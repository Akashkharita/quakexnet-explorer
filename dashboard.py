# dashboard.py
# QuakeXNet Explorer — Interactive seismic event detection dashboard
# Kharita et al. 2026 · https://doi.org/10.26443/seismica.v5i1.2068
#
# Run with: python dashboard.py
# Tunnel:   ssh -L 7860:localhost:7860 ak287@your-server

import time
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import folium
from folium.plugins import Draw

import gradio as gr

from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.signal.filter import envelope
from obspy.geodetics import gps2dist_azimuth

import seisbench.models as sbm

from detect import smooth_moving_avg, detect_event_windows


# ===========================================================================
# STARTUP
# ===========================================================================

print("Loading QuakeXNet model...")
MODEL = sbm.QuakeXNet.from_pretrained("base", version_str="3")
IRIS  = Client("IRIS")
print("Ready.")

CLASS_NAMES      = ["eq", "px", "su"]
CHL_PREFIX       = "QuakeXNet_"
CHANNEL_MAP      = {cls: f"{CHL_PREFIX}{cls}" for cls in CLASS_NAMES}
SECONDS_PER_STEP = 10
CLASS_COLORS     = {"su": "#E24B4A", "eq": "#185FA5", "px": "#3B6D11"}
CLASS_LABELS     = {"su": "Surface (su)", "eq": "Earthquake (eq)", "px": "Explosion (px)"}


# ===========================================================================
# BUILD FOLIUM MAP
# ===========================================================================

def build_folium_map(station_list=None):
    """
    Builds a Folium map centred on Mount Rainier with the Draw plugin.
    The Draw plugin's export=True adds a download button that, when clicked,
    shows the drawn shape's GeoJSON in a popup — the user copies that JSON
    into the GeoJSON textbox and clicks 'Load coordinates from GeoJSON'.

    If station_list is provided, plots them as red circle markers.
    """
    m = folium.Map(location=[46.85, -121.75], zoom_start=9, tiles="OpenStreetMap")

    # Mount Rainier reference marker
    folium.Marker(
        location=[46.8529, -121.7604],
        tooltip="Mount Rainier (4,392 m)",
        icon=folium.Icon(color="red", icon="info-sign"),
    ).add_to(m)

    # Draw plugin with export=True — this is what shows the GeoJSON popup
    Draw(
        export=True,
        draw_options={
            "rectangle":    {"shapeOptions": {"color": "#185FA5", "weight": 2}},
            "polygon":      False,
            "circle":       False,
            "marker":       False,
            "polyline":     False,
            "circlemarker": False,
        },
        edit_options={"edit": False},
    ).add_to(m)

    # Plot stations if provided
    if station_list:
        for s in station_list:
            folium.CircleMarker(
                location=[s["lat"], s["lon"]],
                radius=7,
                color="white",
                fill=True,
                fill_color="#E24B4A",
                fill_opacity=0.9,
                tooltip=f"{s['net']}.{s['sta']}",
            ).add_to(m)

    return m._repr_html_()


# ===========================================================================
# PARSE GEOJSON FROM FOLIUM EXPORT
# ===========================================================================

def parse_geojson_bbox(geojson_str):
    """
    Parses the GeoJSON string exported by the Folium Draw plugin.
    Returns (minlat, maxlat, minlon, maxlon) or raises ValueError.

    The exported GeoJSON looks like:
    {
      "type": "FeatureCollection",
      "features": [{
        "type": "Feature",
        "geometry": {
          "type": "Polygon",
          "coordinates": [[[lon1,lat1],[lon2,lat2],[lon3,lat3],[lon4,lat4],[lon1,lat1]]]
        }
      }]
    }
    """
    geojson_str = geojson_str.strip()
    if not geojson_str:
        raise ValueError("No GeoJSON provided.")

    try:
        data = json.loads(geojson_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")

    # Handle both FeatureCollection and bare Feature
    if data.get("type") == "FeatureCollection":
        features = data.get("features", [])
        if not features:
            raise ValueError("No features found in GeoJSON. Draw a rectangle first.")
        geometry = features[0].get("geometry", {})
    elif data.get("type") == "Feature":
        geometry = data.get("geometry", {})
    else:
        raise ValueError("Unexpected GeoJSON type. Expected FeatureCollection or Feature.")

    coords = geometry.get("coordinates", [])
    if not coords:
        raise ValueError("No coordinates found in geometry.")

    # Polygon coordinates: [[[lon, lat], ...]]
    ring = coords[0]
    lons = [pt[0] for pt in ring]
    lats = [pt[1] for pt in ring]

    return min(lats), max(lats), min(lons), max(lons)


# ===========================================================================
# QUERY IRIS FOR STATIONS
# ===========================================================================

def query_stations(minlat, maxlat, minlon, maxlon, networks="CC,UW"):
    try:
        minlat, maxlat = float(minlat), float(maxlat)
        minlon, maxlon = float(minlon), float(maxlon)
    except ValueError:
        return [], "Invalid coordinates — enter numbers only."

    try:
        inv = IRIS.get_stations(
            minlatitude=minlat, maxlatitude=maxlat,
            minlongitude=minlon, maxlongitude=maxlon,
            network=networks, level="station",
        )
    except Exception as e:
        return [], f"IRIS query failed: {e}"

    stations = []
    for net in inv:
        for sta in net:
            stations.append({
                "net":  net.code,
                "sta":  sta.code,
                "lat":  float(sta.latitude),
                "lon":  float(sta.longitude),
                "elev": float(sta.elevation),
            })

    if not stations:
        return [], "No stations found. Try a larger box or different networks."

    msg = (
        f"Found {len(stations)} stations: "
        + ", ".join(f"{s['net']}.{s['sta']}" for s in stations)
    )
    return stations, msg


# ===========================================================================
# ENVELOPE SNR
# ===========================================================================

def envelope_snr(tr):
    env = envelope(tr.data.astype(np.float64))
    m   = np.mean(env)
    if not np.isfinite(m) or m <= 0:
        return 0.0
    mx = np.max(env)
    return float(mx / m) if np.isfinite(mx) else 0.0


# ===========================================================================
# FETCH WAVEFORMS + RUN DETECTION
# ===========================================================================

def run_detection(
    stations, start_str, end_str,
    ref_lat, ref_lon,
    freqmin=1.0, freqmax=20.0,
    snr_thresh=2.0, peak_thresh=0.50,
    max_dist_km=50.0,
):
    kept       = []
    all_events = []
    log_lines  = []

    st_time = UTCDateTime(start_str)
    et_time = UTCDateTime(end_str)

    for entry in stations:
        net     = entry["net"]
        sta     = entry["sta"]
        sta_lat = entry["lat"]
        sta_lon = entry["lon"]

        dist_m, _, _ = gps2dist_azimuth(ref_lat, ref_lon, sta_lat, sta_lon)
        dist_km = dist_m / 1000.0

        if dist_km > max_dist_km:
            log_lines.append(f"SKIP {net}.{sta}: {dist_km:.1f} km > {max_dist_km} km limit")
            continue

        try:
            st = IRIS.get_waveforms(
                network=net, station=sta,
                channel="*HZ,*HN,*HE",
                location="*",
                starttime=st_time,
                endtime=et_time,
            )
            if len(st) == 0:
                log_lines.append(f"SKIP {net}.{sta}: no data returned")
                continue

            # Display stream — bandpass filtered for visualization only
            st_display = st.copy()
            st_display.resample(50)
            st_display.detrend("linear")
            st_display.taper(0.01)
            st_display.filter("bandpass", freqmin=freqmin, freqmax=freqmax)

            # Model stream — NO bandpass filter
            # QuakeXNet was trained on raw waveforms
            st_model = st.copy()
            st_model.resample(50)
            st_model.detrend("linear")
            st_model.taper(0.01)

            snr = envelope_snr(st_display[0])
            if snr < snr_thresh:
                log_lines.append(f"SKIP {net}.{sta}: SNR={snr:.2f} < {snr_thresh}")
                continue

            probs_st = MODEL.annotate(st_model, stride=500)

            probs = {}
            for cls in CLASS_NAMES:
                sel = probs_st.select(channel=CHANNEL_MAP[cls])
                probs[cls] = sel[0] if len(sel) > 0 else None

            if probs["su"] is None:
                log_lines.append(f"SKIP {net}.{sta}: model returned no output")
                continue

            trace_start = probs["su"].stats.starttime
            t_probs     = probs["su"].times() + 50

            for cls in CLASS_NAMES:
                if probs[cls] is None:
                    continue
                s_cls  = smooth_moving_avg(probs[cls].data)
                events = detect_event_windows(s_cls, peak_thr=peak_thresh)

                for ev in events:
                    start_idx = ev["start"]
                    end_idx   = ev["end"]
                    ev_start  = trace_start + start_idx * SECONDS_PER_STEP
                    ev_end    = trace_start + end_idx   * SECONDS_PER_STEP

                    if ev_end < st_time or ev_start > et_time:
                        continue

                    all_events.append({
                        "station":    sta,
                        "network":    net,
                        "class":      cls,
                        "max_prob":   round(float(ev["max_prob"]),         3),
                        "mean_prob":  round(float(ev["mean_prob"]),        3),
                        "auc":        round(float(ev["area_under_curve"]), 3),
                        "start_time": str(ev_start),
                        "end_time":   str(ev_end),
                        "dist_km":    round(dist_km, 2),
                    })

            tr     = st_display[0].copy()
            t_wave = tr.times("relative")
            y_wave = tr.data / (np.abs(tr.data).max() + 1e-12)

            kept.append({
                "sta":     sta,
                "net":     net,
                "dist_km": dist_km,
                "snr":     round(snr, 2),
                "t_wave":  t_wave,
                "y_wave":  y_wave,
                "t_probs": t_probs,
                "d_su": probs["su"].data if probs["su"] else np.zeros(len(t_probs)),
                "d_eq": probs["eq"].data if probs["eq"] else np.zeros(len(t_probs)),
                "d_px": probs["px"].data if probs["px"] else np.zeros(len(t_probs)),
            })

            log_lines.append(f"OK   {net}.{sta}: {dist_km:.1f} km  SNR={snr:.1f}")

        except Exception as e:
            log_lines.append(f"ERR  {net}.{sta}: {e}")

    kept.sort(key=lambda x: x["dist_km"])
    return kept, all_events, "\n".join(log_lines)


# ===========================================================================
# FIGURE
# ===========================================================================

def make_figure(kept, start_str, end_str, peak_thresh=0.50):
    n = len(kept)

    if n == 0:
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.text(
            0.5, 0.5,
            "No stations passed quality filters.\n"
            "Try lowering SNR threshold or increasing max distance.",
            ha="center", va="center",
            transform=ax.transAxes, fontsize=13,
        )
        ax.axis("off")
        return fig

    plt.rcParams.update({
        "font.size": 13, "axes.titlesize": 15,
        "axes.labelsize": 13, "xtick.labelsize": 12,
        "ytick.labelsize": 12, "legend.fontsize": 12,
    })

    fig, ax = plt.subplots(figsize=(16, max(8, 1.8 * n)), dpi=150)
    spacing = 3.0

    for i in range(n):
        if i % 2 == 0:
            ax.axhspan(spacing * i - 1.3, spacing * i + 1.3,
                       color="0.96", zorder=0)

    for i, item in enumerate(kept):
        offset = spacing * i

        ax.plot(item["t_wave"], item["y_wave"] + offset,
                lw=0.8, color="black", alpha=0.7, zorder=2)

        ax.plot(item["t_probs"], item["d_su"] + offset,
                lw=2.0, color=CLASS_COLORS["su"],
                label=CLASS_LABELS["su"] if i == 0 else "", zorder=3)
        ax.plot(item["t_probs"], item["d_eq"] + offset,
                lw=2.0, color=CLASS_COLORS["eq"],
                label=CLASS_LABELS["eq"] if i == 0 else "", zorder=3)
        ax.plot(item["t_probs"], item["d_px"] + offset,
                lw=2.0, color=CLASS_COLORS["px"],
                label=CLASS_LABELS["px"] if i == 0 else "", zorder=3)

        d_su = item["d_su"]
        if len(d_su) > 2:
            peaks = np.where(
                (d_su[1:-1] > d_su[:-2]) &
                (d_su[1:-1] > d_su[2:])  &
                (d_su[1:-1] >= peak_thresh)
            )[0] + 1
            for p in peaks:
                ax.text(
                    item["t_probs"][p],
                    d_su[p] + offset + 0.12,
                    f"{d_su[p]:.2f}",
                    ha="center", va="bottom",
                    fontsize=10, color="darkred",
                    bbox=dict(boxstyle="round,pad=0.15",
                              fc="white", ec="none", alpha=0.85),
                    zorder=4,
                )

    ax.set_yticks([spacing * i for i in range(n)])
    ax.set_yticklabels([
        f"{item['net']}.{item['sta']}  ({item['dist_km']:.1f} km)  SNR={item['snr']}"
        for item in kept
    ])
    ax.set_ylim(-1.5, spacing * (n - 1) + 2.0)
    ax.set_xlabel("Time relative to stream start (s)")
    ax.set_title(
        f"QuakeXNet detection  ·  {start_str}  →  {end_str}\n"
        f"{n} stations · sorted by distance from reference point",
        fontsize=14, weight="bold", pad=10,
    )
    ax.grid(True, axis="x", alpha=0.2, lw=0.5)
    ax.grid(False, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95)
    plt.tight_layout()
    return fig


# ===========================================================================
# EVENTS TABLE
# ===========================================================================

def make_events_table(all_events):
    if not all_events:
        return pd.DataFrame(columns=[
            "Station", "Network", "Class", "Max prob",
            "Mean prob", "AUC", "Start time", "End time", "Dist (km)"
        ])
    df = pd.DataFrame(all_events).rename(columns={
        "station": "Station", "network": "Network", "class": "Class",
        "max_prob": "Max prob", "mean_prob": "Mean prob", "auc": "AUC",
        "start_time": "Start time", "end_time": "End time", "dist_km": "Dist (km)",
    })
    return df.sort_values(["Dist (km)", "Start time"]).reset_index(drop=True)


# ===========================================================================
# CALLBACKS
# ===========================================================================

_station_cache = []


def cb_load_geojson(geojson_str):
    """
    Parse the GeoJSON from the Folium export popup and return
    the four bounding box coordinates into the textboxes.
    """
    try:
        minlat, maxlat, minlon, maxlon = parse_geojson_bbox(geojson_str)
        msg = (
            f"Loaded bounding box: "
            f"lat [{minlat:.4f}, {maxlat:.4f}]  "
            f"lon [{minlon:.4f}, {maxlon:.4f}]"
        )
        return (
            str(round(minlat, 4)),
            str(round(maxlat, 4)),
            str(round(minlon, 4)),
            str(round(maxlon, 4)),
            msg,
        )
    except ValueError as e:
        return "", "", "", "", f"Error: {e}"


def cb_query_stations(minlat, maxlat, minlon, maxlon, networks):
    global _station_cache
    _station_cache, msg = query_stations(minlat, maxlat, minlon, maxlon, networks)

    # Rebuild map with station markers
    map_html = build_folium_map(station_list=_station_cache if _station_cache else None)
    return msg, map_html


def cb_run_detection(
    minlat, maxlat, minlon, maxlon,
    start_str, end_str,
    use_manual, manual_lat, manual_lon,
    freqmin, freqmax,
    snr_thresh, peak_thresh, max_dist_km,
):
    global _station_cache
    t0 = time.time()

    if not _station_cache:
        return None, pd.DataFrame(), "No stations loaded — click 'Query stations from IRIS' first.", ""

    try:
        st_time = UTCDateTime(start_str)
        et_time = UTCDateTime(end_str)
    except Exception:
        return None, pd.DataFrame(), "Invalid time format. Use YYYY-MM-DDTHH:MM:SS", ""

    if (et_time - st_time) > 6 * 3600:
        return None, pd.DataFrame(), "Maximum window is 6 hours.", ""
    if et_time <= st_time:
        return None, pd.DataFrame(), "End time must be after start time.", ""

    if use_manual and manual_lat and manual_lon:
        try:
            ref_lat = float(manual_lat)
            ref_lon = float(manual_lon)
        except ValueError:
            return None, pd.DataFrame(), "Invalid manual lat/lon.", ""
    else:
        try:
            ref_lat = (float(minlat) + float(maxlat)) / 2
            ref_lon = (float(minlon) + float(maxlon)) / 2
        except ValueError:
            return None, pd.DataFrame(), "Enter bounding box coordinates first.", ""

    kept, all_events, log = run_detection(
        stations=_station_cache,
        start_str=start_str, end_str=end_str,
        ref_lat=ref_lat, ref_lon=ref_lon,
        freqmin=freqmin, freqmax=freqmax,
        snr_thresh=snr_thresh, peak_thresh=peak_thresh,
        max_dist_km=max_dist_km,
    )

    fig = make_figure(kept, start_str, end_str, peak_thresh)
    df  = make_events_table(all_events)

    elapsed = round(time.time() - t0, 1)
    n_su = sum(1 for e in all_events if e["class"] == "su")
    n_eq = sum(1 for e in all_events if e["class"] == "eq")
    n_px = sum(1 for e in all_events if e["class"] == "px")

    summary = (
        f"Processed {len(kept)}/{len(_station_cache)} stations in {elapsed}s  ·  "
        f"Surface events: {n_su}  ·  Earthquakes: {n_eq}  ·  Explosions: {n_px}"
    )
    return fig, df, summary, log


# ===========================================================================
# UI
# ===========================================================================

with gr.Blocks(title="QuakeXNet Explorer", theme=gr.themes.Base()) as demo:

    gr.Markdown("""
    # QuakeXNet Explorer
    **Real-time seismic event detection** · Kharita, Denolle et al. · *Seismica* 2026 ·
    [Paper](https://doi.org/10.26443/seismica.v5i1.2068)
    """)

    with gr.Row():

        # ---- Left column: map + station query ----
        with gr.Column(scale=3):

            gr.Markdown("### 1. Select region")
            gr.Markdown(
                "**How to use the map:**\n"
                "1. Click the rectangle tool (toolbar on the left of the map)\n"
                "2. Draw a rectangle over your region of interest\n"
                "3. Click the **Export** button that appears — a popup shows the GeoJSON\n"
                "4. Copy the entire JSON text from the popup\n"
                "5. Paste it into the box below and click **Load coordinates from GeoJSON**"
            )

            map_component = gr.HTML(value=build_folium_map())

            geojson_box = gr.Textbox(
                label="Paste GeoJSON from map export here",
                placeholder='{"type":"FeatureCollection","features":[...]}',
                lines=3,
            )

            load_btn = gr.Button("Load coordinates from GeoJSON", variant="secondary")

            with gr.Row():
                minlat_box = gr.Textbox(label="Min lat", value="46.7")
                maxlat_box = gr.Textbox(label="Max lat", value="47.0")
                minlon_box = gr.Textbox(label="Min lon", value="-122.0")
                maxlon_box = gr.Textbox(label="Max lon", value="-121.4")

            bbox_status = gr.Textbox(
                label="Bounding box status",
                interactive=False,
                placeholder="Bounding box will appear here after loading GeoJSON...",
            )

            networks_box = gr.Textbox(
                label="Networks (comma-separated)", value="CC,UW"
            )
            query_btn = gr.Button("Query stations from IRIS", variant="secondary")
            station_status = gr.Textbox(
                label="Station query result", interactive=False,
                placeholder="Stations will appear here after querying...",
            )

        # ---- Right column: time + options ----
        with gr.Column(scale=2):

            gr.Markdown("### 2. Time window")
            start_box = gr.Textbox(
                label="Start time (UTC)", value="2025-01-15T08:00:00"
            )
            end_box = gr.Textbox(
                label="End time (UTC)", value="2025-01-15T10:00:00"
            )

            gr.Markdown("### 3. Reference location")
            use_manual = gr.Checkbox(
                label="Use manual event location (otherwise uses bounding box centre)",
                value=False,
            )
            with gr.Row():
                manual_lat = gr.Textbox(label="Event lat", placeholder="e.g. 46.8529")
                manual_lon = gr.Textbox(label="Event lon", placeholder="e.g. -121.7604")

            gr.Markdown("### 4. Detection options")
            with gr.Row():
                freqmin_sl = gr.Slider(0.1, 10,  value=1.0,  step=0.1,  label="Freq min (Hz)")
                freqmax_sl = gr.Slider(5,   40,   value=20.0, step=1.0,  label="Freq max (Hz)")
            with gr.Row():
                snr_sl  = gr.Slider(0, 10, value=2.0,  step=0.5,  label="Min SNR")
                peak_sl = gr.Slider(0, 1,  value=0.50, step=0.05, label="Min peak prob")
            maxdist_sl = gr.Slider(5, 300, value=50, step=5, label="Max distance (km)")

            run_btn = gr.Button("Run detection", variant="primary", size="lg")

    # ---- Results ----
    gr.Markdown("---")
    gr.Markdown("### Results")

    summary_box  = gr.Textbox(
        label="Summary", interactive=False,
        placeholder="Results will appear here after running detection...",
    )
    plot_out     = gr.Plot(label="Waveforms + probability traces")
    events_table = gr.Dataframe(
        label="Detected events", interactive=False, wrap=True
    )
    log_box = gr.Textbox(
        label="Processing log", interactive=False, lines=8,
        placeholder="Per-station log will appear here...",
    )

    # ---- Wire up buttons ----
    load_btn.click(
        fn=cb_load_geojson,
        inputs=[geojson_box],
        outputs=[minlat_box, maxlat_box, minlon_box, maxlon_box, bbox_status],
    )

    query_btn.click(
        fn=cb_query_stations,
        inputs=[minlat_box, maxlat_box, minlon_box, maxlon_box, networks_box],
        outputs=[station_status, map_component],
    )

    run_btn.click(
        fn=cb_run_detection,
        inputs=[
            minlat_box, maxlat_box, minlon_box, maxlon_box,
            start_box, end_box,
            use_manual, manual_lat, manual_lon,
            freqmin_sl, freqmax_sl,
            snr_sl, peak_sl, maxdist_sl,
        ],
        outputs=[plot_out, events_table, summary_box, log_box],
    )

    gr.Markdown("""
    ---
    QuakeXNet · Kharita, Denolle, Hutko, Hartog & Malone · *Seismica* 2026 ·
    [doi:10.26443/seismica.v5i1.2068](https://doi.org/10.26443/seismica.v5i1.2068)
    """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7863,
        share=False,
    )