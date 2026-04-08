# app.py
# QuakeXNet-as-a-Service
# Wraps your existing detection pipeline in a FastAPI web API.
#
# Run with:  uvicorn app:app --reload --port 8000
# Docs at:   http://localhost:8000/docs

import io
import json
import time
import numpy as np
import seisbench.models as sbm

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from obspy import UTCDateTime, read as obspy_read
from obspy.clients.fdsn import Client
from pydantic import BaseModel
from typing import List, Optional

from detect import smooth_moving_avg, detect_event_windows


# ===========================================================================
# 1. CREATE THE APP
# ===========================================================================
# This one line creates your entire web application.
# The title and description show up automatically at /docs.

app = FastAPI(
    title="QuakeXNet Detection API",
    description=(
        "Seismic event detection using QuakeXNet (Kharita et al., 2026). "
        "Classifies waveform windows as earthquake (eq), explosion (px), "
        "surface event (su), or noise."
    ),
    version="1.0.0",
)


# ===========================================================================
# 2. LOAD THE MODEL ONCE AT STARTUP
# ===========================================================================
# This is important. In your script, the model loads every time you run it.
# In an API, we load it ONCE when the server starts, then reuse it for every
# request. This is why APIs are fast — the expensive part happens once.

print("Loading QuakeXNet model...")
model = sbm.QuakeXNet.from_pretrained("base", version_str="3")
print("Model loaded and ready.")

# IRIS client — same as your script, reused across all requests
iris_client = Client("IRIS")

# Constants — same values as your script
SECONDS_PER_STEP = 10
CLASS_NAMES = ["eq", "px", "su"]
CHL_PREFIX = "QuakeXNet_"
CHANNEL_MAP = {cls: f"{CHL_PREFIX}{cls}" for cls in CLASS_NAMES}


# ===========================================================================
# 3. PYDANTIC MODELS — Define what goes IN and what comes OUT
# ===========================================================================
# These are blueprints. FastAPI uses them to:
#   - Validate that responses have the right structure
#   - Auto-generate the /docs page
#   - Catch bugs where your code returns wrong types

class DetectedEvent(BaseModel):
    """One detected event at one station — mirrors your existing event_records dict."""
    station: str
    network: str
    event_class: str        # "eq", "px", or "su"
    auc: float
    mean_prob: float
    max_prob: float
    start_index: int
    end_index: int
    start_time: str
    end_time: str


class DetectionResponse(BaseModel):
    """The full response returned by the API for any detection request."""
    status: str                      # "success" or "error"
    query_start: str                 # what time window was searched
    query_end: str
    stations_processed: int
    stations_failed: int
    total_events_detected: int
    processing_time_seconds: float
    events: List[DetectedEvent]      # list of all detected events


# ===========================================================================
# 4. HELPER — the core detection logic extracted from your script
# ===========================================================================
# This is your existing loop, unchanged, just moved into a function.
# Notice: not a single line of the actual science has changed.

def run_detection_on_stream(st, st_time, et_time, sta, net):
    """
    Runs QuakeXNet inference + detection on an ObsPy Stream.
    This is your existing inner loop from daily_detection.py, verbatim.
    Returns a list of event dicts.
    """
    probs_st = model.annotate(st, stride=500)
    event_records = []

    for cls in CLASS_NAMES:
        probs = probs_st.select(channel=CHANNEL_MAP[cls])

        for prob in probs:
            trace_start = prob.stats.starttime
            probs_array = prob.data

            # Your existing smoothing + detection — completely unchanged
            s_cls = smooth_moving_avg(probs_array)
            events = detect_event_windows(s_cls)

            for event in events:
                start_idx = event["start"]
                end_idx   = event["end"]

                event_start_time = trace_start + start_idx * SECONDS_PER_STEP
                event_end_time   = trace_start + end_idx   * SECONDS_PER_STEP

                # Your existing safety filter — unchanged
                if event_end_time < st_time or event_start_time > et_time:
                    continue

                event_records.append(DetectedEvent(
                    station=sta,
                    network=net,
                    event_class=cls,
                    auc=float(event["area_under_curve"]),
                    mean_prob=float(event["mean_prob"]),
                    max_prob=float(event["max_prob"]),
                    start_index=start_idx,
                    end_index=end_idx,
                    start_time=str(event_start_time),
                    end_time=str(event_end_time),
                ))

    return event_records


# ===========================================================================
# 5. ENDPOINT A — Detect from IRIS (mirrors your existing script exactly)
# ===========================================================================
# POST /detect/iris
#
# The caller sends: start time, end time, list of stations
# The API fetches waveforms from IRIS, runs detection, returns JSON events.
#
# This replaces:
#   python daily_detection.py --start 2025-12-10T00:00:00 \
#                              --end   2025-12-10T23:59:59 \
#                              --stations_json stations.json

@app.post("/detect/iris", response_model=DetectionResponse)
def detect_from_iris(
    start: str = Query(..., description="Start time UTC e.g. 2025-12-10T00:00:00"),
    end:   str = Query(..., description="End time UTC   e.g. 2025-12-10T23:59:59"),
    stations_json: str = Query(..., description="Path to stations JSON file"),
):
    """
    Fetch waveforms from IRIS and run QuakeXNet detection.
    Equivalent to running daily_detection.py from the command line,
    but callable by any program, from anywhere.
    """
    # --- Parse and validate times ---
    # If the user sends a bad time string, FastAPI returns a clean 400 error
    # instead of crashing with a Python traceback.
    try:
        st_time = UTCDateTime(start)
        et_time = UTCDateTime(end)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid time format. Use ISO 8601 e.g. '2025-12-10T00:00:00'"
        )

    if et_time <= st_time:
        raise HTTPException(status_code=400, detail="end must be after start.")

    # --- Load stations ---
    try:
        with open(stations_json) as f:
            stations = json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"stations_json not found: {stations_json}")

    # --- Run detection loop (identical to your script) ---
    t0 = time.time()
    all_events = []
    failed = 0

    for entry in stations:
        net = entry["net"]
        sta = entry["sta"]
        chn = entry.get("chn", "*H")

        try:
            st = iris_client.get_waveforms(
                network=net, station=sta,
                channel=chn + "*", location="*",
                starttime=st_time, endtime=et_time,
            )
            events = run_detection_on_stream(st, st_time, et_time, sta, net)
            all_events.extend(events)

        except Exception as e:
            # In your script this prints and continues — same behaviour here,
            # but we count failures so the caller knows something went wrong.
            print(f"Failed {net}.{sta}: {e}")
            failed += 1

    elapsed = round(time.time() - t0, 2)

    return DetectionResponse(
        status="success",
        query_start=str(st_time),
        query_end=str(et_time),
        stations_processed=len(stations) - failed,
        stations_failed=failed,
        total_events_detected=len(all_events),
        processing_time_seconds=elapsed,
        events=all_events,
    )


# ===========================================================================
# 6. ENDPOINT B — Detect from uploaded miniSEED file
# ===========================================================================
# POST /detect/upload
#
# The caller uploads a miniSEED file directly.
# No IRIS connection needed — useful for local files or offline use.
# This is the endpoint a dashboard or external app would use.

@app.post("/detect/upload", response_model=DetectionResponse)
async def detect_from_upload(
    file: UploadFile = File(..., description="miniSEED waveform file"),
    start: Optional[str] = Query(None, description="Optional start filter (UTC ISO 8601)"),
    end:   Optional[str] = Query(None, description="Optional end filter   (UTC ISO 8601)"),
):
    """
    Upload a miniSEED file and run QuakeXNet detection on it.
    This is the endpoint a web app, dashboard, or colleague would call —
    no IRIS access or Python environment needed on their end.
    """
    # --- Validate file type ---
    if not file.filename.endswith((".mseed", ".miniseed", ".ms")):
        raise HTTPException(
            status_code=400,
            detail="File must be a miniSEED file (.mseed / .miniseed / .ms)"
        )

    # --- Read uploaded bytes into ObsPy Stream ---
    # file.read() gives us raw bytes. io.BytesIO wraps them so ObsPy can read
    # them as if they were a file on disk — no temp file needed.
    raw_bytes = await file.read()
    try:
        st = obspy_read(io.BytesIO(raw_bytes))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read file as miniSEED.")

    # --- Determine time window ---
    # Use provided times if given, otherwise use the stream's own time range.
    try:
        st_time = UTCDateTime(start) if start else st[0].stats.starttime
        et_time = UTCDateTime(end)   if end   else st[0].stats.endtime
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid time format.")

    # --- Get station info from the stream itself ---
    sta = st[0].stats.station
    net = st[0].stats.network

    # --- Run detection ---
    t0 = time.time()
    try:
        events = run_detection_on_stream(st, st_time, et_time, sta, net)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

    elapsed = round(time.time() - t0, 2)

    return DetectionResponse(
        status="success",
        query_start=str(st_time),
        query_end=str(et_time),
        stations_processed=1,
        stations_failed=0,
        total_events_detected=len(events),
        processing_time_seconds=elapsed,
        events=events,
    )


# ===========================================================================
# 7. HEALTH CHECK ENDPOINT
# ===========================================================================
# GET /health
#
# Every production API has this. It lets monitoring systems, Docker, and
# orchestration tools check "is this service alive?" with a simple request.
# Returns instantly without running any ML — just confirms the server is up.

@app.get("/health")
def health():
    """Check that the API is running and the model is loaded."""
    return {
        "status": "healthy",
        "model": "QuakeXNet",
        "version": "3",
        "classes": CLASS_NAMES,
    }


# ===========================================================================
# 8. ROOT ENDPOINT
# ===========================================================================
# GET /
#
# What you see if you open the API URL in a browser.
# Points people to /docs where they can explore and test the API.

@app.get("/")
def root():
    return {
        "message": "QuakeXNet Detection API is running.",
        "docs": "/docs",
        "health": "/health",
        "paper": "https://doi.org/10.26443/seismica.v5i1.2068",
    }