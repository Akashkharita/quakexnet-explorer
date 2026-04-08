import numpy as np
from scipy.ndimage import uniform_filter1d

# we want to smoothen out the probabilities here. 
def smooth_moving_avg(x, window=5):
    pad_width = window // 2
    padded = np.pad(x, (pad_width, pad_width), mode='edge')
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode='valid')






def detect_event_windows(prob_series, enter_thr=0.15, exit_thr=0.15, peak_thr=0.5):
    in_event = False
    events = []
    start = None
    max_val = -np.inf
    running_vals = []

    for i, val in enumerate(prob_series):
        if not in_event and val >= enter_thr:
            start = i
            max_val = val
            running_vals = [val]
            in_event = True
        elif in_event:
            running_vals.append(val)
            max_val = max(max_val, val)
            if val < exit_thr:
                end = i
                if max_val >= peak_thr:
                    mean_val = np.mean(running_vals)
                    auc = np.trapz(running_vals)  # Area under the curve
                    events.append({
                        "start": start,
                        "end": end,
                        "max_prob": max_val,
                        "mean_prob": mean_val,
                        "area_under_curve": auc
                    })
                in_event = False

    # Handle if still in event at end of series
    if in_event and max_val >= peak_thr:
        mean_val = np.mean(running_vals)
        auc = np.trapz(running_vals)
        events.append({
            "start": start,
            "end": len(prob_series) - 1,
            "max_prob": max_val,
            "mean_prob": mean_val,
            "area_under_curve": auc
        })

    return events

