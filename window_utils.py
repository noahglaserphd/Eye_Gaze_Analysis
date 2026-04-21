"""
Shared time-window helpers for app and offline scripts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_windows_from_fix(fix: pd.DataFrame, window_size: float) -> pd.DataFrame:
    if len(fix) == 0 or "end_s" not in fix.columns:
        return pd.DataFrame(columns=["window_start", "window_end"])
    max_time = float(fix["end_s"].max())
    if not np.isfinite(max_time) or max_time <= 0:
        return pd.DataFrame(columns=["window_start", "window_end"])
    starts = np.arange(0, max_time + window_size, window_size)
    return pd.DataFrame(
        {
            "window_start": starts[:-1].astype(float),
            "window_end": (starts[:-1] + window_size).astype(float),
        }
    )


def compute_window_metrics(fix: pd.DataFrame, sac: pd.DataFrame, win_df: pd.DataFrame) -> pd.DataFrame:
    needed = {"fixation_count", "mean_fix_duration", "total_fix_time", "mean_saccade_amp"}
    if needed.issubset(set(win_df.columns)):
        return win_df

    rows = []
    for r in win_df.itertuples(index=False):
        start = float(getattr(r, "window_start"))
        end = float(getattr(r, "window_end"))
        f = fix[(fix["start_s"] >= start) & (fix["start_s"] < end)]
        s = sac[(sac["start_s"] >= start) & (sac["start_s"] < end)]
        rows.append(
            {
                "window_start": start,
                "window_end": end,
                "fixation_count": int(len(f)),
                "mean_fix_duration": float(f["duration_s"].mean()) if (len(f) and "duration_s" in f.columns) else 0.0,
                "total_fix_time": float(f["duration_s"].sum()) if (len(f) and "duration_s" in f.columns) else 0.0,
                "mean_saccade_amp": float(s["amplitude_norm"].mean()) if (len(s) and "amplitude_norm" in s.columns) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def subset_window(fix: pd.DataFrame, sac: pd.DataFrame, start: float, end: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    f = fix[(fix["start_s"] >= start) & (fix["start_s"] < end)].copy()
    s = sac[(sac["start_s"] >= start) & (sac["start_s"] < end)].copy()
    return f, s
