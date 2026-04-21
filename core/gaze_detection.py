"""
I-DT fixation detection and saccade derivation from fixation sequences (numpy/pandas only).

Used by extract_dot.py and analyze_gaze_csv.py so CSV reprocessing does not require OpenCV.
Visualization and windowing defaults (heatmap bins, 30 s windows) live in pipeline_config.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Fixation detection defaults (match extract_dot.py)
DISPERSION_THRESH_NORM = 0.006
MIN_FIX_DURATION_S = 0.10


def _require_columns(df: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")


def detect_fixations_idt(
    df: pd.DataFrame,
    xcol: str = "gaze_x_norm",
    ycol: str = "gaze_y_norm",
    tcol: str = "timestamp",
    dispersion_thresh: float = DISPERSION_THRESH_NORM,
    min_duration_s: float = MIN_FIX_DURATION_S,
) -> pd.DataFrame:
    _require_columns(df, (xcol, ycol, tcol))
    x = df[xcol].to_numpy(dtype=float)
    y = df[ycol].to_numpy(dtype=float)
    t = df[tcol].to_numpy(dtype=float)

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(t)
    idxs = np.where(valid)[0]
    fix: list[dict] = []

    if len(idxs) == 0:
        return pd.DataFrame(fix)

    start_ptr = 0
    while start_ptr < len(idxs):
        start_i = idxs[start_ptr]
        end_ptr = start_ptr

        while end_ptr < len(idxs) and (t[idxs[end_ptr]] - t[start_i]) < min_duration_s:
            end_ptr += 1
        if end_ptr >= len(idxs):
            break

        def dispersion(window: np.ndarray) -> float:
            return (np.max(x[window]) - np.min(x[window])) + (np.max(y[window]) - np.min(y[window]))

        window = idxs[start_ptr : end_ptr + 1]
        while end_ptr < len(idxs) and dispersion(window) <= dispersion_thresh:
            end_ptr += 1
            if end_ptr < len(idxs):
                window = idxs[start_ptr : end_ptr + 1]

        window = idxs[start_ptr:end_ptr]
        if len(window) >= 2:
            t0 = float(t[window[0]])
            t1 = float(t[window[-1]])
            fix.append(
                {
                    "start_s": t0,
                    "end_s": t1,
                    "duration_s": t1 - t0,
                    "x_norm": float(np.mean(x[window])),
                    "y_norm": float(np.mean(y[window])),
                    "n_samples": int(len(window)),
                }
            )

        start_ptr = end_ptr

    return pd.DataFrame(fix)


def saccades_from_fixations(fix_df: pd.DataFrame) -> pd.DataFrame:
    sacc: list[dict] = []
    if len(fix_df) < 2:
        return pd.DataFrame(sacc)

    _require_columns(fix_df, ("x_norm", "y_norm", "start_s", "end_s"))

    for i in range(len(fix_df) - 1):
        a = fix_df.iloc[i]
        b = fix_df.iloc[i + 1]
        dx = b["x_norm"] - a["x_norm"]
        dy = b["y_norm"] - a["y_norm"]
        amp = float(np.sqrt(dx * dx + dy * dy))
        sacc.append(
            {
                "start_s": float(a["end_s"]),
                "end_s": float(b["start_s"]),
                "duration_s": float(b["start_s"] - a["end_s"]),
                "from_x_norm": float(a["x_norm"]),
                "from_y_norm": float(a["y_norm"]),
                "to_x_norm": float(b["x_norm"]),
                "to_y_norm": float(b["y_norm"]),
                "amplitude_norm": amp,
                "direction_rad": float(np.arctan2(dy, dx)),
            }
        )

    return pd.DataFrame(sacc)
