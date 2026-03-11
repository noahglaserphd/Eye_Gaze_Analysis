import numpy as np
import pandas as pd
from pathlib import Path

INFILE = list(Path("gaze_samples").glob("*.csv"))[0]   # change if needed
OUT_FIX = INFILE.with_name(INFILE.stem + "_fixations.csv")
OUT_SAC = INFILE.with_name(INFILE.stem + "_saccades.csv")

def detect_fixations_idt(df, xcol="gaze_x_norm", ycol="gaze_y_norm", tcol="timestamp",
                         dispersion_thresh=0.02, min_duration_s=0.12):
    """
    I-DT fixation detection on normalized coordinates (0..1).
    dispersion = (max(x)-min(x)) + (max(y)-min(y))
    """
    x = df[xcol].to_numpy(dtype=float)
    y = df[ycol].to_numpy(dtype=float)
    t = df[tcol].to_numpy(dtype=float)

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(t)
    idxs = np.where(valid)[0]
    fix = []

    if len(idxs) == 0:
        return pd.DataFrame(fix)

    start_ptr = 0
    while start_ptr < len(idxs):
        start_i = idxs[start_ptr]
        end_ptr = start_ptr

        # grow window until it reaches minimum duration
        while end_ptr < len(idxs) and (t[idxs[end_ptr]] - t[start_i]) < min_duration_s:
            end_ptr += 1
        if end_ptr >= len(idxs):
            break

        def dispersion(window):
            return (np.max(x[window]) - np.min(x[window])) + (np.max(y[window]) - np.min(y[window]))

        # expand while dispersion stays under threshold
        window = idxs[start_ptr:end_ptr+1]
        while end_ptr < len(idxs) and dispersion(window) <= dispersion_thresh:
            end_ptr += 1
            if end_ptr < len(idxs):
                window = idxs[start_ptr:end_ptr+1]

        # last point exceeded threshold; finalize previous window
        window = idxs[start_ptr:end_ptr]
        if len(window) >= 2:
            t0 = float(t[window[0]])
            t1 = float(t[window[-1]])
            fix.append({
                "start_s": t0,
                "end_s": t1,
                "duration_s": t1 - t0,
                "x_norm": float(np.mean(x[window])),
                "y_norm": float(np.mean(y[window])),
                "n_samples": int(len(window)),
            })

        start_ptr = end_ptr

    return pd.DataFrame(fix)

def saccades_from_fixations(fix_df):
    sacc = []
    if len(fix_df) < 2:
        return pd.DataFrame(sacc)

    for i in range(len(fix_df) - 1):
        a = fix_df.iloc[i]
        b = fix_df.iloc[i + 1]
        dx = b["x_norm"] - a["x_norm"]
        dy = b["y_norm"] - a["y_norm"]
        amp = float(np.sqrt(dx*dx + dy*dy))
        sacc.append({
            "start_s": float(a["end_s"]),
            "end_s": float(b["start_s"]),
            "duration_s": float(b["start_s"] - a["end_s"]),
            "from_x_norm": float(a["x_norm"]),
            "from_y_norm": float(a["y_norm"]),
            "to_x_norm": float(b["x_norm"]),
            "to_y_norm": float(b["y_norm"]),
            "amplitude_norm": amp,
            "direction_rad": float(np.arctan2(dy, dx)),
        })
    return pd.DataFrame(sacc)

df = pd.read_csv(INFILE)

fix = detect_fixations_idt(df, dispersion_thresh=0.006, min_duration_s=0.10)
sac = saccades_from_fixations(fix)

fix.to_csv(OUT_FIX, index=False)
sac.to_csv(OUT_SAC, index=False)

print(f"Input rows: {len(df)}")
print(f"Fixations: {len(fix)} -> {OUT_FIX.name}")
print(f"Saccades:  {len(sac)} -> {OUT_SAC.name}")
print("\nFixations preview:")

print(fix.head(10))
