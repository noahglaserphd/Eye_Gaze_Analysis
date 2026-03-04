# extract_dot.py
# Batch pipeline:
#   videos/*.mkv|mp4|mov|avi -> gaze_samples/<video>.csv -> fixations/<video>_fixations.csv
#   -> saccades/<video>_saccades.csv -> metrics/fixation_summary.csv
#
# Requirements (in your conda env):
#   pip install opencv-python pandas numpy

import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# -----------------------------
# Folder layout (relative to this script)
# -----------------------------
ROOT = Path(__file__).resolve().parent
VIDEO_DIR = ROOT / "videos"
GAZE_DIR = ROOT / "gaze_samples"
FIX_DIR = ROOT / "fixations"
SAC_DIR = ROOT / "saccades"
METRICS_DIR = ROOT / "metrics"

for d in (GAZE_DIR, FIX_DIR, SAC_DIR, METRICS_DIR):
    d.mkdir(parents=True, exist_ok=True)

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi"}

# -----------------------------
# Dot detection settings (tune these)
# -----------------------------
# These HSV bounds are a good starting point for a bright GREEN dot.
# If your dot is a different color, change HSV_LOWER/HSV_UPPER.
HSV_LOWER = (35, 120, 120)
HSV_UPPER = (85, 255, 255)

MIN_AREA = 40            # ignore blobs smaller than this
SMOOTH_ALPHA = 0.30      # 0..1, higher = more smoothing
SHOW_PREVIEW = False     # True to debug a single video visually (ESC to quit)

# -----------------------------
# Fixation detection settings (I-DT)
# -----------------------------
DISPERSION_THRESH_NORM = 0.006  # smaller is stricter; video-derived dot often needs 0.004–0.010
MIN_FIX_DURATION_S = 0.10

# Optional: cap extreme fixations for summary stats (set to None to disable)
CAP_FIXATION_AT_S = None  # e.g., 2.0


def extract_gaze_from_video(video_path: Path, out_csv: Path) -> pd.DataFrame:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 60.0

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    kernel = np.ones((3, 3), np.uint8)
    rows = []
    frame_idx = 0
    last = None  # smoothed (x,y) in pixels

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(HSV_LOWER), np.array(HSV_UPPER))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        x = y = np.nan
        conf = 0.0

        if contours:
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)

            if area >= MIN_AREA:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    x_raw = M["m10"] / M["m00"]
                    y_raw = M["m01"] / M["m00"]

                    if last is None:
                        x_s, y_s = x_raw, y_raw
                    else:
                        x_s = (1 - SMOOTH_ALPHA) * x_raw + SMOOTH_ALPHA * last[0]
                        y_s = (1 - SMOOTH_ALPHA) * y_raw + SMOOTH_ALPHA * last[1]

                    last = (x_s, y_s)
                    x, y = x_s, y_s
                    conf = float(min(1.0, area / (MIN_AREA * 10)))

                    if SHOW_PREVIEW:
                        cv2.circle(frame, (int(x), int(y)), 10, (0, 0, 255), 2)

        t = frame_idx / fps
        rows.append(
            {
                "timestamp": t,
                "frame_idx": frame_idx,
                "gaze_x_px": x,
                "gaze_y_px": y,
                "gaze_x_norm": (x / w) if np.isfinite(x) else np.nan,
                "gaze_y_norm": (y / h) if np.isfinite(y) else np.nan,
                "confidence": conf,
                "video_w": w,
                "video_h": h,
                "fps": fps,
            }
        )

        if SHOW_PREVIEW:
            cv2.imshow("frame (centroid)", frame)
            cv2.imshow("mask", mask)
            if (cv2.waitKey(1) & 0xFF) == 27:  # ESC
                break

        frame_idx += 1

    cap.release()
    if SHOW_PREVIEW:
        cv2.destroyAllWindows()

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


def detect_fixations_idt(
    df: pd.DataFrame,
    xcol="gaze_x_norm",
    ycol="gaze_y_norm",
    tcol="timestamp",
    dispersion_thresh=DISPERSION_THRESH_NORM,
    min_duration_s=MIN_FIX_DURATION_S,
) -> pd.DataFrame:
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

        while end_ptr < len(idxs) and (t[idxs[end_ptr]] - t[start_i]) < min_duration_s:
            end_ptr += 1
        if end_ptr >= len(idxs):
            break

        def dispersion(window):
            return (np.max(x[window]) - np.min(x[window])) + (np.max(y[window]) - np.min(y[window]))

        window = idxs[start_ptr : end_ptr + 1]
        while end_ptr < len(idxs) and dispersion(window) <= dispersion_thresh:
            end_ptr += 1
            if end_ptr < len(idxs):
                window = idxs[start_ptr : end_ptr + 1]

        window = idxs[start_ptr:end_ptr]  # exclude the violating point
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
    sacc = []
    if len(fix_df) < 2:
        return pd.DataFrame(sacc)

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


def summarize_fixations(fix_df: pd.DataFrame) -> dict:
    if len(fix_df) == 0:
        return {
            "num_fixations": 0,
            "mean_fixation_duration": np.nan,
            "median_fixation_duration": np.nan,
            "longest_fixation": np.nan,
        }

    durations = fix_df["duration_s"].to_numpy(dtype=float)

    if CAP_FIXATION_AT_S is not None:
        durations = np.minimum(durations, float(CAP_FIXATION_AT_S))

    return {
        "num_fixations": int(len(fix_df)),
        "mean_fixation_duration": float(np.mean(durations)),
        "median_fixation_duration": float(np.median(durations)),
        "longest_fixation": float(np.max(durations)),
    }


def main():
    if not VIDEO_DIR.exists():
        raise SystemExit(f"Missing folder: {VIDEO_DIR}\nCreate it and put videos inside.")

    videos = sorted([p for p in VIDEO_DIR.iterdir() if p.suffix.lower() in VIDEO_EXTS])
    if not videos:
        raise SystemExit(f"No videos found in: {VIDEO_DIR}\nSupported: {sorted(VIDEO_EXTS)}")

    summary_rows = []

    for vp in videos:
        stem = vp.stem

        gaze_csv = GAZE_DIR / f"{stem}.csv"
        fix_csv = FIX_DIR / f"{stem}_fixations.csv"
        sac_csv = SAC_DIR / f"{stem}_saccades.csv"

        print(f"\n=== {vp.name} ===")

        df_gaze = extract_gaze_from_video(vp, gaze_csv)
        print(f"gaze samples: {len(df_gaze)} -> {gaze_csv.relative_to(ROOT)}")

        fix = detect_fixations_idt(df_gaze)
        fix.to_csv(fix_csv, index=False)
        print(f"fixations:    {len(fix)} -> {fix_csv.relative_to(ROOT)}")

        sac = saccades_from_fixations(fix)
        sac.to_csv(sac_csv, index=False)
        print(f"saccades:     {len(sac)} -> {sac_csv.relative_to(ROOT)}")

        metrics = summarize_fixations(fix)
        summary_rows.append({"file": fix_csv.name, **metrics})

    summary_df = pd.DataFrame(summary_rows)
    summary_out = METRICS_DIR / "fixation_summary.csv"
    summary_df.to_csv(summary_out, index=False)
    print(f"\nSaved summary -> {summary_out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()