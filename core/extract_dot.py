# extract_dot.py
# Batch pipeline:
#   videos/*.mkv|mp4|mov|avi -> gaze_samples/<video>.csv -> fixations/<video>_fixations.csv
#   -> saccades/<video>_saccades.csv -> metrics/fixation_summary.csv
#
# Requirements (in your conda env):
#   pip install opencv-python pandas numpy

import numpy as np
import pandas as pd
from pathlib import Path

from core.gaze_detection import detect_fixations_idt, saccades_from_fixations

# -----------------------------
# Folder layout (relative to this script)
# -----------------------------
ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = ROOT / "videos"
GAZE_DIR = ROOT / "gaze_samples"
FIX_DIR = ROOT / "fixations"
SAC_DIR = ROOT / "saccades"
METRICS_DIR = ROOT / "metrics"


def ensure_pipeline_dirs() -> None:
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

# I-DT defaults: gaze_detection.DISPERSION_THRESH_NORM / MIN_FIX_DURATION_S

# Optional: cap extreme fixations for summary stats (set to None to disable)
CAP_FIXATION_AT_S = None  # e.g., 2.0


def extract_gaze_from_video(video_path: Path, out_csv: Path) -> pd.DataFrame:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 60.0

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Avoid division by zero if the backend reports invalid dimensions.
    w_eff = max(1, w)
    h_eff = max(1, h)

    kernel = np.ones((3, 3), np.uint8)
    rows = []
    frame_idx = 0
    last = None  # smoothed (x,y) in pixels

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(HSV_LOWER), np.array(HSV_UPPER))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

        # OpenCV 3 returns (image, contours, hierarchy); OpenCV 4 returns (contours, hierarchy).
        _fc = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = _fc[1] if len(_fc) == 3 else _fc[0]

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
                "gaze_x_norm": (x / w_eff) if np.isfinite(x) else np.nan,
                "gaze_y_norm": (y / h_eff) if np.isfinite(y) else np.nan,
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
    from core.pipeline_runner import run_extract_from_videos

    try:
        run_extract_from_videos(log=print)
    except ValueError as e:
        raise SystemExit(str(e)) from None


if __name__ == "__main__":
    main()
