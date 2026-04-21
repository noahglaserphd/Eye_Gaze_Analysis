"""
Run the gaze analysis pipeline programmatically (CLI scripts and Streamlit dashboard).

Uses matplotlib's non-interactive Agg backend when generating PNG figures.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from collections.abc import Callable
from pathlib import Path
from typing import Optional

import pandas as pd

from extract_dot import (
    FIX_DIR,
    GAZE_DIR,
    METRICS_DIR,
    ROOT,
    SAC_DIR,
    VIDEO_DIR,
    VIDEO_EXTS,
    extract_gaze_from_video,
    summarize_fixations,
)
from extract_dot import ensure_pipeline_dirs as ensure_extract_dirs
from gaze_detection import (
    DISPERSION_THRESH_NORM,
    MIN_FIX_DURATION_S,
    detect_fixations_idt,
    saccades_from_fixations,
)

LogFn = Optional[Callable[[str], None]]


def _emit(msg: str, log: LogFn, lines: list[str]) -> None:
    lines.append(msg)
    if log:
        log(msg)


def _session_stems_from_glob(folder: Path, pattern: str, suffix: str) -> set[str]:
    stems: set[str] = set()
    for p in folder.glob(pattern):
        if p.is_file():
            stems.add(p.stem.removesuffix(suffix))
    return stems


def _cleanup_orphans(directory: Path, pattern: str, keep_stems: set[str], stem_suffix: str, log: LogFn, lines: list[str]) -> None:
    for p in sorted(directory.glob(pattern)):
        if not p.is_file():
            continue
        stem = p.stem.removesuffix(stem_suffix)
        if stem not in keep_stems:
            p.unlink(missing_ok=True)
            _emit(f"Removed stale file: {p.relative_to(ROOT)}", log, lines)


def ensure_project_dirs() -> None:
    """Create all project folders used by the pipeline and dashboard."""
    ensure_extract_dirs()
    for name in (
        "videos",
        "time_windows",
        "figures",
        "aoi",
        "cache_clips",
        "exports",
    ):
        (ROOT / name).mkdir(parents=True, exist_ok=True)


def run_extract_from_videos(log: LogFn = None) -> list[str]:
    """videos/* → gaze_samples/*.csv, fixations/, saccades/, metrics/fixation_summary.csv"""
    lines: list[str] = []
    try:
        import cv2  # noqa: F401
    except ImportError:
        raise ValueError("OpenCV is required for video extraction. Install: pip install opencv-python") from None

    ensure_project_dirs()
    if not VIDEO_DIR.exists():
        raise ValueError(f"Missing folder: {VIDEO_DIR}")

    videos = sorted([p for p in VIDEO_DIR.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS])
    if not videos:
        raise ValueError(f"No supported videos in {VIDEO_DIR}. Extensions: {sorted(VIDEO_EXTS)}")

    keep_stems = {vp.stem for vp in videos}
    _cleanup_orphans(GAZE_DIR, "*.csv", keep_stems, "", log, lines)
    _cleanup_orphans(FIX_DIR, "*_fixations.csv", keep_stems, "_fixations", log, lines)
    _cleanup_orphans(SAC_DIR, "*_saccades.csv", keep_stems, "_saccades", log, lines)

    summary_rows = []
    for vp in videos:
        stem = vp.stem
        gaze_csv = GAZE_DIR / f"{stem}.csv"
        fix_csv = FIX_DIR / f"{stem}_fixations.csv"
        sac_csv = SAC_DIR / f"{stem}_saccades.csv"

        _emit(f"=== {vp.name} ===", log, lines)
        df_gaze = extract_gaze_from_video(vp, gaze_csv)
        _emit(f"  gaze samples: {len(df_gaze)} → {gaze_csv.relative_to(ROOT)}", log, lines)

        fix = detect_fixations_idt(df_gaze)
        fix.to_csv(fix_csv, index=False)
        _emit(f"  fixations: {len(fix)} → {fix_csv.relative_to(ROOT)}", log, lines)

        sac = saccades_from_fixations(fix)
        sac.to_csv(sac_csv, index=False)
        _emit(f"  saccades: {len(sac)} → {sac_csv.relative_to(ROOT)}", log, lines)

        metrics = summarize_fixations(fix)
        summary_rows.append({"file": fix_csv.name, **metrics})

    summary_df = pd.DataFrame(summary_rows)
    summary_out = METRICS_DIR / "fixation_summary.csv"
    summary_df.to_csv(summary_out, index=False)
    _emit(f"Saved summary → {summary_out.relative_to(ROOT)}", log, lines)
    return lines


def run_analyze_gaze_csvs(log: LogFn = None) -> list[str]:
    """gaze_samples/*.csv → fixations/ + saccades/ (re-run I-DT without decoding video)."""
    lines: list[str] = []
    ensure_project_dirs()
    fix_dir = FIX_DIR
    sac_dir = SAC_DIR
    fix_dir.mkdir(parents=True, exist_ok=True)
    sac_dir.mkdir(parents=True, exist_ok=True)

    if not GAZE_DIR.exists():
        raise ValueError(f"Missing folder: {GAZE_DIR}")

    gaze_files = sorted(GAZE_DIR.glob("*.csv"))
    if not gaze_files:
        raise ValueError(f"No gaze sample CSVs in {GAZE_DIR}")

    keep_stems = {p.stem for p in gaze_files}
    _cleanup_orphans(FIX_DIR, "*_fixations.csv", keep_stems, "_fixations", log, lines)
    _cleanup_orphans(SAC_DIR, "*_saccades.csv", keep_stems, "_saccades", log, lines)

    for gaze_path in gaze_files:
        stem = gaze_path.stem
        fix_path = fix_dir / f"{stem}_fixations.csv"
        sac_path = sac_dir / f"{stem}_saccades.csv"
        df = pd.read_csv(gaze_path)
        fix = detect_fixations_idt(
            df,
            dispersion_thresh=DISPERSION_THRESH_NORM,
            min_duration_s=MIN_FIX_DURATION_S,
        )
        sac = saccades_from_fixations(fix)
        fix.to_csv(fix_path, index=False)
        sac.to_csv(sac_path, index=False)
        _emit(f"{stem}: gaze rows={len(df)} fixations={len(fix)} saccades={len(sac)}", log, lines)

    return lines


def run_summarize_fixations(log: LogFn = None) -> list[str]:
    """fixations/* → metrics/fixation_summary.csv"""
    import numpy as np

    lines: list[str] = []
    ensure_project_dirs()
    fix_dir = FIX_DIR
    if not fix_dir.exists():
        raise ValueError(f"Missing folder: {fix_dir}")

    fix_files = sorted(fix_dir.glob("*_fixations.csv"))
    if not fix_files:
        raise ValueError(f"No fixation CSVs in {fix_dir}")

    rows = []
    for fix_file in fix_files:
        fix = pd.read_csv(fix_file)
        if len(fix) == 0 or "duration_s" not in fix.columns:
            rows.append(
                {
                    "file": fix_file.name,
                    "num_fixations": 0,
                    "mean_fixation_duration": np.nan,
                    "median_fixation_duration": np.nan,
                    "longest_fixation": np.nan,
                }
            )
            continue
        d = fix["duration_s"].to_numpy(dtype=float)
        rows.append(
            {
                "file": fix_file.name,
                "num_fixations": int(len(fix)),
                "mean_fixation_duration": float(np.nanmean(d)),
                "median_fixation_duration": float(np.nanmedian(d)),
                "longest_fixation": float(np.nanmax(d)),
            }
        )

    summary = pd.DataFrame(rows)
    out = METRICS_DIR / "fixation_summary.csv"
    summary.to_csv(out, index=False)
    _emit(f"Saved {out.relative_to(ROOT)}", log, lines)
    return lines


def run_time_windows(log: LogFn = None) -> list[str]:
    """fixations/ + saccades/ → time_windows/* + timeline PNGs."""
    import window_analysis

    lines: list[str] = []
    if not FIX_DIR.exists():
        raise ValueError(f"Missing folder: {FIX_DIR}")
    keep_stems = _session_stems_from_glob(FIX_DIR, "*_fixations.csv", "_fixations")
    tw_dir = ROOT / "time_windows"
    if tw_dir.exists():
        _cleanup_orphans(tw_dir, "*_windows.csv", keep_stems, "_windows", log, lines)
        _cleanup_orphans(tw_dir, "*_timeline.png", keep_stems, "_timeline", log, lines)

    window_analysis.main(log=lambda m: _emit(m, log, lines))
    return lines


def run_session_figures(log: LogFn = None) -> list[str]:
    """Per-session PNG summaries in figures/."""
    import make_figures

    lines: list[str] = []
    keep_stems = _session_stems_from_glob(GAZE_DIR, "*.csv", "")
    fig_dir = ROOT / "figures"
    if fig_dir.exists():
        _cleanup_orphans(fig_dir, "*_summary.png", keep_stems, "_summary", log, lines)

    make_figures.main(log=lambda m: _emit(m, log, lines))
    return lines


def run_aggregate_figure(log: LogFn = None) -> list[str]:
    """figures/ALL_sessions_summary.png"""
    import make_aggregate_figure

    lines: list[str] = []
    make_aggregate_figure.main(log=lambda m: _emit(m, log, lines))
    return lines


def run_full_pipeline(log: LogFn = None) -> list[str]:
    """
    Prefer video extraction if files exist in videos/; otherwise use gaze_samples/*.csv.
    Then: metrics summary → time windows → session figures → aggregate figure.
    """
    lines: list[str] = []
    ensure_project_dirs()

    video_files = sorted([p for p in VIDEO_DIR.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS]) if VIDEO_DIR.exists() else []
    gaze_files = sorted(GAZE_DIR.glob("*.csv")) if GAZE_DIR.exists() else []

    if video_files:
        lines.extend(run_extract_from_videos(log=log))
    elif gaze_files:
        _emit("No videos found; processing gaze_samples/*.csv only.", log, lines)
        lines.extend(run_analyze_gaze_csvs(log=log))
    else:
        raise ValueError(
            f"Add video files to {VIDEO_DIR} or gaze CSVs to {GAZE_DIR}, then run the pipeline again."
        )

    lines.extend(run_summarize_fixations(log=log))
    lines.extend(run_time_windows(log=log))
    lines.extend(run_session_figures(log=log))
    lines.extend(run_aggregate_figure(log=log))
    _emit("Full pipeline finished.", log, lines)
    return lines
