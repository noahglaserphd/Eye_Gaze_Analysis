# app.py
# Interactive gaze overlay analytics dashboard (Streamlit)
#
# AOI drawing uses streamlit-drawable-canvas-fix (rect tool).
# Install:
#   pip install streamlit plotly pandas numpy opencv-python matplotlib pillow
#   pip install streamlit-drawable-canvas-fix
#   conda install -c conda-forge ffmpeg
# PNG: Kaleido 1.x + Plotly 6+ can fail on some Windows setups; HTML export always works.
# FFmpeg: if `ffmpeg` is not on PATH (e.g. Cursor/Streamlit started outside conda), set
# FFMPEG_BINARY to the full path of ffmpeg.exe, or use conda-forge ffmpeg in the env.
#
# Run:
#   streamlit run app.py

from __future__ import annotations

from pathlib import Path
import functools
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from core.dashboard_aoi import add_aoi_shapes_to_scanpath, aoi_summary_table, assign_aoi_rects, normalize_aois, sanitize_aoi_label
from core.dashboard_charts import make_heatmap, make_hist_fix_dur, make_hist_sac_amp, make_scanpath
from core.pipeline_config import HEATMAP_BINS_STREAMLIT as HEATMAP_BINS, WINDOW_SIZE
from core.pipeline_runner import (
    ensure_project_dirs,
    run_aggregate_figure,
    run_analyze_gaze_csvs,
    run_extract_from_videos,
    run_full_pipeline,
    run_session_figures,
    run_summarize_fixations,
    run_time_windows,
)
from core.window_utils import compute_window_metrics, compute_windows_from_fix, subset_window

try:
    import cv2  # type: ignore
    from PIL import Image  # type: ignore

    HAS_FRAME = True
except Exception:
    HAS_FRAME = False

try:
    # NOTE: you installed streamlit-drawable-canvas-fix but import stays the same
    from streamlit_drawable_canvas import st_canvas  # type: ignore

    HAS_CANVAS = True
except Exception:
    HAS_CANVAS = False

# -----------------------------
# Config (defaults in pipeline_config.py)
# -----------------------------
SUPPORTED_VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".avi")

# Repo root (parent of this `core/` package)
ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = ROOT / "videos"
GAZE_DIR = ROOT / "gaze_samples"
FIX_DIR = ROOT / "fixations"
SAC_DIR = ROOT / "saccades"
WIN_DIR = ROOT / "time_windows"

AOI_DIR = ROOT / "aoi"
AOI_FILE = AOI_DIR / "aoi_definitions.json"

CLIP_DIR = ROOT / "cache_clips"
EXPORT_DIR = ROOT / "exports"

ensure_project_dirs()
CLIP_DIR.mkdir(exist_ok=True)
EXPORT_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="Gaze Overlay Analytics", layout="wide")
st.title("Interactive Gaze Overlay Analytics")

if "pipeline_log" not in st.session_state:
    st.session_state.pipeline_log = ""


# -----------------------------
# Helpers
# -----------------------------
def safe_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s)


def is_safe_session_id(s: str, max_len: int = 240) -> bool:
    """Reject path-like session stems (e.g. '..', separators) from glob-derived names."""
    if not s or len(s) > max_len:
        return False
    if ".." in s or "/" in s or "\\" in s:
        return False
    return True


def assert_path_under(path: Path, root: Path) -> Path:
    """Ensure resolved path stays under root (prevents path traversal via symlinks or joins)."""
    path = path.resolve()
    root = root.resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Resolved path is outside allowed directory: {path}")
    return path


def _sanitize_upload_basename(name: str) -> str:
    base = Path(name).name
    if not base or base in (".", ".."):
        base = "upload"
    stem = safe_name(Path(base).stem)
    suf = Path(base).suffix.lower()
    return f"{stem}{suf}" if suf else stem


def _as_upload_file_list(uploaded) -> list:
    """Normalize Streamlit file_uploader return value (None, one file, list, or tuple)."""
    if uploaded is None:
        return []
    if isinstance(uploaded, (list, tuple)):
        return list(uploaded)
    return [uploaded]


def _unique_basename_in_dir(directory: Path, basename: str) -> str:
    """Pick a basename under directory that does not yet exist (append _2, _3, … before suffix)."""
    p = Path(basename)
    stem, suf = p.stem, p.suffix
    candidate = f"{stem}{suf}"
    n = 1
    while (directory / candidate).exists():
        n += 1
        candidate = f"{stem}_{n}{suf}"
    return candidate


def save_uploads_to_videos(uploaded) -> list[str]:
    """Write uploaded video files into videos/; returns basenames saved."""
    files = _as_upload_file_list(uploaded)
    if not files:
        return []
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for uf in files:
        base = _sanitize_upload_basename(uf.name)
        if not any(base.lower().endswith(ext) for ext in SUPPORTED_VIDEO_EXTS):
            continue
        out_name = _unique_basename_in_dir(VIDEO_DIR, base)
        dest = (VIDEO_DIR / out_name).resolve()
        assert_path_under(dest, VIDEO_DIR.resolve())
        data = uf.getvalue() if hasattr(uf, "getvalue") else uf.read()
        dest.write_bytes(data)
        saved.append(out_name)
    return saved


def save_uploads_to_gaze_samples(uploaded) -> list[str]:
    """Write uploaded gaze CSVs into gaze_samples/; returns basenames saved."""
    files = _as_upload_file_list(uploaded)
    if not files:
        return []
    GAZE_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for uf in files:
        base = _sanitize_upload_basename(uf.name)
        if not base.lower().endswith(".csv"):
            continue
        out_name = _unique_basename_in_dir(GAZE_DIR, base)
        dest = (GAZE_DIR / out_name).resolve()
        assert_path_under(dest, GAZE_DIR.resolve())
        data = uf.getvalue() if hasattr(uf, "getvalue") else uf.read()
        dest.write_bytes(data)
        saved.append(out_name)
    return saved


def _subprocess_no_window_kw() -> dict:
    if sys.platform == "win32":
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
    return {}


@functools.lru_cache(maxsize=16)
def _resolve_ffmpeg_cached(path_sig: str) -> str | None:
    """Find ffmpeg executable (PATH, conda env, or FFMPEG_BINARY / IMAGEIO_FFMPEG_EXE)."""
    seen: set[str] = set()
    candidates: list[str] = []

    for key in ("FFMPEG_BINARY", "IMAGEIO_FFMPEG_EXE"):
        v = os.environ.get(key)
        if v and v not in seen:
            candidates.append(os.path.expandvars(v))
            seen.add(candidates[-1])

    for name in ("ffmpeg", "ffmpeg.exe"):
        w = shutil.which(name)
        if w and w not in seen:
            candidates.append(w)
            seen.add(w)

    for prefix in filter(None, (os.environ.get("CONDA_PREFIX"), str(sys.prefix))):
        root = Path(prefix)
        for sub in (
            "Library/bin/ffmpeg.exe",
            "Library/bin/ffmpeg",
            "Scripts/ffmpeg.exe",
            "bin/ffmpeg.exe",
            "bin/ffmpeg",
        ):
            p = (root / sub).resolve()
            if p.is_file():
                sp = str(p)
                if sp not in seen:
                    candidates.append(sp)
                    seen.add(sp)

    kw = _subprocess_no_window_kw()
    for exe in candidates:
        try:
            subprocess.run(
                [exe, "-version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
                timeout=8,
                **kw,
            )
            return exe
        except Exception:
            continue
    return None


def _resolve_ffmpeg() -> str | None:
    sig = "|".join(
        (
            os.environ.get("FFMPEG_BINARY", ""),
            os.environ.get("IMAGEIO_FFMPEG_EXE", ""),
            os.environ.get("CONDA_PREFIX", ""),
            str(sys.prefix),
            shutil.which("ffmpeg") or "",
            shutil.which("ffmpeg.exe") or "",
        )
    )
    return _resolve_ffmpeg_cached(sig)


def run_ffmpeg(cmd: list[str]) -> None:
    """Run ffmpeg; on failure include stderr in the exception (not swallowed)."""
    exe = _resolve_ffmpeg()
    if not exe:
        raise RuntimeError(
            "ffmpeg not found. Install ffmpeg, add it to PATH, or set FFMPEG_BINARY to the full path "
            "to ffmpeg.exe (conda-forge: <env>\\Library\\bin\\ffmpeg.exe)."
        )
    if not cmd:
        raise ValueError("ffmpeg command is empty")
    cmd = [exe, *cmd[1:]]
    r = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        **_subprocess_no_window_kw(),
    )
    if r.returncode != 0:
        err = (r.stderr or "").strip()
        tail = err[-800:] if len(err) > 800 else err
        msg = f"ffmpeg exited with code {r.returncode}"
        if tail:
            msg += f": {tail}"
        raise RuntimeError(msg)


def mmss(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}:{s:02d}"


def window_label_pathsafe(start: float, end: float) -> str:
    m1, s1 = int(start // 60), int(start % 60)
    m2, s2 = int(end // 60), int(end % 60)
    return f"{m1:02d}-{s1:02d}_{m2:02d}-{s2:02d}"


def window_label_human(start: float, end: float) -> str:
    return f"{mmss(start)}–{mmss(end)}"


def ffmpeg_available() -> bool:
    return _resolve_ffmpeg() is not None


# -----------------------------
# Loading
# -----------------------------
@st.cache_data
def list_sessions() -> list[str]:
    fix_files = sorted(FIX_DIR.glob("*_fixations.csv"))
    out: list[str] = []
    for p in fix_files:
        s = p.stem.replace("_fixations", "")
        if not is_safe_session_id(s):
            continue
        sac_p = SAC_DIR / f"{s}_saccades.csv"
        if sac_p.exists():
            out.append(s)
    return out


def _session_data_paths(session: str) -> tuple[Path, Path]:
    if not is_safe_session_id(session):
        raise ValueError("Invalid session id")
    fix_p = FIX_DIR / f"{session}_fixations.csv"
    sac_p = SAC_DIR / f"{session}_saccades.csv"
    assert_path_under(fix_p, FIX_DIR)
    assert_path_under(sac_p, SAC_DIR)
    return fix_p, sac_p


@st.cache_data
def load_fix(session: str) -> pd.DataFrame:
    fix_p, _ = _session_data_paths(session)
    return pd.read_csv(fix_p)


@st.cache_data
def load_sac(session: str) -> pd.DataFrame:
    _, sac_p = _session_data_paths(session)
    return pd.read_csv(sac_p)


@st.cache_data
def load_windows_if_exists(session: str) -> pd.DataFrame | None:
    if not is_safe_session_id(session):
        return None
    f = WIN_DIR / f"{session}_windows.csv"
    if not f.exists():
        return None
    assert_path_under(f, WIN_DIR)
    return pd.read_csv(f)


def refresh_data_caches() -> None:
    """Clear only data readers used by Explore, not every cache in the app."""
    list_sessions.clear()
    load_fix.clear()
    load_sac.clear()
    load_windows_if_exists.clear()


# -----------------------------
# Video discovery / ffmpeg
# -----------------------------
def find_video_for_session(session: str) -> Path | None:
    if not is_safe_session_id(session) or not VIDEO_DIR.exists():
        return None
    for ext in SUPPORTED_VIDEO_EXTS:
        p = VIDEO_DIR / f"{session}{ext}"
        if p.exists():
            return assert_path_under(p, VIDEO_DIR)
    for p in VIDEO_DIR.iterdir():
        if p.is_file() and p.suffix.lower() in SUPPORTED_VIDEO_EXTS and p.stem.startswith(session):
            return assert_path_under(p, VIDEO_DIR)
    return None


def make_clip_ffmpeg(src: Path, start_s: float, end_s: float, out_mp4: Path) -> Path:
    assert_path_under(src, VIDEO_DIR)
    assert_path_under(out_mp4, CLIP_DIR)
    dur = max(0.1, end_s - start_s)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_s:.3f}",
        "-i",
        str(src),
        "-t",
        f"{dur:.3f}",
        "-an",
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryfast",
        str(out_mp4),
    ]
    run_ffmpeg(cmd)
    return out_mp4


def export_full_video_to_mp4(src: Path, out_mp4: Path) -> Path:
    assert_path_under(src, VIDEO_DIR)
    assert_path_under(out_mp4, EXPORT_DIR)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryfast",
        "-an",
        str(out_mp4),
    ]
    run_ffmpeg(cmd)
    return out_mp4


# -----------------------------
# AOIs
# -----------------------------
@st.cache_data
def load_aois() -> list[dict]:
    if not AOI_FILE.exists():
        return []
    try:
        data = json.loads(AOI_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []
    aois = data.get("aois", [])
    return normalize_aois(aois if isinstance(aois, list) else [])


def save_aois(aois: list[dict]) -> None:
    AOI_DIR.mkdir(exist_ok=True)
    norm = normalize_aois(aois)
    payload = {"coordinate_space": "norm", "aois": norm}
    AOI_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    load_aois.clear()


# -----------------------------
# Frame grab (for AOI canvas)
# -----------------------------
@st.cache_data
def get_reference_frame(video_path: Path, t_s: float = 1.0):
    if not HAS_FRAME:
        return None, None, None
    assert_path_under(Path(video_path), VIDEO_DIR)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, None, None
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 60.0
    frame_idx = int(max(0.0, float(t_s)) * float(fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None, None, None
    h, w = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb), int(w), int(h)


# -----------------------------
# Charts
# -----------------------------
# -----------------------------
# Export helpers
# -----------------------------
def export_plot(fig, out_base: Path):
    html_path = out_base.with_suffix(".html")
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    png_path = out_base.with_suffix(".png")
    try:
        pio.write_image(fig, str(png_path), width=1200, height=800, scale=2)
    except Exception:
        # HTML already written. Kaleido can fail on Windows (Plotly 6 + Kaleido 1) or version skew.
        pass


def export_window_assets(session: str, start: float, end: float, figs: dict, preview_clip: Path | None) -> Path:
    out_dir = EXPORT_DIR / safe_name(session) / window_label_pathsafe(start, end)
    out_dir.mkdir(parents=True, exist_ok=True)
    assert_path_under(out_dir, EXPORT_DIR)

    export_plot(figs["heatmap"], out_dir / "heatmap")
    export_plot(figs["scanpath"], out_dir / "scanpath")
    export_plot(figs["fixhist"], out_dir / "fix_dur_hist")
    export_plot(figs["sachist"], out_dir / "sac_amp_hist")

    if preview_clip and preview_clip.exists():
        assert_path_under(preview_clip, CLIP_DIR)
        shutil.copyfile(preview_clip, out_dir / "clip.mp4")

    return out_dir


def export_full_session_assets(session: str) -> str:
    if not is_safe_session_id(session):
        raise ValueError("Invalid session id")
    fix = load_fix(session)
    sac = load_sac(session)

    out_dir = EXPORT_DIR / safe_name(session) / "FULL"
    out_dir.mkdir(parents=True, exist_ok=True)
    assert_path_under(out_dir, EXPORT_DIR)

    export_plot(make_heatmap(fix, HEATMAP_BINS), out_dir / "heatmap_full")
    export_plot(make_scanpath(fix), out_dir / "scanpath_full")
    export_plot(make_hist_fix_dur(fix), out_dir / "fix_dur_hist_full")
    export_plot(make_hist_sac_amp(sac), out_dir / "sac_amp_hist_full")

    # --- NEW: AOI export (full session) ---
    aois = load_aois()
    if aois:
        fix_all_aoi = assign_aoi_rects(fix, aois)
        aoi_summary_table(fix_all_aoi).to_csv(out_dir / "aoi_summary_full.csv", index=False)
    # --------------------------------------

    src_video = find_video_for_session(session)
    if src_video is None:
        return f"{session}: full charts exported; AOI CSV exported (if AOIs exist); no source video found in videos/."
    if not ffmpeg_available():
        return f"{session}: full charts exported; AOI CSV exported (if AOIs exist); ffmpeg not available for full video export."

    out_mp4 = out_dir / f"{safe_name(session)}_full.mp4"
    if not out_mp4.exists():
        export_full_video_to_mp4(src_video, out_mp4)

    return f"{session}: full charts exported; AOI CSV exported (if AOIs exist); full video exported."


def run_pipeline_step(action, spinner_text: str, success_text: str | None = None) -> None:
    lines: list[str] = []

    def lg(msg: str) -> None:
        lines.append(msg)

    try:
        with st.spinner(spinner_text):
            action(log=lg)
        st.session_state.pipeline_log = "\n".join(lines)
        refresh_data_caches()
        if success_text:
            st.success(success_text)
        st.rerun()
    except Exception as e:
        st.session_state.pipeline_log = "\n".join(lines) + f"\nError: {e}"
        st.error(str(e))


# -----------------------------
# Sessions + sidebar (pipeline + explore)
# -----------------------------
sessions = list_sessions()
has_sessions = len(sessions) > 0

with st.sidebar:
    with st.expander("Upload data", expanded=False):
        st.caption("Save files into `videos/` and/or `gaze_samples/`, then run the pipeline below.")
        up_videos = st.file_uploader(
            "Videos",
            type=["mp4", "mkv", "mov", "avi"],
            accept_multiple_files=True,
            key="upload_videos",
        )
        up_csvs = st.file_uploader(
            "Gaze CSVs",
            type=["csv"],
            accept_multiple_files=True,
            key="upload_gaze_csvs",
        )
        if st.button("Save uploads to project folders", use_container_width=True, key="save_uploads_btn"):
            msgs: list[str] = []
            try:
                v_saved = save_uploads_to_videos(up_videos)
                if v_saved:
                    msgs.append("videos/: " + ", ".join(v_saved))
                c_saved = save_uploads_to_gaze_samples(up_csvs)
                if c_saved:
                    msgs.append("gaze_samples/: " + ", ".join(c_saved))
                if msgs:
                    st.success("Saved:\n" + "\n".join(msgs))
                else:
                    st.warning("No files selected, or no files matched the allowed types.")
            except Exception as e:
                st.error(f"Upload failed: {e}")

    with st.expander("Data pipeline (run here — no terminal)", expanded=not has_sessions):
        st.caption(
            "Place videos in `videos/` **or** gaze CSVs in `gaze_samples/` (or use **Upload data**), then run. "
            "Requires OpenCV for video extraction; FFmpeg for clips in Explore."
        )
        if st.button("Run full pipeline", type="primary", use_container_width=True):
            run_pipeline_step(run_full_pipeline, "Running full pipeline…", "Pipeline finished.")

        st.markdown("**Individual steps**")
        if st.button("Videos → events", use_container_width=True, help="Decode gaze from videos/"):
            run_pipeline_step(run_extract_from_videos, "Extracting from videos…", "Video extraction finished.")

        if st.button("Gaze CSVs → events", use_container_width=True, help="I-DT on gaze_samples/*.csv"):
            run_pipeline_step(run_analyze_gaze_csvs, "Processing gaze CSVs…", "Done.")

        if st.button("Summarize fixations", use_container_width=True):
            run_pipeline_step(run_summarize_fixations, "Summarizing…")

        if st.button("Time windows", use_container_width=True):
            run_pipeline_step(run_time_windows, "Computing windows…")

        if st.button("Session PNG figures", use_container_width=True):
            run_pipeline_step(run_session_figures, "Building figures…")

        if st.button("Aggregate figure", use_container_width=True):
            run_pipeline_step(run_aggregate_figure, "Building aggregate…")

        st.text_area(
            "Last pipeline log",
            value=st.session_state.pipeline_log,
            height=160,
            disabled=True,
            label_visibility="collapsed",
        )

    st.divider()
    st.header("Explore")
    if has_sessions:
        session = st.selectbox("Session", sessions, key="explore_session")

        fix = load_fix(session)
        sac = load_sac(session)

        win_df = load_windows_if_exists(session)
        if win_df is None:
            win_df = compute_windows_from_fix(fix, WINDOW_SIZE)
        if len(win_df) == 0:
            st.warning("No windows available (empty fixation file).")
            st.stop()

        win_df = compute_window_metrics(fix, sac, win_df)
        win_df = win_df.sort_values("window_start").reset_index(drop=True)

        st.divider()
        st.header("Exports")
        st.caption(
            "Exports write interactive HTML for every chart. PNG is added when static image export succeeds "
            "(Kaleido). On Windows, Plotly 6 + Kaleido 1.x may fail; use `plotly>=5.18,<6` and `kaleido>=0.2.1,<1`, "
            "or rely on HTML. FFmpeg: if clips/full video fail, set **FFMPEG_BINARY** to your `ffmpeg.exe` path "
            "when the app is not started from an activated conda shell."
        )
        export_window_btn = st.button("Export charts + clip (selected window)", use_container_width=True)
        export_all_full_btn = st.button("Export FULL outputs for ALL sessions", use_container_width=True)
    else:
        session = None
        fix = sac = win_df = None
        export_window_btn = False
        export_all_full_btn = False
        st.caption("Run the pipeline above to create sessions.")

if not has_sessions:
    st.info(
        "No fixation data yet. Add video files under `videos/` (or gaze CSVs under `gaze_samples/`), "
        "or use **Upload data** in the sidebar, then open **Data pipeline** and click **Run full pipeline**."
    )
    st.stop()

# -----------------------------
# Timeline + window selection
# -----------------------------
st.markdown("### Session timeline")

t = win_df["window_start"].astype(float)
timeline_fig = go.Figure()
timeline_fig.add_trace(go.Scatter(x=t, y=win_df["fixation_count"], mode="lines+markers", name="Fixation count"))
timeline_fig.add_trace(go.Scatter(x=t, y=win_df["mean_fix_duration"], mode="lines+markers", name="Mean fixation duration (s)", yaxis="y2"))
timeline_fig.add_trace(go.Scatter(x=t, y=win_df["mean_saccade_amp"], mode="lines+markers", name="Mean saccade amplitude", yaxis="y3"))

timeline_fig.update_layout(
    height=320,
    margin=dict(l=10, r=10, t=10, b=10),
    legend=dict(orientation="h"),
    xaxis=dict(title="time (s)"),
    yaxis=dict(title="fixations / window"),
    yaxis2=dict(title="mean fix dur (s)", overlaying="y", side="right", anchor="x"),
    yaxis3=dict(title="mean sacc amp", overlaying="y", side="right", anchor="free", position=1.0),
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="white"),
)
timeline_fig.update_xaxes(gridcolor="rgba(255,255,255,0.12)")
timeline_fig.update_yaxes(gridcolor="rgba(255,255,255,0.12)")
st.plotly_chart(timeline_fig, use_container_width=True)

labels = [window_label_human(float(r.window_start), float(r.window_end)) for r in win_df.itertuples(index=False)]
idx = st.selectbox(
    f"Select {WINDOW_SIZE:g}-second window",
    options=list(range(len(labels))),
    index=0,
    format_func=lambda i: labels[i],
    key="window_dropdown_main",
)

start = float(win_df.loc[idx, "window_start"])
end = float(win_df.loc[idx, "window_end"])
fix_w, sac_w = subset_window(fix, sac, start, end)

aois = load_aois()

# Window AOI-labeled fixations
fix_w_aoi = assign_aoi_rects(fix_w, aois)

# FULL SESSION AOI-labeled fixations (this is the new part)
fix_all_aoi = assign_aoi_rects(fix, aois)

# -----------------------------
# Charts
# -----------------------------
fig_heat = make_heatmap(fix_w_aoi, HEATMAP_BINS)
fig_scan = add_aoi_shapes_to_scanpath(make_scanpath(fix_w_aoi), aois)
fig_fixhist = make_hist_fix_dur(fix_w_aoi)
fig_sachist = make_hist_sac_amp(sac_w)
figs = {"heatmap": fig_heat, "scanpath": fig_scan, "fixhist": fig_fixhist, "sachist": fig_sachist}

# -----------------------------
# Video preview clip + export actions
# -----------------------------
status = st.empty()
preview_clip = None
src_video = find_video_for_session(session)

if src_video is not None and ffmpeg_available():
    preview_clip = CLIP_DIR / f"{safe_name(session)}_{int(start):06d}_{int(end):06d}.mp4"
    if not preview_clip.exists():
        try:
            make_clip_ffmpeg(src_video, start, end, preview_clip)
        except Exception as e:
            status.error(f"Video preview clip failed: {e}")
            preview_clip = None

if export_window_btn:
    try:
        out_dir = export_window_assets(session, start, end, figs, preview_clip)
        status.success(f"Exported to: {out_dir}")
    except Exception as e:
        status.error(f"Export failed: {e}")

if export_all_full_btn:
    msgs = []
    with st.spinner("Exporting full outputs for all sessions…"):
        for sess in sessions:
            try:
                msgs.append(export_full_session_assets(sess))
            except Exception as e:
                msgs.append(f"{sess}: export failed ({e})")
    status.success("Batch export complete:\n" + "\n".join(msgs))

# -----------------------------
# 2x2 layout
# -----------------------------
row1_c1, row1_c2 = st.columns(2)
with row1_c1:
    st.plotly_chart(fig_heat, use_container_width=True)
with row1_c2:
    st.plotly_chart(fig_scan, use_container_width=True)

row2_c1, row2_c2 = st.columns(2)
with row2_c1:
    st.plotly_chart(fig_fixhist, use_container_width=True)
with row2_c2:
    st.plotly_chart(fig_sachist, use_container_width=True)

# -----------------------------
# AOI summaries
# -----------------------------
st.markdown("### AOI summary (selected window)")
if not AOI_FILE.exists():
    st.info("AOIs not enabled yet.")
elif not aois:
    st.warning("AOI file exists but no valid AOIs were found (needs name/x0/y0/x1/y1).")
else:
    st.dataframe(aoi_summary_table(fix_w_aoi), use_container_width=True)

st.markdown("### AOI summary (full session)")
if not AOI_FILE.exists():
    st.info("AOIs not enabled yet.")
elif not aois:
    st.warning("AOI file exists but no valid AOIs were found (needs name/x0/y0/x1/y1).")
else:
    st.dataframe(aoi_summary_table(fix_all_aoi), use_container_width=True)

# -----------------------------
# AOI editor (canvas)
# -----------------------------
with st.expander("AOI editor (draw rectangles and save)", expanded=False):
    if not HAS_CANVAS:
        st.error("Install: pip install streamlit-drawable-canvas-fix")
    elif not HAS_FRAME:
        st.error("Install: pip install opencv-python pillow")
    elif src_video is None:
        st.info("No matching video found for this session.")
    else:
        frame_time = st.number_input("Reference frame time (seconds)", min_value=0.0, value=1.0, step=0.5)
        bg, W, H = get_reference_frame(src_video, t_s=float(frame_time))
        if bg is None:
            st.warning("Could not read frame.")
        else:
            st.caption("Draw rectangles; then click Save.")
            canvas_key = f"aoi_canvas_{safe_name(session)}"

            display_w = min(W, 1100)
            scale = display_w / W
            display_h = int(H * scale)
            bg_disp = bg.resize((display_w, display_h))

            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 0.06)",
                stroke_width=2,
                stroke_color="rgba(255, 255, 255, 0.9)",
                background_image=bg_disp,
                update_streamlit=True,
                height=display_h,
                width=display_w,
                drawing_mode="rect",
                key=canvas_key,
            )

            objs = []
            if canvas_result and canvas_result.json_data and "objects" in canvas_result.json_data:
                objs = canvas_result.json_data["objects"] or []

            aois_from_canvas: list[dict] = []
            for o in objs:
                if not isinstance(o, dict) or o.get("type") != "rect":
                    continue

                left = float(o.get("left", 0.0)) / scale
                top = float(o.get("top", 0.0)) / scale
                width = float(o.get("width", 0.0)) / scale
                height = float(o.get("height", 0.0)) / scale

                x0p = max(0.0, min(float(W), left))
                y0p = max(0.0, min(float(H), top))
                x1p = max(0.0, min(float(W), left + width))
                y1p = max(0.0, min(float(H), top + height))
                if (x1p - x0p) < 2 or (y1p - y0p) < 2:
                    continue

                aois_from_canvas.append(
                    {"name": f"AOI_{len(aois_from_canvas)+1}", "x0": x0p / W, "y0": y0p / H, "x1": x1p / W, "y1": y1p / H}
                )

            if aois_from_canvas:
                df = pd.DataFrame(aois_from_canvas)
                edited = st.data_editor(df, use_container_width=True, num_rows="fixed")
                aois_final = edited.to_dict(orient="records")
            else:
                aois_final = []
                st.caption("No rectangles detected yet.")

            c1, c2 = st.columns(2)
            with c1:
                if st.button("Save AOIs to aoi/aoi_definitions.json", use_container_width=True):
                    save_aois(aois_final)
                    st.success(f"Saved {len(aois_final)} AOIs.")
                    st.rerun()
            with c2:
                if st.button("Clear AOI file", use_container_width=True):
                    save_aois([])
                    st.success("Cleared AOIs.")
                    st.rerun()

# -----------------------------
# Video at bottom
# -----------------------------
st.markdown("### Video (selected window)")
if src_video is None:
    st.info(f"No matching video found for session '{session}'. Put the recording in `videos/` as `{session}.mp4` (or .mkv/.mov/.avi).")
else:
    if not ffmpeg_available():
        st.error(
            "FFmpeg not found for this process (PATH / conda). Install ffmpeg, restart from an activated env, "
            "or set environment variable **FFMPEG_BINARY** to the full path of ffmpeg.exe."
        )
    elif preview_clip and preview_clip.exists():
        assert_path_under(preview_clip, CLIP_DIR)
        st.video(str(preview_clip))
    else:
        st.caption("No preview clip available for this window.")

with st.expander("Show window data tables"):
    st.markdown("#### Fixations in selected window")
    st.dataframe(fix_w_aoi, use_container_width=True)
    st.markdown("#### Saccades in selected window")
    st.dataframe(sac_w, use_container_width=True)
