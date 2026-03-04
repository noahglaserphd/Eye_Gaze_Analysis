# app.py
# Interactive gaze overlay analytics dashboard (Streamlit)
#
# Includes:
# - Session timeline (fixed 30s windows) with "jump" selection:
#     - Click-to-jump if `streamlit-plotly-events` is installed
#     - Otherwise uses a slider
# - 2x2 chart layout for the selected 30s window:
#     - heatmap, scanpath, fixation duration histogram, saccade amplitude histogram
# - Video preview for the selected 30s window shown at the bottom (auto-generates cached clip)
# - Sidebar buttons:
#     - Export charts + clip for the selected window
#     - Export FULL outputs for ALL sessions (charts + full-length MP4)
#
# Folders (relative to this file):
#   videos/                     # source recordings (mp4/mkv/mov/avi)
#   fixations/*_fixations.csv
#   saccades/*_saccades.csv
# Optional:
#   time_windows/*_windows.csv  # if present, will be used; metrics will be computed if missing
#
# Outputs:
#   cache_clips/                # auto-generated window preview clips
#   exports/<session>/<window>/ # window exports (charts + clip)
#   exports/<session>/FULL/     # full-session exports (charts + full video)
#
# Requirements (dashboard env):
#   pip install streamlit plotly pandas numpy
#   conda install -c conda-forge ffmpeg
# Optional (PNG export):
#   pip install kaleido
# Optional (click-to-jump on timeline):
#   pip install streamlit-plotly-events
#
# Run:
#   streamlit run app.py

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# -----------------------------
# Config
# -----------------------------
WINDOW_SIZE = 30.0
HEATMAP_BINS = 120
SUPPORTED_VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".avi")

ROOT = Path(__file__).resolve().parent
VIDEO_DIR = ROOT / "videos"
FIX_DIR = ROOT / "fixations"
SAC_DIR = ROOT / "saccades"
WIN_DIR = ROOT / "time_windows"

CLIP_DIR = ROOT / "cache_clips"
EXPORT_DIR = ROOT / "exports"
CLIP_DIR.mkdir(exist_ok=True)
EXPORT_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="Gaze Overlay Analytics", layout="wide")
st.title("Interactive Gaze Overlay Analytics")

# -----------------------------
# Optional dependency: click events on Plotly
# -----------------------------
try:
    from streamlit_plotly_events import plotly_events  # type: ignore
    HAS_PLOTLY_EVENTS = True
except Exception:
    HAS_PLOTLY_EVENTS = False

# -----------------------------
# Helpers: formatting and safe names
# -----------------------------
def safe_name(s: str) -> str:
    # safe for Windows/macOS/Linux paths
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s)

def mmss(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}:{s:02d}"

def window_label_pathsafe(start: float, end: float) -> str:
    # Windows-safe (no colons)
    m1, s1 = int(start // 60), int(start % 60)
    m2, s2 = int(end // 60), int(end % 60)
    return f"{m1:02d}-{s1:02d}_{m2:02d}-{s2:02d}"

def window_label_human(start: float, end: float) -> str:
    return f"{mmss(start)}–{mmss(end)}"

# -----------------------------
# Helpers: environment / ffmpeg
# -----------------------------
def ffmpeg_available() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False

# -----------------------------
# Helpers: loading
# -----------------------------
@st.cache_data
def list_sessions() -> list[str]:
    fix_files = sorted(FIX_DIR.glob("*_fixations.csv"))
    return [p.stem.replace("_fixations", "") for p in fix_files]

@st.cache_data
def load_fix(session: str) -> pd.DataFrame:
    return pd.read_csv(FIX_DIR / f"{session}_fixations.csv")

@st.cache_data
def load_sac(session: str) -> pd.DataFrame:
    return pd.read_csv(SAC_DIR / f"{session}_saccades.csv")

@st.cache_data
def load_windows_if_exists(session: str) -> pd.DataFrame | None:
    f = WIN_DIR / f"{session}_windows.csv"
    return pd.read_csv(f) if f.exists() else None

def compute_windows_from_fix(fix: pd.DataFrame) -> pd.DataFrame:
    if len(fix) == 0:
        return pd.DataFrame(columns=["window_start", "window_end"])
    max_time = float(fix["end_s"].max())
    starts = np.arange(0, max_time + WINDOW_SIZE, WINDOW_SIZE)
    return pd.DataFrame(
        {"window_start": starts[:-1].astype(float), "window_end": (starts[:-1] + WINDOW_SIZE).astype(float)}
    )

def compute_window_metrics(fix: pd.DataFrame, sac: pd.DataFrame, win_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures per-window metrics exist for timeline:
      fixation_count, mean_fix_duration, total_fix_time, mean_saccade_amp
    If win_df already has them, keeps them.
    """
    needed = {"fixation_count", "mean_fix_duration", "total_fix_time", "mean_saccade_amp"}
    if needed.issubset(set(win_df.columns)):
        return win_df

    rows = []
    for r in win_df.itertuples(index=False):
        start = float(getattr(r, "window_start"))
        end = float(getattr(r, "window_end"))

        f = fix[(fix["start_s"] >= start) & (fix["start_s"] < end)]
        s = sac[(sac["start_s"] >= start) & (sac["start_s"] < end)]

        rows.append({
            "window_start": start,
            "window_end": end,
            "fixation_count": int(len(f)),
            "mean_fix_duration": float(f["duration_s"].mean()) if len(f) else 0.0,
            "total_fix_time": float(f["duration_s"].sum()) if len(f) else 0.0,
            "mean_saccade_amp": float(s["amplitude_norm"].mean()) if (len(s) and "amplitude_norm" in s.columns) else 0.0,
        })
    return pd.DataFrame(rows)

def subset_window(fix: pd.DataFrame, sac: pd.DataFrame, start: float, end: float):
    f = fix[(fix["start_s"] >= start) & (fix["start_s"] < end)].copy()
    s = sac[(sac["start_s"] >= start) & (sac["start_s"] < end)].copy()
    return f, s

# -----------------------------
# Helpers: video discovery and clipping
# -----------------------------
def find_video_for_session(session: str) -> Path | None:
    if not VIDEO_DIR.exists():
        return None
    # exact match
    for ext in SUPPORTED_VIDEO_EXTS:
        p = VIDEO_DIR / f"{session}{ext}"
        if p.exists():
            return p
    # prefix match
    for p in VIDEO_DIR.iterdir():
        if p.is_file() and p.suffix.lower() in SUPPORTED_VIDEO_EXTS and p.stem.startswith(session):
            return p
    return None

def make_clip_ffmpeg(src: Path, start_s: float, end_s: float, out_mp4: Path) -> Path:
    dur = max(0.1, end_s - start_s)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", f"{start_s:.3f}",
        "-i", str(src),
        "-t", f"{dur:.3f}",
        "-an",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        str(out_mp4),
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return out_mp4

def export_full_video_to_mp4(src: Path, out_mp4: Path) -> Path:
    # full-length export in consistent mp4/h264 for sharing
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(src),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        "-an",
        str(out_mp4),
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return out_mp4

# -----------------------------
# Helpers: charts
# -----------------------------
def make_heatmap(fix_df: pd.DataFrame, bins: int = HEATMAP_BINS):
    if len(fix_df) == 0:
        z = np.zeros((bins, bins))
    else:
        x = fix_df["x_norm"].to_numpy(float)
        y = fix_df["y_norm"].to_numpy(float)
        w = fix_df["duration_s"].to_numpy(float)
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
        x, y, w = x[valid], y[valid], w[valid]
        H, _, _ = np.histogram2d(x, y, bins=bins, range=[[0, 1], [0, 1]], weights=w)
        z = H.T
    fig = px.imshow(z, origin="upper", aspect="auto")
    fig.update_layout(margin=dict(l=10, r=10, t=35, b=10), title="Fixation heatmap (duration-weighted)")
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig

def make_scanpath(fix_df: pd.DataFrame):
    fig = go.Figure()
    fig.update_layout(
        title="Scanpath",
        xaxis_title="x (norm)",
        yaxis_title="y (norm)",
        margin=dict(l=10, r=10, t=35, b=10),
    )
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(autorange="reversed", range=[1, 0])

    if len(fix_df) == 0:
        return fig

    max_d = float(fix_df["duration_s"].max()) if float(fix_df["duration_s"].max()) > 0 else 1.0
    sizes = 8 + 30 * (fix_df["duration_s"] / max_d)

    fig.add_trace(go.Scatter(
        x=fix_df["x_norm"], y=fix_df["y_norm"],
        mode="lines", name="path", line=dict(width=1)
    ))
    fig.add_trace(go.Scatter(
        x=fix_df["x_norm"], y=fix_df["y_norm"],
        mode="markers", name="fixations",
        marker=dict(size=sizes, opacity=0.8),
        hovertemplate="start=%{customdata[0]:.2f}s<br>dur=%{customdata[1]:.3f}s<br>x=%{x:.3f}, y=%{y:.3f}<extra></extra>",
        customdata=np.column_stack([fix_df["start_s"], fix_df["duration_s"]]),
    ))
    return fig

def make_hist_fix_dur(fix_df: pd.DataFrame):
    fig = px.histogram(fix_df, x="duration_s", nbins=30, title="Fixation duration distribution")
    fig.update_layout(margin=dict(l=10, r=10, t=35, b=10))
    return fig

def make_hist_sac_amp(sac_df: pd.DataFrame):
    if len(sac_df) == 0 or "amplitude_norm" not in sac_df.columns:
        fig = go.Figure()
        fig.update_layout(margin=dict(l=10, r=10, t=35, b=10), title="Saccade amplitude distribution")
        return fig
    fig = px.histogram(sac_df, x="amplitude_norm", nbins=30, title="Saccade amplitude distribution")
    fig.update_layout(margin=dict(l=10, r=10, t=35, b=10))
    return fig

def window_summary(fix_w: pd.DataFrame, sac_w: pd.DataFrame):
    n_fix = len(fix_w)
    n_sac = len(sac_w)
    mean_fix = float(fix_w["duration_s"].mean()) if n_fix else float("nan")
    med_fix = float(fix_w["duration_s"].median()) if n_fix else float("nan")
    max_fix = float(fix_w["duration_s"].max()) if n_fix else float("nan")
    mean_amp = float(sac_w["amplitude_norm"].mean()) if (n_sac and "amplitude_norm" in sac_w.columns) else float("nan")
    return n_fix, mean_fix, med_fix, max_fix, n_sac, mean_amp

# -----------------------------
# Helpers: export
# -----------------------------
def can_export_png() -> bool:
    try:
        import kaleido  # noqa: F401
        return True
    except Exception:
        return False

def export_plot(fig, out_base: Path):
    # always HTML; PNG if kaleido
    html_path = out_base.with_suffix(".html")
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    if can_export_png():
        png_path = out_base.with_suffix(".png")
        pio.write_image(fig, str(png_path), width=1200, height=800, scale=2)

def export_window_assets(session: str, start: float, end: float, figs: dict, preview_clip: Path | None) -> Path:
    out_dir = EXPORT_DIR / safe_name(session) / window_label_pathsafe(start, end)
    out_dir.mkdir(parents=True, exist_ok=True)

    export_plot(figs["heatmap"], out_dir / "heatmap")
    export_plot(figs["scanpath"], out_dir / "scanpath")
    export_plot(figs["fixhist"], out_dir / "fix_dur_hist")
    export_plot(figs["sachist"], out_dir / "sac_amp_hist")

    if preview_clip and preview_clip.exists():
        shutil.copyfile(preview_clip, out_dir / "clip.mp4")

    return out_dir

def export_full_session_assets(session: str) -> str:
    fix = load_fix(session)
    sac = load_sac(session)

    out_dir = EXPORT_DIR / safe_name(session) / "FULL"
    out_dir.mkdir(parents=True, exist_ok=True)

    export_plot(make_heatmap(fix), out_dir / "heatmap_full")
    export_plot(make_scanpath(fix), out_dir / "scanpath_full")
    export_plot(make_hist_fix_dur(fix), out_dir / "fix_dur_hist_full")
    export_plot(make_hist_sac_amp(sac), out_dir / "sac_amp_hist_full")

    src_video = find_video_for_session(session)
    if src_video is None:
        return f"{session}: full charts exported; no source video found in videos/."

    if not ffmpeg_available():
        return f"{session}: full charts exported; ffmpeg not available for full video export."

    out_mp4 = out_dir / f"{safe_name(session)}_full.mp4"
    if not out_mp4.exists():
        export_full_video_to_mp4(src_video, out_mp4)

    return f"{session}: full charts exported; full video exported."

# -----------------------------
# Guardrails
# -----------------------------
if not FIX_DIR.exists():
    st.error("Missing folder: fixations/")
    st.stop()
if not SAC_DIR.exists():
    st.error("Missing folder: saccades/")
    st.stop()

sessions = list_sessions()
if not sessions:
    st.error("No sessions found. Expected files like fixations/<name>_fixations.csv")
    st.stop()

if "idx" not in st.session_state:
    st.session_state.idx = 0

# -----------------------------
# Sidebar controls + buttons
# -----------------------------
with st.sidebar:
    st.header("Selection")
    session = st.selectbox("Session", sessions)

    fix = load_fix(session)
    sac = load_sac(session)

    win_df = load_windows_if_exists(session)
    if win_df is None:
        win_df = compute_windows_from_fix(fix)
    if len(win_df) == 0:
        st.warning("No windows available (empty fixation file).")
        st.stop()

    win_df = compute_window_metrics(fix, sac, win_df)
    win_df = win_df.sort_values("window_start").reset_index(drop=True)

    st.divider()
    st.header("Exports")

    export_window_btn = st.button("Export charts + clip (selected window)", use_container_width=True)
    export_all_full_btn = st.button("Export FULL outputs for ALL sessions", use_container_width=True)

# -----------------------------
# Timeline + window selection
# -----------------------------
st.markdown("### Session timeline")

t = win_df["window_start"].astype(float)

timeline_fig = go.Figure()
timeline_fig.add_trace(go.Scatter(
    x=t, y=win_df["fixation_count"],
    mode="lines+markers", name="Fixation count"
))
timeline_fig.add_trace(go.Scatter(
    x=t, y=win_df["mean_fix_duration"],
    mode="lines+markers", name="Mean fixation duration (s)",
    yaxis="y2"
))
timeline_fig.add_trace(go.Scatter(
    x=t, y=win_df["mean_saccade_amp"],
    mode="lines+markers", name="Mean saccade amplitude",
    yaxis="y3"
))

# selection indicator
sel_idx = int(np.clip(st.session_state.idx, 0, len(win_df) - 1))
sel_t = float(win_df.loc[sel_idx, "window_start"])
timeline_fig.add_vline(x=sel_t, line_width=2)

timeline_fig.update_layout(
    height=320,
    margin=dict(l=10, r=10, t=10, b=10),
    legend=dict(orientation="h"),
    xaxis=dict(title="time (s)"),
    yaxis=dict(title="fixations / window"),
    yaxis2=dict(title="mean fix dur (s)", overlaying="y", side="right", anchor="x"),
    yaxis3=dict(title="mean sacc amp", overlaying="y", side="right", anchor="free", position=1.0),
)
timeline_fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="white"),
)
timeline_fig.update_xaxes(gridcolor="rgba(255,255,255,0.12)")
timeline_fig.update_yaxes(gridcolor="rgba(255,255,255,0.12)")

# Click-to-jump if available
# Always render the timeline normally (no click/drag component)
st.plotly_chart(timeline_fig, use_container_width=True)

# Dropdown window selector (this is the ONLY control)
labels = [f"{window_label_human(float(r.window_start), float(r.window_end))}"
          for r in win_df.itertuples(index=False)]

idx = st.selectbox(
    "Select 30-second window",
    options=list(range(len(labels))),
    index=0,
    format_func=lambda i: labels[i],
    key="window_dropdown_main",
)

# Apply selected window
start = float(win_df.loc[idx, "window_start"])
end = float(win_df.loc[idx, "window_end"])
fix_w, sac_w = subset_window(fix, sac, start, end)

# Dropdown window selector (human-readable mm:ss–mm:ss)
labels = [f"{window_label_human(float(r.window_start), float(r.window_end))}"
          for r in win_df.itertuples(index=False)]





# -----------------------------
# Build charts for selected window
# -----------------------------
fig_heat = make_heatmap(fix_w)
fig_scan = make_scanpath(fix_w)
fig_fixhist = make_hist_fix_dur(fix_w)
fig_sachist = make_hist_sac_amp(sac_w)

figs = {
    "heatmap": fig_heat,
    "scanpath": fig_scan,
    "fixhist": fig_fixhist,
    "sachist": fig_sachist,
}

# -----------------------------
# Video preview clip (auto-generate cached) + export actions
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

# Handle export buttons
if export_window_btn:
    out_dir = export_window_assets(session, start, end, figs, preview_clip)
    if can_export_png():
        status.success(f"Exported window charts (HTML + PNG) and clip (if available) to: {out_dir}")
    else:
        status.success(f"Exported window charts (HTML only) and clip (if available) to: {out_dir}. Install `kaleido` for PNG export.")

if export_all_full_btn:
    msgs = []
    for sess in sessions:
        try:
            msgs.append(export_full_session_assets(sess))
        except Exception as e:
            msgs.append(f"{sess}: export failed ({e})")
    status.success("Batch export complete:\n" + "\n".join(msgs))

# -----------------------------
# 2x2 chart layout (as requested)
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
# Video at bottom
# -----------------------------
st.markdown("### Video (selected window)")

if src_video is None:
    st.info(f"No matching video found for session '{session}'. Put the recording in `videos/` as `{session}.mp4` (or .mkv/.mov/.avi).")
else:
    if not ffmpeg_available():
        st.error("FFmpeg not found in this environment, so clips cannot be generated.")
    elif preview_clip and preview_clip.exists():
        # Bytes-based rendering tends to be more reliable on Windows
        with open(preview_clip, "rb") as f:
            st.video(f.read())
    else:
        st.caption("No preview clip available for this window.")

with st.expander("Show window data tables"):
    st.markdown("#### Fixations in selected window")
    st.dataframe(fix_w, use_container_width=True)
    st.markdown("#### Saccades in selected window")
    st.dataframe(sac_w, use_container_width=True)