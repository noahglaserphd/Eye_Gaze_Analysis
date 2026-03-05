# app.py
# Interactive gaze overlay analytics dashboard (Streamlit)
#
# AOI drawing uses streamlit-drawable-canvas (rect tool) instead of Plotly events.
# Install:
#   pip install streamlit streamlit-drawable-canvas plotly pandas numpy opencv-python pillow
#   conda install -c conda-forge ffmpeg
# Optional:
#   pip install kaleido
#
# Run:
#   streamlit run app.py

from __future__ import annotations

from pathlib import Path
import json
import shutil
import subprocess

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

try:
    import cv2  # type: ignore
    from PIL import Image  # type: ignore

    HAS_FRAME = True
except Exception:
    HAS_FRAME = False

try:
    from streamlit_drawable_canvas import st_canvas  # type: ignore

    HAS_CANVAS = True
except Exception:
    HAS_CANVAS = False

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

AOI_DIR = ROOT / "aoi"
AOI_FILE = AOI_DIR / "aoi_definitions.json"

CLIP_DIR = ROOT / "cache_clips"
EXPORT_DIR = ROOT / "exports"
CLIP_DIR.mkdir(exist_ok=True)
EXPORT_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="Gaze Overlay Analytics", layout="wide")
st.title("Interactive Gaze Overlay Analytics")


# -----------------------------
# Helpers
# -----------------------------
def safe_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s)


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
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False


# -----------------------------
# Loading
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
    return pd.DataFrame({"window_start": starts[:-1].astype(float), "window_end": (starts[:-1] + WINDOW_SIZE).astype(float)})


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
            dict(
                window_start=start,
                window_end=end,
                fixation_count=int(len(f)),
                mean_fix_duration=float(f["duration_s"].mean()) if len(f) else 0.0,
                total_fix_time=float(f["duration_s"].sum()) if len(f) else 0.0,
                mean_saccade_amp=float(s["amplitude_norm"].mean())
                if (len(s) and "amplitude_norm" in s.columns)
                else 0.0,
            )
        )
    return pd.DataFrame(rows)


def subset_window(fix: pd.DataFrame, sac: pd.DataFrame, start: float, end: float):
    f = fix[(fix["start_s"] >= start) & (fix["start_s"] < end)].copy()
    s = sac[(sac["start_s"] >= start) & (sac["start_s"] < end)].copy()
    return f, s


# -----------------------------
# Video discovery / ffmpeg
# -----------------------------
def find_video_for_session(session: str) -> Path | None:
    if not VIDEO_DIR.exists():
        return None
    for ext in SUPPORTED_VIDEO_EXTS:
        p = VIDEO_DIR / f"{session}{ext}"
        if p.exists():
            return p
    for p in VIDEO_DIR.iterdir():
        if p.is_file() and p.suffix.lower() in SUPPORTED_VIDEO_EXTS and p.stem.startswith(session):
            return p
    return None


def make_clip_ffmpeg(src: Path, start_s: float, end_s: float, out_mp4: Path) -> Path:
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
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return out_mp4


def export_full_video_to_mp4(src: Path, out_mp4: Path) -> Path:
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
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
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
    cleaned: list[dict] = []
    for a in aois:
        if not isinstance(a, dict):
            continue
        if not all(k in a for k in ("name", "x0", "y0", "x1", "y1")):
            continue
        try:
            cleaned.append(
                {"name": str(a["name"]), "x0": float(a["x0"]), "y0": float(a["y0"]), "x1": float(a["x1"]), "y1": float(a["y1"])}
            )
        except Exception:
            continue
    return cleaned


def assign_aoi_rects(fix_df: pd.DataFrame, aois: list[dict]) -> pd.DataFrame:
    out = fix_df.copy()
    if len(out) == 0:
        out["aoi"] = pd.Series(dtype="object")
        return out

    out["aoi"] = "None"
    if not aois:
        return out

    x = out["x_norm"].to_numpy(float)
    y = out["y_norm"].to_numpy(float)
    valid = np.isfinite(x) & np.isfinite(y)

    for a in aois:
        inside = valid & (x >= a["x0"]) & (x <= a["x1"]) & (y >= a["y0"]) & (y <= a["y1"])
        out.loc[inside, "aoi"] = a["name"]

    return out


def aoi_summary_table(fix_df_with_aoi: pd.DataFrame) -> pd.DataFrame:
    if len(fix_df_with_aoi) == 0 or "aoi" not in fix_df_with_aoi.columns:
        return pd.DataFrame(columns=["aoi", "fixation_count", "dwell_time_s", "mean_fix_duration_s"])
    g = fix_df_with_aoi.groupby("aoi", dropna=False)
    return (
        g.agg(
            fixation_count=("aoi", "count"),
            dwell_time_s=("duration_s", "sum"),
            mean_fix_duration_s=("duration_s", "mean"),
        )
        .reset_index()
        .sort_values(["dwell_time_s", "fixation_count"], ascending=False)
        .reset_index(drop=True)
    )


def add_aoi_shapes_to_scanpath(fig: go.Figure, aois: list[dict]) -> go.Figure:
    for a in aois:
        fig.add_shape(
            type="rect",
            x0=a["x0"],
            x1=a["x1"],
            y0=a["y0"],
            y1=a["y1"],
            line=dict(width=2),
            fillcolor="rgba(255,255,255,0.06)",
            layer="below",
        )
        fig.add_annotation(
            x=(a["x0"] + a["x1"]) / 2.0,
            y=a["y0"],
            text=a["name"],
            showarrow=False,
            yanchor="bottom",
            font=dict(size=12),
        )
    return fig


def save_aois(aois: list[dict]) -> None:
    AOI_DIR.mkdir(exist_ok=True)
    payload = {"coordinate_space": "norm", "aois": aois}
    AOI_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    st.cache_data.clear()


# -----------------------------
# Frame grab (for AOI canvas)
# -----------------------------
@st.cache_data
def get_reference_frame(video_path: Path, t_s: float = 1.0):
    if not HAS_FRAME:
        return None, None, None
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


def canvas_objects_to_aois(objs: list[dict], W: int, H: int) -> list[dict]:
    out: list[dict] = []
    for o in objs:
        if not isinstance(o, dict):
            continue
        if o.get("type") != "rect":
            continue
        left = float(o.get("left", 0.0))
        top = float(o.get("top", 0.0))
        width = float(o.get("width", 0.0))
        height = float(o.get("height", 0.0))
        x0p = max(0.0, min(float(W), left))
        y0p = max(0.0, min(float(H), top))
        x1p = max(0.0, min(float(W), left + width))
        y1p = max(0.0, min(float(H), top + height))
        if (x1p - x0p) < 2 or (y1p - y0p) < 2:
            continue
        out.append({"name": f"AOI_{len(out)+1}", "x0": x0p / W, "y0": y0p / H, "x1": x1p / W, "y1": y1p / H})
    return out


# -----------------------------
# Charts
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
        H2, _, _ = np.histogram2d(x, y, bins=bins, range=[[0, 1], [0, 1]], weights=w)
        z = H2.T
    fig = px.imshow(z, origin="upper", aspect="auto")
    fig.update_layout(margin=dict(l=10, r=10, t=35, b=10), title="Fixation heatmap (duration-weighted)")
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig


def make_scanpath(fix_df: pd.DataFrame):
    fig = go.Figure()
    fig.update_layout(title="Scanpath", xaxis_title="x (norm)", yaxis_title="y (norm)", margin=dict(l=10, r=10, t=35, b=10))
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(autorange="reversed", range=[1, 0])
    if len(fix_df) == 0:
        return fig
    max_d = float(fix_df["duration_s"].max()) if float(fix_df["duration_s"].max()) > 0 else 1.0
    sizes = 8 + 30 * (fix_df["duration_s"] / max_d)
    fig.add_trace(go.Scatter(x=fix_df["x_norm"], y=fix_df["y_norm"], mode="lines", name="path", line=dict(width=1)))
    fig.add_trace(
        go.Scatter(
            x=fix_df["x_norm"],
            y=fix_df["y_norm"],
            mode="markers",
            name="fixations",
            marker=dict(size=sizes, opacity=0.8),
            hovertemplate="start=%{customdata[0]:.2f}s<br>dur=%{customdata[1]:.3f}s<br>x=%{x:.3f}, y=%{y:.3f}<extra></extra>",
            customdata=np.column_stack([fix_df["start_s"], fix_df["duration_s"]]),
        )
    )
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


# -----------------------------
# Export helpers
# -----------------------------
def can_export_png() -> bool:
    try:
        import kaleido  # noqa: F401

        return True
    except Exception:
        return False


def export_plot(fig, out_base: Path):
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


# -----------------------------
# Sidebar
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
idx = st.selectbox("Select 30-second window", options=list(range(len(labels))), index=0, format_func=lambda i: labels[i], key="window_dropdown_main")

start = float(win_df.loc[idx, "window_start"])
end = float(win_df.loc[idx, "window_end"])
fix_w, sac_w = subset_window(fix, sac, start, end)

aois = load_aois()
fix_w_aoi = assign_aoi_rects(fix_w, aois)

# -----------------------------
# Charts
# -----------------------------
fig_heat = make_heatmap(fix_w_aoi)
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
    out_dir = export_window_assets(session, start, end, figs, preview_clip)
    status.success(f"Exported to: {out_dir}")

if export_all_full_btn:
    msgs = []
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
# AOI summary
# -----------------------------
st.markdown("### AOI summary (selected window)")
if not AOI_FILE.exists():
    st.info("AOIs not enabled yet.")
elif not aois:
    st.warning("AOI file exists but no valid AOIs were found (needs name/x0/y0/x1/y1).")
else:
    st.dataframe(aoi_summary_table(fix_w_aoi), use_container_width=True)

# -----------------------------
# AOI editor (canvas)
# -----------------------------
with st.expander("AOI editor (draw rectangles and save)", expanded=False):
    if not HAS_CANVAS:
        st.error("Install: pip install streamlit-drawable-canvas")
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

            # Size canvas to the actual frame; if it's huge, cap display width for UI
            display_w = min(W, 1100)
            scale = display_w / W
            display_h = int(H * scale)

            # Render scaled background for UI, but keep mapping back to original pixels
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

            # Convert scaled-pixel rects -> original-pixel rects -> norm AOIs
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
                df["name"] = [f"AOI_{i+1}" for i in range(len(df))]

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
        st.error("FFmpeg not found, so clips cannot be generated.")
    elif preview_clip and preview_clip.exists():
        with open(preview_clip, "rb") as f:
            st.video(f.read())
    else:
        st.caption("No preview clip available for this window.")

with st.expander("Show window data tables"):
    st.markdown("#### Fixations in selected window")
    st.dataframe(fix_w_aoi, use_container_width=True)
    st.markdown("#### Saccades in selected window")
    st.dataframe(sac_w, use_container_width=True)
