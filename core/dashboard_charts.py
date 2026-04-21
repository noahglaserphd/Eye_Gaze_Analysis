from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def make_heatmap(fix_df: pd.DataFrame, bins: int):
    if len(fix_df) == 0:
        z = np.zeros((bins, bins))
    else:
        x = fix_df["x_norm"].to_numpy(float)
        y = fix_df["y_norm"].to_numpy(float)
        w = fix_df["duration_s"].to_numpy(float)
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
        x, y, w = x[valid], y[valid], w[valid]
        h2, _, _ = np.histogram2d(x, y, bins=bins, range=[[0, 1], [0, 1]], weights=w)
        z = h2.T
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
