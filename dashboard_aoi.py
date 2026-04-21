from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def sanitize_aoi_label(name: str, max_len: int = 80) -> str:
    t = str(name)[:max_len]
    return "".join(ch for ch in t if ch.isprintable()).replace("<", "").replace(">", "")


def normalize_aois(raw_aois: list[dict]) -> list[dict]:
    cleaned: list[dict] = []
    for a in raw_aois:
        if not isinstance(a, dict):
            continue
        if not all(k in a for k in ("name", "x0", "y0", "x1", "y1")):
            continue
        try:
            x0, y0 = float(a["x0"]), float(a["y0"])
            x1, y1 = float(a["x1"]), float(a["y1"])
            cleaned.append(
                {
                    "name": sanitize_aoi_label(str(a["name"])),
                    "x0": float(np.clip(min(x0, x1), 0.0, 1.0)),
                    "y0": float(np.clip(min(y0, y1), 0.0, 1.0)),
                    "x1": float(np.clip(max(x0, x1), 0.0, 1.0)),
                    "y1": float(np.clip(max(y0, y1), 0.0, 1.0)),
                }
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
            text=sanitize_aoi_label(str(a.get("name", ""))),
            showarrow=False,
            yanchor="bottom",
            font=dict(size=12),
        )
    return fig
