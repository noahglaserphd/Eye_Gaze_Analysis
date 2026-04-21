"""
Shared visualization helpers: duration-weighted fixation heatmaps (matplotlib path),
Gaussian blur, and safe reductions for empty series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def gaussian_kernel(radius: int, sigma: float) -> np.ndarray:
    ax = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    k /= k.sum()
    return k


def convolve2d_same(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    # Fast path without extra dependencies: vectorized sliding-window contraction.
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kh, kw))
    return np.einsum("ijkl,kl->ij", windows, kernel, optimize=True)


def duration_weighted_fixation_heatmap(
    fix: pd.DataFrame,
    *,
    bins: int,
    gaussian_radius: int,
    gaussian_sigma: float,
) -> np.ndarray:
    """2D histogram of fixation positions weighted by duration, then Gaussian-smoothed."""
    required = ("x_norm", "y_norm", "duration_s")
    missing = [c for c in required if c not in fix.columns]
    if missing:
        raise ValueError(f"Fixation DataFrame missing columns: {missing}")

    if len(fix) == 0:
        return np.zeros((bins, bins), dtype=float)

    x = fix["x_norm"].to_numpy(dtype=float)
    y = fix["y_norm"].to_numpy(dtype=float)
    w = fix["duration_s"].to_numpy(dtype=float)

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
    x, y, w = x[valid], y[valid], w[valid]

    if len(x) == 0:
        return np.zeros((bins, bins), dtype=float)

    h, _, _ = np.histogram2d(x, y, bins=bins, range=[[0, 1], [0, 1]], weights=w)
    h = h.T
    k = gaussian_kernel(gaussian_radius, gaussian_sigma)
    return convolve2d_same(h, k)


def safe_mean(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else float("nan")


def safe_median(series: pd.Series) -> float:
    return float(series.median()) if len(series) else float("nan")


def safe_max(series: pd.Series) -> float:
    return float(series.max()) if len(series) else float("nan")
