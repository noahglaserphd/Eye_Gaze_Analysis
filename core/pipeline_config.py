"""
Central defaults for time windows, heatmap resolution, and PNG heatmap blur.

Streamlit uses a plain histogram2d (no Gaussian blur) for responsiveness; offline
matplotlib figures apply a light Gaussian smooth — see viz_utils.duration_weighted_fixation_heatmap.
"""

from __future__ import annotations

# Temporal windows (seconds), used by app.py and window_analysis.py
WINDOW_SIZE = 30.0

# Heatmap grid resolution (bins per axis on 0..1 normalized coordinates)
HEATMAP_BINS_STREAMLIT = 120
HEATMAP_BINS_SESSION_PNG = 140
HEATMAP_BINS_AGGREGATE_PNG = 180

# Gaussian blur after histogram2d for exported/session PNG heatmaps only
GAUSSIAN_HEATMAP_SESSION_RADIUS = 6
GAUSSIAN_HEATMAP_SESSION_SIGMA = 2.2
GAUSSIAN_HEATMAP_AGGREGATE_RADIUS = 7
GAUSSIAN_HEATMAP_AGGREGATE_SIGMA = 2.6
