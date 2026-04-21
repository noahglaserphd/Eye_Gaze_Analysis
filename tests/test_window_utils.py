from __future__ import annotations

import pandas as pd
import pytest

from core.window_utils import compute_window_metrics, compute_windows_from_fix, subset_window


def test_compute_windows_from_fix_empty() -> None:
    fix = pd.DataFrame(columns=["end_s"])
    win = compute_windows_from_fix(fix, 30.0)
    assert win.empty
    assert list(win.columns) == ["window_start", "window_end"]


def test_compute_windows_from_fix_generates_expected_ranges() -> None:
    fix = pd.DataFrame({"end_s": [12.0, 61.0, 89.0]})
    win = compute_windows_from_fix(fix, 30.0)
    assert len(win) == 3
    assert win.iloc[0]["window_start"] == pytest.approx(0.0)
    assert win.iloc[2]["window_end"] == pytest.approx(90.0)


def test_compute_window_metrics_and_subset_window() -> None:
    fix = pd.DataFrame(
        [
            {"start_s": 5.0, "end_s": 5.2, "duration_s": 0.2, "x_norm": 0.1, "y_norm": 0.1},
            {"start_s": 35.0, "end_s": 35.5, "duration_s": 0.5, "x_norm": 0.2, "y_norm": 0.2},
        ]
    )
    sac = pd.DataFrame(
        [
            {"start_s": 5.2, "amplitude_norm": 0.3},
            {"start_s": 35.5, "amplitude_norm": 0.4},
        ]
    )
    win = pd.DataFrame({"window_start": [0.0, 30.0], "window_end": [30.0, 60.0]})

    out = compute_window_metrics(fix, sac, win)
    assert len(out) == 2
    assert int(out.iloc[0]["fixation_count"]) == 1
    assert out.iloc[1]["mean_fix_duration"] == pytest.approx(0.5)

    f, s = subset_window(fix, sac, 30.0, 60.0)
    assert len(f) == 1
    assert len(s) == 1
