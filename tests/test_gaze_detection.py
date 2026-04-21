from __future__ import annotations

import pandas as pd
import pytest

from gaze_detection import detect_fixations_idt, saccades_from_fixations


def test_detect_fixations_idt_requires_columns() -> None:
    df = pd.DataFrame({"timestamp": [0.0, 0.1], "gaze_x_norm": [0.1, 0.1]})
    with pytest.raises(ValueError):
        detect_fixations_idt(df)


def test_detect_fixations_idt_finds_single_fixation() -> None:
    df = pd.DataFrame(
        {
            "timestamp": [0.00, 0.05, 0.10, 0.15],
            "gaze_x_norm": [0.40, 0.401, 0.399, 0.400],
            "gaze_y_norm": [0.50, 0.501, 0.499, 0.500],
        }
    )
    fix = detect_fixations_idt(df, min_duration_s=0.10, dispersion_thresh=0.02)
    assert len(fix) == 1
    assert fix.loc[0, "duration_s"] == pytest.approx(0.15)
    assert fix.loc[0, "x_norm"] == pytest.approx(0.4, abs=0.01)
    assert fix.loc[0, "y_norm"] == pytest.approx(0.5, abs=0.01)


def test_saccades_from_fixations_produces_amplitude() -> None:
    fix = pd.DataFrame(
        [
            {"start_s": 0.0, "end_s": 0.2, "x_norm": 0.1, "y_norm": 0.1},
            {"start_s": 0.25, "end_s": 0.5, "x_norm": 0.4, "y_norm": 0.5},
        ]
    )
    sac = saccades_from_fixations(fix)
    assert len(sac) == 1
    assert sac.loc[0, "duration_s"] == pytest.approx(0.05)
    assert sac.loc[0, "amplitude_norm"] > 0
