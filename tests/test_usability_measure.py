from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from core.usability_measure import score_nasa_tlx


def test_score_nasa_tlx_raw_and_weighted(tmp_path: Path) -> None:
    src = tmp_path / "responses.csv"
    out = tmp_path / "scores.csv"
    df = pd.DataFrame(
        [
            {
                "participant_id": "P1",
                "session_id": "S1",
                "mental_demand": 60,
                "physical_demand": 20,
                "temporal_demand": 50,
                "performance": 40,
                "effort": 70,
                "frustration": 30,
                "mental_demand_weight": 3,
                "physical_demand_weight": 1,
                "temporal_demand_weight": 2,
                "performance_weight": 2,
                "effort_weight": 5,
                "frustration_weight": 2,
            }
        ]
    )
    df.to_csv(src, index=False)

    scored = score_nasa_tlx(src, out, strict_weighting=True)
    assert out.is_file()
    assert scored.loc[0, "nasa_tlx_raw"] == pytest.approx((60 + 20 + 50 + 40 + 70 + 30) / 6)
    expected_weighted = (60 * 3 + 20 * 1 + 50 * 2 + 40 * 2 + 70 * 5 + 30 * 2) / 15
    assert scored.loc[0, "nasa_tlx_weighted"] == pytest.approx(expected_weighted)


def test_score_nasa_tlx_missing_required_column_raises(tmp_path: Path) -> None:
    src = tmp_path / "responses_missing.csv"
    out = tmp_path / "scores.csv"
    pd.DataFrame(
        [
            {
                "mental_demand": 50,
                "physical_demand": 50,
                "temporal_demand": 50,
                "performance": 50,
                "effort": 50,
            }
        ]
    ).to_csv(src, index=False)

    with pytest.raises(ValueError, match="Missing required NASA-TLX columns"):
        score_nasa_tlx(src, out)
