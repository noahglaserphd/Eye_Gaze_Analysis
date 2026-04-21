from __future__ import annotations

import pandas as pd

from dashboard_aoi import assign_aoi_rects, normalize_aois


def test_normalize_aois_clamps_and_orders() -> None:
    aois = normalize_aois(
        [
            {"name": "HUD", "x0": 0.8, "y0": -0.2, "x1": 0.2, "y1": 1.2},
            {"name": "bad"},
        ]
    )
    assert len(aois) == 1
    a = aois[0]
    assert a["x0"] == 0.2
    assert a["x1"] == 0.8
    assert a["y0"] == 0.0
    assert a["y1"] == 1.0


def test_assign_aoi_rects_labels_inside_points() -> None:
    fix = pd.DataFrame(
        [
            {"x_norm": 0.1, "y_norm": 0.1, "duration_s": 0.2},
            {"x_norm": 0.9, "y_norm": 0.9, "duration_s": 0.2},
        ]
    )
    aois = [{"name": "TopLeft", "x0": 0.0, "y0": 0.0, "x1": 0.5, "y1": 0.5}]
    out = assign_aoi_rects(fix, aois)
    assert out.loc[0, "aoi"] == "TopLeft"
    assert out.loc[1, "aoi"] == "None"
