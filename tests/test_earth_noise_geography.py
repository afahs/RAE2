from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "investigate_earth_noise_geography.py"
SPEC = importlib.util.spec_from_file_location("investigate_earth_noise_geography", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_land_mask_honours_polygon_and_hole() -> None:
    geojson = {
        "features": [
            {
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[-2, -2], [2, -2], [2, 2], [-2, 2], [-2, -2]],
                        [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5]],
                    ],
                }
            }
        ]
    }
    result = MODULE.land_mask_from_geojson(
        np.asarray([1.0, 0.0, 3.0]), np.asarray([0.0, 0.0, 0.0]), geojson
    )
    assert result.tolist() == [True, False, False]


def test_bh_qvalues_are_monotone_in_sorted_p() -> None:
    p = pd.Series([0.04, 0.001, 0.02, np.nan])
    q = MODULE._bh_qvalues(p)
    finite = pd.DataFrame({"p": p, "q": q}).dropna().sort_values("p")
    assert np.all(np.diff(finite["q"]) >= -1e-12)
    assert q.iloc[1] == 0.003
    assert np.isnan(q.iloc[3])


def test_partial_rank_correlation_removes_a_shared_monotonic_control() -> None:
    rng = np.random.default_rng(14)
    control = np.linspace(-3.0, 3.0, 500)
    x = control + rng.normal(0.0, 0.25, len(control))
    y = control + rng.normal(0.0, 0.25, len(control))
    marginal = MODULE._rank_correlation(x, y)
    partial = MODULE._partial_rank_correlation(x, y, control)
    assert marginal > 0.8
    assert abs(partial) < 0.15
