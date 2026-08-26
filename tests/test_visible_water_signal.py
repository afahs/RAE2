from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "plot_pass_signal_vs_visible_water.py"
SPEC = importlib.util.spec_from_file_location("plot_pass_signal_vs_visible_water", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_visible_water_metrics_all_water_and_all_land() -> None:
    xyz = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )
    base = {"xyz": xyz, "area_weight": np.ones(4)}
    observer = np.asarray([[1.0, 0.0, 0.0]])
    distance = np.asarray([384_000.0])
    water = MODULE.visible_water_metrics(
        observer,
        distance,
        {**base, "is_land": np.zeros(4, dtype=bool)},
    )
    land = MODULE.visible_water_metrics(
        observer,
        distance,
        {**base, "is_land": np.ones(4, dtype=bool)},
    )
    assert water.loc[0, "visible_water_percent_projected"] == 100.0
    assert land.loc[0, "visible_water_percent_projected"] == 0.0


def test_visible_water_metrics_uses_projected_area() -> None:
    grid = {
        "xyz": np.asarray([[1.0, 0.0, 0.0], [0.5, np.sqrt(0.75), 0.0]]),
        "area_weight": np.ones(2),
        "is_land": np.asarray([True, False]),
    }
    result = MODULE.visible_water_metrics(
        np.asarray([[1.0, 0.0, 0.0]]),
        np.asarray([1.0e9]),
        grid,
        earth_radius_km=1.0,
    )
    # In the distant-observer limit the water cell has half the projected
    # area of the land cell: 0.5 / (1 + 0.5) = 1/3.
    assert np.isclose(
        result.loc[0, "visible_water_fraction_projected"],
        1.0 / 3.0,
        atol=1e-8,
    )


def test_build_signal_points_arithmetic_average_across_antennas() -> None:
    power = pd.DataFrame(
        {
            "pass_id": [1, 1],
            "frequency_band": [9, 9],
            "frequency_mhz": [9.18, 9.18],
            "antenna": ["rv1_coarse", "rv2_coarse"],
            "window_minutes": [5.0, 5.0],
            "robust_inlier": [True, True],
            "minimum_alignment_angle_deg": [2.0, 2.0],
            "mean_raw_power": [10.0, 30.0],
            "mean_log10_power": [1.0, np.log10(30.0)],
            "month_centered_mean_log10_power": [0.1, 0.3],
            "n_samples": [6, 7],
        }
    )
    metrics = pd.DataFrame(
        {
            "pass_id": [1],
            "center_time": pd.to_datetime(["1974-01-01"]),
            "month": ["1974-01"],
            "minimum_alignment_angle_deg": [2.0],
            "subspacecraft_lon_deg": [10.0],
            "subspacecraft_lat_deg": [20.0],
            "subspacecraft_is_land": [False],
            "surface_class": ["ocean"],
            "visible_water_fraction_projected": [0.7],
            "visible_land_fraction_projected": [0.3],
            "visible_water_percent_projected": [70.0],
            "visible_water_fraction_surface": [0.72],
            "visible_water_percent_surface": [72.0],
            "earth_to_rae2_distance_km": [384_000.0],
        }
    )
    result = MODULE.build_signal_points(power, metrics, 5.0, 10.0)
    assert len(result) == 1
    assert result.loc[0, "average_signal"] == 20.0
    assert np.isclose(result.loc[0, "log10_average_signal"], np.log10(20.0))
    assert result.loc[0, "n_antennas"] == 2
