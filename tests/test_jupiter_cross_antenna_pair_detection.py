from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_jupiter_cross_antenna_pair_detection import (
    build_cross_antenna_pairs,
    daily_pair_denominators,
    daily_pair_rate_points,
)


def test_build_cross_antenna_pairs_uses_nearest_unused_lower_event() -> None:
    events = pd.DataFrame(
        {
            "event_id": [1, 2, 3, 4],
            "burst_peak_time": pd.to_datetime(
                [
                    "1975-01-01 00:00:00",
                    "1975-01-01 00:00:20",
                    "1975-01-01 00:00:05",
                    "1975-01-01 00:02:00",
                ]
            ),
            "date": pd.Timestamp("1975-01-01"),
            "antenna": ["rv1_coarse", "rv1_coarse", "rv2_coarse", "rv2_coarse"],
            "frequency_band": [1, 1, 1, 1],
            "frequency_mhz": [0.45, 0.45, 0.45, 0.45],
            "peak_local_z": [5.0, 6.0, 4.0, 7.0],
            "io_phase_spice_deg": [90.0, 91.0, 90.5, 20.0],
            "jupiter_cml_spice_deg": [100.0, 101.0, 100.5, 50.0],
            "maser_zarka_io_score": [0.9, 0.8, 0.95, 0.1],
            "maser_top10_at_peak": [True, False, True, True],
        }
    )
    pairs = build_cross_antenna_pairs(events, tolerance_s=30.0)
    assert len(pairs) == 1
    row = pairs.iloc[0]
    assert row["upper_event_id"] == 1
    assert row["lower_event_id"] == 3
    assert row["maser_top10_pair"] is True or row["maser_top10_pair"] == np.bool_(True)
    assert row["min_peak_local_z"] == 4.0


def test_daily_pair_rates_use_smaller_upper_lower_denominator() -> None:
    daily = pd.DataFrame(
        {
            "event_set": ["all_bursts"] * 2,
            "selector": ["test_selector", "test_selector"],
            "date": pd.to_datetime(["1975-01-01", "1975-01-01"]),
            "antenna": ["rv1_coarse", "rv2_coarse"],
            "frequency_band": [1, 1],
            "frequency_mhz": [0.45, 0.45],
            "selected_n_samples": [10, 8],
            "control_n_samples": [30, 20],
        }
    )
    denominators = daily_pair_denominators(daily, min_selected_samples=1, min_control_samples=1)
    assert denominators.iloc[0]["selected_pair_samples"] == 8
    assert denominators.iloc[0]["control_pair_samples"] == 20

    pairs = pd.DataFrame(
        {
            "date": pd.to_datetime(["1975-01-01", "1975-01-01"]),
            "frequency_band": [1, 1],
            "frequency_mhz": [0.45, 0.45],
            "test_selector_pair": [True, False],
        }
    )
    rates = daily_pair_rate_points(pairs, denominators, "test_selector")
    row = rates.iloc[0]
    assert row["selected_n_pairs"] == 1
    assert row["control_n_pairs"] == 1
    assert row["selected_pair_rate_per_1000_samples"] == 125.0
    assert row["control_pair_rate_per_1000_samples"] == 50.0
