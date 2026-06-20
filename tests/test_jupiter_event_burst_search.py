from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_jupiter_event_burst_search import (
    annotate_event_coincidences,
    build_event_catalog,
    cluster_high_samples,
    daily_event_rate_points,
)


def test_cluster_high_samples_merges_short_high_z_gaps() -> None:
    group = pd.DataFrame(
        {
            "time": pd.to_datetime(
                [
                    "1975-01-01 00:00:00",
                    "1975-01-01 00:01:00",
                    "1975-01-01 00:03:00",
                    "1975-01-01 00:10:00",
                    "1975-01-01 00:11:00",
                ]
            ),
            "local_z_log_power": [0.0, 3.2, 3.4, 3.3, 0.1],
        }
    )
    assert cluster_high_samples(group, 3.0, 180.0, 2, 30.0) == [(1, 2)]


def test_annotate_event_coincidences_uses_ns_tolerance_and_other_channel() -> None:
    times = np.array(
        [
            "1975-01-01T00:00:00.000000",
            "1975-01-01T00:00:00.000000",
            "1975-01-01T01:00:00.000000",
            "1975-01-01T01:00:10.000000",
            "1975-01-01T02:00:00.000000",
            "1975-01-01T02:00:40.000000",
        ],
        dtype="datetime64[us]",
    )
    events = pd.DataFrame(
        {
            "event_id": range(6),
            "burst_peak_time": times,
            "antenna": ["rv1_coarse", "rv2_coarse", "rv1_coarse", "rv1_coarse", "rv1_coarse", "rv2_coarse"],
            "antenna_label": ["upper V", "lower V", "upper V", "upper V", "upper V", "lower V"],
            "frequency_mhz": [0.45, 0.45, 3.93, 3.93, 0.45, 0.45],
        }
    )
    out = annotate_event_coincidences(events, tolerance_s=30.0).sort_values("event_id")
    assert out["has_any_coincidence"].tolist() == [True, True, False, False, False, False]
    assert out["has_cross_antenna_same_frequency"].tolist() == [True, True, False, False, False, False]
    assert out["has_cross_frequency_same_antenna"].sum() == 0


def test_build_event_catalog_and_daily_event_rates_pair_controls() -> None:
    times = pd.date_range("1975-01-01 00:00", periods=8, freq="1min")
    samples = pd.DataFrame(
        {
            "time": times,
            "date": times.floor("D"),
            "antenna": "rv1_coarse",
            "frequency_band": 1,
            "frequency_mhz": 0.45,
            "power": np.arange(8, dtype=float) + 10.0,
            "log_power": np.log(np.arange(8, dtype=float) + 10.0),
            "local_z_log_power": [3.2, 3.3, 0.0, 0.0, 3.5, 3.6, 0.0, 0.0],
            "jupiter_visible_by_moon": True,
            "sample_id": np.arange(8),
            "jupiter_cml_spice_deg": 100.0,
            "io_phase_spice_deg": 90.0,
            "io_phase_spice_reverse_deg": 270.0,
            "maser_zarka_io_score": 0.9,
            "maser_zarka_full_score": 0.8,
            "earth_visible_by_moon": True,
            "jupiter_limb_angle_deg": 50.0,
            "earth_limb_angle_deg": 60.0,
            "test_selector": [True, True, False, False, False, False, False, False],
        }
    )
    events = build_event_catalog(samples, ["test_selector"], 3.0, 120.0, 2, 30.0)
    assert len(events) == 2
    daily = daily_event_rate_points(
        samples,
        events,
        pd.Series(samples["test_selector"].to_numpy(dtype=bool), index=samples.index),
        selector_name="test_selector",
        min_selected_samples=2,
        min_control_samples=2,
        high_z=3.0,
    )
    assert len(daily) == 1
    row = daily.iloc[0]
    assert row["selected_n_events"] == 1
    assert row["control_n_events"] == 1
    assert row["selected_event_rate_per_1000_samples"] == 500.0
