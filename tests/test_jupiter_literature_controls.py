from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_jupiter_literature_controls import (
    WindowInterval,
    label_permutation_p,
    phase_scramble_summary,
    select_interval_samples,
    spearman_corr,
)
from scripts.run_jupiter_expected_active_selector_analysis import daily_selector_points


def test_spearman_corr_handles_monotonic_values() -> None:
    assert np.isclose(spearman_corr(np.array([1, 2, 3]), np.array([10, 20, 30])), 1.0)
    assert np.isclose(spearman_corr(np.array([1, 2, 3]), np.array([30, 20, 10])), -1.0)


def test_select_interval_samples_uses_selection_bounds_and_reported_start() -> None:
    samples = pd.DataFrame(
        {
            "time": pd.date_range("1975-01-01 00:00", periods=6, freq="10min"),
            "antenna": "rv1_coarse",
            "frequency_band": 4,
            "frequency_mhz": 1.31,
            "daily_z_log_power": np.arange(6, dtype=float),
        }
    )
    interval = WindowInterval(
        label="w001",
        start=pd.Timestamp("1975-01-01 00:10"),
        end=pd.Timestamp("1975-01-01 00:40"),
        report_start=pd.Timestamp("1975-01-01 00:20"),
        report_end=pd.Timestamp("1975-01-01 00:30"),
        parent="w001",
        shift_days=None,
        intensity=2,
        burstiness="SMOOTH",
        reported_freq_range_mhz="10-20",
    )
    selected = select_interval_samples(samples, [interval], "historical_active")
    assert selected["time"].tolist() == samples["time"].iloc[1:5].tolist()
    assert selected["event_start_time"].iloc[0] == pd.Timestamp("1975-01-01 00:20")
    assert selected["dt_from_start_min"].tolist() == [-10.0, 0.0, 10.0, 20.0]


def test_label_permutation_p_returns_valid_probabilities() -> None:
    rows = []
    for wid in ["w001", "w002", "w003", "w004"]:
        rows.append(
            {
                "historical_window_id": wid,
                "selector": "historical_active",
                "antenna": "rv1_coarse",
                "frequency_band": 4,
                "median_daily_z": 2.0,
                "n_samples": 5,
            }
        )
        for shift, value in [(-7, -0.5), (7, 0.0), (14, 0.3)]:
            rows.append(
                {
                    "historical_window_id": wid,
                    "selector": "shifted_control",
                    "antenna": "rv1_coarse",
                    "frequency_band": 4,
                    "median_daily_z": value,
                    "n_samples": 5,
                    "control_shift_days": shift,
                }
            )
    p_two, p_pos = label_permutation_p(
        pd.DataFrame(rows),
        antenna="rv1_coarse",
        frequency_band=4,
        rng=np.random.default_rng(123),
        n_perm=200,
        min_window_samples=5,
    )
    assert 0.0 <= p_two <= 1.0
    assert 0.0 <= p_pos <= 1.0


def test_phase_scramble_summary_emits_channel_row() -> None:
    rows = []
    for cml in [7.5, 22.5, 37.5]:
        for io in [7.5, 22.5, 37.5]:
            rows.append(
                {
                    "antenna": "rv1_coarse",
                    "frequency_band": 4,
                    "frequency_mhz": 1.31,
                    "cml_bin_deg": cml,
                    "io_bin_deg": io,
                    "high_power_fraction": cml + io,
                    "median_zarka_io_score": cml - io,
                    "regime": "jupiter_visible",
                }
            )
    out = phase_scramble_summary(pd.DataFrame(rows), np.random.default_rng(1), n_perm=10)
    assert len(out) == 1
    assert out.iloc[0]["n_phase_bins"] == 9


def test_daily_selector_points_pairs_same_day_channel_controls() -> None:
    times = pd.date_range("1975-01-01 00:00", periods=8, freq="1h")
    samples = pd.DataFrame(
        {
            "time": times,
            "date": times.floor("D"),
            "antenna": "rv1_coarse",
            "frequency_band": 4,
            "frequency_mhz": 1.31,
            "daily_z_log_power": [3.0, 3.2, 0.0, 0.1, 0.0, 0.1, 0.2, 0.3],
            "jupiter_visible_by_moon": True,
            "maser_zarka_io_score": [1.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "io_phase_spice_deg": [90.0, 95.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "jupiter_cml_spice_deg": [100.0] * 8,
        }
    )
    selected_mask = pd.Series([True, True, False, False, False, False, False, False])
    out = daily_selector_points(
        samples,
        selected_mask,
        selector_name="test_selector",
        high_z=2.5,
        min_selected_samples=2,
        min_control_samples=2,
    )
    assert len(out) == 1
    assert out.iloc[0]["selected_n_samples"] == 2
    assert out.iloc[0]["control_n_samples"] == 6
    assert out.iloc[0]["selected_minus_control_high_tail_fraction"] == 1.0
