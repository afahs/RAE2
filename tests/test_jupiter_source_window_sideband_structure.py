from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_jupiter_source_window_sideband_structure import (
    build_window_phase_bin_values,
    interval_stats,
    source_sideband_metrics_for_role,
)


def sample_frame() -> pd.DataFrame:
    times = pd.date_range("1975-01-01 00:00", periods=9, freq="10min")
    daily = np.array([-0.2, -0.1, 0.4, 0.5, 0.6, -0.05, -0.1, -0.2, -0.1])
    return pd.DataFrame(
        {
            "time": times,
            "antenna": ["rv1_coarse"] * len(times),
            "frequency_band": [6] * len(times),
            "frequency_mhz": [3.93] * len(times),
            "daily_log10_residual": daily,
            "local20min_log10_residual": daily - np.median(daily),
        }
    )


def window_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "window_id": ["IoA_0001"],
            "source_box": ["Io-A"],
            "shift_days": [0.0],
            "start_time": pd.to_datetime(["1975-01-01 00:20"]),
            "end_time": pd.to_datetime(["1975-01-01 00:40"]),
            "duration_min": [20.0],
            "median_cml_deg": [230.0],
            "median_io_phase_deg": [230.0],
            "max_maser_zarka_io_score": [1.0],
        }
    )


def test_interval_stats_counts_high_fraction_and_summary_values() -> None:
    stats = interval_stats(np.array([0.0, 0.2, 0.4]), high_threshold=0.3)
    assert stats["n"] == 3
    assert stats["n_high"] == 1
    assert np.isclose(stats["high_fraction"], 1 / 3)
    assert np.isclose(stats["median"], 0.2)


def test_source_sideband_metrics_compare_source_to_pre_post_sidebands() -> None:
    metrics = source_sideband_metrics_for_role(
        sample_frame(),
        window_frame(),
        high_threshold=0.3,
        sideband_gap_min=0.0,
        sideband_duration_scale=1.0,
        role="real",
    )
    row = metrics.iloc[0]
    assert row["source_n_samples"] == 3
    assert row["sideband_n_samples"] == 4
    assert np.isclose(row["source_median_daily_residual"], 0.5)
    assert np.isclose(row["sideband_median_daily_residual"], -0.1)
    assert np.isclose(row["source_minus_sideband_median_daily_residual"], 0.6)


def test_window_phase_bin_values_identifies_inside_source_window_bins() -> None:
    values = build_window_phase_bin_values(
        sample_frame(),
        window_frame(),
        high_threshold=0.3,
        phase_bin_width=0.5,
        sideband_scale=1.0,
        role="real",
    )
    inside = values[values["is_inside_source_window"]]
    outside = values[~values["is_inside_source_window"]]
    assert set(inside["phase_bin"].round(2)) == {0.25, 0.75}
    assert outside["phase_bin"].min() < 0.0
    assert inside["any_high"].all()
