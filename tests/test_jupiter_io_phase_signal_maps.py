from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.plot_jupiter_io_phase_signal_maps import (
    expected_io_bands,
    phase_in_windows,
    shifted_geometry,
    summarize_phase_frequency,
)


def test_expected_io_bands_cover_standard_source_phase_ranges() -> None:
    assert expected_io_bands() == [("Io-B/D", 80.0, 130.0), ("Io-A/C", 205.0, 260.0)]


def test_expected_io_bands_can_select_centered_windows() -> None:
    assert expected_io_bands(mode="centered", center_half_width_deg=20.0) == [
        ("Io-B/D", 85.0, 125.0),
        ("Io-A/C", 212.5, 252.5),
    ]


def test_phase_in_windows_marks_expected_ranges() -> None:
    vals = np.array([20.0, 90.0, 128.0, 170.0, 230.0, 300.0])
    mask = phase_in_windows(vals, [(80.0, 130.0), (205.0, 260.0)])
    assert mask.tolist() == [False, True, True, False, True, False]


def test_shifted_geometry_moves_geometry_time_backward_for_positive_shift() -> None:
    geom = pd.DataFrame({"time": pd.to_datetime(["1975-01-02 00:00"]), "io_phase_spice_deg": [90.0]})
    out = shifted_geometry(geom, shift_days=1.0)
    assert out["time"].iloc[0] == pd.Timestamp("1975-01-01 00:00")


def test_summarize_phase_frequency_uses_direct_high_factor_threshold() -> None:
    times = pd.to_datetime(["1975-01-01 00:00", "1975-01-01 00:01", "1975-01-01 00:02"])
    samples = pd.DataFrame(
        {
            "time": times,
            "antenna": ["rv1_coarse"] * 3,
            "frequency_band": [6] * 3,
            "frequency_mhz": [3.93] * 3,
            "log10_power": [5.0, 5.31, 5.6],
            "daily_log10_residual": [0.0, 0.31, 0.6],
            "jupiter_visible_by_moon": [True, True, True],
            "io_phase_spice_deg": [85.0, 86.0, 87.0],
        }
    )
    summary = summarize_phase_frequency(samples, phase_bin_deg=10.0, high_factor=2.0, min_count=1)
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["io_phase_bin_deg"] == 85.0
    assert row["factor_high_fraction"] == 2 / 3
