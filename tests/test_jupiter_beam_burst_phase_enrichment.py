from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_jupiter_beam_burst_phase_enrichment import (
    annotate_cross_frequency_groups,
    build_burst_catalog,
    cluster_high_samples,
    enrichment_summary,
    in_phase_windows,
    phase_shift_control_summary,
    read_selected_samples,
    shifted_windows,
    threshold_table,
)


def test_in_phase_windows_uses_expected_io_bands() -> None:
    phases = np.array([79.9, 80.0, 110.0, 130.0, 220.0, 260.0])
    assert in_phase_windows(phases).tolist() == [False, True, True, False, True, False]


def test_shifted_windows_rotate_expected_bands() -> None:
    windows = shifted_windows(10.0)
    assert windows[0] == ("Io-B/D", 90.0, 140.0)
    assert in_phase_windows(np.array([85.0]), windows=windows).tolist() == [False]
    assert in_phase_windows(np.array([95.0]), windows=windows).tolist() == [True]


def test_cluster_high_samples_merges_nearby_high_samples() -> None:
    group = pd.DataFrame(
        {
            "time": pd.to_datetime(
                [
                    "1975-01-01 00:00:00",
                    "1975-01-01 00:01:00",
                    "1975-01-01 00:02:00",
                    "1975-01-01 00:10:00",
                    "1975-01-01 00:11:00",
                ]
            ),
            "local_normalized_power": [0.0, 6.0, 7.0, 8.0, 0.0],
        }
    )
    assert cluster_high_samples(group, threshold=5.0, max_gap_seconds=180.0, min_cluster_samples=2, max_event_minutes=30.0) == [
        (1, 2)
    ]


def test_build_burst_catalog_is_blind_to_io_phase_then_flags_window() -> None:
    times = pd.date_range("1975-01-01", periods=5, freq="1min")
    samples = pd.DataFrame(
        {
            "time": times,
            "frequency_band": 1,
            "frequency_mhz": 0.45,
            "power": [1, 2, 3, 4, 5],
            "local_normalized_power": [0.0, 6.0, 7.0, 0.0, 0.0],
            "io_phase_spice_deg": [10.0, 90.0, 95.0, 180.0, 220.0],
            "jupiter_cml_spice_deg": 100.0,
            "jupiter_beam_relative_gain_db": -3.0,
            "jupiter_beam_separation_deg": 30.0,
        }
    )
    thresholds = threshold_table(samples, quantile=0.5, min_local_z=5.0)
    bursts = build_burst_catalog(samples, thresholds, max_gap_seconds=180.0, min_cluster_samples=2, max_event_minutes=30.0)
    assert len(bursts) == 1
    assert bursts.iloc[0]["in_expected_io_window"]
    assert bursts.iloc[0]["io_phase_window"] == "Io-B/D"


def test_annotate_cross_frequency_groups_counts_distinct_frequency_bands() -> None:
    bursts = pd.DataFrame(
        {
            "burst_id": [0, 1, 2],
            "burst_peak_time": pd.to_datetime(
                ["1975-01-01 00:00:00", "1975-01-01 00:03:00", "1975-01-01 01:00:00"]
            ),
            "frequency_band": [1, 2, 1],
            "frequency_mhz": [0.45, 0.70, 0.45],
            "peak_local_normalized_power": [8.0, 7.0, 9.0],
            "io_phase_spice_deg": [90.0, 91.0, 10.0],
            "io_phase_window": ["Io-B/D", "Io-B/D", "outside"],
            "in_expected_io_window": [True, True, False],
        }
    )
    annotated, groups = annotate_cross_frequency_groups(bursts, tolerance_seconds=300.0)
    assert groups.loc[0, "n_group_frequencies"] == 2
    assert groups.loc[1, "n_group_frequencies"] == 1
    assert annotated.loc[0, "n_group_frequencies"] == 2
    assert annotated.loc[2, "n_group_frequencies"] == 1


def test_enrichment_summary_compares_bursts_to_exposure_fraction() -> None:
    samples = pd.DataFrame(
        {
            "frequency_band": [1] * 10,
            "frequency_mhz": [0.45] * 10,
            "io_phase_spice_deg": [90.0, 95.0, 100.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0],
        }
    )
    bursts = pd.DataFrame(
        {
            "frequency_band": [1, 1],
            "frequency_mhz": [0.45, 0.45],
            "io_phase_spice_deg": [90.0, 100.0],
        }
    )
    out = enrichment_summary(samples, bursts)
    row = out[out["group"].eq("frequency")].iloc[0]
    assert row["n_expected_io_exposure_samples"] == 3
    assert row["n_expected_io_bursts"] == 2
    assert row["expected_io_burst_fraction"] == 1.0
    assert row["expected_io_exposure_fraction"] == 0.3


def test_phase_shift_control_marks_zero_shift() -> None:
    samples = pd.DataFrame({"io_phase_spice_deg": [90.0, 10.0, 20.0, 220.0]})
    bursts = pd.DataFrame({"io_phase_spice_deg": [90.0, 220.0]})
    out = phase_shift_control_summary(samples, bursts, shift_step_deg=180.0)
    assert out["shift_deg"].tolist() == [0.0, 180.0]
    assert out["is_nominal_expected_window"].tolist() == [True, False]


def test_read_selected_samples_accepts_npy_recarray(tmp_path) -> None:
    dtype = [
        ("time", "O"),
        ("frequency_band", "i8"),
        ("frequency_mhz", "f8"),
        ("power", "f8"),
        ("local_normalized_power", "f8"),
        ("io_phase_spice_deg", "f8"),
        ("jupiter_cml_spice_deg", "f8"),
        ("jupiter_beam_relative_gain_db", "f8"),
        ("jupiter_beam_separation_deg", "f8"),
    ]
    arr = np.array(
        [("1975-01-01 00:00:00", 1, 0.45, 10.0, 6.0, 90.0, 100.0, -3.0, 40.0)],
        dtype=dtype,
    )
    path = tmp_path / "selected.npy"
    np.save(path, arr)
    out = read_selected_samples(path)
    assert len(out) == 1
    assert out.iloc[0]["frequency_band"] == 1
    assert out.iloc[0]["io_phase_spice_deg"] == 90.0
