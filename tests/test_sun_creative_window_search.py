from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_sun_creative_window_search.py"
SPEC = importlib.util.spec_from_file_location("run_sun_creative_window_search", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_robust_edge_amplitude_uses_source_sign() -> None:
    time = np.arange(-570.0, 571.0, 60.0)
    values = np.where(time < 0.0, 1.5, -0.5) + 0.0002 * time
    amplitude = MODULE.robust_edge_amplitude(time, values, 0.0, 300.0)
    assert amplitude > 1.8


def test_shift_ridge_recovers_common_original_time() -> None:
    shifts = (-1200.0, -600.0, 0.0, 600.0, 1200.0)
    time = np.arange(-2370.0, 2371.0, 60.0)
    true_delta = 120.0
    rows = []
    for shift in shifts:
        original_time = time + shift
        values = np.where(original_time < true_delta, 1.0, -1.0) + 2e-7 * original_time**2
        rows.extend(
            {
                "control_shift_s": shift,
                "t_bin_sec": float(t),
                "value": float(value),
            }
            for t, value in zip(time, values)
        )
    group = pd.DataFrame.from_records(rows)
    ridge = MODULE.compute_shift_ridge_for_group(
        group,
        "value",
        300.0,
        delta_grid_seconds=np.arange(-300.0, 301.0, 60.0),
    )
    best = ridge.loc[ridge["ridge_amplitude"].idxmax()]
    assert best["delta_s"] == true_delta
    assert best["ridge_amplitude"] > 1.5


def test_rstn_parser_handles_tenths_of_minutes() -> None:
    parsed = MODULE.parse_rstn_line("740102    606MANI 05248 05254     9  40    397")
    assert parsed is not None
    assert parsed["frequency_mhz_reported"] == 606.0
    assert parsed["start_time"] == pd.Timestamp("1974-01-02 05:24:48")


def test_rstn_parser_handles_status_prefix_before_date() -> None:
    parsed = MODULE.parse_rstn_line("  FAL750102   2800OTTA 1535  1540     5   26")
    assert parsed is not None
    assert parsed["frequency_mhz_reported"] == 2800.0
    assert parsed["start_time"] == pd.Timestamp("1975-01-02 15:35:00")


def test_persistence_is_within_one_frequency_and_antenna() -> None:
    rows = []
    for window, amplitude in ((240.0, 0.15), (300.0, 0.11), (600.0, 0.10)):
        rows.append(
            {
                "source_name": "sun",
                "frequency_mhz": 0.70,
                "antenna": "rv2_coarse",
                "statistic": "median",
                "half_window_s": window,
                "zero_amplitude": amplitude,
                "zero_shift_mad": 0.01,
                "best_delta_s": 0.0,
                "best_amplitude": amplitude,
                "zero_one_sided_lag_rank": 1.0 / 21.0,
                "positive_delta_fraction": 0.6,
            }
        )
        # A second frequency has large amplitudes but cannot rescue this
        # antenna because persistence is never pooled across frequencies.
        rows.append(
            {
                "source_name": "sun",
                "frequency_mhz": 4.70,
                "antenna": "rv1_coarse",
                "statistic": "median",
                "half_window_s": window,
                "zero_amplitude": 10.0,
                "zero_shift_mad": 0.01,
                "best_delta_s": 600.0,
                "best_amplitude": 11.0,
                "zero_one_sided_lag_rank": 0.5,
                "positive_delta_fraction": 0.8,
            }
        )
    persistence = MODULE.build_channel_persistence(pd.DataFrame(rows))
    passing = persistence[persistence["passes_all_windows"]]
    assert len(passing) == 1
    row = passing.iloc[0]
    assert row["frequency_mhz"] == 0.70
    assert row["antenna"] == "rv2_coarse"
    assert bool(row["foreground_sensitive_band"])
    assert not bool(row["other_antenna_passes_same_channel"])


def test_rstn_timestamp_gate_does_not_require_frequency_overlap() -> None:
    features = pd.DataFrame(
        [
            {
                "analysis_source": "sun",
                "predicted_event_time": "1975-01-02 12:00:00",
                "frequency_mhz": frequency,
                "antenna": "rv2_coarse",
                "control_family": family,
                "control_id": control_id,
                "source_like_log_contrast": contrast,
            }
            for frequency in (0.70, 4.70)
            for family, control_id, contrast in (
                ("real", "real", 0.3),
                ("off_source", "east", -0.1),
            )
        ]
    )
    rstn = pd.DataFrame(
        {
            "start_time": [pd.Timestamp("1975-01-02 12:05:00")],
            "frequency_mhz_reported": [18.0],
        }
    )
    annotated, summary = MODULE.build_rstn_timing_gate_products(features, rstn)
    count_column = "rstn_reported_le_30mhz_15m_count"
    assert np.all(annotated[count_column].to_numpy() == 1)
    selected = summary[
        summary["rstn_scope"].eq("reported_le_30mhz")
        & np.isclose(summary["gate_half_window_h"], 0.25)
    ]
    assert set(selected["frequency_mhz"]) == {0.70, 4.70}
    assert np.all(selected["n_real_selected"].to_numpy() == 1)
    assert not np.any(selected["reported_frequency_matched_to_rae2"].to_numpy())
