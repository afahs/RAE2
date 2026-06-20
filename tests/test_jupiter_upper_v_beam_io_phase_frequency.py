from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.plot_jupiter_upper_v_beam_io_phase_frequency import (
    attach_local_power_normalization,
    beam_half_width_deg_for_frequency,
    compute_beam_separation_deg,
    detect_ra_units,
    interpolate_cyclic,
    nearest_beam_spec,
    relative_gain_db_from_linear,
)


def test_beam_half_width_rule_matches_requested_frequency_breaks() -> None:
    assert beam_half_width_deg_for_frequency(0.45) == 90.0
    assert beam_half_width_deg_for_frequency(0.90) == 90.0
    assert beam_half_width_deg_for_frequency(1.31) == 20.0
    assert beam_half_width_deg_for_frequency(4.70) == 20.0
    assert beam_half_width_deg_for_frequency(6.55) == 10.0
    assert beam_half_width_deg_for_frequency(9.18) == 10.0


def test_compute_beam_separation_deg_uses_angular_distance() -> None:
    beam = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    source = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    sep = compute_beam_separation_deg(beam, source)
    assert np.allclose(sep, [0.0, 90.0])


def test_detect_ra_units_auto_prefers_hours_for_legacy_ra_range() -> None:
    assert detect_ra_units([0.0, 12.0, 23.5], "auto") == "hours"
    assert detect_ra_units([0.0, 120.0, 350.0], "auto") == "degrees"


def test_nearest_beam_spec_uses_available_digitized_frequency() -> None:
    assert nearest_beam_spec(0.45)[0] == 1.31
    assert nearest_beam_spec(2.20)[0] == 1.31
    assert nearest_beam_spec(3.93)[0] == 3.93
    assert nearest_beam_spec(9.18)[0] == 6.55


def test_relative_gain_db_from_linear_normalizes_to_peak() -> None:
    gain_db = relative_gain_db_from_linear(np.array([10.0, 1.0, 0.1, 0.0]))
    assert np.allclose(gain_db[:3], [0.0, -10.0, -20.0])
    assert np.isnan(gain_db[3])


def test_interpolate_cyclic_wraps_at_360_deg() -> None:
    angles = np.array([0.0, 90.0, 180.0, 270.0])
    values = np.array([0.0, 9.0, 18.0, 27.0])
    assert np.allclose(interpolate_cyclic(angles, values, np.array([45.0, 315.0, 405.0])), [4.5, 13.5, 4.5])


def test_attach_local_power_normalization_centers_each_frequency_window() -> None:
    samples = pd.DataFrame(
        {
            "time": pd.date_range("1975-01-01", periods=5, freq="1min"),
            "frequency_band": [1, 1, 1, 1, 1],
            "power": [1.0, 1.0, 11.0, 1.0, 1.0],
        }
    )
    out = attach_local_power_normalization(samples, window_s=300.0, min_periods=3)
    assert np.isclose(out.loc[2, "local_normalization_center_power"], 1.0)
    assert out.loc[2, "local_normalization_scale_power"] > 0.0
    assert out.loc[2, "local_normalized_power"] > 2.0
