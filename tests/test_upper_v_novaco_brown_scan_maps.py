from __future__ import annotations

import numpy as np
import pandas as pd

import scripts.build_upper_v_novaco_brown_scan_maps as nb
from scripts.build_upper_v_novaco_brown_scan_maps import (
    REFERENCE_NEW_MOON,
    SYNODIC_MONTH_DAYS,
    _attach_earth_beam_gain,
    _attach_earth_beam_separation,
    _filter_valid_pointing,
    _iterative_group_reference,
    _normalize_destriped_groups,
    _read_selected_rows,
    interpolate_cyclic,
    nearest_beam_spec,
    relative_gain_db_from_linear,
    detect_ra_units,
    effective_start_time,
    iterative_sigma_clip,
    new_moon_distance_days,
    ten_minute_jump_keep,
)


def _write_tiny_beam_dir(tmp_path):
    rows = pd.DataFrame(
        {
            "angle_deg": [0.0, 90.0, 180.0, 270.0],
            "gain_dB": [0.0, -3.0, -10.0, -3.0],
        }
    )
    specs = []
    for mhz, label in [(1.31, "1p31"), (3.93, "3p93"), (6.55, "6p55")]:
        eplane = tmp_path / f"eplane_{label}MHz.csv"
        hplane = tmp_path / f"hplane_{label}MHz.csv"
        rows.to_csv(eplane, index=False)
        rows.to_csv(hplane, index=False)
        specs.append((mhz, eplane, hplane))
    return specs


def test_detect_ra_units_auto_handles_legacy_hours() -> None:
    assert detect_ra_units(pd.Series([0.0, 12.5, 23.99]), "auto") == "hours"
    assert detect_ra_units(pd.Series([*np.linspace(0.0, 23.0, 100), 9999.0]), "auto") == "hours"
    assert detect_ra_units(pd.Series([0.0, 180.0, 359.0]), "auto") == "degrees"


def test_all_dates_overrides_default_start_time() -> None:
    assert effective_start_time("1974-11-01", all_dates=True) is None
    assert effective_start_time("1974-11-01", all_dates=False) == "1974-11-01"


def test_raw_master_reader_expands_selected_antenna_column(tmp_path) -> None:
    raw_path = tmp_path / "raw.csv"
    pd.DataFrame(
        {
            "time": ["1973-07-12 00:00:00", "1973-07-12 00:01:00", "1973-07-12 00:02:00"],
            "frequency_band": [4, 5, 1],
            "right_ascension": [1.0, 2.0, 3.0],
            "declination": [-10.0, -11.0, -12.0],
            "rv1_fine": [10.0, 0.0, 30.0],
            "rv1_coarse": [100.0, 200.0, 300.0],
        }
    ).to_csv(raw_path, index=False)
    selected = _read_selected_rows(raw_path, "rv1_fine", [4, 5], None, None, 0, "auto")
    assert selected["antenna"].tolist() == ["rv1_fine"]
    assert selected["frequency_band"].tolist() == [4]
    assert selected["power"].tolist() == [10.0]
    assert selected["is_valid"].tolist() == [True]


def test_filter_valid_pointing_removes_bad_declination_and_ra() -> None:
    df = pd.DataFrame(
        {
            "right_ascension": [1.0, 2.0, 25.0],
            "declination": [0.0, 95.0, 0.0],
            "power": [1.0, 1.0, 1.0],
        }
    )
    out, units, n_removed = _filter_valid_pointing(df, "hours")
    assert units == "hours"
    assert n_removed == 2
    assert out["right_ascension"].tolist() == [1.0]


def test_new_moon_distance_wraps_around_synodic_month() -> None:
    times = pd.to_datetime(
        [
            REFERENCE_NEW_MOON,
            REFERENCE_NEW_MOON + pd.Timedelta(days=2.0),
            REFERENCE_NEW_MOON + pd.Timedelta(days=SYNODIC_MONTH_DAYS - 1.0),
        ]
    )
    distance = new_moon_distance_days(times)
    assert np.allclose(distance, [0.0, 2.0, 1.0], atol=1e-6)


def test_earth_beam_separation_uses_radec_axis_in_hours() -> None:
    df = pd.DataFrame(
        {
            "right_ascension": [0.0, 6.0, 12.0],
            "declination": [0.0, 0.0, 0.0],
            "position_x": [0.0, 0.0, 0.0],
            "position_y": [0.0, 0.0, 0.0],
            "position_z": [0.0, 0.0, 0.0],
            "earth_unit_vector_x": [1.0, 1.0, 1.0],
            "earth_unit_vector_y": [0.0, 0.0, 0.0],
            "earth_unit_vector_z": [0.0, 0.0, 0.0],
        }
    )
    out, units = _attach_earth_beam_separation(df, "radec", "hours")
    assert units == "hours"
    assert np.allclose(out["earth_beam_separation_deg"], [0.0, 90.0, 180.0], atol=1e-6)


def test_earth_beam_separation_can_use_radial_upper_axis() -> None:
    df = pd.DataFrame(
        {
            "right_ascension": [0.0, 0.0],
            "declination": [0.0, 0.0],
            "position_x": [1.0, -1.0],
            "position_y": [0.0, 0.0],
            "position_z": [0.0, 0.0],
            "earth_unit_vector_x": [1.0, 1.0],
            "earth_unit_vector_y": [0.0, 0.0],
            "earth_unit_vector_z": [0.0, 0.0],
        }
    )
    out, units = _attach_earth_beam_separation(df, "radial-upper", "hours")
    assert units is None
    assert np.allclose(out["earth_beam_separation_deg"], [0.0, 180.0], atol=1e-3)


def test_earth_beam_gain_attaches_relative_gain_and_visibility(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(nb, "BEAM_SPECS", _write_tiny_beam_dir(tmp_path))
    df = pd.DataFrame(
        {
            "time": [pd.Timestamp("1975-01-01"), pd.Timestamp("1975-01-01 00:01:00")],
            "frequency_mhz": [1.31, 3.93],
            "right_ascension": [0.0, 12.0],
            "declination": [0.0, 0.0],
            "position_x": [10_000.0, 10_000.0],
            "position_y": [0.0, 0.0],
            "position_z": [0.0, 0.0],
            "earth_unit_vector_x": [1.0, 1.0],
            "earth_unit_vector_y": [0.0, 0.0],
            "earth_unit_vector_z": [0.0, 0.0],
        }
    )
    out, units = nb._attach_earth_beam_gain(df, "radec", "hours")
    assert units == "hours"
    assert out["earth_beam_model_frequency_mhz"].tolist() == [1.31, 3.93]
    assert np.isfinite(out["earth_beam_relative_gain_db"]).all()
    assert out["earth_visible_by_moon_center"].dtype == bool


def test_nearest_beam_spec_uses_available_upper_v_patterns() -> None:
    assert nearest_beam_spec(0.45)[0] == 1.31
    assert nearest_beam_spec(3.93)[0] == 3.93
    assert nearest_beam_spec(9.18)[0] == 6.55


def test_relative_gain_db_from_linear_normalizes_to_peak() -> None:
    gain_db = relative_gain_db_from_linear(np.array([10.0, 1.0, 0.1, 0.0]))
    assert np.allclose(gain_db[:3], [0.0, -10.0, -20.0])
    assert np.isnan(gain_db[3])


def test_interpolate_cyclic_wraps_beam_angles() -> None:
    angles = np.array([0.0, 90.0, 180.0, 270.0])
    values = np.array([0.0, 9.0, 18.0, 27.0])
    assert np.allclose(interpolate_cyclic(angles, values, np.array([45.0, 315.0, 405.0])), [4.5, 13.5, 4.5])


def test_destripe_normalization_solves_group_offsets_without_erasing_sky() -> None:
    rows = []
    base = pd.Timestamp("1975-01-01")
    sky_db = {0: -4.0, 1: 6.0}
    group_offsets = {0: 100.0, 1: 120.0}
    for group_id, offset in group_offsets.items():
        for pixel, sky in sky_db.items():
            for rep in range(4):
                raw_db = offset + sky
                rows.append(
                    {
                        "time": base + pd.Timedelta(days=4 * group_id, minutes=rep + 10 * pixel),
                        "frequency_band": 4,
                        "frequency_mhz": 1.31,
                        "power": 10.0 ** (raw_db / 10.0),
                        "pixel_index": pixel,
                    }
                )
    df = pd.DataFrame(rows)
    normalized, group_norm = _normalize_destriped_groups(
        df,
        start_time=None,
        group_days=4.0,
        amplitude_filter_db=20.0,
        normalization_iterations=2,
        destripe_iterations=4,
    )
    med = normalized.groupby("pixel_index")["relative_power_db"].median()
    assert np.isclose(med.loc[1] - med.loc[0], 10.0, atol=1e-6)
    offsets = group_norm.sort_values("four_day_group")["group_reference_db"].to_numpy(dtype=float)
    assert np.isclose(offsets[1] - offsets[0], 20.0, atol=1e-6)


def test_iterative_group_reference_rejects_high_db_outlier() -> None:
    values = np.array([100.0, 101.0, 99.0, 102.0, 400.0])
    reference, n_ref = _iterative_group_reference(values, upper_db=2.0, iterations=3)
    assert 99.0 <= reference <= 102.0
    assert n_ref == 4


def test_ten_minute_jump_filter_removes_later_interval() -> None:
    times = pd.date_range("1975-01-01", periods=18, freq="5min")
    values = np.r_[np.zeros(6), np.full(6, 2.0), np.full(6, 2.1)]
    keep = ten_minute_jump_keep(times, values, threshold_db=1.5, bin_seconds=600.0)
    assert keep[:6].all()
    assert not keep[6:8].any()


def test_iterative_sigma_clip_rejects_single_cell_outlier() -> None:
    values = np.array([0.0, 0.1, -0.1, 0.05, 9.0])
    keep = iterative_sigma_clip(values, sigma=4.0)
    assert keep.tolist() == [True, True, True, True, False]
