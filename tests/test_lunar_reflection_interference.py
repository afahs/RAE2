from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.investigate_lower_v_lunar_reflection_interference import (
    _event_side_coordinate,
    attach_fringe_period,
    fit_fringe_group,
    specular_reflection_geometry,
)


def test_specular_reflection_geometry_symmetric_collinear_case() -> None:
    spacecraft = np.array([[2.0, 0.0, 0.0]])
    earth_unit = np.array([[1.0, 0.0, 0.0]])

    geom = specular_reflection_geometry(
        spacecraft,
        earth_unit,
        moon_radius_km=1.0,
        earth_moon_distance_km=100.0,
        n_iter=5,
    )

    assert np.allclose([geom["specular_x"][0], geom["specular_y"][0], geom["specular_z"][0]], [1.0, 0.0, 0.0])
    assert np.isclose(geom["delta_l_km"][0], 2.0)
    assert bool(geom["reflection_valid"][0])


def test_specular_reflection_geometry_is_exact_near_limb() -> None:
    earth_distance = 100.0
    spacecraft_distance = 2.0
    limb_beta = np.arccos(1.0 / earth_distance) + np.arccos(1.0 / spacecraft_distance)
    beta = limb_beta - 1e-4
    spacecraft = spacecraft_distance * np.array([[np.cos(beta), np.sin(beta), 0.0]])
    earth_unit = np.array([[1.0, 0.0, 0.0]])

    geom = specular_reflection_geometry(
        spacecraft,
        earth_unit,
        moon_radius_km=1.0,
        earth_moon_distance_km=earth_distance,
    )

    assert bool(geom["reflection_valid"][0])
    assert geom["specular_mirror_error"][0] < 1e-10
    assert 0.0 < geom["delta_l_km"][0] < 1e-6


def test_fringe_rate_is_computed_within_dwell_not_across_scan_gap() -> None:
    times = np.array([0.0, 1.0, 2.0, 100.0, 101.0])
    samples = pd.DataFrame(
        {
            "time_shift_s": 0.0,
            "frequency_band": 1,
            "antenna": "rv2_coarse",
            "event_id": 1,
            "event_type": "disappearance",
            "t_rel_sec": times,
            "reflection_valid": True,
            "phase_rad": 2.0 * np.pi * 0.1 * times,
        }
    )

    out = attach_fringe_period(samples)

    assert np.allclose(out["fringe_period_s"], 10.0)
    assert out["dwell_id"].tolist() == [0, 0, 0, 1, 1]


def test_event_side_coordinate_aligns_visible_sides() -> None:
    t = np.array([-30.0, 30.0])

    assert np.allclose(_event_side_coordinate(t, "disappearance"), [30.0, -30.0])
    assert np.allclose(_event_side_coordinate(t, "reappearance"), [-30.0, 30.0])


def test_fringe_fit_recovers_synthetic_phase_amplitude_with_event_offsets() -> None:
    rng = np.random.default_rng(123)
    rows = []
    amp = 0.8
    phase0 = 0.4
    for event_id in range(8):
        t = np.linspace(-120.0, 120.0, 80)
        phase = np.linspace(0.0, 8.0 * np.pi, t.size) + 0.2 * event_id
        baseline = 2.0 + 0.15 * event_id + 0.002 * t
        y = baseline + amp * np.cos(phase + phase0) + rng.normal(0.0, 0.03, size=t.size)
        for tv, ph, yv in zip(t, phase, y):
            rows.append(
                {
                    "event_id": event_id,
                    "event_type": "disappearance",
                    "frequency_band": 2,
                    "frequency_mhz": 0.70,
                    "antenna": "rv2_coarse",
                    "t_rel_sec": tv,
                    "phase_rad": ph,
                    "z_power": yv,
                }
            )
    samples = pd.DataFrame(rows)

    fit = fit_fringe_group(
        samples,
        phase_mode="real",
        baseline_order=1,
        include_step=False,
        rng=np.random.default_rng(456),
    )

    assert abs(fit["fringe_amplitude_z"] - amp) < 0.08
    assert fit["fringe_snr"] > 20.0
