from __future__ import annotations

import pandas as pd
import numpy as np

from rylevonberg.geometry import find_limb_transitions, moon_limb_angle_deg, visible_by_moon


def test_moon_limb_sign_convention() -> None:
    sc = np.array([[2000.0, 0.0, 0.0]])
    toward_moon = np.array([[-1.0, 0.0, 0.0]])
    away = np.array([[1.0, 0.0, 0.0]])
    assert moon_limb_angle_deg(sc, toward_moon)[0] < 0
    assert not visible_by_moon(sc, toward_moon)[0]
    assert moon_limb_angle_deg(sc, away)[0] > 0
    assert visible_by_moon(sc, away)[0]


def test_transition_time_interpolates_zero_crossing() -> None:
    times = pd.date_range("1974-01-01", periods=3, freq="1min")
    limb = np.array([2.0, -2.0, -3.0])
    events = find_limb_transitions(times, limb)
    assert len(events) == 1
    assert events.iloc[0]["event_type"] == "disappearance"
    assert events.iloc[0]["predicted_event_time"] == pd.Timestamp("1974-01-01 00:00:30")


def test_transition_finder_ignores_missing_limb_samples() -> None:
    times = pd.date_range("1974-01-01", periods=5, freq="1min")
    limb = np.array([2.0, np.nan, 1.0, -1.0, -2.0])

    events = find_limb_transitions(times, limb)

    assert len(events) == 1
    assert events.iloc[0]["predicted_event_time"] == pd.Timestamp("1974-01-01 00:02:30")


def test_invalid_orbit_position_does_not_manufacture_source_events() -> None:
    times = pd.date_range("1974-01-01", periods=3, freq="15s")
    positions = np.array([[-1800., 2500., 0.], [-1000., 100., 0.], [-1800., 2500., 0.]])
    directions = np.tile([1., 0., 0.], (3, 1))
    limb = moon_limb_angle_deg(positions, directions)
    assert np.isnan(limb[1])
    assert limb[0] > 0 and limb[2] > 0
    assert find_limb_transitions(times, limb).empty


def test_zero_and_surface_orbit_positions_have_unknown_limb_angle() -> None:
    positions = np.array([[0., 0., 0.], [1737.4, 0., 0.]])
    assert np.isnan(moon_limb_angle_deg(positions, np.tile([-1., 0., 0.], (2, 1)))).all()
