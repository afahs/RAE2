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
