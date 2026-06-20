from __future__ import annotations

import pandas as pd

from rylevonberg.events import predict_events
from rylevonberg.frames import offset_unit_vectors_tangent


def test_predict_fixed_source_disappearance_simple_geometry() -> None:
    df = pd.DataFrame(
        {
            "time": pd.date_range("1974-01-01", periods=5, freq="1min"),
            "position_x": [2000.0, 2000.0, 2000.0, -2000.0, -2000.0],
            "position_y": [0.0] * 5,
            "position_z": [0.0] * 5,
            "earth_unit_vector_x": [1.0] * 5,
            "earth_unit_vector_y": [0.0] * 5,
            "earth_unit_vector_z": [0.0] * 5,
            "frequency_band": [1] * 5,
            "antenna": ["rv1_coarse"] * 5,
            "power": [1.0] * 5,
        }
    )
    sources = pd.DataFrame([{"source_name": "plus_x", "kind": "fixed", "ra_deg": 0.0, "dec_deg": 0.0, "frame": "fk4"}])
    events, states = predict_events(df, sources, frequencies=[1], antennas=["rv1_coarse"])
    assert not states.empty
    assert "disappearance" in set(events["event_type"])


def test_predict_events_can_exclude_limb_contaminants() -> None:
    df = pd.DataFrame(
        {
            "time": pd.date_range("1974-01-01", periods=5, freq="1min"),
            "position_x": [2000.0, 2000.0, 2000.0, -2000.0, -2000.0],
            "position_y": [0.0] * 5,
            "position_z": [0.0] * 5,
            "earth_unit_vector_x": [1.0] * 5,
            "earth_unit_vector_y": [0.0] * 5,
            "earth_unit_vector_z": [0.0] * 5,
            "frequency_band": [1] * 5,
            "antenna": ["rv1_coarse"] * 5,
            "power": [1.0] * 5,
        }
    )
    sources = pd.DataFrame([{"source_name": "plus_x", "kind": "fixed", "ra_deg": 0.0, "dec_deg": 0.0, "frame": "fk4"}])
    near_limb = pd.DataFrame([{"source_name": "near_limb", "kind": "fixed", "ra_deg": 1.0, "dec_deg": 0.0, "frame": "fk4"}])
    far_limb = pd.DataFrame([{"source_name": "far_limb", "kind": "fixed", "ra_deg": 180.0, "dec_deg": 0.0, "frame": "fk4"}])

    filtered, _ = predict_events(df, sources, frequencies=[1], antennas=["rv1_coarse"], limb_exclusion_sources_df=near_limb, limb_exclusion_deg=3.0)
    retained, _ = predict_events(df, sources, frequencies=[1], antennas=["rv1_coarse"], limb_exclusion_sources_df=far_limb, limb_exclusion_deg=3.0)

    assert filtered.empty
    assert not retained.empty
    assert "limb_exclusion_nearest_abs_deg" in retained.columns


def test_offset_unit_vectors_tangent_preserves_unit_norm_and_offset_scale() -> None:
    import numpy as np

    base = np.array([[1.0, 0.0, 0.0]])
    shifted = offset_unit_vectors_tangent(base, east_offset_deg=1.0)
    assert np.isclose(np.linalg.norm(shifted[0]), 1.0)
    sep = np.degrees(np.arccos(np.clip(np.dot(base[0], shifted[0]), -1.0, 1.0)))
    assert 0.9 < sep < 1.1
