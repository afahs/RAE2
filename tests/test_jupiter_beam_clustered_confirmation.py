from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_jupiter_beam_clustered_confirmation import (
    EXPECTED_IO_WINDOWS,
    in_expected_io_windows,
    mark_cluster_peak_candidates,
    shared_daily_roll_plan,
)
from scripts.jupiter_external_stream_common import rolled_vectors


def test_expected_io_windows_have_documented_half_open_edges() -> None:
    phase = np.array([79.99, 80.0, 129.99, 130.0, 204.99, 205.0, 259.99, 260.0])
    assert EXPECTED_IO_WINDOWS == ((80.0, 130.0), (205.0, 260.0))
    assert in_expected_io_windows(phase).tolist() == [False, True, True, False, False, True, True, False]


def test_cluster_gate_marks_only_the_peak_high_sample() -> None:
    source = pd.DataFrame(
        {
            "time": pd.to_datetime(
                [
                    "1974-01-01 00:00:00",
                    "1974-01-01 00:01:00",
                    "1974-01-01 00:02:00",
                    "1974-01-01 00:10:00",
                ]
            ),
            "local_normalized_power": [5.2, 7.0, 6.0, 9.0],
            "antenna": ["rv1_coarse"] * 4,
            "frequency_band": [7] * 4,
            "frequency_mhz": [4.70] * 4,
            "io_phase_deg": [90.0, 91.0, 92.0, 93.0],
            "jupiter_cml_system_iii_deg": [10.0, 11.0, 12.0, 13.0],
            "jupiter_beam_separation_deg": [1.0] * 4,
            "jupiter_beam_relative_gain_db": [-1.0] * 4,
            "beam_half_width_deg": [20.0] * 4,
        }
    )
    gate, bursts = mark_cluster_peak_candidates(source, threshold=5.0, min_cluster_samples=2)
    assert gate.tolist() == [0, 1, 0, 0]
    assert len(bursts) == 1
    assert bursts.iloc[0]["peak_local_normalized_power"] == 7.0
    assert bursts.iloc[0]["n_high_samples"] == 3


def test_shared_roll_preserves_each_days_candidate_count() -> None:
    source = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["1974-01-01", "1974-01-01", "1974-01-01", "1974-01-02", "1974-01-02"]
            )
        }
    )
    uniforms = np.array([[0.0, 0.0], [0.5, 0.5], [0.99, 0.99]])
    columns = {
        pd.Timestamp("1974-01-01"): 0,
        pd.Timestamp("1974-01-02"): 1,
    }
    groups, offsets = shared_daily_roll_plan(source, uniforms, columns)
    rolled = rolled_vectors(np.array([1, 0, 1, 0, 1], dtype=np.int8), groups, offsets)
    assert rolled[:, :3].sum(axis=1).tolist() == [2, 2, 2]
    assert rolled[:, 3:].sum(axis=1).tolist() == [1, 1, 1]
