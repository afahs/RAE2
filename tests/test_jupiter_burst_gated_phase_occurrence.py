from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_jupiter_burst_gated_phase_occurrence.py"
SPEC = importlib.util.spec_from_file_location("jupiter_burst_gated_phase_occurrence", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_gate_vector_counts_extreme_bands() -> None:
    events = pd.DataFrame(
        {
            **{f"band_{band}_percentile": [0.99, 0.20] for band in range(1, 10)},
        }
    )
    events.loc[0, "band_1_percentile"] = 0.999
    events.loc[0, "band_2_percentile"] = 0.998
    one = MODULE.gate_vector(events, 0.995, 1)
    two = MODULE.gate_vector(events, 0.995, 2)
    three = MODULE.gate_vector(events, 0.995, 3)
    assert one.tolist() == [True, False]
    assert two.tolist() == [True, False]
    assert three.tolist() == [False, False]


def test_rolled_null_preserves_candidate_total() -> None:
    source = pd.DataFrame(
        {
            "date": pd.to_datetime(["1975-01-01"] * 6 + ["1975-01-02"] * 6),
        }
    )
    groups, offsets = MODULE.make_roll_offsets(source, 20, np.random.default_rng(7))
    gate = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.int8)
    bins = np.arange(12) % 4
    null = MODULE.rolled_null_counts(gate, bins, 4, groups, offsets)
    assert np.all(null.sum(axis=1) == gate.sum())


def test_expected_io_masks_do_not_overlap() -> None:
    centers = (np.arange(24) + 0.5) * 15.0
    target, control, _ = MODULE.expected_masks(centers, "io_phase")
    assert not np.any(target & control)
    assert np.any(target)
    assert np.any(control)


def test_observed_contrast_rewards_expected_io_windows() -> None:
    centers = (np.arange(24) + 0.5) * 15.0
    source = pd.DataFrame({"io_phase_deg": centers})
    target, _, _ = MODULE.expected_masks(centers, "io_phase")
    contrast = MODULE.observed_contrast_for_gate(
        source, target.astype(bool), "io_phase_deg", "io_phase", 15.0
    )
    assert contrast > 0.9
