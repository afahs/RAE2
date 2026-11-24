from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_jupiter_phase_occurrence_svd.py"
SPEC = importlib.util.spec_from_file_location("jupiter_phase_occurrence_svd", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_entropy_svd_rejects_uniform_and_keeps_sparse_spectral_mode() -> None:
    temporal = np.sin(np.linspace(0.0, 5.0 * np.pi, 100))
    sparse = np.zeros((9, 100))
    sparse[2] = temporal
    sparse_result = MODULE.entropy_svd(sparse, MODULE.PAPER_ENTROPY_LIMIT_NORMALIZED)
    assert sparse_result.retained[0]
    uniform = np.ones((9, 1)) @ temporal[None, :]
    uniform_result = MODULE.entropy_svd(uniform, MODULE.PAPER_ENTROPY_LIMIT_NORMALIZED)
    assert not uniform_result.retained[0]


def test_io_phase_is_zero_at_superior_conjunction_and_90_eastward() -> None:
    observer = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    io = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    phase = MODULE.io_phase_from_vectors(observer, io)
    assert np.allclose(phase, [0.0, 90.0])


def test_top_fraction_never_selects_zero_scores() -> None:
    score = np.array([0.0, 0.0, 1.0, 2.0, 3.0])
    selected = MODULE.select_top_fraction(score, np.ones(5, dtype=bool), 0.4)
    assert selected.sum() == 2
    assert selected[3] and selected[4]
    assert not selected[0]


def test_phase_curve_uses_exposure_denominator_and_preserves_total_candidates() -> None:
    times = pd.date_range("1975-01-01", periods=96, freq="15min")
    phase = np.tile(np.arange(0.0, 360.0, 15.0), 4)
    source = pd.DataFrame(
        {
            "time": times,
            "date": times.floor("D"),
            "valid_interval": True,
            "jupiter_occulted": False,
            "candidate_present": phase == 90.0,
            "io_phase_deg": phase + 0.1,
        }
    )
    summary, null_counts, exposure = MODULE.phase_curve_with_null(
        source, "io_phase_deg", 15.0, 20, np.random.default_rng(4)
    )
    assert int(summary["n_valid"].sum()) == 96
    assert int(summary["n_candidate"].sum()) == 4
    assert np.all(null_counts.sum(axis=1) == 4)
    assert int(exposure.sum()) == 96
