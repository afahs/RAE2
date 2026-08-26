from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_naive_subspacecraft_land_ocean.py"
SPEC = importlib.util.spec_from_file_location("run_naive_subspacecraft_land_ocean", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_alignment_cosine_identifies_earth_facing_side() -> None:
    spacecraft = np.asarray([[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    earth = np.asarray([[1.0, 0.0, 0.0]] * 3)
    result = MODULE.alignment_cosine(spacecraft, earth)
    np.testing.assert_allclose(result, [1.0, -1.0, 0.0], atol=1e-12)


def test_nearest_center_indices_selects_closest_pass() -> None:
    centers = np.asarray([100, 300, 600], dtype=np.int64)
    samples = np.asarray([90, 210, 590, 800], dtype=np.int64)
    index, distance = MODULE.nearest_center_indices(samples, centers)
    assert index.tolist() == [0, 1, 2, 2]
    assert distance.tolist() == [10, 90, 10, 200]


def test_parse_windows_sorts_and_deduplicates() -> None:
    assert MODULE.parse_windows("10, 2,5,2") == [2.0, 5.0, 10.0]
