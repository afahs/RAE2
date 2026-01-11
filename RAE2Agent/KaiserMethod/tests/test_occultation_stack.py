import importlib.util
from pathlib import Path

import numpy as np


STACK_PATH = Path(__file__).resolve().parents[1] / "occultation_stack.py"

def _load_module():
    spec = importlib.util.spec_from_file_location("occultation_stack", STACK_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_sign_bit_mapping():
    ks = _load_module()
    assert ks.sign_bit_from_delta(0.1) == 1
    assert ks.sign_bit_from_delta(-0.1) == -1
    assert ks.sign_bit_from_delta(0.0) == -1


def test_event_type_delta_mapping():
    ks = _load_module()
    mean_before = 10.0
    mean_after = 8.0
    delta_dis = ks.compute_delta_db(mean_before, mean_after, "DISAPPEARANCE")
    delta_rep = ks.compute_delta_db(mean_before, mean_after, "REAPPEARANCE")
    assert np.isclose(delta_dis, 2.0)
    assert np.isclose(delta_rep, -2.0)


def test_run_length_detection():
    ks = _load_module()
    channel_ids = np.array([0, 1, 2, 4, 5, 6])
    deltas = np.array([0.6, 0.6, 0.6, 0.7, 0.7, 0.7])
    signs = np.array([1, 1, 1, 1, 1, 1])
    runs = ks.find_candidate_runs(channel_ids, deltas, signs, min_run=3, delta_threshold_db=0.5)
    assert len(runs) == 2
    assert runs[0]["channel_start"] == 0
    assert runs[0]["channel_end"] == 2
    assert runs[0]["run_length"] == 3
    assert runs[1]["channel_start"] == 4
    assert runs[1]["channel_end"] == 6
    assert runs[1]["run_length"] == 3

    deltas[2] = 0.4
    runs = ks.find_candidate_runs(channel_ids, deltas, signs, min_run=3, delta_threshold_db=0.5)
    assert len(runs) == 1
    assert runs[0]["channel_start"] == 4
    assert runs[0]["channel_end"] == 6
