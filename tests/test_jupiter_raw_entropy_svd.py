from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_jupiter_raw_entropy_svd.py"
SPEC = importlib.util.spec_from_file_location("jupiter_raw_entropy_svd", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_entropy_svd_keeps_sparse_mode_and_rejects_uniform_mode() -> None:
    temporal = np.sin(np.linspace(0.0, 3.0 * np.pi, 80))
    sparse = np.zeros((9, 80))
    sparse[3] = temporal
    sparse_result = MODULE.entropy_svd(sparse)
    assert sparse_result.retained[0]
    assert sparse_result.entropy_normalized[0] < 0.01

    uniform = np.ones((9, 1)) @ temporal[None, :]
    uniform_result = MODULE.entropy_svd(uniform)
    assert not uniform_result.retained[0]
    assert uniform_result.entropy_normalized[0] > 0.99


def test_reappearance_is_folded_to_high_before_low_after() -> None:
    event_time = pd.Timestamp("1975-01-01T00:00:00")
    offsets = np.arange(-300.0, 301.0, 10.0)
    lookup = {}
    for antenna in MODULE.ANTENNAS:
        for band in range(1, 10):
            time_ns = event_time.value + (offsets * 1e9).astype(np.int64)
            # Reappearance: low before the real event and high after it.
            power = np.where(offsets < 0.0, 1.0, 4.0)
            lookup[(antenna, band)] = MODULE.ReceiverSeries(
                time_ns=time_ns,
                power=power,
                is_valid=np.ones_like(power, dtype=bool),
            )
    matrix, times, _ = MODULE.extract_event_matrix(
        lookup,
        event_time,
        "reappearance",
        "rv1_coarse",
        0.0,
        "finite",
        300.0,
        60.0,
    )
    edge = MODULE.edge_spectrum(matrix, times, half_width_s=240.0)
    assert np.all(edge > 0.0)


def test_whitening_centers_and_scales_rows() -> None:
    base = np.linspace(-3.0, 5.0, 80)
    matrix = np.vstack([base + 2.0 * index for index in range(9)])
    whitened, flagged = MODULE.whiten_frequency_rows(matrix)
    assert not np.any(flagged)
    assert np.allclose(np.median(whitened, axis=1), 0.0, atol=1e-12)
    for row in whitened:
        assert np.isclose(np.percentile(row[row > 0.0], 75.0), 1.0, atol=1e-12)


def test_exact_block_null_rewards_zero_aligned_positive_edges() -> None:
    rows = []
    for block in range(3):
        for antenna in MODULE.ANTENNAS:
            for shift in MODULE.ALL_SHIFTS_S:
                for band, frequency in enumerate(MODULE.FREQUENCIES_MHZ, start=1):
                    target = shift == 0.0
                    rows.append(
                        {
                            "source_name": "jupiter",
                            "validity_mode": "finite",
                            "entropy_limit_normalized": MODULE.PAPER_ENTROPY_LIMIT_NORMALIZED,
                            "lunar_block": block,
                            "antenna": antenna,
                            "control_shift_s": shift,
                            "frequency_band": band,
                            "frequency_mhz": frequency,
                            "raw_edge": 2.0 if target and band in (1, 4, 7) else 0.0,
                            "svd_edge": 2.0 if target and band in (1, 4, 7) else 0.0,
                            "n_events": 1,
                        }
                    )
    summary, null = MODULE.exact_block_null(
        pd.DataFrame(rows),
        "jupiter",
        "finite",
        MODULE.PAPER_ENTROPY_LIMIT_NORMALIZED,
        "svd_edge",
    )
    assert summary["target_rank_rv1_of_5"] == 1
    assert summary["target_rank_rv2_of_5"] == 1
    assert summary["replication_p"] == 1.0 / (1.0 + len(null))
