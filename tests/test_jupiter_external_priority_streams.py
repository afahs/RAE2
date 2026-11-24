from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.jupiter_external_stream_common import (
    daily_roll_plan,
    log_rate_ratio,
    rolled_vectors,
)
from scripts.run_jupiter_external_stream1_rae1 import in_circular_windows, slope
from scripts.run_jupiter_external_stream2_uag42 import (
    label_permutation_null,
    parse_frequency_range,
    parse_hhmm,
)
from scripts.run_jupiter_external_stream3_latitude import beam_mask, magnetic_latitude


def test_rolls_preserve_daily_candidate_totals() -> None:
    source = pd.DataFrame(
        {
            "date": pd.to_datetime(["1974-11-01"] * 4 + ["1974-11-02"] * 3),
        }
    )
    gate = np.asarray([1, 0, 1, 0, 0, 1, 0], dtype=np.int8)
    groups, offsets = daily_roll_plan(source, 40, np.random.default_rng(42))
    rolled = rolled_vectors(gate, groups, offsets)
    assert np.all(rolled[:, :4].sum(axis=1) == 2)
    assert np.all(rolled[:, 4:].sum(axis=1) == 1)


def test_log_rate_ratio_is_positive_for_enrichment() -> None:
    gate = np.asarray([1, 1, 0, 0, 0, 0], dtype=np.int8)
    target = np.asarray([1, 1, 1, 0, 0, 0], dtype=bool)
    assert float(log_rate_ratio(gate, target)) > 0


def test_stream1_fixed_io_windows_and_intensity_slope() -> None:
    phase = np.asarray([69.9, 70.0, 109.9, 110.0, 209.9, 210.0, 269.9, 270.0])
    expected = np.asarray([False, True, True, False, False, True, True, False])
    assert np.array_equal(in_circular_windows(phase, ((70, 110), (210, 270))), expected)
    assert float(slope(np.asarray([0.0, 0.1, 0.3, 0.8]))) > 0


def test_stream2_parses_next_day_and_reversed_frequency_range() -> None:
    date = pd.Timestamp("1975-02-23")
    assert parse_hhmm(date, "2535") == pd.Timestamp("1975-02-24 01:35:00")
    assert parse_frequency_range("25-18") == (18.0, 25.0)


def test_stream2_exchange_null_is_centered_for_symmetric_scores() -> None:
    scores = [np.asarray([0.0, 0.5, 1.0])] * 30
    null = label_permutation_null(scores, 2000, np.random.default_rng(5))
    assert abs(float(np.mean(null))) < 0.04


def test_stream3_tilted_dipole_and_beam_mask_are_finite() -> None:
    cml = np.asarray([0.0, 90.0, 180.0, 270.0])
    observer = np.full(4, 2.0)
    magnetic = magnetic_latitude(cml, observer)
    assert np.all(np.isfinite(magnetic))
    assert np.ptp(magnetic) > 15.0
    inside = beam_mask(np.asarray([-2.0, 3.0, 8.0, 8.01]))
    assert np.array_equal(inside, np.asarray([True, True, True, False]))
