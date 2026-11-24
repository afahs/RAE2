from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


decoder = load_module("decode_helios_e5c_summary", ROOT / "scripts" / "decode_helios_e5c_summary.py")
analysis = load_module("run_helios1_rae2_crossmatch", ROOT / "scripts" / "run_helios1_rae2_crossmatch.py")


def test_ibm_hex_float_vectors() -> None:
    words = np.array([0x00000000, 0x41100000, 0xC1200000, 0x42100000], dtype=np.uint32)
    observed = decoder.ibm32_words_to_float(words)
    np.testing.assert_allclose(observed, [0.0, 1.0, -2.0, 16.0], rtol=0, atol=0)


def test_recovered_tape_negative_milliseconds_wrap_at_midnight() -> None:
    stamp = decoder.archived_time(760127.0, -144464.0)
    assert stamp.isoformat() == "1976-01-27T23:57:35.536000+00:00"


def test_documented_frequency_map_and_pairs_are_exact() -> None:
    np.testing.assert_allclose(
        decoder.FREQUENCIES_KHZ,
        [26.5, 50, 65, 85, 115, 150, 195, 255, 340, 445, 585, 765, 1010, 1320, 2280, 3000],
    )
    assert len({rae for rae, _, _ in analysis.PRIMARY_PAIRS}) == len(analysis.PRIMARY_PAIRS)
    assert len({helios for _, helios, _ in analysis.PRIMARY_PAIRS}) == len(analysis.PRIMARY_PAIRS)
    assert sum(primary for _, _, primary in analysis.PRIMARY_PAIRS) == 5


def test_event_window_boundaries_are_closed() -> None:
    center = pd.Timestamp("1975-01-18T09:24:48Z")
    times = pd.Series(pd.to_datetime([
        "1975-01-18T09:19:47Z", "1975-01-18T09:19:48Z",
        "1975-01-18T09:39:48Z", "1975-01-18T09:39:49Z",
    ], utc=True))
    observed = analysis.in_window(times, center, (-5.0, 15.0))
    np.testing.assert_array_equal(observed, [False, True, True, False])
    assert analysis.window_bounds(340)["event"] == (-5.0, 15.0)
    assert analysis.window_bounds(255)["event"] == (0.0, 120.0)


def test_timestamp_shift_and_noaa_exclusion_zone() -> None:
    gate = pd.Timestamp("1975-01-18T09:24:48Z")
    shifted = gate + pd.Timedelta(minutes=-5)
    assert (shifted - gate).total_seconds() == -300
    gates = pd.DatetimeIndex([gate])
    assert not analysis.control_center_is_clean(gate + pd.Timedelta(minutes=119, seconds=59), gates)
    assert analysis.control_center_is_clean(gate + pd.Timedelta(minutes=120), gates)


def test_day_collapse_keeps_channels_and_antennas_separate() -> None:
    frame = pd.DataFrame({
        "gate_date": ["1975-01-18", "1975-01-18", "1975-01-18", "1975-01-19"],
        "antenna": ["A", "A", "B", "A"],
        "frequency_mhz": [0.45, 0.45, 0.45, 0.45],
        "effect": [1.0, 3.0, 100.0, 5.0],
    })
    collapsed = frame.groupby(["antenna", "frequency_mhz", "gate_date"])["effect"].median()
    assert collapsed.loc[("A", 0.45, "1975-01-18")] == 2.0
    assert collapsed.loc[("B", 0.45, "1975-01-18")] == 100.0
    assert len(collapsed) == 3


def test_archive_overlap_resolution_is_same_channel_and_event_blind() -> None:
    frame = pd.DataFrame({
        "time": pd.to_datetime(["1975-01-01T00:00:00Z"] * 3),
        "channel_id": [1, 1, 2],
        "frequency_khz": [26.5, 26.5, 50.0],
        "logical_record": [10, 11, 10],
        "nonzero_count": [15.0, 30.0, 5.0],
        "two_percent_max_10logt": [999.0, 1.0, 50.0],
    })
    selected, audit = analysis.select_archive_science_records(frame)
    channel_one = selected[selected["channel_id"] == 1].iloc[0]
    assert channel_one["logical_record"] == 11
    assert channel_one["two_percent_max_10logt"] == 1.0
    assert audit["duplicate_channel_rows_removed"] == 1
    assert len(selected) == 2


def test_bh_qvalues_known_example() -> None:
    observed = analysis.bh_qvalues(pd.Series([0.01, 0.04, 0.03, 0.20]))
    np.testing.assert_allclose(observed, [0.04, 0.05333333333333334, 0.05333333333333334, 0.20])
