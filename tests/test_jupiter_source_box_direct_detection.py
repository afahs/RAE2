from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_jupiter_source_box_direct_detection import (
    annotate_source_boxes,
    build_alignment_events,
    circular_between,
    mask_times_in_windows,
    shifted_window_mask,
    source_box_definitions,
)


def test_circular_between_handles_wrapped_io_c_cml_range() -> None:
    vals = np.array([10.0, 50.0, 299.0, 300.0, 350.0])
    assert circular_between(vals, 300.0, 20.0).tolist() == [True, False, False, True, True]


def test_annotate_source_boxes_marks_standard_boxes() -> None:
    df = pd.DataFrame(
        {
            "jupiter_cml_spice_deg": [220.0, 150.0, 330.0, 50.0, 250.0],
            "io_phase_spice_deg": [230.0, 90.0, 240.0, 110.0, 20.0],
        }
    )
    out = annotate_source_boxes(df, pad_deg=0.0)
    assert out["source_io_a"].tolist() == [True, False, False, False, False]
    assert out["source_io_b"].tolist() == [False, True, False, False, False]
    assert out["source_io_c"].tolist() == [False, False, True, False, False]
    assert out["source_io_d"].tolist() == [False, False, False, True, False]
    assert out["source_box_any"].tolist() == [True, True, True, True, False]


def test_mask_times_in_windows_and_shifted_controls() -> None:
    times = pd.to_datetime(
        [
            "1975-01-01 00:00",
            "1975-01-01 01:00",
            "1975-01-01 02:00",
            "1975-01-02 01:00",
        ]
    )
    windows = pd.DataFrame(
        {
            "start_time": pd.to_datetime(["1975-01-01 00:30", "1975-01-02 00:30"]),
            "end_time": pd.to_datetime(["1975-01-01 01:30", "1975-01-02 01:30"]),
        }
    )
    assert mask_times_in_windows(pd.Series(times), windows).tolist() == [False, True, False, True]
    assert shifted_window_mask(pd.Series(times), windows, shift_days=1.0).tolist() == [False, True, False, False]


def test_source_box_definitions_records_analysis_padding() -> None:
    defs = source_box_definitions(pad_deg=5.0)
    io_a = defs[defs["source_box"].eq("Io-A")].iloc[0]
    assert io_a["analysis_cml_lo_deg"] == 195.0
    assert io_a["analysis_cml_hi_deg"] == 275.0
    assert io_a["analysis_io_phase_lo_deg"] == 200.0


def test_build_alignment_events_uses_entry_exit_and_maser_peak() -> None:
    geom = pd.DataFrame(
        {
            "time": pd.to_datetime(
                [
                    "1975-01-01 00:00",
                    "1975-01-01 00:10",
                    "1975-01-01 00:20",
                ]
            ),
            "jupiter_visible_by_moon": [True, True, True],
            "source_box_any": [True, True, True],
            "maser_zarka_io_score": [0.2, 0.9, 0.4],
        }
    )
    windows = pd.DataFrame(
        {
            "window_id": ["src0001"],
            "start_time": pd.to_datetime(["1975-01-01 00:00"]),
            "end_time": pd.to_datetime(["1975-01-01 00:20"]),
            "duration_min": [20.0],
            "source_boxes": ["Io-A"],
            "median_cml_deg": [220.0],
            "median_io_phase_deg": [230.0],
        }
    )
    events = build_alignment_events(
        geom,
        windows,
        shift_days=[1.0],
        sample_start=pd.Timestamp("1975-01-01 00:00"),
        sample_end=pd.Timestamp("1975-01-03 00:00"),
    )
    real = events[events["role"].eq("real")].sort_values("align_type")
    assert sorted(real["align_type"].tolist()) == ["entry", "exit", "maser_peak"]
    peak = real[real["align_type"].eq("maser_peak")].iloc[0]
    assert peak["event_time"] == pd.Timestamp("1975-01-01 00:10")
    shifted = events[events["role"].eq("shifted_control")]
    assert len(shifted) == 3
    assert shifted[shifted["align_type"].eq("entry")]["event_time"].iloc[0] == pd.Timestamp("1975-01-02 00:00")
