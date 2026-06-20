from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_jupiter_per_window_event_survey import count_events, shifted_windows, top_candidate_windows


def test_count_events_clusters_high_samples_by_time_gap() -> None:
    time_ns = pd.to_datetime(
        [
            "1975-01-01 00:00:00",
            "1975-01-01 00:01:00",
            "1975-01-01 00:10:00",
            "1975-01-01 00:10:30",
            "1975-01-01 00:30:00",
        ]
    ).to_numpy(dtype="datetime64[ns]").astype("int64")
    high = np.array([True, True, True, True, False])
    assert count_events(time_ns, high, max_gap_s=180.0, min_high_samples=1) == 2
    assert count_events(time_ns, high, max_gap_s=180.0, min_high_samples=2) == 2
    assert count_events(time_ns, high, max_gap_s=20.0, min_high_samples=2) == 0


def test_shifted_windows_preserves_window_metadata_and_clips_to_sample_range() -> None:
    windows = pd.DataFrame(
        {
            "window_id": ["IoA_0001"],
            "source_box": ["Io-A"],
            "start_time": pd.to_datetime(["1975-01-02 00:00"]),
            "end_time": pd.to_datetime(["1975-01-02 01:00"]),
            "duration_min": [60.0],
            "median_cml_deg": [220.0],
            "median_io_phase_deg": [230.0],
            "max_maser_zarka_io_score": [0.9],
        }
    )
    shifted = shifted_windows(
        windows,
        shift_days=[-1.0, 10.0],
        sample_start=pd.Timestamp("1975-01-01 00:00"),
        sample_end=pd.Timestamp("1975-01-03 00:00"),
    )
    assert len(shifted) == 1
    assert shifted["shift_days"].iloc[0] == -1.0
    assert shifted["start_time"].iloc[0] == pd.Timestamp("1975-01-01 00:00")
    assert shifted["source_box"].iloc[0] == "Io-A"


def test_top_candidate_windows_prefers_repeated_highband_support() -> None:
    rows = []
    for window_id, high_positive in [("w1", 2), ("w2", 1)]:
        for freq in [0.45, 3.93, 4.70]:
            rows.append(
                {
                    "window_id": window_id,
                    "source_box": "Io-A",
                    "start_time": "1975-01-01 00:00",
                    "end_time": "1975-01-01 01:00",
                    "duration_min": 60.0,
                    "median_cml_deg": 220.0,
                    "median_io_phase_deg": 230.0,
                    "frequency_mhz": freq,
                    "threshold_name": "daily_factor_2",
                    "event_rate_excess_per_hr": 1.0 if freq != 0.45 and high_positive > 0 else -1.0,
                    "n_events": 1 if freq != 0.45 and high_positive > 0 else 0,
                }
            )
            if freq != 0.45:
                high_positive -= 1
    top = top_candidate_windows(pd.DataFrame(rows), threshold_name="daily_factor_2")
    assert top["window_id"].iloc[0] == "w1"
    assert top["positive_highband_channels"].iloc[0] == 2
