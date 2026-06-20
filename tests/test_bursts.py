from __future__ import annotations

import numpy as np
import pandas as pd

from rylevonberg.bursts import search_burst_events


def test_search_burst_events_finds_impulsive_outlier() -> None:
    times = pd.date_range("1975-01-01", periods=40, freq="1min")
    y = np.ones(len(times)) * 100.0
    y[20] = 300.0
    clean = pd.DataFrame({"time": times, "frequency_band": 4, "frequency_mhz": 1.31, "antenna": "rv2_coarse", "power": y, "is_valid": True})
    events = pd.DataFrame(
        [
            {
                "event_id": 0,
                "source_name": "jupiter",
                "event_type": "disappearance",
                "predicted_event_time": times[20],
                "frequency_band": 4,
                "frequency_mhz": 1.31,
                "antenna": "rv2_coarse",
            }
        ]
    )
    bursts = search_burst_events(clean, events, window_seconds=600, z_threshold=4.0)
    assert not bursts.empty
    assert bursts.iloc[0]["burst_peak_time"] == times[20]
    assert bursts.iloc[0]["peak_z"] > 0
