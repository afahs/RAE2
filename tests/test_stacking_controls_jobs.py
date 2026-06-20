from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rylevonberg.controls import injection_recovery, randomized_event_times, time_reversed_events
from rylevonberg.detection import StepFitConfig
from rylevonberg.jobs import require_interactive_mode
from rylevonberg.stacking import aligned_profiles, stack_profiles


def _mini_data():
    times = pd.date_range("1974-01-01", periods=30, freq="1min")
    event_time = pd.Timestamp("1974-01-01 00:15:00")
    y = np.where(times < event_time, 10.0, 5.0)
    clean = pd.DataFrame({"time": times, "frequency_band": 1, "antenna": "rv1_coarse", "power": y, "is_valid": True})
    events = pd.DataFrame(
        [{"event_id": 0, "source_name": "x", "event_type": "disappearance", "predicted_event_time": event_time, "frequency_band": 1, "antenna": "rv1_coarse"}]
    )
    return clean, events


def test_stack_profiles_outputs_summary() -> None:
    clean, events = _mini_data()
    profiles = aligned_profiles(clean, events, window_seconds=300, bin_seconds=60)
    stacked, summary = stack_profiles(profiles, n_bootstrap=0)
    assert not stacked.empty
    assert not summary.empty


def test_controls_and_job_guard() -> None:
    clean, events = _mini_data()
    rand = randomized_event_times(events, clean["time"].min(), clean["time"].max(), seed=1)
    rev = time_reversed_events(events)
    inj = injection_recovery(clean, events, [1.0], StepFitConfig(window_seconds=300, min_samples_per_side=2))
    assert rand.iloc[0]["predicted_event_time"] != events.iloc[0]["predicted_event_time"]
    assert rev.iloc[0]["event_type"] == "reappearance"
    assert not inj.empty
    assert require_interactive_mode("interactive") == "interactive"
    with pytest.raises(RuntimeError):
        require_interactive_mode("sbatch")

