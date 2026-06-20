from __future__ import annotations

import numpy as np
import pandas as pd

from rylevonberg.detection import StepFitConfig, run_matched_filter, run_stepfit_detections


def test_stepfit_recovers_positive_occultation_amplitude() -> None:
    times = pd.date_range("1974-01-01", periods=41, freq="1min")
    event_time = pd.Timestamp("1974-01-01 00:20:00")
    y = np.where(times < event_time, 10.0, 4.0)
    clean = pd.DataFrame({"time": times, "frequency_band": 1, "antenna": "rv1_coarse", "power": y, "is_valid": True})
    events = pd.DataFrame(
        [{"event_id": 0, "source_name": "x", "event_type": "disappearance", "predicted_event_time": event_time, "frequency_band": 1, "antenna": "rv1_coarse"}]
    )
    det = run_stepfit_detections(clean, events, StepFitConfig(window_seconds=600, min_samples_per_side=2, timing_grid_seconds=(0.0,)))
    assert det.iloc[0]["amplitude"] > 5.0
    mf = run_matched_filter(clean, events, window_seconds=600)
    assert mf.iloc[0]["matched_amp"] > 5.0


def test_stepfit_reports_sparse_baseline_support() -> None:
    times = pd.to_datetime(
        [
            "1974-01-01 00:19:30",
            "1974-01-01 00:19:40",
            "1974-01-01 00:19:50",
            "1974-01-01 00:20:10",
            "1974-01-01 00:20:20",
            "1974-01-01 00:20:30",
        ]
    )
    event_time = pd.Timestamp("1974-01-01 00:20:00")
    y = np.where(times < event_time, 10.0, 4.0)
    clean = pd.DataFrame({"time": times, "frequency_band": 1, "antenna": "rv1_coarse", "power": y, "is_valid": True})
    events = pd.DataFrame(
        [{"event_id": 0, "source_name": "x", "event_type": "disappearance", "predicted_event_time": event_time, "frequency_band": 1, "antenna": "rv1_coarse"}]
    )
    det = run_stepfit_detections(clean, events, StepFitConfig(window_seconds=120, min_samples_per_side=2, timing_grid_seconds=(0.0,)))
    assert det.iloc[0]["n_pre"] == 3
    assert det.iloc[0]["n_post"] == 3
    assert det.iloc[0]["baseline_support_warning"] == "sparse_pre_or_post_samples"
    assert det.iloc[0]["pre_time_span_s"] == 20.0
