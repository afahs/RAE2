from __future__ import annotations

import numpy as np
import pandas as pd

from rylevonberg.controls import injection_recovery_grid, negative_control_event_ensemble
from rylevonberg.detection import StepFitConfig, run_matched_filter, run_stepfit_detections
from rylevonberg.scoring import ScoreConfig, score_detections


def _scoring_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    times = pd.date_range("1974-11-01", periods=80, freq="1min")
    event_time = pd.Timestamp("1974-11-01 00:40:00")
    rows = []
    events = []
    for freq in [4, 8]:
        for antenna, amp in [("rv1_coarse", 30.0), ("rv2_coarse", 45.0)]:
            power = np.where(times < event_time, 100.0, 100.0 - amp)
            for t, y in zip(times, power):
                rows.append({"time": t, "frequency_band": freq, "antenna": antenna, "power": y, "is_valid": True})
            events.append(
                {
                    "event_id": len(events),
                    "source_name": "control",
                    "event_type": "disappearance",
                    "predicted_event_time": event_time,
                    "frequency_band": freq,
                    "antenna": antenna,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(events)


def test_score_detections_adds_empirical_pvalues_and_antenna_orientation() -> None:
    clean, events = _scoring_fixture()
    cfg = StepFitConfig(window_seconds=600, min_samples_per_side=3, timing_grid_seconds=(0.0,))
    step = run_stepfit_detections(clean, events, cfg)
    matched = run_matched_filter(clean, events, window_seconds=600)
    controls = negative_control_event_ensemble(events, clean, n_random=10, seed=44)
    injection = injection_recovery_grid(clean, events, amplitudes=[5.0], window_seconds=[600.0], min_samples_per_side=3)
    scored, null_summary, null_step, null_matched = score_detections(
        clean,
        step,
        matched,
        controls,
        injection_grid=injection,
        config=ScoreConfig(window_seconds=600, min_samples_per_side=3, timing_grid_seconds=(0.0,)),
    )
    assert not scored.empty
    assert not null_summary.empty
    assert not null_step.empty
    assert not null_matched.empty
    assert {"receiver", "moon_pointing", "best_empirical_p", "detection_grade"}.issubset(scored.columns)
    assert set(scored["receiver"]) == {"upper_v", "lower_v"}
    assert set(scored["moon_pointing"]) == {"away_from_moon", "toward_moon"}
    assert (scored["quality_clean_fraction"] == 1.0).all()
    assert {"supporting_frequency_count", "supporting_antenna_count", "supporting_channel_count"}.issubset(scored.columns)
