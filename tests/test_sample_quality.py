from __future__ import annotations

import numpy as np
import pandas as pd

from rylevonberg.detection import StepFitConfig, run_stepfit_detections
from rylevonberg.sample_quality import add_strict_valid_column, strict_power_mask


def test_strict_power_mask_rejects_invalid_power_but_not_normal_plateau() -> None:
    normal = pd.DataFrame({"power": [10.0] * 20 + [4.0] * 20, "is_valid": [True] * 40})
    assert strict_power_mask(normal).all()

    bad = pd.DataFrame({"power": [10.0] * 20 + [0.0, -1.0, np.nan, 1.0e9], "is_valid": [True] * 24})
    masked = add_strict_valid_column(bad)
    assert not bool(masked.loc[20, "strict_is_valid"])
    assert not bool(masked.loc[21, "strict_is_valid"])
    assert not bool(masked.loc[22, "strict_is_valid"])
    assert not bool(masked.loc[23, "strict_is_valid"])
    assert bool(masked.loc[0, "strict_is_valid"])


def test_stepfit_uses_strict_mask_for_stale_clean_files() -> None:
    times = pd.date_range("1974-01-01", periods=41, freq="1min")
    event_time = pd.Timestamp("1974-01-01 00:20:00")
    y = np.where(times < event_time, 10.0, 4.0).astype(float)
    y[19] = 0.0
    y[21] = 1.0e9
    clean = pd.DataFrame({"time": times, "frequency_band": 1, "antenna": "rv1_coarse", "power": y, "is_valid": True})
    events = pd.DataFrame(
        [{"event_id": 0, "source_name": "x", "event_type": "disappearance", "predicted_event_time": event_time, "frequency_band": 1, "antenna": "rv1_coarse"}]
    )
    det = run_stepfit_detections(clean, events, StepFitConfig(window_seconds=600, min_samples_per_side=2, timing_grid_seconds=(0.0,)))
    assert det.iloc[0]["n_used"] == 19
    assert det.iloc[0]["amplitude"] > 5.0
