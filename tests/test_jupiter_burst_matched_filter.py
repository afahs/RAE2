from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_jupiter_burst_matched_filter import (
    _component,
    literature_weight,
    pair_occultation_intervals,
    score_components,
)


def test_pair_occultation_intervals_rejects_unresolved_pair() -> None:
    events = pd.DataFrame(
        {
            "source_name": ["jupiter"] * 4,
            "event_id": [0, 1, 2, 3],
            "event_type": ["disappearance", "reappearance", "disappearance", "reappearance"],
            "predicted_event_time": pd.to_datetime(
                ["1975-01-01 00:00", "1975-01-01 00:02", "1975-02-01 00:00", "1975-02-01 00:08"]
            ),
        }
    )
    paired = pair_occultation_intervals(events, "jupiter", min_duration_s=240, max_duration_s=1800)
    assert len(paired) == 2
    assert paired["resolvable"].tolist() == [False, True]
    assert paired["duration_s"].tolist() == [120.0, 480.0]


def test_burst_matched_filter_prefers_visible_side_energy() -> None:
    rng = np.random.default_rng(42)
    epoch = pd.Timestamp("1975-01-01 00:00:00")
    seconds = np.arange(-900, 901, 30, dtype=float)
    times = epoch.value + (seconds * 1e9).astype(np.int64)
    baseline = 10.0 + 0.0001 * seconds
    noise = rng.normal(0.0, 0.08, size=len(seconds))
    visible = (seconds < -240) | (seconds > 240)
    bursts = np.zeros(len(seconds), dtype=float)
    burst_indices = np.flatnonzero(visible)[::5]
    bursts[burst_indices] = 0.9
    payload = {"time_ns": times, "log_power": baseline + noise + bursts}
    component = _component(
        payload,
        epoch - pd.Timedelta(seconds=240),
        epoch + pd.Timedelta(seconds=240),
        padding_s=600,
    )
    assert component is not None
    linear = score_components([component], None)
    gated = score_components([component], 0.75)
    assert linear is not None and gated is not None
    assert linear["score"] > 0
    assert gated["score"] > 0
    assert gated["contrast"] > 0


def test_literature_weight_is_soft_and_bounded() -> None:
    peak = literature_weight(100.0, 50.0)
    trough = literature_weight(150.0, 120.0)
    assert 0.35 <= trough < peak <= 2.5
