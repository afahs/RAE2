"""The stack jackknife must report a standard error, not the leave-one-out spread."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rylevonberg.stacking import stack_profiles
from rylevonberg.util import jackknife_standard_error


def test_jackknife_of_mean_equals_standard_error_of_mean() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=50)
    loo = [(x.sum() - xi) / (x.size - 1) for xi in x]
    assert np.isclose(jackknife_standard_error(loo), x.std(ddof=1) / np.sqrt(x.size))


def test_stack_jackknife_matches_bootstrap_scale() -> None:
    rng = np.random.default_rng(7)
    t = np.linspace(-600, 600, 41)
    template = np.where(t < 0, 0.5, -0.5)
    rows = []
    for event_id in range(200):
        amp = 1.0 + rng.normal(scale=2.0)  # event-to-event scatter dominates
        for tv, mv in zip(t, template):
            rows.append({"source_name": "s", "frequency_band": 1, "antenna": "rv1_coarse",
                         "event_id": event_id, "t_bin_sec": tv,
                         "profile_value": amp * mv + rng.normal(scale=0.1), "template": mv})
    _, summary = stack_profiles(pd.DataFrame(rows), n_bootstrap=200)
    row = summary.iloc[0]
    assert 0.7 < row["jackknife_std"] / row["bootstrap_std"] < 1.4
