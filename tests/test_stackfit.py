from __future__ import annotations

import numpy as np

from rylevonberg.stackfit import StackedStepFitConfig, finite_duration_event_template, fit_stacked_step, stacked_event_template


def test_stacked_step_fit_recovers_positive_disappearance_amplitude() -> None:
    rng = np.random.default_rng(123)
    t = np.arange(-300.0, 301.0, 30.0)
    template = stacked_event_template(t, "disappearance", timing_offset_sec=0.0)
    y = 0.2 + 2.5 * template + rng.normal(0.0, 0.05, size=t.size)
    fit = fit_stacked_step(
        t,
        y,
        "disappearance",
        uncertainty=np.full_like(t, 0.05, dtype=float),
        config=StackedStepFitConfig(baseline_order=0, timing_offsets_seconds=(0.0,)),
    )
    assert fit["amplitude"] > 2.0
    assert fit["stack_fit_snr"] > 10.0
    assert fit["delta_bic"] > 20.0


def test_stacked_step_fit_combines_event_type_signs() -> None:
    t_one = np.arange(-300.0, 301.0, 30.0)
    t = np.r_[t_one, t_one]
    event_types = np.array(["disappearance"] * t_one.size + ["reappearance"] * t_one.size)
    template = stacked_event_template(t, event_types, timing_offset_sec=30.0)
    y = 1.0 + 1.5 * template
    fit = fit_stacked_step(
        t,
        y,
        event_types,
        uncertainty=np.full_like(t, 0.1, dtype=float),
        config=StackedStepFitConfig(baseline_order=0, timing_offsets_seconds=(-30.0, 0.0, 30.0)),
    )
    assert abs(fit["best_timing_offset_s"] - 30.0) < 1e-9
    assert fit["amplitude"] > 1.0


def test_finite_duration_template_has_linear_transition() -> None:
    t = np.array([-120.0, -60.0, 0.0, 60.0, 120.0])
    dis = finite_duration_event_template(t, "disappearance", transition_duration_sec=120.0)
    rep = finite_duration_event_template(t, "reappearance", transition_duration_sec=120.0)
    assert np.allclose(dis, [0.5, 0.5, -0.0, -0.5, -0.5])
    assert np.allclose(rep, [-0.5, -0.5, 0.0, 0.5, 0.5])


def test_stacked_fit_recovers_finite_transition_duration() -> None:
    rng = np.random.default_rng(456)
    t = np.arange(-600.0, 601.0, 60.0)
    template = stacked_event_template(t, "reappearance", transition_duration_sec=300.0)
    y = 0.1 + 1.8 * template + rng.normal(0.0, 0.03, size=t.size)
    fit = fit_stacked_step(
        t,
        y,
        "reappearance",
        uncertainty=np.full_like(t, 0.03, dtype=float),
        config=StackedStepFitConfig(
            baseline_order=0,
            timing_offsets_seconds=(0.0,),
            transition_durations_seconds=(0.0, 120.0, 300.0, 600.0),
        ),
    )
    assert fit["best_transition_duration_s"] == 300.0
    assert fit["amplitude"] > 1.5
    assert fit["delta_bic"] > 20.0
