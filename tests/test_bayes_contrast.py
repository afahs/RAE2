from __future__ import annotations

import numpy as np
import pandas as pd

from rylevonberg.bayes_contrast import (
    centered_design_matrix,
    draw_posterior,
    fit_bayesian_contrast_model,
    summarize_draws,
)


def test_centered_design_intercept_matches_mean_for_balanced_categories() -> None:
    frame = pd.DataFrame(
        {
            "contrast": [1.0, 2.0, 3.0, 4.0],
            "event_type": ["disappearance", "disappearance", "reappearance", "reappearance"],
        }
    )
    X, names = centered_design_matrix(frame, ["event_type"])
    intercept = names.index("global_amplitude")
    assert np.allclose(X[:, intercept], 1.0)
    assert np.allclose(X[:, 1:].mean(axis=0), 0.0)


def test_bayesian_contrast_model_recovers_positive_signal() -> None:
    rng = np.random.default_rng(123)
    frame = pd.DataFrame(
        {
            "contrast": 1.5 + rng.normal(0.0, 0.2, 60),
            "event_type": ["disappearance", "reappearance"] * 30,
            "month_block": [f"1975-{1 + i % 4:02d}" for i in range(60)],
        }
    )
    fit = fit_bayesian_contrast_model(frame, categorical_columns=["event_type", "month_block"])
    draws = draw_posterior(fit, n_draws=4000, seed=42)
    summary = summarize_draws(draws["global_amplitude"])
    assert summary["posterior_mean"] > 1.0
    assert summary["p_gt_0"] > 0.99


def test_bayesian_contrast_model_recovers_negative_signal() -> None:
    rng = np.random.default_rng(456)
    frame = pd.DataFrame(
        {
            "contrast": -0.8 + rng.normal(0.0, 0.15, 50),
            "event_type": ["disappearance", "reappearance"] * 25,
        }
    )
    fit = fit_bayesian_contrast_model(frame, categorical_columns=["event_type"])
    draws = draw_posterior(fit, n_draws=4000, seed=43)
    summary = summarize_draws(draws["global_amplitude"])
    assert summary["posterior_mean"] < -0.5
    assert summary["p_lt_0"] > 0.99
