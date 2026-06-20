"""Lightweight Bayesian models for occultation contrast diagnostics.

This module intentionally avoids heavyweight MCMC dependencies.  It uses a
Gaussian prior on a linear contrast model, then samples the resulting
Gaussian posterior approximation.  The target quantity is the occultation
contrast that is already signed so that positive means source-like behavior.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rylevonberg.util import robust_sigma


@dataclass(frozen=True)
class BayesianContrastFit:
    """Posterior approximation for a centered contrast regression."""

    coefficient_names: list[str]
    posterior_mean: np.ndarray
    posterior_covariance: np.ndarray
    residual_sigma: float
    n_observations: int
    n_parameters: int


def centered_design_matrix(
    frame: pd.DataFrame,
    categorical_columns: list[str] | tuple[str, ...] = (),
) -> tuple[np.ndarray, list[str]]:
    """Build an intercept plus sample-centered categorical deviation columns.

    For each category level, the column is 1(level present) minus that level's
    sample fraction.  This makes the intercept the sample-weighted global
    contrast amplitude after accounting for category deviations.
    """

    pieces = [np.ones(len(frame), dtype=float)]
    names = ["global_amplitude"]
    for column in categorical_columns:
        values = frame[column].fillna("missing").astype(str)
        levels = sorted(values.unique())
        fractions = values.value_counts(normalize=True).to_dict()
        for level in levels:
            pieces.append((values.eq(level).astype(float) - float(fractions[level])).to_numpy(dtype=float))
            names.append(f"{column}={level}")
    if len(pieces) == 1:
        return pieces[0][:, None], names
    return np.column_stack(pieces), names


def fit_bayesian_contrast_model(
    frame: pd.DataFrame,
    response_column: str = "contrast",
    categorical_columns: list[str] | tuple[str, ...] = (),
    prior_scale: float = 5.0,
    intercept_prior_scale: float = 20.0,
) -> BayesianContrastFit:
    """Fit a Gaussian-prior linear model to event-level contrasts.

    The posterior covariance uses a robust residual scale estimate.  This is a
    pragmatic empirical-Bayes diagnostic, not a substitute for a full physical
    radiometer likelihood.
    """

    work = frame.copy()
    y = pd.to_numeric(work[response_column], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(y)
    for column in categorical_columns:
        keep &= work[column].notna().to_numpy()
    work = work.loc[keep].reset_index(drop=True)
    y = y[keep]
    if y.size == 0:
        raise ValueError("no finite contrast values available for Bayesian contrast model")

    X, names = centered_design_matrix(work, categorical_columns)
    prior_precision = np.eye(X.shape[1], dtype=float) / float(prior_scale) ** 2
    prior_precision[0, 0] = 1.0 / float(intercept_prior_scale) ** 2

    precision = X.T @ X + prior_precision
    rhs = X.T @ y
    beta = np.linalg.solve(precision, rhs)
    residual = y - X @ beta
    sigma = robust_sigma(residual)
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.std(residual, ddof=1)) if residual.size > 1 else 1.0
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0
    covariance = float(sigma) ** 2 * np.linalg.inv(precision)
    return BayesianContrastFit(
        coefficient_names=names,
        posterior_mean=beta,
        posterior_covariance=covariance,
        residual_sigma=float(sigma),
        n_observations=int(y.size),
        n_parameters=int(X.shape[1]),
    )


def draw_posterior(
    fit: BayesianContrastFit,
    n_draws: int = 20_000,
    seed: int = 20260518,
) -> pd.DataFrame:
    """Draw coefficient samples from the Gaussian posterior approximation."""

    rng = np.random.default_rng(int(seed))
    draws = rng.multivariate_normal(fit.posterior_mean, fit.posterior_covariance, size=int(n_draws))
    return pd.DataFrame(draws, columns=fit.coefficient_names)


def summarize_draws(draws: pd.Series | np.ndarray) -> dict[str, float]:
    """Return compact posterior summaries for one scalar quantity."""

    values = np.asarray(draws, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "posterior_mean": np.nan,
            "posterior_sd": np.nan,
            "ci_2p5": np.nan,
            "ci_97p5": np.nan,
            "p_gt_0": np.nan,
            "p_lt_0": np.nan,
        }
    return {
        "posterior_mean": float(np.mean(values)),
        "posterior_sd": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "ci_2p5": float(np.quantile(values, 0.025)),
        "ci_97p5": float(np.quantile(values, 0.975)),
        "p_gt_0": float(np.mean(values > 0.0)),
        "p_lt_0": float(np.mean(values < 0.0)),
    }


def summarize_coefficients(draws: pd.DataFrame) -> pd.DataFrame:
    """Summarize every sampled coefficient."""

    rows = []
    for column in draws.columns:
        rows.append({"quantity": column, **summarize_draws(draws[column])})
    return pd.DataFrame.from_records(rows)
