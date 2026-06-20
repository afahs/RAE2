"""Fit occultation templates to already-stacked event profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .detection import baseline_matrix, event_template


@dataclass(frozen=True)
class StackedStepFitConfig:
    """Configuration for stack-first template fitting."""

    baseline_order: int = 0
    timing_offsets_seconds: tuple[float, ...] = (-180.0, -120.0, -90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0, 120.0, 180.0)
    transition_durations_seconds: tuple[float, ...] = (0.0,)
    min_bins: int = 6


def finite_duration_event_template(
    t_rel_sec: np.ndarray,
    event_type: str,
    timing_offset_sec: float = 0.0,
    transition_duration_sec: float = 0.0,
) -> np.ndarray:
    """Return a positive-source occultation template with finite transition time.

    The zero-duration case is the existing ideal step. For positive amplitudes,
    disappearance decreases from +1/2 to -1/2 and reappearance increases from
    -1/2 to +1/2. A nonzero transition duration linearly interpolates between
    those levels over that many seconds.
    """

    t = np.asarray(t_rel_sec, dtype=float) - float(timing_offset_sec)
    duration = float(transition_duration_sec)
    if duration <= 0:
        return event_template(np.asarray(t_rel_sec, dtype=float), event_type, timing_offset_sec=float(timing_offset_sec))
    ramp = np.clip(t / duration, -0.5, 0.5)
    if str(event_type).lower() == "reappearance":
        return ramp
    return -ramp


def stacked_event_template(
    t_rel_sec: np.ndarray,
    event_types: Iterable[str] | str,
    timing_offset_sec: float = 0.0,
    transition_duration_sec: float = 0.0,
) -> np.ndarray:
    """Return the expected positive-source occultation template for stack bins.

    A positive fitted amplitude means the stacked profile changes with the
    expected sign for a positive source: disappearance drops and reappearance
    rises. A negative amplitude means the stacked profile changes opposite to
    that convention.
    """
    t = np.asarray(t_rel_sec, dtype=float)
    if isinstance(event_types, str):
        return finite_duration_event_template(
            t,
            event_types,
            timing_offset_sec=float(timing_offset_sec),
            transition_duration_sec=float(transition_duration_sec),
        )
    event_arr = np.asarray(list(event_types), dtype=object)
    if event_arr.size != t.size:
        raise ValueError("event_types must be a scalar string or match t_rel_sec length")
    out = np.empty_like(t, dtype=float)
    for event_type in np.unique(event_arr.astype(str)):
        mask = event_arr.astype(str) == event_type
        out[mask] = finite_duration_event_template(
            t[mask],
            event_type,
            timing_offset_sec=float(timing_offset_sec),
            transition_duration_sec=float(transition_duration_sec),
        )
    return out


def _weighted_lstsq(X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    w = np.sqrt(np.asarray(weights, dtype=float))
    beta, *_ = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)
    resid = y - X @ beta
    wrss = float(np.sum(weights * resid**2))
    dof = max(int(len(y) - X.shape[1]), 1)
    sigma2 = wrss / dof
    cov = np.linalg.pinv((X.T * weights) @ X) * sigma2
    return beta, resid, wrss, cov


def _bic(wrss: float, n: int, n_params: int) -> float:
    wrss = max(float(wrss), np.finfo(float).tiny)
    n = max(int(n), 1)
    return float(n * np.log(wrss / n) + int(n_params) * np.log(n))


def fit_stacked_step(
    t_rel_sec: np.ndarray,
    values: np.ndarray,
    event_types: Iterable[str] | str,
    uncertainty: np.ndarray | None = None,
    config: StackedStepFitConfig | None = None,
) -> dict[str, float]:
    """Fit a baseline-only model and baseline-plus-template model to a stack.

    This is intentionally stack-first: individual events are only used to build
    the stack and uncertainty per time bin. The template amplitude and model
    comparison are computed on the stacked profile.
    """
    cfg = config or StackedStepFitConfig()
    t = np.asarray(t_rel_sec, dtype=float)
    y = np.asarray(values, dtype=float)
    if uncertainty is None:
        sigma = np.ones_like(y, dtype=float)
    else:
        sigma = np.asarray(uncertainty, dtype=float)
    mask = np.isfinite(t) & np.isfinite(y)
    mask &= np.isfinite(sigma) & (sigma > 0)
    t = t[mask]
    y = y[mask]
    sigma = sigma[mask]
    if not isinstance(event_types, str):
        event_arr = np.asarray(list(event_types), dtype=object)[mask]
    else:
        event_arr = event_types
    if y.size < int(cfg.min_bins):
        return {
            "n_bins": int(y.size),
            "amplitude": np.nan,
            "uncertainty": np.nan,
            "stack_fit_snr": np.nan,
            "best_timing_offset_s": np.nan,
            "delta_bic": np.nan,
            "baseline_order": int(cfg.baseline_order),
        }
    weights = 1.0 / np.maximum(sigma, np.nanmedian(sigma[sigma > 0]) if np.any(sigma > 0) else 1.0) ** 2
    weights = weights / np.nanmedian(weights)
    B = baseline_matrix(t, int(cfg.baseline_order))
    _, resid0, wrss0, _ = _weighted_lstsq(B, y, weights)
    bic0 = _bic(wrss0, y.size, B.shape[1])
    best: dict[str, float] | None = None
    for offset in cfg.timing_offsets_seconds:
        for transition_duration in cfg.transition_durations_seconds:
            tmpl = stacked_event_template(
                t,
                event_arr,
                timing_offset_sec=float(offset),
                transition_duration_sec=float(transition_duration),
            )
            X = np.column_stack([B, tmpl])
            beta, resid1, wrss1, cov = _weighted_lstsq(X, y, weights)
            amp = float(beta[-1])
            var = float(cov[-1, -1]) if cov.size else np.nan
            se = float(np.sqrt(var)) if np.isfinite(var) and var >= 0 else np.nan
            bic1 = _bic(wrss1, y.size, X.shape[1])
            row = {
                "n_bins": int(y.size),
                "amplitude": amp,
                "uncertainty": se,
                "stack_fit_snr": float(amp / se) if np.isfinite(se) and se > 0 else np.nan,
                "best_timing_offset_s": float(offset),
                "best_transition_duration_s": float(transition_duration),
                "delta_bic": float(bic0 - bic1),
                "baseline_order": int(cfg.baseline_order),
                "weighted_rss_baseline": float(wrss0),
                "weighted_rss_step": float(wrss1),
                "residual_rms": float(np.sqrt(np.nanmean(resid1**2))) if resid1.size else np.nan,
            }
            if best is None or row["delta_bic"] > best["delta_bic"]:
                best = row
    return best or {}
