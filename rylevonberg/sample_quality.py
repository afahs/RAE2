"""Shared row-level validity masks for Ryle-Vonberg power samples."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _boolean_series(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.to_numpy(dtype=bool)
    return series.astype(str).str.lower().isin(["true", "1", "yes"]).to_numpy(dtype=bool)


def strict_power_mask(
    frame: pd.DataFrame,
    upper_clip_quantile: float = 0.9999,
    upper_clip_sigma: float = 12.0,
    use_existing_valid: bool = True,
) -> np.ndarray:
    """Return a conservative sample mask for raw radiometer power.

    The mask rejects samples that are non-finite, nonpositive, already flagged
    invalid, or at/above the channel's high-end clipping threshold.  It is
    intentionally row-level: event windows should drop bad points first, then
    allow existing minimum-sample cuts to reject windows that lose too many
    points.
    """
    if "power" not in frame.columns:
        return np.zeros(len(frame), dtype=bool)
    power = pd.to_numeric(frame["power"], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(power) & (power > 0.0)
    if use_existing_valid and "is_valid" in frame.columns:
        keep &= _boolean_series(frame["is_valid"])
    q = float(upper_clip_quantile)
    if 0.0 < q < 1.0:
        finite_positive = np.isfinite(power) & (power > 0.0)
        if np.count_nonzero(finite_positive) >= 10:
            positive_values = power[finite_positive]
            threshold = float(np.nanquantile(positive_values, q))
            median = float(np.nanmedian(positive_values))
            mad = float(np.nanmedian(np.abs(positive_values - median)))
            robust_scale = 1.4826 * mad
            extreme_threshold = (np.isfinite(robust_scale) and robust_scale > 0.0 and threshold > median + float(upper_clip_sigma) * robust_scale) or (
                np.isfinite(median) and median > 0.0 and threshold > 100.0 * median
            )
            if np.isfinite(threshold):
                if extreme_threshold:
                    keep &= power < threshold
    return keep


def add_strict_valid_column(
    frame: pd.DataFrame,
    column: str = "strict_is_valid",
    upper_clip_quantile: float = 0.9999,
    upper_clip_sigma: float = 12.0,
) -> pd.DataFrame:
    """Return a copy with a strict row-level validity column."""
    out = frame.copy()
    power = pd.to_numeric(out["power"], errors="coerce").to_numpy(dtype=float) if "power" in out.columns else np.full(len(out), np.nan)
    finite = np.isfinite(power)
    positive = finite & (power > 0.0)
    high_clip = np.zeros(len(out), dtype=bool)
    q = float(upper_clip_quantile)
    if 0.0 < q < 1.0 and np.count_nonzero(positive) >= 10:
        positive_values = power[positive]
        threshold = float(np.nanquantile(positive_values, q))
        median = float(np.nanmedian(positive_values))
        mad = float(np.nanmedian(np.abs(positive_values - median)))
        robust_scale = 1.4826 * mad
        extreme_threshold = (np.isfinite(robust_scale) and robust_scale > 0.0 and threshold > median + float(upper_clip_sigma) * robust_scale) or (
            np.isfinite(median) and median > 0.0 and threshold > 100.0 * median
        )
        if np.isfinite(threshold):
            high_clip = extreme_threshold & positive & (power >= threshold)
    out["strict_nonpositive_power"] = finite & (power <= 0.0)
    out["strict_high_clip_power"] = high_clip
    out[column] = strict_power_mask(out, upper_clip_quantile=upper_clip_quantile, upper_clip_sigma=upper_clip_sigma)
    return out
