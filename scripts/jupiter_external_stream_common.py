"""Shared helpers for the three externally constrained Jupiter streams."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_jupiter_burst_gated_phase_occurrence import (
    DETECTORS,
    LABELS,
    detector_events,
)


COLORS = {
    "rv1_coarse": "#3568a8",
    "rv2_coarse": "#d4681f",
    "rv1_and_rv2": "#7a3e9d",
}


def detector_source(events: pd.DataFrame, detector: str, require_geometry: bool = False) -> pd.DataFrame:
    """Return complete, unocculted five-minute windows for one detector."""

    source = detector_events(events, detector)
    keep = source["valid_interval"].astype(bool) & ~source["jupiter_occulted"].astype(bool)
    if require_geometry:
        keep &= source["io_phase_deg"].notna() & source["jupiter_cml_system_iii_deg"].notna()
    return source.loc[keep].sort_values("time").reset_index(drop=True)


def band_gate(source: pd.DataFrame, bands: tuple[int, ...], quantile: float, mode: str = "any") -> np.ndarray:
    """Threshold daily channel percentiles in a fixed list of bands."""

    values = source[[f"band_{band}_percentile" for band in bands]].to_numpy(dtype=float)
    passed = values >= float(quantile)
    if mode == "any":
        return np.any(passed, axis=1)
    if mode == "all":
        return np.all(passed, axis=1)
    raise ValueError(f"Unknown gate mode: {mode}")


def daily_roll_plan(
    source: pd.DataFrame,
    n_permutations: int,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Create shared within-day circular-roll offsets."""

    groups = [group.index.to_numpy(dtype=int) for _, group in source.groupby("date", sort=True)]
    offsets = np.zeros((int(n_permutations), len(groups)), dtype=np.int32)
    for group_index, indices in enumerate(groups):
        if len(indices) > 1:
            offsets[:, group_index] = rng.integers(0, len(indices), size=int(n_permutations))
    return groups, offsets


def rolled_vectors(gate: np.ndarray, groups: list[np.ndarray], offsets: np.ndarray) -> np.ndarray:
    """Apply a shared within-day roll plan to a candidate vector."""

    gate = np.asarray(gate, dtype=np.int8)
    result = np.broadcast_to(gate, (offsets.shape[0], len(gate))).copy()
    for group_index, indices in enumerate(groups):
        if len(indices) <= 1:
            continue
        positions = (
            np.arange(len(indices), dtype=np.int32)[None, :]
            - offsets[:, group_index, None]
        ) % len(indices)
        result[:, indices] = gate[indices[positions]]
    return result


def rate_difference(gate: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Candidate rate in target exposure minus rate in complementary exposure."""

    gate = np.asarray(gate, dtype=float)
    target = np.asarray(target, dtype=bool)
    if gate.ndim == 1:
        return np.asarray(gate[target].mean() - gate[~target].mean())
    return gate[:, target].mean(axis=1) - gate[:, ~target].mean(axis=1)


def log_rate_ratio(gate: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Jeffreys-smoothed log candidate-rate ratio for target vs complement."""

    gate = np.asarray(gate, dtype=float)
    target = np.asarray(target, dtype=bool)
    n_target = int(target.sum())
    n_control = int((~target).sum())
    if n_target == 0 or n_control == 0:
        shape = gate.shape[:-1] if gate.ndim > 1 else ()
        return np.full(shape, np.nan, dtype=float)
    if gate.ndim == 1:
        count_target = float(gate[target].sum())
        count_control = float(gate[~target].sum())
    else:
        count_target = gate[:, target].sum(axis=1)
        count_control = gate[:, ~target].sum(axis=1)
    target_rate = (count_target + 0.5) / (n_target + 1.0)
    control_rate = (count_control + 0.5) / (n_control + 1.0)
    return np.log(target_rate / control_rate)


def upper_tail_p(observed: float, null: np.ndarray) -> float:
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    if not np.isfinite(observed) or len(null) == 0:
        return np.nan
    return float((1 + np.sum(null >= float(observed))) / (len(null) + 1))


def familywise_upper_p(observed: np.ndarray, null: np.ndarray) -> np.ndarray:
    """Max-statistic FWER p-values across peer detectors."""

    observed = np.asarray(observed, dtype=float)
    null = np.asarray(null, dtype=float)
    max_null = np.nanmax(null, axis=0)
    return np.asarray([upper_tail_p(value, max_null) for value in observed], dtype=float)


__all__ = [
    "COLORS",
    "DETECTORS",
    "LABELS",
    "band_gate",
    "daily_roll_plan",
    "detector_source",
    "familywise_upper_p",
    "log_rate_ratio",
    "rate_difference",
    "rolled_vectors",
    "upper_tail_p",
]
