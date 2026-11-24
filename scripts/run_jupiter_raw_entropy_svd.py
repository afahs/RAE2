#!/usr/bin/env python
"""Event-level entropy-SVD trial on raw RAE-2 Ryle-Vonberg powers.

This adapts the frequency whitening, low-spectral-entropy SVD reconstruction,
SNR diagnostic, and residual-correlation check from Wille et al. (2026) to
lunar-occultation windows.  The two Ryle-Vonberg receivers are processed
independently.  Jupiter is tested against fixed +/-10 and +/-20 minute
alignments with an exact lunar-block null; matched Earth events are the
positive control.

The input is the retained cleaned mission stream, but the analyzed quantity is
the minimally transformed raw receiver ``power`` column (finite positive
samples, log transformed for the instrument's very large dynamic range).
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
LOCAL_DEPS = WORKSPACE / ".deps"
if LOCAL_DEPS.is_dir():
    sys.path.insert(0, str(LOCAL_DEPS))
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


PAPER_URL = "https://arxiv.org/abs/2608.25075"
PAPER_ENTROPY_LIMIT_NORMALIZED = 3.0 / math.log(90.0)
ENTROPY_LIMITS = (0.55, PAPER_ENTROPY_LIMIT_NORMALIZED, 0.78)
ANTENNAS = ("rv1_coarse", "rv2_coarse")
FREQUENCIES_MHZ = np.array([0.45, 0.70, 0.90, 1.31, 2.20, 3.93, 4.70, 6.55, 9.18])
CONTROL_SHIFTS_S = (-1200.0, -600.0, 600.0, 1200.0)
ALL_SHIFTS_S = (-1200.0, -600.0, 0.0, 600.0, 1200.0)
VALIDITY_MODES = ("finite", "valid")
MISSION_EPOCH = pd.Timestamp("1974-11-01T00:00:00")
LUNAR_BLOCK_DAYS = 29.53059
EDGE_HALF_WIDTH_S = 240.0
EDGE_GAP_S = 30.0
SCAN_LIMIT_S = 1200.0

DEFAULT_CLEANED = ROOT / "runtime_inputs/occultogram_v1/cleaned_timeseries.parquet"
DEFAULT_EVENTS = ROOT / "runtime_inputs/occultogram_v1/planet_events_science_baseline.parquet"
DEFAULT_OUTPUT = Path("/pscratch/sd/a/afahs/RAE2/jupiter_raw_entropy_svd_v1")


@dataclass(frozen=True)
class ReceiverSeries:
    time_ns: np.ndarray
    power: np.ndarray
    is_valid: np.ndarray


@dataclass(frozen=True)
class SVDResult:
    reconstruction: np.ndarray
    residual: np.ndarray
    singular_values: np.ndarray
    entropy_nats: np.ndarray
    entropy_normalized: np.ndarray
    retained: np.ndarray


def _robust_scale(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return np.nan
    center = float(np.median(values))
    scale = float(1.4826 * np.median(np.abs(values - center)))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.std(values, ddof=1))
    return scale


def _sustained(mask: np.ndarray, minimum_run: int = 2) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    result = np.zeros_like(mask)
    start = 0
    while start < len(mask):
        if not mask[start]:
            start += 1
            continue
        stop = start + 1
        while stop < len(mask) and mask[stop]:
            stop += 1
        if stop - start >= minimum_run:
            result[start:stop] = True
        start = stop
    return result


def _interpolate_rows(matrix: np.ndarray, minimum_finite: int = 12) -> tuple[np.ndarray, float]:
    matrix = np.asarray(matrix, dtype=float)
    output = matrix.copy()
    x = np.arange(matrix.shape[1], dtype=float)
    coverage = float(np.isfinite(matrix).mean())
    required = min(int(minimum_finite), max(2, matrix.shape[1] // 2))
    for index, row in enumerate(matrix):
        finite = np.isfinite(row)
        if int(np.sum(finite)) < required:
            output[index] = np.nan
            continue
        output[index] = np.interp(x, x[finite], row[finite])
    return output, coverage


def whiten_frequency_rows(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Paper-style row whitening and deterministic broadband inpainting."""

    matrix = np.asarray(matrix, dtype=float)
    whitened = np.full_like(matrix, np.nan)
    for row_index, row in enumerate(matrix):
        finite = np.isfinite(row)
        if not np.any(finite):
            continue
        center = float(np.median(row[finite]))
        residual = row - center
        positive = residual[finite & (residual > 0.0)]
        scale = float(np.percentile(positive, 75.0)) if positive.size >= 3 else np.nan
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = _robust_scale(residual[finite])
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = 1.0
        whitened[row_index, finite] = residual[finite] / scale

    column_mean = np.nanmean(whitened, axis=0)
    column_std = np.nanstd(whitened, axis=0)
    std_center = float(np.nanmedian(column_std))
    std_scale = _robust_scale(column_std)
    mean_center = float(np.nanmedian(column_mean))
    mean_scale = _robust_scale(column_mean)
    if not np.isfinite(std_scale) or not np.isfinite(mean_scale):
        flagged = np.zeros(matrix.shape[1], dtype=bool)
    else:
        high_variance = column_std > std_center + 6.0 * std_scale
        near_zero_mean = np.abs(column_mean - mean_center) <= max(1.5 * mean_scale, 0.25)
        flagged = _sustained(high_variance & near_zero_mean, minimum_run=2)

    if np.any(flagged):
        clean = ~flagged
        for column in np.flatnonzero(flagged):
            distance = np.abs(np.arange(matrix.shape[1]) - column)
            neighbors = clean & (distance <= 3)
            if not np.any(neighbors):
                neighbors = clean
            for row_index in range(whitened.shape[0]):
                values = whitened[row_index, neighbors]
                values = values[np.isfinite(values)]
                if values.size:
                    whitened[row_index, column] = float(np.median(values))
    return whitened, flagged


def entropy_svd(
    matrix: np.ndarray,
    entropy_limit_normalized: float = PAPER_ENTROPY_LIMIT_NORMALIZED,
) -> SVDResult:
    """Retain modes below a spectral-entropy cutoff normalized by log(N)."""

    matrix = np.asarray(matrix, dtype=float)
    clean = np.where(np.isfinite(matrix), matrix, 0.0)
    u, singular_values, vt = np.linalg.svd(clean, full_matrices=False)
    probabilities = np.square(np.abs(u))
    probabilities /= np.maximum(probabilities.sum(axis=0, keepdims=True), 1e-15)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(probabilities > 0.0, probabilities * np.log(probabilities), 0.0)
    entropy_nats = -np.sum(terms, axis=0)
    entropy_normalized = entropy_nats / math.log(max(clean.shape[0], 2))
    retained = entropy_normalized < float(entropy_limit_normalized)
    if np.any(retained):
        reconstruction = (u[:, retained] * singular_values[retained]) @ vt[retained]
    else:
        reconstruction = np.zeros_like(clean)
    return SVDResult(
        reconstruction=reconstruction,
        residual=clean - reconstruction,
        singular_values=singular_values,
        entropy_nats=entropy_nats,
        entropy_normalized=entropy_normalized,
        retained=retained,
    )


def lag1_autocorrelation(matrix: np.ndarray) -> float:
    matrix = np.asarray(matrix, dtype=float)
    correlations: list[float] = []
    for row_lag in (-1, 0, 1):
        for column_lag in (-1, 0, 1):
            if row_lag == 0 and column_lag == 0:
                continue
            row_a = slice(max(0, row_lag), matrix.shape[0] + min(0, row_lag))
            row_b = slice(max(0, -row_lag), matrix.shape[0] + min(0, -row_lag))
            col_a = slice(max(0, column_lag), matrix.shape[1] + min(0, column_lag))
            col_b = slice(max(0, -column_lag), matrix.shape[1] + min(0, -column_lag))
            a = matrix[row_a, col_a].ravel()
            b = matrix[row_b, col_b].ravel()
            finite = np.isfinite(a) & np.isfinite(b)
            a = a[finite]
            b = b[finite]
            if a.size < 3:
                continue
            a = a - np.mean(a)
            b = b - np.mean(b)
            denominator = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
            if denominator > 0.0:
                correlations.append(float(np.sum(a * b) / denominator))
    return float(np.mean(correlations)) if correlations else np.nan


def empirical_snr_gain_db(raw: np.ndarray, reconstruction: np.ndarray) -> float:
    def snr(matrix: np.ndarray) -> np.ndarray:
        squared = np.square(np.asarray(matrix, dtype=float))
        baseline = np.percentile(squared, 10.0, axis=1)
        positive = baseline[np.isfinite(baseline) & (baseline > 1e-12)]
        fallback = float(np.median(positive)) if positive.size else 1e-12
        baseline = np.where(baseline > 1e-12, baseline, fallback)
        return 10.0 * np.log10(np.maximum(squared, 1e-12) / baseline[:, None])

    raw_snr = snr(raw)
    reconstruction_snr = snr(reconstruction)
    active = np.isfinite(raw_snr) & np.isfinite(reconstruction_snr) & (raw_snr > 3.0)
    if not np.any(active):
        return np.nan
    return float(np.mean(reconstruction_snr[active]) - np.mean(raw_snr[active]))


def edge_spectrum(
    matrix: np.ndarray,
    times_s: np.ndarray,
    center_s: float = 0.0,
    half_width_s: float = EDGE_HALF_WIDTH_S,
    gap_s: float = EDGE_GAP_S,
) -> np.ndarray:
    relative = np.asarray(times_s, dtype=float) - float(center_s)
    before = (relative >= -half_width_s) & (relative < -gap_s)
    after = (relative > gap_s) & (relative <= half_width_s)
    if not np.any(before) or not np.any(after):
        return np.full(np.asarray(matrix).shape[0], np.nan)
    return np.nanmean(matrix[:, before], axis=1) - np.nanmean(matrix[:, after], axis=1)


def signed_sparse_edge_score(edge_values: np.ndarray) -> float:
    edge_values = np.asarray(edge_values, dtype=float)
    finite = np.isfinite(edge_values)
    if not np.any(finite):
        return np.nan
    values = edge_values[finite]
    positive = float(np.sqrt(np.mean(np.square(np.maximum(values, 0.0)))))
    negative = float(np.sqrt(np.mean(np.square(np.maximum(-values, 0.0)))))
    return positive - negative


def signed_sparse_edge_scores(edge_values: np.ndarray) -> np.ndarray:
    values = np.asarray(edge_values, dtype=float)
    positive = np.sqrt(np.nanmean(np.square(np.maximum(values, 0.0)), axis=-1))
    negative = np.sqrt(np.nanmean(np.square(np.maximum(-values, 0.0)), axis=-1))
    return positive - negative


def edge_scan(matrix: np.ndarray, times_s: np.ndarray) -> pd.DataFrame:
    cadence = float(np.median(np.diff(np.asarray(times_s, dtype=float))))
    centers = np.arange(-SCAN_LIMIT_S, SCAN_LIMIT_S + 0.5 * cadence, cadence)
    return pd.DataFrame(
        {
            "center_s": centers,
            "score": [
                signed_sparse_edge_score(edge_spectrum(matrix, times_s, center))
                for center in centers
            ],
        }
    )


def build_receiver_lookup(cleaned: pd.DataFrame) -> dict[tuple[str, int], ReceiverSeries]:
    required = {"time", "frequency_band", "antenna", "power", "is_valid"}
    missing = required - set(cleaned.columns)
    if missing:
        raise ValueError(f"cleaned receiver table is missing {sorted(missing)}")
    work = cleaned.loc[:, sorted(required)].copy()
    work["time"] = pd.to_datetime(work["time"], errors="coerce")
    work["power"] = pd.to_numeric(work["power"], errors="coerce")
    work = work[
        work["time"].notna()
        & work["antenna"].astype(str).isin(ANTENNAS)
        & work["frequency_band"].between(1, 9)
        & np.isfinite(work["power"])
        & work["power"].gt(0.0)
    ].copy()
    work["time_ns"] = work["time"].astype("int64")
    lookup: dict[tuple[str, int], ReceiverSeries] = {}
    for (antenna, band), group in work.groupby(["antenna", "frequency_band"], observed=True, sort=True):
        group = group.sort_values("time_ns")
        lookup[(str(antenna), int(band))] = ReceiverSeries(
            time_ns=group["time_ns"].to_numpy(dtype=np.int64),
            power=group["power"].to_numpy(dtype=float),
            is_valid=group["is_valid"].fillna(False).to_numpy(dtype=bool),
        )
    expected = {(antenna, band) for antenna in ANTENNAS for band in range(1, 10)}
    missing_series = expected - set(lookup)
    if missing_series:
        raise ValueError(f"receiver series are missing: {sorted(missing_series)}")
    return lookup


def extract_event_matrix(
    lookup: dict[tuple[str, int], ReceiverSeries],
    event_time: pd.Timestamp,
    event_type: str,
    antenna: str,
    shift_s: float,
    validity_mode: str,
    half_window_s: float,
    bin_seconds: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Bin raw receiver powers into a folded frequency x event-time matrix."""

    n_bins = int(round(2.0 * half_window_s / bin_seconds))
    centers = -half_window_s + (np.arange(n_bins, dtype=float) + 0.5) * bin_seconds
    matrix = np.full((9, n_bins), np.nan)
    center_ns = int(pd.Timestamp(event_time).value + round(float(shift_s) * 1e9))
    window_ns = int(round(half_window_s * 1e9))
    reverse = str(event_type).lower().startswith("reappear")

    for row_index, band in enumerate(range(1, 10)):
        series = lookup[(antenna, band)]
        left = int(np.searchsorted(series.time_ns, center_ns - window_ns, side="left"))
        right = int(np.searchsorted(series.time_ns, center_ns + window_ns, side="right"))
        if right <= left:
            continue
        relative_s = (series.time_ns[left:right] - center_ns) / 1e9
        if reverse:
            relative_s = -relative_s
        power = series.power[left:right]
        keep = np.isfinite(power) & (power > 0.0)
        if validity_mode == "valid":
            keep &= series.is_valid[left:right]
        elif validity_mode != "finite":
            raise ValueError(f"unknown validity mode: {validity_mode}")
        relative_s = relative_s[keep]
        values = np.log(power[keep])
        bin_index = np.floor((relative_s + half_window_s) / bin_seconds).astype(int)
        inside = (bin_index >= 0) & (bin_index < n_bins)
        bin_index = bin_index[inside]
        values = values[inside]
        for index in np.unique(bin_index):
            matrix[row_index, index] = float(np.median(values[bin_index == index]))

    interpolated, coverage = _interpolate_rows(matrix)
    return interpolated, centers, coverage


def _prepare_events(event_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"event_id", "source_name", "event_type", "predicted_event_time"}
    missing = required - set(event_table.columns)
    if missing:
        raise ValueError(f"event table is missing {sorted(missing)}")
    events = event_table[list(required)].copy()
    events["predicted_event_time"] = pd.to_datetime(events["predicted_event_time"], errors="coerce")
    events = events.dropna(subset=["predicted_event_time"])
    events = events.drop_duplicates(
        ["source_name", "event_id", "event_type", "predicted_event_time"]
    ).sort_values(["source_name", "predicted_event_time", "event_id"])

    jupiter = events[events["source_name"].astype(str).eq("jupiter")].copy()
    earth = events[events["source_name"].astype(str).eq("earth")].copy()
    if jupiter.empty or earth.empty:
        raise ValueError("both Jupiter and Earth events are required")

    available = earth.reset_index(drop=True)
    unused = np.ones(len(available), dtype=bool)
    selected_indices: list[int] = []
    earth_time_ns = available["predicted_event_time"].astype("int64").to_numpy()
    for target_ns in jupiter["predicted_event_time"].astype("int64").to_numpy():
        candidates = np.flatnonzero(unused)
        chosen = int(candidates[np.argmin(np.abs(earth_time_ns[candidates] - target_ns))])
        selected_indices.append(chosen)
        unused[chosen] = False
    earth_matched = available.iloc[selected_indices].copy()

    output: list[pd.DataFrame] = []
    for source_events in (jupiter, earth_matched):
        source_events = source_events.copy().reset_index(drop=True)
        delta_days = (
            source_events["predicted_event_time"] - MISSION_EPOCH
        ).dt.total_seconds() / 86400.0
        source_events["lunar_block"] = np.floor(delta_days / LUNAR_BLOCK_DAYS).astype(int)
        source_events["event_uid"] = [
            f"{row.source_name}:{row.event_id}:{pd.Timestamp(row.predicted_event_time).value}"
            for row in source_events.itertuples(index=False)
        ]
        output.append(source_events)
    return output[0], output[1]


def analyze_events(
    lookup: dict[tuple[str, int], ReceiverSeries],
    events: pd.DataFrame,
    half_window_s: float,
    bin_seconds: float,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    edge_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    matrix_records: list[dict[str, object]] = []

    for event in events.itertuples(index=False):
        for antenna in ANTENNAS:
            for shift_s in ALL_SHIFTS_S:
                for validity_mode in VALIDITY_MODES:
                    raw_log, times_s, coverage = extract_event_matrix(
                        lookup,
                        pd.Timestamp(event.predicted_event_time),
                        str(event.event_type),
                        antenna,
                        shift_s,
                        validity_mode,
                        half_window_s,
                        bin_seconds,
                    )
                    if np.sum(np.all(np.isfinite(raw_log), axis=1)) < 3:
                        continue
                    whitened, flagged = whiten_frequency_rows(raw_log)
                    whitened = np.where(np.isfinite(whitened), whitened, 0.0)
                    raw_edge = edge_spectrum(whitened, times_s)
                    for entropy_limit in ENTROPY_LIMITS:
                        result = entropy_svd(whitened, entropy_limit)
                        svd_edge = edge_spectrum(result.reconstruction, times_s)
                        energy_total = float(np.sum(np.square(result.singular_values)))
                        energy_retained = float(
                            np.sum(np.square(result.singular_values[result.retained]))
                        )
                        common = {
                            "source_name": str(event.source_name),
                            "event_uid": str(event.event_uid),
                            "event_id": int(event.event_id),
                            "event_type": str(event.event_type),
                            "predicted_event_time": pd.Timestamp(event.predicted_event_time),
                            "lunar_block": int(event.lunar_block),
                            "antenna": antenna,
                            "control_shift_s": float(shift_s),
                            "validity_mode": validity_mode,
                            "entropy_limit_normalized": float(entropy_limit),
                        }
                        event_rows.append(
                            {
                                **common,
                                "matrix_coverage_fraction": coverage,
                                "n_broadband_columns_masked": int(np.sum(flagged)),
                                "retained_rank": int(np.sum(result.retained)),
                                "retained_indices": ",".join(
                                    str(index + 1) for index in np.flatnonzero(result.retained)
                                ),
                                "retained_energy_fraction": (
                                    energy_retained / energy_total if energy_total > 0.0 else np.nan
                                ),
                                "snr_gain_db": empirical_snr_gain_db(
                                    whitened, result.reconstruction
                                ),
                                "reconstruction_lag1": lag1_autocorrelation(
                                    result.reconstruction
                                ),
                                "residual_lag1": lag1_autocorrelation(result.residual),
                                "raw_edge_score": signed_sparse_edge_score(raw_edge),
                                "svd_edge_score": signed_sparse_edge_score(svd_edge),
                            }
                        )
                        for band_index, (frequency, raw_value, svd_value) in enumerate(
                            zip(FREQUENCIES_MHZ, raw_edge, svd_edge, strict=True), start=1
                        ):
                            edge_rows.append(
                                {
                                    **common,
                                    "frequency_band": band_index,
                                    "frequency_mhz": float(frequency),
                                    "raw_edge": float(raw_value),
                                    "svd_edge": float(svd_value),
                                }
                            )
                        if (
                            validity_mode == "finite"
                            and np.isclose(entropy_limit, PAPER_ENTROPY_LIMIT_NORMALIZED)
                        ):
                            matrix_records.append(
                                {
                                    **common,
                                    "times_s": times_s,
                                    "whitened": whitened,
                                    "reconstruction": result.reconstruction,
                                    "residual": result.residual,
                                }
                            )
    return pd.DataFrame(edge_rows), pd.DataFrame(event_rows), matrix_records


def _block_edge_table(edge_rows: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "source_name",
        "validity_mode",
        "entropy_limit_normalized",
        "lunar_block",
        "antenna",
        "control_shift_s",
        "frequency_band",
        "frequency_mhz",
    ]
    return (
        edge_rows.groupby(keys, observed=True, sort=True)
        .agg(
            raw_edge=("raw_edge", "median"),
            svd_edge=("svd_edge", "median"),
            n_events=("event_uid", "nunique"),
        )
        .reset_index()
    )


def exact_block_null(
    block_edges: pd.DataFrame,
    source_name: str,
    validity_mode: str,
    entropy_limit: float,
    value_column: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    subset = block_edges[
        block_edges["source_name"].eq(source_name)
        & block_edges["validity_mode"].eq(validity_mode)
        & np.isclose(block_edges["entropy_limit_normalized"], entropy_limit)
    ].copy()
    pivot = subset.pivot_table(
        index="lunar_block",
        columns=["antenna", "control_shift_s", "frequency_band"],
        values=value_column,
        aggfunc="median",
    ).sort_index()
    required_columns = pd.MultiIndex.from_product(
        [ANTENNAS, ALL_SHIFTS_S, range(1, 10)],
        names=["antenna", "control_shift_s", "frequency_band"],
    )
    pivot = pivot.reindex(columns=required_columns).dropna(axis=0, how="any")
    if pivot.empty:
        raise ValueError(
            f"no complete lunar blocks for {source_name}/{validity_mode}/{entropy_limit:g}"
        )
    blocks = pivot.index.to_numpy(dtype=int)
    n_blocks = len(blocks)
    values = pivot.to_numpy(dtype=float).reshape(n_blocks, 2, 5, 9)
    target_edges = values[:, :, 2, :]
    control_edges = values[:, :, [0, 1, 3, 4], :]
    target_vectors = np.nanmedian(target_edges, axis=0)
    target_scores = signed_sparse_edge_scores(target_vectors)
    fixed_control_vectors = np.nanmedian(control_edges, axis=0)
    fixed_control_scores = signed_sparse_edge_scores(fixed_control_vectors)

    combinations = np.asarray(
        list(itertools.product(range(4), repeat=n_blocks)), dtype=np.uint8
    )
    null_scores = np.empty((len(combinations), 2), dtype=float)
    block_index = np.arange(n_blocks)[None, :]
    for antenna_index in range(2):
        chosen = control_edges[
            block_index,
            antenna_index,
            combinations,
            :,
        ]
        combined = np.nanmedian(chosen, axis=1)
        null_scores[:, antenna_index] = signed_sparse_edge_scores(combined)

    p_values = (1.0 + np.sum(null_scores >= target_scores[None, :], axis=0)) / (
        1.0 + len(null_scores)
    )
    target_any = float(np.max(target_scores))
    null_any = np.max(null_scores, axis=1)
    any_p = float((1.0 + np.sum(null_any >= target_any)) / (1.0 + len(null_any)))
    target_replication = float(np.min(target_scores))
    null_replication = np.min(null_scores, axis=1)
    replication_p = float(
        (1.0 + np.sum(null_replication >= target_replication))
        / (1.0 + len(null_replication))
    )
    target_ranks = 1 + np.sum(
        fixed_control_scores > target_scores[:, None], axis=1
    )

    loo_margins = np.full((n_blocks, 2), np.nan)
    if n_blocks >= 2:
        for omitted in range(n_blocks):
            keep = np.arange(n_blocks) != omitted
            target_loo = signed_sparse_edge_scores(np.nanmedian(target_edges[keep], axis=0))
            control_loo = signed_sparse_edge_scores(
                np.nanmedian(control_edges[keep], axis=0)
            )
            loo_margins[omitted] = target_loo - np.max(control_loo, axis=1)

    block_scores = signed_sparse_edge_scores(target_edges)
    summary: dict[str, object] = {
        "source_name": source_name,
        "validity_mode": validity_mode,
        "entropy_limit_normalized": float(entropy_limit),
        "method": value_column,
        "n_lunar_blocks": int(n_blocks),
        "n_exact_null_combinations": int(len(combinations)),
        "target_score_rv1": float(target_scores[0]),
        "target_score_rv2": float(target_scores[1]),
        "empirical_p_rv1": float(p_values[0]),
        "empirical_p_rv2": float(p_values[1]),
        "target_rank_rv1_of_5": int(target_ranks[0]),
        "target_rank_rv2_of_5": int(target_ranks[1]),
        "familywise_any_antenna_p": any_p,
        "replication_score": target_replication,
        "replication_p": replication_p,
        "correct_sign_block_fraction_rv1": float(np.mean(block_scores[:, 0] > 0.0)),
        "correct_sign_block_fraction_rv2": float(np.mean(block_scores[:, 1] > 0.0)),
        "minimum_leave_one_block_margin_rv1": float(np.nanmin(loo_margins[:, 0])),
        "minimum_leave_one_block_margin_rv2": float(np.nanmin(loo_margins[:, 1])),
        "null_q95_rv1": float(np.quantile(null_scores[:, 0], 0.95)),
        "null_q95_rv2": float(np.quantile(null_scores[:, 1], 0.95)),
        "null_q95_replication": float(np.quantile(null_replication, 0.95)),
    }
    null_frame = pd.DataFrame(
        {
            "source_name": source_name,
            "validity_mode": validity_mode,
            "entropy_limit_normalized": float(entropy_limit),
            "method": value_column,
            "combination_index": np.arange(len(combinations), dtype=int),
            "null_score_rv1": null_scores[:, 0],
            "null_score_rv2": null_scores[:, 1],
            "null_score_any_antenna": null_any,
            "null_score_replication": null_replication,
        }
    )
    return summary, null_frame


def summarize_statistics(
    block_edges: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, object]] = []
    null_frames: list[pd.DataFrame] = []
    for source_name in ("earth", "jupiter"):
        for validity_mode in VALIDITY_MODES:
            for entropy_limit in ENTROPY_LIMITS:
                for method in ("raw_edge", "svd_edge"):
                    summary, null_frame = exact_block_null(
                        block_edges,
                        source_name,
                        validity_mode,
                        entropy_limit,
                        method,
                    )
                    summaries.append(summary)
                    null_frames.append(null_frame)
    return pd.DataFrame(summaries), pd.concat(null_frames, ignore_index=True)


def aggregate_matrices(
    matrix_records: list[dict[str, object]],
) -> dict[tuple[str, str, float], dict[str, object]]:
    output: dict[tuple[str, str, float], dict[str, object]] = {}
    records = pd.DataFrame(
        [
            {
                "source_name": row["source_name"],
                "antenna": row["antenna"],
                "control_shift_s": row["control_shift_s"],
                "lunar_block": row["lunar_block"],
                "record_index": index,
            }
            for index, row in enumerate(matrix_records)
        ]
    )
    for (source, antenna, shift), group in records.groupby(
        ["source_name", "antenna", "control_shift_s"], observed=True, sort=True
    ):
        block_matrices: dict[str, list[np.ndarray]] = {
            "whitened": [],
            "reconstruction": [],
            "residual": [],
        }
        for _, block_group in group.groupby("lunar_block", observed=True, sort=True):
            indices = block_group["record_index"].to_numpy(dtype=int)
            for name in block_matrices:
                block_matrices[name].append(
                    np.nanmedian(
                        np.stack([np.asarray(matrix_records[index][name]) for index in indices]),
                        axis=0,
                    )
                )
        item = {
            name: np.nanmedian(np.stack(matrices), axis=0)
            for name, matrices in block_matrices.items()
        }
        first_record = matrix_records[int(group["record_index"].iloc[0])]
        item["times_s"] = np.asarray(first_record["times_s"], dtype=float)
        item["n_blocks"] = len(block_matrices["whitened"])
        item["scan"] = edge_scan(item["reconstruction"], item["times_s"])
        output[(str(source), str(antenna), float(shift))] = item
    return output


def add_localization_metrics(
    statistics: pd.DataFrame,
    aggregate: dict[tuple[str, str, float], dict[str, object]],
) -> pd.DataFrame:
    output = statistics.copy()
    for source in ("earth", "jupiter"):
        for antenna_index, antenna in enumerate(ANTENNAS, start=1):
            item = aggregate[(source, antenna, 0.0)]
            scan: pd.DataFrame = item["scan"]  # type: ignore[assignment]
            peak = scan.iloc[int(np.nanargmax(scan["score"].to_numpy(dtype=float)))]
            mask = (
                output["source_name"].eq(source)
                & output["validity_mode"].eq("finite")
                & np.isclose(
                    output["entropy_limit_normalized"], PAPER_ENTROPY_LIMIT_NORMALIZED
                )
                & output["method"].eq("svd_edge")
            )
            output.loc[mask, f"peak_offset_s_rv{antenna_index}"] = float(peak["center_s"])
            output.loc[mask, f"peak_scan_score_rv{antenna_index}"] = float(peak["score"])
    return output


def _primary_row(statistics: pd.DataFrame, source: str) -> pd.Series:
    subset = statistics[
        statistics["source_name"].eq(source)
        & statistics["validity_mode"].eq("finite")
        & np.isclose(statistics["entropy_limit_normalized"], PAPER_ENTROPY_LIMIT_NORMALIZED)
        & statistics["method"].eq("svd_edge")
    ]
    if len(subset) != 1:
        raise ValueError(f"expected one primary row for {source}, found {len(subset)}")
    return subset.iloc[0]


def determine_verdict(statistics: pd.DataFrame) -> tuple[str, list[str]]:
    earth = _primary_row(statistics, "earth")
    jupiter = _primary_row(statistics, "jupiter")
    reasons: list[str] = []
    if float(earth["replication_p"]) > 0.05:
        reasons.append("the matched Earth positive control does not pass the two-receiver replication null")
    if float(jupiter["replication_p"]) > 0.05:
        reasons.append("Jupiter does not pass the exact lunar-block replication null at the paper-scaled cutoff")
    if int(jupiter["target_rank_rv1_of_5"]) != 1 or int(jupiter["target_rank_rv2_of_5"]) != 1:
        reasons.append("the predicted Jupiter alignment is not strongest among all five alignments in both receivers")
    for index in (1, 2):
        peak = float(jupiter.get(f"peak_offset_s_rv{index}", np.nan))
        if not np.isfinite(peak) or abs(peak) > 120.0:
            reasons.append(f"RV{index}'s strongest aggregate Jupiter edge is not within two minutes of zero")
    if (
        float(jupiter["minimum_leave_one_block_margin_rv1"]) <= 0.0
        or float(jupiter["minimum_leave_one_block_margin_rv2"]) <= 0.0
    ):
        reasons.append("the Jupiter advantage over fixed-time controls is not stable to every leave-one-lunar-block omission")
    sensitivity = statistics[
        statistics["source_name"].eq("jupiter")
        & statistics["validity_mode"].eq("finite")
        & statistics["method"].eq("svd_edge")
    ]
    if int(np.sum(sensitivity["replication_p"].to_numpy(dtype=float) <= 0.05)) < 2:
        reasons.append("the result is not significant at two of the three predeclared entropy cutoffs")
    valid_sensitivity = statistics[
        statistics["source_name"].eq("jupiter")
        & statistics["validity_mode"].eq("valid")
        & np.isclose(statistics["entropy_limit_normalized"], PAPER_ENTROPY_LIMIT_NORMALIZED)
        & statistics["method"].eq("svd_edge")
    ].iloc[0]
    if float(valid_sensitivity["replication_p"]) > 0.10:
        reasons.append("the candidate does not survive the legacy-valid-sample sensitivity run")
    if reasons:
        return "not detected", reasons
    return (
        "candidate only",
        [
            "the event-level SVD tests pass, but RAE-2 lacks the paper's circular-polarization product and this run does not yet supply an independent Io/System-III occurrence validation"
        ],
    )


def plot_overview(
    aggregate: dict[tuple[str, str, float], dict[str, object]],
    block_edges: pd.DataFrame,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(2, 5, figsize=(18, 8), constrained_layout=True)
    for row, antenna in enumerate(ANTENNAS):
        target = aggregate[("jupiter", antenna, 0.0)]
        times = np.asarray(target["times_s"], dtype=float)
        color_limit = float(np.nanpercentile(np.abs(target["whitened"]), 98.0))
        for column, name in enumerate(("whitened", "reconstruction", "residual")):
            axis = axes[row, column]
            image = axis.imshow(
                np.asarray(target[name]),
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                extent=(times[0] / 60.0, times[-1] / 60.0, -0.5, 8.5),
                cmap="RdBu_r",
                vmin=-color_limit,
                vmax=color_limit,
            )
            axis.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
            axis.set_yticks(np.arange(9))
            axis.set_yticklabels([f"{value:g}" for value in FREQUENCIES_MHZ])
            axis.set_xlabel("minutes from predicted event")
            if column == 0:
                axis.set_ylabel(f"{antenna}\nfrequency (MHz)")
            axis.set_title(
                {
                    "whitened": "Block-equal whitened raw log power",
                    "reconstruction": "Entropy-SVD reconstruction",
                    "residual": "Residual",
                }[name]
            )
            figure.colorbar(image, ax=axis, shrink=0.75)

        edge_axis = axes[row, 3]
        selection = block_edges[
            block_edges["source_name"].eq("jupiter")
            & block_edges["validity_mode"].eq("finite")
            & np.isclose(
                block_edges["entropy_limit_normalized"], PAPER_ENTROPY_LIMIT_NORMALIZED
            )
            & block_edges["antenna"].eq(antenna)
        ]
        for shift in CONTROL_SHIFTS_S:
            control = selection[np.isclose(selection["control_shift_s"], shift)]
            vector = control.groupby("frequency_mhz", sort=True)["svd_edge"].median()
            frequency_index = np.arange(len(FREQUENCIES_MHZ))
            edge_axis.plot(
                frequency_index,
                vector.values,
                color="0.68",
                alpha=0.8,
                linewidth=1.0,
            )
        predicted = selection[np.isclose(selection["control_shift_s"], 0.0)]
        vector = predicted.groupby("frequency_mhz", sort=True)["svd_edge"].median()
        edge_axis.plot(frequency_index, vector.values, "o-", color="#d62728", linewidth=2.0, label="predicted")
        edge_axis.axhline(0.0, color="black", linewidth=0.8)
        edge_axis.set_xticks(frequency_index)
        edge_axis.set_xticklabels([f"{value:g}" for value in FREQUENCIES_MHZ], rotation=45)
        edge_axis.set_xlabel("frequency (MHz)")
        edge_axis.set_ylabel("high-before minus low-after")
        edge_axis.set_title("Block-median edge spectrum")
        edge_axis.legend(fontsize=8)

        scan_axis = axes[row, 4]
        scan: pd.DataFrame = target["scan"]  # type: ignore[assignment]
        scan_axis.plot(scan["center_s"] / 60.0, scan["score"], color="#d62728", linewidth=2.0)
        for shift in CONTROL_SHIFTS_S:
            control_scan: pd.DataFrame = aggregate[("jupiter", antenna, shift)]["scan"]  # type: ignore[assignment]
            scan_axis.plot(
                control_scan["center_s"] / 60.0,
                control_scan["score"],
                color="0.7",
                linewidth=1.0,
                alpha=0.75,
            )
        scan_axis.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
        scan_axis.axhline(0.0, color="black", linewidth=0.8)
        scan_axis.set_xlabel("tested edge center (minutes)")
        scan_axis.set_ylabel("signed sparse-edge score")
        scan_axis.set_title("Edge localization")

    figure.suptitle(
        "Jupiter in raw RAE-2 Ryle-Vonberg powers: paper-style whitening and entropy-SVD\n"
        "Individual event matrices, then equal weight per 29.53-day lunar block",
        fontsize=15,
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_nulls(
    statistics: pd.DataFrame,
    null_scores: pd.DataFrame,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for row, source in enumerate(("earth", "jupiter")):
        primary = _primary_row(statistics, source)
        null = null_scores[
            null_scores["source_name"].eq(source)
            & null_scores["validity_mode"].eq("finite")
            & np.isclose(
                null_scores["entropy_limit_normalized"], PAPER_ENTROPY_LIMIT_NORMALIZED
            )
            & null_scores["method"].eq("svd_edge")
        ]
        for column, (null_column, target_column, label) in enumerate(
            (
                ("null_score_rv1", "target_score_rv1", "RV1"),
                ("null_score_rv2", "target_score_rv2", "RV2"),
                ("null_score_replication", "replication_score", "both-receiver replication"),
            )
        ):
            axis = axes[row, column]
            axis.hist(null[null_column], bins=45, color="0.72", density=True)
            axis.axvline(float(primary[target_column]), color="#d62728", linewidth=2.0, label="predicted")
            p_value = (
                float(primary[f"empirical_p_rv{column + 1}"])
                if column < 2
                else float(primary["replication_p"])
            )
            axis.set_title(f"{source.title()} - {label}\np={p_value:.4g}")
            axis.set_xlabel("signed edge statistic")
            axis.set_ylabel("exact-null density")
            axis.legend(fontsize=8)
    figure.suptitle("Exact lunar-block null from +/-10 and +/-20 minute controls", fontsize=15)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_event_scores(event_summaries: pd.DataFrame, output_path: Path) -> None:
    selection = event_summaries[
        event_summaries["source_name"].eq("jupiter")
        & event_summaries["validity_mode"].eq("finite")
        & np.isclose(
            event_summaries["entropy_limit_normalized"], PAPER_ENTROPY_LIMIT_NORMALIZED
        )
        & np.isclose(event_summaries["control_shift_s"], 0.0)
    ].copy()
    figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    for axis, antenna, color in zip(axes, ANTENNAS, ("#1f77b4", "#ff7f0e"), strict=True):
        subset = selection[selection["antenna"].eq(antenna)].sort_values("predicted_event_time")
        axis.scatter(
            subset["predicted_event_time"],
            subset["svd_edge_score"],
            c=subset["lunar_block"],
            cmap="viridis",
            s=38,
            edgecolor="black",
            linewidth=0.35,
        )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_yscale("symlog", linthresh=0.2)
        axis.set_ylabel("event SVD edge score")
        axis.set_title(antenna)
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("predicted Jupiter occultation time")
    figure.suptitle("Individual Jupiter event scores (colors identify lunar blocks)", fontsize=15)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def write_report(
    statistics: pd.DataFrame,
    event_summaries: pd.DataFrame,
    jupiter_events: pd.DataFrame,
    earth_events: pd.DataFrame,
    verdict: str,
    reasons: list[str],
    half_window_s: float,
    bin_seconds: float,
    output_path: Path,
) -> None:
    primary_rows = statistics[
        statistics["validity_mode"].eq("finite")
        & np.isclose(statistics["entropy_limit_normalized"], PAPER_ENTROPY_LIMIT_NORMALIZED)
        & statistics["method"].eq("svd_edge")
    ].set_index("source_name")
    event_primary = event_summaries[
        event_summaries["validity_mode"].eq("finite")
        & np.isclose(
            event_summaries["entropy_limit_normalized"], PAPER_ENTROPY_LIMIT_NORMALIZED
        )
        & np.isclose(event_summaries["control_shift_s"], 0.0)
    ]
    lines = [
        "# Raw RAE-2 Jupiter entropy-SVD trial",
        "",
        f"Paper adapted: [Wille et al. (2026)]({PAPER_URL}), *A Singular Value Decomposition Framework for Jovian Radio Emissions from Parker Solar Probe*.",
        "",
        f"## Outcome: {verdict}",
        "",
    ]
    lines.extend(f"- {reason}." for reason in reasons)
    lines.extend(
        [
            "",
            "## Data and design",
            "",
            f"- Raw receiver input: `{DEFAULT_CLEANED}` (`power` from `rv1_coarse` and `rv2_coarse`).",
            f"- Jupiter events: {len(jupiter_events)} transitions in {jupiter_events['lunar_block'].nunique()} 29.53-day blocks.",
            f"- Earth benchmark: {len(earth_events)} nearest-in-time, non-reused transitions in {earth_events['lunar_block'].nunique()} blocks.",
            f"- Each event is a 9-frequency by {int(round(2.0 * half_window_s / bin_seconds))}-bin matrix covering +/-{half_window_s / 60.0:g} minutes at {bin_seconds:g}-second cadence. Reappearances are time reversed so a source is high before and low after zero.",
            "- Raw positive receiver power is log transformed, median centered per frequency, and divided by the 75th percentile of its positive residuals. Sustained high-variance/near-zero-mean broadband columns are median inpainted.",
            f"- The paper's 3.0-nat cutoff for 90 channels is scaled to normalized entropy `< {PAPER_ENTROPY_LIMIT_NORMALIZED:.3f}` for RAE-2's nine bands. Cutoffs 0.55 and 0.78 are predeclared sensitivity checks.",
            "- Inference uses one median edge spectrum per lunar block. The exact null chooses one of four off-time alignments independently in every block. RV1 and RV2 are processed independently; the replication statistic is the weaker of the two receiver scores.",
            "",
            "## Primary results",
            "",
            "| Source | Blocks | RV1 score | RV1 p | RV1 rank/5 | RV2 score | RV2 p | RV2 rank/5 | Replication p | Any-receiver p | RV1 peak (min) | RV2 peak (min) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for source in ("earth", "jupiter"):
        row = primary_rows.loc[source]
        lines.append(
            f"| {source.title()} | {int(row['n_lunar_blocks'])} | "
            f"{float(row['target_score_rv1']):+.3f} | {float(row['empirical_p_rv1']):.4g} | {int(row['target_rank_rv1_of_5'])} | "
            f"{float(row['target_score_rv2']):+.3f} | {float(row['empirical_p_rv2']):.4g} | {int(row['target_rank_rv2_of_5'])} | "
            f"{float(row['replication_p']):.4g} | {float(row['familywise_any_antenna_p']):.4g} | "
            f"{float(row.get('peak_offset_s_rv1', np.nan)) / 60.0:+.1f} | {float(row.get('peak_offset_s_rv2', np.nan)) / 60.0:+.1f} |"
        )
    lines.extend(
        [
            "",
            "Paper-style event diagnostics at the predicted alignment:",
            "",
            "| Source | Antenna | Mean retained rank | Mean retained energy | Mean SNR change (dB) | Mean recon lag-1 | Mean residual lag-1 | Median coverage |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for source in ("earth", "jupiter"):
        for antenna in ANTENNAS:
            rows = event_primary[
                event_primary["source_name"].eq(source)
                & event_primary["antenna"].eq(antenna)
            ]
            lines.append(
                f"| {source.title()} | {antenna} | {rows['retained_rank'].mean():.2f} | "
                f"{rows['retained_energy_fraction'].mean():.3f} | {rows['snr_gain_db'].mean():+.2f} | "
                f"{rows['reconstruction_lag1'].mean():+.3f} | {rows['residual_lag1'].mean():+.3f} | "
                f"{rows['matrix_coverage_fraction'].median():.3f} |"
            )
    lines.extend(
        [
            "",
            "## Sensitivity summary",
            "",
            "| Source | Validity | Entropy cutoff | Method | Replication p | RV1 p | RV2 p |",
            "|---|---|---:|---|---:|---:|---:|",
        ]
    )
    for row in statistics.sort_values(
        ["source_name", "validity_mode", "method", "entropy_limit_normalized"]
    ).itertuples(index=False):
        lines.append(
            f"| {str(row.source_name).title()} | {row.validity_mode} | "
            f"{float(row.entropy_limit_normalized):.3f} | {row.method} | "
            f"{float(row.replication_p):.4g} | {float(row.empirical_p_rv1):.4g} | {float(row.empirical_p_rv2):.4g} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "SVD is a denoiser, not proof of source identity: smooth Galactic, receiver, or beam structure can also be low rank. The exact fixed-time null, zero localization, lunar-block stability, and Earth recovery are therefore the decision evidence; an SNR increase alone is not treated as a detection.",
            "",
            "This experiment cannot reproduce the paper's circular-polarization isolation because RAE-2 provides independent total-power receiver streams rather than the PSP imaginary cross-correlation/Stokes-V proxy. It also does not train an Eigenfaces classifier: there is no defensible labeled set of Jupiter-positive RAE-2 events. If a candidate survives here, the next independent check is event-level enrichment in the expected Io phase/System-III regions, with SPICE light-time correction and held-out event labels.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleaned", type=Path, default=DEFAULT_CLEANED)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--half-window-seconds", type=float, default=2400.0)
    parser.add_argument("--bin-seconds", type=float, default=60.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Reading raw receiver powers from {args.cleaned}", flush=True)
    cleaned = pd.read_parquet(
        args.cleaned,
        columns=["time", "frequency_band", "antenna", "power", "is_valid"],
    )
    lookup = build_receiver_lookup(cleaned)
    del cleaned
    event_table = pd.read_parquet(args.events)
    jupiter_events, earth_events = _prepare_events(event_table)
    del event_table
    selected_events = pd.concat([jupiter_events, earth_events], ignore_index=True)
    selected_events.to_parquet(args.output_dir / "selected_events.parquet", index=False)

    print(
        f"Analyzing {len(jupiter_events)} Jupiter and {len(earth_events)} matched Earth events",
        flush=True,
    )
    edge_rows, event_summaries, matrix_records = analyze_events(
        lookup,
        selected_events,
        float(args.half_window_seconds),
        float(args.bin_seconds),
    )
    edge_rows.to_parquet(args.output_dir / "event_frequency_edges.parquet", index=False)
    event_summaries.to_parquet(args.output_dir / "event_svd_diagnostics.parquet", index=False)
    block_edges = _block_edge_table(edge_rows)
    block_edges.to_parquet(args.output_dir / "lunar_block_edge_spectra.parquet", index=False)
    statistics, null_scores = summarize_statistics(block_edges)
    aggregate = aggregate_matrices(matrix_records)
    statistics = add_localization_metrics(statistics, aggregate)
    statistics.to_csv(args.output_dir / "exact_block_statistics.csv", index=False)
    null_scores.to_parquet(args.output_dir / "exact_block_null_scores.parquet", index=False)

    verdict, reasons = determine_verdict(statistics)
    run_summary = {
        "paper": PAPER_URL,
        "cleaned_receiver_input": str(args.cleaned),
        "event_input": str(args.events),
        "n_jupiter_events": int(len(jupiter_events)),
        "n_earth_matched_events": int(len(earth_events)),
        "jupiter_lunar_blocks": int(jupiter_events["lunar_block"].nunique()),
        "earth_lunar_blocks": int(earth_events["lunar_block"].nunique()),
        "matrix_shape": [9, int(round(2 * args.half_window_seconds / args.bin_seconds))],
        "paper_entropy_limit_normalized": PAPER_ENTROPY_LIMIT_NORMALIZED,
        "entropy_sensitivity_limits": list(ENTROPY_LIMITS),
        "verdict": verdict,
        "reasons": reasons,
    }
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_report(
        statistics,
        event_summaries,
        jupiter_events,
        earth_events,
        verdict,
        reasons,
        float(args.half_window_seconds),
        float(args.bin_seconds),
        args.output_dir / "JUPITER_RAW_ENTROPY_SVD_REPORT.md",
    )
    plot_overview(
        aggregate,
        block_edges,
        args.output_dir / "jupiter_raw_entropy_svd_overview.png",
    )
    plot_nulls(
        statistics,
        null_scores,
        args.output_dir / "earth_jupiter_exact_block_nulls.png",
    )
    plot_event_scores(
        event_summaries,
        args.output_dir / "jupiter_individual_event_scores.png",
    )
    print(json.dumps(run_summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
