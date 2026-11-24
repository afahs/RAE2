#!/usr/bin/env python
"""Phase-blind burst-gated Jupiter occurrence test for raw RAE-2 powers.

RAE-2 scans its nine bands sequentially: almost every five-minute window has
all nine bands, while one-minute windows generally do not.  This script keeps
the native one-minute single-band measurements, ranks log power within each
receiver/band/day, then makes complete five-minute event windows.  Gates are
defined without Io or System-III information by:

* per-channel daily tail threshold: 95, 97.5, 99, or 99.5 percentile;
* minimum extreme channels in the five-minute scan: one, two, or three;
* detector: RV1, RV2, or same-band/exact-window RV1 & RV2 coincidence.

Occurrence curves use exposure denominators and within-day circular-roll
nulls.  Because the best gate is selected from a grid, family-wise p-values
compare every observed gate with the maximum null statistic across the full
gate family for that detector and coordinate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_AGGREGATED = Path(
    "/pscratch/sd/a/afahs/RAE2/jupiter_phase_occurrence_svd_1min_v1/aggregated_raw_1min.parquet"
)
DEFAULT_GEOMETRY = Path(
    "/pscratch/sd/a/afahs/RAE2/jupiter_phase_occurrence_svd_v1/spice_geometry.parquet"
)
DEFAULT_OCCULTATIONS = Path(
    "/pscratch/sd/a/afahs/RAE2/jupiter_phase_occurrence_svd_v1/jupiter_occulted_intervals.csv"
)
DEFAULT_OUTPUT = Path("/pscratch/sd/a/afahs/RAE2/jupiter_burst_gated_phase_occurrence_v1")

ANTENNAS = ("rv1_coarse", "rv2_coarse")
DETECTORS = (*ANTENNAS, "rv1_and_rv2")
LABELS = {
    "rv1_coarse": "RV1 (upper V)",
    "rv2_coarse": "RV2 (lower V)",
    "rv1_and_rv2": "Coincident RV1 & RV2",
}
COLORS = {
    "rv1_coarse": "#3568a8",
    "rv2_coarse": "#d4681f",
    "rv1_and_rv2": "#7a3e9d",
}
TAIL_QUANTILES = (0.95, 0.975, 0.99, 0.995)
MIN_EXTREME_BANDS = (1, 2, 3)


def circular_smooth(counts: np.ndarray, exposure: np.ndarray) -> np.ndarray:
    smooth_counts = np.roll(counts, -1, axis=-1) + counts + np.roll(counts, 1, axis=-1)
    smooth_exposure = np.roll(exposure, -1, axis=-1) + exposure + np.roll(exposure, 1, axis=-1)
    return np.divide(
        smooth_counts,
        smooth_exposure,
        out=np.full_like(smooth_counts, np.nan, dtype=float),
        where=smooth_exposure > 0,
    )


def mark_occulted(times: pd.Series, intervals: pd.DataFrame) -> np.ndarray:
    result = np.zeros(len(times), dtype=bool)
    values = pd.to_datetime(times).to_numpy(dtype="datetime64[ns]")
    for row in intervals.itertuples(index=False):
        result |= (values >= np.datetime64(row.start)) & (values <= np.datetime64(row.stop))
    return result


def prepare_channel_percentiles(aggregated: pd.DataFrame, min_daily_samples: int) -> pd.DataFrame:
    source = aggregated.copy()
    source["time"] = pd.to_datetime(source["time"])
    source["date"] = source["time"].dt.floor("D")
    keys = ["antenna", "frequency_band", "date"]
    grouped = source.groupby(keys, sort=False, observed=True)["log_power"]
    source["daily_channel_samples"] = grouped.transform("size")
    source["daily_channel_percentile"] = grouped.rank(method="average", pct=True)
    source.loc[
        source["daily_channel_samples"] < int(min_daily_samples), "daily_channel_percentile"
    ] = np.nan
    source["event_time"] = source["time"].dt.floor("5min") + pd.Timedelta(minutes=2.5)
    return source


def build_event_table(
    channel_scores: pd.DataFrame,
    geometry: pd.DataFrame,
    intervals: pd.DataFrame,
    min_channels: int,
) -> pd.DataFrame:
    pivot = channel_scores.pivot_table(
        index=["antenna", "event_time"],
        columns="frequency_band",
        values="daily_channel_percentile",
        aggfunc="max",
    ).reindex(columns=np.arange(1, 10))
    pivot.columns = [f"band_{int(value)}_percentile" for value in pivot.columns]
    events = pivot.reset_index().rename(columns={"event_time": "time"})
    percentile_cols = [f"band_{band}_percentile" for band in range(1, 10)]
    events["n_channels"] = events[percentile_cols].notna().sum(axis=1)
    events["valid_interval"] = events["n_channels"] >= int(min_channels)
    events["date"] = events["time"].dt.floor("D")
    geometry = geometry.copy()
    geometry["time"] = pd.to_datetime(geometry["time"])
    events = events.merge(geometry, on="time", how="left", validate="many_to_one")
    events["jupiter_occulted"] = mark_occulted(events["time"], intervals)
    return events.sort_values(["antenna", "time"]).reset_index(drop=True)


def gate_vector(events: pd.DataFrame, tail_quantile: float, min_extreme_bands: int) -> np.ndarray:
    columns = [f"band_{band}_percentile" for band in range(1, 10)]
    values = events[columns].to_numpy(dtype=float)
    return np.sum(values >= float(tail_quantile), axis=1) >= int(min_extreme_bands)


def detector_events(events: pd.DataFrame, detector: str) -> pd.DataFrame:
    if detector in ANTENNAS:
        return events[events["antenna"].eq(detector)].copy()
    index_cols = [
        "time",
        "date",
        "jupiter_cml_system_iii_deg",
        "io_phase_deg",
        "jupiter_occulted",
    ]
    left = events[events["antenna"].eq(ANTENNAS[0])].set_index("time")
    right = events[events["antenna"].eq(ANTENNAS[1])].set_index("time")
    common = left.index.intersection(right.index).sort_values()
    base = left.loc[common, index_cols[1:]].reset_index()
    base["antenna"] = detector
    base["valid_interval"] = (
        left.loc[common, "valid_interval"].to_numpy(dtype=bool)
        & right.loc[common, "valid_interval"].to_numpy(dtype=bool)
    )
    for band in range(1, 10):
        column = f"band_{band}_percentile"
        base[column] = np.minimum(
            left.loc[common, column].to_numpy(dtype=float),
            right.loc[common, column].to_numpy(dtype=float),
        )
    base["n_channels"] = base[[f"band_{band}_percentile" for band in range(1, 10)]].notna().sum(axis=1)
    return base


def valid_detector_source(events: pd.DataFrame, detector: str) -> pd.DataFrame:
    source = detector_events(events, detector)
    return source[
        source["valid_interval"].astype(bool)
        & ~source["jupiter_occulted"].astype(bool)
        & source["io_phase_deg"].notna()
        & source["jupiter_cml_system_iii_deg"].notna()
    ].sort_values("time").reset_index(drop=True)


def make_roll_offsets(
    source: pd.DataFrame, n_permutations: int, rng: np.random.Generator
) -> tuple[list[np.ndarray], np.ndarray]:
    groups = [group.index.to_numpy(dtype=int) for _, group in source.groupby("date", sort=True)]
    offsets = np.zeros((int(n_permutations), len(groups)), dtype=np.int32)
    for group_index, indices in enumerate(groups):
        if len(indices) > 1:
            offsets[:, group_index] = rng.integers(0, len(indices), size=int(n_permutations))
    return groups, offsets


def rolled_null_counts(
    gate: np.ndarray,
    bin_index: np.ndarray,
    n_bins: int,
    groups: list[np.ndarray],
    offsets: np.ndarray,
) -> np.ndarray:
    result = np.zeros((offsets.shape[0], int(n_bins)), dtype=float)
    for permutation in range(offsets.shape[0]):
        rolled = gate.copy()
        for group_index, indices in enumerate(groups):
            if len(indices) > 1:
                rolled[indices] = np.roll(gate[indices], int(offsets[permutation, group_index]))
        result[permutation] = np.bincount(
            bin_index, weights=rolled.astype(float), minlength=int(n_bins)
        )
    return result


def expected_masks(centers: np.ndarray, coordinate: str) -> tuple[np.ndarray, np.ndarray, str]:
    if coordinate == "io_phase":
        target = ((centers >= 75.0) & (centers < 125.0)) | (
            (centers >= 235.0) & (centers < 285.0)
        )
        control = (centers >= 125.0) & (centers < 175.0)
        label = "Io peaks minus 125-175 trough"
    else:
        target = ((centers >= 25.0) & (centers < 75.0)) | (
            (centers >= 195.0) & (centers < 245.0)
        )
        control = ~target
        label = "CML 25-75,195-245 minus complement"
    return target, control, label


def pooled_rate(counts: np.ndarray, exposure: np.ndarray, mask: np.ndarray) -> np.ndarray:
    denominator = float(np.sum(exposure[mask]))
    if denominator <= 0.0:
        return np.full(counts.shape[:-1], np.nan)
    return np.sum(counts[..., mask], axis=-1) / denominator


def analyze_detector_coordinate(
    source: pd.DataFrame,
    detector: str,
    coordinate: str,
    angle_col: str,
    phase_bin_deg: float,
    n_permutations: int,
    min_candidates: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_bins = int(round(360.0 / float(phase_bin_deg)))
    centers = (np.arange(n_bins) + 0.5) * float(phase_bin_deg)
    bin_index = np.floor(source[angle_col].to_numpy(dtype=float) / float(phase_bin_deg)).astype(int) % n_bins
    exposure = np.bincount(bin_index, minlength=n_bins).astype(float)
    groups, offsets = make_roll_offsets(source, n_permutations, rng)
    target, control, contrast_label = expected_masks(centers, coordinate)
    curve_rows = []
    metric_rows = []
    null_contrasts = []
    null_amplitudes = []
    eligible = []
    gate_results: list[tuple[np.ndarray, np.ndarray, np.ndarray, float, int]] = []

    for tail_quantile in TAIL_QUANTILES:
        for min_bands in MIN_EXTREME_BANDS:
            gate = gate_vector(source, tail_quantile, min_bands).astype(np.int8)
            counts = np.bincount(bin_index, weights=gate, minlength=n_bins).astype(float)
            null_counts = rolled_null_counts(gate, bin_index, n_bins, groups, offsets)
            smooth = circular_smooth(counts, exposure)
            null_smooth = circular_smooth(null_counts, exposure[None, :])
            contrast = float(
                pooled_rate(counts, exposure, target) - pooled_rate(counts, exposure, control)
            )
            null_contrast = pooled_rate(null_counts, exposure, target) - pooled_rate(
                null_counts, exposure, control
            )
            amplitude = float(np.nanmax(smooth) - np.nanmin(smooth))
            null_amplitude = np.nanmax(null_smooth, axis=1) - np.nanmin(null_smooth, axis=1)
            enough = int(np.sum(gate)) >= int(min_candidates)
            eligible.append(enough)
            null_contrasts.append(null_contrast)
            null_amplitudes.append(null_amplitude)
            gate_results.append((counts, null_smooth, smooth, float(tail_quantile), int(min_bands)))
            metric_rows.append(
                {
                    "detector": detector,
                    "coordinate": coordinate,
                    "tail_quantile": float(tail_quantile),
                    "min_extreme_bands": int(min_bands),
                    "n_valid": int(np.sum(exposure)),
                    "n_candidate": int(np.sum(gate)),
                    "overall_occurrence": float(np.mean(gate)),
                    "prespecified_contrast": contrast,
                    "prespecified_local_p": (
                        float((1 + np.sum(null_contrast >= contrast)) / (len(null_contrast) + 1))
                        if enough
                        else np.nan
                    ),
                    "smoothed_amplitude": amplitude,
                    "amplitude_local_p": (
                        float((1 + np.sum(null_amplitude >= amplitude)) / (len(null_amplitude) + 1))
                        if enough
                        else np.nan
                    ),
                    "contrast_definition": contrast_label,
                    "maximum_phase_center_deg": float(centers[int(np.nanargmax(smooth))]),
                    "minimum_phase_center_deg": float(centers[int(np.nanargmin(smooth))]),
                    "eligible_for_gate_scan": bool(enough),
                }
            )

    eligible_array = np.asarray(eligible, dtype=bool)
    contrast_stack = np.asarray(null_contrasts, dtype=float)
    amplitude_stack = np.asarray(null_amplitudes, dtype=float)
    max_null_contrast = np.nanmax(contrast_stack[eligible_array], axis=0)
    max_null_amplitude = np.nanmax(amplitude_stack[eligible_array], axis=0)
    for index, (counts, null_smooth, smooth, tail_quantile, min_bands) in enumerate(gate_results):
        metric = metric_rows[index]
        if eligible_array[index]:
            metric["prespecified_familywise_p"] = float(
                (1 + np.sum(max_null_contrast >= metric["prespecified_contrast"]))
                / (len(max_null_contrast) + 1)
            )
            metric["amplitude_familywise_p"] = float(
                (1 + np.sum(max_null_amplitude >= metric["smoothed_amplitude"]))
                / (len(max_null_amplitude) + 1)
            )
        else:
            metric["prespecified_familywise_p"] = np.nan
            metric["amplitude_familywise_p"] = np.nan
        for phase_index, center in enumerate(centers):
            curve_rows.append(
                {
                    "detector": detector,
                    "coordinate": coordinate,
                    "tail_quantile": tail_quantile,
                    "min_extreme_bands": min_bands,
                    "phase_center_deg": center,
                    "n_valid": int(exposure[phase_index]),
                    "n_candidate": int(counts[phase_index]),
                    "occurrence": float(counts[phase_index] / exposure[phase_index]),
                    "occurrence_smoothed": float(smooth[phase_index]),
                    "null_median_smoothed": float(np.nanmedian(null_smooth[:, phase_index])),
                    "null_low_smoothed": float(np.nanquantile(null_smooth[:, phase_index], 0.025)),
                    "null_high_smoothed": float(np.nanquantile(null_smooth[:, phase_index], 0.975)),
                }
            )
    return pd.DataFrame(curve_rows), pd.DataFrame(metric_rows)


def run_gate_scan(
    events: pd.DataFrame,
    phase_bin_deg: float,
    n_permutations: int,
    min_candidates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    curves = []
    metrics = []
    rng = np.random.default_rng(int(seed))
    coordinates = {
        "io_phase": "io_phase_deg",
        "system_iii_cml": "jupiter_cml_system_iii_deg",
    }
    for detector in DETECTORS:
        source = valid_detector_source(events, detector)
        for coordinate, angle_col in coordinates.items():
            curve, metric = analyze_detector_coordinate(
                source,
                detector,
                coordinate,
                angle_col,
                phase_bin_deg,
                n_permutations,
                min_candidates,
                rng,
            )
            curves.append(curve)
            metrics.append(metric)
    return pd.concat(curves, ignore_index=True), pd.concat(metrics, ignore_index=True)


def select_best(metrics: pd.DataFrame, statistic: str = "prespecified_contrast") -> pd.DataFrame:
    source = metrics[metrics["eligible_for_gate_scan"]].copy()
    indices = source.groupby(["detector", "coordinate"], sort=True)[statistic].idxmax()
    return source.loc[indices].sort_values(["detector", "coordinate"]).reset_index(drop=True)


def observed_contrast_for_gate(
    source: pd.DataFrame,
    gate: np.ndarray,
    angle_col: str,
    coordinate: str,
    phase_bin_deg: float,
) -> float:
    n_bins = int(round(360.0 / float(phase_bin_deg)))
    centers = (np.arange(n_bins) + 0.5) * float(phase_bin_deg)
    bins = np.floor(source[angle_col].to_numpy(dtype=float) / float(phase_bin_deg)).astype(int) % n_bins
    exposure = np.bincount(bins, minlength=n_bins).astype(float)
    counts = np.bincount(bins, weights=np.asarray(gate, dtype=float), minlength=n_bins)
    target, control, _ = expected_masks(centers, coordinate)
    return float(
        pooled_rate(counts, exposure, target) - pooled_rate(counts, exposure, control)
    )


def choose_gate_on_training_half(
    source: pd.DataFrame,
    angle_col: str,
    coordinate: str,
    phase_bin_deg: float,
    min_candidates: int,
) -> tuple[float, int, float]:
    choices = []
    for tail_quantile in TAIL_QUANTILES:
        for min_bands in MIN_EXTREME_BANDS:
            gate = gate_vector(source, tail_quantile, min_bands)
            if int(np.sum(gate)) < int(min_candidates):
                continue
            contrast = observed_contrast_for_gate(
                source, gate, angle_col, coordinate, phase_bin_deg
            )
            choices.append((contrast, float(tail_quantile), int(min_bands)))
    if not choices:
        raise RuntimeError("No burst gate has enough candidates in a training half")
    contrast, tail_quantile, min_bands = max(choices, key=lambda value: value[0])
    return tail_quantile, min_bands, contrast


def crossfit_detector_coordinate(
    source: pd.DataFrame,
    detector: str,
    coordinate: str,
    angle_col: str,
    phase_bin_deg: float,
    n_permutations: int,
    min_training_candidates: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, object]]:
    unique_dates = np.sort(source["date"].unique())
    split_index = len(unique_dates) // 2
    first_dates = set(unique_dates[:split_index])
    first_mask = source["date"].isin(first_dates).to_numpy(dtype=bool)
    second_mask = ~first_mask
    first = source[first_mask].reset_index(drop=True)
    second = source[second_mask].reset_index(drop=True)

    # Each validation half uses a gate chosen solely on the other half.
    q_for_second, bands_for_second, first_train_contrast = choose_gate_on_training_half(
        first,
        angle_col,
        coordinate,
        phase_bin_deg,
        min_training_candidates,
    )
    q_for_first, bands_for_first, second_train_contrast = choose_gate_on_training_half(
        second,
        angle_col,
        coordinate,
        phase_bin_deg,
        min_training_candidates,
    )
    gate = np.zeros(len(source), dtype=np.int8)
    gate[first_mask] = gate_vector(source[first_mask], q_for_first, bands_for_first)
    gate[second_mask] = gate_vector(source[second_mask], q_for_second, bands_for_second)

    n_bins = int(round(360.0 / float(phase_bin_deg)))
    centers = (np.arange(n_bins) + 0.5) * float(phase_bin_deg)
    bins = np.floor(source[angle_col].to_numpy(dtype=float) / float(phase_bin_deg)).astype(int) % n_bins
    exposure = np.bincount(bins, minlength=n_bins).astype(float)
    counts = np.bincount(bins, weights=gate, minlength=n_bins).astype(float)
    groups, offsets = make_roll_offsets(source, n_permutations, rng)
    null_counts = rolled_null_counts(gate, bins, n_bins, groups, offsets)
    smooth = circular_smooth(counts, exposure)
    null_smooth = circular_smooth(null_counts, exposure[None, :])
    target, control, contrast_label = expected_masks(centers, coordinate)
    contrast = float(
        pooled_rate(counts, exposure, target) - pooled_rate(counts, exposure, control)
    )
    null_contrast = pooled_rate(null_counts, exposure, target) - pooled_rate(
        null_counts, exposure, control
    )
    p_value = float((1 + np.sum(null_contrast >= contrast)) / (len(null_contrast) + 1))
    curve = pd.DataFrame(
        {
            "detector": detector,
            "coordinate": coordinate,
            "phase_center_deg": centers,
            "n_valid": exposure.astype(int),
            "n_candidate": counts.astype(int),
            "occurrence": counts / exposure,
            "occurrence_smoothed": smooth,
            "null_median_smoothed": np.nanmedian(null_smooth, axis=0),
            "null_low_smoothed": np.nanquantile(null_smooth, 0.025, axis=0),
            "null_high_smoothed": np.nanquantile(null_smooth, 0.975, axis=0),
        }
    )
    metric: dict[str, object] = {
        "detector": detector,
        "coordinate": coordinate,
        "split_date": str(pd.Timestamp(unique_dates[split_index]).date()),
        "gate_for_first_half_tail_quantile": q_for_first,
        "gate_for_first_half_min_extreme_bands": bands_for_first,
        "gate_for_first_half_selected_on_second_contrast": second_train_contrast,
        "gate_for_second_half_tail_quantile": q_for_second,
        "gate_for_second_half_min_extreme_bands": bands_for_second,
        "gate_for_second_half_selected_on_first_contrast": first_train_contrast,
        "first_half_candidates": int(np.sum(gate[first_mask])),
        "second_half_candidates": int(np.sum(gate[second_mask])),
        "n_candidate": int(np.sum(gate)),
        "overall_occurrence": float(np.mean(gate)),
        "crossfit_prespecified_contrast": contrast,
        "crossfit_p": p_value,
        "contrast_definition": contrast_label,
        "maximum_phase_center_deg": float(centers[int(np.nanargmax(smooth))]),
        "minimum_phase_center_deg": float(centers[int(np.nanargmin(smooth))]),
    }
    return curve, metric


def run_crossfit(
    events: pd.DataFrame,
    phase_bin_deg: float,
    n_permutations: int,
    min_training_candidates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    curves = []
    metrics = []
    rng = np.random.default_rng(int(seed) + 1701)
    coordinates = {
        "io_phase": "io_phase_deg",
        "system_iii_cml": "jupiter_cml_system_iii_deg",
    }
    for detector in DETECTORS:
        source = valid_detector_source(events, detector)
        for coordinate, angle_col in coordinates.items():
            curve, metric = crossfit_detector_coordinate(
                source,
                detector,
                coordinate,
                angle_col,
                phase_bin_deg,
                n_permutations,
                min_training_candidates,
                rng,
            )
            curves.append(curve)
            metrics.append(metric)
    return pd.concat(curves, ignore_index=True), pd.DataFrame(metrics)


def curve_slice(curves: pd.DataFrame, metric: pd.Series) -> pd.DataFrame:
    return curves[
        curves["detector"].eq(metric["detector"])
        & curves["coordinate"].eq(metric["coordinate"])
        & np.isclose(curves["tail_quantile"], float(metric["tail_quantile"]))
        & curves["min_extreme_bands"].eq(int(metric["min_extreme_bands"]))
    ].sort_values("phase_center_deg")


def plot_strict_tail(curves: pd.DataFrame, metrics: pd.DataFrame, output: Path) -> Path:
    tail_quantile = 0.995
    min_bands = 1
    fig, axes = plt.subplots(3, 2, figsize=(12.4, 10.2), sharex=True)
    for row, detector in enumerate(DETECTORS):
        for column, coordinate in enumerate(("io_phase", "system_iii_cml")):
            ax = axes[row, column]
            source = curves[
                curves["detector"].eq(detector)
                & curves["coordinate"].eq(coordinate)
                & np.isclose(curves["tail_quantile"], tail_quantile)
                & curves["min_extreme_bands"].eq(min_bands)
            ].sort_values("phase_center_deg")
            metric = metrics[
                metrics["detector"].eq(detector)
                & metrics["coordinate"].eq(coordinate)
                & np.isclose(metrics["tail_quantile"], tail_quantile)
                & metrics["min_extreme_bands"].eq(min_bands)
            ].iloc[0]
            x = source["phase_center_deg"].to_numpy(dtype=float)
            ax.fill_between(
                x,
                100.0 * source["null_low_smoothed"],
                100.0 * source["null_high_smoothed"],
                color="0.82",
                alpha=0.8,
            )
            ax.plot(x, 100.0 * source["occurrence_smoothed"], color=COLORS[detector], lw=2.0)
            ax.scatter(x, 100.0 * source["occurrence"], color=COLORS[detector], s=12)
            ax.set_title("Io phase" if coordinate == "io_phase" else "System III CML")
            ax.text(
                0.02,
                0.96,
                f"n={int(metric['n_candidate'])}; local p={metric['prespecified_local_p']:.3f}; "
                f"gate-corrected p={metric['prespecified_familywise_p']:.3f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
            )
            ax.grid(True, color="0.91", lw=0.6)
            ax.set_xlim(0, 360)
            ax.set_xticks(np.arange(0, 361, 60))
            if column == 0:
                ax.set_ylabel(f"{LABELS[detector]}\nOccurrence (%)")
            if row == 2:
                ax.set_xlabel("Phase / longitude (deg)")
    fig.suptitle("Very-bursty gate: at least one channel in its daily top 0.5%")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = output / "jupiter_burst_gate_top_0p5pct.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_best_gate(curves: pd.DataFrame, best: pd.DataFrame, output: Path) -> Path:
    fig, axes = plt.subplots(3, 2, figsize=(12.4, 10.2), sharex=True)
    for row, detector in enumerate(DETECTORS):
        for column, coordinate in enumerate(("io_phase", "system_iii_cml")):
            ax = axes[row, column]
            metric = best[
                best["detector"].eq(detector) & best["coordinate"].eq(coordinate)
            ].iloc[0]
            source = curve_slice(curves, metric)
            x = source["phase_center_deg"].to_numpy(dtype=float)
            ax.fill_between(
                x,
                100.0 * source["null_low_smoothed"],
                100.0 * source["null_high_smoothed"],
                color="0.82",
                alpha=0.8,
            )
            ax.plot(x, 100.0 * source["occurrence_smoothed"], color=COLORS[detector], lw=2.0)
            ax.scatter(x, 100.0 * source["occurrence"], color=COLORS[detector], s=12)
            ax.set_title("Io phase" if coordinate == "io_phase" else "System III CML")
            ax.text(
                0.02,
                0.96,
                f"q={100*float(metric['tail_quantile']):.1f}%; bands≥{int(metric['min_extreme_bands'])}; "
                f"n={int(metric['n_candidate'])}\nlocal p={metric['prespecified_local_p']:.3f}; "
                f"gate-corrected p={metric['prespecified_familywise_p']:.3f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
            )
            ax.grid(True, color="0.91", lw=0.6)
            ax.set_xlim(0, 360)
            ax.set_xticks(np.arange(0, 361, 60))
            if column == 0:
                ax.set_ylabel(f"{LABELS[detector]}\nOccurrence (%)")
            if row == 2:
                ax.set_xlabel("Phase / longitude (deg)")
    fig.suptitle("Best paper-window burst gate (selection corrected)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = output / "jupiter_burst_gate_best_prespecified.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_contrast_scan(metrics: pd.DataFrame, output: Path) -> Path:
    fig, axes = plt.subplots(3, 2, figsize=(12.4, 9.7), sharex=True)
    markers = {1: "o", 2: "s", 3: "^"}
    for row, detector in enumerate(DETECTORS):
        for column, coordinate in enumerate(("io_phase", "system_iii_cml")):
            ax = axes[row, column]
            source = metrics[
                metrics["detector"].eq(detector) & metrics["coordinate"].eq(coordinate)
            ]
            for min_bands in MIN_EXTREME_BANDS:
                line = source[source["min_extreme_bands"].eq(min_bands)].sort_values("tail_quantile")
                ax.plot(
                    100.0 * (1.0 - line["tail_quantile"]),
                    100.0 * line["prespecified_contrast"],
                    marker=markers[min_bands],
                    label=f"≥{min_bands} extreme band{'s' if min_bands > 1 else ''}",
                )
                significant = line["prespecified_familywise_p"].lt(0.05)
                ax.scatter(
                    100.0 * (1.0 - line.loc[significant, "tail_quantile"]),
                    100.0 * line.loc[significant, "prespecified_contrast"],
                    s=80,
                    facecolor="none",
                    edgecolor="black",
                    linewidth=1.4,
                )
            ax.axhline(0.0, color="0.35", lw=0.8)
            ax.set_xscale("log")
            ax.invert_xaxis()
            ax.set_xticks([0.5, 1.0, 2.5, 5.0])
            ax.set_xticklabels(["0.5", "1", "2.5", "5"])
            ax.grid(True, color="0.91", lw=0.6)
            ax.set_title("Io expected windows" if coordinate == "io_phase" else "System III expected windows")
            if column == 0:
                ax.set_ylabel(f"{LABELS[detector]}\nContrast (percentage points)")
            if row == 2:
                ax.set_xlabel("Daily per-channel upper-tail size (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Does stricter phase-blind time gating strengthen the paper's phase windows?")
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    path = output / "jupiter_burst_gate_contrast_scan.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_crossfit(curves: pd.DataFrame, metrics: pd.DataFrame, output: Path) -> Path:
    fig, axes = plt.subplots(3, 2, figsize=(12.4, 10.2), sharex=True)
    for row, detector in enumerate(DETECTORS):
        for column, coordinate in enumerate(("io_phase", "system_iii_cml")):
            ax = axes[row, column]
            source = curves[
                curves["detector"].eq(detector) & curves["coordinate"].eq(coordinate)
            ].sort_values("phase_center_deg")
            metric = metrics[
                metrics["detector"].eq(detector) & metrics["coordinate"].eq(coordinate)
            ].iloc[0]
            x = source["phase_center_deg"].to_numpy(dtype=float)
            ax.fill_between(
                x,
                100.0 * source["null_low_smoothed"],
                100.0 * source["null_high_smoothed"],
                color="0.82",
                alpha=0.8,
            )
            ax.plot(x, 100.0 * source["occurrence_smoothed"], color=COLORS[detector], lw=2.0)
            ax.scatter(x, 100.0 * source["occurrence"], color=COLORS[detector], s=12)
            ax.set_title("Io phase" if coordinate == "io_phase" else "System III CML")
            ax.text(
                0.02,
                0.96,
                f"cross-fit n={int(metric['n_candidate'])}; contrast="
                f"{100*float(metric['crossfit_prespecified_contrast']):+.2f} pp; "
                f"p={float(metric['crossfit_p']):.3f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
            )
            ax.grid(True, color="0.91", lw=0.6)
            ax.set_xlim(0, 360)
            ax.set_xticks(np.arange(0, 361, 60))
            if column == 0:
                ax.set_ylabel(f"{LABELS[detector]}\nOccurrence (%)")
            if row == 2:
                ax.set_xlabel("Phase / longitude (deg)")
    fig.suptitle("Chronological cross-fit: each half uses a gate selected on the other half")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = output / "jupiter_burst_gate_crossfit_validation.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def write_report(
    output: Path,
    config: dict[str, object],
    events: pd.DataFrame,
    metrics: pd.DataFrame,
    best: pd.DataFrame,
    crossfit_metrics: pd.DataFrame,
    paths: list[Path],
) -> Path:
    columns = [
        "detector",
        "coordinate",
        "tail_quantile",
        "min_extreme_bands",
        "n_candidate",
        "overall_occurrence",
        "maximum_phase_center_deg",
        "prespecified_contrast",
        "prespecified_local_p",
        "prespecified_familywise_p",
        "smoothed_amplitude",
        "amplitude_local_p",
        "amplitude_familywise_p",
    ]
    corrected = best["prespecified_familywise_p"].lt(0.05)
    replicated = False
    for coordinate in ("io_phase", "system_iii_cml"):
        rows = best[best["coordinate"].eq(coordinate)].set_index("detector")
        if all(
            detector in rows.index
            and float(rows.loc[detector, "prespecified_contrast"]) > 0.0
            and float(rows.loc[detector, "prespecified_familywise_p"]) < 0.05
            for detector in ANTENNAS
        ):
            replicated = True
    if replicated:
        conclusion = (
            "A selection-corrected expected-window excess replicates across RV1 and RV2; this is a burst-gated Jupiter candidate requiring independent validation."
        )
    elif corrected.any():
        conclusion = (
            "At least one selected burst gate is significant after the within-detector gate scan, but the effect does not replicate across RV1 and RV2 and is not a secure Jupiter detection."
        )
    else:
        conclusion = (
            "No burst gate remains significant after correcting for the threshold/minimum-band scan. "
            f"The chronological cross-fit is also null (smallest p={float(crossfit_metrics['crossfit_p'].min()):.3f}). "
            "Time gating does not reveal the paper's Jupiter phase pattern."
        )
    strict = metrics[
        np.isclose(metrics["tail_quantile"], 0.995)
        & metrics["min_extreme_bands"].eq(1)
    ].copy()
    rv2_cml_hint = strict[
        strict["detector"].eq("rv2_coarse") & strict["coordinate"].eq("system_iii_cml")
    ].iloc[0]
    hint_statement = (
        "The strict RV2 System-III curve has an uncorrected expected-window p="
        f"{float(rv2_cml_hint['prespecified_local_p']):.3f}, but its gate-scan-corrected p="
        f"{float(rv2_cml_hint['prespecified_familywise_p']):.3f}; the cross-fit does not validate it."
    )
    lines = [
        "# RAE-2 burst-gated Jupiter phase-occurrence scan",
        "",
        "## Bottom line",
        "",
        conclusion,
        "",
        "## Best paper-window gate per detector and coordinate",
        "",
        best[columns].to_string(index=False),
        "",
        "## Predeclared very-bursty gate: top 0.5% in at least one channel",
        "",
        strict[columns].to_string(index=False),
        "",
        hint_statement,
        "",
        "## Chronological cross-fit validation",
        "",
        crossfit_metrics.to_string(index=False),
        "",
        "## Method and anti-selection controls",
        "",
        "- The sub-five-minute SVD attempt was rejected: RAE-2 scans bands sequentially, so only about 160 one-minute intervals contained seven or more bands.",
        "- Native one-minute single-band log powers were ranked within receiver, frequency band, and UTC day without using Jupiter geometry.",
        "- Complete five-minute scan windows were gated at daily per-channel percentiles 95, 97.5, 99, and 99.5, requiring at least one, two, or three extreme bands.",
        "- RV1/RV2 coincidence requires the same frequency band in both receivers to pass the gate in the same five-minute window.",
        f"- Each gate used {config['n_permutations']} within-day circular rolls. Family-wise p-values compare the observed statistic with the maximum null statistic over all 12 gates for that detector and coordinate.",
        "- The expected-window statistic was fixed before seeing these gated curves: Io 75-125 plus 235-285 versus 125-175; System III 25-75 plus 195-245 versus the complement.",
        "- The separate amplitude test allows a peak anywhere, and is also corrected over the gate scan.",
        "- In the cross-fit, the first and second chronological halves independently choose gates for each other. The reported cross-fit p-value therefore tests phase data that did not select its own gate.",
        "",
        "## Data accounting",
        "",
        f"- Five-minute receiver-event rows: {len(events):,}",
        f"- RV1 valid unocculted windows: {len(valid_detector_source(events, 'rv1_coarse')):,}",
        f"- RV2 valid unocculted windows: {len(valid_detector_source(events, 'rv2_coarse')):,}",
        f"- Coincident valid unocculted windows: {len(valid_detector_source(events, 'rv1_and_rv2')):,}",
        "",
        "## Outputs",
        "",
        *[f"- `{path.name}`" for path in paths],
        "- `burst_gate_statistics.csv`",
        "- `burst_gate_occurrence_curves.csv`",
        "- `burst_gate_best_statistics.csv`",
        "- `five_minute_channel_percentile_events.parquet`",
    ]
    path = output / "RAE2_JUPITER_BURST_GATE_REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregated", type=Path, default=DEFAULT_AGGREGATED)
    parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--occultations", type=Path, default=DEFAULT_OCCULTATIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-daily-samples", type=int, default=20)
    parser.add_argument("--min-channels", type=int, default=7)
    parser.add_argument("--min-candidates", type=int, default=20)
    parser.add_argument("--phase-bin-deg", type=float, default=15.0)
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=260825075)
    args = parser.parse_args()

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "aggregated": str(args.aggregated),
        "geometry": str(args.geometry),
        "occultations": str(args.occultations),
        "min_daily_samples": int(args.min_daily_samples),
        "min_channels": int(args.min_channels),
        "min_candidates": int(args.min_candidates),
        "phase_bin_deg": float(args.phase_bin_deg),
        "n_permutations": int(args.n_permutations),
        "seed": int(args.seed),
        "tail_quantiles": list(TAIL_QUANTILES),
        "min_extreme_bands": list(MIN_EXTREME_BANDS),
    }
    (output / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    aggregated = pd.read_parquet(args.aggregated)
    geometry = pd.read_parquet(args.geometry)
    intervals = pd.read_csv(args.occultations, parse_dates=["start", "stop"])
    channel_scores = prepare_channel_percentiles(aggregated, int(args.min_daily_samples))
    events = build_event_table(channel_scores, geometry, intervals, int(args.min_channels))
    events.to_parquet(output / "five_minute_channel_percentile_events.parquet", index=False)
    curves, metrics = run_gate_scan(
        events,
        float(args.phase_bin_deg),
        int(args.n_permutations),
        int(args.min_candidates),
        int(args.seed),
    )
    curves.to_csv(output / "burst_gate_occurrence_curves.csv", index=False)
    metrics.to_csv(output / "burst_gate_statistics.csv", index=False)
    best = select_best(metrics)
    best.to_csv(output / "burst_gate_best_statistics.csv", index=False)
    crossfit_curves, crossfit_metrics = run_crossfit(
        events,
        float(args.phase_bin_deg),
        int(args.n_permutations),
        max(10, int(args.min_candidates) // 2),
        int(args.seed),
    )
    crossfit_curves.to_csv(output / "burst_gate_crossfit_curves.csv", index=False)
    crossfit_metrics.to_csv(output / "burst_gate_crossfit_statistics.csv", index=False)
    paths = [
        plot_strict_tail(curves, metrics, output),
        plot_best_gate(curves, best, output),
        plot_contrast_scan(metrics, output),
        plot_crossfit(crossfit_curves, crossfit_metrics, output),
    ]
    report = write_report(output, config, events, metrics, best, crossfit_metrics, paths)
    print(report)


if __name__ == "__main__":
    main()
