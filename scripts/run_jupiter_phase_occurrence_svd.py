#!/usr/bin/env python
"""Paper-style Jupiter occurrence curves from raw RAE-2 RV receiver powers.

The PSP analysis in Wille et al. (2026) defines occurrence probability as
the fraction of valid, phase-folded intervals classified emission-present.
RAE-2 does not provide the signed cross-spectrum used there, so this script
constructs a deliberately labelled *candidate-emission* analogue:

1. median raw powers in fixed time bins and take log power;
2. whiten each frequency row using its median and positive-side 75th
   percentile within a UTC day;
3. retain low-spectral-entropy SVD modes, with the paper's 3/log(90)
   normalized entropy cutoff scaled to RAE-2's nine bands;
4. call the strongest configured fraction of SVD scores within each day
   candidate-present;
5. fold valid intervals by light-time-corrected Io phase and System-III
   central meridian longitude;
6. compare the observed curves with within-day circular-roll nulls.

RV1 and RV2 are independent total-power receivers, not LHCP/RHCP channels.
All plots and tables therefore avoid polarization labels.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import urllib.request

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_CLEAN = Path(
    "/global/u2/a/afahs/RAE2/RyleVonberg/runtime_inputs/occultogram_v1/cleaned_timeseries.parquet"
)
DEFAULT_EVENTS = Path(
    "/global/u2/a/afahs/RAE2/RyleVonberg/runtime_inputs/occultogram_v1/planet_events_science_baseline.parquet"
)
DEFAULT_OUTPUT = Path("/pscratch/sd/a/afahs/RAE2/jupiter_phase_occurrence_svd_v1")

ANTENNAS = ("rv1_coarse", "rv2_coarse")
ANTENNA_LABEL = {"rv1_coarse": "RV1 (upper V)", "rv2_coarse": "RV2 (lower V)"}
ANTENNA_COLOR = {"rv1_coarse": "#3568a8", "rv2_coarse": "#d4681f"}
FREQUENCIES_MHZ = np.array([0.45, 0.70, 0.90, 1.31, 2.20, 3.93, 4.70, 6.55, 9.18])
PAPER_ENTROPY_LIMIT_NORMALIZED = 3.0 / math.log(90.0)
ENTROPY_SENSITIVITY = (0.55, PAPER_ENTROPY_LIMIT_NORMALIZED, 0.78)

NAIF_KERNELS = {
    "naif0012.tls": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls",
    "pck00010.tpc": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00010.tpc",
    "jup348.bsp": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/jup348.bsp",
    "jup100.bsp": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/a_old_versions/jup100.bsp",
}


@dataclass
class SVDResult:
    reconstruction: np.ndarray
    singular_values: np.ndarray
    entropy_normalized: np.ndarray
    retained: np.ndarray


def robust_sigma(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return np.nan
    q25, q75 = np.percentile(values, [25.0, 75.0])
    return float((q75 - q25) / 1.349)


def sustained(mask: np.ndarray, minimum_run: int = 2) -> np.ndarray:
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
        if stop - start >= int(minimum_run):
            result[start:stop] = True
        start = stop
    return result


def whiten_frequency_rows(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Median/positive-Q75 whitening plus the paper's transient screen."""

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
            scale = robust_sigma(residual[finite])
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = 1.0
        whitened[row_index, finite] = residual[finite] / scale

    column_mean = np.nanmean(whitened, axis=0)
    column_std = np.nanstd(whitened, axis=0)
    std_center = float(np.nanmedian(column_std))
    std_scale = robust_sigma(column_std)
    mean_center = float(np.nanmedian(column_mean))
    mean_scale = robust_sigma(column_mean)
    if not np.isfinite(std_scale) or not np.isfinite(mean_scale):
        flagged = np.zeros(matrix.shape[1], dtype=bool)
    else:
        high_variance = column_std > std_center + 6.0 * std_scale
        near_zero_mean = np.abs(column_mean - mean_center) <= max(1.5 * mean_scale, 0.25)
        flagged = sustained(high_variance & near_zero_mean, minimum_run=2)

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


def entropy_svd(matrix: np.ndarray, entropy_limit_normalized: float) -> SVDResult:
    clean = np.where(np.isfinite(np.asarray(matrix, dtype=float)), matrix, 0.0)
    u, singular_values, vt = np.linalg.svd(clean, full_matrices=False)
    probabilities = np.square(np.abs(u))
    probabilities /= np.maximum(probabilities.sum(axis=0, keepdims=True), 1e-15)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(probabilities > 0.0, probabilities * np.log(probabilities), 0.0)
    entropy = -np.sum(terms, axis=0) / math.log(max(clean.shape[0], 2))
    retained = entropy < float(entropy_limit_normalized)
    reconstruction = (
        (u[:, retained] * singular_values[retained]) @ vt[retained]
        if np.any(retained)
        else np.zeros_like(clean)
    )
    return SVDResult(reconstruction, singular_values, entropy, retained)


def aggregate_raw(clean_path: Path, bin_minutes: int) -> pd.DataFrame:
    columns = ["time", "frequency_band", "antenna", "power", "is_valid"]
    raw = pd.read_parquet(clean_path, columns=columns)
    raw["time"] = pd.to_datetime(raw["time"])
    raw["power"] = pd.to_numeric(raw["power"], errors="coerce")
    raw = raw[
        raw["antenna"].astype(str).isin(ANTENNAS)
        & raw["is_valid"].astype(bool)
        & raw["power"].gt(0.0)
        & raw["frequency_band"].between(1, 9)
    ].copy()
    raw["antenna"] = raw["antenna"].astype(str)
    raw["frequency_band"] = raw["frequency_band"].astype(np.int8)
    raw["log_power"] = np.log(raw["power"].to_numpy(dtype=float))
    raw["time_bin"] = (
        raw["time"].dt.floor(f"{int(bin_minutes)}min")
        + pd.Timedelta(minutes=float(bin_minutes) / 2.0)
    )
    aggregated = (
        raw.groupby(["antenna", "frequency_band", "time_bin"], sort=True, observed=True)
        .agg(log_power=("log_power", "median"), n_raw=("log_power", "size"))
        .reset_index()
        .rename(columns={"time_bin": "time"})
    )
    aggregated["frequency_mhz"] = FREQUENCIES_MHZ[
        aggregated["frequency_band"].to_numpy(dtype=int) - 1
    ]
    return aggregated


def select_top_fraction(scores: np.ndarray, valid: np.ndarray, fraction: float) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(scores) & (scores > 0.0)
    selected = np.zeros(len(scores), dtype=bool)
    candidates = np.flatnonzero(valid)
    if candidates.size == 0:
        return selected
    n_select = max(1, int(math.ceil(float(fraction) * candidates.size)))
    order = candidates[np.argsort(scores[candidates], kind="mergesort")[-n_select:]]
    selected[order] = True
    return selected


def score_daily_svd(
    aggregated: pd.DataFrame,
    entropy_limits: tuple[float, ...],
    candidate_fraction: float,
    min_channels: int,
    min_day_intervals: int,
) -> pd.DataFrame:
    work = aggregated.copy()
    work["date"] = work["time"].dt.floor("D")
    rows: list[pd.DataFrame] = []
    for (antenna, date), source in work.groupby(["antenna", "date"], sort=True, observed=True):
        pivot = source.pivot_table(
            index="frequency_band", columns="time", values="log_power", aggfunc="median"
        ).reindex(np.arange(1, 10))
        if pivot.shape[1] < int(min_day_intervals):
            continue
        times = pd.DatetimeIndex(pivot.columns)
        matrix = pivot.to_numpy(dtype=float)
        n_channels = np.sum(np.isfinite(matrix), axis=0)
        whitened, transient_flag = whiten_frequency_rows(matrix)
        valid = (n_channels >= int(min_channels)) & ~transient_flag
        for entropy_limit in entropy_limits:
            svd = entropy_svd(whitened, entropy_limit)
            positive = np.maximum(svd.reconstruction, 0.0)
            score = np.sqrt(np.mean(np.square(positive), axis=0))
            candidate = select_top_fraction(score, valid, candidate_fraction)
            percentile = np.full(len(score), np.nan, dtype=float)
            if np.any(valid):
                percentile[valid] = pd.Series(score[valid]).rank(method="average", pct=True).to_numpy()
            rows.append(
                pd.DataFrame(
                    {
                        "time": times,
                        "date": date,
                        "antenna": str(antenna),
                        "entropy_limit_normalized": float(entropy_limit),
                        "svd_score": score,
                        "svd_score_percentile": percentile,
                        "candidate_present": candidate,
                        "valid_interval": valid,
                        "n_channels": n_channels,
                        "transient_flag": transient_flag,
                        "n_retained_modes": int(np.sum(svd.retained)),
                        "leading_singular_fraction": (
                            float(svd.singular_values[0] / np.sum(svd.singular_values))
                            if np.sum(svd.singular_values) > 0.0
                            else np.nan
                        ),
                    }
                )
            )
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(
        ["entropy_limit_normalized", "antenna", "time"]
    ).reset_index(drop=True)


def ensure_kernels(kernel_dir: Path, refresh: bool = False) -> dict[str, Path]:
    kernel_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, url in NAIF_KERNELS.items():
        path = kernel_dir / name
        if refresh or not path.exists():
            temporary = path.with_suffix(path.suffix + ".part")
            with urllib.request.urlopen(url, timeout=600) as response, temporary.open("wb") as handle:
                while True:
                    block = response.read(8 * 1024 * 1024)
                    if not block:
                        break
                    handle.write(block)
            temporary.replace(path)
        paths[name] = path
    return paths


def io_phase_from_vectors(observer_body: np.ndarray, io_body: np.ndarray) -> np.ndarray:
    """Io phase, zero at superior conjunction, increasing eastward."""

    observer_body = np.atleast_2d(np.asarray(observer_body, dtype=float))
    io_body = np.atleast_2d(np.asarray(io_body, dtype=float))
    reference = -observer_body[:, :2]
    target = io_body[:, :2]
    cross = reference[:, 0] * target[:, 1] - reference[:, 1] * target[:, 0]
    dot = np.sum(reference * target, axis=1)
    return np.degrees(np.arctan2(cross, dot)) % 360.0


def compute_spice_geometry(times: pd.DatetimeIndex, kernels: dict[str, Path]) -> pd.DataFrame:
    try:
        import spiceypy as spice
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError("spiceypy is required; install it in the Perlmutter analysis environment") from exc

    ordered_times = pd.DatetimeIndex(pd.to_datetime(times)).sort_values().unique()
    time_strings = [timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f") for timestamp in ordered_times]
    spice.kclear()
    for name in ("naif0012.tls", "pck00010.tpc", "jup348.bsp", "jup100.bsp"):
        spice.furnsh(str(kernels[name]))
    try:
        receive_et = np.asarray(spice.str2et(time_strings), dtype=float)
        earth_to_jupiter, light_time_s = spice.spkpos(
            "JUPITER", receive_et, "J2000", "LT+S", "EARTH"
        )
        earth_to_jupiter = np.asarray(earth_to_jupiter, dtype=float)
        light_time_s = np.asarray(light_time_s, dtype=float)
        emission_et = receive_et - light_time_s
        io_j2000 = np.asarray(
            spice.spkpos("IO", emission_et, "J2000", "NONE", "JUPITER")[0], dtype=float
        )
        observer_body = np.empty_like(earth_to_jupiter)
        io_body = np.empty_like(io_j2000)
        for index, epoch in enumerate(emission_et):
            rotation = np.asarray(spice.pxform("J2000", "IAU_JUPITER", float(epoch)))
            observer_body[index] = rotation @ (-earth_to_jupiter[index])
            io_body[index] = rotation @ io_j2000[index]
        observer_east_lon = np.degrees(
            np.arctan2(observer_body[:, 1], observer_body[:, 0])
        ) % 360.0
        cml_west = (360.0 - observer_east_lon) % 360.0
        io_phase = io_phase_from_vectors(observer_body, io_body)
        return pd.DataFrame(
            {
                "time": ordered_times,
                "jupiter_cml_system_iii_deg": cml_west,
                "io_phase_deg": io_phase,
                "jupiter_light_time_s": light_time_s,
                "jupiter_range_au": np.linalg.norm(earth_to_jupiter, axis=1) / 149_597_870.7,
            }
        )
    finally:
        spice.kclear()


def occultation_intervals(events_path: Path) -> pd.DataFrame:
    events = pd.read_parquet(
        events_path, columns=["source_name", "event_id", "event_type", "predicted_event_time"]
    )
    events = events[events["source_name"].astype(str).str.lower().eq("jupiter")].copy()
    events["predicted_event_time"] = pd.to_datetime(events["predicted_event_time"])
    events = events.drop_duplicates(["event_id", "event_type", "predicted_event_time"])
    events = events.sort_values("predicted_event_time")
    rows = []
    start: pd.Timestamp | None = None
    for row in events.itertuples(index=False):
        if str(row.event_type).lower() == "disappearance":
            start = pd.Timestamp(row.predicted_event_time)
        elif str(row.event_type).lower() == "reappearance" and start is not None:
            stop = pd.Timestamp(row.predicted_event_time)
            duration = (stop - start).total_seconds()
            if 0.0 < duration <= 12.0 * 3600.0:
                rows.append({"start": start, "stop": stop, "duration_s": duration})
            start = None
    return pd.DataFrame(rows)


def mark_occulted(times: pd.Series, intervals: pd.DataFrame) -> np.ndarray:
    result = np.zeros(len(times), dtype=bool)
    time_values = pd.to_datetime(times).to_numpy(dtype="datetime64[ns]")
    for row in intervals.itertuples(index=False):
        result |= (time_values >= np.datetime64(row.start)) & (time_values <= np.datetime64(row.stop))
    return result


def add_coincident_detector(scored: pd.DataFrame) -> pd.DataFrame:
    pieces = [scored.copy()]
    for entropy_limit, source in scored.groupby("entropy_limit_normalized", sort=True):
        compact = source[
            ["time", "date", "antenna", "candidate_present", "valid_interval", "svd_score_percentile"]
        ].copy()
        candidates = compact.pivot(index="time", columns="antenna", values="candidate_present")
        valid = compact.pivot(index="time", columns="antenna", values="valid_interval")
        percentiles = compact.pivot(index="time", columns="antenna", values="svd_score_percentile")
        if not all(name in candidates.columns for name in ANTENNAS):
            continue
        both_valid = valid[list(ANTENNAS)].eq(True).all(axis=1)  # noqa: E712
        both_candidate = candidates[list(ANTENNAS)].eq(True).all(axis=1) & both_valid  # noqa: E712
        coincident = pd.DataFrame(
            {
                "time": candidates.index,
                "date": candidates.index.floor("D"),
                "antenna": "rv1_and_rv2",
                "entropy_limit_normalized": float(entropy_limit),
                "svd_score": percentiles[list(ANTENNAS)].min(axis=1).to_numpy(),
                "svd_score_percentile": percentiles[list(ANTENNAS)].min(axis=1).to_numpy(),
                "candidate_present": both_candidate.to_numpy(),
                "valid_interval": both_valid.to_numpy(),
                "n_channels": np.nan,
                "transient_flag": False,
                "n_retained_modes": np.nan,
                "leading_singular_fraction": np.nan,
            }
        )
        pieces.append(coincident)
    return pd.concat(pieces, ignore_index=True)


def circular_smooth(counts: np.ndarray, exposures: np.ndarray) -> np.ndarray:
    smooth_counts = np.roll(counts, -1, axis=-1) + counts + np.roll(counts, 1, axis=-1)
    smooth_exposure = np.roll(exposures, -1, axis=-1) + exposures + np.roll(exposures, 1, axis=-1)
    return np.divide(
        smooth_counts,
        smooth_exposure,
        out=np.full_like(smooth_counts, np.nan, dtype=float),
        where=smooth_exposure > 0,
    )


def phase_curve_with_null(
    source: pd.DataFrame,
    angle_col: str,
    phase_bin_deg: float,
    n_permutations: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    source = source[
        source["valid_interval"] & ~source["jupiter_occulted"] & source[angle_col].notna()
    ].sort_values("time").copy()
    n_bins = int(round(360.0 / float(phase_bin_deg)))
    bin_index = np.floor(source[angle_col].to_numpy(dtype=float) / float(phase_bin_deg)).astype(int) % n_bins
    candidate = source["candidate_present"].to_numpy(dtype=np.int8)
    exposure = np.bincount(bin_index, minlength=n_bins).astype(float)
    counts = np.bincount(bin_index, weights=candidate, minlength=n_bins).astype(float)
    null_counts = np.zeros((int(n_permutations), n_bins), dtype=float)
    groups = [
        group.index.to_numpy(dtype=int)
        for _, group in source.reset_index(drop=True).groupby("date", sort=True)
    ]
    for permutation in range(int(n_permutations)):
        shuffled = candidate.copy()
        for indices in groups:
            if len(indices) > 1:
                shuffled[indices] = np.roll(candidate[indices], int(rng.integers(0, len(indices))))
        null_counts[permutation] = np.bincount(
            bin_index, weights=shuffled, minlength=n_bins
        )
    smooth = circular_smooth(counts, exposure)
    null_smooth = circular_smooth(null_counts, exposure[None, :])
    centers = (np.arange(n_bins) + 0.5) * float(phase_bin_deg)
    summary = pd.DataFrame(
        {
            "phase_center_deg": centers,
            "n_valid": exposure.astype(int),
            "n_candidate": counts.astype(int),
            "occurrence": np.divide(counts, exposure, out=np.full(n_bins, np.nan), where=exposure > 0),
            "occurrence_smoothed": smooth,
            "null_median_smoothed": np.nanmedian(null_smooth, axis=0),
            "null_low_smoothed": np.nanquantile(null_smooth, 0.025, axis=0),
            "null_high_smoothed": np.nanquantile(null_smooth, 0.975, axis=0),
        }
    )
    return summary, null_counts, exposure


def curve_metrics(
    summary: pd.DataFrame,
    null_counts: np.ndarray,
    exposure: np.ndarray,
    coordinate: str,
) -> dict[str, float | int | str]:
    observed = summary["occurrence_smoothed"].to_numpy(dtype=float)
    null_smooth = circular_smooth(null_counts, exposure[None, :])
    amplitude = float(np.nanmax(observed) - np.nanmin(observed))
    null_amplitude = np.nanmax(null_smooth, axis=1) - np.nanmin(null_smooth, axis=1)
    amplitude_p = float((1 + np.sum(null_amplitude >= amplitude)) / (len(null_amplitude) + 1))
    centers = summary["phase_center_deg"].to_numpy(dtype=float)
    if coordinate == "io_phase":
        target = ((centers >= 75.0) & (centers < 125.0)) | (
            (centers >= 235.0) & (centers < 285.0)
        )
        control = (centers >= 125.0) & (centers < 175.0)
        contrast_name = "Io peaks (75-125,235-285) minus trough (125-175)"
    else:
        target = ((centers >= 25.0) & (centers < 75.0)) | (
            (centers >= 195.0) & (centers < 245.0)
        )
        control = ~target
        contrast_name = "CML paper peaks (25-75,195-245) minus complement"

    def pooled_rate(count_array: np.ndarray, mask: np.ndarray) -> np.ndarray:
        denominator = float(np.sum(exposure[mask]))
        return np.sum(count_array[..., mask], axis=-1) / denominator if denominator > 0 else np.nan

    observed_counts = summary["n_candidate"].to_numpy(dtype=float)
    contrast = float(pooled_rate(observed_counts, target) - pooled_rate(observed_counts, control))
    null_contrast = pooled_rate(null_counts, target) - pooled_rate(null_counts, control)
    contrast_p = float((1 + np.sum(null_contrast >= contrast)) / (len(null_contrast) + 1))
    maximum_bin = int(np.nanargmax(observed))
    minimum_bin = int(np.nanargmin(observed))
    return {
        "n_valid": int(np.sum(exposure)),
        "n_candidate": int(np.sum(observed_counts)),
        "overall_occurrence": float(np.sum(observed_counts) / np.sum(exposure)),
        "smoothed_amplitude": amplitude,
        "omnibus_amplitude_p": amplitude_p,
        "prespecified_contrast": contrast,
        "prespecified_contrast_p": contrast_p,
        "contrast_definition": contrast_name,
        "maximum_phase_center_deg": float(centers[maximum_bin]),
        "minimum_phase_center_deg": float(centers[minimum_bin]),
    }


def build_phase_products(
    scored: pd.DataFrame,
    phase_bin_deg: float,
    n_permutations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    metrics = []
    rng = np.random.default_rng(int(seed))
    coordinate_columns = {
        "io_phase": "io_phase_deg",
        "system_iii_cml": "jupiter_cml_system_iii_deg",
    }
    for (entropy_limit, detector), source in scored.groupby(
        ["entropy_limit_normalized", "antenna"], sort=True
    ):
        for coordinate, angle_col in coordinate_columns.items():
            summary, null_counts, exposure = phase_curve_with_null(
                source, angle_col, phase_bin_deg, n_permutations, rng
            )
            summary["entropy_limit_normalized"] = float(entropy_limit)
            summary["detector"] = str(detector)
            summary["coordinate"] = coordinate
            summaries.append(summary)
            metric = curve_metrics(summary, null_counts, exposure, coordinate)
            metric.update(
                {
                    "entropy_limit_normalized": float(entropy_limit),
                    "detector": str(detector),
                    "coordinate": coordinate,
                }
            )
            metrics.append(metric)
    return pd.concat(summaries, ignore_index=True), pd.DataFrame(metrics)


def two_dimensional_occurrence(
    scored: pd.DataFrame, entropy_limit: float, phase_bin_deg: float, min_exposure: int
) -> pd.DataFrame:
    source = scored[
        np.isclose(scored["entropy_limit_normalized"], float(entropy_limit))
        & scored["valid_interval"]
        & ~scored["jupiter_occulted"]
    ].copy()
    source["io_bin_deg"] = (
        np.floor(source["io_phase_deg"] / phase_bin_deg) * phase_bin_deg + phase_bin_deg / 2.0
    )
    source["cml_bin_deg"] = (
        np.floor(source["jupiter_cml_system_iii_deg"] / phase_bin_deg) * phase_bin_deg
        + phase_bin_deg / 2.0
    )
    result = (
        source.groupby(["antenna", "io_bin_deg", "cml_bin_deg"], sort=True)
        .agg(n_valid=("candidate_present", "size"), n_candidate=("candidate_present", "sum"))
        .reset_index()
    )
    result["occurrence"] = result["n_candidate"] / result["n_valid"]
    result.loc[result["n_valid"] < int(min_exposure), "occurrence"] = np.nan
    return result


def plot_figure4_analogue(curves: pd.DataFrame, entropy_limit: float, output: Path) -> Path:
    detectors = list(ANTENNAS)
    coordinates = ["io_phase", "system_iii_cml"]
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.8), sharex=True)
    for row, detector in enumerate(detectors):
        for column, coordinate in enumerate(coordinates):
            ax = axes[row, column]
            source = curves[
                np.isclose(curves["entropy_limit_normalized"], float(entropy_limit))
                & curves["detector"].eq(detector)
                & curves["coordinate"].eq(coordinate)
            ].sort_values("phase_center_deg")
            x = source["phase_center_deg"].to_numpy(dtype=float)
            y = 100.0 * source["occurrence_smoothed"].to_numpy(dtype=float)
            lo = 100.0 * source["null_low_smoothed"].to_numpy(dtype=float)
            hi = 100.0 * source["null_high_smoothed"].to_numpy(dtype=float)
            ax.fill_between(x, lo, hi, color="0.82", alpha=0.75, label="95% circular-roll null")
            ax.plot(x, y, color=ANTENNA_COLOR[detector], lw=2.0, label="3-bin smoothed")
            ax.scatter(
                x,
                100.0 * source["occurrence"],
                s=13,
                facecolor="white",
                edgecolor=ANTENNA_COLOR[detector],
                linewidth=0.7,
                zorder=3,
                label="15° bins",
            )
            if coordinate == "io_phase":
                ax.axvspan(75, 125, color="#6a51a3", alpha=0.07)
                ax.axvspan(235, 285, color="#6a51a3", alpha=0.07)
                ax.axvspan(125, 175, color="#cb181d", alpha=0.05)
                title = "Io phase"
            else:
                ax.axvspan(25, 75, color="#238b45", alpha=0.06)
                ax.axvspan(195, 245, color="#238b45", alpha=0.06)
                title = "System III central meridian longitude"
            ax.set_title(title)
            ax.set_xlim(0, 360)
            ax.set_xticks(np.arange(0, 361, 60))
            ax.grid(True, color="0.91", lw=0.6)
            if column == 0:
                ax.set_ylabel(f"{ANTENNA_LABEL[detector]}\nCandidate occurrence (%)")
            if row == 1:
                ax.set_xlabel("Phase / longitude (deg)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(
        "RAE-2 analogue of paper Figure 4: entropy-SVD candidate occurrence\n"
        "Receiver panels are total power, not circular polarization"
    )
    fig.tight_layout(rect=[0, 0.07, 1, 0.93])
    path = output / "jupiter_phase_occurrence_figure4_analogue.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_coincident(curves: pd.DataFrame, entropy_limit: float, output: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.1), sharey=True)
    for ax, coordinate, title in zip(
        axes,
        ("io_phase", "system_iii_cml"),
        ("Io phase", "System III central meridian longitude"),
        strict=True,
    ):
        source = curves[
            np.isclose(curves["entropy_limit_normalized"], float(entropy_limit))
            & curves["detector"].eq("rv1_and_rv2")
            & curves["coordinate"].eq(coordinate)
        ].sort_values("phase_center_deg")
        x = source["phase_center_deg"].to_numpy(dtype=float)
        ax.fill_between(
            x,
            100.0 * source["null_low_smoothed"],
            100.0 * source["null_high_smoothed"],
            color="0.82",
            alpha=0.75,
        )
        ax.plot(x, 100.0 * source["occurrence_smoothed"], color="#7a3e9d", lw=2.0)
        ax.scatter(x, 100.0 * source["occurrence"], s=14, color="#7a3e9d")
        ax.set_title(title)
        ax.set_xlim(0, 360)
        ax.set_xticks(np.arange(0, 361, 60))
        ax.set_xlabel("Phase / longitude (deg)")
        ax.grid(True, color="0.91", lw=0.6)
    axes[0].set_ylabel("Coincident RV1 & RV2 occurrence (%)")
    fig.suptitle("Stricter two-receiver candidate occurrence")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    path = output / "jupiter_phase_occurrence_coincident_receivers.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_exposure(curves: pd.DataFrame, entropy_limit: float, output: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.3), sharex=True)
    for row, detector in enumerate(ANTENNAS):
        for column, coordinate in enumerate(("io_phase", "system_iii_cml")):
            ax = axes[row, column]
            source = curves[
                np.isclose(curves["entropy_limit_normalized"], float(entropy_limit))
                & curves["detector"].eq(detector)
                & curves["coordinate"].eq(coordinate)
            ].sort_values("phase_center_deg")
            ax.bar(source["phase_center_deg"], source["n_valid"], width=13.5, color=ANTENNA_COLOR[detector])
            ax.set_xlim(0, 360)
            ax.set_xticks(np.arange(0, 361, 60))
            ax.grid(True, axis="y", color="0.91", lw=0.6)
            ax.set_title("Io phase" if coordinate == "io_phase" else "System III longitude")
            if column == 0:
                ax.set_ylabel(f"{ANTENNA_LABEL[detector]}\nValid 5-min intervals")
            if row == 1:
                ax.set_xlabel("Phase / longitude (deg)")
    fig.suptitle("Exposure denominator for the occurrence curves")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = output / "jupiter_phase_occurrence_exposure.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_2d_maps(maps: pd.DataFrame, output: Path) -> Path:
    detectors = [*ANTENNAS, "rv1_and_rv2"]
    labels = [ANTENNA_LABEL[name] for name in ANTENNAS] + ["Coincident RV1 & RV2"]
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.4), sharex=True, sharey=True)
    image = None
    for ax, detector, label in zip(axes, detectors, labels, strict=True):
        source = maps[maps["antenna"].eq(detector)]
        matrix = source.pivot(index="io_bin_deg", columns="cml_bin_deg", values="occurrence")
        matrix = matrix.reindex(
            index=np.sort(maps["io_bin_deg"].unique()),
            columns=np.sort(maps["cml_bin_deg"].unique()),
        )
        detector_mean = float(
            maps.loc[maps["antenna"].eq(detector), "n_candidate"].sum()
            / maps.loc[maps["antenna"].eq(detector), "n_valid"].sum()
        )
        image = ax.imshow(
            100.0 * (matrix.to_numpy(dtype=float) - detector_mean),
            origin="lower",
            extent=(0, 360, 0, 360),
            aspect="auto",
            cmap="RdBu_r",
            vmin=-5,
            vmax=5,
            interpolation="nearest",
        )
        ax.set_title(label)
        ax.set_xlabel("System III CML (deg)")
        ax.set_xticks(np.arange(0, 361, 90))
    axes[0].set_ylabel("Io phase (deg)")
    fig.suptitle("Joint Io-phase / System-III candidate-occurrence maps")
    fig.subplots_adjust(left=0.06, right=0.87, bottom=0.14, top=0.84, wspace=0.08)
    colorbar_axis = fig.add_axes([0.895, 0.18, 0.012, 0.58])
    fig.colorbar(
        image,
        cax=colorbar_axis,
        label="Occurrence minus detector mean (percentage points)",
    )
    path = output / "jupiter_io_cml_occurrence_maps.png"
    fig.savefig(path, dpi=210)
    plt.close(fig)
    return path


def write_report(
    output: Path,
    config: dict[str, object],
    aggregated: pd.DataFrame,
    scored: pd.DataFrame,
    metrics: pd.DataFrame,
    intervals: pd.DataFrame,
    plot_paths: list[Path],
) -> Path:
    primary = metrics[
        np.isclose(metrics["entropy_limit_normalized"], PAPER_ENTROPY_LIMIT_NORMALIZED)
    ].copy()
    significant_primary = primary[
        primary["omnibus_amplitude_p"].lt(0.05)
        | primary["prespecified_contrast_p"].lt(0.05)
    ]
    if significant_primary.empty:
        result_statement = (
            "No primary Io-phase or System-III curve is significant at p < 0.05. "
            "The RAE-2 data therefore do not reproduce the paper's phase-occurrence signature under this classifier."
        )
    else:
        result_statement = (
            "At least one primary phase curve reaches p < 0.05; it must still replicate across both receivers "
            "and entropy cutoffs before being interpreted as Jupiter."
        )
    table_columns = [
        "detector",
        "coordinate",
        "n_valid",
        "overall_occurrence",
        "maximum_phase_center_deg",
        "minimum_phase_center_deg",
        "smoothed_amplitude",
        "omnibus_amplitude_p",
        "prespecified_contrast",
        "prespecified_contrast_p",
    ]
    sensitivity_columns = [
        "entropy_limit_normalized",
        "detector",
        "coordinate",
        "smoothed_amplitude",
        "omnibus_amplitude_p",
        "prespecified_contrast",
        "prespecified_contrast_p",
    ]
    transient_fraction = (
        scored[scored["antenna"].isin(ANTENNAS)]
        .groupby("antenna")["transient_flag"]
        .mean()
        .to_dict()
    )
    lines = [
        "# RAE-2 Jupiter phase-occurrence entropy-SVD trial",
        "",
        "## Bottom line",
        "",
        result_statement,
        "These are candidate-emission occurrence curves, not a polarization-separated replication of the PSP result. "
        "A Jovian interpretation would require repeatable phase structure outside the 95% circular-roll null in both receivers, "
        "preferably with a coincident RV1/RV2 excess and stability across entropy cutoffs.",
        "",
        "## Primary paper-scaled results",
        "",
        primary[table_columns].to_string(index=False),
        "",
        "## Entropy-cutoff sensitivity",
        "",
        metrics[sensitivity_columns].to_string(index=False),
        "",
        "## Method",
        "",
        f"- Raw valid positive powers were median-aggregated into {config['bin_minutes']}-minute intervals.",
        "- Each UTC day and receiver was represented as a nine-frequency dynamic spectrum of log power.",
        "- Frequency rows were median-centered and scaled by the 75th percentile of positive residuals.",
        "- Sustained high-variance, near-zero-mean columns were inpainted for SVD and excluded from the occurrence denominator.",
        f"- Low-spectral-entropy modes were retained at normalized entropy < {PAPER_ENTROPY_LIMIT_NORMALIZED:.6f} "
        "(the paper's 3.0/log(90), dimension-scaled to nine RAE-2 bands).",
        f"- The strongest {100.0 * float(config['candidate_fraction']):.1f}% of positive SVD scores per valid receiver-day "
        "were flagged candidate-present. This fixes the approximate background rate and tests phase concentration.",
        "- Receive timestamps were corrected for Jupiter-Earth light time. Io phase is zero at superior conjunction; "
        "System III longitude is the west longitude of the Earth-facing central meridian.",
        "- Occurrence is candidate intervals divided by valid intervals in each 15-degree bin.",
        f"- Null bands and p-values use {config['n_permutations']} independent within-day circular rolls, preserving "
        "daily rates, data gaps, and burst autocorrelation while breaking the phase association.",
        "",
        "## Data accounting",
        "",
        f"- Aggregated receiver-frequency rows: {len(aggregated):,}",
        f"- Unique aggregated times: {aggregated['time'].nunique():,}",
        f"- Time span: {aggregated['time'].min()} to {aggregated['time'].max()}",
        f"- Paired Jupiter occulted intervals excluded: {len(intervals)} "
        f"({intervals['duration_s'].sum() / 60.0:.1f} total minutes)" if not intervals.empty else "- Paired Jupiter occulted intervals excluded: 0",
        f"- Paper-style transient-screen fraction by receiver: {transient_fraction}",
        "",
        "## Instrument limitation",
        "",
        "The paper separates occurrence by the sign of an imaginary cross-correlation proportional to Stokes V. "
        "RAE-2 RV1/RV2 contain independent total power only, so receiver differences cannot be interpreted as LHCP/RHCP. "
        "Solar and terrestrial bursts can enter the same statistic; receiver coincidence and circular-roll nulls mitigate but do not remove that ambiguity.",
        "",
        "## Outputs",
        "",
        *[f"- `{path.name}`" for path in plot_paths],
        "- `phase_occurrence_curves.csv`",
        "- `phase_occurrence_statistics.csv`",
        "- `io_cml_occurrence_map.csv`",
        "- `interval_candidate_scores.parquet`",
        "- `spice_geometry.parquet`",
    ]
    path = output / "RAE2_JUPITER_PHASE_OCCURRENCE_REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", type=Path, default=DEFAULT_CLEAN)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--kernel-dir", type=Path, default=None)
    parser.add_argument("--bin-minutes", type=int, default=5)
    parser.add_argument("--phase-bin-deg", type=float, default=15.0)
    parser.add_argument("--candidate-fraction", type=float, default=0.10)
    parser.add_argument("--min-channels", type=int, default=7)
    parser.add_argument("--min-day-intervals", type=int, default=48)
    parser.add_argument("--min-map-exposure", type=int, default=30)
    parser.add_argument("--n-permutations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=260825075)
    parser.add_argument("--refresh-kernels", action="store_true")
    parser.add_argument("--reuse-aggregated", action="store_true")
    args = parser.parse_args()

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    kernel_dir = args.kernel_dir or output / "spice_kernels"
    config = {
        "clean": str(args.clean),
        "events": str(args.events),
        "output_dir": str(output),
        "kernel_dir": str(kernel_dir),
        "bin_minutes": int(args.bin_minutes),
        "phase_bin_deg": float(args.phase_bin_deg),
        "candidate_fraction": float(args.candidate_fraction),
        "min_channels": int(args.min_channels),
        "min_day_intervals": int(args.min_day_intervals),
        "min_map_exposure": int(args.min_map_exposure),
        "n_permutations": int(args.n_permutations),
        "seed": int(args.seed),
        "entropy_limits_normalized": [float(value) for value in ENTROPY_SENSITIVITY],
        "kernel_urls": NAIF_KERNELS,
    }
    (output / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    aggregate_path = output / f"aggregated_raw_{int(args.bin_minutes)}min.parquet"
    if args.reuse_aggregated and aggregate_path.exists():
        aggregated = pd.read_parquet(aggregate_path)
        aggregated["time"] = pd.to_datetime(aggregated["time"])
    else:
        aggregated = aggregate_raw(args.clean, int(args.bin_minutes))
        aggregated.to_parquet(aggregate_path, index=False)

    scored = score_daily_svd(
        aggregated,
        ENTROPY_SENSITIVITY,
        float(args.candidate_fraction),
        int(args.min_channels),
        int(args.min_day_intervals),
    )
    if scored.empty:
        raise RuntimeError("No receiver-days passed the interval/channel coverage requirements")

    scored = add_coincident_detector(scored)
    kernels = ensure_kernels(kernel_dir, bool(args.refresh_kernels))
    geometry = compute_spice_geometry(pd.DatetimeIndex(scored["time"].unique()), kernels)
    geometry.to_parquet(output / "spice_geometry.parquet", index=False)
    scored = scored.merge(geometry, on="time", how="inner", validate="many_to_one")
    intervals = occultation_intervals(args.events)
    intervals.to_csv(output / "jupiter_occulted_intervals.csv", index=False)
    scored["jupiter_occulted"] = mark_occulted(scored["time"], intervals)
    scored["valid_interval"] = scored["valid_interval"].astype(bool)
    scored["candidate_present"] = scored["candidate_present"].astype(bool)
    scored.to_parquet(output / "interval_candidate_scores.parquet", index=False)

    curves, metrics = build_phase_products(
        scored, float(args.phase_bin_deg), int(args.n_permutations), int(args.seed)
    )
    curves.to_csv(output / "phase_occurrence_curves.csv", index=False)
    metrics.to_csv(output / "phase_occurrence_statistics.csv", index=False)
    maps = two_dimensional_occurrence(
        scored,
        PAPER_ENTROPY_LIMIT_NORMALIZED,
        float(args.phase_bin_deg),
        int(args.min_map_exposure),
    )
    maps.to_csv(output / "io_cml_occurrence_map.csv", index=False)

    plot_paths = [
        plot_figure4_analogue(curves, PAPER_ENTROPY_LIMIT_NORMALIZED, output),
        plot_coincident(curves, PAPER_ENTROPY_LIMIT_NORMALIZED, output),
        plot_exposure(curves, PAPER_ENTROPY_LIMIT_NORMALIZED, output),
        plot_2d_maps(maps, output),
    ]
    report = write_report(output, config, aggregated, scored, metrics, intervals, plot_paths)
    print(report)


if __name__ == "__main__":
    main()
