#!/usr/bin/env python
"""Cross-fitted low-rank foreground removal and covariance-aware Earth fits.

The edge is fixed at the ephemeris time (t=0).  Antennas are analysed
separately: no agreement requirement or shared-amplitude constraint is used.
Every learned foreground, covariance, and spectral-template quantity is fit
without the lunar block to which it is subsequently applied.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.linalg import cholesky, solve_triangular  # noqa: E402

from rylevonberg.occultogram import fold_occultogram_points  # noqa: E402
from rylevonberg.table_io import write_table  # noqa: E402


DEFAULT_PROFILES = Path(
    "/pscratch/sd/a/afahs/RAE2/occultograms_morphology_v1/occultogram_event_profiles.parquet"
)
DEFAULT_OUT = Path("/pscratch/sd/a/afahs/RAE2/earth_lowrank_covariance_v1")
TRACER_FREQUENCIES = (0.70, 0.90, 1.31, 2.20)
CLEAR_EARTH_FREQUENCIES = (0.45, 3.93, 6.55, 9.18)
BLANK_CONTROL_SHIFTS = (-1200.0, 1200.0)
FIT_HALF_WINDOW_SECONDS = 600.0
DISPLAY_HALF_WINDOW_SECONDS = 1200.0
RANK_CANDIDATES = (1, 2, 3)
INJECTION_LEVELS = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00)
TRACER_LEAKAGE_FRACTIONS = (0.0, 0.10, 0.25, 0.50)


def _robust_scale(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return np.nan
    center = float(np.median(values))
    scale = float(1.4826 * np.median(np.abs(values - center)))
    if not np.isfinite(scale) or scale <= 1e-10:
        scale = float(np.std(values, ddof=1))
    return scale


def _huber_fit(x: np.ndarray, y: np.ndarray, max_iter: int = 50) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    x = x[finite]
    y = y[finite]
    if len(y) < x.shape[1] + 2:
        return np.full(x.shape[1], np.nan)
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    for _ in range(max_iter):
        residual = y - x @ beta
        scale = _robust_scale(residual)
        if not np.isfinite(scale) or scale <= 1e-10:
            break
        u = np.abs(residual - np.median(residual)) / (1.345 * scale)
        weights = np.ones_like(u)
        large = u > 1.0
        weights[large] = 1.0 / u[large]
        root_weight = np.sqrt(weights)
        updated = np.linalg.lstsq(x * root_weight[:, None], y * root_weight, rcond=None)[0]
        if np.max(np.abs(updated - beta)) <= 1e-8 * (1.0 + np.max(np.abs(beta))):
            beta = updated
            break
        beta = updated
    return beta


def _edge_vector(times: np.ndarray) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    return np.where(times < 0.0, 0.5, np.where(times > 0.0, -0.5, 0.0))


def _edge_design(times: np.ndarray) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    return np.column_stack(
        [np.ones_like(times), times / FIT_HALF_WINDOW_SECONDS, _edge_vector(times)]
    )


def _edge_amplitude(times: np.ndarray, values: np.ndarray) -> float:
    beta = _huber_fit(_edge_design(times), np.asarray(values, dtype=float))
    return float(beta[2])


def _block_profiles(points: pd.DataFrame) -> pd.DataFrame:
    keys = ["lunar_block", "frequency_mhz", "antenna", "control_shift_s", "t_bin_sec"]
    return (
        points.groupby(keys, observed=True, sort=False)
        .agg(value=("normalized_power", "median"), n_events=("event_uid", "nunique"))
        .reset_index()
    )


def _wide_profiles(profiles: pd.DataFrame) -> pd.DataFrame:
    wide = profiles.pivot_table(
        index=["lunar_block", "antenna", "control_shift_s", "t_bin_sec"],
        columns="frequency_mhz",
        values="value",
        aggfunc="median",
    ).reset_index()
    wide.columns.name = None
    return wide.sort_values(
        ["lunar_block", "antenna", "control_shift_s", "t_bin_sec"]
    ).reset_index(drop=True)


def _frequency_column(columns: list[object], requested: float) -> float:
    matches = [float(column) for column in columns if isinstance(column, (float, int)) and np.isclose(column, requested)]
    if not matches:
        raise KeyError(f"frequency {requested:.2f} MHz is unavailable")
    return matches[0]


def _available_frequencies(wide: pd.DataFrame) -> tuple[float, ...]:
    metadata = {"lunar_block", "antenna", "control_shift_s", "t_bin_sec"}
    return tuple(sorted(float(column) for column in wide.columns if column not in metadata))


def _blank_rows(wide: pd.DataFrame, blocks: list[int], antenna: str) -> pd.DataFrame:
    block_values = pd.to_numeric(wide["lunar_block"], errors="coerce")
    shifts = pd.to_numeric(wide["control_shift_s"], errors="coerce").to_numpy(dtype=float)
    times = pd.to_numeric(wide["t_bin_sec"], errors="coerce").to_numpy(dtype=float)
    blank_shift = np.zeros(len(wide), dtype=bool)
    for shift in BLANK_CONTROL_SHIFTS:
        blank_shift |= np.isclose(shifts, shift)
    return wide[
        block_values.isin(blocks)
        & wide["antenna"].astype(str).eq(str(antenna))
        & blank_shift
        & (np.abs(times) <= FIT_HALF_WINDOW_SECONDS)
    ].copy()


@dataclass(frozen=True)
class TargetFactorModel:
    target_frequency: float
    tracer_frequencies: tuple[float, ...]
    center: np.ndarray
    scale: np.ndarray
    loadings: np.ndarray
    regression: np.ndarray
    rank: int

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        x = rows.loc[:, list(self.tracer_frequencies)].to_numpy(dtype=float)
        z = np.clip((x - self.center[None, :]) / self.scale[None, :], -8.0, 8.0)
        # Event-level profiles do not always contain every tracer in every
        # 60-second bin. Missing standardized values are conservatively set to
        # the training median (zero), so an unavailable tracer contributes no
        # inferred common-mode excursion instead of discarding the target bin.
        z[~np.isfinite(z)] = 0.0
        scores = z @ self.loadings
        design = np.column_stack([np.ones(len(rows)), scores])
        prediction = design @ self.regression
        return prediction


def _fit_target_factor(
    rows: pd.DataFrame,
    target_frequency: float,
    rank: int,
) -> TargetFactorModel:
    tracer_frequencies = tuple(
        frequency for frequency in TRACER_FREQUENCIES if not np.isclose(frequency, target_frequency)
    )
    rank = min(int(rank), len(tracer_frequencies))
    x = rows.loc[:, list(tracer_frequencies)].to_numpy(dtype=float)
    y = rows[target_frequency].to_numpy(dtype=float)
    complete = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    x = x[complete]
    y = y[complete]
    if len(y) < 10:
        raise RuntimeError(f"insufficient blank rows for {target_frequency:.2f} MHz factor model")
    center = np.median(x, axis=0)
    scale = np.asarray([_robust_scale(x[:, index]) for index in range(x.shape[1])])
    scale[~np.isfinite(scale) | (scale <= 1e-10)] = 1.0
    z = np.clip((x - center[None, :]) / scale[None, :], -8.0, 8.0)
    _, _, vt = np.linalg.svd(z, full_matrices=False)
    loadings = vt[:rank].T
    scores = z @ loadings
    regression = _huber_fit(np.column_stack([np.ones(len(scores)), scores]), y)
    return TargetFactorModel(
        target_frequency=float(target_frequency),
        tracer_frequencies=tracer_frequencies,
        center=center,
        scale=scale,
        loadings=loadings,
        regression=regression,
        rank=rank,
    )


def _fit_factor_family(
    wide: pd.DataFrame,
    blocks: list[int],
    antenna: str,
    rank: int,
    frequencies: tuple[float, ...],
) -> dict[float, TargetFactorModel]:
    rows = _blank_rows(wide, blocks, antenna)
    return {
        frequency: _fit_target_factor(rows, frequency, rank)
        for frequency in frequencies
    }


def _factor_validation_loss(
    wide: pd.DataFrame,
    training_blocks: list[int],
    antenna: str,
    rank: int,
    frequencies: tuple[float, ...],
) -> float:
    detection_frequencies = tuple(
        frequency
        for frequency in frequencies
        if any(np.isclose(frequency, clear) for clear in CLEAR_EARTH_FREQUENCIES)
        or np.isclose(frequency, 4.70)
    )
    errors: list[float] = []
    for validation_block in training_blocks:
        inner_blocks = [block for block in training_blocks if block != validation_block]
        if len(inner_blocks) < 2:
            continue
        family = _fit_factor_family(wide, inner_blocks, antenna, rank, frequencies)
        validation = _blank_rows(wide, [validation_block], antenna)
        training = _blank_rows(wide, inner_blocks, antenna)
        for frequency in detection_frequencies:
            prediction = family[frequency].predict(validation)
            observed = validation[frequency].to_numpy(dtype=float)
            train_scale = _robust_scale(training[frequency].to_numpy(dtype=float))
            if not np.isfinite(train_scale) or train_scale <= 1e-10:
                continue
            finite = np.isfinite(prediction) & np.isfinite(observed)
            errors.extend((np.abs(observed[finite] - prediction[finite]) / train_scale).tolist())
    return float(np.median(errors)) if errors else np.inf


def _select_factor_rank(
    wide: pd.DataFrame,
    training_blocks: list[int],
    antenna: str,
    frequencies: tuple[float, ...],
) -> tuple[int, list[dict[str, object]]]:
    records: list[dict[str, object]] = []
    for rank in RANK_CANDIDATES:
        loss = _factor_validation_loss(wide, training_blocks, antenna, rank, frequencies)
        records.append({"rank": int(rank), "validation_nmae": float(loss)})
    best = min(records, key=lambda record: (record["validation_nmae"], record["rank"]))
    return int(best["rank"]), records


def _crossfit_lowrank(
    wide: pd.DataFrame,
    frequencies: tuple[float, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[int, str], dict[float, TargetFactorModel]]]:
    blocks = sorted(int(value) for value in pd.unique(wide["lunar_block"]))
    antennas = sorted(str(value) for value in pd.unique(wide["antenna"]))
    transformed: list[pd.DataFrame] = []
    rank_records: list[dict[str, object]] = []
    families: dict[tuple[int, str], dict[float, TargetFactorModel]] = {}
    for block in blocks:
        training_blocks = [candidate for candidate in blocks if candidate != block]
        for antenna in antennas:
            rank, losses = _select_factor_rank(wide, training_blocks, antenna, frequencies)
            for record in losses:
                rank_records.append(
                    {
                        "held_out_lunar_block": block,
                        "antenna": antenna,
                        **record,
                        "selected": bool(record["rank"] == rank),
                    }
                )
            family = _fit_factor_family(wide, training_blocks, antenna, rank, frequencies)
            families[(block, antenna)] = family
            current = wide[
                wide["lunar_block"].eq(block) & wide["antenna"].astype(str).eq(antenna)
            ].copy()
            metadata = current.loc[:, ["lunar_block", "antenna", "control_shift_s", "t_bin_sec"]]
            for frequency in frequencies:
                prediction = family[frequency].predict(current)
                frame = metadata.copy()
                frame["frequency_mhz"] = float(frequency)
                frame["raw_value"] = current[frequency].to_numpy(dtype=float)
                frame["foreground_prediction"] = prediction
                frame["residual_value"] = frame["raw_value"] - prediction
                frame["selected_rank"] = int(rank)
                transformed.append(frame)
    return (
        pd.concat(transformed, ignore_index=True),
        pd.DataFrame.from_records(rank_records),
        families,
    )


def _residual_wide(lowrank: pd.DataFrame) -> pd.DataFrame:
    wide = lowrank.pivot_table(
        index=["lunar_block", "antenna", "control_shift_s", "t_bin_sec"],
        columns="frequency_mhz",
        values="residual_value",
        aggfunc="median",
    ).reset_index()
    wide.columns.name = None
    return wide.sort_values(
        ["lunar_block", "antenna", "control_shift_s", "t_bin_sec"]
    ).reset_index(drop=True)


def _profile_matrix(
    wide: pd.DataFrame,
    block: int,
    antenna: str,
    shift: float,
    frequencies: tuple[float, ...],
    times: np.ndarray,
) -> np.ndarray:
    rows = wide[
        wide["lunar_block"].eq(block)
        & wide["antenna"].astype(str).eq(antenna)
        & np.isclose(pd.to_numeric(wide["control_shift_s"], errors="coerce"), shift)
    ].set_index("t_bin_sec")
    matrix = rows.reindex(times).loc[:, list(frequencies)].to_numpy(dtype=float)
    if np.any(~np.isfinite(matrix)):
        matrix = (
            pd.DataFrame(matrix, index=times)
            .interpolate(method="index", limit_direction="both")
            .to_numpy(dtype=float)
        )
    if np.any(~np.isfinite(matrix)):
        raise RuntimeError(
            f"incomplete profile matrix for block={block}, antenna={antenna}, shift={shift}"
        )
    return matrix


def _oas_covariance(samples: np.ndarray) -> tuple[np.ndarray, float]:
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.all(np.isfinite(samples), axis=1)]
    if samples.shape[0] < 2:
        raise RuntimeError("at least two complete samples are required for covariance estimation")
    centered = samples - np.mean(samples, axis=0, keepdims=True)
    covariance = centered.T @ centered / float(len(centered))
    dimension = covariance.shape[0]
    mu = float(np.trace(covariance) / dimension)
    trace_square = float(np.sum(covariance * covariance))
    denominator = (len(centered) + 1.0 - 2.0 / dimension) * (
        trace_square - (float(np.trace(covariance)) ** 2) / dimension
    )
    numerator = (1.0 - 2.0 / dimension) * trace_square + float(np.trace(covariance)) ** 2
    shrinkage = 1.0 if denominator <= 1e-15 else float(np.clip(numerator / denominator, 0.0, 1.0))
    shrunk = (1.0 - shrinkage) * covariance + shrinkage * mu * np.eye(dimension)
    floor = max(mu, 1.0) * 1e-8
    shrunk += floor * np.eye(dimension)
    return shrunk, shrinkage


@dataclass(frozen=True)
class SeparableCovariance:
    time_covariance: np.ndarray
    frequency_covariance: np.ndarray
    time_shrinkage: float
    frequency_shrinkage: float
    residual_scales: np.ndarray
    n_control_profiles: int


def _estimate_covariance(
    wide: pd.DataFrame,
    training_blocks: list[int],
    antenna: str,
    frequencies: tuple[float, ...],
    times: np.ndarray,
) -> SeparableCovariance:
    residuals: list[np.ndarray] = []
    nuisance = _edge_design(times)[:, :2]
    for block in training_blocks:
        for shift in BLANK_CONTROL_SHIFTS:
            matrix = _profile_matrix(wide, block, antenna, shift, frequencies, times)
            fitted = nuisance @ np.linalg.lstsq(nuisance, matrix, rcond=None)[0]
            residuals.append(matrix - fitted)
    if len(residuals) < 4:
        raise RuntimeError("insufficient cross-fitted blank profiles for covariance estimation")
    stack = np.stack(residuals)
    scales = np.asarray([_robust_scale(stack[:, :, index].ravel()) for index in range(len(frequencies))])
    scales[~np.isfinite(scales) | (scales <= 1e-10)] = 1.0
    standardized = stack / scales[None, None, :]
    time_samples = np.concatenate([matrix.T for matrix in standardized], axis=0)
    frequency_samples = np.concatenate([matrix for matrix in standardized], axis=0)
    time_covariance, time_shrinkage = _oas_covariance(time_samples)
    time_covariance /= float(np.trace(time_covariance) / len(times))
    frequency_correlation, frequency_shrinkage = _oas_covariance(frequency_samples)
    frequency_covariance = (
        np.diag(scales) @ frequency_correlation @ np.diag(scales)
    )
    return SeparableCovariance(
        time_covariance=time_covariance,
        frequency_covariance=frequency_covariance,
        time_shrinkage=float(time_shrinkage),
        frequency_shrinkage=float(frequency_shrinkage),
        residual_scales=scales,
        n_control_profiles=len(residuals),
    )


def _student_t_linear_fit(
    y: np.ndarray,
    design: np.ndarray,
    covariance: np.ndarray,
    degrees_freedom: float = 4.0,
    max_iter: int = 40,
) -> tuple[np.ndarray, np.ndarray, float]:
    covariance = np.asarray(covariance, dtype=float)
    lower = cholesky(covariance, lower=True, check_finite=False)
    whitened_y = solve_triangular(lower, np.asarray(y, dtype=float), lower=True, check_finite=False)
    whitened_x = solve_triangular(lower, np.asarray(design, dtype=float), lower=True, check_finite=False)
    beta = np.linalg.lstsq(whitened_x, whitened_y, rcond=None)[0]
    weights = np.ones(len(whitened_y), dtype=float)
    scale = 1.0
    for _ in range(max_iter):
        residual = whitened_y - whitened_x @ beta
        scale = _robust_scale(residual)
        if not np.isfinite(scale) or scale <= 1e-10:
            scale = 1.0
        standardized = residual / scale
        weights = (degrees_freedom + 1.0) / (degrees_freedom + standardized**2)
        root_weight = np.sqrt(weights)
        updated = np.linalg.lstsq(
            whitened_x * root_weight[:, None], whitened_y * root_weight, rcond=None
        )[0]
        if np.max(np.abs(updated - beta)) <= 1e-8 * (1.0 + np.max(np.abs(beta))):
            beta = updated
            break
        beta = updated
    precision = whitened_x.T @ (weights[:, None] * whitened_x)
    parameter_covariance = np.linalg.pinv(precision) * scale**2
    return beta, parameter_covariance, float(scale)


def _covariance_for_frequencies(
    model: SeparableCovariance,
    all_frequencies: tuple[float, ...],
    selected_frequencies: tuple[float, ...],
) -> np.ndarray:
    indices = [
        next(index for index, value in enumerate(all_frequencies) if np.isclose(value, frequency))
        for frequency in selected_frequencies
    ]
    frequency_covariance = model.frequency_covariance[np.ix_(indices, indices)]
    return np.kron(frequency_covariance, model.time_covariance)


def _free_covariance_fit(
    matrix: np.ndarray,
    times: np.ndarray,
    model: SeparableCovariance,
) -> tuple[np.ndarray, np.ndarray]:
    frequencies = matrix.shape[1]
    per_frequency = _edge_design(times)
    design = np.kron(np.eye(frequencies), per_frequency)
    covariance = np.kron(model.frequency_covariance, model.time_covariance)
    beta, parameter_covariance, _ = _student_t_linear_fit(
        matrix.reshape(-1, order="F"), design, covariance
    )
    amplitude_indices = np.arange(frequencies) * 3 + 2
    return beta[amplitude_indices], np.sqrt(np.maximum(np.diag(parameter_covariance)[amplitude_indices], 0.0))


def _joint_covariance_fit(
    matrix: np.ndarray,
    times: np.ndarray,
    covariance: np.ndarray,
    spectral_template: np.ndarray,
) -> tuple[float, float]:
    frequency_count = matrix.shape[1]
    nuisance = _edge_design(times)[:, :2]
    nuisance_design = np.kron(np.eye(frequency_count), nuisance)
    template_column = np.concatenate(
        [float(amplitude) * _edge_vector(times) for amplitude in spectral_template]
    )
    design = np.column_stack([nuisance_design, template_column])
    beta, parameter_covariance, _ = _student_t_linear_fit(
        matrix.reshape(-1, order="F"), design, covariance
    )
    return float(beta[-1]), float(np.sqrt(max(parameter_covariance[-1, -1], 0.0)))


def _raw_spectral_template(
    raw_wide: pd.DataFrame,
    training_blocks: list[int],
    antenna: str,
    clear_frequencies: tuple[float, ...],
    times: np.ndarray,
) -> np.ndarray:
    amplitudes: list[np.ndarray] = []
    for block in training_blocks:
        matrix = _profile_matrix(raw_wide, block, antenna, 0.0, clear_frequencies, times)
        amplitudes.append(
            np.asarray([_edge_amplitude(times, matrix[:, index]) for index in range(len(clear_frequencies))])
        )
    template = np.nanmedian(np.stack(amplitudes), axis=0)
    positive = template[np.isfinite(template) & (template > 0.0)]
    floor = 0.05 * float(np.median(positive)) if positive.size else 1e-3
    template[~np.isfinite(template) | (template <= floor)] = floor
    return template


def _baseline_joint_score(
    matrix: np.ndarray,
    times: np.ndarray,
    spectral_template: np.ndarray,
) -> tuple[float, np.ndarray]:
    amplitudes = np.asarray(
        [_edge_amplitude(times, matrix[:, index]) for index in range(matrix.shape[1])]
    )
    ratios = amplitudes / spectral_template
    return float(np.median(ratios[np.isfinite(ratios)])), amplitudes


def _lowrank_transfer(
    families: dict[tuple[int, str], dict[float, TargetFactorModel]],
    blocks: list[int],
    antennas: list[str],
    frequencies: tuple[float, ...],
    times: np.ndarray,
    raw_templates: dict[tuple[int, str], np.ndarray],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    clear_indices = {
        frequency: index for index, frequency in enumerate(CLEAR_EARTH_FREQUENCIES)
    }
    median_clear_by_fold = {
        key: float(np.median(value)) for key, value in raw_templates.items()
    }
    for block in blocks:
        for antenna in antennas:
            family = families[(block, antenna)]
            clear_template = raw_templates[(block, antenna)]
            for leakage_fraction in TRACER_LEAKAGE_FRACTIONS:
                injection = pd.DataFrame(0.0, index=np.arange(len(times)), columns=list(frequencies))
                edge = _edge_vector(times)
                for frequency, index in clear_indices.items():
                    injection[frequency] = edge * clear_template[index]
                tracer_amplitude = leakage_fraction * median_clear_by_fold[(block, antenna)]
                for frequency in TRACER_FREQUENCIES:
                    injection[frequency] = edge * tracer_amplitude
                for frequency in CLEAR_EARTH_FREQUENCIES:
                    prediction_change = family[frequency].predict(injection)
                    residual_signal = injection[frequency].to_numpy(dtype=float) - prediction_change
                    recovered = _edge_amplitude(times, residual_signal)
                    input_amplitude = clear_template[clear_indices[frequency]]
                    records.append(
                        {
                            "held_out_lunar_block": block,
                            "antenna": antenna,
                            "frequency_mhz": frequency,
                            "tracer_leakage_fraction": float(leakage_fraction),
                            "input_edge_amplitude": float(input_amplitude),
                            "recovered_edge_amplitude": float(recovered),
                            "retained_fraction": float(recovered / input_amplitude),
                        }
                    )
    return pd.DataFrame.from_records(records)


def _run_block_fits_and_injections(
    raw_wide: pd.DataFrame,
    residual_wide: pd.DataFrame,
    frequencies: tuple[float, ...],
    times: np.ndarray,
    injection_levels: tuple[float, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[tuple[int, str], np.ndarray]]:
    blocks = sorted(int(value) for value in pd.unique(raw_wide["lunar_block"]))
    antennas = sorted(str(value) for value in pd.unique(raw_wide["antenna"]))
    clear = tuple(_frequency_column(list(frequencies), value) for value in CLEAR_EARTH_FREQUENCIES)
    clear_indices = [
        next(index for index, value in enumerate(frequencies) if np.isclose(value, selected))
        for selected in clear
    ]
    fit_records: list[dict[str, object]] = []
    injection_records: list[dict[str, object]] = []
    covariance_records: list[dict[str, object]] = []
    template_records: list[dict[str, object]] = []
    raw_templates: dict[tuple[int, str], np.ndarray] = {}

    for block in blocks:
        training_blocks = [candidate for candidate in blocks if candidate != block]
        for antenna in antennas:
            template = _raw_spectral_template(
                raw_wide, training_blocks, antenna, clear, times
            )
            raw_templates[(block, antenna)] = template
            for frequency, amplitude in zip(clear, template):
                template_records.append(
                    {
                        "held_out_lunar_block": block,
                        "antenna": antenna,
                        "frequency_mhz": frequency,
                        "template_edge_amplitude": float(amplitude),
                    }
                )
            models: dict[str, SeparableCovariance] = {}
            for data_kind, data_wide in (("raw", raw_wide), ("lowrank", residual_wide)):
                model = _estimate_covariance(
                    data_wide, training_blocks, antenna, frequencies, times
                )
                models[data_kind] = model
                time_corr = model.time_covariance / np.sqrt(
                    np.outer(np.diag(model.time_covariance), np.diag(model.time_covariance))
                )
                freq_corr = model.frequency_covariance / np.sqrt(
                    np.outer(np.diag(model.frequency_covariance), np.diag(model.frequency_covariance))
                )
                for row in range(len(times)):
                    for column in range(len(times)):
                        covariance_records.append(
                            {
                                "held_out_lunar_block": block,
                                "antenna": antenna,
                                "data_kind": data_kind,
                                "matrix_kind": "time",
                                "row_label": float(times[row]),
                                "column_label": float(times[column]),
                                "covariance": float(model.time_covariance[row, column]),
                                "correlation": float(time_corr[row, column]),
                                "shrinkage": model.time_shrinkage,
                                "n_control_profiles": model.n_control_profiles,
                            }
                        )
                for row, row_frequency in enumerate(frequencies):
                    for column, column_frequency in enumerate(frequencies):
                        covariance_records.append(
                            {
                                "held_out_lunar_block": block,
                                "antenna": antenna,
                                "data_kind": data_kind,
                                "matrix_kind": "frequency",
                                "row_label": float(row_frequency),
                                "column_label": float(column_frequency),
                                "covariance": float(model.frequency_covariance[row, column]),
                                "correlation": float(freq_corr[row, column]),
                                "shrinkage": model.frequency_shrinkage,
                                "n_control_profiles": model.n_control_profiles,
                            }
                        )

            all_shifts = sorted(
                float(value)
                for value in pd.unique(
                    raw_wide[
                        raw_wide["lunar_block"].eq(block)
                        & raw_wide["antenna"].astype(str).eq(antenna)
                    ]["control_shift_s"]
                )
            )
            for shift in all_shifts:
                matrices = {
                    "raw": _profile_matrix(raw_wide, block, antenna, shift, frequencies, times),
                    "lowrank": _profile_matrix(
                        residual_wide, block, antenna, shift, frequencies, times
                    ),
                }
                for data_kind, matrix in matrices.items():
                    clear_matrix = matrix[:, clear_indices]
                    baseline_score, baseline_amplitudes = _baseline_joint_score(
                        clear_matrix, times, template
                    )
                    fit_records.append(
                        {
                            "held_out_lunar_block": block,
                            "antenna": antenna,
                            "control_shift_s": shift,
                            "data_kind": data_kind,
                            "fit_kind": "fixed_zero_huber_joint_score",
                            "frequency_mhz": np.nan,
                            "estimate": baseline_score,
                            "standard_error": np.nan,
                        }
                    )
                    covariance = _covariance_for_frequencies(
                        models[data_kind], frequencies, clear
                    )
                    joint_estimate, joint_error = _joint_covariance_fit(
                        clear_matrix, times, covariance, template
                    )
                    fit_records.append(
                        {
                            "held_out_lunar_block": block,
                            "antenna": antenna,
                            "control_shift_s": shift,
                            "data_kind": data_kind,
                            "fit_kind": "joint_student_t_gls",
                            "frequency_mhz": np.nan,
                            "estimate": joint_estimate,
                            "standard_error": joint_error,
                        }
                    )
                    free_amplitudes, free_errors = _free_covariance_fit(
                        matrix, times, models[data_kind]
                    )
                    for frequency, estimate, error in zip(
                        frequencies, free_amplitudes, free_errors
                    ):
                        fit_records.append(
                            {
                                "held_out_lunar_block": block,
                                "antenna": antenna,
                                "control_shift_s": shift,
                                "data_kind": data_kind,
                                "fit_kind": "free_student_t_gls",
                                "frequency_mhz": frequency,
                                "estimate": float(estimate),
                                "standard_error": float(error),
                            }
                        )
                    all_huber_amplitudes = np.asarray(
                        [
                            _edge_amplitude(times, matrix[:, frequency_index])
                            for frequency_index in range(len(frequencies))
                        ]
                    )
                    for frequency, estimate in zip(frequencies, all_huber_amplitudes):
                        fit_records.append(
                            {
                                "held_out_lunar_block": block,
                                "antenna": antenna,
                                "control_shift_s": shift,
                                "data_kind": data_kind,
                                "fit_kind": "free_huber",
                                "frequency_mhz": frequency,
                                "estimate": float(estimate),
                                "standard_error": np.nan,
                            }
                        )

            for shift in BLANK_CONTROL_SHIFTS:
                raw_base = _profile_matrix(raw_wide, block, antenna, shift, frequencies, times)
                lowrank_base = _profile_matrix(
                    residual_wide, block, antenna, shift, frequencies, times
                )
                injection = np.zeros_like(raw_base)
                for clear_index, all_index in enumerate(clear_indices):
                    injection[:, all_index] = _edge_vector(times) * template[clear_index]
                for level in injection_levels:
                    raw_injected = raw_base + level * injection
                    lowrank_injected = lowrank_base + level * injection
                    for method, matrix, model_kind, estimator in (
                        ("raw_fixed_zero", raw_injected, "raw", "baseline"),
                        ("lowrank_fixed_zero", lowrank_injected, "lowrank", "baseline"),
                        ("raw_covariance_student_t", raw_injected, "raw", "covariance"),
                        (
                            "lowrank_covariance_student_t",
                            lowrank_injected,
                            "lowrank",
                            "covariance",
                        ),
                    ):
                        clear_matrix = matrix[:, clear_indices]
                        if estimator == "baseline":
                            score, _ = _baseline_joint_score(clear_matrix, times, template)
                        else:
                            covariance = _covariance_for_frequencies(
                                models[model_kind], frequencies, clear
                            )
                            score, _ = _joint_covariance_fit(
                                clear_matrix, times, covariance, template
                            )
                        injection_records.append(
                            {
                                "held_out_lunar_block": block,
                                "antenna": antenna,
                                "control_shift_s": float(shift),
                                "injection_level": float(level),
                                "method": method,
                                "block_score": float(score),
                            }
                        )
    return (
        pd.DataFrame.from_records(fit_records),
        pd.DataFrame.from_records(injection_records),
        pd.DataFrame.from_records(covariance_records),
        pd.DataFrame.from_records(template_records),
        raw_templates,
    )


def _frequency_fit_summary(fits: pd.DataFrame) -> pd.DataFrame:
    selected = fits[
        np.isclose(fits["control_shift_s"], 0.0)
        & fits["frequency_mhz"].notna()
        & fits["fit_kind"].isin(["free_huber", "free_student_t_gls"])
    ].copy()
    records: list[dict[str, object]] = []
    keys = ["antenna", "data_kind", "fit_kind", "frequency_mhz"]
    for group_key, group in selected.groupby(keys, observed=True, sort=True):
        antenna, data_kind, fit_kind, frequency = group_key
        values = pd.to_numeric(group["estimate"], errors="coerce").dropna()
        records.append(
            {
                "antenna": str(antenna),
                "data_kind": str(data_kind),
                "fit_kind": str(fit_kind),
                "frequency_mhz": float(frequency),
                "target_q16": float(values.quantile(0.16)),
                "target_median": float(values.median()),
                "target_q84": float(values.quantile(0.84)),
                "n_lunar_blocks": int(values.size),
            }
        )
    return pd.DataFrame.from_records(records)


def _method_name(data_kind: str, fit_kind: str) -> str | None:
    mapping = {
        ("raw", "fixed_zero_huber_joint_score"): "raw_fixed_zero",
        ("lowrank", "fixed_zero_huber_joint_score"): "lowrank_fixed_zero",
        ("raw", "joint_student_t_gls"): "raw_covariance_student_t",
        ("lowrank", "joint_student_t_gls"): "lowrank_covariance_student_t",
    }
    return mapping.get((str(data_kind), str(fit_kind)))


def _summarize_methods(
    fits: pd.DataFrame,
    injection_scores: pd.DataFrame,
    injection_trials: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    joint = fits[fits["frequency_mhz"].isna()].copy()
    joint["method"] = [
        _method_name(data_kind, fit_kind)
        for data_kind, fit_kind in zip(joint["data_kind"], joint["fit_kind"])
    ]
    joint = joint[joint["method"].notna()]
    summary_records: list[dict[str, object]] = []
    power_records: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)

    for antenna in sorted(joint["antenna"].astype(str).unique()):
        antenna_injections = injection_scores[injection_scores["antenna"].astype(str).eq(antenna)]
        blocks = sorted(int(value) for value in antenna_injections["held_out_lunar_block"].unique())
        possible_combinations = len(BLANK_CONTROL_SHIFTS) ** len(blocks)
        if possible_combinations <= injection_trials:
            combination_ids = np.arange(possible_combinations, dtype=np.int64)
            choice_indices = np.column_stack(
                [
                    (combination_ids // (len(BLANK_CONTROL_SHIFTS) ** block_index))
                    % len(BLANK_CONTROL_SHIFTS)
                    for block_index in range(len(blocks))
                ]
            )
            combination_mode = "exhaustive"
        else:
            choice_indices = rng.integers(
                0, len(BLANK_CONTROL_SHIFTS), size=(injection_trials, len(blocks))
            )
            combination_mode = "random_sample"
        for method in sorted(joint["method"].unique()):
            method_fits = joint[
                joint["antenna"].astype(str).eq(antenna) & joint["method"].eq(method)
            ]
            target = method_fits[np.isclose(method_fits["control_shift_s"], 0.0)].set_index(
                "held_out_lunar_block"
            )["estimate"].reindex(blocks).to_numpy(dtype=float)
            observed = float(np.median(target))
            method_injections = antenna_injections[antenna_injections["method"].eq(method)]
            level_statistics: dict[float, np.ndarray] = {}
            for level in INJECTION_LEVELS:
                selected_level = method_injections[np.isclose(method_injections["injection_level"], level)]
                cube = np.empty((len(blocks), len(BLANK_CONTROL_SHIFTS)), dtype=float)
                for block_index, block in enumerate(blocks):
                    for shift_index, shift in enumerate(BLANK_CONTROL_SHIFTS):
                        value = selected_level[
                            selected_level["held_out_lunar_block"].eq(block)
                            & np.isclose(selected_level["control_shift_s"], shift)
                        ]["block_score"]
                        if len(value) != 1:
                            raise RuntimeError("injection score grid is incomplete")
                        cube[block_index, shift_index] = float(value.iloc[0])
                chosen = cube[np.arange(len(blocks))[None, :], choice_indices]
                level_statistics[float(level)] = np.median(chosen, axis=1)
            null = level_statistics[0.0]
            threshold_95 = float(np.quantile(null, 0.95))
            threshold_99 = float(np.quantile(null, 0.99))
            summary_records.append(
                {
                    "antenna": antenna,
                    "method": method,
                    "target_score_median": observed,
                    "target_block_q16": float(np.quantile(target, 0.16)),
                    "target_block_q84": float(np.quantile(target, 0.84)),
                    "null_q16": float(np.quantile(null, 0.16)),
                    "null_median": float(np.median(null)),
                    "null_q84": float(np.quantile(null, 0.84)),
                    "null_q95": threshold_95,
                    "null_q99": threshold_99,
                    "target_percentile": float(100.0 * (1 + np.sum(null <= observed)) / (len(null) + 1)),
                    "one_sided_empirical_p": float((1 + np.sum(null >= observed)) / (len(null) + 1)),
                    "n_lunar_blocks": len(blocks),
                    "control_combination_mode": combination_mode,
                    "control_combinations_evaluated": int(len(choice_indices)),
                    "control_combinations_possible": int(possible_combinations),
                }
            )
            for level, statistics in level_statistics.items():
                power_records.append(
                    {
                        "antenna": antenna,
                        "method": method,
                        "injection_level": float(level),
                        "power_at_95pct_null": float(np.mean(statistics > threshold_95)),
                        "power_at_99pct_null": float(np.mean(statistics > threshold_99)),
                        "score_q16": float(np.quantile(statistics, 0.16)),
                        "score_median": float(np.median(statistics)),
                        "score_q84": float(np.quantile(statistics, 0.84)),
                        "control_combination_mode": combination_mode,
                        "control_combinations_evaluated": int(len(choice_indices)),
                    }
                )
    return pd.DataFrame.from_records(summary_records), pd.DataFrame.from_records(power_records)


def _plot_lowrank_occultograms(
    raw_wide: pd.DataFrame,
    residual_wide: pd.DataFrame,
    frequencies: tuple[float, ...],
    output_dir: Path,
) -> list[Path]:
    paths: list[Path] = []
    for antenna in sorted(raw_wide["antenna"].astype(str).unique()):
        raw = raw_wide[
            raw_wide["antenna"].astype(str).eq(antenna)
            & np.isclose(raw_wide["control_shift_s"], 0.0)
            & (np.abs(raw_wide["t_bin_sec"]) <= DISPLAY_HALF_WINDOW_SECONDS)
        ]
        residual = residual_wide[
            residual_wide["antenna"].astype(str).eq(antenna)
            & np.isclose(residual_wide["control_shift_s"], 0.0)
            & (np.abs(residual_wide["t_bin_sec"]) <= DISPLAY_HALF_WINDOW_SECONDS)
        ]
        fig, axes = plt.subplots(
            len(frequencies), 2, figsize=(13, 2.15 * len(frequencies)), constrained_layout=True
        )
        for row, frequency in enumerate(frequencies):
            raw_grid = raw.pivot_table(
                index="lunar_block", columns="t_bin_sec", values=frequency, aggfunc="median"
            ).sort_index()
            residual_grid = residual.pivot_table(
                index="lunar_block", columns="t_bin_sec", values=frequency, aggfunc="median"
            ).reindex(index=raw_grid.index, columns=raw_grid.columns)
            combined = np.concatenate([raw_grid.to_numpy().ravel(), residual_grid.to_numpy().ravel()])
            finite = combined[np.isfinite(combined)]
            limit = float(np.quantile(np.abs(finite), 0.98)) if finite.size else 1.0
            limit = max(limit, 1e-6)
            extent = [
                float(raw_grid.columns.min()),
                float(raw_grid.columns.max()),
                len(raw_grid.index) - 0.5,
                -0.5,
            ]
            for column, (grid, title) in enumerate(
                ((raw_grid, "raw"), (residual_grid, "low-rank residual"))
            ):
                ax = axes[row, column]
                ax.imshow(
                    grid.to_numpy(dtype=float),
                    aspect="auto",
                    interpolation="nearest",
                    cmap="RdBu_r",
                    vmin=-limit,
                    vmax=limit,
                    extent=extent,
                )
                ax.axvline(0.0, color="black", linestyle="--", linewidth=0.8)
                ax.set_yticks(range(len(raw_grid.index)), [str(value) for value in raw_grid.index])
                if column == 0:
                    ax.set_ylabel(f"{frequency:.2f} MHz\nlunar block")
                else:
                    ax.set_yticklabels([])
                if row == 0:
                    ax.set_title(title)
                if row == len(frequencies) - 1:
                    ax.set_xlabel("time from fixed edge (s)")
                else:
                    ax.set_xticklabels([])
        fig.suptitle(
            f"Earth {antenna}: cross-fitted common-mode removal (t=0 fixed)", fontsize=14
        )
        path = output_dir / f"earth_{antenna}_lowrank_occultograms.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def _plot_lowrank_profiles(
    raw_wide: pd.DataFrame,
    residual_wide: pd.DataFrame,
    frequencies: tuple[float, ...],
    output_dir: Path,
) -> list[Path]:
    paths: list[Path] = []
    for antenna in sorted(raw_wide["antenna"].astype(str).unique()):
        raw = raw_wide[
            raw_wide["antenna"].astype(str).eq(antenna)
            & np.isclose(raw_wide["control_shift_s"], 0.0)
            & (np.abs(raw_wide["t_bin_sec"]) <= DISPLAY_HALF_WINDOW_SECONDS)
        ]
        residual = residual_wide[
            residual_wide["antenna"].astype(str).eq(antenna)
            & np.isclose(residual_wide["control_shift_s"], 0.0)
            & (np.abs(residual_wide["t_bin_sec"]) <= DISPLAY_HALF_WINDOW_SECONDS)
        ]
        fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharex=True, constrained_layout=True)
        for ax, frequency in zip(axes.ravel(), frequencies):
            raw_summary = raw.groupby("t_bin_sec", observed=True)[frequency].median().sort_index()
            residual_group = residual.groupby("t_bin_sec", observed=True)[frequency]
            residual_median = residual_group.median().sort_index()
            residual_q16 = residual_group.quantile(0.16).reindex(residual_median.index)
            residual_q84 = residual_group.quantile(0.84).reindex(residual_median.index)
            t = residual_median.index.to_numpy(dtype=float)
            ax.plot(raw_summary.index, raw_summary.values, color="0.55", linewidth=1.1, label="raw")
            ax.plot(t, residual_median.values, color="#7b2cbf", linewidth=1.7, label="low-rank residual")
            ax.fill_between(t, residual_q16.values, residual_q84.values, color="#7b2cbf", alpha=0.14)
            ax.axvline(0.0, color="black", linestyle="--", linewidth=0.8)
            ax.axhline(0.0, color="0.4", linewidth=0.6)
            ax.set_title(f"{frequency:.2f} MHz")
            ax.grid(alpha=0.17)
        axes[0, 0].legend(frameon=False)
        for ax in axes[-1, :]:
            ax.set_xlabel("time from fixed edge (s)")
        for ax in axes[:, 0]:
            ax.set_ylabel("normalized power")
        fig.suptitle(f"Earth {antenna}: raw and common-mode-residual profiles")
        path = output_dir / f"earth_{antenna}_lowrank_profiles.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def _plot_rank_and_transfer(
    rank_selection: pd.DataFrame,
    transfer: pd.DataFrame,
    output: Path,
) -> None:
    antennas = sorted(rank_selection["antenna"].astype(str).unique())
    fig, axes = plt.subplots(2, len(antennas), figsize=(12, 8), constrained_layout=True)
    axes = np.asarray(axes).reshape(2, len(antennas))
    for column, antenna in enumerate(antennas):
        rank_data = rank_selection[rank_selection["antenna"].astype(str).eq(antenna)]
        summary = rank_data.groupby("rank", observed=True)["validation_nmae"].agg(["median", "min", "max"])
        axes[0, column].plot(summary.index, summary["median"], marker="o", color="#1f77b4")
        axes[0, column].fill_between(summary.index, summary["min"], summary["max"], color="#1f77b4", alpha=0.15)
        axes[0, column].set_xticks(list(RANK_CANDIDATES))
        axes[0, column].set_xlabel("foreground rank")
        axes[0, column].set_ylabel("held-out normalized MAE")
        axes[0, column].set_title(antenna)
        axes[0, column].grid(alpha=0.2)

        transfer_data = transfer[transfer["antenna"].astype(str).eq(antenna)]
        for frequency in CLEAR_EARTH_FREQUENCIES:
            current = transfer_data[np.isclose(transfer_data["frequency_mhz"], frequency)]
            grouped = current.groupby("tracer_leakage_fraction", observed=True)["retained_fraction"].median()
            axes[1, column].plot(grouped.index, grouped.values, marker="o", label=f"{frequency:.2f}")
        axes[1, column].axhline(1.0, color="0.3", linestyle="--", linewidth=0.8)
        axes[1, column].set_xlabel("injected tracer leakage / median clear-band edge")
        axes[1, column].set_ylabel("fraction of clear-band edge retained")
        axes[1, column].set_ylim(-0.1, 1.25)
        axes[1, column].grid(alpha=0.2)
    axes[1, 0].legend(title="MHz", frameon=False, ncol=2)
    fig.suptitle("Low-rank selection and signal-preservation stress test")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_covariance(covariances: pd.DataFrame, output: Path) -> None:
    antennas = sorted(covariances["antenna"].astype(str).unique())
    fig, axes = plt.subplots(len(antennas), 4, figsize=(16, 4.2 * len(antennas)), constrained_layout=True)
    axes = np.asarray(axes).reshape(len(antennas), 4)
    for row, antenna in enumerate(antennas):
        subset = covariances[covariances["antenna"].astype(str).eq(antenna)]
        for column, (data_kind, matrix_kind) in enumerate(
            (("raw", "time"), ("lowrank", "time"), ("raw", "frequency"), ("lowrank", "frequency"))
        ):
            current = subset[
                subset["data_kind"].eq(data_kind) & subset["matrix_kind"].eq(matrix_kind)
            ]
            pivot = current.pivot_table(
                index="row_label", columns="column_label", values="correlation", aggfunc="median"
            ).sort_index().sort_index(axis=1)
            image = axes[row, column].imshow(
                pivot.to_numpy(dtype=float), vmin=-1.0, vmax=1.0, cmap="coolwarm", origin="lower", aspect="auto"
            )
            if matrix_kind == "frequency":
                labels = [f"{value:.2f}" for value in pivot.index]
                axes[row, column].set_xticks(range(len(labels)), labels, rotation=60, fontsize=7)
                axes[row, column].set_yticks(range(len(labels)), labels, fontsize=7)
            else:
                positions = np.linspace(0, len(pivot.index) - 1, 5, dtype=int)
                labels = [f"{pivot.index[index] / 60:.0f}m" for index in positions]
                axes[row, column].set_xticks(positions, labels)
                axes[row, column].set_yticks(positions, labels)
            axes[row, column].set_title(f"{antenna}: {data_kind} {matrix_kind} correlation")
    fig.colorbar(image, ax=axes, shrink=0.72, label="cross-fitted residual correlation")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_method_comparison(summary: pd.DataFrame, output: Path) -> None:
    antennas = sorted(summary["antenna"].astype(str).unique())
    method_order = [
        "raw_fixed_zero",
        "lowrank_fixed_zero",
        "raw_covariance_student_t",
        "lowrank_covariance_student_t",
    ]
    labels = ["raw\nfixed-zero", "low-rank\nfixed-zero", "raw\ncovariance", "low-rank +\ncovariance"]
    fig, axes = plt.subplots(1, len(antennas), figsize=(12, 5), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, antenna in zip(axes, antennas):
        data = summary[summary["antenna"].astype(str).eq(antenna)].set_index("method").reindex(method_order)
        x = np.arange(len(method_order))
        ax.fill_between(x, data["null_q16"], data["null_q84"], color="0.82", label="blank-control 16–84%")
        ax.plot(x, data["null_q95"], color="#d62728", linestyle="--", linewidth=1.2, label="blank-control 95%")
        target = data["target_score_median"].to_numpy(dtype=float)
        ax.errorbar(
            x,
            target,
            yerr=np.vstack([target - data["target_block_q16"], data["target_block_q84"] - target]),
            color="#1f77b4",
            marker="o",
            capsize=3,
            linewidth=1.4,
            label="Earth at t=0",
        )
        for index, percentile in enumerate(data["target_percentile"]):
            ax.annotate(f"{percentile:.1f}%", (index, target[index]), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)
        ax.axhline(0.0, color="0.25", linewidth=0.7)
        ax.set_xticks(x, labels)
        ax.set_ylabel("Earth-template amplitude (1 = training-block Earth)")
        ax.set_title(antenna)
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Earth: four cross-fitted fixed-zero pipelines (antennas evaluated separately)")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_frequency_amplitudes(summary: pd.DataFrame, output: Path) -> None:
    antennas = sorted(summary["antenna"].astype(str).unique())
    styles = {
        ("raw", "free_huber"): ("raw fixed-zero", "#777777", "o"),
        ("lowrank", "free_huber"): ("low-rank fixed-zero", "#7b2cbf", "o"),
        ("raw", "free_student_t_gls"): ("raw covariance", "#0077b6", "^"),
        ("lowrank", "free_student_t_gls"): ("low-rank + covariance", "#d1495b", "^"),
    }
    fig, axes = plt.subplots(1, len(antennas), figsize=(13, 5.3), sharey=False, constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, antenna in zip(axes, antennas):
        antenna_data = summary[summary["antenna"].astype(str).eq(antenna)]
        frequencies = sorted(float(value) for value in antenna_data["frequency_mhz"].unique())
        x = np.arange(len(frequencies), dtype=float)
        offsets = np.linspace(-0.18, 0.18, len(styles))
        for offset, ((data_kind, fit_kind), (label, color, marker)) in zip(offsets, styles.items()):
            current = (
                antenna_data[
                    antenna_data["data_kind"].eq(data_kind)
                    & antenna_data["fit_kind"].eq(fit_kind)
                ]
                .set_index("frequency_mhz")
                .reindex(frequencies)
            )
            median = current["target_median"].to_numpy(dtype=float)
            ax.errorbar(
                x + offset,
                median,
                yerr=np.vstack(
                    [
                        median - current["target_q16"].to_numpy(dtype=float),
                        current["target_q84"].to_numpy(dtype=float) - median,
                    ]
                ),
                color=color,
                marker=marker,
                linewidth=1.1,
                capsize=2,
                markersize=4,
                label=label,
            )
        ax.axhline(0.0, color="0.25", linewidth=0.7)
        for frequency in TRACER_FREQUENCIES:
            index = next(i for i, value in enumerate(frequencies) if np.isclose(value, frequency))
            ax.axvspan(index - 0.42, index + 0.42, color="#e9c46a", alpha=0.10)
        ax.set_xticks(x, [f"{value:.2f}" for value in frequencies], rotation=45)
        ax.set_xlabel("frequency (MHz); shaded bands are foreground tracers")
        ax.set_ylabel("fixed-zero edge amplitude")
        ax.set_title(antenna)
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Earth: per-frequency amplitudes across nine held-out lunar blocks")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_injection_power(power: pd.DataFrame, output: Path) -> None:
    antennas = sorted(power["antenna"].astype(str).unique())
    colors = {
        "raw_fixed_zero": "#777777",
        "lowrank_fixed_zero": "#7b2cbf",
        "raw_covariance_student_t": "#0077b6",
        "lowrank_covariance_student_t": "#d1495b",
    }
    labels = {
        "raw_fixed_zero": "raw fixed-zero",
        "lowrank_fixed_zero": "low-rank fixed-zero",
        "raw_covariance_student_t": "raw covariance",
        "lowrank_covariance_student_t": "low-rank + covariance",
    }
    fig, axes = plt.subplots(1, len(antennas), figsize=(12, 4.8), sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, antenna in zip(axes, antennas):
        data = power[power["antenna"].astype(str).eq(antenna)]
        for method in labels:
            current = data[data["method"].eq(method)].sort_values("injection_level")
            ax.plot(
                100.0 * current["injection_level"],
                100.0 * current["power_at_95pct_null"],
                color=colors[method],
                marker="o",
                linewidth=1.7,
                label=labels[method],
            )
        ax.axhline(95.0, color="0.4", linestyle=":", linewidth=0.8)
        ax.set_xlabel("injected fraction of observed Earth template (%)")
        ax.set_title(antenna)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("recovery probability at 5% empirical false-positive rate (%)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Attenuated-Earth recovery in held-out ±20 minute control windows")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return json.loads(frame.to_json(orient="records"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--injection-trials",
        type=int,
        default=4000,
        help="maximum control combinations; all combinations are enumerated when feasible",
    )
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "source_name",
        "event_uid",
        "event_order",
        "lunar_block",
        "event_type",
        "frequency_mhz",
        "antenna",
        "control_shift_s",
        "t_bin_sec",
        "normalized_power",
    ]
    points = pd.read_parquet(
        args.profiles, columns=columns, filters=[("source_name", "==", "earth")]
    )
    points = points[np.isfinite(pd.to_numeric(points["normalized_power"], errors="coerce"))].copy()
    points = fold_occultogram_points(points)
    profiles = _block_profiles(points)
    raw_wide = _wide_profiles(profiles)
    frequencies = _available_frequencies(raw_wide)
    blocks = sorted(int(value) for value in pd.unique(raw_wide["lunar_block"]))
    antennas = sorted(str(value) for value in pd.unique(raw_wide["antenna"]))
    if len(blocks) < 5:
        raise RuntimeError("at least five lunar blocks are required for nested cross-fitting")
    for required in TRACER_FREQUENCIES + CLEAR_EARTH_FREQUENCIES:
        _frequency_column(list(frequencies), required)
    times = np.asarray(
        sorted(
            float(value)
            for value in pd.unique(raw_wide["t_bin_sec"])
            if abs(float(value)) <= FIT_HALF_WINDOW_SECONDS
        ),
        dtype=float,
    )

    lowrank, rank_selection, families = _crossfit_lowrank(raw_wide, frequencies)
    residual_wide = _residual_wide(lowrank)
    fits, injection_scores, covariances, templates, raw_templates = _run_block_fits_and_injections(
        raw_wide,
        residual_wide,
        frequencies,
        times,
        INJECTION_LEVELS,
    )
    transfer = _lowrank_transfer(
        families, blocks, antennas, frequencies, times, raw_templates
    )
    frequency_summary = _frequency_fit_summary(fits)
    method_summary, injection_power = _summarize_methods(
        fits, injection_scores, args.injection_trials, args.seed
    )

    write_table(lowrank, args.out_dir / "lowrank_crossfit_profiles.parquet", index=False)
    write_table(rank_selection, args.out_dir / "lowrank_rank_selection.parquet", index=False)
    write_table(transfer, args.out_dir / "lowrank_signal_transfer.parquet", index=False)
    write_table(fits, args.out_dir / "covariance_block_fits.parquet", index=False)
    write_table(covariances, args.out_dir / "crossfit_covariance_matrices.parquet", index=False)
    write_table(templates, args.out_dir / "crossfit_earth_spectral_templates.parquet", index=False)
    write_table(injection_scores, args.out_dir / "injection_block_scores.parquet", index=False)
    write_table(frequency_summary, args.out_dir / "frequency_fit_summary.parquet", index=False)
    write_table(method_summary, args.out_dir / "four_method_summary.parquet", index=False)
    write_table(injection_power, args.out_dir / "attenuated_earth_recovery.parquet", index=False)

    plot_paths = []
    plot_paths.extend(_plot_lowrank_occultograms(raw_wide, residual_wide, frequencies, args.out_dir))
    plot_paths.extend(_plot_lowrank_profiles(raw_wide, residual_wide, frequencies, args.out_dir))
    rank_path = args.out_dir / "earth_lowrank_rank_and_transfer.png"
    _plot_rank_and_transfer(rank_selection, transfer, rank_path)
    plot_paths.append(rank_path)
    covariance_path = args.out_dir / "earth_covariance_structure.png"
    _plot_covariance(covariances, covariance_path)
    plot_paths.append(covariance_path)
    comparison_path = args.out_dir / "earth_four_method_comparison.png"
    _plot_method_comparison(method_summary, comparison_path)
    plot_paths.append(comparison_path)
    frequency_path = args.out_dir / "earth_frequency_amplitude_comparison.png"
    _plot_frequency_amplitudes(frequency_summary, frequency_path)
    plot_paths.append(frequency_path)
    injection_path = args.out_dir / "earth_attenuated_injection_recovery.png"
    _plot_injection_power(injection_power, injection_path)
    plot_paths.append(injection_path)

    visual_data = {
        "config": {
            "source": "earth",
            "edge_time_s": 0.0,
            "antennas_fit_separately": True,
            "antenna_consistency_required": False,
            "independent_unit": "29.53059-day lunar block",
            "tracer_frequencies_mhz": TRACER_FREQUENCIES,
            "clear_earth_frequencies_mhz": CLEAR_EARTH_FREQUENCIES,
            "blank_control_shifts_s": BLANK_CONTROL_SHIFTS,
            "fit_half_window_s": FIT_HALF_WINDOW_SECONDS,
            "rank_candidates": RANK_CANDIDATES,
            "injection_levels": INJECTION_LEVELS,
            "maximum_injection_trials": args.injection_trials,
        },
        "method_summary": _records(method_summary),
        "frequency_summary": _records(frequency_summary),
        "injection_power": _records(injection_power),
        "rank_selection": _records(rank_selection),
        "signal_transfer": _records(transfer),
    }
    (args.out_dir / "earth_lowrank_covariance_visual_data.json").write_text(
        json.dumps(visual_data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "profiles": str(args.profiles),
        "output_directory": str(args.out_dir),
        "earth_profile_rows": int(len(points)),
        "lunar_blocks": len(blocks),
        "antennas": antennas,
        "frequencies_mhz": frequencies,
        "edge_time_fixed_seconds": 0.0,
        "antennas_fit_separately": True,
        "antenna_consistency_required": False,
        "crossfit_unit": "lunar_block",
        "generated_plots": [path.name for path in plot_paths],
        "claim_status": "Earth positive-control method comparison; no extension to other sources yet",
    }
    (args.out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
