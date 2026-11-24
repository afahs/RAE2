#!/usr/bin/env python
"""Run covariance-aware, fixed-zero Fornax A occultation sensitivity tests.

The analysis keeps RV1 and RV2 independent, treats lunar blocks as the
independent units, and learns foreground/covariance quantities only from
other blocks' +/-20-minute controls.  The joint spectral directions are
defined from control noise scales rather than the Fornax t=0 data.  A sharp
edge and three fixed-centered ramps are calibrated together with an exact
block-control null, so selecting a favorable width or preprocessing branch
does not create an uncorrected detection statistic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from rylevonberg.occultogram import fold_occultogram_points  # noqa: E402
from rylevonberg.table_io import write_table  # noqa: E402
from run_earth_lowrank_covariance import (  # noqa: E402
    BLANK_CONTROL_SHIFTS,
    FIT_HALF_WINDOW_SECONDS,
    TRACER_FREQUENCIES,
    _available_frequencies,
    _block_profiles,
    _crossfit_lowrank,
    _estimate_covariance,
    _frequency_column,
    _huber_fit,
    _plot_covariance,
    _profile_matrix,
    _residual_wide,
    _robust_scale,
    _student_t_linear_fit,
    _wide_profiles,
)


DEFAULT_PROFILES = Path(
    "/pscratch/sd/a/afahs/RAE2/occultograms_morphology_v1/occultogram_event_profiles.parquet"
)
DEFAULT_OUT = Path("/pscratch/sd/a/afahs/RAE2/fornax_covariance_fixedzero_v2")
SOURCE_NAME = "fornax_a"
DETECTION_FREQUENCIES = (0.45, 3.93, 4.70, 6.55, 9.18)
EDGE_WIDTHS_SECONDS = (0.0, 60.0, 120.0, 240.0)
TEMPLATE_KINDS = ("flat_control_median", "control_scaled")
INJECTION_LEVELS = (0.0, 0.10, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0)
TRACER_LEAKAGE_FRACTIONS = (0.0, 0.10, 0.25, 0.50, 1.0)
NULL_CONTROL_SHIFTS = (-1200.0, -600.0, 600.0, 1200.0)
MAX_INJECTION_CONTROL_COMBINATIONS = 8192


def _edge_vector(times: np.ndarray, width_seconds: float) -> np.ndarray:
    """Return a unit before-minus-after edge centered exactly at zero."""
    times = np.asarray(times, dtype=float)
    width = float(width_seconds)
    if width <= 0.0:
        return np.where(times < 0.0, 0.5, np.where(times > 0.0, -0.5, 0.0))
    return np.clip(-times / width, -0.5, 0.5)


def _nuisance_design(times: np.ndarray) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    return np.column_stack([np.ones_like(times), times / FIT_HALF_WINDOW_SECONDS])


def _edge_amplitude(times: np.ndarray, values: np.ndarray, width_seconds: float) -> float:
    design = np.column_stack([_nuisance_design(times), _edge_vector(times, width_seconds)])
    return float(_huber_fit(design, np.asarray(values, dtype=float))[-1])


def _control_templates(
    covariance_model: object,
    all_frequencies: tuple[float, ...],
    selected_frequencies: tuple[float, ...],
) -> dict[str, np.ndarray]:
    indices = [
        next(index for index, value in enumerate(all_frequencies) if np.isclose(value, frequency))
        for frequency in selected_frequencies
    ]
    scales = np.asarray(covariance_model.residual_scales, dtype=float)[indices]
    finite = scales[np.isfinite(scales) & (scales > 1e-10)]
    fallback = float(np.median(finite)) if finite.size else 1.0
    scales[~np.isfinite(scales) | (scales <= 1e-10)] = fallback
    return {
        "flat_control_median": np.full(len(selected_frequencies), fallback, dtype=float),
        "control_scaled": scales,
    }


def _selected_covariance(
    covariance_model: object,
    all_frequencies: tuple[float, ...],
    selected_frequencies: tuple[float, ...],
) -> np.ndarray:
    indices = [
        next(index for index, value in enumerate(all_frequencies) if np.isclose(value, frequency))
        for frequency in selected_frequencies
    ]
    frequency_covariance = covariance_model.frequency_covariance[np.ix_(indices, indices)]
    return np.kron(frequency_covariance, covariance_model.time_covariance)


def _joint_fit(
    matrix: np.ndarray,
    times: np.ndarray,
    covariance: np.ndarray,
    spectral_template: np.ndarray,
    width_seconds: float,
) -> tuple[float, float, float]:
    frequency_count = matrix.shape[1]
    nuisance = _nuisance_design(times)
    nuisance_design = np.kron(np.eye(frequency_count), nuisance)
    edge = _edge_vector(times, width_seconds)
    template_column = np.concatenate(
        [float(amplitude) * edge for amplitude in spectral_template]
    )
    design = np.column_stack([nuisance_design, template_column])
    beta, parameter_covariance, _ = _student_t_linear_fit(
        matrix.reshape(-1, order="F"), design, covariance
    )
    estimate = float(beta[-1])
    standard_error = float(np.sqrt(max(parameter_covariance[-1, -1], 0.0)))
    score = estimate / standard_error if standard_error > 1e-12 else np.nan
    return estimate, standard_error, float(score)


def _free_fit(
    matrix: np.ndarray,
    times: np.ndarray,
    covariance: np.ndarray,
    width_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    per_frequency = np.column_stack(
        [_nuisance_design(times), _edge_vector(times, width_seconds)]
    )
    design = np.kron(np.eye(matrix.shape[1]), per_frequency)
    beta, parameter_covariance, _ = _student_t_linear_fit(
        matrix.reshape(-1, order="F"), design, covariance
    )
    indices = np.arange(matrix.shape[1]) * 3 + 2
    errors = np.sqrt(np.maximum(np.diag(parameter_covariance)[indices], 0.0))
    return beta[indices], errors


def _covariance_records(
    block: int,
    antenna: str,
    data_kind: str,
    model: object,
    times: np.ndarray,
    frequencies: tuple[float, ...],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    time_corr = model.time_covariance / np.sqrt(
        np.outer(np.diag(model.time_covariance), np.diag(model.time_covariance))
    )
    frequency_corr = model.frequency_covariance / np.sqrt(
        np.outer(np.diag(model.frequency_covariance), np.diag(model.frequency_covariance))
    )
    for row, row_label in enumerate(times):
        for column, column_label in enumerate(times):
            records.append(
                {
                    "held_out_lunar_block": block,
                    "antenna": antenna,
                    "data_kind": data_kind,
                    "matrix_kind": "time",
                    "row_label": float(row_label),
                    "column_label": float(column_label),
                    "covariance": float(model.time_covariance[row, column]),
                    "correlation": float(time_corr[row, column]),
                    "shrinkage": float(model.time_shrinkage),
                    "n_control_profiles": int(model.n_control_profiles),
                }
            )
    for row, row_label in enumerate(frequencies):
        for column, column_label in enumerate(frequencies):
            records.append(
                {
                    "held_out_lunar_block": block,
                    "antenna": antenna,
                    "data_kind": data_kind,
                    "matrix_kind": "frequency",
                    "row_label": float(row_label),
                    "column_label": float(column_label),
                    "covariance": float(model.frequency_covariance[row, column]),
                    "correlation": float(frequency_corr[row, column]),
                    "shrinkage": float(model.frequency_shrinkage),
                    "n_control_profiles": int(model.n_control_profiles),
                }
            )
    return records


def _run_fits(
    raw_wide: pd.DataFrame,
    residual_wide: pd.DataFrame,
    frequencies: tuple[float, ...],
    selected_frequencies: tuple[float, ...],
    times: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[tuple[int, str, str], object]]:
    blocks = sorted(int(value) for value in pd.unique(raw_wide["lunar_block"]))
    antennas = sorted(str(value) for value in pd.unique(raw_wide["antenna"]))
    fit_records: list[dict[str, object]] = []
    free_records: list[dict[str, object]] = []
    covariance_records: list[dict[str, object]] = []
    template_records: list[dict[str, object]] = []
    models: dict[tuple[int, str, str], object] = {}

    for block in blocks:
        training_blocks = [candidate for candidate in blocks if candidate != block]
        for antenna in antennas:
            print(f"fitting block={block} antenna={antenna}", flush=True)
            for data_kind, data_wide in (("raw", raw_wide), ("lowrank", residual_wide)):
                model = _estimate_covariance(
                    data_wide, training_blocks, antenna, frequencies, times
                )
                models[(block, antenna, data_kind)] = model
                covariance_records.extend(
                    _covariance_records(block, antenna, data_kind, model, times, frequencies)
                )
                templates = _control_templates(model, frequencies, selected_frequencies)
                for template_kind, spectral_template in templates.items():
                    for frequency, amplitude in zip(selected_frequencies, spectral_template):
                        template_records.append(
                            {
                                "held_out_lunar_block": block,
                                "antenna": antenna,
                                "data_kind": data_kind,
                                "template_kind": template_kind,
                                "frequency_mhz": float(frequency),
                                "template_edge_scale": float(amplitude),
                                "template_source": "other-block +/-20-minute control residual scale",
                            }
                        )
                covariance = _selected_covariance(model, frequencies, selected_frequencies)
                for shift in (0.0,) + NULL_CONTROL_SHIFTS:
                    full_matrix = _profile_matrix(
                        data_wide, block, antenna, shift, frequencies, times
                    )
                    selected_indices = [
                        next(
                            index
                            for index, value in enumerate(frequencies)
                            if np.isclose(value, frequency)
                        )
                        for frequency in selected_frequencies
                    ]
                    matrix = full_matrix[:, selected_indices]
                    for width_seconds in EDGE_WIDTHS_SECONDS:
                        amplitudes, errors = _free_fit(
                            matrix, times, covariance, width_seconds
                        )
                        for frequency, estimate, error in zip(
                            selected_frequencies, amplitudes, errors
                        ):
                            free_records.append(
                                {
                                    "held_out_lunar_block": block,
                                    "antenna": antenna,
                                    "data_kind": data_kind,
                                    "control_shift_s": float(shift),
                                    "edge_width_s": float(width_seconds),
                                    "frequency_mhz": float(frequency),
                                    "estimate": float(estimate),
                                    "standard_error": float(error),
                                }
                            )
                        for template_kind, spectral_template in templates.items():
                            estimate, standard_error, score = _joint_fit(
                                matrix,
                                times,
                                covariance,
                                spectral_template,
                                width_seconds,
                            )
                            fit_records.append(
                                {
                                    "held_out_lunar_block": block,
                                    "antenna": antenna,
                                    "data_kind": data_kind,
                                    "control_shift_s": float(shift),
                                    "template_kind": template_kind,
                                    "edge_width_s": float(width_seconds),
                                    "estimate_control_scale_units": estimate,
                                    "standard_error": standard_error,
                                    "standardized_matched_score": score,
                                }
                            )
    return (
        pd.DataFrame.from_records(fit_records),
        pd.DataFrame.from_records(free_records),
        pd.DataFrame.from_records(covariance_records),
        pd.DataFrame.from_records(template_records),
        models,
    )


def _run_injections(fits: pd.DataFrame) -> pd.DataFrame:
    """Inject the fitted template analytically into matched control fits.

    Adding ``level * template`` to a linear model increases its fitted template
    coefficient by exactly ``level`` while leaving every residual, robust
    weight, and standard error unchanged.  Using that identity avoids
    thousands of numerically redundant refits.  This is a post-subtraction,
    zero-tracer-leakage sensitivity curve; low-rank transfer is audited
    separately.
    """
    records: list[dict[str, object]] = []
    shifts = pd.to_numeric(fits["control_shift_s"], errors="coerce").to_numpy(float)
    control_mask = np.zeros(len(fits), dtype=bool)
    for shift in NULL_CONTROL_SHIFTS:
        control_mask |= np.isclose(shifts, shift)
    controls = fits[control_mask]
    for row in controls.itertuples(index=False):
        for level in INJECTION_LEVELS:
            estimate = float(row.estimate_control_scale_units) + float(level)
            standard_error = float(row.standard_error)
            score = estimate / standard_error if standard_error > 1e-12 else np.nan
            records.append(
                {
                    "held_out_lunar_block": int(row.held_out_lunar_block),
                    "antenna": str(row.antenna),
                    "data_kind": str(row.data_kind),
                    "control_shift_s": float(row.control_shift_s),
                    "template_kind": str(row.template_kind),
                    "edge_width_s": float(row.edge_width_s),
                    "injection_level_control_scales": float(level),
                    "estimate_control_scale_units": estimate,
                    "standard_error": standard_error,
                    "standardized_matched_score": float(score),
                    "injection_stage": "post-foreground-subtraction; zero tracer leakage",
                }
            )
    return pd.DataFrame.from_records(records)


def _choice_indices(block_count: int) -> np.ndarray:
    combinations = len(NULL_CONTROL_SHIFTS) ** int(block_count)
    identifiers = np.arange(combinations, dtype=np.int64)
    return np.column_stack(
        [
            (identifiers // (len(NULL_CONTROL_SHIFTS) ** block_index))
            % len(NULL_CONTROL_SHIFTS)
            for block_index in range(block_count)
        ]
    )


def _candidate_key(row: pd.Series | dict[str, object]) -> tuple[str, str, str, float]:
    return (
        str(row["antenna"]),
        str(row["data_kind"]),
        str(row["template_kind"]),
        float(row["edge_width_s"]),
    )


def _candidate_summaries(
    fits: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, str, str, float], np.ndarray]]:
    blocks = sorted(int(value) for value in pd.unique(fits["held_out_lunar_block"]))
    choices = _choice_indices(len(blocks))
    keys = ["antenna", "data_kind", "template_kind", "edge_width_s"]
    records: list[dict[str, object]] = []
    nulls: dict[tuple[str, str, str, float], np.ndarray] = {}
    for group_key, group in fits.groupby(keys, observed=True, sort=True):
        antenna, data_kind, template_kind, width_seconds = group_key
        target_rows = group[np.isclose(group["control_shift_s"], 0.0)].set_index(
            "held_out_lunar_block"
        )
        target_scores = target_rows["standardized_matched_score"].reindex(blocks).to_numpy(float)
        target_estimates = target_rows["estimate_control_scale_units"].reindex(blocks).to_numpy(float)
        cube = np.empty((len(blocks), len(NULL_CONTROL_SHIFTS)), dtype=float)
        for block_index, block in enumerate(blocks):
            for shift_index, shift in enumerate(NULL_CONTROL_SHIFTS):
                selected = group[
                    group["held_out_lunar_block"].eq(block)
                    & np.isclose(group["control_shift_s"], shift)
                ]["standardized_matched_score"]
                if len(selected) != 1:
                    raise RuntimeError("candidate control grid is incomplete")
                cube[block_index, shift_index] = float(selected.iloc[0])
        chosen = cube[np.arange(len(blocks))[None, :], choices]
        null = np.median(chosen, axis=1)
        key = (str(antenna), str(data_kind), str(template_kind), float(width_seconds))
        nulls[key] = null
        observed = float(np.median(target_scores))
        records.append(
            {
                "antenna": str(antenna),
                "data_kind": str(data_kind),
                "template_kind": str(template_kind),
                "edge_width_s": float(width_seconds),
                "target_median_standardized_score": observed,
                "target_score_q16": float(np.quantile(target_scores, 0.16)),
                "target_score_q84": float(np.quantile(target_scores, 0.84)),
                "target_median_estimate_control_scales": float(np.median(target_estimates)),
                "correct_sign_block_fraction": float(np.mean(target_estimates > 0.0)),
                "null_median": float(np.median(null)),
                "null_q95": float(np.quantile(null, 0.95)),
                "null_q99": float(np.quantile(null, 0.99)),
                "one_sided_empirical_p": float((1 + np.sum(null >= observed)) / (len(null) + 1)),
                "n_lunar_blocks": len(blocks),
                "control_combinations": len(null),
            }
        )
    candidates = pd.DataFrame.from_records(records)

    family_records: list[dict[str, object]] = []
    scopes: list[tuple[str, pd.DataFrame]] = []
    for (antenna, data_kind), group in candidates.groupby(
        ["antenna", "data_kind"], observed=True, sort=True
    ):
        scopes.append((f"pipeline:{antenna}:{data_kind}", group))
    for antenna, group in candidates.groupby("antenna", observed=True, sort=True):
        scopes.append((f"antenna:{antenna}:raw_and_lowrank", group))
    scopes.append(("all_antennas_and_methods", candidates))
    for scope, group in scopes:
        observed_values = group["target_median_standardized_score"].to_numpy(float)
        best_index = int(np.nanargmax(observed_values))
        best = group.iloc[best_index]
        arrays = np.stack([nulls[_candidate_key(row)] for _, row in group.iterrows()])
        family_null = np.max(arrays, axis=0)
        observed = float(observed_values[best_index])
        family_records.append(
            {
                "scope": scope,
                "best_antenna": str(best["antenna"]),
                "best_data_kind": str(best["data_kind"]),
                "best_template_kind": str(best["template_kind"]),
                "best_edge_width_s": float(best["edge_width_s"]),
                "observed_max_median_standardized_score": observed,
                "family_null_median": float(np.median(family_null)),
                "family_null_q95": float(np.quantile(family_null, 0.95)),
                "family_null_q99": float(np.quantile(family_null, 0.99)),
                "familywise_one_sided_empirical_p": float(
                    (1 + np.sum(family_null >= observed)) / (len(family_null) + 1)
                ),
                "candidates_in_family": int(len(group)),
                "control_combinations": int(len(family_null)),
            }
        )
    return candidates, pd.DataFrame.from_records(family_records), nulls


def _run_leave_one_frequency_out(
    raw_wide: pd.DataFrame,
    residual_wide: pd.DataFrame,
    frequencies: tuple[float, ...],
    selected_frequencies: tuple[float, ...],
    times: np.ndarray,
    models: dict[tuple[int, str, str], object],
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """Refit each winning pipeline candidate after dropping one band.

    Candidate template and width are chosen by the full-band analysis, so this
    is explicitly a stability diagnostic rather than an additional
    confirmatory p-value.
    """
    blocks = sorted(int(value) for value in pd.unique(raw_wide["lunar_block"]))
    records: list[dict[str, object]] = []
    for (antenna, data_kind), group in candidates.groupby(
        ["antenna", "data_kind"], observed=True, sort=True
    ):
        best = group.loc[group["target_median_standardized_score"].idxmax()]
        data_wide = raw_wide if str(data_kind) == "raw" else residual_wide
        for dropped_frequency in selected_frequencies:
            subset = tuple(
                frequency
                for frequency in selected_frequencies
                if not np.isclose(frequency, dropped_frequency)
            )
            subset_indices = [
                next(index for index, value in enumerate(frequencies) if np.isclose(value, frequency))
                for frequency in subset
            ]
            for block in blocks:
                model = models[(block, str(antenna), str(data_kind))]
                covariance = _selected_covariance(model, frequencies, subset)
                spectral_template = _control_templates(model, frequencies, subset)[
                    str(best["template_kind"])
                ]
                for shift in (0.0,) + NULL_CONTROL_SHIFTS:
                    matrix = _profile_matrix(
                        data_wide, block, str(antenna), shift, frequencies, times
                    )[:, subset_indices]
                    estimate, standard_error, score = _joint_fit(
                        matrix,
                        times,
                        covariance,
                        spectral_template,
                        float(best["edge_width_s"]),
                    )
                    records.append(
                        {
                            "held_out_lunar_block": block,
                            "antenna": str(antenna),
                            "data_kind": str(data_kind),
                            "selected_template_kind": str(best["template_kind"]),
                            "selected_edge_width_s": float(best["edge_width_s"]),
                            "dropped_frequency_mhz": float(dropped_frequency),
                            "control_shift_s": float(shift),
                            "estimate_control_scale_units": estimate,
                            "standard_error": standard_error,
                            "standardized_matched_score": score,
                            "interpretation": "outcome-informed stability diagnostic; not confirmatory",
                        }
                    )
    return pd.DataFrame.from_records(records)


def _leave_one_frequency_out_summary(fits: pd.DataFrame) -> pd.DataFrame:
    blocks = sorted(int(value) for value in pd.unique(fits["held_out_lunar_block"]))
    choices = _choice_indices(len(blocks))
    keys = [
        "antenna",
        "data_kind",
        "selected_template_kind",
        "selected_edge_width_s",
        "dropped_frequency_mhz",
    ]
    records: list[dict[str, object]] = []
    for group_key, group in fits.groupby(keys, observed=True, sort=True):
        target = group[np.isclose(group["control_shift_s"], 0.0)].set_index(
            "held_out_lunar_block"
        )["standardized_matched_score"].reindex(blocks).to_numpy(float)
        cube = np.empty((len(blocks), len(NULL_CONTROL_SHIFTS)), dtype=float)
        for block_index, block in enumerate(blocks):
            for shift_index, shift in enumerate(NULL_CONTROL_SHIFTS):
                value = group[
                    group["held_out_lunar_block"].eq(block)
                    & np.isclose(group["control_shift_s"], shift)
                ]["standardized_matched_score"]
                if len(value) != 1:
                    raise RuntimeError("leave-one-frequency-out control grid is incomplete")
                cube[block_index, shift_index] = float(value.iloc[0])
        chosen = cube[np.arange(len(blocks))[None, :], choices]
        null = np.median(chosen, axis=1)
        observed = float(np.median(target))
        records.append(
            {
                "antenna": str(group_key[0]),
                "data_kind": str(group_key[1]),
                "selected_template_kind": str(group_key[2]),
                "selected_edge_width_s": float(group_key[3]),
                "dropped_frequency_mhz": float(group_key[4]),
                "target_median_standardized_score": observed,
                "target_score_q16": float(np.quantile(target, 0.16)),
                "target_score_q84": float(np.quantile(target, 0.84)),
                "null_q95": float(np.quantile(null, 0.95)),
                "diagnostic_one_sided_empirical_p": float(
                    (1 + np.sum(null >= observed)) / (len(null) + 1)
                ),
                "control_combinations": int(len(null)),
                "interpretation": "outcome-informed stability diagnostic; not confirmatory",
            }
        )
    return pd.DataFrame.from_records(records)


def _injection_power(
    injections: pd.DataFrame,
    nulls: dict[tuple[str, str, str, float], np.ndarray],
) -> pd.DataFrame:
    blocks = sorted(int(value) for value in pd.unique(injections["held_out_lunar_block"]))
    choices = _choice_indices(len(blocks))
    possible_combinations = len(choices)
    if possible_combinations > MAX_INJECTION_CONTROL_COMBINATIONS:
        rng = np.random.default_rng(20260826)
        choices = choices[
            np.sort(
                rng.choice(
                    possible_combinations,
                    size=MAX_INJECTION_CONTROL_COMBINATIONS,
                    replace=False,
                )
            )
        ]
    keys = ["antenna", "data_kind", "template_kind", "edge_width_s"]
    records: list[dict[str, object]] = []
    for group_key, group in injections.groupby(keys, observed=True, sort=True):
        key = (str(group_key[0]), str(group_key[1]), str(group_key[2]), float(group_key[3]))
        null = nulls[key]
        threshold_95 = float(np.quantile(null, 0.95))
        threshold_99 = float(np.quantile(null, 0.99))
        for level, selected_level in group.groupby(
            "injection_level_control_scales", observed=True, sort=True
        ):
            cube = np.empty((len(blocks), len(NULL_CONTROL_SHIFTS)), dtype=float)
            for block_index, block in enumerate(blocks):
                for shift_index, shift in enumerate(NULL_CONTROL_SHIFTS):
                    value = selected_level[
                        selected_level["held_out_lunar_block"].eq(block)
                        & np.isclose(selected_level["control_shift_s"], shift)
                    ]["standardized_matched_score"]
                    if len(value) != 1:
                        raise RuntimeError("injection control grid is incomplete")
                    cube[block_index, shift_index] = float(value.iloc[0])
            chosen = cube[np.arange(len(blocks))[None, :], choices]
            statistics = np.median(chosen, axis=1)
            records.append(
                {
                    "antenna": key[0],
                    "data_kind": key[1],
                    "template_kind": key[2],
                    "edge_width_s": key[3],
                    "injection_level_control_scales": float(level),
                    "power_at_candidate_95pct_null": float(np.mean(statistics > threshold_95)),
                    "power_at_candidate_99pct_null": float(np.mean(statistics > threshold_99)),
                    "score_q16": float(np.quantile(statistics, 0.16)),
                    "score_median": float(np.median(statistics)),
                    "score_q84": float(np.quantile(statistics, 0.84)),
                    "control_combinations": int(len(statistics)),
                    "control_combinations_possible": int(possible_combinations),
                    "combination_mode": (
                        "exact"
                        if len(statistics) == possible_combinations
                        else "fixed random sample"
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def _frequency_summary(free_fits: pd.DataFrame) -> pd.DataFrame:
    selected = free_fits[
        np.isclose(free_fits["control_shift_s"], 0.0)
        & np.isclose(free_fits["edge_width_s"], 0.0)
    ]
    records: list[dict[str, object]] = []
    keys = ["antenna", "data_kind", "frequency_mhz"]
    for group_key, group in selected.groupby(keys, observed=True, sort=True):
        values = pd.to_numeric(group["estimate"], errors="coerce").dropna()
        errors = pd.to_numeric(group["standard_error"], errors="coerce").dropna()
        records.append(
            {
                "antenna": str(group_key[0]),
                "data_kind": str(group_key[1]),
                "frequency_mhz": float(group_key[2]),
                "target_block_median": float(values.median()),
                "target_block_q16": float(values.quantile(0.16)),
                "target_block_q84": float(values.quantile(0.84)),
                "correct_sign_block_fraction": float(np.mean(values > 0.0)),
                "median_internal_standard_error": float(errors.median()),
                "n_lunar_blocks": int(values.size),
            }
        )
    return pd.DataFrame.from_records(records)


def _lowrank_signal_transfer(
    families: dict[tuple[int, str], dict[float, object]],
    templates: pd.DataFrame,
    selected_frequencies: tuple[float, ...],
    times: np.ndarray,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    edge = _edge_vector(times, 0.0)
    lowrank_templates = templates[templates["data_kind"].eq("lowrank")]
    for (block, antenna), family in families.items():
        for template_kind in TEMPLATE_KINDS:
            selected = lowrank_templates[
                lowrank_templates["held_out_lunar_block"].eq(block)
                & lowrank_templates["antenna"].astype(str).eq(antenna)
                & lowrank_templates["template_kind"].eq(template_kind)
            ].set_index("frequency_mhz")["template_edge_scale"]
            spectrum = np.asarray([float(selected.loc[frequency]) for frequency in selected_frequencies])
            tracer_reference = float(np.median(spectrum))
            for leakage in TRACER_LEAKAGE_FRACTIONS:
                for frequency, input_amplitude in zip(selected_frequencies, spectrum):
                    model = family[frequency]
                    baseline = pd.DataFrame(index=np.arange(len(times)))
                    injected = pd.DataFrame(index=np.arange(len(times)))
                    for tracer_index, tracer_frequency in enumerate(model.tracer_frequencies):
                        center = float(model.center[tracer_index])
                        baseline[tracer_frequency] = center
                        injected[tracer_frequency] = (
                            center + float(leakage) * tracer_reference * edge
                        )
                    prediction_change = model.predict(injected) - model.predict(baseline)
                    residual_signal = input_amplitude * edge - prediction_change
                    recovered = _edge_amplitude(times, residual_signal, 0.0)
                    records.append(
                        {
                            "held_out_lunar_block": int(block),
                            "antenna": str(antenna),
                            "template_kind": template_kind,
                            "frequency_mhz": float(frequency),
                            "tracer_edge_fraction_of_median_detection_edge": float(leakage),
                            "input_edge_amplitude": float(input_amplitude),
                            "recovered_edge_amplitude": float(recovered),
                            "retained_fraction": float(recovered / input_amplitude),
                        }
                    )
    return pd.DataFrame.from_records(records)


def _plot_candidate_grid(summary: pd.DataFrame, familywise: pd.DataFrame, output: Path) -> None:
    antennas = sorted(summary["antenna"].unique())
    data_kinds = ["raw", "lowrank"]
    fig, axes = plt.subplots(len(antennas), len(data_kinds), figsize=(12, 7.5), constrained_layout=True)
    axes = np.asarray(axes).reshape(len(antennas), len(data_kinds))
    image = None
    common_limit = max(
        2.0,
        float(np.nanmax(np.abs(summary["target_median_standardized_score"].to_numpy(float)))),
    )
    for row, antenna in enumerate(antennas):
        for column, data_kind in enumerate(data_kinds):
            current = summary[
                summary["antenna"].eq(antenna) & summary["data_kind"].eq(data_kind)
            ]
            pivot = current.pivot_table(
                index="template_kind",
                columns="edge_width_s",
                values="target_median_standardized_score",
            ).reindex(index=TEMPLATE_KINDS, columns=EDGE_WIDTHS_SECONDS)
            image = axes[row, column].imshow(
                pivot.to_numpy(dtype=float),
                cmap="RdBu_r",
                vmin=-common_limit,
                vmax=common_limit,
                aspect="auto",
            )
            for y, template_kind in enumerate(TEMPLATE_KINDS):
                for x, width in enumerate(EDGE_WIDTHS_SECONDS):
                    match = current[
                        current["template_kind"].eq(template_kind)
                        & np.isclose(current["edge_width_s"], width)
                    ].iloc[0]
                    axes[row, column].text(
                        x,
                        y,
                        f"{match['target_median_standardized_score']:.2f}\np={match['one_sided_empirical_p']:.3g}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="black",
                    )
            axes[row, column].set_xticks(
                range(len(EDGE_WIDTHS_SECONDS)),
                ["step" if width == 0 else f"{width:.0f}s" for width in EDGE_WIDTHS_SECONDS],
            )
            axes[row, column].set_yticks(
                range(len(TEMPLATE_KINDS)), ["flat-control", "control-scaled"]
            )
            pipeline = familywise[
                familywise["scope"].eq(f"pipeline:{antenna}:{data_kind}")
            ].iloc[0]
            axes[row, column].set_title(
                f"{antenna} {data_kind}: familywise p={pipeline['familywise_one_sided_empirical_p']:.3g}"
            )
    if image is not None:
        fig.colorbar(image, ax=axes, shrink=0.75, label="median block standardized matched score")
    overall = familywise[familywise["scope"].eq("all_antennas_and_methods")].iloc[0]
    fig.suptitle(
        "Fornax A fixed-zero covariance candidates\n"
        f"all-antennas/methods familywise p={overall['familywise_one_sided_empirical_p']:.3g}"
    )
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_block_scores(
    fits: pd.DataFrame, candidates: pd.DataFrame, output: Path
) -> None:
    antennas = sorted(fits["antenna"].unique())
    data_kinds = ["raw", "lowrank"]
    fig, axes = plt.subplots(len(antennas), len(data_kinds), figsize=(13, 7.5), constrained_layout=True)
    axes = np.asarray(axes).reshape(len(antennas), len(data_kinds))
    for row, antenna in enumerate(antennas):
        for column, data_kind in enumerate(data_kinds):
            candidate_group = candidates[
                candidates["antenna"].eq(antenna) & candidates["data_kind"].eq(data_kind)
            ]
            best = candidate_group.loc[
                candidate_group["target_median_standardized_score"].idxmax()
            ]
            current = fits[
                fits["antenna"].eq(antenna)
                & fits["data_kind"].eq(data_kind)
                & fits["template_kind"].eq(best["template_kind"])
                & np.isclose(fits["edge_width_s"], best["edge_width_s"])
            ]
            blocks = sorted(int(value) for value in current["held_out_lunar_block"].unique())
            for shift, label, color, marker in (
                (-1200.0, "-20m control", "0.75", "<"),
                (-600.0, "-10m control", "0.58", "s"),
                (600.0, "+10m control", "0.42", "d"),
                (1200.0, "+20m control", "0.25", ">"),
                (0.0, "t=0", "#d62728", "o"),
            ):
                values = current[np.isclose(current["control_shift_s"], shift)].set_index(
                    "held_out_lunar_block"
                )["standardized_matched_score"].reindex(blocks)
                axes[row, column].plot(
                    blocks, values, marker=marker, color=color, linewidth=1.0, label=label
                )
            axes[row, column].axhline(0.0, color="black", linewidth=0.7)
            axes[row, column].set_title(
                f"{antenna} {data_kind}: {best['template_kind']}, "
                f"{'step' if best['edge_width_s'] == 0 else f'{best['edge_width_s']:.0f}s ramp'}"
            )
            axes[row, column].set_xlabel("lunar block")
            axes[row, column].set_ylabel("standardized matched score")
            axes[row, column].grid(alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Fornax A: best candidate in each preprocessing branch")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_frequency_summary(summary: pd.DataFrame, output: Path) -> None:
    antennas = sorted(summary["antenna"].unique())
    fig, axes = plt.subplots(1, len(antennas), figsize=(12, 4.5), sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    colors = {"raw": "#777777", "lowrank": "#7b2cbf"}
    for ax, antenna in zip(axes, antennas):
        for data_kind in ("raw", "lowrank"):
            current = summary[
                summary["antenna"].eq(antenna) & summary["data_kind"].eq(data_kind)
            ].sort_values("frequency_mhz")
            y = current["target_block_median"].to_numpy(float)
            lower = y - current["target_block_q16"].to_numpy(float)
            upper = current["target_block_q84"].to_numpy(float) - y
            ax.errorbar(
                current["frequency_mhz"],
                y,
                yerr=np.vstack([lower, upper]),
                marker="o",
                capsize=3,
                color=colors[data_kind],
                label=data_kind,
            )
        ax.axhline(0.0, color="black", linewidth=0.7)
        ax.set_xscale("log")
        ax.set_xticks(DETECTION_FREQUENCIES, [f"{value:.2f}" for value in DETECTION_FREQUENCIES])
        ax.set_xlabel("frequency (MHz)")
        ax.set_title(antenna)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("sharp-edge amplitude; median and block 16-84%")
    axes[0].legend(frameon=False)
    fig.suptitle("Fornax A covariance-aware free-frequency diagnostics")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_leave_one_frequency_out(summary: pd.DataFrame, output: Path) -> None:
    antennas = sorted(summary["antenna"].unique())
    data_kinds = ["raw", "lowrank"]
    fig, axes = plt.subplots(
        len(antennas), len(data_kinds), figsize=(12, 7.5), constrained_layout=True
    )
    axes = np.asarray(axes).reshape(len(antennas), len(data_kinds))
    for row, antenna in enumerate(antennas):
        for column, data_kind in enumerate(data_kinds):
            current = summary[
                summary["antenna"].eq(antenna) & summary["data_kind"].eq(data_kind)
            ].sort_values("dropped_frequency_mhz")
            x = np.arange(len(current))
            axes[row, column].plot(
                x,
                current["target_median_standardized_score"],
                marker="o",
                color="#d62728",
                label="t=0",
            )
            axes[row, column].plot(
                x,
                current["null_q95"],
                marker="s",
                color="0.35",
                label="control 95%",
            )
            for position, (_, record) in enumerate(current.iterrows()):
                axes[row, column].text(
                    position,
                    float(record["target_median_standardized_score"]),
                    f"p={record['diagnostic_one_sided_empirical_p']:.3g}",
                    fontsize=7,
                    ha="center",
                    va="bottom",
                )
            axes[row, column].axhline(0.0, color="black", linewidth=0.7)
            axes[row, column].set_xticks(
                x,
                [f"drop\n{value:.2f}" for value in current["dropped_frequency_mhz"]],
            )
            axes[row, column].set_title(f"{antenna} {data_kind}")
            axes[row, column].set_ylabel("median block standardized score")
            axes[row, column].grid(alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Fornax A leave-one-frequency-out stability\n"
        "winning full-band candidate held fixed; p-values are diagnostic"
    )
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_injection_power(
    power: pd.DataFrame, candidates: pd.DataFrame, output: Path
) -> None:
    antennas = sorted(power["antenna"].unique())
    data_kinds = ["raw", "lowrank"]
    fig, axes = plt.subplots(len(antennas), len(data_kinds), figsize=(12, 7.5), constrained_layout=True)
    axes = np.asarray(axes).reshape(len(antennas), len(data_kinds))
    for row, antenna in enumerate(antennas):
        for column, data_kind in enumerate(data_kinds):
            group = candidates[
                candidates["antenna"].eq(antenna) & candidates["data_kind"].eq(data_kind)
            ]
            best = group.loc[group["target_median_standardized_score"].idxmax()]
            current = power[
                power["antenna"].eq(antenna)
                & power["data_kind"].eq(data_kind)
                & power["template_kind"].eq(best["template_kind"])
                & np.isclose(power["edge_width_s"], best["edge_width_s"])
            ].sort_values("injection_level_control_scales")
            axes[row, column].plot(
                current["injection_level_control_scales"],
                current["power_at_candidate_95pct_null"],
                marker="o",
                label="95% candidate null",
            )
            axes[row, column].plot(
                current["injection_level_control_scales"],
                current["power_at_candidate_99pct_null"],
                marker="s",
                label="99% candidate null",
            )
            axes[row, column].set_ylim(-0.03, 1.03)
            axes[row, column].set_title(f"{antenna} {data_kind}")
            axes[row, column].set_xlabel("injected edge (control-scale units)")
            axes[row, column].set_ylabel("recovery fraction")
            axes[row, column].grid(alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Fornax A post-subtraction fixed-zero injection recovery")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_signal_transfer(transfer: pd.DataFrame, output: Path) -> None:
    antennas = sorted(transfer["antenna"].unique())
    fig, axes = plt.subplots(1, len(antennas), figsize=(12, 4.5), sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, antenna in zip(axes, antennas):
        current = transfer[
            transfer["antenna"].eq(antenna)
            & transfer["template_kind"].eq("control_scaled")
        ]
        summary = current.groupby(
            "tracer_edge_fraction_of_median_detection_edge", observed=True
        )["retained_fraction"].agg(
            median="median",
            q16=lambda values: values.quantile(0.16),
            q84=lambda values: values.quantile(0.84),
        )
        x = summary.index.to_numpy(float)
        ax.plot(x, summary["median"], marker="o", color="#7b2cbf")
        ax.fill_between(x, summary["q16"], summary["q84"], color="#7b2cbf", alpha=0.18)
        ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
        ax.axhline(0.0, color="black", linewidth=0.6)
        ax.set_title(antenna)
        ax.set_xlabel("tracer-band edge / median detection-band edge")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("fraction of injected detection-band edge retained")
    fig.suptitle("Fornax A low-rank signal-transfer sensitivity")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return json.loads(frame.to_json(orient="records"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
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
        args.profiles, columns=columns, filters=[("source_name", "==", SOURCE_NAME)]
    )
    if points.empty:
        raise RuntimeError("no Fornax A occultogram profiles were found")
    points = points[np.isfinite(pd.to_numeric(points["normalized_power"], errors="coerce"))].copy()
    points = fold_occultogram_points(points)
    raw_wide = _wide_profiles(_block_profiles(points))
    frequencies = _available_frequencies(raw_wide)
    selected_frequencies = tuple(
        _frequency_column(list(frequencies), requested) for requested in DETECTION_FREQUENCIES
    )
    for tracer in TRACER_FREQUENCIES:
        _frequency_column(list(frequencies), tracer)
    blocks = sorted(int(value) for value in pd.unique(raw_wide["lunar_block"]))
    antennas = sorted(str(value) for value in pd.unique(raw_wide["antenna"]))
    if len(blocks) < 5:
        raise RuntimeError("at least five lunar blocks are required")
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
    fits, free_fits, covariances, templates, models = _run_fits(
        raw_wide, residual_wide, frequencies, selected_frequencies, times
    )
    candidates, familywise, nulls = _candidate_summaries(fits)
    leave_one_out_fits = _run_leave_one_frequency_out(
        raw_wide,
        residual_wide,
        frequencies,
        selected_frequencies,
        times,
        models,
        candidates,
    )
    leave_one_out_summary = _leave_one_frequency_out_summary(leave_one_out_fits)
    injections = _run_injections(fits)
    injection_power = _injection_power(injections, nulls)
    frequency_summary = _frequency_summary(free_fits)
    signal_transfer = _lowrank_signal_transfer(
        families, templates, selected_frequencies, times
    )

    write_table(lowrank, args.out_dir / "lowrank_crossfit_block_profiles.parquet", index=False)
    write_table(rank_selection, args.out_dir / "lowrank_rank_selection.parquet", index=False)
    write_table(fits, args.out_dir / "covariance_joint_block_fits.parquet", index=False)
    write_table(free_fits, args.out_dir / "covariance_free_frequency_fits.parquet", index=False)
    write_table(covariances, args.out_dir / "crossfit_covariance_matrices.parquet", index=False)
    write_table(templates, args.out_dir / "control_only_spectral_templates.parquet", index=False)
    write_table(candidates, args.out_dir / "candidate_empirical_null_summary.parquet", index=False)
    write_table(familywise, args.out_dir / "familywise_empirical_null_summary.parquet", index=False)
    write_table(
        leave_one_out_fits,
        args.out_dir / "leave_one_frequency_out_block_fits.parquet",
        index=False,
    )
    write_table(
        leave_one_out_summary,
        args.out_dir / "leave_one_frequency_out_summary.parquet",
        index=False,
    )
    write_table(injections, args.out_dir / "postsubtraction_injection_scores.parquet", index=False)
    write_table(injection_power, args.out_dir / "postsubtraction_injection_power.parquet", index=False)
    write_table(frequency_summary, args.out_dir / "free_frequency_summary.parquet", index=False)
    write_table(signal_transfer, args.out_dir / "lowrank_signal_transfer.parquet", index=False)

    plot_paths: list[Path] = []
    candidate_path = args.out_dir / "fornax_covariance_candidate_grid.png"
    _plot_candidate_grid(candidates, familywise, candidate_path)
    plot_paths.append(candidate_path)
    block_path = args.out_dir / "fornax_covariance_best_block_scores.png"
    _plot_block_scores(fits, candidates, block_path)
    plot_paths.append(block_path)
    frequency_path = args.out_dir / "fornax_covariance_frequency_diagnostics.png"
    _plot_frequency_summary(frequency_summary, frequency_path)
    plot_paths.append(frequency_path)
    leave_one_out_path = args.out_dir / "fornax_covariance_leave_one_frequency_out.png"
    _plot_leave_one_frequency_out(leave_one_out_summary, leave_one_out_path)
    plot_paths.append(leave_one_out_path)
    power_path = args.out_dir / "fornax_covariance_injection_power.png"
    _plot_injection_power(injection_power, candidates, power_path)
    plot_paths.append(power_path)
    transfer_path = args.out_dir / "fornax_lowrank_signal_transfer.png"
    _plot_signal_transfer(signal_transfer, transfer_path)
    plot_paths.append(transfer_path)
    covariance_path = args.out_dir / "fornax_covariance_structure.png"
    _plot_covariance(covariances, covariance_path)
    plot_paths.append(covariance_path)

    visual_data = {
        "config": {
            "source": SOURCE_NAME,
            "edge_time_fixed_seconds": 0.0,
            "edge_widths_seconds": EDGE_WIDTHS_SECONDS,
            "detection_frequencies_mhz": selected_frequencies,
            "foreground_tracer_frequencies_mhz": TRACER_FREQUENCIES,
            "spectral_templates": TEMPLATE_KINDS,
            "spectral_template_training": "other-block +/-20-minute controls only",
            "covariance_training_control_shifts_seconds": BLANK_CONTROL_SHIFTS,
            "empirical_null_control_shifts_seconds": NULL_CONTROL_SHIFTS,
            "antennas_fit_separately": True,
            "antenna_consistency_required": False,
            "independent_unit": "29.53059-day lunar block",
            "injection_levels_control_scales": INJECTION_LEVELS,
        },
        "candidate_summary": _records(candidates),
        "familywise_summary": _records(familywise),
        "frequency_summary": _records(frequency_summary),
        "leave_one_frequency_out_summary": _records(leave_one_out_summary),
        "injection_power": _records(injection_power),
        "signal_transfer": _records(signal_transfer),
    }
    (args.out_dir / "fornax_covariance_visual_data.json").write_text(
        json.dumps(visual_data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "profiles": str(args.profiles),
        "output_directory": str(args.out_dir),
        "source": SOURCE_NAME,
        "input_profile_rows": int(len(points)),
        "lunar_blocks": len(blocks),
        "antennas": antennas,
        "frequencies_mhz": frequencies,
        "detection_frequencies_mhz": selected_frequencies,
        "edge_time_fixed_seconds": 0.0,
        "edge_widths_seconds": EDGE_WIDTHS_SECONDS,
        "crossfit_unit": "lunar_block",
        "antennas_fit_separately": True,
        "control_shifts_seconds": NULL_CONTROL_SHIFTS,
        "control_combinations": int(len(NULL_CONTROL_SHIFTS) ** len(blocks)),
        "generated_plots": [path.name for path in plot_paths],
        "claim_status": "Fornax A covariance sensitivity experiment; detection assessment requires visual review",
    }
    (args.out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
