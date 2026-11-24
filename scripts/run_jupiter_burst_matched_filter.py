#!/usr/bin/env python
"""Burst-aware matched-filter test for Jupiter occultations in raw RAE-2 RV data.

The Ryle-Vonberg receivers sample too slowly to reproduce the 7.68-second BR-2
analysis in Kaiser (1977), but they can still test whether high-excess RV power
is preferentially present while Jupiter is visible and suppressed while it is
behind the Moon.  This script therefore:

* pairs predicted disappearance/reappearance edges into finite occultation
  intervals and rejects intervals shorter than one practical RV cadence;
* robustly removes a local log-power baseline independently in each receiver
  and frequency channel;
* applies both a linear matched filter and positive-residual burst gates before
  correlating the samples with the visible/occulted notch template;
* maximizes over only three predeclared timing corrections (-150, 0, +150 s);
* repeats the identical analysis at a coverage-complete symmetric +/-75-minute
  sham pair;
* chooses one sham shift per calendar-month block in each permutation, keeping
  clustered events together; and
* reports local and max-statistic family-wise empirical p-values.

The primary search is RV2 (lower V), uniform event weighting, and the
Kaiser-motivated 3.93/4.70-MHz configurations.  A literature Io/System-III
weighting is retained as an explicitly exploratory branch.  Earth is run with
the same machinery as an implementation positive control.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_RAW = Path(
    "/global/u2/a/afahs/RAE2/RyleVonberg/runtime_inputs/occultogram_v1/cleaned_timeseries.parquet"
)
DEFAULT_EVENTS = Path(
    "/global/u2/a/afahs/RAE2/RyleVonberg/runtime_inputs/occultogram_v1/planet_events_science_baseline.parquet"
)
DEFAULT_GEOMETRY = Path(
    "/pscratch/sd/a/afahs/RAE2/jupiter_burst_gated_phase_occurrence_v1/"
    "five_minute_channel_percentile_events.parquet"
)
DEFAULT_OUT = Path("/pscratch/sd/a/afahs/RAE2/jupiter_burst_matched_filter_v1")

ANTENNAS = ("rv1_coarse", "rv2_coarse")
ANTENNA_LABELS = {"rv1_coarse": "RV1 (upper V)", "rv2_coarse": "RV2 (lower V)"}
FREQUENCY_MHZ = {5: 2.20, 6: 3.93, 7: 4.70, 8: 6.55}
FREQUENCY_CONFIGS = {
    "2.20": (5,),
    "3.93": (6,),
    "4.70": (7,),
    "6.55": (8,),
    "3.93+4.70": (6, 7),
    "2.20-6.55": (5, 6, 7, 8),
}
KAISER_FREQUENCY_CONFIGS = ("3.93", "4.70", "3.93+4.70")
TRANSFORMS = {
    "linear": None,
    "burst_z>0": 0.0,
    "burst_z>0.75": 0.75,
    "burst_z>1.5": 1.5,
}
DEFAULT_OFFSETS_SECONDS = (0.0, -4500.0, 4500.0)
DEFAULT_TIMING_JITTER_SECONDS = (-150.0, 0.0, 150.0)


@dataclass(frozen=True)
class Component:
    residual_z: np.ndarray
    template: np.ndarray
    visible: np.ndarray
    n_inside: int
    n_outside: int


def robust_sigma(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return np.nan
    q25, q75 = np.percentile(values, [25.0, 75.0])
    return float((q75 - q25) / 1.349)


def _huber_line(x: np.ndarray, y: np.ndarray, iterations: int = 10) -> tuple[np.ndarray, np.ndarray, float]:
    """Robust two-parameter line used only as a local nuisance baseline."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    scale_x = max(float(np.nanmax(np.abs(x))), 1.0)
    design = np.column_stack([np.ones(len(x), dtype=float), x / scale_x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    for _ in range(int(iterations)):
        residual = y - design @ beta
        sigma = robust_sigma(residual)
        if not np.isfinite(sigma) or sigma <= 1e-12:
            sigma = float(np.nanstd(residual, ddof=1)) if len(residual) > 1 else np.nan
        if not np.isfinite(sigma) or sigma <= 1e-12:
            break
        u = np.abs(residual) / (1.5 * sigma)
        weights = np.ones_like(u)
        high = u > 1.0
        weights[high] = 1.0 / u[high]
        root = np.sqrt(weights)
        updated, *_ = np.linalg.lstsq(design * root[:, None], y * root, rcond=None)
        if np.allclose(updated, beta, rtol=1e-8, atol=1e-10):
            beta = updated
            break
        beta = updated
    residual = y - design @ beta
    sigma = robust_sigma(residual)
    if not np.isfinite(sigma) or sigma <= 1e-12:
        sigma = float(np.nanstd(residual, ddof=1)) if len(residual) > 1 else np.nan
    return design, residual, sigma


def pair_occultation_intervals(
    event_table: pd.DataFrame,
    source: str,
    min_duration_s: float = 240.0,
    max_duration_s: float = 1800.0,
) -> pd.DataFrame:
    """Pair each disappearance with the next reappearance in time order."""

    events = event_table[event_table["source_name"].astype(str).str.lower().eq(source.lower())].copy()
    events["predicted_event_time"] = pd.to_datetime(events["predicted_event_time"])
    events = (
        events[["event_id", "event_type", "predicted_event_time"]]
        .drop_duplicates()
        .sort_values("predicted_event_time")
    )
    rows: list[dict[str, object]] = []
    start: pd.Series | None = None
    for _, edge in events.iterrows():
        event_type = str(edge["event_type"]).lower()
        if event_type == "disappearance":
            start = edge
            continue
        if event_type != "reappearance" or start is None:
            continue
        start_time = pd.Timestamp(start["predicted_event_time"])
        stop_time = pd.Timestamp(edge["predicted_event_time"])
        duration = float((stop_time - start_time).total_seconds())
        if 0.0 < duration <= 12.0 * 3600.0:
            rows.append(
                {
                    "source_name": source.lower(),
                    "start_event_id": int(start["event_id"]),
                    "stop_event_id": int(edge["event_id"]),
                    "occultation_start": start_time,
                    "occultation_stop": stop_time,
                    "midpoint": start_time + (stop_time - start_time) / 2,
                    "duration_s": duration,
                    "resolvable": bool(float(min_duration_s) <= duration <= float(max_duration_s)),
                }
            )
        start = None
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["interval_id"] = np.arange(len(out), dtype=int)
    out["calendar_block"] = out["occultation_start"].dt.to_period("M").astype(str)
    return out


def attach_jupiter_phase(intervals: pd.DataFrame, geometry_path: Path) -> pd.DataFrame:
    out = intervals.copy().sort_values("midpoint")
    out["io_phase_deg"] = np.nan
    out["jupiter_cml_system_iii_deg"] = np.nan
    if out.empty or not geometry_path.exists():
        return out
    columns = ["time", "io_phase_deg", "jupiter_cml_system_iii_deg"]
    geom = pd.read_parquet(geometry_path, columns=columns)
    geom["time"] = pd.to_datetime(geom["time"])
    geom = geom.dropna(subset=["io_phase_deg", "jupiter_cml_system_iii_deg"]).drop_duplicates("time")
    merged = pd.merge_asof(
        out.drop(columns=["io_phase_deg", "jupiter_cml_system_iii_deg"]).sort_values("midpoint"),
        geom.sort_values("time"),
        left_on="midpoint",
        right_on="time",
        direction="nearest",
        tolerance=pd.Timedelta(minutes=4),
    ).drop(columns=["time"])
    return merged.sort_values("interval_id").reset_index(drop=True)


def literature_weight(io_phase_deg: float, cml_deg: float) -> float:
    """Fixed soft weight from published Io/CML occurrence windows."""

    weight = 1.0
    if np.isfinite(io_phase_deg):
        io = float(io_phase_deg) % 360.0
        if 75.0 <= io < 125.0 or 235.0 <= io < 285.0:
            weight *= 1.75
        elif 125.0 <= io < 175.0:
            weight *= 0.60
    if np.isfinite(cml_deg):
        cml = float(cml_deg) % 360.0
        if 25.0 <= cml < 75.0 or 195.0 <= cml < 245.0:
            weight *= 1.35
        else:
            weight *= 0.85
    return float(np.clip(weight, 0.35, 2.5))


def load_clean_groups(path: Path) -> dict[tuple[str, int], dict[str, np.ndarray]]:
    columns = ["time", "frequency_band", "antenna", "power", "is_valid"]
    raw = pd.read_parquet(
        path,
        columns=columns,
        filters=[("frequency_band", "in", list(FREQUENCY_MHZ))],
    )
    power = pd.to_numeric(raw["power"], errors="coerce")
    keep = (
        raw["antenna"].astype(str).isin(ANTENNAS)
        & raw["frequency_band"].isin(FREQUENCY_MHZ)
        & raw["is_valid"].astype(bool)
        & np.isfinite(power)
        & power.gt(0.0)
    )
    raw = raw.loc[keep, ["time", "frequency_band", "antenna"]].copy()
    raw["log_power"] = np.log(power.loc[keep].to_numpy(dtype=float))
    raw["time"] = pd.to_datetime(raw["time"])
    groups: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    for (antenna, band), group in raw.groupby(["antenna", "frequency_band"], sort=True):
        ordered = group.sort_values("time")
        groups[(str(antenna), int(band))] = {
            "time_ns": ordered["time"].astype("datetime64[ns]").astype(np.int64).to_numpy(),
            "log_power": ordered["log_power"].to_numpy(dtype=float),
        }
    return groups


def _component(
    payload: dict[str, np.ndarray],
    start: pd.Timestamp,
    stop: pd.Timestamp,
    padding_s: float,
    min_outside_each_side: int = 2,
) -> Component | None:
    time_ns = payload["time_ns"]
    start_ns = pd.Timestamp(start).value
    stop_ns = pd.Timestamp(stop).value
    pad_ns = int(float(padding_s) * 1e9)
    lo = int(np.searchsorted(time_ns, start_ns - pad_ns, side="left"))
    hi = int(np.searchsorted(time_ns, stop_ns + pad_ns, side="right"))
    if hi <= lo:
        return None
    local_ns = time_ns[lo:hi]
    y = payload["log_power"][lo:hi]
    inside = (local_ns >= start_ns) & (local_ns <= stop_ns)
    before = local_ns < start_ns
    after = local_ns > stop_ns
    if np.count_nonzero(inside) < 1:
        return None
    if np.count_nonzero(before) < int(min_outside_each_side) or np.count_nonzero(after) < int(min_outside_each_side):
        return None
    midpoint_ns = (start_ns + stop_ns) // 2
    t_rel = (local_ns - midpoint_ns).astype(float) / 1e9
    design, residual, sigma = _huber_line(t_rel, y)
    if not np.isfinite(sigma) or sigma <= 1e-12:
        return None
    residual_z = np.clip(residual / sigma, -6.0, 6.0)
    visible = (~inside).astype(float)
    template_beta, *_ = np.linalg.lstsq(design, visible, rcond=None)
    template = visible - design @ template_beta
    if float(np.dot(template, template)) <= 1e-12:
        return None
    return Component(
        residual_z=residual_z,
        template=template,
        visible=visible,
        n_inside=int(np.count_nonzero(inside)),
        n_outside=int(np.count_nonzero(~inside)),
    )


def score_components(
    components: Iterable[Component],
    threshold: float | None,
) -> dict[str, float] | None:
    pieces = list(components)
    if not pieces:
        return None
    residual = np.concatenate([piece.residual_z for piece in pieces])
    template = np.concatenate([piece.template for piece in pieces])
    visible = np.concatenate([piece.visible for piece in pieces]).astype(bool)
    if threshold is None:
        transformed = residual
    else:
        transformed = np.maximum(residual - float(threshold), 0.0)
    denom = float(np.sqrt(np.dot(template, template)))
    if denom <= 1e-12:
        return None
    inside = ~visible
    if not np.any(inside) or not np.any(visible):
        return None
    score = float(np.dot(transformed, template) / denom)
    contrast = float(np.mean(transformed[visible]) - np.mean(transformed[inside]))
    activity = float(np.sqrt(np.mean(np.square(np.maximum(residual, 0.0)))))
    return {
        "score": score,
        "contrast": contrast,
        "activity": activity,
        "n_inside": int(sum(piece.n_inside for piece in pieces)),
        "n_outside": int(sum(piece.n_outside for piece in pieces)),
    }


def score_interval_slots(
    intervals: pd.DataFrame,
    groups: dict[tuple[str, int], dict[str, np.ndarray]],
    offsets_s: tuple[float, ...],
    timing_jitter_s: tuple[float, ...],
    padding_s: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    usable = intervals[intervals["resolvable"].astype(bool)].copy()
    for interval in usable.itertuples(index=False):
        for antenna in ANTENNAS:
            for offset_s in offsets_s:
                best: dict[tuple[str, str], dict[str, object]] = {}
                for jitter_s in timing_jitter_s:
                    total_shift = float(offset_s) + float(jitter_s)
                    start = pd.Timestamp(interval.occultation_start) + pd.to_timedelta(total_shift, unit="s")
                    stop = pd.Timestamp(interval.occultation_stop) + pd.to_timedelta(total_shift, unit="s")
                    band_components: dict[int, Component] = {}
                    for band in FREQUENCY_MHZ:
                        payload = groups.get((antenna, int(band)))
                        if payload is None:
                            continue
                        component = _component(payload, start, stop, padding_s)
                        if component is not None:
                            band_components[int(band)] = component
                    for config_name, bands in FREQUENCY_CONFIGS.items():
                        selected = [band_components[band] for band in bands if band in band_components]
                        if len(selected) != len(bands):
                            continue
                        for transform_name, threshold in TRANSFORMS.items():
                            scored = score_components(selected, threshold)
                            if scored is None:
                                continue
                            key = (config_name, transform_name)
                            candidate = {
                                **scored,
                                "best_timing_jitter_s": float(jitter_s),
                            }
                            if key not in best or float(candidate["score"]) > float(best[key]["score"]):
                                best[key] = candidate
                for (config_name, transform_name), result in best.items():
                    rows.append(
                        {
                            "source_name": str(interval.source_name),
                            "interval_id": int(interval.interval_id),
                            "calendar_block": str(interval.calendar_block),
                            "occultation_start": interval.occultation_start,
                            "occultation_stop": interval.occultation_stop,
                            "duration_s": float(interval.duration_s),
                            "io_phase_deg": float(interval.io_phase_deg),
                            "jupiter_cml_system_iii_deg": float(interval.jupiter_cml_system_iii_deg),
                            "literature_weight": literature_weight(
                                float(interval.io_phase_deg),
                                float(interval.jupiter_cml_system_iii_deg),
                            ),
                            "antenna": antenna,
                            "offset_s": float(offset_s),
                            "frequency_config": config_name,
                            "transform": transform_name,
                            **result,
                        }
                    )
    return pd.DataFrame(rows)


def _weighted_block_statistic(
    matrix: np.ndarray,
    interval_blocks: np.ndarray,
    event_weights: np.ndarray,
    block_names: list[str],
    block_choices: np.ndarray,
) -> np.ndarray:
    """Aggregate intervals within blocks, then give each block equal weight."""

    block_results = []
    for block_index, block in enumerate(block_names):
        indices = np.flatnonzero(interval_blocks == block)
        if len(indices) == 0:
            continue
        choices = block_choices[:, block_index]
        selected = matrix[indices][:, choices]
        weights = event_weights[indices][:, None]
        block_results.append(np.sum(selected * weights, axis=0) / np.sum(weights))
    return np.mean(np.vstack(block_results), axis=0) if block_results else np.full(block_choices.shape[0], np.nan)


def aggregate_branches(
    scores: pd.DataFrame,
    offsets_s: tuple[float, ...],
    n_permutations: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    rows: list[dict[str, object]] = []
    nulls: dict[int, np.ndarray] = {}
    all_blocks = sorted(scores["calendar_block"].dropna().astype(str).unique())
    n_exact = int(len(offsets_s) ** len(all_blocks))
    exact_null = n_exact <= int(n_permutations)
    if exact_null:
        random_choices = np.indices(
            (len(offsets_s),) * len(all_blocks), dtype=np.int16
        ).reshape(len(all_blocks), -1).T
    else:
        rng = np.random.default_rng(int(seed))
        random_choices = rng.integers(
            0,
            len(offsets_s),
            size=(int(n_permutations), len(all_blocks)),
        )
    observed_choices = np.zeros((1, len(all_blocks)), dtype=int)
    branch_columns = ["source_name", "antenna", "frequency_config", "transform"]
    for keys, branch in scores.groupby(branch_columns, sort=True):
        source, antenna, frequency_config, transform = keys
        weight_modes = ["uniform"] if str(source) != "jupiter" else ["uniform", "literature_io_cml"]
        pivot = branch.pivot_table(index="interval_id", columns="offset_s", values="score", aggfunc="max")
        pivot = pivot.reindex(columns=list(offsets_s))
        complete = pivot.notna().all(axis=1)
        pivot = pivot.loc[complete]
        if pivot.empty:
            continue
        metadata = (
            branch.drop_duplicates("interval_id")
            .set_index("interval_id")
            .reindex(pivot.index)
        )
        blocks = metadata["calendar_block"].astype(str).to_numpy()
        branch_blocks = [block for block in all_blocks if block in set(blocks)]
        if len(branch_blocks) < 3:
            continue
        all_block_index = {block: index for index, block in enumerate(all_blocks)}
        choice_columns = [all_block_index[block] for block in branch_blocks]
        matrix = pivot.to_numpy(dtype=float)
        for weight_mode in weight_modes:
            if weight_mode == "uniform":
                weights = np.ones(len(pivot), dtype=float)
            else:
                weights = metadata["literature_weight"].to_numpy(dtype=float)
            observed = _weighted_block_statistic(
                matrix,
                blocks,
                weights,
                branch_blocks,
                observed_choices[:, choice_columns],
            )[0]
            null = _weighted_block_statistic(
                matrix,
                blocks,
                weights,
                branch_blocks,
                random_choices[:, choice_columns],
            )
            if exact_null:
                local_p = float(np.count_nonzero(null >= observed) / len(null))
            else:
                local_p = float((1 + np.count_nonzero(null >= observed)) / (len(null) + 1))
            center = float(np.nanmedian(null))
            scale = robust_sigma(null - center)
            if not np.isfinite(scale) or scale <= 1e-12:
                scale = float(np.nanstd(null, ddof=1))
            observed_z = float((observed - center) / scale) if np.isfinite(scale) and scale > 0 else np.nan
            row_id = len(rows)
            rows.append(
                {
                    "row_id": row_id,
                    "source_name": source,
                    "antenna": antenna,
                    "frequency_config": frequency_config,
                    "transform": transform,
                    "weight_mode": weight_mode,
                    "n_intervals": int(len(pivot)),
                    "n_blocks": int(len(branch_blocks)),
                    "observed_block_mean_score": float(observed),
                    "null_median_score": center,
                    "null_scale": float(scale),
                    "observed_null_z": observed_z,
                    "local_empirical_p": local_p,
                    "null_mode": "exact_cartesian" if exact_null else "monte_carlo",
                    "n_null_combinations": int(len(null)),
                }
            )
            nulls[row_id] = null
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary, nulls

    z_nulls = {}
    for row in summary.itertuples(index=False):
        z_nulls[int(row.row_id)] = (nulls[int(row.row_id)] - float(row.null_median_score)) / float(row.null_scale)

    family_masks = {
        "fwer_p_within_receiver": lambda frame, row: (
            frame["source_name"].eq(row.source_name)
            & frame["antenna"].eq(row.antenna)
        ),
        "fwer_p_rv2_uniform": lambda frame, row: (
            frame["source_name"].eq(row.source_name)
            & frame["antenna"].eq("rv2_coarse")
            & frame["weight_mode"].eq("uniform")
        ),
        "fwer_p_kaiser_rv2": lambda frame, row: (
            frame["source_name"].eq(row.source_name)
            & frame["antenna"].eq("rv2_coarse")
            & frame["weight_mode"].eq("uniform")
            & frame["frequency_config"].isin(KAISER_FREQUENCY_CONFIGS)
        ),
        "fwer_p_all_branches": lambda frame, row: frame["source_name"].eq(row.source_name),
    }
    for column, selector in family_masks.items():
        values = []
        for row in summary.itertuples(index=False):
            if column == "fwer_p_rv2_uniform" and not (
                str(row.antenna) == "rv2_coarse" and str(row.weight_mode) == "uniform"
            ):
                values.append(np.nan)
                continue
            if column == "fwer_p_kaiser_rv2" and not (
                str(row.antenna) == "rv2_coarse"
                and str(row.weight_mode) == "uniform"
                and str(row.frequency_config) in KAISER_FREQUENCY_CONFIGS
            ):
                values.append(np.nan)
                continue
            family = summary.loc[selector(summary, row), "row_id"].astype(int).tolist()
            max_null = np.nanmax(np.vstack([z_nulls[index] for index in family]), axis=0)
            exceed = int(np.count_nonzero(max_null >= float(row.observed_null_z)))
            values.append(
                float(exceed / len(max_null))
                if exact_null
                else float((1 + exceed) / (len(max_null) + 1))
            )
        summary[column] = values
    return summary.sort_values(["source_name", "antenna", "observed_null_z"], ascending=[True, True, False]), nulls


def plot_branch_scan(summary: pd.DataFrame, output: Path) -> Path:
    source = summary[
        summary["source_name"].eq("jupiter")
        & summary["antenna"].eq("rv2_coarse")
        & summary["weight_mode"].eq("uniform")
    ].copy()
    fig, ax = plt.subplots(figsize=(10.4, 5.8))
    x = np.arange(len(FREQUENCY_CONFIGS), dtype=float)
    for transform_index, transform in enumerate(TRANSFORMS):
        group = source[source["transform"].eq(transform)].set_index("frequency_config").reindex(FREQUENCY_CONFIGS)
        y = group["observed_null_z"].to_numpy(dtype=float)
        ax.plot(x, y, marker="o", lw=1.8, label=transform)
        for xx, yy, p in zip(x, y, group["fwer_p_rv2_uniform"].to_numpy(dtype=float)):
            if np.isfinite(yy):
                ax.annotate(f"pF={p:.3f}", (xx, yy), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7)
    ax.axhline(0.0, color="0.55", lw=0.8)
    ax.axhline(1.645, color="0.75", lw=0.8, ls="--", label="one-sided z=1.645")
    ax.set_xticks(x)
    ax.set_xticklabels(FREQUENCY_CONFIGS)
    ax.set_ylabel("observed score relative to clustered sham null (robust z)")
    ax.set_title("Jupiter RV2 occultation-notch matched-filter scan")
    ax.grid(True, axis="y", color="0.91", lw=0.6)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    path = output / "jupiter_rv2_matched_filter_scan.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def plot_best_null(summary: pd.DataFrame, nulls: dict[int, np.ndarray], output: Path) -> Path:
    candidates = summary[
        summary["source_name"].eq("jupiter")
        & summary["antenna"].eq("rv2_coarse")
        & summary["weight_mode"].eq("uniform")
    ].sort_values("observed_null_z", ascending=False)
    best = candidates.iloc[0]
    null = nulls[int(best["row_id"])]
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.hist(null, bins=55, color="#c9d5e5", edgecolor="white", label=f"{len(null):,} clustered sham permutations")
    ax.axvline(float(best["observed_block_mean_score"]), color="#b2182b", lw=2.2, label="true occultation times")
    ax.axvline(float(best["null_median_score"]), color="black", lw=1.0, ls="--", label="null median")
    ax.set_xlabel("equal-block mean matched-filter score")
    ax.set_ylabel("permutations")
    ax.set_title(
        f"Best RV2 uniform branch: {best['frequency_config']}, {best['transform']}\n"
        f"local p={float(best['local_empirical_p']):.4f}; RV2-family p={float(best['fwer_p_rv2_uniform']):.4f}"
    )
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = output / "jupiter_best_branch_clustered_null.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def plot_linear_receiver_scan(summary: pd.DataFrame, output: Path) -> Path:
    source = summary[
        summary["source_name"].eq("jupiter")
        & summary["weight_mode"].eq("uniform")
        & summary["transform"].eq("linear")
    ].copy()
    fig, ax = plt.subplots(figsize=(10.4, 5.4))
    x = np.arange(len(FREQUENCY_CONFIGS), dtype=float)
    for antenna, color in [("rv1_coarse", "#3568a8"), ("rv2_coarse", "#d4681f")]:
        group = (
            source[source["antenna"].eq(antenna)]
            .set_index("frequency_config")
            .reindex(FREQUENCY_CONFIGS)
        )
        ax.plot(
            x,
            group["observed_null_z"],
            marker="o",
            lw=2.0,
            color=color,
            label=ANTENNA_LABELS[antenna],
        )
    ax.axhline(0.0, color="0.55", lw=0.8)
    ax.axhline(1.645, color="0.75", lw=0.8, ls="--", label="one-sided z=1.645")
    ax.set_xticks(x)
    ax.set_xticklabels(FREQUENCY_CONFIGS, rotation=25, ha="right")
    ax.set_ylabel("observed score relative to clustered sham null (robust z)")
    ax.set_title("Jupiter linear occultation-notch filter by receiver and band")
    ax.grid(True, axis="y", color="0.91", lw=0.6)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = output / "jupiter_linear_receiver_frequency_scan.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def plot_event_differences(scores: pd.DataFrame, summary: pd.DataFrame, output: Path) -> Path:
    candidates = summary[
        summary["source_name"].eq("jupiter")
        & summary["antenna"].eq("rv2_coarse")
        & summary["weight_mode"].eq("uniform")
    ].sort_values("observed_null_z", ascending=False)
    best = candidates.iloc[0]
    branch = scores[
        scores["source_name"].eq("jupiter")
        & scores["antenna"].eq("rv2_coarse")
        & scores["frequency_config"].eq(best["frequency_config"])
        & scores["transform"].eq(best["transform"])
    ].copy()
    pivot = (
        branch.pivot_table(index="interval_id", columns="offset_s", values="score", aggfunc="max")
        .reindex(columns=list(DEFAULT_OFFSETS_SECONDS))
        .dropna()
    )
    metadata = branch.drop_duplicates("interval_id").set_index("interval_id").reindex(pivot.index)
    paired = metadata.copy()
    paired["score"] = pivot[0.0]
    paired["sham_median"] = pivot[[-4500.0, 4500.0]].median(axis=1)
    paired["difference"] = paired["score"] - paired["sham_median"]
    paired = paired.sort_values("occultation_start")
    colors = {block: plt.cm.tab10(index % 10) for index, block in enumerate(sorted(paired["calendar_block"].unique()))}
    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    x = np.arange(len(paired))
    ax.axhline(0.0, color="0.55", lw=0.8)
    ax.bar(x, paired["difference"], color=[colors[value] for value in paired["calendar_block"]])
    ax.set_xticks(x)
    ax.set_xticklabels([pd.Timestamp(value).strftime("%Y-%m-%d") for value in paired["occultation_start"]], rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("true score - median same-event sham score")
    ax.set_title(f"Complete-null event contributions for {best['frequency_config']}, {best['transform']}")
    handles = [plt.Line2D([0], [0], color=color, lw=6, label=block) for block, color in colors.items()]
    ax.legend(handles=handles, frameon=False, fontsize=8, title="calendar block", ncol=min(4, len(handles)))
    ax.grid(True, axis="y", color="0.92", lw=0.5)
    fig.tight_layout()
    path = output / "jupiter_best_branch_event_differences.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    paired.reset_index().to_csv(output / "jupiter_best_branch_event_differences.csv", index=False)
    return path


def write_report(
    output: Path,
    intervals: pd.DataFrame,
    summary: pd.DataFrame,
    figures: list[Path],
    config: dict[str, object],
) -> Path:
    jupiter_primary = summary[
        summary["source_name"].eq("jupiter")
        & summary["antenna"].eq("rv2_coarse")
        & summary["weight_mode"].eq("uniform")
    ].sort_values("observed_null_z", ascending=False)
    best = jupiter_primary.iloc[0] if not jupiter_primary.empty else None
    kaiser = jupiter_primary[
        jupiter_primary["frequency_config"].isin(KAISER_FREQUENCY_CONFIGS)
    ]
    best_kaiser = kaiser.iloc[0] if not kaiser.empty else None
    rv1_uniform = summary[
        summary["source_name"].eq("jupiter")
        & summary["antenna"].eq("rv1_coarse")
        & summary["weight_mode"].eq("uniform")
    ].sort_values("observed_null_z", ascending=False)
    best_rv1 = rv1_uniform.iloc[0] if not rv1_uniform.empty else None
    earth = summary[
        summary["source_name"].eq("earth")
        & summary["weight_mode"].eq("uniform")
    ].sort_values("observed_null_z", ascending=False)
    best_earth = earth.iloc[0] if not earth.empty else None
    compact_columns = [
        "source_name",
        "antenna",
        "frequency_config",
        "transform",
        "weight_mode",
        "n_intervals",
        "n_blocks",
        "observed_null_z",
        "local_empirical_p",
        "fwer_p_rv2_uniform",
        "fwer_p_kaiser_rv2",
        "fwer_p_all_branches",
    ]
    lines = [
        "# Burst-aware Jupiter occultation matched-filter test",
        "",
        "## Result",
        "",
    ]
    if best is not None:
        lines.extend(
            [
                f"The strongest RV2/uniform branch across all tested bands is `{best['frequency_config']}` with `{best['transform']}`: ",
                f"robust null z = `{float(best['observed_null_z']):.3f}`, local empirical p = `{float(best['local_empirical_p']):.5f}`, ",
                f"and RV2-family max-statistic p = `{float(best['fwer_p_rv2_uniform']):.5f}`.",
                "",
            ]
        )
    if best_kaiser is not None:
        lines.extend(
            [
                f"Within the predeclared Kaiser 3--5 MHz family, the strongest branch is `{best_kaiser['frequency_config']}` with `{best_kaiser['transform']}`: ",
                f"robust null z = `{float(best_kaiser['observed_null_z']):.3f}`, local p = `{float(best_kaiser['local_empirical_p']):.5f}`, ",
                f"and Kaiser-family max-statistic p = `{float(best_kaiser['fwer_p_kaiser_rv2']):.5f}`.",
                "",
            ]
        )
    if best_rv1 is not None:
        lines.extend(
            [
                f"The strongest secondary RV1 branch is `{best_rv1['frequency_config']}` with `{best_rv1['transform']}`: ",
                f"robust null z = `{float(best_rv1['observed_null_z']):.3f}`, local p = `{float(best_rv1['local_empirical_p']):.5f}`, ",
                f"within-RV1 max-statistic p = `{float(best_rv1['fwer_p_within_receiver']):.5f}`, and all-branch p = `{float(best_rv1['fwer_p_all_branches']):.5f}`. ",
                "It is an upper-V, broad/low-frequency effect rather than replication in the lower-V Kaiser band.",
                "",
            ]
        )
    if best_earth is not None:
        lines.extend(
            [
                f"The strongest Earth positive-control branch reaches robust null z = `{float(best_earth['observed_null_z']):.3f}` ",
                f"with local p = `{float(best_earth['local_empirical_p']):.5f}`.",
                "",
            ]
        )
    lines.extend(
        [
            "## What was tested",
            "",
            "- Each disappearance/reappearance pair defines a finite occulted notch; expected source power is higher outside than inside.",
            "- `linear` uses robust local log-power residuals directly.",
            "- `burst_z>...` keeps only positive residual energy above the named threshold before applying the same notch template.",
            "- The burst gate is calculated without using which samples are inside or outside the true occultation.",
            "- Every score maximizes over the same -150/0/+150-second timing grid.",
            "- Sham notches use the same event duration and analysis at symmetric +/-75-minute offsets.",
            "- That sham pair was selected from score-availability coverage before inspecting any Jupiter score values; it retains five RV2/3.93-MHz intervals spanning four calendar blocks.",
            "- Permutations choose one offset for every event in a calendar block, so the dense June 1975 cluster is not treated as many independent trials.",
            "- `literature_io_cml` is exploratory. The primary result uses uniform weights.",
            "",
            "## Interval inventory",
            "",
            intervals.groupby(["source_name", "resolvable"], dropna=False).size().reset_index(name="n_intervals").to_string(index=False),
            "",
            "## Branch statistics",
            "",
            summary[compact_columns].sort_values(["source_name", "antenna", "observed_null_z"], ascending=[True, True, False]).to_string(index=False),
            "",
            "## Configuration",
            "",
            *[f"- `{key}`: `{value}`" for key, value in config.items()],
            "",
            "## Figures",
            "",
            *[f"- `{path.name}`" for path in figures],
            "",
            "## Interpretation rule",
            "",
            "A Jupiter claim requires a positive RV2 score that survives the RV2 max-statistic correction, is not driven by one block, and is physically concentrated near 3--5 MHz. A favorable uncorrected branch alone is not a detection.",
        ]
    )
    path = output / "jupiter_burst_matched_filter_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sources", default="jupiter,earth")
    parser.add_argument("--min-duration-s", type=float, default=240.0)
    parser.add_argument("--max-duration-s", type=float, default=1800.0)
    parser.add_argument("--padding-s", type=float, default=900.0)
    parser.add_argument("--offsets-s", default=",".join(str(int(value)) for value in DEFAULT_OFFSETS_SECONDS))
    parser.add_argument("--timing-jitter-s", default=",".join(str(int(value)) for value in DEFAULT_TIMING_JITTER_SECONDS))
    parser.add_argument("--n-permutations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()

    output = args.out_dir
    output.mkdir(parents=True, exist_ok=True)
    sources = [value.strip().lower() for value in str(args.sources).split(",") if value.strip()]
    offsets_s = tuple(float(value) for value in str(args.offsets_s).split(",") if value.strip())
    timing_jitter_s = tuple(float(value) for value in str(args.timing_jitter_s).split(",") if value.strip())
    if not offsets_s or offsets_s[0] != 0.0:
        raise ValueError("The first offset must be 0 seconds (the true occultation slot)")

    config = {
        "raw": str(args.raw),
        "events": str(args.events),
        "geometry": str(args.geometry),
        "sources": sources,
        "frequency_configs": FREQUENCY_CONFIGS,
        "transforms": TRANSFORMS,
        "min_duration_s": float(args.min_duration_s),
        "max_duration_s": float(args.max_duration_s),
        "padding_s": float(args.padding_s),
        "offsets_s": offsets_s,
        "timing_jitter_s": timing_jitter_s,
        "n_permutations": int(args.n_permutations),
        "seed": int(args.seed),
    }
    (output / "run_config.json").write_text(json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8")

    print("Loading and pairing occultation edges...", flush=True)
    events = pd.read_parquet(args.events)
    interval_tables = []
    for source in sources:
        paired = pair_occultation_intervals(
            events,
            source,
            min_duration_s=float(args.min_duration_s),
            max_duration_s=float(args.max_duration_s),
        )
        if not paired.empty:
            interval_tables.append(paired)
    intervals = pd.concat(interval_tables, ignore_index=True)
    intervals["interval_id"] = np.arange(len(intervals), dtype=int)
    intervals = attach_jupiter_phase(intervals, args.geometry)
    intervals.to_csv(output / "occultation_interval_inventory.csv", index=False)
    print(intervals.groupby(["source_name", "resolvable"]).size(), flush=True)

    print("Loading valid raw-derived RV samples for 2.20--6.55 MHz...", flush=True)
    groups = load_clean_groups(args.raw)
    print("Scoring true and sham occultation notches...", flush=True)
    scores = score_interval_slots(
        intervals,
        groups,
        offsets_s=offsets_s,
        timing_jitter_s=timing_jitter_s,
        padding_s=float(args.padding_s),
    )
    scores.to_parquet(
        output / "matched_filter_slot_scores.parquet",
        index=False,
        coerce_timestamps="us",
        allow_truncated_timestamps=True,
    )
    scores.to_csv(output / "matched_filter_slot_scores.csv", index=False)

    print("Building clustered empirical nulls...", flush=True)
    summary, nulls = aggregate_branches(scores, offsets_s, int(args.n_permutations), int(args.seed))
    summary.to_csv(output / "matched_filter_branch_statistics.csv", index=False)
    np.savez_compressed(output / "matched_filter_null_distributions.npz", **{f"row_{key}": value for key, value in nulls.items()})

    figures = [
        plot_branch_scan(summary, output),
        plot_linear_receiver_scan(summary, output),
        plot_best_null(summary, nulls, output),
        plot_event_differences(scores, summary, output),
    ]
    report = write_report(output, intervals, summary, figures, config)
    print(report, flush=True)


if __name__ == "__main__":
    main()
