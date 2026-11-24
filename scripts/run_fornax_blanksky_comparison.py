#!/usr/bin/env python
"""Compare fixed-zero Fornax A occultation statistics with nearby blank sky.

The confirmatory statistic is frozen from the completed Fornax A covariance
trial before any spatial controls are inspected.  Twenty-four positions on
2- and 5-degree rings provide a local sky null while retaining broadly similar
Galactic foreground and lunar/spacecraft geometry.  Antennas remain separate.

Only compact block profiles, scores, selected event occultograms, plots, and
metadata are retained.  Large event-level tables for every blank position are
never written.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import astropy.units as u  # noqa: E402
from astropy.coordinates import SkyCoord  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from rylevonberg.constants import frequency_mhz_for_band  # noqa: E402
from rylevonberg.events import predict_source_events  # noqa: E402
from rylevonberg.occultogram import (  # noqa: E402
    OccultogramConfig,
    fold_occultogram_points,
    lunar_block_id,
)
from rylevonberg.table_io import write_table  # noqa: E402
from rylevonberg.util import robust_sigma, software_versions  # noqa: E402
from run_earth_lowrank_covariance import (  # noqa: E402
    FIT_HALF_WINDOW_SECONDS,
    _available_frequencies,
    _block_profiles,
    _crossfit_lowrank,
    _estimate_covariance,
    _frequency_column,
    _profile_matrix,
    _residual_wide,
    _wide_profiles,
)
from run_fornax_covariance_fixedzero import (  # noqa: E402
    DETECTION_FREQUENCIES,
    NULL_CONTROL_SHIFTS,
    _candidate_summaries,
    _control_templates,
    _joint_fit,
    _selected_covariance,
)


DEFAULT_CLEAN = Path(
    "/global/u2/a/afahs/RAE2/RyleVonberg/runtime_inputs/occultogram_v1/cleaned_timeseries.parquet"
)
DEFAULT_TARGET_PROFILES = Path(
    "/pscratch/sd/a/afahs/RAE2/occultograms_morphology_v1/occultogram_event_profiles.parquet"
)
DEFAULT_REFERENCE = Path(
    "/pscratch/sd/a/afahs/RAE2/fornax_covariance_fixedzero_v2/candidate_empirical_null_summary.parquet"
)
DEFAULT_OUT = Path("/pscratch/sd/a/afahs/RAE2/fornax_blanksky_comparison_v1")

TARGET_NAME = "fornax_a"
TARGET_RA_ICRS_DEG = 50.67383
TARGET_DEC_ICRS_DEG = -37.20823
RING_RADII_DEG = (2.0, 5.0)
POSITIONS_PER_RING = 12
FREQUENCY_BANDS = tuple(range(1, 10))
ANTENNAS = ("rv1_coarse", "rv2_coarse")
PROFILE_DISPLAY_HALF_WINDOW_SECONDS = 1200.0

# These configurations were frozen from the target-only v2 analysis.  The
# first entry is the primary spatial test.  The sharp RV1 raw edge is retained
# as a duration sensitivity check, not as a second opportunity to select.
FROZEN_CANDIDATES = (
    {
        "candidate_id": "primary_rv1_raw_ramp240",
        "antenna": "rv1_coarse",
        "data_kind": "raw",
        "template_kind": "control_scaled",
        "edge_width_s": 240.0,
        "role": "primary spatial test",
    },
    {
        "candidate_id": "sharp_rv1_raw_step",
        "antenna": "rv1_coarse",
        "data_kind": "raw",
        "template_kind": "control_scaled",
        "edge_width_s": 0.0,
        "role": "duration sensitivity",
    },
    {
        "candidate_id": "rv1_lowrank_ramp60",
        "antenna": "rv1_coarse",
        "data_kind": "lowrank",
        "template_kind": "flat_control_median",
        "edge_width_s": 60.0,
        "role": "target-branch winner",
    },
    {
        "candidate_id": "rv2_raw_ramp240",
        "antenna": "rv2_coarse",
        "data_kind": "raw",
        "template_kind": "flat_control_median",
        "edge_width_s": 240.0,
        "role": "target-branch winner",
    },
    {
        "candidate_id": "rv2_lowrank_ramp120",
        "antenna": "rv2_coarse",
        "data_kind": "lowrank",
        "template_kind": "control_scaled",
        "edge_width_s": 120.0,
        "role": "target-branch winner",
    },
)


def _position_table() -> pd.DataFrame:
    target = SkyCoord(
        ra=TARGET_RA_ICRS_DEG * u.deg,
        dec=TARGET_DEC_ICRS_DEG * u.deg,
        frame="icrs",
    )
    galactic_center = SkyCoord(l=0.0 * u.deg, b=0.0 * u.deg, frame="galactic")

    def record(
        name: str,
        coordinate: SkyCoord,
        role: str,
        ring_radius: float,
        position_angle: float,
        east_offset: float,
        north_offset: float,
    ) -> dict[str, object]:
        galactic = coordinate.galactic
        return {
            "source_name": name,
            "kind": "fixed",
            "body_name": "",
            "ra_deg": float(coordinate.icrs.ra.deg),
            "dec_deg": float(coordinate.icrs.dec.deg),
            "frame": "icrs",
            "source_role": role,
            "ring_radius_deg": float(ring_radius),
            "position_angle_deg_east_of_north": float(position_angle),
            "east_offset_deg": float(east_offset),
            "north_offset_deg": float(north_offset),
            "galactic_longitude_deg": float(galactic.l.deg),
            "galactic_latitude_deg": float(galactic.b.deg),
            "separation_from_galactic_center_deg": float(
                coordinate.separation(galactic_center).deg
            ),
            "notes": (
                "Fornax A catalog position"
                if role == "target"
                else "nearby empirical sky control; not guaranteed devoid of every faint radio source"
            ),
        }

    rows = [record(TARGET_NAME, target, "target", 0.0, 0.0, 0.0, 0.0)]
    for radius in RING_RADII_DEG:
        for index in range(POSITIONS_PER_RING):
            angle = 360.0 * index / POSITIONS_PER_RING
            coordinate = target.directional_offset_by(angle * u.deg, radius * u.deg)
            angle_rad = np.deg2rad(angle)
            east = radius * np.sin(angle_rad)
            north = radius * np.cos(angle_rad)
            name = f"fornax_blank_r{int(radius):02d}_pa{int(round(angle)) % 360:03d}"
            rows.append(record(name, coordinate, "blank", radius, angle, east, north))
    return pd.DataFrame.from_records(rows)


def _read_clean(path: Path) -> pd.DataFrame:
    columns = [
        "time",
        "frequency_band",
        "position_x",
        "position_y",
        "position_z",
        "antenna",
        "power",
    ]
    clean = pd.read_parquet(path, columns=columns)
    clean["time"] = pd.to_datetime(clean["time"], errors="coerce")
    clean = clean.dropna(subset=["time"]).reset_index(drop=True)
    return clean


def _channel_arrays(
    clean: pd.DataFrame,
) -> dict[tuple[int, str], tuple[np.ndarray, np.ndarray]]:
    groups: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
    for (band, antenna), group in clean.groupby(
        ["frequency_band", "antenna"], observed=True, sort=False
    ):
        ordered = group.sort_values("time")
        times = pd.DatetimeIndex(ordered["time"]).asi8
        power = pd.to_numeric(ordered["power"], errors="coerce").to_numpy(float)
        groups[(int(band), str(antenna))] = (times, power)
    return groups


def _profile_at_time(
    group_times_ns: np.ndarray,
    group_power: np.ndarray,
    center_time: pd.Timestamp,
    config: OccultogramConfig,
    output_half_window_seconds: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    center_ns = pd.Timestamp(center_time).value
    half_ns = int(config.window_seconds * 1e9)
    lo = int(np.searchsorted(group_times_ns, center_ns - half_ns, side="left"))
    hi = int(np.searchsorted(group_times_ns, center_ns + half_ns, side="right"))
    if hi <= lo:
        return None
    relative = (group_times_ns[lo:hi] - center_ns).astype(float) / 1e9
    power = group_power[lo:hi]
    finite = np.isfinite(power) & (np.abs(relative) <= config.window_seconds)
    if np.count_nonzero(finite) < config.minimum_window_samples:
        return None
    relative = relative[finite]
    power = power[finite]
    sideband = np.abs(relative) >= config.sideband_start_seconds
    if np.count_nonzero(sideband) < config.minimum_sideband_samples:
        return None
    center = float(np.median(power[sideband]))
    scale = float(robust_sigma(power[sideband] - center))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(np.std(power[sideband], ddof=1))
    if not np.isfinite(scale) or scale <= 0.0:
        return None
    normalized = (power - center) / scale
    edges = np.arange(
        -config.window_seconds,
        config.window_seconds + config.bin_seconds,
        config.bin_seconds,
        dtype=float,
    )
    n_bins = len(edges) - 1
    index = np.digitize(relative, edges, right=False) - 1
    index[relative == edges[-1]] = n_bins - 1
    values = np.full(n_bins, np.nan, dtype=float)
    counts = np.zeros(n_bins, dtype=np.int16)
    for bin_index in np.unique(index):
        if bin_index < 0 or bin_index >= n_bins:
            continue
        if abs(float(0.5 * (edges[bin_index] + edges[bin_index + 1]))) > float(
            output_half_window_seconds
        ):
            continue
        selected = index == bin_index
        values[bin_index] = float(np.median(normalized[selected]))
        counts[bin_index] = int(np.count_nonzero(selected))
    return values, counts


def _physical_event_uid(source_name: str, event_time: pd.Timestamp, event_type: str) -> str:
    return "|".join([source_name, pd.Timestamp(event_time).isoformat(), event_type])


def _extract_position(
    source_name: str,
    events: pd.DataFrame,
    channels: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]],
    config: OccultogramConfig,
    capture_event_profiles: bool = False,
    build_block_profiles: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    edges = np.arange(
        -config.window_seconds,
        config.window_seconds + config.bin_seconds,
        config.bin_seconds,
        dtype=float,
    )
    centers = 0.5 * (edges[:-1] + edges[1:])
    block_values: dict[tuple[int, float, str, float], list[np.ndarray]] = defaultdict(list)
    event_records: list[dict[str, object]] = []
    event_order_lookup: dict[str, int] = {}

    for event in events.itertuples(index=False):
        band = int(event.frequency_band)
        antenna = str(event.antenna)
        channel = channels.get((band, antenna))
        if channel is None:
            continue
        frequency = float(frequency_mhz_for_band(band))
        if capture_event_profiles and not build_block_profiles:
            if antenna != "rv1_coarse":
                continue
            if not any(np.isclose(frequency, value) for value in DETECTION_FREQUENCIES):
                continue
        event_time = pd.Timestamp(event.predicted_event_time)
        event_type = str(event.event_type).lower()
        block = lunar_block_id(event_time)
        physical_uid = _physical_event_uid(source_name, event_time, event_type)
        if physical_uid not in event_order_lookup:
            event_order_lookup[physical_uid] = len(event_order_lookup)
        shifts: Iterable[float] = (
            (0.0,)
            if capture_event_profiles and not build_block_profiles
            else config.control_shifts_seconds
        )
        for shift in shifts:
            center_time = event_time + pd.to_timedelta(float(shift), unit="s")
            output_half_window = (
                PROFILE_DISPLAY_HALF_WINDOW_SECONDS
                if np.isclose(shift, 0.0)
                else FIT_HALF_WINDOW_SECONDS
            )
            extracted = _profile_at_time(
                channel[0],
                channel[1],
                center_time,
                config,
                output_half_window_seconds=output_half_window,
            )
            if extracted is None:
                continue
            values, counts = extracted
            if event_type == "reappearance":
                values = values[::-1]
                counts = counts[::-1]
            if build_block_profiles:
                block_values[(block, frequency, antenna, float(shift))].append(values)

            if not capture_event_profiles:
                continue
            if not np.isclose(shift, 0.0):
                continue
            if antenna != "rv1_coarse":
                continue
            if not any(np.isclose(frequency, value) for value in DETECTION_FREQUENCIES):
                continue
            finite = np.isfinite(values)
            for time_value, value, count in zip(
                centers[finite], values[finite], counts[finite]
            ):
                event_records.append(
                    {
                        "source_name": source_name,
                        "physical_event_uid": physical_uid,
                        "event_order": int(event_order_lookup[physical_uid]),
                        "lunar_block": int(block),
                        "event_type": "folded_source_like",
                        "frequency_band": band,
                        "frequency_mhz": frequency,
                        "antenna": antenna,
                        "control_shift_s": 0.0,
                        "t_bin_sec": float(time_value),
                        "normalized_power": float(value),
                        "n_samples": int(count),
                    }
                )

    profile_records: list[dict[str, object]] = []
    for (block, frequency, antenna, shift), arrays in block_values.items():
        matrix = np.stack(arrays)
        median = np.full(matrix.shape[1], np.nan, dtype=float)
        populated = np.any(np.isfinite(matrix), axis=0)
        median[populated] = np.nanmedian(matrix[:, populated], axis=0)
        n_events = np.sum(np.isfinite(matrix), axis=0)
        for time_value, value, count in zip(centers, median, n_events):
            if not np.isfinite(value):
                continue
            profile_records.append(
                {
                    "lunar_block": int(block),
                    "frequency_mhz": float(frequency),
                    "antenna": antenna,
                    "control_shift_s": float(shift),
                    "t_bin_sec": float(time_value),
                    "value": float(value),
                    "n_events": int(count),
                }
            )
    profiles = pd.DataFrame.from_records(profile_records)
    wide = _wide_profiles(profiles) if not profiles.empty else pd.DataFrame()
    return wide, pd.DataFrame.from_records(event_records)


def _target_inputs(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        "n_samples",
    ]
    points = pd.read_parquet(
        path,
        columns=columns,
        filters=[("source_name", "==", TARGET_NAME)],
    )
    points = points[np.isfinite(pd.to_numeric(points["normalized_power"], errors="coerce"))]
    folded = fold_occultogram_points(points)
    raw_wide = _wide_profiles(_block_profiles(folded))

    visual = folded[
        folded["antenna"].astype(str).eq("rv1_coarse")
        & np.isclose(pd.to_numeric(folded["control_shift_s"], errors="coerce"), 0.0)
        & (np.abs(pd.to_numeric(folded["t_bin_sec"], errors="coerce")) <= PROFILE_DISPLAY_HALF_WINDOW_SECONDS)
    ].copy()
    visual = visual[
        visual["frequency_mhz"].apply(
            lambda value: any(np.isclose(float(value), frequency) for frequency in DETECTION_FREQUENCIES)
        )
    ]
    parts = visual["event_uid"].astype(str).str.split("|", expand=True, regex=False)
    visual["physical_event_uid"] = parts[0] + "|" + parts[1] + "|" + parts[2]
    keep = [
        "source_name",
        "physical_event_uid",
        "event_order",
        "lunar_block",
        "event_type",
        "frequency_mhz",
        "antenna",
        "control_shift_s",
        "t_bin_sec",
        "normalized_power",
        "n_samples",
    ]
    return raw_wide, visual.loc[:, keep].reset_index(drop=True)


def _candidate_lookup() -> dict[tuple[str, str, str, float], dict[str, object]]:
    return {
        (
            str(item["antenna"]),
            str(item["data_kind"]),
            str(item["template_kind"]),
            float(item["edge_width_s"]),
        ): item
        for item in FROZEN_CANDIDATES
    }


def _run_frozen_fits(
    raw_wide: pd.DataFrame,
    residual_wide: pd.DataFrame,
    frequencies: tuple[float, ...],
    selected_frequencies: tuple[float, ...],
    times: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    blocks = sorted(int(value) for value in pd.unique(raw_wide["lunar_block"]))
    if len(blocks) < 5:
        raise RuntimeError(f"at least five lunar blocks are required; found {len(blocks)}")
    selected_indices = [
        next(index for index, value in enumerate(frequencies) if np.isclose(value, requested))
        for requested in selected_frequencies
    ]
    model_cache: dict[tuple[int, str, str], object] = {}
    fit_records: list[dict[str, object]] = []
    template_records: list[dict[str, object]] = []

    for candidate in FROZEN_CANDIDATES:
        antenna = str(candidate["antenna"])
        data_kind = str(candidate["data_kind"])
        template_kind = str(candidate["template_kind"])
        width = float(candidate["edge_width_s"])
        data_wide = raw_wide if data_kind == "raw" else residual_wide
        for block in blocks:
            cache_key = (block, antenna, data_kind)
            if cache_key not in model_cache:
                training_blocks = [value for value in blocks if value != block]
                model_cache[cache_key] = _estimate_covariance(
                    data_wide, training_blocks, antenna, frequencies, times
                )
            model = model_cache[cache_key]
            templates = _control_templates(model, frequencies, selected_frequencies)
            spectral_template = templates[template_kind]
            covariance = _selected_covariance(model, frequencies, selected_frequencies)
            for frequency, scale in zip(selected_frequencies, spectral_template):
                template_records.append(
                    {
                        "candidate_id": str(candidate["candidate_id"]),
                        "held_out_lunar_block": int(block),
                        "antenna": antenna,
                        "data_kind": data_kind,
                        "template_kind": template_kind,
                        "frequency_mhz": float(frequency),
                        "template_edge_scale": float(scale),
                    }
                )
            for shift in (0.0,) + NULL_CONTROL_SHIFTS:
                matrix = _profile_matrix(
                    data_wide, block, antenna, shift, frequencies, times
                )[:, selected_indices]
                estimate, standard_error, score = _joint_fit(
                    matrix, times, covariance, spectral_template, width
                )
                fit_records.append(
                    {
                        "candidate_id": str(candidate["candidate_id"]),
                        "candidate_role": str(candidate["role"]),
                        "held_out_lunar_block": int(block),
                        "antenna": antenna,
                        "data_kind": data_kind,
                        "control_shift_s": float(shift),
                        "template_kind": template_kind,
                        "edge_width_s": width,
                        "estimate_control_scale_units": float(estimate),
                        "standard_error": float(standard_error),
                        "standardized_matched_score": float(score),
                    }
                )
    return pd.DataFrame.from_records(fit_records), pd.DataFrame.from_records(template_records)


def _annotate_candidate_summary(summary: pd.DataFrame) -> pd.DataFrame:
    lookup = _candidate_lookup()
    out = summary.copy()
    out["candidate_id"] = [
        lookup[(str(row.antenna), str(row.data_kind), str(row.template_kind), float(row.edge_width_s))][
            "candidate_id"
        ]
        for row in out.itertuples(index=False)
    ]
    out["candidate_role"] = [
        lookup[(str(row.antenna), str(row.data_kind), str(row.template_kind), float(row.edge_width_s))][
            "role"
        ]
        for row in out.itertuples(index=False)
    ]
    return out


def _long_block_profiles(raw_wide: pd.DataFrame, source_name: str) -> pd.DataFrame:
    frequencies = _available_frequencies(raw_wide)
    out = raw_wide.melt(
        id_vars=["lunar_block", "antenna", "control_shift_s", "t_bin_sec"],
        value_vars=list(frequencies),
        var_name="frequency_mhz",
        value_name="value",
    )
    out["frequency_mhz"] = pd.to_numeric(out["frequency_mhz"], errors="coerce")
    out.insert(0, "source_name", source_name)
    return out[np.isfinite(pd.to_numeric(out["value"], errors="coerce"))].reset_index(drop=True)


def _spatial_comparison(scores: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
    annotated = scores.merge(
        positions[
            [
                "source_name",
                "source_role",
                "ring_radius_deg",
                "east_offset_deg",
                "north_offset_deg",
            ]
        ],
        on="source_name",
        how="left",
        validate="many_to_one",
    )
    rows: list[dict[str, object]] = []
    for candidate_id, group in annotated.groupby("candidate_id", observed=True, sort=False):
        target = group[group["source_role"].eq("target")]
        blanks = group[group["source_role"].eq("blank")].copy()
        if len(target) != 1 or blanks.empty:
            continue
        target_row = target.iloc[0]
        target_score = float(target_row["target_median_standardized_score"])
        blank_scores = pd.to_numeric(
            blanks["target_median_standardized_score"], errors="coerce"
        ).dropna().to_numpy(float)
        exceed = int(np.sum(blank_scores >= target_score))
        center = float(np.median(blank_scores))
        scale = float(1.4826 * np.median(np.abs(blank_scores - center)))
        rows.append(
            {
                "candidate_id": candidate_id,
                "candidate_role": str(target_row["candidate_role"]),
                "antenna": str(target_row["antenna"]),
                "data_kind": str(target_row["data_kind"]),
                "template_kind": str(target_row["template_kind"]),
                "edge_width_s": float(target_row["edge_width_s"]),
                "fornax_score": target_score,
                "blank_median_score": center,
                "blank_q05_score": float(np.quantile(blank_scores, 0.05)),
                "blank_q95_score": float(np.quantile(blank_scores, 0.95)),
                "blank_max_score": float(np.max(blank_scores)),
                "fornax_rank_descending": int(1 + exceed),
                "n_blank_positions": int(len(blank_scores)),
                "blank_positions_at_least_as_large": exceed,
                "one_sided_spatial_empirical_p": float((1 + exceed) / (1 + len(blank_scores))),
                "descriptive_blank_robust_z": (
                    float((target_score - center) / scale)
                    if np.isfinite(scale) and scale > 1e-12
                    else np.nan
                ),
            }
        )
    return pd.DataFrame.from_records(rows)


def _select_visual_blanks(scores: pd.DataFrame) -> tuple[str, str]:
    primary = scores[
        scores["candidate_id"].astype(str).eq("primary_rv1_raw_ramp240")
        & ~scores["source_name"].astype(str).eq(TARGET_NAME)
    ].sort_values("target_median_standardized_score")
    if primary.empty:
        raise RuntimeError("no primary blank-sky scores were available")
    strongest = str(primary.iloc[-1]["source_name"])
    median_score = float(primary["target_median_standardized_score"].median())
    typical = str(
        primary.iloc[
            np.argmin(
                np.abs(
                    primary["target_median_standardized_score"].to_numpy(float)
                    - median_score
                )
            )
        ]["source_name"]
    )
    return strongest, typical


def _candidate_label(candidate_id: str) -> str:
    return {
        "primary_rv1_raw_ramp240": "RV1 raw, 240 s ramp",
        "sharp_rv1_raw_step": "RV1 raw, sharp step",
        "rv1_lowrank_ramp60": "RV1 low-rank, 60 s ramp",
        "rv2_raw_ramp240": "RV2 raw, 240 s ramp",
        "rv2_lowrank_ramp120": "RV2 low-rank, 120 s ramp",
    }.get(candidate_id, candidate_id)


def _plot_spatial_scores(
    scores: pd.DataFrame,
    positions: pd.DataFrame,
    comparison: pd.DataFrame,
    output: Path,
) -> None:
    primary = scores[scores["candidate_id"].eq("primary_rv1_raw_ramp240")].merge(
        positions, on="source_name", how="left", validate="one_to_one"
    )
    target = primary[primary["source_role"].eq("target")].iloc[0]
    blanks = primary[primary["source_role"].eq("blank")].copy()
    values = primary["target_median_standardized_score"].to_numpy(float)
    limit = max(1.0, float(np.nanmax(np.abs(values))))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True)

    scatter = axes[0].scatter(
        blanks["east_offset_deg"],
        blanks["north_offset_deg"],
        c=blanks["target_median_standardized_score"],
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        s=105,
        edgecolor="0.2",
        linewidth=0.6,
    )
    axes[0].scatter(
        [0.0],
        [0.0],
        c=[float(target["target_median_standardized_score"])],
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        marker="*",
        s=360,
        edgecolor="black",
        linewidth=1.0,
        label="Fornax A",
        zorder=5,
    )
    for radius in RING_RADII_DEG:
        circle = plt.Circle((0, 0), radius, fill=False, color="0.65", linestyle="--", linewidth=0.8)
        axes[0].add_patch(circle)
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("east offset from Fornax A (deg)")
    axes[0].set_ylabel("north offset from Fornax A (deg)")
    axes[0].set_title("Frozen primary score on nearby sky")
    axes[0].legend(frameon=False, loc="upper right")
    colorbar = fig.colorbar(scatter, ax=axes[0], pad=0.02)
    colorbar.set_label("median block matched score")

    ordered = blanks.sort_values("target_median_standardized_score").reset_index(drop=True)
    axes[1].scatter(
        np.arange(len(ordered)),
        ordered["target_median_standardized_score"],
        c=ordered["ring_radius_deg"],
        cmap="viridis",
        s=42,
        label="blank positions",
    )
    axes[1].axhline(
        float(target["target_median_standardized_score"]),
        color="#b2182b",
        linewidth=2.0,
        label="Fornax A",
    )
    axes[1].axhline(0.0, color="0.65", linewidth=0.8)
    row = comparison[comparison["candidate_id"].eq("primary_rv1_raw_ramp240")].iloc[0]
    axes[1].set_title(
        f"Fornax rank {int(row['fornax_rank_descending'])}/{int(row['n_blank_positions']) + 1}; "
        f"spatial p={float(row['one_sided_spatial_empirical_p']):.3f}"
    )
    axes[1].set_xlabel("blank positions, sorted by score")
    axes[1].set_ylabel("median block matched score")
    axes[1].legend(frameon=False)
    fig.suptitle("Fornax A versus matched nearby-sky occultations", fontsize=13)
    fig.savefig(output, dpi=190)
    plt.close(fig)


def _plot_all_candidate_scores(scores: pd.DataFrame, output: Path) -> None:
    candidate_order = [str(item["candidate_id"]) for item in FROZEN_CANDIDATES]
    fig, axes = plt.subplots(1, len(candidate_order), figsize=(15.5, 4.7), sharey=True, constrained_layout=True)
    for axis, candidate_id in zip(axes, candidate_order):
        group = scores[scores["candidate_id"].eq(candidate_id)].copy()
        target = group[group["source_name"].eq(TARGET_NAME)].iloc[0]
        blanks = group[~group["source_name"].eq(TARGET_NAME)].sort_values(
            "target_median_standardized_score"
        )
        axis.scatter(
            np.zeros(len(blanks)),
            blanks["target_median_standardized_score"],
            color="0.55",
            alpha=0.8,
            s=25,
            label="blanks" if candidate_id == candidate_order[0] else None,
        )
        axis.scatter(
            [0.0],
            [float(target["target_median_standardized_score"])],
            marker="*",
            color="#b2182b",
            edgecolor="black",
            linewidth=0.7,
            s=190,
            zorder=4,
            label="Fornax A" if candidate_id == candidate_order[0] else None,
        )
        axis.axhline(0.0, color="0.8", linewidth=0.8)
        axis.set_xlim(-0.35, 0.35)
        axis.set_xticks([])
        axis.set_title(_candidate_label(candidate_id), fontsize=9)
    axes[0].set_ylabel("median block matched score")
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle("Every branch uses its target-frozen template and width", fontsize=12)
    fig.savefig(output, dpi=190)
    plt.close(fig)


def _profile_summary(
    raw_profiles: pd.DataFrame,
    lowrank_profiles: pd.DataFrame,
) -> pd.DataFrame:
    raw = raw_profiles.rename(columns={"value": "profile_value"}).copy()
    raw["data_kind"] = "raw"
    lowrank = lowrank_profiles.rename(columns={"residual_value": "profile_value"}).copy()
    lowrank["data_kind"] = "lowrank"
    keep = [
        "source_name",
        "data_kind",
        "lunar_block",
        "antenna",
        "control_shift_s",
        "t_bin_sec",
        "frequency_mhz",
        "profile_value",
    ]
    combined = pd.concat([raw.loc[:, keep], lowrank.loc[:, keep]], ignore_index=True)
    combined = combined[
        combined["antenna"].astype(str).eq("rv1_coarse")
        & np.isclose(pd.to_numeric(combined["control_shift_s"], errors="coerce"), 0.0)
        & (np.abs(pd.to_numeric(combined["t_bin_sec"], errors="coerce")) <= PROFILE_DISPLAY_HALF_WINDOW_SECONDS)
    ]
    return (
        combined.groupby(
            ["source_name", "data_kind", "frequency_mhz", "t_bin_sec"],
            observed=True,
            sort=True,
        )["profile_value"]
        .agg(median="median", q16=lambda values: values.quantile(0.16), q84=lambda values: values.quantile(0.84))
        .reset_index()
    )


def _plot_profile_comparison(
    summary: pd.DataFrame,
    strongest_blank: str,
    typical_blank: str,
    output: Path,
) -> None:
    fig, axes = plt.subplots(
        len(DETECTION_FREQUENCIES),
        2,
        figsize=(13.5, 12.5),
        sharex=True,
        constrained_layout=True,
    )
    colors = {TARGET_NAME: "black", strongest_blank: "#d95f02", typical_blank: "#1b9e77"}
    labels = {TARGET_NAME: "Fornax A", strongest_blank: "strongest blank", typical_blank: "typical blank"}
    blank_names = [
        name for name in summary["source_name"].astype(str).unique() if name != TARGET_NAME
    ]
    for row_index, frequency in enumerate(DETECTION_FREQUENCIES):
        for column_index, data_kind in enumerate(("raw", "lowrank")):
            axis = axes[row_index, column_index]
            channel = summary[
                summary["data_kind"].eq(data_kind)
                & np.isclose(summary["frequency_mhz"], frequency)
            ]
            blank = channel[channel["source_name"].isin(blank_names)]
            spatial = (
                blank.groupby("t_bin_sec", observed=True)["median"]
                .agg(blank_median="median", blank_q16=lambda values: values.quantile(0.16), blank_q84=lambda values: values.quantile(0.84))
                .reset_index()
            )
            axis.fill_between(
                spatial["t_bin_sec"].to_numpy(float),
                spatial["blank_q16"].to_numpy(float),
                spatial["blank_q84"].to_numpy(float),
                color="0.80",
                alpha=0.45,
                linewidth=0,
                label="blank-sky 16–84%" if row_index == 0 and column_index == 0 else None,
            )
            for name in (TARGET_NAME, strongest_blank, typical_blank):
                line = channel[channel["source_name"].eq(name)].sort_values("t_bin_sec")
                axis.plot(
                    line["t_bin_sec"],
                    line["median"],
                    color=colors[name],
                    linewidth=1.8 if name == TARGET_NAME else 1.0,
                    alpha=1.0 if name == TARGET_NAME else 0.85,
                    label=labels[name] if row_index == 0 and column_index == 0 else None,
                )
            axis.axvline(0.0, color="black", linestyle="--", linewidth=0.8)
            axis.axhline(0.0, color="0.75", linewidth=0.7)
            if row_index == 0:
                axis.set_title("raw normalized profiles" if data_kind == "raw" else "cross-fitted low-rank residuals")
            axis.set_ylabel(f"{frequency:.2f} MHz")
            if row_index == len(DETECTION_FREQUENCIES) - 1:
                axis.set_xlabel("seconds from predicted occultation")
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle("RV1 lunar-block medians: Fornax A versus nearby blank-sky positions", fontsize=13)
    fig.savefig(output, dpi=190)
    plt.close(fig)


def _heatmap_matrix(points: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    pivot = points.pivot_table(
        index="physical_event_uid",
        columns="t_bin_sec",
        values="normalized_power",
        aggfunc="median",
    )
    pivot = pivot.sort_index()
    return pivot.to_numpy(float), pivot.columns.to_numpy(float)


def _plot_event_occultograms(
    event_profiles: pd.DataFrame,
    strongest_blank: str,
    typical_blank: str,
    output: Path,
) -> None:
    names = (TARGET_NAME, strongest_blank, typical_blank)
    labels = ("Fornax A", "strongest blank by frozen score", "typical blank")
    values = pd.to_numeric(event_profiles["normalized_power"], errors="coerce").dropna()
    limit = min(5.0, max(1.0, float(np.quantile(np.abs(values), 0.98))))
    fig, axes = plt.subplots(
        len(names),
        len(DETECTION_FREQUENCIES),
        figsize=(15.5, 9.0),
        squeeze=False,
        constrained_layout=True,
    )
    image = None
    for row_index, (name, label) in enumerate(zip(names, labels)):
        for column_index, frequency in enumerate(DETECTION_FREQUENCIES):
            axis = axes[row_index, column_index]
            channel = event_profiles[
                event_profiles["source_name"].astype(str).eq(name)
                & np.isclose(event_profiles["frequency_mhz"], frequency)
            ]
            matrix, times = _heatmap_matrix(channel)
            image = axis.imshow(
                matrix,
                aspect="auto",
                interpolation="nearest",
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
                extent=[times.min(), times.max(), matrix.shape[0] - 0.5, -0.5],
            )
            axis.axvline(0.0, color="black", linestyle="--", linewidth=0.8)
            if row_index == 0:
                axis.set_title(f"{frequency:.2f} MHz")
            if column_index == 0:
                axis.set_ylabel(f"{label}\nfolded events")
            else:
                axis.set_yticks([])
            if row_index == len(names) - 1:
                axis.set_xlabel("seconds from predicted time")
            else:
                axis.tick_params(labelbottom=False)
    if image is not None:
        colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), pad=0.008, aspect=40)
        colorbar.set_label("normalized raw power")
    fig.suptitle(
        "RV1 event-level occultograms — identical extraction and fixed t=0 alignment",
        fontsize=13,
    )
    fig.savefig(output, dpi=190)
    plt.close(fig)


def _reference_validation(scores: pd.DataFrame, reference_path: Path) -> pd.DataFrame:
    reference = pd.read_parquet(reference_path)
    target = scores[scores["source_name"].eq(TARGET_NAME)].copy()
    keys = ["antenna", "data_kind", "template_kind", "edge_width_s"]
    keep = keys + ["target_median_standardized_score"]
    merged = target.merge(
        reference.loc[:, keep],
        on=keys,
        how="left",
        suffixes=("_blanksky_run", "_reference_v2"),
        validate="one_to_one",
    )
    merged["absolute_score_difference"] = np.abs(
        merged["target_median_standardized_score_blanksky_run"]
        - merged["target_median_standardized_score_reference_v2"]
    )
    return merged


def _write_report(
    out_dir: Path,
    comparison: pd.DataFrame,
    scores: pd.DataFrame,
    strongest_blank: str,
    typical_blank: str,
    validation: pd.DataFrame,
) -> None:
    primary = comparison[comparison["candidate_id"].eq("primary_rv1_raw_ramp240")].iloc[0]
    primary_scores = scores[scores["candidate_id"].eq("primary_rv1_raw_ramp240")]
    target_temporal = primary_scores[primary_scores["source_name"].eq(TARGET_NAME)].iloc[0]
    strongest = primary_scores[primary_scores["source_name"].eq(strongest_blank)].iloc[0]
    lines = [
        "# Fornax A Matched Blank-Sky Occultation Comparison",
        "",
        "## Outcome",
        "",
        (
            f"The frozen primary Fornax statistic ranks **{int(primary['fornax_rank_descending'])} of "
            f"{int(primary['n_blank_positions']) + 1}** positions (target plus 24 nearby controls). "
            f"Its one-sided spatial empirical p-value is **{float(primary['one_sided_spatial_empirical_p']):.4f}**."
        ),
        "",
        (
            f"Fornax has median block score {float(primary['fornax_score']):.4f}; the blank-sky median is "
            f"{float(primary['blank_median_score']):.4f}, the 95th percentile is "
            f"{float(primary['blank_q95_score']):.4f}, and the maximum is "
            f"{float(primary['blank_max_score']):.4f}."
        ),
        "",
        (
            f"The strongest blank is `{strongest_blank}` with temporal-control p="
            f"{float(strongest['one_sided_empirical_p']):.6g}; the representative median blank is `{typical_blank}`."
        ),
        "",
        "## Visual interpretation",
        "",
        "The raw statistic is spatially localized, but this is not an Earth-like visual detection. The Fornax block-median profiles largely overlap the blank-sky envelope, and the event heatmaps are dominated by horizontal event/block structure instead of a clean vertical edge at zero.",
        "",
        "The foreground-subtracted branches are also less distinctive: RV1 low-rank ranks fifth of 25 positions (`p = 0.20`), while RV2 low-rank ranks second (`p = 0.08`). Keep the result classified as a localized raw candidate, not a Fornax A detection.",
        "",
        "## What was frozen",
        "",
        "- Primary: RV1 raw, control-scaled spectral template, 240-second fixed-centered ramp.",
        "- Edge center: exactly 0 seconds for the target and every spatial control.",
        "- Detection bands: 0.45, 3.93, 4.70, 6.55, and 9.18 MHz.",
        "- Covariance and spectral scales: trained only on other lunar blocks' off-time controls.",
        "- Antennas: evaluated separately; cross-antenna consistency is not a detection requirement.",
        "- The blank positions were not allowed to select their own width, template, or antenna.",
        "",
        "## Spatial controls",
        "",
        "Twelve positions lie on a 2-degree ring and twelve on a 5-degree ring around the ICRS Fornax A coordinate. "
        "This is a local empirical sky null: it preserves much of the foreground and observing geometry while moving the lunar limb away from Fornax A. "
        "The labels mean control fields, not a guarantee that no faint catalog source exists there.",
        "",
        "## Temporal versus spatial evidence",
        "",
        (
            f"The target's exact off-time temporal-control p-value remains {float(target_temporal['one_sided_empirical_p']):.6g}. "
            "That tests whether t=0 is unusual relative to ±10/±20 minutes for the same sky position. "
            "The spatial rank above tests the more relevant localization question: whether nearby sky positions produce a comparable fixed-zero feature."
        ),
        "",
        "## Reproducibility check",
        "",
        (
            f"Maximum absolute difference between the target scores recomputed here and the finalized v2 Fornax table: "
            f"{float(validation['absolute_score_difference'].max()):.3g}."
        ),
        "",
        "## Files to inspect",
        "",
        "- `fornax_blanksky_spatial_scores.png`: local sky map and target rank.",
        "- `fornax_blanksky_all_frozen_candidates.png`: target versus blanks for every frozen branch.",
        "- `fornax_blanksky_block_profile_comparison.png`: raw and low-rank lunar-block median morphology.",
        "- `fornax_blanksky_selected_event_occultograms.png`: event-level Fornax, strongest-blank, and typical-blank occultograms.",
        "- `spatial_empirical_comparison.parquet`: compact spatial inference table.",
        "- `frozen_candidate_scores.parquet`: position-level temporal-control results.",
        "",
        "No CSV or NumPy array products are written.",
    ]
    (out_dir / "FORNAX_BLANKSKY_REVIEW.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _event_summary(events: pd.DataFrame, source_name: str) -> pd.DataFrame:
    physical = events.drop_duplicates(["predicted_event_time", "event_type"]).copy()
    physical["lunar_block"] = physical["predicted_event_time"].apply(lunar_block_id)
    return (
        physical.groupby(["lunar_block", "event_type"], observed=True, sort=True)
        .agg(
            physical_events=("predicted_event_time", "size"),
            first_event=("predicted_event_time", "min"),
            last_event=("predicted_event_time", "max"),
            median_prediction_gap_s=("gap_seconds", "median"),
        )
        .reset_index()
        .assign(source_name=source_name)
        .loc[
            :,
            [
                "source_name",
                "lunar_block",
                "event_type",
                "physical_events",
                "first_event",
                "last_event",
                "median_prediction_gap_s",
            ],
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", type=Path, default=DEFAULT_CLEAN)
    parser.add_argument("--target-profiles", type=Path, default=DEFAULT_TARGET_PROFILES)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--max-blank-positions",
        type=int,
        default=None,
        help="Debugging only; the scientific comparison requires all 24 controls.",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    config = OccultogramConfig(
        window_seconds=2400.0,
        bin_seconds=60.0,
        sideband_start_seconds=1200.0,
        control_shifts_seconds=(-1200.0, -600.0, 0.0, 600.0, 1200.0),
        minimum_window_samples=8,
        minimum_sideband_samples=6,
        validity_mode="finite",
    )
    positions = _position_table()
    if args.max_blank_positions is not None:
        blanks = positions[positions["source_role"].eq("blank")].head(
            int(args.max_blank_positions)
        )
        positions = pd.concat(
            [positions[positions["source_role"].eq("target")], blanks], ignore_index=True
        )
    write_table(positions, args.out_dir / "control_positions.parquet", index=False)

    print(f"reading compact clean table: {args.clean}", flush=True)
    clean = _read_clean(args.clean)
    geometry = (
        clean.loc[:, ["time", "position_x", "position_y", "position_z"]]
        .drop_duplicates("time")
        .sort_values("time")
        .reset_index(drop=True)
    )
    channels = _channel_arrays(clean)

    target_raw, target_visual = _target_inputs(args.target_profiles)
    frequencies = _available_frequencies(target_raw)
    selected_frequencies = tuple(
        _frequency_column(list(frequencies), requested) for requested in DETECTION_FREQUENCIES
    )
    times = np.asarray(
        sorted(
            float(value)
            for value in pd.unique(target_raw["t_bin_sec"])
            if abs(float(value)) <= FIT_HALF_WINDOW_SECONDS
        ),
        dtype=float,
    )

    all_raw: list[pd.DataFrame] = []
    all_lowrank: list[pd.DataFrame] = []
    all_fits: list[pd.DataFrame] = []
    all_scores: list[pd.DataFrame] = []
    all_templates: list[pd.DataFrame] = []
    event_summaries: list[pd.DataFrame] = []
    event_cache: dict[str, pd.DataFrame] = {}

    position_rows = list(positions.itertuples(index=False))
    for index, position in enumerate(position_rows, start=1):
        source_name = str(position.source_name)
        print(f"[{index}/{len(position_rows)}] {source_name}", flush=True)
        if source_name == TARGET_NAME:
            raw_wide = target_raw.copy()
        else:
            source = pd.Series(
                {
                    "source_name": source_name,
                    "kind": "fixed",
                    "body_name": "",
                    "ra_deg": float(position.ra_deg),
                    "dec_deg": float(position.dec_deg),
                    "frame": "icrs",
                }
            )
            events, _ = predict_source_events(
                geometry,
                source,
                target_frame="fk4",
                equinox="B1950",
                ephemeris="builtin",
                max_gap_seconds=600.0,
                prediction_cadence_seconds=300.0,
                frequencies=list(FREQUENCY_BANDS),
                antennas=list(ANTENNAS),
            )
            if events.empty:
                raise RuntimeError(f"no events predicted for {source_name}")
            event_summaries.append(_event_summary(events, source_name))
            event_cache[source_name] = events
            raw_wide, _ = _extract_position(source_name, events, channels, config)
            print(f"  extracted {len(raw_wide):,} compact block-profile rows", flush=True)
        source_frequencies = _available_frequencies(raw_wide)
        if source_frequencies != frequencies:
            raise RuntimeError(
                f"frequency mismatch for {source_name}: {source_frequencies} != {frequencies}"
            )
        lowrank, rank_selection, _ = _crossfit_lowrank(raw_wide, frequencies)
        print("  completed cross-fitted low-rank foreground model", flush=True)
        residual_wide = _residual_wide(lowrank)
        fits, templates = _run_frozen_fits(
            raw_wide, residual_wide, frequencies, selected_frequencies, times
        )
        scores, _, _ = _candidate_summaries(fits)
        scores = _annotate_candidate_summary(scores)
        primary_score = float(
            scores.loc[
                scores["candidate_id"].eq("primary_rv1_raw_ramp240"),
                "target_median_standardized_score",
            ].iloc[0]
        )
        print(f"  frozen primary score={primary_score:.4f}", flush=True)

        raw_long = _long_block_profiles(raw_wide, source_name)
        raw_long = raw_long[
            np.abs(pd.to_numeric(raw_long["t_bin_sec"], errors="coerce"))
            <= PROFILE_DISPLAY_HALF_WINDOW_SECONDS
        ]
        lowrank = lowrank.copy()
        lowrank.insert(0, "source_name", source_name)
        lowrank = lowrank[
            np.abs(pd.to_numeric(lowrank["t_bin_sec"], errors="coerce"))
            <= PROFILE_DISPLAY_HALF_WINDOW_SECONDS
        ]
        fits.insert(0, "source_name", source_name)
        scores.insert(0, "source_name", source_name)
        templates.insert(0, "source_name", source_name)
        rank_selection = rank_selection.copy()
        rank_selection.insert(0, "source_name", source_name)
        all_raw.append(raw_long)
        all_lowrank.append(lowrank)
        all_fits.append(fits)
        all_scores.append(scores)
        all_templates.append(templates)

    raw_profiles = pd.concat(all_raw, ignore_index=True)
    lowrank_profiles = pd.concat(all_lowrank, ignore_index=True)
    fits = pd.concat(all_fits, ignore_index=True)
    scores = pd.concat(all_scores, ignore_index=True)
    templates = pd.concat(all_templates, ignore_index=True)
    comparison = _spatial_comparison(scores, positions)
    validation = _reference_validation(scores, args.reference)
    if float(validation["absolute_score_difference"].max()) > 1e-8:
        raise RuntimeError("target recomputation does not match the finalized v2 score table")

    strongest_blank, typical_blank = _select_visual_blanks(scores)
    selected_visuals = [target_visual]
    for name in (strongest_blank, typical_blank):
        _, visual = _extract_position(
            name,
            event_cache[name],
            channels,
            config,
            capture_event_profiles=True,
            build_block_profiles=False,
        )
        visual = visual[
            np.abs(pd.to_numeric(visual["t_bin_sec"], errors="coerce"))
            <= PROFILE_DISPLAY_HALF_WINDOW_SECONDS
        ]
        selected_visuals.append(visual)
    event_profiles = pd.concat(selected_visuals, ignore_index=True)

    write_table(
        pd.concat(event_summaries, ignore_index=True),
        args.out_dir / "predicted_event_summary.parquet",
        index=False,
    )
    write_table(raw_profiles, args.out_dir / "raw_block_profiles.parquet", index=False)
    write_table(
        lowrank_profiles,
        args.out_dir / "lowrank_crossfit_block_profiles.parquet",
        index=False,
    )
    write_table(fits, args.out_dir / "frozen_candidate_block_fits.parquet", index=False)
    write_table(scores, args.out_dir / "frozen_candidate_scores.parquet", index=False)
    write_table(templates, args.out_dir / "control_only_spectral_templates.parquet", index=False)
    write_table(comparison, args.out_dir / "spatial_empirical_comparison.parquet", index=False)
    write_table(validation, args.out_dir / "target_reference_validation.parquet", index=False)
    write_table(
        event_profiles,
        args.out_dir / "selected_event_occultogram_profiles.parquet",
        index=False,
    )

    spatial_plot = args.out_dir / "fornax_blanksky_spatial_scores.png"
    candidates_plot = args.out_dir / "fornax_blanksky_all_frozen_candidates.png"
    profiles_plot = args.out_dir / "fornax_blanksky_block_profile_comparison.png"
    occultogram_plot = args.out_dir / "fornax_blanksky_selected_event_occultograms.png"
    _plot_spatial_scores(scores, positions, comparison, spatial_plot)
    _plot_all_candidate_scores(scores, candidates_plot)
    profile_summary = _profile_summary(raw_profiles, lowrank_profiles)
    _plot_profile_comparison(
        profile_summary, strongest_blank, typical_blank, profiles_plot
    )
    _plot_event_occultograms(
        event_profiles, strongest_blank, typical_blank, occultogram_plot
    )
    _write_report(
        args.out_dir,
        comparison,
        scores,
        strongest_blank,
        typical_blank,
        validation,
    )

    manifest = {
        "output_directory": str(args.out_dir),
        "clean_input": str(args.clean),
        "target_profile_input": str(args.target_profiles),
        "reference_target_scores": str(args.reference),
        "target": TARGET_NAME,
        "blank_positions": int(positions["source_role"].eq("blank").sum()),
        "ring_radii_deg": RING_RADII_DEG,
        "positions_per_ring": POSITIONS_PER_RING,
        "edge_time_fixed_seconds": 0.0,
        "occultogram_config": asdict(config),
        "detection_frequencies_mhz": selected_frequencies,
        "frozen_candidates": FROZEN_CANDIDATES,
        "antennas_fit_separately": True,
        "antenna_consistency_required": False,
        "strongest_blank": strongest_blank,
        "typical_blank": typical_blank,
        "software_versions": software_versions(),
        "generated_plots": [
            spatial_plot.name,
            candidates_plot.name,
            profiles_plot.name,
            occultogram_plot.name,
        ],
        "output_formats": ["parquet_zstd", "png", "json", "markdown"],
        "csv_or_npy_outputs": False,
        "claim_status": "localized raw candidate; visual and low-rank controls do not support a Fornax A detection claim",
    }
    (args.out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
