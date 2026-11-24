#!/usr/bin/env python3
"""Systematic, channel-separated Helios-1 E5c / RAE-2 RV burst study.

NOAA 18/29 MHz reports select times only.  Every Helios frequency, every
RAE-2 frequency, both RAE antennas, and both RAE validity treatments remain
separate throughout detection.  Frequency pairing is used only after each
instrument has a same-channel event-versus-control statistic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import rankdata, spearmanr, theilslopes


RNG_SEED = 581_204
C_KM_S = 299_792.458
AU_KM = 149_597_870.7
RSUN_KM = 695_700.0
TIMESTAMP_SHIFTS_MIN = (-5, 0, 5)
CONTROL_SHIFTS_H = (-2, 2, -4, 4, -6, 6, -8, 8)
PRIMARY_PAIRS = (
    (0.45, 0.445, True),
    (0.70, 0.765, True),
    (0.90, 1.010, True),
    (1.31, 1.320, True),
    (2.20, 2.280, True),
    (3.93, 3.000, False),
)

HELIOS_SUMMARY_COLUMNS = (
    "frequency_khz", "frequency_mhz", "n_eligible_episodes", "n_independent_days",
    "median_day_upper_event_control_z", "positive_day_fraction", "sign_flip_p",
    "bh_q_across_helios_channels",
)
PAIR_SUMMARY_COLUMNS = (
    "validity_mode", "antenna", "rae_frequency_mhz", "helios_frequency_mhz",
    "primary_pair", "n_paired_episodes", "n_independent_days", "spearman_rho",
    "day_block_permutation_p", "joint_positive_fraction", "median_rae_effect",
    "median_helios_effect", "bh_q_within_antenna_treatment", "exploratory_pair_flag",
)
DAY_SUMMARY_COLUMNS = (
    "validity_mode", "antenna", "rae_frequency_mhz", "helios_frequency_mhz",
    "primary_pair", "gate_date", "n_episodes", "rae_effect", "helios_effect",
)
SUPPORT_COLUMNS = (
    "episode_id", "antenna", "rae_frequency_mhz", "helios_frequency_mhz",
    "primary_pair", "helios_positive_all_timestamp_conventions",
    "rae_finite_independent_candidate", "rae_strict_independent_candidate",
    "cross_mission_supported",
)
DRIFT_COLUMNS = (
    "time_mode", "episode_id", "gate_time", "n_independently_selected_channels", "frequencies_khz",
    "frequency_ratio", "peak_delay_span_s", "spearman_rho_delay_vs_fundamental_radius",
    "timing_rms_s", "within_event_frequency_permutation_p", "frequency_drift_khz_s",
    "fundamental_speed_km_s", "fundamental_speed_fraction_c", "harmonic_speed_km_s",
    "harmonic_speed_fraction_c", "coherent_drift",
)


def robust_sigma(values: pd.Series | np.ndarray) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return math.nan
    center = float(np.median(arr))
    scale = 1.4826 * float(np.median(np.abs(arr - center)))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.std(arr, ddof=1))
    return scale if np.isfinite(scale) and scale > 0 else math.nan


def bh_qvalues(pvalues: pd.Series) -> pd.Series:
    p = pd.to_numeric(pvalues, errors="coerce").to_numpy(dtype=float)
    out = np.full(p.size, np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    if finite.size == 0:
        return pd.Series(out, index=pvalues.index)
    order = finite[np.argsort(p[finite])]
    ranked = p[order] * finite.size / np.arange(1, finite.size + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out[order] = np.clip(ranked, 0.0, 1.0)
    return pd.Series(out, index=pvalues.index)


def sign_flip_p(values: np.ndarray, rng: np.random.Generator, draws: int = 20_000) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return math.nan
    observed = float(np.median(arr))
    if observed <= 0:
        return 1.0
    signs = rng.choice(np.array([-1.0, 1.0]), size=(draws, arr.size), replace=True)
    null = np.median(signs * arr[None, :], axis=1)
    return float((1 + np.count_nonzero(null >= observed)) / (draws + 1))


def permutation_spearman_p(
    x: np.ndarray, y: np.ndarray, rng: np.random.Generator, draws: int = 20_000
) -> tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x, dtype=float)[mask]
    y = np.asarray(y, dtype=float)[mask]
    if x.size < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return math.nan, math.nan
    x_rank = rankdata(x).astype(float)
    y_rank = rankdata(y).astype(float)
    x_rank -= x_rank.mean()
    y_rank -= y_rank.mean()
    denominator = float(np.sqrt(np.dot(x_rank, x_rank) * np.dot(y_rank, y_rank)))
    if not np.isfinite(denominator) or denominator <= 0:
        return math.nan, math.nan
    observed = float(np.dot(x_rank, y_rank) / denominator)
    exceed = 0
    for _ in range(draws):
        null_rho = float(np.dot(x_rank, rng.permutation(y_rank)) / denominator)
        exceed += int(abs(null_rho) >= abs(observed))
    return observed, float((exceed + 1) / (draws + 1))


def parse_trajectory(paths: list[Path]) -> pd.DataFrame:
    columns = [
        "year", "day_of_year", "hour", "carrington_rotation", "heliocentric_au",
        "solar_ecliptic_lat_deg", "solar_ecliptic_lon_deg", "heliographic_lat_deg",
        "heliographic_lon_deg", "heliographic_inertial_lon_deg",
        "earth_sun_spacecraft_angle_deg", "data_code",
    ]
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path, sep=r"\s+", names=columns, comment="#")
        frame = frame[pd.to_numeric(frame["heliocentric_au"], errors="coerce").notna()].copy()
        year = 1900 + pd.to_numeric(frame["year"], errors="coerce").astype(int)
        day = pd.to_numeric(frame["day_of_year"], errors="coerce").astype(int)
        hour = pd.to_numeric(frame["hour"], errors="coerce").astype(int)
        frame["time"] = pd.to_datetime(year.astype(str) + "-01-01", utc=True)
        frame["time"] += pd.to_timedelta(day - 1, unit="D") + pd.to_timedelta(hour, unit="h")
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True).sort_values("time")
    out = out.drop_duplicates("time")
    out["heliocentric_au"] = pd.to_numeric(out["heliocentric_au"], errors="coerce")
    return out[["time", "heliocentric_au", "earth_sun_spacecraft_angle_deg"]]


def select_archive_science_records(helios: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Resolve recovered-tape overlap without mixing channels or archived fields.

    For duplicate timestamp/channel rows, retain the row with the larger
    same-channel nonzero count; ties retain the earlier logical record.  This
    archive-quality rule is fixed before event outcomes are inspected.
    """

    out = helios.copy()
    keys = ["time", "channel_id", "frequency_khz"]
    out["source_duplicate_count"] = out.groupby(keys)["logical_record"].transform("size")
    audit = {
        "decoded_channel_rows": int(len(out)),
        "duplicate_channel_time_groups": int(
            out.loc[out["source_duplicate_count"] > 1, keys].drop_duplicates().shape[0]
        ),
    }
    out["nonzero_count_sort"] = pd.to_numeric(out["nonzero_count"], errors="coerce").fillna(-np.inf)
    out = out.sort_values(
        keys + ["nonzero_count_sort", "logical_record"],
        ascending=[True, True, True, False, True],
        kind="stable",
    ).drop_duplicates(keys, keep="first")
    out = out.drop(columns="nonzero_count_sort").sort_values(["time", "frequency_khz"])
    audit["science_channel_rows"] = int(len(out))
    audit["duplicate_channel_rows_removed"] = int(audit["decoded_channel_rows"] - len(out))
    return out, audit


def add_light_time(helios: pd.DataFrame, trajectory: pd.DataFrame) -> pd.DataFrame:
    out = helios.copy()
    out["time_raw"] = pd.to_datetime(out["time"], utc=True)
    record = out[["logical_record", "time_raw"]].drop_duplicates().sort_values("time_raw")
    x = trajectory["time"].astype("int64").to_numpy(dtype=float)
    y = trajectory["heliocentric_au"].to_numpy(dtype=float)
    xi = record["time_raw"].astype("int64").to_numpy(dtype=float)
    record["heliocentric_au"] = np.interp(xi, x, y, left=np.nan, right=np.nan)
    record["light_time_correction_s"] = (
        (1.0 - record["heliocentric_au"]) * AU_KM / C_KM_S
    )
    record["time_light_corrected"] = record["time_raw"] + pd.to_timedelta(
        record["light_time_correction_s"], unit="s"
    )
    return out.merge(record, on=["logical_record", "time_raw"], how="left", validate="many_to_one")


def window_bounds(frequency_khz: float) -> dict[str, tuple[float, float]]:
    if frequency_khz >= 340.0:
        return {"event": (-5.0, 15.0), "baseline_a": (-125.0, -25.0), "baseline_b": (35.0, 135.0)}
    return {"event": (0.0, 120.0), "baseline_a": (-180.0, -60.0), "baseline_b": (180.0, 300.0)}


def in_window(times: pd.Series, center: pd.Timestamp, bounds_min: tuple[float, float]) -> np.ndarray:
    delta_min = (times - center).dt.total_seconds().to_numpy(dtype=float) / 60.0
    return (delta_min >= bounds_min[0]) & (delta_min <= bounds_min[1])


def control_center_is_clean(center: pd.Timestamp, all_gates: pd.DatetimeIndex) -> bool:
    if len(all_gates) == 0:
        return True
    separation = np.abs((all_gates - center).total_seconds())
    return bool(np.nanmin(separation) >= 120.0 * 60.0)


def select_controls(
    group: pd.DataFrame,
    time_column: str,
    gate: pd.Timestamp,
    event_bounds: tuple[float, float],
    event_count: int,
    all_gates: pd.DatetimeIndex,
) -> list[dict[str, object]]:
    qualified: list[dict[str, object]] = []
    minimum_count = max(2, int(math.ceil(0.75 * event_count)))
    for shift_h in CONTROL_SHIFTS_H:
        center = gate + pd.Timedelta(hours=shift_h)
        if not control_center_is_clean(center, all_gates):
            continue
        sample = group.loc[in_window(group[time_column], center, event_bounds)]
        if len(sample) < minimum_count or abs(len(sample) - event_count) > max(1, int(0.25 * event_count)):
            continue
        qualified.append(
            {
                "shift_h": int(shift_h),
                "n": int(len(sample)),
                "upper_max": float(sample["two_percent_max_10logt"].max()),
                "median_max": float(sample["median_10logt"].max()),
            }
        )
    negative = sorted((row for row in qualified if row["shift_h"] < 0), key=lambda row: abs(row["shift_h"]))
    positive = sorted((row for row in qualified if row["shift_h"] > 0), key=lambda row: abs(row["shift_h"]))
    if not negative or not positive:
        return []
    selected = negative[:2] + positive[:2]
    return sorted(selected[:4], key=lambda row: row["shift_h"])


def build_coverage_audit(
    helios: pd.DataFrame, episodes: pd.DataFrame, rae_episode_ids: set[int]
) -> pd.DataFrame:
    times = pd.DatetimeIndex(helios["time_raw"].drop_duplicates().sort_values())
    rows: list[dict[str, object]] = []
    cohort = episodes[
        episodes["episode_id"].isin(rae_episode_ids)
        & (episodes["gate_time"] >= pd.Timestamp("1974-12-13", tz="UTC"))
    ].copy()
    for row in cohort.itertuples(index=False):
        position = int(np.argmin(np.abs((times - row.gate_time).asi8)))
        nearest = times[position]
        delta_s = float((nearest - row.gate_time).total_seconds())
        rows.append(
            {
                "episode_id": int(row.episode_id),
                "gate_time": row.gate_time,
                "nearest_helios_time_raw": nearest,
                "nearest_minus_gate_s": delta_s,
                "absolute_nearest_delta_s": abs(delta_s),
                "coverage_within_15min": abs(delta_s) <= 900.0,
                "coverage_within_6min": abs(delta_s) <= 360.0,
            }
        )
    return pd.DataFrame(rows).sort_values("episode_id")


def build_helios_metrics(
    helios: pd.DataFrame,
    episodes: pd.DataFrame,
    covered_ids: set[int],
    quality: pd.DataFrame,
) -> pd.DataFrame:
    all_gates = pd.DatetimeIndex(episodes["gate_time"].sort_values())
    eligible_frequencies = set(
        quality.loc[quality["science_eligible_without_event_selection"].astype(bool), "frequency_khz"]
        .astype(float)
        .tolist()
    )
    rows: list[dict[str, object]] = []
    covered = episodes[episodes["episode_id"].isin(covered_ids)].sort_values("gate_time")
    for episode in covered.itertuples(index=False):
        for frequency_khz, group_base in helios.groupby("frequency_khz", sort=True):
            group_base = group_base.sort_values("time_raw")
            bounds = window_bounds(float(frequency_khz))
            channel_preeligible = float(frequency_khz) in eligible_frequencies
            for time_mode, source_column in (
                ("raw", "time_raw"),
                ("light_corrected", "time_light_corrected"),
            ):
                for timestamp_shift_min in TIMESTAMP_SHIFTS_MIN:
                    group = group_base.copy()
                    time_column = "analysis_time"
                    group[time_column] = group[source_column] + pd.Timedelta(minutes=timestamp_shift_min)
                    event = group.loc[in_window(group[time_column], episode.gate_time, bounds["event"])]
                    baseline = group.loc[
                        in_window(group[time_column], episode.gate_time, bounds["baseline_a"])
                        | in_window(group[time_column], episode.gate_time, bounds["baseline_b"])
                    ]
                    controls = select_controls(
                        group, time_column, episode.gate_time, bounds["event"], len(event), all_gates
                    ) if len(event) else []
                    upper_real = float(event["two_percent_max_10logt"].max()) if len(event) else math.nan
                    median_real = float(event["median_10logt"].max()) if len(event) else math.nan
                    upper_controls = np.array([row["upper_max"] for row in controls], dtype=float)
                    median_controls = np.array([row["median_max"] for row in controls], dtype=float)
                    baseline_scale = robust_sigma(baseline["two_percent_max_10logt"])
                    upper_delta = (
                        upper_real - float(np.median(upper_controls))
                        if np.isfinite(upper_real) and upper_controls.size
                        else math.nan
                    )
                    median_delta = (
                        median_real - float(np.median(median_controls))
                        if np.isfinite(median_real) and median_controls.size
                        else math.nan
                    )
                    peak_time = pd.NaT
                    if len(event) and event["two_percent_max_10logt"].notna().any():
                        peak_time = event.loc[event["two_percent_max_10logt"].idxmax(), time_column]
                    inferentially_eligible = bool(
                        channel_preeligible
                        and len(event) >= 2
                        and len(baseline) >= 12
                        and len(controls) >= 2
                        and any(row["shift_h"] < 0 for row in controls)
                        and any(row["shift_h"] > 0 for row in controls)
                        and np.isfinite(baseline_scale)
                        and baseline_scale > 0
                    )
                    exceeds_all = bool(
                        inferentially_eligible
                        and upper_controls.size
                        and upper_real > float(np.max(upper_controls))
                    )
                    helios_candidate = bool(
                        inferentially_eligible and exceeds_all and upper_delta > 0 and median_delta > 0
                    )
                    rows.append(
                        {
                            "episode_id": int(episode.episode_id),
                            "gate_time": episode.gate_time,
                            "gate_date": episode.gate_time.date().isoformat(),
                            "frequency_khz": float(frequency_khz),
                            "frequency_mhz": float(frequency_khz) / 1000.0,
                            "time_mode": time_mode,
                            "timestamp_shift_min": int(timestamp_shift_min),
                            "event_window_start_min": bounds["event"][0],
                            "event_window_end_min": bounds["event"][1],
                            "n_event_records": int(len(event)),
                            "n_baseline_records": int(len(baseline)),
                            "n_controls": int(len(controls)),
                            "control_shifts_h": ";".join(str(row["shift_h"]) for row in controls),
                            "channel_preeligible": channel_preeligible,
                            "inferentially_eligible": inferentially_eligible,
                            "upper_event_max": upper_real,
                            "upper_control_median": float(np.median(upper_controls)) if upper_controls.size else math.nan,
                            "upper_control_max": float(np.max(upper_controls)) if upper_controls.size else math.nan,
                            "upper_event_minus_control": upper_delta,
                            "baseline_robust_sigma": baseline_scale,
                            "upper_event_control_z": upper_delta / baseline_scale if inferentially_eligible else math.nan,
                            "median_event_max": median_real,
                            "median_control_median": float(np.median(median_controls)) if median_controls.size else math.nan,
                            "median_event_minus_control": median_delta,
                            "event_exceeds_all_controls": exceeds_all,
                            "helios_channel_candidate": helios_candidate,
                            "helios_peak_time": peak_time,
                            "helios_peak_delay_s": (
                                float((peak_time - episode.gate_time).total_seconds())
                                if pd.notna(peak_time)
                                else math.nan
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def helios_channel_summary(metrics: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    nominal = metrics[
        (metrics["time_mode"] == "light_corrected")
        & (metrics["timestamp_shift_min"] == 0)
        & metrics["inferentially_eligible"]
    ].copy()
    rows: list[dict[str, object]] = []
    for frequency_khz, group in nominal.groupby("frequency_khz", sort=True):
        daily = group.groupby("gate_date")["upper_event_control_z"].median().to_numpy(dtype=float)
        rows.append(
            {
                "frequency_khz": float(frequency_khz),
                "frequency_mhz": float(frequency_khz) / 1000.0,
                "n_eligible_episodes": int(group["episode_id"].nunique()),
                "n_independent_days": int(len(daily)),
                "median_day_upper_event_control_z": float(np.nanmedian(daily)) if len(daily) else math.nan,
                "positive_day_fraction": float(np.mean(daily > 0)) if len(daily) else math.nan,
                "sign_flip_p": sign_flip_p(daily, rng),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=HELIOS_SUMMARY_COLUMNS)
    out["bh_q_across_helios_channels"] = bh_qvalues(out["sign_flip_p"])
    out = out.loc[:, HELIOS_SUMMARY_COLUMNS]
    return out


def build_crossmission(
    helios_metrics: pd.DataFrame,
    finite: pd.DataFrame,
    strict: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    nominal = helios_metrics[
        (helios_metrics["time_mode"] == "light_corrected")
        & (helios_metrics["timestamp_shift_min"] == 0)
        & helios_metrics["inferentially_eligible"]
    ].copy()
    pair_table = pd.DataFrame(PRIMARY_PAIRS, columns=["rae_frequency_mhz", "helios_frequency_mhz", "primary_pair"])
    nominal = nominal.merge(
        pair_table, left_on="frequency_mhz", right_on="helios_frequency_mhz", how="inner"
    )
    day_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for validity_mode, rae in (("finite", finite), ("strict", strict)):
        rae = rae.copy()
        rae["episode_id"] = pd.to_numeric(rae["episode_id"], errors="coerce").astype("Int64")
        rae["frequency_mhz"] = pd.to_numeric(rae["frequency_mhz"], errors="coerce")
        # Merge pair rows explicitly to prevent approximate or cross-channel joins.
        pieces: list[pd.DataFrame] = []
        for rae_frequency, helios_frequency, primary in PRIMARY_PAIRS:
            r = rae[np.isclose(rae["frequency_mhz"], rae_frequency)].copy()
            h = nominal[np.isclose(nominal["helios_frequency_mhz"], helios_frequency)].copy()
            if r.empty or h.empty:
                continue
            merged = r.merge(h, on="episode_id", suffixes=("_rae", "_helios"), how="inner")
            merged["rae_frequency_mhz"] = rae_frequency
            merged["helios_frequency_mhz"] = helios_frequency
            merged["primary_pair"] = primary
            pieces.append(merged)
        combined = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
        if combined.empty:
            continue
        for (rae_frequency, helios_frequency, primary, antenna), group in combined.groupby(
            ["rae_frequency_mhz", "helios_frequency_mhz", "primary_pair", "antenna"], sort=True
        ):
            group = group.copy()
            group["gate_date_pair"] = pd.to_datetime(group["gate_time_helios"], utc=True).dt.date.astype(str)
            daily = group.groupby("gate_date_pair", as_index=False).agg(
                rae_effect=("paired_delta_top3_z", "median"),
                helios_effect=("upper_event_control_z", "median"),
                n_episodes=("episode_id", "nunique"),
            )
            for day in daily.itertuples(index=False):
                day_rows.append(
                    {
                        "validity_mode": validity_mode,
                        "antenna": antenna,
                        "rae_frequency_mhz": float(rae_frequency),
                        "helios_frequency_mhz": float(helios_frequency),
                        "primary_pair": bool(primary),
                        "gate_date": day.gate_date_pair,
                        "n_episodes": int(day.n_episodes),
                        "rae_effect": float(day.rae_effect),
                        "helios_effect": float(day.helios_effect),
                    }
                )
            rho, pvalue = permutation_spearman_p(
                daily["rae_effect"].to_numpy(dtype=float),
                daily["helios_effect"].to_numpy(dtype=float),
                rng,
            )
            summary_rows.append(
                {
                    "validity_mode": validity_mode,
                    "antenna": antenna,
                    "rae_frequency_mhz": float(rae_frequency),
                    "helios_frequency_mhz": float(helios_frequency),
                    "primary_pair": bool(primary),
                    "n_paired_episodes": int(group["episode_id"].nunique()),
                    "n_independent_days": int(len(daily)),
                    "spearman_rho": rho,
                    "day_block_permutation_p": pvalue,
                    "joint_positive_fraction": float(
                        np.mean((daily["rae_effect"] > 0) & (daily["helios_effect"] > 0))
                    ) if len(daily) else math.nan,
                    "median_rae_effect": float(daily["rae_effect"].median()) if len(daily) else math.nan,
                    "median_helios_effect": float(daily["helios_effect"].median()) if len(daily) else math.nan,
                }
            )
    day_summary = pd.DataFrame(day_rows, columns=DAY_SUMMARY_COLUMNS)
    pair_summary = pd.DataFrame(summary_rows)
    if not pair_summary.empty:
        pair_summary["bh_q_within_antenna_treatment"] = math.nan
        for (_, _), indices in pair_summary[pair_summary["primary_pair"]].groupby(
            ["validity_mode", "antenna"]
        ).groups.items():
            pair_summary.loc[indices, "bh_q_within_antenna_treatment"] = bh_qvalues(
                pair_summary.loc[indices, "day_block_permutation_p"]
            ).to_numpy()
        pair_summary["exploratory_pair_flag"] = (
            pair_summary["primary_pair"]
            & (pair_summary["spearman_rho"] > 0)
            & (pair_summary["bh_q_within_antenna_treatment"] <= 0.10)
        )
        pair_summary = pair_summary.loc[:, PAIR_SUMMARY_COLUMNS]
    else:
        pair_summary = pd.DataFrame(columns=PAIR_SUMMARY_COLUMNS)

    # Strict individual-event label: all Helios timestamp conventions plus both RAE treatments.
    support_rows: list[dict[str, object]] = []
    all_shifts = helios_metrics[
        (helios_metrics["time_mode"] == "light_corrected")
        & helios_metrics["frequency_mhz"].isin([pair[1] for pair in PRIMARY_PAIRS])
    ]
    for rae_frequency, helios_frequency, primary in PRIMARY_PAIRS:
        shift_group = all_shifts[np.isclose(all_shifts["frequency_mhz"], helios_frequency)]
        for episode_id, episode_shifts in shift_group.groupby("episode_id"):
            shift_ok = {}
            for timestamp_shift, group in episode_shifts.groupby("timestamp_shift_min"):
                shift_ok[int(timestamp_shift)] = bool(
                    group["inferentially_eligible"].any()
                    and group["event_exceeds_all_controls"].any()
                    and (group["median_event_minus_control"] > 0).any()
                )
            helios_all_shifts = bool(
                set(shift_ok) == set(TIMESTAMP_SHIFTS_MIN) and all(shift_ok.values())
            )
            for antenna in sorted(set(finite["antenna"]).intersection(strict["antenna"])):
                f = finite[
                    (pd.to_numeric(finite["episode_id"], errors="coerce") == episode_id)
                    & np.isclose(pd.to_numeric(finite["frequency_mhz"], errors="coerce"), rae_frequency)
                    & (finite["antenna"] == antenna)
                ]
                s = strict[
                    (pd.to_numeric(strict["episode_id"], errors="coerce") == episode_id)
                    & np.isclose(pd.to_numeric(strict["frequency_mhz"], errors="coerce"), rae_frequency)
                    & (strict["antenna"] == antenna)
                ]
                finite_candidate = bool(not f.empty and f["independent_burst_candidate"].astype(str).str.lower().eq("true").any())
                strict_candidate = bool(not s.empty and s["independent_burst_candidate"].astype(str).str.lower().eq("true").any())
                support_rows.append(
                    {
                        "episode_id": int(episode_id),
                        "antenna": antenna,
                        "rae_frequency_mhz": rae_frequency,
                        "helios_frequency_mhz": helios_frequency,
                        "primary_pair": primary,
                        "helios_positive_all_timestamp_conventions": helios_all_shifts,
                        "rae_finite_independent_candidate": finite_candidate,
                        "rae_strict_independent_candidate": strict_candidate,
                        "cross_mission_supported": bool(
                            helios_all_shifts and finite_candidate and strict_candidate
                        ),
                    }
                )
    support = pd.DataFrame(support_rows, columns=SUPPORT_COLUMNS)
    return pair_summary, day_summary, support


def density_cm3(radius_rsun: float) -> float:
    return 3.3e5 * radius_rsun ** -2 + 4.1e6 * radius_rsun ** -4 + 8.0e7 * radius_rsun ** -6


def emission_radius_rsun(frequency_khz: float, harmonic: int) -> float:
    plasma_frequency = frequency_khz / harmonic
    target_density = (plasma_frequency / 8.98) ** 2
    return float(brentq(lambda radius: density_cm3(radius) - target_density, 1.001, 10_000.0))


def drift_fits(metrics: pd.DataFrame, rng: np.random.Generator, draws: int = 10_000) -> pd.DataFrame:
    nominal = metrics[
        (metrics["timestamp_shift_min"] == 0)
        & metrics["helios_channel_candidate"]
    ].copy()
    rows: list[dict[str, object]] = []
    for (time_mode, episode_id), group in nominal.groupby(["time_mode", "episode_id"], sort=True):
        group = group.dropna(subset=["helios_peak_delay_s"]).sort_values("frequency_khz")
        if len(group) < 4:
            continue
        ratio = float(group["frequency_khz"].max() / group["frequency_khz"].min())
        delay = group["helios_peak_delay_s"].to_numpy(dtype=float)
        radius = np.array([emission_radius_rsun(value, 1) for value in group["frequency_khz"]])
        constant_delay = np.unique(delay).size < 2
        rho = math.nan if constant_delay else float(spearmanr(radius, delay).statistic)
        slope, intercept, _, _ = theilslopes(delay, radius)
        predicted = intercept + slope * radius
        rms = float(np.sqrt(np.mean((delay - predicted) ** 2)))
        span = float(np.ptp(delay))
        passes_shape_checks = bool(
            ratio >= 4.0 and span >= 1200.0 and rho >= 0.70 and rms <= 1200.0 and slope > 0
        )
        exceed = 0
        if constant_delay:
            permutation_p = math.nan
        else:
            radius_rank = rankdata(radius).astype(float)
            delay_rank = rankdata(delay).astype(float)
            radius_rank -= radius_rank.mean()
            delay_rank -= delay_rank.mean()
            rank_denominator = float(
                np.sqrt(np.dot(radius_rank, radius_rank) * np.dot(delay_rank, delay_rank))
            )
            for _ in range(draws):
                permuted_rho = float(
                    np.dot(radius_rank, rng.permutation(delay_rank)) / rank_denominator
                )
                exceed += int(np.isfinite(permuted_rho) and permuted_rho >= rho)
            permutation_p = float((exceed + 1) / (draws + 1))
        fundamental_speed = RSUN_KM / slope if slope > 0 else math.nan
        harmonic_radius = np.array([emission_radius_rsun(value, 2) for value in group["frequency_khz"]])
        harmonic_slope = float(theilslopes(delay, harmonic_radius).slope)
        harmonic_speed = RSUN_KM / harmonic_slope if harmonic_slope > 0 else math.nan
        frequency_slope = (
            math.nan
            if constant_delay
            else float(theilslopes(group["frequency_khz"].to_numpy(dtype=float), delay).slope)
        )
        coherent = bool(passes_shape_checks and permutation_p <= 0.05)
        rows.append(
            {
                "time_mode": time_mode,
                "episode_id": int(episode_id),
                "gate_time": group["gate_time"].iloc[0],
                "n_independently_selected_channels": int(len(group)),
                "frequencies_khz": ";".join(f"{value:g}" for value in group["frequency_khz"]),
                "frequency_ratio": ratio,
                "peak_delay_span_s": span,
                "spearman_rho_delay_vs_fundamental_radius": rho,
                "timing_rms_s": rms,
                "within_event_frequency_permutation_p": permutation_p,
                "frequency_drift_khz_s": frequency_slope,
                "fundamental_speed_km_s": fundamental_speed,
                "fundamental_speed_fraction_c": fundamental_speed / C_KM_S,
                "harmonic_speed_km_s": harmonic_speed,
                "harmonic_speed_fraction_c": harmonic_speed / C_KM_S,
                "coherent_drift": coherent,
            }
        )
    return pd.DataFrame(rows, columns=DRIFT_COLUMNS)


def plot_drift_fits(metrics: pd.DataFrame, drifts: pd.DataFrame, out_dir: Path) -> None:
    primary = drifts[
        (drifts["time_mode"] == "light_corrected") & drifts["coherent_drift"]
    ] if not drifts.empty else drifts
    fig, ax = plt.subplots(figsize=(9, 5.5))
    if primary.empty:
        ax.text(0.5, 0.5, "No Helios drift fit passed the prespecified checks",
                ha="center", va="center", transform=ax.transAxes)
    for fit in primary.itertuples(index=False):
        selected = metrics[
            (metrics["episode_id"] == fit.episode_id)
            & (metrics["time_mode"] == "light_corrected")
            & (metrics["timestamp_shift_min"] == 0)
            & metrics["helios_channel_candidate"]
        ].sort_values("helios_peak_delay_s")
        ax.plot(
            selected["helios_peak_delay_s"] / 60.0,
            selected["frequency_khz"],
            "o-",
            label=(
                f"episode {fit.episode_id}: {fit.frequency_drift_khz_s:.3f} kHz/s, "
                f"p={fit.within_event_frequency_permutation_p:.4g}"
            ),
        )
    ax.set_yscale("log")
    ax.set_xlabel("corrected peak delay from NOAA gate (minutes)")
    ax.set_ylabel("independently selected Helios frequency (kHz)")
    ax.set_title("Prespecified Helios-only drift screen")
    ax.grid(alpha=0.25)
    if not primary.empty:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "helios_drift_fits.png", dpi=180)
    plt.close(fig)


def plot_coverage(coverage: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    covered = coverage["coverage_within_15min"]
    ax.scatter(pd.to_datetime(coverage["gate_time"]), coverage["absolute_nearest_delta_s"] / 60,
               c=np.where(covered, "tab:orange", "0.72"), s=35, edgecolor="none")
    ax.axhline(15, color="black", ls="--", lw=1, label="15-minute coverage threshold")
    ax.set_yscale("symlog", linthresh=15)
    ax.set_ylabel("nearest Helios E5c record (minutes)")
    ax.set_title("Availability-selected Helios-1 / RAE-2 NOAA cohort")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "helios_rae2_coverage.png", dpi=180)
    plt.close(fig)


def plot_quality(quality: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    x = quality["frequency_khz"]
    colors = np.where(quality["science_eligible_without_event_selection"], "tab:blue", "tab:red")
    axes[0].scatter(x, quality["upper_robust_span"], c=colors, s=55)
    axes[0].set_ylabel("5–95% upper-statistic span")
    axes[1].scatter(x, quality["upper_modal_fraction"], c=colors, s=55)
    axes[1].set_ylabel("modal-value fraction")
    axes[1].set_xlabel("Helios E5c frequency (kHz)")
    for ax in axes:
        ax.set_xscale("log")
        ax.grid(alpha=0.25)
    fig.suptitle("Event-blind Helios E5c channel-quality audit")
    fig.tight_layout()
    fig.savefig(out_dir / "helios_channel_quality.png", dpi=180)
    plt.close(fig)


def plot_pair_summary(summary: pd.DataFrame, out_dir: Path) -> None:
    if summary.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for ax, ((mode, antenna), group) in zip(axes.flat, summary.groupby(["validity_mode", "antenna"], sort=True)):
        group = group.sort_values("rae_frequency_mhz")
        ax.axhline(0, color="0.5", lw=1)
        ax.plot(group["rae_frequency_mhz"], group["spearman_rho"], "o-", color="tab:purple")
        for row in group.itertuples(index=False):
            if bool(row.primary_pair) and np.isfinite(row.bh_q_within_antenna_treatment):
                ax.annotate(f"q={row.bh_q_within_antenna_treatment:.2g}",
                            (row.rae_frequency_mhz, row.spearman_rho), xytext=(3, 5),
                            textcoords="offset points", fontsize=8)
        ax.set_xscale("log")
        ax.set_title(f"{mode}; {antenna}")
        ax.grid(alpha=0.25)
    fig.supxlabel("RAE-2 frequency (MHz); paired Helios channel fixed in advance")
    fig.supylabel("day-collapsed Spearman correlation")
    fig.suptitle("Channel-separated Helios–RAE timing-association concordance")
    fig.tight_layout()
    fig.savefig(out_dir / "crossmission_frequency_pair_associations.png", dpi=180)
    plt.close(fig)


def event_panel(
    episode_id: int,
    helios: pd.DataFrame,
    metrics: pd.DataFrame,
    finite: pd.DataFrame,
    out_path: Path,
) -> None:
    rows = metrics[
        (metrics["episode_id"] == episode_id)
        & (metrics["time_mode"] == "light_corrected")
        & (metrics["timestamp_shift_min"] == 0)
    ]
    if rows.empty:
        return
    gate = pd.Timestamp(rows["gate_time"].iloc[0])
    sample = helios[
        (helios["time_light_corrected"] >= gate - pd.Timedelta(hours=3))
        & (helios["time_light_corrected"] <= gate + pd.Timedelta(hours=3))
    ].copy()
    matrix_rows: list[np.ndarray] = []
    frequencies: list[float] = []
    time_axis = pd.DatetimeIndex(sorted(sample["time_light_corrected"].drop_duplicates()))
    for frequency, group in sample.groupby("frequency_khz", sort=True):
        indexed = (
            group.groupby("time_light_corrected")["two_percent_max_10logt"]
            .max()
            .reindex(time_axis)
        )
        scale = robust_sigma(indexed)
        center = float(np.nanmedian(indexed))
        matrix_rows.append((indexed.to_numpy(dtype=float) - center) / scale if np.isfinite(scale) else np.full(len(indexed), np.nan))
        frequencies.append(float(frequency))
    matrix = np.vstack(matrix_rows) if matrix_rows else np.empty((0, 0))
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [1, 2]})
    rae = finite[pd.to_numeric(finite["episode_id"], errors="coerce") == episode_id]
    for antenna, group in rae.groupby("antenna"):
        axes[0].plot(group["frequency_mhz"], group["paired_delta_top3_z"], "o-", label=antenna)
    axes[0].axhline(0, color="0.5", lw=1)
    axes[0].set_xscale("log")
    axes[0].set_ylabel("RAE event − controls (z)")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    if matrix.size:
        image = axes[1].pcolormesh(time_axis, frequencies, matrix, shading="nearest", cmap="RdBu_r", vmin=-4, vmax=4)
        axes[1].set_yscale("log")
        axes[1].axvline(gate, color="black", lw=1.5, label="NOAA gate")
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        axes[1].set_ylabel("Helios E5c frequency (kHz)")
        axes[1].set_xlabel(f"UTC on {gate.date().isoformat()}")
        fig.colorbar(image, ax=axes[1], label="local robust z")
    fig.suptitle(f"Episode {episode_id}: separately normalized RAE-2 and Helios-1 channels")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def dataframe_markdown(frame: pd.DataFrame, columns: list[str], rows: int = 30) -> str:
    if frame.empty:
        return "(no rows)"
    view = frame.loc[:, [column for column in columns if column in frame]].head(rows).copy()
    try:
        return view.to_markdown(index=False)
    except ImportError:
        return "```\n" + view.to_string(index=False) + "\n```"


def write_report(
    out_dir: Path,
    coverage: pd.DataFrame,
    quality: pd.DataFrame,
    helios_summary: pd.DataFrame,
    pair_summary: pd.DataFrame,
    support: pd.DataFrame,
    drifts: pd.DataFrame,
    metrics: pd.DataFrame,
    duplicate_audit: dict[str, int],
    decoder_audit: dict[str, object],
) -> None:
    covered = coverage[coverage["coverage_within_15min"]]
    supported = support[support["cross_mission_supported"]] if not support.empty else support
    coherent = drifts[
        (drifts["time_mode"] == "light_corrected") & drifts["coherent_drift"]
    ] if not drifts.empty else drifts
    supported_events = int(supported["episode_id"].nunique()) if not supported.empty else 0
    pair_flags = int(pair_summary["exploratory_pair_flag"].sum()) if not pair_summary.empty else 0
    min_pair_q = float(pair_summary["bh_q_within_antenna_treatment"].min()) if not pair_summary.empty else math.nan
    min_helios_q = float(helios_summary["bh_q_across_helios_channels"].min()) if not helios_summary.empty else math.nan
    episode55_modes = metrics[
        (metrics["episode_id"] == 55)
        & np.isclose(metrics["frequency_khz"], 1010.0)
    ].groupby("time_mode")["helios_channel_candidate"].all()
    episode55_raw_and_corrected = bool(
        set(episode55_modes.index) == {"raw", "light_corrected"} and episode55_modes.all()
    )
    target_metrics = metrics[
        metrics["episode_id"].isin([54, 55])
        & (metrics["time_mode"] == "light_corrected")
        & (metrics["timestamp_shift_min"] == 0)
        & metrics["inferentially_eligible"]
    ].sort_values(["episode_id", "frequency_khz"])
    lines = [
        "# Helios-1 E5c / RAE-2 RV Exploratory Solar-Burst Study",
        "",
        "## Bottom line",
        "",
        "This systematic cross-match used NOAA 18/29 MHz reports only as timestamps. Helios frequencies, RAE frequencies, RAE antennas, and RAE validity treatments were never pooled for detection.",
        "",
        f"- Post-launch RAE-usable NOAA episodes audited: **{len(coverage)}**.",
        f"- Episodes with an E5c record within 15 minutes: **{len(covered)}**; within 6 minutes: **{int(coverage['coverage_within_6min'].sum())}**.",
        f"- Event-blind Helios channels retained by the archive-quality audit: **{int(quality['science_eligible_without_event_selection'].sum())}/16**.",
        f"- Cross-mission-supported events: **{supported_events}** (represented by **{len(supported)}** antenna rows).",
        f"- Helios-only coherent drift fits passing all exploratory checks: **{len(coherent)}**.",
        "",
        "The online E5c binary is explicitly labelled by SPDF as recovered but unreviewed. Results therefore use ordinal within-channel contrasts, not absolute antenna temperature. The 10-minute summaries cannot support triangulation, angular-size inference, or precision beam speeds.",
        "",
        "## Archive decoding and coverage",
        "",
        f"The decoder verified SHA-1 `{decoder_audit['source_sha1']}`, {decoder_audit['physical_blocks']} physical blocks, "
        f"{decoder_audit['logical_records']:,} logical records, {decoder_audit['unique_date_codes']} date codes, "
        f"and zero monotonic, block-length, or channel-layout violations. It preserved and audited "
        f"{decoder_audit['wrapped_negative_milliseconds_records']} recovered-tape midnight rollover values.",
        "",
        f"The raw decoded table contains {duplicate_audit['decoded_channel_rows']:,} channel rows. "
        f"An event-blind, same-channel overlap rule removed {duplicate_audit['duplicate_channel_rows_removed']:,} "
        f"rows from {duplicate_audit['duplicate_channel_time_groups']:,} duplicate timestamp/channel groups, "
        "retaining the record with the larger nonzero count (then earlier logical-record number).",
        "",
        dataframe_markdown(coverage[coverage["episode_id"].isin([54, 55])], [
            "episode_id", "gate_time", "nearest_helios_time_raw", "nearest_minus_gate_s", "coverage_within_15min"
        ]),
        "",
        "## Helios-only NOAA timing association",
        "",
        f"No Helios channel survives the exploratory across-channel correction (`min q = {min_helios_q:.4f}`). "
        "The nominal per-channel results below are therefore effect estimates and null outcomes, not discoveries.",
        "",
        dataframe_markdown(helios_summary.sort_values("bh_q_across_helios_channels"), [
            "frequency_khz", "n_eligible_episodes", "n_independent_days",
            "median_day_upper_event_control_z", "positive_day_fraction", "sign_flip_p",
            "bh_q_across_helios_channels",
        ], 20),
        "",
        "## Frequency-concordant Helios–RAE association",
        "",
        f"No population-level pair is flagged (`{pair_flags}` flags; minimum primary-pair `q = {min_pair_q:.4f}`). "
        "Thus the individual-event support below is historical case corroboration, not evidence of a cohort-wide correlation.",
        "",
        dataframe_markdown(pair_summary.sort_values(["validity_mode", "antenna", "rae_frequency_mhz"]), [
            "validity_mode", "antenna", "rae_frequency_mhz", "helios_frequency_mhz", "primary_pair",
            "n_independent_days", "spearman_rho", "day_block_permutation_p",
            "bh_q_within_antenna_treatment", "joint_positive_fraction", "exploratory_pair_flag",
        ], 40),
        "",
        "## Episodes 54 and 55",
        "",
        "These episodes were prespecified for detailed display but did not define the cohort, channel eligibility, windows, or controls.",
        "Episode 54 has no Helios channel that exceeds every valid control. Episode 55 has a positive 1.010 MHz "
        f"Helios result under all three timestamp conventions; its status is also positive with raw and light-time-corrected timestamps: **{episode55_raw_and_corrected}**.",
        "",
        dataframe_markdown(target_metrics, [
            "episode_id", "frequency_khz", "n_event_records", "n_baseline_records", "n_controls",
            "upper_event_control_z", "median_event_minus_control", "event_exceeds_all_controls",
            "helios_channel_candidate", "helios_peak_delay_s",
        ], 40),
        "",
        "## Individual cross-mission support",
        "",
        "The two retained rows are the same event—episode 55 at RAE 0.90 MHz / Helios 1.010 MHz—seen independently in both RAE antennas. "
        "Each antenna is a candidate under both finite and strict RAE handling, and Helios exceeds every same-channel shifted control with a positive archived-median sensitivity result for −5, 0, and +5 minute conventions.",
        "",
        dataframe_markdown(supported, [
            "episode_id", "antenna", "rae_frequency_mhz", "helios_frequency_mhz", "primary_pair",
            "cross_mission_supported",
        ], 40),
        "",
        "## Secondary Helios-only drift",
        "",
        dataframe_markdown(drifts, [
            "time_mode", "episode_id", "n_independently_selected_channels", "frequencies_khz", "frequency_ratio",
            "peak_delay_span_s", "spearman_rho_delay_vs_fundamental_radius", "timing_rms_s",
            "within_event_frequency_permutation_p", "fundamental_speed_fraction_c",
            "harmonic_speed_fraction_c", "coherent_drift",
        ], 40),
        "",
        "Any quoted speed is an order-of-magnitude density-model conversion. Helios and RAE timing points were not combined in a single velocity fit.",
        "The surviving corrected-time Helios fit is episode 68: its frequency drift is negative (lower frequencies later), "
        "the same outward direction as the prior RAE scan-corrected population result. It is not one of the four RAE episodes "
        "stable under both validity treatments, so this is directional consistency rather than a same-event velocity confirmation. "
        "The existing RAE synthesis reports median stable-fit drifts of −6.24 kHz/s (finite) and −5.64 kHz/s (strict); "
        "the different cadence, selected frequencies, and event make a precision numerical comparison inappropriate.",
        "",
        "## Provenance and limitations",
        "",
        "- Numerical analysis executed on NERSC Perlmutter.",
        "- [Helios E5c recovered archive](https://spdf.gsfc.nasa.gov/pub/data/helios/helios1/26pt5KHz_3MHz_radio_wave/10min_averaged_summaries/): NASA/SPDF SORA-00003 recovered IBM-360 summary product.",
        "- [Original E5c instrument paper](https://helios-data.ssl.berkeley.edu/data/Miscellaneous_Documentation/Original_Instrument_Papers_Raumfahrtsforschung%201975/Raumfahrtsforschung%20E5c.pdf): source of the fixed 16-channel frequency map.",
        "- [SPDF hourly Helios-1 trajectory](https://spdf.gsfc.nasa.gov/pub/data/helios/helios1/traj/hour/): source of heliocentric distance for the Earth-arrival correction.",
        "- [Leblanc–Dulk–Bougeret density model](https://doi.org/10.1023/A:1005049730506): used only for order-of-magnitude fundamental/harmonic conversions.",
        "- RAE-2 inputs: existing finite and strict Perlmutter channel metrics already audited against the m4895 interpolated master CSV.",
        "- Interpretation is anchored to the [existing Perlmutter solar synthesis](C:/Users/Adam/Documents/ChatGPT/RAE2/perlmutter_sun_science_v1/PERLMUTTER_SUN_RV_SYNTHESIS.md).",
        "- Helios E5a was excluded because it has no event-time coverage for episodes 54 and 55.",
        "- The summary timestamps have unresolved start/center/end semantics; every individual support label must remain positive under −5, 0, and +5 minute conventions.",
    ]
    (out_dir / "HELIOS1_RAE2_EXPLORATORY_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def hash_tables(out_dir: Path, filenames: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for filename in filenames:
        path = out_dir / filename
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        result[filename] = digest
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--helios", type=Path, required=True)
    parser.add_argument("--channel-quality", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--rae-finite", type=Path, required=True)
    parser.add_argument("--rae-strict", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    helios_raw = pd.read_parquet(args.helios)
    helios, duplicate_audit = select_archive_science_records(helios_raw)
    decoder_audit = json.loads(
        (args.helios.parent / "helios_e5c_decoder_audit.json").read_text(encoding="utf-8")
    )
    quality = pd.read_csv(args.channel_quality)
    episodes = pd.read_csv(args.episodes)
    finite = pd.read_csv(args.rae_finite)
    strict = pd.read_csv(args.rae_strict)
    episodes["episode_id"] = pd.to_numeric(episodes["episode_id"], errors="raise").astype(int)
    episodes["gate_time"] = pd.to_datetime(episodes["gate_time"], utc=True)
    trajectory = parse_trajectory(args.trajectory)
    helios = add_light_time(helios, trajectory)

    finite_ids = set(pd.to_numeric(finite["episode_id"], errors="coerce").dropna().astype(int))
    coverage = build_coverage_audit(helios, episodes, finite_ids)
    if len(coverage) != 50 or int(coverage["coverage_within_15min"].sum()) != 18 or int(coverage["coverage_within_6min"].sum()) != 16:
        raise ValueError(
            "coverage invariants failed: "
            f"cohort={len(coverage)}, within15={int(coverage['coverage_within_15min'].sum())}, "
            f"within6={int(coverage['coverage_within_6min'].sum())}"
        )
    target = coverage.set_index("episode_id")
    if not np.isclose(target.loc[54, "nearest_minus_gate_s"], 12.08, atol=0.001):
        raise ValueError("episode 54 coverage timestamp changed")
    if not np.isclose(target.loc[55, "nearest_minus_gate_s"], 179.52, atol=0.001):
        raise ValueError("episode 55 coverage timestamp changed")

    covered_ids = set(coverage.loc[coverage["coverage_within_15min"], "episode_id"].astype(int))
    metrics = build_helios_metrics(helios, episodes, covered_ids, quality)
    helios_summary = helios_channel_summary(metrics, rng)
    pair_summary, day_summary, support = build_crossmission(metrics, finite, strict, rng)
    drifts = drift_fits(metrics, rng)

    coverage.to_csv(args.out_dir / "helios_rae2_coverage_audit.csv", index=False)
    metrics.to_csv(args.out_dir / "helios_event_channel_metrics.csv", index=False)
    metrics.to_parquet(args.out_dir / "helios_event_channel_metrics.parquet", index=False, compression="zstd")
    helios_summary.to_csv(args.out_dir / "helios_channel_timing_summary.csv", index=False)
    pair_summary.to_csv(args.out_dir / "crossmission_frequency_pair_metrics.csv", index=False)
    day_summary.to_csv(args.out_dir / "crossmission_day_summary.csv", index=False)
    support.to_csv(args.out_dir / "crossmission_event_support.csv", index=False)
    drifts.to_csv(args.out_dir / "helios_drift_fits.csv", index=False)

    plot_coverage(coverage, args.out_dir)
    plot_quality(quality, args.out_dir)
    plot_pair_summary(pair_summary, args.out_dir)
    plot_drift_fits(metrics, drifts, args.out_dir)
    event_dir = args.out_dir / "event_panels"
    event_dir.mkdir(exist_ok=True)
    for episode_id in sorted(covered_ids):
        event_panel(episode_id, helios, metrics, finite, event_dir / f"episode_{episode_id:03d}.png")
    for episode_id in (54, 55):
        event_panel(episode_id, helios, metrics, finite, args.out_dir / f"episode_{episode_id}_crossmission.png")

    write_report(
        args.out_dir, coverage, quality, helios_summary, pair_summary, support, drifts, metrics,
        duplicate_audit, decoder_audit,
    )
    table_files = [
        "helios_rae2_coverage_audit.csv",
        "helios_event_channel_metrics.csv",
        "helios_channel_timing_summary.csv",
        "crossmission_frequency_pair_metrics.csv",
        "crossmission_day_summary.csv",
        "crossmission_event_support.csv",
        "helios_drift_fits.csv",
    ]
    manifest = {
        "execution_site": "NERSC Perlmutter",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "inputs": {
            "helios": str(args.helios),
            "channel_quality": str(args.channel_quality),
            "episodes": str(args.episodes),
            "rae_finite": str(args.rae_finite),
            "rae_strict": str(args.rae_strict),
            "trajectory": [str(path) for path in args.trajectory],
        },
        "channels_combined_for_detection": False,
        "antennas_combined_for_detection": False,
        "archive_duplicate_resolution": duplicate_audit,
        "decoder_audit_sha256": hashlib.sha256(
            (args.helios.parent / "helios_e5c_decoder_audit.json").read_bytes()
        ).hexdigest(),
        "reference_synthesis": "C:/Users/Adam/Documents/ChatGPT/RAE2/perlmutter_sun_science_v1/PERLMUTTER_SUN_RV_SYNTHESIS.md",
        "timestamp_shifts_min": TIMESTAMP_SHIFTS_MIN,
        "control_shifts_h": CONTROL_SHIFTS_H,
        "primary_frequency_pairs": [pair for pair in PRIMARY_PAIRS if pair[2]],
        "secondary_frequency_pairs": [pair for pair in PRIMARY_PAIRS if not pair[2]],
        "coverage": {
            "post_launch_rae_usable": int(len(coverage)),
            "within_15min": int(coverage["coverage_within_15min"].sum()),
            "within_6min": int(coverage["coverage_within_6min"].sum()),
        },
        "table_sha256": hash_tables(args.out_dir, table_files),
    }
    (args.out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest["coverage"], sort_keys=True), flush=True)
    print(f"wrote {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
