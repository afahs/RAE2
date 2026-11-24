#!/usr/bin/env python
"""Search raw RAE-2 Ryle-Vonberg data at NOAA low-frequency burst times.

The historical NOAA 18/29 MHz reports are used only as timestamps.  Every
RAE-2 frequency and antenna is evaluated independently; no multi-frequency
coincidence rule or cross-channel score is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FREQUENCY_MHZ = {
    1: 0.45,
    2: 0.70,
    3: 0.90,
    4: 1.31,
    5: 2.20,
    6: 3.93,
    7: 4.70,
    8: 6.55,
    9: 9.18,
}
ANTENNAS = ("rv1_coarse", "rv2_coarse")
CONTROL_SHIFTS_H = (-6.0, -3.0, 3.0, 6.0)
SEARCH_WINDOWS_S = {
    "same_burst_20m": (-300.0, 900.0),
    "activity_75m": (-900.0, 3600.0),
}
EXTRACTION_HALF_WINDOW_S = 5400.0
MISSION_START = pd.Timestamp("1974-11-01 00:00:00")
MISSION_END = pd.Timestamp("1975-06-28 23:59:59")


def datetime_ns(values: pd.Series) -> np.ndarray:
    """Return nanoseconds explicitly (pandas may otherwise preserve microseconds)."""

    return pd.to_datetime(values, errors="coerce").astype("datetime64[ns]").astype("int64").to_numpy()


def robust_scale(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    center = float(np.median(values))
    scale = float(1.4826 * np.median(np.abs(values - center)))
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = float(np.std(values, ddof=1)) if values.size > 1 else np.nan
    return scale


def cluster_low_frequency_reports(
    reports: pd.DataFrame,
    maximum_frequency_mhz: float = 30.0,
    cluster_gap_minutes: float = 30.0,
) -> pd.DataFrame:
    """Collapse duplicate/station reports into low-frequency activity episodes."""

    work = reports.copy()
    work["start_time"] = pd.to_datetime(work["start_time"], errors="coerce")
    work["frequency_mhz_reported"] = pd.to_numeric(
        work["frequency_mhz_reported"], errors="coerce"
    )
    work = work[
        work["start_time"].between(MISSION_START, MISSION_END)
        & (work["frequency_mhz_reported"] <= float(maximum_frequency_mhz))
    ].sort_values("start_time")
    if work.empty:
        return pd.DataFrame()
    gaps = work["start_time"].diff().dt.total_seconds().fillna(np.inf)
    work["episode_id"] = (gaps > float(cluster_gap_minutes) * 60.0).cumsum().astype(int)
    episodes = (
        work.groupby("episode_id", observed=True, sort=True)
        .agg(
            gate_time=("start_time", "min"),
            last_report_time=("start_time", "max"),
            n_low_frequency_reports=("start_time", "size"),
            minimum_reported_frequency_mhz=("frequency_mhz_reported", "min"),
        )
        .reset_index()
    )
    return episodes


def build_centers(episodes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    gate_ns = datetime_ns(episodes["gate_time"])
    quiet_radius_ns = int(90.0 * 60.0 * 1e9)
    for episode in episodes.itertuples(index=False):
        for shift_h in (0.0, *CONTROL_SHIFTS_H):
            center_time = pd.Timestamp(episode.gate_time) + pd.to_timedelta(shift_h, unit="h")
            distance = np.min(np.abs(gate_ns - center_time.value)) if gate_ns.size else np.inf
            is_real = np.isclose(shift_h, 0.0)
            rows.append(
                {
                    "center_id": len(rows),
                    "episode_id": int(episode.episode_id),
                    "gate_time": pd.Timestamp(episode.gate_time),
                    "center_time": center_time,
                    "gate_role": "real" if is_real else "time_shift_control",
                    "control_shift_h": float(shift_h),
                    "valid_quiet_control": bool(is_real or distance > quiet_radius_ns),
                    "n_low_frequency_reports": int(episode.n_low_frequency_reports),
                    "minimum_reported_frequency_mhz": float(
                        episode.minimum_reported_frequency_mhz
                    ),
                }
            )
    return pd.DataFrame.from_records(rows)


def extract_raw_windows(
    master_csv: Path,
    centers: pd.DataFrame,
    chunksize: int = 500_000,
) -> dict[int, dict[str, list[np.ndarray]]]:
    """Stream the 3 GB master CSV and retain only samples near gate/control times."""

    center_times = datetime_ns(centers["center_time"])
    half_ns = int(EXTRACTION_HALF_WINDOW_S * 1e9)
    buffers: dict[int, dict[str, list[np.ndarray]]] = {
        int(center_id): {"relative_s": [], "frequency_band": [], **{name: [] for name in ANTENNAS}}
        for center_id in centers["center_id"]
    }
    usecols = ["time", "frequency_band", *ANTENNAS]
    global_lo = int(center_times.min() - half_ns)
    global_hi = int(center_times.max() + half_ns)
    reader = pd.read_csv(master_csv, usecols=usecols, chunksize=int(chunksize), low_memory=False)
    for chunk_index, chunk in enumerate(reader, start=1):
        parsed = pd.to_datetime(chunk["time"], errors="coerce")
        valid_time = ~parsed.isna()
        if not np.any(valid_time):
            continue
        chunk = chunk.loc[valid_time].copy()
        chunk["time"] = parsed.loc[valid_time]
        time_ns = datetime_ns(chunk["time"])
        order = np.argsort(time_ns, kind="stable")
        if not np.all(order == np.arange(len(order))):
            chunk = chunk.iloc[order].reset_index(drop=True)
            time_ns = time_ns[order]
        if time_ns[0] > global_hi:
            print("[direct-burst] passed the final requested window; stopping CSV scan", flush=True)
            break
        if time_ns[-1] < global_lo:
            continue
        first_center = int(np.searchsorted(center_times, time_ns[0] - half_ns, side="left"))
        last_center = int(np.searchsorted(center_times, time_ns[-1] + half_ns, side="right"))
        for center_position in range(first_center, last_center):
            center_ns = int(center_times[center_position])
            lo = int(np.searchsorted(time_ns, center_ns - half_ns, side="left"))
            hi = int(np.searchsorted(time_ns, center_ns + half_ns, side="right"))
            if hi <= lo:
                continue
            center_id = int(centers.iloc[center_position]["center_id"])
            target = buffers[center_id]
            target["relative_s"].append((time_ns[lo:hi] - center_ns).astype(float) / 1e9)
            target["frequency_band"].append(
                pd.to_numeric(chunk.iloc[lo:hi]["frequency_band"], errors="coerce").to_numpy()
            )
            for antenna in ANTENNAS:
                target[antenna].append(
                    pd.to_numeric(chunk.iloc[lo:hi][antenna], errors="coerce").to_numpy()
                )
        if chunk_index % 10 == 0:
            print(f"[direct-burst] streamed {chunk_index:,} chunks", flush=True)
    return buffers


def calculate_center_metrics(
    centers: pd.DataFrame,
    buffers: dict[int, dict[str, list[np.ndarray]]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for center in centers.itertuples(index=False):
        payload = buffers[int(center.center_id)]
        if not payload["relative_s"]:
            continue
        relative_s = np.concatenate(payload["relative_s"])
        bands = np.concatenate(payload["frequency_band"])
        antenna_power = {
            antenna: np.concatenate(payload[antenna])
            for antenna in ANTENNAS
        }
        for band, frequency_mhz in FREQUENCY_MHZ.items():
            band_mask = np.isclose(bands, float(band), equal_nan=False)
            if not np.any(band_mask):
                continue
            channel_time = relative_s[band_mask]
            for antenna in ANTENNAS:
                raw_power = antenna_power[antenna][band_mask]
                finite = np.isfinite(channel_time) & np.isfinite(raw_power) & (raw_power > 0.0)
                t = channel_time[finite]
                log_power = np.log10(raw_power[finite])
                baseline_mask = (
                    ((t >= -5400.0) & (t < -3600.0))
                    | ((t > 3600.0) & (t <= 5400.0))
                )
                baseline = log_power[baseline_mask]
                if baseline.size < 10:
                    continue
                baseline_center = float(np.median(baseline))
                baseline_scale = robust_scale(baseline)
                if not np.isfinite(baseline_scale) or baseline_scale <= 0.0:
                    continue
                for window_name, (window_lo, window_hi) in SEARCH_WINDOWS_S.items():
                    candidate_mask = (t >= window_lo) & (t <= window_hi)
                    candidate = log_power[candidate_mask]
                    candidate_t = t[candidate_mask]
                    if candidate.size < 3:
                        continue
                    z = (candidate - baseline_center) / baseline_scale
                    order = np.argsort(z)
                    top = z[order[-min(3, len(order)):]]
                    peak_index = int(np.nanargmax(z))
                    rows.append(
                        {
                            "center_id": int(center.center_id),
                            "episode_id": int(center.episode_id),
                            "gate_time": pd.Timestamp(center.gate_time),
                            "center_time": pd.Timestamp(center.center_time),
                            "gate_role": str(center.gate_role),
                            "control_shift_h": float(center.control_shift_h),
                            "valid_quiet_control": bool(center.valid_quiet_control),
                            "frequency_band": int(band),
                            "frequency_mhz": float(frequency_mhz),
                            "antenna": antenna,
                            "search_window": window_name,
                            "n_baseline_samples": int(baseline.size),
                            "n_candidate_samples": int(candidate.size),
                            "baseline_log_power": baseline_center,
                            "baseline_log_scale": baseline_scale,
                            "peak_z": float(z[peak_index]),
                            "peak_delta_s": float(candidate_t[peak_index]),
                            "top3_median_z": float(np.median(top)),
                            "q90_z": float(np.quantile(z, 0.90)),
                        }
                    )
    return pd.DataFrame.from_records(rows)


def build_paired_differences(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["frequency_band", "frequency_mhz", "antenna", "search_window"]
    for key, channel in metrics.groupby(keys, observed=True, sort=True):
        for episode_id, episode in channel.groupby("episode_id", observed=True, sort=True):
            real = episode[episode["gate_role"].eq("real")]
            controls = episode[
                episode["gate_role"].eq("time_shift_control")
                & episode["valid_quiet_control"]
            ]
            if real.empty or len(controls) < 2:
                continue
            real_row = real.iloc[0]
            control_value = float(controls["top3_median_z"].median())
            rows.append(
                {
                    **dict(zip(keys, key)),
                    "episode_id": int(episode_id),
                    "gate_time": real_row["gate_time"],
                    "real_top3_median_z": float(real_row["top3_median_z"]),
                    "control_median_top3_z": control_value,
                    "paired_delta_top3_z": float(real_row["top3_median_z"] - control_value),
                    "real_peak_z": float(real_row["peak_z"]),
                    "real_peak_delta_s": float(real_row["peak_delta_s"]),
                    "n_valid_controls": int(len(controls)),
                    "real_exceeds_all_controls": bool(
                        float(real_row["top3_median_z"]) > float(controls["top3_median_z"].max())
                    ),
                }
            )
    return pd.DataFrame.from_records(rows)


def sign_flip_pvalue(differences: np.ndarray, seed: int, draws: int = 20_000) -> float:
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    observed = float(np.mean(values))
    if observed <= 0.0:
        return 1.0
    rng = np.random.default_rng(seed)
    exceed = 0
    completed = 0
    batch = 1000
    while completed < int(draws):
        size = min(batch, int(draws) - completed)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(size, len(values)))
        permuted = np.mean(signs * values[None, :], axis=1)
        exceed += int(np.sum(permuted >= observed))
        completed += size
    return float((1 + exceed) / (1 + int(draws)))


def benjamini_hochberg(pvalues: pd.Series) -> np.ndarray:
    values = pd.to_numeric(pvalues, errors="coerce").to_numpy(dtype=float)
    result = np.full(len(values), np.nan)
    finite = np.isfinite(values)
    if not np.any(finite):
        return result
    selected = np.where(finite)[0]
    order = selected[np.argsort(values[selected])]
    ranked = values[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    result[order] = np.minimum(ranked, 1.0)
    return result


def summarize_channels(paired: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["frequency_band", "frequency_mhz", "antenna", "search_window"]
    for index, (key, channel) in enumerate(
        paired.groupby(keys, observed=True, sort=True), start=1
    ):
        delta = channel["paired_delta_top3_z"].to_numpy(dtype=float)
        rows.append(
            {
                **dict(zip(keys, key)),
                "n_paired_episodes": int(len(channel)),
                "median_real_top3_z": float(channel["real_top3_median_z"].median()),
                "median_control_top3_z": float(channel["control_median_top3_z"].median()),
                "median_paired_delta_z": float(np.median(delta)),
                "mean_paired_delta_z": float(np.mean(delta)),
                "positive_paired_fraction": float(np.mean(delta > 0.0)),
                "real_exceeds_all_controls_fraction": float(
                    channel["real_exceeds_all_controls"].mean()
                ),
                "median_real_peak_delta_s": float(channel["real_peak_delta_s"].median()),
                "one_sided_sign_flip_p": sign_flip_pvalue(delta, seed=20260831 + index),
            }
        )
    out = pd.DataFrame.from_records(rows)
    if out.empty:
        return out
    out["bh_q_across_reported_tests"] = benjamini_hochberg(out["one_sided_sign_flip_p"])
    out["evidence_class"] = np.where(
        (out["n_paired_episodes"] >= 15)
        & (out["median_paired_delta_z"] > 0.0)
        & (out["positive_paired_fraction"] >= 0.60)
        & (out["bh_q_across_reported_tests"] <= 0.10),
        "timestamp_associated_burst_candidate",
        "not_detected",
    )
    return out.sort_values(
        ["bh_q_across_reported_tests", "one_sided_sign_flip_p", "frequency_mhz", "antenna"]
    ).reset_index(drop=True)


def summarize_day_blocks(
    paired: pd.DataFrame,
    delta_column: str,
    key_columns: list[str],
    seed_base: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Repeat inference after reducing clustered episodes to one value per day."""

    work = paired.copy()
    work["gate_date"] = pd.to_datetime(work["gate_time"], errors="coerce").dt.floor("D")
    daily = (
        work.groupby([*key_columns, "gate_date"], observed=True, sort=True)[delta_column]
        .median()
        .rename("day_median_delta")
        .reset_index()
    )
    rows: list[dict[str, object]] = []
    for index, (key, channel) in enumerate(
        daily.groupby(key_columns, observed=True, sort=True), start=1
    ):
        if not isinstance(key, tuple):
            key = (key,)
        values = channel["day_median_delta"].to_numpy(dtype=float)
        rows.append(
            {
                **dict(zip(key_columns, key)),
                "n_independent_days": int(len(values)),
                "median_day_delta": float(np.median(values)),
                "mean_day_delta": float(np.mean(values)),
                "positive_day_fraction": float(np.mean(values > 0.0)),
                "one_sided_day_sign_flip_p": sign_flip_pvalue(
                    values, seed=int(seed_base + index)
                ),
            }
        )
    summary = pd.DataFrame.from_records(rows)
    if summary.empty:
        return daily, summary
    summary["bh_q_across_day_block_tests"] = benjamini_hochberg(
        summary["one_sided_day_sign_flip_p"]
    )
    summary["day_block_evidence_class"] = np.where(
        (summary["n_independent_days"] >= 10)
        & (summary["median_day_delta"] > 0.0)
        & (summary["positive_day_fraction"] >= 0.60)
        & (summary["bh_q_across_day_block_tests"] <= 0.10),
        "day_block_timestamp_candidate",
        "not_detected",
    )
    summary = summary.sort_values(
        ["bh_q_across_day_block_tests", "one_sided_day_sign_flip_p", *key_columns]
    ).reset_index(drop=True)
    return daily, summary


def write_report(
    output_dir: Path,
    master_csv: Path,
    episodes: pd.DataFrame,
    summary: pd.DataFrame,
    day_summary: pd.DataFrame,
) -> Path:
    top = summary.head(12)
    focus = summary[
        np.isclose(summary["frequency_mhz"], 0.70)
        | np.isclose(summary["frequency_mhz"], 4.70)
    ]
    candidates = summary[summary["evidence_class"].eq("timestamp_associated_burst_candidate")]
    day_candidates = day_summary[
        day_summary["day_block_evidence_class"].eq("day_block_timestamp_candidate")
    ]
    lines = [
        "# Direct RAE-2 Search at NOAA 18/29 MHz Burst Times",
        "",
        "Each Ryle–Vonberg frequency and antenna was tested independently. NOAA frequencies were used only to choose times; no RAE frequency channels were combined.",
        "",
        f"- Raw mission file: `{master_csv}`",
        f"- Low-frequency NOAA episodes: {len(episodes)}",
        "- Real windows were compared with local ±3 h and ±6 h time-shift controls.",
        "- Primary statistic: median of the three largest locally normalized log-power samples.",
        "- Two declared time windows: −5 to +15 minutes and −15 to +60 minutes.",
        "",
        "## Result",
        "",
        (
            f"{len(candidates)} channel/window rows meet the exploratory timestamp-association rule."
            if not candidates.empty
            else "No channel/window row meets the exploratory timestamp-association rule after correction across the reported channel-local tests."
        ),
        "",
        "## 0.70 and 4.70 MHz",
        "",
        focus.to_string(index=False) if not focus.empty else "No usable paired rows.",
        "",
        "## Strongest channel-local rows",
        "",
        top.to_string(index=False) if not top.empty else "No usable paired rows.",
        "",
        "## Interpretation limits",
        "",
        "This is a direct burst-timing association test, not an occultation test. Local normalization and nearby controls reduce slow Galactic-background changes, but telemetry artifacts and solar reports missed by the ground network can still affect individual windows. The BH column corrects multiplicity without averaging or pooling channels.",
        "",
        "## Day-block robustness",
        "",
        f"{len(day_candidates)} channel/window rows survive after reducing all episodes on the same UTC day to one median paired difference.",
        "",
        day_summary.head(12).to_string(index=False) if not day_summary.empty else "No usable day blocks.",
    ]
    report = output_dir / "DIRECT_NOAA_RAE_BURST_SEARCH.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-csv", type=Path, required=True)
    parser.add_argument("--noaa-events", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunksize", type=int, default=500_000)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = pd.read_csv(args.noaa_events)
    episodes = cluster_low_frequency_reports(reports)
    centers = build_centers(episodes).sort_values("center_time").reset_index(drop=True)
    centers["center_id"] = np.arange(len(centers), dtype=int)
    buffers = extract_raw_windows(args.master_csv, centers, chunksize=args.chunksize)
    metrics = calculate_center_metrics(centers, buffers)
    paired = build_paired_differences(metrics)
    summary = summarize_channels(paired)
    daily, day_summary = summarize_day_blocks(
        paired,
        delta_column="paired_delta_top3_z",
        key_columns=["frequency_band", "frequency_mhz", "antenna", "search_window"],
        seed_base=20270000,
    )

    episodes.to_csv(args.output_dir / "noaa_low_frequency_episodes.csv", index=False)
    centers.to_csv(args.output_dir / "real_and_control_centers.csv", index=False)
    metrics.to_csv(args.output_dir / "channel_event_metrics.csv", index=False)
    paired.to_csv(args.output_dir / "paired_real_control_differences.csv", index=False)
    summary.to_csv(args.output_dir / "channel_timing_association_summary.csv", index=False)
    daily.to_csv(args.output_dir / "daily_channel_differences.csv", index=False)
    day_summary.to_csv(args.output_dir / "channel_day_block_summary.csv", index=False)
    report = write_report(args.output_dir, args.master_csv, episodes, summary, day_summary)
    manifest = {
        "master_csv": str(args.master_csv),
        "noaa_events": str(args.noaa_events),
        "n_low_frequency_episodes": int(len(episodes)),
        "frequency_channels_combined": False,
        "antennas_combined": False,
        "search_windows_seconds": SEARCH_WINDOWS_S,
        "control_shifts_hours": CONTROL_SHIFTS_H,
        "report": report.name,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
