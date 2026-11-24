#!/usr/bin/env python
"""Test RAE-2 burst-receiver channels at NOAA 18/29 MHz report times."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_noaa_direct_rae_burst_search import (
    benjamini_hochberg,
    cluster_low_frequency_reports,
    datetime_ns,
    sign_flip_pvalue,
    summarize_day_blocks,
)


WINDOWS_S = {
    "same_burst_30m": (-600.0, 1200.0),
    "activity_70m": (-600.0, 3600.0),
}
CONTROL_SHIFTS_H = (-6.0, -3.0, 3.0, 6.0)


def parse_vector(value: object) -> np.ndarray:
    text = str(value).strip().replace("\n", " ")
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    return np.fromstring(text, sep=" ", dtype=float)


def load_burst_receiver(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    columns = ["Datetime", "Energies", "T_MAX_1", "T_MAX_2", "T_MODE_1", "T_MODE_2"]
    data = pd.read_csv(path, usecols=columns, low_memory=False)
    data["Datetime"] = pd.to_datetime(data["Datetime"], errors="coerce")
    data = data.dropna(subset=["Datetime"]).sort_values("Datetime").drop_duplicates("Datetime")
    energies = parse_vector(data.iloc[0]["Energies"])
    arrays: dict[str, np.ndarray] = {}
    for column in ("T_MAX_1", "T_MAX_2", "T_MODE_1", "T_MODE_2"):
        parsed = [parse_vector(value) for value in data[column]]
        if any(len(values) != len(energies) for values in parsed):
            raise ValueError(f"inconsistent vector length in {column}")
        arrays[column] = np.vstack(parsed)
    result = pd.DataFrame({"time": data["Datetime"].to_numpy()})
    for receiver in (1, 2):
        maximum = arrays[f"T_MAX_{receiver}"]
        mode = arrays[f"T_MODE_{receiver}"]
        excess = maximum - mode
        excess[(maximum <= 0.0) | (mode <= 0.0) | (excess < 0.0)] = np.nan
        result[f"excess_{receiver}"] = list(excess)
    return result.reset_index(drop=True), energies


def build_metrics(
    receiver: pd.DataFrame,
    energies_khz: np.ndarray,
    episodes: pd.DataFrame,
) -> pd.DataFrame:
    times = pd.to_datetime(receiver["time"])
    time_ns = datetime_ns(times)
    excess = {
        index: np.vstack(receiver[f"excess_{index}"].to_numpy())
        for index in (1, 2)
    }
    gate_ns = datetime_ns(episodes["gate_time"])
    rows: list[dict[str, object]] = []
    for episode in episodes.itertuples(index=False):
        for shift_h in (0.0, *CONTROL_SHIFTS_H):
            center = pd.Timestamp(episode.gate_time) + pd.to_timedelta(shift_h, unit="h")
            is_real = np.isclose(shift_h, 0.0)
            nearest_gate_s = (
                float(np.min(np.abs(gate_ns - center.value)) / 1e9) if gate_ns.size else np.inf
            )
            quiet = bool(is_real or nearest_gate_s > 5400.0)
            for window_name, (window_lo, window_hi) in WINDOWS_S.items():
                lo = int(np.searchsorted(time_ns, center.value + int(window_lo * 1e9), side="left"))
                hi = int(np.searchsorted(time_ns, center.value + int(window_hi * 1e9), side="right"))
                if hi <= lo:
                    continue
                for receiver_index in (1, 2):
                    local = excess[receiver_index][lo:hi]
                    valid = np.isfinite(local)
                    for channel_index, energy_khz in enumerate(energies_khz):
                        values = local[:, channel_index]
                        finite = values[np.isfinite(values)]
                        if finite.size == 0:
                            continue
                        rows.append(
                            {
                                "episode_id": int(episode.episode_id),
                                "gate_time": pd.Timestamp(episode.gate_time),
                                "center_time": center,
                                "gate_role": "real" if is_real else "time_shift_control",
                                "control_shift_h": float(shift_h),
                                "valid_quiet_control": quiet,
                                "receiver": f"burst_receiver_{receiver_index}",
                                "frequency_khz": float(energy_khz),
                                "frequency_mhz": float(energy_khz / 1000.0),
                                "search_window": window_name,
                                "n_valid_bins": int(finite.size),
                                "peak_max_minus_mode": float(np.max(finite)),
                                "median_max_minus_mode": float(np.median(finite)),
                            }
                        )
    return pd.DataFrame.from_records(rows)


def summarize(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    paired_rows: list[dict[str, object]] = []
    keys = ["receiver", "frequency_khz", "frequency_mhz", "search_window"]
    for key, channel in metrics.groupby(keys, observed=True, sort=True):
        for episode_id, episode in channel.groupby("episode_id", observed=True, sort=True):
            real = episode[episode["gate_role"].eq("real")]
            controls = episode[
                episode["gate_role"].eq("time_shift_control")
                & episode["valid_quiet_control"]
            ]
            if real.empty or len(controls) < 2:
                continue
            real_value = float(real.iloc[0]["peak_max_minus_mode"])
            control_values = controls["peak_max_minus_mode"].to_numpy(dtype=float)
            paired_rows.append(
                {
                    **dict(zip(keys, key)),
                    "episode_id": int(episode_id),
                    "gate_time": real.iloc[0]["gate_time"],
                    "real_peak_excess": real_value,
                    "control_median_peak_excess": float(np.median(control_values)),
                    "paired_delta_excess": float(real_value - np.median(control_values)),
                    "real_exceeds_all_controls": bool(real_value > np.max(control_values)),
                    "n_valid_controls": int(len(control_values)),
                }
            )
    paired = pd.DataFrame.from_records(paired_rows)
    summary_rows: list[dict[str, object]] = []
    for index, (key, channel) in enumerate(
        paired.groupby(keys, observed=True, sort=True), start=1
    ):
        delta = channel["paired_delta_excess"].to_numpy(dtype=float)
        summary_rows.append(
            {
                **dict(zip(keys, key)),
                "n_paired_episodes": int(len(channel)),
                "median_real_peak_excess": float(channel["real_peak_excess"].median()),
                "median_control_peak_excess": float(
                    channel["control_median_peak_excess"].median()
                ),
                "median_paired_delta_excess": float(np.median(delta)),
                "mean_paired_delta_excess": float(np.mean(delta)),
                "positive_paired_fraction": float(np.mean(delta > 0.0)),
                "real_exceeds_all_controls_fraction": float(
                    channel["real_exceeds_all_controls"].mean()
                ),
                "one_sided_sign_flip_p": sign_flip_pvalue(delta, seed=19750000 + index),
            }
        )
    summary = pd.DataFrame.from_records(summary_rows)
    if not summary.empty:
        summary["bh_q_across_channel_local_tests"] = benjamini_hochberg(
            summary["one_sided_sign_flip_p"]
        )
        summary["evidence_class"] = np.where(
            (summary["n_paired_episodes"] >= 15)
            & (summary["median_paired_delta_excess"] > 0.0)
            & (summary["positive_paired_fraction"] >= 0.60)
            & (summary["bh_q_across_channel_local_tests"] <= 0.10),
            "timestamp_associated_burst_candidate",
            "not_detected",
        )
        summary = summary.sort_values(
            ["bh_q_across_channel_local_tests", "one_sided_sign_flip_p", "frequency_mhz"]
        ).reset_index(drop=True)
    return paired, summary


def write_report(
    output_dir: Path,
    receiver_path: Path,
    episodes: pd.DataFrame,
    summary: pd.DataFrame,
    day_summary: pd.DataFrame,
) -> Path:
    candidates = summary[summary["evidence_class"].eq("timestamp_associated_burst_candidate")]
    day_candidates = day_summary[
        day_summary["day_block_evidence_class"].eq("day_block_timestamp_candidate")
    ]
    focus = summary[
        np.isclose(summary["frequency_mhz"], 0.70)
        | np.isclose(summary["frequency_mhz"], 4.70)
    ]
    lines = [
        "# NOAA-Timed RAE-2 Burst-Receiver Search",
        "",
        "NOAA 18/29 MHz reports were used only as timestamps. Each of the 32 burst-receiver frequencies and two receiver outputs was tested independently.",
        "",
        f"- Burst-receiver table: `{receiver_path}`",
        f"- Usable low-frequency NOAA episodes in receiver coverage: {len(episodes)}",
        "- Statistic: within-bin maximum minus modal telemetry level.",
        "- Controls: the same channel at ±3 and ±6 hours, excluding controls near another low-frequency report.",
        "",
        "## Result",
        "",
        (
            f"{len(candidates)} channel/window rows meet the exploratory corrected rule."
            if not candidates.empty
            else "No individual channel/window row meets the exploratory rule after multiplicity correction."
        ),
        "",
        "## 0.70 and 4.70 MHz",
        "",
        focus.to_string(index=False) if not focus.empty else "No usable rows.",
        "",
        "## Strongest individual channel rows",
        "",
        summary.head(16).to_string(index=False) if not summary.empty else "No usable rows.",
        "",
        "No frequencies or receivers are averaged. The BH column is a multiplicity correction over separately reported tests, not a composite signal.",
        "",
        "## Day-block robustness",
        "",
        f"{len(day_candidates)} channel/window rows survive after collapsing same-day NOAA episodes.",
        "",
        day_summary.head(16).to_string(index=False) if not day_summary.empty else "No usable day blocks.",
    ]
    report = output_dir / "NOAA_TIMED_BURST_RECEIVER_SEARCH.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--burst-receiver", type=Path, required=True)
    parser.add_argument("--noaa-events", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    receiver, energies = load_burst_receiver(args.burst_receiver)
    reports = pd.read_csv(args.noaa_events)
    episodes = cluster_low_frequency_reports(reports)
    coverage_start = pd.Timestamp(receiver["time"].min())
    coverage_end = pd.Timestamp(receiver["time"].max())
    episodes = episodes[episodes["gate_time"].between(coverage_start, coverage_end)].copy()
    metrics = build_metrics(receiver, energies, episodes)
    paired, summary = summarize(metrics)
    daily, day_summary = summarize_day_blocks(
        paired,
        delta_column="paired_delta_excess",
        key_columns=["receiver", "frequency_khz", "frequency_mhz", "search_window"],
        seed_base=19760000,
    )

    episodes.to_csv(args.output_dir / "noaa_episodes_in_receiver_coverage.csv", index=False)
    metrics.to_csv(args.output_dir / "burst_receiver_event_metrics.csv", index=False)
    paired.to_csv(args.output_dir / "paired_real_control_differences.csv", index=False)
    summary.to_csv(args.output_dir / "channel_timing_association_summary.csv", index=False)
    daily.to_csv(args.output_dir / "daily_channel_differences.csv", index=False)
    day_summary.to_csv(args.output_dir / "channel_day_block_summary.csv", index=False)
    report = write_report(args.output_dir, args.burst_receiver, episodes, summary, day_summary)
    manifest = {
        "burst_receiver": str(args.burst_receiver),
        "coverage_start": str(coverage_start),
        "coverage_end": str(coverage_end),
        "n_noaa_episodes_in_coverage": int(len(episodes)),
        "n_frequency_channels": int(len(energies)),
        "frequency_channels_combined": False,
        "receivers_combined": False,
        "report": report.name,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
