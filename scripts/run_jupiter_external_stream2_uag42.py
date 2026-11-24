#!/usr/bin/env python
"""Stream 2: test RAE-2 bursts in independently reported UAG-42 windows.

Warwick, Dulk, and Riddle (1975, UAG-42) tabulated dated high-frequency
Jupiter activity intervals.  This stream treats those times as an external,
phase-free selector and asks whether RAE-2's two highest bands are unusually
bursty during the reported activity.  Same-duration, nearby shifted windows
provide the controls.
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

from scripts.jupiter_external_stream_common import (
    COLORS,
    DETECTORS,
    LABELS,
    detector_source,
    familywise_upper_p,
    upper_tail_p,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVENTS = Path(
    "/pscratch/sd/a/afahs/RAE2/jupiter_burst_gated_phase_occurrence_v1/"
    "five_minute_channel_percentile_events.parquet"
)
DEFAULT_WINDOWS = ROOT / "configs/jupiter_warwick_dulk_riddle_1975_active_windows.csv"
DEFAULT_OUTPUT = Path("/pscratch/sd/a/afahs/RAE2/jupiter_external_stream2_uag42_v1")

PRIMARY_BANDS = (8, 9)  # 6.55 and 9.18 MHz: closest RAE-2 bands to UAG-42.
PRIMARY_QUANTILE = 0.99
CONTROL_SHIFTS_DAYS = (-7, -3, -2, -1, 1, 2, 3, 7)


def parse_hhmm(date: pd.Timestamp, value: object) -> pd.Timestamp:
    """Parse UAG-style HHMM, including values greater than 2400."""

    text = str(value).strip().replace(".", "")
    if not text or text.lower() == "nan":
        raise ValueError(f"missing HHMM value for {date}")
    sign = -1 if text.startswith("-") else 1
    text = text.lstrip("+-").zfill(4)
    hour = int(text[:-2])
    minute = int(text[-2:])
    if minute >= 60:
        raise ValueError(f"malformed HHMM minute in {value!r}")
    delta = pd.Timedelta(hours=hour, minutes=minute)
    return date + sign * delta


def parse_frequency_range(value: object) -> tuple[float, float]:
    text = str(value).strip()
    try:
        left, right = (float(part) for part in text.split("-", 1))
    except (ValueError, TypeError):
        return np.nan, np.nan
    return min(left, right), max(left, right)


def load_windows(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype=str)
    rows: list[dict[str, object]] = []
    for index, row in raw.iterrows():
        date = pd.Timestamp(row["date"])
        start = parse_hhmm(date, row["event_start_hhmm"])
        end = parse_hhmm(date, row["event_end_hhmm"])
        if end <= start:
            end += pd.Timedelta(days=1)
        low, high = parse_frequency_range(row.get("reported_freq_range_mhz", ""))
        rows.append(
            {
                "historical_window_id": f"w{index + 1:03d}",
                "start": start,
                "end": end,
                "duration_min": (end - start).total_seconds() / 60.0,
                "intensity": int(row["intensity"]),
                "burstiness": str(row["burstiness"]).strip().upper(),
                "reported_freq_low_mhz": low,
                "reported_freq_high_mhz": high,
                "reported_range_includes_9p18": bool(low <= 9.18 <= high),
            }
        )
    return pd.DataFrame(rows)


def union_mask(times: np.ndarray, windows: pd.DataFrame) -> np.ndarray:
    result = np.zeros(len(times), dtype=bool)
    for row in windows.itertuples(index=False):
        result |= (times >= np.datetime64(row.start)) & (times <= np.datetime64(row.end))
    return result


def score_interval(
    times: np.ndarray,
    candidate_cells: np.ndarray,
    start: pd.Timestamp,
    end: pd.Timestamp,
    forbidden: np.ndarray | None = None,
) -> tuple[float, int, int]:
    keep = (times >= np.datetime64(start)) & (times <= np.datetime64(end))
    if forbidden is not None:
        keep &= ~forbidden
    n_samples = int(keep.sum())
    if n_samples == 0:
        return np.nan, 0, 0
    cells = candidate_cells[keep]
    return float(np.mean(cells)), n_samples, int(cells.size)


def label_permutation_null(
    score_sets: list[np.ndarray],
    n_permutations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Exchange the active label among active plus matched control windows."""

    null = np.empty(int(n_permutations), dtype=float)
    for permutation in range(int(n_permutations)):
        contrasts = []
        for scores in score_sets:
            chosen = int(rng.integers(0, len(scores)))
            rest = np.delete(scores, chosen)
            contrasts.append(float(scores[chosen] - np.median(rest)))
        null[permutation] = float(np.mean(contrasts))
    return null


def analyze_detector(
    events: pd.DataFrame,
    windows: pd.DataFrame,
    detector: str,
    n_permutations: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, object], np.ndarray]:
    source = detector_source(events, detector, require_geometry=False)
    times = pd.to_datetime(source["time"]).to_numpy(dtype="datetime64[ns]")
    candidate_cells = np.column_stack(
        [source[f"band_{band}_percentile"].to_numpy(dtype=float) >= PRIMARY_QUANTILE for band in PRIMARY_BANDS]
    )
    active_union = union_mask(times, windows)

    rows: list[dict[str, object]] = []
    score_sets: list[np.ndarray] = []
    for window in windows.itertuples(index=False):
        active_score, active_samples, active_cells = score_interval(
            times, candidate_cells, window.start, window.end
        )
        controls: list[float] = []
        control_samples = 0
        for shift_days in CONTROL_SHIFTS_DAYS:
            shift = pd.Timedelta(days=int(shift_days))
            score, n_samples, _ = score_interval(
                times,
                candidate_cells,
                window.start + shift,
                window.end + shift,
                forbidden=active_union,
            )
            if np.isfinite(score):
                controls.append(score)
                control_samples += n_samples
        usable = np.isfinite(active_score) and len(controls) >= 2
        control_median = float(np.median(controls)) if controls else np.nan
        contrast = float(active_score - control_median) if usable else np.nan
        overlap_active_score = np.nan
        overlap_control_median = np.nan
        overlap_contrast = np.nan
        if window.reported_range_includes_9p18:
            overlap_active_score, _, _ = score_interval(
                times, candidate_cells[:, [1]], window.start, window.end
            )
            overlap_controls = []
            for shift_days in CONTROL_SHIFTS_DAYS:
                shift = pd.Timedelta(days=int(shift_days))
                score, n_samples, _ = score_interval(
                    times,
                    candidate_cells[:, [1]],
                    window.start + shift,
                    window.end + shift,
                    forbidden=active_union,
                )
                if n_samples and np.isfinite(score):
                    overlap_controls.append(score)
            if np.isfinite(overlap_active_score) and len(overlap_controls) >= 2:
                overlap_control_median = float(np.median(overlap_controls))
                overlap_contrast = float(overlap_active_score - overlap_control_median)
        if usable:
            score_sets.append(np.asarray([active_score, *controls], dtype=float))
        rows.append(
            {
                "detector": detector,
                "historical_window_id": window.historical_window_id,
                "start": window.start,
                "end": window.end,
                "duration_min": window.duration_min,
                "intensity": window.intensity,
                "burstiness": window.burstiness,
                "reported_range_includes_9p18": window.reported_range_includes_9p18,
                "active_score": active_score,
                "control_median_score": control_median,
                "paired_difference": contrast,
                "n_active_samples": active_samples,
                "n_active_band_time_cells": active_cells,
                "n_available_controls": len(controls),
                "n_control_samples": control_samples,
                "usable": usable,
                "direct_9p18_active_score": overlap_active_score,
                "direct_9p18_control_median_score": overlap_control_median,
                "direct_9p18_paired_difference": overlap_contrast,
            }
        )

    detail = pd.DataFrame(rows)
    usable = detail[detail["usable"]].copy()
    observed = float(usable["paired_difference"].mean())
    rng = np.random.default_rng(int(seed))
    null = label_permutation_null(score_sets, n_permutations, rng)

    # Direct-overlap diagnostic: 9.18 MHz only, and only catalog ranges that
    # explicitly include it.  It is intentionally secondary because n is small.
    overlap = usable[
        usable["reported_range_includes_9p18"]
        & usable["direct_9p18_paired_difference"].notna()
    ]
    overlap_difference = (
        float(overlap["direct_9p18_paired_difference"].mean()) if len(overlap) else np.nan
    )
    bootstrap = np.empty(int(n_permutations), dtype=float)
    paired = usable["paired_difference"].to_numpy(dtype=float)
    for iteration in range(int(n_permutations)):
        bootstrap[iteration] = float(np.mean(rng.choice(paired, size=len(paired), replace=True)))
    summary: dict[str, object] = {
        "detector": detector,
        "n_valid": int(len(source)),
        "n_catalog_windows": int(len(windows)),
        "n_usable_windows": int(len(usable)),
        "active_mean_candidate_fraction": float(usable["active_score"].mean()),
        "control_mean_candidate_fraction": float(usable["control_median_score"].mean()),
        "primary_paired_difference": observed,
        "primary_bootstrap_low": float(np.quantile(bootstrap, 0.025)),
        "primary_bootstrap_high": float(np.quantile(bootstrap, 0.975)),
        "primary_local_p": upper_tail_p(observed, null),
        "n_direct_9p18_windows": int(len(overlap)),
        "direct_9p18_paired_difference": overlap_difference,
    }
    return detail, summary, null


def plot_summary(summary: pd.DataFrame, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    x = np.arange(len(summary))
    observed = summary["primary_paired_difference"].to_numpy(dtype=float)
    low = summary["primary_bootstrap_low"].to_numpy(dtype=float)
    high = summary["primary_bootstrap_high"].to_numpy(dtype=float)
    colors = [COLORS[value] for value in summary["detector"]]
    ax.bar(x, observed, color=colors, alpha=0.84, width=0.64)
    ax.errorbar(x, observed, yerr=np.vstack([observed - low, high - observed]), fmt="none", color="0.15", capsize=4)
    ax.axhline(0.0, color="0.25", lw=0.9)
    ax.set_xticks(x, [LABELS[value] for value in summary["detector"]])
    ax.set_ylabel("active - nearby-control\nburst-cell fraction")
    ax.set_title("Stream 2 - externally dated UAG-42 activity windows")
    ax.grid(True, axis="y", color="0.90", lw=0.5)
    fig.tight_layout()
    path = output / "stream2_uag42_paired_effects.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def plot_windows(detail: pd.DataFrame, output: Path) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(10.2, 8.0), sharex=True)
    for ax, detector in zip(axes, DETECTORS):
        sub = detail[detail["detector"].eq(detector) & detail["usable"]].sort_values("start")
        ax.scatter(sub["start"], sub["paired_difference"], s=18, color=COLORS[detector], alpha=0.78)
        ax.axhline(0.0, color="0.35", lw=0.8)
        ax.set_ylabel("active -\ncontrol")
        ax.set_title(LABELS[detector], loc="left", fontsize=10)
        ax.grid(True, color="0.92", lw=0.45)
    axes[-1].set_xlabel("UAG-42 reported activity date")
    fig.suptitle("Window-level RAE-2 top-1% burst contrasts")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = output / "stream2_uag42_window_differences.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def write_report(output: Path, summary: pd.DataFrame, files: list[Path], config: dict[str, object]) -> Path:
    columns = [
        "detector",
        "n_usable_windows",
        "active_mean_candidate_fraction",
        "control_mean_candidate_fraction",
        "primary_paired_difference",
        "primary_local_p",
        "primary_fwer_p",
        "n_direct_9p18_windows",
    ]
    lines = [
        "# Stream 2 - UAG-42 dated activity prior",
        "",
        "The primary statistic is the mean across catalog windows of the top-1% burst-cell fraction in RAE-2 bands 8 and 9 (6.55 and 9.18 MHz), minus the median of same-duration controls shifted by 1, 2, 3, or 7 days. Controls are not allowed to reuse any catalog-active exposure.",
        "",
        "## Results",
        "",
        summary[columns].to_string(index=False),
        "",
        "## Controls and interpretation",
        "",
        f"- {config['n_permutations']} label exchanges among each active window and its available shifted controls.",
        "- FWER p-values use the maximum statistic across RV1, RV2, and receiver coincidence.",
        "- UAG-42 is mostly above RAE-2's 9.18 MHz ceiling, so this is a cross-frequency simultaneity test, not a same-frequency replication.",
        "- The direct 9.18 MHz overlap subset is reported only as a low-power diagnostic.",
        "",
        "## Files",
        "",
        *[f"- `{path.name}`" for path in files],
    ]
    report = output / "STREAM2_UAG42_REPORT.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--windows", type=Path, default=DEFAULT_WINDOWS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-permutations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=260825201)
    args = parser.parse_args()

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    events = pd.read_parquet(args.events)
    windows = load_windows(args.windows)
    config = {
        "events": str(args.events),
        "windows": str(args.windows),
        "primary_bands": list(PRIMARY_BANDS),
        "primary_quantile": PRIMARY_QUANTILE,
        "control_shifts_days": list(CONTROL_SHIFTS_DAYS),
        "n_permutations": int(args.n_permutations),
        "seed": int(args.seed),
    }
    (output / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    details = []
    summaries = []
    nulls = []
    for detector_index, detector in enumerate(DETECTORS):
        detail, summary, null = analyze_detector(
            events,
            windows,
            detector,
            int(args.n_permutations),
            int(args.seed) + detector_index,
        )
        details.append(detail)
        summaries.append(summary)
        nulls.append(null)
    detail_table = pd.concat(details, ignore_index=True)
    summary_table = pd.DataFrame(summaries)
    summary_table["primary_fwer_p"] = familywise_upper_p(
        summary_table["primary_paired_difference"].to_numpy(dtype=float),
        np.vstack(nulls),
    )
    detail_table.to_csv(output / "stream2_uag42_window_statistics.csv", index=False)
    summary_table.to_csv(output / "stream2_uag42_summary.csv", index=False)
    windows.to_csv(output / "stream2_uag42_windows_parsed.csv", index=False)
    files = [
        plot_summary(summary_table, output),
        plot_windows(detail_table, output),
        output / "stream2_uag42_summary.csv",
        output / "stream2_uag42_window_statistics.csv",
    ]
    report = write_report(output, summary_table, files, config)
    print(report)


if __name__ == "__main__":
    main()
