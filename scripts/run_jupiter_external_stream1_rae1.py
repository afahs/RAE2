#!/usr/bin/env python
"""Stream 1: test RAE-2 bursts against fixed RAE-1 Io/intensity priors.

Desch & Carr (1978) found the strongest RAE-1 activity at 6.55, 4.70,
3.93, and 2.20 MHz near Io phases 90 and 240 degrees, while System-III
control was marginal.  They also argued that Io control becomes stronger as
the flux-density threshold is raised.  This stream therefore fixes the Io
windows and threshold ladder from that paper before looking at RAE-2.
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
    daily_roll_plan,
    detector_source,
    familywise_upper_p,
    log_rate_ratio,
    rolled_vectors,
    upper_tail_p,
)


DEFAULT_EVENTS = Path(
    "/pscratch/sd/a/afahs/RAE2/jupiter_burst_gated_phase_occurrence_v1/"
    "five_minute_channel_percentile_events.parquet"
)
DEFAULT_OUTPUT = Path("/pscratch/sd/a/afahs/RAE2/jupiter_external_stream1_rae1_v1")

TAIL_QUANTILES = (0.95, 0.975, 0.99, 0.995)
PRIMARY_BANDS = (5, 6, 7, 8)
SECONDARY_BANDS = (4,)
FREQUENCY_MHZ = {4: 1.31, 5: 2.20, 6: 3.93, 7: 4.70, 8: 6.55}
IO_WINDOWS = ((70.0, 110.0), (210.0, 270.0))


def in_circular_windows(values: np.ndarray, windows: tuple[tuple[float, float], ...]) -> np.ndarray:
    values = np.asarray(values, dtype=float) % 360.0
    result = np.zeros(len(values), dtype=bool)
    for low, high in windows:
        if low <= high:
            result |= (values >= low) & (values < high)
        else:
            result |= (values >= low) | (values < high)
    return result


def slope(values: np.ndarray) -> np.ndarray:
    """Slope against increasing burst extremeness."""

    values = np.asarray(values, dtype=float)
    x = -np.log10(1.0 - np.asarray(TAIL_QUANTILES))
    xc = x - x.mean()
    denominator = float(np.sum(xc**2))
    if values.ndim == 1:
        return np.asarray(np.sum((values - np.nanmean(values)) * xc) / denominator)
    return np.nansum((values - np.nanmean(values, axis=1, keepdims=True)) * xc, axis=1) / denominator


def analyze_detector(
    events: pd.DataFrame,
    detector: str,
    n_permutations: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame, np.ndarray, np.ndarray]:
    source = detector_source(events, detector, require_geometry=True)
    phase = source["io_phase_deg"].to_numpy(dtype=float)
    target = in_circular_windows(phase, IO_WINDOWS)
    rng = np.random.default_rng(int(seed))
    groups, offsets = daily_roll_plan(source, n_permutations, rng)

    band_rows: list[dict[str, object]] = []
    combined_observed = np.empty(len(TAIL_QUANTILES), dtype=float)
    combined_null = np.empty((int(n_permutations), len(TAIL_QUANTILES)), dtype=float)

    for quantile_index, quantile in enumerate(TAIL_QUANTILES):
        primary_observed = []
        primary_null = []
        for band in (*SECONDARY_BANDS, *PRIMARY_BANDS):
            percentile = source[f"band_{band}_percentile"].to_numpy(dtype=float)
            gate = np.asarray(percentile >= float(quantile), dtype=np.int8)
            rolled = rolled_vectors(gate, groups, offsets)
            observed = float(log_rate_ratio(gate, target))
            null = log_rate_ratio(rolled, target)
            band_rows.append(
                {
                    "detector": detector,
                    "frequency_band": int(band),
                    "frequency_mhz": FREQUENCY_MHZ[int(band)],
                    "tail_quantile": float(quantile),
                    "n_valid": int(len(source)),
                    "n_candidates": int(gate.sum()),
                    "n_io_window_exposure": int(target.sum()),
                    "log_rate_ratio_io_windows": observed,
                    "rate_ratio_io_windows": float(np.exp(observed)),
                    "local_p": upper_tail_p(observed, null),
                    "rae1_primary_band": bool(band in PRIMARY_BANDS),
                }
            )
            if band in PRIMARY_BANDS:
                primary_observed.append(observed)
                primary_null.append(null)
        combined_observed[quantile_index] = float(np.mean(primary_observed))
        combined_null[:, quantile_index] = np.mean(np.vstack(primary_null), axis=0)

    primary_index = TAIL_QUANTILES.index(0.995)
    observed_primary = float(combined_observed[primary_index])
    null_primary = combined_null[:, primary_index]
    observed_slope = float(slope(combined_observed))
    null_slope = slope(combined_null)
    summary: dict[str, object] = {
        "detector": detector,
        "n_valid": int(len(source)),
        "n_io_window_exposure": int(target.sum()),
        "primary_tail_quantile": 0.995,
        "primary_bands_mhz": ",".join(f"{FREQUENCY_MHZ[b]:.2f}" for b in PRIMARY_BANDS),
        "primary_mean_log_rate_ratio": observed_primary,
        "primary_mean_rate_ratio": float(np.exp(observed_primary)),
        "primary_local_p": upper_tail_p(observed_primary, null_primary),
        "intensity_trend_slope": observed_slope,
        "intensity_trend_local_p": upper_tail_p(observed_slope, null_slope),
    }

    ladder = pd.DataFrame(
        {
            "detector": detector,
            "tail_quantile": TAIL_QUANTILES,
            "tail_fraction_percent": [100.0 * (1.0 - q) for q in TAIL_QUANTILES],
            "mean_log_rate_ratio": combined_observed,
            "mean_rate_ratio": np.exp(combined_observed),
            "null_low_log_rate_ratio": np.quantile(combined_null, 0.025, axis=0),
            "null_high_log_rate_ratio": np.quantile(combined_null, 0.975, axis=0),
        }
    )

    phase_centers = np.arange(10.0, 360.0, 20.0)
    phase_bins = (np.floor(phase / 20.0).astype(int) % 18)
    exposure = np.bincount(phase_bins, minlength=18).astype(float)
    band_gates = np.vstack(
        [
            source[f"band_{band}_percentile"].to_numpy(dtype=float) >= 0.995
            for band in PRIMARY_BANDS
        ]
    ).mean(axis=0)
    counts = np.bincount(phase_bins, weights=band_gates, minlength=18)
    phase_curve = pd.DataFrame(
        {
            "detector": detector,
            "io_phase_center_deg": phase_centers,
            "n_valid": exposure.astype(int),
            "mean_band_candidate_fraction": np.divide(
                counts,
                exposure,
                out=np.full(18, np.nan),
                where=exposure > 0,
            ),
            "in_rae1_io_window": in_circular_windows(phase_centers, IO_WINDOWS),
        }
    )
    return pd.DataFrame(band_rows), summary, ladder, phase_curve, combined_null


def plot_ladder(ladder: pd.DataFrame, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    for detector in DETECTORS:
        sub = ladder[ladder["detector"].eq(detector)].sort_values("tail_fraction_percent", ascending=False)
        ax.plot(
            sub["tail_fraction_percent"],
            sub["mean_log_rate_ratio"],
            marker="o",
            lw=1.7,
            color=COLORS[detector],
            label=LABELS[detector],
        )
        ax.fill_between(
            sub["tail_fraction_percent"],
            sub["null_low_log_rate_ratio"],
            sub["null_high_log_rate_ratio"],
            color=COLORS[detector],
            alpha=0.10,
            linewidth=0,
        )
    ax.axhline(0.0, color="0.35", lw=0.9)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xticks([5.0, 2.5, 1.0, 0.5])
    ax.set_xticklabels(["top 5%", "top 2.5%", "top 1%", "top 0.5%"])
    ax.set_xlabel("RAE-2 per-band daily burst gate")
    ax.set_ylabel("mean log rate ratio in RAE-1 Io windows")
    ax.set_title("Stream 1 - RAE-1 frequency/intensity prior")
    ax.grid(True, color="0.90", lw=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = output / "stream1_rae1_intensity_ladder.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def plot_phase_curves(curves: pd.DataFrame, output: Path) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(9.2, 8.2), sharex=True)
    for ax, detector in zip(axes, DETECTORS):
        sub = curves[curves["detector"].eq(detector)].sort_values("io_phase_center_deg")
        ax.plot(
            sub["io_phase_center_deg"],
            sub["mean_band_candidate_fraction"],
            marker="o",
            lw=1.5,
            color=COLORS[detector],
        )
        for low, high in IO_WINDOWS:
            ax.axvspan(low, high, color=COLORS[detector], alpha=0.08, lw=0)
        ax.set_ylabel("top-0.5%\nfraction")
        ax.set_title(LABELS[detector], loc="left", fontsize=10)
        ax.grid(True, color="0.92", lw=0.45)
    axes[-1].set_xlabel("Io phase (deg; paper convention)")
    axes[-1].set_xlim(0, 360)
    axes[-1].set_xticks(np.arange(0, 361, 45))
    fig.suptitle("RAE-2 burst occurrence in the fixed RAE-1 Io-phase frame")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = output / "stream1_rae1_io_phase_curves.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def write_report(output: Path, summary: pd.DataFrame, files: list[Path], config: dict[str, object]) -> Path:
    columns = [
        "detector",
        "n_valid",
        "primary_mean_rate_ratio",
        "primary_local_p",
        "primary_fwer_p",
        "intensity_trend_slope",
        "intensity_trend_local_p",
        "intensity_trend_fwer_p",
    ]
    lines = [
        "# Stream 1 - RAE-1 frequency/intensity prior",
        "",
        "The primary test was fixed from Desch & Carr (1978): the exact-match 2.20, 3.93, 4.70, and 6.55 MHz bands, the top 0.5% daily per-band powers, and Io windows 70-110 plus 210-270 degrees. The statistic is the mean bandwise rate ratio inside versus outside those windows.",
        "",
        "The intensity-trend test asks whether Io-window enrichment increases as the gate tightens from the top 5% to the top 0.5%.",
        "",
        "## Results",
        "",
        summary[columns].to_string(index=False),
        "",
        "## Controls",
        "",
        f"- {config['n_permutations']} within-day circular rolls preserve burst clustering, daily normalization, and cross-band timing.",
        "- FWER p-values use the maximum statistic across RV1, RV2, and receiver coincidence.",
        "- The 1.31 MHz result is retained as a secondary band because RAE-1 identification was less reliable there.",
        "- System III was not included in this stream because the source paper found only marginal rotation-phase control below 8 MHz.",
        "",
        "## Files",
        "",
        *[f"- `{path.name}`" for path in files],
    ]
    report = output / "STREAM1_RAE1_REPORT.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-permutations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=260825101)
    args = parser.parse_args()

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "events": str(args.events),
        "n_permutations": int(args.n_permutations),
        "seed": int(args.seed),
        "tail_quantiles": list(TAIL_QUANTILES),
        "primary_bands": list(PRIMARY_BANDS),
        "io_windows_deg": [list(window) for window in IO_WINDOWS],
    }
    (output / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    events = pd.read_parquet(args.events)

    bands = []
    summaries = []
    ladders = []
    curves = []
    primary_nulls = []
    slope_nulls = []
    for detector_index, detector in enumerate(DETECTORS):
        band, summary, ladder, curve, combined_null = analyze_detector(
            events,
            detector,
            int(args.n_permutations),
            int(args.seed) + detector_index,
        )
        bands.append(band)
        summaries.append(summary)
        ladders.append(ladder)
        curves.append(curve)
        primary_nulls.append(combined_null[:, TAIL_QUANTILES.index(0.995)])
        slope_nulls.append(slope(combined_null))

    band_table = pd.concat(bands, ignore_index=True)
    summary_table = pd.DataFrame(summaries)
    ladder_table = pd.concat(ladders, ignore_index=True)
    curve_table = pd.concat(curves, ignore_index=True)
    summary_table["primary_fwer_p"] = familywise_upper_p(
        summary_table["primary_mean_log_rate_ratio"].to_numpy(dtype=float),
        np.vstack(primary_nulls),
    )
    summary_table["intensity_trend_fwer_p"] = familywise_upper_p(
        summary_table["intensity_trend_slope"].to_numpy(dtype=float),
        np.vstack(slope_nulls),
    )

    band_table.to_csv(output / "stream1_rae1_band_statistics.csv", index=False)
    summary_table.to_csv(output / "stream1_rae1_summary.csv", index=False)
    ladder_table.to_csv(output / "stream1_rae1_intensity_ladder.csv", index=False)
    curve_table.to_csv(output / "stream1_rae1_io_phase_curves.csv", index=False)
    files = [
        plot_ladder(ladder_table, output),
        plot_phase_curves(curve_table, output),
        output / "stream1_rae1_summary.csv",
        output / "stream1_rae1_band_statistics.csv",
    ]
    report = write_report(output, summary_table, files, config)
    print(report)


if __name__ == "__main__":
    main()
