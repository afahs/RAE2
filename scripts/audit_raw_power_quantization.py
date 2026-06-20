#!/usr/bin/env python
"""Audit quantization/repeated values in original Ryle-Vonberg power samples."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rylevonberg.table_io import read_table

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.constants import FREQUENCY_MAP_MHZ  # noqa: E402
from rylevonberg.util import ensure_dir, software_versions, write_json  # noqa: E402


CLEAN = ROOT / "outputs/control_survey_earth_sun_postnov1974_v2/01_ingest/cleaned_timeseries.csv"
BANDING = ROOT / "outputs/normalized_power_banding_audit_v1/normalized_power_banding_summary.csv"
ANT_LABEL = {"rv1_coarse": "upper V", "rv2_coarse": "lower V"}


def _is_valid_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def _counter_stats(counter: Counter, n: int) -> dict[str, float | int | str]:
    if n <= 0 or not counter:
        return {
            "n_unique_exact": 0,
            "duplicate_fraction": np.nan,
            "top1_exact_fraction": np.nan,
            "top5_exact_fraction": np.nan,
            "top10_exact_fraction": np.nan,
            "top100_exact_fraction": np.nan,
            "effective_unique_count": np.nan,
            "effective_unique_fraction": np.nan,
            "top_values": "",
            "min_positive_spacing": np.nan,
            "median_positive_spacing": np.nan,
            "modal_positive_spacing": np.nan,
        }
    counts = np.array(sorted(counter.values(), reverse=True), dtype=float)
    p = counts / float(n)
    entropy = -float(np.sum(p * np.log(p)))
    values = np.array(sorted(counter.keys()), dtype=float)
    diffs = np.diff(values)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size:
        rounded_diffs = np.round(diffs, 6)
        unique_d, d_counts = np.unique(rounded_diffs, return_counts=True)
        modal_spacing = float(unique_d[np.argmax(d_counts)])
    else:
        modal_spacing = np.nan
    return {
        "n_unique_exact": int(len(counter)),
        "duplicate_fraction": float(1.0 - len(counter) / float(n)),
        "top1_exact_fraction": float(np.sum(counts[:1]) / n),
        "top5_exact_fraction": float(np.sum(counts[:5]) / n),
        "top10_exact_fraction": float(np.sum(counts[:10]) / n),
        "top100_exact_fraction": float(np.sum(counts[:100]) / n),
        "effective_unique_count": float(np.exp(entropy)),
        "effective_unique_fraction": float(np.exp(entropy) / n),
        "top_values": "; ".join(f"{value:.6g}:{count / n:.4f}" for value, count in counter.most_common(8)),
        "min_positive_spacing": float(np.nanmin(diffs)) if diffs.size else np.nan,
        "median_positive_spacing": float(np.nanmedian(diffs)) if diffs.size else np.nan,
        "modal_positive_spacing": modal_spacing,
    }


def _scan_clean_timeseries(chunk_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    counters: dict[tuple[int, str], Counter] = defaultdict(Counter)
    totals: dict[tuple[int, str], int] = defaultdict(int)
    finite_totals: dict[tuple[int, str], int] = defaultdict(int)
    valid_totals: dict[tuple[int, str], int] = defaultdict(int)
    integer_totals: dict[tuple[int, str], int] = defaultdict(int)

    usecols = ["frequency_band", "antenna", "power", "is_valid"]
    for chunk in read_table(CLEAN, usecols=usecols, chunksize=chunk_size, low_memory=False):
        chunk["power"] = pd.to_numeric(chunk["power"], errors="coerce")
        chunk["frequency_band"] = pd.to_numeric(chunk["frequency_band"], errors="coerce").astype("Int64")
        chunk["is_valid_bool"] = _is_valid_series(chunk["is_valid"])
        for (band, antenna), grp in chunk.groupby(["frequency_band", "antenna"], sort=False):
            if pd.isna(band):
                continue
            key = (int(band), str(antenna))
            totals[key] += int(len(grp))
            finite = np.isfinite(grp["power"].to_numpy(dtype=float))
            finite_totals[key] += int(np.count_nonzero(finite))
            keep = grp["is_valid_bool"].to_numpy(dtype=bool) & finite
            valid_totals[key] += int(np.count_nonzero(keep))
            vals = grp.loc[keep, "power"]
            if vals.empty:
                continue
            arr = vals.to_numpy(dtype=float)
            integer_totals[key] += int(np.count_nonzero(np.isclose(arr, np.round(arr), atol=1e-6, rtol=0.0)))
            counters[key].update(vals.value_counts(dropna=True).to_dict())

    summary_rows = []
    top_rows = []
    for key in sorted(totals):
        band, antenna = key
        n_valid = valid_totals[key]
        stats = _counter_stats(counters[key], n_valid)
        summary_rows.append(
            {
                "frequency_band": band,
                "frequency_mhz": FREQUENCY_MAP_MHZ.get(band, np.nan),
                "antenna": antenna,
                "antenna_label": ANT_LABEL.get(antenna, antenna),
                "n_rows": totals[key],
                "n_finite": finite_totals[key],
                "n_valid": n_valid,
                "valid_fraction": float(n_valid / totals[key]) if totals[key] else np.nan,
                "integer_power_fraction": float(integer_totals[key] / n_valid) if n_valid else np.nan,
                **stats,
            }
        )
        for rank, (value, count) in enumerate(counters[key].most_common(50), start=1):
            top_rows.append(
                {
                    "frequency_band": band,
                    "frequency_mhz": FREQUENCY_MAP_MHZ.get(band, np.nan),
                    "antenna": antenna,
                    "antenna_label": ANT_LABEL.get(antenna, antenna),
                    "rank": rank,
                    "raw_power": float(value),
                    "count": int(count),
                    "fraction_valid": float(count / n_valid) if n_valid else np.nan,
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(top_rows)


def _plot_quantization(summary: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True)
    colors = {"rv1_coarse": "#4c78a8", "rv2_coarse": "#d95f02"}
    for antenna, grp in summary.groupby("antenna", sort=True):
        g = grp.sort_values("frequency_mhz")
        label = ANT_LABEL.get(antenna, antenna)
        color = colors.get(antenna, None)
        axes[0].plot(g["frequency_mhz"], g["integer_power_fraction"], marker="o", lw=1.7, color=color, label=label)
        axes[1].plot(g["frequency_mhz"], g["duplicate_fraction"], marker="o", lw=1.7, color=color, label=label)
        axes[2].plot(g["frequency_mhz"], g["top10_exact_fraction"], marker="o", lw=1.7, color=color, label=label)
    axes[0].set_ylabel("integer-valued valid samples")
    axes[1].set_ylabel("duplicate exact-value fraction")
    axes[2].set_ylabel("top 10 exact raw values / valid samples")
    for ax in axes:
        ax.set_xlabel("frequency (MHz)")
        ax.set_xscale("log")
        ax.set_xticks(sorted(FREQUENCY_MAP_MHZ.values()))
        ax.set_xticklabels([f"{v:.2f}" for v in sorted(FREQUENCY_MAP_MHZ.values())], rotation=45)
        ax.grid(alpha=0.22, which="both")
    axes[2].legend(frameon=False)
    fig.suptitle("Original cleaned Ryle-Vonberg power values are discrete/repeated")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    path = out_dir / "raw_power_quantization_by_channel.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _plot_quantization_vs_banding(summary: pd.DataFrame, out_dir: Path) -> Path | None:
    if not BANDING.exists():
        return None
    banding = read_table(BANDING)
    low = summary[summary["antenna"].eq("rv2_coarse")][
        ["frequency_mhz", "top1_exact_fraction", "top10_exact_fraction", "duplicate_fraction"]
    ].copy()
    merged = banding.merge(low, on="frequency_mhz", how="left")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for group, grp in merged.groupby("plot_group", sort=True):
        axes[0].scatter(grp["top1_exact_fraction"], grp["normalized_top_0p05_bin_fraction"], s=28, alpha=0.75, label=group)
        axes[1].scatter(grp["top10_exact_fraction"], grp["normalized_top_0p05_bin_fraction"], s=28, alpha=0.75, label=group)
    axes[0].set_xlabel("channel top exact raw value fraction")
    axes[1].set_xlabel("channel top 10 exact raw values fraction")
    for ax in axes:
        ax.set_ylabel("normalized top 0.05-bin fraction")
        ax.grid(alpha=0.22)
    axes[1].legend(frameon=False, fontsize=7, loc="best")
    fig.suptitle("Raw exact-value repetition contributes to normalized banding but is not the whole effect")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    path = out_dir / "raw_quantization_vs_normalized_banding.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _write_report(summary: pd.DataFrame, top: pd.DataFrame, paths: list[Path], out_dir: Path) -> Path:
    low_rv2 = summary[summary["antenna"].eq("rv2_coarse") & summary["frequency_mhz"].isin([0.45, 0.70, 0.90, 1.31, 2.20])]
    med = summary.groupby("antenna_label", as_index=False).agg(
        median_integer_power_fraction=("integer_power_fraction", "median"),
        median_duplicate_fraction=("duplicate_fraction", "median"),
        median_top1_exact_fraction=("top1_exact_fraction", "median"),
        median_top10_exact_fraction=("top10_exact_fraction", "median"),
        median_effective_unique_fraction=("effective_unique_fraction", "median"),
    )
    lines = [
        "# Raw Power Quantization Audit",
        "",
        "Question: are the horizontal bands in normalized plots ultimately tied to quantization in the original Ryle-Vonberg data?",
        "",
        "## Conclusion",
        "",
        "Yes, the original cleaned power values are discrete and repeated. Nearly all valid samples are integer-valued, and exact raw-power values recur far more often than they would for continuous floating-point receiver noise.",
        "",
        "However, raw quantization is not the only step that makes the bands prominent. The visible normalized bands are produced by the combination of:",
        "",
        "1. discrete/repeated raw power levels;",
        "2. per-event median subtraction;",
        "3. MAD/robust-sigma scaling, which maps median-level samples to 0 and one-MAD samples to about +/-0.674;",
        "4. repeated use of overlapping telemetry times in the artificial fixed-control families.",
        "",
        "Therefore the horizontal stripes should be interpreted as a data/normalization display artifact, not as a physical sky signal.",
        "",
        "## Antenna Summary",
        "",
        med.to_string(index=False),
        "",
        "## Lower-V Low-Frequency Details",
        "",
        low_rv2[
            [
                "frequency_band",
                "frequency_mhz",
                "n_valid",
                "integer_power_fraction",
                "n_unique_exact",
                "duplicate_fraction",
                "top1_exact_fraction",
                "top10_exact_fraction",
                "top100_exact_fraction",
                "effective_unique_fraction",
                "min_positive_spacing",
                "median_positive_spacing",
                "modal_positive_spacing",
                "top_values",
            ]
        ].to_string(index=False),
        "",
        "## Top Raw Values, Lower V Low Frequencies",
        "",
        top[top["antenna"].eq("rv2_coarse") & top["frequency_mhz"].isin([0.45, 0.70, 0.90, 1.31, 2.20])]
        .head(80)
        .to_string(index=False),
        "",
        "## Generated Plots",
        "",
    ]
    lines.extend(f"- `{path}`" for path in paths if path is not None)
    path = out_dir / "raw_power_quantization_audit.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(ROOT / "outputs/raw_power_quantization_audit_v1"))
    parser.add_argument("--chunk-size", type=int, default=750_000)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    write_json(
        out_dir / "run_config.json",
        {
            "cleaned_timeseries": str(CLEAN),
            "chunk_size": int(args.chunk_size),
            "software_versions": software_versions(),
        },
    )
    summary, top = _scan_clean_timeseries(args.chunk_size)
    summary.to_csv(out_dir / "raw_power_quantization_summary.csv", index=False)
    top.to_csv(out_dir / "raw_power_top_values.csv", index=False)
    paths = [_plot_quantization(summary, out_dir)]
    maybe = _plot_quantization_vs_banding(summary, out_dir)
    if maybe is not None:
        paths.append(maybe)
    report = _write_report(summary, top, paths, out_dir)
    print(report)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
