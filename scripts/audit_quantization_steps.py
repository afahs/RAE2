#!/usr/bin/env python
"""Estimate raw Ryle-Vonberg quantization steps from value differences and occupancy."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from math import gcd
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
ANT_LABEL = {"rv1_coarse": "upper V", "rv2_coarse": "lower V"}
ANT_COLOR = {"rv1_coarse": "#4c78a8", "rv2_coarse": "#d95f02"}


def _is_valid_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def _scan(
    clean_csv: Path,
    chunk_size: int,
    integer_tolerance: float,
    diff_clip: int,
    max_step: int,
    residue_min_fraction: float,
    acf_max_lag: int,
) -> tuple[dict[tuple[int, str], Counter], pd.DataFrame, dict[tuple[int, str], tuple[np.ndarray, np.ndarray]]]:
    diff_counts: dict[tuple[int, str], Counter] = {}
    acfs: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
    usecols = ["frequency_band", "antenna", "power", "is_valid"]

    rows = []
    for band in sorted(FREQUENCY_MAP_MHZ):
        for antenna in ["rv1_coarse", "rv2_coarse"]:
            key = (int(band), antenna)
            value_counter: Counter = Counter()
            diff_counter: Counter = Counter()
            total = 0
            integer_total = 0
            diff_total = 0
            diff_integer_total = 0
            diff_clipped_total = 0
            last_value: float | None = None
            for chunk in read_table(clean_csv, usecols=usecols, chunksize=chunk_size, low_memory=False):
                band_values = pd.to_numeric(chunk["frequency_band"], errors="coerce")
                mask = band_values.eq(band) & chunk["antenna"].eq(antenna)
                if not mask.any():
                    continue
                sub = chunk.loc[mask, ["power", "is_valid"]].copy()
                sub["power"] = pd.to_numeric(sub["power"], errors="coerce")
                valid = _is_valid_series(sub["is_valid"]).to_numpy(dtype=bool)
                finite = np.isfinite(sub["power"].to_numpy(dtype=float))
                values = sub.loc[valid & finite, "power"].to_numpy(dtype=float)
                if values.size == 0:
                    continue
                rounded = np.rint(values).astype(np.int64)
                integer_like = np.abs(values - rounded) <= integer_tolerance
                total += int(values.size)
                integer_total += int(np.count_nonzero(integer_like))
                if np.any(integer_like):
                    unique_values, unique_counts = np.unique(rounded[integer_like], return_counts=True)
                    value_counter.update({int(v): int(c) for v, c in zip(unique_values, unique_counts)})

                if last_value is not None:
                    values_for_diff = np.concatenate([[last_value], values])
                else:
                    values_for_diff = values
                last_value = float(values[-1])
                if values_for_diff.size < 2:
                    continue
                diffs = np.diff(values_for_diff)
                diff_total += int(diffs.size)
                rounded_diffs = np.rint(diffs).astype(np.int64)
                diff_integer = np.abs(diffs - rounded_diffs) <= integer_tolerance
                diff_integer_total += int(np.count_nonzero(diff_integer))
                keep = diff_integer & (np.abs(rounded_diffs) <= diff_clip)
                diff_clipped_total += int(np.count_nonzero(keep))
                if np.any(keep):
                    unique_diffs, unique_counts = np.unique(rounded_diffs[keep], return_counts=True)
                    diff_counter.update({int(v): int(c) for v, c in zip(unique_diffs, unique_counts)})
            diff_counts[key] = diff_counter
            base_row = {
                "frequency_band": key[0],
                "frequency_mhz": FREQUENCY_MAP_MHZ.get(key[0], np.nan),
                "antenna": key[1],
                "antenna_label": ANT_LABEL.get(key[1], key[1]),
                "n_valid": total,
                "integer_power_fraction": integer_total / total if total else np.nan,
                "n_first_differences": diff_total,
                "integer_first_difference_fraction": diff_integer_total / diff_total if diff_total else np.nan,
                "n_clipped_integer_first_differences": diff_clipped_total,
            }
            values = np.array(sorted(value_counter), dtype=int)
            residue_step, residue_fraction, residue = _dominant_residue_step(values, max_step, residue_min_fraction)
            diff_gcd, diff_mod_step, diff_mod_fraction, top_diffs = _diff_gcd_and_mod(diff_counter, max_step)
            lags, acf, acf_peak_lag, acf_peak_value = _occupancy_acf(value_counter, acf_max_lag, 0.98)
            acfs[key] = (lags, acf)
            if base_row["integer_power_fraction"] < 0.5:
                preferred = np.nan
                confidence = "mixed_fractional"
            elif residue_step > 1 and residue_fraction >= residue_min_fraction:
                preferred = residue_step
                confidence = "strong_integer_grid"
            elif diff_mod_step > 1 and diff_mod_fraction >= 0.95:
                preferred = diff_mod_step
                confidence = "first_difference_grid"
            else:
                preferred = 1
                confidence = "unit_or_irregular_integer_grid"
            rows.append(
                {
                    **base_row,
                    "n_occupied_integer_levels": int(values.size),
                    "occupancy_residue_step_raw_units": residue_step,
                    "occupancy_residue_fraction": residue_fraction,
                    "occupancy_residue": residue,
                    "first_difference_top50_gcd_raw_units": diff_gcd,
                    "first_difference_mod_step_raw_units": diff_mod_step,
                    "first_difference_mod_fraction": diff_mod_fraction,
                    "occupancy_acf_peak_lag_raw_units": acf_peak_lag,
                    "occupancy_acf_peak_value": acf_peak_value,
                    "estimated_quant_step_raw_units": preferred,
                    "step_confidence": confidence,
                    "top_integer_first_differences": top_diffs,
                }
            )
            print(f"scanned {ANT_LABEL.get(antenna, antenna)} {FREQUENCY_MAP_MHZ.get(band, np.nan):.2f} MHz", flush=True)
    return diff_counts, pd.DataFrame(rows), acfs


def _dominant_residue_step(values: np.ndarray, max_step: int, min_fraction: float) -> tuple[int, float, int]:
    if values.size == 0:
        return 0, np.nan, -1
    best_step = 1
    best_fraction = 1.0
    best_residue = 0
    # Prefer the largest step with a strong single-residue concentration.  Step=1
    # is always perfect, so it is only the fallback.
    for step in range(2, max_step + 1):
        residues = np.mod(values, step)
        counts = np.bincount(residues, minlength=step)
        idx = int(np.argmax(counts))
        frac = float(counts[idx] / values.size)
        if frac >= min_fraction and step > best_step:
            best_step = step
            best_fraction = frac
            best_residue = idx
    return best_step, best_fraction, best_residue


def _diff_gcd_and_mod(diff_counter: Counter, max_step: int) -> tuple[int, int, float, str]:
    if not diff_counter:
        return 0, 0, np.nan, ""
    nonzero = [(abs(int(k)), int(v)) for k, v in diff_counter.items() if int(k) != 0]
    if not nonzero:
        return 0, 0, np.nan, ""
    # Use the most common non-zero first differences so rare artifacts do not
    # force the gcd down to one.
    nonzero_sorted = sorted(nonzero, key=lambda kv: kv[1], reverse=True)
    top = nonzero_sorted[:50]
    g = 0
    for value, _ in top:
        g = value if g == 0 else gcd(g, value)
    total = sum(v for _, v in nonzero)
    best_step = 1
    best_fraction = 1.0
    for step in range(2, max_step + 1):
        good = sum(v for value, v in nonzero if value % step == 0)
        frac = good / total if total else np.nan
        if frac > best_fraction or (frac >= 0.95 and step > best_step):
            best_step = step
            best_fraction = frac
    top_text = "; ".join(f"{value}:{count}" for value, count in top[:10])
    return int(g), int(best_step), float(best_fraction), top_text


def _occupancy_acf(value_counter: Counter, max_lag: int, central_fraction: float) -> tuple[np.ndarray, np.ndarray, int, float]:
    if not value_counter:
        lags = np.arange(1, max_lag + 1)
        return lags, np.full_like(lags, np.nan, dtype=float), 0, np.nan
    values = np.array(sorted(value_counter), dtype=int)
    counts = np.array([value_counter[int(v)] for v in values], dtype=float)
    order = np.argsort(values)
    values = values[order]
    counts = counts[order]
    cdf = np.cumsum(counts) / np.sum(counts)
    lo = values[np.searchsorted(cdf, (1 - central_fraction) / 2, side="left")]
    hi = values[np.searchsorted(cdf, 1 - (1 - central_fraction) / 2, side="left")]
    keep = (values >= lo) & (values <= hi)
    values = values[keep]
    counts = counts[keep]
    if values.size < max_lag + 3:
        lags = np.arange(1, max_lag + 1)
        return lags, np.full_like(lags, np.nan, dtype=float), 0, np.nan
    count_map = {int(v): float(c) for v, c in zip(values, counts)}
    denom = float(np.sum(counts * counts))
    lags = np.arange(1, max_lag + 1)
    acf = np.full(max_lag, np.nan, dtype=float)
    if denom <= 0:
        return lags, acf, 0, np.nan
    for idx, lag in enumerate(lags):
        numerator = 0.0
        for value, count in count_map.items():
            numerator += count * count_map.get(value + int(lag), 0.0)
        acf[idx] = float(numerator / denom)
    if np.all(~np.isfinite(acf)):
        return lags, acf, 0, np.nan
    # The first local maximum is the cleanest occupancy-periodicity diagnostic;
    # if none exists, fall back to the global maximum over positive lags.
    finite = np.where(np.isfinite(acf))[0]
    best_idx = int(finite[np.nanargmax(acf[finite])])
    for i in finite:
        if i == 0 or i == len(acf) - 1:
            continue
        if acf[i] >= acf[i - 1] and acf[i] >= acf[i + 1]:
            best_idx = int(i)
            break
    return lags, acf, int(lags[best_idx]), float(acf[best_idx])


def _estimate_steps(
    value_counts: dict[tuple[int, str], Counter],
    diff_counts: dict[tuple[int, str], Counter],
    base: pd.DataFrame,
    max_step: int,
    residue_min_fraction: float,
    acf_max_lag: int,
) -> tuple[pd.DataFrame, dict[tuple[int, str], tuple[np.ndarray, np.ndarray]]]:
    rows = []
    acfs = {}
    for _, row in base.iterrows():
        key = (int(row["frequency_band"]), str(row["antenna"]))
        values = np.array(sorted(value_counts[key]), dtype=int)
        residue_step, residue_fraction, residue = _dominant_residue_step(values, max_step, residue_min_fraction)
        diff_gcd, diff_mod_step, diff_mod_fraction, top_diffs = _diff_gcd_and_mod(diff_counts[key], max_step)
        lags, acf, acf_peak_lag, acf_peak_value = _occupancy_acf(value_counts[key], acf_max_lag, 0.98)
        acfs[key] = (lags, acf)
        if row["integer_power_fraction"] < 0.5:
            preferred = np.nan
            confidence = "mixed_fractional"
        elif residue_step > 1 and residue_fraction >= residue_min_fraction:
            preferred = residue_step
            confidence = "strong_integer_grid"
        elif diff_mod_step > 1 and diff_mod_fraction >= 0.95:
            preferred = diff_mod_step
            confidence = "first_difference_grid"
        else:
            preferred = 1
            confidence = "unit_or_irregular_integer_grid"
        rows.append(
            {
                **row.to_dict(),
                "n_occupied_integer_levels": int(values.size),
                "occupancy_residue_step_raw_units": residue_step,
                "occupancy_residue_fraction": residue_fraction,
                "occupancy_residue": residue,
                "first_difference_top50_gcd_raw_units": diff_gcd,
                "first_difference_mod_step_raw_units": diff_mod_step,
                "first_difference_mod_fraction": diff_mod_fraction,
                "occupancy_acf_peak_lag_raw_units": acf_peak_lag,
                "occupancy_acf_peak_value": acf_peak_value,
                "estimated_quant_step_raw_units": preferred,
                "step_confidence": confidence,
                "top_integer_first_differences": top_diffs,
            }
        )
    return pd.DataFrame(rows), acfs


def _plot_first_differences(summary: pd.DataFrame, diff_counts: dict[tuple[int, str], Counter], out_dir: Path, antenna: str) -> Path:
    rows = summary[summary["antenna"].eq(antenna)].sort_values("frequency_mhz")
    n = len(rows)
    fig, axes = plt.subplots(2, int(np.ceil(n / 2)), figsize=(16, 7.5), sharey=False)
    axes = np.ravel(axes)
    for ax, (_, row) in zip(axes, rows.iterrows()):
        key = (int(row["frequency_band"]), antenna)
        counter = diff_counts[key]
        values = np.array(list(counter.keys()), dtype=int)
        counts = np.array(list(counter.values()), dtype=float)
        if values.size:
            order = np.argsort(values)
            values = values[order]
            counts = counts[order]
            nz = values != 0
            if np.any(nz):
                abs_values = np.abs(values[nz])
                abs_counts = counts[nz]
                order_abs = np.argsort(abs_values)
                abs_values = abs_values[order_abs]
                abs_counts = abs_counts[order_abs]
                cdf = np.cumsum(abs_counts) / np.sum(abs_counts)
                vmax = int(abs_values[np.searchsorted(cdf, 0.98, side="left")])
            else:
                vmax = 100
            vmax = max(25, min(2500, vmax))
            keep = (values >= -vmax) & (values <= vmax)
            ax.bar(values[keep], counts[keep], width=1.0, color=ANT_COLOR.get(antenna, "#555555"), alpha=0.85)
            ax.set_yscale("log")
            ax.set_xlim(-vmax, vmax)
        ax.axvline(0, color="black", lw=0.8, alpha=0.5)
        step = row["estimated_quant_step_raw_units"]
        if np.isfinite(step) and step > 1:
            ax.axvline(step, color="#b2182b", ls="--", lw=0.9)
            ax.axvline(-step, color="#b2182b", ls="--", lw=0.9)
        ax.set_title(f"{row['frequency_mhz']:.2f} MHz")
        ax.set_xlabel("first difference in raw power units")
        ax.grid(alpha=0.18)
    for ax in axes[n:]:
        ax.axis("off")
    axes[0].set_ylabel("count, log scale")
    fig.suptitle(f"First-difference histograms: {ANT_LABEL.get(antenna, antenna)}")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = out_dir / f"first_difference_histograms_{ANT_LABEL.get(antenna, antenna).lower().replace(' ', '_')}.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _plot_occupancy_acf(summary: pd.DataFrame, acfs: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]], out_dir: Path, antenna: str) -> Path:
    rows = summary[summary["antenna"].eq(antenna)].sort_values("frequency_mhz")
    n = len(rows)
    fig, axes = plt.subplots(2, int(np.ceil(n / 2)), figsize=(16, 7.5), sharey=True)
    axes = np.ravel(axes)
    for ax, (_, row) in zip(axes, rows.iterrows()):
        key = (int(row["frequency_band"]), antenna)
        lags, acf = acfs[key]
        ax.plot(lags, acf, color=ANT_COLOR.get(antenna, "#555555"), lw=1.5)
        step = row["estimated_quant_step_raw_units"]
        if np.isfinite(step) and step > 0:
            ax.axvline(step, color="#b2182b", ls="--", lw=0.9, label="estimated step")
        ax.axhline(0, color="black", lw=0.7, alpha=0.45)
        ax.set_title(f"{row['frequency_mhz']:.2f} MHz")
        ax.set_xlabel("lag in raw power units")
        ax.grid(alpha=0.2)
    for ax in axes[n:]:
        ax.axis("off")
    axes[0].set_ylabel("sparse occupancy autocorrelation")
    fig.suptitle(f"Autocorrelation of occupied raw-value levels: {ANT_LABEL.get(antenna, antenna)}")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = out_dir / f"value_occupancy_autocorrelation_{ANT_LABEL.get(antenna, antenna).lower().replace(' ', '_')}.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _write_report(summary: pd.DataFrame, paths: list[Path], out_dir: Path) -> Path:
    lower = summary[summary["antenna"].eq("rv2_coarse")].sort_values("frequency_mhz")
    upper = summary[summary["antenna"].eq("rv1_coarse")].sort_values("frequency_mhz")
    lines = [
        "# Quantization Step Audit",
        "",
        "This audit quantifies raw Ryle-Vonberg digitization using two direct diagnostics:",
        "",
        "1. histograms of adjacent first differences in each antenna/frequency time series;",
        "2. autocorrelation of the occupied raw-power value levels.",
        "",
        "The estimated step is in the original raw power units used by the cleaned time-series table. It is not a physical flux unit.",
        "",
        "## Main Interpretation",
        "",
        "The low-frequency channels are not continuous-valued. They occupy repeated integer or near-integer raw levels, and adjacent first differences often fall on a coarse grid. This supports the interpretation that horizontal bands in normalized control plots are partly inherited from raw telemetry/digitization quantization.",
        "",
        "The step is not identical across all bands. Some channels are dominated by an integer grid with an apparent multi-unit spacing, while others are mixed/fractional and should not be summarized by a single clean quantum.",
        "",
        "## Lower V Step Estimates",
        "",
        lower[
            [
                "frequency_mhz",
                "integer_power_fraction",
                "integer_first_difference_fraction",
                "occupancy_residue_step_raw_units",
                "occupancy_residue_fraction",
                "first_difference_top50_gcd_raw_units",
                "first_difference_mod_step_raw_units",
                "first_difference_mod_fraction",
                "occupancy_acf_peak_lag_raw_units",
                "estimated_quant_step_raw_units",
                "step_confidence",
            ]
        ].to_string(index=False),
        "",
        "## Upper V Step Estimates",
        "",
        upper[
            [
                "frequency_mhz",
                "integer_power_fraction",
                "integer_first_difference_fraction",
                "occupancy_residue_step_raw_units",
                "occupancy_residue_fraction",
                "first_difference_top50_gcd_raw_units",
                "first_difference_mod_step_raw_units",
                "first_difference_mod_fraction",
                "occupancy_acf_peak_lag_raw_units",
                "estimated_quant_step_raw_units",
                "step_confidence",
            ]
        ].to_string(index=False),
        "",
        "## Diagnostic Plots",
        "",
    ]
    for path in paths:
        lines.append(f"- {path}")
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- First differences are adjacent samples in file/time order for each antenna/frequency channel. Large astrophysical or instrumental changes broaden the histogram, so the central comb structure is more important than the full tails.",
            "- Occupancy autocorrelation is computed over integer-rounded raw levels. Channels with low integer-power fraction are labeled mixed/fractional because a single integer quantum is not a complete description.",
            "- A unit raw-power step can still produce visible normalized bands after median/MAD scaling, especially when many event windows share the same few telemetry levels.",
            "",
            "## Software Versions",
            "",
            "```",
            repr(software_versions()),
            "```",
        ]
    )
    path = out_dir / "quantization_step_audit.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-csv", type=Path, default=CLEAN)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/raw_power_quantization_step_audit_v1")
    parser.add_argument("--chunk-size", type=int, default=150_000)
    parser.add_argument("--integer-tolerance", type=float, default=1e-3)
    parser.add_argument("--diff-clip", type=int, default=5_000)
    parser.add_argument("--max-step", type=int, default=80)
    parser.add_argument("--residue-min-fraction", type=float, default=0.985)
    parser.add_argument("--acf-max-lag", type=int, default=120)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    write_json(
        out_dir / "run_config.json",
        {
            "clean_csv": str(args.clean_csv),
            "chunk_size": args.chunk_size,
            "integer_tolerance": args.integer_tolerance,
            "diff_clip": args.diff_clip,
            "max_step": args.max_step,
            "residue_min_fraction": args.residue_min_fraction,
            "acf_max_lag": args.acf_max_lag,
            "software_versions": software_versions(),
        },
    )
    diff_counts, summary, acfs = _scan(
        args.clean_csv,
        args.chunk_size,
        args.integer_tolerance,
        args.diff_clip,
        args.max_step,
        args.residue_min_fraction,
        args.acf_max_lag,
    )
    summary = summary.sort_values(["antenna", "frequency_mhz"])
    summary.to_csv(out_dir / "quantization_step_summary.csv", index=False)

    paths = [
        _plot_first_differences(summary, diff_counts, out_dir, "rv2_coarse"),
        _plot_first_differences(summary, diff_counts, out_dir, "rv1_coarse"),
        _plot_occupancy_acf(summary, acfs, out_dir, "rv2_coarse"),
        _plot_occupancy_acf(summary, acfs, out_dir, "rv1_coarse"),
    ]
    report = _write_report(summary, paths, out_dir)
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
