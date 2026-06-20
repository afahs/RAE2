#!/usr/bin/env python
"""Quantify horizontal banding in normalized source-control grids."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.util import ensure_dir, software_versions, write_json  # noqa: E402
from scripts.build_normalized_ecliptic_control_visuals import _normalize_samples  # noqa: E402
from scripts.build_raw_ecliptic_control_visuals import (  # noqa: E402
    ANTENNA,
    ANTENNA_LABEL,
    FREQS_MHZ,
    FREQ_TO_BAND,
    GROUP_COLORS,
    GROUP_ORDER,
    _all_events,
    _collect_raw_samples,
    _load_clean_subset,
)


def _rounded(values: pd.Series, step: float) -> pd.Series:
    return (pd.to_numeric(values, errors="coerce") / float(step)).round() * float(step)


def _top_bins(values: pd.Series, step: float, n: int = 5) -> str:
    bins = _rounded(values, step).dropna()
    if bins.empty:
        return ""
    counts = bins.value_counts().head(n)
    total = float(len(bins))
    return "; ".join(f"{idx:.3g}:{count / total:.3f}" for idx, count in counts.items())


def _sample_time_ns(samples: pd.DataFrame) -> pd.Series:
    predicted = pd.to_datetime(samples["predicted_event_time"], errors="coerce").astype("int64")
    dt_ns = (pd.to_numeric(samples["t_rel_sec"], errors="coerce") * 1e9).round().astype("Int64")
    out = predicted + dt_ns.astype("int64")
    return pd.Series(out, index=samples.index)


def _banding_summary(samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    work = samples.copy()
    work["sample_time_ns"] = _sample_time_ns(work)
    by = ["plot_group", "frequency_mhz", "event_type"]
    for keys, grp in work.groupby(by, sort=True):
        group, freq, event_type = keys
        n = len(grp)
        if n == 0:
            continue
        z05 = _rounded(grp["normalized_power"], 0.05)
        z10 = _rounded(grp["normalized_power"], 0.10)
        raw = pd.to_numeric(grp["raw_power"], errors="coerce")
        scale = pd.to_numeric(grp["normalization_scale_raw_power"], errors="coerce")
        sample_time_counts = grp["sample_time_ns"].value_counts()
        rows.append(
            {
                "plot_group": group,
                "frequency_mhz": float(freq),
                "event_type": event_type,
                "n_samples": int(n),
                "n_events": int(grp["event_uid"].nunique()),
                "n_tracks": int(grp["control_name_for_plot"].nunique()),
                "normalized_unique_fraction_exact": float(grp["normalized_power"].nunique(dropna=True) / n),
                "normalized_top_0p05_bin_fraction": float(z05.value_counts().iloc[0] / n) if not z05.empty else np.nan,
                "normalized_top_0p10_bin_fraction": float(z10.value_counts().iloc[0] / n) if not z10.empty else np.nan,
                "normalized_bins_0p05_for_50pct": int(_bins_for_fraction(z05, 0.50)),
                "top_normalized_0p05_bins": _top_bins(grp["normalized_power"], 0.05),
                "raw_power_unique_fraction_exact": float(raw.nunique(dropna=True) / n),
                "raw_power_top_exact_fraction": float(raw.value_counts().iloc[0] / n) if not raw.dropna().empty else np.nan,
                "top_raw_power_values": _top_exact(raw),
                "normalization_scale_unique_fraction_exact": float(scale.nunique(dropna=True) / max(grp["event_uid"].nunique(), 1)),
                "top_normalization_scales": _top_exact(scale, n=3),
                "sample_time_unique_fraction": float(grp["sample_time_ns"].nunique(dropna=True) / n),
                "max_same_sample_time_reuse": int(sample_time_counts.iloc[0]) if not sample_time_counts.empty else 0,
            }
        )
    return pd.DataFrame(rows)


def _bins_for_fraction(values: pd.Series, fraction: float) -> int:
    counts = values.dropna().value_counts()
    if counts.empty:
        return 0
    target = float(counts.sum()) * float(fraction)
    return int(np.searchsorted(np.cumsum(counts.to_numpy()), target, side="left") + 1)


def _top_exact(values: pd.Series, n: int = 5) -> str:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty:
        return ""
    counts = vals.value_counts().head(n)
    total = float(len(vals))
    return "; ".join(f"{idx:.6g}:{count / total:.3f}" for idx, count in counts.items())


def _plot_banding(summary: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    for group in GROUP_ORDER:
        sub = summary[summary["plot_group"].eq(group)]
        if sub.empty:
            continue
        g = sub.groupby("frequency_mhz", as_index=False).agg(
            normalized_top=("normalized_top_0p05_bin_fraction", "median"),
            raw_top=("raw_power_top_exact_fraction", "median"),
        )
        color = GROUP_COLORS.get(group, "black")
        axes[0].plot(g["frequency_mhz"], g["normalized_top"], marker="o", lw=1.5, color=color, label=group)
        axes[1].plot(g["frequency_mhz"], g["raw_top"], marker="o", lw=1.5, color=color, label=group)
    axes[0].set_ylabel("median top normalized-bin fraction")
    axes[1].set_ylabel("median top exact raw-value fraction")
    for ax in axes:
        ax.set_xlabel("frequency (MHz)")
        ax.grid(alpha=0.22)
        ax.set_xticks(FREQS_MHZ)
        ax.set_xticklabels([f"{f:.2f}" for f in FREQS_MHZ], rotation=45)
    axes[0].set_title("Horizontal banding after local normalization")
    axes[1].set_title("Raw telemetry value repetition")
    axes[1].legend(frameon=False, fontsize=7, loc="best")
    fig.suptitle(f"Banding audit, {ANTENNA_LABEL}")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    path = out_dir / "normalized_power_banding_audit.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _write_report(out_dir: Path, summary: pd.DataFrame, plot_path: Path, args: argparse.Namespace) -> Path:
    display = summary.sort_values(["plot_group", "frequency_mhz", "event_type"])
    worst = summary.sort_values("normalized_top_0p05_bin_fraction", ascending=False).head(20)
    lines = [
        "# Normalized Power Banding Audit",
        "",
        "Purpose: quantify whether the apparent horizontal lines in the normalized source-control grids are real repeated values.",
        "",
        "Definitions:",
        "",
        "- `normalized_top_0p05_bin_fraction`: fraction of samples in the most populated normalized-power bin after rounding to 0.05.",
        "- `raw_power_top_exact_fraction`: fraction of samples with the single most common exact raw telemetry power value.",
        "- `sample_time_unique_fraction`: fraction of plotted samples that correspond to unique underlying telemetry timestamps.",
        "- `max_same_sample_time_reuse`: maximum number of plotted points sharing the same underlying telemetry timestamp in that group/frequency/event type.",
        "",
        "A strong horizontal-line artifact would usually show a large normalized top-bin fraction and/or large exact raw-value repetition.",
        "",
        "## Main Conclusion",
        "",
        "The fixed ecliptic, near-ecliptic, and off-ecliptic controls do show measurable banding after normalization, but the diagnostic separates three effects:",
        "",
        "1. The normalization is median/MAD-like. Samples equal to the event-window sideband median map to normalized power 0.",
        "2. Samples about one MAD from the median map to about +/-0.674 after robust-sigma scaling, so common bands appear near 0 and +/-0.65.",
        "3. Ryle-Vonberg coarse powers are quantized/repeated, which makes those median/MAD ladder values more visually obvious.",
        "",
        "The effect is not unique to the ecliptic controls; Earth, Sun, and Fornax A also show it.",
        "It should be treated as a normalization/display feature, not by itself as evidence for or against occultation.",
        "The median trend lines are more interpretable than the individual scatter bands for these plots.",
        "",
        "## Worst Normalized-Banding Cases",
        "",
        worst[
            [
                "plot_group",
                "frequency_mhz",
                "event_type",
                "n_samples",
                "normalized_top_0p05_bin_fraction",
                "raw_power_top_exact_fraction",
                "sample_time_unique_fraction",
                "max_same_sample_time_reuse",
                "top_normalized_0p05_bins",
            ]
        ].to_string(index=False),
        "",
        "## Full Summary",
        "",
        display.to_string(index=False),
        "",
        "## Event-Bin Meaning",
        "",
        "In the source-control grids, time is measured relative to the predicted event time: `dt = sample_time - predicted_event_time`.",
        "For the median curve, samples are grouped into 60-second `dt` bins, such as -900 s, -840 s, ..., 0 s, ..., +900 s.",
        "`events/bin` means the number of distinct occultation events that contributed at least one valid sample to a particular 60-second relative-time bin.",
        "Because telemetry coverage is gappy, not every event contributes samples to every relative-time bin.",
        "The label `max N events/bin` is the busiest such bin, not the total event count.",
        "",
        f"Generated plot: `{plot_path}`",
        "",
        "## Run Configuration",
        "",
        f"- antenna: {ANTENNA} ({ANTENNA_LABEL})",
        f"- window_s: {args.window_s}",
        f"- inner_s: {args.inner_s}",
        f"- max_events_per_group: {args.max_events_per_group} (0 means all events)",
        "",
    ]
    path = out_dir / "normalized_power_banding_audit.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(ROOT / "outputs/normalized_power_banding_audit_v1"))
    parser.add_argument("--window-s", type=float, default=900.0)
    parser.add_argument("--inner-s", type=float, default=15.0)
    parser.add_argument("--max-events-per-group", type=int, default=0, help="0 or negative means use all available events.")
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    write_json(
        out_dir / "run_config.json",
        {
            "antenna": ANTENNA,
            "antenna_label": ANTENNA_LABEL,
            "frequencies_mhz": FREQS_MHZ,
            "window_s": float(args.window_s),
            "inner_s": float(args.inner_s),
            "max_events_per_group": int(args.max_events_per_group),
            "software_versions": software_versions(),
        },
    )

    bands = {FREQ_TO_BAND[f] for f in FREQS_MHZ}
    clean = _load_clean_subset(bands)
    events = _all_events(bands)
    raw_samples = _collect_raw_samples(clean, events, args.window_s, args.max_events_per_group)
    samples = _normalize_samples(raw_samples, args.inner_s)
    summary = _banding_summary(samples)
    summary.to_csv(out_dir / "normalized_power_banding_summary.csv", index=False)
    plot_path = _plot_banding(summary, out_dir)
    report = _write_report(out_dir, summary, plot_path, args)
    print(report)
    print(plot_path)


if __name__ == "__main__":
    main()
