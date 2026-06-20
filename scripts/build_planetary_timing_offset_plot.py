#!/usr/bin/env python
"""Build a slide-ready timing-offset plot for the planetary survey."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rylevonberg.table_io import read_table


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "outputs/planetary_confirmation_survey/planetary_confirmation_summary.csv"
OUT = ROOT / "outputs/slide_assets/planetary_timing_offsets"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = read_table(SUMMARY)
    df["source_label"] = df["source_name"].astype(str).str.title()
    df["antenna_label"] = df["target_antenna"].map({"rv2_coarse": "lower V", "rv1_coarse": "upper V"}).fillna(df["target_antenna"])
    df["abs_best_snr"] = pd.to_numeric(df["best_confirmed_snr"], errors="coerce").abs()
    df["timing_offset_s"] = pd.to_numeric(df["best_timing_offset_s"], errors="coerce")
    df["strict_quality_snr"] = pd.to_numeric(df["strict_quality_snr"], errors="coerce")
    order = ["mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune"]
    df["planet_order"] = df["source_name"].map({name: idx for idx, name in enumerate(order)})
    df = df.sort_values("planet_order").reset_index(drop=True)

    out_csv = OUT / "planetary_timing_offsets.csv"
    df[
        [
            "source_name",
            "target_frequency_mhz",
            "antenna_label",
            "target_window_s",
            "best_timing_offset_s",
            "best_confirmed_snr",
            "strict_quality_snr",
            "max_wrong_control_abs_snr",
            "min_leave_one_month_abs_snr",
            "status",
            "decision_reason",
        ]
    ].to_csv(out_csv, index=False)
    _plot(df)
    _write_readme(out_csv)


def _plot(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    y = np.arange(len(df))
    colors = np.where(df["status"].astype(str).eq("repeatable_candidate"), "#1b9e77", "#7570b3")
    ax.hlines(y, 0, df["timing_offset_s"], color="0.70", lw=1.6, zorder=1)
    sizes = 45 + 14 * np.clip(df["abs_best_snr"], 0, 20)
    ax.scatter(df["timing_offset_s"], y, s=sizes, c=colors, edgecolor="black", linewidth=0.7, zorder=3)

    for _, row in df.iterrows():
        label = f"{row['target_frequency_mhz']:.2g} MHz, {row['antenna_label']}"
        x = float(row["timing_offset_s"])
        ha = "left" if x >= 0 else "right"
        dx = 18 if x >= 0 else -18
        ax.text(x + dx, row.name, label, ha=ha, va="center", fontsize=8, color="0.20")

    ax.axvspan(-60, 60, color="#1b9e77", alpha=0.10, label="good timing: |dt| <= 60 s")
    ax.axvline(0, color="black", lw=1.0)
    ax.axvline(-120, color="0.45", lw=0.9, ls="--")
    ax.axvline(120, color="0.45", lw=0.9, ls="--", label="caution boundary: |dt| = 120 s")
    ax.set_yticks(y)
    ax.set_yticklabels(df["source_label"])
    ax.invert_yaxis()
    ax.set_xlim(-650, 170)
    ax.set_xlabel("Best timing offset from predicted occultation time, dt (s)")
    ax.set_title("Planetary survey timing offsets")
    ax.grid(axis="x", color="0.88", lw=0.8)
    ax.legend(loc="lower left", fontsize=8, frameon=False)

    caption = "Marker area scales with |best stack SNR|; green marks Earth positive-control status."
    fig.text(0.01, 0.01, caption, fontsize=8, color="0.30")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "planetary_timing_offsets.png", dpi=180)
    plt.close(fig)


def _write_readme(out_csv: Path) -> None:
    md = [
        "# Planetary Timing Offsets",
        "",
        "This plot summarizes the best timing offset for each planet in the planetary confirmation survey.",
        "A source-like occultation should peak near the predicted lunar-limb crossing time. The green band marks |dt| <= 60 s, the current good-timing region; dashed lines mark |dt| = 120 s.",
        "",
        f"- Plot: `{OUT / 'planetary_timing_offsets.png'}`",
        f"- CSV: `{out_csv}`",
        "",
        "Marker area scales with the absolute value of the best stack SNR. The SNR itself is not control-corrected; timing, strict-quality survival, wrong controls, and leave-one-month tests are separate diagnostics.",
    ]
    (OUT / "README.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
