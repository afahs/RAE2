#!/usr/bin/env python
"""Plot point-source MIE all-frequency grids from sampling-model summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rylevonberg.table_io import read_table


ROOT = Path(__file__).resolve().parents[1]


def _plot_point_grid(summary: pd.DataFrame, source: str, out_dir: Path, window_s: float) -> Path:
    freqs = sorted(summary["frequency_mhz"].dropna().unique())
    fig, axes = plt.subplots(len(freqs), 2, figsize=(12, max(10, 1.25 * len(freqs))), sharex=True)
    if len(freqs) == 1:
        axes = np.asarray([axes])
    for i, freq in enumerate(freqs):
        for j, event_type in enumerate(["disappearance", "reappearance"]):
            ax = axes[i, j]
            sub = summary[
                np.isclose(summary["frequency_mhz"].astype(float), float(freq))
                & summary["event_type"].astype(str).eq(event_type)
            ].sort_values("t_bin_sec")
            if not sub.empty:
                ax.errorbar(
                    sub["t_bin_sec"] / 60.0,
                    sub["point_mie_median"],
                    yerr=sub.get("point_mie_err"),
                    marker="o",
                    markersize=2.2,
                    linewidth=1.05,
                    elinewidth=0.55,
                    capsize=1.0,
                    color="0.25",
                    ecolor="0.45",
                    label="point-source MIE sampled by RAE lower V" if i == 0 and j == 1 else None,
                )
            ax.axvline(0, color="black", linestyle="--", linewidth=0.75)
            ax.axhline(0, color="0.7", linewidth=0.65)
            ax.set_title(f"{freq:.2f} MHz {event_type}", fontsize=9)
            ax.grid(alpha=0.22)
            if j == 0:
                ax.set_ylabel("relative source contribution")
            if i == len(freqs) - 1:
                ax.set_xlabel("minutes from predicted event")
            if i == 0 and j == 1:
                ax.legend(frameon=False, fontsize=8)
    source_label = source.replace("_", " ").title()
    fig.suptitle(f"Point-source MIE model with actual lower-V sampling for {source_label} event times, +/-{window_s/60:.0f} min", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    path = out_dir / f"{source}_event_times_point_source_mie_all_frequency_grid_{int(window_s)}s.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["sun", "earth"], required=True)
    parser.add_argument("--summary", default="")
    parser.add_argument("--out-dir", default=str(ROOT / "outputs/point_source_mie_sampling_grids_10min_v1"))
    parser.add_argument("--window-s", type=float, default=600.0)
    args = parser.parse_args()

    source = str(args.source).lower()
    summary = (
        Path(args.summary)
        if args.summary
        else ROOT
        / f"outputs/{source}_sized_mie_sampling_forward_model_10min_v1/actual_rae_sampled_{source}_sized_mie_model_summary.csv"
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = read_table(summary)
    path = _plot_point_grid(df, source, out_dir, float(args.window_s))
    (out_dir / f"{source}_point_source_mie_grid_inputs.txt").write_text(
        "\n".join(
            [
                f"source_event_schedule={source}",
                f"summary={summary}",
                f"window_s={float(args.window_s)}",
                f"plot={path}",
                "model=point-source MIE response sampled at actual lower-V RAE timestamps",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(path)


if __name__ == "__main__":
    main()
