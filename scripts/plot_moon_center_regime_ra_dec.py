#!/usr/bin/env python
"""Plot Moon-center RA/Dec of moving-body events by regime."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rylevonberg.table_io import read_table


ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "outputs/moving_body_regime_physical_differences_v1/moving_body_event_regime_geometry_table.csv"
OUT = ROOT / "outputs/moving_body_regime_physical_differences_v1/moon_center_regime_plots"

COLORS = {"source_like": "#2ca02c", "anti_template": "#d62728"}
MARKERS = {"disappearance": "o", "reappearance": "^"}


def _plot_source(df: pd.DataFrame, source: str) -> Path:
    sub = df[df["source_name"].eq(source) & df["regime"].isin(["source_like", "anti_template"])].copy()
    freqs = sorted(sub["frequency_mhz"].dropna().unique())
    ncols = 3
    nrows = int(np.ceil(len(freqs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.6 * nrows), sharex=True, sharey=True)
    axes = np.asarray(axes).ravel()

    for ax, freq in zip(axes, freqs):
        f = sub[np.isclose(sub["frequency_mhz"], freq)]
        for regime in ["source_like", "anti_template"]:
            for event_type in ["disappearance", "reappearance"]:
                g = f[f["regime"].eq(regime) & f["event_type"].eq(event_type)]
                if g.empty:
                    continue
                ax.scatter(
                    g["moon_center_dec_deg"],
                    g["moon_center_ra_deg"],
                    s=14,
                    alpha=0.68,
                    c=COLORS[regime],
                    marker=MARKERS[event_type],
                    linewidths=0,
                    label=f"{regime} {event_type}",
                )
        ax.set_title(f"{freq:.2f} MHz")
        ax.grid(alpha=0.22)
        ax.set_xlim(-90, 90)
        ax.set_ylim(0, 360)

    for ax in axes[len(freqs):]:
        ax.axis("off")

    for row in range(nrows):
        axes[row * ncols].set_ylabel("Moon-center RA (deg)")
    for ax in axes[-ncols:]:
        if ax.has_data():
            ax.set_xlabel("Moon-center Dec (deg)")

    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="upper center", ncol=4, frameon=False, fontsize=8)
    fig.suptitle(
        f"{source}: Moon-center coordinates of lower-V events by regime\n"
        "x = Moon-center Dec, y = Moon-center RA; green = source-like, red = anti-template",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{source}_moon_center_ra_dec_by_regime_all_frequencies.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _plot_source_event_type(df: pd.DataFrame, source: str, event_type: str) -> Path:
    sub = df[
        df["source_name"].eq(source)
        & df["event_type"].eq(event_type)
        & df["regime"].isin(["source_like", "anti_template"])
    ].copy()
    freqs = sorted(sub["frequency_mhz"].dropna().unique())
    ncols = 3
    nrows = int(np.ceil(len(freqs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.6 * nrows), sharex=True, sharey=True)
    axes = np.asarray(axes).ravel()

    for ax, freq in zip(axes, freqs):
        f = sub[np.isclose(sub["frequency_mhz"], freq)]
        for regime in ["source_like", "anti_template"]:
            g = f[f["regime"].eq(regime)]
            if g.empty:
                continue
            ax.scatter(
                g["moon_center_dec_deg"],
                g["moon_center_ra_deg"],
                s=15,
                alpha=0.72,
                c=COLORS[regime],
                marker="o",
                linewidths=0,
                label=regime,
            )
        ax.set_title(f"{freq:.2f} MHz")
        ax.grid(alpha=0.22)
        ax.set_xlim(-90, 90)
        ax.set_ylim(0, 360)

    for ax in axes[len(freqs):]:
        ax.axis("off")
    for row in range(nrows):
        axes[row * ncols].set_ylabel("Moon-center RA (deg)")
    for ax in axes[-ncols:]:
        if ax.has_data():
            ax.set_xlabel("Moon-center Dec (deg)")

    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="upper center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle(
        f"{source} {event_type}: Moon-center coordinates by regime\n"
        "x = Moon-center Dec, y = Moon-center RA; green = source-like, red = anti-template",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{source}_{event_type}_moon_center_ra_dec_by_regime_all_frequencies.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def main() -> None:
    df = read_table(TABLE, low_memory=False)
    paths = []
    for source in ["earth", "sun"]:
        paths.append(_plot_source(df, source))
        for event_type in ["disappearance", "reappearance"]:
            paths.append(_plot_source_event_type(df, source, event_type))

    lines = [
        "# Moon-Center RA/Dec Regime Plots",
        "",
        "Each panel is one frequency. The x-axis is Moon-center Dec and the y-axis is Moon-center RA at the predicted event.",
        "",
        "- Green: source-like event",
        "- Red: anti-template event",
        "- Circles: disappearance",
        "- Triangles: reappearance",
        "",
        "Generated plots:",
        "",
    ]
    lines.extend(f"- `{p}`" for p in paths)
    (OUT / "moon_center_ra_dec_regime_plot_index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
