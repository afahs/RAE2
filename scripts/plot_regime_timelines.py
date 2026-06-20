#!/usr/bin/env python
"""Plot source-like and anti-template moving-body event regimes versus time."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rylevonberg.table_io import read_table


ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "outputs/moving_body_regime_physical_differences_v1/moving_body_event_regime_geometry_table.csv"
OUT = ROOT / "outputs/moving_body_regime_physical_differences_v1/moon_center_regime_plots"

COLORS = {"source_like": "#2ca02c", "anti_template": "#d62728"}
MARKERS = {"disappearance": "o", "reappearance": "^"}


def _format_time_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=35, labelsize=8)


def _plot_frequency_timeline(df: pd.DataFrame, source: str) -> Path:
    sub = df[df["source_name"].eq(source) & df["regime"].isin(COLORS)].copy()
    sub["event_time"] = pd.to_datetime(sub["predicted_event_time"], errors="coerce")
    sub = sub.dropna(subset=["event_time", "frequency_mhz"])

    fig, ax = plt.subplots(figsize=(13.5, 5.2))
    for regime in ["source_like", "anti_template"]:
        for event_type in ["disappearance", "reappearance"]:
            g = sub[sub["regime"].eq(regime) & sub["event_type"].eq(event_type)]
            if g.empty:
                continue
            ax.scatter(
                g["event_time"],
                g["frequency_mhz"],
                s=18,
                alpha=0.62,
                c=COLORS[regime],
                marker=MARKERS[event_type],
                linewidths=0,
                label=f"{regime} {event_type}",
            )

    ax.set_yscale("log")
    ax.set_ylabel("Frequency (MHz)")
    ax.set_xlabel("Predicted event time")
    ax.grid(alpha=0.22)
    _format_time_axis(ax)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper center", ncol=4, frameon=False, fontsize=8)
    ax.set_title(
        f"{source}: event regime versus time and frequency\n"
        "green = source-like, red = anti-template; circles = disappearance, triangles = reappearance"
    )
    fig.tight_layout()

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{source}_event_regime_frequency_timeline.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _plot_contrast_by_frequency(df: pd.DataFrame, source: str) -> Path:
    sub = df[df["source_name"].eq(source) & df["regime"].isin(COLORS)].copy()
    sub["event_time"] = pd.to_datetime(sub["predicted_event_time"], errors="coerce")
    sub = sub.dropna(subset=["event_time", "frequency_mhz", "source_like_fractional_contrast"])
    freqs = sorted(sub["frequency_mhz"].dropna().unique())

    ncols = 3
    nrows = int(np.ceil(len(freqs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.25 * nrows), sharex=True, sharey=False)
    axes = np.asarray(axes).ravel()

    for ax, freq in zip(axes, freqs):
        f = sub[np.isclose(sub["frequency_mhz"], freq)]
        for regime in ["source_like", "anti_template"]:
            for event_type in ["disappearance", "reappearance"]:
                g = f[f["regime"].eq(regime) & f["event_type"].eq(event_type)]
                if g.empty:
                    continue
                ax.scatter(
                    g["event_time"],
                    g["source_like_fractional_contrast"],
                    s=16,
                    alpha=0.66,
                    c=COLORS[regime],
                    marker=MARKERS[event_type],
                    linewidths=0,
                    label=f"{regime} {event_type}",
                )
        ax.axhline(0.0, color="0.2", lw=0.8, alpha=0.65)
        ax.set_title(f"{freq:.2f} MHz")
        ax.grid(alpha=0.2)
        _format_time_axis(ax)

    for ax in axes[len(freqs) :]:
        ax.axis("off")

    for row in range(nrows):
        axes[row * ncols].set_ylabel("Source-like fractional contrast")
    for ax in axes[-ncols:]:
        if ax.has_data():
            ax.set_xlabel("Predicted event time")

    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="upper center", ncol=4, frameon=False, fontsize=8)
    fig.suptitle(
        f"{source}: per-frequency event contrast versus time\n"
        "positive contrast is source-like; negative contrast is anti-template",
        y=0.997,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955))

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{source}_event_regime_contrast_timeline_by_frequency.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def main() -> None:
    df = read_table(TABLE, low_memory=False)
    paths = []
    for source in ["earth", "sun"]:
        paths.append(_plot_frequency_timeline(df, source))
        paths.append(_plot_contrast_by_frequency(df, source))

    index = OUT / "moon_center_ra_dec_regime_plot_index.md"
    existing = index.read_text(encoding="utf-8") if index.exists() else "# Regime Plots\n"
    lines = [
        "",
        "## Time-Based Regime Plots",
        "",
        "These plots use the same source-like / anti-template classifications as the Moon-center RA/Dec plots,",
        "but place events on the time axis.",
        "",
        "- Frequency timeline: x = predicted event time, y = frequency.",
        "- Contrast timeline: one panel per frequency, x = predicted event time, y = source-like fractional contrast.",
        "- Positive contrast is source-like; negative contrast is anti-template.",
        "",
    ]
    lines.extend(f"- `{p}`" for p in paths)
    index.write_text(existing.rstrip() + "\n" + "\n".join(lines) + "\n", encoding="utf-8")

    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
