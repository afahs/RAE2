#!/usr/bin/env python
"""Build inspection plots for the planetary confirmation survey."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import build_source_inspection_plots as plots


ROOT = Path(__file__).resolve().parents[1]
SURVEY = ROOT / "outputs" / "planetary_confirmation_survey"
OUT = ROOT / "outputs" / "planetary_confirmation_survey" / "inspection_plots"
PLANETS = ["mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune"]


def _summary_plot() -> Path | None:
    df = plots._read(SURVEY / "planetary_confirmation_summary.csv")
    if df.empty:
        return None
    fig, ax = plots.plt.subplots(figsize=(10, 4))
    vals = pd.to_numeric(df["best_confirmed_snr"], errors="coerce").to_numpy(dtype=float)
    colors = ["#4c78a8" if v >= 0 else "#d95f02" for v in vals]
    ax.bar(df["source_name"], vals, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Planetary survey: best confirmed SNR by planet")
    ax.set_ylabel("Stacked SNR")
    ax.tick_params(axis="x", labelrotation=25)
    path = OUT / "planetary_best_confirmed_snr.png"
    plots._save(fig, path)
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    paths_by_planet: dict[str, list[Path]] = {}
    for planet in PLANETS:
        suite = SURVEY / planet
        paths: list[Path] = []
        for func in [
            plots._plot_timing,
            plots._plot_window,
            plots._plot_event_type,
            plots._plot_quality_survival,
            plots._plot_wrong_controls,
            plots._plot_leave_one_month,
        ]:
            path = func(planet, suite, OUT)
            if path is not None:
                paths.append(path)
        paths_by_planet[planet] = paths
    summary = _summary_plot()
    lines = [
        "# Planetary Inspection Plots",
        "",
        "Generated from `outputs/planetary_confirmation_survey`.",
        "",
    ]
    if summary is not None:
        lines.extend(["## Summary", "", f"- [{summary.name}]({summary.relative_to(OUT)})", ""])
    for planet, paths in paths_by_planet.items():
        lines.extend([f"## {planet.title()}", ""])
        for path in paths:
            lines.append(f"- [{path.name}]({path.relative_to(OUT)})")
        lines.append("")
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(OUT / "README.md")


if __name__ == "__main__":
    main()
