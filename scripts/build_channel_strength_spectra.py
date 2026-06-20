#!/usr/bin/env python
"""Build slide-ready lower-V channel-strength spectra from planetary survey outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rylevonberg.table_io import read_table


ROOT = Path(__file__).resolve().parents[1]
SURVEY = ROOT / "outputs/planetary_confirmation_survey"
OUT = ROOT / "outputs/slide_assets/channel_strength_spectra"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = read_table(SURVEY / "planetary_confirmation_summary.csv")
    rows = []
    for _, source_row in summary.iterrows():
        source = str(source_row["source_name"])
        scan_path = SURVEY / source / "initial_channel_scan.csv"
        if not scan_path.exists():
            continue
        scan = read_table(scan_path)
        sub = scan[
            scan["antenna"].astype(str).eq("rv2_coarse")
            & pd.to_numeric(scan["window_s"], errors="coerce").eq(float(source_row["target_window_s"]))
        ].copy()
        sub["planetary_survey_target_window_s"] = float(source_row["target_window_s"])
        sub["planetary_survey_best_confirmed_snr"] = float(source_row["best_confirmed_snr"])
        sub["planetary_survey_best_timing_offset_s"] = float(source_row["best_timing_offset_s"])
        sub["planetary_survey_status"] = source_row["status"]
        rows.append(sub)
    spectra = pd.concat(rows, ignore_index=True)
    spectra.to_csv(OUT / "planetary_lower_v_channel_snr_spectrum.csv", index=False)

    _plot_earth(spectra)
    _plot_planets(spectra, summary["source_name"].astype(str).tolist())
    _write_summary(spectra)


def _plot_earth(spectra: pd.DataFrame) -> None:
    earth = spectra[spectra["source_name"].eq("earth")].sort_values("frequency_mhz")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(earth["frequency_mhz"], earth["stacked_snr"], marker="o", lw=2.0, color="#1f77b4")
    ax.axhline(0, color="0.25", lw=0.9)
    ax.axhline(3, color="0.6", lw=0.8, ls="--")
    ax.axhline(-3, color="0.6", lw=0.8, ls="--")
    for _, row in earth.iterrows():
        ax.text(
            row["frequency_mhz"],
            row["stacked_snr"],
            f" {row['frequency_mhz']:.2g}",
            fontsize=7,
            va="bottom" if row["stacked_snr"] >= 0 else "top",
        )
    ax.set_xscale("log")
    ax.set_xticks(earth["frequency_mhz"].to_list())
    ax.set_xticklabels([f"{x:.2g}" for x in earth["frequency_mhz"]])
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Stacked SNR, lower V")
    ax.set_title("Earth lower-V channel-strength spectrum (planetary survey)")
    fig.tight_layout()
    fig.savefig(OUT / "earth_lower_v_planetary_snr_spectrum.png", dpi=180)
    plt.close(fig)


def _plot_planets(spectra: pd.DataFrame, planets: list[str]) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(11, 12), sharex=True, constrained_layout=True)
    ymax = np.nanmax(np.abs(spectra["stacked_snr"].to_numpy(dtype=float)))
    ylim = max(5, min(ymax * 1.1, 70))
    colors = {"repeatable_candidate": "#1b9e77", "unresolved_or_episodic": "#7570b3"}
    for ax, planet in zip(axes.ravel(), planets):
        sub = spectra[spectra["source_name"].eq(planet)].sort_values("frequency_mhz")
        status = str(sub["planetary_survey_status"].iloc[0]) if not sub.empty else ""
        ax.plot(sub["frequency_mhz"], sub["stacked_snr"], marker="o", lw=1.6, color=colors.get(status, "#4c78a8"))
        ax.axhline(0, color="0.25", lw=0.8)
        ax.axhline(3, color="0.75", lw=0.7, ls="--")
        ax.axhline(-3, color="0.75", lw=0.7, ls="--")
        ax.set_xscale("log")
        title = f"{planet.capitalize()} lower V ({int(sub['window_s'].iloc[0])} s)" if not sub.empty else planet.capitalize()
        ax.set_title(title, fontsize=10)
        ax.set_ylim(-ylim, ylim)
        ax.grid(alpha=0.18)
    ticks = sorted(spectra["frequency_mhz"].unique())
    for ax in axes.ravel():
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{x:.2g}" for x in ticks])
    for ax in axes[-1, :]:
        ax.set_xlabel("Frequency (MHz)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Stacked SNR")
    fig.suptitle("Planetary survey lower-V channel-strength spectra", fontsize=14)
    fig.savefig(OUT / "planetary_lower_v_snr_spectra.png", dpi=180)
    plt.close(fig)


def _write_summary(spectra: pd.DataFrame) -> None:
    best_rows = []
    for source, group in spectra.groupby("source_name", sort=True):
        best = group.assign(abs_snr=group["stacked_snr"].abs()).sort_values("abs_snr", ascending=False).iloc[0]
        best_rows.append(
            {
                "source": source,
                "strongest_lower_v_frequency_mhz": best["frequency_mhz"],
                "strongest_lower_v_snr": best["stacked_snr"],
                "window_s": best["window_s"],
                "planetary_survey_status": best["planetary_survey_status"],
            }
        )
    best_df = pd.DataFrame(best_rows)
    best_df.to_csv(OUT / "planetary_lower_v_spectrum_best_channels.csv", index=False)
    md = [
        "# Lower-V Channel-Strength Spectra",
        "",
        "These plots use planetary survey per-channel stacked SNR values, filtered to `rv2_coarse` only. In slide text this is the lower V antenna, pointing toward the Moon.",
        "",
        f"- Earth spectrum: `{OUT / 'earth_lower_v_planetary_snr_spectrum.png'}`",
        f"- Planetary small multiples: `{OUT / 'planetary_lower_v_snr_spectra.png'}`",
        f"- Full CSV: `{OUT / 'planetary_lower_v_channel_snr_spectrum.csv'}`",
        "",
        "| source | strongest lower-V frequency (MHz) | lower-V SNR | window (s) | survey status |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for _, row in best_df.iterrows():
        md.append(
            f"| {row['source']} | {row['strongest_lower_v_frequency_mhz']:.2f} | "
            f"{row['strongest_lower_v_snr']:.2f} | {int(row['window_s'])} | {row['planetary_survey_status']} |"
        )
    (OUT / "README.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
