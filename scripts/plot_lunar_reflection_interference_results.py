#!/usr/bin/env python
"""Plot summary figures from a lunar-reflection interference run."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.table_io import read_table  # noqa: E402


ANT_LABEL = {"rv1_coarse": "upper V", "rv2_coarse": "lower V"}
MODE_COLOR = {
    "real": "#1b9e77",
    "scrambled": "#7570b3",
    "wrong_frequency_low": "#d95f02",
    "wrong_frequency_high": "#e7298a",
}


def _read(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return read_table(path, low_memory=False)


def _freq_label(values: pd.Series) -> list[str]:
    return [f"{float(v):.2f}" for v in values]


def plot_true_time_phase_controls(fits: pd.DataFrame, out_dir: Path, antenna: str) -> Path | None:
    sub = fits[
        fits["antenna"].astype(str).eq(str(antenna))
        & fits["event_type"].astype(str).eq("combined")
        & np.isclose(pd.to_numeric(fits["time_shift_s"], errors="coerce"), 0.0)
    ].copy()
    if sub.empty:
        return None
    sub["frequency_mhz"] = pd.to_numeric(sub["frequency_mhz"], errors="coerce")
    sub["fringe_snr"] = pd.to_numeric(sub["fringe_snr"], errors="coerce")
    sub["fringe_amplitude_z"] = pd.to_numeric(sub["fringe_amplitude_z"], errors="coerce")
    freqs = sorted(sub["frequency_mhz"].dropna().unique())

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.0), sharex=True)
    for mode, grp in sub.groupby("phase_mode", sort=True):
        g = grp.sort_values("frequency_mhz")
        label = str(mode).replace("_", " ")
        color = MODE_COLOR.get(str(mode))
        axes[0].plot(g["frequency_mhz"], g["fringe_snr"], marker="o", lw=1.5, label=label, color=color)
        axes[1].plot(g["frequency_mhz"], g["fringe_amplitude_z"], marker="o", lw=1.5, label=label, color=color)

    axes[0].axhline(2.0, color="0.45", lw=0.8, ls="--")
    axes[0].axhline(3.0, color="0.25", lw=0.8, ls=":")
    axes[0].set_ylabel("fringe fit SNR")
    axes[1].set_ylabel("fringe amplitude (z)")
    axes[1].set_xlabel("frequency (MHz)")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(freqs)
        ax.set_xticklabels(_freq_label(pd.Series(freqs)))
        ax.grid(alpha=0.25, which="both")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"True-time phase-law controls, {ANT_LABEL.get(antenna, antenna)}")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = out_dir / f"{antenna}_true_time_phase_control_summary.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_shift_control_heatmap(fits: pd.DataFrame, out_dir: Path, antenna: str, value_col: str = "fringe_snr") -> Path | None:
    sub = fits[
        fits["antenna"].astype(str).eq(str(antenna))
        & fits["event_type"].astype(str).eq("combined")
        & fits["phase_mode"].astype(str).eq("real")
    ].copy()
    if sub.empty:
        return None
    sub["frequency_mhz"] = pd.to_numeric(sub["frequency_mhz"], errors="coerce")
    sub["time_shift_s"] = pd.to_numeric(sub["time_shift_s"], errors="coerce")
    sub[value_col] = pd.to_numeric(sub[value_col], errors="coerce")
    pivot = sub.pivot_table(index="time_shift_s", columns="frequency_mhz", values=value_col, aggfunc="first")
    pivot = pivot.sort_index().sort_index(axis=1)
    if pivot.empty:
        return None

    data = pivot.to_numpy(dtype=float)
    vmax = np.nanmax(np.abs(data)) if np.isfinite(data).any() else 1.0
    vmax = max(vmax, 1.0)
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    im = ax.imshow(data, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(pivot.columns)), [f"{float(x):.2f}" for x in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)), [f"{float(x):.0f}" for x in pivot.index])
    ax.set_xlabel("frequency (MHz)")
    ax.set_ylabel("event-time shift (s)")
    ax.set_title(f"Real phase fit under event-time shifts, {ANT_LABEL.get(antenna, antenna)}")
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            if np.isfinite(data[y, x]):
                ax.text(x, y, f"{data[y, x]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=value_col.replace("_", " "))
    fig.tight_layout()
    path = out_dir / f"{antenna}_real_phase_shift_control_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_phase_mode_heatmap(fits: pd.DataFrame, out_dir: Path, antenna: str, value_col: str = "fringe_snr") -> Path | None:
    sub = fits[
        fits["antenna"].astype(str).eq(str(antenna))
        & fits["event_type"].astype(str).eq("combined")
        & np.isclose(pd.to_numeric(fits["time_shift_s"], errors="coerce"), 0.0)
    ].copy()
    if sub.empty:
        return None
    sub["frequency_mhz"] = pd.to_numeric(sub["frequency_mhz"], errors="coerce")
    sub[value_col] = pd.to_numeric(sub[value_col], errors="coerce")
    order = ["real", "scrambled", "wrong_frequency_low", "wrong_frequency_high"]
    pivot = sub.pivot_table(index="phase_mode", columns="frequency_mhz", values=value_col, aggfunc="first")
    pivot = pivot.reindex([m for m in order if m in pivot.index]).sort_index(axis=1)
    if pivot.empty:
        return None

    data = pivot.to_numpy(dtype=float)
    vmax = np.nanmax(np.abs(data)) if np.isfinite(data).any() else 1.0
    vmax = max(vmax, 1.0)
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    im = ax.imshow(data, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(pivot.columns)), [f"{float(x):.2f}" for x in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)), [str(x).replace("_", " ") for x in pivot.index])
    ax.set_xlabel("frequency (MHz)")
    ax.set_title(f"True-time phase-law controls, {ANT_LABEL.get(antenna, antenna)}")
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            if np.isfinite(data[y, x]):
                ax.text(x, y, f"{data[y, x]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=value_col.replace("_", " "))
    fig.tight_layout()
    path = out_dir / f"{antenna}_phase_mode_control_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_phase_folded_with_fit(phase_stack: pd.DataFrame, fits: pd.DataFrame, out_dir: Path, antenna: str) -> Path | None:
    sub = phase_stack[phase_stack["antenna"].astype(str).eq(str(antenna))].copy()
    fit_rows = fits[
        fits["antenna"].astype(str).eq(str(antenna))
        & fits["event_type"].astype(str).eq("combined")
        & fits["phase_mode"].astype(str).eq("real")
        & np.isclose(pd.to_numeric(fits["time_shift_s"], errors="coerce"), 0.0)
    ].copy()
    if sub.empty or fit_rows.empty:
        return None
    sub["frequency_mhz"] = pd.to_numeric(sub["frequency_mhz"], errors="coerce")
    sub["phase_bin_center_rad"] = pd.to_numeric(sub["phase_bin_center_rad"], errors="coerce")
    sub["median_value"] = pd.to_numeric(sub["median_value"], errors="coerce")
    sub["value_sem"] = pd.to_numeric(sub["value_sem"], errors="coerce")
    fit_rows["frequency_mhz"] = pd.to_numeric(fit_rows["frequency_mhz"], errors="coerce")
    fit_rows["fringe_cos_coeff"] = pd.to_numeric(fit_rows["fringe_cos_coeff"], errors="coerce")
    fit_rows["fringe_sin_coeff"] = pd.to_numeric(fit_rows["fringe_sin_coeff"], errors="coerce")

    freqs = sorted(sub["frequency_mhz"].dropna().unique())
    fig, axes = plt.subplots(len(freqs), 1, figsize=(11.5, max(7, 1.55 * len(freqs))), sharex=True, sharey=False)
    axes = np.asarray(axes).reshape(-1)
    phase_grid = np.linspace(0.0, 2.0 * np.pi, 300)
    for ax, freq in zip(axes, freqs):
        grp = sub[np.isclose(sub["frequency_mhz"], float(freq))].sort_values("phase_bin_center_rad")
        x = grp["phase_bin_center_rad"] / (2.0 * np.pi)
        ax.errorbar(
            x,
            grp["median_value"],
            yerr=grp["value_sem"],
            marker="o",
            markersize=2.6,
            lw=1.0,
            elinewidth=0.55,
            capsize=1.0,
            color="#4c78a8",
            label="phase-binned stack",
        )
        fit = fit_rows[np.isclose(fit_rows["frequency_mhz"], float(freq))]
        if not fit.empty:
            row = fit.iloc[0]
            y_fit = float(row["fringe_cos_coeff"]) * np.cos(phase_grid) + float(row["fringe_sin_coeff"]) * np.sin(phase_grid)
            ax.plot(phase_grid / (2.0 * np.pi), y_fit, color="#1b9e77", lw=1.4, label="sample-level fit")
        ax.axhline(0.0, color="0.65", lw=0.7)
        ax.set_title(f"{float(freq):.2f} MHz", fontsize=9)
        ax.set_ylabel("residual z")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8, loc="best")
    axes[-1].set_xlabel("wrapped reflection phase (cycles)")
    fig.suptitle(f"Phase-folded residuals with fitted real-phase sinusoid, {ANT_LABEL.get(antenna, antenna)}")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = out_dir / f"{antenna}_phase_folded_with_fit_overlay.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_phase_rate_gate(rate: pd.DataFrame, out_dir: Path, antenna: str) -> Path | None:
    sub = rate[
        rate["antenna"].astype(str).eq(str(antenna))
        & np.isclose(pd.to_numeric(rate["time_shift_s"], errors="coerce"), 0.0)
    ].copy()
    if sub.empty:
        return None
    sub["frequency_mhz"] = pd.to_numeric(sub["frequency_mhz"], errors="coerce")
    sub["median_fringe_period_s"] = pd.to_numeric(sub["median_fringe_period_s"], errors="coerce")
    sub["median_samples_per_fringe"] = pd.to_numeric(sub["median_samples_per_fringe"], errors="coerce")
    sub["reflection_valid_fraction"] = pd.to_numeric(sub["reflection_valid_fraction"], errors="coerce")

    fig, axes = plt.subplots(3, 1, figsize=(11.5, 8.5), sharex=True)
    for event_type, grp in sub.groupby("event_type", sort=True):
        g = grp.sort_values("frequency_mhz")
        axes[0].plot(g["frequency_mhz"], g["median_fringe_period_s"], marker="o", lw=1.4, label=event_type)
        axes[1].plot(g["frequency_mhz"], g["median_samples_per_fringe"], marker="o", lw=1.4, label=event_type)
        axes[2].plot(g["frequency_mhz"], g["reflection_valid_fraction"], marker="o", lw=1.4, label=event_type)
    axes[0].set_ylabel("period (s)")
    axes[1].set_ylabel("samples / fringe")
    axes[2].set_ylabel("valid reflection fraction")
    axes[2].set_xlabel("frequency (MHz)")
    for ax in axes:
        ax.set_xscale("log")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Sampling and reflection-geometry gate, {ANT_LABEL.get(antenna, antenna)}")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = out_dir / f"{antenna}_phase_rate_sampling_gate.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(ROOT / "outputs/lower_v_lunar_reflection_interference_v1"))
    parser.add_argument("--antennas", default="rv2_coarse")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    fits = _read(out_dir / "reflection_fringe_fit_summary.csv")
    rate = _read(out_dir / "reflection_phase_rate_summary.csv")
    phase_stack = _read(out_dir / "phase_folded_stack.csv")
    antennas = [x.strip() for x in str(args.antennas).split(",") if x.strip()]

    paths: list[Path] = []
    for antenna in antennas:
        for path in [
            plot_phase_rate_gate(rate, out_dir, antenna),
            plot_true_time_phase_controls(fits, out_dir, antenna),
            plot_shift_control_heatmap(fits, out_dir, antenna),
            plot_phase_mode_heatmap(fits, out_dir, antenna),
            plot_phase_folded_with_fit(phase_stack, fits, out_dir, antenna),
        ]:
            if path is not None:
                paths.append(path)
                print(path)


if __name__ == "__main__":
    main()
