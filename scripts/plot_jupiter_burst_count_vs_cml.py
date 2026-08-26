#!/usr/bin/env python
"""Plot blind burst counts versus Jupiter System III longitude.

The primary plot is the raw number of per-frequency blind burst events in each
System III CML bin.  A companion table and rate plot include exposure counts
from the gain-filtered sample table, because the beam-selected coverage is not
uniform in CML.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.util import ensure_dir, write_json  # noqa: E402
from scripts.run_jupiter_beam_burst_phase_enrichment import read_selected_samples  # noqa: E402


DEFAULT_BURSTS = (
    ROOT
    / "outputs/jupiter_upper_v_beam_single_frequency_burst_phase_v1/jupiter_blind_burst_catalog.csv"
)
DEFAULT_SELECTED = (
    ROOT
    / "outputs/jupiter_upper_v_beam_gain10db_io_phase_frequency_v1/jupiter_upper_v_beam_selected_samples.npy"
)
DEFAULT_OUT = ROOT / "outputs/jupiter_upper_v_beam_single_frequency_burst_phase_v1"


def cml_bin_centers(bin_deg: float) -> np.ndarray:
    return np.arange(0.5 * float(bin_deg), 360.0, float(bin_deg))


def attach_cml_bins(df: pd.DataFrame, bin_deg: float) -> pd.DataFrame:
    out = df.copy()
    cml = pd.to_numeric(out["jupiter_cml_spice_deg"], errors="coerce").to_numpy(dtype=float) % 360.0
    out["cml_bin_deg"] = np.floor(cml / float(bin_deg)) * float(bin_deg) + 0.5 * float(bin_deg)
    return out


def cml_frequency_summary(samples: pd.DataFrame, bursts: pd.DataFrame, bin_deg: float) -> pd.DataFrame:
    sample_bins = attach_cml_bins(samples, bin_deg)
    burst_bins = attach_cml_bins(bursts, bin_deg)
    freqs = sample_bins[["frequency_band", "frequency_mhz"]].drop_duplicates().sort_values("frequency_mhz")
    full = pd.MultiIndex.from_product(
        [freqs["frequency_band"].to_numpy(dtype=int), cml_bin_centers(bin_deg)],
        names=["frequency_band", "cml_bin_deg"],
    ).to_frame(index=False)
    full = full.merge(freqs, on="frequency_band", how="left")
    exposure = (
        sample_bins.groupby(["frequency_band", "cml_bin_deg"], sort=True)
        .size()
        .rename("n_exposure_samples")
        .reset_index()
    )
    counts = (
        burst_bins.groupby(["frequency_band", "cml_bin_deg"], sort=True)
        .size()
        .rename("n_bursts")
        .reset_index()
    )
    out = full.merge(exposure, on=["frequency_band", "cml_bin_deg"], how="left")
    out = out.merge(counts, on=["frequency_band", "cml_bin_deg"], how="left")
    out[["n_exposure_samples", "n_bursts"]] = out[["n_exposure_samples", "n_bursts"]].fillna(0).astype(int)
    out["burst_rate_per_100k_samples"] = np.divide(
        out["n_bursts"] * 100000.0,
        out["n_exposure_samples"],
        out=np.full(len(out), np.nan),
        where=out["n_exposure_samples"].to_numpy(dtype=float) > 0.0,
    )
    return out.sort_values(["frequency_mhz", "cml_bin_deg"]).reset_index(drop=True)


def plot_count_heatmap(summary: pd.DataFrame, out_dir: Path, bin_deg: float) -> Path:
    centers = cml_bin_centers(bin_deg)
    mat = summary.pivot_table(
        index="frequency_mhz",
        columns="cml_bin_deg",
        values="n_bursts",
        aggfunc="first",
    ).reindex(columns=centers)
    vals = mat.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(12.6, 5.8))
    im = ax.imshow(
        vals,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[0.0, 360.0, -0.5, len(mat.index) - 0.5],
        cmap="viridis",
        norm=Normalize(vmin=0.0, vmax=max(1.0, float(np.nanmax(vals)))),
    )
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels([f"{v:.2f}" for v in mat.index])
    ax.set_xlim(0, 360)
    ax.set_xticks(np.arange(0, 361, 30))
    ax.set_xlabel("Jupiter System III CML (deg)")
    ax.set_ylabel("frequency (MHz)")
    ax.set_title("Blind burst count by frequency and Jupiter System III longitude")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="burst count")
    fig.tight_layout()
    path = out_dir / "jupiter_blind_burst_count_system_iii_cml_frequency.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_count_panels(summary: pd.DataFrame, out_dir: Path, bin_deg: float) -> Path:
    freqs = summary[["frequency_band", "frequency_mhz"]].drop_duplicates().sort_values("frequency_mhz")
    n_cols = 3
    n_rows = int(np.ceil(len(freqs) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14.0, 3.0 * n_rows), sharex=True)
    axes_arr = np.asarray(axes).reshape(-1)
    width = float(bin_deg) * 0.86
    for ax, row in zip(axes_arr, freqs.itertuples(index=False)):
        work = summary[summary["frequency_band"].eq(int(row.frequency_band))].sort_values("cml_bin_deg")
        x = work["cml_bin_deg"].to_numpy(dtype=float)
        y = work["n_bursts"].to_numpy(dtype=float)
        ax.bar(x, y, width=width, color="#386cb0", alpha=0.82, edgecolor="white", linewidth=0.35)
        ymax = float(np.nanmax(y)) if y.size else 1.0
        ax.set_ylim(0.0, ymax + 1.0)
        ax.set_title(f"{float(row.frequency_mhz):.2f} MHz")
        ax.grid(True, axis="y", color="0.90", lw=0.45)
        ax.text(
            0.02,
            0.94,
            f"{int(work['n_bursts'].sum())} bursts",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.76, "pad": 1.0},
        )
    for ax in axes_arr[len(freqs) :]:
        ax.axis("off")
    for ax in axes_arr[: len(freqs)]:
        ax.set_xlim(0, 360)
        ax.set_xticks(np.arange(0, 361, 60))
    for ax in axes_arr[::n_cols]:
        ax.set_ylabel("burst count")
    for ax in axes_arr[max(0, (n_rows - 1) * n_cols) : n_rows * n_cols]:
        if ax.has_data():
            ax.set_xlabel("Jupiter System III CML (deg)")
    fig.suptitle("Blind burst count versus Jupiter System III longitude", y=0.995)
    fig.tight_layout()
    path = out_dir / "jupiter_blind_burst_count_by_system_iii_cml_per_frequency.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_rate_heatmap(summary: pd.DataFrame, out_dir: Path, bin_deg: float) -> Path:
    centers = cml_bin_centers(bin_deg)
    mat = summary.pivot_table(
        index="frequency_mhz",
        columns="cml_bin_deg",
        values="burst_rate_per_100k_samples",
        aggfunc="first",
    ).reindex(columns=centers)
    vals = mat.to_numpy(dtype=float)
    scaled = np.full_like(vals, np.nan, dtype=float)
    ylabels = []
    for idx, freq in enumerate(mat.index):
        row = vals[idx]
        finite = row[np.isfinite(row)]
        if finite.size == 0:
            ylabels.append(f"{freq:.2f}\nno data")
            continue
        lo = float(np.nanmin(finite))
        hi = float(np.nanpercentile(finite, 98)) if finite.size >= 4 else float(np.nanmax(finite))
        if not np.isfinite(hi) or np.isclose(lo, hi):
            hi = float(np.nanmax(finite))
        if not np.isfinite(hi) or np.isclose(lo, hi):
            scaled[idx] = 0.0
        else:
            scaled[idx] = np.clip((row - lo) / (hi - lo), 0.0, 1.0)
        ylabels.append(f"{freq:.2f}")
    fig, ax = plt.subplots(figsize=(12.6, 5.8))
    im = ax.imshow(
        scaled,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[0.0, 360.0, -0.5, len(mat.index) - 0.5],
        cmap="magma",
        norm=Normalize(vmin=0.0, vmax=1.0),
    )
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels(ylabels)
    ax.set_xlim(0, 360)
    ax.set_xticks(np.arange(0, 361, 30))
    ax.set_xlabel("Jupiter System III CML (deg)")
    ax.set_ylabel("frequency (MHz)")
    ax.set_title("Exposure-normalized blind burst rate by System III longitude, row-scaled")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="row-scaled bursts / exposure")
    fig.tight_layout()
    path = out_dir / "jupiter_blind_burst_rate_system_iii_cml_frequency_row_scaled.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--burst-catalog", type=Path, default=DEFAULT_BURSTS)
    parser.add_argument("--selected-samples", type=Path, default=DEFAULT_SELECTED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cml-bin-deg", type=float, default=10.0)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    write_json(
        out_dir / "jupiter_blind_burst_system_iii_cml_plot_config.json",
        {
            "burst_catalog": str(args.burst_catalog),
            "selected_samples": str(args.selected_samples),
            "cml_bin_deg": float(args.cml_bin_deg),
        },
    )
    bursts = pd.read_csv(args.burst_catalog, parse_dates=["burst_start_time", "burst_end_time", "burst_peak_time"])
    bursts = bursts.dropna(subset=["frequency_band", "frequency_mhz", "jupiter_cml_spice_deg"]).copy()
    bursts["frequency_band"] = pd.to_numeric(bursts["frequency_band"], errors="coerce").astype(int)
    bursts["frequency_mhz"] = pd.to_numeric(bursts["frequency_mhz"], errors="coerce")
    bursts["jupiter_cml_spice_deg"] = pd.to_numeric(bursts["jupiter_cml_spice_deg"], errors="coerce") % 360.0

    samples = read_selected_samples(args.selected_samples)
    summary = cml_frequency_summary(samples, bursts, bin_deg=float(args.cml_bin_deg))
    summary_path = out_dir / "jupiter_blind_burst_system_iii_cml_frequency_summary.csv"
    summary.to_csv(summary_path, index=False)

    paths = [
        summary_path,
        plot_count_heatmap(summary, out_dir, bin_deg=float(args.cml_bin_deg)),
        plot_count_panels(summary, out_dir, bin_deg=float(args.cml_bin_deg)),
        plot_rate_heatmap(summary, out_dir, bin_deg=float(args.cml_bin_deg)),
    ]
    print("Wrote:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
