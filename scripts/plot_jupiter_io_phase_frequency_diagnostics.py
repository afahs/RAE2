#!/usr/bin/env python
"""Io-phase/frequency diagnostics for Jupiter RAE-2 samples.

These plots follow the expert recommendation to focus on Io phase rather than
System III longitude when looking for hectometric/decametric Jupiter emission.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np
import pandas as pd
from rylevonberg.table_io import read_table


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLES = ROOT / "outputs/jupiter_phase_pattern_survey_v1/jupiter_phase_pattern_sampled_points.csv"
DEFAULT_HISTORICAL = ROOT / "outputs/jupiter_historical_active_windows_v1/jupiter_historical_active_samples.csv"
DEFAULT_OUT = ROOT / "outputs/jupiter_io_phase_frequency_diagnostics_v1"

ANTENNA_LABEL = {
    "rv1_coarse": "upper V",
    "rv2_coarse": "lower V",
}

def circular_rolling_mean(y: np.ndarray, width: int = 3) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if len(y) == 0 or width <= 1:
        return y
    half = int(width) // 2
    padded = np.concatenate([y[-half:], y, y[:half]])
    out = []
    for i in range(half, half + len(y)):
        out.append(np.nanmean(padded[i - half : i + half + 1]))
    return np.asarray(out, dtype=float)


def read_samples(path: Path, label: str) -> pd.DataFrame:
    usecols = [
        "time",
        "frequency_band",
        "frequency_mhz",
        "antenna",
        "daily_z_log_power",
        "jupiter_visible_by_moon",
        "earth_visible_by_moon",
        "jupiter_cml_spice_deg",
        "io_phase_spice_deg",
    ]
    available = read_table(path, nrows=0).columns
    cols = [c for c in usecols if c in available]
    df = read_table(path, usecols=cols, parse_dates=["time"])
    if "jupiter_visible_by_moon" in df.columns:
        df = df[df["jupiter_visible_by_moon"].astype(bool)].copy()
    df["dataset"] = label
    df["io_phase_spice_deg"] = df["io_phase_spice_deg"].astype(float) % 360.0
    df["frequency_mhz"] = df["frequency_mhz"].astype(float)
    df["daily_z_log_power"] = df["daily_z_log_power"].astype(float)
    return df


def phase_bin_summary(df: pd.DataFrame, phase_bin_deg: float, high_z: float, min_count: int) -> pd.DataFrame:
    work = df.copy()
    work["io_phase_bin_deg"] = (
        np.floor(work["io_phase_spice_deg"] / float(phase_bin_deg)) * float(phase_bin_deg)
        + 0.5 * float(phase_bin_deg)
    )
    work["high_power"] = work["daily_z_log_power"] > float(high_z)
    summary = (
        work.groupby(["dataset", "antenna", "frequency_band", "frequency_mhz", "io_phase_bin_deg"], sort=True)
        .agg(
            n_samples=("daily_z_log_power", "size"),
            median_daily_z=("daily_z_log_power", "median"),
            q90_daily_z=("daily_z_log_power", lambda s: float(np.nanquantile(s, 0.90))),
            high_tail_fraction=("high_power", "mean"),
        )
        .reset_index()
    )
    summary.loc[summary["n_samples"] < int(min_count), ["median_daily_z", "q90_daily_z", "high_tail_fraction"]] = np.nan
    phase_median = summary.groupby(["dataset", "antenna", "frequency_mhz"])["high_tail_fraction"].transform("median")
    summary["high_tail_fraction_minus_frequency_median"] = summary["high_tail_fraction"] - phase_median
    return summary


def _pivot(summary: pd.DataFrame, antenna: str, value_col: str) -> pd.DataFrame:
    sub = summary[summary["antenna"].astype(str).eq(antenna)]
    mat = sub.pivot_table(index="frequency_mhz", columns="io_phase_bin_deg", values=value_col, aggfunc="mean")
    return mat.sort_index()


def _decorate_io_axis(ax: plt.Axes, expected_phases: list[float]) -> None:
    phases = [float(p) % 360.0 for p in expected_phases]
    for phase in phases:
        ax.axvline(phase, color="white", lw=1.1, ls="--", alpha=0.92)
        ax.axvline(phase, color="black", lw=0.45, ls="--", alpha=0.7)
        ax.text(
            phase,
            1.02,
            f"Io ~{phase:g} deg",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    if len(phases) >= 2:
        left, right = sorted(phases[:2])
        ax.axvspan(left, right, color="black", alpha=0.045, lw=0)


def plot_heatmaps(summary: pd.DataFrame, dataset: str, out_dir: Path, expected_phases: list[float]) -> list[Path]:
    paths = []
    ds = summary[summary["dataset"].eq(dataset)].copy()
    for value_col, title, cmap, filename, label, centered in [
        (
            "high_tail_fraction",
            "high-power fraction",
            "magma",
            "high_tail_fraction",
            "fraction with daily z > threshold",
            False,
        ),
        (
            "high_tail_fraction_minus_frequency_median",
            "high-power fraction relative to each frequency median",
            "coolwarm",
            "phase_excess_high_tail_fraction",
            "phase-bin high-tail fraction minus frequency median",
            True,
        ),
        (
            "median_daily_z",
            "median daily-normalized log power",
            "coolwarm",
            "median_daily_z",
            "median daily z",
            True,
        ),
    ]:
        fig, axes = plt.subplots(2, 1, figsize=(12.5, 8.0), sharex=True)
        im = None
        for ax, antenna in zip(axes, ["rv1_coarse", "rv2_coarse"]):
            mat = _pivot(ds, antenna, value_col)
            if mat.empty:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center")
                continue
            arr = mat.to_numpy(dtype=float)
            finite = arr[np.isfinite(arr)]
            if centered:
                vmax = float(np.nanquantile(np.abs(finite), 0.97)) if len(finite) else 1.0
                norm = TwoSlopeNorm(vcenter=0.0, vmin=-max(vmax, 0.02), vmax=max(vmax, 0.02))
            else:
                norm = Normalize(vmin=0.0, vmax=max(0.02, float(np.nanquantile(finite, 0.98))) if len(finite) else 0.1)
            im = ax.imshow(
                arr,
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                extent=[
                    float(mat.columns.min() - 0.5),
                    float(mat.columns.max() + 0.5),
                    float(mat.index.min()),
                    float(mat.index.max()),
                ],
                cmap=cmap,
                norm=norm,
            )
            _decorate_io_axis(ax, expected_phases)
            ax.set_yscale("log")
            ax.set_yticks(mat.index.to_list())
            ax.get_yaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
            ax.set_ylabel("frequency (MHz)")
            ax.set_title(ANTENNA_LABEL.get(antenna, antenna))
        axes[-1].set_xlim(0, 360)
        axes[-1].set_xlabel("Io phase (deg)")
        fig.suptitle(f"Jupiter {dataset}: Io phase-frequency {title}")
        if im is not None:
            fig.colorbar(im, ax=axes.tolist(), pad=0.012, label=label)
        path = out_dir / f"jupiter_{dataset}_io_phase_frequency_{filename}.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def plot_phase_profiles(summary: pd.DataFrame, dataset: str, out_dir: Path, expected_phases: list[float]) -> Path:
    ds = summary[summary["dataset"].eq(dataset)].copy()
    freqs = sorted(ds["frequency_mhz"].dropna().unique())
    fig, axes = plt.subplots(len(freqs), 2, figsize=(13.0, max(2.2 * len(freqs), 7.0)), sharex=True, sharey=False)
    axes = np.asarray(axes)
    for row_i, freq in enumerate(freqs):
        for col_i, antenna in enumerate(["rv1_coarse", "rv2_coarse"]):
            ax = axes[row_i, col_i]
            sub = ds[ds["antenna"].astype(str).eq(antenna) & np.isclose(ds["frequency_mhz"].astype(float), float(freq))]
            sub = sub.sort_values("io_phase_bin_deg")
            ax.plot(sub["io_phase_bin_deg"], sub["high_tail_fraction"], marker="o", ms=2.8, lw=1.1)
            _decorate_io_axis(ax, expected_phases)
            ax.set_xlim(0, 360)
            ax.axhline(sub["high_tail_fraction"].median(), color="0.35", lw=0.7, ls=":")
            if row_i == 0:
                ax.set_title(ANTENNA_LABEL.get(antenna, antenna))
            if col_i == 0:
                ax.set_ylabel(f"{freq:.2f} MHz\nhigh-tail frac.")
        axes[row_i, 1].set_ylabel("")
    axes[-1, 0].set_xlabel("Io phase (deg)")
    axes[-1, 1].set_xlabel("Io phase (deg)")
    fig.suptitle(f"Jupiter {dataset}: high-power fraction versus Io phase")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = out_dir / f"jupiter_{dataset}_io_phase_profiles_by_frequency.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_high_power_scatter(
    df: pd.DataFrame,
    dataset: str,
    out_dir: Path,
    high_z: float,
    seed: int,
    max_points: int,
    expected_phases: list[float],
) -> Path:
    sub = df[df["dataset"].eq(dataset) & (df["daily_z_log_power"] > float(high_z))].copy()
    rng = np.random.default_rng(seed)
    if len(sub) > int(max_points):
        sub = sub.iloc[rng.choice(len(sub), size=int(max_points), replace=False)].copy()
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 8.0), sharex=True, sharey=True)
    sc = None
    for ax, antenna in zip(axes, ["rv1_coarse", "rv2_coarse"]):
        a = sub[sub["antenna"].astype(str).eq(antenna)]
        if a.empty:
            ax.text(0.5, 0.5, "no high-power samples", transform=ax.transAxes, ha="center", va="center")
            continue
        sc = ax.scatter(
            a["io_phase_spice_deg"],
            a["frequency_mhz"],
            c=a["daily_z_log_power"].clip(upper=8),
            s=6,
            alpha=0.42,
            cmap="magma",
            vmin=float(high_z),
            vmax=8,
            rasterized=True,
        )
        _decorate_io_axis(ax, expected_phases)
        ax.set_yscale("log")
        ax.set_yticks(sorted(df["frequency_mhz"].unique()))
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
        ax.set_ylabel("frequency (MHz)")
        ax.set_title(ANTENNA_LABEL.get(antenna, antenna))
    axes[-1].set_xlim(0, 360)
    axes[-1].set_xlabel("Io phase (deg)")
    fig.suptitle(f"Jupiter {dataset}: individual high-power samples in Io phase-frequency space")
    if sc is not None:
        fig.colorbar(sc, ax=axes.tolist(), pad=0.012, label="daily-normalized log power")
    path = out_dir / f"jupiter_{dataset}_io_phase_frequency_high_power_scatter.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_collapsed_phase_profiles(
    df: pd.DataFrame,
    dataset: str,
    out_dir: Path,
    phase_bin_deg: float,
    high_z: float,
    expected_phases: list[float],
) -> Path:
    ds = df[df["dataset"].eq(dataset)].copy()
    ds["io_phase_bin_deg"] = (
        np.floor(ds["io_phase_spice_deg"] / float(phase_bin_deg)) * float(phase_bin_deg)
        + 0.5 * float(phase_bin_deg)
    )
    ds["high_power"] = ds["daily_z_log_power"] > float(high_z)
    ds["frequency_group"] = np.where(ds["frequency_mhz"] <= 2.20, "0.45-2.20 MHz", "3.93-9.18 MHz")
    prof = (
        ds.groupby(["antenna", "frequency_group", "io_phase_bin_deg"], sort=True)
        .agg(high_tail_fraction=("high_power", "mean"), n_samples=("high_power", "size"))
        .reset_index()
    )
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.0), sharex=True, sharey=False)
    colors = {"0.45-2.20 MHz": "tab:blue", "3.93-9.18 MHz": "tab:orange"}
    for ax, antenna in zip(axes, ["rv1_coarse", "rv2_coarse"]):
        for group, sub in prof[prof["antenna"].astype(str).eq(antenna)].groupby("frequency_group", sort=True):
            sub = sub.sort_values("io_phase_bin_deg")
            x = sub["io_phase_bin_deg"].to_numpy(dtype=float)
            y = sub["high_tail_fraction"].to_numpy(dtype=float)
            color = colors.get(str(group), None)
            ax.plot(x, y, marker="o", ms=2.8, lw=0.8, alpha=0.25, color=color)
            ax.plot(x, circular_rolling_mean(y, width=3), lw=2.0, color=color, label=f"{group}, 30 deg smooth")
        _decorate_io_axis(ax, expected_phases)
        ax.axhline(
            prof[prof["antenna"].astype(str).eq(antenna)]["high_tail_fraction"].median(),
            color="0.35",
            lw=0.7,
            ls=":",
        )
        ax.set_xlim(0, 360)
        ax.set_ylabel("high-tail fraction")
        ax.set_title(ANTENNA_LABEL.get(antenna, antenna))
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Io phase (deg)")
    fig.suptitle(f"Jupiter {dataset}: Io-phase profile collapsed into broad frequency ranges")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = out_dir / f"jupiter_{dataset}_io_phase_profiles_collapsed_frequency_ranges.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_guide(out_dir: Path, paths: list[Path], config: dict[str, object]) -> Path:
    lines = [
        "# Jupiter Io-Phase / Frequency Diagnostics",
        "",
        "These plots use Io phase as the primary horizontal coordinate, following the Jupiter-radio guidance that hectometric/decametric emission should be strongly Io modulated.",
        "",
        "Expected qualitative behavior from the expert note:",
        "",
        "- hectometric/decametric emission: blobby structure in Io phase-frequency space, often strongest near preferred Io phases;",
        "- vertex arcs, if present, may appear as curved parentheses-like structures in individual phase-frequency scatter;",
        "- kilometric/auroral emission would be more phase independent and broadband.",
        "",
        "## Plot Conventions",
        "",
        f"- Dashed vertical lines mark the configured expected Io phases: `{config['expected_phases_deg']}` degrees.",
        "- The light gray vertical band spans the first two configured phase markers as a visual reference only.",
        "- Heatmaps show all Jupiter-visible RAE-2 samples unless the filename says `historical_active_windows`.",
        "- `high_tail_fraction` means the fraction of samples in a phase/frequency bin with daily-normalized log power above the configured threshold.",
        "- The collapsed phase-profile plots show faint unsmoothed 10-degree bins plus a thicker 30-degree circular-smoothed curve.",
        "",
        "## Configuration",
        "",
        *[f"- `{k}`: `{v}`" for k, v in config.items()],
        "",
        "## Files",
        "",
        *[f"- `{p}`" for p in paths],
    ]
    path = out_dir / "jupiter_io_phase_frequency_diagnostics_guide.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--historical-active", type=Path, default=DEFAULT_HISTORICAL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--phase-bin-deg", type=float, default=10.0)
    parser.add_argument("--high-z", type=float, default=2.5)
    parser.add_argument("--min-count-per-bin", type=int, default=20)
    parser.add_argument("--max-scatter-points", type=int, default=45000)
    parser.add_argument("--expected-phases", type=float, nargs="+", default=[80.0, 240.0])
    parser.add_argument("--seed", type=int, default=20260610)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_samples = read_samples(args.samples, "all_jupiter_visible")
    pieces = [all_samples]
    if args.historical_active.exists():
        pieces.append(read_samples(args.historical_active, "historical_active_windows"))
    samples = pd.concat(pieces, ignore_index=True)
    summary = phase_bin_summary(
        samples,
        phase_bin_deg=float(args.phase_bin_deg),
        high_z=float(args.high_z),
        min_count=int(args.min_count_per_bin),
    )
    summary.to_csv(args.out_dir / "jupiter_io_phase_frequency_binned_summary.csv", index=False)

    paths: list[Path] = []
    expected_phases = [float(p) for p in args.expected_phases]
    for dataset in summary["dataset"].drop_duplicates().tolist():
        paths.extend(plot_heatmaps(summary, dataset, args.out_dir, expected_phases=expected_phases))
        paths.append(plot_phase_profiles(summary, dataset, args.out_dir, expected_phases=expected_phases))
        paths.append(
            plot_high_power_scatter(
                samples,
                dataset,
                args.out_dir,
                high_z=float(args.high_z),
                seed=int(args.seed),
                max_points=int(args.max_scatter_points),
                expected_phases=expected_phases,
            )
        )
        paths.append(
            plot_collapsed_phase_profiles(
                samples,
                dataset,
                args.out_dir,
                phase_bin_deg=float(args.phase_bin_deg),
                high_z=float(args.high_z),
                expected_phases=expected_phases,
            )
        )

    config = {
        "samples": str(args.samples),
        "historical_active": str(args.historical_active),
        "phase_bin_deg": float(args.phase_bin_deg),
        "high_z": float(args.high_z),
        "min_count_per_bin": int(args.min_count_per_bin),
        "max_scatter_points": int(args.max_scatter_points),
        "expected_phases_deg": expected_phases,
        "seed": int(args.seed),
    }
    guide = write_guide(args.out_dir, paths, config)
    print(guide)


if __name__ == "__main__":
    main()
