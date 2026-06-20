#!/usr/bin/env python
"""Direct Io-phase/frequency signal maps for Jupiter.

This is the deliberately visual diagnostic: Io phase on the x-axis,
frequency on the y-axis, and measured RAE-2 signal in each cell.  It excludes
0.45 MHz by default and compares the real Io phase assignment with shifted
phase controls that preserve the RAE-2 signal times.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.constants import FREQUENCY_MAP_MHZ  # noqa: E402
from rylevonberg.util import ensure_dir, software_versions, write_json  # noqa: E402
from scripts.run_jupiter_source_box_direct_detection import (  # noqa: E402
    ANTENNAS,
    ANTENNA_LABEL,
    SOURCE_BOXES,
    merge_geometry,
    read_clean_npy_subset,
)


DEFAULT_CLEAN_NPY = ROOT / "outputs/control_survey_earth_sun_postnov1974_v2/01_ingest/cleaned_timeseries.npy"
DEFAULT_GEOMETRY = ROOT / "outputs/jupiter_phase_pattern_survey_v1/jupiter_spice_visibility_geometry_grid.csv"
DEFAULT_OUT = ROOT / "outputs/jupiter_io_phase_signal_maps_v1"


def load_geometry(path: Path) -> pd.DataFrame:
    cols = [
        "time",
        "jupiter_visible_by_moon",
        "earth_visible_by_moon",
        "jupiter_limb_angle_deg",
        "earth_limb_angle_deg",
        "jupiter_cml_spice_deg",
        "io_phase_spice_deg",
        "maser_zarka_io_score",
    ]
    geom = pd.read_csv(path, usecols=cols, parse_dates=["time"], low_memory=False)
    geom = geom.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    for col in ["jupiter_visible_by_moon", "earth_visible_by_moon"]:
        geom[col] = geom[col].astype(bool)
    return geom


def expected_io_bands(mode: str = "broad", center_half_width_deg: float = 20.0) -> list[tuple[str, float, float]]:
    """Return Io-phase windows used for overlays and summaries."""
    broad = [
        ("Io-B/D", 80.0, 130.0),
        ("Io-A/C", 205.0, 260.0),
    ]
    if mode == "broad":
        return broad
    if mode != "centered":
        raise ValueError(f"unknown expected Io-band mode: {mode}")
    half_width = float(center_half_width_deg)
    return [(label, 0.5 * (lo + hi) - half_width, 0.5 * (lo + hi) + half_width) for label, lo, hi in broad]


def phase_in_windows(values: pd.Series | np.ndarray, windows: list[tuple[float, float]]) -> np.ndarray:
    vals = np.asarray(values, dtype=float) % 360.0
    mask = np.zeros(len(vals), dtype=bool)
    for lo, hi in windows:
        lo = float(lo) % 360.0
        hi = float(hi) % 360.0
        if lo <= hi:
            mask |= (vals >= lo) & (vals <= hi)
        else:
            mask |= (vals >= lo) | (vals <= hi)
    return mask


def annotate_phase_bins(samples: pd.DataFrame, phase_bin_deg: float, high_factor: float) -> pd.DataFrame:
    out = samples.copy()
    out = out[out["jupiter_visible_by_moon"].astype(bool)].copy()
    out["io_phase_spice_deg"] = pd.to_numeric(out["io_phase_spice_deg"], errors="coerce") % 360.0
    out["io_phase_bin_deg"] = (
        np.floor(out["io_phase_spice_deg"] / float(phase_bin_deg)) * float(phase_bin_deg)
        + 0.5 * float(phase_bin_deg)
    )
    out["factor_high"] = out["daily_log10_residual"] >= float(np.log10(float(high_factor)))
    return out.dropna(subset=["io_phase_bin_deg"]).reset_index(drop=True)


def summarize_phase_frequency(samples: pd.DataFrame, phase_bin_deg: float, high_factor: float, min_count: int) -> pd.DataFrame:
    work = annotate_phase_bins(samples, phase_bin_deg=phase_bin_deg, high_factor=high_factor)
    summary = (
        work.groupby(["antenna", "frequency_band", "frequency_mhz", "io_phase_bin_deg"], sort=True)
        .agg(
            n_samples=("daily_log10_residual", "size"),
            median_daily_residual=("daily_log10_residual", "median"),
            q75_daily_residual=("daily_log10_residual", lambda x: float(np.nanquantile(x, 0.75))),
            factor_high_fraction=("factor_high", "mean"),
            median_raw_log10_power=("log10_power", "median"),
        )
        .reset_index()
    )
    mask_low = summary["n_samples"] < int(min_count)
    for col in ["median_daily_residual", "q75_daily_residual", "factor_high_fraction", "median_raw_log10_power"]:
        summary.loc[mask_low, col] = np.nan
    return summary


def shifted_geometry(geom: pd.DataFrame, shift_days: float) -> pd.DataFrame:
    shifted = geom.copy()
    shifted["time"] = shifted["time"] - pd.Timedelta(days=float(shift_days))
    return shifted.sort_values("time").reset_index(drop=True)


def shifted_control_summary(
    samples: pd.DataFrame,
    geom: pd.DataFrame,
    shift_days: list[float],
    tolerance_s: float,
    phase_bin_deg: float,
    high_factor: float,
    min_count: int,
) -> pd.DataFrame:
    rows = []
    base_cols = ["time", "frequency_band", "frequency_mhz", "antenna", "power", "log10_power", "date", "daily_log10_residual"]
    base_samples = samples[base_cols].copy()
    for shift in shift_days:
        shifted = merge_geometry(base_samples, shifted_geometry(geom, shift), tolerance_s=tolerance_s, pad_deg=0.0)
        summary = summarize_phase_frequency(
            shifted,
            phase_bin_deg=phase_bin_deg,
            high_factor=high_factor,
            min_count=min_count,
        )
        summary["shift_days"] = float(shift)
        rows.append(summary)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def combine_real_and_shifted(real: pd.DataFrame, shifted: pd.DataFrame) -> pd.DataFrame:
    keys = ["antenna", "frequency_band", "frequency_mhz", "io_phase_bin_deg"]
    if shifted.empty:
        return real.copy()
    control = (
        shifted.groupby(keys, sort=True)
        .agg(
            control_median_daily_residual=("median_daily_residual", "median"),
            control_factor_high_fraction=("factor_high_fraction", "median"),
            control_factor_high_q10=("factor_high_fraction", lambda x: float(np.nanquantile(x, 0.10))),
            control_factor_high_q90=("factor_high_fraction", lambda x: float(np.nanquantile(x, 0.90))),
            n_control_shifts=("shift_days", "nunique"),
        )
        .reset_index()
    )
    merged = real.merge(control, on=keys, how="left")
    merged["median_daily_residual_minus_shifted"] = (
        merged["median_daily_residual"] - merged["control_median_daily_residual"]
    )
    merged["factor_high_fraction_minus_shifted"] = (
        merged["factor_high_fraction"] - merged["control_factor_high_fraction"]
    )
    return merged


def decorate_expected_io_bands(ax: plt.Axes, bands: list[tuple[str, float, float]]) -> None:
    colors = {"Io-B/D": "#d95f02", "Io-A/C": "#1b9e77"}
    for label, lo, hi in bands:
        ax.axvspan(lo, hi, color=colors[label], alpha=0.10, lw=0)
        ax.axvline(lo, color=colors[label], lw=1.0, alpha=0.9)
        ax.axvline(hi, color=colors[label], lw=1.0, alpha=0.9)
        ax.text(
            (lo + hi) / 2.0,
            1.015,
            label,
            color=colors[label],
            fontsize=8,
            ha="center",
            va="bottom",
            transform=ax.get_xaxis_transform(),
        )


def matrix_for(summary: pd.DataFrame, antenna: str, value_col: str, phase_bin_deg: float) -> pd.DataFrame:
    sub = summary[summary["antenna"].eq(antenna)].copy()
    phases = np.arange(0.5 * phase_bin_deg, 360.0, phase_bin_deg)
    freqs = sorted(sub["frequency_mhz"].dropna().unique())
    mat = sub.pivot_table(index="frequency_mhz", columns="io_phase_bin_deg", values=value_col, aggfunc="mean")
    return mat.reindex(index=freqs, columns=phases)


def plot_heatmap_pair(
    summary: pd.DataFrame,
    out_dir: Path,
    value_col: str,
    title: str,
    colorbar_label: str,
    filename: str,
    phase_bin_deg: float,
    centered: bool,
    cmap: str,
    expected_bands: list[tuple[str, float, float]],
) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(12.8, 7.4), sharex=True, sharey=True)
    im = None
    for ax, antenna in zip(axes, ANTENNAS):
        mat = matrix_for(summary, antenna=antenna, value_col=value_col, phase_bin_deg=phase_bin_deg)
        vals = mat.to_numpy(dtype=float)
        finite = vals[np.isfinite(vals)]
        if centered:
            vmax = max(0.02, float(np.nanpercentile(np.abs(finite), 98))) if finite.size else 0.05
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        else:
            vmax = max(0.01, float(np.nanpercentile(finite, 98))) if finite.size else 0.05
            norm = Normalize(vmin=0.0, vmax=vmax)
        im = ax.imshow(
            vals,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=[
                float(mat.columns.min() - 0.5 * phase_bin_deg),
                float(mat.columns.max() + 0.5 * phase_bin_deg),
                -0.5,
                len(mat.index) - 0.5,
            ],
            cmap=cmap,
            norm=norm,
        )
        ax.set_yticks(np.arange(len(mat.index)))
        ax.set_yticklabels([f"{freq:.2f}" for freq in mat.index])
        ax.set_ylabel("frequency (MHz)")
        ax.set_title(ANTENNA_LABEL.get(antenna, antenna), loc="left")
        ax.grid(True, axis="x", color="white", alpha=0.12, lw=0.35)
        decorate_expected_io_bands(ax, expected_bands)
    axes[-1].set_xlim(0, 360)
    axes[-1].set_xticks(np.arange(0, 361, 30))
    axes[-1].set_xlabel("Io phase (deg)")
    fig.suptitle(title)
    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.08, top=0.91, hspace=0.18)
    if im is not None:
        cax = fig.add_axes([0.90, 0.18, 0.018, 0.64])
        fig.colorbar(im, cax=cax, label=colorbar_label)
    path = out_dir / filename
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def circular_rolling_mean(y: np.ndarray, width: int = 3) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if len(y) == 0 or width <= 1:
        return y
    half = int(width) // 2
    padded = np.concatenate([y[-half:], y, y[:half]])
    return np.asarray([np.nanmean(padded[i - half : i + half + 1]) for i in range(half, half + len(y))])


def plot_profiles(
    summary: pd.DataFrame,
    out_dir: Path,
    phase_bin_deg: float,
    high_factor: float,
    expected_bands: list[tuple[str, float, float]],
) -> Path:
    freqs = sorted(summary["frequency_mhz"].dropna().unique())
    fig, axes = plt.subplots(len(freqs), 2, figsize=(13.0, max(2.1 * len(freqs), 8.5)), sharex=True)
    axes = np.atleast_2d(axes)
    for row, freq in enumerate(freqs):
        for col, antenna in enumerate(ANTENNAS):
            ax = axes[row, col]
            sub = summary[summary["antenna"].eq(antenna) & np.isclose(summary["frequency_mhz"], float(freq))].sort_values(
                "io_phase_bin_deg"
            )
            x = sub["io_phase_bin_deg"].to_numpy(dtype=float)
            y = sub["factor_high_fraction"].to_numpy(dtype=float)
            yc = sub["control_factor_high_fraction"].to_numpy(dtype=float)
            ax.plot(x, yc, color="0.45", lw=1.1, alpha=0.9, label="shifted-control median" if row == 0 and col == 0 else None)
            ax.plot(x, y, color="#386cb0" if antenna == "rv1_coarse" else "#bf5b17", lw=0.8, alpha=0.35)
            ax.plot(
                x,
                circular_rolling_mean(y, width=3),
                color="#386cb0" if antenna == "rv1_coarse" else "#bf5b17",
                lw=1.7,
                label="real, 3-bin smooth" if row == 0 and col == 0 else None,
            )
            decorate_expected_io_bands(ax, expected_bands)
            ax.set_xlim(0, 360)
            ax.grid(True, color="0.9", lw=0.45)
            if row == 0:
                ax.set_title(ANTENNA_LABEL[antenna])
            if col == 0:
                ax.set_ylabel(f"{freq:.2f} MHz\nfrac >= {high_factor:g}x")
    axes[-1, 0].set_xlabel("Io phase (deg)")
    axes[-1, 1].set_xlabel("Io phase (deg)")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Jupiter-visible samples: high-power fraction versus Io phase by frequency")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = out_dir / "jupiter_io_phase_profiles_factor_high_by_frequency.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_expected_band_contrast(band_summary: pd.DataFrame, out_dir: Path) -> Path:
    order = []
    for freq in sorted(band_summary["frequency_mhz"].dropna().unique()):
        for antenna in ANTENNAS:
            sub = band_summary[band_summary["antenna"].eq(antenna) & np.isclose(band_summary["frequency_mhz"], float(freq))]
            if not sub.empty:
                order.append((antenna, float(freq)))
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 7.4), sharex=True)
    metrics = [
        ("expected_minus_off_factor_high_fraction", "selected Io windows - off-phase high fraction"),
        ("expected_phase_mean_factor_high_excess_vs_shifted", "selected Io windows - shifted-control high fraction"),
    ]
    colors = {"rv1_coarse": "#386cb0", "rv2_coarse": "#bf5b17"}
    for ax, (metric, ylabel) in zip(axes, metrics):
        for xpos, (antenna, freq) in enumerate(order):
            sub = band_summary[band_summary["antenna"].eq(antenna) & np.isclose(band_summary["frequency_mhz"], freq)]
            if sub.empty:
                continue
            ax.scatter(
                [xpos],
                [float(sub[metric].iloc[0])],
                marker="D",
                s=58,
                color=colors.get(antenna, "black"),
                edgecolor="black",
                linewidth=0.35,
            )
        ax.axhline(0.0, color="0.25", lw=0.9)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", color="0.9", lw=0.45)
    labels = [f"{freq:.2f}\n{ANTENNA_LABEL[antenna]}" for antenna, freq in order]
    axes[-1].set_xticks(np.arange(len(order)))
    axes[-1].set_xticklabels(labels)
    fig.suptitle("Selected Io-phase window contrast by channel")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = out_dir / "jupiter_io_phase_centered_window_contrast_by_channel.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_highband_overlay_profiles(
    summary: pd.DataFrame,
    out_dir: Path,
    expected_bands: list[tuple[str, float, float]],
    high_factor: float,
) -> list[Path]:
    high = summary[summary["frequency_mhz"].ge(3.0)].copy()
    freqs = sorted(high["frequency_mhz"].dropna().unique())
    colors = {
        3.93: "#1b9e77",
        4.70: "#d95f02",
        6.55: "#7570b3",
        9.18: "#e7298a",
    }
    plot_specs = [
        (
            "factor_high_fraction",
            f"fraction >= {high_factor:g}x same-day median",
            "jupiter_io_phase_1d_highband_factor_high_fraction.png",
            False,
        ),
        (
            "factor_high_fraction_minus_shifted",
            f"fraction >= {high_factor:g}x minus shifted controls",
            "jupiter_io_phase_1d_highband_factor_high_fraction_minus_shifted.png",
            True,
        ),
        (
            "median_daily_residual",
            "median daily residual (dex)",
            "jupiter_io_phase_1d_highband_median_daily_residual.png",
            True,
        ),
    ]
    paths = []
    for value_col, ylabel, filename, centered_y in plot_specs:
        fig, axes = plt.subplots(2, 1, figsize=(12.5, 7.2), sharex=True, sharey=centered_y)
        values_for_limits = []
        for ax, antenna in zip(axes, ANTENNAS):
            for freq in freqs:
                sub = high[high["antenna"].eq(antenna) & np.isclose(high["frequency_mhz"], float(freq))].sort_values(
                    "io_phase_bin_deg"
                )
                if sub.empty:
                    continue
                x = sub["io_phase_bin_deg"].to_numpy(dtype=float)
                y = sub[value_col].to_numpy(dtype=float)
                color = colors.get(round(float(freq), 2), None)
                ax.plot(x, y, marker="o", ms=3.0, lw=1.5, color=color, label=f"{freq:.2f} MHz")
                values_for_limits.extend(y[np.isfinite(y)].tolist())
            if value_col == "factor_high_fraction" and "control_factor_high_fraction" in high.columns:
                for freq in freqs:
                    sub = high[high["antenna"].eq(antenna) & np.isclose(high["frequency_mhz"], float(freq))].sort_values(
                        "io_phase_bin_deg"
                    )
                    if sub.empty:
                        continue
                    x = sub["io_phase_bin_deg"].to_numpy(dtype=float)
                    yc = sub["control_factor_high_fraction"].to_numpy(dtype=float)
                    color = colors.get(round(float(freq), 2), "0.5")
                    ax.plot(x, yc, lw=0.9, ls="--", alpha=0.55, color=color)
            decorate_expected_io_bands(ax, expected_bands)
            ax.grid(True, color="0.9", lw=0.45)
            ax.set_xlim(0, 360)
            ax.set_ylabel(ylabel)
            ax.set_title(ANTENNA_LABEL.get(antenna, antenna), loc="left")
            if centered_y:
                ax.axhline(0.0, color="0.25", lw=0.8)
        axes[-1].set_xticks(np.arange(0, 361, 30))
        axes[-1].set_xlabel("Io phase (deg)")
        axes[0].legend(frameon=False, ncol=4, fontsize=8, loc="upper right")
        if centered_y and values_for_limits:
            vals = np.asarray(values_for_limits, dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                lim = max(0.01, float(np.nanmax(np.abs(vals))))
                lim *= 1.08
                for ax in axes:
                    ax.set_ylim(-lim, lim)
        fig.suptitle("High-band 1D Io-phase profiles")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        path = out_dir / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def expected_band_summary(summary: pd.DataFrame, bands: list[tuple[str, float, float]]) -> pd.DataFrame:
    rows = []
    windows = [(lo, hi) for _, lo, hi in bands]
    for (antenna, band, freq), grp in summary.groupby(["antenna", "frequency_band", "frequency_mhz"], sort=True):
        expected = phase_in_windows(grp["io_phase_bin_deg"], windows)
        off = ~expected
        rows.append(
            {
                "antenna": antenna,
                "antenna_label": ANTENNA_LABEL.get(str(antenna), str(antenna)),
                "frequency_band": int(band),
                "frequency_mhz": float(freq),
                "expected_phase_mean_factor_high_fraction": float(grp.loc[expected, "factor_high_fraction"].mean()),
                "off_phase_mean_factor_high_fraction": float(grp.loc[off, "factor_high_fraction"].mean()),
                "expected_minus_off_factor_high_fraction": float(
                    grp.loc[expected, "factor_high_fraction"].mean() - grp.loc[off, "factor_high_fraction"].mean()
                ),
                "expected_phase_mean_factor_high_excess_vs_shifted": float(
                    grp.loc[expected, "factor_high_fraction_minus_shifted"].mean()
                ),
                "off_phase_mean_factor_high_excess_vs_shifted": float(
                    grp.loc[off, "factor_high_fraction_minus_shifted"].mean()
                ),
                "expected_phase_mean_median_daily_residual": float(grp.loc[expected, "median_daily_residual"].mean()),
                "off_phase_mean_median_daily_residual": float(grp.loc[off, "median_daily_residual"].mean()),
            }
        )
    return pd.DataFrame(rows)


def write_report(
    out_dir: Path,
    samples: pd.DataFrame,
    combined: pd.DataFrame,
    band_summary: pd.DataFrame,
    paths: list[Path],
    config: dict[str, object],
    expected_bands: list[tuple[str, float, float]],
) -> Path:
    top = band_summary.sort_values("expected_phase_mean_factor_high_excess_vs_shifted", ascending=False).head(16)
    lines = [
        "# Jupiter Io-Phase / Frequency Signal Maps",
        "",
        "These plots put Io phase on the x-axis and frequency on the y-axis, with each cell colored by direct RAE-2 signal summaries. The default run excludes 0.45 MHz.",
        "",
        "## Expected Io-Phase Bands",
        "",
        "The shaded bands are the Io-phase windows used for this run's overlays and expected-band summary:",
        "",
        *[f"- `{label}`: `{lo:g}-{hi:g} deg`" for label, lo, hi in expected_bands],
        "",
        "Individual source-box Io ranges used for that collapse:",
        "",
        *[f"- `{box.name}`: `{box.io_lo:g}-{box.io_hi:g} deg`" for box in SOURCE_BOXES],
        "",
        "## Configuration",
        "",
        *[f"- `{k}`: `{v}`" for k, v in config.items() if k != "software_versions"],
        "",
        "## Coverage",
        "",
        f"- merged samples: `{len(samples)}`",
        f"- Jupiter-visible samples: `{int(samples['jupiter_visible_by_moon'].astype(bool).sum())}`",
        "",
        "## Expected-Band Summary",
        "",
        top[
            [
                "antenna_label",
                "frequency_mhz",
                "expected_minus_off_factor_high_fraction",
                "expected_phase_mean_factor_high_excess_vs_shifted",
                "expected_phase_mean_factor_high_fraction",
                "off_phase_mean_factor_high_fraction",
            ]
        ].to_string(index=False),
        "",
        "## What Would Be Convincing",
        "",
        "- A vertical warm band near `80-130 deg` or `205-260 deg` that recurs across adjacent frequencies and preferably both antennas.",
        "- The same phase band should remain positive in the real-minus-shifted-control map, not just the direct signal map.",
        "- The effect should be visible in high bands such as `3.93-9.18 MHz`; lower bands can be context, but the case should not rest on `0.45 MHz`.",
        "- The 1D phase profiles should show peaks in the shaded bands, not equally strong peaks everywhere else.",
        "",
        "## Files",
        "",
        *[f"- `{p}`" for p in paths],
    ]
    path = out_dir / "jupiter_io_phase_signal_maps_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-npy", type=Path, default=DEFAULT_CLEAN_NPY)
    parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--frequency-band", type=int, nargs="+", default=[2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument("--phase-bin-deg", type=float, default=10.0)
    parser.add_argument("--high-factor", type=float, default=2.0)
    parser.add_argument("--min-count-per-bin", type=int, default=25)
    parser.add_argument("--geometry-tolerance-s", type=float, default=360.0)
    parser.add_argument("--shift-days", type=float, nargs="+", default=[-7, -5, -3, -1, 1, 3, 5, 7])
    parser.add_argument("--expected-band-mode", choices=["broad", "centered"], default="broad")
    parser.add_argument("--center-window-half-width-deg", type=float, default=20.0)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    expected_bands = expected_io_bands(
        mode=str(args.expected_band_mode),
        center_half_width_deg=float(args.center_window_half_width_deg),
    )
    config = {
        "clean_npy": str(args.clean_npy),
        "geometry": str(args.geometry),
        "frequency_bands": [int(v) for v in args.frequency_band],
        "frequencies_mhz": [FREQUENCY_MAP_MHZ.get(int(v), np.nan) for v in args.frequency_band],
        "excluded_frequency_note": "0.45 MHz is excluded by default.",
        "phase_bin_deg": float(args.phase_bin_deg),
        "high_factor": float(args.high_factor),
        "min_count_per_bin": int(args.min_count_per_bin),
        "geometry_tolerance_s": float(args.geometry_tolerance_s),
        "shift_days": [float(v) for v in args.shift_days],
        "expected_band_mode": str(args.expected_band_mode),
        "center_window_half_width_deg": float(args.center_window_half_width_deg),
        "expected_io_bands": expected_bands,
        "software_versions": software_versions(),
    }
    write_json(out_dir / "run_config.json", config)

    print("Loading samples...", flush=True)
    samples = read_clean_npy_subset(args.clean_npy, [int(v) for v in args.frequency_band])
    geom = load_geometry(args.geometry)
    print("Merging real Jupiter geometry...", flush=True)
    samples = merge_geometry(samples, geom, tolerance_s=float(args.geometry_tolerance_s), pad_deg=0.0)
    print("Summarizing real Io-phase/frequency map...", flush=True)
    real = summarize_phase_frequency(
        samples,
        phase_bin_deg=float(args.phase_bin_deg),
        high_factor=float(args.high_factor),
        min_count=int(args.min_count_per_bin),
    )
    print("Building shifted Io-phase controls...", flush=True)
    shifted = shifted_control_summary(
        samples,
        geom,
        shift_days=[float(v) for v in args.shift_days],
        tolerance_s=float(args.geometry_tolerance_s),
        phase_bin_deg=float(args.phase_bin_deg),
        high_factor=float(args.high_factor),
        min_count=int(args.min_count_per_bin),
    )
    combined = combine_real_and_shifted(real, shifted)
    real.to_csv(out_dir / "jupiter_io_phase_frequency_real_summary.csv", index=False)
    shifted.to_csv(out_dir / "jupiter_io_phase_frequency_shifted_control_summary.csv", index=False)
    combined.to_csv(out_dir / "jupiter_io_phase_frequency_real_with_shifted_controls.csv", index=False)
    band_summary = expected_band_summary(combined, expected_bands)
    band_summary.to_csv(out_dir / "jupiter_io_phase_expected_band_summary.csv", index=False)

    paths = [
        out_dir / "run_config.json",
        out_dir / "jupiter_io_phase_frequency_real_summary.csv",
        out_dir / "jupiter_io_phase_frequency_shifted_control_summary.csv",
        out_dir / "jupiter_io_phase_frequency_real_with_shifted_controls.csv",
        out_dir / "jupiter_io_phase_expected_band_summary.csv",
    ]
    print("Making plots...", flush=True)
    paths.extend(
        [
            plot_heatmap_pair(
                combined,
                out_dir,
                value_col="median_daily_residual",
                title="Jupiter-visible samples: median daily residual by Io phase and frequency",
                colorbar_label="median log10(power) minus same-day channel median (dex)",
                filename="jupiter_io_phase_frequency_median_daily_residual.png",
                phase_bin_deg=float(args.phase_bin_deg),
                centered=True,
                cmap="coolwarm",
                expected_bands=expected_bands,
            ),
            plot_heatmap_pair(
                combined,
                out_dir,
                value_col="factor_high_fraction",
                title=f"Jupiter-visible samples: fraction >= {float(args.high_factor):g}x same-day median",
                colorbar_label=f"fraction >= {float(args.high_factor):g}x same-day channel median",
                filename="jupiter_io_phase_frequency_factor_high_fraction.png",
                phase_bin_deg=float(args.phase_bin_deg),
                centered=False,
                cmap="magma",
                expected_bands=expected_bands,
            ),
            plot_heatmap_pair(
                combined,
                out_dir,
                value_col="factor_high_fraction_minus_shifted",
                title="Io-phase/frequency signal after subtracting shifted-phase controls",
                colorbar_label="real high fraction minus shifted-control median",
                filename="jupiter_io_phase_frequency_factor_high_fraction_minus_shifted_controls.png",
                phase_bin_deg=float(args.phase_bin_deg),
                centered=True,
                cmap="coolwarm",
                expected_bands=expected_bands,
            ),
            plot_profiles(
                combined,
                out_dir,
                phase_bin_deg=float(args.phase_bin_deg),
                high_factor=float(args.high_factor),
                expected_bands=expected_bands,
            ),
            plot_expected_band_contrast(band_summary, out_dir),
            *plot_highband_overlay_profiles(
                combined,
                out_dir,
                expected_bands=expected_bands,
                high_factor=float(args.high_factor),
            ),
        ]
    )
    report = write_report(
        out_dir,
        samples=samples,
        combined=combined,
        band_summary=band_summary,
        paths=paths,
        config=config,
        expected_bands=expected_bands,
    )
    print(f"Wrote {report}", flush=True)


if __name__ == "__main__":
    main()
