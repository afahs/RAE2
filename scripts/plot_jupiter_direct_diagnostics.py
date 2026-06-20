#!/usr/bin/env python
"""Direct Jupiter diagnostic plots using raw/log-power quantities.

These plots avoid event-rate scoring. They show raw log10 power and simple
log10 residuals relative to same-day channel medians for physically plausible
Jupiter bands, with Io/MASER and occultation context overlaid.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.constants import FREQUENCY_MAP_MHZ  # noqa: E402
from rylevonberg.util import ensure_dir, software_versions, write_json  # noqa: E402
from scripts.select_jupiter_expected_active_times import phase_in_windows  # noqa: E402


DEFAULT_CLEAN_NPY = ROOT / "outputs/control_survey_earth_sun_postnov1974_v2/01_ingest/cleaned_timeseries.npy"
DEFAULT_GEOMETRY = ROOT / "outputs/jupiter_phase_pattern_survey_v1/jupiter_spice_visibility_geometry_grid.csv"
DEFAULT_OUT = ROOT / "outputs/jupiter_direct_diagnostic_plots_v1"
ANTENNAS = ["rv1_coarse", "rv2_coarse"]
ANTENNA_LABEL = {"rv1_coarse": "upper V", "rv2_coarse": "lower V"}
ANTENNA_COLOR = {"rv1_coarse": "#386cb0", "rv2_coarse": "#bf5b17"}


def read_clean_npy_subset(path: Path, frequency_bands: list[int]) -> pd.DataFrame:
    arr = np.load(path, allow_pickle=True)
    keep = np.isin(arr["frequency_band"], np.asarray(frequency_bands, dtype=int))
    keep &= np.isin(arr["antenna"], np.asarray(ANTENNAS, dtype=object))
    keep &= arr["is_valid"].astype(bool)
    keep &= np.isfinite(arr["power"].astype(float)) & (arr["power"].astype(float) > 0)
    sub = arr[keep]
    out = pd.DataFrame(
        {
            "time": pd.to_datetime(sub["time"].astype(str)),
            "frequency_band": sub["frequency_band"].astype(int),
            "frequency_mhz": sub["frequency_mhz"].astype(float),
            "antenna": sub["antenna"].astype(str),
            "power": sub["power"].astype(float),
        }
    )
    out["log10_power"] = np.log10(out["power"].to_numpy(dtype=float))
    out["date"] = out["time"].dt.floor("D")
    stats = (
        out.groupby(["date", "antenna", "frequency_band"], sort=True)["log10_power"]
        .median()
        .rename("daily_median_log10_power")
        .reset_index()
    )
    out = out.merge(stats, on=["date", "antenna", "frequency_band"], how="left")
    out["daily_log10_residual"] = out["log10_power"] - out["daily_median_log10_power"]
    return out.sort_values("time").reset_index(drop=True)


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
    return geom.sort_values("time").drop_duplicates("time").reset_index(drop=True)


def merge_geometry(samples: pd.DataFrame, geom: pd.DataFrame, tolerance_s: float) -> pd.DataFrame:
    merged = pd.merge_asof(
        samples.sort_values("time"),
        geom.sort_values("time"),
        on="time",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=float(tolerance_s)),
    )
    merged = merged.dropna(subset=["io_phase_spice_deg", "jupiter_cml_spice_deg", "maser_zarka_io_score"]).copy()
    for col in ["jupiter_visible_by_moon", "earth_visible_by_moon"]:
        merged[col] = merged[col].astype(bool)
    return merged.reset_index(drop=True)


def add_activity_flags(samples: pd.DataFrame, io_windows: list[tuple[float, float]]) -> tuple[pd.DataFrame, dict[str, float]]:
    out = samples.copy()
    visible = out["jupiter_visible_by_moon"].astype(bool)
    score = pd.to_numeric(out["maser_zarka_io_score"], errors="coerce")
    score_visible = score[visible]
    thresholds = {
        "maser_q90_visible": float(score_visible.quantile(0.90)),
        "maser_q95_visible": float(score_visible.quantile(0.95)),
    }
    out["io_window"] = visible & phase_in_windows(out["io_phase_spice_deg"], io_windows)
    out["maser_top10"] = visible & score.ge(thresholds["maser_q90_visible"])
    out["maser_top05"] = visible & score.ge(thresholds["maser_q95_visible"])
    out["io_window_and_maser_top10"] = out["io_window"] & out["maser_top10"]
    return out, thresholds


def contiguous_windows(geom: pd.DataFrame, mask: pd.Series, max_gap_min: float) -> pd.DataFrame:
    selected = geom[mask.to_numpy()].sort_values("time").copy()
    if selected.empty:
        return pd.DataFrame()
    group = selected["time"].diff().dt.total_seconds().div(60).gt(float(max_gap_min)).fillna(True).cumsum()
    rows = []
    for _, grp in selected.groupby(group, sort=True):
        rows.append(
            {
                "start_time": grp["time"].min(),
                "end_time": grp["time"].max(),
                "duration_min": (grp["time"].max() - grp["time"].min()).total_seconds() / 60.0,
                "median_io_phase_deg": float(grp["io_phase_spice_deg"].median()),
                "median_cml_deg": float(grp["jupiter_cml_spice_deg"].median()),
                "median_maser_score": float(grp["maser_zarka_io_score"].median()),
                "n_geometry_points": int(len(grp)),
            }
        )
    return pd.DataFrame(rows)


def plot_io_phase_profiles(samples: pd.DataFrame, out_dir: Path, io_windows: list[tuple[float, float]]) -> list[Path]:
    paths = []
    metrics = [
        ("log10_power", "raw log10(power)", "raw log10(power)", "jupiter_highband_raw_log10_power_vs_io_phase.png"),
        (
            "daily_log10_residual",
            "daily residual (dex)",
            "log10(power) - same-day channel median",
            "jupiter_highband_daily_residual_vs_io_phase.png",
        ),
    ]
    work = samples[samples["jupiter_visible_by_moon"].astype(bool)].copy()
    work["io_bin"] = np.floor(work["io_phase_spice_deg"] / 10.0) * 10.0 + 5.0
    freqs = sorted(work["frequency_mhz"].dropna().unique())
    for metric, ylabel, title_metric, filename in metrics:
        fig, axes = plt.subplots(len(freqs), 2, figsize=(12.5, max(2.4 * len(freqs), 5.0)), sharex=True)
        axes = np.atleast_2d(axes)
        for row, freq in enumerate(freqs):
            for col, antenna in enumerate(ANTENNAS):
                ax = axes[row, col]
                sub = work[work["frequency_mhz"].eq(freq) & work["antenna"].eq(antenna)]
                prof = (
                    sub.groupby("io_bin", sort=True)[metric]
                    .agg(median="median", q25=lambda x: np.quantile(x, 0.25), q75=lambda x: np.quantile(x, 0.75), n="size")
                    .reset_index()
                )
                ax.plot(prof["io_bin"], prof["median"], color=ANTENNA_COLOR[antenna], lw=1.5)
                ax.fill_between(prof["io_bin"], prof["q25"], prof["q75"], color=ANTENNA_COLOR[antenna], alpha=0.18, lw=0)
                for lo, hi in io_windows:
                    ax.axvspan(lo, hi, color="#d95f02", alpha=0.10, lw=0)
                if metric.endswith("residual"):
                    ax.axhline(0, color="0.35", lw=0.8)
                ax.set_title(f"{ANTENNA_LABEL[antenna]} {freq:.2f} MHz", loc="left", fontsize=10)
                ax.set_ylabel(ylabel if col == 0 else "")
                ax.grid(True, color="0.9", lw=0.45)
        axes[-1, 0].set_xlabel("Io phase (deg)")
        axes[-1, 1].set_xlabel("Io phase (deg)")
        for ax in axes[-1, :]:
            ax.set_xlim(0, 360)
            ax.set_xticks(np.arange(0, 361, 60))
        fig.suptitle(f"Jupiter-visible samples: {title_metric} binned by Io phase")
        fig.tight_layout(rect=[0, 0, 1, 0.975])
        path = out_dir / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_active_control_daily_scatter(samples: pd.DataFrame, out_dir: Path) -> Path:
    visible = samples[samples["jupiter_visible_by_moon"].astype(bool)].copy()
    selector = "io_window_and_maser_top10"
    selected = visible[visible[selector]].copy()
    control = visible[~visible[selector]].copy()
    keys = ["date", "antenna", "frequency_band", "frequency_mhz"]
    sel = selected.groupby(keys, sort=True)["daily_log10_residual"].median().rename("active_median_resid").reset_index()
    ctl = control.groupby(keys, sort=True)["daily_log10_residual"].median().rename("control_median_resid").reset_index()
    paired = sel.merge(ctl, on=keys, how="inner")
    paired["active_minus_control_median_resid"] = paired["active_median_resid"] - paired["control_median_resid"]
    paired["channel"] = paired["frequency_mhz"].map(lambda v: f"{v:.2f}") + "\n" + paired["antenna"].map(ANTENNA_LABEL)
    channel_order = []
    for freq in sorted(paired["frequency_mhz"].dropna().unique()):
        for antenna in ANTENNAS:
            label = f"{freq:.2f}\n{ANTENNA_LABEL[antenna]}"
            if label in set(paired["channel"]):
                channel_order.append(label)
    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    rng = np.random.default_rng(12345)
    y_all = paired["active_minus_control_median_resid"].to_numpy(dtype=float)
    lim = min(max(0.06, float(np.nanpercentile(np.abs(y_all[np.isfinite(y_all)]), 99))), 0.6)
    for xpos, label in enumerate(channel_order):
        sub = paired[paired["channel"].eq(label)].copy()
        x = xpos + rng.normal(0.0, 0.055, size=len(sub))
        color = ANTENNA_COLOR["rv1_coarse"] if "upper" in label else ANTENNA_COLOR["rv2_coarse"]
        ax.scatter(x, sub["active_minus_control_median_resid"], s=14, alpha=0.45, color=color)
        q25, med, q75 = np.nanquantile(sub["active_minus_control_median_resid"], [0.25, 0.5, 0.75])
        ax.plot([xpos - 0.24, xpos + 0.24], [med, med], color="black", lw=1.5)
        ax.vlines(xpos, q25, q75, color="black", lw=3.0, alpha=0.75)
        ax.text(xpos, lim * 0.92, f"n={len(sub)}", ha="center", va="top", fontsize=8)
    ax.axhline(0, color="0.25", lw=0.9)
    ax.set_xticks(np.arange(len(channel_order)))
    ax.set_xticklabels(channel_order)
    ax.set_ylim(-lim, lim)
    ax.set_ylabel("active - same-day inactive median residual (dex)")
    ax.set_title("Daily Io-window + MASER-top-10 residuals relative to same-day inactive visible samples")
    ax.grid(True, axis="y", color="0.9", lw=0.45)
    fig.tight_layout()
    path = out_dir / "jupiter_highband_active_minus_control_daily_residuals.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paired.to_csv(out_dir / "jupiter_highband_active_minus_control_daily_residual_points.csv", index=False)
    return path


def plot_active_window_dynamic_spectra(
    samples: pd.DataFrame,
    geom: pd.DataFrame,
    out_dir: Path,
    io_windows: list[tuple[float, float]],
    n_windows: int,
) -> Path | None:
    visible = geom["jupiter_visible_by_moon"].astype(bool)
    score = pd.to_numeric(geom["maser_zarka_io_score"], errors="coerce")
    threshold = float(score[visible].quantile(0.90))
    active = visible & phase_in_windows(geom["io_phase_spice_deg"], io_windows) & score.ge(threshold)
    windows = contiguous_windows(geom, active, max_gap_min=15.0)
    if windows.empty:
        return None
    windows = windows.sort_values(["median_maser_score", "duration_min"], ascending=False).head(int(n_windows)).reset_index(drop=True)
    freqs = sorted(samples["frequency_mhz"].dropna().unique())
    fig, axes = plt.subplots(len(windows), 2, figsize=(13.0, max(2.15 * len(windows), 4.5)), sharex=False, sharey=True)
    axes = np.atleast_2d(axes)
    for row, win in windows.iterrows():
        start = pd.Timestamp(win["start_time"])
        end = pd.Timestamp(win["end_time"])
        center = start + (end - start) / 2
        lo = center - pd.Timedelta(minutes=75)
        hi = center + pd.Timedelta(minutes=75)
        for col, antenna in enumerate(ANTENNAS):
            ax = axes[row, col]
            sub = samples[(samples["time"] >= lo) & (samples["time"] <= hi) & samples["antenna"].eq(antenna)].copy()
            if sub.empty:
                continue
            sub["minute_bin"] = np.round((sub["time"] - center).dt.total_seconds() / 60.0 / 2.0) * 2.0
            mat = sub.pivot_table(index="frequency_mhz", columns="minute_bin", values="daily_log10_residual", aggfunc="median")
            mat = mat.reindex(freqs)
            vals = mat.to_numpy(dtype=float)
            finite = vals[np.isfinite(vals)]
            lim = max(0.08, float(np.nanpercentile(np.abs(finite), 98))) if finite.size else 0.1
            im = ax.imshow(
                vals,
                origin="lower",
                aspect="auto",
                extent=[mat.columns.min(), mat.columns.max(), -0.5, len(freqs) - 0.5],
                cmap="coolwarm",
                norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim),
            )
            ax.axvspan((start - center).total_seconds() / 60.0, (end - center).total_seconds() / 60.0, color="black", alpha=0.08)
            ax.axvline(0, color="0.2", lw=0.8)
            ax.set_yticks(np.arange(len(freqs)))
            ax.set_yticklabels([f"{f:.2f}" for f in freqs])
            ax.set_title(
                f"{ANTENNA_LABEL[antenna]}  {center:%Y-%m-%d %H:%M}  Io={win['median_io_phase_deg']:.1f} CML={win['median_cml_deg']:.1f}",
                loc="left",
                fontsize=9,
            )
        axes[row, 0].set_ylabel("MHz")
    axes[-1, 0].set_xlabel("minutes from window center")
    axes[-1, 1].set_xlabel("minutes from window center")
    fig.subplots_adjust(right=0.88, hspace=0.45, wspace=0.08, top=0.94, bottom=0.06)
    cax = fig.add_axes([0.90, 0.16, 0.018, 0.68])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("daily log10 residual (dex)")
    fig.suptitle("Strongest Io-window + MASER-top-10 windows: direct time-frequency residuals")
    path = out_dir / "jupiter_highband_active_window_dynamic_spectra.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    windows.to_csv(out_dir / "jupiter_highband_active_window_dynamic_spectra_windows.csv", index=False)
    return path


def plot_occultation_transition_stack(samples: pd.DataFrame, geom: pd.DataFrame, out_dir: Path) -> Path | None:
    states = geom.sort_values("time").copy()
    visible = states["jupiter_visible_by_moon"].astype(bool).to_numpy()
    change_idx = np.flatnonzero(visible[1:] != visible[:-1]) + 1
    if len(change_idx) == 0:
        return None
    transitions = states.iloc[change_idx][["time", "jupiter_visible_by_moon"]].copy()
    transitions = transitions.rename(columns={"time": "transition_time", "jupiter_visible_by_moon": "visible_after_transition"})
    merged = pd.merge_asof(
        samples.sort_values("time"),
        transitions.sort_values("transition_time"),
        left_on="time",
        right_on="transition_time",
        direction="nearest",
        tolerance=pd.Timedelta(minutes=45),
    ).dropna(subset=["transition_time"])
    if merged.empty:
        return None
    merged["dt_min"] = (merged["time"] - merged["transition_time"]).dt.total_seconds() / 60.0
    merged["dt_bin"] = np.floor(merged["dt_min"] / 5.0) * 5.0 + 2.5
    freqs = sorted(merged["frequency_mhz"].dropna().unique())
    fig, axes = plt.subplots(len(freqs), 2, figsize=(12.5, max(2.2 * len(freqs), 5.0)), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    for row, freq in enumerate(freqs):
        for col, antenna in enumerate(ANTENNAS):
            ax = axes[row, col]
            sub = merged[merged["frequency_mhz"].eq(freq) & merged["antenna"].eq(antenna)]
            prof = sub.groupby("dt_bin", sort=True)["daily_log10_residual"].median().reset_index()
            ax.scatter(sub["dt_min"], sub["daily_log10_residual"], s=3, alpha=0.08, color=ANTENNA_COLOR[antenna])
            ax.plot(prof["dt_bin"], prof["daily_log10_residual"], color="black", lw=1.3)
            ax.axvline(0, color="0.2", lw=0.9)
            ax.axhline(0, color="0.6", lw=0.7)
            ax.set_title(f"{ANTENNA_LABEL[antenna]} {freq:.2f} MHz", loc="left", fontsize=10)
            ax.grid(True, color="0.9", lw=0.45)
        axes[row, 0].set_ylabel("daily residual (dex)")
    axes[-1, 0].set_xlabel("minutes from nearest Jupiter visibility transition")
    axes[-1, 1].set_xlabel("minutes from nearest Jupiter visibility transition")
    fig.suptitle(f"Jupiter occultation-transition sanity plot ({len(transitions)} transitions; all high-band samples within 45 min)")
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    path = out_dir / "jupiter_highband_occultation_transition_stack.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report(
    out_dir: Path,
    samples: pd.DataFrame,
    geom: pd.DataFrame,
    thresholds: dict[str, float],
    paths: list[Path],
    config: dict[str, object],
) -> Path:
    visible = samples["jupiter_visible_by_moon"].astype(bool)
    active = samples["io_window_and_maser_top10"].astype(bool)
    rows = []
    for (antenna, freq), grp in samples.groupby(["antenna", "frequency_mhz"], sort=True):
        rows.append(
            {
                "antenna_label": ANTENNA_LABEL.get(str(antenna), str(antenna)),
                "frequency_mhz": float(freq),
                "n_samples": int(len(grp)),
                "n_jupiter_visible": int(grp["jupiter_visible_by_moon"].sum()),
                "n_io_maser_top10": int(grp["io_window_and_maser_top10"].sum()),
                "median_raw_log10_power": float(grp["log10_power"].median()),
                "median_active_daily_resid": float(grp.loc[grp["io_window_and_maser_top10"], "daily_log10_residual"].median()),
                "median_visible_control_daily_resid": float(
                    grp.loc[grp["jupiter_visible_by_moon"].astype(bool) & ~grp["io_window_and_maser_top10"], "daily_log10_residual"].median()
                ),
            }
        )
    summary = pd.DataFrame(rows)
    summary_path = out_dir / "jupiter_highband_direct_diagnostic_summary.csv"
    summary.to_csv(summary_path, index=False)
    paths = [summary_path, *paths]
    geom_visible = geom["jupiter_visible_by_moon"].astype(bool)
    lines = [
        "# Jupiter Direct Diagnostic Plots",
        "",
        "This run intentionally avoids treating 0.45 MHz as evidence. It uses the high Ryle-Vonberg bands only and plots raw log10 power plus simple same-day channel residuals in dex.",
        "",
        "## Configuration",
        "",
        *[f"- `{k}`: `{v}`" for k, v in config.items() if k != "software_versions"],
        "",
        "## Coverage",
        "",
        f"- merged high-band samples: `{len(samples)}`",
        f"- Jupiter-visible high-band samples: `{int(visible.sum())}`",
        f"- Io-window + MASER-top-10 high-band samples: `{int(active.sum())}`",
        f"- geometry grid Jupiter-occulted points: `{int((~geom_visible).sum())}` of `{len(geom)}`",
        "",
        "## MASER Thresholds",
        "",
        *[f"- `{k}`: `{v:.6f}`" for k, v in thresholds.items()],
        "",
        "## Direct Channel Summary",
        "",
        summary.to_string(index=False),
        "",
        "## How To Read The Plots",
        "",
        "- `raw log10(power)` is the direct measured power scale, binned only for readability.",
        "- `daily log10 residual` is `log10(power)` minus that UTC day's median for the same antenna and frequency. `+0.3 dex` is about a factor of two above the same-day channel median.",
        "- The dynamic spectra shade the selected Io-window + MASER-top-10 interval and show residuals by frequency through time.",
        "- The occultation-transition plot is a sanity check only; there are very few Jupiter-occulted geometry samples, so it is not a strong detection test by itself.",
        "",
        "## Files",
        "",
        *[f"- `{p}`" for p in paths],
    ]
    path = out_dir / "jupiter_direct_diagnostic_plots_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-npy", type=Path, default=DEFAULT_CLEAN_NPY)
    parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--frequency-band", type=int, nargs="+", default=[6, 7, 8, 9])
    parser.add_argument("--io-window", type=float, nargs=2, action="append", default=[(80.0, 100.0), (235.0, 260.0)])
    parser.add_argument("--geometry-tolerance-s", type=float, default=360.0)
    parser.add_argument("--n-active-windows", type=int, default=6)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    io_windows = [(float(a), float(b)) for a, b in args.io_window]
    config = {
        "clean_npy": str(args.clean_npy),
        "geometry": str(args.geometry),
        "frequency_bands": [int(v) for v in args.frequency_band],
        "frequencies_mhz": [FREQUENCY_MAP_MHZ.get(int(v), np.nan) for v in args.frequency_band],
        "io_windows_deg": [[float(a), float(b)] for a, b in io_windows],
        "geometry_tolerance_s": float(args.geometry_tolerance_s),
        "n_active_windows": int(args.n_active_windows),
        "software_versions": software_versions(),
    }
    write_json(out_dir / "run_config.json", config)

    samples = read_clean_npy_subset(args.clean_npy, [int(v) for v in args.frequency_band])
    geom = load_geometry(args.geometry)
    samples = merge_geometry(samples, geom, tolerance_s=float(args.geometry_tolerance_s))
    samples, thresholds = add_activity_flags(samples, io_windows)

    paths: list[Path] = []
    paths.extend(plot_io_phase_profiles(samples, out_dir, io_windows))
    paths.append(plot_active_control_daily_scatter(samples, out_dir))
    for maybe_path in [
        plot_active_window_dynamic_spectra(samples, geom, out_dir, io_windows, n_windows=int(args.n_active_windows)),
        plot_occultation_transition_stack(samples, geom, out_dir),
    ]:
        if maybe_path is not None:
            paths.append(maybe_path)
    report_path = write_report(out_dir, samples=samples, geom=geom, thresholds=thresholds, paths=paths, config=config)
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
