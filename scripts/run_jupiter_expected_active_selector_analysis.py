#!/usr/bin/env python
"""Test expected-active Jupiter selectors against same-day controls."""

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
from scripts.run_jupiter_historical_window_phase_survey import load_windows  # noqa: E402
from scripts.run_jupiter_literature_controls import (  # noqa: E402
    bootstrap_mean_ci,
    load_full_samples,
    sign_flip_p,
)
from scripts.select_jupiter_expected_active_times import phase_in_windows  # noqa: E402


DEFAULT_CLEAN = ROOT / "outputs/control_survey_earth_sun_postnov1974_v2/01_ingest/cleaned_timeseries.csv"
DEFAULT_GEOMETRY = ROOT / "outputs/jupiter_phase_pattern_survey_v1/jupiter_spice_visibility_geometry_grid.csv"
DEFAULT_WINDOWS = ROOT / "configs/jupiter_warwick_dulk_riddle_1975_active_windows.csv"
DEFAULT_OUT = ROOT / "outputs/jupiter_expected_active_selector_analysis_v1"

ANTENNA_LABEL = {"rv1_coarse": "upper V", "rv2_coarse": "lower V"}
ANTENNA_COLOR = {"rv1_coarse": "#386cb0", "rv2_coarse": "#bf5b17"}

SELECTOR_LABEL = {
    "literature_io_phase_windows": "Io phase windows",
    "maser_top25": "MASER top 25%",
    "maser_top10": "MASER top 10%",
    "maser_top05": "MASER top 5%",
    "io_windows_and_maser_top10": "Io windows + MASER top 10%",
    "historical_wdr_reported_windows": "Historical WDR windows",
}


def interval_mask(times: pd.Series, intervals: pd.DataFrame) -> pd.Series:
    arr = pd.to_datetime(times).to_numpy(dtype="datetime64[ns]")
    mask = np.zeros(len(times), dtype=bool)
    for _, row in intervals.iterrows():
        start = np.datetime64(pd.Timestamp(row["event_start_time"]))
        end = np.datetime64(pd.Timestamp(row["event_end_time"]))
        mask |= (arr >= start) & (arr <= end)
    return pd.Series(mask, index=times.index)


def build_selector_masks(
    samples: pd.DataFrame,
    historical_windows: pd.DataFrame,
    io_windows: list[tuple[float, float]],
) -> tuple[dict[str, pd.Series], dict[str, float]]:
    visible = samples["jupiter_visible_by_moon"].astype(bool)
    score = pd.to_numeric(samples["maser_zarka_io_score"], errors="coerce")
    score_visible = score[visible]
    thresholds = {
        "maser_q75_visible": float(score_visible.quantile(0.75)),
        "maser_q90_visible": float(score_visible.quantile(0.90)),
        "maser_q95_visible": float(score_visible.quantile(0.95)),
    }
    io_mask = visible & phase_in_windows(samples["io_phase_spice_deg"], io_windows)
    masks = {
        "literature_io_phase_windows": io_mask,
        "maser_top25": visible & (score >= thresholds["maser_q75_visible"]),
        "maser_top10": visible & (score >= thresholds["maser_q90_visible"]),
        "maser_top05": visible & (score >= thresholds["maser_q95_visible"]),
        "io_windows_and_maser_top10": io_mask & (score >= thresholds["maser_q90_visible"]),
        "historical_wdr_reported_windows": visible & interval_mask(samples["time"], historical_windows),
    }
    return masks, thresholds


def daily_selector_points(
    samples: pd.DataFrame,
    selected_mask: pd.Series,
    selector_name: str,
    high_z: float,
    min_selected_samples: int,
    min_control_samples: int,
) -> pd.DataFrame:
    visible = samples["jupiter_visible_by_moon"].astype(bool)
    selected = samples[visible & selected_mask].copy()
    control = samples[visible & ~selected_mask].copy()
    if selected.empty or control.empty:
        return pd.DataFrame()
    keys = ["date", "antenna", "frequency_band", "frequency_mhz"]
    selected["high_tail"] = selected["daily_z_log_power"].to_numpy(dtype=float) > float(high_z)
    control["high_tail"] = control["daily_z_log_power"].to_numpy(dtype=float) > float(high_z)
    selected_daily = (
        selected.groupby(keys, sort=True)
        .agg(
            selected_n_samples=("daily_z_log_power", "size"),
            selected_high_tail_fraction=("high_tail", "mean"),
            selected_median_daily_z=("daily_z_log_power", "median"),
            selected_mean_daily_z=("daily_z_log_power", "mean"),
            selected_median_maser_score=("maser_zarka_io_score", "median"),
            selected_median_io_phase=("io_phase_spice_deg", "median"),
            selected_median_cml=("jupiter_cml_spice_deg", "median"),
        )
        .reset_index()
    )
    control_daily = (
        control.groupby(keys, sort=True)
        .agg(
            control_n_samples=("daily_z_log_power", "size"),
            control_high_tail_fraction=("high_tail", "mean"),
            control_median_daily_z=("daily_z_log_power", "median"),
            control_mean_daily_z=("daily_z_log_power", "mean"),
            control_median_maser_score=("maser_zarka_io_score", "median"),
        )
        .reset_index()
    )
    paired = selected_daily.merge(control_daily, on=keys, how="inner")
    paired = paired[
        (paired["selected_n_samples"] >= int(min_selected_samples))
        & (paired["control_n_samples"] >= int(min_control_samples))
    ].copy()
    if paired.empty:
        return paired
    paired["selector"] = selector_name
    paired["selected_minus_control_high_tail_fraction"] = (
        paired["selected_high_tail_fraction"] - paired["control_high_tail_fraction"]
    )
    paired["selected_minus_control_median_z"] = (
        paired["selected_median_daily_z"] - paired["control_median_daily_z"]
    )
    paired["selected_minus_control_mean_z"] = paired["selected_mean_daily_z"] - paired["control_mean_daily_z"]
    paired["selected_minus_control_maser_score"] = (
        paired["selected_median_maser_score"] - paired["control_median_maser_score"]
    )
    return paired


def summarize_daily_points(
    daily: pd.DataFrame,
    rng: np.random.Generator,
    n_boot: int,
    n_perm: int,
) -> pd.DataFrame:
    rows = []
    group_cols = ["selector", "antenna", "frequency_band", "frequency_mhz"]
    for (selector, antenna, band, freq), grp in daily.groupby(group_cols, sort=True):
        tail = grp["selected_minus_control_high_tail_fraction"].to_numpy(dtype=float)
        med = grp["selected_minus_control_median_z"].to_numpy(dtype=float)
        mean_z = grp["selected_minus_control_mean_z"].to_numpy(dtype=float)
        tail_mean, tail_lo, tail_hi = bootstrap_mean_ci(tail, rng, n_boot)
        med_mean, med_lo, med_hi = bootstrap_mean_ci(med, rng, n_boot)
        mean_z_mean, mean_z_lo, mean_z_hi = bootstrap_mean_ci(mean_z, rng, n_boot)
        tail_p_two, tail_p_pos = sign_flip_p(tail, rng, n_perm)
        med_p_two, med_p_pos = sign_flip_p(med, rng, n_perm)
        rows.append(
            {
                "selector": selector,
                "selector_label": SELECTOR_LABEL.get(str(selector), str(selector)),
                "antenna": antenna,
                "antenna_label": ANTENNA_LABEL.get(str(antenna), str(antenna)),
                "frequency_band": int(band),
                "frequency_mhz": float(freq),
                "n_paired_days": int(len(grp)),
                "selected_total_samples": int(grp["selected_n_samples"].sum()),
                "control_total_samples": int(grp["control_n_samples"].sum()),
                "selected_high_tail_fraction_mean": float(grp["selected_high_tail_fraction"].mean()),
                "control_high_tail_fraction_mean": float(grp["control_high_tail_fraction"].mean()),
                "mean_selected_minus_control_high_tail_fraction": tail_mean,
                "boot_lo_selected_minus_control_high_tail_fraction": tail_lo,
                "boot_hi_selected_minus_control_high_tail_fraction": tail_hi,
                "positive_day_fraction_high_tail": float((tail > 0).mean()) if len(tail) else np.nan,
                "signflip_p_two_sided_high_tail": tail_p_two,
                "signflip_p_positive_high_tail": tail_p_pos,
                "mean_selected_minus_control_median_z": med_mean,
                "boot_lo_selected_minus_control_median_z": med_lo,
                "boot_hi_selected_minus_control_median_z": med_hi,
                "positive_day_fraction_median_z": float((med > 0).mean()) if len(med) else np.nan,
                "signflip_p_two_sided_median_z": med_p_two,
                "signflip_p_positive_median_z": med_p_pos,
                "mean_selected_minus_control_mean_z": mean_z_mean,
                "boot_lo_selected_minus_control_mean_z": mean_z_lo,
                "boot_hi_selected_minus_control_mean_z": mean_z_hi,
                "selected_median_maser_score_median": float(grp["selected_median_maser_score"].median()),
                "control_median_maser_score_median": float(grp["control_median_maser_score"].median()),
            }
        )
    return pd.DataFrame(rows)


def plot_selector_spectrum(summary: pd.DataFrame, out_dir: Path, metric: str, ylabel: str, filename: str) -> Path:
    selectors = [s for s in SELECTOR_LABEL if s in set(summary["selector"])]
    n = len(selectors)
    fig, axes = plt.subplots(n, 1, figsize=(10.8, max(2.2 * n, 5.0)), sharex=True, sharey=False)
    axes = np.atleast_1d(axes)
    lo_col = metric.replace("mean_", "boot_lo_")
    hi_col = metric.replace("mean_", "boot_hi_")
    p_col = "signflip_p_positive_high_tail" if "high_tail" in metric else "signflip_p_positive_median_z"
    for ax, selector in zip(axes, selectors):
        subsel = summary[summary["selector"].eq(selector)].copy()
        for antenna, grp in subsel.groupby("antenna", sort=True):
            grp = grp.sort_values("frequency_mhz")
            y = grp[metric].to_numpy(dtype=float)
            lo = grp[lo_col].to_numpy(dtype=float)
            hi = grp[hi_col].to_numpy(dtype=float)
            err = np.vstack([y - lo, hi - y])
            err[~np.isfinite(err)] = 0.0
            ax.errorbar(
                grp["frequency_mhz"],
                y,
                yerr=err,
                marker="o",
                lw=1.35,
                capsize=2.5,
                color=ANTENNA_COLOR.get(str(antenna), "black"),
                label=ANTENNA_LABEL.get(str(antenna), str(antenna)),
            )
            sig = pd.to_numeric(grp[p_col], errors="coerce").to_numpy(dtype=float) < 0.05
            if np.any(sig):
                ax.scatter(
                    grp.loc[sig, "frequency_mhz"],
                    grp.loc[sig, metric],
                    s=72,
                    facecolors="none",
                    edgecolors="black",
                    linewidths=1.1,
                    zorder=5,
                )
        ax.axhline(0, color="0.35", lw=0.85)
        ax.set_xscale("log")
        ax.set_xticks(sorted(FREQUENCY_MAP_MHZ.values()))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.set_ylabel(ylabel)
        ax.set_title(SELECTOR_LABEL.get(selector, selector), loc="left", fontsize=10)
        ax.grid(True, color="0.9", lw=0.45)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    axes[-1].set_xlabel("frequency (MHz)")
    fig.suptitle("Expected-active Jupiter selector test: selected samples vs same-day controls")
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    path = out_dir / filename
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_selector_heatmap(summary: pd.DataFrame, out_dir: Path) -> Path:
    work = summary.copy()
    work["channel"] = work["antenna_label"] + " " + work["frequency_mhz"].map(lambda v: f"{v:.2f}")
    mat = work.pivot_table(
        index="selector_label",
        columns="channel",
        values="mean_selected_minus_control_high_tail_fraction",
        aggfunc="first",
    )
    selector_order = [SELECTOR_LABEL[s] for s in SELECTOR_LABEL if SELECTOR_LABEL[s] in mat.index]
    mat = mat.reindex(selector_order)
    channel_order = []
    for antenna_label in ["upper V", "lower V"]:
        for freq in sorted(work["frequency_mhz"].dropna().unique()):
            label = f"{antenna_label} {freq:.2f}"
            if label in mat.columns:
                channel_order.append(label)
    mat = mat[channel_order]
    vals = mat.to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    lim = max(0.005, float(np.nanpercentile(np.abs(finite), 98))) if finite.size else 0.01
    fig, ax = plt.subplots(figsize=(13.5, 5.8))
    im = ax.imshow(
        vals,
        aspect="auto",
        cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim),
    )
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels(mat.index)
    ax.set_xticks(np.arange(len(mat.columns)))
    ax.set_xticklabels(mat.columns, rotation=45, ha="right", fontsize=8)
    ax.set_title("Expected-active selected minus same-day control high-tail fraction")
    cbar = fig.colorbar(im, ax=ax, pad=0.015)
    cbar.set_label("selected - control high-tail fraction")
    fig.tight_layout()
    path = out_dir / "jupiter_expected_active_selector_high_tail_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report(
    out_dir: Path,
    summary: pd.DataFrame,
    paths: list[Path],
    config: dict[str, object],
    thresholds: dict[str, float],
) -> Path:
    rank_cols = [
        "selector_label",
        "antenna_label",
        "frequency_mhz",
        "n_paired_days",
        "mean_selected_minus_control_high_tail_fraction",
        "boot_lo_selected_minus_control_high_tail_fraction",
        "boot_hi_selected_minus_control_high_tail_fraction",
        "signflip_p_positive_high_tail",
        "mean_selected_minus_control_median_z",
        "signflip_p_positive_median_z",
    ]
    top = summary.sort_values("mean_selected_minus_control_high_tail_fraction", ascending=False)
    lines = [
        "# Jupiter Expected-Active Selector Analysis",
        "",
        "This run applies a priori Jupiter activity selectors to full merged RAE-2 samples and compares selected samples with same-day, same-channel Jupiter-visible controls.",
        "",
        "## Selectors",
        "",
        "- `literature_io_phase_windows`: Jupiter visible and Io phase in the configured coarse Io windows.",
        "- `maser_top25`, `maser_top10`, `maser_top05`: Jupiter visible and MASER/Zarka Io-CML score above the visible-sample quantile threshold.",
        "- `io_windows_and_maser_top10`: intersection of coarse Io windows and top-10% MASER score.",
        "- `historical_wdr_reported_windows`: Warwick/Dulk/Riddle reported active windows, used as an independent time selector.",
        "",
        "## Configuration",
        "",
        *[f"- `{k}`: `{v}`" for k, v in config.items() if k != "software_versions"],
        "",
        "## MASER Score Thresholds",
        "",
        *[f"- `{k}`: `{v:.6f}`" for k, v in thresholds.items()],
        "",
        "## Strongest Positive High-Tail Results",
        "",
        top[rank_cols].head(18).to_string(index=False) if not top.empty else "(none)",
        "",
        "## Strongest Negative High-Tail Results",
        "",
        top.tail(12).sort_values("mean_selected_minus_control_high_tail_fraction")[rank_cols].to_string(index=False)
        if not top.empty
        else "(none)",
        "",
        "## Interpretation",
        "",
        "- The primary quantity is selected minus same-day control fraction of samples with daily-normalized log power above the configured high-z threshold.",
        "- Bootstrap intervals are over UTC day/channel pairs, not individual samples.",
        "- Open circles in the spectra mark one-sided sign-flip p < 0.05 for that channel and selector before any multiple-comparison correction.",
        "- A convincing Jupiter result should repeat across adjacent frequencies, appear in a physically plausible selector, and not be dominated by one isolated channel.",
        "",
        "## Files",
        "",
        *[f"- `{p}`" for p in paths],
    ]
    path = out_dir / "jupiter_expected_active_selector_analysis_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", type=Path, default=DEFAULT_CLEAN)
    parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--historical-windows", type=Path, default=DEFAULT_WINDOWS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--io-window", type=float, nargs=2, action="append", default=[(80.0, 100.0), (235.0, 260.0)])
    parser.add_argument("--high-z", type=float, default=2.5)
    parser.add_argument("--min-selected-samples-per-day", type=int, default=5)
    parser.add_argument("--min-control-samples-per-day", type=int, default=25)
    parser.add_argument("--geometry-tolerance-s", type=float, default=360.0)
    parser.add_argument("--bootstrap-samples", type=int, default=3000)
    parser.add_argument("--permutations", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260610)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    rng = np.random.default_rng(int(args.seed))
    config = {
        "clean": str(args.clean),
        "geometry": str(args.geometry),
        "historical_windows": str(args.historical_windows),
        "io_windows_deg": [[float(a), float(b)] for a, b in args.io_window],
        "high_z": float(args.high_z),
        "min_selected_samples_per_day": int(args.min_selected_samples_per_day),
        "min_control_samples_per_day": int(args.min_control_samples_per_day),
        "geometry_tolerance_s": float(args.geometry_tolerance_s),
        "bootstrap_samples": int(args.bootstrap_samples),
        "permutations": int(args.permutations),
        "seed": int(args.seed),
        "software_versions": software_versions(),
    }
    write_json(out_dir / "run_config.json", config)

    samples = load_full_samples(args.clean, args.geometry, tolerance_s=float(args.geometry_tolerance_s))
    windows = load_windows(args.historical_windows, padding_min=0.0)
    masks, thresholds = build_selector_masks(samples, windows, [(float(a), float(b)) for a, b in args.io_window])

    daily_tables = []
    for selector, mask in masks.items():
        daily = daily_selector_points(
            samples,
            mask,
            selector_name=selector,
            high_z=float(args.high_z),
            min_selected_samples=int(args.min_selected_samples_per_day),
            min_control_samples=int(args.min_control_samples_per_day),
        )
        if not daily.empty:
            daily_tables.append(daily)
    daily_all = pd.concat(daily_tables, ignore_index=True) if daily_tables else pd.DataFrame()
    daily_all.to_csv(out_dir / "jupiter_expected_active_selector_daily_points.csv", index=False)
    summary = summarize_daily_points(
        daily_all,
        rng=rng,
        n_boot=int(args.bootstrap_samples),
        n_perm=int(args.permutations),
    )
    summary.to_csv(out_dir / "jupiter_expected_active_selector_summary.csv", index=False)

    paths: list[Path] = []
    paths.append(
        plot_selector_spectrum(
            summary,
            out_dir,
            metric="mean_selected_minus_control_high_tail_fraction",
            ylabel="selected - control\nhigh-tail fraction",
            filename="jupiter_expected_active_selector_high_tail_spectrum.png",
        )
    )
    paths.append(
        plot_selector_spectrum(
            summary,
            out_dir,
            metric="mean_selected_minus_control_median_z",
            ylabel="selected - control\nmedian daily z",
            filename="jupiter_expected_active_selector_median_z_spectrum.png",
        )
    )
    paths.append(plot_selector_heatmap(summary, out_dir))
    report = write_report(out_dir, summary, paths, config, thresholds)
    print(report)


if __name__ == "__main__":
    main()
