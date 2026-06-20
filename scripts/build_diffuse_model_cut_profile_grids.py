#!/usr/bin/env python
"""Build lower-V all-frequency profile grids after diffuse-model cuts."""

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

from rylevonberg.util import ensure_dir, software_versions, write_json  # noqa: E402
from scripts.build_all_frequency_occultation_profile_grids import (  # noqa: E402
    CLEAN,
    _read,
    collect_profiles,
    summarize_profiles,
)


AUDIT_ROOT = ROOT / "outputs/beam_weighted_diffuse_sign_audit_v1"
OUT = ROOT / "outputs/diffuse_model_cut_profile_grids_v1"
LOWER_V = "rv2_coarse"
SOURCE_LABEL = {"sun": "Sun", "fornax_a": "Fornax A"}


def _load_audit(source: str) -> pd.DataFrame:
    path = AUDIT_ROOT / source / f"{source}_beam_weighted_diffuse_row_audit.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    rows = _read(path, parse_dates=["predicted_event_time"])
    return rows[rows["antenna"].astype(str).eq(LOWER_V)].copy()


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def _prepare_events(rows: pd.DataFrame, cut_mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    usable = rows[_bool_series(rows["analysis_usable"])].copy()
    usable["selection_class"] = "all_usable"
    if cut_mode == "exclude_model_anti_strong":
        selected = usable[~usable["model_class"].astype(str).eq("model_anti_strong")].copy()
        selected["selection_class"] = "diffuse_model_cut"
    elif cut_mode == "model_weak_or_source":
        selected = usable[usable["model_class"].astype(str).isin(["model_weak_change", "model_source_strong"])].copy()
        selected["selection_class"] = "diffuse_model_cut"
    else:
        raise ValueError(f"unknown cut mode: {cut_mode}")
    return usable, selected


def _count_summary(usable: pd.DataFrame, selected: pd.DataFrame, points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, grp in usable.groupby(["frequency_mhz", "event_type"], sort=True):
        freq, event_type = keys
        sel = selected[np.isclose(selected["frequency_mhz"], float(freq)) & selected["event_type"].astype(str).eq(str(event_type))]
        prof = (
            points[np.isclose(points["frequency_mhz"], float(freq)) & points["event_type"].astype(str).eq(str(event_type))]
            if not points.empty
            else pd.DataFrame()
        )
        rows.append(
            {
                "frequency_mhz": float(freq),
                "event_type": str(event_type),
                "usable_rows_before_cut": int(len(grp)),
                "selected_rows_after_cut": int(len(sel)),
                "removed_rows": int(len(grp) - len(sel)),
                "removed_fraction": float((len(grp) - len(sel)) / len(grp)) if len(grp) else np.nan,
                "usable_unique_events_before_cut": int(grp["event_id"].nunique()),
                "selected_unique_events_after_cut": int(sel["event_id"].nunique()) if not sel.empty else 0,
                "events_with_profile_points_after_cut": int(prof["event_id"].nunique()) if not prof.empty else 0,
            }
        )
    return pd.DataFrame(rows)


def _plot_cut_grid(summary: pd.DataFrame, source: str, out_dir: Path, window_s: float) -> Path:
    freqs = sorted(summary["frequency_mhz"].dropna().unique())
    fig, axes = plt.subplots(len(freqs), 2, figsize=(12.5, max(10, 1.45 * len(freqs))), sharex=True)
    if len(freqs) == 1:
        axes = np.asarray([axes])
    for i, freq in enumerate(freqs):
        for j, event_type in enumerate(["disappearance", "reappearance"]):
            ax = axes[i, j]
            sub = summary[
                np.isclose(summary["frequency_mhz"], float(freq))
                & summary["event_type"].astype(str).eq(event_type)
            ].sort_values("t_bin_sec")
            if not sub.empty:
                ax.errorbar(
                    sub["t_bin_sec"],
                    sub["median_z_power"],
                    yerr=sub["median_z_power_err"],
                    color="#d95f02",
                    ecolor="#d95f02",
                    marker="o",
                    markersize=2.6,
                    linewidth=1.25,
                    elinewidth=0.65,
                    capsize=1.2,
                )
                med_n = float(np.nanmedian(sub["n_events"]))
                ax.text(
                    0.02,
                    0.93,
                    f"median n/bin={med_n:.0f}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=7,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
                )
            ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
            ax.axhline(0, color="0.62", linewidth=0.7)
            ax.set_title(f"{freq:.2f} MHz {event_type}", fontsize=9)
            if i == len(freqs) - 1:
                ax.set_xlabel("seconds from predicted event")
    fig.supylabel("normalized raw power", x=0.01)
    fig.suptitle(
        f"{SOURCE_LABEL.get(source, source)} lower V after diffuse-model cut\n"
        "Cut: analysis usable rows, excluding model_anti_strong. No trendline subtraction.",
        y=0.996,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    path = out_dir / f"{source}_lower_v_diffuse_model_cut_all_frequency_grid_{int(window_s)}s.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_before_after_grid(all_summary: pd.DataFrame, cut_summary: pd.DataFrame, source: str, out_dir: Path, window_s: float) -> Path:
    freqs = sorted(all_summary["frequency_mhz"].dropna().unique())
    fig, axes = plt.subplots(len(freqs), 2, figsize=(13, max(10, 1.45 * len(freqs))), sharex=True)
    if len(freqs) == 1:
        axes = np.asarray([axes])
    for i, freq in enumerate(freqs):
        for j, event_type in enumerate(["disappearance", "reappearance"]):
            ax = axes[i, j]
            all_sub = all_summary[
                np.isclose(all_summary["frequency_mhz"], float(freq))
                & all_summary["event_type"].astype(str).eq(event_type)
            ].sort_values("t_bin_sec")
            cut_sub = cut_summary[
                np.isclose(cut_summary["frequency_mhz"], float(freq))
                & cut_summary["event_type"].astype(str).eq(event_type)
            ].sort_values("t_bin_sec")
            if not all_sub.empty:
                ax.errorbar(
                    all_sub["t_bin_sec"],
                    all_sub["median_z_power"],
                    yerr=all_sub["median_z_power_err"],
                    color="0.45",
                    ecolor="0.65",
                    marker="o",
                    markersize=2.0,
                    linewidth=1.0,
                    elinewidth=0.45,
                    capsize=0.8,
                    alpha=0.65,
                    label="all usable" if i == 0 and j == 1 else None,
                )
            if not cut_sub.empty:
                ax.errorbar(
                    cut_sub["t_bin_sec"],
                    cut_sub["median_z_power"],
                    yerr=cut_sub["median_z_power_err"],
                    color="#d95f02",
                    ecolor="#d95f02",
                    marker="o",
                    markersize=2.5,
                    linewidth=1.35,
                    elinewidth=0.6,
                    capsize=1.0,
                    label="after cut" if i == 0 and j == 1 else None,
                )
            ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
            ax.axhline(0, color="0.62", linewidth=0.7)
            ax.set_title(f"{freq:.2f} MHz {event_type}", fontsize=9)
            if i == len(freqs) - 1:
                ax.set_xlabel("seconds from predicted event")
            if i == 0 and j == 1:
                ax.legend(frameon=False, fontsize=8)
    fig.supylabel("normalized raw power", x=0.01)
    fig.suptitle(
        f"{SOURCE_LABEL.get(source, source)} lower V diffuse-model cut comparison\n"
        "Gray: all analysis-usable rows. Orange: excludes model_anti_strong rows.",
        y=0.996,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    path = out_dir / f"{source}_lower_v_diffuse_model_cut_before_after_grid_{int(window_s)}s.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_report(
    out_dir: Path,
    source: str,
    cut_mode: str,
    count_summary: pd.DataFrame,
    plot_paths: list[Path],
    window_s: float,
) -> Path:
    lines = [
        f"# {SOURCE_LABEL.get(source, source)} Diffuse-Model Cut Profile Grid",
        "",
        "## Cut",
        "",
        f"- source: `{source}`",
        f"- antenna: `{LOWER_V}` / lower V",
        f"- cut mode: `{cut_mode}`",
        "- selection: `analysis_usable == True` and `model_class != model_anti_strong`",
        f"- window: +/- {window_s:.0f} s",
        "",
        "This cut is based only on the independent beam-weighted diffuse-background model class.",
        "It does not use the observed raw source-like sign to decide which rows to keep.",
        "",
        "## Counts",
        "",
        count_summary.to_string(index=False),
        "",
        "## Plots",
        "",
    ]
    lines.extend(f"- `{path}`" for path in plot_paths)
    report = out_dir / f"{source}_diffuse_model_cut_profile_grid_report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run_source(source: str, clean: pd.DataFrame, args: argparse.Namespace) -> Path:
    source_dir = ensure_dir(Path(args.out_dir) / source)
    rows = _load_audit(source)
    usable, selected = _prepare_events(rows, str(args.cut_mode))

    all_points = collect_profiles(clean, usable, source, float(args.window_s), float(args.bin_s), float(args.inner_s))
    cut_points = collect_profiles(clean, selected, source, float(args.window_s), float(args.bin_s), float(args.inner_s))
    all_summary = summarize_profiles(all_points)
    cut_summary = summarize_profiles(cut_points)
    counts = _count_summary(usable, selected, cut_points)

    usable.to_csv(source_dir / f"{source}_lower_v_all_usable_diffuse_model_rows.csv", index=False)
    selected.to_csv(source_dir / f"{source}_lower_v_diffuse_model_cut_selected_rows.csv", index=False)
    all_points.to_csv(source_dir / f"{source}_lower_v_all_usable_profile_points_{int(args.window_s)}s.csv", index=False)
    cut_points.to_csv(source_dir / f"{source}_lower_v_diffuse_model_cut_profile_points_{int(args.window_s)}s.csv", index=False)
    all_summary.to_csv(source_dir / f"{source}_lower_v_all_usable_profile_summary_{int(args.window_s)}s.csv", index=False)
    cut_summary.to_csv(source_dir / f"{source}_lower_v_diffuse_model_cut_profile_summary_{int(args.window_s)}s.csv", index=False)
    counts.to_csv(source_dir / f"{source}_lower_v_diffuse_model_cut_count_summary.csv", index=False)

    paths = [
        _plot_cut_grid(cut_summary, source, source_dir, float(args.window_s)),
        _plot_before_after_grid(all_summary, cut_summary, source, source_dir, float(args.window_s)),
    ]
    return _write_report(source_dir, source, str(args.cut_mode), counts, paths, float(args.window_s))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", default="sun,fornax_a")
    parser.add_argument("--out-dir", default=str(OUT))
    parser.add_argument("--cut-mode", default="exclude_model_anti_strong", choices=["exclude_model_anti_strong", "model_weak_or_source"])
    parser.add_argument("--window-s", type=float, default=600.0)
    parser.add_argument("--bin-s", type=float, default=60.0)
    parser.add_argument("--inner-s", type=float, default=30.0)
    args = parser.parse_args()

    out_dir = ensure_dir(Path(args.out_dir))
    sources = [x.strip().lower() for x in str(args.sources).split(",") if x.strip()]
    audit_rows = [_load_audit(source) for source in sources]
    bands = {int(x) for x in pd.concat(audit_rows, ignore_index=True)["frequency_band"].dropna().astype(int).unique()}
    clean = _read(CLEAN, parse_dates=["time"])
    clean = clean[clean["antenna"].astype(str).eq(LOWER_V) & clean["frequency_band"].astype(int).isin(bands)].copy()
    write_json(
        out_dir / "run_config.json",
        {
            "sources": sources,
            "antenna": LOWER_V,
            "cut_mode": str(args.cut_mode),
            "window_s": float(args.window_s),
            "bin_s": float(args.bin_s),
            "inner_s": float(args.inner_s),
            "audit_root": str(AUDIT_ROOT),
            "software_versions": software_versions(),
        },
    )
    reports = [run_source(source, clean, args) for source in sources]
    index = out_dir / "diffuse_model_cut_profile_grid_index.md"
    index.write_text(
        "# Diffuse-Model Cut Profile Grids\n\n" + "\n".join(f"- `{path}`" for path in reports) + "\n",
        encoding="utf-8",
    )
    print(index)


if __name__ == "__main__":
    main()
