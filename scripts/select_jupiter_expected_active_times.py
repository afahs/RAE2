#!/usr/bin/env python
"""Select RAE-2 times where Jupiter radio emission is expected a priori.

This is a selector builder, not a detection test. It records two kinds of
priors:

1. broad Io-phase windows commonly used for Io-controlled DAM;
2. high-score bins from published Io-CML probability maps already sampled by
   the Jupiter phase-pattern pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rylevonberg.table_io import read_table


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.util import ensure_dir, write_json  # noqa: E402
from scripts.run_jupiter_historical_window_phase_survey import load_windows  # noqa: E402


DEFAULT_GEOMETRY = ROOT / "outputs/jupiter_phase_pattern_survey_v1/jupiter_spice_visibility_geometry_grid.csv"
DEFAULT_PHASE_SUMMARY = ROOT / "outputs/jupiter_phase_pattern_survey_v1/jupiter_phase_binned_summary.csv"
DEFAULT_HISTORICAL_WINDOWS = ROOT / "configs/jupiter_warwick_dulk_riddle_1975_active_windows.csv"
DEFAULT_OUT = ROOT / "outputs/jupiter_expected_active_selectors_v1"


def phase_in_windows(phase_deg: pd.Series, windows: list[tuple[float, float]]) -> pd.Series:
    phase = pd.to_numeric(phase_deg, errors="coerce") % 360.0
    mask = pd.Series(False, index=phase_deg.index)
    for lo, hi in windows:
        lo = float(lo) % 360.0
        hi = float(hi) % 360.0
        if lo <= hi:
            mask |= (phase >= lo) & (phase <= hi)
        else:
            mask |= (phase >= lo) | (phase <= hi)
    return mask


def bin_centers_for_windows(windows: list[tuple[float, float]], bin_deg: float) -> list[float]:
    centers = np.arange(0.5 * bin_deg, 360.0, bin_deg)
    keep = []
    for center in centers:
        if phase_in_windows(pd.Series([center]), windows).iloc[0]:
            keep.append(float(center))
    return keep


def contiguous_windows(
    geom: pd.DataFrame,
    mask: pd.Series,
    selector_name: str,
    max_gap_min: float,
) -> pd.DataFrame:
    selected = geom[mask.to_numpy()].sort_values("time").copy()
    if selected.empty:
        return pd.DataFrame()
    gaps = selected["time"].diff().dt.total_seconds().div(60.0)
    group = gaps.gt(float(max_gap_min)).fillna(True).cumsum()
    rows = []
    for idx, grp in selected.groupby(group, sort=True):
        rows.append(
            {
                "selector": selector_name,
                "window_id": f"{selector_name}_{int(idx):04d}",
                "start_time": grp["time"].min(),
                "end_time": grp["time"].max(),
                "duration_min": (grp["time"].max() - grp["time"].min()).total_seconds() / 60.0,
                "n_geometry_points": int(len(grp)),
                "median_io_phase_deg": float(grp["io_phase_spice_deg"].median()),
                "median_cml_deg": float(grp["jupiter_cml_spice_deg"].median()),
                "median_maser_zarka_io_score": float(grp["maser_zarka_io_score"].median()),
                "max_maser_zarka_io_score": float(grp["maser_zarka_io_score"].max()),
                "jupiter_visible_fraction": float(grp["jupiter_visible_by_moon"].astype(bool).mean()),
                "earth_occulted_fraction": float((~grp["earth_visible_by_moon"].astype(bool)).mean()),
            }
        )
    return pd.DataFrame(rows)


def attach_historical_mask(geom: pd.DataFrame, historical_windows: pd.DataFrame) -> pd.Series:
    times = pd.to_datetime(geom["time"]).to_numpy(dtype="datetime64[ns]")
    mask = np.zeros(len(geom), dtype=bool)
    for _, row in historical_windows.iterrows():
        start = np.datetime64(pd.Timestamp(row["event_start_time"]))
        end = np.datetime64(pd.Timestamp(row["event_end_time"]))
        mask |= (times >= start) & (times <= end)
    return pd.Series(mask, index=geom.index)


def expected_bin_table(
    phase_summary: pd.DataFrame,
    io_windows: list[tuple[float, float]],
    io_bin_deg: float,
    top_n: int,
) -> pd.DataFrame:
    rows = []
    for lo, hi in io_windows:
        rows.append(
            {
                "selector": "literature_io_phase_window",
                "io_phase_min_deg": float(lo),
                "io_phase_max_deg": float(hi),
                "io_bin_deg": float(io_bin_deg),
                "expected_io_bin_centers_deg": ",".join(f"{v:g}" for v in bin_centers_for_windows([(lo, hi)], io_bin_deg)),
                "cml_bin_deg": "",
                "median_maser_zarka_io_score": np.nan,
                "notes": "Coarse Io-controlled DAM prior; use with Jupiter-visible samples. CML still matters.",
            }
        )

    score_bins = (
        phase_summary[phase_summary["regime"].astype(str).eq("jupiter_visible")]
        [["cml_bin_deg", "io_bin_deg", "median_zarka_io_score"]]
        .dropna()
        .groupby(["cml_bin_deg", "io_bin_deg"], as_index=False)
        .agg(median_zarka_io_score=("median_zarka_io_score", "max"))
        .sort_values("median_zarka_io_score", ascending=False)
        .head(int(top_n))
    )
    for _, row in score_bins.iterrows():
        rows.append(
            {
                "selector": "maser_zarka_io_top_iocml_bin",
                "io_phase_min_deg": float(row["io_bin_deg"]) - 7.5,
                "io_phase_max_deg": float(row["io_bin_deg"]) + 7.5,
                "io_bin_deg": 15.0,
                "expected_io_bin_centers_deg": f"{float(row['io_bin_deg']):g}",
                "cml_bin_deg": float(row["cml_bin_deg"]),
                "median_maser_zarka_io_score": float(row["median_zarka_io_score"]),
                "notes": "Top published-probability Io-CML bin sampled from the MASER/Zarka Io-controlled map.",
            }
        )
    return pd.DataFrame(rows)


def plot_selectors(geom: pd.DataFrame, masks: dict[str, pd.Series], out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))
    ax = axes[0]
    visible = geom[geom["jupiter_visible_by_moon"].astype(bool)]
    ax.hist(visible["io_phase_spice_deg"], bins=np.arange(0, 361, 10), color="0.82", label="Jupiter visible")
    for name, mask in masks.items():
        sub = geom[mask & geom["jupiter_visible_by_moon"].astype(bool)]
        ax.hist(sub["io_phase_spice_deg"], bins=np.arange(0, 361, 10), histtype="step", lw=1.6, label=name)
    ax.set_xlabel("Io phase (deg)")
    ax.set_ylabel("geometry-grid count")
    ax.set_title("Expected-active selectors in Io phase")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, color="0.92", lw=0.5)

    ax = axes[1]
    score = pd.to_numeric(geom["maser_zarka_io_score"], errors="coerce")
    sc = ax.scatter(
        geom["jupiter_cml_spice_deg"],
        geom["io_phase_spice_deg"],
        c=score,
        s=4,
        cmap="magma",
        alpha=0.35,
        linewidths=0,
    )
    for name, mask in masks.items():
        if "top10" not in name:
            continue
        sub = geom[mask]
        ax.scatter(sub["jupiter_cml_spice_deg"], sub["io_phase_spice_deg"], s=9, facecolors="none", edgecolors="cyan", lw=0.7, label=name)
    ax.set_xlabel("Jupiter System III CML (deg)")
    ax.set_ylabel("Io phase (deg)")
    ax.set_title("MASER Io-CML selector on RAE-2 geometry")
    ax.set_xlim(0, 360)
    ax.set_ylim(0, 360)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    cbar = fig.colorbar(sc, ax=ax, pad=0.015)
    cbar.set_label("MASER/Zarka Io score")
    fig.tight_layout()
    path = out_dir / "jupiter_expected_active_selectors.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--phase-summary", type=Path, default=DEFAULT_PHASE_SUMMARY)
    parser.add_argument("--historical-windows", type=Path, default=DEFAULT_HISTORICAL_WINDOWS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--io-window", type=float, nargs=2, action="append", default=[(80.0, 100.0), (235.0, 260.0)])
    parser.add_argument("--io-bin-deg", type=float, default=10.0)
    parser.add_argument("--top-iocml-bins", type=int, default=30)
    parser.add_argument("--max-gap-min", type=float, default=20.0)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    geom = read_table(args.geometry, parse_dates=["time"], low_memory=False).sort_values("time")
    phase_summary = read_table(args.phase_summary, low_memory=False)
    hist = load_windows(args.historical_windows, padding_min=0.0)

    visible = geom["jupiter_visible_by_moon"].astype(bool)
    score = pd.to_numeric(geom["maser_zarka_io_score"], errors="coerce")
    q75 = float(score[visible].quantile(0.75))
    q90 = float(score[visible].quantile(0.90))
    q95 = float(score[visible].quantile(0.95))
    masks = {
        "literature_io_phase_windows": visible & phase_in_windows(geom["io_phase_spice_deg"], args.io_window),
        "maser_zarka_io_top25_visible": visible & (score >= q75),
        "maser_zarka_io_top10_visible": visible & (score >= q90),
        "maser_zarka_io_top05_visible": visible & (score >= q95),
        "historical_wdr_reported_windows": visible & attach_historical_mask(geom, hist),
    }

    bins = expected_bin_table(phase_summary, args.io_window, float(args.io_bin_deg), int(args.top_iocml_bins))
    bins.to_csv(out_dir / "jupiter_expected_active_bins.csv", index=False)

    window_tables = []
    for name, mask in masks.items():
        table = contiguous_windows(geom, mask, name, max_gap_min=float(args.max_gap_min))
        if not table.empty:
            window_tables.append(table)
    windows = pd.concat(window_tables, ignore_index=True) if window_tables else pd.DataFrame()
    windows.to_csv(out_dir / "jupiter_expected_active_geometry_windows.csv", index=False)

    selector_summary = []
    for name, mask in masks.items():
        sub = geom[mask]
        selector_summary.append(
            {
                "selector": name,
                "n_geometry_points": int(mask.sum()),
                "n_contiguous_windows": int((windows["selector"] == name).sum()) if not windows.empty else 0,
                "median_io_phase_deg": float(sub["io_phase_spice_deg"].median()) if len(sub) else np.nan,
                "median_cml_deg": float(sub["jupiter_cml_spice_deg"].median()) if len(sub) else np.nan,
                "median_maser_zarka_io_score": float(sub["maser_zarka_io_score"].median()) if len(sub) else np.nan,
                "max_maser_zarka_io_score": float(sub["maser_zarka_io_score"].max()) if len(sub) else np.nan,
            }
        )
    summary = pd.DataFrame(selector_summary)
    summary.to_csv(out_dir / "jupiter_expected_active_selector_summary.csv", index=False)

    plot_path = plot_selectors(geom, masks, out_dir)
    write_json(
        out_dir / "run_config.json",
        {
            "geometry": str(args.geometry),
            "phase_summary": str(args.phase_summary),
            "historical_windows": str(args.historical_windows),
            "io_windows_deg": [[float(a), float(b)] for a, b in args.io_window],
            "io_bin_deg": float(args.io_bin_deg),
            "top_iocml_bins": int(args.top_iocml_bins),
            "max_gap_min": float(args.max_gap_min),
            "maser_zarka_io_score_q75_visible": q75,
            "maser_zarka_io_score_q90_visible": q90,
            "maser_zarka_io_score_q95_visible": q95,
        },
    )
    print(out_dir / "jupiter_expected_active_bins.csv")
    print(out_dir / "jupiter_expected_active_geometry_windows.csv")
    print(out_dir / "jupiter_expected_active_selector_summary.csv")
    print(plot_path)


if __name__ == "__main__":
    main()
