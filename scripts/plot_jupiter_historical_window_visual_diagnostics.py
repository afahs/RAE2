#!/usr/bin/env python
"""Visual diagnostics for historical-window selected Jupiter analysis.

The plots here are designed to make repeatability visible without relying on a
single scalar detection metric.
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
DEFAULT_STACK_DIR = ROOT / "outputs/jupiter_historical_window_stacks_v1"

ANTENNA_LABEL = {
    "rv1_coarse": "upper V",
    "rv2_coarse": "lower V",
}


def channel_label(antenna: object, freq: object) -> str:
    return f"{ANTENNA_LABEL.get(str(antenna), str(antenna))}\n{float(freq):.2f} MHz"


def load_inputs(stack_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paired = read_table(stack_dir / "jupiter_historical_window_paired_window_stats.csv")
    summary = read_table(stack_dir / "jupiter_historical_window_selected_stack_summary.csv")
    active = read_table(
        stack_dir / "jupiter_historical_window_selected_active_samples.csv",
        parse_dates=["time", "event_start_time", "event_end_time"],
    )
    control = read_table(
        stack_dir / "jupiter_historical_window_selected_shifted_control_samples.csv",
        parse_dates=["time", "event_start_time", "event_end_time"],
    )
    return paired, summary, active, control


def add_plot_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["channel"] = [channel_label(a, f) for a, f in zip(out["antenna"], out["frequency_mhz"])]
    out["sort_key"] = out["antenna"].map({"rv1_coarse": 0, "rv2_coarse": 1}).fillna(9) * 100 + out[
        "frequency_mhz"
    ].astype(float)
    return out


def window_order(active: pd.DataFrame) -> pd.DataFrame:
    meta = (
        active.groupby("historical_window_id", sort=True)
        .agg(
            event_start_time=("event_start_time", "first"),
            intensity=("intensity", "first"),
            burstiness=("burstiness", "first"),
        )
        .reset_index()
        .sort_values("event_start_time")
    )
    meta["window_label"] = (
        meta["historical_window_id"].astype(str)
        + "  I="
        + meta["intensity"].astype(int).astype(str)
        + " "
        + meta["burstiness"].astype(str).str.slice(0, 3)
    )
    return meta


def plot_window_channel_heatmaps(paired: pd.DataFrame, active: pd.DataFrame, out_dir: Path) -> list[Path]:
    paired = add_plot_columns(paired)
    order = window_order(active)
    channels = paired[["channel", "sort_key"]].drop_duplicates().sort_values("sort_key")
    rows = order["historical_window_id"].tolist()
    row_labels = order["window_label"].tolist()
    paths = []

    for value_col, title, cmap, centered, filename, cbar_label in [
        (
            "active_median_daily_z",
            "Historical Jupiter windows: active-window median power",
            "viridis",
            False,
            "jupiter_window_channel_active_power_heatmap.png",
            "active-window median daily z",
        ),
        (
            "paired_median_daily_z_excess",
            "Historical Jupiter windows: active minus shifted-control power",
            "coolwarm",
            True,
            "jupiter_window_channel_excess_heatmap.png",
            "active - shifted-control median daily z",
        ),
    ]:
        mat = (
            paired.pivot_table(
                index="historical_window_id",
                columns="channel",
                values=value_col,
                aggfunc="median",
            )
            .reindex(index=rows, columns=channels["channel"].tolist())
            .to_numpy(dtype=float)
        )
        fig, ax = plt.subplots(figsize=(13.5, max(8.0, 0.22 * len(rows))))
        if centered:
            finite = mat[np.isfinite(mat)]
            vmax = float(np.nanquantile(np.abs(finite), 0.95)) if len(finite) else 1.0
            norm = TwoSlopeNorm(vcenter=0.0, vmin=-max(vmax, 0.5), vmax=max(vmax, 0.5))
        else:
            finite = mat[np.isfinite(mat)]
            norm = Normalize(
                vmin=float(np.nanquantile(finite, 0.05)) if len(finite) else -2,
                vmax=float(np.nanquantile(finite, 0.95)) if len(finite) else 2,
            )
        im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
        ax.set_xticks(np.arange(len(channels)))
        ax.set_xticklabels(channels["channel"], rotation=45, ha="right", fontsize=8)
        step = max(1, len(row_labels) // 26)
        ax.set_yticks(np.arange(len(row_labels))[::step])
        ax.set_yticklabels(row_labels[::step], fontsize=7)
        ax.set_xlabel("antenna / frequency")
        ax.set_ylabel("historical active window, sorted by date")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, pad=0.012, label=cbar_label)
        fig.tight_layout()
        path = out_dir / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_ladder_controls(paired: pd.DataFrame, summary: pd.DataFrame, out_dir: Path, top_n: int) -> Path:
    top = summary.sort_values("paired_median_daily_z_excess_mean", ascending=False).head(int(top_n)).copy()
    fig, axes = plt.subplots(len(top), 1, figsize=(8.5, max(2.6 * len(top), 5.0)), sharex=True, sharey=False)
    axes = np.atleast_1d(axes)
    for ax, (_, row) in zip(axes, top.iterrows()):
        sub = paired[
            paired["antenna"].astype(str).eq(str(row["antenna"]))
            & np.isclose(paired["frequency_mhz"].astype(float), float(row["frequency_mhz"]))
        ].copy()
        sub = sub.sort_values("paired_median_daily_z_excess")
        for _, w in sub.iterrows():
            y0 = float(w["shifted_median_daily_z"])
            y1 = float(w["active_median_daily_z"])
            color = "tab:blue" if y1 > y0 else "tab:red"
            ax.plot([0, 1], [y0, y1], color=color, alpha=0.45, lw=1.0)
            ax.scatter([0, 1], [y0, y1], color=color, s=12, alpha=0.7)
        ax.scatter([0, 1], [sub["shifted_median_daily_z"].median(), sub["active_median_daily_z"].median()], s=80, color="black")
        ax.axhline(0, color="0.35", lw=0.7)
        ax.set_xlim(-0.18, 1.18)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["shifted controls", "historical active"])
        ax.set_ylabel("window median daily z")
        ax.set_title(
            f"{ANTENNA_LABEL.get(str(row['antenna']), row['antenna'])} {float(row['frequency_mhz']):.2f} MHz: "
            f"one line per historical window"
        )
    fig.suptitle("Paired active/control ladders: visually check repeatability")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = out_dir / "jupiter_active_vs_shifted_ladder_top_channels.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_phase_raster(active: pd.DataFrame, summary: pd.DataFrame, out_dir: Path, top_n: int) -> Path:
    top = summary.sort_values("paired_median_daily_z_excess_mean", ascending=False).head(int(top_n)).copy()
    order = window_order(active)
    row_index = {wid: i for i, wid in enumerate(order["historical_window_id"])}
    fig, axes = plt.subplots(len(top), 1, figsize=(12.0, max(3.0 * len(top), 5.0)), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (_, row) in zip(axes, top.iterrows()):
        sub = active[
            active["antenna"].astype(str).eq(str(row["antenna"]))
            & np.isclose(active["frequency_mhz"].astype(float), float(row["frequency_mhz"]))
        ].copy()
        sub["y"] = sub["historical_window_id"].map(row_index)
        sc = ax.scatter(
            sub["window_phase"],
            sub["y"],
            c=sub["daily_z_log_power"],
            s=18,
            cmap="coolwarm",
            vmin=-3,
            vmax=5,
            alpha=0.82,
            rasterized=True,
        )
        ax.axvspan(0, 1, color="gold", alpha=0.08)
        ax.axvline(0, color="0.25", lw=0.8)
        ax.axvline(1, color="0.25", lw=0.8)
        ax.set_xlim(-0.18, 1.18)
        ax.set_ylabel("historical windows")
        ax.set_title(
            f"{ANTENNA_LABEL.get(str(row['antenna']), row['antenna'])} {float(row['frequency_mhz']):.2f} MHz"
        )
    axes[-1].set_xlabel("fraction through reported historical window; 0=start, 1=end")
    fig.colorbar(sc, ax=axes.tolist(), pad=0.012, label="daily-normalized log power")
    fig.suptitle("Raster view: actual samples in historical Jupiter windows")
    fig.savefig(out_dir / "jupiter_historical_window_phase_raster_top_channels.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_dir / "jupiter_historical_window_phase_raster_top_channels.png"


def plot_multifrequency_window_strips(active: pd.DataFrame, out_dir: Path, top_n: int) -> Path:
    core = active[active["inside_reported_window"]].copy()
    top_windows = (
        core.groupby("historical_window_id", sort=True)
        .agg(max_daily_z=("daily_z_log_power", "max"), event_start_time=("event_start_time", "first"))
        .reset_index()
        .sort_values("max_daily_z", ascending=False)
        .head(int(top_n))
    )
    fig, axes = plt.subplots(len(top_windows), 1, figsize=(13.0, max(2.6 * len(top_windows), 5.0)), sharex=False)
    axes = np.atleast_1d(axes)
    sc = None
    for ax, (_, row) in zip(axes, top_windows.iterrows()):
        sub = active[active["historical_window_id"].astype(str).eq(str(row["historical_window_id"]))].copy()
        marker_map = {"rv1_coarse": "o", "rv2_coarse": "^"}
        for antenna, grp in sub.groupby("antenna", sort=True):
            sc = ax.scatter(
                grp["dt_from_start_min"],
                grp["frequency_mhz"],
                c=grp["daily_z_log_power"],
                cmap="coolwarm",
                vmin=-3,
                vmax=5,
                s=36 if antenna == "rv2_coarse" else 25,
                marker=marker_map.get(str(antenna), "o"),
                edgecolors="black" if antenna == "rv2_coarse" else "none",
                linewidths=0.25,
                alpha=0.82,
                rasterized=True,
            )
        dur = float(sub["event_duration_min"].dropna().iloc[0]) if sub["event_duration_min"].notna().any() else 0.0
        ax.axvspan(0, dur, color="gold", alpha=0.09)
        ax.axvline(0, color="0.25", lw=0.8)
        ax.axvline(dur, color="0.25", lw=0.8)
        ax.set_yscale("log")
        ax.set_yticks(sorted(active["frequency_mhz"].unique()))
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
        ax.set_ylabel("MHz")
        ax.set_title(f"{row['historical_window_id']}  max daily z={float(row['max_daily_z']):+.2f}")
    axes[-1].set_xlabel("minutes from reported window start")
    fig.suptitle("Individual historical windows across all Ryle-Vonberg frequencies")
    fig.text(0.99, 0.01, "circles: upper V; triangles with outline: lower V", ha="right", fontsize=9)
    fig.subplots_adjust(left=0.075, right=0.84, top=0.955, bottom=0.045, hspace=0.78)
    if sc is not None:
        cax = fig.add_axes([0.87, 0.14, 0.018, 0.72])
        fig.colorbar(sc, cax=cax, label="daily-normalized log power")
    path = out_dir / "jupiter_top_windows_all_frequency_strips.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_guide(out_dir: Path, paths: list[Path], stack_dir: Path) -> Path:
    lines = [
        "# Jupiter Historical-Window Visual Diagnostics",
        "",
        "These plots are meant to be judged visually before trusting any scalar metric.",
        "",
        "## How To Read Them",
        "",
        "- `active_power_heatmap`: each cell is the median daily-normalized log power inside one historical window for one antenna/frequency.",
        "- `excess_heatmap`: the same cell after subtracting the matched shifted-control window median. A real repeatable activity selector should produce coherent warm bands or repeated warm rows, not isolated pixels.",
        "- `ladder`: each line is one historical window. Blue upward lines mean the historical window is higher than its shifted controls; red downward lines mean the opposite. A robust detection should have most lines moving upward.",
        "- `phase_raster`: actual samples plotted at their location within the reported historical window. This exposes whether a stack is based on dense repeated sampling or a few isolated points.",
        "- `all_frequency_strips`: raw sample layout for individual high-power windows across all Ryle-Vonberg frequencies and both antennas.",
        "",
        "## Source Stack Directory",
        "",
        f"`{stack_dir}`",
        "",
        "## Plots",
        "",
        *[f"- `{p}`" for p in paths],
    ]
    path = out_dir / "jupiter_historical_window_visual_diagnostics_guide.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack-dir", type=Path, default=DEFAULT_STACK_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--top-channels", type=int, default=6)
    parser.add_argument("--top-windows", type=int, default=10)
    args = parser.parse_args()

    out_dir = args.out_dir or (args.stack_dir / "visual_diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)
    paired, summary, active, control = load_inputs(args.stack_dir)

    paths: list[Path] = []
    paths.extend(plot_window_channel_heatmaps(paired, active, out_dir))
    paths.append(plot_ladder_controls(paired, summary, out_dir, top_n=int(args.top_channels)))
    paths.append(plot_phase_raster(active, summary, out_dir, top_n=int(args.top_channels)))
    paths.append(plot_multifrequency_window_strips(active, out_dir, top_n=int(args.top_windows)))
    guide = write_guide(out_dir, paths, args.stack_dir)
    print(guide)


if __name__ == "__main__":
    main()
