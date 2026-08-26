#!/usr/bin/env python
"""Make data-first plots for the lunar-reflection interference run."""

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
from rylevonberg.util import robust_sigma  # noqa: E402


ANT_LABEL = {"rv1_coarse": "upper V", "rv2_coarse": "lower V"}
EVENT_COLOR = {"disappearance": "#4c78a8", "reappearance": "#d95f02"}


def _read(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return read_table(path, low_memory=False)


def _robust_sem(values: pd.Series | np.ndarray) -> float:
    vals = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size <= 1:
        return np.nan
    scale = robust_sigma(vals - np.nanmedian(vals))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.nanstd(vals, ddof=1))
    return float(scale / np.sqrt(vals.size)) if np.isfinite(scale) and scale > 0 else np.nan


def _event_median_stack(frame: pd.DataFrame, x_col: str, value_col: str, group_cols: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    required = set(group_cols + ["event_id", x_col, value_col])
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    per_event_cols = group_cols + ["event_id", x_col]
    per_event = (
        frame.groupby(per_event_cols, as_index=False, dropna=False)
        .agg(event_value=(value_col, "median"), n_samples=(value_col, "size"))
    )
    rows = []
    for keys, grp in per_event.groupby(group_cols + [x_col], sort=True, dropna=False):
        vals = grp["event_value"]
        rows.append(
            {
                **dict(zip(group_cols + [x_col], keys)),
                "median_value": float(np.nanmedian(vals)),
                "value_sem": _robust_sem(vals),
                "n_events": int(len(grp)),
                "n_samples": int(grp["n_samples"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _binned_samples(samples: pd.DataFrame, time_bin_s: float, phase_bins: int) -> pd.DataFrame:
    work = samples.copy()
    work["t_bin_sec"] = np.round(pd.to_numeric(work["t_rel_sec"], errors="coerce").to_numpy(dtype=float) / time_bin_s) * time_bin_s
    work["oriented_t_bin_sec"] = (
        np.round(pd.to_numeric(work["oriented_t_sec"], errors="coerce").to_numpy(dtype=float) / time_bin_s) * time_bin_s
    )
    phase = pd.to_numeric(work["phase_wrapped_rad"], errors="coerce").to_numpy(dtype=float)
    phase_bin = np.full(len(work), -1, dtype=int)
    finite = np.isfinite(phase)
    phase_bin[finite] = np.floor((phase[finite] / (2.0 * np.pi)) * phase_bins).astype(int)
    work["phase_bin"] = np.clip(phase_bin, -1, phase_bins - 1)
    work["phase_cycle"] = np.where(work["phase_bin"].to_numpy(dtype=int) >= 0, (work["phase_bin"] + 0.5) / phase_bins, np.nan)
    return work


def _set_robust_ylim(ax: plt.Axes, y: pd.Series, pad: float = 0.15) -> None:
    vals = pd.to_numeric(y, errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size < 4:
        return
    lo, hi = np.nanpercentile(vals, [2, 98])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return
    span = hi - lo
    ax.set_ylim(lo - pad * span, hi + pad * span)


def plot_event_centered_normalized(samples: pd.DataFrame, out_dir: Path, antenna: str, time_bin_s: float) -> Path | None:
    sub = samples[samples["antenna"].astype(str).eq(str(antenna)) & samples["time_shift_s"].eq(0.0)].copy()
    if sub.empty:
        return None
    stack = _event_median_stack(
        sub,
        "t_bin_sec",
        "z_power",
        ["antenna", "frequency_mhz", "event_type"],
    )
    if stack.empty:
        return None
    freqs = sorted(stack["frequency_mhz"].dropna().unique())
    fig, axes = plt.subplots(len(freqs), 2, figsize=(12.5, max(8, 1.45 * len(freqs))), sharex=True, sharey=False)
    if len(freqs) == 1:
        axes = np.asarray([axes])
    for i, freq in enumerate(freqs):
        for j, event_type in enumerate(["disappearance", "reappearance"]):
            ax = axes[i, j]
            grp = stack[np.isclose(stack["frequency_mhz"], float(freq)) & stack["event_type"].astype(str).eq(event_type)].sort_values(
                "t_bin_sec"
            )
            ax.errorbar(
                grp["t_bin_sec"],
                grp["median_value"],
                yerr=grp["value_sem"],
                marker="o",
                markersize=2.0,
                lw=1.0,
                elinewidth=0.55,
                capsize=1.0,
                color=EVENT_COLOR[event_type],
            )
            ax.axvline(0.0, color="black", lw=0.7, ls="--")
            ax.axhline(0.0, color="0.65", lw=0.7)
            ax.set_title(f"{float(freq):.2f} MHz {event_type}", fontsize=9)
            ax.set_ylabel("normalized power")
            _set_robust_ylim(ax, grp["median_value"])
            if i == len(freqs) - 1:
                ax.set_xlabel("seconds from predicted event")
            ax.grid(alpha=0.22)
    fig.suptitle(f"Event-centered normalized power, {ANT_LABEL.get(antenna, antenna)}; {time_bin_s:g} s bins")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = out_dir / f"data_view_{antenna}_event_centered_normalized_power.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_oriented_normalized(samples: pd.DataFrame, out_dir: Path, antenna: str, time_bin_s: float) -> Path | None:
    sub = samples[samples["antenna"].astype(str).eq(str(antenna)) & samples["time_shift_s"].eq(0.0)].copy()
    if sub.empty:
        return None
    stack = _event_median_stack(
        sub,
        "oriented_t_bin_sec",
        "z_power",
        ["antenna", "frequency_mhz"],
    )
    if stack.empty:
        return None
    fig, ax = plt.subplots(figsize=(12.0, 5.8))
    for freq, grp in stack.groupby("frequency_mhz", sort=True):
        g = grp.sort_values("oriented_t_bin_sec")
        ax.plot(g["oriented_t_bin_sec"], g["median_value"], marker="o", markersize=2.2, lw=1.25, label=f"{float(freq):.2f} MHz")
    ax.axvline(0.0, color="black", lw=0.8, ls="--")
    ax.axhline(0.0, color="0.65", lw=0.7)
    ax.set_xlabel("seconds from event; positive is direct-visible side")
    ax.set_ylabel("normalized power")
    ax.grid(alpha=0.24)
    ax.legend(frameon=False, ncol=3, fontsize=8)
    fig.suptitle(f"Disappearance/reappearance combined in event-centered coordinates, {ANT_LABEL.get(antenna, antenna)}")
    fig.tight_layout()
    path = out_dir / f"data_view_{antenna}_oriented_normalized_power.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_phase_folded_data(
    samples: pd.DataFrame,
    out_dir: Path,
    antenna: str,
    value_col: str,
    label: str,
    fit_side: str,
) -> Path | None:
    sub = samples[samples["antenna"].astype(str).eq(str(antenna)) & samples["time_shift_s"].eq(0.0)].copy()
    if sub.empty:
        return None
    sub = sub[sub["reflection_valid"].astype(bool) & (sub["phase_bin"].to_numpy(dtype=int) >= 0)].copy()
    if fit_side == "visible":
        sub = sub[sub["event_visible_side"].astype(bool) & sub["direct_visible"].astype(bool)].copy()
    elif fit_side == "occulted":
        sub = sub[(~sub["event_visible_side"].astype(bool)) & (~sub["direct_visible"].astype(bool))].copy()
    if sub.empty:
        return None
    stack = _event_median_stack(
        sub,
        "phase_cycle",
        value_col,
        ["antenna", "frequency_mhz"],
    )
    if stack.empty:
        return None

    freqs = sorted(stack["frequency_mhz"].dropna().unique())
    fig, axes = plt.subplots(len(freqs), 1, figsize=(12.0, max(7, 1.45 * len(freqs))), sharex=True, sharey=False)
    axes = np.asarray(axes).reshape(-1)
    for ax, freq in zip(axes, freqs):
        grp = stack[np.isclose(stack["frequency_mhz"], float(freq))].sort_values("phase_cycle")
        ax.errorbar(
            grp["phase_cycle"],
            grp["median_value"],
            yerr=grp["value_sem"],
            marker="o",
            markersize=2.8,
            lw=1.0,
            elinewidth=0.6,
            capsize=1.0,
            color="#4c78a8",
        )
        ax.axhline(0.0, color="0.65", lw=0.7)
        ax.set_title(f"{float(freq):.2f} MHz", fontsize=9)
        ax.set_ylabel(label)
        _set_robust_ylim(ax, grp["median_value"])
        ax.grid(alpha=0.24)
    axes[-1].set_xlabel("wrapped reflection phase (cycles)")
    fig.suptitle(f"Phase-folded {label}, {ANT_LABEL.get(antenna, antenna)}; {fit_side} side")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    suffix = value_col.replace("_", "-")
    path = out_dir / f"data_view_{antenna}_phase_folded_{suffix}_{fit_side}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_045_summary(samples: pd.DataFrame, grid: pd.DataFrame, out_dir: Path, antenna: str) -> Path | None:
    sub = samples[
        samples["antenna"].astype(str).eq(str(antenna))
        & samples["time_shift_s"].eq(0.0)
        & np.isclose(pd.to_numeric(samples["frequency_mhz"], errors="coerce"), 0.45)
    ].copy()
    if sub.empty:
        return None
    sample_time = _event_median_stack(sub, "t_bin_sec", "z_power", ["event_type"])
    phase = sub[sub["reflection_valid"].astype(bool) & (sub["phase_bin"].to_numpy(dtype=int) >= 0)].copy()
    phase = phase[phase["event_visible_side"].astype(bool) & phase["direct_visible"].astype(bool)].copy()
    phase_stack = _event_median_stack(phase, "phase_cycle", "z_resid", [])

    grid_sub = grid[
        np.isclose(pd.to_numeric(grid["frequency_mhz"], errors="coerce"), 0.45)
        & grid["antenna"].astype(str).eq(str(antenna))
    ].copy()
    if grid_sub.empty or sample_time.empty:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.0))
    for event_type, grp in grid_sub.groupby("event_type", sort=True):
        g = grp.sort_values("t_bin_sec")
        axes[0, 0].errorbar(
            g["t_bin_sec"],
            g["median_z_power"],
            yerr=g["median_z_power_err"],
            marker="o",
            markersize=2.2,
            lw=1.0,
            color=EVENT_COLOR.get(str(event_type)),
            label=event_type,
        )
    axes[0, 0].set_title("Existing all-frequency grid, 60 s bins")
    axes[0, 0].set_xlabel("seconds from event")
    axes[0, 0].set_ylabel("normalized power")
    axes[0, 0].legend(frameon=False, fontsize=8)

    for event_type, grp in sample_time.groupby("event_type", sort=True):
        g = grp.sort_values("t_bin_sec")
        axes[0, 1].errorbar(
            g["t_bin_sec"],
            g["median_value"],
            yerr=g["value_sem"],
            marker="o",
            markersize=2.0,
            lw=1.0,
            color=EVENT_COLOR.get(str(event_type)),
            label=event_type,
        )
    axes[0, 1].set_title("Sample-level extraction, 10 s bins")
    axes[0, 1].set_xlabel("seconds from event")
    axes[0, 1].set_ylabel("normalized power")

    oriented = _event_median_stack(sub, "oriented_t_bin_sec", "z_power", [])
    if not oriented.empty:
        g = oriented.sort_values("oriented_t_bin_sec")
        axes[1, 0].errorbar(g["oriented_t_bin_sec"], g["median_value"], yerr=g["value_sem"], marker="o", markersize=2.0, lw=1.0)
    axes[1, 0].set_title("Event-type combined; positive is direct-visible side")
    axes[1, 0].set_xlabel("oriented seconds from event")
    axes[1, 0].set_ylabel("normalized power")

    if not phase_stack.empty:
        g = phase_stack.sort_values("phase_cycle")
        axes[1, 1].errorbar(g["phase_cycle"], g["median_value"], yerr=g["value_sem"], marker="o", markersize=2.8, lw=1.0)
    axes[1, 1].set_title("Phase-folded residuals; visible side")
    axes[1, 1].set_xlabel("wrapped reflection phase (cycles)")
    axes[1, 1].set_ylabel("baseline residual z")

    for ax in axes.ravel():
        ax.axhline(0.0, color="0.65", lw=0.7)
        ax.grid(alpha=0.24)
    axes[0, 0].axvline(0.0, color="black", lw=0.75, ls="--")
    axes[0, 1].axvline(0.0, color="black", lw=0.75, ls="--")
    axes[1, 0].axvline(0.0, color="black", lw=0.75, ls="--")
    fig.suptitle(f"0.45 MHz lower-V data views")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = out_dir / f"data_view_{antenna}_045mhz_drilldown.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(ROOT / "outputs/lower_v_lunar_reflection_interference_v1"))
    parser.add_argument("--grid-summary", default=str(ROOT / "outputs/all_frequency_profile_grids_v1/earth_all_frequency_profile_summary_900s.csv"))
    parser.add_argument("--antennas", default="rv2_coarse")
    parser.add_argument("--time-bin-s", type=float, default=10.0)
    parser.add_argument("--phase-bins", type=int, default=24)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    samples = _read(out_dir / "reflection_event_samples.npy")
    grid = _read(Path(args.grid_summary))
    if samples.empty:
        raise SystemExit(f"No samples found in {out_dir}")
    samples = _binned_samples(samples, float(args.time_bin_s), int(args.phase_bins))
    antennas = [x.strip() for x in str(args.antennas).split(",") if x.strip()]

    for antenna in antennas:
        paths = [
            plot_event_centered_normalized(samples, out_dir, antenna, float(args.time_bin_s)),
            plot_oriented_normalized(samples, out_dir, antenna, float(args.time_bin_s)),
            plot_phase_folded_data(samples, out_dir, antenna, "z_power", "normalized power", "visible"),
            plot_phase_folded_data(samples, out_dir, antenna, "z_resid", "baseline residual z", "visible"),
            plot_045_summary(samples, grid, out_dir, antenna),
        ]
        for path in paths:
            if path is not None:
                print(path)


if __name__ == "__main__":
    main()
