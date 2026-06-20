#!/usr/bin/env python
"""Stack RAE-2 samples selected by historical Jupiter activity windows.

This is intentionally not an occultation analysis.  The Warwick/Dulk/Riddle
windows are used as an external selector for times when Jupiter was active in
ground-based data.  RAE-2 samples inside those windows are compared with
duration-matched windows shifted in time.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rylevonberg.table_io import read_table

from run_jupiter_historical_window_phase_survey import (
    ANTENNA_LABEL,
    _intervals_from_windows,
    assign_intervals,
    build_shifted_controls,
    load_windows,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLES = ROOT / "outputs/jupiter_phase_pattern_survey_v1/jupiter_phase_pattern_sampled_points.csv"
DEFAULT_WINDOWS = ROOT / "configs/jupiter_warwick_dulk_riddle_1975_active_windows.csv"
DEFAULT_OUT = ROOT / "outputs/jupiter_historical_window_stacks_v1"


def _safe_name(value: object) -> str:
    return str(value).replace(" ", "_").replace(".", "p").replace("/", "_")


def _bootstrap_ci(values: np.ndarray, seed: int, n_boot: int = 2000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    if len(values) == 1:
        return float(values[0]), np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(int(n_boot), len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _window_metadata(windows: pd.DataFrame) -> pd.DataFrame:
    meta = windows.copy()
    for col in ["event_start_time", "event_end_time", "expanded_start_time", "expanded_end_time"]:
        meta[col] = pd.to_datetime(meta[col])
    meta["event_duration_min"] = (
        meta["event_end_time"] - meta["event_start_time"]
    ).dt.total_seconds() / 60.0
    meta["event_center_time"] = meta["event_start_time"] + (
        meta["event_end_time"] - meta["event_start_time"]
    ) / 2
    return meta


def label_samples(
    samples: pd.DataFrame,
    windows: pd.DataFrame,
    label_col: str,
    parent_col: str = "historical_window_id",
) -> pd.DataFrame:
    interval_windows = windows.copy()
    if label_col != "historical_window_id":
        interval_windows["historical_window_id"] = interval_windows[label_col]
    labels = assign_intervals(
        samples["time"], _intervals_from_windows(interval_windows, "expanded_start_time", "expanded_end_time")
    )
    selected = samples[labels.astype(str).ne("")].copy()
    selected[label_col] = labels[labels.astype(str).ne("")].to_numpy()
    meta = _window_metadata(windows).copy()
    if label_col != "historical_window_id":
        meta = meta.rename(columns={label_col: label_col})
    if parent_col not in meta.columns:
        meta[parent_col] = meta[label_col]
    keep_cols = [
        label_col,
        parent_col,
        "event_start_time",
        "event_end_time",
        "expanded_start_time",
        "expanded_end_time",
        "event_duration_min",
        "event_center_time",
        "intensity",
        "burstiness",
        "reported_freq_range_mhz",
    ]
    keep_cols = list(dict.fromkeys(keep_cols))
    selected = selected.merge(meta[keep_cols], on=label_col, how="left")
    selected["dt_from_start_min"] = (
        pd.to_datetime(selected["time"]) - pd.to_datetime(selected["event_start_time"])
    ).dt.total_seconds() / 60.0
    selected["dt_from_center_min"] = (
        pd.to_datetime(selected["time"]) - pd.to_datetime(selected["event_center_time"])
    ).dt.total_seconds() / 60.0
    selected["window_phase"] = selected["dt_from_start_min"] / selected["event_duration_min"]
    selected["inside_reported_window"] = (
        (pd.to_datetime(selected["time"]) >= pd.to_datetime(selected["event_start_time"]))
        & (pd.to_datetime(selected["time"]) <= pd.to_datetime(selected["event_end_time"]))
    )
    return selected


def make_window_level_table(active: pd.DataFrame, control: pd.DataFrame) -> pd.DataFrame:
    keys = ["antenna", "frequency_band", "frequency_mhz"]
    active_core = active[active["inside_reported_window"]].copy()
    control_core = control[control["inside_reported_window"]].copy()

    aw = (
        active_core.groupby(["historical_window_id", *keys], sort=True)
        .agg(
            active_n_samples=("daily_z_log_power", "size"),
            active_median_daily_z=("daily_z_log_power", "median"),
            active_mean_daily_z=("daily_z_log_power", "mean"),
            active_max_daily_z=("daily_z_log_power", "max"),
            active_median_log_power=("log_power", "median"),
            active_high_tail_fraction=("daily_z_log_power", lambda s: float((s > 2.5).mean())),
            intensity=("intensity", "first"),
            burstiness=("burstiness", "first"),
        )
        .reset_index()
    )
    cw = (
        control_core.groupby(["historical_window_id", *keys], sort=True)
        .agg(
            shifted_n_samples=("daily_z_log_power", "size"),
            shifted_median_daily_z=("daily_z_log_power", "median"),
            shifted_mean_daily_z=("daily_z_log_power", "mean"),
            shifted_max_daily_z=("daily_z_log_power", "max"),
            shifted_median_log_power=("log_power", "median"),
            shifted_high_tail_fraction=("daily_z_log_power", lambda s: float((s > 2.5).mean())),
        )
        .reset_index()
    )
    paired = aw.merge(cw, on=["historical_window_id", *keys], how="inner")
    paired["paired_median_daily_z_excess"] = paired["active_median_daily_z"] - paired["shifted_median_daily_z"]
    paired["paired_mean_daily_z_excess"] = paired["active_mean_daily_z"] - paired["shifted_mean_daily_z"]
    paired["paired_high_tail_fraction_excess"] = (
        paired["active_high_tail_fraction"] - paired["shifted_high_tail_fraction"]
    )
    return paired


def summarize_paired_window_stack(paired: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows = []
    for (antenna, band, freq), grp in paired.groupby(["antenna", "frequency_band", "frequency_mhz"], sort=True):
        values = grp["paired_median_daily_z_excess"].to_numpy(dtype=float)
        mean_excess, lo, hi = _bootstrap_ci(values, seed=seed + int(band) * 13 + (1 if antenna == "rv1_coarse" else 2))
        rows.append(
            {
                "antenna": antenna,
                "antenna_label": ANTENNA_LABEL.get(str(antenna), str(antenna)),
                "frequency_band": int(band),
                "frequency_mhz": float(freq),
                "n_paired_windows": int(len(grp)),
                "active_total_samples": int(grp["active_n_samples"].sum()),
                "shifted_total_samples": int(grp["shifted_n_samples"].sum()),
                "active_window_median_daily_z_median": float(grp["active_median_daily_z"].median()),
                "shifted_window_median_daily_z_median": float(grp["shifted_median_daily_z"].median()),
                "paired_median_daily_z_excess_mean": mean_excess,
                "paired_median_daily_z_excess_boot_lo": lo,
                "paired_median_daily_z_excess_boot_hi": hi,
                "paired_median_daily_z_excess_median": float(np.nanmedian(values)) if len(values) else np.nan,
                "positive_window_excess_fraction": float((grp["paired_median_daily_z_excess"] > 0).mean()),
                "active_high_tail_fraction_mean": float(grp["active_high_tail_fraction"].mean()),
                "shifted_high_tail_fraction_mean": float(grp["shifted_high_tail_fraction"].mean()),
                "paired_high_tail_fraction_excess_mean": float(grp["paired_high_tail_fraction_excess"].mean()),
                "active_max_daily_z_median": float(grp["active_max_daily_z"].median()),
                "active_max_daily_z_max": float(grp["active_max_daily_z"].max()),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["abs_paired_median_daily_z_excess_mean"] = out["paired_median_daily_z_excess_mean"].abs()
    return out


def make_phase_profile_table(active: pd.DataFrame, control: pd.DataFrame, phase_bin_width: float) -> pd.DataFrame:
    pieces = []
    for label, df, id_col in [
        ("historical_active", active[active["inside_reported_window"]].copy(), "historical_window_id"),
        ("shifted_control", control[control["inside_reported_window"]].copy(), "shifted_control_window_id"),
    ]:
        if df.empty:
            continue
        df["phase_bin"] = (
            np.floor(df["window_phase"].clip(0, 1 - 1e-9) / float(phase_bin_width)) * float(phase_bin_width)
            + 0.5 * float(phase_bin_width)
        )
        window_bin = (
            df.groupby([id_col, "antenna", "frequency_band", "frequency_mhz", "phase_bin"], sort=True)
            .agg(
                window_bin_median_daily_z=("daily_z_log_power", "median"),
                window_bin_median_log_power=("log_power", "median"),
                n_samples=("daily_z_log_power", "size"),
            )
            .reset_index()
        )
        summary = (
            window_bin.groupby(["antenna", "frequency_band", "frequency_mhz", "phase_bin"], sort=True)
            .agg(
                median_daily_z=("window_bin_median_daily_z", "median"),
                mean_daily_z=("window_bin_median_daily_z", "mean"),
                q16_daily_z=("window_bin_median_daily_z", lambda s: float(np.nanquantile(s, 0.16))),
                q84_daily_z=("window_bin_median_daily_z", lambda s: float(np.nanquantile(s, 0.84))),
                n_windows=("window_bin_median_daily_z", "size"),
                n_samples=("n_samples", "sum"),
                median_log_power=("window_bin_median_log_power", "median"),
            )
            .reset_index()
        )
        summary["selector"] = label
        pieces.append(summary)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def make_edge_profile_table(
    active: pd.DataFrame,
    control: pd.DataFrame,
    edge_window_min: float,
    edge_bin_min: float,
) -> pd.DataFrame:
    pieces = []
    for selector, df, id_col in [
        ("historical_active", active.copy(), "historical_window_id"),
        ("shifted_control", control.copy(), "shifted_control_window_id"),
    ]:
        if df.empty:
            continue
        work = df.copy()
        work["dt_from_end_min"] = (
            pd.to_datetime(work["time"]) - pd.to_datetime(work["event_end_time"])
        ).dt.total_seconds() / 60.0
        for edge, coord_col, lo, hi in [
            ("start", "dt_from_start_min", -float(edge_bin_min) * 3.0, float(edge_window_min)),
            ("end", "dt_from_end_min", -float(edge_window_min), float(edge_bin_min) * 3.0),
        ]:
            sub = work[(work[coord_col] >= lo) & (work[coord_col] <= hi)].copy()
            if sub.empty:
                continue
            sub["edge_bin_min"] = (
                np.floor(sub[coord_col] / float(edge_bin_min)) * float(edge_bin_min)
                + 0.5 * float(edge_bin_min)
            )
            window_bin = (
                sub.groupby([id_col, "antenna", "frequency_band", "frequency_mhz", "edge_bin_min"], sort=True)
                .agg(
                    window_bin_median_daily_z=("daily_z_log_power", "median"),
                    n_samples=("daily_z_log_power", "size"),
                )
                .reset_index()
            )
            summary = (
                window_bin.groupby(["antenna", "frequency_band", "frequency_mhz", "edge_bin_min"], sort=True)
                .agg(
                    median_daily_z=("window_bin_median_daily_z", "median"),
                    q16_daily_z=("window_bin_median_daily_z", lambda s: float(np.nanquantile(s, 0.16))),
                    q84_daily_z=("window_bin_median_daily_z", lambda s: float(np.nanquantile(s, 0.84))),
                    n_windows=("window_bin_median_daily_z", "size"),
                    n_samples=("n_samples", "sum"),
                )
                .reset_index()
            )
            summary["selector"] = selector
            summary["edge"] = edge
            pieces.append(summary)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def plot_excess_spectrum(summary: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10.0, 5.4))
    for antenna, sub in summary.sort_values("frequency_mhz").groupby("antenna", sort=True):
        y = sub["paired_median_daily_z_excess_mean"].to_numpy(dtype=float)
        lo = sub["paired_median_daily_z_excess_boot_lo"].to_numpy(dtype=float)
        hi = sub["paired_median_daily_z_excess_boot_hi"].to_numpy(dtype=float)
        err = np.vstack([y - lo, hi - y])
        err[~np.isfinite(err)] = 0.0
        ax.errorbar(
            sub["frequency_mhz"],
            y,
            yerr=err,
            marker="o",
            lw=1.4,
            capsize=3,
            label=ANTENNA_LABEL.get(str(antenna), str(antenna)),
        )
    ax.axhline(0, color="0.25", lw=1.0)
    ax.set_xscale("log")
    ax.set_xticks(sorted(summary["frequency_mhz"].unique()))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
    ax.set_xlabel("Ryle-Vonberg frequency (MHz)")
    ax.set_ylabel("active-window excess daily-normalized log power\n(mean paired window-median difference)")
    ax.set_title("Jupiter historical-window selected stack: active windows vs shifted controls")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "jupiter_historical_window_selected_excess_spectrum.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_edge_aligned_top_channels(edge_profile: pd.DataFrame, summary: pd.DataFrame, out_dir: Path, top_n: int) -> Path | None:
    if edge_profile.empty or summary.empty:
        return None
    top = summary.sort_values("paired_median_daily_z_excess_mean", ascending=False).head(int(top_n))
    if top.empty:
        return None
    fig, axes = plt.subplots(len(top), 2, figsize=(13.0, max(2.8 * len(top), 5.0)), sharey=True)
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes.reshape(1, -1)
    for row_i, (_, channel) in enumerate(top.iterrows()):
        for col_i, edge in enumerate(["start", "end"]):
            ax = axes[row_i, col_i]
            for selector, color, label in [
                ("historical_active", "tab:blue", "historical active"),
                ("shifted_control", "tab:orange", "shifted controls"),
            ]:
                sub = edge_profile[
                    edge_profile["selector"].eq(selector)
                    & edge_profile["edge"].eq(edge)
                    & edge_profile["antenna"].astype(str).eq(str(channel["antenna"]))
                    & np.isclose(edge_profile["frequency_mhz"].astype(float), float(channel["frequency_mhz"]))
                ].sort_values("edge_bin_min")
                if sub.empty:
                    continue
                ax.plot(
                    sub["edge_bin_min"],
                    sub["median_daily_z"],
                    marker="o",
                    ms=3,
                    lw=1.2,
                    color=color,
                    label=label,
                )
                ax.fill_between(sub["edge_bin_min"], sub["q16_daily_z"], sub["q84_daily_z"], color=color, alpha=0.14)
            ax.axhline(0, color="0.35", lw=0.7)
            ax.axvline(0, color="0.2", lw=0.85)
            if row_i == 0:
                ax.set_title(f"aligned to reported window {edge}")
            if col_i == 0:
                ax.set_ylabel(
                    f"{ANTENNA_LABEL.get(str(channel['antenna']), channel['antenna'])}\n"
                    f"{float(channel['frequency_mhz']):.2f} MHz\nmedian daily z"
                )
            ax.set_xlabel(f"minutes from reported {edge}")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.985, 0.985))
    fig.suptitle("Historical Jupiter windows: edge-aligned selected stacks")
    fig.tight_layout(rect=[0, 0, 0.91, 0.965])
    path = out_dir / "jupiter_historical_window_edge_aligned_top_channels.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_phase_profile_grid(profile: pd.DataFrame, antenna: str, out_dir: Path) -> Path | None:
    suba = profile[profile["antenna"].astype(str).eq(antenna)].copy()
    if suba.empty:
        return None
    freqs = sorted(suba["frequency_mhz"].unique())
    ncols = 3
    nrows = int(np.ceil(len(freqs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.0, max(3.2 * nrows, 4.5)), sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(-1)
    for ax, freq in zip(axes, freqs):
        for selector, color, label in [
            ("historical_active", "tab:blue", "historical active"),
            ("shifted_control", "tab:orange", "shifted controls"),
        ]:
            sub = suba[(suba["selector"] == selector) & np.isclose(suba["frequency_mhz"], float(freq))]
            sub = sub.sort_values("phase_bin")
            if sub.empty:
                continue
            ax.plot(sub["phase_bin"], sub["median_daily_z"], marker="o", ms=3, lw=1.2, color=color, label=label)
            ax.fill_between(sub["phase_bin"], sub["q16_daily_z"], sub["q84_daily_z"], color=color, alpha=0.16, lw=0)
        ax.axhline(0, color="0.3", lw=0.7)
        ax.set_title(f"{float(freq):.2f} MHz")
        ax.set_xlim(0, 1)
        ax.set_xlabel("fraction through historical window")
        ax.set_ylabel("window-median daily z")
    for ax in axes[len(freqs) :]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.975, 0.975))
    fig.suptitle(f"Historical Jupiter active-window stack, {ANTENNA_LABEL.get(antenna, antenna)}")
    fig.tight_layout(rect=[0, 0, 0.93, 0.95])
    path = out_dir / f"jupiter_historical_window_phase_stack_grid_{antenna}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_paired_window_differences(paired: pd.DataFrame, summary: pd.DataFrame, out_dir: Path, top_n: int) -> Path:
    top = summary.sort_values("paired_median_daily_z_excess_mean", ascending=False).head(int(top_n))
    n = len(top)
    fig, axes = plt.subplots(n, 1, figsize=(11.5, max(2.6 * n, 4.5)), sharex=False)
    axes = np.atleast_1d(axes)
    for ax, (_, row) in zip(axes, top.iterrows()):
        sub = paired[
            paired["antenna"].astype(str).eq(str(row["antenna"]))
            & np.isclose(paired["frequency_mhz"].astype(float), float(row["frequency_mhz"]))
        ].copy()
        sub = sub.sort_values("historical_window_id")
        x = np.arange(len(sub))
        colors = np.where(sub["paired_median_daily_z_excess"] >= 0, "tab:blue", "tab:red")
        ax.bar(x, sub["paired_median_daily_z_excess"], color=colors, alpha=0.78)
        ax.axhline(0, color="0.25", lw=0.8)
        ax.set_ylabel("active - shifted\nmedian z")
        ax.set_title(
            f"{ANTENNA_LABEL.get(str(row['antenna']), row['antenna'])} "
            f"{float(row['frequency_mhz']):.2f} MHz; "
            f"mean={float(row['paired_median_daily_z_excess_mean']):+.3f}, "
            f"positive windows={float(row['positive_window_excess_fraction']):.2f}"
        )
        step = max(1, len(sub) // 12)
        ax.set_xticks(x[::step])
        ax.set_xticklabels(sub["historical_window_id"].iloc[::step], rotation=45, ha="right")
    axes[-1].set_xlabel("historical active window")
    fig.suptitle("Window-by-window contribution to the historical Jupiter selector stack")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = out_dir / "jupiter_historical_window_paired_differences_top_channels.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_individual_channel(
    active: pd.DataFrame,
    channel_row: pd.Series,
    out_dir: Path,
    n_windows: int,
    y_col: str,
    y_label: str,
    suffix: str,
) -> Path | None:
    antenna = str(channel_row["antenna"])
    freq = float(channel_row["frequency_mhz"])
    chan = active[
        active["antenna"].astype(str).eq(antenna) & np.isclose(active["frequency_mhz"].astype(float), freq)
    ].copy()
    if chan.empty:
        return None
    core_stats = (
        chan[chan["inside_reported_window"]]
        .groupby("historical_window_id", sort=True)
        .agg(
            median_daily_z=("daily_z_log_power", "median"),
            max_daily_z=("daily_z_log_power", "max"),
            event_duration_min=("event_duration_min", "first"),
            intensity=("intensity", "first"),
            burstiness=("burstiness", "first"),
        )
        .reset_index()
        .sort_values(["median_daily_z", "max_daily_z"], ascending=False)
        .head(int(n_windows))
    )
    if core_stats.empty:
        return None
    n = len(core_stats)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.0, max(3.0 * nrows, 4.2)), sharex=False, sharey=False)
    axes = np.asarray(axes).reshape(-1)
    for ax, (_, row) in zip(axes, core_stats.iterrows()):
        sub = chan[chan["historical_window_id"].astype(str).eq(str(row["historical_window_id"]))].sort_values("time")
        ax.scatter(sub["dt_from_start_min"], sub[y_col], s=14, color="tab:blue", alpha=0.78)
        ax.axvspan(0, float(row["event_duration_min"]), color="gold", alpha=0.14, label="reported active window")
        ax.axvline(0, color="0.2", lw=0.8)
        ax.axvline(float(row["event_duration_min"]), color="0.2", lw=0.8)
        if y_col == "daily_z_log_power":
            ax.axhline(0, color="0.35", lw=0.7)
            ax.axhline(2.5, color="crimson", lw=0.8, ls="--")
        ax.set_title(
            f"{row['historical_window_id']}  I={int(row['intensity'])} {row['burstiness']}; "
            f"median z={float(row['median_daily_z']):+.2f}"
        )
        ax.set_xlabel("minutes from reported window start")
        ax.set_ylabel(y_label)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(
        f"Individual historical Jupiter windows: {ANTENNA_LABEL.get(antenna, antenna)} {freq:.2f} MHz"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    path = out_dir / f"individual_{_safe_name(antenna)}_{_safe_name(f'{freq:.2f}mhz')}_{suffix}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_individual_windows(
    active: pd.DataFrame,
    summary: pd.DataFrame,
    out_dir: Path,
    n_channels: int,
    n_windows: int,
) -> list[Path]:
    indiv_dir = out_dir / "individual_windows"
    indiv_dir.mkdir(parents=True, exist_ok=True)
    top = summary.sort_values("paired_median_daily_z_excess_mean", ascending=False).head(int(n_channels))
    paths: list[Path] = []
    for _, row in top.iterrows():
        raw = _plot_individual_channel(
            active,
            row,
            indiv_dir,
            n_windows=n_windows,
            y_col="log_power",
            y_label="raw log power",
            suffix="raw_log_power",
        )
        norm = _plot_individual_channel(
            active,
            row,
            indiv_dir,
            n_windows=n_windows,
            y_col="daily_z_log_power",
            y_label="daily-normalized log power",
            suffix="daily_z",
        )
        paths.extend([p for p in [raw, norm] if p is not None])
    return paths


def write_report(
    out_dir: Path,
    windows: pd.DataFrame,
    active: pd.DataFrame,
    control: pd.DataFrame,
    paired: pd.DataFrame,
    summary: pd.DataFrame,
    paths: list[Path],
    config: dict[str, object],
) -> Path:
    top_cols = [
        "antenna_label",
        "frequency_mhz",
        "n_paired_windows",
        "active_total_samples",
        "shifted_total_samples",
        "paired_median_daily_z_excess_mean",
        "paired_median_daily_z_excess_boot_lo",
        "paired_median_daily_z_excess_boot_hi",
        "positive_window_excess_fraction",
        "paired_high_tail_fraction_excess_mean",
    ]
    top_positive = summary.sort_values("paired_median_daily_z_excess_mean", ascending=False).head(12)
    top_negative = summary.sort_values("paired_median_daily_z_excess_mean", ascending=True).head(8)
    lines = [
        "# Jupiter Historical-Window Selected Stacks",
        "",
        "This analysis uses the Warwick/Dulk/Riddle historical Jupiter activity windows as an external selector.",
        "It does not assume an occultation time.  Instead, it asks whether RAE-2 power is systematically higher inside those independently reported active intervals than inside duration-matched windows shifted by the requested control offsets.",
        "",
        "## Configuration",
        "",
        *[f"- `{k}`: `{v}`" for k, v in config.items()],
        "",
        "## Coverage",
        "",
        f"- Historical windows loaded: `{len(windows)}`",
        f"- Active-window selected samples, including padding: `{len(active)}`",
        f"- Shifted-control selected samples, including padding: `{len(control)}`",
        f"- Paired active/control window-channel rows: `{len(paired)}`",
        f"- Historical windows contributing at least one paired channel: `{paired['historical_window_id'].nunique() if not paired.empty else 0}`",
        "",
        "## How The Stack Is Built",
        "",
        "For each frequency and antenna, the script computes the median daily-normalized log power inside each historical active window.  It then computes the same quantity for the same-duration shifted control windows, paired by historical window ID.",
        "",
        "The primary plotted quantity is:",
        "",
        "`active-window excess = median_z(active historical window) - median_z(shifted control windows)`",
        "",
        "The spectrum plot averages this paired excess across historical windows.  Error bars are bootstrap intervals over historical windows, not sample-by-sample errors.",
        "",
        "## Strongest Positive Window-Selected Excesses",
        "",
        top_positive[top_cols].to_string(index=False),
        "",
        "## Strongest Negative Window-Selected Excesses",
        "",
        top_negative[top_cols].to_string(index=False),
        "",
        "## Interpretation",
        "",
        "- This is a stricter test than just plotting high-power samples in Io-CML space, because the comparison uses inactive windows with matching duration and similar observing cadence.",
        "- A convincing Jupiter signature would show positive excess across several adjacent frequencies, consistency across many historical windows, and individual-window plots where high points are not dominated by one or two windows.",
        "- The output should still be treated as evidence for or against activity-correlated excess, not proof of a Jovian source by itself.",
        "",
        "## Files",
        "",
        *[f"- `{p}`" for p in paths],
    ]
    path = out_dir / "jupiter_historical_window_selected_stack_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--historical-windows", type=Path, default=DEFAULT_WINDOWS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--padding-min", type=float, default=30.0)
    parser.add_argument("--control-shift-days", type=int, nargs="*", default=[-7, 7])
    parser.add_argument("--min-intensity", type=int, default=None)
    parser.add_argument("--burstiness", nargs="*", default=None, help="Optional allowed historical burstiness labels.")
    parser.add_argument("--phase-bin-width", type=float, default=0.1)
    parser.add_argument("--edge-window-min", type=float, default=120.0)
    parser.add_argument("--edge-bin-min", type=float, default=10.0)
    parser.add_argument("--individual-channel-count", type=int, default=4)
    parser.add_argument("--individual-windows-per-channel", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260609)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    windows = load_windows(args.historical_windows, padding_min=float(args.padding_min))
    windows = _window_metadata(windows)
    if args.min_intensity is not None:
        windows = windows[windows["intensity"].astype(int) >= int(args.min_intensity)].copy()
    if args.burstiness:
        allowed_burstiness = {str(x).strip().upper() for x in args.burstiness}
        windows = windows[windows["burstiness"].astype(str).str.upper().isin(allowed_burstiness)].copy()

    samples = read_table(args.samples, parse_dates=["time"]).sort_values("time").reset_index(drop=True)

    active = label_samples(samples, windows, "historical_window_id", parent_col="historical_window_id")
    shifted = build_shifted_controls(windows, list(args.control_shift_days), padding_min=float(args.padding_min))
    shifted["shifted_control_window_id"] = shifted["historical_window_id"]
    shifted["historical_window_id"] = shifted["shifted_control_window_id"].str.replace(r"_shift_[+-]\d+d$", "", regex=True)
    control = label_samples(samples, shifted, "shifted_control_window_id", parent_col="historical_window_id")

    paired = make_window_level_table(active, control)
    summary = summarize_paired_window_stack(paired, seed=int(args.seed))
    profile = make_phase_profile_table(active, control, phase_bin_width=float(args.phase_bin_width))
    edge_profile = make_edge_profile_table(
        active,
        control,
        edge_window_min=float(args.edge_window_min),
        edge_bin_min=float(args.edge_bin_min),
    )

    active.to_csv(args.out_dir / "jupiter_historical_window_selected_active_samples.csv", index=False)
    control.to_csv(args.out_dir / "jupiter_historical_window_selected_shifted_control_samples.csv", index=False)
    paired.to_csv(args.out_dir / "jupiter_historical_window_paired_window_stats.csv", index=False)
    summary.to_csv(args.out_dir / "jupiter_historical_window_selected_stack_summary.csv", index=False)
    profile.to_csv(args.out_dir / "jupiter_historical_window_phase_stack_profiles.csv", index=False)
    edge_profile.to_csv(args.out_dir / "jupiter_historical_window_edge_stack_profiles.csv", index=False)

    paths: list[Path] = [
        plot_excess_spectrum(summary, args.out_dir),
        plot_paired_window_differences(paired, summary, args.out_dir, top_n=6),
    ]
    for antenna in ["rv1_coarse", "rv2_coarse"]:
        p = plot_phase_profile_grid(profile, antenna, args.out_dir)
        if p is not None:
            paths.append(p)
    edge_plot = plot_edge_aligned_top_channels(edge_profile, summary, args.out_dir, top_n=6)
    if edge_plot is not None:
        paths.append(edge_plot)
    paths.extend(
        plot_individual_windows(
            active,
            summary,
            args.out_dir,
            n_channels=int(args.individual_channel_count),
            n_windows=int(args.individual_windows_per_channel),
        )
    )

    config = {
        "samples": str(args.samples),
        "historical_windows": str(args.historical_windows),
        "padding_min": float(args.padding_min),
        "control_shift_days": list(args.control_shift_days),
        "min_intensity": args.min_intensity,
        "burstiness": args.burstiness,
        "phase_bin_width": float(args.phase_bin_width),
        "edge_window_min": float(args.edge_window_min),
        "edge_bin_min": float(args.edge_bin_min),
        "individual_channel_count": int(args.individual_channel_count),
        "individual_windows_per_channel": int(args.individual_windows_per_channel),
        "seed": int(args.seed),
    }
    report = write_report(args.out_dir, windows, active, control, paired, summary, paths, config)
    print(report)


if __name__ == "__main__":
    main()
