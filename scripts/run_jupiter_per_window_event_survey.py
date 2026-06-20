#!/usr/bin/env python
"""Per-window Jupiter Io-A/B/C/D event survey.

This pass keeps individual source-box windows separate.  It includes all
Ryle-Vonberg frequencies, including 0.45 MHz, and tests whether bright-event
occurrence repeats across multiple Io/CML source-box windows instead of
appearing only in an averaged profile.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
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
    add_local_residual,
    load_geometry,
    read_clean_npy_subset,
)


DEFAULT_CLEAN_NPY = ROOT / "outputs/control_survey_earth_sun_postnov1974_v2/01_ingest/cleaned_timeseries.npy"
DEFAULT_GEOMETRY = ROOT / "outputs/jupiter_phase_pattern_survey_v1/jupiter_spice_visibility_geometry_grid.csv"
DEFAULT_OUT = ROOT / "outputs/jupiter_per_window_event_survey_v1"

FREQ_COLOR = {
    0.45: "#222222",
    0.70: "#8dd3c7",
    0.90: "#ffffb3",
    1.31: "#bebada",
    2.20: "#fb8072",
    3.93: "#1b9e77",
    4.70: "#d95f02",
    6.55: "#7570b3",
    9.18: "#e7298a",
}


THRESHOLDS = {
    "daily_factor_1p5": ("daily_log10_residual", float(np.log10(1.5))),
    "daily_factor_2": ("daily_log10_residual", float(np.log10(2.0))),
    "daily_factor_3": ("daily_log10_residual", float(np.log10(3.0))),
    "daily_top10": ("daily_percentile", 0.90),
    "daily_top05": ("daily_percentile", 0.95),
    "local20_factor_2": ("local20min_log10_residual", float(np.log10(2.0))),
}


def source_box_windows(geom: pd.DataFrame, max_gap_min: float) -> pd.DataFrame:
    rows = []
    for box in SOURCE_BOXES:
        col = f"source_{box.name.lower().replace('-', '_')}"
        selected = geom[geom["jupiter_visible_by_moon"].astype(bool) & geom[col].astype(bool)].sort_values("time").copy()
        if selected.empty:
            continue
        group = selected["time"].diff().dt.total_seconds().div(60).gt(float(max_gap_min)).fillna(True).cumsum()
        for idx, grp in selected.groupby(group, sort=True):
            start = grp["time"].min()
            end = grp["time"].max()
            rows.append(
                {
                    "window_id": f"{box.name.replace('-', '')}_{int(idx):04d}",
                    "source_box": box.name,
                    "start_time": start,
                    "end_time": end,
                    "duration_min": (end - start).total_seconds() / 60.0,
                    "median_cml_deg": float(grp["jupiter_cml_spice_deg"].median()),
                    "median_io_phase_deg": float(grp["io_phase_spice_deg"].median()),
                    "max_maser_zarka_io_score": float(grp["maser_zarka_io_score"].max()),
                    "median_maser_zarka_io_score": float(grp["maser_zarka_io_score"].median()),
                    "n_geometry_points": int(len(grp)),
                }
            )
    return pd.DataFrame(rows).sort_values(["start_time", "source_box"]).reset_index(drop=True)


def shifted_windows(windows: pd.DataFrame, shift_days: list[float], sample_start: pd.Timestamp, sample_end: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for shift in shift_days:
        delta = pd.Timedelta(days=float(shift))
        shifted = windows.copy()
        shifted["role"] = "shifted_control"
        shifted["shift_days"] = float(shift)
        shifted["start_time"] = pd.to_datetime(shifted["start_time"]) + delta
        shifted["end_time"] = pd.to_datetime(shifted["end_time"]) + delta
        shifted = shifted[(shifted["end_time"] >= sample_start) & (shifted["start_time"] <= sample_end)].copy()
        rows.append(shifted)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def add_daily_percentile(samples: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for _, grp in samples.groupby(["date", "antenna", "frequency_band"], sort=False):
        work = grp.copy()
        rank = work["log10_power"].rank(method="average", pct=True)
        work["daily_percentile"] = rank.to_numpy(dtype=float)
        pieces.append(work)
    return pd.concat(pieces, ignore_index=True).sort_values("time").reset_index(drop=True)


def count_events(time_ns: np.ndarray, high_mask: np.ndarray, max_gap_s: float, min_high_samples: int) -> int:
    idx = np.flatnonzero(np.asarray(high_mask, dtype=bool))
    if idx.size < int(min_high_samples):
        return 0
    if idx.size == 1:
        return int(min_high_samples <= 1)
    t = np.asarray(time_ns, dtype=np.int64)[idx]
    breaks = np.flatnonzero(np.diff(t) > int(float(max_gap_s) * 1e9)) + 1
    starts = np.r_[0, breaks]
    ends = np.r_[breaks, len(idx)]
    return int(np.count_nonzero((ends - starts) >= int(min_high_samples)))


def window_metrics_for_role(
    samples: pd.DataFrame,
    windows: pd.DataFrame,
    thresholds: dict[str, tuple[str, float]],
    role: str,
    max_event_gap_s: float,
    min_high_samples_per_event: int,
) -> pd.DataFrame:
    rows = []
    if windows.empty:
        return pd.DataFrame()
    win = windows.sort_values("start_time").reset_index(drop=True)
    starts = pd.to_datetime(win["start_time"]).to_numpy(dtype="datetime64[ns]").astype("int64")
    ends = pd.to_datetime(win["end_time"]).to_numpy(dtype="datetime64[ns]").astype("int64")
    window_records = win.to_dict("records")
    for (antenna, band, freq), grp in samples.groupby(["antenna", "frequency_band", "frequency_mhz"], sort=True):
        work = grp.sort_values("time")
        time_ns = work["time"].to_numpy(dtype="datetime64[ns]").astype("int64")
        for i, (start_ns, end_ns) in enumerate(zip(starts, ends)):
            lo = int(np.searchsorted(time_ns, start_ns, side="left"))
            hi = int(np.searchsorted(time_ns, end_ns, side="right"))
            rec = window_records[i]
            duration_hr = max(float(rec["duration_min"]) / 60.0, 1e-9)
            if hi <= lo:
                for threshold_name in thresholds:
                    rows.append(empty_metric_row(rec, role, antenna, band, freq, threshold_name, duration_hr))
                continue
            seg = work.iloc[lo:hi]
            seg_time_ns = time_ns[lo:hi]
            for threshold_name, (value_col, threshold_value) in thresholds.items():
                values = pd.to_numeric(seg[value_col], errors="coerce").to_numpy(dtype=float)
                high = np.isfinite(values) & (values >= float(threshold_value))
                n_high = int(np.count_nonzero(high))
                n_events = count_events(seg_time_ns, high, max_gap_s=max_event_gap_s, min_high_samples=int(min_high_samples_per_event))
                daily_resid = pd.to_numeric(seg["daily_log10_residual"], errors="coerce")
                local_resid = pd.to_numeric(seg["local20min_log10_residual"], errors="coerce")
                rows.append(
                    {
                        "role": role,
                        "window_id": rec["window_id"],
                        "source_box": rec["source_box"],
                        "shift_days": float(rec.get("shift_days", 0.0)),
                        "start_time": rec["start_time"],
                        "end_time": rec["end_time"],
                        "duration_min": float(rec["duration_min"]),
                        "median_cml_deg": float(rec["median_cml_deg"]),
                        "median_io_phase_deg": float(rec["median_io_phase_deg"]),
                        "max_maser_zarka_io_score": float(rec["max_maser_zarka_io_score"]),
                        "antenna": antenna,
                        "antenna_label": ANTENNA_LABEL.get(str(antenna), str(antenna)),
                        "frequency_band": int(band),
                        "frequency_mhz": float(freq),
                        "threshold_name": threshold_name,
                        "threshold_value": float(threshold_value),
                        "threshold_value_col": value_col,
                        "n_samples": int(len(seg)),
                        "n_high_samples": n_high,
                        "n_events": n_events,
                        "event_rate_per_hr": float(n_events / duration_hr),
                        "high_sample_fraction": float(n_high / len(seg)) if len(seg) else np.nan,
                        "median_daily_residual": float(daily_resid.median()) if len(seg) else np.nan,
                        "max_daily_residual": float(daily_resid.max()) if len(seg) else np.nan,
                        "median_local20min_residual": float(local_resid.median()) if len(seg) else np.nan,
                        "max_local20min_residual": float(local_resid.max()) if len(seg) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def empty_metric_row(
    rec: dict[str, object],
    role: str,
    antenna: str,
    band: int,
    freq: float,
    threshold_name: str,
    duration_hr: float,
) -> dict[str, object]:
    return {
        "role": role,
        "window_id": rec["window_id"],
        "source_box": rec["source_box"],
        "shift_days": float(rec.get("shift_days", 0.0)),
        "start_time": rec["start_time"],
        "end_time": rec["end_time"],
        "duration_min": float(rec["duration_min"]),
        "median_cml_deg": float(rec["median_cml_deg"]),
        "median_io_phase_deg": float(rec["median_io_phase_deg"]),
        "max_maser_zarka_io_score": float(rec["max_maser_zarka_io_score"]),
        "antenna": antenna,
        "antenna_label": ANTENNA_LABEL.get(str(antenna), str(antenna)),
        "frequency_band": int(band),
        "frequency_mhz": float(freq),
        "threshold_name": threshold_name,
        "threshold_value": np.nan,
        "threshold_value_col": "",
        "n_samples": 0,
        "n_high_samples": 0,
        "n_events": 0,
        "event_rate_per_hr": float(0.0 / duration_hr),
        "high_sample_fraction": np.nan,
        "median_daily_residual": np.nan,
        "max_daily_residual": np.nan,
        "median_local20min_residual": np.nan,
        "max_local20min_residual": np.nan,
    }


def add_shifted_controls(real: pd.DataFrame, controls: pd.DataFrame) -> pd.DataFrame:
    keys = ["window_id", "source_box", "antenna", "frequency_band", "frequency_mhz", "threshold_name"]
    ctrl = (
        controls.groupby(keys, sort=True)
        .agg(
            control_event_rate_per_hr_median=("event_rate_per_hr", "median"),
            control_event_rate_per_hr_q10=("event_rate_per_hr", lambda x: float(np.nanquantile(x, 0.10))),
            control_event_rate_per_hr_q90=("event_rate_per_hr", lambda x: float(np.nanquantile(x, 0.90))),
            control_high_sample_fraction_median=("high_sample_fraction", "median"),
            n_control_shifts=("shift_days", "nunique"),
        )
        .reset_index()
    )
    out = real.merge(ctrl, on=keys, how="left")
    out["event_rate_excess_per_hr"] = out["event_rate_per_hr"] - out["control_event_rate_per_hr_median"]
    out["high_sample_fraction_excess"] = out["high_sample_fraction"] - out["control_high_sample_fraction_median"]
    return out


def channel_threshold_summary(with_controls: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["source_box", "antenna", "antenna_label", "frequency_band", "frequency_mhz", "threshold_name"]
    for keys, grp in with_controls.groupby(group_cols, sort=True):
        positive = grp["event_rate_excess_per_hr"].gt(0)
        rows.append(
            {
                **dict(zip(group_cols, keys)),
                "n_windows": int(len(grp)),
                "n_windows_with_events": int(grp["n_events"].gt(0).sum()),
                "event_window_fraction": float(grp["n_events"].gt(0).mean()),
                "n_windows_positive_excess": int(positive.sum()),
                "positive_excess_window_fraction": float(positive.mean()),
                "median_event_rate_excess_per_hr": float(grp["event_rate_excess_per_hr"].median()),
                "q75_event_rate_excess_per_hr": float(np.nanquantile(grp["event_rate_excess_per_hr"], 0.75)),
                "total_real_events": int(grp["n_events"].sum()),
                "total_real_samples": int(grp["n_samples"].sum()),
            }
        )
    return pd.DataFrame(rows)


def look_elsewhere_summary(summary: pd.DataFrame, controls: pd.DataFrame) -> pd.DataFrame:
    """Compare best real channel/source-box results with best shifted controls."""
    rows = []
    ctrl_group_cols = ["shift_days", "source_box", "antenna", "frequency_band", "frequency_mhz", "threshold_name"]
    ctrl = (
        controls.groupby(ctrl_group_cols, sort=True)
        .agg(control_positive_window_fraction=("n_events", lambda x: float(np.mean(np.asarray(x) > 0))))
        .reset_index()
    )
    for threshold_name, real_grp in summary.groupby("threshold_name", sort=True):
        real_best = real_grp.sort_values("event_window_fraction", ascending=False).iloc[0]
        shift_best = (
            ctrl[ctrl["threshold_name"].eq(threshold_name)]
            .groupby("shift_days", sort=True)["control_positive_window_fraction"]
            .max()
            .to_numpy(dtype=float)
        )
        real_val = float(real_best["event_window_fraction"])
        percentile = float((np.count_nonzero(shift_best <= real_val) + 1.0) / (len(shift_best) + 1.0)) if len(shift_best) else np.nan
        rows.append(
            {
                "threshold_name": threshold_name,
                "best_real_source_box": real_best["source_box"],
                "best_real_antenna_label": real_best["antenna_label"],
                "best_real_frequency_mhz": float(real_best["frequency_mhz"]),
                "best_real_event_window_fraction": real_val,
                "best_shift_control_positive_window_fraction_median": float(np.nanmedian(shift_best)) if len(shift_best) else np.nan,
                "best_shift_control_positive_window_fraction_max": float(np.nanmax(shift_best)) if len(shift_best) else np.nan,
                "best_real_vs_shift_best_percentile": percentile,
            }
        )
    return pd.DataFrame(rows)


def plot_window_channel_heatmap(with_controls: pd.DataFrame, threshold_name: str, out_dir: Path, top_n_per_box: int) -> Path:
    work = with_controls[with_controls["threshold_name"].eq(threshold_name)].copy()
    score = (
        work.groupby(["window_id", "source_box", "start_time", "end_time"], sort=True)
        .agg(
            positive_channels=("event_rate_excess_per_hr", lambda x: int(np.count_nonzero(np.asarray(x) > 0))),
            max_excess=("event_rate_excess_per_hr", "max"),
            total_events=("n_events", "sum"),
        )
        .reset_index()
    )
    selected_ids = []
    for _, grp in score.sort_values(["positive_channels", "max_excess", "total_events"], ascending=False).groupby("source_box", sort=True):
        selected_ids.extend(grp.head(int(top_n_per_box))["window_id"].tolist())
    plot = work[work["window_id"].isin(selected_ids)].copy()
    plot["channel"] = plot["frequency_mhz"].map(lambda x: f"{x:.2f}") + "\n" + plot["antenna"].map(ANTENNA_LABEL)
    channel_order = []
    for freq in sorted(plot["frequency_mhz"].dropna().unique()):
        for ant in ANTENNAS:
            label = f"{freq:.2f}\n{ANTENNA_LABEL[ant]}"
            if label in set(plot["channel"]):
                channel_order.append(label)
    fig, axes = plt.subplots(4, 1, figsize=(14.0, 13.0), sharex=True)
    for ax, box in zip(axes, [b.name for b in SOURCE_BOXES]):
        sub = plot[plot["source_box"].eq(box)].copy()
        if sub.empty:
            ax.text(0.5, 0.5, "no windows", transform=ax.transAxes, ha="center", va="center")
            ax.set_title(box, loc="left")
            continue
        row_order = (
            sub.groupby(["window_id", "start_time"], sort=True)["event_rate_excess_per_hr"]
            .max()
            .sort_values(ascending=False)
            .index.get_level_values("window_id")
            .tolist()
        )
        mat = sub.pivot_table(index="window_id", columns="channel", values="event_rate_excess_per_hr", aggfunc="mean")
        mat = mat.reindex(index=row_order, columns=channel_order)
        vals = mat.to_numpy(dtype=float)
        finite = vals[np.isfinite(vals)]
        vmax = max(1.0, float(np.nanpercentile(np.abs(finite), 98))) if finite.size else 1.0
        im = ax.imshow(vals, aspect="auto", interpolation="nearest", cmap="coolwarm", norm=TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax))
        labels = []
        starts = sub.drop_duplicates("window_id").set_index("window_id")["start_time"]
        for wid in row_order:
            labels.append(f"{pd.Timestamp(starts.loc[wid]):%m-%d %H:%M}")
        ax.set_yticks(np.arange(len(row_order)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_ylabel(box)
        ax.set_title(f"{box}: each row is one source-box window", loc="left", fontsize=10)
    axes[-1].set_xticks(np.arange(len(channel_order)))
    axes[-1].set_xticklabels(channel_order, rotation=0, fontsize=8)
    fig.suptitle(f"Per-window event-rate excess vs shifted controls ({threshold_name})")
    fig.subplots_adjust(left=0.12, right=0.90, bottom=0.08, top=0.94, hspace=0.25)
    cax = fig.add_axes([0.92, 0.18, 0.018, 0.64])
    fig.colorbar(im, cax=cax, label="real event rate - shifted-control median (events/hr)")
    path = out_dir / f"jupiter_per_window_event_rate_excess_heatmap_{threshold_name}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_threshold_sweep(summary: pd.DataFrame, out_dir: Path) -> Path:
    high = summary[summary["frequency_mhz"].isin([0.45, 3.93, 4.70, 6.55, 9.18])].copy()
    high["channel"] = high["source_box"] + "\n" + high["frequency_mhz"].map(lambda x: f"{x:.2f}") + " " + high["antenna_label"]
    thresholds = list(THRESHOLDS.keys())
    fig, axes = plt.subplots(len(thresholds), 1, figsize=(15.0, 2.15 * len(thresholds)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, threshold_name in zip(axes, thresholds):
        sub = high[high["threshold_name"].eq(threshold_name)].sort_values(
            ["source_box", "frequency_mhz", "antenna_label"]
        )
        x = np.arange(len(sub))
        ax.scatter(x, sub["positive_excess_window_fraction"], s=22, color="#386cb0")
        ax.axhline(0.5, color="0.55", lw=0.8, ls=":")
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel(threshold_name)
        ax.grid(True, axis="y", color="0.9", lw=0.45)
    axes[-1].set_xticks(np.arange(len(sub)))
    axes[-1].set_xticklabels(sub["channel"], rotation=90, fontsize=7)
    fig.suptitle("Threshold sweep: fraction of windows with positive event-rate excess")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = out_dir / "jupiter_threshold_sweep_positive_excess_window_fraction.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def top_candidate_windows(with_controls: pd.DataFrame, threshold_name: str) -> pd.DataFrame:
    work = with_controls[with_controls["threshold_name"].eq(threshold_name)].copy()
    work["is_highband"] = work["frequency_mhz"].isin([3.93, 4.70, 6.55, 9.18])
    rows = []
    for (window_id, source_box), grp in work.groupby(["window_id", "source_box"], sort=True):
        rows.append(
            {
                "window_id": window_id,
                "source_box": source_box,
                "start_time": grp["start_time"].iloc[0],
                "end_time": grp["end_time"].iloc[0],
                "duration_min": float(grp["duration_min"].iloc[0]),
                "median_cml_deg": float(grp["median_cml_deg"].iloc[0]),
                "median_io_phase_deg": float(grp["median_io_phase_deg"].iloc[0]),
                "positive_all_channels": int(grp["event_rate_excess_per_hr"].gt(0).sum()),
                "positive_highband_channels": int(grp.loc[grp["is_highband"], "event_rate_excess_per_hr"].gt(0).sum()),
                "total_events_all_channels": int(grp["n_events"].sum()),
                "total_events_highbands": int(grp.loc[grp["is_highband"], "n_events"].sum()),
                "max_event_rate_excess_per_hr": float(grp["event_rate_excess_per_hr"].max()),
                "max_highband_event_rate_excess_per_hr": float(grp.loc[grp["is_highband"], "event_rate_excess_per_hr"].max()),
                "has_0p45_events": bool(grp.loc[np.isclose(grp["frequency_mhz"], 0.45), "n_events"].sum() > 0),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["positive_highband_channels", "positive_all_channels", "max_highband_event_rate_excess_per_hr", "total_events_highbands"],
        ascending=False,
    ).reset_index(drop=True)


def plot_top_window_dynamic_spectra(samples: pd.DataFrame, top_windows: pd.DataFrame, out_dir: Path, n_windows: int) -> Path | None:
    if top_windows.empty:
        return None
    wins = top_windows.head(int(n_windows)).copy()
    freqs = sorted(samples["frequency_mhz"].dropna().unique())
    fig, axes = plt.subplots(len(wins), 2, figsize=(13.5, max(2.15 * len(wins), 5.0)), sharex=False, sharey=True)
    axes = np.atleast_2d(axes)
    for row, (_, win) in enumerate(wins.iterrows()):
        start = pd.Timestamp(win["start_time"])
        end = pd.Timestamp(win["end_time"])
        center = start + (end - start) / 2
        lo = start - pd.Timedelta(minutes=35)
        hi = end + pd.Timedelta(minutes=35)
        for col, antenna in enumerate(ANTENNAS):
            ax = axes[row, col]
            sub = samples[(samples["time"] >= lo) & (samples["time"] <= hi) & samples["antenna"].eq(antenna)].copy()
            if sub.empty:
                continue
            sub["minute_bin"] = np.floor((sub["time"] - center).dt.total_seconds() / 60.0 / 3.0) * 3.0 + 1.5
            mat = sub.pivot_table(index="frequency_mhz", columns="minute_bin", values="daily_log10_residual", aggfunc="median")
            mat = mat.reindex(index=freqs)
            vals = mat.to_numpy(dtype=float)
            finite = vals[np.isfinite(vals)]
            lim = max(0.08, float(np.nanpercentile(np.abs(finite), 98))) if finite.size else 0.1
            im = ax.imshow(
                vals,
                origin="lower",
                aspect="auto",
                extent=[float(mat.columns.min()), float(mat.columns.max()), -0.5, len(freqs) - 0.5],
                cmap="coolwarm",
                norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim),
            )
            ax.axvspan((start - center).total_seconds() / 60.0, (end - center).total_seconds() / 60.0, color="black", alpha=0.08)
            ax.axvline(0, color="0.2", lw=0.8)
            ax.set_yticks(np.arange(len(freqs)))
            ax.set_yticklabels([f"{f:.2f}" for f in freqs])
            ax.set_title(
                f"{ANTENNA_LABEL[antenna]} {win['source_box']} {center:%Y-%m-%d %H:%M} "
                f"Io={float(win['median_io_phase_deg']):.0f} CML={float(win['median_cml_deg']):.0f}",
                loc="left",
                fontsize=8.5,
            )
            if col == 0:
                ax.set_ylabel("MHz")
    axes[-1, 0].set_xlabel("minutes from window center")
    axes[-1, 1].set_xlabel("minutes from window center")
    fig.subplots_adjust(right=0.89, hspace=0.45, wspace=0.08, top=0.94, bottom=0.06)
    cax = fig.add_axes([0.91, 0.16, 0.018, 0.68])
    fig.colorbar(im, cax=cax, label="daily residual (dex)")
    fig.suptitle("Top repeated source-box windows: all-frequency direct residuals")
    path = out_dir / "jupiter_top_repeated_source_box_windows_dynamic_spectra.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report(
    out_dir: Path,
    windows: pd.DataFrame,
    summary: pd.DataFrame,
    look_elsewhere: pd.DataFrame,
    top_windows: pd.DataFrame,
    paths: list[Path],
    config: dict[str, object],
) -> Path:
    lines = [
        "# Jupiter Per-Window Event Survey",
        "",
        "This run keeps individual Io-A/B/C/D source-box windows separate and includes all Ryle-Vonberg bands, including 0.45 MHz. It tests repeated event occurrence across many windows rather than averaging source-box windows together.",
        "",
        "## Configuration",
        "",
        *[f"- `{k}`: `{v}`" for k, v in config.items() if k != "software_versions"],
        "",
        "## Coverage",
        "",
        f"- source-box windows: `{len(windows)}`",
        *[f"- `{box}` windows: `{int((windows['source_box'] == box).sum())}`" for box in [b.name for b in SOURCE_BOXES]],
        "",
        "## Look-Elsewhere Screen",
        "",
        look_elsewhere.to_string(index=False),
        "",
        "## Strongest Channel/Source-Box Repetition",
        "",
        summary.sort_values(
            ["positive_excess_window_fraction", "total_real_events"], ascending=False
        )
        .head(20)[
            [
                "source_box",
                "antenna_label",
                "frequency_mhz",
                "threshold_name",
                "n_windows",
                "n_windows_with_events",
                "event_window_fraction",
                "positive_excess_window_fraction",
                "median_event_rate_excess_per_hr",
                "total_real_events",
            ]
        ]
        .to_string(index=False),
        "",
        "## Top Individual Windows",
        "",
        top_windows.head(16).to_string(index=False),
        "",
        "## How To Read These Products",
        "",
        "- Each heatmap row is one real Io-A/B/C/D source-box window; rows are not averaged.",
        "- Positive event-rate excess means that window/channel has more bright events per hour than the median shifted-time control for the same window duration.",
        "- Threshold sweeps check whether the same source-box/frequency behavior survives several brightness definitions.",
        "- 0.45 MHz is included for context, but the strongest Jupiter interpretation should still make sense in neighboring/plausible bands and raw-window strips.",
        "",
        "## Files",
        "",
        *[f"- `{p}`" for p in paths],
    ]
    path = out_dir / "jupiter_per_window_event_survey_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-npy", type=Path, default=DEFAULT_CLEAN_NPY)
    parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--frequency-band", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument("--geometry-tolerance-s", type=float, default=360.0)
    parser.add_argument("--source-window-max-gap-min", type=float, default=18.0)
    parser.add_argument("--shift-days", type=float, nargs="+", default=[-7, -5, -3, -1, 1, 3, 5, 7])
    parser.add_argument("--local-window", type=str, default="20min")
    parser.add_argument("--local-min-periods", type=int, default=8)
    parser.add_argument("--max-event-gap-s", type=float, default=180.0)
    parser.add_argument("--min-high-samples-per-event", type=int, default=1)
    parser.add_argument("--heatmap-top-n-per-box", type=int, default=28)
    parser.add_argument("--top-raw-windows", type=int, default=10)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    config = {
        "clean_npy": str(args.clean_npy),
        "geometry": str(args.geometry),
        "frequency_bands": [int(v) for v in args.frequency_band],
        "frequencies_mhz": [FREQUENCY_MAP_MHZ.get(int(v), np.nan) for v in args.frequency_band],
        "geometry_tolerance_s": float(args.geometry_tolerance_s),
        "source_window_max_gap_min": float(args.source_window_max_gap_min),
        "shift_days": [float(v) for v in args.shift_days],
        "local_window": str(args.local_window),
        "local_min_periods": int(args.local_min_periods),
        "max_event_gap_s": float(args.max_event_gap_s),
        "min_high_samples_per_event": int(args.min_high_samples_per_event),
        "thresholds": {k: {"column": v[0], "value": v[1]} for k, v in THRESHOLDS.items()},
        "software_versions": software_versions(),
    }
    write_json(out_dir / "run_config.json", config)

    print("Loading all-band samples...", flush=True)
    samples = read_clean_npy_subset(args.clean_npy, [int(v) for v in args.frequency_band])
    print(f"Loaded {len(samples)} samples; adding local residuals and daily percentiles...", flush=True)
    samples = add_local_residual(samples, window=str(args.local_window), min_periods=int(args.local_min_periods))
    samples = add_daily_percentile(samples)
    print("Loading source-box geometry...", flush=True)
    geom = load_geometry(args.geometry, pad_deg=0.0)
    windows = source_box_windows(geom, max_gap_min=float(args.source_window_max_gap_min))
    windows["role"] = "real"
    windows["shift_days"] = 0.0
    windows.to_csv(out_dir / "jupiter_source_box_individual_windows.csv", index=False)
    shifted = shifted_windows(windows, [float(v) for v in args.shift_days], samples["time"].min(), samples["time"].max())
    shifted.to_csv(out_dir / "jupiter_source_box_shifted_control_windows.csv", index=False)
    print(f"Scoring {len(windows)} real windows and {len(shifted)} shifted controls...", flush=True)
    real = window_metrics_for_role(
        samples,
        windows,
        THRESHOLDS,
        role="real",
        max_event_gap_s=float(args.max_event_gap_s),
        min_high_samples_per_event=int(args.min_high_samples_per_event),
    )
    controls = window_metrics_for_role(
        samples,
        shifted,
        THRESHOLDS,
        role="shifted_control",
        max_event_gap_s=float(args.max_event_gap_s),
        min_high_samples_per_event=int(args.min_high_samples_per_event),
    )
    real.to_csv(out_dir / "jupiter_real_window_channel_event_metrics.csv", index=False)
    controls.to_csv(out_dir / "jupiter_shifted_window_channel_event_metrics.csv", index=False)
    with_controls = add_shifted_controls(real, controls)
    with_controls.to_csv(out_dir / "jupiter_real_window_channel_event_metrics_with_shifted_controls.csv", index=False)
    summary = channel_threshold_summary(with_controls)
    summary.to_csv(out_dir / "jupiter_channel_threshold_sourcebox_summary.csv", index=False)
    look_elsewhere = look_elsewhere_summary(summary, controls)
    look_elsewhere.to_csv(out_dir / "jupiter_look_elsewhere_threshold_summary.csv", index=False)
    top_windows = top_candidate_windows(with_controls, threshold_name="daily_factor_2")
    top_windows.to_csv(out_dir / "jupiter_top_candidate_source_box_windows.csv", index=False)

    print("Making plots...", flush=True)
    paths: list[Path] = [
        out_dir / "run_config.json",
        out_dir / "jupiter_source_box_individual_windows.csv",
        out_dir / "jupiter_source_box_shifted_control_windows.csv",
        out_dir / "jupiter_real_window_channel_event_metrics.csv",
        out_dir / "jupiter_shifted_window_channel_event_metrics.csv",
        out_dir / "jupiter_real_window_channel_event_metrics_with_shifted_controls.csv",
        out_dir / "jupiter_channel_threshold_sourcebox_summary.csv",
        out_dir / "jupiter_look_elsewhere_threshold_summary.csv",
        out_dir / "jupiter_top_candidate_source_box_windows.csv",
    ]
    for threshold_name in ["daily_factor_2", "daily_factor_3", "local20_factor_2"]:
        paths.append(plot_window_channel_heatmap(with_controls, threshold_name, out_dir, top_n_per_box=int(args.heatmap_top_n_per_box)))
    paths.append(plot_threshold_sweep(summary, out_dir))
    maybe = plot_top_window_dynamic_spectra(samples, top_windows, out_dir, n_windows=int(args.top_raw_windows))
    if maybe is not None:
        paths.append(maybe)
    report = write_report(out_dir, windows, summary, look_elsewhere, top_windows, paths, config)
    print(f"Wrote {report}", flush=True)


if __name__ == "__main__":
    main()
