#!/usr/bin/env python
"""All-band source-window versus nearby-sideband Jupiter diagnostics.

This analysis keeps Io-A/B/C/D windows separate and asks a direct question:
is the signal inside each expected source-box interval brighter than nearby
pre/post time intervals, and does that behavior repeat across windows and
channels more than shifted-time controls?
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
from scripts.run_jupiter_per_window_event_survey import (  # noqa: E402
    add_daily_percentile,
    shifted_windows,
    source_box_windows,
)
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
DEFAULT_OUT = ROOT / "outputs/jupiter_source_window_sideband_structure_allbands_v1"

FREQ_COLOR = {
    0.45: "#222222",
    0.70: "#8dd3c7",
    0.90: "#b3a800",
    1.31: "#bebada",
    2.20: "#fb8072",
    3.93: "#1b9e77",
    4.70: "#d95f02",
    6.55: "#7570b3",
    9.18: "#e7298a",
}


def interval_values(
    time_ns: np.ndarray,
    values: np.ndarray,
    start_ns: int,
    end_ns: int,
) -> np.ndarray:
    lo = int(np.searchsorted(time_ns, int(start_ns), side="left"))
    hi = int(np.searchsorted(time_ns, int(end_ns), side="right"))
    if hi <= lo:
        return np.asarray([], dtype=float)
    vals = np.asarray(values[lo:hi], dtype=float)
    return vals[np.isfinite(vals)]


def interval_stats(values: np.ndarray, high_threshold: float) -> dict[str, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {
            "n": 0,
            "n_high": 0,
            "high_fraction": np.nan,
            "median": np.nan,
            "q90": np.nan,
            "max": np.nan,
        }
    high = vals >= float(high_threshold)
    return {
        "n": int(vals.size),
        "n_high": int(np.count_nonzero(high)),
        "high_fraction": float(np.mean(high)),
        "median": float(np.nanmedian(vals)),
        "q90": float(np.nanquantile(vals, 0.90)),
        "max": float(np.nanmax(vals)),
    }


def timestamp_ns(value: object) -> int:
    return int(pd.Timestamp(value).to_datetime64().astype("datetime64[ns]").astype(np.int64))


def source_sideband_metrics_for_role(
    samples: pd.DataFrame,
    windows: pd.DataFrame,
    high_threshold: float,
    sideband_gap_min: float,
    sideband_duration_scale: float,
    role: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if windows.empty:
        return pd.DataFrame()
    ordered_windows = windows.sort_values(["start_time", "source_box"]).reset_index(drop=True)
    win_records = ordered_windows.to_dict("records")
    starts = pd.to_datetime(ordered_windows["start_time"]).to_numpy(dtype="datetime64[ns]").astype("int64")
    ends = pd.to_datetime(ordered_windows["end_time"]).to_numpy(dtype="datetime64[ns]").astype("int64")
    gap_ns = int(float(sideband_gap_min) * 60.0 * 1e9)
    for (antenna, band, freq), grp in samples.groupby(["antenna", "frequency_band", "frequency_mhz"], sort=True):
        work = grp.sort_values("time")
        time_ns = work["time"].to_numpy(dtype="datetime64[ns]").astype("int64")
        daily = pd.to_numeric(work["daily_log10_residual"], errors="coerce").to_numpy(dtype=float)
        local = pd.to_numeric(work["local20min_log10_residual"], errors="coerce").to_numpy(dtype=float)
        for rec, start_ns, end_ns in zip(win_records, starts, ends):
            duration_ns = max(int(end_ns - start_ns), int(60e9))
            side_ns = int(round(duration_ns * float(sideband_duration_scale)))
            pre_start = int(start_ns - gap_ns - side_ns)
            pre_end = int(start_ns - gap_ns - 1)
            post_start = int(end_ns + gap_ns + 1)
            post_end = int(end_ns + gap_ns + side_ns)

            source_daily = interval_values(time_ns, daily, int(start_ns), int(end_ns))
            pre_daily = interval_values(time_ns, daily, pre_start, pre_end)
            post_daily = interval_values(time_ns, daily, post_start, post_end)
            side_daily = np.concatenate([pre_daily, post_daily]) if pre_daily.size or post_daily.size else np.asarray([], dtype=float)

            source_local = interval_values(time_ns, local, int(start_ns), int(end_ns))
            side_local = np.concatenate(
                [
                    interval_values(time_ns, local, pre_start, pre_end),
                    interval_values(time_ns, local, post_start, post_end),
                ]
            )

            s_daily = interval_stats(source_daily, high_threshold)
            b_daily = interval_stats(side_daily, high_threshold)
            s_local = interval_stats(source_local, high_threshold)
            b_local = interval_stats(side_local, high_threshold)
            pre_stats = interval_stats(pre_daily, high_threshold)
            post_stats = interval_stats(post_daily, high_threshold)
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
                    "source_n_samples": int(s_daily["n"]),
                    "sideband_n_samples": int(b_daily["n"]),
                    "pre_n_samples": int(pre_stats["n"]),
                    "post_n_samples": int(post_stats["n"]),
                    "source_n_high_samples": int(s_daily["n_high"]),
                    "sideband_n_high_samples": int(b_daily["n_high"]),
                    "source_high_fraction": float(s_daily["high_fraction"]),
                    "sideband_high_fraction": float(b_daily["high_fraction"]),
                    "source_minus_sideband_high_fraction": float(s_daily["high_fraction"] - b_daily["high_fraction"])
                    if np.isfinite(s_daily["high_fraction"]) and np.isfinite(b_daily["high_fraction"])
                    else np.nan,
                    "source_median_daily_residual": float(s_daily["median"]),
                    "sideband_median_daily_residual": float(b_daily["median"]),
                    "source_minus_sideband_median_daily_residual": float(s_daily["median"] - b_daily["median"])
                    if np.isfinite(s_daily["median"]) and np.isfinite(b_daily["median"])
                    else np.nan,
                    "source_q90_daily_residual": float(s_daily["q90"]),
                    "sideband_q90_daily_residual": float(b_daily["q90"]),
                    "source_minus_sideband_q90_daily_residual": float(s_daily["q90"] - b_daily["q90"])
                    if np.isfinite(s_daily["q90"]) and np.isfinite(b_daily["q90"])
                    else np.nan,
                    "source_max_daily_residual": float(s_daily["max"]),
                    "sideband_max_daily_residual": float(b_daily["max"]),
                    "source_median_local20min_residual": float(s_local["median"]),
                    "sideband_median_local20min_residual": float(b_local["median"]),
                    "source_minus_sideband_median_local20min_residual": float(s_local["median"] - b_local["median"])
                    if np.isfinite(s_local["median"]) and np.isfinite(b_local["median"])
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def add_shifted_control_baseline(real: pd.DataFrame, shifted: pd.DataFrame) -> pd.DataFrame:
    keys = ["window_id", "source_box", "antenna", "frequency_band", "frequency_mhz"]
    metric_cols = [
        "source_minus_sideband_median_daily_residual",
        "source_minus_sideband_high_fraction",
        "source_minus_sideband_q90_daily_residual",
        "source_minus_sideband_median_local20min_residual",
    ]
    agg = {}
    for col in metric_cols:
        agg[f"shift_median_{col}"] = (col, "median")
        agg[f"shift_q10_{col}"] = (col, lambda x: float(np.nanquantile(x, 0.10)))
        agg[f"shift_q90_{col}"] = (col, lambda x: float(np.nanquantile(x, 0.90)))
    ctrl = shifted.groupby(keys, sort=True).agg(**agg, n_shift_controls=("shift_days", "nunique")).reset_index()
    out = real.merge(ctrl, on=keys, how="left")
    for col in metric_cols:
        out[f"{col}_minus_shift_median"] = out[col] - out[f"shift_median_{col}"]
    return out


def channel_summary(with_controls: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ["source_box", "antenna", "antenna_label", "frequency_band", "frequency_mhz"]
    for keys, grp in with_controls.groupby(group_cols, sort=True):
        source_vs_side = grp["source_minus_sideband_median_daily_residual"]
        source_vs_shift = grp["source_minus_sideband_median_daily_residual_minus_shift_median"]
        high_vs_shift = grp["source_minus_sideband_high_fraction_minus_shift_median"]
        rows.append(
            {
                **dict(zip(group_cols, keys)),
                "n_windows": int(len(grp)),
                "n_windows_with_source_samples": int(grp["source_n_samples"].gt(0).sum()),
                "source_any_high_window_fraction": float(grp["source_n_high_samples"].gt(0).mean()),
                "sideband_any_high_window_fraction": float(grp["sideband_n_high_samples"].gt(0).mean()),
                "positive_source_minus_sideband_window_fraction": float(source_vs_side.gt(0).mean()),
                "positive_vs_shift_window_fraction": float(source_vs_shift.gt(0).mean()),
                "positive_high_fraction_vs_shift_window_fraction": float(high_vs_shift.gt(0).mean()),
                "median_source_minus_sideband_daily_residual": float(source_vs_side.median()),
                "median_source_minus_sideband_daily_residual_minus_shift": float(source_vs_shift.median()),
                "median_source_minus_sideband_high_fraction": float(grp["source_minus_sideband_high_fraction"].median()),
                "median_source_minus_sideband_high_fraction_minus_shift": float(high_vs_shift.median()),
                "total_source_high_samples": int(grp["source_n_high_samples"].sum()),
                "total_sideband_high_samples": int(grp["sideband_n_high_samples"].sum()),
            }
        )
    return pd.DataFrame(rows)


def shift_channel_distribution(shifted: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ["shift_days", "source_box", "antenna", "antenna_label", "frequency_band", "frequency_mhz"]
    for keys, grp in shifted.groupby(group_cols, sort=True):
        metric = grp["source_minus_sideband_median_daily_residual"]
        rows.append(
            {
                **dict(zip(group_cols, keys)),
                "shift_positive_source_minus_sideband_window_fraction": float(metric.gt(0).mean()),
                "shift_median_source_minus_sideband_daily_residual": float(metric.median()),
                "shift_median_source_minus_sideband_high_fraction": float(grp["source_minus_sideband_high_fraction"].median()),
                "n_shift_windows": int(len(grp)),
            }
        )
    return pd.DataFrame(rows)


def build_window_phase_bin_values(
    samples: pd.DataFrame,
    windows: pd.DataFrame,
    high_threshold: float,
    phase_bin_width: float,
    sideband_scale: float,
    role: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if windows.empty:
        return pd.DataFrame()
    win = windows.sort_values(["start_time", "source_box"]).reset_index(drop=True)
    records = win.to_dict("records")
    starts = pd.to_datetime(win["start_time"]).to_numpy(dtype="datetime64[ns]").astype("int64")
    ends = pd.to_datetime(win["end_time"]).to_numpy(dtype="datetime64[ns]").astype("int64")
    phase_lo = -float(sideband_scale)
    phase_hi = 1.0 + float(sideband_scale)
    bin_width = float(phase_bin_width)
    for (antenna, band, freq), grp in samples.groupby(["antenna", "frequency_band", "frequency_mhz"], sort=True):
        work = grp.sort_values("time")
        time_ns = work["time"].to_numpy(dtype="datetime64[ns]").astype("int64")
        daily = pd.to_numeric(work["daily_log10_residual"], errors="coerce").to_numpy(dtype=float)
        for rec, start_ns, end_ns in zip(records, starts, ends):
            dur_ns = max(int(end_ns - start_ns), int(60e9))
            lo_ns = int(start_ns + phase_lo * dur_ns)
            hi_ns = int(start_ns + phase_hi * dur_ns)
            lo = int(np.searchsorted(time_ns, lo_ns, side="left"))
            hi = int(np.searchsorted(time_ns, hi_ns, side="right"))
            if hi <= lo:
                continue
            phase = (time_ns[lo:hi].astype(float) - float(start_ns)) / float(dur_ns)
            values = daily[lo:hi]
            finite = np.isfinite(phase) & np.isfinite(values)
            if not finite.any():
                continue
            phase = phase[finite]
            values = values[finite]
            bins = np.floor(phase / bin_width) * bin_width + 0.5 * bin_width
            for phase_bin in np.unique(bins):
                mask = bins == phase_bin
                vals = values[mask]
                rows.append(
                    {
                        "role": role,
                        "window_id": rec["window_id"],
                        "source_box": rec["source_box"],
                        "shift_days": float(rec.get("shift_days", 0.0)),
                        "antenna": antenna,
                        "antenna_label": ANTENNA_LABEL.get(str(antenna), str(antenna)),
                        "frequency_band": int(band),
                        "frequency_mhz": float(freq),
                        "phase_bin": float(phase_bin),
                        "is_inside_source_window": bool(0.0 <= float(phase_bin) <= 1.0),
                        "n_samples": int(vals.size),
                        "any_high": bool(np.any(vals >= float(high_threshold))),
                        "high_fraction": float(np.mean(vals >= float(high_threshold))),
                        "median_daily_residual": float(np.nanmedian(vals)),
                    }
                )
    return pd.DataFrame(rows)


def summarize_phase_profiles(values: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if values.empty:
        return pd.DataFrame(), pd.DataFrame()
    keys = ["source_box", "antenna", "antenna_label", "frequency_band", "frequency_mhz", "phase_bin"]
    real = values[values["role"].eq("real")].copy()
    real_profile = (
        real.groupby(keys, sort=True)
        .agg(
            real_window_any_high_fraction=("any_high", "mean"),
            real_median_high_fraction=("high_fraction", "median"),
            real_median_daily_residual=("median_daily_residual", "median"),
            n_real_window_bins=("window_id", "nunique"),
        )
        .reset_index()
    )
    shifted = values[values["role"].eq("shifted_control")].copy()
    if shifted.empty:
        return real_profile, pd.DataFrame()
    per_shift = (
        shifted.groupby(["shift_days", *keys], sort=True)
        .agg(
            shift_window_any_high_fraction=("any_high", "mean"),
            shift_median_high_fraction=("high_fraction", "median"),
            shift_median_daily_residual=("median_daily_residual", "median"),
        )
        .reset_index()
    )
    control_profile = (
        per_shift.groupby(keys, sort=True)
        .agg(
            control_window_any_high_fraction_median=("shift_window_any_high_fraction", "median"),
            control_window_any_high_fraction_q10=("shift_window_any_high_fraction", lambda x: float(np.nanquantile(x, 0.10))),
            control_window_any_high_fraction_q90=("shift_window_any_high_fraction", lambda x: float(np.nanquantile(x, 0.90))),
            control_median_daily_residual=("shift_median_daily_residual", "median"),
            n_control_shifts=("shift_days", "nunique"),
        )
        .reset_index()
    )
    return real_profile, control_profile


def phase_inside_excess_summary(
    real_profile: pd.DataFrame,
    control_profile: pd.DataFrame,
    min_real_window_bins: int = 1,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ["source_box", "antenna", "antenna_label", "frequency_band", "frequency_mhz"]
    merged = real_profile.merge(
        control_profile,
        on=["source_box", "antenna", "antenna_label", "frequency_band", "frequency_mhz", "phase_bin"],
        how="left",
    )
    merged = merged[merged["n_real_window_bins"].ge(int(min_real_window_bins))].copy()
    merged["real_minus_control_window_any_high_fraction"] = (
        merged["real_window_any_high_fraction"] - merged["control_window_any_high_fraction_median"]
    )
    for keys, grp in merged.groupby(group_cols, sort=True):
        inside = grp[(grp["phase_bin"] >= 0.0) & (grp["phase_bin"] <= 1.0)]
        outside = grp[(grp["phase_bin"] < 0.0) | (grp["phase_bin"] > 1.0)]
        rows.append(
            {
                **dict(zip(group_cols, keys)),
                "inside_real_window_any_high_fraction_mean": float(inside["real_window_any_high_fraction"].mean()),
                "outside_real_window_any_high_fraction_mean": float(outside["real_window_any_high_fraction"].mean()),
                "inside_minus_outside_real_window_any_high_fraction": float(
                    inside["real_window_any_high_fraction"].mean() - outside["real_window_any_high_fraction"].mean()
                ),
                "inside_real_minus_control_window_any_high_fraction_mean": float(
                    inside["real_minus_control_window_any_high_fraction"].mean()
                ),
                "outside_real_minus_control_window_any_high_fraction_mean": float(
                    outside["real_minus_control_window_any_high_fraction"].mean()
                ),
                "inside_minus_outside_real_minus_control_window_any_high_fraction": float(
                    inside["real_minus_control_window_any_high_fraction"].mean()
                    - outside["real_minus_control_window_any_high_fraction"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def channel_labels(df: pd.DataFrame) -> list[str]:
    labels: list[str] = []
    for freq in sorted(df["frequency_mhz"].dropna().unique()):
        for antenna in ANTENNAS:
            label = f"{freq:.2f}\n{ANTENNA_LABEL[antenna]}"
            if ((np.isclose(df["frequency_mhz"], float(freq))) & df["antenna"].eq(antenna)).any():
                labels.append(label)
    return labels


def add_channel_label(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["channel"] = out["frequency_mhz"].map(lambda v: f"{v:.2f}") + "\n" + out["antenna"].map(ANTENNA_LABEL)
    return out


def plot_sideband_summary_heatmaps(summary: pd.DataFrame, out_dir: Path) -> list[Path]:
    work = add_channel_label(summary)
    channels = channel_labels(work)
    boxes = [box.name for box in SOURCE_BOXES]
    plots = [
        (
            "median_source_minus_sideband_daily_residual_minus_shift",
            "median source-sideband residual, minus shifted-control median (dex)",
            "coolwarm",
            True,
            "jupiter_sideband_summary_median_residual_excess_heatmap.png",
        ),
        (
            "positive_vs_shift_window_fraction",
            "fraction of windows with source-sideband residual above shifted-control median",
            "viridis",
            False,
            "jupiter_sideband_summary_positive_window_fraction_heatmap.png",
        ),
        (
            "median_source_minus_sideband_high_fraction_minus_shift",
            "median source-sideband factor-high fraction, minus shifted-control median",
            "coolwarm",
            True,
            "jupiter_sideband_summary_high_fraction_excess_heatmap.png",
        ),
    ]
    paths: list[Path] = []
    for value_col, title, cmap, centered, filename in plots:
        mat = work.pivot_table(index="source_box", columns="channel", values=value_col, aggfunc="mean")
        mat = mat.reindex(index=boxes, columns=channels)
        vals = mat.to_numpy(dtype=float)
        finite = vals[np.isfinite(vals)]
        fig, ax = plt.subplots(figsize=(14.0, 4.4))
        if centered:
            vmax = max(0.01, float(np.nanpercentile(np.abs(finite), 98))) if finite.size else 0.05
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        else:
            norm = Normalize(vmin=0.0, vmax=1.0)
        im = ax.imshow(vals, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
        ax.set_xticks(np.arange(len(channels)))
        ax.set_xticklabels(channels, fontsize=8)
        ax.set_yticks(np.arange(len(boxes)))
        ax.set_yticklabels(boxes)
        ax.set_title(title)
        ax.set_xlabel("frequency / antenna")
        ax.set_ylabel("source box")
        for y in range(vals.shape[0]):
            for x in range(vals.shape[1]):
                val = vals[y, x]
                if np.isfinite(val):
                    ax.text(x, y, f"{val:.2f}", ha="center", va="center", fontsize=6.5, color="white" if abs(val) > 0.15 else "black")
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        fig.tight_layout()
        path = out_dir / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def top_windows_by_direct_structure(with_controls: pd.DataFrame) -> pd.DataFrame:
    work = with_controls.copy()
    work["positive_vs_shift"] = work["source_minus_sideband_median_daily_residual_minus_shift_median"] > 0
    work["positive_highband_vs_shift"] = work["positive_vs_shift"] & work["frequency_mhz"].isin([3.93, 4.70, 6.55, 9.18])
    rows: list[dict[str, object]] = []
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
                "positive_all_channels_vs_shift": int(grp["positive_vs_shift"].sum()),
                "positive_3p93_to_9p18_channels_vs_shift": int(grp["positive_highband_vs_shift"].sum()),
                "median_channel_residual_excess_vs_shift": float(
                    grp["source_minus_sideband_median_daily_residual_minus_shift_median"].median()
                ),
                "max_channel_residual_excess_vs_shift": float(
                    grp["source_minus_sideband_median_daily_residual_minus_shift_median"].max()
                ),
                "total_source_high_samples": int(grp["source_n_high_samples"].sum()),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(
        [
            "positive_all_channels_vs_shift",
            "positive_3p93_to_9p18_channels_vs_shift",
            "median_channel_residual_excess_vs_shift",
            "total_source_high_samples",
        ],
        ascending=False,
    ).reset_index(drop=True)


def plot_per_window_direct_heatmap(with_controls: pd.DataFrame, out_dir: Path, top_n_per_box: int) -> Path:
    work = add_channel_label(with_controls)
    channels = channel_labels(work)
    score = top_windows_by_direct_structure(with_controls)
    selected_ids: list[str] = []
    for _, grp in score.groupby("source_box", sort=True):
        selected_ids.extend(grp.head(int(top_n_per_box))["window_id"].tolist())
    plot = work[work["window_id"].isin(selected_ids)].copy()
    boxes = [box.name for box in SOURCE_BOXES]
    fig, axes = plt.subplots(len(boxes), 1, figsize=(14.0, 13.0), sharex=True)
    axes = np.atleast_1d(axes)
    im = None
    for ax, box in zip(axes, boxes):
        sub = plot[plot["source_box"].eq(box)].copy()
        if sub.empty:
            ax.text(0.5, 0.5, "no selected windows", transform=ax.transAxes, ha="center", va="center")
            ax.set_ylabel(box)
            continue
        row_order = (
            sub.groupby(["window_id", "start_time"], sort=True)
            .agg(score=("source_minus_sideband_median_daily_residual_minus_shift_median", "median"))
            .sort_values("score", ascending=False)
            .index.get_level_values("window_id")
            .tolist()
        )
        mat = sub.pivot_table(
            index="window_id",
            columns="channel",
            values="source_minus_sideband_median_daily_residual_minus_shift_median",
            aggfunc="mean",
        ).reindex(index=row_order, columns=channels)
        vals = mat.to_numpy(dtype=float)
        finite = vals[np.isfinite(vals)]
        vmax = max(0.03, float(np.nanpercentile(np.abs(finite), 98))) if finite.size else 0.05
        im = ax.imshow(vals, aspect="auto", interpolation="nearest", cmap="coolwarm", norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax))
        starts = sub.drop_duplicates("window_id").set_index("window_id")["start_time"]
        ylabels = [f"{pd.Timestamp(starts.loc[wid]):%m-%d %H:%M}" for wid in row_order]
        ax.set_yticks(np.arange(len(row_order)))
        ax.set_yticklabels(ylabels, fontsize=7)
        ax.set_ylabel(box)
        ax.set_title(f"{box}: source interval minus nearby sidebands, after shifted-control baseline", loc="left", fontsize=10)
    axes[-1].set_xticks(np.arange(len(channels)))
    axes[-1].set_xticklabels(channels, fontsize=8)
    fig.suptitle("Per-window direct residual structure across all bands")
    fig.subplots_adjust(left=0.12, right=0.90, bottom=0.08, top=0.94, hspace=0.25)
    if im is not None:
        cax = fig.add_axes([0.92, 0.18, 0.018, 0.64])
        fig.colorbar(im, cax=cax, label="source-sideband residual excess vs shifted controls (dex)")
    path = out_dir / "jupiter_per_window_direct_sideband_residual_excess_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_phase_profiles(
    real_profile: pd.DataFrame,
    control_profile: pd.DataFrame,
    out_dir: Path,
    min_real_window_bins: int,
) -> Path:
    merged = real_profile.merge(
        control_profile,
        on=["source_box", "antenna", "antenna_label", "frequency_band", "frequency_mhz", "phase_bin"],
        how="left",
    )
    merged = merged[merged["n_real_window_bins"].ge(int(min_real_window_bins))].copy()
    boxes = [box.name for box in SOURCE_BOXES]
    fig, axes = plt.subplots(len(boxes), len(ANTENNAS), figsize=(14.0, 10.5), sharex=True, sharey=True)
    axes = np.asarray(axes)
    for row, box in enumerate(boxes):
        for col, antenna in enumerate(ANTENNAS):
            ax = axes[row, col]
            sub = merged[merged["source_box"].eq(box) & merged["antenna"].eq(antenna)].copy()
            for freq, grp in sub.groupby("frequency_mhz", sort=True):
                grp = grp.sort_values("phase_bin")
                color = FREQ_COLOR.get(round(float(freq), 2), None)
                y = grp["real_window_any_high_fraction"].to_numpy(dtype=float)
                ctrl = grp["control_window_any_high_fraction_median"].to_numpy(dtype=float)
                ax.plot(
                    grp["phase_bin"],
                    y,
                    color=color,
                    lw=1.25,
                    alpha=0.92,
                    label=f"{freq:.2f} MHz" if row == 0 and col == 0 else None,
                )
                if np.isfinite(ctrl).any():
                    ax.plot(grp["phase_bin"], ctrl, color=color, lw=0.7, alpha=0.22)
            ax.axvspan(0.0, 1.0, color="black", alpha=0.06)
            ax.axvline(0.0, color="0.35", lw=0.75)
            ax.axvline(1.0, color="0.35", lw=0.75)
            ax.axvline(0.5, color="0.35", lw=0.55, ls=":")
            ax.grid(True, color="0.91", lw=0.45)
            ax.set_title(f"{box} {ANTENNA_LABEL[antenna]}", loc="left", fontsize=9.5)
            if col == 0:
                ax.set_ylabel("fraction of windows with >=2x sample")
            if row == len(boxes) - 1:
                ax.set_xlabel("normalized window phase")
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=3, loc="upper left")
    fig.suptitle(f"Where bright samples occur relative to each Io/CML source window (phase bins with n>={int(min_real_window_bins)} windows)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = out_dir / "jupiter_window_phase_high_occurrence_profiles_allbands.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_phase_inside_excess_heatmap(inside_summary: pd.DataFrame, out_dir: Path) -> Path:
    work = add_channel_label(inside_summary)
    channels = channel_labels(work)
    boxes = [box.name for box in SOURCE_BOXES]
    value_col = "inside_minus_outside_real_minus_control_window_any_high_fraction"
    mat = work.pivot_table(index="source_box", columns="channel", values=value_col, aggfunc="mean").reindex(index=boxes, columns=channels)
    vals = mat.to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    vmax = max(0.01, float(np.nanpercentile(np.abs(finite), 98))) if finite.size else 0.05
    fig, ax = plt.subplots(figsize=(14.0, 4.4))
    im = ax.imshow(vals, aspect="auto", interpolation="nearest", cmap="coolwarm", norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax))
    ax.set_xticks(np.arange(len(channels)))
    ax.set_xticklabels(channels, fontsize=8)
    ax.set_yticks(np.arange(len(boxes)))
    ax.set_yticklabels(boxes)
    ax.set_title("Inside-window bright-sample excess over outside-window bins and shifted controls")
    ax.set_xlabel("frequency / antenna")
    ax.set_ylabel("source box")
    for y in range(vals.shape[0]):
        for x in range(vals.shape[1]):
            val = vals[y, x]
            if np.isfinite(val):
                ax.text(x, y, f"{val:.2f}", ha="center", va="center", fontsize=6.5, color="white" if abs(val) > 0.08 else "black")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="inside-outside excess")
    fig.tight_layout()
    path = out_dir / "jupiter_window_phase_inside_excess_heatmap_allbands.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_top_window_dynamic_spectra(samples: pd.DataFrame, top_windows: pd.DataFrame, out_dir: Path, n_windows: int) -> Path | None:
    if top_windows.empty:
        return None
    wins = top_windows.head(int(n_windows)).copy()
    freqs = sorted(samples["frequency_mhz"].dropna().unique())
    fig, axes = plt.subplots(len(wins), len(ANTENNAS), figsize=(13.5, max(2.1 * len(wins), 5.0)), sharex=False, sharey=True)
    axes = np.atleast_2d(axes)
    im = None
    for row, (_, win) in enumerate(wins.iterrows()):
        start = pd.Timestamp(win["start_time"])
        end = pd.Timestamp(win["end_time"])
        center = start + (end - start) / 2
        duration = max((end - start).total_seconds() / 60.0, 1.0)
        lo = start - pd.Timedelta(minutes=duration)
        hi = end + pd.Timedelta(minutes=duration)
        for col, antenna in enumerate(ANTENNAS):
            ax = axes[row, col]
            sub = samples[(samples["time"] >= lo) & (samples["time"] <= hi) & samples["antenna"].eq(antenna)].copy()
            if sub.empty:
                continue
            sub["minute_bin"] = np.floor((sub["time"] - center).dt.total_seconds() / 60.0 / 3.0) * 3.0 + 1.5
            mat = sub.pivot_table(index="frequency_mhz", columns="minute_bin", values="daily_log10_residual", aggfunc="median").reindex(index=freqs)
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
            ax.axvline(0.0, color="0.2", lw=0.75)
            ax.set_yticks(np.arange(len(freqs)))
            ax.set_yticklabels([f"{freq:.2f}" for freq in freqs])
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
    if im is not None:
        cax = fig.add_axes([0.91, 0.16, 0.018, 0.68])
        fig.colorbar(im, cax=cax, label="daily residual (dex)")
    fig.suptitle("Top source windows by all-band direct sideband structure")
    path = out_dir / "jupiter_top_direct_sideband_windows_dynamic_spectra_allbands.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report(
    out_dir: Path,
    windows: pd.DataFrame,
    summary: pd.DataFrame,
    top_windows: pd.DataFrame,
    inside_summary: pd.DataFrame,
    paths: list[Path],
    config: dict[str, object],
) -> Path:
    top_summary = summary.sort_values(
        ["positive_vs_shift_window_fraction", "median_source_minus_sideband_daily_residual_minus_shift"],
        ascending=False,
    ).head(24)
    inside_top = inside_summary.sort_values(
        "inside_minus_outside_real_minus_control_window_any_high_fraction", ascending=False
    ).head(20)
    lines = [
        "# Jupiter Source-Window Sideband Structure",
        "",
        "This all-band pass keeps Io-A/B/C/D windows separate. For every window/channel it compares the source interval with nearby pre/post sidebands of the same scaled duration, then compares that direct residual contrast with shifted-time controls.",
        "",
        "## Configuration",
        "",
        *[f"- `{k}`: `{v}`" for k, v in config.items() if k != "software_versions"],
        "",
        "## Coverage",
        "",
        f"- real source-box windows: `{len(windows)}`",
        *[f"- `{box.name}` windows: `{int((windows['source_box'] == box.name).sum())}`" for box in SOURCE_BOXES],
        "",
        "## Strongest Repeated Direct Sideband Structure",
        "",
        top_summary[
            [
                "source_box",
                "antenna_label",
                "frequency_mhz",
                "n_windows",
                "positive_vs_shift_window_fraction",
                "median_source_minus_sideband_daily_residual_minus_shift",
                "median_source_minus_sideband_high_fraction_minus_shift",
                "source_any_high_window_fraction",
                "sideband_any_high_window_fraction",
                "total_source_high_samples",
            ]
        ].to_string(index=False),
        "",
        "## Strongest Inside-Window Phase Localization",
        "",
        inside_top[
            [
                "source_box",
                "antenna_label",
                "frequency_mhz",
                "inside_real_window_any_high_fraction_mean",
                "outside_real_window_any_high_fraction_mean",
                "inside_minus_outside_real_window_any_high_fraction",
                "inside_minus_outside_real_minus_control_window_any_high_fraction",
            ]
        ].to_string(index=False),
        "",
        "## Top Individual Windows",
        "",
        top_windows.head(16).to_string(index=False),
        "",
        "## How To Read Direct Residuals",
        "",
        "The window center is just the midpoint of the CML/Io source-box interval. It is not a physical prediction that the burst must peak exactly at zero minutes. A Jovian DAM burst can occur anywhere inside the favorable source interval, and it can be offset by beaming, frequency drift, sparse sampling, and the fact that the source boxes are broad empirical regions.",
        "",
        "What we want in direct residual plots is structure that is preferentially inside the shaded source interval, or close to its entry/exit if sampling is sparse, and that repeats across windows/channels more than the nearby sidebands and shifted-time controls. A broad day-scale offset that is equally bright before, during, and after the window is weaker evidence.",
        "",
        "A useful pattern would be several separate windows showing positive residuals inside the shaded span, preferably in adjacent plausible bands and in the expected polarization sense. A 0.45-only pattern is included here for diagnosis but should not be treated as a clean Jovian detection by itself.",
        "",
        "## Files",
        "",
        *[f"- `{p}`" for p in paths],
    ]
    path = out_dir / "jupiter_source_window_sideband_structure_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-npy", type=Path, default=DEFAULT_CLEAN_NPY)
    parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--frequency-band", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument("--source-window-max-gap-min", type=float, default=18.0)
    parser.add_argument("--shift-days", type=float, nargs="+", default=[-7, -5, -3, -1, 1, 3, 5, 7])
    parser.add_argument("--local-window", type=str, default="20min")
    parser.add_argument("--local-min-periods", type=int, default=8)
    parser.add_argument("--high-factor", type=float, default=2.0)
    parser.add_argument("--sideband-gap-min", type=float, default=5.0)
    parser.add_argument("--sideband-duration-scale", type=float, default=1.0)
    parser.add_argument("--phase-bin-width", type=float, default=0.10)
    parser.add_argument("--phase-sideband-scale", type=float, default=1.0)
    parser.add_argument("--min-phase-window-bins", type=int, default=10)
    parser.add_argument("--top-n-per-box", type=int, default=30)
    parser.add_argument("--top-dynamic-windows", type=int, default=12)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    high_threshold = float(np.log10(float(args.high_factor)))
    config = {
        "clean_npy": str(args.clean_npy),
        "geometry": str(args.geometry),
        "frequency_bands": [int(v) for v in args.frequency_band],
        "frequencies_mhz": [FREQUENCY_MAP_MHZ.get(int(v), np.nan) for v in args.frequency_band],
        "source_window_max_gap_min": float(args.source_window_max_gap_min),
        "shift_days": [float(v) for v in args.shift_days],
        "local_window": str(args.local_window),
        "local_min_periods": int(args.local_min_periods),
        "high_factor": float(args.high_factor),
        "high_threshold_dex": high_threshold,
        "sideband_gap_min": float(args.sideband_gap_min),
        "sideband_duration_scale": float(args.sideband_duration_scale),
        "phase_bin_width": float(args.phase_bin_width),
        "phase_sideband_scale": float(args.phase_sideband_scale),
        "min_phase_window_bins": int(args.min_phase_window_bins),
        "software_versions": software_versions(),
    }
    write_json(out_dir / "run_config.json", config)

    print("Loading all-band samples...", flush=True)
    samples = read_clean_npy_subset(args.clean_npy, [int(v) for v in args.frequency_band])
    samples = add_local_residual(samples, window=str(args.local_window), min_periods=int(args.local_min_periods))
    samples = add_daily_percentile(samples)
    print(f"Loaded {len(samples)} samples; building source-box windows...", flush=True)
    geom = load_geometry(args.geometry, pad_deg=0.0)
    windows = source_box_windows(geom, max_gap_min=float(args.source_window_max_gap_min))
    windows["role"] = "real"
    windows["shift_days"] = 0.0
    windows.to_csv(out_dir / "jupiter_source_box_individual_windows.csv", index=False)
    shifted = shifted_windows(windows, [float(v) for v in args.shift_days], samples["time"].min(), samples["time"].max())
    shifted.to_csv(out_dir / "jupiter_source_box_shifted_windows.csv", index=False)

    print(f"Scoring {len(windows)} real windows and {len(shifted)} shifted windows...", flush=True)
    real_metrics = source_sideband_metrics_for_role(
        samples,
        windows,
        high_threshold=high_threshold,
        sideband_gap_min=float(args.sideband_gap_min),
        sideband_duration_scale=float(args.sideband_duration_scale),
        role="real",
    )
    shifted_metrics = source_sideband_metrics_for_role(
        samples,
        shifted,
        high_threshold=high_threshold,
        sideband_gap_min=float(args.sideband_gap_min),
        sideband_duration_scale=float(args.sideband_duration_scale),
        role="shifted_control",
    )
    real_metrics.to_csv(out_dir / "jupiter_real_window_sideband_metrics.csv", index=False)
    shifted_metrics.to_csv(out_dir / "jupiter_shifted_window_sideband_metrics.csv", index=False)
    with_controls = add_shifted_control_baseline(real_metrics, shifted_metrics)
    with_controls.to_csv(out_dir / "jupiter_real_window_sideband_metrics_with_shifted_controls.csv", index=False)
    summary = channel_summary(with_controls)
    summary.to_csv(out_dir / "jupiter_sideband_channel_summary.csv", index=False)
    shift_distribution = shift_channel_distribution(shifted_metrics)
    shift_distribution.to_csv(out_dir / "jupiter_sideband_shift_channel_distribution.csv", index=False)
    top_windows = top_windows_by_direct_structure(with_controls)
    top_windows.to_csv(out_dir / "jupiter_top_direct_sideband_windows.csv", index=False)

    print("Building normalized window-phase profiles...", flush=True)
    phase_real = build_window_phase_bin_values(
        samples,
        windows,
        high_threshold=high_threshold,
        phase_bin_width=float(args.phase_bin_width),
        sideband_scale=float(args.phase_sideband_scale),
        role="real",
    )
    phase_shifted = build_window_phase_bin_values(
        samples,
        shifted,
        high_threshold=high_threshold,
        phase_bin_width=float(args.phase_bin_width),
        sideband_scale=float(args.phase_sideband_scale),
        role="shifted_control",
    )
    phase_values = pd.concat([phase_real, phase_shifted], ignore_index=True)
    phase_values.to_csv(out_dir / "jupiter_window_phase_bin_values.csv", index=False)
    real_profile, control_profile = summarize_phase_profiles(phase_values)
    real_profile.to_csv(out_dir / "jupiter_window_phase_real_profile.csv", index=False)
    control_profile.to_csv(out_dir / "jupiter_window_phase_shifted_control_profile.csv", index=False)
    inside_summary = phase_inside_excess_summary(
        real_profile,
        control_profile,
        min_real_window_bins=int(args.min_phase_window_bins),
    )
    inside_summary.to_csv(out_dir / "jupiter_window_phase_inside_excess_summary.csv", index=False)

    print("Making plots...", flush=True)
    paths: list[Path] = [
        out_dir / "run_config.json",
        out_dir / "jupiter_source_box_individual_windows.csv",
        out_dir / "jupiter_source_box_shifted_windows.csv",
        out_dir / "jupiter_real_window_sideband_metrics.csv",
        out_dir / "jupiter_shifted_window_sideband_metrics.csv",
        out_dir / "jupiter_real_window_sideband_metrics_with_shifted_controls.csv",
        out_dir / "jupiter_sideband_channel_summary.csv",
        out_dir / "jupiter_sideband_shift_channel_distribution.csv",
        out_dir / "jupiter_top_direct_sideband_windows.csv",
        out_dir / "jupiter_window_phase_real_profile.csv",
        out_dir / "jupiter_window_phase_shifted_control_profile.csv",
        out_dir / "jupiter_window_phase_inside_excess_summary.csv",
    ]
    paths.extend(plot_sideband_summary_heatmaps(summary, out_dir))
    paths.append(plot_per_window_direct_heatmap(with_controls, out_dir, top_n_per_box=int(args.top_n_per_box)))
    paths.append(
        plot_phase_profiles(
            real_profile,
            control_profile,
            out_dir,
            min_real_window_bins=int(args.min_phase_window_bins),
        )
    )
    paths.append(plot_phase_inside_excess_heatmap(inside_summary, out_dir))
    maybe = plot_top_window_dynamic_spectra(samples, top_windows, out_dir, n_windows=int(args.top_dynamic_windows))
    if maybe is not None:
        paths.append(maybe)

    report = write_report(out_dir, windows, summary, top_windows, inside_summary, paths, config)
    print(f"Wrote {report}", flush=True)


if __name__ == "__main__":
    main()
