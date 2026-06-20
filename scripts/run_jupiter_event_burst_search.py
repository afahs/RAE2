#!/usr/bin/env python
"""Event-level burst search for expected-active Jupiter intervals.

This pass looks for short positive excursions in locally normalized RAE-2
power at 0.45 and 3.93 MHz, then asks whether the event rate is higher inside
a priori Jupiter-active selectors than in same-day Jupiter-visible controls.
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
from rylevonberg.table_io import read_table


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.constants import FREQUENCY_MAP_MHZ  # noqa: E402
from rylevonberg.util import ensure_dir, software_versions, write_json  # noqa: E402
from scripts.run_jupiter_expected_active_selector_analysis import (  # noqa: E402
    ANTENNA_COLOR,
    ANTENNA_LABEL,
    SELECTOR_LABEL,
    build_selector_masks,
)
from scripts.run_jupiter_historical_window_phase_survey import load_windows  # noqa: E402
from scripts.run_jupiter_literature_controls import bootstrap_mean_ci, sign_flip_p  # noqa: E402


DEFAULT_CLEAN = ROOT / "outputs/control_survey_earth_sun_postnov1974_v2/01_ingest/cleaned_timeseries.csv"
DEFAULT_GEOMETRY = ROOT / "outputs/jupiter_phase_pattern_survey_v1/jupiter_spice_visibility_geometry_grid.csv"
DEFAULT_WINDOWS = ROOT / "configs/jupiter_warwick_dulk_riddle_1975_active_windows.csv"
DEFAULT_OUT = ROOT / "outputs/jupiter_event_burst_search_v1"
DEFAULT_FREQUENCY_BANDS = [1, 6]
DEFAULT_ANTENNAS = ["rv1_coarse", "rv2_coarse"]
SELECTOR_ORDER = [
    "literature_io_phase_windows",
    "maser_top25",
    "maser_top10",
    "maser_top05",
    "io_windows_and_maser_top10",
    "historical_wdr_reported_windows",
]


def read_clean_subset(
    clean_path: Path,
    frequency_bands: list[int],
    antennas: list[str],
    chunksize: int,
) -> pd.DataFrame:
    usecols = ["time", "frequency_band", "frequency_mhz", "antenna", "power", "is_valid"]
    frames: list[pd.DataFrame] = []
    band_set = {int(v) for v in frequency_bands}
    antenna_set = {str(v) for v in antennas}
    for chunk in read_table(
        clean_path,
        usecols=usecols,
        parse_dates=["time"],
        chunksize=int(chunksize),
        low_memory=False,
    ):
        keep = (
            chunk["frequency_band"].isin(band_set)
            & chunk["antenna"].astype(str).isin(antenna_set)
            & chunk["is_valid"].astype(bool)
        )
        sub = chunk.loc[keep].copy()
        if sub.empty:
            continue
        sub["power"] = pd.to_numeric(sub["power"], errors="coerce")
        sub = sub[np.isfinite(sub["power"]) & sub["power"].gt(0)].copy()
        if sub.empty:
            continue
        sub["frequency_band"] = sub["frequency_band"].astype(int)
        sub["frequency_mhz"] = pd.to_numeric(sub["frequency_mhz"], errors="coerce")
        sub["antenna"] = sub["antenna"].astype(str)
        sub["log_power"] = np.log(sub["power"].to_numpy(dtype=float))
        frames.append(sub)
    if not frames:
        return pd.DataFrame(columns=usecols + ["log_power", "date"])
    out = pd.concat(frames, ignore_index=True)
    out["date"] = out["time"].dt.floor("D")
    return out.sort_values(["antenna", "frequency_band", "time"]).reset_index(drop=True)


def add_local_normalization(samples: pd.DataFrame, window: str, min_periods: int) -> pd.DataFrame:
    if samples.empty:
        return samples.copy()
    groups: list[pd.DataFrame] = []
    for _, group in samples.groupby(["antenna", "frequency_band"], sort=True):
        work = group.sort_values("time").copy()
        series = work.set_index("time")["log_power"]
        rolling = series.rolling(window, center=True, min_periods=int(min_periods))
        work["local_median_log_power"] = rolling.median().to_numpy(dtype=float)
        work["local_q25_log_power"] = rolling.quantile(0.25).to_numpy(dtype=float)
        work["local_q75_log_power"] = rolling.quantile(0.75).to_numpy(dtype=float)
        local_sigma = (work["local_q75_log_power"] - work["local_q25_log_power"]) / 1.349
        global_q25 = float(series.quantile(0.25))
        global_q50 = float(series.quantile(0.50))
        global_q75 = float(series.quantile(0.75))
        global_sigma = (global_q75 - global_q25) / 1.349
        if not np.isfinite(global_sigma) or global_sigma <= 1e-9:
            global_sigma = float(series.std(ddof=0))
        work.loc[~np.isfinite(work["local_median_log_power"]), "local_median_log_power"] = global_q50
        bad_scale = ~np.isfinite(local_sigma) | (local_sigma <= 1e-9)
        work["local_sigma_log_power"] = local_sigma
        work.loc[bad_scale, "local_sigma_log_power"] = global_sigma
        work["local_z_log_power"] = (
            work["log_power"] - work["local_median_log_power"]
        ) / work["local_sigma_log_power"]
        groups.append(work)
    out = pd.concat(groups, ignore_index=True)
    return out[np.isfinite(out["local_z_log_power"])].sort_values("time").reset_index(drop=True)


def load_geometry(geometry_path: Path) -> pd.DataFrame:
    geom_cols = [
        "time",
        "jupiter_visible_by_moon",
        "earth_visible_by_moon",
        "jupiter_limb_angle_deg",
        "earth_limb_angle_deg",
        "jupiter_cml_spice_deg",
        "io_phase_spice_deg",
        "io_phase_spice_reverse_deg",
        "jupiter_range_au",
        "maser_zarka_full_score",
        "maser_zarka_io_score",
        "maser_leblanc_1978_score",
    ]
    geom = read_table(geometry_path, usecols=geom_cols, parse_dates=["time"], low_memory=False)
    return geom.sort_values("time").drop_duplicates("time").reset_index(drop=True)


def merge_geometry(samples: pd.DataFrame, geometry: pd.DataFrame, tolerance_s: float) -> pd.DataFrame:
    if samples.empty:
        return samples.copy()
    merged = pd.merge_asof(
        samples.sort_values("time"),
        geometry.sort_values("time"),
        on="time",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=float(tolerance_s)),
    )
    merged = merged.dropna(subset=["jupiter_cml_spice_deg", "io_phase_spice_deg"]).copy()
    for col in ["jupiter_visible_by_moon", "earth_visible_by_moon"]:
        merged[col] = merged[col].astype(bool)
    merged["date"] = merged["time"].dt.floor("D")
    merged["sample_id"] = np.arange(len(merged), dtype=np.int64)
    return merged.reset_index(drop=True)


def cluster_high_samples(
    group: pd.DataFrame,
    z_threshold: float,
    max_gap_seconds: float,
    min_cluster_samples: int,
    max_event_minutes: float,
) -> list[tuple[int, int]]:
    work = group.sort_values("time").reset_index(drop=True)
    high_pos = np.where(work["local_z_log_power"].to_numpy(dtype=float) >= float(z_threshold))[0]
    if high_pos.size == 0:
        return []
    clusters: list[tuple[int, int]] = []
    start = int(high_pos[0])
    last = int(high_pos[0])
    times = pd.DatetimeIndex(work["time"])
    for pos_value in high_pos[1:]:
        pos = int(pos_value)
        if (times[pos] - times[last]).total_seconds() <= float(max_gap_seconds):
            last = pos
        else:
            clusters.append((start, last))
            start = last = pos
    clusters.append((start, last))

    out: list[tuple[int, int]] = []
    for start, end in clusters:
        z = work.loc[start:end, "local_z_log_power"].to_numpy(dtype=float)
        high_count = int(np.count_nonzero(z >= float(z_threshold)))
        duration_min = (times[end] - times[start]).total_seconds() / 60.0
        if high_count >= int(min_cluster_samples) and duration_min <= float(max_event_minutes):
            out.append((start, end))
    return out


def build_event_catalog(
    samples: pd.DataFrame,
    selector_names: list[str],
    z_threshold: float,
    max_gap_seconds: float,
    min_cluster_samples: int,
    max_event_minutes: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    visible = samples[samples["jupiter_visible_by_moon"].astype(bool)].copy()
    for (antenna, band), group in visible.groupby(["antenna", "frequency_band"], sort=True):
        work = group.sort_values("time").reset_index(drop=True)
        clusters = cluster_high_samples(
            work,
            z_threshold=z_threshold,
            max_gap_seconds=max_gap_seconds,
            min_cluster_samples=min_cluster_samples,
            max_event_minutes=max_event_minutes,
        )
        for start, end in clusters:
            segment = work.iloc[start : end + 1].copy()
            high = segment[segment["local_z_log_power"].ge(float(z_threshold))]
            if high.empty:
                continue
            peak_idx = int(high["local_z_log_power"].idxmax())
            peak = work.loc[peak_idx]
            start_time = pd.Timestamp(segment["time"].iloc[0])
            end_time = pd.Timestamp(segment["time"].iloc[-1])
            flags = {f"{name}_at_peak": bool(peak.get(name, False)) for name in selector_names}
            rows.append(
                {
                    "antenna": str(antenna),
                    "antenna_label": ANTENNA_LABEL.get(str(antenna), str(antenna)),
                    "frequency_band": int(band),
                    "frequency_mhz": float(peak["frequency_mhz"]),
                    "date": pd.Timestamp(peak["time"]).floor("D"),
                    "burst_start_time": start_time,
                    "burst_end_time": end_time,
                    "burst_peak_time": pd.Timestamp(peak["time"]),
                    "duration_min": float((end_time - start_time).total_seconds() / 60.0),
                    "n_high_samples": int(len(high)),
                    "peak_sample_id": int(peak["sample_id"]),
                    "peak_power": float(peak["power"]),
                    "peak_log_power": float(peak["log_power"]),
                    "peak_local_z": float(peak["local_z_log_power"]),
                    "sum_excess_local_z": float((high["local_z_log_power"] - float(z_threshold)).sum()),
                    "jupiter_cml_spice_deg": float(peak["jupiter_cml_spice_deg"]),
                    "io_phase_spice_deg": float(peak["io_phase_spice_deg"]),
                    "io_phase_spice_reverse_deg": float(peak["io_phase_spice_reverse_deg"]),
                    "maser_zarka_io_score": float(peak["maser_zarka_io_score"]),
                    "maser_zarka_full_score": float(peak["maser_zarka_full_score"]),
                    "earth_visible_by_moon": bool(peak["earth_visible_by_moon"]),
                    "jupiter_limb_angle_deg": float(peak["jupiter_limb_angle_deg"]),
                    "earth_limb_angle_deg": float(peak["earth_limb_angle_deg"]),
                    "n_expected_selectors_at_peak": int(sum(flags.values())),
                    **flags,
                }
            )
    out = pd.DataFrame.from_records(rows)
    if out.empty:
        return out
    out = out.sort_values(["burst_peak_time", "antenna", "frequency_mhz"]).reset_index(drop=True)
    out.insert(0, "event_id", np.arange(len(out), dtype=int))
    return out


def annotate_event_coincidences(events: pd.DataFrame, tolerance_s: float) -> pd.DataFrame:
    """Flag events with another burst candidate nearby in time."""
    if events.empty:
        return events.copy()
    out = events.sort_values("burst_peak_time").reset_index(drop=True).copy()
    times = out["burst_peak_time"].to_numpy(dtype="datetime64[ns]").astype("int64")
    tol_ns = int(float(tolerance_s) * 1e9)
    rows = []
    for idx, row in out.iterrows():
        lo = int(np.searchsorted(times, times[idx] - tol_ns, side="left"))
        hi = int(np.searchsorted(times, times[idx] + tol_ns, side="right"))
        near = out.iloc[lo:hi].drop(index=idx, errors="ignore")
        same_freq = near[np.isclose(near["frequency_mhz"].to_numpy(dtype=float), float(row["frequency_mhz"]))]
        same_antenna = near[near["antenna"].astype(str).eq(str(row["antenna"]))]
        other_channel = near[
            near["antenna"].astype(str).ne(str(row["antenna"]))
            | ~np.isclose(near["frequency_mhz"].to_numpy(dtype=float), float(row["frequency_mhz"]))
        ]
        other_freq_same_antenna = same_antenna[
            ~np.isclose(same_antenna["frequency_mhz"].to_numpy(dtype=float), float(row["frequency_mhz"]))
        ]
        rows.append(
            {
                "coincident_event_count": int(len(near)),
                "coincident_other_channel_event_count": int(len(other_channel)),
                "coincident_antenna_count": int(other_channel["antenna"].nunique()) if not other_channel.empty else 0,
                "coincident_frequency_count": int(other_channel["frequency_mhz"].nunique()) if not other_channel.empty else 0,
                "has_any_coincidence": bool(len(other_channel) > 0),
                "has_cross_antenna_same_frequency": bool(
                    not same_freq.empty and same_freq["antenna"].astype(str).ne(str(row["antenna"])).any()
                ),
                "has_cross_frequency_same_antenna": bool(not other_freq_same_antenna.empty),
            }
        )
    annotated = pd.concat([out, pd.DataFrame(rows)], axis=1)
    return annotated.sort_values("event_id").reset_index(drop=True)


def daily_event_rate_points(
    samples: pd.DataFrame,
    events: pd.DataFrame,
    selected_mask: pd.Series,
    selector_name: str,
    min_selected_samples: int,
    min_control_samples: int,
    high_z: float = 3.0,
) -> pd.DataFrame:
    visible = samples["jupiter_visible_by_moon"].astype(bool)
    selected_samples = samples[visible & selected_mask.astype(bool)].copy()
    control_samples = samples[visible & ~selected_mask.astype(bool)].copy()
    keys = ["date", "antenna", "frequency_band", "frequency_mhz"]
    if selected_samples.empty or control_samples.empty:
        return pd.DataFrame()
    selected_daily = (
        selected_samples.groupby(keys, sort=True)
        .agg(
            selected_n_samples=("local_z_log_power", "size"),
            selected_median_local_z=("local_z_log_power", "median"),
            selected_high_sample_fraction=("local_z_log_power", lambda x: float(np.mean(np.asarray(x) >= float(high_z)))),
        )
        .reset_index()
    )
    control_daily = (
        control_samples.groupby(keys, sort=True)
        .agg(
            control_n_samples=("local_z_log_power", "size"),
            control_median_local_z=("local_z_log_power", "median"),
            control_high_sample_fraction=("local_z_log_power", lambda x: float(np.mean(np.asarray(x) >= float(high_z)))),
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

    selected_event_daily = pd.DataFrame(columns=keys + ["selected_n_events", "selected_peak_z_max"])
    control_event_daily = pd.DataFrame(columns=keys + ["control_n_events", "control_peak_z_max"])
    flag_col = f"{selector_name}_at_peak"
    if not events.empty and flag_col in events.columns:
        selected_events = events[events[flag_col].astype(bool)].copy()
        control_events = events[~events[flag_col].astype(bool)].copy()
        if not selected_events.empty:
            selected_event_daily = (
                selected_events.groupby(keys, sort=True)
                .agg(selected_n_events=("event_id", "size"), selected_peak_z_max=("peak_local_z", "max"))
                .reset_index()
            )
        if not control_events.empty:
            control_event_daily = (
                control_events.groupby(keys, sort=True)
                .agg(control_n_events=("event_id", "size"), control_peak_z_max=("peak_local_z", "max"))
                .reset_index()
            )
    paired = paired.merge(selected_event_daily, on=keys, how="left")
    paired = paired.merge(control_event_daily, on=keys, how="left")
    paired[["selected_n_events", "control_n_events"]] = paired[
        ["selected_n_events", "control_n_events"]
    ].fillna(0).astype(int)
    paired["selector"] = selector_name
    paired["selector_label"] = SELECTOR_LABEL.get(selector_name, selector_name)
    paired["antenna_label"] = paired["antenna"].map(lambda v: ANTENNA_LABEL.get(str(v), str(v)))
    paired["selected_event_rate_per_1000_samples"] = paired["selected_n_events"] / paired["selected_n_samples"] * 1000.0
    paired["control_event_rate_per_1000_samples"] = paired["control_n_events"] / paired["control_n_samples"] * 1000.0
    paired["selected_minus_control_event_rate_per_1000_samples"] = (
        paired["selected_event_rate_per_1000_samples"] - paired["control_event_rate_per_1000_samples"]
    )
    paired["selected_minus_control_high_sample_fraction"] = (
        paired["selected_high_sample_fraction"] - paired["control_high_sample_fraction"]
    )
    paired["selected_minus_control_median_local_z"] = (
        paired["selected_median_local_z"] - paired["control_median_local_z"]
    )
    return paired


def summarize_daily_event_rates(
    daily: pd.DataFrame,
    rng: np.random.Generator,
    n_boot: int,
    n_perm: int,
) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    has_event_set = "event_set" in daily.columns
    group_cols = (["event_set"] if has_event_set else []) + ["selector", "antenna", "frequency_band", "frequency_mhz"]
    for key, grp in daily.groupby(group_cols, sort=True):
        if has_event_set:
            event_set, selector, antenna, band, freq = key
        else:
            event_set = "all_bursts"
            selector, antenna, band, freq = key
        diff = grp["selected_minus_control_event_rate_per_1000_samples"].to_numpy(dtype=float)
        high_diff = grp["selected_minus_control_high_sample_fraction"].to_numpy(dtype=float)
        med_diff = grp["selected_minus_control_median_local_z"].to_numpy(dtype=float)
        mean_diff, lo_diff, hi_diff = bootstrap_mean_ci(diff, rng, n_boot)
        high_mean, high_lo, high_hi = bootstrap_mean_ci(high_diff, rng, n_boot)
        med_mean, med_lo, med_hi = bootstrap_mean_ci(med_diff, rng, n_boot)
        p_two, p_pos = sign_flip_p(diff, rng, n_perm)
        high_p_two, high_p_pos = sign_flip_p(high_diff, rng, n_perm)
        rows.append(
            {
                "event_set": event_set,
                "selector": selector,
                "selector_label": SELECTOR_LABEL.get(str(selector), str(selector)),
                "antenna": antenna,
                "antenna_label": ANTENNA_LABEL.get(str(antenna), str(antenna)),
                "frequency_band": int(band),
                "frequency_mhz": float(freq),
                "n_paired_days": int(len(grp)),
                "selected_total_samples": int(grp["selected_n_samples"].sum()),
                "control_total_samples": int(grp["control_n_samples"].sum()),
                "selected_total_events": int(grp["selected_n_events"].sum()),
                "control_total_events": int(grp["control_n_events"].sum()),
                "selected_event_rate_per_1000_samples_mean": float(grp["selected_event_rate_per_1000_samples"].mean()),
                "control_event_rate_per_1000_samples_mean": float(grp["control_event_rate_per_1000_samples"].mean()),
                "mean_selected_minus_control_event_rate_per_1000_samples": mean_diff,
                "boot_lo_selected_minus_control_event_rate_per_1000_samples": lo_diff,
                "boot_hi_selected_minus_control_event_rate_per_1000_samples": hi_diff,
                "positive_day_fraction_event_rate": float((diff > 0).mean()) if len(diff) else np.nan,
                "signflip_p_two_sided_event_rate": p_two,
                "signflip_p_positive_event_rate": p_pos,
                "mean_selected_minus_control_high_sample_fraction": high_mean,
                "boot_lo_selected_minus_control_high_sample_fraction": high_lo,
                "boot_hi_selected_minus_control_high_sample_fraction": high_hi,
                "signflip_p_two_sided_high_sample_fraction": high_p_two,
                "signflip_p_positive_high_sample_fraction": high_p_pos,
                "mean_selected_minus_control_median_local_z": med_mean,
                "boot_lo_selected_minus_control_median_local_z": med_lo,
                "boot_hi_selected_minus_control_median_local_z": med_hi,
            }
        )
    return pd.DataFrame(rows)


def plot_event_rate_spectrum(
    summary: pd.DataFrame,
    out_dir: Path,
    filename: str = "jupiter_event_rate_excess_spectrum.png",
    title: str = "Jupiter expected-active burst search: event-rate excess over same-day controls",
) -> Path | None:
    if summary.empty:
        return None
    selectors = [s for s in SELECTOR_ORDER if s in set(summary["selector"])]
    fig, axes = plt.subplots(
        len(selectors),
        1,
        figsize=(10.5, max(2.15 * len(selectors), 5.5)),
        sharex=True,
        sharey=False,
    )
    axes = np.atleast_1d(axes)
    metric = "mean_selected_minus_control_event_rate_per_1000_samples"
    lo_col = "boot_lo_selected_minus_control_event_rate_per_1000_samples"
    hi_col = "boot_hi_selected_minus_control_event_rate_per_1000_samples"
    for ax, selector in zip(axes, selectors):
        sub = summary[summary["selector"].eq(selector)].copy()
        for antenna, grp in sub.groupby("antenna", sort=True):
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
            sig = pd.to_numeric(grp["signflip_p_positive_event_rate"], errors="coerce").to_numpy(dtype=float) < 0.05
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
        ax.set_xticks([FREQUENCY_MAP_MHZ[b] for b in DEFAULT_FREQUENCY_BANDS])
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.set_ylabel("events / 1000 samples")
        ax.set_title(SELECTOR_LABEL.get(selector, selector), loc="left", fontsize=10)
        ax.grid(True, color="0.9", lw=0.45)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    axes[-1].set_xlabel("frequency (MHz)")
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    path = out_dir / filename
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_event_rate_heatmap(
    summary: pd.DataFrame,
    out_dir: Path,
    filename: str = "jupiter_event_rate_excess_heatmap.png",
    title: str = "Burst event-rate excess: selected minus same-day control",
) -> Path | None:
    if summary.empty:
        return None
    work = summary.copy()
    work["channel"] = work["antenna_label"] + " " + work["frequency_mhz"].map(lambda v: f"{v:.2f}")
    mat = work.pivot_table(
        index="selector_label",
        columns="channel",
        values="mean_selected_minus_control_event_rate_per_1000_samples",
        aggfunc="first",
    )
    selector_order = [SELECTOR_LABEL[s] for s in SELECTOR_ORDER if SELECTOR_LABEL[s] in mat.index]
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
    lim = max(0.05, float(np.nanpercentile(np.abs(finite), 98))) if finite.size else 0.1
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    im = ax.imshow(vals, aspect="auto", cmap="coolwarm", norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim))
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels(mat.index)
    ax.set_xticks(np.arange(len(mat.columns)))
    ax.set_xticklabels(mat.columns, rotation=40, ha="right", fontsize=8)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, pad=0.015)
    cbar.set_label("events / 1000 valid visible samples")
    fig.tight_layout()
    path = out_dir / filename
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_top_event_strips(
    samples: pd.DataFrame,
    events: pd.DataFrame,
    out_dir: Path,
    z_threshold: float,
    minutes: float,
    top_n: int,
) -> Path | None:
    if samples.empty or events.empty:
        return None
    ranked = events.sort_values(
        ["n_expected_selectors_at_peak", "peak_local_z", "n_high_samples"],
        ascending=[False, False, False],
    ).head(int(top_n))
    if ranked.empty:
        return None
    fig, axes = plt.subplots(len(ranked), 1, figsize=(12.0, max(2.3 * len(ranked), 4.0)), sharex=True)
    axes = np.atleast_1d(axes)
    line_styles = {0.45: "-", 3.93: "--"}
    for ax, (_, event) in zip(axes, ranked.iterrows()):
        peak_time = pd.Timestamp(event["burst_peak_time"])
        lo = peak_time - pd.Timedelta(minutes=float(minutes))
        hi = peak_time + pd.Timedelta(minutes=float(minutes))
        window = samples[(samples["time"] >= lo) & (samples["time"] <= hi)].copy()
        for (antenna, freq), grp in window.groupby(["antenna", "frequency_mhz"], sort=True):
            dx = (grp["time"] - peak_time).dt.total_seconds() / 60.0
            label = f"{ANTENNA_LABEL.get(str(antenna), str(antenna))} {float(freq):.2f} MHz"
            ax.plot(
                dx,
                grp["local_z_log_power"],
                lw=0.8,
                alpha=0.72,
                linestyle=line_styles.get(round(float(freq), 2), ":"),
                color=ANTENNA_COLOR.get(str(antenna), "0.2"),
                label=label,
            )
        ax.axvline(0, color="0.2", lw=0.85)
        ax.axhline(float(z_threshold), color="0.25", lw=0.8, linestyle=":")
        finite_y = window["local_z_log_power"].to_numpy(dtype=float)
        finite_y = finite_y[np.isfinite(finite_y)]
        if finite_y.size:
            y_lo = max(-15.0, float(np.nanpercentile(finite_y, 1)))
            y_hi = min(45.0, max(float(z_threshold) + 2.0, float(np.nanpercentile(finite_y, 99))))
            if y_hi > y_lo:
                ax.set_ylim(y_lo, y_hi)
        ax.set_ylabel("local z")
        ax.grid(True, color="0.9", lw=0.45)
        ax.set_title(
            f"event {int(event['event_id'])}  {peak_time}  "
            f"{event['antenna_label']} {float(event['frequency_mhz']):.2f} MHz  "
            f"z={float(event['peak_local_z']):.2f}  Io={float(event['io_phase_spice_deg']):.1f}  "
            f"CML={float(event['jupiter_cml_spice_deg']):.1f}  selectors={int(event['n_expected_selectors_at_peak'])}",
            loc="left",
            fontsize=9,
        )
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, loc="upper right", fontsize=8)
    axes[-1].set_xlabel("minutes from event peak")
    fig.suptitle("Top locally normalized positive burst candidates near expected-active selectors")
    fig.tight_layout(rect=[0, 0, 0.94, 0.97])
    path = out_dir / "jupiter_top_event_candidate_strips.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report(
    out_dir: Path,
    samples: pd.DataFrame,
    events: pd.DataFrame,
    summary: pd.DataFrame,
    paths: list[Path],
    config: dict[str, object],
    thresholds: dict[str, float],
) -> Path:
    rank_cols = [
        "event_set",
        "selector_label",
        "antenna_label",
        "frequency_mhz",
        "n_paired_days",
        "selected_total_events",
        "control_total_events",
        "mean_selected_minus_control_event_rate_per_1000_samples",
        "boot_lo_selected_minus_control_event_rate_per_1000_samples",
        "boot_hi_selected_minus_control_event_rate_per_1000_samples",
        "signflip_p_positive_event_rate",
        "mean_selected_minus_control_high_sample_fraction",
        "signflip_p_positive_high_sample_fraction",
    ]
    top = summary.sort_values("mean_selected_minus_control_event_rate_per_1000_samples", ascending=False) if not summary.empty else summary
    coincident = summary[summary["event_set"].eq("coincident_bursts")].copy() if "event_set" in summary.columns else pd.DataFrame()
    coincident = coincident.sort_values("mean_selected_minus_control_event_rate_per_1000_samples", ascending=False)
    event_counts = (
        events.groupby(["antenna_label", "frequency_mhz"], sort=True)
        .size()
        .reset_index(name="n_events")
        .to_string(index=False)
        if not events.empty
        else "(none)"
    )
    coincidence_counts = (
        events.groupby(["antenna_label", "frequency_mhz"], sort=True)
        .agg(
            n_events=("event_id", "size"),
            n_any_coincidence=("has_any_coincidence", "sum"),
            n_cross_antenna_same_frequency=("has_cross_antenna_same_frequency", "sum"),
            n_cross_frequency_same_antenna=("has_cross_frequency_same_antenna", "sum"),
        )
        .reset_index()
        .to_string(index=False)
        if not events.empty and "has_any_coincidence" in events.columns
        else "(none)"
    )
    top_events = (
        events.sort_values(["n_expected_selectors_at_peak", "peak_local_z"], ascending=[False, False])
        .head(12)[
            [
                "event_id",
                "burst_peak_time",
                "antenna_label",
                "frequency_mhz",
                "peak_local_z",
                "n_high_samples",
                "io_phase_spice_deg",
                "jupiter_cml_spice_deg",
                "maser_zarka_io_score",
                "n_expected_selectors_at_peak",
            ]
        ]
        .to_string(index=False)
        if not events.empty
        else "(none)"
    )
    lines = [
        "# Jupiter Event-Level Burst Search",
        "",
        "This run clusters short positive excursions in 20-minute locally normalized log-power at 0.45 and 3.93 MHz. Event rates inside expected-active selectors are compared with same-day, same-channel Jupiter-visible controls.",
        "",
        "## Configuration",
        "",
        *[f"- `{k}`: `{v}`" for k, v in config.items() if k != "software_versions"],
        "",
        "## MASER Score Thresholds",
        "",
        *[f"- `{k}`: `{v:.6f}`" for k, v in thresholds.items()],
        "",
        "## Sample And Event Counts",
        "",
        f"- merged locally normalized samples: `{len(samples)}`",
        f"- Jupiter-visible samples: `{int(samples['jupiter_visible_by_moon'].sum()) if not samples.empty else 0}`",
        f"- clustered burst candidates: `{len(events)}`",
        "",
        event_counts,
        "",
        "## Coincidence Counts",
        "",
        coincidence_counts,
        "",
        "## Strongest Positive Event-Rate Excesses",
        "",
        top[rank_cols].head(18).to_string(index=False) if not top.empty else "(none)",
        "",
        "## Strongest Positive Coincident Event-Rate Excesses",
        "",
        coincident[rank_cols].head(18).to_string(index=False) if not coincident.empty else "(none)",
        "",
        "## Strongest Negative Event-Rate Excesses",
        "",
        top.tail(12).sort_values("mean_selected_minus_control_event_rate_per_1000_samples")[rank_cols].to_string(index=False)
        if not top.empty
        else "(none)",
        "",
        "## Top Individual Candidates",
        "",
        top_events,
        "",
        "## Interpretation Notes",
        "",
        "- The primary statistic is selected minus same-day control burst events per 1000 valid Jupiter-visible samples.",
        "- Bootstrap intervals and sign-flip tests are over paired UTC day/channel rows, not individual samples.",
        "- A credible Jupiter signature should recur in expected Io/CML regions and preferably appear across neighboring frequencies or both antennas; isolated single-channel excesses remain suspect.",
        "- The event catalog is intentionally permissive so candidate strips can be visually inspected before making a claim.",
        "",
        "## Files",
        "",
        *[f"- `{p}`" for p in paths],
    ]
    path = out_dir / "jupiter_event_burst_search_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", type=Path, default=DEFAULT_CLEAN)
    parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--historical-windows", type=Path, default=DEFAULT_WINDOWS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--frequency-band", type=int, action="append", default=None)
    parser.add_argument("--antenna", type=str, action="append", default=None)
    parser.add_argument("--io-window", type=float, nargs=2, action="append", default=[(80.0, 100.0), (235.0, 260.0)])
    parser.add_argument("--local-window", type=str, default="20min")
    parser.add_argument("--local-min-periods", type=int, default=8)
    parser.add_argument("--z-threshold", type=float, default=3.0)
    parser.add_argument("--max-gap-seconds", type=float, default=180.0)
    parser.add_argument("--coincidence-seconds", type=float, default=30.0)
    parser.add_argument("--min-cluster-samples", type=int, default=2)
    parser.add_argument("--max-event-minutes", type=float, default=30.0)
    parser.add_argument("--min-selected-samples-per-day", type=int, default=5)
    parser.add_argument("--min-control-samples-per-day", type=int, default=25)
    parser.add_argument("--geometry-tolerance-s", type=float, default=360.0)
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    parser.add_argument("--bootstrap-samples", type=int, default=3000)
    parser.add_argument("--permutations", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--write-samples", action="store_true")
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    rng = np.random.default_rng(int(args.seed))
    frequency_bands = [int(v) for v in (args.frequency_band or DEFAULT_FREQUENCY_BANDS)]
    antennas = [str(v) for v in (args.antenna or DEFAULT_ANTENNAS)]
    io_windows = [(float(a), float(b)) for a, b in args.io_window]
    config = {
        "clean": str(args.clean),
        "geometry": str(args.geometry),
        "historical_windows": str(args.historical_windows),
        "frequency_bands": frequency_bands,
        "antennas": antennas,
        "io_windows_deg": [[float(a), float(b)] for a, b in io_windows],
        "local_window": str(args.local_window),
        "local_min_periods": int(args.local_min_periods),
        "z_threshold": float(args.z_threshold),
        "max_gap_seconds": float(args.max_gap_seconds),
        "coincidence_seconds": float(args.coincidence_seconds),
        "min_cluster_samples": int(args.min_cluster_samples),
        "max_event_minutes": float(args.max_event_minutes),
        "min_selected_samples_per_day": int(args.min_selected_samples_per_day),
        "min_control_samples_per_day": int(args.min_control_samples_per_day),
        "geometry_tolerance_s": float(args.geometry_tolerance_s),
        "chunksize": int(args.chunksize),
        "bootstrap_samples": int(args.bootstrap_samples),
        "permutations": int(args.permutations),
        "seed": int(args.seed),
        "write_samples": bool(args.write_samples),
        "software_versions": software_versions(),
    }
    write_json(out_dir / "run_config.json", config)

    clean = read_clean_subset(args.clean, frequency_bands=frequency_bands, antennas=antennas, chunksize=int(args.chunksize))
    clean = add_local_normalization(clean, window=str(args.local_window), min_periods=int(args.local_min_periods))
    geometry = load_geometry(args.geometry)
    samples = merge_geometry(clean, geometry, tolerance_s=float(args.geometry_tolerance_s))
    windows = load_windows(args.historical_windows, padding_min=0.0)
    masks, thresholds = build_selector_masks(samples, windows, io_windows)
    for name, mask in masks.items():
        samples[name] = mask.to_numpy(dtype=bool)

    selector_names = [name for name in SELECTOR_ORDER if name in masks]
    events = build_event_catalog(
        samples,
        selector_names=selector_names,
        z_threshold=float(args.z_threshold),
        max_gap_seconds=float(args.max_gap_seconds),
        min_cluster_samples=int(args.min_cluster_samples),
        max_event_minutes=float(args.max_event_minutes),
    )
    events = annotate_event_coincidences(events, tolerance_s=float(args.coincidence_seconds))

    daily_tables = []
    event_sets = [("all_bursts", events)]
    if not events.empty and "has_any_coincidence" in events.columns:
        event_sets.append(("coincident_bursts", events[events["has_any_coincidence"].astype(bool)].copy()))
    for event_set, event_frame in event_sets:
        for selector in selector_names:
            daily = daily_event_rate_points(
                samples,
                event_frame,
                pd.Series(samples[selector].to_numpy(dtype=bool), index=samples.index),
                selector_name=selector,
                min_selected_samples=int(args.min_selected_samples_per_day),
                min_control_samples=int(args.min_control_samples_per_day),
                high_z=float(args.z_threshold),
            )
            if not daily.empty:
                daily["event_set"] = event_set
                daily_tables.append(daily)
    daily_all = pd.concat(daily_tables, ignore_index=True) if daily_tables else pd.DataFrame()
    summary = summarize_daily_event_rates(
        daily_all,
        rng=rng,
        n_boot=int(args.bootstrap_samples),
        n_perm=int(args.permutations),
    )

    paths: list[Path] = []
    event_path = out_dir / "jupiter_burst_event_catalog.csv"
    daily_path = out_dir / "jupiter_burst_event_selector_daily_rates.csv"
    summary_path = out_dir / "jupiter_burst_event_selector_summary.csv"
    sample_summary_path = out_dir / "jupiter_event_search_sample_summary.csv"
    events.to_csv(event_path, index=False)
    daily_all.to_csv(daily_path, index=False)
    summary.to_csv(summary_path, index=False)
    paths.extend([event_path, daily_path, summary_path])

    sample_summary = (
        samples.groupby(["antenna", "frequency_band", "frequency_mhz", "jupiter_visible_by_moon"], sort=True)
        .agg(
            n_samples=("local_z_log_power", "size"),
            median_local_z=("local_z_log_power", "median"),
            high_z_fraction=("local_z_log_power", lambda x: float(np.mean(np.asarray(x) >= float(args.z_threshold)))),
        )
        .reset_index()
    )
    sample_summary.to_csv(sample_summary_path, index=False)
    paths.append(sample_summary_path)

    if bool(args.write_samples):
        sample_cols = [
            "sample_id",
            "time",
            "date",
            "antenna",
            "frequency_band",
            "frequency_mhz",
            "power",
            "log_power",
            "local_z_log_power",
            "jupiter_visible_by_moon",
            "earth_visible_by_moon",
            "jupiter_cml_spice_deg",
            "io_phase_spice_deg",
            "maser_zarka_io_score",
            *selector_names,
        ]
        sample_path = out_dir / "jupiter_local_normalized_samples_0p45_3p93.csv.gz"
        samples[sample_cols].to_csv(sample_path, index=False, compression="gzip")
        paths.append(sample_path)

    plot_inputs = [
        (
            summary[summary["event_set"].eq("all_bursts")] if "event_set" in summary.columns else summary,
            "jupiter_event_rate_excess_spectrum.png",
            "jupiter_event_rate_excess_heatmap.png",
            "Jupiter expected-active burst search: event-rate excess over same-day controls",
            "Burst event-rate excess: selected minus same-day control",
        ),
        (
            summary[summary["event_set"].eq("coincident_bursts")] if "event_set" in summary.columns else pd.DataFrame(),
            "jupiter_coincident_event_rate_excess_spectrum.png",
            "jupiter_coincident_event_rate_excess_heatmap.png",
            "Jupiter burst search: coincident-event excess over same-day controls",
            "Coincident burst event-rate excess: selected minus same-day control",
        ),
    ]
    maybe_paths = []
    for plot_summary, spectrum_name, heatmap_name, spectrum_title, heatmap_title in plot_inputs:
        maybe_paths.append(plot_event_rate_spectrum(plot_summary, out_dir, filename=spectrum_name, title=spectrum_title))
        maybe_paths.append(plot_event_rate_heatmap(plot_summary, out_dir, filename=heatmap_name, title=heatmap_title))
    maybe_paths.append(plot_top_event_strips(samples, events, out_dir, z_threshold=float(args.z_threshold), minutes=45.0, top_n=8))
    for maybe_path in maybe_paths:
        if maybe_path is not None:
            paths.append(maybe_path)

    report_path = write_report(
        out_dir,
        samples=samples,
        events=events,
        summary=summary,
        paths=paths,
        config=config,
        thresholds=thresholds,
    )
    print(f"Wrote {event_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
