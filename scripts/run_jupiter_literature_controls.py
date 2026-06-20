#!/usr/bin/env python
"""Literature-guided controls for Jupiter in RAE-2 Ryle-Vonberg data.

The tests here follow the usual Jupiter-radio analysis framing: occurrence or
high-tail probability in System-III CML / Io phase space, time-frequency visual
inspection, and controls that preserve observing cadence while breaking the
Jupiter selector.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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
from scripts.run_jupiter_historical_window_phase_survey import load_windows  # noqa: E402
from scripts.run_jupiter_phase_pattern_survey import (  # noqa: E402
    _add_daily_channel_normalization,
    _merge_geometry,
    _read_clean,
)


DEFAULT_CLEAN = ROOT / "outputs/control_survey_earth_sun_postnov1974_v2/01_ingest/cleaned_timeseries.csv"
DEFAULT_GEOMETRY = ROOT / "outputs/jupiter_phase_pattern_survey_v1/jupiter_spice_visibility_geometry_grid.csv"
DEFAULT_PHASE_SUMMARY = ROOT / "outputs/jupiter_phase_pattern_survey_v1/jupiter_phase_binned_summary.csv"
DEFAULT_WINDOWS = ROOT / "configs/jupiter_warwick_dulk_riddle_1975_active_windows.csv"
DEFAULT_OUT = ROOT / "outputs/jupiter_literature_controls_v1"

ANTENNA_LABEL = {"rv1_coarse": "upper V", "rv2_coarse": "lower V"}
ANTENNA_COLOR = {"rv1_coarse": "#386cb0", "rv2_coarse": "#bf5b17"}


@dataclass(frozen=True)
class WindowInterval:
    label: str
    start: pd.Timestamp
    end: pd.Timestamp
    report_start: pd.Timestamp
    report_end: pd.Timestamp
    parent: str
    shift_days: int | None
    intensity: int
    burstiness: str
    reported_freq_range_mhz: str


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Small dependency-free Spearman correlation for finite paired values."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3:
        return np.nan
    xr = pd.Series(x[keep]).rank(method="average").to_numpy(dtype=float)
    yr = pd.Series(y[keep]).rank(method="average").to_numpy(dtype=float)
    if np.nanstd(xr) <= 0 or np.nanstd(yr) <= 0:
        return np.nan
    return float(np.corrcoef(xr, yr)[0, 1])


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    mean = float(values.mean())
    if len(values) == 1 or n_boot <= 0:
        return mean, np.nan, np.nan
    draws = rng.choice(values, size=(int(n_boot), len(values)), replace=True).mean(axis=1)
    return mean, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def sign_flip_p(values: np.ndarray, rng: np.random.Generator, n_perm: int) -> tuple[float, float]:
    """Two-sided and positive-tail p-values for paired differences."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    obs = float(values.mean())
    if len(values) == 1 or n_perm <= 0:
        return np.nan, np.nan
    signs = rng.choice([-1.0, 1.0], size=(int(n_perm), len(values)), replace=True)
    null = (signs * values.reshape(1, -1)).mean(axis=1)
    p_two = (1.0 + np.sum(np.abs(null) >= abs(obs))) / (len(null) + 1.0)
    p_positive = (1.0 + np.sum(null >= obs)) / (len(null) + 1.0)
    return float(p_two), float(p_positive)


def intervals_overlap(start: pd.Timestamp, end: pd.Timestamp, intervals: list[tuple[pd.Timestamp, pd.Timestamp]]) -> bool:
    for other_start, other_end in intervals:
        if start <= other_end and end >= other_start:
            return True
    return False


def make_intervals(
    windows: pd.DataFrame,
    shift_days: list[int] | None = None,
    exclude_overlaps: bool = False,
    start_col: str = "event_start_time",
    end_col: str = "event_end_time",
) -> list[WindowInterval]:
    active_ranges = [
        (pd.Timestamp(row["event_start_time"]), pd.Timestamp(row["event_end_time"]))
        for _, row in windows.iterrows()
    ]
    shifts: list[int | None] = [None] if shift_days is None else [int(s) for s in shift_days]
    intervals: list[WindowInterval] = []
    for _, row in windows.iterrows():
        parent = str(row["historical_window_id"])
        for shift in shifts:
            delta = pd.Timedelta(days=0 if shift is None else int(shift))
            start = pd.Timestamp(row[start_col]) + delta
            end = pd.Timestamp(row[end_col]) + delta
            report_start = pd.Timestamp(row["event_start_time"]) + delta
            report_end = pd.Timestamp(row["event_end_time"]) + delta
            if shift is not None and exclude_overlaps and intervals_overlap(start, end, active_ranges):
                continue
            label = parent if shift is None else f"{parent}_shift_{shift:+d}d"
            intervals.append(
                WindowInterval(
                    label=label,
                    start=start,
                    end=end,
                    report_start=report_start,
                    report_end=report_end,
                    parent=parent,
                    shift_days=shift,
                    intensity=int(row["intensity"]),
                    burstiness=str(row["burstiness"]),
                    reported_freq_range_mhz=str(row["reported_freq_range_mhz"]),
                )
            )
    return intervals


def select_interval_samples(samples: pd.DataFrame, intervals: list[WindowInterval], selector: str) -> pd.DataFrame:
    if not intervals or samples.empty:
        return pd.DataFrame(columns=list(samples.columns))
    work = samples.sort_values("time").reset_index(drop=True)
    times = pd.to_datetime(work["time"]).to_numpy(dtype="datetime64[ns]")
    pieces = []
    for iv in intervals:
        lo = int(np.searchsorted(times, np.datetime64(iv.start), side="left"))
        hi = int(np.searchsorted(times, np.datetime64(iv.end), side="right"))
        if hi <= lo:
            continue
        sub = work.iloc[lo:hi].copy()
        sub["selector"] = selector
        sub["window_id"] = iv.label
        sub["historical_window_id"] = iv.parent
        sub["control_shift_days"] = np.nan if iv.shift_days is None else float(iv.shift_days)
        sub["selection_start_time"] = iv.start
        sub["selection_end_time"] = iv.end
        sub["event_start_time"] = iv.report_start
        sub["event_end_time"] = iv.report_end
        sub["intensity"] = iv.intensity
        sub["burstiness"] = iv.burstiness
        sub["reported_freq_range_mhz"] = iv.reported_freq_range_mhz
        sub["dt_from_start_min"] = (pd.to_datetime(sub["time"]) - iv.report_start).dt.total_seconds() / 60.0
        pieces.append(sub)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=list(samples.columns))


def load_full_samples(clean_path: Path, geometry_path: Path, tolerance_s: float) -> pd.DataFrame:
    geom = read_table(geometry_path, parse_dates=["time"], low_memory=False)
    clean = _read_clean(clean_path)
    clean = _add_daily_channel_normalization(clean)
    samples = _merge_geometry(clean, geom, tolerance_s)
    keep_cols = [
        "time",
        "date",
        "antenna",
        "frequency_band",
        "frequency_mhz",
        "log_power",
        "daily_z_log_power",
        "jupiter_visible_by_moon",
        "earth_visible_by_moon",
        "jupiter_limb_angle_deg",
        "earth_limb_angle_deg",
        "jupiter_cml_spice_deg",
        "io_phase_spice_deg",
        "jupiter_range_au",
        "maser_zarka_io_score",
        "maser_zarka_full_score",
        "maser_leblanc_1978_score",
    ]
    cols = [c for c in keep_cols if c in samples.columns]
    out = samples[cols].copy()
    out["time"] = pd.to_datetime(out["time"])
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values("time").reset_index(drop=True)


def phase_scramble_summary(
    phase_summary: pd.DataFrame,
    rng: np.random.Generator,
    n_perm: int,
    regime: str = "jupiter_visible",
) -> pd.DataFrame:
    rows = []
    src = phase_summary[phase_summary["regime"].astype(str).eq(regime)].copy()
    for (antenna, band, freq), grp in src.groupby(["antenna", "frequency_band", "frequency_mhz"], sort=True):
        mat_h = grp.pivot(index="io_bin_deg", columns="cml_bin_deg", values="high_power_fraction").sort_index()
        mat_s = grp.pivot(index="io_bin_deg", columns="cml_bin_deg", values="median_zarka_io_score").sort_index()
        mat_s = mat_s.reindex(index=mat_h.index, columns=mat_h.columns)
        h = mat_h.to_numpy(dtype=float)
        s = mat_s.to_numpy(dtype=float)
        obs = spearman_corr(h.ravel(), s.ravel())
        null = []
        for _ in range(int(n_perm)):
            scrambled = h.copy()
            for col in range(scrambled.shape[1]):
                scrambled[:, col] = np.roll(scrambled[:, col], int(rng.integers(0, scrambled.shape[0])))
            null.append(spearman_corr(scrambled.ravel(), s.ravel()))
        null_arr = np.asarray(null, dtype=float)
        null_arr = null_arr[np.isfinite(null_arr)]
        p_two = (
            float((1.0 + np.sum(np.abs(null_arr) >= abs(obs))) / (len(null_arr) + 1.0))
            if np.isfinite(obs) and len(null_arr)
            else np.nan
        )
        rows.append(
            {
                "antenna": antenna,
                "antenna_label": ANTENNA_LABEL.get(str(antenna), str(antenna)),
                "frequency_band": int(band),
                "frequency_mhz": float(freq),
                "regime": regime,
                "n_phase_bins": int(np.isfinite(h).sum()),
                "observed_spearman_high_tail_vs_maser": obs,
                "phase_scramble_null_median": float(np.nanmedian(null_arr)) if len(null_arr) else np.nan,
                "phase_scramble_null_lo": float(np.nanquantile(null_arr, 0.025)) if len(null_arr) else np.nan,
                "phase_scramble_null_hi": float(np.nanquantile(null_arr, 0.975)) if len(null_arr) else np.nan,
                "phase_scramble_p_two_sided": p_two,
            }
        )
    return pd.DataFrame(rows)


def score_daily_controls(
    samples: pd.DataFrame,
    high_z: float,
    score_quantile: float,
    min_daily_samples: int,
    rng: np.random.Generator,
    n_boot: int,
    n_perm: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    visible = samples[samples["jupiter_visible_by_moon"].astype(bool)].copy()
    score = pd.to_numeric(visible["maser_zarka_io_score"], errors="coerce")
    high_threshold = float(score.quantile(float(score_quantile)))
    visible["score_class"] = np.where(score >= high_threshold, "high_maser_score", "low_or_zero_maser_score")
    visible["high_tail"] = visible["daily_z_log_power"].to_numpy(dtype=float) > float(high_z)
    daily = (
        visible.groupby(["date", "antenna", "frequency_band", "frequency_mhz", "score_class"], sort=True)
        .agg(
            n_samples=("daily_z_log_power", "size"),
            high_tail_fraction=("high_tail", "mean"),
            median_daily_z=("daily_z_log_power", "median"),
        )
        .reset_index()
    )
    daily = daily[daily["n_samples"] >= int(min_daily_samples)].copy()
    wide = daily.pivot_table(
        index=["date", "antenna", "frequency_band", "frequency_mhz"],
        columns="score_class",
        values=["n_samples", "high_tail_fraction", "median_daily_z"],
        aggfunc="first",
    )
    wide.columns = ["_".join(map(str, c)).strip() for c in wide.columns.to_flat_index()]
    wide = wide.reset_index()
    needed = [
        "high_tail_fraction_high_maser_score",
        "high_tail_fraction_low_or_zero_maser_score",
        "median_daily_z_high_maser_score",
        "median_daily_z_low_or_zero_maser_score",
    ]
    for col in needed:
        if col not in wide.columns:
            wide[col] = np.nan
    wide = wide.dropna(subset=needed).copy()
    wide["daily_high_minus_low_high_tail_fraction"] = (
        wide["high_tail_fraction_high_maser_score"] - wide["high_tail_fraction_low_or_zero_maser_score"]
    )
    wide["daily_high_minus_low_median_z"] = (
        wide["median_daily_z_high_maser_score"] - wide["median_daily_z_low_or_zero_maser_score"]
    )
    rows = []
    for (antenna, band, freq), grp in wide.groupby(["antenna", "frequency_band", "frequency_mhz"], sort=True):
        tail = grp["daily_high_minus_low_high_tail_fraction"].to_numpy(dtype=float)
        med = grp["daily_high_minus_low_median_z"].to_numpy(dtype=float)
        tail_mean, tail_lo, tail_hi = bootstrap_mean_ci(tail, rng, n_boot)
        med_mean, med_lo, med_hi = bootstrap_mean_ci(med, rng, n_boot)
        tail_p_two, tail_p_pos = sign_flip_p(tail, rng, n_perm)
        med_p_two, med_p_pos = sign_flip_p(med, rng, n_perm)
        rows.append(
            {
                "antenna": antenna,
                "antenna_label": ANTENNA_LABEL.get(str(antenna), str(antenna)),
                "frequency_band": int(band),
                "frequency_mhz": float(freq),
                "score_high_quantile": float(score_quantile),
                "score_high_threshold": high_threshold,
                "n_paired_days": int(len(grp)),
                "mean_daily_high_minus_low_high_tail_fraction": tail_mean,
                "boot_lo_daily_high_minus_low_high_tail_fraction": tail_lo,
                "boot_hi_daily_high_minus_low_high_tail_fraction": tail_hi,
                "signflip_p_two_sided_high_tail": tail_p_two,
                "signflip_p_positive_high_tail": tail_p_pos,
                "positive_day_fraction_high_tail": float((tail > 0).mean()) if len(tail) else np.nan,
                "mean_daily_high_minus_low_median_z": med_mean,
                "boot_lo_daily_high_minus_low_median_z": med_lo,
                "boot_hi_daily_high_minus_low_median_z": med_hi,
                "signflip_p_two_sided_median_z": med_p_two,
                "signflip_p_positive_median_z": med_p_pos,
            }
        )
    return pd.DataFrame(rows), wide


def aggregate_window_samples(selected: pd.DataFrame, high_z: float) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame()
    work = selected.copy()
    work["high_tail"] = work["daily_z_log_power"].to_numpy(dtype=float) > float(high_z)
    return (
        work.groupby(
            [
                "selector",
                "window_id",
                "historical_window_id",
                "control_shift_days",
                "antenna",
                "frequency_band",
                "frequency_mhz",
            ],
            dropna=False,
            sort=True,
        )
        .agg(
            n_samples=("daily_z_log_power", "size"),
            median_daily_z=("daily_z_log_power", "median"),
            mean_daily_z=("daily_z_log_power", "mean"),
            max_daily_z=("daily_z_log_power", "max"),
            high_tail_fraction=("high_tail", "mean"),
            intensity=("intensity", "first"),
            burstiness=("burstiness", "first"),
            reported_freq_range_mhz=("reported_freq_range_mhz", "first"),
        )
        .reset_index()
    )


def build_historical_matched_tables(
    samples: pd.DataFrame,
    windows: pd.DataFrame,
    shift_days: list[int],
    high_z: float,
    min_window_samples: int,
    min_control_shifts: int,
    rng: np.random.Generator,
    n_boot: int,
    n_perm: int,
    exclude_control_overlaps: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    active = select_interval_samples(samples, make_intervals(windows), "historical_active")
    control = select_interval_samples(
        samples,
        make_intervals(windows, shift_days=shift_days, exclude_overlaps=exclude_control_overlaps),
        "shifted_control",
    )
    active_stats = aggregate_window_samples(active, high_z)
    control_stats = aggregate_window_samples(control, high_z)
    active_stats = active_stats[active_stats["n_samples"] >= int(min_window_samples)].copy()
    control_stats = control_stats[control_stats["n_samples"] >= int(min_window_samples)].copy()
    control_rollup = (
        control_stats.groupby(["historical_window_id", "antenna", "frequency_band", "frequency_mhz"], sort=True)
        .agg(
            n_control_shifts=("window_id", "nunique"),
            control_total_samples=("n_samples", "sum"),
            control_median_daily_z=("median_daily_z", "median"),
            control_mean_daily_z=("mean_daily_z", "median"),
            control_high_tail_fraction=("high_tail_fraction", "median"),
        )
        .reset_index()
    )
    paired = active_stats.merge(
        control_rollup,
        on=["historical_window_id", "antenna", "frequency_band", "frequency_mhz"],
        how="inner",
    )
    paired = paired[paired["n_control_shifts"] >= int(min_control_shifts)].copy()
    paired["active_minus_control_median_daily_z"] = paired["median_daily_z"] - paired["control_median_daily_z"]
    paired["active_minus_control_mean_daily_z"] = paired["mean_daily_z"] - paired["control_mean_daily_z"]
    paired["active_minus_control_high_tail_fraction"] = (
        paired["high_tail_fraction"] - paired["control_high_tail_fraction"]
    )

    summary_rows = []
    candidate_stats = pd.concat([active_stats, control_stats], ignore_index=True)
    for (antenna, band, freq), grp in paired.groupby(["antenna", "frequency_band", "frequency_mhz"], sort=True):
        diffs = grp["active_minus_control_median_daily_z"].to_numpy(dtype=float)
        tail = grp["active_minus_control_high_tail_fraction"].to_numpy(dtype=float)
        mean_diff, lo, hi = bootstrap_mean_ci(diffs, rng, n_boot)
        tail_mean, tail_lo, tail_hi = bootstrap_mean_ci(tail, rng, n_boot)
        sign_p_two, sign_p_pos = sign_flip_p(diffs, rng, n_perm)
        label_p_two, label_p_pos = label_permutation_p(
            candidate_stats,
            antenna=str(antenna),
            frequency_band=int(band),
            rng=rng,
            n_perm=n_perm,
            min_window_samples=min_window_samples,
        )
        summary_rows.append(
            {
                "antenna": antenna,
                "antenna_label": ANTENNA_LABEL.get(str(antenna), str(antenna)),
                "frequency_band": int(band),
                "frequency_mhz": float(freq),
                "n_paired_windows": int(len(grp)),
                "active_total_samples": int(grp["n_samples"].sum()),
                "control_total_samples": int(grp["control_total_samples"].sum()),
                "mean_active_minus_control_median_z": mean_diff,
                "boot_lo_active_minus_control_median_z": lo,
                "boot_hi_active_minus_control_median_z": hi,
                "median_active_minus_control_median_z": float(np.nanmedian(diffs)) if len(diffs) else np.nan,
                "positive_window_fraction_median_z": float((diffs > 0).mean()) if len(diffs) else np.nan,
                "signflip_p_two_sided_median_z": sign_p_two,
                "signflip_p_positive_median_z": sign_p_pos,
                "label_permutation_p_two_sided_median_z": label_p_two,
                "label_permutation_p_positive_median_z": label_p_pos,
                "mean_active_minus_control_high_tail_fraction": tail_mean,
                "boot_lo_active_minus_control_high_tail_fraction": tail_lo,
                "boot_hi_active_minus_control_high_tail_fraction": tail_hi,
            }
        )
    return active, control, paired, pd.DataFrame(summary_rows)


def label_permutation_p(
    candidate_stats: pd.DataFrame,
    antenna: str,
    frequency_band: int,
    rng: np.random.Generator,
    n_perm: int,
    min_window_samples: int,
) -> tuple[float, float]:
    channel = candidate_stats[
        candidate_stats["antenna"].astype(str).eq(str(antenna))
        & candidate_stats["frequency_band"].astype(int).eq(int(frequency_band))
        & (candidate_stats["n_samples"] >= int(min_window_samples))
    ].copy()
    if channel.empty:
        return np.nan, np.nan
    groups = []
    obs_values = []
    for _, grp in channel.groupby("historical_window_id", sort=True):
        vals = grp["median_daily_z"].to_numpy(dtype=float)
        selectors = grp["selector"].astype(str).to_numpy()
        keep = np.isfinite(vals)
        vals = vals[keep]
        selectors = selectors[keep]
        active_idx = np.flatnonzero(selectors == "historical_active")
        control_idx = np.flatnonzero(selectors == "shifted_control")
        if len(active_idx) != 1 or len(control_idx) < 1:
            continue
        active_val = float(vals[active_idx[0]])
        control_val = float(np.nanmedian(vals[control_idx]))
        obs_values.append(active_val - control_val)
        groups.append(vals)
    if len(groups) < 2:
        return np.nan, np.nan
    obs = float(np.mean(obs_values))
    null = np.empty(int(n_perm), dtype=float)
    for i in range(int(n_perm)):
        vals = []
        for arr in groups:
            idx = int(rng.integers(0, len(arr)))
            rest = np.delete(arr, idx)
            vals.append(float(arr[idx] - np.nanmedian(rest)))
        null[i] = float(np.mean(vals))
    p_two = (1.0 + np.sum(np.abs(null) >= abs(obs))) / (len(null) + 1.0)
    p_pos = (1.0 + np.sum(null >= obs)) / (len(null) + 1.0)
    return float(p_two), float(p_pos)


def select_plot_windows(
    active_samples: pd.DataFrame,
    paired: pd.DataFrame,
    max_windows: int,
    plot_padding_min: float,
) -> pd.DataFrame:
    if active_samples.empty or paired.empty:
        return pd.DataFrame()
    ranked = (
        paired.sort_values(["active_minus_control_median_daily_z", "max_daily_z"], ascending=False)
        .drop_duplicates("historical_window_id")
        .head(int(max_windows))
    )
    keep_ids = ranked["historical_window_id"].astype(str).tolist()
    sub = active_samples[active_samples["historical_window_id"].astype(str).isin(keep_ids)].copy()
    starts = (
        sub.groupby("historical_window_id")["event_start_time"]
        .first()
        .map(pd.Timestamp)
        .to_dict()
    )
    ends = sub.groupby("historical_window_id")["event_end_time"].first().map(pd.Timestamp).to_dict()
    out = []
    for wid in keep_ids:
        wsub = sub[sub["historical_window_id"].astype(str).eq(wid)].copy()
        start = starts[wid]
        end = ends[wid]
        lo = start - pd.Timedelta(minutes=float(plot_padding_min))
        hi = end + pd.Timedelta(minutes=float(plot_padding_min))
        out.append(wsub[(pd.to_datetime(wsub["time"]) >= lo) & (pd.to_datetime(wsub["time"]) <= hi)])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def plot_phase_scramble(summary: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    for antenna, grp in summary.groupby("antenna", sort=True):
        grp = grp.sort_values("frequency_mhz")
        y = grp["observed_spearman_high_tail_vs_maser"].to_numpy(dtype=float)
        lo = grp["phase_scramble_null_lo"].to_numpy(dtype=float)
        hi = grp["phase_scramble_null_hi"].to_numpy(dtype=float)
        ax.fill_between(
            grp["frequency_mhz"],
            lo,
            hi,
            color=ANTENNA_COLOR.get(str(antenna), "0.5"),
            alpha=0.16,
            linewidth=0,
        )
        ax.plot(
            grp["frequency_mhz"],
            y,
            marker="o",
            lw=1.6,
            label=f"{ANTENNA_LABEL.get(str(antenna), str(antenna))} observed",
            color=ANTENNA_COLOR.get(str(antenna), "black"),
        )
    ax.axhline(0, color="0.35", lw=0.9)
    ax.set_xscale("log")
    ax.set_xticks(sorted(FREQUENCY_MAP_MHZ.values()))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_xlabel("frequency (MHz)")
    ax.set_ylabel("Spearman r: high-tail fraction vs MASER Io-CML score")
    ax.set_title("Jupiter phase-map test with Io-phase scramble null")
    ax.grid(True, color="0.9", lw=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = out_dir / "jupiter_phase_scramble_correlation_spectrum.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_score_daily(summary: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    for antenna, grp in summary.groupby("antenna", sort=True):
        grp = grp.sort_values("frequency_mhz")
        y = grp["mean_daily_high_minus_low_high_tail_fraction"].to_numpy(dtype=float)
        lo = grp["boot_lo_daily_high_minus_low_high_tail_fraction"].to_numpy(dtype=float)
        hi = grp["boot_hi_daily_high_minus_low_high_tail_fraction"].to_numpy(dtype=float)
        err = np.vstack([y - lo, hi - y])
        err[~np.isfinite(err)] = 0.0
        ax.errorbar(
            grp["frequency_mhz"],
            y,
            yerr=err,
            marker="o",
            lw=1.5,
            capsize=3,
            color=ANTENNA_COLOR.get(str(antenna), "black"),
            label=ANTENNA_LABEL.get(str(antenna), str(antenna)),
        )
    ax.axhline(0, color="0.35", lw=0.9)
    ax.set_xscale("log")
    ax.set_xticks(sorted(FREQUENCY_MAP_MHZ.values()))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_xlabel("frequency (MHz)")
    ax.set_ylabel("daily paired high-tail fraction difference\nhigh MASER-score samples - lower-score samples")
    ax.set_title("MASER Io-CML selector, paired by UTC day")
    ax.grid(True, color="0.9", lw=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = out_dir / "jupiter_maser_score_daily_paired_controls.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_historical_spectrum(summary: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    for antenna, grp in summary.groupby("antenna", sort=True):
        grp = grp.sort_values("frequency_mhz")
        y = grp["mean_active_minus_control_median_z"].to_numpy(dtype=float)
        lo = grp["boot_lo_active_minus_control_median_z"].to_numpy(dtype=float)
        hi = grp["boot_hi_active_minus_control_median_z"].to_numpy(dtype=float)
        err = np.vstack([y - lo, hi - y])
        err[~np.isfinite(err)] = 0.0
        p = grp["label_permutation_p_positive_median_z"].to_numpy(dtype=float)
        ax.errorbar(
            grp["frequency_mhz"],
            y,
            yerr=err,
            marker="o",
            lw=1.5,
            capsize=3,
            color=ANTENNA_COLOR.get(str(antenna), "black"),
            label=ANTENNA_LABEL.get(str(antenna), str(antenna)),
        )
        sig = np.isfinite(p) & (p < 0.05)
        ax.scatter(
            grp.loc[sig, "frequency_mhz"],
            grp.loc[sig, "mean_active_minus_control_median_z"],
            s=95,
            facecolors="none",
            edgecolors="black",
            linewidths=1.2,
            zorder=5,
        )
    ax.axhline(0, color="0.35", lw=0.9)
    ax.set_xscale("log")
    ax.set_xticks(sorted(FREQUENCY_MAP_MHZ.values()))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_xlabel("frequency (MHz)")
    ax.set_ylabel("active historical-window excess daily z\nactive median - median shifted controls")
    ax.set_title("Historical Jupiter windows with multi-shift matched controls")
    ax.grid(True, color="0.9", lw=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = out_dir / "jupiter_historical_multi_shift_control_spectrum.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_historical_window_heatmap(paired: pd.DataFrame, summary: pd.DataFrame, out_dir: Path) -> Path:
    if paired.empty or summary.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No paired historical windows", ha="center", va="center")
        path = out_dir / "jupiter_historical_window_channel_excess_heatmap.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path
    top_channels = summary.sort_values("mean_active_minus_control_median_z", ascending=False).head(10)
    pieces = []
    labels = []
    for _, row in top_channels.iterrows():
        sub = paired[
            paired["antenna"].astype(str).eq(str(row["antenna"]))
            & paired["frequency_band"].astype(int).eq(int(row["frequency_band"]))
        ][["historical_window_id", "active_minus_control_median_daily_z"]].copy()
        label = f"{ANTENNA_LABEL.get(str(row['antenna']), row['antenna'])} {float(row['frequency_mhz']):.2f}"
        sub = sub.rename(columns={"active_minus_control_median_daily_z": label})
        pieces.append(sub.set_index("historical_window_id"))
        labels.append(label)
    mat = pd.concat(pieces, axis=1).sort_index()
    lim = max(0.5, float(np.nanpercentile(np.abs(mat.to_numpy(dtype=float)), 95)))
    fig, ax = plt.subplots(figsize=(11.5, max(5.8, 0.18 * len(mat) + 2.0)))
    im = ax.imshow(
        mat.to_numpy(dtype=float),
        aspect="auto",
        cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim),
    )
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels(mat.index, fontsize=7)
    ax.set_xticks(np.arange(len(mat.columns)))
    ax.set_xticklabels(mat.columns, rotation=45, ha="right", fontsize=8)
    ax.set_title("Window-by-window historical excess for the strongest channels")
    ax.set_ylabel("historical window")
    cbar = fig.colorbar(im, ax=ax, pad=0.015)
    cbar.set_label("active - shifted-control median daily z")
    fig.tight_layout()
    path = out_dir / "jupiter_historical_window_channel_excess_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_dynamic_window_gallery(active_samples: pd.DataFrame, paired: pd.DataFrame, out_dir: Path, max_windows: int) -> Path:
    if active_samples.empty or paired.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No historical active samples for gallery", ha="center", va="center")
        path = out_dir / "jupiter_dynamic_window_gallery.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path
    ranked = (
        paired.sort_values(["active_minus_control_median_daily_z", "max_daily_z"], ascending=False)
        .drop_duplicates("historical_window_id")
        .head(int(max_windows))
    )
    window_ids = ranked["historical_window_id"].astype(str).tolist()
    freqs = sorted(active_samples["frequency_mhz"].dropna().unique())
    n = len(window_ids)
    fig, axes = plt.subplots(n, 2, figsize=(13.5, max(3.0 * n, 5.0)), sharex=False, sharey=True)
    axes = np.asarray(axes).reshape(n, 2)
    norm = TwoSlopeNorm(vmin=-2.5, vcenter=0.0, vmax=6.0)
    last_sc = None
    for row_i, wid in enumerate(window_ids):
        meta = ranked[ranked["historical_window_id"].astype(str).eq(wid)].iloc[0]
        for col_i, antenna in enumerate(["rv1_coarse", "rv2_coarse"]):
            ax = axes[row_i, col_i]
            sub = active_samples[
                active_samples["historical_window_id"].astype(str).eq(wid)
                & active_samples["antenna"].astype(str).eq(antenna)
            ].copy()
            if not sub.empty:
                start = pd.Timestamp(sub["event_start_time"].iloc[0])
                end = pd.Timestamp(sub["event_end_time"].iloc[0])
                sub["minute"] = (pd.to_datetime(sub["time"]) - start).dt.total_seconds() / 60.0
                last_sc = ax.scatter(
                    sub["minute"],
                    sub["frequency_mhz"],
                    c=sub["daily_z_log_power"],
                    cmap="coolwarm",
                    norm=norm,
                    s=16,
                    alpha=0.85,
                    linewidths=0,
                )
                ax.axvspan(0, (end - start).total_seconds() / 60.0, color="gold", alpha=0.14)
                ax.axvline(0, color="0.25", lw=0.8)
                ax.axvline((end - start).total_seconds() / 60.0, color="0.25", lw=0.8)
            ax.set_yscale("log")
            ax.set_yticks(freqs)
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            ax.grid(True, color="0.9", lw=0.45)
            if row_i == 0:
                ax.set_title(ANTENNA_LABEL.get(antenna, antenna))
            if col_i == 0:
                ax.set_ylabel(
                    f"{wid}\nI={int(meta['intensity'])} {meta['burstiness']}\nfrequency MHz"
                )
            ax.set_xlabel("minutes from reported start")
    fig.suptitle("Historical Jupiter windows: full-frequency sample strips")
    fig.subplots_adjust(left=0.09, right=0.875, top=0.94, bottom=0.055, hspace=0.48, wspace=0.10)
    if last_sc is not None:
        cax = fig.add_axes([0.905, 0.15, 0.018, 0.70])
        cbar = fig.colorbar(last_sc, cax=cax)
        cbar.set_label("daily-normalized log power z")
    path = out_dir / "jupiter_dynamic_window_gallery.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_dashboard(
    phase: pd.DataFrame,
    score: pd.DataFrame,
    hist: pd.DataFrame,
    paired: pd.DataFrame,
    out_dir: Path,
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 10.2))
    ax = axes[0, 0]
    for antenna, grp in phase.groupby("antenna", sort=True):
        grp = grp.sort_values("frequency_mhz")
        ax.plot(
            grp["frequency_mhz"],
            grp["observed_spearman_high_tail_vs_maser"],
            marker="o",
            color=ANTENNA_COLOR.get(str(antenna), "black"),
            label=ANTENNA_LABEL.get(str(antenna), str(antenna)),
        )
        ax.fill_between(
            grp["frequency_mhz"],
            grp["phase_scramble_null_lo"],
            grp["phase_scramble_null_hi"],
            color=ANTENNA_COLOR.get(str(antenna), "0.5"),
            alpha=0.14,
        )
    ax.axhline(0, color="0.35", lw=0.8)
    ax.set_title("Io-CML occurrence map alignment")
    ax.set_ylabel("Spearman r")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    for antenna, grp in score.groupby("antenna", sort=True):
        grp = grp.sort_values("frequency_mhz")
        ax.errorbar(
            grp["frequency_mhz"],
            grp["mean_daily_high_minus_low_high_tail_fraction"],
            yerr=np.vstack(
                [
                    grp["mean_daily_high_minus_low_high_tail_fraction"]
                    - grp["boot_lo_daily_high_minus_low_high_tail_fraction"],
                    grp["boot_hi_daily_high_minus_low_high_tail_fraction"]
                    - grp["mean_daily_high_minus_low_high_tail_fraction"],
                ]
            ),
            marker="o",
            capsize=2.5,
            color=ANTENNA_COLOR.get(str(antenna), "black"),
        )
    ax.axhline(0, color="0.35", lw=0.8)
    ax.set_title("High MASER-score daily control")
    ax.set_ylabel("high-tail fraction difference")

    ax = axes[1, 0]
    for antenna, grp in hist.groupby("antenna", sort=True):
        grp = grp.sort_values("frequency_mhz")
        ax.errorbar(
            grp["frequency_mhz"],
            grp["mean_active_minus_control_median_z"],
            yerr=np.vstack(
                [
                    grp["mean_active_minus_control_median_z"] - grp["boot_lo_active_minus_control_median_z"],
                    grp["boot_hi_active_minus_control_median_z"] - grp["mean_active_minus_control_median_z"],
                ]
            ),
            marker="o",
            capsize=2.5,
            color=ANTENNA_COLOR.get(str(antenna), "black"),
        )
    ax.axhline(0, color="0.35", lw=0.8)
    ax.set_title("Historical windows vs shifted controls")
    ax.set_ylabel("active - control median z")

    ax = axes[1, 1]
    if not paired.empty and not hist.empty:
        top = hist.sort_values("mean_active_minus_control_median_z", ascending=False).head(8)
        mats = []
        for _, row in top.iterrows():
            label = f"{ANTENNA_LABEL.get(str(row['antenna']), row['antenna'])} {float(row['frequency_mhz']):.2f}"
            sub = paired[
                paired["antenna"].astype(str).eq(str(row["antenna"]))
                & paired["frequency_band"].astype(int).eq(int(row["frequency_band"]))
            ][["historical_window_id", "active_minus_control_median_daily_z"]].rename(
                columns={"active_minus_control_median_daily_z": label}
            )
            mats.append(sub.set_index("historical_window_id"))
        mat = pd.concat(mats, axis=1).sort_index()
        lim = max(0.5, float(np.nanpercentile(np.abs(mat.to_numpy(dtype=float)), 95)))
        im = ax.imshow(
            mat.to_numpy(dtype=float),
            aspect="auto",
            cmap="coolwarm",
            norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim),
        )
        ax.set_yticks(np.arange(len(mat.index))[:: max(1, len(mat.index) // 12)])
        ax.set_yticklabels(mat.index[:: max(1, len(mat.index) // 12)], fontsize=7)
        ax.set_xticks(np.arange(len(mat.columns)))
        ax.set_xticklabels(mat.columns, rotation=45, ha="right", fontsize=8)
        ax.set_title("Window contributions")
        fig.colorbar(im, ax=ax, pad=0.01, shrink=0.85)
    else:
        ax.text(0.5, 0.5, "No paired windows", ha="center", va="center")

    for ax in axes.ravel()[:3]:
        ax.set_xscale("log")
        ax.set_xticks(sorted(FREQUENCY_MAP_MHZ.values()))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.set_xlabel("frequency (MHz)")
        ax.grid(True, color="0.9", lw=0.45)
    fig.suptitle("RAE-2 Jupiter literature-guided control dashboard")
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    path = out_dir / "jupiter_literature_control_dashboard.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _format_table(df: pd.DataFrame, columns: list[str], n: int = 12) -> str:
    if df.empty:
        return "(none)"
    return df[columns].head(int(n)).to_string(index=False)


def write_report(
    out_dir: Path,
    paths: list[Path],
    config: dict[str, object],
    n_samples: int,
    phase: pd.DataFrame,
    score: pd.DataFrame,
    hist: pd.DataFrame,
) -> Path:
    phase_cols = [
        "antenna_label",
        "frequency_mhz",
        "observed_spearman_high_tail_vs_maser",
        "phase_scramble_null_lo",
        "phase_scramble_null_hi",
        "phase_scramble_p_two_sided",
    ]
    score_cols = [
        "antenna_label",
        "frequency_mhz",
        "n_paired_days",
        "mean_daily_high_minus_low_high_tail_fraction",
        "boot_lo_daily_high_minus_low_high_tail_fraction",
        "boot_hi_daily_high_minus_low_high_tail_fraction",
        "signflip_p_positive_high_tail",
    ]
    hist_cols = [
        "antenna_label",
        "frequency_mhz",
        "n_paired_windows",
        "mean_active_minus_control_median_z",
        "boot_lo_active_minus_control_median_z",
        "boot_hi_active_minus_control_median_z",
        "positive_window_fraction_median_z",
        "label_permutation_p_positive_median_z",
    ]
    phase_rank = phase.copy()
    phase_rank["abs_observed"] = phase_rank["observed_spearman_high_tail_vs_maser"].abs()
    score_rank = score.sort_values("mean_daily_high_minus_low_high_tail_fraction", ascending=False)
    hist_rank = hist.sort_values("mean_active_minus_control_median_z", ascending=False)

    lines = [
        "# Jupiter Literature-Guided Controls",
        "",
        "This run treats Jupiter as a bursty, geometry-dependent radio source rather than a steady occultation source.",
        "The tests are designed to be falsifiable: the same statistic should weaken under phase scrambles or shifted-time controls.",
        "",
        "## Literature Basis Used",
        "",
        "- Jupiter low-frequency emission is strongly anisotropic and often appears as time-frequency arcs, so dynamic spectra and visibility geometry matter: https://www.frontiersin.org/journals/astronomy-and-space-sciences/articles/10.3389/fspas.2023.1091967/full",
        "- Io-controlled decametric emission is commonly analyzed in System-III CML versus Io-phase occurrence maps: https://space.physics.uiowa.edu/~dag/publications/2008_AngluarBeamingModelOfJupitersDecametricRadioEmissionsBasedOnCassiniRPWSDataAnalysis_GRL.pdf",
        "- RAE-2 overlaps the low-MHz HOM / low-DAM regime, where Io-independent and solar-wind-controlled variability can matter: https://space.physics.uiowa.edu/~dag/publications/2002_ControlOfJupitersRadioEmissionAndAuroraeByTheSolarWind_N.pdf",
        "- The historical windows are from Warwick, Dulk, and Riddle's 7.6-80 MHz catalog and are used only as external time selectors: https://repository.library.noaa.gov/view/noaa/1033",
        "",
        "## Configuration",
        "",
        *[f"- `{k}`: `{v}`" for k, v in config.items() if k != "software_versions"],
        "",
        f"Full merged RAE-2 sample count used for time-domain controls: `{n_samples}`.",
        "",
        "## Tests Added",
        "",
        "1. `Io-CML phase-scramble null`: correlates RAE-2 high-tail fraction with the MASER Io-controlled probability map, then circularly scrambles Io phase within CML columns.",
        "2. `Daily paired MASER-score control`: compares high-MASER-score samples with lower-score samples within the same UTC day, antenna, and frequency.",
        "3. `Historical multi-shift control`: compares Warwick/Dulk/Riddle windows with same-duration windows shifted by the configured day offsets, with known historical-window overlaps removed from controls.",
        "4. `Dynamic window gallery`: plots full-frequency strips for the strongest historical-window leads so individual events can be inspected visually.",
        "",
        "## Strongest Io-CML Phase-Map Alignments",
        "",
        _format_table(phase_rank.sort_values("abs_observed", ascending=False), phase_cols),
        "",
        "## Strongest Daily MASER-Score Excesses",
        "",
        _format_table(score_rank, score_cols),
        "",
        "## Strongest Historical-Window Excesses",
        "",
        _format_table(hist_rank, hist_cols),
        "",
        "## Reading The Plots",
        "",
        "- A credible Jupiter detection should be coherent across adjacent frequencies or repeat across several windows, not appear as a single isolated channel.",
        "- A positive historical-window excess is more compelling when the bootstrap interval stays above zero and the label-permutation p-value is small.",
        "- A phase-map correlation is more compelling when it lies outside the Io-phase scramble envelope and does not have comparable control behavior.",
        "- The dynamic gallery should show clustered broadband or drifting time-frequency structure inside the reported windows; isolated single-frequency points are weak evidence.",
        "",
        "## Output Files",
        "",
        *[f"- `{p}`" for p in paths],
    ]
    path = out_dir / "jupiter_literature_controls_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", type=Path, default=DEFAULT_CLEAN)
    parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--phase-summary", type=Path, default=DEFAULT_PHASE_SUMMARY)
    parser.add_argument("--historical-windows", type=Path, default=DEFAULT_WINDOWS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--geometry-tolerance-s", type=float, default=360.0)
    parser.add_argument("--high-z", type=float, default=2.5)
    parser.add_argument("--score-high-quantile", type=float, default=0.90)
    parser.add_argument("--min-daily-score-samples", type=int, default=20)
    parser.add_argument("--padding-min", type=float, default=90.0)
    parser.add_argument("--control-shift-days", type=int, nargs="*", default=[-28, -21, -14, -7, 7, 14, 21, 28])
    parser.add_argument("--min-window-samples", type=int, default=5)
    parser.add_argument("--min-control-shifts", type=int, default=3)
    parser.add_argument("--phase-null-permutations", type=int, default=2000)
    parser.add_argument("--bootstrap-samples", type=int, default=3000)
    parser.add_argument("--permutations", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--plot-window-count", type=int, default=8)
    parser.add_argument("--exclude-control-overlaps", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    rng = np.random.default_rng(int(args.seed))
    config = {
        "clean": str(args.clean),
        "geometry": str(args.geometry),
        "phase_summary": str(args.phase_summary),
        "historical_windows": str(args.historical_windows),
        "geometry_tolerance_s": float(args.geometry_tolerance_s),
        "high_z": float(args.high_z),
        "score_high_quantile": float(args.score_high_quantile),
        "min_daily_score_samples": int(args.min_daily_score_samples),
        "padding_min": float(args.padding_min),
        "control_shift_days": [int(v) for v in args.control_shift_days],
        "min_window_samples": int(args.min_window_samples),
        "min_control_shifts": int(args.min_control_shifts),
        "phase_null_permutations": int(args.phase_null_permutations),
        "bootstrap_samples": int(args.bootstrap_samples),
        "permutations": int(args.permutations),
        "seed": int(args.seed),
        "exclude_control_overlaps": bool(args.exclude_control_overlaps),
        "software_versions": software_versions(),
    }
    write_json(out_dir / "run_config.json", config)

    phase_summary = read_table(args.phase_summary, low_memory=False)
    phase = phase_scramble_summary(
        phase_summary,
        rng=rng,
        n_perm=int(args.phase_null_permutations),
    )
    phase.to_csv(out_dir / "jupiter_phase_scramble_correlation_summary.csv", index=False)

    samples = load_full_samples(args.clean, args.geometry, tolerance_s=float(args.geometry_tolerance_s))
    score, score_daily = score_daily_controls(
        samples,
        high_z=float(args.high_z),
        score_quantile=float(args.score_high_quantile),
        min_daily_samples=int(args.min_daily_score_samples),
        rng=rng,
        n_boot=int(args.bootstrap_samples),
        n_perm=int(args.permutations),
    )
    score.to_csv(out_dir / "jupiter_maser_score_daily_control_summary.csv", index=False)
    score_daily.to_csv(out_dir / "jupiter_maser_score_daily_control_points.csv", index=False)

    windows = load_windows(args.historical_windows, padding_min=float(args.padding_min))
    active, control, paired, hist = build_historical_matched_tables(
        samples,
        windows=windows,
        shift_days=[int(v) for v in args.control_shift_days],
        high_z=float(args.high_z),
        min_window_samples=int(args.min_window_samples),
        min_control_shifts=int(args.min_control_shifts),
        rng=rng,
        n_boot=int(args.bootstrap_samples),
        n_perm=int(args.permutations),
        exclude_control_overlaps=bool(args.exclude_control_overlaps),
    )
    active.to_csv(out_dir / "jupiter_historical_active_full_samples.csv", index=False)
    control.to_csv(out_dir / "jupiter_historical_shifted_control_full_samples.csv", index=False)
    paired.to_csv(out_dir / "jupiter_historical_multi_shift_paired_window_stats.csv", index=False)
    hist.to_csv(out_dir / "jupiter_historical_multi_shift_control_summary.csv", index=False)

    paths: list[Path] = []
    paths.append(plot_phase_scramble(phase, out_dir))
    paths.append(plot_score_daily(score, out_dir))
    paths.append(plot_historical_spectrum(hist, out_dir))
    paths.append(plot_historical_window_heatmap(paired, hist, out_dir))
    active_for_gallery = select_interval_samples(
        samples,
        make_intervals(windows, start_col="expanded_start_time", end_col="expanded_end_time"),
        "historical_active_padded",
    )
    paths.append(
        plot_dynamic_window_gallery(active_for_gallery, paired, out_dir, max_windows=int(args.plot_window_count))
    )
    paths.append(plot_dashboard(phase, score, hist, paired, out_dir))

    report = write_report(out_dir, paths, config, len(samples), phase, score, hist)
    print(report)


if __name__ == "__main__":
    main()
