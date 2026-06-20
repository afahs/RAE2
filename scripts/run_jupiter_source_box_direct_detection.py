#!/usr/bin/env python
"""Direct configured-band Jupiter search in standard CML/Io source boxes.

This analysis tests whether configured Ryle-Vonberg bands are enhanced when
Jupiter is in the standard Io-controlled DAM source boxes, and compares that
result with time-shifted phase controls that preserve the RAE-2 observing
times.
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
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.constants import FREQUENCY_MAP_MHZ  # noqa: E402
from rylevonberg.util import ensure_dir, software_versions, write_json  # noqa: E402


DEFAULT_CLEAN_NPY = ROOT / "outputs/control_survey_earth_sun_postnov1974_v2/01_ingest/cleaned_timeseries.npy"
DEFAULT_GEOMETRY = ROOT / "outputs/jupiter_phase_pattern_survey_v1/jupiter_spice_visibility_geometry_grid.csv"
DEFAULT_OUT = ROOT / "outputs/jupiter_source_box_direct_detection_v1"

ANTENNAS = ["rv1_coarse", "rv2_coarse"]
ANTENNA_LABEL = {"rv1_coarse": "upper V", "rv2_coarse": "lower V"}
ANTENNA_COLOR = {"rv1_coarse": "#386cb0", "rv2_coarse": "#bf5b17"}
FREQ_COLOR = {3.93: "#1b9e77", 4.70: "#d95f02", 6.55: "#7570b3", 9.18: "#e7298a"}


@dataclass(frozen=True)
class SourceBox:
    name: str
    cml_lo: float
    cml_hi: float
    io_lo: float
    io_hi: float
    max_frequency_mhz: float
    polarization: str


# Standard Io-controlled DAM boxes, expressed in System-III CML and Io phase
# measured from superior geocentric conjunction.  The ranges are intentionally
# written as source definitions, not as fitted outcomes from these data.
SOURCE_BOXES = [
    SourceBox("Io-A", 200.0, 270.0, 205.0, 260.0, 38.0, "RH"),
    SourceBox("Io-B", 105.0, 185.0, 80.0, 110.0, 39.5, "RH"),
    SourceBox("Io-C", 300.0, 20.0, 225.0, 260.0, 36.0, "RH/LH"),
    SourceBox("Io-D", 0.0, 200.0, 95.0, 130.0, 18.0, "LH"),
]


def circular_width(lo: float, hi: float) -> float:
    lo = float(lo) % 360.0
    hi = float(hi) % 360.0
    return (hi - lo) % 360.0


def circular_between(values: pd.Series | np.ndarray, lo: float, hi: float) -> np.ndarray:
    vals = np.asarray(values, dtype=float) % 360.0
    lo = float(lo) % 360.0
    hi = float(hi) % 360.0
    if np.isclose(lo, hi):
        return np.ones(len(vals), dtype=bool)
    if lo < hi:
        return (vals >= lo) & (vals <= hi)
    return (vals >= lo) | (vals <= hi)


def padded_interval(lo: float, hi: float, pad_deg: float) -> tuple[float, float] | None:
    width = circular_width(lo, hi)
    if width + 2.0 * float(pad_deg) >= 360.0:
        return None
    return (float(lo) - float(pad_deg)) % 360.0, (float(hi) + float(pad_deg)) % 360.0


def source_box_definitions(pad_deg: float) -> pd.DataFrame:
    rows = []
    for box in SOURCE_BOXES:
        cml = padded_interval(box.cml_lo, box.cml_hi, pad_deg)
        io = padded_interval(box.io_lo, box.io_hi, pad_deg)
        rows.append(
            {
                "source_box": box.name,
                "cml_lo_deg": box.cml_lo,
                "cml_hi_deg": box.cml_hi,
                "io_phase_lo_deg": box.io_lo,
                "io_phase_hi_deg": box.io_hi,
                "analysis_cml_lo_deg": np.nan if cml is None else cml[0],
                "analysis_cml_hi_deg": np.nan if cml is None else cml[1],
                "analysis_io_phase_lo_deg": np.nan if io is None else io[0],
                "analysis_io_phase_hi_deg": np.nan if io is None else io[1],
                "padding_deg": float(pad_deg),
                "max_frequency_mhz": box.max_frequency_mhz,
                "dominant_polarization": box.polarization,
            }
        )
    return pd.DataFrame(rows)


def annotate_source_boxes(df: pd.DataFrame, pad_deg: float) -> pd.DataFrame:
    out = df.copy()
    cml = pd.to_numeric(out["jupiter_cml_spice_deg"], errors="coerce")
    io = pd.to_numeric(out["io_phase_spice_deg"], errors="coerce")
    any_mask = np.zeros(len(out), dtype=bool)
    names = np.full(len(out), "", dtype=object)
    for box in SOURCE_BOXES:
        cml_interval = padded_interval(box.cml_lo, box.cml_hi, pad_deg)
        io_interval = padded_interval(box.io_lo, box.io_hi, pad_deg)
        if cml_interval is None:
            cml_mask = np.ones(len(out), dtype=bool)
        else:
            cml_mask = circular_between(cml, cml_interval[0], cml_interval[1])
        if io_interval is None:
            io_mask = np.ones(len(out), dtype=bool)
        else:
            io_mask = circular_between(io, io_interval[0], io_interval[1])
        mask = cml_mask & io_mask
        col = f"source_{box.name.lower().replace('-', '_')}"
        out[col] = mask
        any_mask |= mask
        empty = names == ""
        names[mask & empty] = box.name
        names[mask & ~empty] = names[mask & ~empty] + "+" + box.name
    out["source_box_any"] = any_mask
    out["source_box_names"] = names
    return out


def read_clean_npy_subset(path: Path, frequency_bands: list[int]) -> pd.DataFrame:
    arr = np.load(path, allow_pickle=True)
    keep = np.isin(arr["frequency_band"], np.asarray(frequency_bands, dtype=int))
    keep &= np.isin(arr["antenna"], np.asarray(ANTENNAS, dtype=object))
    keep &= arr["is_valid"].astype(bool)
    power = arr["power"].astype(float)
    keep &= np.isfinite(power) & (power > 0.0)
    sub = arr[keep]
    out = pd.DataFrame(
        {
            "time": pd.to_datetime(sub["time"].astype(str)),
            "frequency_band": sub["frequency_band"].astype(int),
            "frequency_mhz": sub["frequency_mhz"].astype(float),
            "antenna": sub["antenna"].astype(str),
            "power": sub["power"].astype(float),
        }
    )
    out["log10_power"] = np.log10(out["power"].to_numpy(dtype=float))
    out["date"] = out["time"].dt.floor("D")
    stats = (
        out.groupby(["date", "antenna", "frequency_band"], sort=True)["log10_power"]
        .median()
        .rename("daily_median_log10_power")
        .reset_index()
    )
    out = out.merge(stats, on=["date", "antenna", "frequency_band"], how="left")
    out["daily_log10_residual"] = out["log10_power"] - out["daily_median_log10_power"]
    return out.sort_values("time").reset_index(drop=True)


def add_local_residual(samples: pd.DataFrame, window: str, min_periods: int) -> pd.DataFrame:
    pieces = []
    for _, grp in samples.groupby(["antenna", "frequency_band"], sort=False):
        work = grp.sort_values("time").copy()
        roll = work.set_index("time")["log10_power"].rolling(window=window, center=True, min_periods=int(min_periods))
        local_median = roll.median().to_numpy(dtype=float)
        work["local_median_log10_power"] = local_median
        fallback = float(work["log10_power"].median())
        work.loc[~np.isfinite(work["local_median_log10_power"]), "local_median_log10_power"] = fallback
        work["local20min_log10_residual"] = work["log10_power"] - work["local_median_log10_power"]
        pieces.append(work)
    return pd.concat(pieces, ignore_index=True).sort_values("time").reset_index(drop=True)


def load_geometry(path: Path, pad_deg: float) -> pd.DataFrame:
    cols = [
        "time",
        "jupiter_visible_by_moon",
        "earth_visible_by_moon",
        "jupiter_limb_angle_deg",
        "earth_limb_angle_deg",
        "jupiter_cml_spice_deg",
        "io_phase_spice_deg",
        "maser_zarka_io_score",
    ]
    geom = pd.read_csv(path, usecols=cols, parse_dates=["time"], low_memory=False)
    geom = geom.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    for col in ["jupiter_visible_by_moon", "earth_visible_by_moon"]:
        geom[col] = geom[col].astype(bool)
    return annotate_source_boxes(geom, pad_deg=pad_deg)


def merge_geometry(samples: pd.DataFrame, geom: pd.DataFrame, tolerance_s: float, pad_deg: float) -> pd.DataFrame:
    keep_cols = [
        "time",
        "jupiter_visible_by_moon",
        "earth_visible_by_moon",
        "jupiter_limb_angle_deg",
        "earth_limb_angle_deg",
        "jupiter_cml_spice_deg",
        "io_phase_spice_deg",
        "maser_zarka_io_score",
    ]
    merged = pd.merge_asof(
        samples.sort_values("time"),
        geom[keep_cols].sort_values("time"),
        on="time",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=float(tolerance_s)),
    )
    merged = merged.dropna(subset=["jupiter_cml_spice_deg", "io_phase_spice_deg"]).copy()
    for col in ["jupiter_visible_by_moon", "earth_visible_by_moon"]:
        merged[col] = merged[col].astype(bool)
    return annotate_source_boxes(merged, pad_deg=pad_deg).reset_index(drop=True)


def contiguous_windows(geom: pd.DataFrame, mask: pd.Series | np.ndarray, max_gap_min: float) -> pd.DataFrame:
    selected = geom[np.asarray(mask, dtype=bool)].sort_values("time").copy()
    if selected.empty:
        return pd.DataFrame()
    group = selected["time"].diff().dt.total_seconds().div(60).gt(float(max_gap_min)).fillna(True).cumsum()
    rows = []
    source_cols = [c for c in selected.columns if c.startswith("source_io_")]
    for idx, grp in selected.groupby(group, sort=True):
        names = []
        for col in source_cols:
            if grp[col].astype(bool).any():
                names.append(col.replace("source_", "").replace("_", "-").title())
        rows.append(
            {
                "window_id": f"src{int(idx):04d}",
                "start_time": grp["time"].min(),
                "end_time": grp["time"].max(),
                "duration_min": (grp["time"].max() - grp["time"].min()).total_seconds() / 60.0,
                "source_boxes": "+".join(names),
                "median_cml_deg": float(grp["jupiter_cml_spice_deg"].median()),
                "median_io_phase_deg": float(grp["io_phase_spice_deg"].median()),
                "median_maser_zarka_io_score": float(grp["maser_zarka_io_score"].median()),
                "n_geometry_points": int(len(grp)),
            }
        )
    return pd.DataFrame(rows)


def mask_times_in_windows(times: pd.Series, windows: pd.DataFrame) -> np.ndarray:
    if windows.empty:
        return np.zeros(len(times), dtype=bool)
    ordered = windows.sort_values("start_time")
    starts = pd.to_datetime(ordered["start_time"]).to_numpy(dtype="datetime64[ns]").astype("int64")
    ends = pd.to_datetime(ordered["end_time"]).to_numpy(dtype="datetime64[ns]").astype("int64")
    vals = pd.to_datetime(times).to_numpy(dtype="datetime64[ns]").astype("int64")
    idx = np.searchsorted(starts, vals, side="right") - 1
    valid = idx >= 0
    out = np.zeros(len(vals), dtype=bool)
    out[valid] = vals[valid] <= ends[idx[valid]]
    return out


def shifted_window_mask(times: pd.Series, source_windows: pd.DataFrame, shift_days: float) -> np.ndarray:
    shifted = source_windows[["start_time", "end_time"]].copy()
    delta = pd.Timedelta(days=float(shift_days))
    shifted["start_time"] = pd.to_datetime(shifted["start_time"]) - delta
    shifted["end_time"] = pd.to_datetime(shifted["end_time"]) - delta
    return mask_times_in_windows(times, shifted)


def build_alignment_events(
    geom: pd.DataFrame,
    source_windows: pd.DataFrame,
    shift_days: list[float],
    sample_start: pd.Timestamp,
    sample_end: pd.Timestamp,
) -> pd.DataFrame:
    """Return real and shifted alignment times for source-box intervals."""
    rows = []
    if source_windows.empty:
        return pd.DataFrame()
    geom = geom.sort_values("time").reset_index(drop=True)
    for _, win in source_windows.iterrows():
        start = pd.Timestamp(win["start_time"])
        end = pd.Timestamp(win["end_time"])
        sub = geom[
            (geom["time"] >= start)
            & (geom["time"] <= end)
            & geom["jupiter_visible_by_moon"].astype(bool)
            & geom["source_box_any"].astype(bool)
        ].copy()
        if sub.empty:
            peak_time = start + (end - start) / 2
            peak_score = np.nan
        else:
            peak_idx = pd.to_numeric(sub["maser_zarka_io_score"], errors="coerce").idxmax()
            peak_time = pd.Timestamp(sub.loc[peak_idx, "time"])
            peak_score = float(sub.loc[peak_idx, "maser_zarka_io_score"])
        base_events = [
            ("entry", start),
            ("exit", end),
            ("maser_peak", peak_time),
        ]
        for align_type, event_time in base_events:
            rows.append(
                {
                    "alignment_event_id": f"{win['window_id']}_{align_type}_real",
                    "window_id": win["window_id"],
                    "role": "real",
                    "shift_days": 0.0,
                    "align_type": align_type,
                    "event_time": event_time,
                    "source_boxes": win.get("source_boxes", ""),
                    "window_start_time": start,
                    "window_end_time": end,
                    "duration_min": float(win.get("duration_min", np.nan)),
                    "median_cml_deg": float(win.get("median_cml_deg", np.nan)),
                    "median_io_phase_deg": float(win.get("median_io_phase_deg", np.nan)),
                    "peak_maser_zarka_io_score": peak_score,
                }
            )
            for shift in shift_days:
                shifted_time = event_time + pd.Timedelta(days=float(shift))
                if shifted_time < sample_start or shifted_time > sample_end:
                    continue
                rows.append(
                    {
                        "alignment_event_id": f"{win['window_id']}_{align_type}_shift_{shift:+g}d",
                        "window_id": win["window_id"],
                        "role": "shifted_control",
                        "shift_days": float(shift),
                        "align_type": align_type,
                        "event_time": shifted_time,
                        "source_boxes": win.get("source_boxes", ""),
                        "window_start_time": start,
                        "window_end_time": end,
                        "duration_min": float(win.get("duration_min", np.nan)),
                        "median_cml_deg": float(win.get("median_cml_deg", np.nan)),
                        "median_io_phase_deg": float(win.get("median_io_phase_deg", np.nan)),
                        "peak_maser_zarka_io_score": peak_score,
                    }
                )
    return pd.DataFrame(rows)


def aligned_event_bin_values(
    samples: pd.DataFrame,
    events: pd.DataFrame,
    metric: str,
    window_minutes: float,
    bin_minutes: float,
) -> pd.DataFrame:
    """Median metric per event/channel/time-bin, so each event carries equal weight."""
    if events.empty:
        return pd.DataFrame()
    rows = []
    event_times = pd.to_datetime(events["event_time"]).to_numpy(dtype="datetime64[ns]").astype("int64")
    half_window_ns = int(float(window_minutes) * 60.0 * 1e9)
    bin_minutes = float(bin_minutes)
    event_records = events.reset_index(drop=True).to_dict("records")
    for (antenna, band, freq), grp in samples.groupby(["antenna", "frequency_band", "frequency_mhz"], sort=True):
        work = grp.sort_values("time")
        time_ns = work["time"].to_numpy(dtype="datetime64[ns]").astype("int64")
        values = pd.to_numeric(work[metric], errors="coerce").to_numpy(dtype=float)
        for event_i, ev_ns in enumerate(event_times):
            lo = int(np.searchsorted(time_ns, ev_ns - half_window_ns, side="left"))
            hi = int(np.searchsorted(time_ns, ev_ns + half_window_ns, side="right"))
            if hi <= lo:
                continue
            rel_min = (time_ns[lo:hi].astype(float) - float(ev_ns)) / 1e9 / 60.0
            vals = values[lo:hi]
            finite = np.isfinite(vals) & np.isfinite(rel_min)
            if not finite.any():
                continue
            rel_min = rel_min[finite]
            vals = vals[finite]
            t_bins = np.floor(rel_min / bin_minutes) * bin_minutes + 0.5 * bin_minutes
            for t_bin in np.unique(t_bins):
                bmask = t_bins == t_bin
                record = event_records[event_i]
                rows.append(
                    {
                        "alignment_event_id": record["alignment_event_id"],
                        "window_id": record["window_id"],
                        "role": record["role"],
                        "shift_days": record["shift_days"],
                        "align_type": record["align_type"],
                        "source_boxes": record["source_boxes"],
                        "antenna": antenna,
                        "antenna_label": ANTENNA_LABEL.get(str(antenna), str(antenna)),
                        "frequency_band": int(band),
                        "frequency_mhz": float(freq),
                        "t_bin_min": float(t_bin),
                        "value": float(np.nanmedian(vals[bmask])),
                        "n_samples_in_event_bin": int(np.count_nonzero(bmask)),
                    }
                )
    return pd.DataFrame(rows)


def summarize_aligned_profiles(bin_values: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if bin_values.empty:
        return pd.DataFrame(), pd.DataFrame()
    keys = ["align_type", "antenna", "antenna_label", "frequency_band", "frequency_mhz", "t_bin_min"]
    real = bin_values[bin_values["role"].eq("real")].copy()
    real_profile = (
        real.groupby(keys, sort=True)["value"]
        .agg(
            real_median="median",
            real_q25=lambda x: float(np.nanquantile(x, 0.25)),
            real_q75=lambda x: float(np.nanquantile(x, 0.75)),
            n_real_event_bins="size",
        )
        .reset_index()
    )
    control = bin_values[bin_values["role"].eq("shifted_control")].copy()
    if control.empty:
        return real_profile, pd.DataFrame()
    shift_profile = (
        control.groupby(["shift_days", *keys], sort=True)["value"]
        .median()
        .rename("shift_median")
        .reset_index()
    )
    control_profile = (
        shift_profile.groupby(keys, sort=True)["shift_median"]
        .agg(
            control_median="median",
            control_q10=lambda x: float(np.nanquantile(x, 0.10)),
            control_q90=lambda x: float(np.nanquantile(x, 0.90)),
            n_control_shifts="size",
        )
        .reset_index()
    )
    return real_profile, control_profile


def summarize_aligned_high_fraction(bin_values: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if bin_values.empty:
        return pd.DataFrame(), pd.DataFrame()
    work = bin_values.copy()
    work["is_high"] = pd.to_numeric(work["value"], errors="coerce") >= float(threshold)
    keys = ["align_type", "antenna", "antenna_label", "frequency_band", "frequency_mhz", "t_bin_min"]
    real = work[work["role"].eq("real")].copy()
    real_profile = (
        real.groupby(keys, sort=True)["is_high"]
        .agg(real_high_fraction="mean", n_real_event_bins="size")
        .reset_index()
    )
    control = work[work["role"].eq("shifted_control")].copy()
    if control.empty:
        return real_profile, pd.DataFrame()
    shift_profile = (
        control.groupby(["shift_days", *keys], sort=True)["is_high"]
        .mean()
        .rename("shift_high_fraction")
        .reset_index()
    )
    control_profile = (
        shift_profile.groupby(keys, sort=True)["shift_high_fraction"]
        .agg(
            control_high_fraction_median="median",
            control_high_fraction_q10=lambda x: float(np.nanquantile(x, 0.10)),
            control_high_fraction_q90=lambda x: float(np.nanquantile(x, 0.90)),
            n_control_shifts="size",
        )
        .reset_index()
    )
    return real_profile, control_profile


def summarize_channel_contrast(
    samples: pd.DataFrame,
    source_mask: np.ndarray,
    label: str,
    high_resid_threshold: float,
) -> pd.DataFrame:
    source_mask = np.asarray(source_mask, dtype=bool)
    visible = samples["jupiter_visible_by_moon"].to_numpy(dtype=bool)
    rows = []
    for (antenna, band, freq), grp in samples.groupby(["antenna", "frequency_band", "frequency_mhz"], sort=True):
        idx = grp.index.to_numpy(dtype=int)
        source = source_mask[idx] & visible[idx]
        control = (~source_mask[idx]) & visible[idx]
        source_values = grp.loc[source, "daily_log10_residual"]
        control_values = grp.loc[control, "daily_log10_residual"]
        source_raw = grp.loc[source, "log10_power"]
        control_raw = grp.loc[control, "log10_power"]
        rows.append(
            {
                "contrast_label": label,
                "antenna": antenna,
                "antenna_label": ANTENNA_LABEL.get(str(antenna), str(antenna)),
                "frequency_band": int(band),
                "frequency_mhz": float(freq),
                "n_source": int(source.sum()),
                "n_visible_control": int(control.sum()),
                "source_median_raw_log10_power": float(source_raw.median()) if len(source_raw) else np.nan,
                "control_median_raw_log10_power": float(control_raw.median()) if len(control_raw) else np.nan,
                "source_median_daily_resid": float(source_values.median()) if len(source_values) else np.nan,
                "control_median_daily_resid": float(control_values.median()) if len(control_values) else np.nan,
                "source_minus_control_median_daily_resid": (
                    float(source_values.median() - control_values.median()) if len(source_values) and len(control_values) else np.nan
                ),
                "source_factor_high_fraction": float(np.mean(source_values >= high_resid_threshold)) if len(source_values) else np.nan,
                "control_factor_high_fraction": float(np.mean(control_values >= high_resid_threshold)) if len(control_values) else np.nan,
                "source_minus_control_factor_high_fraction": (
                    float(np.mean(source_values >= high_resid_threshold) - np.mean(control_values >= high_resid_threshold))
                    if len(source_values) and len(control_values)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def build_shift_control_summary(
    samples: pd.DataFrame,
    source_windows: pd.DataFrame,
    shift_days: list[float],
    high_resid_threshold: float,
) -> pd.DataFrame:
    rows = []
    for shift in shift_days:
        mask = shifted_window_mask(samples["time"], source_windows, shift_days=float(shift))
        summary = summarize_channel_contrast(
            samples,
            source_mask=mask,
            label=f"shift_{shift:+g}d",
            high_resid_threshold=high_resid_threshold,
        )
        summary["shift_days"] = float(shift)
        rows.append(summary)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def add_shift_percentiles(real: pd.DataFrame, shifts: pd.DataFrame) -> pd.DataFrame:
    out = real.copy()
    rows = []
    metrics = ["source_minus_control_median_daily_resid", "source_minus_control_factor_high_fraction"]
    for _, row in out.iterrows():
        match = shifts[
            shifts["antenna"].eq(row["antenna"])
            & shifts["frequency_band"].eq(row["frequency_band"])
            & shifts["frequency_mhz"].eq(row["frequency_mhz"])
        ]
        add = row.to_dict()
        for metric in metrics:
            vals = match[metric].dropna().to_numpy(dtype=float)
            real_val = float(row[metric])
            if len(vals) == 0 or not np.isfinite(real_val):
                add[f"{metric}_shift_percentile"] = np.nan
                add[f"{metric}_shift_two_sided_tail"] = np.nan
                continue
            percentile = float((np.count_nonzero(vals <= real_val) + 1.0) / (len(vals) + 1.0))
            add[f"{metric}_shift_percentile"] = percentile
            add[f"{metric}_shift_two_sided_tail"] = float(min(1.0, 2.0 * min(percentile, 1.0 - percentile)))
        rows.append(add)
    return pd.DataFrame(rows)


def plot_source_box_phase_maps(samples: pd.DataFrame, out_dir: Path, high_resid_threshold: float) -> list[Path]:
    visible = samples[samples["jupiter_visible_by_moon"].astype(bool)].copy()
    visible["cml_bin"] = np.floor(visible["jupiter_cml_spice_deg"] / 10.0) * 10.0 + 5.0
    visible["io_bin"] = np.floor(visible["io_phase_spice_deg"] / 10.0) * 10.0 + 5.0
    visible["factor_high"] = visible["daily_log10_residual"] >= float(high_resid_threshold)
    paths = []
    for value_col, title, filename, cmap, centered in [
        (
            "median_daily_resid",
            "median daily log10 residual",
            "jupiter_source_box_cml_io_median_daily_residual.png",
            "coolwarm",
            True,
        ),
        (
            "factor_high_fraction",
            f"fraction at least {10 ** high_resid_threshold:.1f}x same-day median",
            "jupiter_source_box_cml_io_factor_high_fraction.png",
            "magma",
            False,
        ),
    ]:
        summary = (
            visible.groupby(["antenna", "frequency_mhz", "cml_bin", "io_bin"], sort=True)
            .agg(
                median_daily_resid=("daily_log10_residual", "median"),
                factor_high_fraction=("factor_high", "mean"),
                n_samples=("factor_high", "size"),
            )
            .reset_index()
        )
        freqs = sorted(summary["frequency_mhz"].dropna().unique())
        fig, axes = plt.subplots(2, len(freqs), figsize=(3.5 * len(freqs), 7.0), sharex=True, sharey=True)
        axes = np.asarray(axes)
        im = None
        for row, antenna in enumerate(ANTENNAS):
            for col, freq in enumerate(freqs):
                ax = axes[row, col]
                sub = summary[summary["antenna"].eq(antenna) & np.isclose(summary["frequency_mhz"], float(freq))]
                mat = sub.pivot_table(index="io_bin", columns="cml_bin", values=value_col, aggfunc="mean")
                mat = mat.reindex(index=np.arange(5.0, 360.0, 10.0), columns=np.arange(5.0, 360.0, 10.0))
                vals = mat.to_numpy(dtype=float)
                finite = vals[np.isfinite(vals)]
                if centered:
                    vmax = max(0.02, float(np.nanpercentile(np.abs(finite), 98))) if finite.size else 0.05
                    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
                else:
                    vmax = max(0.005, float(np.nanpercentile(finite, 98))) if finite.size else 0.02
                    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=vmax)
                im = ax.imshow(
                    vals,
                    origin="lower",
                    extent=[0, 360, 0, 360],
                    aspect="auto",
                    interpolation="nearest",
                    cmap=cmap,
                    norm=norm,
                )
                draw_source_boxes(ax)
                ax.set_title(f"{ANTENNA_LABEL[antenna]} {freq:.2f} MHz", fontsize=9)
                ax.set_xlim(0, 360)
                ax.set_ylim(0, 360)
                ax.grid(True, color="white", alpha=0.15, lw=0.4)
                if col == 0:
                    ax.set_ylabel("Io phase (deg)")
                if row == 1:
                    ax.set_xlabel("System III CML (deg)")
        fig.suptitle(f"Jupiter-visible configured-band samples in CML/Io source boxes: {title}")
        fig.subplots_adjust(left=0.06, right=0.88, bottom=0.08, top=0.90, wspace=0.09, hspace=0.25)
        if im is not None:
            cax = fig.add_axes([0.905, 0.18, 0.018, 0.64])
            fig.colorbar(im, cax=cax, label=title)
        path = out_dir / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def draw_source_boxes(ax: plt.Axes) -> None:
    colors = {"Io-A": "#1b9e77", "Io-B": "#d95f02", "Io-C": "#7570b3", "Io-D": "#e7298a"}
    for box in SOURCE_BOXES:
        color = colors.get(box.name, "black")
        cml_parts = [(box.cml_lo, box.cml_hi)] if box.cml_lo <= box.cml_hi else [(box.cml_lo, 360.0), (0.0, box.cml_hi)]
        for cml_lo, cml_hi in cml_parts:
            ax.add_patch(
                plt.Rectangle(
                    (cml_lo, box.io_lo),
                    cml_hi - cml_lo,
                    box.io_hi - box.io_lo,
                    fill=False,
                    edgecolor=color,
                    lw=1.15,
                )
            )
        label_x = box.cml_lo if box.cml_lo <= box.cml_hi else 302.0
        ax.text(label_x + 2.0, box.io_hi - 2.0, box.name, color=color, fontsize=7, va="top", ha="left")


def plot_shift_control_summary(real: pd.DataFrame, shifts: pd.DataFrame, out_dir: Path) -> Path:
    channel_order = []
    for freq in sorted(real["frequency_mhz"].dropna().unique()):
        for antenna in ANTENNAS:
            row = real[real["antenna"].eq(antenna) & np.isclose(real["frequency_mhz"], float(freq))]
            if not row.empty:
                channel_order.append((antenna, float(freq)))
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 8.0), sharex=True)
    metrics = [
        ("source_minus_control_median_daily_resid", "source-box - visible-control median residual (dex)"),
        ("source_minus_control_factor_high_fraction", "source-box - visible-control high fraction"),
    ]
    rng = np.random.default_rng(20260611)
    for ax, (metric, ylabel) in zip(axes, metrics):
        for xpos, (antenna, freq) in enumerate(channel_order):
            sub_shift = shifts[shifts["antenna"].eq(antenna) & np.isclose(shifts["frequency_mhz"], freq)]
            xs = xpos + rng.normal(0, 0.045, size=len(sub_shift))
            ax.scatter(xs, sub_shift[metric], s=28, color="0.68", alpha=0.75, label="shifted phase controls" if xpos == 0 else None)
            sub_real = real[real["antenna"].eq(antenna) & np.isclose(real["frequency_mhz"], freq)]
            color = ANTENNA_COLOR.get(antenna, "black")
            if not sub_real.empty:
                ax.scatter(
                    [xpos],
                    [float(sub_real[metric].iloc[0])],
                    marker="D",
                    s=54,
                    color=color,
                    edgecolor="black",
                    linewidth=0.45,
                    zorder=5,
                    label="real CML/Io source boxes" if xpos == 0 else None,
                )
        ax.axhline(0.0, color="0.25", lw=0.9)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", color="0.9", lw=0.5)
    labels = [f"{freq:.2f}\n{ANTENNA_LABEL[antenna]}" for antenna, freq in channel_order]
    axes[-1].set_xticks(np.arange(len(channel_order)))
    axes[-1].set_xticklabels(labels)
    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle("Jupiter source-box contrast: real phase windows vs time-shifted phase controls")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = out_dir / "jupiter_source_box_real_vs_shifted_controls.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def rank_source_windows(samples: pd.DataFrame, source_windows: pd.DataFrame, high_resid_threshold: float, min_samples: int) -> pd.DataFrame:
    if source_windows.empty:
        return pd.DataFrame()
    rows = []
    times = samples["time"]
    for _, win in source_windows.iterrows():
        start = pd.Timestamp(win["start_time"])
        end = pd.Timestamp(win["end_time"])
        sub = samples[
            (times >= start)
            & (times <= end)
            & samples["jupiter_visible_by_moon"].astype(bool)
            & samples["source_box_any"].astype(bool)
        ].copy()
        if len(sub) < int(min_samples):
            continue
        rows.append(
            {
                **win.to_dict(),
                "n_samples": int(len(sub)),
                "n_factor_high_samples": int((sub["daily_log10_residual"] >= float(high_resid_threshold)).sum()),
                "factor_high_fraction": float(np.mean(sub["daily_log10_residual"] >= float(high_resid_threshold))),
                "median_daily_resid": float(sub["daily_log10_residual"].median()),
                "q90_daily_resid": float(np.nanquantile(sub["daily_log10_residual"], 0.90)),
                "max_daily_resid": float(sub["daily_log10_residual"].max()),
                "median_local20min_resid": float(sub["local20min_log10_residual"].median())
                if "local20min_log10_residual" in sub.columns
                else np.nan,
                "n_antennas": int(sub["antenna"].nunique()),
                "n_frequencies": int(sub["frequency_mhz"].nunique()),
            }
        )
    if not rows:
        return pd.DataFrame()
    ranked = pd.DataFrame(rows)
    return ranked.sort_values(
        ["factor_high_fraction", "q90_daily_resid", "n_factor_high_samples", "n_samples"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def plot_top_window_raw_strips(
    samples: pd.DataFrame,
    ranked_windows: pd.DataFrame,
    out_dir: Path,
    n_windows: int,
    minutes_each_side: float,
    metric: str,
    ylabel: str,
    filename: str,
) -> Path | None:
    if ranked_windows.empty:
        return None
    windows = ranked_windows.head(int(n_windows)).copy()
    fig, axes = plt.subplots(len(windows), 2, figsize=(13.0, max(2.05 * len(windows), 4.8)), sharex=False, sharey=False)
    axes = np.atleast_2d(axes)
    for row, (_, win) in enumerate(windows.iterrows()):
        start = pd.Timestamp(win["start_time"])
        end = pd.Timestamp(win["end_time"])
        center = start + (end - start) / 2
        lo = center - pd.Timedelta(minutes=float(minutes_each_side))
        hi = center + pd.Timedelta(minutes=float(minutes_each_side))
        for col, antenna in enumerate(ANTENNAS):
            ax = axes[row, col]
            sub = samples[(samples["time"] >= lo) & (samples["time"] <= hi) & samples["antenna"].eq(antenna)].copy()
            sub["t_min"] = (sub["time"] - center).dt.total_seconds() / 60.0
            for freq, fgrp in sub.groupby("frequency_mhz", sort=True):
                color = FREQ_COLOR.get(round(float(freq), 2), None)
                fgrp = fgrp.sort_values("t_min")
                gap_group = fgrp["t_min"].diff().abs().gt(6.0).fillna(False).cumsum()
                first_segment = True
                for _, seg in fgrp.groupby(gap_group, sort=True):
                    ax.plot(
                        seg["t_min"],
                        seg[metric],
                        marker=".",
                        ms=2.0,
                        lw=0.65 if len(seg) > 1 else 0.0,
                        alpha=0.78,
                        color=color,
                        label=f"{freq:.2f} MHz" if first_segment else None,
                    )
                    first_segment = False
            ax.axvspan((start - center).total_seconds() / 60.0, (end - center).total_seconds() / 60.0, color="black", alpha=0.08)
            ax.axvline(0, color="0.25", lw=0.75)
            if metric != "log10_power":
                ax.axhline(0, color="0.55", lw=0.6)
            finite = sub[metric].to_numpy(dtype=float)
            finite = finite[np.isfinite(finite)]
            if finite.size:
                qlo, qhi = np.nanquantile(finite, [0.01, 0.99])
                pad = max(0.02, 0.08 * (qhi - qlo))
                ax.set_ylim(qlo - pad, qhi + pad)
            ax.set_title(
                f"{ANTENNA_LABEL[antenna]}  {win['window_id']} {win['source_boxes']}  "
                f"Io={float(win['median_io_phase_deg']):.0f} CML={float(win['median_cml_deg']):.0f}",
                loc="left",
                fontsize=8.5,
            )
            ax.grid(True, color="0.9", lw=0.45)
            if col == 0:
                ax.set_ylabel(ylabel)
        axes[row, 0].text(
            0.01,
            0.92,
            f"{center:%Y-%m-%d %H:%M}  high frac={float(win['factor_high_fraction']):.2f}",
            transform=axes[row, 0].transAxes,
            fontsize=8,
            va="top",
            ha="left",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
        )
    axes[0, 1].legend(frameon=False, ncol=2, fontsize=7, loc="upper right")
    axes[-1, 0].set_xlabel("minutes from source-box window center")
    axes[-1, 1].set_xlabel("minutes from source-box window center")
    fig.suptitle(f"Top Jupiter CML/Io source-box intervals: {ylabel}")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = out_dir / filename
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_aligned_stack_profiles(
    real_profile: pd.DataFrame,
    control_profile: pd.DataFrame,
    out_dir: Path,
    metric_label: str,
    filename_suffix: str,
    window_minutes: float,
) -> list[Path]:
    if real_profile.empty:
        return []
    paths = []
    freqs = sorted(real_profile["frequency_mhz"].dropna().unique())
    if not freqs:
        return []
    align_titles = {
        "entry": "source-box entry aligned",
        "maser_peak": "peak MASER-score aligned",
        "exit": "source-box exit aligned",
    }
    for align_type in ["entry", "maser_peak", "exit"]:
        fig_width = max(15.5, 2.75 * len(freqs))
        fig, axes = plt.subplots(2, len(freqs), figsize=(fig_width, 7.0), sharex=True, sharey=True)
        axes = np.asarray(axes)
        if axes.ndim == 1:
            axes = axes[:, np.newaxis]
        values_for_limits = []
        for row, antenna in enumerate(ANTENNAS):
            for col, freq in enumerate(freqs):
                ax = axes[row, col]
                real = real_profile[
                    real_profile["align_type"].eq(align_type)
                    & real_profile["antenna"].eq(antenna)
                    & np.isclose(real_profile["frequency_mhz"], float(freq))
                ].sort_values("t_bin_min")
                control = control_profile[
                    control_profile["align_type"].eq(align_type)
                    & control_profile["antenna"].eq(antenna)
                    & np.isclose(control_profile["frequency_mhz"], float(freq))
                ].sort_values("t_bin_min")
                if not control.empty:
                    x = control["t_bin_min"].to_numpy(dtype=float)
                    cmed = control["control_median"].to_numpy(dtype=float)
                    cq10 = control["control_q10"].to_numpy(dtype=float)
                    cq90 = control["control_q90"].to_numpy(dtype=float)
                    ax.fill_between(x, cq10, cq90, color="0.72", alpha=0.40, lw=0, label="shifted controls 10-90%" if row == 0 and col == 0 else None)
                    ax.plot(x, cmed, color="0.38", lw=1.0, alpha=0.85, label="shifted controls median" if row == 0 and col == 0 else None)
                    values_for_limits.extend(cq10[np.isfinite(cq10)].tolist())
                    values_for_limits.extend(cq90[np.isfinite(cq90)].tolist())
                if not real.empty:
                    x = real["t_bin_min"].to_numpy(dtype=float)
                    rmed = real["real_median"].to_numpy(dtype=float)
                    rq25 = real["real_q25"].to_numpy(dtype=float)
                    rq75 = real["real_q75"].to_numpy(dtype=float)
                    color = ANTENNA_COLOR.get(antenna, "black")
                    ax.fill_between(x, rq25, rq75, color=color, alpha=0.16, lw=0)
                    ax.plot(x, rmed, color=color, lw=1.7, label="real source-box windows" if row == 0 and col == 0 else None)
                    values_for_limits.extend(rq25[np.isfinite(rq25)].tolist())
                    values_for_limits.extend(rq75[np.isfinite(rq75)].tolist())
                    ax.text(
                        0.02,
                        0.92,
                        f"n={int(real['n_real_event_bins'].max())}",
                        transform=ax.transAxes,
                        fontsize=8,
                        ha="left",
                        va="top",
                    )
                ax.axvline(0.0, color="0.20", lw=0.9)
                ax.axhline(0.0, color="0.55", lw=0.65)
                ax.set_xlim(-float(window_minutes), float(window_minutes))
                ax.grid(True, color="0.91", lw=0.45)
                ax.set_title(f"{ANTENNA_LABEL[antenna]} {freq:.2f} MHz", fontsize=9)
                if col == 0:
                    ax.set_ylabel(metric_label)
                if row == 1:
                    ax.set_xlabel("minutes from alignment time")
        if values_for_limits:
            finite = np.asarray(values_for_limits, dtype=float)
            finite = finite[np.isfinite(finite)]
            if finite.size:
                lim = max(0.03, float(np.nanpercentile(np.abs(finite), 98)))
                lim = min(lim, 0.45)
                for ax in axes.ravel():
                    ax.set_ylim(-lim, lim)
        axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left")
        fig.suptitle(f"Jupiter source-box aligned stack: {align_titles.get(align_type, align_type)}")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        path = out_dir / f"jupiter_source_box_{align_type}_aligned_{filename_suffix}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_aligned_high_fraction_profiles(
    real_profile: pd.DataFrame,
    control_profile: pd.DataFrame,
    out_dir: Path,
    high_factor: float,
    window_minutes: float,
) -> list[Path]:
    if real_profile.empty:
        return []
    paths = []
    freqs = sorted(real_profile["frequency_mhz"].dropna().unique())
    if not freqs:
        return []
    align_titles = {
        "entry": "source-box entry aligned",
        "maser_peak": "peak MASER-score aligned",
        "exit": "source-box exit aligned",
    }
    for align_type in ["entry", "maser_peak", "exit"]:
        fig_width = max(15.5, 2.75 * len(freqs))
        fig, axes = plt.subplots(2, len(freqs), figsize=(fig_width, 7.0), sharex=True, sharey=True)
        axes = np.asarray(axes)
        if axes.ndim == 1:
            axes = axes[:, np.newaxis]
        values_for_limits = []
        for row, antenna in enumerate(ANTENNAS):
            for col, freq in enumerate(freqs):
                ax = axes[row, col]
                real = real_profile[
                    real_profile["align_type"].eq(align_type)
                    & real_profile["antenna"].eq(antenna)
                    & np.isclose(real_profile["frequency_mhz"], float(freq))
                ].sort_values("t_bin_min")
                control = control_profile[
                    control_profile["align_type"].eq(align_type)
                    & control_profile["antenna"].eq(antenna)
                    & np.isclose(control_profile["frequency_mhz"], float(freq))
                ].sort_values("t_bin_min")
                if not control.empty:
                    x = control["t_bin_min"].to_numpy(dtype=float)
                    cmed = control["control_high_fraction_median"].to_numpy(dtype=float)
                    cq10 = control["control_high_fraction_q10"].to_numpy(dtype=float)
                    cq90 = control["control_high_fraction_q90"].to_numpy(dtype=float)
                    ax.fill_between(x, cq10, cq90, color="0.72", alpha=0.40, lw=0, label="shifted controls 10-90%" if row == 0 and col == 0 else None)
                    ax.plot(x, cmed, color="0.38", lw=1.0, alpha=0.85, label="shifted controls median" if row == 0 and col == 0 else None)
                    values_for_limits.extend(cq90[np.isfinite(cq90)].tolist())
                if not real.empty:
                    x = real["t_bin_min"].to_numpy(dtype=float)
                    y = real["real_high_fraction"].to_numpy(dtype=float)
                    color = ANTENNA_COLOR.get(antenna, "black")
                    ax.plot(x, y, color=color, lw=1.7, label="real source-box windows" if row == 0 and col == 0 else None)
                    ax.scatter(x, y, color=color, s=8, alpha=0.75)
                    values_for_limits.extend(y[np.isfinite(y)].tolist())
                    ax.text(
                        0.02,
                        0.92,
                        f"n={int(real['n_real_event_bins'].max())}",
                        transform=ax.transAxes,
                        fontsize=8,
                        ha="left",
                        va="top",
                    )
                ax.axvline(0.0, color="0.20", lw=0.9)
                ax.set_xlim(-float(window_minutes), float(window_minutes))
                ax.set_ylim(bottom=0.0)
                ax.grid(True, color="0.91", lw=0.45)
                ax.set_title(f"{ANTENNA_LABEL[antenna]} {freq:.2f} MHz", fontsize=9)
                if col == 0:
                    ax.set_ylabel(f"fraction >= {high_factor:g}x daily median")
                if row == 1:
                    ax.set_xlabel("minutes from alignment time")
        if values_for_limits:
            vmax = max(0.03, float(np.nanpercentile(np.asarray(values_for_limits, dtype=float), 98)))
            vmax = min(0.45, vmax * 1.12)
            for ax in axes.ravel():
                ax.set_ylim(0.0, vmax)
        axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left")
        fig.suptitle(f"Jupiter source-box aligned high-power fraction: {align_titles.get(align_type, align_type)}")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        path = out_dir / f"jupiter_source_box_{align_type}_aligned_factor_high_fraction_stack.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def write_report(
    out_dir: Path,
    samples: pd.DataFrame,
    geom: pd.DataFrame,
    source_windows: pd.DataFrame,
    alignment_events: pd.DataFrame,
    real_with_percentiles: pd.DataFrame,
    shifts: pd.DataFrame,
    ranked_windows: pd.DataFrame,
    paths: list[Path],
    config: dict[str, object],
) -> Path:
    visible = samples["jupiter_visible_by_moon"].astype(bool)
    source = samples["source_box_any"].astype(bool) & visible
    source_counts = {}
    for box in SOURCE_BOXES:
        col = f"source_{box.name.lower().replace('-', '_')}"
        source_counts[box.name] = int((samples[col].astype(bool) & visible).sum()) if col in samples else 0

    compact_cols = [
        "antenna_label",
        "frequency_mhz",
        "n_source",
        "source_minus_control_median_daily_resid",
        "source_minus_control_median_daily_resid_shift_percentile",
        "source_minus_control_factor_high_fraction",
        "source_minus_control_factor_high_fraction_shift_percentile",
    ]
    compact = real_with_percentiles[compact_cols].copy()
    compact = compact.sort_values(["frequency_mhz", "antenna_label"])
    top_cols = [
        "window_id",
        "start_time",
        "end_time",
        "source_boxes",
        "n_samples",
        "factor_high_fraction",
        "median_daily_resid",
        "q90_daily_resid",
        "max_daily_resid",
    ]
    top = ranked_windows[top_cols].head(int(config["top_windows"])) if not ranked_windows.empty else pd.DataFrame(columns=top_cols)
    lines = [
        "# Jupiter Source-Box Direct Detection",
        "",
        "This run tests the configured Ryle-Vonberg bands against standard Io-controlled Jupiter CML/Io phase boxes. If 0.45 MHz is configured, it is included for comparison but should not carry the Jupiter interpretation by itself.",
        "",
        "## Source-Box Basis",
        "",
        "The source boxes are the usual Io-A, Io-B, Io-C, and Io-D regions in System-III CML versus Io phase. They are used as an a-priori selector, not fitted from the RAE-2 data.",
        "",
        "Useful references checked for this run:",
        "",
        "- https://www.radiosky.com/jupmodes.html",
        "- https://radiojove.gsfc.nasa.gov/library/sci_briefs/decametric.htm",
        "- https://www.aanda.org/articles/aa/full_html/2017/08/aa30025-16/aa30025-16.html",
        "",
        "## Configuration",
        "",
        *[f"- `{k}`: `{v}`" for k, v in config.items() if k != "software_versions"],
        "",
        "## Coverage",
        "",
        f"- merged configured-band samples: `{len(samples)}`",
        f"- Jupiter-visible configured-band samples: `{int(visible.sum())}`",
        f"- visible source-box configured-band samples: `{int(source.sum())}`",
        f"- source-box geometry windows: `{len(source_windows)}`",
        f"- real alignment events: `{int(alignment_events['role'].eq('real').sum()) if not alignment_events.empty else 0}`",
        f"- shifted-control alignment events: `{int(alignment_events['role'].eq('shifted_control').sum()) if not alignment_events.empty else 0}`",
        f"- geometry grid Jupiter-occulted rows: `{int((~geom['jupiter_visible_by_moon'].astype(bool)).sum())}` of `{len(geom)}`",
        "",
        "Visible configured-band sample counts by source box:",
        "",
        *[f"- `{name}`: `{count}`" for name, count in source_counts.items()],
        "",
        "## Real Source Boxes Versus Shifted Phase Controls",
        "",
        compact.to_string(index=False),
        "",
        "Percentiles are the rank of the real source-box contrast among the configured time-shifted CML/Io controls. With only a small control set, these are screening diagnostics rather than formal p-values.",
        "",
        "## Top Source-Box Windows",
        "",
        top.to_string(index=False),
        "",
        "## What A Detection Should Look Like",
        "",
        "A convincing detection in these diagnostics should not depend on every favorable phase window being bright. Jupiter is bursty, so the expected signature is a repeated excess in the right geometry, not a steady continuum.",
        "",
        "The strongest evidence would look like this:",
        "",
        "- In the entry-, exit-, or MASER-peak-aligned stacks, the real source-box curve rises above the gray shifted-control band for adjacent time bins, not just one isolated point.",
        "- The timing should make physical sense: entry-aligned emission should prefer times after entry, exit-aligned emission should prefer times before exit, and MASER-peak-aligned emission should be near zero minutes if the score peak is useful.",
        "- The excess should recur in plausible Jovian bands, especially 3.93-9.18 MHz in this data set, and should not rely on 0.45 MHz alone.",
        "- The raw top-window strips should show actual points inside the source-box interval, with plausible multi-frequency or cross-antenna structure rather than a plot artifact from gaps.",
        "- The same feature should be weaker or absent in shifted phase-control stacks and in same-day visible non-source-box controls.",
        "",
        "Non-detections or weak cases look like real curves sitting inside the shifted-control band, broad day-scale offsets that are not tied to the alignment time, or isolated single-channel excursions with no repeated phase/timing behavior.",
        "",
        "## How To Read The Plots",
        "",
        "- CML/Io maps show direct configured-band data in the phase plane; boxes outline the a-priori Jupiter source regions.",
        "- The shifted-control plot compares the real source-box contrast with the same data tested against source boxes shifted in time. A real diamond sitting inside the gray control cloud is not compelling by itself.",
        "- The aligned-stack plots compare event-weighted real source-box profiles with shifted-control profiles. The vertical line is the alignment time, not necessarily a predicted maximum for every burst.",
        "- The aligned high-power-fraction plots are usually more useful for sparse bursts than median residual plots, because a few strong Jupiter-like bursts can be erased by median stacking.",
        f"- `factor-high` means `log10(power)` is at least `{config['high_factor']}x` the same-day median for that antenna/frequency.",
        "- Raw strips show the actual high-band log10 power near the top source-box intervals. The shaded span is the source-box interval.",
        "",
        "## Files",
        "",
        *[f"- `{p}`" for p in paths],
    ]
    path = out_dir / "jupiter_source_box_direct_detection_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-npy", type=Path, default=DEFAULT_CLEAN_NPY)
    parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--frequency-band", type=int, nargs="+", default=[6, 7, 8, 9])
    parser.add_argument("--geometry-tolerance-s", type=float, default=360.0)
    parser.add_argument("--source-box-pad-deg", type=float, default=0.0)
    parser.add_argument("--high-factor", type=float, default=2.0)
    parser.add_argument("--shift-days", type=float, nargs="+", default=[-7, -5, -3, -2, -1, 1, 2, 3, 5, 7])
    parser.add_argument("--source-window-max-gap-min", type=float, default=18.0)
    parser.add_argument("--local-window", type=str, default="20min")
    parser.add_argument("--local-min-periods", type=int, default=8)
    parser.add_argument("--top-windows", type=int, default=8)
    parser.add_argument("--top-window-min-samples", type=int, default=24)
    parser.add_argument("--plot-window-minutes", type=float, default=90.0)
    parser.add_argument("--alignment-window-minutes", type=float, default=90.0)
    parser.add_argument("--alignment-bin-minutes", type=float, default=5.0)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    high_resid_threshold = float(np.log10(float(args.high_factor)))
    config = {
        "clean_npy": str(args.clean_npy),
        "geometry": str(args.geometry),
        "frequency_bands": [int(v) for v in args.frequency_band],
        "frequencies_mhz": [FREQUENCY_MAP_MHZ.get(int(v), np.nan) for v in args.frequency_band],
        "geometry_tolerance_s": float(args.geometry_tolerance_s),
        "source_box_pad_deg": float(args.source_box_pad_deg),
        "high_factor": float(args.high_factor),
        "high_resid_threshold_dex": high_resid_threshold,
        "shift_days": [float(v) for v in args.shift_days],
        "source_window_max_gap_min": float(args.source_window_max_gap_min),
        "local_window": str(args.local_window),
        "local_min_periods": int(args.local_min_periods),
        "top_windows": int(args.top_windows),
        "top_window_min_samples": int(args.top_window_min_samples),
        "plot_window_minutes": float(args.plot_window_minutes),
        "alignment_window_minutes": float(args.alignment_window_minutes),
        "alignment_bin_minutes": float(args.alignment_bin_minutes),
        "software_versions": software_versions(),
    }
    write_json(out_dir / "run_config.json", config)
    source_defs = source_box_definitions(float(args.source_box_pad_deg))
    source_defs.to_csv(out_dir / "jupiter_source_box_definitions.csv", index=False)

    print("Loading configured-band clean samples...", flush=True)
    samples = read_clean_npy_subset(args.clean_npy, [int(v) for v in args.frequency_band])
    print(f"Loaded {len(samples)} configured-band samples; adding local residuals...", flush=True)
    samples = add_local_residual(samples, window=str(args.local_window), min_periods=int(args.local_min_periods))
    print("Loading and merging Jupiter geometry...", flush=True)
    geom = load_geometry(args.geometry, pad_deg=float(args.source_box_pad_deg))
    samples = merge_geometry(samples, geom, tolerance_s=float(args.geometry_tolerance_s), pad_deg=float(args.source_box_pad_deg))
    samples["factor_high"] = samples["daily_log10_residual"] >= high_resid_threshold

    active_geom = geom["jupiter_visible_by_moon"].astype(bool) & geom["source_box_any"].astype(bool)
    source_windows = contiguous_windows(geom, active_geom, max_gap_min=float(args.source_window_max_gap_min))
    source_windows.to_csv(out_dir / "jupiter_source_box_geometry_windows.csv", index=False)

    real_mask = samples["source_box_any"].to_numpy(dtype=bool)
    real_summary = summarize_channel_contrast(
        samples,
        source_mask=real_mask,
        label="real_source_boxes",
        high_resid_threshold=high_resid_threshold,
    )
    shift_summary = build_shift_control_summary(
        samples,
        source_windows=source_windows,
        shift_days=[float(v) for v in args.shift_days],
        high_resid_threshold=high_resid_threshold,
    )
    real_with_percentiles = add_shift_percentiles(real_summary, shift_summary)
    real_with_percentiles.to_csv(out_dir / "jupiter_source_box_real_channel_summary.csv", index=False)
    shift_summary.to_csv(out_dir / "jupiter_source_box_shifted_control_channel_summary.csv", index=False)

    ranked_windows = rank_source_windows(
        samples,
        source_windows,
        high_resid_threshold=high_resid_threshold,
        min_samples=int(args.top_window_min_samples),
    )
    ranked_windows.to_csv(out_dir / "jupiter_source_box_ranked_windows.csv", index=False)
    alignment_events = build_alignment_events(
        geom,
        source_windows,
        shift_days=[float(v) for v in args.shift_days],
        sample_start=pd.Timestamp(samples["time"].min()),
        sample_end=pd.Timestamp(samples["time"].max()),
    )
    alignment_events.to_csv(out_dir / "jupiter_source_box_alignment_events.csv", index=False)

    aligned_plot_paths: list[Path] = []
    for metric, label, suffix in [
        ("daily_log10_residual", "daily residual (dex)", "daily_residual_stack"),
        ("local20min_log10_residual", "20-min local residual (dex)", "local20min_residual_stack"),
    ]:
        print(f"Building aligned stack profiles for {metric}...", flush=True)
        bin_values = aligned_event_bin_values(
            samples,
            alignment_events,
            metric=metric,
            window_minutes=float(args.alignment_window_minutes),
            bin_minutes=float(args.alignment_bin_minutes),
        )
        real_profile, control_profile = summarize_aligned_profiles(bin_values)
        real_profile.to_csv(out_dir / f"jupiter_source_box_aligned_{suffix}_real_profile.csv", index=False)
        control_profile.to_csv(out_dir / f"jupiter_source_box_aligned_{suffix}_shifted_control_profile.csv", index=False)
        aligned_plot_paths.extend(
            plot_aligned_stack_profiles(
                real_profile,
                control_profile,
                out_dir,
                metric_label=label,
                filename_suffix=suffix,
                window_minutes=float(args.alignment_window_minutes),
            )
        )
        if metric == "daily_log10_residual":
            high_real, high_control = summarize_aligned_high_fraction(bin_values, threshold=high_resid_threshold)
            high_real.to_csv(out_dir / "jupiter_source_box_aligned_factor_high_fraction_real_profile.csv", index=False)
            high_control.to_csv(out_dir / "jupiter_source_box_aligned_factor_high_fraction_shifted_control_profile.csv", index=False)
            aligned_plot_paths.extend(
                plot_aligned_high_fraction_profiles(
                    high_real,
                    high_control,
                    out_dir,
                    high_factor=float(args.high_factor),
                    window_minutes=float(args.alignment_window_minutes),
                )
            )

    paths: list[Path] = [
        out_dir / "run_config.json",
        out_dir / "jupiter_source_box_definitions.csv",
        out_dir / "jupiter_source_box_real_channel_summary.csv",
        out_dir / "jupiter_source_box_shifted_control_channel_summary.csv",
        out_dir / "jupiter_source_box_ranked_windows.csv",
        out_dir / "jupiter_source_box_geometry_windows.csv",
        out_dir / "jupiter_source_box_alignment_events.csv",
        out_dir / "jupiter_source_box_aligned_daily_residual_stack_real_profile.csv",
        out_dir / "jupiter_source_box_aligned_daily_residual_stack_shifted_control_profile.csv",
        out_dir / "jupiter_source_box_aligned_local20min_residual_stack_real_profile.csv",
        out_dir / "jupiter_source_box_aligned_local20min_residual_stack_shifted_control_profile.csv",
        out_dir / "jupiter_source_box_aligned_factor_high_fraction_real_profile.csv",
        out_dir / "jupiter_source_box_aligned_factor_high_fraction_shifted_control_profile.csv",
    ]
    print("Making plots...", flush=True)
    paths.extend(plot_source_box_phase_maps(samples, out_dir, high_resid_threshold=high_resid_threshold))
    paths.append(plot_shift_control_summary(real_with_percentiles, shift_summary, out_dir))
    paths.extend(aligned_plot_paths)
    for maybe_path in [
        plot_top_window_raw_strips(
            samples,
            ranked_windows,
            out_dir,
            n_windows=int(args.top_windows),
            minutes_each_side=float(args.plot_window_minutes),
            metric="log10_power",
            ylabel="raw log10(power)",
            filename="jupiter_source_box_top_windows_raw_log10_power.png",
        ),
        plot_top_window_raw_strips(
            samples,
            ranked_windows,
            out_dir,
            n_windows=int(args.top_windows),
            minutes_each_side=float(args.plot_window_minutes),
            metric="daily_log10_residual",
            ylabel="daily residual (dex)",
            filename="jupiter_source_box_top_windows_daily_residual.png",
        ),
        plot_top_window_raw_strips(
            samples,
            ranked_windows,
            out_dir,
            n_windows=int(args.top_windows),
            minutes_each_side=float(args.plot_window_minutes),
            metric="local20min_log10_residual",
            ylabel="20-min local residual (dex)",
            filename="jupiter_source_box_top_windows_local20min_residual.png",
        ),
    ]:
        if maybe_path is not None:
            paths.append(maybe_path)

    report = write_report(
        out_dir,
        samples=samples,
        geom=geom,
        source_windows=source_windows,
        alignment_events=alignment_events,
        real_with_percentiles=real_with_percentiles,
        shifts=shift_summary,
        ranked_windows=ranked_windows,
        paths=paths,
        config=config,
    )
    print(f"Wrote {report}", flush=True)


if __name__ == "__main__":
    main()
