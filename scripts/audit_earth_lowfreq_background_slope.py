#!/usr/bin/env python
"""Audit why Earth low-frequency profiles have consistent opposite slopes.

This script intentionally uses raw power pre/post medians rather than SNR or
source-like scoring.  The goal is to separate three effects:

1. MIE/sampling edge smoothing, which can make a source edge less sharp.
2. A true compact-source occultation, which should be centered on the predicted
   event time and have the ordinary disappearance/reappearance sign.
3. A background-slope/orbital-phase effect, where the event table selects
   repeated spacecraft/beam/sky states whose raw power is already trending up
   or down regardless of the exact limb time.
"""

from __future__ import annotations

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

from rylevonberg.util import datetime_ns, ensure_dir, robust_sigma, software_versions, write_json  # noqa: E402


CLEAN = ROOT / "outputs/control_survey_earth_sun_postnov1974_v2/01_ingest/cleaned_timeseries.csv"
EVENTS = ROOT / "outputs/planetary_confirmation_survey_science_baseline_v2/events/all_planet_predicted_events.csv"
PRECOMPUTED_SHIFTS = ROOT / "outputs/earth_lowfreq_profile_sign_diagnostics_v1/earth_event_level_prepost_contrasts_with_shifts.csv"
OUT = ROOT / "outputs/earth_lowfreq_background_slope_audit_v1"

ANTENNA = "rv2_coarse"
LOW_FREQS = [0.70, 0.90, 1.31, 2.20]
LOW_BANDS = [2, 3, 4, 5]
TIME_SHIFTS_S = [-1800.0, -1200.0, -600.0, -300.0, 0.0, 300.0, 600.0, 1200.0, 1800.0]
WINDOW_S = 900.0
PRE_NEAR = (-180.0, -60.0)
POST_NEAR = (60.0, 180.0)
SIDE_INNER_S = 15.0
EXPECTED_SIGN = {"disappearance": -1.0, "reappearance": 1.0}


def _read_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    clean_cols = [
        "time",
        "frequency_band",
        "frequency_mhz",
        "antenna",
        "power",
        "is_valid",
        "position_x",
        "position_y",
        "position_z",
        "right_ascension",
        "declination",
    ]
    clean = read_table(CLEAN, usecols=clean_cols, parse_dates=["time"], low_memory=False)
    clean = clean[
        clean["antenna"].astype(str).eq(ANTENNA)
        & clean["frequency_band"].isin(LOW_BANDS)
        & clean["is_valid"].astype(bool)
    ].copy()
    events = read_table(EVENTS, parse_dates=["predicted_event_time"], low_memory=False)
    events = events[
        events["source_name"].astype(str).str.lower().eq("earth")
        & events["antenna"].astype(str).eq(ANTENNA)
        & events["frequency_band"].isin(LOW_BANDS)
    ].copy()
    return clean, events


def _read_precomputed_shifts(events: pd.DataFrame) -> pd.DataFrame:
    shifts = read_table(PRECOMPUTED_SHIFTS, low_memory=False)
    shifts = shifts[
        shifts["antenna"].astype(str).eq(ANTENNA)
        & shifts["frequency_band"].isin(LOW_BANDS)
        & shifts["time_shift_s"].isin(TIME_SHIFTS_S)
    ].copy()
    meta = events[
        [
            "event_id",
            "frequency_band",
            "predicted_event_time",
            "moon_center_x",
            "moon_center_y",
            "moon_center_z",
        ]
    ].drop_duplicates(["event_id", "frequency_band"])
    shifts = shifts.merge(meta, on=["event_id", "frequency_band"], how="left")
    shifts["predicted_event_time"] = pd.to_datetime(shifts["predicted_event_time"], errors="coerce")
    shifts["center_time"] = shifts["predicted_event_time"] + pd.to_timedelta(shifts["time_shift_s"], unit="s")
    shifts = shifts.rename(
        columns={
            "post_minus_pre_z": "normalized_post_minus_pre",
            "source_like_signed_contrast_z": "source_like_normalized_contrast",
        }
    )
    return shifts


def _groups(clean: pd.DataFrame) -> dict[int, tuple[pd.DataFrame, np.ndarray]]:
    out = {}
    for band, grp in clean.groupby("frequency_band", sort=True):
        g = grp.sort_values("time").reset_index(drop=True)
        out[int(band)] = (g, datetime_ns(g["time"]))
    return out


def _window(group: pd.DataFrame, group_ns: np.ndarray, center_time: pd.Timestamp) -> pd.DataFrame:
    center_ns = pd.Timestamp(center_time).value
    half_ns = int(WINDOW_S * 1e9)
    lo = int(np.searchsorted(group_ns, center_ns - half_ns, side="left"))
    hi = int(np.searchsorted(group_ns, center_ns + half_ns, side="right"))
    if hi <= lo:
        return pd.DataFrame()
    local = group.iloc[lo:hi].copy()
    local["t_rel_sec"] = (datetime_ns(local["time"]) - center_ns).astype(float) / 1e9
    return local[np.abs(local["t_rel_sec"]) <= WINDOW_S].copy()


def _nearest_row(group: pd.DataFrame, group_ns: np.ndarray, center_time: pd.Timestamp) -> pd.Series | None:
    center_ns = pd.Timestamp(center_time).value
    idx = int(np.searchsorted(group_ns, center_ns, side="left"))
    candidates = [idx]
    if idx > 0:
        candidates.append(idx - 1)
    if idx + 1 < len(group):
        candidates.append(idx + 1)
    candidates = [i for i in candidates if 0 <= i < len(group)]
    if not candidates:
        return None
    best = min(candidates, key=lambda i: abs(int(group_ns[i]) - center_ns))
    return group.iloc[best]


def _phase_deg(row: pd.Series) -> float:
    return float(np.degrees(np.arctan2(float(row["position_y"]), float(row["position_x"]))) % 360.0)


def _moon_ra_dec(ev: pd.Series) -> tuple[float, float]:
    x = float(ev["moon_center_x"])
    y = float(ev["moon_center_y"])
    z = float(ev["moon_center_z"])
    r = np.sqrt(x * x + y * y + z * z)
    if not np.isfinite(r) or r <= 0:
        return np.nan, np.nan
    ra = float(np.degrees(np.arctan2(y, x)) % 360.0)
    dec = float(np.degrees(np.arcsin(np.clip(z / r, -1.0, 1.0))))
    return ra, dec


def _contrast_for_window(local: pd.DataFrame) -> dict[str, float]:
    if local.empty:
        return {
            "n_pre": 0,
            "n_post": 0,
            "raw_pre": np.nan,
            "raw_post": np.nan,
            "raw_post_minus_pre": np.nan,
            "normalized_post_minus_pre": np.nan,
            "side_sigma": np.nan,
        }
    t = pd.to_numeric(local["t_rel_sec"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(local["power"], errors="coerce").to_numpy(dtype=float)
    good = np.isfinite(t) & np.isfinite(y)
    t = t[good]
    y = y[good]
    pre = (t >= PRE_NEAR[0]) & (t <= PRE_NEAR[1])
    post = (t >= POST_NEAR[0]) & (t <= POST_NEAR[1])
    side = np.abs(t) >= SIDE_INNER_S
    if np.count_nonzero(pre) < 2 or np.count_nonzero(post) < 2 or np.count_nonzero(side) < 6:
        return {
            "n_pre": int(np.count_nonzero(pre)),
            "n_post": int(np.count_nonzero(post)),
            "raw_pre": np.nan,
            "raw_post": np.nan,
            "raw_post_minus_pre": np.nan,
            "normalized_post_minus_pre": np.nan,
            "side_sigma": np.nan,
        }
    raw_pre = float(np.nanmedian(y[pre]))
    raw_post = float(np.nanmedian(y[post]))
    center = float(np.nanmedian(y[side]))
    scale = robust_sigma(y[side] - center)
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.nanstd(y[side]))
    norm_delta = np.nan if not np.isfinite(scale) or scale <= 0 else (raw_post - raw_pre) / scale
    return {
        "n_pre": int(np.count_nonzero(pre)),
        "n_post": int(np.count_nonzero(post)),
        "raw_pre": raw_pre,
        "raw_post": raw_post,
        "raw_post_minus_pre": float(raw_post - raw_pre),
        "normalized_post_minus_pre": float(norm_delta) if np.isfinite(norm_delta) else np.nan,
        "side_sigma": float(scale) if np.isfinite(scale) else np.nan,
    }


def build_event_shift_table(clean: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    groups = _groups(clean)
    rows = []
    for ev in events.itertuples(index=False):
        band = int(ev.frequency_band)
        payload = groups.get(band)
        if payload is None:
            continue
        group, group_ns = payload
        moon_ra, moon_dec = _moon_ra_dec(pd.Series(ev._asdict()))
        for shift in TIME_SHIFTS_S:
            center_time = pd.Timestamp(ev.predicted_event_time) + pd.to_timedelta(float(shift), unit="s")
            local = _window(group, group_ns, center_time)
            nearest = _nearest_row(group, group_ns, center_time)
            if nearest is None:
                continue
            stats = _contrast_for_window(local)
            sign = EXPECTED_SIGN[str(ev.event_type)]
            rows.append(
                {
                    "event_id": ev.event_id,
                    "event_type": ev.event_type,
                    "frequency_band": band,
                    "frequency_mhz": float(ev.frequency_mhz),
                    "time_shift_s": float(shift),
                    "predicted_event_time": pd.Timestamp(ev.predicted_event_time),
                    "center_time": center_time,
                    "spacecraft_phase_deg": _phase_deg(nearest),
                    "spacecraft_z_km": float(nearest["position_z"]),
                    "moon_center_ra_deg": moon_ra,
                    "moon_center_dec_deg": moon_dec,
                    **stats,
                    "source_like_normalized_contrast": sign * stats["normalized_post_minus_pre"]
                    if np.isfinite(stats["normalized_post_minus_pre"])
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def attach_phase_to_shift_table(clean: pd.DataFrame, event_shift: pd.DataFrame) -> pd.DataFrame:
    groups = _groups(clean)
    rows = []
    for band, sub in event_shift.groupby("frequency_band", sort=True):
        payload = groups.get(int(band))
        if payload is None:
            continue
        group, group_ns = payload
        phase_vals = []
        z_vals = []
        for center_time in pd.to_datetime(sub["center_time"], errors="coerce"):
            if pd.isna(center_time):
                phase_vals.append(np.nan)
                z_vals.append(np.nan)
                continue
            nearest = _nearest_row(group, group_ns, center_time)
            if nearest is None:
                phase_vals.append(np.nan)
                z_vals.append(np.nan)
            else:
                phase_vals.append(_phase_deg(nearest))
                z_vals.append(float(nearest["position_z"]))
        out = sub.copy()
        out["spacecraft_phase_deg"] = phase_vals
        out["spacecraft_z_km"] = z_vals
        moon_ra = []
        moon_dec = []
        for _, row in out.iterrows():
            ra, dec = _moon_ra_dec(row)
            moon_ra.append(ra)
            moon_dec.append(dec)
        out["moon_center_ra_deg"] = moon_ra
        out["moon_center_dec_deg"] = moon_dec
        rows.append(out)
    return pd.concat(rows, ignore_index=True) if rows else event_shift.copy()


def summarize_shift_table(event_shift: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, grp in event_shift.groupby(["frequency_mhz", "event_type", "time_shift_s"], sort=True):
        freq, event_type, shift = keys
        vals = pd.to_numeric(grp["normalized_post_minus_pre"], errors="coerce").dropna()
        src_vals = pd.to_numeric(grp["source_like_normalized_contrast"], errors="coerce").dropna()
        raw_vals = pd.to_numeric(grp["raw_post_minus_pre"], errors="coerce").dropna()
        rows.append(
            {
                "frequency_mhz": float(freq),
                "event_type": str(event_type),
                "time_shift_s": float(shift),
                "n_events": int(grp["event_id"].nunique()),
                "median_raw_post_minus_pre": float(raw_vals.median()) if not raw_vals.empty else np.nan,
                "median_normalized_post_minus_pre": float(vals.median()) if not vals.empty else np.nan,
                "positive_raw_slope_fraction": float((raw_vals > 0).mean()) if not raw_vals.empty else np.nan,
                "median_source_like_normalized_contrast": float(src_vals.median()) if not src_vals.empty else np.nan,
                "source_like_fraction": float((src_vals > 0).mean()) if not src_vals.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def phase_binned_background(clean: pd.DataFrame) -> pd.DataFrame:
    bins = np.arange(0.0, 361.0, 20.0)
    frames = []
    for band, grp in clean.groupby("frequency_band", sort=True):
        g = grp.sort_values("time").reset_index(drop=True)
        times = datetime_ns(g["time"])
        y = pd.to_numeric(g["power"], errors="coerce").to_numpy(dtype=float)
        phase = np.degrees(np.arctan2(g["position_y"].to_numpy(dtype=float), g["position_x"].to_numpy(dtype=float))) % 360.0
        # Slope from nearest samples about 2-6 minutes apart. This is deliberately
        # a local raw-power diagnostic, not an occultation model.
        if len(g) < 3:
            continue
        dt = (times[2:] - times[:-2]).astype(float) / 1e9
        slope = (y[2:] - y[:-2]) / dt
        phase_mid = phase[1:-1]
        keep = np.isfinite(slope) & np.isfinite(phase_mid) & (dt > 0) & (dt <= 900)
        if not np.any(keep):
            continue
        idx = np.clip(np.digitize(phase_mid[keep], bins) - 1, 0, len(bins) - 2)
        phase_bin = 0.5 * (bins[idx] + bins[idx + 1])
        frames.append(
            pd.DataFrame(
                {
                    "frequency_band": int(band),
                    "frequency_mhz": float(g["frequency_mhz"].iloc[0]),
                    "spacecraft_phase_deg": phase_mid[keep],
                    "raw_slope_per_s": slope[keep],
                    "phase_bin_deg": phase_bin,
                }
            )
        )
    slopes = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if slopes.empty:
        return slopes
    summary = (
        slopes.groupby(["frequency_band", "frequency_mhz", "phase_bin_deg"], as_index=False)
        .agg(
            median_raw_slope_per_s=("raw_slope_per_s", "median"),
            positive_slope_fraction=("raw_slope_per_s", lambda x: float((pd.Series(x).dropna() > 0).mean())),
            n_samples=("raw_slope_per_s", "size"),
        )
    )
    return summary


def event_phase_summary(event_shift: pd.DataFrame) -> pd.DataFrame:
    true = event_shift[np.isclose(event_shift["time_shift_s"], 0.0)].copy()
    rows = []
    for keys, grp in true.groupby(["frequency_mhz", "event_type"], sort=True):
        freq, event_type = keys
        vals = pd.to_numeric(grp["spacecraft_phase_deg"], errors="coerce").dropna()
        norm = pd.to_numeric(grp["normalized_post_minus_pre"], errors="coerce").dropna()
        raw = pd.to_numeric(grp["raw_post_minus_pre"], errors="coerce").dropna()
        phase_bin = np.nan
        phase_bin_fraction = np.nan
        if not vals.empty:
            bins = np.arange(0.0, 361.0, 20.0)
            idx = np.clip(np.digitize(vals.to_numpy(dtype=float), bins) - 1, 0, len(bins) - 2)
            centers = 0.5 * (bins[idx] + bins[idx + 1])
            counts = pd.Series(centers).value_counts()
            phase_bin = float(counts.index[0])
            phase_bin_fraction = float(counts.iloc[0] / len(vals))
        rows.append(
            {
                "frequency_mhz": float(freq),
                "event_type": str(event_type),
                "n_events": int(grp["event_id"].nunique()),
                "median_spacecraft_phase_deg": float(vals.median()) if not vals.empty else np.nan,
                "most_common_phase_bin_deg": phase_bin,
                "most_common_phase_bin_fraction": phase_bin_fraction,
                "phase_q16_deg": float(vals.quantile(0.16)) if not vals.empty else np.nan,
                "phase_q84_deg": float(vals.quantile(0.84)) if not vals.empty else np.nan,
                "median_normalized_post_minus_pre": float(norm.median()) if not norm.empty else np.nan,
                "positive_raw_slope_fraction": float((raw > 0).mean()) if not raw.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def plot_time_shift(summary: pd.DataFrame) -> Path:
    freqs = sorted(summary["frequency_mhz"].dropna().unique())
    fig, axes = plt.subplots(len(freqs), 1, figsize=(9.5, max(8, 1.8 * len(freqs))), sharex=True)
    if len(freqs) == 1:
        axes = np.asarray([axes])
    colors = {"disappearance": "#d95f02", "reappearance": "#1f77b4"}
    for ax, freq in zip(axes, freqs):
        sub = summary[np.isclose(summary["frequency_mhz"], freq)]
        for event_type, grp in sub.groupby("event_type", sort=True):
            grp = grp.sort_values("time_shift_s")
            ax.plot(
                grp["time_shift_s"] / 60.0,
                grp["median_normalized_post_minus_pre"],
                marker="o",
                linewidth=1.5,
                color=colors.get(event_type),
                label=event_type,
            )
        ax.axhline(0, color="0.5", linewidth=0.8)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_ylabel(f"{freq:.2f} MHz\npost-pre / local sigma")
        ax.grid(alpha=0.18)
    axes[0].legend(frameon=False, loc="best")
    axes[-1].set_xlabel("pseudo-event shift relative to predicted Earth event (minutes)")
    fig.suptitle(
        "Earth lower-V low-frequency raw trend versus time shift\n"
        "If the sign were only a sharp occultation edge, the effect should peak sharply at 0 minutes.",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    path = OUT / "earth_lowfreq_time_shift_raw_trend.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_phase(event_shift: pd.DataFrame, phase_bg: pd.DataFrame) -> Path:
    true = event_shift[np.isclose(event_shift["time_shift_s"], 0.0)].copy()
    freqs = sorted(true["frequency_mhz"].dropna().unique())
    fig, axes = plt.subplots(len(freqs), 1, figsize=(10.5, max(8, 1.9 * len(freqs))), sharex=True)
    if len(freqs) == 1:
        axes = np.asarray([axes])
    colors = {"disappearance": "#d95f02", "reappearance": "#1f77b4"}
    for ax, freq in zip(axes, freqs):
        bg = phase_bg[np.isclose(phase_bg["frequency_mhz"], freq)].sort_values("phase_bin_deg")
        if not bg.empty:
            ax.plot(
                bg["phase_bin_deg"],
                bg["median_raw_slope_per_s"],
                color="0.35",
                linewidth=1.2,
                label="all lower-V samples: median local raw slope",
            )
            ax.axhline(0, color="0.5", linewidth=0.8)
        sub = true[np.isclose(true["frequency_mhz"], freq)]
        ymax = np.nanmax(np.abs(bg["median_raw_slope_per_s"])) if not bg.empty else np.nan
        if not np.isfinite(ymax) or ymax <= 0:
            ymax = 1.0
        for event_type, grp in sub.groupby("event_type", sort=True):
            slope = pd.to_numeric(grp["raw_post_minus_pre"], errors="coerce").to_numpy(dtype=float) / 240.0
            phase = pd.to_numeric(grp["spacecraft_phase_deg"], errors="coerce").to_numpy(dtype=float)
            keep = np.isfinite(phase) & np.isfinite(slope)
            ax.scatter(
                phase[keep],
                np.clip(slope[keep], -3 * ymax, 3 * ymax),
                s=12,
                alpha=0.25,
                color=colors.get(event_type),
                label=event_type,
            )
        ax.set_ylabel(f"{freq:.2f} MHz\nraw slope")
        ax.grid(alpha=0.18)
    axes[0].legend(frameon=False, loc="best", ncol=2)
    axes[-1].set_xlabel("spacecraft orbital phase around Moon, atan2(y, x), deg")
    fig.suptitle(
        "Earth event raw trends versus spacecraft orbital phase\n"
        "Event type is not random in phase; this is where beam/sky background slope can enter the stack.",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    path = OUT / "earth_lowfreq_phase_background_slope.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report(
    shift_summary: pd.DataFrame,
    phase_summary: pd.DataFrame,
    paths: list[Path],
) -> None:
    focus = shift_summary[
        shift_summary["frequency_mhz"].isin([0.90, 1.31, 2.20])
        & shift_summary["time_shift_s"].isin([-600.0, -300.0, 0.0, 300.0, 600.0])
    ].copy()
    true_phase = phase_summary[phase_summary["frequency_mhz"].isin([0.90, 1.31, 2.20])].copy()
    lines = [
        "# Earth Low-Frequency Background-Slope Audit",
        "",
        "## Purpose",
        "",
        "This audit treats the MIE result as an edge-smoothing result only. It asks a separate question:",
        "",
        "> Why do low-frequency Earth disappearance stacks often rise and reappearance stacks often fall?",
        "",
        "All diagnostics here use lower V (`rv2_coarse`) raw power around Earth events. The key value is `post - pre` across",
        "`60-180 s` after versus `60-180 s` before the event, with optional normalization by local side-window scatter.",
        "",
        "## Time-Shift Test",
        "",
        "If the sign were mainly produced by the exact Earth limb crossing, the raw trend should be strongest and most coherent at zero shift.",
        "Instead, the same sign often persists when pseudo-events are moved by several minutes.",
        "",
        focus.to_string(index=False),
        "",
        "Interpretation: the low-frequency reversal is not behaving like a sharp compact-source edge. It is riding on a local raw-power",
        "trend that is selected by the same event geometry over a broader time interval.",
        "",
        "## Event-Type / Orbital-Phase Test",
        "",
        "Earth disappearance and reappearance events are not random samples of the spacecraft orbit. They occur on opposite limb-crossing",
        "branches, so lower-V sees them at different Moon-facing beam/orbital phases. The table below summarizes the phase and raw trend.",
        "",
        true_phase.to_string(index=False),
        "",
        "This gives a physically plausible reason that the average can be consistent: the event label selects a repeated orbital/beam/sky",
        "state. At low frequency the radiometer is dominated by diffuse structured sky through a broad beam, so a repeatable background",
        "slope can survive stacking and appear as disappearance-up / reappearance-down even when the exact compact Earth occultation would",
        "have the opposite sign.",
        "",
        "## What This Does And Does Not Prove",
        "",
        "- It supports MIE/sampling as a contributor to visual smoothing, not as a sign-reversal mechanism.",
        "- It supports orbital-phase/background-slope selection as the reason the Earth low-frequency average is consistently signed.",
        "- It does not prove a full beam-convolved Galactic forward model yet; that would require using the digitized beam and a sky map.",
        "- It argues against interpreting the low-frequency Earth/Sun profiles as simple positive compact-source occultations.",
        "",
        "## Current Best Explanation",
        "",
        "The low-frequency Earth stack is probably combining two signals:",
        "",
        "1. a real Earth occultation/diffraction edge, which may be smoothed by MIE physics and sparse grouped sampling;",
        "2. a stronger low-frequency background trend selected by Earth limb geometry and lower-V orbital phase.",
        "",
        "The second term can dominate the stack because the diffuse Galactic background is very bright and structured below a few MHz.",
        "The Galaxy is present both before and after, but the beam-weighted sky seen by the Moon-facing antenna is changing during the",
        "event window. Disappearance and reappearance select opposite sweep directions/limb branches, so their average background slopes",
        "can have opposite signs.",
        "",
        "## Generated Plots",
        "",
    ]
    lines.extend(f"- `{path}`" for path in paths)
    (OUT / "earth_lowfreq_background_slope_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dir(OUT)
    write_json(
        OUT / "run_config.json",
        {
            "antenna": ANTENNA,
            "low_frequency_bands": LOW_BANDS,
            "low_frequency_mhz": LOW_FREQS,
            "time_shifts_s": TIME_SHIFTS_S,
            "window_s": WINDOW_S,
            "pre_near_s": PRE_NEAR,
            "post_near_s": POST_NEAR,
            "software_versions": software_versions(),
        },
    )
    clean, events = _read_inputs()
    event_shift = attach_phase_to_shift_table(clean, _read_precomputed_shifts(events))
    shift_summary = summarize_shift_table(event_shift)
    phase_bg = phase_binned_background(clean)
    phase_summary = event_phase_summary(event_shift)

    event_shift.to_csv(OUT / "earth_lowfreq_event_shift_raw_trends.csv", index=False)
    shift_summary.to_csv(OUT / "earth_lowfreq_time_shift_raw_trend_summary.csv", index=False)
    phase_bg.to_csv(OUT / "earth_lowfreq_phase_binned_background_slopes.csv", index=False)
    phase_summary.to_csv(OUT / "earth_lowfreq_event_phase_summary.csv", index=False)

    paths = [plot_time_shift(shift_summary), plot_phase(event_shift, phase_bg)]
    write_report(shift_summary, phase_summary, paths)
    print(OUT / "earth_lowfreq_background_slope_audit_report.md")


if __name__ == "__main__":
    main()
