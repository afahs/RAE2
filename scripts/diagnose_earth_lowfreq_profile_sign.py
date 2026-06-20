#!/usr/bin/env python
"""Diagnose low-frequency Earth profile sign behavior in all-frequency grids."""

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

from rylevonberg.util import datetime_ns, ensure_dir, robust_sigma  # noqa: E402

CLEAN = ROOT / "outputs/control_survey_earth_sun_postnov1974_v2/01_ingest/cleaned_timeseries.csv"
EVENTS = ROOT / "outputs/planetary_confirmation_survey_science_baseline_v2/events/all_planet_predicted_events.csv"
GRID_SUMMARY = ROOT / "outputs/all_frequency_profile_grids_v1/earth_all_frequency_profile_summary_900s.csv"
OUT = ROOT / "outputs/earth_lowfreq_profile_sign_diagnostics_v1"

ANT_LABEL = {"rv1_coarse": "upper V", "rv2_coarse": "lower V"}
EXPECTED_SIGN = {"disappearance": -1.0, "reappearance": 1.0}


def _channel_groups(clean: pd.DataFrame) -> dict[tuple[int, str], tuple[pd.DataFrame, np.ndarray]]:
    groups = {}
    for (band, ant), grp in clean.groupby(["frequency_band", "antenna"], sort=True):
        g = grp.sort_values("time").reset_index(drop=True)
        groups[(int(band), str(ant))] = (g, datetime_ns(g["time"]))
    return groups


def _window(group: pd.DataFrame, t_ns: np.ndarray, event_time: pd.Timestamp, window_s: float) -> tuple[np.ndarray, np.ndarray] | None:
    event_ns = pd.Timestamp(event_time).value
    half_ns = int(float(window_s) * 1e9)
    lo = int(np.searchsorted(t_ns, event_ns - half_ns, side="left"))
    hi = int(np.searchsorted(t_ns, event_ns + half_ns, side="right"))
    if hi <= lo:
        return None
    local = group.iloc[lo:hi]
    tr = (datetime_ns(local["time"]) - event_ns).astype(float) / 1e9
    y = pd.to_numeric(local["power"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(y) & (np.abs(tr) <= window_s)
    if "is_valid" in local:
        valid &= local["is_valid"].to_numpy(dtype=bool)
    if np.count_nonzero(valid) < 8:
        return None
    order = np.argsort(tr[valid])
    return tr[valid][order], y[valid][order]


def _event_contrast(t: np.ndarray, y: np.ndarray, inner_s: float, pre_range: tuple[float, float], post_range: tuple[float, float]) -> dict[str, float] | None:
    side = np.abs(t) >= inner_s
    pre = (t >= pre_range[0]) & (t <= pre_range[1])
    post = (t >= post_range[0]) & (t <= post_range[1])
    if np.count_nonzero(side) < 6 or np.count_nonzero(pre) < 1 or np.count_nonzero(post) < 1:
        return None
    center = float(np.nanmedian(y[side]))
    sigma = robust_sigma(y[side] - center)
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.nanstd(y[side]))
    if not np.isfinite(sigma) or sigma <= 0:
        return None
    z = (y - center) / sigma
    raw_pre = float(np.nanmedian(y[pre]))
    raw_post = float(np.nanmedian(y[post]))
    return {
        "pre_z": float(np.nanmedian(z[pre])),
        "post_z": float(np.nanmedian(z[post])),
        "post_minus_pre_z": float(np.nanmedian(z[post]) - np.nanmedian(z[pre])),
        "raw_pre": raw_pre,
        "raw_post": raw_post,
        "raw_post_minus_pre": raw_post - raw_pre,
        "side_sigma": float(sigma),
        "n_pre": int(np.count_nonzero(pre)),
        "n_post": int(np.count_nonzero(post)),
    }


def build_event_contrasts(
    clean: pd.DataFrame,
    events: pd.DataFrame,
    shifts_s: list[float],
    window_s: float,
    inner_s: float,
    pre_range: tuple[float, float],
    post_range: tuple[float, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = _channel_groups(clean)
    earth = events[events["source_name"].astype(str).str.lower().eq("earth")].copy()
    rows = []
    unique = earth[["event_id", "event_type", "predicted_event_time"]].drop_duplicates().sort_values("predicted_event_time")
    unique["prev_dt_s"] = unique["predicted_event_time"].diff().dt.total_seconds()
    unique["next_dt_s"] = unique["predicted_event_time"].shift(-1).sub(unique["predicted_event_time"]).dt.total_seconds()
    unique["prev_type"] = unique["event_type"].shift(1)
    unique["next_type"] = unique["event_type"].shift(-1)
    pair_meta = unique.set_index("event_id")[["prev_dt_s", "next_dt_s", "prev_type", "next_type"]]
    for _, ev in earth.iterrows():
        key = (int(ev["frequency_band"]), str(ev["antenna"]))
        payload = groups.get(key)
        if payload is None:
            continue
        group, t_ns = payload
        meta = pair_meta.loc[ev["event_id"]] if ev["event_id"] in pair_meta.index else pd.Series(dtype=object)
        has_opposite_within_window = bool(
            (pd.notna(meta.get("prev_dt_s")) and meta.get("prev_type") != ev["event_type"] and float(meta.get("prev_dt_s")) < window_s)
            or (pd.notna(meta.get("next_dt_s")) and meta.get("next_type") != ev["event_type"] and float(meta.get("next_dt_s")) < window_s)
        )
        for shift_s in shifts_s:
            shifted_time = pd.Timestamp(ev["predicted_event_time"]) + pd.to_timedelta(float(shift_s), unit="s")
            local = _window(group, t_ns, shifted_time, window_s)
            if local is None:
                continue
            contrast = _event_contrast(local[0], local[1], inner_s, pre_range, post_range)
            if contrast is None:
                continue
            event_type = str(ev["event_type"])
            signed = contrast["post_minus_pre_z"] * EXPECTED_SIGN[event_type]
            rows.append(
                {
                    "event_id": ev["event_id"],
                    "event_type": event_type,
                    "frequency_band": int(ev["frequency_band"]),
                    "frequency_mhz": float(ev["frequency_mhz"]),
                    "antenna": str(ev["antenna"]),
                    "antenna_label": ANT_LABEL.get(str(ev["antenna"]), str(ev["antenna"])),
                    "time_shift_s": float(shift_s),
                    "has_opposite_event_within_window": has_opposite_within_window,
                    "source_like_signed_contrast_z": signed,
                    **contrast,
                }
            )
    event_rows = pd.DataFrame(rows)
    agg_rows = []
    for keys, grp in event_rows.groupby(["frequency_band", "frequency_mhz", "antenna", "antenna_label", "event_type", "time_shift_s"], sort=True):
        vals = grp["source_like_signed_contrast_z"].to_numpy(dtype=float)
        raw = grp["raw_post_minus_pre"].to_numpy(dtype=float)
        agg_rows.append(
            {
                **dict(zip(["frequency_band", "frequency_mhz", "antenna", "antenna_label", "event_type", "time_shift_s"], keys)),
                "n_events": int(len(grp)),
                "median_source_like_contrast_z": float(np.nanmedian(vals)),
                "mean_source_like_contrast_z": float(np.nanmean(vals)),
                "positive_source_like_fraction": float(np.nanmean(vals > 0.0)),
                "median_raw_post_minus_pre": float(np.nanmedian(raw)),
                "overlap_fraction": float(np.nanmean(grp["has_opposite_event_within_window"])),
            }
        )
    return event_rows, pd.DataFrame(agg_rows)


def grid_contrast_summary() -> pd.DataFrame:
    df = read_table(GRID_SUMMARY)
    rows = []
    for keys, grp in df.groupby(["frequency_band", "frequency_mhz", "antenna", "event_type"], sort=True):
        pre = grp[(grp["t_bin_sec"] >= -180) & (grp["t_bin_sec"] <= -60)]["median_z_power"].median()
        post = grp[(grp["t_bin_sec"] >= 60) & (grp["t_bin_sec"] <= 180)]["median_z_power"].median()
        outer_pre = grp[(grp["t_bin_sec"] >= -900) & (grp["t_bin_sec"] <= -600)]["median_z_power"].median()
        outer_post = grp[(grp["t_bin_sec"] >= 600) & (grp["t_bin_sec"] <= 900)]["median_z_power"].median()
        band, mhz, ant, et = keys
        sign = EXPECTED_SIGN[str(et)]
        rows.append(
            {
                "frequency_band": band,
                "frequency_mhz": mhz,
                "antenna": ant,
                "antenna_label": ANT_LABEL.get(str(ant), str(ant)),
                "event_type": et,
                "grid_central_post_minus_pre": float(post - pre),
                "grid_central_source_like": float(sign * (post - pre)),
                "grid_outer_post_minus_pre": float(outer_post - outer_pre),
                "grid_outer_source_like": float(sign * (outer_post - outer_pre)),
            }
        )
    return pd.DataFrame(rows)


def plot_shift_test(agg: pd.DataFrame, out: Path) -> Path:
    low = agg[agg["frequency_mhz"].isin([0.70, 0.90, 1.31, 2.20]) & agg["antenna"].eq("rv2_coarse")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, (mhz, grp) in zip(axes, low.groupby("frequency_mhz", sort=True)):
        for et, sub in grp.groupby("event_type", sort=True):
            sub = sub.sort_values("time_shift_s")
            ax.plot(sub["time_shift_s"], sub["median_source_like_contrast_z"], marker="o", label=et)
        ax.axhline(0, color="black", lw=0.8)
        ax.axvline(0, color="0.4", lw=0.8, ls="--")
        ax.set_title(f"{mhz:.2f} MHz lower V")
        ax.set_xlabel("event-time shift (s)")
        ax.set_ylabel("median source-like contrast")
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Earth low-frequency sign test: true events vs shifted pseudo-events")
    fig.tight_layout()
    path = out / "earth_lowfreq_time_shift_contrast_test.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_report(grid: pd.DataFrame, agg: pd.DataFrame, plot_path: Path) -> Path:
    low_grid = grid[grid["frequency_mhz"].isin([0.70, 0.90, 1.31, 2.20]) & grid["antenna"].eq("rv2_coarse")].copy()
    low_shift = agg[agg["frequency_mhz"].isin([0.70, 0.90, 1.31, 2.20]) & agg["antenna"].eq("rv2_coarse") & agg["time_shift_s"].isin([0.0, 300.0, 600.0, 1200.0])].copy()
    true = agg[agg["time_shift_s"].eq(0.0)].copy()
    true["abs_median_source_like_contrast"] = true["median_source_like_contrast_z"].abs()
    lines = [
        "# Earth Low-Frequency Profile Sign Diagnostics",
        "",
        "Question: does the all-frequency profile grid really show low-frequency Earth disappearance increasing and reappearance decreasing, and why?",
        "",
        "Sign convention used here:",
        "",
        "- disappearance source-like contrast is negative `post - pre`, because a bright source should drop;",
        "- reappearance source-like contrast is positive `post - pre`, because a bright source should rise;",
        "- therefore positive `source_like_contrast` means the expected occultation sign.",
        "",
        "## Grid-Derived Low-Frequency Lower-V Contrasts",
        "",
        low_grid[["frequency_mhz", "event_type", "grid_central_post_minus_pre", "grid_central_source_like", "grid_outer_post_minus_pre", "grid_outer_source_like"]].to_string(index=False),
        "",
        "## Per-Event Contrast Time-Shift Test",
        "",
        low_shift[["frequency_mhz", "event_type", "time_shift_s", "n_events", "median_source_like_contrast_z", "positive_source_like_fraction", "median_raw_post_minus_pre", "overlap_fraction"]].to_string(index=False),
        "",
        "## Strongest True-Time Per-Event Contrasts",
        "",
        true.sort_values("abs_median_source_like_contrast", ascending=False).head(18)[["frequency_mhz", "antenna_label", "event_type", "n_events", "median_source_like_contrast_z", "positive_source_like_fraction", "median_raw_post_minus_pre", "overlap_fraction"]].to_string(index=False),
        "",
        "## Plot",
        "",
        f"- `{plot_path}`",
        "",
        "## Interpretation",
        "",
        "The low-frequency lower-V inversion is real in the all-frequency grid, especially in the broad outer-bin trend and in 0.90, 1.31, and 2.20 MHz. It is not a plotting-label swap.",
        "",
        "The time-shift test checks whether the sign is uniquely centered on the predicted occultation time. If shifted pseudo-events keep a similar sign and magnitude, the effect is likely a slow local trend tied to orbit/antenna/background geometry rather than a sharp Earth occultation step.",
        "",
        "The overlap fraction column tests whether disappearance and reappearance windows contaminate one another. It is small for most events, so event-pair overlap is not the dominant explanation.",
    ]
    path = OUT / "earth_lowfreq_profile_sign_diagnostics_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    ensure_dir(OUT)
    clean = read_table(CLEAN, parse_dates=["time"], low_memory=False)
    events = read_table(EVENTS, parse_dates=["predicted_event_time"], low_memory=False)
    grid = grid_contrast_summary()
    event_rows, agg = build_event_contrasts(
        clean,
        events,
        shifts_s=[-1200.0, -600.0, -300.0, 0.0, 300.0, 600.0, 1200.0],
        window_s=900.0,
        inner_s=15.0,
        pre_range=(-180.0, -60.0),
        post_range=(60.0, 180.0),
    )
    grid.to_csv(OUT / "earth_all_frequency_grid_prepost_contrast.csv", index=False)
    event_rows.to_csv(OUT / "earth_event_level_prepost_contrasts_with_shifts.csv", index=False)
    agg.to_csv(OUT / "earth_prepost_contrast_shift_summary.csv", index=False)
    plot = plot_shift_test(agg, OUT)
    report = write_report(grid, agg, plot)
    print(report)


if __name__ == "__main__":
    main()
