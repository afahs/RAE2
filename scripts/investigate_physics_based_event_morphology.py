#!/usr/bin/env python
"""Physics-based morphology checks for moving-body source-like classifications.

The existing source-like/anti-template event split is based on the sign of a
near-event pre/post contrast. That is useful bookkeeping, but it is not a
physical detection classifier by itself. This script asks whether the selected
events look like localized occultation steps rather than slow drifts or noisy
segments.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rylevonberg.table_io import read_table


ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "outputs/moving_body_regime_physical_differences_v1/moving_body_event_regime_geometry_table.csv"
POINTS = ROOT / "outputs/moving_body_stack_type_subset_tests_v1/moving_body_stack_points.csv"
SHIFT_CONTRASTS = ROOT / "outputs/earth_lowfreq_profile_sign_diagnostics_v1/earth_event_level_prepost_contrasts_with_shifts.csv"
OUT = ROOT / "outputs/physics_based_event_morphology_v1"

EXPECTED_SIGN = {"disappearance": -1.0, "reappearance": 1.0}
COLORS = {"source_like": "#2ca02c", "anti_template": "#d62728", "neutral": "#7f7f7f"}

PRE_SIDE = (-900.0, -240.0)
POST_SIDE = (240.0, 900.0)
NEAR = (-180.0, 180.0)
WINDOW = (-900.0, 900.0)
REGIME_THRESHOLD = 0.02


def _robust_sigma(vals: np.ndarray) -> float:
    vals = vals[np.isfinite(vals)]
    if vals.size <= 1:
        return np.nan
    center = np.nanmedian(vals)
    sigma = 1.4826 * np.nanmedian(np.abs(vals - center))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.nanstd(vals, ddof=1))
    return sigma if np.isfinite(sigma) and sigma > 0 else np.nan


def _robust_se(vals: pd.Series) -> float:
    arr = pd.to_numeric(vals, errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size <= 1:
        return np.nan
    sigma = _robust_sigma(arr)
    if not np.isfinite(sigma) or sigma <= 0:
        return np.nan
    return float(sigma / np.sqrt(arr.size))


def _classify_regime(values: pd.Series) -> pd.Series:
    vals = pd.to_numeric(values, errors="coerce")
    out = pd.Series("neutral", index=values.index, dtype=object)
    out.loc[vals > REGIME_THRESHOLD] = "source_like"
    out.loc[vals < -REGIME_THRESHOLD] = "anti_template"
    out.loc[~np.isfinite(vals)] = "invalid"
    return out


def _ols_slope(t: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(t) & np.isfinite(y)
    if np.count_nonzero(mask) < 4:
        return np.nan
    tt = t[mask].astype(float)
    yy = y[mask].astype(float)
    if np.nanmax(tt) <= np.nanmin(tt):
        return np.nan
    x = tt - np.nanmean(tt)
    denom = float(np.nansum(x * x))
    if denom <= 0:
        return np.nan
    return float(np.nansum(x * (yy - np.nanmean(yy))) / denom)


def _event_side_slope_table(points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    work = points[points["source_name"].eq("earth")].copy()
    for keys, grp in work.groupby(["event_id", "event_type", "frequency_band", "frequency_mhz"], sort=True):
        event_id, event_type, band, freq = keys
        t = pd.to_numeric(grp["t_rel_sec"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(grp["raw_fractional"], errors="coerce").to_numpy(dtype=float)
        pre = (t >= PRE_SIDE[0]) & (t <= PRE_SIDE[1])
        post = (t >= POST_SIDE[0]) & (t <= POST_SIDE[1])
        near = (t >= NEAR[0]) & (t <= NEAR[1])
        sign = EXPECTED_SIGN[str(event_type)]
        pre_slope = _ols_slope(t[pre], y[pre])
        post_slope = _ols_slope(t[post], y[post])
        near_slope = _ols_slope(t[near], y[near])
        side_slopes = np.asarray([pre_slope, post_slope], dtype=float)
        max_side_slope = float(np.nanmax(np.abs(side_slopes))) if np.isfinite(side_slopes).any() else np.nan
        side_drift_10min = max_side_slope * 600.0 if np.isfinite(max_side_slope) else np.nan
        rows.append(
            {
                "event_id": event_id,
                "event_type": event_type,
                "frequency_band": int(band),
                "frequency_mhz": float(freq),
                "pre_side_slope_frac_per_s": pre_slope,
                "post_side_slope_frac_per_s": post_slope,
                "near_slope_frac_per_s": near_slope,
                "expected_signed_near_slope_frac_per_s": sign * near_slope if np.isfinite(near_slope) else np.nan,
                "max_side_slope_abs_frac_per_s": max_side_slope,
                "side_drift_10min_frac": side_drift_10min,
                "n_pre_side_points": int(np.count_nonzero(pre & np.isfinite(y))),
                "n_post_side_points": int(np.count_nonzero(post & np.isfinite(y))),
                "n_near_points": int(np.count_nonzero(near & np.isfinite(y))),
            }
        )
    return pd.DataFrame(rows)


def _event_bin_points(points: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    meta = events[
        ["event_id", "event_type", "frequency_band", "frequency_mhz", "regime", "source_like_fractional_contrast"]
    ].copy()
    earth = points[points["source_name"].eq("earth")].merge(
        meta,
        on=["event_id", "event_type", "frequency_band", "frequency_mhz"],
        how="inner",
        suffixes=("", "_event"),
    )
    event_bin = (
        earth.groupby(["event_id", "event_type", "frequency_band", "frequency_mhz", "regime", "t_bin_sec"], as_index=False)
        .agg(raw_fractional=("raw_fractional", "median"), n_samples=("raw_fractional", "size"))
    )
    return event_bin


def _stack_profiles(event_bin: pd.DataFrame) -> pd.DataFrame:
    rows = []
    by = ["regime", "event_type", "frequency_band", "frequency_mhz", "t_bin_sec"]
    for keys, grp in event_bin.groupby(by, sort=True):
        vals = pd.to_numeric(grp["raw_fractional"], errors="coerce")
        rows.append(
            {
                **dict(zip(by, keys)),
                "median_raw_fractional": float(np.nanmedian(vals)),
                "err_raw_fractional": _robust_se(vals),
                "n_events": int(grp["event_id"].nunique()),
                "n_points": int(grp["n_samples"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _fit_wls(t: np.ndarray, y: np.ndarray, err: np.ndarray, design: np.ndarray) -> dict[str, object]:
    mask = np.isfinite(t) & np.isfinite(y) & np.all(np.isfinite(design), axis=1)
    if err is not None:
        mask &= np.isfinite(err)
    t = t[mask]
    y = y[mask]
    x = design[mask]
    if y.size <= x.shape[1] + 1:
        return {"ok": False}
    if err is None:
        sigma = np.ones_like(y)
    else:
        sigma = err[mask].astype(float)
        floor = np.nanmedian(sigma[np.isfinite(sigma) & (sigma > 0)]) * 0.25
        if not np.isfinite(floor) or floor <= 0:
            floor = 1.0
        sigma = np.clip(sigma, floor, np.inf)
    xw = x / sigma[:, None]
    yw = y / sigma
    coef, *_ = np.linalg.lstsq(xw, yw, rcond=None)
    resid = y - x @ coef
    chi2 = float(np.nansum((resid / sigma) ** 2))
    n = int(y.size)
    k = int(x.shape[1])
    bic = chi2 + k * np.log(max(n, 2))
    return {"ok": True, "coef": coef, "chi2": chi2, "bic": bic, "n": n, "k": k, "resid": resid}


def _model_diagnostics(stacked: pd.DataFrame) -> pd.DataFrame:
    rows = []
    work = stacked[stacked["regime"].isin(["source_like", "anti_template"])].copy()
    for keys, grp in work.groupby(["regime", "event_type", "frequency_band", "frequency_mhz"], sort=True):
        regime, event_type, band, freq = keys
        grp = grp.sort_values("t_bin_sec")
        t = pd.to_numeric(grp["t_bin_sec"], errors="coerce").to_numpy(dtype=float)
        y_raw = pd.to_numeric(grp["median_raw_fractional"], errors="coerce").to_numpy(dtype=float)
        err = pd.to_numeric(grp["err_raw_fractional"], errors="coerce").to_numpy(dtype=float)
        sign = EXPECTED_SIGN[str(event_type)]
        # In this orientation, a positive occulted source is an upward step at t=0.
        y = sign * y_raw
        tt = t / 900.0
        step = np.where(t >= 0, 0.5, -0.5)
        const = np.ones_like(tt)
        fits = {
            "constant": _fit_wls(t, y, err, const[:, None]),
            "line": _fit_wls(t, y, err, np.column_stack([const, tt])),
            "step": _fit_wls(t, y, err, np.column_stack([const, step])),
            "line_plus_step": _fit_wls(t, y, err, np.column_stack([const, tt, step])),
        }
        if not all(fit.get("ok") for fit in fits.values()):
            continue
        line_coef = fits["line"]["coef"]
        step_coef = fits["step"]["coef"]
        combo_coef = fits["line_plus_step"]["coef"]
        rows.append(
            {
                "regime": regime,
                "event_type": event_type,
                "frequency_band": int(band),
                "frequency_mhz": float(freq),
                "n_bins": int(fits["line_plus_step"]["n"]),
                "n_events": int(grp["n_events"].max()),
                "bic_constant": fits["constant"]["bic"],
                "bic_line": fits["line"]["bic"],
                "bic_step": fits["step"]["bic"],
                "bic_line_plus_step": fits["line_plus_step"]["bic"],
                "delta_bic_line_minus_step": fits["line"]["bic"] - fits["step"]["bic"],
                "delta_bic_line_minus_line_plus_step": fits["line"]["bic"] - fits["line_plus_step"]["bic"],
                "step_only_amplitude_source_oriented": float(step_coef[1]),
                "line_plus_step_amplitude_source_oriented": float(combo_coef[2]),
                "line_slope_source_oriented_per_window": float(line_coef[1]),
                "line_plus_step_slope_source_oriented_per_window": float(combo_coef[1]),
                "trend_to_step_abs": float(abs(combo_coef[1]) / max(abs(combo_coef[2]), 1e-12)),
            }
        )
    diag = pd.DataFrame(rows)
    if diag.empty:
        return diag
    conditions = [
        (diag["line_plus_step_amplitude_source_oriented"] > 0)
        & (diag["delta_bic_line_minus_line_plus_step"] > 6.0)
        & (diag["trend_to_step_abs"] < 2.0),
        (diag["line_plus_step_amplitude_source_oriented"] < 0)
        & (diag["delta_bic_line_minus_line_plus_step"] > 6.0),
        (diag["delta_bic_line_minus_line_plus_step"] <= 2.0)
        & (diag["trend_to_step_abs"] >= 2.0),
    ]
    choices = ["step_like_source", "step_like_anti_template", "trend_dominated_or_not_localized"]
    diag["morphology_class"] = np.select(conditions, choices, default="ambiguous")
    return diag


def _plot_suspect_profile(stacked: pd.DataFrame, diag: pd.DataFrame) -> Path:
    freq = 0.90
    event_type = "disappearance"
    regime = "source_like"
    grp = stacked[
        stacked["regime"].eq(regime)
        & stacked["event_type"].eq(event_type)
        & np.isclose(stacked["frequency_mhz"], freq)
    ].sort_values("t_bin_sec")
    t = grp["t_bin_sec"].to_numpy(dtype=float)
    y = grp["median_raw_fractional"].to_numpy(dtype=float)
    err = grp["err_raw_fractional"].to_numpy(dtype=float)
    sign = EXPECTED_SIGN[event_type]
    q = sign * y
    tt = t / 900.0
    step = np.where(t >= 0, 0.5, -0.5)
    const = np.ones_like(tt)
    line = _fit_wls(t, q, err, np.column_stack([const, tt]))
    step_fit = _fit_wls(t, q, err, np.column_stack([const, step]))
    combo = _fit_wls(t, q, err, np.column_stack([const, tt, step]))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), sharex=True)
    axes[0].errorbar(t / 60.0, y, yerr=err, marker="o", ms=3, lw=1.2, elinewidth=0.7, capsize=1.5, color=COLORS[regime])
    axes[0].axvline(0, color="black", ls="--", lw=0.9)
    axes[0].axhline(0, color="0.6", lw=0.8)
    axes[0].set_title("plotted orientation")
    axes[0].set_ylabel("raw fractional power")
    axes[0].set_xlabel("minutes from predicted event")
    axes[0].grid(alpha=0.2)

    axes[1].errorbar(t / 60.0, q, yerr=err, marker="o", ms=3, lw=0, elinewidth=0.7, capsize=1.5, color="0.35", label="stack")
    if line.get("ok"):
        axes[1].plot(t / 60.0, np.column_stack([const, tt]) @ line["coef"], color="#4c78a8", lw=1.5, label="line")
    if step_fit.get("ok"):
        axes[1].plot(t / 60.0, np.column_stack([const, step]) @ step_fit["coef"], color="#59a14f", lw=1.5, label="fixed step")
    if combo.get("ok"):
        axes[1].plot(t / 60.0, np.column_stack([const, tt, step]) @ combo["coef"], color="#f28e2b", lw=1.5, label="line + step")
    axes[1].axvline(0, color="black", ls="--", lw=0.9)
    axes[1].axhline(0, color="0.6", lw=0.8)
    axes[1].set_title("source-oriented morphology fits")
    axes[1].set_ylabel("source-oriented profile")
    axes[1].set_xlabel("minutes from predicted event")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(alpha=0.2)

    row = diag[
        diag["regime"].eq(regime)
        & diag["event_type"].eq(event_type)
        & np.isclose(diag["frequency_mhz"], freq)
    ]
    text = ""
    if not row.empty:
        r = row.iloc[0]
        text = (
            f"line-step ΔBIC={r['delta_bic_line_minus_line_plus_step']:.1f}, "
            f"A_step={r['line_plus_step_amplitude_source_oriented']:.3f}, "
            f"class={r['morphology_class']}"
        )
    fig.suptitle(f"Earth 0.90 MHz disappearance, current source-like subset\n{text}", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    path = OUT / "earth_090mhz_disappearance_source_like_morphology_fit.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _plot_shift_by_regime(events: pd.DataFrame) -> Path | None:
    if not SHIFT_CONTRASTS.exists():
        return None
    shifts = read_table(SHIFT_CONTRASTS, low_memory=False)
    meta = events[
        ["event_id", "event_type", "frequency_band", "frequency_mhz", "regime"]
    ].copy()
    work = shifts[
        shifts["antenna"].eq("rv2_coarse")
        & shifts["event_type"].eq("disappearance")
        & np.isclose(shifts["frequency_mhz"], 0.90)
    ].merge(meta, on=["event_id", "event_type", "frequency_band", "frequency_mhz"], how="inner")
    work = work[work["regime"].isin(["source_like", "anti_template"])]
    rows = []
    for keys, grp in work.groupby(["regime", "time_shift_s"], sort=True):
        regime, shift = keys
        vals = pd.to_numeric(grp["source_like_signed_contrast_z"], errors="coerce")
        rows.append(
            {
                "regime": regime,
                "time_shift_s": float(shift),
                "median_source_like_contrast_z": float(np.nanmedian(vals)),
                "err_source_like_contrast_z": _robust_se(vals),
                "n_events": int(grp["event_id"].nunique()),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "earth_090mhz_disappearance_shift_by_regime.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.8, 4.5))
    for regime, grp in summary.groupby("regime", sort=True):
        grp = grp.sort_values("time_shift_s")
        ax.errorbar(
            grp["time_shift_s"] / 60.0,
            grp["median_source_like_contrast_z"],
            yerr=grp["err_source_like_contrast_z"],
            marker="o",
            lw=1.4,
            elinewidth=0.8,
            capsize=2,
            color=COLORS[regime],
            label=regime,
        )
    ax.axvline(0, color="black", ls="--", lw=0.9)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xlabel("pseudo-event shift from predicted time (minutes)")
    ax.set_ylabel("median source-like contrast (z units)")
    ax.set_title("Earth 0.90 MHz disappearance: timing localization by current regime")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    path = OUT / "earth_090mhz_disappearance_shift_by_regime.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _plot_morphology_grid(stacked: pd.DataFrame, diag: pd.DataFrame) -> Path:
    source_like_diag = diag[
        diag["regime"].eq("source_like")
        & diag["morphology_class"].eq("step_like_source")
    ][["event_type", "frequency_band", "frequency_mhz"]]
    keep = stacked.merge(
        source_like_diag.assign(morphology_class="step_like_source"),
        on=["event_type", "frequency_band", "frequency_mhz"],
        how="inner",
    )
    freqs = sorted(keep["frequency_mhz"].dropna().unique())
    fig, axes = plt.subplots(len(freqs), 2, figsize=(12, max(9, 1.35 * len(freqs))), sharex=True)
    if len(freqs) == 1:
        axes = np.asarray([axes])
    for i, freq in enumerate(freqs):
        for j, event_type in enumerate(["disappearance", "reappearance"]):
            ax = axes[i, j]
            g = keep[np.isclose(keep["frequency_mhz"], freq) & keep["event_type"].eq(event_type)].sort_values("t_bin_sec")
            if not g.empty:
                ax.errorbar(
                    g["t_bin_sec"] / 60.0,
                    g["median_raw_fractional"],
                    yerr=g["err_raw_fractional"],
                    marker="o",
                    ms=2.8,
                    lw=1.25,
                    elinewidth=0.65,
                    capsize=1.3,
                    color=COLORS["source_like"],
                )
                n = int(g["n_events"].max())
            else:
                n = 0
            ax.axvline(0, color="black", ls="--", lw=0.8)
            ax.axhline(0, color="0.6", lw=0.7)
            ax.set_title(f"{freq:.2f} MHz {event_type} (n={n})", fontsize=9)
            if j == 0:
                ax.set_ylabel("raw fractional power")
            if i == len(freqs) - 1:
                ax.set_xlabel("minutes from predicted event")
            ax.grid(alpha=0.18)
    fig.suptitle(
        "Earth lower V all-frequency grid: groups passing stack-level step-like morphology\n"
        "This is not an event-level throwaway filter; it only shows source-like groups where line+step beats line.",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = OUT / "earth_step_like_source_groups_all_frequency_profile_grid_900s.png"
    fig.savefig(path, dpi=175)
    plt.close(fig)
    return path


def _write_report(
    events: pd.DataFrame,
    model_diag: pd.DataFrame,
    paths: list[Path],
) -> Path:
    suspect = model_diag[
        model_diag["regime"].eq("source_like")
        & model_diag["event_type"].eq("disappearance")
        & np.isclose(model_diag["frequency_mhz"], 0.90)
    ]
    top_cols = [
        "regime",
        "event_type",
        "frequency_mhz",
        "n_events",
        "delta_bic_line_minus_step",
        "delta_bic_line_minus_line_plus_step",
        "line_plus_step_amplitude_source_oriented",
        "trend_to_step_abs",
        "morphology_class",
    ]
    source_like = model_diag[model_diag["regime"].eq("source_like")].sort_values(
        ["event_type", "frequency_mhz"]
    )
    counts = (
        source_like.groupby("morphology_class", as_index=False)
        .agg(n_groups=("frequency_mhz", "size"), total_events=("n_events", "sum"))
        .sort_values("n_groups", ascending=False)
    )
    event_counts = (
        events[events["source_name"].eq("earth")]
        .groupby(["frequency_mhz", "event_type", "regime"], as_index=False)
        .size()
        .pivot_table(index=["frequency_mhz", "event_type"], columns="regime", values="size", fill_value=0)
        .reset_index()
    )
    lines = [
        "# Physics-Based Event Morphology Investigation",
        "",
        "This revisits the current source-like/anti-template split because the Earth 0.90 MHz",
        "disappearance source-like stack does not visually look like a clean disappearance.",
        "",
        "## Key Point",
        "",
        "The current event class is a sign label, not a physical occultation morphology label.",
        "It says whether the median post-near value moved in the expected direction relative",
        "to the median pre-near value. It does not require a localized step, stable before/after",
        "plateaus, or a better fit than a slow trend.",
        "",
        "## Tests Added",
        "",
        "- Within-side plateau slopes: pre-side and post-side slopes are estimated away from the event.",
        "- Stack model comparison: source-oriented stacked profiles are fit with constant, line, fixed step, and line+step models.",
        "- Shifted pseudo-event control for Earth 0.90 MHz disappearance: a true occultation should be strongest near zero shift.",
        "- Step-like morphology grid: shows only source-like groups where adding a fixed step improves a line model by ΔBIC > 6.",
        "",
        "## Earth 0.90 MHz Disappearance, Current Source-Like Subset",
        "",
        suspect[top_cols].to_string(index=False) if not suspect.empty else "No row found.",
        "",
        "## Source-Like Group Morphology Counts",
        "",
        counts.to_string(index=False),
        "",
        "## Existing Event Regime Counts",
        "",
        event_counts.to_string(index=False),
        "",
        "## Source-Like Stack-Level Model Diagnostics",
        "",
        source_like[top_cols].to_string(index=False),
        "",
        "## Interpretation",
        "",
        "The suspect 0.90 MHz disappearance source-like group is better understood as a sign-selected",
        "subset, not necessarily a clean physical disappearance. If the line or line+step model shows a",
        "large trend component, or if shifted pseudo-events retain comparable contrast, the apparent",
        "source-like sign can come from local background trajectory and sampling rather than a localized",
        "lunar occultation.",
        "",
        "For pipeline purposes, the better classification should be two-stage:",
        "",
        "1. Keep the current sign label as a descriptive variable: source-like, anti-template, neutral.",
        "2. Add an independent morphology grade from stack-level and event-level tests: step-like, trend-dominated, timing-nonlocalized, ambiguous.",
        "",
        "Events should not be thrown away solely because they are anti-template or source-like. A more",
        "defensible cut is to downweight or exclude groups/events whose morphology is trend-dominated,",
        "whose shifted controls are comparable to the true event, or whose side-window slopes are large",
        "enough to explain the central contrast.",
        "",
        "## Generated Plots",
        "",
    ]
    lines.extend(f"- `{p}`" for p in paths if p is not None)
    path = OUT / "physics_based_event_morphology_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    events = read_table(EVENTS, low_memory=False)
    points = read_table(POINTS, low_memory=False)
    events["regime"] = _classify_regime(events["source_like_fractional_contrast"])

    side = _event_side_slope_table(points)
    events = events.merge(side, on=["event_id", "event_type", "frequency_band", "frequency_mhz"], how="left")
    central = pd.to_numeric(events["source_like_fractional_contrast"], errors="coerce").abs()
    side_drift = pd.to_numeric(events["side_drift_10min_frac"], errors="coerce")
    events["side_drift_to_central_contrast"] = side_drift / central.replace(0, np.nan)
    events["side_drift_flag"] = events["side_drift_to_central_contrast"] > 1.0

    event_bin = _event_bin_points(points, events)
    stacked = _stack_profiles(event_bin)
    model_diag = _model_diagnostics(stacked)

    events.to_csv(OUT / "earth_event_morphology_metrics.csv", index=False)
    stacked.to_csv(OUT / "earth_regime_stack_profiles_with_errors.csv", index=False)
    model_diag.to_csv(OUT / "earth_stack_model_morphology_diagnostics.csv", index=False)

    paths = [
        _plot_suspect_profile(stacked, model_diag),
        _plot_shift_by_regime(events),
        _plot_morphology_grid(stacked, model_diag),
    ]
    report = _write_report(events, model_diag, paths)
    print(report)
    for path in paths:
        if path is not None:
            print(path)


if __name__ == "__main__":
    main()
