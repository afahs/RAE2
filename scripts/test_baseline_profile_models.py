#!/usr/bin/env python
"""Compare baseline subtraction choices for event-type stacked profiles."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rylevonberg.table_io import read_table

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.detection import baseline_matrix, event_template
from rylevonberg.util import datetime_ns, robust_sigma


CLEAN = ROOT / "outputs/control_survey_earth_sun_postnov1974_v2/01_ingest/cleaned_timeseries.csv"
EARTH_SUN_EVENTS = ROOT / "outputs/control_survey_earth_sun_postnov1974_v2/02_events/predicted_events.csv"
JUPITER_EVENTS = ROOT / "outputs/planetary_confirmation_survey/events/jupiter_predicted_events.csv"
OUT = ROOT / "outputs/baseline_profile_model_tests"


@dataclass(frozen=True)
class Target:
    source_name: str
    frequency_band: int
    frequency_mhz: float
    antenna: str
    window_s: float
    events_path: Path
    label: str


TARGETS = (
    Target("earth", 7, 4.70, "rv2_coarse", 600.0, EARTH_SUN_EVENTS, "Earth 4.70 MHz lower V"),
    Target("sun", 6, 3.93, "rv2_coarse", 900.0, EARTH_SUN_EVENTS, "Sun 3.93 MHz lower V"),
    Target("jupiter", 4, 1.31, "rv2_coarse", 900.0, JUPITER_EVENTS, "Jupiter 1.31 MHz lower V"),
)

MODES = ("linear_all", "constant_all", "sideband_linear", "sideband_constant", "joint_step_linear")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    clean = read_table(CLEAN, parse_dates=["time"], low_memory=False)
    summaries = []
    for target in TARGETS:
        events_all = read_table(target.events_path, parse_dates=["predicted_event_time"], low_memory=False)
        events = events_all[
            events_all["source_name"].astype(str).eq(target.source_name)
            & events_all["frequency_band"].astype(int).eq(target.frequency_band)
            & events_all["antenna"].astype(str).eq(target.antenna)
        ].copy()
        for mode in MODES:
            profile = extract_profile(clean, events, target, mode=mode, bin_s=60.0, sideband_exclusion_s=120.0)
            stem = f"{target.source_name}_{target.frequency_mhz:.2f}mhz_{target.antenna}_{int(target.window_s)}s_{mode}".replace(".", "p")
            profile.to_csv(OUT / f"{stem}_profile.csv", index=False)
            summary = summarize_profile(profile, target, mode)
            summaries.append(summary)
        plot_target_comparison(target)
    summary_df = pd.DataFrame.from_records(summaries)
    summary_df.to_csv(OUT / "baseline_model_comparison_summary.csv", index=False)
    write_report(summary_df)


def extract_profile(
    clean: pd.DataFrame,
    events: pd.DataFrame,
    target: Target,
    mode: str,
    bin_s: float,
    sideband_exclusion_s: float,
) -> pd.DataFrame:
    group = clean[
        clean["frequency_band"].astype(int).eq(target.frequency_band)
        & clean["antenna"].astype(str).eq(target.antenna)
    ].sort_values("time").reset_index(drop=True)
    t_ns = datetime_ns(group["time"])
    rows = []
    half_ns = int(target.window_s * 1e9)
    for _, ev in events.iterrows():
        event_ns = pd.Timestamp(ev["predicted_event_time"]).value
        lo = int(np.searchsorted(t_ns, event_ns - half_ns, side="left"))
        hi = int(np.searchsorted(t_ns, event_ns + half_ns, side="right"))
        if hi <= lo:
            continue
        local = group.iloc[lo:hi]
        rel = (datetime_ns(local["time"]) - event_ns).astype(float) / 1e9
        y = pd.to_numeric(local["power"], errors="coerce").to_numpy(dtype=float)
        valid = local["is_valid"].to_numpy(dtype=bool) if "is_valid" in local.columns else np.ones(len(local), dtype=bool)
        keep = valid & np.isfinite(y) & (np.abs(rel) <= target.window_s)
        if np.count_nonzero(keep) < 8:
            continue
        tr = rel[keep]
        yy = y[keep]
        tmpl = event_template(tr, str(ev["event_type"]))
        yd, sigma, n_baseline = detrend_event(tr, yy, tmpl, mode, sideband_exclusion_s)
        if not np.isfinite(sigma) or sigma <= 0:
            continue
        zn = yd / sigma
        aligned = zn * tmpl
        tbin = np.round(tr / bin_s) * bin_s
        for trel, tb, value, aligned_value, template_value in zip(tr, tbin, zn, aligned, tmpl):
            rows.append(
                {
                    "source_name": target.source_name,
                    "frequency_band": target.frequency_band,
                    "frequency_mhz": target.frequency_mhz,
                    "antenna": target.antenna,
                    "window_s": target.window_s,
                    "mode": mode,
                    "event_type": ev["event_type"],
                    "event_id": ev.get("event_id"),
                    "predicted_event_time": ev["predicted_event_time"],
                    "t_rel_sec": float(trel),
                    "t_bin_sec": float(tb),
                    "profile_value": float(value),
                    "template_aligned_value": float(aligned_value),
                    "template": float(template_value),
                    "n_baseline_samples": int(n_baseline),
                }
            )
    raw = pd.DataFrame.from_records(rows)
    if raw.empty:
        return raw
    out_rows = []
    for keys, grp in raw.groupby(["source_name", "frequency_band", "frequency_mhz", "antenna", "window_s", "mode", "event_type", "t_bin_sec"], sort=True):
        source, band, mhz, ant, window, mode_name, event_type, tbin = keys
        vals = grp["template_aligned_value"].to_numpy(dtype=float)
        out_rows.append(
            {
                "source_name": source,
                "frequency_band": band,
                "frequency_mhz": mhz,
                "antenna": ant,
                "window_s": window,
                "mode": mode_name,
                "event_type": event_type,
                "t_bin_sec": float(tbin),
                "n_samples": int(vals.size),
                "n_events": int(grp["event_id"].nunique()),
                "mean_template_aligned": float(np.mean(vals)),
                "median_template_aligned": float(np.median(vals)),
                "sem_template_aligned": float(np.std(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else np.nan,
                "median_n_baseline_samples": float(np.nanmedian(grp["n_baseline_samples"].to_numpy(dtype=float))),
            }
        )
    return pd.DataFrame.from_records(out_rows)


def detrend_event(tr: np.ndarray, y: np.ndarray, tmpl: np.ndarray, mode: str, sideband_exclusion_s: float) -> tuple[np.ndarray, float, int]:
    if mode == "linear_all":
        fit_mask = np.ones(len(y), dtype=bool)
        B = baseline_matrix(tr[fit_mask], 1)
        beta, *_ = np.linalg.lstsq(B, y[fit_mask], rcond=None)
        baseline = baseline_matrix(tr, 1) @ beta
        resid = y - baseline
    elif mode == "constant_all":
        fit_mask = np.ones(len(y), dtype=bool)
        baseline = np.full_like(y, np.nanmedian(y))
        resid = y - baseline
    elif mode == "sideband_linear":
        fit_mask = np.abs(tr) >= float(sideband_exclusion_s)
        if np.count_nonzero(fit_mask) < 6:
            fit_mask = np.ones(len(y), dtype=bool)
        B = baseline_matrix(tr[fit_mask], 1)
        beta, *_ = np.linalg.lstsq(B, y[fit_mask], rcond=None)
        baseline = baseline_matrix(tr, 1) @ beta
        resid = y - baseline
    elif mode == "sideband_constant":
        fit_mask = np.abs(tr) >= float(sideband_exclusion_s)
        if np.count_nonzero(fit_mask) < 4:
            fit_mask = np.ones(len(y), dtype=bool)
        baseline = np.full_like(y, np.nanmedian(y[fit_mask]))
        resid = y - baseline
    elif mode == "joint_step_linear":
        fit_mask = np.ones(len(y), dtype=bool)
        B = baseline_matrix(tr, 1)
        X = np.column_stack([B, tmpl])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        baseline = B @ beta[:-1]
        resid_for_sigma = y - X @ beta
        resid = y - baseline
        sigma = robust_sigma(resid_for_sigma)
        return resid, float(sigma), int(np.count_nonzero(fit_mask))
    else:
        raise ValueError(f"unknown baseline mode: {mode}")
    sigma = robust_sigma(resid[fit_mask]) if np.count_nonzero(fit_mask) else robust_sigma(resid)
    return resid, float(sigma), int(np.count_nonzero(fit_mask))


def summarize_profile(profile: pd.DataFrame, target: Target, mode: str) -> dict[str, object]:
    base = {
        "source_name": target.source_name,
        "frequency_mhz": target.frequency_mhz,
        "antenna": target.antenna,
        "window_s": target.window_s,
        "mode": mode,
    }
    if profile.empty:
        return {**base, "n_events": 0}
    center = profile[np.abs(profile["t_bin_sec"]) <= 120.0]
    far = profile[np.abs(profile["t_bin_sec"]) >= 0.5 * target.window_s]
    all_vals = profile["mean_template_aligned"].to_numpy(dtype=float)
    center_peak = float(center["mean_template_aligned"].max()) if not center.empty else np.nan
    center_abs_peak = float(center["mean_template_aligned"].abs().max()) if not center.empty else np.nan
    far_abs_peak = float(far["mean_template_aligned"].abs().max()) if not far.empty else np.nan
    far_lobe_ratio = float(far_abs_peak / center_abs_peak) if np.isfinite(center_abs_peak) and center_abs_peak > 0 else np.nan
    negative_fraction = float(np.mean(all_vals < 0.0)) if all_vals.size else np.nan
    return {
        **base,
        "n_events": int(profile["n_events"].max()),
        "center_peak": center_peak,
        "center_abs_peak": center_abs_peak,
        "far_abs_peak": far_abs_peak,
        "far_lobe_ratio": far_lobe_ratio,
        "negative_bin_fraction": negative_fraction,
        "median_baseline_samples": float(profile["median_n_baseline_samples"].median()),
    }


def plot_target_comparison(target: Target) -> None:
    fig, axes = plt.subplots(len(MODES), 1, figsize=(8.5, 10.5), sharex=True, sharey=True)
    colors = {"disappearance": "#4c78a8", "reappearance": "#d95f02"}
    for ax, mode in zip(axes, MODES):
        stem = f"{target.source_name}_{target.frequency_mhz:.2f}mhz_{target.antenna}_{int(target.window_s)}s_{mode}".replace(".", "p")
        profile = read_table(OUT / f"{stem}_profile.csv")
        for event_type, grp in profile.groupby("event_type", sort=True):
            grp = grp.sort_values("t_bin_sec")
            x = grp["t_bin_sec"] / 60.0
            y = grp["mean_template_aligned"]
            sem = grp["sem_template_aligned"].fillna(0.0)
            ax.plot(x, y, lw=1.6, color=colors.get(str(event_type)), label=str(event_type))
            ax.fill_between(x, y - sem, y + sem, color=colors.get(str(event_type)), alpha=0.15, linewidth=0)
        ax.axvline(0.0, color="black", lw=0.9)
        ax.axhline(0.0, color="0.35", lw=0.8)
        ax.set_ylabel(mode.replace("_", "\n"), fontsize=8)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title(f"{target.label}: event-type profile under alternate baseline models")
    axes[-1].set_xlabel("Relative time from predicted event (min)")
    fig.text(0.01, 0.5, "Template-aligned normalized residual", va="center", rotation="vertical")
    fig.tight_layout(rect=(0.04, 0, 1, 1))
    fig.savefig(OUT / f"{target.source_name}_baseline_model_event_type_comparison.png", dpi=180)
    plt.close(fig)


def write_report(summary: pd.DataFrame) -> None:
    lines = [
        "# Baseline Model Profile Comparison",
        "",
        "Purpose: test whether event-type profile crossover depends on the local background subtraction model.",
        "",
        "Modes:",
        "",
        "- `linear_all`: current-style linear baseline fit over the full event window.",
        "- `constant_all`: subtract one robust constant level from the full event window.",
        "- `sideband_linear`: fit a linear baseline only outside |t| < 120 s, then apply it across the event.",
        "- `sideband_constant`: subtract the sideband median outside |t| < 120 s.",
        "- `joint_step_linear`: fit linear baseline and occultation step jointly, then subtract only the fitted baseline.",
        "",
        "Diagnostics:",
        "",
        "- `center_abs_peak`: strongest absolute stacked response within |t| <= 120 s.",
        "- `far_abs_peak`: strongest absolute stacked response outside half the event window.",
        "- `far_lobe_ratio`: far_abs_peak / center_abs_peak. Larger values mean more crossover/ringing away from the event.",
        "- `negative_bin_fraction`: fraction of stacked profile bins below zero after template alignment.",
        "",
        "Summary table:",
        "",
        _markdown_table(summary),
        "",
        "Plots:",
        "",
    ]
    for target in TARGETS:
        lines.append(f"- `{OUT / f'{target.source_name}_baseline_model_event_type_comparison.png'}`")
    (OUT / "baseline_model_comparison_report.md").write_text("\n".join(lines) + "\n")


def _markdown_table(df: pd.DataFrame) -> str:
    cols = ["source_name", "mode", "n_events", "center_abs_peak", "far_lobe_ratio", "negative_bin_fraction", "median_baseline_samples"]
    work = df[cols].copy()
    for col in ["center_abs_peak", "far_lobe_ratio", "negative_bin_fraction", "median_baseline_samples"]:
        work[col] = pd.to_numeric(work[col], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.3g}")
    widths = {col: max(len(col), *(len(str(v)) for v in work[col])) for col in cols}
    lines = [
        "| " + " | ".join(col.ljust(widths[col]) for col in cols) + " |",
        "| " + " | ".join("-" * widths[col] for col in cols) + " |",
    ]
    for _, row in work.iterrows():
        lines.append("| " + " | ".join(str(row[col]).ljust(widths[col]) for col in cols) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
