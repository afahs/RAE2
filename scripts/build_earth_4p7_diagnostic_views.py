#!/usr/bin/env python
"""Build Earth 4.70 MHz lower-V baseline diagnostic views."""

from __future__ import annotations

import sys
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
EVENTS = ROOT / "outputs/control_survey_earth_sun_postnov1974_v2/02_events/stack_input_events.csv"
OUT = ROOT / "outputs/earth_4p7_profile_crossover_diagnostics/four_view_diagnostics"

SOURCE = "earth"
BAND = 7
FREQUENCY_MHZ = 4.70
ANTENNA = "rv2_coarse"
ANTENNA_LABEL = "lower V"
WINDOW_S = 600.0
BIN_S = 60.0
SIDEBAND_EXCLUSION_S = 120.0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    clean = read_table(CLEAN, parse_dates=["time"], low_memory=False)
    events = read_table(EVENTS, parse_dates=["predicted_event_time"], low_memory=False)
    events = events[
        events["source_name"].astype(str).str.lower().eq(SOURCE)
        & events["frequency_band"].astype(int).eq(BAND)
        & events["antenna"].astype(str).eq(ANTENNA)
    ].sort_values("predicted_event_time").reset_index(drop=True)

    group = clean[
        clean["frequency_band"].astype(int).eq(BAND)
        & clean["antenna"].astype(str).eq(ANTENNA)
    ].sort_values("time").reset_index(drop=True)
    group_ns = datetime_ns(group["time"])

    raw_rows, profile_rows, fit_rows = [], [], []
    for _, event in events.iterrows():
        local = _event_window(group, group_ns, event)
        if local is None:
            continue
        tr, y = local
        tmpl = event_template(tr, str(event["event_type"]))
        if tr.size < 8:
            continue
        for mode in ("sideband_linear", "joint_step_linear", "pre_event_anchor"):
            profile, baseline, model, sigma, n_base, amp = _profile_for_mode(
                tr,
                y,
                tmpl,
                mode,
                str(event["event_type"]),
            )
            if not np.isfinite(sigma) or sigma <= 0:
                continue
            tbin = np.round(tr / BIN_S) * BIN_S
            fit_rows.append(
                {
                    **_event_meta(event),
                    "mode": mode,
                    "n_samples": int(tr.size),
                    "n_baseline_samples": int(n_base),
                    "sigma": float(sigma),
                    "amplitude": float(amp) if np.isfinite(amp) else np.nan,
                    "abs_amplitude_over_sigma": float(abs(amp) / sigma) if np.isfinite(amp) and sigma > 0 else np.nan,
                }
            )
            for trel, tb, yy, base, mod, prof, tv in zip(tr, tbin, y, baseline, model, profile, tmpl):
                profile_rows.append(
                    {
                        **_event_meta(event),
                        "mode": mode,
                        "t_rel_sec": float(trel),
                        "t_bin_sec": float(tb),
                        "power": float(yy),
                        "baseline": float(base),
                        "model": float(mod),
                        "profile_value": float(prof),
                        "template": float(tv),
                        "sigma": float(sigma),
                    }
                )
        for trel, yy in zip(tr, y):
            raw_rows.append({**_event_meta(event), "t_rel_sec": float(trel), "power": float(yy)})

    raw = pd.DataFrame.from_records(raw_rows)
    profiles = pd.DataFrame.from_records(profile_rows)
    fits = pd.DataFrame.from_records(fit_rows)
    raw.to_csv(OUT / "earth_4p70mhz_lower_v_raw_event_samples.csv", index=False)
    profiles.to_csv(OUT / "earth_4p70mhz_lower_v_profile_samples.csv", index=False)
    fits.to_csv(OUT / "earth_4p70mhz_lower_v_event_fit_summary.csv", index=False)

    raw_stack = _stack_raw(raw)
    raw_stack.to_csv(OUT / "earth_4p70mhz_lower_v_raw_power_stack.csv", index=False)
    stacked = _stack_profiles(profiles)
    stacked.to_csv(OUT / "earth_4p70mhz_lower_v_residual_stack_by_mode.csv", index=False)

    _plot_raw_model_examples(group, group_ns, events, fits)
    _plot_raw_power_stack(raw_stack)
    for mode in ("sideband_linear", "joint_step_linear", "pre_event_anchor"):
        _plot_profile(
            stacked[stacked["mode"].eq(mode)],
            OUT / f"earth_4p70mhz_lower_v_{mode}_event_type_profile.png",
            mode,
        )
    _plot_four_view_summary(raw_stack, stacked)
    _write_report(events, fits, raw_stack, stacked)


def _event_meta(event: pd.Series) -> dict[str, object]:
    return {
        "event_id": event.get("event_id"),
        "source_name": SOURCE,
        "frequency_band": BAND,
        "frequency_mhz": FREQUENCY_MHZ,
        "antenna": ANTENNA,
        "antenna_label": ANTENNA_LABEL,
        "event_type": event["event_type"],
        "predicted_event_time": event["predicted_event_time"],
    }


def _event_window(group: pd.DataFrame, group_ns: np.ndarray, event: pd.Series) -> tuple[np.ndarray, np.ndarray] | None:
    event_ns = pd.Timestamp(event["predicted_event_time"]).value
    half_ns = int(WINDOW_S * 1e9)
    lo = int(np.searchsorted(group_ns, event_ns - half_ns, side="left"))
    hi = int(np.searchsorted(group_ns, event_ns + half_ns, side="right"))
    if hi <= lo:
        return None
    local = group.iloc[lo:hi]
    tr = (datetime_ns(local["time"]) - event_ns).astype(float) / 1e9
    y = pd.to_numeric(local["power"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(y) & (np.abs(tr) <= WINDOW_S)
    if "is_valid" in local.columns:
        valid &= local["is_valid"].to_numpy(dtype=bool)
    if np.count_nonzero(valid) < 8:
        return None
    tr = tr[valid]
    y = y[valid]
    order = np.argsort(tr)
    return tr[order], y[order]


def _profile_for_mode(
    tr: np.ndarray,
    y: np.ndarray,
    tmpl: np.ndarray,
    mode: str,
    event_type: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int, float]:
    if mode == "sideband_linear":
        fit_mask = np.abs(tr) >= SIDEBAND_EXCLUSION_S
        if np.count_nonzero(fit_mask) < 6:
            fit_mask = np.ones(len(y), dtype=bool)
        B_fit = baseline_matrix(tr[fit_mask], 1)
        beta, *_ = np.linalg.lstsq(B_fit, y[fit_mask], rcond=None)
        baseline = baseline_matrix(tr, 1) @ beta
        amp = _fit_amp_given_baseline(y, baseline, tmpl)
        model = baseline + amp * tmpl
        resid_for_sigma = y[fit_mask] - baseline[fit_mask]
        sigma = robust_sigma(resid_for_sigma)
        profile = (y - baseline) / sigma if np.isfinite(sigma) and sigma > 0 else y * np.nan
        return profile, baseline, model, float(sigma), int(np.count_nonzero(fit_mask)), float(amp)

    if mode == "joint_step_linear":
        fit_mask = np.ones(len(y), dtype=bool)
        B = baseline_matrix(tr, 1)
        X = np.column_stack([B, tmpl])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        baseline = B @ beta[:-1]
        amp = float(beta[-1])
        model = X @ beta
        resid_for_sigma = y - model
        sigma = robust_sigma(resid_for_sigma)
        profile = (y - baseline) / sigma if np.isfinite(sigma) and sigma > 0 else y * np.nan
        return profile, baseline, model, float(sigma), int(np.count_nonzero(fit_mask)), amp

    if mode == "pre_event_anchor":
        fit_mask = tr <= -SIDEBAND_EXCLUSION_S
        if np.count_nonzero(fit_mask) < 4:
            fit_mask = tr < 0
        if np.count_nonzero(fit_mask) < 4:
            fit_mask = np.ones(len(y), dtype=bool)
        baseline_level = float(np.nanmedian(y[fit_mask]))
        baseline = np.full_like(y, baseline_level)
        amp = _fit_amp_given_baseline(y, baseline, tmpl)
        model = baseline + amp * tmpl
        resid_for_sigma = y[fit_mask] - baseline[fit_mask]
        sigma = robust_sigma(resid_for_sigma)
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = robust_sigma(y - model)
        profile = (y - baseline) / sigma if np.isfinite(sigma) and sigma > 0 else y * np.nan
        return profile, baseline, model, float(sigma), int(np.count_nonzero(fit_mask)), float(amp)

    raise ValueError(mode)


def _fit_amp_given_baseline(y: np.ndarray, baseline: np.ndarray, tmpl: np.ndarray) -> float:
    denom = float(np.dot(tmpl, tmpl))
    if denom <= 0:
        return np.nan
    return float(np.dot(y - baseline, tmpl) / denom)


def _stack_raw(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw
    work = raw.copy()
    work["t_bin_sec"] = np.round(work["t_rel_sec"] / BIN_S) * BIN_S
    rows = []
    for keys, grp in work.groupby(["event_type", "t_bin_sec"], sort=True):
        event_type, tbin = keys
        vals = grp["power"].to_numpy(dtype=float)
        rows.append(
            {
                "event_type": event_type,
                "t_bin_sec": float(tbin),
                "n_samples": int(vals.size),
                "n_events": int(grp["event_id"].nunique()),
                "mean_power": float(np.nanmean(vals)),
                "median_power": float(np.nanmedian(vals)),
                "sem_power": float(np.nanstd(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else np.nan,
            }
        )
    return pd.DataFrame.from_records(rows)


def _stack_profiles(profiles: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, grp in profiles.groupby(["mode", "event_type", "t_bin_sec"], sort=True):
        mode, event_type, tbin = keys
        vals = grp["profile_value"].to_numpy(dtype=float)
        rows.append(
            {
                "mode": mode,
                "event_type": event_type,
                "t_bin_sec": float(tbin),
                "n_samples": int(vals.size),
                "n_events": int(grp["event_id"].nunique()),
                "mean": float(np.nanmean(vals)),
                "median": float(np.nanmedian(vals)),
                "sem": float(np.nanstd(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else np.nan,
            }
        )
    return pd.DataFrame.from_records(rows)


def _plot_raw_model_examples(group: pd.DataFrame, group_ns: np.ndarray, events: pd.DataFrame, fits: pd.DataFrame) -> None:
    ranked = (
        fits[fits["mode"].eq("joint_step_linear")]
        .sort_values("abs_amplitude_over_sigma", ascending=False)
        .groupby("event_type", sort=True)
        .head(3)
    )
    selected = events[events["event_id"].isin(ranked["event_id"])].copy()
    selected = selected.merge(ranked[["event_id", "abs_amplitude_over_sigma"]], on="event_id", how="left")
    selected = selected.sort_values(["event_type", "abs_amplitude_over_sigma"], ascending=[True, False]).head(6)
    fig, axes = plt.subplots(3, 2, figsize=(13.5, 10.5), sharex=True, constrained_layout=True)
    axes = axes.ravel()
    for ax, (_, event) in zip(axes, selected.iterrows()):
        local = _event_window(group, group_ns, event)
        if local is None:
            ax.axis("off")
            continue
        tr, y = local
        tmpl = event_template(tr, str(event["event_type"]))
        _profile, baseline, model, _sigma, _n, amp = _profile_for_mode(tr, y, tmpl, "joint_step_linear", str(event["event_type"]))
        ax.plot(tr / 60.0, y, ".", color="0.65", ms=3, label="raw valid samples")
        ax.plot(tr / 60.0, baseline, "--", color="#ff7f0e", lw=1.6, label="joint-fit baseline")
        ax.plot(tr / 60.0, model, "-", color="#d62728", lw=1.8, label="baseline + step")
        ax.axvline(0.0, color="black", lw=0.9)
        ax.set_title(f"{event['event_type']} {pd.Timestamp(event['predicted_event_time']).date()} A={amp:.3g}")
        ax.set_ylabel("Raw power")
    for ax in axes[len(selected) :]:
        ax.axis("off")
    axes[0].legend(fontsize=8)
    for ax in axes[-2:]:
        ax.set_xlabel("Relative time from predicted event (min)")
    fig.suptitle(f"Earth {FREQUENCY_MHZ:.2f} MHz {ANTENNA_LABEL}: original power with joint baseline+step model")
    fig.savefig(OUT / "earth_4p70mhz_lower_v_raw_power_model_overlays.png", dpi=180)
    plt.close(fig)


def _plot_raw_power_stack(raw_stack: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)
    colors = {"disappearance": "#4c78a8", "reappearance": "#d95f02"}
    for event_type, grp in raw_stack.groupby("event_type", sort=True):
        grp = grp.sort_values("t_bin_sec")
        x = grp["t_bin_sec"] / 60.0
        y = grp["mean_power"]
        sem = grp["sem_power"].fillna(0.0)
        ax.plot(x, y, lw=1.8, color=colors.get(str(event_type)), label=str(event_type))
        ax.fill_between(x, y - sem, y + sem, alpha=0.15, color=colors.get(str(event_type)), linewidth=0)
    ax.axvline(0.0, color="black", lw=0.9)
    ax.set_xlabel("Relative time from predicted event (min)")
    ax.set_ylabel("Mean original raw power")
    ax.set_title(f"Earth {FREQUENCY_MHZ:.2f} MHz {ANTENNA_LABEL}: raw power stack")
    ax.legend()
    fig.savefig(OUT / "earth_4p70mhz_lower_v_raw_power_stack.png", dpi=180)
    plt.close(fig)


def _plot_profile(profile: pd.DataFrame, path: Path, mode: str) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)
    colors = {"disappearance": "#4c78a8", "reappearance": "#d95f02"}
    for event_type, grp in profile.groupby("event_type", sort=True):
        grp = grp.sort_values("t_bin_sec")
        x = grp["t_bin_sec"] / 60.0
        y = grp["mean"]
        sem = grp["sem"].fillna(0.0)
        ax.plot(x, y, lw=1.8, color=colors.get(str(event_type)), label=str(event_type))
        ax.fill_between(x, y - sem, y + sem, alpha=0.15, color=colors.get(str(event_type)), linewidth=0)
    ax.axvline(0.0, color="black", lw=0.9)
    ax.axhline(0.0, color="0.35", lw=0.8)
    ax.set_xlabel("Relative time from predicted event (min)")
    ax.set_ylabel("Mean normalized residual")
    ax.set_title(f"Earth {FREQUENCY_MHZ:.2f} MHz {ANTENNA_LABEL}: {mode.replace('_', ' ')}")
    ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_four_view_summary(raw_stack: pd.DataFrame, stacked: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0), constrained_layout=True)
    colors = {"disappearance": "#4c78a8", "reappearance": "#d95f02"}

    ax = axes[0, 0]
    for event_type, grp in raw_stack.groupby("event_type", sort=True):
        grp = grp.sort_values("t_bin_sec")
        ax.plot(grp["t_bin_sec"] / 60.0, grp["mean_power"], lw=1.7, color=colors.get(str(event_type)), label=str(event_type))
    ax.set_title("1. Raw power stack")
    ax.set_ylabel("Mean raw power")
    ax.legend(fontsize=8)

    for ax, mode, title in [
        (axes[0, 1], "sideband_linear", "2. Sideband-linear residual"),
        (axes[1, 0], "joint_step_linear", "3. Joint baseline+step residual"),
        (axes[1, 1], "pre_event_anchor", "4. Pre-event anchored residual"),
    ]:
        sub = stacked[stacked["mode"].eq(mode)]
        for event_type, grp in sub.groupby("event_type", sort=True):
            grp = grp.sort_values("t_bin_sec")
            ax.plot(grp["t_bin_sec"] / 60.0, grp["mean"], lw=1.7, color=colors.get(str(event_type)), label=str(event_type))
        ax.axhline(0.0, color="0.35", lw=0.8)
        ax.set_title(title)
        ax.set_ylabel("Mean normalized residual")
    for ax in axes.ravel():
        ax.axvline(0.0, color="black", lw=0.9)
        ax.set_xlabel("Relative time from predicted event (min)")
    fig.suptitle(f"Earth {FREQUENCY_MHZ:.2f} MHz {ANTENNA_LABEL}: four diagnostic views")
    fig.savefig(OUT / "earth_4p70mhz_lower_v_four_diagnostic_views.png", dpi=180)
    plt.close(fig)


def _write_report(events: pd.DataFrame, fits: pd.DataFrame, raw_stack: pd.DataFrame, stacked: pd.DataFrame) -> None:
    metric_rows = []
    for mode, grp in stacked.groupby("mode", sort=True):
        center = grp[np.abs(grp["t_bin_sec"]) <= 120.0]
        far = grp[np.abs(grp["t_bin_sec"]) >= 360.0]
        center_abs = float(center["mean"].abs().max()) if not center.empty else np.nan
        far_abs = float(far["mean"].abs().max()) if not far.empty else np.nan
        metric_rows.append(
            {
                "mode": mode,
                "max_events_per_bin": int(grp["n_events"].max()) if "n_events" in grp else 0,
                "center_abs_peak": center_abs,
                "far_abs_peak": far_abs,
                "far_to_center_ratio": float(far_abs / center_abs) if np.isfinite(center_abs) and center_abs > 0 else np.nan,
            }
        )
    metrics = pd.DataFrame.from_records(metric_rows)
    metrics.to_csv(OUT / "earth_4p70mhz_lower_v_diagnostic_metrics.csv", index=False)
    lines = [
        "# Earth 4.70 MHz Lower-V Four Diagnostic Views",
        "",
        f"Events used: {len(events)} from `stack_input_events.csv`.",
        "",
        "Generated views:",
        "",
        f"1. Raw power with fitted model overlays: `{OUT / 'earth_4p70mhz_lower_v_raw_power_model_overlays.png'}`",
        f"2. Raw power stack: `{OUT / 'earth_4p70mhz_lower_v_raw_power_stack.png'}`",
        f"3. Sideband-linear residual profile: `{OUT / 'earth_4p70mhz_lower_v_sideband_linear_event_type_profile.png'}`",
        f"4. Joint baseline+step residual profile: `{OUT / 'earth_4p70mhz_lower_v_joint_step_linear_event_type_profile.png'}`",
        f"5. Pre-event anchored residual profile: `{OUT / 'earth_4p70mhz_lower_v_pre_event_anchor_event_type_profile.png'}`",
        f"6. Combined four-panel comparison: `{OUT / 'earth_4p70mhz_lower_v_four_diagnostic_views.png'}`",
        "",
        "Interpretation:",
        "",
        "- The raw-power overlay plot shows what the model is fitting before any subtraction.",
        "- The sideband-linear view excludes the central event region from the baseline fit.",
        "- The joint-step view fits baseline and occultation step together, then subtracts only the baseline.",
        "- The pre-event anchored view uses the pre-event side as the zero point, so it is easier to read as a before/after change.",
        "",
        "Residual-shape metrics:",
        "",
        _markdown_table(metrics),
        "",
    ]
    (OUT / "earth_4p70mhz_lower_v_four_view_report.md").write_text("\n".join(lines) + "\n")


def _markdown_table(df: pd.DataFrame) -> str:
    work = df.copy()
    for col in ("center_abs_peak", "far_abs_peak", "far_to_center_ratio"):
        work[col] = pd.to_numeric(work[col], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.3g}")
    cols = list(work.columns)
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
