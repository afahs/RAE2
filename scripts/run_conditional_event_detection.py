#!/usr/bin/env python
"""Conditional lower-V event selection tests for Sun and Jupiter.

This script tries to avoid two failure modes seen in the global stacks:

1. Slowly varying diffuse/background structure can dominate the pre/post
   contrast.  The `stable_background` gate keeps events whose nearby wrong-time
   windows have small absolute contrast.
2. Jupiter is episodic.  The `activity_gated` branch keeps events where the
   side expected to contain the source is locally bright/active, applying the
   same rule to controls.

The detection statistic remains simple raw pre/post log contrast; no trend
line or diffuse model is subtracted.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.constants import FREQUENCY_MAP_MHZ  # noqa: E402
from rylevonberg.util import ensure_dir, robust_sigma, software_versions, write_json  # noqa: E402
from scripts.run_lower_v_stackfirst_detection_attempt import (  # noqa: E402
    ANTENNA,
    CLEAN,
    _load_clean_groups,
    _load_event_table,
    _make_randomized_controls,
    _make_time_shift_controls,
    _read,
)
from scripts.run_prepost_rank_detection import (  # noqa: E402
    EARTH_EVENT_ROWS,
    _binom_sf_one_sided,
    _cliffs_delta,
    _local_window,
    _mannwhitney_greater,
)


DEFAULT_OUT = ROOT / "outputs/lower_v_conditional_event_detection_sun_jupiter_v1"
JUPITER_EVENTS = ROOT / "outputs/control_survey_jupiter_postnov1974_v1/02_events/predicted_events.csv"
SOURCE_LABEL = {"earth": "Earth", "sun": "Sun", "jupiter": "Jupiter"}
MODE_LABEL = {
    "all_events": "all events",
    "stable_background": "stable wrong-time background",
    "activity_gated": "activity gated",
    "stable_and_active": "stable + active",
}
MODE_COLOR = {
    "all_events": "#4d4d4d",
    "stable_background": "#1b9e77",
    "activity_gated": "#d95f02",
    "stable_and_active": "#7570b3",
}


def load_events(start_date: str, shifts_s: list[float], n_random: int, random_seed: int, outer_s: float, max_offsource_controls: int) -> pd.DataFrame:
    earth = _read(EARTH_EVENT_ROWS, parse_dates=["predicted_event_time"])
    earth = earth[
        earth["analysis_source"].astype(str).eq("earth")
        & earth["antenna"].astype(str).eq(ANTENNA)
        & (pd.to_datetime(earth["predicted_event_time"]) >= pd.Timestamp(start_date))
    ].copy()
    if "time_shift_s" in earth.columns:
        is_shift = earth["control_family"].astype(str).eq("time_shift")
        shift_ok = pd.to_numeric(earth["time_shift_s"], errors="coerce").abs() >= min(abs(x) for x in shifts_s)
        earth = earth[(~is_shift) | shift_ok].copy()

    moving = _load_event_table(
        start_date,
        shifts_s=shifts_s,
        n_random=int(n_random),
        random_seed=int(random_seed),
        window_s=float(outer_s),
        max_offsource_controls=int(max_offsource_controls),
    )
    sun = moving[moving["analysis_source"].astype(str).eq("sun")].copy()

    jupiter = _read(JUPITER_EVENTS, parse_dates=["predicted_event_time"])
    jupiter = jupiter[
        jupiter["source_name"].astype(str).str.lower().eq("jupiter")
        & jupiter["antenna"].astype(str).eq(ANTENNA)
        & (pd.to_datetime(jupiter["predicted_event_time"]) >= pd.Timestamp(start_date))
    ].copy()
    jupiter["analysis_source"] = "jupiter"
    jupiter["control_family"] = "real"
    jupiter["control_id"] = "real"
    jupiter["control_type"] = "true_prediction"
    jupiter_shifts = _make_time_shift_controls(jupiter, shifts_s)
    clean_times = _read(CLEAN, usecols=["time"], parse_dates=["time"])["time"]
    jupiter_random = _make_randomized_controls(
        jupiter,
        clean_times,
        n_random=int(n_random),
        seed=int(random_seed) + 991,
        window_s=float(outer_s),
    )

    events = pd.concat([earth, sun, jupiter, jupiter_shifts, jupiter_random], ignore_index=True)
    events = events[events["antenna"].astype(str).eq(ANTENNA)].copy()
    events["event_uid_input"] = events.get("event_uid", np.arange(len(events)))
    events["event_uid"] = np.arange(len(events), dtype=int)
    events["predicted_event_time"] = pd.to_datetime(events["predicted_event_time"])
    return events.reset_index(drop=True)


def _raw_prepost_for_time(
    payload: dict[str, np.ndarray],
    event_time: pd.Timestamp,
    event_type: str,
    inner_s: float,
    outer_s: float,
    min_side_samples: int,
) -> dict[str, float] | None:
    local = _local_window(payload, pd.Timestamp(event_time), outer_s)
    if local is None:
        return None
    t, y = local
    pre = y[(t >= -float(outer_s)) & (t <= -float(inner_s))]
    post = y[(t >= float(inner_s)) & (t <= float(outer_s))]
    if len(pre) < int(min_side_samples) or len(post) < int(min_side_samples):
        return None
    pre_med = float(np.nanmedian(pre))
    post_med = float(np.nanmedian(post))
    if not np.isfinite(pre_med) or not np.isfinite(post_med) or pre_med <= 0 or post_med <= 0:
        return None
    log_pre = float(np.log(pre_med))
    log_post = float(np.log(post_med))
    if str(event_type) == "disappearance":
        contrast = log_pre - log_post
        visible_log_median = log_pre
    else:
        contrast = log_post - log_pre
        visible_log_median = log_post
    all_log = np.log(y[y > 0])
    center = float(np.nanmedian(all_log)) if len(all_log) else np.nan
    scale = robust_sigma(all_log - center) if len(all_log) else np.nan
    return {
        "n_pre": int(len(pre)),
        "n_post": int(len(post)),
        "log_pre_median_power": log_pre,
        "log_post_median_power": log_post,
        "source_like_log_contrast": float(contrast),
        "visible_side_log_excess": float(visible_log_median - center) if np.isfinite(center) else np.nan,
        "local_log_power_scale": float(scale) if np.isfinite(scale) else np.nan,
    }


def extract_conditional_features(
    events: pd.DataFrame,
    clean_groups: dict[tuple[int, str], dict[str, np.ndarray]],
    inner_s: float,
    outer_s: float,
    min_side_samples: int,
    stability_shifts_s: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    status: list[dict[str, object]] = []
    for ev in events.sort_values(["analysis_source", "control_family", "control_id", "predicted_event_time", "frequency_band"]).itertuples(index=False):
        band = int(ev.frequency_band)
        payload = clean_groups.get((band, ANTENNA))
        base = {
            "analysis_source": str(ev.analysis_source),
            "source_name": str(ev.source_name),
            "control_family": str(ev.control_family),
            "control_id": str(ev.control_id),
            "event_uid": int(ev.event_uid),
            "event_id": getattr(ev, "event_id", np.nan),
            "event_type": str(ev.event_type),
            "predicted_event_time": ev.predicted_event_time,
            "frequency_band": band,
            "frequency_mhz": float(ev.frequency_mhz),
            "antenna": ANTENNA,
        }
        if payload is None:
            status.append({**base, "used": False, "failure": "missing_clean_group"})
            continue
        main = _raw_prepost_for_time(payload, pd.Timestamp(ev.predicted_event_time), str(ev.event_type), inner_s, outer_s, min_side_samples)
        if main is None:
            status.append({**base, "used": False, "failure": "prepost_failed"})
            continue
        drift_values = []
        for shift in stability_shifts_s:
            shifted = pd.Timestamp(ev.predicted_event_time) + pd.to_timedelta(float(shift), unit="s")
            ctrl = _raw_prepost_for_time(payload, shifted, str(ev.event_type), inner_s, outer_s, min_side_samples)
            if ctrl is not None and np.isfinite(ctrl["source_like_log_contrast"]):
                drift_values.append(abs(float(ctrl["source_like_log_contrast"])))
        rows.append(
            {
                **base,
                **main,
                "wrong_time_abs_contrast_median": float(np.nanmedian(drift_values)) if drift_values else np.nan,
                "wrong_time_abs_contrast_n": int(len(drift_values)),
            }
        )
        status.append({**base, "used": True, "failure": ""})
    return pd.DataFrame(rows), pd.DataFrame(status)


def add_selection_modes(features: pd.DataFrame, stable_quantile: float, active_quantile: float) -> pd.DataFrame:
    out = features.copy()
    out["mode_all_events"] = True
    out["mode_stable_background"] = False
    out["mode_activity_gated"] = False
    out["mode_stable_and_active"] = False
    group_cols = ["analysis_source", "control_family", "frequency_band"]
    for _, idx in out.groupby(group_cols, dropna=False).groups.items():
        idx = list(idx)
        drift = pd.to_numeric(out.loc[idx, "wrong_time_abs_contrast_median"], errors="coerce")
        activity = pd.to_numeric(out.loc[idx, "visible_side_log_excess"], errors="coerce")
        if drift.notna().sum() >= 4:
            stable_thr = float(drift.quantile(float(stable_quantile)))
            stable = drift <= stable_thr
        else:
            stable = pd.Series(False, index=idx)
        if activity.notna().sum() >= 4:
            active_thr = float(activity.quantile(float(active_quantile)))
            active = activity >= active_thr
        else:
            active = pd.Series(False, index=idx)
        out.loc[idx, "mode_stable_background"] = stable.to_numpy(dtype=bool)
        out.loc[idx, "mode_activity_gated"] = active.to_numpy(dtype=bool)
        out.loc[idx, "mode_stable_and_active"] = (stable & active).to_numpy(dtype=bool)
    return out


def summarize_mode(features: pd.DataFrame, mode: str) -> pd.DataFrame:
    flag = f"mode_{mode}"
    subset = features[features[flag].astype(bool)].copy()
    rows: list[dict[str, object]] = []
    real = subset[subset["control_family"].astype(str).eq("real")].copy()
    controls = subset[~subset["control_family"].astype(str).eq("real")].copy()
    for keys, grp in real.groupby(["analysis_source", "frequency_band", "frequency_mhz"], sort=True):
        source, band, freq = keys
        vals = pd.to_numeric(grp["source_like_log_contrast"], errors="coerce").dropna().to_numpy(dtype=float)
        same_controls = controls[
            controls["analysis_source"].astype(str).eq(str(source))
            & controls["frequency_band"].astype(int).eq(int(band))
        ].copy()
        cvals = pd.to_numeric(same_controls["source_like_log_contrast"], errors="coerce").dropna().to_numpy(dtype=float)
        group_meds = same_controls.groupby(["control_family", "control_id"], sort=True)["source_like_log_contrast"].median().dropna().to_numpy(dtype=float)
        n = len(vals)
        k = int(np.count_nonzero(vals > 0))
        med = float(np.nanmedian(vals)) if n else np.nan
        empirical = float((1 + np.count_nonzero(group_meds >= med)) / (1 + len(group_meds))) if len(group_meds) else np.nan
        rows.append(
            {
                "mode": mode,
                "analysis_source": source,
                "frequency_band": int(band),
                "frequency_mhz": float(freq),
                "n_real_events": int(n),
                "real_sign_fraction": float(k / n) if n else np.nan,
                "real_median_log_contrast": med,
                "real_median_fractional_contrast": float(np.exp(med) - 1.0) if np.isfinite(med) else np.nan,
                "one_sided_sign_p": _binom_sf_one_sided(k, n),
                "mannwhitney_real_gt_controls_p": _mannwhitney_greater(vals, cvals),
                "cliffs_delta_real_vs_controls": _cliffs_delta(vals, cvals),
                "control_group_empirical_p_median_ge_real": empirical,
                "control_group_median_log_contrast": float(np.nanmedian(group_meds)) if len(group_meds) else np.nan,
                "control_group_q25_log_contrast": float(np.nanquantile(group_meds, 0.25)) if len(group_meds) else np.nan,
                "control_group_q75_log_contrast": float(np.nanquantile(group_meds, 0.75)) if len(group_meds) else np.nan,
                "n_control_events": int(len(cvals)),
                "n_control_groups": int(len(group_meds)),
            }
        )
    return pd.DataFrame(rows)


def classify(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    classes = []
    for _, row in out.iterrows():
        med = float(row.get("real_median_log_contrast", np.nan))
        sign_frac = float(row.get("real_sign_fraction", np.nan))
        sign_p = float(row.get("one_sided_sign_p", np.nan))
        mw_p = float(row.get("mannwhitney_real_gt_controls_p", np.nan))
        delta = float(row.get("cliffs_delta_real_vs_controls", np.nan))
        emp = float(row.get("control_group_empirical_p_median_ge_real", np.nan))
        positive = med > 0 and sign_frac >= 0.60 and sign_p <= 1e-4 and mw_p <= 1e-4 and delta >= 0.10
        if str(row["analysis_source"]) == "earth" and positive and emp <= 0.25:
            cls = "positive_control"
        elif positive and emp <= 0.10:
            cls = "source_specific_candidate"
        elif positive:
            cls = "positive_but_control_comparable"
        elif med < 0 and sign_frac <= 0.40:
            cls = "anti_template"
        else:
            cls = "not_detected"
        classes.append(cls)
    out["conditional_evidence_class"] = classes
    return out


def summarize_all_modes(features: pd.DataFrame) -> pd.DataFrame:
    tables = [summarize_mode(features, mode) for mode in ["all_events", "stable_background", "activity_gated", "stable_and_active"]]
    return classify(pd.concat([t for t in tables if not t.empty], ignore_index=True))


def plot_mode_spectrum(summary: pd.DataFrame, out_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for source, src in summary.groupby("analysis_source", sort=True):
        fig, ax = plt.subplots(figsize=(10.5, 5.2))
        ax.axhline(0, color="0.65", lw=0.8)
        for mode, grp in src.groupby("mode", sort=False):
            grp = grp.sort_values("frequency_mhz")
            ax.plot(
                grp["frequency_mhz"],
                grp["real_median_log_contrast"],
                marker="o",
                lw=1.5,
                color=MODE_COLOR.get(mode, None),
                label=MODE_LABEL.get(mode, mode),
            )
            for _, row in grp.iterrows():
                if str(row.get("conditional_evidence_class")) in ["source_specific_candidate", "positive_control"]:
                    ax.scatter(float(row["frequency_mhz"]), float(row["real_median_log_contrast"]), s=115, facecolor="none", edgecolor="black", lw=1.5)
        ax.set_xscale("log")
        ax.set_xticks(list(FREQUENCY_MAP_MHZ.values()))
        ax.set_xticklabels([f"{v:g}" for v in FREQUENCY_MAP_MHZ.values()], rotation=45, ha="right")
        ax.set_xlabel("frequency (MHz)")
        ax.set_ylabel("median source-like log contrast")
        ax.set_title(f"{SOURCE_LABEL.get(source, source)} conditional lower-V raw pre/post test")
        ax.grid(True, color="0.9", lw=0.5)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        path = out_dir / f"{source}_conditional_mode_spectrum.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def select_visual_cases(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source, src in summary[summary["analysis_source"].isin(["sun", "jupiter"])].groupby("analysis_source", sort=True):
        g = src.copy()
        g["_score"] = (
            g["real_median_log_contrast"].fillna(-999)
            + 0.25 * g["real_sign_fraction"].fillna(0)
            - g["control_group_empirical_p_median_ge_real"].fillna(1)
        )
        cand = g[g["conditional_evidence_class"].isin(["source_specific_candidate", "positive_but_control_comparable"])]
        if cand.empty:
            cand = g
        rows.append(cand.sort_values("_score", ascending=False).head(3))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def plot_selected_distributions(features: pd.DataFrame, selected: pd.DataFrame, out_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for source, cases in selected.groupby("analysis_source", sort=True):
        fig, axes = plt.subplots(len(cases), 1, figsize=(9.5, 3.1 * len(cases)), sharex=False)
        if len(cases) == 1:
            axes = [axes]
        for ax, (_, row) in zip(axes, cases.iterrows()):
            mode = str(row["mode"])
            flag = f"mode_{mode}"
            freq = float(row["frequency_mhz"])
            sub = features[
                features[flag].astype(bool)
                & features["analysis_source"].astype(str).eq(source)
                & np.isclose(pd.to_numeric(features["frequency_mhz"], errors="coerce"), freq)
            ].copy()
            real = sub[sub["control_family"].astype(str).eq("real")]["source_like_log_contrast"].dropna().to_numpy(dtype=float)
            controls = sub[~sub["control_family"].astype(str).eq("real")]["source_like_log_contrast"].dropna().to_numpy(dtype=float)
            if len(real) + len(controls) == 0:
                continue
            lo = np.nanpercentile(np.r_[real, controls], 1)
            hi = np.nanpercentile(np.r_[real, controls], 99)
            if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                lo, hi = -1, 1
            bins = np.linspace(lo, hi, 45)
            ax.hist(real, bins=bins, density=True, histtype="step", color="black", lw=2.0, label=f"real n={len(real)}")
            ax.hist(controls, bins=bins, density=True, histtype="step", color="#7570b3", lw=1.5, label=f"controls n={len(controls)}")
            ax.axvline(0, color="0.55", lw=0.8)
            ax.axvline(float(row["real_median_log_contrast"]), color="black", ls="--", lw=1.2)
            ax.set_title(
                f"{SOURCE_LABEL.get(source, source)} {freq:.2f} MHz {MODE_LABEL.get(mode, mode)}: "
                f"{row['conditional_evidence_class']}, sign={float(row['real_sign_fraction']):.2f}, "
                f"emp p={float(row['control_group_empirical_p_median_ge_real']):.3g}"
            )
            ax.set_xlabel("source-like log contrast")
            ax.set_ylabel("density")
            ax.grid(True, color="0.92", lw=0.5)
            ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        path = out_dir / f"{source}_conditional_selected_distributions.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_selected_event_lines(features: pd.DataFrame, selected: pd.DataFrame, out_dir: Path, max_events: int, seed: int) -> list[Path]:
    paths: list[Path] = []
    rng = np.random.default_rng(seed)
    for source, cases in selected.groupby("analysis_source", sort=True):
        fig, axes = plt.subplots(len(cases), 2, figsize=(10.5, 3.0 * len(cases)), sharex=True, sharey=False)
        if len(cases) == 1:
            axes = np.asarray([axes])
        for i, (_, row) in enumerate(cases.iterrows()):
            mode = str(row["mode"])
            flag = f"mode_{mode}"
            freq = float(row["frequency_mhz"])
            for j, event_type in enumerate(["disappearance", "reappearance"]):
                ax = axes[i, j]
                sub = features[
                    features[flag].astype(bool)
                    & features["analysis_source"].astype(str).eq(source)
                    & features["control_family"].astype(str).eq("real")
                    & np.isclose(pd.to_numeric(features["frequency_mhz"], errors="coerce"), freq)
                    & features["event_type"].astype(str).eq(event_type)
                ].copy()
                if len(sub) > max_events:
                    sub = sub.iloc[rng.choice(np.arange(len(sub)), size=max_events, replace=False)].copy()
                for _, ev in sub.iterrows():
                    y0 = float(ev["log_pre_median_power"])
                    y1 = float(ev["log_post_median_power"])
                    center = 0.5 * (y0 + y1)
                    contrast = float(ev["source_like_log_contrast"])
                    color = "#1a9850" if contrast > 0 else "#d73027"
                    ax.plot([-1, 1], [y0 - center, y1 - center], color=color, alpha=0.23, lw=0.85)
                ax.axhline(0, color="0.7", lw=0.7)
                ax.set_xticks([-1, 1])
                ax.set_xticklabels(["pre", "post"])
                ax.set_title(f"{freq:.2f} MHz {MODE_LABEL.get(mode, mode)} {event_type}")
                if j == 0:
                    ax.set_ylabel("event-centered log raw power")
                ax.grid(True, color="0.92", lw=0.5)
        fig.suptitle(f"{SOURCE_LABEL.get(source, source)} selected real events; green lines are source-like")
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        path = out_dir / f"{source}_conditional_selected_event_lines.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def write_report(out_dir: Path, summary: pd.DataFrame, status: pd.DataFrame, selected: pd.DataFrame, paths: list[Path], config: dict[str, object]) -> Path:
    compact_cols = [
        "mode",
        "analysis_source",
        "frequency_mhz",
        "conditional_evidence_class",
        "n_real_events",
        "real_sign_fraction",
        "real_median_log_contrast",
        "one_sided_sign_p",
        "mannwhitney_real_gt_controls_p",
        "control_group_empirical_p_median_ge_real",
    ]
    compact = summary[compact_cols].sort_values(["analysis_source", "mode", "frequency_mhz"])
    lines = [
        "# Conditional Event-Selection Detection Test",
        "",
        "This run tests event subsets selected without using the target occultation sign:",
        "",
        "- `stable_background`: wrong-time windows near the event have low absolute pre/post contrast.",
        "- `activity_gated`: the side expected to contain the source is locally bright/active; this is intended mainly for episodic Jupiter.",
        "- `stable_and_active`: both conditions.",
        "",
        "The detection statistic is still raw lower-V pre/post log contrast. No trend or diffuse model is subtracted.",
        "",
        "## Configuration",
        "",
        *[f"- `{k}`: `{v}`" for k, v in config.items() if k != "software_versions"],
        "",
        "## Usable Event Counts",
        "",
        status.groupby(["analysis_source", "control_family", "used"], dropna=False).size().reset_index(name="n_rows").to_string(index=False),
        "",
        "## Selected Visual Cases",
        "",
        selected[compact_cols].to_string(index=False) if not selected.empty else "(none)",
        "",
        "## Summary Table",
        "",
        compact.to_string(index=False),
        "",
        "## Plots",
        "",
        *[f"- `{p}`" for p in paths],
        "",
        "## Interpretation Notes",
        "",
        "- `source_specific_candidate` means the selected real events are positive and exceed the selected controls.",
        "- `positive_but_control_comparable` means the subset looks source-like, but controls can reproduce similar contrast.",
        "- The activity gate is physically motivated for Jupiter, but it is still not sufficient unless controls fail the same test.",
    ]
    path = out_dir / "conditional_event_detection_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--start-date", default="1974-11-01")
    parser.add_argument("--inner-s", type=float, default=120.0)
    parser.add_argument("--outer-s", type=float, default=600.0)
    parser.add_argument("--min-side-samples", type=int, default=2)
    parser.add_argument("--time-shifts-s", default="-7200,-3600,-1800,1800,3600,7200")
    parser.add_argument("--stability-shifts-s", default="-7200,-3600,-1800,1800,3600,7200")
    parser.add_argument("--stable-quantile", type=float, default=0.40)
    parser.add_argument("--active-quantile", type=float, default=0.75)
    parser.add_argument("--n-random", type=int, default=8)
    parser.add_argument("--random-seed", type=int, default=20260609)
    parser.add_argument("--max-offsource-controls", type=int, default=16)
    parser.add_argument("--max-event-lines", type=int, default=120)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    shifts = [float(x.strip()) for x in str(args.time_shifts_s).split(",") if x.strip()]
    stability_shifts = [float(x.strip()) for x in str(args.stability_shifts_s).split(",") if x.strip()]
    config = {
        "antenna": ANTENNA,
        "clean": str(CLEAN),
        "start_date": args.start_date,
        "inner_s": float(args.inner_s),
        "outer_s": float(args.outer_s),
        "min_side_samples": int(args.min_side_samples),
        "time_shifts_s": shifts,
        "stability_shifts_s": stability_shifts,
        "stable_quantile": float(args.stable_quantile),
        "active_quantile": float(args.active_quantile),
        "n_random": int(args.n_random),
        "random_seed": int(args.random_seed),
        "max_offsource_controls": int(args.max_offsource_controls),
        "software_versions": software_versions(),
    }
    write_json(out_dir / "run_config.json", config)

    print("Loading real and control event tables...", flush=True)
    events = load_events(args.start_date, shifts, int(args.n_random), int(args.random_seed), float(args.outer_s), int(args.max_offsource_controls))
    bands = sorted(events["frequency_band"].dropna().astype(int).unique())
    print(f"Loading clean lower-V groups for bands {bands}...", flush=True)
    clean_groups = _load_clean_groups(bands)
    print("Extracting raw pre/post features and wrong-time stability scores...", flush=True)
    features, status = extract_conditional_features(events, clean_groups, float(args.inner_s), float(args.outer_s), int(args.min_side_samples), stability_shifts)
    features = add_selection_modes(features, float(args.stable_quantile), float(args.active_quantile))
    status.to_csv(out_dir / "conditional_event_status.csv", index=False)
    features.to_csv(out_dir / "conditional_selected_event_features.csv", index=False)

    print("Summarizing conditional modes...", flush=True)
    summary = summarize_all_modes(features)
    summary.to_csv(out_dir / "conditional_detection_summary.csv", index=False)
    selected = select_visual_cases(summary)
    selected.to_csv(out_dir / "conditional_selected_visual_cases.csv", index=False)

    print("Writing plots...", flush=True)
    paths: list[Path] = []
    paths.extend(plot_mode_spectrum(summary, out_dir))
    paths.extend(plot_selected_distributions(features, selected, out_dir))
    paths.extend(plot_selected_event_lines(features, selected, out_dir, int(args.max_event_lines), int(args.random_seed)))
    report = write_report(out_dir, summary, status, selected, paths, config)
    print(report)


if __name__ == "__main__":
    main()
