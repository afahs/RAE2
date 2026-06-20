#!/usr/bin/env python
"""Raw pre/post occultation contrast test for lower-V detections.

This is a deliberately simple alternative to baseline subtraction and fitted
stack amplitudes.  For each predicted lower-V occultation event, it compares
raw power before and after the event:

    disappearance: contrast = log(median_pre_power) - log(median_post_power)
    reappearance:  contrast = log(median_post_power) - log(median_pre_power)

Positive contrast is source-like for both event types.  The same calculation is
applied to time-shift, randomized-time, and off-source controls.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from math import comb

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rylevonberg.table_io import read_table

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.constants import FREQUENCY_MAP_MHZ  # noqa: E402
from rylevonberg.util import datetime_ns, ensure_dir, robust_sigma, software_versions, write_json  # noqa: E402
from scripts.run_lower_v_stackfirst_detection_attempt import (  # noqa: E402
    ANTENNA,
    CLEAN,
    _load_clean_groups,
    _load_event_table,
)


DEFAULT_OUT = ROOT / "outputs/lower_v_prepost_rank_detection_v1"
EARTH_EVENT_ROWS = ROOT / "outputs/lower_v_control_manifold_earth_positive_control_v1/lower_v_stackfirst_event_rows.csv"
SOURCE_LABEL = {"earth": "Earth", "sun": "Sun", "fornax_a": "Fornax-A"}
CONTROL_COLORS = {"time_shift": "#d95f02", "randomized_time": "#7570b3", "offsource": "#1b9e77"}


def _read(path: Path, **kwargs) -> pd.DataFrame:
    return read_table(path, low_memory=False, **kwargs)


def _binom_sf_one_sided(k: int, n: int) -> float:
    if n <= 0:
        return np.nan
    try:
        from scipy.stats import binomtest

        return float(binomtest(int(k), int(n), p=0.5, alternative="greater").pvalue)
    except Exception:
        if n > 1024:
            # Continuity-corrected normal fallback.
            z = ((k - 0.5) - 0.5 * n) / np.sqrt(0.25 * n)
            return float(0.5 * np.math.erfc(z / np.sqrt(2)))
        return float(sum(comb(n, i) for i in range(k, n + 1)) / 2**n)


def _mannwhitney_greater(real: np.ndarray, controls: np.ndarray) -> float:
    real = real[np.isfinite(real)]
    controls = controls[np.isfinite(controls)]
    if len(real) == 0 or len(controls) == 0:
        return np.nan
    try:
        from scipy.stats import mannwhitneyu

        return float(mannwhitneyu(real, controls, alternative="greater").pvalue)
    except Exception:
        return np.nan


def _cliffs_delta(real: np.ndarray, controls: np.ndarray) -> float:
    real = np.sort(real[np.isfinite(real)])
    controls = np.sort(controls[np.isfinite(controls)])
    if len(real) == 0 or len(controls) == 0:
        return np.nan
    less = np.searchsorted(controls, real, side="left").sum()
    leq = np.searchsorted(controls, real, side="right").sum()
    greater = len(controls) * len(real) - leq
    return float((less - greater) / (len(real) * len(controls)))


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
    earth["analysis_source"] = "earth"

    moving = _load_event_table(
        start_date,
        shifts_s=shifts_s,
        n_random=int(n_random),
        random_seed=int(random_seed),
        window_s=float(outer_s),
        max_offsource_controls=int(max_offsource_controls),
    )
    moving = moving[moving["analysis_source"].astype(str).isin(["sun", "fornax_a"])].copy()
    events = pd.concat([earth, moving], ignore_index=True)
    events = events[events["antenna"].astype(str).eq(ANTENNA)].copy()
    events["event_uid_input"] = events.get("event_uid", np.arange(len(events)))
    events["event_uid"] = np.arange(len(events), dtype=int)
    events["predicted_event_time"] = pd.to_datetime(events["predicted_event_time"])
    return events.reset_index(drop=True)


def _local_window(payload: dict[str, np.ndarray], event_time: pd.Timestamp, outer_s: float) -> tuple[np.ndarray, np.ndarray] | None:
    event_ns = pd.Timestamp(event_time).value
    half_ns = int(float(outer_s) * 1e9)
    time_ns = payload["time_ns"]
    lo = int(np.searchsorted(time_ns, event_ns - half_ns, side="left"))
    hi = int(np.searchsorted(time_ns, event_ns + half_ns, side="right"))
    if hi <= lo:
        return None
    t = (time_ns[lo:hi] - event_ns).astype(float) / 1e9
    y = payload["power"][lo:hi]
    keep = payload["valid"][lo:hi] & np.isfinite(y) & (y > 0) & (np.abs(t) <= float(outer_s))
    if np.count_nonzero(keep) == 0:
        return None
    order = np.argsort(t[keep])
    return t[keep][order], y[keep][order]


def extract_prepost_contrasts(
    events: pd.DataFrame,
    clean_groups: dict[tuple[int, str], dict[str, np.ndarray]],
    inner_s: float,
    outer_s: float,
    min_side_samples: int,
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
            status.append({**base, "used": False, "failure": "missing_clean_group", "n_pre": 0, "n_post": 0})
            continue
        local = _local_window(payload, pd.Timestamp(ev.predicted_event_time), outer_s)
        if local is None:
            status.append({**base, "used": False, "failure": "no_valid_samples", "n_pre": 0, "n_post": 0})
            continue
        t, y = local
        pre = y[(t >= -float(outer_s)) & (t <= -float(inner_s))]
        post = y[(t >= float(inner_s)) & (t <= float(outer_s))]
        if len(pre) < int(min_side_samples) or len(post) < int(min_side_samples):
            status.append({**base, "used": False, "failure": "too_few_pre_or_post_samples", "n_pre": int(len(pre)), "n_post": int(len(post))})
            continue
        pre_med = float(np.nanmedian(pre))
        post_med = float(np.nanmedian(post))
        if not np.isfinite(pre_med) or not np.isfinite(post_med) or pre_med <= 0 or post_med <= 0:
            status.append({**base, "used": False, "failure": "bad_pre_or_post_median", "n_pre": int(len(pre)), "n_post": int(len(post))})
            continue
        log_pre = float(np.log(pre_med))
        log_post = float(np.log(post_med))
        if str(ev.event_type) == "disappearance":
            contrast = log_pre - log_post
            raw_delta = pre_med - post_med
        else:
            contrast = log_post - log_pre
            raw_delta = post_med - pre_med
        all_log = np.log(y[y > 0])
        local_scale = robust_sigma(all_log - np.nanmedian(all_log)) if len(all_log) else np.nan
        rows.append(
            {
                **base,
                "used": True,
                "n_pre": int(len(pre)),
                "n_post": int(len(post)),
                "pre_median_power": pre_med,
                "post_median_power": post_med,
                "log_pre_median_power": log_pre,
                "log_post_median_power": log_post,
                "source_like_log_contrast": float(contrast),
                "source_like_fractional_contrast": float(np.exp(contrast) - 1.0),
                "source_like_raw_delta_power": float(raw_delta),
                "local_log_power_scale": float(local_scale) if np.isfinite(local_scale) else np.nan,
            }
        )
        status.append({**base, "used": True, "failure": "", "n_pre": int(len(pre)), "n_post": int(len(post))})
    return pd.DataFrame(rows), pd.DataFrame(status)


def summarize_contrasts(contrasts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    real = contrasts[contrasts["control_family"].astype(str).eq("real")].copy()
    controls = contrasts[~contrasts["control_family"].astype(str).eq("real")].copy()
    for keys, grp in real.groupby(["analysis_source", "frequency_band", "frequency_mhz"], sort=True):
        source, band, freq = keys
        vals = pd.to_numeric(grp["source_like_log_contrast"], errors="coerce").dropna().to_numpy(dtype=float)
        same_controls = controls[
            controls["analysis_source"].astype(str).eq(str(source))
            & controls["frequency_band"].astype(int).eq(int(band))
        ].copy()
        cvals = pd.to_numeric(same_controls["source_like_log_contrast"], errors="coerce").dropna().to_numpy(dtype=float)
        control_group_medians = (
            same_controls.groupby(["control_family", "control_id"], sort=True)["source_like_log_contrast"].median().dropna().to_numpy(dtype=float)
        )
        n = int(len(vals))
        k = int(np.count_nonzero(vals > 0))
        med = float(np.nanmedian(vals)) if len(vals) else np.nan
        frac = float(np.nanmedian(np.exp(vals) - 1.0)) if len(vals) else np.nan
        sign_p = _binom_sf_one_sided(k, n)
        rank_p = _mannwhitney_greater(vals, cvals)
        delta = _cliffs_delta(vals, cvals)
        if len(control_group_medians):
            empirical_p = float((1 + np.count_nonzero(control_group_medians >= med)) / (1 + len(control_group_medians)))
            control_group_q25 = float(np.nanquantile(control_group_medians, 0.25))
            control_group_q75 = float(np.nanquantile(control_group_medians, 0.75))
            control_group_median = float(np.nanmedian(control_group_medians))
        else:
            empirical_p = np.nan
            control_group_q25 = np.nan
            control_group_q75 = np.nan
            control_group_median = np.nan
        out = {
            "analysis_source": source,
            "frequency_band": int(band),
            "frequency_mhz": float(freq),
            "n_real_events": n,
            "real_sign_positive_n": k,
            "real_sign_fraction": float(k / n) if n else np.nan,
            "real_median_log_contrast": med,
            "real_median_fractional_contrast": frac,
            "one_sided_sign_p": sign_p,
            "mannwhitney_real_gt_controls_p": rank_p,
            "cliffs_delta_real_vs_controls": delta,
            "control_group_empirical_p_median_ge_real": empirical_p,
            "control_group_median_log_contrast": control_group_median,
            "control_group_q25_log_contrast": control_group_q25,
            "control_group_q75_log_contrast": control_group_q75,
            "n_control_events": int(len(cvals)),
            "n_control_groups": int(len(control_group_medians)),
        }
        for family in ["time_shift", "randomized_time", "offsource"]:
            fam = same_controls[same_controls["control_family"].astype(str).eq(family)]
            fvals = pd.to_numeric(fam["source_like_log_contrast"], errors="coerce").dropna().to_numpy(dtype=float)
            fgroups = fam.groupby("control_id", sort=True)["source_like_log_contrast"].median().dropna().to_numpy(dtype=float)
            out[f"{family}_n_events"] = int(len(fvals))
            out[f"{family}_median_log_contrast"] = float(np.nanmedian(fvals)) if len(fvals) else np.nan
            out[f"{family}_q25_log_contrast"] = float(np.nanquantile(fvals, 0.25)) if len(fvals) else np.nan
            out[f"{family}_q75_log_contrast"] = float(np.nanquantile(fvals, 0.75)) if len(fvals) else np.nan
            out[f"{family}_n_groups"] = int(len(fgroups))
            out[f"{family}_group_median_log_contrast"] = float(np.nanmedian(fgroups)) if len(fgroups) else np.nan
            out[f"{family}_group_empirical_p_median_ge_real"] = (
                float((1 + np.count_nonzero(fgroups >= med)) / (1 + len(fgroups))) if len(fgroups) else np.nan
            )
        rows.append(out)
    return classify_prepost_evidence(pd.DataFrame(rows))


def classify_prepost_evidence(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    classes: list[str] = []
    reasons: list[str] = []
    for _, row in out.iterrows():
        source = str(row["analysis_source"])
        median = float(row.get("real_median_log_contrast", np.nan))
        sign_frac = float(row.get("real_sign_fraction", np.nan))
        sign_p = float(row.get("one_sided_sign_p", np.nan))
        mw_p = float(row.get("mannwhitney_real_gt_controls_p", np.nan))
        delta = float(row.get("cliffs_delta_real_vs_controls", np.nan))
        empirical = float(row.get("control_group_empirical_p_median_ge_real", np.nan))
        offsource_p = float(row.get("offsource_group_empirical_p_median_ge_real", np.nan))

        positive = (
            np.isfinite(median)
            and median > 0
            and np.isfinite(sign_frac)
            and sign_frac >= 0.60
            and np.isfinite(sign_p)
            and sign_p <= 1e-6
            and np.isfinite(mw_p)
            and mw_p <= 1e-6
            and np.isfinite(delta)
            and delta >= 0.10
        )
        source_specific = np.isfinite(empirical) and empirical <= 0.10
        offsource_specific = (not np.isfinite(offsource_p)) or offsource_p <= 0.10
        anti = np.isfinite(median) and median < 0 and np.isfinite(sign_frac) and sign_frac <= 0.40

        if source == "earth" and positive and empirical <= 0.20:
            cls = "positive_control_detected"
            reason = "raw pre/post signs, ranks, and control comparison pass for Earth"
        elif source != "earth" and positive and source_specific and offsource_specific:
            cls = "source_specific_candidate"
            reason = "real predicted events are positive and exceed time/random/off-source controls"
        elif positive:
            cls = "non_specific_positive_shift"
            reason = "real predicted events are positive, but control medians are comparable"
        elif anti:
            cls = "anti_template"
            reason = "real predicted events preferentially move opposite the occultation expectation"
        else:
            cls = "not_detected"
            reason = "no robust positive raw pre/post shift"
        classes.append(cls)
        reasons.append(reason)
    out["prepost_evidence_class"] = classes
    out["prepost_evidence_reason"] = reasons
    return out


def choose_visual_channels(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source, grp in summary.groupby("analysis_source", sort=True):
        g = grp.copy()
        g["_score"] = (
            g["real_median_log_contrast"].fillna(-999)
            - g["control_group_q75_log_contrast"].fillna(0)
            + g["real_sign_fraction"].fillna(0)
        )
        if source == "earth":
            g = g.sort_values(["control_group_empirical_p_median_ge_real", "_score"], ascending=[True, False])
        else:
            g = g.sort_values(["mannwhitney_real_gt_controls_p", "_score"], ascending=[True, False])
        rows.append(g.head(2))
    return pd.concat(rows, ignore_index=True)


def plot_contrast_spectrum(summary: pd.DataFrame, out_dir: Path) -> Path:
    sources = [s for s in ["earth", "sun", "fornax_a"] if s in set(summary["analysis_source"].astype(str))]
    fig, axes = plt.subplots(len(sources), 1, figsize=(12.2, 3.0 * len(sources)), sharex=True, constrained_layout=True)
    if len(sources) == 1:
        axes = [axes]
    for ax, source in zip(axes, sources):
        sub = summary[summary["analysis_source"].astype(str).eq(source)].sort_values("frequency_mhz")
        x = sub["frequency_mhz"].to_numpy(dtype=float)
        ax.axhline(0, color="0.55", lw=0.8)
        ax.fill_between(
            x,
            sub["control_group_q25_log_contrast"].to_numpy(dtype=float),
            sub["control_group_q75_log_contrast"].to_numpy(dtype=float),
            color="0.65",
            alpha=0.22,
            label="control group median IQR",
        )
        ax.plot(
            x,
            sub["control_group_median_log_contrast"],
            color="0.35",
            lw=1.0,
            ls="--",
            label="control group median",
        )
        pts = ax.scatter(
            x,
            sub["real_median_log_contrast"],
            c=sub["real_sign_fraction"],
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            s=65,
            edgecolor="black",
            label="real predicted events",
            zorder=3,
        )
        ax.plot(x, sub["real_median_log_contrast"], color="black", lw=1.0, alpha=0.65)
        for _, row in sub.iterrows():
            ax.text(float(row["frequency_mhz"]), float(row["real_median_log_contrast"]), f"{float(row['real_sign_fraction']):.2f}", fontsize=7, ha="center", va="bottom")
        ax.set_xscale("log")
        ax.set_ylabel("median source-like log contrast")
        ax.set_title(f"{SOURCE_LABEL.get(source, source)} raw pre/post contrast")
        ax.grid(True, color="0.9", lw=0.5)
    axes[-1].set_xticks(list(FREQUENCY_MAP_MHZ.values()))
    axes[-1].set_xticklabels([f"{v:g}" for v in FREQUENCY_MAP_MHZ.values()], rotation=45, ha="right")
    axes[-1].set_xlabel("frequency (MHz)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    cbar = fig.colorbar(pts, ax=axes, pad=0.02, shrink=0.88)
    cbar.set_label("fraction of real events with source-like sign")
    fig.suptitle("Lower-V raw pre/post detection test; positive is occultation-like")
    path = out_dir / "lower_v_prepost_contrast_spectrum.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_distribution_panels(contrasts: pd.DataFrame, summary: pd.DataFrame, selected: pd.DataFrame, out_dir: Path) -> list[Path]:
    paths = []
    for source, rows in selected.groupby("analysis_source", sort=True):
        fig, axes = plt.subplots(len(rows), 1, figsize=(9.5, 3.2 * len(rows)), sharex=False)
        if len(rows) == 1:
            axes = [axes]
        for ax, (_, row) in zip(axes, rows.iterrows()):
            freq = float(row["frequency_mhz"])
            sub = contrasts[
                contrasts["analysis_source"].astype(str).eq(str(source))
                & np.isclose(pd.to_numeric(contrasts["frequency_mhz"], errors="coerce"), freq)
            ].copy()
            real = sub[sub["control_family"].astype(str).eq("real")]["source_like_log_contrast"].dropna().to_numpy(dtype=float)
            controls = sub[~sub["control_family"].astype(str).eq("real")]
            bins = np.linspace(
                np.nanpercentile(sub["source_like_log_contrast"], 1),
                np.nanpercentile(sub["source_like_log_contrast"], 99),
                45,
            )
            ax.hist(real, bins=bins, density=True, histtype="step", lw=2.0, color="black", label=f"real n={len(real)}")
            for family, color in CONTROL_COLORS.items():
                vals = controls[controls["control_family"].astype(str).eq(family)]["source_like_log_contrast"].dropna().to_numpy(dtype=float)
                if len(vals):
                    ax.hist(vals, bins=bins, density=True, histtype="step", lw=1.3, color=color, label=f"{family} n={len(vals)}")
            ax.axvline(0, color="0.55", lw=0.8)
            ax.axvline(float(row["real_median_log_contrast"]), color="black", ls="--", lw=1.2)
            ax.set_title(
                f"{SOURCE_LABEL.get(source, source)} {freq:.2f} MHz: "
                f"median={float(row['real_median_log_contrast']):.3g}, "
                f"sign frac={float(row['real_sign_fraction']):.2f}, "
                f"MW p={float(row['mannwhitney_real_gt_controls_p']):.3g}"
            )
            ax.set_xlabel("source-like log contrast")
            ax.set_ylabel("density")
            ax.grid(True, color="0.92", lw=0.5)
            ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        path = out_dir / f"{source}_selected_prepost_contrast_distributions.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_prepost_event_lines(contrasts: pd.DataFrame, selected: pd.DataFrame, out_dir: Path, max_events: int, seed: int) -> list[Path]:
    paths = []
    rng = np.random.default_rng(seed)
    for source, rows in selected.groupby("analysis_source", sort=True):
        fig, axes = plt.subplots(len(rows), 2, figsize=(10.5, 3.0 * len(rows)), sharex=True, sharey=False)
        if len(rows) == 1:
            axes = np.asarray([axes])
        for i, (_, row) in enumerate(rows.iterrows()):
            freq = float(row["frequency_mhz"])
            for j, event_type in enumerate(["disappearance", "reappearance"]):
                ax = axes[i, j]
                sub = contrasts[
                    contrasts["analysis_source"].astype(str).eq(str(source))
                    & contrasts["control_family"].astype(str).eq("real")
                    & np.isclose(pd.to_numeric(contrasts["frequency_mhz"], errors="coerce"), freq)
                    & contrasts["event_type"].astype(str).eq(event_type)
                ].copy()
                if len(sub) > max_events:
                    sub = sub.iloc[rng.choice(np.arange(len(sub)), size=max_events, replace=False)].copy()
                for _, ev in sub.iterrows():
                    y0 = float(ev["log_pre_median_power"])
                    y1 = float(ev["log_post_median_power"])
                    center = 0.5 * (y0 + y1)
                    contrast = float(ev["source_like_log_contrast"])
                    color = "#1a9850" if contrast > 0 else "#d73027"
                    ax.plot([-1, 1], [y0 - center, y1 - center], color=color, alpha=0.20, lw=0.8)
                ax.axhline(0, color="0.7", lw=0.7)
                ax.set_xticks([-1, 1])
                ax.set_xticklabels(["pre", "post"])
                ax.set_title(f"{freq:.2f} MHz {event_type}")
                if j == 0:
                    ax.set_ylabel("event-centered log raw power")
                ax.grid(True, color="0.92", lw=0.5)
        fig.suptitle(f"{SOURCE_LABEL.get(source, source)} real event pre/post raw medians; green lines are source-like")
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        path = out_dir / f"{source}_selected_prepost_event_lines.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def write_report(out_dir: Path, summary: pd.DataFrame, status: pd.DataFrame, paths: list[Path], config: dict[str, object]) -> Path:
    compact = summary[
        [
            "analysis_source",
            "frequency_mhz",
            "prepost_evidence_class",
            "n_real_events",
            "real_sign_fraction",
            "real_median_log_contrast",
            "real_median_fractional_contrast",
            "one_sided_sign_p",
            "mannwhitney_real_gt_controls_p",
            "cliffs_delta_real_vs_controls",
            "control_group_empirical_p_median_ge_real",
            "offsource_group_empirical_p_median_ge_real",
        ]
    ].sort_values(["analysis_source", "frequency_mhz"])
    lines = [
        "# Lower-V Raw Pre/Post Rank Detection Test",
        "",
        "This run avoids fitted baselines and background subtraction. It uses raw lower-V power only.",
        "",
        "For every event, the statistic is:",
        "",
        "    disappearance: log(median pre-event power) - log(median post-event power)",
        "    reappearance:  log(median post-event power) - log(median pre-event power)",
        "",
        "Positive values are source-like for both event types.",
        "",
        "## Why This Is More Principled",
        "",
        "- The test is event-local and uses only two robust medians per event.",
        "- No local trend line is fit.",
        "- No diffuse-background model is subtracted.",
        "- The sign test asks whether predicted events go in the expected direction more often than chance.",
        "- The rank test asks whether predicted events are shifted positive relative to time/off-source/random controls.",
        "- The plots show actual event pre/post raw power medians, not just summary statistics.",
        "",
        "## Configuration",
        "",
        *[f"- `{k}`: `{v}`" for k, v in config.items() if k != "software_versions"],
        "",
        "## Usable Event Counts",
        "",
        status.groupby(["analysis_source", "control_family", "used"], dropna=False).size().reset_index(name="n_rows").to_string(index=False),
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
        "- A convincing channel should have positive median contrast, high source-like sign fraction, small one-sided sign p, small Mann-Whitney p against controls, and visible separation in the distribution plots.",
        "- If real and controls overlap visually, the channel is not detection-grade even if one metric is nominally small.",
        "- Neighboring frequencies should be inspected as coherence evidence, not treated as wrong-frequency controls.",
        "- `non_specific_positive_shift` means the predicted events move in the source-like direction, but the same effect is also present in controls; this is not a source detection.",
    ]
    path = out_dir / "lower_v_prepost_rank_detection_report.md"
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
    parser.add_argument("--n-random", type=int, default=8)
    parser.add_argument("--random-seed", type=int, default=20260609)
    parser.add_argument("--max-offsource-controls", type=int, default=16)
    parser.add_argument("--max-event-lines", type=int, default=120)
    parser.add_argument("--save-large-tables", action="store_true", help="Save event-row, event-contrast, and per-event status tables.")
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    shifts = [float(x.strip()) for x in str(args.time_shifts_s).split(",") if x.strip()]
    config = {
        "antenna": ANTENNA,
        "clean": str(CLEAN),
        "start_date": args.start_date,
        "inner_s": float(args.inner_s),
        "outer_s": float(args.outer_s),
        "min_side_samples": int(args.min_side_samples),
        "time_shifts_s": shifts,
        "n_random": int(args.n_random),
        "random_seed": int(args.random_seed),
        "max_offsource_controls": int(args.max_offsource_controls),
        "save_large_tables": bool(args.save_large_tables),
        "software_versions": software_versions(),
    }
    write_json(out_dir / "run_config.json", config)

    print("Loading events and controls...", flush=True)
    events = load_events(args.start_date, shifts, int(args.n_random), int(args.random_seed), float(args.outer_s), int(args.max_offsource_controls))
    bands = sorted(events["frequency_band"].dropna().astype(int).unique())
    print(f"Loading clean lower-V groups for bands {bands}...", flush=True)
    clean_groups = _load_clean_groups(bands)
    print("Extracting raw pre/post contrasts...", flush=True)
    contrasts, status = extract_prepost_contrasts(events, clean_groups, float(args.inner_s), float(args.outer_s), int(args.min_side_samples))
    if bool(args.save_large_tables):
        events.to_csv(out_dir / "prepost_input_event_rows.csv", index=False)
        contrasts.to_csv(out_dir / "prepost_event_contrasts.csv", index=False)
        status.to_csv(out_dir / "prepost_event_status.csv", index=False)
    status_summary = status.groupby(["analysis_source", "control_family", "used", "failure"], dropna=False).size().reset_index(name="n_rows")
    status_summary.to_csv(out_dir / "prepost_event_status_summary.csv", index=False)
    print("Summarizing rank/sign tests...", flush=True)
    summary = summarize_contrasts(contrasts)
    summary.to_csv(out_dir / "prepost_rank_detection_summary.csv", index=False)
    selected = choose_visual_channels(summary)
    selected.to_csv(out_dir / "prepost_selected_visual_channels.csv", index=False)
    selected_keys = selected[["analysis_source", "frequency_band", "frequency_mhz"]].drop_duplicates()
    selected_contrasts = contrasts.merge(selected_keys, on=["analysis_source", "frequency_band", "frequency_mhz"], how="inner")
    selected_contrasts.to_csv(out_dir / "prepost_selected_event_contrasts.csv", index=False)
    print("Writing plots...", flush=True)
    paths = [plot_contrast_spectrum(summary, out_dir)]
    paths.extend(plot_distribution_panels(contrasts, summary, selected, out_dir))
    paths.extend(plot_prepost_event_lines(contrasts, selected, out_dir, int(args.max_event_lines), int(args.random_seed)))
    report = write_report(out_dir, summary, status, paths, config)
    print(report)


if __name__ == "__main__":
    main()
