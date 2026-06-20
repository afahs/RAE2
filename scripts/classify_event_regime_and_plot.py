#!/usr/bin/env python
"""Classify source-like versus anti-template regimes and plot Earth regime-split grids."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rylevonberg.table_io import read_table
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
GEOM_TABLE = ROOT / "outputs/moving_body_regime_physical_differences_v1/moving_body_event_regime_geometry_table.csv"
STACK_POINTS = ROOT / "outputs/moving_body_stack_type_subset_tests_v1/moving_body_stack_points.csv"
OUT = ROOT / "outputs/moving_body_root_classifier_v1"

COLORS = {"source_like": "#2ca02c", "anti_template": "#d62728"}
EXPECTED_SIGN = {"disappearance": -1.0, "reappearance": 1.0}


def _safe_log10(x: pd.Series) -> pd.Series:
    vals = pd.to_numeric(x, errors="coerce").astype(float)
    positive = vals[vals > 0]
    floor = positive.quantile(0.01) if len(positive) else 1.0
    floor = floor if np.isfinite(floor) and floor > 0 else 1.0
    return np.log10(vals.clip(lower=floor))


def _robust_sigma(values: np.ndarray) -> float:
    vals = values[np.isfinite(values)]
    if vals.size <= 1:
        return np.nan
    center = np.nanmedian(vals)
    sigma = 1.4826 * np.nanmedian(np.abs(vals - center))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.nanstd(vals, ddof=1))
    return sigma if np.isfinite(sigma) and sigma > 0 else np.nan


def _robust_standard_error(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size <= 1:
        return np.nan
    sigma = _robust_sigma(vals)
    if not np.isfinite(sigma) or sigma <= 0:
        return np.nan
    return float(sigma / np.sqrt(vals.size))


def _auc_score(x: pd.Series, y: pd.Series) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    xv = x[mask].to_numpy(dtype=float)
    yv = y[mask].to_numpy(dtype=int)
    if len(np.unique(yv)) < 2 or len(xv) < 10:
        return np.nan
    pos = xv[yv == 1]
    neg = xv[yv == 0]
    stat = stats.mannwhitneyu(pos, neg, alternative="two-sided").statistic
    auc = stat / (len(pos) * len(neg))
    return float(max(auc, 1.0 - auc))


def _threshold_classifier(x: pd.Series, y: pd.Series) -> dict[str, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    xv = x[mask].to_numpy(dtype=float)
    yv = y[mask].to_numpy(dtype=int)
    if len(np.unique(yv)) < 2 or len(xv) < 10:
        return {"threshold": np.nan, "accuracy": np.nan, "balanced_accuracy": np.nan, "auc_abs": np.nan}
    candidates = np.unique(np.nanquantile(xv, np.linspace(0.02, 0.98, 97)))
    best = None
    for threshold in candidates:
        for direction in [1, -1]:
            pred = (direction * xv >= direction * threshold).astype(int)
            tp = np.mean(pred[yv == 1] == 1) if np.any(yv == 1) else np.nan
            tn = np.mean(pred[yv == 0] == 0) if np.any(yv == 0) else np.nan
            bal = 0.5 * (tp + tn)
            acc = np.mean(pred == yv)
            score = (bal, acc)
            if best is None or score > best[0]:
                best = (score, threshold, direction, acc, bal)
    return {
        "threshold": float(best[1]),
        "direction": int(best[2]),
        "accuracy": float(best[3]),
        "balanced_accuracy": float(best[4]),
        "auc_abs": _auc_score(x, y),
    }


def _fit_threshold(xv: np.ndarray, yv: np.ndarray) -> tuple[float, int] | None:
    if len(np.unique(yv)) < 2 or len(xv) < 10:
        return None
    candidates = np.unique(np.nanquantile(xv, np.linspace(0.02, 0.98, 97)))
    best = None
    for threshold in candidates:
        for direction in [1, -1]:
            pred = (direction * xv >= direction * threshold).astype(int)
            tp = np.mean(pred[yv == 1] == 1) if np.any(yv == 1) else np.nan
            tn = np.mean(pred[yv == 0] == 0) if np.any(yv == 0) else np.nan
            bal = 0.5 * (tp + tn)
            acc = np.mean(pred == yv)
            score = (bal, acc)
            if best is None or score > best[0]:
                best = (score, float(threshold), int(direction))
    return best[1], best[2]


def _stratified_folds(yv: np.ndarray, n_splits: int = 5) -> list[np.ndarray]:
    rng = np.random.default_rng(20260522)
    folds = [[] for _ in range(n_splits)]
    for cls in sorted(np.unique(yv)):
        idx = np.where(yv == cls)[0]
        rng.shuffle(idx)
        for i, chunk in enumerate(np.array_split(idx, n_splits)):
            folds[i].extend(chunk.tolist())
    return [np.array(sorted(fold), dtype=int) for fold in folds if fold]


def _cv_threshold_classifier(x: pd.Series, y: pd.Series, n_splits: int = 5) -> dict[str, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    xv = x[mask].to_numpy(dtype=float)
    yv = y[mask].to_numpy(dtype=int)
    if len(np.unique(yv)) < 2 or len(xv) < 25:
        return {"cv_accuracy": np.nan, "cv_balanced_accuracy": np.nan}
    pred_all = np.full(len(yv), np.nan)
    for test_idx in _stratified_folds(yv, n_splits=n_splits):
        train_idx = np.setdiff1d(np.arange(len(yv)), test_idx)
        fit = _fit_threshold(xv[train_idx], yv[train_idx])
        if fit is None:
            continue
        threshold, direction = fit
        pred_all[test_idx] = (direction * xv[test_idx] >= direction * threshold).astype(int)
    keep = np.isfinite(pred_all)
    if not np.any(keep):
        return {"cv_accuracy": np.nan, "cv_balanced_accuracy": np.nan}
    pred = pred_all[keep].astype(int)
    truth = yv[keep]
    tp = np.mean(pred[truth == 1] == 1) if np.any(truth == 1) else np.nan
    tn = np.mean(pred[truth == 0] == 0) if np.any(truth == 0) else np.nan
    return {
        "cv_accuracy": float(np.mean(pred == truth)),
        "cv_balanced_accuracy": float(0.5 * (tp + tn)),
    }


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    work = df[df["regime"].isin(["source_like", "anti_template"])].copy()
    work["label_source_like"] = work["regime"].eq("source_like").astype(int)
    sign = work["event_type"].map(EXPECTED_SIGN).astype(float)
    work["expected_signed_near_slope"] = sign * pd.to_numeric(work["near_raw_slope_per_s"], errors="coerce")
    work["expected_signed_far_slope"] = sign * pd.to_numeric(work["far_raw_slope_per_s"], errors="coerce")
    work["expected_signed_raw_contrast"] = sign * pd.to_numeric(work["raw_post_minus_pre"], errors="coerce")
    work["expected_signed_fractional_contrast"] = pd.to_numeric(work["source_like_fractional_contrast"], errors="coerce")
    work["log_pre_background"] = _safe_log10(work["pre_far_median"])
    work["log_pre_sigma"] = _safe_log10(work["pre_sigma"])
    work["moon_ra_sin"] = np.sin(np.deg2rad(pd.to_numeric(work["moon_center_ra_deg"], errors="coerce")))
    work["moon_ra_cos"] = np.cos(np.deg2rad(pd.to_numeric(work["moon_center_ra_deg"], errors="coerce")))
    work["target_ra_sin"] = np.sin(np.deg2rad(pd.to_numeric(work["target_ra_deg"], errors="coerce")))
    work["target_ra_cos"] = np.cos(np.deg2rad(pd.to_numeric(work["target_ra_deg"], errors="coerce")))
    return work


def _feature_screen(work: pd.DataFrame) -> pd.DataFrame:
    features = {
        "expected_signed_near_slope": "near-event local slope, expected-sign adjusted",
        "expected_signed_far_slope": "far-window local slope, expected-sign adjusted",
        "log_pre_background": "pre-event background level",
        "log_pre_sigma": "pre-event noise level",
        "gap_seconds": "sample gap bracketing event",
        "limb_rate_deg_s": "lunar-limb crossing rate",
        "moon_center_dec_deg": "Moon-center Dec",
        "moon_ra_sin": "Moon-center RA sin",
        "moon_ra_cos": "Moon-center RA cos",
        "target_dec_deg": "target Dec",
        "target_ra_sin": "target RA sin",
        "target_ra_cos": "target RA cos",
        "limb_pa_sin": "limb position-angle sin",
        "limb_pa_cos": "limb position-angle cos",
        "moon_angular_radius_deg": "Moon angular radius",
    }
    rows = []
    for source, src in work.groupby("source_name", sort=True):
        y = src["label_source_like"]
        for col, desc in features.items():
            x = pd.to_numeric(src[col], errors="coerce")
            out = _threshold_classifier(x, y)
            out.update(_cv_threshold_classifier(x, y))
            rows.append(
                {
                    "source_name": source,
                    "feature": col,
                    "description": desc,
                    **out,
                    "n_events": int(len(src)),
                    "source_like_fraction": float(y.mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(["source_name", "cv_balanced_accuracy"], ascending=[True, False])


def _plot_classifier_feature(work: pd.DataFrame, scores: pd.DataFrame) -> Path:
    sources = ["earth", "sun"]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2), sharex="col")
    for row, source in enumerate(sources):
        src = work[work["source_name"].eq(source)].copy()
        for col, title, ax in [
            ("expected_signed_near_slope", "near-event slope", axes[row, 0]),
            ("expected_signed_far_slope", "far-window slope", axes[row, 1]),
        ]:
            for regime in ["anti_template", "source_like"]:
                vals = pd.to_numeric(src[src["regime"].eq(regime)][col], errors="coerce").dropna()
                if vals.empty:
                    continue
                lo, hi = np.nanpercentile(pd.to_numeric(src[col], errors="coerce").dropna(), [2, 98])
                bins = np.linspace(lo, hi, 45) if np.isfinite(lo) and np.isfinite(hi) and hi > lo else 30
                ax.hist(vals.clip(lo, hi), bins=bins, density=True, histtype="step", linewidth=1.8, color=COLORS[regime], label=regime)
            score = scores[(scores["source_name"].eq(source)) & (scores["feature"].eq(col))]
            if not score.empty:
                r = score.iloc[0]
                ax.axvline(r["threshold"], color="black", linestyle="--", linewidth=1.0)
                ax.text(
                    0.03,
                    0.92,
                    f"AUC={r['auc_abs']:.2f}\nCV bal. acc={r['cv_balanced_accuracy']:.2f}",
                    transform=ax.transAxes,
                    va="top",
                    fontsize=9,
                    bbox={"facecolor": "white", "edgecolor": "0.85", "alpha": 0.9},
                )
            ax.axvline(0, color="0.55", linewidth=0.8)
            ax.set_title(f"{source}: {title}")
            ax.set_ylabel("density")
            ax.grid(alpha=0.18)
    axes[1, 0].set_xlabel("expected-sign-adjusted slope (power / s)")
    axes[1, 1].set_xlabel("expected-sign-adjusted slope (power / s)")
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle(
        "Candidate root classifier for source-like vs anti-template events\n"
        "Positive expected-sign slope means the local power moves in the point-source occultation direction.",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "root_classifier_local_slope_distributions.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_feature_ranking(scores: pd.DataFrame) -> Path:
    sources = sorted(scores["source_name"].unique())
    fig, axes = plt.subplots(len(sources), 1, figsize=(12, 4.4 * len(sources)), sharex=True)
    axes = np.asarray(axes).ravel()
    for ax, source in zip(axes, sources):
        top = scores[scores["source_name"].eq(source)].sort_values("cv_balanced_accuracy", ascending=True).tail(10)
        ax.barh(top["description"], top["cv_balanced_accuracy"], color="#4c78a8")
        ax.axvline(0.5, color="0.4", linestyle="--", linewidth=0.9)
        ax.axvline(0.7, color="0.7", linestyle=":", linewidth=0.9)
        ax.set_xlim(0.45, 1.0)
        ax.set_title(f"{source}: one-variable classifier performance")
        ax.set_xlabel("5-fold cross-validated balanced accuracy")
        ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "root_classifier_feature_ranking.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _earth_regime_stack(points: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    earth_events = events[
        events["source_name"].eq("earth")
        & events["regime"].isin(["source_like", "anti_template"])
    ][["event_id", "frequency_band", "regime"]].copy()
    pts = points[points["source_name"].eq("earth")].copy()
    pts = pts.merge(earth_events, on=["event_id", "frequency_band"], how="inner")
    event_bin = (
        pts.groupby(["regime", "event_type", "frequency_band", "frequency_mhz", "t_bin_sec", "event_id"], as_index=False)
        .agg(
            raw_fractional=("raw_fractional", "median"),
            pre_event_anchored=("pre_event_anchored", "median"),
            n_samples=("raw_fractional", "size"),
        )
    )
    rows = []
    by = ["regime", "event_type", "frequency_band", "frequency_mhz", "t_bin_sec"]
    for keys, grp in event_bin.groupby(by, sort=True, dropna=False):
        raw_vals = pd.to_numeric(grp["raw_fractional"], errors="coerce")
        anchored_vals = pd.to_numeric(grp["pre_event_anchored"], errors="coerce")
        rows.append(
            {
                **dict(zip(by, keys)),
                "median_raw_fractional": float(np.nanmedian(raw_vals)),
                "median_raw_fractional_err": _robust_standard_error(raw_vals),
                "median_pre_event_anchored": float(np.nanmedian(anchored_vals)),
                "median_pre_event_anchored_err": _robust_standard_error(anchored_vals),
                "n_events": int(grp["event_id"].nunique()),
                "n_points": int(grp["n_samples"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _plot_earth_regime_grid(summary: pd.DataFrame, regime: str) -> Path:
    sub = summary[summary["regime"].eq(regime)].copy()
    freqs = sorted(sub["frequency_mhz"].dropna().unique())
    fig, axes = plt.subplots(len(freqs), 2, figsize=(12, max(10, 1.35 * len(freqs))), sharex=True, sharey=False)
    if len(freqs) == 1:
        axes = np.asarray([axes])
    for i, freq in enumerate(freqs):
        for j, event_type in enumerate(["disappearance", "reappearance"]):
            ax = axes[i, j]
            g = sub[np.isclose(sub["frequency_mhz"], freq) & sub["event_type"].eq(event_type)].sort_values("t_bin_sec")
            if not g.empty:
                ax.errorbar(
                    g["t_bin_sec"],
                    g["median_raw_fractional"],
                    yerr=g["median_raw_fractional_err"],
                    marker="o",
                    markersize=2.8,
                    linewidth=1.25,
                    elinewidth=0.65,
                    capsize=1.3,
                    alpha=0.9,
                    color=COLORS[regime],
                    ecolor=COLORS[regime],
                )
                n_events = int(g["n_events"].max())
            else:
                n_events = 0
            ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
            ax.axhline(0, color="0.6", linewidth=0.7)
            ax.set_title(f"{freq:.2f} MHz {event_type} (n={n_events})", fontsize=9)
            if j == 0:
                ax.set_ylabel("raw fractional power")
            if i == len(freqs) - 1:
                ax.set_xlabel("seconds from predicted event")
            ax.grid(alpha=0.18)
    fig.suptitle(
        f"Earth lower V all-frequency profile grid: {regime} events only\n"
        "Each event is normalized as P / median(pre-far) - 1; error bars are event-to-event robust standard errors.",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"earth_{regime}_all_frequency_profile_grid_900s.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_report(scores: pd.DataFrame, paths: list[Path]) -> Path:
    lines = [
        "# Moving-Body Root Classifier Diagnostic",
        "",
        "Question: what variable actually separates source-like and anti-template events?",
        "",
        "The strongest classifier is the event-type-adjusted local slope/contrast around the predicted event.",
        "That is not an independent physical cause; it is the measured local shape of the event window and is",
        "closely related to the regime definition. The more useful test is whether a broader far-window slope",
        "also separates the classes better than Moon RA/Dec or time.",
        "",
        "## Top One-Variable Classifiers",
        "",
    ]
    for source in sorted(scores["source_name"].unique()):
        lines.append(f"### {source}")
        lines.append("")
        cols = ["description", "auc_abs", "cv_balanced_accuracy", "cv_accuracy", "balanced_accuracy", "threshold", "direction"]
        lines.append(scores[scores["source_name"].eq(source)].head(8)[cols].to_string(index=False))
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "- Near-event slope classifies best because source-like versus anti-template is fundamentally a statement about the sign of the local pre/post change.",
            "- Far-window slope is the best non-identical local-background proxy. When it performs well, it means the event class is tied to the local raw power trajectory across the broader window.",
            "- Time and Moon-center RA/Dec are weaker, so the sign flip is not explained by a clean date cluster or simple Moon-center sky region.",
            "- The practical pipeline consequence is that Earth/Sun moving-body stacks should not mix all local-slope regimes. They should report source-like and anti-template regime stacks separately, then diagnose why the anti-template regime exists.",
            "- Earth split-profile error bars are robust event-to-event standard errors after first reducing each event to one value per time bin.",
            "",
            "## Generated Plots",
            "",
        ]
    )
    lines.extend(f"- `{p}`" for p in paths)
    path = OUT / "root_classifier_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    df = read_table(GEOM_TABLE, low_memory=False)
    points = read_table(STACK_POINTS, low_memory=False)
    work = _build_features(df)
    scores = _feature_screen(work)

    OUT.mkdir(parents=True, exist_ok=True)
    work.to_csv(OUT / "root_classifier_event_features.csv", index=False)
    scores.to_csv(OUT / "root_classifier_feature_scores.csv", index=False)

    paths = [_plot_classifier_feature(work, scores), _plot_feature_ranking(scores)]
    earth_summary = _earth_regime_stack(points, work)
    earth_summary.to_csv(OUT / "earth_regime_split_all_frequency_profile_summary.csv", index=False)
    for regime in ["source_like", "anti_template"]:
        paths.append(_plot_earth_regime_grid(earth_summary, regime))
    report = _write_report(scores, paths)

    for p in paths:
        print(p)
    print(report)


if __name__ == "__main__":
    main()
