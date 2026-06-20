#!/usr/bin/env python
"""Investigate time/geometry cuts for raw expected-sign occultation events.

The input sign is the raw pre/post median behavior already computed by
``flag_raw_occultation_candidates.py``:

    disappearance source-like: post median < pre median
    reappearance source-like: post median > pre median

This script does not use the sign to define the cut itself.  It asks whether
source-like rows/events cluster in independent variables such as event time,
Moon-center sky position, lunar-limb speed, or event type.
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
from rylevonberg.table_io import read_table
from astropy.coordinates import FK4, Galactic, SkyCoord
import astropy.units as u

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.util import ensure_dir, write_json, software_versions  # noqa: E402


RAW_SCORE_ROOT = ROOT / "outputs/lower_v_structure_selected_stacks_all_sources_v1"
EARTH_SUN_EVENTS = ROOT / "outputs/control_survey_earth_sun_postnov1974_v2/02_events/predicted_events.csv"
BRIGHT_EVENTS = ROOT / "outputs/control_survey_bright_sources_postnov1974_v1/02_events/predicted_events.csv"
OUT = ROOT / "outputs/sign_geometry_temporal_cut_investigation_v1"

SOURCE_EVENT_PATH = {
    "sun": EARTH_SUN_EVENTS,
    "fornax_a": BRIGHT_EVENTS,
}
SOURCE_LABEL = {
    "sun": "Sun",
    "fornax_a": "Fornax A",
}


def _read(path: Path) -> pd.DataFrame:
    return read_table(path, low_memory=False)


def _moon_ra_dec_from_xyz(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x = pd.to_numeric(df["moon_center_x"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df["moon_center_y"], errors="coerce").to_numpy(dtype=float)
    z = pd.to_numeric(df["moon_center_z"], errors="coerce").to_numpy(dtype=float)
    norm = np.sqrt(x * x + y * y + z * z)
    x = np.divide(x, norm, out=np.full_like(x, np.nan), where=norm > 0)
    y = np.divide(y, norm, out=np.full_like(y, np.nan), where=norm > 0)
    z = np.divide(z, norm, out=np.full_like(z, np.nan), where=norm > 0)
    ra = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0
    dec = np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))
    return ra, dec


def _add_geometry(events: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    ra, dec = _moon_ra_dec_from_xyz(out)
    out["moon_center_ra_deg"] = ra
    out["moon_center_dec_deg"] = dec
    good = np.isfinite(ra) & np.isfinite(dec)
    gal_l = np.full(len(out), np.nan)
    gal_b = np.full(len(out), np.nan)
    if np.any(good):
        coords = SkyCoord(ra=ra[good] * u.deg, dec=dec[good] * u.deg, frame=FK4(equinox="B1950"))
        gal = coords.transform_to(Galactic())
        gal_l[good] = gal.l.deg
        gal_b[good] = gal.b.deg
    out["moon_center_gal_l_deg"] = gal_l
    out["moon_center_gal_b_deg"] = gal_b
    pre = pd.to_numeric(out["pre_limb_angle_deg"], errors="coerce")
    post = pd.to_numeric(out["post_limb_angle_deg"], errors="coerce")
    gap = pd.to_numeric(out["gap_seconds"], errors="coerce")
    out["limb_speed_deg_per_min"] = (post - pre).abs() / gap * 60.0
    return out


def _load_source(source: str) -> pd.DataFrame:
    scores = _read(RAW_SCORE_ROOT / source / "raw_occultation_candidate_scores.csv")
    scores = scores[scores["antenna"].astype(str).eq("rv2_coarse")].copy()
    events = _read(SOURCE_EVENT_PATH[source])
    events = events[
        events["source_name"].astype(str).str.lower().eq(source)
        & events["antenna"].astype(str).eq("rv2_coarse")
    ].copy()
    events = _add_geometry(events)
    keys = ["event_id", "event_type", "predicted_event_time", "frequency_band", "frequency_mhz", "antenna"]
    keep_cols = keys + [
        "source_ra_deg",
        "source_dec_deg",
        "moon_center_ra_deg",
        "moon_center_dec_deg",
        "moon_center_gal_l_deg",
        "moon_center_gal_b_deg",
        "moon_angular_radius_deg",
        "limb_speed_deg_per_min",
        "gap_seconds",
    ]
    merged = scores.merge(events[keep_cols], on=keys, how="left", validate="many_to_one")
    merged["predicted_event_time"] = pd.to_datetime(merged["predicted_event_time"], errors="coerce")
    t0 = merged["predicted_event_time"].min()
    merged["days_since_first"] = (merged["predicted_event_time"] - t0).dt.total_seconds() / 86400.0
    merged["month"] = merged["predicted_event_time"].dt.to_period("M").astype(str)
    merged["day_of_year"] = merged["predicted_event_time"].dt.dayofyear.astype(float)
    merged["source_like_row"] = pd.to_numeric(merged["predicted_signed_delta"], errors="coerce") > 0
    merged["anti_template_row"] = pd.to_numeric(merged["predicted_signed_delta"], errors="coerce") < 0
    usable = merged["usable"].astype(bool)
    usable &= pd.to_numeric(merged["predicted_n_pre"], errors="coerce").ge(4)
    usable &= pd.to_numeric(merged["predicted_n_post"], errors="coerce").ge(4)
    usable &= pd.to_numeric(merged["predicted_signed_delta"], errors="coerce").notna()
    merged["analysis_usable"] = usable
    return merged


def _event_level(rows: pd.DataFrame) -> pd.DataFrame:
    usable = rows[rows["analysis_usable"]].copy()
    group_keys = ["source_name", "event_id", "event_type", "predicted_event_time"]
    aggs = {
        "source_like_row": "mean",
        "anti_template_row": "mean",
        "frequency_band": "nunique",
        "valid_fraction": "median",
        "valid_samples": "median",
        "predicted_step_z": lambda s: float(np.nanmedian(np.abs(pd.to_numeric(s, errors="coerce")))),
        "best_abs_offset_s": "median",
        "moon_center_ra_deg": "median",
        "moon_center_dec_deg": "median",
        "moon_center_gal_l_deg": "median",
        "moon_center_gal_b_deg": "median",
        "moon_angular_radius_deg": "median",
        "limb_speed_deg_per_min": "median",
        "days_since_first": "median",
        "day_of_year": "median",
        "month": "first",
    }
    ev = usable.groupby(group_keys, as_index=False).agg(aggs)
    ev = ev.rename(
        columns={
            "source_like_row": "source_like_fraction",
            "anti_template_row": "anti_template_fraction",
            "frequency_band": "n_usable_bands",
            "predicted_step_z": "median_abs_predicted_step_z",
        }
    )
    ev["source_like_event"] = ev["source_like_fraction"].ge(0.60)
    ev["anti_template_event"] = ev["source_like_fraction"].le(0.40)
    ev["ambiguous_event"] = ~(ev["source_like_event"] | ev["anti_template_event"])
    return ev


def _bin_feature(values: pd.Series, bins: int = 8) -> pd.Series:
    vals = pd.to_numeric(values, errors="coerce")
    if vals.dropna().nunique() <= 1:
        return pd.Series(["all"] * len(vals), index=values.index)
    try:
        return pd.qcut(vals, q=min(bins, vals.dropna().nunique()), duplicates="drop").astype(str)
    except ValueError:
        return pd.cut(vals, bins=min(bins, vals.dropna().nunique()), duplicates="drop").astype(str)


def _safe_nanmedian(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(np.median(vals)) if vals.size else np.nan


def _feature_bin_summary(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    features = [
        "month",
        "event_type",
        "moon_center_ra_deg",
        "moon_center_dec_deg",
        "moon_center_gal_l_deg",
        "moon_center_gal_b_deg",
        "moon_angular_radius_deg",
        "limb_speed_deg_per_min",
        "days_since_first",
        "day_of_year",
        "valid_fraction",
        "valid_samples",
        "median_abs_predicted_step_z",
        "best_abs_offset_s",
    ]
    base = float(events["source_like_event"].mean()) if len(events) else np.nan
    for feature in features:
        if feature in ["month", "event_type"]:
            labels = events[feature].astype(str)
        else:
            labels = _bin_feature(events[feature])
        for label, grp in events.groupby(labels, dropna=False):
            n = len(grp)
            if n == 0:
                continue
            frac = float(grp["source_like_event"].mean())
            rows.append(
                {
                    "feature": feature,
                    "bin": str(label),
                    "n_events": int(n),
                    "source_like_fraction": frac,
                    "anti_template_fraction": float(grp["anti_template_event"].mean()),
                    "ambiguous_fraction": float(grp["ambiguous_event"].mean()),
                    "baseline_source_like_fraction": base,
                    "lift_vs_baseline": frac - base,
                    "median_moon_ra_deg": _safe_nanmedian(grp["moon_center_ra_deg"]),
                    "median_moon_dec_deg": _safe_nanmedian(grp["moon_center_dec_deg"]),
                    "median_days_since_first": _safe_nanmedian(grp["days_since_first"]),
                }
            )
    return pd.DataFrame(rows).sort_values(["feature", "bin"]).reset_index(drop=True)


def _candidate_bin_cuts(bin_summary: pd.DataFrame, n_total: int, min_coverage: float, min_lift: float) -> pd.DataFrame:
    if bin_summary.empty:
        return pd.DataFrame()
    work = bin_summary.copy()
    work["coverage"] = work["n_events"] / max(1, int(n_total))
    base = pd.to_numeric(work["baseline_source_like_fraction"], errors="coerce")
    work = work[
        work["coverage"].ge(float(min_coverage))
        & work["lift_vs_baseline"].ge(float(min_lift))
        & work["source_like_fraction"].ge(base + float(min_lift))
    ].copy()
    return work.sort_values(["lift_vs_baseline", "source_like_fraction", "coverage"], ascending=[False, False, False])


def _best_threshold_cuts(events: pd.DataFrame, min_coverage: float) -> pd.DataFrame:
    rows = []
    base = float(events["source_like_event"].mean()) if len(events) else np.nan
    features = [
        "moon_center_ra_deg",
        "moon_center_dec_deg",
        "moon_center_gal_l_deg",
        "moon_center_gal_b_deg",
        "moon_angular_radius_deg",
        "limb_speed_deg_per_min",
        "days_since_first",
        "day_of_year",
        "valid_fraction",
        "valid_samples",
        "median_abs_predicted_step_z",
        "best_abs_offset_s",
    ]
    y = events["source_like_event"].to_numpy(dtype=bool)
    for feature in features:
        x = pd.to_numeric(events[feature], errors="coerce").to_numpy(dtype=float)
        good = np.isfinite(x)
        if np.count_nonzero(good) < 20 or np.unique(x[good]).size < 4:
            continue
        thresholds = np.nanpercentile(x[good], np.arange(10, 91, 5))
        for threshold in np.unique(thresholds):
            for direction, mask in [
                ("<=", x <= threshold),
                (">", x > threshold),
            ]:
                sel = good & mask
                n = int(np.count_nonzero(sel))
                coverage = n / max(1, len(events))
                if coverage < float(min_coverage):
                    continue
                frac = float(np.mean(y[sel]))
                rows.append(
                    {
                        "feature": feature,
                        "direction": direction,
                        "threshold": float(threshold),
                        "n_events": n,
                        "coverage": coverage,
                        "source_like_fraction": frac,
                        "baseline_source_like_fraction": base,
                        "lift_vs_baseline": frac - base,
                    }
                )
    return pd.DataFrame(rows).sort_values(["lift_vs_baseline", "source_like_fraction", "coverage"], ascending=[False, False, False])


def _geometry_time_feature_matrix(events: pd.DataFrame) -> pd.DataFrame:
    """Return features allowed for an independent time/geometry cut audit."""
    out = pd.DataFrame(index=events.index)
    for col in [
        "moon_center_dec_deg",
        "moon_center_gal_b_deg",
        "moon_angular_radius_deg",
        "limb_speed_deg_per_min",
        "days_since_first",
        "day_of_year",
    ]:
        out[col] = pd.to_numeric(events[col], errors="coerce")
    for col in ["moon_center_ra_deg", "moon_center_gal_l_deg"]:
        angle = np.deg2rad(pd.to_numeric(events[col], errors="coerce").to_numpy(dtype=float))
        out[f"{col}_sin"] = np.sin(angle)
        out[f"{col}_cos"] = np.cos(angle)
    out["is_disappearance"] = events["event_type"].astype(str).str.lower().eq("disappearance").astype(float)
    return out


def _candidate_rules_for_training(
    features: pd.DataFrame,
    y: np.ndarray,
    min_coverage: float,
) -> pd.DataFrame:
    rows = []
    base = float(np.mean(y)) if len(y) else np.nan
    for feature in features.columns:
        x = pd.to_numeric(features[feature], errors="coerce").to_numpy(dtype=float)
        good = np.isfinite(x)
        if np.count_nonzero(good) < 20 or np.unique(x[good]).size < 2:
            continue
        if np.unique(x[good]).size == 2:
            thresholds = [0.5]
        else:
            thresholds = np.nanpercentile(x[good], np.arange(10, 91, 5))
        for threshold in np.unique(thresholds):
            for direction, mask in [("<=", x <= threshold), (">", x > threshold)]:
                sel = good & mask
                n = int(np.count_nonzero(sel))
                coverage = n / max(1, len(y))
                if coverage < min_coverage:
                    continue
                frac = float(np.mean(y[sel]))
                rows.append(
                    {
                        "feature": feature,
                        "direction": direction,
                        "threshold": float(threshold),
                        "train_n_selected": n,
                        "train_coverage": coverage,
                        "train_source_like_fraction": frac,
                        "train_baseline_source_like_fraction": base,
                        "train_lift_vs_baseline": frac - base,
                    }
                )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["train_lift_vs_baseline", "train_source_like_fraction", "train_coverage"],
        ascending=[False, False, False],
    )


def _apply_rule(features: pd.DataFrame, rule: pd.Series) -> np.ndarray:
    x = pd.to_numeric(features[rule["feature"]], errors="coerce").to_numpy(dtype=float)
    good = np.isfinite(x)
    if rule["direction"] == "<=":
        return good & (x <= float(rule["threshold"]))
    return good & (x > float(rule["threshold"]))


def _cross_validated_geometry_time_rules(
    events: pd.DataFrame,
    min_coverage: float,
    n_folds: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Choose the best time/geometry threshold on training folds and test it held out."""
    features = _geometry_time_feature_matrix(events)
    y = events["source_like_event"].to_numpy(dtype=bool)
    n = len(events)
    if n < n_folds * 10:
        empty = pd.DataFrame()
        return empty, empty
    order = np.argsort(pd.to_datetime(events["predicted_event_time"], errors="coerce").astype("int64").to_numpy())
    fold_ids = np.empty(n, dtype=int)
    for rank, idx in enumerate(order):
        fold_ids[idx] = rank % n_folds

    rows = []
    for fold in range(n_folds):
        train = fold_ids != fold
        test = fold_ids == fold
        rules = _candidate_rules_for_training(features.loc[train], y[train], min_coverage)
        if rules.empty:
            continue
        rule = rules.iloc[0]
        test_sel = _apply_rule(features.loc[test], rule)
        train_sel = _apply_rule(features.loc[train], rule)
        n_test_selected = int(np.count_nonzero(test_sel))
        test_base = float(np.mean(y[test])) if np.count_nonzero(test) else np.nan
        test_frac = float(np.mean(y[test][test_sel])) if n_test_selected else np.nan
        rows.append(
            {
                "fold": int(fold),
                "feature": rule["feature"],
                "direction": rule["direction"],
                "threshold": float(rule["threshold"]),
                "train_n": int(np.count_nonzero(train)),
                "test_n": int(np.count_nonzero(test)),
                "train_n_selected": int(np.count_nonzero(train_sel)),
                "test_n_selected": n_test_selected,
                "train_coverage": float(np.count_nonzero(train_sel) / max(1, np.count_nonzero(train))),
                "test_coverage": float(n_test_selected / max(1, np.count_nonzero(test))),
                "train_source_like_fraction": float(np.mean(y[train][train_sel])) if np.count_nonzero(train_sel) else np.nan,
                "train_baseline_source_like_fraction": float(np.mean(y[train])) if np.count_nonzero(train) else np.nan,
                "train_lift_vs_baseline": float(np.mean(y[train][train_sel]) - np.mean(y[train]))
                if np.count_nonzero(train_sel)
                else np.nan,
                "test_source_like_fraction": test_frac,
                "test_baseline_source_like_fraction": test_base,
                "test_lift_vs_baseline": test_frac - test_base if np.isfinite(test_frac) and np.isfinite(test_base) else np.nan,
            }
        )
    fold_table = pd.DataFrame(rows)
    if fold_table.empty:
        return fold_table, pd.DataFrame()
    valid = fold_table[np.isfinite(pd.to_numeric(fold_table["test_lift_vs_baseline"], errors="coerce"))]
    summary = pd.DataFrame(
        [
            {
                "n_folds": int(len(fold_table)),
                "mean_train_lift": float(fold_table["train_lift_vs_baseline"].mean()),
                "mean_test_lift": float(valid["test_lift_vs_baseline"].mean()) if len(valid) else np.nan,
                "median_test_lift": float(valid["test_lift_vs_baseline"].median()) if len(valid) else np.nan,
                "min_test_lift": float(valid["test_lift_vs_baseline"].min()) if len(valid) else np.nan,
                "max_test_lift": float(valid["test_lift_vs_baseline"].max()) if len(valid) else np.nan,
                "mean_test_coverage": float(fold_table["test_coverage"].mean()),
                "folds_positive_test_lift": int((valid["test_lift_vs_baseline"] > 0).sum()) if len(valid) else 0,
                "folds_with_selected_events": int((fold_table["test_n_selected"] > 0).sum()),
                "feature_family": "time_geometry_only",
            }
        ]
    )
    return fold_table, summary


def _permutation_max_lift(events: pd.DataFrame, bin_summary: pd.DataFrame, seed: int, n_perm: int) -> pd.DataFrame:
    """Estimate whether the strongest binned enrichment exceeds shuffled labels."""
    rng = np.random.default_rng(seed)
    observed = float(bin_summary["lift_vs_baseline"].max()) if not bin_summary.empty else np.nan
    if not np.isfinite(observed) or events.empty:
        return pd.DataFrame([{"observed_max_lift": observed, "empirical_p": np.nan, "n_permutations": n_perm}])
    work = events.copy()
    labels = work["source_like_event"].to_numpy(dtype=bool)
    max_lifts = []
    for _ in range(int(n_perm)):
        work["source_like_event"] = rng.permutation(labels)
        bs = _feature_bin_summary(work)
        max_lifts.append(float(bs["lift_vs_baseline"].max()) if not bs.empty else np.nan)
    arr = np.asarray(max_lifts, dtype=float)
    arr = arr[np.isfinite(arr)]
    p = (1 + np.count_nonzero(arr >= observed)) / (1 + len(arr)) if len(arr) else np.nan
    return pd.DataFrame(
        [
            {
                "observed_max_lift": observed,
                "median_permuted_max_lift": float(np.nanmedian(arr)) if len(arr) else np.nan,
                "p95_permuted_max_lift": float(np.nanpercentile(arr, 95)) if len(arr) else np.nan,
                "empirical_p": float(p) if np.isfinite(p) else np.nan,
                "n_permutations": int(n_perm),
            }
        ]
    )


def _plot_time_geometry(events: pd.DataFrame, source: str, out_dir: Path) -> Path:
    color = np.where(events["source_like_event"], "#2ca02c", np.where(events["anti_template_event"], "#d62728", "0.55"))
    label = SOURCE_LABEL.get(source, source)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    axes[0, 0].scatter(events["moon_center_dec_deg"], events["moon_center_ra_deg"], c=color, s=18, alpha=0.75)
    axes[0, 0].set_xlabel("Moon-center Dec deg")
    axes[0, 0].set_ylabel("Moon-center RA deg")
    axes[0, 0].set_title("Moon-center sky position")
    axes[0, 1].scatter(events["days_since_first"], events["source_like_fraction"], c=color, s=18, alpha=0.75)
    axes[0, 1].set_xlabel("days since first event")
    axes[0, 1].set_ylabel("event source-like fraction across bands")
    axes[0, 1].set_title("Event sign fraction vs time")
    axes[1, 0].scatter(events["moon_center_gal_l_deg"], events["moon_center_gal_b_deg"], c=color, s=18, alpha=0.75)
    axes[1, 0].set_xlabel("Moon-center Galactic l deg")
    axes[1, 0].set_ylabel("Moon-center Galactic b deg")
    axes[1, 0].set_title("Galactic coordinates")
    axes[1, 1].scatter(events["limb_speed_deg_per_min"], events["source_like_fraction"], c=color, s=18, alpha=0.75)
    axes[1, 1].set_xlabel("limb speed deg/min")
    axes[1, 1].set_ylabel("event source-like fraction across bands")
    axes[1, 1].set_title("Limb speed")
    fig.suptitle(f"{label}: source-like raw pre/post sign vs time/geometry")
    path = out_dir / f"{source}_sign_time_geometry_scatter.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _plot_feature_enrichment(bin_summary: pd.DataFrame, source: str, out_dir: Path) -> Path:
    top = bin_summary.sort_values("lift_vs_baseline", ascending=False).head(24).copy()
    labels = top["feature"].astype(str) + "\n" + top["bin"].astype(str).str.slice(0, 24)
    fig, ax = plt.subplots(figsize=(12, max(5, 0.34 * len(top))))
    y = np.arange(len(top))
    ax.barh(y, top["lift_vs_baseline"], color="#4c78a8")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("source-like fraction lift above baseline")
    ax.set_title(f"{SOURCE_LABEL.get(source, source)}: strongest binned time/geometry enrichments")
    for yi, (_, row) in enumerate(top.iterrows()):
        ax.text(row["lift_vs_baseline"], yi, f" n={int(row['n_events'])}", va="center", fontsize=8)
    fig.tight_layout()
    path = out_dir / f"{source}_feature_enrichment_top_bins.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _write_report(
    out_dir: Path,
    source: str,
    rows: pd.DataFrame,
    events: pd.DataFrame,
    bin_summary: pd.DataFrame,
    candidate_bins: pd.DataFrame,
    thresholds: pd.DataFrame,
    permutation: pd.DataFrame,
    cv_folds: pd.DataFrame,
    cv_summary: pd.DataFrame,
    paths: list[Path],
) -> Path:
    n_rows = len(rows)
    n_usable_rows = int(rows["analysis_usable"].sum())
    n_events = len(events)
    base = float(events["source_like_event"].mean()) if n_events else np.nan
    top_bins = candidate_bins.head(12)
    top_thresh = thresholds.head(12)
    perm_line = permutation.to_string(index=False) if not permutation.empty else "No permutation summary."
    cv_line = cv_summary.to_string(index=False) if not cv_summary.empty else "No cross-validation summary."
    cv_fold_line = (
        cv_folds[
            [
                "fold",
                "feature",
                "direction",
                "threshold",
                "test_n_selected",
                "test_coverage",
                "test_source_like_fraction",
                "test_baseline_source_like_fraction",
                "test_lift_vs_baseline",
            ]
        ].to_string(index=False)
        if not cv_folds.empty
        else "No cross-validation fold table."
    )
    has_candidate = (not top_bins.empty) or (not top_thresh.empty)
    interpretation = [
        "The current result supports trying a cut only if the enriched bins/thresholds are physically interpretable,",
        "cover a non-trivial fraction of events, and remain stronger than the shuffled-label expectation.",
    ]
    if not has_candidate:
        interpretation.append("No robust candidate cut met the configured coverage/lift thresholds.")
    lines = [
        f"# {SOURCE_LABEL.get(source, source)} Sign Geometry/Temporal Cut Investigation",
        "",
        "## Question",
        "",
        "Can raw source-like median changes be converted into a time- or geometry-informed cut,",
        "instead of simply selecting events because their pre/post sign already agrees with the template?",
        "",
        "## Class Definition",
        "",
        "- Row source-like: `predicted_signed_delta > 0` from raw pre/post medians.",
        "- Event source-like: at least 60% of usable frequency rows for that event have source-like sign.",
        "- Event anti-template: at most 40% of usable rows have source-like sign.",
        "- Rows must be usable and have at least 4 valid pre and 4 valid post samples.",
        "",
        "## Counts",
        "",
        f"- scored rows: {n_rows}",
        f"- analysis-usable rows: {n_usable_rows}",
        f"- event-level rows: {n_events}",
        f"- baseline event source-like fraction: {base:.3f}" if np.isfinite(base) else "- baseline event source-like fraction: NaN",
        "",
        "## Top Candidate Binned Cuts",
        "",
        top_bins[
            [
                "feature",
                "bin",
                "n_events",
                "coverage",
                "source_like_fraction",
                "baseline_source_like_fraction",
                "lift_vs_baseline",
            ]
        ].to_string(index=False)
        if not top_bins.empty
        else "No binned cut exceeded the configured threshold.",
        "",
        "## Top One-Threshold Cuts",
        "",
        top_thresh[
            [
                "feature",
                "direction",
                "threshold",
                "n_events",
                "coverage",
                "source_like_fraction",
                "baseline_source_like_fraction",
                "lift_vs_baseline",
            ]
        ].to_string(index=False)
        if not top_thresh.empty
        else "No threshold cut exceeded the configured threshold.",
        "",
        "## Permutation Check",
        "",
        perm_line,
        "",
        "## Held-Out Geometry/Time Cut Audit",
        "",
        "This audit only allows time and geometry variables, chooses one threshold rule on each",
        "training fold, then evaluates it on held-out events. It is intentionally simple:",
        "a useful physics cut should generalize even in this restricted test.",
        "",
        cv_line,
        "",
        "### Fold Details",
        "",
        cv_fold_line,
        "",
        "## Interpretation",
        "",
        *[f"- {line}" for line in interpretation],
        "",
        "## Output Plots",
        "",
    ]
    lines.extend(f"- `{path}`" for path in paths)
    report = out_dir / f"{source}_sign_geometry_temporal_cut_report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run_source(source: str, out_root: Path, min_coverage: float, min_lift: float, n_perm: int, seed: int) -> Path:
    out_dir = ensure_dir(out_root / source)
    rows = _load_source(source)
    events = _event_level(rows)
    bin_summary = _feature_bin_summary(events)
    candidate_bins = _candidate_bin_cuts(bin_summary, len(events), min_coverage, min_lift)
    thresholds = _best_threshold_cuts(events, min_coverage)
    thresholds = thresholds[thresholds["lift_vs_baseline"].ge(min_lift)].copy()
    permutation = _permutation_max_lift(events, bin_summary, seed=seed, n_perm=n_perm)
    cv_folds, cv_summary = _cross_validated_geometry_time_rules(events, min_coverage=min_coverage)

    rows.to_csv(out_dir / f"{source}_row_level_sign_geometry.csv", index=False)
    events.to_csv(out_dir / f"{source}_event_level_sign_geometry.csv", index=False)
    bin_summary.to_csv(out_dir / f"{source}_feature_bin_source_like_summary.csv", index=False)
    candidate_bins.to_csv(out_dir / f"{source}_candidate_binned_cuts.csv", index=False)
    thresholds.to_csv(out_dir / f"{source}_candidate_threshold_cuts.csv", index=False)
    permutation.to_csv(out_dir / f"{source}_permutation_enrichment_check.csv", index=False)
    cv_folds.to_csv(out_dir / f"{source}_heldout_geometry_time_cut_folds.csv", index=False)
    cv_summary.to_csv(out_dir / f"{source}_heldout_geometry_time_cut_summary.csv", index=False)

    paths = [
        _plot_time_geometry(events, source, out_dir),
        _plot_feature_enrichment(bin_summary, source, out_dir),
    ]
    return _write_report(out_dir, source, rows, events, bin_summary, candidate_bins, thresholds, permutation, cv_folds, cv_summary, paths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", default="sun,fornax_a")
    parser.add_argument("--out-dir", default=str(OUT))
    parser.add_argument("--min-coverage", type=float, default=0.15)
    parser.add_argument("--min-lift", type=float, default=0.15)
    parser.add_argument("--n-permutations", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260603)
    args = parser.parse_args()

    out_root = ensure_dir(args.out_dir)
    config = {
        "sources": args.sources,
        "min_coverage": float(args.min_coverage),
        "min_lift": float(args.min_lift),
        "n_permutations": int(args.n_permutations),
        "seed": int(args.seed),
        "software_versions": software_versions(),
    }
    write_json(out_root / "run_config.json", config)
    reports = []
    for source in [x.strip().lower() for x in str(args.sources).split(",") if x.strip()]:
        if source not in SOURCE_EVENT_PATH:
            raise ValueError(f"unsupported source: {source}")
        reports.append(run_source(source, out_root, float(args.min_coverage), float(args.min_lift), int(args.n_permutations), int(args.seed)))
    index = out_root / "sign_geometry_temporal_cut_investigation_index.md"
    index.write_text(
        "# Sign Geometry/Temporal Cut Investigation\n\n" + "\n".join(f"- `{path}`" for path in reports) + "\n",
        encoding="utf-8",
    )
    print(index)


if __name__ == "__main__":
    main()
