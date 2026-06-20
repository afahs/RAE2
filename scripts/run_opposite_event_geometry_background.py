#!/usr/bin/env python
"""Subtract geometry-matched opposite-event-type backgrounds for Earth/Sun.

For a disappearance event, use reappearance events with similar geometry as a
background control pool, and vice versa.  This avoids using same-event-type
neighbors, which would subtract the common occultation morphology we are trying
to recover.  Matching variables are intentionally physical and broad:

- limb position angle on the Moon;
- apparent target Galactic latitude;
- broad far-window raw slope.

The corrected value is:

    corrected_event_bin = event_far_line_residual_bin - median(matched_opposite_event_bins)

This is a diagnostic, not a production detection statistic.
"""

from __future__ import annotations

from pathlib import Path
import sys
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rylevonberg.table_io import read_table

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.util import ensure_dir, robust_sigma, software_versions, write_json  # noqa: E402


POINTS = ROOT / "outputs/lowfreq_regime_recovery_profiles_v1/event_far_line_residual_points.csv"
OUT = ROOT / "outputs/opposite_event_geometry_background_v1"
LOW_FREQS = [0.45, 0.70, 0.90, 1.31, 2.20]
EXPECTED_SIGN = {"disappearance": -1.0, "reappearance": 1.0}
PRE = (-180.0, -60.0)
POST = (60.0, 180.0)


def _read(path: Path) -> pd.DataFrame:
    return read_table(path, low_memory=False)


def _robust_sem(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size <= 1:
        return np.nan
    scale = robust_sigma(vals - np.nanmedian(vals))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.nanstd(vals, ddof=1))
    return float(scale / np.sqrt(vals.size)) if np.isfinite(scale) and scale > 0 else np.nan


def _event_features(points: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "source_name",
        "event_id",
        "event_type",
        "frequency_mhz",
        "limb_position_angle_deg",
        "target_galactic_b_deg",
        "far_raw_slope_per_s",
    ]
    feat = points[cols].drop_duplicates(["source_name", "event_id", "event_type", "frequency_mhz"]).copy()
    pa = np.deg2rad(pd.to_numeric(feat["limb_position_angle_deg"], errors="coerce").to_numpy(dtype=float))
    feat["limb_pa_sin"] = np.sin(pa)
    feat["limb_pa_cos"] = np.cos(pa)
    return feat


def _standardize(pool: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    out = pool.copy()
    stats = {}
    for col in cols:
        x = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float)
        med = float(np.nanmedian(x))
        scale = robust_sigma(x - med)
        if not np.isfinite(scale) or scale <= 0:
            scale = float(np.nanstd(x))
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        out[col + "_z"] = (x - med) / scale
        stats[col] = (med, scale)
    return out, stats


def _apply_standardize(row: pd.Series, stats: dict[str, tuple[float, float]], cols: list[str]) -> np.ndarray:
    vals = []
    for col in cols:
        med, scale = stats[col]
        val = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
        vals.append((float(val) - med) / scale if np.isfinite(val) else np.nan)
    return np.asarray(vals, dtype=float)


DEFAULT_FEATURES = ["limb_pa_sin", "limb_pa_cos", "target_galactic_b_deg", "far_raw_slope_per_s"]


def matched_background_correct(
    points: pd.DataFrame,
    k_neighbors: int = 30,
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feat = _event_features(points)
    profile_lookup = {}
    profile_cols = ["source_name", "frequency_mhz", "event_type"]
    for keys, grp in points.groupby(profile_cols, sort=True):
        profile_lookup[keys] = grp.pivot_table(
            index="event_id",
            columns="t_bin_sec",
            values="far_line_residual",
            aggfunc="median",
        )
    rows = []
    match_rows = []
    feature_cols = list(feature_cols or DEFAULT_FEATURES)
    for keys, group_feat in feat.groupby(["source_name", "frequency_mhz"], sort=True):
        source, freq = keys
        for _, target in group_feat.iterrows():
            opposite = "reappearance" if target["event_type"] == "disappearance" else "disappearance"
            pool = group_feat[group_feat["event_type"].eq(opposite)].copy()
            if len(pool) < 5:
                continue
            pool_z, stats = _standardize(pool, feature_cols)
            target_z = _apply_standardize(target, stats, feature_cols)
            mat = pool_z[[c + "_z" for c in feature_cols]].to_numpy(dtype=float)
            good = np.isfinite(mat).all(axis=1) & np.isfinite(target_z).all()
            if not good.any():
                continue
            dist = np.sqrt(np.nansum((mat[good] - target_z) ** 2, axis=1))
            candidates = pool_z.loc[good].copy()
            candidates["match_distance"] = dist
            candidates = candidates.sort_values("match_distance").head(k_neighbors)
            matched_ids = list(candidates["event_id"].astype(int))
            target_matrix = profile_lookup.get((source, freq, target["event_type"]))
            control_matrix = profile_lookup.get((source, freq, opposite))
            if target_matrix is None or control_matrix is None or target["event_id"] not in target_matrix.index:
                continue
            matched_ids = [event_id for event_id in matched_ids if event_id in control_matrix.index]
            if len(matched_ids) < 5:
                continue
            target_profile = target_matrix.loc[int(target["event_id"])]
            control_profile = control_matrix.loc[matched_ids].median(axis=0)
            common_bins = target_profile.index.intersection(control_profile.index)
            if common_bins.empty:
                continue
            corrected = target_profile.loc[common_bins] - control_profile.loc[common_bins]
            rows.append(
                pd.DataFrame(
                    {
                        "source_name": source,
                        "event_id": int(target["event_id"]),
                        "event_type": target["event_type"],
                        "frequency_mhz": freq,
                        "t_bin_sec": common_bins.to_numpy(dtype=float),
                        "far_line_residual": target_profile.loc[common_bins].to_numpy(dtype=float),
                        "matched_opposite_background": control_profile.loc[common_bins].to_numpy(dtype=float),
                        "geometry_background_corrected": corrected.to_numpy(dtype=float),
                        "n_matched_background_events": len(matched_ids),
                    }
                )
            )
            match_rows.append(
                {
                    "source_name": source,
                    "frequency_mhz": freq,
                    "event_id": int(target["event_id"]),
                    "event_type": target["event_type"],
                    "opposite_event_type": opposite,
                    "n_matched_background_events": len(matched_ids),
                    "median_match_distance": float(candidates["match_distance"].median()),
                    "max_match_distance": float(candidates["match_distance"].max()),
                    "feature_set": ",".join(feature_cols),
                }
            )
    corrected = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return corrected, pd.DataFrame(match_rows)


def summarize(points: pd.DataFrame, value_col: str, method: str) -> pd.DataFrame:
    rows = []
    keys = ["source_name", "method", "frequency_mhz", "event_type", "t_bin_sec"]
    for vals_key, grp in points.assign(method=method).groupby(keys, sort=True, dropna=False):
        vals = pd.to_numeric(grp[value_col], errors="coerce").dropna()
        if vals.empty:
            continue
        rows.append(
            {
                **dict(zip(keys, vals_key)),
                "median_profile": float(vals.median()),
                "profile_err": _robust_sem(vals),
                "n_events": int(grp["event_id"].nunique()),
                "n_points": int(len(vals)),
            }
        )
    return pd.DataFrame(rows)


def contrast(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["source_name", "method", "frequency_mhz", "event_type"]
    for vals_key, grp in summary.groupby(keys, sort=True, dropna=False):
        pre = grp[(grp["t_bin_sec"] >= PRE[0]) & (grp["t_bin_sec"] <= PRE[1])]["median_profile"]
        post = grp[(grp["t_bin_sec"] >= POST[0]) & (grp["t_bin_sec"] <= POST[1])]["median_profile"]
        if pre.empty or post.empty:
            continue
        event_type = str(vals_key[-1])
        delta = float(np.nanmedian(post) - np.nanmedian(pre))
        rows.append(
            {
                **dict(zip(keys, vals_key)),
                "n_events": int(np.nanmedian(grp["n_events"])),
                "post_minus_pre": delta,
                "source_like_contrast": float(EXPECTED_SIGN[event_type] * delta),
            }
        )
    return pd.DataFrame(rows)


def plot_profiles(summary: pd.DataFrame, out_dir: Path, source: str) -> Path:
    methods = ["far_line_residual", "geometry_opposite_event_subtracted"]
    colors = {"far_line_residual": "0.45", "geometry_opposite_event_subtracted": "#0072B2"}
    fig, axes = plt.subplots(len(LOW_FREQS), 2, figsize=(13, 10), sharex=True)
    for i, freq in enumerate(LOW_FREQS):
        for j, event_type in enumerate(["disappearance", "reappearance"]):
            ax = axes[i, j]
            for method in methods:
                sub = summary[
                    summary["source_name"].eq(source)
                    & np.isclose(summary["frequency_mhz"], freq)
                    & summary["event_type"].eq(event_type)
                    & summary["method"].eq(method)
                ].sort_values("t_bin_sec")
                if sub.empty:
                    continue
                ax.errorbar(
                    sub["t_bin_sec"],
                    sub["median_profile"],
                    yerr=sub["profile_err"],
                    marker="o",
                    markersize=2.1,
                    linewidth=1.15,
                    elinewidth=0.45,
                    capsize=0.9,
                    alpha=0.9,
                    color=colors[method],
                    ecolor=colors[method],
                    label=method.replace("_", " "),
                )
            ax.axvline(0, color="black", linestyle="--", linewidth=0.7)
            ax.axhline(0, color="0.65", linewidth=0.7)
            ax.set_title(f"{freq:.2f} MHz {event_type}", fontsize=9)
            if j == 0:
                ax.set_ylabel("profile")
            if i == len(LOW_FREQS) - 1:
                ax.set_xlabel("seconds from predicted event")
            if i == 0 and j == 1:
                ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"{source.title()}: opposite-event geometry-background subtraction", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = out_dir / f"{source}_opposite_event_geometry_background_grid.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report(out_dir: Path, contrasts: pd.DataFrame, match_table: pd.DataFrame, paths: list[Path]) -> None:
    lines = [
        "# Opposite-Event Geometry Background Subtraction",
        "",
        "## Method",
        "",
        "This subtracts a background profile built from opposite-event-type windows with similar limb position angle,",
        "apparent target Galactic latitude, and broad far-window slope. It is a diagnostic attempt to remove repeatable",
        "geometry/background morphology without using same-event-type windows as controls.",
        "",
        "## Match Counts",
        "",
        match_table.groupby(["source_name", "frequency_mhz", "event_type"])["n_matched_background_events"].describe().to_string()
        if not match_table.empty
        else "No matches.",
        "",
        "## Source-Like Contrasts",
        "",
        contrasts[
            ["source_name", "method", "frequency_mhz", "event_type", "n_events", "source_like_contrast"]
        ].sort_values(["source_name", "frequency_mhz", "event_type", "method"]).to_string(index=False),
        "",
        "## Interpretation",
        "",
        "If this works, corrected profiles should move toward the Earth 0.45 MHz morphology while retaining event counts.",
        "If it fails or flips signs, then opposite-event-type windows are not a valid background pool for this source/frequency.",
        "",
        "## Plots",
        "",
    ]
    lines.extend(f"- `{path}`" for path in paths)
    (out_dir / "opposite_event_geometry_background_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(OUT))
    parser.add_argument("--k-neighbors", type=int, default=30)
    parser.add_argument("--feature-set", default=",".join(DEFAULT_FEATURES))
    args = parser.parse_args()
    out_dir = ensure_dir(args.out_dir)
    feature_cols = [x.strip() for x in str(args.feature_set).split(",") if x.strip()]
    write_json(
        out_dir / "run_config.json",
        {
            "points": str(POINTS),
            "k_neighbors": int(args.k_neighbors),
            "feature_cols": feature_cols,
            "software_versions": software_versions(),
        },
    )
    points = _read(POINTS)
    points = points[points["source_name"].isin(["earth", "sun"]) & points["frequency_mhz"].isin(LOW_FREQS)].copy()
    corrected, matches = matched_background_correct(points, k_neighbors=args.k_neighbors, feature_cols=feature_cols)
    corrected.to_csv(out_dir / "geometry_opposite_event_corrected_points.csv", index=False)
    matches.to_csv(out_dir / "geometry_opposite_event_match_table.csv", index=False)

    base = points.rename(columns={"far_line_residual": "profile_value"}).copy()
    base_summary = summarize(base, "profile_value", "far_line_residual")
    corr_summary = summarize(corrected, "geometry_background_corrected", "geometry_opposite_event_subtracted")
    summary = pd.concat([base_summary, corr_summary], ignore_index=True, sort=False)
    contrasts = contrast(summary)
    summary.to_csv(out_dir / "geometry_background_profile_summary.csv", index=False)
    contrasts.to_csv(out_dir / "geometry_background_prepost_contrasts.csv", index=False)
    paths = [plot_profiles(summary, out_dir, source) for source in ["earth", "sun"]]
    write_report(out_dir, contrasts, matches, paths)
    print(out_dir / "opposite_event_geometry_background_report.md")


if __name__ == "__main__":
    main()
