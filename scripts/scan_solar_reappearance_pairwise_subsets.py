#!/usr/bin/env python
"""Scan moving-body subset morphology after geometry-background subtraction.

This is a visual/morphology diagnostic for hard moving-body cases.
It does not select on the near-event occultation sign.  Candidate subsets are
defined from independent geometry/background labels, then compared with the
Earth 0.45 MHz reference shape from the same corrected-point table.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rylevonberg.table_io import read_table
from astropy import units as u
from astropy.coordinates import FK4, SkyCoord
from astropy.time import Time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.util import ensure_dir, robust_sigma, software_versions, write_json  # noqa: E402


METRICS = ROOT / "outputs/moving_body_regime_physical_differences_v1/moving_body_event_regime_geometry_table.csv"
DEFAULT_POINTS = ROOT / "outputs/opposite_event_geometry_background_k10_v1/geometry_opposite_event_corrected_points.csv"
DEFAULT_OUT = ROOT / "outputs/solar_reappearance_pairwise_subset_scan_v1"

DEFAULT_FREQS = [0.70, 1.31]
MIN_EVENTS = 10
PRE = (-180.0, -60.0)
POST = (60.0, 180.0)
EXPECTED_SIGN = {"disappearance": -1.0, "reappearance": 1.0}


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


def _qcut_labels(values: pd.Series, labels: list[str]) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    out = pd.Series(index=values.index, dtype=object)
    finite = numeric[np.isfinite(numeric)]
    if finite.empty:
        return out
    try:
        out.loc[finite.index] = pd.qcut(finite, q=len(labels), labels=labels, duplicates="drop").astype(str)
    except ValueError:
        ranks = finite.rank(method="first")
        out.loc[finite.index] = pd.qcut(ranks, q=len(labels), labels=labels, duplicates="drop").astype(str)
    return out


def _metadata() -> pd.DataFrame:
    metrics = _read(METRICS)
    meta_cols = [
        "source_name",
        "event_id",
        "event_type",
        "frequency_band",
        "frequency_mhz",
        "far_raw_slope_per_s",
        "near_raw_slope_per_s",
        "pre_far_median",
        "pre_sigma",
        "limb_rate_deg_s",
        "gap_seconds",
        "limb_position_angle_deg",
        "target_ra_deg",
        "target_dec_deg",
        "month",
    ]
    meta = metrics[meta_cols].drop_duplicates(["source_name", "event_id", "event_type", "frequency_mhz"]).copy()
    finite_coord = np.isfinite(meta["target_ra_deg"]) & np.isfinite(meta["target_dec_deg"])
    meta["target_galactic_b_deg"] = np.nan
    if finite_coord.any():
        coord = SkyCoord(
            ra=meta.loc[finite_coord, "target_ra_deg"].to_numpy(dtype=float) * u.deg,
            dec=meta.loc[finite_coord, "target_dec_deg"].to_numpy(dtype=float) * u.deg,
            frame=FK4(equinox=Time("B1950")),
        )
        meta.loc[finite_coord, "target_galactic_b_deg"] = np.asarray(coord.galactic.b.deg, dtype=float)
    return meta


def _with_metadata(points: pd.DataFrame) -> pd.DataFrame:
    meta = _metadata()
    return points.merge(
        meta,
        on=["source_name", "event_id", "event_type", "frequency_mhz"],
        how="left",
        suffixes=("", "_meta"),
    )


def _base_labels(events: pd.DataFrame) -> dict[int, set[str]]:
    work = events.drop_duplicates("event_id").copy()
    labels: dict[int, set[str]] = {int(eid): set() for eid in work["event_id"]}

    pa = pd.to_numeric(work["limb_position_angle_deg"], errors="coerce") % 360.0
    work["pa_sector"] = pd.cut(
        pa,
        bins=[0, 90, 180, 270, 360],
        labels=["pa_0_90", "pa_90_180", "pa_180_270", "pa_270_360"],
        include_lowest=True,
        right=False,
    ).astype(str)
    work["far_slope_bin"] = _qcut_labels(work["far_raw_slope_per_s"], ["low_far_slope", "mid_far_slope", "high_far_slope"])
    work["near_slope_bin"] = _qcut_labels(work["near_raw_slope_per_s"], ["low_near_slope", "mid_near_slope", "high_near_slope"])
    work["noise_bin"] = _qcut_labels(work["pre_sigma"], ["low_noise", "mid_noise", "high_noise"])
    work["background_bin"] = _qcut_labels(work["pre_far_median"], ["low_background", "mid_background", "high_background"])
    work["limb_rate_bin"] = _qcut_labels(work["limb_rate_deg_s"], ["low_limb_rate", "mid_limb_rate", "high_limb_rate"])
    work["gap_bin"] = _qcut_labels(work["gap_seconds"], ["low_gap", "mid_gap", "high_gap"])
    work["abs_gal_b_bin"] = _qcut_labels(work["target_galactic_b_deg"].abs(), ["low_abs_gal_b", "mid_abs_gal_b", "high_abs_gal_b"])

    for _, row in work.iterrows():
        eid = int(row["event_id"])
        for col in [
            "month",
            "pa_sector",
            "far_slope_bin",
            "near_slope_bin",
            "noise_bin",
            "background_bin",
            "limb_rate_bin",
            "gap_bin",
            "abs_gal_b_bin",
        ]:
            val = row.get(col)
            if isinstance(val, str) and val and val != "nan":
                labels[eid].add(val)
        if np.isfinite(row.get("far_raw_slope_per_s", np.nan)) and row["far_raw_slope_per_s"] >= 0:
            labels[eid].add("favorable_far_slope")
        if np.isfinite(row.get("target_galactic_b_deg", np.nan)):
            if abs(row["target_galactic_b_deg"]) >= 30:
                labels[eid].add("high_abs_gal_b")
            else:
                labels[eid].add("low_abs_gal_b")
    return labels


def _expand_subset_labels(events: pd.DataFrame) -> pd.DataFrame:
    labels = _base_labels(events)
    rows = []
    for eid, base in labels.items():
        for label in sorted(base):
            rows.append({"event_id": eid, "subset": label, "complexity": 1})
        for a, b in combinations(sorted(base), 2):
            rows.append({"event_id": eid, "subset": f"{a}&{b}", "complexity": 2})
    return pd.DataFrame(rows)


def _summarize(points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, grp in points.groupby(["frequency_mhz", "subset", "t_bin_sec"], sort=True):
        vals = pd.to_numeric(grp["geometry_background_corrected"], errors="coerce").dropna()
        if vals.empty:
            continue
        rows.append(
            {
                "frequency_mhz": keys[0],
                "subset": keys[1],
                "t_bin_sec": keys[2],
                "median_profile": float(vals.median()),
                "profile_err": _robust_sem(vals),
                "n_events": int(grp["event_id"].nunique()),
                "n_points": int(len(vals)),
            }
        )
    return pd.DataFrame(rows)


def _parse_freqs(text: str) -> list[float]:
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


def _reference_shape(points: pd.DataFrame, source: str, event_type: str, freq: float) -> pd.DataFrame:
    ref = points[
        points["source_name"].eq(source)
        & points["event_type"].eq(event_type)
        & np.isclose(points["frequency_mhz"], freq)
    ].copy()
    sign = EXPECTED_SIGN[event_type]
    rows = []
    for t, grp in ref.groupby("t_bin_sec", sort=True):
        vals = pd.to_numeric(grp["geometry_background_corrected"], errors="coerce").dropna()
        if not vals.empty:
            rows.append({"t_bin_sec": t, "reference_profile": float(sign * vals.median())})
    return pd.DataFrame(rows)


def _metrics(summary: pd.DataFrame, ref: pd.DataFrame, event_type: str) -> pd.DataFrame:
    sign = EXPECTED_SIGN[event_type]
    rows = []
    for keys, grp in summary.groupby(["frequency_mhz", "subset"], sort=True):
        n_events = int(np.nanmedian(grp["n_events"]))
        if n_events < MIN_EVENTS:
            continue
        pre = grp[(grp["t_bin_sec"] >= PRE[0]) & (grp["t_bin_sec"] <= PRE[1])]["median_profile"]
        post = grp[(grp["t_bin_sec"] >= POST[0]) & (grp["t_bin_sec"] <= POST[1])]["median_profile"]
        if pre.empty or post.empty:
            continue
        central_delta = float(np.nanmedian(post) - np.nanmedian(pre))
        source_like_delta = float(sign * central_delta)
        oriented = grp[["t_bin_sec", "median_profile"]].copy()
        oriented["oriented_profile"] = sign * oriented["median_profile"]
        merged = oriented[["t_bin_sec", "oriented_profile"]].merge(ref, on="t_bin_sec", how="inner")
        corr = np.nan
        if len(merged) >= 6:
            x = merged["oriented_profile"].to_numpy(dtype=float)
            y = merged["reference_profile"].to_numpy(dtype=float)
            good = np.isfinite(x) & np.isfinite(y)
            if np.count_nonzero(good) >= 6 and np.nanstd(x[good]) > 0 and np.nanstd(y[good]) > 0:
                corr = float(np.corrcoef(x[good], y[good])[0, 1])
        complexity = 2 if "&" in str(keys[1]) else 1
        if source_like_delta > 0 and np.isfinite(corr) and corr >= 0.35:
            status = "reference_like_shape"
            status_rank = 0
        elif source_like_delta > 0:
            status = "positive_contrast_only"
            status_rank = 1
        else:
            status = "not_step_like"
            status_rank = 2
        rows.append(
            {
                "frequency_mhz": keys[0],
                "subset": keys[1],
                "n_events": n_events,
                "central_delta": central_delta,
                "source_like_delta": source_like_delta,
                "reference_corr": corr,
                "complexity": complexity,
                "status": status,
                "status_rank": status_rank,
            }
        )
    return pd.DataFrame(rows)


def _plot(summary: pd.DataFrame, metrics: pd.DataFrame, out: Path, source: str, event_type: str, freqs: list[float]) -> list[Path]:
    paths = []
    sign = EXPECTED_SIGN[event_type]
    for freq in freqs:
        best = metrics[np.isclose(metrics["frequency_mhz"], freq)].sort_values(
            ["status_rank", "complexity", "reference_corr", "source_like_delta"],
            ascending=[True, True, False, False],
        ).head(8)
        if best.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, 5.8))
        for _, row in best.iterrows():
            sub = summary[np.isclose(summary["frequency_mhz"], freq) & summary["subset"].eq(row["subset"])].sort_values("t_bin_sec")
            ax.errorbar(
                sub["t_bin_sec"],
                sign * sub["median_profile"],
                yerr=sub["profile_err"],
                marker="o",
                markersize=2.4,
                linewidth=1.15,
                elinewidth=0.55,
                capsize=1.1,
                label=f"{row['subset']} n={int(row['n_events'])} corr={row['reference_corr']:.2f}",
            )
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.axhline(0, color="0.65", linewidth=0.7)
        ax.set_title(f"{source} {event_type} {freq:.2f} MHz: oriented geometry-corrected subsets")
        ax.set_xlabel(f"seconds from predicted {event_type}")
        ax.set_ylabel("source-like oriented corrected normalized power")
        ax.legend(frameon=False, fontsize=7)
        fig.tight_layout()
        path = out / f"{source}_{event_type}_{freq:.2f}_pairwise_subset_scan.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def _write_report(
    best: pd.DataFrame,
    metrics: pd.DataFrame,
    paths: list[Path],
    out: Path,
    points_path: Path,
    source: str,
    event_type: str,
    reference_source: str,
    reference_event_type: str,
    reference_freq: float,
) -> None:
    like = metrics[metrics["status"].eq("reference_like_shape")].copy()
    lines = [
        "# Moving-Body Pairwise Subset Scan",
        "",
        f"Target: `{source}` `{event_type}`",
        f"Reference: `{reference_source}` `{reference_event_type}` `{reference_freq:.2f} MHz`, oriented so source-like is positive",
        f"Corrected points: `{points_path}`",
        "",
        "Subsets are based on month, limb PA sector, far/near slope bins, noise/background bins, limb rate, gap, and Galactic latitude.",
        "They do not select on near-event occultation sign.",
        "",
        "## Best Subsets",
        "",
        best.to_string(index=False) if not best.empty else "No subsets passed the minimum event count.",
        "",
        "## Reference-Like Rows",
        "",
        like[["frequency_mhz", "subset", "n_events", "central_delta", "source_like_delta", "reference_corr", "complexity", "status"]].to_string(index=False)
        if not like.empty
        else "No reference-like rows.",
        "",
        "## Plots",
        "",
    ]
    lines.extend(f"- `{path.name}`" for path in paths)
    (out / "solar_reappearance_pairwise_subset_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", default=str(DEFAULT_POINTS))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--target-source", default="sun")
    parser.add_argument("--event-type", default="reappearance", choices=["disappearance", "reappearance"])
    parser.add_argument("--freqs", default=",".join(str(f) for f in DEFAULT_FREQS))
    parser.add_argument("--reference-source", default="earth")
    parser.add_argument("--reference-event-type", default=None, choices=["disappearance", "reappearance"])
    parser.add_argument("--reference-freq", type=float, default=0.45)
    args = parser.parse_args()

    points_path = Path(args.points)
    out = ensure_dir(Path(args.out_dir))
    target_source = str(args.target_source).lower()
    event_type = str(args.event_type)
    freqs = _parse_freqs(args.freqs)
    reference_source = str(args.reference_source).lower()
    reference_event_type = str(args.reference_event_type or event_type)
    write_json(
        out / "run_config.json",
        {
            "points": str(points_path),
            "metrics": str(METRICS),
            "target_source": target_source,
            "event_type": event_type,
            "frequencies_mhz": freqs,
            "reference_source": reference_source,
            "reference_event_type": reference_event_type,
            "reference_freq": args.reference_freq,
            "min_events": MIN_EVENTS,
            "software_versions": software_versions(),
        },
    )

    points = _with_metadata(_read(points_path))
    ref = _reference_shape(points, reference_source, reference_event_type, args.reference_freq)
    target = points[
        points["source_name"].eq(target_source)
        & points["event_type"].eq(event_type)
        & points["frequency_mhz"].isin(freqs)
    ].copy()
    labels = _expand_subset_labels(target)
    expanded = target.merge(labels, on="event_id", how="inner")
    summary = _summarize(expanded)
    metric_table = _metrics(summary, ref, event_type)
    best = metric_table.sort_values(
        ["frequency_mhz", "status_rank", "complexity", "reference_corr", "source_like_delta"],
        ascending=[True, True, True, False, False],
    ).groupby("frequency_mhz", as_index=False).head(15)

    summary.to_csv(out / "solar_reappearance_pairwise_subset_profiles.csv", index=False)
    metric_table.to_csv(out / "solar_reappearance_pairwise_subset_metrics.csv", index=False)
    best.to_csv(out / "solar_reappearance_pairwise_best_subsets.csv", index=False)
    paths = _plot(summary, metric_table, out, target_source, event_type, freqs)
    _write_report(
        best,
        metric_table,
        paths,
        out,
        points_path,
        target_source,
        event_type,
        reference_source,
        reference_event_type,
        args.reference_freq,
    )
    print(out / "solar_reappearance_pairwise_subset_report.md")


if __name__ == "__main__":
    main()
