#!/usr/bin/env python
"""Subtract feature-matched empirical background windows from event profiles.

This builds on the time-shift control profiles.  Instead of subtracting the
median of every available shifted control for the same event, each true event
profile is matched to shifted non-event windows with similar outer-window
morphology.  The default features deliberately avoid the central before/after
contrast used to identify an occultation:

- pre-event outer-window slope;
- post-event outer-window slope;
- pre-event outer-window scatter;
- post-event outer-window scatter.

The corrected event profile is:

    corrected(t) = true_event(t) - median(nearest_matched_shift_controls(t))

This is an empirical diagnostic/background-removal test, not a theoretical
forward model.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.util import ensure_dir, robust_sigma, software_versions, write_json  # noqa: E402

from scripts.build_all_frequency_occultation_profile_grids import ANT_COLOR, ANT_LABEL  # noqa: E402


EXPECTED_SIGN = {"disappearance": -1.0, "reappearance": 1.0}
ID_COLS = ["source_name", "event_id", "event_type", "frequency_band", "frequency_mhz", "antenna"]
CONTROL_ID_COLS = ID_COLS + ["time_shift_s"]
DEFAULT_FEATURES = ["pre_slope_per_min", "post_slope_per_min", "pre_scatter", "post_scatter"]


def _read(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return read_table(path, low_memory=False)


def _robust_sem(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size <= 1:
        return np.nan
    scale = robust_sigma(vals - np.nanmedian(vals))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.nanstd(vals, ddof=1))
    return float(scale / np.sqrt(vals.size)) if np.isfinite(scale) and scale > 0 else np.nan


def _slope_per_min(t_sec: np.ndarray, z: np.ndarray) -> float:
    ok = np.isfinite(t_sec) & np.isfinite(z)
    if np.count_nonzero(ok) < 3:
        return np.nan
    x = t_sec[ok] / 60.0
    y = z[ok]
    if np.nanstd(x) <= 0:
        return np.nan
    try:
        return float(np.polyfit(x, y, deg=1)[0])
    except Exception:
        return np.nan


def _scatter(z: np.ndarray) -> float:
    vals = np.asarray(z, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return np.nan
    scale = robust_sigma(vals - np.nanmedian(vals))
    if np.isfinite(scale) and scale > 0:
        return float(scale)
    return float(np.nanstd(vals, ddof=1))


def build_profile_features(points: pd.DataFrame, id_cols: list[str], feature_start_s: float, feature_end_s: float) -> pd.DataFrame:
    rows = []
    for keys, grp in points.groupby(id_cols, sort=True, dropna=False):
        t = pd.to_numeric(grp["t_bin_sec"], errors="coerce").to_numpy(dtype=float)
        z = pd.to_numeric(grp["z_power"], errors="coerce").to_numpy(dtype=float)
        pre = (t >= -float(feature_end_s)) & (t <= -float(feature_start_s))
        post = (t >= float(feature_start_s)) & (t <= float(feature_end_s))
        row = dict(zip(id_cols, keys))
        row.update(
            {
                "pre_median": float(np.nanmedian(z[pre])) if np.count_nonzero(pre) else np.nan,
                "post_median": float(np.nanmedian(z[post])) if np.count_nonzero(post) else np.nan,
                "outer_delta": float(np.nanmedian(z[post]) - np.nanmedian(z[pre]))
                if np.count_nonzero(pre) and np.count_nonzero(post)
                else np.nan,
                "pre_slope_per_min": _slope_per_min(t[pre], z[pre]),
                "post_slope_per_min": _slope_per_min(t[post], z[post]),
                "pre_scatter": _scatter(z[pre]),
                "post_scatter": _scatter(z[post]),
                "n_pre_outer_bins": int(np.count_nonzero(pre & np.isfinite(z))),
                "n_post_outer_bins": int(np.count_nonzero(post & np.isfinite(z))),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _standardize(pool: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    out = pool.copy()
    stats: dict[str, tuple[float, float]] = {}
    for col in feature_cols:
        vals = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float)
        med = float(np.nanmedian(vals))
        scale = robust_sigma(vals - med)
        if not np.isfinite(scale) or scale <= 0:
            scale = float(np.nanstd(vals))
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        out[col + "_z"] = (vals - med) / scale
        stats[col] = (med, scale)
    return out, stats


def _standardize_row(row: pd.Series, stats: dict[str, tuple[float, float]], feature_cols: list[str]) -> np.ndarray:
    vals = []
    for col in feature_cols:
        med, scale = stats[col]
        val = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
        vals.append((float(val) - med) / scale if np.isfinite(val) else np.nan)
    return np.asarray(vals, dtype=float)


def _pivot_profiles(points: pd.DataFrame, index_cols: list[str], value_col: str = "z_power") -> dict[tuple, pd.DataFrame]:
    out = {}
    group_cols = ["source_name", "event_type", "frequency_band", "frequency_mhz", "antenna"]
    for keys, grp in points.groupby(group_cols, sort=True, dropna=False):
        out[keys] = grp.pivot_table(index=index_cols, columns="t_bin_sec", values=value_col, aggfunc="median")
    return out


def matched_background_subtract(
    true_points: pd.DataFrame,
    control_points: pd.DataFrame,
    k_neighbors: int,
    min_controls: int,
    feature_cols: list[str],
    feature_start_s: float,
    feature_end_s: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    true_features = build_profile_features(true_points, ID_COLS, feature_start_s, feature_end_s)
    control_features = build_profile_features(control_points, CONTROL_ID_COLS, feature_start_s, feature_end_s)
    true_profiles = _pivot_profiles(true_points, ["event_id"])
    control_profiles = _pivot_profiles(control_points, ["event_id", "time_shift_s"])

    corrected_rows = []
    match_rows = []
    feature_suffix = [col + "_z" for col in feature_cols]
    group_cols = ["source_name", "event_type", "frequency_band", "frequency_mhz", "antenna"]

    for group_key, targets in true_features.groupby(group_cols, sort=True, dropna=False):
        pool = control_features
        for col, val in zip(group_cols, group_key):
            pool = pool[pool[col].eq(val)]
        if len(pool) < min_controls:
            continue
        pool_z, stats = _standardize(pool, feature_cols)
        profile_true = true_profiles.get(group_key)
        profile_controls = control_profiles.get(group_key)
        if profile_true is None or profile_controls is None:
            continue
        matrix = pool_z[feature_suffix].to_numpy(dtype=float)
        matrix_good = np.isfinite(matrix).all(axis=1)
        if not matrix_good.any():
            continue
        pool_good = pool_z.loc[matrix_good].copy()
        matrix = pool_good[feature_suffix].to_numpy(dtype=float)

        for _, target in targets.iterrows():
            target_id = int(target["event_id"])
            if target_id not in profile_true.index:
                continue
            target_z = _standardize_row(target, stats, feature_cols)
            if not np.isfinite(target_z).all():
                continue
            distances = np.sqrt(np.sum((matrix - target_z) ** 2, axis=1))
            candidates = pool_good.copy()
            candidates["match_distance"] = distances
            candidates = candidates.sort_values("match_distance").head(int(k_neighbors))
            control_ids = [(int(row.event_id), float(row.time_shift_s)) for row in candidates.itertuples()]
            control_ids = [control_id for control_id in control_ids if control_id in profile_controls.index]
            if len(control_ids) < int(min_controls):
                continue

            true_profile = profile_true.loc[target_id]
            background = profile_controls.loc[control_ids].median(axis=0)
            common_bins = true_profile.index.intersection(background.index)
            if common_bins.empty:
                continue
            corrected = true_profile.loc[common_bins] - background.loc[common_bins]
            corrected_rows.append(
                pd.DataFrame(
                    {
                        "source_name": group_key[0],
                        "event_id": target_id,
                        "event_type": group_key[1],
                        "frequency_band": int(group_key[2]),
                        "frequency_mhz": float(group_key[3]),
                        "antenna": group_key[4],
                        "t_bin_sec": common_bins.to_numpy(dtype=float),
                        "z_power": true_profile.loc[common_bins].to_numpy(dtype=float),
                        "matched_background_z_power": background.loc[common_bins].to_numpy(dtype=float),
                        "matched_background_corrected_z_power": corrected.to_numpy(dtype=float),
                        "n_matched_controls": len(control_ids),
                        "median_match_distance": float(candidates["match_distance"].median()),
                        "max_match_distance": float(candidates["match_distance"].max()),
                    }
                )
            )
            match_rows.append(
                {
                    "source_name": group_key[0],
                    "event_id": target_id,
                    "event_type": group_key[1],
                    "frequency_band": int(group_key[2]),
                    "frequency_mhz": float(group_key[3]),
                    "antenna": group_key[4],
                    "n_matched_controls": len(control_ids),
                    "median_match_distance": float(candidates["match_distance"].median()),
                    "max_match_distance": float(candidates["match_distance"].max()),
                    "matched_control_event_ids": ";".join(str(x[0]) for x in control_ids),
                    "matched_control_time_shifts_s": ";".join(f"{x[1]:.0f}" for x in control_ids),
                }
            )
    corrected_points = pd.concat(corrected_rows, ignore_index=True) if corrected_rows else pd.DataFrame()
    return corrected_points, pd.DataFrame(match_rows), pd.concat(
        [
            true_features.assign(profile_role="true"),
            control_features.assign(profile_role="shift_control"),
        ],
        ignore_index=True,
        sort=False,
    )


def summarize(points: pd.DataFrame, value_col: str, method: str) -> pd.DataFrame:
    if points.empty:
        return pd.DataFrame()
    rows = []
    keys = ["source_name", "event_type", "frequency_band", "frequency_mhz", "antenna", "t_bin_sec"]
    for vals_key, grp in points.groupby(keys, sort=True, dropna=False):
        raw = pd.to_numeric(grp[value_col], errors="coerce")
        good = raw.notna() & np.isfinite(raw)
        vals = raw.loc[good]
        if vals.empty:
            continue
        rows.append(
            {
                **dict(zip(keys, vals_key)),
                "method": method,
                "median_z_power": float(vals.median()),
                "median_z_power_err": _robust_sem(vals),
                "n_events": int(grp.loc[good, "event_id"].nunique()),
                "n_points": int(len(vals)),
                "median_n_matched_controls": float(np.nanmedian(grp.loc[good, "n_matched_controls"]))
                if "n_matched_controls" in grp
                else np.nan,
                "median_match_distance": float(np.nanmedian(grp.loc[good, "median_match_distance"]))
                if "median_match_distance" in grp
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def prepost_contrast(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pre = (-180.0, -60.0)
    post = (60.0, 180.0)
    keys = ["method", "source_name", "frequency_band", "frequency_mhz", "antenna", "event_type"]
    for vals_key, grp in summary.groupby(keys, sort=True, dropna=False):
        before = grp[(grp["t_bin_sec"] >= pre[0]) & (grp["t_bin_sec"] <= pre[1])]["median_z_power"]
        after = grp[(grp["t_bin_sec"] >= post[0]) & (grp["t_bin_sec"] <= post[1])]["median_z_power"]
        if before.empty or after.empty:
            continue
        event_type = str(vals_key[-1])
        delta = float(np.nanmedian(after) - np.nanmedian(before))
        rows.append(
            {
                **dict(zip(keys, vals_key)),
                "n_events": int(np.nanmedian(grp["n_events"])) if "n_events" in grp else np.nan,
                "post_minus_pre": delta,
                "source_like_contrast": float(EXPECTED_SIGN[event_type] * delta),
            }
        )
    return pd.DataFrame(rows)


def contrast_comparison(uncorrected: pd.DataFrame, corrected: pd.DataFrame) -> pd.DataFrame:
    keys = ["source_name", "frequency_band", "frequency_mhz", "antenna", "event_type"]
    left = uncorrected.rename(
        columns={
            "n_events": "n_events_uncorrected",
            "post_minus_pre": "post_minus_pre_uncorrected",
            "source_like_contrast": "source_like_contrast_uncorrected",
        }
    )
    right = corrected.rename(
        columns={
            "n_events": "n_events_corrected",
            "post_minus_pre": "post_minus_pre_corrected",
            "source_like_contrast": "source_like_contrast_corrected",
        }
    )
    merged = left[
        keys + ["n_events_uncorrected", "post_minus_pre_uncorrected", "source_like_contrast_uncorrected"]
    ].merge(
        right[keys + ["n_events_corrected", "post_minus_pre_corrected", "source_like_contrast_corrected"]],
        on=keys,
        how="outer",
    )
    merged["source_like_contrast_change"] = (
        merged["source_like_contrast_corrected"] - merged["source_like_contrast_uncorrected"]
    )
    return merged.sort_values(keys).reset_index(drop=True)


def plot_corrected_grid(summary: pd.DataFrame, source: str, out_dir: Path, window_s: float) -> Path:
    freqs = sorted(summary["frequency_mhz"].dropna().unique())
    fig, axes = plt.subplots(len(freqs), 2, figsize=(12, max(10, 1.35 * len(freqs))), sharex=True)
    if len(freqs) == 1:
        axes = np.asarray([axes])
    for i, freq in enumerate(freqs):
        for j, event_type in enumerate(["disappearance", "reappearance"]):
            ax = axes[i, j]
            sub = summary[np.isclose(summary["frequency_mhz"], freq) & summary["event_type"].eq(event_type)]
            for antenna, grp in sub.groupby("antenna", sort=True):
                grp = grp.sort_values("t_bin_sec")
                ax.errorbar(
                    grp["t_bin_sec"],
                    grp["median_z_power"],
                    yerr=grp["median_z_power_err"],
                    marker="o",
                    markersize=2.3,
                    linewidth=1.1,
                    elinewidth=0.55,
                    capsize=1.0,
                    alpha=0.9,
                    color=ANT_COLOR.get(str(antenna)),
                    ecolor=ANT_COLOR.get(str(antenna)),
                    label=ANT_LABEL.get(str(antenna), str(antenna)),
                )
            ax.axvline(0, color="black", linestyle="--", linewidth=0.75)
            ax.axhline(0, color="0.65", linewidth=0.7)
            ax.set_title(f"{freq:.2f} MHz {event_type}", fontsize=9)
            if j == 0:
                ax.set_ylabel("true - matched background")
            if i == len(freqs) - 1:
                ax.set_xlabel("seconds from predicted event")
            if i == 0 and j == 1:
                ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"{source.title()}: matched empirical background subtraction, +/-{window_s:.0f} s", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    path = out_dir / f"{source}_matched_background_subtracted_all_frequency_profile_grid_{int(window_s)}s.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_lower_v_overlay(uncorrected: pd.DataFrame, corrected: pd.DataFrame, source: str, out_dir: Path, window_s: float) -> Path:
    unc = uncorrected[uncorrected["antenna"].eq("rv2_coarse")].copy()
    cor = corrected[corrected["antenna"].eq("rv2_coarse")].copy()
    freqs = sorted(set(unc["frequency_mhz"].dropna()).union(cor["frequency_mhz"].dropna()))
    fig, axes = plt.subplots(len(freqs), 2, figsize=(12, max(10, 1.35 * len(freqs))), sharex=True)
    if len(freqs) == 1:
        axes = np.asarray([axes])
    for i, freq in enumerate(freqs):
        for j, event_type in enumerate(["disappearance", "reappearance"]):
            ax = axes[i, j]
            for data, label, color, ls in [
                (unc, "uncorrected", "0.35", "--"),
                (cor, "matched-background subtracted", "#d95f02", "-"),
            ]:
                sub = data[np.isclose(data["frequency_mhz"], freq) & data["event_type"].eq(event_type)].sort_values(
                    "t_bin_sec"
                )
                if sub.empty:
                    continue
                ax.errorbar(
                    sub["t_bin_sec"],
                    sub["median_z_power"],
                    yerr=sub["median_z_power_err"],
                    marker="o",
                    markersize=2.3,
                    linewidth=1.1,
                    elinewidth=0.55,
                    capsize=1.0,
                    alpha=0.9,
                    color=color,
                    ecolor=color,
                    linestyle=ls,
                    label=label if i == 0 and j == 1 else None,
                )
            ax.axvline(0, color="black", linestyle="--", linewidth=0.75)
            ax.axhline(0, color="0.65", linewidth=0.7)
            ax.set_title(f"{freq:.2f} MHz {event_type}", fontsize=9)
            if j == 0:
                ax.set_ylabel("normalized power")
            if i == len(freqs) - 1:
                ax.set_xlabel("seconds from predicted event")
            if i == 0 and j == 1:
                ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"{source.title()} lower V: uncorrected vs matched empirical background, +/-{window_s:.0f} s", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    path = out_dir / f"{source}_lower_v_uncorrected_vs_matched_background_grid_{int(window_s)}s.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report(
    out_dir: Path,
    source: str,
    feature_cols: list[str],
    comparison: pd.DataFrame,
    matches: pd.DataFrame,
    paths: list[Path],
) -> None:
    low = comparison[
        comparison["frequency_mhz"].isin([0.45, 0.70, 0.90, 1.31, 2.20])
        & comparison["antenna"].eq("rv2_coarse")
    ].copy()
    match_summary = (
        matches.groupby(["frequency_mhz", "antenna", "event_type"])["n_matched_controls"].describe().reset_index()
        if not matches.empty
        else pd.DataFrame()
    )
    lines = [
        "# Matched Empirical Background Subtraction",
        "",
        "## Method",
        "",
        "This uses shifted non-event windows as an empirical background pool, but it does not subtract all shifted controls.",
        "For each true event, it selects the nearest shifted-control profiles using only outer-window morphology.",
        "The default features are side slopes and side scatter, so central before/after occultation contrast is not used as a matching feature.",
        "",
        f"Source: `{source}`",
        f"Feature columns: `{', '.join(feature_cols)}`",
        "",
        "## Low-Frequency Lower-V Contrast Comparison",
        "",
    ]
    if low.empty:
        lines.append("No low-frequency lower-V rows.")
    else:
        lines.append(
            low[
                [
                    "frequency_mhz",
                    "event_type",
                    "n_events_uncorrected",
                    "n_events_corrected",
                    "source_like_contrast_uncorrected",
                    "source_like_contrast_corrected",
                    "source_like_contrast_change",
                ]
            ].to_string(index=False)
        )
    lines.extend(["", "## Match Count Summary", ""])
    lines.append(match_summary.to_string(index=False) if not match_summary.empty else "No matched controls.")
    lines.extend(
        [
            "",
            "## Interpretation Guide",
            "",
            "- If the low-frequency Earth reversal is mostly removable background morphology, the corrected lower-V profiles should move toward the 0.45 MHz Earth shape.",
            "- If they remain anti-template, this feature-matched empirical subtraction is not capturing the contaminating term or the term is coupled to the event geometry itself.",
            "- If the 0.45 MHz and high-frequency Earth profiles vanish, the subtraction is too aggressive and is removing real occultation structure.",
            "",
            "## Plots",
            "",
        ]
    )
    lines.extend(f"- `{path}`" for path in paths)
    (out_dir / "matched_empirical_background_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="earth")
    parser.add_argument("--input-dir", default=str(ROOT / "outputs/shift_control_background_profiles_earth_20min_v1"))
    parser.add_argument("--out-dir", default=str(ROOT / "outputs/matched_empirical_background_profiles_earth_20min_v1"))
    parser.add_argument("--window-s", type=float, default=1200.0)
    parser.add_argument("--feature-start-s", type=float, default=720.0)
    parser.add_argument("--feature-end-s", type=float, default=1200.0)
    parser.add_argument("--k-neighbors", type=int, default=20)
    parser.add_argument("--min-controls", type=int, default=8)
    parser.add_argument("--feature-cols", default=",".join(DEFAULT_FEATURES))
    args = parser.parse_args()

    source = str(args.source).strip().lower()
    input_dir = Path(args.input_dir)
    out_dir = ensure_dir(args.out_dir)
    feature_cols = [x.strip() for x in str(args.feature_cols).split(",") if x.strip()]
    true_points = _read(input_dir / f"{source}_true_profile_points.csv")
    control_points = _read(input_dir / f"{source}_shift_control_profile_points.csv")
    if true_points.empty or control_points.empty:
        raise SystemExit(f"Missing true/control points in {input_dir}")

    write_json(
        out_dir / "run_config.json",
        {
            "source": source,
            "input_dir": str(input_dir),
            "window_s": float(args.window_s),
            "feature_start_s": float(args.feature_start_s),
            "feature_end_s": float(args.feature_end_s),
            "k_neighbors": int(args.k_neighbors),
            "min_controls": int(args.min_controls),
            "feature_cols": feature_cols,
            "software_versions": software_versions(),
        },
    )

    corrected, matches, features = matched_background_subtract(
        true_points=true_points,
        control_points=control_points,
        k_neighbors=args.k_neighbors,
        min_controls=args.min_controls,
        feature_cols=feature_cols,
        feature_start_s=args.feature_start_s,
        feature_end_s=args.feature_end_s,
    )
    corrected.to_csv(out_dir / f"{source}_matched_background_corrected_points.csv", index=False)
    matches.to_csv(out_dir / f"{source}_matched_background_match_table.csv", index=False)
    features.to_csv(out_dir / f"{source}_matched_background_profile_features.csv", index=False)

    uncorrected_summary = summarize(true_points, "z_power", "uncorrected")
    corrected_summary = summarize(corrected, "matched_background_corrected_z_power", "matched_background_subtracted")
    uncorrected_contrast = prepost_contrast(uncorrected_summary)
    corrected_contrast = prepost_contrast(corrected_summary)
    comparison = contrast_comparison(uncorrected_contrast, corrected_contrast)

    uncorrected_summary.to_csv(out_dir / f"{source}_uncorrected_summary.csv", index=False)
    corrected_summary.to_csv(out_dir / f"{source}_matched_background_corrected_summary.csv", index=False)
    comparison.to_csv(out_dir / f"{source}_matched_background_contrast_comparison.csv", index=False)

    paths = []
    if not corrected_summary.empty:
        paths.append(plot_corrected_grid(corrected_summary, source, out_dir, args.window_s))
        paths.append(plot_lower_v_overlay(uncorrected_summary, corrected_summary, source, out_dir, args.window_s))
    write_report(out_dir, source, feature_cols, comparison, matches, paths)
    print(out_dir / "matched_empirical_background_report.md")


if __name__ == "__main__":
    main()
