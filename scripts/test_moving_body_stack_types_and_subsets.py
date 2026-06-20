#!/usr/bin/env python
"""Compare Earth/Sun stack normalizations and event subsets."""

from __future__ import annotations

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

from rylevonberg.util import datetime_ns, ensure_dir, robust_sigma  # noqa: E402


CLEAN = ROOT / "outputs/control_survey_earth_sun_postnov1974_v2/01_ingest/cleaned_timeseries.csv"
EVENTS = ROOT / "outputs/planetary_confirmation_survey_science_baseline_v2/events/all_planet_predicted_events.csv"
OUT = ROOT / "outputs/moving_body_stack_type_subset_tests_v1"

SOURCES = ["earth", "sun"]
ANTENNA = "rv2_coarse"
WINDOW_S = 900.0
BIN_S = 60.0
PRE_FAR = (-900.0, -300.0)
POST_FAR = (300.0, 900.0)
PRE_NEAR = (-180.0, -60.0)
POST_NEAR = (60.0, 180.0)
EXPECTED_SIGN = {"disappearance": -1.0, "reappearance": 1.0}

STACK_TYPES = {
    "raw_power_mean": "raw power, mean across event-bin values",
    "no_normalization_median": "raw power, median across event-bin values",
    "raw_fractional": "P / median(pre-far) - 1 for each event",
    "pre_event_anchored": "(P - median(pre-far)) / robust_sigma(pre-far) for each event",
    "current_side_normalized": "(P - median(|t|>=15 s)) / robust_sigma(|t|>=15 s), matching all-frequency grid logic",
}


def _read_clean() -> pd.DataFrame:
    cols = ["time", "frequency_band", "frequency_mhz", "antenna", "power", "is_valid"]
    clean = read_table(CLEAN, usecols=cols, parse_dates=["time"], low_memory=False)
    clean = clean[clean["antenna"].eq(ANTENNA)].copy()
    return clean


def _read_events() -> pd.DataFrame:
    events = read_table(EVENTS, parse_dates=["predicted_event_time"], low_memory=False)
    events = events[
        events["source_name"].astype(str).isin(SOURCES)
        & events["antenna"].astype(str).eq(ANTENNA)
    ].copy()
    events["month"] = events["predicted_event_time"].dt.to_period("M").astype(str)
    events["limb_rate_deg_s"] = (
        events["post_limb_angle_deg"].astype(float) - events["pre_limb_angle_deg"].astype(float)
    ).abs() / events["gap_seconds"].astype(float).clip(lower=1e-9)
    return events


def _groups(clean: pd.DataFrame) -> dict[int, tuple[pd.DataFrame, np.ndarray]]:
    out = {}
    for band, grp in clean.groupby("frequency_band", sort=True):
        g = grp.sort_values("time").reset_index(drop=True)
        out[int(band)] = (g, datetime_ns(g["time"]))
    return out


def _window(group: pd.DataFrame, group_ns: np.ndarray, event_time: pd.Timestamp) -> tuple[np.ndarray, np.ndarray] | None:
    event_ns = pd.Timestamp(event_time).value
    half_ns = int(WINDOW_S * 1e9)
    lo = int(np.searchsorted(group_ns, event_ns - half_ns, side="left"))
    hi = int(np.searchsorted(group_ns, event_ns + half_ns, side="right"))
    if hi <= lo:
        return None
    local = group.iloc[lo:hi]
    t = (datetime_ns(local["time"]) - event_ns).astype(float) / 1e9
    y = pd.to_numeric(local["power"], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(y) & (np.abs(t) <= WINDOW_S)
    if "is_valid" in local:
        keep &= local["is_valid"].to_numpy(dtype=bool)
    if np.count_nonzero(keep) < 8:
        return None
    order = np.argsort(t[keep])
    return t[keep][order], y[keep][order]


def _scale(vals: np.ndarray) -> float:
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return np.nan
    s = robust_sigma(vals - np.nanmedian(vals))
    if not np.isfinite(s) or s <= 0:
        s = float(np.nanstd(vals))
    return s if np.isfinite(s) and s > 0 else np.nan


def _safe_quantile_labels(values: pd.Series, labels: list[str]) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    finite = x[np.isfinite(x)]
    out = pd.Series("unknown", index=values.index, dtype=object)
    if finite.nunique() < len(labels):
        out.loc[np.isfinite(x)] = "all"
        return out
    try:
        out.loc[np.isfinite(x)] = pd.qcut(x[np.isfinite(x)], q=len(labels), labels=labels, duplicates="drop").astype(str)
    except ValueError:
        out.loc[np.isfinite(x)] = "all"
    return out


def build_event_tables(clean: pd.DataFrame, events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = _groups(clean)
    point_rows = []
    event_rows = []
    bins = np.arange(-WINDOW_S, WINDOW_S + BIN_S, BIN_S)
    for ev in events.itertuples(index=False):
        band = int(ev.frequency_band)
        payload = groups.get(band)
        if payload is None:
            continue
        local = _window(payload[0], payload[1], pd.Timestamp(ev.predicted_event_time))
        if local is None:
            continue
        t, y = local
        pre_far = (t >= PRE_FAR[0]) & (t <= PRE_FAR[1])
        post_far = (t >= POST_FAR[0]) & (t <= POST_FAR[1])
        pre_near = (t >= PRE_NEAR[0]) & (t <= PRE_NEAR[1])
        post_near = (t >= POST_NEAR[0]) & (t <= POST_NEAR[1])
        side = np.abs(t) >= 15.0
        if np.count_nonzero(pre_far) < 2 or np.count_nonzero(pre_near) < 1 or np.count_nonzero(post_near) < 1:
            continue
        pre_far_median = float(np.nanmedian(y[pre_far]))
        post_far_median = float(np.nanmedian(y[post_far])) if np.count_nonzero(post_far) else np.nan
        pre_sigma = _scale(y[pre_far])
        side_median = float(np.nanmedian(y[side])) if np.count_nonzero(side) >= 6 else np.nan
        side_sigma = _scale(y[side]) if np.count_nonzero(side) >= 6 else np.nan
        raw_pre = float(np.nanmedian(y[pre_near]))
        raw_post = float(np.nanmedian(y[post_near]))
        sign = EXPECTED_SIGN[str(ev.event_type)]
        if np.isfinite(pre_far_median) and pre_far_median != 0:
            frac = y / pre_far_median - 1.0
            frac_pre = raw_pre / pre_far_median - 1.0
            frac_post = raw_post / pre_far_median - 1.0
        else:
            frac = np.full_like(y, np.nan)
            frac_pre = frac_post = np.nan
        anchored = (y - pre_far_median) / pre_sigma if np.isfinite(pre_sigma) and pre_sigma > 0 else np.full_like(y, np.nan)
        side_norm = (y - side_median) / side_sigma if np.isfinite(side_sigma) and side_sigma > 0 else np.full_like(y, np.nan)
        t_bin = 0.5 * (bins[np.clip(np.digitize(t, bins) - 1, 0, len(bins) - 2)] + bins[np.clip(np.digitize(t, bins) - 1, 0, len(bins) - 2) + 1])

        near_slope = (raw_post - raw_pre) / max(POST_NEAR[0] - PRE_NEAR[1], 1.0)
        far_slope = (post_far_median - pre_far_median) / (0.5 * (POST_FAR[0] + POST_FAR[1]) - 0.5 * (PRE_FAR[0] + PRE_FAR[1])) if np.isfinite(post_far_median) else np.nan
        event_rows.append(
            {
                "source_name": ev.source_name,
                "event_id": ev.event_id,
                "event_type": ev.event_type,
                "frequency_band": band,
                "frequency_mhz": float(ev.frequency_mhz),
                "month": ev.month,
                "gap_seconds": float(ev.gap_seconds),
                "limb_rate_deg_s": float(ev.limb_rate_deg_s),
                "pre_far_median": pre_far_median,
                "post_far_median": post_far_median,
                "pre_sigma": pre_sigma,
                "side_sigma": side_sigma,
                "raw_pre_near": raw_pre,
                "raw_post_near": raw_post,
                "raw_post_minus_pre": raw_post - raw_pre,
                "source_like_raw_contrast": sign * (raw_post - raw_pre),
                "fractional_post_minus_pre": frac_post - frac_pre if np.isfinite(frac_post) and np.isfinite(frac_pre) else np.nan,
                "source_like_fractional_contrast": sign * (frac_post - frac_pre) if np.isfinite(frac_post) and np.isfinite(frac_pre) else np.nan,
                "anchored_post_minus_pre": float(np.nanmedian(anchored[post_near]) - np.nanmedian(anchored[pre_near])) if np.isfinite(anchored).any() else np.nan,
                "source_like_anchored_contrast": sign * (float(np.nanmedian(anchored[post_near]) - np.nanmedian(anchored[pre_near])) if np.isfinite(anchored).any() else np.nan),
                "side_norm_post_minus_pre": float(np.nanmedian(side_norm[post_near]) - np.nanmedian(side_norm[pre_near])) if np.isfinite(side_norm).any() else np.nan,
                "source_like_side_norm_contrast": sign * (float(np.nanmedian(side_norm[post_near]) - np.nanmedian(side_norm[pre_near])) if np.isfinite(side_norm).any() else np.nan),
                "near_raw_slope_per_s": near_slope,
                "far_raw_slope_per_s": far_slope,
            }
        )
        for trel, tb, yy, ff, aa, ss in zip(t, t_bin, y, frac, anchored, side_norm):
            if not np.isfinite(tb) or abs(trel) > WINDOW_S:
                continue
            point_rows.append(
                {
                    "source_name": ev.source_name,
                    "event_id": ev.event_id,
                    "event_type": ev.event_type,
                    "frequency_band": band,
                    "frequency_mhz": float(ev.frequency_mhz),
                    "month": ev.month,
                    "t_rel_sec": float(trel),
                    "t_bin_sec": float(tb),
                    "raw_power": float(yy),
                    "raw_fractional": float(ff) if np.isfinite(ff) else np.nan,
                    "pre_event_anchored": float(aa) if np.isfinite(aa) else np.nan,
                    "current_side_normalized": float(ss) if np.isfinite(ss) else np.nan,
                }
            )
    events_out = pd.DataFrame(event_rows)
    if not events_out.empty:
        events_out["slope_subset"] = "flat"
        s = events_out["near_raw_slope_per_s"]
        q1, q2 = np.nanquantile(s, [1 / 3, 2 / 3])
        events_out.loc[s <= q1, "slope_subset"] = "falling_local_power"
        events_out.loc[s >= q2, "slope_subset"] = "rising_local_power"
        events_out["limb_rate_subset"] = events_out.groupby(["source_name", "frequency_band"], group_keys=False)["limb_rate_deg_s"].apply(lambda x: _safe_quantile_labels(x, ["slow_limb_rate", "mid_limb_rate", "fast_limb_rate"]))
        events_out["gap_subset"] = events_out.groupby(["source_name", "frequency_band"], group_keys=False)["gap_seconds"].apply(lambda x: _safe_quantile_labels(x, ["short_gap", "mid_gap", "long_gap"]))
        events_out["background_subset"] = events_out.groupby(["source_name", "frequency_band"], group_keys=False)["pre_far_median"].apply(lambda x: _safe_quantile_labels(x, ["low_background", "mid_background", "high_background"]))
        point_meta = events_out[["source_name", "event_id", "frequency_band", "slope_subset", "limb_rate_subset", "gap_subset", "background_subset"]]
        points_out = pd.DataFrame(point_rows).merge(point_meta, on=["source_name", "event_id", "frequency_band"], how="left") if point_rows else pd.DataFrame()
    else:
        points_out = pd.DataFrame(point_rows)
    return points_out, events_out


def _subset_assignments(events: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "all_events": pd.Series("all", index=events.index),
        "local_slope": events["slope_subset"],
        "limb_rate": events["limb_rate_subset"],
        "bracketing_gap": events["gap_subset"],
        "pre_event_background": events["background_subset"],
        "month": events["month"],
    }


def stack_summary(points: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    if points.empty or events.empty:
        return pd.DataFrame()
    rows = []
    point_cols = {
        "raw_power_mean": ("raw_power", "mean"),
        "no_normalization_median": ("raw_power", "median"),
        "raw_fractional": ("raw_fractional", "median"),
        "pre_event_anchored": ("pre_event_anchored", "median"),
        "current_side_normalized": ("current_side_normalized", "median"),
    }
    for subset_family, labels in _subset_assignments(events).items():
        label_df = events[["source_name", "event_id", "frequency_band"]].copy()
        label_df["subset_label"] = labels.to_numpy()
        pts = points.merge(label_df, on=["source_name", "event_id", "frequency_band"], how="inner")
        for keys, grp in pts.groupby(["source_name", "frequency_band", "frequency_mhz", "event_type", "subset_label"], sort=True):
            source, band, mhz, event_type, subset_label = keys
            sign = EXPECTED_SIGN[str(event_type)]
            pre_mask = (grp["t_bin_sec"] >= PRE_NEAR[0]) & (grp["t_bin_sec"] <= PRE_NEAR[1])
            post_mask = (grp["t_bin_sec"] >= POST_NEAR[0]) & (grp["t_bin_sec"] <= POST_NEAR[1])
            if not pre_mask.any() or not post_mask.any():
                continue
            for stack_type, (col, agg_name) in point_cols.items():
                pre_vals = pd.to_numeric(grp.loc[pre_mask, col], errors="coerce").dropna()
                post_vals = pd.to_numeric(grp.loc[post_mask, col], errors="coerce").dropna()
                if pre_vals.empty or post_vals.empty:
                    continue
                pre = float(pre_vals.mean()) if agg_name == "mean" else float(pre_vals.median())
                post = float(post_vals.mean()) if agg_name == "mean" else float(post_vals.median())
                rows.append(
                    {
                        "source_name": source,
                        "frequency_band": int(band),
                        "frequency_mhz": float(mhz),
                        "event_type": event_type,
                        "subset_family": subset_family,
                        "subset_label": str(subset_label),
                        "stack_type": stack_type,
                        "stack_description": STACK_TYPES[stack_type],
                        "n_events": int(grp["event_id"].nunique()),
                        "pre_value": pre,
                        "post_value": post,
                        "post_minus_pre": post - pre,
                        "source_like_contrast": sign * (post - pre),
                        "source_like": bool(sign * (post - pre) > 0),
                    }
                )
    return pd.DataFrame(rows)


def contrast_summary(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    contrast_cols = {
        "per_event_raw_contrast": "source_like_raw_contrast",
        "per_event_fractional_contrast": "source_like_fractional_contrast",
        "per_event_pre_event_anchored_contrast": "source_like_anchored_contrast",
        "per_event_current_side_normalized_contrast": "source_like_side_norm_contrast",
    }
    for subset_family, labels in _subset_assignments(events).items():
        work = events.copy()
        work["subset_label"] = labels.to_numpy()
        for keys, grp in work.groupby(["source_name", "frequency_band", "frequency_mhz", "event_type", "subset_label"], sort=True):
            source, band, mhz, event_type, subset_label = keys
            for stack_type, col in contrast_cols.items():
                vals = pd.to_numeric(grp[col], errors="coerce").dropna()
                if vals.empty:
                    continue
                rows.append(
                    {
                        "source_name": source,
                        "frequency_band": int(band),
                        "frequency_mhz": float(mhz),
                        "event_type": event_type,
                        "subset_family": subset_family,
                        "subset_label": str(subset_label),
                        "stack_type": stack_type,
                        "stack_description": "event-level post/pre contrast signed so positive means expected occultation direction",
                        "n_events": int(len(vals)),
                        "median_source_like_contrast": float(vals.median()),
                        "mean_source_like_contrast": float(vals.mean()),
                        "source_like_fraction": float((vals > 0).mean()),
                    }
                )
    return pd.DataFrame(rows)


def plot_stack_type_heatmap(summary: pd.DataFrame, out_dir: Path) -> list[Path]:
    paths = []
    base = summary[(summary["subset_family"].eq("all_events")) & summary["subset_label"].eq("all")].copy()
    if base.empty:
        return paths
    for source in SOURCES:
        sub = base[base["source_name"].eq(source)].copy()
        for event_type in ["disappearance", "reappearance"]:
            pivot = sub[sub["event_type"].eq(event_type)].pivot_table(index="stack_type", columns="frequency_mhz", values="source_like_contrast", aggfunc="median")
            if pivot.empty:
                continue
            fig, ax = plt.subplots(figsize=(10, 4.6))
            vals = pivot.to_numpy(dtype=float)
            vmax = np.nanmax(np.abs(vals)) if np.isfinite(vals).any() else 1.0
            im = ax.imshow(vals, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_xticks(np.arange(len(pivot.columns)))
            ax.set_xticklabels([f"{c:.2f}" for c in pivot.columns], rotation=45)
            ax.set_yticks(np.arange(len(pivot.index)))
            ax.set_yticklabels(pivot.index)
            ax.set_xlabel("frequency (MHz)")
            ax.set_title(f"{source} {event_type}: source-like contrast by stack type")
            fig.colorbar(im, ax=ax, label="positive = expected occultation sign")
            fig.tight_layout()
            path = out_dir / f"{source}_{event_type}_stack_type_contrast_heatmap.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            paths.append(path)
    return paths


def plot_subset_bars(summary: pd.DataFrame, out_dir: Path) -> list[Path]:
    paths = []
    focus = summary[
        summary["frequency_mhz"].isin([0.90, 1.31, 2.20, 4.70, 9.18])
        & summary["stack_type"].eq("raw_fractional")
        & summary["subset_family"].isin(["local_slope", "limb_rate", "pre_event_background"])
    ].copy()
    for source in SOURCES:
        sub_source = focus[focus["source_name"].eq(source)]
        if sub_source.empty:
            continue
        for family in ["local_slope", "limb_rate", "pre_event_background"]:
            sub = sub_source[sub_source["subset_family"].eq(family)].copy()
            if sub.empty:
                continue
            g = sub.groupby(["frequency_mhz", "subset_label"], as_index=False)["source_like_contrast"].median()
            labels = sorted(g["subset_label"].unique())
            freqs = sorted(g["frequency_mhz"].unique())
            x = np.arange(len(freqs))
            width = 0.8 / max(len(labels), 1)
            fig, ax = plt.subplots(figsize=(10, 4.5))
            for i, label in enumerate(labels):
                vals = []
                for freq in freqs:
                    match = g[(g["frequency_mhz"].eq(freq)) & g["subset_label"].eq(label)]
                    vals.append(float(match["source_like_contrast"].iloc[0]) if not match.empty else np.nan)
                ax.bar(x + (i - (len(labels) - 1) / 2) * width, vals, width=width, label=label)
            ax.axhline(0, color="black", lw=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{f:.2f}" for f in freqs])
            ax.set_xlabel("frequency (MHz)")
            ax.set_ylabel("source-like contrast")
            ax.set_title(f"{source}: raw fractional stack by {family}")
            ax.legend(frameon=False, fontsize=8)
            fig.tight_layout()
            path = out_dir / f"{source}_{family}_raw_fractional_subset_bars.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            paths.append(path)
    return paths


def _table(df: pd.DataFrame, cols: list[str], max_rows: int = 40) -> str:
    if df.empty:
        return "(empty)"
    return df[cols].head(max_rows).to_string(index=False)


def write_report(summary: pd.DataFrame, contrasts: pd.DataFrame, event_metrics: pd.DataFrame, paths: list[Path]) -> Path:
    all_low = summary[
        summary["subset_family"].eq("all_events")
        & summary["subset_label"].eq("all")
        & summary["frequency_mhz"].isin([0.70, 0.90, 1.31, 2.20, 3.93, 4.70, 9.18])
    ].copy()
    all_low = all_low.sort_values(["source_name", "frequency_mhz", "event_type", "stack_type"])
    subset_focus = summary[
        summary["stack_type"].eq("raw_fractional")
        & summary["subset_family"].isin(["local_slope", "limb_rate", "pre_event_background"])
        & summary["frequency_mhz"].isin([0.90, 1.31, 2.20, 4.70, 9.18])
    ].copy()
    subset_focus = subset_focus.sort_values(["source_name", "frequency_mhz", "event_type", "subset_family", "subset_label"])
    contrast_focus = contrasts[
        contrasts["subset_family"].eq("all_events")
        & contrasts["frequency_mhz"].isin([0.70, 0.90, 1.31, 2.20, 3.93, 4.70, 9.18])
    ].sort_values(["source_name", "frequency_mhz", "event_type", "stack_type"])
    event_counts = (
        event_metrics.groupby(["source_name", "frequency_mhz", "event_type"], as_index=False)
        .agg(n_events=("event_id", "size"), median_limb_rate=("limb_rate_deg_s", "median"), median_gap_s=("gap_seconds", "median"))
        .sort_values(["source_name", "frequency_mhz", "event_type"])
    )
    lines = [
        "# Moving-Body Stack Type and Subset Tests",
        "",
        "Goal: test whether Earth/Sun low-frequency anti-template behavior is a product of stacking/normalization or a stable property of the selected moving-body event geometry.",
        "",
        "All tests here use lower V (`rv2_coarse`) and a +/-900 s window. Positive `source_like_contrast` means the expected point-source occultation direction: disappearance drops and reappearance rises.",
        "",
        "## Stack Types Tested",
        "",
    ]
    lines.extend([f"- `{name}`: {desc}" for name, desc in STACK_TYPES.items()])
    lines.append("- `per_event_*_contrast`: event-level post/pre contrast signed so positive means expected occultation direction.")
    lines.extend(
        [
            "",
            "## Event Counts",
            "",
            _table(event_counts, ["source_name", "frequency_mhz", "event_type", "n_events", "median_limb_rate", "median_gap_s"], 80),
            "",
            "## All-Event Stack-Type Comparison",
            "",
            _table(all_low, ["source_name", "frequency_mhz", "event_type", "stack_type", "n_events", "source_like_contrast", "source_like"], 120),
            "",
            "## Per-Event Sign-Preserving Contrast Summary",
            "",
            _table(contrast_focus, ["source_name", "frequency_mhz", "event_type", "stack_type", "n_events", "median_source_like_contrast", "source_like_fraction"], 120),
            "",
            "## Subset Tests: Raw Fractional Stack",
            "",
            _table(subset_focus, ["source_name", "frequency_mhz", "event_type", "subset_family", "subset_label", "n_events", "source_like_contrast", "source_like"], 160),
            "",
            "## Plots",
            "",
        ]
    )
    lines.extend([f"- `{p}`" for p in paths])
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- If a sign reversal appears only in `current_side_normalized`, it is mostly a normalization artifact.",
            "- If it appears in raw fractional, pre-event anchored, and event-level contrast summaries, it is a real repeated pre/post behavior in the selected windows.",
            "- If only one local-slope subset drives the sign, stacking is mixing subpopulations and should split by that diagnostic.",
            "- If all subsets retain the same sign, the behavior is more likely tied to the moving-body geometry itself.",
        ]
    )
    path = OUT / "moving_body_stack_type_subset_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    ensure_dir(OUT)
    clean = _read_clean()
    events = _read_events()
    points, event_metrics = build_event_tables(clean, events)
    points.to_csv(OUT / "moving_body_stack_points.csv", index=False)
    event_metrics.to_csv(OUT / "moving_body_event_metrics.csv", index=False)
    summary = stack_summary(points, event_metrics)
    contrasts = contrast_summary(event_metrics)
    summary.to_csv(OUT / "moving_body_stack_type_subset_summary.csv", index=False)
    contrasts.to_csv(OUT / "moving_body_per_event_contrast_summary.csv", index=False)
    paths = []
    paths.extend(plot_stack_type_heatmap(summary, OUT))
    paths.extend(plot_subset_bars(summary, OUT))
    report = write_report(summary, contrasts, event_metrics, paths)
    print(report)


if __name__ == "__main__":
    main()
