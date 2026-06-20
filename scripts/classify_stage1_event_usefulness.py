#!/usr/bin/env python
"""Stage-1 event usefulness classifier for moving-body occultation profiles.

This classifier is intentionally independent of whether the event's central
change is source-like or anti-template. It asks whether a local event window is
useful for occultation interpretation:

1. Is it sufficiently sampled near the event and in both side windows?
2. Is there a measurable central change?
3. Can slow side-window drift plausibly explain that central change?
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rylevonberg.table_io import read_table


ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "outputs/physics_based_event_morphology_v1/earth_event_morphology_metrics.csv"
POINTS = ROOT / "outputs/moving_body_stack_type_subset_tests_v1/moving_body_stack_points.csv"
STACK_DIAG = ROOT / "outputs/physics_based_event_morphology_v1/earth_stack_model_morphology_diagnostics.csv"
OUT = ROOT / "outputs/stage1_event_usefulness_v1"

MIN_NEAR_POINTS = 10
MIN_SIDE_POINTS = 10
MIN_ABS_CENTRAL_CONTRAST = 0.02
LOW_DRIFT_RATIO = 2.0
HIGH_DRIFT_RATIO = 6.0

CLASS_ORDER = ["usable_low_drift", "drift_competing", "low_information_or_weak"]
CLASS_COLOR = {
    "usable_low_drift": "#2ca02c",
    "drift_competing": "#ff7f0e",
    "low_information_or_weak": "#7f7f7f",
}


def _robust_sigma(vals: np.ndarray) -> float:
    vals = vals[np.isfinite(vals)]
    if vals.size <= 1:
        return np.nan
    center = np.nanmedian(vals)
    sigma = 1.4826 * np.nanmedian(np.abs(vals - center))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.nanstd(vals, ddof=1))
    return sigma if np.isfinite(sigma) and sigma > 0 else np.nan


def _robust_se(vals: pd.Series) -> float:
    arr = pd.to_numeric(vals, errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size <= 1:
        return np.nan
    sigma = _robust_sigma(arr)
    if not np.isfinite(sigma) or sigma <= 0:
        return np.nan
    return float(sigma / np.sqrt(arr.size))


def _stage1_classify(events: pd.DataFrame) -> pd.DataFrame:
    work = events[events["source_name"].eq("earth")].copy()
    contrast_abs = pd.to_numeric(work["source_like_fractional_contrast"], errors="coerce").abs()
    drift_ratio = pd.to_numeric(work["side_drift_to_central_contrast"], errors="coerce")
    sample_ok = (
        pd.to_numeric(work["n_near_points"], errors="coerce").ge(MIN_NEAR_POINTS)
        & pd.to_numeric(work["n_pre_side_points"], errors="coerce").ge(MIN_SIDE_POINTS)
        & pd.to_numeric(work["n_post_side_points"], errors="coerce").ge(MIN_SIDE_POINTS)
    )
    contrast_ok = contrast_abs.ge(MIN_ABS_CENTRAL_CONTRAST)
    drift_ok = np.isfinite(drift_ratio)

    work["abs_central_fractional_contrast"] = contrast_abs
    work["stage1_sample_ok"] = sample_ok
    work["stage1_contrast_ok"] = contrast_ok
    work["stage1_drift_ratio"] = drift_ratio

    cls = pd.Series("low_information_or_weak", index=work.index, dtype=object)
    interpretable = sample_ok & contrast_ok & drift_ok
    cls.loc[interpretable & drift_ratio.le(LOW_DRIFT_RATIO)] = "usable_low_drift"
    cls.loc[interpretable & drift_ratio.gt(LOW_DRIFT_RATIO) & drift_ratio.le(HIGH_DRIFT_RATIO)] = "drift_competing"
    cls.loc[interpretable & drift_ratio.gt(HIGH_DRIFT_RATIO)] = "low_information_or_weak"
    work["stage1_usefulness_class"] = cls

    reasons = pd.Series("", index=work.index, dtype=object)
    reasons.loc[~sample_ok] = "insufficient near/side samples"
    reasons.loc[sample_ok & ~contrast_ok] = "central contrast below threshold"
    reasons.loc[sample_ok & contrast_ok & ~drift_ok] = "side drift ratio unavailable"
    reasons.loc[interpretable & drift_ratio.le(LOW_DRIFT_RATIO)] = "side-window drift small compared with central change"
    reasons.loc[interpretable & drift_ratio.gt(LOW_DRIFT_RATIO) & drift_ratio.le(HIGH_DRIFT_RATIO)] = (
        "side-window drift competes with central change"
    )
    reasons.loc[interpretable & drift_ratio.gt(HIGH_DRIFT_RATIO)] = "side-window drift dominates central change"
    work["stage1_reason"] = reasons
    return work


def _event_bin_points(points: pd.DataFrame, classified: pd.DataFrame) -> pd.DataFrame:
    meta = classified[
        [
            "event_id",
            "event_type",
            "frequency_band",
            "frequency_mhz",
            "stage1_usefulness_class",
            "stage1_reason",
            "regime",
        ]
    ].copy()
    earth = points[points["source_name"].eq("earth")].merge(
        meta,
        on=["event_id", "event_type", "frequency_band", "frequency_mhz"],
        how="inner",
    )
    return (
        earth.groupby(
            [
                "event_id",
                "event_type",
                "frequency_band",
                "frequency_mhz",
                "stage1_usefulness_class",
                "stage1_reason",
                "regime",
                "t_bin_sec",
            ],
            as_index=False,
        )
        .agg(raw_fractional=("raw_fractional", "median"), n_samples=("raw_fractional", "size"))
    )


def _stack(event_bin: pd.DataFrame) -> pd.DataFrame:
    rows = []
    by = ["stage1_usefulness_class", "event_type", "frequency_band", "frequency_mhz", "t_bin_sec"]
    for keys, grp in event_bin.groupby(by, sort=True):
        vals = pd.to_numeric(grp["raw_fractional"], errors="coerce")
        rows.append(
            {
                **dict(zip(by, keys)),
                "median_raw_fractional": float(np.nanmedian(vals)),
                "err_raw_fractional": _robust_se(vals),
                "n_events": int(grp["event_id"].nunique()),
                "n_points": int(grp["n_samples"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _plot_class_grid(stacked: pd.DataFrame, class_name: str) -> Path:
    sub = stacked[stacked["stage1_usefulness_class"].eq(class_name)].copy()
    freqs = sorted(sub["frequency_mhz"].dropna().unique())
    fig, axes = plt.subplots(len(freqs), 2, figsize=(12, max(10, 1.35 * len(freqs))), sharex=True, sharey=False)
    if len(freqs) == 1:
        axes = np.asarray([axes])
    color = CLASS_COLOR[class_name]
    for i, freq in enumerate(freqs):
        for j, event_type in enumerate(["disappearance", "reappearance"]):
            ax = axes[i, j]
            g = sub[np.isclose(sub["frequency_mhz"], freq) & sub["event_type"].eq(event_type)].sort_values("t_bin_sec")
            if not g.empty:
                ax.errorbar(
                    g["t_bin_sec"] / 60.0,
                    g["median_raw_fractional"],
                    yerr=g["err_raw_fractional"],
                    marker="o",
                    markersize=2.8,
                    linewidth=1.2,
                    elinewidth=0.65,
                    capsize=1.3,
                    color=color,
                    ecolor=color,
                    alpha=0.92,
                )
                n_events = int(g["n_events"].max())
            else:
                n_events = 0
            ax.axvline(0.0, color="black", linestyle="--", linewidth=0.8)
            ax.axhline(0.0, color="0.6", linewidth=0.7)
            ax.grid(alpha=0.18)
            ax.set_title(f"{freq:.2f} MHz {event_type} (n={n_events})", fontsize=9)
            if j == 0:
                ax.set_ylabel("raw fractional power")
            if i == len(freqs) - 1:
                ax.set_xlabel("minutes from predicted event")
    fig.suptitle(
        f"Earth lower V all-frequency profile grid: stage-1 class = {class_name}\n"
        "Class uses sample support, central contrast magnitude, and side-window drift; not source-like sign.",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.972))
    path = OUT / f"earth_stage1_{class_name}_all_frequency_profile_grid_900s.png"
    fig.savefig(path, dpi=175)
    plt.close(fig)
    return path


def _plot_classifier_space(classified: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    for cls in CLASS_ORDER:
        g = classified[classified["stage1_usefulness_class"].eq(cls)]
        if g.empty:
            continue
        ax.scatter(
            g["abs_central_fractional_contrast"],
            g["stage1_drift_ratio"],
            s=14,
            alpha=0.45,
            c=CLASS_COLOR[cls],
            label=f"{cls} (n={len(g)})",
            linewidths=0,
        )
    ax.axvline(MIN_ABS_CENTRAL_CONTRAST, color="black", linestyle="--", linewidth=0.9)
    ax.axhline(LOW_DRIFT_RATIO, color="black", linestyle="--", linewidth=0.9)
    ax.axhline(HIGH_DRIFT_RATIO, color="black", linestyle=":", linewidth=0.9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("|central fractional contrast|")
    ax.set_ylabel("side-window drift / central contrast")
    ax.set_title("Stage-1 event usefulness classifier space")
    ax.grid(alpha=0.2, which="both")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = OUT / "stage1_event_usefulness_classifier_space.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _summaries(classified: pd.DataFrame) -> dict[str, pd.DataFrame]:
    counts = (
        classified.groupby(["frequency_mhz", "event_type", "stage1_usefulness_class"], as_index=False)
        .size()
        .pivot_table(
            index=["frequency_mhz", "event_type"],
            columns="stage1_usefulness_class",
            values="size",
            fill_value=0,
        )
        .reset_index()
    )
    for cls in CLASS_ORDER:
        if cls not in counts.columns:
            counts[cls] = 0
    counts["n_events"] = counts[CLASS_ORDER].sum(axis=1)
    for cls in CLASS_ORDER:
        counts[f"{cls}_fraction"] = counts[cls] / counts["n_events"].replace(0, np.nan)

    sign_mix = (
        classified.groupby(["stage1_usefulness_class", "regime"], as_index=False)
        .size()
        .pivot_table(index="stage1_usefulness_class", columns="regime", values="size", fill_value=0)
        .reset_index()
    )

    suspect = classified[
        classified["event_type"].eq("disappearance")
        & np.isclose(classified["frequency_mhz"], 0.90)
    ]
    suspect_counts = (
        suspect.groupby(["regime", "stage1_usefulness_class"], as_index=False)
        .size()
        .pivot_table(index="regime", columns="stage1_usefulness_class", values="size", fill_value=0)
        .reset_index()
    )

    model_assoc = pd.DataFrame()
    if STACK_DIAG.exists():
        diag = read_table(STACK_DIAG)
        class_frac = (
            classified.groupby(["frequency_mhz", "event_type", "stage1_usefulness_class"], as_index=False)
            .size()
            .pivot_table(
                index=["frequency_mhz", "event_type"],
                columns="stage1_usefulness_class",
                values="size",
                fill_value=0,
            )
            .reset_index()
        )
        for cls in CLASS_ORDER:
            if cls not in class_frac.columns:
                class_frac[cls] = 0
        class_frac["n_events_stage1"] = class_frac[CLASS_ORDER].sum(axis=1)
        for cls in CLASS_ORDER:
            class_frac[f"{cls}_fraction"] = class_frac[cls] / class_frac["n_events_stage1"].replace(0, np.nan)
        model_assoc = diag[diag["regime"].eq("source_like")].merge(
            class_frac,
            on=["frequency_mhz", "event_type"],
            how="left",
        )
    return {
        "stage1_counts_by_frequency_event_type": counts,
        "stage1_sign_mix": sign_mix,
        "earth_090mhz_disappearance_stage1_by_sign": suspect_counts,
        "stage1_vs_stack_morphology": model_assoc,
    }


def _write_report(paths: list[Path], tables: dict[str, pd.DataFrame]) -> Path:
    counts = tables["stage1_counts_by_frequency_event_type"]
    suspect = tables["earth_090mhz_disappearance_stage1_by_sign"]
    sign_mix = tables["stage1_sign_mix"]
    assoc = tables["stage1_vs_stack_morphology"]
    lines = [
        "# Stage-1 Event Usefulness Classifier",
        "",
        "This implements a first-stage event classifier for Earth lower-V windows that does not use",
        "whether the event is source-like or anti-template. It asks whether the event window is",
        "physically useful for a lunar-occultation interpretation.",
        "",
        "## Class Definitions",
        "",
        f"- `usable_low_drift`: at least {MIN_NEAR_POINTS} near-event samples, at least {MIN_SIDE_POINTS} samples in each side window,",
        f"  absolute central fractional contrast >= {MIN_ABS_CENTRAL_CONTRAST}, and side-window drift / central contrast <= {LOW_DRIFT_RATIO}.",
        f"- `drift_competing`: same sampling and contrast requirements, but {LOW_DRIFT_RATIO} < side-window drift / central contrast <= {HIGH_DRIFT_RATIO}.",
        f"- `low_information_or_weak`: weak central contrast, poor sampling, unavailable drift estimate, or side-window drift / central contrast > {HIGH_DRIFT_RATIO}.",
        "",
        "The side-window drift ratio is a physics-motivated usefulness check: for an occultation step,",
        "the signal should be localized near the predicted limb crossing and the pre/post side windows",
        "should be reasonably plateau-like. If a slow side-window drift over 10 minutes is larger than",
        "the central change, the event can mimic a step without requiring a lunar occultation.",
        "",
        "## Earth 0.90 MHz Disappearance Class Counts By Existing Sign Label",
        "",
        suspect.to_string(index=False),
        "",
        "## Overall Existing Sign Mix Within Stage-1 Classes",
        "",
        sign_mix.to_string(index=False),
        "",
        "## Stage-1 Counts By Frequency And Event Type",
        "",
        counts.to_string(index=False),
        "",
    ]
    if not assoc.empty:
        assoc_cols = [
            "event_type",
            "frequency_mhz",
            "morphology_class",
            "usable_low_drift_fraction",
            "drift_competing_fraction",
            "low_information_or_weak_fraction",
            "trend_to_step_abs",
            "line_plus_step_amplitude_source_oriented",
        ]
        lines.extend(
            [
                "## Association With Stack-Level Morphology",
                "",
                assoc[assoc_cols].sort_values(["event_type", "frequency_mhz"]).to_string(index=False),
                "",
            ]
        )
    lines.extend(["## Generated Plots", ""])
    lines.extend(f"- `{path}`" for path in paths)
    path = OUT / "stage1_event_usefulness_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    events = read_table(METRICS, low_memory=False)
    points = read_table(POINTS, low_memory=False)
    classified = _stage1_classify(events)
    event_bin = _event_bin_points(points, classified)
    stacked = _stack(event_bin)
    tables = _summaries(classified)

    classified.to_csv(OUT / "earth_stage1_event_usefulness_metrics.csv", index=False)
    event_bin.to_csv(OUT / "earth_stage1_event_bin_points.csv", index=False)
    stacked.to_csv(OUT / "earth_stage1_stacked_profiles.csv", index=False)
    for name, table in tables.items():
        table.to_csv(OUT / f"{name}.csv", index=False)

    paths = [_plot_classifier_space(classified)]
    for cls in CLASS_ORDER:
        paths.append(_plot_class_grid(stacked, cls))
    report = _write_report(paths, tables)

    print(report)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
