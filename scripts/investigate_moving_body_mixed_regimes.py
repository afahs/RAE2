#!/usr/bin/env python
"""Investigate whether Earth/Sun stacks mix different event regimes."""

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

from rylevonberg.util import ensure_dir  # noqa: E402


IN = ROOT / "outputs/moving_body_stack_type_subset_tests_v1"
OUT = ROOT / "outputs/moving_body_mixed_regime_investigation_v1"
EVENT_METRICS = IN / "moving_body_event_metrics.csv"
POINTS = IN / "moving_body_stack_points.csv"

FOCUS_FREQS = [0.90, 1.31, 2.20, 4.70, 9.18]
LOW_FREQS = [0.90, 1.31, 2.20]
STACK_VALUE = "raw_fractional"
REGIME_THRESHOLD = 0.02


def classify_regime(values: pd.Series, threshold: float = REGIME_THRESHOLD) -> pd.Series:
    vals = pd.to_numeric(values, errors="coerce")
    out = pd.Series("neutral", index=values.index, dtype=object)
    out.loc[vals > threshold] = "source_like"
    out.loc[vals < -threshold] = "anti_template"
    out.loc[~np.isfinite(vals)] = "invalid"
    return out


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    events = read_table(EVENT_METRICS, low_memory=False)
    points = read_table(POINTS, low_memory=False)
    events["regime"] = classify_regime(events["source_like_fractional_contrast"])
    # A less-thresholded sign label helps expose marginal structure.
    events["sign_class"] = np.where(
        pd.to_numeric(events["source_like_fractional_contrast"], errors="coerce") > 0,
        "positive_source_like",
        "negative_anti_template",
    )
    points = points.merge(
        events[
            [
                "source_name",
                "event_id",
                "frequency_band",
                "frequency_mhz",
                "event_type",
                "regime",
                "sign_class",
                "source_like_fractional_contrast",
                "source_like_raw_contrast",
                "near_raw_slope_per_s",
                "far_raw_slope_per_s",
                "pre_far_median",
                "pre_sigma",
                "limb_rate_deg_s",
                "gap_seconds",
                "month",
            ]
        ],
        on=["source_name", "event_id", "frequency_band", "frequency_mhz", "event_type", "month"],
        how="inner",
    )
    return events, points


def regime_counts(events: pd.DataFrame) -> pd.DataFrame:
    work = events[events["frequency_mhz"].isin(FOCUS_FREQS)].copy()
    rows = []
    for keys, grp in work.groupby(["source_name", "frequency_mhz", "event_type"], sort=True):
        source, freq, event_type = keys
        counts = grp["regime"].value_counts().to_dict()
        n = len(grp)
        rows.append(
            {
                "source_name": source,
                "frequency_mhz": freq,
                "event_type": event_type,
                "n_events": n,
                "source_like_n": int(counts.get("source_like", 0)),
                "neutral_n": int(counts.get("neutral", 0)),
                "anti_template_n": int(counts.get("anti_template", 0)),
                "source_like_fraction": float(counts.get("source_like", 0) / n) if n else np.nan,
                "anti_template_fraction": float(counts.get("anti_template", 0) / n) if n else np.nan,
                "median_source_like_fractional_contrast": float(np.nanmedian(grp["source_like_fractional_contrast"])),
            }
        )
    return pd.DataFrame(rows)


def regime_predictor_summary(events: pd.DataFrame) -> pd.DataFrame:
    work = events[events["frequency_mhz"].isin(FOCUS_FREQS)].copy()
    rows = []
    numeric = [
        "source_like_fractional_contrast",
        "source_like_raw_contrast",
        "near_raw_slope_per_s",
        "far_raw_slope_per_s",
        "pre_far_median",
        "pre_sigma",
        "limb_rate_deg_s",
        "gap_seconds",
    ]
    for keys, grp in work.groupby(["source_name", "frequency_mhz", "event_type", "regime"], sort=True):
        source, freq, event_type, regime = keys
        row = {
            "source_name": source,
            "frequency_mhz": freq,
            "event_type": event_type,
            "regime": regime,
            "n_events": int(len(grp)),
        }
        for col in numeric:
            vals = pd.to_numeric(grp[col], errors="coerce")
            row[f"median_{col}"] = float(np.nanmedian(vals)) if np.isfinite(vals).any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def categorical_mixture_tables(events: pd.DataFrame) -> dict[str, pd.DataFrame]:
    work = events[events["frequency_mhz"].isin(FOCUS_FREQS)].copy()
    tables: dict[str, pd.DataFrame] = {}
    for cat in ["month", "background_subset", "limb_rate_subset", "gap_subset", "slope_subset"]:
        rows = []
        for keys, grp in work.groupby(["source_name", "frequency_mhz", "event_type", cat], sort=True):
            source, freq, event_type, label = keys
            n = len(grp)
            rows.append(
                {
                    "source_name": source,
                    "frequency_mhz": freq,
                    "event_type": event_type,
                    "subset_family": cat,
                    "subset_label": label,
                    "n_events": n,
                    "source_like_fraction": float((grp["regime"].eq("source_like")).mean()),
                    "anti_template_fraction": float((grp["regime"].eq("anti_template")).mean()),
                    "median_source_like_fractional_contrast": float(np.nanmedian(grp["source_like_fractional_contrast"])),
                }
            )
        tables[cat] = pd.DataFrame(rows)
    return tables


def stack_by_regime(points: pd.DataFrame) -> pd.DataFrame:
    work = points[points["frequency_mhz"].isin(FOCUS_FREQS) & points["regime"].isin(["source_like", "anti_template", "neutral"])].copy()
    rows = []
    for keys, grp in work.groupby(["source_name", "frequency_mhz", "event_type", "regime", "t_bin_sec"], sort=True):
        source, freq, event_type, regime, tbin = keys
        vals = pd.to_numeric(grp[STACK_VALUE], errors="coerce").dropna()
        if vals.empty:
            continue
        rows.append(
            {
                "source_name": source,
                "frequency_mhz": freq,
                "event_type": event_type,
                "regime": regime,
                "t_bin_sec": float(tbin),
                "median_raw_fractional": float(vals.median()),
                "mean_raw_fractional": float(vals.mean()),
                "n_points": int(len(vals)),
                "n_events": int(grp["event_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def plot_regime_profiles(stacked: pd.DataFrame, out_dir: Path) -> list[Path]:
    paths = []
    colors = {"source_like": "#2ca02c", "anti_template": "#d62728", "neutral": "#7f7f7f"}
    for source in sorted(stacked["source_name"].unique()):
        for freq in FOCUS_FREQS:
            sub = stacked[(stacked["source_name"].eq(source)) & np.isclose(stacked["frequency_mhz"], freq)]
            if sub.empty:
                continue
            fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
            for ax, event_type in zip(axes, ["disappearance", "reappearance"]):
                et = sub[sub["event_type"].eq(event_type)]
                for regime, grp in et.groupby("regime", sort=True):
                    grp = grp.sort_values("t_bin_sec")
                    ax.plot(
                        grp["t_bin_sec"] / 60.0,
                        grp["median_raw_fractional"],
                        marker="o",
                        ms=2.5,
                        lw=1.4,
                        color=colors.get(regime, "black"),
                        label=f"{regime} ({int(grp['n_events'].max())} events)",
                    )
                ax.axvline(0, color="black", lw=0.8)
                ax.axhline(0, color="0.5", lw=0.7)
                ax.set_title(event_type)
                ax.set_xlabel("minutes from predicted event")
                ax.grid(alpha=0.2)
            axes[0].set_ylabel("median raw fractional power")
            axes[1].legend(frameon=False, fontsize=8)
            fig.suptitle(f"{source} {freq:.2f} MHz lower V: stacks split by event-level regime")
            fig.tight_layout()
            path = out_dir / f"{source}_{freq:.2f}mhz_regime_split_profiles.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            paths.append(path)
    return paths


def plot_regime_fractions(counts: pd.DataFrame, out_dir: Path) -> Path:
    focus = counts[counts["frequency_mhz"].isin(FOCUS_FREQS)].copy()
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey=True)
    axes = axes.ravel()
    panels = [
        ("earth", "disappearance"),
        ("earth", "reappearance"),
        ("sun", "disappearance"),
        ("sun", "reappearance"),
    ]
    for ax, (source, event_type) in zip(axes, panels):
        sub = focus[(focus["source_name"].eq(source)) & focus["event_type"].eq(event_type)].sort_values("frequency_mhz")
        x = np.arange(len(sub))
        ax.bar(x, sub["anti_template_fraction"], color="#d62728", label="anti-template")
        ax.bar(x, sub["source_like_fraction"], bottom=sub["anti_template_fraction"], color="#2ca02c", label="source-like")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:.2f}" for v in sub["frequency_mhz"]], rotation=45)
        ax.set_title(f"{source} {event_type}")
        ax.axhline(0.5, color="black", lw=0.8, ls="--")
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("event fraction")
    axes[2].set_ylabel("event fraction")
    axes[2].set_xlabel("frequency (MHz)")
    axes[3].set_xlabel("frequency (MHz)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Event-level regime fractions, lower V")
    fig.tight_layout()
    path = out_dir / "moving_body_event_regime_fractions.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_predictor_box(events: pd.DataFrame, out_dir: Path) -> list[Path]:
    paths = []
    focus = events[
        events["frequency_mhz"].isin(LOW_FREQS)
        & events["regime"].isin(["source_like", "anti_template"])
    ].copy()
    metrics = [
        ("far_raw_slope_per_s", "broad far-window slope"),
        ("pre_far_median", "pre-event background"),
        ("pre_sigma", "pre-event robust sigma"),
        ("limb_rate_deg_s", "limb rate"),
    ]
    for source in ["earth", "sun"]:
        sub_source = focus[focus["source_name"].eq(source)]
        if sub_source.empty:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        axes = axes.ravel()
        for ax, (col, title) in zip(axes, metrics):
            data = []
            labels = []
            for regime in ["source_like", "anti_template"]:
                vals = pd.to_numeric(sub_source[sub_source["regime"].eq(regime)][col], errors="coerce").dropna()
                if vals.empty:
                    continue
                data.append(vals.to_numpy())
                labels.append(regime)
            if data:
                ax.boxplot(data, tick_labels=labels, showfliers=False)
            ax.set_title(title)
            ax.grid(axis="y", alpha=0.25)
        fig.suptitle(f"{source}: low-frequency event-regime predictor distributions")
        fig.tight_layout()
        path = out_dir / f"{source}_lowfreq_regime_predictor_boxplots.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


def top_examples(events: pd.DataFrame) -> pd.DataFrame:
    work = events[events["frequency_mhz"].isin(FOCUS_FREQS)].copy()
    work["abs_contrast"] = work["source_like_fractional_contrast"].abs()
    cols = [
        "source_name",
        "frequency_mhz",
        "event_type",
        "event_id",
        "month",
        "regime",
        "source_like_fractional_contrast",
        "source_like_raw_contrast",
        "near_raw_slope_per_s",
        "far_raw_slope_per_s",
        "pre_far_median",
        "limb_rate_deg_s",
        "gap_seconds",
    ]
    rows = []
    for (source, freq, event_type, regime), grp in work.groupby(["source_name", "frequency_mhz", "event_type", "regime"], sort=True):
        if regime not in {"source_like", "anti_template"}:
            continue
        rows.append(grp.sort_values("abs_contrast", ascending=False).head(5)[cols])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=cols)


def _table(df: pd.DataFrame, cols: list[str], max_rows: int = 80) -> str:
    if df.empty:
        return "(empty)"
    return df[cols].head(max_rows).to_string(index=False)


def write_report(
    counts: pd.DataFrame,
    predictors: pd.DataFrame,
    cat_tables: dict[str, pd.DataFrame],
    examples: pd.DataFrame,
    plot_paths: list[Path],
) -> Path:
    low_counts = counts[counts["frequency_mhz"].isin(FOCUS_FREQS)].copy()
    low_counts = low_counts.sort_values(["source_name", "frequency_mhz", "event_type"])
    pred_focus = predictors[
        predictors["frequency_mhz"].isin(LOW_FREQS)
        & predictors["regime"].isin(["source_like", "anti_template"])
    ].sort_values(["source_name", "frequency_mhz", "event_type", "regime"])
    slope_table = cat_tables["slope_subset"]
    slope_focus = slope_table[slope_table["frequency_mhz"].isin(FOCUS_FREQS)].sort_values(
        ["source_name", "frequency_mhz", "event_type", "subset_label"]
    )
    month_table = cat_tables["month"]
    month_focus = (
        month_table[month_table["frequency_mhz"].isin(LOW_FREQS)]
        .sort_values(["source_name", "frequency_mhz", "event_type", "anti_template_fraction"], ascending=[True, True, True, False])
        .groupby(["source_name", "frequency_mhz", "event_type"])
        .head(3)
    )
    lines = [
        "# Moving-Body Mixed-Regime Investigation",
        "",
        "Purpose: test whether Earth/Sun stacks are averaging distinct event populations instead of one coherent occultation response.",
        "",
        f"Regime definition uses per-event raw-fractional source-like contrast with threshold `{REGIME_THRESHOLD}`:",
        "",
        "- `source_like`: expected point-source occultation direction;",
        "- `anti_template`: opposite direction;",
        "- `neutral`: small absolute contrast.",
        "",
        "## Event Regime Fractions",
        "",
        _table(
            low_counts,
            [
                "source_name",
                "frequency_mhz",
                "event_type",
                "n_events",
                "source_like_n",
                "neutral_n",
                "anti_template_n",
                "source_like_fraction",
                "anti_template_fraction",
                "median_source_like_fractional_contrast",
            ],
            120,
        ),
        "",
        "## Numeric Predictors By Regime",
        "",
        _table(
            pred_focus,
            [
                "source_name",
                "frequency_mhz",
                "event_type",
                "regime",
                "n_events",
                "median_source_like_fractional_contrast",
                "median_near_raw_slope_per_s",
                "median_far_raw_slope_per_s",
                "median_pre_far_median",
                "median_pre_sigma",
                "median_limb_rate_deg_s",
                "median_gap_seconds",
            ],
            120,
        ),
        "",
        "## Local-Slope Subset Table",
        "",
        _table(
            slope_focus,
            [
                "source_name",
                "frequency_mhz",
                "event_type",
                "subset_label",
                "n_events",
                "source_like_fraction",
                "anti_template_fraction",
                "median_source_like_fractional_contrast",
            ],
            160,
        ),
        "",
        "## Highest Anti-Template Months At Low Frequency",
        "",
        _table(
            month_focus,
            [
                "source_name",
                "frequency_mhz",
                "event_type",
                "subset_label",
                "n_events",
                "source_like_fraction",
                "anti_template_fraction",
                "median_source_like_fractional_contrast",
            ],
            120,
        ),
        "",
        "## Strong Individual Examples",
        "",
        _table(examples, list(examples.columns), 120),
        "",
        "## Plots",
        "",
    ]
    lines.extend(f"- `{path}`" for path in plot_paths)
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The all-event stack is not a single homogeneous population. At the same frequency and event type, there are source-like, neutral, and anti-template events.",
            "",
            "The strongest separator is the local power trend through the event window. This is expected because a moving bright body occultation is being measured on top of large low-frequency baseline/background motion. If that local trend has the same sign as the point-source occultation, the event appears source-like; if it has the opposite sign and larger magnitude, the event appears anti-template.",
            "",
            "The regime-split profile plots show whether source-like events have the ordinary disappearance/reappearance shape while anti-template events have the opposite shape. If both are present, a single all-event stack is physically misleading for Earth/Sun.",
        ]
    )
    path = OUT / "moving_body_mixed_regime_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    ensure_dir(OUT)
    events, points = load_tables()
    counts = regime_counts(events)
    predictors = regime_predictor_summary(events)
    cat_tables = categorical_mixture_tables(events)
    stacked = stack_by_regime(points)
    examples = top_examples(events)

    counts.to_csv(OUT / "event_regime_counts.csv", index=False)
    predictors.to_csv(OUT / "event_regime_predictor_summary.csv", index=False)
    stacked.to_csv(OUT / "regime_split_stacked_profiles.csv", index=False)
    examples.to_csv(OUT / "strong_individual_regime_examples.csv", index=False)
    for name, table in cat_tables.items():
        table.to_csv(OUT / f"regime_by_{name}.csv", index=False)

    paths = [plot_regime_fractions(counts, OUT)]
    paths.extend(plot_regime_profiles(stacked, OUT))
    paths.extend(plot_predictor_box(events, OUT))
    report = write_report(counts, predictors, cat_tables, examples, paths)
    print(report)


if __name__ == "__main__":
    main()
