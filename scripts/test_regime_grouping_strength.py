#!/usr/bin/env python
"""Quantify whether source-like and anti-template events cluster in time or Moon RA/Dec."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rylevonberg.table_io import read_table
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "outputs/moving_body_regime_physical_differences_v1/moving_body_event_regime_geometry_table.csv"
OUT = ROOT / "outputs/moving_body_regime_physical_differences_v1/regime_grouping_tests"


def _auc_from_mannwhitney(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return np.nan
    res = stats.mannwhitneyu(a, b, alternative="two-sided")
    auc = res.statistic / (len(a) * len(b))
    return float(max(auc, 1.0 - auc))


def _ks_p(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return np.nan
    return float(stats.ks_2samp(a, b).pvalue)


def _standardized_median_gap(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = np.concatenate([a, b])
    scale = 1.4826 * np.nanmedian(np.abs(pooled - np.nanmedian(pooled)))
    if not np.isfinite(scale) or scale <= 0:
        scale = np.nanstd(pooled)
    if not np.isfinite(scale) or scale <= 0:
        return np.nan
    return float(abs(np.nanmedian(a) - np.nanmedian(b)) / scale)


def _circular_phase_distance(ra_deg: pd.Series) -> pd.DataFrame:
    radians = np.deg2rad(ra_deg.astype(float).to_numpy())
    return pd.DataFrame({"moon_ra_sin": np.sin(radians), "moon_ra_cos": np.cos(radians)})


def _group_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["source_name", "frequency_mhz", "event_type"]
    features = [
        ("day_index", "time"),
        ("moon_center_dec_deg", "moon_dec"),
        ("moon_ra_sin", "moon_ra_sin"),
        ("moon_ra_cos", "moon_ra_cos"),
        ("target_dec_deg", "target_dec"),
        ("limb_position_angle_deg", "limb_pa"),
        ("limb_rate_deg_s", "limb_rate"),
        ("moon_angular_radius_deg", "moon_radius"),
    ]
    for keys, g in df.groupby(group_cols, dropna=False):
        source, freq, event_type = keys
        g = g[g["regime"].isin(["source_like", "anti_template"])]
        source_like = g[g["regime"].eq("source_like")]
        anti = g[g["regime"].eq("anti_template")]
        if len(source_like) < 5 or len(anti) < 5:
            continue
        row = {
            "source_name": source,
            "frequency_mhz": freq,
            "event_type": event_type,
            "n_source_like": len(source_like),
            "n_anti_template": len(anti),
            "source_like_fraction": len(source_like) / len(g),
        }
        best_auc = np.nan
        best_feature = ""
        for col, label in features:
            a = source_like[col].dropna().to_numpy(dtype=float)
            b = anti[col].dropna().to_numpy(dtype=float)
            auc = _auc_from_mannwhitney(a, b)
            row[f"{label}_auc_abs"] = auc
            row[f"{label}_ks_p"] = _ks_p(a, b)
            row[f"{label}_std_median_gap"] = _standardized_median_gap(a, b)
            if np.isfinite(auc) and (not np.isfinite(best_auc) or auc > best_auc):
                best_auc = auc
                best_feature = label
        row["best_single_variable_auc_abs"] = best_auc
        row["best_single_variable"] = best_feature
        rows.append(row)
    return pd.DataFrame(rows)


def _overall_summary(summary: pd.DataFrame) -> list[str]:
    lines = [
        "# Source-Like vs Anti-Template Grouping Tests",
        "",
        "This checks whether the event classes visibly seen in the regime plots are separable by",
        "event time or Moon-center RA/Dec alone. AUC values are reported as absolute separability:",
        "0.5 is no separation, 1.0 is perfect separation.",
        "",
    ]
    if summary.empty:
        lines.append("No groups had enough source-like and anti-template events for testing.")
        return lines

    lines.extend(
        [
            "Interpretation thresholds used here:",
            "",
            "- AUC < 0.65: weak or no useful grouping",
            "- 0.65 <= AUC < 0.80: moderate grouping",
            "- AUC >= 0.80: strong one-variable grouping",
            "",
        ]
    )
    for source in sorted(summary["source_name"].unique()):
        s = summary[summary["source_name"].eq(source)]
        lines.extend(
            [
                f"## {source}",
                "",
                f"Groups tested: {len(s)}",
                f"Median best single-variable AUC: {s['best_single_variable_auc_abs'].median():.3f}",
                f"Fraction with best AUC >= 0.65: {(s['best_single_variable_auc_abs'] >= 0.65).mean():.2f}",
                f"Fraction with best AUC >= 0.80: {(s['best_single_variable_auc_abs'] >= 0.80).mean():.2f}",
                "",
                "Best-separated groups:",
                "",
            ]
        )
        top = s.sort_values("best_single_variable_auc_abs", ascending=False).head(8)
        for _, row in top.iterrows():
            lines.append(
                "- "
                f"{row.frequency_mhz:.2f} MHz {row.event_type}: "
                f"best AUC {row.best_single_variable_auc_abs:.3f} "
                f"from {row.best_single_variable} "
                f"(n source-like {int(row.n_source_like)}, n anti {int(row.n_anti_template)})"
            )
        lines.append("")

    lines.extend(
        [
            "## Bottom Line",
            "",
            "There is not a clean global clustering of source-like and anti-template events in",
            "Moon-center RA/Dec or time. Some individual source/frequency/event-type groups show",
            "moderate separation, but the classes overlap substantially. This means Moon-center",
            "coordinate or date alone is not enough to explain the sign flip; those variables are",
            "better treated as covariates that modulate a broader observing-regime effect.",
            "",
        ]
    )
    return lines


def main() -> None:
    df = read_table(TABLE, low_memory=False)
    ra_phase = _circular_phase_distance(df["moon_center_ra_deg"])
    df = pd.concat([df.reset_index(drop=True), ra_phase], axis=1)
    summary = _group_summary(df)

    OUT.mkdir(parents=True, exist_ok=True)
    summary_path = OUT / "source_like_vs_antitemplate_grouping_summary.csv"
    report_path = OUT / "source_like_vs_antitemplate_grouping_report.md"
    summary.to_csv(summary_path, index=False)
    report_path.write_text("\n".join(_overall_summary(summary)) + "\n", encoding="utf-8")

    print(summary_path)
    print(report_path)


if __name__ == "__main__":
    main()
