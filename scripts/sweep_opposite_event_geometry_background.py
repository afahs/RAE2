#!/usr/bin/env python
"""Sweep opposite-event geometry-background subtraction configurations."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.util import ensure_dir, software_versions, write_json  # noqa: E402
from scripts.run_opposite_event_geometry_background import (  # noqa: E402
    LOW_FREQS,
    POINTS,
    _read,
    contrast,
    matched_background_correct,
    summarize,
)


OUT = ROOT / "outputs/opposite_event_geometry_background_sweep_v1"
FEATURE_SETS = {
    "pa_b_slope": ["limb_pa_sin", "limb_pa_cos", "target_galactic_b_deg", "far_raw_slope_per_s"],
    "pa_b": ["limb_pa_sin", "limb_pa_cos", "target_galactic_b_deg"],
    "pa_slope": ["limb_pa_sin", "limb_pa_cos", "far_raw_slope_per_s"],
    "b_slope": ["target_galactic_b_deg", "far_raw_slope_per_s"],
    "pa_only": ["limb_pa_sin", "limb_pa_cos"],
}
K_VALUES = [10, 20, 30, 50, 80]
FAST_FEATURE_SETS = {
    "pa_b_slope": FEATURE_SETS["pa_b_slope"],
    "pa_b": FEATURE_SETS["pa_b"],
    "pa_slope": FEATURE_SETS["pa_slope"],
}
FAST_K_VALUES = [20, 30, 50]
FOCUS_ROWS = [
    ("earth", 0.45, "disappearance"),
    ("earth", 0.45, "reappearance"),
    ("earth", 0.70, "disappearance"),
    ("earth", 0.70, "reappearance"),
    ("earth", 0.90, "disappearance"),
    ("earth", 0.90, "reappearance"),
    ("earth", 1.31, "disappearance"),
    ("earth", 1.31, "reappearance"),
    ("earth", 2.20, "disappearance"),
    ("earth", 2.20, "reappearance"),
    ("sun", 0.45, "disappearance"),
    ("sun", 0.45, "reappearance"),
    ("sun", 0.70, "disappearance"),
    ("sun", 0.70, "reappearance"),
    ("sun", 0.90, "disappearance"),
    ("sun", 0.90, "reappearance"),
    ("sun", 1.31, "disappearance"),
    ("sun", 1.31, "reappearance"),
    ("sun", 2.20, "disappearance"),
    ("sun", 2.20, "reappearance"),
]


def _config_score(contrasts: pd.DataFrame) -> dict[str, float]:
    work = contrasts[contrasts["method"].eq("geometry_opposite_event_subtracted")].copy()
    focus = pd.DataFrame(FOCUS_ROWS, columns=["source_name", "frequency_mhz", "event_type"])
    merged = focus.merge(work, on=["source_name", "frequency_mhz", "event_type"], how="left")
    vals = pd.to_numeric(merged["source_like_contrast"], errors="coerce")
    earth_control = vals.iloc[0:2].min()
    hard = merged[
        ~(
            merged["source_name"].eq("earth")
            & merged["frequency_mhz"].eq(0.45)
        )
    ].copy()
    hard_vals = pd.to_numeric(hard["source_like_contrast"], errors="coerce")
    return {
        "earth_045_min": float(earth_control) if np.isfinite(earth_control) else np.nan,
        "n_positive_hard_rows": int((hard_vals > 0).sum()),
        "median_hard_contrast": float(np.nanmedian(hard_vals)),
        "min_hard_contrast": float(np.nanmin(hard_vals)),
        "n_rows": int(vals.notna().sum()),
    }


def run_sweep(points: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_contrasts = []
    score_rows = []
    for feature_name, features in FAST_FEATURE_SETS.items():
        for k in FAST_K_VALUES:
            corrected, _ = matched_background_correct(points, k_neighbors=k, feature_cols=features)
            if corrected.empty:
                continue
            corr_summary = summarize(corrected, "geometry_background_corrected", "geometry_opposite_event_subtracted")
            contrasts = contrast(corr_summary)
            contrasts["feature_name"] = feature_name
            contrasts["k_neighbors"] = int(k)
            all_contrasts.append(contrasts)
            row = {"feature_name": feature_name, "k_neighbors": int(k), **_config_score(contrasts)}
            score_rows.append(row)
    contrast_table = pd.concat(all_contrasts, ignore_index=True) if all_contrasts else pd.DataFrame()
    score_table = pd.DataFrame(score_rows)
    return contrast_table, score_table


def plot_sweep_scores(scores: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
    metrics = [
        ("n_positive_hard_rows", "positive hard rows"),
        ("median_hard_contrast", "median hard contrast"),
        ("earth_045_min", "Earth 0.45 minimum"),
    ]
    for ax, (col, title) in zip(axes, metrics):
        for feature_name, grp in scores.groupby("feature_name", sort=True):
            grp = grp.sort_values("k_neighbors")
            ax.plot(grp["k_neighbors"], grp[col], marker="o", lw=1.4, label=feature_name)
        ax.set_title(title)
        ax.set_xlabel("k neighbors")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("score")
    axes[-1].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("Opposite-event geometry background sweep")
    fig.tight_layout()
    path = out_dir / "opposite_event_geometry_sweep_scores.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report(out_dir: Path, scores: pd.DataFrame, contrasts: pd.DataFrame, plot_path: Path) -> None:
    ranked = scores.sort_values(
        ["n_positive_hard_rows", "earth_045_min", "median_hard_contrast"],
        ascending=[False, False, False],
    )
    best = ranked.iloc[0] if not ranked.empty else pd.Series(dtype=object)
    lines = [
        "# Opposite-Event Geometry Background Sweep",
        "",
        "This sweeps the physical matching variables and number of opposite-event neighbors.",
        "The ranking rewards positive source-like contrast in hard Earth/Sun rows while requiring Earth 0.45 to stay positive.",
        "",
        "## Ranked Configurations",
        "",
        ranked.to_string(index=False) if not ranked.empty else "No configurations.",
        "",
        "## Best Configuration Contrasts",
        "",
    ]
    if not best.empty:
        sub = contrasts[
            contrasts["feature_name"].eq(best["feature_name"])
            & contrasts["k_neighbors"].eq(best["k_neighbors"])
        ].copy()
        lines.append(
            sub[
                ["source_name", "frequency_mhz", "event_type", "n_events", "source_like_contrast"]
            ].sort_values(["source_name", "frequency_mhz", "event_type"]).to_string(index=False)
        )
    lines.extend(
        [
            "",
            "## Plot",
            "",
            f"- `{plot_path}`",
        ]
    )
    (out_dir / "opposite_event_geometry_sweep_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = ensure_dir(OUT)
    write_json(
        out_dir / "run_config.json",
        {
            "feature_sets": FEATURE_SETS,
            "k_values": K_VALUES,
            "executed_feature_sets": FAST_FEATURE_SETS,
            "executed_k_values": FAST_K_VALUES,
            "points": str(POINTS),
            "software_versions": software_versions(),
        },
    )
    points = _read(POINTS)
    points = points[points["source_name"].isin(["earth", "sun"]) & points["frequency_mhz"].isin(LOW_FREQS)].copy()
    contrasts, scores = run_sweep(points, out_dir)
    contrasts.to_csv(out_dir / "opposite_event_geometry_sweep_contrasts.csv", index=False)
    scores.to_csv(out_dir / "opposite_event_geometry_sweep_scores.csv", index=False)
    plot_path = plot_sweep_scores(scores, out_dir)
    write_report(out_dir, scores, contrasts, plot_path)
    print(out_dir / "opposite_event_geometry_sweep_report.md")


if __name__ == "__main__":
    main()
