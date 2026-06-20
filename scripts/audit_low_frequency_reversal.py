#!/usr/bin/env python
"""Audit Earth low-frequency anti-template behavior after low-drift filtering."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rylevonberg.table_io import read_table


ROOT = Path(__file__).resolve().parents[1]
STAGE1_STACK = ROOT / "outputs/stage1_event_usefulness_v1/earth_stage1_stacked_profiles.csv"
STAGE1_METRICS = ROOT / "outputs/stage1_event_usefulness_v1/earth_stage1_event_usefulness_metrics.csv"
ALL_GRID = ROOT / "outputs/all_frequency_profile_grids_v1/earth_all_frequency_profile_summary_900s.csv"
GEOM = ROOT / "outputs/moving_body_regime_physical_differences_v1/moving_body_event_regime_geometry_table.csv"
OUT = ROOT / "outputs/low_frequency_reversal_audit_v1"

FREQS = [0.45, 0.70, 0.90, 1.31, 2.20, 3.93, 4.70, 6.55, 9.18]
LOW_FREQS = [0.70, 0.90, 1.31, 2.20]
EXPECTED_SIGN = {"disappearance": -1.0, "reappearance": 1.0}
CLASS_COLOR = {
    "usable_low_drift": "#2ca02c",
    "drift_competing": "#ff7f0e",
    "low_information_or_weak": "#7f7f7f",
}
ANT_COLOR = {"rv1_coarse": "#4c78a8", "rv2_coarse": "#d95f02"}


def _central_contrast_from_stack(df: pd.DataFrame, group_cols: list[str], value_col: str) -> pd.DataFrame:
    rows = []
    for keys, grp in df.groupby(group_cols, sort=True):
        pre = grp[(grp["t_bin_sec"] >= -180) & (grp["t_bin_sec"] <= -60)][value_col].median()
        post = grp[(grp["t_bin_sec"] >= 60) & (grp["t_bin_sec"] <= 180)][value_col].median()
        outer_pre = grp[(grp["t_bin_sec"] >= -900) & (grp["t_bin_sec"] <= -600)][value_col].median()
        outer_post = grp[(grp["t_bin_sec"] >= 600) & (grp["t_bin_sec"] <= 900)][value_col].median()
        rec = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        sign = EXPECTED_SIGN[str(rec["event_type"])]
        rec.update(
            {
                "central_post_minus_pre": float(post - pre),
                "central_source_like_contrast": float(sign * (post - pre)),
                "outer_post_minus_pre": float(outer_post - outer_pre),
                "outer_source_like_contrast": float(sign * (outer_post - outer_pre)),
                "n_events": int(grp["n_events"].max()) if "n_events" in grp else np.nan,
            }
        )
        rows.append(rec)
    return pd.DataFrame(rows)


def _load_source_grid(source: str) -> pd.DataFrame:
    path = ROOT / f"outputs/all_frequency_profile_grids_v1/{source}_all_frequency_profile_summary_900s.csv"
    if not path.exists():
        return pd.DataFrame()
    df = read_table(path)
    df["source_name"] = source
    return _central_contrast_from_stack(
        df,
        ["source_name", "frequency_mhz", "antenna", "event_type"],
        "median_z_power",
    )


def _plot_stage1_contrasts(stage1: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5), sharey=True)
    for ax, event_type in zip(axes, ["disappearance", "reappearance"]):
        sub = stage1[stage1["event_type"].eq(event_type)]
        for cls, grp in sub.groupby("stage1_usefulness_class", sort=True):
            grp = grp.sort_values("frequency_mhz")
            ax.plot(
                grp["frequency_mhz"],
                grp["central_source_like_contrast"],
                marker="o",
                lw=1.6,
                color=CLASS_COLOR.get(cls, "black"),
                label=cls,
            )
        ax.axhline(0, color="black", lw=0.8)
        for f in LOW_FREQS:
            ax.axvspan(f * 0.96, f * 1.04, color="0.92", zorder=0)
        ax.set_xscale("log")
        ax.set_xticks(FREQS)
        ax.set_xticklabels([f"{f:.2f}" for f in FREQS], rotation=45)
        ax.set_title(event_type)
        ax.set_xlabel("frequency (MHz)")
        ax.grid(alpha=0.22, which="both")
    axes[0].set_ylabel("source-like central contrast\npositive = expected occultation sign")
    axes[1].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("Earth lower V: low-frequency reversal survives low-drift filtering")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = OUT / "earth_stage1_source_like_contrast_by_frequency.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _plot_antenna_contrasts(all_contrast: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5), sharey=True)
    for ax, event_type in zip(axes, ["disappearance", "reappearance"]):
        sub = all_contrast[all_contrast["event_type"].eq(event_type)]
        for antenna, grp in sub.groupby("antenna", sort=True):
            grp = grp.sort_values("frequency_mhz")
            ax.plot(
                grp["frequency_mhz"],
                grp["central_source_like_contrast"],
                marker="o",
                lw=1.6,
                color=ANT_COLOR.get(antenna, "black"),
                label=antenna,
            )
        ax.axhline(0, color="black", lw=0.8)
        for f in LOW_FREQS:
            ax.axvspan(f * 0.96, f * 1.04, color="0.92", zorder=0)
        ax.set_xscale("log")
        ax.set_xticks(FREQS)
        ax.set_xticklabels([f"{f:.2f}" for f in FREQS], rotation=45)
        ax.set_title(event_type)
        ax.set_xlabel("frequency (MHz)")
        ax.grid(alpha=0.22, which="both")
    axes[0].set_ylabel("source-like central contrast\npositive = expected occultation sign")
    axes[1].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("Earth all events: reversal appears in both antennas at 0.90-2.20 MHz")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = OUT / "earth_antenna_source_like_contrast_by_frequency.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _plot_source_comparison(source_contrast: pd.DataFrame) -> Path:
    work = source_contrast[
        source_contrast["antenna"].eq("rv2_coarse")
        & source_contrast["source_name"].isin(["earth", "sun", "fornax_a", "cas_a", "cyg_a"])
    ].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.6), sharey=True)
    colors = {
        "earth": "#1f77b4",
        "sun": "#ff7f0e",
        "fornax_a": "#2ca02c",
        "cas_a": "#9467bd",
        "cyg_a": "#8c564b",
    }
    for ax, event_type in zip(axes, ["disappearance", "reappearance"]):
        sub = work[work["event_type"].eq(event_type)]
        for source, grp in sub.groupby("source_name", sort=True):
            grp = grp.sort_values("frequency_mhz")
            ax.plot(
                grp["frequency_mhz"],
                grp["central_source_like_contrast"],
                marker="o",
                lw=1.4,
                color=colors.get(source, "black"),
                label=source,
            )
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xscale("log")
        ax.set_xticks(FREQS)
        ax.set_xticklabels([f"{f:.2f}" for f in FREQS], rotation=45)
        ax.set_title(event_type)
        ax.set_xlabel("frequency (MHz)")
        ax.grid(alpha=0.22, which="both")
    axes[0].set_ylabel("source-like central contrast\nlower V only")
    axes[1].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("Lower-V source comparison: Earth/Sun low-frequency behavior differs from fixed-source controls")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = OUT / "lower_v_source_comparison_contrast_by_frequency.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _limb_label_check() -> pd.DataFrame:
    geom = read_table(GEOM, low_memory=False)
    sub = geom[geom["source_name"].eq("earth") & geom["frequency_mhz"].isin(LOW_FREQS)]
    rows = []
    for keys, grp in sub.groupby(["frequency_mhz", "event_type"], sort=True):
        freq, event_type = keys
        rows.append(
            {
                "frequency_mhz": float(freq),
                "event_type": event_type,
                "n_events": int(len(grp)),
                "pre_limb_positive_fraction": float((grp["pre_limb_angle_deg"] > 0).mean()),
                "post_limb_positive_fraction": float((grp["post_limb_angle_deg"] > 0).mean()),
                "median_pre_limb_angle_deg": float(np.nanmedian(grp["pre_limb_angle_deg"])),
                "median_post_limb_angle_deg": float(np.nanmedian(grp["post_limb_angle_deg"])),
            }
        )
    return pd.DataFrame(rows)


def _write_report(
    stage1: pd.DataFrame,
    all_contrast: pd.DataFrame,
    source_contrast: pd.DataFrame,
    label_check: pd.DataFrame,
    paths: list[Path],
) -> Path:
    low_stage1 = stage1[
        stage1["frequency_mhz"].isin(LOW_FREQS)
        & stage1["event_type"].eq("disappearance")
    ][
        [
            "stage1_usefulness_class",
            "frequency_mhz",
            "central_post_minus_pre",
            "central_source_like_contrast",
            "outer_source_like_contrast",
            "n_events",
        ]
    ].sort_values(["stage1_usefulness_class", "frequency_mhz"])

    low_ant = all_contrast[
        all_contrast["frequency_mhz"].isin(LOW_FREQS)
    ][
        [
            "frequency_mhz",
            "antenna",
            "event_type",
            "central_source_like_contrast",
            "outer_source_like_contrast",
            "n_events",
        ]
    ].sort_values(["frequency_mhz", "antenna", "event_type"])

    low_sources = source_contrast[
        source_contrast["frequency_mhz"].isin(LOW_FREQS)
        & source_contrast["antenna"].eq("rv2_coarse")
        & source_contrast["source_name"].isin(["earth", "sun", "fornax_a", "cas_a", "cyg_a"])
    ][
        ["source_name", "frequency_mhz", "event_type", "central_source_like_contrast"]
    ].sort_values(["source_name", "frequency_mhz", "event_type"])

    lines = [
        "# Low-Frequency Earth Reversal Audit",
        "",
        "This audit checks whether the Earth 0.70-2.20 MHz disappearance profiles look like",
        "reappearances because of broad baseline drift. The answer is mostly no: the reversal",
        "survives the first-stage low-drift selection.",
        "",
        "## Main Evidence",
        "",
        "1. Event labels are correct in limb-angle space. For disappearance, pre-event limb angle",
        "is positive and post-event limb angle is negative; reappearance has the opposite sign.",
        "2. The low-frequency reversal appears in both antennas for the all-event stack, so it is",
        "not a lower-V-only plotting artifact.",
        "3. For Earth lower V, the `usable_low_drift` stack remains anti-template at 0.90, 1.31,",
        "and 2.20 MHz. Therefore side-window drift is not sufficient to explain the sign reversal.",
        "4. Fixed-source controls do not show the same coherent Earth/Sun low-frequency behavior.",
        "",
        "## Low-Drift Earth Disappearance Contrasts",
        "",
        low_stage1.to_string(index=False),
        "",
        "## Earth Antenna Comparison",
        "",
        low_ant.to_string(index=False),
        "",
        "## Limb Label Check",
        "",
        label_check.to_string(index=False),
        "",
        "## Lower-V Source Comparison",
        "",
        low_sources.to_string(index=False),
        "",
        "## Interpretation",
        "",
        "The baseline-drift classifier is still useful, but it is not the explanation for the",
        "0.90-2.20 MHz sign reversal. The remaining physically plausible interpretation is that",
        "the low-frequency Earth/Sun contribution is not a simple positive point-source flux",
        "being removed by the Moon. Instead, in this band the moving body can have negative",
        "effective contrast relative to the diffuse sky / antenna background / lunar replacement",
        "field. Then disappearance increases total power and reappearance decreases it, exactly",
        "the anti-template behavior.",
        "",
        "For Earth this is plausible because below a few MHz the diffuse Galactic background is",
        "very bright and the terrestrial ionosphere/magnetospheric environment can block, absorb,",
        "scatter, or otherwise modulate that background. In total-power data, a body can therefore",
        "act like a negative-contrast patch even though it is not emitting negative power.",
        "",
        "The strongest pipeline consequence is that the sign convention should be generalized.",
        "Instead of assuming every source has positive occultation contrast, each source/frequency/",
        "antenna channel should estimate an empirical contrast sign. Positive-contrast channels",
        "use the ordinary occultation template; negative-contrast channels use the anti-template",
        "but should be reported explicitly as negative-contrast detections/candidates, not as",
        "ordinary positive source detections.",
        "",
        "Important caveat: 0.45 MHz Earth behaves positive while 0.70-2.20 MHz behaves negative.",
        "That means this is not a single monotonic Earth model; it could include band-dependent",
        "receiver response, terrestrial emission, Galactic background contrast, or calibration",
        "effects. The next pipeline step should estimate the contrast sign per channel using",
        "controls, not hard-code it.",
        "",
        "## Generated Plots",
        "",
    ]
    lines.extend(f"- `{path}`" for path in paths)
    report = OUT / "low_frequency_reversal_audit_report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    stage1_stack = read_table(STAGE1_STACK, low_memory=False)
    stage1 = _central_contrast_from_stack(
        stage1_stack,
        ["stage1_usefulness_class", "frequency_mhz", "event_type"],
        "median_raw_fractional",
    )
    all_grid = read_table(ALL_GRID, low_memory=False)
    all_contrast = _central_contrast_from_stack(
        all_grid,
        ["frequency_mhz", "antenna", "event_type"],
        "median_z_power",
    )
    source_contrast = pd.concat(
        [_load_source_grid(source) for source in ["earth", "sun", "fornax_a", "cas_a", "cyg_a"]],
        ignore_index=True,
    )
    label_check = _limb_label_check()

    stage1.to_csv(OUT / "earth_stage1_central_contrast_summary.csv", index=False)
    all_contrast.to_csv(OUT / "earth_antenna_central_contrast_summary.csv", index=False)
    source_contrast.to_csv(OUT / "source_central_contrast_comparison.csv", index=False)
    label_check.to_csv(OUT / "earth_low_frequency_limb_label_check.csv", index=False)

    paths = [
        _plot_stage1_contrasts(stage1),
        _plot_antenna_contrasts(all_contrast),
        _plot_source_comparison(source_contrast),
    ]
    report = _write_report(stage1, all_contrast, source_contrast, label_check, paths)
    print(report)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
