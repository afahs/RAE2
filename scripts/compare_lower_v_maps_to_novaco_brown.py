#!/usr/bin/env python
"""Compare lower-V differential occultation maps to Novaco & Brown RAE-2 contours.

This comparison is intentionally qualitative. Novaco & Brown published
absolute brightness-temperature contour maps, while the lower-V occultation
product is a mean-zero differential inversion. The tables here therefore
compare sky morphology and feature signs, not absolute scale.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rylevonberg.table_io import read_table
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.util import ensure_dir, software_versions, write_json  # noqa: E402
from scripts.plot_lower_v_haslam_style_maps import (  # noqa: E402
    _add_haslam_axes,
    _display_vmax,
    _format_scale,
    _mollweide_xy,
    _regularize_band,
)


DEFAULT_MAP_TABLE = ROOT / "outputs/lower_v_differential_occultation_maps_allbands_15deg_v1/lower_v_relative_map_table.csv"
DEFAULT_SUMMARY = ROOT / "outputs/lower_v_differential_occultation_maps_allbands_15deg_v1/occultation_design_matrix_summary.csv"
DEFAULT_OUT = ROOT / "outputs/lower_v_differential_occultation_maps_allbands_15deg_v1/novaco_brown_comparison"

NOVACO_BROWN_FREQUENCIES_MHZ = (1.31, 2.20, 3.93, 4.70, 6.55, 9.18)


@dataclass(frozen=True)
class FeatureProbe:
    key: str
    label: str
    lon_deg: float
    lat_deg: float
    lon_half_width_deg: float
    lat_half_width_deg: float
    expected: str
    applies_to_mhz: tuple[float, ...]
    note: str
    score_stat: str = "region_median"


FEATURE_PROBES = (
    FeatureProbe(
        key="local_field_1p31",
        label="1.31 local-field maximum",
        lon_deg=100.0,
        lat_deg=0.0,
        lon_half_width_deg=22.5,
        lat_half_width_deg=15.0,
        expected="high",
        applies_to_mhz=(1.31,),
        note="Novaco & Brown describe the 1.31 MHz maximum as lying near l~100 on the Galactic plane.",
        score_stat="nearest_value",
    ),
    FeatureProbe(
        key="north_galactic_spur",
        label="North Galactic Spur",
        lon_deg=35.0,
        lat_deg=35.0,
        lon_half_width_deg=22.5,
        lat_half_width_deg=22.5,
        expected="high",
        applies_to_mhz=(2.20, 3.93, 4.70, 6.55, 9.18),
        note="Expected as an emission feature in the Novaco & Brown maps above 1.31 MHz.",
    ),
    FeatureProbe(
        key="cetus_arc",
        label="Cetus Arc / southern loop",
        lon_deg=40.0,
        lat_deg=-35.0,
        lon_half_width_deg=22.5,
        lat_half_width_deg=22.5,
        expected="high",
        applies_to_mhz=(2.20, 3.93, 4.70, 6.55, 9.18),
        note="Expected as an emission feature in the Novaco & Brown maps above 1.31 MHz.",
    ),
    FeatureProbe(
        key="gum_nebula",
        label="Gum Nebula",
        lon_deg=260.0,
        lat_deg=-5.0,
        lon_half_width_deg=22.5,
        lat_half_width_deg=22.5,
        expected="low",
        applies_to_mhz=(2.20, 3.93, 4.70, 6.55, 9.18),
        note="Expected as an absorption/low-brightness feature in Novaco & Brown.",
    ),
    FeatureProbe(
        key="orion_association",
        label="Orion Association",
        lon_deg=205.0,
        lat_deg=-15.0,
        lon_half_width_deg=22.5,
        lat_half_width_deg=22.5,
        expected="low",
        applies_to_mhz=(2.20, 3.93, 4.70, 6.55, 9.18),
        note="Expected as an absorption/low-brightness feature in Novaco & Brown.",
    ),
)


REFERENCE_NOTES = [
    "Novaco & Brown (1978), 'Nonthermal Galactic Emission Below 10 MHz', ApJ 221, 114-123.",
    "The published RAE-2 contour maps are at 1.31, 2.20, 3.93, 4.70, 6.55, and 9.18 MHz.",
    "Their contours are absolute brightness temperatures, scaled using IMP-6 average Galactic calibration.",
    "Their summary identifies the North Galactic Spur and Cetus Arc in emission, the Gum Nebula and Orion Association in absorption, and a distinct 1.31 MHz local-field-dominated maximum near Galactic longitude 100 deg.",
    "This lower-V product is a relative, mean-zero differential occultation map, so agreement should be judged by morphology/sign only.",
]


def _model_modes_from_summary(summary: pd.DataFrame | None) -> list[str]:
    if summary is None or "model_mode" not in summary.columns:
        return []
    return sorted(str(value) for value in summary["model_mode"].dropna().unique())


def _product_description(model_modes: list[str]) -> str:
    modes = set(model_modes)
    if modes == {"ring_only"}:
        return "a relative, mean-zero, beam-free ring-only differential occultation inversion"
    if modes == {"beam_weighted"}:
        return "a relative, mean-zero, beam-weighted differential occultation inversion"
    if modes:
        return f"a relative, mean-zero differential occultation inversion with model modes {', '.join(model_modes)}"
    return "a relative, mean-zero differential occultation inversion"


def _lon_delta_deg(lon: pd.Series | np.ndarray, center: float) -> np.ndarray:
    lon_arr = np.asarray(lon, dtype=float)
    return ((lon_arr - float(center) + 180.0) % 360.0) - 180.0


def _covered_band(map_table: pd.DataFrame, frequency_mhz: float) -> pd.DataFrame:
    band = map_table[np.isclose(map_table["frequency_mhz"], frequency_mhz)].copy()
    if "passes_coverage" in band.columns:
        band = band[band["passes_coverage"].astype(bool)].copy()
    return band


def _nearest_row(band: pd.DataFrame, lon_deg: float, lat_deg: float) -> pd.Series:
    distance2 = _lon_delta_deg(band["galactic_l_deg"], lon_deg) ** 2 + (
        pd.to_numeric(band["galactic_b_deg"], errors="coerce").to_numpy(dtype=float) - float(lat_deg)
    ) ** 2
    return band.iloc[int(np.nanargmin(distance2))]


def _percentile(values: pd.Series | np.ndarray, value: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0 or not np.isfinite(value):
        return np.nan
    return float(100.0 * (np.sum(arr < value) + 0.5 * np.sum(arr == value)) / arr.size)


def _region_subset(band: pd.DataFrame, probe: FeatureProbe) -> pd.DataFrame:
    lon_ok = np.abs(_lon_delta_deg(band["galactic_l_deg"], probe.lon_deg)) <= probe.lon_half_width_deg
    lat = pd.to_numeric(band["galactic_b_deg"], errors="coerce")
    lat_ok = np.abs(lat - probe.lat_deg) <= probe.lat_half_width_deg
    return band[lon_ok & lat_ok].copy()


def _interpret(expected: str, percentile: float) -> str:
    if not np.isfinite(percentile):
        return "no covered cells"
    if expected == "high":
        if percentile >= 67.0:
            return "matches high"
        if percentile <= 33.0:
            return "opposite sign"
        return "weak/ambiguous"
    if expected == "low":
        if percentile <= 33.0:
            return "matches low"
        if percentile >= 67.0:
            return "opposite sign"
        return "weak/ambiguous"
    return "not scored"


def build_feature_table(map_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for frequency_mhz in NOVACO_BROWN_FREQUENCIES_MHZ:
        band = _covered_band(map_table, frequency_mhz)
        values = pd.to_numeric(band["relative_brightness"], errors="coerce")
        for probe in FEATURE_PROBES:
            if not any(np.isclose(frequency_mhz, f) for f in probe.applies_to_mhz):
                continue
            near = _nearest_row(band, probe.lon_deg, probe.lat_deg)
            region = _region_subset(band, probe)
            region_values = pd.to_numeric(region["relative_brightness"], errors="coerce")
            if region.empty:
                region_median = np.nan
                region_min = np.nan
                region_max = np.nan
                region_n = 0
            else:
                region_median = float(np.nanmedian(region_values))
                region_min = float(np.nanmin(region_values))
                region_max = float(np.nanmax(region_values))
                region_n = int(region_values.notna().sum())
            region_percentile = _percentile(values, region_median)
            if probe.score_stat == "nearest_value":
                score_value = float(near["relative_brightness"])
                score_percentile = _percentile(values, score_value)
            elif probe.score_stat == "region_max":
                score_value = region_max
                score_percentile = _percentile(values, score_value)
            elif probe.score_stat == "region_min":
                score_value = region_min
                score_percentile = _percentile(values, score_value)
            else:
                score_value = region_median
                score_percentile = region_percentile
            rows.append(
                {
                    "frequency_mhz": frequency_mhz,
                    "feature": probe.label,
                    "expected": probe.expected,
                    "score_stat": probe.score_stat,
                    "score_value": score_value,
                    "score_percentile": score_percentile,
                    "probe_l_deg": probe.lon_deg,
                    "probe_b_deg": probe.lat_deg,
                    "nearest_l_deg": float(near["galactic_l_deg"]),
                    "nearest_b_deg": float(near["galactic_b_deg"]),
                    "nearest_value": float(near["relative_brightness"]),
                    "nearest_percentile": _percentile(values, float(near["relative_brightness"])),
                    "region_n_cells": region_n,
                    "region_median": region_median,
                    "region_min": region_min,
                    "region_max": region_max,
                    "region_percentile": region_percentile,
                    "interpretation": _interpret(probe.expected, score_percentile),
                    "note": probe.note,
                }
            )
    return pd.DataFrame(rows)


def build_extrema_table(map_table: pd.DataFrame, summary: pd.DataFrame | None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for frequency_mhz in NOVACO_BROWN_FREQUENCIES_MHZ:
        band = _covered_band(map_table, frequency_mhz)
        value = pd.to_numeric(band["relative_brightness"], errors="coerce")
        max_row = band.loc[value.idxmax()]
        min_row = band.loc[value.idxmin()]
        near_100 = _nearest_row(band, 100.0, 0.0)
        row = {
            "frequency_mhz": frequency_mhz,
            "max_l_deg": float(max_row["galactic_l_deg"]),
            "max_b_deg": float(max_row["galactic_b_deg"]),
            "max_value": float(max_row["relative_brightness"]),
            "min_l_deg": float(min_row["galactic_l_deg"]),
            "min_b_deg": float(min_row["galactic_b_deg"]),
            "min_value": float(min_row["relative_brightness"]),
            "local_field_probe_value": float(near_100["relative_brightness"]),
            "local_field_probe_percentile": _percentile(value, float(near_100["relative_brightness"])),
            "n_pixels": int(len(band)),
        }
        if summary is not None and not summary.empty:
            match = summary[np.isclose(summary["frequency_mhz"], frequency_mhz)]
            if not match.empty:
                s = match.iloc[0]
                row.update(
                    {
                        "n_measurements": int(s["n_measurements"]),
                        "weighted_rmse": float(s["weighted_rmse"]),
                        "median_abs_residual": float(s["median_abs_residual"]),
                        "beam_model_frequency_mhz": float(s["beam_model_frequency_mhz"]),
                    }
                )
        rows.append(row)
    return pd.DataFrame(rows)


def _galactic_lon_to_plot_x(lon_deg: float) -> float:
    x_deg = ((-float(lon_deg) + 180.0) % 360.0) - 180.0
    return float(np.deg2rad(x_deg))


def _plot_feature_panel(map_table: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(15.6, 7.4), subplot_kw={"projection": "mollweide"})
    for ax, frequency_mhz in zip(axes.ravel(), NOVACO_BROWN_FREQUENCIES_MHZ):
        band = _covered_band(map_table, frequency_mhz)
        lon, lat, data, covered = _regularize_band(
            band,
            "relative_brightness",
            lon_step_deg=1.0,
            lat_step_deg=1.0,
            smooth_sigma_deg=7.5,
            min_coverage_count=8,
        )
        x, y = _mollweide_xy(lon, lat)
        vmax = _display_vmax(data, 97.0)
        image = ax.pcolormesh(x, y, data, shading="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, rasterized=True)
        ax.contour(x, y, data, levels=[0.0], colors="black", linewidths=0.45, alpha=0.55)
        for probe in FEATURE_PROBES:
            if not any(np.isclose(frequency_mhz, f) for f in probe.applies_to_mhz):
                continue
            px = _galactic_lon_to_plot_x(probe.lon_deg)
            py = np.deg2rad(probe.lat_deg)
            if probe.expected == "high":
                ax.scatter(px, py, marker="*", s=72, c="#ffd447", edgecolors="black", linewidths=0.55, zorder=5)
            else:
                ax.scatter(px, py, marker="v", s=48, c="black", edgecolors="white", linewidths=0.45, zorder=5)
        _add_haslam_axes(ax)
        ax.set_title(f"{frequency_mhz:.2f} MHz, scale +/-{_format_scale(vmax)}", fontsize=9.2)
        cbar = fig.colorbar(image, ax=ax, orientation="horizontal", fraction=0.045, pad=0.08)
        cbar.ax.tick_params(labelsize=6, length=2)
    legend_handles = [
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#ffd447", markeredgecolor="black", markersize=9, label="Novaco & Brown emission/high probe"),
        Line2D([0], [0], marker="v", color="none", markerfacecolor="black", markeredgecolor="white", markersize=8, label="Novaco & Brown absorption/low probe"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle(
        "Current lower-V differential maps with Novaco & Brown feature probes\n"
        "Markers are historical-feature probes only; map colors are relative mean-zero occultation brightness.",
        y=0.985,
        fontsize=13,
    )
    fig.subplots_adjust(left=0.025, right=0.985, top=0.88, bottom=0.13, wspace=0.08, hspace=0.28)
    path = out_dir / "novaco_brown_feature_probe_panel.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def _fmt(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return ""
        if abs(value) >= 100:
            return f"{value:.0f}"
        if abs(value) >= 10:
            return f"{value:.1f}"
        return f"{value:.3g}"
    return str(value)


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in df[columns].iterrows():
        rows.append("| " + " | ".join(_fmt(row[col]) for col in columns) + " |")
    return "\n".join([header, divider, *rows])


def _write_report(
    out_dir: Path,
    map_table_path: Path,
    summary_path: Path,
    feature_table: pd.DataFrame,
    extrema_table: pd.DataFrame,
    figure_path: Path,
    config: dict[str, object],
) -> Path:
    scored = feature_table["interpretation"].value_counts().to_dict()
    n_matches = int(scored.get("matches high", 0) + scored.get("matches low", 0))
    n_opposite = int(scored.get("opposite sign", 0))
    n_ambiguous = int(scored.get("weak/ambiguous", 0))

    one31 = extrema_table[np.isclose(extrema_table["frequency_mhz"], 1.31)].iloc[0]
    one31_feature = feature_table[
        np.isclose(feature_table["frequency_mhz"], 1.31) & (feature_table["feature"] == "1.31 local-field maximum")
    ].iloc[0]
    two20_feature = feature_table[
        np.isclose(feature_table["frequency_mhz"], 2.20) & (feature_table["feature"] == "Cetus Arc / southern loop")
    ]
    two20_note = ""
    if not two20_feature.empty:
        t = two20_feature.iloc[0]
        two20_note = (
            f"At the 2.20 MHz Cetus/southern-loop probe, the region median is {t['region_median']:.3g} "
            f"at percentile {t['region_percentile']:.1f}, scored as {t['interpretation']}."
        )

    model_modes = [str(mode) for mode in config.get("model_modes", [])]
    product_description = _product_description(model_modes)
    reference_notes = list(REFERENCE_NOTES[:-1])
    reference_notes.append(f"This lower-V product is {product_description}, so agreement should be judged by morphology/sign only.")
    interpretation_bullets = [
        "- Novaco & Brown use absolute, calibrated brightness-temperature contours; this product has no recoverable monopole and is normalized to mean zero.",
        "- Their published analysis used the Ryle-Vonberg fine output for Galactic radiation, while this lower-V run is based on the current rv2_coarse pipeline assumption.",
        "- The differential occultation rows are sensitive to limb geometry and broad null modes; very broad Galactic emission can be partly absorbed by the mean-zero constraint and regularization.",
    ]
    if set(model_modes) != {"ring_only"}:
        interpretation_bullets.append(
            "- Some frequencies use the nearest available digitized beam model rather than a frequency-specific beam, as shown in the extrema table."
        )

    lines = [
        "# Novaco & Brown Comparison",
        "",
        "## Scope",
        "",
        "This compares the current lower-V differential occultation maps with the RAE-2 contour maps published by Novaco & Brown.",
        f"The comparison is morphological only: Novaco & Brown plotted absolute brightness temperature contours, while this pipeline output is {product_description}.",
        "",
        "## Reference Notes",
        "",
    ]
    lines.extend(f"- {note}" for note in reference_notes)
    lines.extend(
        [
            "",
            "Sources:",
            "- NASA NTRS record: https://ntrs.nasa.gov/citations/19780047176",
            "- NASA NTRS PDF/preprint: https://ntrs.nasa.gov/api/citations/19770024123/downloads/19770024123.pdf",
            "",
            "## Result",
            "",
            f"Feature-probe score across the Novaco & Brown map frequencies: {n_matches} matching signs, {n_ambiguous} weak/ambiguous, and {n_opposite} opposite-sign probes.",
            f"The 1.31 MHz expected local-field probe near l=100 deg, b=0 deg is positive at percentile {one31_feature['score_percentile']:.1f}, but the current map maximum is elsewhere at l={one31['max_l_deg']:.1f}, b={one31['max_b_deg']:.1f}.",
        ]
    )
    if two20_note:
        lines.append(two20_note)
    lines.extend(
        [
            "",
            "The current maps therefore do not reproduce the Novaco & Brown contour morphology in a clean way.",
            "The 1.31 MHz map has a high cell near the reported local-field direction, but it is not the dominant maximum; the 2.20 MHz southern-loop/Cetus probe is not recovered as a strong high feature.",
            "The 3.93 and 6.55 MHz products also need caution because their weighted residuals are much larger than the surrounding bands.",
            "",
            "## Feature Probes",
            "",
            _markdown_table(
                feature_table,
                [
                    "frequency_mhz",
                    "feature",
                    "expected",
                    "score_stat",
                    "score_value",
                    "score_percentile",
                    "region_median",
                    "nearest_l_deg",
                    "nearest_b_deg",
                    "nearest_value",
                    "interpretation",
                ],
            ),
            "",
            "## Current Map Extrema",
            "",
            _markdown_table(
                extrema_table,
                [
                    "frequency_mhz",
                    "max_l_deg",
                    "max_b_deg",
                    "max_value",
                    "min_l_deg",
                    "min_b_deg",
                    "min_value",
                    "weighted_rmse",
                    "beam_model_frequency_mhz",
                ],
            ),
            "",
            "## Diagnostic Figure",
            "",
            f"- `{figure_path}`",
            "",
            "## Interpretation",
            "",
            "The main likely causes of mismatch are methodological rather than a simple plotting-coordinate issue:",
            "",
            *interpretation_bullets,
            "",
            "## Files",
            "",
            f"- Input map table: `{map_table_path}`",
            f"- Inversion summary: `{summary_path}`",
            f"- Feature table: `{out_dir / 'novaco_brown_feature_probes.csv'}`",
            f"- Extrema table: `{out_dir / 'novaco_brown_extrema_comparison.csv'}`",
            f"- Run config: `{out_dir / 'run_config.json'}`",
            "",
            "## Configuration",
            "",
            "```",
            pd.Series(config).to_string(),
            "```",
            "",
        ]
    )
    path = out_dir / "novaco_brown_comparison_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-table", default=str(DEFAULT_MAP_TABLE))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    map_table_path = Path(args.map_table)
    summary_path = Path(args.summary)
    out_dir = ensure_dir(args.out_dir)

    map_table = read_table(map_table_path)
    summary = read_table(summary_path) if summary_path.exists() else None
    config = {
        "map_table": str(map_table_path),
        "summary": str(summary_path),
        "out_dir": str(out_dir),
        "model_modes": _model_modes_from_summary(summary),
        "novaco_brown_frequencies_mhz": list(NOVACO_BROWN_FREQUENCIES_MHZ),
        "feature_probe_keys": [probe.key for probe in FEATURE_PROBES],
        "software_versions": software_versions(),
    }
    write_json(out_dir / "run_config.json", config)

    feature_table = build_feature_table(map_table)
    extrema_table = build_extrema_table(map_table, summary)
    feature_table.to_csv(out_dir / "novaco_brown_feature_probes.csv", index=False)
    extrema_table.to_csv(out_dir / "novaco_brown_extrema_comparison.csv", index=False)
    figure_path = _plot_feature_panel(map_table, out_dir)
    report_path = _write_report(out_dir, map_table_path, summary_path, feature_table, extrema_table, figure_path, config)
    print(report_path)


if __name__ == "__main__":
    main()
