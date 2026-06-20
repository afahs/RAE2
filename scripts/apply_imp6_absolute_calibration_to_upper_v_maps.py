#!/usr/bin/env python
"""Apply an IMP-6-tied absolute brightness scale to upper-V scan maps.

The Novaco & Brown RAE-2 maps used RAE-2 for spatial contrast and IMP-6 for
the absolute average Galactic brightness scale.  This script applies the same
kind of operation to a local relative upper-V map table:

1. convert the map's relative dB values to linear ratios;
2. normalize the ratios so the chosen covered-cell weighted mean is unity;
3. multiply by an external IMP-6 brightness-temperature scale by frequency.

The built-in calibration table is Brown (1973) Table 1, the IMP-6 minimum
Galactic spectrum.  Frequencies above 2.6 MHz are extrapolated in brightness
temperature with a configurable spectral index.  This is a provisional
IMP-6-tied scale, because the exact Novaco & Brown average/extrapolated
calibration curve is not available as a machine-readable table in this repo.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.util import ensure_dir, software_versions, write_json  # noqa: E402
from scripts.build_upper_v_novaco_brown_scan_maps import (  # noqa: E402
    _add_haslam_axes,
    _format_scale,
    _mollweide_xy,
    _overlay_uncovered,
    _regularize_band,
)


DEFAULT_MAP_DIR = ROOT / "outputs/upper_v_novaco_brown_scan_maps_rv1_coarse_allfreq_5deg_all_dates_earthgain10db_destripe_v1"
DEFAULT_MAP_TABLE = DEFAULT_MAP_DIR / "upper_v_novaco_brown_map_table.csv"
DEFAULT_OUT = ROOT / "outputs/upper_v_novaco_brown_scan_maps_rv1_coarse_allfreq_5deg_all_dates_earthgain10db_destripe_imp6cal_v1"

C_LIGHT_M_S = 299_792_458.0
K_BOLTZMANN_J_K = 1.380_649e-23

# Brown 1973, ApJ 180, 359, Table 1.  Brightness units are
# 1e-23 W m^-2 Hz^-1 sr^-1.  This table is the IMP-6 minimum Galactic
# spectrum; Novaco & Brown later used an IMP-6 average scale, so outputs are
# labeled as a provisional tied scale unless a user supplies another table.
BROWN_1973_MINIMUM_SPECTRUM = [
    # frequency_khz, brightness_1e_minus_23_w_m2_hz_sr, total_error_percent
    (130.0, 1.5, 46.0),
    (155.0, 1.9, 31.0),
    (185.0, 2.7, 23.0),
    (210.0, 3.7, 25.0),
    (250.0, 7.3, 36.0),
    (292.0, 11.0, 26.0),
    (375.0, 52.0, 17.0),
    (425.0, 84.9, 14.0),
    (475.0, 149.0, 14.0),
    (600.0, 200.0, 13.0),
    (737.0, 365.0, 11.0),
    (825.0, 424.0, 11.0),
    (870.0, 486.0, 10.0),
    (950.0, 504.0, 10.0),
    (1030.0, 585.0, 11.0),
    (1100.0, 672.0, 10.0),
    (1270.0, 754.0, 10.0),
    (1450.0, 825.0, 11.0),
    (1630.0, 915.0, 11.0),
    (1850.0, 910.0, 11.0),
    (2200.0, 1130.0, 12.0),
    (2600.0, 1170.0, 12.0),
]


def brightness_to_temperature_k(frequency_mhz: np.ndarray, brightness_w_m2_hz_sr: np.ndarray) -> np.ndarray:
    freq_hz = np.asarray(frequency_mhz, dtype=float) * 1.0e6
    brightness = np.asarray(brightness_w_m2_hz_sr, dtype=float)
    return brightness * C_LIGHT_M_S**2 / (2.0 * K_BOLTZMANN_J_K * freq_hz**2)


def built_in_brown_table() -> pd.DataFrame:
    rows = []
    for freq_khz, brightness, error_percent in BROWN_1973_MINIMUM_SPECTRUM:
        freq_mhz = float(freq_khz) / 1000.0
        brightness_si = float(brightness) * 1.0e-23
        rows.append(
            {
                "frequency_mhz": freq_mhz,
                "frequency_khz": float(freq_khz),
                "brightness_1e_minus_23_w_m2_hz_sr": float(brightness),
                "brightness_w_m2_hz_sr": brightness_si,
                "brightness_temperature_k": float(brightness_to_temperature_k(freq_mhz, brightness_si)),
                "total_error_percent": float(error_percent),
                "calibration_source": "Brown 1973 Table 1 minimum Galactic spectrum",
            }
        )
    return pd.DataFrame.from_records(rows)


def load_calibration_table(path: Path | None) -> pd.DataFrame:
    if path is None:
        return built_in_brown_table()
    table = pd.read_csv(path)
    if "frequency_mhz" not in table.columns:
        raise ValueError("calibration table must include frequency_mhz")
    if "brightness_temperature_k" in table.columns:
        table = table.copy()
    elif "brightness_w_m2_hz_sr" in table.columns:
        table = table.copy()
        table["brightness_temperature_k"] = brightness_to_temperature_k(
            table["frequency_mhz"].to_numpy(dtype=float),
            table["brightness_w_m2_hz_sr"].to_numpy(dtype=float),
        )
    elif "brightness_1e_minus_23_w_m2_hz_sr" in table.columns:
        table = table.copy()
        table["brightness_w_m2_hz_sr"] = table["brightness_1e_minus_23_w_m2_hz_sr"].to_numpy(dtype=float) * 1.0e-23
        table["brightness_temperature_k"] = brightness_to_temperature_k(
            table["frequency_mhz"].to_numpy(dtype=float),
            table["brightness_w_m2_hz_sr"].to_numpy(dtype=float),
        )
    else:
        raise ValueError(
            "calibration table must include brightness_temperature_k, brightness_w_m2_hz_sr, "
            "or brightness_1e_minus_23_w_m2_hz_sr"
        )
    if "total_error_percent" not in table.columns:
        table["total_error_percent"] = np.nan
    if "calibration_source" not in table.columns:
        table["calibration_source"] = str(path)
    return table


def interpolation_slope_beta(table: pd.DataFrame, n_points: int) -> float:
    work = table.sort_values("frequency_mhz").tail(int(n_points))
    x = np.log(work["frequency_mhz"].to_numpy(dtype=float))
    y = np.log(work["brightness_temperature_k"].to_numpy(dtype=float))
    if len(work) < 2:
        return 2.5
    return float(-np.polyfit(x, y, 1)[0])


def make_frequency_calibration(
    requested_mhz: list[float],
    table: pd.DataFrame,
    high_frequency_beta: float | None,
    high_frequency_beta_fit_points: int,
) -> pd.DataFrame:
    cal = table.sort_values("frequency_mhz").copy()
    cal = cal[np.isfinite(cal["frequency_mhz"]) & np.isfinite(cal["brightness_temperature_k"])]
    if cal.empty:
        raise ValueError("calibration table has no finite frequency/temperature rows")
    x = np.log(cal["frequency_mhz"].to_numpy(dtype=float))
    y = np.log(cal["brightness_temperature_k"].to_numpy(dtype=float))
    err = cal["total_error_percent"].to_numpy(dtype=float)
    freq_min = float(cal["frequency_mhz"].min())
    freq_max = float(cal["frequency_mhz"].max())
    beta = float(high_frequency_beta) if high_frequency_beta is not None else interpolation_slope_beta(cal, high_frequency_beta_fit_points)
    rows = []
    for freq in sorted(float(v) for v in requested_mhz):
        if freq < freq_min:
            # Low-frequency extrapolation is not expected for the upper-V bins,
            # but use the nearest log-log slope rather than failing abruptly.
            slope = (y[1] - y[0]) / (x[1] - x[0]) if len(cal) > 1 else -beta
            log_temp = y[0] + slope * (np.log(freq) - x[0])
            source_kind = "low_frequency_loglog_extrapolated"
        elif freq <= freq_max:
            log_temp = float(np.interp(np.log(freq), x, y))
            source_kind = "loglog_interpolated"
        else:
            log_temp = float(y[-1] - beta * (np.log(freq) - x[-1]))
            source_kind = f"high_frequency_extrapolated_beta_{beta:.4g}"
        error_percent = float(np.interp(np.log(np.clip(freq, freq_min, freq_max)), x, err))
        rows.append(
            {
                "frequency_mhz": freq,
                "imp6_brightness_temperature_k": float(np.exp(log_temp)),
                "imp6_total_error_percent": error_percent,
                "calibration_source_kind": source_kind,
                "high_frequency_temperature_spectral_index_beta": beta,
                "calibration_table_min_frequency_mhz": freq_min,
                "calibration_table_max_frequency_mhz": freq_max,
            }
        )
    return pd.DataFrame.from_records(rows)


def area_weights(lat_deg: pd.Series | np.ndarray) -> np.ndarray:
    return np.clip(np.cos(np.deg2rad(np.asarray(lat_deg, dtype=float))), 0.0, None)


def apply_absolute_scale(map_table: pd.DataFrame, cal_by_frequency: pd.DataFrame, mean_weighting: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = map_table.copy()
    out["relative_linear_ratio"] = 10.0 ** (out["relative_brightness"].to_numpy(dtype=float) / 10.0)
    out["area_weight"] = area_weights(out["galactic_b_deg"])
    mode = str(mean_weighting).strip().lower().replace("_", "-")
    rows = []
    out["calibration_ratio_normalization"] = np.nan
    out["imp6_brightness_temperature_k"] = np.nan
    out["absolute_brightness_temperature_k"] = np.nan
    out["log10_absolute_brightness_temperature_k"] = np.nan
    out["absolute_brightness_temperature_mk"] = np.nan
    for _, cal in cal_by_frequency.iterrows():
        freq = float(cal["frequency_mhz"])
        mask = np.isclose(out["frequency_mhz"].to_numpy(dtype=float), freq)
        band = out.loc[mask].copy()
        covered = band["passes_coverage"].astype(bool) & np.isfinite(band["relative_linear_ratio"])
        if mode == "area":
            weights = band.loc[covered, "area_weight"].to_numpy(dtype=float)
        elif mode == "sample":
            weights = band.loc[covered, "coverage_count"].to_numpy(dtype=float)
        elif mode == "uniform":
            weights = np.ones(int(np.count_nonzero(covered)), dtype=float)
        else:
            raise ValueError(f"unsupported mean weighting: {mean_weighting}")
        ratios = band.loc[covered, "relative_linear_ratio"].to_numpy(dtype=float)
        norm = float(np.average(ratios, weights=weights)) if ratios.size and np.any(weights > 0) else np.nan
        temp0 = float(cal["imp6_brightness_temperature_k"])
        absolute = temp0 * band["relative_linear_ratio"].to_numpy(dtype=float) / norm if np.isfinite(norm) and norm > 0 else np.full(len(band), np.nan)
        out.loc[mask, "calibration_ratio_normalization"] = norm
        out.loc[mask, "imp6_brightness_temperature_k"] = temp0
        out.loc[mask, "absolute_brightness_temperature_k"] = absolute
        out.loc[mask, "log10_absolute_brightness_temperature_k"] = np.log10(np.where(absolute > 0, absolute, np.nan))
        out.loc[mask, "absolute_brightness_temperature_mk"] = absolute / 1.0e6
        finite_abs = absolute[covered.to_numpy(dtype=bool)] if len(absolute) else np.array([], dtype=float)
        rows.append(
            {
                "frequency_band": int(band["frequency_band"].iloc[0]) if not band.empty else np.nan,
                "frequency_mhz": freq,
                "imp6_brightness_temperature_k": temp0,
                "imp6_brightness_temperature_mk": temp0 / 1.0e6,
                "imp6_total_error_percent": float(cal["imp6_total_error_percent"]),
                "calibration_source_kind": str(cal["calibration_source_kind"]),
                "calibration_ratio_normalization": norm,
                "mean_weighting": mode,
                "n_grid_cells": int(len(band)),
                "n_covered_cells": int(np.count_nonzero(covered)),
                "covered_temperature_min_k": float(np.nanmin(finite_abs)) if finite_abs.size else np.nan,
                "covered_temperature_median_k": float(np.nanmedian(finite_abs)) if finite_abs.size else np.nan,
                "covered_temperature_mean_k": float(np.average(finite_abs, weights=weights)) if finite_abs.size and np.any(weights > 0) else np.nan,
                "covered_temperature_max_k": float(np.nanmax(finite_abs)) if finite_abs.size else np.nan,
                "covered_log10_temperature_min": float(np.nanmin(np.log10(finite_abs))) if finite_abs.size else np.nan,
                "covered_log10_temperature_max": float(np.nanmax(np.log10(finite_abs))) if finite_abs.size else np.nan,
            }
        )
    return out, pd.DataFrame.from_records(rows)


def display_limits(data: np.ndarray, percentile: float) -> tuple[float, float]:
    finite = np.asarray(data, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 5.0, 7.0
    lo = float(np.nanpercentile(finite, 100.0 - float(percentile)))
    hi = float(np.nanpercentile(finite, float(percentile)))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo = float(np.nanmin(finite))
        hi = float(np.nanmax(finite))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        center = float(np.nanmedian(finite)) if finite.size else 6.0
        lo, hi = center - 0.1, center + 0.1
    return lo, hi


def _add_log_colorbar(fig: plt.Figure, ax: plt.Axes, image, lo: float, hi: float) -> None:
    cbar = fig.colorbar(image, ax=ax, orientation="horizontal", fraction=0.052, pad=0.05)
    ticks = np.linspace(lo, hi, 3)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{10.0 ** tick / 1.0e6:.2g}" for tick in ticks])
    cbar.ax.tick_params(labelsize=5.5, length=1.8, pad=1)
    cbar.outline.set_linewidth(0.4)


def plot_panel(
    map_table: pd.DataFrame,
    out_dir: Path,
    percentile: float,
    smooth_sigma_deg: float,
    calibration_name: str,
    output_prefix: str,
) -> Path:
    freqs = sorted(map_table["frequency_mhz"].dropna().unique())
    ncols = 3
    nrows = int(np.ceil(len(freqs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16.0, 3.15 * nrows + 1.0), subplot_kw={"projection": "mollweide"})
    axes_arr = np.asarray(axes).ravel()
    for ax, freq in zip(axes_arr, freqs):
        band = map_table[np.isclose(map_table["frequency_mhz"], freq)].copy()
        lon, lat, data, covered = _regularize_band(
            band,
            "log10_absolute_brightness_temperature_k",
            lon_step_deg=1.0,
            lat_step_deg=1.0,
            smooth_sigma_deg=smooth_sigma_deg,
        )
        x, y = _mollweide_xy(lon, lat)
        lo, hi = display_limits(data, percentile)
        image = ax.pcolormesh(x, y, data, shading="auto", cmap="inferno", vmin=lo, vmax=hi, rasterized=True)
        _overlay_uncovered(ax, x, y, covered)
        levels = np.linspace(lo, hi, 7)
        if np.isfinite(data).any():
            ax.contour(x, y, data, levels=levels, colors="white", linewidths=0.35, alpha=0.55)
        _add_haslam_axes(ax)
        scale_mk = float(band["imp6_brightness_temperature_k"].dropna().iloc[0]) / 1.0e6 if band["imp6_brightness_temperature_k"].notna().any() else np.nan
        ax.set_title(f"{freq:.2f} MHz, scale {scale_mk:.2g} MK", fontsize=9.2)
        _add_log_colorbar(fig, ax, image, lo, hi)
    for ax in axes_arr[len(freqs) :]:
        ax.set_visible(False)
    fig.suptitle(
        f"Upper-V scan maps with {calibration_name} absolute brightness scale\n"
        "Galactic Mollweide; colorbar ticks are MK; gray hatch marks cells below coverage threshold.",
        y=0.995,
        fontsize=13,
    )
    fig.legend(
        handles=[Patch(facecolor="0.70", edgecolor="0.25", hatch="////", label="not covered / below sample threshold")],
        loc="lower right",
        bbox_to_anchor=(0.985, 0.02),
        frameon=False,
        fontsize=8,
    )
    fig.subplots_adjust(left=0.03, right=0.985, top=0.88, bottom=0.12, wspace=0.08, hspace=0.25)
    path = out_dir / f"{output_prefix}_absolute_brightness_temperature_panel.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def write_report(
    out_dir: Path,
    config: dict[str, object],
    summary: pd.DataFrame,
    panel_path: Path,
    paths: list[Path],
) -> Path:
    calibration_name = str(config.get("calibration_name", "external calibration"))
    lines = [
        f"# Upper-V {calibration_name} Absolute Calibration",
        "",
        f"This product applies an external `{calibration_name}` brightness-temperature scale to an existing relative upper-V scan map.",
        "It uses the local map for spatial contrast and the external calibration table for the absolute average scale.",
        "",
        "## Calibration Formula",
        "",
        "`T_cell = T_cal(freq) * 10^(relative_dB_cell / 10) / <10^(relative_dB / 10)>_covered`",
        "",
        f"The covered-cell mean uses `{config['mean_weighting']}` weighting.",
        "",
        "## Important Caveats",
        "",
        "- This fixes the absolute scale only. It does not fix pointing, beam-shape, coverage, or filtering differences.",
        "- Interpolation/extrapolation choices are recorded in `upper_v_absolute_calibration_by_frequency.csv`.",
        "- If the calibration table is digitized from a figure rather than a machine-readable table, treat the absolute scale as approximate.",
        "- The local map remains a scan-map product from the selected RAE-2 channel and cuts; it is not a beam-deconvolved all-sky model.",
        "",
        "## Summary",
        "",
        summary.to_string(index=False),
        "",
        "## Outputs",
        "",
    ]
    lines.extend(f"- `{path}`" for path in paths)
    lines.extend(
        [
            "",
            "## Sources",
            "",
            str(config.get("source_note", "- External calibration table specified in the run configuration.")),
            "",
            "## Configuration",
            "",
            pd.Series(config).to_string(),
            "",
        ]
    )
    report = out_dir / f"{config.get('output_prefix', 'upper_v')}_absolute_calibration_report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run(args: argparse.Namespace) -> Path:
    out_dir = ensure_dir(args.out_dir)
    map_table_path = Path(args.map_table)
    map_table = pd.read_csv(map_table_path)
    cal_source = load_calibration_table(Path(args.calibration_table) if args.calibration_table else None)
    freqs = sorted(float(v) for v in map_table["frequency_mhz"].dropna().unique())
    cal_by_freq = make_frequency_calibration(
        freqs,
        cal_source,
        args.high_frequency_temperature_spectral_index,
        int(args.high_frequency_beta_fit_points),
    )
    calibrated, summary = apply_absolute_scale(map_table, cal_by_freq, args.mean_weighting)

    config = {
        "map_table": str(map_table_path),
        "calibration_table": str(args.calibration_table) if args.calibration_table else "built_in_brown_1973_table_1_minimum_spectrum",
        "calibration_name": str(args.calibration_name),
        "source_note": str(args.source_note),
        "output_prefix": str(args.output_prefix),
        "out_dir": str(out_dir),
        "mean_weighting": str(args.mean_weighting),
        "high_frequency_temperature_spectral_index": None
        if args.high_frequency_temperature_spectral_index is None
        else float(args.high_frequency_temperature_spectral_index),
        "high_frequency_beta_fit_points": int(args.high_frequency_beta_fit_points),
        "resolved_high_frequency_temperature_spectral_index": float(
            cal_by_freq["high_frequency_temperature_spectral_index_beta"].iloc[0]
        ),
        "scale_percentile": float(args.scale_percentile),
        "smooth_sigma_deg": float(args.smooth_sigma_deg),
        "software_versions": software_versions(),
    }
    write_json(out_dir / "run_config.json", config)
    source_table = out_dir / f"{args.output_prefix}_source_calibration_table.csv"
    by_frequency = out_dir / f"{args.output_prefix}_absolute_calibration_by_frequency.csv"
    calibrated_table = out_dir / f"{args.output_prefix}_calibrated_map_table.csv"
    summary_path = out_dir / f"{args.output_prefix}_absolute_calibration_summary.csv"
    cal_source.to_csv(source_table, index=False)
    cal_by_freq.to_csv(by_frequency, index=False)
    calibrated.to_csv(calibrated_table, index=False)
    summary.to_csv(summary_path, index=False)
    panel = plot_panel(
        calibrated,
        out_dir,
        float(args.scale_percentile),
        float(args.smooth_sigma_deg),
        str(args.calibration_name),
        str(args.output_prefix),
    )
    paths = [
        out_dir / "run_config.json",
        source_table,
        by_frequency,
        calibrated_table,
        summary_path,
        panel,
    ]
    report = write_report(out_dir, config, summary, panel, paths)
    print(report)
    print(summary.to_string(index=False))
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-table", default=str(DEFAULT_MAP_TABLE))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--calibration-table", default=None)
    parser.add_argument("--calibration-name", default="IMP-6/Brown 1973")
    parser.add_argument("--output-prefix", default="upper_v_imp6")
    parser.add_argument(
        "--source-note",
        default="- Brown 1973, 'The Galactic Radio Spectrum Between 130 and 2600 kHz', ApJ 180, 359-370.\n- Novaco & Brown 1978, 'Nonthermal Galactic Emission Below 10 MHz', ApJ 221, 114-123.",
    )
    parser.add_argument("--mean-weighting", choices=["area", "sample", "uniform"], default="area")
    parser.add_argument(
        "--high-frequency-temperature-spectral-index",
        type=float,
        default=None,
        help="Temperature spectral index beta for T proportional to nu^-beta above the calibration table. Default fits the high-frequency tail of the calibration table.",
    )
    parser.add_argument("--high-frequency-beta-fit-points", type=int, default=5)
    parser.add_argument("--smooth-sigma-deg", type=float, default=4.0)
    parser.add_argument("--scale-percentile", type=float, default=97.0)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
