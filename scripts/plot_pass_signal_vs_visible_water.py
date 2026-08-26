#!/usr/bin/env python
"""Attach apparent Earth-disk water coverage to RAE-2 pass-average spectra.

The observer for each point is the exact RAE-2 position relative to Earth at
the closest Moon--RAE--Earth alignment in a lunar orbit.  Water percentage is
the fraction of the apparent Earth disk covered by water, not the binary
surface type at disk centre.  The projected-area calculation includes the
finite Earth-to-spacecraft distance.
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
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.constants import (  # noqa: E402
    EARTH_MOON_DISTANCE_KM,
    EARTH_RADIUS_KM,
    EARTH_UNIT_COLUMNS,
    SPACECRAFT_COLUMNS,
)
from rylevonberg.util import ensure_dir, software_versions, write_json  # noqa: E402
from scripts.investigate_earth_noise_geography import (  # noqa: E402
    _bh_qvalues,
    _markdown,
    _unit_from_lon_lat,
    build_surface_grid,
    load_land_geojson,
)


DEFAULT_INPUT_DIR = ROOT / "outputs/earth_naive_subspacecraft_land_ocean_v1"
DEFAULT_OUT_DIR = ROOT / "outputs/earth_visible_water_signal_v1"


def observer_geometry(passes: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return Earth-fixed observer directions and exact Earth-RAE distances."""
    directions = _unit_from_lon_lat(
        passes["subspacecraft_lon_deg"].to_numpy(dtype=float),
        passes["subspacecraft_lat_deg"].to_numpy(dtype=float),
    )
    spacecraft = passes[SPACECRAFT_COLUMNS].to_numpy(dtype=float)
    moon_to_earth = passes[EARTH_UNIT_COLUMNS].to_numpy(dtype=float, copy=True)
    moon_to_earth /= np.linalg.norm(moon_to_earth, axis=1, keepdims=True)
    earth_to_spacecraft = -float(EARTH_MOON_DISTANCE_KM) * moon_to_earth + spacecraft
    distances = np.linalg.norm(earth_to_spacecraft, axis=1)
    return directions, distances


def visible_water_metrics(
    observer_unit: np.ndarray,
    observer_distance_km: np.ndarray,
    grid: dict[str, np.ndarray],
    earth_radius_km: float = EARTH_RADIUS_KM,
) -> pd.DataFrame:
    """Calculate water fractions of the apparent disk and visible surface.

    For a surface element with outward unit normal ``n`` and observer vector
    ``O``, its exact solid-angle weight is proportional to

        (n.O - R) / |O - R n|^3 dA,

    when ``n.O > R``.  This reduces to ordinary cosine/projected-area
    weighting for a distant observer.
    """
    directions = np.asarray(observer_unit, dtype=float)
    distances = np.asarray(observer_distance_km, dtype=float)
    xyz = np.asarray(grid["xyz"], dtype=float)
    area = np.asarray(grid["area_weight"], dtype=float)
    is_water = ~np.asarray(grid["is_land"], dtype=bool)
    if directions.ndim != 2 or directions.shape[1] != 3:
        raise ValueError("observer_unit must have shape (n, 3)")
    if len(directions) != len(distances):
        raise ValueError("observer directions and distances must have equal length")
    if np.any(~np.isfinite(distances)) or np.any(distances <= float(earth_radius_km)):
        raise ValueError("each observer must be outside Earth")

    rows: list[dict[str, float]] = []
    for direction, distance in zip(directions, distances, strict=True):
        direction = direction / np.linalg.norm(direction)
        mu = xyz @ direction
        radius_ratio = float(earth_radius_km) / float(distance)
        visible = mu > radius_ratio

        # Common distance factors cancel in the fraction.  This dimensionless
        # form is numerically better behaved than using kilometre-scale cubes.
        denominator = np.power(
            np.maximum(1.0 + radius_ratio**2 - 2.0 * radius_ratio * mu, 1e-30),
            1.5,
        )
        projected_weight = np.where(
            visible,
            area * (mu - radius_ratio) / denominator,
            0.0,
        )
        projected_total = float(np.sum(projected_weight))
        surface_weight = np.where(visible, area, 0.0)
        surface_total = float(np.sum(surface_weight))
        projected_water = (
            float(np.sum(projected_weight[is_water]) / projected_total)
            if projected_total > 0.0
            else np.nan
        )
        surface_water = (
            float(np.sum(surface_weight[is_water]) / surface_total)
            if surface_total > 0.0
            else np.nan
        )
        rows.append(
            {
                "visible_water_fraction_projected": projected_water,
                "visible_land_fraction_projected": 1.0 - projected_water,
                "visible_water_percent_projected": 100.0 * projected_water,
                "visible_water_fraction_surface": surface_water,
                "visible_water_percent_surface": 100.0 * surface_water,
                "earth_to_rae2_distance_km": float(distance),
            }
        )
    return pd.DataFrame(rows)


def attach_visible_water(
    passes: pd.DataFrame,
    grid: dict[str, np.ndarray],
) -> pd.DataFrame:
    directions, distances = observer_geometry(passes)
    metrics = visible_water_metrics(directions, distances, grid)
    return pd.concat([passes.reset_index(drop=True), metrics], axis=1)


def build_signal_points(
    pass_power: pd.DataFrame,
    pass_metrics: pd.DataFrame,
    window_minutes: float,
    maximum_alignment_angle_deg: float,
) -> pd.DataFrame:
    """Reduce the two antennas to one pass-average signal per point/channel."""
    inlier = pass_power["robust_inlier"]
    if inlier.dtype != bool:
        inlier = inlier.astype(str).str.lower().eq("true")
    selected = pass_power[
        np.isclose(pass_power["window_minutes"], float(window_minutes))
        & inlier
        & (
            pass_power["minimum_alignment_angle_deg"]
            <= float(maximum_alignment_angle_deg)
        )
    ].copy()
    combined = (
        selected.groupby(
            ["pass_id", "frequency_band", "frequency_mhz"],
            as_index=False,
        )
        .agg(
            average_signal=("mean_raw_power", "mean"),
            average_log10_signal_across_antennas=("mean_log10_power", "mean"),
            month_centered_log10_signal=("month_centered_mean_log10_power", "mean"),
            n_antennas=("antenna", "nunique"),
            n_samples_total=("n_samples", "sum"),
            n_samples_minimum_antenna=("n_samples", "min"),
        )
    )
    combined["log10_average_signal"] = np.log10(combined["average_signal"])

    antenna = selected.pivot_table(
        index=["pass_id", "frequency_band", "frequency_mhz"],
        columns="antenna",
        values=["mean_raw_power", "mean_log10_power"],
        aggfunc="first",
    )
    antenna.columns = [f"{receiver}_{metric}" for metric, receiver in antenna.columns]
    antenna = antenna.reset_index()
    combined = combined.merge(
        antenna,
        on=["pass_id", "frequency_band", "frequency_mhz"],
        how="left",
        validate="one_to_one",
    )
    metadata_columns = [
        "pass_id",
        "center_time",
        "month",
        "minimum_alignment_angle_deg",
        "subspacecraft_lon_deg",
        "subspacecraft_lat_deg",
        "subspacecraft_is_land",
        "surface_class",
        "visible_water_fraction_projected",
        "visible_land_fraction_projected",
        "visible_water_percent_projected",
        "visible_water_fraction_surface",
        "visible_water_percent_surface",
        "earth_to_rae2_distance_km",
    ]
    return combined.merge(
        pass_metrics[metadata_columns],
        on="pass_id",
        how="left",
        validate="many_to_one",
    ).sort_values(["center_time", "frequency_band"])


def association_summary(
    points: pd.DataFrame,
    primary_alignment_angle_deg: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    scopes = {
        f"center_le_{primary_alignment_angle_deg:g}_deg": points[
            points["minimum_alignment_angle_deg"] <= float(primary_alignment_angle_deg)
        ],
        "all_candidate_points": points,
    }
    metrics = {
        "raw_log10_average_signal": "log10_average_signal",
        "month_centered_log10_signal": "month_centered_log10_signal",
    }
    for scope, scoped in scopes.items():
        for metric_name, metric_column in metrics.items():
            for (band, frequency), group in scoped.groupby(
                ["frequency_band", "frequency_mhz"], sort=True
            ):
                x = group["visible_water_percent_projected"].to_numpy(dtype=float)
                y = group[metric_column].to_numpy(dtype=float)
                good = np.isfinite(x) & np.isfinite(y)
                x = x[good]
                y = y[good]
                if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
                    rho = p_value = slope = np.nan
                else:
                    result = stats.spearmanr(x, y)
                    regression = stats.linregress(x, y)
                    rho = float(result.statistic)
                    p_value = float(result.pvalue)
                    slope = float(regression.slope)
                rows.append(
                    {
                        "scope": scope,
                        "metric": metric_name,
                        "frequency_band": int(band),
                        "frequency_mhz": float(frequency),
                        "n_points": int(len(x)),
                        "water_percent_min": float(np.nanmin(x)),
                        "water_percent_max": float(np.nanmax(x)),
                        "spearman_rho": rho,
                        "spearman_p": p_value,
                        "linear_slope_dex_per_10_water_percent": 10.0 * slope,
                    }
                )
    out = pd.DataFrame(rows)
    out["fdr_q_within_scope_metric"] = np.nan
    for _, indices in out.groupby(["scope", "metric"]).groups.items():
        out.loc[indices, "fdr_q_within_scope_metric"] = _bh_qvalues(
            out.loc[indices, "spearman_p"]
        )
    return out


def _binned_means(
    group: pd.DataFrame,
    y_column: str,
    bin_edges: np.ndarray,
    minimum_points: int = 8,
) -> pd.DataFrame:
    work = group[["visible_water_percent_projected", y_column]].dropna().copy()
    work["water_bin"] = pd.cut(
        work["visible_water_percent_projected"],
        bins=bin_edges,
        include_lowest=True,
    )
    binned = work.groupby("water_bin", observed=True).agg(
        water_percent_mean=("visible_water_percent_projected", "mean"),
        signal_mean=(y_column, "mean"),
        signal_sem=(y_column, "sem"),
        n_points=(y_column, "size"),
    )
    return binned[binned["n_points"] >= int(minimum_points)].reset_index(drop=True)


def plot_signal_facets(
    points: pd.DataFrame,
    y_column: str,
    y_label: str,
    title: str,
    path: Path,
) -> Path:
    frequencies = sorted(points["frequency_mhz"].dropna().unique())
    water = points["visible_water_percent_projected"].to_numpy(dtype=float)
    lower = 5.0 * np.floor(np.nanmin(water) / 5.0)
    upper = 5.0 * np.ceil(np.nanmax(water) / 5.0)
    bin_edges = np.arange(lower, upper + 5.01, 5.0)
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True)
    for ax, frequency in zip(axes.flat, frequencies, strict=True):
        group = points[np.isclose(points["frequency_mhz"], frequency)].copy()
        ax.scatter(
            group["visible_water_percent_projected"],
            group[y_column],
            s=10,
            alpha=0.22,
            color="#4e79a7",
            linewidths=0,
            rasterized=True,
        )
        binned = _binned_means(group, y_column, bin_edges)
        if not binned.empty:
            ax.errorbar(
                binned["water_percent_mean"],
                binned["signal_mean"],
                yerr=binned["signal_sem"],
                fmt="o-",
                color="#e15759",
                lw=1.5,
                ms=4,
                capsize=2,
                label="5%-bin mean ± SE",
            )
        valid = group[["visible_water_percent_projected", y_column]].dropna()
        rho = stats.spearmanr(
            valid["visible_water_percent_projected"], valid[y_column]
        ).statistic
        ax.text(
            0.03,
            0.96,
            f"{frequency:g} MHz\nn={len(valid):,}, ρ={rho:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
        )
        ax.grid(alpha=0.18)
    for ax in axes[-1, :]:
        ax.set_xlabel("Water coverage of apparent Earth disk (%)")
    for ax in axes[:, 0]:
        ax.set_ylabel(y_label)
    axes[0, 0].legend(frameon=False, loc="lower left", fontsize=9)
    fig.suptitle(title, fontsize=17)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def spectral_bin_summary(
    points: pd.DataFrame,
    quantiles: int = 5,
) -> pd.DataFrame:
    unique_passes = points[["pass_id", "visible_water_percent_projected"]].drop_duplicates()
    _, edges = pd.qcut(
        unique_passes["visible_water_percent_projected"],
        q=int(quantiles),
        retbins=True,
        duplicates="drop",
    )
    work = points.copy()
    work["water_group"] = pd.cut(
        work["visible_water_percent_projected"],
        bins=edges,
        include_lowest=True,
    )
    out = work.groupby(
        ["water_group", "frequency_band", "frequency_mhz"],
        observed=True,
        as_index=False,
    ).agg(
        water_percent_mean=("visible_water_percent_projected", "mean"),
        water_percent_min=("visible_water_percent_projected", "min"),
        water_percent_max=("visible_water_percent_projected", "max"),
        n_points=("pass_id", "nunique"),
        mean_log10_average_signal=("log10_average_signal", "mean"),
        sem_log10_average_signal=("log10_average_signal", "sem"),
        mean_month_centered_log10_signal=("month_centered_log10_signal", "mean"),
    )
    out["geometric_mean_average_signal"] = 10.0 ** out["mean_log10_average_signal"]
    out["water_group"] = out["water_group"].astype(str)
    return out


def plot_water_group_spectra(summary: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    groups = list(summary["water_group"].drop_duplicates())
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(groups)))
    for color, water_group in zip(colors, groups, strict=True):
        group = summary[summary["water_group"].eq(water_group)].sort_values("frequency_mhz")
        label = (
            f"{group['water_percent_min'].min():.1f}–"
            f"{group['water_percent_max'].max():.1f}% water"
        )
        ax.errorbar(
            group["frequency_mhz"],
            group["mean_log10_average_signal"],
            yerr=group["sem_log10_average_signal"],
            marker="o",
            lw=1.6,
            capsize=2,
            color=color,
            label=label,
        )
    frequencies = sorted(summary["frequency_mhz"].unique())
    ax.set_xscale("log")
    ax.set_xticks(frequencies)
    ax.set_xticklabels([f"{frequency:g}" for frequency in frequencies])
    ax.set_xlabel("RV channel frequency (MHz)")
    ax.set_ylabel("Mean log10 pass-average signal (instrument units)")
    ax.set_title("Average RAE-2 spectrum by apparent Earth-disk water coverage")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, title="Pass groups")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report(
    out_dir: Path,
    pass_metrics: pd.DataFrame,
    points: pd.DataFrame,
    associations: pd.DataFrame,
    primary_alignment_angle_deg: float,
    window_minutes: float,
    grid_step_deg: float,
    plots: list[Path],
) -> Path:
    selected = associations[
        associations["scope"].eq(f"center_le_{primary_alignment_angle_deg:g}_deg")
        & associations["metric"].eq("month_centered_log10_signal")
    ][
        [
            "frequency_mhz",
            "n_points",
            "spearman_rho",
            "spearman_p",
            "fdr_q_within_scope_metric",
            "linear_slope_dex_per_10_water_percent",
        ]
    ].sort_values("frequency_mhz")
    lines = [
        "# RAE-2 Signal Versus Visible-Earth Water Coverage",
        "",
        "## Definition",
        "",
        "Each point is one closest-alignment Earth-facing lunar-orbit pass. The water metric is the percentage of the apparent Earth disk covered by water as seen from the exact RAE-2 position at the pass center. Surface cells are weighted by their exact finite-distance solid angle, so cells near the apparent limb contribute less than cells near disk center.",
        "",
        f"Land/water is sampled on a {grid_step_deg:g}-degree global grid using the cached Natural Earth 1:110m country polygons. Inland water represented as polygon holes counts as water; coastlines and small islands are limited by that map resolution.",
        "",
        f"Signal at each point is the arithmetic mean of the available upper- and lower-V pass-average powers in a ±{window_minutes:g}-minute window. Plots use log10 signal because RV power spans a large dynamic range. Red curves are 5-percentage-point water-bin means, not fitted models.",
        "",
        "## Coverage",
        "",
        f"- Pass points with water metrics: {pass_metrics['pass_id'].nunique():,}",
        f"- Pass-frequency signal points: {len(points):,}",
        f"- Projected water range: {pass_metrics['visible_water_percent_projected'].min():.2f}% to {pass_metrics['visible_water_percent_projected'].max():.2f}%",
        f"- Primary near-line passes (center ≤{primary_alignment_angle_deg:g}°): {pass_metrics.loc[pass_metrics['minimum_alignment_angle_deg'] <= primary_alignment_angle_deg, 'pass_id'].nunique():,}",
        "",
        "## Near-Line Month-Centered Association",
        "",
        _markdown(selected, list(selected.columns)),
        "",
        "The table is descriptive. Month-centering reduces long-term receiver drift; FDR is applied across the nine frequencies. The raw-signal scatterplots remain the direct answer to the requested averaging-and-plotting operation.",
        "",
        "## Files",
        "",
        "- `pass_visible_water_metrics.csv`",
        "- `pass_frequency_signal_with_visible_water.csv`",
        "- `visible_water_signal_associations.csv`",
        "- `visible_water_binned_spectral_averages.csv`",
    ]
    lines.extend(f"- `{plot.name}`" for plot in plots)
    path = out_dir / "visible_water_signal_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--window-minutes", type=float, default=5.0)
    parser.add_argument("--maximum-alignment-angle-deg", type=float, default=30.0)
    parser.add_argument("--primary-alignment-angle-deg", type=float, default=10.0)
    parser.add_argument("--surface-grid-step-deg", type=float, default=1.0)
    args = parser.parse_args()
    if not 0.0 < args.primary_alignment_angle_deg <= args.maximum_alignment_angle_deg:
        raise ValueError("primary alignment angle must be positive and no larger than maximum")
    if not 0.0 < args.surface_grid_step_deg <= 5.0:
        raise ValueError("surface grid step must be in (0, 5] degrees")

    out_dir = ensure_dir(args.out_dir)
    passes = pd.read_csv(
        args.input_dir / "earth_facing_pass_centers.csv",
        parse_dates=["start_time", "end_time", "center_time"],
    )
    passes = passes[
        passes["minimum_alignment_angle_deg"] <= float(args.maximum_alignment_angle_deg)
    ].copy()
    power = pd.read_csv(
        args.input_dir / "earth_facing_pass_power_averages.csv",
        parse_dates=["center_time"],
    )
    geojson_path = args.input_dir / "natural_earth_110m_admin0_countries.geojson"
    geojson = load_land_geojson(geojson_path)
    grid = build_surface_grid(geojson, step_deg=args.surface_grid_step_deg)

    pass_metrics = attach_visible_water(passes, grid)
    pass_metrics.to_csv(out_dir / "pass_visible_water_metrics.csv", index=False)
    points = build_signal_points(
        power,
        pass_metrics,
        args.window_minutes,
        args.maximum_alignment_angle_deg,
    )
    points.to_csv(out_dir / "pass_frequency_signal_with_visible_water.csv", index=False)
    associations = association_summary(points, args.primary_alignment_angle_deg)
    associations.to_csv(out_dir / "visible_water_signal_associations.csv", index=False)
    spectral_bins = spectral_bin_summary(points)
    spectral_bins.to_csv(out_dir / "visible_water_binned_spectral_averages.csv", index=False)

    plots = [
        plot_signal_facets(
            points,
            "log10_average_signal",
            "log10 average signal",
            "Pass-average RAE-2 signal versus apparent Earth-disk water coverage",
            out_dir / "signal_vs_visible_water_by_frequency.png",
        ),
        plot_signal_facets(
            points,
            "month_centered_log10_signal",
            "Month-centered log10 signal",
            "Month-centered RAE-2 signal versus apparent Earth-disk water coverage",
            out_dir / "month_centered_signal_vs_visible_water_by_frequency.png",
        ),
        plot_signal_facets(
            points[
                points["minimum_alignment_angle_deg"]
                <= float(args.primary_alignment_angle_deg)
            ],
            "log10_average_signal",
            "log10 average signal",
            f"Near-line pass-average signal versus water coverage (center ≤{args.primary_alignment_angle_deg:g}°)",
            out_dir / "primary_signal_vs_visible_water_by_frequency.png",
        ),
        plot_water_group_spectra(
            spectral_bins,
            out_dir / "average_spectrum_by_visible_water_group.png",
        ),
    ]
    report = write_report(
        out_dir,
        pass_metrics,
        points,
        associations,
        args.primary_alignment_angle_deg,
        args.window_minutes,
        args.surface_grid_step_deg,
        plots,
    )
    write_json(
        out_dir / "run_config.json",
        {
            **vars(args),
            "input_dir": str(args.input_dir),
            "out_dir": str(args.out_dir),
            "software_versions": software_versions(),
        },
    )
    print(report)


if __name__ == "__main__":
    main()
