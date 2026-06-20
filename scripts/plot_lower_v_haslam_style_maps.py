#!/usr/bin/env python
"""Plot lower-V differential occultation maps in a Haslam-style Galactic view."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd
from rylevonberg.table_io import read_table
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.util import ensure_dir, write_json, software_versions  # noqa: E402


DEFAULT_INPUT = ROOT / "outputs/lower_v_differential_occultation_maps_allbands_15deg_v1/lower_v_relative_map_table.csv"
DEFAULT_OUT = ROOT / "outputs/lower_v_differential_occultation_maps_allbands_15deg_v1/haslam_style"
UNCOVERED_CMAP = ListedColormap([(0.70, 0.70, 0.70, 0.62)])


def _regularize_band(
    band: pd.DataFrame,
    value_col: str,
    lon_step_deg: float,
    lat_step_deg: float,
    smooth_sigma_deg: float,
    min_coverage_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pivot = (
        band.pivot_table(index="galactic_b_deg", columns="galactic_l_deg", values=value_col, aggfunc="first")
        .sort_index()
        .sort_index(axis=1)
    )
    lons = pivot.columns.to_numpy(dtype=float)
    lats = pivot.index.to_numpy(dtype=float)
    vals = pivot.to_numpy(dtype=float)
    coverage = _coverage_pivot(band, pivot, min_coverage_count)

    lon_ext = np.r_[lons[-1] - 360.0, lons, lons[0] + 360.0]
    vals_ext = np.column_stack([vals[:, -1], vals, vals[:, 0]])
    coverage_ext = np.column_stack([coverage[:, -1], coverage, coverage[:, 0]])
    interp = RegularGridInterpolator(
        (lats, lon_ext),
        vals_ext,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    coverage_interp = RegularGridInterpolator(
        (lats, lon_ext),
        coverage_ext.astype(float),
        method="nearest",
        bounds_error=False,
        fill_value=0.0,
    )

    lon_plot = np.arange(-180.0, 180.0 + 0.5 * lon_step_deg, lon_step_deg)
    lon_fine = (-lon_plot) % 360.0
    lat_fine = np.arange(-89.5, 90.0, lat_step_deg)
    lon_mesh, lat_mesh = np.meshgrid(lon_fine, lat_fine)
    lon_plot_mesh, _ = np.meshgrid(lon_plot, lat_fine)
    query = np.column_stack([lat_mesh.ravel(), lon_mesh.ravel()])
    image = interp(query).reshape(lat_mesh.shape)
    covered = coverage_interp(query).reshape(lat_mesh.shape) >= 0.5

    # Fill polar interpolation gaps with the nearest finite latitude row, then
    # lightly smooth so a 15-degree inversion grid reads as an all-sky map.
    for row in range(image.shape[0]):
        if np.isfinite(image[row]).any():
            first = row
            break
    else:
        return lon_plot_mesh, lat_mesh, image, covered
    for row in range(image.shape[0] - 1, -1, -1):
        if np.isfinite(image[row]).any():
            last = row
            break
    image[:first] = image[first]
    image[last + 1 :] = image[last]
    covered[:first] = covered[first]
    covered[last + 1 :] = covered[last]

    if smooth_sigma_deg > 0:
        sigma = smooth_sigma_deg / max(lat_step_deg, 1e-9)
        finite = np.isfinite(image) & covered
        weighted = gaussian_filter(np.where(finite, image, 0.0), sigma=sigma, mode=("nearest", "wrap"))
        norm = gaussian_filter(finite.astype(float), sigma=sigma, mode=("nearest", "wrap"))
        image = np.divide(weighted, norm, out=np.full_like(weighted, np.nan), where=norm > 0)
    image = np.where(covered, image, np.nan)
    return lon_plot_mesh, lat_mesh, image, covered


def _coverage_pivot(band: pd.DataFrame, value_pivot: pd.DataFrame, min_coverage_count: int) -> np.ndarray:
    if "passes_coverage" in band.columns:
        cov = (
            band.assign(_covered=band["passes_coverage"].astype(bool))
            .pivot_table(index="galactic_b_deg", columns="galactic_l_deg", values="_covered", aggfunc="first")
            .reindex(index=value_pivot.index, columns=value_pivot.columns)
        )
    elif "coverage_count" in band.columns:
        cov = (
            band.assign(_covered=pd.to_numeric(band["coverage_count"], errors="coerce").ge(int(min_coverage_count)))
            .pivot_table(index="galactic_b_deg", columns="galactic_l_deg", values="_covered", aggfunc="first")
            .reindex(index=value_pivot.index, columns=value_pivot.columns)
        )
    else:
        cov = pd.DataFrame(True, index=value_pivot.index, columns=value_pivot.columns)
    return cov.fillna(False).to_numpy(dtype=bool)


def _mollweide_xy(lon_mesh: np.ndarray, lat_mesh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Galactic convention used by many Haslam renderings: l=0 centered, l
    # increases to the left.
    x = np.deg2rad(lon_mesh)
    y = np.deg2rad(lat_mesh)
    return x, y


def _add_haslam_axes(ax: plt.Axes) -> None:
    ax.grid(color="white", alpha=0.28, linewidth=0.55)
    ticks = np.array([-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150])
    labels = [f"{int((-tick) % 360)}°" for tick in ticks]
    ax.set_xticks(np.deg2rad(ticks))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_yticks(np.deg2rad([-60, -30, 0, 30, 60]))
    ax.set_yticklabels(["-60°", "-30°", "0°", "+30°", "+60°"], fontsize=7)
    ax.text(0.5, -0.08, "Galactic longitude", transform=ax.transAxes, ha="center", va="top", fontsize=8)


def _overlay_uncovered(ax: plt.Axes, x: np.ndarray, y: np.ndarray, covered: np.ndarray) -> None:
    uncovered = np.where(~covered, 1.0, np.nan)
    if np.isfinite(uncovered).any():
        ax.pcolormesh(x, y, uncovered, shading="auto", cmap=UNCOVERED_CMAP, vmin=0.0, vmax=1.0, rasterized=True)
        ax.contourf(x, y, uncovered, levels=[0.5, 1.5], colors="none", hatches=["////"], alpha=0.0)


def _coverage_counts(map_table: pd.DataFrame, min_coverage_count: int) -> pd.DataFrame:
    work = map_table.copy()
    if "passes_coverage" in work.columns:
        work["_covered"] = work["passes_coverage"].astype(bool)
    elif "coverage_count" in work.columns:
        work["_covered"] = pd.to_numeric(work["coverage_count"], errors="coerce").ge(int(min_coverage_count))
    else:
        work["_covered"] = True
    return (
        work.groupby("frequency_mhz", as_index=False)
        .agg(n_pixels=("pixel_index", "size"), n_uncovered=("_covered", lambda s: int((~s.astype(bool)).sum())))
        .sort_values("frequency_mhz")
    )


def _covered_values(map_table: pd.DataFrame, value_col: str, min_coverage_count: int) -> pd.Series:
    values = pd.to_numeric(map_table[value_col], errors="coerce")
    if "passes_coverage" in map_table.columns:
        return values.where(map_table["passes_coverage"].astype(bool))
    if "coverage_count" in map_table.columns:
        return values.where(pd.to_numeric(map_table["coverage_count"], errors="coerce").ge(int(min_coverage_count)))
    return values


def _display_vmax(values: np.ndarray | pd.Series, percentile: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 1.0
    vmax = float(np.nanpercentile(np.abs(arr), float(percentile)))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.nanmax(np.abs(arr)))
    return vmax if np.isfinite(vmax) and vmax > 0 else 1.0


def _format_scale(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    if abs(value) >= 1:
        return f"{value:.2f}"
    return f"{value:.2g}"


def _add_panel_colorbar(fig: plt.Figure, ax: plt.Axes, image, vmax: float) -> None:
    cax = inset_axes(ax, width="44%", height="4.5%", loc="lower center", borderpad=1.15)
    cbar = fig.colorbar(image, cax=cax, orientation="horizontal")
    cbar.set_ticks([-vmax, 0.0, vmax])
    label = _format_scale(vmax)
    cbar.set_ticklabels([f"-{label}", "0", f"+{label}"])
    cbar.ax.tick_params(labelsize=5.5, length=1.8, pad=1)
    cbar.outline.set_linewidth(0.4)


def _plot_panel(
    map_table: pd.DataFrame,
    out_dir: Path,
    value_col: str,
    smooth_sigma_deg: float,
    min_coverage_count: int,
    scale_percentile: float,
) -> Path:
    freqs = sorted(map_table["frequency_mhz"].dropna().unique())
    fig, axes = plt.subplots(3, 3, figsize=(16.0, 9.2), subplot_kw={"projection": "mollweide"})
    for ax, freq in zip(axes.ravel(), freqs):
        band = map_table[np.isclose(map_table["frequency_mhz"], freq)].copy()
        lon, lat, data, covered = _regularize_band(
            band,
            value_col,
            lon_step_deg=1.0,
            lat_step_deg=1.0,
            smooth_sigma_deg=smooth_sigma_deg,
            min_coverage_count=min_coverage_count,
        )
        x, y = _mollweide_xy(lon, lat)
        vmax = _display_vmax(data, scale_percentile)
        image = ax.pcolormesh(x, y, data, shading="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, rasterized=True)
        _overlay_uncovered(ax, x, y, covered)
        ax.contour(x, y, data, levels=[0.0], colors="black", linewidths=0.45, alpha=0.55)
        _add_haslam_axes(ax)
        ax.set_title(f"{freq:.2f} MHz, scale +/-{_format_scale(vmax)}", fontsize=9.2)
        _add_panel_colorbar(fig, ax, image, vmax)
    fig.suptitle(
        "Lower-V differential occultation maps, Haslam-style Galactic Mollweide\n"
        "Each frequency uses its own color scale; gray hatch marks uncovered map cells.",
        y=0.99,
        fontsize=13,
    )
    fig.legend(
        handles=[Patch(facecolor="0.70", edgecolor="0.25", hatch="////", label="not covered / below coverage threshold")],
        loc="lower right",
        bbox_to_anchor=(0.985, 0.02),
        frameon=False,
        fontsize=8,
    )
    fig.subplots_adjust(left=0.03, right=0.985, top=0.90, bottom=0.12, wspace=0.08, hspace=0.22)
    path = out_dir / "lower_v_relative_maps_haslam_style_panel.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def _plot_single_bands(
    map_table: pd.DataFrame,
    out_dir: Path,
    value_col: str,
    smooth_sigma_deg: float,
    min_coverage_count: int,
    scale_percentile: float,
) -> list[Path]:
    paths = []
    for freq in sorted(map_table["frequency_mhz"].dropna().unique()):
        band = map_table[np.isclose(map_table["frequency_mhz"], freq)].copy()
        lon, lat, data, covered = _regularize_band(
            band,
            value_col,
            lon_step_deg=0.5,
            lat_step_deg=0.5,
            smooth_sigma_deg=smooth_sigma_deg,
            min_coverage_count=min_coverage_count,
        )
        x, y = _mollweide_xy(lon, lat)
        vmax = _display_vmax(data, scale_percentile)
        fig = plt.figure(figsize=(12.4, 6.8))
        ax = fig.add_subplot(111, projection="mollweide")
        image = ax.pcolormesh(x, y, data, shading="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, rasterized=True)
        _overlay_uncovered(ax, x, y, covered)
        ax.contour(x, y, data, levels=[0.0], colors="black", linewidths=0.55, alpha=0.6)
        _add_haslam_axes(ax)
        uncovered_count = int((~band["passes_coverage"].astype(bool)).sum()) if "passes_coverage" in band else 0
        ax.set_title(
            f"Lower-V differential occultation map, {freq:.2f} MHz; scale +/-{_format_scale(vmax)}; uncovered cells: {uncovered_count}",
            fontsize=12.5,
        )
        cbar = fig.colorbar(image, ax=ax, orientation="horizontal", fraction=0.055, pad=0.09)
        cbar.set_label("relative mean-zero occultation brightness")
        ax.legend(
            handles=[Patch(facecolor="0.70", edgecolor="0.25", hatch="////", label="not covered / below coverage threshold")],
            loc="lower right",
            bbox_to_anchor=(1.0, -0.11),
            frameon=False,
            fontsize=8,
        )
        fig.subplots_adjust(left=0.03, right=0.98, top=0.91, bottom=0.13)
        freq_label = f"{freq:.2f}".replace(".", "p")
        path = out_dir / f"lower_v_{freq_label}mhz_haslam_style.png"
        fig.savefig(path, dpi=210)
        plt.close(fig)
        paths.append(path)
    return paths


def _write_index(out_dir: Path, input_path: Path, paths: list[Path], config: dict[str, object], coverage: pd.DataFrame) -> Path:
    lines = [
        "# Haslam-Style Lower-V Differential Occultation Maps",
        "",
        "These are presentation-oriented Galactic Mollweide renderings of the differential lower-V inversion output.",
        "They use the familiar Haslam-map convention: Galactic center centered, Galactic plane horizontal, and longitude increasing to the left.",
        "",
        "The plotted values remain the relative mean-zero occultation map values from the inversion, not absolute brightness temperature.",
        "",
        "Uncovered regions, if present, are masked from the color map and overlaid in gray with diagonal hatching.",
        "Each frequency panel uses its own symmetric color scale, printed in the subplot title and shown on its colorbar.",
        "For this input table, the coverage counts are:",
        "",
        coverage.to_string(index=False),
        "",
        "## Input",
        "",
        f"- `{input_path}`",
        "",
        "## Figures",
        "",
    ]
    lines.extend(f"- `{path}`" for path in paths)
    lines.extend(["", "## Configuration", "", pd.Series(config).to_string(), ""])
    path = out_dir / "haslam_style_map_index.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--value-col", default="relative_brightness")
    parser.add_argument("--smooth-sigma-deg", type=float, default=7.5)
    parser.add_argument("--min-coverage-count", type=int, default=8)
    parser.add_argument("--scale-percentile", type=float, default=97.0)
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = ensure_dir(args.out_dir)
    table = read_table(input_path)
    if args.value_col not in table.columns:
        raise SystemExit(f"missing value column: {args.value_col}")
    config = {
        "input": str(input_path),
        "value_col": str(args.value_col),
        "smooth_sigma_deg": float(args.smooth_sigma_deg),
        "min_coverage_count": int(args.min_coverage_count),
        "scale_percentile": float(args.scale_percentile),
        "software_versions": software_versions(),
    }
    write_json(out_dir / "run_config.json", config)
    coverage = _coverage_counts(table, int(args.min_coverage_count))
    paths = [
        _plot_panel(
            table,
            out_dir,
            args.value_col,
            args.smooth_sigma_deg,
            int(args.min_coverage_count),
            float(args.scale_percentile),
        )
    ]
    paths.extend(
        _plot_single_bands(
            table,
            out_dir,
            args.value_col,
            args.smooth_sigma_deg,
            int(args.min_coverage_count),
            float(args.scale_percentile),
        )
    )
    index = _write_index(out_dir, input_path, paths, config, coverage)
    print(index)


if __name__ == "__main__":
    main()
