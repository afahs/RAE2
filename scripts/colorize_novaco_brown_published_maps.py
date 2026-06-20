#!/usr/bin/env python
"""Colorize and recenter the published Novaco & Brown contour maps.

This script works from the ADS scanned journal PDF of Novaco & Brown (1978).
The published maps are contour-line figures, not a machine-readable sky grid.
The output therefore preserves the scanned contour linework and reprojects the
linework into the same l=0-centered Galactic Mollweide convention used by the
pipeline presentation plots.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from urllib.request import urlretrieve

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.util import ensure_dir, software_versions, write_json  # noqa: E402


ADS_PDF_URL = "https://articles.adsabs.harvard.edu/pdf/1978ApJ...221..114N"
NTRS_RECORD_URL = "https://ntrs.nasa.gov/citations/19780047176"
NTRS_PDF_URL = "https://ntrs.nasa.gov/api/citations/19770024123/downloads/19770024123.pdf"

DEFAULT_OUT = ROOT / "outputs/novaco_brown_published_color_maps_v1"
DEFAULT_PDF = DEFAULT_OUT / "source/ads/1978ApJ_221_114N.pdf"

# The source figures are in an ADS PDF, pages 5-7, two maps per page.
# The source map projection is centered near l=100 deg: its horizontal axis
# labels run ... 140, 120, 100, 80, 60 ... from left to right.
SOURCE_CENTER_L_DEG = 100.0
SOURCE_LAT_LIMIT_DEG = 80.0


@dataclass(frozen=True)
class MapSpec:
    frequency_mhz: float
    pdf_page: int
    panel: str
    unit_label: str
    contour_units: float


MAP_SPECS = (
    MapSpec(9.18, 5, "top", "10^5 K", 1.0e5),
    MapSpec(6.55, 5, "bottom", "10^5 K", 1.0e5),
    MapSpec(4.70, 6, "top", "10^6 K", 1.0e6),
    MapSpec(3.93, 6, "bottom", "10^6 K", 1.0e6),
    MapSpec(2.20, 7, "top", "10^6 K", 1.0e6),
    MapSpec(1.31, 7, "bottom", "10^7 K", 1.0e7),
)


def _page_output_path(render_dir: Path, pdf_page: int) -> Path:
    # Ghostscript emits pages 5,6,7 as page_01,page_02,page_03 because the
    # render command starts at FirstPage=5.
    return render_dir / f"ads_page_{pdf_page:02d}.png"


def _download_pdf(pdf_path: Path, refresh: bool) -> None:
    if pdf_path.exists() and not refresh:
        return
    ensure_dir(pdf_path.parent)
    urlretrieve(ADS_PDF_URL, pdf_path)


def _render_pdf_pages(pdf_path: Path, render_dir: Path, dpi: int, refresh: bool) -> list[Path]:
    ensure_dir(render_dir)
    expected = [_page_output_path(render_dir, page) for page in (5, 6, 7)]
    if all(path.exists() for path in expected) and not refresh:
        return expected
    tmp_pattern = render_dir / "tmp_page_%02d.png"
    cmd = [
        "gs",
        "-q",
        "-dNOSAFER",
        "-dBATCH",
        "-dNOPAUSE",
        "-sDEVICE=png16m",
        f"-r{int(dpi)}",
        "-dFirstPage=5",
        "-dLastPage=7",
        f"-sOutputFile={tmp_pattern}",
        str(pdf_path),
    ]
    subprocess.run(cmd, check=True)
    for i, page in enumerate((5, 6, 7), start=1):
        src = render_dir / f"tmp_page_{i:02d}.png"
        dst = _page_output_path(render_dir, page)
        src.replace(dst)
    return expected


def _source_bbox(page_shape: tuple[int, int], panel: str) -> tuple[float, float, float, float]:
    """Approximate source projection bbox for the ADS page render.

    The ADS page render is stable at a given DPI; these fractions were chosen
    against a 220-dpi render and scale with the rendered page dimensions.  The
    box maps the visible l-center +/-180 deg equator span and b=+/-80 deg
    center-meridian span.
    """
    height, width = page_shape
    left = 0.138 * width
    right = 0.865 * width
    if panel == "top":
        top = 0.145 * height
        bottom = 0.370 * height
    elif panel == "bottom":
        top = 0.540 * height
        bottom = 0.760 * height
    else:
        raise ValueError(f"unsupported panel: {panel}")
    return left, top, right, bottom


def _wrap180(deg: np.ndarray | float) -> np.ndarray:
    return (np.asarray(deg, dtype=float) + 180.0) % 360.0 - 180.0


def _mollweide_theta(lat_rad: np.ndarray) -> np.ndarray:
    theta = np.asarray(lat_rad, dtype=float).copy()
    theta = np.clip(theta, -np.pi / 2.0, np.pi / 2.0)
    for _ in range(12):
        denom = 2.0 + 2.0 * np.cos(2.0 * theta)
        step = (2.0 * theta + np.sin(2.0 * theta) - np.pi * np.sin(lat_rad)) / np.where(
            np.abs(denom) < 1e-12, np.nan, denom
        )
        step = np.nan_to_num(step, nan=0.0, posinf=0.0, neginf=0.0)
        theta -= step
    return theta


def _mollweide_xy_for_lonlat(lon_centered_deg: np.ndarray, lat_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lam = np.deg2rad(_wrap180(lon_centered_deg))
    phi = np.deg2rad(lat_deg)
    theta = _mollweide_theta(phi)
    x = (2.0 * np.sqrt(2.0) / np.pi) * lam * np.cos(theta)
    y = np.sqrt(2.0) * np.sin(theta)
    return x, y


def _load_grayscale(path: Path) -> np.ndarray:
    import matplotlib.image as mpimg

    arr = mpimg.imread(path)
    if arr.dtype.kind in {"u", "i"}:
        arr = arr.astype(float) / np.iinfo(arr.dtype).max
    if arr.ndim == 3:
        arr = arr[..., :3].mean(axis=2)
    return np.asarray(arr, dtype=float)


def _reproject_ink(
    page_gray: np.ndarray,
    spec: MapSpec,
    lon_step_deg: float,
    lat_step_deg: float,
    source_center_l_deg: float,
    source_lat_limit_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lon_plot = np.arange(-180.0, 180.0 + 0.5 * lon_step_deg, lon_step_deg)
    lat_plot = np.arange(-float(source_lat_limit_deg), float(source_lat_limit_deg) + 0.5 * lat_step_deg, lat_step_deg)
    lon_mesh, lat_mesh = np.meshgrid(lon_plot, lat_plot)

    gal_l = (-lon_mesh) % 360.0
    source_centered_l = -(gal_l - float(source_center_l_deg))
    src_x_proj, src_y_proj = _mollweide_xy_for_lonlat(source_centered_l, lat_mesh)
    _, y_limit = _mollweide_xy_for_lonlat(np.zeros_like(lat_mesh), np.full_like(lat_mesh, float(source_lat_limit_deg)))
    y_limit_value = float(np.nanmedian(np.abs(y_limit[np.isfinite(y_limit)])))

    left, top, right, bottom = _source_bbox(page_gray.shape, spec.panel)
    cx = 0.5 * (left + right)
    cy = 0.5 * (top + bottom)
    half_width = 0.5 * (right - left)
    half_height = 0.5 * (bottom - top)

    src_x = cx + src_x_proj / (2.0 * np.sqrt(2.0)) * half_width
    src_y = cy - src_y_proj / max(y_limit_value, 1e-9) * half_height

    valid = (
        np.isfinite(src_x)
        & np.isfinite(src_y)
        & (src_x >= 0)
        & (src_x <= page_gray.shape[1] - 1)
        & (src_y >= 0)
        & (src_y <= page_gray.shape[0] - 1)
        & (np.abs(lat_mesh) <= float(source_lat_limit_deg))
    )
    sampled = map_coordinates(page_gray, [src_y, src_x], order=1, mode="constant", cval=1.0)
    ink = np.clip((0.94 - sampled) / 0.78, 0.0, 1.0)
    ink[ink < 0.07] = 0.0
    ink = np.where(valid, ink**0.75, np.nan)
    return lon_mesh, lat_mesh, ink, valid


def _add_haslam_axes(ax: plt.Axes) -> None:
    ax.grid(color="0.35", alpha=0.18, linewidth=0.55)
    ticks = np.array([-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150])
    ax.set_xticks(np.deg2rad(ticks))
    ax.set_xticklabels([f"{int((-tick) % 360)}°" for tick in ticks], fontsize=7)
    ax.set_yticks(np.deg2rad([-60, -30, 0, 30, 60]))
    ax.set_yticklabels(["-60°", "-30°", "0°", "+30°", "+60°"], fontsize=7)
    ax.text(0.5, -0.08, "Galactic longitude", transform=ax.transAxes, ha="center", va="top", fontsize=8)


def _plot_one(
    ax: plt.Axes,
    lon_mesh: np.ndarray,
    lat_mesh: np.ndarray,
    ink: np.ndarray,
    spec: MapSpec,
    title_prefix: str = "",
) -> None:
    x = np.deg2rad(lon_mesh)
    y = np.deg2rad(lat_mesh)
    # Background is zero/white; the scanned published linework is positive
    # relative ink in the same diverging, zero-centered visual convention as
    # the pipeline maps. This is intentionally not a temperature color scale.
    image = ax.pcolormesh(x, y, ink, shading="auto", cmap="RdBu_r", vmin=-1.0, vmax=1.0, rasterized=True)
    ax.contour(x, y, np.where(np.isfinite(ink), ink, np.nan), levels=[0.18], colors="0.10", linewidths=0.22, alpha=0.55)
    _add_haslam_axes(ax)
    ax.set_title(f"{title_prefix}{spec.frequency_mhz:.2f} MHz; published contours in {spec.unit_label}", fontsize=9.2)
    return image


def _plot_panel(reprojected: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MapSpec]], out_dir: Path) -> Path:
    order = [9.18, 6.55, 4.70, 3.93, 2.20, 1.31]
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 7.2), subplot_kw={"projection": "mollweide"})
    image = None
    for ax, freq in zip(axes.ravel(), order):
        lon, lat, ink, _, spec = reprojected[freq]
        image = _plot_one(ax, lon, lat, ink, spec)
    fig.suptitle(
        "Novaco & Brown published RAE-2 contour maps, colorized and recentered to l=0\n"
        "Color is scanned contour ink density; contour labels retain the published temperature units.",
        y=0.995,
        fontsize=13,
    )
    if image is not None:
        cbar = fig.colorbar(image, ax=axes.ravel().tolist(), orientation="horizontal", fraction=0.035, pad=0.08)
        cbar.set_ticks([-1.0, 0.0, 1.0])
        cbar.set_ticklabels(["", "0", "+ink"])
        cbar.set_label("zero-centered published figure linework density (not digitized brightness temperature)")
    fig.subplots_adjust(left=0.035, right=0.985, top=0.87, bottom=0.12, wspace=0.08, hspace=0.28)
    path = out_dir / "novaco_brown_published_colorized_l0_centered_panel.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def _plot_individual(
    reprojected: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MapSpec]],
    out_dir: Path,
) -> list[Path]:
    paths: list[Path] = []
    for freq in sorted(reprojected, reverse=True):
        lon, lat, ink, _, spec = reprojected[freq]
        fig = plt.figure(figsize=(12.4, 6.8))
        ax = fig.add_subplot(111, projection="mollweide")
        image = _plot_one(ax, lon, lat, ink, spec, title_prefix="Novaco & Brown, ")
        cbar = fig.colorbar(image, ax=ax, orientation="horizontal", fraction=0.055, pad=0.09)
        cbar.set_ticks([-1.0, 0.0, 1.0])
        cbar.set_ticklabels(["", "0", "+ink"])
        cbar.set_label("zero-centered published figure linework density (not digitized brightness temperature)")
        fig.subplots_adjust(left=0.03, right=0.98, top=0.91, bottom=0.13)
        label = f"{freq:.2f}".replace(".", "p")
        path = out_dir / f"novaco_brown_published_{label}mhz_colorized_l0_centered.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        paths.append(path)
    return paths


def _write_report(out_dir: Path, config: dict[str, object], paths: list[Path]) -> Path:
    lines = [
        "# Novaco & Brown Published Maps: Colorized l=0-Centered Views",
        "",
        "## What This Product Is",
        "",
        "This product starts from the ADS scanned journal PDF of Novaco & Brown (1978), extracts the six published RAE-2 contour-map figures, and reprojects the scanned contour linework into the same Galactic Mollweide convention used by the local pipeline maps: Galactic longitude `l=0 deg` is centered and longitude increases to the left.",
        "",
        "The output is a colorized scan reprojection, not a numerical digitization of the published brightness-temperature field. Color encodes printed contour-line ink density so the contour map is readable in the same plotting layout; the printed contour labels remain the source of the temperature levels.",
        "",
        "## Frequencies",
        "",
        ", ".join(f"{spec.frequency_mhz:.2f} MHz" for spec in MAP_SPECS),
        "",
        "## Important Limitations",
        "",
        "- The published maps are contour figures, not machine-readable FITS or CSV maps.",
        "- The reprojection assumes the source figures are Mollweide-like, centered at approximately `l=100 deg`, with visible latitude coverage to about `|b|=80 deg`.",
        "- The regions outside the source scanned map coverage are left blank by construction.",
        "- The color scale is not calibrated sky temperature; use the printed contour labels for the original brightness-temperature values.",
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
            f"- ADS scanned article PDF: {ADS_PDF_URL}",
            f"- NASA NTRS record: {NTRS_RECORD_URL}",
            f"- NASA NTRS preprint PDF: {NTRS_PDF_URL}",
            "",
            "## Configuration",
            "",
            "\n".join(f"{key}: {value}" for key, value in config.items()),
            "",
        ]
    )
    path = out_dir / "novaco_brown_published_colorized_map_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run(args: argparse.Namespace) -> Path:
    out_dir = ensure_dir(args.out_dir)
    source_dir = ensure_dir(out_dir / "source/ads")
    render_dir = ensure_dir(out_dir / "source/ads_rendered_pages")
    pdf_path = Path(args.pdf) if args.pdf else source_dir / "1978ApJ_221_114N.pdf"
    _download_pdf(pdf_path, bool(args.refresh))
    _render_pdf_pages(pdf_path, render_dir, int(args.dpi), bool(args.refresh))

    pages = {page: _load_grayscale(_page_output_path(render_dir, page)) for page in (5, 6, 7)}
    reprojected: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MapSpec]] = {}
    for spec in MAP_SPECS:
        lon, lat, ink, valid = _reproject_ink(
            pages[spec.pdf_page],
            spec,
            float(args.lon_step_deg),
            float(args.lat_step_deg),
            float(args.source_center_l_deg),
            float(args.source_lat_limit_deg),
        )
        reprojected[spec.frequency_mhz] = (lon, lat, ink, valid, spec)

    paths: list[Path] = []
    paths.append(_plot_panel(reprojected, out_dir))
    paths.extend(_plot_individual(reprojected, out_dir))

    config = {
        "ads_pdf_url": ADS_PDF_URL,
        "pdf_path": str(pdf_path),
        "out_dir": str(out_dir),
        "dpi": int(args.dpi),
        "lon_step_deg": float(args.lon_step_deg),
        "lat_step_deg": float(args.lat_step_deg),
        "source_center_l_deg": float(args.source_center_l_deg),
        "source_lat_limit_deg": float(args.source_lat_limit_deg),
        "frequencies_mhz": [spec.frequency_mhz for spec in MAP_SPECS],
        "software_versions": software_versions(),
    }
    write_json(out_dir / "run_config.json", config)
    report = _write_report(out_dir, config, paths)
    print(report)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--pdf", default=str(DEFAULT_PDF))
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--dpi", type=int, default=260)
    parser.add_argument("--lon-step-deg", type=float, default=0.5)
    parser.add_argument("--lat-step-deg", type=float, default=0.5)
    parser.add_argument("--source-center-l-deg", type=float, default=SOURCE_CENTER_L_DEG)
    parser.add_argument("--source-lat-limit-deg", type=float, default=SOURCE_LAT_LIMIT_DEG)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
