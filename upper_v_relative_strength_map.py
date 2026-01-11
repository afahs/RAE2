#!/usr/bin/env python3
"""Build 50x50 Mollweide relative-strength maps from RAE2 Upper V data.

This standalone script was requested so the large interpolated RAE2 CSV can be
reduced into sky maps per frequency band using the Upper V (rv1_coarse) channel.
Given the 3 GB source table we stream it in chunks, bin entries in longitude/
latitude, iteratively 4σ-clip each bin until convergence (per user instruction),
and then visualise the resulting median and standard deviation on Mollweide
projections.
"""

from __future__ import annotations

import argparse
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

# Keep plots working on headless interactive nodes.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Instruction reference: use the shared interpolated CSV unless a different path is passed.
DEFAULT_DATA_PATH = Path("/global/cfs/projectdirs/m4895/RAE2Data/interpolatedRAE2MasterFile.csv")

# Only read the heavy CSV columns needed for Upper V sky maps (instruction update).
UPPER_V_COLUMN = "rv1_coarse"  # clarified by user that rv1_coarse is Upper V.
USECOLS = ["frequency_band", "right_ascension", "declination", UPPER_V_COLUMN]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Mollweide relative-strength maps from RAE2 Upper V (rv1_coarse) data."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the interpolated RAE2 CSV (defaults to the shared project file).",
    )
    script_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "upper_v_maps",
        help="Directory where figures and bin statistics will be written.",
    )
    parser.add_argument(
        "--ra-bins",
        type=int,
        default=50,
        help="Number of longitude bins (default 50 as requested).",
    )
    parser.add_argument(
        "--dec-bins",
        type=int,
        default=50,
        help="Number of latitude bins (default 50 as requested).",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=4.0,
        help="Sigma threshold for iterative clipping (instruction: 4σ).",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=10,
        help="Maximum sigma-clipping passes per bin (ensure convergence).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500_000,
        help="Rows per chunk when streaming the CSV (tweak for memory/performance).",
    )
    parser.add_argument(
        "--frequencies",
        type=int,
        nargs="*",
        help="Optional list of frequency_band integers to process; defaults to all bands present.",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=5,
        help="Minimum number of clipped samples required to keep a bin (avoid noisy outliers).",
    )
    parser.add_argument(
        "--log-scale",
        action="store_true",
        help="Plot colour maps on a log10 scale (useful for wide dynamic range).",
    )
    return parser.parse_args(argv)


def discover_frequencies(csv_path: Path, chunk_size: int) -> List[int]:
    logging.info("Scanning %s for available frequency bands...", csv_path)
    freqs: set[int] = set()
    for chunk in pd.read_csv(csv_path, usecols=["frequency_band"], chunksize=chunk_size):
        freqs.update(int(band) for band in chunk["frequency_band"].dropna().unique())
    discovered = sorted(freqs)
    logging.info("Found frequency bands: %s", discovered)
    return discovered


def iterative_sigma_clip(
    values: np.ndarray,
    sigma: float,
    max_iters: int,
) -> np.ndarray:
    """Clip outliers iteratively until every point lies within sigma*std of the median.

    Instruction link: repeat 4σ clipping until convergence on each bin.
    """
    clipped = values[np.isfinite(values)]
    if clipped.size == 0:
        return clipped

    for iteration in range(max_iters):
        median = np.median(clipped)
        std = np.std(clipped)
        if std == 0 or np.isnan(std):
            break
        distance = np.abs(clipped - median)
        mask = distance <= sigma * std
        if mask.all():
            break
        clipped = clipped[mask]
        if clipped.size == 0:
            break
    return clipped


def collect_bin_values(
    csv_path: Path,
    frequency: int,
    ra_edges: np.ndarray,
    dec_edges: np.ndarray,
    chunk_size: int,
) -> Dict[int, List[float]]:
    """Stream the CSV and collect rv1_coarse samples into 2D sky bins for one frequency band."""
    n_ra = len(ra_edges) - 1
    n_dec = len(dec_edges) - 1
    bins: Dict[int, List[float]] = defaultdict(list)

    reader = pd.read_csv(csv_path, usecols=USECOLS, chunksize=chunk_size)
    total_rows = 0
    kept_rows = 0

    for chunk in reader:
        total_rows += len(chunk)
        freq_chunk = chunk[chunk["frequency_band"] == frequency]
        if freq_chunk.empty:
            continue

        freq_chunk = freq_chunk.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["right_ascension", "declination", UPPER_V_COLUMN]
        )
        if freq_chunk.empty:
            continue

        ra_deg = (freq_chunk["right_ascension"].to_numpy(dtype=float) * 15.0) % 360.0
        dec_deg = freq_chunk["declination"].to_numpy(dtype=float)
        values = freq_chunk[UPPER_V_COLUMN].to_numpy(dtype=float)

        valid = np.isfinite(ra_deg) & np.isfinite(dec_deg) & np.isfinite(values)
        valid &= (dec_deg >= dec_edges[0]) & (dec_deg <= dec_edges[-1])
        if not np.any(valid):
            continue

        ra_valid = ra_deg[valid]
        dec_valid = dec_deg[valid]
        vals_valid = values[valid]

        ra_idx = np.digitize(ra_valid, ra_edges, right=False) - 1
        dec_idx = np.digitize(dec_valid, dec_edges, right=False) - 1

        inside = (ra_idx >= 0) & (ra_idx < n_ra) & (dec_idx >= 0) & (dec_idx < n_dec)
        if not np.any(inside):
            continue

        ra_idx = ra_idx[inside]
        dec_idx = dec_idx[inside]
        vals_valid = vals_valid[inside]

        bin_ids = (dec_idx * n_ra) + ra_idx
        order = np.argsort(bin_ids, kind="mergesort")
        bin_ids = bin_ids[order]
        vals_sorted = vals_valid[order]

        split_idx = np.flatnonzero(np.diff(bin_ids)) + 1
        starts = np.concatenate(([0], split_idx))
        ends = np.concatenate((split_idx, [len(bin_ids)]))

        for start, end in zip(starts, ends):
            bin_id = int(bin_ids[start])
            bins[bin_id].extend(vals_sorted[start:end].tolist())

        kept_rows += int(vals_valid.size)

    logging.info(
        "Frequency %s: scanned %d rows, retained %d Upper V samples inside the sky grid.",
        frequency,
        total_rows,
        kept_rows,
    )
    return bins


def compute_bin_statistics(
    raw_bins: Dict[int, List[float]],
    n_dec: int,
    n_ra: int,
    sigma: float,
    max_iters: int,
    min_count: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply iterative sigma clipping and return median/std/count grids."""
    median_grid = np.full((n_dec, n_ra), np.nan, dtype=float)
    std_grid = np.full_like(median_grid, np.nan)
    count_grid = np.zeros_like(median_grid, dtype=int)
    clipped_count_grid = np.zeros_like(median_grid, dtype=int)

    for bin_id, samples in raw_bins.items():
        if not samples:
            continue
        dec_idx, ra_idx = divmod(bin_id, n_ra)
        data = np.asarray(samples, dtype=float)
        data = data[np.isfinite(data)]
        if data.size == 0:
            continue
        clipped = iterative_sigma_clip(data, sigma=sigma, max_iters=max_iters)
        if clipped.size < min_count:
            continue
        median_grid[dec_idx, ra_idx] = float(np.median(clipped))
        std_grid[dec_idx, ra_idx] = float(np.std(clipped))
        count_grid[dec_idx, ra_idx] = int(data.size)
        clipped_count_grid[dec_idx, ra_idx] = int(clipped.size)

    return median_grid, std_grid, count_grid, clipped_count_grid


def build_longitude_latitude_edges(n_ra: int, n_dec: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create longitude/latitude bin edges in degrees."""
    ra_edges = np.linspace(0.0, 360.0, num=n_ra + 1)
    dec_edges = np.linspace(-90.0, 90.0, num=n_dec + 1)
    return ra_edges, dec_edges


def normalise_to_relative(median_grid: np.ndarray) -> np.ndarray:
    """Convert medians into a relative strength map."""
    reference = np.nanmedian(median_grid)
    if not math.isfinite(reference) or reference == 0:
        return np.full_like(median_grid, np.nan)
    return median_grid / reference


def make_mollweide_mesh(ra_edges: np.ndarray, dec_edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert spherical edges to Mollweide-ready radian grids."""
    lon = np.deg2rad(ra_edges) - np.pi  # Shift so RA=180° is centred.
    lat = np.deg2rad(dec_edges)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    return lon_grid, lat_grid


def render_frequency_maps(
    frequency: int,
    median_grid: np.ndarray,
    std_grid: np.ndarray,
    ra_edges: np.ndarray,
    dec_edges: np.ndarray,
    output_dir: Path,
    use_log_scale: bool,
) -> None:
    """Plot relative strength and sigma grids on Mollweide projections."""
    lon_grid, lat_grid = make_mollweide_mesh(ra_edges, dec_edges)
    relative_grid = normalise_to_relative(median_grid)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6),
        subplot_kw={"projection": "mollweide"},
        constrained_layout=True,
    )

    cmap_strength = plt.get_cmap("viridis")
    cmap_std = plt.get_cmap("magma")

    if use_log_scale:
        with np.errstate(invalid="ignore"):
            strength_data = np.log10(relative_grid)
        strength_label = "log10(relative median Upper V)"
    else:
        strength_data = relative_grid
        strength_label = "Relative median Upper V"

    # Clip colour scaling to reduce outlier dominance where possible.
    finite_strength = strength_data[np.isfinite(strength_data)]
    if finite_strength.size:
        vmin_strength = np.nanpercentile(finite_strength, 5)
        vmax_strength = np.nanpercentile(finite_strength, 95)
    else:
        vmin_strength = vmax_strength = None

    mesh_strength = axes[0].pcolormesh(
        lon_grid,
        lat_grid,
        strength_data,
        cmap=cmap_strength,
        shading="auto",
        vmin=vmin_strength,
        vmax=vmax_strength,
    )
    cbar_strength = fig.colorbar(mesh_strength, ax=axes[0], orientation="horizontal", pad=0.05)
    cbar_strength.set_label(strength_label)
    axes[0].set_title(f"Band {frequency} relative Upper V median")
    axes[0].grid(True, linestyle=":", alpha=0.5)

    if use_log_scale:
        with np.errstate(invalid="ignore"):
            std_data = np.log10(std_grid)
        std_label = "log10(std Upper V)"
    else:
        std_data = std_grid
        std_label = "Std Upper V"

    finite_std = std_data[np.isfinite(std_data)]
    if finite_std.size:
        vmin_std = np.nanpercentile(finite_std, 5)
        vmax_std = np.nanpercentile(finite_std, 95)
    else:
        vmin_std = vmax_std = None

    mesh_std = axes[1].pcolormesh(
        lon_grid,
        lat_grid,
        std_data,
        cmap=cmap_std,
        shading="auto",
        vmin=vmin_std,
        vmax=vmax_std,
    )
    cbar_std = fig.colorbar(mesh_std, ax=axes[1], orientation="horizontal", pad=0.05)
    cbar_std.set_label(std_label)
    axes[1].set_title(f"Band {frequency} Upper V dispersion")
    axes[1].grid(True, linestyle=":", alpha=0.5)

    tick_deg = np.arange(-150, 181, 30)
    tick_labels = [f"{int((deg + 180) % 360)}°" for deg in tick_deg]
    tick_positions = np.deg2rad(tick_deg)
    axes[0].set_xticks(tick_positions)
    axes[0].set_xticklabels(tick_labels, fontsize=9)
    axes[1].set_xticks(tick_positions)
    axes[1].set_xticklabels(tick_labels, fontsize=9)

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / f"band_{frequency:02d}_upper_v_mollweide.png"
    fig.suptitle(f"RAE2 Upper V relative-strength summary (band {frequency})", fontsize=14)
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)
    logging.info("Saved Mollweide map for band %s to %s", frequency, figure_path)


def save_statistics(
    frequency: int,
    median_grid: np.ndarray,
    std_grid: np.ndarray,
    count_grid: np.ndarray,
    clipped_count_grid: np.ndarray,
    output_dir: Path,
) -> None:
    """Persist per-bin statistics for downstream analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / f"band_{frequency:02d}_upper_v_stats.npz"
    np.savez_compressed(
        npz_path,
        median=median_grid,
        std=std_grid,
        raw_counts=count_grid,
        clipped_counts=clipped_count_grid,
    )
    logging.info("Wrote bin statistics for band %s to %s", frequency, npz_path)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    data_path = args.data.expanduser()
    if not data_path.exists():
        raise FileNotFoundError(f"CSV not found: {data_path}")

    ra_edges, dec_edges = build_longitude_latitude_edges(args.ra_bins, args.dec_bins)

    if args.frequencies:
        frequencies = sorted(set(args.frequencies))
    else:
        frequencies = discover_frequencies(data_path, args.chunk_size)

    if not frequencies:
        raise RuntimeError("No frequency bands available to process.")

    for frequency in frequencies:
        logging.info("Processing frequency band %s...", frequency)
        bin_samples = collect_bin_values(
            csv_path=data_path,
            frequency=frequency,
            ra_edges=ra_edges,
            dec_edges=dec_edges,
            chunk_size=args.chunk_size,
        )
        median_grid, std_grid, count_grid, clipped_count_grid = compute_bin_statistics(
            raw_bins=bin_samples,
            n_dec=args.dec_bins,
            n_ra=args.ra_bins,
            sigma=args.sigma,
            max_iters=args.max_iters,
            min_count=args.min_count,
        )
        save_statistics(
            frequency=frequency,
            median_grid=median_grid,
            std_grid=std_grid,
            count_grid=count_grid,
            clipped_count_grid=clipped_count_grid,
            output_dir=args.output_dir,
        )
        render_frequency_maps(
            frequency=frequency,
            median_grid=median_grid,
            std_grid=std_grid,
            ra_edges=ra_edges,
            dec_edges=dec_edges,
            output_dir=args.output_dir,
            use_log_scale=args.log_scale,
        )

    logging.info("Finished building Upper V relative-strength maps.")


if __name__ == "__main__":
    main()
