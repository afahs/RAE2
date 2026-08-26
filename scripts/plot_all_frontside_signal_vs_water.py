#!/usr/bin/env python
"""Plot signal versus visible water for every Earth-facing RAE-2 orbit pass.

Unlike the near-line analysis, this includes a pass whenever RAE-2 is in the
Earth-facing lunar half-space (spacecraft dot Moon-to-Earth direction > 0).
Telemetry-gap fragments less than one hour apart are reduced to the single
best-aligned point before pass-average powers are extracted.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.util import ensure_dir, software_versions, write_json  # noqa: E402
from scripts.investigate_earth_noise_geography import (  # noqa: E402
    build_surface_grid,
    load_land_geojson,
)
from scripts.plot_pass_signal_vs_visible_water import (  # noqa: E402
    attach_visible_water,
    build_signal_points,
    plot_signal_facets,
)
from scripts.run_naive_subspacecraft_land_ocean import (  # noqa: E402
    aggregate_pass_power,
    attach_subspacecraft_points,
    find_frontside_passes,
    prepare_pass_residuals,
)


DEFAULT_INPUT = Path(
    os.environ.get(
        "RAE2_MASTER_CSV",
        "data/interpolatedRAE2MasterFile.csv",
    )
)
DEFAULT_GEOGRAPHY_DIR = ROOT / "outputs/earth_naive_subspacecraft_land_ocean_v1"
DEFAULT_OUT_DIR = ROOT / "outputs/earth_visible_water_all_frontside_v1"


def merge_same_orbit_fragments(
    fragments: pd.DataFrame,
    maximum_center_gap_hours: float = 1.0,
) -> pd.DataFrame:
    """Merge short telemetry-gap fragments and retain one point per orbit.

    The measured orbit period is about 3.71 hours.  A one-hour center gap is
    therefore conservative: it removes obvious fragments without combining
    adjacent physical orbits.
    """
    if fragments.empty:
        return fragments.copy()
    work = fragments.sort_values("center_time").reset_index(drop=True).copy()
    center_gap_hours = work["center_time"].diff().dt.total_seconds() / 3600.0
    work["physical_pass_id"] = (
        center_gap_hours.isna()
        | (center_gap_hours >= float(maximum_center_gap_hours))
    ).cumsum()
    grouped = work.groupby("physical_pass_id", sort=True)
    summary = grouped.agg(
        start_time=("start_time", "min"),
        end_time=("end_time", "max"),
        n_geometry_rows=("n_geometry_rows", "sum"),
        observed_frontside_duration_s=("frontside_duration_s", "sum"),
        longest_fragment_duration_s=("frontside_duration_s", "max"),
        fragment_count=("pass_id", "size"),
    )
    best = work.loc[grouped["alignment_cosine"].idxmax()].set_index("physical_pass_id")
    best = best.drop(
        columns=[
            "pass_id",
            "start_time",
            "end_time",
            "n_geometry_rows",
            "frontside_duration_s",
        ]
    )
    out = summary.join(best).reset_index().rename(columns={"physical_pass_id": "pass_id"})
    out["frontside_span_s"] = (out["end_time"] - out["start_time"]).dt.total_seconds()
    # Downstream tables expect this legacy field; here it explicitly denotes
    # the sum of observed fragment durations, excluding telemetry gaps.
    out["frontside_duration_s"] = out["observed_frontside_duration_s"]
    return out.sort_values("center_time").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--geography-dir", type=Path, default=DEFAULT_GEOGRAPHY_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--chunksize", type=int, default=750_000)
    parser.add_argument("--maximum-telemetry-gap-seconds", type=float, default=600.0)
    parser.add_argument("--same-orbit-fragment-gap-hours", type=float, default=1.0)
    parser.add_argument("--minimum-observed-frontside-minutes", type=float, default=20.0)
    parser.add_argument("--window-minutes", type=float, default=5.0)
    parser.add_argument("--minimum-samples-per-pass-channel", type=int, default=5)
    parser.add_argument("--surface-grid-step-deg", type=float, default=1.0)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    geojson_path = args.geography_dir / "natural_earth_110m_admin0_countries.geojson"
    geojson = load_land_geojson(geojson_path)

    fragments = find_frontside_passes(
        args.input,
        args.chunksize,
        args.maximum_telemetry_gap_seconds,
    )
    passes = merge_same_orbit_fragments(
        fragments,
        args.same_orbit_fragment_gap_hours,
    )
    passes = passes[
        passes["observed_frontside_duration_s"]
        >= float(args.minimum_observed_frontside_minutes) * 60.0
    ].copy()
    # No close-alignment cut: every retained center is merely required to be
    # in the Earth-facing half-space, which find_frontside_passes guarantees.
    if not (passes["alignment_cosine"] > 0.0).all():
        raise RuntimeError("non-frontside point survived frontside selection")
    passes = attach_subspacecraft_points(passes, geojson)
    grid = build_surface_grid(geojson, step_deg=args.surface_grid_step_deg)
    passes = attach_visible_water(passes, grid)
    passes.to_csv(out_dir / "all_frontside_pass_points_with_water.csv", index=False)

    power = aggregate_pass_power(
        args.input,
        passes,
        args.chunksize,
        [args.window_minutes],
        90.0,
    )
    power = prepare_pass_residuals(power, args.minimum_samples_per_pass_channel)
    power.to_csv(out_dir / "all_frontside_pass_power_averages.csv", index=False)
    points = build_signal_points(power, passes, args.window_minutes, 90.0)
    points.to_csv(out_dir / "all_frontside_pass_frequency_signal_with_water.csv", index=False)

    plot_path = plot_signal_facets(
        points,
        "log10_average_signal",
        "log10 average signal",
        "All Earth-facing RAE-2 passes: signal versus apparent Earth-disk water",
        out_dir / "all_frontside_signal_vs_visible_water_by_frequency.png",
    )
    write_json(
        out_dir / "run_config.json",
        {
            **vars(args),
            "input": str(args.input),
            "geography_dir": str(args.geography_dir),
            "out_dir": str(args.out_dir),
            "selection": "spacecraft dot Moon-to-Earth direction > 0; no close-alignment cut",
            "n_raw_frontside_fragments": int(len(fragments)),
            "n_physical_passes_after_coverage_filter": int(len(passes)),
            "n_pass_frequency_points": int(len(points)),
            "software_versions": software_versions(),
        },
    )
    print(plot_path)


if __name__ == "__main__":
    main()
