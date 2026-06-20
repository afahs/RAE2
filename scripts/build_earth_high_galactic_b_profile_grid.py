#!/usr/bin/env python
"""Build an Earth all-frequency profile grid using only high Galactic-latitude Earth events."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import astropy.units as u
from astropy.coordinates import CartesianRepresentation, FK4, SkyCoord
from astropy.time import Time
import numpy as np
import pandas as pd
from rylevonberg.table_io import read_table

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.constants import EARTH_UNIT_COLUMNS, SPACECRAFT_COLUMNS  # noqa: E402
from rylevonberg.geometry import earth_direction_from_spacecraft  # noqa: E402
from rylevonberg.util import ensure_dir, software_versions, write_json  # noqa: E402

from scripts.build_all_frequency_occultation_profile_grids import (  # noqa: E402
    CLEAN,
    EARTH_EVENTS,
    _read,
    collect_profiles,
    plot_grid,
    summarize_profiles,
)


def _earth_galactic_b_for_events(events: pd.DataFrame, clean_path: Path) -> pd.DataFrame:
    """Attach apparent Earth Galactic coordinates at each predicted event time.

    The event table has one row per event/frequency/antenna. The apparent sky
    position only depends on event time, so it is computed once per unique event
    and then merged back onto the full event table.
    """
    unique = (
        events[["event_id", "predicted_event_time"]]
        .drop_duplicates("event_id")
        .sort_values("predicted_event_time")
        .reset_index(drop=True)
    )
    unique["predicted_event_time"] = pd.to_datetime(unique["predicted_event_time"], errors="coerce").astype("datetime64[ns]")
    usecols = ["time", *SPACECRAFT_COLUMNS, *EARTH_UNIT_COLUMNS]
    geom = read_table(clean_path, usecols=usecols, parse_dates=["time"], low_memory=False)
    geom["time"] = pd.to_datetime(geom["time"], errors="coerce").astype("datetime64[ns]")
    geom = geom.dropna(subset=["time"]).drop_duplicates("time").sort_values("time").reset_index(drop=True)
    matched = pd.merge_asof(
        unique,
        geom,
        left_on="predicted_event_time",
        right_on="time",
        direction="nearest",
        tolerance=pd.Timedelta(minutes=5),
    )
    missing = matched["time"].isna()
    if missing.any():
        raise RuntimeError(f"Could not match {int(missing.sum())} Earth events to cleaned geometry within 5 minutes")

    earth_vec = earth_direction_from_spacecraft(
        matched[SPACECRAFT_COLUMNS].to_numpy(dtype=float),
        matched[EARTH_UNIT_COLUMNS].to_numpy(dtype=float),
    )
    rep = CartesianRepresentation(earth_vec[:, 0] * u.one, earth_vec[:, 1] * u.one, earth_vec[:, 2] * u.one)
    gal = SkyCoord(rep, frame=FK4(equinox=Time("B1950"))).galactic
    matched["earth_galactic_l_deg"] = np.asarray(gal.l.deg, dtype=float)
    matched["earth_galactic_b_deg"] = np.asarray(gal.b.deg, dtype=float)
    matched["geometry_match_dt_s"] = (
        matched["time"].astype("datetime64[ns]") - matched["predicted_event_time"].astype("datetime64[ns]")
    ).dt.total_seconds()

    return events.merge(
        matched[["event_id", "earth_galactic_l_deg", "earth_galactic_b_deg", "geometry_match_dt_s"]],
        on="event_id",
        how="left",
    )


def _write_report(
    out_dir: Path,
    plot_path: Path | None,
    events_with_b: pd.DataFrame,
    filtered_events: pd.DataFrame,
    window_s: float,
    bin_s: float,
    min_galactic_b_deg: float,
) -> None:
    event_level = events_with_b.drop_duplicates("event_id")
    kept_level = filtered_events.drop_duplicates("event_id")
    by_type = (
        events_with_b.drop_duplicates("event_id")
        .groupby("event_type", dropna=False)
        .agg(n_total=("event_id", "size"))
        .join(
            filtered_events.drop_duplicates("event_id")
            .groupby("event_type", dropna=False)
            .agg(n_kept=("event_id", "size")),
            how="left",
        )
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    lines = [
        "# Earth High-Galactic-Latitude Profile Grid",
        "",
        f"Selection: Earth apparent Galactic latitude `b >= {min_galactic_b_deg:.1f} deg` at the predicted occultation time.",
        "",
        "What changed relative to the normal Earth all-frequency profile grid:",
        "",
        "- the same cleaned Ryle-Vonberg time series was used;",
        "- the same all-frequency normalized raw-power profile method was used;",
        "- each event was assigned the apparent spacecraft-to-Earth Galactic latitude using the cleaned geometry;",
        f"- only event IDs with `earth_galactic_b_deg >= {min_galactic_b_deg:.1f}` were retained;",
        "- frequency/antenna rows for retained event IDs were then stacked exactly as in the usual grid.",
        "",
        "This is intended to test whether the low-frequency Earth profile morphology changes when Earth is well above the Galactic plane.",
        "",
        "## Event Counts",
        "",
        f"- total unique Earth events before Galactic-b filter: {len(event_level)}",
        f"- retained unique Earth events after Galactic-b filter: {len(kept_level)}",
        f"- retained event/frequency/antenna rows: {len(filtered_events)}",
        "",
        "By event type:",
        "",
        by_type.to_string(index=False),
        "",
        "## Geometry Match",
        "",
        f"- max absolute nearest-geometry match offset: {event_level['geometry_match_dt_s'].abs().max():.3f} s",
        f"- median absolute nearest-geometry match offset: {event_level['geometry_match_dt_s'].abs().median():.3f} s",
        "",
        "## Plot",
        "",
        f"- `{plot_path.name}`" if plot_path is not None else "- No plot generated; no profile summary rows survived.",
        "",
        f"Window: {window_s:.0f} s. Bin size: {bin_s:.0f} s.",
    ]
    (out_dir / "earth_high_galactic_b_profile_grid_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(ROOT / "outputs/earth_high_galactic_b_profile_grid_v1"))
    parser.add_argument("--min-galactic-b-deg", type=float, default=30.0)
    parser.add_argument("--window-s", type=float, default=900.0)
    parser.add_argument("--bin-s", type=float, default=60.0)
    parser.add_argument("--inner-s", type=float, default=15.0)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    write_json(
        out_dir / "run_config.json",
        {
            "source_name": "earth",
            "min_galactic_b_deg": float(args.min_galactic_b_deg),
            "window_s": float(args.window_s),
            "bin_s": float(args.bin_s),
            "inner_s": float(args.inner_s),
            "event_table": str(EARTH_EVENTS.relative_to(ROOT)),
            "cleaned_timeseries": str(CLEAN.relative_to(ROOT)),
            "software_versions": software_versions(),
        },
    )

    clean = _read(CLEAN, parse_dates=["time"])
    events = _read(EARTH_EVENTS, parse_dates=["predicted_event_time"])
    events = events[events["source_name"].astype(str).str.lower().eq("earth")].copy()
    events_with_b = _earth_galactic_b_for_events(events, CLEAN)
    filtered_events = events_with_b[pd.to_numeric(events_with_b["earth_galactic_b_deg"], errors="coerce").ge(args.min_galactic_b_deg)].copy()

    events_with_b.to_csv(out_dir / "earth_events_with_apparent_galactic_b.csv", index=False)
    filtered_events.to_csv(out_dir / "earth_events_b_ge_30deg.csv", index=False)

    points = collect_profiles(clean, filtered_events, "earth", args.window_s, args.bin_s, args.inner_s)
    points.to_csv(out_dir / f"earth_b_ge_30_all_frequency_profile_points_{int(args.window_s)}s.csv", index=False)
    summary = summarize_profiles(points)
    summary.to_csv(out_dir / f"earth_b_ge_30_all_frequency_profile_summary_{int(args.window_s)}s.csv", index=False)

    plot_path = None
    if not summary.empty:
        raw_plot = plot_grid(summary, "earth_b_ge_30", args.window_s, out_dir)
        plot_path = out_dir / f"earth_b_ge_30_all_frequency_profile_grid_{int(args.window_s)}s.png"
        raw_plot.rename(plot_path)
        print(plot_path)

    _write_report(out_dir, plot_path, events_with_b, filtered_events, args.window_s, args.bin_s, args.min_galactic_b_deg)


if __name__ == "__main__":
    main()
