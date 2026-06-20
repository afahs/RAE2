#!/usr/bin/env python
"""Build all-frequency normalized grids for custom fixed FK4 sources.

This script is intended for interactive-node use.  It predicts lunar
occultation events for a small source table and then reuses the existing
all-frequency profile-grid implementation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
from rylevonberg.table_io import read_table

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.events import predict_events  # noqa: E402
from rylevonberg.util import ensure_dir, software_versions, write_json  # noqa: E402

from scripts.build_all_frequency_occultation_profile_grids import (  # noqa: E402
    CLEAN,
    collect_profiles,
    plot_grid,
    summarize_profiles,
)


DEFAULT_SOURCES = [
    {
        "source_name": "3c_295",
        "kind": "fixed",
        "body_name": "",
        "ra_deg": 212.38958600312114,
        "dec_deg": 52.43697551841423,
        "frame": "fk4",
        "notes": "3C 295; J2000 position transformed to FK4/B1950 for RAE-2 pipeline",
    },
    {
        "source_name": "3c_273",
        "kind": "fixed",
        "body_name": "",
        "ra_deg": 186.63855070362368,
        "dec_deg": 2.3287299569472175,
        "frame": "fk4",
        "notes": "3C 273; J2000 position transformed to FK4/B1950 for RAE-2 pipeline",
    },
    {
        "source_name": "vir_a",
        "kind": "fixed",
        "body_name": "",
        "ra_deg": 187.70593,
        "dec_deg": 12.39112,
        "frame": "fk4",
        "notes": "Virgo A / M87; existing RyleVonberg FK4 bright-source coordinate",
    },
    {
        "source_name": "tau_a",
        "kind": "fixed",
        "body_name": "",
        "ra_deg": 83.63308,
        "dec_deg": 22.01450,
        "frame": "fk4",
        "notes": "Tau A / Crab Nebula; existing RyleVonberg FK4 bright-source coordinate",
    },
]


def _parse_sources(names: str) -> list[str]:
    return [x.strip().lower() for x in str(names).split(",") if x.strip()]


def _write_report(
    out_dir: Path,
    source_table: pd.DataFrame,
    events: pd.DataFrame,
    paths: list[Path],
    config: dict[str, object],
) -> None:
    event_counts = (
        events.groupby("source_name", sort=True)
        .agg(
            event_rows=("event_id", "size"),
            unique_event_ids=("event_id", "nunique"),
            first_event=("predicted_event_time", "min"),
            last_event=("predicted_event_time", "max"),
        )
        .reset_index()
        if not events.empty
        else pd.DataFrame()
    )
    lines = [
        "# Custom Fixed-Source All-Frequency Profile Grids",
        "",
        "These outputs were generated on the current interactive session.  No batch queue submission was used.",
        "",
        "## Sources",
        "",
        source_table.to_string(index=False),
        "",
        "## Prediction / Plot Configuration",
        "",
        pd.Series(config).to_string(),
        "",
        "## Event Counts",
        "",
        event_counts.to_string(index=False) if not event_counts.empty else "No events predicted.",
        "",
        "## How To Read The Plots",
        "",
        "- x-axis: seconds from predicted occultation event;",
        "- y-axis: locally normalized raw power, with no trendline removal;",
        "- rows: frequency channels;",
        "- columns: disappearance and reappearance;",
        "- blue: upper V / `rv1_coarse`; orange: lower V / `rv2_coarse`;",
        "- error bars: robust event-to-event standard error per time bin.",
        "",
        "Generated plots:",
        "",
    ]
    lines.extend(f"- `{path.name}`" for path in paths)
    (out_dir / "custom_fixed_source_profile_grid_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(ROOT / "outputs/custom_fixed_source_profile_grids_3c295_3c273_virgoa_v1"))
    parser.add_argument("--sources", default="3c_295,3c_273,vir_a")
    parser.add_argument("--window-s", type=float, default=900.0)
    parser.add_argument("--bin-s", type=float, default=60.0)
    parser.add_argument("--inner-s", type=float, default=15.0)
    parser.add_argument("--prediction-cadence-seconds", type=float, default=300.0)
    parser.add_argument("--max-gap-seconds", type=float, default=600.0)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    requested = set(_parse_sources(args.sources))
    source_table = pd.DataFrame(DEFAULT_SOURCES)
    source_table = source_table[source_table["source_name"].isin(requested)].copy()
    if source_table.empty:
        raise SystemExit("no requested sources found in DEFAULT_SOURCES")
    missing = sorted(requested - set(source_table["source_name"]))
    if missing:
        raise SystemExit(f"unknown custom source(s): {', '.join(missing)}")

    frequencies = list(range(1, 10))
    antennas = ["rv1_coarse", "rv2_coarse"]
    config = {
        "sources": sorted(requested),
        "target_frame": "fk4",
        "equinox": "B1950",
        "ephemeris": "builtin",
        "frequencies": frequencies,
        "antennas": antennas,
        "window_s": float(args.window_s),
        "bin_s": float(args.bin_s),
        "inner_s": float(args.inner_s),
        "prediction_cadence_seconds": float(args.prediction_cadence_seconds),
        "max_gap_seconds": float(args.max_gap_seconds),
        "run_mode": "interactive",
        "software_versions": software_versions(),
    }
    write_json(out_dir / "run_config.json", config)
    source_table.to_csv(out_dir / "custom_fixed_source_list.csv", index=False)

    clean = read_table(CLEAN, parse_dates=["time"], low_memory=False)
    events, states = predict_events(
        clean,
        source_table,
        target_frame="fk4",
        equinox="B1950",
        ephemeris="builtin",
        max_gap_seconds=float(args.max_gap_seconds),
        prediction_cadence_seconds=float(args.prediction_cadence_seconds),
        frequencies=frequencies,
        antennas=antennas,
    )
    events.to_csv(out_dir / "custom_fixed_source_predicted_events.csv", index=False)
    states.to_csv(out_dir / "custom_fixed_source_limb_visibility_states.csv", index=False)

    paths: list[Path] = []
    for source in sorted(requested):
        source_events = events[events["source_name"].astype(str).str.lower().eq(source)].copy()
        points = collect_profiles(clean, source_events, source, args.window_s, args.bin_s, args.inner_s)
        points.to_csv(out_dir / f"{source}_all_frequency_profile_points_{int(args.window_s)}s.csv", index=False)
        summary = summarize_profiles(points)
        summary.to_csv(out_dir / f"{source}_all_frequency_profile_summary_{int(args.window_s)}s.csv", index=False)
        if not summary.empty:
            path = plot_grid(summary, source, args.window_s, out_dir)
            paths.append(path)
            print(path)

    _write_report(out_dir, source_table, events, paths, config)
    print(out_dir / "custom_fixed_source_profile_grid_report.md")


if __name__ == "__main__":
    main()
