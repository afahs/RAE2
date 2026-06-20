#!/usr/bin/env python
"""Predict fixed ecliptic controls with Sun/Earth lunar-limb vetoes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rylevonberg.table_io import read_table

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.events import predict_events  # noqa: E402
from rylevonberg.util import ensure_dir, software_versions, write_json  # noqa: E402
from scripts.audit_ecliptic_control_points import _source_table  # noqa: E402
from scripts.build_all_frequency_occultation_profile_grids import CLEAN  # noqa: E402


UNFILTERED_EVENTS = ROOT / "outputs/ecliptic_control_points_v1/ecliptic_control_predicted_events.csv"


def _read(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return read_table(path, low_memory=False, **kwargs)


def _limb_exclusion_sources() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_name": "earth",
                "kind": "earth",
                "body_name": "earth",
                "frame": "fk4",
                "ra_deg": np.nan,
                "dec_deg": np.nan,
            },
            {
                "source_name": "sun",
                "kind": "body",
                "body_name": "sun",
                "frame": "fk4",
                "ra_deg": np.nan,
                "dec_deg": np.nan,
            },
        ]
    )


def _count_table(events: pd.DataFrame, label: str) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    return (
        events.groupby(["control_class", "frequency_mhz", "event_type"], as_index=False)
        .agg(n_rows=("source_name", "size"), n_tracks=("source_name", "nunique"), n_event_ids=("event_id", "nunique"))
        .assign(run_label=label)
        .sort_values(["control_class", "frequency_mhz", "event_type"])
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(ROOT / "outputs/ecliptic_control_points_sun_earth_limb5_v1"))
    parser.add_argument("--limb-exclusion-deg", type=float, default=5.0)
    parser.add_argument("--prediction-cadence-seconds", type=float, default=300.0)
    parser.add_argument("--max-gap-seconds", type=float, default=600.0)
    parser.add_argument("--frequencies", default="1,2,3,4,5")
    parser.add_argument("--write-states", action="store_true")
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    sources = _source_table()
    frequencies = [int(x.strip()) for x in str(args.frequencies).split(",") if x.strip()]
    sources.to_csv(out_dir / "ecliptic_control_source_list.csv", index=False)
    write_json(
        out_dir / "run_config.json",
        {
            "limb_exclusion_deg": float(args.limb_exclusion_deg),
            "limb_exclusion_sources": ["earth", "sun"],
            "prediction_cadence_seconds": float(args.prediction_cadence_seconds),
            "max_gap_seconds": float(args.max_gap_seconds),
            "frequencies": frequencies,
            "antennas": ["rv2_coarse"],
            "write_states": bool(args.write_states),
            "software_versions": software_versions(),
        },
    )

    clean = _read(CLEAN, parse_dates=["time"])
    events, states = predict_events(
        clean,
        sources,
        target_frame="fk4",
        equinox="B1950",
        ephemeris="builtin",
        max_gap_seconds=float(args.max_gap_seconds),
        prediction_cadence_seconds=float(args.prediction_cadence_seconds),
        frequencies=frequencies,
        antennas=["rv2_coarse"],
        limb_exclusion_sources_df=_limb_exclusion_sources(),
        limb_exclusion_deg=float(args.limb_exclusion_deg),
    )
    if not events.empty:
        events = events.merge(
            sources[["source_name", "ecliptic_lon_deg", "ecliptic_lat_deg", "control_class"]],
            on="source_name",
            how="left",
        )
    events.to_csv(out_dir / "ecliptic_control_predicted_events.csv", index=False)
    if args.write_states:
        states.to_csv(out_dir / "ecliptic_control_limb_visibility_states.csv", index=False)

    unfiltered = _read(UNFILTERED_EVENTS)
    counts = pd.concat(
        [
            _count_table(unfiltered, "unfiltered"),
            _count_table(events, f"sun_earth_limb_gt_{args.limb_exclusion_deg:g}deg"),
        ],
        ignore_index=True,
    )
    if not counts.empty:
        pivot = counts.pivot_table(
            index=["control_class", "frequency_mhz", "event_type"],
            columns="run_label",
            values="n_event_ids",
            aggfunc="first",
        ).reset_index()
        filtered_col = f"sun_earth_limb_gt_{args.limb_exclusion_deg:g}deg"
        if "unfiltered" in pivot.columns and filtered_col in pivot.columns:
            pivot["removed_event_ids"] = pivot["unfiltered"] - pivot[filtered_col]
            pivot["retained_fraction"] = pivot[filtered_col] / pivot["unfiltered"]
        pivot.to_csv(out_dir / "ecliptic_control_limb_filter_event_counts.csv", index=False)
    counts.to_csv(out_dir / "ecliptic_control_limb_filter_event_count_long.csv", index=False)

    lines = [
        "# Ecliptic Controls With Sun/Earth Limb Filter",
        "",
        f"Applied veto: reject fixed-control events where either Sun or Earth is within {args.limb_exclusion_deg:g} deg of the lunar limb at the predicted control event time.",
        "",
        f"Filtered event rows: {len(events)}",
        f"Unfiltered event rows: {len(unfiltered)}",
        "",
        "Outputs:",
        "",
        f"- `{out_dir / 'ecliptic_control_predicted_events.csv'}`",
        f"- `{out_dir / 'ecliptic_control_limb_filter_event_counts.csv'}`",
        "",
    ]
    (out_dir / "ecliptic_control_limb_filter_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(out_dir / "ecliptic_control_limb_filter_report.md")


if __name__ == "__main__":
    main()
