#!/usr/bin/env python
"""Sweep upper-V Earth beam-gain veto thresholds without building full maps."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.constants import FREQUENCY_MAP_MHZ  # noqa: E402
from rylevonberg.util import ensure_dir, software_versions, write_json  # noqa: E402

from scripts.build_upper_v_novaco_brown_scan_maps import (  # noqa: E402
    _attach_earth_beam_gain,
    _filter_valid_pointing,
    _initial_selection_summary,
    _read_selected_rows,
    _detect_input_format,
    new_moon_distance_days,
    parse_frequencies,
)


DEFAULT_RAW = Path(os.environ.get("RAE2_MASTER_CSV", "data/interpolatedRAE2MasterFile.csv"))
DEFAULT_OUT = ROOT / "outputs/upper_v_earth_beam_gain_threshold_sweep_v1"


def parse_thresholds(text: str) -> list[float]:
    items = [part.strip() for part in str(text).split(",") if part.strip()]
    if not items:
        raise ValueError("at least one threshold is required")
    return [float(item) for item in items]


def summarize_thresholds(
    selected: pd.DataFrame,
    antenna: str,
    thresholds: list[float],
) -> pd.DataFrame:
    rows = []
    base = _initial_selection_summary(selected)
    if base.empty:
        return base
    gain = selected["earth_beam_relative_gain_db"].to_numpy(dtype=float)
    finite_gain = np.isfinite(gain)
    for band, grp in selected.groupby("frequency_band", sort=True):
        idx = grp.index.to_numpy()
        local_gain = selected.loc[idx, "earth_beam_relative_gain_db"].to_numpy(dtype=float)
        local_sep = selected.loc[idx, "earth_beam_separation_deg"].to_numpy(dtype=float)
        finite_local = np.isfinite(local_gain)
        for threshold in thresholds:
            keep = finite_local & (local_gain < float(threshold))
            angle_keep = selected.loc[idx, "earth_beam_separation_deg"].to_numpy(dtype=float) > 90.0
            row = {
                "antenna": str(antenna),
                "frequency_band": int(band),
                "frequency_mhz": float(FREQUENCY_MAP_MHZ[int(band)]),
                "earth_gain_threshold_db": float(threshold),
                "n_loaded_valid_positive": int(len(grp)),
                "n_finite_earth_gain": int(np.count_nonzero(finite_local)),
                "n_keep_earth_gain_lt_threshold": int(np.count_nonzero(keep)),
                "n_reject_earth_gain_ge_threshold": int(len(grp) - np.count_nonzero(keep)),
                "frac_keep_earth_gain_lt_threshold": float(np.count_nonzero(keep) / len(grp)) if len(grp) else np.nan,
                "n_keep_old_earth_sep_gt_90deg": int(np.count_nonzero(angle_keep)),
                "frac_keep_old_earth_sep_gt_90deg": float(np.count_nonzero(angle_keep) / len(grp)) if len(grp) else np.nan,
                "n_earth_center_visible_by_moon": int(selected.loc[idx, "earth_visible_by_moon_center"].astype(bool).sum()),
                "earth_beam_model_frequency_mhz": float(np.nanmedian(selected.loc[idx, "earth_beam_model_frequency_mhz"])),
                "earth_beam_relative_gain_median_db": float(np.nanmedian(local_gain[finite_local])) if np.any(finite_local) else np.nan,
                "earth_beam_relative_gain_p90_db": float(np.nanpercentile(local_gain[finite_local], 90)) if np.any(finite_local) else np.nan,
                "earth_beam_separation_median_deg": float(np.nanmedian(local_sep)),
            }
            rows.append(row)
    return pd.DataFrame.from_records(rows)


def run(args: argparse.Namespace) -> Path:
    out_dir = ensure_dir(args.out_dir)
    thresholds = parse_thresholds(args.thresholds)
    bands = parse_frequencies(args.frequencies)
    input_format = _detect_input_format(args.input, args.input_format)
    config = {
        "input": str(args.input),
        "input_format": str(args.input_format),
        "input_format_resolved": input_format,
        "antennas": list(args.antennas),
        "frequencies": bands,
        "frequencies_mhz": [FREQUENCY_MAP_MHZ[band] for band in bands],
        "thresholds_db": thresholds,
        "earth_beam_axis": str(args.earth_beam_axis),
        "ra_units": str(args.ra_units),
        "all_dates": True,
        "software_versions": software_versions(),
    }
    write_json(out_dir / "run_config.json", config)
    frames = []
    for antenna in args.antennas:
        selected = _read_selected_rows(
            args.input,
            str(antenna),
            bands,
            start_time=None,
            end_time=None,
            max_rows=0,
            input_format=input_format,
        )
        if selected.empty:
            continue
        selected, pointing_ra_units, n_removed = _filter_valid_pointing(selected, str(args.ra_units))
        selected["new_moon_distance_days"] = new_moon_distance_days(selected["time"])
        selected["keep_new_moon"] = selected["new_moon_distance_days"].le(float(args.new_moon_half_window_days))
        selected, earth_ra_units = _attach_earth_beam_gain(selected, str(args.earth_beam_axis), str(args.ra_units))
        summary = summarize_thresholds(selected, str(antenna), thresholds)
        summary["n_removed_invalid_pointing"] = int(n_removed)
        summary["pointing_ra_units_interpreted"] = str(pointing_ra_units)
        summary["earth_beam_ra_units_interpreted"] = str(earth_ra_units)
        frames.append(summary)
    if not frames:
        raise SystemExit("No matching samples found for threshold sweep.")
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(out_dir / "earth_beam_gain_threshold_sweep_by_frequency.csv", index=False)
    compact = (
        out.groupby(["antenna", "earth_gain_threshold_db"], as_index=False)
        .agg(
            n_loaded_valid_positive=("n_loaded_valid_positive", "sum"),
            n_keep_earth_gain_lt_threshold=("n_keep_earth_gain_lt_threshold", "sum"),
            n_reject_earth_gain_ge_threshold=("n_reject_earth_gain_ge_threshold", "sum"),
            n_keep_old_earth_sep_gt_90deg=("n_keep_old_earth_sep_gt_90deg", "sum"),
            n_earth_center_visible_by_moon=("n_earth_center_visible_by_moon", "sum"),
        )
        .sort_values(["antenna", "earth_gain_threshold_db"], ascending=[True, False])
    )
    compact["frac_keep_earth_gain_lt_threshold"] = (
        compact["n_keep_earth_gain_lt_threshold"] / compact["n_loaded_valid_positive"]
    )
    compact["frac_keep_old_earth_sep_gt_90deg"] = compact["n_keep_old_earth_sep_gt_90deg"] / compact["n_loaded_valid_positive"]
    compact.to_csv(out_dir / "earth_beam_gain_threshold_sweep_compact.csv", index=False)
    print(out_dir / "earth_beam_gain_threshold_sweep_compact.csv")
    print(compact.to_string(index=False))
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--input-format", choices=["auto", "cleaned", "raw-master"], default="raw-master")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--antennas", nargs="+", default=["rv1_fine", "rv1_coarse"])
    parser.add_argument("--frequencies", default="all")
    parser.add_argument("--thresholds", default="-10,-12,-14,-16,-18,-20")
    parser.add_argument("--earth-beam-axis", choices=["radec", "radial-upper"], default="radec")
    parser.add_argument("--ra-units", choices=["auto", "hours", "degrees"], default="auto")
    parser.add_argument("--new-moon-half-window-days", type=float, default=6.0)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
