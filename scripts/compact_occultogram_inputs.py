#!/usr/bin/env python
"""Preserve only the compact inputs needed to rebuild the occultograms."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype  # noqa: E402

from rylevonberg.table_io import read_table, write_table  # noqa: E402


INPUTS = {
    "cleaned_timeseries": Path("control_survey_earth_sun_postnov1974_v2/01_ingest/cleaned_timeseries.npy"),
    "sun_events": Path("pipeline_confidence_audit_v2/sun_audit_input_events.csv"),
    "bright_source_events": Path("control_survey_bright_sources_postnov1974_v1/02_events/predicted_events.csv"),
    "planet_events_science_baseline": Path(
        "planetary_confirmation_survey_science_baseline_v2/events/all_planet_predicted_events.csv"
    ),
    "virgo_events": Path(
        "custom_fixed_source_profile_grids_3c295_3c273_virgoa_v1/custom_fixed_source_predicted_events.csv"
    ),
    "outer_planet_events": Path("planetary_confirmation_survey/events/all_planet_predicted_events.npy"),
}


def _validate_series(left: pd.Series, right: pd.Series) -> None:
    if is_datetime64_any_dtype(left.dtype) or is_datetime64_any_dtype(right.dtype):
        left_values = pd.to_datetime(left, errors="coerce").to_numpy(dtype="datetime64[ns]")
        right_values = pd.to_datetime(right, errors="coerce").to_numpy(dtype="datetime64[ns]")
        if not np.array_equal(left_values, right_values, equal_nan=True):
            raise RuntimeError(f"datetime values changed in column {left.name}")
        return
    if is_numeric_dtype(left.dtype) and is_numeric_dtype(right.dtype):
        left_values = pd.to_numeric(left, errors="coerce").to_numpy()
        right_values = pd.to_numeric(right, errors="coerce").to_numpy()
        if not np.array_equal(left_values, right_values, equal_nan=True):
            raise RuntimeError(f"numeric values changed in column {left.name}")
        return
    left_values = left.astype("string").fillna("<NA>").to_numpy(dtype=str)
    right_values = right.astype("string").fillna("<NA>").to_numpy(dtype=str)
    if not np.array_equal(left_values, right_values):
        raise RuntimeError(f"text values changed in column {left.name}")


def _validate_round_trip(original: pd.DataFrame, compact: pd.DataFrame, name: str) -> None:
    if len(original) != len(compact):
        raise RuntimeError(f"{name}: row count changed: {len(original)} -> {len(compact)}")
    if list(original.columns) != list(compact.columns):
        raise RuntimeError(f"{name}: columns changed")
    for column in original.columns:
        _validate_series(original[column].reset_index(drop=True), compact[column].reset_index(drop=True))


def compact_inputs(outputs_root: Path, out_dir: Path) -> dict[str, object]:
    outputs_root = outputs_root.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tables: dict[str, object] = {}
    for name, relative_path in INPUTS.items():
        source = outputs_root / relative_path
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = out_dir / f"{name}.parquet"
        original = read_table(source)
        if not isinstance(original, pd.DataFrame) or original.empty:
            raise RuntimeError(f"{name}: source table is empty")
        write_table(original, destination, index=False)
        compact = read_table(destination)
        _validate_round_trip(original, compact, name)
        tables[name] = {
            "source": str(source),
            "source_bytes": int(source.stat().st_size),
            "compact": str(destination),
            "compact_bytes": int(destination.stat().st_size),
            "rows": int(len(compact)),
            "columns": list(compact.columns),
        }
        del original, compact
        gc.collect()

    manifest: dict[str, object] = {
        "format": "Parquet with Zstandard compression",
        "purpose": "minimal durable inputs required to rebuild the morphology-first occultograms",
        "tables": tables,
        "validation": "exact row count, column order, and value round trip for every table",
    }
    manifest_path = out_dir / "input_compaction_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runtime_inputs/occultogram_v1")
    args = parser.parse_args()
    print(json.dumps(compact_inputs(args.outputs_root, args.out_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
