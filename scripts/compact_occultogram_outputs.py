#!/usr/bin/env python
"""Convert legacy occultogram NPY/CSV products to compact Zstd Parquet."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from rylevonberg.occultogram import compact_occultogram_profiles, occultogram_event_metadata  # noqa: E402
from rylevonberg.table_io import read_table, write_table  # noqa: E402


def _file_size(path: Path) -> int:
    return int(path.stat().st_size) if path.exists() else 0


def _validate_profile_round_trip(original: pd.DataFrame, compact_path: Path, metadata_path: Path) -> dict[str, int]:
    expected = compact_occultogram_profiles(original)
    compact = read_table(compact_path)
    metadata = read_table(metadata_path)
    keys = ["event_uid", "control_shift_s"]
    if len(compact) != len(expected):
        raise RuntimeError(f"profile row count changed: {len(expected)} -> {len(compact)}")
    if int(compact["n_samples"].sum()) != int(expected["n_samples"].sum()):
        raise RuntimeError("profile n_samples total changed")
    if compact[keys].drop_duplicates().shape[0] != expected[keys].drop_duplicates().shape[0]:
        raise RuntimeError("profile event/control key count changed")
    if metadata.duplicated(keys).any():
        raise RuntimeError("metadata contains duplicate event/control keys")
    if metadata[keys].shape[0] != expected[keys].drop_duplicates().shape[0]:
        raise RuntimeError("metadata does not cover every event/control key")
    for column in ["raw_power", "normalized_power"]:
        left = pd.to_numeric(expected[column], errors="coerce").to_numpy(dtype=float)
        right = pd.to_numeric(compact[column], errors="coerce").to_numpy(dtype=float)
        if not np.allclose(left, right, rtol=0.0, atol=0.0, equal_nan=True):
            raise RuntimeError(f"{column} values changed during Parquet conversion")
    return {
        "profile_rows": int(len(compact)),
        "event_control_keys": int(compact[keys].drop_duplicates().shape[0]),
        "sample_total": int(compact["n_samples"].sum()),
        "metadata_rows": int(len(metadata)),
    }


def _convert_csv(csv_path: Path) -> tuple[Path, int] | None:
    if not csv_path.exists():
        return None
    parquet_path = csv_path.with_suffix(".parquet")
    table = pd.read_csv(csv_path, low_memory=False)
    write_table(table, parquet_path, index=False)
    round_trip = read_table(parquet_path)
    if len(round_trip) != len(table) or list(round_trip.columns) != list(table.columns):
        raise RuntimeError(f"CSV/Parquet validation failed for {csv_path.name}")
    return parquet_path, int(len(table))


def compact_directory(directory: Path, delete_originals: bool = False) -> dict[str, object]:
    directory = directory.resolve()
    profile_npy = directory / "occultogram_event_profiles.npy"
    if not profile_npy.exists():
        raise FileNotFoundError(profile_npy)
    original_sizes = {
        path.name: _file_size(path)
        for path in [
            profile_npy,
            directory / "occultogram_event_and_lunar_block_summary.csv",
            directory / "occultogram_event_signed_contrasts.csv",
        ]
        if path.exists()
    }
    original = read_table(profile_npy)
    compact_path = directory / "occultogram_event_profiles.parquet"
    metadata_path = directory / "occultogram_event_metadata.parquet"
    write_table(compact_occultogram_profiles(original), compact_path, index=False)
    write_table(occultogram_event_metadata(original), metadata_path, index=False)
    validation = _validate_profile_round_trip(original, compact_path, metadata_path)

    converted_csvs: dict[str, int] = {}
    for name in [
        "occultogram_event_and_lunar_block_summary.csv",
        "occultogram_event_signed_contrasts.csv",
    ]:
        result = _convert_csv(directory / name)
        if result is not None:
            converted_csvs[result[0].name] = result[1]

    compact_paths = [compact_path, metadata_path, *(directory / name for name in converted_csvs)]
    compact_sizes = {path.name: _file_size(path) for path in compact_paths}
    if any(size <= 0 for size in compact_sizes.values()):
        raise RuntimeError("one or more compact outputs are empty")
    manifest: dict[str, object] = {
        "directory": str(directory),
        "format": "Parquet with Zstandard compression",
        "original_sizes_bytes": original_sizes,
        "compact_sizes_bytes": compact_sizes,
        "validation": validation,
        "converted_table_rows": converted_csvs,
        "originals_deleted": False,
    }
    manifest_path = directory / "compaction_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if delete_originals:
        delete_paths = [profile_npy]
        delete_paths.extend(
            path
            for path in [
                directory / "occultogram_event_and_lunar_block_summary.csv",
                directory / "occultogram_event_signed_contrasts.csv",
            ]
            if path.exists()
        )
        for path in delete_paths:
            path.unlink()
        manifest["originals_deleted"] = True
        manifest["deleted_originals"] = [path.name for path in delete_paths]
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--delete-originals", action="store_true")
    args = parser.parse_args()
    manifest = compact_directory(args.directory, delete_originals=args.delete_originals)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
