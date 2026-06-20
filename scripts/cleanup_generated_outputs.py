#!/usr/bin/env python
"""Cleanup generated Ryle-Vonberg outputs without touching source/config data."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rylevonberg.table_io import write_table


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".pdf"}


def _file_size(path: Path) -> int:
    return path.stat().st_size


def _iter_large_csv(root: Path, min_size_mb: float) -> list[Path]:
    min_bytes = int(min_size_mb * 1024 * 1024)
    return sorted(p for p in root.rglob("*.csv") if p.is_file() and _file_size(p) >= min_bytes)


def _iter_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def _verify_npy(path: Path, expected_rows: int, expected_columns: list[str]) -> tuple[bool, str]:
    try:
        arr = np.load(path, allow_pickle=True)
        names = list(arr.dtype.names or [])
        if len(arr) != expected_rows:
            return False, f"row_count_mismatch:{len(arr)}!={expected_rows}"
        if names != expected_columns:
            return False, "column_mismatch"
    except Exception as exc:  # pragma: no cover - defensive report path
        return False, f"load_failed:{type(exc).__name__}:{exc}"
    return True, "ok"


def convert_large_csvs(
    output_root: Path,
    manifest_dir: Path,
    min_size_mb: float,
    remove_csv: bool,
    limit: int | None,
) -> pd.DataFrame:
    rows: list[dict] = []
    manifest_path = manifest_dir / "large_csv_conversion_manifest.csv"
    csv_paths = _iter_large_csv(output_root, min_size_mb)
    if limit is not None:
        csv_paths = csv_paths[:limit]
    for csv_path in csv_paths:
        if not csv_path.exists():
            rows.append(
                {
                    "csv_path": str(csv_path),
                    "npy_path": str(csv_path.with_suffix(".npy")),
                    "csv_size_bytes": None,
                    "npy_size_bytes": None,
                    "rows": None,
                    "columns": None,
                    "status": "skipped",
                    "seconds": 0.0,
                    "csv_removed": False,
                    "notes": "csv_path_missing_at_conversion_time",
                }
            )
            write_table(pd.DataFrame(rows), manifest_path, index=False)
            continue
        t0 = time.monotonic()
        npy_path = csv_path.with_suffix(".npy")
        row = {
            "csv_path": str(csv_path),
            "npy_path": str(npy_path),
            "csv_size_bytes": _file_size(csv_path),
            "npy_size_bytes": None,
            "rows": None,
            "columns": None,
            "status": "started",
            "seconds": None,
            "csv_removed": False,
            "notes": "",
        }
        try:
            df = pd.read_csv(csv_path, low_memory=False)
            row["rows"] = len(df)
            row["columns"] = len(df.columns)
            write_table(df, npy_path)
            ok, message = _verify_npy(npy_path, len(df), list(df.columns))
            if not ok:
                row["status"] = "failed"
                row["notes"] = message
            else:
                row["npy_size_bytes"] = _file_size(npy_path)
                row["status"] = "converted"
                row["notes"] = message
                if remove_csv:
                    csv_path.unlink()
                    row["csv_removed"] = True
        except Exception as exc:  # pragma: no cover - report and continue
            row["status"] = "failed"
            row["notes"] = f"{type(exc).__name__}: {exc}"
            if npy_path.exists() and row["status"] != "converted":
                npy_path.unlink()
        row["seconds"] = round(time.monotonic() - t0, 3)
        rows.append(row)
        write_table(pd.DataFrame(rows), manifest_path, index=False)
        print(f"{row['status']}: {csv_path} -> {npy_path} ({row['notes']})", flush=True)
    manifest = pd.DataFrame(rows)
    write_table(manifest, manifest_path, index=False)
    return manifest


def write_image_inventory(output_root: Path, manifest_dir: Path) -> pd.DataFrame:
    rows = [
        {
            "path": str(path),
            "size_bytes": _file_size(path),
            "suffix": path.suffix.lower(),
            "parent": str(path.parent),
        }
        for path in _iter_images(output_root)
    ]
    inventory = pd.DataFrame(rows).sort_values("size_bytes", ascending=False) if rows else pd.DataFrame(rows)
    write_table(inventory, manifest_dir / "image_inventory.csv", index=False)
    return inventory


def write_post_cleanup_inventory(output_root: Path, manifest_dir: Path, min_size_mb: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    npy_rows = [
        {
            "path": str(path),
            "size_bytes": _file_size(path),
            "csv_sidecar_exists": path.with_suffix(".csv").exists(),
        }
        for path in sorted(output_root.rglob("*.npy"))
        if path.is_file()
    ]
    npy_inventory = pd.DataFrame(npy_rows).sort_values("size_bytes", ascending=False) if npy_rows else pd.DataFrame(npy_rows)
    write_table(npy_inventory, manifest_dir / "npy_inventory.csv", index=False)

    large_csv_rows = [
        {
            "path": str(path),
            "size_bytes": _file_size(path),
        }
        for path in _iter_large_csv(output_root, min_size_mb)
    ]
    large_csv_inventory = (
        pd.DataFrame(large_csv_rows).sort_values("size_bytes", ascending=False)
        if large_csv_rows
        else pd.DataFrame(large_csv_rows)
    )
    write_table(large_csv_inventory, manifest_dir / "remaining_large_csv_inventory.csv", index=False)
    return npy_inventory, large_csv_inventory


def remove_empty_dirs(root: Path) -> int:
    removed = 0
    for path in sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        try:
            path.rmdir()
            removed += 1
        except OSError:
            pass
    return removed


def write_report(
    report_path: Path,
    output_root: Path,
    manifest: pd.DataFrame,
    images: pd.DataFrame,
    npy_inventory: pd.DataFrame,
    remaining_large_csvs: pd.DataFrame,
    removed_empty_dirs: int,
    args: argparse.Namespace,
) -> None:
    converted = int((manifest["status"] == "converted").sum()) if not manifest.empty else 0
    failed = int((manifest["status"] == "failed").sum()) if not manifest.empty else 0
    csv_removed = int(manifest["csv_removed"].sum()) if "csv_removed" in manifest else 0
    csv_before = int(manifest["csv_size_bytes"].sum()) if "csv_size_bytes" in manifest else 0
    npy_after = int(manifest["npy_size_bytes"].fillna(0).sum()) if "npy_size_bytes" in manifest else 0
    image_count = len(images)
    image_bytes = int(images["size_bytes"].sum()) if "size_bytes" in images else 0
    npy_count = len(npy_inventory)
    npy_bytes = int(npy_inventory["size_bytes"].sum()) if "size_bytes" in npy_inventory else 0
    remaining_large_csv_count = len(remaining_large_csvs)
    remaining_large_csv_bytes = int(remaining_large_csvs["size_bytes"].sum()) if "size_bytes" in remaining_large_csvs else 0
    lines = [
        "# Ryle-Vonberg Output Cleanup Report",
        "",
        f"Output root: `{output_root}`",
        f"CSV conversion threshold: `{args.csv_min_size_mb}` MB",
        f"Remove CSV after verified NPY: `{args.remove_converted_csv}`",
        "",
        "## Large CSV Conversion",
        "",
        "This section reports conversions performed in this invocation. The post-cleanup audit below is the authoritative current filesystem state.",
        "",
        f"- converted files: `{converted}`",
        f"- failed files: `{failed}`",
        f"- CSV files removed: `{csv_removed}`",
        f"- original converted CSV bytes: `{csv_before}`",
        f"- replacement NPY bytes: `{npy_after}`",
        f"- approximate bytes saved from converted files: `{csv_before - npy_after if csv_removed else 0}`",
        "",
        "Manifest:",
        "",
        "- `large_csv_conversion_manifest.csv`",
        "",
        "## Post-Cleanup Table Audit",
        "",
        f"- NPY output tables present: `{npy_count}`",
        f"- NPY output table bytes: `{npy_bytes}`",
        f"- remaining CSV files above threshold: `{remaining_large_csv_count}`",
        f"- remaining CSV bytes above threshold: `{remaining_large_csv_bytes}`",
        "- `npy_inventory.csv`",
        "- `remaining_large_csv_inventory.csv`",
        "",
        "## Image Inventory",
        "",
        "No images were removed by this script unless a future run adds an explicit image-removal mode.",
        f"- image-like files inventoried: `{image_count}`",
        f"- image-like bytes inventoried: `{image_bytes}`",
        "- `image_inventory.csv`",
        "",
        "## Empty Directories",
        "",
        f"- empty directories removed: `{removed_empty_dirs}`",
    ]
    report_path.write_text("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(ROOT / "outputs"))
    parser.add_argument("--manifest-dir", default=str(ROOT / "reports" / "cleanup_generated_outputs"))
    parser.add_argument("--csv-min-size-mb", type=float, default=10.0)
    parser.add_argument("--convert-large-csv", action="store_true")
    parser.add_argument("--remove-converted-csv", action="store_true")
    parser.add_argument("--image-inventory", action="store_true")
    parser.add_argument("--remove-empty-dirs", action="store_true")
    parser.add_argument("--limit", type=int, help="Convert at most this many large CSVs, for smoke testing.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    output_root = Path(args.output_root)
    manifest_dir = Path(args.manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    if args.convert_large_csv:
        manifest = convert_large_csvs(
            output_root,
            manifest_dir,
            min_size_mb=args.csv_min_size_mb,
            remove_csv=args.remove_converted_csv,
            limit=args.limit,
        )
    else:
        manifest = pd.DataFrame()
        write_table(manifest, manifest_dir / "large_csv_conversion_manifest.csv", index=False)

    images = write_image_inventory(output_root, manifest_dir) if args.image_inventory else pd.DataFrame()
    npy_inventory, remaining_large_csvs = write_post_cleanup_inventory(output_root, manifest_dir, args.csv_min_size_mb)
    removed_empty_dirs = remove_empty_dirs(output_root) if args.remove_empty_dirs else 0
    write_report(
        manifest_dir / "cleanup_generated_outputs_report.md",
        output_root,
        manifest,
        images,
        npy_inventory,
        remaining_large_csvs,
        removed_empty_dirs,
        args,
    )
    print(manifest_dir / "cleanup_generated_outputs_report.md")


if __name__ == "__main__":
    main()
