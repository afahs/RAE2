"""Minimal programmatic use of the Ryle-Vonberg package."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rylevonberg.ingest import IngestOptions, ingest_csv
from rylevonberg.sources import load_source_list


def main() -> None:
    csv_path = os.environ.get("RAE2_MASTER_CSV")
    if not csv_path:
        raise SystemExit("Set RAE2_MASTER_CSV to the local RAE-2 master CSV path.")

    cleaned, report = ingest_csv(
        csv_path,
        IngestOptions(
            start_time="1973-10-03 04:00:00",
            end_time="1973-10-03 06:00:00",
            value_columns=("rv1_coarse", "rv2_coarse"),
            use_existing_loader=False,
        ),
    )
    sources = load_source_list(Path("configs") / "bright_sources.csv")

    print(cleaned.head())
    print(report.head())
    print(sources[["source_name", "source_type"]].head())


if __name__ == "__main__":
    main()
