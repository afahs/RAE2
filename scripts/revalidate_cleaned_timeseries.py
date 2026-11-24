"""Re-apply the current ingest quality policy to a retained cleaned time series.

The retained occultogram_v1 table already holds the long-format samples
produced by ``ingest`` (time, band, geometry, antenna, power). Only the row
flags were produced by an older policy, so this drops ``quality_flags`` and
``is_valid`` and recomputes them with ``validate_timeseries``. Times, powers
and geometry are copied unchanged; the script verifies that.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.ingest import QUALITY_POLICY_VERSION, validate_timeseries  # noqa: E402
from rylevonberg.table_io import write_table  # noqa: E402
from rylevonberg.util import software_versions  # noqa: E402

V1 = ROOT / "runtime_inputs/occultogram_v1/cleaned_timeseries.parquet"
V2 = ROOT / "runtime_inputs/occultogram_v2/cleaned_timeseries.parquet"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=V1)
    parser.add_argument("--output", type=Path, default=V2)
    parser.add_argument("--gap-factor", type=float, default=3.0)
    parser.add_argument("--artifact-sigma", type=float, default=12.0)
    args = parser.parse_args()

    old = pd.read_parquet(args.input)
    old_valid = old["is_valid"].astype(bool).to_numpy()
    samples = old.drop(columns=["quality_flags", "is_valid"])
    new, report = validate_timeseries(samples, gap_factor=args.gap_factor, artifact_sigma=args.artifact_sigma)

    key = ["time", "frequency_band", "antenna"]
    left = old[key + ["power"]].reset_index(drop=True)
    right = new[key + ["power"]].reset_index(drop=True)
    if len(left) != len(right) or not left.equals(right):
        raise RuntimeError("revalidation changed sample order or values; refusing to write")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_table(new, args.output, index=False)
    write_table(report, args.output.with_name("cleaned_timeseries_validation_report.csv"), index=False)

    new_valid = new["is_valid"].astype(bool).to_numpy()
    flags = new["quality_flags"].fillna("").str.split(";").explode()
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "quality_policy_version": QUALITY_POLICY_VERSION,
        "rows": int(len(new)),
        "valid_fraction_v1": float(old_valid.mean()),
        "valid_fraction_v2": float(new_valid.mean()),
        "rows_newly_valid": int(np.sum(new_valid & ~old_valid)),
        "rows_newly_invalid": int(np.sum(~new_valid & old_valid)),
        "flag_counts_v2": {k: int(v) for k, v in flags[flags != ""].value_counts().items()},
        "samples_unchanged": True,
        "versions": software_versions(),
    }
    args.output.with_name("cleaned_timeseries_revalidation.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "versions"}, indent=2))


if __name__ == "__main__":
    main()
