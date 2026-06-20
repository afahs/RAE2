"""CSV ingestion, validation, and internal time-series model."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import (
    DEFAULT_VALUE_COLUMNS,
    EARTH_UNIT_COLUMNS,
    FREQUENCY_COLUMN,
    GEOMETRY_COLUMNS,
    SPACECRAFT_COLUMNS,
    TIME_COLUMN,
    add_frequency_mhz_column,
)
from .table_io import read_table
from .util import append_flag, datetime_ns, robust_sigma


@dataclass(frozen=True)
class IngestOptions:
    start_time: str | None = None
    end_time: str | None = None
    value_columns: tuple[str, ...] = tuple(DEFAULT_VALUE_COLUMNS)
    gap_factor: float = 3.0
    artifact_sigma: float = 12.0
    use_existing_loader: bool = True


def _try_existing_loader() -> object | None:
    repo_rae2 = Path(__file__).resolve().parents[2] / "RAE2AgentV2"
    if repo_rae2.exists() and str(repo_rae2) not in sys.path:
        sys.path.insert(0, str(repo_rae2))
    try:
        from rae2.io import load_time_slice_from_csv  # type: ignore

        return load_time_slice_from_csv
    except Exception:
        return None


def required_columns(value_columns: list[str] | tuple[str, ...]) -> list[str]:
    cols = [TIME_COLUMN, FREQUENCY_COLUMN, *GEOMETRY_COLUMNS, *value_columns]
    return list(dict.fromkeys(cols))


def load_raw_csv(
    csv_path: str | Path,
    options: IngestOptions | None = None,
    extra_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load a raw RAE-2 CSV slice while preserving source columns."""
    opts = options or IngestOptions()
    columns = required_columns(opts.value_columns)
    if extra_columns:
        columns = list(dict.fromkeys([*columns, *extra_columns]))

    loader = _try_existing_loader() if opts.use_existing_loader else None
    if loader is not None and opts.start_time is not None and opts.end_time is not None:
        try:
            return loader(csv_path, opts.start_time, opts.end_time, columns=columns)
        except Exception:
            pass

    df = read_table(csv_path, usecols=lambda col: col in set(columns), low_memory=False)
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN], errors="coerce")
    if opts.start_time is not None:
        df = df[df[TIME_COLUMN] >= pd.Timestamp(opts.start_time)]
    if opts.end_time is not None:
        df = df[df[TIME_COLUMN] <= pd.Timestamp(opts.end_time)]
    return df.sort_values(TIME_COLUMN).reset_index(drop=True)


def to_long_timeseries(raw_df: pd.DataFrame, value_columns: list[str] | tuple[str, ...]) -> pd.DataFrame:
    """Convert wide Ryle-Vonberg columns into a composable long time-series table."""
    id_cols = [TIME_COLUMN, FREQUENCY_COLUMN, *GEOMETRY_COLUMNS]
    id_cols = [c for c in id_cols if c in raw_df.columns]
    work = raw_df.copy()
    work["original_time"] = raw_df[TIME_COLUMN].astype(str)
    id_cols = ["original_time", *id_cols]
    long_df = work.melt(
        id_vars=id_cols,
        value_vars=[c for c in value_columns if c in work.columns],
        var_name="antenna",
        value_name="power",
    )
    long_df["time"] = pd.to_datetime(long_df["time"], errors="coerce")
    long_df["power"] = pd.to_numeric(long_df["power"], errors="coerce")
    long_df = add_frequency_mhz_column(long_df)
    return long_df.sort_values(["antenna", FREQUENCY_COLUMN, "time"]).reset_index(drop=True)


def validate_timeseries(
    ts: pd.DataFrame,
    gap_factor: float = 3.0,
    artifact_sigma: float = 12.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach validation flags and return row-level data plus a compact report."""
    out = ts.copy()
    out["quality_flags"] = ""

    malformed = out["time"].isna() | out["power"].isna()
    for col in SPACECRAFT_COLUMNS + EARTH_UNIT_COLUMNS:
        if col in out.columns:
            malformed = malformed | ~np.isfinite(pd.to_numeric(out[col], errors="coerce"))
    out["quality_flags"] = append_flag(out["quality_flags"], malformed.to_numpy(), "malformed")

    dup = out.duplicated(subset=["time", FREQUENCY_COLUMN, "antenna"], keep=False)
    out["quality_flags"] = append_flag(out["quality_flags"], dup.to_numpy(), "duplicate_timestamp")

    report_rows: list[dict] = []
    for (antenna, freq), idx in out.groupby(["antenna", FREQUENCY_COLUMN], sort=True).groups.items():
        group = out.loc[idx].sort_values("time")
        times = pd.DatetimeIndex(group["time"])
        valid_times = times[~times.isna()]
        gap_mask = np.zeros(len(group), dtype=bool)
        missing_est = 0
        cadence_s = np.nan
        max_gap_s = np.nan
        if len(valid_times) >= 3:
            dt = np.diff(datetime_ns(valid_times) / 1e9)
            dt = dt[np.isfinite(dt) & (dt > 0)]
            if dt.size:
                cadence_s = float(np.median(dt))
                max_gap_s = float(np.max(dt))
                big = dt > float(gap_factor) * cadence_s
                missing_est = int(np.sum(np.maximum(np.rint(dt[big] / cadence_s) - 1, 0)))
                gap_positions = np.where(big)[0] + 1
                locs = group.index.to_numpy()[gap_positions]
                gap_mask[np.isin(group.index.to_numpy(), locs)] = True

        values = group["power"].to_numpy(dtype=float)
        finite = np.isfinite(values)
        sat = finite & ((values <= 0.0) | (values >= np.nanpercentile(values[finite], 99.99) if finite.any() else False))
        diffs = np.diff(values)
        sig = robust_sigma(diffs)
        artifact = np.zeros(len(values), dtype=bool)
        if np.isfinite(sig) and sig > 0.0:
            jump = np.abs(diffs) > float(artifact_sigma) * sig
            artifact[np.where(jump)[0] + 1] = True

        out.loc[group.index, "quality_flags"] = append_flag(out.loc[group.index, "quality_flags"], gap_mask, "gap_after_previous")
        out.loc[group.index, "quality_flags"] = append_flag(out.loc[group.index, "quality_flags"], sat, "saturation_or_nonpositive")
        out.loc[group.index, "quality_flags"] = append_flag(out.loc[group.index, "quality_flags"], artifact, "telemetry_artifact_jump")

        report_rows.append(
            {
                "antenna": antenna,
                "frequency_band": freq,
                "n_rows": int(len(group)),
                "n_malformed": int(malformed.loc[group.index].sum()),
                "n_duplicates": int(dup.loc[group.index].sum()),
                "cadence_seconds_median": cadence_s,
                "max_gap_seconds": max_gap_s,
                "estimated_missing_samples": missing_est,
                "n_gap_flags": int(np.count_nonzero(gap_mask)),
                "n_saturation_flags": int(np.count_nonzero(sat)),
                "n_artifact_flags": int(np.count_nonzero(artifact)),
            }
        )

    out["is_valid"] = out["quality_flags"].eq("")
    return out.reset_index(drop=True), pd.DataFrame.from_records(report_rows)


def ingest_csv(csv_path: str | Path, options: IngestOptions | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    opts = options or IngestOptions()
    raw = load_raw_csv(csv_path, opts)
    long_df = to_long_timeseries(raw, opts.value_columns)
    return validate_timeseries(long_df, gap_factor=opts.gap_factor, artifact_sigma=opts.artifact_sigma)
