"""Source-list loading."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .table_io import read_table


SOURCE_COLUMNS = ["source_name", "kind", "ra_deg", "dec_deg", "frame"]


def load_source_list(path: str | Path) -> pd.DataFrame:
    df = read_table(path)
    missing = sorted(set(["source_name", "kind"]) - set(df.columns))
    if missing:
        raise ValueError(f"source list is missing required columns: {missing}")
    for col in SOURCE_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df["kind"] = df["kind"].str.lower()
    df["frame"] = df["frame"].replace("", "fk4").fillna("fk4")
    return df


def filter_sources(df: pd.DataFrame, names: list[str] | None) -> pd.DataFrame:
    if not names:
        return df.reset_index(drop=True)
    wanted = {n.lower() for n in names}
    out = df[df["source_name"].str.lower().isin(wanted)].copy()
    missing = sorted(wanted - set(out["source_name"].str.lower()))
    if missing:
        raise ValueError(f"Unknown source(s): {', '.join(missing)}")
    return out.reset_index(drop=True)
