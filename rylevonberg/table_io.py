"""Small table I/O helpers for generated pipeline artifacts.

Generated Ryle-Vonberg tables can become very large as CSV files.  These helpers
keep CSV compatibility for small/source tables while allowing large generated
tables to be replaced by NumPy ``.npy`` record arrays.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CSV_SUFFIX = ".csv"
NPY_SUFFIX = ".npy"


def resolve_table_path(path: str | Path) -> Path:
    """Return an existing table path, accepting CSV/NPY sidecar swaps.

    Most older scripts point at ``*.csv`` outputs.  After cleanup those large
    generated files may exist only as ``*.npy``.  A requested CSV path therefore
    resolves to the NPY sidecar when the CSV is absent.
    """
    table_path = Path(path)
    if table_path.exists():
        return table_path
    if table_path.suffix.lower() == CSV_SUFFIX:
        npy_path = table_path.with_suffix(NPY_SUFFIX)
        if npy_path.exists():
            return npy_path
    if table_path.suffix.lower() == NPY_SUFFIX:
        csv_path = table_path.with_suffix(CSV_SUFFIX)
        if csv_path.exists():
            return csv_path
    return table_path


def _apply_usecols(df: pd.DataFrame, usecols: Any) -> pd.DataFrame:
    if usecols is None:
        return df
    if callable(usecols):
        cols = [c for c in df.columns if usecols(c)]
        return df.loc[:, cols]
    return df.loc[:, [c for c in usecols if c in df.columns]]


def _apply_parse_dates(df: pd.DataFrame, parse_dates: Any) -> pd.DataFrame:
    if parse_dates is None or parse_dates is False:
        return df
    out = df.copy()
    if parse_dates is True:
        cols: Iterable[Any] = []
    elif isinstance(parse_dates, dict):
        cols = parse_dates.keys()
    elif isinstance(parse_dates, (str, bytes)):
        cols = [parse_dates]
    else:
        cols = parse_dates
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def _load_npy_dataframe(path: Path) -> pd.DataFrame:
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.lib.npyio.NpzFile):
        raise ValueError(f"Expected a .npy table, got npz archive: {path}")
    if getattr(arr.dtype, "names", None):
        return pd.DataFrame.from_records(arr)
    if arr.ndim == 0:
        item = arr.item()
        if isinstance(item, pd.DataFrame):
            return item
        if isinstance(item, dict):
            return pd.DataFrame(item)
    raise ValueError(f"Unsupported NPY table layout: {path}")


def _iter_chunks(df: pd.DataFrame, chunksize: int) -> Iterator[pd.DataFrame]:
    for start in range(0, len(df), chunksize):
        yield df.iloc[start : start + chunksize].copy()


def read_table(path: str | Path, **kwargs: Any) -> pd.DataFrame | Iterator[pd.DataFrame]:
    """Read a CSV or generated NPY table.

    This intentionally mirrors the subset of ``pandas.read_csv`` options used in
    the pipeline.  CSV paths are passed through to pandas.  NPY paths are loaded
    as DataFrames and then filtered for ``usecols``, ``parse_dates``, ``nrows``,
    and ``chunksize`` when those options are supplied.
    """
    table_path = resolve_table_path(path)
    if table_path.suffix.lower() != NPY_SUFFIX:
        return pd.read_csv(table_path, **kwargs)

    usecols = kwargs.pop("usecols", None)
    parse_dates = kwargs.pop("parse_dates", None)
    nrows = kwargs.pop("nrows", None)
    chunksize = kwargs.pop("chunksize", None)
    skiprows = kwargs.pop("skiprows", None)

    # CSV-only parser knobs are harmless for generated NPY tables.
    for ignored in (
        "low_memory",
        "engine",
        "dtype_backend",
        "memory_map",
        "on_bad_lines",
        "comment",
        "header",
        "index_col",
    ):
        kwargs.pop(ignored, None)
    if kwargs:
        unsupported = ", ".join(sorted(kwargs))
        raise TypeError(f"Unsupported read_table options for NPY input {table_path}: {unsupported}")

    df = _load_npy_dataframe(table_path)
    df = _apply_usecols(df, usecols)
    df = _apply_parse_dates(df, parse_dates)
    if skiprows is not None:
        if isinstance(skiprows, int):
            df = df.iloc[skiprows:]
        else:
            raise TypeError("NPY read_table only supports integer skiprows")
    if nrows is not None:
        df = df.iloc[: int(nrows)]
    if chunksize is not None:
        return _iter_chunks(df, int(chunksize))
    return df.reset_index(drop=True)


def write_table(df: pd.DataFrame, path: str | Path, **kwargs: Any) -> Path:
    """Write a table as CSV or NPY based on the filename suffix."""
    table_path = Path(path)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    if table_path.suffix.lower() == NPY_SUFFIX:
        if "index" in kwargs:
            kwargs.pop("index")
        if kwargs:
            unsupported = ", ".join(sorted(kwargs))
            raise TypeError(f"Unsupported write_table options for NPY output {table_path}: {unsupported}")
        np.save(table_path, df.to_records(index=False), allow_pickle=True)
        return table_path
    df.to_csv(table_path, **kwargs)
    return table_path


def output_table_path(directory: str | Path, stem: str, table_format: str = "csv") -> Path:
    """Build a generated table path from an output directory and format name."""
    fmt = table_format.lower().lstrip(".")
    if fmt not in {"csv", "npy"}:
        raise ValueError(f"Unsupported table format: {table_format}")
    return Path(directory) / f"{stem}.{fmt}"
