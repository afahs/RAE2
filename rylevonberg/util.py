"""Small shared utilities."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def software_versions() -> dict[str, str]:
    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    for name in ["astropy", "scipy", "matplotlib"]:
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[name] = "not-importable"
    try:
        versions["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        versions["git_commit"] = "unknown"
    return versions


def robust_sigma(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    if mad > 0.0 and np.isfinite(mad):
        return 1.4826 * mad
    std = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
    return std if np.isfinite(std) else float("nan")


def datetime_ns(values: pd.Series | pd.DatetimeIndex | np.ndarray) -> np.ndarray:
    """Return timestamps as integer nanoseconds, independent of pandas storage unit."""
    return pd.DatetimeIndex(values).to_numpy(dtype="datetime64[ns]").astype("int64")


def append_flag(existing: pd.Series, mask: np.ndarray, flag: str) -> pd.Series:
    out = existing.fillna("").astype(str).copy()
    mask = np.asarray(mask, dtype=bool)
    out.loc[mask & (out != "")] = out.loc[mask & (out != "")] + ";" + flag
    out.loc[mask & (out == "")] = flag
    return out
