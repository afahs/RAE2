"""Structured off-source sky-control generation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OffsourceConfig:
    ra_offsets_deg: tuple[float, ...] = (-20.0, -10.0, -5.0, -2.0, 2.0, 5.0, 10.0, 20.0)
    dec_offsets_deg: tuple[float, ...] = (-20.0, -10.0, -5.0, -2.0, 2.0, 5.0, 10.0, 20.0)
    annulus_radius_deg: float = 10.0
    annulus_positions: int = 16
    frame: str = "fk4"


def _wrap_ra(ra_deg: float) -> float:
    return float(ra_deg % 360.0)


def _clip_dec(dec_deg: float) -> float:
    return float(np.clip(dec_deg, -89.999, 89.999))


def _control_row(
    source: pd.Series,
    control_name: str,
    control_type: str,
    offset_deg: float,
    ra_deg: float,
    dec_deg: float,
    notes: str,
) -> dict[str, object]:
    parent = str(source["source_name"])
    frame = str(source.get("frame", "fk4") or "fk4")
    return {
        "source_name": control_name,
        "kind": "fixed",
        "body_name": "",
        "parent_source": parent,
        "control_name": control_name,
        "control_type": control_type,
        "offset_deg": float(offset_deg),
        "ra_deg": _wrap_ra(float(ra_deg)),
        "dec_deg": _clip_dec(float(dec_deg)),
        "ra_fk4": _wrap_ra(float(ra_deg)),
        "dec_fk4": _clip_dec(float(dec_deg)),
        "frame": frame,
        "notes": notes,
    }


def generate_offsource_controls(sources_df: pd.DataFrame, config: OffsourceConfig | None = None) -> pd.DataFrame:
    """Generate deterministic nearby fixed-position controls for fixed sources."""
    cfg = config or OffsourceConfig()
    rows: list[dict[str, object]] = []
    fixed = sources_df[sources_df.get("kind", "").astype(str).str.lower().eq("fixed")].copy()
    for _, source in fixed.iterrows():
        parent = str(source["source_name"])
        ra = float(source["ra_deg"])
        dec = float(source["dec_deg"])
        for offset in cfg.ra_offsets_deg:
            sign = "p" if offset > 0 else "m"
            name = f"{parent}_off_ra_{sign}{abs(offset):g}"
            rows.append(
                _control_row(
                    source,
                    name,
                    "same_dec_ra_offset",
                    abs(offset),
                    ra + offset,
                    dec,
                    f"same FK4 Dec as {parent}; RA offset {offset:+g} deg",
                )
            )
        for offset in cfg.dec_offsets_deg:
            sign = "p" if offset > 0 else "m"
            name = f"{parent}_off_dec_{sign}{abs(offset):g}"
            rows.append(
                _control_row(
                    source,
                    name,
                    "same_ra_dec_offset",
                    abs(offset),
                    ra,
                    dec + offset,
                    f"same FK4 RA as {parent}; Dec offset {offset:+g} deg",
                )
            )
        n = int(cfg.annulus_positions)
        radius = float(cfg.annulus_radius_deg)
        cos_dec = max(math.cos(math.radians(dec)), 0.1)
        for idx in range(n):
            theta = 2.0 * math.pi * idx / n
            dra = radius * math.cos(theta) / cos_dec
            ddec = radius * math.sin(theta)
            name = f"{parent}_off_annulus_{idx:02d}"
            rows.append(
                _control_row(
                    source,
                    name,
                    "local_annulus",
                    radius,
                    ra + dra,
                    dec + ddec,
                    f"FK4 local annulus around {parent}; radius {radius:g} deg; index {idx}",
                )
            )
    out = pd.DataFrame.from_records(rows)
    return out.sort_values(["parent_source", "control_type", "control_name"]).reset_index(drop=True) if not out.empty else out

