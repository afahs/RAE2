#!/usr/bin/env python
"""Audit lower-V beam-offset direction against local geometry/aspect columns."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from rylevonberg.table_io import read_table

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.constants import SPACECRAFT_COLUMNS  # noqa: E402
from rylevonberg.geometry import normalize_vectors  # noqa: E402
from rylevonberg.util import ensure_dir, software_versions, write_json  # noqa: E402
from scripts.build_all_frequency_occultation_profile_grids import CLEAN  # noqa: E402
from scripts.build_lower_v_occultation_contrast_maps import lower_v_beam_axes  # noqa: E402


DEFAULT_OUT = ROOT / "outputs/lower_v_differential_occultation_maps_allbands_15deg_v1/beam_offset_audit"


def _aspect_vectors_from_ra_hours_dec_deg(df: pd.DataFrame) -> np.ndarray:
    ra_deg = pd.to_numeric(df["right_ascension"], errors="coerce").to_numpy(dtype=float) * 15.0
    dec_deg = pd.to_numeric(df["declination"], errors="coerce").to_numpy(dtype=float)
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    return normalize_vectors(np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)]))


def _angular_sep_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = normalize_vectors(a)
    bb = normalize_vectors(b)
    return np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", aa, bb), -1.0, 1.0)))


def _tangent_direction(from_axis: np.ndarray, to_axis: np.ndarray) -> np.ndarray:
    base = normalize_vectors(from_axis)
    target = normalize_vectors(to_axis)
    return normalize_vectors(target - np.einsum("ij,ij->i", target, base)[:, None] * base)


def _velocity_vectors(times: pd.Series, positions: np.ndarray) -> np.ndarray:
    t = pd.DatetimeIndex(times).asi8.astype(float) / 1e9
    pos = np.asarray(positions, dtype=float)
    vel = np.full_like(pos, np.nan, dtype=float)
    if len(pos) < 2:
        return vel
    vel[0] = (pos[1] - pos[0]) / (t[1] - t[0])
    vel[-1] = (pos[-1] - pos[-2]) / (t[-1] - t[-2])
    dt = t[2:] - t[:-2]
    good = np.isfinite(dt) & (dt > 0)
    vel[1:-1][good] = (pos[2:][good] - pos[:-2][good]) / dt[good, None]
    return vel


def _project_tangent(vectors: np.ndarray, axes: np.ndarray) -> np.ndarray:
    axis = normalize_vectors(axes)
    vec = np.asarray(vectors, dtype=float)
    return normalize_vectors(vec - np.einsum("ij,ij->i", vec, axis)[:, None] * axis)


def _summary_stats(name: str, values: np.ndarray) -> dict[str, float | str | int]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return {
        "quantity": name,
        "n": int(arr.size),
        "median": float(np.nanmedian(arr)) if arr.size else np.nan,
        "p05": float(np.nanpercentile(arr, 5)) if arr.size else np.nan,
        "p95": float(np.nanpercentile(arr, 95)) if arr.size else np.nan,
    }


def _markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)

    def fmt(value: object) -> str:
        if isinstance(value, (float, np.floating)):
            if not np.isfinite(value):
                return ""
            if abs(value) >= 100:
                return f"{value:.0f}"
            if abs(value) >= 10:
                return f"{value:.2f}"
            return f"{value:.3g}"
        return str(value)

    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> Path:
    out_dir = ensure_dir(args.out_dir)
    usecols = ["time", *SPACECRAFT_COLUMNS, "right_ascension", "declination"]
    frames = []
    for chunk in read_table(args.clean, usecols=usecols, chunksize=750_000, low_memory=False):
        chunk["time"] = pd.to_datetime(chunk["time"], errors="coerce")
        chunk = chunk.dropna(subset=["time"]).drop_duplicates("time")
        frames.append(chunk)
    df = pd.concat(frames, ignore_index=True).drop_duplicates("time").sort_values("time").reset_index(drop=True)
    if args.sample_stride > 1:
        df = df.iloc[:: int(args.sample_stride)].reset_index(drop=True)

    pos = df[SPACECRAFT_COLUMNS].to_numpy(dtype=float)
    position_axis = normalize_vectors(pos)
    moon_axis = normalize_vectors(-pos)
    aspect_axis = _aspect_vectors_from_ra_hours_dec_deg(df)
    lower_aspect_axis = -aspect_axis
    velocity = _velocity_vectors(df["time"], pos)

    aspect_offset = _angular_sep_deg(position_axis, aspect_axis)
    lower_aspect_offset = _angular_sep_deg(moon_axis, lower_aspect_axis)

    aspect_tangent = _tangent_direction(position_axis, aspect_axis)
    vel_tangent_about_position = _project_tangent(velocity, position_axis)
    dot_aspect_velocity = np.einsum("ij,ij->i", aspect_tangent, vel_tangent_about_position)
    dot_aspect_anti_velocity = np.einsum("ij,ij->i", aspect_tangent, -vel_tangent_about_position)

    rows_for_axes = df[["time", *SPACECRAFT_COLUMNS]].copy()
    beam_anti = lower_v_beam_axes(rows_for_axes, moon_axis, float(args.offset_deg), "anti_velocity")
    beam_vel = lower_v_beam_axes(rows_for_axes, moon_axis, float(args.offset_deg), "velocity")
    anti_to_lower_aspect = _angular_sep_deg(beam_anti, lower_aspect_axis)
    vel_to_lower_aspect = _angular_sep_deg(beam_vel, lower_aspect_axis)
    anti_vs_vel = _angular_sep_deg(beam_anti, beam_vel)

    summary = pd.DataFrame(
        [
            _summary_stats("position_to_ra_hours_dec_angle_deg", aspect_offset),
            _summary_stats("moon_axis_to_minus_ra_hours_dec_angle_deg", lower_aspect_offset),
            _summary_stats("aspect_offset_tangent_dot_velocity", dot_aspect_velocity),
            _summary_stats("aspect_offset_tangent_dot_anti_velocity", dot_aspect_anti_velocity),
            _summary_stats(f"anti_velocity_{args.offset_deg:g}deg_beam_to_minus_aspect_angle_deg", anti_to_lower_aspect),
            _summary_stats(f"velocity_{args.offset_deg:g}deg_beam_to_minus_aspect_angle_deg", vel_to_lower_aspect),
            _summary_stats(f"anti_vs_velocity_{args.offset_deg:g}deg_beam_axis_angle_deg", anti_vs_vel),
        ]
    )
    summary.to_csv(out_dir / "lower_v_beam_offset_direction_summary.csv", index=False)
    config = {
        "clean": str(args.clean),
        "out_dir": str(out_dir),
        "sample_stride": int(args.sample_stride),
        "offset_deg": float(args.offset_deg),
        "software_versions": software_versions(),
    }
    write_json(out_dir / "run_config.json", config)
    lines = [
        "# Lower-V Beam Offset Direction Audit",
        "",
        "This audit uses the cleaned `position_x/y/z` vectors, finite-difference orbital velocity, and the historical RA-hours/Dec aspect columns.",
        "It does not assume the RA/Dec aspect columns are the Moon occultation geometry; they are used only as an independent direction clue.",
        "",
        "## Summary",
        "",
        _markdown_table(summary),
        "",
        "## Interpretation",
        "",
        "- The RA column must be interpreted as hours. With that convention, the RA/Dec aspect direction is about 9 deg from the `+position` direction.",
        "- The tangent direction from `+position` toward RA/Dec has a stronger anti-velocity than velocity component.",
        "- If the lower-V beam is treated as the opposite of that RA/Dec aspect direction, the 7 deg velocity-side beam hypothesis is closer to `-RA/Dec` than the anti-velocity hypothesis.",
        "- Because the boom figure does not define the complete body-frame sign convention, the map builder exposes both `velocity` and `anti_velocity` offset modes.",
        "",
    ]
    report = out_dir / "lower_v_beam_offset_direction_audit.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", default=str(CLEAN))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--sample-stride", type=int, default=500)
    parser.add_argument("--offset-deg", type=float, default=7.0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
