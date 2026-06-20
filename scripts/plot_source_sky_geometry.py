#!/usr/bin/env python
"""Plot source positions with Galactic and ecliptic reference geometry."""

from __future__ import annotations

from pathlib import Path
import sys

import astropy.units as u
from astropy.coordinates import CartesianRepresentation
from astropy.coordinates import BarycentricTrueEcliptic, FK4, Galactic, SkyCoord
from astropy.time import Time
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rylevonberg.table_io import read_table

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.sources import load_source_list  # noqa: E402
from rylevonberg.constants import EARTH_UNIT_COLUMNS, SPACECRAFT_COLUMNS  # noqa: E402
from rylevonberg.events import source_vectors_for_rows  # noqa: E402
from rylevonberg.frames import body_unit_vectors_from_moon  # noqa: E402
from rylevonberg.util import ensure_dir, software_versions, write_json  # noqa: E402


OUT = ROOT / "outputs/source_sky_geometry_v1"
SOURCE_LIST = ROOT / "configs/bright_sources.csv"
CLEAN = ROOT / "outputs/control_survey_earth_sun_postnov1974_v2/01_ingest/cleaned_timeseries.csv"

SOURCE_COLORS = {
    "fornax_a": "#9467bd",
    "cas_a": "#8c564b",
    "cyg_a": "#17becf",
    "tau_a": "#2ca02c",
    "vir_a": "#bcbd22",
    "sgr_a": "#e377c2",
    "galactic_center": "black",
}


def _wrap_lon_rad(lon_deg: np.ndarray | float) -> np.ndarray:
    lon = np.asarray(lon_deg, dtype=float)
    wrapped = ((lon + 180.0) % 360.0) - 180.0
    return np.deg2rad(-wrapped)


def _fixed_source_rows(src: pd.DataFrame) -> pd.DataFrame:
    rows = []
    fixed = src[src["kind"].astype(str).str.lower().eq("fixed")].copy()
    for _, row in fixed.iterrows():
        coord = SkyCoord(
            ra=float(row["ra_deg"]) * u.deg,
            dec=float(row["dec_deg"]) * u.deg,
            frame=FK4(equinox=Time("B1950")),
        )
        gal = coord.galactic
        ecl = coord.transform_to(BarycentricTrueEcliptic(equinox=Time("B1950")))
        rows.append(
            {
                "source_name": row["source_name"],
                "ra_fk4_deg": float(row["ra_deg"]),
                "dec_fk4_deg": float(row["dec_deg"]),
                "ecliptic_lon_deg": float(ecl.lon.deg),
                "ecliptic_lat_deg": float(ecl.lat.deg),
                "galactic_l_deg": float(gal.l.deg),
                "galactic_b_deg": float(gal.b.deg),
                "abs_ecliptic_lat_deg": abs(float(ecl.lat.deg)),
                "abs_galactic_lat_deg": abs(float(gal.b.deg)),
                "notes": row.get("notes", ""),
            }
        )
    return pd.DataFrame(rows).sort_values("abs_ecliptic_lat_deg")


def _vector_to_galactic_lonlat(vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rep = CartesianRepresentation(vec[:, 0] * u.one, vec[:, 1] * u.one, vec[:, 2] * u.one)
    coord = SkyCoord(rep, frame=FK4(equinox=Time("B1950")))
    gal = coord.galactic
    return np.asarray(gal.l.deg, dtype=float), np.asarray(gal.b.deg, dtype=float)


def _sample_geometry_track_rows(cadence_days: float = 3.0) -> pd.DataFrame:
    usecols = ["time", *SPACECRAFT_COLUMNS, *EARTH_UNIT_COLUMNS]
    rows = []
    last_time = None
    step = pd.Timedelta(days=float(cadence_days))
    for chunk in read_table(CLEAN, usecols=usecols, chunksize=750_000, low_memory=False):
        chunk["time"] = pd.to_datetime(chunk["time"], errors="coerce")
        chunk = chunk[chunk["time"].notna()].drop_duplicates("time").sort_values("time")
        for _, row in chunk.iterrows():
            t = row["time"]
            if last_time is None or t - last_time >= step:
                rows.append(row)
                last_time = t
    if not rows:
        return pd.DataFrame(columns=usecols)
    return pd.DataFrame(rows).sort_values("time").reset_index(drop=True)


def _moving_body_tracks() -> pd.DataFrame:
    base = _sample_geometry_track_rows()
    if base.empty:
        return pd.DataFrame()
    times = pd.DatetimeIndex(base["time"])
    earth_vec = source_vectors_for_rows(pd.Series({"source_name": "earth", "kind": "earth", "frame": "fk4"}), times, base)
    sun_vec = body_unit_vectors_from_moon("sun", times, target_frame="fk4", equinox="B1950", ephemeris="builtin")
    rows = []
    for name, vec in [("earth_track", earth_vec), ("sun_track", sun_vec)]:
        lon, lat = _vector_to_galactic_lonlat(vec)
        for t, l, b in zip(times, lon, lat):
            rows.append({"track_name": name, "time": t, "galactic_l_deg": float(l), "galactic_b_deg": float(b)})
    return pd.DataFrame(rows)


def _ecliptic_lat_curve(beta_deg: float, n: int = 720) -> tuple[np.ndarray, np.ndarray]:
    lon = np.linspace(0.0, 360.0, n)
    lat = np.full_like(lon, float(beta_deg))
    coord = SkyCoord(
        lon=lon * u.deg,
        lat=lat * u.deg,
        frame=BarycentricTrueEcliptic(equinox=Time("B1950")),
    )
    gal = coord.transform_to(Galactic())
    x = _wrap_lon_rad(gal.l.deg)
    y = np.deg2rad(gal.b.deg)
    jumps = np.where(np.abs(np.diff(x)) > np.pi)[0]
    if jumps.size:
        x = x.copy()
        y = y.copy()
        x[jumps + 1] = np.nan
        y[jumps + 1] = np.nan
    return x, y


def _galactic_lat_curve(b_deg: float, n: int = 720) -> tuple[np.ndarray, np.ndarray]:
    lon = np.linspace(-180.0, 180.0, n)
    return np.deg2rad(-lon), np.full_like(lon, np.deg2rad(float(b_deg)))


def _plot_track_segments(ax, track: pd.DataFrame, color: str, label: str) -> None:
    if track.empty:
        return
    data = track.sort_values("time")
    x = _wrap_lon_rad(data["galactic_l_deg"].to_numpy(dtype=float))
    y = np.deg2rad(data["galactic_b_deg"].to_numpy(dtype=float))
    jumps = np.where(np.abs(np.diff(x)) > np.pi)[0]
    x = x.copy()
    y = y.copy()
    if jumps.size:
        x[jumps + 1] = np.nan
        y[jumps + 1] = np.nan
    ax.plot(x, y, color=color, lw=1.8, alpha=0.9, label=label, zorder=3)
    if np.isfinite(x).any():
        ax.scatter(x[np.isfinite(x)][0], y[np.isfinite(y)][0], color=color, s=28, marker=">", zorder=4)


def _plot_aitoff(rows: pd.DataFrame, tracks: pd.DataFrame, out_dir: Path) -> Path:
    fig = plt.figure(figsize=(14, 7.4))
    ax = fig.add_subplot(111, projection="aitoff")
    ax.grid(True, alpha=0.32)

    for b, color, lw, alpha, label in [
        (0.0, "0.12", 1.4, 0.9, "Galactic plane"),
        (10.0, "0.55", 0.8, 0.55, "Galactic +/-10 deg"),
        (-10.0, "0.55", 0.8, 0.55, None),
        (30.0, "0.72", 0.65, 0.45, "Galactic +/-30 deg"),
        (-30.0, "0.72", 0.65, 0.45, None),
    ]:
        x, y = _galactic_lat_curve(b)
        ax.plot(x, y, color=color, lw=lw, alpha=alpha, ls="-" if b == 0 else ":", label=label)

    for beta, color, lw, ls, label in [
        (0.0, "#d62728", 1.6, "-", "Ecliptic plane"),
        (5.0, "#d62728", 0.9, "--", "Ecliptic +/-5 deg"),
        (-5.0, "#d62728", 0.9, "--", None),
        (10.0, "#ff9896", 0.9, "--", "Ecliptic +/-10 deg"),
        (-10.0, "#ff9896", 0.9, "--", None),
        (15.0, "#f7b6b2", 0.85, "--", "Ecliptic +/-15 deg"),
        (-15.0, "#f7b6b2", 0.85, "--", None),
    ]:
        x, y = _ecliptic_lat_curve(beta)
        ax.plot(x, y, color=color, lw=lw, ls=ls, alpha=0.9, label=label)

    if not tracks.empty:
        _plot_track_segments(ax, tracks[tracks["track_name"].eq("earth_track")], "#1f77b4", "Earth apparent track")
        _plot_track_segments(ax, tracks[tracks["track_name"].eq("sun_track")], "#ff7f0e", "Sun track")

    for _, row in rows.iterrows():
        name = str(row["source_name"])
        x = _wrap_lon_rad(row["galactic_l_deg"])
        y = np.deg2rad(row["galactic_b_deg"])
        color = SOURCE_COLORS.get(name, "#333333")
        marker = "*" if name in {"galactic_center", "sgr_a"} else "o"
        size = 95 if marker == "*" else 62
        ax.scatter(x, y, s=size, color=color, marker=marker, edgecolor="white", linewidth=0.7, zorder=5)
        ax.text(x + 0.035, y + 0.025, name, fontsize=8, color=color, zorder=6)

    ax.set_xticklabels(["150", "120", "90", "60", "30", "0", "330", "300", "270", "240", "210"])
    ax.set_title("Fixed source positions with Sun/Earth tracks and Galactic/ecliptic reference bands", pad=20)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.19), ncols=4, fontsize=8, frameon=False)
    path = out_dir / "source_positions_galactic_aitoff.png"
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_latitude_scatter(rows: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    ax.axhline(0.0, color="0.25", lw=1.0, label="Galactic plane")
    ax.axvline(0.0, color="#d62728", lw=1.0, label="Ecliptic plane")
    for value, color, label in [(5, "#d62728", "+/-5 deg"), (10, "#ff9896", "+/-10 deg"), (15, "#f7b6b2", "+/-15 deg")]:
        ax.axvline(value, color=color, ls="--", lw=0.8, alpha=0.8)
        ax.axvline(-value, color=color, ls="--", lw=0.8, alpha=0.8, label=label)
    for value in [10, 30, 60]:
        ax.axhline(value, color="0.65", ls=":", lw=0.8)
        ax.axhline(-value, color="0.65", ls=":", lw=0.8)
    for _, row in rows.iterrows():
        name = str(row["source_name"])
        ax.scatter(row["ecliptic_lat_deg"], row["galactic_b_deg"], s=70, color=SOURCE_COLORS.get(name, "#333333"))
        ax.text(row["ecliptic_lat_deg"] + 1.2, row["galactic_b_deg"] + 1.2, name, fontsize=8)
    ax.set_xlim(-70, 70)
    ax.set_ylim(-85, 85)
    ax.set_xlabel("Ecliptic latitude beta (deg)")
    ax.set_ylabel("Galactic latitude b (deg)")
    ax.set_title("Fixed-source ecliptic vs Galactic latitude")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    path = out_dir / "source_ecliptic_vs_galactic_latitude.png"
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _write_report(rows: pd.DataFrame, tracks: pd.DataFrame, paths: list[Path], out_dir: Path) -> Path:
    near_ecl = rows[rows["abs_ecliptic_lat_deg"].le(15.0)]
    near_gal = rows[rows["abs_galactic_lat_deg"].le(10.0)]
    lines = [
        "# Source Sky Geometry Visualization",
        "",
        "Source positions are fixed-source FK4/B1950 coordinates transformed to Galactic and B1950 true ecliptic coordinates.",
        "Sun and Earth tracks are sampled from the post-November-1974 cleaned Ryle-Vonberg geometry at about 3 day cadence.",
        "The Earth track uses the pipeline spacecraft-to-Earth direction; the Sun track uses the FK4/B1950 Moon-to-Sun ephemeris direction.",
        "",
        "## Fixed Source Coordinates",
        "",
        rows[
            [
                "source_name",
                "ra_fk4_deg",
                "dec_fk4_deg",
                "ecliptic_lon_deg",
                "ecliptic_lat_deg",
                "galactic_l_deg",
                "galactic_b_deg",
            ]
        ].to_string(index=False),
        "",
        "## Within +/-15 deg Of The Ecliptic",
        "",
        near_ecl[["source_name", "ecliptic_lat_deg", "galactic_b_deg"]].to_string(index=False) if not near_ecl.empty else "None.",
        "",
        "## Within +/-10 deg Of The Galactic Plane",
        "",
        near_gal[["source_name", "ecliptic_lat_deg", "galactic_b_deg"]].to_string(index=False) if not near_gal.empty else "None.",
        "",
        "## Track Samples",
        "",
        tracks.groupby("track_name").agg(
            n_samples=("time", "size"),
            start_time=("time", "min"),
            end_time=("time", "max"),
            min_galactic_b_deg=("galactic_b_deg", "min"),
            max_galactic_b_deg=("galactic_b_deg", "max"),
        ).to_string() if not tracks.empty else "No track samples generated.",
        "",
        "## Generated Figures",
        "",
    ]
    lines.extend(f"- `{p}`" for p in paths)
    path = out_dir / "source_sky_geometry_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    out_dir = ensure_dir(OUT)
    write_json(out_dir / "run_config.json", {"source_list": str(SOURCE_LIST), "software_versions": software_versions()})
    rows = _fixed_source_rows(load_source_list(SOURCE_LIST))
    tracks = _moving_body_tracks()
    rows.to_csv(out_dir / "fixed_source_sky_coordinates.csv", index=False)
    tracks.to_csv(out_dir / "sun_earth_track_samples.csv", index=False)
    paths = [_plot_aitoff(rows, tracks, out_dir), _plot_latitude_scatter(rows, out_dir)]
    report = _write_report(rows, tracks, paths, out_dir)
    print(report)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
