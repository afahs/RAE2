#!/usr/bin/env python3
"""
RAE2 simulation: compute upper/lower V receiver power vs time and frequency.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import healpy as hp
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError(
        "healpy is required for skymap IO. Activate luseepy_env or install healpy."
    ) from exc


MOON_RADIUS_KM = 1737.4


@dataclass
class BeamPattern:
    theta_deg: np.ndarray
    e_plane: np.ndarray
    h_plane: np.ndarray
    in_db: bool = False


def _parse_freqs(freqs: str) -> List[float]:
    parts = [p.strip() for p in freqs.split(",") if p.strip()]
    if not parts:
        raise ValueError("No frequencies provided.")
    return [float(p) for p in parts]


def read_positions_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["time", "x_km", "y_km", "z_km", "ex", "ey", "ez"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Positions CSV missing columns: {missing}")
    return df


def read_beam_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    if "theta_deg" not in df.columns or "gain" not in df.columns:
        raise ValueError(f"Beam CSV {path} must have columns: theta_deg, gain")
    return df["theta_deg"].to_numpy(), df["gain"].to_numpy()


def default_dipole_pattern(theta_deg: np.ndarray) -> np.ndarray:
    theta_rad = np.deg2rad(theta_deg)
    return np.sin(theta_rad) ** 2


def build_beam_pattern(
    e_plane_path: Optional[Path],
    h_plane_path: Optional[Path],
    in_db: bool,
) -> BeamPattern:
    if e_plane_path is None or h_plane_path is None:
        # Default dipole if no beam is given (user instruction).
        theta_deg = np.linspace(0.0, 180.0, 721)
        pattern = default_dipole_pattern(theta_deg)
        return BeamPattern(theta_deg=theta_deg, e_plane=pattern, h_plane=pattern, in_db=False)

    e_theta, e_gain = read_beam_csv(e_plane_path)
    h_theta, h_gain = read_beam_csv(h_plane_path)
    if not np.allclose(e_theta, h_theta):
        raise ValueError("E-plane and H-plane theta grids must match.")
    return BeamPattern(theta_deg=e_theta, e_plane=e_gain, h_plane=h_gain, in_db=in_db)


def _interp_gain(theta_deg: np.ndarray, beam: BeamPattern) -> Tuple[np.ndarray, np.ndarray]:
    e_gain = np.interp(theta_deg, beam.theta_deg, beam.e_plane, left=beam.e_plane[0], right=beam.e_plane[-1])
    h_gain = np.interp(theta_deg, beam.theta_deg, beam.h_plane, left=beam.h_plane[0], right=beam.h_plane[-1])
    if beam.in_db:
        e_gain = 10 ** (e_gain / 10.0)
        h_gain = 10 ** (h_gain / 10.0)
    return e_gain, h_gain


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    norm = np.where(norm == 0, 1.0, norm)
    return vec / norm


def _earth_reference_axis(axis: np.ndarray, earth_vec: np.ndarray) -> np.ndarray:
    # Use Earth direction to define the E-plane reference (instruction: Earth unit vector provided).
    ref = earth_vec - np.dot(earth_vec, axis) * axis
    if np.linalg.norm(ref) < 1e-12:
        # Fallback reference if Earth aligns with axis.
        fallback = np.array([1.0, 0.0, 0.0])
        ref = fallback - np.dot(fallback, axis) * axis
    return _normalize(ref)


def _phi_angle(axis: np.ndarray, ref: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    # Compute azimuth around axis with ref as phi=0 plane.
    proj = dirs - np.outer(np.dot(dirs, axis), axis)
    proj = _normalize(proj)
    x = np.dot(proj, ref)
    y = np.dot(proj, np.cross(axis, ref))
    return np.arctan2(y, x)


def _occultation_mask(dirs: np.ndarray, r_hat: np.ndarray, radius_km: float, range_km: float) -> np.ndarray:
    # Moon occultation: directions within the Moon disk about -r_hat are blocked (user instruction).
    alpha = math.asin(min(1.0, radius_km / range_km))
    cos_alpha = math.cos(alpha)
    cos_angle = np.dot(dirs, -r_hat)
    return cos_angle >= cos_alpha


def antenna_temperature(
    skymap: np.ndarray,
    dirs: np.ndarray,
    d_omega: float,
    axis: np.ndarray,
    earth_vec: np.ndarray,
    beam: BeamPattern,
    occult_mask: np.ndarray,
) -> float:
    axis = _normalize(axis)
    earth_vec = _normalize(earth_vec)
    ref = _earth_reference_axis(axis, earth_vec)
    theta = np.arccos(np.clip(np.dot(dirs, axis), -1.0, 1.0))
    theta_deg = np.rad2deg(theta)
    phi = _phi_angle(axis, ref, dirs)

    e_gain, h_gain = _interp_gain(theta_deg, beam)
    # Combine E/H planes into a 3D pattern using a standard weighted model.
    beam_power = (e_gain * np.cos(phi)) ** 2 + (h_gain * np.sin(phi)) ** 2

    visible = ~occult_mask
    if not np.any(visible):
        return 0.0
    beam_visible = beam_power[visible]
    sky_visible = skymap[visible]
    numerator = np.sum(sky_visible * beam_visible) * d_omega
    denom = np.sum(beam_visible) * d_omega
    if denom == 0.0:
        return 0.0
    return numerator / denom


def simulate(
    positions: pd.DataFrame,
    skymaps: Sequence[np.ndarray],
    freqs_hz: Sequence[float],
    beam: BeamPattern,
    system_temperature_k: float,
    skymap_paths: Optional[Sequence[str]] = None,
    beam_e_plane_path: Optional[str] = None,
    beam_h_plane_path: Optional[str] = None,
    moon_radius_km: float = MOON_RADIUS_KM,
) -> pd.DataFrame:
    if len(freqs_hz) != len(skymaps):
        raise ValueError("Number of freqs must match number of skymaps (or reuse a single map).")

    nside = hp.get_nside(skymaps[0])
    npix = hp.nside2npix(nside)
    dirs = np.array(hp.pix2vec(nside, np.arange(npix))).T
    d_omega = 4.0 * math.pi / npix

    rows = []
    for _, row in positions.iterrows():
        r_vec = np.array([row["x_km"], row["y_km"], row["z_km"]], dtype=float)
        earth_vec = np.array([row["ex"], row["ey"], row["ez"]], dtype=float)
        range_km = np.linalg.norm(r_vec)
        if range_km == 0.0:
            raise ValueError("Spacecraft position cannot be at the Moon center.")
        r_hat = r_vec / range_km

        # Gravity-gradient stabilized: lower-V points to the Moon, upper-V points away (instruction).
        lower_axis = -r_hat
        upper_axis = r_hat
        occult_mask = _occultation_mask(dirs, r_hat, moon_radius_km, range_km)

        for idx, (skymap, freq_hz) in enumerate(zip(skymaps, freqs_hz)):
            skymap_path = skymap_paths[idx] if skymap_paths else ""
            t_upper = antenna_temperature(
                skymap=skymap,
                dirs=dirs,
                d_omega=d_omega,
                axis=upper_axis,
                earth_vec=earth_vec,
                beam=beam,
                occult_mask=occult_mask,
            )
            t_lower = antenna_temperature(
                skymap=skymap,
                dirs=dirs,
                d_omega=d_omega,
                axis=lower_axis,
                earth_vec=earth_vec,
                beam=beam,
                occult_mask=occult_mask,
            )

            rows.append(
                {
                    "time": row["time"],
                    "freq_hz": freq_hz,
                    "power_upper_k": t_upper + system_temperature_k,
                    "power_lower_k": t_lower + system_temperature_k,
                    "x_km": row["x_km"],
                    "y_km": row["y_km"],
                    "z_km": row["z_km"],
                    "ex": row["ex"],
                    "ey": row["ey"],
                    "ez": row["ez"],
                    "system_temperature_k": system_temperature_k,
                    "beam_model": "dipole" if (beam.e_plane is beam.h_plane) else "e_h_plane",
                    "beam_e_plane_csv": beam_e_plane_path or "",
                    "beam_h_plane_csv": beam_h_plane_path or "",
                    "skymap_path": skymap_path,
                    "moon_occultation": True,
                }
            )

    return pd.DataFrame(rows)


def _load_skymaps(paths: Sequence[Path]) -> List[np.ndarray]:
    maps = []
    for path in paths:
        maps.append(hp.read_map(path, verbose=False))
    return maps


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Simulate RAE2 upper/lower V power.")
    parser.add_argument("--positions", required=True, type=Path, help="Positions CSV.")
    parser.add_argument("--skymap", required=True, action="append", type=Path, help="Healpy skymap FITS.")
    parser.add_argument("--freqs", required=True, type=str, help="Comma-separated list of frequencies in Hz.")
    parser.add_argument("--output", required=True, type=Path, help="Output CSV.")
    parser.add_argument("--e-plane", type=Path, default=None, help="E-plane CSV (theta_deg,gain).")
    parser.add_argument("--h-plane", type=Path, default=None, help="H-plane CSV (theta_deg,gain).")
    parser.add_argument("--beam-db", action="store_true", help="Interpret beam gain values as dB.")
    parser.add_argument("--system-temperature", type=float, default=0.0, help="System temperature in K.")
    args = parser.parse_args(argv)

    positions = read_positions_csv(args.positions)
    freqs_hz = _parse_freqs(args.freqs)

    skymaps = _load_skymaps(args.skymap)
    skymap_paths = [str(path) for path in args.skymap]
    if len(skymaps) == 1 and len(freqs_hz) > 1:
        skymaps = skymaps * len(freqs_hz)
        skymap_paths = skymap_paths * len(freqs_hz)
    if len(skymaps) != len(freqs_hz):
        raise ValueError("Provide one skymap per frequency, or a single map for all frequencies.")

    beam = build_beam_pattern(args.e_plane, args.h_plane, args.beam_db)
    out_df = simulate(
        positions=positions,
        skymaps=skymaps,
        freqs_hz=freqs_hz,
        beam=beam,
        system_temperature_k=args.system_temperature,
        skymap_paths=skymap_paths,
        beam_e_plane_path=str(args.e_plane) if args.e_plane else None,
        beam_h_plane_path=str(args.h_plane) if args.h_plane else None,
    )
    out_df.to_csv(args.output, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
