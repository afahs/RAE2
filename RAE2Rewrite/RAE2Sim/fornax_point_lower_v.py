#!/usr/bin/env python3
"""
Generate a lower-V-only RAE2 test CSV for an empty sky with a single
Fornax-A point source (B1950) at amplitude 1, per instructions.
"""

from __future__ import annotations

import argparse
import math
from collections import deque
from pathlib import Path

import healpy as hp
import numpy as np
import pandas as pd
from astropy.coordinates import FK4, SkyCoord


MOON_RADIUS_KM = 1737.4
DEFAULT_DATA_PATH = "/global/cfs/projectdirs/m4895/RAE2Data/interpolatedRAE2MasterFile.csv"
DEFAULT_START = "1974-09-07 14:00"
DEFAULT_END = "1975-06-27 16:00"
DEFAULT_WINDOW_DAYS = 14


def fornax_a_b1950_uvec() -> np.ndarray:
    # Per instruction: Fornax-A position in B1950 (from ICRS -> FK4 B1950).
    coord = SkyCoord(ra="03h22m41.7s", dec="-37d12m30s", frame="icrs")
    coord_b1950 = coord.transform_to(FK4(equinox="B1950"))
    vec = coord_b1950.cartesian.xyz.to_value()
    return vec / np.linalg.norm(vec)


def build_fornax_map(nside: int, amplitude: float) -> np.ndarray:
    # Per instruction: point source at Fornax-A (B1950) on an otherwise empty map.
    npix = hp.nside2npix(nside)
    sky_map = np.zeros(npix, dtype=np.float64)
    source_uvec = fornax_a_b1950_uvec()
    theta, phi = hp.vec2ang(source_uvec)
    ipix = hp.ang2pix(nside, theta, phi)
    sky_map[ipix] += amplitude
    return sky_map


def compute_occulted_flags(pos_km: np.ndarray, source_uvec: np.ndarray) -> np.ndarray:
    # Per instruction: Moon occultation is the disk around -r_hat.
    range_km = np.linalg.norm(pos_km, axis=1)
    valid = range_km > 0.0
    r_hat = np.zeros_like(pos_km, dtype=np.float64)
    r_hat[valid] = pos_km[valid] / range_km[valid, None]
    cos_theta = np.zeros(range_km.shape[0], dtype=np.float64)
    cos_theta[valid] = -np.einsum("ij,j->i", r_hat[valid], source_uvec)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    ratio = np.zeros_like(range_km, dtype=np.float64)
    ratio[valid] = np.clip(MOON_RADIUS_KM / range_km[valid], 0.0, 1.0)
    cos_moon = np.sqrt(1.0 - ratio ** 2)
    blocked = cos_theta > cos_moon
    blocked[~valid] = True
    return blocked


def find_window(
    path: Path,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    window_days: int,
    chunk_size: int,
    source_uvec: np.ndarray,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    # Per instruction: auto-select a 2-week window with at least one occultation.
    window = np.timedelta64(window_days, "D")
    start_np = np.datetime64(start_time)
    end_np = np.datetime64(end_time)
    window_deque: deque[tuple[np.datetime64, bool]] = deque()
    blocked_count = 0
    visible_count = 0

    usecols = ["time", "position_x", "position_y", "position_z"]
    for chunk in pd.read_csv(
        path,
        usecols=usecols,
        chunksize=chunk_size,
        parse_dates=["time"],
    ):
        chunk = chunk[(chunk["time"] >= start_time) & (chunk["time"] <= end_time)]
        if chunk.empty:
            continue

        times = chunk["time"].to_numpy()
        pos = chunk[["position_x", "position_y", "position_z"]].to_numpy(dtype=np.float64)
        blocked = compute_occulted_flags(pos, source_uvec)

        for t, b in zip(times, blocked):
            window_deque.append((t, bool(b)))
            if b:
                blocked_count += 1
            else:
                visible_count += 1

            cutoff = t - window
            while window_deque and window_deque[0][0] < cutoff:
                _, b0 = window_deque.popleft()
                if b0:
                    blocked_count -= 1
                else:
                    visible_count -= 1

            if cutoff < start_np or t > end_np:
                continue
            if blocked_count > 0 and visible_count > 0:
                return pd.to_datetime(cutoff), pd.to_datetime(t)

    raise RuntimeError("Could not find a 2-week window with both occulted and visible Fornax-A.")


def integrate_lower_v(
    pos_km: np.ndarray,
    sky_map: np.ndarray,
    dirs: np.ndarray,
    d_omega: float,
) -> np.ndarray:
    # Per instruction: full healpy map integration (lower-V dipole, Moon occultation).
    powers = np.zeros(pos_km.shape[0], dtype=np.float64)
    for i in range(pos_km.shape[0]):
        range_km = np.linalg.norm(pos_km[i])
        if range_km <= 0.0:
            continue
        r_hat = pos_km[i] / range_km
        axis = -r_hat  # lower-V points to the Moon, per instruction
        cos_theta = dirs @ axis
        sin2 = 1.0 - cos_theta ** 2
        sin2 = np.clip(sin2, 0.0, None)
        beam_power = sin2 ** 2  # dipole power pattern ~ sin^4(theta)

        ratio = min(1.0, MOON_RADIUS_KM / range_km)
        cos_moon = math.sqrt(1.0 - ratio ** 2)
        visible = cos_theta < cos_moon
        if not np.any(visible):
            continue

        numerator = np.sum(sky_map[visible] * beam_power[visible]) * d_omega
        denom = np.sum(beam_power[visible]) * d_omega
        powers[i] = 0.0 if denom == 0.0 else numerator / denom

    return powers.astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Fornax-A lower-V test data.")
    parser.add_argument("--input", type=Path, default=Path(DEFAULT_DATA_PATH))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--frequency-band", type=int, default=1)
    parser.add_argument("--nside", type=int, default=32)
    parser.add_argument("--amplitude", type=float, default=1.0)
    parser.add_argument("--window-days", type=int, default=DEFAULT_WINDOW_DAYS)
    parser.add_argument("--chunk-size", type=int, default=200000)
    args = parser.parse_args()

    start_time = pd.to_datetime(args.start)
    end_time = pd.to_datetime(args.end)

    source_uvec = fornax_a_b1950_uvec()
    window_start, window_end = find_window(
        args.input,
        start_time,
        end_time,
        args.window_days,
        args.chunk_size,
        source_uvec,
    )
    print(f"Selected window: {window_start} to {window_end}")

    sky_map = build_fornax_map(args.nside, args.amplitude)
    npix = hp.nside2npix(args.nside)
    dirs = np.array(hp.pix2vec(args.nside, np.arange(npix))).T
    d_omega = 4.0 * math.pi / npix

    usecols = [
        "time",
        "position_x",
        "position_y",
        "position_z",
        "earth_unit_vector_x",
        "earth_unit_vector_y",
        "earth_unit_vector_z",
    ]

    wrote_header = False
    for chunk in pd.read_csv(
        args.input,
        usecols=usecols,
        chunksize=args.chunk_size,
        parse_dates=["time"],
    ):
        chunk = chunk[(chunk["time"] >= window_start) & (chunk["time"] <= window_end)]
        if chunk.empty:
            continue

        pos = chunk[["position_x", "position_y", "position_z"]].to_numpy(dtype=np.float64)
        power = integrate_lower_v(pos, sky_map, dirs, d_omega)

        out = chunk.copy()
        out["frequency_band"] = int(args.frequency_band)
        out["rv2_coarse"] = power

        out.to_csv(args.output, mode="a", header=not wrote_header, index=False)
        wrote_header = True

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
