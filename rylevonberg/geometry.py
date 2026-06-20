"""Spacecraft/Moon geometry, limb visibility, and transition finding."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import EARTH_MOON_DISTANCE_KM, EARTH_RADIUS_KM, MOON_RADIUS_KM


def normalize_vectors(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(vectors, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    out = np.full(arr.shape, np.nan, dtype=float)
    good = np.isfinite(arr).all(axis=1) & (norms[:, 0] >= eps)
    out[good] = arr[good] / norms[good]
    return out


def angular_separation_deg(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    a = normalize_vectors(v1)
    b = normalize_vectors(v2)
    dots = np.einsum("ij,ij->i", a, b)
    return np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))


def moon_center_direction(spacecraft_position_km: np.ndarray) -> np.ndarray:
    return normalize_vectors(-np.asarray(spacecraft_position_km, dtype=float))


def moon_angular_radius_deg(spacecraft_position_km: np.ndarray, moon_radius_km: float = MOON_RADIUS_KM) -> np.ndarray:
    dist = np.linalg.norm(np.asarray(spacecraft_position_km, dtype=float), axis=1)
    return np.degrees(np.arcsin(np.clip(float(moon_radius_km) / np.maximum(dist, 1e-12), 0.0, 1.0)))


def moon_limb_angle_deg(
    spacecraft_position_km: np.ndarray,
    target_unit_vectors: np.ndarray,
    moon_radius_km: float = MOON_RADIUS_KM,
) -> np.ndarray:
    """Signed angular distance from the lunar limb.

    Positive means visible/outside the lunar disk. Negative means occulted.
    """
    center = moon_center_direction(spacecraft_position_km)
    sep = angular_separation_deg(center, target_unit_vectors)
    return sep - moon_angular_radius_deg(spacecraft_position_km, moon_radius_km)


def visible_by_moon(spacecraft_position_km: np.ndarray, target_unit_vectors: np.ndarray) -> np.ndarray:
    return moon_limb_angle_deg(spacecraft_position_km, target_unit_vectors) >= 0.0


def earth_direction_from_spacecraft(
    spacecraft_position_km: np.ndarray,
    earth_unit_vector_from_moon: np.ndarray,
    earth_moon_distance_km: float = EARTH_MOON_DISTANCE_KM,
) -> np.ndarray:
    moon_to_earth = normalize_vectors(earth_unit_vector_from_moon) * float(earth_moon_distance_km)
    return normalize_vectors(moon_to_earth - np.asarray(spacecraft_position_km, dtype=float))


def earth_limb_angle_deg(
    spacecraft_position_km: np.ndarray,
    earth_unit_vector_from_moon: np.ndarray,
    target_unit_vectors: np.ndarray,
    earth_radius_km: float = EARTH_RADIUS_KM,
    earth_moon_distance_km: float = EARTH_MOON_DISTANCE_KM,
) -> np.ndarray:
    earth_dir = earth_direction_from_spacecraft(spacecraft_position_km, earth_unit_vector_from_moon, earth_moon_distance_km)
    moon_to_earth = normalize_vectors(earth_unit_vector_from_moon) * float(earth_moon_distance_km)
    sc_to_earth = moon_to_earth - np.asarray(spacecraft_position_km, dtype=float)
    radius = np.degrees(np.arcsin(np.clip(float(earth_radius_km) / np.linalg.norm(sc_to_earth, axis=1), 0.0, 1.0)))
    return angular_separation_deg(earth_dir, target_unit_vectors) - radius


def linear_zero_crossing_time(t0: pd.Timestamp, t1: pd.Timestamp, y0: float, y1: float) -> pd.Timestamp:
    if not np.isfinite(y0) or not np.isfinite(y1) or y0 == y1:
        return t0 + (t1 - t0) / 2
    frac = float(np.clip(-y0 / (y1 - y0), 0.0, 1.0))
    return t0 + pd.to_timedelta(frac * (t1 - t0).total_seconds(), unit="s")


def find_limb_transitions(
    times: pd.Series | pd.DatetimeIndex,
    limb_angle_deg: np.ndarray,
    max_gap_seconds: float | None = None,
) -> pd.DataFrame:
    """Find disappearance/reappearance events from a signed limb-angle series."""
    time_index = pd.DatetimeIndex(times)
    limb = np.asarray(limb_angle_deg, dtype=float)
    if len(time_index) != len(limb):
        raise ValueError("times and limb_angle_deg length mismatch")
    if len(limb) < 2:
        return pd.DataFrame()

    visible = limb >= 0.0
    rows = []
    for idx in np.where(visible[1:] != visible[:-1])[0] + 1:
        t0 = pd.Timestamp(time_index[idx - 1])
        t1 = pd.Timestamp(time_index[idx])
        gap_s = float((t1 - t0).total_seconds())
        if max_gap_seconds is not None and gap_s > float(max_gap_seconds):
            continue
        event_type = "disappearance" if visible[idx - 1] and not visible[idx] else "reappearance"
        event_time = linear_zero_crossing_time(t0, t1, float(limb[idx - 1]), float(limb[idx]))
        rows.append(
            {
                "event_type": event_type,
                "pre_idx": int(idx - 1),
                "post_idx": int(idx),
                "pre_time": t0,
                "post_time": t1,
                "predicted_event_time": event_time,
                "gap_seconds": gap_s,
                "pre_limb_angle_deg": float(limb[idx - 1]),
                "post_limb_angle_deg": float(limb[idx]),
            }
        )
    return pd.DataFrame.from_records(rows)

