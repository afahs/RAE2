"""Celestial-frame handling for RAE-2 occultation analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import FK4, FK5, ICRS, CartesianRepresentation, SkyCoord, get_body_barycentric, solar_system_ephemeris
from astropy.time import Time

from .constants import DEFAULT_EQUINOX, DEFAULT_FRAME
from .geometry import normalize_vectors


def frame_object(frame: str = DEFAULT_FRAME, equinox: str = DEFAULT_EQUINOX):
    key = frame.lower()
    if key == "fk4":
        return FK4(equinox=Time(equinox))
    if key == "fk5":
        return FK5(equinox=Time("J2000" if equinox.upper().startswith("B") else equinox))
    if key == "icrs":
        return ICRS()
    raise ValueError(f"Unsupported celestial frame: {frame}")


def fixed_source_unit_vector(
    ra_deg: float,
    dec_deg: float,
    source_frame: str = DEFAULT_FRAME,
    target_frame: str = DEFAULT_FRAME,
    equinox: str = DEFAULT_EQUINOX,
) -> np.ndarray:
    """Return a fixed source unit vector in the target inertial frame.

    RAE-2 legacy source catalogs are commonly B1950/FK4. This function treats
    the input frame explicitly and defaults to FK4 B1950.
    """
    coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame=frame_object(source_frame, equinox))
    transformed = coord.transform_to(frame_object(target_frame, equinox))
    vec = np.array([transformed.cartesian.x.value, transformed.cartesian.y.value, transformed.cartesian.z.value])
    return vec / np.linalg.norm(vec)


def repeated_unit_vector(vec: np.ndarray, n_samples: int) -> np.ndarray:
    unit = np.asarray(vec, dtype=float).reshape(1, 3)
    unit = unit / np.linalg.norm(unit)
    return np.repeat(unit, int(n_samples), axis=0)


def offset_unit_vectors_tangent(
    unit_vectors: np.ndarray,
    east_offset_deg: float = 0.0,
    north_offset_deg: float = 0.0,
) -> np.ndarray:
    """Offset moving-source vectors along an exact great-circle displacement.

    ``east_offset_deg`` and ``north_offset_deg`` define a direction and angular
    distance in the local tangent basis.  The tangent vector is mapped back to
    the unit sphere with the spherical exponential map, so large control
    offsets retain their requested great-circle separation.
    """
    uvec = normalize_vectors(unit_vectors)
    out = np.full_like(uvec, np.nan, dtype=float)
    east_rad = np.deg2rad(float(east_offset_deg))
    north_rad = np.deg2rad(float(north_offset_deg))
    offset_rad = float(np.hypot(east_rad, north_rad))
    if not np.isfinite(offset_rad):
        raise ValueError("moving-source offsets must be finite")
    z_axis = np.array([0.0, 0.0, 1.0])
    x_axis = np.array([1.0, 0.0, 0.0])
    for idx, vec in enumerate(uvec):
        if not np.isfinite(vec).all():
            continue
        east = np.cross(z_axis, vec)
        if np.linalg.norm(east) < 1e-10:
            east = np.cross(x_axis, vec)
        east = east / np.linalg.norm(east)
        north = np.cross(vec, east)
        north = north / np.linalg.norm(north)
        if offset_rad == 0.0:
            out[idx] = vec
            continue
        tangent_direction = (east_rad * east + north_rad * north) / offset_rad
        shifted = np.cos(offset_rad) * vec + np.sin(offset_rad) * tangent_direction
        out[idx] = shifted / np.linalg.norm(shifted)
    return out


def body_positions_from_moon(
    body_name: str,
    times: pd.Series | pd.DatetimeIndex,
    target_frame: str = DEFAULT_FRAME,
    equinox: str = DEFAULT_EQUINOX,
    ephemeris: str = "builtin",
) -> np.ndarray:
    """Geometric Moon-to-body positions in km in the requested axes.

    Preserve distance until after subtracting the Moon-centered spacecraft
    position. Archived Earth unit vectors are deliberately not inputs. NaT
    rows stay unknown; repeated epochs are evaluated only once.
    """
    time_index = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
    out = np.full((len(time_index), 3), np.nan)
    valid = ~time_index.isna()
    if not valid.any():
        return out
    epochs, inverse = np.unique(time_index[valid].as_unit("ns").asi8, return_inverse=True)
    t_ast = Time(epochs / 1e9, format="unix", scale="utc")
    with solar_system_ephemeris.set(ephemeris):
        moon = get_body_barycentric("moon", t_ast)
        body = get_body_barycentric(body_name.lower(), t_ast)
    vec_km = (body.xyz - moon.xyz).to_value(u.km).T

    if target_frame.lower() == "icrs":
        positions = vec_km
    else:
        rep = CartesianRepresentation(vec_km[:, 0] * u.km, vec_km[:, 1] * u.km, vec_km[:, 2] * u.km)
        coords = SkyCoord(rep, frame="icrs", obstime=t_ast)
        transformed = coords.transform_to(frame_object(target_frame, equinox))
        positions = transformed.cartesian.xyz.to_value(u.km).T
    out[valid] = positions[inverse]
    return out


def body_unit_vectors_from_moon(
    body_name: str, times: pd.Series | pd.DatetimeIndex,
    target_frame: str = DEFAULT_FRAME, equinox: str = DEFAULT_EQUINOX,
    ephemeris: str = "builtin",
) -> np.ndarray:
    """Moon-centered directions; use the observer helper for occultations."""
    return normalize_vectors(body_positions_from_moon(body_name, times, target_frame, equinox, ephemeris))


def body_vectors_from_spacecraft(
    body_name: str, times: pd.Series | pd.DatetimeIndex,
    spacecraft_position_km: np.ndarray, target_frame: str = DEFAULT_FRAME,
    equinox: str = DEFAULT_EQUINOX, ephemeris: str = "builtin",
) -> np.ndarray:
    """Spacecraft-to-body vectors in km; spacecraft must use target_frame."""
    spacecraft = np.asarray(spacecraft_position_km, dtype=float)
    if spacecraft.shape != (len(times), 3):
        raise ValueError("spacecraft positions must have shape (number of times, 3)")
    return body_positions_from_moon(body_name, times, target_frame, equinox, ephemeris) - spacecraft


def body_unit_vectors_from_spacecraft(
    body_name: str, times: pd.Series | pd.DatetimeIndex,
    spacecraft_position_km: np.ndarray, target_frame: str = DEFAULT_FRAME,
    equinox: str = DEFAULT_EQUINOX, ephemeris: str = "builtin",
) -> np.ndarray:
    return normalize_vectors(body_vectors_from_spacecraft(
        body_name, times, spacecraft_position_km, target_frame, equinox, ephemeris))
