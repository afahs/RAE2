#!/usr/bin/env python
"""Build lower-V differential occultation maps.

This script treats each event window as a differential occultation
measurement.  The design row is not stamped onto the crossing source alone:

    A[e, p] = median_post(beam[p, t] * visible[p, t])
              - median_pre(beam[p, t] * visible[p, t])

Cells disappearing behind one limb therefore get negative coefficients while
simultaneously reappearing cells on the opposite limb get positive
coefficients in the same measurement row.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rylevonberg.table_io import read_table
from astropy.coordinates import FK4, Galactic, SkyCoord
from astropy.time import Time
import astropy.units as u

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.constants import EARTH_UNIT_COLUMNS, FREQUENCY_MAP_MHZ, SPACECRAFT_COLUMNS  # noqa: E402
from rylevonberg.events import contaminant_limb_angles, spacecraft_positions  # noqa: E402
from rylevonberg.frames import fixed_source_unit_vector, repeated_unit_vector  # noqa: E402
from rylevonberg.geometry import find_limb_transitions, moon_angular_radius_deg, moon_center_direction, moon_limb_angle_deg, normalize_vectors  # noqa: E402
from rylevonberg.util import datetime_ns, ensure_dir, robust_sigma, software_versions, write_json  # noqa: E402
from scripts.build_all_frequency_occultation_profile_grids import CLEAN  # noqa: E402


LOWER_V = "rv2_coarse"
OUT = ROOT / "outputs/lower_v_differential_occultation_maps_allbands_15deg_v1"
BEAM_DIR = Path(os.environ.get("RAE2_ANTENNA_DIGITIZATION_DIR", "data/antennaDigitization"))
BEAM_SPECS = [
    (1.31, BEAM_DIR / "eplane_1p31MHz.csv", BEAM_DIR / "hplane_1p31MHz.csv"),
    (3.93, BEAM_DIR / "eplane_3p93MHz.csv", BEAM_DIR / "hplane_3p93MHz.csv"),
    (6.55, BEAM_DIR / "eplane_6p55MHz.csv", BEAM_DIR / "hplane_6p55MHz.csv"),
]
PRE_RANGE = (-180.0, -60.0)
POST_RANGE = (60.0, 180.0)


def parse_frequencies(value: str) -> list[int]:
    text = str(value).strip().lower()
    if text in {"all", "*"}:
        return sorted(FREQUENCY_MAP_MHZ)
    out = []
    reverse = {float(v): int(k) for k, v in FREQUENCY_MAP_MHZ.items()}
    for part in text.split(","):
        item = part.strip()
        if not item:
            continue
        if "." in item:
            freq = float(item)
            if freq not in reverse:
                raise ValueError(f"unsupported frequency MHz: {freq}")
            out.append(reverse[freq])
        else:
            band = int(item)
            if band not in FREQUENCY_MAP_MHZ:
                raise ValueError(f"unsupported frequency band: {band}")
            out.append(band)
    return sorted(set(out))


def _coord_tag(value: float, prefix: str) -> str:
    sign = "m" if value < 0 else "p"
    return f"{prefix}{sign}{abs(value):05.1f}".replace(".", "p")


def build_galactic_grid(step_deg: float) -> pd.DataFrame:
    if step_deg <= 0 or 360.0 % step_deg > 1e-9 or 180.0 % step_deg > 1e-9:
        raise ValueError("grid step must evenly divide both 360 and 180 degrees")
    n_lon = int(round(360.0 / step_deg))
    n_lat = int(round(180.0 / step_deg))
    lon_centers = np.arange(n_lon, dtype=float) * step_deg + 0.5 * step_deg
    lat_centers = -90.0 + np.arange(n_lat, dtype=float) * step_deg + 0.5 * step_deg
    rows = []
    for lat_idx, lat in enumerate(lat_centers):
        for lon_idx, lon in enumerate(lon_centers):
            gal = SkyCoord(l=lon * u.deg, b=lat * u.deg, frame=Galactic())
            fk4 = gal.transform_to(FK4(equinox=Time("B1950")))
            pixel_index = lat_idx * n_lon + lon_idx
            rows.append(
                {
                    "source_name": f"gal_{_coord_tag(lon, 'l')}_{_coord_tag(lat, 'b')}",
                    "kind": "fixed",
                    "body_name": "",
                    "ra_deg": float(fk4.ra.deg),
                    "dec_deg": float(fk4.dec.deg),
                    "frame": "fk4",
                    "galactic_l_deg": float(lon),
                    "galactic_b_deg": float(lat),
                    "lon_index": int(lon_idx),
                    "lat_index": int(lat_idx),
                    "pixel_index": int(pixel_index),
                }
            )
    return pd.DataFrame.from_records(rows)


def grid_unit_vectors(grid: pd.DataFrame) -> np.ndarray:
    ra = np.deg2rad(pd.to_numeric(grid["ra_deg"], errors="coerce").to_numpy(dtype=float))
    dec = np.deg2rad(pd.to_numeric(grid["dec_deg"], errors="coerce").to_numpy(dtype=float))
    return np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def _load_prediction_geometry() -> pd.DataFrame:
    usecols = ["time", *SPACECRAFT_COLUMNS, *EARTH_UNIT_COLUMNS]
    frames = []
    for chunk in read_table(CLEAN, usecols=usecols, chunksize=750_000, low_memory=False):
        chunk["time"] = pd.to_datetime(chunk["time"], errors="coerce")
        chunk = chunk[chunk["time"].notna()]
        frames.append(chunk.drop_duplicates("time"))
    if not frames:
        return pd.DataFrame(columns=usecols)
    return pd.concat(frames, ignore_index=True).drop_duplicates("time").sort_values("time").reset_index(drop=True)


def _limb_exclusion_sources() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_name": "earth",
                "kind": "earth",
                "body_name": "earth",
                "frame": "fk4",
                "ra_deg": np.nan,
                "dec_deg": np.nan,
            },
            {
                "source_name": "sun",
                "kind": "body",
                "body_name": "sun",
                "frame": "fk4",
                "ra_deg": np.nan,
                "dec_deg": np.nan,
            },
        ]
    )


def _downsample_prediction_grid(clean_geom: pd.DataFrame, cadence_s: float) -> pd.DataFrame:
    base = clean_geom.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    if cadence_s <= 0 or len(base) <= 2:
        return base
    times = pd.DatetimeIndex(base["time"])
    elapsed = (datetime_ns(times) - pd.Timestamp(times[0]).value).astype(float) / 1e9
    bucket = np.floor(elapsed / float(cadence_s)).astype(np.int64)
    keep = np.r_[True, bucket[1:] != bucket[:-1]]
    keep[-1] = True
    return base.loc[keep].reset_index(drop=True)


def _interpolate_transition_value(times: pd.DatetimeIndex, values: np.ndarray, transition: pd.Series) -> float:
    pre_idx = int(transition["pre_idx"])
    post_idx = int(transition["post_idx"])
    if pre_idx < 0 or post_idx >= len(times):
        return np.nan
    t0 = times[pre_idx].value
    t1 = times[post_idx].value
    te = pd.Timestamp(transition["predicted_event_time"]).value
    if t1 == t0:
        return float(values[post_idx])
    frac = float((te - t0) / (t1 - t0))
    return float(values[pre_idx] + frac * (values[post_idx] - values[pre_idx]))


def _load_lower_v_band(band: int) -> pd.DataFrame:
    usecols = ["time", "frequency_band", "antenna", "power", "is_valid", *SPACECRAFT_COLUMNS]
    frames = []
    for chunk in read_table(CLEAN, usecols=usecols, chunksize=750_000, low_memory=False):
        mask = chunk["antenna"].astype(str).eq(LOWER_V) & chunk["frequency_band"].astype(int).eq(int(band))
        if not mask.any():
            continue
        sub = chunk.loc[mask].copy()
        sub["time"] = pd.to_datetime(sub["time"], errors="coerce")
        sub["power"] = pd.to_numeric(sub["power"], errors="coerce")
        if "is_valid" in sub.columns:
            if sub["is_valid"].dtype != bool:
                sub["is_valid"] = sub["is_valid"].astype(str).str.lower().isin(["true", "1", "yes"])
        else:
            sub["is_valid"] = np.isfinite(sub["power"])
        sub = sub[sub["time"].notna()]
        frames.append(sub)
    if not frames:
        return pd.DataFrame(columns=usecols)
    out = pd.concat(frames, ignore_index=True).sort_values("time").reset_index(drop=True)
    out["power"] = pd.to_numeric(out["power"], errors="coerce")
    return out


def _nearest_beam(freq_mhz: float) -> tuple[float, Path, Path]:
    return min(BEAM_SPECS, key=lambda spec: abs(spec[0] - float(freq_mhz)))


def _load_beam_mean(freq_mhz: float) -> tuple[float, np.ndarray, np.ndarray, Path, Path]:
    beam_freq, eplane, hplane = _nearest_beam(freq_mhz)
    e = read_table(eplane)
    h = read_table(hplane)
    angles = e["angle_deg"].to_numpy(dtype=float)
    gain = 0.5 * (
        10.0 ** (e["gain_dB"].to_numpy(dtype=float) / 10.0)
        + 10.0 ** (h["gain_dB"].to_numpy(dtype=float) / 10.0)
    )
    finite = gain[np.isfinite(gain)]
    max_gain = float(np.nanmax(finite)) if finite.size else np.nan
    if np.isfinite(max_gain) and max_gain > 0:
        gain = gain / max_gain
    return float(beam_freq), angles, gain, eplane, hplane


def _interp_cyclic(angles: np.ndarray, values: np.ndarray, angle_deg: np.ndarray) -> np.ndarray:
    a = np.asarray(angle_deg, dtype=float) % 360.0
    x = np.concatenate(([angles[-1] - 360.0], angles, [angles[0] + 360.0]))
    y = np.concatenate(([values[-1]], values, [values[0]]))
    return np.interp(a, x, y)


def estimate_velocity_vectors(times: pd.Series | pd.DatetimeIndex, positions: np.ndarray) -> np.ndarray:
    """Estimate spacecraft velocity vectors from adjacent position samples."""
    pos = np.asarray(positions, dtype=float)
    if len(pos) == 0:
        return np.empty_like(pos)
    out = np.full_like(pos, np.nan, dtype=float)
    time_ns = datetime_ns(times).astype(float)
    if len(pos) == 1:
        return out
    for idx in range(len(pos)):
        if idx == 0:
            lo, hi = 0, 1
        elif idx == len(pos) - 1:
            lo, hi = len(pos) - 2, len(pos) - 1
        else:
            lo, hi = idx - 1, idx + 1
        dt = (time_ns[hi] - time_ns[lo]) / 1e9
        if not np.isfinite(dt) or dt <= 0:
            continue
        out[idx] = (pos[hi] - pos[lo]) / dt
    return out


def tangent_unit_vectors(vectors: np.ndarray, axes: np.ndarray) -> np.ndarray:
    """Project vectors into the plane perpendicular to axes and normalize."""
    vec = np.asarray(vectors, dtype=float)
    axis = normalize_vectors(axes)
    projection = vec - np.einsum("ij,ij->i", vec, axis)[:, None] * axis
    return normalize_vectors(projection)


def offset_axes_toward_tangent(axes: np.ndarray, tangent: np.ndarray, offset_deg: float) -> np.ndarray:
    """Rotate axes by offset_deg toward a tangent-plane direction."""
    axis = normalize_vectors(axes)
    tan = normalize_vectors(tangent)
    theta = np.deg2rad(float(offset_deg))
    out = np.full_like(axis, np.nan, dtype=float)
    good = np.isfinite(axis).all(axis=1) & np.isfinite(tan).all(axis=1)
    out[good] = normalize_vectors(np.cos(theta) * axis[good] + np.sin(theta) * tan[good])
    return out


def lower_v_beam_axes(
    band_rows: pd.DataFrame,
    moon_axes: np.ndarray,
    offset_deg: float,
    offset_direction: str,
) -> np.ndarray:
    """Return lower-V beam axes, optionally offset from Moon center.

    The lunar occultation disk is still centered on ``moon_axes``.  This only
    changes the beam-gain direction used to weight sky cells.
    """
    direction = str(offset_direction).lower()
    if abs(float(offset_deg)) <= 0.0 or direction in {"none", "moon_center"}:
        return moon_axes
    pos = band_rows[SPACECRAFT_COLUMNS].to_numpy(dtype=float)
    vel = estimate_velocity_vectors(pd.DatetimeIndex(band_rows["time"]), pos)
    velocity_tangent = tangent_unit_vectors(vel, moon_axes)
    if direction == "anti_velocity":
        tangent = -velocity_tangent
    elif direction == "velocity":
        tangent = velocity_tangent
    else:
        raise ValueError(f"unsupported beam offset direction: {offset_direction}")
    axes = offset_axes_toward_tangent(moon_axes, tangent, float(offset_deg))
    bad = ~np.isfinite(axes).all(axis=1)
    if bad.any():
        axes[bad] = moon_axes[bad]
    return axes


def beam_visible_matrix(
    band_rows: pd.DataFrame,
    pixel_vectors: np.ndarray,
    beam_angles: np.ndarray,
    beam_gains: np.ndarray,
    beam_offset_deg: float = 0.0,
    beam_offset_direction: str = "none",
    model_mode: str = "beam_weighted",
    chunk_size: int = 256,
) -> np.ndarray:
    mode = str(model_mode).lower()
    if mode not in {"beam_weighted", "ring_only"}:
        raise ValueError(f"unsupported model mode: {model_mode}")
    pos = band_rows[SPACECRAFT_COLUMNS].to_numpy(dtype=float)
    moon_axes = normalize_vectors(-pos)
    if mode == "beam_weighted":
        beam_axes = lower_v_beam_axes(band_rows, moon_axes, float(beam_offset_deg), beam_offset_direction)
    else:
        beam_axes = moon_axes
    radius = moon_angular_radius_deg(pos)
    out = np.full((len(band_rows), len(pixel_vectors)), np.nan, dtype=np.float32)
    for start in range(0, len(band_rows), chunk_size):
        stop = min(start + chunk_size, len(band_rows))
        moon_block = moon_axes[start:stop]
        beam_block = beam_axes[start:stop]
        good = np.isfinite(moon_block).all(axis=1) & np.isfinite(beam_block).all(axis=1)
        if not good.any():
            continue
        moon_dots = np.clip(moon_block[good] @ pixel_vectors.T, -1.0, 1.0)
        moon_sep = np.degrees(np.arccos(moon_dots))
        visible = moon_sep >= radius[start:stop][good, None]
        if mode == "ring_only":
            modeled = visible.astype(np.float32)
        else:
            beam_dots = np.clip(beam_block[good] @ pixel_vectors.T, -1.0, 1.0)
            beam_sep = np.degrees(np.arccos(beam_dots))
            gain = _interp_cyclic(beam_angles, beam_gains, beam_sep)
            modeled = (gain * visible).astype(np.float32)
        out_block = out[start:stop]
        out_block[good] = modeled
        out[start:stop] = out_block
    return out


def design_row_from_model(pre_model: np.ndarray, post_model: np.ndarray) -> np.ndarray:
    if pre_model.size == 0 or post_model.size == 0:
        return np.array([], dtype=float)
    return np.nanmedian(post_model, axis=0) - np.nanmedian(pre_model, axis=0)


def _robust_event_z(power: np.ndarray, t_rel: np.ndarray, inner_s: float) -> tuple[np.ndarray, float, float] | None:
    side = np.isfinite(power) & np.isfinite(t_rel) & (np.abs(t_rel) >= float(inner_s))
    if np.count_nonzero(side) < 6:
        return None
    center = float(np.nanmedian(power[side]))
    scale = robust_sigma(power[side] - center)
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.nanstd(power[side], ddof=1))
    if not np.isfinite(scale) or scale <= 0:
        return None
    return (power - center) / scale, center, scale


def _sample_events(events: pd.DataFrame, max_measurements: int, seed: int) -> pd.DataFrame:
    if max_measurements <= 0 or len(events) <= max_measurements:
        return events
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(events.index.to_numpy(), size=max_measurements, replace=False))
    return events.loc[idx].copy()


def collect_measurements_for_band(
    band_rows: pd.DataFrame,
    events: pd.DataFrame,
    model_matrix: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    times_ns = datetime_ns(band_rows["time"])
    power = pd.to_numeric(band_rows["power"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(power)
    if "is_valid" in band_rows.columns:
        valid &= band_rows["is_valid"].to_numpy(dtype=bool)

    work = events.sort_values("predicted_event_time").reset_index(drop=True)
    work = _sample_events(work, int(args.max_measurements), seed)
    rows = []
    design_rows = []
    y_values = []
    weights = []
    half_ns = int(float(args.window_s) * 1e9)
    min_side = int(args.min_side_samples)
    min_norm = int(args.min_normalization_samples)
    measurement_id = 0
    for ev in work.itertuples(index=False):
        event_time = pd.Timestamp(ev.predicted_event_time)
        event_ns = event_time.value
        lo = int(np.searchsorted(times_ns, event_ns - half_ns, side="left"))
        hi = int(np.searchsorted(times_ns, event_ns + half_ns, side="right"))
        if hi <= lo:
            continue
        t_rel = (times_ns[lo:hi] - event_ns).astype(float) / 1e9
        keep = valid[lo:hi] & (np.abs(t_rel) <= float(args.window_s))
        if np.count_nonzero(keep) < max(min_norm, 8):
            continue
        local_idx = np.arange(lo, hi)[keep]
        local_t = t_rel[keep]
        local_power = power[lo:hi][keep]
        pre = (local_t >= PRE_RANGE[0]) & (local_t <= PRE_RANGE[1])
        post = (local_t >= POST_RANGE[0]) & (local_t <= POST_RANGE[1])
        if np.count_nonzero(pre) < min_side or np.count_nonzero(post) < min_side:
            continue
        z_result = _robust_event_z(local_power, local_t, float(args.inner_s))
        if z_result is None:
            continue
        z_power, norm_center, norm_scale = z_result
        if np.count_nonzero(np.isfinite(z_power) & (np.abs(local_t) >= float(args.inner_s))) < min_norm:
            continue
        y = float(np.nanmedian(z_power[post]) - np.nanmedian(z_power[pre]))
        row = design_row_from_model(model_matrix[local_idx[pre]], model_matrix[local_idx[post]])
        if row.size == 0 or not np.isfinite(row).any():
            continue
        row = np.nan_to_num(row.astype(float), nan=0.0)
        row_norm = float(np.linalg.norm(row))
        if not np.isfinite(row_norm) or row_norm < float(args.min_design_norm):
            continue
        target_idx = int(ev.pixel_index)
        target_coeff = float(row[target_idx]) if 0 <= target_idx < len(row) else np.nan
        signs = np.sign(row)
        target_sign = np.sign(target_coeff) if np.isfinite(target_coeff) and target_coeff != 0 else 0.0
        if target_sign == 0:
            same_sum = np.nan
            opposite_sum = np.nan
            strongest_opposite = np.nan
        else:
            same = signs == target_sign
            opposite = signs == -target_sign
            same_sum = float(np.nansum(np.abs(row[same])))
            opposite_sum = float(np.nansum(np.abs(row[opposite])))
            strongest = np.abs(row[opposite])
            strongest_opposite = float(np.nanmax(strongest)) if strongest.size else np.nan
        n_pre = int(np.count_nonzero(pre))
        n_post = int(np.count_nonzero(post))
        weight = float(min(min(n_pre, n_post), int(args.max_weight)))
        rows.append(
            {
                "measurement_id": int(measurement_id),
                "source_name": ev.source_name,
                "target_pixel_index": target_idx,
                "galactic_l_deg": float(ev.galactic_l_deg),
                "galactic_b_deg": float(ev.galactic_b_deg),
                "event_type": ev.event_type,
                "predicted_event_time": event_time,
                "frequency_band": int(ev.frequency_band),
                "frequency_mhz": float(ev.frequency_mhz),
                "antenna": LOWER_V,
                "observed_post_minus_pre_z": y,
                "measurement_weight": weight,
                "n_pre_samples": n_pre,
                "n_post_samples": n_post,
                "normalization_center_raw_power": norm_center,
                "normalization_scale_raw_power": norm_scale,
                "target_design_coeff": target_coeff,
                "same_limb_abs_coeff_sum": same_sum,
                "opposite_limb_abs_coeff_sum": opposite_sum,
                "strongest_opposite_limb_coeff_abs": strongest_opposite,
                "design_row_norm": row_norm,
            }
        )
        design_rows.append(row)
        y_values.append(y)
        weights.append(weight)
        measurement_id += 1
    if not design_rows:
        return pd.DataFrame(), np.empty((0, model_matrix.shape[1])), np.array([]), np.array([])
    return pd.DataFrame.from_records(rows), np.vstack(design_rows), np.asarray(y_values), np.asarray(weights)


def build_smoothing_laplacian(n_lat: int, n_lon: int) -> np.ndarray:
    n_pix = n_lat * n_lon
    lap = np.zeros((n_pix, n_pix), dtype=float)

    def index(lat_idx: int, lon_idx: int) -> int:
        return lat_idx * n_lon + (lon_idx % n_lon)

    edges = []
    for lat_idx in range(n_lat):
        for lon_idx in range(n_lon):
            p = index(lat_idx, lon_idx)
            edges.append((p, index(lat_idx, lon_idx + 1)))
            if lat_idx + 1 < n_lat:
                edges.append((p, index(lat_idx + 1, lon_idx)))
    for a, b in edges:
        if a == b:
            continue
        lap[a, a] += 1.0
        lap[b, b] += 1.0
        lap[a, b] -= 1.0
        lap[b, a] -= 1.0
    return lap


def solve_relative_map(
    design: np.ndarray,
    observed: np.ndarray,
    weights: np.ndarray,
    laplacian: np.ndarray,
    smooth_lambda: float,
    ridge_lambda: float,
    mean_constraint_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    if design.size == 0 or observed.size == 0:
        n_pix = laplacian.shape[0]
        return np.full(n_pix, np.nan), np.full(n_pix, np.nan), np.array([]), {"rank": 0.0, "condition": np.nan}
    w = np.asarray(weights, dtype=float)
    w = np.where(np.isfinite(w) & (w > 0), w, 1.0)
    sqrt_w = np.sqrt(w)
    aw = design * sqrt_w[:, None]
    yw = observed * sqrt_w
    n_pix = design.shape[1]
    ones = np.ones(n_pix, dtype=float)
    hessian = (
        aw.T @ aw
        + float(smooth_lambda) * laplacian
        + float(ridge_lambda) * np.eye(n_pix)
        + float(mean_constraint_weight) * np.outer(ones, ones) / float(n_pix)
    )
    rhs = aw.T @ yw
    try:
        solution = np.linalg.solve(hessian, rhs)
    except np.linalg.LinAlgError:
        solution = np.linalg.lstsq(hessian, rhs, rcond=None)[0]
    solution = solution - float(np.nanmean(solution))
    residuals = observed - design @ solution
    dof = max(len(observed) - n_pix, 1)
    sigma2 = float(np.nansum(w * residuals**2) / dof)
    try:
        cov_diag = np.diag(np.linalg.inv(hessian))
        stderr = np.sqrt(np.maximum(cov_diag * sigma2, 0.0))
    except np.linalg.LinAlgError:
        stderr = np.full(n_pix, np.nan)
    stats = {
        "rank": float(np.linalg.matrix_rank(hessian)),
        "condition": float(np.linalg.cond(hessian)),
        "weighted_rmse": float(np.sqrt(np.nanmean(w * residuals**2))),
        "median_abs_residual": float(np.nanmedian(np.abs(residuals))),
        "map_mean_after_constraint": float(np.nanmean(solution)),
    }
    return solution, stderr, residuals, stats


def _coverage(design: np.ndarray, weights: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    if design.size == 0:
        return np.array([]), np.array([])
    w = np.asarray(weights, dtype=float)
    abs_design = np.abs(design)
    weighted_abs = np.nansum(abs_design * w[:, None], axis=0)
    count = np.sum(abs_design >= float(threshold), axis=0)
    return weighted_abs, count.astype(int)


def _residual_by_target(measurements: pd.DataFrame, residuals: np.ndarray) -> pd.DataFrame:
    if measurements.empty or residuals.size == 0:
        return pd.DataFrame()
    work = measurements[["frequency_band", "target_pixel_index"]].copy()
    work["fit_residual"] = residuals
    rows = []
    for keys, grp in work.groupby(["frequency_band", "target_pixel_index"], sort=True):
        vals = pd.to_numeric(grp["fit_residual"], errors="coerce").dropna()
        if vals.empty:
            continue
        rows.append(
            {
                "frequency_band": int(keys[0]),
                "pixel_index": int(keys[1]),
                "target_residual_median": float(vals.median()),
                "target_residual_n": int(len(vals)),
            }
        )
    return pd.DataFrame.from_records(rows)


def _plot_maps(
    map_table: pd.DataFrame,
    value_col: str,
    out_path: Path,
    title: str,
    cmap: str,
    symmetric: bool,
    mask_by_coverage: bool,
    min_coverage_count: int,
) -> Path:
    freqs = sorted(map_table["frequency_mhz"].dropna().unique())
    fig, axes = plt.subplots(3, 3, figsize=(15.5, 9.5), sharex=True, sharey=True)
    values = pd.to_numeric(map_table[value_col], errors="coerce")
    if mask_by_coverage:
        values = values.where(map_table["coverage_count"].ge(min_coverage_count))
    if symmetric:
        vmax = float(np.nanpercentile(np.abs(values), 95)) if values.notna().any() else 1.0
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
        vmin = -vmax
    else:
        vmin = float(np.nanpercentile(values, 5)) if values.notna().any() else 0.0
        vmax = float(np.nanpercentile(values, 95)) if values.notna().any() else 1.0
        if not np.isfinite(vmin):
            vmin = 0.0
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = vmin + 1.0
    image = None
    for ax, freq in zip(axes.ravel(), freqs):
        sub = map_table[np.isclose(map_table["frequency_mhz"], freq)].copy()
        vals = pd.to_numeric(sub[value_col], errors="coerce")
        if mask_by_coverage:
            vals = vals.where(sub["coverage_count"].ge(min_coverage_count))
        sub[value_col] = vals
        pivot = sub.pivot_table(index="lat_index", columns="lon_index", values=value_col, aggfunc="first")
        pivot = pivot.sort_index().sort_index(axis=1)
        arr = pivot.to_numpy(dtype=float)
        image = ax.imshow(arr, origin="lower", extent=[0, 360, -90, 90], aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"{freq:.2f} MHz", fontsize=9)
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.set_yticks([-60, -30, 0, 30, 60])
        ax.grid(alpha=0.16, color="white", linewidth=0.5)
    for ax in axes[-1, :]:
        ax.set_xlabel("Galactic longitude l (deg)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Galactic latitude b (deg)")
    for ax in axes.ravel()[len(freqs) :]:
        ax.axis("off")
    if image is not None:
        cbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.82, pad=0.012)
        cbar.set_label(value_col.replace("_", " "))
    fig.suptitle(title, y=0.995, fontsize=12)
    fig.subplots_adjust(left=0.06, right=0.89, bottom=0.08, top=0.91, wspace=0.12, hspace=0.28)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return out_path


def _write_report(out_dir: Path, config: dict, design_summary: pd.DataFrame, paths: list[Path]) -> None:
    lines = [
        "# Lower-V Differential Occultation Maps",
        "",
        "This run builds relative lower-V maps using differential lunar-occultation measurements.",
        "Each design row includes every grid cell, so simultaneous ingress on one lunar limb and egress on the other",
        "are represented in the same linear equation.",
        "",
        "## Measurement Model",
        "",
        "    y = median(post z-power) - median(pre z-power)",
        "    A[p] = median_post(beam[p,t] * visible_by_moon[p,t]) - median_pre(beam[p,t] * visible_by_moon[p,t])",
        "",
        "A disappearance-side cell has a negative coefficient when it becomes occulted.  A simultaneous opposite-limb",
        "reappearance-side cell has a positive coefficient.  The map is solved from all coefficients together rather",
        "than assigning a signed value only to the crossing cell.",
        "",
        "## Configuration",
        "",
        pd.Series(config).to_string(),
        "",
        "## Design / Fit Summary",
        "",
        design_summary.to_string(index=False) if not design_summary.empty else "No usable measurements.",
        "",
        "## Outputs",
        "",
    ]
    lines.extend(f"- `{path}`" for path in paths)
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- Map values are relative and mean-zero; this is not an absolute sky-temperature calibration.",
            "- `model_mode=beam_weighted` uses the normalized axisymmetric mean of available digitized E/H-plane cuts.",
            "- `model_mode=ring_only` ignores antenna gain and uses only lunar visibility changes around the limb.",
            "- When configured for beam-weighted runs, the lower-V beam axis is offset from the Moon-center direction; the lunar occultation disk remains centered on the Moon.",
            "- Residual maps are binned by the event's crossing grid cell and are diagnostic only.",
            "",
        ]
    )
    (out_dir / "lower_v_differential_occultation_map_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _predict_grid_events(grid: pd.DataFrame, bands: list[int], args: argparse.Namespace, out_dir: Path) -> pd.DataFrame:
    clean_geom = _load_prediction_geometry()
    base = _downsample_prediction_grid(clean_geom, float(args.prediction_cadence_seconds))
    times = pd.DatetimeIndex(base["time"])
    sc = spacecraft_positions(base)
    center = moon_center_direction(sc)
    moon_radius = moon_angular_radius_deg(sc)
    exclusion_limb = contaminant_limb_angles(
        base,
        _limb_exclusion_sources(),
        target_frame="fk4",
        equinox="B1950",
        ephemeris="builtin",
    )
    rows = []
    for source in grid.itertuples(index=False):
        vec = fixed_source_unit_vector(
            float(source.ra_deg),
            float(source.dec_deg),
            source_frame="fk4",
            target_frame="fk4",
            equinox="B1950",
        )
        target = repeated_unit_vector(vec, len(times))
        limb = moon_limb_angle_deg(sc, target)
        transitions = find_limb_transitions(times, limb, max_gap_seconds=float(args.max_gap_seconds))
        if transitions.empty:
            continue
        for event_id, ev in transitions.reset_index(drop=True).iterrows():
            exclusion_values = {
                name: _interpolate_transition_value(times, values, ev)
                for name, values in exclusion_limb.items()
            }
            finite_abs = {name: abs(value) for name, value in exclusion_values.items() if np.isfinite(value)}
            nearest_source = min(finite_abs, key=finite_abs.get) if finite_abs else ""
            nearest_abs = float(finite_abs[nearest_source]) if nearest_source else np.nan
            if np.isfinite(nearest_abs) and nearest_abs <= float(args.limb_exclusion_deg):
                continue
            idx = int(ev["post_idx"])
            exclusion_metadata = ";".join(f"{name}:{value:.6g}" for name, value in sorted(exclusion_values.items()))
            for band in bands:
                rows.append(
                    {
                        "event_id": int(event_id),
                        "source_name": source.source_name,
                        "source_ra_deg": float(source.ra_deg),
                        "source_dec_deg": float(source.dec_deg),
                        "frame": "fk4",
                        "event_type": ev["event_type"],
                        "predicted_event_time": ev["predicted_event_time"],
                        "frequency_band": int(band),
                        "frequency_mhz": float(FREQUENCY_MAP_MHZ[int(band)]),
                        "antenna": LOWER_V,
                        "limb_angle_deg": 0.0,
                        "pre_limb_angle_deg": ev["pre_limb_angle_deg"],
                        "post_limb_angle_deg": ev["post_limb_angle_deg"],
                        "moon_center_x": float(center[idx, 0]),
                        "moon_center_y": float(center[idx, 1]),
                        "moon_center_z": float(center[idx, 2]),
                        "moon_angular_radius_deg": float(moon_radius[idx]),
                        "gap_seconds": ev["gap_seconds"],
                        "limb_exclusion_deg": float(args.limb_exclusion_deg),
                        "limb_exclusion_nearest_source": nearest_source,
                        "limb_exclusion_nearest_abs_deg": nearest_abs,
                        "limb_exclusion_source_angles_deg": exclusion_metadata,
                        "quality_flags": "",
                        "galactic_l_deg": float(source.galactic_l_deg),
                        "galactic_b_deg": float(source.galactic_b_deg),
                        "lon_index": int(source.lon_index),
                        "lat_index": int(source.lat_index),
                        "pixel_index": int(source.pixel_index),
                    }
                )
    events = pd.DataFrame.from_records(rows)
    if bool(args.write_predicted_events):
        events.to_csv(out_dir / "galactic_grid_predicted_events.csv", index=False)
    return events


def run(args: argparse.Namespace) -> Path:
    out_dir = ensure_dir(args.out_dir)
    bands = parse_frequencies(args.frequencies)
    freqs = [FREQUENCY_MAP_MHZ[b] for b in bands]
    grid = build_galactic_grid(float(args.grid_step_deg))
    grid.to_csv(out_dir / "galactic_grid_sources.csv", index=False)
    n_lon = int(grid["lon_index"].max()) + 1
    n_lat = int(grid["lat_index"].max()) + 1
    pixel_vectors = grid_unit_vectors(grid)
    laplacian = build_smoothing_laplacian(n_lat, n_lon)
    config = {
        "antenna": LOWER_V,
        "frequencies": bands,
        "frequencies_mhz": freqs,
        "grid_step_deg": float(args.grid_step_deg),
        "n_grid_pixels": int(len(grid)),
        "window_s": float(args.window_s),
        "inner_s": float(args.inner_s),
        "pre_range_s": PRE_RANGE,
        "post_range_s": POST_RANGE,
        "limb_exclusion_deg": float(args.limb_exclusion_deg),
        "prediction_cadence_seconds": float(args.prediction_cadence_seconds),
        "max_gap_seconds": float(args.max_gap_seconds),
        "max_measurements": int(args.max_measurements),
        "smooth_lambda": float(args.smooth_lambda),
        "ridge_lambda": float(args.ridge_lambda),
        "mean_constraint_weight": float(args.mean_constraint_weight),
        "coverage_threshold": float(args.coverage_threshold),
        "min_coverage_count": int(args.min_coverage_count),
        "model_mode": str(args.model_mode),
        "write_predicted_events": bool(args.write_predicted_events),
        "beam_offset_deg": float(args.beam_offset_deg),
        "beam_offset_direction": str(args.beam_offset_direction),
        "software_versions": software_versions(),
    }
    write_json(out_dir / "run_config.json", config)

    events = _predict_grid_events(grid, bands, args, out_dir)
    measurement_frames = []
    map_frames = []
    summary_rows = []
    residual_frames = []
    for band in bands:
        freq = FREQUENCY_MAP_MHZ[band]
        band_events = events[events["frequency_band"].astype(int).eq(int(band))].copy()
        band_rows = _load_lower_v_band(band)
        if band_events.empty or band_rows.empty:
            summary_rows.append(
                {
                    "frequency_band": int(band),
                    "frequency_mhz": float(freq),
                    "n_measurements": 0,
                    "status": "no_events_or_no_data",
                }
            )
            continue
        if str(args.model_mode) == "ring_only":
            beam_freq = np.nan
            beam_angles = np.array([0.0, 180.0], dtype=float)
            beam_gains = np.ones(2, dtype=float)
            eplane = Path("")
            hplane = Path("")
        else:
            beam_freq, beam_angles, beam_gains, eplane, hplane = _load_beam_mean(freq)
        model = beam_visible_matrix(
            band_rows,
            pixel_vectors,
            beam_angles,
            beam_gains,
            beam_offset_deg=float(args.beam_offset_deg),
            beam_offset_direction=str(args.beam_offset_direction),
            model_mode=str(args.model_mode),
            chunk_size=int(args.model_chunk_size),
        )
        measurements, design, observed, weights = collect_measurements_for_band(
            band_rows,
            band_events,
            model,
            args,
            seed=int(args.seed) + int(band),
        )
        if measurements.empty:
            summary_rows.append(
                {
                    "frequency_band": int(band),
                    "frequency_mhz": float(freq),
                    "n_measurements": 0,
                    "status": "no_usable_measurements",
                    "model_mode": str(args.model_mode),
                    "beam_model_frequency_mhz": beam_freq,
                    "beam_offset_deg": float(args.beam_offset_deg),
                    "beam_offset_direction": str(args.beam_offset_direction),
                    "eplane_beam": str(eplane),
                    "hplane_beam": str(hplane),
                }
            )
            continue
        solution, stderr, residuals, stats = solve_relative_map(
            design,
            observed,
            weights,
            laplacian,
            smooth_lambda=float(args.smooth_lambda),
            ridge_lambda=float(args.ridge_lambda),
            mean_constraint_weight=float(args.mean_constraint_weight),
        )
        predicted = design @ solution
        measurements["fit_predicted_post_minus_pre_z"] = predicted
        measurements["fit_residual"] = residuals
        measurements["measurement_id"] = measurements["measurement_id"].astype(int) + 1_000_000 * int(band)
        measurement_frames.append(measurements)
        coverage_abs, coverage_count = _coverage(design, weights, float(args.coverage_threshold))
        freq_map = grid.copy()
        freq_map["frequency_band"] = int(band)
        freq_map["frequency_mhz"] = float(freq)
        freq_map["relative_brightness"] = solution
        freq_map["relative_brightness_stderr"] = stderr
        freq_map["coverage_weighted_abs_coeff"] = coverage_abs
        freq_map["coverage_count"] = coverage_count
        freq_map["passes_coverage"] = coverage_count >= int(args.min_coverage_count)
        map_frames.append(freq_map)
        residual_by_target = _residual_by_target(measurements, residuals)
        residual_frames.append(residual_by_target)
        summary_rows.append(
            {
                "frequency_band": int(band),
                "frequency_mhz": float(freq),
                "n_measurements": int(len(measurements)),
                "n_pixels": int(len(grid)),
                "n_pixels_passing_coverage": int(np.count_nonzero(coverage_count >= int(args.min_coverage_count))),
                "matrix_rank": stats["rank"],
                "hessian_condition": stats["condition"],
                "weighted_rmse": stats["weighted_rmse"],
                "median_abs_residual": stats["median_abs_residual"],
                "map_mean_after_constraint": stats["map_mean_after_constraint"],
                "model_mode": str(args.model_mode),
                "beam_model_frequency_mhz": beam_freq,
                "beam_offset_deg": float(args.beam_offset_deg),
                "beam_offset_direction": str(args.beam_offset_direction),
                "eplane_beam": str(eplane),
                "hplane_beam": str(hplane),
                "status": "ok",
            }
        )

    measurements_all = pd.concat(measurement_frames, ignore_index=True) if measurement_frames else pd.DataFrame()
    measurements_all.to_csv(out_dir / "lower_v_measurement_contrasts.csv", index=False)
    if not measurements_all.empty:
        measurements_all[
            [
                "measurement_id",
                "predicted_event_time",
                "frequency_band",
                "frequency_mhz",
                "event_type",
                "source_name",
                "target_pixel_index",
                "observed_post_minus_pre_z",
                "measurement_weight",
            ]
        ].to_csv(out_dir / "measurement_times.csv", index=False)
    else:
        pd.DataFrame().to_csv(out_dir / "measurement_times.csv", index=False)

    map_table = pd.concat(map_frames, ignore_index=True) if map_frames else pd.DataFrame()
    residual_targets = pd.concat(residual_frames, ignore_index=True) if residual_frames else pd.DataFrame()
    if not map_table.empty and not residual_targets.empty:
        map_table = map_table.merge(residual_targets, on=["frequency_band", "pixel_index"], how="left")
    else:
        map_table["target_residual_median"] = np.nan if not map_table.empty else []
        map_table["target_residual_n"] = np.nan if not map_table.empty else []
    map_table.to_csv(out_dir / "lower_v_relative_map_table.csv", index=False)

    design_summary = pd.DataFrame.from_records(summary_rows)
    design_summary.to_csv(out_dir / "occultation_design_matrix_summary.csv", index=False)

    paths = [
        out_dir / "run_config.json",
        out_dir / "galactic_grid_sources.csv",
        out_dir / "measurement_times.csv",
        out_dir / "occultation_design_matrix_summary.csv",
        out_dir / "lower_v_measurement_contrasts.csv",
        out_dir / "lower_v_relative_map_table.csv",
    ]
    if bool(args.write_predicted_events):
        paths.insert(2, out_dir / "galactic_grid_predicted_events.csv")
    if not map_table.empty:
        paths.append(
            _plot_maps(
                map_table,
                "relative_brightness",
                out_dir / "lower_v_relative_maps_by_frequency.png",
                "Lower-V differential occultation relative maps",
                "RdBu_r",
                symmetric=True,
                mask_by_coverage=True,
                min_coverage_count=int(args.min_coverage_count),
            )
        )
        paths.append(
            _plot_maps(
                map_table,
                "coverage_weighted_abs_coeff",
                out_dir / "lower_v_coverage_maps_by_frequency.png",
                "Lower-V differential occultation coverage",
                "viridis",
                symmetric=False,
                mask_by_coverage=False,
                min_coverage_count=int(args.min_coverage_count),
            )
        )
        paths.append(
            _plot_maps(
                map_table,
                "target_residual_median",
                out_dir / "lower_v_residual_maps_by_frequency.png",
                "Lower-V fit residuals binned by crossing cell",
                "RdBu_r",
                symmetric=True,
                mask_by_coverage=False,
                min_coverage_count=int(args.min_coverage_count),
            )
        )
    _write_report(out_dir, config, design_summary, paths)
    print(out_dir / "lower_v_differential_occultation_map_report.md")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(OUT))
    parser.add_argument("--grid-step-deg", type=float, default=15.0)
    parser.add_argument("--frequencies", default="all")
    parser.add_argument("--window-s", type=float, default=900.0)
    parser.add_argument("--inner-s", type=float, default=15.0)
    parser.add_argument("--limb-exclusion-deg", type=float, default=5.0)
    parser.add_argument("--prediction-cadence-seconds", type=float, default=300.0)
    parser.add_argument("--max-gap-seconds", type=float, default=600.0)
    parser.add_argument("--max-measurements", type=int, default=0, help="0 means use all usable measurements per frequency.")
    parser.add_argument("--min-normalization-samples", type=int, default=6)
    parser.add_argument("--min-side-samples", type=int, default=2)
    parser.add_argument("--min-design-norm", type=float, default=1e-6)
    parser.add_argument("--max-weight", type=int, default=20)
    parser.add_argument("--smooth-lambda", type=float, default=1.0)
    parser.add_argument("--ridge-lambda", type=float, default=1e-6)
    parser.add_argument("--mean-constraint-weight", type=float, default=1000.0)
    parser.add_argument("--coverage-threshold", type=float, default=0.01)
    parser.add_argument("--min-coverage-count", type=int, default=8)
    parser.add_argument(
        "--write-predicted-events",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write the large per-grid-cell predicted-event CSV diagnostic.",
    )
    parser.add_argument(
        "--model-mode",
        choices=["beam_weighted", "ring_only"],
        default="beam_weighted",
        help="Use beam-weighted lunar visibility or beam-free lunar visibility only.",
    )
    parser.add_argument("--beam-offset-deg", type=float, default=0.0)
    parser.add_argument(
        "--beam-offset-direction",
        choices=["none", "moon_center", "anti_velocity", "velocity"],
        default="none",
        help="Direction for lower-V beam-axis offset; Moon occultation geometry is not shifted.",
    )
    parser.add_argument("--model-chunk-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
