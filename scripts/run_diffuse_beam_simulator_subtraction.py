#!/usr/bin/env python
"""Subtract a beam-weighted diffuse-sky simulator from occultation windows.

This is a first RAE-2/Ryle-Vonberg simulator for the specific low-frequency
background problem.  It is intentionally limited and explicit:

1. Use the actual cleaned RAE-2 timestamps, spacecraft positions, antennas,
   and frequency channels.
2. Use the digitized Ryle-Vonberg E/H-plane beam cuts already used elsewhere
   in this repository.
3. Use PySM diffuse synchrotron sky maps as the sky brightness model.
4. For each real event, fit observed raw power = intercept + scale * simulated
   diffuse pickup using nearby shifted non-event windows only.
5. Apply that calibrated simulator to the true event window and stack the
   residuals.

The default model is now an approximate azimuth-dependent beam built from the
1D E/H cuts.  The E-plane and H-plane are used as principal planes with a
cos^2/sin^2 interpolation in antenna azimuth.  The antenna-frame azimuth is
defined from the local along-track direction projected perpendicular to the
boresight and then rotated by the supplied yaw angle.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import healpy as hp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rylevonberg.table_io import read_table
from astropy.coordinates import FK4, Galactic, SkyCoord
import astropy.units as u
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.geometry import normalize_vectors  # noqa: E402
from rylevonberg.util import datetime_ns, ensure_dir, robust_sigma, software_versions, write_json  # noqa: E402

from scripts.build_all_frequency_occultation_profile_grids import (  # noqa: E402
    ANT_COLOR,
    ANT_LABEL,
    CLEAN,
    _events_for_source,
    _read,
)


BEAM_DIR = Path(os.environ.get("RAE2_ANTENNA_DIGITIZATION_DIR", "data/antennaDigitization"))
SKY_DIR = Path(os.environ.get("RAE2_SKY_MAP_DIR", "data/pysmMaps"))
MODEL_NSIDE = 8
EXPECTED_SIGN = {"disappearance": -1.0, "reappearance": 1.0}
BEAM_SPECS = [
    (1.31, BEAM_DIR / "eplane_1p31MHz.csv", BEAM_DIR / "hplane_1p31MHz.csv"),
    (3.93, BEAM_DIR / "eplane_3p93MHz.csv", BEAM_DIR / "hplane_3p93MHz.csv"),
    (6.55, BEAM_DIR / "eplane_6p55MHz.csv", BEAM_DIR / "hplane_6p55MHz.csv"),
]


def _nearest_beam(freq_mhz: float) -> tuple[float, Path, Path]:
    return min(BEAM_SPECS, key=lambda spec: abs(spec[0] - float(freq_mhz)))


def _nearest_sky_path(freq_mhz: float) -> Path:
    mhz = int(np.clip(round(float(freq_mhz)), 1, 50))
    return SKY_DIR / f"synch_pysm_s1_{mhz:02d}MHz_IQU.fits"


def _load_sky_i(freq_mhz: float) -> tuple[np.ndarray, Path]:
    path = _nearest_sky_path(freq_mhz)
    with fits.open(path, memmap=True) as hdul:
        arr = np.asarray(hdul[1].data.field(0), dtype=np.float64).reshape(-1)
    arr[~np.isfinite(arr)] = np.nanmedian(arr[np.isfinite(arr)])
    nside_in = hp.npix2nside(len(arr))
    if nside_in != MODEL_NSIDE:
        arr = hp.ud_grade(arr, nside_out=MODEL_NSIDE, order_in="RING", order_out="RING", power=0)
    return arr, path


def _pixel_fk4_vectors(nside: int) -> np.ndarray:
    pix = np.arange(hp.nside2npix(nside))
    theta, phi = hp.pix2ang(nside, pix, nest=False)
    gal = SkyCoord(l=phi * u.rad, b=(0.5 * np.pi - theta) * u.rad, frame=Galactic())
    fk4 = gal.transform_to(FK4(equinox="B1950"))
    ra = fk4.ra.rad
    dec = fk4.dec.rad
    return np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def _load_beam(eplane: Path, hplane: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    e = read_table(eplane)
    h = read_table(hplane)
    angles = e["angle_deg"].to_numpy(dtype=float)
    gain_e = 10.0 ** (e["gain_dB"].to_numpy(dtype=float) / 10.0)
    gain_h = 10.0 ** (h["gain_dB"].to_numpy(dtype=float) / 10.0)
    return angles, gain_e, gain_h


def _interp_cyclic(angles: np.ndarray, values: np.ndarray, angle_deg: np.ndarray) -> np.ndarray:
    a = np.asarray(angle_deg, dtype=float) % 360.0
    x = np.concatenate(([angles[-1] - 360.0], angles, [angles[0] + 360.0]))
    y = np.concatenate(([values[-1]], values, [values[0]]))
    return np.interp(a, x, y)


def _beam_weighted_sky(
    axes: np.ndarray,
    sky_i: np.ndarray,
    pixel_vecs: np.ndarray,
    beam_angles: np.ndarray,
    beam_gain_e: np.ndarray,
    beam_gain_h: np.ndarray,
    e_axes: np.ndarray | None = None,
    h_axes: np.ndarray | None = None,
    beam_mode: str = "axisymmetric_mean",
    chunk_size: int = 128,
) -> np.ndarray:
    axes = np.asarray(axes, dtype=float)
    out = np.full(len(axes), np.nan)
    good = np.isfinite(axes).all(axis=1)
    for start in range(0, len(axes), chunk_size):
        stop = min(start + chunk_size, len(axes))
        idx = np.arange(start, stop)
        valid = good[start:stop]
        if not np.any(valid):
            continue
        block = axes[start:stop][valid]
        dots = np.clip(block @ pixel_vecs.T, -1.0, 1.0)
        sep = np.degrees(np.arccos(dots))
        if beam_mode == "axisymmetric_mean":
            gain = 0.5 * (
                _interp_cyclic(beam_angles, beam_gain_e, sep)
                + _interp_cyclic(beam_angles, beam_gain_h, sep)
            )
        elif beam_mode == "eh_azimuth":
            if e_axes is None or h_axes is None:
                raise ValueError("e_axes and h_axes are required for eh_azimuth beam mode")
            e_block = e_axes[start:stop][valid]
            h_block = h_axes[start:stop][valid]
            # Azimuthal weights relative to the local antenna E/H axes.  This
            # is equivalent to G(theta, phi) = Ge(theta) cos^2(phi) +
            # Gh(theta) sin^2(phi), where phi is measured from the E plane.
            e_proj = pixel_vecs @ e_block.T
            h_proj = pixel_vecs @ h_block.T
            e_weight = e_proj.T**2
            h_weight = h_proj.T**2
            weight_sum = e_weight + h_weight
            e_weight = np.divide(e_weight, weight_sum, out=np.full_like(e_weight, 0.5), where=weight_sum > 0)
            h_weight = 1.0 - e_weight
            gain = (
                _interp_cyclic(beam_angles, beam_gain_e, sep) * e_weight
                + _interp_cyclic(beam_angles, beam_gain_h, sep) * h_weight
            )
        else:
            raise ValueError(f"unsupported beam_mode: {beam_mode}")
        denom = np.nansum(gain, axis=1)
        numer = np.nansum(gain * sky_i.reshape(1, -1), axis=1)
        out[idx[valid]] = np.divide(numer, denom, out=np.full_like(numer, np.nan), where=denom > 0)
    return out


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    norm = np.linalg.norm(arr, axis=1)
    out = np.full_like(arr, np.nan)
    good = np.isfinite(arr).all(axis=1) & (norm > 0)
    out[good] = arr[good] / norm[good, None]
    return out


def _project_reference_axis(z_axis: np.ndarray, reference: np.ndarray) -> np.ndarray:
    ref = np.asarray(reference, dtype=float)
    proj = ref - np.sum(ref * z_axis, axis=1)[:, None] * z_axis
    x_axis = _normalize_rows(proj)
    bad = ~np.isfinite(x_axis).all(axis=1)
    if np.any(bad):
        fallback = np.tile(np.array([0.0, 0.0, 1.0]), (len(z_axis), 1))
        proj = fallback - np.sum(fallback * z_axis, axis=1)[:, None] * z_axis
        x_axis = _normalize_rows(proj)
    bad = ~np.isfinite(x_axis).all(axis=1)
    if np.any(bad):
        fallback = np.tile(np.array([1.0, 0.0, 0.0]), (len(z_axis), 1))
        proj = fallback - np.sum(fallback * z_axis, axis=1)[:, None] * z_axis
        x_axis = _normalize_rows(proj)
    return x_axis


def _antenna_frame_axes(local: pd.DataFrame, antenna: str, yaw_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z_axis = _axis_for_rows(local, antenna)
    velocity = local[["velocity_x", "velocity_y", "velocity_z"]].to_numpy(dtype=float)
    x0 = _project_reference_axis(z_axis, velocity)
    y0 = _normalize_rows(np.cross(z_axis, x0))
    yaw = np.deg2rad(float(yaw_deg))
    e_axis = np.cos(yaw) * x0 + np.sin(yaw) * y0
    h_axis = -np.sin(yaw) * x0 + np.cos(yaw) * y0
    return z_axis, _normalize_rows(e_axis), _normalize_rows(h_axis)


def _make_groups(clean: pd.DataFrame) -> dict[tuple[int, str], tuple[pd.DataFrame, np.ndarray]]:
    groups = {}
    for key, grp in clean.groupby(["frequency_band", "antenna"], sort=True):
        g = grp.sort_values("time").reset_index(drop=True)
        pos = g[["position_x", "position_y", "position_z"]].to_numpy(dtype=float)
        if len(g) >= 2:
            vel = np.gradient(pos, axis=0)
        else:
            vel = np.full_like(pos, np.nan)
        g["velocity_x"] = vel[:, 0]
        g["velocity_y"] = vel[:, 1]
        g["velocity_z"] = vel[:, 2]
        groups[(int(key[0]), str(key[1]))] = (g, datetime_ns(g["time"]))
    return groups


def _event_window(
    group: pd.DataFrame,
    group_ns: np.ndarray,
    event_time: pd.Timestamp,
    window_s: float,
) -> pd.DataFrame:
    event_ns = pd.Timestamp(event_time).value
    half_ns = int(float(window_s) * 1e9)
    lo = int(np.searchsorted(group_ns, event_ns - half_ns, side="left"))
    hi = int(np.searchsorted(group_ns, event_ns + half_ns, side="right"))
    if hi <= lo:
        return pd.DataFrame()
    local = group.iloc[lo:hi].copy()
    local["group_index"] = np.arange(lo, hi)
    local["t_rel_sec"] = (datetime_ns(local["time"]) - event_ns).astype(float) / 1e9
    local = local[np.abs(local["t_rel_sec"]) <= float(window_s)].copy()
    valid = np.isfinite(pd.to_numeric(local["power"], errors="coerce").to_numpy(dtype=float))
    if "is_valid" in local.columns:
        valid &= local["is_valid"].to_numpy(dtype=bool)
    return local.loc[valid].copy()


def _axis_for_rows(local: pd.DataFrame, antenna: str) -> np.ndarray:
    pos = local[["position_x", "position_y", "position_z"]].to_numpy(dtype=float)
    lower = normalize_vectors(-pos)
    if antenna == "rv2_coarse":
        return lower
    if antenna == "rv1_coarse":
        return -lower
    raise ValueError(f"unsupported antenna: {antenna}")


def _model_for_local(
    local: pd.DataFrame,
    antenna: str,
    sky_i: np.ndarray,
    pixel_vecs: np.ndarray,
    beam_angles: np.ndarray,
    beam_gain_e: np.ndarray,
    beam_gain_h: np.ndarray,
    model_cache: dict[int, float],
    beam_mode: str,
    yaw_deg: float,
) -> np.ndarray:
    idx = local["group_index"].to_numpy(dtype=int)
    out = np.full(len(idx), np.nan)
    missing_mask = np.array([int(i) not in model_cache for i in idx], dtype=bool)
    if missing_mask.any():
        missing = local.iloc[np.where(missing_mask)[0]]
        axes, e_axes, h_axes = _antenna_frame_axes(missing, antenna, yaw_deg)
        vals = _beam_weighted_sky(
            axes,
            sky_i,
            pixel_vecs,
            beam_angles,
            beam_gain_e,
            beam_gain_h,
            e_axes=e_axes,
            h_axes=h_axes,
            beam_mode=beam_mode,
        )
        for row_idx, value in zip(idx[missing_mask], vals):
            model_cache[int(row_idx)] = float(value)
    for j, row_idx in enumerate(idx):
        out[j] = model_cache.get(int(row_idx), np.nan)
    return out


def _robust_linear_fit(x: np.ndarray, y: np.ndarray, max_iter: int = 4) -> tuple[float, float, float, int]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(good) < 8 or np.nanstd(x[good]) <= 0:
        return np.nan, np.nan, np.nan, 0
    keep = good.copy()
    coef = np.array([np.nan, np.nan])
    for _ in range(max_iter):
        mat = np.column_stack([np.ones(np.count_nonzero(keep)), x[keep]])
        coef, *_ = np.linalg.lstsq(mat, y[keep], rcond=None)
        resid = y - (coef[0] + coef[1] * x)
        sigma = robust_sigma(resid[keep] - np.nanmedian(resid[keep]))
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = float(np.nanstd(resid[keep]))
        if not np.isfinite(sigma) or sigma <= 0:
            break
        next_keep = good & (np.abs(resid - np.nanmedian(resid[keep])) <= 6.0 * sigma)
        if np.array_equal(next_keep, keep):
            break
        keep = next_keep
    resid = y - (coef[0] + coef[1] * x)
    resid_keep = resid[keep]
    sigma = robust_sigma(resid_keep - np.nanmedian(resid_keep))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.nanstd(resid_keep))
    return float(coef[0]), float(coef[1]), float(sigma), int(np.count_nonzero(keep))


def _build_exclusion_index(sources: list[str], antennas: list[str]) -> dict[tuple[int, str], np.ndarray]:
    frames = []
    for source in sources:
        ev = _events_for_source(source)
        if ev.empty:
            continue
        frames.append(ev[ev["antenna"].isin(antennas)].copy())
    if not frames:
        return {}
    events = pd.concat(frames, ignore_index=True)
    out: dict[tuple[int, str], np.ndarray] = {}
    for (band, antenna), grp in events.groupby(["frequency_band", "antenna"], sort=True):
        times = pd.to_datetime(grp["predicted_event_time"], errors="coerce").dropna()
        if not times.empty:
            out[(int(band), str(antenna))] = np.sort(datetime_ns(times))
    return out


def _shift_overlaps_real_event(
    band: int,
    antenna: str,
    center: pd.Timestamp,
    exclusion_index: dict[tuple[int, str], np.ndarray],
    radius_s: float,
) -> bool:
    arr = exclusion_index.get((int(band), str(antenna)))
    if arr is None or arr.size == 0:
        return False
    center_ns = pd.Timestamp(center).value
    radius_ns = int(float(radius_s) * 1e9)
    pos = int(np.searchsorted(arr, center_ns))
    if pos < arr.size and abs(int(arr[pos]) - center_ns) <= radius_ns:
        return True
    if pos > 0 and abs(int(arr[pos - 1]) - center_ns) <= radius_ns:
        return True
    return False


def _bin_rows(
    event: pd.Series,
    t: np.ndarray,
    values: np.ndarray,
    bins: np.ndarray,
    value_col: str,
    extra: dict[str, float | int | str],
) -> list[dict[str, float | int | str]]:
    rows = []
    idx = np.digitize(t, bins) - 1
    for bin_idx in sorted(set(idx)):
        if bin_idx < 0 or bin_idx >= len(bins) - 1:
            continue
        mask = idx == bin_idx
        vals = values[mask]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        row = {
            "source_name": str(event["source_name"]).lower(),
            "event_id": int(event["event_id"]),
            "event_type": str(event["event_type"]),
            "frequency_band": int(event["frequency_band"]),
            "frequency_mhz": float(event["frequency_mhz"]),
            "antenna": str(event["antenna"]),
            "t_bin_sec": float(0.5 * (bins[bin_idx] + bins[bin_idx + 1])),
            value_col: float(np.nanmedian(vals)),
            "n_samples": int(vals.size),
        }
        row.update(extra)
        rows.append(row)
    return rows


def run_simulator_subtraction(
    clean: pd.DataFrame,
    events: pd.DataFrame,
    source: str,
    antennas: list[str],
    window_s: float,
    bin_s: float,
    control_shifts_s: list[float],
    min_control_samples: int,
    scale_mode: str,
    beam_mode: str,
    yaw_deg: float,
    fixed_scales: dict[tuple[int, str], float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    groups = _make_groups(clean)
    bins = np.arange(-float(window_s), float(window_s) + float(bin_s), float(bin_s))
    pixel_vecs = _pixel_fk4_vectors(MODEL_NSIDE)
    exclusion_index = _build_exclusion_index(["earth", "sun"], antennas)
    point_rows = []
    fit_rows = []
    input_rows = []
    work = events[events["source_name"].astype(str).str.lower().eq(source.lower()) & events["antenna"].isin(antennas)].copy()

    for (band, freq, antenna), ev_group in work.groupby(["frequency_band", "frequency_mhz", "antenna"], sort=True):
        payload = groups.get((int(band), str(antenna)))
        if payload is None:
            continue
        group, group_ns = payload
        sky_i, sky_path = _load_sky_i(float(freq))
        beam_freq, eplane, hplane = _nearest_beam(float(freq))
        beam_angles, beam_gain_e, beam_gain_h = _load_beam(eplane, hplane)
        model_cache: dict[int, float] = {}
        input_rows.append(
            {
                "frequency_band": int(band),
                "frequency_mhz": float(freq),
                "antenna": str(antenna),
                "sky_map": str(sky_path),
                "beam_model_frequency_mhz": float(beam_freq),
                "eplane_beam": str(eplane),
                "hplane_beam": str(hplane),
                "model_nside": MODEL_NSIDE,
                "beam_mode": beam_mode,
                "yaw_deg": float(yaw_deg),
            }
        )
        for _, ev in ev_group.iterrows():
            event_time = pd.Timestamp(ev["predicted_event_time"])
            control_power = []
            control_model = []
            n_control_windows = 0
            for shift_s in control_shifts_s:
                center = event_time + pd.to_timedelta(float(shift_s), unit="s")
                if _shift_overlaps_real_event(int(band), str(antenna), center, exclusion_index, window_s):
                    continue
                local_control = _event_window(group, group_ns, center, window_s)
                if local_control.empty:
                    continue
                model_control = _model_for_local(
                    local_control,
                    str(antenna),
                    sky_i,
                    pixel_vecs,
                    beam_angles,
                    beam_gain_e,
                    beam_gain_h,
                    model_cache,
                    beam_mode,
                    yaw_deg,
                )
                power_control = pd.to_numeric(local_control["power"], errors="coerce").to_numpy(dtype=float)
                good = np.isfinite(model_control) & np.isfinite(power_control)
                if np.count_nonzero(good) < 4:
                    continue
                control_model.append(model_control[good])
                control_power.append(power_control[good])
                n_control_windows += 1
            if not control_power:
                continue
            x_control = np.concatenate(control_model)
            y_control = np.concatenate(control_power)
            if len(y_control) < min_control_samples:
                continue
            fitted_intercept, fitted_scale, fitted_sigma, fitted_n = _robust_linear_fit(x_control, y_control)
            if scale_mode == "per_event":
                intercept, scale, sigma, n_fit = fitted_intercept, fitted_scale, fitted_sigma, fitted_n
            elif scale_mode == "channel_positive_median":
                fixed_scale = (fixed_scales or {}).get((int(band), str(antenna)), np.nan)
                if not np.isfinite(fixed_scale):
                    fixed_scale = 0.0
                control_resid_for_intercept = y_control - fixed_scale * x_control
                intercept = float(np.nanmedian(control_resid_for_intercept))
                scale = float(fixed_scale)
                control_resid_for_sigma = y_control - (intercept + scale * x_control)
                sigma = robust_sigma(control_resid_for_sigma - np.nanmedian(control_resid_for_sigma))
                if not np.isfinite(sigma) or sigma <= 0:
                    sigma = float(np.nanstd(control_resid_for_sigma))
                n_fit = int(np.count_nonzero(np.isfinite(control_resid_for_sigma)))
            else:
                raise ValueError(f"unsupported scale_mode: {scale_mode}")
            if not np.isfinite(intercept) or not np.isfinite(scale) or not np.isfinite(sigma) or sigma <= 0:
                continue
            control_resid = y_control - (intercept + scale * x_control)
            resid_center = float(np.nanmedian(control_resid[np.isfinite(control_resid)]))

            local_true = _event_window(group, group_ns, event_time, window_s)
            if local_true.empty:
                continue
            model_true = _model_for_local(
                local_true,
                str(antenna),
                sky_i,
                pixel_vecs,
                beam_angles,
                beam_gain_e,
                beam_gain_h,
                model_cache,
                beam_mode,
                yaw_deg,
            )
            power_true = pd.to_numeric(local_true["power"], errors="coerce").to_numpy(dtype=float)
            t_true = local_true["t_rel_sec"].to_numpy(dtype=float)
            sim_true = intercept + scale * model_true
            residual_z = (power_true - sim_true - resid_center) / sigma
            raw_control_z = (power_true - np.nanmedian(y_control)) / sigma
            good_true = np.isfinite(t_true) & np.isfinite(residual_z) & np.isfinite(raw_control_z)
            if np.count_nonzero(good_true) < 8:
                continue
            point_rows.extend(
                _bin_rows(
                    ev,
                    t_true[good_true],
                    residual_z[good_true],
                    bins,
                    "simulator_subtracted_z_power",
                    {
                        "simulator_intercept": intercept,
                        "simulator_scale": scale,
                        "simulator_fitted_intercept": fitted_intercept,
                        "simulator_fitted_scale": fitted_scale,
                        "simulator_scale_mode": scale_mode,
                        "control_resid_sigma": sigma,
                        "n_control_samples": int(len(y_control)),
                        "n_control_windows": int(n_control_windows),
                        "n_fit_samples": int(n_fit),
                        "beam_model_frequency_mhz": float(beam_freq),
                        "beam_mode": beam_mode,
                        "yaw_deg": float(yaw_deg),
                    },
                )
            )
            point_rows.extend(
                _bin_rows(
                    ev,
                    t_true[good_true],
                    raw_control_z[good_true],
                    bins,
                    "control_normalized_raw_z_power",
                    {
                        "simulator_intercept": intercept,
                        "simulator_scale": scale,
                        "simulator_fitted_intercept": fitted_intercept,
                        "simulator_fitted_scale": fitted_scale,
                        "simulator_scale_mode": scale_mode,
                        "control_resid_sigma": sigma,
                        "n_control_samples": int(len(y_control)),
                        "n_control_windows": int(n_control_windows),
                        "n_fit_samples": int(n_fit),
                        "beam_model_frequency_mhz": float(beam_freq),
                        "beam_mode": beam_mode,
                        "yaw_deg": float(yaw_deg),
                    },
                )
            )
            fit_rows.append(
                {
                    "source_name": str(ev["source_name"]).lower(),
                    "event_id": int(ev["event_id"]),
                    "event_type": str(ev["event_type"]),
                    "frequency_band": int(band),
                    "frequency_mhz": float(freq),
                    "antenna": str(antenna),
                    "predicted_event_time": event_time,
                    "simulator_intercept": intercept,
                    "simulator_scale": scale,
                    "simulator_fitted_intercept": fitted_intercept,
                    "simulator_fitted_scale": fitted_scale,
                    "simulator_fitted_sigma": fitted_sigma,
                    "simulator_fitted_n": fitted_n,
                    "simulator_scale_mode": scale_mode,
                    "control_resid_sigma": sigma,
                    "n_control_samples": int(len(y_control)),
                    "n_control_windows": int(n_control_windows),
                    "n_fit_samples": int(n_fit),
                    "beam_model_frequency_mhz": float(beam_freq),
                    "beam_mode": beam_mode,
                    "yaw_deg": float(yaw_deg),
                }
            )

    points = pd.DataFrame(point_rows)
    if not points.empty:
        id_cols = [
            "source_name",
            "event_id",
            "event_type",
            "frequency_band",
            "frequency_mhz",
            "antenna",
            "t_bin_sec",
        ]
        value_cols = ["simulator_subtracted_z_power", "control_normalized_raw_z_power"]
        merged = None
        for value_col in value_cols:
            sub = points.dropna(subset=[value_col]).copy()
            keep = id_cols + [value_col] + [
                "simulator_intercept",
                "simulator_scale",
                "control_resid_sigma",
                "n_control_samples",
                "n_control_windows",
                "n_fit_samples",
                "beam_model_frequency_mhz",
                "beam_mode",
                "yaw_deg",
            ]
            sub = sub[keep]
            if merged is None:
                merged = sub
            else:
                merged = merged.merge(sub[id_cols + [value_col]], on=id_cols, how="outer")
        points = merged if merged is not None else pd.DataFrame()
    return points, pd.DataFrame(fit_rows), pd.DataFrame(input_rows)


def build_positive_channel_scales(
    fit_table: pd.DataFrame,
    min_positive_fraction: float,
    max_abs_scale_quantile: float = 0.95,
) -> tuple[dict[tuple[int, str], float], pd.DataFrame]:
    rows = []
    scales: dict[tuple[int, str], float] = {}
    for (band, antenna), grp in fit_table.groupby(["frequency_band", "antenna"], sort=True):
        vals = pd.to_numeric(grp["simulator_scale"], errors="coerce")
        vals = vals[np.isfinite(vals)]
        if vals.empty:
            scale = np.nan
            frac_pos = np.nan
            n_used = 0
        else:
            abs_limit = float(vals.abs().quantile(float(max_abs_scale_quantile)))
            trimmed = vals[vals.abs().le(abs_limit)]
            pos = trimmed[trimmed.gt(0)]
            frac_pos = float((trimmed > 0).mean()) if not trimmed.empty else np.nan
            if len(pos) >= 5 and np.isfinite(frac_pos) and frac_pos >= float(min_positive_fraction):
                scale = float(pos.median())
                n_used = int(len(pos))
            else:
                scale = 0.0
                n_used = int(len(pos))
        scales[(int(band), str(antenna))] = scale
        freq = float(pd.to_numeric(grp["frequency_mhz"], errors="coerce").dropna().iloc[0])
        rows.append(
            {
                "frequency_band": int(band),
                "frequency_mhz": freq,
                "antenna": str(antenna),
                "channel_positive_median_scale": scale,
                "positive_fraction_after_trim": frac_pos,
                "n_positive_scales_used": n_used,
                "scale_selection_rule": (
                    "positive_median" if np.isfinite(scale) and scale > 0 else "zero_unreliable_or_inconsistent"
                ),
            }
        )
    return scales, pd.DataFrame(rows)


def _robust_sem(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size <= 1:
        return np.nan
    scale = robust_sigma(vals - np.nanmedian(vals))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.nanstd(vals, ddof=1))
    return float(scale / np.sqrt(vals.size)) if np.isfinite(scale) and scale > 0 else np.nan


def summarize(points: pd.DataFrame, value_col: str, method: str) -> pd.DataFrame:
    rows = []
    keys = ["source_name", "event_type", "frequency_band", "frequency_mhz", "antenna", "t_bin_sec"]
    for vals_key, grp in points.groupby(keys, sort=True, dropna=False):
        raw = pd.to_numeric(grp[value_col], errors="coerce")
        good = raw.notna() & np.isfinite(raw)
        vals = raw.loc[good]
        if vals.empty:
            continue
        rows.append(
            {
                **dict(zip(keys, vals_key)),
                "method": method,
                "median_z_power": float(vals.median()),
                "median_z_power_err": _robust_sem(vals),
                "n_events": int(grp.loc[good, "event_id"].nunique()),
                "n_points": int(len(vals)),
                "median_n_control_windows": float(np.nanmedian(grp.loc[good, "n_control_windows"]))
                if "n_control_windows" in grp
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def prepost_contrast(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pre = (-180.0, -60.0)
    post = (60.0, 180.0)
    keys = ["method", "source_name", "frequency_band", "frequency_mhz", "antenna", "event_type"]
    for vals_key, grp in summary.groupby(keys, sort=True, dropna=False):
        before = grp[(grp["t_bin_sec"] >= pre[0]) & (grp["t_bin_sec"] <= pre[1])]["median_z_power"]
        after = grp[(grp["t_bin_sec"] >= post[0]) & (grp["t_bin_sec"] <= post[1])]["median_z_power"]
        if before.empty or after.empty:
            continue
        event_type = str(vals_key[-1])
        delta = float(np.nanmedian(after) - np.nanmedian(before))
        rows.append(
            {
                **dict(zip(keys, vals_key)),
                "n_events": int(np.nanmedian(grp["n_events"])),
                "post_minus_pre": delta,
                "source_like_contrast": float(EXPECTED_SIGN[event_type] * delta),
            }
        )
    return pd.DataFrame(rows)


def _plot_grid(summary: pd.DataFrame, source: str, method: str, out_dir: Path, window_s: float) -> Path:
    sub_summary = summary[summary["method"].eq(method)].copy()
    freqs = sorted(sub_summary["frequency_mhz"].dropna().unique())
    fig, axes = plt.subplots(len(freqs), 2, figsize=(12, max(10, 1.35 * len(freqs))), sharex=True)
    if len(freqs) == 1:
        axes = np.asarray([axes])
    for i, freq in enumerate(freqs):
        for j, event_type in enumerate(["disappearance", "reappearance"]):
            ax = axes[i, j]
            sub = sub_summary[np.isclose(sub_summary["frequency_mhz"], freq) & sub_summary["event_type"].eq(event_type)]
            for antenna, grp in sub.groupby("antenna", sort=True):
                grp = grp.sort_values("t_bin_sec")
                ax.errorbar(
                    grp["t_bin_sec"],
                    grp["median_z_power"],
                    yerr=grp["median_z_power_err"],
                    marker="o",
                    markersize=2.4,
                    linewidth=1.15,
                    elinewidth=0.55,
                    capsize=1.0,
                    alpha=0.9,
                    color=ANT_COLOR.get(str(antenna)),
                    ecolor=ANT_COLOR.get(str(antenna)),
                    label=ANT_LABEL.get(str(antenna), str(antenna)),
                )
            ax.axvline(0, color="black", linestyle="--", linewidth=0.75)
            ax.axhline(0, color="0.65", linewidth=0.7)
            ax.set_title(f"{freq:.2f} MHz {event_type}", fontsize=9)
            if j == 0:
                ax.set_ylabel(method.replace("_", " "))
            if i == len(freqs) - 1:
                ax.set_xlabel("seconds from predicted event")
            if i == 0 and j == 1:
                ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"{source.title()}: {method.replace('_', ' ')}, +/-{window_s:.0f} s", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    path = out_dir / f"{source}_{method}_all_frequency_profile_grid_{int(window_s)}s.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_overlay(summary: pd.DataFrame, source: str, antenna: str, out_dir: Path, window_s: float) -> Path:
    raw = summary[summary["method"].eq("control_normalized_raw") & summary["antenna"].eq(antenna)].copy()
    corr = summary[summary["method"].eq("simulator_subtracted") & summary["antenna"].eq(antenna)].copy()
    freqs = sorted(set(raw["frequency_mhz"].dropna()).union(corr["frequency_mhz"].dropna()))
    fig, axes = plt.subplots(len(freqs), 2, figsize=(12, max(10, 1.35 * len(freqs))), sharex=True)
    if len(freqs) == 1:
        axes = np.asarray([axes])
    for i, freq in enumerate(freqs):
        for j, event_type in enumerate(["disappearance", "reappearance"]):
            ax = axes[i, j]
            for data, label, color, ls in [
                (raw, "raw normalized by control windows", "0.35", "--"),
                (corr, "simulator subtracted", "#d95f02" if antenna == "rv2_coarse" else "#4c78a8", "-"),
            ]:
                sub = data[np.isclose(data["frequency_mhz"], freq) & data["event_type"].eq(event_type)].sort_values(
                    "t_bin_sec"
                )
                if sub.empty:
                    continue
                ax.errorbar(
                    sub["t_bin_sec"],
                    sub["median_z_power"],
                    yerr=sub["median_z_power_err"],
                    marker="o",
                    markersize=2.3,
                    linewidth=1.1,
                    elinewidth=0.55,
                    capsize=1.0,
                    alpha=0.9,
                    color=color,
                    ecolor=color,
                    linestyle=ls,
                    label=label if i == 0 and j == 1 else None,
                )
            ax.axvline(0, color="black", linestyle="--", linewidth=0.75)
            ax.axhline(0, color="0.65", linewidth=0.7)
            ax.set_title(f"{freq:.2f} MHz {event_type}", fontsize=9)
            if j == 0:
                ax.set_ylabel("control-normalized z")
            if i == len(freqs) - 1:
                ax.set_xlabel("seconds from predicted event")
            if i == 0 and j == 1:
                ax.legend(frameon=False, fontsize=8)
    fig.suptitle(
        f"{source.title()} {ANT_LABEL.get(antenna, antenna)}: raw vs diffuse-simulator subtraction, +/-{window_s:.0f} s",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    path = out_dir / f"{source}_{antenna}_raw_vs_simulator_subtracted_grid_{int(window_s)}s.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _comparison_table(contrasts: pd.DataFrame) -> pd.DataFrame:
    raw = contrasts[contrasts["method"].eq("control_normalized_raw")].rename(
        columns={"post_minus_pre": "post_minus_pre_raw", "source_like_contrast": "source_like_contrast_raw"}
    )
    corr = contrasts[contrasts["method"].eq("simulator_subtracted")].rename(
        columns={
            "post_minus_pre": "post_minus_pre_simulator_subtracted",
            "source_like_contrast": "source_like_contrast_simulator_subtracted",
        }
    )
    keys = ["source_name", "frequency_band", "frequency_mhz", "antenna", "event_type"]
    out = raw[keys + ["n_events", "post_minus_pre_raw", "source_like_contrast_raw"]].merge(
        corr[
            keys
            + [
                "post_minus_pre_simulator_subtracted",
                "source_like_contrast_simulator_subtracted",
            ]
        ],
        on=keys,
        how="outer",
    )
    out["source_like_contrast_change"] = (
        out["source_like_contrast_simulator_subtracted"] - out["source_like_contrast_raw"]
    )
    return out.sort_values(keys).reset_index(drop=True)


def write_report(
    out_dir: Path,
    source: str,
    inputs: pd.DataFrame,
    fits: pd.DataFrame,
    comparison: pd.DataFrame,
    paths: list[Path],
    config: dict,
) -> None:
    low = comparison[
        comparison["frequency_mhz"].isin([0.45, 0.70, 0.90, 1.31, 2.20])
        & comparison["antenna"].eq("rv2_coarse")
    ].copy()
    fit_summary = (
        fits.groupby(["frequency_mhz", "antenna", "event_type"])
        .agg(
            n_fit_events=("event_id", "nunique"),
            median_control_windows=("n_control_windows", "median"),
            median_control_samples=("n_control_samples", "median"),
            median_fit_samples=("n_fit_samples", "median"),
            median_control_resid_sigma=("control_resid_sigma", "median"),
        )
        .reset_index()
        if not fits.empty
        else pd.DataFrame()
    )
    lines = [
        "# RAE-2 Diffuse Beam Simulator Subtraction",
        "",
        "## What Was Built",
        "",
        "This run simulates the diffuse Galactic/synchrotron pickup seen by the Ryle-Vonberg antenna using:",
        "",
        "- actual RAE-2 timestamps and spacecraft positions;",
        "- the existing digitized Ryle-Vonberg E/H-plane beam patterns;",
        "- PySM synchrotron sky maps;",
        "- lower/upper V boresight convention from the pipeline.",
        "",
        "For each real event, the simulator scale and offset are fit on shifted non-event windows only.",
        "The calibrated simulator is then subtracted from the true event window before stacking.",
        (
            "Beam-frame convention: in `eh_azimuth` mode, the E-plane reference is the local along-track direction "
            "projected perpendicular to the antenna boresight and rotated by `yaw_deg`; H is the perpendicular in-boresight-plane axis."
        )
        if config.get("beam_mode") == "eh_azimuth"
        else "Beam-frame convention: `axisymmetric_mean` mode ignores antenna azimuth and uses the mean E/H radial power beam.",
        "",
        "## Configuration",
        "",
        pd.Series(config).to_string(),
        "",
        "## Beam / Sky Inputs",
        "",
        inputs.to_string(index=False) if not inputs.empty else "No model inputs recorded.",
        "",
        "## Low-Frequency Lower-V Comparison",
        "",
        low[
            [
                "frequency_mhz",
                "event_type",
                "n_events",
                "source_like_contrast_raw",
                "source_like_contrast_simulator_subtracted",
                "source_like_contrast_change",
            ]
        ].to_string(index=False)
        if not low.empty
        else "No low-frequency lower-V rows.",
        "",
        "## Fit Counts",
        "",
        fit_summary.to_string(index=False) if not fit_summary.empty else "No fit rows.",
        "",
        "## Interpretation",
        "",
        "A successful subtraction would preserve known positive-control morphology while moving the problematic low-frequency",
        "Earth profiles toward the 0.45 MHz pattern. If the low-frequency disappearance rows remain anti-template, then",
        "the available axisymmetric 1D-beam diffuse simulator is not sufficient to remove the contaminating term.",
        "",
        (
            "The main approximation is still the beam. In `eh_azimuth` mode the model has azimuthal dependence, but it is "
            "only a cos^2/sin^2 interpolation between two 1D principal-plane cuts. It is not a measured 2D beam and it "
            "does not include spin-phase-dependent asymmetries, polarization response, lunar thermal/blocked-sky terms, "
            "or receiver calibration changes."
        )
        if config.get("beam_mode") == "eh_azimuth"
        else (
            "The main approximation is the beam: we only have 1D E/H cuts, not a full spin/azimuth-dependent 2D beam. "
            "That means this simulator can test plausibility and remove simple beam-weighted diffuse trends, but it is "
            "not a final calibrated RAE-2 response model."
        ),
        "",
        "## Plots",
        "",
    ]
    lines.extend(f"- `{path}`" for path in paths)
    (out_dir / "diffuse_beam_simulator_subtraction_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="earth")
    parser.add_argument("--antennas", default="rv2_coarse")
    parser.add_argument("--out-dir", default=str(ROOT / "outputs/diffuse_beam_simulator_subtraction_earth_20min_v1"))
    parser.add_argument("--window-s", type=float, default=1200.0)
    parser.add_argument("--bin-s", type=float, default=60.0)
    parser.add_argument("--control-shifts-s", default="-3600,-2400,-1800,1800,2400,3600")
    parser.add_argument("--min-control-samples", type=int, default=24)
    parser.add_argument("--scale-mode", choices=["per_event", "channel_positive_median"], default="per_event")
    parser.add_argument(
        "--beam-mode",
        choices=["axisymmetric_mean", "eh_azimuth"],
        default="eh_azimuth",
        help="axisymmetric_mean reproduces the previous E/H averaged radial beam; eh_azimuth uses E/H principal planes.",
    )
    parser.add_argument("--yaw-deg", type=float, default=13.0)
    parser.add_argument(
        "--fixed-scale-table",
        default="",
        help="Fit table from a per-event run, used for channel_positive_median mode.",
    )
    parser.add_argument("--min-positive-scale-fraction", type=float, default=0.6)
    args = parser.parse_args()

    source = str(args.source).strip().lower()
    antennas = [x.strip() for x in str(args.antennas).split(",") if x.strip()]
    shifts = [float(x.strip()) for x in str(args.control_shifts_s).split(",") if x.strip()]
    out_dir = ensure_dir(args.out_dir)
    config = {
        "source": source,
        "antennas": antennas,
        "window_s": float(args.window_s),
        "bin_s": float(args.bin_s),
        "control_shifts_s": shifts,
        "min_control_samples": int(args.min_control_samples),
        "scale_mode": args.scale_mode,
        "beam_mode": args.beam_mode,
        "yaw_deg": float(args.yaw_deg),
        "fixed_scale_table": args.fixed_scale_table,
        "min_positive_scale_fraction": float(args.min_positive_scale_fraction),
        "beam_model": (
            "axisymmetric mean of digitized E/H plane cuts"
            if args.beam_mode == "axisymmetric_mean"
            else "azimuth-dependent cos2/sin2 interpolation of digitized E/H plane cuts"
        ),
        "model_nside": MODEL_NSIDE,
    }
    write_json(out_dir / "run_config.json", {**config, "software_versions": software_versions()})

    clean_cols = [
        "time",
        "frequency_band",
        "frequency_mhz",
        "antenna",
        "position_x",
        "position_y",
        "position_z",
        "power",
        "is_valid",
    ]
    clean = _read(CLEAN, usecols=clean_cols, parse_dates=["time"])
    clean = clean[clean["antenna"].isin(antennas)].copy()
    events = _events_for_source(source)
    events = events[events["source_name"].astype(str).str.lower().eq(source) & events["antenna"].isin(antennas)].copy()
    if events.empty:
        raise SystemExit(f"No events found for source={source}, antennas={antennas}")

    fixed_scales = None
    scale_summary = pd.DataFrame()
    if args.scale_mode == "channel_positive_median":
        if not args.fixed_scale_table:
            raise SystemExit("--fixed-scale-table is required for channel_positive_median mode")
        prior_fits = _read(Path(args.fixed_scale_table))
        fixed_scales, scale_summary = build_positive_channel_scales(
            prior_fits,
            min_positive_fraction=args.min_positive_scale_fraction,
        )
        scale_summary.to_csv(out_dir / f"{source}_simulator_channel_positive_scale_table.csv", index=False)

    points, fits, inputs = run_simulator_subtraction(
        clean=clean,
        events=events,
        source=source,
        antennas=antennas,
        window_s=args.window_s,
        bin_s=args.bin_s,
        control_shifts_s=shifts,
        min_control_samples=args.min_control_samples,
        scale_mode=args.scale_mode,
        beam_mode=args.beam_mode,
        yaw_deg=args.yaw_deg,
        fixed_scales=fixed_scales,
    )
    points.to_csv(out_dir / f"{source}_simulator_subtracted_points.csv", index=False)
    fits.to_csv(out_dir / f"{source}_simulator_fit_table.csv", index=False)
    inputs.to_csv(out_dir / f"{source}_simulator_input_table.csv", index=False)

    summaries = []
    if not points.empty:
        summaries.append(summarize(points, "control_normalized_raw_z_power", "control_normalized_raw"))
        summaries.append(summarize(points, "simulator_subtracted_z_power", "simulator_subtracted"))
    summary = pd.concat(summaries, ignore_index=True, sort=False) if summaries else pd.DataFrame()
    contrasts = prepost_contrast(summary) if not summary.empty else pd.DataFrame()
    comparison = _comparison_table(contrasts) if not contrasts.empty else pd.DataFrame()
    summary.to_csv(out_dir / f"{source}_simulator_subtracted_summary.csv", index=False)
    contrasts.to_csv(out_dir / f"{source}_simulator_subtracted_contrasts.csv", index=False)
    comparison.to_csv(out_dir / f"{source}_simulator_subtraction_comparison.csv", index=False)

    paths = []
    if not summary.empty:
        paths.append(_plot_grid(summary, source, "simulator_subtracted", out_dir, args.window_s))
        for antenna in antennas:
            paths.append(_plot_overlay(summary, source, antenna, out_dir, args.window_s))
    write_report(out_dir, source, inputs, fits, comparison, paths, config)
    print(out_dir / "diffuse_beam_simulator_subtraction_report.md")


if __name__ == "__main__":
    main()
