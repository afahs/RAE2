#!/usr/bin/env python
"""Search lower-V Earth event windows for lunar-reflection interference.

This diagnostic keeps the data at sample level.  The existing all-frequency
profile grids are useful for broad occultation morphology, but their 60 s bins
can smooth over a sea-cliff-like direct/reflected interference term.  Here each
sample is centered on a predicted Earth occultation event, assigned a
lunar-surface specular-reflection phase, and then stacked in event-relative
time and in reflection phase.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.table_io import read_table, write_table  # noqa: E402
from rylevonberg.constants import (  # noqa: E402
    EARTH_MOON_DISTANCE_KM,
    EARTH_UNIT_COLUMNS,
    FREQUENCY_MAP_MHZ,
    MOON_RADIUS_KM,
    SPACECRAFT_COLUMNS,
)
from rylevonberg.detection import baseline_matrix, event_template  # noqa: E402
from rylevonberg.geometry import (  # noqa: E402
    earth_direction_from_spacecraft,
    moon_limb_angle_deg,
    normalize_vectors,
)
from rylevonberg.util import datetime_ns, ensure_dir, robust_sigma, software_versions, write_json  # noqa: E402


C_KM_S = 299_792.458
RV_EFFECTIVE_BANDWIDTH_HZ = 40_000.0
RV_COARSE_TIME_CONSTANT_S = 0.1
CLEAN = ROOT / "outputs/control_survey_earth_sun_postnov1974_v2/01_ingest/cleaned_timeseries.csv"
EVENTS = ROOT / "outputs/planetary_confirmation_survey_science_baseline_v2/events/all_planet_predicted_events.csv"
OUT = ROOT / "outputs/lower_v_lunar_reflection_interference_v1"
ANT_LABEL = {"rv1_coarse": "upper V", "rv2_coarse": "lower V"}
ANT_COLOR = {"rv1_coarse": "#4c78a8", "rv2_coarse": "#d95f02"}
EMPTY_SAMPLE_COLUMNS = [
    "time",
    "frequency_band",
    "antenna",
    "power",
    "is_valid",
    "t_rel_sec",
    "z_power",
    "source_name",
    "event_id",
    "event_type",
    "predicted_event_time",
    "centered_event_time",
    "time_shift_s",
    "frequency_mhz",
    "phase_rad",
    "phase_cycles",
    "phase_wrapped_rad",
    "delta_l_km",
    "reflection_valid",
    "direct_limb_angle_deg",
    "direct_visible",
    "oriented_t_sec",
    "event_visible_side",
    "geometry_model",
    "fringe_period_s",
    "z_resid",
]


def parse_csv_list(text: str, cast=str) -> list:
    return [cast(x.strip()) for x in str(text).split(",") if x.strip()]


def wavelength_km(frequency_mhz: np.ndarray | float) -> np.ndarray:
    freq = np.asarray(frequency_mhz, dtype=float)
    return C_KM_S / (freq * 1e6)


def _is_valid_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def _event_side_coordinate(t_rel_sec: np.ndarray, event_type: str | np.ndarray) -> np.ndarray:
    """Return positive time on the direct-visible side of the event."""
    t = np.asarray(t_rel_sec, dtype=float)
    et = np.asarray(event_type)
    if et.ndim == 0:
        sign = -1.0 if str(et.item()).lower() == "disappearance" else 1.0
        return sign * t
    signs = np.where(pd.Series(et.astype(str)).str.lower().to_numpy() == "disappearance", -1.0, 1.0)
    return signs * t


def _robust_sem(values: pd.Series | np.ndarray) -> float:
    vals = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size <= 1:
        return np.nan
    scale = robust_sigma(vals - np.nanmedian(vals))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.nanstd(vals, ddof=1))
    return float(scale / np.sqrt(vals.size)) if np.isfinite(scale) and scale > 0 else np.nan


def _finite_percentile(values: pd.Series | np.ndarray, q: float) -> float:
    vals = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    return float(np.nanpercentile(vals, q)) if vals.size else np.nan


def _markdown_table(frame: pd.DataFrame, max_rows: int | None = None) -> str:
    if frame.empty:
        return "No rows."
    work = frame.head(max_rows).copy() if max_rows else frame.copy()
    for col in work.columns:
        if pd.api.types.is_numeric_dtype(work[col]):
            work[col] = work[col].map(lambda x: "" if pd.isna(x) else f"{x:.4g}")
    cols = list(work.columns)
    widths = {col: max(len(str(col)), *(len(str(v)) for v in work[col])) for col in cols}
    lines = [
        "| " + " | ".join(str(col).ljust(widths[col]) for col in cols) + " |",
        "| " + " | ".join("-" * widths[col] for col in cols) + " |",
    ]
    for _, row in work.iterrows():
        lines.append("| " + " | ".join(str(row[col]).ljust(widths[col]) for col in cols) + " |")
    return "\n".join(lines)


def specular_reflection_geometry(
    spacecraft_position_km: np.ndarray,
    earth_unit_vector_from_moon: np.ndarray,
    moon_radius_km: float = MOON_RADIUS_KM,
    earth_moon_distance_km: float = EARTH_MOON_DISTANCE_KM,
    n_iter: int = 64,
) -> dict[str, np.ndarray]:
    """Solve the spherical-lunar specular point and exact path difference.

    The source, lunar centre, specular point, and receiver are coplanar.  In
    that plane, ``alpha`` is the surface-normal angle measured from the
    Moon-to-source direction toward the Moon-to-spacecraft direction.  The
    physical specular point is the root of

    ``(u_to_source + u_to_spacecraft) dot d(n)/d(alpha) = 0``.

    A vectorised bisection is used on the portion of the lunar surface visible
    to both endpoints.  This is slower than the former fixed-point update but
    remains well conditioned as the common visible arc collapses at the lunar
    limb.  That limit is precisely where sub-wavelength path accuracy matters
    for this analysis.
    """
    spacecraft = np.asarray(spacecraft_position_km, dtype=float)
    earth_unit = normalize_vectors(earth_unit_vector_from_moon)
    earth = earth_unit * np.asarray(earth_moon_distance_km)[..., None]
    sc_unit = normalize_vectors(spacecraft)
    radius = float(moon_radius_km)

    earth_distance = np.linalg.norm(earth, axis=1)
    spacecraft_distance = np.linalg.norm(spacecraft, axis=1)
    cos_beta = np.clip(np.einsum("ij,ij->i", earth_unit, sc_unit), -1.0, 1.0)
    beta = np.arccos(cos_beta)

    # Complete an orthonormal basis for the source-spacecraft great-circle.
    tangent = sc_unit - cos_beta[:, None] * earth_unit
    tangent_norm = np.linalg.norm(tangent, axis=1)
    regular = tangent_norm > 1e-13
    tangent[regular] /= tangent_norm[regular, None]
    if np.any(~regular):
        seed = np.zeros_like(earth_unit[~regular])
        seed[:, 0] = 1.0
        parallel = np.abs(np.einsum("ij,ij->i", seed, earth_unit[~regular])) > 0.9
        seed[parallel] = np.array([0.0, 1.0, 0.0])
        seed -= np.einsum("ij,ij->i", seed, earth_unit[~regular])[:, None] * earth_unit[~regular]
        tangent[~regular] = normalize_vectors(seed)

    with np.errstate(divide="ignore", invalid="ignore"):
        earth_horizon = np.arccos(np.clip(radius / earth_distance, -1.0, 1.0))
        spacecraft_horizon = np.arccos(np.clip(radius / spacecraft_distance, -1.0, 1.0))
    common_arc = (
        np.isfinite(beta)
        & np.isfinite(earth_horizon)
        & np.isfinite(spacecraft_horizon)
        & (earth_distance > radius)
        & (spacecraft_distance > radius)
        & (beta <= earth_horizon + spacecraft_horizon + 1e-12)
    )

    lo = np.maximum(0.0, beta - spacecraft_horizon)
    hi = np.minimum(beta, earth_horizon)
    alpha = 0.5 * (lo + hi)

    def mirror_tangent_component(angle: np.ndarray) -> np.ndarray:
        normal = np.cos(angle)[:, None] * earth_unit + np.sin(angle)[:, None] * tangent
        normal_tangent = -np.sin(angle)[:, None] * earth_unit + np.cos(angle)[:, None] * tangent
        point_local = radius * normal
        u_earth_local = normalize_vectors(earth - point_local)
        u_spacecraft_local = normalize_vectors(spacecraft - point_local)
        return np.einsum("ij,ij->i", u_earth_local + u_spacecraft_local, normal_tangent)

    g_lo = mirror_tangent_component(lo)
    g_hi = mirror_tangent_component(hi)
    bracketed = common_arc & np.isfinite(g_lo) & np.isfinite(g_hi) & (g_lo * g_hi <= 0.0)
    exact_lo = bracketed & np.isclose(g_lo, 0.0, atol=1e-14)
    exact_hi = bracketed & np.isclose(g_hi, 0.0, atol=1e-14)
    alpha[exact_lo] = lo[exact_lo]
    alpha[exact_hi] = hi[exact_hi]
    active = bracketed & ~exact_lo & ~exact_hi
    left = lo.copy()
    right = hi.copy()
    left_sign = np.sign(g_lo)
    for _ in range(int(n_iter)):
        mid = 0.5 * (left + right)
        g_mid = mirror_tangent_component(mid)
        move_left = active & (np.sign(g_mid) == left_sign)
        left[move_left] = mid[move_left]
        right[active & ~move_left] = mid[active & ~move_left]
    alpha[active] = 0.5 * (left[active] + right[active])

    n = np.cos(alpha)[:, None] * earth_unit + np.sin(alpha)[:, None] * tangent

    point = radius * n
    earth_range = np.linalg.norm(earth - point, axis=1)
    spacecraft_range = np.linalg.norm(spacecraft - point, axis=1)
    direct_range = np.linalg.norm(earth - spacecraft, axis=1)
    delta_l = earth_range + spacecraft_range - direct_range
    # Envelope-theorem derivative of the stationary optical path with respect
    # to the source-spacecraft angle.  Its magnitude is the effective projected
    # baseline that controls spatial coherence of an extended source.
    reflected_angle_derivative = (
        spacecraft_distance * radius * np.sin(beta - alpha) / spacecraft_range
    )
    direct_angle_derivative = (
        earth_distance * spacecraft_distance * np.sin(beta) / direct_range
    )
    source_angle_path_derivative = reflected_angle_derivative - direct_angle_derivative

    u_earth = normalize_vectors(earth - point)
    u_spacecraft = normalize_vectors(spacecraft - point)
    earth_above_tangent = np.einsum("ij,ij->i", u_earth, n)
    spacecraft_above_tangent = np.einsum("ij,ij->i", u_spacecraft, n)
    mirror_error = np.linalg.norm(n - normalize_vectors(u_earth + u_spacecraft), axis=1)
    reflection_valid = (
        common_arc
        & bracketed
        & np.isfinite(delta_l)
        & np.isfinite(earth_above_tangent)
        & np.isfinite(spacecraft_above_tangent)
        & (earth_above_tangent > 0.0)
        & (spacecraft_above_tangent > 0.0)
    )

    return {
        "delta_l_km": delta_l,
        "reflection_valid": reflection_valid,
        "specular_x": point[:, 0],
        "specular_y": point[:, 1],
        "specular_z": point[:, 2],
        "earth_incidence_sin": earth_above_tangent,
        "spacecraft_emergence_sin": spacecraft_above_tangent,
        "specular_mirror_error": mirror_error,
        "specular_alpha_rad": alpha,
        "source_spacecraft_angle_rad": beta,
        "source_angle_path_derivative_km_per_rad": source_angle_path_derivative,
    }


def tangent_proxy_path_km(
    spacecraft_position_km: np.ndarray,
    earth_unit_vector_from_moon: np.ndarray,
    earth_moon_distance_km: float | np.ndarray = EARTH_MOON_DISTANCE_KM,
) -> tuple[np.ndarray, np.ndarray]:
    """Flat-surface proxy retained as a sanity check, not the science model."""
    spacecraft = np.asarray(spacecraft_position_km, dtype=float)
    earth_dir = earth_direction_from_spacecraft(spacecraft, earth_unit_vector_from_moon, earth_moon_distance_km)
    local_up = normalize_vectors(spacecraft)
    altitude = np.linalg.norm(spacecraft, axis=1) - MOON_RADIUS_KM
    sin_elevation = np.einsum("ij,ij->i", earth_dir, local_up)
    delta = 2.0 * altitude * np.abs(sin_elevation)
    valid = np.isfinite(delta) & (altitude > 0.0)
    return delta, valid


def _make_groups(clean: pd.DataFrame) -> dict[tuple[int, str], tuple[pd.DataFrame, np.ndarray]]:
    groups = {}
    for (band, antenna), grp in clean.groupby(["frequency_band", "antenna"], sort=True):
        g = grp.sort_values("time").reset_index(drop=True)
        groups[(int(band), str(antenna))] = (g, datetime_ns(g["time"]))
    return groups


def _event_window(
    group: pd.DataFrame,
    group_ns: np.ndarray,
    event_time: pd.Timestamp,
    window_s: float,
    valid_only: bool = True,
) -> pd.DataFrame:
    event_ns = pd.Timestamp(event_time).value
    half_ns = int(float(window_s) * 1e9)
    lo = int(np.searchsorted(group_ns, event_ns - half_ns, side="left"))
    hi = int(np.searchsorted(group_ns, event_ns + half_ns, side="right"))
    if hi <= lo:
        return pd.DataFrame()
    local = group.iloc[lo:hi].copy()
    local["t_rel_sec"] = (datetime_ns(local["time"]) - event_ns).astype(float) / 1e9
    keep = np.abs(local["t_rel_sec"].to_numpy(dtype=float)) <= float(window_s)
    keep &= np.isfinite(pd.to_numeric(local["power"], errors="coerce").to_numpy(dtype=float))
    if valid_only and "is_valid" in local.columns:
        keep &= _is_valid_series(local["is_valid"]).to_numpy(dtype=bool)
    return local.loc[keep].copy()


def _normalize_event_power(local: pd.DataFrame, inner_s: float) -> pd.DataFrame:
    out = local.copy()
    t = out["t_rel_sec"].to_numpy(dtype=float)
    y = pd.to_numeric(out["power"], errors="coerce").to_numpy(dtype=float)
    side = np.abs(t) >= float(inner_s)
    if np.count_nonzero(side) < 6:
        out["z_power"] = np.nan
        out["event_power_center"] = np.nan
        out["event_power_sigma"] = np.nan
        return out
    center = float(np.nanmedian(y[side]))
    sigma = robust_sigma(y[side] - center)
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.nanstd(y[side], ddof=1))
    if not np.isfinite(sigma) or sigma <= 0:
        out["z_power"] = np.nan
        out["event_power_center"] = center
        out["event_power_sigma"] = np.nan
        return out
    out["z_power"] = (y - center) / sigma
    out["event_power_center"] = center
    out["event_power_sigma"] = sigma
    return out


def _prepare_events(
    events: pd.DataFrame,
    source: str,
    antennas: list[str],
    bands: list[int],
    max_events_per_band_type: int,
) -> pd.DataFrame:
    work = events[
        events["source_name"].astype(str).str.lower().eq(source.lower())
        & events["antenna"].astype(str).isin(antennas)
        & events["frequency_band"].astype(int).isin(bands)
    ].copy()
    if work.empty:
        return work
    work["predicted_event_time"] = pd.to_datetime(work["predicted_event_time"], errors="coerce")
    work = work.dropna(subset=["predicted_event_time"]).sort_values("predicted_event_time")
    if max_events_per_band_type <= 0:
        return work.reset_index(drop=True)
    parts = []
    by = ["frequency_band", "antenna", "event_type"]
    for _, grp in work.groupby(by, sort=True, dropna=False):
        parts.append(grp.head(int(max_events_per_band_type)))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def collect_reflection_samples(
    clean: pd.DataFrame,
    events: pd.DataFrame,
    source: str,
    antennas: list[str],
    bands: list[int],
    window_s: float,
    inner_s: float,
    time_shifts_s: list[float],
    geometry_model: str,
    max_events_per_band_type: int = 0,
    require_normalized: bool = True,
    valid_only: bool = True,
) -> pd.DataFrame:
    groups = _make_groups(clean)
    work = _prepare_events(events, source, antennas, bands, max_events_per_band_type)
    rows = []

    for _, ev in work.iterrows():
        band = int(ev["frequency_band"])
        antenna = str(ev["antenna"])
        payload = groups.get((band, antenna))
        if payload is None:
            continue
        group, group_ns = payload
        for shift_s in time_shifts_s:
            event_time = pd.Timestamp(ev["predicted_event_time"]) + pd.to_timedelta(float(shift_s), unit="s")
            local = _event_window(group, group_ns, event_time, window_s, valid_only=valid_only)
            if local.empty:
                continue
            local = _normalize_event_power(local, inner_s)
            if require_normalized:
                local = local[np.isfinite(local["z_power"].to_numpy(dtype=float))].copy()
                if local.empty:
                    continue

            spacecraft = local[SPACECRAFT_COLUMNS].to_numpy(dtype=float)
            from rylevonberg.frames import body_positions_from_moon
            earth_position = body_positions_from_moon("earth", local["time"])
            earth_distance = np.linalg.norm(earth_position, axis=1)
            earth_unit = normalize_vectors(earth_position)
            earth_dir = normalize_vectors(earth_position - spacecraft)
            limb_angle = moon_limb_angle_deg(spacecraft, earth_dir)

            if geometry_model == "tangent_proxy":
                delta_l, reflection_valid = tangent_proxy_path_km(spacecraft, earth_unit, earth_distance)
                geom = {
                    "delta_l_km": delta_l,
                    "reflection_valid": reflection_valid,
                    "specular_x": np.full(len(local), np.nan),
                    "specular_y": np.full(len(local), np.nan),
                    "specular_z": np.full(len(local), np.nan),
                    "earth_incidence_sin": np.full(len(local), np.nan),
                    "spacecraft_emergence_sin": np.full(len(local), np.nan),
                    "specular_mirror_error": np.full(len(local), np.nan),
                    "specular_alpha_rad": np.full(len(local), np.nan),
                    "source_spacecraft_angle_rad": np.full(len(local), np.nan),
                    "source_angle_path_derivative_km_per_rad": np.full(len(local), np.nan),
                }
            elif geometry_model == "spherical_specular":
                geom = specular_reflection_geometry(spacecraft, earth_unit, earth_moon_distance_km=earth_distance)
            else:
                raise ValueError(f"Unsupported geometry model: {geometry_model}")

            freq_mhz = float(ev["frequency_mhz"])
            phase = 2.0 * np.pi * geom["delta_l_km"] / float(wavelength_km(freq_mhz))
            local["source_name"] = str(ev["source_name"]).lower()
            local["event_id"] = ev["event_id"]
            local["event_type"] = str(ev["event_type"])
            local["predicted_event_time"] = pd.Timestamp(ev["predicted_event_time"])
            local["centered_event_time"] = event_time
            local["time_shift_s"] = float(shift_s)
            local["frequency_mhz"] = freq_mhz
            local["phase_rad"] = phase
            local["phase_cycles"] = phase / (2.0 * np.pi)
            local["phase_wrapped_rad"] = np.mod(phase, 2.0 * np.pi)
            local["delta_l_km"] = geom["delta_l_km"]
            local["delay_us"] = geom["delta_l_km"] / C_KM_S * 1e6
            local["bandwidth_coherence"] = np.sinc(
                RV_EFFECTIVE_BANDWIDTH_HZ * geom["delta_l_km"] / C_KM_S
            )
            local["reflection_valid"] = geom["reflection_valid"]
            local["direct_limb_angle_deg"] = limb_angle
            local["direct_visible"] = limb_angle >= 0.0
            local["oriented_t_sec"] = _event_side_coordinate(
                local["t_rel_sec"].to_numpy(dtype=float),
                str(ev["event_type"]),
            )
            local["event_visible_side"] = local["oriented_t_sec"] >= 0.0
            local["geometry_model"] = geometry_model
            for col in [
                "specular_x",
                "specular_y",
                "specular_z",
                "earth_incidence_sin",
                "spacecraft_emergence_sin",
                "specular_mirror_error",
                "specular_alpha_rad",
                "source_spacecraft_angle_rad",
                "source_angle_path_derivative_km_per_rad",
            ]:
                local[col] = geom[col]
            effective_baseline = np.abs(geom["source_angle_path_derivative_km_per_rad"])
            local["effective_baseline_km"] = effective_baseline
            local["phase_cycles_per_source_degree"] = (
                effective_baseline / float(wavelength_km(freq_mhz)) * np.pi / 180.0
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                local["source_fwhm_50coherence_arcmin"] = (
                    0.4412712
                    * float(wavelength_km(freq_mhz))
                    / effective_baseline
                    * 180.0
                    / np.pi
                    * 60.0
                )
            rows.append(local)

    if not rows:
        return pd.DataFrame(columns=EMPTY_SAMPLE_COLUMNS)
    out = pd.concat(rows, ignore_index=True, sort=False)
    return attach_fringe_period(out)


def attach_fringe_period(samples: pd.DataFrame) -> pd.DataFrame:
    if samples.empty:
        return samples
    out = samples.copy()
    out["fringe_period_s"] = np.nan
    out["fringe_rate_hz"] = np.nan
    out["temporal_response"] = np.nan
    out["dwell_id"] = -1
    out["within_dwell_cadence_s"] = np.nan
    out["locally_nyquist_resolved"] = False
    group_cols = ["time_shift_s", "frequency_band", "antenna", "event_id", "event_type"]
    for _, idx in out.groupby(group_cols, sort=False).groups.items():
        g = out.loc[idx].sort_values("t_rel_sec")
        valid = g["reflection_valid"].to_numpy(dtype=bool) & np.isfinite(g["phase_rad"].to_numpy(dtype=float))
        if np.count_nonzero(valid) < 2:
            continue
        t = g.loc[valid, "t_rel_sec"].to_numpy(dtype=float)
        phase = g.loc[valid, "phase_rad"].to_numpy(dtype=float)
        valid_index = g.loc[valid].index.to_numpy()

        # RAE-2 samples a frequency in a short dwell, then leaves it while the
        # other eight channels are visited.  Never unwrap or differentiate
        # across that roughly 126 s gap: doing so aliases the predicted phase
        # rate and was the main defect in the original smoke analysis.
        dt = np.diff(t)
        dwell = np.r_[0, np.cumsum(~(np.isfinite(dt) & (dt <= 5.0)))]
        out.loc[valid_index, "dwell_id"] = dwell
        for dwell_number in np.unique(dwell):
            take = np.flatnonzero(dwell == dwell_number)
            if take.size < 2:
                continue
            td_all = t[take]
            unique = np.r_[True, np.diff(td_all) > 0.0]
            take = take[unique]
            if take.size < 2:
                continue
            td = t[take]
            pd = phase[take]
            positive_dt = np.diff(td)
            positive_dt = positive_dt[np.isfinite(positive_dt) & (positive_dt > 0.0)]
            if positive_dt.size == 0:
                continue
            cadence = float(np.nanmedian(positive_dt))
            rate_rad_s = np.gradient(pd, td)
            frequency_hz = np.abs(rate_rad_s) / (2.0 * np.pi)
            period = np.full_like(frequency_hz, np.nan, dtype=float)
            changing = np.isfinite(frequency_hz) & (frequency_hz > 0.0)
            period[changing] = 1.0 / frequency_hz[changing]
            response = 1.0 / np.sqrt(
                1.0 + (2.0 * np.pi * frequency_hz * RV_COARSE_TIME_CONSTANT_S) ** 2
            )
            target_index = valid_index[take]
            out.loc[target_index, "fringe_period_s"] = period
            out.loc[target_index, "fringe_rate_hz"] = frequency_hz
            out.loc[target_index, "temporal_response"] = response
            out.loc[target_index, "within_dwell_cadence_s"] = cadence
            out.loc[target_index, "locally_nyquist_resolved"] = period >= 2.0 * cadence
    return out


def phase_rate_summary(samples: pd.DataFrame) -> pd.DataFrame:
    if samples.empty:
        return pd.DataFrame()
    rows = []
    by = ["source_name", "antenna", "frequency_band", "frequency_mhz", "event_type", "time_shift_s"]
    for keys, grp in samples.groupby(by, sort=True, dropna=False):
        valid_period = grp["reflection_valid"] & np.isfinite(grp["fringe_period_s"])
        periods = grp.loc[valid_period, "fringe_period_s"]
        cadence_rows = []
        for _, evg in grp.groupby("event_id", sort=False):
            t = np.sort(evg["t_rel_sec"].to_numpy(dtype=float))
            dt = np.diff(t)
            dt = dt[np.isfinite(dt) & (dt > 0)]
            if dt.size:
                cadence_rows.append(float(np.nanmedian(dt)))
        median_cadence = float(np.nanmedian(cadence_rows)) if cadence_rows else np.nan
        median_period = float(np.nanmedian(periods)) if not periods.empty else np.nan
        rows.append(
            {
                **dict(zip(by, keys)),
                "n_samples": int(len(grp)),
                "n_events": int(grp["event_id"].nunique()),
                "reflection_valid_fraction": float(np.nanmean(grp["reflection_valid"].to_numpy(dtype=bool))),
                "direct_visible_fraction": float(np.nanmean(grp["direct_visible"].to_numpy(dtype=bool))),
                "median_sample_cadence_s": median_cadence,
                "median_fringe_period_s": median_period,
                "p10_fringe_period_s": _finite_percentile(periods, 10),
                "p90_fringe_period_s": _finite_percentile(periods, 90),
                "median_samples_per_fringe": (
                    float(median_period / median_cadence)
                    if np.isfinite(median_period) and np.isfinite(median_cadence) and median_cadence > 0
                    else np.nan
                ),
                "fraction_period_gt_2cadence": (
                    float(np.nanmean(periods.to_numpy(dtype=float) > 2.0 * median_cadence))
                    if not periods.empty and np.isfinite(median_cadence)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _nuisance_matrix(group: pd.DataFrame, baseline_order: int, include_step: bool) -> np.ndarray:
    t = group["t_rel_sec"].to_numpy(dtype=float)
    cols = [baseline_matrix(t, baseline_order)]
    if include_step:
        if "diffraction_power" in group.columns:
            tmpl = pd.to_numeric(group["diffraction_power"], errors="coerce").to_numpy(dtype=float)
        else:
            tmpl = event_template(t, str(group["event_type"].iloc[0]))
        cols.append(tmpl.reshape(-1, 1))
    return np.column_stack(cols)


def _residualize(values: np.ndarray, X: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    good = np.isfinite(vals) & np.isfinite(X).all(axis=1)
    out = np.full(vals.shape, np.nan, dtype=float)
    if np.count_nonzero(good) <= X.shape[1]:
        return out
    beta, *_ = np.linalg.lstsq(X[good], vals[good], rcond=None)
    out[good] = vals[good] - X[good] @ beta
    return out


def add_event_residuals(samples: pd.DataFrame, baseline_order: int = 1, include_step: bool = True) -> pd.DataFrame:
    if samples.empty:
        return samples
    out = samples.copy()
    out["z_resid"] = np.nan
    group_cols = ["time_shift_s", "frequency_band", "antenna", "event_id", "event_type"]
    for _, idx in out.groupby(group_cols, sort=False).groups.items():
        group = out.loc[idx]
        X = _nuisance_matrix(group, baseline_order, include_step)
        out.loc[idx, "z_resid"] = _residualize(group["z_power"].to_numpy(dtype=float), X)
    return out


def _select_fit_side(samples: pd.DataFrame, fit_side: str) -> pd.DataFrame:
    if fit_side == "visible":
        return samples[samples["event_visible_side"] & samples["direct_visible"]].copy()
    if fit_side == "occulted":
        return samples[(~samples["event_visible_side"]) & (~samples["direct_visible"])].copy()
    if fit_side == "all":
        return samples.copy()
    raise ValueError(f"Unsupported fit side: {fit_side}")


def _phase_for_mode(group: pd.DataFrame, phase_mode: str, rng: np.random.Generator) -> np.ndarray:
    phase = group["phase_rad"].to_numpy(dtype=float)
    freq = float(group["frequency_mhz"].iloc[0])
    band = int(group["frequency_band"].iloc[0])
    if phase_mode == "real":
        return phase
    if phase_mode == "scrambled":
        return rng.permutation(phase)
    if phase_mode == "wrong_frequency_low":
        min_band = min(FREQUENCY_MAP_MHZ)
        wrong_band = band - 1 if band > min_band else band + 1
        wrong_freq = FREQUENCY_MAP_MHZ.get(wrong_band, freq)
        return phase * float(wrong_freq) / freq
    if phase_mode == "wrong_frequency_high":
        max_band = max(FREQUENCY_MAP_MHZ)
        wrong_band = band + 1 if band < max_band else band - 1
        wrong_freq = FREQUENCY_MAP_MHZ.get(wrong_band, freq)
        return phase * float(wrong_freq) / freq
    raise ValueError(f"Unsupported phase mode: {phase_mode}")


def _fit_design(cos_r: np.ndarray, sin_r: np.ndarray, y_r: np.ndarray) -> dict[str, float]:
    good = np.isfinite(cos_r) & np.isfinite(sin_r) & np.isfinite(y_r)
    if np.count_nonzero(good) < 5:
        return {
            "fringe_cos_coeff": np.nan,
            "fringe_sin_coeff": np.nan,
            "fringe_amplitude_z": np.nan,
            "fringe_phase_rad": np.nan,
            "fringe_snr": np.nan,
            "fringe_delta_rss": np.nan,
        }
    X = np.column_stack([cos_r[good], sin_r[good]])
    y = y_r[good]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    rss1 = float(np.nansum(resid**2))
    rss0 = float(np.nansum(y**2))
    dof = max(int(len(y) - X.shape[1]), 1)
    sigma2 = rss1 / dof
    cov = np.linalg.pinv(X.T @ X) * sigma2
    amp = float(np.hypot(beta[0], beta[1]))
    if amp > 0 and np.isfinite(cov).all():
        grad = np.array([beta[0] / amp, beta[1] / amp])
        amp_var = float(grad @ cov @ grad)
        amp_se = float(np.sqrt(amp_var)) if np.isfinite(amp_var) and amp_var >= 0 else np.nan
    else:
        amp_se = np.nan
    return {
        "fringe_cos_coeff": float(beta[0]),
        "fringe_sin_coeff": float(beta[1]),
        "fringe_amplitude_z": amp,
        "fringe_amplitude_se_z": amp_se,
        "fringe_phase_rad": float(np.arctan2(-beta[1], beta[0])),
        "fringe_snr": float(amp / amp_se) if np.isfinite(amp_se) and amp_se > 0 else np.nan,
        "fringe_delta_rss": float(max(rss0 - rss1, 0.0)),
    }


def fit_fringe_group(
    group: pd.DataFrame,
    phase_mode: str,
    baseline_order: int,
    include_step: bool,
    rng: np.random.Generator,
) -> dict[str, float]:
    if group.empty:
        return {}
    group = group.reset_index(drop=True).copy()
    phase = _phase_for_mode(group, phase_mode, rng)
    y_parts = []
    cos_parts = []
    sin_parts = []
    for _, evg in group.groupby(["event_id", "event_type"], sort=False):
        local = evg.copy()
        local_phase = phase[evg.index.to_numpy()]
        X = _nuisance_matrix(local, baseline_order, include_step)
        y = local["z_power"].to_numpy(dtype=float)
        if "bandwidth_coherence" in local.columns:
            weight = pd.to_numeric(local["bandwidth_coherence"], errors="coerce").to_numpy(dtype=float).copy()
        else:
            weight = np.ones(len(local), dtype=float)
        if "temporal_response" in local.columns:
            temporal = pd.to_numeric(local["temporal_response"], errors="coerce").to_numpy(dtype=float).copy()
            temporal[~np.isfinite(temporal)] = 1.0
            weight *= temporal
        carrier = np.exp(1j * local_phase)
        if {"direct_diffraction_field_real", "direct_diffraction_field_imag"}.issubset(local.columns):
            direct_field = (
                pd.to_numeric(local["direct_diffraction_field_real"], errors="coerce").to_numpy(dtype=float)
                + 1j * pd.to_numeric(local["direct_diffraction_field_imag"], errors="coerce").to_numpy(dtype=float)
            )
            carrier *= direct_field
        cosv = weight * np.real(carrier)
        sinv = weight * np.imag(carrier)
        y_parts.append(_residualize(y, X))
        cos_parts.append(_residualize(cosv, X))
        sin_parts.append(_residualize(sinv, X))
    y_r = np.concatenate(y_parts) if y_parts else np.array([], dtype=float)
    cos_r = np.concatenate(cos_parts) if cos_parts else np.array([], dtype=float)
    sin_r = np.concatenate(sin_parts) if sin_parts else np.array([], dtype=float)
    fit = _fit_design(cos_r, sin_r, y_r)
    fit.update(
        {
            "n_fit_samples": int(np.count_nonzero(np.isfinite(y_r) & np.isfinite(cos_r) & np.isfinite(sin_r))),
            "n_fit_events": int(group["event_id"].nunique()),
        }
    )
    return fit


def bootstrap_fringe_amplitude(
    group: pd.DataFrame,
    phase_mode: str,
    baseline_order: int,
    include_step: bool,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if n_bootstrap <= 0 or group.empty:
        return np.nan, np.nan
    event_keys = list(group[["event_id", "event_type"]].drop_duplicates().itertuples(index=False, name=None))
    if len(event_keys) < 4:
        return np.nan, np.nan
    by_event = {
        key: grp.copy()
        for key, grp in group.groupby(["event_id", "event_type"], sort=False)
    }
    amps = []
    for _ in range(int(n_bootstrap)):
        draws = rng.integers(0, len(event_keys), len(event_keys))
        parts = []
        for draw_i, event_pos in enumerate(draws):
            part = by_event[event_keys[int(event_pos)]].copy()
            part["bootstrap_event_id"] = draw_i
            parts.append(part)
        boot = pd.concat(parts, ignore_index=True)
        boot["event_id"] = boot["bootstrap_event_id"]
        fit = fit_fringe_group(boot, phase_mode, baseline_order, include_step, rng)
        amp = fit.get("fringe_amplitude_z", np.nan)
        if np.isfinite(amp):
            amps.append(float(amp))
    if not amps:
        return np.nan, np.nan
    return float(np.nanpercentile(amps, 16)), float(np.nanpercentile(amps, 84))


def fit_fringe_models(
    samples: pd.DataFrame,
    fit_side: str,
    phase_modes: list[str],
    baseline_order: int,
    include_step: bool,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    if samples.empty:
        return pd.DataFrame()
    rows = []
    rng = np.random.default_rng(seed)
    base = samples[samples["reflection_valid"].astype(bool)].copy()
    base = _select_fit_side(base, fit_side)
    base = base.reset_index(drop=True)
    group_cols = ["source_name", "antenna", "frequency_band", "frequency_mhz", "time_shift_s"]
    for keys, group in base.groupby(group_cols, sort=True, dropna=False):
        group_variants = [("combined", group)]
        for event_type, et_group in group.groupby("event_type", sort=True):
            group_variants.append((str(event_type), et_group))
        for event_type_label, fit_group in group_variants:
            if fit_group.empty:
                continue
            for phase_mode in phase_modes:
                local_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
                fit = fit_fringe_group(
                    fit_group.reset_index(drop=True),
                    phase_mode,
                    baseline_order,
                    include_step,
                    local_rng,
                )
                lo, hi = bootstrap_fringe_amplitude(
                    fit_group.reset_index(drop=True),
                    phase_mode,
                    baseline_order,
                    include_step,
                    n_bootstrap,
                    np.random.default_rng(rng.integers(0, 2**32 - 1)),
                )
                rows.append(
                    {
                        **dict(zip(group_cols, keys)),
                        "event_type": event_type_label,
                        "fit_side": fit_side,
                        "phase_mode": phase_mode,
                        **fit,
                        "fringe_amplitude_p16_z": lo,
                        "fringe_amplitude_p84_z": hi,
                    }
                )
    return pd.DataFrame(rows)


def stack_event_centered(
    samples: pd.DataFrame,
    value_col: str,
    time_bin_s: float,
    phase_bins: int,
    fit_side: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if samples.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    work = samples[np.isfinite(samples[value_col].to_numpy(dtype=float))].copy()
    work = work[work["time_shift_s"].eq(0.0)].copy()
    if work.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Event-level medians prevent high-cadence windows from dominating a stack.
    work["t_bin_sec"] = np.round(work["t_rel_sec"].to_numpy(dtype=float) / float(time_bin_s)) * float(time_bin_s)
    work["oriented_t_bin_sec"] = (
        np.round(work["oriented_t_sec"].to_numpy(dtype=float) / float(time_bin_s)) * float(time_bin_s)
    )
    phase_wrapped = work["phase_wrapped_rad"].to_numpy(dtype=float)
    phase_bin = np.full(len(work), -1, dtype=int)
    finite_phase = np.isfinite(phase_wrapped)
    phase_bin[finite_phase] = np.floor((phase_wrapped[finite_phase] / (2.0 * np.pi)) * int(phase_bins)).astype(int)
    work["phase_bin"] = np.clip(phase_bin, -1, int(phase_bins) - 1)
    work["phase_bin_center_rad"] = np.where(
        work["phase_bin"].to_numpy(dtype=int) >= 0,
        (work["phase_bin"].to_numpy(dtype=float) + 0.5) * 2.0 * np.pi / int(phase_bins),
        np.nan,
    )

    side = _select_fit_side(work, fit_side)
    time_stack = _summarize_stack(
        work,
        ["source_name", "antenna", "frequency_band", "frequency_mhz", "event_type", "t_bin_sec"],
        ["source_name", "antenna", "frequency_band", "frequency_mhz", "event_type", "event_id", "t_bin_sec"],
        value_col,
    )
    oriented_stack = _summarize_stack(
        work,
        ["source_name", "antenna", "frequency_band", "frequency_mhz", "oriented_t_bin_sec"],
        ["source_name", "antenna", "frequency_band", "frequency_mhz", "event_id", "oriented_t_bin_sec"],
        value_col,
    )
    phase_stack = _summarize_stack(
        side[side["reflection_valid"].astype(bool) & (side["phase_bin"].to_numpy(dtype=int) >= 0)],
        ["source_name", "antenna", "frequency_band", "frequency_mhz", "phase_bin", "phase_bin_center_rad"],
        ["source_name", "antenna", "frequency_band", "frequency_mhz", "event_id", "phase_bin", "phase_bin_center_rad"],
        value_col,
    )
    return time_stack, oriented_stack, phase_stack


def _summarize_stack(
    frame: pd.DataFrame,
    stack_cols: list[str],
    event_bin_cols: list[str],
    value_col: str,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    per_event = (
        frame.groupby(event_bin_cols, as_index=False, dropna=False)
        .agg(event_bin_value=(value_col, "median"), n_samples=(value_col, "size"))
    )
    rows = []
    for keys, grp in per_event.groupby(stack_cols, sort=True, dropna=False):
        vals = grp["event_bin_value"]
        rows.append(
            {
                **dict(zip(stack_cols, keys)),
                "median_value": float(np.nanmedian(vals)),
                "value_sem": _robust_sem(vals),
                "n_events": int(len(grp)),
                "n_samples": int(grp["n_samples"].sum()),
            }
        )
    return pd.DataFrame(rows)


def plot_fringe_period_summary(summary: pd.DataFrame, out_dir: Path) -> Path | None:
    if summary.empty or "time_shift_s" not in summary.columns:
        return None
    sub = summary[summary["time_shift_s"].eq(0.0)].copy()
    if sub.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for antenna, grp in sub.groupby("antenna", sort=True):
        g = grp.groupby("frequency_mhz", as_index=False).agg(
            median_fringe_period_s=("median_fringe_period_s", "median"),
            median_samples_per_fringe=("median_samples_per_fringe", "median"),
        )
        label = ANT_LABEL.get(str(antenna), str(antenna))
        color = ANT_COLOR.get(str(antenna))
        axes[0].plot(g["frequency_mhz"], g["median_fringe_period_s"], marker="o", label=label, color=color)
        axes[1].plot(g["frequency_mhz"], g["median_samples_per_fringe"], marker="o", label=label, color=color)
    axes[0].set_ylabel("median predicted fringe period (s)")
    axes[1].set_ylabel("samples per fringe")
    for ax in axes:
        ax.set_xlabel("frequency (MHz)")
        ax.set_xscale("log")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    fig.suptitle("Specular lunar-reflection phase rate near Earth occultations")
    fig.tight_layout()
    path = out_dir / "fringe_period_by_frequency.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def plot_event_centered_stack(stack: pd.DataFrame, out_dir: Path, antenna: str) -> Path | None:
    if stack.empty or "antenna" not in stack.columns:
        return None
    sub = stack[stack["antenna"].astype(str).eq(str(antenna))].copy()
    if sub.empty:
        return None
    freqs = sorted(sub["frequency_mhz"].dropna().unique())
    fig, axes = plt.subplots(len(freqs), 2, figsize=(12, max(8, 1.35 * len(freqs))), sharex=True, sharey=True)
    if len(freqs) == 1:
        axes = np.asarray([axes])
    for i, freq in enumerate(freqs):
        for j, event_type in enumerate(["disappearance", "reappearance"]):
            ax = axes[i, j]
            grp = sub[np.isclose(sub["frequency_mhz"], float(freq)) & sub["event_type"].astype(str).eq(event_type)].sort_values(
                "t_bin_sec"
            )
            ax.errorbar(
                grp["t_bin_sec"],
                grp["median_value"],
                yerr=grp["value_sem"],
                marker="o",
                markersize=2.0,
                linewidth=1.0,
                elinewidth=0.6,
                capsize=1.0,
                color=ANT_COLOR.get(antenna),
            )
            ax.axvline(0.0, color="black", ls="--", lw=0.7)
            ax.axhline(0.0, color="0.65", lw=0.7)
            ax.set_title(f"{freq:.2f} MHz {event_type}", fontsize=9)
            if j == 0:
                ax.set_ylabel("baseline residual z")
            if i == len(freqs) - 1:
                ax.set_xlabel("seconds from predicted event")
    fig.suptitle(f"Event-centered stack, {ANT_LABEL.get(antenna, antenna)}")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = out_dir / f"{antenna}_event_centered_residual_stack.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def plot_oriented_stack(stack: pd.DataFrame, out_dir: Path, antenna: str) -> Path | None:
    if stack.empty or "antenna" not in stack.columns:
        return None
    sub = stack[stack["antenna"].astype(str).eq(str(antenna))].copy()
    if sub.empty:
        return None
    fig, ax = plt.subplots(figsize=(11, 5.4))
    for freq, grp in sub.groupby("frequency_mhz", sort=True):
        g = grp.sort_values("oriented_t_bin_sec")
        ax.plot(g["oriented_t_bin_sec"], g["median_value"], marker="o", markersize=2.2, lw=1.2, label=f"{float(freq):.2f} MHz")
    ax.axvline(0.0, color="black", ls="--", lw=0.8)
    ax.axhline(0.0, color="0.65", lw=0.7)
    ax.set_xlabel("seconds from event; positive is direct-visible side")
    ax.set_ylabel("baseline residual z")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=3, fontsize=8)
    fig.suptitle(f"Disappearance/reappearance combined in event-centered coordinates, {ANT_LABEL.get(antenna, antenna)}")
    fig.tight_layout()
    path = out_dir / f"{antenna}_oriented_event_centered_stack.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def plot_phase_stack(stack: pd.DataFrame, out_dir: Path, antenna: str) -> Path | None:
    if stack.empty or "antenna" not in stack.columns:
        return None
    sub = stack[stack["antenna"].astype(str).eq(str(antenna))].copy()
    if sub.empty:
        return None
    freqs = sorted(sub["frequency_mhz"].dropna().unique())
    fig, axes = plt.subplots(len(freqs), 1, figsize=(11, max(7, 1.45 * len(freqs))), sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(-1)
    for ax, freq in zip(axes, freqs):
        grp = sub[np.isclose(sub["frequency_mhz"], float(freq))].sort_values("phase_bin_center_rad")
        x = grp["phase_bin_center_rad"] / (2.0 * np.pi)
        ax.errorbar(
            x,
            grp["median_value"],
            yerr=grp["value_sem"],
            marker="o",
            markersize=2.8,
            linewidth=1.2,
            elinewidth=0.65,
            capsize=1.0,
            color=ANT_COLOR.get(antenna),
        )
        ax.axhline(0.0, color="0.65", lw=0.7)
        ax.set_title(f"{freq:.2f} MHz", fontsize=9)
        ax.set_ylabel("residual z")
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("wrapped reflection phase (cycles)")
    fig.suptitle(f"Phase-folded residual stack on direct-visible side, {ANT_LABEL.get(antenna, antenna)}")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = out_dir / f"{antenna}_phase_folded_residual_stack.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def plot_fit_controls(fits: pd.DataFrame, out_dir: Path, antenna: str) -> Path | None:
    if fits.empty or "antenna" not in fits.columns:
        return None
    sub = fits[fits["antenna"].astype(str).eq(str(antenna)) & fits["event_type"].eq("combined")].copy()
    if sub.empty:
        return None
    real = sub[sub["phase_mode"].eq("real")].copy()
    phase_ctl = sub[sub["time_shift_s"].eq(0.0) & sub["phase_mode"].isin(["real", "scrambled", "wrong_frequency_low", "wrong_frequency_high"])].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    for shift_s, grp in real.groupby("time_shift_s", sort=True):
        g = grp.sort_values("frequency_mhz")
        label = "real time" if float(shift_s) == 0.0 else f"shift {float(shift_s):.0f} s"
        axes[0].plot(g["frequency_mhz"], g["fringe_amplitude_z"], marker="o", lw=1.2, label=label)
    for mode, grp in phase_ctl.groupby("phase_mode", sort=True):
        g = grp.sort_values("frequency_mhz")
        axes[1].plot(g["frequency_mhz"], g["fringe_amplitude_z"], marker="o", lw=1.2, label=mode.replace("_", " "))
    for ax in axes:
        ax.set_xlabel("frequency (MHz)")
        ax.set_ylabel("fringe amplitude (z)")
        ax.set_xscale("log")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_title("Real phase at true and shifted event centers")
    axes[1].set_title("True-time phase-law controls")
    fig.suptitle(f"Cosine/sine fringe fit controls, {ANT_LABEL.get(antenna, antenna)}")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = out_dir / f"{antenna}_fringe_fit_controls.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def write_report(
    out_dir: Path,
    config: dict[str, object],
    rate_summary: pd.DataFrame,
    fits: pd.DataFrame,
    paths: list[Path],
) -> Path:
    if fits.empty or not {"time_shift_s", "phase_mode", "event_type"}.issubset(fits.columns):
        real_combined = pd.DataFrame()
    else:
        real_combined = fits[
            fits["time_shift_s"].eq(0.0)
            & fits["phase_mode"].eq("real")
            & fits["event_type"].eq("combined")
        ].copy()
    lines = [
        "# Lower-V Lunar Reflection Interference Search",
        "",
        "## Purpose",
        "",
        "This run searches for sea-cliff-like interference between direct Earth emission and lunar-reflected Earth emission.",
        "Samples are centered on predicted Earth occultation times before stacking, so narrow structure is not averaged away by the all-frequency grid bins.",
        "",
        "## Configuration",
        "",
        _markdown_table(pd.DataFrame([config]).T.reset_index().rename(columns={"index": "parameter", 0: "value"})),
        "",
        "## Phase-Rate Gate",
        "",
    ]
    cols = [
        "antenna",
        "frequency_mhz",
        "event_type",
        "time_shift_s",
        "n_events",
        "reflection_valid_fraction",
        "direct_visible_fraction",
        "median_sample_cadence_s",
        "median_fringe_period_s",
        "median_samples_per_fringe",
        "fraction_period_gt_2cadence",
    ]
    if rate_summary.empty or "time_shift_s" not in rate_summary.columns:
        gate = pd.DataFrame()
    else:
        gate = rate_summary[rate_summary["time_shift_s"].eq(0.0)].copy()
    lines.append(_markdown_table(gate[cols].sort_values(["antenna", "frequency_mhz", "event_type"]), max_rows=40) if not gate.empty else "No phase-rate rows.")
    lines.extend(["", "## True-Time Combined Fringe Fits", ""])
    fit_cols = [
        "antenna",
        "frequency_mhz",
        "fit_side",
        "n_fit_events",
        "n_fit_samples",
        "fringe_amplitude_z",
        "fringe_amplitude_se_z",
        "fringe_snr",
        "fringe_amplitude_p16_z",
        "fringe_amplitude_p84_z",
    ]
    lines.append(_markdown_table(real_combined[fit_cols].sort_values(["antenna", "frequency_mhz"]), max_rows=40) if not real_combined.empty else "No real combined fit rows.")
    lines.extend(
        [
            "",
            "## Read Rules",
            "",
            "- A plausible reflection-interference signature should be strongest at `time_shift_s = 0` with `phase_mode = real`.",
            "- Wrong-frequency and scrambled phase modes should reduce the fitted amplitude.",
            "- Lower V should be favored over upper V if the Moon-facing receiver is seeing the reflected component.",
            "- The `median_samples_per_fringe` gate matters: values near or below 2 mean the phase model is poorly sampled.",
            "- Because low-frequency Earth windows already contain broad event-selected background slopes, shifted-time controls are mandatory.",
            "",
            "## Generated Files",
            "",
            "- `reflection_event_samples.npy`",
            "- `reflection_phase_rate_summary.csv`",
            "- `reflection_fringe_fit_summary.csv`",
            "- `event_centered_time_stack.csv`",
            "- `oriented_event_centered_time_stack.csv`",
            "- `phase_folded_stack.csv`",
            "",
            "## Plots",
            "",
        ]
    )
    lines.extend(f"- `{path.name}`" for path in paths)
    path = out_dir / "lower_v_lunar_reflection_interference_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(OUT))
    parser.add_argument("--clean", default=str(CLEAN))
    parser.add_argument("--events", default=str(EVENTS))
    parser.add_argument("--source", default="earth")
    parser.add_argument("--antennas", default="rv2_coarse,rv1_coarse")
    parser.add_argument("--bands", default="1,2,3,4,5")
    parser.add_argument("--window-s", type=float, default=600.0)
    parser.add_argument("--inner-s", type=float, default=120.0)
    parser.add_argument("--time-shifts-s", default="0,-300,300,-600,600")
    parser.add_argument("--geometry-model", choices=["spherical_specular", "tangent_proxy"], default="spherical_specular")
    parser.add_argument("--fit-side", choices=["visible", "occulted", "all"], default="visible")
    parser.add_argument("--phase-modes", default="real,scrambled,wrong_frequency_low,wrong_frequency_high")
    parser.add_argument("--baseline-order", type=int, default=1)
    parser.add_argument("--no-step-nuisance", action="store_true")
    parser.add_argument("--time-bin-s", type=float, default=10.0)
    parser.add_argument("--phase-bins", type=int, default=24)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--max-events-per-band-type", type=int, default=0)
    parser.add_argument("--table-format", choices=["csv", "npy"], default="npy")
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    antennas = parse_csv_list(args.antennas, str)
    bands = parse_csv_list(args.bands, int)
    time_shifts = parse_csv_list(args.time_shifts_s, float)
    phase_modes = parse_csv_list(args.phase_modes, str)
    config = {
        "source": args.source,
        "antennas": ",".join(antennas),
        "bands": ",".join(str(x) for x in bands),
        "window_s": float(args.window_s),
        "inner_s": float(args.inner_s),
        "time_shifts_s": ",".join(f"{x:g}" for x in time_shifts),
        "geometry_model": args.geometry_model,
        "fit_side": args.fit_side,
        "phase_modes": ",".join(phase_modes),
        "baseline_order": int(args.baseline_order),
        "include_step_nuisance": not args.no_step_nuisance,
        "time_bin_s": float(args.time_bin_s),
        "phase_bins": int(args.phase_bins),
        "bootstrap": int(args.bootstrap),
        "max_events_per_band_type": int(args.max_events_per_band_type),
        "cleaned_timeseries": str(args.clean),
        "events": str(args.events),
        "software_versions": software_versions(),
    }
    write_json(out_dir / "run_config.json", config)

    usecols = [
        "time",
        "frequency_band",
        "antenna",
        "power",
        "is_valid",
        *SPACECRAFT_COLUMNS,
        *EARTH_UNIT_COLUMNS,
    ]
    clean = read_table(args.clean, usecols=usecols, parse_dates=["time"], low_memory=False)
    clean = clean[
        clean["antenna"].astype(str).isin(antennas)
        & clean["frequency_band"].astype(int).isin(bands)
    ].copy()
    events = read_table(args.events, parse_dates=["predicted_event_time"], low_memory=False)

    samples = collect_reflection_samples(
        clean,
        events,
        args.source,
        antennas,
        bands,
        float(args.window_s),
        float(args.inner_s),
        time_shifts,
        args.geometry_model,
        int(args.max_events_per_band_type),
    )
    samples = add_event_residuals(samples, int(args.baseline_order), include_step=not args.no_step_nuisance)
    write_table(samples, out_dir / f"reflection_event_samples.{args.table_format}", index=False)

    rate_summary = phase_rate_summary(samples)
    rate_summary.to_csv(out_dir / "reflection_phase_rate_summary.csv", index=False)

    fits = fit_fringe_models(
        samples,
        args.fit_side,
        phase_modes,
        int(args.baseline_order),
        include_step=not args.no_step_nuisance,
        n_bootstrap=int(args.bootstrap),
        seed=int(args.seed),
    )
    fits.to_csv(out_dir / "reflection_fringe_fit_summary.csv", index=False)

    time_stack, oriented_stack, phase_stack = stack_event_centered(
        samples,
        "z_resid",
        float(args.time_bin_s),
        int(args.phase_bins),
        args.fit_side,
    )
    time_stack.to_csv(out_dir / "event_centered_time_stack.csv", index=False)
    oriented_stack.to_csv(out_dir / "oriented_event_centered_time_stack.csv", index=False)
    phase_stack.to_csv(out_dir / "phase_folded_stack.csv", index=False)

    paths: list[Path] = []
    for path in [
        plot_fringe_period_summary(rate_summary, out_dir),
        *[plot_event_centered_stack(time_stack, out_dir, antenna) for antenna in antennas],
        *[plot_oriented_stack(oriented_stack, out_dir, antenna) for antenna in antennas],
        *[plot_phase_stack(phase_stack, out_dir, antenna) for antenna in antennas],
        *[plot_fit_controls(fits, out_dir, antenna) for antenna in antennas],
    ]:
        if path is not None:
            paths.append(path)
    report = write_report(out_dir, config, rate_summary, fits, paths)
    print(report)


if __name__ == "__main__":
    main()
