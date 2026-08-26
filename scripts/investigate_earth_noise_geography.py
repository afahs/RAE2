#!/usr/bin/env python
"""Identify RAE-2 Earth-noise bands and test coarse geographic associations.

The Ryle-Vonberg antennas do not image the Earth.  This analysis therefore
does not assign a surface origin to an individual sample.  It asks the more
limited questions the archive can support: whether occultation contrast is
associated with the projected land, nighttime-land, or nighttime auroral-zone
fraction of the Earth-facing disk, and whether a disk-centre land/ocean split
is repeatable after month/event-type stratification.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from urllib.request import urlopen

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.table_io import read_table  # noqa: E402
from rylevonberg.util import ensure_dir, robust_sigma, software_versions, write_json  # noqa: E402


DEFAULT_EVENTS = ROOT / "outputs/planetary_confirmation_survey_science_baseline_v2/events/earth_predicted_events.csv"
DEFAULT_CONTRASTS = ROOT / "outputs/earth_lowfreq_profile_sign_diagnostics_v1/earth_event_level_prepost_contrasts_with_shifts.csv"
DEFAULT_CHANNEL_SCAN = ROOT / "outputs/planetary_confirmation_survey_science_baseline_v2/earth/initial_channel_scan.csv"
DEFAULT_OUT = ROOT / "outputs/earth_noise_geography_v1"
NATURAL_EARTH_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
    "master/geojson/ne_110m_admin_0_countries.geojson"
)

GEOMAGNETIC_NORTH_POLE_1975_LAT_DEG = 78.76
GEOMAGNETIC_NORTH_POLE_1975_LON_DEG = -70.47

CHANNEL_INTERPRETATION = {
    1: ("natural magnetospheric", "inside the 50-750 kHz AKR/TKR range; close to its upper side"),
    2: ("mixed AKR/MF", "near the AKR upper cutoff and inside the MF broadcast allocation"),
    3: ("MF broadcast", "inside the 525/535-1605 kHz medium-wave broadcast allocation"),
    4: ("MF broadcast", "inside the 525/535-1605 kHz medium-wave broadcast allocation"),
    5: ("maritime/MF", "40 kHz channel overlaps the 2182 kHz maritime distress/calling region"),
    6: ("HF ground breakthrough", "near 75 m broadcast/amateur use and the historical ~4 MHz breakthrough turn-up"),
    7: ("HF ground breakthrough", "above the historical ~4 MHz terrestrial ground-breakthrough transition"),
    8: ("HF ground breakthrough", "HF fixed/mobile/aeronautical region; ionospheric escape expected"),
    9: ("HF ground breakthrough", "HF/shortwave region; strongest historical RAE terrestrial turn-up"),
}

ASSOCIATION_METRICS = [
    "visible_land_fraction",
    "visible_night_land_fraction",
    "night_land_share_of_night_disk",
    "visible_night_fraction",
    "visible_night_auroral_fraction",
    "visible_auroral_fraction",
    "night_auroral_share_of_night_disk",
    "abs_moon_geomagnetic_lat_deg",
    "submoon_point_is_land",
]

COMPOSITION_METRICS = [
    "night_land_share_of_night_disk",
    "night_auroral_share_of_night_disk",
]


def _unit_from_lon_lat(lon_deg: np.ndarray | float, lat_deg: np.ndarray | float) -> np.ndarray:
    lon = np.deg2rad(np.asarray(lon_deg, dtype=float))
    lat = np.deg2rad(np.asarray(lat_deg, dtype=float))
    return np.column_stack(
        [
            np.cos(lat) * np.cos(lon),
            np.cos(lat) * np.sin(lon),
            np.sin(lat),
        ]
    )


def _lon_lat_from_unit(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vec = np.asarray(vectors, dtype=float)
    lon = np.degrees(np.arctan2(vec[:, 1], vec[:, 0]))
    lat = np.degrees(np.arcsin(np.clip(vec[:, 2], -1.0, 1.0)))
    return lon, lat


def _normalise(vectors: np.ndarray) -> np.ndarray:
    vec = np.asarray(vectors, dtype=float)
    norm = np.linalg.norm(vec, axis=1, keepdims=True)
    return vec / np.maximum(norm, 1e-30)


def load_land_geojson(cache_path: Path) -> dict:
    """Load Natural Earth countries, caching the generated analysis input."""
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    with urlopen(NATURAL_EARTH_URL, timeout=60) as response:
        payload = response.read()
    cache_path.write_bytes(payload)
    return json.loads(payload)


def _iter_polygons(geometry: dict):
    kind = geometry.get("type")
    coords = geometry.get("coordinates", [])
    if kind == "Polygon":
        yield coords
    elif kind == "MultiPolygon":
        yield from coords


def land_mask_from_geojson(lon_deg: np.ndarray, lat_deg: np.ndarray, geojson: dict) -> np.ndarray:
    """Classify lon/lat points using Natural Earth polygon exteriors and holes."""
    lon = np.asarray(lon_deg, dtype=float)
    lat = np.asarray(lat_deg, dtype=float)
    points = np.column_stack([lon, lat])
    land = np.zeros(len(points), dtype=bool)
    for feature in geojson.get("features", []):
        geometry = feature.get("geometry") or {}
        for polygon in _iter_polygons(geometry):
            if not polygon:
                continue
            exterior = np.asarray(polygon[0], dtype=float)
            if exterior.ndim != 2 or len(exterior) < 3:
                continue
            inside = MplPath(exterior).contains_points(points, radius=1e-10)
            for hole in polygon[1:]:
                ring = np.asarray(hole, dtype=float)
                if ring.ndim == 2 and len(ring) >= 3:
                    inside &= ~MplPath(ring).contains_points(points, radius=1e-10)
            land |= inside
    return land


def build_surface_grid(geojson: dict, step_deg: float = 2.0) -> dict[str, np.ndarray]:
    lat_axis = np.arange(-90.0 + step_deg / 2.0, 90.0, step_deg)
    lon_axis = np.arange(-180.0 + step_deg / 2.0, 180.0, step_deg)
    lon, lat = np.meshgrid(lon_axis, lat_axis)
    lon_flat = lon.ravel()
    lat_flat = lat.ravel()
    xyz = _unit_from_lon_lat(lon_flat, lat_flat)
    pole = _unit_from_lon_lat(
        np.asarray([GEOMAGNETIC_NORTH_POLE_1975_LON_DEG]),
        np.asarray([GEOMAGNETIC_NORTH_POLE_1975_LAT_DEG]),
    )[0]
    magnetic_lat = np.degrees(np.arcsin(np.clip(xyz @ pole, -1.0, 1.0)))
    return {
        "lon_deg": lon_flat,
        "lat_deg": lat_flat,
        "xyz": xyz,
        "area_weight": np.cos(np.deg2rad(lat_flat)),
        "is_land": land_mask_from_geojson(lon_flat, lat_flat, geojson),
        "magnetic_lat_deg": magnetic_lat,
    }


def earth_fixed_directions(times: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Return geocentric Earth-fixed directions to Moon and Sun."""
    from astropy.coordinates import ITRS, get_body, get_sun
    from astropy.time import Time
    from astropy.utils import iers

    iers.conf.auto_download = False
    obstime = Time(pd.to_datetime(times, utc=True).dt.to_pydatetime(), scale="utc")
    moon = get_body("moon", obstime).transform_to(ITRS(obstime=obstime)).cartesian
    sun = get_sun(obstime).transform_to(ITRS(obstime=obstime)).cartesian
    moon_xyz = np.column_stack([moon.x.value, moon.y.value, moon.z.value])
    sun_xyz = np.column_stack([sun.x.value, sun.y.value, sun.z.value])
    return _normalise(moon_xyz), _normalise(sun_xyz)


def event_geometry(events: pd.DataFrame, grid: dict[str, np.ndarray], geojson: dict) -> pd.DataFrame:
    unique = (
        events[["event_id", "event_type", "predicted_event_time"]]
        .drop_duplicates("event_id")
        .sort_values("predicted_event_time")
        .reset_index(drop=True)
    )
    moon, sun = earth_fixed_directions(unique["predicted_event_time"])
    moon_lon, moon_lat = _lon_lat_from_unit(moon)
    sun_lon, sun_lat = _lon_lat_from_unit(sun)
    center_land = land_mask_from_geojson(moon_lon, moon_lat, geojson)
    pole = _unit_from_lon_lat(
        np.asarray([GEOMAGNETIC_NORTH_POLE_1975_LON_DEG]),
        np.asarray([GEOMAGNETIC_NORTH_POLE_1975_LAT_DEG]),
    )[0]

    xyz = grid["xyz"]
    area = grid["area_weight"]
    is_land = grid["is_land"]
    magnetic_lat = grid["magnetic_lat_deg"]
    auroral = (np.abs(magnetic_lat) >= 55.0) & (np.abs(magnetic_lat) <= 80.0)
    north_auroral = (magnetic_lat >= 55.0) & (magnetic_lat <= 80.0)
    south_auroral = (magnetic_lat <= -55.0) & (magnetic_lat >= -80.0)
    rows: list[dict[str, object]] = []
    for i in range(len(unique)):
        projection = xyz @ moon[i]
        visible = projection > 0.0
        weight = np.where(visible, projection * area, 0.0)
        total = float(np.sum(weight))
        night = (xyz @ sun[i]) < 0.0

        def fraction(mask: np.ndarray) -> float:
            return float(np.sum(weight[mask]) / total) if total > 0.0 else np.nan

        rows.append(
            {
                "event_id": unique.loc[i, "event_id"],
                "event_type": unique.loc[i, "event_type"],
                "predicted_event_time": unique.loc[i, "predicted_event_time"],
                "month": pd.Timestamp(unique.loc[i, "predicted_event_time"]).to_period("M").strftime("%Y-%m"),
                "submoon_lon_deg": moon_lon[i],
                "submoon_lat_deg": moon_lat[i],
                "subsolar_lon_deg": sun_lon[i],
                "subsolar_lat_deg": sun_lat[i],
                "submoon_point_is_land": bool(center_land[i]),
                "submoon_solar_zenith_deg": float(
                    np.degrees(np.arccos(np.clip(np.dot(moon[i], sun[i]), -1.0, 1.0)))
                ),
                "visible_land_fraction": fraction(is_land),
                "visible_ocean_fraction": fraction(~is_land),
                "visible_night_fraction": fraction(night),
                "visible_night_land_fraction": fraction(night & is_land),
                "visible_day_land_fraction": fraction((~night) & is_land),
                "visible_night_ocean_fraction": fraction(night & (~is_land)),
                "visible_night_auroral_fraction": fraction(night & auroral),
                "visible_auroral_fraction": fraction(auroral),
                "visible_night_north_auroral_fraction": fraction(night & north_auroral),
                "visible_night_south_auroral_fraction": fraction(night & south_auroral),
                "moon_geomagnetic_lat_deg": float(
                    np.degrees(np.arcsin(np.clip(np.dot(moon[i], pole), -1.0, 1.0)))
                ),
            }
        )
    out = pd.DataFrame(rows)
    out["abs_moon_geomagnetic_lat_deg"] = out["moon_geomagnetic_lat_deg"].abs()
    sufficiently_night = out["visible_night_fraction"] >= 0.10
    out["night_land_share_of_night_disk"] = np.where(
        sufficiently_night,
        out["visible_night_land_fraction"] / out["visible_night_fraction"],
        np.nan,
    )
    out["night_auroral_share_of_night_disk"] = np.where(
        sufficiently_night,
        out["visible_night_auroral_fraction"] / out["visible_night_fraction"],
        np.nan,
    )
    return out


def prepare_event_responses(contrasts: pd.DataFrame, geometry: pd.DataFrame) -> pd.DataFrame:
    key = ["event_id", "event_type", "frequency_band", "frequency_mhz", "antenna"]
    usable = contrasts[
        (contrasts["n_pre"] >= 3)
        & (contrasts["n_post"] >= 3)
        & np.isfinite(contrasts["source_like_signed_contrast_z"])
        & (~contrasts["has_opposite_event_within_window"].astype(bool))
    ].copy()
    control = usable[
        usable["time_shift_s"].abs().isin([600.0, 1200.0])
    ].groupby(key, as_index=False).agg(
        shifted_control_median_z=("source_like_signed_contrast_z", "median"),
        n_shifted_controls=("source_like_signed_contrast_z", "size"),
    )
    true = usable[usable["time_shift_s"].eq(0.0)].copy()
    true = true.merge(control, on=key, how="left")
    true["shifted_control_median_z"] = true["shifted_control_median_z"].fillna(0.0)
    true["response_excess_z"] = (
        true["source_like_signed_contrast_z"] - true["shifted_control_median_z"]
    )
    return true.merge(geometry, on=["event_id", "event_type"], how="inner", validate="many_to_one")


def _robust_se(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 2:
        return np.nan
    sigma = robust_sigma(vals)
    return float(sigma / np.sqrt(len(vals))) if np.isfinite(sigma) else np.nan


def spectral_summary(responses: pd.DataFrame, channel_scan: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    scan = channel_scan[channel_scan["window_s"].eq(600.0) & channel_scan["timing_offset_s"].eq(0.0)]
    scan_lookup = scan.set_index(["frequency_band", "antenna"])
    for keys, group in responses.groupby(["frequency_band", "frequency_mhz", "antenna"], sort=True):
        band, freq, antenna = keys
        y = group["response_excess_z"].to_numpy(dtype=float)
        med = float(np.nanmedian(y))
        se = _robust_se(y)
        lookup_key = (int(band), str(antenna))
        historical_snr = (
            float(scan_lookup.loc[lookup_key, "robust_stack_snr"])
            if lookup_key in scan_lookup.index
            else np.nan
        )
        family, interpretation = CHANNEL_INTERPRETATION[int(band)]
        rows.append(
            {
                "frequency_band": int(band),
                "frequency_mhz": float(freq),
                "antenna": str(antenna),
                "n_events": int(len(group)),
                "median_true_source_like_z": float(np.nanmedian(group["source_like_signed_contrast_z"])),
                "median_shift_corrected_source_like_z": med,
                "robust_se_shift_corrected_z": se,
                "robust_z_shift_corrected": med / se if np.isfinite(se) and se > 0.0 else np.nan,
                "positive_shift_corrected_fraction": float(np.mean(y > 0.0)),
                "existing_survey_robust_stack_snr": historical_snr,
                "expected_emission_family": family,
                "band_context": interpretation,
            }
        )
    return pd.DataFrame(rows)


def _rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    xr = rankdata(x)
    yr = rankdata(y)
    if np.std(xr) == 0.0 or np.std(yr) == 0.0:
        return np.nan
    return float(np.corrcoef(xr, yr)[0, 1])


def _stratified_permutation_p(
    x: np.ndarray,
    y: np.ndarray,
    strata: np.ndarray,
    trials: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    observed = _rank_correlation(x, y)
    if not np.isfinite(observed) or trials <= 0:
        return observed, np.nan
    blocks = [np.flatnonzero(strata == value) for value in np.unique(strata)]
    exceed = 0
    for _ in range(int(trials)):
        permuted = y.copy()
        for idx in blocks:
            if len(idx) > 1:
                permuted[idx] = y[rng.permutation(idx)]
        value = _rank_correlation(x, permuted)
        exceed += int(np.isfinite(value) and abs(value) >= abs(observed))
    return observed, (1.0 + exceed) / (1.0 + int(trials))


def _bh_qvalues(pvalues: pd.Series) -> pd.Series:
    p = pd.to_numeric(pvalues, errors="coerce").to_numpy(dtype=float)
    q = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if not len(valid):
        return pd.Series(q, index=pvalues.index)
    order = valid[np.argsort(p[valid])]
    adjusted = p[order] * len(order) / np.arange(1, len(order) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    q[order] = np.clip(adjusted, 0.0, 1.0)
    return pd.Series(q, index=pvalues.index)


def _rank_residuals(values: np.ndarray, control: np.ndarray) -> np.ndarray:
    """Residualise ranks against one continuous control, including an intercept."""
    ranked = rankdata(np.asarray(values, dtype=float))
    ranked_control = rankdata(np.asarray(control, dtype=float))
    design = np.column_stack([np.ones(len(ranked_control)), ranked_control])
    coefficients = np.linalg.lstsq(design, ranked, rcond=None)[0]
    return ranked - design @ coefficients


def _partial_rank_correlation(x: np.ndarray, y: np.ndarray, control: np.ndarray) -> float:
    x_residual = _rank_residuals(x, control)
    y_residual = _rank_residuals(y, control)
    if np.std(x_residual) == 0.0 or np.std(y_residual) == 0.0:
        return np.nan
    return float(np.corrcoef(x_residual, y_residual)[0, 1])


def association_tests(responses: pd.DataFrame, trials: int, seed: int) -> pd.DataFrame:
    combined = (
        responses.groupby(["event_id", "event_type", "frequency_band", "frequency_mhz"], as_index=False)
        .agg(
            response_excess_z=("response_excess_z", "median"),
            month=("month", "first"),
            **{metric: (metric, "first") for metric in ASSOCIATION_METRICS},
        )
    )
    combined["antenna"] = "combined"
    analysis = pd.concat(
        [responses[["event_id", "event_type", "frequency_band", "frequency_mhz", "antenna", "response_excess_z", "month", *ASSOCIATION_METRICS]], combined],
        ignore_index=True,
        sort=False,
    )
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
    for keys, group in analysis.groupby(["frequency_band", "frequency_mhz", "antenna"], sort=True):
        band, freq, antenna = keys
        y_all = group["response_excess_z"].to_numpy(dtype=float)
        strata_all = (group["month"].astype(str) + "|" + group["event_type"].astype(str)).to_numpy()
        for metric in ASSOCIATION_METRICS:
            x_all = group[metric].astype(float).to_numpy()
            good = np.isfinite(x_all) & np.isfinite(y_all)
            if np.count_nonzero(good) < 30:
                continue
            rho, p = _stratified_permutation_p(
                x_all[good], y_all[good], strata_all[good], trials, rng
            )
            land = good & group["submoon_point_is_land"].to_numpy(dtype=bool)
            ocean = good & ~group["submoon_point_is_land"].to_numpy(dtype=bool)
            rows.append(
                {
                    "frequency_band": int(band),
                    "frequency_mhz": float(freq),
                    "antenna": str(antenna),
                    "metric": metric,
                    "n_events": int(np.count_nonzero(good)),
                    "spearman_rho": rho,
                    "stratified_permutation_p": p,
                    "median_response_submoon_land_z": float(np.nanmedian(y_all[land])) if np.any(land) else np.nan,
                    "median_response_submoon_ocean_z": float(np.nanmedian(y_all[ocean])) if np.any(ocean) else np.nan,
                    "n_submoon_land": int(np.count_nonzero(land)),
                    "n_submoon_ocean": int(np.count_nonzero(ocean)),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["fdr_q_global"] = _bh_qvalues(out["stratified_permutation_p"])
        out["fdr_q_within_metric"] = out.groupby("metric", group_keys=False)[
            "stratified_permutation_p"
        ].apply(_bh_qvalues)
    return out


def composition_partial_tests(responses: pd.DataFrame, trials: int, seed: int) -> pd.DataFrame:
    """Test disk composition after holding total visible nightside fraction fixed."""
    combined = (
        responses.groupby(["event_id", "event_type", "frequency_band", "frequency_mhz"], as_index=False)
        .agg(
            response_excess_z=("response_excess_z", "median"),
            month=("month", "first"),
            visible_night_fraction=("visible_night_fraction", "first"),
            **{metric: (metric, "first") for metric in COMPOSITION_METRICS},
        )
    )
    combined["antenna"] = "combined"
    columns = [
        "event_id",
        "event_type",
        "frequency_band",
        "frequency_mhz",
        "antenna",
        "response_excess_z",
        "month",
        "visible_night_fraction",
        *COMPOSITION_METRICS,
    ]
    analysis = pd.concat([responses[columns], combined], ignore_index=True, sort=False)
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
    for keys, group in analysis.groupby(["frequency_band", "frequency_mhz", "antenna"], sort=True):
        band, freq, antenna = keys
        for metric in COMPOSITION_METRICS:
            usable = group.dropna(subset=[metric, "visible_night_fraction", "response_excess_z"])
            if len(usable) < 30:
                continue
            x = _rank_residuals(
                usable[metric].to_numpy(dtype=float),
                usable["visible_night_fraction"].to_numpy(dtype=float),
            )
            y = _rank_residuals(
                usable["response_excess_z"].to_numpy(dtype=float),
                usable["visible_night_fraction"].to_numpy(dtype=float),
            )
            observed = float(np.corrcoef(x, y)[0, 1])
            strata = (
                usable["month"].astype(str) + "|" + usable["event_type"].astype(str)
            ).to_numpy()
            blocks = [np.flatnonzero(strata == value) for value in np.unique(strata)]
            exceed = 0
            for _ in range(int(trials)):
                permuted = y.copy()
                for idx in blocks:
                    if len(idx) > 1:
                        permuted[idx] = y[rng.permutation(idx)]
                value = float(np.corrcoef(x, permuted)[0, 1])
                exceed += int(np.isfinite(value) and abs(value) >= abs(observed))
            p_value = (
                (1.0 + exceed) / (1.0 + int(trials))
                if np.isfinite(observed) and trials > 0
                else np.nan
            )
            rows.append(
                {
                    "frequency_band": int(band),
                    "frequency_mhz": float(freq),
                    "antenna": str(antenna),
                    "metric": metric,
                    "control": "visible_night_fraction",
                    "n_events": int(len(usable)),
                    "partial_spearman_rho": observed,
                    "stratified_residual_permutation_p": p_value,
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["fdr_q_global"] = _bh_qvalues(out["stratified_residual_permutation_p"])
        out["fdr_q_within_metric"] = out.groupby("metric", group_keys=False)[
            "stratified_residual_permutation_p"
        ].apply(_bh_qvalues)
    return out


def quartile_effects(responses: pd.DataFrame) -> pd.DataFrame:
    combined = (
        responses.groupby(["event_id", "event_type", "frequency_band", "frequency_mhz"], as_index=False)
        .agg(
            response_excess_z=("response_excess_z", "median"),
            night_land_share_of_night_disk=("night_land_share_of_night_disk", "first"),
            night_auroral_share_of_night_disk=("night_auroral_share_of_night_disk", "first"),
        )
    )
    combined["antenna"] = "combined"
    base = responses[
        [
            "event_id",
            "event_type",
            "frequency_band",
            "frequency_mhz",
            "antenna",
            "response_excess_z",
            "night_land_share_of_night_disk",
            "night_auroral_share_of_night_disk",
        ]
    ]
    analysis = pd.concat([base, combined], ignore_index=True, sort=False)
    rows: list[dict[str, object]] = []
    for keys, group in analysis.groupby(["frequency_band", "frequency_mhz", "antenna"], sort=True):
        for metric in ["night_land_share_of_night_disk", "night_auroral_share_of_night_disk"]:
            g = group.dropna(subset=[metric, "response_excess_z"])
            if len(g) < 40:
                continue
            q25, q75 = g[metric].quantile([0.25, 0.75])
            low = g[g[metric] <= q25]["response_excess_z"].to_numpy(dtype=float)
            high = g[g[metric] >= q75]["response_excess_z"].to_numpy(dtype=float)
            rows.append(
                {
                    **dict(zip(["frequency_band", "frequency_mhz", "antenna"], keys)),
                    "metric": metric,
                    "n_available": int(len(g)),
                    "low_quartile_max": float(q25),
                    "high_quartile_min": float(q75),
                    "n_low_quartile": int(len(low)),
                    "n_high_quartile": int(len(high)),
                    "median_response_low_quartile_z": float(np.nanmedian(low)),
                    "median_response_high_quartile_z": float(np.nanmedian(high)),
                    "high_minus_low_median_response_z": float(np.nanmedian(high) - np.nanmedian(low)),
                }
            )
    return pd.DataFrame(rows)


def band_correlation_matrix(responses: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_event = (
        responses.groupby(["event_id", "frequency_mhz"], as_index=False)
        .agg(response_excess_z=("response_excess_z", "median"))
    )
    wide = per_event.pivot(index="event_id", columns="frequency_mhz", values="response_excess_z")
    freqs = list(wide.columns)
    rows: list[dict[str, object]] = []
    matrix = pd.DataFrame(np.nan, index=freqs, columns=freqs)
    for f1 in freqs:
        for f2 in freqs:
            pair = wide[[f1, f2]].dropna()
            rho = _rank_correlation(pair[f1].to_numpy(), pair[f2].to_numpy()) if len(pair) >= 20 else np.nan
            matrix.loc[f1, f2] = rho
            if f2 > f1:
                rows.append({"frequency_mhz_1": f1, "frequency_mhz_2": f2, "n_events": len(pair), "spearman_rho": rho})
    return matrix, pd.DataFrame(rows)


def plot_spectrum(summary: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for antenna, group in summary.groupby("antenna", sort=True):
        g = group.sort_values("frequency_mhz")
        label = "upper V" if antenna == "rv1_coarse" else "lower V"
        ax.errorbar(
            g["frequency_mhz"],
            g["median_shift_corrected_source_like_z"],
            yerr=g["robust_se_shift_corrected_z"],
            marker="o",
            capsize=3,
            label=label,
        )
    ax.axhline(0.0, color="black", lw=0.8)
    ax.axvspan(0.05, 0.75, color="#59a14f", alpha=0.10, label="typical AKR range")
    ax.axvspan(4.0, 10.0, color="#f28e2b", alpha=0.09, label="historical ground-breakthrough turn-up")
    ax.set_xscale("log")
    ax.set_xticks(sorted(summary["frequency_mhz"].unique()))
    ax.set_xticklabels([f"{v:g}" for v in sorted(summary["frequency_mhz"].unique())])
    ax.set_xlabel("RV channel frequency (MHz)")
    ax.set_ylabel("median true-minus-shifted Earth contrast (local σ)")
    ax.set_title("RAE-2 Earth-occultation spectrum")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    path = out_dir / "earth_occultation_spectrum.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_associations(tests: pd.DataFrame, out_dir: Path) -> Path:
    selected = tests[tests["metric"].isin(["visible_night_fraction", "night_land_share_of_night_disk"])]
    styles = {
        "combined": ("combined antennas", "#2ca02c", 0.0),
        "rv1_coarse": ("upper V", "#1f77b4", -0.012),
        "rv2_coarse": ("lower V", "#ff7f0e", 0.012),
    }
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    for ax, metric in zip(axes, ["visible_night_fraction", "night_land_share_of_night_disk"]):
        group = selected[selected["metric"].eq(metric)].copy()
        for antenna, sub in group.groupby("antenna", sort=True):
            label, color, offset = styles[str(antenna)]
            x = np.log10(sub["frequency_mhz"].to_numpy()) + offset
            ax.scatter(x, sub["spearman_rho"], s=55, color=color, label=label)
            significant = sub["fdr_q_within_metric"] < 0.05
            ax.scatter(
                x[significant],
                sub.loc[significant, "spearman_rho"],
                s=120,
                facecolors="none",
                edgecolors="black",
            )
        ax.axhline(0.0, color="black", lw=0.8)
        ax.set_ylabel("Spearman ρ")
        ax.set_title(metric.replace("_", " "))
        ax.grid(alpha=0.22)
    freqs = sorted(tests["frequency_mhz"].unique())
    axes[-1].set_xticks(np.log10(freqs))
    axes[-1].set_xticklabels([f"{v:g}" for v in freqs])
    axes[-1].set_xlabel("RV channel frequency (MHz)")
    axes[0].legend(frameon=False, ncol=2)
    fig.suptitle("Geographic association tests; rings mark within-family FDR q<0.05")
    fig.tight_layout()
    path = out_dir / "earth_geographic_association_spectrum.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_partial_composition(partial_tests: pd.DataFrame, out_dir: Path) -> Path:
    combined = partial_tests[partial_tests["antenna"].eq("combined")].copy()
    styles = {
        "night_land_share_of_night_disk": ("nighttime land share", "#f28e2b", "o"),
        "night_auroral_share_of_night_disk": ("nighttime auroral-zone share", "#59a14f", "s"),
    }
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for metric, group in combined.groupby("metric", sort=True):
        label, color, marker = styles[str(metric)]
        group = group.sort_values("frequency_mhz")
        ax.plot(
            group["frequency_mhz"],
            group["partial_spearman_rho"],
            color=color,
            marker=marker,
            label=label,
        )
        significant = group["fdr_q_global"] < 0.05
        ax.scatter(
            group.loc[significant, "frequency_mhz"],
            group.loc[significant, "partial_spearman_rho"],
            s=130,
            facecolors="none",
            edgecolors="black",
            zorder=4,
        )
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xscale("log")
    freqs = sorted(combined["frequency_mhz"].unique())
    ax.set_xticks(freqs)
    ax.set_xticklabels([f"{value:g}" for value in freqs])
    ax.set_xlabel("RV channel frequency (MHz)")
    ax.set_ylabel("partial Spearman rho")
    ax.set_title("Disk composition after controlling total nightside visibility")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = out_dir / "earth_composition_partial_associations.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_submoon_coverage(geometry: pd.DataFrame, geojson: dict, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for feature in geojson.get("features", []):
        for polygon in _iter_polygons((feature.get("geometry") or {})):
            if polygon:
                ring = np.asarray(polygon[0], dtype=float)
                ax.plot(ring[:, 0], ring[:, 1], color="0.55", lw=0.45)
    ax.scatter(
        geometry["submoon_lon_deg"],
        geometry["submoon_lat_deg"],
        c=geometry["visible_night_land_fraction"],
        cmap="viridis",
        s=12,
        alpha=0.7,
    )
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("sub-lunar longitude on Earth (deg)")
    ax.set_ylabel("sub-lunar latitude on Earth (deg)")
    ax.set_title("Earth-facing disk-centre coverage; color is projected nighttime-land fraction")
    ax.grid(alpha=0.18)
    colorbar = fig.colorbar(ax.collections[-1], ax=ax, pad=0.015)
    colorbar.set_label("projected nighttime-land fraction")
    fig.tight_layout()
    path = out_dir / "submoon_geometry_coverage.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _markdown(frame: pd.DataFrame, columns: list[str], max_rows: int = 50) -> str:
    if frame.empty:
        return "No rows."
    work = frame.loc[:, columns].head(max_rows).copy()
    for column in work.columns:
        if pd.api.types.is_numeric_dtype(work[column]):
            work[column] = work[column].map(lambda value: "" if pd.isna(value) else f"{value:.4g}")
    widths = {column: max(len(column), *(len(str(value)) for value in work[column])) for column in work.columns}
    lines = [
        "| " + " | ".join(column.ljust(widths[column]) for column in work.columns) + " |",
        "| " + " | ".join("-" * widths[column] for column in work.columns) + " |",
    ]
    for _, row in work.iterrows():
        lines.append("| " + " | ".join(str(row[column]).ljust(widths[column]) for column in work.columns) + " |")
    return "\n".join(lines)


def write_report(
    out_dir: Path,
    geometry: pd.DataFrame,
    responses: pd.DataFrame,
    spectrum: pd.DataFrame,
    tests: pd.DataFrame,
    partial_tests: pd.DataFrame,
    quartiles: pd.DataFrame,
    correlations: pd.DataFrame,
    plots: list[Path],
) -> Path:
    akr = tests[
        tests["frequency_band"].eq(1)
        & tests["metric"].isin(
            [
                "visible_night_auroral_fraction",
                "night_auroral_share_of_night_disk",
                "visible_auroral_fraction",
                "abs_moon_geomagnetic_lat_deg",
            ]
        )
    ].sort_values(["metric", "antenna"])
    hf_land = tests[
        tests["frequency_band"].isin([6, 7, 8, 9])
        & tests["antenna"].eq("combined")
        & tests["metric"].isin(
            [
                "visible_night_fraction",
                "visible_night_land_fraction",
                "night_land_share_of_night_disk",
                "visible_land_fraction",
            ]
        )
    ].sort_values(["frequency_mhz", "metric", "antenna"])
    hf_land_partial = partial_tests[
        partial_tests["frequency_band"].isin([6, 7, 8, 9])
        & partial_tests["antenna"].eq("combined")
        & partial_tests["metric"].eq("night_land_share_of_night_disk")
    ].sort_values("frequency_mhz")
    akr_partial = partial_tests[
        partial_tests["frequency_band"].eq(1)
        & partial_tests["metric"].eq("night_auroral_share_of_night_disk")
    ].sort_values("antenna")
    akr_land_control = tests[
        tests["frequency_band"].eq(1)
        & tests["antenna"].eq("combined")
        & tests["metric"].eq("night_land_share_of_night_disk")
    ]
    land_point = tests[
        tests["metric"].eq("submoon_point_is_land") & tests["antenna"].eq("combined")
    ].sort_values(
        ["frequency_mhz", "antenna"]
    )
    quartile_focus = quartiles[
        (
            quartiles["frequency_band"].eq(1)
            & quartiles["metric"].eq("night_auroral_share_of_night_disk")
        )
        | (
            quartiles["frequency_band"].isin([6, 7, 8, 9])
            & quartiles["metric"].eq("night_land_share_of_night_disk")
        )
    ].sort_values(["frequency_mhz", "antenna"])
    strongest = tests.assign(abs_rho=tests["spearman_rho"].abs()).sort_values("abs_rho", ascending=False)
    hf_significant = hf_land_partial[hf_land_partial["fdr_q_global"] < 0.05]
    hf_frequency_text = ", ".join(f"{value:g}" for value in hf_significant["frequency_mhz"])
    hf_rho_min = hf_significant["partial_spearman_rho"].min()
    hf_rho_max = hf_significant["partial_spearman_rho"].max()
    hf_q_max = hf_significant["fdr_q_global"].max()
    akr_combined = akr_partial[akr_partial["antenna"].eq("combined")].iloc[0]
    akr_land = akr_land_control.iloc[0]
    lines = [
        "# RAE-2 Earth-Noise Spectrum and Geographic Tests",
        "",
        "## Decision",
        "",
        "The Earth occultation spectrum is consistent with two historically known components: 0.45 MHz magnetospheric AKR/TKR and a separate 3.93-9.18 MHz terrestrial ground-breakthrough/HF component.",
        "The sparse nine-channel RV data do not identify individual services or transmitters, and the antennas do not image the Earth.",
        "",
        f"At population level, the RV data do differentiate the components across {geometry['event_id'].nunique()} physical occultations. After controlling for how much nightside is visible, the combined-antenna HF response still increases with nighttime land share at {hf_frequency_text} MHz (partial Spearman rho {hf_rho_min:.2f}-{hf_rho_max:.2f}, global FDR q<={hf_q_max:.3g}).",
        f"At 0.45 MHz the response instead tracks the nighttime auroral-zone share (combined partial rho {akr_combined['partial_spearman_rho']:.2f}, p={akr_combined['stratified_residual_permutation_p']:.3g}, global q={akr_combined['fdr_q_global']:.3g}; replicated in both antennas) and has no corresponding nighttime-land association (rho {akr_land['spearman_rho']:.2f}, p={akr_land['stratified_permutation_p']:.3g}).",
        "This is a statistically supported population-level separation, not an event-by-event source map.",
        "",
        "## Occultation Spectrum",
        "",
        _markdown(
            spectrum.sort_values(["frequency_mhz", "antenna"]),
            [
                "antenna",
                "frequency_mhz",
                "n_events",
                "median_shift_corrected_source_like_z",
                "robust_z_shift_corrected",
                "existing_survey_robust_stack_snr",
                "expected_emission_family",
            ],
        ),
        "",
        "The response is the per-event source-like occultation contrast minus the median of that same event/channel at ±600 and ±1200 s controls. Events with fewer than three samples on either side or an overlapping opposite limb are excluded.",
        "The existing survey statistic is included as an independent extraction; its very large formal stack S/N should not be read as spectral resolution.",
        "",
        "Band interpretation:",
        "",
    ]
    for band, (family, context) in CHANNEL_INTERPRETATION.items():
        freq = float(spectrum[spectrum["frequency_band"].eq(band)]["frequency_mhz"].iloc[0])
        lines.append(f"- {freq:g} MHz: **{family}** — {context}.")
    lines.extend(
        [
            "",
            "## Land/Ocean Tests",
            "",
            "`submoon_point_is_land` asks whether the point at the center of the Earth disk was land. This is easy to visualize but is not a source-localization observable.",
            "`visible_night_land_fraction` integrates projected nighttime land over the whole Earth-facing disk, while `night_land_share_of_night_disk` asks whether land composition matters after separating how much nightside is visible.",
            "",
            _markdown(
                hf_land,
                [
                    "antenna",
                    "frequency_mhz",
                    "metric",
                    "n_events",
                    "spearman_rho",
                    "stratified_permutation_p",
                    "fdr_q_within_metric",
                    "fdr_q_global",
                ],
            ),
            "",
            "Composition-only test after partialling out the total visible nightside fraction:",
            "",
            _markdown(
                hf_land_partial,
                [
                    "antenna",
                    "frequency_mhz",
                    "metric",
                    "control",
                    "n_events",
                    "partial_spearman_rho",
                    "stratified_residual_permutation_p",
                    "fdr_q_global",
                ],
            ),
            "",
            "The partial test retains the land-composition association at 3.93, 6.55, and 9.18 MHz. The 4.70 MHz estimate remains positive but does not survive correction once total nightside visibility is controlled, and that channel is also strongly antenna-dependent.",
            "At the retained bands both antennas have positive effect-size estimates, but global corrected significance is carried mainly by upper V; the lower-V estimates are weaker. This limits the claim to a population trend.",
            "",
            "Disk-centre land/ocean split:",
            "",
            _markdown(
                land_point,
                [
                    "antenna",
                    "frequency_mhz",
                    "n_submoon_land",
                    "n_submoon_ocean",
                    "median_response_submoon_land_z",
                    "median_response_submoon_ocean_z",
                    "spearman_rho",
                    "stratified_permutation_p",
                    "fdr_q_global",
                ],
            ),
            "",
            "Low-versus-high geometry quartiles (effect size, not an additional significance test):",
            "",
            _markdown(
                quartile_focus,
                [
                    "antenna",
                    "frequency_mhz",
                    "metric",
                    "n_available",
                    "median_response_low_quartile_z",
                    "median_response_high_quartile_z",
                    "high_minus_low_median_response_z",
                ],
            ),
            "",
            "## AKR / High-Latitude Tests",
            "",
            "The 1975 centered-dipole pole (78.76° N, 70.47° W) defines a 55°-80° geomagnetic auroral-zone proxy.",
            "`visible_night_auroral_fraction` is its projected nightside fraction on the Earth-facing disk; `night_auroral_share_of_night_disk` separates composition from total nightside visibility, and `abs_moon_geomagnetic_lat_deg` tests viewing latitude.",
            "",
            _markdown(
                akr,
                [
                    "antenna",
                    "metric",
                    "n_events",
                    "spearman_rho",
                    "stratified_permutation_p",
                    "fdr_q_within_metric",
                    "fdr_q_global",
                ],
            ),
            "",
            "Auroral-composition test after partialling out the total visible nightside fraction:",
            "",
            _markdown(
                akr_partial,
                [
                    "antenna",
                    "metric",
                    "control",
                    "n_events",
                    "partial_spearman_rho",
                    "stratified_residual_permutation_p",
                    "fdr_q_global",
                ],
            ),
            "",
            "The positive, two-antenna 0.45 MHz association is consistent with a nightside high-latitude AKR origin; it is not proof of a specific emitting oval. AKR is bursty, beamed, and emitted thousands of kilometres above the oval, while this proxy uses a surface-centered dipole and no geomagnetic-activity index.",
            "The RV cadence and total-power response do not provide two-dimensional direction finding, so northern versus southern auroral source localization is not statistically identifiable without an external activity/source prior.",
            "",
            "## Strongest Geographic Correlations (diagnostic, not preselected)",
            "",
            _markdown(
                strongest,
                [
                    "antenna",
                    "frequency_mhz",
                    "metric",
                    "spearman_rho",
                    "stratified_permutation_p",
                    "fdr_q_global",
                ],
                max_rows=20,
            ),
            "",
            "Permutations are restricted within calendar month and ingress/egress type. Global and within-metric Benjamini-Hochberg corrections account for the tested channels, antennas, and predictors.",
            "",
            "## Cross-Frequency Event Correlations",
            "",
            _markdown(
                correlations.assign(abs_rho=correlations["spearman_rho"].abs()).sort_values("abs_rho", ascending=False),
                ["frequency_mhz_1", "frequency_mhz_2", "n_events", "spearman_rho"],
                max_rows=20,
            ),
            "",
            "Correlated high-frequency channels can share ionospheric transmission and broad terrestrial interference even when their 40 kHz passbands do not contain the same service. Correlation with 0.45 MHz would instead suggest common activity/background selection, not proof of a shared emission mechanism.",
            "",
            "## Practical Conclusion",
            "",
            "- Treat 0.45 MHz as the only RV channel that directly samples the established AKR/TKR band; its independent high-latitude/nightside association strengthens that identification.",
            "- The data distinguish land-rich from ocean-rich viewing statistically at 3.93, 6.55, and 9.18 MHz, after controlling total nightside visibility. Treat this as disk-integrated ground breakthrough, not as a resolved map of transmitters.",
            "- The 4.70 MHz evidence is weaker and antenna-dependent; 0.70-2.20 MHz do not yield a reliable positive Earth component in this strict extraction.",
            "- Do not label individual events land, ocean, north-auroral, or south-auroral sources from RV alone.",
            "- A stronger geographic attribution would require external 1974-1975 geomagnetic indices and AKR observations, historical transmitter schedules, and a forward model of ionospheric escape and the RV antenna beam.",
            "",
            "## Files",
            "",
            "- `earth_event_geometry.csv`",
            "- `earth_event_responses_with_geometry.csv`",
            "- `earth_noise_spectral_summary.csv`",
            "- `earth_noise_geographic_association_tests.csv`",
            "- `earth_noise_composition_partial_tests.csv`",
            "- `earth_noise_geometry_quartile_effects.csv`",
            "- `earth_noise_cross_frequency_correlations.csv`",
            "- `earth_noise_cross_frequency_matrix.csv`",
        ]
    )
    lines.extend([f"- `{path.name}`" for path in plots])
    lines.extend(
        [
            "",
            "## Primary References",
            "",
            "- NASA, *Characteristics of magnetospheric radio noise spectra*: https://ntrs.nasa.gov/citations/19760016374",
            "- NASA/JPL, *Terrestrial Radio Noise at Low Frequencies*: https://ntrs.nasa.gov/api/citations/19900009307/downloads/19900009307.pdf",
            "- Grabbe, *Auroral kilometric radiation: A theoretical review*: https://ntrs.nasa.gov/citations/19820030700",
            "- ITU Geneva 1975 LF/MF plan: https://www.itu.int/en/ITU-R/terrestrial/broadcast/plans/Pages/plans.aspx",
            "- ITU 1974 maritime conference documents: https://search.itu.int/history/HistoryDigitalCollectionDocLibrary/4.96.51.en.103.pdf",
            "- Natural Earth 1:110m countries: https://www.naturalearthdata.com/",
        ]
    )
    path = out_dir / "earth_noise_geography_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", default=str(DEFAULT_EVENTS))
    parser.add_argument("--contrasts", default=str(DEFAULT_CONTRASTS))
    parser.add_argument("--channel-scan", default=str(DEFAULT_CHANNEL_SCAN))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--land-grid-step-deg", type=float, default=2.0)
    parser.add_argument("--permutation-trials", type=int, default=999)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    write_json(
        out_dir / "run_config.json",
        {
            **vars(args),
            "natural_earth_url": NATURAL_EARTH_URL,
            "geomagnetic_north_pole_1975_lat_deg": GEOMAGNETIC_NORTH_POLE_1975_LAT_DEG,
            "geomagnetic_north_pole_1975_lon_deg": GEOMAGNETIC_NORTH_POLE_1975_LON_DEG,
            "software_versions": software_versions(),
        },
    )
    geojson = load_land_geojson(out_dir / "natural_earth_110m_admin0_countries.geojson")
    grid = build_surface_grid(geojson, float(args.land_grid_step_deg))
    events = read_table(args.events, parse_dates=["predicted_event_time"], low_memory=False)
    contrasts = read_table(args.contrasts, low_memory=False)
    channel_scan = read_table(args.channel_scan, low_memory=False)

    geometry = event_geometry(events, grid, geojson)
    geometry.to_csv(out_dir / "earth_event_geometry.csv", index=False)
    responses = prepare_event_responses(contrasts, geometry)
    responses.to_csv(out_dir / "earth_event_responses_with_geometry.csv", index=False)
    spectrum = spectral_summary(responses, channel_scan)
    spectrum.to_csv(out_dir / "earth_noise_spectral_summary.csv", index=False)
    tests = association_tests(responses, int(args.permutation_trials), int(args.seed))
    tests.to_csv(out_dir / "earth_noise_geographic_association_tests.csv", index=False)
    partial_tests = composition_partial_tests(responses, int(args.permutation_trials), int(args.seed))
    partial_tests.to_csv(out_dir / "earth_noise_composition_partial_tests.csv", index=False)
    quartiles = quartile_effects(responses)
    quartiles.to_csv(out_dir / "earth_noise_geometry_quartile_effects.csv", index=False)
    matrix, correlations = band_correlation_matrix(responses)
    matrix.to_csv(out_dir / "earth_noise_cross_frequency_matrix.csv")
    correlations.to_csv(out_dir / "earth_noise_cross_frequency_correlations.csv", index=False)

    plots = [
        plot_spectrum(spectrum, out_dir),
        plot_associations(tests, out_dir),
        plot_partial_composition(partial_tests, out_dir),
        plot_submoon_coverage(geometry, geojson, out_dir),
    ]
    report = write_report(
        out_dir,
        geometry,
        responses,
        spectrum,
        tests,
        partial_tests,
        quartiles,
        correlations,
        plots,
    )
    print(report)


if __name__ == "__main__":
    main()
