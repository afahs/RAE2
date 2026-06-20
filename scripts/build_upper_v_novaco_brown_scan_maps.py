#!/usr/bin/env python
"""Recreate Novaco & Brown-style upper-V Galactic scan maps.

Novaco & Brown's published RAE-2 contour maps were made from the outward
Ryle-Vonberg antenna as a scanning experiment, not from lunar-occultation
inversion rows.  This script follows that reduction style with the locally
available Ryle-Vonberg table:

1. select upper-V samples with either the historical new-Moon quiet-window
   proxy or a direct Earth-in-beam geometry cut;
2. normalize four-day groups to a local group average;
3. reject high-amplitude samples and ten-minute jumps;
4. bin accepted pointing samples into 5-degree Galactic cells;
5. iteratively 4-sigma clip each cell and render contour maps.

The current default input table contains `rv1_coarse`, not the RV fine output
named in the paper, so outputs are labeled with the actual antenna column.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd
from astropy.coordinates import FK4, Galactic, SkyCoord
from astropy.time import Time
import astropy.units as u
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.constants import (  # noqa: E402
    EARTH_MOON_DISTANCE_KM,
    EARTH_UNIT_COLUMNS,
    FREQUENCY_MAP_MHZ,
    MOON_RADIUS_KM,
    SPACECRAFT_COLUMNS,
)
from rylevonberg.frames import body_unit_vectors_from_moon  # noqa: E402
from rylevonberg.geometry import normalize_vectors  # noqa: E402
from rylevonberg.table_io import read_table  # noqa: E402
from rylevonberg.util import datetime_ns, ensure_dir, robust_sigma, software_versions, write_json  # noqa: E402


DEFAULT_CLEAN = ROOT / "outputs/control_survey_earth_sun_postnov1974_v2/01_ingest/cleaned_timeseries.csv"
DEFAULT_OUT = ROOT / "outputs/upper_v_novaco_brown_scan_maps_rv1_coarse_5deg_v1"
NOVACO_BROWN_BANDS = [4, 5, 6, 7, 8, 9]
SYNODIC_MONTH_DAYS = 29.530588853
REFERENCE_NEW_MOON = pd.Timestamp("1970-01-07 20:35:00")
UNCOVERED_CMAP = ListedColormap([(0.70, 0.70, 0.70, 0.62)])
EARTH_BEAM_GEOMETRY_COLUMNS = [*SPACECRAFT_COLUMNS, *EARTH_UNIT_COLUMNS]
BEAM_DIR = Path(os.environ.get("RAE2_ANTENNA_DIGITIZATION_DIR", "data/antennaDigitization"))
BEAM_SPECS = [
    (1.31, BEAM_DIR / "eplane_1p31MHz.csv", BEAM_DIR / "hplane_1p31MHz.csv"),
    (3.93, BEAM_DIR / "eplane_3p93MHz.csv", BEAM_DIR / "hplane_3p93MHz.csv"),
    (6.55, BEAM_DIR / "eplane_6p55MHz.csv", BEAM_DIR / "hplane_6p55MHz.csv"),
]


def effective_start_time(start_time: str | None, all_dates: bool = False) -> str | None:
    if all_dates:
        return None
    if start_time is None:
        return None
    text = str(start_time).strip()
    return text or None


def parse_frequencies(value: str) -> list[int]:
    text = str(value).strip().lower()
    if text in {"novaco_brown", "nb", "paper"}:
        return list(NOVACO_BROWN_BANDS)
    if text in {"all", "*"}:
        return sorted(FREQUENCY_MAP_MHZ)
    reverse = {float(v): int(k) for k, v in FREQUENCY_MAP_MHZ.items()}
    out = []
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


def detect_ra_units(ra_values: pd.Series | np.ndarray, mode: str = "auto") -> str:
    mode = str(mode).lower()
    if mode in {"hour", "hours", "h"}:
        return "hours"
    if mode in {"degree", "degrees", "deg"}:
        return "degrees"
    if mode != "auto":
        raise ValueError(f"unsupported RA unit mode: {mode}")
    values = pd.to_numeric(pd.Series(ra_values), errors="coerce").to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return "degrees"
    abs_finite = np.abs(finite)
    if float(np.nanmax(abs_finite)) <= 24.0001:
        return "hours"
    return "hours" if float(np.nanmean(abs_finite <= 24.0001)) >= 0.99 else "degrees"


def _filter_valid_pointing(df: pd.DataFrame, ra_units: str) -> tuple[pd.DataFrame, str, int]:
    units = detect_ra_units(df["right_ascension"], ra_units)
    ra = pd.to_numeric(df["right_ascension"], errors="coerce")
    dec = pd.to_numeric(df["declination"], errors="coerce")
    mask = np.isfinite(ra) & np.isfinite(dec) & dec.between(-90.0, 90.0, inclusive="both")
    if units == "hours":
        mask &= ra.between(0.0, 24.0001, inclusive="both")
    else:
        mask &= ra.between(0.0, 360.0001, inclusive="both")
    out = df.loc[mask].copy()
    return out, units, int(len(df) - len(out))


def new_moon_distance_days(times: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    ns = datetime_ns(times).astype(float)
    elapsed_days = (ns - float(REFERENCE_NEW_MOON.value)) / 86400.0e9
    phase = np.mod(elapsed_days, SYNODIC_MONTH_DAYS)
    return np.minimum(phase, SYNODIC_MONTH_DAYS - phase)


def radec_unit_vectors(
    ra_values: pd.Series | np.ndarray,
    dec_values: pd.Series | np.ndarray,
    ra_units: str,
) -> tuple[np.ndarray, str]:
    units = detect_ra_units(ra_values, ra_units)
    ra_deg = pd.to_numeric(pd.Series(ra_values), errors="coerce").to_numpy(dtype=float)
    if units == "hours":
        ra_deg = ra_deg * 15.0
    dec_deg = pd.to_numeric(pd.Series(dec_values), errors="coerce").to_numpy(dtype=float)
    ra_rad = np.deg2rad(ra_deg)
    dec_rad = np.deg2rad(dec_deg)
    vectors = np.column_stack(
        [
            np.cos(dec_rad) * np.cos(ra_rad),
            np.cos(dec_rad) * np.sin(ra_rad),
            np.sin(dec_rad),
        ]
    )
    return normalize_vectors(vectors), units


def compute_beam_separation_deg(beam_vectors: np.ndarray, source_vectors: np.ndarray) -> np.ndarray:
    beam = normalize_vectors(np.asarray(beam_vectors, dtype=float))
    source = normalize_vectors(np.asarray(source_vectors, dtype=float))
    dots = np.einsum("ij,ij->i", beam, source)
    return np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))


def nearest_beam_spec(frequency_mhz: float) -> tuple[float, Path, Path]:
    return min(BEAM_SPECS, key=lambda spec: abs(spec[0] - float(frequency_mhz)))


def relative_gain_db_from_linear(gain: np.ndarray) -> np.ndarray:
    values = np.asarray(gain, dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)
    good = np.isfinite(values) & (values > 0.0)
    if not np.any(good):
        return out
    peak = float(np.nanmax(values[good]))
    if not np.isfinite(peak) or peak <= 0.0:
        return out
    out[good] = 10.0 * np.log10(np.clip(values[good] / peak, np.finfo(float).tiny, None))
    return out


def interpolate_cyclic(angles: np.ndarray, values: np.ndarray, angle_deg: np.ndarray) -> np.ndarray:
    x = np.asarray(angles, dtype=float) % 360.0
    y = np.asarray(values, dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    if x.size == 0:
        return np.full_like(np.asarray(angle_deg, dtype=float), np.nan, dtype=float)
    query = np.asarray(angle_deg, dtype=float) % 360.0
    x_pad = np.concatenate(([x[-1] - 360.0], x, [x[0] + 360.0]))
    y_pad = np.concatenate(([y[-1]], y, [y[0]]))
    return np.interp(query, x_pad, y_pad)


def load_relative_beam_model(eplane: Path, hplane: Path) -> tuple[np.ndarray, np.ndarray]:
    e = read_table(eplane)
    h = read_table(hplane)
    e = e[["angle_deg", "gain_dB"]].rename(columns={"gain_dB": "e_gain_dB"})
    h = h[["angle_deg", "gain_dB"]].rename(columns={"gain_dB": "h_gain_dB"})
    beam = e.merge(h, on="angle_deg", how="inner").sort_values("angle_deg")
    angles = beam["angle_deg"].to_numpy(dtype=float)
    e_gain = 10.0 ** (beam["e_gain_dB"].to_numpy(dtype=float) / 10.0)
    h_gain = 10.0 ** (beam["h_gain_dB"].to_numpy(dtype=float) / 10.0)
    mean_power_gain = 0.5 * (e_gain + h_gain)
    return angles, relative_gain_db_from_linear(mean_power_gain)


def sun_unit_vectors_for_times(
    times: pd.Series | pd.DatetimeIndex,
    cadence_hours: float,
    target_frame: str = "fk4",
) -> np.ndarray:
    time_index = pd.DatetimeIndex(times)
    if len(time_index) == 0:
        return np.empty((0, 3), dtype=float)
    ns = datetime_ns(time_index).astype(np.int64)
    cadence_ns = max(int(float(cadence_hours) * 3600.0e9), int(60.0e9))
    lo = int((int(ns.min()) // cadence_ns) * cadence_ns)
    hi = int(((int(ns.max()) + cadence_ns - 1) // cadence_ns) * cadence_ns)
    grid_ns = np.arange(lo, hi + cadence_ns, cadence_ns, dtype=np.int64)
    if grid_ns.size < 2:
        grid_ns = np.array([lo, lo + cadence_ns], dtype=np.int64)
    grid_times = pd.to_datetime(grid_ns)
    grid_vec = body_unit_vectors_from_moon("sun", grid_times, target_frame=target_frame, equinox="B1950")
    x = ns.astype(float)
    gx = grid_ns.astype(float)
    vec = np.column_stack([np.interp(x, gx, grid_vec[:, dim]) for dim in range(3)])
    return normalize_vectors(vec)


def _earth_direction_vectors(df: pd.DataFrame) -> np.ndarray:
    missing = [col for col in EARTH_BEAM_GEOMETRY_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "Earth-beam selection requires geometry columns missing from input: "
            + ", ".join(missing)
        )
    spacecraft = df[SPACECRAFT_COLUMNS].to_numpy(dtype=float)
    moon_to_earth = normalize_vectors(df[EARTH_UNIT_COLUMNS].to_numpy(dtype=float)) * float(EARTH_MOON_DISTANCE_KM)
    return normalize_vectors(moon_to_earth - spacecraft)


def _beam_axis_vectors(df: pd.DataFrame, axis_mode: str, ra_units: str) -> tuple[np.ndarray, str | None]:
    mode = str(axis_mode).strip().lower().replace("_", "-")
    if mode == "radec":
        return radec_unit_vectors(df["right_ascension"], df["declination"], ra_units)
    if mode in {"radial-upper", "upper-radial"}:
        missing = [col for col in SPACECRAFT_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(
                "Radial upper-V Earth-beam selection requires spacecraft position columns missing from input: "
                + ", ".join(missing)
            )
        return normalize_vectors(df[SPACECRAFT_COLUMNS].to_numpy(dtype=float)), None
    raise ValueError(f"unsupported Earth-beam axis mode: {axis_mode}")


def _attach_earth_beam_separation(df: pd.DataFrame, axis_mode: str, ra_units: str) -> tuple[pd.DataFrame, str | None]:
    out = df.copy()
    earth = _earth_direction_vectors(out)
    beam, interpreted_ra_units = _beam_axis_vectors(out, axis_mode, ra_units)
    out["earth_beam_separation_deg"] = compute_beam_separation_deg(beam, earth)
    return out, interpreted_ra_units


def _earth_visible_by_moon_center(df: pd.DataFrame, earth_vectors: np.ndarray) -> np.ndarray:
    spacecraft = df[SPACECRAFT_COLUMNS].to_numpy(dtype=float)
    moon_center = normalize_vectors(-spacecraft)
    moon_range = np.linalg.norm(spacecraft, axis=1)
    moon_radius_deg = np.degrees(np.arcsin(np.clip(float(MOON_RADIUS_KM) / np.maximum(moon_range, 1e-12), 0.0, 1.0)))
    return compute_beam_separation_deg(moon_center, earth_vectors) > moon_radius_deg


def _attach_earth_beam_gain(df: pd.DataFrame, axis_mode: str, ra_units: str) -> tuple[pd.DataFrame, str | None]:
    out = df.copy()
    earth = _earth_direction_vectors(out)
    beam, interpreted_ra_units = _beam_axis_vectors(out, axis_mode, ra_units)
    out["earth_beam_separation_deg"] = compute_beam_separation_deg(beam, earth)
    out["earth_visible_by_moon_center"] = _earth_visible_by_moon_center(out, earth)
    out["earth_beam_model_frequency_mhz"] = np.nan
    out["earth_beam_model_eplane"] = ""
    out["earth_beam_model_hplane"] = ""
    out["earth_beam_relative_gain_db"] = np.nan
    beam_cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    freq_values = out["frequency_mhz"].to_numpy(dtype=float)
    for freq in sorted(out["frequency_mhz"].dropna().unique()):
        beam_freq, eplane, hplane = nearest_beam_spec(float(freq))
        if beam_freq not in beam_cache:
            beam_cache[beam_freq] = load_relative_beam_model(eplane, hplane)
        angles, gain_db = beam_cache[beam_freq]
        mask = np.isclose(freq_values, float(freq))
        sep = out.loc[mask, "earth_beam_separation_deg"].to_numpy(dtype=float)
        out.loc[mask, "earth_beam_model_frequency_mhz"] = float(beam_freq)
        out.loc[mask, "earth_beam_model_eplane"] = str(eplane)
        out.loc[mask, "earth_beam_model_hplane"] = str(hplane)
        out.loc[mask, "earth_beam_relative_gain_db"] = interpolate_cyclic(angles, gain_db, sep)
    return out, interpreted_ra_units


def _attach_sun_beam_gain(
    df: pd.DataFrame,
    axis_mode: str,
    ra_units: str,
    sun_vector_cadence_hours: float,
) -> tuple[pd.DataFrame, str | None]:
    out = df.copy()
    beam, interpreted_ra_units = _beam_axis_vectors(out, axis_mode, ra_units)
    sun = sun_unit_vectors_for_times(out["time"], cadence_hours=float(sun_vector_cadence_hours), target_frame="fk4")
    out["sun_beam_separation_deg"] = compute_beam_separation_deg(beam, sun)
    out["sun_beam_model_frequency_mhz"] = np.nan
    out["sun_beam_model_eplane"] = ""
    out["sun_beam_model_hplane"] = ""
    out["sun_beam_relative_gain_db"] = np.nan
    beam_cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    freq_values = out["frequency_mhz"].to_numpy(dtype=float)
    for freq in sorted(out["frequency_mhz"].dropna().unique()):
        beam_freq, eplane, hplane = nearest_beam_spec(float(freq))
        if beam_freq not in beam_cache:
            beam_cache[beam_freq] = load_relative_beam_model(eplane, hplane)
        angles, gain_db = beam_cache[beam_freq]
        mask = np.isclose(freq_values, float(freq))
        sep = out.loc[mask, "sun_beam_separation_deg"].to_numpy(dtype=float)
        out.loc[mask, "sun_beam_model_frequency_mhz"] = float(beam_freq)
        out.loc[mask, "sun_beam_model_eplane"] = str(eplane)
        out.loc[mask, "sun_beam_model_hplane"] = str(hplane)
        out.loc[mask, "sun_beam_relative_gain_db"] = interpolate_cyclic(angles, gain_db, sep)
    return out, interpreted_ra_units


def _truthy(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    return values.astype(str).str.lower().isin(["true", "1", "yes"])


def _detect_input_format(path: Path, input_format: str) -> str:
    mode = str(input_format).strip().lower().replace("_", "-")
    if mode in {"cleaned", "raw-master"}:
        return mode
    if mode != "auto":
        raise ValueError(f"unsupported input format: {input_format}")
    header = read_table(path, nrows=0)
    cols = set(header.columns)
    if {"antenna", "power", "is_valid"}.issubset(cols):
        return "cleaned"
    return "raw-master"


def _read_selected_rows(
    input_path: Path,
    antenna: str,
    bands: list[int],
    start_time: str | None,
    end_time: str | None,
    max_rows: int,
    input_format: str = "cleaned",
) -> pd.DataFrame:
    resolved = _detect_input_format(input_path, input_format)
    if resolved == "raw-master":
        return _read_selected_raw_master_rows(input_path, antenna, bands, start_time, end_time, max_rows)
    return _read_selected_cleaned_rows(input_path, antenna, bands, start_time, end_time, max_rows)


def _read_selected_cleaned_rows(
    clean_path: Path,
    antenna: str,
    bands: list[int],
    start_time: str | None,
    end_time: str | None,
    max_rows: int,
) -> pd.DataFrame:
    header = read_table(clean_path, nrows=0)
    available_cols = set(header.columns)
    usecols = [
        "time",
        "frequency_band",
        "antenna",
        "power",
        "is_valid",
        "right_ascension",
        "declination",
    ]
    usecols.extend(col for col in EARTH_BEAM_GEOMETRY_COLUMNS if col in available_cols)
    frames = []
    remaining = int(max_rows)
    wanted_bands = set(int(band) for band in bands)
    start = pd.Timestamp(start_time) if start_time else None
    end = pd.Timestamp(end_time) if end_time else None
    for chunk in read_table(clean_path, usecols=usecols, chunksize=750_000, low_memory=False):
        mask = chunk["antenna"].astype(str).eq(str(antenna))
        mask &= chunk["frequency_band"].astype(int).isin(wanted_bands)
        if not mask.any():
            continue
        sub = chunk.loc[mask].copy()
        sub["time"] = pd.to_datetime(sub["time"], errors="coerce")
        sub = sub[sub["time"].notna()]
        if start is not None:
            sub = sub[sub["time"] >= start]
        if end is not None:
            sub = sub[sub["time"] <= end]
        if sub.empty:
            continue
        sub["frequency_band"] = sub["frequency_band"].astype(int)
        sub["frequency_mhz"] = sub["frequency_band"].map(FREQUENCY_MAP_MHZ).astype(float)
        sub["power"] = pd.to_numeric(sub["power"], errors="coerce")
        sub["right_ascension"] = pd.to_numeric(sub["right_ascension"], errors="coerce")
        sub["declination"] = pd.to_numeric(sub["declination"], errors="coerce")
        sub["is_valid"] = _truthy(sub["is_valid"])
        sub = sub[
            sub["is_valid"]
            & np.isfinite(sub["power"])
            & sub["power"].gt(0.0)
            & np.isfinite(sub["right_ascension"])
            & np.isfinite(sub["declination"])
        ].copy()
        if sub.empty:
            continue
        if max_rows > 0:
            take = min(remaining, len(sub))
            frames.append(sub.iloc[:take].copy())
            remaining -= take
            if remaining <= 0:
                break
        else:
            frames.append(sub)
    if not frames:
        return pd.DataFrame(columns=usecols)
    return pd.concat(frames, ignore_index=True).sort_values("time").reset_index(drop=True)


def _read_selected_raw_master_rows(
    raw_path: Path,
    antenna: str,
    bands: list[int],
    start_time: str | None,
    end_time: str | None,
    max_rows: int,
) -> pd.DataFrame:
    header = read_table(raw_path, nrows=0)
    cols = set(header.columns)
    if antenna not in cols:
        raise ValueError(f"raw master file is missing antenna/value column: {antenna}")
    frequency_col = "frequency_band" if "frequency_band" in cols else "frequencyNumber"
    usecols = ["time", frequency_col, "right_ascension", "declination", antenna]
    usecols.extend(col for col in EARTH_BEAM_GEOMETRY_COLUMNS if col in cols)
    frames = []
    remaining = int(max_rows)
    wanted_bands = set(int(band) for band in bands)
    start = pd.Timestamp(start_time) if start_time else None
    end = pd.Timestamp(end_time) if end_time else None
    for chunk in read_table(raw_path, usecols=usecols, chunksize=750_000, low_memory=False):
        mask = chunk[frequency_col].astype(int).isin(wanted_bands)
        if not mask.any():
            continue
        sub = chunk.loc[mask].copy()
        sub["time"] = pd.to_datetime(sub["time"], errors="coerce")
        sub = sub[sub["time"].notna()]
        if start is not None:
            sub = sub[sub["time"] >= start]
        if end is not None:
            sub = sub[sub["time"] <= end]
        if sub.empty:
            continue
        sub = sub.rename(columns={frequency_col: "frequency_band", antenna: "power"})
        sub["antenna"] = str(antenna)
        sub["is_valid"] = True
        sub["frequency_band"] = sub["frequency_band"].astype(int)
        sub["frequency_mhz"] = sub["frequency_band"].map(FREQUENCY_MAP_MHZ).astype(float)
        sub["power"] = pd.to_numeric(sub["power"], errors="coerce")
        sub["right_ascension"] = pd.to_numeric(sub["right_ascension"], errors="coerce")
        sub["declination"] = pd.to_numeric(sub["declination"], errors="coerce")
        output_cols = [
            "time",
            "frequency_band",
            "antenna",
            "power",
            "is_valid",
            "right_ascension",
            "declination",
            "frequency_mhz",
        ]
        output_cols.extend(col for col in EARTH_BEAM_GEOMETRY_COLUMNS if col in sub.columns)
        sub = sub[
            np.isfinite(sub["power"])
            & sub["power"].gt(0.0)
            & np.isfinite(sub["right_ascension"])
            & np.isfinite(sub["declination"])
        ][output_cols]
        if sub.empty:
            continue
        if max_rows > 0:
            take = min(remaining, len(sub))
            frames.append(sub.iloc[:take].copy())
            remaining -= take
            if remaining <= 0:
                break
        else:
            frames.append(sub)
    if not frames:
        return pd.DataFrame(
            columns=[
                "time",
                "frequency_band",
                "antenna",
                "power",
                "is_valid",
                "right_ascension",
                "declination",
                "frequency_mhz",
                *[col for col in EARTH_BEAM_GEOMETRY_COLUMNS if col in cols],
            ]
        )
    return pd.concat(frames, ignore_index=True).sort_values("time").reset_index(drop=True)


def _attach_galactic_pointing(df: pd.DataFrame, ra_units: str, grid_step_deg: float) -> pd.DataFrame:
    out = df.copy()
    units = detect_ra_units(out["right_ascension"], ra_units)
    ra_deg = out["right_ascension"].to_numpy(dtype=float) * (15.0 if units == "hours" else 1.0)
    dec_deg = out["declination"].to_numpy(dtype=float)
    coords = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame=FK4(equinox=Time("B1950")))
    gal = coords.transform_to(Galactic())
    lon = np.asarray(gal.l.deg, dtype=float) % 360.0
    lat = np.asarray(gal.b.deg, dtype=float)
    n_lon = int(round(360.0 / float(grid_step_deg)))
    n_lat = int(round(180.0 / float(grid_step_deg)))
    lon_idx = np.floor(lon / float(grid_step_deg)).astype(int)
    lat_idx = np.floor((lat + 90.0) / float(grid_step_deg)).astype(int)
    lon_idx = np.clip(lon_idx, 0, n_lon - 1)
    lat_idx = np.clip(lat_idx, 0, n_lat - 1)
    out["ra_units_interpreted"] = units
    out["ra_deg_interpreted"] = ra_deg
    out["galactic_l_sample_deg"] = lon
    out["galactic_b_sample_deg"] = lat
    out["lon_index"] = lon_idx
    out["lat_index"] = lat_idx
    out["pixel_index"] = lat_idx * n_lon + lon_idx
    out["galactic_l_deg"] = lon_idx.astype(float) * float(grid_step_deg) + 0.5 * float(grid_step_deg)
    out["galactic_b_deg"] = -90.0 + lat_idx.astype(float) * float(grid_step_deg) + 0.5 * float(grid_step_deg)
    return out


def _iterative_group_reference(power: np.ndarray, upper_db: float, iterations: int) -> tuple[float, int]:
    values = np.asarray(power, dtype=float)
    keep = np.isfinite(values) & (values > 0)
    if not keep.any():
        return np.nan, 0
    reference = float(np.nanmedian(values[keep]))
    for _ in range(max(int(iterations), 1)):
        if not np.isfinite(reference) or reference <= 0:
            return np.nan, 0
        limit = reference * 10.0 ** (float(upper_db) / 10.0)
        next_keep = np.isfinite(values) & (values > 0) & (values <= limit)
        if not next_keep.any():
            break
        next_ref = float(np.nanmedian(values[next_keep]))
        if np.isclose(next_ref, reference, rtol=1e-6, atol=0.0):
            keep = next_keep
            reference = next_ref
            break
        keep = next_keep
        reference = next_ref
    return float(reference), int(np.count_nonzero(keep))


def _normalize_four_day_groups(
    df: pd.DataFrame,
    start_time: str | None,
    group_days: float,
    amplitude_filter_db: float,
    normalization_iterations: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    origin = pd.Timestamp(start_time) if start_time else pd.Timestamp(out["time"].min())
    group_ns = int(float(group_days) * 86400.0e9)
    out["four_day_group"] = ((datetime_ns(out["time"]) - origin.value) // group_ns).astype(int)
    out["group_reference_power"] = np.nan
    out["group_reference_n"] = 0
    out["keep_amplitude"] = False
    rows = []
    for (band, group_id), idx in out.groupby(["frequency_band", "four_day_group"], sort=True).groups.items():
        values = out.loc[idx, "power"].to_numpy(dtype=float)
        ref, n_ref = _iterative_group_reference(values, amplitude_filter_db, normalization_iterations)
        out.loc[idx, "group_reference_power"] = ref
        out.loc[idx, "group_reference_n"] = n_ref
        if np.isfinite(ref) and ref > 0:
            db = 10.0 * np.log10(values / ref)
            keep = np.isfinite(db) & (db <= float(amplitude_filter_db))
            out.loc[idx, "keep_amplitude"] = keep
        else:
            db = np.full(len(idx), np.nan)
        rows.append(
            {
                "frequency_band": int(band),
                "frequency_mhz": float(FREQUENCY_MAP_MHZ[int(band)]),
                "four_day_group": int(group_id),
                "group_start_time": origin + pd.to_timedelta(int(group_id) * float(group_days), unit="D"),
                "n_samples": int(len(idx)),
                "group_reference_power": ref,
                "group_reference_n": int(n_ref),
                "n_keep_amplitude": int(np.count_nonzero(out.loc[idx, "keep_amplitude"].to_numpy(dtype=bool))),
            }
        )
    out["normalized_power_ratio"] = out["power"] / out["group_reference_power"]
    out["relative_power_db"] = 10.0 * np.log10(out["normalized_power_ratio"])
    return out, pd.DataFrame.from_records(rows)


def _normalize_destriped_groups(
    df: pd.DataFrame,
    start_time: str | None,
    group_days: float,
    amplitude_filter_db: float,
    normalization_iterations: int,
    destripe_iterations: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    origin = pd.Timestamp(start_time) if start_time else pd.Timestamp(out["time"].min())
    group_ns = int(float(group_days) * 86400.0e9)
    out["four_day_group"] = ((datetime_ns(out["time"]) - origin.value) // group_ns).astype(int)
    out["raw_power_db"] = 10.0 * np.log10(out["power"].to_numpy(dtype=float))
    out["initial_group_reference_power"] = np.nan
    out["initial_group_reference_db"] = np.nan
    out["group_reference_n"] = 0
    out["keep_amplitude"] = False

    rows = []
    group_keys = ["frequency_band", "four_day_group"]
    for (band, group_id), idx in out.groupby(group_keys, sort=True).groups.items():
        values = out.loc[idx, "power"].to_numpy(dtype=float)
        ref, n_ref = _iterative_group_reference(values, amplitude_filter_db, normalization_iterations)
        out.loc[idx, "initial_group_reference_power"] = ref
        out.loc[idx, "group_reference_n"] = n_ref
        if np.isfinite(ref) and ref > 0:
            ref_db = 10.0 * np.log10(ref)
            db = out.loc[idx, "raw_power_db"].to_numpy(dtype=float) - ref_db
            keep = np.isfinite(db) & (db <= float(amplitude_filter_db))
            out.loc[idx, "initial_group_reference_db"] = ref_db
            out.loc[idx, "keep_amplitude"] = keep

    keep = out["keep_amplitude"].to_numpy(dtype=bool) & np.isfinite(out["raw_power_db"].to_numpy(dtype=float))
    work = out.loc[keep, ["frequency_band", "frequency_mhz", "four_day_group", "pixel_index", "raw_power_db"]].copy()
    if work.empty:
        out["group_reference_power"] = np.nan
        out["normalized_power_ratio"] = np.nan
        out["relative_power_db"] = np.nan
        return out, pd.DataFrame.from_records(rows)

    work["group_offset_db"] = work.groupby(group_keys, sort=True)["raw_power_db"].transform("median")
    work["sky_cell_db"] = 0.0
    for _ in range(max(int(destripe_iterations), 1)):
        work["_sky_input"] = work["raw_power_db"] - work["group_offset_db"]
        sky = (
            work.groupby(["frequency_band", "frequency_mhz", "pixel_index"], sort=True)["_sky_input"]
            .median()
            .rename("sky_cell_db")
            .reset_index()
        )
        sky_center = sky.groupby("frequency_band", sort=True)["sky_cell_db"].transform("median")
        sky["sky_cell_db"] = sky["sky_cell_db"] - sky_center
        work = work.drop(columns=["sky_cell_db"], errors="ignore").merge(
            sky,
            on=["frequency_band", "frequency_mhz", "pixel_index"],
            how="left",
        )
        work["_group_input"] = work["raw_power_db"] - work["sky_cell_db"]
        offsets = (
            work.groupby(group_keys, sort=True)["_group_input"]
            .median()
            .rename("group_offset_db")
            .reset_index()
        )
        work = work.drop(columns=["group_offset_db"], errors="ignore").merge(offsets, on=group_keys, how="left")

    offsets = (
        work.groupby(["frequency_band", "frequency_mhz", "four_day_group"], sort=True)["group_offset_db"]
        .median()
        .reset_index()
    )
    fallback_offsets = (
        out.groupby(group_keys, sort=True)["initial_group_reference_db"]
        .median()
        .rename("fallback_group_offset_db")
        .reset_index()
    )
    offsets = offsets.merge(fallback_offsets, on=group_keys, how="outer")
    offsets["group_offset_db"] = offsets["group_offset_db"].fillna(offsets["fallback_group_offset_db"])
    offsets["group_reference_power"] = 10.0 ** (offsets["group_offset_db"] / 10.0)
    out = out.merge(
        offsets[group_keys + ["group_offset_db", "group_reference_power"]],
        on=group_keys,
        how="left",
    )
    out["relative_power_db"] = out["raw_power_db"] - out["group_offset_db"]
    out["normalized_power_ratio"] = 10.0 ** (out["relative_power_db"] / 10.0)

    sky_final = (
        work.groupby(["frequency_band", "pixel_index"], sort=True)["sky_cell_db"]
        .median()
        .rename("sky_cell_db")
        .reset_index()
    )
    sky_counts = (
        work.groupby(["frequency_band", "pixel_index"], sort=True)
        .size()
        .rename("n_destripe_samples")
        .reset_index()
    )
    sky_final = sky_final.merge(sky_counts, on=["frequency_band", "pixel_index"], how="left")
    for (band, group_id), idx in out.groupby(group_keys, sort=True).groups.items():
        group_offset = float(np.nanmedian(out.loc[idx, "group_offset_db"].to_numpy(dtype=float)))
        group_reference = 10.0 ** (group_offset / 10.0) if np.isfinite(group_offset) else np.nan
        band_sky = sky_final[sky_final["frequency_band"].astype(int).eq(int(band))]
        rows.append(
            {
                "frequency_band": int(band),
                "frequency_mhz": float(FREQUENCY_MAP_MHZ[int(band)]),
                "four_day_group": int(group_id),
                "group_start_time": origin + pd.to_timedelta(int(group_id) * float(group_days), unit="D"),
                "n_samples": int(len(idx)),
                "group_reference_power": group_reference,
                "group_reference_db": group_offset,
                "group_reference_n": int(np.count_nonzero(out.loc[idx, "keep_amplitude"].to_numpy(dtype=bool))),
                "n_keep_amplitude": int(np.count_nonzero(out.loc[idx, "keep_amplitude"].to_numpy(dtype=bool))),
                "normalization_mode": "destripe",
                "destripe_iterations": int(destripe_iterations),
                "n_destripe_sky_cells": int(len(band_sky)),
                "median_destripe_sky_cell_db": float(np.nanmedian(band_sky["sky_cell_db"])) if not band_sky.empty else np.nan,
            }
        )
    return out, pd.DataFrame.from_records(rows)


def _normalize_selected_samples(
    df: pd.DataFrame,
    start_time: str | None,
    group_days: float,
    amplitude_filter_db: float,
    normalization_iterations: int,
    normalization_mode: str,
    destripe_iterations: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mode = str(normalization_mode).strip().lower().replace("_", "-")
    if mode in {"four-day-median", "median", "legacy"}:
        normalized, group_norm = _normalize_four_day_groups(
            df,
            start_time,
            group_days,
            amplitude_filter_db,
            normalization_iterations,
        )
        group_norm["normalization_mode"] = "four-day-median"
        return normalized, group_norm
    if mode == "destripe":
        return _normalize_destriped_groups(
            df,
            start_time,
            group_days,
            amplitude_filter_db,
            normalization_iterations,
            destripe_iterations,
        )
    raise ValueError(f"unsupported normalization mode: {normalization_mode}")


def ten_minute_jump_keep(
    times: pd.Series | pd.DatetimeIndex,
    values_db: np.ndarray,
    threshold_db: float,
    bin_seconds: float,
) -> np.ndarray:
    time_index = pd.DatetimeIndex(times)
    values = np.asarray(values_db, dtype=float)
    keep = np.isfinite(values)
    if len(values) == 0:
        return keep
    origin_ns = int(time_index.min().value)
    bin_ns = int(float(bin_seconds) * 1e9)
    bins = ((datetime_ns(time_index) - origin_ns) // bin_ns).astype(int)
    med = pd.DataFrame({"bin": bins, "value": values}).groupby("bin")["value"].median().sort_index()
    if len(med) <= 1:
        return keep
    bad_bins = set(med.index.to_numpy()[1:][np.abs(np.diff(med.to_numpy(dtype=float))) > float(threshold_db)])
    if bad_bins:
        keep &= ~np.isin(bins, list(bad_bins))
    return keep


def _apply_ten_minute_jump_filter(df: pd.DataFrame, threshold_db: float, bin_seconds: float) -> pd.DataFrame:
    out = df.copy()
    out["keep_10min_jump"] = False
    for (_, _), idx in out[out["keep_amplitude"]].groupby(["frequency_band", "four_day_group"], sort=True).groups.items():
        local = out.loc[idx].sort_values("time")
        keep = ten_minute_jump_keep(local["time"], local["relative_power_db"].to_numpy(dtype=float), threshold_db, bin_seconds)
        out.loc[local.index, "keep_10min_jump"] = keep
    return out


def iterative_sigma_clip(values: np.ndarray, sigma: float = 4.0, max_iter: int = 6) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    keep = np.isfinite(arr)
    for _ in range(int(max_iter)):
        if np.count_nonzero(keep) < 3:
            break
        center = float(np.nanmedian(arr[keep]))
        scale = robust_sigma(arr[keep] - center)
        if not np.isfinite(scale) or scale <= 0:
            break
        next_keep = keep & (np.abs(arr - center) <= float(sigma) * scale)
        if np.array_equal(next_keep, keep):
            break
        keep = next_keep
    return keep


def _build_map_tables(df: pd.DataFrame, grid_step_deg: float, min_bin_samples: int, sigma_clip: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df[df["keep_10min_jump"]].copy()
    rows = []
    for (band, lon_idx, lat_idx), grp in work.groupby(["frequency_band", "lon_index", "lat_index"], sort=True):
        values = grp["relative_power_db"].to_numpy(dtype=float)
        clip_keep = iterative_sigma_clip(values, sigma=sigma_clip)
        clipped = values[clip_keep]
        rows.append(
            {
                "frequency_band": int(band),
                "frequency_mhz": float(FREQUENCY_MAP_MHZ[int(band)]),
                "lon_index": int(lon_idx),
                "lat_index": int(lat_idx),
                "pixel_index": int(lat_idx) * int(round(360.0 / grid_step_deg)) + int(lon_idx),
                "galactic_l_deg": float(lon_idx) * float(grid_step_deg) + 0.5 * float(grid_step_deg),
                "galactic_b_deg": -90.0 + float(lat_idx) * float(grid_step_deg) + 0.5 * float(grid_step_deg),
                "n_input_samples": int(len(values)),
                "n_clipped_samples": int(len(clipped)),
                "n_sigma_rejected": int(len(values) - len(clipped)),
                "relative_power_db_mean": float(np.nanmean(clipped)) if len(clipped) else np.nan,
                "relative_power_db_median": float(np.nanmedian(clipped)) if len(clipped) else np.nan,
                "relative_power_db_std": float(np.nanstd(clipped, ddof=1)) if len(clipped) > 1 else np.nan,
                "normalized_power_ratio_mean": float(np.nanmean(10.0 ** (clipped / 10.0))) if len(clipped) else np.nan,
            }
        )
    bin_table = pd.DataFrame.from_records(rows)
    n_lon = int(round(360.0 / grid_step_deg))
    n_lat = int(round(180.0 / grid_step_deg))
    grid_rows = []
    bands = sorted(df["frequency_band"].dropna().astype(int).unique())
    for band in bands:
        for lat_idx in range(n_lat):
            for lon_idx in range(n_lon):
                pixel = lat_idx * n_lon + lon_idx
                grid_rows.append(
                    {
                        "source_name": f"upper_v_scan_l{lon_idx:02d}_b{lat_idx:02d}",
                        "frequency_band": int(band),
                        "frequency_mhz": float(FREQUENCY_MAP_MHZ[int(band)]),
                        "lon_index": int(lon_idx),
                        "lat_index": int(lat_idx),
                        "pixel_index": int(pixel),
                        "galactic_l_deg": float(lon_idx) * float(grid_step_deg) + 0.5 * float(grid_step_deg),
                        "galactic_b_deg": -90.0 + float(lat_idx) * float(grid_step_deg) + 0.5 * float(grid_step_deg),
                    }
                )
    map_table = pd.DataFrame.from_records(grid_rows)
    if not bin_table.empty:
        map_table = map_table.merge(
            bin_table,
            on=["frequency_band", "frequency_mhz", "lon_index", "lat_index", "pixel_index", "galactic_l_deg", "galactic_b_deg"],
            how="left",
        )
    else:
        for col in [
            "n_input_samples",
            "n_clipped_samples",
            "n_sigma_rejected",
            "relative_power_db_mean",
            "relative_power_db_median",
            "relative_power_db_std",
            "normalized_power_ratio_mean",
        ]:
            map_table[col] = np.nan
    map_table["coverage_count"] = pd.to_numeric(map_table["n_clipped_samples"], errors="coerce").fillna(0).astype(int)
    map_table["passes_coverage"] = map_table["coverage_count"].ge(int(min_bin_samples))
    map_table["relative_brightness"] = map_table["relative_power_db_mean"]
    return bin_table, map_table


def _stage_summary(df: pd.DataFrame, map_table: pd.DataFrame, min_bin_samples: int) -> pd.DataFrame:
    rows = []
    for band, grp in df.groupby("frequency_band", sort=True):
        m = map_table[map_table["frequency_band"].astype(int).eq(int(band))]
        rows.append(
            {
                "frequency_band": int(band),
                "frequency_mhz": float(FREQUENCY_MAP_MHZ[int(band)]),
                "n_loaded_valid_positive": int(len(grp)),
                "n_after_amplitude_filter": int(grp["keep_amplitude"].sum()),
                "n_after_10min_jump_filter": int(grp["keep_10min_jump"].sum()),
                "n_map_cells": int(len(m)),
                "n_cells_passing_coverage": int(m["passes_coverage"].sum()) if not m.empty else 0,
                "min_bin_samples": int(min_bin_samples),
                "median_cell_samples": float(np.nanmedian(m["coverage_count"])) if not m.empty else np.nan,
                "max_cell_samples": int(np.nanmax(m["coverage_count"])) if not m.empty else 0,
            }
        )
    return pd.DataFrame.from_records(rows)


def _initial_selection_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for band, grp in df.groupby("frequency_band", sort=True):
        row = {
            "frequency_band": int(band),
            "n_loaded_valid_positive": int(len(grp)),
        }
        if "keep_new_moon" in grp.columns:
            row["n_new_moon_window"] = int(grp["keep_new_moon"].sum())
        if "keep_earth_beam" in grp.columns:
            row["n_earth_beam_window"] = int(grp["keep_earth_beam"].sum())
        if "keep_earth_beam_gain_veto" in grp.columns:
            row["n_earth_beam_gain_window"] = int(grp["keep_earth_beam_gain_veto"].sum())
            row["n_earth_beam_gain_veto_rejected"] = int((~grp["keep_earth_beam_gain_veto"].astype(bool)).sum())
        if "keep_sun_beam_veto" in grp.columns:
            row["n_sun_beam_window"] = int(grp["keep_sun_beam_veto"].sum())
            row["n_sun_beam_veto_rejected"] = int((~grp["keep_sun_beam_veto"].astype(bool)).sum())
        if "keep_initial_selection" in grp.columns:
            row["n_initial_selection"] = int(grp["keep_initial_selection"].sum())
        if "earth_beam_separation_deg" in grp.columns:
            sep = pd.to_numeric(grp["earth_beam_separation_deg"], errors="coerce")
            row["earth_beam_separation_median_deg"] = float(np.nanmedian(sep))
            row["earth_beam_separation_p10_deg"] = float(np.nanpercentile(sep, 10))
            row["earth_beam_separation_p90_deg"] = float(np.nanpercentile(sep, 90))
        if "earth_beam_relative_gain_db" in grp.columns:
            gain = pd.to_numeric(grp["earth_beam_relative_gain_db"], errors="coerce")
            row["earth_beam_relative_gain_median_db"] = float(np.nanmedian(gain))
            row["earth_beam_relative_gain_p90_db"] = float(np.nanpercentile(gain, 90))
            row["earth_beam_model_frequency_mhz"] = float(np.nanmedian(grp["earth_beam_model_frequency_mhz"].to_numpy(dtype=float)))
        if "earth_visible_by_moon_center" in grp.columns:
            row["n_earth_center_visible_by_moon"] = int(grp["earth_visible_by_moon_center"].astype(bool).sum())
        if "sun_beam_separation_deg" in grp.columns:
            sep = pd.to_numeric(grp["sun_beam_separation_deg"], errors="coerce")
            row["sun_beam_separation_median_deg"] = float(np.nanmedian(sep))
            row["sun_beam_separation_p10_deg"] = float(np.nanpercentile(sep, 10))
            row["sun_beam_separation_p90_deg"] = float(np.nanpercentile(sep, 90))
        if "sun_beam_relative_gain_db" in grp.columns:
            gain = pd.to_numeric(grp["sun_beam_relative_gain_db"], errors="coerce")
            row["sun_beam_relative_gain_median_db"] = float(np.nanmedian(gain))
            row["sun_beam_relative_gain_p90_db"] = float(np.nanpercentile(gain, 90))
            row["sun_beam_model_frequency_mhz"] = float(np.nanmedian(grp["sun_beam_model_frequency_mhz"].to_numpy(dtype=float)))
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def _coverage_counts(map_table: pd.DataFrame) -> pd.DataFrame:
    return (
        map_table.groupby("frequency_mhz", as_index=False)
        .agg(n_pixels=("pixel_index", "size"), n_uncovered=("passes_coverage", lambda s: int((~s.astype(bool)).sum())))
        .sort_values("frequency_mhz")
    )


def _regularize_band(
    band: pd.DataFrame,
    value_col: str,
    lon_step_deg: float,
    lat_step_deg: float,
    smooth_sigma_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pivot = (
        band.pivot_table(index="galactic_b_deg", columns="galactic_l_deg", values=value_col, aggfunc="first")
        .sort_index()
        .sort_index(axis=1)
    )
    lons = pivot.columns.to_numpy(dtype=float)
    lats = pivot.index.to_numpy(dtype=float)
    vals = pivot.to_numpy(dtype=float)
    coverage = (
        band.assign(_covered=band["passes_coverage"].astype(bool))
        .pivot_table(index="galactic_b_deg", columns="galactic_l_deg", values="_covered", aggfunc="first")
        .reindex(index=pivot.index, columns=pivot.columns)
        .fillna(False)
        .to_numpy(dtype=bool)
    )
    lon_ext = np.r_[lons[-1] - 360.0, lons, lons[0] + 360.0]
    vals_ext = np.column_stack([vals[:, -1], vals, vals[:, 0]])
    cov_ext = np.column_stack([coverage[:, -1], coverage, coverage[:, 0]])
    interp = RegularGridInterpolator((lats, lon_ext), vals_ext, bounds_error=False, fill_value=np.nan)
    cov_interp = RegularGridInterpolator((lats, lon_ext), cov_ext.astype(float), method="nearest", bounds_error=False, fill_value=0.0)
    lon_plot = np.arange(-180.0, 180.0 + 0.5 * lon_step_deg, lon_step_deg)
    lon_fine = (-lon_plot) % 360.0
    lat_fine = np.arange(-89.5, 90.0, lat_step_deg)
    lon_mesh, lat_mesh = np.meshgrid(lon_fine, lat_fine)
    lon_plot_mesh, _ = np.meshgrid(lon_plot, lat_fine)
    query = np.column_stack([lat_mesh.ravel(), lon_mesh.ravel()])
    image = interp(query).reshape(lat_mesh.shape)
    covered = cov_interp(query).reshape(lat_mesh.shape) >= 0.5
    if smooth_sigma_deg > 0:
        sigma = float(smooth_sigma_deg) / max(float(lat_step_deg), 1e-9)
        finite = np.isfinite(image) & covered
        weighted = gaussian_filter(np.where(finite, image, 0.0), sigma=sigma, mode=("nearest", "wrap"))
        norm = gaussian_filter(finite.astype(float), sigma=sigma, mode=("nearest", "wrap"))
        image = np.divide(weighted, norm, out=np.full_like(weighted, np.nan), where=norm > 0)
    image = np.where(covered, image, np.nan)
    return lon_plot_mesh, lat_mesh, image, covered


def _mollweide_xy(lon_mesh: np.ndarray, lat_mesh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.deg2rad(lon_mesh), np.deg2rad(lat_mesh)


def _add_haslam_axes(ax: plt.Axes) -> None:
    ax.grid(color="white", alpha=0.28, linewidth=0.55)
    ticks = np.array([-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150])
    ax.set_xticks(np.deg2rad(ticks))
    ax.set_xticklabels([f"{int((-tick) % 360)}°" for tick in ticks], fontsize=7)
    ax.set_yticks(np.deg2rad([-60, -30, 0, 30, 60]))
    ax.set_yticklabels(["-60°", "-30°", "0°", "+30°", "+60°"], fontsize=7)
    ax.text(0.5, -0.08, "Galactic longitude", transform=ax.transAxes, ha="center", va="top", fontsize=8)


def _display_vmax(values: np.ndarray, percentile: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 1.0
    vmax = float(np.nanpercentile(np.abs(arr), float(percentile)))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.nanmax(np.abs(arr)))
    return vmax if np.isfinite(vmax) and vmax > 0 else 1.0


def _format_scale(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    if abs(value) >= 1:
        return f"{value:.2f}"
    return f"{value:.2g}"


def _map_value_label(config: dict[str, object]) -> str:
    mode = str(config.get("normalization_mode", "four-day-median")).strip().lower().replace("_", "-")
    if mode == "destripe":
        return "destriped relative dB; sky cells solved jointly with four-day offsets"
    return "four-day normalized relative dB"


def _overlay_uncovered(ax: plt.Axes, x: np.ndarray, y: np.ndarray, covered: np.ndarray) -> None:
    uncovered = np.where(~covered, 1.0, np.nan)
    if np.isfinite(uncovered).any():
        ax.pcolormesh(x, y, uncovered, shading="auto", cmap=UNCOVERED_CMAP, vmin=0.0, vmax=1.0, rasterized=True)
        ax.contourf(x, y, uncovered, levels=[0.5, 1.5], colors="none", hatches=["////"], alpha=0.0)


def _add_colorbar(fig: plt.Figure, ax: plt.Axes, image, vmax: float) -> None:
    cax = inset_axes(ax, width="46%", height="5%", loc="lower center", borderpad=1.08)
    cbar = fig.colorbar(image, cax=cax, orientation="horizontal")
    label = _format_scale(vmax)
    cbar.set_ticks([-vmax, 0.0, vmax])
    cbar.set_ticklabels([f"-{label}", "0", f"+{label}"])
    cbar.ax.tick_params(labelsize=5.5, length=1.8, pad=1)
    cbar.outline.set_linewidth(0.4)


def _plot_panel(map_table: pd.DataFrame, out_dir: Path, config: dict[str, object]) -> Path:
    freqs = sorted(map_table["frequency_mhz"].dropna().unique())
    ncols = 3
    nrows = int(np.ceil(len(freqs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16.0, 3.15 * nrows + 1.0), subplot_kw={"projection": "mollweide"})
    axes_arr = np.asarray(axes).ravel()
    for ax, freq in zip(axes_arr, freqs):
        band = map_table[np.isclose(map_table["frequency_mhz"], freq)].copy()
        lon, lat, data, covered = _regularize_band(
            band,
            "relative_brightness",
            lon_step_deg=1.0,
            lat_step_deg=1.0,
            smooth_sigma_deg=float(config["smooth_sigma_deg"]),
        )
        x, y = _mollweide_xy(lon, lat)
        vmax = _display_vmax(data, float(config["scale_percentile"]))
        image = ax.pcolormesh(x, y, data, shading="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, rasterized=True)
        _overlay_uncovered(ax, x, y, covered)
        levels = np.linspace(-vmax, vmax, 9)
        levels = levels[np.abs(levels) > 1e-12]
        if levels.size:
            ax.contour(x, y, data, levels=levels, colors="black", linewidths=0.35, alpha=0.48)
        ax.contour(x, y, data, levels=[0.0], colors="black", linewidths=0.65, alpha=0.72)
        _add_haslam_axes(ax)
        ax.set_title(f"{freq:.2f} MHz, scale +/-{_format_scale(vmax)} dB", fontsize=9.2)
        _add_colorbar(fig, ax, image, vmax)
    for ax in axes_arr[len(freqs) :]:
        ax.set_visible(False)
    fig.suptitle(
        f"Upper-V Novaco-Brown-style scan maps ({config['antenna']}); Galactic Mollweide\n"
        f"{_map_value_label(config)}; each frequency uses its own scale; gray hatch marks uncovered cells.",
        y=0.995,
        fontsize=13,
    )
    fig.legend(
        handles=[Patch(facecolor="0.70", edgecolor="0.25", hatch="////", label="not covered / below sample threshold")],
        loc="lower right",
        bbox_to_anchor=(0.985, 0.02),
        frameon=False,
        fontsize=8,
    )
    fig.subplots_adjust(left=0.03, right=0.985, top=0.88, bottom=0.12, wspace=0.08, hspace=0.25)
    path = out_dir / "upper_v_novaco_brown_style_maps_panel.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def _plot_single_maps(map_table: pd.DataFrame, out_dir: Path, config: dict[str, object]) -> list[Path]:
    paths = []
    for freq in sorted(map_table["frequency_mhz"].dropna().unique()):
        band = map_table[np.isclose(map_table["frequency_mhz"], freq)].copy()
        lon, lat, data, covered = _regularize_band(
            band,
            "relative_brightness",
            lon_step_deg=0.5,
            lat_step_deg=0.5,
            smooth_sigma_deg=float(config["smooth_sigma_deg"]),
        )
        x, y = _mollweide_xy(lon, lat)
        vmax = _display_vmax(data, float(config["scale_percentile"]))
        fig = plt.figure(figsize=(12.4, 6.8))
        ax = fig.add_subplot(111, projection="mollweide")
        image = ax.pcolormesh(x, y, data, shading="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, rasterized=True)
        _overlay_uncovered(ax, x, y, covered)
        levels = np.linspace(-vmax, vmax, 11)
        levels = levels[np.abs(levels) > 1e-12]
        if levels.size:
            ax.contour(x, y, data, levels=levels, colors="black", linewidths=0.45, alpha=0.55)
        ax.contour(x, y, data, levels=[0.0], colors="black", linewidths=0.75, alpha=0.72)
        _add_haslam_axes(ax)
        uncovered = int((~band["passes_coverage"].astype(bool)).sum())
        ax.set_title(
            f"Upper-V Novaco-Brown-style scan map, {freq:.2f} MHz; scale +/-{_format_scale(vmax)} dB; uncovered cells: {uncovered}",
            fontsize=12.0,
        )
        cbar = fig.colorbar(image, ax=ax, orientation="horizontal", fraction=0.055, pad=0.09)
        cbar.set_label(_map_value_label(config))
        ax.legend(
            handles=[Patch(facecolor="0.70", edgecolor="0.25", hatch="////", label="not covered / below sample threshold")],
            loc="lower right",
            bbox_to_anchor=(1.0, -0.11),
            frameon=False,
            fontsize=8,
        )
        fig.subplots_adjust(left=0.03, right=0.98, top=0.91, bottom=0.13)
        label = f"{freq:.2f}".replace(".", "p")
        path = out_dir / f"upper_v_{label}mhz_novaco_brown_style.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        paths.append(path)
    return paths


def _write_report(
    out_dir: Path,
    config: dict[str, object],
    summary: pd.DataFrame,
    coverage: pd.DataFrame,
    group_norm: pd.DataFrame,
    paths: list[Path],
) -> Path:
    selection_mode = str(config.get("initial_selection_mode", "new-moon")).replace("_", "-")
    if selection_mode == "new-moon":
        selection_rule = "- Select quiet intervals centered on new Moon to suppress terrestrial contamination."
    elif selection_mode == "earth-beam":
        selection_rule = (
            f"- Reject samples where Earth is within {float(config['earth_beam_min_separation_deg']):g} deg "
            f"of the `{config['earth_beam_axis']}` upper-V beam axis."
        )
    elif selection_mode == "new-moon-and-earth-beam":
        selection_rule = (
            "- Select quiet intervals centered on new Moon and also reject samples where Earth is within "
            f"{float(config['earth_beam_min_separation_deg']):g} deg of the `{config['earth_beam_axis']}` upper-V beam axis."
        )
    elif selection_mode == "earth-beam-gain-veto":
        selection_rule = (
            f"- Reject samples where Earth has upper-V relative beam gain >= "
            f"{float(config['earth_beam_gain_threshold_db']):g} dB using the nearest digitized beam pattern for each frequency bin."
        )
    elif selection_mode == "new-moon-and-earth-beam-gain-veto":
        selection_rule = (
            "- Select quiet intervals centered on new Moon and also reject samples where Earth has upper-V relative beam gain >= "
            f"{float(config['earth_beam_gain_threshold_db']):g} dB using the nearest digitized beam pattern for each frequency bin."
        )
    elif selection_mode == "sun-beam-veto":
        selection_rule = (
            f"- Reject samples where the Sun has upper-V relative beam gain >= "
            f"{float(config['sun_beam_gain_threshold_db']):g} dB using the nearest digitized beam pattern for each frequency bin."
        )
    elif selection_mode == "new-moon-and-sun-beam-veto":
        selection_rule = (
            "- Select quiet intervals centered on new Moon and also reject samples where the Sun has upper-V relative beam gain >= "
            f"{float(config['sun_beam_gain_threshold_db']):g} dB using the nearest digitized beam pattern for each frequency bin."
        )
    elif selection_mode == "none":
        selection_rule = "- Use all valid positive samples before statistical editing."
    else:
        selection_rule = f"- Initial selection mode: `{selection_mode}`."
    lines = [
        "# Upper-V Novaco-Brown-Style Scan Maps",
        "",
        "## What This Run Does",
        "",
        "This is a scan-map reduction, not a lunar-occultation inversion.",
        "Samples are assigned to the Galactic position of the upper-V pointing direction, then statistically edited and averaged in 5 degree cells.",
        "",
        "## Paper-Derived Reduction Rules Implemented",
        "",
        "- Use the outward upper-V Ryle-Vonberg antenna as the sky-scanning channel.",
        selection_rule,
        "- Normalize data groups before combining them.",
        "- Reject samples more than the configured dB threshold above the group average.",
        "- Reject ten-minute intervals with jumps larger than the configured dB threshold.",
        "- Bin samples into Galactic longitude/latitude cells and iteratively 4-sigma clip each cell.",
        "",
        "## Important Deviations From The Published Maps",
        "",
        f"- Current antenna column: `{config['antenna']}`.",
        f"- Input format: `{config.get('input_format_resolved', config['input_format'])}`.",
        f"- Pointing coordinates come from `right_ascension`/`declination`; RA was interpreted as `{config.get('ra_units_interpreted', config['ra_units'])}`.",
        f"- Initial sample selection mode: `{selection_mode}`.",
        f"- Normalization mode: `{config.get('normalization_mode', 'four-day-median')}`.",
        "- Novaco & Brown state that their Galactic contour maps used the upper-V fine output; coarse-channel runs are useful for coverage but are not an exact receiver-mode match.",
        "- This run does not apply the IMP-6 absolute brightness-temperature calibration. Values are relative dB after the configured normalization.",
        "- Manual visual editing described in the paper is approximated here by reproducible statistical filters.",
        "- Date handling: "
        + (
            "this run uses all available dates in the input table."
            if config.get("all_dates")
            else "this run starts at the configured post-deployment start time, not the full multiyear mission interval."
        ),
        "",
        "## Data Summary",
        "",
        summary.to_string(index=False) if not summary.empty else "No selected samples.",
        "",
        "## Coverage",
        "",
        coverage.to_string(index=False) if not coverage.empty else "No map cells.",
        "",
        "## Group Normalization Summary",
        "",
        group_norm.groupby("frequency_mhz", as_index=False)
        .agg(n_groups=("four_day_group", "nunique"), median_group_reference_power=("group_reference_power", "median"))
        .to_string(index=False)
        if not group_norm.empty
        else "No groups.",
        "",
        "## Outputs",
        "",
    ]
    lines.extend(f"- `{path}`" for path in paths)
    lines.extend(
        [
            "",
            "## Sources",
            "",
            "- Novaco & Brown, 'Nonthermal Galactic Emission Below 10 MHz', ApJ 221, 114-123.",
            "- NASA NTRS record: https://ntrs.nasa.gov/citations/19780047176",
            "- NASA NTRS PDF/preprint: https://ntrs.nasa.gov/api/citations/19770024123/downloads/19770024123.pdf",
            "",
            "## Configuration",
            "",
            pd.Series(config).to_string(),
            "",
        ]
    )
    path = out_dir / "upper_v_novaco_brown_scan_map_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run(args: argparse.Namespace) -> Path:
    out_dir = ensure_dir(args.out_dir)
    bands = parse_frequencies(args.frequencies)
    start_time = effective_start_time(args.start_time, bool(getattr(args, "all_dates", False)))
    input_format_resolved = _detect_input_format(Path(args.cleaned), str(args.input_format))
    config = {
        "cleaned_timeseries": str(args.cleaned),
        "input_format": str(args.input_format),
        "input_format_resolved": input_format_resolved,
        "antenna": str(args.antenna),
        "frequencies": bands,
        "frequencies_mhz": [FREQUENCY_MAP_MHZ[band] for band in bands],
        "start_time": start_time,
        "end_time": args.end_time,
        "all_dates": bool(getattr(args, "all_dates", False)),
        "grid_step_deg": float(args.grid_step_deg),
        "ra_units": str(args.ra_units),
        "initial_selection_mode": str(args.initial_selection_mode),
        "new_moon_half_window_days": float(args.new_moon_half_window_days),
        "earth_beam_axis": str(args.earth_beam_axis),
        "earth_beam_min_separation_deg": float(args.earth_beam_min_separation_deg),
        "earth_beam_gain_threshold_db": float(args.earth_beam_gain_threshold_db),
        "sun_beam_axis": str(args.sun_beam_axis),
        "sun_beam_gain_threshold_db": float(args.sun_beam_gain_threshold_db),
        "sun_vector_cadence_hours": float(args.sun_vector_cadence_hours),
        "beam_pattern_directory": str(BEAM_DIR),
        "group_days": float(args.group_days),
        "amplitude_filter_db": float(args.amplitude_filter_db),
        "normalization_iterations": int(args.normalization_iterations),
        "normalization_mode": str(args.normalization_mode),
        "destripe_iterations": int(args.destripe_iterations),
        "jump_bin_seconds": float(args.jump_bin_seconds),
        "jump_threshold_db": float(args.jump_threshold_db),
        "sigma_clip": float(args.sigma_clip),
        "min_bin_samples": int(args.min_bin_samples),
        "smooth_sigma_deg": float(args.smooth_sigma_deg),
        "scale_percentile": float(args.scale_percentile),
        "max_rows": int(args.max_rows),
        "new_moon_reference_utc": str(REFERENCE_NEW_MOON),
        "synodic_month_days": SYNODIC_MONTH_DAYS,
        "software_versions": software_versions(),
    }
    write_json(out_dir / "run_config.json", config)

    selected = _read_selected_rows(
        Path(args.cleaned),
        str(args.antenna),
        bands,
        start_time,
        args.end_time,
        int(args.max_rows),
        input_format_resolved,
    )
    if selected.empty:
        raise SystemExit("No matching valid positive upper-V samples found.")
    selected, pointing_ra_units, n_removed_invalid_pointing = _filter_valid_pointing(selected, str(args.ra_units))
    config["pointing_ra_units_interpreted_for_validation"] = str(pointing_ra_units)
    config["n_removed_invalid_pointing"] = int(n_removed_invalid_pointing)
    if selected.empty:
        raise SystemExit("No samples remain after RA/Dec pointing validity filtering.")
    selected["new_moon_distance_days"] = new_moon_distance_days(selected["time"])
    selected["keep_new_moon"] = selected["new_moon_distance_days"].le(float(args.new_moon_half_window_days))
    selected["keep_earth_beam"] = True
    selected["keep_earth_beam_gain_veto"] = True
    selected["keep_sun_beam_veto"] = True
    selected["keep_initial_selection"] = True
    selection_mode = str(args.initial_selection_mode).strip().lower().replace("_", "-")
    if selection_mode in {"earth-beam", "new-moon-and-earth-beam"}:
        selected, earth_beam_ra_units = _attach_earth_beam_separation(
            selected,
            str(args.earth_beam_axis),
            str(args.ra_units),
        )
        selected["keep_earth_beam"] = selected["earth_beam_separation_deg"].gt(float(args.earth_beam_min_separation_deg))
        if earth_beam_ra_units is not None:
            config["earth_beam_ra_units_interpreted"] = str(earth_beam_ra_units)
    if selection_mode in {"earth-beam-gain-veto", "new-moon-and-earth-beam-gain-veto"}:
        selected, earth_gain_ra_units = _attach_earth_beam_gain(
            selected,
            str(args.earth_beam_axis),
            str(args.ra_units),
        )
        selected["keep_earth_beam"] = selected["earth_beam_separation_deg"].gt(float(args.earth_beam_min_separation_deg))
        selected["keep_earth_beam_gain_veto"] = (
            np.isfinite(selected["earth_beam_relative_gain_db"].to_numpy(dtype=float))
            & selected["earth_beam_relative_gain_db"].lt(float(args.earth_beam_gain_threshold_db))
        )
        if earth_gain_ra_units is not None:
            config["earth_beam_ra_units_interpreted"] = str(earth_gain_ra_units)
    if selection_mode in {"sun-beam-veto", "new-moon-and-sun-beam-veto"}:
        selected, sun_beam_ra_units = _attach_sun_beam_gain(
            selected,
            str(args.sun_beam_axis),
            str(args.ra_units),
            float(args.sun_vector_cadence_hours),
        )
        selected["keep_sun_beam_veto"] = (
            np.isfinite(selected["sun_beam_relative_gain_db"].to_numpy(dtype=float))
            & selected["sun_beam_relative_gain_db"].lt(float(args.sun_beam_gain_threshold_db))
        )
        if sun_beam_ra_units is not None:
            config["sun_beam_ra_units_interpreted"] = str(sun_beam_ra_units)
    if selection_mode == "new-moon":
        selected["keep_initial_selection"] = selected["keep_new_moon"]
    elif selection_mode == "earth-beam":
        selected["keep_initial_selection"] = selected["keep_earth_beam"]
    elif selection_mode == "new-moon-and-earth-beam":
        selected["keep_initial_selection"] = selected["keep_new_moon"] & selected["keep_earth_beam"]
    elif selection_mode == "earth-beam-gain-veto":
        selected["keep_initial_selection"] = selected["keep_earth_beam_gain_veto"]
    elif selection_mode == "new-moon-and-earth-beam-gain-veto":
        selected["keep_initial_selection"] = selected["keep_new_moon"] & selected["keep_earth_beam_gain_veto"]
    elif selection_mode == "sun-beam-veto":
        selected["keep_initial_selection"] = selected["keep_sun_beam_veto"]
    elif selection_mode == "new-moon-and-sun-beam-veto":
        selected["keep_initial_selection"] = selected["keep_new_moon"] & selected["keep_sun_beam_veto"]
    elif selection_mode == "none":
        selected["keep_initial_selection"] = True
    else:
        raise ValueError(f"unsupported initial selection mode: {args.initial_selection_mode}")
    initial_summary = _initial_selection_summary(selected)
    selected = selected[selected["keep_initial_selection"]].copy()
    if selected.empty:
        raise SystemExit(f"No samples remain after initial selection mode: {selection_mode}.")
    selected = selected.drop(columns=EARTH_BEAM_GEOMETRY_COLUMNS, errors="ignore")
    selected = _attach_galactic_pointing(selected, str(args.ra_units), float(args.grid_step_deg))
    config["ra_units_interpreted"] = str(selected["ra_units_interpreted"].iloc[0])
    write_json(out_dir / "run_config.json", config)
    normalized, group_norm = _normalize_selected_samples(
        selected,
        start_time,
        float(args.group_days),
        float(args.amplitude_filter_db),
        int(args.normalization_iterations),
        str(args.normalization_mode),
        int(args.destripe_iterations),
    )
    filtered = _apply_ten_minute_jump_filter(normalized, float(args.jump_threshold_db), float(args.jump_bin_seconds))
    bin_table, map_table = _build_map_tables(filtered, float(args.grid_step_deg), int(args.min_bin_samples), float(args.sigma_clip))
    summary = _stage_summary(filtered, map_table, int(args.min_bin_samples))
    if not initial_summary.empty:
        summary = summary.drop(columns=["n_loaded_valid_positive", "n_new_moon_window"], errors="ignore").merge(
            initial_summary,
            on="frequency_band",
            how="left",
        )
        ordered = [
            "frequency_band",
            "frequency_mhz",
            "n_loaded_valid_positive",
            "n_new_moon_window",
            "n_earth_beam_window",
            "n_earth_beam_gain_window",
            "n_earth_beam_gain_veto_rejected",
            "n_sun_beam_window",
            "n_sun_beam_veto_rejected",
            "n_initial_selection",
            "n_after_amplitude_filter",
            "n_after_10min_jump_filter",
            "n_map_cells",
            "n_cells_passing_coverage",
            "min_bin_samples",
            "median_cell_samples",
            "max_cell_samples",
            "earth_beam_separation_median_deg",
            "earth_beam_separation_p10_deg",
            "earth_beam_separation_p90_deg",
            "earth_beam_model_frequency_mhz",
            "earth_beam_relative_gain_median_db",
            "earth_beam_relative_gain_p90_db",
            "n_earth_center_visible_by_moon",
            "sun_beam_model_frequency_mhz",
            "sun_beam_separation_median_deg",
            "sun_beam_separation_p10_deg",
            "sun_beam_separation_p90_deg",
            "sun_beam_relative_gain_median_db",
            "sun_beam_relative_gain_p90_db",
        ]
        summary = summary[[col for col in ordered if col in summary.columns]]
    coverage = _coverage_counts(map_table)

    group_norm.to_csv(out_dir / "upper_v_group_normalization.csv", index=False)
    summary.to_csv(out_dir / "upper_v_novaco_brown_filter_summary.csv", index=False)
    bin_table.to_csv(out_dir / "upper_v_novaco_brown_bin_table.csv", index=False)
    map_table.to_csv(out_dir / "upper_v_novaco_brown_map_table.csv", index=False)
    coverage.to_csv(out_dir / "upper_v_novaco_brown_coverage_summary.csv", index=False)

    paths = [
        out_dir / "run_config.json",
        out_dir / "upper_v_group_normalization.csv",
        out_dir / "upper_v_novaco_brown_filter_summary.csv",
        out_dir / "upper_v_novaco_brown_bin_table.csv",
        out_dir / "upper_v_novaco_brown_map_table.csv",
        out_dir / "upper_v_novaco_brown_coverage_summary.csv",
    ]
    paths.append(_plot_panel(map_table, out_dir, config))
    paths.extend(_plot_single_maps(map_table, out_dir, config))
    report = _write_report(out_dir, config, summary, coverage, group_norm, paths)
    print(report)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleaned", default=str(DEFAULT_CLEAN))
    parser.add_argument("--input-format", choices=["auto", "cleaned", "raw-master"], default="auto")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--antenna", default="rv1_coarse")
    parser.add_argument("--frequencies", default="novaco_brown")
    parser.add_argument("--start-time", default="1974-11-01")
    parser.add_argument("--all-dates", action="store_true", help="Use every date present in --cleaned, overriding --start-time.")
    parser.add_argument("--end-time", default=None)
    parser.add_argument("--grid-step-deg", type=float, default=5.0)
    parser.add_argument("--ra-units", choices=["auto", "hours", "degrees"], default="auto")
    parser.add_argument(
        "--initial-selection-mode",
        choices=[
            "new-moon",
            "earth-beam",
            "new-moon-and-earth-beam",
            "earth-beam-gain-veto",
            "new-moon-and-earth-beam-gain-veto",
            "sun-beam-veto",
            "new-moon-and-sun-beam-veto",
            "none",
        ],
        default="new-moon",
        help="Initial contamination/quiet-sample selection before normalization.",
    )
    parser.add_argument("--new-moon-half-window-days", type=float, default=6.0)
    parser.add_argument(
        "--earth-beam-axis",
        choices=["radec", "radial-upper"],
        default="radec",
        help="Beam axis used when --initial-selection-mode includes earth-beam.",
    )
    parser.add_argument(
        "--earth-beam-min-separation-deg",
        type=float,
        default=90.0,
        help="Keep samples only when Earth is farther than this angle from the beam axis.",
    )
    parser.add_argument(
        "--earth-beam-gain-threshold-db",
        type=float,
        default=-10.0,
        help="Reject samples when the Earth relative beam gain is at or above this dB threshold.",
    )
    parser.add_argument(
        "--sun-beam-axis",
        choices=["radec", "radial-upper"],
        default="radec",
        help="Beam axis used when --initial-selection-mode includes sun-beam-veto.",
    )
    parser.add_argument(
        "--sun-beam-gain-threshold-db",
        type=float,
        default=-10.0,
        help="Reject samples when the Sun relative beam gain is at or above this dB threshold.",
    )
    parser.add_argument(
        "--sun-vector-cadence-hours",
        type=float,
        default=1.0,
        help="Cadence for interpolated Astropy Sun vectors; 1 hour is sub-degree accurate for this veto.",
    )
    parser.add_argument("--group-days", type=float, default=4.0)
    parser.add_argument("--amplitude-filter-db", type=float, default=2.0)
    parser.add_argument("--normalization-iterations", type=int, default=2)
    parser.add_argument(
        "--normalization-mode",
        choices=["four-day-median", "destripe"],
        default="four-day-median",
        help="Use the legacy four-day median normalization or solve sky cells plus four-day offsets.",
    )
    parser.add_argument("--destripe-iterations", type=int, default=4)
    parser.add_argument("--jump-bin-seconds", type=float, default=600.0)
    parser.add_argument("--jump-threshold-db", type=float, default=1.5)
    parser.add_argument("--sigma-clip", type=float, default=4.0)
    parser.add_argument("--min-bin-samples", type=int, default=5)
    parser.add_argument("--smooth-sigma-deg", type=float, default=4.0)
    parser.add_argument("--scale-percentile", type=float, default=97.0)
    parser.add_argument("--max-rows", type=int, default=0, help="0 means no row cap after initial channel/date filtering.")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
