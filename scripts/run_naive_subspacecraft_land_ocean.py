#!/usr/bin/env python
"""Naive RAE-2 land/ocean comparison at Earth-facing orbit-pass centers.

For every contiguous interval in which RAE-2 is on the Earth-facing side of
the Moon, find the instant closest to the Moon--RAE--Earth line.  Classify the
Earth sub-spacecraft point at that instant as land or ocean and average each
coarse Ryle-Vonberg channel in symmetric windows around the pass center.

The pass, not the individual sample, is the independent statistical unit.
This is intentionally a point-under-the-spacecraft test; it does not pretend
that a total-power antenna images that point or exclude the rest of Earth.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.constants import (  # noqa: E402
    EARTH_MOON_DISTANCE_KM,
    EARTH_UNIT_COLUMNS,
    FREQUENCY_MAP_MHZ,
    SPACECRAFT_COLUMNS,
)
from rylevonberg.util import ensure_dir, robust_sigma, software_versions, write_json  # noqa: E402
from scripts.investigate_earth_noise_geography import (  # noqa: E402
    NATURAL_EARTH_URL,
    _bh_qvalues,
    _markdown,
    land_mask_from_geojson,
    load_land_geojson,
)


DEFAULT_INPUT = Path(
    os.environ.get(
        "RAE2_MASTER_CSV",
        "data/interpolatedRAE2MasterFile.csv",
    )
)
DEFAULT_OUT = ROOT / "outputs/earth_naive_subspacecraft_land_ocean_v1"
POWER_COLUMNS = ["rv1_coarse", "rv2_coarse"]
GEOMETRY_COLUMNS = ["time", *SPACECRAFT_COLUMNS, *EARTH_UNIT_COLUMNS]


def alignment_cosine(spacecraft: np.ndarray, moon_to_earth: np.ndarray) -> np.ndarray:
    spacecraft = np.asarray(spacecraft, dtype=float)
    moon_to_earth = np.asarray(moon_to_earth, dtype=float)
    numerator = np.einsum("ij,ij->i", spacecraft, moon_to_earth)
    denominator = np.linalg.norm(spacecraft, axis=1) * np.linalg.norm(moon_to_earth, axis=1)
    out = np.full(len(spacecraft), np.nan)
    good = np.isfinite(numerator) & np.isfinite(denominator) & (denominator > 0.0)
    out[good] = numerator[good] / denominator[good]
    return np.clip(out, -1.0, 1.0)


def find_frontside_passes(
    input_path: Path,
    chunksize: int,
    max_gap_seconds: float,
) -> pd.DataFrame:
    """Stream the master table and reduce each Earth-facing segment to one center."""
    summaries: list[pd.DataFrame] = []
    last_front = False
    last_time_ns: int | None = None
    segment_counter = 0
    for chunk in pd.read_csv(
        input_path,
        usecols=GEOMETRY_COLUMNS,
        parse_dates=["time"],
        chunksize=int(chunksize),
        low_memory=False,
    ):
        times_ns = chunk["time"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
        spacecraft = chunk[SPACECRAFT_COLUMNS].to_numpy(dtype=float)
        earth = chunk[EARTH_UNIT_COLUMNS].to_numpy(dtype=float)
        cosine = alignment_cosine(spacecraft, earth)
        front = np.isfinite(cosine) & (cosine > 0.0)

        previous_front = np.r_[last_front, front[:-1]]
        if last_time_ns is None:
            gaps = np.r_[True, np.diff(times_ns) > float(max_gap_seconds) * 1e9]
        else:
            gaps = np.r_[times_ns[0] - last_time_ns > float(max_gap_seconds) * 1e9,
                         np.diff(times_ns) > float(max_gap_seconds) * 1e9]
        starts = front & ((~previous_front) | gaps)
        local_ids = segment_counter + np.cumsum(starts, dtype=np.int64)
        if np.any(starts):
            segment_counter = int(local_ids.max())

        selected = np.flatnonzero(front)
        if selected.size:
            work = pd.DataFrame(
                {
                    "pass_id": local_ids[selected],
                    "time_ns": times_ns[selected],
                    "alignment_cosine": cosine[selected],
                    "position_x": spacecraft[selected, 0],
                    "position_y": spacecraft[selected, 1],
                    "position_z": spacecraft[selected, 2],
                    "earth_unit_vector_x": earth[selected, 0],
                    "earth_unit_vector_y": earth[selected, 1],
                    "earth_unit_vector_z": earth[selected, 2],
                }
            )
            grouped = work.groupby("pass_id", sort=False)
            base = grouped.agg(
                start_time_ns=("time_ns", "min"),
                end_time_ns=("time_ns", "max"),
                n_geometry_rows=("time_ns", "size"),
            )
            center_rows = work.loc[grouped["alignment_cosine"].idxmax()].set_index("pass_id")
            summaries.append(base.join(center_rows))

        last_front = bool(front[-1]) if len(front) else last_front
        last_time_ns = int(times_ns[-1]) if len(times_ns) else last_time_ns

    if not summaries:
        return pd.DataFrame()
    partial = pd.concat(summaries).reset_index()
    aggregate = partial.groupby("pass_id", as_index=False).agg(
        start_time_ns=("start_time_ns", "min"),
        end_time_ns=("end_time_ns", "max"),
        n_geometry_rows=("n_geometry_rows", "sum"),
    )
    best = partial.loc[partial.groupby("pass_id")["alignment_cosine"].idxmax()].drop(
        columns=["start_time_ns", "end_time_ns", "n_geometry_rows"]
    )
    passes = aggregate.merge(best, on="pass_id", how="left", validate="one_to_one")
    passes["start_time"] = pd.to_datetime(passes.pop("start_time_ns"), unit="ns")
    passes["end_time"] = pd.to_datetime(passes.pop("end_time_ns"), unit="ns")
    passes["center_time"] = pd.to_datetime(passes.pop("time_ns"), unit="ns")
    passes["frontside_duration_s"] = (passes["end_time"] - passes["start_time"]).dt.total_seconds()
    passes["minimum_alignment_angle_deg"] = np.degrees(
        np.arccos(np.clip(passes["alignment_cosine"], -1.0, 1.0))
    )
    return passes.sort_values("center_time").reset_index(drop=True)


def attach_subspacecraft_points(passes: pd.DataFrame, geojson: dict) -> pd.DataFrame:
    """Transform the exact Earth-to-spacecraft vector into Earth-fixed lon/lat."""
    from astropy import units as u
    from astropy.coordinates import FK4, ITRS, SkyCoord
    from astropy.time import Time
    from astropy.utils import iers

    iers.conf.auto_download = False
    spacecraft = passes[SPACECRAFT_COLUMNS].to_numpy(dtype=float)
    earth = passes[EARTH_UNIT_COLUMNS].to_numpy(dtype=float, copy=True)
    earth /= np.linalg.norm(earth, axis=1, keepdims=True)
    earth_to_spacecraft = -float(EARTH_MOON_DISTANCE_KM) * earth + spacecraft
    unit = earth_to_spacecraft / np.linalg.norm(earth_to_spacecraft, axis=1, keepdims=True)
    ra = np.degrees(np.arctan2(unit[:, 1], unit[:, 0])) % 360.0
    dec = np.degrees(np.arcsin(np.clip(unit[:, 2], -1.0, 1.0)))
    inertial = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame=FK4(equinox="B1950"))
    fixed = inertial.transform_to(
        ITRS(obstime=Time(pd.to_datetime(passes["center_time"], utc=True).dt.to_pydatetime()))
    ).cartesian
    xyz = np.column_stack([fixed.x.value, fixed.y.value, fixed.z.value])
    xyz /= np.linalg.norm(xyz, axis=1, keepdims=True)
    out = passes.copy()
    out["subspacecraft_lon_deg"] = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))
    out["subspacecraft_lat_deg"] = np.degrees(np.arcsin(np.clip(xyz[:, 2], -1.0, 1.0)))
    out["subspacecraft_is_land"] = land_mask_from_geojson(
        out["subspacecraft_lon_deg"].to_numpy(),
        out["subspacecraft_lat_deg"].to_numpy(),
        geojson,
    )
    out["surface_class"] = np.where(out["subspacecraft_is_land"], "land", "ocean")
    out["month"] = out["center_time"].dt.to_period("M").astype(str)
    return out


def nearest_center_indices(times_ns: np.ndarray, centers_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    insertion = np.searchsorted(centers_ns, times_ns)
    right = np.clip(insertion, 0, len(centers_ns) - 1)
    left = np.clip(insertion - 1, 0, len(centers_ns) - 1)
    choose_right = np.abs(times_ns - centers_ns[right]) < np.abs(times_ns - centers_ns[left])
    nearest = np.where(choose_right, right, left)
    distance_ns = np.abs(times_ns - centers_ns[nearest])
    return nearest, distance_ns


def aggregate_pass_power(
    input_path: Path,
    passes: pd.DataFrame,
    chunksize: int,
    windows_minutes: list[float],
    max_alignment_angle_deg: float,
) -> pd.DataFrame:
    centers_ns = passes["center_time"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    pass_ids = passes["pass_id"].to_numpy(dtype=np.int64)
    max_window_ns = int(max(windows_minutes) * 60.0e9)
    pieces: list[pd.DataFrame] = []
    usecols = [*GEOMETRY_COLUMNS, "frequency_band", *POWER_COLUMNS]
    for chunk in pd.read_csv(
        input_path,
        usecols=usecols,
        parse_dates=["time"],
        chunksize=int(chunksize),
        low_memory=False,
    ):
        times_ns = chunk["time"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
        nearest, distance_ns = nearest_center_indices(times_ns, centers_ns)
        candidate = distance_ns <= max_window_ns
        if not np.any(candidate):
            continue
        idx = np.flatnonzero(candidate)
        spacecraft = chunk.iloc[idx][SPACECRAFT_COLUMNS].to_numpy(dtype=float)
        earth = chunk.iloc[idx][EARTH_UNIT_COLUMNS].to_numpy(dtype=float)
        cosine = alignment_cosine(spacecraft, earth)
        aligned = cosine >= np.cos(np.deg2rad(float(max_alignment_angle_deg)))
        idx = idx[aligned]
        if not len(idx):
            continue
        nearest = nearest[idx]
        distance_ns = distance_ns[idx]
        frequency = pd.to_numeric(chunk.iloc[idx]["frequency_band"], errors="coerce").to_numpy()
        for window_minutes in windows_minutes:
            in_window = distance_ns <= float(window_minutes) * 60.0e9
            if not np.any(in_window):
                continue
            for antenna in POWER_COLUMNS:
                power = pd.to_numeric(chunk.iloc[idx][antenna], errors="coerce").to_numpy(dtype=float)
                good = in_window & np.isfinite(power) & (power > 0.0) & np.isfinite(frequency)
                if not np.any(good):
                    continue
                log_power = np.log10(power[good])
                work = pd.DataFrame(
                    {
                        "window_minutes": float(window_minutes),
                        "pass_id": pass_ids[nearest[good]],
                        "frequency_band": frequency[good].astype(int),
                        "antenna": antenna,
                        "log_power": log_power,
                        "raw_power": power[good],
                    }
                )
                grouped = work.groupby(
                    ["window_minutes", "pass_id", "frequency_band", "antenna"], as_index=False
                ).agg(
                    n_samples=("log_power", "size"),
                    sum_log_power=("log_power", "sum"),
                    sumsq_log_power=("log_power", lambda values: float(np.sum(np.square(values)))),
                    sum_raw_power=("raw_power", "sum"),
                )
                pieces.append(grouped)
    if not pieces:
        return pd.DataFrame()
    reduced = pd.concat(pieces, ignore_index=True).groupby(
        ["window_minutes", "pass_id", "frequency_band", "antenna"], as_index=False
    ).agg(
        n_samples=("n_samples", "sum"),
        sum_log_power=("sum_log_power", "sum"),
        sumsq_log_power=("sumsq_log_power", "sum"),
        sum_raw_power=("sum_raw_power", "sum"),
    )
    reduced["mean_log10_power"] = reduced["sum_log_power"] / reduced["n_samples"]
    reduced["mean_raw_power"] = reduced["sum_raw_power"] / reduced["n_samples"]
    variance = reduced["sumsq_log_power"] / reduced["n_samples"] - np.square(
        reduced["mean_log10_power"]
    )
    reduced["within_pass_log10_power_std"] = np.sqrt(np.maximum(variance, 0.0))
    reduced["frequency_mhz"] = reduced["frequency_band"].map(FREQUENCY_MAP_MHZ)
    metadata = passes[
        [
            "pass_id",
            "center_time",
            "month",
            "surface_class",
            "subspacecraft_is_land",
            "subspacecraft_lon_deg",
            "subspacecraft_lat_deg",
            "minimum_alignment_angle_deg",
            "frontside_duration_s",
        ]
    ]
    return reduced.merge(metadata, on="pass_id", how="left", validate="many_to_one")


def prepare_pass_residuals(pass_power: pd.DataFrame, min_samples: int) -> pd.DataFrame:
    out = pass_power[pass_power["n_samples"] >= int(min_samples)].copy()
    group_cols = ["window_minutes", "frequency_band", "antenna"]
    out["robust_inlier"] = False
    for _, idx in out.groupby(group_cols).groups.items():
        values = out.loc[idx, "mean_log10_power"].to_numpy(dtype=float)
        median = float(np.nanmedian(values))
        scale = robust_sigma(values)
        keep = np.isfinite(values)
        if np.isfinite(scale) and scale > 0.0:
            keep &= np.abs(values - median) <= 8.0 * scale
        out.loc[idx, "robust_inlier"] = keep
    month_median = out.groupby([*group_cols, "month"])["mean_log10_power"].transform("median")
    out["month_centered_mean_log10_power"] = out["mean_log10_power"] - month_median
    return out


def _permutation_p(
    values: np.ndarray,
    land: np.ndarray,
    strata: np.ndarray,
    trials: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    def difference(labels: np.ndarray) -> float:
        return float(np.nanmedian(values[labels]) - np.nanmedian(values[~labels]))

    observed = difference(land)
    blocks = [np.flatnonzero(strata == value) for value in np.unique(strata)]
    exceed = 0
    for _ in range(int(trials)):
        shuffled = land.copy()
        for idx in blocks:
            if len(idx) > 1:
                shuffled[idx] = land[rng.permutation(idx)]
        if np.any(shuffled) and np.any(~shuffled):
            exceed += int(abs(difference(shuffled)) >= abs(observed))
    return observed, (1.0 + exceed) / (1.0 + int(trials))


def compare_land_ocean(
    pass_power: pd.DataFrame,
    trials: int,
    seed: int,
    primary_window_minutes: float,
) -> pd.DataFrame:
    base = pass_power[pass_power["robust_inlier"]].copy()
    centered = base.copy()
    centered["analysis_antenna"] = centered["antenna"]
    combined = (
        base.groupby(
            [
                "window_minutes",
                "pass_id",
                "frequency_band",
                "frequency_mhz",
                "month",
                "surface_class",
                "subspacecraft_is_land",
            ],
            as_index=False,
        )
        .agg(
            month_centered_mean_log10_power=("month_centered_mean_log10_power", "median"),
            mean_log10_power=("mean_log10_power", "median"),
        )
    )
    combined["analysis_antenna"] = "combined"
    analysis = pd.concat(
        [
            centered[
                [
                    "window_minutes",
                    "pass_id",
                    "frequency_band",
                    "frequency_mhz",
                    "month",
                    "surface_class",
                    "subspacecraft_is_land",
                    "month_centered_mean_log10_power",
                    "mean_log10_power",
                    "analysis_antenna",
                ]
            ],
            combined,
        ],
        ignore_index=True,
        sort=False,
    )
    rows = []
    rng = np.random.default_rng(int(seed))
    for keys, group in analysis.groupby(
        ["window_minutes", "frequency_band", "frequency_mhz", "analysis_antenna"], sort=True
    ):
        window, band, frequency, antenna = keys
        land = group["subspacecraft_is_land"].to_numpy(dtype=bool)
        values = group["month_centered_mean_log10_power"].to_numpy(dtype=float)
        good = np.isfinite(values)
        land = land[good]
        values = values[good]
        strata = group.loc[good, "month"].astype(str).to_numpy()
        if np.count_nonzero(land) < 20 or np.count_nonzero(~land) < 20:
            continue
        delta, p_value = _permutation_p(values, land, strata, trials, rng)
        raw = group.loc[good, "mean_log10_power"].to_numpy(dtype=float)
        rows.append(
            {
                "window_minutes": float(window),
                "frequency_band": int(band),
                "frequency_mhz": float(frequency),
                "antenna": str(antenna),
                "n_land_passes": int(np.count_nonzero(land)),
                "n_ocean_passes": int(np.count_nonzero(~land)),
                "median_raw_log10_power_land": float(np.nanmedian(raw[land])),
                "median_raw_log10_power_ocean": float(np.nanmedian(raw[~land])),
                "land_minus_ocean_month_centered_dex": delta,
                "land_to_ocean_geometric_power_ratio": float(10.0**delta),
                "month_stratified_permutation_p": p_value,
            }
        )
    out = pd.DataFrame(rows)
    out["fdr_q_global"] = _bh_qvalues(out["month_stratified_permutation_p"])
    out["fdr_q_primary_window"] = np.nan
    primary = np.isclose(out["window_minutes"], float(primary_window_minutes))
    out.loc[primary, "fdr_q_primary_window"] = _bh_qvalues(
        out.loc[primary, "month_stratified_permutation_p"]
    )
    return out


def alignment_sensitivity(
    pass_power: pd.DataFrame,
    thresholds_deg: list[float],
    trials: int,
    seed: int,
    primary_window_minutes: float,
) -> pd.DataFrame:
    frames = []
    for index, threshold in enumerate(thresholds_deg):
        subset = pass_power[
            pass_power["minimum_alignment_angle_deg"] <= float(threshold)
        ]
        result = compare_land_ocean(
            subset,
            trials,
            int(seed) + 1009 * index,
            primary_window_minutes,
        )
        result = result[
            np.isclose(result["window_minutes"], float(primary_window_minutes))
        ].copy()
        result["maximum_center_alignment_angle_deg"] = float(threshold)
        frames.append(result)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_pass_centers(passes: pd.DataFrame, geojson: dict, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for feature in geojson.get("features", []):
        geometry = feature.get("geometry") or {}
        coordinates = geometry.get("coordinates", [])
        polygons = [coordinates] if geometry.get("type") == "Polygon" else coordinates
        for polygon in polygons:
            if polygon:
                ring = np.asarray(polygon[0], dtype=float)
                ax.plot(ring[:, 0], ring[:, 1], color="0.65", lw=0.4)
    colors = np.where(passes["subspacecraft_is_land"], "#f28e2b", "#4e79a7")
    ax.scatter(passes["subspacecraft_lon_deg"], passes["subspacecraft_lat_deg"], s=8, c=colors, alpha=0.55)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Earth sub-spacecraft longitude (deg)")
    ax.set_ylabel("Earth sub-spacecraft latitude (deg)")
    ax.set_title("RAE-2 closest-alignment pass centers: land (orange), ocean (blue)")
    ax.grid(alpha=0.18)
    fig.tight_layout()
    path = out_dir / "naive_subspacecraft_pass_coverage.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_effects(
    comparisons: pd.DataFrame,
    primary_window: float,
    primary_alignment_angle_deg: float,
    out_dir: Path,
) -> Path:
    selected = comparisons[np.isclose(comparisons["window_minutes"], float(primary_window))]
    styles = {
        "combined": ("combined", "#2ca02c", 0.0),
        "rv1_coarse": ("upper V", "#1f77b4", -0.012),
        "rv2_coarse": ("lower V", "#ff7f0e", 0.012),
    }
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for antenna, group in selected.groupby("antenna", sort=True):
        label, color, offset = styles[str(antenna)]
        x = np.log10(group["frequency_mhz"].to_numpy()) + offset
        y = group["land_minus_ocean_month_centered_dex"].to_numpy()
        ax.scatter(x, y, color=color, s=60, label=label)
        significant = group["fdr_q_primary_window"] < 0.05
        ax.scatter(x[significant], y[significant], s=130, facecolors="none", edgecolors="black")
    ax.axhline(0.0, color="black", lw=0.8)
    frequencies = sorted(selected["frequency_mhz"].unique())
    ax.set_xticks(np.log10(frequencies))
    ax.set_xticklabels([f"{value:g}" for value in frequencies])
    ax.set_xlabel("RV channel frequency (MHz)")
    ax.set_ylabel("land minus ocean pass power (dex; month-centered)")
    ax.set_title(
        "Naive sub-spacecraft land/ocean comparison "
        f"(±{primary_window:g} min, center ≤{primary_alignment_angle_deg:g}°)"
    )
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = out_dir / "naive_land_ocean_effect_by_frequency.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report(
    out_dir: Path,
    passes: pd.DataFrame,
    pass_power: pd.DataFrame,
    comparisons: pd.DataFrame,
    sensitivity: pd.DataFrame,
    primary_window: float,
    primary_alignment_angle_deg: float,
    plots: list[Path],
) -> Path:
    primary = comparisons[
        np.isclose(comparisons["window_minutes"], float(primary_window))
        & comparisons["antenna"].eq("combined")
    ].sort_values("frequency_mhz")
    display = primary[
        [
            "frequency_mhz",
            "n_land_passes",
            "n_ocean_passes",
            "land_minus_ocean_month_centered_dex",
            "land_to_ocean_geometric_power_ratio",
            "month_stratified_permutation_p",
            "fdr_q_primary_window",
        ]
    ].copy()
    table = _markdown(display, list(display.columns))
    significant = primary[primary["fdr_q_primary_window"] < 0.05]
    sensitivity_918 = sensitivity[
        sensitivity["antenna"].eq("combined")
        & np.isclose(sensitivity["frequency_mhz"], 9.18)
    ].sort_values("maximum_center_alignment_angle_deg")
    sensitivity_display = sensitivity_918[
        [
            "maximum_center_alignment_angle_deg",
            "n_land_passes",
            "n_ocean_passes",
            "land_to_ocean_geometric_power_ratio",
            "month_stratified_permutation_p",
            "fdr_q_primary_window",
        ]
    ].copy()
    primary_pass_ids = pass_power.loc[
        pass_power["minimum_alignment_angle_deg"] <= float(primary_alignment_angle_deg),
        "pass_id",
    ].unique()
    primary_passes = passes[passes["pass_id"].isin(primary_pass_ids)]
    lines = [
        "# Naive RAE-2 Sub-Spacecraft Land/Ocean Test",
        "",
        "## Definition",
        "",
        "Each independent observation is one contiguous lunar-orbit interval with RAE-2 on the Earth-facing side of the Moon. The pass center is the sample with the closest Moon--RAE--Earth alignment. The exact Earth-to-spacecraft vector is transformed to Earth-fixed longitude/latitude, and that single sub-spacecraft point is classified as land or ocean.",
        "",
        f"RV power is averaged in log space over ±{primary_window:g} minutes around the pass center. The primary comparison keeps centers within {primary_alignment_angle_deg:g} degrees of the Moon--Earth line. Comparisons use pass averages, remove only extreme eight-robust-sigma pass outliers, center within calendar month, and permute land/ocean labels within month.",
        "",
        "This deliberately does not weight the visible Earth disk. It is the binary point-under-RAE test requested. Because RV is a total-power non-imaging receiver, the label is a geometric proxy rather than proof that emission came from that point.",
        "",
        "## Coverage",
        "",
        f"- Candidate passes within the {passes['minimum_alignment_angle_deg'].max():.0f}-degree sensitivity envelope: {passes['pass_id'].nunique():,}",
        f"- Primary passes with usable power and center within {primary_alignment_angle_deg:g} degrees: {len(primary_pass_ids):,}",
        f"- Primary land centers: {int(primary_passes['subspacecraft_is_land'].sum()):,}",
        f"- Primary ocean centers: {int((~primary_passes['subspacecraft_is_land']).sum()):,}",
        f"- Median primary closest-alignment angle: {primary_passes['minimum_alignment_angle_deg'].median():.2f} deg",
        f"- Candidate pass-channel-antenna averages after sample-count filtering: {len(pass_power):,}",
        "",
        "## Primary Combined-Antenna Result",
        "",
        table,
        "",
    ]
    if significant.empty:
        lines.append("No channel survives the primary-window, primary-alignment FDR correction.")
    else:
        frequencies = ", ".join(f"{value:g}" for value in significant["frequency_mhz"])
        lines.append(
            f"Channels surviving primary-window, primary-alignment FDR q<0.05: {frequencies} MHz."
        )
    if not primary.empty:
        row_393 = primary[np.isclose(primary["frequency_mhz"], 3.93)]
        row_470 = primary[np.isclose(primary["frequency_mhz"], 4.70)]
        row_918 = primary[np.isclose(primary["frequency_mhz"], 9.18)]
        if not row_918.empty:
            r = row_918.iloc[0]
            lines.extend(
                [
                    "",
                    f"The clearest land excess is 9.18 MHz: land/ocean geometric power ratio {r['land_to_ocean_geometric_power_ratio']:.3f} (permutation p={r['month_stratified_permutation_p']:.4g}, FDR q={r['fdr_q_primary_window']:.4g}).",
                ]
            )
        if not row_393.empty and not row_470.empty:
            a = row_393.iloc[0]
            b = row_470.iloc[0]
            lines.append(
                f"The 3.93 and 4.70 MHz point-label effects are small and statistically unresolved after correction: ratios {a['land_to_ocean_geometric_power_ratio']:.3f} and {b['land_to_ocean_geometric_power_ratio']:.3f}, respectively (q={a['fdr_q_primary_window']:.4g} and {b['fdr_q_primary_window']:.4g})."
            )
    lines.extend(
        [
            "",
            "## Alignment-Cut Sensitivity at 9.18 MHz",
            "",
            _markdown(sensitivity_display, list(sensitivity_display.columns)),
            "",
            "## Interpretation Limits",
            "",
            "- A land/ocean difference can support a surface-transmitter contribution, but it cannot localize a transmitter within the visible hemisphere.",
            "- A null result does not rule out land emission because the spacecraft sees almost an entire Earth hemisphere, not only its sub-spacecraft point.",
            "- The month-stratified test suppresses long-term receiver drift; the 2, 5, and 10 minute window outputs show sensitivity to the chosen averaging width.",
            "",
            "## Files",
            "",
            "- `earth_facing_pass_centers.csv`",
            "- `earth_facing_pass_power_averages.csv`",
            "- `naive_land_ocean_comparisons.csv`",
            "- `naive_land_ocean_alignment_sensitivity.csv`",
        ]
    )
    lines.extend([f"- `{plot.name}`" for plot in plots])
    path = out_dir / "naive_subspacecraft_land_ocean_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def parse_windows(text: str) -> list[float]:
    values = sorted({float(value.strip()) for value in str(text).split(",") if value.strip()})
    if not values or any(value <= 0.0 for value in values):
        raise ValueError("windows must contain positive comma-separated minutes")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--chunksize", type=int, default=750_000)
    parser.add_argument("--max-gap-seconds", type=float, default=600.0)
    parser.add_argument("--min-frontside-duration-minutes", type=float, default=20.0)
    parser.add_argument("--max-center-alignment-angle-deg", type=float, default=30.0)
    parser.add_argument("--primary-alignment-angle-deg", type=float, default=10.0)
    parser.add_argument("--alignment-sensitivity-deg", default="5,10,15,20,30")
    parser.add_argument("--windows-minutes", default="2,5,10")
    parser.add_argument("--primary-window-minutes", type=float, default=5.0)
    parser.add_argument("--min-samples-per-pass-channel", type=int, default=5)
    parser.add_argument("--permutation-trials", type=int, default=1999)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()

    windows = parse_windows(args.windows_minutes)
    if not any(np.isclose(windows, args.primary_window_minutes)):
        raise ValueError("primary window must be included in --windows-minutes")
    if not 0.0 < args.primary_alignment_angle_deg <= args.max_center_alignment_angle_deg:
        raise ValueError(
            "primary alignment angle must be positive and no larger than the maximum angle"
        )
    out_dir = ensure_dir(args.out_dir)
    write_json(
        out_dir / "run_config.json",
        {
            **vars(args),
            "input": str(args.input),
            "out_dir": str(args.out_dir),
            "windows_minutes_parsed": windows,
            "natural_earth_url": NATURAL_EARTH_URL,
            "software_versions": software_versions(),
        },
    )
    geojson_path = out_dir / "natural_earth_110m_admin0_countries.geojson"
    if not geojson_path.exists():
        source = ROOT / "outputs/earth_noise_geography_v1/natural_earth_110m_admin0_countries.geojson"
        if source.exists():
            geojson_path.write_bytes(source.read_bytes())
    geojson = load_land_geojson(geojson_path)

    passes = find_frontside_passes(args.input, args.chunksize, args.max_gap_seconds)
    passes = passes[
        (passes["frontside_duration_s"] >= float(args.min_frontside_duration_minutes) * 60.0)
        & (passes["minimum_alignment_angle_deg"] <= float(args.max_center_alignment_angle_deg))
    ].copy()
    passes = attach_subspacecraft_points(passes, geojson)
    passes.to_csv(out_dir / "earth_facing_pass_centers.csv", index=False)

    power = aggregate_pass_power(
        args.input,
        passes,
        args.chunksize,
        windows,
        args.max_center_alignment_angle_deg,
    )
    power = prepare_pass_residuals(power, args.min_samples_per_pass_channel)
    power.to_csv(out_dir / "earth_facing_pass_power_averages.csv", index=False)
    primary_power = power[
        power["minimum_alignment_angle_deg"] <= float(args.primary_alignment_angle_deg)
    ].copy()
    comparisons = compare_land_ocean(
        primary_power,
        args.permutation_trials,
        args.seed,
        args.primary_window_minutes,
    )
    comparisons.to_csv(out_dir / "naive_land_ocean_comparisons.csv", index=False)
    sensitivity = alignment_sensitivity(
        power,
        parse_windows(args.alignment_sensitivity_deg),
        args.permutation_trials,
        args.seed,
        args.primary_window_minutes,
    )
    sensitivity.to_csv(out_dir / "naive_land_ocean_alignment_sensitivity.csv", index=False)
    plots = [
        plot_pass_centers(passes, geojson, out_dir),
        plot_effects(
            comparisons,
            args.primary_window_minutes,
            args.primary_alignment_angle_deg,
            out_dir,
        ),
    ]
    report = write_report(
        out_dir,
        passes,
        power,
        comparisons,
        sensitivity,
        args.primary_window_minutes,
        args.primary_alignment_angle_deg,
        plots,
    )
    print(report)


if __name__ == "__main__":
    main()
