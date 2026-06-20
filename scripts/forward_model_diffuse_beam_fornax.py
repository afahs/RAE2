#!/usr/bin/env python
"""Approximate diffuse-sky + digitized-beam forward model for Fornax A events.

This is a diagnostic, not a final antenna simulation.  The digitized antenna
files available here are 1D E/H-plane cuts, not a full 2D polarized beam.  The
model therefore uses those cuts as an axisymmetric gain approximation around
the lower/upper-V boresight.

The purpose is to test a specific claim: can a broad, structured diffuse
Galactic background passing through a sidelobe/beam gradient produce linear
pre/post trends that remain visible even after shifting Fornax event times by
600 s?
"""

from __future__ import annotations

from dataclasses import dataclass
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


CLEAN = ROOT / "outputs/control_survey_earth_sun_postnov1974_v2/01_ingest/cleaned_timeseries.csv"
EVENTS = ROOT / "outputs/control_survey_bright_sources_postnov1974_v1/02_events/predicted_events.csv"
OBS_TRUE = ROOT / "outputs/all_frequency_profile_grids_v1/fornax_a_all_frequency_profile_summary_900s.csv"
OBS_SHIFT = ROOT / "outputs/all_frequency_profile_grids_fornax_a_shift_plus600s_v1/fornax_a_all_frequency_profile_summary_900s.csv"
TAU_EVENTS = ROOT / "outputs/custom_fixed_source_profile_grids_tau_a_v1/custom_fixed_source_predicted_events.csv"
TAU_OBS_TRUE = ROOT / "outputs/custom_fixed_source_profile_grids_tau_a_v1/tau_a_all_frequency_profile_summary_900s.csv"
EARTH_EVENTS = ROOT / "outputs/planetary_confirmation_survey_science_baseline_v2/events/all_planet_predicted_events.csv"
EARTH_OBS_TRUE = ROOT / "outputs/all_frequency_profile_grids_v1/earth_all_frequency_profile_summary_900s.csv"
EARTH_OBS_SHIFT = ROOT / "outputs/all_frequency_profile_grids_earth_shift_plus600s_v1/earth_all_frequency_profile_summary_900s.csv"
BEAM_DIR = Path(os.environ.get("RAE2_ANTENNA_DIGITIZATION_DIR", "data/antennaDigitization"))
SKY_DIR = Path(os.environ.get("RAE2_SKY_MAP_DIR", "data/pysmMaps"))
OUT = ROOT / "outputs/fornax_diffuse_beam_forward_model_v1"

SOURCE = "fornax_a"
WINDOW_S = 900.0
BIN_S = 60.0
INNER_S = 15.0
TIME_SHIFTS_S = [0.0, 600.0]
ANTENNAS = ["rv1_coarse", "rv2_coarse"]
FREQUENCY_BANDS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
MAX_EVENTS_PER_GROUP = 80
MODEL_NSIDE = 8
EXPECTED_SIGN = {"disappearance": -1.0, "reappearance": 1.0}
ANT_LABEL = {"rv1_coarse": "upper V", "rv2_coarse": "lower V"}
ANT_COLOR = {"rv1_coarse": "#4c78a8", "rv2_coarse": "#d95f02"}


def _source_label(source: str) -> str:
    return {"fornax_a": "Fornax A", "earth": "Earth", "tau_a": "Tau A"}.get(source, source)


def _source_paths(source: str, out_dir: str | None) -> tuple[Path, Path, Path, Path]:
    if source == "earth":
        default_out = ROOT / "outputs/earth_diffuse_beam_forward_model_v1"
        return EARTH_EVENTS, EARTH_OBS_TRUE, EARTH_OBS_SHIFT, Path(out_dir) if out_dir else default_out
    if source == "fornax_a":
        default_out = ROOT / "outputs/fornax_diffuse_beam_forward_model_v1"
        return EVENTS, OBS_TRUE, OBS_SHIFT, Path(out_dir) if out_dir else default_out
    if source == "tau_a":
        default_out = ROOT / "outputs/tau_a_diffuse_beam_forward_model_v1"
        return TAU_EVENTS, TAU_OBS_TRUE, Path("__missing_tau_a_shift_observed__.csv"), Path(out_dir) if out_dir else default_out
    raise ValueError(f"unsupported source for this diagnostic: {source}")


@dataclass(frozen=True)
class BeamSpec:
    model_frequency_mhz: float
    eplane_path: Path
    hplane_path: Path


BEAM_SPECS = [
    BeamSpec(1.31, BEAM_DIR / "eplane_1p31MHz.csv", BEAM_DIR / "hplane_1p31MHz.csv"),
    BeamSpec(3.93, BEAM_DIR / "eplane_3p93MHz.csv", BEAM_DIR / "hplane_3p93MHz.csv"),
    BeamSpec(6.55, BEAM_DIR / "eplane_6p55MHz.csv", BEAM_DIR / "hplane_6p55MHz.csv"),
]


def _read(path: Path, **kwargs) -> pd.DataFrame:
    return read_table(path, low_memory=False, **kwargs)


def _nearest_beam(freq_mhz: float) -> BeamSpec:
    return min(BEAM_SPECS, key=lambda spec: abs(spec.model_frequency_mhz - float(freq_mhz)))


def _nearest_sky_path(freq_mhz: float) -> Path:
    mhz = int(np.clip(round(float(freq_mhz)), 1, 50))
    return SKY_DIR / f"synch_pysm_s1_{mhz:02d}MHz_IQU.fits"


def _load_sky_i(freq_mhz: float) -> tuple[np.ndarray, Path]:
    path = _nearest_sky_path(freq_mhz)
    with fits.open(path, memmap=True) as hdul:
        data = hdul[1].data
        # PySM IQU table: 12 rows of 1024 pixels for nside 32, RING ordering.
        arr = np.asarray(data.field(0), dtype=np.float64).reshape(-1)
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


def _load_beam(spec: BeamSpec) -> tuple[np.ndarray, np.ndarray]:
    e = _read(spec.eplane_path)
    h = _read(spec.hplane_path)
    angles = e["angle_deg"].to_numpy(dtype=float)
    gain_e = 10.0 ** (e["gain_dB"].to_numpy(dtype=float) / 10.0)
    gain_h = 10.0 ** (h["gain_dB"].to_numpy(dtype=float) / 10.0)
    # Mean E/H cut in linear power.  This preserves sidelobe structure while
    # avoiding a false precision from choosing a single unknown spin azimuth.
    gain = 0.5 * (gain_e + gain_h)
    return angles, gain


def _interp_cyclic(angles: np.ndarray, values: np.ndarray, angle_deg: np.ndarray) -> np.ndarray:
    a = np.asarray(angle_deg, dtype=float) % 360.0
    x = np.concatenate(([angles[-1] - 360.0], angles, [angles[0] + 360.0]))
    y = np.concatenate(([values[-1]], values, [values[0]]))
    return np.interp(a, x, y)


def _beam_gain_for_axis(axis: np.ndarray, pixel_vecs: np.ndarray, angles: np.ndarray, gains: np.ndarray) -> np.ndarray:
    dots = np.clip(pixel_vecs @ axis, -1.0, 1.0)
    sep = np.degrees(np.arccos(dots))
    return _interp_cyclic(angles, gains, sep)


def _beam_weighted_sky_for_axes(
    axes: np.ndarray,
    sky_i: np.ndarray,
    pixel_vecs: np.ndarray,
    angles: np.ndarray,
    gains: np.ndarray,
    chunk_size: int = 128,
) -> np.ndarray:
    axes = np.asarray(axes, dtype=float)
    out = np.full(len(axes), np.nan, dtype=float)
    good = np.isfinite(axes).all(axis=1)
    for start in range(0, len(axes), chunk_size):
        stop = min(start + chunk_size, len(axes))
        idx = np.arange(start, stop)
        valid = good[start:stop]
        if not np.any(valid):
            continue
        block_axes = axes[start:stop][valid]
        dots = np.clip(block_axes @ pixel_vecs.T, -1.0, 1.0)
        sep = np.degrees(np.arccos(dots))
        gain = _interp_cyclic(angles, gains, sep)
        denom = np.nansum(gain, axis=1)
        numer = np.nansum(gain * sky_i.reshape(1, -1), axis=1)
        vals = np.divide(numer, denom, out=np.full_like(numer, np.nan), where=denom > 0)
        out[idx[valid]] = vals
    return out


def _moon_axis_from_position(pos_xyz: np.ndarray, antenna: str) -> np.ndarray:
    lower = normalize_vectors(-np.asarray(pos_xyz, dtype=float))[0]
    if antenna == "rv2_coarse":
        return lower
    if antenna == "rv1_coarse":
        return -lower
    raise ValueError(f"unsupported antenna {antenna}")


def _groups(clean: pd.DataFrame) -> dict[tuple[int, str], tuple[pd.DataFrame, np.ndarray]]:
    out = {}
    for key, grp in clean.groupby(["frequency_band", "antenna"], sort=True):
        g = grp.sort_values("time").reset_index(drop=True)
        out[(int(key[0]), str(key[1]))] = (g, datetime_ns(g["time"]))
    return out


def _event_window(group: pd.DataFrame, group_ns: np.ndarray, event_time: pd.Timestamp) -> pd.DataFrame:
    event_ns = pd.Timestamp(event_time).value
    half_ns = int(WINDOW_S * 1e9)
    lo = int(np.searchsorted(group_ns, event_ns - half_ns, side="left"))
    hi = int(np.searchsorted(group_ns, event_ns + half_ns, side="right"))
    if hi <= lo:
        return pd.DataFrame()
    local = group.iloc[lo:hi].copy()
    local["t_rel_sec"] = (datetime_ns(local["time"]) - event_ns).astype(float) / 1e9
    return local[np.abs(local["t_rel_sec"]) <= WINDOW_S].copy()


def _normalize_event(t: np.ndarray, y: np.ndarray) -> np.ndarray | None:
    side = np.abs(t) >= INNER_S
    if np.count_nonzero(side) < 6:
        return None
    center = float(np.nanmedian(y[side]))
    scale = robust_sigma(y[side] - center)
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.nanstd(y[side]))
    if not np.isfinite(scale) or scale <= 0:
        return None
    return (y - center) / scale


def _model_power_for_rows(
    local: pd.DataFrame,
    antenna: str,
    sky_i: np.ndarray,
    pixel_vecs: np.ndarray,
    beam_angles: np.ndarray,
    beam_gains: np.ndarray,
) -> np.ndarray:
    pos = local[["position_x", "position_y", "position_z"]].to_numpy(dtype=float)
    lower = normalize_vectors(-pos)
    axes = lower if antenna == "rv2_coarse" else -lower
    return _beam_weighted_sky_for_axes(axes, sky_i, pixel_vecs, beam_angles, beam_gains)


def build_model_profiles(clean: pd.DataFrame, events: pd.DataFrame, source: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = _groups(clean)
    point_rows = []
    model_meta = []
    bins = np.arange(-WINDOW_S, WINDOW_S + BIN_S, BIN_S)
    for band in FREQUENCY_BANDS:
        band_events = events[events["frequency_band"].eq(band)].copy()
        if band_events.empty:
            continue
        freq = float(band_events["frequency_mhz"].iloc[0])
        sky_i, sky_path = _load_sky_i(freq)
        nside = hp.npix2nside(len(sky_i))
        pixel_vecs = _pixel_fk4_vectors(nside)
        spec = _nearest_beam(freq)
        beam_angles, beam_gains = _load_beam(spec)
        model_meta.append(
            {
                "frequency_band": band,
                "frequency_mhz": freq,
                "sky_map": str(sky_path),
                "beam_model_frequency_mhz": spec.model_frequency_mhz,
                "eplane_beam": str(spec.eplane_path),
                "hplane_beam": str(spec.hplane_path),
                "nside": nside,
            }
        )
        for antenna in ANTENNAS:
            payload = groups.get((band, antenna))
            if payload is None:
                continue
            group, group_ns = payload
            evs = band_events[band_events["antenna"].astype(str).eq(antenna)]
            if not evs.empty:
                evs = (
                    evs.sort_values("predicted_event_time")
                    .groupby("event_type", group_keys=False)
                    .head(MAX_EVENTS_PER_GROUP)
                )
            for shift_s in TIME_SHIFTS_S:
                for ev in evs.itertuples(index=False):
                    event_time = pd.Timestamp(ev.predicted_event_time) + pd.to_timedelta(float(shift_s), unit="s")
                    local = _event_window(group, group_ns, event_time)
                    if local.empty or len(local) < 8:
                        continue
                    t = local["t_rel_sec"].to_numpy(dtype=float)
                    y = _model_power_for_rows(local, antenna, sky_i, pixel_vecs, beam_angles, beam_gains)
                    z = _normalize_event(t, y)
                    if z is None:
                        continue
                    idx = np.digitize(t, bins) - 1
                    for bidx in sorted(set(idx)):
                        if bidx < 0 or bidx >= len(bins) - 1:
                            continue
                        mask = idx == bidx
                        if not np.any(mask):
                            continue
                        point_rows.append(
                            {
                                "source_name": source,
                                "event_id": ev.event_id,
                                "event_type": ev.event_type,
                                "frequency_band": band,
                                "frequency_mhz": freq,
                                "antenna": antenna,
                                "time_shift_s": float(shift_s),
                                "t_bin_sec": float(0.5 * (bins[bidx] + bins[bidx + 1])),
                                "model_z_power": float(np.nanmedian(z[mask])),
                                "n_samples": int(np.count_nonzero(mask)),
                            }
                        )
    return pd.DataFrame(point_rows), pd.DataFrame(model_meta)


def summarize(points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    by = ["source_name", "event_type", "frequency_band", "frequency_mhz", "antenna", "time_shift_s", "t_bin_sec"]
    for keys, grp in points.groupby(by, sort=True):
        vals = pd.to_numeric(grp["model_z_power"], errors="coerce").dropna()
        if vals.empty:
            continue
        err = robust_sigma(vals.to_numpy(dtype=float) - float(vals.median())) / np.sqrt(len(vals)) if len(vals) > 1 else np.nan
        rows.append(
            {
                **dict(zip(by, keys)),
                "median_model_z_power": float(vals.median()),
                "model_z_power_err": float(err) if np.isfinite(err) else np.nan,
                "n_events": int(grp["event_id"].nunique()),
                "n_points": int(len(grp)),
            }
        )
    return pd.DataFrame(rows)


def _prepost_contrast(summary: pd.DataFrame, value_col: str) -> pd.DataFrame:
    rows = []
    pre = (-180.0, -60.0)
    post = (60.0, 180.0)
    by = ["frequency_mhz", "antenna", "event_type", "time_shift_s"]
    for keys, grp in summary.groupby(by, sort=True):
        pre_vals = grp[(grp["t_bin_sec"] >= pre[0]) & (grp["t_bin_sec"] <= pre[1])][value_col]
        post_vals = grp[(grp["t_bin_sec"] >= post[0]) & (grp["t_bin_sec"] <= post[1])][value_col]
        if pre_vals.empty or post_vals.empty:
            continue
        event_type = str(keys[2])
        delta = float(np.nanmedian(post_vals) - np.nanmedian(pre_vals))
        rows.append(
            {
                **dict(zip(by, keys)),
                "post_minus_pre": delta,
                "source_like_contrast": float(EXPECTED_SIGN[event_type] * delta),
            }
        )
    return pd.DataFrame(rows)


def _load_observed_summary(path: Path, shift_s: float, source: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = _read(path)
    df = df[df["source_name"].astype(str).str.lower().eq(source)].copy()
    df["time_shift_s"] = float(shift_s)
    return df.rename(columns={"median_z_power": "observed_z_power", "median_z_power_err": "observed_z_power_err"})


def plot_model_vs_observed(model_summary: pd.DataFrame, observed: pd.DataFrame, source: str, out: Path) -> list[Path]:
    paths = []
    for shift_s in TIME_SHIFTS_S:
        fig, axes = plt.subplots(len(FREQUENCY_BANDS), 2, figsize=(12, 1.45 * len(FREQUENCY_BANDS) + 2), sharex=True)
        for i, band in enumerate(FREQUENCY_BANDS):
            freq_vals = model_summary[model_summary["frequency_band"].eq(band)]["frequency_mhz"].dropna().unique()
            freq_label = f"band {band}" if len(freq_vals) == 0 else f"{float(freq_vals[0]):.2f} MHz"
            for j, event_type in enumerate(["disappearance", "reappearance"]):
                ax = axes[i, j]
                for antenna in ANTENNAS:
                    m = model_summary[
                        model_summary["frequency_band"].eq(band)
                        & model_summary["event_type"].astype(str).eq(event_type)
                        & model_summary["antenna"].astype(str).eq(antenna)
                        & np.isclose(model_summary["time_shift_s"], shift_s)
                    ].sort_values("t_bin_sec")
                    if not m.empty:
                        ax.plot(
                            m["t_bin_sec"],
                            m["median_model_z_power"],
                            color=ANT_COLOR[antenna],
                            linewidth=1.7,
                            label=f"model {ANT_LABEL[antenna]}",
                        )
                    o = observed[
                        observed["frequency_band"].eq(band)
                        & observed["event_type"].astype(str).eq(event_type)
                        & observed["antenna"].astype(str).eq(antenna)
                        & np.isclose(observed["time_shift_s"], shift_s)
                    ].sort_values("t_bin_sec")
                    if not o.empty:
                        ax.errorbar(
                            o["t_bin_sec"],
                            o["observed_z_power"],
                            yerr=o.get("observed_z_power_err"),
                            color=ANT_COLOR[antenna],
                            marker="o",
                            linestyle="none",
                            markersize=2.3,
                            alpha=0.35,
                            elinewidth=0.5,
                            label=f"observed {ANT_LABEL[antenna]}",
                        )
                ax.axvline(0, color="black", linestyle="--", linewidth=0.7)
                ax.axhline(0, color="0.65", linewidth=0.7)
                ax.set_title(f"{freq_label} {event_type}", fontsize=9)
                if j == 0:
                    ax.set_ylabel("local normalized power")
                if i == len(FREQUENCY_BANDS) - 1:
                    ax.set_xlabel("seconds from event center")
                if i == 0 and j == 1:
                    ax.legend(frameon=False, fontsize=7, ncol=2)
        fig.suptitle(
            f"{_source_label(source)} diffuse-sky beam forward model vs observed profiles, shift={shift_s:.0f} s\n"
            "Model uses PySM synchrotron map + axisymmetric E/H beam-cut approximation.",
            y=0.995,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.965])
        path = out / f"{source}_diffuse_beam_model_vs_observed_shift_{int(shift_s)}s.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_contrast_comparison(model_contrast: pd.DataFrame, obs_contrast: pd.DataFrame, source: str, out: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for ax, shift_s in zip(axes.ravel(), [0.0, 600.0, 0.0, 600.0]):
        event_type = "disappearance" if ax in axes[0] else "reappearance"
        sub_m = model_contrast[
            np.isclose(model_contrast["time_shift_s"], shift_s) & model_contrast["event_type"].astype(str).eq(event_type)
        ]
        sub_o = obs_contrast[
            np.isclose(obs_contrast["time_shift_s"], shift_s) & obs_contrast["event_type"].astype(str).eq(event_type)
        ]
        merged = sub_o.merge(
            sub_m,
            on=["frequency_mhz", "antenna", "event_type", "time_shift_s"],
            suffixes=("_observed", "_model"),
        )
        for antenna, grp in merged.groupby("antenna", sort=True):
            ax.scatter(
                grp["source_like_contrast_observed"],
                grp["source_like_contrast_model"],
                label=ANT_LABEL.get(antenna, antenna),
                color=ANT_COLOR.get(antenna),
                s=36,
                alpha=0.85,
            )
            for _, row in grp.iterrows():
                ax.text(
                    row["source_like_contrast_observed"],
                    row["source_like_contrast_model"],
                    f"{row['frequency_mhz']:.1f}",
                    fontsize=7,
                    alpha=0.75,
                )
        lim = 2.5
        ax.plot([-lim, lim], [-lim, lim], color="0.5", linestyle="--", linewidth=0.8)
        ax.axhline(0, color="0.75", linewidth=0.7)
        ax.axvline(0, color="0.75", linewidth=0.7)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_title(f"{event_type}, shift={shift_s:.0f} s")
        ax.set_xlabel("observed source-like contrast")
        ax.set_ylabel("model source-like contrast")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"{_source_label(source)} pre/post contrast: observed vs diffuse-sky beam model", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = out / f"{source}_diffuse_beam_observed_vs_model_contrast.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report(
    model_meta: pd.DataFrame,
    model_contrast: pd.DataFrame,
    obs_contrast: pd.DataFrame,
    paths: list[Path],
    source: str,
    out: Path,
) -> None:
    merged = obs_contrast.merge(
        model_contrast,
        on=["frequency_mhz", "antenna", "event_type", "time_shift_s"],
        suffixes=("_observed", "_model"),
    )
    if not merged.empty:
        corr = float(
            np.corrcoef(
                merged["source_like_contrast_observed"].to_numpy(dtype=float),
                merged["source_like_contrast_model"].to_numpy(dtype=float),
            )[0, 1]
        )
        sign_agreement = float(
            (
                np.sign(merged["source_like_contrast_observed"].to_numpy(dtype=float))
                == np.sign(merged["source_like_contrast_model"].to_numpy(dtype=float))
            ).mean()
        )
    else:
        corr = np.nan
        sign_agreement = np.nan
    low = merged[merged["frequency_mhz"].isin([0.70, 0.90, 1.31, 2.20])].copy()
    low_lower = low[low["antenna"].astype(str).eq("rv2_coarse")].copy()
    lower_sign_agreement = (
        float(
            (
                np.sign(low_lower["source_like_contrast_observed"].to_numpy(dtype=float))
                == np.sign(low_lower["source_like_contrast_model"].to_numpy(dtype=float))
            ).mean()
        )
        if not low_lower.empty
        else np.nan
    )
    label = _source_label(source)
    if source == "earth":
        bottom_line = [
            "Bottom line: the approximate model does **partly** reproduce the Earth lower-V low-frequency anti-template",
            "direction. That is the specific behavior we were trying to explain. But the global observed-vs-model comparison",
            "is still poor because upper V and several non-low-frequency rows disagree. This supports a diffuse-sky/beam-gradient",
            "contribution, not a complete forward model.",
            f"Lower-V low-frequency sign agreement alone: {lower_sign_agreement:.3f}",
        ]
    else:
        bottom_line = [
            "Bottom line: this approximate diffuse-sky beam model produces slow event-aligned trends, including at +600 s shifted",
            f"event centers, but it does **not** reproduce the observed {label} signs or amplitudes well enough to claim that the",
            "current shifted profiles are explained by this model alone.",
            f"Lower-V low-frequency sign agreement alone: {lower_sign_agreement:.3f}",
        ]
    lines = [
        f"# {label} Diffuse-Sky Beam Forward Model",
        "",
        "## Model",
        "",
        f"This is a first-order forward model for the hypothesis that the straight/linear trends in the {label} profile grids",
        "are caused by broad diffuse Galactic background structure passing through the Ryle-Vonberg beam/sidelobes.",
        "",
        "Inputs:",
        "",
        f"- PySM synchrotron HEALPix maps from `{SKY_DIR}`;",
        f"- digitized Ryle-Vonberg E/H-plane beam cuts from `{BEAM_DIR}`;",
        "- spacecraft position and event times from the existing RyleVonberg cleaned/event tables;",
        "- lower V boresight is taken as Moon-facing; upper V is the opposite direction.",
        "",
        "Important limitation: the beam cuts are 1D. I used the mean E/H cut as an axisymmetric power beam.",
        "This can test whether the effect is plausible, but it cannot validate azimuthal sidelobe structure or spacecraft spin phase.",
        "",
        "## Beam / Sky Files Used",
        "",
        model_meta.to_string(index=False),
        "",
        "## Contrast Comparison",
        "",
        f"Observed-vs-model source-like contrast correlation: {corr:.3f}",
        f"Observed-vs-model sign agreement: {sign_agreement:.3f}",
        "",
        *bottom_line,
        "",
        "Low-frequency comparison rows:",
        "",
        low[
            [
                "frequency_mhz",
                "antenna",
                "event_type",
                "time_shift_s",
                "source_like_contrast_observed",
                "source_like_contrast_model",
            ]
        ].to_string(index=False)
        if not low.empty
        else "No low-frequency merged rows.",
        "",
        "## Interpretation",
        "",
        "The model reproduces the qualitative fact that a diffuse sky/beam-gradient term can survive event stacking and can remain",
        "visible at both true and +600 s shifted event centers. That supports the timescale plausibility of the sidelobe/background",
        "hypothesis.",
        "",
        (
            "For Earth, the useful result is the lower-V low-frequency sign match. The model predicts disappearance-up and "
            "reappearance-down behavior in those bands, which is exactly the anti-template morphology seen in the data. "
            "That makes the diffuse Galactic background plus Moon-facing beam gradient a physically credible contributor. "
            "The mismatch elsewhere means this is not yet a calibrated model."
            if source == "earth"
            else (
                "For Tau A, this comparison is best treated as a beam/diffuse-background morphology check. Tau A is close to the "
                "Galactic plane, so diffuse sky picked up by the broad Ryle-Vonberg beam is physically plausible. Agreement in broad "
                "slope or sign would support background contamination as part of the observed stack; disagreement means this 1D beam "
                "+ PySM model is missing important structure or that the observed profile is dominated by measured-data systematics."
            )
            if source == "tau_a"
            else (
                "However, the sign test fails for this first-order implementation. The model is not correlated with the observed pre/post "
                f"contrast signs, so the available 1D beam cuts plus a coarse synchrotron map are not sufficient to explain the {label} "
                "shifted profiles. The likely missing ingredients are the real 2D beam, spacecraft spin/azimuth phase, polarization, "
                "lunar blocking/emission, and receiver calibration."
            )
        ),
        "",
        "## Generated Plots",
        "",
    ]
    lines.extend(f"- `{path}`" for path in paths)
    (out / f"{source}_diffuse_beam_forward_model_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["fornax_a", "earth", "tau_a"], default=SOURCE)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    source = args.source
    events_path, obs_true_path, obs_shift_path, out = _source_paths(source, args.out_dir)
    ensure_dir(out)
    write_json(
        out / "run_config.json",
        {
            "source": source,
            "window_s": WINDOW_S,
            "bin_s": BIN_S,
            "inner_s": INNER_S,
            "time_shifts_s": TIME_SHIFTS_S,
            "antennas": ANTENNAS,
            "frequency_bands": FREQUENCY_BANDS,
            "max_events_per_frequency_antenna_event_type": MAX_EVENTS_PER_GROUP,
            "model_nside": MODEL_NSIDE,
            "beam_model": "axisymmetric mean of digitized E/H plane cuts",
            "software_versions": software_versions(),
        },
    )
    clean_cols = ["time", "frequency_band", "frequency_mhz", "antenna", "position_x", "position_y", "position_z"]
    clean = _read(CLEAN, usecols=clean_cols, parse_dates=["time"])
    clean = clean[clean["frequency_band"].isin(FREQUENCY_BANDS) & clean["antenna"].isin(ANTENNAS)].copy()
    events = _read(events_path, parse_dates=["predicted_event_time"])
    events = (
        events[events["source_name"].astype(str).str.lower().eq(source)]
        .loc[lambda df: df["frequency_band"].isin(FREQUENCY_BANDS) & df["antenna"].isin(ANTENNAS)]
        .copy()
    )
    points, model_meta = build_model_profiles(clean, events, source)
    model_summary = summarize(points)
    model_contrast = _prepost_contrast(model_summary.rename(columns={"median_model_z_power": "value"}), "value")
    model_contrast = model_contrast.rename(
        columns={"post_minus_pre": "post_minus_pre_model", "source_like_contrast": "source_like_contrast_model"}
    )

    obs = pd.concat(
        [_load_observed_summary(obs_true_path, 0.0, source), _load_observed_summary(obs_shift_path, 600.0, source)],
        ignore_index=True,
    )
    obs = obs[obs["frequency_band"].isin(FREQUENCY_BANDS) & obs["antenna"].isin(ANTENNAS)].copy()
    obs_contrast = _prepost_contrast(obs.rename(columns={"observed_z_power": "value"}), "value")
    obs_contrast = obs_contrast.rename(
        columns={"post_minus_pre": "post_minus_pre_observed", "source_like_contrast": "source_like_contrast_observed"}
    )

    points.to_csv(out / f"{source}_diffuse_beam_model_points.csv", index=False)
    model_summary.to_csv(out / f"{source}_diffuse_beam_model_summary.csv", index=False)
    model_contrast.to_csv(out / f"{source}_diffuse_beam_model_contrast.csv", index=False)
    obs_contrast.to_csv(out / f"{source}_observed_profile_contrast_for_model_comparison.csv", index=False)
    model_meta.to_csv(out / f"{source}_diffuse_beam_model_inputs.csv", index=False)

    paths = plot_model_vs_observed(model_summary, obs, source, out)
    paths.append(plot_contrast_comparison(model_contrast, obs_contrast, source, out))
    write_report(model_meta, model_contrast, obs_contrast, paths, source, out)
    print(out / f"{source}_diffuse_beam_forward_model_report.md")


if __name__ == "__main__":
    main()
