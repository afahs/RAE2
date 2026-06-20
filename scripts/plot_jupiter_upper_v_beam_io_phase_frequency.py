#!/usr/bin/env python
"""Upper-V Io phase/frequency maps after a Jupiter beam-response cut."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.constants import FREQUENCY_MAP_MHZ  # noqa: E402
from rylevonberg.frames import body_unit_vectors_from_moon  # noqa: E402
from rylevonberg.geometry import normalize_vectors  # noqa: E402
from rylevonberg.table_io import read_table  # noqa: E402
from rylevonberg.util import ensure_dir, software_versions, write_json  # noqa: E402


DEFAULT_CLEAN_NPY = ROOT / "outputs/control_survey_earth_sun_postnov1974_v2/01_ingest/cleaned_timeseries.npy"
DEFAULT_GEOMETRY = ROOT / "outputs/jupiter_phase_pattern_survey_v1/jupiter_spice_visibility_geometry_grid.csv"
DEFAULT_OUT = ROOT / "outputs/jupiter_upper_v_beam_gain10db_io_phase_frequency_v1"
UPPER_V = "rv1_coarse"
BEAM_DIR = Path(os.environ.get("RAE2_ANTENNA_DIGITIZATION_DIR", "data/antennaDigitization"))
BEAM_SPECS = [
    (1.31, BEAM_DIR / "eplane_1p31MHz.csv", BEAM_DIR / "hplane_1p31MHz.csv"),
    (3.93, BEAM_DIR / "eplane_3p93MHz.csv", BEAM_DIR / "hplane_3p93MHz.csv"),
    (6.55, BEAM_DIR / "eplane_6p55MHz.csv", BEAM_DIR / "hplane_6p55MHz.csv"),
]


def detect_ra_units(values: pd.Series | np.ndarray, mode: str) -> str:
    requested = str(mode).strip().lower()
    if requested in {"hours", "degrees"}:
        return requested
    vals = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size == 0:
        return "hours"
    return "hours" if np.nanmax(np.abs(vals)) <= 24.0 else "degrees"


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


def beam_half_width_deg_for_frequency(frequency_mhz: float) -> float:
    freq = float(frequency_mhz)
    if freq < 1.31:
        return 90.0
    if freq < 6.55:
        return 20.0
    return 10.0


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
    rel = values[good] / peak
    out[good] = 10.0 * np.log10(np.clip(rel, np.finfo(float).tiny, None))
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
    a = np.asarray(angle_deg, dtype=float) % 360.0
    x_pad = np.concatenate(([x[-1] - 360.0], x, [x[0] + 360.0]))
    y_pad = np.concatenate(([y[-1]], y, [y[0]]))
    return np.interp(a, x_pad, y_pad)


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


def attach_digitized_beam_gain(merged: pd.DataFrame) -> pd.DataFrame:
    out = merged.copy()
    out["beam_model_frequency_mhz"] = np.nan
    out["beam_model_eplane"] = ""
    out["beam_model_hplane"] = ""
    out["jupiter_beam_relative_gain_db"] = np.nan
    beam_cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    freq_values = out["frequency_mhz"].to_numpy(dtype=float)
    for freq in sorted(out["frequency_mhz"].dropna().unique()):
        beam_freq, eplane, hplane = nearest_beam_spec(float(freq))
        if beam_freq not in beam_cache:
            beam_cache[beam_freq] = load_relative_beam_model(eplane, hplane)
        angles, gain_db = beam_cache[beam_freq]
        mask = np.isclose(freq_values, float(freq))
        sep = out.loc[mask, "jupiter_beam_separation_deg"].to_numpy(dtype=float)
        out.loc[mask, "beam_model_frequency_mhz"] = float(beam_freq)
        out.loc[mask, "beam_model_eplane"] = str(eplane)
        out.loc[mask, "beam_model_hplane"] = str(hplane)
        out.loc[mask, "jupiter_beam_relative_gain_db"] = interpolate_cyclic(angles, gain_db, sep)
    return out


def compute_beam_separation_deg(beam_vectors: np.ndarray, source_vectors: np.ndarray) -> np.ndarray:
    beam = normalize_vectors(np.asarray(beam_vectors, dtype=float))
    source = normalize_vectors(np.asarray(source_vectors, dtype=float))
    dots = np.einsum("ij,ij->i", beam, source)
    return np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))


def read_upper_v_samples(path: Path, frequency_bands: list[int]) -> pd.DataFrame:
    arr = np.load(path, allow_pickle=True)
    keep = np.isin(arr["frequency_band"], np.asarray(frequency_bands, dtype=int))
    keep &= arr["antenna"].astype(str) == UPPER_V
    keep &= arr["is_valid"].astype(bool)
    power = arr["power"].astype(float)
    keep &= np.isfinite(power) & (power > 0.0)
    sub = arr[keep]
    out = pd.DataFrame(
        {
            "time": pd.to_datetime(sub["time"].astype(str)),
            "frequency_band": sub["frequency_band"].astype(int),
            "frequency_mhz": sub["frequency_mhz"].astype(float),
            "antenna": sub["antenna"].astype(str),
            "power": sub["power"].astype(float),
            "right_ascension": sub["right_ascension"].astype(float),
            "declination": sub["declination"].astype(float),
        }
    )
    out = out[np.isfinite(out["right_ascension"]) & np.isfinite(out["declination"])].copy()
    out["log10_power"] = np.log10(out["power"].to_numpy(dtype=float))
    out["date"] = out["time"].dt.floor("D")
    stats = (
        out.groupby(["date", "frequency_band"], sort=True)["log10_power"]
        .median()
        .rename("daily_median_log10_power")
        .reset_index()
    )
    out = out.merge(stats, on=["date", "frequency_band"], how="left")
    out["daily_log10_residual"] = out["log10_power"] - out["daily_median_log10_power"]
    return out.sort_values("time").reset_index(drop=True)


def attach_local_power_normalization(
    samples: pd.DataFrame,
    window_s: float,
    min_periods: int,
) -> pd.DataFrame:
    """Attach centered rolling local-normalized raw power within each frequency.

    This mirrors the Earth profile normalization idea: subtract a local median
    raw power and divide by a robust local scale.  For this continuous Io-phase
    view there is no event center or sideband, so the local statistics use the
    centered time window around each sample.
    """
    out = samples.copy()
    out["local_normalization_window_s"] = float(window_s)
    out["local_normalization_center_power"] = np.nan
    out["local_normalization_scale_power"] = np.nan
    out["local_normalized_power"] = np.nan
    window = pd.to_timedelta(float(window_s), unit="s")
    min_count = int(min_periods)
    for _, grp in out.groupby("frequency_band", sort=True):
        ordered = grp.sort_values("time")
        power = pd.to_numeric(ordered["power"], errors="coerce").to_numpy(dtype=float)
        time_index = pd.DatetimeIndex(ordered["time"])
        series = pd.Series(power, index=time_index)
        rolling = series.rolling(window=window, center=True, min_periods=min_count)
        center = rolling.median()
        abs_dev = (series - center).abs()
        scale = 1.4826 * abs_dev.rolling(window=window, center=True, min_periods=min_count).median()
        fallback_scale = rolling.std(ddof=1)
        scale = scale.where(np.isfinite(scale) & (scale > 0.0), fallback_scale)
        values = (series - center) / scale
        invalid = ~np.isfinite(scale.to_numpy(dtype=float)) | (scale.to_numpy(dtype=float) <= 0.0)
        values = values.mask(invalid)
        out.loc[ordered.index, "local_normalization_center_power"] = center.to_numpy(dtype=float)
        out.loc[ordered.index, "local_normalization_scale_power"] = scale.to_numpy(dtype=float)
        out.loc[ordered.index, "local_normalized_power"] = values.to_numpy(dtype=float)
    return out


def load_geometry_with_jupiter_vectors(path: Path, target_frame: str) -> pd.DataFrame:
    cols = [
        "time",
        "jupiter_visible_by_moon",
        "jupiter_limb_angle_deg",
        "earth_visible_by_moon",
        "earth_limb_angle_deg",
        "jupiter_cml_spice_deg",
        "io_phase_spice_deg",
        "maser_zarka_io_score",
    ]
    geom = pd.read_csv(path, usecols=cols, parse_dates=["time"], low_memory=False)
    geom = geom.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    for col in ["jupiter_visible_by_moon", "earth_visible_by_moon"]:
        geom[col] = geom[col].astype(bool)
    vectors = body_unit_vectors_from_moon("jupiter", geom["time"], target_frame=str(target_frame))
    geom["jupiter_unit_x"] = vectors[:, 0]
    geom["jupiter_unit_y"] = vectors[:, 1]
    geom["jupiter_unit_z"] = vectors[:, 2]
    return geom


def merge_geometry_and_beam(
    samples: pd.DataFrame,
    geom: pd.DataFrame,
    tolerance_s: float,
    ra_units: str,
) -> tuple[pd.DataFrame, str]:
    keep_cols = [
        "time",
        "jupiter_visible_by_moon",
        "jupiter_limb_angle_deg",
        "earth_visible_by_moon",
        "earth_limb_angle_deg",
        "jupiter_cml_spice_deg",
        "io_phase_spice_deg",
        "maser_zarka_io_score",
        "jupiter_unit_x",
        "jupiter_unit_y",
        "jupiter_unit_z",
    ]
    merged = pd.merge_asof(
        samples.sort_values("time"),
        geom[keep_cols].sort_values("time"),
        on="time",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=float(tolerance_s)),
    )
    merged = merged.dropna(subset=["io_phase_spice_deg", "jupiter_unit_x"]).copy()
    for col in ["jupiter_visible_by_moon", "earth_visible_by_moon"]:
        merged[col] = merged[col].astype(bool)
    beam, interpreted_ra_units = radec_unit_vectors(merged["right_ascension"], merged["declination"], ra_units)
    jupiter = merged[["jupiter_unit_x", "jupiter_unit_y", "jupiter_unit_z"]].to_numpy(dtype=float)
    merged["jupiter_beam_separation_deg"] = compute_beam_separation_deg(beam, jupiter)
    merged["beam_half_width_deg"] = merged["frequency_mhz"].map(beam_half_width_deg_for_frequency)
    merged["jupiter_passes_angular_beam_cut"] = (
        merged["jupiter_beam_separation_deg"] <= merged["beam_half_width_deg"]
    )
    merged["jupiter_within_frequency_beam"] = merged["jupiter_passes_angular_beam_cut"]
    merged = attach_digitized_beam_gain(merged)
    return merged, interpreted_ra_units


def summarize_counts(
    merged: pd.DataFrame,
    selected: pd.DataFrame,
    selection_mode: str,
    gain_threshold_db: float,
) -> pd.DataFrame:
    rows = []
    for band, freq in sorted(merged[["frequency_band", "frequency_mhz"]].drop_duplicates().itertuples(index=False)):
        all_band = merged[merged["frequency_band"].eq(int(band))]
        visible = all_band[all_band["jupiter_visible_by_moon"]]
        beam = selected[selected["frequency_band"].eq(int(band))]
        model = all_band[["beam_model_frequency_mhz", "beam_model_eplane", "beam_model_hplane"]].dropna(
            subset=["beam_model_frequency_mhz"]
        )
        model_row = model.iloc[0] if len(model) else None
        rows.append(
            {
                "frequency_band": int(band),
                "frequency_mhz": float(freq),
                "selection_mode": str(selection_mode),
                "beam_half_width_deg": beam_half_width_deg_for_frequency(float(freq)),
                "beam_gain_threshold_db": float(gain_threshold_db),
                "beam_model_frequency_mhz": float(model_row["beam_model_frequency_mhz"]) if model_row is not None else np.nan,
                "beam_model_eplane": str(model_row["beam_model_eplane"]) if model_row is not None else "",
                "beam_model_hplane": str(model_row["beam_model_hplane"]) if model_row is not None else "",
                "n_geometry_merged_upper_v_samples": int(len(all_band)),
                "n_jupiter_visible_upper_v_samples": int(len(visible)),
                "n_angular_selected_upper_v_samples": int(visible["jupiter_passes_angular_beam_cut"].sum()),
                "n_gain_selected_upper_v_samples": int(visible["jupiter_passes_gain_beam_cut"].sum())
                if "jupiter_passes_gain_beam_cut" in visible
                else np.nan,
                "n_beam_selected_upper_v_samples": int(len(beam)),
                "beam_selected_fraction_of_visible": float(len(beam) / len(visible)) if len(visible) else np.nan,
                "median_jupiter_beam_separation_selected_deg": float(beam["jupiter_beam_separation_deg"].median())
                if len(beam)
                else np.nan,
                "p90_jupiter_beam_separation_selected_deg": float(beam["jupiter_beam_separation_deg"].quantile(0.90))
                if len(beam)
                else np.nan,
                "median_jupiter_beam_relative_gain_selected_db": float(beam["jupiter_beam_relative_gain_db"].median())
                if len(beam)
                else np.nan,
                "p10_jupiter_beam_relative_gain_selected_db": float(beam["jupiter_beam_relative_gain_db"].quantile(0.10))
                if len(beam)
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize_phase_frequency(
    selected: pd.DataFrame,
    phase_bin_deg: float,
    high_factor: float,
    min_count_per_bin: int,
) -> pd.DataFrame:
    work = selected.copy()
    work["io_phase_spice_deg"] = pd.to_numeric(work["io_phase_spice_deg"], errors="coerce") % 360.0
    work["io_phase_bin_deg"] = (
        np.floor(work["io_phase_spice_deg"] / float(phase_bin_deg)) * float(phase_bin_deg)
        + 0.5 * float(phase_bin_deg)
    )
    work["factor_high"] = work["daily_log10_residual"] >= float(np.log10(float(high_factor)))
    summary = (
        work.groupby(["frequency_band", "frequency_mhz", "io_phase_bin_deg"], sort=True)
        .agg(
            n_samples=("daily_log10_residual", "size"),
            median_daily_residual=("daily_log10_residual", "median"),
            q75_daily_residual=("daily_log10_residual", lambda x: float(np.nanquantile(x, 0.75))),
            factor_high_fraction=("factor_high", "mean"),
            median_raw_log10_power=("log10_power", "median"),
            mean_local_normalized_power=("local_normalized_power", "mean"),
            median_local_normalized_power=("local_normalized_power", "median"),
            q75_local_normalized_power=("local_normalized_power", lambda x: float(np.nanquantile(x, 0.75))),
            median_beam_separation_deg=("jupiter_beam_separation_deg", "median"),
            median_beam_relative_gain_db=("jupiter_beam_relative_gain_db", "median"),
        )
        .reset_index()
    )
    low = summary["n_samples"] < int(min_count_per_bin)
    for col in [
        "median_daily_residual",
        "q75_daily_residual",
        "factor_high_fraction",
        "median_raw_log10_power",
        "mean_local_normalized_power",
        "median_local_normalized_power",
        "q75_local_normalized_power",
    ]:
        summary.loc[low, col] = np.nan
    return summary


def matrix_for(summary: pd.DataFrame, value_col: str, phase_bin_deg: float) -> pd.DataFrame:
    phases = np.arange(0.5 * phase_bin_deg, 360.0, phase_bin_deg)
    freqs = sorted(summary["frequency_mhz"].dropna().unique())
    mat = summary.pivot_table(index="frequency_mhz", columns="io_phase_bin_deg", values=value_col, aggfunc="mean")
    return mat.reindex(index=freqs, columns=phases)


def decorate_io_bands(ax: plt.Axes) -> None:
    bands = [("Io-B/D", 80.0, 130.0, "#d95f02"), ("Io-A/C", 205.0, 260.0, "#1b9e77")]
    for label, lo, hi, color in bands:
        ax.axvspan(lo, hi, color=color, alpha=0.10, lw=0)
        ax.axvline(lo, color=color, lw=1.0, alpha=0.85)
        ax.axvline(hi, color=color, lw=1.0, alpha=0.85)
        ax.text(
            (lo + hi) / 2.0,
            0.98,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
            color=color,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.70, "pad": 1.0},
        )


def plot_heatmap(
    summary: pd.DataFrame,
    out_dir: Path,
    value_col: str,
    title: str,
    cbar_label: str,
    filename: str,
    phase_bin_deg: float,
    cmap: str,
    centered: bool,
    log_counts: bool = False,
) -> Path:
    mat = matrix_for(summary, value_col=value_col, phase_bin_deg=phase_bin_deg)
    vals = mat.to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    fig, ax = plt.subplots(figsize=(12.8, 5.8))
    if log_counts:
        vals = vals.copy()
        vals[vals <= 0] = np.nan
        finite = vals[np.isfinite(vals)]
        norm = LogNorm(vmin=1.0, vmax=max(1.0, float(np.nanmax(finite))) if finite.size else 1.0)
    elif centered:
        vmax = max(0.02, float(np.nanpercentile(np.abs(finite), 98))) if finite.size else 0.05
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    else:
        vmax = max(0.01, float(np.nanpercentile(finite, 98))) if finite.size else 0.05
        norm = Normalize(vmin=0.0, vmax=vmax)
    im = ax.imshow(
        vals,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[
            float(mat.columns.min() - 0.5 * phase_bin_deg),
            float(mat.columns.max() + 0.5 * phase_bin_deg),
            -0.5,
            len(mat.index) - 0.5,
        ],
        cmap=cmap,
        norm=norm,
    )
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels([f"{freq:.2f}" for freq in mat.index])
    ax.set_xlim(0, 360)
    ax.set_xticks(np.arange(0, 361, 30))
    ax.set_xlabel("Io phase (deg)")
    ax.set_ylabel("frequency (MHz)")
    ax.set_title(title)
    ax.grid(True, axis="x", color="white", alpha=0.15, lw=0.35)
    decorate_io_bands(ax)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label=cbar_label)
    fig.tight_layout()
    path = out_dir / filename
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _fmt_scale(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    if abs(value) >= 10:
        return f"{value:.0f}"
    if abs(value) >= 1:
        return f"{value:.2g}"
    return f"{value:.2f}"


def plot_heatmap_row_scaled(
    summary: pd.DataFrame,
    out_dir: Path,
    value_col: str,
    title: str,
    cbar_label: str,
    filename: str,
    phase_bin_deg: float,
    cmap: str,
    centered: bool,
) -> Path:
    mat = matrix_for(summary, value_col=value_col, phase_bin_deg=phase_bin_deg)
    vals = mat.to_numpy(dtype=float)
    scaled = np.full_like(vals, np.nan, dtype=float)
    tick_labels = []
    for idx, freq in enumerate(mat.index):
        row = vals[idx]
        finite = row[np.isfinite(row)]
        if finite.size == 0:
            tick_labels.append(f"{freq:.2f}\nno data")
            continue
        if centered:
            scale = float(np.nanpercentile(np.abs(finite), 98)) if finite.size >= 4 else float(np.nanmax(np.abs(finite)))
            if not np.isfinite(scale) or scale <= 0.0:
                scale = 1.0
            scaled[idx] = np.clip(row / scale, -1.0, 1.0)
            tick_labels.append(f"{freq:.2f}\n+/-{_fmt_scale(scale)}")
        else:
            lo, hi = np.nanpercentile(finite, [2, 98]) if finite.size >= 4 else (np.nanmin(finite), np.nanmax(finite))
            if not np.isfinite(lo) or not np.isfinite(hi) or np.isclose(lo, hi):
                lo = float(np.nanmin(finite))
                hi = float(np.nanmax(finite))
            if not np.isfinite(lo) or not np.isfinite(hi) or np.isclose(lo, hi):
                scaled[idx] = 0.5
                tick_labels.append(f"{freq:.2f}\nflat")
            else:
                scaled[idx] = np.clip((row - lo) / (hi - lo), 0.0, 1.0)
                tick_labels.append(f"{freq:.2f}\n{_fmt_scale(lo)}-{_fmt_scale(hi)}")

    fig, ax = plt.subplots(figsize=(12.8, 5.8))
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0) if centered else Normalize(vmin=0.0, vmax=1.0)
    im = ax.imshow(
        scaled,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[
            float(mat.columns.min() - 0.5 * phase_bin_deg),
            float(mat.columns.max() + 0.5 * phase_bin_deg),
            -0.5,
            len(mat.index) - 0.5,
        ],
        cmap=cmap,
        norm=norm,
    )
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels(tick_labels)
    ax.set_xlim(0, 360)
    ax.set_xticks(np.arange(0, 361, 30))
    ax.set_xlabel("Io phase (deg)")
    ax.set_ylabel("frequency (MHz)\nrow scale")
    ax.set_title(title)
    ax.grid(True, axis="x", color="white", alpha=0.15, lw=0.35)
    decorate_io_bands(ax)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label=cbar_label)
    fig.tight_layout()
    path = out_dir / filename
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_1d_profiles(
    summary: pd.DataFrame,
    out_dir: Path,
    value_col: str,
    ylabel: str,
    title: str,
    filename: str,
    phase_bin_deg: float,
    centered: bool,
) -> Path:
    freqs = sorted(summary["frequency_mhz"].dropna().unique())
    fig, axes = plt.subplots(len(freqs), 1, figsize=(12.8, max(2.0 * len(freqs), 7.5)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, freq in zip(axes, freqs):
        sub = summary[np.isclose(summary["frequency_mhz"], float(freq))].sort_values("io_phase_bin_deg").copy()
        x = sub["io_phase_bin_deg"].to_numpy(dtype=float)
        y = sub[value_col].to_numpy(dtype=float)
        n = sub["n_samples"].to_numpy(dtype=float)
        decorate_io_bands(ax)
        if np.isfinite(y).any():
            ax.plot(x, y, marker="o", ms=3.0, lw=1.25, color="#386cb0")
            ax.scatter(x, y, s=np.clip(n, 12, 90), color="#386cb0", alpha=0.55, edgecolor="white", linewidth=0.35)
            finite = y[np.isfinite(y)]
            ylo, yhi = np.nanquantile(finite, [0.05, 0.95]) if finite.size >= 4 else (np.nanmin(finite), np.nanmax(finite))
            if np.isclose(ylo, yhi):
                pad = max(0.01, abs(float(yhi)) * 0.10)
            else:
                pad = max(0.01, 0.18 * float(yhi - ylo))
            if centered:
                lim = max(abs(float(ylo)), abs(float(yhi))) + pad
                ax.set_ylim(-lim, lim)
                ax.axhline(0.0, color="0.35", lw=0.8)
            else:
                ax.set_ylim(float(ylo) - pad, float(yhi) + pad)
        else:
            ax.text(0.5, 0.52, "no plotted bins", transform=ax.transAxes, ha="center", va="center", color="0.35")
        count_ax = ax.twinx()
        if np.isfinite(n).any() and np.nanmax(n) > 0:
            count_ax.bar(x, n, width=float(phase_bin_deg) * 0.85, color="0.82", alpha=0.35, align="center", zorder=0)
            count_ax.set_ylim(0, max(1.0, float(np.nanmax(n)) * 1.18))
        count_ax.set_ylabel("n", fontsize=8, color="0.45")
        count_ax.tick_params(axis="y", labelsize=7, colors="0.45")
        ax.set_ylabel(f"{freq:.2f} MHz\n{ylabel}")
        ax.set_xlim(0, 360)
        ax.grid(True, color="0.90", lw=0.45)
        total = int(np.nansum(n)) if np.isfinite(n).any() else 0
        plotted = int(np.count_nonzero(np.isfinite(y)))
        ax.text(
            0.99,
            0.88,
            f"total n={total:,}; plotted bins={plotted}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
        )
    axes[-1].set_xticks(np.arange(0, 361, 30))
    axes[-1].set_xlabel("Io phase (deg)")
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    path = out_dir / filename
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_1d_raw_samples(selected: pd.DataFrame, out_dir: Path, filename: str) -> Path:
    freqs = sorted(selected["frequency_mhz"].dropna().unique())
    fig, axes = plt.subplots(len(freqs), 1, figsize=(12.8, max(2.0 * len(freqs), 7.5)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, freq in zip(axes, freqs):
        sub = selected[np.isclose(selected["frequency_mhz"], float(freq))].copy()
        x = (pd.to_numeric(sub["io_phase_spice_deg"], errors="coerce").to_numpy(dtype=float) % 360.0)
        y = pd.to_numeric(sub["power"], errors="coerce").to_numpy(dtype=float)
        good = np.isfinite(x) & np.isfinite(y) & (y > 0.0)
        decorate_io_bands(ax)
        if np.any(good):
            ax.scatter(
                x[good],
                y[good],
                s=1.0,
                alpha=0.045,
                color="#386cb0",
                linewidths=0,
                rasterized=True,
            )
            ax.set_yscale("log")
        else:
            ax.text(0.5, 0.52, "no selected samples", transform=ax.transAxes, ha="center", va="center", color="0.35")
        ax.set_ylabel(f"{freq:.2f} MHz\nraw power")
        ax.set_xlim(0, 360)
        ax.grid(True, color="0.90", lw=0.45)
        ax.text(
            0.99,
            0.88,
            f"n={int(np.count_nonzero(good)):,}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
        )
    axes[-1].set_xticks(np.arange(0, 361, 30))
    axes[-1].set_xlabel("Io phase (deg)")
    fig.suptitle("Upper V beam-filtered Io-phase profiles: raw samples")
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    path = out_dir / filename
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_1d_local_normalized_samples(selected: pd.DataFrame, out_dir: Path, filename: str) -> Path:
    freqs = sorted(selected["frequency_mhz"].dropna().unique())
    fig, axes = plt.subplots(len(freqs), 1, figsize=(12.8, max(2.0 * len(freqs), 7.5)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, freq in zip(axes, freqs):
        sub = selected[np.isclose(selected["frequency_mhz"], float(freq))].copy()
        x = (pd.to_numeric(sub["io_phase_spice_deg"], errors="coerce").to_numpy(dtype=float) % 360.0)
        y = pd.to_numeric(sub["local_normalized_power"], errors="coerce").to_numpy(dtype=float)
        good = np.isfinite(x) & np.isfinite(y)
        decorate_io_bands(ax)
        if np.any(good):
            ax.scatter(
                x[good],
                y[good],
                s=1.0,
                alpha=0.045,
                color="#386cb0",
                linewidths=0,
                rasterized=True,
            )
            finite = y[good]
            lo, hi = np.nanpercentile(finite, [1.0, 99.0]) if finite.size >= 20 else (np.nanmin(finite), np.nanmax(finite))
            lim = max(3.0, abs(float(lo)), abs(float(hi)))
            ax.set_ylim(-lim, lim)
            ax.axhline(0.0, color="0.35", lw=0.8)
        else:
            ax.text(0.5, 0.52, "no selected samples", transform=ax.transAxes, ha="center", va="center", color="0.35")
        ax.set_ylabel(f"{freq:.2f} MHz\nlocal z")
        ax.set_xlim(0, 360)
        ax.grid(True, color="0.90", lw=0.45)
        ax.text(
            0.99,
            0.88,
            f"n={int(np.count_nonzero(good)):,}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
        )
    axes[-1].set_xticks(np.arange(0, 361, 30))
    axes[-1].set_xlabel("Io phase (deg)")
    fig.suptitle("Upper V beam-filtered Io-phase profiles: local-normalized raw samples")
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    path = out_dir / filename
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_frequency_counts(counts: pd.DataFrame, out_dir: Path, gain_threshold_db: float) -> Path:
    rows = counts.sort_values("frequency_mhz")
    freqs = rows["frequency_mhz"].to_numpy(dtype=float)
    selected = rows["n_beam_selected_upper_v_samples"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    ax.bar(np.arange(len(rows)), selected, color="#386cb0", width=0.72)
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels([f"{freq:.2f}" for freq in freqs])
    ax.set_xlabel("frequency (MHz)")
    ax.set_ylabel("selected upper-V samples")
    ax.set_title(f"Jupiter lunar-visible samples with beam gain >= {float(gain_threshold_db):g} dB")
    ax.grid(True, axis="y", color="0.88", lw=0.6)
    ymax = max(1.0, float(np.nanmax(selected)) if selected.size else 1.0)
    ax.set_ylim(0, ymax * 1.16)
    for i, n in enumerate(selected):
        ax.text(i, n + ymax * 0.015, f"{int(n):,}", ha="center", va="bottom", fontsize=8, rotation=0)
    fig.tight_layout()
    path = out_dir / "jupiter_upper_v_beam_frequency_sample_counts.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report(
    out_dir: Path,
    counts: pd.DataFrame,
    paths: list[Path],
    config: dict[str, object],
) -> Path:
    report_counts = counts.drop(columns=["beam_model_eplane", "beam_model_hplane"], errors="ignore")
    lines = [
        "# Jupiter Upper-V Beam-Filtered Io Phase/Frequency Plot",
        "",
        "This run uses upper V only (`rv1_coarse`) and keeps samples where Jupiter is lunar-visible and passes the configured beam-selection rule.",
        "",
        "## Beam Cut",
        "",
        "- Default rule: interpolate the nearest digitized Ryle-Vonberg E/H-plane beam pattern at Jupiter's off-axis angle.",
        "- The E/H planes are averaged in linear power and normalized to the beam peak.",
        "- Samples are kept when the relative gain toward Jupiter is at least the configured threshold.",
        "- The older hard angular cut is still available with `--selection-mode angular`.",
        "",
        "## Local Normalization",
        "",
        "Local-normalized raw power is computed within each frequency channel as `(power - centered rolling median power) / centered rolling robust sigma`.  The default rolling window is 20 minutes.",
        "",
        "The upper-V beam axis is taken from the sample `right_ascension`/`declination`; Jupiter's apparent lunar-centered direction is computed in FK4/B1950 and merged to each sample by nearest geometry time.",
        "",
        "## Configuration",
        "",
        *[f"- `{k}`: `{v}`" for k, v in config.items() if k != "software_versions"],
        "",
        "## Samples Kept By Frequency",
        "",
        report_counts.to_string(index=False),
        "",
        "## Files",
        "",
        *[f"- `{p}`" for p in paths],
    ]
    path = out_dir / "jupiter_upper_v_beam_io_phase_frequency_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-npy", type=Path, default=DEFAULT_CLEAN_NPY)
    parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--frequency-band", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument("--phase-bin-deg", type=float, default=10.0)
    parser.add_argument("--high-factor", type=float, default=2.0)
    parser.add_argument("--min-count-per-bin", type=int, default=3)
    parser.add_argument("--geometry-tolerance-s", type=float, default=360.0)
    parser.add_argument("--ra-units", choices=["auto", "hours", "degrees"], default="auto")
    parser.add_argument("--target-frame", choices=["fk4", "fk5", "icrs"], default="fk4")
    parser.add_argument("--selection-mode", choices=["beam-gain", "angular"], default="beam-gain")
    parser.add_argument("--gain-threshold-db", type=float, default=-10.0)
    parser.add_argument("--local-normalization-window-minutes", type=float, default=20.0)
    parser.add_argument("--local-normalization-min-samples", type=int, default=6)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    config = {
        "clean_npy": str(args.clean_npy),
        "geometry": str(args.geometry),
        "frequency_bands": [int(v) for v in args.frequency_band],
        "frequencies_mhz": [FREQUENCY_MAP_MHZ.get(int(v), np.nan) for v in args.frequency_band],
        "antenna": UPPER_V,
        "phase_bin_deg": float(args.phase_bin_deg),
        "high_factor": float(args.high_factor),
        "min_count_per_bin": int(args.min_count_per_bin),
        "geometry_tolerance_s": float(args.geometry_tolerance_s),
        "ra_units_requested": str(args.ra_units),
        "target_frame": str(args.target_frame),
        "selection_mode": str(args.selection_mode),
        "gain_threshold_db": float(args.gain_threshold_db),
        "local_normalization_window_minutes": float(args.local_normalization_window_minutes),
        "local_normalization_min_samples": int(args.local_normalization_min_samples),
        "beam_pattern_directory": str(BEAM_DIR),
        "beam_cut_rule": {
            "freq_mhz_lt_1p31": 90.0,
            "freq_mhz_ge_1p31_lt_6p55": 20.0,
            "freq_mhz_ge_6p55": 10.0,
        },
        "software_versions": software_versions(),
    }

    print("Loading upper-V samples...", flush=True)
    samples = read_upper_v_samples(args.clean_npy, [int(v) for v in args.frequency_band])
    print("Applying local rolling raw-power normalization...", flush=True)
    samples = attach_local_power_normalization(
        samples,
        window_s=float(args.local_normalization_window_minutes) * 60.0,
        min_periods=int(args.local_normalization_min_samples),
    )
    print(f"Loaded {len(samples)} upper-V samples; computing Jupiter direction on geometry grid...", flush=True)
    geom = load_geometry_with_jupiter_vectors(args.geometry, target_frame=str(args.target_frame))
    print("Merging geometry and applying beam cuts...", flush=True)
    merged, interpreted_ra_units = merge_geometry_and_beam(
        samples,
        geom,
        tolerance_s=float(args.geometry_tolerance_s),
        ra_units=str(args.ra_units),
    )
    config["ra_units_interpreted"] = interpreted_ra_units
    merged["jupiter_passes_gain_beam_cut"] = merged["jupiter_beam_relative_gain_db"] >= float(args.gain_threshold_db)
    if str(args.selection_mode) == "beam-gain":
        selected_mask = merged["jupiter_visible_by_moon"] & merged["jupiter_passes_gain_beam_cut"]
    else:
        selected_mask = merged["jupiter_visible_by_moon"] & merged["jupiter_passes_angular_beam_cut"]
    merged["jupiter_selected_by_beam"] = selected_mask
    merged["jupiter_within_frequency_beam"] = selected_mask
    write_json(out_dir / "run_config.json", config)
    selected = merged[selected_mask].copy()
    counts = summarize_counts(
        merged,
        selected,
        selection_mode=str(args.selection_mode),
        gain_threshold_db=float(args.gain_threshold_db),
    )
    summary = summarize_phase_frequency(
        selected,
        phase_bin_deg=float(args.phase_bin_deg),
        high_factor=float(args.high_factor),
        min_count_per_bin=int(args.min_count_per_bin),
    )
    merged_cols = [
        "time",
        "frequency_band",
        "frequency_mhz",
        "power",
        "log10_power",
        "daily_log10_residual",
        "local_normalization_window_s",
        "local_normalization_center_power",
        "local_normalization_scale_power",
        "local_normalized_power",
        "io_phase_spice_deg",
        "jupiter_cml_spice_deg",
        "jupiter_visible_by_moon",
        "jupiter_beam_separation_deg",
        "beam_half_width_deg",
        "beam_model_frequency_mhz",
        "jupiter_beam_relative_gain_db",
        "jupiter_passes_angular_beam_cut",
        "jupiter_passes_gain_beam_cut",
        "jupiter_selected_by_beam",
        "jupiter_within_frequency_beam",
    ]
    counts.to_csv(out_dir / "jupiter_upper_v_beam_frequency_sample_counts.csv", index=False)
    summary.to_csv(out_dir / "jupiter_upper_v_beam_io_phase_frequency_summary.csv", index=False)
    selected[merged_cols].to_csv(out_dir / "jupiter_upper_v_beam_selected_samples.csv", index=False)

    print("Making plots...", flush=True)
    paths: list[Path] = [
        out_dir / "run_config.json",
        out_dir / "jupiter_upper_v_beam_frequency_sample_counts.csv",
        out_dir / "jupiter_upper_v_beam_io_phase_frequency_summary.csv",
        out_dir / "jupiter_upper_v_beam_selected_samples.csv",
    ]
    paths.append(plot_frequency_counts(counts, out_dir, gain_threshold_db=float(args.gain_threshold_db)))
    paths.append(
        plot_heatmap(
            summary,
            out_dir,
            value_col="median_daily_residual",
            title="Upper V: beam-filtered Jupiter median daily residual by Io phase/frequency",
            cbar_label="median log10(power) minus same-day band median (dex)",
            filename="jupiter_upper_v_beam_io_phase_frequency_median_daily_residual.png",
            phase_bin_deg=float(args.phase_bin_deg),
            cmap="coolwarm",
            centered=True,
        )
    )
    paths.append(
        plot_heatmap(
            summary,
            out_dir,
            value_col="q75_daily_residual",
            title="Upper V: beam-filtered q75 daily residual by Io phase/frequency",
            cbar_label="75th percentile log10(power) minus same-day band median (dex)",
            filename="jupiter_upper_v_beam_io_phase_frequency_q75_daily_residual.png",
            phase_bin_deg=float(args.phase_bin_deg),
            cmap="coolwarm",
            centered=True,
        )
    )
    paths.append(
        plot_heatmap(
            summary,
            out_dir,
            value_col="median_raw_log10_power",
            title="Upper V: beam-filtered median raw log10 power by Io phase/frequency",
            cbar_label="median raw log10(power)",
            filename="jupiter_upper_v_beam_io_phase_frequency_median_raw_log10_power.png",
            phase_bin_deg=float(args.phase_bin_deg),
            cmap="viridis",
            centered=False,
        )
    )
    paths.append(
        plot_heatmap(
            summary,
            out_dir,
            value_col="median_local_normalized_power",
            title="Upper V: beam-filtered median local-normalized raw power by Io phase/frequency",
            cbar_label="median local-normalized raw power",
            filename="jupiter_upper_v_beam_io_phase_frequency_median_local_normalized_power.png",
            phase_bin_deg=float(args.phase_bin_deg),
            cmap="coolwarm",
            centered=True,
        )
    )
    paths.append(
        plot_heatmap(
            summary,
            out_dir,
            value_col="mean_local_normalized_power",
            title="Upper V: beam-filtered mean local-normalized raw power by Io phase/frequency",
            cbar_label="mean local-normalized raw power",
            filename="jupiter_upper_v_beam_io_phase_frequency_mean_local_normalized_power.png",
            phase_bin_deg=float(args.phase_bin_deg),
            cmap="coolwarm",
            centered=True,
        )
    )
    paths.append(
        plot_heatmap_row_scaled(
            summary,
            out_dir,
            value_col="mean_local_normalized_power",
            title="Upper V: beam-filtered mean local-normalized raw power by Io phase/frequency, row-scaled",
            cbar_label="row-scaled mean local-normalized power",
            filename="jupiter_upper_v_beam_io_phase_frequency_mean_local_normalized_power_row_scaled.png",
            phase_bin_deg=float(args.phase_bin_deg),
            cmap="coolwarm",
            centered=True,
        )
    )
    paths.append(
        plot_heatmap(
            summary,
            out_dir,
            value_col="q75_local_normalized_power",
            title="Upper V: beam-filtered q75 local-normalized raw power by Io phase/frequency",
            cbar_label="75th percentile local-normalized raw power",
            filename="jupiter_upper_v_beam_io_phase_frequency_q75_local_normalized_power.png",
            phase_bin_deg=float(args.phase_bin_deg),
            cmap="coolwarm",
            centered=True,
        )
    )
    paths.append(
        plot_heatmap_row_scaled(
            summary,
            out_dir,
            value_col="q75_local_normalized_power",
            title="Upper V: beam-filtered q75 local-normalized raw power by Io phase/frequency, row-scaled",
            cbar_label="row-scaled q75 local-normalized power",
            filename="jupiter_upper_v_beam_io_phase_frequency_q75_local_normalized_power_row_scaled.png",
            phase_bin_deg=float(args.phase_bin_deg),
            cmap="magma",
            centered=False,
        )
    )
    paths.append(
        plot_heatmap(
            summary,
            out_dir,
            value_col="factor_high_fraction",
            title=f"Upper V: beam-filtered fraction >= {float(args.high_factor):g}x daily median",
            cbar_label=f"fraction >= {float(args.high_factor):g}x",
            filename="jupiter_upper_v_beam_io_phase_frequency_factor_high_fraction.png",
            phase_bin_deg=float(args.phase_bin_deg),
            cmap="magma",
            centered=False,
        )
    )
    paths.append(
        plot_heatmap(
            summary,
            out_dir,
            value_col="n_samples",
            title="Upper V: beam-filtered sample count by Io phase/frequency",
            cbar_label="samples per Io-phase bin",
            filename="jupiter_upper_v_beam_io_phase_frequency_sample_counts.png",
            phase_bin_deg=float(args.phase_bin_deg),
            cmap="viridis",
            centered=False,
            log_counts=True,
        )
    )
    paths.append(
        plot_1d_raw_samples(
            selected,
            out_dir,
            filename="jupiter_upper_v_beam_io_phase_1d_raw_power_samples.png",
        )
    )
    paths.append(
        plot_1d_local_normalized_samples(
            selected,
            out_dir,
            filename="jupiter_upper_v_beam_io_phase_1d_local_normalized_power_samples.png",
        )
    )
    paths.append(
        plot_1d_profiles(
            summary,
            out_dir,
            value_col="median_daily_residual",
            ylabel="median residual",
            title="Upper V beam-filtered Io-phase profiles: median daily residual",
            filename="jupiter_upper_v_beam_io_phase_1d_median_daily_residual.png",
            phase_bin_deg=float(args.phase_bin_deg),
            centered=True,
        )
    )
    paths.append(
        plot_1d_profiles(
            summary,
            out_dir,
            value_col="q75_daily_residual",
            ylabel="q75 residual",
            title="Upper V beam-filtered Io-phase profiles: q75 daily residual",
            filename="jupiter_upper_v_beam_io_phase_1d_q75_daily_residual.png",
            phase_bin_deg=float(args.phase_bin_deg),
            centered=True,
        )
    )
    paths.append(
        plot_1d_profiles(
            summary,
            out_dir,
            value_col="median_raw_log10_power",
            ylabel="raw log10 power",
            title="Upper V beam-filtered Io-phase profiles: median raw log10 power",
            filename="jupiter_upper_v_beam_io_phase_1d_median_raw_log10_power.png",
            phase_bin_deg=float(args.phase_bin_deg),
            centered=False,
        )
    )
    paths.append(
        plot_1d_profiles(
            summary,
            out_dir,
            value_col="median_local_normalized_power",
            ylabel="median local z",
            title="Upper V beam-filtered Io-phase profiles: median local-normalized raw power",
            filename="jupiter_upper_v_beam_io_phase_1d_median_local_normalized_power.png",
            phase_bin_deg=float(args.phase_bin_deg),
            centered=True,
        )
    )
    paths.append(
        plot_1d_profiles(
            summary,
            out_dir,
            value_col="mean_local_normalized_power",
            ylabel="mean local z",
            title="Upper V beam-filtered Io-phase profiles: mean local-normalized raw power",
            filename="jupiter_upper_v_beam_io_phase_1d_mean_local_normalized_power.png",
            phase_bin_deg=float(args.phase_bin_deg),
            centered=True,
        )
    )
    report = write_report(out_dir, counts=counts, paths=paths, config=config)
    print(f"Wrote {report}", flush=True)


if __name__ == "__main__":
    main()
