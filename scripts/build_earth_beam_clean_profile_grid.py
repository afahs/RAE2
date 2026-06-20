#!/usr/bin/env python
"""Build Earth all-frequency grids using a beam-weighted diffuse-Galaxy cut.

This is stricter than a boresight-only Galactic-latitude cut.  For every
lower-V Earth event/frequency row, it predicts the diffuse synchrotron pickup
seen by the Moon-facing antenna using digitized Ryle-Vonberg E/H-plane beam
cuts and PySM low-frequency sky maps.  It then keeps rows where both the
beam-weighted diffuse level and its local time slope are small.

The beam model is still approximate: only 1D E/H cuts are available, so the
script uses their mean as an axisymmetric power beam.
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
    CLEAN,
    EARTH_EVENTS,
    _read,
    collect_profiles,
    summarize_profiles,
)


BEAM_DIR = Path(os.environ.get("RAE2_ANTENNA_DIGITIZATION_DIR", "data/antennaDigitization"))
SKY_DIR = Path(os.environ.get("RAE2_SKY_MAP_DIR", "data/pysmMaps"))
LOWER_V = "rv2_coarse"
WINDOW_S = 900.0
BIN_S = 60.0
INNER_S = 15.0
MODEL_NSIDE = 8

BEAM_SPECS = [
    (1.31, BEAM_DIR / "eplane_1p31MHz.csv", BEAM_DIR / "hplane_1p31MHz.csv"),
    (3.93, BEAM_DIR / "eplane_3p93MHz.csv", BEAM_DIR / "hplane_3p93MHz.csv"),
    (6.55, BEAM_DIR / "eplane_6p55MHz.csv", BEAM_DIR / "hplane_6p55MHz.csv"),
]


def _nearest_beam(freq_mhz: float) -> tuple[float, Path, Path]:
    return min(BEAM_SPECS, key=lambda spec: abs(spec[0] - float(freq_mhz)))


def _nearest_sky_path(freq_mhz: float) -> Path:
    # Available PySM maps are integer MHz files.  Clamp sub-MHz channels to the
    # 1 MHz map, which is also what the existing diffuse-beam diagnostic used.
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


def _load_beam(eplane: Path, hplane: Path) -> tuple[np.ndarray, np.ndarray]:
    e = read_table(eplane)
    h = read_table(hplane)
    angles = e["angle_deg"].to_numpy(dtype=float)
    gain = 0.5 * (10.0 ** (e["gain_dB"].to_numpy(dtype=float) / 10.0) + 10.0 ** (h["gain_dB"].to_numpy(dtype=float) / 10.0))
    return angles, gain


def _interp_cyclic(angles: np.ndarray, values: np.ndarray, angle_deg: np.ndarray) -> np.ndarray:
    a = np.asarray(angle_deg, dtype=float) % 360.0
    x = np.concatenate(([angles[-1] - 360.0], angles, [angles[0] + 360.0]))
    y = np.concatenate(([values[-1]], values, [values[0]]))
    return np.interp(a, x, y)


def _beam_weighted_sky(axes: np.ndarray, sky_i: np.ndarray, pixel_vecs: np.ndarray, beam_angles: np.ndarray, beam_gains: np.ndarray) -> np.ndarray:
    axes = np.asarray(axes, dtype=float)
    out = np.full(len(axes), np.nan)
    good = np.isfinite(axes).all(axis=1)
    for start in range(0, len(axes), 128):
        stop = min(start + 128, len(axes))
        idx = np.arange(start, stop)
        valid = good[start:stop]
        if not np.any(valid):
            continue
        block = axes[start:stop][valid]
        dots = np.clip(block @ pixel_vecs.T, -1.0, 1.0)
        sep = np.degrees(np.arccos(dots))
        gain = _interp_cyclic(beam_angles, beam_gains, sep)
        denom = np.nansum(gain, axis=1)
        numer = np.nansum(gain * sky_i.reshape(1, -1), axis=1)
        out[idx[valid]] = np.divide(numer, denom, out=np.full_like(numer, np.nan), where=denom > 0)
    return out


def _groups(clean: pd.DataFrame) -> dict[tuple[int, str], tuple[pd.DataFrame, np.ndarray]]:
    groups = {}
    for key, grp in clean.groupby(["frequency_band", "antenna"], sort=True):
        g = grp.sort_values("time").reset_index(drop=True)
        groups[(int(key[0]), str(key[1]))] = (g, datetime_ns(g["time"]))
    return groups


def _event_window(group: pd.DataFrame, group_ns: np.ndarray, event_time: pd.Timestamp, window_s: float) -> pd.DataFrame:
    event_ns = pd.Timestamp(event_time).value
    half_ns = int(float(window_s) * 1e9)
    lo = int(np.searchsorted(group_ns, event_ns - half_ns, side="left"))
    hi = int(np.searchsorted(group_ns, event_ns + half_ns, side="right"))
    if hi <= lo:
        return pd.DataFrame()
    local = group.iloc[lo:hi].copy()
    local["t_rel_sec"] = (datetime_ns(local["time"]) - event_ns).astype(float) / 1e9
    return local[np.abs(local["t_rel_sec"]) <= float(window_s)].copy()


def _robust_slope(t: np.ndarray, y: np.ndarray) -> float:
    good = np.isfinite(t) & np.isfinite(y)
    if np.count_nonzero(good) < 6:
        return np.nan
    # Ordinary least squares is enough for the smooth model values; robust scale
    # is used later for normalization of the score.
    return float(np.polyfit(t[good] / WINDOW_S, y[good], deg=1)[0])


def _event_model_metrics(clean: pd.DataFrame, events: pd.DataFrame, window_s: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = _groups(clean)
    rows = []
    input_rows = []
    pixel_vecs = _pixel_fk4_vectors(MODEL_NSIDE)
    for (band, freq), band_events in events.groupby(["frequency_band", "frequency_mhz"], sort=True):
        sky_i, sky_path = _load_sky_i(float(freq))
        beam_freq, eplane, hplane = _nearest_beam(float(freq))
        beam_angles, beam_gains = _load_beam(eplane, hplane)
        input_rows.append(
            {
                "frequency_band": int(band),
                "frequency_mhz": float(freq),
                "sky_map": str(sky_path),
                "beam_model_frequency_mhz": float(beam_freq),
                "eplane_beam": str(eplane),
                "hplane_beam": str(hplane),
                "nside": MODEL_NSIDE,
            }
        )
        payload = groups.get((int(band), LOWER_V))
        if payload is None:
            continue
        group, group_ns = payload
        for ev in band_events.itertuples(index=False):
            local = _event_window(group, group_ns, pd.Timestamp(ev.predicted_event_time), window_s)
            if local.empty or len(local) < 8:
                continue
            pos = local[["position_x", "position_y", "position_z"]].to_numpy(dtype=float)
            axes = normalize_vectors(-pos)
            model = _beam_weighted_sky(axes, sky_i, pixel_vecs, beam_angles, beam_gains)
            t = local["t_rel_sec"].to_numpy(dtype=float)
            finite = np.isfinite(model)
            if np.count_nonzero(finite) < 8:
                continue
            level = float(np.nanmedian(model[finite]))
            slope = _robust_slope(t, model)
            # A fractional slope is more comparable across frequency maps.
            frac_slope = float(slope / level) if np.isfinite(level) and level != 0 else np.nan
            rows.append(
                {
                    "event_id": ev.event_id,
                    "event_type": ev.event_type,
                    "predicted_event_time": ev.predicted_event_time,
                    "frequency_band": int(band),
                    "frequency_mhz": float(freq),
                    "antenna": LOWER_V,
                    "model_diffuse_level": level,
                    "model_diffuse_abs_frac_slope_per_window": abs(frac_slope) if np.isfinite(frac_slope) else np.nan,
                    "model_diffuse_frac_slope_per_window": frac_slope,
                    "n_model_samples": int(np.count_nonzero(finite)),
                    "beam_model_frequency_mhz": float(beam_freq),
                    "sky_map": str(sky_path),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(input_rows)


def _rank01(values: pd.Series) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    finite = x[np.isfinite(x)]
    out = pd.Series(np.nan, index=values.index, dtype=float)
    if finite.empty:
        return out
    if finite.nunique() == 1:
        out.loc[finite.index] = 0.5
    else:
        out.loc[finite.index] = (finite.rank(method="average") - 1.0) / max(len(finite) - 1, 1)
    return out


def _select_clean_events(metrics: pd.DataFrame, keep_fraction: float, max_level_rank: float, max_slope_rank: float) -> pd.DataFrame:
    pieces = []
    for _, grp in metrics.groupby(["frequency_mhz", "event_type"], sort=True):
        g = grp.copy()
        g["level_rank"] = _rank01(g["model_diffuse_level"])
        g["slope_rank"] = _rank01(g["model_diffuse_abs_frac_slope_per_window"])
        g["beam_clean_score"] = 0.5 * (g["level_rank"] + g["slope_rank"])
        strict = g[g["level_rank"].le(max_level_rank) & g["slope_rank"].le(max_slope_rank)].copy()
        if strict.empty:
            n_keep = max(1, int(np.ceil(len(g) * keep_fraction)))
            strict = g.sort_values("beam_clean_score").head(n_keep).copy()
            strict["beam_clean_selection_mode"] = "score_fallback"
        else:
            strict["beam_clean_selection_mode"] = "level_and_slope_rank"
        pieces.append(strict)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def _plot_grid(summary: pd.DataFrame, out_dir: Path, window_s: float) -> Path:
    freqs = sorted(summary["frequency_mhz"].dropna().unique())
    fig, axes = plt.subplots(len(freqs), 2, figsize=(12, max(10, 1.35 * len(freqs))), sharex=True)
    if len(freqs) == 1:
        axes = np.asarray([axes])
    for i, freq in enumerate(freqs):
        for j, event_type in enumerate(["disappearance", "reappearance"]):
            ax = axes[i, j]
            sub = summary[
                np.isclose(summary["frequency_mhz"], freq)
                & summary["event_type"].astype(str).eq(event_type)
                & summary["antenna"].astype(str).eq(LOWER_V)
            ].sort_values("t_bin_sec")
            if not sub.empty:
                ax.errorbar(
                    sub["t_bin_sec"],
                    sub["median_z_power"],
                    yerr=sub["median_z_power_err"],
                    marker="o",
                    markersize=2.5,
                    linewidth=1.2,
                    elinewidth=0.65,
                    capsize=1.2,
                    color="#d95f02",
                    ecolor="#d95f02",
                )
                ax.text(
                    0.02,
                    0.94,
                    f"median n/bin={np.nanmedian(sub['n_events']):.0f}",
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=7,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.65),
                )
            ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
            ax.axhline(0, color="0.6", linewidth=0.7)
            ax.set_title(f"{freq:.2f} MHz {event_type}", fontsize=9)
            if j == 0:
                ax.set_ylabel("normalized power")
            if i == len(freqs) - 1:
                ax.set_xlabel("seconds from predicted event")
    fig.suptitle(
        "Earth lower-V all-frequency normalized profiles: beam-clean diffuse-Galaxy filter\n"
        "Filter uses digitized beam cuts + PySM sky map; no trendline subtraction",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    path = out_dir / f"earth_lower_v_beam_clean_all_frequency_profile_grid_{int(window_s)}s.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _count_summary(all_events: pd.DataFrame, selected: pd.DataFrame, points: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for freq in sorted(all_events["frequency_mhz"].dropna().unique()):
        for event_type in ["disappearance", "reappearance"]:
            pred = all_events[
                np.isclose(all_events["frequency_mhz"], freq) & all_events["event_type"].astype(str).eq(event_type)
            ]["event_id"].nunique()
            kept = selected[
                np.isclose(selected["frequency_mhz"], freq) & selected["event_type"].astype(str).eq(event_type)
            ]["event_id"].nunique()
            prof = points[
                np.isclose(points.get("frequency_mhz", pd.Series(dtype=float)), freq)
                & points.get("event_type", pd.Series(dtype=object)).astype(str).eq(event_type)
            ]["event_id"].nunique() if not points.empty else 0
            med = np.nan
            if not summary.empty:
                sub = summary[np.isclose(summary["frequency_mhz"], freq) & summary["event_type"].astype(str).eq(event_type)]
                if not sub.empty:
                    med = float(np.nanmedian(sub["n_events"]))
            rows.append(
                {
                    "frequency_mhz": float(freq),
                    "event_type": event_type,
                    "predicted_unique_events": int(pred),
                    "beam_clean_selected_events": int(kept),
                    "events_with_profile_points": int(prof),
                    "median_events_per_time_bin": med,
                }
            )
    return pd.DataFrame(rows)


def _write_report(
    out: Path,
    plot_path: Path,
    model_inputs: pd.DataFrame,
    metrics: pd.DataFrame,
    selected: pd.DataFrame,
    counts: pd.DataFrame,
    keep_fraction: float,
    max_level_rank: float,
    max_slope_rank: float,
) -> None:
    lines = [
        "# Earth Beam-Clean Lower-V Profile Grid",
        "",
        "## Method",
        "",
        "This run uses the digitized Ryle-Vonberg beam cuts directly. For each Earth lower-V event/frequency row, the script:",
        "",
        "1. loads the nearest PySM synchrotron sky map;",
        "2. loads the nearest available digitized Ryle-Vonberg E/H-plane beam cut;",
        "3. treats the mean E/H cut as an axisymmetric power beam;",
        "4. evaluates beam-weighted diffuse sky brightness across the event window;",
        "5. ranks events by diffuse level and absolute fractional diffuse slope;",
        "6. keeps rows with both low level and low slope.",
        "",
        "This is still approximate because the available beam data are 1D cuts, not a full 2D spin-resolved beam.",
        "",
        "## Selection",
        "",
        f"- requested maximum level rank: {max_level_rank:.2f}",
        f"- requested maximum slope rank: {max_slope_rank:.2f}",
        f"- fallback keep fraction, if a group has no strict rows: {keep_fraction:.2f}",
        f"- model rows computed: {len(metrics)}",
        f"- selected event/frequency rows: {len(selected)}",
        f"- selected unique Earth events: {selected['event_id'].nunique()}",
        "",
        "## Beam/Sky Inputs",
        "",
        model_inputs.to_string(index=False),
        "",
        "## Count Summary",
        "",
        counts.to_string(index=False),
        "",
        "## Outputs",
        "",
        f"- `{plot_path}`",
        "- `earth_beam_clean_model_metrics.csv`",
        "- `earth_beam_clean_selected_events.csv`",
        "- `earth_beam_clean_profile_points.csv`",
        "- `earth_beam_clean_profile_summary.csv`",
        "- `earth_beam_clean_count_summary.csv`",
    ]
    (out / "earth_beam_clean_profile_grid_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(ROOT / "outputs/earth_beam_clean_profile_grid_v1"))
    parser.add_argument("--keep-fraction", type=float, default=0.25)
    parser.add_argument("--max-level-rank", type=float, default=0.35)
    parser.add_argument("--max-slope-rank", type=float, default=0.35)
    parser.add_argument("--window-s", type=float, default=WINDOW_S)
    parser.add_argument("--bin-s", type=float, default=BIN_S)
    parser.add_argument("--inner-s", type=float, default=INNER_S)
    args = parser.parse_args()

    out = ensure_dir(Path(args.out_dir))
    clean = _read(CLEAN, parse_dates=["time"])
    events = _read(EARTH_EVENTS, parse_dates=["predicted_event_time"])
    events = events[
        events["source_name"].astype(str).str.lower().eq("earth")
        & events["antenna"].astype(str).eq(LOWER_V)
    ].copy()

    metrics, model_inputs = _event_model_metrics(clean, events, args.window_s)
    selected_metrics = _select_clean_events(metrics, args.keep_fraction, args.max_level_rank, args.max_slope_rank)
    selected_events = events.merge(
        selected_metrics[["event_id", "frequency_band", "frequency_mhz", "beam_clean_score", "level_rank", "slope_rank", "beam_clean_selection_mode"]],
        on=["event_id", "frequency_band", "frequency_mhz"],
        how="inner",
    )
    points = collect_profiles(clean, selected_events, "earth", args.window_s, args.bin_s, args.inner_s)
    points = points[points["antenna"].astype(str).eq(LOWER_V)].copy() if not points.empty else points
    summary = summarize_profiles(points)
    counts = _count_summary(events, selected_metrics, points, summary)

    write_json(
        out / "run_config.json",
        {
            "source_name": "earth",
            "antenna": LOWER_V,
            "window_s": float(args.window_s),
            "bin_s": float(args.bin_s),
            "inner_s": float(args.inner_s),
            "keep_fraction": float(args.keep_fraction),
            "max_level_rank": float(args.max_level_rank),
            "max_slope_rank": float(args.max_slope_rank),
            "beam_model": "axisymmetric mean of digitized E/H plane cuts",
            "model_nside": MODEL_NSIDE,
            "software_versions": software_versions(),
        },
    )
    model_inputs.to_csv(out / "earth_beam_clean_model_inputs.csv", index=False)
    metrics.to_csv(out / "earth_beam_clean_model_metrics.csv", index=False)
    selected_metrics.to_csv(out / "earth_beam_clean_selected_metrics.csv", index=False)
    selected_events.to_csv(out / "earth_beam_clean_selected_events.csv", index=False)
    points.to_csv(out / "earth_beam_clean_profile_points.csv", index=False)
    summary.to_csv(out / "earth_beam_clean_profile_summary.csv", index=False)
    counts.to_csv(out / "earth_beam_clean_count_summary.csv", index=False)
    plot_path = _plot_grid(summary, out, args.window_s)
    _write_report(out, plot_path, model_inputs, metrics, selected_metrics, counts, args.keep_fraction, args.max_level_rank, args.max_slope_rank)
    print(plot_path)
    print(out / "earth_beam_clean_profile_grid_report.md")


if __name__ == "__main__":
    main()
