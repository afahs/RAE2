#!/usr/bin/env python
"""Extended-source lunar occultation toy model with MIE response and RAE sampling.

This diagnostic combines three effects that are easy to confuse in the RAE-2
profiles:

1. a local MIE point-source lunar-shadow response, configured with
   ``RAE2_MIE_DIR`` or ``RAE2_MIE_POINT_CSV``;
2. finite angular source size, using a uniform disk with the apparent source
   radius as seen from the Moon;
3. actual lower-V Ryle-Vonberg sample times around predicted source events.

The goal is not absolute calibration.  The goal is to test whether these
ingredients can naturally smear an occultation edge into a multi-minute
transition even before Galactic background or receiver terms are added.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.util import datetime_ns, ensure_dir, robust_sigma, software_versions, write_json  # noqa: E402


MIE_DIR = Path(os.environ.get("RAE2_MIE_DIR", "data/mie"))
MIE_POINT = Path(os.environ.get("RAE2_MIE_POINT_CSV", MIE_DIR / "figures/lunar_source_occultation_700khz.csv"))
BEAM_DIR = Path(os.environ.get("RAE2_ANTENNA_DIGITIZATION_DIR", "data/antennaDigitization"))
CLEAN = ROOT / "outputs/control_survey_earth_sun_postnov1974_v2/01_ingest/cleaned_timeseries.csv"
EVENTS = ROOT / "outputs/control_survey_earth_sun_postnov1974_v2/02_events/predicted_events.csv"
OUT = ROOT / "outputs/extended_source_mie_sampling_forward_model_v1"
DEFAULT_MODEL_RADIUS_DEG = {
    "sun_disk": 0.266,
    "earth_disk": float(np.degrees(np.arcsin(6378.137 / 384400.0))),
    "point_source": 0.0,
}

BEAM_SPECS = [
    (1.31, BEAM_DIR / "eplane_1p31MHz.csv", BEAM_DIR / "hplane_1p31MHz.csv"),
    (3.93, BEAM_DIR / "eplane_3p93MHz.csv", BEAM_DIR / "hplane_3p93MHz.csv"),
    (6.55, BEAM_DIR / "eplane_6p55MHz.csv", BEAM_DIR / "hplane_6p55MHz.csv"),
]
EXPECTED_EVENT_TYPES = ["disappearance", "reappearance"]


def _read(path: Path, **kwargs) -> pd.DataFrame:
    return read_table(path, low_memory=False, **kwargs)


def _nearest_beam(freq_mhz: float) -> tuple[float, Path, Path]:
    return min(BEAM_SPECS, key=lambda spec: abs(spec[0] - float(freq_mhz)))


def _load_beam(freq_mhz: float) -> tuple[float, np.ndarray, np.ndarray]:
    beam_freq, eplane, hplane = _nearest_beam(freq_mhz)
    e = _read(eplane)
    h = _read(hplane)
    angles = pd.to_numeric(e["angle_deg"], errors="coerce").to_numpy(dtype=float)
    gain_e = 10.0 ** (pd.to_numeric(e["gain_dB"], errors="coerce").to_numpy(dtype=float) / 10.0)
    gain_h = 10.0 ** (pd.to_numeric(h["gain_dB"], errors="coerce").to_numpy(dtype=float) / 10.0)
    return beam_freq, angles, 0.5 * (gain_e + gain_h)


def _interp_cyclic(angles: np.ndarray, values: np.ndarray, angle_deg: np.ndarray) -> np.ndarray:
    a = np.asarray(angle_deg, dtype=float) % 360.0
    x = np.concatenate(([angles[-1] - 360.0], angles, [angles[0] + 360.0]))
    y = np.concatenate(([values[-1]], values, [values[0]]))
    return np.interp(a, x, y)


def _load_mie_transmission() -> tuple[np.ndarray, np.ndarray]:
    df = _read(MIE_POINT)
    x = pd.to_numeric(df["source_position_from_limb_deg"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df["observed_over_intrinsic"], errors="coerce").to_numpy(dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    order = np.argsort(x[good])
    return x[good][order], np.clip(y[good][order], 0.0, np.inf)


def _transmission_from_limb_coordinate(x_deg: np.ndarray, mie_x: np.ndarray, mie_t: np.ndarray) -> np.ndarray:
    # Negative x is outside the geometric lunar disk. Positive x is behind the
    # limb.  Clamp far outside to unocculted and far inside to the deepest
    # available MIE value rather than extrapolating oscillations.
    x = np.asarray(x_deg, dtype=float)
    return np.interp(x, mie_x, mie_t, left=float(np.nanmedian(mie_t[mie_x < -20.0])), right=float(mie_t[-1]))


def _disk_samples(radius_deg: float, radial_steps: int = 34, az_steps: int = 96) -> tuple[np.ndarray, np.ndarray]:
    if float(radius_deg) <= 0.0:
        return np.asarray([0.0], dtype=float), np.asarray([1.0], dtype=float)
    # Equal-area radial rings.  Only the radial-inward coordinate affects a
    # straight local limb, but the tangential coordinate is kept in the weights
    # by sampling the full disk.
    radii = radius_deg * np.sqrt((np.arange(radial_steps, dtype=float) + 0.5) / radial_steps)
    phis = 2.0 * np.pi * (np.arange(az_steps, dtype=float) + 0.5) / az_steps
    rr, pp = np.meshgrid(radii, phis, indexing="ij")
    inward = (rr * np.cos(pp)).reshape(-1)
    weights = np.full(inward.size, 1.0 / inward.size)
    return inward, weights


def _extended_disk_model(
    t_rel_s: np.ndarray,
    event_type: str,
    angular_speed_deg_s: float,
    moon_radius_deg: float,
    source_radius_deg: float,
    mie_x: np.ndarray,
    mie_t: np.ndarray,
    beam_angles: np.ndarray | None = None,
    beam_gain: np.ndarray | None = None,
) -> np.ndarray:
    t = np.asarray(t_rel_s, dtype=float)
    sign = 1.0 if event_type == "disappearance" else -1.0
    center_x = sign * float(angular_speed_deg_s) * t
    inward_offsets, weights = _disk_samples(source_radius_deg)
    x = center_x[:, None] + inward_offsets[None, :]
    trans = _transmission_from_limb_coordinate(x, mie_x, mie_t)
    if beam_angles is None or beam_gain is None:
        return np.sum(trans * weights[None, :], axis=1)
    theta = float(moon_radius_deg) - x
    gain = _interp_cyclic(beam_angles, beam_gain, theta)
    numer = np.sum(trans * gain * weights[None, :], axis=1)
    denom = np.sum(gain * weights[None, :], axis=1)
    return np.divide(numer, denom, out=np.full_like(numer, np.nan), where=denom > 0)


def _point_model(
    t_rel_s: np.ndarray,
    event_type: str,
    angular_speed_deg_s: float,
    mie_x: np.ndarray,
    mie_t: np.ndarray,
) -> np.ndarray:
    sign = 1.0 if event_type == "disappearance" else -1.0
    return _transmission_from_limb_coordinate(sign * float(angular_speed_deg_s) * np.asarray(t_rel_s), mie_x, mie_t)


def _geometric_disk_model(
    t_rel_s: np.ndarray,
    event_type: str,
    angular_speed_deg_s: float,
    source_radius_deg: float,
) -> np.ndarray:
    if float(source_radius_deg) <= 0.0:
        sign = 1.0 if event_type == "disappearance" else -1.0
        center_x = sign * float(angular_speed_deg_s) * np.asarray(t_rel_s, dtype=float)
        return (center_x <= 0.0).astype(float)
    sign = 1.0 if event_type == "disappearance" else -1.0
    center_x = sign * float(angular_speed_deg_s) * np.asarray(t_rel_s, dtype=float)
    inward_offsets, weights = _disk_samples(source_radius_deg)
    visible = (center_x[:, None] + inward_offsets[None, :]) <= 0.0
    return np.sum(visible * weights[None, :], axis=1)


def _event_speed(events: pd.DataFrame) -> pd.Series:
    pre = pd.to_numeric(events["pre_limb_angle_deg"], errors="coerce")
    post = pd.to_numeric(events["post_limb_angle_deg"], errors="coerce")
    gap = pd.to_numeric(events["gap_seconds"], errors="coerce")
    speed = (post - pre).abs() / gap
    speed = speed.where(np.isfinite(speed) & speed.gt(0))
    return speed


def _build_groups(clean: pd.DataFrame) -> dict[tuple[int, str], tuple[pd.DataFrame, np.ndarray]]:
    out = {}
    clean = clean.copy()
    clean["time"] = pd.to_datetime(clean["time"], errors="coerce")
    for key, grp in clean.groupby(["frequency_band", "antenna"], sort=True):
        g = grp.sort_values("time").reset_index(drop=True)
        out[(int(key[0]), str(key[1]))] = (g, datetime_ns(g["time"]))
    return out


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
    local["t_rel_sec"] = (datetime_ns(local["time"]) - event_ns).astype(float) / 1e9
    local = local[np.abs(local["t_rel_sec"]) <= float(window_s)].copy()
    if "is_valid" in local.columns:
        local = local[local["is_valid"].astype(bool)].copy()
    return local


def _normalize_profile(y: np.ndarray) -> np.ndarray:
    arr = np.asarray(y, dtype=float)
    base = np.nanmedian(arr)
    scale = robust_sigma(arr - base)
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.nanstd(arr))
    if not np.isfinite(scale) or scale <= 0:
        return arr - base
    return (arr - base) / scale


def _bin_sampled_models(rows: pd.DataFrame, bin_s: float, window_s: float) -> pd.DataFrame:
    bins = np.arange(-float(window_s), float(window_s) + float(bin_s), float(bin_s))
    out = []
    value_cols = ["point_mie", "geometric_disk", "extended_disk_mie", "extended_disk_mie_beam"]
    rows = rows.copy()
    rows["t_bin_sec"] = pd.cut(rows["t_rel_sec"], bins=bins, labels=0.5 * (bins[:-1] + bins[1:]))
    for keys, grp in rows.groupby(["event_type", "frequency_band", "frequency_mhz", "t_bin_sec"], sort=True, observed=True):
        event_type, band, freq, t_bin = keys
        rec = {
            "event_type": event_type,
            "frequency_band": int(band),
            "frequency_mhz": float(freq),
            "t_bin_sec": float(t_bin),
            "n_events": int(grp["event_id"].nunique()),
            "n_samples": int(len(grp)),
        }
        for col in value_cols:
            vals = pd.to_numeric(grp[col], errors="coerce")
            vals = vals[np.isfinite(vals)]
            if vals.empty:
                rec[f"{col}_median"] = np.nan
                rec[f"{col}_err"] = np.nan
            else:
                rec[f"{col}_median"] = float(vals.median())
                sig = robust_sigma(vals.to_numpy(dtype=float) - float(vals.median()))
                rec[f"{col}_err"] = float(sig / np.sqrt(len(vals))) if np.isfinite(sig) and sig > 0 else np.nan
        out.append(rec)
    return pd.DataFrame(out)


def _transition_width(t: np.ndarray, y: np.ndarray, event_type: str) -> tuple[float, float, float]:
    order = np.argsort(t)
    tt = np.asarray(t, dtype=float)[order]
    yy = np.asarray(y, dtype=float)[order]
    good = np.isfinite(tt) & np.isfinite(yy)
    tt = tt[good]
    yy = yy[good]
    if len(tt) < 4:
        return np.nan, np.nan, np.nan
    if event_type == "reappearance":
        yy = yy[::-1]
        tt = -tt[::-1]
    high = float(np.nanpercentile(yy[tt < -0.5 * np.nanmax(np.abs(tt))], 50)) if np.any(tt < 0) else float(np.nanmax(yy))
    low = float(np.nanpercentile(yy[tt > 0.5 * np.nanmax(np.abs(tt))], 50)) if np.any(tt > 0) else float(np.nanmin(yy))
    amp = high - low
    if not np.isfinite(amp) or amp == 0:
        return np.nan, np.nan, np.nan
    y90 = low + 0.9 * amp
    y10 = low + 0.1 * amp
    # Disappearance-like after any reappearance reversal: curve declines with t.
    def crossings(level: float) -> np.ndarray:
        diff = yy - level
        idx = np.where(np.signbit(diff[:-1]) != np.signbit(diff[1:]))[0]
        if idx.size == 0:
            return np.array([], dtype=float)
        vals = []
        for i in idx:
            i = int(i)
            if yy[i + 1] == yy[i]:
                vals.append(float(tt[i]))
            else:
                frac = (level - yy[i]) / (yy[i + 1] - yy[i])
                vals.append(float(tt[i] + frac * (tt[i + 1] - tt[i])))
        return np.asarray(vals, dtype=float)

    # MIE produces small pre-limb ripples.  For an edge-duration diagnostic we
    # want the central limb transition, not the first far-pre-event ripple that
    # happens to cross the same level.  Choose the crossing nearest the
    # predicted geometric limb, then the matching lower-level crossing after it.
    c90 = crossings(y90)
    c10 = crossings(y10)
    if c90.size == 0 or c10.size == 0:
        return np.nan, np.nan, np.nan
    t90 = float(c90[np.argmin(np.abs(c90))])
    after = c10[c10 >= t90]
    t10 = float(after[np.argmin(np.abs(after - t90))]) if after.size else float(c10[np.argmin(np.abs(c10))])
    return float(abs(t10 - t90)) if np.isfinite(t10) and np.isfinite(t90) else np.nan, t90, t10


def _plot_continuous(
    curves: pd.DataFrame,
    sampled_summary: pd.DataFrame,
    event_source: str,
    model_source: str,
    freq_mhz: float,
    out_dir: Path,
    window_s: float,
) -> Path:
    event_label = event_source.replace("_", " ").title()
    model_label = model_source.replace("_", " ").title()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, constrained_layout=True)
    cols = [
        ("point_mie", "point MIE"),
        ("geometric_disk", "extended disk, geometric"),
        ("extended_disk_mie", "extended disk + MIE"),
        ("extended_disk_mie_beam", "extended disk + MIE + lower-V beam"),
    ]
    for j, event_type in enumerate(EXPECTED_EVENT_TYPES):
        cont = curves[curves["event_type"].eq(event_type)].sort_values("t_rel_sec")
        samp = sampled_summary[
            sampled_summary["event_type"].eq(event_type)
            & np.isclose(sampled_summary["frequency_mhz"].astype(float), float(freq_mhz))
        ].sort_values("t_bin_sec")
        for ax, (col, label) in zip(axes[j], cols[:2]):
            ax.plot(cont["t_rel_sec"] / 60.0, cont[col], lw=1.7, label=f"continuous {label}")
            ycol = f"{col}_median"
            ecol = f"{col}_err"
            if ycol in samp:
                ax.errorbar(
                    samp["t_bin_sec"] / 60.0,
                    samp[ycol],
                    yerr=samp.get(ecol),
                    marker="o",
                    markersize=2.2,
                    linewidth=0.9,
                    capsize=1.0,
                    label="actual RAE sampling, binned",
                )
            ax.axvline(0, color="black", ls="--", lw=0.8)
            ax.set_title(f"{event_type}: {label}")
            ax.set_ylabel("relative source contribution")
            ax.grid(alpha=0.25)
        # Put the two physically combined curves on the second column by
        # overplotting them for easy comparison.
        ax = axes[j, 1]
        for col, label, color in [
            ("extended_disk_mie", "extended disk + MIE", "#4c78a8"),
            ("extended_disk_mie_beam", "extended disk + MIE + lower-V beam", "#d95f02"),
        ]:
            ax.plot(cont["t_rel_sec"] / 60.0, cont[col], lw=1.5, color=color, alpha=0.85, label=label)
            ycol = f"{col}_median"
            if ycol in samp:
                ax.plot(samp["t_bin_sec"] / 60.0, samp[ycol], marker=".", lw=0.8, color=color, alpha=0.85)
        ax.legend(frameon=False, fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("minutes from predicted limb crossing")
    fig.suptitle(f"{model_label} occultation model on {event_label} event times at {freq_mhz:.2f} MHz, +/-{window_s/60:.0f} min")
    freq_label = f"{freq_mhz:.2f}".replace(".", "p")
    path = out_dir / f"{model_source}_on_{event_source}_events_mie_sampling_continuous_vs_sampled_{freq_label}mhz.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_all_frequency_sampled(summary: pd.DataFrame, event_source: str, model_source: str, out_dir: Path, window_s: float) -> Path:
    event_label = event_source.replace("_", " ").title()
    model_label = model_source.replace("_", " ").title()
    freqs = sorted(summary["frequency_mhz"].dropna().unique())
    fig, axes = plt.subplots(len(freqs), 2, figsize=(12, max(10, 1.25 * len(freqs))), sharex=True)
    if len(freqs) == 1:
        axes = np.asarray([axes])
    for i, freq in enumerate(freqs):
        for j, event_type in enumerate(EXPECTED_EVENT_TYPES):
            ax = axes[i, j]
            sub = summary[np.isclose(summary["frequency_mhz"], freq) & summary["event_type"].eq(event_type)].sort_values(
                "t_bin_sec"
            )
            series = [("point_mie_median", "point MIE", "0.45")]
            if model_source != "point_source":
                series.extend(
                    [
                        ("extended_disk_mie_median", "extended disk + MIE", "#4c78a8"),
                        ("extended_disk_mie_beam_median", "extended disk + MIE + beam", "#d95f02"),
                    ]
                )
            for col, label, color in series:
                if col not in sub:
                    continue
                ax.plot(sub["t_bin_sec"] / 60.0, sub[col], marker="o", markersize=2.0, lw=0.95, color=color, label=label)
            ax.axvline(0, color="black", ls="--", lw=0.75)
            ax.set_title(f"{freq:.2f} MHz {event_type}", fontsize=9)
            ax.grid(alpha=0.22)
            if j == 0:
                ax.set_ylabel("relative source contribution")
            if i == len(freqs) - 1:
                ax.set_xlabel("minutes from predicted event")
            if i == 0 and j == 1:
                ax.legend(frameon=False, fontsize=7)
    fig.suptitle(f"Actual lower-V sampling: {model_label} model on {event_label} event times, +/-{window_s/60:.0f} min", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    path = out_dir / f"{model_source}_on_{event_source}_events_mie_actual_sampling_all_frequency_grid_{int(window_s)}s.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_cadence(sampled: pd.DataFrame, event_source: str, model_source: str, out_dir: Path) -> Path:
    event_label = event_source.replace("_", " ").title()
    model_label = model_source.replace("_", " ").title()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    per_event = sampled.groupby(["frequency_mhz", "event_id"], as_index=False).agg(
        n_samples=("t_rel_sec", "size"),
        event_type=("event_type", "first"),
        angular_speed_deg_s=("angular_speed_deg_s", "median"),
        transition_time_s=("extended_disk_geometric_transition_s", "median"),
    )
    axes[0].hist(per_event["n_samples"], bins=40, color="#4c78a8", alpha=0.85)
    axes[0].set_xlabel("valid samples in +/- window")
    axes[0].set_ylabel("event count")
    axes[0].set_title(f"RAE samples per {event_label} event window")
    axes[1].scatter(
        per_event["angular_speed_deg_s"] * 60.0,
        per_event["transition_time_s"] / 60.0,
        s=8,
        alpha=0.35,
        color="#d95f02",
    )
    axes[1].set_xlabel("local limb speed (deg/min)")
    axes[1].set_ylabel("extended-disk geometric crossing time (min)")
    axes[1].set_title("Finite source size timescale")
    axes[1].grid(alpha=0.25)
    path = out_dir / f"{model_source}_on_{event_source}_events_mie_sampling_cadence_and_duration.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _write_report(
    out_dir: Path,
    config: dict,
    event_summary: pd.DataFrame,
    transition_summary: pd.DataFrame,
    paths: list[Path],
) -> Path:
    event_label = str(config.get("event_source", "source")).replace("_", " ").title()
    model_label = str(config.get("model_source", "model")).replace("_", " ").title()
    lines = [
        f"# {model_label} MIE + Sampling Forward Model",
        "",
        "## Purpose",
        "",
        f"This diagnostic asks how a `{model_label}` source model looks when sampled at predicted {event_label} occultation times.",
        "would look like a sharp step after applying the local MIE lunar-shadow response and the actual",
        "RAE-2 lower-V sampling times.",
        "",
        "It is a source-only model. It does not include diffuse Galactic pickup, moving-source contamination,",
        "receiver drift, quantization, or absolute flux calibration.",
        "",
        "## Configuration",
        "",
        pd.Series(config).to_string(),
        "",
        "## Event / Sampling Counts",
        "",
        event_summary.to_string(index=False) if not event_summary.empty else "No sampled events.",
        "",
        "## Transition Widths",
        "",
        transition_summary.to_string(index=False) if not transition_summary.empty else "No transition rows.",
        "",
        "## Interpretation",
        "",
        "- A point-source MIE curve remains a relatively compact edge on the RAE event scale.",
    ]
    if config.get("model_source") == "point_source":
        lines.extend(
            [
                "- This run is the generic compact-source limit: `source_radius_deg = 0`.",
                "- Any broadening in this plot is from the MIE response and actual RAE sampling, not finite source size.",
            ]
        )
    else:
        lines.extend(
            [
                "- An extended disk broadens the edge by roughly `2 * source_angular_radius / limb_speed`.",
                f"- For the median {event_label}-event limb speed in this dataset, that finite-disk crossing time is listed above.",
            ]
        )
    lines.extend(
        [
        "- Actual RAE sampling can make that broadened transition look sparse or line-like in individual events,",
        "  especially when only one small sample group falls near the event.",
        "- This model can explain some visual smoothing, but by itself it should preserve the ordinary occultation",
        "  sign: disappearance decreases and reappearance increases. It should not create the persistent opposite-sign",
        "  low-frequency Earth/Sun behavior seen elsewhere.",
        "",
        "## Plots",
        "",
        ]
    )
    lines.extend(f"- `{path}`" for path in paths)
    path = out_dir / f"{config.get('model_source', 'model')}_on_{config.get('event_source', 'source')}_events_mie_sampling_forward_model_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run(args: argparse.Namespace) -> Path:
    event_source = str(args.source).strip().lower()
    model_source = str(args.model_source or f"{event_source}_disk").strip().lower()
    if event_source not in ["sun", "earth"]:
        raise ValueError(f"unsupported event source for this diagnostic: {event_source}")
    if model_source not in DEFAULT_MODEL_RADIUS_DEG:
        raise ValueError(f"unsupported model source for this diagnostic: {model_source}")
    source_radius_deg = (
        float(args.source_radius_deg)
        if args.source_radius_deg is not None
        else float(DEFAULT_MODEL_RADIUS_DEG[model_source])
    )
    out_dir = ensure_dir(
        args.out_dir
        or (ROOT / f"outputs/{model_source}_on_{event_source}_events_mie_sampling_forward_model_v1")
    )
    mie_x, mie_t = _load_mie_transmission()
    clean = _read(CLEAN)
    events = _read(EVENTS)
    clean = clean[clean["antenna"].astype(str).eq(args.antenna)].copy()
    events = events[
        events["source_name"].astype(str).str.lower().eq(event_source)
        & events["antenna"].astype(str).eq(args.antenna)
    ].copy()
    if args.frequency_bands:
        bands = [int(x) for x in args.frequency_bands.split(",") if x.strip()]
        clean = clean[clean["frequency_band"].isin(bands)].copy()
        events = events[events["frequency_band"].isin(bands)].copy()
    groups = _build_groups(clean)
    t_cont = np.linspace(-float(args.window_s), float(args.window_s), 1201)
    point_rows = []
    sampled_rows = []
    transition_rows = []

    unique_events = events.drop_duplicates(["event_id", "event_type", "predicted_event_time"]).copy()
    speeds = _event_speed(unique_events)
    representative_speed = float(speeds.dropna().median())
    representative_radius = float(pd.to_numeric(unique_events["moon_angular_radius_deg"], errors="coerce").dropna().median())
    beam_freq, beam_angles, beam_gain = _load_beam(float(args.representative_frequency_mhz))
    for event_type in EXPECTED_EVENT_TYPES:
        point = _point_model(t_cont, event_type, representative_speed, mie_x, mie_t)
        geom = _geometric_disk_model(t_cont, event_type, representative_speed, source_radius_deg)
        disk = _extended_disk_model(
            t_cont,
            event_type,
            representative_speed,
            representative_radius,
            source_radius_deg,
            mie_x,
            mie_t,
        )
        disk_beam = _extended_disk_model(
            t_cont,
            event_type,
            representative_speed,
            representative_radius,
            source_radius_deg,
            mie_x,
            mie_t,
            beam_angles,
            beam_gain,
        )
        width, t90, t10 = _transition_width(t_cont, disk, event_type)
        transition_rows.append(
            {
                "model": "continuous_representative_extended_disk_mie",
                "event_type": event_type,
                "frequency_mhz": float(args.representative_frequency_mhz),
                "limb_speed_deg_per_min": representative_speed * 60.0,
                "source_radius_deg": source_radius_deg,
                "transition_width_90_to_10_s": width,
                "t90_s": t90,
                "t10_s": t10,
            }
        )
        for t, p, g, d, db in zip(t_cont, point, geom, disk, disk_beam):
            point_rows.append(
                {
                    "event_type": event_type,
                    "t_rel_sec": float(t),
                    "point_mie": float(p),
                    "geometric_disk": float(g),
                    "extended_disk_mie": float(d),
                    "extended_disk_mie_beam": float(db),
                    "representative_limb_speed_deg_s": representative_speed,
                    "representative_moon_radius_deg": representative_radius,
                    "beam_model_frequency_mhz": beam_freq,
                }
            )

    events["angular_speed_deg_s"] = _event_speed(events)
    events["predicted_event_time"] = pd.to_datetime(events["predicted_event_time"], errors="coerce")
    for (band, freq), ev_group in events.groupby(["frequency_band", "frequency_mhz"], sort=True):
        payload = groups.get((int(band), str(args.antenna)))
        if payload is None:
            continue
        group, group_ns = payload
        beam_freq, beam_angles, beam_gain = _load_beam(float(freq))
        evs = ev_group.sort_values("predicted_event_time")
        if int(args.max_events_per_band_type) > 0:
            evs = evs.groupby("event_type", group_keys=False).head(int(args.max_events_per_band_type))
        for _, ev in evs.iterrows():
            speed = float(ev.get("angular_speed_deg_s", np.nan))
            if not np.isfinite(speed) or speed <= 0:
                continue
            local = _event_window(group, group_ns, pd.Timestamp(ev["predicted_event_time"]), float(args.window_s))
            if local.empty:
                continue
            t = local["t_rel_sec"].to_numpy(dtype=float)
            radius = float(ev.get("moon_angular_radius_deg", representative_radius))
            point = _point_model(t, str(ev["event_type"]), speed, mie_x, mie_t)
            geom = _geometric_disk_model(t, str(ev["event_type"]), speed, source_radius_deg)
            disk = _extended_disk_model(
                t,
                str(ev["event_type"]),
                speed,
                radius,
                source_radius_deg,
                mie_x,
                mie_t,
            )
            disk_beam = _extended_disk_model(
                t,
                str(ev["event_type"]),
                speed,
                radius,
                source_radius_deg,
                mie_x,
                mie_t,
                beam_angles,
                beam_gain,
            )
            geom_transition_s = 2.0 * source_radius_deg / speed
            for tt, p, g, d, db in zip(t, point, geom, disk, disk_beam):
                sampled_rows.append(
                    {
                        "event_id": int(ev["event_id"]),
                        "event_type": str(ev["event_type"]),
                        "predicted_event_time": ev["predicted_event_time"],
                        "frequency_band": int(band),
                        "frequency_mhz": float(freq),
                        "antenna": str(args.antenna),
                        "t_rel_sec": float(tt),
                        "angular_speed_deg_s": speed,
                        "moon_radius_deg": radius,
                        "extended_disk_geometric_transition_s": geom_transition_s,
                        "point_mie": float(p),
                        "geometric_disk": float(g),
                        "extended_disk_mie": float(d),
                        "extended_disk_mie_beam": float(db),
                        "beam_model_frequency_mhz": float(beam_freq),
                    }
                )

    curves = pd.DataFrame(point_rows)
    sampled = pd.DataFrame(sampled_rows)
    summary = _bin_sampled_models(sampled, float(args.bin_s), float(args.window_s)) if not sampled.empty else pd.DataFrame()
    prefix = f"{model_source}_on_{event_source}_events"
    curves.to_csv(out_dir / f"continuous_representative_{prefix}_mie_model.csv", index=False)
    sampled.to_csv(out_dir / f"actual_rae_sampled_{prefix}_mie_model_points.csv", index=False)
    summary.to_csv(out_dir / f"actual_rae_sampled_{prefix}_mie_model_summary.csv", index=False)
    transition_summary = pd.DataFrame(transition_rows)
    if not sampled.empty:
        per_event = sampled.drop_duplicates(["frequency_mhz", "event_id", "event_type"])
        transition_summary = pd.concat(
            [
                transition_summary,
                per_event.groupby(["frequency_mhz", "event_type"], as_index=False).agg(
                    model=("event_type", lambda _: "actual_event_extended_disk_geometric"),
                    n_events=("event_id", "nunique"),
                    median_limb_speed_deg_per_min=("angular_speed_deg_s", lambda s: float(np.nanmedian(s) * 60.0)),
                    median_source_size_crossing_s=("extended_disk_geometric_transition_s", "median"),
                    p10_source_size_crossing_s=("extended_disk_geometric_transition_s", lambda s: float(np.nanpercentile(s, 10))),
                    p90_source_size_crossing_s=("extended_disk_geometric_transition_s", lambda s: float(np.nanpercentile(s, 90))),
                ),
            ],
            ignore_index=True,
            sort=False,
        )
    transition_summary.to_csv(out_dir / f"{prefix}_mie_transition_widths.csv", index=False)
    event_summary = (
        sampled.groupby(["frequency_band", "frequency_mhz", "event_type"], as_index=False)
        .agg(
            n_events=("event_id", "nunique"),
            n_samples=("t_rel_sec", "size"),
            median_samples_per_event=("event_id", lambda s: float(len(s) / max(1, s.nunique()))),
            median_limb_speed_deg_per_min=("angular_speed_deg_s", lambda s: float(np.nanmedian(s) * 60.0)),
            median_source_size_crossing_s=("extended_disk_geometric_transition_s", "median"),
        )
        if not sampled.empty
        else pd.DataFrame()
    )
    event_summary.to_csv(out_dir / f"{prefix}_mie_sampling_event_summary.csv", index=False)

    paths = []
    if not summary.empty:
        paths.append(
            _plot_continuous(
                curves,
                summary,
                event_source,
                model_source,
                float(args.representative_frequency_mhz),
                out_dir,
                float(args.window_s),
            )
        )
        paths.append(_plot_all_frequency_sampled(summary, event_source, model_source, out_dir, float(args.window_s)))
    if not sampled.empty:
        paths.append(_plot_cadence(sampled, event_source, model_source, out_dir))
    config = {
        "event_source": event_source,
        "model_source": model_source,
        "mie_point_source_file": str(MIE_POINT),
        "mie_frequency_mhz": 0.70,
        "cleaned_timeseries": str(CLEAN),
        "predicted_events": str(EVENTS),
        "antenna": str(args.antenna),
        "source_radius_deg": source_radius_deg,
        "source_diameter_deg": 2.0 * source_radius_deg,
        "window_s": float(args.window_s),
        "bin_s": float(args.bin_s),
        "representative_frequency_mhz": float(args.representative_frequency_mhz),
        "max_events_per_band_type": int(args.max_events_per_band_type),
        "software_versions": software_versions(),
    }
    write_json(out_dir / "run_config.json", config)
    report = _write_report(out_dir, config, event_summary, transition_summary, paths)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["earth", "sun"], default="sun", help="Event schedule to sample.")
    parser.add_argument(
        "--model-source",
        choices=sorted(DEFAULT_MODEL_RADIUS_DEG),
        default="",
        help="Physical source model. Defaults to <source>_disk; use point_source for a generic compact source.",
    )
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--antenna", default="rv2_coarse")
    parser.add_argument(
        "--source-radius-deg",
        type=float,
        default=None,
        help="Angular radius from the Moon. Defaults: Sun=0.266 deg, Earth=arcsin(Rearth/Moon distance).",
    )
    parser.add_argument("--window-s", type=float, default=1200.0)
    parser.add_argument("--bin-s", type=float, default=60.0)
    parser.add_argument("--representative-frequency-mhz", type=float, default=0.70)
    parser.add_argument("--frequency-bands", default="", help="Optional comma-separated subset, e.g. 2 for 0.70 MHz.")
    parser.add_argument(
        "--max-events-per-band-type",
        type=int,
        default=160,
        help="Cap per frequency/event type for runtime; <=0 uses every available event.",
    )
    args = parser.parse_args()
    report = run(args)
    print(report)


if __name__ == "__main__":
    main()
