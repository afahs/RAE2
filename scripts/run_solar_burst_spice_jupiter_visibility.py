#!/usr/bin/env python
"""External solar-burst, SPICE Jupiter phase, and direct-visible Jupiter tests.

This script deliberately separates three physically different questions:

1. Solar occultation events selected by independent RSTN solar radio burst
   reports.
2. Jupiter occultation events folded with SPICE-derived CML/Io geometry.
3. A non-occultation Jupiter scan: when Jupiter is visible to lower V, does the
   raw power or high-power sample rate change, especially while Earth is
   occulted by the Moon?

All detection-style statistics use lower V only and raw powers/contrasts.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import re
import sys
import urllib.request

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rylevonberg.table_io import read_table

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.constants import FREQUENCY_MAP_MHZ  # noqa: E402
from rylevonberg.util import ensure_dir, robust_sigma, software_versions, write_json  # noqa: E402
from scripts.run_prepost_rank_detection import (  # noqa: E402
    _binom_sf_one_sided,
    _cliffs_delta,
    _mannwhitney_greater,
)

try:  # noqa: E402
    import spiceypy as spice
except Exception:  # pragma: no cover - tested through runtime report
    spice = None


DEFAULT_FEATURES = (
    ROOT
    / "outputs/lower_v_conditional_event_detection_sun_jupiter_v1/"
    / "conditional_selected_event_features.csv"
)
DEFAULT_CLEAN = ROOT / "outputs/control_survey_earth_sun_postnov1974_v2/01_ingest/cleaned_timeseries.csv"
DEFAULT_JUPITER_STATES = ROOT / "outputs/planetary_confirmation_survey_science_baseline_v2/events/jupiter_limb_visibility_states.csv"
DEFAULT_EARTH_STATES = ROOT / "outputs/planetary_confirmation_survey_science_baseline_v2/events/earth_limb_visibility_states.csv"
DEFAULT_OUT = ROOT / "outputs/lower_v_solar_burst_spice_jupiter_visibility_v1"

RSTN_FIXED_URL_TEMPLATE = (
    "https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-features/"
    "solar-radio/radio-bursts/reports/fixed-frequency-listings/"
    "solar-radio_bursts_fixed-frequency_{year}.txt"
)
NAIF_KERNELS = {
    "naif0012.tls": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls",
    "pck00010.tpc": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00010.tpc",
    "jup100.bsp": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/a_old_versions/jup100.bsp",
    "jup348.bsp": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/jup348.bsp",
}

ANTENNA = "rv2_coarse"
SOURCE_LABEL = {"sun": "Sun", "jupiter": "Jupiter"}
MODE_COLOR = {
    "all_events": "#4d4d4d",
    "rstn_near_1h": "#b2182b",
    "rstn_near_6h": "#d95f02",
    "rstn_high_day": "#e6ab02",
    "spice_io_abc_gate": "#1b9e77",
    "spice_io_abc_or_reversed_gate": "#7570b3",
}


def _download(url: str, path: Path, refresh: bool = False) -> Path:
    if refresh or not path.exists():
        ensure_dir(path.parent)
        with urllib.request.urlopen(url, timeout=180) as response:
            path.write_bytes(response.read())
    return path


def parse_rstn_time_token(token: object) -> float | None:
    """Parse NOAA RSTN HHMM or HHMMt tokens into decimal hours."""
    s = re.sub(r"[^0-9]", "", str(token or ""))
    if len(s) < 4:
        return None
    hour = int(s[:2])
    minute = int(s[2:4])
    if hour > 23 or minute > 59:
        return None
    frac_min = 0.0
    if len(s) >= 5:
        frac_min = int(s[4]) / 10.0
    return hour + (minute + frac_min) / 60.0


def parse_rstn_fixed_frequency_line(line: str) -> dict[str, object] | None:
    parts = line.split()
    if len(parts) < 3 or not re.fullmatch(r"\d{6}", parts[0]):
        return None
    yy = int(parts[0][:2])
    year = 1900 + yy if yy >= 50 else 2000 + yy
    month = int(parts[0][2:4])
    day = int(parts[0][4:6])
    m = re.match(r"(\d+)([A-Za-z]+)?", parts[1])
    if not m:
        return None
    start_h = parse_rstn_time_token(parts[2])
    end_h = parse_rstn_time_token(parts[3]) if len(parts) > 3 else None
    if start_h is None:
        return None
    date = pd.Timestamp(year=year, month=month, day=day)
    start = date + pd.to_timedelta(start_h, unit="h")
    if end_h is not None:
        end = date + pd.to_timedelta(end_h, unit="h")
        if end < start:
            end = end + pd.Timedelta(days=1)
    else:
        end = pd.NaT
    duration_min = np.nan
    if len(parts) > 4:
        dur_raw = re.sub(r"[^0-9.]", "", parts[4])
        if dur_raw:
            duration_min = float(dur_raw)
    if pd.isna(end) and np.isfinite(duration_min):
        end = start + pd.to_timedelta(duration_min, unit="min")
    peak_flux = np.nan
    # The files are fixed-width legacy listings. After duration/type fields, the
    # largest remaining number is a useful conservative proxy for event size.
    numeric_tail = []
    for token in parts[5:]:
        cleaned = re.sub(r"[^0-9.]", "", token)
        if cleaned:
            try:
                numeric_tail.append(float(cleaned))
            except ValueError:
                pass
    if numeric_tail:
        peak_flux = float(max(numeric_tail))
    return {
        "date": date,
        "start_time": start,
        "end_time": end,
        "frequency_mhz_reported": float(m.group(1)),
        "station": m.group(2) or "",
        "duration_min_reported": duration_min,
        "peak_flux_proxy": peak_flux,
        "raw_line": line.rstrip("\n"),
    }


def download_parse_rstn(start_year: int, end_year: int, out_dir: Path, refresh: bool = False) -> pd.DataFrame:
    rows = []
    raw_dir = ensure_dir(out_dir / "external_data" / "rstn_fixed_frequency")
    for year in range(int(start_year), int(end_year) + 1):
        url = RSTN_FIXED_URL_TEMPLATE.format(year=year)
        path = _download(url, raw_dir / f"solar-radio_bursts_fixed-frequency_{year}.txt", refresh=refresh)
        for line in path.read_text(encoding="latin1", errors="replace").splitlines():
            row = parse_rstn_fixed_frequency_line(line)
            if row is not None:
                rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("start_time").reset_index(drop=True)
        out.to_csv(out_dir / "rstn_fixed_frequency_events_parsed.csv", index=False)
    return out


def count_bursts_near(times: pd.Series, burst_times: pd.Series, window_h: float) -> np.ndarray:
    event_ns = pd.to_datetime(times).astype("int64").to_numpy()
    burst_ns = pd.to_datetime(burst_times).dropna().astype("int64").to_numpy().copy()
    burst_ns.sort()
    half = int(float(window_h) * 3600 * 1e9)
    left = np.searchsorted(burst_ns, event_ns - half, side="left")
    right = np.searchsorted(burst_ns, event_ns + half, side="right")
    return right - left


def annotate_solar_rstn(features: pd.DataFrame, rstn: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    out["predicted_event_time"] = pd.to_datetime(out["predicted_event_time"])
    out["event_date"] = out["predicted_event_time"].dt.floor("D")
    if rstn.empty:
        out["rstn_burst_count_1h"] = 0
        out["rstn_burst_count_6h"] = 0
        out["rstn_daily_count"] = 0
        out["rstn_daily_peak_flux_proxy"] = np.nan
    else:
        out["rstn_burst_count_1h"] = count_bursts_near(out["predicted_event_time"], rstn["start_time"], 1.0)
        out["rstn_burst_count_6h"] = count_bursts_near(out["predicted_event_time"], rstn["start_time"], 6.0)
        daily = (
            rstn.groupby("date", sort=True)
            .agg(rstn_daily_count=("start_time", "size"), rstn_daily_peak_flux_proxy=("peak_flux_proxy", "max"))
            .reset_index()
            .rename(columns={"date": "event_date"})
        )
        out = out.merge(daily, on="event_date", how="left")
        out["rstn_daily_count"] = out["rstn_daily_count"].fillna(0).astype(int)
    sun_real = out[out["analysis_source"].astype(str).eq("sun") & out["control_family"].astype(str).eq("real")]
    q75 = float(sun_real["rstn_daily_count"].quantile(0.75)) if not sun_real.empty else np.inf
    out["rstn_high_day_threshold"] = q75
    out["rstn_near_1h"] = out["rstn_burst_count_1h"].astype(int) > 0
    out["rstn_near_6h"] = out["rstn_burst_count_6h"].astype(int) > 0
    out["rstn_high_day"] = out["rstn_daily_count"].astype(float) >= q75
    return out


def _circular_between(values: pd.Series | np.ndarray, lo: float, hi: float) -> np.ndarray:
    v = np.asarray(values, dtype=float) % 360.0
    if lo <= hi:
        return (v >= lo) & (v <= hi)
    return (v >= lo) | (v <= hi)


def _spice_time_strings(times: pd.Series) -> list[str]:
    return [pd.Timestamp(t).strftime("%Y-%m-%dT%H:%M:%S.%f") for t in times]


def ensure_spice_kernels(out_dir: Path, refresh: bool = False) -> dict[str, Path]:
    kernel_dir = ensure_dir(out_dir / "external_data" / "spice_kernels")
    return {name: _download(url, kernel_dir / name, refresh=refresh) for name, url in NAIF_KERNELS.items()}


def _signed_phase_deg(reference_xy: np.ndarray, vector_xy: np.ndarray) -> np.ndarray:
    cross = reference_xy[:, 0] * vector_xy[:, 1] - reference_xy[:, 1] * vector_xy[:, 0]
    dot = np.einsum("ij,ij->i", reference_xy, vector_xy)
    return np.degrees(np.arctan2(cross, dot)) % 360.0


def annotate_jupiter_spice(features: pd.DataFrame, out_dir: Path, refresh_kernels: bool = False) -> pd.DataFrame:
    out = features.copy()
    out["predicted_event_time"] = pd.to_datetime(out["predicted_event_time"])
    out["event_date"] = out["predicted_event_time"].dt.floor("D")
    for col in ["jupiter_cml_spice_deg", "io_phase_spice_deg", "io_phase_spice_reverse_deg"]:
        out[col] = np.nan
    if spice is None:
        out["spice_geometry_status"] = "spiceypy_not_available"
        return out

    jup_idx = out["analysis_source"].astype(str).eq("jupiter")
    if not jup_idx.any():
        out["spice_geometry_status"] = ""
        return out

    kernels = ensure_spice_kernels(out_dir, refresh=refresh_kernels)
    spice.kclear()
    for name in ["naif0012.tls", "pck00010.tpc", "jup100.bsp", "jup348.bsp"]:
        spice.furnsh(str(kernels[name]))
    try:
        unique_times = pd.Series(pd.to_datetime(out.loc[jup_idx, "predicted_event_time"]).drop_duplicates().sort_values())
        et = np.asarray(spice.str2et(_spice_time_strings(unique_times)), dtype=float)
        # Use geometric positions. The broad CML/Io gates are far wider than
        # light-time corrections, and the compact merged Jupiter kernel does
        # not always provide the full SSB route needed by aberration-corrected
        # Io states.
        earth_vec = np.asarray(spice.spkpos("EARTH", et, "J2000", "NONE", "JUPITER")[0], dtype=float)
        io_vec = np.asarray(spice.spkpos("IO", et, "J2000", "NONE", "JUPITER")[0], dtype=float)
        cml = []
        io_phase = []
        io_phase_reverse = []
        for e, ev, iv in zip(et, earth_vec, io_vec):
            rot = np.asarray(spice.pxform("J2000", "IAU_JUPITER", float(e)), dtype=float)
            earth_body = rot @ ev
            io_body = rot @ iv
            lon_east = np.degrees(np.arctan2(earth_body[1], earth_body[0])) % 360.0
            cml.append((360.0 - lon_east) % 360.0)
            ref = -earth_body[:2].reshape(1, 2)
            io_xy = io_body[:2].reshape(1, 2)
            phase = float(_signed_phase_deg(ref, io_xy)[0])
            io_phase.append(phase)
            io_phase_reverse.append((-phase) % 360.0)
        geom = pd.DataFrame(
            {
                "predicted_event_time": unique_times.to_numpy(),
                "jupiter_cml_spice_deg": cml,
                "io_phase_spice_deg": io_phase,
                "io_phase_spice_reverse_deg": io_phase_reverse,
            }
        )
        out = out.merge(geom, on="predicted_event_time", how="left", suffixes=("", "_new"))
        for col in ["jupiter_cml_spice_deg", "io_phase_spice_deg", "io_phase_spice_reverse_deg"]:
            new_col = f"{col}_new"
            if new_col in out.columns:
                out[col] = out[new_col].combine_first(out[col])
                out = out.drop(columns=[new_col])
        out["spice_geometry_status"] = np.where(jup_idx, "spice_ok", "")
    finally:
        spice.kclear()

    cml = out["jupiter_cml_spice_deg"]
    io = out["io_phase_spice_deg"]
    ior = out["io_phase_spice_reverse_deg"]
    io_a = _circular_between(cml, 180, 300) & _circular_between(io, 180, 260)
    io_b = _circular_between(cml, 15, 240) & _circular_between(io, 40, 110)
    io_c = _circular_between(cml, 60, 280) & _circular_between(io, 200, 260)
    io_ar = _circular_between(cml, 180, 300) & _circular_between(ior, 180, 260)
    io_br = _circular_between(cml, 15, 240) & _circular_between(ior, 40, 110)
    io_cr = _circular_between(cml, 60, 280) & _circular_between(ior, 200, 260)
    out["spice_io_abc_gate"] = io_a | io_b | io_c
    out["spice_io_abc_reversed_gate"] = io_ar | io_br | io_cr
    out["spice_io_abc_or_reversed_gate"] = out["spice_io_abc_gate"] | out["spice_io_abc_reversed_gate"]
    return out


def mode_mask(df: pd.DataFrame, source: str, mode: str) -> pd.Series:
    src = df["analysis_source"].astype(str).eq(source)
    if mode == "all_events":
        return src
    if mode in {"rstn_near_1h", "rstn_near_6h", "rstn_high_day"}:
        return src & df[mode].astype(bool)
    if mode in {"spice_io_abc_gate", "spice_io_abc_or_reversed_gate"}:
        return src & df[mode].astype(bool)
    raise ValueError(mode)


def summarize_modes(features: pd.DataFrame, source_modes: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for source, modes in source_modes.items():
        for mode in modes:
            sub = features[mode_mask(features, source, mode)].copy()
            real = sub[sub["control_family"].astype(str).eq("real")]
            controls = sub[~sub["control_family"].astype(str).eq("real")]
            for keys, grp in real.groupby(["frequency_band", "frequency_mhz"], sort=True):
                band, freq = keys
                vals = pd.to_numeric(grp["source_like_log_contrast"], errors="coerce").dropna().to_numpy(dtype=float)
                same_controls = controls[controls["frequency_band"].astype(int).eq(int(band))]
                cvals = pd.to_numeric(same_controls["source_like_log_contrast"], errors="coerce").dropna().to_numpy(dtype=float)
                group_meds = same_controls.groupby(["control_family", "control_id"], sort=True)["source_like_log_contrast"].median().dropna().to_numpy(dtype=float)
                n = int(len(vals))
                k = int(np.count_nonzero(vals > 0))
                med = float(np.nanmedian(vals)) if n else np.nan
                emp = float((1 + np.count_nonzero(group_meds >= med)) / (1 + len(group_meds))) if len(group_meds) else np.nan
                rows.append(
                    {
                        "source": source,
                        "mode": mode,
                        "frequency_band": int(band),
                        "frequency_mhz": float(freq),
                        "n_real_events": n,
                        "real_sign_fraction": float(k / n) if n else np.nan,
                        "real_median_log_contrast": med,
                        "real_median_fractional_contrast": float(np.exp(med) - 1.0) if np.isfinite(med) else np.nan,
                        "one_sided_sign_p": _binom_sf_one_sided(k, n),
                        "mannwhitney_real_gt_controls_p": _mannwhitney_greater(vals, cvals),
                        "cliffs_delta_real_vs_controls": _cliffs_delta(vals, cvals),
                        "control_group_empirical_p_median_ge_real": emp,
                        "control_group_median_log_contrast": float(np.nanmedian(group_meds)) if len(group_meds) else np.nan,
                        "n_control_events": int(len(cvals)),
                        "n_control_groups": int(len(group_meds)),
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    classes = []
    for _, row in out.iterrows():
        n = int(row["n_real_events"])
        med = float(row["real_median_log_contrast"])
        sign = float(row["real_sign_fraction"])
        mw = float(row["mannwhitney_real_gt_controls_p"])
        emp = float(row["control_group_empirical_p_median_ge_real"])
        delta = float(row["cliffs_delta_real_vs_controls"])
        if n < 8:
            cls = "too_few_selected_events"
        elif med > 0 and sign >= 0.65 and mw <= 0.01 and emp <= 0.10 and delta >= 0.15:
            cls = "externally_selected_candidate"
        elif med > 0 and sign >= 0.60 and mw <= 0.05:
            cls = "positive_but_control_limited"
        elif med < 0 and sign <= 0.40:
            cls = "anti_template"
        else:
            cls = "not_detected"
        classes.append(cls)
    out["evidence_class"] = classes
    return out


def select_visual_cases(summary: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for source, src in summary.groupby("source", sort=True):
        cand = src[src["mode"].ne("all_events")].copy()
        cand["_score"] = (
            cand["real_median_log_contrast"].fillna(-999)
            + cand["real_sign_fraction"].fillna(0)
            - cand["control_group_empirical_p_median_ge_real"].fillna(1)
            + 0.5 * cand["evidence_class"].eq("externally_selected_candidate").astype(float)
        )
        pieces.append(cand.sort_values("_score", ascending=False).head(4))
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def plot_mode_spectra(summary: pd.DataFrame, out_dir: Path, prefix: str) -> list[Path]:
    paths = []
    for source, src in summary.groupby("source", sort=True):
        fig, ax = plt.subplots(figsize=(10.5, 5.0))
        ax.axhline(0, color="0.65", lw=0.8)
        for mode, grp in src.groupby("mode", sort=False):
            grp = grp.sort_values("frequency_mhz")
            ax.plot(
                grp["frequency_mhz"],
                grp["real_median_log_contrast"],
                marker="o",
                lw=1.6,
                label=mode,
                color=MODE_COLOR.get(str(mode), None),
            )
        ax.set_xscale("log")
        ax.set_xticks(list(FREQUENCY_MAP_MHZ.values()))
        ax.set_xticklabels([f"{v:g}" for v in FREQUENCY_MAP_MHZ.values()], rotation=45, ha="right")
        ax.set_xlabel("frequency (MHz)")
        ax.set_ylabel("median source-like log contrast")
        ax.set_title(f"{SOURCE_LABEL.get(source, source)} externally selected lower-V contrast")
        ax.grid(True, color="0.9", lw=0.5)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        path = out_dir / f"{prefix}_{source}_mode_spectrum.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_selected_distributions(features: pd.DataFrame, selected: pd.DataFrame, out_dir: Path, prefix: str) -> list[Path]:
    paths = []
    if selected.empty:
        return paths
    for source, cases in selected.groupby("source", sort=True):
        fig, axes = plt.subplots(len(cases), 1, figsize=(9.7, 3.1 * len(cases)))
        if len(cases) == 1:
            axes = [axes]
        for ax, (_, case) in zip(axes, cases.iterrows()):
            mode = str(case["mode"])
            freq = float(case["frequency_mhz"])
            sub = features[
                mode_mask(features, source, mode)
                & np.isclose(pd.to_numeric(features["frequency_mhz"], errors="coerce"), freq)
            ].copy()
            real = sub[sub["control_family"].astype(str).eq("real")]["source_like_log_contrast"].dropna().to_numpy(dtype=float)
            ctrl = sub[~sub["control_family"].astype(str).eq("real")]["source_like_log_contrast"].dropna().to_numpy(dtype=float)
            vals = np.r_[real, ctrl]
            if len(vals) == 0:
                continue
            lo, hi = np.nanpercentile(vals, [1, 99])
            if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                lo, hi = -1.0, 1.0
            bins = np.linspace(float(lo), float(hi), 45)
            ax.hist(real, bins=bins, density=True, histtype="step", color="black", lw=2.0, label=f"real n={len(real)}")
            ax.hist(ctrl, bins=bins, density=True, histtype="step", color="#7570b3", lw=1.5, label=f"controls n={len(ctrl)}")
            ax.axvline(0, color="0.55", lw=0.8)
            ax.axvline(float(case["real_median_log_contrast"]), color="black", ls="--", lw=1.2)
            ax.set_title(
                f"{SOURCE_LABEL.get(source, source)} {freq:.2f} MHz {mode}: "
                f"{case['evidence_class']}, sign={float(case['real_sign_fraction']):.2f}, "
                f"emp p={float(case['control_group_empirical_p_median_ge_real']):.3g}"
            )
            ax.set_xlabel("source-like log contrast")
            ax.set_ylabel("density")
            ax.grid(True, color="0.92", lw=0.5)
            ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        path = out_dir / f"{prefix}_{source}_selected_distributions.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_rstn_timeline(features: pd.DataFrame, rstn: pd.DataFrame, out_dir: Path) -> Path:
    sun = features[features["analysis_source"].astype(str).eq("sun") & features["control_family"].astype(str).eq("real")].copy()
    daily = sun.groupby("event_date")["source_like_log_contrast"].median().reset_index()
    rstn_daily = rstn.groupby("date").size().rename("rstn_burst_count").reset_index() if not rstn.empty else pd.DataFrame(columns=["date", "rstn_burst_count"])
    fig, ax = plt.subplots(figsize=(11.0, 4.8))
    ax.bar(rstn_daily["date"], rstn_daily["rstn_burst_count"], width=1.0, color="0.75", label="RSTN fixed-frequency burst reports/day")
    ax2 = ax.twinx()
    ax2.scatter(daily["event_date"], daily["source_like_log_contrast"], s=18, color="#b2182b", alpha=0.75, label="Sun event median contrast")
    ax.set_ylabel("RSTN reports/day")
    ax2.set_ylabel("median source-like log contrast")
    ax.set_title("Solar RSTN burst reports against RAE-2 Sun event dates")
    ax.grid(True, color="0.9", lw=0.5)
    ax.legend(frameon=False, loc="upper left")
    ax2.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    path = out_dir / "sun_rstn_burst_timeline.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_jupiter_spice_phase(features: pd.DataFrame, out_dir: Path) -> Path:
    real = features[features["analysis_source"].astype(str).eq("jupiter") & features["control_family"].astype(str).eq("real")].copy()
    fig, ax = plt.subplots(figsize=(8.6, 6.8))
    if not real.empty:
        band = real[real["frequency_mhz"].eq(0.45)].copy()
        if band.empty:
            band = real.copy()
        sc = ax.scatter(
            band["jupiter_cml_spice_deg"],
            band["io_phase_spice_deg"],
            c=band["source_like_log_contrast"],
            cmap="coolwarm",
            s=45,
            edgecolor="black",
            linewidth=0.25,
        )
        fig.colorbar(sc, ax=ax, label="source-like log contrast")
    boxes = [
        ("Io-A", 180, 300, 180, 260, "#1b9e77"),
        ("Io-B", 15, 240, 40, 110, "#d95f02"),
        ("Io-C", 60, 280, 200, 260, "#7570b3"),
    ]
    for label, x0, x1, y0, y1, color in boxes:
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, lw=1.5))
        ax.text(x0 + 4, y1 - 6, label, color=color, fontsize=9, va="top")
    ax.set_xlim(0, 360)
    ax.set_ylim(0, 360)
    ax.set_xlabel("SPICE System-III CML proxy (deg)")
    ax.set_ylabel("SPICE Io phase, 0 deg near superior conjunction (deg)")
    ax.set_title("Jupiter real lower-V occultation events in SPICE CML/Io plane")
    ax.grid(True, color="0.9", lw=0.5)
    fig.tight_layout()
    path = out_dir / "jupiter_spice_cml_io_phase_real_events.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def load_visibility_states(jupiter_states: Path, earth_states: Path) -> pd.DataFrame:
    jup = read_table(jupiter_states, parse_dates=["time"])
    earth = read_table(earth_states, parse_dates=["time"])
    states = jup[["time", "visible_by_moon", "limb_angle_deg"]].rename(
        columns={"visible_by_moon": "jupiter_visible_by_moon", "limb_angle_deg": "jupiter_limb_angle_deg"}
    )
    states = states.merge(
        earth[["time", "visible_by_moon", "limb_angle_deg"]].rename(
            columns={"visible_by_moon": "earth_visible_by_moon", "limb_angle_deg": "earth_limb_angle_deg"}
        ),
        on="time",
        how="inner",
    ).sort_values("time")
    return states


def direct_jupiter_visible_scan(clean_path: Path, states: pd.DataFrame, out_dir: Path, chunksize: int = 750_000) -> tuple[pd.DataFrame, list[Path]]:
    values: dict[tuple[int, str], list[np.ndarray]] = defaultdict(list)
    state_cols = ["time", "jupiter_visible_by_moon", "earth_visible_by_moon"]
    states = states[state_cols].sort_values("time").reset_index(drop=True)
    usecols = ["time", "frequency_band", "antenna", "power", "is_valid"]
    for chunk in read_table(clean_path, usecols=usecols, parse_dates=["time"], chunksize=int(chunksize), low_memory=False):
        chunk = chunk[
            chunk["antenna"].astype(str).eq(ANTENNA)
            & chunk["is_valid"].astype(bool)
            & pd.to_numeric(chunk["power"], errors="coerce").gt(0)
        ].copy()
        if chunk.empty:
            continue
        chunk["power"] = pd.to_numeric(chunk["power"], errors="coerce")
        chunk = chunk.dropna(subset=["power"]).sort_values("time")
        merged = pd.merge_asof(chunk, states, on="time", direction="nearest", tolerance=pd.Timedelta(seconds=360))
        merged = merged.dropna(subset=["jupiter_visible_by_moon", "earth_visible_by_moon"])
        if merged.empty:
            continue
        merged["log_power"] = np.log(merged["power"].to_numpy(dtype=float))
        masks = {
            "jupiter_visible": merged["jupiter_visible_by_moon"].astype(bool),
            "jupiter_occulted": ~merged["jupiter_visible_by_moon"].astype(bool),
            "jupiter_visible_earth_occulted": merged["jupiter_visible_by_moon"].astype(bool)
            & ~merged["earth_visible_by_moon"].astype(bool),
            "jupiter_visible_earth_visible": merged["jupiter_visible_by_moon"].astype(bool)
            & merged["earth_visible_by_moon"].astype(bool),
        }
        for band, grp in merged.groupby("frequency_band", sort=True):
            band = int(band)
            for regime, mask in masks.items():
                vals = grp.loc[mask.loc[grp.index], "log_power"].to_numpy(dtype=float)
                if len(vals):
                    values[(band, regime)].append(vals)
            vals_all = grp["log_power"].to_numpy(dtype=float)
            if len(vals_all):
                values[(band, "all_lower_v")].append(vals_all)

    rows = []
    all_values: dict[tuple[int, str], np.ndarray] = {}
    for key, pieces in values.items():
        all_values[key] = np.concatenate(pieces) if pieces else np.array([], dtype=float)
    for band in sorted({k[0] for k in all_values}):
        freq = float(FREQUENCY_MAP_MHZ.get(int(band), np.nan))
        all_band = all_values.get((band, "all_lower_v"), np.array([], dtype=float))
        center = float(np.nanmedian(all_band)) if len(all_band) else np.nan
        scale = robust_sigma(all_band - center) if len(all_band) else np.nan
        threshold = center + 3.0 * scale if np.isfinite(center) and np.isfinite(scale) and scale > 0 else np.nan
        for regime in ["all_lower_v", "jupiter_visible", "jupiter_occulted", "jupiter_visible_earth_occulted", "jupiter_visible_earth_visible"]:
            vals = all_values.get((band, regime), np.array([], dtype=float))
            rows.append(
                {
                    "frequency_band": band,
                    "frequency_mhz": freq,
                    "regime": regime,
                    "n_samples": int(len(vals)),
                    "median_log_power": float(np.nanmedian(vals)) if len(vals) else np.nan,
                    "robust_sigma_log_power": robust_sigma(vals - np.nanmedian(vals)) if len(vals) else np.nan,
                    "high_power_threshold_log": threshold,
                    "high_power_fraction": float(np.mean(vals > threshold)) if len(vals) and np.isfinite(threshold) else np.nan,
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "jupiter_direct_visible_power_summary.csv", index=False)
    paths = plot_direct_jupiter_scan(summary, all_values, out_dir)
    return summary, paths


def plot_direct_jupiter_scan(summary: pd.DataFrame, all_values: dict[tuple[int, str], np.ndarray], out_dir: Path) -> list[Path]:
    paths = []
    wide = summary.pivot(index=["frequency_band", "frequency_mhz"], columns="regime", values="median_log_power").reset_index()
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    ax.axhline(0, color="0.65", lw=0.8)
    if {"jupiter_visible", "jupiter_occulted"}.issubset(wide.columns):
        ax.plot(
            wide["frequency_mhz"],
            wide["jupiter_visible"] - wide["jupiter_occulted"],
            marker="o",
            label="Jupiter visible - Jupiter occulted",
        )
    if {"jupiter_visible_earth_occulted", "jupiter_visible_earth_visible"}.issubset(wide.columns):
        ax.plot(
            wide["frequency_mhz"],
            wide["jupiter_visible_earth_occulted"] - wide["jupiter_visible_earth_visible"],
            marker="o",
            label="Jupiter visible, Earth occulted - Earth visible",
        )
    ax.set_xscale("log")
    ax.set_xticks(list(FREQUENCY_MAP_MHZ.values()))
    ax.set_xticklabels([f"{v:g}" for v in FREQUENCY_MAP_MHZ.values()], rotation=45, ha="right")
    ax.set_xlabel("frequency (MHz)")
    ax.set_ylabel("median log-power difference")
    ax.set_title("Direct lower-V Jupiter-visible raw-power comparison")
    ax.grid(True, color="0.9", lw=0.5)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = out_dir / "jupiter_direct_visible_median_power_spectrum.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    frac = summary.pivot(index=["frequency_band", "frequency_mhz"], columns="regime", values="high_power_fraction").reset_index()
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    for regime in ["jupiter_visible", "jupiter_occulted", "jupiter_visible_earth_occulted", "jupiter_visible_earth_visible"]:
        if regime in frac.columns:
            ax.plot(frac["frequency_mhz"], frac[regime], marker="o", label=regime)
    ax.set_xscale("log")
    ax.set_xticks(list(FREQUENCY_MAP_MHZ.values()))
    ax.set_xticklabels([f"{v:g}" for v in FREQUENCY_MAP_MHZ.values()], rotation=45, ha="right")
    ax.set_xlabel("frequency (MHz)")
    ax.set_ylabel("fraction above band median + 3 robust sigma")
    ax.set_title("Direct lower-V high-power sample fraction")
    ax.grid(True, color="0.9", lw=0.5)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = out_dir / "jupiter_direct_visible_high_power_fraction.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    # Show actual log-power distributions for the three bands where the
    # Earth-occulted Jupiter-visible comparison differs most.
    deltas = wide.copy()
    if {"jupiter_visible_earth_occulted", "jupiter_visible_earth_visible"}.issubset(deltas.columns):
        deltas["abs_delta"] = (deltas["jupiter_visible_earth_occulted"] - deltas["jupiter_visible_earth_visible"]).abs()
        top = deltas.sort_values("abs_delta", ascending=False).head(3)
    else:
        top = wide.head(3)
    fig, axes = plt.subplots(len(top), 1, figsize=(9.6, 3.1 * len(top)))
    if len(top) == 1:
        axes = [axes]
    for ax, (_, row) in zip(axes, top.iterrows()):
        band = int(row["frequency_band"])
        freq = float(row["frequency_mhz"])
        a = all_values.get((band, "jupiter_visible_earth_occulted"), np.array([], dtype=float))
        b = all_values.get((band, "jupiter_visible_earth_visible"), np.array([], dtype=float))
        vals = np.r_[a, b]
        if len(vals) == 0:
            continue
        lo, hi = np.nanpercentile(vals, [1, 99])
        bins = np.linspace(float(lo), float(hi), 55)
        ax.hist(a, bins=bins, density=True, histtype="step", color="black", lw=2, label=f"Jup visible + Earth occulted n={len(a)}")
        ax.hist(b, bins=bins, density=True, histtype="step", color="#7570b3", lw=1.5, label=f"Jup visible + Earth visible n={len(b)}")
        ax.set_title(f"Jupiter direct scan {freq:.2f} MHz lower V raw log power")
        ax.set_xlabel("log raw power")
        ax.set_ylabel("density")
        ax.grid(True, color="0.92", lw=0.5)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = out_dir / "jupiter_direct_visible_power_distributions.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)
    return paths


def write_report(
    out_dir: Path,
    summary: pd.DataFrame,
    rstn: pd.DataFrame,
    direct_summary: pd.DataFrame,
    paths: list[Path],
    config: dict[str, object],
) -> Path:
    cols = [
        "source",
        "mode",
        "frequency_mhz",
        "evidence_class",
        "n_real_events",
        "real_sign_fraction",
        "real_median_log_contrast",
        "mannwhitney_real_gt_controls_p",
        "control_group_empirical_p_median_ge_real",
    ]
    lines = [
        "# Solar Burst / SPICE Jupiter / Direct Visibility Test",
        "",
        "This run implements two external-selection upgrades plus a non-occultation Jupiter scan.",
        "",
        "## Solar RSTN Gate",
        "",
        "- Uses NOAA/NCEI fixed-frequency solar radio burst listings.",
        "- RSTN reports are independent of RAE-2 and cover the mission years.",
        "- This is still an activity proxy: RSTN is mostly 18 MHz and above, while RAE-2 bands here are 0.45-9.18 MHz.",
        f"- Parsed RSTN records: `{len(rstn)}`.",
        "",
        "## Jupiter SPICE Gate",
        "",
        "- Uses NAIF SPICE kernels to compute Jupiter body-fixed CML proxy and Io phase.",
        "- The script also tests a reversed Io-phase convention because historical CML-Io plots differ in phase-direction conventions.",
        "",
        "## Direct Jupiter-Visible Scan",
        "",
        "- Ignores lunar occultation timing.",
        "- Compares raw lower-V power when Jupiter is visible vs occulted by the Moon.",
        "- Also compares Jupiter-visible samples when Earth is occulted vs Earth visible.",
        "- This is a burst/background screen, not a source occultation claim.",
        "",
        "## Configuration",
        "",
        *[f"- `{k}`: `{v}`" for k, v in config.items() if k != "software_versions"],
        "",
        "## External-Gate Summary",
        "",
        summary[cols].sort_values(["source", "mode", "frequency_mhz"]).to_string(index=False) if not summary.empty else "(none)",
        "",
        "## Direct Jupiter Summary",
        "",
        direct_summary.to_string(index=False) if not direct_summary.empty else "(not run)",
        "",
        "## Plots",
        "",
        *[f"- `{p}`" for p in paths],
        "",
        "## Interpretation Guardrails",
        "",
        "- A useful gate should separate real events from time/off-source controls, not merely select positive-looking real subsets.",
        "- Direct Jupiter-visible excess can indicate bursty Jupiter, residual Earth/Galaxy/background geometry, or receiver state; it is not by itself an occultation detection.",
        "- The strongest next Jupiter-specific check would be a known Jovian radio occurrence-probability model or actual historical DAM observation logs in the RAE-2 band.",
    ]
    path = out_dir / "solar_burst_spice_jupiter_visibility_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--clean", type=Path, default=DEFAULT_CLEAN)
    parser.add_argument("--jupiter-states", type=Path, default=DEFAULT_JUPITER_STATES)
    parser.add_argument("--earth-states", type=Path, default=DEFAULT_EARTH_STATES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--refresh-external", action="store_true")
    parser.add_argument("--skip-direct-jupiter", action="store_true")
    parser.add_argument(
        "--save-large-tables",
        action="store_true",
        help="Save the full annotated Sun/Jupiter event-feature table.",
    )
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    features = read_table(args.features, parse_dates=["predicted_event_time"], low_memory=False)
    features = features[features["antenna"].astype(str).eq(ANTENNA)].copy()
    min_year = int(pd.to_datetime(features["predicted_event_time"]).dt.year.min())
    max_year = int(pd.to_datetime(features["predicted_event_time"]).dt.year.max())

    rstn = download_parse_rstn(min_year, max_year, out_dir, refresh=bool(args.refresh_external))
    annotated = annotate_solar_rstn(features, rstn)
    annotated = annotate_jupiter_spice(annotated, out_dir, refresh_kernels=bool(args.refresh_external))
    if args.save_large_tables:
        annotated[annotated["analysis_source"].isin(["sun", "jupiter"])].to_csv(
            out_dir / "solar_jupiter_external_selected_event_features.csv",
            index=False,
        )

    source_modes = {
        "sun": ["all_events", "rstn_near_1h", "rstn_near_6h", "rstn_high_day"],
        "jupiter": ["all_events", "spice_io_abc_gate", "spice_io_abc_or_reversed_gate"],
    }
    summary = summarize_modes(annotated, source_modes)
    summary.to_csv(out_dir / "solar_burst_spice_jupiter_gate_summary.csv", index=False)
    selected = select_visual_cases(summary)
    selected.to_csv(out_dir / "solar_burst_spice_jupiter_selected_visual_cases.csv", index=False)
    if not selected.empty:
        selected_rows = []
        for _, case in selected.iterrows():
            source = str(case["source"])
            mode = str(case["mode"])
            freq = float(case["frequency_mhz"])
            sub = annotated[
                mode_mask(annotated, source, mode)
                & np.isclose(pd.to_numeric(annotated["frequency_mhz"], errors="coerce"), freq)
            ].copy()
            sub["selected_external_mode"] = mode
            selected_rows.append(sub)
        pd.concat(selected_rows, ignore_index=True).to_csv(
            out_dir / "solar_burst_spice_jupiter_selected_event_features.csv",
            index=False,
        )

    paths: list[Path] = []
    paths.extend(plot_mode_spectra(summary, out_dir, "external_gate"))
    paths.extend(plot_selected_distributions(annotated, selected, out_dir, "external_gate"))
    paths.append(plot_rstn_timeline(annotated, rstn, out_dir))
    paths.append(plot_jupiter_spice_phase(annotated, out_dir))

    direct_summary = pd.DataFrame()
    if not args.skip_direct_jupiter:
        states = load_visibility_states(args.jupiter_states, args.earth_states)
        direct_summary, direct_paths = direct_jupiter_visible_scan(args.clean, states, out_dir)
        paths.extend(direct_paths)

    config = {
        "features": str(args.features),
        "clean": str(args.clean),
        "jupiter_states": str(args.jupiter_states),
        "earth_states": str(args.earth_states),
        "rstn_url_template": RSTN_FIXED_URL_TEMPLATE,
        "spice_kernels": NAIF_KERNELS,
        "antenna": ANTENNA,
        "software_versions": software_versions(),
    }
    write_json(out_dir / "run_config.json", config)
    report = write_report(out_dir, summary, rstn, direct_summary, paths, config)
    print(report)


if __name__ == "__main__":
    main()
