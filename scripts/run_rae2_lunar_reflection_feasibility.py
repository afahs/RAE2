#!/usr/bin/env python
"""Quantitative RAE-2 direct-plus-lunar-reflected Earth feasibility audit.

This script is deliberately data-first.  It inventories the samples that are
actually archived, solves the exact spherical specular geometry, applies the
finite RV bandwidth and post-detection response, models point-source lunar-edge
diffraction, and reports how many Earth limb events contain a locally sampled
fringe.  Its matched-filter result is labelled optimistic because the apparent
AKR/TKR source position and lunar reflection phase are not known event by event.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import os
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import fresnel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.constants import (  # noqa: E402
    EARTH_MOON_DISTANCE_KM,
    EARTH_UNIT_COLUMNS,
    FREQUENCY_MAP_MHZ,
    MOON_RADIUS_KM,
    SPACECRAFT_COLUMNS,
)
from rylevonberg.table_io import read_table, write_table  # noqa: E402
from rylevonberg.util import ensure_dir, robust_sigma, software_versions, write_json  # noqa: E402
from scripts.investigate_lower_v_lunar_reflection_interference import (  # noqa: E402
    C_KM_S,
    RV_COARSE_TIME_CONSTANT_S,
    RV_EFFECTIVE_BANDWIDTH_HZ,
    collect_reflection_samples,
    fit_fringe_group,
    fit_fringe_models,
    tangent_proxy_path_km,
    wavelength_km,
)


DEFAULT_MASTER = Path(os.environ.get("RAE2_MASTER_CSV", "data/interpolatedRAE2MasterFile.csv"))
DEFAULT_CLEAN = ROOT / "outputs/control_survey_earth_sun_postnov1974_v2/01_ingest/cleaned_timeseries.csv"
DEFAULT_EVENTS = ROOT / "outputs/planetary_confirmation_survey_science_baseline_v2/events/all_planet_predicted_events.csv"
DEFAULT_OUT = ROOT / "outputs/rae2_lunar_reflection_feasibility_v1"


def _valid(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.fillna(False).to_numpy(dtype=bool)
    return series.astype(str).str.lower().isin(["true", "1", "yes"]).to_numpy(dtype=bool)


def _finite_median(values: pd.Series | np.ndarray) -> float:
    vals = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(np.nanmedian(vals)) if vals.size else np.nan


def _finite_percentile(values: pd.Series | np.ndarray, q: float) -> float:
    vals = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(np.nanpercentile(vals, q)) if vals.size else np.nan


def _markdown(frame: pd.DataFrame, columns: list[str] | None = None, max_rows: int = 40) -> str:
    if frame.empty:
        return "No rows."
    work = frame.loc[:, columns].head(max_rows).copy() if columns else frame.head(max_rows).copy()
    for col in work.columns:
        if pd.api.types.is_numeric_dtype(work[col]):
            work[col] = work[col].map(lambda x: "" if pd.isna(x) else f"{x:.5g}")
    widths = {col: max(len(str(col)), *(len(str(x)) for x in work[col])) for col in work.columns}
    lines = [
        "| " + " | ".join(str(col).ljust(widths[col]) for col in work.columns) + " |",
        "| " + " | ".join("-" * widths[col] for col in work.columns) + " |",
    ]
    for _, row in work.iterrows():
        lines.append("| " + " | ".join(str(row[col]).ljust(widths[col]) for col in work.columns) + " |")
    return "\n".join(lines)


def audit_raw_sampling(master_csv: Path, chunksize: int = 500_000) -> pd.DataFrame:
    """Inventory dwell structure and retained fine/coarse samples in the master."""
    usecols = ["time", "frequency_band", "rv1_coarse", "rv2_coarse", "rv1_fine", "rv2_fine"]
    within_dt: list[float] = []
    revisit_dt: list[float] = []
    run_lengths: list[int] = []
    run_spans: list[float] = []
    last_start_by_band: dict[int, int] = {}
    pending_band: int | None = None
    pending_count = 0
    pending_start_ns = 0
    pending_end_ns = 0
    n_rows = 0
    fine_nonzero = Counter()

    def finish_run(band: int, count: int, start_ns: int, end_ns: int) -> None:
        run_lengths.append(int(count))
        run_spans.append((end_ns - start_ns) / 1e9)
        prior = last_start_by_band.get(int(band))
        if prior is not None:
            delta = (start_ns - prior) / 1e9
            if 30.0 <= delta <= 300.0:
                revisit_dt.append(float(delta))
        last_start_by_band[int(band)] = int(start_ns)

    for chunk in pd.read_csv(master_csv, usecols=usecols, chunksize=int(chunksize), low_memory=False):
        n_rows += len(chunk)
        times = (
            pd.to_datetime(chunk["time"], errors="coerce")
            .to_numpy(dtype="datetime64[ns]")
            .astype(np.int64)
        )
        bands = pd.to_numeric(chunk["frequency_band"], errors="coerce").fillna(-1).to_numpy(dtype=int)
        for col in ["rv1_fine", "rv2_fine"]:
            vals = pd.to_numeric(chunk[col], errors="coerce").to_numpy(dtype=float)
            fine_nonzero[col] += int(np.count_nonzero(np.isfinite(vals) & (vals != 0.0)))
        if len(chunk) == 0:
            continue
        if pending_band is not None:
            same = bands[0] == pending_band
            if same:
                pending_count += 1
                boundary_delta = (int(times[0]) - pending_end_ns) / 1e9
                if 0.0 < boundary_delta < 10.0:
                    within_dt.append(float(boundary_delta))
                pending_end_ns = int(times[0])
            else:
                finish_run(pending_band, pending_count, pending_start_ns, pending_end_ns)
                pending_band = int(bands[0])
                pending_count = 1
                pending_start_ns = pending_end_ns = int(times[0])
        else:
            pending_band = int(bands[0])
            pending_count = 1
            pending_start_ns = pending_end_ns = int(times[0])
        for i in range(1, len(chunk)):
            if bands[i] == pending_band:
                delta = (int(times[i]) - pending_end_ns) / 1e9
                if 0.0 < delta < 10.0:
                    within_dt.append(float(delta))
                pending_count += 1
                pending_end_ns = int(times[i])
            else:
                finish_run(pending_band, pending_count, pending_start_ns, pending_end_ns)
                pending_band = int(bands[i])
                pending_count = 1
                pending_start_ns = pending_end_ns = int(times[i])
    if pending_band is not None:
        finish_run(pending_band, pending_count, pending_start_ns, pending_end_ns)

    mode_count = Counter(run_lengths).most_common(1)[0][0] if run_lengths else 0
    typical = np.asarray(run_lengths) == mode_count
    spans = np.asarray(run_spans, dtype=float)
    return pd.DataFrame(
        [
            {
                "master_rows": int(n_rows),
                "n_frequency_dwells": int(len(run_lengths)),
                "coarse_samples_per_typical_dwell": int(mode_count),
                "typical_dwell_fraction": float(np.mean(typical)) if len(typical) else np.nan,
                "median_within_dwell_cadence_s": _finite_median(within_dt),
                "median_typical_dwell_span_s": _finite_median(spans[typical]) if np.any(typical) else np.nan,
                "median_same_band_revisit_s": _finite_median(revisit_dt),
                "rv1_fine_nonzero_samples": int(fine_nonzero["rv1_fine"]),
                "rv2_fine_nonzero_samples": int(fine_nonzero["rv2_fine"]),
                "rv1_fine_samples_per_dwell": fine_nonzero["rv1_fine"] / max(len(run_lengths), 1),
                "rv2_fine_samples_per_dwell": fine_nonzero["rv2_fine"] / max(len(run_lengths), 1),
            }
        ]
    )


def attach_edge_diffraction(samples: pd.DataFrame) -> pd.DataFrame:
    """Attach finite-distance spherical-limb clearance and knife-edge field."""
    if samples.empty:
        return samples
    out = samples.copy()
    spacecraft = out[SPACECRAFT_COLUMNS].to_numpy(dtype=float)
    source_unit = out[EARTH_UNIT_COLUMNS].to_numpy(dtype=float)
    source = source_unit * EARTH_MOON_DISTANCE_KM
    ray = source - spacecraft
    ray2 = np.einsum("ij,ij->i", ray, ray)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = -np.einsum("ij,ij->i", spacecraft, ray) / ray2
    frac = np.clip(frac, 0.0, 1.0)
    closest = spacecraft + frac[:, None] * ray
    impact = np.linalg.norm(closest, axis=1)
    clearance = impact - MOON_RADIUS_KM
    d_receiver = np.linalg.norm(closest - spacecraft, axis=1)
    d_source = np.linalg.norm(source - closest, axis=1)
    d_eff = d_receiver * d_source / (d_receiver + d_source)
    lam = wavelength_km(out["frequency_mhz"].to_numpy(dtype=float))
    fresnel_scale = np.sqrt(np.maximum(lam * d_eff / 2.0, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        v = clearance / fresnel_scale
    s_int, c_int = fresnel(v)
    field = ((0.5 + c_int) + 1j * (0.5 + s_int)) / np.sqrt(2.0)
    out["edge_clearance_km"] = clearance
    out["edge_effective_distance_km"] = d_eff
    out["edge_fresnel_scale_km"] = fresnel_scale
    out["edge_fresnel_coordinate"] = v
    out["direct_diffraction_field_real"] = np.real(field)
    out["direct_diffraction_field_imag"] = np.imag(field)
    out["diffraction_power"] = np.abs(field) ** 2
    return out


def normalize_events(samples: pd.DataFrame, window_s: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate local noise and direct Earth contrast without averaging dwells."""
    out = samples.copy()
    out["analysis_z"] = np.nan
    rows: list[dict[str, object]] = []
    group_cols = ["antenna", "frequency_band", "event_id", "event_type"]
    for keys, idx in out.groupby(group_cols, sort=False).groups.items():
        g = out.loc[idx]
        is_valid = _valid(g["is_valid"]) if "is_valid" in g else np.ones(len(g), dtype=bool)
        y = pd.to_numeric(g["power"], errors="coerce").to_numpy(dtype=float)
        t_oriented = g["oriented_t_sec"].to_numpy(dtype=float)
        finite = is_valid & np.isfinite(y)
        outer = min(float(window_s), 240.0)
        visible = finite & (t_oriented >= 45.0) & (t_oriented <= outer)
        occulted = finite & (t_oriented <= -45.0) & (t_oriented >= -outer)
        visible_level = float(np.nanmedian(y[visible])) if np.any(visible) else np.nan
        occulted_level = float(np.nanmedian(y[occulted])) if np.any(occulted) else np.nan
        residual_parts = []
        if np.any(visible):
            residual_parts.append(y[visible] - visible_level)
        if np.any(occulted):
            residual_parts.append(y[occulted] - occulted_level)
        residuals = np.concatenate(residual_parts) if residual_parts else np.array([], dtype=float)
        sigma = robust_sigma(residuals) if residuals.size else np.nan
        if not np.isfinite(sigma) or sigma <= 0.0:
            sigma = float(np.nanstd(residuals, ddof=1)) if residuals.size > 1 else np.nan
        center = occulted_level if np.isfinite(occulted_level) else float(np.nanmedian(y[finite])) if np.any(finite) else np.nan
        if np.isfinite(sigma) and sigma > 0.0:
            out.loc[g.index[finite], "analysis_z"] = (y[finite] - center) / sigma
        contrast = visible_level - occulted_level
        rows.append(
            {
                **dict(zip(group_cols, keys)),
                "n_valid_visible_sideband": int(np.count_nonzero(visible)),
                "n_valid_occulted_sideband": int(np.count_nonzero(occulted)),
                "direct_contrast_power": contrast,
                "local_noise_power": sigma,
                "direct_contrast_sigma": contrast / sigma if np.isfinite(sigma) and sigma > 0 else np.nan,
            }
        )
    out["z_power"] = out["analysis_z"]
    return out, pd.DataFrame(rows)


def event_feasibility(samples: pd.DataFrame, contrasts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = [
        "antenna",
        "frequency_band",
        "frequency_mhz",
        "event_id",
        "event_type",
        "predicted_event_time",
    ]
    contrast_lookup = contrasts.set_index(["antenna", "frequency_band", "event_id", "event_type"])
    for keys, g in samples.groupby(group_cols, sort=False, dropna=False):
        valid = _valid(g["is_valid"]) if "is_valid" in g else np.ones(len(g), dtype=bool)
        coherent = (
            valid
            & g["reflection_valid"].to_numpy(dtype=bool)
            & g["direct_visible"].to_numpy(dtype=bool)
            & (g["delay_us"].to_numpy(dtype=float) >= 0.0)
            & (g["delay_us"].to_numpy(dtype=float) <= 1e6 / RV_EFFECTIVE_BANDWIDTH_HZ)
        )
        useful = (
            coherent
            & (np.abs(g["bandwidth_coherence"].to_numpy(dtype=float)) >= 0.2)
            & g["locally_nyquist_resolved"].to_numpy(dtype=bool)
        )
        dwell_spans: list[float] = []
        dwell_counts: list[int] = []
        useful_frame = g.loc[useful]
        if not useful_frame.empty:
            for _, dwell in useful_frame.groupby("dwell_id", sort=False):
                phase_cycles = dwell["phase_cycles"].to_numpy(dtype=float)
                dwell_spans.append(float(np.nanmax(phase_cycles) - np.nanmin(phase_cycles)))
                dwell_counts.append(int(len(dwell)))
        near = valid & (np.abs(g["t_rel_sec"].to_numpy(dtype=float)) <= 150.0)
        clearance_rate = np.nan
        if np.count_nonzero(near) >= 3:
            clearance_rate = float(
                np.polyfit(
                    g.loc[near, "t_rel_sec"].to_numpy(dtype=float),
                    g.loc[near, "edge_clearance_km"].to_numpy(dtype=float),
                    1,
                )[0]
            )
        fresnel_time = (
            _finite_median(g.loc[coherent, "edge_fresnel_scale_km"]) / abs(clearance_rate)
            if np.isfinite(clearance_rate) and abs(clearance_rate) > 0.0 and np.any(coherent)
            else np.nan
        )
        ant, band, freq, event_id, event_type, event_time = keys
        ckey = (ant, int(band), event_id, event_type)
        contrast_sigma = np.nan
        if ckey in contrast_lookup.index:
            value = contrast_lookup.loc[ckey, "direct_contrast_sigma"]
            contrast_sigma = float(value.iloc[0] if isinstance(value, pd.Series) else value)
        max_span = max(dwell_spans, default=0.0)
        rows.append(
            {
                **dict(zip(group_cols, keys)),
                "n_window_samples": int(len(g)),
                "n_valid_window_samples": int(np.count_nonzero(valid)),
                "n_first_lobe_samples": int(np.count_nonzero(coherent)),
                "n_useful_samples": int(np.count_nonzero(useful)),
                "n_useful_dwells": int(len(dwell_counts)),
                "max_useful_samples_in_dwell": max(dwell_counts, default=0),
                "max_useful_dwell_phase_span_cycles": max_span,
                "median_useful_delay_us": _finite_median(g.loc[useful, "delay_us"]),
                "median_useful_bandwidth_coherence": _finite_median(g.loc[useful, "bandwidth_coherence"]),
                "median_useful_fringe_period_s": _finite_median(g.loc[useful, "fringe_period_s"]),
                "median_effective_baseline_km": _finite_median(g.loc[useful, "effective_baseline_km"]),
                "median_source_fwhm_50coherence_arcmin": _finite_median(
                    g.loc[useful, "source_fwhm_50coherence_arcmin"]
                ),
                "median_phase_cycles_per_0p1deg": _finite_median(
                    g.loc[useful, "phase_cycles_per_source_degree"]
                )
                / 10.0,
                "fresnel_one_scale_time_s": fresnel_time,
                "direct_contrast_sigma": contrast_sigma,
                "locally_searchable": bool(np.count_nonzero(useful) >= 4 and max_span >= 0.15),
                "half_cycle_in_one_dwell": bool(max_span >= 0.5),
            }
        )
    return pd.DataFrame(rows)


def channel_summary(event_metrics: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    catalog = events[events["source_name"].astype(str).str.lower().eq("earth")]
    by = ["antenna", "frequency_band", "frequency_mhz"]
    for keys, g in event_metrics.groupby(by, sort=True):
        ant, band, freq = keys
        n_catalog = len(
            catalog[
                catalog["antenna"].astype(str).eq(str(ant))
                & catalog["frequency_band"].astype(int).eq(int(band))
            ]
        )
        useful = g[g["n_useful_samples"] > 0]
        searched = g[g["locally_searchable"]]
        baseline = _finite_median(useful["median_effective_baseline_km"])
        lam = float(wavelength_km(float(freq)))
        theta = np.deg2rad(0.1)
        log10_coherence_0p1 = -((np.pi * baseline * theta / lam) ** 2) / (4.0 * np.log(2.0) * np.log(10.0))
        rows.append(
            {
                "antenna": ant,
                "frequency_band": int(band),
                "frequency_mhz": float(freq),
                "catalog_occultations": int(n_catalog),
                "observed_occultations_pm4min": int(len(g)),
                "occultations_with_first_lobe_sample": int((g["n_first_lobe_samples"] > 0).sum()),
                "locally_searchable_occultations": int(g["locally_searchable"].sum()),
                "half_cycle_dwell_occultations": int(g["half_cycle_in_one_dwell"].sum()),
                "total_useful_samples": int(g["n_useful_samples"].sum()),
                "median_useful_samples_per_contributing_event": _finite_median(useful["n_useful_samples"]),
                "median_useful_dwells_per_contributing_event": _finite_median(useful["n_useful_dwells"]),
                "median_fringe_period_s": _finite_median(useful["median_useful_fringe_period_s"]),
                "p10_fringe_period_s": _finite_percentile(useful["median_useful_fringe_period_s"], 10),
                "median_bandwidth_coherence": _finite_median(useful["median_useful_bandwidth_coherence"]),
                "median_effective_baseline_km": baseline,
                "median_source_fwhm_50coherence_arcmin": _finite_median(
                    useful["median_source_fwhm_50coherence_arcmin"]
                ),
                "median_phase_cycles_per_0p1deg": _finite_median(useful["median_phase_cycles_per_0p1deg"]),
                "median_fresnel_one_scale_time_s": _finite_median(useful["fresnel_one_scale_time_s"]),
                "median_direct_contrast_sigma": _finite_median(g["direct_contrast_sigma"]),
                "log10_spatial_coherence_if_source_fwhm_0p1deg": log10_coherence_0p1,
                "log10_spatial_coherence_if_source_fwhm_1deg": 100.0 * log10_coherence_0p1,
            }
        )
    return pd.DataFrame(rows)


def time_to_limb_summary(samples: pd.DataFrame) -> pd.DataFrame:
    """Tabulate the exact geometry and instrumental response versus limb time."""
    use = samples[
        _valid(samples["is_valid"])
        & samples["reflection_valid"].to_numpy(dtype=bool)
        & samples["direct_visible"].to_numpy(dtype=bool)
        & (samples["delay_us"].to_numpy(dtype=float) >= 0.0)
        & (samples["delay_us"].to_numpy(dtype=float) <= 1e6 / RV_EFFECTIVE_BANDWIDTH_HZ)
    ].copy()
    if use.empty:
        return pd.DataFrame()
    edges = np.asarray([0.0, 15.0, 30.0, 45.0, 60.0, 90.0, 120.0, 180.0, 240.0])
    use["time_to_limb_bin"] = pd.cut(
        use["oriented_t_sec"], edges, include_lowest=True, right=False
    ).astype(str)
    use["time_to_limb_bin_start_s"] = edges[
        np.clip(np.searchsorted(edges, use["oriented_t_sec"].to_numpy(dtype=float), side="right") - 1, 0, len(edges) - 1)
    ]
    return (
        use.groupby(
            ["antenna", "frequency_band", "frequency_mhz", "time_to_limb_bin_start_s", "time_to_limb_bin"],
            observed=True,
        )
        .agg(
            n_samples=("time", "size"),
            n_events=("event_id", "nunique"),
            median_delay_us=("delay_us", "median"),
            median_bandwidth_coherence=("bandwidth_coherence", "median"),
            median_fringe_period_s=("fringe_period_s", "median"),
            median_temporal_response=("temporal_response", "median"),
            median_effective_baseline_km=("effective_baseline_km", "median"),
            median_source_fwhm_50coherence_arcmin=("source_fwhm_50coherence_arcmin", "median"),
        )
        .reset_index()
    )


def _fit_sample_gate(samples: pd.DataFrame) -> pd.DataFrame:
    return samples[
        _valid(samples["is_valid"])
        & samples["reflection_valid"].to_numpy(dtype=bool)
        & samples["direct_visible"].to_numpy(dtype=bool)
        & (samples["delay_us"].to_numpy(dtype=float) <= 1e6 / RV_EFFECTIVE_BANDWIDTH_HZ)
        & (np.abs(samples["bandwidth_coherence"].to_numpy(dtype=float)) >= 0.2)
        & np.isfinite(samples["z_power"].to_numpy(dtype=float))
    ].copy()


def matched_filter_audit(samples: pd.DataFrame, event_metrics: pd.DataFrame, n_null: int, seed: int) -> pd.DataFrame:
    fit_samples = _fit_sample_gate(samples)
    if fit_samples.empty:
        return pd.DataFrame()
    base = fit_fringe_models(
        fit_samples,
        fit_side="visible",
        phase_modes=["real", "wrong_frequency_low", "wrong_frequency_high"],
        baseline_order=1,
        include_step=True,
        n_bootstrap=0,
        seed=seed,
    )
    base = base[base["event_type"].eq("combined")].copy()
    rng = np.random.default_rng(seed + 1)
    null_rows: dict[tuple[str, int], list[float]] = defaultdict(list)
    group_cols = ["antenna", "frequency_band"]
    for keys, group in fit_samples.groupby(group_cols, sort=True):
        for _ in range(int(n_null)):
            fit = fit_fringe_group(group, "scrambled", 1, True, rng)
            amp = fit.get("fringe_amplitude_z", np.nan)
            if np.isfinite(amp):
                null_rows[(str(keys[0]), int(keys[1]))].append(float(amp))
    metric_lookup = event_metrics.groupby(group_cols).agg(
        median_direct_contrast_sigma=("direct_contrast_sigma", "median")
    )
    rows: list[dict[str, object]] = []
    for _, row in base.iterrows():
        key = (str(row["antenna"]), int(row["frequency_band"]))
        null = np.asarray(null_rows.get(key, []), dtype=float)
        amplitude = float(row["fringe_amplitude_z"])
        empirical_p = (1.0 + np.count_nonzero(null >= amplitude)) / (1.0 + len(null)) if len(null) else np.nan
        contrast = (
            float(metric_lookup.loc[key, "median_direct_contrast_sigma"])
            if key in metric_lookup.index
            else np.nan
        )
        amp_5sigma = 5.0 * float(row["fringe_amplitude_se_z"])
        voltage_ratio = amp_5sigma / (2.0 * contrast) if np.isfinite(contrast) and contrast > 0 else np.nan
        rows.append(
            {
                **row.to_dict(),
                "phase_scramble_trials": int(len(null)),
                "phase_scramble_empirical_p": empirical_p,
                "phase_scramble_p95_amplitude_z": float(np.nanpercentile(null, 95)) if len(null) else np.nan,
                "median_direct_contrast_sigma": contrast,
                "optimistic_5sigma_unattenuated_modulation_over_direct": (
                    amp_5sigma / contrast if np.isfinite(contrast) and contrast > 0 else np.nan
                ),
                "optimistic_5sigma_reflected_voltage_ratio": voltage_ratio,
                "optimistic_5sigma_reflected_power_fraction": voltage_ratio**2 if np.isfinite(voltage_ratio) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def event_type_fit_audit(samples: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Fit ingress and egress separately as a repeatability control."""
    fit_samples = _fit_sample_gate(samples)
    if fit_samples.empty:
        return pd.DataFrame()
    fits = fit_fringe_models(
        fit_samples,
        fit_side="visible",
        phase_modes=["real"],
        baseline_order=1,
        include_step=True,
        n_bootstrap=0,
        seed=seed,
    )
    return fits[~fits["event_type"].eq("combined")].copy()


def incorrect_geometry_audit(samples: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Fit a deliberately flat/tangent path proxy as a geometry null."""
    fit_samples = _fit_sample_gate(samples)
    if fit_samples.empty:
        return pd.DataFrame()
    delta_km, valid = tangent_proxy_path_km(
        fit_samples[SPACECRAFT_COLUMNS].to_numpy(dtype=float),
        fit_samples[EARTH_UNIT_COLUMNS].to_numpy(dtype=float),
    )
    fit_samples["phase_rad"] = 2.0 * np.pi * delta_km / wavelength_km(
        fit_samples["frequency_mhz"].to_numpy(dtype=float)
    )
    fit_samples["bandwidth_coherence"] = np.sinc(
        (delta_km / C_KM_S) * RV_EFFECTIVE_BANDWIDTH_HZ
    )
    fit_samples["proxy_in_first_lobe"] = (
        valid
        & np.isfinite(fit_samples["phase_rad"].to_numpy(dtype=float))
        & (np.abs(fit_samples["bandwidth_coherence"].to_numpy(dtype=float)) >= 0.2)
    )
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed + 2)
    for _, group in fit_samples.groupby(["antenna", "frequency_band"], sort=True):
        accepted = group[group["proxy_in_first_lobe"]].copy()
        fit = fit_fringe_group(accepted, "real", 1, True, rng) if not accepted.empty else {
            "fringe_amplitude_z": np.nan,
            "fringe_snr": np.nan,
            "n_fit_samples": 0,
            "n_fit_events": 0,
        }
        rows.append(
            {
                "source_name": str(group["source_name"].iloc[0]),
                "antenna": str(group["antenna"].iloc[0]),
                "frequency_band": int(group["frequency_band"].iloc[0]),
                "frequency_mhz": float(group["frequency_mhz"].iloc[0]),
                "time_shift_s": 0.0,
                "event_type": "combined",
                "fit_side": "visible",
                "phase_mode": "flat_tangent_proxy",
                **fit,
            }
        )
    return pd.DataFrame(rows)


def event_time_scramble_audit(
    samples: pd.DataFrame,
    real_fits: pd.DataFrame,
    n_trials: int,
    seed: int,
) -> pd.DataFrame:
    """Scramble whole event-geometry tracks among like-sized occultation records."""
    fit_samples = _fit_sample_gate(samples)
    if fit_samples.empty or n_trials <= 0:
        return pd.DataFrame()
    real_lookup = (
        real_fits[real_fits["phase_mode"].eq("real")]
        .set_index(["antenna", "frequency_band"])["fringe_amplitude_z"]
    )
    rng = np.random.default_rng(seed + 3)
    rows: list[dict[str, object]] = []
    for (antenna, band), group in fit_samples.groupby(["antenna", "frequency_band"], sort=True):
        base = group.reset_index(drop=True).copy()
        event_indices: dict[tuple[str, int], list[np.ndarray]] = defaultdict(list)
        for (_, event_type), idx in base.groupby(["event_id", "event_type"], sort=False).groups.items():
            ordered = base.loc[idx].sort_values("oriented_t_sec").index.to_numpy(dtype=int)
            event_indices[(str(event_type), len(ordered))].append(ordered)
        amplitudes: list[float] = []
        for _ in range(int(n_trials)):
            shuffled = base.copy()
            for blocks in event_indices.values():
                if len(blocks) < 2:
                    continue
                donors = rng.permutation(len(blocks))
                for recipient_pos, donor_pos in enumerate(donors):
                    recipient = blocks[recipient_pos]
                    donor = blocks[int(donor_pos)]
                    for column in ["phase_rad", "bandwidth_coherence", "temporal_response"]:
                        shuffled.loc[recipient, column] = base.loc[donor, column].to_numpy()
            fit = fit_fringe_group(shuffled, "real", 1, True, rng)
            amp = fit.get("fringe_amplitude_z", np.nan)
            if np.isfinite(amp):
                amplitudes.append(float(amp))
        key = (str(antenna), int(band))
        real_amp = float(real_lookup.loc[key]) if key in real_lookup.index else np.nan
        null = np.asarray(amplitudes, dtype=float)
        rows.append(
            {
                "antenna": str(antenna),
                "frequency_band": int(band),
                "frequency_mhz": float(base["frequency_mhz"].iloc[0]),
                "event_time_scramble_trials": int(len(null)),
                "real_geometry_amplitude_z": real_amp,
                "event_time_scramble_median_amplitude_z": _finite_median(null),
                "event_time_scramble_p95_amplitude_z": _finite_percentile(null, 95),
                "event_time_scramble_max_amplitude_z": float(np.nanmax(null)) if len(null) else np.nan,
                "event_time_scramble_empirical_p": (
                    (1.0 + np.count_nonzero(null >= real_amp)) / (1.0 + len(null))
                    if len(null) and np.isfinite(real_amp)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def off_limb_control_audit(
    clean: pd.DataFrame,
    earth_events: pd.DataFrame,
    antennas: list[str],
    shifts_s: list[float],
) -> pd.DataFrame:
    """Inventory equal-duration windows centered well away from the true limb."""
    controls = collect_reflection_samples(
        clean,
        earth_events,
        source="earth",
        antennas=antennas,
        bands=[1],
        window_s=120.0,
        inner_s=45.0,
        time_shifts_s=shifts_s,
        geometry_model="spherical_specular",
        max_events_per_band_type=0,
        require_normalized=False,
        valid_only=False,
    )
    if controls.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for keys, group in controls.groupby(["antenna", "frequency_band", "frequency_mhz", "time_shift_s"], sort=True):
        valid = _valid(group["is_valid"])
        direct = group["direct_visible"].to_numpy(dtype=bool)
        exact = group["reflection_valid"].to_numpy(dtype=bool)
        far_from_limb = np.abs(group["direct_limb_angle_deg"].to_numpy(dtype=float)) >= 5.0
        first_lobe = (
            valid
            & direct
            & exact
            & (group["delay_us"].to_numpy(dtype=float) >= 0.0)
            & (group["delay_us"].to_numpy(dtype=float) <= 1e6 / RV_EFFECTIVE_BANDWIDTH_HZ)
        )
        useful = (
            first_lobe
            & (np.abs(group["bandwidth_coherence"].to_numpy(dtype=float)) >= 0.2)
            & group["locally_nyquist_resolved"].to_numpy(dtype=bool)
        )
        rows.append(
            {
                **dict(zip(["antenna", "frequency_band", "frequency_mhz", "time_shift_s"], keys)),
                "n_catalog_events": int(group["event_id"].nunique()),
                "n_valid_samples": int(np.count_nonzero(valid)),
                "n_direct_visible_samples": int(np.count_nonzero(valid & direct)),
                "n_direct_visible_samples_at_least_5deg_from_limb": int(
                    np.count_nonzero(valid & direct & far_from_limb)
                ),
                "n_first_lobe_samples": int(np.count_nonzero(first_lobe)),
                "n_first_lobe_samples_at_least_5deg_from_limb": int(
                    np.count_nonzero(first_lobe & far_from_limb)
                ),
                "n_locally_resolved_first_lobe_samples": int(np.count_nonzero(useful)),
                "n_events_with_locally_resolved_first_lobe_sample": int(group.loc[useful, "event_id"].nunique()),
                "median_delay_us_for_direct_visible_samples": _finite_median(group.loc[valid & direct, "delay_us"]),
                "median_abs_bandwidth_coherence_for_direct_visible_samples": _finite_median(
                    np.abs(group.loc[valid & direct, "bandwidth_coherence"])
                ),
                "median_delay_us_at_least_5deg_from_limb": _finite_median(
                    group.loc[valid & direct & far_from_limb, "delay_us"]
                ),
                "median_abs_bandwidth_coherence_at_least_5deg_from_limb": _finite_median(
                    np.abs(group.loc[valid & direct & far_from_limb, "bandwidth_coherence"])
                ),
            }
        )
    return pd.DataFrame(rows)


def select_representative(metrics: pd.DataFrame) -> pd.DataFrame:
    """Choose a typical, positive-contrast, one-dwell lower-V 0.45 MHz event."""
    all_lower = metrics[
        metrics["antenna"].eq("rv2_coarse")
        & metrics["frequency_band"].eq(1)
        & (metrics["n_useful_samples"] > 0)
        & np.isfinite(metrics["median_useful_fringe_period_s"])
    ]
    if all_lower.empty:
        return all_lower
    target_period = _finite_median(all_lower["median_useful_fringe_period_s"])
    candidates = all_lower[
        all_lower["locally_searchable"]
        & (all_lower["direct_contrast_sigma"] > 0.0)
        & all_lower["n_useful_dwells"].eq(1)
    ].copy()
    if candidates.empty:
        candidates = all_lower.copy()
    candidates["representative_score"] = np.abs(
        np.log(candidates["median_useful_fringe_period_s"] / target_period)
    )
    return candidates.sort_values(
        ["representative_score", "max_useful_dwell_phase_span_cycles"],
        ascending=[True, False],
    ).head(1)


def plot_representative(samples: pd.DataFrame, metrics: pd.DataFrame, out_dir: Path) -> Path | None:
    candidates = select_representative(metrics)
    if candidates.empty:
        return None
    best = candidates.iloc[0]
    g = samples[
        samples["antenna"].eq(best["antenna"])
        & samples["frequency_band"].eq(int(best["frequency_band"]))
        & samples["event_id"].eq(best["event_id"])
        & samples["event_type"].eq(best["event_type"])
    ].sort_values("oriented_t_sec")
    if g.empty:
        return None
    valid = _valid(g["is_valid"])
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 9), sharex=True)
    axes[0].plot(g["oriented_t_sec"], g["delay_us"], color="#4c78a8")
    axes[0].axhline(1e6 / RV_EFFECTIVE_BANDWIDTH_HZ, color="black", ls="--", lw=0.9, label="40 kHz first zero")
    axes[0].set_ylabel("delay (µs)")
    axes[0].legend(frameon=False)
    axes[1].plot(g["oriented_t_sec"], g["bandwidth_coherence"], color="#f58518")
    axes[1].set_ylabel("bandwidth sinc")
    axes[2].scatter(
        g.loc[valid, "oriented_t_sec"],
        g.loc[valid, "fringe_period_s"],
        s=22,
        color="#54a24b",
        label="predicted fringe period",
    )
    fresnel_time = float(best["fresnel_one_scale_time_s"])
    if np.isfinite(fresnel_time):
        axes[2].axhline(fresnel_time, color="#e45756", ls="--", label="one Fresnel-scale time")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("time scale (s)")
    axes[2].set_xlabel("seconds from limb; positive is Earth-visible side")
    axes[2].legend(frameon=False)
    for ax in axes:
        ax.axvline(0.0, color="0.4", lw=0.8)
        ax.grid(alpha=0.22)
    fig.suptitle(
        f"Representative 0.45 MHz lower-V event: {best['event_type']} {pd.Timestamp(best['predicted_event_time'])}"
    )
    fig.tight_layout()
    path = out_dir / "representative_045mhz_geometry.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_channel_summary(summary: pd.DataFrame, out_dir: Path) -> Path | None:
    if summary.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for antenna, g in summary.groupby("antenna", sort=True):
        g = g.sort_values("frequency_mhz")
        label = "lower V" if antenna == "rv2_coarse" else "upper V"
        axes[0].plot(g["frequency_mhz"], g["locally_searchable_occultations"], marker="o", label=label)
        axes[1].plot(g["frequency_mhz"], g["median_fringe_period_s"], marker="o", label=label)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("frequency (MHz)")
        ax.grid(alpha=0.22)
        ax.legend(frameon=False)
    axes[0].set_ylabel("locally searchable occultations")
    axes[1].set_ylabel("median fringe period in retained samples (s)")
    axes[1].set_yscale("log")
    fig.suptitle("RAE-2 lunar-reflection sampling feasibility")
    fig.tight_layout()
    path = out_dir / "channel_feasibility_summary.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report(
    out_dir: Path,
    sampling: pd.DataFrame,
    summary: pd.DataFrame,
    metrics: pd.DataFrame,
    fits: pd.DataFrame,
    event_type_fits: pd.DataFrame,
    time_scrambles: pd.DataFrame,
    off_limb_controls: pd.DataFrame,
    limb_summary: pd.DataFrame,
    representative_path: Path | None,
    channel_plot: Path | None,
) -> Path:
    sample = sampling.iloc[0]
    band_table = summary.sort_values(["frequency_mhz", "antenna"])
    lower_045 = band_table[(band_table["antenna"].eq("rv2_coarse")) & (band_table["frequency_band"].eq(1))]
    rep = select_representative(metrics)
    real_fits = fits[fits["phase_mode"].eq("real")].copy() if not fits.empty else fits
    control_fits = fits[~fits["phase_mode"].eq("real")].copy() if not fits.empty else fits
    lower_event_types = event_type_fits[
        event_type_fits["antenna"].eq("rv2_coarse")
        & event_type_fits["frequency_band"].eq(1)
    ] if not event_type_fits.empty else event_type_fits
    lower_time_scrambles = time_scrambles[
        time_scrambles["frequency_band"].eq(1)
    ] if not time_scrambles.empty else time_scrambles
    lower_limb = limb_summary[
        limb_summary["antenna"].eq("rv2_coarse")
        & limb_summary["frequency_band"].eq(1)
    ].sort_values("time_to_limb_bin_start_s") if not limb_summary.empty else limb_summary
    lines = [
        "# RAE-2 Earth–Moon Reflection Interferometry Feasibility",
        "",
        "## Decision",
        "",
        "**No-go for a statistically credible reflection detection from the present RV archive; go for an explicitly labelled exploratory/null search.**",
        "",
        "The coarse archive retains real sub-dwell samples, so finite receiver bandwidth does not erase the effect everywhere.",
        "The decisive limitations are the clustered scan sampling, event-dependent arcminute-scale source coherence requirement,",
        "unknown AKR/TKR source direction, and a Fresnel-diffraction time scale comparable to the predicted fringe time scale.",
        "A coherent Earth-centre matched filter is therefore an optimistic diagnostic, not a detection statistic with a defensible physical phase prior.",
        "",
        "## What is actually archived",
        "",
        f"- Master rows audited: {int(sample['master_rows']):,}.",
        f"- Typical coarse dwell: {int(sample['coarse_samples_per_typical_dwell'])} retained readings, spaced {sample['median_within_dwell_cadence_s']:.3f} s over {sample['median_typical_dwell_span_s']:.3f} s.",
        f"- Same-band revisit: {sample['median_same_band_revisit_s']:.3f} s; the intervening gap is about {sample['median_same_band_revisit_s'] - sample['median_typical_dwell_span_s']:.1f} s.",
        f"- Retained nonzero fine values: {int(sample['rv1_fine_nonzero_samples']):,} (upper V) and {int(sample['rv2_fine_nonzero_samples']):,} (lower V), about one value per dwell per antenna.",
        "- Consequently the seven coarse readings survive separately; the fine stream cannot resolve a tens-of-seconds fringe.",
        "",
        "NASA documentation independently gives a 15.4 s frequency dwell, eight coarse and two fine telemetry samples,",
        "and warns that the first coarse sample is unreliable. The seven samples in this file are exactly consistent with removal of that first reading.",
        "The RV effective bandwidth is 40 kHz, the coarse post-detection time constant is 0.1 s, and a nine-frequency scan takes 138 s.",
        "",
        "## Event and channel feasibility",
        "",
        _markdown(
            band_table,
            [
                "antenna",
                "frequency_mhz",
                "catalog_occultations",
                "observed_occultations_pm4min",
                "occultations_with_first_lobe_sample",
                "locally_searchable_occultations",
                "half_cycle_dwell_occultations",
                "total_useful_samples",
                "median_useful_samples_per_contributing_event",
                "median_fringe_period_s",
                "median_bandwidth_coherence",
                "median_direct_contrast_sigma",
            ],
        ),
        "",
        "`Locally searchable` requires at least four valid first-lobe samples and at least 0.15 predicted phase cycle within one dwell.",
        "This is only a local sampling gate. It does not claim that phase can be connected across the roughly 126 s inter-dwell gap.",
        "At 0.45 MHz lower V, the searchable set divides into 224 disappearances and 217 reappearances; 126 and 125, respectively, contain at least half a phase cycle in one dwell.",
        "The 0.70 MHz channel has more locally sampled events, but its median direct Earth contrast is -0.14σ upper V and 0.00σ lower V, versus +2.37σ and +2.25σ at 0.45 MHz.",
        "Together with the slower 29–30 s fringes and less severe spatial-coherence requirement, that makes 0.45 MHz the best existing channel for this specific terrestrial-emission test.",
        "",
        "## Representative 0.45 MHz geometry",
        "",
    ]
    if not rep.empty:
        lines.append(
            _markdown(
                rep,
                [
                    "event_type",
                    "predicted_event_time",
                    "n_useful_samples",
                    "n_useful_dwells",
                    "max_useful_dwell_phase_span_cycles",
                    "median_useful_delay_us",
                    "median_useful_bandwidth_coherence",
                    "median_useful_fringe_period_s",
                    "fresnel_one_scale_time_s",
                    "median_effective_baseline_km",
                    "median_source_fwhm_50coherence_arcmin",
                    "median_phase_cycles_per_0p1deg",
                ],
            )
        )
    else:
        lines.append("No valid lower-V 0.45 MHz representative event was found.")
    lines.extend(
        [
            "",
            "The rectangular 40 kHz bandpass gives its first zero at 25 µs (7.495 km excess path).",
            "The 0.1 s coarse-channel response produces negligible attenuation for the retained tens-of-seconds fringes.",
            "The exact spherical solver enforces the vector reflection law and does not use the flat 2h cos(theta) path approximation.",
            "The exact time-dependent delay, fringe rate, bandwidth response, and effective baseline are retained per sample in `reflection_event_samples.npy`.",
            "The lower-V 0.45 MHz medians versus time from the limb are:",
            "",
            _markdown(
                lower_limb,
                [
                    "time_to_limb_bin",
                    "n_samples",
                    "n_events",
                    "median_delay_us",
                    "median_bandwidth_coherence",
                    "median_fringe_period_s",
                    "median_effective_baseline_km",
                ],
            ),
            "",
            "## Source position and spatial coherence",
            "",
        ]
    )
    if not lower_045.empty:
        b = lower_045.iloc[0]
        lines.extend(
            [
                f"At 0.45 MHz lower V, the median effective projected baseline in retained samples is {b['median_effective_baseline_km']:.1f} km.",
                f"A circular Gaussian source reaches 50% visibility at only {b['median_source_fwhm_50coherence_arcmin']:.3f} arcmin FWHM.",
                f"A 0.1° shift changes the predicted phase by a median {b['median_phase_cycles_per_0p1deg']:.2f} cycles.",
                f"For a 0.1° Gaussian source the median-baseline visibility is about 10^{b['log10_spatial_coherence_if_source_fwhm_0p1deg']:.2f}; for a 1° source it is about 10^{b['log10_spatial_coherence_if_source_fwhm_1deg']:.2f}.",
                "Auroral emission spread over a substantial fraction of the terrestrial auroral region is therefore resolved out.",
                "Kaguya proves that compact/narrowband AKR components can sometimes remain coherent, but that precedent does not supply RAE-2 with an event-specific source position and its 40 kHz channel averages frequency structure.",
            ]
        )
    lines.extend(
        [
            "",
            "The source offset cannot be represented by one harmless constant phase: it also shifts the fringe rate and the occultation/diffraction transition.",
            "An event-specific 2-D source-position search would require a very fine grid and a large trials correction; with usually one short dwell in the first lobe, that nuisance fit is underconstrained.",
            "",
            "## Diffraction comparison",
            "",
            "The output samples include finite-distance ray clearance, the effective Fresnel distance, the complex knife-edge field, and its power.",
            "The matched-filter nuisance model uses the point-source diffraction power instead of a hard step, and the reflection quadratures include the complex diffracted direct field.",
            "The representative table compares one Fresnel-scale crossing time directly with the reflection fringe period.",
            "Their comparable scales, combined with one 11.55 s dwell and a roughly 126 s gap, prevent clean event-by-event separation without an external source-position prior.",
            "Extended-source diffraction would be smoother than the point-source model and adds another nuisance scale.",
            "At 0.45 MHz and a nominal 1,050 km altitude, even the normal-incidence first-Fresnel diameter is about 53 km and becomes elongated at grazing incidence.",
            "No lunar roughness map or measured RV filter transfer function is present in this archive, so the calculation uses a smooth sphere and a rectangular 40 kHz passband; both assumptions are optimistic.",
            "",
            "## Optimistic Earth-centre matched-filter audit",
            "",
        ]
    )
    if real_fits.empty:
        lines.append("No matched-filter rows passed the sample gate.")
    else:
        lines.append(
            _markdown(
                real_fits.sort_values(["frequency_mhz", "antenna"]),
                [
                    "antenna",
                    "frequency_mhz",
                    "phase_mode",
                    "n_fit_events",
                    "n_fit_samples",
                    "fringe_amplitude_z",
                    "fringe_snr",
                    "phase_scramble_empirical_p",
                    "optimistic_5sigma_reflected_power_fraction",
                ],
            )
        )
        lines.extend(
            [
                "",
                "No real-geometry row exceeds 2σ. The nominal upper-V 0.45 MHz phase-scramble tail probability is small, but a wrong-frequency control is stronger, so it is not a reflection detection.",
                "The deliberately flat/tangent geometry and wrong-frequency controls are:",
                "",
                _markdown(
                    control_fits.sort_values(["frequency_mhz", "antenna", "phase_mode"]),
                    [
                        "antenna",
                        "frequency_mhz",
                        "phase_mode",
                        "n_fit_samples",
                        "fringe_amplitude_z",
                        "fringe_snr",
                    ],
                ),
                "The flat/tangent proxy leaves no 0.45 or 0.70 MHz samples inside the same bandwidth gate; it is not an acceptable approximation to the spherical solution.",
                "",
                "Lower-V 0.45 MHz ingress/egress repeatability fits are:",
                "",
                _markdown(
                    lower_event_types,
                    ["event_type", "n_fit_events", "n_fit_samples", "fringe_amplitude_z", "fringe_snr"],
                ),
                "",
                "Whole-event time/geometry scrambles (phase tracks are reassigned only among events with the same type and sample count) give:",
                "",
                _markdown(
                    lower_time_scrambles,
                    [
                        "antenna",
                        "event_time_scramble_trials",
                        "real_geometry_amplitude_z",
                        "event_time_scramble_median_amplitude_z",
                        "event_time_scramble_p95_amplitude_z",
                        "event_time_scramble_empirical_p",
                    ],
                ),
                "",
                "Equal-duration 0.45 MHz windows centered 10 minutes away from the predicted limb give:",
                "",
                _markdown(
                    off_limb_controls,
                    [
                        "antenna",
                        "time_shift_s",
                        "n_valid_samples",
                        "n_direct_visible_samples",
                        "n_direct_visible_samples_at_least_5deg_from_limb",
                        "n_first_lobe_samples_at_least_5deg_from_limb",
                        "median_delay_us_at_least_5deg_from_limb",
                        "median_abs_bandwidth_coherence_at_least_5deg_from_limb",
                    ],
                ),
                "The few nominal first-lobe samples in the full shifted windows are cases that are still geometrically close to a limb; after requiring at least 5° separation, none survive the first-lobe gate.",
            ]
        )
        upper_045_fit = real_fits[
            real_fits["antenna"].eq("rv1_coarse") & real_fits["frequency_band"].eq(1)
        ]
        lower_045_fit = real_fits[
            real_fits["antenna"].eq("rv2_coarse") & real_fits["frequency_band"].eq(1)
        ]
        upper_045_band = band_table[
            band_table["antenna"].eq("rv1_coarse") & band_table["frequency_band"].eq(1)
        ]
        if not upper_045_fit.empty and not lower_045_fit.empty and not upper_045_band.empty:
            upper_threshold = float(upper_045_fit.iloc[0]["optimistic_5sigma_reflected_power_fraction"])
            lower_threshold = float(lower_045_fit.iloc[0]["optimistic_5sigma_reflected_power_fraction"])
            upper_band = upper_045_band.iloc[0]
            visibility_0p1deg = 10.0 ** float(
                upper_band["log10_spatial_coherence_if_source_fwhm_0p1deg"]
            )
            visibility_1deg = 10.0 ** float(
                upper_band["log10_spatial_coherence_if_source_fwhm_1deg"]
            )
            lines.extend(
                [
                    "",
                    f"Under the optimistic compact-source/common-phase model, a 5σ detection needs reflected power about {upper_threshold:.2f} times the direct Earth component in upper V and {lower_threshold:.0f} times in lower V.",
                    f"Even a 0.1° Gaussian raises the upper-V threshold to about {upper_threshold / visibility_0p1deg**2:.2f}; a 1° source raises it to about {upper_threshold / visibility_1deg**2:.2e} times direct power before lunar-surface losses.",
                ]
            )
    lines.extend(
        [
            "",
            "The quoted reflected-power threshold is optimistic: it assumes Earth-centre geometry, a common phase across events,",
            "the rectangular bandpass, a compact coherent source, and no surface-coherence loss. Divide the voltage visibility by",
            "source and surface coherence before translating it into physical reflectivity; doing so makes the realistic threshold much worse.",
            "A real claim would require the true geometry to beat phase scrambles, wrong frequencies, event-time scrambles, upper/lower-V controls,",
            "ingress/egress splits, and incorrect source offsets after a predeclared trials correction. The present archive cannot constrain all of these nuisances simultaneously.",
            "",
            "## Practical exploratory pipeline",
            "",
            "1. Keep all seven coarse values; never average a dwell or use the 60 s profile products.",
            "2. Recompute spherical specular delay and knife-edge diffraction at every timestamp.",
            "3. Restrict to the Earth-visible first bandpass lobe and retain the signed sinc weight.",
            "4. Fit smooth baseline + diffraction + two reflection quadratures within each event.",
            "5. Combine per-event likelihoods, not time stacks, and calibrate the entire source-offset search with scrambled-event and wrong-geometry runs.",
            "6. Treat any result as exploratory unless it repeats in ingress/egress and frequency scaling while failing the null geometries.",
            "",
            "Reproduce this audit with:",
            "",
            "```bash",
            "conda run --no-capture-output -n luseepy_env python scripts/run_rae2_lunar_reflection_feasibility.py",
            "```",
            "",
            "## Files",
            "",
            "- `sampling_structure_audit.csv`",
            "- `reflection_event_samples.npy`",
            "- `event_feasibility.csv`",
            "- `channel_feasibility.csv`",
            "- `time_to_limb_geometry.csv`",
            "- `optimistic_matched_filter_audit.csv`",
            "- `ingress_egress_fit_audit.csv`",
            "- `event_time_scramble_audit.csv`",
            "- `off_limb_control_audit.csv`",
        ]
    )
    if representative_path:
        lines.append(f"- `{representative_path.name}`")
    if channel_plot:
        lines.append(f"- `{channel_plot.name}`")
    lines.extend(
        [
            "",
            "## Primary references",
            "",
            "- Gorham et al., *Lunar Reflective Interferometry*: https://arxiv.org/abs/2608.10284",
            "- NASA RAE-2 receiver/scanning description: https://ntrs.nasa.gov/citations/19770024123",
            "- NASA RAE-2 instrumentation report: https://ntrs.nasa.gov/citations/19750010212",
            "- Kaguya AKR reflection/interference study: https://doi.org/10.5047/eps.2011.01.005",
        ]
    )
    path = out_dir / "rae2_lunar_reflection_feasibility_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", default=str(DEFAULT_MASTER))
    parser.add_argument("--clean", default=str(DEFAULT_CLEAN))
    parser.add_argument("--events", default=str(DEFAULT_EVENTS))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--bands", default="1,2,3,4,5")
    parser.add_argument("--antennas", default="rv1_coarse,rv2_coarse")
    parser.add_argument("--window-s", type=float, default=240.0)
    parser.add_argument("--phase-scramble-trials", type=int, default=128)
    parser.add_argument("--event-time-scramble-trials", type=int, default=32)
    parser.add_argument("--off-limb-shifts-s", default="-600,600")
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--skip-raw-audit", action="store_true")
    parser.add_argument("--reuse-samples", action="store_true")
    parser.add_argument("--reuse-fits", action="store_true")
    parser.add_argument("--refresh-off-limb-controls", action="store_true")
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    bands = [int(x) for x in args.bands.split(",") if x.strip()]
    antennas = [x.strip() for x in args.antennas.split(",") if x.strip()]
    config = {
        **vars(args),
        "effective_bandwidth_hz": RV_EFFECTIVE_BANDWIDTH_HZ,
        "coarse_time_constant_s": RV_COARSE_TIME_CONSTANT_S,
        "software_versions": software_versions(),
    }
    write_json(out_dir / "run_config.json", config)

    sampling_path = out_dir / "sampling_structure_audit.csv"
    if args.reuse_samples and sampling_path.exists():
        sampling = pd.read_csv(sampling_path)
    elif args.skip_raw_audit:
        sampling = pd.DataFrame(
            [
                {
                    "master_rows": np.nan,
                    "n_frequency_dwells": np.nan,
                    "coarse_samples_per_typical_dwell": 7,
                    "typical_dwell_fraction": np.nan,
                    "median_within_dwell_cadence_s": 1.925,
                    "median_typical_dwell_span_s": 11.55,
                    "median_same_band_revisit_s": 138.23,
                    "rv1_fine_nonzero_samples": np.nan,
                    "rv2_fine_nonzero_samples": np.nan,
                    "rv1_fine_samples_per_dwell": np.nan,
                    "rv2_fine_samples_per_dwell": np.nan,
                }
            ]
        )
    else:
        sampling = audit_raw_sampling(Path(args.master))
    sampling.to_csv(sampling_path, index=False)

    events = read_table(args.events, parse_dates=["predicted_event_time"], low_memory=False)
    earth_events = events[
        events["source_name"].astype(str).str.lower().eq("earth")
        & events["frequency_band"].astype(int).isin(bands)
        & events["antenna"].astype(str).isin(antennas)
    ].copy()

    sample_path = out_dir / "reflection_event_samples.npy"
    clean: pd.DataFrame | None = None
    if args.reuse_samples and sample_path.exists():
        samples = read_table(sample_path)
        contrast_path = out_dir / "event_direct_contrast.csv"
        contrasts = pd.read_csv(contrast_path) if contrast_path.exists() else normalize_events(samples, float(args.window_s))[1]
    else:
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
            clean["frequency_band"].astype(int).isin(bands)
            & clean["antenna"].astype(str).isin(antennas)
        ].copy()
        samples = collect_reflection_samples(
            clean,
            earth_events,
            source="earth",
            antennas=antennas,
            bands=bands,
            window_s=float(args.window_s),
            inner_s=45.0,
            time_shifts_s=[0.0],
            geometry_model="spherical_specular",
            max_events_per_band_type=0,
            require_normalized=False,
            valid_only=False,
        )
        samples = attach_edge_diffraction(samples)
        samples, contrasts = normalize_events(samples, float(args.window_s))
        write_table(samples, sample_path, index=False)
        contrasts.to_csv(out_dir / "event_direct_contrast.csv", index=False)

    off_limb_path = out_dir / "off_limb_control_audit.csv"
    if args.reuse_fits and off_limb_path.exists() and not args.refresh_off_limb_controls:
        off_limb_controls = pd.read_csv(off_limb_path)
    else:
        if clean is None:
            off_limb_usecols = [
                "time",
                "frequency_band",
                "antenna",
                "power",
                "is_valid",
                *SPACECRAFT_COLUMNS,
                *EARTH_UNIT_COLUMNS,
            ]
            clean = read_table(args.clean, usecols=off_limb_usecols, parse_dates=["time"], low_memory=False)
            clean = clean[
                clean["frequency_band"].astype(int).eq(1)
                & clean["antenna"].astype(str).isin(antennas)
            ].copy()
        shifts = [float(value) for value in args.off_limb_shifts_s.split(",") if value.strip()]
        off_limb_controls = off_limb_control_audit(clean, earth_events, antennas, shifts)
        off_limb_controls.to_csv(off_limb_path, index=False)

    metrics = event_feasibility(samples, contrasts)
    metrics.to_csv(out_dir / "event_feasibility.csv", index=False)
    summary = channel_summary(metrics, earth_events)
    summary.to_csv(out_dir / "channel_feasibility.csv", index=False)
    limb_summary = time_to_limb_summary(samples)
    limb_summary.to_csv(out_dir / "time_to_limb_geometry.csv", index=False)
    fit_path = out_dir / "optimistic_matched_filter_audit.csv"
    if args.reuse_fits and fit_path.exists():
        fits = pd.read_csv(fit_path)
        fits = fits[~fits["phase_mode"].eq("flat_tangent_proxy")].copy()
    else:
        fits = matched_filter_audit(samples, metrics, int(args.phase_scramble_trials), int(args.seed))
    wrong_geometry = incorrect_geometry_audit(samples, int(args.seed))
    fits = pd.concat([fits, wrong_geometry], ignore_index=True, sort=False)
    fits.to_csv(out_dir / "optimistic_matched_filter_audit.csv", index=False)
    event_type_fits = event_type_fit_audit(samples, int(args.seed))
    event_type_fits.to_csv(out_dir / "ingress_egress_fit_audit.csv", index=False)
    time_scramble_path = out_dir / "event_time_scramble_audit.csv"
    if args.reuse_fits and time_scramble_path.exists():
        time_scrambles = pd.read_csv(time_scramble_path)
    else:
        time_scrambles = event_time_scramble_audit(
            samples,
            fits,
            int(args.event_time_scramble_trials),
            int(args.seed),
        )
        time_scrambles.to_csv(time_scramble_path, index=False)

    representative_path = plot_representative(samples, metrics, out_dir)
    channel_plot = plot_channel_summary(summary, out_dir)
    report = write_report(
        out_dir,
        sampling,
        summary,
        metrics,
        fits,
        event_type_fits,
        time_scrambles,
        off_limb_controls,
        limb_summary,
        representative_path,
        channel_plot,
    )
    print(report)


if __name__ == "__main__":
    main()
