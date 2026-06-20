#!/usr/bin/env python
"""Weak hierarchical event-offset occultation model for lower-V profiles.

Model, per source/frequency/control curve:

    z_ij = b_event[j] + A h_ij(tau) + eps_ij

where `b_event[j]` is a per-event nuisance offset with ridge shrinkage and
`h_ij` is the positive-source occultation template. There are no event-level
trend lines and no flexible background shapes. The only nuisance term is a
constant offset per event.
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
from rylevonberg.table_io import read_table

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.constants import FREQUENCY_MAP_MHZ  # noqa: E402
from rylevonberg.stackfit import stacked_event_template  # noqa: E402
from rylevonberg.util import ensure_dir, robust_sigma, software_versions, write_json  # noqa: E402
from scripts.run_lower_v_stackfirst_detection_attempt import (  # noqa: E402
    ANTENNA,
    _load_clean_groups,
    _load_event_table,
    collect_profiles,
)


DEFAULT_OUT = ROOT / "outputs/lower_v_hierarchical_event_offset_model_v1"
EARTH_EVENT_ROWS = ROOT / "outputs/lower_v_control_manifold_earth_positive_control_v1/lower_v_stackfirst_event_rows.csv"
SOURCE_LABEL = {"earth": "Earth", "sun": "Sun", "fornax_a": "Fornax-A"}
EVENT_ORDER = {"disappearance": 0, "reappearance": 1}
GROUP_COLS = ["analysis_source", "control_family", "control_id", "frequency_band", "frequency_mhz"]


def _read(path: Path, **kwargs) -> pd.DataFrame:
    return read_table(path, low_memory=False, **kwargs)


def _robust_sem(values: pd.Series | np.ndarray) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size <= 1:
        return np.nan
    scale = robust_sigma(vals - np.nanmedian(vals))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.nanstd(vals, ddof=1))
    return float(scale / np.sqrt(vals.size)) if np.isfinite(scale) and scale > 0 else np.nan


def _bic(wrss: float, n: int, effective_params: float) -> float:
    wrss = max(float(wrss), np.finfo(float).tiny)
    n = max(int(n), 1)
    return float(n * np.log(wrss / n) + float(effective_params) * np.log(n))


def _fit_design(X: np.ndarray, y: np.ndarray, weights: np.ndarray, penalty_diag: np.ndarray) -> dict[str, object]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(weights, dtype=float)
    w = np.where(np.isfinite(w) & (w > 0), w, 1.0)
    median_w = float(np.nanmedian(w[np.isfinite(w) & (w > 0)])) if np.any(np.isfinite(w) & (w > 0)) else 1.0
    if not np.isfinite(median_w) or median_w <= 0:
        median_w = 1.0
    w = w / median_w
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    penalty_diag = np.asarray(penalty_diag, dtype=float)
    penalty_diag = np.where(np.isfinite(penalty_diag) & (penalty_diag >= 0), penalty_diag, 0.0)
    xtw = X.T * w
    xtwx = xtw @ X
    normal = xtwx + np.diag(penalty_diag + 1e-8)
    rhs = xtw @ y
    try:
        beta = np.linalg.solve(normal, rhs)
        inv = np.linalg.inv(normal)
    except np.linalg.LinAlgError:
        beta, *_ = np.linalg.lstsq(normal, rhs, rcond=None)
        try:
            inv = np.linalg.pinv(normal, hermitian=True)
        except np.linalg.LinAlgError:
            inv = np.diag(1.0 / np.maximum(np.diag(normal), 1e-8))
    resid = y - X @ beta
    wrss = float(np.sum(w * resid**2))
    # Ridge effective parameter count: trace((X'WX + P)^-1 X'WX).
    eff = float(np.trace(inv @ xtwx))
    dof = max(float(len(y)) - eff, 1.0)
    sigma2 = wrss / dof
    cov = inv * sigma2
    return {"beta": beta, "resid": resid, "wrss": wrss, "cov": cov, "effective_params": eff, "sigma2": sigma2}


def _group_arrays(grp: pd.DataFrame) -> dict[str, object] | None:
    g = grp.copy()
    g["event_sort"] = g["event_type"].astype(str).map(EVENT_ORDER).fillna(99).astype(int)
    g = g.sort_values(["event_sort", "event_uid", "t_bin_sec"]).reset_index(drop=True)
    y = pd.to_numeric(g["z_power"], errors="coerce").to_numpy(dtype=float)
    t = pd.to_numeric(g["t_bin_sec"], errors="coerce").to_numpy(dtype=float)
    n_samples = pd.to_numeric(g.get("n_samples", 1), errors="coerce").fillna(1.0).to_numpy(dtype=float)
    event_codes, event_labels = pd.factorize(g["event_uid"].astype(str), sort=True)
    mask = np.isfinite(y) & np.isfinite(t) & (event_codes >= 0)
    if np.count_nonzero(mask) < 8:
        return None
    g = g.loc[mask].reset_index(drop=True)
    return {
        "frame": g,
        "y": y[mask],
        "t": t[mask],
        "weights": np.sqrt(np.maximum(n_samples[mask], 1.0)),
        "event_codes": event_codes[mask],
        "event_labels": np.asarray(event_labels, dtype=object),
    }


def _normalized_weights(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    w = np.where(np.isfinite(w) & (w > 0), w, 1.0)
    median_w = float(np.nanmedian(w[np.isfinite(w) & (w > 0)])) if np.any(np.isfinite(w) & (w > 0)) else 1.0
    if not np.isfinite(median_w) or median_w <= 0:
        median_w = 1.0
    return w / median_w


def _closed_form_random_intercept_fit(
    y: np.ndarray,
    template: np.ndarray,
    event_codes: np.ndarray,
    weights: np.ndarray,
    event_offset_penalty: float,
) -> dict[str, object]:
    """Fit y = A template + event_offset using event-level ridge offsets.

    The event offsets are eliminated analytically. For a fixed amplitude A, the
    best offset for event e is:

        b_e(A) = sum_i w_i (y_i - A h_i) / (sum_i w_i + lambda)

    Substituting this back leaves a one-parameter quadratic in A.
    """

    y = np.asarray(y, dtype=float)
    h = np.asarray(template, dtype=float)
    codes = np.asarray(event_codes, dtype=int)
    w = _normalized_weights(weights)
    mask = np.isfinite(y) & np.isfinite(h) & np.isfinite(w) & (codes >= 0)
    y = y[mask]
    h = h[mask]
    w = w[mask]
    codes = codes[mask]
    n = int(y.size)
    n_events = int(np.max(codes) + 1) if n else 0
    if n < 8 or n_events <= 0:
        return {
            "amplitude": np.nan,
            "uncertainty": np.nan,
            "offsets": np.array([]),
            "resid": np.array([]),
            "wrss": np.nan,
            "null_wrss": np.nan,
            "effective_params": np.nan,
            "null_effective_params": np.nan,
        }
    lam = max(float(event_offset_penalty), 0.0)
    sw = np.bincount(codes, weights=w, minlength=n_events)
    sy = np.bincount(codes, weights=w * y, minlength=n_events)
    sh = np.bincount(codes, weights=w * h, minlength=n_events)
    syy = np.bincount(codes, weights=w * y * y, minlength=n_events)
    syh = np.bincount(codes, weights=w * y * h, minlength=n_events)
    shh = np.bincount(codes, weights=w * h * h, minlength=n_events)
    denom = sw + lam
    denom = np.where(denom > 0, denom, np.inf)
    null_wrss = float(np.sum(syy - (sy * sy) / denom))
    a = float(np.sum(shh - (sh * sh) / denom))
    b = float(np.sum(syh - (sy * sh) / denom))
    amp = b / a if np.isfinite(a) and abs(a) > 1e-12 else np.nan
    offsets = (sy - amp * sh) / denom if np.isfinite(amp) else np.full(n_events, np.nan)
    resid = y - amp * h - offsets[codes] if np.isfinite(amp) else np.full(n, np.nan)
    wrss = float(np.sum(w * resid**2)) if np.isfinite(amp) else np.nan
    offset_eff = float(np.sum(sw / denom))
    eff = offset_eff + (1.0 if np.isfinite(amp) else 0.0)
    dof = max(float(n) - eff, 1.0)
    sigma2 = wrss / dof if np.isfinite(wrss) else np.nan
    se = float(np.sqrt(sigma2 / a)) if np.isfinite(sigma2) and np.isfinite(a) and a > 0 else np.nan
    return {
        "amplitude": float(amp) if np.isfinite(amp) else np.nan,
        "uncertainty": se,
        "offsets": offsets,
        "resid": resid,
        "wrss": wrss,
        "null_wrss": null_wrss,
        "template_information": a,
        "template_information_per_point": float(a / n) if n > 0 and np.isfinite(a) else np.nan,
        "effective_params": eff,
        "null_effective_params": offset_eff,
    }


def fit_event_offset_group(
    grp: pd.DataFrame,
    transition_durations: list[float],
    event_offset_penalty: float,
    fixed_tau_s: float | None = None,
) -> tuple[dict[str, object], pd.DataFrame]:
    arrays = _group_arrays(grp)
    if arrays is None:
        return (
            {
                "n_points": int(len(grp)),
                "n_events": int(grp["event_uid"].nunique()) if "event_uid" in grp.columns else 0,
                "amplitude": np.nan,
                "uncertainty": np.nan,
                "event_offset_fit_snr": np.nan,
                "best_transition_duration_s": np.nan,
                "delta_bic": np.nan,
            },
            pd.DataFrame(),
        )
    g = arrays["frame"]
    y = arrays["y"]
    t = arrays["t"]
    weights = arrays["weights"]
    event_codes = arrays["event_codes"]
    n_events = int(np.max(event_codes) + 1)
    event_types = g["event_type"].astype(str).to_numpy()

    taus = [float(fixed_tau_s)] if fixed_tau_s is not None else [float(x) for x in transition_durations]
    best: dict[str, object] | None = None
    best_adjusted = pd.DataFrame()
    for tau in taus:
        template = stacked_event_template(t, event_types, timing_offset_sec=0.0, transition_duration_sec=float(tau))
        fit = _closed_form_random_intercept_fit(y, template, event_codes, weights, float(event_offset_penalty))
        amp = float(fit["amplitude"])
        se = float(fit["uncertainty"])
        null_bic = _bic(float(fit["null_wrss"]), len(y), float(fit["null_effective_params"]))
        bic = _bic(float(fit["wrss"]), len(y), float(fit["effective_params"]))
        offsets = np.asarray(fit["offsets"], dtype=float)
        adjusted_values = y - offsets[event_codes]
        model_values = amp * template
        adjusted_q95 = float(np.nanquantile(np.abs(adjusted_values[np.isfinite(adjusted_values)]), 0.95)) if np.any(np.isfinite(adjusted_values)) else np.nan
        model_abs_max = float(np.nanmax(np.abs(model_values[np.isfinite(model_values)]))) if np.any(np.isfinite(model_values)) else np.nan
        model_to_data_ratio = model_abs_max / adjusted_q95 if np.isfinite(model_abs_max) and np.isfinite(adjusted_q95) and adjusted_q95 > 0 else np.nan
        template_info_per_point = float(fit.get("template_information_per_point", np.nan))
        if not np.isfinite(amp) or not np.isfinite(se):
            fit_quality_flag = "fit_failed"
        elif np.isfinite(model_to_data_ratio) and model_to_data_ratio > 8.0:
            fit_quality_flag = "template_extrapolates_beyond_adjusted_data"
        elif np.isfinite(template_info_per_point) and template_info_per_point < 1e-5:
            fit_quality_flag = "low_template_identifiability"
        else:
            fit_quality_flag = "ok"
        adjusted = g.copy()
        adjusted["event_offset"] = offsets[event_codes]
        adjusted["hierarchical_adjusted_z_power"] = adjusted_values
        adjusted["hierarchical_model_z_power"] = model_values
        adjusted["best_transition_duration_s"] = float(tau)
        adjusted["event_offset_penalty"] = float(event_offset_penalty)
        profile_medians = (
            adjusted.groupby(["event_type", "t_bin_sec"], sort=True)["hierarchical_adjusted_z_power"]
            .median()
            .to_numpy(dtype=float)
        )
        profile_abs_q95 = float(np.nanquantile(np.abs(profile_medians[np.isfinite(profile_medians)]), 0.95)) if np.any(np.isfinite(profile_medians)) else np.nan
        model_to_profile_ratio = model_abs_max / profile_abs_q95 if np.isfinite(model_abs_max) and np.isfinite(profile_abs_q95) and profile_abs_q95 > 0 else np.nan
        if fit_quality_flag == "ok" and np.isfinite(model_to_profile_ratio) and model_to_profile_ratio > 8.0:
            fit_quality_flag = "template_extrapolates_beyond_stacked_profile"
        adjusted["fit_quality_flag"] = fit_quality_flag
        row = {
            "n_points": int(len(y)),
            "n_events": int(n_events),
            "amplitude": amp,
            "uncertainty": se,
            "event_offset_fit_snr": float(amp / se) if np.isfinite(se) and se > 0 else np.nan,
            "best_transition_duration_s": float(tau),
            "event_offset_penalty": float(event_offset_penalty),
            "delta_bic": float(null_bic - bic),
            "weighted_rss_event_offsets_only": float(fit["null_wrss"]),
            "weighted_rss_event_offsets_plus_template": float(fit["wrss"]),
            "effective_params_event_offsets_only": float(fit["null_effective_params"]),
            "effective_params_event_offsets_plus_template": float(fit["effective_params"]),
            "event_offset_rms": float(np.sqrt(np.nanmean(offsets**2))) if offsets.size else np.nan,
            "residual_rms": float(np.sqrt(np.nanmean(np.asarray(fit["resid"], dtype=float) ** 2))),
            "template_information": float(fit.get("template_information", np.nan)),
            "template_information_per_point": template_info_per_point,
            "adjusted_abs_q95": adjusted_q95,
            "profile_abs_q95": profile_abs_q95,
            "model_abs_max": model_abs_max,
            "model_to_adjusted_q95_ratio": model_to_data_ratio,
            "model_to_profile_q95_ratio": model_to_profile_ratio,
            "fit_quality_flag": fit_quality_flag,
        }
        if best is None or float(row["delta_bic"]) > float(best["delta_bic"]):
            best = row
            best_adjusted = adjusted
    return best or {}, best_adjusted


def fit_all_groups(
    points: pd.DataFrame,
    transition_durations: list[float],
    event_offset_penalty: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit_rows = []
    adjusted_rows = []
    for keys, grp in points.groupby(GROUP_COLS, sort=True, dropna=False):
        meta = dict(zip(GROUP_COLS, keys))
        fit, adjusted = fit_event_offset_group(grp, transition_durations, event_offset_penalty)
        fit_rows.append({**meta, **fit})
        if not adjusted.empty:
            for col, val in meta.items():
                adjusted[col] = val
            adjusted_rows.append(adjusted)
    fits = pd.DataFrame(fit_rows)
    adjusted_points = pd.concat(adjusted_rows, ignore_index=True) if adjusted_rows else pd.DataFrame()
    return fits, adjusted_points


def summarize_adjusted_profiles(adjusted_points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    by = GROUP_COLS + ["event_type", "t_bin_sec"]
    for keys, grp in adjusted_points.groupby(by, sort=True, dropna=False):
        vals = pd.to_numeric(grp["hierarchical_adjusted_z_power"], errors="coerce").dropna().to_numpy(dtype=float)
        model_vals = pd.to_numeric(grp["hierarchical_model_z_power"], errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue
        rows.append(
            {
                **dict(zip(by, keys)),
                "median_adjusted_z_power": float(np.nanmedian(vals)),
                "mean_adjusted_z_power": float(np.nanmean(vals)),
                "sem_adjusted_z_power": _robust_sem(vals),
                "q25_adjusted_z_power": float(np.nanquantile(vals, 0.25)),
                "q75_adjusted_z_power": float(np.nanquantile(vals, 0.75)),
                "median_model_z_power": float(np.nanmedian(model_vals)) if model_vals.size else np.nan,
                "fit_quality_flag": str(grp["fit_quality_flag"].dropna().iloc[0])
                if "fit_quality_flag" in grp.columns and not grp["fit_quality_flag"].dropna().empty
                else "",
                "n_events": int(grp["event_uid"].nunique()),
                "n_points": int(vals.size),
            }
        )
    return pd.DataFrame(rows)


def summarize_fit_controls(fits: pd.DataFrame) -> pd.DataFrame:
    real = fits[fits["control_family"].eq("real")].copy()
    controls = fits[~fits["control_family"].eq("real")].copy()
    rows = []
    for _, row in real.iterrows():
        same = controls[
            controls["analysis_source"].eq(row["analysis_source"])
            & controls["frequency_band"].astype(int).eq(int(row["frequency_band"]))
        ].copy()
        vals = pd.to_numeric(same["amplitude"], errors="coerce").dropna().to_numpy(dtype=float)
        abs_vals = np.abs(vals)
        amp = float(row["amplitude"])
        rows.append(
            {
                "analysis_source": row["analysis_source"],
                "frequency_band": int(row["frequency_band"]),
                "frequency_mhz": float(row["frequency_mhz"]),
                "real_amplitude": amp,
                "real_uncertainty": float(row["uncertainty"]),
                "real_event_offset_fit_snr": float(row["event_offset_fit_snr"]),
                "real_delta_bic": float(row["delta_bic"]),
                "real_best_transition_duration_s": float(row["best_transition_duration_s"]),
                "real_n_events": int(row["n_events"]),
                "real_n_points": int(row["n_points"]),
                "real_fit_quality_flag": row.get("fit_quality_flag", ""),
                "real_model_to_adjusted_q95_ratio": float(row.get("model_to_adjusted_q95_ratio", np.nan)),
                "real_model_to_profile_q95_ratio": float(row.get("model_to_profile_q95_ratio", np.nan)),
                "real_template_information_per_point": float(row.get("template_information_per_point", np.nan)),
                "n_controls": int(vals.size),
                "control_median_amplitude": float(np.nanmedian(vals)) if vals.size else np.nan,
                "control_abs_q75_amplitude": float(np.nanquantile(abs_vals, 0.75)) if vals.size else np.nan,
                "empirical_p_abs_amp_ge_real": float((1 + np.count_nonzero(abs_vals >= abs(amp))) / (1 + vals.size))
                if vals.size
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def injection_recovery(
    points: pd.DataFrame,
    transition_durations: list[float],
    event_offset_penalty: float,
    injection_taus: list[float],
    injection_amplitude: float,
) -> pd.DataFrame:
    rows = []
    real = points[points["control_family"].eq("real")].copy()
    for keys, grp in real.groupby(["analysis_source", "frequency_band", "frequency_mhz"], sort=True, dropna=False):
        source, band, freq = keys
        for tau in injection_taus:
            baseline_fit, _ = fit_event_offset_group(grp, transition_durations, event_offset_penalty, fixed_tau_s=float(tau))
            injected = grp.copy()
            template = stacked_event_template(
                injected["t_bin_sec"].to_numpy(dtype=float),
                injected["event_type"].astype(str).to_numpy(),
                timing_offset_sec=0.0,
                transition_duration_sec=float(tau),
            )
            injected["z_power"] = pd.to_numeric(injected["z_power"], errors="coerce").to_numpy(dtype=float) + float(injection_amplitude) * template
            injected_fit, _ = fit_event_offset_group(injected, transition_durations, event_offset_penalty, fixed_tau_s=float(tau))
            base_amp = float(baseline_fit.get("amplitude", np.nan))
            inj_amp = float(injected_fit.get("amplitude", np.nan))
            recovery = (inj_amp - base_amp) / float(injection_amplitude) if np.isfinite(base_amp) and np.isfinite(inj_amp) else np.nan
            if not np.isfinite(recovery) or abs(recovery) > 3:
                flag = "unstable_or_nonphysical"
            elif recovery <= 0:
                flag = "negative_transfer"
            elif recovery < 0.1:
                flag = "low_recovery"
            elif recovery < 0.5:
                flag = "attenuated"
            elif recovery <= 1.5:
                flag = "usable"
            else:
                flag = "amplified"
            rows.append(
                {
                    "analysis_source": source,
                    "frequency_band": int(band),
                    "frequency_mhz": float(freq),
                    "injection_tau_s": float(tau),
                    "injection_amplitude": float(injection_amplitude),
                    "baseline_fixed_tau_amplitude": base_amp,
                    "injected_fixed_tau_amplitude": inj_amp,
                    "recovery_fraction": recovery,
                    "transfer_stability_flag": flag,
                }
            )
    return pd.DataFrame(rows)


def _control_envelope(profile_summary: pd.DataFrame, source: str, freq: float, event_type: str) -> pd.DataFrame:
    controls = profile_summary[
        profile_summary["analysis_source"].eq(source)
        & profile_summary["control_family"].ne("real")
        & np.isclose(profile_summary["frequency_mhz"].astype(float), float(freq))
        & profile_summary["event_type"].eq(event_type)
    ].copy()
    rows = []
    for t, grp in controls.groupby("t_bin_sec", sort=True):
        vals = pd.to_numeric(grp["median_adjusted_z_power"], errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue
        rows.append(
            {
                "t_bin_sec": float(t),
                "control_median": float(np.nanmedian(vals)),
                "control_q25": float(np.nanquantile(vals, 0.25)),
                "control_q75": float(np.nanquantile(vals, 0.75)),
            }
        )
    return pd.DataFrame(rows)


def plot_adjusted_profile_grids(profile_summary: pd.DataFrame, out_dir: Path) -> list[Path]:
    paths = []
    for source in sorted(profile_summary["analysis_source"].dropna().unique()):
        freqs = sorted(profile_summary.loc[profile_summary["analysis_source"].eq(source), "frequency_mhz"].dropna().unique())
        fig, axes = plt.subplots(len(freqs), 2, figsize=(13.5, max(10, 1.45 * len(freqs))), sharex=True)
        if len(freqs) == 1:
            axes = np.asarray([axes])
        for i, freq in enumerate(freqs):
            for j, event_type in enumerate(["disappearance", "reappearance"]):
                ax = axes[i, j]
                real = profile_summary[
                    profile_summary["analysis_source"].eq(source)
                    & profile_summary["control_family"].eq("real")
                    & np.isclose(profile_summary["frequency_mhz"].astype(float), float(freq))
                    & profile_summary["event_type"].eq(event_type)
                ].sort_values("t_bin_sec")
                env = _control_envelope(profile_summary, source, float(freq), event_type)
                if not env.empty:
                    x = env["t_bin_sec"].to_numpy(dtype=float) / 60.0
                    ax.fill_between(x, env["control_q25"], env["control_q75"], color="0.65", alpha=0.22, linewidth=0, label="control IQR")
                    ax.plot(x, env["control_median"], color="0.45", lw=1.0, label="control median")
                if not real.empty:
                    x = real["t_bin_sec"].to_numpy(dtype=float) / 60.0
                    fit_quality_flag = (
                        str(real["fit_quality_flag"].dropna().iloc[0])
                        if "fit_quality_flag" in real.columns and not real["fit_quality_flag"].dropna().empty
                        else "ok"
                    )
                    ax.errorbar(
                        x,
                        real["median_adjusted_z_power"],
                        yerr=real["sem_adjusted_z_power"],
                        marker="o",
                        ms=2.4,
                        lw=1.2,
                        elinewidth=0.5,
                        capsize=1.0,
                        color="#1f78b4",
                        ecolor="#1f78b4",
                        label="real, event-offset adjusted",
                    )
                    if fit_quality_flag == "ok":
                        ax.plot(x, real["median_model_z_power"], color="#d95f02", lw=1.7, label="shared template fit")
                    else:
                        ax.text(
                            0.02,
                            0.92,
                            fit_quality_flag.replace("_", " "),
                            transform=ax.transAxes,
                            ha="left",
                            va="top",
                            fontsize=6.8,
                            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.0},
                        )
                ax.axvline(0, color="black", ls=":", lw=0.8)
                ax.axhline(0, color="0.75", lw=0.7)
                ax.grid(True, color="0.92", lw=0.5)
                ax.set_title(f"{float(freq):.2f} MHz {event_type}", fontsize=8.5)
                if j == 0 and i == len(freqs) // 2:
                    ax.set_ylabel("event-offset adjusted lower-V power")
                if i == len(freqs) - 1:
                    ax.set_xlabel("minutes from predicted event")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        fig.legend(by_label.values(), by_label.keys(), loc="lower center", bbox_to_anchor=(0.5, 0.006), ncol=min(4, len(by_label)), frameon=False)
        fig.suptitle(f"{SOURCE_LABEL.get(source, source)} weak hierarchical event-offset model", y=0.992)
        fig.tight_layout(rect=[0, 0.04, 1, 0.965])
        path = out_dir / f"{source}_hierarchical_event_offset_profile_grid.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_amplitude_spectrum(summary: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), sharey=False)
    for ax, source in zip(axes, ["earth", "sun", "fornax_a"]):
        sub = summary[summary["analysis_source"].eq(source)].sort_values("frequency_mhz")
        if sub.empty:
            ax.set_title(SOURCE_LABEL.get(source, source))
            continue
        x = sub["frequency_mhz"].to_numpy(dtype=float)
        ok = sub["real_fit_quality_flag"].fillna("").eq("ok").to_numpy()
        amp = sub["real_amplitude"].to_numpy(dtype=float)
        unc = sub["real_uncertainty"].to_numpy(dtype=float)
        ctrl = sub["control_abs_q75_amplitude"].to_numpy(dtype=float)

        scale_terms: list[float] = []
        if np.any(ok):
            scale_terms.extend(np.abs(amp[ok]).tolist())
            scale_terms.extend(np.abs(amp[ok] + unc[ok]).tolist())
            scale_terms.extend(np.abs(amp[ok] - unc[ok]).tolist())
            scale_terms.extend(np.abs(ctrl[ok]).tolist())
        else:
            scale_terms.extend(np.abs(ctrl[np.isfinite(ctrl)]).tolist())
        scale_terms = [v for v in scale_terms if np.isfinite(v)]
        y_bound = max(1.0, float(np.nanpercentile(scale_terms, 95)) * 1.35) if scale_terms else 1.0
        y_bound = min(y_bound, max(scale_terms) * 1.35 if scale_terms else y_bound)
        ctrl_for_plot = np.clip(ctrl, 0, y_bound)

        ax.axhline(0, color="0.75", lw=0.8)
        ax.fill_between(
            x,
            -ctrl_for_plot,
            ctrl_for_plot,
            color="0.65",
            alpha=0.22,
            label="control |A| q75",
        )
        if np.any(ok):
            ax.errorbar(
                x[ok],
                amp[ok],
                yerr=unc[ok],
                marker="o",
                lw=1.4,
                capsize=2,
                color="#1f78b4",
                label="real A, fit-quality ok",
            )
        flagged = ~ok
        if np.any(flagged):
            signs = np.sign(amp[flagged])
            signs[signs == 0] = 1
            ax.scatter(
                x[flagged],
                signs * y_bound * 0.92,
                marker="x",
                s=42,
                color="#d73027",
                label="flagged fit, off-scale",
                zorder=4,
            )
        ax.set_ylim(-y_bound, y_bound)
        ax.set_xscale("log")
        ax.set_xticks(list(FREQUENCY_MAP_MHZ.values()))
        ax.set_xticklabels([f"{v:g}" for v in FREQUENCY_MAP_MHZ.values()], rotation=45, ha="right")
        ax.set_title(SOURCE_LABEL.get(source, source))
        ax.set_xlabel("frequency (MHz)")
        ax.grid(True, color="0.9", lw=0.5)
    axes[0].set_ylabel("shared positive-source template amplitude")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Weak hierarchical event-offset fitted amplitudes")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    path = out_dir / "hierarchical_event_offset_amplitude_spectrum.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_transfer(recovery: pd.DataFrame, out_dir: Path, main_tau_s: float) -> Path:
    sub = recovery[np.isclose(recovery["injection_tau_s"].astype(float), float(main_tau_s))].copy()
    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    colors = {"earth": "#333333", "sun": "#d95f02", "fornax_a": "#1f78b4"}
    ax.axhline(1, color="0.55", ls="--", lw=0.8)
    ax.axhline(0, color="0.75", lw=0.8)
    for source, grp in sub.groupby("analysis_source", sort=True):
        grp = grp.sort_values("frequency_mhz")
        ax.plot(
            grp["frequency_mhz"],
            grp["recovery_fraction"],
            marker="o",
            lw=1.4,
            color=colors.get(source, None),
            label=SOURCE_LABEL.get(source, source),
        )
    ax.set_xscale("log")
    ax.set_xticks(list(FREQUENCY_MAP_MHZ.values()))
    ax.set_xticklabels([f"{v:g}" for v in FREQUENCY_MAP_MHZ.values()], rotation=45, ha="right")
    ax.set_xlabel("frequency (MHz)")
    ax.set_ylabel("recovered / injected amplitude")
    ax.set_title(f"Injection recovery for event-offset model, tau={float(main_tau_s):.0f} s")
    ax.set_ylim(-0.1, 1.15)
    ax.grid(True, color="0.9", lw=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = out_dir / f"hierarchical_event_offset_transfer_tau{int(main_tau_s)}s.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def load_events_for_sources(
    sources: list[str],
    start_date: str,
    window_s: float,
    shifts_s: list[float],
    n_random: int,
    random_seed: int,
    max_offsource_controls: int,
) -> pd.DataFrame:
    tables = []
    if "earth" in sources:
        earth = _read(EARTH_EVENT_ROWS, parse_dates=["predicted_event_time"])
        earth = earth[earth["antenna"].astype(str).eq(ANTENNA)].copy()
        earth = earth[earth["predicted_event_time"] >= pd.Timestamp(start_date)].copy()
        tables.append(earth)
    needed = [s for s in sources if s in {"sun", "fornax_a"}]
    if needed:
        sf = _load_event_table(
            start_date,
            shifts_s=shifts_s,
            n_random=n_random,
            random_seed=random_seed,
            window_s=window_s,
            max_offsource_controls=max_offsource_controls,
        )
        sf = sf[sf["analysis_source"].astype(str).isin(needed)].copy()
        tables.append(sf)
    events = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    events = events[events["analysis_source"].astype(str).isin(sources)].copy()
    events = events[events["antenna"].astype(str).eq(ANTENNA)].copy()
    events["event_uid"] = np.arange(len(events), dtype=int)
    return events.reset_index(drop=True)


def write_report(
    out_dir: Path,
    summary: pd.DataFrame,
    recovery: pd.DataFrame,
    status_counts: pd.DataFrame,
    plot_paths: list[Path],
    event_offset_penalty: float,
    save_large_tables: bool,
) -> None:
    rec_counts = (
        recovery.groupby(["analysis_source", "injection_tau_s", "transfer_stability_flag"])
        .size()
        .rename("n_channels")
        .reset_index()
        .sort_values(["analysis_source", "injection_tau_s", "transfer_stability_flag"])
    )
    cols = [
        "analysis_source",
        "frequency_mhz",
        "real_amplitude",
        "real_uncertainty",
        "real_event_offset_fit_snr",
        "real_delta_bic",
        "real_best_transition_duration_s",
        "real_n_events",
        "real_fit_quality_flag",
        "real_model_to_profile_q95_ratio",
        "control_abs_q75_amplitude",
        "empirical_p_abs_amp_ge_real",
    ]
    lines = [
        "# Weak Hierarchical Event-Offset Model",
        "",
        "This is option 6: a simple hierarchical-style model with per-event offsets only.",
        "",
        "Model:",
        "",
        "    z_ij = b_event[j] + A h_ij(tau) + eps_ij",
        "",
        "There are no per-event slopes, no trend lines, and no flexible background basis. The nuisance term is a constant event offset with ridge shrinkage.",
        "",
        f"Event-offset ridge penalty: {float(event_offset_penalty):g}",
        "",
        "## Usable Window Counts",
        "",
        status_counts.to_string(index=False),
        "",
        "## Real Fits Against Control Amplitudes",
        "",
        summary[cols].sort_values(["analysis_source", "frequency_mhz"]).to_string(index=False),
        "",
        "## Injection-Recovery Stability",
        "",
        rec_counts.to_string(index=False),
        "",
        "## Plots",
        "",
    ]
    lines.extend(f"- `{p.name}`" for p in plot_paths)
    lines += [
        "",
        "## Interpretation Guide",
        "",
        "- A positive amplitude means disappearance decreases and reappearance increases after removing only per-event offsets.",
        "- The gray control bands in the profile grids show whether wrong-time/off-source/randomized events can reproduce the adjusted morphology.",
        "- Panels marked `template extrapolates beyond adjusted data` are not detection-grade fits; the template coefficient is poorly constrained relative to the actual adjusted data scale.",
        "- If injection recovery is low, then even this simple per-event-offset model can absorb the source template for that frequency.",
        "- This model is intentionally less aggressive than trend subtraction; if it fails to show a source while Earth remains visible, the limiting issue is likely source/background contrast rather than overflexible baseline fitting.",
        f"- Large event-bin tables saved: {bool(save_large_tables)}.",
    ]
    (out_dir / "hierarchical_event_offset_model_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--sources", default="earth,sun,fornax_a")
    parser.add_argument("--start-date", default="1974-11-01")
    parser.add_argument("--window-s", type=float, default=900.0)
    parser.add_argument("--bin-s", type=float, default=60.0)
    parser.add_argument("--sideband-s", type=float, default=600.0)
    parser.add_argument("--time-shifts-s", default="-1200,-600,-300,300,600,1200")
    parser.add_argument("--n-random", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=20260608)
    parser.add_argument("--max-offsource-controls", type=int, default=16)
    parser.add_argument("--transition-durations-s", default="0,300,900")
    parser.add_argument("--event-offset-penalty", type=float, default=1.0)
    parser.add_argument("--injection-taus-s", default="0,300,900")
    parser.add_argument("--injection-amplitude", type=float, default=0.2)
    parser.add_argument("--main-transfer-tau-s", type=float, default=300.0)
    parser.add_argument("--save-large-tables", action="store_true")
    args = parser.parse_args()

    out_dir = ensure_dir(Path(args.out_dir))
    sources = [x.strip() for x in str(args.sources).split(",") if x.strip()]
    shifts = [float(x.strip()) for x in str(args.time_shifts_s).split(",") if x.strip()]
    transition_durations = [float(x.strip()) for x in str(args.transition_durations_s).split(",") if x.strip()]
    injection_taus = [float(x.strip()) for x in str(args.injection_taus_s).split(",") if x.strip()]
    write_json(
        out_dir / "run_config.json",
        {
            "sources": sources,
            "antenna": ANTENNA,
            "start_date": str(args.start_date),
            "window_s": float(args.window_s),
            "bin_s": float(args.bin_s),
            "sideband_s": float(args.sideband_s),
            "time_shifts_s": shifts,
            "n_random": int(args.n_random),
            "random_seed": int(args.random_seed),
            "max_offsource_controls": int(args.max_offsource_controls),
            "transition_durations_s": transition_durations,
            "event_offset_penalty": float(args.event_offset_penalty),
            "injection_taus_s": injection_taus,
            "injection_amplitude": float(args.injection_amplitude),
            "main_transfer_tau_s": float(args.main_transfer_tau_s),
            "save_large_tables": bool(args.save_large_tables),
            "software_versions": software_versions(),
        },
    )

    print("Loading event tables for hierarchical event-offset model...", flush=True)
    events = load_events_for_sources(
        sources=sources,
        start_date=str(args.start_date),
        window_s=float(args.window_s),
        shifts_s=shifts,
        n_random=int(args.n_random),
        random_seed=int(args.random_seed),
        max_offsource_controls=int(args.max_offsource_controls),
    )
    event_counts = (
        events.groupby(["analysis_source", "control_family"])["event_uid"]
        .agg(["count", "nunique"])
        .reset_index()
        .rename(columns={"count": "n_event_frequency_rows", "nunique": "n_unique_event_rows"})
    )
    event_counts.to_csv(out_dir / "hierarchical_event_offset_input_event_counts.csv", index=False)
    if bool(args.save_large_tables):
        events.to_csv(out_dir / "hierarchical_event_offset_input_events.csv", index=False)
    bands = sorted(events["frequency_band"].dropna().astype(int).unique())
    print(f"Loading lower-V data for bands {bands}...", flush=True)
    clean_groups = _load_clean_groups(bands)
    print(f"Collecting event-level normalized profiles from {len(events)} event/control rows...", flush=True)
    points, status = collect_profiles(events, clean_groups, float(args.window_s), float(args.bin_s), float(args.sideband_s))
    status_counts = (
        status.groupby(["analysis_source", "control_family", "used_in_stack"])["event_uid"]
        .agg(["count", "nunique"])
        .reset_index()
        .rename(columns={"count": "n_event_frequency_rows", "nunique": "n_unique_event_rows"})
    )
    status_counts.to_csv(out_dir / "hierarchical_event_offset_profile_status_counts.csv", index=False)
    if bool(args.save_large_tables):
        points.to_csv(out_dir / "hierarchical_event_offset_profile_points.csv", index=False)
        status.to_csv(out_dir / "hierarchical_event_offset_profile_status.csv", index=False)
    print("Fitting weak event-offset model to real and control groups...", flush=True)
    fits, adjusted_points = fit_all_groups(points, transition_durations, float(args.event_offset_penalty))
    fits.to_csv(out_dir / "hierarchical_event_offset_fit_by_group.csv", index=False)
    profile_summary = summarize_adjusted_profiles(adjusted_points)
    profile_summary.to_csv(out_dir / "hierarchical_event_offset_adjusted_profile_summary.csv", index=False)
    if bool(args.save_large_tables):
        adjusted_points.to_csv(out_dir / "hierarchical_event_offset_adjusted_points.csv", index=False)
    summary = summarize_fit_controls(fits)
    summary.to_csv(out_dir / "hierarchical_event_offset_source_summary.csv", index=False)
    print("Running injection recovery through event-offset model...", flush=True)
    recovery = injection_recovery(
        points,
        transition_durations,
        float(args.event_offset_penalty),
        injection_taus,
        float(args.injection_amplitude),
    )
    recovery.to_csv(out_dir / "hierarchical_event_offset_injection_recovery.csv", index=False)
    print("Writing plots...", flush=True)
    paths: list[Path] = []
    paths.extend(plot_adjusted_profile_grids(profile_summary, out_dir))
    paths.append(plot_amplitude_spectrum(summary, out_dir))
    paths.append(plot_transfer(recovery, out_dir, float(args.main_transfer_tau_s)))
    write_report(
        out_dir,
        summary,
        recovery,
        status_counts,
        paths,
        float(args.event_offset_penalty),
        bool(args.save_large_tables),
    )
    print(out_dir / "hierarchical_event_offset_model_report.md")


if __name__ == "__main__":
    main()
