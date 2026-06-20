#!/usr/bin/env python
"""Compare baseline/signal models on selected source occultation windows."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rylevonberg.table_io import read_table
from scipy.interpolate import UnivariateSpline
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.detection import baseline_matrix, event_template
from rylevonberg.util import datetime_ns, ensure_dir, robust_sigma


CLEAN = ROOT / "outputs/control_survey_earth_sun_postnov1974_v2/01_ingest/cleaned_timeseries.csv"
ANT_LABEL = {"rv1_coarse": "upper V", "rv2_coarse": "lower V"}
METHOD_LABELS = {
    "joint_constant": "joint constant",
    "robust_joint_constant": "robust joint constant",
    "joint_linear": "joint linear",
    "robust_joint_huber": "robust joint Huber",
    "sideband_constant": "sideband constant",
    "sideband_linear": "sideband linear",
    "sideband_quadratic": "sideband quadratic",
    "sideband_spline": "sideband spline",
    "prepost_median": "pre/post median",
    "detrended_prepost_median": "detrended pre/post median",
    "detrended_inner_median": "detrended inner median",
    "inner_median": "inner median",
    "ar1_whitened_sideband": "AR(1) whitened sideband",
}


@dataclass(frozen=True)
class FitResult:
    method: str
    amp: float
    sigma: float
    n_samples: int
    n_baseline: int
    timing_offset_s: float = 0.0
    warning: str = ""
    delta_bic: float = np.nan
    residual_ar1: float = np.nan
    runs_z: float = np.nan

    @property
    def amp_over_sigma(self) -> float:
        if np.isfinite(self.amp) and np.isfinite(self.sigma) and self.sigma > 0:
            return float(self.amp / self.sigma)
        return np.nan


def _read(path: Path, **kwargs) -> pd.DataFrame:
    return read_table(path, low_memory=False, **kwargs)


def _antenna_label(antenna: str) -> str:
    return ANT_LABEL.get(str(antenna), str(antenna))


def _source_title(source: str) -> str:
    return source.capitalize()


def _fit_linear(X: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    if weights is None:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    else:
        w = np.sqrt(np.asarray(weights, dtype=float))
        beta, *_ = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)
    return beta, y - X @ beta


def _fit_joint(
    tr: np.ndarray,
    y: np.ndarray,
    tmpl: np.ndarray,
    method: str,
    robust: bool = False,
    baseline_order: int = 1,
) -> FitResult:
    B = baseline_matrix(tr, baseline_order)
    X = np.column_stack([B, tmpl])
    weights = np.ones(len(y), dtype=float)
    beta, resid = _fit_linear(X, y, weights if robust else None)
    if robust:
        for _ in range(8):
            sig = robust_sigma(resid)
            if not np.isfinite(sig) or sig <= 0:
                break
            cutoff = 1.5 * sig
            weights = np.minimum(1.0, cutoff / np.maximum(np.abs(resid), cutoff))
            beta, resid = _fit_linear(X, y, weights)
    amp = float(beta[-1])
    sigma = robust_sigma(resid)
    _, resid0 = _fit_linear(B, y, weights if robust else None)
    return FitResult(
        method=method,
        amp=amp,
        sigma=float(sigma),
        n_samples=len(y),
        n_baseline=len(y),
        delta_bic=_delta_bic(resid0, resid, n_params0=B.shape[1], n_params1=X.shape[1]),
        residual_ar1=_estimate_ar1(resid),
        runs_z=_runs_z(resid),
    )


def _fit_sideband_poly(
    tr: np.ndarray,
    y: np.ndarray,
    tmpl: np.ndarray,
    method: str,
    order: int,
    exclusion_s: float,
) -> FitResult:
    fit_mask = np.abs(tr) >= float(exclusion_s)
    min_fit = max(6, order + 3)
    warning = ""
    if np.count_nonzero(fit_mask) < min_fit:
        fit_mask = np.ones(len(y), dtype=bool)
        warning = "fallback_full_window"
    B_fit = baseline_matrix(tr[fit_mask], order)
    beta, _ = _fit_linear(B_fit, y[fit_mask])
    baseline = baseline_matrix(tr, order) @ beta
    resid = y - baseline
    den = float(np.dot(tmpl, tmpl))
    amp = float(np.dot(resid, tmpl) / den) if den > 0 else np.nan
    sigma = robust_sigma(resid[fit_mask]) if np.count_nonzero(fit_mask) else robust_sigma(resid)
    model_resid = resid - amp * tmpl if np.isfinite(amp) else resid
    return FitResult(
        method=method,
        amp=amp,
        sigma=float(sigma),
        n_samples=len(y),
        n_baseline=int(np.count_nonzero(fit_mask)),
        warning=warning,
        delta_bic=_delta_bic(resid, model_resid, n_params0=order + 1, n_params1=order + 2),
        residual_ar1=_estimate_ar1(model_resid),
        runs_z=_runs_z(model_resid),
    )


def _fit_sideband_spline(tr: np.ndarray, y: np.ndarray, tmpl: np.ndarray, exclusion_s: float) -> FitResult:
    fit_mask = np.abs(tr) >= float(exclusion_s)
    warning = ""
    if np.count_nonzero(fit_mask) < 12 or np.unique(tr[fit_mask]).size < 8:
        return _fit_sideband_poly(tr, y, tmpl, "sideband_spline", 1, exclusion_s)
    x = tr[fit_mask]
    yy = y[fit_mask]
    order = np.argsort(x)
    x = x[order]
    yy = yy[order]
    try:
        # Smooth enough to capture broad drift but not individual sample jumps.
        scale = robust_sigma(yy - np.nanmedian(yy))
        smooth = len(yy) * (scale if np.isfinite(scale) and scale > 0 else np.nanstd(yy)) ** 2
        spline = UnivariateSpline(x, yy, k=min(3, len(x) - 1), s=smooth)
        baseline = spline(tr)
    except Exception:
        warning = "spline_failed_fallback_linear"
        lin = _fit_sideband_poly(tr, y, tmpl, "sideband_spline", 1, exclusion_s)
        return FitResult(**{**lin.__dict__, "warning": warning})
    resid = y - baseline
    den = float(np.dot(tmpl, tmpl))
    amp = float(np.dot(resid, tmpl) / den) if den > 0 else np.nan
    sigma = robust_sigma(resid[fit_mask])
    model_resid = resid - amp * tmpl if np.isfinite(amp) else resid
    return FitResult(
        method="sideband_spline",
        amp=amp,
        sigma=float(sigma),
        n_samples=len(y),
        n_baseline=int(np.count_nonzero(fit_mask)),
        warning=warning,
        delta_bic=_delta_bic(resid, model_resid, n_params0=4, n_params1=5),
        residual_ar1=_estimate_ar1(model_resid),
        runs_z=_runs_z(model_resid),
    )


def _side_median_amp(tr: np.ndarray, y: np.ndarray, event_type: str, method: str, pre: tuple[float, float], post: tuple[float, float]) -> FitResult:
    pre_mask = (tr >= pre[0]) & (tr <= pre[1])
    post_mask = (tr >= post[0]) & (tr <= post[1])
    if np.count_nonzero(pre_mask) < 3 or np.count_nonzero(post_mask) < 3:
        return FitResult(method=method, amp=np.nan, sigma=np.nan, n_samples=len(y), n_baseline=int(np.count_nonzero(pre_mask) + np.count_nonzero(post_mask)), warning="too_few_side_samples")
    pre_med = float(np.nanmedian(y[pre_mask]))
    post_med = float(np.nanmedian(y[post_mask]))
    if str(event_type) == "reappearance":
        amp = post_med - pre_med
    else:
        amp = pre_med - post_med
    resid = np.concatenate([y[pre_mask] - pre_med, y[post_mask] - post_med])
    sigma = robust_sigma(resid)
    pooled = np.concatenate([y[pre_mask], y[post_mask]])
    null_resid = pooled - np.nanmedian(pooled)
    return FitResult(
        method=method,
        amp=float(amp),
        sigma=float(sigma),
        n_samples=len(y),
        n_baseline=int(len(resid)),
        delta_bic=_delta_bic(null_resid, resid, n_params0=1, n_params1=2),
        residual_ar1=_estimate_ar1(resid),
        runs_z=_runs_z(resid),
    )


def _detrended_side_median_amp(
    tr: np.ndarray,
    y: np.ndarray,
    event_type: str,
    method: str,
    pre: tuple[float, float],
    post: tuple[float, float],
    exclusion_s: float,
) -> FitResult:
    fit_mask = np.abs(tr) >= float(exclusion_s)
    warning = ""
    if np.count_nonzero(fit_mask) < 6:
        fit_mask = np.ones(len(y), dtype=bool)
        warning = "fallback_full_window"
    B_fit = baseline_matrix(tr[fit_mask], 1)
    beta, _ = _fit_linear(B_fit, y[fit_mask])
    baseline = baseline_matrix(tr, 1) @ beta
    level = float(np.nanmedian(y[fit_mask])) if np.count_nonzero(fit_mask) else float(np.nanmedian(y))
    detrended = y - baseline + level
    fit = _side_median_amp(tr, detrended, event_type, method, pre, post)
    return FitResult(**{**fit.__dict__, "warning": warning or fit.warning})


def _estimate_ar1(resid: np.ndarray) -> float:
    r = np.asarray(resid, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 4:
        return 0.0
    r = r - np.nanmedian(r)
    den = float(np.dot(r[:-1], r[:-1]))
    if den <= 0:
        return 0.0
    phi = float(np.dot(r[1:], r[:-1]) / den)
    return float(np.clip(phi, -0.85, 0.85))


def _fit_ar1_whitened(tr: np.ndarray, y: np.ndarray, tmpl: np.ndarray, exclusion_s: float) -> FitResult:
    base = _fit_sideband_poly(tr, y, tmpl, "ar1_whitened_sideband", 1, exclusion_s)
    fit_mask = np.abs(tr) >= float(exclusion_s)
    if np.count_nonzero(fit_mask) < 4:
        fit_mask = np.ones(len(y), dtype=bool)
    B_fit = baseline_matrix(tr[fit_mask], 1)
    beta, _ = _fit_linear(B_fit, y[fit_mask])
    baseline = baseline_matrix(tr, 1) @ beta
    resid = y - baseline
    phi = _estimate_ar1(resid[fit_mask])
    yw = resid[1:] - phi * resid[:-1]
    tw = tmpl[1:] - phi * tmpl[:-1]
    den = float(np.dot(tw, tw))
    amp = float(np.dot(yw, tw) / den) if den > 0 else np.nan
    sigma = robust_sigma(yw)
    model_resid = yw - amp * tw if np.isfinite(amp) else yw
    return FitResult(
        method="ar1_whitened_sideband",
        amp=amp,
        sigma=float(sigma),
        n_samples=len(y),
        n_baseline=base.n_baseline,
        warning=f"phi={phi:.3f}",
        delta_bic=_delta_bic(yw, model_resid, n_params0=3, n_params1=4),
        residual_ar1=_estimate_ar1(model_resid),
        runs_z=_runs_z(model_resid),
    )


def _delta_bic(resid0: np.ndarray, resid1: np.ndarray, n_params0: int, n_params1: int) -> float:
    """Return BIC(null) - BIC(step). Positive favors adding the step."""
    r0 = np.asarray(resid0, dtype=float)
    r1 = np.asarray(resid1, dtype=float)
    mask = np.isfinite(r0) & np.isfinite(r1)
    r0 = r0[mask]
    r1 = r1[mask]
    n = int(r0.size)
    if n < max(n_params1 + 2, 4):
        return np.nan
    rss0 = max(float(np.sum(r0**2)), np.finfo(float).tiny)
    rss1 = max(float(np.sum(r1**2)), np.finfo(float).tiny)
    bic0 = n * np.log(rss0 / n) + int(n_params0) * np.log(n)
    bic1 = n * np.log(rss1 / n) + int(n_params1) * np.log(n)
    return float(bic0 - bic1)


def _runs_z(resid: np.ndarray) -> float:
    r = np.asarray(resid, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 8:
        return np.nan
    r = r - np.nanmedian(r)
    signs = np.sign(r)
    signs = signs[signs != 0]
    if signs.size < 8:
        return np.nan
    n_pos = int(np.count_nonzero(signs > 0))
    n_neg = int(np.count_nonzero(signs < 0))
    if n_pos == 0 or n_neg == 0:
        return np.nan
    runs = 1 + int(np.count_nonzero(signs[1:] != signs[:-1]))
    n = n_pos + n_neg
    mean = 1 + 2 * n_pos * n_neg / n
    var = (2 * n_pos * n_neg * (2 * n_pos * n_neg - n)) / (n**2 * (n - 1))
    return float((runs - mean) / np.sqrt(var)) if var > 0 else np.nan


def _fit_methods(tr: np.ndarray, y: np.ndarray, event_type: str, timing_offset_s: float, exclusion_s: float) -> list[FitResult]:
    tmpl = event_template(tr, event_type, timing_offset_sec=timing_offset_s)
    return [
        _fit_joint(tr, y, tmpl, "joint_constant", robust=False, baseline_order=0),
        _fit_joint(tr, y, tmpl, "robust_joint_constant", robust=True, baseline_order=0),
        _fit_joint(tr, y, tmpl, "joint_linear", robust=False),
        _fit_joint(tr, y, tmpl, "robust_joint_huber", robust=True),
        _fit_sideband_poly(tr, y, tmpl, "sideband_constant", 0, exclusion_s),
        _fit_sideband_poly(tr, y, tmpl, "sideband_linear", 1, exclusion_s),
        _fit_sideband_poly(tr, y, tmpl, "sideband_quadratic", 2, exclusion_s),
        _fit_sideband_spline(tr, y, tmpl, exclusion_s),
        _side_median_amp(tr, y, event_type, "prepost_median", (-float("inf"), -exclusion_s), (exclusion_s, float("inf"))),
        _detrended_side_median_amp(
            tr,
            y,
            event_type,
            "detrended_prepost_median",
            (-float("inf"), -exclusion_s),
            (exclusion_s, float("inf")),
            exclusion_s,
        ),
        _side_median_amp(tr, y, event_type, "inner_median", (-3 * exclusion_s, -exclusion_s), (exclusion_s, 3 * exclusion_s)),
        _detrended_side_median_amp(
            tr,
            y,
            event_type,
            "detrended_inner_median",
            (-3 * exclusion_s, -exclusion_s),
            (exclusion_s, 3 * exclusion_s),
            exclusion_s,
        ),
        _fit_ar1_whitened(tr, y, tmpl, exclusion_s),
    ]


def _event_window(group: pd.DataFrame, group_ns: np.ndarray, ev: pd.Series, window_s: float) -> tuple[np.ndarray, np.ndarray] | None:
    event_ns = pd.Timestamp(ev["predicted_event_time"]).value
    half_ns = int(window_s * 1e9)
    lo = int(np.searchsorted(group_ns, event_ns - half_ns, side="left"))
    hi = int(np.searchsorted(group_ns, event_ns + half_ns, side="right"))
    if hi <= lo:
        return None
    local = group.iloc[lo:hi]
    tr = (datetime_ns(local["time"]) - event_ns).astype(float) / 1e9
    y = pd.to_numeric(local["power"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(y) & (np.abs(tr) <= window_s)
    if "is_valid" in local.columns:
        valid &= local["is_valid"].to_numpy(dtype=bool)
    if np.count_nonzero(valid) < 8:
        return None
    tr = tr[valid]
    y = y[valid]
    order = np.argsort(tr)
    return tr[order], y[order]


def _event_level_table(clean: pd.DataFrame, events: pd.DataFrame, source_row: pd.Series, exclusion_s: float) -> pd.DataFrame:
    source = str(source_row["source_name"])
    target_id = str(source_row.get("target_id", source))
    band = int(source_row["target_band"])
    antenna = str(source_row["target_antenna"])
    window_s = float(source_row["target_window_s"])
    timing_offset_s = float(source_row["best_timing_offset_s"])
    group = clean[
        clean["frequency_band"].astype(int).eq(band)
        & clean["antenna"].astype(str).eq(antenna)
    ].sort_values("time").reset_index(drop=True)
    group_ns = datetime_ns(group["time"])
    sub_events = events[
        events["source_name"].astype(str).eq(source)
        & events["frequency_band"].astype(int).eq(band)
        & events["antenna"].astype(str).eq(antenna)
    ].copy()
    rows = []
    for _, ev in sub_events.iterrows():
        local = _event_window(group, group_ns, ev, window_s)
        if local is None:
            continue
        tr, y = local
        for fit in _fit_methods(tr, y, str(ev["event_type"]), timing_offset_s, exclusion_s):
            rows.append(
                {
                    "source_name": source,
                    "target_id": target_id,
                    "event_id": ev.get("event_id"),
                    "event_type": ev["event_type"],
                    "predicted_event_time": ev["predicted_event_time"],
                    "month_block": pd.Timestamp(ev["predicted_event_time"]).strftime("%Y-%m"),
                    "frequency_band": band,
                    "frequency_mhz": float(source_row["target_frequency_mhz"]),
                    "antenna": antenna,
                    "window_s": window_s,
                    "timing_offset_s": timing_offset_s,
                    "method": fit.method,
                    "amplitude": fit.amp,
                    "local_sigma": fit.sigma,
                    "event_snr": fit.amp_over_sigma,
                    "n_samples": fit.n_samples,
                    "n_baseline": fit.n_baseline,
                    "delta_bic": fit.delta_bic,
                    "residual_ar1": fit.residual_ar1,
                    "runs_z": fit.runs_z,
                    "warning": fit.warning,
                }
            )
    return pd.DataFrame.from_records(rows)


def _aggregate(events: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows = []
    for keys, grp in events.groupby(by, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        meta = dict(zip(by, keys))
        amps = pd.to_numeric(grp["amplitude"], errors="coerce").dropna().to_numpy(dtype=float)
        if amps.size == 0:
            continue
        med = float(np.nanmedian(amps))
        sig = robust_sigma(amps)
        robust_snr = float(med / (sig / np.sqrt(amps.size))) if np.isfinite(sig) and sig > 0 else np.nan
        signs = np.sign(amps)
        n_same = int(np.count_nonzero(signs == np.sign(med))) if np.isfinite(med) and np.sign(med) != 0 else 0
        n_nonzero = int(np.count_nonzero(signs != 0))
        sign_p = float(binomtest(n_same, n_nonzero, 0.5, alternative="greater").pvalue) if n_nonzero > 0 else np.nan
        delta_bic = pd.to_numeric(grp["delta_bic"], errors="coerce")
        resid_ar1 = pd.to_numeric(grp["residual_ar1"], errors="coerce")
        runs_z = pd.to_numeric(grp["runs_z"], errors="coerce")
        bic_positive_fraction = float((delta_bic > 0).mean()) if len(delta_bic) else np.nan
        bic_strong_fraction = float((delta_bic > 6).mean()) if len(delta_bic) else np.nan
        median_delta_bic = float(np.nanmedian(delta_bic)) if len(delta_bic) else np.nan
        median_abs_ar1 = float(np.nanmedian(np.abs(resid_ar1))) if len(resid_ar1) else np.nan
        median_abs_runs_z = float(np.nanmedian(np.abs(runs_z))) if len(runs_z) else np.nan
        adequacy_score = _fit_quality_score(
            robust_snr=robust_snr,
            sign_fraction=float(n_same / n_nonzero) if n_nonzero else np.nan,
            sign_p=sign_p,
            median_delta_bic=median_delta_bic,
            bic_strong_fraction=bic_strong_fraction,
            median_abs_ar1=median_abs_ar1,
            median_abs_runs_z=median_abs_runs_z,
        )
        rows.append(
            {
                **meta,
                "n_events": int(grp["event_id"].nunique()),
                "n_fit_rows": int(len(grp)),
                "median_amplitude": med,
                "mean_amplitude": float(np.nanmean(amps)),
                "event_amp_robust_sigma": float(sig) if np.isfinite(sig) else np.nan,
                "robust_stack_snr": robust_snr,
                "event_sign_fraction": float(n_same / n_nonzero) if n_nonzero else np.nan,
                "sign_binomial_p": sign_p,
                "median_delta_bic": median_delta_bic,
                "bic_positive_fraction": bic_positive_fraction,
                "bic_strong_fraction": bic_strong_fraction,
                "median_abs_residual_ar1": median_abs_ar1,
                "median_abs_runs_z": median_abs_runs_z,
                "fit_quality_score": adequacy_score,
                "median_event_snr": float(np.nanmedian(pd.to_numeric(grp["event_snr"], errors="coerce"))),
                "warning_fraction": float(grp["warning"].fillna("").astype(str).ne("").mean()),
            }
        )
    return pd.DataFrame.from_records(rows)


def _fit_quality_score(
    robust_snr: float,
    sign_fraction: float,
    sign_p: float,
    median_delta_bic: float,
    bic_strong_fraction: float,
    median_abs_ar1: float,
    median_abs_runs_z: float,
) -> float:
    """0-1 heuristic fit-quality score, not a discovery probability."""
    score = 0.0
    if np.isfinite(robust_snr):
        score += 0.20 * min(abs(robust_snr) / 8.0, 1.0)
    if np.isfinite(sign_fraction):
        score += 0.20 * np.clip((sign_fraction - 0.5) / 0.35, 0.0, 1.0)
    if np.isfinite(sign_p):
        score += 0.15 * min(-np.log10(max(sign_p, 1e-300)) / 6.0, 1.0)
    if np.isfinite(median_delta_bic):
        score += 0.20 * np.clip((median_delta_bic + 2.0) / 12.0, 0.0, 1.0)
    if np.isfinite(bic_strong_fraction):
        score += 0.10 * np.clip(bic_strong_fraction / 0.5, 0.0, 1.0)
    if np.isfinite(median_abs_ar1):
        score += 0.075 * (1.0 - np.clip(median_abs_ar1 / 0.5, 0.0, 1.0))
    if np.isfinite(median_abs_runs_z):
        score += 0.075 * (1.0 - np.clip(median_abs_runs_z / 3.0, 0.0, 1.0))
    return float(score)


def _leave_one_month(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    source_col = "target_id" if "target_id" in events.columns else "source_name"
    for (source, method), grp in events.groupby([source_col, "method"], sort=True):
        months = sorted(grp["month_block"].dropna().unique())
        for month in months:
            sub = grp[~grp["month_block"].eq(month)]
            if sub.empty:
                continue
            agg = _aggregate(sub, [source_col, "method"])
            if agg.empty:
                continue
            row = agg.iloc[0].to_dict()
            row["left_out_month"] = month
            rows.append(row)
    return pd.DataFrame.from_records(rows)


def _plot_method_bars(source: str, summary: pd.DataFrame, out_dir: Path) -> Path:
    sub = summary[summary["source_name"].eq(source)].copy()
    sub["abs_snr"] = sub["robust_stack_snr"].abs()
    sub = sub.sort_values("abs_snr", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    labels = [METHOD_LABELS.get(m, m) for m in sub["method"]]
    vals = sub["robust_stack_snr"].to_numpy(dtype=float)
    ax.bar(labels, vals, color=["#4c78a8" if v >= 0 else "#d95f02" for v in vals])
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(3, color="0.65", lw=0.8, ls="--")
    ax.axhline(-3, color="0.65", lw=0.8, ls="--")
    ax.set_title(f"{_source_title(source)} baseline-model comparison")
    ax.set_ylabel("Robust stack SNR")
    ax.tick_params(axis="x", labelrotation=35)
    fig.tight_layout()
    path = out_dir / source / f"{source}_baseline_model_comparison.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_event_type_heatmap(source: str, event_type_summary: pd.DataFrame, out_dir: Path) -> Path:
    sub = event_type_summary[event_type_summary["source_name"].eq(source)].copy()
    pivot = sub.pivot_table(index="method", columns="event_type", values="robust_stack_snr", aggfunc="first")
    pivot = pivot.reindex([m for m in METHOD_LABELS if m in pivot.index])
    fig, ax = plt.subplots(figsize=(6.5, max(3.0, 0.45 * len(pivot))))
    data = pivot.to_numpy(dtype=float)
    vmax = np.nanmax(np.abs(data)) if np.isfinite(data).any() else 1.0
    im = ax.imshow(data, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([METHOD_LABELS.get(m, m) for m in pivot.index])
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8)
    ax.set_title(f"{_source_title(source)} event-type robustness by baseline model")
    fig.colorbar(im, ax=ax, label="Robust stack SNR")
    fig.tight_layout()
    path = out_dir / source / f"{source}_baseline_model_event_type_heatmap.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_quality_bars(source: str, summary: pd.DataFrame, out_dir: Path) -> Path:
    sub = summary[summary["source_name"].eq(source)].copy()
    sub = sub.sort_values("fit_quality_score", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    labels = [METHOD_LABELS.get(m, m) for m in sub["method"]]
    vals = sub["fit_quality_score"].to_numpy(dtype=float)
    ax.bar(labels, vals, color="#4c78a8")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"{_source_title(source)} fit-quality score by baseline model")
    ax.set_ylabel("Fit-quality score (0-1)")
    ax.tick_params(axis="x", labelrotation=35)
    fig.tight_layout()
    path = out_dir / source / f"{source}_fit_quality_score_comparison.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _consensus_quality(summary: pd.DataFrame, leave: pd.DataFrame) -> pd.DataFrame:
    raw_methods = {"prepost_median", "inner_median"}
    detrended_methods = {"detrended_prepost_median", "detrended_inner_median"}
    rows = []
    source_col = "target_id" if "target_id" in summary.columns else "source_name"
    leave_source_col = "target_id" if "target_id" in leave.columns else "source_name"
    for source, grp in summary.groupby(source_col, sort=True):
        raw = grp[grp["method"].isin(raw_methods)]
        detrended = grp[grp["method"].isin(detrended_methods)]
        model = grp[~grp["method"].isin(raw_methods | detrended_methods)]
        best_quality = grp.sort_values("fit_quality_score", ascending=False).iloc[0]
        best_snr = grp.iloc[grp["robust_stack_snr"].abs().argmax()]
        raw_pass = raw[(raw["fit_quality_score"] >= 0.55) & (raw["robust_stack_snr"].abs() >= 3)]
        detrended_pass = detrended[(detrended["fit_quality_score"] >= 0.55) & (detrended["robust_stack_snr"].abs() >= 3)]
        model_pass = model[(model["fit_quality_score"] >= 0.55) & (model["robust_stack_snr"].abs() >= 3)]
        raw_sign = np.sign(np.nanmedian(raw_pass["median_amplitude"])) if not raw_pass.empty else 0
        detrended_sign = np.sign(np.nanmedian(detrended_pass["median_amplitude"])) if not detrended_pass.empty else 0
        model_sign = np.sign(np.nanmedian(model_pass["median_amplitude"])) if not model_pass.empty else 0
        leave_sub = leave[leave[leave_source_col].eq(source)] if leave_source_col in leave.columns else pd.DataFrame()
        min_leave_quality = np.nan
        min_leave_abs_snr = np.nan
        if not leave_sub.empty and "fit_quality_score" in leave_sub:
            min_leave_quality = float(leave_sub.groupby("method")["fit_quality_score"].median().max())
            min_leave_abs_snr = float(leave_sub.groupby("method")["robust_stack_snr"].apply(lambda x: np.nanmin(np.abs(x))).max())
        consensus = "unresolved"
        reasons = []
        if len(raw_pass) == 0:
            reasons.append("no raw level-difference method passes quality")
        if len(detrended_pass) == 0:
            reasons.append("no detrended level-difference method passes quality")
        if len(model_pass) == 0:
            reasons.append("no model-based method passes quality")
        if raw_sign == 0 or detrended_sign == 0 or raw_sign != detrended_sign:
            reasons.append("raw/detrended sign disagreement")
        if raw_sign == 0 or model_sign == 0 or raw_sign != model_sign:
            reasons.append("raw/model sign disagreement")
        if np.isfinite(min_leave_abs_snr) and min_leave_abs_snr < 3:
            reasons.append("date-block stability below 3")
        if not reasons:
            consensus = "fit_quality_supported"
        elif len(raw_pass) > 0 and len(detrended_pass) == 0:
            consensus = "baseline_drift_sensitive"
        elif len(model_pass) > 0 and len(raw_pass) == 0:
            consensus = "model_only"
        elif len(model_pass) > 0 and len(raw_pass) > 0 and (raw_sign != model_sign):
            consensus = "baseline_sensitive"
        rows.append(
            {
                source_col: source,
                "best_quality_method": best_quality["method"],
                "best_fit_quality_score": float(best_quality["fit_quality_score"]),
                "best_snr_method": best_snr["method"],
                "best_abs_robust_snr": float(abs(best_snr["robust_stack_snr"])),
                "raw_methods_quality_pass": int(len(raw_pass)),
                "detrended_methods_quality_pass": int(len(detrended_pass)),
                "model_methods_quality_pass": int(len(model_pass)),
                "raw_detrended_sign_agree": bool(raw_sign != 0 and detrended_sign != 0 and raw_sign == detrended_sign),
                "raw_model_sign_agree": bool(raw_sign != 0 and model_sign != 0 and raw_sign == model_sign),
                "best_leave_one_month_min_abs_snr": min_leave_abs_snr,
                "consensus_status": consensus,
                "consensus_reason": "; ".join(reasons) if reasons else "raw and model families agree with acceptable quality",
            }
        )
    return pd.DataFrame.from_records(rows)


def _write_report(out_dir: Path, summary: pd.DataFrame, event_type: pd.DataFrame, leave: pd.DataFrame, paths: dict[str, list[Path]]) -> None:
    source_col = "target_id" if "target_id" in summary.columns else "source_name"
    best = summary.sort_values([source_col, "fit_quality_score"], ascending=[True, False]).groupby(source_col).head(3)
    consensus = _consensus_quality(summary, leave)
    consensus.to_csv(out_dir / "fit_quality_consensus_summary.csv", index=False)
    lines = [
        "# Baseline Model Playground",
        "",
        "Purpose: compare multiple baseline/signal estimators on the same selected source channels.",
        "",
        "Model families tested:",
        "",
        "- `joint_linear`: simultaneous linear baseline plus occultation template fit.",
        "- `joint_constant`: simultaneous constant baseline plus occultation template fit.",
        "- `robust_joint_constant`: constant baseline plus occultation template with Huber-style downweighting.",
        "- `robust_joint_huber`: same model with Huber-style iterative downweighting.",
        "- `sideband_constant`: fit a constant baseline only outside the event core.",
        "- `sideband_linear` and `sideband_quadratic`: fit baseline only outside the event core.",
        "- `sideband_spline`: smooth sideband baseline interpolation.",
        "- `prepost_median`: robust raw before/after sideband level difference.",
        "- `detrended_prepost_median`: linearly detrend sidebands, then take robust before/after sideband level difference.",
        "- `inner_median`: robust level difference close to the event.",
        "- `detrended_inner_median`: linearly detrend sidebands, then take robust near-event level difference.",
        "- `ar1_whitened_sideband`: sideband-linear residual with simple AR(1) prewhitening.",
        "",
        "Top methods by fit-quality score:",
        "",
        _markdown_table(best[[source_col, "method", "n_events", "median_amplitude", "robust_stack_snr", "event_sign_fraction", "median_delta_bic", "bic_strong_fraction", "median_abs_residual_ar1", "median_abs_runs_z", "fit_quality_score"]]),
        "",
        "Consensus quality summary:",
        "",
        _markdown_table(consensus),
        "",
        "Interpretation notes:",
        "",
        "- `robust_stack_snr` measures amplitude consistency, not fit adequacy.",
        "- `fit_quality_score` combines amplitude consistency, sign consistency, BIC improvement, residual autocorrelation, and residual runs behavior.",
        "- If only flexible residual models score well but raw median before/after models are weak, the signal is baseline-model-sensitive.",
        "- If raw before/after methods score well but linearly detrended before/after methods collapse, the signal is baseline-drift-sensitive.",
        "- A source-like occultation should be stable across raw level-difference, detrended level-difference, and at least one model-based method.",
        "- The Sun should not be promoted if the raw before/after estimators fail after simple linear detrending.",
        "",
        "Plots:",
        "",
    ]
    for source, source_paths in paths.items():
        lines.extend([f"## {_source_title(source)}", ""])
        for path in source_paths:
            lines.append(f"- `{path}`")
        lines.append("")
    (out_dir / "baseline_model_playground_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    work = df.copy()
    for col in work.columns:
        if pd.api.types.is_numeric_dtype(work[col]):
            work[col] = work[col].map(lambda x: "" if pd.isna(x) else f"{x:.4g}")
    cols = list(work.columns)
    widths = {col: max(len(col), *(len(str(v)) for v in work[col])) for col in cols}
    lines = [
        "| " + " | ".join(col.ljust(widths[col]) for col in cols) + " |",
        "| " + " | ".join("-" * widths[col] for col in cols) + " |",
    ]
    for _, row in work.iterrows():
        lines.append("| " + " | ".join(str(row[col]).ljust(widths[col]) for col in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--survey-root", default="outputs/planetary_confirmation_survey_science_baseline_v2")
    parser.add_argument("--events-path", default=None)
    parser.add_argument("--source-summary-path", default=None)
    parser.add_argument("--clean-path", default=str(CLEAN))
    parser.add_argument("--output-dir", default="outputs/baseline_model_playground_v1")
    parser.add_argument("--sideband-exclusion-seconds", type=float, default=120.0)
    args = parser.parse_args()

    survey = ROOT / args.survey_root
    out_dir = ensure_dir(ROOT / args.output_dir)
    clean_path = Path(args.clean_path)
    if not clean_path.is_absolute():
        clean_path = ROOT / clean_path
    clean = _read(clean_path, parse_dates=["time"])
    events_path = Path(args.events_path) if args.events_path else survey / "events" / "all_planet_predicted_events.csv"
    if not events_path.is_absolute():
        events_path = ROOT / events_path
    source_summary_path = Path(args.source_summary_path) if args.source_summary_path else survey / "planetary_confirmation_summary.csv"
    if not source_summary_path.is_absolute():
        source_summary_path = ROOT / source_summary_path
    events = _read(events_path, parse_dates=["predicted_event_time"])
    source_summary = _read(source_summary_path)
    event_tables = []
    for _, source_row in source_summary.iterrows():
        source = str(source_row["source_name"])
        target_id = str(source_row.get("target_id", source))
        table = _event_level_table(clean, events, source_row, args.sideband_exclusion_seconds)
        source_dir = ensure_dir(out_dir / target_id)
        table.to_csv(source_dir / f"{target_id}_baseline_model_event_fits.csv", index=False)
        event_tables.append(table)
    event_fits = pd.concat(event_tables, ignore_index=True) if event_tables else pd.DataFrame()
    source_col = "target_id" if "target_id" in event_fits.columns else "source_name"
    summary = _aggregate(event_fits, [source_col, "source_name", "frequency_band", "frequency_mhz", "antenna", "window_s", "timing_offset_s", "method"])
    event_type = _aggregate(event_fits, [source_col, "source_name", "method", "event_type"])
    leave = _leave_one_month(event_fits)
    summary.to_csv(out_dir / "baseline_model_summary.csv", index=False)
    event_type.to_csv(out_dir / "baseline_model_event_type_summary.csv", index=False)
    leave.to_csv(out_dir / "baseline_model_leave_one_month.csv", index=False)
    paths: dict[str, list[Path]] = {}
    plot_summary = summary.copy()
    plot_event_type = event_type.copy()
    if source_col != "source_name":
        plot_summary["source_name"] = plot_summary[source_col].astype(str)
        plot_event_type["source_name"] = plot_event_type[source_col].astype(str)
    for _, source_row in source_summary.iterrows():
        target_id = str(source_row.get("target_id", source_row["source_name"]))
        paths[target_id] = [
            _plot_method_bars(target_id, plot_summary, out_dir),
            _plot_quality_bars(target_id, plot_summary, out_dir),
            _plot_event_type_heatmap(target_id, plot_event_type, out_dir),
        ]
    _write_report(out_dir, summary, event_type, leave, paths)
    print(out_dir / "baseline_model_playground_report.md")
    print(out_dir / "baseline_model_summary.csv")


if __name__ == "__main__":
    main()
