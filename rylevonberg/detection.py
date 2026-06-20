"""Step-template and matched-filter detection models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .util import datetime_ns, robust_sigma
from .sample_quality import add_strict_valid_column


@dataclass(frozen=True)
class StepFitConfig:
    window_seconds: float = 600.0
    baseline_order: int = 1
    min_samples_per_side: int = 4
    smooth_seconds: float = 0.0
    timing_grid_seconds: tuple[float, ...] = (-120.0, -60.0, -30.0, 0.0, 30.0, 60.0, 120.0)


def event_template(t_rel_sec: np.ndarray, event_type: str, smooth_seconds: float = 0.0, timing_offset_sec: float = 0.0) -> np.ndarray:
    x = np.asarray(t_rel_sec, dtype=float) - float(timing_offset_sec)
    if smooth_seconds and smooth_seconds > 0.0:
        step = -np.tanh(x / float(smooth_seconds))
    else:
        step = np.where(x < 0.0, 1.0, -1.0)
    if str(event_type).lower() == "reappearance":
        step = -step
    return 0.5 * step


def baseline_matrix(t_rel_sec: np.ndarray, order: int) -> np.ndarray:
    cols = [np.ones_like(t_rel_sec, dtype=float)]
    if int(order) >= 1:
        scale = max(float(np.nanmax(np.abs(t_rel_sec))), 1.0)
        cols.append(t_rel_sec / scale)
    if int(order) >= 2:
        scale = max(float(np.nanmax(np.abs(t_rel_sec))), 1.0)
        cols.append((t_rel_sec / scale) ** 2)
    return np.column_stack(cols)


def _fit_linear(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(int(len(y) - X.shape[1]), 1)
    rss = float(np.sum(resid**2))
    sigma2 = rss / dof
    cov = np.linalg.pinv(X.T @ X) * sigma2
    return beta, resid, rss, cov


def fit_step_template(
    times: pd.DatetimeIndex,
    values: np.ndarray,
    event_time: pd.Timestamp,
    event_type: str,
    config: StepFitConfig,
    valid_mask: np.ndarray | None = None,
) -> dict[str, float]:
    t_ns = datetime_ns(times)
    event_ns = pd.Timestamp(event_time).value
    half_ns = int(float(config.window_seconds) * 1e9)
    in_window = (t_ns >= event_ns - half_ns) & (t_ns <= event_ns + half_ns)
    if valid_mask is not None:
        in_window &= np.asarray(valid_mask, dtype=bool)
    vals = np.asarray(values, dtype=float)
    in_window &= np.isfinite(vals)
    t_rel = (t_ns[in_window] - event_ns).astype(float) / 1e9
    y = vals[in_window].astype(float)
    n_pre = int(np.count_nonzero(t_rel < 0.0))
    n_post = int(np.count_nonzero(t_rel >= 0.0))
    pre_times = t_rel[t_rel < 0.0]
    post_times = t_rel[t_rel >= 0.0]
    pre_span = float(np.nanmax(pre_times) - np.nanmin(pre_times)) if pre_times.size > 1 else 0.0
    post_span = float(np.nanmax(post_times) - np.nanmin(post_times)) if post_times.size > 1 else 0.0
    cadence = np.diff(np.sort(t_rel))
    cadence = cadence[cadence > 0]
    median_cadence = float(np.nanmedian(cadence)) if cadence.size else np.nan
    baseline_warning = ""
    if n_pre < max(6, config.baseline_order + 3) or n_post < max(6, config.baseline_order + 3):
        baseline_warning = "sparse_pre_or_post_samples"
    elif np.isfinite(median_cadence) and median_cadence > 0 and (pre_span < 2.0 * median_cadence or post_span < 2.0 * median_cadence):
        baseline_warning = "narrow_pre_or_post_time_span"
    if n_pre < config.min_samples_per_side or n_post < config.min_samples_per_side:
        return {
            "n_used": int(y.size),
            "n_pre": n_pre,
            "n_post": n_post,
            "pre_time_span_s": pre_span,
            "post_time_span_s": post_span,
            "median_sample_spacing_s": median_cadence,
            "baseline_support_warning": baseline_warning or "too_few_samples_for_step_fit",
            "amplitude": np.nan,
            "uncertainty": np.nan,
            "detection_snr": np.nan,
            "chi2_improvement": np.nan,
            "timing_offset_sec": np.nan,
        }

    B = baseline_matrix(t_rel, config.baseline_order)
    _, _, rss0, _ = _fit_linear(B, y)
    best: dict[str, float] | None = None
    for offset in config.timing_grid_seconds:
        step = event_template(t_rel, event_type, smooth_seconds=config.smooth_seconds, timing_offset_sec=offset)
        X = np.column_stack([B, step])
        beta, resid, rss1, cov = _fit_linear(X, y)
        amp = float(beta[-1])
        var = float(cov[-1, -1]) if cov.size else np.nan
        se = float(np.sqrt(var)) if np.isfinite(var) and var >= 0 else np.nan
        snr = float(amp / se) if se and np.isfinite(se) and se > 0 else np.nan
        chi2_imp = float(max(rss0 - rss1, 0.0))
        row = {
            "n_used": int(y.size),
            "n_pre": n_pre,
            "n_post": n_post,
            "pre_time_span_s": pre_span,
            "post_time_span_s": post_span,
            "median_sample_spacing_s": median_cadence,
            "baseline_support_warning": baseline_warning,
            "amplitude": amp,
            "uncertainty": se,
            "sign": float(np.sign(amp)) if np.isfinite(amp) else np.nan,
            "timing_offset_sec": float(offset),
            "chi2_improvement": chi2_imp,
            "detection_snr": snr,
            "sigma_resid": float(np.sqrt(np.mean(resid**2))) if resid.size else np.nan,
        }
        if best is None or row["chi2_improvement"] > best["chi2_improvement"]:
            best = row
    return best or {}


def run_stepfit_detections(clean_df: pd.DataFrame, events_df: pd.DataFrame, config: StepFitConfig) -> pd.DataFrame:
    rows = []
    if events_df.empty:
        return pd.DataFrame()
    groups = {
        (freq, antenna): add_strict_valid_column(group.sort_values("time").reset_index(drop=True))
        for (freq, antenna), group in clean_df.groupby(["frequency_band", "antenna"], dropna=False, sort=True)
    }
    for _, event in events_df.iterrows():
        freq = event.get("frequency_band")
        antenna = event.get("antenna")
        group = groups.get((freq, antenna))
        if group is None:
            group = clean_df.sort_values("time")
            if pd.notna(freq):
                group = group[group["frequency_band"] == freq]
            if pd.notna(antenna):
                group = group[group["antenna"] == antenna]
            group = add_strict_valid_column(group.reset_index(drop=True))
        if group.empty:
            continue
        times_all = pd.DatetimeIndex(group["time"])
        event_ns = pd.Timestamp(event["predicted_event_time"]).value
        half_ns = int(float(config.window_seconds) * 1e9)
        t_ns = datetime_ns(times_all)
        lo = int(np.searchsorted(t_ns, event_ns - half_ns, side="left"))
        hi = int(np.searchsorted(t_ns, event_ns + half_ns, side="right"))
        if hi <= lo:
            continue
        local = group.iloc[lo:hi]
        times = pd.DatetimeIndex(local["time"])
        valid = local["strict_is_valid"].to_numpy(dtype=bool) if "strict_is_valid" in local.columns else None
        fit = fit_step_template(
            times,
            local["power"].to_numpy(dtype=float),
            pd.Timestamp(event["predicted_event_time"]),
            str(event["event_type"]),
            config,
            valid,
        )
        rows.append({**event.to_dict(), "method": "local_step_template", **fit})
    return pd.DataFrame.from_records(rows)


def matched_filter_event(
    times: pd.DatetimeIndex,
    values: np.ndarray,
    event_time: pd.Timestamp,
    event_type: str,
    window_seconds: float,
    baseline_order: int = 1,
    valid_mask: np.ndarray | None = None,
) -> dict[str, float]:
    t_ns = datetime_ns(times)
    event_ns = pd.Timestamp(event_time).value
    in_window = np.abs(t_ns - event_ns) <= int(float(window_seconds) * 1e9)
    vals = np.asarray(values, dtype=float)
    if valid_mask is not None:
        in_window &= np.asarray(valid_mask, dtype=bool)
    in_window &= np.isfinite(vals)
    t_rel = (t_ns[in_window] - event_ns).astype(float) / 1e9
    y = vals[in_window].astype(float)
    if y.size < max(6, baseline_order + 3):
        return {"matched_amp": np.nan, "matched_snr": np.nan, "matched_n": int(y.size)}
    B = baseline_matrix(t_rel, baseline_order)
    beta, *_ = np.linalg.lstsq(B, y, rcond=None)
    yd = y - B @ beta
    template = event_template(t_rel, event_type)
    tbeta, *_ = np.linalg.lstsq(B, template, rcond=None)
    td = template - B @ tbeta
    denom = float(np.dot(td, td))
    if denom <= 0:
        return {"matched_amp": np.nan, "matched_snr": np.nan, "matched_n": int(y.size)}
    amp = float(np.dot(yd, td) / denom)
    sigma = robust_sigma(yd)
    se = float(sigma / np.sqrt(denom)) if np.isfinite(sigma) and sigma > 0 else np.nan
    return {"matched_amp": amp, "matched_snr": float(amp / se) if se and se > 0 else np.nan, "matched_n": int(y.size)}


def run_matched_filter(clean_df: pd.DataFrame, events_df: pd.DataFrame, window_seconds: float = 600.0, baseline_order: int = 1) -> pd.DataFrame:
    rows = []
    if events_df.empty:
        return pd.DataFrame()
    groups = {
        (freq, antenna): add_strict_valid_column(group.sort_values("time").reset_index(drop=True))
        for (freq, antenna), group in clean_df.groupby(["frequency_band", "antenna"], dropna=False, sort=True)
    }
    for _, event in events_df.iterrows():
        freq = event.get("frequency_band")
        antenna = event.get("antenna")
        group = groups.get((freq, antenna))
        if group is None:
            group = clean_df.sort_values("time")
            if pd.notna(freq):
                group = group[group["frequency_band"] == freq]
            if pd.notna(antenna):
                group = group[group["antenna"] == antenna]
            group = add_strict_valid_column(group.reset_index(drop=True))
        if group.empty:
            continue
        times_all = pd.DatetimeIndex(group["time"])
        event_ns = pd.Timestamp(event["predicted_event_time"]).value
        half_ns = int(float(window_seconds) * 1e9)
        t_ns = datetime_ns(times_all)
        lo = int(np.searchsorted(t_ns, event_ns - half_ns, side="left"))
        hi = int(np.searchsorted(t_ns, event_ns + half_ns, side="right"))
        if hi <= lo:
            continue
        local = group.iloc[lo:hi]
        row = matched_filter_event(
            pd.DatetimeIndex(local["time"]),
            local["power"].to_numpy(dtype=float),
            pd.Timestamp(event["predicted_event_time"]),
            str(event["event_type"]),
            window_seconds=window_seconds,
            baseline_order=baseline_order,
            valid_mask=local["strict_is_valid"].to_numpy(dtype=bool) if "strict_is_valid" in local.columns else None,
        )
        rows.append({**event.to_dict(), "method": "matched_filter", **row})
    return pd.DataFrame.from_records(rows)
