"""Event-window quality diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .detection import baseline_matrix
from .sample_quality import add_strict_valid_column
from .util import datetime_ns, robust_sigma


def _flag_fraction(flags: pd.Series, pattern: str) -> float:
    if flags.empty:
        return np.nan
    return float(flags.astype(str).str.contains(pattern, regex=False).mean())


def _primary_failure(row: dict[str, object]) -> str:
    if int(row.get("valid_samples_pre", 0)) < 4 or int(row.get("valid_samples_post", 0)) < 4:
        return "too_few_valid_samples"
    if float(row.get("gap_fraction", 0.0)) > 0.25:
        return "severe_gap_fraction"
    if float(row.get("nonpositive_fraction", 0.0)) > 0.25:
        return "severe_nonpositive_fraction"
    if float(row.get("high_clip_fraction", 0.0)) > 0.25:
        return "severe_high_clip_fraction"
    if float(row.get("outlier_fraction", 0.0)) > 0.2:
        return "outlier_fraction"
    return ""


def event_window_quality(clean_df: pd.DataFrame, events_df: pd.DataFrame, window_seconds: float) -> pd.DataFrame:
    """Compute pre-fit/post-fit quality diagnostics for event windows."""
    if events_df.empty:
        return pd.DataFrame()
    groups = {
        (freq, antenna): add_strict_valid_column(group.sort_values("time").reset_index(drop=True))
        for (freq, antenna), group in clean_df.groupby(["frequency_band", "antenna"], dropna=False, sort=True)
    }
    rows: list[dict[str, object]] = []
    for _, ev in events_df.iterrows():
        freq = ev.get("frequency_band")
        antenna = ev.get("antenna")
        group = groups.get((freq, antenna))
        base = {
            "source_name": ev.get("source_name"),
            "event_type": ev.get("event_type"),
            "predicted_event_time": ev.get("predicted_event_time"),
            "frequency_band": freq,
            "frequency_mhz": ev.get("frequency_mhz"),
            "antenna": antenna,
            "window_s": float(window_seconds),
        }
        if group is None or group.empty:
            rows.append({**base, "n_samples": 0, "primary_quality_failure": "no_channel_data"})
            continue
        t_ns = datetime_ns(group["time"])
        event_ns = pd.Timestamp(ev["predicted_event_time"]).value
        half_ns = int(float(window_seconds) * 1e9)
        lo = int(np.searchsorted(t_ns, event_ns - half_ns, side="left"))
        hi = int(np.searchsorted(t_ns, event_ns + half_ns, side="right"))
        local = group.iloc[lo:hi].copy()
        if local.empty:
            rows.append({**base, "n_samples": 0, "primary_quality_failure": "empty_window"})
            continue
        rel = (datetime_ns(local["time"]) - event_ns).astype(float) / 1e9
        power = pd.to_numeric(local["power"], errors="coerce").to_numpy(dtype=float)
        valid = local["strict_is_valid"].to_numpy(dtype=bool) if "strict_is_valid" in local else np.isfinite(power)
        flags = local["quality_flags"] if "quality_flags" in local else pd.Series([""] * len(local))
        flag_text = flags.astype(str)
        jump_mask = flag_text.str.contains("telemetry_artifact_jump", regex=False).to_numpy(dtype=bool)
        finite = np.isfinite(power)
        nonpositive_mask = local["strict_nonpositive_power"].to_numpy(dtype=bool) if "strict_nonpositive_power" in local else (finite & (power <= 0.0))
        high_clip_mask = local["strict_high_clip_power"].to_numpy(dtype=bool) if "strict_high_clip_power" in local else np.zeros(len(local), dtype=bool)
        valid_finite = valid & finite
        pre = rel < 0.0
        post = rel >= 0.0
        dt = np.diff(datetime_ns(local["time"]).astype(float) / 1e9)
        cadence_irregularity = np.nan
        if dt.size > 2:
            med = np.nanmedian(dt[dt > 0])
            cadence_irregularity = float(robust_sigma(dt / med - 1.0)) if np.isfinite(med) and med > 0 else np.nan
        baseline_slope = np.nan
        baseline_curvature = np.nan
        if np.count_nonzero(valid_finite) >= 6:
            order = 2 if np.count_nonzero(valid_finite) >= 8 else 1
            X = baseline_matrix(rel[valid_finite], order)
            beta, *_ = np.linalg.lstsq(X, power[valid_finite], rcond=None)
            baseline_slope = float(beta[1]) if len(beta) > 1 else np.nan
            baseline_curvature = float(beta[2]) if len(beta) > 2 else np.nan
        resid = power[valid_finite] - np.nanmedian(power[valid_finite]) if np.count_nonzero(valid_finite) else np.array([])
        sig = robust_sigma(resid) if resid.size else np.nan
        outlier_fraction = float(np.mean(np.abs(resid) > 6.0 * sig)) if np.isfinite(sig) and sig > 0 else 0.0
        row = {
            **base,
            "n_samples": int(len(local)),
            "valid_samples_pre": int(np.count_nonzero(valid_finite & pre)),
            "valid_samples_post": int(np.count_nonzero(valid_finite & post)),
            "valid_fraction": float(np.mean(valid_finite)),
            "gap_fraction": _flag_fraction(flags, "gap_after_previous"),
            "nonpositive_fraction": float(np.mean(nonpositive_mask)),
            "saturation_fraction": float(np.mean(high_clip_mask)),
            "high_clip_fraction": float(np.mean(high_clip_mask)),
            "invalid_power_fraction": float(np.mean(nonpositive_mask | high_clip_mask)),
            "jump_count": int(np.count_nonzero(jump_mask)),
            "jump_fraction": float(np.mean(jump_mask)),
            "jump_count_pre": int(np.count_nonzero(jump_mask & pre)),
            "jump_count_post": int(np.count_nonzero(jump_mask & post)),
            "jump_count_near_event": int(np.count_nonzero(jump_mask & (np.abs(rel) <= 60.0))),
            "local_rms_pre": float(np.nanstd(power[valid_finite & pre], ddof=1)) if np.count_nonzero(valid_finite & pre) > 1 else np.nan,
            "local_rms_post": float(np.nanstd(power[valid_finite & post], ddof=1)) if np.count_nonzero(valid_finite & post) > 1 else np.nan,
            "baseline_slope": baseline_slope,
            "baseline_curvature": baseline_curvature,
            "cadence_irregularity": cadence_irregularity,
            "outlier_fraction": outlier_fraction,
        }
        row["primary_quality_failure"] = _primary_failure(row)
        rows.append(row)
    return pd.DataFrame.from_records(rows)
