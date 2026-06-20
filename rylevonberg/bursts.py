"""Burst-like transient search around predicted event windows."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .util import datetime_ns, robust_sigma


def _cluster_boolean(mask: np.ndarray) -> list[tuple[int, int]]:
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []
    clusters: list[tuple[int, int]] = []
    start = int(idx[0])
    last = int(idx[0])
    for value in idx[1:]:
        value = int(value)
        if value == last + 1:
            last = value
        else:
            clusters.append((start, last))
            start = last = value
    clusters.append((start, last))
    return clusters


def search_burst_events(
    clean_df: pd.DataFrame,
    events_df: pd.DataFrame,
    window_seconds: float = 900.0,
    z_threshold: float = 8.0,
    min_cluster_samples: int = 1,
) -> pd.DataFrame:
    """Find impulsive outlier clusters in event-centered windows."""
    rows: list[dict] = []
    if clean_df.empty or events_df.empty:
        return pd.DataFrame()
    groups = {
        (freq, antenna): group.sort_values("time").reset_index(drop=True)
        for (freq, antenna), group in clean_df.groupby(["frequency_band", "antenna"], dropna=False, sort=True)
    }
    burst_id = 0
    for _, event in events_df.iterrows():
        freq = event.get("frequency_band")
        antenna = event.get("antenna")
        group = groups.get((freq, antenna))
        if group is None or group.empty:
            continue
        times = pd.DatetimeIndex(group["time"])
        t_ns = datetime_ns(times)
        event_time = pd.Timestamp(event["predicted_event_time"])
        event_ns = event_time.value
        half_ns = int(float(window_seconds) * 1e9)
        lo = int(np.searchsorted(t_ns, event_ns - half_ns, side="left"))
        hi = int(np.searchsorted(t_ns, event_ns + half_ns, side="right"))
        if hi <= lo:
            continue
        local = group.iloc[lo:hi].reset_index(drop=True)
        valid = local["is_valid"].to_numpy(dtype=bool) if "is_valid" in local.columns else np.ones(len(local), dtype=bool)
        y = local["power"].to_numpy(dtype=float)
        finite = np.isfinite(y) & valid
        if np.count_nonzero(finite) < max(6, int(min_cluster_samples)):
            continue
        baseline = float(np.nanmedian(y[finite]))
        sigma = robust_sigma(y[finite] - baseline)
        if not np.isfinite(sigma) or sigma <= 0:
            continue
        z = (y - baseline) / sigma
        keep = finite & (np.abs(z) >= float(z_threshold))
        for start, end in _cluster_boolean(keep):
            n = end - start + 1
            if n < int(min_cluster_samples):
                continue
            segment_z = z[start : end + 1]
            peak_local = int(start + np.nanargmax(np.abs(segment_z)))
            peak_time = pd.Timestamp(local.loc[peak_local, "time"])
            rows.append(
                {
                    "burst_id": burst_id,
                    "source_name": event.get("source_name"),
                    "event_id": event.get("event_id"),
                    "event_type": event.get("event_type"),
                    "predicted_event_time": event_time,
                    "frequency_band": freq,
                    "frequency_mhz": event.get("frequency_mhz", np.nan),
                    "antenna": antenna,
                    "burst_start_time": pd.Timestamp(local.loc[start, "time"]),
                    "burst_end_time": pd.Timestamp(local.loc[end, "time"]),
                    "burst_peak_time": peak_time,
                    "peak_time_offset_sec": float((peak_time - event_time).total_seconds()),
                    "n_samples": int(n),
                    "baseline_power": baseline,
                    "robust_sigma": float(sigma),
                    "peak_power": float(y[peak_local]),
                    "peak_z": float(z[peak_local]),
                    "abs_peak_z": float(abs(z[peak_local])),
                    "sign": int(np.sign(z[peak_local])),
                    "quality_flags": "burst_candidate_not_occultation_step",
                }
            )
            burst_id += 1
    out = pd.DataFrame.from_records(rows)
    if out.empty:
        return out
    return out.sort_values(["abs_peak_z", "burst_peak_time"], ascending=[False, True]).reset_index(drop=True)
