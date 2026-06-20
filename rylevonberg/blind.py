"""Blind step-like change-point search and limb-constraint cataloging."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import SPACECRAFT_COLUMNS, add_frequency_mhz_column, frequency_mhz_for_band
from .geometry import moon_angular_radius_deg, moon_center_direction
from .util import robust_sigma


def blind_changepoints(
    clean_df: pd.DataFrame,
    window_samples: int = 8,
    snr_threshold: float = 5.0,
    max_candidates_per_series: int = 100,
) -> pd.DataFrame:
    rows = []
    for (freq, ant), group in clean_df.sort_values("time").groupby(["frequency_band", "antenna"], sort=True):
        valid = group["is_valid"].to_numpy(dtype=bool) if "is_valid" in group.columns else np.ones(len(group), dtype=bool)
        group = group[valid].reset_index(drop=True)
        y = group["power"].to_numpy(dtype=float)
        if len(group) < 2 * window_samples + 1:
            continue
        amps = []
        centers = []
        for idx in range(window_samples, len(y) - window_samples):
            pre = y[idx - window_samples : idx]
            post = y[idx : idx + window_samples]
            amp = float(np.median(post) - np.median(pre))
            amps.append(amp)
            centers.append(idx)
        amps_arr = np.asarray(amps)
        sig = robust_sigma(amps_arr)
        if not np.isfinite(sig) or sig <= 0:
            continue
        snr = amps_arr / sig
        keep = np.where(np.abs(snr) >= float(snr_threshold))[0]
        if keep.size > max_candidates_per_series:
            order = np.argsort(np.abs(snr[keep]))[::-1][:max_candidates_per_series]
            keep = keep[order]
        sc = group[SPACECRAFT_COLUMNS].to_numpy(dtype=float)
        center_dirs = moon_center_direction(sc)
        radii = moon_angular_radius_deg(sc)
        for k in keep:
            idx = int(centers[int(k)])
            rows.append(
                {
                    "candidate_time": group.loc[idx, "time"],
                    "frequency_band": freq,
                    "frequency_mhz": frequency_mhz_for_band(freq),
                    "antenna": ant,
                    "step_amplitude_post_minus_pre": float(amps_arr[k]),
                    "changepoint_snr": float(snr[k]),
                    "limb_center_x": float(center_dirs[idx, 0]),
                    "limb_center_y": float(center_dirs[idx, 1]),
                    "limb_center_z": float(center_dirs[idx, 2]),
                    "limb_radius_deg": float(radii[idx]),
                    "quality_flags": "limb_circle_constraint_not_unique_position",
                }
            )
    out = pd.DataFrame.from_records(rows)
    return add_frequency_mhz_column(out.sort_values("candidate_time").reset_index(drop=True)) if not out.empty else out


def cluster_blind_constraints(candidates: pd.DataFrame, time_tolerance_seconds: float = 120.0) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    work = candidates.sort_values("candidate_time").reset_index(drop=True)
    times = pd.DatetimeIndex(work["candidate_time"])
    cluster_ids = []
    cid = 0
    last = times[0]
    for t in times:
        if (t - last).total_seconds() > float(time_tolerance_seconds):
            cid += 1
        cluster_ids.append(cid)
        last = t
    work["cluster_id"] = cluster_ids
    rows = []
    for cid, grp in work.groupby("cluster_id"):
        rows.append(
            {
                "cluster_id": int(cid),
                "n_candidates": int(len(grp)),
                "start_time": grp["candidate_time"].min(),
                "end_time": grp["candidate_time"].max(),
                "max_abs_snr": float(np.max(np.abs(grp["changepoint_snr"]))),
                "mean_limb_center_x": float(np.mean(grp["limb_center_x"])),
                "mean_limb_center_y": float(np.mean(grp["limb_center_y"])),
                "mean_limb_center_z": float(np.mean(grp["limb_center_z"])),
                "mean_limb_radius_deg": float(np.mean(grp["limb_radius_deg"])),
            }
        )
    return pd.DataFrame.from_records(rows)
