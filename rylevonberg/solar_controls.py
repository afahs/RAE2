"""Solar moving-source controls and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .util import datetime_ns, robust_sigma


@dataclass(frozen=True)
class MovingBodyOffsetConfig:
    body_name: str = "sun"
    parent_source: str = "sun"
    radial_offsets_deg: tuple[float, ...] = (2.0, 5.0, 10.0, 20.0)
    annulus_radius_deg: float = 10.0
    annulus_positions: int = 8


def generate_moving_body_offset_controls(config: MovingBodyOffsetConfig | None = None) -> pd.DataFrame:
    """Generate off-ephemeris moving controls for the Sun.

    Controls follow the same apparent motion as the parent body but are
    displaced in the local tangent plane by east/north angular offsets.
    """
    cfg = config or MovingBodyOffsetConfig()
    rows: list[dict[str, object]] = []

    def add(control_type: str, east: float, north: float, offset: float, pa: float, notes: str) -> None:
        rows.append(
            {
                "source_name": f"{cfg.parent_source}_offephem_{len(rows):03d}",
                "kind": "body_offset",
                "body_name": cfg.body_name,
                "parent_source": cfg.parent_source,
                "control_name": f"{cfg.parent_source}_offephem_{len(rows):03d}",
                "control_type": control_type,
                "offset_deg": float(offset),
                "offset_pa_deg": float(pa),
                "offset_east_deg": float(east),
                "offset_north_deg": float(north),
                "frame": "fk4",
                "notes": notes,
            }
        )

    for off in cfg.radial_offsets_deg:
        add("same_track_east_west", off, 0.0, off, 90.0, "moving body track offset east in tangent plane")
        add("same_track_east_west", -off, 0.0, off, 270.0, "moving body track offset west in tangent plane")
        add("same_track_north_south", 0.0, off, off, 0.0, "moving body track offset north in tangent plane")
        add("same_track_north_south", 0.0, -off, off, 180.0, "moving body track offset south in tangent plane")

    for idx in range(int(cfg.annulus_positions)):
        pa = 360.0 * idx / float(cfg.annulus_positions)
        east = cfg.annulus_radius_deg * np.sin(np.deg2rad(pa))
        north = cfg.annulus_radius_deg * np.cos(np.deg2rad(pa))
        add("moving_annulus", east, north, cfg.annulus_radius_deg, pa, "moving annulus control around solar ephemeris")

    return pd.DataFrame.from_records(rows)


def event_burst_metrics(clean_df: pd.DataFrame, events_df: pd.DataFrame, window_seconds: float) -> pd.DataFrame:
    """Compute burst/variance metrics for event windows."""
    if events_df.empty:
        return pd.DataFrame()
    groups = {
        (freq, antenna): group.sort_values("time").reset_index(drop=True)
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
            rows.append({**base, "n_samples": 0})
            continue
        t_ns = datetime_ns(group["time"])
        event_ns = pd.Timestamp(ev["predicted_event_time"]).value
        half_ns = int(float(window_seconds) * 1e9)
        lo = int(np.searchsorted(t_ns, event_ns - half_ns, side="left"))
        hi = int(np.searchsorted(t_ns, event_ns + half_ns, side="right"))
        local = group.iloc[lo:hi]
        if local.empty:
            rows.append({**base, "n_samples": 0})
            continue
        rel = (datetime_ns(local["time"]) - event_ns).astype(float) / 1e9
        power = pd.to_numeric(local["power"], errors="coerce").to_numpy(dtype=float)
        valid = local["is_valid"].to_numpy(dtype=bool) if "is_valid" in local else np.isfinite(power)
        keep = valid & np.isfinite(power)
        pre = keep & (rel < 0.0)
        post = keep & (rel >= 0.0)
        vals = power[keep]
        med = float(np.nanmedian(vals)) if vals.size else np.nan
        sig = robust_sigma(vals - med) if vals.size else np.nan
        max_abs_z = float(np.nanmax(np.abs((vals - med) / sig))) if vals.size and np.isfinite(sig) and sig > 0 else np.nan
        pre_rms = float(np.nanstd(power[pre], ddof=1)) if np.count_nonzero(pre) > 1 else np.nan
        post_rms = float(np.nanstd(power[post], ddof=1)) if np.count_nonzero(post) > 1 else np.nan
        variance_ratio = float(max(pre_rms, post_rms) / max(min(pre_rms, post_rms), 1e-12)) if np.isfinite(pre_rms) and np.isfinite(post_rms) else np.nan
        rows.append(
            {
                **base,
                "n_samples": int(len(local)),
                "n_valid": int(np.count_nonzero(keep)),
                "local_median": med,
                "local_robust_sigma": float(sig) if np.isfinite(sig) else np.nan,
                "max_abs_robust_z": max_abs_z,
                "pre_rms": pre_rms,
                "post_rms": post_rms,
                "variance_ratio": variance_ratio,
                "burst_score": max_abs_z,
                "burst_like": bool(np.isfinite(max_abs_z) and max_abs_z >= 6.0),
            }
        )
    return pd.DataFrame.from_records(rows)


def date_cluster_summary(scored_df: pd.DataFrame, burst_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Summarize candidate/burst concentration by date."""
    if scored_df.empty:
        return pd.DataFrame()
    work = scored_df.copy()
    work["date"] = pd.to_datetime(work["predicted_event_time"]).dt.date.astype(str)
    work["candidate_like"] = work["detection_grade"].astype(str).isin(["candidate_needs_review", "strong_control_validated"])
    if burst_df is not None and not burst_df.empty:
        b = burst_df.copy()
        b["predicted_event_time"] = pd.to_datetime(b["predicted_event_time"]).astype(str)
        work["predicted_event_time"] = pd.to_datetime(work["predicted_event_time"]).astype(str)
        work = work.merge(
            b[["predicted_event_time", "frequency_band", "antenna", "burst_score", "burst_like"]],
            on=["predicted_event_time", "frequency_band", "antenna"],
            how="left",
        )
    rows = []
    for date, grp in work.groupby("date", sort=True):
        best = float(pd.to_numeric(grp["best_abs_snr"], errors="coerce").max()) if "best_abs_snr" in grp else np.nan
        rows.append(
            {
                "date": date,
                "n_rows": int(len(grp)),
                "n_candidate_like": int(grp["candidate_like"].sum()),
                "max_best_abs_snr": best,
                "n_burst_like": int(grp["burst_like"].astype(bool).sum()) if "burst_like" in grp else 0,
                "max_burst_score": float(pd.to_numeric(grp.get("burst_score", pd.Series(dtype=float)), errors="coerce").max()) if "burst_score" in grp else np.nan,
            }
        )
    return pd.DataFrame.from_records(rows).sort_values(["n_candidate_like", "max_best_abs_snr"], ascending=[False, False])
