"""Event profile extraction and stacking."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import add_frequency_mhz_column
from .detection import baseline_matrix, event_template
from .sample_quality import add_strict_valid_column
from .util import datetime_ns, robust_sigma


def detrend_profile_values(
    t_rel_sec: np.ndarray,
    values: np.ndarray,
    event_type: str,
    baseline_mode: str = "sideband_linear",
    sideband_exclusion_seconds: float = 120.0,
    timing_offset_sec: float = 0.0,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Return detrended profile values and template for one event window.

    `sideband_linear` is the science default for profile stacks: it fits the
    baseline outside the central occultation region so the baseline cannot
    directly absorb the predicted step. `linear_all` is retained for legacy
    comparisons.
    """
    tr = np.asarray(t_rel_sec, dtype=float)
    y = np.asarray(values, dtype=float)
    template = event_template(tr, str(event_type), timing_offset_sec=float(timing_offset_sec))
    mode = str(baseline_mode)
    if mode == "linear_all":
        fit_mask = np.ones(len(y), dtype=bool)
        B_fit = baseline_matrix(tr[fit_mask], 1)
        beta, *_ = np.linalg.lstsq(B_fit, y[fit_mask], rcond=None)
        baseline = baseline_matrix(tr, 1) @ beta
        residual_for_sigma = y - baseline
        profile = residual_for_sigma
    elif mode == "constant_all":
        fit_mask = np.ones(len(y), dtype=bool)
        baseline = np.full_like(y, np.nanmedian(y))
        residual_for_sigma = y - baseline
        profile = residual_for_sigma
    elif mode == "sideband_linear":
        fit_mask = np.abs(tr) >= float(sideband_exclusion_seconds)
        if np.count_nonzero(fit_mask) < 6:
            fit_mask = np.ones(len(y), dtype=bool)
        B_fit = baseline_matrix(tr[fit_mask], 1)
        beta, *_ = np.linalg.lstsq(B_fit, y[fit_mask], rcond=None)
        baseline = baseline_matrix(tr, 1) @ beta
        profile = y - baseline
        residual_for_sigma = profile
    elif mode == "sideband_constant":
        fit_mask = np.abs(tr) >= float(sideband_exclusion_seconds)
        if np.count_nonzero(fit_mask) < 4:
            fit_mask = np.ones(len(y), dtype=bool)
        baseline = np.full_like(y, np.nanmedian(y[fit_mask]))
        profile = y - baseline
        residual_for_sigma = profile
    elif mode == "pre_event_anchor":
        fit_mask = tr <= -float(sideband_exclusion_seconds)
        if np.count_nonzero(fit_mask) < 4:
            fit_mask = tr < 0.0
        if np.count_nonzero(fit_mask) < 4:
            fit_mask = np.ones(len(y), dtype=bool)
        baseline = np.full_like(y, np.nanmedian(y[fit_mask]))
        profile = y - baseline
        residual_for_sigma = profile
    elif mode == "joint_step_linear":
        fit_mask = np.ones(len(y), dtype=bool)
        B = baseline_matrix(tr, 1)
        X = np.column_stack([B, template])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        baseline = B @ beta[:-1]
        residual_for_sigma = y - X @ beta
        profile = y - baseline
    else:
        raise ValueError(f"unknown baseline_mode: {baseline_mode}")
    sigma = robust_sigma(residual_for_sigma)
    if normalize and np.isfinite(sigma) and sigma > 0:
        profile = profile / sigma
    return profile, template, float(sigma) if np.isfinite(sigma) else np.nan, int(np.count_nonzero(fit_mask))


def aligned_profiles(
    clean_df: pd.DataFrame,
    events_df: pd.DataFrame,
    window_seconds: float = 600.0,
    bin_seconds: float = 30.0,
    normalize: bool = True,
    baseline_mode: str = "sideband_linear",
    sideband_exclusion_seconds: float = 120.0,
) -> pd.DataFrame:
    rows = []
    if events_df.empty:
        return pd.DataFrame()
    groups = {
        (freq, antenna): add_strict_valid_column(group.sort_values("time").reset_index(drop=True))
        for (freq, antenna), group in clean_df.groupby(["frequency_band", "antenna"], dropna=False, sort=True)
    }
    for _, ev in events_df.iterrows():
        freq = ev.get("frequency_band")
        antenna = ev.get("antenna")
        group = groups.get((freq, antenna))
        if group is None:
            group = clean_df
            if pd.notna(freq):
                group = group[group["frequency_band"] == freq]
            if pd.notna(antenna):
                group = group[group["antenna"] == antenna]
            group = add_strict_valid_column(group.sort_values("time").reset_index(drop=True))
        if group.empty:
            continue
        t = pd.DatetimeIndex(group["time"])
        event_ns = pd.Timestamp(ev["predicted_event_time"]).value
        half_ns = int(float(window_seconds) * 1e9)
        t_ns = datetime_ns(t)
        lo = int(np.searchsorted(t_ns, event_ns - half_ns, side="left"))
        hi = int(np.searchsorted(t_ns, event_ns + half_ns, side="right"))
        if hi <= lo:
            continue
        local = group.iloc[lo:hi]
        t_rel = (datetime_ns(local["time"]) - event_ns).astype(float) / 1e9
        valid = local["strict_is_valid"].to_numpy(dtype=bool) if "strict_is_valid" in local.columns else np.ones(len(local), dtype=bool)
        keep = (np.abs(t_rel) <= float(window_seconds)) & valid
        y = local["power"].to_numpy(dtype=float)
        keep &= np.isfinite(y)
        if np.count_nonzero(keep) < 6:
            continue
        tr = t_rel[keep]
        yy = y[keep]
        yd, tmpl, sig, n_baseline = detrend_profile_values(
            tr,
            yy,
            str(ev["event_type"]),
            baseline_mode=baseline_mode,
            sideband_exclusion_seconds=sideband_exclusion_seconds,
            normalize=normalize,
        )
        tb = np.round(tr / float(bin_seconds)) * float(bin_seconds)
        for trel, tbin, val, tv in zip(tr, tb, yd, tmpl):
            rows.append({
                **ev.to_dict(),
                "t_rel_sec": float(trel),
                "t_bin_sec": float(tbin),
                "profile_value": float(val),
                "template": float(tv),
                "baseline_mode": str(baseline_mode),
                "n_baseline_samples": int(n_baseline),
                "baseline_sigma": float(sig) if np.isfinite(sig) else np.nan,
            })
    return add_frequency_mhz_column(pd.DataFrame.from_records(rows))


def stack_profiles(profile_df: pd.DataFrame, by: list[str] | None = None, seed: int = 12345, n_bootstrap: int = 200) -> tuple[pd.DataFrame, pd.DataFrame]:
    if profile_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    group_cols = list(by or ["source_name", "frequency_band", "antenna"])
    rows = []
    for keys, grp in profile_df.groupby([*group_cols, "t_bin_sec"], dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        meta = dict(zip([*group_cols, "t_bin_sec"], keys))
        vals = grp["profile_value"].to_numpy(dtype=float)
        rows.append(
            {
                **meta,
                "n_samples": int(vals.size),
                "n_events": int(grp["event_id"].nunique()),
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "ivar_mean": float(np.average(vals)),
                "sem": float(np.std(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else np.nan,
            }
        )
    stacked = pd.DataFrame.from_records(rows)

    summary_rows = []
    rng = np.random.default_rng(seed)
    for keys, grp in profile_df.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        meta = dict(zip(group_cols, keys))
        vals = grp["profile_value"].to_numpy(dtype=float)
        tmpl = grp["template"].to_numpy(dtype=float)
        denom = float(np.dot(tmpl, tmpl))
        amp = float(np.dot(vals, tmpl) / denom) if denom > 0 else np.nan
        sig = robust_sigma(vals)
        snr = float(amp * np.sqrt(denom) / sig) if np.isfinite(sig) and sig > 0 else np.nan
        boots = []
        event_ids = grp["event_id"].drop_duplicates().to_numpy()
        if n_bootstrap > 0 and event_ids.size > 1:
            for _ in range(int(n_bootstrap)):
                sample = rng.choice(event_ids, size=event_ids.size, replace=True)
                boot = pd.concat([grp[grp["event_id"] == sid] for sid in sample], ignore_index=True)
                bv = boot["profile_value"].to_numpy(dtype=float)
                bt = boot["template"].to_numpy(dtype=float)
                bd = float(np.dot(bt, bt))
                if bd > 0:
                    boots.append(float(np.dot(bv, bt) / bd))
        jack = []
        if event_ids.size > 2:
            for sid in event_ids:
                j = grp[grp["event_id"] != sid]
                jv = j["profile_value"].to_numpy(dtype=float)
                jt = j["template"].to_numpy(dtype=float)
                jd = float(np.dot(jt, jt))
                if jd > 0:
                    jack.append(float(np.dot(jv, jt) / jd))
        summary_rows.append(
            {
                **meta,
                "n_events": int(event_ids.size),
                "stacked_amplitude": amp,
                "stacked_snr": snr,
                "bootstrap_std": float(np.std(boots, ddof=1)) if len(boots) > 1 else np.nan,
                "jackknife_std": float(np.std(jack, ddof=1)) if len(jack) > 1 else np.nan,
            }
        )
    return add_frequency_mhz_column(stacked), add_frequency_mhz_column(pd.DataFrame.from_records(summary_rows))
