"""Event profile extraction and stacking."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import add_frequency_mhz_column
from .detection import baseline_matrix, event_template
from .sample_quality import add_strict_valid_column
from .util import datetime_ns, robust_sigma

DEFAULT_BASELINE_MODE = "joint_step_linear"
STACK_PROCESSING_VERSION = "source_preserving_v4_ephemeris"


def _stack_estimate(values: np.ndarray, template: np.ndarray) -> tuple[float, float]:
    """Project the signal and estimate noise from the fitted residual.

    Including the fitted signal in the noise scale makes the score saturate
    as a source brightens. This remains an empirical control-calibrated score,
    not a Gaussian significance for correlated spacecraft samples.
    """
    denominator = float(np.dot(template, template))
    amplitude = float(np.dot(values, template) / denominator) if denominator > 0 else np.nan
    sigma = robust_sigma(values - amplitude * template)
    floor = 64 * np.finfo(float).eps * max(1., float(np.max(np.abs(values)))) if len(values) else np.inf
    snr = float(amplitude * np.sqrt(denominator) / sigma) if np.isfinite(sigma) and sigma > floor else np.nan
    return amplitude, snr


def stack_cache_matches(frame: pd.DataFrame, window_seconds: float,
                        baseline_mode: str = DEFAULT_BASELINE_MODE,
                        sideband_exclusion_seconds: float = 120.0) -> bool:
    """Reject legacy summaries and caches using different processing settings."""
    required = {"stack_processing_version", "baseline_mode", "window_s", "sideband_exclusion_seconds"}
    return bool(not frame.empty and required.issubset(frame.columns)
                and frame["stack_processing_version"].eq(STACK_PROCESSING_VERSION).all()
                and frame["baseline_mode"].eq(baseline_mode).all()
                and np.isclose(pd.to_numeric(frame["window_s"], errors="coerce"), window_seconds).all()
                and np.isclose(pd.to_numeric(frame["sideband_exclusion_seconds"], errors="coerce"), sideband_exclusion_seconds).all())


def detrend_profile_values(
    t_rel_sec: np.ndarray,
    values: np.ndarray,
    event_type: str,
    baseline_mode: str = DEFAULT_BASELINE_MODE,
    sideband_exclusion_seconds: float = 120.0,
    timing_offset_sec: float = 0.0,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Return detrended profile values and template for one event window.

    The default jointly fits the predicted step and a linear baseline, then
    subtracts only the baseline. Explicit sideband_linear/linear_all modes
    retain the legacy attenuation for sensitivity comparisons.
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
    if np.isfinite(sigma) and sigma <= 64 * np.finfo(float).eps * max(1., float(np.max(np.abs(y)))):
        sigma = np.nan  # Do not amplify floating-point residuals of exact fits.
    if normalize and np.isfinite(sigma) and sigma > 0:
        profile = profile / sigma
    return profile, template, float(sigma) if np.isfinite(sigma) else np.nan, int(np.count_nonzero(fit_mask))


def _iter_profile_windows(clean_df, events_df, window_seconds, normalize,
                          baseline_mode, sideband_exclusion_seconds):
    """One extraction/mask/detrending path shared by target and control stacks."""
    if events_df.empty:
        return
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
        yield ev, tr, yd, tmpl, sig, n_baseline


def aligned_profiles(
    clean_df: pd.DataFrame,
    events_df: pd.DataFrame,
    window_seconds: float = 600.0,
    bin_seconds: float = 30.0,
    normalize: bool = True,
    baseline_mode: str = DEFAULT_BASELINE_MODE,
    sideband_exclusion_seconds: float = 120.0,
) -> pd.DataFrame:
    rows = []
    for ev, tr, yd, tmpl, sig, n_baseline in _iter_profile_windows(
        clean_df, events_df, window_seconds, normalize, baseline_mode, sideband_exclusion_seconds
    ):
        tb = np.round(tr / float(bin_seconds)) * float(bin_seconds)
        for trel, tbin, val, tv in zip(tr, tb, yd, tmpl):
            rows.append({
                **ev.to_dict(),
                "t_rel_sec": float(trel),
                "t_bin_sec": float(tbin),
                "profile_value": float(val),
                "template": float(tv),
                "baseline_mode": str(baseline_mode),
                "stack_processing_version": STACK_PROCESSING_VERSION,
                "window_s": float(window_seconds),
                "sideband_exclusion_seconds": float(sideband_exclusion_seconds),
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
                # Profiles are already normalized by each event's residual
                # sigma, so equal weights are the inverse-variance weights.
                "ivar_mean": float(np.mean(vals)),
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
        amp, snr = _stack_estimate(vals, tmpl)
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
                **{c: grp[c].iloc[0] for c in ("baseline_mode", "stack_processing_version", "window_s", "sideband_exclusion_seconds") if c in grp},
                "stacked_amplitude": amp,
                "stacked_snr": snr,
                "bootstrap_std": float(np.std(boots, ddof=1)) if len(boots) > 1 else np.nan,
                "jackknife_std": float(np.std(jack, ddof=1)) if len(jack) > 1 else np.nan,
            }
        )
    return add_frequency_mhz_column(stacked), add_frequency_mhz_column(pd.DataFrame.from_records(summary_rows))


def fast_stack_summary(
    clean_df: pd.DataFrame,
    events_df: pd.DataFrame,
    window_seconds: float,
    by: list[str] | None = None,
    baseline_mode: str = DEFAULT_BASELINE_MODE,
    sideband_exclusion_seconds: float = 120.0,
) -> pd.DataFrame:
    """Compute stack summaries without materializing aligned profile rows.

    Additional grouping columns such as ``control_id`` keep randomized
    realizations separate, yielding the null distribution required for a
    stack-level empirical p-value.
    """
    if events_df.empty:
        return pd.DataFrame()
    group_cols = list(by or ["source_name", "frequency_band", "antenna"])
    input_counts = events_df.groupby(group_cols, dropna=False, sort=False).size().to_dict()
    accum: dict[tuple[object, ...], dict[str, object]] = {}
    for event, _, profile, template, _, _ in _iter_profile_windows(
        clean_df, events_df, window_seconds, True, baseline_mode, sideband_exclusion_seconds
    ):
        key = tuple(event.get(column) for column in group_cols)
        slot = accum.setdefault(key, {"values": [], "templates": [], "events": set()})
        slot["values"].append(profile)
        slot["templates"].append(template)
        slot["events"].add((event.get("event_id"), str(event.get("predicted_event_time"))))

    rows: list[dict[str, object]] = []
    for key, slot in accum.items():
        values = np.concatenate(slot["values"]) if slot["values"] else np.array([], dtype=float)
        template = np.concatenate(slot["templates"]) if slot["templates"] else np.array([], dtype=float)
        amplitude, snr = _stack_estimate(values, template)
        rows.append(
            {
                **dict(zip(group_cols, key)),
                "n_input_events": int(input_counts.get(key if len(key) > 1 else key[0], 0)),
                "baseline_mode": baseline_mode,
                "stack_processing_version": STACK_PROCESSING_VERSION,
                "window_s": float(window_seconds),
                "sideband_exclusion_seconds": float(sideband_exclusion_seconds),
                "n_events": int(len(slot["events"])),
                "stacked_amplitude": amplitude,
                "stacked_snr": snr,
                "bootstrap_std": np.nan,
                "jackknife_std": np.nan,
            }
        )
    return add_frequency_mhz_column(pd.DataFrame.from_records(rows))


def randomized_stack_distribution(
    clean_df: pd.DataFrame,
    events_df: pd.DataFrame,
    window_seconds: float,
    n_random: int = 30,
    seed: int = 12345,
    baseline_mode: str = DEFAULT_BASELINE_MODE,
    sideband_exclusion_seconds: float = 120.0,
) -> pd.DataFrame:
    """Build complete randomized-time stack realizations.

    Every realization starts from the full event table.  This differs from
    event-level scoring ensembles, which may be capped for runtime and must
    not be reused as a source-stack null distribution.
    """
    if events_df.empty or int(n_random) <= 0:
        return pd.DataFrame()
    start_ns = pd.Timestamp(clean_df["time"].min()).value
    end_ns = pd.Timestamp(clean_df["time"].max()).value
    if end_ns <= start_ns:
        return pd.DataFrame()
    rng = np.random.default_rng(int(seed))
    realizations = []
    for control_id in range(int(n_random)):
        randomized = events_df.copy()
        randomized["predicted_event_time"] = pd.to_datetime(
            rng.integers(start_ns, end_ns, size=len(randomized)),
            unit="ns",
        )
        randomized["control_type"] = "randomized_time_stack"
        randomized["control_id"] = int(control_id)
        realizations.append(randomized)
    controls = pd.concat(realizations, ignore_index=True)
    return fast_stack_summary(
        clean_df,
        controls,
        window_seconds,
        by=["source_name", "frequency_band", "antenna", "control_type", "control_id"],
        baseline_mode=baseline_mode,
        sideband_exclusion_seconds=sideband_exclusion_seconds,
    )
