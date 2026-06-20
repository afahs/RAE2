"""Detection scoring against empirical controls and validation metadata."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .constants import ANTENNA_METADATA, add_frequency_mhz_column, frequency_mhz_for_band
from .detection import StepFitConfig, run_matched_filter, run_stepfit_detections
from .util import datetime_ns


@dataclass(frozen=True)
class ScoreConfig:
    window_seconds: float = 600.0
    baseline_order: int = 1
    min_samples_per_side: int = 4
    smooth_seconds: float = 0.0
    timing_grid_seconds: tuple[float, ...] = (-60.0, 0.0, 60.0)
    null_percentile_threshold: float = 99.0
    min_abs_snr: float = 3.0
    clean_fraction_threshold: float = 0.8
    empirical_control_types: tuple[str, ...] = ("randomized_time",)


def antenna_metadata_frame(antennas: pd.Series) -> pd.DataFrame:
    rows = []
    for antenna in antennas.astype(str):
        meta = ANTENNA_METADATA.get(
            antenna,
            {
                "receiver": "unknown",
                "moon_pointing": "unknown",
                "expected_directionality": "unknown antenna orientation",
            },
        )
        rows.append({"antenna": antenna, **meta})
    return pd.DataFrame.from_records(rows)


def _empirical_pvalue(value: float, null_values: np.ndarray) -> float:
    if not np.isfinite(value):
        return np.nan
    null = np.asarray(null_values, dtype=float)
    null = np.abs(null[np.isfinite(null)])
    if null.size == 0:
        return np.nan
    return float((1 + np.count_nonzero(null >= abs(value))) / (null.size + 1))


def _quality_fraction(clean_df: pd.DataFrame, event: pd.Series, window_seconds: float) -> tuple[float, int]:
    group = clean_df
    if "frequency_band" in event and pd.notna(event["frequency_band"]):
        group = group[group["frequency_band"] == event["frequency_band"]]
    if "antenna" in event and pd.notna(event["antenna"]):
        group = group[group["antenna"] == event["antenna"]]
    if group.empty:
        return np.nan, 0
    event_time = pd.Timestamp(event["predicted_event_time"])
    dt = np.abs((pd.DatetimeIndex(group["time"]) - event_time).total_seconds())
    in_window = dt <= float(window_seconds)
    n = int(np.count_nonzero(in_window))
    if n == 0:
        return np.nan, 0
    if "is_valid" not in group.columns:
        return 1.0, n
    return float(np.mean(group.loc[in_window, "is_valid"].to_numpy(dtype=bool))), n


def _quality_fraction_from_groups(groups: dict[tuple[object, object], pd.DataFrame], event: pd.Series, window_seconds: float) -> tuple[float, int]:
    freq = event.get("frequency_band")
    antenna = event.get("antenna")
    group = groups.get((freq, antenna))
    if group is None or group.empty:
        return np.nan, 0
    t_ns = datetime_ns(group["time"])
    event_ns = pd.Timestamp(event["predicted_event_time"]).value
    half_ns = int(float(window_seconds) * 1e9)
    lo = int(np.searchsorted(t_ns, event_ns - half_ns, side="left"))
    hi = int(np.searchsorted(t_ns, event_ns + half_ns, side="right"))
    if hi <= lo:
        return np.nan, 0
    local = group.iloc[lo:hi]
    if "is_valid" not in local.columns:
        return 1.0, int(len(local))
    return float(np.mean(local["is_valid"].to_numpy(dtype=bool))), int(len(local))


def _merge_real_methods(step: pd.DataFrame, matched: pd.DataFrame) -> pd.DataFrame:
    keys = ["source_name", "event_type", "predicted_event_time", "frequency_band", "antenna"]
    left = step.copy()
    right = matched.copy()
    for df in [left, right]:
        df["predicted_event_time"] = pd.to_datetime(df["predicted_event_time"]).astype(str)
    cols = keys + ["amplitude", "uncertainty", "detection_snr", "timing_offset_sec", "n_used", "n_pre", "n_post"]
    out = left[[c for c in cols if c in left.columns]].merge(
        right[[c for c in keys + ["matched_amp", "matched_snr", "matched_n"] if c in right.columns]],
        on=keys,
        how="outer",
    )
    return out


def _null_stats(null_step: pd.DataFrame, null_matched: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, df, snr_col in [
        ("stepfit", null_step, "detection_snr"),
        ("matched_filter", null_matched, "matched_snr"),
    ]:
        if df.empty or snr_col not in df.columns:
            continue
        for (source, freq, antenna), group in df.groupby(["source_name", "frequency_band", "antenna"], dropna=False):
            vals = group[snr_col].to_numpy(dtype=float)
            vals = np.abs(vals[np.isfinite(vals)])
            rows.append(
                {
                    "source_name": source,
                    "frequency_band": freq,
                    "frequency_mhz": frequency_mhz_for_band(freq),
                    "antenna": antenna,
                    "method": method,
                    "n_null": int(vals.size),
                    "null_abs_snr_p95": float(np.percentile(vals, 95)) if vals.size else np.nan,
                    "null_abs_snr_p99": float(np.percentile(vals, 99)) if vals.size else np.nan,
                    "null_abs_snr_max": float(np.max(vals)) if vals.size else np.nan,
                }
            )
    return pd.DataFrame.from_records(rows)


def _add_multiband_support(scored: pd.DataFrame, snr_threshold: float) -> pd.DataFrame:
    out = scored.copy()
    pass_mask = (out["stepfit_empirical_p"] <= 0.05) | (out["matched_empirical_p"] <= 0.05)
    pass_mask &= (out[["detection_snr", "matched_snr"]].abs().max(axis=1) >= float(snr_threshold))
    out["_passes_support"] = pass_mask
    rows = []
    for (source, event_type, event_time), group in out.groupby(["source_name", "event_type", "predicted_event_time"], dropna=False):
        support = group[group["_passes_support"]]
        rows.append(
            {
                "source_name": source,
                "event_type": event_type,
                "predicted_event_time": event_time,
                "supporting_frequency_count": int(support["frequency_band"].nunique()),
                "supporting_antenna_count": int(support["antenna"].nunique()),
                "supporting_channel_count": int(len(support)),
            }
        )
    support_df = pd.DataFrame.from_records(rows)
    out = out.merge(support_df, on=["source_name", "event_type", "predicted_event_time"], how="left")
    return out.drop(columns=["_passes_support"])


def _merge_injection_summary(scored: pd.DataFrame, injection_grid: pd.DataFrame | None) -> pd.DataFrame:
    if injection_grid is None or injection_grid.empty:
        scored["injection_n_recovered_max"] = np.nan
        scored["injection_best_abs_median_snr"] = np.nan
        return scored
    rows = []
    for (freq, antenna), group in injection_grid.groupby(["frequency_band", "antenna"], dropna=False):
        rows.append(
            {
                "frequency_band": freq,
                "antenna": antenna,
                "injection_n_recovered_max": float(group["n_recovered"].max()) if "n_recovered" in group else np.nan,
                "injection_best_abs_median_snr": float(group["median_snr"].abs().max()) if "median_snr" in group else np.nan,
            }
        )
    return scored.merge(pd.DataFrame.from_records(rows), on=["frequency_band", "antenna"], how="left")


def grade_detection(row: pd.Series, cfg: ScoreConfig) -> str:
    best_p = np.nanmin([row.get("stepfit_empirical_p", np.nan), row.get("matched_empirical_p", np.nan)])
    best_snr = np.nanmax([abs(row.get("detection_snr", np.nan)), abs(row.get("matched_snr", np.nan))])
    method_agree = bool(row.get("method_sign_agreement", False))
    enough_support = int(row.get("supporting_frequency_count", 0)) >= 2 or int(row.get("supporting_antenna_count", 0)) >= 2
    clean = row.get("quality_clean_fraction", np.nan)
    clean_ok = np.isfinite(clean) and clean >= cfg.clean_fraction_threshold
    if np.isfinite(best_p) and best_p <= 0.01 and best_snr >= 5.0 and method_agree and enough_support and clean_ok:
        return "strong_control_validated"
    if np.isfinite(best_p) and best_p <= 0.05 and best_snr >= cfg.min_abs_snr and clean_ok:
        return "candidate_needs_review"
    if not clean_ok:
        return "quality_limited"
    return "not_significant_against_controls"


def score_detections(
    clean_df: pd.DataFrame,
    real_step: pd.DataFrame,
    real_matched: pd.DataFrame,
    control_events: pd.DataFrame,
    injection_grid: pd.DataFrame | None = None,
    config: ScoreConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = config or ScoreConfig()
    empirical_controls = control_events.copy()
    if "control_type" in empirical_controls.columns and cfg.empirical_control_types:
        empirical_controls = empirical_controls[
            empirical_controls["control_type"].astype(str).isin(set(cfg.empirical_control_types))
        ].reset_index(drop=True)
    step_cfg = StepFitConfig(
        window_seconds=cfg.window_seconds,
        baseline_order=cfg.baseline_order,
        min_samples_per_side=cfg.min_samples_per_side,
        smooth_seconds=cfg.smooth_seconds,
        timing_grid_seconds=cfg.timing_grid_seconds,
    )
    null_step = run_stepfit_detections(clean_df, empirical_controls, step_cfg)
    null_matched = run_matched_filter(
        clean_df,
        empirical_controls,
        window_seconds=cfg.window_seconds,
        baseline_order=cfg.baseline_order,
    )
    null_summary = _null_stats(null_step, null_matched)
    scored = _merge_real_methods(real_step, real_matched)
    if scored.empty:
        return scored, null_summary, null_step, null_matched

    for col in ["frequency_band"]:
        if col in scored.columns:
            scored[col] = pd.to_numeric(scored[col], errors="coerce")
    scored["antenna"] = scored["antenna"].astype(str)
    scored = add_frequency_mhz_column(scored)
    scored = scored.merge(antenna_metadata_frame(scored["antenna"]).drop_duplicates("antenna"), on="antenna", how="left")

    step_p = []
    matched_p = []
    for _, row in scored.iterrows():
        mask_step = (
            (null_step.get("source_name") == row["source_name"])
            & (pd.to_numeric(null_step.get("frequency_band"), errors="coerce") == row["frequency_band"])
            & (null_step.get("antenna").astype(str) == row["antenna"])
        ) if not null_step.empty else np.array([], dtype=bool)
        mask_matched = (
            (null_matched.get("source_name") == row["source_name"])
            & (pd.to_numeric(null_matched.get("frequency_band"), errors="coerce") == row["frequency_band"])
            & (null_matched.get("antenna").astype(str) == row["antenna"])
        ) if not null_matched.empty else np.array([], dtype=bool)
        step_p.append(_empirical_pvalue(row.get("detection_snr", np.nan), null_step.loc[mask_step, "detection_snr"].to_numpy(dtype=float) if len(mask_step) else np.array([])))
        matched_p.append(_empirical_pvalue(row.get("matched_snr", np.nan), null_matched.loc[mask_matched, "matched_snr"].to_numpy(dtype=float) if len(mask_matched) else np.array([])))
    scored["stepfit_empirical_p"] = step_p
    scored["matched_empirical_p"] = matched_p
    scored["best_empirical_p"] = scored[["stepfit_empirical_p", "matched_empirical_p"]].min(axis=1)
    scored["best_abs_snr"] = scored[["detection_snr", "matched_snr"]].abs().max(axis=1)
    scored["method_sign_agreement"] = np.sign(scored["amplitude"].fillna(0.0)) == np.sign(scored["matched_amp"].fillna(0.0))

    quality = []
    n_quality = []
    quality_groups = {
        (freq, antenna): group.sort_values("time").reset_index(drop=True)
        for (freq, antenna), group in clean_df.groupby(["frequency_band", "antenna"], dropna=False, sort=True)
    }
    for _, row in scored.iterrows():
        frac, n = _quality_fraction_from_groups(quality_groups, row, cfg.window_seconds)
        quality.append(frac)
        n_quality.append(n)
    scored["quality_clean_fraction"] = quality
    scored["quality_window_samples"] = n_quality
    scored = _add_multiband_support(scored, cfg.min_abs_snr)
    scored = _merge_injection_summary(scored, injection_grid)
    scored["detection_grade"] = scored.apply(lambda row: grade_detection(row, cfg), axis=1)
    scored = scored.sort_values(["best_empirical_p", "best_abs_snr"], ascending=[True, False]).reset_index(drop=True)
    return add_frequency_mhz_column(scored), add_frequency_mhz_column(null_summary), add_frequency_mhz_column(null_step), add_frequency_mhz_column(null_matched)
