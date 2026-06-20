#!/usr/bin/env python
"""Audit whether raw-shape-selected stacks are source-specific or selection-biased.

This script treats the raw occultation-shape selector as a manual-inspection
trigger, not as evidence by itself.  It applies the same lower-V selector to
real Sun/Fornax-A events and to controls that should not know about the true
source: shifted event times, randomized event times, and off-source positions.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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

from rylevonberg.constants import (  # noqa: E402
    EARTH_UNIT_COLUMNS,
    FREQUENCY_MAP_MHZ,
    SPACECRAFT_COLUMNS,
)
from rylevonberg.events import predict_events  # noqa: E402
from rylevonberg.util import datetime_ns, ensure_dir, robust_sigma, software_versions, write_json  # noqa: E402
from scripts.flag_raw_occultation_candidates import _score_at_offset  # noqa: E402


CLEAN_PATH = ROOT / "outputs/control_survey_earth_sun_postnov1974_v2/01_ingest/cleaned_timeseries.csv"
SUN_EVENTS = ROOT / "outputs/pipeline_confidence_audit_v2/sun_audit_input_events.csv"
FORNAX_EVENTS = ROOT / "outputs/control_survey_bright_sources_postnov1974_v1/02_events/predicted_events.csv"
SUN_OFFEPHEM_ALLBAND_EVENTS = (
    ROOT / "outputs/sun_whole_dataset_validation_allbands_mincontrols/02_events/sun_offephemeris_predicted_events.csv"
)
FIXED_OFFSOURCE_CONTROLS = ROOT / "outputs/focused_validation/controls/offsource_controls.csv"
DIFFUSE_AUDIT_ROOT = ROOT / "outputs/beam_weighted_diffuse_sign_audit_v1"
DEFAULT_OUT = ROOT / "outputs/post_selection_bias_audit_sun_fornax_v1"

ANTENNA = "rv2_coarse"
EXPECTED_ANY_PRIORITIES = {"high_priority", "offset_candidate", "weak_predicted_candidate"}
SOURCE_LABELS = {"sun": "Sun", "fornax_a": "Fornax-A"}


@dataclass(frozen=True)
class SourceSpec:
    source: str
    events_path: Path
    offsource_kind: str


SOURCE_SPECS = {
    "sun": SourceSpec("sun", SUN_EVENTS, "moving_offephemeris"),
    "fornax_a": SourceSpec("fornax_a", FORNAX_EVENTS, "fixed_offsource"),
}


def _read(path: Path, **kwargs) -> pd.DataFrame:
    return read_table(path, low_memory=False, **kwargs)


def _source_label(source: str) -> str:
    return SOURCE_LABELS.get(source, source)


def _freq_label(freq: float) -> str:
    return f"{freq:.2f} MHz"


def _load_clean_for_scoring(path: Path, bands: list[int], antenna: str) -> pd.DataFrame:
    cols = ["time", "frequency_band", "frequency_mhz", "antenna", "power", "is_valid"]
    clean = _read(path, usecols=cols, parse_dates=["time"])
    return clean[
        clean["frequency_band"].astype(int).isin({int(b) for b in bands})
        & clean["antenna"].astype(str).eq(str(antenna))
    ].copy()


def _bool_values(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.to_numpy(dtype=bool)
    return series.astype(str).str.lower().isin(["true", "1", "yes"]).to_numpy(dtype=bool)


def _make_clean_groups(clean: pd.DataFrame) -> dict[tuple[int, str], dict[str, np.ndarray]]:
    groups: dict[tuple[int, str], dict[str, np.ndarray]] = {}
    for (band, antenna), grp in clean.groupby(["frequency_band", "antenna"], sort=True):
        g = grp.sort_values("time").reset_index(drop=True)
        power = pd.to_numeric(g["power"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(power) & (power > 0.0)
        if "is_valid" in g.columns:
            valid &= _bool_values(g["is_valid"])
        groups[(int(band), str(antenna))] = {
            "time_ns": datetime_ns(g["time"]),
            "power": power,
            "valid": valid,
        }
    return groups


def _score_event_arrays(
    time_ns: np.ndarray,
    power: np.ndarray,
    valid: np.ndarray,
    event_time: pd.Timestamp,
    event_type: str,
    scan_offsets_s: np.ndarray,
    window_s: float,
    inner_s: float,
    prepost_s: float,
    min_side_samples: int,
) -> dict[str, object]:
    event_ns = pd.Timestamp(event_time).value
    half_ns = int(float(window_s) * 1e9)
    lo = int(np.searchsorted(time_ns, event_ns - half_ns, side="left"))
    hi = int(np.searchsorted(time_ns, event_ns + half_ns, side="right"))
    if hi <= lo:
        return {"usable": False, "primary_failure": "no_samples_in_window"}
    local_t = (time_ns[lo:hi] - event_ns).astype(float) / 1e9
    in_window = np.abs(local_t) <= float(window_s)
    if not np.any(in_window):
        return {"usable": False, "primary_failure": "no_samples_in_window"}
    local_t = local_t[in_window]
    local_power = power[lo:hi][in_window]
    local_valid = valid[lo:hi][in_window]
    total_samples = int(len(local_t))
    valid_samples = int(np.count_nonzero(local_valid))
    if valid_samples == 0:
        return {
            "usable": False,
            "primary_failure": "no_valid_samples",
            "total_samples": total_samples,
            "valid_samples": 0,
            "valid_fraction": 0.0,
        }
    t = local_t[local_valid]
    y = local_power[local_valid]
    predicted = _score_at_offset(t, y, event_type, 0.0, inner_s, prepost_s, min_side_samples)
    scans = [_score_at_offset(t, y, event_type, float(offset), inner_s, prepost_s, min_side_samples) for offset in scan_offsets_s]
    valid_steps = [s for s in scans if np.isfinite(float(s.get("step_z", np.nan)))]
    best = max(valid_steps, key=lambda s: (float(s["step_z"]), int(s["support_bins"]))) if valid_steps else predicted
    valid_opposite = [s for s in scans if np.isfinite(float(s.get("opposite_step_z", np.nan)))]
    best_opposite = max(valid_opposite, key=lambda s: float(s["opposite_step_z"])) if valid_opposite else predicted

    outer = np.abs(t) >= float(inner_s)
    side_scale = robust_sigma(y[outer] - np.nanmedian(y[outer])) if np.count_nonzero(outer) else np.nan
    if not np.isfinite(side_scale) or side_scale <= 0:
        side_scale = float(np.nanstd(y[outer], ddof=1)) if np.count_nonzero(outer) > 1 else np.nan
    central = np.abs(t) < float(inner_s)
    central_peak_z = np.nan
    if np.count_nonzero(central) > 0 and np.isfinite(side_scale) and side_scale > 0:
        central_peak_z = float(np.nanmax(np.abs(y[central] - np.nanmedian(y[outer]))) / side_scale)

    failure = ""
    if int(predicted["n_pre"]) < min_side_samples:
        failure = "too_few_pre_samples"
    elif int(predicted["n_post"]) < min_side_samples:
        failure = "too_few_post_samples"
    elif valid_samples / total_samples < 0.5:
        failure = "low_valid_fraction"

    return {
        "usable": failure == "",
        "primary_failure": failure,
        "total_samples": total_samples,
        "valid_samples": valid_samples,
        "valid_fraction": float(valid_samples / total_samples),
        "predicted_n_pre": int(predicted["n_pre"]),
        "predicted_n_post": int(predicted["n_post"]),
        "predicted_pre_median": predicted["pre_median"],
        "predicted_post_median": predicted["post_median"],
        "predicted_raw_delta_post_minus_pre": predicted["raw_delta_post_minus_pre"],
        "predicted_signed_delta": predicted["signed_delta"],
        "predicted_step_z": predicted["step_z"],
        "predicted_fractional_signed_delta": predicted["fractional_signed_delta"],
        "predicted_support_bins": int(predicted["support_bins"]),
        "best_offset_s": best["offset_s"],
        "best_abs_offset_s": abs(float(best["offset_s"])) if np.isfinite(float(best["offset_s"])) else np.nan,
        "best_n_pre": int(best["n_pre"]),
        "best_n_post": int(best["n_post"]),
        "best_pre_median": best["pre_median"],
        "best_post_median": best["post_median"],
        "best_raw_delta_post_minus_pre": best["raw_delta_post_minus_pre"],
        "best_signed_delta": best["signed_delta"],
        "best_step_z": best["step_z"],
        "best_fractional_signed_delta": best["fractional_signed_delta"],
        "best_support_bins": int(best["support_bins"]),
        "best_pre_slope_per_min": best["pre_slope_per_min"],
        "best_post_slope_per_min": best["post_slope_per_min"],
        "best_opposite_offset_s": best_opposite["offset_s"],
        "best_opposite_step_z": best_opposite["opposite_step_z"],
        "best_opposite_support_bins": int(best_opposite["support_bins"]),
        "best_opposite_fractional_signed_delta": best_opposite["best_opposite_fractional_signed_delta"]
        if "best_opposite_fractional_signed_delta" in best_opposite
        else best_opposite.get("fractional_signed_delta", np.nan),
        "central_peak_z": central_peak_z,
    }


def _priority_reason_fast(
    row: dict[str, object],
    min_predicted_z: float,
    min_best_z: float,
    max_abs_offset_s: float,
) -> tuple[str, str]:
    failure = str(row.get("primary_failure", "") or "")
    if failure:
        return "unusable", failure
    predicted_z = float(row.get("predicted_step_z", np.nan))
    best_z = float(row.get("best_step_z", np.nan))
    opposite_z = float(row.get("best_opposite_step_z", np.nan))
    offset = float(row.get("best_abs_offset_s", np.nan))
    support = int(row.get("best_support_bins", 0) or 0)
    opposite_support = int(row.get("best_opposite_support_bins", 0) or 0)
    min_support = int(row.get("min_support_bins", 1) or 1)
    if np.isfinite(opposite_z) and opposite_z >= min_best_z and opposite_support >= min_support and (
        not np.isfinite(best_z) or opposite_z > best_z
    ):
        return "anti_template_review", f"opposite-sign raw step stronger than expected sign; opposite_z={opposite_z:.2f}"
    if (
        np.isfinite(predicted_z)
        and np.isfinite(best_z)
        and predicted_z >= min_predicted_z
        and best_z >= min_best_z
        and np.isfinite(offset)
        and offset <= max_abs_offset_s
        and support >= min_support
    ):
        return "high_priority", f"expected-sign raw step near prediction; predicted_z={predicted_z:.2f}, best_z={best_z:.2f}, offset={offset:.0f}s"
    if np.isfinite(best_z) and best_z >= min_best_z and np.isfinite(offset) and offset <= max_abs_offset_s and support >= min_support:
        return "offset_candidate", f"expected-sign step found near event but weaker at exact prediction; best_z={best_z:.2f}, offset={offset:.0f}s"
    if np.isfinite(predicted_z) and predicted_z >= min_predicted_z:
        return "weak_predicted_candidate", f"expected-sign raw step at predicted time but weak scan support; predicted_z={predicted_z:.2f}"
    return "not_flagged", "no strong raw pre/post change by configured thresholds"


def _load_clean_for_prediction(path: Path) -> pd.DataFrame:
    cols = ["time", *SPACECRAFT_COLUMNS, *EARTH_UNIT_COLUMNS]
    geom = _read(path, usecols=cols, parse_dates=["time"])
    return geom.drop_duplicates("time").sort_values("time").reset_index(drop=True)


def _load_real_events(spec: SourceSpec, antenna: str, start_date: str | None) -> pd.DataFrame:
    events = _read(spec.events_path, parse_dates=["predicted_event_time"])
    out = events[
        events["source_name"].astype(str).str.lower().eq(spec.source)
        & events["antenna"].astype(str).eq(str(antenna))
    ].copy()
    if start_date:
        out = out[out["predicted_event_time"] >= pd.Timestamp(start_date)].copy()
    if out.empty:
        raise SystemExit(f"No real {spec.source} events found in {spec.events_path}")
    out["source_name"] = spec.source
    return out.reset_index(drop=True)


def _unique_event_time_map(events: pd.DataFrame) -> pd.DataFrame:
    keys = ["event_id", "event_type", "predicted_event_time"]
    return events[keys].drop_duplicates().reset_index(drop=True)


def _assign_unique_ids(events: pd.DataFrame, start: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = events.reset_index(drop=True).copy()
    out["original_event_id"] = out["event_id"]
    out["event_id"] = np.arange(start, start + len(out), dtype=int)
    meta_cols = [
        c
        for c in [
            "event_id",
            "original_event_id",
            "run_role",
            "control_family",
            "control_id",
            "control_name",
            "control_type",
            "offset_deg",
            "time_shift_s",
            "random_realization",
        ]
        if c in out.columns
    ]
    return out, out[meta_cols].drop_duplicates("event_id")


def _score_event_table(
    clean_groups: dict[tuple[int, str], dict[str, np.ndarray]],
    events: pd.DataFrame,
    source: str,
    window_s: float,
    prepost_s: float,
    inner_s: float,
    scan_radius_s: float,
    scan_step_s: float,
    min_side_samples: int,
    min_predicted_z: float,
    min_best_z: float,
    max_abs_offset_s: float,
    min_support_bins: int,
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    work = events[
        events["source_name"].astype(str).str.lower().eq(source.lower())
        & events["antenna"].astype(str).eq(ANTENNA)
    ].copy()
    if work.empty:
        return pd.DataFrame()
    offsets = np.arange(-float(scan_radius_s), float(scan_radius_s) + 0.5 * float(scan_step_s), float(scan_step_s))
    rows: list[dict[str, object]] = []
    for ev in work.sort_values(["predicted_event_time", "frequency_mhz", "antenna"]).itertuples(index=False):
        band = int(ev.frequency_band)
        antenna = str(ev.antenna)
        event_type = str(ev.event_type)
        payload = clean_groups.get((band, antenna))
        if payload is None:
            metrics = {"usable": False, "primary_failure": "no_clean_channel_group"}
        else:
            metrics = _score_event_arrays(
                payload["time_ns"],
                payload["power"],
                payload["valid"],
                pd.Timestamp(ev.predicted_event_time),
                event_type,
                offsets,
                window_s,
                inner_s,
                prepost_s,
                min_side_samples,
            )
        row = {
            "source_name": source.lower(),
            "event_id": getattr(ev, "event_id"),
            "event_type": event_type,
            "predicted_event_time": getattr(ev, "predicted_event_time"),
            "frequency_band": band,
            "frequency_mhz": float(ev.frequency_mhz),
            "antenna": antenna,
        }
        row.update(metrics)
        row["min_support_bins"] = int(min_support_bins)
        priority, reason = _priority_reason_fast(row, min_predicted_z, min_best_z, max_abs_offset_s)
        row["manual_review_priority"] = priority
        row["manual_review_reason"] = reason
        rows.append(row)
    return pd.DataFrame(rows)


def _score_with_metadata(
    clean_groups: dict[tuple[int, str], dict[str, np.ndarray]],
    events: pd.DataFrame,
    source: str,
    metadata_cols: list[str],
    score_kwargs: dict[str, object],
    start_event_id: int,
) -> tuple[pd.DataFrame, int]:
    if events.empty:
        return pd.DataFrame(), start_event_id
    work, meta = _assign_unique_ids(events, start=start_event_id)
    scored = _score_event_table(clean_groups, work, source, **score_kwargs)
    if scored.empty:
        return scored, int(work["event_id"].max()) + 1
    keep_meta = [c for c in ["event_id", *metadata_cols] if c in meta.columns]
    scored = scored.merge(meta[keep_meta], on="event_id", how="left")
    return scored, int(work["event_id"].max()) + 1


def _make_time_shift_controls(real_events: pd.DataFrame, source: str, shifts_s: list[float]) -> pd.DataFrame:
    tables = []
    for shift_s in shifts_s:
        shifted = real_events.copy()
        shifted["predicted_event_time"] = shifted["predicted_event_time"] + pd.to_timedelta(float(shift_s), unit="s")
        shifted["source_name"] = source
        shifted["control_family"] = "time_shift"
        shifted["control_id"] = f"shift_{int(shift_s):+d}s"
        shifted["time_shift_s"] = float(shift_s)
        shifted["control_name"] = shifted["control_id"]
        shifted["control_type"] = "temporal_offset"
        tables.append(shifted)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def _make_random_time_controls(
    real_events: pd.DataFrame,
    source: str,
    clean: pd.DataFrame,
    n_random: int,
    seed: int,
    window_s: float,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    unique_times = pd.Series(pd.to_datetime(clean["time"]).drop_duplicates().sort_values().to_numpy())
    lo = unique_times.min() + pd.to_timedelta(float(window_s), unit="s")
    hi = unique_times.max() - pd.to_timedelta(float(window_s), unit="s")
    choices = unique_times[(unique_times >= lo) & (unique_times <= hi)]
    if choices.empty:
        raise SystemExit("No clean times available for randomized controls after edge trimming.")
    event_keys = _unique_event_time_map(real_events)
    tables = []
    for i in range(int(n_random)):
        mapping = event_keys.copy()
        mapping["randomized_event_time"] = rng.choice(choices.to_numpy(), size=len(mapping), replace=True)
        randomized = real_events.merge(event_keys.assign(_event_key=np.arange(len(event_keys))), on=["event_id", "event_type", "predicted_event_time"], how="left")
        randomized = randomized.merge(
            mapping.assign(_event_key=np.arange(len(mapping)))[["_event_key", "randomized_event_time"]],
            on="_event_key",
            how="left",
        )
        randomized["predicted_event_time"] = pd.to_datetime(randomized["randomized_event_time"])
        randomized = randomized.drop(columns=["_event_key", "randomized_event_time"])
        randomized["source_name"] = source
        randomized["control_family"] = "randomized_time"
        randomized["control_id"] = f"random_{i:03d}"
        randomized["random_realization"] = i
        randomized["control_name"] = randomized["control_id"]
        randomized["control_type"] = "randomized_event_time"
        tables.append(randomized)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def _load_sun_offephemeris_controls(start_date: str | None) -> pd.DataFrame:
    controls = _read(SUN_OFFEPHEM_ALLBAND_EVENTS, parse_dates=["predicted_event_time"])
    controls = controls[controls["antenna"].astype(str).eq(ANTENNA)].copy()
    if start_date:
        controls = controls[controls["predicted_event_time"] >= pd.Timestamp(start_date)].copy()
    controls["control_family"] = "offsource"
    controls["control_id"] = controls["control_name"].astype(str)
    controls["source_name"] = "sun"
    return controls.reset_index(drop=True)


def _load_or_predict_fornax_offsource_controls(
    clean_prediction: pd.DataFrame,
    out_dir: Path,
    start_date: str | None,
    prediction_cadence_seconds: float,
    force: bool,
) -> pd.DataFrame:
    controls_dir = ensure_dir(out_dir / "controls")
    path = controls_dir / "fornax_a_offsource_predicted_events.csv"
    if path.exists() and not force:
        events = _read(path, parse_dates=["predicted_event_time"])
    else:
        controls = _read(FIXED_OFFSOURCE_CONTROLS)
        controls = controls[controls["parent_source"].astype(str).str.lower().eq("fornax_a")].copy()
        if controls.empty:
            raise SystemExit("No Fornax-A off-source controls found.")
        events, states = predict_events(
            clean_prediction,
            controls,
            target_frame="fk4",
            equinox="B1950",
            ephemeris="builtin",
            max_gap_seconds=600.0,
            prediction_cadence_seconds=float(prediction_cadence_seconds),
            frequencies=sorted(FREQUENCY_MAP_MHZ),
            antennas=[ANTENNA],
        )
        events = events.merge(
            controls[["source_name", "parent_source", "control_name", "control_type", "offset_deg", "notes"]],
            on="source_name",
            how="left",
        )
        events.to_csv(path, index=False)
        states.to_csv(controls_dir / "fornax_a_offsource_limb_visibility_states.csv", index=False)
    events = events[events["antenna"].astype(str).eq(ANTENNA)].copy()
    if start_date:
        events = events[events["predicted_event_time"] >= pd.Timestamp(start_date)].copy()
    events["control_family"] = "offsource"
    events["control_id"] = events["control_name"].astype(str)
    events["source_name"] = "fornax_a"
    return events.reset_index(drop=True)


def _add_selection_flags(scored: pd.DataFrame) -> pd.DataFrame:
    out = scored.copy()
    priority = out.get("manual_review_priority", pd.Series(index=out.index, dtype=object)).astype(str)
    usable = out.get("usable", pd.Series(False, index=out.index))
    if usable.dtype != bool:
        usable = usable.astype(str).str.lower().isin(["true", "1", "yes"])
    out["usable_bool"] = usable.to_numpy(dtype=bool)
    out["expected_shape_strict"] = priority.eq("high_priority") & out["usable_bool"]
    out["expected_shape_any"] = priority.isin(EXPECTED_ANY_PRIORITIES) & out["usable_bool"]
    out["anti_template_shape"] = priority.eq("anti_template_review") & out["usable_bool"]
    return out


def _aggregate_scores(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame()
    work = _add_selection_flags(scored)
    group_cols = ["source_name", "run_role", "control_family", "control_id", "frequency_band", "frequency_mhz", "event_type", "antenna"]
    for col in group_cols:
        if col not in work.columns:
            work[col] = ""
    rows = []
    for keys, grp in work.groupby(group_cols, dropna=False, sort=True):
        usable = grp["usable_bool"].to_numpy(dtype=bool)
        n_rows = len(grp)
        n_usable = int(np.count_nonzero(usable))
        denom = max(n_usable, 1)
        selected = grp["expected_shape_any"].to_numpy(dtype=bool)
        strict = grp["expected_shape_strict"].to_numpy(dtype=bool)
        anti = grp["anti_template_shape"].to_numpy(dtype=bool)
        usable_grp = grp.loc[usable].copy()
        selected_grp = grp.loc[selected].copy()
        row = dict(zip(group_cols, keys))
        row.update(
            {
                "n_rows": int(n_rows),
                "n_usable": n_usable,
                "n_expected_shape_strict": int(np.count_nonzero(strict)),
                "n_expected_shape_any": int(np.count_nonzero(selected)),
                "n_anti_template_shape": int(np.count_nonzero(anti)),
                "strict_selection_fraction": float(np.count_nonzero(strict) / denom),
                "any_selection_fraction": float(np.count_nonzero(selected) / denom),
                "anti_template_fraction": float(np.count_nonzero(anti) / denom),
                "median_predicted_step_z": float(pd.to_numeric(usable_grp.get("predicted_step_z"), errors="coerce").median())
                if n_usable
                else np.nan,
                "median_best_step_z": float(pd.to_numeric(usable_grp.get("best_step_z"), errors="coerce").median())
                if n_usable
                else np.nan,
                "median_abs_best_offset_s": float(pd.to_numeric(usable_grp.get("best_abs_offset_s"), errors="coerce").median())
                if n_usable
                else np.nan,
                "selected_median_predicted_step_z": float(
                    pd.to_numeric(selected_grp.get("predicted_step_z"), errors="coerce").median()
                )
                if len(selected_grp)
                else np.nan,
                "selected_median_fractional_signed_delta": float(
                    pd.to_numeric(selected_grp.get("predicted_fractional_signed_delta"), errors="coerce").median()
                )
                if len(selected_grp)
                else np.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _empirical_p_values(agg: pd.DataFrame) -> pd.DataFrame:
    if agg.empty:
        return pd.DataFrame()
    real = agg[agg["run_role"].eq("real")].copy()
    controls = agg[agg["run_role"].eq("control")].copy()
    if real.empty or controls.empty:
        return pd.DataFrame()
    rows = []
    metrics = ["any_selection_fraction", "strict_selection_fraction", "anti_template_fraction"]
    families = ["all_controls", *sorted(controls["control_family"].dropna().astype(str).unique())]
    for _, row in real.iterrows():
        base = {
            "source_name": row["source_name"],
            "frequency_band": row["frequency_band"],
            "frequency_mhz": row["frequency_mhz"],
            "event_type": row["event_type"],
            "antenna": row["antenna"],
        }
        same = controls[
            controls["source_name"].eq(row["source_name"])
            & controls["frequency_band"].eq(row["frequency_band"])
            & controls["event_type"].eq(row["event_type"])
            & controls["antenna"].eq(row["antenna"])
        ].copy()
        for family in families:
            fam = same if family == "all_controls" else same[same["control_family"].astype(str).eq(family)]
            if fam.empty:
                continue
            out = dict(base)
            out["control_family"] = family
            out["n_control_groups"] = int(len(fam))
            for metric in metrics:
                real_val = float(row[metric])
                vals = pd.to_numeric(fam[metric], errors="coerce").dropna().to_numpy(dtype=float)
                if len(vals) == 0:
                    p = np.nan
                    median = np.nan
                    q25 = np.nan
                    q75 = np.nan
                else:
                    p = (1.0 + float(np.count_nonzero(vals >= real_val))) / (1.0 + float(len(vals)))
                    median = float(np.median(vals))
                    q25 = float(np.quantile(vals, 0.25))
                    q75 = float(np.quantile(vals, 0.75))
                out[f"real_{metric}"] = real_val
                out[f"control_median_{metric}"] = median
                out[f"control_q25_{metric}"] = q25
                out[f"control_q75_{metric}"] = q75
                out[f"p_{metric}"] = p
            rows.append(out)
    return pd.DataFrame(rows)


def _split_stability(scored_real: pd.DataFrame) -> pd.DataFrame:
    if scored_real.empty:
        return pd.DataFrame()
    work = _add_selection_flags(scored_real)
    work["predicted_event_time"] = pd.to_datetime(work["predicted_event_time"])
    rows = []
    for source, source_grp in work.groupby("source_name"):
        median_time = source_grp["predicted_event_time"].median()
        split_specs = {
            "date_half": np.where(source_grp["predicted_event_time"] <= median_time, "early", "late"),
            "event_parity": np.where(pd.to_numeric(source_grp["event_id"], errors="coerce").fillna(0).astype(int) % 2 == 0, "even", "odd"),
        }
        for split_kind, values in split_specs.items():
            temp = source_grp.copy()
            temp["split_kind"] = split_kind
            temp["split_value"] = values
            for keys, grp in temp.groupby(["source_name", "frequency_band", "frequency_mhz", "event_type", "antenna", "split_kind", "split_value"], sort=True):
                usable = grp["usable_bool"].to_numpy(dtype=bool)
                denom = max(int(np.count_nonzero(usable)), 1)
                rows.append(
                    {
                        "source_name": keys[0],
                        "frequency_band": keys[1],
                        "frequency_mhz": keys[2],
                        "event_type": keys[3],
                        "antenna": keys[4],
                        "split_kind": keys[5],
                        "split_value": keys[6],
                        "n_usable": int(np.count_nonzero(usable)),
                        "any_selection_fraction": float(np.count_nonzero(grp["expected_shape_any"]) / denom),
                        "strict_selection_fraction": float(np.count_nonzero(grp["expected_shape_strict"]) / denom),
                        "anti_template_fraction": float(np.count_nonzero(grp["anti_template_shape"]) / denom),
                    }
                )
    split = pd.DataFrame(rows)
    if split.empty:
        return split
    spreads = []
    for keys, grp in split.groupby(["source_name", "frequency_band", "frequency_mhz", "event_type", "antenna", "split_kind"], sort=True):
        out = dict(zip(["source_name", "frequency_band", "frequency_mhz", "event_type", "antenna", "split_kind"], keys))
        for metric in ["any_selection_fraction", "strict_selection_fraction", "anti_template_fraction"]:
            vals = pd.to_numeric(grp[metric], errors="coerce").dropna().to_numpy(dtype=float)
            out[f"{metric}_min"] = float(np.min(vals)) if len(vals) else np.nan
            out[f"{metric}_max"] = float(np.max(vals)) if len(vals) else np.nan
            out[f"{metric}_spread"] = float(np.max(vals) - np.min(vals)) if len(vals) else np.nan
        spreads.append(out)
    return pd.DataFrame(spreads)


def _diffuse_class_summary(source: str) -> pd.DataFrame:
    path = DIFFUSE_AUDIT_ROOT / source / f"{source}_beam_weighted_diffuse_row_audit.csv"
    if not path.exists():
        return pd.DataFrame()
    df = _read(path, parse_dates=["predicted_event_time"])
    df = df[df["antenna"].astype(str).eq(ANTENNA)].copy()
    if df.empty:
        return pd.DataFrame()
    df["run_role"] = "real"
    df["control_family"] = "diffuse_model_class"
    df["control_id"] = df.get("model_class", "model_unknown").astype(str)
    flagged = _add_selection_flags(df)
    rows = []
    for keys, grp in flagged.groupby(["source_name", "frequency_band", "frequency_mhz", "event_type", "antenna", "model_class"], sort=True):
        usable = grp["usable_bool"].to_numpy(dtype=bool)
        denom = max(int(np.count_nonzero(usable)), 1)
        rows.append(
            {
                "source_name": keys[0],
                "frequency_band": keys[1],
                "frequency_mhz": keys[2],
                "event_type": keys[3],
                "antenna": keys[4],
                "model_class": keys[5],
                "n_rows": int(len(grp)),
                "n_usable": int(np.count_nonzero(usable)),
                "any_selection_fraction": float(np.count_nonzero(grp["expected_shape_any"]) / denom),
                "strict_selection_fraction": float(np.count_nonzero(grp["expected_shape_strict"]) / denom),
                "anti_template_fraction": float(np.count_nonzero(grp["anti_template_shape"]) / denom),
            }
        )
    return pd.DataFrame(rows)


def _plot_selection_spectrum(source: str, agg: pd.DataFrame, out_dir: Path, metric: str = "any_selection_fraction") -> Path:
    source_agg = agg[agg["source_name"].astype(str).eq(source)].copy()
    event_types = ["disappearance", "reappearance"]
    families = [
        ("time_shift", "#d95f02", "time-shift controls"),
        ("randomized_time", "#7570b3", "randomized-time controls"),
        ("offsource", "#1b9e77", "off-source controls"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7), sharey=True)
    for ax, event_type in zip(axes, event_types):
        real = source_agg[(source_agg["run_role"].eq("real")) & (source_agg["event_type"].eq(event_type))]
        if not real.empty:
            real = real.sort_values("frequency_mhz")
            ax.plot(real["frequency_mhz"], real[metric], "o-", color="black", lw=2.0, label="real source")
        for family, color, label in families:
            fam = source_agg[
                (source_agg["run_role"].eq("control"))
                & (source_agg["control_family"].eq(family))
                & (source_agg["event_type"].eq(event_type))
            ]
            if fam.empty:
                continue
            rows = []
            for freq, grp in fam.groupby("frequency_mhz", sort=True):
                vals = pd.to_numeric(grp[metric], errors="coerce").dropna().to_numpy(dtype=float)
                if len(vals):
                    rows.append(
                        {
                            "frequency_mhz": float(freq),
                            "median": float(np.median(vals)),
                            "q25": float(np.quantile(vals, 0.25)),
                            "q75": float(np.quantile(vals, 0.75)),
                        }
                    )
            if not rows:
                continue
            frame = pd.DataFrame(rows).sort_values("frequency_mhz")
            ax.plot(frame["frequency_mhz"], frame["median"], "o-", color=color, alpha=0.9, label=label)
            ax.fill_between(frame["frequency_mhz"], frame["q25"], frame["q75"], color=color, alpha=0.18, linewidth=0)
        ax.set_title(event_type)
        ax.set_xlabel("frequency (MHz)")
        ax.set_xscale("log")
        ax.set_xticks(list(FREQUENCY_MAP_MHZ.values()))
        ax.set_xticklabels([f"{v:g}" for v in FREQUENCY_MAP_MHZ.values()], rotation=45, ha="right")
        ax.grid(True, color="0.9", linewidth=0.6)
    axes[0].set_ylabel(metric.replace("_", " "))
    axes[1].legend(loc="best", fontsize=8, frameon=False)
    fig.suptitle(f"{_source_label(source)} lower-V expected-shape selection rate versus controls", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = out_dir / f"{source}_{metric}_control_spectrum.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_time_shift_heatmap(source: str, agg: pd.DataFrame, out_dir: Path, metric: str = "any_selection_fraction") -> Path | None:
    source_agg = agg[
        agg["source_name"].astype(str).eq(source)
        & agg["run_role"].eq("control")
        & agg["control_family"].eq("time_shift")
    ].copy()
    if source_agg.empty:
        return None
    event_types = ["disappearance", "reappearance"]
    freqs = sorted(source_agg["frequency_mhz"].dropna().astype(float).unique())
    shifts = sorted(source_agg["control_id"].dropna().astype(str).unique(), key=lambda x: int(x.replace("shift_", "").replace("s", "")))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, event_type in zip(axes, event_types):
        grid = np.full((len(shifts), len(freqs)), np.nan)
        sub = source_agg[source_agg["event_type"].eq(event_type)]
        for i, shift in enumerate(shifts):
            for j, freq in enumerate(freqs):
                vals = pd.to_numeric(
                    sub[sub["control_id"].eq(shift) & sub["frequency_mhz"].eq(freq)][metric],
                    errors="coerce",
                ).dropna()
                if len(vals):
                    grid[i, j] = float(vals.iloc[0])
        im = ax.imshow(grid, aspect="auto", origin="lower", interpolation="nearest", vmin=0, vmax=np.nanmax(grid) if np.isfinite(grid).any() else 1)
        ax.set_title(event_type)
        ax.set_xticks(range(len(freqs)))
        ax.set_xticklabels([f"{f:g}" for f in freqs], rotation=45, ha="right")
        ax.set_yticks(range(len(shifts)))
        ax.set_yticklabels(shifts)
        ax.set_xlabel("frequency (MHz)")
    axes[0].set_ylabel("time-shift control")
    fig.colorbar(im, ax=axes.ravel().tolist(), label=metric.replace("_", " "))
    fig.suptitle(f"{_source_label(source)} lower-V time-shift specificity", y=0.98)
    path = out_dir / f"{source}_{metric}_time_shift_heatmap.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_p_value_heatmap(source: str, pvals: pd.DataFrame, out_dir: Path, metric: str = "any_selection_fraction") -> Path | None:
    sub = pvals[
        pvals["source_name"].astype(str).eq(source)
        & pvals["control_family"].eq("all_controls")
    ].copy()
    if sub.empty:
        return None
    event_types = ["disappearance", "reappearance"]
    freqs = sorted(sub["frequency_mhz"].dropna().astype(float).unique())
    grid = np.full((len(event_types), len(freqs)), np.nan)
    col = f"p_{metric}"
    for i, event_type in enumerate(event_types):
        for j, freq in enumerate(freqs):
            vals = pd.to_numeric(sub[sub["event_type"].eq(event_type) & sub["frequency_mhz"].eq(freq)][col], errors="coerce").dropna()
            if len(vals):
                grid[i, j] = float(vals.iloc[0])
    fig, ax = plt.subplots(figsize=(9, 3.2))
    im = ax.imshow(grid, aspect="auto", origin="lower", interpolation="nearest", vmin=0, vmax=1, cmap="viridis_r")
    ax.set_yticks(range(len(event_types)))
    ax.set_yticklabels(event_types)
    ax.set_xticks(range(len(freqs)))
    ax.set_xticklabels([f"{f:g}" for f in freqs], rotation=45, ha="right")
    ax.set_xlabel("frequency (MHz)")
    ax.set_title(f"{_source_label(source)} empirical p-value: real selection rate versus all controls")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if np.isfinite(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", color="white" if grid[i, j] < 0.45 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label=f"p({metric})")
    fig.tight_layout()
    path = out_dir / f"{source}_{metric}_empirical_p_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_report(
    out_dir: Path,
    config: dict[str, object],
    agg: pd.DataFrame,
    pvals: pd.DataFrame,
    split: pd.DataFrame,
    diffuse: pd.DataFrame,
    plot_paths: list[Path],
) -> None:
    lines = [
        "# Post-Selection Bias Audit: Sun and Fornax-A",
        "",
        "This audit asks whether raw-shape-selected occultation stacks are source-specific, or whether the selection step itself can manufacture source-like stacks from noise/background structure.",
        "",
        "The raw-shape selector is treated as a trigger only. A selected stack is not independent detection evidence unless the real events are selected more often than temporal and off-source controls.",
        "",
        "## Configuration",
        "",
        pd.Series(config).to_string(),
        "",
        "## Controls Applied",
        "",
        "- real events: true predicted occultation times for the source;",
        "- time-shift controls: the same events shifted by fixed offsets, so source geometry is wrong in time;",
        "- randomized-time controls: event times randomized inside the available lower-V time coverage;",
        "- off-source controls: nearby wrong sky tracks or positions, passed through the same prediction/scoring path;",
        "- split-half stability: early/late and even/odd real-event splits;",
        "- diffuse-model class summary: checks whether beam-weighted diffuse-background sign predicts which events pass the raw selector.",
        "",
        "## Interpretation Rule",
        "",
        "The key quantity is the expected-shape selection fraction: the fraction of usable event windows that already show the expected raw pre/post change before stacking. If a real source is not higher than shifted/random/off-source controls, then selecting those windows and stacking them is circular.",
        "",
        "The conditional selected-window amplitudes are saved, but they are intentionally not treated as detection evidence because they are measured after selecting on the same raw shape.",
        "",
        "## Top-Level Selection-Rate Summary",
        "",
    ]
    if agg.empty:
        lines.append("No aggregate rows were produced.")
    else:
        real_summary = (
            agg[agg["run_role"].eq("real")]
            .sort_values(["source_name", "event_type", "frequency_mhz"])
            [
                [
                    "source_name",
                    "frequency_mhz",
                    "event_type",
                    "n_usable",
                    "any_selection_fraction",
                    "strict_selection_fraction",
                    "anti_template_fraction",
                    "median_abs_best_offset_s",
                ]
            ]
        )
        lines.append(real_summary.to_string(index=False))
    lines += [
        "",
        "## Empirical P-Value Summary",
        "",
    ]
    if pvals.empty:
        lines.append("No p-values were produced.")
    else:
        view = pvals[pvals["control_family"].eq("all_controls")].copy()
        view = view.sort_values(["source_name", "event_type", "frequency_mhz"])
        cols = [
            "source_name",
            "frequency_mhz",
            "event_type",
            "n_control_groups",
            "real_any_selection_fraction",
            "control_median_any_selection_fraction",
            "p_any_selection_fraction",
            "real_strict_selection_fraction",
            "control_median_strict_selection_fraction",
            "p_strict_selection_fraction",
        ]
        lines.append(view[cols].to_string(index=False))
    lines += [
        "",
        "## Split-Half Stability",
        "",
    ]
    if split.empty:
        lines.append("No split-half summary was produced.")
    else:
        view = split.sort_values(["source_name", "split_kind", "event_type", "frequency_mhz"])
        cols = [
            "source_name",
            "frequency_mhz",
            "event_type",
            "split_kind",
            "any_selection_fraction_min",
            "any_selection_fraction_max",
            "any_selection_fraction_spread",
        ]
        lines.append(view[cols].to_string(index=False))
    lines += [
        "",
        "## Diffuse-Model Diagnostic",
        "",
    ]
    if diffuse.empty:
        lines.append("No diffuse-model class summary was available.")
    else:
        view = diffuse.sort_values(["source_name", "frequency_mhz", "event_type", "model_class"])
        cols = [
            "source_name",
            "frequency_mhz",
            "event_type",
            "model_class",
            "n_usable",
            "any_selection_fraction",
            "anti_template_fraction",
        ]
        lines.append(view[cols].to_string(index=False))
    lines += [
        "",
        "## Plots",
        "",
    ]
    for path in plot_paths:
        lines.append(f"- `{path.relative_to(out_dir) if path.is_relative_to(out_dir) else path}`")
    lines += [
        "",
        "## Output Tables",
        "",
        "- `post_selection_all_scores.csv`: every real/control scored event row;",
        "- `post_selection_selection_rate_summary.csv`: one row per source/control/frequency/event type;",
        "- `post_selection_empirical_p_values.csv`: empirical p-values comparing real selection rates against controls;",
        "- `post_selection_real_split_stability.csv`: early/late and even/odd stability of real events;",
        "- `post_selection_diffuse_class_summary.csv`: selection rates split by beam-weighted diffuse model class.",
        "",
        "## Practical Conclusion",
        "",
        "Use this audit to decide whether a visually good selected stack is source-specific. If shifted, randomized, or off-source controls produce comparable selection rates, the selected stack should be treated as a biased manual-review product rather than a detection.",
    ]
    (out_dir / "post_selection_bias_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--sources", default="sun,fornax_a")
    parser.add_argument("--start-date", default="1974-11-01")
    parser.add_argument("--window-s", type=float, default=600.0)
    parser.add_argument("--prepost-s", type=float, default=300.0)
    parser.add_argument("--inner-s", type=float, default=30.0)
    parser.add_argument("--scan-radius-s", type=float, default=300.0)
    parser.add_argument("--scan-step-s", type=float, default=60.0)
    parser.add_argument("--min-side-samples", type=int, default=4)
    parser.add_argument("--min-predicted-z", type=float, default=2.0)
    parser.add_argument("--min-best-z", type=float, default=3.0)
    parser.add_argument("--max-abs-offset-s", type=float, default=180.0)
    parser.add_argument("--min-support-bins", type=int, default=2)
    parser.add_argument("--time-shifts-s", default="-1200,-600,-300,300,600,1200")
    parser.add_argument("--n-random", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=20260603)
    parser.add_argument("--prediction-cadence-seconds", type=float, default=300.0)
    parser.add_argument("--force-predict-offsource", action="store_true")
    args = parser.parse_args()

    out_dir = ensure_dir(Path(args.out_dir))
    requested = [s.strip().lower() for s in str(args.sources).split(",") if s.strip()]
    specs = [SOURCE_SPECS[s] for s in requested if s in SOURCE_SPECS]
    missing = sorted(set(requested) - set(SOURCE_SPECS))
    if missing:
        raise SystemExit(f"Unknown source(s): {', '.join(missing)}")

    real_events_by_source = {spec.source: _load_real_events(spec, ANTENNA, args.start_date) for spec in specs}
    all_bands = sorted(
        {
            int(band)
            for events in real_events_by_source.values()
            for band in events["frequency_band"].astype(int).dropna().unique()
        }
    )
    print("Loading lower-V clean time series for scoring...", flush=True)
    clean_score = _load_clean_for_scoring(CLEAN_PATH, all_bands, ANTENNA)
    print(f"Loaded {len(clean_score)} lower-V clean rows; building channel groups...", flush=True)
    clean_groups = _make_clean_groups(clean_score)
    print(f"Built {len(clean_groups)} channel groups.", flush=True)
    clean_predict = None
    score_kwargs = {
        "window_s": float(args.window_s),
        "prepost_s": float(args.prepost_s),
        "inner_s": float(args.inner_s),
        "scan_radius_s": float(args.scan_radius_s),
        "scan_step_s": float(args.scan_step_s),
        "min_side_samples": int(args.min_side_samples),
        "min_predicted_z": float(args.min_predicted_z),
        "min_best_z": float(args.min_best_z),
        "max_abs_offset_s": float(args.max_abs_offset_s),
        "min_support_bins": int(args.min_support_bins),
    }
    shifts = [float(x.strip()) for x in str(args.time_shifts_s).split(",") if x.strip()]

    all_scores = []
    next_event_id = 0
    for spec in specs:
        source = spec.source
        real_events = real_events_by_source[source]
        print(f"Scoring {source}: {len(real_events)} real event rows...", flush=True)

        real_work = real_events.copy()
        real_work["run_role"] = "real"
        real_work["control_family"] = "real"
        real_work["control_id"] = "real"
        scored_real, next_event_id = _score_with_metadata(
            clean_groups,
            real_work,
            source,
            ["run_role", "control_family", "control_id"],
            score_kwargs,
            next_event_id,
        )
        all_scores.append(scored_real)

        shifted = _make_time_shift_controls(real_events, source, shifts)
        shifted["run_role"] = "control"
        print(f"Scoring {source}: {len(shifted)} time-shift control rows...", flush=True)
        scored_shifted, next_event_id = _score_with_metadata(
            clean_groups,
            shifted,
            source,
            ["run_role", "control_family", "control_id", "control_name", "control_type", "time_shift_s"],
            score_kwargs,
            next_event_id,
        )
        all_scores.append(scored_shifted)

        randomized = _make_random_time_controls(
            real_events,
            source,
            clean_score,
            n_random=int(args.n_random),
            seed=int(args.random_seed) + sum((idx + 1) * ord(ch) for idx, ch in enumerate(source)),
            window_s=float(args.window_s),
        )
        randomized["run_role"] = "control"
        print(f"Scoring {source}: {len(randomized)} randomized-time control rows...", flush=True)
        scored_random, next_event_id = _score_with_metadata(
            clean_groups,
            randomized,
            source,
            ["run_role", "control_family", "control_id", "control_name", "control_type", "random_realization"],
            score_kwargs,
            next_event_id,
        )
        all_scores.append(scored_random)

        if spec.offsource_kind == "moving_offephemeris":
            offsource = _load_sun_offephemeris_controls(args.start_date)
        else:
            fornax_cache = out_dir / "controls" / "fornax_a_offsource_predicted_events.csv"
            if clean_predict is None and (bool(args.force_predict_offsource) or not fornax_cache.exists()):
                print("Loading geometry rows for fixed-source off-source prediction...", flush=True)
                clean_predict = _load_clean_for_prediction(CLEAN_PATH)
                print(f"Loaded {len(clean_predict)} unique geometry times.", flush=True)
            offsource = _load_or_predict_fornax_offsource_controls(
                clean_predict,
                out_dir,
                args.start_date,
                float(args.prediction_cadence_seconds),
                bool(args.force_predict_offsource),
            )
        offsource["run_role"] = "control"
        print(f"Scoring {source}: {len(offsource)} off-source control rows...", flush=True)
        scored_off, next_event_id = _score_with_metadata(
            clean_groups,
            offsource,
            source,
            [
                "run_role",
                "control_family",
                "control_id",
                "control_name",
                "control_type",
                "offset_deg",
            ],
            score_kwargs,
            next_event_id,
        )
        all_scores.append(scored_off)

    scores = pd.concat([s for s in all_scores if s is not None and not s.empty], ignore_index=True)
    scores = _add_selection_flags(scores)
    scores.to_csv(out_dir / "post_selection_all_scores.csv", index=False)

    agg = _aggregate_scores(scores)
    agg.to_csv(out_dir / "post_selection_selection_rate_summary.csv", index=False)
    pvals = _empirical_p_values(agg)
    pvals.to_csv(out_dir / "post_selection_empirical_p_values.csv", index=False)
    split = _split_stability(scores[scores["run_role"].eq("real")].copy())
    split.to_csv(out_dir / "post_selection_real_split_stability.csv", index=False)
    diffuse = pd.concat([_diffuse_class_summary(spec.source) for spec in specs], ignore_index=True)
    diffuse.to_csv(out_dir / "post_selection_diffuse_class_summary.csv", index=False)

    plot_paths: list[Path] = []
    for spec in specs:
        plot_paths.append(_plot_selection_spectrum(spec.source, agg, out_dir, metric="any_selection_fraction"))
        plot_paths.append(_plot_selection_spectrum(spec.source, agg, out_dir, metric="strict_selection_fraction"))
        heat = _plot_time_shift_heatmap(spec.source, agg, out_dir, metric="any_selection_fraction")
        if heat is not None:
            plot_paths.append(heat)
        pheat = _plot_p_value_heatmap(spec.source, pvals, out_dir, metric="any_selection_fraction")
        if pheat is not None:
            plot_paths.append(pheat)

    config = {
        "clean": str(CLEAN_PATH),
        "antenna": ANTENNA,
        "sources": ",".join([s.source for s in specs]),
        "start_date": args.start_date,
        "window_s": float(args.window_s),
        "prepost_s": float(args.prepost_s),
        "inner_s": float(args.inner_s),
        "scan_radius_s": float(args.scan_radius_s),
        "scan_step_s": float(args.scan_step_s),
        "min_side_samples": int(args.min_side_samples),
        "min_predicted_z": float(args.min_predicted_z),
        "min_best_z": float(args.min_best_z),
        "max_abs_offset_s": float(args.max_abs_offset_s),
        "min_support_bins": int(args.min_support_bins),
        "time_shifts_s": shifts,
        "n_random": int(args.n_random),
        "random_seed": int(args.random_seed),
        "prediction_cadence_seconds": float(args.prediction_cadence_seconds),
        "software_versions": software_versions(),
    }
    write_json(out_dir / "run_config.json", config)
    _write_report(out_dir, config, agg, pvals, split, diffuse, plot_paths)
    print(f"Wrote post-selection bias audit to {out_dir}")


if __name__ == "__main__":
    main()
