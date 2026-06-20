"""Source-level result aggregation and conservative decision grading."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import add_frequency_mhz_column
from .scoring import _empirical_pvalue
from .util import robust_sigma


def _mad(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    med = np.median(arr)
    return float(np.median(np.abs(arr - med)))


def block_jackknife_from_profiles(profile_df: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Compute date-block jackknife stability for stacked profile groups."""
    if profile_df.empty:
        return pd.DataFrame()
    group_cols = list(by or ["source_name", "frequency_band", "antenna"])
    work = profile_df.copy()
    work["date_block"] = pd.to_datetime(work["predicted_event_time"]).dt.date.astype(str)
    rows: list[dict[str, object]] = []
    for keys, grp in work.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        meta = dict(zip(group_cols, keys))
        vals = grp["profile_value"].to_numpy(dtype=float)
        tmpl = grp["template"].to_numpy(dtype=float)
        denom = float(np.dot(tmpl, tmpl))
        full = float(np.dot(vals, tmpl) / denom) if denom > 0 else np.nan
        estimates = []
        for block, sub in grp.groupby("date_block", sort=True):
            rest = grp[grp["date_block"] != block]
            if rest.empty:
                continue
            rv = rest["profile_value"].to_numpy(dtype=float)
            rt = rest["template"].to_numpy(dtype=float)
            rd = float(np.dot(rt, rt))
            if rd > 0:
                estimates.append(float(np.dot(rv, rt) / rd))
        rows.append(
            {
                **meta,
                "n_date_blocks": int(grp["date_block"].nunique()),
                "block_jackknife_std": float(np.std(estimates, ddof=1)) if len(estimates) > 1 else np.nan,
                "max_block_leverage": float(np.nanmax(np.abs(np.asarray(estimates) - full))) if estimates and np.isfinite(full) else np.nan,
            }
        )
    return add_frequency_mhz_column(pd.DataFrame.from_records(rows))


def offsource_empirical_p(real_score: float, control_scores: pd.Series) -> float:
    return _empirical_pvalue(float(real_score), pd.to_numeric(control_scores, errors="coerce").to_numpy(dtype=float))


def add_offsource_pvalues(source_level: pd.DataFrame, offsource_stack: pd.DataFrame) -> pd.DataFrame:
    """Attach p_offsource from off-source stacked SNRs."""
    out = source_level.copy()
    if offsource_stack is None or offsource_stack.empty or source_level.empty:
        out["offsource_empirical_p"] = np.nan
        return out
    off = offsource_stack.copy()
    parent_col = "parent_source" if "parent_source" in off.columns else None
    pvals = []
    for _, row in out.iterrows():
        sub = off
        if parent_col and pd.notna(row.get("source")):
            sub = sub[sub[parent_col].astype(str) == str(row["source"])]
        for col in ["frequency_band", "antenna"]:
            if col in sub and col in row and pd.notna(row[col]):
                sub = sub[sub[col].astype(str) == str(row[col])]
        controls = sub["stacked_snr"] if "stacked_snr" in sub else pd.Series(dtype=float)
        pvals.append(offsource_empirical_p(row.get("stacked_snr", np.nan), controls))
    out["offsource_empirical_p"] = pvals
    if "decision_grade" in out.columns:
        decisions = out.apply(decision_for_source_row, axis=1, result_type="expand")
        out["decision_grade"] = decisions[0]
        out["decision_reason"] = decisions[1]
    return out


def aggregate_source_level(
    events: pd.DataFrame,
    scored: pd.DataFrame,
    stack_summary: pd.DataFrame,
    quality: pd.DataFrame | None = None,
    block_summary: pd.DataFrame | None = None,
    window_s: float = 600.0,
    event_type: str = "combined",
) -> pd.DataFrame:
    """Aggregate event rows into source/frequency/antenna/window rows."""
    rows: list[dict[str, object]] = []
    if stack_summary.empty:
        return pd.DataFrame()
    scored_work = scored.copy() if scored is not None else pd.DataFrame()
    if not scored_work.empty:
        scored_work["predicted_event_time"] = pd.to_datetime(scored_work["predicted_event_time"])
    events_work = events.copy() if events is not None else pd.DataFrame()
    if not events_work.empty:
        events_work["predicted_event_time"] = pd.to_datetime(events_work["predicted_event_time"])
    for _, st in stack_summary.iterrows():
        source = st["source_name"]
        freq = st["frequency_band"]
        antenna = st["antenna"]
        ev_sub = events_work
        sc_sub = scored_work
        for df_name, df in [("events", ev_sub), ("scored", sc_sub)]:
            pass
        if not ev_sub.empty:
            ev_sub = ev_sub[(ev_sub["source_name"] == source) & (ev_sub["frequency_band"] == freq) & (ev_sub["antenna"] == antenna)]
        if not sc_sub.empty:
            sc_sub = sc_sub[(sc_sub["source_name"] == source) & (sc_sub["frequency_band"] == freq) & (sc_sub["antenna"] == antenna)]
        q_sub = pd.DataFrame()
        if quality is not None and not quality.empty:
            q_sub = quality[(quality["source_name"] == source) & (quality["frequency_band"] == freq) & (quality["antenna"] == antenna)]
        b_sub = pd.DataFrame()
        if block_summary is not None and not block_summary.empty:
            b_sub = block_summary[(block_summary["source_name"] == source) & (block_summary["frequency_band"] == freq) & (block_summary["antenna"] == antenna)]
        timing = pd.to_numeric(sc_sub.get("timing_offset_sec"), errors="coerce") if not sc_sub.empty else pd.Series(dtype=float)
        event_snr = pd.to_numeric(sc_sub.get("detection_snr"), errors="coerce") if not sc_sub.empty else pd.Series(dtype=float)
        best_p = pd.to_numeric(sc_sub.get("best_empirical_p"), errors="coerce") if not sc_sub.empty else pd.Series(dtype=float)
        quality_fail = ""
        if not q_sub.empty and "primary_quality_failure" in q_sub:
            failure_text = q_sub["primary_quality_failure"].astype(str)
            failures = failure_text[failure_text.ne("")]
            quality_fail = str(failures.value_counts().idxmax()) if not failures.empty else ""
        clean_events = 0
        if not q_sub.empty:
            clean_events = int((q_sub.get("primary_quality_failure", "") == "").sum())
        clean_fraction = float(clean_events / len(q_sub)) if len(q_sub) else np.nan
        row = {
            "source": source,
            "source_name": source,
            "frequency_band": freq,
            "frequency_mhz": st.get("frequency_mhz"),
            "antenna": antenna,
            "window_s": float(window_s),
            "event_type": event_type,
            "n_predicted_events": int(ev_sub["predicted_event_time"].nunique()) if not ev_sub.empty else np.nan,
            "n_used_events": int(sc_sub["predicted_event_time"].nunique()) if not sc_sub.empty else int(st.get("n_events", 0)),
            "n_clean_events": clean_events if len(q_sub) else np.nan,
            "clean_fraction": clean_fraction,
            "stacked_snr": float(st.get("stacked_snr", np.nan)),
            "stacked_amplitude": float(st.get("stacked_amplitude", np.nan)),
            "stacked_uncertainty": float(abs(st.get("stacked_amplitude", np.nan)) / abs(st.get("stacked_snr", np.nan))) if np.isfinite(st.get("stacked_snr", np.nan)) and st.get("stacked_snr", 0) != 0 else np.nan,
            "randomized_empirical_p": float(best_p.min()) if best_p.notna().any() else np.nan,
            "offsource_empirical_p": np.nan,
            "sign_fraction_positive": float((event_snr > 0).mean()) if event_snr.notna().any() else np.nan,
            "median_event_snr": float(event_snr.median()) if event_snr.notna().any() else np.nan,
            "median_abs_event_snr": float(event_snr.abs().median()) if event_snr.notna().any() else np.nan,
            "median_timing_offset_s": float(timing.median()) if timing.notna().any() else np.nan,
            "mad_timing_offset_s": _mad(timing) if timing.notna().any() else np.nan,
            "ordinary_bootstrap_std": st.get("bootstrap_std", np.nan),
            "jackknife_std": st.get("jackknife_std", np.nan),
            "block_bootstrap_std": np.nan,
            "block_jackknife_std": b_sub["block_jackknife_std"].iloc[0] if not b_sub.empty and "block_jackknife_std" in b_sub else np.nan,
            "max_block_leverage": b_sub["max_block_leverage"].iloc[0] if not b_sub.empty and "max_block_leverage" in b_sub else np.nan,
            "primary_quality_failure": quality_fail,
        }
        rows.append(row)
    out = add_frequency_mhz_column(pd.DataFrame.from_records(rows))
    if not out.empty:
        decisions = out.apply(decision_for_source_row, axis=1, result_type="expand")
        out["decision_grade"] = decisions[0]
        out["decision_reason"] = decisions[1]
    return out


def decision_for_source_row(row: pd.Series) -> tuple[str, str]:
    source = str(row.get("source", row.get("source_name", ""))).lower()
    abs_stack = abs(float(row.get("stacked_snr", np.nan))) if np.isfinite(row.get("stacked_snr", np.nan)) else np.nan
    rand_p = row.get("randomized_empirical_p", np.nan)
    off_p = row.get("offsource_empirical_p", np.nan)
    clean = row.get("clean_fraction", np.nan)
    med_offset = abs(float(row.get("median_timing_offset_s", np.nan))) if np.isfinite(row.get("median_timing_offset_s", np.nan)) else np.nan
    leverage = row.get("max_block_leverage", np.nan)
    clean_ok = np.isfinite(clean) and clean >= 0.8
    timing_ok = (not np.isfinite(med_offset)) or med_offset <= 60.0
    timing_poor = np.isfinite(med_offset) and med_offset > 120.0
    block_bad = np.isfinite(leverage) and np.isfinite(row.get("stacked_amplitude", np.nan)) and abs(leverage) > max(abs(row.get("stacked_amplitude", 0.0)), 0.1)
    rand_ok = np.isfinite(rand_p) and rand_p <= 0.05
    off_ok = np.isfinite(off_p) and off_p <= 0.05
    earth_quality_ok = np.isfinite(clean) and clean >= 0.5
    if source == "earth" and np.isfinite(abs_stack) and abs_stack >= 10.0 and earth_quality_ok and timing_ok and rand_ok:
        return "A_positive_control", "Earth stack is strong, timing-compatible, and passes randomized controls"
    if not np.isfinite(abs_stack) or abs_stack < 3.0:
        return "F_not_detected", "stacked SNR below current threshold"
    if timing_poor:
        return "E_likely_systematic", "median timing offset is far from prediction"
    if source != "earth" and np.isfinite(off_p) and off_p > 0.1 and abs_stack >= 5.0:
        return "E_likely_systematic", "off-source controls are comparable to real stack"
    if not clean_ok:
        return "D_unresolved", "insufficient clean event fraction"
    if block_bad:
        return "D_unresolved", "stack appears block-dominated"
    if source != "earth" and rand_ok and off_ok and abs_stack >= 5.0 and timing_ok:
        return "B_strong_candidate", "passes randomized/off-source controls with stable timing"
    if abs_stack >= 5.0:
        return "C_candidate", "strong stack but missing one or more major controls"
    return "D_unresolved", "marginal stack or incomplete controls"


def final_source_summary(source_level: pd.DataFrame) -> pd.DataFrame:
    """Collapse source-level rows to one row per source."""
    if source_level.empty:
        return pd.DataFrame()
    rows = []
    for source, grp in source_level.groupby("source", sort=True):
        ranked = grp.assign(abs_stack=grp["stacked_snr"].abs()).sort_values("abs_stack", ascending=False)
        best = ranked.iloc[0]
        strong_rows = grp[grp["decision_grade"].isin(["A_positive_control", "B_strong_candidate"])]
        multi_band = int(grp[grp["stacked_snr"].abs() >= 3.0]["frequency_band"].nunique())
        multi_ant = int(grp[grp["stacked_snr"].abs() >= 3.0]["antenna"].nunique())
        timing = pd.to_numeric(grp["median_timing_offset_s"], errors="coerce")
        clean = pd.to_numeric(grp["clean_fraction"], errors="coerce")
        if str(source).lower() == "earth" and not strong_rows.empty:
            status = "positive_control_confirmed"
        elif (grp["decision_grade"] == "E_likely_systematic").any() and best["abs_stack"] >= 5.0:
            status = "likely_systematic"
        elif (grp["decision_grade"] == "B_strong_candidate").any():
            status = "candidate"
        elif best["abs_stack"] >= 3.0:
            status = "unresolved"
        else:
            status = "not_detected"
        rows.append(
            {
                "source": source,
                "strongest_channel": f"band {int(best['frequency_band'])} / {best.get('frequency_mhz', np.nan):.2f} MHz / {best['antenna']} / {int(best['window_s'])}s",
                "strongest_stacked_snr": float(best["stacked_snr"]),
                "best_randomized_empirical_p": float(pd.to_numeric(grp["randomized_empirical_p"], errors="coerce").min()),
                "best_offsource_empirical_p": float(pd.to_numeric(grp["offsource_empirical_p"], errors="coerce").min()) if pd.to_numeric(grp["offsource_empirical_p"], errors="coerce").notna().any() else np.nan,
                "multi_band_support": multi_band,
                "multi_antenna_support": multi_ant,
                "timing_consistency_summary": f"median={timing.median():.1f}s; mad={_mad(timing):.1f}s" if timing.notna().any() else "unavailable",
                "quality_summary": f"median_clean_fraction={clean.median():.3f}" if clean.notna().any() else "unavailable",
                "final_status": status,
            }
        )
    return pd.DataFrame.from_records(rows)
