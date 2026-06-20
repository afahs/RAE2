#!/usr/bin/env python
"""Analyze digitized burst-receiver samples against Ryle-Vonberg source events."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rylevonberg.table_io import read_table

ROOT = Path(__file__).resolve().parents[1]
RAE2_V2 = ROOT.parents[0] / "RAE2AgentV2"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(RAE2_V2))

from rae2.burst import (  # noqa: E402
    BURST_COLUMNS,
    _parse_workbook_range,
    _repair_timestamp_year,
    infer_receiver_id,
    read_master_time_bounds,
    resolve_workbook_path,
    source_priority,
)
from rae2.burst_fold import normalize_burst_values  # noqa: E402
from rylevonberg.detection import baseline_matrix, event_template  # noqa: E402
from rylevonberg.events import predict_events  # noqa: E402
from rylevonberg.util import datetime_ns, ensure_dir, robust_sigma  # noqa: E402


DEFAULT_BURST = Path(os.environ.get("RAE2_BURST_RECEIVER_CSV", "data/burstReceiverCSV/burst_receiver_2.csv"))
DEFAULT_MASTER = Path(os.environ.get("RAE2_MASTER_CSV", "data/interpolatedRAE2MasterFile.csv"))


def _log(message: str) -> None:
    print(f"[burst-source] {message}", flush=True)


def _robust_z(values: pd.Series) -> pd.Series:
    arr = pd.to_numeric(values, errors="coerce").astype(float)
    med = float(np.nanmedian(arr))
    sig = float(robust_sigma(arr.to_numpy(dtype=float)))
    if not np.isfinite(sig) or sig <= 0:
        return arr - med
    return (arr - med) / sig


def _normalize_combined_burst_file(csv_path: Path, master_start: pd.Timestamp, master_end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = read_table(csv_path, usecols=lambda column: column in set(BURST_COLUMNS), low_memory=False)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(
            [
                {
                    "csv_path": str(csv_path),
                    "source_priority": source_priority(csv_path),
                    "rows_total": 0,
                    "rows_valid": 0,
                    "rows_repaired": 0,
                    "rows_dropped": 0,
                    "rows_outside_master": 0,
                    "rows_outside_workbook": 0,
                }
            ]
        )

    df["source_file"] = str(csv_path)
    df["source_priority"] = source_priority(csv_path)
    receiver_series = df["Receiver_ID"] if "Receiver_ID" in df.columns else pd.Series(np.nan, index=df.index)
    receiver_numeric = pd.to_numeric(receiver_series, errors="coerce")
    inferred_receiver = infer_receiver_id(df["Receiver"] if "Receiver" in df.columns else pd.Series("", index=df.index))
    df["receiver_id"] = receiver_numeric.fillna(inferred_receiver).astype("Int64")
    df["channel_freq_mhz"] = pd.to_numeric(df.get("Channel_Freq", pd.Series(np.nan, index=df.index)), errors="coerce")
    df["value_log_cal_temp"] = pd.to_numeric(df.get("Log_Cal_Temp", pd.Series(np.nan, index=df.index)), errors="coerce")
    df["point_confidence"] = pd.to_numeric(df.get("Point_Confidence", pd.Series(1.0, index=df.index)), errors="coerce").fillna(1.0)
    df["time_raw"] = pd.to_datetime(df.get("Actual_Time", pd.Series(pd.NaT, index=df.index)), errors="coerce")

    workbook_cache = {}
    workbook_parse_failures = 0
    repaired_chunks = []
    group_key = "XLSX_Report_Path" if "XLSX_Report_Path" in df.columns else None
    groups = df.groupby(group_key, dropna=False, sort=False) if group_key else [(None, df)]
    for workbook_path, group in groups:
        workbook_range = None
        if pd.notna(workbook_path) and str(workbook_path):
            key = str(workbook_path)
            workbook_range = workbook_cache.get(key)
            if key not in workbook_cache:
                try:
                    workbook_range = _parse_workbook_range(resolve_workbook_path(key))
                except Exception:
                    workbook_parse_failures += 1
                    workbook_range = None
                workbook_cache[key] = workbook_range
        repaired_times = []
        repair_status = []
        for ts in group["time_raw"]:
            repaired, status = _repair_timestamp_year(pd.Timestamp(ts) if pd.notna(ts) else pd.NaT, workbook_range)
            repaired_times.append(repaired)
            repair_status.append(status)
        chunk = group.copy()
        chunk["time"] = repaired_times
        chunk["repair_status"] = repair_status
        repaired_chunks.append(chunk)

    df = pd.concat(repaired_chunks, ignore_index=True)
    master_mask = df["time"].notna() & (df["time"] >= master_start) & (df["time"] <= master_end)
    workbook_fail_mask = df["repair_status"].eq("outside_workbook")
    df["drop_reason"] = np.where(workbook_fail_mask, "outside_workbook", "")
    df.loc[df["time"].notna() & ~master_mask & ~workbook_fail_mask, "drop_reason"] = "outside_master"
    df.loc[df["time"].isna() & ~workbook_fail_mask, "drop_reason"] = "missing_time"

    keep_mask = (
        df["drop_reason"].eq("")
        & df["receiver_id"].notna()
        & df["channel_freq_mhz"].notna()
        & df["value_log_cal_temp"].notna()
    )
    valid = df.loc[keep_mask].copy()
    valid["receiver_id"] = valid["receiver_id"].astype(int)
    valid["repair_applied"] = valid["repair_status"].isin({"repaired_year", "repaired_year_ambiguous"})
    valid["source_group"] = valid["Source_Path"].fillna("").astype(str) if "Source_Path" in valid.columns else ""
    valid.loc[valid["source_group"].eq(""), "source_group"] = valid["source_file"]

    audit = pd.DataFrame(
        [
            {
                "csv_path": str(csv_path),
                "source_priority": source_priority(csv_path),
                "rows_total": int(len(df)),
                "rows_valid": int(len(valid)),
                "rows_repaired": int(valid["repair_applied"].sum()),
                "rows_dropped": int((~keep_mask).sum()),
                "rows_outside_master": int((df["drop_reason"] == "outside_master").sum()),
                "rows_outside_workbook": int((df["drop_reason"] == "outside_workbook").sum()),
                "workbooks_seen": int(len(workbook_cache)),
                "workbooks_parse_failed": int(workbook_parse_failures),
            }
        ]
    )
    keep_columns = [
        "time",
        "receiver_id",
        "channel_freq_mhz",
        "value_log_cal_temp",
        "point_confidence",
        "Receiver",
        "Date",
        "MP_ID",
        "Frame_ID",
        "Source_Path",
        "XLSX_Report_Path",
        "Issues",
        "source_file",
        "source_group",
        "source_priority",
        "repair_status",
        "repair_applied",
    ]
    return valid[[column for column in keep_columns if column in valid.columns]].reset_index(drop=True), audit


def _read_or_build_normalized(args: argparse.Namespace, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    normalized_path = out_dir / "normalized_burst_samples.pkl"
    audit_path = out_dir / "normalization_audit.csv"
    if normalized_path.exists() and audit_path.exists() and not args.force:
        _log(f"reading cached normalized burst samples: {normalized_path}")
        return pd.read_pickle(normalized_path), read_table(audit_path)

    _log("normalizing burst CSV and repairing workbook-bounded timestamps")
    master_start, master_end = read_master_time_bounds(args.master_csv)
    burst, audit = _normalize_combined_burst_file(args.burst_csv, master_start, master_end)
    if burst.empty:
        raise RuntimeError("No usable burst rows after timestamp repair and master-range filtering")
    burst["time"] = pd.to_datetime(burst["time"])
    burst = normalize_burst_values(burst, "value_log_cal_temp")
    burst["global_channel_z"] = burst.groupby(["receiver_id", "channel_freq_mhz"], group_keys=False)[
        "value_log_cal_temp"
    ].apply(_robust_z)
    burst.to_pickle(normalized_path)
    audit.to_csv(audit_path, index=False)
    return burst, audit


def _write_content_stats(burst: pd.DataFrame, audit: pd.DataFrame, out_dir: Path, burst_csv: Path) -> None:
    overview = pd.DataFrame(
        [
            {
                "input_csv": str(burst_csv),
                "usable_rows": int(len(burst)),
                "time_min": burst["time"].min(),
                "time_max": burst["time"].max(),
                "unique_dates": int(burst["time"].dt.date.nunique()),
                "unique_months": int(burst["time"].dt.to_period("M").nunique()),
                "receiver_ids": ",".join(map(str, sorted(burst["receiver_id"].dropna().unique()))),
                "unique_channel_freq_mhz": int(burst["channel_freq_mhz"].nunique()),
                "unique_mp_ids": int(burst["MP_ID"].nunique()),
                "unique_frame_ids": int(burst["Frame_ID"].nunique()),
                "unique_source_paths": int(burst["Source_Path"].nunique()),
                "mean_point_confidence": float(burst["point_confidence"].mean()),
                "median_point_confidence": float(burst["point_confidence"].median()),
                "repair_applied_rows": int(burst["repair_applied"].sum()),
            }
        ]
    )
    overview.to_csv(out_dir / "file_overview.csv", index=False)
    audit.to_csv(out_dir / "normalization_audit.csv", index=False)

    by_date = (
        burst.assign(date=burst["time"].dt.date.astype(str))
        .groupby("date", as_index=False)
        .agg(
            n_rows=("time", "size"),
            n_mp_ids=("MP_ID", pd.Series.nunique),
            n_frames=("Frame_ID", pd.Series.nunique),
            n_source_paths=("Source_Path", pd.Series.nunique),
            time_min=("time", "min"),
            time_max=("time", "max"),
        )
    )
    by_date.to_csv(out_dir / "date_coverage.csv", index=False)

    by_mp = (
        burst.groupby("MP_ID", as_index=False)
        .agg(
            n_rows=("time", "size"),
            n_dates=("time", lambda s: s.dt.date.nunique()),
            n_frames=("Frame_ID", pd.Series.nunique),
            n_source_paths=("Source_Path", pd.Series.nunique),
            time_min=("time", "min"),
            time_max=("time", "max"),
        )
        .sort_values(["time_min", "MP_ID"])
    )
    by_mp.to_csv(out_dir / "mp_region_coverage.csv", index=False)

    by_freq = (
        burst.groupby("channel_freq_mhz", as_index=False)
        .agg(
            n_rows=("time", "size"),
            n_dates=("time", lambda s: s.dt.date.nunique()),
            n_frames=("Frame_ID", pd.Series.nunique),
            value_median=("value_log_cal_temp", "median"),
            value_p05=("value_log_cal_temp", lambda s: float(np.nanquantile(s, 0.05))),
            value_p95=("value_log_cal_temp", lambda s: float(np.nanquantile(s, 0.95))),
            confidence_median=("point_confidence", "median"),
        )
        .sort_values("channel_freq_mhz")
    )
    by_freq.to_csv(out_dir / "frequency_summary.csv", index=False)

    repair = burst["repair_status"].value_counts(dropna=False).rename_axis("repair_status").reset_index(name="n_rows")
    repair.to_csv(out_dir / "repair_status_counts.csv", index=False)

    top_regions = by_mp.sort_values("n_rows", ascending=False).head(12)
    lines = [
        "# Burst Receiver 2 File Content Summary",
        "",
        f"Input CSV: `{burst_csv}`",
        "",
        "## Overview",
        "",
        overview.to_string(index=False),
        "",
        "## Largest MP Regions",
        "",
        top_regions.to_string(index=False),
        "",
        "## Frequency Channels",
        "",
        by_freq[["channel_freq_mhz", "n_rows", "n_dates", "n_frames", "confidence_median"]].to_string(index=False),
        "",
        "## Timestamp Repair",
        "",
        repair.to_string(index=False),
    ]
    (out_dir / "file_content_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _source_table(source_config: Path, names: list[str] | None) -> pd.DataFrame:
    sources = read_table(source_config)
    if names:
        wanted = {name.lower() for name in names}
        sources = sources[sources["source_name"].str.lower().isin(wanted)].copy()
    return sources.reset_index(drop=True)


def _load_master_geometry_slice(master_csv: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    required = [
        "time",
        "frequency_band",
        "position_x",
        "position_y",
        "position_z",
        "earth_unit_vector_x",
        "earth_unit_vector_y",
        "earth_unit_vector_z",
    ]
    cache_roots = [
        RAE2_V2 / "cache",
        RAE2_V2 / "cache" / "master",
    ]
    candidates: list[Path] = []
    for root in cache_roots:
        candidates.extend(sorted(root.glob("master_cache_*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True))

    for path in candidates:
        try:
            df = pd.read_pickle(path)
        except Exception:
            continue
        if set(required).issubset(df.columns):
            _log(f"using cached master geometry: {path}")
            out = df[required].copy()
            out["time"] = pd.to_datetime(out["time"])
            out = out[(out["time"] >= start) & (out["time"] <= end)].copy()
            if not out.empty:
                return out.sort_values("time").reset_index(drop=True)

    raise RuntimeError(
        "Could not find a cached master table with spacecraft and Earth-vector geometry. "
        f"Checked {len(candidates)} cache files for {master_csv}."
    )


def _predict_source_events(
    burst: pd.DataFrame,
    sources: pd.DataFrame,
    master_csv: Path,
    out_dir: Path,
    force: bool,
    cadence_s: float,
) -> pd.DataFrame:
    event_path = out_dir / "predicted_source_events.csv"
    state_path = out_dir / "source_visibility_states.csv"
    if event_path.exists() and not force:
        _log(f"reading cached source events: {event_path}")
        return read_table(event_path, parse_dates=["predicted_event_time"])

    start = burst["time"].min() - pd.Timedelta(hours=2)
    end = burst["time"].max() + pd.Timedelta(hours=2)
    _log(f"loading master geometry slice {start} to {end}")
    try:
        master = _load_master_geometry_slice(master_csv, start, end)
    except RuntimeError as exc:
        message = (
            f"{exc} The repaired burst range is {burst['time'].min()} to {burst['time'].max()}, "
            "but the available cached master geometry has no rows in that interval."
        )
        _log(message)
        pd.DataFrame().to_csv(event_path, index=False)
        pd.DataFrame([{"status": "not_predicted", "reason": message}]).to_csv(out_dir / "event_prediction_status.csv", index=False)
        return pd.DataFrame()
    _log(f"predicting events for {len(sources)} sources")
    events, states = predict_events(
        master,
        sources,
        target_frame="fk4",
        equinox="B1950",
        ephemeris="builtin",
        max_gap_seconds=900.0,
        prediction_cadence_seconds=cadence_s,
        frequencies=[1],
        antennas=["burst_receiver_2"],
    )
    if not events.empty:
        lo = burst["time"].min() - pd.Timedelta(seconds=1800)
        hi = burst["time"].max() + pd.Timedelta(seconds=1800)
        events = events[(events["predicted_event_time"] >= lo) & (events["predicted_event_time"] <= hi)].copy()
    events.to_csv(event_path, index=False)
    states.to_csv(state_path, index=False)
    return events


def _make_groups(burst: pd.DataFrame) -> dict[float, tuple[pd.DataFrame, np.ndarray]]:
    groups = {}
    for freq, group in burst.groupby("channel_freq_mhz", sort=True):
        g = group.sort_values("time").reset_index(drop=True)
        groups[float(freq)] = (g, datetime_ns(g["time"]))
    return groups


def _event_contrib(
    groups: dict[float, tuple[pd.DataFrame, np.ndarray]],
    events: pd.DataFrame,
    window_s: float,
    timing_offset_s: float,
    value_column: str,
    freqs: list[float],
) -> pd.DataFrame:
    rows = []
    half_ns = int(float(window_s) * 1e9)
    for _, ev in events.iterrows():
        event_time = pd.Timestamp(ev["predicted_event_time"])
        event_ns = event_time.value
        for freq in freqs:
            group, t_ns = groups[freq]
            lo = int(np.searchsorted(t_ns, event_ns - half_ns, side="left"))
            hi = int(np.searchsorted(t_ns, event_ns + half_ns, side="right"))
            if hi <= lo:
                continue
            local = group.iloc[lo:hi]
            rel = (datetime_ns(local["time"]) - event_ns).astype(float) / 1e9
            y = pd.to_numeric(local[value_column], errors="coerce").to_numpy(dtype=float)
            weight = np.clip(pd.to_numeric(local["point_confidence"], errors="coerce").fillna(1.0).to_numpy(dtype=float), 0.0, None)
            keep = np.isfinite(y) & np.isfinite(rel) & (weight > 0)
            if np.count_nonzero(keep) < 6:
                continue
            tr = rel[keep]
            yy = y[keep]
            baseline = baseline_matrix(tr, 1)
            beta, *_ = np.linalg.lstsq(baseline, yy, rcond=None)
            yd = yy - baseline @ beta
            sig = robust_sigma(yd)
            if np.isfinite(sig) and sig > 0:
                yd = yd / sig
            tmpl = event_template(tr, str(ev["event_type"]), timing_offset_sec=float(timing_offset_s))
            den = float(np.dot(tmpl, tmpl))
            if den <= 0:
                continue
            rows.append(
                {
                    "source_name": ev["source_name"],
                    "event_id": ev["event_id"],
                    "event_type": ev["event_type"],
                    "predicted_event_time": event_time,
                    "month_block": event_time.strftime("%Y-%m"),
                    "channel_freq_mhz": freq,
                    "window_s": float(window_s),
                    "timing_offset_s": float(timing_offset_s),
                    "n_used_samples": int(len(yd)),
                    "num": float(np.dot(yd, tmpl)),
                    "den": den,
                }
            )
    return pd.DataFrame.from_records(rows)


def _aggregate(contrib: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    if contrib.empty:
        return pd.DataFrame()
    rows = []
    for keys, group in contrib.groupby(by, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        meta = dict(zip(by, keys))
        num = float(group["num"].sum())
        den = float(group["den"].sum())
        amp = num / den if den > 0 else np.nan
        event_amp = group["num"].to_numpy(dtype=float) / group["den"].to_numpy(dtype=float)
        rows.append(
            {
                **meta,
                "n_events_with_data": int(group["event_id"].nunique()),
                "n_month_blocks": int(group["month_block"].nunique()),
                "n_samples_used": int(group["n_used_samples"].sum()),
                "stacked_amplitude": amp,
                "stacked_snr": float(amp * np.sqrt(den)) if den > 0 else np.nan,
                "event_sign_fraction": float((np.sign(event_amp) == np.sign(amp)).mean()) if np.isfinite(amp) else np.nan,
            }
        )
    return pd.DataFrame.from_records(rows)


def _random_control_p(
    groups: dict[float, tuple[pd.DataFrame, np.ndarray]],
    events: pd.DataFrame,
    observed_abs_snr: float,
    freq: float,
    window_s: float,
    timing_offset_s: float,
    value_column: str,
    rng: np.random.Generator,
    n_controls: int,
    time_min: pd.Timestamp,
    time_max: pd.Timestamp,
) -> tuple[float, float]:
    if events.empty or n_controls <= 0 or not np.isfinite(observed_abs_snr):
        return np.nan, np.nan
    span_ns = time_max.value - time_min.value
    if span_ns <= 0:
        return np.nan, np.nan
    null_abs = []
    base = events.copy()
    for _ in range(n_controls):
        random_ns = time_min.value + rng.integers(0, span_ns, size=len(base), dtype=np.int64)
        base["predicted_event_time"] = pd.to_datetime(random_ns)
        contrib = _event_contrib(groups, base, window_s, timing_offset_s, value_column, [freq])
        agg = _aggregate(contrib, ["source_name", "channel_freq_mhz", "window_s", "timing_offset_s"])
        if not agg.empty:
            null_abs.append(abs(float(agg["stacked_snr"].iloc[0])))
    if not null_abs:
        return np.nan, np.nan
    arr = np.asarray(null_abs, dtype=float)
    p = (float(np.count_nonzero(arr >= observed_abs_snr)) + 1.0) / (len(arr) + 1.0)
    return p, float(np.nanquantile(arr, 0.95))


def _run_source_stacks(
    burst: pd.DataFrame,
    events: pd.DataFrame,
    out_dir: Path,
    windows: list[float],
    timing_offsets: list[float],
    value_column: str,
    n_controls: int,
) -> pd.DataFrame:
    if events.empty:
        summary = pd.DataFrame(
            columns=[
                "source_name",
                "channel_freq_mhz",
                "window_s",
                "timing_offset_s",
                "n_events_with_data",
                "n_month_blocks",
                "n_samples_used",
                "stacked_amplitude",
                "stacked_snr",
                "event_sign_fraction",
                "random_time_empirical_p",
                "random_time_abs_snr_p95",
                "predicted_transitions_total",
            ]
        )
        summary.to_csv(out_dir / "burst_source_stack_summary.csv", index=False)
        return summary
    groups = _make_groups(burst)
    freqs = sorted(groups)
    all_scan = []
    all_event_type = []
    all_month = []
    rng = np.random.default_rng(20260509)
    for source, source_events in events.groupby("source_name", sort=True):
        _log(f"stacking burst samples for {source}: {source_events['event_id'].nunique()} predicted transitions")
        source_dir = ensure_dir(out_dir / str(source))
        scan_rows = []
        for window in windows:
            for offset in timing_offsets:
                contrib = _event_contrib(groups, source_events, window, offset, value_column, freqs)
                agg = _aggregate(contrib, ["source_name", "channel_freq_mhz", "window_s", "timing_offset_s"])
                if not agg.empty:
                    scan_rows.append(agg)
        scan = pd.concat(scan_rows, ignore_index=True) if scan_rows else pd.DataFrame()
        if scan.empty:
            continue
        scan.to_csv(source_dir / "channel_window_timing_scan.csv", index=False)
        best = scan.assign(abs_snr=scan["stacked_snr"].abs()).sort_values("abs_snr", ascending=False).iloc[0]
        best_freq = float(best["channel_freq_mhz"])
        best_window = float(best["window_s"])
        best_offset = float(best["timing_offset_s"])
        best_contrib = _event_contrib(groups, source_events, best_window, best_offset, value_column, [best_freq])
        etype = _aggregate(best_contrib, ["source_name", "channel_freq_mhz", "window_s", "timing_offset_s", "event_type"])
        month = _aggregate(best_contrib, ["source_name", "channel_freq_mhz", "window_s", "timing_offset_s", "month_block"])
        etype.to_csv(source_dir / "best_event_type_split.csv", index=False)
        month.to_csv(source_dir / "best_month_blocks.csv", index=False)
        p_random, null95 = _random_control_p(
            groups,
            source_events,
            abs(float(best["stacked_snr"])),
            best_freq,
            best_window,
            best_offset,
            value_column,
            rng,
            n_controls,
            burst["time"].min(),
            burst["time"].max(),
        )
        row = best.to_dict()
        row["random_time_empirical_p"] = p_random
        row["random_time_abs_snr_p95"] = null95
        row["predicted_transitions_total"] = int(source_events["event_id"].nunique())
        all_scan.append(row)
        if not etype.empty:
            all_event_type.append(etype)
        if not month.empty:
            all_month.append(month)
    summary = pd.DataFrame.from_records(all_scan)
    summary.to_csv(out_dir / "burst_source_stack_summary.csv", index=False)
    if all_event_type:
        pd.concat(all_event_type, ignore_index=True).to_csv(out_dir / "burst_source_best_event_type.csv", index=False)
    if all_month:
        pd.concat(all_month, ignore_index=True).to_csv(out_dir / "burst_source_best_month_blocks.csv", index=False)
    return summary


def _write_report(summary: pd.DataFrame, events: pd.DataFrame, out_dir: Path) -> None:
    ranked = summary.assign(abs_snr=summary["stacked_snr"].abs()).sort_values("abs_snr", ascending=False) if not summary.empty else summary
    if events.empty or "source_name" not in events.columns:
        event_counts = pd.DataFrame()
    else:
        event_counts = (
            events.groupby("source_name", as_index=False)
            .agg(predicted_transitions=("event_id", pd.Series.nunique), first_event=("predicted_event_time", "min"), last_event=("predicted_event_time", "max"))
            .sort_values("source_name")
        )
    lines = [
        "# Burst Receiver Source Occultation Analysis",
        "",
        "This run uses the repaired burst-receiver timestamps, predicts Ryle-Vonberg lunar-limb transitions over the same date range, "
        "and stacks digitized burst log-cal-temperature samples by source and burst frequency channel.",
        "",
        "## Source Stack Summary",
        "",
        ranked.to_string(index=False) if not ranked.empty else "No stack rows were produced.",
        "",
        "## Predicted Event Coverage",
        "",
        event_counts.to_string(index=False) if not event_counts.empty else "No predicted events in the burst date range.",
        "",
        "## Caveat",
        "",
        "This is a burst-receiver adaptation of the RyleVonberg source workflow. The burst file does not contain the original Ryle-Vonberg "
        "coarse antenna channels, so results are grouped by digitized burst `Channel_Freq` and receiver 2 rather than by RV band/antenna.",
    ]
    (out_dir / "burst_source_analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--burst-csv", type=Path, default=DEFAULT_BURST)
    parser.add_argument("--master-csv", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--source-config", type=Path, default=ROOT / "configs" / "bright_sources.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "burst_receiver_2_source_analysis")
    parser.add_argument("--sources", nargs="*", default=None)
    parser.add_argument("--windows", nargs="+", type=float, default=[300.0, 600.0, 900.0])
    parser.add_argument("--timing-offsets", nargs="+", type=float, default=[-300, -180, -120, -60, -30, 0, 30, 60, 120, 180, 300])
    parser.add_argument("--prediction-cadence-seconds", type=float, default=300.0)
    parser.add_argument("--value-column", default="normalized_value")
    parser.add_argument("--random-controls", type=int, default=100)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_dir = ensure_dir(args.output_dir)
    burst, audit = _read_or_build_normalized(args, out_dir)
    _write_content_stats(burst, audit, out_dir, args.burst_csv)
    sources = _source_table(args.source_config, args.sources)
    events = _predict_source_events(burst, sources, args.master_csv, out_dir, args.force, args.prediction_cadence_seconds)
    summary = _run_source_stacks(burst, events, out_dir, args.windows, args.timing_offsets, args.value_column, args.random_controls)
    _write_report(summary, events, out_dir)
    (out_dir / "run_config.json").write_text(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2) + "\n")
    print(out_dir / "file_content_summary.md")
    print(out_dir / "burst_source_analysis_report.md")
    print(out_dir / "burst_source_stack_summary.csv")


if __name__ == "__main__":
    main()
