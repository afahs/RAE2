#!/usr/bin/env python
"""Whole-dataset Sun validation with stack-level empirical controls."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rylevonberg.table_io import read_table

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rylevonberg.constants import add_frequency_mhz_column
from rylevonberg.events import predict_events
from rylevonberg.ingest import IngestOptions, ingest_csv
from rylevonberg.plotting import plot_event_window
from rylevonberg.quality import event_window_quality
from rylevonberg.solar_controls import MovingBodyOffsetConfig, event_burst_metrics, generate_moving_body_offset_controls
from rylevonberg.sources import load_source_list
from rylevonberg.stacking import detrend_profile_values
from rylevonberg.util import datetime_ns, ensure_dir, robust_sigma, write_json


def _read(path: str | Path, **kwargs) -> pd.DataFrame:
    return read_table(path, low_memory=False, **kwargs)


def _log(message: str) -> None:
    print(f"[sun-validation] {message}", flush=True)


def _prepare_clean(args: argparse.Namespace, root: Path) -> pd.DataFrame:
    clean_path = root / "01_ingest" / "cleaned_timeseries.csv"
    report_path = root / "01_ingest" / "validation_report.csv"
    if clean_path.exists() and not args.force_ingest:
        _log(f"reading cached cleaned table: {clean_path}")
        return _read(clean_path, parse_dates=["time"])
    _log(f"ingesting raw CSV: {args.data}")
    ensure_dir(clean_path.parent)
    clean, report = ingest_csv(
        args.data,
        IngestOptions(
            start_time=None,
            end_time=None,
            value_columns=tuple(args.value_columns),
            gap_factor=args.gap_factor,
            artifact_sigma=args.artifact_sigma,
        ),
    )
    if args.cache_cleaned:
        _log(f"writing cached cleaned table: {clean_path}")
        clean.to_csv(clean_path, index=False)
    report.to_csv(report_path, index=False)
    return clean


def _read_summary(path: Path) -> pd.DataFrame:
    if path.exists():
        return _read(path)
    return pd.DataFrame()


def _write_summary(table: pd.DataFrame, path: Path) -> pd.DataFrame:
    ensure_dir(path.parent)
    table.to_csv(path, index=False)
    return table


def _stack_summary_fast(
    clean: pd.DataFrame,
    events: pd.DataFrame,
    window_s: float,
    source_label: str | None = None,
    baseline_mode: str = "sideband_linear",
    sideband_exclusion_seconds: float = 120.0,
) -> pd.DataFrame:
    """Compute template-stack summaries without materializing aligned profiles."""
    if events.empty:
        return pd.DataFrame()
    clean_groups = {}
    for (freq, antenna), group in clean.groupby(["frequency_band", "antenna"], dropna=False, sort=True):
        sorted_group = group.sort_values("time").reset_index(drop=True)
        clean_groups[(freq, antenna)] = (sorted_group, datetime_ns(sorted_group["time"]))
    accum: dict[tuple[object, object, object], dict[str, object]] = {}
    for _, ev in events.iterrows():
        freq = ev.get("frequency_band")
        antenna = ev.get("antenna")
        payload = clean_groups.get((freq, antenna))
        if payload is None:
            continue
        group, t_ns = payload
        if group.empty:
            continue
        event_time = pd.Timestamp(ev["predicted_event_time"])
        event_ns = event_time.value
        half_ns = int(float(window_s) * 1e9)
        lo = int(np.searchsorted(t_ns, event_ns - half_ns, side="left"))
        hi = int(np.searchsorted(t_ns, event_ns + half_ns, side="right"))
        if hi <= lo:
            continue
        local = group.iloc[lo:hi]
        rel = (datetime_ns(local["time"]) - event_ns).astype(float) / 1e9
        y = pd.to_numeric(local["power"], errors="coerce").to_numpy(dtype=float)
        valid = local["is_valid"].to_numpy(dtype=bool) if "is_valid" in local else np.ones(len(local), dtype=bool)
        keep = valid & np.isfinite(y) & (np.abs(rel) <= float(window_s))
        if np.count_nonzero(keep) < 6:
            continue
        tr = rel[keep]
        yy = y[keep]
        yd, tmpl, _sig, _n_baseline = detrend_profile_values(
            tr,
            yy,
            str(ev["event_type"]),
            baseline_mode=baseline_mode,
            sideband_exclusion_seconds=sideband_exclusion_seconds,
            normalize=True,
        )
        key = (source_label or ev["source_name"], freq, antenna)
        slot = accum.setdefault(key, {"num": 0.0, "den": 0.0, "values": [], "events": set(), "blocks": {}})
        num = float(np.dot(yd, tmpl))
        den = float(np.dot(tmpl, tmpl))
        slot["num"] = float(slot["num"]) + num
        slot["den"] = float(slot["den"]) + den
        slot["values"].append(yd)
        slot["events"].add((ev.get("event_id"), str(event_time)))
        block = event_time.strftime("%Y-%m")
        b = slot["blocks"].setdefault(block, {"num": 0.0, "den": 0.0})
        b["num"] += num
        b["den"] += den
    rows = []
    for (source, freq, antenna), slot in accum.items():
        vals = np.concatenate(slot["values"]) if slot["values"] else np.array([], dtype=float)
        den = float(slot["den"])
        amp = float(slot["num"] / den) if den > 0 else np.nan
        sig = robust_sigma(vals)
        snr = float(amp * np.sqrt(den) / sig) if np.isfinite(sig) and sig > 0 else np.nan
        jack = []
        for block, b in slot["blocks"].items():
            rest_num = float(slot["num"]) - float(b["num"])
            rest_den = float(slot["den"]) - float(b["den"])
            if rest_den > 0:
                jack.append(rest_num / rest_den)
        max_lev = float(np.nanmax(np.abs(np.asarray(jack) - amp))) if jack and np.isfinite(amp) else np.nan
        rows.append(
            {
                "source_name": source,
                "frequency_band": freq,
                "antenna": antenna,
                "window_s": float(window_s),
                "n_events": int(len(slot["events"])),
                "n_month_blocks": int(len(slot["blocks"])),
                "stacked_amplitude": amp,
                "stacked_snr": snr,
                "month_jackknife_std": float(np.std(jack, ddof=1)) if len(jack) > 1 else np.nan,
                "max_month_leverage": max_lev,
            }
        )
    return add_frequency_mhz_column(pd.DataFrame.from_records(rows))


def _randomized_stack_controls(
    events: pd.DataFrame,
    clean: pd.DataFrame,
    window_s: float,
    n_random: int,
    seed: int,
    baseline_mode: str,
    sideband_exclusion_seconds: float,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(clean["time"].min()).value
    end = pd.Timestamp(clean["time"].max()).value
    tables = []
    for idx in range(int(n_random)):
        rand = events.copy()
        rand["source_name"] = f"sun_random_{idx:03d}"
        rand["predicted_event_time"] = pd.to_datetime(rng.integers(start, end, size=len(rand)))
        tables.append(_stack_summary_fast(
            clean,
            rand,
            window_s,
            source_label=f"sun_random_{idx:03d}",
            baseline_mode=baseline_mode,
            sideband_exclusion_seconds=sideband_exclusion_seconds,
        ))
    out = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    out["control_type"] = "randomized_time_stack"
    return out


def _subsample_events_per_group(events: pd.DataFrame, max_per_group: int | None, seed: int) -> pd.DataFrame:
    if events.empty or not max_per_group or int(max_per_group) <= 0:
        return events
    group_cols = [c for c in ["source_name", "event_type", "frequency_band", "antenna"] if c in events.columns]
    rows = []
    rng = np.random.default_rng(seed)
    for key, group in events.groupby(group_cols, dropna=False, sort=True):
        if len(group) <= int(max_per_group):
            rows.append(group)
            continue
        rows.append(group.sample(n=int(max_per_group), random_state=int(rng.integers(0, 2**31 - 1))).sort_values("predicted_event_time"))
    return pd.concat(rows, ignore_index=True) if rows else events.head(0)


def _pvalue(real: float, controls: pd.Series) -> float:
    vals = pd.to_numeric(controls, errors="coerce").abs().dropna().to_numpy(dtype=float)
    if not np.isfinite(real) or vals.size == 0:
        return np.nan
    return float((1 + np.count_nonzero(vals >= abs(float(real)))) / (1 + vals.size))


def _score_real(real: pd.DataFrame, off: pd.DataFrame, rand: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in real.iterrows():
        freq = row["frequency_band"]
        antenna = row["antenna"]
        off_sub = off[(off["frequency_band"].eq(freq)) & (off["antenna"].eq(antenna))] if not off.empty else pd.DataFrame()
        rand_sub = rand[(rand["frequency_band"].eq(freq)) & (rand["antenna"].eq(antenna))] if not rand.empty else pd.DataFrame()
        q_sub = quality[(quality["frequency_band"].eq(freq)) & (quality["antenna"].eq(antenna))] if not quality.empty else pd.DataFrame()
        clean_fraction = float((q_sub["primary_quality_failure"].astype(str).eq("")).mean()) if not q_sub.empty else np.nan
        primary_failure = ""
        if not q_sub.empty:
            failures = q_sub["primary_quality_failure"].astype(str)
            failures = failures[failures.ne("")]
            primary_failure = str(failures.value_counts().idxmax()) if not failures.empty else ""
        p_off = _pvalue(row["stacked_snr"], off_sub["stacked_snr"] if "stacked_snr" in off_sub else pd.Series(dtype=float))
        p_rand = _pvalue(row["stacked_snr"], rand_sub["stacked_snr"] if "stacked_snr" in rand_sub else pd.Series(dtype=float))
        status = "unresolved"
        reason = "missing one or more controls"
        if abs(row["stacked_snr"]) < 3:
            status = "not_detected"
            reason = "stacked SNR below threshold"
        elif np.isfinite(p_off) and p_off > 0.1 and abs(row["stacked_snr"]) >= 5:
            status = "likely_systematic"
            reason = "moving off-ephemeris controls are comparable"
        elif np.isfinite(p_off) and p_off <= 0.05 and np.isfinite(p_rand) and p_rand <= 0.05 and clean_fraction >= 0.5:
            status = "candidate"
            reason = "beats moving and randomized controls with acceptable quality"
        rows.append({**row.to_dict(), "randomized_stack_p": p_rand, "offephemeris_stack_p": p_off, "clean_fraction": clean_fraction, "primary_quality_failure": primary_failure, "status": status, "decision_reason": reason})
    return pd.DataFrame.from_records(rows).sort_values("stacked_snr", key=lambda s: s.abs(), ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=os.environ.get("RAE2_MASTER_CSV", "data/interpolatedRAE2MasterFile.csv"))
    parser.add_argument("--sources", default="configs/bright_sources.csv")
    parser.add_argument("--output-root", default="outputs/sun_whole_dataset_validation")
    parser.add_argument("--value-columns", nargs="+", default=["rv1_coarse", "rv2_coarse"])
    parser.add_argument("--gap-factor", type=float, default=3.0)
    parser.add_argument("--artifact-sigma", type=float, default=12.0)
    parser.add_argument("--window", type=float, default=600.0)
    parser.add_argument("--prediction-cadence-seconds", type=float, default=300.0)
    parser.add_argument("--frequencies", nargs="+", type=int, default=list(range(1, 10)))
    parser.add_argument("--antennas", nargs="+", default=["rv1_coarse", "rv2_coarse"])
    parser.add_argument("--radial-offsets", nargs="+", type=float, default=[2.0, 5.0, 10.0])
    parser.add_argument("--annulus-positions", type=int, default=4)
    parser.add_argument("--n-random-stack-controls", type=int, default=16)
    parser.add_argument("--max-control-events-per-group", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260506)
    parser.add_argument("--force-ingest", action="store_true")
    parser.add_argument("--force-predict", action="store_true")
    parser.add_argument("--force-stack", action="store_true", help="Recompute stack summaries even if cached summary CSVs exist.")
    parser.add_argument("--force-diagnostics", action="store_true", help="Recompute quality and burst diagnostics even if cached CSVs exist.")
    parser.add_argument("--force-plots", action="store_true", help="Regenerate review plots even if they already exist.")
    parser.add_argument("--skip-plots", action="store_true", help="Skip review plot generation.")
    parser.add_argument("--cache-cleaned", action="store_true", help="Write the full cleaned long table. This can require several GB.")
    parser.add_argument("--baseline-mode", default="sideband_linear", choices=["linear_all", "constant_all", "sideband_linear", "sideband_constant", "joint_step_linear", "pre_event_anchor"])
    parser.add_argument("--sideband-exclusion-seconds", type=float, default=120.0)
    args = parser.parse_args()

    root = ensure_dir(args.output_root)
    clean = _prepare_clean(args, root)
    summary_dir = ensure_dir(root / "summary")
    event_dir = ensure_dir(root / "02_events")
    source_table = load_source_list(args.sources)
    sun = source_table[source_table["source_name"].eq("sun")].reset_index(drop=True)
    events_path = event_dir / "sun_predicted_events.csv"
    if events_path.exists() and not args.force_predict:
        _log(f"reading cached Sun events: {events_path}")
        events = _read(events_path, parse_dates=["predicted_event_time"])
    else:
        _log("predicting Sun events")
        events, _states = predict_events(
            clean,
            sun,
            target_frame="fk4",
            equinox="B1950",
            ephemeris="builtin",
            max_gap_seconds=600.0,
            prediction_cadence_seconds=args.prediction_cadence_seconds,
            frequencies=args.frequencies,
            antennas=args.antennas,
        )
        events.to_csv(events_path, index=False)

    controls = generate_moving_body_offset_controls(MovingBodyOffsetConfig(radial_offsets_deg=tuple(args.radial_offsets), annulus_positions=args.annulus_positions))
    controls_path = root / "02_events" / "sun_offephemeris_controls.csv"
    controls.to_csv(controls_path, index=False)
    off_events_path = event_dir / "sun_offephemeris_predicted_events.csv"
    if off_events_path.exists() and not args.force_predict:
        _log(f"reading cached off-ephemeris events: {off_events_path}")
        off_events = _read(off_events_path, parse_dates=["predicted_event_time"])
    else:
        _log("predicting off-ephemeris Sun-control events")
        off_events, _ = predict_events(
            clean,
            controls,
            target_frame="fk4",
            equinox="B1950",
            ephemeris="builtin",
            max_gap_seconds=600.0,
            prediction_cadence_seconds=args.prediction_cadence_seconds,
            frequencies=args.frequencies,
            antennas=args.antennas,
        )
        off_events = off_events.merge(controls[["source_name", "parent_source", "control_name", "control_type", "offset_deg", "offset_pa_deg"]], on="source_name", how="left")
        off_events.to_csv(off_events_path, index=False)

    real_stack_path = summary_dir / "sun_real_stack_summary.csv"
    off_stack_path = summary_dir / "sun_offephemeris_stack_summary.csv"
    rand_stack_path = summary_dir / "sun_randomized_stack_summary.csv"
    quality_path = summary_dir / "sun_event_window_quality.csv"
    bursts_path = summary_dir / "sun_event_burst_metrics.csv"
    scored_path = summary_dir / "sun_whole_dataset_scored_stacks.csv"

    if real_stack_path.exists() and not args.force_stack:
        _log(f"reading cached real stack: {real_stack_path}")
        real_stack = _read_summary(real_stack_path)
    else:
        _log("computing real Sun stack")
        real_stack = _write_summary(_stack_summary_fast(
            clean,
            events,
            args.window,
            baseline_mode=args.baseline_mode,
            sideband_exclusion_seconds=args.sideband_exclusion_seconds,
        ), real_stack_path)

    if off_stack_path.exists() and not args.force_stack:
        _log(f"reading cached off-ephemeris stack: {off_stack_path}")
        off_stack = _read_summary(off_stack_path)
    else:
        _log("computing off-ephemeris control stack")
        off_events_for_stack = _subsample_events_per_group(off_events, args.max_control_events_per_group, args.seed + 17)
        off_stack = _stack_summary_fast(
            clean,
            off_events_for_stack,
            args.window,
            baseline_mode=args.baseline_mode,
            sideband_exclusion_seconds=args.sideband_exclusion_seconds,
        )
        if not off_stack.empty:
            off_stack = off_stack.merge(off_events[["source_name", "parent_source", "control_name", "control_type", "offset_deg", "offset_pa_deg"]].drop_duplicates("source_name"), on="source_name", how="left")
        off_stack = _write_summary(off_stack, off_stack_path)

    if rand_stack_path.exists() and not args.force_stack:
        _log(f"reading cached randomized stack: {rand_stack_path}")
        rand_stack = _read_summary(rand_stack_path)
    else:
        _log("computing randomized-time control stacks")
        real_events_for_random = _subsample_events_per_group(events, args.max_control_events_per_group, args.seed + 23)
        rand_stack = _write_summary(_randomized_stack_controls(
            real_events_for_random,
            clean,
            args.window,
            args.n_random_stack_controls,
            args.seed,
            args.baseline_mode,
            args.sideband_exclusion_seconds,
        ), rand_stack_path)

    if quality_path.exists() and not args.force_diagnostics:
        _log(f"reading cached quality diagnostics: {quality_path}")
        quality = _read_summary(quality_path)
    else:
        _log("computing event-window quality diagnostics")
        quality = _write_summary(event_window_quality(clean, events, args.window), quality_path)

    if bursts_path.exists() and not args.force_diagnostics:
        _log(f"reading cached burst metrics: {bursts_path}")
        bursts = _read_summary(bursts_path)
    else:
        _log("computing event-window burst metrics")
        bursts = _write_summary(event_burst_metrics(clean, events, args.window), bursts_path)

    scored = _score_real(real_stack, off_stack, rand_stack, quality)

    scored.to_csv(scored_path, index=False)

    top = scored.head(12)
    plots = ensure_dir(root / "review_plots")
    if not args.skip_plots and not events.empty:
        candidates = events.merge(scored[["frequency_band", "antenna", "status"]], on=["frequency_band", "antenna"], how="left")
        candidates = candidates[candidates["status"].isin(["candidate", "likely_systematic", "unresolved"])].head(16)
        for idx, (_, ev) in enumerate(candidates.iterrows()):
            plot_path = plots / f"sun_event_{idx:04d}.png"
            if args.force_plots or not plot_path.exists():
                plot_event_window(clean, ev, plot_path, args.window)

    date_range = f"{pd.Timestamp(clean['time'].min())} through {pd.Timestamp(clean['time'].max())}"
    report = [
        "# Whole-Dataset Sun Validation",
        "",
        f"Data span: `{date_range}`.",
        f"Window: `{int(args.window)} s`.",
        "",
        "## Technique",
        "",
        "1. Ingest the full RAE-2 Ryle-Vonberg CSV into the same cleaned long-form table used by the rest of the pipeline.",
        "2. Predict solar lunar-limb disappearance/reappearance events over the whole available dataset.",
        "3. Compute robust local baseline-subtracted template stacks per frequency band and antenna.",
        "4. Generate moving off-ephemeris fake-Sun tracks and run them through the same event prediction and stack calculation.",
        "5. Generate randomized-time stack controls over the same full time span.",
        "6. Estimate month-block jackknife stability, event-window quality, and burst-like local variance metrics.",
        "7. Classify each channel by whether it beats both randomized and moving off-ephemeris controls.",
        "",
        "## Channel Results",
        "",
        top[["frequency_band", "frequency_mhz", "antenna", "n_events", "stacked_snr", "randomized_stack_p", "offephemeris_stack_p", "clean_fraction", "month_jackknife_std", "max_month_leverage", "status", "decision_reason"]].to_string(index=False),
        "",
        "## Interpretation",
        "",
        "A solar detection claim requires the real solar stack to beat randomized-time controls and nearby moving fake-Sun tracks. "
        "If off-ephemeris p-values are large, the stack is likely geometry/background/systematics rather than uniquely solar. "
        "Burst-like date clusters may still be useful, but they should be interpreted separately from a repeated occultation stack.",
        "",
        "## Output Files",
        "",
        f"- Real stack summary: `{summary_dir / 'sun_real_stack_summary.csv'}`",
        f"- Off-ephemeris stack summary: `{summary_dir / 'sun_offephemeris_stack_summary.csv'}`",
        f"- Randomized stack summary: `{summary_dir / 'sun_randomized_stack_summary.csv'}`",
        f"- Scored channel summary: `{summary_dir / 'sun_whole_dataset_scored_stacks.csv'}`",
        f"- Quality diagnostics: `{summary_dir / 'sun_event_window_quality.csv'}`",
        f"- Burst metrics: `{summary_dir / 'sun_event_burst_metrics.csv'}`",
        f"- Review plots: `{plots}`",
    ]
    (root / "sun_whole_dataset_validation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(root / "run_config.json", vars(args))
    print(root / "sun_whole_dataset_validation_report.md")
    print(summary_dir / "sun_whole_dataset_scored_stacks.csv")


if __name__ == "__main__":
    main()
