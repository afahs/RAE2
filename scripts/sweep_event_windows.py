#!/usr/bin/env python
"""Sweep detection/stacking window sizes for an existing control-survey run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from rylevonberg.table_io import read_table

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rylevonberg.detection import StepFitConfig, run_matched_filter, run_stepfit_detections
from rylevonberg.scoring import ScoreConfig, score_detections
from rylevonberg.stacking import aligned_profiles, stack_profiles
from rylevonberg.util import ensure_dir


def _grade_counts(scored: pd.DataFrame) -> dict[str, int]:
    if scored.empty or "detection_grade" not in scored:
        return {}
    return {str(k): int(v) for k, v in scored["detection_grade"].value_counts().items()}


def _best_stack(stack_summary: pd.DataFrame, source: str) -> dict[str, object]:
    if stack_summary.empty or "stacked_snr" not in stack_summary:
        return {}
    sub = stack_summary[stack_summary["source_name"].astype(str).str.lower() == source.lower()].copy()
    if sub.empty:
        return {}
    sub["_abs"] = sub["stacked_snr"].abs()
    row = sub.sort_values("_abs", ascending=False).iloc[0]
    return {
        f"{source}_best_stack_snr": float(row["stacked_snr"]),
        f"{source}_best_stack_abs_snr": float(abs(row["stacked_snr"])),
        f"{source}_best_stack_band": int(row["frequency_band"]),
        f"{source}_best_stack_mhz": float(row.get("frequency_mhz", float("nan"))),
        f"{source}_best_stack_antenna": row["antenna"],
        f"{source}_best_stack_n_events": int(row["n_events"]),
    }


def _source_counts(scored: pd.DataFrame, source: str) -> dict[str, int]:
    if scored.empty or "source_name" not in scored:
        return {}
    sub = scored[scored["source_name"].astype(str).str.lower() == source.lower()]
    if sub.empty:
        return {}
    grades = sub["detection_grade"].value_counts().to_dict() if "detection_grade" in sub else {}
    return {
        f"{source}_candidate_needs_review": int(grades.get("candidate_needs_review", 0)),
        f"{source}_quality_limited": int(grades.get("quality_limited", 0)),
        f"{source}_strong_control_validated": int(grades.get("strong_control_validated", 0)),
    }


def run_sweep(args: argparse.Namespace) -> pd.DataFrame:
    out_root = ensure_dir(args.output_dir)
    clean = read_table(args.cleaned, parse_dates=["time"])
    events = read_table(args.events, parse_dates=["predicted_event_time"])
    stack_events = read_table(args.stack_events, parse_dates=["predicted_event_time"])
    controls = read_table(args.control_events, parse_dates=["predicted_event_time"])

    rows: list[dict[str, object]] = []
    for window in args.windows:
        window = float(window)
        label = f"window_{int(window)}s"
        out = ensure_dir(out_root / label) if args.write_tables else out_root / label
        print(f"running {label}", flush=True)

        step_cfg = StepFitConfig(
            window_seconds=window,
            baseline_order=args.baseline_order,
            min_samples_per_side=args.min_samples_per_side,
            smooth_seconds=args.smooth_seconds,
            timing_grid_seconds=tuple(args.timing_offsets),
        )
        step = run_stepfit_detections(clean, events, step_cfg)
        matched = run_matched_filter(clean, events, window_seconds=window, baseline_order=args.baseline_order)
        if args.write_tables:
            step.to_csv(out / "per_event_stepfit_detections.csv", index=False)
            matched.to_csv(out / "per_event_matched_filter.csv", index=False)

        profiles = aligned_profiles(clean, stack_events, window_seconds=window, bin_seconds=args.bin_seconds)
        stacked, stack_summary = stack_profiles(profiles, n_bootstrap=args.bootstrap)
        if args.write_tables:
            profiles.to_csv(out / "event_aligned_profiles.csv", index=False)
            stacked.to_csv(out / "stacked_event_profiles.csv", index=False)
            stack_summary.to_csv(out / "stacked_detection_summary.csv", index=False)

        score_cfg = ScoreConfig(
            window_seconds=window,
            baseline_order=args.baseline_order,
            min_samples_per_side=args.min_samples_per_side,
            smooth_seconds=args.smooth_seconds,
            timing_grid_seconds=tuple(args.timing_offsets),
            null_percentile_threshold=args.null_percentile_threshold,
            min_abs_snr=args.min_abs_snr,
            clean_fraction_threshold=args.clean_fraction_threshold,
            empirical_control_types=tuple(args.empirical_control_types),
        )
        scored, null_summary, null_step, null_matched = score_detections(clean, step, matched, controls, config=score_cfg)
        if args.write_tables:
            scored.to_csv(out / "scored_detections.csv", index=False)
            null_summary.to_csv(out / "null_snr_summary.csv", index=False)
            null_step.to_csv(out / "control_stepfit_detections.csv", index=False)
            null_matched.to_csv(out / "control_matched_filter.csv", index=False)

        grades = _grade_counts(scored)
        finite_p = pd.to_numeric(scored.get("best_empirical_p"), errors="coerce")
        finite_snr = pd.to_numeric(scored.get("best_abs_snr"), errors="coerce")
        clean_frac = pd.to_numeric(scored.get("quality_clean_fraction"), errors="coerce")
        row: dict[str, object] = {
            "window_seconds": window,
            "n_scored": int(len(scored)),
            "candidate_needs_review": grades.get("candidate_needs_review", 0),
            "strong_control_validated": grades.get("strong_control_validated", 0),
            "quality_limited": grades.get("quality_limited", 0),
            "not_significant_against_controls": grades.get("not_significant_against_controls", 0),
            "n_empirical_p_le_0p05": int((finite_p <= 0.05).sum()),
            "min_empirical_p": float(finite_p.min()) if finite_p.notna().any() else float("nan"),
            "max_abs_single_event_snr": float(finite_snr.max()) if finite_snr.notna().any() else float("nan"),
            "median_quality_clean_fraction": float(clean_frac.median()) if clean_frac.notna().any() else float("nan"),
            "n_stack_groups": int(len(stack_summary)),
            "outputs": str(out) if args.write_tables else "",
        }
        row.update(_best_stack(stack_summary, "earth"))
        row.update(_best_stack(stack_summary, "sun"))
        for source in args.summary_sources:
            row.update(_source_counts(scored, source))
            row.update(_best_stack(stack_summary, source))
        rows.append(row)
        pd.DataFrame.from_records(rows).sort_values("window_seconds").to_csv(out_root / "window_sweep_summary.csv", index=False)

    summary = pd.DataFrame.from_records(rows).sort_values("window_seconds")
    summary.to_csv(out_root / "window_sweep_summary.csv", index=False)
    write_report(out_root / "window_sweep_report.md", summary)
    return summary


def write_report(path: Path, summary: pd.DataFrame) -> None:
    best_earth = summary.sort_values("earth_best_stack_abs_snr", ascending=False).iloc[0] if "earth_best_stack_abs_snr" in summary else None
    best_sun = summary.sort_values("sun_best_stack_abs_snr", ascending=False).iloc[0] if "sun_best_stack_abs_snr" in summary else None
    best_candidates = summary.sort_values("candidate_needs_review", ascending=False).iloc[0]
    lines = [
        "# Event Window Sweep",
        "",
        "This sweep reruns local step fits, matched filters, randomized-time scoring, and capped repeated-event stacking for each window size.",
        "",
        "## Summary",
        "",
        summary.to_string(index=False),
        "",
        "## Interpretation",
        "",
        f"- Most single-event candidates: {int(best_candidates['window_seconds'])} s ({int(best_candidates['candidate_needs_review'])} candidates).",
    ]
    if best_earth is not None:
        lines.append(
            f"- Strongest Earth stack: {int(best_earth['window_seconds'])} s, "
            f"SNR {best_earth['earth_best_stack_snr']:.2f}, band {int(best_earth['earth_best_stack_band'])} "
            f"({best_earth['earth_best_stack_mhz']:.2f} MHz), {best_earth['earth_best_stack_antenna']}."
        )
    if best_sun is not None:
        lines.append(
            f"- Strongest Sun stack: {int(best_sun['window_seconds'])} s, "
            f"SNR {best_sun['sun_best_stack_snr']:.2f}, band {int(best_sun['sun_best_stack_band'])} "
            f"({best_sun['sun_best_stack_mhz']:.2f} MHz), {best_sun['sun_best_stack_antenna']}."
        )
    lines.extend(
        [
            "",
            "Short windows reduce baseline contamination but can miss slow transitions and produce fewer usable samples.",
            "Long windows improve sample count but increase sensitivity to baseline drift, telemetry artifacts, and unrelated sky/background changes.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cleaned", required=True)
    p.add_argument("--events", required=True)
    p.add_argument("--stack-events", required=True)
    p.add_argument("--control-events", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--windows", nargs="+", type=float, default=[180.0, 300.0, 600.0, 900.0, 1200.0])
    p.add_argument("--baseline-order", type=int, default=1)
    p.add_argument("--min-samples-per-side", type=int, default=4)
    p.add_argument("--smooth-seconds", type=float, default=0.0)
    p.add_argument("--timing-offsets", nargs="+", type=float, default=[-120.0, -60.0, -30.0, 0.0, 30.0, 60.0, 120.0])
    p.add_argument("--bin-seconds", type=float, default=60.0)
    p.add_argument("--bootstrap", type=int, default=30)
    p.add_argument("--null-percentile-threshold", type=float, default=99.0)
    p.add_argument("--min-abs-snr", type=float, default=3.0)
    p.add_argument("--clean-fraction-threshold", type=float, default=0.8)
    p.add_argument("--empirical-control-types", nargs="+", default=["randomized_time"])
    p.add_argument("--summary-sources", nargs="*", default=[])
    p.add_argument("--write-tables", action="store_true", help="Write per-window tables. By default only summary/report files are written.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    summary = run_sweep(args)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
