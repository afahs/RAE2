#!/usr/bin/env python
"""Run lower-V raw triage and structure-selected stacks for known event sources."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from rylevonberg.table_io import read_table

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.util import ensure_dir, software_versions, write_json  # noqa: E402
from scripts.flag_raw_occultation_candidates import (  # noqa: E402
    _plot_candidate_grid,
    _plot_score_summary,
    _write_report as write_triage_report,
    score_events,
)
from scripts.stack_structure_selected_raw_events import (  # noqa: E402
    collect_selected_profiles,
    plot_stack_grid,
    stack_profiles,
    stack_statistics,
    write_report as write_stack_report,
)


CLEAN = ROOT / "outputs/control_survey_earth_sun_postnov1974_v2/01_ingest/cleaned_timeseries.csv"


@dataclass(frozen=True)
class SourceSpec:
    source: str
    events_path: Path
    family: str


SOURCE_SPECS = [
    SourceSpec("sun", ROOT / "outputs/pipeline_confidence_audit_v2/sun_audit_input_events.csv", "moving_body"),
    SourceSpec("earth", ROOT / "outputs/planetary_confirmation_survey_science_baseline_v2/events/all_planet_predicted_events.csv", "moving_body"),
    SourceSpec("jupiter", ROOT / "outputs/planetary_confirmation_survey_science_baseline_v2/events/all_planet_predicted_events.csv", "moving_body"),
    SourceSpec("mercury", ROOT / "outputs/planetary_confirmation_survey/events/all_planet_predicted_events.csv", "moving_body"),
    SourceSpec("venus", ROOT / "outputs/planetary_confirmation_survey/events/all_planet_predicted_events.csv", "moving_body"),
    SourceSpec("mars", ROOT / "outputs/planetary_confirmation_survey/events/all_planet_predicted_events.csv", "moving_body"),
    SourceSpec("saturn", ROOT / "outputs/planetary_confirmation_survey/events/all_planet_predicted_events.csv", "moving_body"),
    SourceSpec("uranus", ROOT / "outputs/planetary_confirmation_survey/events/all_planet_predicted_events.csv", "moving_body"),
    SourceSpec("neptune", ROOT / "outputs/planetary_confirmation_survey/events/all_planet_predicted_events.csv", "moving_body"),
    SourceSpec("fornax_a", ROOT / "outputs/control_survey_bright_sources_postnov1974_v1/02_events/predicted_events.csv", "bright_fixed"),
    SourceSpec("cas_a", ROOT / "outputs/control_survey_bright_sources_postnov1974_v1/02_events/predicted_events.csv", "bright_fixed"),
    SourceSpec("cyg_a", ROOT / "outputs/control_survey_bright_sources_postnov1974_v1/02_events/predicted_events.csv", "bright_fixed"),
    SourceSpec("tau_a", ROOT / "outputs/custom_fixed_source_profile_grids_tau_a_v1/custom_fixed_source_predicted_events.csv", "custom_fixed"),
    SourceSpec("3c_273", ROOT / "outputs/custom_fixed_source_profile_grids_3c295_3c273_virgoa_v1/custom_fixed_source_predicted_events.csv", "custom_fixed"),
    SourceSpec("3c_295", ROOT / "outputs/custom_fixed_source_profile_grids_3c295_3c273_virgoa_v1/custom_fixed_source_predicted_events.csv", "custom_fixed"),
    SourceSpec("vir_a", ROOT / "outputs/custom_fixed_source_profile_grids_3c295_3c273_virgoa_v1/custom_fixed_source_predicted_events.csv", "custom_fixed"),
]


def _read(path: Path, **kwargs) -> pd.DataFrame:
    return read_table(path, low_memory=False, **kwargs)


def _parse_sources(text: str) -> list[str]:
    if str(text).strip().lower() in {"", "all"}:
        return [s.source for s in SOURCE_SPECS]
    return [x.strip().lower() for x in str(text).split(",") if x.strip()]


def _load_event_tables(specs: list[SourceSpec]) -> dict[Path, pd.DataFrame]:
    tables: dict[Path, pd.DataFrame] = {}
    for path in sorted({s.events_path for s in specs}):
        if not path.exists():
            continue
        tables[path] = _read(path, parse_dates=["predicted_event_time"])
    return tables


def _write_source_index(out_dir: Path, run_rows: list[dict[str, object]], config: dict[str, object]) -> None:
    table = pd.DataFrame(run_rows)
    if not table.empty:
        table.to_csv(out_dir / "all_source_structure_selected_run_summary.csv", index=False)
    lines = [
        "# Lower-V Structure-Selected Stack Batch",
        "",
        "This batch applies the same lower-V raw-triage and high-priority structure-selected stacking workflow to every requested source with an available event table.",
        "",
        "The workflow is intentionally selection-biased: it first finds individual lower-V raw windows that already have expected-sign pre/post morphology, then stacks that subset.",
        "Therefore these stacks are manual-review diagnostics, not source detection claims.",
        "",
        "## Configuration",
        "",
        pd.Series(config).to_string(),
        "",
        "## Source Outputs",
        "",
        table.to_string(index=False) if not table.empty else "No source outputs generated.",
        "",
    ]
    (out_dir / "all_source_structure_selected_stack_index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(ROOT / "outputs/lower_v_structure_selected_stacks_all_sources_v1"))
    parser.add_argument("--sources", default="all")
    parser.add_argument("--antenna", default="rv2_coarse")
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
    parser.add_argument("--shortlist-per-group", type=int, default=3)
    parser.add_argument("--top-plot-panels", type=int, default=24)
    args = parser.parse_args()

    out_dir = ensure_dir(Path(args.out_dir))
    requested = set(_parse_sources(args.sources))
    specs = [s for s in SOURCE_SPECS if s.source in requested]
    missing_request = sorted(requested - {s.source for s in specs})
    if missing_request:
        raise SystemExit(f"unknown requested source(s): {', '.join(missing_request)}")
    tables = _load_event_tables(specs)

    available_specs = []
    all_event_subsets = []
    for spec in specs:
        events = tables.get(spec.events_path)
        if events is None or events.empty:
            continue
        subset = events[
            events["source_name"].astype(str).str.lower().eq(spec.source)
            & events["antenna"].astype(str).eq(str(args.antenna))
        ].copy()
        if subset.empty:
            continue
        available_specs.append(spec)
        all_event_subsets.append(subset)
    if not available_specs:
        raise SystemExit("No matching source event rows found.")

    all_events = pd.concat(all_event_subsets, ignore_index=True)
    bands = sorted(all_events["frequency_band"].astype(int).unique())
    clean_cols = ["time", "frequency_band", "frequency_mhz", "antenna", "power", "is_valid"]
    clean = _read(CLEAN, usecols=clean_cols, parse_dates=["time"])
    clean = clean[clean["frequency_band"].astype(int).isin(bands) & clean["antenna"].astype(str).eq(str(args.antenna))].copy()

    run_rows: list[dict[str, object]] = []
    for spec in available_specs:
        source_out = ensure_dir(out_dir / spec.source)
        events = tables[spec.events_path]
        event_subset = events[
            events["source_name"].astype(str).str.lower().eq(spec.source)
            & events["antenna"].astype(str).eq(str(args.antenna))
        ].copy()
        event_types = set(event_subset["event_type"].astype(str).dropna().unique())
        frequencies = {float(x) for x in event_subset["frequency_mhz"].dropna().unique()}
        candidates = score_events(
            clean=clean,
            events=event_subset,
            source=spec.source,
            event_types=event_types,
            antennas={str(args.antenna)},
            frequencies=frequencies,
            window_s=float(args.window_s),
            prepost_s=float(args.prepost_s),
            inner_s=float(args.inner_s),
            scan_radius_s=float(args.scan_radius_s),
            scan_step_s=float(args.scan_step_s),
            min_side_samples=int(args.min_side_samples),
            use_existing_valid=True,
            min_predicted_z=float(args.min_predicted_z),
            min_best_z=float(args.min_best_z),
            max_abs_offset_s=float(args.max_abs_offset_s),
            min_support_bins=int(args.min_support_bins),
        )
        candidates.to_csv(source_out / "raw_occultation_candidate_scores.csv", index=False)
        flagged = candidates[candidates["manual_review_priority"].ne("not_flagged") & candidates["manual_review_priority"].ne("unusable")].copy()
        priority_order = {"high_priority": 0, "offset_candidate": 1, "anti_template_review": 2, "weak_predicted_candidate": 3}
        if not flagged.empty:
            flagged["priority_order"] = flagged["manual_review_priority"].map(priority_order).fillna(9)
            flagged["review_rank_score"] = np.where(
                flagged["manual_review_priority"].eq("anti_template_review"),
                pd.to_numeric(flagged["best_opposite_step_z"], errors="coerce"),
                pd.to_numeric(flagged["best_step_z"], errors="coerce"),
            )
            flagged = flagged.sort_values(["priority_order", "review_rank_score", "predicted_step_z"], ascending=[True, False, False])
        flagged.to_csv(source_out / "manual_review_candidate_flags.csv", index=False)
        if flagged.empty:
            shortlist = flagged.copy()
        else:
            shortlist = (
                flagged.groupby(["frequency_mhz", "event_type", "antenna", "manual_review_priority"], group_keys=False, dropna=False)
                .head(int(args.shortlist_per_group))
                .copy()
            )
        shortlist.to_csv(source_out / "manual_review_shortlist.csv", index=False)
        channel_summary = (
            candidates.groupby(["frequency_mhz", "event_type", "antenna", "manual_review_priority"], dropna=False)
            .size()
            .rename("n_rows")
            .reset_index()
        )
        channel_summary.to_csv(source_out / "raw_occultation_candidate_channel_summary.csv", index=False)
        triage_plots = [_plot_score_summary(candidates, source_out, spec.source)]
        candidate_plot = _plot_candidate_grid(shortlist if not shortlist.empty else candidates, clean, source_out, spec.source, float(args.window_s), int(args.top_plot_panels))
        if candidate_plot is not None:
            triage_plots.append(candidate_plot)
        triage_config = {
            "clean": str(CLEAN),
            "events": str(spec.events_path),
            "source": spec.source,
            "family": spec.family,
            "event_types": sorted(event_types),
            "antennas": [str(args.antenna)],
            "frequencies": sorted(frequencies),
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
            "shortlist_per_group": int(args.shortlist_per_group),
            "use_existing_valid": True,
            "n_scored_rows": int(len(candidates)),
            "n_flagged_rows": int(len(flagged)),
            "n_shortlist_rows": int(len(shortlist)),
            "software_versions": software_versions(),
        }
        write_json(source_out / "raw_triage_run_config.json", triage_config)
        write_triage_report(source_out, spec.source, candidates, shortlist, triage_config, triage_plots)

        selected = candidates[
            candidates["source_name"].astype(str).str.lower().eq(spec.source)
            & candidates["antenna"].astype(str).eq(str(args.antenna))
            & candidates["manual_review_priority"].astype(str).eq("high_priority")
        ].copy()
        selected.to_csv(source_out / "structure_selected_candidate_rows.csv", index=False)
        stack_plots: list[Path] = []
        stats = pd.DataFrame()
        status = pd.DataFrame()
        stack = pd.DataFrame()
        if not selected.empty:
            points_all = []
            status_all = []
            for center_mode in ["predicted_centered", "best_offset_centered"]:
                points, stat = collect_selected_profiles(
                    clean=clean,
                    candidates=selected,
                    window_s=float(args.window_s),
                    bin_s=60.0,
                    sideband_s=float(args.inner_s),
                    center_mode=center_mode,
                    use_existing_valid=True,
                )
                points_all.append(points)
                status_all.append(stat)
            points = pd.concat(points_all, ignore_index=True) if points_all else pd.DataFrame()
            status = pd.concat(status_all, ignore_index=True) if status_all else pd.DataFrame()
            stack = stack_profiles(points)
            stats = stack_statistics(stack, status)
            points.to_csv(source_out / "structure_selected_stack_points.csv", index=False)
            status.to_csv(source_out / "structure_selected_event_status.csv", index=False)
            stack.to_csv(source_out / "structure_selected_stack_summary.csv", index=False)
            stats.to_csv(source_out / "structure_selected_stack_statistics.csv", index=False)
            if not stack.empty:
                for center_mode in ["predicted_centered", "best_offset_centered"]:
                    stack_plots.append(plot_stack_grid(stack, stats, spec.source, center_mode, source_out))
        else:
            pd.DataFrame().to_csv(source_out / "structure_selected_stack_points.csv", index=False)
            pd.DataFrame().to_csv(source_out / "structure_selected_event_status.csv", index=False)
            pd.DataFrame().to_csv(source_out / "structure_selected_stack_summary.csv", index=False)
            pd.DataFrame().to_csv(source_out / "structure_selected_stack_statistics.csv", index=False)

        stack_config = {
            "clean": str(CLEAN),
            "candidates": str(source_out / "raw_occultation_candidate_scores.csv"),
            "source": spec.source,
            "antenna": str(args.antenna),
            "priorities": ["high_priority"],
            "window_s": float(args.window_s),
            "bin_s": 60.0,
            "sideband_s": float(args.inner_s),
            "n_selected_candidate_rows": int(len(selected)),
            "software_versions": software_versions(),
        }
        write_json(source_out / "structure_selected_stack_run_config.json", stack_config)
        write_stack_report(source_out, spec.source, selected, status, stats, stack_plots, stack_config)

        run_rows.append(
            {
                "source": spec.source,
                "family": spec.family,
                "event_rows_scored": int(len(candidates)),
                "high_priority_rows": int(len(selected)),
                "shortlist_rows": int(len(shortlist)),
                "stack_stat_rows": int(len(stats)),
                "output_dir": str(source_out),
                "stack_report": str(source_out / f"{spec.source}_lower_v_structure_selected_stack_report.md"),
            }
        )
        print(f"{spec.source}: scored={len(candidates)} high_priority={len(selected)} stats={len(stats)}")

    config = {
        "sources": [s.source for s in available_specs],
        "antenna": str(args.antenna),
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
        "shortlist_per_group": int(args.shortlist_per_group),
        "software_versions": software_versions(),
    }
    write_json(out_dir / "run_config.json", config)
    _write_source_index(out_dir, run_rows, config)
    print(out_dir / "all_source_structure_selected_stack_index.md")


if __name__ == "__main__":
    main()
