"""Command-line pipeline stages."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .blind import blind_changepoints, cluster_blind_constraints
from .bursts import search_burst_events
from .controls import injection_recovery, injection_recovery_grid, negative_control_event_ensemble, randomized_event_times, subsample_control_events, time_reversed_events
from .detection import StepFitConfig, run_matched_filter, run_stepfit_detections
from .events import predict_events
from .ingest import IngestOptions, ingest_csv
from .jobs import require_interactive_mode
from .plotting import plot_event_window, plot_stack
from .reporting import write_control_validation_report, write_methodology_report
from .scoring import ScoreConfig, score_detections
from .sources import filter_sources, load_source_list
from .stacking import aligned_profiles, stack_profiles
from .table_io import read_table, write_table
from .util import ensure_dir, software_versions, write_json


def _list_arg(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    out: list[str] = []
    for value in values:
        out.extend([v.strip() for v in value.split(",") if v.strip()])
    return out or None


def cmd_ingest(args: argparse.Namespace) -> dict[str, str]:
    require_interactive_mode(args.run_mode)
    out = ensure_dir(args.output_dir)
    clean, report = ingest_csv(
        args.data,
        IngestOptions(
            start_time=args.start,
            end_time=args.end,
            value_columns=tuple(args.value_columns),
            gap_factor=args.gap_factor,
            artifact_sigma=args.artifact_sigma,
        ),
    )
    clean_path = out / "cleaned_timeseries.csv"
    report_path = out / "validation_report.csv"
    write_table(clean, clean_path, index=False)
    write_table(report, report_path, index=False)
    write_json(out / "run_metadata.json", {"stage": "ingest", "args": vars(args), "versions": software_versions()})
    return {"cleaned_timeseries": str(clean_path), "validation_report": str(report_path)}


def cmd_predict(args: argparse.Namespace) -> dict[str, str]:
    require_interactive_mode(args.run_mode)
    out = ensure_dir(args.output_dir)
    clean = read_table(args.cleaned, parse_dates=["time"])
    source_table = load_source_list(args.sources)
    sources = filter_sources(source_table, _list_arg(args.source))
    limb_exclusion_names = _list_arg(getattr(args, "exclude_limb_source", None))
    limb_exclusion_sources = filter_sources(source_table, limb_exclusion_names) if limb_exclusion_names else None
    freqs = [int(v) for v in _list_arg(args.frequency) or []] or None
    ants = _list_arg(args.antenna)
    events, states = predict_events(
        clean,
        sources,
        target_frame=args.frame,
        equinox=args.equinox,
        ephemeris=args.ephemeris,
        max_gap_seconds=args.max_gap_seconds,
        prediction_cadence_seconds=getattr(args, "prediction_cadence_seconds", None),
        frequencies=freqs,
        antennas=ants,
        limb_exclusion_sources_df=limb_exclusion_sources,
        limb_exclusion_deg=getattr(args, "exclude_limb_deg", None),
    )
    events_path = out / "predicted_events.csv"
    states_path = out / "limb_visibility_states.csv"
    write_table(events, events_path, index=False)
    write_table(states, states_path, index=False)
    write_json(out / "run_metadata_predict.json", {"stage": "predict", "args": vars(args), "versions": software_versions()})
    return {"predicted_events": str(events_path), "limb_states": str(states_path)}


def cmd_detect(args: argparse.Namespace) -> dict[str, str]:
    require_interactive_mode(args.run_mode)
    out = ensure_dir(args.output_dir)
    clean = read_table(args.cleaned, parse_dates=["time"])
    events = read_table(args.events, parse_dates=["predicted_event_time"])
    config = StepFitConfig(
        window_seconds=args.window_seconds,
        baseline_order=args.baseline_order,
        min_samples_per_side=args.min_samples_per_side,
        smooth_seconds=args.smooth_seconds,
        timing_grid_seconds=tuple(float(v) for v in args.timing_offsets),
    )
    step = run_stepfit_detections(clean, events, config)
    matched = run_matched_filter(clean, events, window_seconds=args.window_seconds, baseline_order=args.baseline_order)
    step_path = out / "per_event_stepfit_detections.csv"
    matched_path = out / "per_event_matched_filter.csv"
    write_table(step, step_path, index=False)
    write_table(matched, matched_path, index=False)
    if args.plots and not events.empty:
        plot_dir = ensure_dir(out / "plots")
        for i, (_, ev) in enumerate(events.head(args.max_plots).iterrows()):
            plot_event_window(clean, ev, plot_dir / f"event_{i:04d}.png", args.window_seconds)
    write_json(out / "run_metadata_detect.json", {"stage": "detect", "args": vars(args), "versions": software_versions()})
    return {"stepfit_detections": str(step_path), "matched_filter": str(matched_path)}


def cmd_stack(args: argparse.Namespace) -> dict[str, str]:
    require_interactive_mode(args.run_mode)
    out = ensure_dir(args.output_dir)
    clean = read_table(args.cleaned, parse_dates=["time"])
    events = read_table(args.events, parse_dates=["predicted_event_time"])
    profiles = aligned_profiles(
        clean,
        events,
        window_seconds=args.window_seconds,
        bin_seconds=args.bin_seconds,
        baseline_mode=getattr(args, "baseline_mode", "sideband_linear"),
        sideband_exclusion_seconds=getattr(args, "sideband_exclusion_seconds", 120.0),
    )
    stacked, summary = stack_profiles(profiles, n_bootstrap=args.bootstrap)
    profiles_path = out / "event_aligned_profiles.csv"
    stacked_path = out / "stacked_event_profiles.csv"
    summary_path = out / "stacked_detection_summary.csv"
    write_table(profiles, profiles_path, index=False)
    write_table(stacked, stacked_path, index=False)
    write_table(summary, summary_path, index=False)
    if args.plots:
        plot_stack(stacked, out / "stacked_profiles.png")
    return {"profiles": str(profiles_path), "stacked_profiles": str(stacked_path), "stack_summary": str(summary_path)}


def cmd_blind(args: argparse.Namespace) -> dict[str, str]:
    require_interactive_mode(args.run_mode)
    out = ensure_dir(args.output_dir)
    clean = read_table(args.cleaned, parse_dates=["time"])
    candidates = blind_changepoints(clean, window_samples=args.window_samples, snr_threshold=args.snr_threshold)
    clusters = cluster_blind_constraints(candidates)
    cand_path = out / "blind_candidate_event_catalog.csv"
    cluster_path = out / "blind_candidate_clusters.csv"
    write_table(candidates, cand_path, index=False)
    write_table(clusters, cluster_path, index=False)
    return {"blind_candidates": str(cand_path), "blind_clusters": str(cluster_path)}


def cmd_burst_search(args: argparse.Namespace) -> dict[str, str]:
    require_interactive_mode(args.run_mode)
    out = ensure_dir(args.output_dir)
    clean = read_table(args.cleaned, parse_dates=["time"])
    events = read_table(args.events, parse_dates=["predicted_event_time"])
    bursts = search_burst_events(
        clean,
        events,
        window_seconds=args.window_seconds,
        z_threshold=args.z_threshold,
        min_cluster_samples=args.min_cluster_samples,
    )
    burst_path = out / "burst_event_catalog.csv"
    write_table(bursts, burst_path, index=False)
    write_json(out / "run_metadata_burst_search.json", {"stage": "burst-search", "args": vars(args), "versions": software_versions()})
    return {"burst_event_catalog": str(burst_path)}


def cmd_validate(args: argparse.Namespace) -> dict[str, str]:
    require_interactive_mode(args.run_mode)
    out = ensure_dir(args.output_dir)
    clean = read_table(args.cleaned, parse_dates=["time"])
    events = read_table(args.events, parse_dates=["predicted_event_time"])
    config = StepFitConfig(window_seconds=args.window_seconds, smooth_seconds=args.smooth_seconds)
    amp = [float(v) for v in args.amplitudes]
    inj = injection_recovery(clean, events, amp, config)
    grid = injection_recovery_grid(
        clean,
        events,
        amp,
        window_seconds=[float(v) for v in args.injection_windows],
        smooth_seconds=args.smooth_seconds,
    )
    rand = randomized_event_times(events, clean["time"].min(), clean["time"].max(), seed=args.seed)
    rev = time_reversed_events(events)
    ensemble = negative_control_event_ensemble(
        events,
        clean,
        n_random=args.n_random_controls,
        seed=args.seed,
        exclusion_seconds=args.random_exclusion_seconds,
    )
    ensemble = subsample_control_events(ensemble, getattr(args, "max_control_events_per_group", None), seed=args.seed)
    inj_path = out / "injection_recovery_summary.csv"
    grid_path = out / "injection_recovery_grid.csv"
    rand_path = out / "negative_control_randomized_events.csv"
    rev_path = out / "negative_control_time_reversed_events.csv"
    ensemble_path = out / "negative_control_event_ensemble.csv"
    write_table(inj, inj_path, index=False)
    write_table(grid, grid_path, index=False)
    write_table(rand, rand_path, index=False)
    write_table(rev, rev_path, index=False)
    write_table(ensemble, ensemble_path, index=False)
    return {
        "injection_recovery": str(inj_path),
        "injection_recovery_grid": str(grid_path),
        "randomized_events": str(rand_path),
        "time_reversed_events": str(rev_path),
        "negative_control_ensemble": str(ensemble_path),
    }


def cmd_score(args: argparse.Namespace) -> dict[str, str]:
    require_interactive_mode(args.run_mode)
    out = ensure_dir(args.output_dir)
    clean = read_table(args.cleaned, parse_dates=["time"])
    real_step = read_table(args.stepfit, parse_dates=["predicted_event_time"])
    real_matched = read_table(args.matched, parse_dates=["predicted_event_time"])
    controls = read_table(args.control_events, parse_dates=["predicted_event_time"])
    injection_grid = read_table(args.injection_grid) if args.injection_grid else None
    cfg = ScoreConfig(
        window_seconds=args.window_seconds,
        baseline_order=args.baseline_order,
        min_samples_per_side=args.min_samples_per_side,
        smooth_seconds=args.smooth_seconds,
        timing_grid_seconds=tuple(float(v) for v in args.timing_offsets),
        null_percentile_threshold=args.null_percentile_threshold,
        min_abs_snr=args.min_abs_snr,
        clean_fraction_threshold=args.clean_fraction_threshold,
        empirical_control_types=tuple(args.empirical_control_types),
    )
    scored, null_summary, null_step, null_matched = score_detections(
        clean,
        real_step,
        real_matched,
        controls,
        injection_grid=injection_grid,
        config=cfg,
    )
    scored_path = out / "scored_detections.csv"
    null_summary_path = out / "null_snr_summary.csv"
    null_step_path = out / "control_stepfit_detections.csv"
    null_matched_path = out / "control_matched_filter.csv"
    write_table(scored, scored_path, index=False)
    write_table(null_summary, null_summary_path, index=False)
    write_table(null_step, null_step_path, index=False)
    write_table(null_matched, null_matched_path, index=False)
    write_json(out / "run_metadata_score.json", {"stage": "score", "args": vars(args), "versions": software_versions()})
    return {
        "scored_detections": str(scored_path),
        "null_snr_summary": str(null_summary_path),
        "control_stepfit_detections": str(null_step_path),
        "control_matched_filter": str(null_matched_path),
    }


def cmd_run_smoke(args: argparse.Namespace) -> dict[str, str]:
    require_interactive_mode(args.run_mode)
    root = ensure_dir(args.output_dir)
    outputs: dict[str, str] = {}
    ingest_payload = vars(args).copy()
    ingest_payload.update({"output_dir": str(root / "01_ingest"), "value_columns": args.value_columns, "gap_factor": 3.0, "artifact_sigma": 12.0})
    ingest_args = argparse.Namespace(**ingest_payload)
    outputs.update(cmd_ingest(ingest_args))
    pred_args = argparse.Namespace(
        cleaned=outputs["cleaned_timeseries"],
        sources=args.sources,
        source=args.source,
        frequency=args.frequency,
        antenna=args.antenna,
        output_dir=str(root / "02_events"),
        frame=args.frame,
        equinox=args.equinox,
        ephemeris=args.ephemeris,
        max_gap_seconds=args.max_gap_seconds,
        prediction_cadence_seconds=getattr(args, "prediction_cadence_seconds", None),
        exclude_limb_source=getattr(args, "exclude_limb_source", None),
        exclude_limb_deg=getattr(args, "exclude_limb_deg", None),
        run_mode=args.run_mode,
    )
    outputs.update(cmd_predict(pred_args))
    detect_args = argparse.Namespace(
        cleaned=outputs["cleaned_timeseries"],
        events=outputs["predicted_events"],
        output_dir=str(root / "03_detection"),
        window_seconds=args.window_seconds,
        baseline_order=1,
        min_samples_per_side=2,
        smooth_seconds=args.smooth_seconds,
        timing_offsets=[-60.0, 0.0, 60.0],
        plots=True,
        max_plots=3,
        run_mode=args.run_mode,
    )
    outputs.update(cmd_detect(detect_args))
    stack_args = argparse.Namespace(
        cleaned=outputs["cleaned_timeseries"],
        events=outputs["predicted_events"],
        output_dir=str(root / "04_stack"),
        window_seconds=args.window_seconds,
        bin_seconds=60.0,
        bootstrap=50,
        baseline_mode=getattr(args, "baseline_mode", "sideband_linear"),
        sideband_exclusion_seconds=getattr(args, "sideband_exclusion_seconds", 120.0),
        plots=True,
        run_mode=args.run_mode,
    )
    outputs.update(cmd_stack(stack_args))
    blind_args = argparse.Namespace(cleaned=outputs["cleaned_timeseries"], output_dir=str(root / "05_blind"), window_samples=4, snr_threshold=8.0, run_mode=args.run_mode)
    outputs.update(cmd_blind(blind_args))
    val_args = argparse.Namespace(
        cleaned=outputs["cleaned_timeseries"],
        events=outputs["predicted_events"],
        output_dir=str(root / "06_validation"),
        amplitudes=["1000", "5000"],
        injection_windows=[args.window_seconds],
        n_random_controls=25,
        max_control_events_per_group=None,
        random_exclusion_seconds=args.window_seconds,
        window_seconds=args.window_seconds,
        smooth_seconds=args.smooth_seconds,
        seed=12345,
        run_mode=args.run_mode,
    )
    outputs.update(cmd_validate(val_args))
    score_args = argparse.Namespace(
        cleaned=outputs["cleaned_timeseries"],
        stepfit=outputs["stepfit_detections"],
        matched=outputs["matched_filter"],
        control_events=outputs["negative_control_ensemble"],
        injection_grid=outputs["injection_recovery_grid"],
        output_dir=str(root / "07_scoring"),
        window_seconds=args.window_seconds,
        baseline_order=1,
        min_samples_per_side=2,
        smooth_seconds=args.smooth_seconds,
        timing_offsets=[-60.0, 0.0, 60.0],
        null_percentile_threshold=99.0,
        min_abs_snr=3.0,
        clean_fraction_threshold=0.8,
        empirical_control_types=["randomized_time"],
        run_mode=args.run_mode,
    )
    outputs.update(cmd_score(score_args))
    report_path = root / "ryle_vonberg_pipeline_report.md"
    write_methodology_report(report_path, outputs)
    outputs["report"] = str(report_path)
    return outputs


def cmd_run_control_survey(args: argparse.Namespace) -> dict[str, str]:
    """Run the Earth/Sun control-validation workflow over an extended date range."""
    require_interactive_mode(args.run_mode)
    root = ensure_dir(args.output_dir)
    outputs: dict[str, str] = {}
    ingest_args = argparse.Namespace(
        data=args.data,
        start=args.start,
        end=args.end,
        output_dir=str(root / "01_ingest"),
        value_columns=args.value_columns,
        gap_factor=args.gap_factor,
        artifact_sigma=args.artifact_sigma,
        run_mode=args.run_mode,
    )
    outputs.update(cmd_ingest(ingest_args))
    pred_args = argparse.Namespace(
        cleaned=outputs["cleaned_timeseries"],
        sources=args.sources,
        source=args.source,
        frequency=args.frequency,
        antenna=args.antenna,
        output_dir=str(root / "02_events"),
        frame=args.frame,
        equinox=args.equinox,
        ephemeris=args.ephemeris,
        max_gap_seconds=args.max_gap_seconds,
        prediction_cadence_seconds=args.prediction_cadence_seconds,
        exclude_limb_source=getattr(args, "exclude_limb_source", None),
        exclude_limb_deg=getattr(args, "exclude_limb_deg", None),
        run_mode=args.run_mode,
    )
    outputs.update(cmd_predict(pred_args))
    predicted_events = read_table(outputs["predicted_events"], parse_dates=["predicted_event_time"])
    stack_events_path = outputs["predicted_events"]
    if args.max_stack_events_per_group:
        stack_events = subsample_control_events(predicted_events, args.max_stack_events_per_group, seed=args.seed)
        stack_events_path = str(root / "02_events" / "stack_input_events.csv")
        write_table(stack_events, stack_events_path, index=False)
        outputs["stack_input_events"] = stack_events_path
    validation_events_path = outputs["predicted_events"]
    if args.max_validation_events_per_group:
        validation_events = subsample_control_events(predicted_events, args.max_validation_events_per_group, seed=args.seed + 101)
        validation_events_path = str(root / "02_events" / "validation_input_events.csv")
        write_table(validation_events, validation_events_path, index=False)
        outputs["validation_input_events"] = validation_events_path
    detect_args = argparse.Namespace(
        cleaned=outputs["cleaned_timeseries"],
        events=outputs["predicted_events"],
        output_dir=str(root / "03_detection"),
        window_seconds=args.window_seconds,
        baseline_order=args.baseline_order,
        min_samples_per_side=args.min_samples_per_side,
        smooth_seconds=args.smooth_seconds,
        timing_offsets=args.timing_offsets,
        plots=args.initial_plots,
        max_plots=args.max_plots,
        run_mode=args.run_mode,
    )
    outputs.update(cmd_detect(detect_args))
    stack_args = argparse.Namespace(
        cleaned=outputs["cleaned_timeseries"],
        events=stack_events_path,
        output_dir=str(root / "04_stack"),
        window_seconds=args.window_seconds,
        bin_seconds=args.stack_bin_seconds,
        bootstrap=args.bootstrap,
        baseline_mode=args.baseline_mode,
        sideband_exclusion_seconds=args.sideband_exclusion_seconds,
        plots=True,
        run_mode=args.run_mode,
    )
    outputs.update(cmd_stack(stack_args))
    if args.include_blind:
        blind_args = argparse.Namespace(
            cleaned=outputs["cleaned_timeseries"],
            output_dir=str(root / "05_blind"),
            window_samples=args.blind_window_samples,
            snr_threshold=args.blind_snr_threshold,
            run_mode=args.run_mode,
        )
        outputs.update(cmd_blind(blind_args))
    val_args = argparse.Namespace(
        cleaned=outputs["cleaned_timeseries"],
        events=validation_events_path,
        output_dir=str(root / "06_validation"),
        amplitudes=args.amplitudes,
        injection_windows=args.injection_windows,
        n_random_controls=args.n_random_controls,
        max_control_events_per_group=args.max_control_events_per_group,
        random_exclusion_seconds=args.random_exclusion_seconds,
        window_seconds=args.window_seconds,
        smooth_seconds=args.smooth_seconds,
        seed=args.seed,
        run_mode=args.run_mode,
    )
    outputs.update(cmd_validate(val_args))
    score_args = argparse.Namespace(
        cleaned=outputs["cleaned_timeseries"],
        stepfit=outputs["stepfit_detections"],
        matched=outputs["matched_filter"],
        control_events=outputs["negative_control_ensemble"],
        injection_grid=outputs["injection_recovery_grid"],
        output_dir=str(root / "07_scoring"),
        window_seconds=args.window_seconds,
        baseline_order=args.baseline_order,
        min_samples_per_side=args.min_samples_per_side,
        smooth_seconds=args.smooth_seconds,
        timing_offsets=args.timing_offsets,
        null_percentile_threshold=args.null_percentile_threshold,
        min_abs_snr=args.min_abs_snr,
        clean_fraction_threshold=args.clean_fraction_threshold,
        empirical_control_types=args.empirical_control_types,
        run_mode=args.run_mode,
    )
    outputs.update(cmd_score(score_args))

    clean = read_table(outputs["cleaned_timeseries"], parse_dates=["time"])
    scored = read_table(outputs["scored_detections"], parse_dates=["predicted_event_time"])
    stack_summary = read_table(outputs["stack_summary"])
    review_dir = ensure_dir(root / "08_review_plots")
    review_cols = ["source_name", "event_type", "predicted_event_time", "frequency_band", "frequency_mhz", "antenna", "best_empirical_p", "best_abs_snr", "detection_grade"]
    top = scored.sort_values(["best_empirical_p", "best_abs_snr"], ascending=[True, False]).head(args.review_plots)
    write_table(top[[c for c in review_cols if c in top.columns]], review_dir / "review_plot_index.csv", index=False)
    for i, (_, ev) in enumerate(top.iterrows()):
        plot_event_window(clean, ev, review_dir / f"top_scored_event_{i:04d}.png", args.window_seconds)
    outputs["review_plot_index"] = str(review_dir / "review_plot_index.csv")
    outputs["review_plots"] = str(review_dir)

    report_path = root / "earth_sun_control_validation_report.md"
    write_control_validation_report(report_path, outputs, scored, stack_summary, vars(args))
    outputs["control_validation_report"] = str(report_path)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RAE-2 Ryle-Vonberg lunar-occultation pipeline")
    sub = p.add_subparsers(dest="command", required=True)

    def common_run(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--run-mode", default="local", help="Allowed: local, interactive, salloc, srun-interactive. Batch modes fail.")

    def common_io(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--data", required=True)
        sp.add_argument("--start")
        sp.add_argument("--end")
        sp.add_argument("--output-dir", required=True)
        sp.add_argument("--value-columns", nargs="+", default=["rv1_coarse", "rv2_coarse"])
        common_run(sp)

    sp = sub.add_parser("ingest")
    common_io(sp)
    sp.add_argument("--gap-factor", type=float, default=3.0)
    sp.add_argument("--artifact-sigma", type=float, default=12.0)
    sp.set_defaults(func=cmd_ingest)

    sp = sub.add_parser("predict")
    sp.add_argument("--cleaned", required=True)
    sp.add_argument("--sources", required=True)
    sp.add_argument("--source", nargs="*")
    sp.add_argument("--frequency", nargs="*")
    sp.add_argument("--antenna", nargs="*")
    sp.add_argument("--output-dir", required=True)
    sp.add_argument("--frame", default="fk4")
    sp.add_argument("--equinox", default="B1950")
    sp.add_argument("--ephemeris", default="builtin")
    sp.add_argument("--max-gap-seconds", type=float, default=240.0)
    sp.add_argument("--prediction-cadence-seconds", type=float)
    sp.add_argument("--exclude-limb-source", nargs="*", help="Drop events when these sources are close to the lunar limb.")
    sp.add_argument("--exclude-limb-deg", type=float, help="Absolute limb-angle threshold in degrees for --exclude-limb-source.")
    common_run(sp)
    sp.set_defaults(func=cmd_predict)

    sp = sub.add_parser("detect")
    sp.add_argument("--cleaned", required=True)
    sp.add_argument("--events", required=True)
    sp.add_argument("--output-dir", required=True)
    sp.add_argument("--window-seconds", type=float, default=600.0)
    sp.add_argument("--baseline-order", type=int, default=1)
    sp.add_argument("--min-samples-per-side", type=int, default=4)
    sp.add_argument("--smooth-seconds", type=float, default=0.0)
    sp.add_argument("--timing-offsets", nargs="+", type=float, default=[-120, -60, -30, 0, 30, 60, 120])
    sp.add_argument("--plots", action="store_true")
    sp.add_argument("--max-plots", type=int, default=10)
    common_run(sp)
    sp.set_defaults(func=cmd_detect)

    sp = sub.add_parser("stack")
    sp.add_argument("--cleaned", required=True)
    sp.add_argument("--events", required=True)
    sp.add_argument("--output-dir", required=True)
    sp.add_argument("--window-seconds", type=float, default=600.0)
    sp.add_argument("--bin-seconds", type=float, default=30.0)
    sp.add_argument("--bootstrap", type=int, default=200)
    sp.add_argument("--max-stack-events-per-group", type=int, default=200)
    sp.add_argument("--baseline-mode", default="sideband_linear", choices=["linear_all", "constant_all", "sideband_linear", "sideband_constant", "joint_step_linear", "pre_event_anchor"])
    sp.add_argument("--sideband-exclusion-seconds", type=float, default=120.0)
    sp.add_argument("--plots", action="store_true")
    common_run(sp)
    sp.set_defaults(func=cmd_stack)

    sp = sub.add_parser("blind")
    sp.add_argument("--cleaned", required=True)
    sp.add_argument("--output-dir", required=True)
    sp.add_argument("--window-samples", type=int, default=8)
    sp.add_argument("--snr-threshold", type=float, default=5.0)
    common_run(sp)
    sp.set_defaults(func=cmd_blind)

    sp = sub.add_parser("burst-search")
    sp.add_argument("--cleaned", required=True)
    sp.add_argument("--events", required=True)
    sp.add_argument("--output-dir", required=True)
    sp.add_argument("--window-seconds", type=float, default=900.0)
    sp.add_argument("--z-threshold", type=float, default=8.0)
    sp.add_argument("--min-cluster-samples", type=int, default=1)
    common_run(sp)
    sp.set_defaults(func=cmd_burst_search)

    sp = sub.add_parser("validate")
    sp.add_argument("--cleaned", required=True)
    sp.add_argument("--events", required=True)
    sp.add_argument("--output-dir", required=True)
    sp.add_argument("--amplitudes", nargs="+", default=["1000", "5000", "10000"])
    sp.add_argument("--injection-windows", nargs="+", type=float, default=[300.0, 600.0, 1200.0])
    sp.add_argument("--n-random-controls", type=int, default=100)
    sp.add_argument("--max-control-events-per-group", type=int)
    sp.add_argument("--random-exclusion-seconds", type=float, default=900.0)
    sp.add_argument("--window-seconds", type=float, default=600.0)
    sp.add_argument("--smooth-seconds", type=float, default=0.0)
    sp.add_argument("--seed", type=int, default=12345)
    sp.add_argument("--max-validation-events-per-group", type=int, default=100)
    common_run(sp)
    sp.set_defaults(func=cmd_validate)

    sp = sub.add_parser("score-detections")
    sp.add_argument("--cleaned", required=True)
    sp.add_argument("--stepfit", required=True)
    sp.add_argument("--matched", required=True)
    sp.add_argument("--control-events", required=True)
    sp.add_argument("--injection-grid")
    sp.add_argument("--output-dir", required=True)
    sp.add_argument("--window-seconds", type=float, default=600.0)
    sp.add_argument("--baseline-order", type=int, default=1)
    sp.add_argument("--min-samples-per-side", type=int, default=4)
    sp.add_argument("--smooth-seconds", type=float, default=0.0)
    sp.add_argument("--timing-offsets", nargs="+", type=float, default=[-60.0, 0.0, 60.0])
    sp.add_argument("--null-percentile-threshold", type=float, default=99.0)
    sp.add_argument("--min-abs-snr", type=float, default=3.0)
    sp.add_argument("--clean-fraction-threshold", type=float, default=0.8)
    sp.add_argument("--empirical-control-types", nargs="+", default=["randomized_time"])
    common_run(sp)
    sp.set_defaults(func=cmd_score)

    sp = sub.add_parser("run-smoke")
    common_io(sp)
    sp.add_argument("--sources", default=str(Path(__file__).resolve().parents[1] / "configs" / "bright_sources.csv"))
    sp.add_argument("--source", nargs="*", default=["jupiter"])
    sp.add_argument("--frequency", nargs="*", default=["4"])
    sp.add_argument("--antenna", nargs="*", default=["rv2_coarse"])
    sp.add_argument("--frame", default="fk4")
    sp.add_argument("--equinox", default="B1950")
    sp.add_argument("--ephemeris", default="builtin")
    sp.add_argument("--max-gap-seconds", type=float, default=600.0)
    sp.add_argument("--window-seconds", type=float, default=600.0)
    sp.add_argument("--smooth-seconds", type=float, default=0.0)
    sp.add_argument("--exclude-limb-source", nargs="*")
    sp.add_argument("--exclude-limb-deg", type=float)
    sp.set_defaults(func=cmd_run_smoke)

    sp = sub.add_parser("run-control-survey")
    common_io(sp)
    sp.add_argument("--sources", default=str(Path(__file__).resolve().parents[1] / "configs" / "bright_sources.csv"))
    sp.add_argument("--source", nargs="*", default=["earth", "sun"])
    sp.add_argument("--frequency", nargs="*", default=[str(v) for v in range(1, 10)])
    sp.add_argument("--antenna", nargs="*", default=["rv1_coarse", "rv2_coarse"])
    sp.add_argument("--frame", default="fk4")
    sp.add_argument("--equinox", default="B1950")
    sp.add_argument("--ephemeris", default="builtin")
    sp.add_argument("--max-gap-seconds", type=float, default=600.0)
    sp.add_argument("--prediction-cadence-seconds", type=float, default=300.0)
    sp.add_argument("--exclude-limb-source", nargs="*")
    sp.add_argument("--exclude-limb-deg", type=float)
    sp.add_argument("--window-seconds", type=float, default=600.0)
    sp.add_argument("--baseline-order", type=int, default=1)
    sp.add_argument("--min-samples-per-side", type=int, default=4)
    sp.add_argument("--smooth-seconds", type=float, default=0.0)
    sp.add_argument("--timing-offsets", nargs="+", type=float, default=[-120.0, -60.0, -30.0, 0.0, 30.0, 60.0, 120.0])
    sp.add_argument("--stack-bin-seconds", type=float, default=60.0)
    sp.add_argument("--baseline-mode", default="sideband_linear", choices=["linear_all", "constant_all", "sideband_linear", "sideband_constant", "joint_step_linear", "pre_event_anchor"])
    sp.add_argument("--sideband-exclusion-seconds", type=float, default=120.0)
    sp.add_argument("--bootstrap", type=int, default=200)
    sp.add_argument("--max-stack-events-per-group", type=int, default=200)
    sp.add_argument("--amplitudes", nargs="+", default=["1000", "5000", "10000"])
    sp.add_argument("--injection-windows", nargs="+", type=float, default=[300.0, 600.0, 1200.0])
    sp.add_argument("--n-random-controls", type=int, default=100)
    sp.add_argument("--max-control-events-per-group", type=int, default=200)
    sp.add_argument("--max-validation-events-per-group", type=int, default=100)
    sp.add_argument("--random-exclusion-seconds", type=float, default=900.0)
    sp.add_argument("--seed", type=int, default=12345)
    sp.add_argument("--null-percentile-threshold", type=float, default=99.0)
    sp.add_argument("--min-abs-snr", type=float, default=3.0)
    sp.add_argument("--clean-fraction-threshold", type=float, default=0.8)
    sp.add_argument("--empirical-control-types", nargs="+", default=["randomized_time"])
    sp.add_argument("--initial-plots", action="store_true")
    sp.add_argument("--max-plots", type=int, default=10)
    sp.add_argument("--review-plots", type=int, default=20)
    sp.add_argument("--include-blind", action="store_true")
    sp.add_argument("--blind-window-samples", type=int, default=8)
    sp.add_argument("--blind-snr-threshold", type=float, default=8.0)
    sp.add_argument("--gap-factor", type=float, default=3.0)
    sp.add_argument("--artifact-sigma", type=float, default=12.0)
    sp.set_defaults(func=cmd_run_control_survey, start="1974-11-01 00:00:00", end="1976-01-02 00:00:00")
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    outputs = args.func(args)
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
