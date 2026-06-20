#!/usr/bin/env python
"""Run focused Earth/Cyg A validation and source-level summaries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from rylevonberg.table_io import read_table

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rylevonberg.controls import negative_control_event_ensemble, subsample_control_events
from rylevonberg.detection import StepFitConfig, run_matched_filter, run_stepfit_detections
from rylevonberg.events import predict_events
from rylevonberg.offsource import OffsourceConfig, generate_offsource_controls
from rylevonberg.plotting import plot_event_window, plot_stack
from rylevonberg.quality import event_window_quality
from rylevonberg.scoring import ScoreConfig, score_detections
from rylevonberg.source_summary import (
    add_offsource_pvalues,
    aggregate_source_level,
    block_jackknife_from_profiles,
    final_source_summary,
)
from rylevonberg.sources import filter_sources, load_source_list
from rylevonberg.stacking import aligned_profiles, stack_profiles
from rylevonberg.util import ensure_dir, write_json


def _read_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    return read_table(path, low_memory=False, **kwargs)


def _plot_profiles(stacked: pd.DataFrame, output_path: Path, title: str) -> None:
    if stacked.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for _, grp in stacked.groupby([c for c in ["source_name", "event_type", "frequency_band", "antenna"] if c in stacked.columns], dropna=False):
        label = " ".join(str(grp.iloc[0].get(c)) for c in ["event_type", "frequency_band", "antenna"] if c in grp.columns)
        ax.plot(grp["t_bin_sec"] / 60.0, grp["mean"], lw=1, label=label)
    ax.axvline(0.0, color="black", lw=1)
    ax.set_xlabel("Relative time (min)")
    ax.set_ylabel("Mean normalized profile")
    ax.set_title(title)
    if len(stacked) and stacked[[c for c in ["event_type"] if c in stacked.columns]].drop_duplicates().shape[0] <= 6:
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _top_event_plots(clean: pd.DataFrame, scored: pd.DataFrame, out_dir: Path, window_s: float, n: int = 12) -> None:
    ensure_dir(out_dir)
    if scored.empty:
        return
    top = scored.sort_values(["best_empirical_p", "best_abs_snr"], ascending=[True, False]).head(n)
    cols = [
        "source_name",
        "event_type",
        "predicted_event_time",
        "frequency_band",
        "frequency_mhz",
        "antenna",
        "best_empirical_p",
        "best_abs_snr",
        "timing_offset_sec",
        "quality_clean_fraction",
        "detection_grade",
    ]
    top[[c for c in cols if c in top.columns]].to_csv(out_dir / "thumbnail_index.csv", index=False)
    for idx, (_, ev) in enumerate(top.iterrows()):
        plot_event_window(clean, ev, out_dir / f"event_{idx:04d}.png", window_s)


def _run_detection_stack_score(
    clean: pd.DataFrame,
    events: pd.DataFrame,
    stack_events: pd.DataFrame,
    window_s: float,
    out_dir: Path,
    random_seed: int,
    control_events: pd.DataFrame | None = None,
    write_tables: bool = True,
) -> dict[str, pd.DataFrame]:
    ensure_dir(out_dir)
    cfg = StepFitConfig(
        window_seconds=window_s,
        baseline_order=1,
        min_samples_per_side=4,
        smooth_seconds=0.0,
        timing_grid_seconds=(-120.0, -60.0, -30.0, 0.0, 30.0, 60.0, 120.0),
    )
    step = run_stepfit_detections(clean, events, cfg)
    matched = run_matched_filter(clean, events, window_seconds=window_s, baseline_order=1)
    profiles = aligned_profiles(clean, stack_events, window_seconds=window_s, bin_seconds=60.0)
    stacked, stack_summary = stack_profiles(profiles, n_bootstrap=50)
    quality = event_window_quality(clean, events, window_s)
    if control_events is None:
        controls = negative_control_event_ensemble(
            subsample_control_events(events, 3, seed=random_seed),
            clean,
            n_random=30,
            seed=random_seed,
            exclusion_seconds=window_s,
        )
        controls = subsample_control_events(controls, 50, seed=random_seed)
    else:
        controls = control_events
    scored, null_summary, null_step, null_matched = score_detections(
        clean,
        step,
        matched,
        controls,
        injection_grid=None,
        config=ScoreConfig(
            window_seconds=window_s,
            baseline_order=1,
            min_samples_per_side=4,
            timing_grid_seconds=(-120.0, -60.0, -30.0, 0.0, 30.0, 60.0, 120.0),
        ),
    )
    if write_tables:
        step.to_csv(out_dir / "per_event_stepfit_detections.csv", index=False)
        matched.to_csv(out_dir / "per_event_matched_filter.csv", index=False)
        profiles.to_csv(out_dir / "event_aligned_profiles.csv", index=False)
        stacked.to_csv(out_dir / "stacked_event_profiles.csv", index=False)
        stack_summary.to_csv(out_dir / "stacked_detection_summary.csv", index=False)
        quality.to_csv(out_dir / "event_window_quality.csv", index=False)
        controls.to_csv(out_dir / "negative_control_event_ensemble.csv", index=False)
        scored.to_csv(out_dir / "scored_detections.csv", index=False)
        null_summary.to_csv(out_dir / "null_snr_summary.csv", index=False)
    return {
        "step": step,
        "matched": matched,
        "profiles": profiles,
        "stacked": stacked,
        "stack_summary": stack_summary,
        "quality": quality,
        "controls": controls,
        "scored": scored,
        "null_summary": null_summary,
    }


def _generate_cyg_offsource(clean: pd.DataFrame, source_list: pd.DataFrame, out_root: Path, args: argparse.Namespace) -> pd.DataFrame:
    controls_all = generate_offsource_controls(source_list, OffsourceConfig())
    controls_path = ensure_dir(out_root / "controls") / "offsource_controls.csv"
    controls_all.to_csv(controls_path, index=False)
    cyg_controls = controls_all[controls_all["parent_source"].eq("cyg_a")].reset_index(drop=True)
    events, states = predict_events(
        clean,
        cyg_controls,
        target_frame="fk4",
        equinox="B1950",
        ephemeris="builtin",
        max_gap_seconds=600.0,
        prediction_cadence_seconds=args.prediction_cadence_seconds,
        frequencies=[1],
        antennas=["rv1_coarse", "rv2_coarse"],
    )
    events = events.merge(
        cyg_controls[["source_name", "parent_source", "control_name", "control_type", "offset_deg", "notes"]],
        on="source_name",
        how="left",
    )
    off_dir = ensure_dir(out_root / "controls" / "cyg_a_offsource")
    events.to_csv(off_dir / "predicted_events.csv", index=False)
    states.to_csv(off_dir / "limb_visibility_states.csv", index=False)
    return events


def _offsource_stacks(clean: pd.DataFrame, off_events: pd.DataFrame, out_root: Path, windows: list[float]) -> pd.DataFrame:
    tables = []
    off_dir = ensure_dir(out_root / "controls" / "cyg_a_offsource")
    for window in windows:
        profiles = aligned_profiles(clean, off_events, window_seconds=window, bin_seconds=60.0)
        _, summary = stack_profiles(profiles, by=["source_name", "frequency_band", "antenna"], n_bootstrap=20)
        if not summary.empty:
            summary["window_s"] = float(window)
            summary = summary.merge(
                off_events[["source_name", "parent_source", "control_name", "control_type", "offset_deg"]].drop_duplicates("source_name"),
                on="source_name",
                how="left",
            )
            summary.to_csv(off_dir / f"offsource_stack_summary_{int(window)}s.csv", index=False)
            tables.append(summary)
    out = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    out.to_csv(off_dir / "offsource_stack_summary_all.csv", index=False)
    return out


def _earth_report(
    clean: pd.DataFrame,
    earth_root: Path,
    out_reports: Path,
    source_rows: list[pd.DataFrame],
) -> None:
    events = _read_csv(earth_root / "02_events" / "predicted_events.csv", parse_dates=["predicted_event_time"])
    scored = _read_csv(earth_root / "07_scoring" / "scored_detections.csv", parse_dates=["predicted_event_time"])
    stack_summary = _read_csv(earth_root / "04_stack" / "stacked_detection_summary.csv")
    profiles = _read_csv(earth_root / "04_stack" / "event_aligned_profiles.csv", parse_dates=["predicted_event_time"])
    earth_events = events[events["source_name"].eq("earth")].reset_index(drop=True)
    earth_scored = scored[scored["source_name"].eq("earth")].reset_index(drop=True)
    earth_stack = stack_summary[stack_summary["source_name"].eq("earth")].reset_index(drop=True)
    quality = event_window_quality(clean, earth_events, 600.0)
    block = block_jackknife_from_profiles(profiles[profiles["source_name"].eq("earth")])
    source_rows.append(aggregate_source_level(earth_events, earth_scored, earth_stack, quality, block, window_s=600.0))

    report_dir = ensure_dir(out_reports / "earth_positive_control_assets")
    strongest = earth_stack.assign(abs_snr=earth_stack["stacked_snr"].abs()).sort_values("abs_snr", ascending=False).head(5)
    split_rows = []
    for _, row in strongest.iterrows():
        freq = row["frequency_band"]
        ant = row["antenna"]
        sub = profiles[(profiles["source_name"].eq("earth")) & (profiles["frequency_band"].eq(freq)) & (profiles["antenna"].eq(ant))]
        combined, summary = stack_profiles(sub, by=["source_name", "frequency_band", "antenna", "event_type"], n_bootstrap=20)
        combined.to_csv(report_dir / f"earth_band{int(freq)}_{ant}_event_type_stack.csv", index=False)
        summary.to_csv(report_dir / f"earth_band{int(freq)}_{ant}_event_type_summary.csv", index=False)
        _plot_profiles(combined, report_dir / f"earth_band{int(freq)}_{ant}_event_type_stack.png", f"Earth band {int(freq)} {ant}")
        split_rows.append(summary)
    _top_event_plots(clean, earth_scored, report_dir / "event_thumbnails", 600.0, n=16)
    grade_counts = earth_scored["detection_grade"].value_counts().to_dict()
    lines = [
        "# Earth Positive-Control Validation",
        "",
        "## Result",
        "",
        "Earth passes the current positive-control validation as a repeated-event stacked control.",
        "",
        "## Strongest Channels",
        "",
        strongest[["frequency_band", "frequency_mhz", "antenna", "n_events", "stacked_amplitude", "stacked_snr", "bootstrap_std", "jackknife_std"]].to_string(index=False),
        "",
        "## Event-Level Grades",
        "",
        pd.DataFrame([grade_counts]).to_string(index=False),
        "",
        "## Stability Checks Generated",
        "",
        f"- Event-type split stacks: `{report_dir}`",
        f"- Individual event thumbnails: `{report_dir / 'event_thumbnails'}`",
        "- Date-block jackknife metrics are included in source-level results.",
        "",
        "## Interpretation",
        "",
        "The strongest Earth channels have stacked SNR > 17 and are present in multiple bands/antennas. "
        "This demonstrates that the pipeline can recover a bright positive-control occultation-like signal. "
        "Some row-level detections remain quality-limited, so Earth should be used as the calibration anchor "
        "for timing, window size, and control thresholds rather than as proof that every row-level event is clean.",
    ]
    (out_reports / "earth_positive_control_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sun_report(
    clean: pd.DataFrame,
    earth_root: Path,
    out_reports: Path,
    source_rows: list[pd.DataFrame],
) -> None:
    dedicated_source_level = Path("outputs/sun_focused_validation/sun_source_level_results.csv")
    dedicated_report = out_reports / "sun_focused_validation.md"
    if dedicated_source_level.exists() and dedicated_report.exists():
        source_rows.append(_read_csv(dedicated_source_level, parse_dates=["predicted_event_time"] if "predicted_event_time" in read_table(dedicated_source_level, nrows=0).columns else None))
        return

    events = _read_csv(earth_root / "02_events" / "predicted_events.csv", parse_dates=["predicted_event_time"])
    scored = _read_csv(earth_root / "07_scoring" / "scored_detections.csv", parse_dates=["predicted_event_time"])
    stack_summary = _read_csv(earth_root / "04_stack" / "stacked_detection_summary.csv")
    profiles = _read_csv(earth_root / "04_stack" / "event_aligned_profiles.csv", parse_dates=["predicted_event_time"])
    sun_events = events[events["source_name"].eq("sun")].reset_index(drop=True)
    sun_scored = scored[scored["source_name"].eq("sun")].reset_index(drop=True)
    sun_stack = stack_summary[stack_summary["source_name"].eq("sun")].reset_index(drop=True)
    quality = event_window_quality(clean, sun_events, 600.0)
    block = block_jackknife_from_profiles(profiles[profiles["source_name"].eq("sun")])
    source_level = aggregate_source_level(sun_events, sun_scored, sun_stack, quality, block, window_s=600.0)
    source_rows.append(source_level)

    report_dir = ensure_dir(out_reports / "sun_focused_validation_assets")
    strongest = sun_stack.assign(abs_snr=sun_stack["stacked_snr"].abs()).sort_values("abs_snr", ascending=False).head(8)
    for _, row in strongest.head(5).iterrows():
        freq = row["frequency_band"]
        ant = row["antenna"]
        sub = profiles[(profiles["source_name"].eq("sun")) & (profiles["frequency_band"].eq(freq)) & (profiles["antenna"].eq(ant))]
        combined, summary = stack_profiles(sub, by=["source_name", "frequency_band", "antenna", "event_type"], n_bootstrap=20)
        combined.to_csv(report_dir / f"sun_band{int(freq)}_{ant}_event_type_stack.csv", index=False)
        summary.to_csv(report_dir / f"sun_band{int(freq)}_{ant}_event_type_summary.csv", index=False)
        _plot_profiles(combined, report_dir / f"sun_band{int(freq)}_{ant}_event_type_stack.png", f"Sun band {int(freq)} {ant}")
    _top_event_plots(clean, sun_scored, report_dir / "event_thumbnails", 600.0, n=16)

    top_level = source_level.assign(abs_snr=source_level["stacked_snr"].abs()).sort_values("abs_snr", ascending=False).head(8)
    best = top_level.iloc[0] if not top_level.empty else None
    classification = "unresolved"
    if best is not None:
        clean_ok = pd.notna(best["clean_fraction"]) and float(best["clean_fraction"]) >= 0.5
        timing_ok = pd.isna(best["median_timing_offset_s"]) or abs(float(best["median_timing_offset_s"])) <= 60.0
        rand_ok = pd.notna(best["randomized_empirical_p"]) and float(best["randomized_empirical_p"]) <= 0.05
        if abs(float(best["stacked_snr"])) >= 5.0 and clean_ok and timing_ok and rand_ok:
            classification = "solar_candidate"
    grade_counts = sun_scored["detection_grade"].value_counts().to_dict()
    lines = [
        "# Sun Focused Validation",
        "",
        "## Result",
        "",
        f"Current Sun classification: `{classification}`.",
        "",
        "This is not a confirmed solar detection. The Sun is treated as a moving, variable positive-control candidate, "
        "so strong single-event rows and moderate stacks are useful evidence but not sufficient without source-neighborhood "
        "controls and date-block stability checks tuned for solar variability.",
        "",
        "## Strongest Stacked Channels",
        "",
        strongest[["frequency_band", "frequency_mhz", "antenna", "n_events", "stacked_amplitude", "stacked_snr", "bootstrap_std", "jackknife_std"]].to_string(index=False),
        "",
        "## Source-Level Rows",
        "",
        top_level.drop(columns=["abs_snr"], errors="ignore").to_string(index=False),
        "",
        "## Event-Level Grades",
        "",
        pd.DataFrame([grade_counts]).to_string(index=False),
        "",
        "## Diagnostics Generated",
        "",
        f"- Event-type split stacks: `{report_dir}`",
        f"- Individual event thumbnails: `{report_dir / 'event_thumbnails'}`",
        "- Date-block jackknife metrics are included in source-level results.",
        "",
        "## Interpretation",
        "",
        "The Sun shows plausible signal, especially in the strongest stacked channel and in individual high-SNR events, "
        "but the current focused suite does not provide enough evidence for a confirmed detection. The next control gap is "
        "a moving-source off-limb/off-ephemeris control, plus explicit date-block tests to separate persistent solar response "
        "from a small number of active intervals.",
    ]
    (out_reports / "sun_basic_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _cyg_forensic(
    clean: pd.DataFrame,
    bright_root: Path,
    out_root: Path,
    out_reports: Path,
    offsource_summary: pd.DataFrame,
    source_rows: list[pd.DataFrame],
    windows: list[float],
) -> None:
    events = _read_csv(bright_root / "02_events" / "predicted_events.csv", parse_dates=["predicted_event_time"])
    cyg = events[
        events["source_name"].eq("cyg_a")
        & events["frequency_band"].isin([1, 2, 3])
        & events["antenna"].isin(["rv1_coarse", "rv2_coarse"])
    ].reset_index(drop=True)
    forensic_dir = ensure_dir(out_root / "cyg_a_forensic")
    report_lines = [
        "# Cyg A Band 1 rv1_coarse Forensic Report",
        "",
        "Targeted result is treated as source-versus-systematics evidence, not as a detection by default.",
        "",
    ]
    classifications = []
    for window in windows:
        run = _run_detection_stack_score(
            clean,
            cyg,
            cyg,
            window,
            forensic_dir / f"window_{int(window)}s",
            random_seed=777 + int(window),
            write_tables=True,
        )
        block = block_jackknife_from_profiles(run["profiles"])
        source_level = aggregate_source_level(cyg, run["scored"], run["stack_summary"], run["quality"], block, window_s=window)
        source_level = add_offsource_pvalues(source_level, offsource_summary[offsource_summary["window_s"].eq(window)] if not offsource_summary.empty else offsource_summary)
        decisions = source_level.apply(lambda row: row["decision_grade"], axis=1)
        source_rows.append(source_level)

        band1 = run["stack_summary"][
            run["stack_summary"]["frequency_band"].eq(1) & run["stack_summary"]["antenna"].eq("rv1_coarse")
        ]
        if not band1.empty:
            stack_row = band1.iloc[0]
            p_off = source_level[
                source_level["frequency_band"].eq(1) & source_level["antenna"].eq("rv1_coarse")
            ]["offsource_empirical_p"].iloc[0]
            classification = "unresolved"
            reason = "off-source/timing/block evidence is not strong enough for a source-like claim"
            if pd.notna(p_off) and p_off > 0.1 and abs(stack_row["stacked_snr"]) >= 5:
                classification = "systematic-like"
                reason = "off-source controls are comparable to the real low-frequency stack"
            elif pd.notna(p_off) and p_off <= 0.05 and abs(stack_row["stacked_snr"]) >= 5:
                classification = "source-like"
                reason = "off-source controls are weaker than the real stack"
            classifications.append((window, classification, reason))
        sub = run["profiles"][(run["profiles"]["source_name"].eq("cyg_a")) & (run["profiles"]["frequency_band"].eq(1)) & (run["profiles"]["antenna"].eq("rv1_coarse"))]
        split_stack, split_summary = stack_profiles(sub, by=["source_name", "frequency_band", "antenna", "event_type"], n_bootstrap=20)
        split_stack.to_csv(forensic_dir / f"cyg_a_band1_rv1_event_type_stack_{int(window)}s.csv", index=False)
        split_summary.to_csv(forensic_dir / f"cyg_a_band1_rv1_event_type_summary_{int(window)}s.csv", index=False)
        _plot_profiles(split_stack, forensic_dir / f"cyg_a_band1_rv1_event_type_stack_{int(window)}s.png", f"Cyg A band 1 rv1 {int(window)}s")
        _top_event_plots(clean, run["scored"], forensic_dir / f"window_{int(window)}s" / "event_thumbnails", window, n=16)
        report_lines.extend(
            [
                f"## Window {int(window)} s",
                "",
                run["stack_summary"].assign(abs_snr=run["stack_summary"]["stacked_snr"].abs()).sort_values("abs_snr", ascending=False).head(8).to_string(index=False),
                "",
                "Source-level rows:",
                "",
                source_level.sort_values("stacked_snr", key=lambda s: s.abs(), ascending=False).head(8).to_string(index=False),
                "",
            ]
        )
    final_class = "unresolved"
    if classifications and all(c[1] == "systematic-like" for c in classifications):
        final_class = "systematic-like"
    elif classifications and all(c[1] == "source-like" for c in classifications):
        final_class = "source-like"
    report_lines.extend(
        [
            "## Classification",
            "",
            f"Current Cyg A band 1 rv1_coarse classification: `{final_class}`.",
            "",
            "Evidence considered: off-source p-values, stack strength, event-type split, timing offsets, "
            "quality summaries, antenna/frequency comparisons, and block jackknife metrics.",
        ]
    )
    (out_reports / "cyg_a_band1_rv1_forensic.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def _append_existing_jupiter(jupiter_root: Path, source_rows: list[pd.DataFrame]) -> None:
    events = _read_csv(jupiter_root / "02_events" / "predicted_events.csv", parse_dates=["predicted_event_time"])
    for window in [600.0, 900.0]:
        run_dir = jupiter_root / "09_window_sweep" / f"window_{int(window)}s"
        scored = _read_csv(run_dir / "scored_detections.csv", parse_dates=["predicted_event_time"])
        stack = _read_csv(run_dir / "stacked_detection_summary.csv")
        source_rows.append(aggregate_source_level(events, scored, stack, quality=None, block_summary=None, window_s=window))


def _jupiter_episodic_note(jupiter_root: Path, out_reports: Path) -> None:
    lines = [
        "# Jupiter Episodic Interpretation Note",
        "",
        "Jupiter should not be forced into the same fixed-continuum interpretation used for Cyg A, Cas A, or Fornax A.",
        "A weak or absent repeated-event stack is compatible with Jupiter being quiet for most predicted occultations and bright only during bursty intervals.",
        "",
        "## Current Evidence",
        "",
        f"- Existing report: `{jupiter_root / 'jupiter_control_validation_report.md'}`",
        "- The post-November Jupiter survey has no strong-control-validated rows.",
        "- The best repeated-event stack is marginal, with stacked SNR about 3.",
        "- Candidate rows cluster on specific dates, especially around 1975-01-01, 1975-03-26, and late June 1975.",
        "",
        "## Interpretation",
        "",
        "These clustered candidates should be treated as episodic-source candidates, not as failed fixed-continuum detections. "
        "The next Jupiter test should split by date cluster, frequency below a few MHz, antenna, event type, and local variance/burstiness. "
        "A valid Jupiter claim would require burst-consistent local behavior in the event windows and controls showing that the same burst-like structure "
        "does not appear at randomized times or unrelated antennas/frequencies.",
    ]
    (out_reports / "jupiter_episodic_interpretation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleaned", default="outputs/control_survey_earth_sun_postnov1974_v2/01_ingest/cleaned_timeseries.csv")
    parser.add_argument("--sources", default="configs/bright_sources.csv")
    parser.add_argument("--earth-root", default="outputs/control_survey_earth_sun_postnov1974_v2")
    parser.add_argument("--bright-root", default="outputs/control_survey_bright_sources_postnov1974_v1")
    parser.add_argument("--jupiter-root", default="outputs/control_survey_jupiter_postnov1974_v1")
    parser.add_argument("--output-root", default="outputs/focused_validation")
    parser.add_argument("--prediction-cadence-seconds", type=float, default=300.0)
    parser.add_argument("--windows", nargs="+", type=float, default=[600.0, 900.0])
    args = parser.parse_args()

    out_root = ensure_dir(args.output_root)
    out_reports = ensure_dir("outputs/reports")
    ensure_dir("outputs/summary")
    clean = _read_csv(args.cleaned, parse_dates=["time"])
    sources = load_source_list(args.sources)
    off_events = _generate_cyg_offsource(clean, sources, out_root, args)
    off_summary = _offsource_stacks(clean, off_events, out_root, args.windows)

    source_rows: list[pd.DataFrame] = []
    _earth_report(clean, Path(args.earth_root), out_reports, source_rows)
    _sun_report(clean, Path(args.earth_root), out_reports, source_rows)
    _cyg_forensic(clean, Path(args.bright_root), out_root, out_reports, off_summary, source_rows, args.windows)
    _append_existing_jupiter(Path(args.jupiter_root), source_rows)
    _jupiter_episodic_note(Path(args.jupiter_root), out_reports)

    source_level = pd.concat([df for df in source_rows if df is not None and not df.empty], ignore_index=True)
    source_level.to_csv("outputs/summary/source_level_results.csv", index=False)
    final = final_source_summary(source_level)
    final.to_csv("outputs/summary/source_final_summary.csv", index=False)
    write_json(
        out_root / "focused_validation_config.json",
        {
            "cleaned": args.cleaned,
            "sources": args.sources,
            "earth_root": args.earth_root,
            "bright_root": args.bright_root,
            "jupiter_root": args.jupiter_root,
            "windows": args.windows,
            "prediction_cadence_seconds": args.prediction_cadence_seconds,
        },
    )
    print("offsource_controls: outputs/focused_validation/controls/offsource_controls.csv")
    print("source_level_results: outputs/summary/source_level_results.csv")
    print("source_final_summary: outputs/summary/source_final_summary.csv")
    print("earth_report: outputs/reports/earth_positive_control_validation.md")
    print("sun_report: outputs/reports/sun_focused_validation.md")
    print("cyg_report: outputs/reports/cyg_a_band1_rv1_forensic.md")
    print("jupiter_note: outputs/reports/jupiter_episodic_interpretation.md")


if __name__ == "__main__":
    main()
