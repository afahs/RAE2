#!/usr/bin/env python
"""Validate lower-V control-manifold subtraction on Earth positive control.

This script builds an Earth-only lower-V stack/control table in the same schema
used by the Sun/Fornax lower-V stack-first run, then applies the same
sideband-trained control-manifold subtraction. It intentionally uses no upper-V
data and does not treat neighboring frequency bands as negative controls.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from rylevonberg.table_io import read_table

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.util import ensure_dir, software_versions, write_json  # noqa: E402
from scripts.run_lower_v_stackfirst_detection_attempt import (  # noqa: E402
    ANTENNA,
    CLEAN,
    _load_clean_groups,
    _make_randomized_controls,
    _make_time_shift_controls,
    _read,
    collect_profiles,
    empirical_fit_summary,
    fit_stacks,
    plot_amplitude_spectrum,
    plot_real_vs_controls_grid,
    plot_top_fit_profiles,
    stack_by_control,
    summarize_control_curves,
)


EARTH_EVENTS = ROOT / "outputs/planetary_confirmation_survey_science_baseline_v2/events/earth_predicted_events.csv"
DEFAULT_OUT = ROOT / "outputs/lower_v_control_manifold_earth_positive_control_v1"


def _earth_real_events(start_date: str) -> pd.DataFrame:
    events = _read(EARTH_EVENTS, parse_dates=["predicted_event_time"])
    events = events[
        events["source_name"].astype(str).str.lower().eq("earth")
        & events["antenna"].astype(str).eq(ANTENNA)
    ].copy()
    events["predicted_event_time"] = pd.to_datetime(events["predicted_event_time"])
    events = events[events["predicted_event_time"] >= pd.Timestamp(start_date)].copy()
    events["analysis_source"] = "earth"
    events["source_name"] = "earth"
    events["control_family"] = "real"
    events["control_id"] = "real"
    events["control_type"] = "true_prediction"
    return events.reset_index(drop=True)


def _earth_event_table(
    start_date: str,
    shifts_s: list[float],
    n_random: int,
    random_seed: int,
    window_s: float,
) -> pd.DataFrame:
    real = _earth_real_events(start_date)
    clean_times = _read(CLEAN, usecols=["time"], parse_dates=["time"])["time"]
    tables = [
        real,
        _make_time_shift_controls(real, shifts_s),
        _make_randomized_controls(real, clean_times, n_random=n_random, seed=random_seed, window_s=window_s),
    ]
    events = pd.concat([t for t in tables if not t.empty], ignore_index=True)
    events["event_uid"] = np.arange(len(events), dtype=int)
    return events


def _write_stack_report(
    out_dir: Path,
    events: pd.DataFrame,
    status: pd.DataFrame,
    summary: pd.DataFrame,
    plot_paths: list[Path],
) -> None:
    counts = (
        status.groupby(["analysis_source", "control_family"])["used_in_stack"]
        .agg(["sum", "count"])
        .reset_index()
        .rename(columns={"sum": "n_used_windows", "count": "n_candidate_windows"})
    )
    cols = [
        "analysis_source",
        "frequency_mhz",
        "real_amplitude",
        "real_uncertainty",
        "real_fit_snr",
        "real_delta_bic",
        "real_best_transition_duration_s",
        "time_shift_p_amp_ge_real",
        "randomized_time_p_amp_ge_real",
    ]
    lines = [
        "# Earth Lower-V Stack-First Input For Control-Manifold Validation",
        "",
        "This is the Earth positive-control input used for the control-manifold background subtraction validation.",
        "",
        "Constraints:",
        "",
        "- antenna: lower V only (`rv2_coarse`);",
        "- controls: time-shift and randomized-time only;",
        "- no upper-V data;",
        "- no neighboring-band negative controls;",
        "- no forward sky/beam model.",
        "",
        f"Real Earth event rows: {len(events[events['control_family'].eq('real')])}",
        "",
        "## Usable Window Counts",
        "",
        counts.to_string(index=False),
        "",
        "## Stack-First Fit Summary Before Manifold Subtraction",
        "",
        summary[cols].sort_values(["frequency_mhz"]).to_string(index=False),
        "",
        "## Stack-First Plots",
        "",
    ]
    lines.extend(f"- `{p.name}`" for p in plot_paths)
    (out_dir / "earth_stackfirst_input_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_final_report(out_dir: Path) -> None:
    summary_path = out_dir / "control_manifold_fit_summary.csv"
    if not summary_path.exists():
        return
    summary = read_table(summary_path)
    main = summary[
        summary["family_set"].eq("all_controls")
        & summary["n_components_requested"].eq(2)
        & summary["analysis_source"].eq("earth")
    ].copy()
    cols = [
        "frequency_mhz",
        "real_residual_amplitude",
        "real_residual_uncertainty",
        "real_residual_fit_snr",
        "real_delta_bic",
        "empirical_p_abs_amp_ge_real",
    ]
    lines = [
        "# Earth Positive-Control Control-Manifold Validation",
        "",
        "This applies the same lower-V sideband-trained control-manifold subtraction used for Sun/Fornax-A to Earth.",
        "",
        "The background model is learned from lower-V time-shift and randomized-time control stacks. The real Earth curve is fit to that control manifold only in sideband bins away from the predicted event, then the inferred background is subtracted across the full event window.",
        "",
        "A successful positive-control result should retain positive-source occultation residuals after subtraction. A failure would mean the empirical background model is too aggressive or is erasing real occultations.",
        "",
        "## Main k=2 all-controls residual summary",
        "",
        main[cols].sort_values("frequency_mhz").to_string(index=False),
        "",
        "## Interpretation",
        "",
    ]
    if not main.empty:
        positive = main[pd.to_numeric(main["real_residual_amplitude"], errors="coerce") > 0]
        strong = main[
            (pd.to_numeric(main["real_residual_amplitude"], errors="coerce") > 0)
            & (pd.to_numeric(main["real_delta_bic"], errors="coerce") > 2)
        ]
        lines += [
            f"- Positive residual channels: {len(positive)} / {len(main)}.",
            f"- Positive residual channels with DeltaBIC > 2: {len(strong)} / {len(main)}.",
            "- This should be judged visually from the residual grids as well as numerically; high fit SNR alone is not the decision rule.",
        ]
    lines += [
        "",
        "## Key plots",
        "",
        "- `earth_all_controls_k2_control_manifold_residual_grid.png`",
        "- `earth_all_controls_control_manifold_amplitude_spectrum.png`",
        "- `earth_all_controls_k2_top_control_manifold_residual_fits.png`",
    ]
    (out_dir / "earth_control_manifold_validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--start-date", default="1974-11-01")
    parser.add_argument("--window-s", type=float, default=900.0)
    parser.add_argument("--bin-s", type=float, default=60.0)
    parser.add_argument("--sideband-s", type=float, default=600.0)
    parser.add_argument("--time-shifts-s", default="-1200,-600,-300,300,600,1200")
    parser.add_argument("--n-random", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=20260604)
    parser.add_argument("--timing-offsets-s", default="0")
    parser.add_argument("--transition-durations-s", default="0,120,300,600,900")
    parser.add_argument("--skip-stack-plots", action="store_true")
    args = parser.parse_args()

    out_dir = ensure_dir(Path(args.out_dir))
    shifts = [float(x.strip()) for x in str(args.time_shifts_s).split(",") if x.strip()]
    timing_offsets = [float(x.strip()) for x in str(args.timing_offsets_s).split(",") if x.strip()]
    transition_durations = [float(x.strip()) for x in str(args.transition_durations_s).split(",") if x.strip()]
    write_json(
        out_dir / "run_config.json",
        {
            "earth_events": str(EARTH_EVENTS),
            "clean": str(CLEAN),
            "antenna": ANTENNA,
            "start_date": str(args.start_date),
            "window_s": float(args.window_s),
            "bin_s": float(args.bin_s),
            "sideband_s": float(args.sideband_s),
            "time_shifts_s": shifts,
            "n_random": int(args.n_random),
            "random_seed": int(args.random_seed),
            "timing_offsets_s": timing_offsets,
            "transition_durations_s": transition_durations,
            "software_versions": software_versions(),
        },
    )

    print("Building Earth lower-V event/control table...", flush=True)
    events = _earth_event_table(
        start_date=str(args.start_date),
        shifts_s=shifts,
        n_random=int(args.n_random),
        random_seed=int(args.random_seed),
        window_s=float(args.window_s),
    )
    events.to_csv(out_dir / "lower_v_stackfirst_event_rows.csv", index=False)
    bands = sorted(events["frequency_band"].dropna().astype(int).unique())
    print(f"Loading lower-V clean groups for bands {bands}...", flush=True)
    clean_groups = _load_clean_groups(bands)
    print(f"Collecting normalized Earth profiles from {len(events)} real/control rows...", flush=True)
    points, status = collect_profiles(events, clean_groups, float(args.window_s), float(args.bin_s), float(args.sideband_s))
    status.to_csv(out_dir / "lower_v_stackfirst_profile_status.csv", index=False)
    print("Stacking by real/control group...", flush=True)
    stacks = stack_by_control(points)
    stacks.to_csv(out_dir / "lower_v_stackfirst_control_stacks.csv", index=False)
    del points

    control_curves = summarize_control_curves(stacks[stacks["control_family"].ne("real")].copy())
    control_curves.to_csv(out_dir / "lower_v_stackfirst_control_curve_summary.csv", index=False)
    print("Fitting stack-first Earth profiles before manifold subtraction...", flush=True)
    fits = fit_stacks(stacks, timing_offsets, transition_durations)
    fits.to_csv(out_dir / "lower_v_stackfirst_fit_summary_by_group.csv", index=False)
    stack_summary = empirical_fit_summary(fits)
    stack_summary.to_csv(out_dir / "lower_v_stackfirst_empirical_fit_summary.csv", index=False)

    stack_plot_paths: list[Path] = []
    if not args.skip_stack_plots:
        print("Writing Earth stack-first input plots...", flush=True)
        stack_plot_paths.extend(plot_real_vs_controls_grid(stacks, control_curves, out_dir))
        stack_plot_paths.extend(plot_amplitude_spectrum(stack_summary, out_dir))
        stack_plot_paths.extend(plot_top_fit_profiles(stacks, fits, control_curves, stack_summary, out_dir, n_channels=4))
    _write_stack_report(out_dir, events, status, stack_summary, stack_plot_paths)

    print("Running control-manifold subtraction on Earth stack table...", flush=True)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_control_manifold_background_subtraction.py"),
            "--stack-root",
            str(out_dir),
            "--out-dir",
            str(out_dir),
            "--family-sets",
            "time_off,all_controls",
            "--components",
            "1,2,3",
            "--main-components",
            "2",
            "--sideband-s",
            str(float(args.sideband_s)),
            "--timing-offsets-s",
            str(args.timing_offsets_s),
            "--transition-durations-s",
            str(args.transition_durations_s),
        ],
        check=True,
    )
    _write_final_report(out_dir)
    print(out_dir / "earth_control_manifold_validation_report.md")


if __name__ == "__main__":
    main()
