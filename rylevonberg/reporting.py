"""Markdown report generation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_methodology_report(path: str | Path, outputs: dict[str, str]) -> None:
    lines = [
        "# RAE-2 Ryle-Vonberg Pipeline Report",
        "",
        "## Implemented",
        "",
        "- CSV ingestion with an adapter for the existing `RAE2AgentV2/rae2/io.py` loader and a local pandas fallback.",
        "- Long-form internal time-series model preserving original timestamps, frequency bands, antenna identifiers, power values, geometry columns, and quality flags.",
        "- Validation for malformed rows, duplicate timestamps, gaps, nonpositive/saturated values, and large telemetry jumps.",
        "- FK4/B1950-default source handling with explicit frame transforms through astropy.",
        "- Separated spacecraft/Moon geometry, sky-frame transforms, lunar-limb visibility, and event prediction.",
        "- Event tables for disappearance/reappearance with limb metadata and optional frequency/antenna expansion.",
        "- Local step-template fits, matched filters, event stacking, blind change-point limb constraints, negative controls, and injection recovery.",
        "",
        "## Geometry Assumptions",
        "",
        "- The master CSV spacecraft position is interpreted as a Moon-centered inertial vector in kilometers.",
        "- The default celestial analysis frame is FK4 with equinox B1950, matching the RAE-2 source-coordinate context.",
        "- Fixed source coordinates are never silently assumed to be ICRS; each source row carries a `frame` column.",
        "- A source is occulted when its spacecraft-frame direction lies within the Moon disk centered on `-spacecraft_position` with angular radius `asin(R_moon / range)`.",
        "- Moving-body vectors are computed as body barycentric minus Moon barycentric and transformed into the requested analysis frame.",
        "- Earth positive-control geometry can use the CSV Earth unit vector converted to spacecraft-to-Earth direction.",
        "",
        "## Remaining Uncertainties",
        "",
        "- The repository contains older geometry shortcuts and frame assumptions; they were inspected only for file/schema context.",
        "- The exact provenance and time scale of CSV timestamps should be checked against mission documentation; current handling treats them as UTC-like naive timestamps.",
        "- The FK4 interpretation of spacecraft vectors should be verified against independent ephemeris anchors before production claims.",
        "- Blind limb-circle clustering is a first-pass constraint catalog, not a full sky-position inversion.",
        "",
        "## Interactive-Node Use",
        "",
        "Run inside an interactive allocation, for example:",
        "",
        "```bash",
        "salloc --qos=interactive --nodes=1 --time=02:00:00",
        "source .venv/bin/activate",
        "cd /path/to/RyleVonberg",
        "python -m rylevonberg.pipeline run-smoke --data /path/to/interpolatedRAE2MasterFile.csv --start '1973-10-03 04:00:00' --end '1973-10-03 06:00:00'",
        "```",
        "",
        "The pipeline refuses explicit `batch`, `sbatch`, `qsub`, or noninteractive queue modes.",
        "",
        "## Output Files",
        "",
    ]
    for label, out_path in outputs.items():
        lines.append(f"- `{label}`: `{out_path}`")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def compact_table_summary(df: pd.DataFrame, max_rows: int = 8) -> str:
    if df.empty:
        return "(empty)"
    return df.head(max_rows).to_string(index=False)


def write_control_validation_report(
    path: str | Path,
    outputs: dict[str, str],
    scored: pd.DataFrame,
    stack_summary: pd.DataFrame,
    run_args: dict,
) -> None:
    grade_counts = scored["detection_grade"].value_counts().to_dict() if not scored.empty and "detection_grade" in scored else {}
    top_cols = [
        "source_name",
        "event_type",
        "predicted_event_time",
        "frequency_band",
        "frequency_mhz",
        "antenna",
        "receiver",
        "moon_pointing",
        "best_empirical_p",
        "best_abs_snr",
        "timing_offset_sec",
        "quality_clean_fraction",
        "detection_grade",
    ]
    stack_cols = [
        "source_name",
        "frequency_band",
        "frequency_mhz",
        "antenna",
        "n_events",
        "stacked_amplitude",
        "stacked_snr",
        "bootstrap_std",
        "jackknife_std",
    ]
    top = scored[[c for c in top_cols if c in scored.columns]].head(20) if not scored.empty else pd.DataFrame()
    stack = stack_summary[[c for c in stack_cols if c in stack_summary.columns]].sort_values("stacked_snr", key=lambda s: s.abs(), ascending=False).head(20) if not stack_summary.empty and "stacked_snr" in stack_summary else stack_summary.head(20)
    lines = [
        "# Earth/Sun Post-November-1974 Control Validation",
        "",
        "## Scope",
        "",
        f"- Sources: `{', '.join(run_args.get('source', []))}`.",
        f"- Date range: `{run_args.get('start')}` through `{run_args.get('end')}`.",
        f"- Frequencies: `{', '.join(run_args.get('frequency', []))}`.",
        f"- Antennas: `{', '.join(run_args.get('antenna', []))}`.",
        "- Frequency coherence across adjacent bands is intentionally not used as a grading criterion in this run.",
        "",
        "## Pipeline Explanation",
        "",
        "1. `ingest` reads the CSV slice, converts `rv1_coarse`/`rv2_coarse` into a long time-series table, attaches MHz labels, and flags gaps, duplicated timestamps, malformed rows, saturation/nonpositive values, and telemetry jumps.",
        "2. `predict` computes FK4/B1950 lunar-limb disappearance and reappearance times for Earth and Sun, then expands each event over the requested band/antenna grid.",
        "3. `detect` runs both the local step-template model and the matched filter for every predicted event/channel.",
        "4. `stack` is involved: events are aligned by predicted time, local baselines are removed, profiles are normalized, and stacked amplitudes/SNRs plus bootstrap/jackknife scatter are written to `04_stack`.",
        "5. `validate` builds randomized-time, reversed-template, wrong-frequency, and wrong-antenna controls and runs injection-recovery grids.",
        "6. `score-detections` compares the real detections to randomized-time empirical nulls, merges antenna orientation metadata, records timing offsets and clean-sample fractions, and assigns review grades.",
        "",
        "## Grade Counts",
        "",
        compact_table_summary(pd.DataFrame([grade_counts])) if grade_counts else "(no scored detections)",
        "",
        "## Top Scored Events",
        "",
        compact_table_summary(top, max_rows=20),
        "",
        "## Strongest Stacked Summaries",
        "",
        compact_table_summary(stack, max_rows=20),
        "",
        "## Output Files",
        "",
    ]
    for label, out_path in outputs.items():
        lines.append(f"- `{label}`: `{out_path}`")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
