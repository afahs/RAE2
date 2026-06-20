#!/usr/bin/env python
"""Build source-by-source detection technique summary tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rylevonberg.table_io import read_table


ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> pd.DataFrame:
    return read_table(ROOT / path, low_memory=False)


def _read_if_exists(path: str) -> pd.DataFrame:
    full = ROOT / path
    if full.exists():
        return read_table(full, low_memory=False)
    return pd.DataFrame()


def _grade_counts(path: str) -> dict[str, dict[str, int]]:
    df = _read(path)
    out: dict[str, dict[str, int]] = {}
    for source, group in df.groupby("source_name"):
        out[str(source)] = group["detection_grade"].value_counts().to_dict()
    return out


def _markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)

    def clean(value: object) -> str:
        text = "" if pd.isna(value) else str(value)
        return text.replace("\n", " ").replace("|", "\\|")

    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(clean(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    out_dir = ROOT / "outputs" / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    final = _read("outputs/summary/source_final_summary.csv")
    final_by_source = {row["source"]: row for _, row in final.iterrows()}
    source_level = _read("outputs/summary/source_level_results.csv")
    sun_level = _read("outputs/sun_focused_validation/sun_source_level_results.csv")
    sun_high = _read_if_exists("outputs/sun_whole_dataset_validation_highcontrols_bands4_8/summary/sun_whole_dataset_scored_stacks.csv")
    confirmation = _read_if_exists("outputs/candidate_confirmation_checks/candidate_confirmation_summary.csv")
    solar_suite = _read_if_exists("outputs/solar_detection_confirmation_suite_v3/window_sweep.csv")
    solar_suite_strict = _read_if_exists("outputs/solar_detection_confirmation_suite_v3/strict_quality.csv")
    solar_suite_event_type = _read_if_exists("outputs/solar_detection_confirmation_suite_v3/event_type.csv")
    jupiter_suite = _read_if_exists("outputs/jupiter_detection_confirmation_suite/window_sweep.csv")
    jupiter_suite_strict = _read_if_exists("outputs/jupiter_detection_confirmation_suite/strict_quality.csv")
    jupiter_suite_controls = _read_if_exists("outputs/jupiter_detection_confirmation_suite/wrong_controls.csv")
    fornax_suite = _read_if_exists("outputs/fornax_a_detection_confirmation_suite/window_sweep.csv")
    fornax_suite_strict = _read_if_exists("outputs/fornax_a_detection_confirmation_suite/strict_quality.csv")
    cas_suite = _read_if_exists("outputs/cas_a_detection_confirmation_suite/window_sweep.csv")
    cas_suite_strict = _read_if_exists("outputs/cas_a_detection_confirmation_suite/strict_quality.csv")
    cyg_suite = _read_if_exists("outputs/cyg_a_detection_confirmation_suite/window_sweep.csv")
    cyg_suite_strict = _read_if_exists("outputs/cyg_a_detection_confirmation_suite/strict_quality.csv")
    bright = _read("outputs/control_survey_bright_sources_postnov1974_v1/09_window_sweep/window_sweep_summary.csv")

    earth_sun_counts = _grade_counts("outputs/control_survey_earth_sun_postnov1974_v2/07_scoring/scored_detections.csv")
    j600_counts = _grade_counts("outputs/control_survey_jupiter_postnov1974_v1/09_window_sweep/window_600s/scored_detections.csv")
    j900_counts = _grade_counts("outputs/control_survey_jupiter_postnov1974_v1/09_window_sweep/window_900s/scored_detections.csv")

    bright600 = bright[bright["window_seconds"].eq(600.0)].iloc[0]
    bright900 = bright[bright["window_seconds"].eq(900.0)].iloc[0]

    def final_row(source: str) -> pd.Series | None:
        return final_by_source.get(source)

    def best_source_level(source: str) -> pd.Series:
        rows = source_level[source_level["source"].eq(source)].copy()
        return rows.assign(abs_snr=rows["stacked_snr"].abs()).sort_values("abs_snr", ascending=False).iloc[0]

    earth = final_row("earth")
    sun = final_row("sun")
    jupiter = final_row("jupiter")
    cyg = final_row("cyg_a")

    sun_high_best = None
    sun_high_candidates = 0
    if not sun_high.empty:
        sun_high_best = sun_high.assign(abs_snr=sun_high["stacked_snr"].abs()).sort_values("abs_snr", ascending=False).iloc[0]
        sun_high_candidates = int(sun_high["status"].eq("candidate").sum())
    confirmation_by_source = {}
    if not confirmation.empty:
        for source, group in confirmation.groupby("source_name"):
            confirmation_by_source[str(source)] = group.assign(abs_snr=group["stacked_snr"].abs()).sort_values("abs_snr", ascending=False).iloc[0]
    solar_suite_best = None
    if not solar_suite.empty:
        solar_suite_best = solar_suite.assign(abs_snr=solar_suite["stacked_snr"].abs()).sort_values("abs_snr", ascending=False).iloc[0]
    solar_strict_snr = float(solar_suite_strict["stacked_snr"].iloc[0]) if not solar_suite_strict.empty else None
    solar_event_snrs = ""
    if not solar_suite_event_type.empty:
        solar_event_snrs = "; ".join(
            f"{row['event_type']} SNR {row['stacked_snr']:.2f}" for _, row in solar_suite_event_type.iterrows()
        )
    if solar_suite_best is not None and solar_strict_snr is not None:
        sun_interpretation = (
            f"Solar confirmation suite favors band {int(solar_suite_best['frequency_band'])} / {solar_suite_best['frequency_mhz']:.2f} MHz / "
            f"{solar_suite_best['antenna']} / {int(solar_suite_best['window_s'])}s, SNR {solar_suite_best['stacked_snr']:.2f}; "
            f"strict-quality SNR {solar_strict_snr:.2f}; {solar_event_snrs}. Treat as strong solar candidate evidence, not final confirmation until off-ephemeris p<=0.01."
        )
    elif "sun" in confirmation_by_source:
        sun_row = confirmation_by_source["sun"]
        sun_interpretation = (
            f"Confirmation checks favor Sun band {int(sun_row['frequency_band'])} / {sun_row['frequency_mhz']:.2f} MHz / "
            f"{sun_row['antenna']} / {int(sun_row['window_s'])}s as {sun_row['status']}; "
            "treat as solar candidate evidence, not final confirmation."
        )
    else:
        sun_interpretation = "High-control whole-dataset stacks now support solar candidate channels, especially rv2_coarse band 6 and band 8; treat as candidate until event-type plots and independent repeatability checks are reviewed."
    jupiter_suite_best = None
    if not jupiter_suite.empty:
        jupiter_suite_best = jupiter_suite.assign(abs_snr=jupiter_suite["stacked_snr"].abs()).sort_values("abs_snr", ascending=False).iloc[0]
    jupiter_strict_snr = float(jupiter_suite_strict["stacked_snr"].iloc[0]) if not jupiter_suite_strict.empty else None
    jupiter_wrong_snr = np.nan
    if not jupiter_suite_controls.empty:
        wrong = jupiter_suite_controls[jupiter_suite_controls["control_type"].eq("wrong_antenna")]
        if not wrong.empty:
            jupiter_wrong_snr = float(wrong["stacked_snr"].iloc[0])
    if jupiter_suite_best is not None and jupiter_strict_snr is not None:
        jupiter_interpretation = (
            f"Confirmation suite best is band {int(jupiter_suite_best['frequency_band'])} / {jupiter_suite_best['frequency_mhz']:.2f} MHz / "
            f"{jupiter_suite_best['antenna']} / {int(jupiter_suite_best['window_s'])}s, SNR {jupiter_suite_best['stacked_snr']:.2f}; "
            f"strict-quality SNR {jupiter_strict_snr:.2f}; wrong-antenna SNR {jupiter_wrong_snr:.2f}. "
            "Plausible episodic candidate, but not confirmed because significance is modest and wrong-antenna response is non-negligible."
        )
    elif "jupiter" in confirmation_by_source:
        jupiter_interpretation = (
            f"Confirmation check best is band {int(confirmation_by_source['jupiter']['frequency_band'])} / {confirmation_by_source['jupiter']['frequency_mhz']:.2f} MHz / "
            f"{confirmation_by_source['jupiter']['antenna']} / {int(confirmation_by_source['jupiter']['window_s'])}s with status {confirmation_by_source['jupiter']['status']}; "
            "not a confirmed repeated-stack detection."
        )
    else:
        jupiter_interpretation = "Not a confirmed repeated-stack detection; clustered candidates may reflect quiet/bursting Jupiter behavior."

    fornax_suite_best = None
    if not fornax_suite.empty:
        fornax_suite_best = fornax_suite.assign(abs_snr=fornax_suite["stacked_snr"].abs()).sort_values("abs_snr", ascending=False).iloc[0]
    fornax_strict_snr = float(fornax_suite_strict["stacked_snr"].iloc[0]) if not fornax_suite_strict.empty else None
    if fornax_suite_best is not None and fornax_strict_snr is not None:
        fornax_interpretation = (
            f"Confirmation suite best is band {int(fornax_suite_best['frequency_band'])} / {fornax_suite_best['frequency_mhz']:.2f} MHz / "
            f"{fornax_suite_best['antenna']} / {int(fornax_suite_best['window_s'])}s, SNR {fornax_suite_best['stacked_snr']:.2f}; "
            f"strict-quality SNR {fornax_strict_snr:.2f}. Strong full-stack signal fails strict-quality survival, so it is not confirmed."
        )
    elif "fornax_a" in confirmation_by_source:
        fornax_interpretation = (
            f"Confirmation check best is band {int(confirmation_by_source['fornax_a']['frequency_band'])} / {confirmation_by_source['fornax_a']['frequency_mhz']:.2f} MHz / "
            f"{confirmation_by_source['fornax_a']['antenna']} / {int(confirmation_by_source['fornax_a']['window_s'])}s with status {confirmation_by_source['fornax_a']['status']}; "
            "strong-looking low-band stack is not control-clean."
        )
    else:
        fornax_interpretation = "Many candidates because many crossings; strongest stacks are not control-clean and change with window."

    cas_suite_best = None
    if not cas_suite.empty:
        cas_suite_best = cas_suite.assign(abs_snr=cas_suite["stacked_snr"].abs()).sort_values("abs_snr", ascending=False).iloc[0]
    cas_strict_snr = float(cas_suite_strict["stacked_snr"].iloc[0]) if not cas_suite_strict.empty else None
    if cas_suite_best is not None and cas_strict_snr is not None:
        cas_interpretation = (
            f"Confirmation suite best is band {int(cas_suite_best['frequency_band'])} / {cas_suite_best['frequency_mhz']:.2f} MHz / "
            f"{cas_suite_best['antenna']} / {int(cas_suite_best['window_s'])}s, SNR {cas_suite_best['stacked_snr']:.2f}; "
            f"strict-quality SNR {cas_strict_snr:.2f}. Large timing offset and strict-quality failure prevent confirmation."
        )
    else:
        cas_interpretation = "Moderate stacks only; not confirmed."

    cyg_suite_best = None
    if not cyg_suite.empty:
        cyg_suite_best = cyg_suite.assign(abs_snr=cyg_suite["stacked_snr"].abs()).sort_values("abs_snr", ascending=False).iloc[0]
    cyg_strict_snr = float(cyg_suite_strict["stacked_snr"].iloc[0]) if not cyg_suite_strict.empty else None
    if cyg_suite_best is not None and cyg_strict_snr is not None:
        cyg_interpretation = (
            f"Confirmation suite best is band {int(cyg_suite_best['frequency_band'])} / {cyg_suite_best['frequency_mhz']:.2f} MHz / "
            f"{cyg_suite_best['antenna']} / {int(cyg_suite_best['window_s'])}s, SNR {cyg_suite_best['stacked_snr']:.2f}; "
            f"strict-quality SNR {cyg_strict_snr:.2f}. Huge aggregate signal collapses under strict-quality cuts and is systematic-like."
        )
    else:
        cyg_interpretation = "Large negative low-band stack but unresolved because off-source/quality controls do not support confirmation."

    rows = [
        {
            "source": "Earth",
            "source_class": "moving positive control",
            "event_prediction": "lunar-limb disappearance/reappearance from Earth vector",
            "stepfit_matched_filter": f"{earth_sun_counts['earth'].get('candidate_needs_review', 0)} candidate rows; {earth_sun_counts['earth'].get('quality_limited', 0)} quality-limited rows; 0 strong-control rows",
            "stacking_result": f"best {earth['strongest_channel']}; stacked SNR {earth['strongest_stacked_snr']:.2f}",
            "empirical_controls": f"randomized-time p={earth['best_randomized_empirical_p']:.3g}; off-source not applicable",
            "special_controls": "Earth positive-control report; event-type split stacks; block jackknife; thumbnails",
            "quality_timing": earth["timing_consistency_summary"] + "; " + earth["quality_summary"],
            "final_status": earth["final_status"],
            "interpretation": "Confirmed positive control for the pipeline, not an astrophysical discovery claim.",
        },
        {
            "source": "Sun",
            "source_class": "moving variable solar source",
            "event_prediction": "solar ephemeris limb crossings plus moving off-ephemeris fake-Sun tracks",
            "stepfit_matched_filter": f"{earth_sun_counts['sun'].get('candidate_needs_review', 0)} candidate rows; {earth_sun_counts['sun'].get('quality_limited', 0)} quality-limited rows; 0 strong-control rows",
            "stacking_result": (
                f"high-control best band {int(sun_high_best['frequency_band'])} / {sun_high_best['frequency_mhz']:.2f} MHz / {sun_high_best['antenna']}, "
                f"SNR {sun_high_best['stacked_snr']:.2f}; {sun_high_candidates} high-control candidate channels"
                if sun_high_best is not None
                else f"best {sun['strongest_channel']}; stacked SNR {sun['strongest_stacked_snr']:.2f}"
            ),
            "empirical_controls": (
                f"high-control randomized-time p={sun_high_best['randomized_stack_p']:.3g}; moving off-ephemeris p={sun_high_best['offephemeris_stack_p']:.3g}"
                if sun_high_best is not None
                else f"randomized-time p={sun['best_randomized_empirical_p']:.3g}; moving off-ephemeris best p={sun['best_offsource_empirical_p']:.3g}"
            ),
            "special_controls": "high-control whole-dataset candidate-band run; off-ephemeris fake-Sun controls; date-cluster/burst metrics; antenna/frequency coherence",
            "quality_timing": sun["timing_consistency_summary"] + "; " + sun["quality_summary"],
            "final_status": "candidate",
            "interpretation": sun_interpretation,
        },
        {
            "source": "Jupiter",
            "source_class": "moving episodic/bursty source",
            "event_prediction": "Jupiter ephemeris limb crossings",
            "stepfit_matched_filter": f"600 s: {j600_counts['jupiter'].get('candidate_needs_review', 0)} candidate rows; 900 s: {j900_counts['jupiter'].get('candidate_needs_review', 0)} candidate rows; 0 strong-control rows",
            "stacking_result": f"best {jupiter['strongest_channel']}; stacked SNR {jupiter['strongest_stacked_snr']:.2f}",
            "empirical_controls": f"best randomized-time p={jupiter['best_randomized_empirical_p']:.3g}; no off-source Jupiter controls yet",
            "special_controls": "episodic interpretation note; date-cluster review recommended",
            "quality_timing": jupiter["timing_consistency_summary"] + "; " + jupiter["quality_summary"],
            "final_status": jupiter["final_status"],
            "interpretation": jupiter_interpretation,
        },
        {
            "source": "Cyg A",
            "source_class": "fixed bright low-frequency source",
            "event_prediction": "FK4 fixed-source lunar-limb crossings",
            "stepfit_matched_filter": f"600 s: {int(bright600['cyg_a_candidate_needs_review'])} candidate rows; 900 s: {int(bright900['cyg_a_candidate_needs_review'])} candidate rows; 0 strong-control rows",
            "stacking_result": f"best {cyg['strongest_channel']}; stacked SNR {cyg['strongest_stacked_snr']:.2f}",
            "empirical_controls": f"randomized-time p={cyg['best_randomized_empirical_p']:.3g}; off-source p={cyg['best_offsource_empirical_p']:.3g}",
            "special_controls": "Cyg A 0.45 MHz forensic report; off-source controls; antenna/frequency comparisons; block jackknife",
            "quality_timing": cyg["timing_consistency_summary"] + "; " + cyg["quality_summary"],
            "final_status": cyg["final_status"],
            "interpretation": cyg_interpretation,
        },
        {
            "source": "Cas A",
            "source_class": "fixed bright low-frequency source",
            "event_prediction": "FK4 fixed-source lunar-limb crossings",
            "stepfit_matched_filter": f"600 s: {int(bright600['cas_a_candidate_needs_review'])} candidate rows; 900 s: {int(bright900['cas_a_candidate_needs_review'])} candidate rows; 0 strong-control rows",
            "stacking_result": f"600 s best band {int(bright600['cas_a_best_stack_band'])} / {bright600['cas_a_best_stack_mhz']:.2f} MHz / {bright600['cas_a_best_stack_antenna']}, SNR {bright600['cas_a_best_stack_snr']:.2f}; 900 s best SNR {bright900['cas_a_best_stack_snr']:.2f}",
            "empirical_controls": "randomized-time and time-reversed controls in bright-source survey; no dedicated off-source forensic table yet",
            "special_controls": "bright-source survey window sweep",
            "quality_timing": f"quality-limited rows: 600 s {int(bright600['cas_a_quality_limited'])}, 900 s {int(bright900['cas_a_quality_limited'])}",
            "final_status": "not_confirmed",
            "interpretation": cas_interpretation,
        },
        {
            "source": "Fornax A",
            "source_class": "fixed bright low-frequency source",
            "event_prediction": "FK4 fixed-source lunar-limb crossings",
            "stepfit_matched_filter": f"600 s: {int(bright600['fornax_a_candidate_needs_review'])} candidate rows; 900 s: {int(bright900['fornax_a_candidate_needs_review'])} candidate rows; 0 strong-control rows",
            "stacking_result": f"600 s best SNR {bright600['fornax_a_best_stack_snr']:.2f}; 900 s best band {int(bright900['fornax_a_best_stack_band'])} / {bright900['fornax_a_best_stack_mhz']:.2f} MHz / {bright900['fornax_a_best_stack_antenna']}, SNR {bright900['fornax_a_best_stack_snr']:.2f}",
            "empirical_controls": "randomized-time and time-reversed controls in bright-source survey; no dedicated off-source forensic table yet",
            "special_controls": "bright-source survey window sweep",
            "quality_timing": f"quality-limited rows: 600 s {int(bright600['fornax_a_quality_limited'])}, 900 s {int(bright900['fornax_a_quality_limited'])}",
            "final_status": "not_confirmed",
            "interpretation": fornax_interpretation,
        },
    ]

    wide = pd.DataFrame.from_records(rows)
    wide.to_csv(out_dir / "source_detection_technique_summary.csv", index=False)

    technique_rows = []
    technique_cols = [
        ("event_prediction", "Event prediction"),
        ("stepfit_matched_filter", "Local step fit + matched filter"),
        ("stacking_result", "Event stacking"),
        ("empirical_controls", "Empirical controls"),
        ("special_controls", "Special diagnostics"),
        ("quality_timing", "Quality/timing"),
        ("final_status", "Final decision"),
    ]
    for row in rows:
        for key, technique in technique_cols:
            technique_rows.append(
                {
                    "source": row["source"],
                    "source_class": row["source_class"],
                    "technique": technique,
                    "result": row[key],
                    "interpretation": row["interpretation"] if technique == "Final decision" else "",
                }
            )
    long = pd.DataFrame.from_records(technique_rows)
    long.to_csv(out_dir / "source_detection_technique_summary_long.csv", index=False)

    md_lines = [
        "# Source Detection Technique Summary",
        "",
        "Conservative status labels are used. Earth is confirmed as a positive-control validation of the pipeline. The Sun now has high-control candidate channels, but is not promoted to a final confirmed astrophysical detection here.",
        "",
        _markdown_table(wide),
        "",
        "## Long Technique Table",
        "",
        _markdown_table(long),
        "",
    ]
    (out_dir / "source_detection_technique_summary.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(out_dir / "source_detection_technique_summary.md")
    print(out_dir / "source_detection_technique_summary.csv")
    print(out_dir / "source_detection_technique_summary_long.csv")


if __name__ == "__main__":
    main()
