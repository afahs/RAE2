#!/usr/bin/env python
"""Audit Sun events for Earth-near-lunar-limb contamination.

This is a focused diagnostic, not a blind search. It answers whether the
current lower-V Sun candidate changes when Sun occultation events are rejected
if Earth is within a configurable angular distance of the lunar limb at the
same time.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rylevonberg.table_io import read_table

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.events import predict_events
from rylevonberg.util import ensure_dir, write_json


PLAYGROUND_SCRIPT = ROOT / "scripts" / "run_baseline_model_playground.py"


def _load_playground():
    spec = importlib.util.spec_from_file_location("baseline_playground", PLAYGROUND_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load helpers from {PLAYGROUND_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PLAY = _load_playground()


def _read(path: Path, **kwargs) -> pd.DataFrame:
    return read_table(path, low_memory=False, **kwargs)


def _source_rows(name: str) -> pd.DataFrame:
    if name.lower() == "earth":
        return pd.DataFrame(
            [{"source_name": "earth", "kind": "earth", "body_name": "earth", "frame": "fk4"}]
        )
    return pd.DataFrame(
        [{"source_name": name.lower(), "kind": "body", "body_name": name.lower(), "frame": "fk4"}]
    )


def _event_key(df: pd.DataFrame) -> pd.Series:
    return (
        df["source_name"].astype(str)
        + "|"
        + df["event_type"].astype(str)
        + "|"
        + pd.to_datetime(df["predicted_event_time"]).astype(str)
        + "|"
        + df["frequency_band"].astype(str)
        + "|"
        + df["antenna"].astype(str)
    )


def _count_summary(label: str, events: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "selection": label,
            "frequency_band": "all",
            "frequency_mhz": np.nan,
            "antenna": "all",
            "event_rows": int(len(events)),
            "unique_limb_crossings": int(
                events[["event_type", "predicted_event_time"]].drop_duplicates().shape[0]
            )
            if not events.empty
            else 0,
        }
    ]
    if not events.empty:
        for (band, mhz, antenna), grp in events.groupby(
            ["frequency_band", "frequency_mhz", "antenna"], dropna=False, sort=True
        ):
            rows.append(
                {
                    "selection": label,
                    "frequency_band": int(band),
                    "frequency_mhz": float(mhz) if pd.notna(mhz) else np.nan,
                    "antenna": str(antenna),
                    "event_rows": int(len(grp)),
                    "unique_limb_crossings": int(
                        grp[["event_type", "predicted_event_time"]].drop_duplicates().shape[0]
                    ),
                }
            )
    return pd.DataFrame.from_records(rows)


def _evaluate_channel(
    clean: pd.DataFrame,
    events: pd.DataFrame,
    label: str,
    band: int,
    frequency_mhz: float,
    antenna: str,
    window_s: float,
    timing_offset_s: float,
    exclusion_s: float,
    out_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source_row = pd.Series(
        {
            "source_name": "sun",
            "target_band": int(band),
            "target_frequency_mhz": float(frequency_mhz),
            "target_antenna": str(antenna),
            "target_window_s": float(window_s),
            "best_timing_offset_s": float(timing_offset_s),
        }
    )
    event_fits = PLAY._event_level_table(clean, events, source_row, float(exclusion_s))
    event_fits["selection"] = label
    summary = PLAY._aggregate(
        event_fits,
        [
            "selection",
            "source_name",
            "frequency_band",
            "frequency_mhz",
            "antenna",
            "window_s",
            "timing_offset_s",
            "method",
        ],
    )
    event_type = PLAY._aggregate(
        event_fits,
        [
            "selection",
            "source_name",
            "frequency_band",
            "frequency_mhz",
            "antenna",
            "window_s",
            "timing_offset_s",
            "method",
            "event_type",
        ],
    )
    event_fits.to_csv(out_dir / f"{label}_sun_lower_v_event_fits.csv", index=False)
    summary.to_csv(out_dir / f"{label}_sun_lower_v_method_summary.csv", index=False)
    event_type.to_csv(out_dir / f"{label}_sun_lower_v_event_type_summary.csv", index=False)
    return event_fits, summary, event_type


def _plot_method_comparison(summary: pd.DataFrame, out_dir: Path) -> Path:
    if summary.empty:
        return out_dir / "sun_lower_v_earth_exclusion_method_comparison.png"
    methods = [m for m in PLAY.METHOD_LABELS if m in set(summary["method"])]
    labels = [PLAY.METHOD_LABELS.get(m, m) for m in methods]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    width = 0.38
    x = np.arange(len(methods))
    colors = {"no_exclusion": "#4c78a8", "earth_limb_gt_3deg": "#f58518"}
    for j, selection in enumerate(["no_exclusion", "earth_limb_gt_3deg"]):
        sub = summary[summary["selection"].eq(selection)].set_index("method")
        vals = [float(sub.loc[m, "robust_stack_snr"]) if m in sub.index else np.nan for m in methods]
        ax.bar(x + (j - 0.5) * width, vals, width=width, label=selection.replace("_", " "), color=colors[selection])
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(3, color="0.65", lw=0.8, ls="--")
    ax.axhline(-3, color="0.65", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Robust stack SNR")
    ax.set_title("Sun lower V, 6.55 MHz: effect of rejecting Earth-near-limb events")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = out_dir / "sun_lower_v_earth_exclusion_method_comparison.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_limb_hist(annotated: pd.DataFrame, filtered: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    vals = pd.to_numeric(annotated["limb_exclusion_nearest_abs_deg"], errors="coerce").dropna()
    ax.hist(vals, bins=np.linspace(0, min(60, max(6, float(vals.max()) if len(vals) else 6)), 60), color="#4c78a8")
    ax.axvline(3.0, color="#d95f02", lw=2, label="3 deg veto")
    ax.set_xlabel("Earth absolute distance from lunar limb at Sun event (deg)")
    ax.set_ylabel("Sun event rows")
    ax.set_title("Earth-near-limb distribution for predicted Sun events")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = out_dir / "sun_event_earth_limb_distance_histogram.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _format_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "No rows."
    show = df.head(max_rows).copy()
    return "```\n" + show.to_string(index=False) + "\n```"


def _write_report(
    out_dir: Path,
    args: argparse.Namespace,
    counts: pd.DataFrame,
    comparison: pd.DataFrame,
    event_type: pd.DataFrame,
    paths: list[Path],
) -> None:
    selected_counts = counts[
        counts["frequency_band"].astype(str).eq(str(args.target_band))
        & counts["antenna"].astype(str).eq(args.target_antenna)
    ].copy()
    top = comparison.sort_values(["selection", "fit_quality_score"], ascending=[True, False]).groupby("selection").head(3)
    lines = [
        "# Sun/Earth Limb-Exclusion Audit",
        "",
        "Purpose: test whether Sun occultation windows are contaminated by Earth being close to the lunar limb at the same predicted Sun event time.",
        "",
        "Important interpretation point: negative fitted amplitude is not negative physical emission. It means the observed local power change has the opposite sign from the occultation template for a positive source. That can be caused by contamination, baseline/background structure, antenna response, or a wrong sign convention. This audit does not clip signs; it tests whether a known contaminant explains the sign.",
        "",
        f"Earth-limb veto: reject Sun event rows where Earth is within {args.limb_exclusion_deg:g} degrees of the lunar limb.",
        "",
        f"Targeted lower-V channel checked: band {args.target_band}, {args.target_frequency_mhz:g} MHz, `{args.target_antenna}`, window {args.window_s:g} s, timing offset {args.timing_offset_s:g} s.",
        "",
        "## Event Counts",
        "",
        _format_table(selected_counts),
        "",
        "## Best Method Rows",
        "",
        _format_table(
            top[
                [
                    "selection",
                    "method",
                    "n_events",
                    "median_amplitude",
                    "robust_stack_snr",
                    "event_sign_fraction",
                    "median_delta_bic",
                    "fit_quality_score",
                ]
            ]
        ),
        "",
        "## Event-Type Split",
        "",
        _format_table(
            event_type[
                [
                    "selection",
                    "method",
                    "event_type",
                    "n_events",
                    "median_amplitude",
                    "robust_stack_snr",
                    "fit_quality_score",
                ]
            ].sort_values(["selection", "method", "event_type"])
        ),
        "",
        "## Figures",
        "",
        *[f"- {path}" for path in paths],
        "",
        "## Pipeline Implication",
        "",
        "- The existing event-prediction code already supports limb-contaminant metadata and exclusion.",
        "- The recent planetary science-baseline run did not request this exclusion, so the Sun results were not filtered for Earth within 3 degrees of the lunar limb.",
        "- For Sun confirmation runs, the recommended default is to pass Earth as a limb-exclusion source with a 3 degree threshold and to report both the pre-veto and post-veto event counts.",
        "- Negative amplitudes should remain visible as signed diagnostics. They should not be converted to positive emission unless a physical sign convention is explicitly modeled and validated.",
    ]
    (out_dir / "sun_earth_limb_exclusion_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", default="outputs/control_survey_earth_sun_postnov1974_v2/01_ingest/cleaned_timeseries.csv")
    parser.add_argument("--output-root", default="outputs/sun_earth_limb_exclusion_audit_v1")
    parser.add_argument("--frequencies", nargs="+", type=int, default=list(range(1, 10)))
    parser.add_argument("--antennas", nargs="+", default=["rv1_coarse", "rv2_coarse"])
    parser.add_argument("--prediction-cadence-seconds", type=float, default=300.0)
    parser.add_argument("--limb-exclusion-deg", type=float, default=3.0)
    parser.add_argument("--target-band", type=int, default=8)
    parser.add_argument("--target-frequency-mhz", type=float, default=6.55)
    parser.add_argument("--target-antenna", default="rv2_coarse")
    parser.add_argument("--window-s", type=float, default=600.0)
    parser.add_argument("--timing-offset-s", type=float, default=60.0)
    parser.add_argument("--sideband-exclusion-s", type=float, default=120.0)
    args = parser.parse_args()

    out_dir = ensure_dir(ROOT / args.output_root)
    clean = _read(ROOT / args.clean, parse_dates=["time"])
    sun = _source_rows("sun")
    earth = _source_rows("earth")

    common = dict(
        target_frame="fk4",
        equinox="B1950",
        ephemeris="builtin",
        max_gap_seconds=600.0,
        prediction_cadence_seconds=float(args.prediction_cadence_seconds),
        frequencies=args.frequencies,
        antennas=args.antennas,
    )
    annotated, states = predict_events(
        clean,
        sun,
        limb_exclusion_sources_df=earth,
        limb_exclusion_deg=None,
        **common,
    )
    filtered, filtered_states = predict_events(
        clean,
        sun,
        limb_exclusion_sources_df=earth,
        limb_exclusion_deg=float(args.limb_exclusion_deg),
        **common,
    )

    annotated.to_csv(out_dir / "sun_events_annotated_with_earth_limb.csv", index=False)
    filtered.to_csv(out_dir / "sun_events_earth_limb_gt_3deg.csv", index=False)
    states.to_csv(out_dir / "sun_limb_visibility_states.csv", index=False)

    veto_mask = pd.to_numeric(annotated["limb_exclusion_nearest_abs_deg"], errors="coerce") <= float(args.limb_exclusion_deg)
    excluded = annotated[veto_mask].copy()
    excluded.to_csv(out_dir / "sun_events_rejected_by_earth_limb_3deg.csv", index=False)

    counts = pd.concat(
        [
            _count_summary("no_exclusion", annotated),
            _count_summary("earth_limb_gt_3deg", filtered),
            _count_summary("rejected_by_earth_limb_3deg", excluded),
        ],
        ignore_index=True,
    )
    counts.to_csv(out_dir / "earth_limb_exclusion_event_counts.csv", index=False)

    no_fits, no_summary, no_event_type = _evaluate_channel(
        clean,
        annotated,
        "no_exclusion",
        args.target_band,
        args.target_frequency_mhz,
        args.target_antenna,
        args.window_s,
        args.timing_offset_s,
        args.sideband_exclusion_s,
        out_dir,
    )
    filt_fits, filt_summary, filt_event_type = _evaluate_channel(
        clean,
        filtered,
        "earth_limb_gt_3deg",
        args.target_band,
        args.target_frequency_mhz,
        args.target_antenna,
        args.window_s,
        args.timing_offset_s,
        args.sideband_exclusion_s,
        out_dir,
    )
    summary = pd.concat([no_summary, filt_summary], ignore_index=True)
    event_type = pd.concat([no_event_type, filt_event_type], ignore_index=True)
    summary.to_csv(out_dir / "sun_lower_v_earth_exclusion_method_comparison.csv", index=False)
    event_type.to_csv(out_dir / "sun_lower_v_earth_exclusion_event_type_comparison.csv", index=False)

    paths = [
        _plot_limb_hist(annotated, filtered, out_dir),
        _plot_method_comparison(summary, out_dir),
    ]
    _write_report(out_dir, args, counts, summary, event_type, paths)
    write_json(out_dir / "run_config.json", vars(args))
    print(out_dir / "sun_earth_limb_exclusion_audit.md")
    print(out_dir / "sun_lower_v_earth_exclusion_method_comparison.csv")


if __name__ == "__main__":
    main()
