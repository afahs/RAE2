#!/usr/bin/env python
"""Run the Sun-focused moving-source validation suite."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from rylevonberg.table_io import read_table

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rylevonberg.events import predict_events
from rylevonberg.constants import add_frequency_mhz_column
from rylevonberg.detection import baseline_matrix, event_template
from rylevonberg.plotting import plot_event_window
from rylevonberg.quality import event_window_quality
from rylevonberg.solar_controls import (
    MovingBodyOffsetConfig,
    date_cluster_summary,
    event_burst_metrics,
    generate_moving_body_offset_controls,
)
from rylevonberg.source_summary import (
    add_offsource_pvalues,
    aggregate_source_level,
    block_jackknife_from_profiles,
    final_source_summary,
)
from rylevonberg.stacking import aligned_profiles, stack_profiles
from rylevonberg.util import datetime_ns, ensure_dir, robust_sigma, write_json


def _read_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    return read_table(path, low_memory=False, **kwargs)


def _plot_profiles(stacked: pd.DataFrame, output_path: Path, title: str) -> None:
    if stacked.empty:
        return
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3.5))
    group_cols = [c for c in ["event_type", "frequency_band", "antenna"] if c in stacked.columns]
    for _, grp in stacked.groupby(group_cols, dropna=False, sort=True):
        label = " ".join(str(grp.iloc[0].get(c)) for c in group_cols)
        ax.plot(grp["t_bin_sec"] / 60.0, grp["mean"], lw=1, label=label)
    ax.axvline(0.0, color="black", lw=1)
    ax.set_xlabel("Relative time (min)")
    ax.set_ylabel("Mean normalized profile")
    ax.set_title(title)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _top_event_plots(clean: pd.DataFrame, scored: pd.DataFrame, out_dir: Path, window_s: float, n: int = 20) -> None:
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


def _frequency_antenna_coherence(source_level: pd.DataFrame) -> pd.DataFrame:
    if source_level.empty:
        return pd.DataFrame()
    rows = []
    for freq, grp in source_level.groupby("frequency_band", sort=True):
        rv1 = grp[grp["antenna"].eq("rv1_coarse")]
        rv2 = grp[grp["antenna"].eq("rv2_coarse")]
        rv1_snr = float(rv1["stacked_snr"].iloc[0]) if not rv1.empty else pd.NA
        rv2_snr = float(rv2["stacked_snr"].iloc[0]) if not rv2.empty else pd.NA
        rows.append(
            {
                "frequency_band": int(freq),
                "frequency_mhz": float(grp["frequency_mhz"].iloc[0]),
                "rv1_stacked_snr": rv1_snr,
                "rv2_stacked_snr": rv2_snr,
                "rv1_sign": float(pd.Series([rv1_snr]).dropna().apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0).iloc[0]) if pd.notna(rv1_snr) else pd.NA,
                "rv2_sign": float(pd.Series([rv2_snr]).dropna().apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0).iloc[0]) if pd.notna(rv2_snr) else pd.NA,
                "abs_max_snr": max(abs(rv1_snr) if pd.notna(rv1_snr) else 0.0, abs(rv2_snr) if pd.notna(rv2_snr) else 0.0),
                "stronger_antenna": "rv1_coarse" if (pd.notna(rv1_snr) and abs(rv1_snr) >= (abs(rv2_snr) if pd.notna(rv2_snr) else -1)) else "rv2_coarse",
                "antenna_note": "rv1 upper V away from Moon; rv2 lower V toward Moon",
            }
        )
    return pd.DataFrame.from_records(rows).sort_values("frequency_band")


def _fast_stack_summary(clean: pd.DataFrame, events: pd.DataFrame, window_s: float) -> pd.DataFrame:
    """Control-stack summary without materializing aligned profile rows."""
    if events.empty:
        return pd.DataFrame()
    clean_groups = {
        (freq, antenna): group.sort_values("time").reset_index(drop=True)
        for (freq, antenna), group in clean.groupby(["frequency_band", "antenna"], dropna=False, sort=True)
    }
    accum: dict[tuple[object, object, object], dict[str, object]] = {}
    for _, ev in events.iterrows():
        freq = ev.get("frequency_band")
        antenna = ev.get("antenna")
        group = clean_groups.get((freq, antenna))
        if group is None or group.empty:
            continue
        times = pd.DatetimeIndex(group["time"])
        t_ns = datetime_ns(times)
        event_ns = pd.Timestamp(ev["predicted_event_time"]).value
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
        B = baseline_matrix(tr, 1)
        beta, *_ = np.linalg.lstsq(B, yy, rcond=None)
        yd = yy - B @ beta
        sig = robust_sigma(yd)
        if np.isfinite(sig) and sig > 0:
            yd = yd / sig
        tmpl = event_template(tr, str(ev["event_type"]))
        key = (ev["source_name"], freq, antenna)
        slot = accum.setdefault(key, {"values": [], "templates": [], "events": set()})
        slot["values"].append(yd)
        slot["templates"].append(tmpl)
        slot["events"].add((ev.get("event_id"), str(ev.get("predicted_event_time"))))
    rows = []
    for (source, freq, antenna), slot in accum.items():
        vals = np.concatenate(slot["values"]) if slot["values"] else np.array([], dtype=float)
        tmpl = np.concatenate(slot["templates"]) if slot["templates"] else np.array([], dtype=float)
        denom = float(np.dot(tmpl, tmpl))
        amp = float(np.dot(vals, tmpl) / denom) if denom > 0 else np.nan
        sig = robust_sigma(vals)
        snr = float(amp * np.sqrt(denom) / sig) if np.isfinite(sig) and sig > 0 else np.nan
        rows.append(
            {
                "source_name": source,
                "frequency_band": freq,
                "antenna": antenna,
                "n_events": int(len(slot["events"])),
                "stacked_amplitude": amp,
                "stacked_snr": snr,
                "bootstrap_std": np.nan,
                "jackknife_std": np.nan,
            }
        )
    return add_frequency_mhz_column(pd.DataFrame.from_records(rows))


def _classify_sun(source_level: pd.DataFrame, clusters: pd.DataFrame) -> tuple[str, str]:
    if source_level.empty:
        return "unresolved", "no source-level rows were produced"
    ranked = source_level.assign(abs_snr=source_level["stacked_snr"].abs()).sort_values("abs_snr", ascending=False)
    best = ranked.iloc[0]
    abs_snr = float(abs(best["stacked_snr"]))
    rand_p = best.get("randomized_empirical_p", pd.NA)
    off_p = best.get("offsource_empirical_p", pd.NA)
    clean = best.get("clean_fraction", pd.NA)
    timing = abs(float(best.get("median_timing_offset_s", 999.0))) if pd.notna(best.get("median_timing_offset_s", pd.NA)) else 0.0
    block_leverage = best.get("max_block_leverage", pd.NA)
    rand_ok = pd.notna(rand_p) and float(rand_p) <= 0.05
    off_ok = pd.notna(off_p) and float(off_p) <= 0.05
    clean_ok = pd.notna(clean) and float(clean) >= 0.5
    timing_ok = timing <= 60.0
    block_ok = pd.isna(block_leverage) or abs(float(block_leverage)) <= max(abs(float(best.get("stacked_amplitude", 0.0))) * 1.5, 0.1)
    burst_dates = int((clusters.get("n_burst_like", pd.Series(dtype=int)) > 0).sum()) if clusters is not None and not clusters.empty else 0

    if abs_snr >= 8.0 and rand_ok and off_ok and clean_ok and timing_ok and block_ok:
        return "solar_candidate", "strong stack passes randomized/off-ephemeris controls with acceptable quality and timing"
    if abs_snr >= 5.0 and pd.notna(off_p) and float(off_p) > 0.1:
        return "systematic_like", "off-ephemeris controls are comparable to the real solar stack"
    if burst_dates >= 2:
        return "solar_burst_candidate", "candidate/burst-like rows cluster on multiple dates, but the repeated stack is not fully control-clean"
    return "unresolved", "solar evidence is present but missing one or more major controls"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleaned", default="outputs/control_survey_earth_sun_postnov1974_v2/01_ingest/cleaned_timeseries.csv")
    parser.add_argument("--earth-sun-root", default="outputs/control_survey_earth_sun_postnov1974_v2")
    parser.add_argument("--output-root", default="outputs/sun_focused_validation")
    parser.add_argument("--window", type=float, default=600.0)
    parser.add_argument("--prediction-cadence-seconds", type=float, default=300.0)
    parser.add_argument("--frequencies", nargs="+", type=int, default=[4, 5, 6, 7, 8])
    parser.add_argument("--antennas", nargs="+", default=["rv1_coarse", "rv2_coarse"])
    parser.add_argument("--annulus-positions", type=int, default=4)
    parser.add_argument("--radial-offsets", nargs="+", type=float, default=[2.0, 5.0, 10.0])
    parser.add_argument("--force-predict", action="store_true", help="Regenerate off-ephemeris predicted events even if the table exists.")
    args = parser.parse_args()

    out_root = ensure_dir(args.output_root)
    reports = ensure_dir("outputs/reports")
    ensure_dir("outputs/summary")

    clean = _read_csv(args.cleaned, parse_dates=["time"])
    root = Path(args.earth_sun_root)
    events = _read_csv(root / "02_events" / "predicted_events.csv", parse_dates=["predicted_event_time"])
    scored = _read_csv(root / "07_scoring" / "scored_detections.csv", parse_dates=["predicted_event_time"])
    stack_summary = _read_csv(root / "04_stack" / "stacked_detection_summary.csv")
    profiles = _read_csv(root / "04_stack" / "event_aligned_profiles.csv", parse_dates=["predicted_event_time"])

    sun_events = events[events["source_name"].eq("sun")].reset_index(drop=True)
    sun_scored = scored[scored["source_name"].eq("sun")].reset_index(drop=True)
    sun_stack = stack_summary[stack_summary["source_name"].eq("sun")].reset_index(drop=True)
    sun_profiles = profiles[profiles["source_name"].eq("sun")].reset_index(drop=True)

    controls = generate_moving_body_offset_controls(
        MovingBodyOffsetConfig(
            annulus_positions=args.annulus_positions,
            radial_offsets_deg=tuple(args.radial_offsets),
        )
    )
    controls_dir = ensure_dir(out_root / "controls")
    controls.to_csv(controls_dir / "sun_offephemeris_controls.csv", index=False)
    off_events_path = controls_dir / "sun_offephemeris_predicted_events.csv"
    if off_events_path.exists() and not args.force_predict:
        off_events = _read_csv(off_events_path, parse_dates=["predicted_event_time"])
    else:
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
        off_events = off_events.merge(
            controls[["source_name", "parent_source", "control_name", "control_type", "offset_deg", "offset_pa_deg", "offset_east_deg", "offset_north_deg", "notes"]],
            on="source_name",
            how="left",
        )
        off_events.to_csv(off_events_path, index=False)
    off_events = off_events[
        off_events["frequency_band"].isin(args.frequencies) & off_events["antenna"].astype(str).isin(set(args.antennas))
    ].reset_index(drop=True)
    off_stack = _fast_stack_summary(clean, off_events, args.window)
    if not off_stack.empty:
        off_stack["window_s"] = float(args.window)
        off_stack = off_stack.merge(
            off_events[["source_name", "parent_source", "control_name", "control_type", "offset_deg", "offset_pa_deg"]].drop_duplicates("source_name"),
            on="source_name",
            how="left",
        )
    off_stack.to_csv(controls_dir / "sun_offephemeris_stack_summary.csv", index=False)

    quality = event_window_quality(clean, sun_events, args.window)
    burst = event_burst_metrics(clean, sun_events, args.window)
    block = block_jackknife_from_profiles(sun_profiles)
    source_level = aggregate_source_level(sun_events, sun_scored, sun_stack, quality, block, window_s=args.window)
    source_level = add_offsource_pvalues(source_level, off_stack)
    coherence = _frequency_antenna_coherence(source_level)
    clusters = date_cluster_summary(sun_scored, burst)
    classification, reason = _classify_sun(source_level, clusters)

    source_level.to_csv(out_root / "sun_source_level_results.csv", index=False)
    quality.to_csv(out_root / "sun_event_window_quality.csv", index=False)
    burst.to_csv(out_root / "sun_event_burst_metrics.csv", index=False)
    clusters.to_csv(out_root / "sun_date_cluster_summary.csv", index=False)
    coherence.to_csv(out_root / "sun_frequency_antenna_coherence.csv", index=False)

    assets = ensure_dir(reports / "sun_focused_validation_assets")
    strongest = sun_stack.assign(abs_snr=sun_stack["stacked_snr"].abs()).sort_values("abs_snr", ascending=False).head(8)
    for _, row in strongest.head(6).iterrows():
        freq = row["frequency_band"]
        ant = row["antenna"]
        sub = sun_profiles[(sun_profiles["frequency_band"].eq(freq)) & (sun_profiles["antenna"].eq(ant))]
        split_stack, split_summary = stack_profiles(sub, by=["source_name", "frequency_band", "antenna", "event_type"], n_bootstrap=20)
        split_stack.to_csv(assets / f"sun_band{int(freq)}_{ant}_event_type_stack.csv", index=False)
        split_summary.to_csv(assets / f"sun_band{int(freq)}_{ant}_event_type_summary.csv", index=False)
        _plot_profiles(split_stack, assets / f"sun_band{int(freq)}_{ant}_event_type_stack.png", f"Sun band {int(freq)} {ant}")
    _top_event_plots(clean, sun_scored, assets / "event_thumbnails", args.window, n=24)

    top_level = source_level.assign(abs_snr=source_level["stacked_snr"].abs()).sort_values("abs_snr", ascending=False).head(10)
    lines = [
        "# Sun Focused Validation",
        "",
        "## Result",
        "",
        f"Current Sun classification: `{classification}`.",
        "",
        reason,
        "",
        "This is not a confirmed solar detection unless the classification is explicitly upgraded after reviewing the "
        "moving-source controls and date-block diagnostics. The Sun is treated as a moving, variable source.",
        "",
        "## Strongest Stacked Channels",
        "",
        strongest[["frequency_band", "frequency_mhz", "antenna", "n_events", "stacked_amplitude", "stacked_snr", "bootstrap_std", "jackknife_std"]].to_string(index=False),
        "",
        "## Source-Level Rows With Off-Ephemeris Controls",
        "",
        top_level.drop(columns=["abs_snr"], errors="ignore").to_string(index=False),
        "",
        "## Frequency / Antenna Coherence",
        "",
        coherence.to_string(index=False),
        "",
        "## Date Clusters",
        "",
        clusters.head(12).to_string(index=False),
        "",
        "## Event-Level Grades",
        "",
        pd.DataFrame([sun_scored["detection_grade"].value_counts().to_dict()]).to_string(index=False),
        "",
        "## Diagnostics Generated",
        "",
        f"- Moving-source controls: `{controls_dir / 'sun_offephemeris_controls.csv'}`",
        f"- Off-ephemeris event table: `{controls_dir / 'sun_offephemeris_predicted_events.csv'}`",
        f"- Off-ephemeris stack summary: `{controls_dir / 'sun_offephemeris_stack_summary.csv'}`",
        f"- Event-type split stacks and plots: `{assets}`",
        f"- Event thumbnails: `{assets / 'event_thumbnails'}`",
        f"- Burst metrics: `{out_root / 'sun_event_burst_metrics.csv'}`",
        f"- Date clusters: `{out_root / 'sun_date_cluster_summary.csv'}`",
        "",
        "## Interpretation",
        "",
        "The key test is whether the real solar stack is stronger than nearby moving fake-Sun tracks. "
        "A real solar result should also be timing-compatible, not dominated by one date block, and should have plausible "
        "frequency/antenna behavior given that rv1_coarse is the upper V away from the Moon and rv2_coarse is the lower V toward the Moon.",
    ]
    (reports / "sun_focused_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary_path = Path("outputs/summary/source_level_results.csv")
    if summary_path.exists():
        existing = _read_csv(summary_path)
        existing = existing[~existing["source"].astype(str).eq("sun")]
        combined = pd.concat([existing, source_level], ignore_index=True)
    else:
        combined = source_level
    combined.to_csv(summary_path, index=False)
    final = final_source_summary(combined)
    final.to_csv("outputs/summary/source_final_summary.csv", index=False)

    write_json(
        out_root / "sun_focused_validation_config.json",
        {
            "cleaned": args.cleaned,
            "earth_sun_root": args.earth_sun_root,
            "window": args.window,
            "prediction_cadence_seconds": args.prediction_cadence_seconds,
            "frequencies": args.frequencies,
            "antennas": args.antennas,
            "annulus_positions": args.annulus_positions,
            "radial_offsets": args.radial_offsets,
        },
    )
    print("sun_report: outputs/reports/sun_focused_validation.md")
    print("sun_source_level: outputs/sun_focused_validation/sun_source_level_results.csv")
    print("source_level_results: outputs/summary/source_level_results.csv")
    print("source_final_summary: outputs/summary/source_final_summary.csv")


if __name__ == "__main__":
    main()
