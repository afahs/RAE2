#!/usr/bin/env python
"""Run occultation confirmation diagnostics for all major planets except Pluto."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rylevonberg.table_io import read_table

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rylevonberg.events import predict_events
from rylevonberg.quality import event_window_quality
from rylevonberg.util import ensure_dir, write_json


ROOT = Path(__file__).resolve().parents[1]
CONFIRMATION_SCRIPT = ROOT / "scripts" / "run_source_detection_confirmation_suite.py"


def _load_confirmation_helpers():
    spec = importlib.util.spec_from_file_location("source_confirmation", CONFIRMATION_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load confirmation helpers from {CONFIRMATION_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONFIRM = _load_confirmation_helpers()


PLANETS = ["mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune"]


def _score_column(table: pd.DataFrame) -> str:
    return "robust_stack_snr" if "robust_stack_snr" in table.columns else "stacked_snr"


def _abs_score(table: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(table[_score_column(table)], errors="coerce").abs()


def _log(message: str) -> None:
    print(f"[planet-survey] {message}", flush=True)


def _read(path: Path, **kwargs) -> pd.DataFrame:
    return read_table(path, low_memory=False, **kwargs)


def _planet_sources(planets: list[str]) -> pd.DataFrame:
    rows = []
    for planet in planets:
        rows.append(
            {
                "source_name": planet,
                "kind": "earth" if planet == "earth" else "body",
                "body_name": planet,
                "ra_deg": np.nan,
                "dec_deg": np.nan,
                "frame": "fk4",
                "notes": "planet confirmation survey",
            }
        )
    return pd.DataFrame.from_records(rows)


def _aggregate_for_specs(
    groups: dict[tuple[int, str], tuple[pd.DataFrame, np.ndarray]],
    events: pd.DataFrame,
    windows: list[float],
    timing_offset: float = 0.0,
    baseline_mode: str = "sideband_linear",
    sideband_exclusion_seconds: float = 120.0,
) -> pd.DataFrame:
    rows = []
    for window in windows:
        contrib = CONFIRM._event_contributions(
            groups,
            events,
            float(window),
            float(timing_offset),
            baseline_mode=baseline_mode,
            sideband_exclusion_seconds=sideband_exclusion_seconds,
        )
        agg = CONFIRM._aggregate(contrib, ["source_name", "frequency_band", "frequency_mhz", "antenna", "window_s", "timing_offset_s"])
        if not agg.empty:
            rows.append(agg)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _source_events(events: pd.DataFrame, source_name: str, freq: int | None = None, antenna: str | None = None) -> pd.DataFrame:
    out = events[events["source_name"].astype(str).eq(source_name)].copy()
    if freq is not None:
        out = out[out["frequency_band"].astype(int).eq(int(freq))]
    if antenna is not None:
        out = out[out["antenna"].astype(str).eq(str(antenna))]
    return out.reset_index(drop=True)


def _confirm_source(
    clean: pd.DataFrame,
    groups: dict[tuple[int, str], tuple[pd.DataFrame, np.ndarray]],
    events: pd.DataFrame,
    source_name: str,
    target_band: int,
    target_antenna: str,
    target_window: float,
    windows: list[float],
    timing_offsets: list[float],
    out_dir: Path,
    strict_min_side_samples: int,
    baseline_mode: str,
    sideband_exclusion_seconds: float,
) -> dict[str, pd.DataFrame]:
    target_events = _source_events(events, source_name, target_band, target_antenna)
    quality = event_window_quality(clean, target_events, target_window)
    quality.to_csv(out_dir / "target_window_quality.csv", index=False)

    timing_rows = []
    timing_contribs: dict[float, pd.DataFrame] = {}
    for offset in timing_offsets:
        contrib = CONFIRM._event_contributions(
            groups,
            target_events,
            target_window,
            float(offset),
            baseline_mode=baseline_mode,
            sideband_exclusion_seconds=sideband_exclusion_seconds,
        )
        timing_contribs[float(offset)] = contrib
        timing_rows.append(CONFIRM._aggregate(contrib, ["source_name", "frequency_band", "frequency_mhz", "antenna", "window_s", "timing_offset_s"]))
    timing_scan = pd.concat(timing_rows, ignore_index=True) if timing_rows else pd.DataFrame()
    best_offset = float(timing_scan.assign(abs_snr=timing_scan["stacked_snr"].abs()).sort_values("abs_snr", ascending=False).iloc[0]["timing_offset_s"]) if not timing_scan.empty else 0.0
    best_contrib = timing_contribs.get(best_offset, pd.DataFrame())

    window_rows = []
    for window in windows:
        contrib = CONFIRM._event_contributions(
            groups,
            target_events,
            float(window),
            best_offset,
            baseline_mode=baseline_mode,
            sideband_exclusion_seconds=sideband_exclusion_seconds,
        )
        window_rows.append(CONFIRM._aggregate(contrib, ["source_name", "frequency_band", "frequency_mhz", "antenna", "window_s", "timing_offset_s"]))
    window_sweep = pd.concat(window_rows, ignore_index=True) if window_rows else pd.DataFrame()

    event_type = CONFIRM._aggregate(best_contrib, ["source_name", "frequency_band", "frequency_mhz", "antenna", "window_s", "timing_offset_s", "event_type"])
    month = CONFIRM._aggregate(best_contrib, ["source_name", "frequency_band", "frequency_mhz", "antenna", "window_s", "timing_offset_s", "month_block"])
    leave_one = CONFIRM._leave_one_month(best_contrib)

    strict_keys = CONFIRM._quality_keys(quality, target_band, target_antenna, strict_min_side_samples)
    strict_contrib = CONFIRM._event_contributions(
        groups,
        target_events,
        target_window,
        best_offset,
        strict_good_keys=strict_keys,
        baseline_mode=baseline_mode,
        sideband_exclusion_seconds=sideband_exclusion_seconds,
    )
    strict_quality = CONFIRM._aggregate(strict_contrib, ["source_name", "frequency_band", "frequency_mhz", "antenna", "window_s", "timing_offset_s"])

    control_specs = [
        (target_band, "rv1_coarse" if target_antenna == "rv2_coarse" else "rv2_coarse", "wrong_antenna"),
        (max(1, target_band - 1), target_antenna, "neighboring_band"),
        (min(9, target_band + 1), target_antenna, "neighboring_band"),
    ]
    controls = []
    for freq, ant, ctype in control_specs:
        if int(freq) == int(target_band) and str(ant) == str(target_antenna):
            continue
        sub = _source_events(events, source_name, int(freq), str(ant))
        contrib = CONFIRM._event_contributions(
            groups,
            sub,
            target_window,
            best_offset,
            baseline_mode=baseline_mode,
            sideband_exclusion_seconds=sideband_exclusion_seconds,
        )
        agg = CONFIRM._aggregate(contrib, ["source_name", "frequency_band", "frequency_mhz", "antenna", "window_s", "timing_offset_s"])
        if not agg.empty:
            agg["control_type"] = ctype
            controls.append(agg)
    wrong_controls = pd.concat(controls, ignore_index=True) if controls else pd.DataFrame()

    tables = {
        "timing_scan": timing_scan,
        "window_sweep": window_sweep,
        "event_type": event_type,
        "month_blocks": month,
        "leave_one_month": leave_one,
        "strict_quality": strict_quality,
        "wrong_controls": wrong_controls,
    }
    for name, table in tables.items():
        table.to_csv(out_dir / f"{name}.csv", index=False)
    return tables


def _classify(tables: dict[str, pd.DataFrame]) -> tuple[str, str]:
    timing = tables["timing_scan"]
    window = tables["window_sweep"]
    event_type = tables["event_type"]
    strict = tables["strict_quality"]
    controls = tables["wrong_controls"]
    leave = tables["leave_one_month"]
    if timing.empty:
        return "not_evaluated", "no timing rows"
    score_col = _score_column(timing)
    best = timing.assign(abs_snr=_abs_score(timing)).sort_values("abs_snr", ascending=False).iloc[0]
    best_snr = float(best[score_col])
    best_offset = float(best["timing_offset_s"])
    strict_snr = float(strict[_score_column(strict)].iloc[0]) if not strict.empty else np.nan
    max_control = float(_abs_score(controls).max()) if not controls.empty else np.nan
    min_leave = float(_abs_score(leave).min()) if not leave.empty else np.nan
    etype_ok = False
    if not event_type.empty and len(event_type) >= 2:
        vals = event_type[_score_column(event_type)].to_numpy(dtype=float)
        etype_ok = bool(np.all(np.sign(vals) == np.sign(best_snr)) and np.nanmin(np.abs(vals)) >= 2.0)
    reasons = []
    if abs(best_snr) < 3:
        reasons.append("best SNR below 3")
    if abs(best_offset) > 120:
        reasons.append(f"large timing offset {best_offset:.0f}s")
    if not np.isfinite(strict_snr) or np.sign(strict_snr) != np.sign(best_snr) or abs(strict_snr) < 3:
        reasons.append(f"strict-quality fails ({strict_snr:.2f})")
    if not etype_ok:
        reasons.append("event-type split weak or inconsistent")
    if np.isfinite(max_control) and max_control >= 0.7 * abs(best_snr):
        reasons.append(f"wrong/neighboring control is comparable ({max_control:.2f})")
    if np.isfinite(min_leave) and min_leave < 3:
        reasons.append(f"leave-one-month minimum below 3 ({min_leave:.2f})")
    if not reasons and abs(best_snr) >= 5:
        return "repeatable_candidate", "passes timing, quality, event-type, control, and month checks"
    if abs(best_snr) >= 3:
        return "unresolved_or_episodic", "; ".join(reasons)
    return "not_detected", "; ".join(reasons)


def _write_report(out_root: Path, summary: pd.DataFrame) -> None:
    lines = [
        "# Planetary Occultation Confirmation Survey",
        "",
        "Sources included: " + ", ".join(str(v) for v in summary["source_name"].to_list()) + ".",
        "",
        "Each planet was scanned across frequency bands 1-9, both Ryle-Vonberg antennas, and selected windows. "
        "The strongest aggregate channel was then passed through timing-offset, window, event-type, strict-quality, wrong-control, and leave-one-month-out checks.",
        "",
        "The primary decision statistic is `robust_stack_snr`, the median per-event template amplitude divided by its robust event-level standard error. "
        "`stacked_snr` is still written for continuity but can be over-sensitive to profile normalization and should not be used alone with sideband baselines.",
        "",
        "## Summary",
        "",
        summary.to_string(index=False),
        "",
        "Statuses are diagnostic labels, not final discovery claims.",
    ]
    (out_root / "planetary_confirmation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", default="outputs/control_survey_earth_sun_postnov1974_v2/01_ingest/cleaned_timeseries.csv")
    parser.add_argument("--output-root", default="outputs/planetary_confirmation_survey")
    parser.add_argument("--planets", nargs="+", default=PLANETS)
    parser.add_argument("--frequencies", nargs="+", type=int, default=list(range(1, 10)))
    parser.add_argument("--antennas", nargs="+", default=["rv1_coarse", "rv2_coarse"])
    parser.add_argument("--scan-windows", nargs="+", type=float, default=[600.0, 900.0])
    parser.add_argument("--confirm-windows", nargs="+", type=float, default=[300.0, 600.0, 900.0, 1200.0])
    parser.add_argument("--timing-offsets", nargs="+", type=float, default=[-600, -510, -420, -330, -240, -180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180, 240, 330, 420, 510, 600])
    parser.add_argument("--prediction-cadence-seconds", type=float, default=300.0)
    parser.add_argument("--strict-min-side-samples", type=int, default=20)
    parser.add_argument("--baseline-mode", default="sideband_linear", choices=["linear_all", "constant_all", "sideband_linear", "sideband_constant", "joint_step_linear", "pre_event_anchor"])
    parser.add_argument("--sideband-exclusion-seconds", type=float, default=120.0)
    parser.add_argument(
        "--sun-exclude-earth-limb-deg",
        type=float,
        default=None,
        help="For Sun runs, reject events where Earth is this many degrees or less from the lunar limb.",
    )
    parser.add_argument("--force-predict", action="store_true")
    args = parser.parse_args()

    out_root = ensure_dir(ROOT / args.output_root)
    event_dir = ensure_dir(out_root / "events")
    _log(f"reading clean table: {args.clean}")
    clean = _read(ROOT / args.clean, parse_dates=["time"])
    groups = CONFIRM._make_groups(clean)
    sources = _planet_sources(args.planets)
    summary_rows = []
    all_events = []
    for _, source in sources.iterrows():
        planet = str(source["source_name"])
        planet_dir = ensure_dir(out_root / planet)
        event_path = event_dir / f"{planet}_predicted_events.csv"
        state_path = event_dir / f"{planet}_limb_visibility_states.csv"
        if event_path.exists() and not args.force_predict:
            _log(f"reading cached events for {planet}")
            events = _read(event_path, parse_dates=["predicted_event_time"])
        else:
            _log(f"predicting events for {planet}")
            exclusion_sources = None
            exclusion_deg = None
            if planet == "sun" and args.sun_exclude_earth_limb_deg is not None:
                exclusion_sources = _planet_sources(["earth"])
                exclusion_deg = float(args.sun_exclude_earth_limb_deg)
                _log(f"{planet}: applying Earth lunar-limb exclusion <= {exclusion_deg:g} deg")
            events, states = predict_events(
                clean,
                pd.DataFrame([source]),
                target_frame="fk4",
                equinox="B1950",
                ephemeris="builtin",
                max_gap_seconds=600.0,
                prediction_cadence_seconds=args.prediction_cadence_seconds,
                frequencies=args.frequencies,
                antennas=args.antennas,
                limb_exclusion_sources_df=exclusion_sources,
                limb_exclusion_deg=exclusion_deg,
            )
            events.to_csv(event_path, index=False)
            states.to_csv(state_path, index=False)
        all_events.append(events)
        if events.empty:
            summary_rows.append({"source_name": planet, "status": "not_evaluated", "decision_reason": "no predicted events"})
            continue
        scan = _aggregate_for_specs(
            groups,
            events,
            args.scan_windows,
            timing_offset=0.0,
            baseline_mode=args.baseline_mode,
            sideband_exclusion_seconds=args.sideband_exclusion_seconds,
        )
        scan.to_csv(planet_dir / "initial_channel_scan.csv", index=False)
        if scan.empty:
            summary_rows.append({"source_name": planet, "status": "not_evaluated", "decision_reason": "no stack rows"})
            continue
        best = scan.assign(abs_snr=_abs_score(scan)).sort_values("abs_snr", ascending=False).iloc[0]
        target_band = int(best["frequency_band"])
        target_antenna = str(best["antenna"])
        target_window = float(best["window_s"])
        _log(f"{planet}: best initial channel B{target_band} {target_antenna} {target_window:.0f}s robust SNR {best[_score_column(scan)]:.2f}")
        tables = _confirm_source(
            clean,
            groups,
            events,
            planet,
            target_band,
            target_antenna,
            target_window,
            args.confirm_windows,
            args.timing_offsets,
            planet_dir,
            args.strict_min_side_samples,
            args.baseline_mode,
            args.sideband_exclusion_seconds,
        )
        status, reason = _classify(tables)
        timing = tables["timing_scan"]
        best_confirm = timing.assign(abs_snr=_abs_score(timing)).sort_values("abs_snr", ascending=False).iloc[0]
        strict = tables["strict_quality"]
        controls = tables["wrong_controls"]
        leave = tables["leave_one_month"]
        confirm_col = _score_column(timing)
        summary_rows.append(
            {
                "source_name": planet,
                "target_band": target_band,
                "target_frequency_mhz": float(best.get("frequency_mhz", np.nan)),
                "target_antenna": target_antenna,
                "target_window_s": target_window,
                "initial_stacked_snr": float(best["stacked_snr"]),
                "initial_robust_stack_snr": float(best.get("robust_stack_snr", np.nan)),
                "best_timing_offset_s": float(best_confirm["timing_offset_s"]),
                "best_confirmed_snr": float(best_confirm["stacked_snr"]),
                "best_confirmed_robust_snr": float(best_confirm.get("robust_stack_snr", np.nan)),
                "n_events": int(best_confirm["n_events"]),
                "n_month_blocks": int(best_confirm["n_month_blocks"]),
                "strict_quality_snr": float(strict["stacked_snr"].iloc[0]) if not strict.empty else np.nan,
                "strict_quality_robust_snr": float(strict.get("robust_stack_snr", pd.Series([np.nan])).iloc[0]) if not strict.empty else np.nan,
                "max_wrong_control_abs_snr": float(controls["stacked_snr"].abs().max()) if not controls.empty else np.nan,
                "max_wrong_control_abs_robust_snr": float(_abs_score(controls).max()) if not controls.empty else np.nan,
                "min_leave_one_month_abs_snr": float(leave["stacked_snr"].abs().min()) if not leave.empty else np.nan,
                "min_leave_one_month_abs_robust_snr": float(_abs_score(leave).min()) if not leave.empty else np.nan,
                "score_column": confirm_col,
                "status": status,
                "decision_reason": reason,
            }
        )
    if all_events:
        pd.concat(all_events, ignore_index=True).to_csv(event_dir / "all_planet_predicted_events.csv", index=False)
    summary = pd.DataFrame.from_records(summary_rows)
    summary.to_csv(out_root / "planetary_confirmation_summary.csv", index=False)
    _write_report(out_root, summary)
    write_json(out_root / "run_config.json", vars(args))
    print(out_root / "planetary_confirmation_report.md")
    print(out_root / "planetary_confirmation_summary.csv")


if __name__ == "__main__":
    main()
