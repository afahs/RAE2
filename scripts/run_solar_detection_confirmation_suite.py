#!/usr/bin/env python
"""Solar confirmation suite for the strongest Ryle-Vonberg candidate."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rylevonberg.table_io import read_table

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rylevonberg.detection import baseline_matrix, event_template
from rylevonberg.ingest import IngestOptions, ingest_csv
from rylevonberg.quality import event_window_quality
from rylevonberg.util import datetime_ns, ensure_dir, robust_sigma, write_json


ROOT = Path(__file__).resolve().parents[1]


def _log(message: str) -> None:
    print(f"[solar-confirmation] {message}", flush=True)


def _read(path: Path, **kwargs) -> pd.DataFrame:
    return read_table(path, low_memory=False, **kwargs)


def _load_clean(args: argparse.Namespace) -> pd.DataFrame:
    _log(f"ingesting full dataset: {args.data}")
    clean, report = ingest_csv(
        args.data,
        IngestOptions(
            start_time=None,
            end_time=None,
            value_columns=("rv1_coarse", "rv2_coarse"),
            gap_factor=args.gap_factor,
            artifact_sigma=args.artifact_sigma,
        ),
    )
    return clean


def _make_groups(clean: pd.DataFrame) -> dict[tuple[int, str], tuple[pd.DataFrame, np.ndarray]]:
    groups = {}
    for (freq, antenna), group in clean.groupby(["frequency_band", "antenna"], dropna=False, sort=True):
        sorted_group = group.sort_values("time").reset_index(drop=True)
        groups[(int(freq), str(antenna))] = (sorted_group, datetime_ns(sorted_group["time"]))
    return groups


def _event_contributions(
    groups: dict[tuple[int, str], tuple[pd.DataFrame, np.ndarray]],
    events: pd.DataFrame,
    window_s: float,
    timing_offset_s: float,
    strict_good_keys: set[tuple[str, int, str]] | None = None,
) -> pd.DataFrame:
    rows = []
    half_ns = int(float(window_s) * 1e9)
    for _, ev in events.iterrows():
        freq = int(ev["frequency_band"])
        antenna = str(ev["antenna"])
        event_time = pd.Timestamp(ev["predicted_event_time"])
        event_key = (str(event_time), freq, antenna)
        if strict_good_keys is not None and event_key not in strict_good_keys:
            continue
        payload = groups.get((freq, antenna))
        if payload is None:
            continue
        group, t_ns = payload
        event_ns = event_time.value
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
        baseline = baseline_matrix(tr, 1)
        beta, *_ = np.linalg.lstsq(baseline, yy, rcond=None)
        yd = yy - baseline @ beta
        sigma = robust_sigma(yd)
        if np.isfinite(sigma) and sigma > 0:
            yd = yd / sigma
        tmpl = event_template(tr, str(ev["event_type"]), timing_offset_sec=float(timing_offset_s))
        den = float(np.dot(tmpl, tmpl))
        if den <= 0:
            continue
        rows.append(
            {
                "source_name": ev.get("source_name"),
                "event_id": ev.get("event_id"),
                "event_type": ev.get("event_type"),
                "predicted_event_time": event_time,
                "month_block": event_time.strftime("%Y-%m"),
                "frequency_band": freq,
                "frequency_mhz": ev.get("frequency_mhz"),
                "antenna": antenna,
                "window_s": float(window_s),
                "timing_offset_s": float(timing_offset_s),
                "n_used_samples": int(len(yd)),
                "num": float(np.dot(yd, tmpl)),
                "den": den,
                "local_sigma": float(sigma) if np.isfinite(sigma) else np.nan,
            }
        )
    return pd.DataFrame.from_records(rows)


def _aggregate(contrib: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    if contrib.empty:
        return pd.DataFrame()
    rows = []
    for keys, group in contrib.groupby(by, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        meta = dict(zip(by, keys))
        num = float(group["num"].sum())
        den = float(group["den"].sum())
        amp = num / den if den > 0 else np.nan
        event_amp = group["num"].to_numpy(dtype=float) / group["den"].to_numpy(dtype=float)
        scatter = robust_sigma(event_amp)
        stack_snr = amp * np.sqrt(den)
        rows.append(
            {
                **meta,
                "n_events": int(group["event_id"].nunique()),
                "n_month_blocks": int(group["month_block"].nunique()),
                "stacked_amplitude": amp,
                "stacked_snr": float(stack_snr),
                "event_amp_median": float(np.nanmedian(event_amp)) if event_amp.size else np.nan,
                "event_amp_robust_sigma": float(scatter) if np.isfinite(scatter) else np.nan,
                "event_sign_fraction": float((np.sign(event_amp) == np.sign(amp)).mean()) if np.isfinite(amp) and event_amp.size else np.nan,
            }
        )
    return pd.DataFrame.from_records(rows)


def _quality_keys(quality: pd.DataFrame, freq: int, antenna: str) -> set[tuple[str, int, str]]:
    q = quality[
        quality["frequency_band"].astype(int).eq(int(freq))
        & quality["antenna"].astype(str).eq(str(antenna))
        & quality["primary_quality_failure"].fillna("").astype(str).eq("")
        & (pd.to_numeric(quality["valid_samples_pre"], errors="coerce").fillna(0) >= 20)
        & (pd.to_numeric(quality["valid_samples_post"], errors="coerce").fillna(0) >= 20)
    ].copy()
    return set((str(pd.Timestamp(r["predicted_event_time"])), int(r["frequency_band"]), str(r["antenna"])) for _, r in q.iterrows())


def _leave_one_month(contrib: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if contrib.empty:
        return pd.DataFrame()
    months = sorted(contrib["month_block"].dropna().unique())
    full = _aggregate(contrib, ["source_name", "frequency_band", "antenna", "window_s", "timing_offset_s"]).iloc[0]
    full_snr = float(full["stacked_snr"])
    for month in months:
        sub = contrib[~contrib["month_block"].eq(month)]
        if sub.empty:
            continue
        row = _aggregate(sub, ["source_name", "frequency_band", "antenna", "window_s", "timing_offset_s"]).iloc[0].to_dict()
        row["left_out_month"] = month
        row["full_stacked_snr"] = full_snr
        row["snr_fraction_of_full"] = float(row["stacked_snr"] / full_snr) if full_snr else np.nan
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def _build_report(out_root: Path, tables: dict[str, pd.DataFrame]) -> None:
    timing = tables["timing_scan"]
    best = timing.assign(abs_snr=timing["stacked_snr"].abs()).sort_values("abs_snr", ascending=False).head(1)
    window = tables["window_sweep"]
    event_type = tables["event_type"]
    strict = tables["strict_quality"]
    controls = tables["wrong_controls"]
    loo = tables["leave_one_month"]
    empirical = tables.get("high_control_empirical_p", pd.DataFrame())
    lines = [
        "# Solar Detection Confirmation Suite",
        "",
        "Target: Sun band 6 / 3.93 MHz / `rv2_coarse`.",
        "",
        "## Best Timing Offset",
        "",
        best.to_string(index=False),
        "",
        "## Window Sweep",
        "",
        window.to_string(index=False),
        "",
        "## Event-Type Split At Best Offset",
        "",
        event_type.to_string(index=False),
        "",
        "## Strict-Quality Survival",
        "",
        strict.to_string(index=False),
        "",
        "## Existing High-Control Empirical P-Values",
        "",
        empirical.to_string(index=False) if not empirical.empty else "No empirical p-value table found.",
        "",
        "## Wrong Antenna / Neighboring Band Controls",
        "",
        controls.to_string(index=False),
        "",
        "## Leave-One-Month-Out Summary",
        "",
        loo.describe(include="all").to_string() if not loo.empty else "No leave-one-month rows.",
        "",
        "## Interpretation",
        "",
        "A confident solar detection requires a strong centered timing-offset peak, survival under strict quality cuts, "
        "same-sign disappearance/reappearance behavior, stability when any month is removed, and weaker wrong-antenna/band controls.",
    ]
    (out_root / "solar_detection_confirmation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=os.environ.get("RAE2_MASTER_CSV", "data/interpolatedRAE2MasterFile.csv"))
    parser.add_argument("--events", default="outputs/sun_whole_dataset_validation_highcontrols_bands4_8/02_events/sun_predicted_events.csv")
    parser.add_argument("--scored", default="outputs/sun_whole_dataset_validation_highcontrols_bands4_8/summary/sun_whole_dataset_scored_stacks.csv")
    parser.add_argument("--output-root", default="outputs/solar_detection_confirmation_suite")
    parser.add_argument("--target-band", type=int, default=6)
    parser.add_argument("--target-antenna", default="rv2_coarse")
    parser.add_argument("--target-window", type=float, default=900.0)
    parser.add_argument("--windows", nargs="+", type=float, default=[450.0, 600.0, 750.0, 900.0, 1050.0])
    parser.add_argument("--timing-offsets", nargs="+", type=float, default=[-300, -240, -180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180, 240, 300])
    parser.add_argument("--gap-factor", type=float, default=3.0)
    parser.add_argument("--artifact-sigma", type=float, default=12.0)
    args = parser.parse_args()

    out_root = ensure_dir(ROOT / args.output_root)
    clean = _load_clean(args)
    groups = _make_groups(clean)
    events = _read(ROOT / args.events, parse_dates=["predicted_event_time"])
    events = events[events["source_name"].astype(str).eq("sun")].reset_index(drop=True)

    target_events = events[
        events["frequency_band"].astype(int).eq(args.target_band)
        & events["antenna"].astype(str).eq(args.target_antenna)
    ].copy()
    quality = event_window_quality(clean, target_events, args.target_window)
    quality.to_csv(out_root / "target_window_quality.csv", index=False)

    _log("running timing-offset scan")
    timing_rows = []
    timing_contribs: dict[float, pd.DataFrame] = {}
    for offset in args.timing_offsets:
        contrib = _event_contributions(groups, target_events, args.target_window, offset)
        timing_contribs[float(offset)] = contrib
        timing_rows.append(_aggregate(contrib, ["source_name", "frequency_band", "frequency_mhz", "antenna", "window_s", "timing_offset_s"]))
    timing_scan = pd.concat(timing_rows, ignore_index=True)
    best_offset = float(timing_scan.assign(abs_snr=timing_scan["stacked_snr"].abs()).sort_values("abs_snr", ascending=False).iloc[0]["timing_offset_s"])
    best_contrib = timing_contribs[best_offset]

    _log(f"best timing offset: {best_offset}s")
    window_rows = []
    for window in args.windows:
        contrib = _event_contributions(groups, target_events, window, best_offset)
        window_rows.append(_aggregate(contrib, ["source_name", "frequency_band", "frequency_mhz", "antenna", "window_s", "timing_offset_s"]))
    window_sweep = pd.concat(window_rows, ignore_index=True)

    event_type = _aggregate(best_contrib, ["source_name", "frequency_band", "frequency_mhz", "antenna", "window_s", "timing_offset_s", "event_type"])
    month = _aggregate(best_contrib, ["source_name", "frequency_band", "frequency_mhz", "antenna", "window_s", "timing_offset_s", "month_block"])
    leave_one = _leave_one_month(best_contrib)

    _log("running strict-quality survival")
    strict_keys = _quality_keys(quality, args.target_band, args.target_antenna)
    strict_contrib = _event_contributions(groups, target_events, args.target_window, best_offset, strict_good_keys=strict_keys)
    strict_quality = _aggregate(strict_contrib, ["source_name", "frequency_band", "frequency_mhz", "antenna", "window_s", "timing_offset_s"])

    _log("running wrong-antenna and neighboring-band controls")
    control_specs = [
        (args.target_band, "rv1_coarse", "wrong_antenna"),
        (5, args.target_antenna, "neighboring_band"),
        (7, args.target_antenna, "neighboring_band"),
        (8, args.target_antenna, "neighboring_band"),
    ]
    controls = []
    for freq, ant, ctype in control_specs:
        sub = events[events["frequency_band"].astype(int).eq(int(freq)) & events["antenna"].astype(str).eq(str(ant))]
        contrib = _event_contributions(groups, sub, args.target_window, best_offset)
        agg = _aggregate(contrib, ["source_name", "frequency_band", "frequency_mhz", "antenna", "window_s", "timing_offset_s"])
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
    scored_path = ROOT / args.scored
    if scored_path.exists():
        scored = _read(scored_path)
        scored = scored[
            scored["frequency_band"].astype(int).eq(args.target_band)
            & scored["antenna"].astype(str).eq(args.target_antenna)
        ]
        if not scored.empty:
            tables["high_control_empirical_p"] = scored[
                [
                    "source_name",
                    "frequency_band",
                    "frequency_mhz",
                    "antenna",
                    "window_s",
                    "stacked_snr",
                    "randomized_stack_p",
                    "offephemeris_stack_p",
                    "clean_fraction",
                    "status",
                ]
            ]
    for name, table in tables.items():
        table.to_csv(out_root / f"{name}.csv", index=False)
    _build_report(out_root, tables)
    write_json(out_root / "run_config.json", vars(args) | {"best_timing_offset_s": best_offset})
    print(out_root / "solar_detection_confirmation_report.md")
    print(out_root / "timing_scan.csv")


if __name__ == "__main__":
    main()
