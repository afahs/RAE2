#!/usr/bin/env python
"""Targeted confirmation suite for a candidate source/channel."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rylevonberg.table_io import read_table

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rylevonberg.quality import event_window_quality
from rylevonberg.stacking import detrend_profile_values
from rylevonberg.util import datetime_ns, ensure_dir, robust_sigma, write_json


ROOT = Path(__file__).resolve().parents[1]


def _log(message: str) -> None:
    print(f"[source-confirmation] {message}", flush=True)


def _read(path: Path, **kwargs) -> pd.DataFrame:
    return read_table(path, low_memory=False, **kwargs)


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
    baseline_mode: str = "sideband_linear",
    sideband_exclusion_seconds: float = 120.0,
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
        yd, tmpl, sigma, n_baseline = detrend_profile_values(
            tr,
            yy,
            str(ev["event_type"]),
            baseline_mode=baseline_mode,
            sideband_exclusion_seconds=sideband_exclusion_seconds,
            timing_offset_sec=float(timing_offset_s),
            normalize=True,
        )
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
                "baseline_mode": str(baseline_mode),
                "n_baseline_samples": int(n_baseline),
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
        event_amp_sigma = robust_sigma(event_amp) if event_amp.size else np.nan
        event_amp_median = float(np.nanmedian(event_amp)) if event_amp.size else np.nan
        robust_se = float(event_amp_sigma / np.sqrt(event_amp.size)) if event_amp.size and np.isfinite(event_amp_sigma) and event_amp_sigma > 0 else np.nan
        robust_stack_snr = float(event_amp_median / robust_se) if np.isfinite(event_amp_median) and np.isfinite(robust_se) and robust_se > 0 else np.nan
        rows.append(
            {
                **meta,
                "n_events": int(group["event_id"].nunique()),
                "n_month_blocks": int(group["month_block"].nunique()),
                "stacked_amplitude": amp,
                "stacked_snr": float(amp * np.sqrt(den)) if den > 0 else np.nan,
                "robust_stack_snr": robust_stack_snr,
                "event_amp_median": event_amp_median,
                "event_amp_robust_sigma": float(event_amp_sigma) if np.isfinite(event_amp_sigma) else np.nan,
                "event_sign_fraction": float((np.sign(event_amp) == np.sign(amp)).mean()) if np.isfinite(amp) and event_amp.size else np.nan,
            }
        )
    return pd.DataFrame.from_records(rows)


def _quality_keys(quality: pd.DataFrame, freq: int, antenna: str, min_side: int) -> set[tuple[str, int, str]]:
    q = quality[
        quality["frequency_band"].astype(int).eq(int(freq))
        & quality["antenna"].astype(str).eq(str(antenna))
        & quality["primary_quality_failure"].fillna("").astype(str).eq("")
        & (pd.to_numeric(quality["valid_samples_pre"], errors="coerce").fillna(0) >= int(min_side))
        & (pd.to_numeric(quality["valid_samples_post"], errors="coerce").fillna(0) >= int(min_side))
    ].copy()
    return set((str(pd.Timestamp(r["predicted_event_time"])), int(r["frequency_band"]), str(r["antenna"])) for _, r in q.iterrows())


def _leave_one_month(contrib: pd.DataFrame) -> pd.DataFrame:
    if contrib.empty:
        return pd.DataFrame()
    rows = []
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


def _build_report(out_root: Path, args: argparse.Namespace, tables: dict[str, pd.DataFrame]) -> None:
    timing = tables["timing_scan"]
    best = timing.assign(abs_snr=timing["stacked_snr"].abs()).sort_values("abs_snr", ascending=False).head(1)
    lines = [
        f"# {args.source_name} Detection Confirmation Suite",
        "",
        f"Target: {args.source_name} band {args.target_band} / `{args.target_antenna}`.",
        "",
        "## Best Timing Offset",
        "",
        best.to_string(index=False),
        "",
        "## Window Sweep",
        "",
        tables["window_sweep"].to_string(index=False),
        "",
        "## Event-Type Split At Best Offset",
        "",
        tables["event_type"].to_string(index=False),
        "",
        "## Strict-Quality Survival",
        "",
        tables["strict_quality"].to_string(index=False) if not tables["strict_quality"].empty else "No events survived strict-quality cuts.",
        "",
        "## Wrong Antenna / Neighboring Band Controls",
        "",
        tables["wrong_controls"].to_string(index=False) if not tables["wrong_controls"].empty else "No wrong-control rows.",
        "",
        "## Leave-One-Month-Out Summary",
        "",
        tables["leave_one_month"].describe(include="all").to_string() if not tables["leave_one_month"].empty else "No leave-one-month rows.",
        "",
        "## Interpretation",
        "",
        "A robust detection should peak near the predicted time, survive quality cuts, show plausible event-type behavior, "
        "remain stable when date blocks are removed, and be stronger than wrong antenna or neighboring-band controls.",
    ]
    (out_root / f"{args.source_name}_detection_confirmation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_control_specs(items: list[str]) -> list[tuple[int, str, str]]:
    specs = []
    for item in items:
        parts = item.split(":")
        if len(parts) != 3:
            raise ValueError(f"control spec must be band:antenna:type, got {item!r}")
        specs.append((int(parts[0]), parts[1], parts[2]))
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--clean", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--target-band", type=int, required=True)
    parser.add_argument("--target-antenna", required=True)
    parser.add_argument("--target-window", type=float, required=True)
    parser.add_argument("--windows", nargs="+", type=float, default=[300.0, 600.0, 900.0])
    parser.add_argument("--timing-offsets", nargs="+", type=float, default=[-600, -510, -420, -330, -240, -180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180, 240, 330, 420, 510, 600])
    parser.add_argument("--control", nargs="*", default=[])
    parser.add_argument("--strict-min-side-samples", type=int, default=20)
    parser.add_argument("--baseline-mode", default="sideband_linear", choices=["linear_all", "constant_all", "sideband_linear", "sideband_constant", "joint_step_linear", "pre_event_anchor"])
    parser.add_argument("--sideband-exclusion-seconds", type=float, default=120.0)
    args = parser.parse_args()

    out_root = ensure_dir(ROOT / args.output_root)
    _log(f"reading clean table: {args.clean}")
    clean = _read(ROOT / args.clean, parse_dates=["time"])
    groups = _make_groups(clean)
    events = _read(ROOT / args.events, parse_dates=["predicted_event_time"])
    events = events[events["source_name"].astype(str).eq(args.source_name)].reset_index(drop=True)
    target_events = events[
        events["frequency_band"].astype(int).eq(args.target_band)
        & events["antenna"].astype(str).eq(args.target_antenna)
    ].copy()

    _log("computing target-window quality")
    quality = event_window_quality(clean, target_events, args.target_window)
    quality.to_csv(out_root / "target_window_quality.csv", index=False)

    _log("running timing-offset scan")
    timing_rows = []
    timing_contribs: dict[float, pd.DataFrame] = {}
    for offset in args.timing_offsets:
        contrib = _event_contributions(
            groups,
            target_events,
            args.target_window,
            offset,
            baseline_mode=args.baseline_mode,
            sideband_exclusion_seconds=args.sideband_exclusion_seconds,
        )
        timing_contribs[float(offset)] = contrib
        timing_rows.append(_aggregate(contrib, ["source_name", "frequency_band", "frequency_mhz", "antenna", "window_s", "timing_offset_s"]))
    timing_scan = pd.concat(timing_rows, ignore_index=True)
    best_offset = float(timing_scan.assign(abs_snr=timing_scan["stacked_snr"].abs()).sort_values("abs_snr", ascending=False).iloc[0]["timing_offset_s"])
    best_contrib = timing_contribs[best_offset]
    _log(f"best timing offset: {best_offset}s")

    window_rows = []
    for window in args.windows:
        contrib = _event_contributions(
            groups,
            target_events,
            window,
            best_offset,
            baseline_mode=args.baseline_mode,
            sideband_exclusion_seconds=args.sideband_exclusion_seconds,
        )
        window_rows.append(_aggregate(contrib, ["source_name", "frequency_band", "frequency_mhz", "antenna", "window_s", "timing_offset_s"]))
    window_sweep = pd.concat(window_rows, ignore_index=True)

    event_type = _aggregate(best_contrib, ["source_name", "frequency_band", "frequency_mhz", "antenna", "window_s", "timing_offset_s", "event_type"])
    month = _aggregate(best_contrib, ["source_name", "frequency_band", "frequency_mhz", "antenna", "window_s", "timing_offset_s", "month_block"])
    leave_one = _leave_one_month(best_contrib)

    _log("running strict-quality survival")
    strict_keys = _quality_keys(quality, args.target_band, args.target_antenna, args.strict_min_side_samples)
    strict_contrib = _event_contributions(
        groups,
        target_events,
        args.target_window,
        best_offset,
        strict_good_keys=strict_keys,
        baseline_mode=args.baseline_mode,
        sideband_exclusion_seconds=args.sideband_exclusion_seconds,
    )
    strict_quality = _aggregate(strict_contrib, ["source_name", "frequency_band", "frequency_mhz", "antenna", "window_s", "timing_offset_s"])

    controls = []
    for freq, ant, ctype in _parse_control_specs(args.control):
        sub = events[events["frequency_band"].astype(int).eq(int(freq)) & events["antenna"].astype(str).eq(str(ant))]
        contrib = _event_contributions(
            groups,
            sub,
            args.target_window,
            best_offset,
            baseline_mode=args.baseline_mode,
            sideband_exclusion_seconds=args.sideband_exclusion_seconds,
        )
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
    for name, table in tables.items():
        table.to_csv(out_root / f"{name}.csv", index=False)
    _build_report(out_root, args, tables)
    write_json(out_root / "run_config.json", vars(args) | {"best_timing_offset_s": best_offset})
    print(out_root / f"{args.source_name}_detection_confirmation_report.md")
    print(out_root / "timing_scan.csv")


if __name__ == "__main__":
    main()
