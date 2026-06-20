#!/usr/bin/env python
"""Build occultation profile grids after subtracting nearby time-shift controls.

This is a deliberately simple background-removal test.  For each real event
window we extract additional windows from the same antenna/frequency channel at
nearby time offsets.  Those shifted windows should contain the same broad
receiver/background/spacecraft-scan behavior but no correctly centered lunar
occultation of the requested source.  The corrected event profile is:

    corrected(t) = true_event(t) - median(shifted_control_profiles(t))

The corrected profiles are then stacked exactly like the ordinary
all-frequency profile grids.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.util import datetime_ns, ensure_dir, robust_sigma, software_versions, write_json  # noqa: E402

from scripts.build_all_frequency_occultation_profile_grids import (  # noqa: E402
    ANT_COLOR,
    ANT_LABEL,
    CLEAN,
    _events_for_source,
    _read,
    collect_profiles,
)


EXPECTED_SIGN = {"disappearance": -1.0, "reappearance": 1.0}


def _robust_sem(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size <= 1:
        return np.nan
    scale = robust_sigma(vals - np.nanmedian(vals))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.nanstd(vals, ddof=1))
    return float(scale / np.sqrt(vals.size)) if np.isfinite(scale) and scale > 0 else np.nan


def _source_title(source: str) -> str:
    return {"earth": "Earth", "sun": "Sun"}.get(source.lower(), source.replace("_", " ").title())


def _collect_source_with_shifts(
    clean: pd.DataFrame,
    source: str,
    window_s: float,
    bin_s: float,
    inner_s: float,
    control_shifts_s: list[float],
    exclude_sources: list[str] | None = None,
    exclude_margin_s: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    events = _events_for_source(source)
    true_points = collect_profiles(clean, events, source, window_s, bin_s, inner_s, time_shift_s=0.0)
    exclusion_index = _build_control_exclusion_index(exclude_sources or [])
    controls = []
    stats = []
    for shift_s in control_shifts_s:
        filtered_events, shift_stats = _filter_shift_control_events(
            events=events,
            shift_s=shift_s,
            window_s=window_s,
            exclusion_index=exclusion_index,
            exclude_margin_s=exclude_margin_s,
        )
        shift_stats["source_name"] = source.lower()
        stats.append(shift_stats)
        part = collect_profiles(clean, filtered_events, source, window_s, bin_s, inner_s, time_shift_s=shift_s)
        if not part.empty:
            controls.append(part)
    control_points = pd.concat(controls, ignore_index=True) if controls else pd.DataFrame()
    status = pd.DataFrame(stats)
    return true_points, control_points, status


def _build_control_exclusion_index(sources: list[str]) -> dict[tuple[int, str], np.ndarray]:
    frames = []
    for source in sources:
        events = _events_for_source(source)
        if not events.empty:
            frames.append(events)
    if not frames:
        return {}
    all_events = pd.concat(frames, ignore_index=True)
    index: dict[tuple[int, str], np.ndarray] = {}
    for (band, antenna), grp in all_events.groupby(["frequency_band", "antenna"], sort=True):
        times = pd.to_datetime(grp["predicted_event_time"], errors="coerce").dropna()
        if times.empty:
            continue
        index[(int(band), str(antenna))] = np.sort(datetime_ns(times))
    return index


def _filter_shift_control_events(
    events: pd.DataFrame,
    shift_s: float,
    window_s: float,
    exclusion_index: dict[tuple[int, str], np.ndarray],
    exclude_margin_s: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    if events.empty or not exclusion_index:
        return events.copy(), {
            "time_shift_s": float(shift_s),
            "n_events_before_overlap_filter": int(len(events)),
            "n_events_after_overlap_filter": int(len(events)),
            "n_events_removed_by_overlap_filter": 0,
        }

    radius_ns = int((float(window_s) + float(exclude_margin_s)) * 1e9)
    shifted_ns = datetime_ns(pd.to_datetime(events["predicted_event_time"]) + pd.to_timedelta(float(shift_s), unit="s"))
    keep = np.ones(len(events), dtype=bool)
    for i, (_, ev) in enumerate(events.iterrows()):
        arr = exclusion_index.get((int(ev["frequency_band"]), str(ev["antenna"])))
        if arr is None or arr.size == 0:
            continue
        pos = int(np.searchsorted(arr, shifted_ns[i]))
        contaminated = False
        if pos < arr.size and abs(int(arr[pos]) - int(shifted_ns[i])) <= radius_ns:
            contaminated = True
        if pos > 0 and abs(int(arr[pos - 1]) - int(shifted_ns[i])) <= radius_ns:
            contaminated = True
        keep[i] = not contaminated

    return events.loc[keep].copy(), {
        "time_shift_s": float(shift_s),
        "n_events_before_overlap_filter": int(len(events)),
        "n_events_after_overlap_filter": int(np.count_nonzero(keep)),
        "n_events_removed_by_overlap_filter": int(len(events) - np.count_nonzero(keep)),
    }


def subtract_shift_controls(true_points: pd.DataFrame, control_points: pd.DataFrame) -> pd.DataFrame:
    if true_points.empty or control_points.empty:
        return pd.DataFrame()
    keys = ["source_name", "event_id", "event_type", "frequency_band", "frequency_mhz", "antenna", "t_bin_sec"]
    ctrl = (
        control_points.groupby(keys, sort=True, dropna=False)
        .agg(
            shift_control_median_z_power=("z_power", "median"),
            n_shift_control_points=("z_power", "size"),
            n_shift_controls=("time_shift_s", "nunique"),
        )
        .reset_index()
    )
    merged = true_points.merge(ctrl, on=keys, how="inner")
    merged["shift_control_corrected_z_power"] = merged["z_power"] - merged["shift_control_median_z_power"]
    return merged


def summarize(points: pd.DataFrame, value_col: str, method: str) -> pd.DataFrame:
    rows = []
    keys = ["source_name", "event_type", "frequency_band", "frequency_mhz", "antenna", "t_bin_sec"]
    for vals_key, grp in points.groupby(keys, sort=True, dropna=False):
        raw = pd.to_numeric(grp[value_col], errors="coerce")
        good = raw.notna() & np.isfinite(raw)
        vals = raw.loc[good]
        if vals.empty:
            continue
        rows.append(
            {
                **dict(zip(keys, vals_key)),
                "method": method,
                "median_z_power": float(vals.median()),
                "median_z_power_err": _robust_sem(vals),
                "n_events": int(grp.loc[good, "event_id"].nunique()),
                "n_points": int(len(vals)),
                "median_n_shift_controls": float(np.nanmedian(grp.loc[good, "n_shift_controls"]))
                if "n_shift_controls" in grp
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def prepost_contrast(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pre = (-180.0, -60.0)
    post = (60.0, 180.0)
    keys = ["method", "source_name", "frequency_band", "frequency_mhz", "antenna", "event_type"]
    for vals_key, grp in summary.groupby(keys, sort=True, dropna=False):
        before = grp[(grp["t_bin_sec"] >= pre[0]) & (grp["t_bin_sec"] <= pre[1])]["median_z_power"]
        after = grp[(grp["t_bin_sec"] >= post[0]) & (grp["t_bin_sec"] <= post[1])]["median_z_power"]
        if before.empty or after.empty:
            continue
        event_type = str(vals_key[-1])
        delta = float(np.nanmedian(after) - np.nanmedian(before))
        rows.append(
            {
                **dict(zip(keys, vals_key)),
                "n_events": int(np.nanmedian(grp["n_events"])) if "n_events" in grp else np.nan,
                "n_points": int(np.nanmedian(grp["n_points"])) if "n_points" in grp else np.nan,
                "post_minus_pre": delta,
                "source_like_contrast": float(EXPECTED_SIGN[event_type] * delta),
            }
        )
    return pd.DataFrame(rows)


def contrast_comparison(uncorrected: pd.DataFrame, corrected: pd.DataFrame) -> pd.DataFrame:
    keys = ["source_name", "frequency_band", "frequency_mhz", "antenna", "event_type"]
    left = uncorrected.rename(
        columns={
            "n_events": "n_events_uncorrected",
            "n_points": "n_points_uncorrected",
            "post_minus_pre": "post_minus_pre_uncorrected",
            "source_like_contrast": "source_like_contrast_uncorrected",
        }
    )
    right = corrected.rename(
        columns={
            "n_events": "n_events_corrected",
            "n_points": "n_points_corrected",
            "post_minus_pre": "post_minus_pre_corrected",
            "source_like_contrast": "source_like_contrast_corrected",
        }
    )
    keep_left = keys + [
        "n_events_uncorrected",
        "n_points_uncorrected",
        "post_minus_pre_uncorrected",
        "source_like_contrast_uncorrected",
    ]
    keep_right = keys + [
        "n_events_corrected",
        "n_points_corrected",
        "post_minus_pre_corrected",
        "source_like_contrast_corrected",
    ]
    merged = left[keep_left].merge(right[keep_right], on=keys, how="outer")
    merged["source_like_contrast_change"] = (
        merged["source_like_contrast_corrected"] - merged["source_like_contrast_uncorrected"]
    )
    return merged.sort_values(keys).reset_index(drop=True)


def plot_grid(summary: pd.DataFrame, source: str, out_dir: Path, window_s: float) -> Path:
    freqs = sorted(summary["frequency_mhz"].dropna().unique())
    fig, axes = plt.subplots(len(freqs), 2, figsize=(12, max(10, 1.35 * len(freqs))), sharex=True)
    if len(freqs) == 1:
        axes = np.asarray([axes])
    for i, freq in enumerate(freqs):
        for j, event_type in enumerate(["disappearance", "reappearance"]):
            ax = axes[i, j]
            sub = summary[np.isclose(summary["frequency_mhz"], freq) & summary["event_type"].eq(event_type)]
            for antenna, grp in sub.groupby("antenna", sort=True):
                grp = grp.sort_values("t_bin_sec")
                ax.errorbar(
                    grp["t_bin_sec"],
                    grp["median_z_power"],
                    yerr=grp["median_z_power_err"],
                    marker="o",
                    markersize=2.4,
                    linewidth=1.15,
                    elinewidth=0.55,
                    capsize=1.1,
                    alpha=0.9,
                    color=ANT_COLOR.get(str(antenna)),
                    ecolor=ANT_COLOR.get(str(antenna)),
                    label=ANT_LABEL.get(str(antenna), str(antenna)),
                )
            ax.axvline(0, color="black", linestyle="--", linewidth=0.75)
            ax.axhline(0, color="0.65", linewidth=0.7)
            ax.set_title(f"{freq:.2f} MHz {event_type}", fontsize=9)
            if j == 0:
                ax.set_ylabel("true - shifted controls")
            if i == len(freqs) - 1:
                ax.set_xlabel("seconds from predicted event")
            if i == 0 and j == 1:
                ax.legend(frameon=False, fontsize=8)
    fig.suptitle(
        f"{_source_title(source)}: time-shift-control-subtracted profiles\n"
        f"Each event window minus median nearby shifted controls, +/-{window_s:.0f} s",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = out_dir / f"{source}_shift_control_subtracted_all_frequency_profile_grid.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report(
    out_dir: Path,
    sources: list[str],
    shifts: list[float],
    contrast_table: pd.DataFrame,
    comparison_table: pd.DataFrame,
    overlap_status: pd.DataFrame,
    paths: list[Path],
) -> None:
    lines = [
        "# Time-Shift Control Background Subtraction",
        "",
        "## Method",
        "",
        "For every real occultation event, nearby windows from the same source event list, frequency, antenna, and data channel",
        "are extracted at fixed time shifts. The median shifted-control profile is subtracted from the true event profile",
        "before stacking. This tests whether the low-frequency morphology is dominated by local scan/background structure",
        "that is not locked to the predicted occultation time.",
        "",
        f"Sources: {', '.join(sources)}",
        f"Control shifts, seconds: {', '.join(f'{x:.0f}' for x in shifts)}",
        "",
        "Shift controls whose centers would put another Earth/Sun predicted event inside the extraction window are removed.",
        "",
        "## Low-Frequency Lower-V Source-Like Contrasts",
        "",
    ]
    low = comparison_table[
        comparison_table["frequency_mhz"].isin([0.45, 0.70, 0.90, 1.31, 2.20])
        & comparison_table["antenna"].eq("rv2_coarse")
    ].copy()
    if low.empty:
        lines.append("No low-frequency lower-V rows.")
    else:
        lines.append(
            low[
                [
                    "source_name",
                    "frequency_mhz",
                    "event_type",
                    "n_events_uncorrected",
                    "n_events_corrected",
                    "source_like_contrast_uncorrected",
                    "source_like_contrast_corrected",
                    "source_like_contrast_change",
                ]
            ].to_string(index=False)
        )
    lines.extend(["", "## Control-Window Overlap Filter", ""])
    if overlap_status.empty:
        lines.append("No overlap-filter status rows.")
    else:
        lines.append(overlap_status.to_string(index=False))
    lines.extend(["", "## Plots", ""])
    lines.extend(f"- `{path}`" for path in paths)
    (out_dir / "shift_control_background_subtraction_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", default="earth,sun")
    parser.add_argument("--out-dir", default=str(ROOT / "outputs/shift_control_background_profiles_v1"))
    parser.add_argument("--window-s", type=float, default=900.0)
    parser.add_argument("--bin-s", type=float, default=60.0)
    parser.add_argument("--inner-s", type=float, default=15.0)
    parser.add_argument("--control-shifts-s", default="-1800,-1200,1200,1800")
    parser.add_argument(
        "--exclude-control-near-sources",
        default="earth,sun",
        help="Comma-separated sources whose real occultation windows are not allowed to overlap shifted controls.",
    )
    parser.add_argument("--exclude-control-margin-s", type=float, default=0.0)
    args = parser.parse_args()

    sources = [x.strip().lower() for x in str(args.sources).split(",") if x.strip()]
    shifts = [float(x.strip()) for x in str(args.control_shifts_s).split(",") if x.strip()]
    exclude_sources = [x.strip().lower() for x in str(args.exclude_control_near_sources).split(",") if x.strip()]
    out_dir = ensure_dir(args.out_dir)
    write_json(
        out_dir / "run_config.json",
        {
            "sources": sources,
            "window_s": float(args.window_s),
            "bin_s": float(args.bin_s),
            "inner_s": float(args.inner_s),
            "control_shifts_s": shifts,
            "exclude_control_near_sources": exclude_sources,
            "exclude_control_margin_s": float(args.exclude_control_margin_s),
            "software_versions": software_versions(),
        },
    )

    clean = _read(CLEAN, parse_dates=["time"])
    all_summaries = []
    all_contrasts = []
    all_uncorrected_contrasts = []
    all_comparisons = []
    all_overlap_status = []
    plot_paths = []
    for source in sources:
        true_points, control_points, overlap_status = _collect_source_with_shifts(
            clean,
            source,
            args.window_s,
            args.bin_s,
            args.inner_s,
            shifts,
            exclude_sources,
            args.exclude_control_margin_s,
        )
        corrected = subtract_shift_controls(true_points, control_points)
        true_points.to_csv(out_dir / f"{source}_true_profile_points.csv", index=False)
        control_points.to_csv(out_dir / f"{source}_shift_control_profile_points.csv", index=False)
        corrected.to_csv(out_dir / f"{source}_shift_control_corrected_points.csv", index=False)
        uncorrected_summary = summarize(true_points, "z_power", "uncorrected")
        uncorrected_contrast = prepost_contrast(uncorrected_summary)
        summary = summarize(corrected, "shift_control_corrected_z_power", "shift_control_subtracted")
        contrast = prepost_contrast(summary)
        comparison = contrast_comparison(uncorrected_contrast, contrast)
        uncorrected_summary.to_csv(out_dir / f"{source}_uncorrected_summary.csv", index=False)
        uncorrected_contrast.to_csv(out_dir / f"{source}_uncorrected_prepost_contrast.csv", index=False)
        summary.to_csv(out_dir / f"{source}_shift_control_corrected_summary.csv", index=False)
        contrast.to_csv(out_dir / f"{source}_shift_control_corrected_prepost_contrast.csv", index=False)
        comparison.to_csv(out_dir / f"{source}_shift_control_contrast_comparison.csv", index=False)
        overlap_status.to_csv(out_dir / f"{source}_shift_control_overlap_filter_status.csv", index=False)
        if not summary.empty:
            plot_paths.append(plot_grid(summary, source, out_dir, args.window_s))
        all_summaries.append(summary)
        all_contrasts.append(contrast)
        all_uncorrected_contrasts.append(uncorrected_contrast)
        all_comparisons.append(comparison)
        all_overlap_status.append(overlap_status)

    combined_summary = pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame()
    combined_contrast = pd.concat(all_contrasts, ignore_index=True) if all_contrasts else pd.DataFrame()
    combined_uncorrected_contrast = (
        pd.concat(all_uncorrected_contrasts, ignore_index=True) if all_uncorrected_contrasts else pd.DataFrame()
    )
    combined_comparison = pd.concat(all_comparisons, ignore_index=True) if all_comparisons else pd.DataFrame()
    combined_overlap_status = pd.concat(all_overlap_status, ignore_index=True) if all_overlap_status else pd.DataFrame()
    combined_summary.to_csv(out_dir / "shift_control_corrected_summary_all_sources.csv", index=False)
    combined_contrast.to_csv(out_dir / "shift_control_corrected_prepost_contrast_all_sources.csv", index=False)
    combined_uncorrected_contrast.to_csv(out_dir / "uncorrected_prepost_contrast_all_sources.csv", index=False)
    combined_comparison.to_csv(out_dir / "shift_control_contrast_comparison_all_sources.csv", index=False)
    combined_overlap_status.to_csv(out_dir / "shift_control_overlap_filter_status_all_sources.csv", index=False)
    write_report(out_dir, sources, shifts, combined_contrast, combined_comparison, combined_overlap_status, plot_paths)
    print(out_dir / "shift_control_background_subtraction_report.md")


if __name__ == "__main__":
    main()
