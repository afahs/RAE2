#!/usr/bin/env python
"""Audit repeated/stuck lower-V raw values across time and frequency bands."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rylevonberg.table_io import read_table

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.constants import FREQUENCY_MAP_MHZ  # noqa: E402
from rylevonberg.sample_quality import strict_power_mask  # noqa: E402
from rylevonberg.util import ensure_dir, software_versions, write_json  # noqa: E402


CLEAN = ROOT / "outputs/control_survey_earth_sun_postnov1974_v2/01_ingest/cleaned_timeseries.csv"


def _is_valid_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def _load_lower_v(clean_csv: Path, chunk_size: int, require_valid: bool) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    usecols = ["time", "frequency_band", "antenna", "power", "is_valid"]
    for chunk in read_table(clean_csv, usecols=usecols, chunksize=chunk_size, low_memory=False):
        chunk = chunk[chunk["antenna"].astype(str).eq("rv2_coarse")].copy()
        if chunk.empty:
            continue
        chunk["power"] = pd.to_numeric(chunk["power"], errors="coerce")
        chunk["frequency_band"] = pd.to_numeric(chunk["frequency_band"], errors="coerce")
        valid = _is_valid_series(chunk["is_valid"])
        keep = np.isfinite(chunk["power"].to_numpy(dtype=float)) & chunk["frequency_band"].notna().to_numpy()
        if require_valid:
            keep = keep & valid.to_numpy(dtype=bool)
        sub = chunk.loc[keep, ["time", "frequency_band", "power"]].copy()
        sub["frequency_band"] = sub["frequency_band"].astype(np.int16)
        pieces.append(sub)
    if not pieces:
        return pd.DataFrame(columns=["time", "frequency_band", "power", "power_int", "frequency_mhz"])
    out = pd.concat(pieces, ignore_index=True)
    out["time"] = pd.to_datetime(out["time"], errors="coerce")
    out = out[out["time"].notna()].copy()
    if require_valid and not out.empty:
        keep = np.zeros(len(out), dtype=bool)
        for _, idx in out.groupby("frequency_band", sort=False).groups.items():
            pos = out.index.get_indexer(idx)
            keep[pos] = strict_power_mask(out.loc[idx], upper_clip_quantile=0.9999)
        out = out.loc[keep].copy()
    out["power_int"] = np.rint(out["power"].to_numpy(dtype=float)).astype(np.int64)
    out["frequency_mhz"] = out["frequency_band"].map(FREQUENCY_MAP_MHZ).astype(float)
    return out.sort_values("time").reset_index(drop=True)


def _run_lengths(values: np.ndarray) -> list[tuple[int, int, int]]:
    if values.size == 0:
        return []
    runs: list[tuple[int, int, int]] = []
    start = 0
    current = int(values[0])
    for i in range(1, values.size):
        value = int(values[i])
        if value != current:
            runs.append((current, start, i - start))
            current = value
            start = i
    runs.append((current, start, values.size - start))
    return runs


def _temporal_stuck_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    run_rows = []
    for band, grp in df.groupby("frequency_band", sort=True):
        g = grp.sort_values("time")
        values = g["power_int"].to_numpy(dtype=np.int64)
        same_prev = values[1:] == values[:-1]
        runs = _run_lengths(values)
        long_runs = [r for r in runs if r[2] >= 3]
        very_long_runs = [r for r in runs if r[2] >= 5]
        top_values = Counter()
        for value, _, length in long_runs:
            top_values[value] += length
        rows.append(
            {
                "frequency_band": int(band),
                "frequency_mhz": FREQUENCY_MAP_MHZ.get(int(band), np.nan),
                "n_samples": int(len(g)),
                "same_as_previous_fraction": float(np.mean(same_prev)) if same_prev.size else np.nan,
                "n_runs_len_ge_3": int(len(long_runs)),
                "n_runs_len_ge_5": int(len(very_long_runs)),
                "max_run_length": int(max((r[2] for r in runs), default=0)),
                "fraction_samples_in_runs_len_ge_3": float(sum(r[2] for r in long_runs) / len(g)) if len(g) else np.nan,
                "top_stuck_values": "; ".join(f"{v}:{c}" for v, c in top_values.most_common(8)),
            }
        )
        time_values = g["time"].to_numpy()
        for value, start, length in sorted(long_runs, key=lambda r: r[2], reverse=True)[:100]:
            run_rows.append(
                {
                    "frequency_band": int(band),
                    "frequency_mhz": FREQUENCY_MAP_MHZ.get(int(band), np.nan),
                    "power_int": int(value),
                    "run_length": int(length),
                    "start_time": str(pd.Timestamp(time_values[start])),
                    "end_time": str(pd.Timestamp(time_values[start + length - 1])),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(run_rows)


def _sequence_stuck_summary(df: pd.DataFrame, max_gap_s: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Find exact repeated values in adjacent frequency bands in time-ordered telemetry."""
    g = df.sort_values("time").reset_index(drop=True)
    t = g["time"].astype("int64").to_numpy()
    bands = g["frequency_band"].to_numpy(dtype=np.int16)
    values = g["power_int"].to_numpy(dtype=np.int64)
    dt = np.diff(t) / 1e9
    adjacent = (np.abs(np.diff(bands)) == 1) & (dt >= 0) & (dt <= max_gap_s)
    exact_match = adjacent & (values[1:] == values[:-1])
    rounded_match = adjacent & (np.abs(values[1:] - values[:-1]) <= 1)
    rows = [
        {
            "comparison": "adjacent_band_pairs_in_time_order",
            "max_gap_s": float(max_gap_s),
            "n_pairs": int(np.count_nonzero(adjacent)),
            "n_exact_same_raw_integer": int(np.count_nonzero(exact_match)),
            "exact_same_fraction": float(np.count_nonzero(exact_match) / np.count_nonzero(adjacent))
            if np.count_nonzero(adjacent)
            else np.nan,
            "n_within_1_raw_unit": int(np.count_nonzero(rounded_match)),
            "within_1_raw_unit_fraction": float(np.count_nonzero(rounded_match) / np.count_nonzero(adjacent))
            if np.count_nonzero(adjacent)
            else np.nan,
        }
    ]

    run_rows = []
    start = 0
    for i in range(1, len(g)):
        continues = bool(adjacent[i - 1] and values[i] == values[i - 1])
        if continues:
            continue
        if i - start >= 2:
            band_seq = bands[start:i]
            run_rows.append(
                {
                    "start_time": str(pd.Timestamp(g.loc[int(start), "time"])),
                    "end_time": str(pd.Timestamp(g.loc[int(i - 1), "time"])),
                    "power_int": int(values[start]),
                    "n_samples": int(i - start),
                    "n_frequency_bands": int(len(set(int(x) for x in band_seq))),
                    "band_sequence": " ".join(str(int(x)) for x in band_seq),
                    "frequency_mhz_sequence": " ".join(f"{FREQUENCY_MAP_MHZ.get(int(x), np.nan):.2f}" for x in band_seq),
                }
            )
        start = i
    if len(g) - start >= 2:
        band_seq = bands[start : len(g)]
        run_rows.append(
            {
                "start_time": str(pd.Timestamp(g.loc[int(start), "time"])),
                "end_time": str(pd.Timestamp(g.loc[int(len(g) - 1), "time"])),
                "power_int": int(values[start]),
                "n_samples": int(len(g) - start),
                "n_frequency_bands": int(len(set(int(x) for x in band_seq))),
                "band_sequence": " ".join(str(int(x)) for x in band_seq),
                "frequency_mhz_sequence": " ".join(f"{FREQUENCY_MAP_MHZ.get(int(x), np.nan):.2f}" for x in band_seq),
            }
        )
    run_table = pd.DataFrame(run_rows)
    if not run_table.empty:
        run_table = run_table.sort_values(["n_frequency_bands", "n_samples"], ascending=False).head(500)
        rows.append(
            {
                "comparison": "same_value_adjacent_band_runs",
                "max_gap_s": float(max_gap_s),
                "n_pairs": np.nan,
                "n_exact_same_raw_integer": int(len(run_rows)),
                "exact_same_fraction": np.nan,
                "n_within_1_raw_unit": np.nan,
                "within_1_raw_unit_fraction": np.nan,
            }
        )
    return pd.DataFrame(rows), run_table


def _sweep_summary(df: pd.DataFrame, sweep_gap_s: float, max_samples: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Group short monotonic frequency sequences and test repeated values across bands."""
    g = df.sort_values("time").reset_index(drop=True)
    times = g["time"].to_numpy()
    t_ns = g["time"].astype("int64").to_numpy()
    bands = g["frequency_band"].to_numpy(dtype=np.int16)
    values = g["power_int"].to_numpy(dtype=np.int64)
    n_sweeps = 0
    n_same_2 = 0
    n_same_3 = 0
    max_mults = []
    repeat_rows = []
    start = 0
    sid = 0
    n = len(g)
    while start < n:
        end = start + 1
        while end < n and end - start < max_samples:
            dt = (t_ns[end] - t_ns[end - 1]) / 1e9
            if not np.isfinite(dt) or dt > sweep_gap_s or bands[end] <= bands[end - 1]:
                break
            end += 1
        if end - start < 2:
            start = max(end, start + 1)
            sid += 1
            continue
        band_seq = bands[start:end]
        value_seq = values[start:end]
        band_count = int(len(set(int(x) for x in band_seq)))
        if band_count < 2:
            start = end
            sid += 1
            continue
        by_value: dict[int, set[int]] = defaultdict(set)
        for band, value in zip(band_seq, value_seq):
            by_value[int(value)].add(int(band))
        repeated = [(value, sorted(bset)) for value, bset in by_value.items() if len(bset) >= 2]
        max_mult = max((len(bset) for bset in by_value.values()), default=0)
        n_sweeps += 1
        n_same_2 += int(max_mult >= 2)
        n_same_3 += int(max_mult >= 3)
        max_mults.append(max_mult)
        for value, band_list in sorted(repeated, key=lambda item: len(item[1]), reverse=True)[:5]:
            repeat_rows.append(
                {
                    "sweep_id": int(sid),
                    "start_time": str(pd.Timestamp(times[start])),
                    "power_int": int(value),
                    "n_frequency_bands": int(len(band_list)),
                    "bands": " ".join(str(x) for x in band_list),
                    "frequencies_mhz": " ".join(f"{FREQUENCY_MAP_MHZ.get(x, np.nan):.2f}" for x in band_list),
                }
            )
        start = end
        sid += 1
    repeats = pd.DataFrame(repeat_rows)
    summary_rows = []
    if n_sweeps:
        summary_rows.append(
            {
                "comparison": "near_time_sweeps",
                "sweep_gap_s": float(sweep_gap_s),
                "max_samples_per_sweep": int(max_samples),
                "n_sweeps_with_2plus_bands": int(n_sweeps),
                "fraction_sweeps_same_value_2plus_bands": float(n_same_2 / n_sweeps),
                "fraction_sweeps_same_value_3plus_bands": float(n_same_3 / n_sweeps),
                "median_max_same_value_band_count": float(np.median(max_mults)),
                "max_same_value_band_count": int(max(max_mults)),
            }
        )
    summary = pd.DataFrame(summary_rows)
    if not repeats.empty:
        repeats = repeats.sort_values(["n_frequency_bands", "start_time"], ascending=[False, True]).head(500)
    return summary, repeats


def _plot_histograms(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(3, 3, figsize=(16, 11), constrained_layout=True)
    axes = axes.ravel()
    for ax, band in zip(axes, sorted(FREQUENCY_MAP_MHZ)):
        sub = df[df["frequency_band"].eq(band)]
        values = sub["power"].to_numpy(dtype=float)
        if values.size:
            lo, hi = np.nanpercentile(values, [0.5, 99.5])
            central = values[(values >= lo) & (values <= hi)]
            ax.hist(central, bins=100, color="#d95f02", alpha=0.88, log=True)
            clipped = 1.0 - len(central) / len(values)
            ax.text(
                0.02,
                0.95,
                f"n={len(values):,}\ncentral 99% shown\nclipped={clipped:.1%}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
            )
        ax.set_title(f"{FREQUENCY_MAP_MHZ.get(band, np.nan):.2f} MHz")
        ax.set_xlabel("raw lower-V power")
        ax.set_ylabel("count, log scale")
        ax.grid(alpha=0.18)
    path = out_dir / "lower_v_raw_power_histograms_by_frequency.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _plot_same_prev(summary: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax1 = plt.subplots(figsize=(9, 5.2))
    x = summary["frequency_mhz"].to_numpy(dtype=float)
    ax1.plot(x, summary["same_as_previous_fraction"], marker="o", color="#d95f02", label="same as previous")
    ax1.plot(
        x,
        summary["fraction_samples_in_runs_len_ge_3"],
        marker="s",
        color="#7570b3",
        label="in runs length >= 3",
    )
    ax1.set_xscale("log")
    ax1.set_xticks(sorted(FREQUENCY_MAP_MHZ.values()))
    ax1.set_xticklabels([f"{v:.2f}" for v in sorted(FREQUENCY_MAP_MHZ.values())], rotation=45)
    ax1.set_xlabel("frequency (MHz)")
    ax1.set_ylabel("fraction of lower-V samples")
    ax1.grid(alpha=0.25, which="both")
    ax1.legend(frameon=False)
    fig.suptitle("Temporal repeated raw values in lower V")
    fig.tight_layout()
    path = out_dir / "lower_v_temporal_stuck_fraction.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _write_report(
    out_dir: Path,
    temporal: pd.DataFrame,
    temporal_runs: pd.DataFrame,
    sequence_summary: pd.DataFrame,
    sequence_runs: pd.DataFrame,
    sweep_summary: pd.DataFrame,
    sweep_repeats: pd.DataFrame,
    paths: list[Path],
    require_valid: bool,
) -> Path:
    lines = [
        "# Lower-V Stuck/Repeated Raw Value Audit",
        "",
        f"Input rows: lower V (`rv2_coarse`), {'strict valid-only' if require_valid else 'all finite rows'}.",
        "",
        "Strict valid-only means finite power, `is_valid=True`, power > 0, and below the per-band 99.99 percentile high-clip threshold.",
        "",
        "## What Was Tested",
        "",
        "1. Temporal repeats: whether a frequency channel repeats the same rounded raw power in consecutive samples.",
        "2. Adjacent-band sequence repeats: whether time-ordered neighboring frequency bands carry the same rounded raw power within a short gap.",
        "3. Near-time sweep repeats: whether multiple frequency bands inside a short telemetry sweep share the same rounded raw value.",
        "4. Per-frequency raw-value histograms for lower V.",
        "",
        "## Interpretation",
        "",
        "This separates ordinary quantization from a stronger stuck-code behavior. Quantization means values live on a discrete grid. Stuck-code behavior means the same raw value persists across consecutive samples or appears across multiple channels in the same short telemetry interval.",
        "",
        "## Temporal Repeat Summary",
        "",
        temporal.to_string(index=False),
        "",
        "## Adjacent-Band Sequence Summary",
        "",
        sequence_summary.to_string(index=False) if not sequence_summary.empty else "No adjacent-band sequence comparisons found.",
        "",
        "## Near-Time Sweep Summary",
        "",
        sweep_summary.to_string(index=False) if not sweep_summary.empty else "No near-time sweeps with repeated bands found.",
        "",
        "## Top Temporal Runs",
        "",
        temporal_runs.head(30).to_string(index=False) if not temporal_runs.empty else "No temporal runs length >= 3.",
        "",
        "## Top Adjacent-Band Same-Value Runs",
        "",
        sequence_runs.head(30).to_string(index=False) if not sequence_runs.empty else "No adjacent-band same-value runs.",
        "",
        "## Top Near-Time Sweep Repeats",
        "",
        sweep_repeats.head(30).to_string(index=False) if not sweep_repeats.empty else "No near-time repeated values across bands.",
        "",
        "## Diagnostic Plots",
        "",
    ]
    lines.extend(f"- {path}" for path in paths)
    lines.extend(["", "## Software Versions", "", "```", repr(software_versions()), "```"])
    path = out_dir / "lower_v_stuck_value_audit.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-csv", type=Path, default=CLEAN)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/lower_v_stuck_value_audit_v1")
    parser.add_argument("--chunk-size", type=int, default=250_000)
    parser.add_argument("--include-invalid", action="store_true")
    parser.add_argument("--adjacent-max-gap-s", type=float, default=12.0)
    parser.add_argument("--sweep-gap-s", type=float, default=6.0)
    parser.add_argument("--sweep-max-samples", type=int, default=12)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    require_valid = not args.include_invalid
    write_json(
        out_dir / "run_config.json",
        {
            "clean_csv": str(args.clean_csv),
            "chunk_size": args.chunk_size,
            "require_valid": require_valid,
            "adjacent_max_gap_s": args.adjacent_max_gap_s,
            "sweep_gap_s": args.sweep_gap_s,
            "sweep_max_samples": args.sweep_max_samples,
            "software_versions": software_versions(),
        },
    )
    df = _load_lower_v(args.clean_csv, args.chunk_size, require_valid=require_valid)
    print(f"loaded lower V rows: {len(df):,}", flush=True)
    temporal, temporal_runs = _temporal_stuck_summary(df)
    print("computed temporal repeated-value summary", flush=True)
    sequence_summary, sequence_runs = _sequence_stuck_summary(df, args.adjacent_max_gap_s)
    print("computed adjacent-band sequence summary", flush=True)
    sweep_summary, sweep_repeats = _sweep_summary(df, args.sweep_gap_s, args.sweep_max_samples)
    print("computed near-time sweep summary", flush=True)
    temporal.to_csv(out_dir / "lower_v_temporal_stuck_summary.csv", index=False)
    temporal_runs.to_csv(out_dir / "lower_v_temporal_stuck_runs.csv", index=False)
    sequence_summary.to_csv(out_dir / "lower_v_adjacent_band_same_value_summary.csv", index=False)
    sequence_runs.to_csv(out_dir / "lower_v_adjacent_band_same_value_runs.csv", index=False)
    sweep_summary.to_csv(out_dir / "lower_v_sweep_same_value_summary.csv", index=False)
    sweep_repeats.to_csv(out_dir / "lower_v_sweep_same_value_repeats.csv", index=False)
    paths = [_plot_histograms(df, out_dir), _plot_same_prev(temporal, out_dir)]
    print("wrote diagnostic plots", flush=True)
    report = _write_report(
        out_dir,
        temporal,
        temporal_runs,
        sequence_summary,
        sequence_runs,
        sweep_summary,
        sweep_repeats,
        paths,
        require_valid,
    )
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
