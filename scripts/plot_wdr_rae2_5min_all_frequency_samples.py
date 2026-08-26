#!/usr/bin/env python
"""Plot RAE-2 upper-V samples in +/-5 minute WDR Jupiter burst windows.

The input WDR table is the manually checked overlap transcription.  For each
reported Warwick/Dulk/Riddle interval, this script expands the interval by the
configured padding and plots all upper-V Jupiter beam-selected RAE-2 samples at
all frequencies in that expanded interval.
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

from rylevonberg.table_io import read_table  # noqa: E402
from rylevonberg.util import ensure_dir, software_versions, write_json  # noqa: E402
from scripts.run_jupiter_historical_window_phase_survey import load_windows  # noqa: E402


DEFAULT_SELECTED = (
    ROOT
    / "outputs/jupiter_upper_v_beam_gain10db_io_phase_frequency_v1/jupiter_upper_v_beam_selected_samples.npy"
)
DEFAULT_WINDOWS = ROOT / "configs/jupiter_warwick_dulk_riddle_1975_active_windows.csv"
DEFAULT_BURSTS = ROOT / "outputs/jupiter_upper_v_beam_single_frequency_burst_phase_v1/jupiter_blind_burst_catalog.csv"
DEFAULT_OUT = ROOT / "outputs/jupiter_wdr_rae2_5min_all_frequency_samples_v1"


PLOT_COLUMNS = [
    "time",
    "frequency_band",
    "frequency_mhz",
    "power",
    "log10_power",
    "daily_log10_residual",
    "local_normalization_window_s",
    "local_normalization_center_power",
    "local_normalized_power",
    "io_phase_spice_deg",
    "jupiter_cml_spice_deg",
    "jupiter_beam_relative_gain_db",
]


def load_selected_samples(path: Path) -> pd.DataFrame:
    samples = read_table(path, usecols=PLOT_COLUMNS, parse_dates=["time"])
    samples = samples.copy()
    samples["time"] = pd.to_datetime(samples["time"], errors="coerce")
    samples = samples.dropna(subset=["time", "frequency_mhz"]).sort_values("time").reset_index(drop=True)
    for col in [
        "frequency_band",
        "frequency_mhz",
        "power",
        "log10_power",
        "daily_log10_residual",
        "local_normalization_window_s",
        "local_normalization_center_power",
        "local_normalized_power",
        "io_phase_spice_deg",
        "jupiter_cml_spice_deg",
        "jupiter_beam_relative_gain_db",
    ]:
        if col in samples:
            samples[col] = pd.to_numeric(samples[col], errors="coerce")
    center = samples["local_normalization_center_power"].to_numpy(dtype=float)
    power = samples["power"].to_numpy(dtype=float)
    good = np.isfinite(power) & np.isfinite(center) & (center > 0.0)
    samples["relative_raw_power"] = np.nan
    samples.loc[good, "relative_raw_power"] = power[good] / center[good] - 1.0
    samples["relative_raw_power_percent"] = 100.0 * samples["relative_raw_power"]
    return samples


def load_burst_catalog(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    bursts = read_table(path, parse_dates=["burst_start_time", "burst_end_time", "burst_peak_time"])
    bursts = bursts.copy()
    bursts["burst_peak_time"] = pd.to_datetime(bursts["burst_peak_time"], errors="coerce")
    return bursts.dropna(subset=["burst_peak_time", "frequency_mhz"]).reset_index(drop=True)


def sample_window(samples: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    mask = samples["time"].ge(start) & samples["time"].le(end)
    return samples.loc[mask].copy()


def burst_window(bursts: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if bursts.empty:
        return bursts.copy()
    mask = bursts["burst_peak_time"].ge(start) & bursts["burst_peak_time"].le(end)
    return bursts.loc[mask].copy()


def rel_minutes(times: pd.Series, start: pd.Timestamp) -> np.ndarray:
    return (pd.to_datetime(times) - start).dt.total_seconds().to_numpy(dtype=float) / 60.0


def _window_slug(row: pd.Series) -> str:
    start = pd.Timestamp(row["event_start_time"])
    end = pd.Timestamp(row["event_end_time"])
    return (
        f"{row['historical_window_id']}_"
        f"{start.strftime('%Y%m%d_%H%M')}_{end.strftime('%H%M')}"
    )


def _decorate_time_axis(ax: plt.Axes, duration_min: float) -> None:
    ax.axvspan(0.0, float(duration_min), color="#fdb863", alpha=0.13, lw=0)
    ax.axvline(0.0, color="#d95f02", lw=1.0, alpha=0.88)
    ax.axvline(float(duration_min), color="#d95f02", lw=1.0, alpha=0.88)
    ax.grid(True, color="0.90", lw=0.45)


def _finite_percentile_limits(values: np.ndarray, lower: float, upper: float) -> tuple[float, float] | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    if finite.size >= 8:
        lo, hi = np.nanpercentile(finite, [lower, upper])
    else:
        lo, hi = np.nanmin(finite), np.nanmax(finite)
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    if np.isclose(lo, hi):
        pad = max(1.0, abs(float(hi)) * 0.10)
        lo -= pad
        hi += pad
    return float(lo), float(hi)


def plot_window_metric(
    window: pd.Series,
    samples: pd.DataFrame,
    bursts: pd.DataFrame,
    frequencies: np.ndarray,
    out_dir: Path,
    value_col: str,
    ylabel: str,
    filename_suffix: str,
    yscale: str = "linear",
    centered: bool = False,
) -> Path:
    event_start = pd.Timestamp(window["event_start_time"])
    event_end = pd.Timestamp(window["event_end_time"])
    expanded_start = pd.Timestamp(window["expanded_start_time"])
    expanded_end = pd.Timestamp(window["expanded_end_time"])
    duration_min = float((event_end - event_start).total_seconds() / 60.0)
    x_min = float((expanded_start - event_start).total_seconds() / 60.0)
    x_max = float((expanded_end - event_start).total_seconds() / 60.0)

    fig, axes = plt.subplots(
        len(frequencies),
        1,
        figsize=(13.0, max(1.82 * len(frequencies), 8.6)),
        sharex=True,
    )
    axes = np.atleast_1d(axes)
    for ax, freq in zip(axes, frequencies):
        sub = samples[np.isclose(samples["frequency_mhz"], float(freq))].copy()
        bsub = bursts[np.isclose(pd.to_numeric(bursts.get("frequency_mhz", pd.Series(dtype=float)), errors="coerce"), float(freq))].copy()
        _decorate_time_axis(ax, duration_min)
        if not sub.empty:
            x = rel_minutes(sub["time"], event_start)
            y = pd.to_numeric(sub[value_col], errors="coerce").to_numpy(dtype=float)
            good = np.isfinite(x) & np.isfinite(y)
            if yscale == "log":
                good &= y > 0.0
            if np.any(good):
                ax.scatter(
                    x[good],
                    y[good],
                    s=7.0,
                    alpha=0.62,
                    color="#386cb0",
                    linewidths=0,
                    rasterized=True,
                )
                if yscale == "log":
                    ax.set_yscale("log")
                    limits = _finite_percentile_limits(y[good], 2.0, 98.0)
                    if limits is not None and limits[0] > 0:
                        ax.set_ylim(limits[0] * 0.82, limits[1] * 1.22)
                else:
                    limits = _finite_percentile_limits(y[good], 1.0, 99.0)
                    if limits is not None:
                        lo, hi = limits
                        if centered:
                            lim = max(1.0, abs(lo), abs(hi))
                            ax.set_ylim(-1.12 * lim, 1.12 * lim)
                            ax.axhline(0.0, color="0.35", lw=0.8)
                        else:
                            pad = 0.12 * (hi - lo)
                            ax.set_ylim(lo - pad, hi + pad)
            else:
                ax.text(0.5, 0.52, "no finite samples", transform=ax.transAxes, ha="center", va="center", color="0.40")
        else:
            ax.text(0.5, 0.52, "no samples", transform=ax.transAxes, ha="center", va="center", color="0.40")

        if not bsub.empty:
            bx = rel_minutes(bsub["burst_peak_time"], event_start)
            ax.scatter(
                bx,
                np.full(len(bx), 0.94),
                transform=ax.get_xaxis_transform(),
                marker="v",
                s=23,
                color="#d73027",
                alpha=0.82,
                linewidths=0,
                label="blind RAE-2 burst peak",
                clip_on=False,
            )

        n_samples = int(len(sub))
        n_bursts = int(len(bsub))
        ax.set_ylabel(f"{freq:.2f} MHz\n{ylabel}")
        ax.text(
            0.99,
            0.86,
            f"n={n_samples:,}; bursts={n_bursts}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.4},
        )

    axes[-1].set_xlim(x_min, x_max)
    axes[-1].set_xlabel("minutes since WDR reported interval start")
    metadata = (
        f"{window['historical_window_id']}  {event_start:%Y-%m-%d %H:%M} to {event_end:%H:%M} UTC; "
        f"pad +/-{(event_start - expanded_start).total_seconds() / 60.0:.0f} min; "
        f"WDR {window['reported_freq_range_mhz']} MHz, intensity {window['intensity']}, {window['burstiness']}"
    )
    fig.suptitle(f"RAE-2 upper-V Jupiter beam-selected samples near WDR burst window\n{metadata}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.958])
    path = out_dir / f"{_window_slug(window)}_{filename_suffix}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def summarize_windows(windows: pd.DataFrame, samples: pd.DataFrame, bursts: pd.DataFrame, frequencies: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    event_rows: list[dict[str, object]] = []
    freq_rows: list[dict[str, object]] = []
    for _, window in windows.iterrows():
        expanded_start = pd.Timestamp(window["expanded_start_time"])
        expanded_end = pd.Timestamp(window["expanded_end_time"])
        sub = sample_window(samples, expanded_start, expanded_end)
        bsub = burst_window(bursts, expanded_start, expanded_end)
        event_rows.append(
            {
                "historical_window_id": str(window["historical_window_id"]),
                "event_start_time": pd.Timestamp(window["event_start_time"]),
                "event_end_time": pd.Timestamp(window["event_end_time"]),
                "expanded_start_time": expanded_start,
                "expanded_end_time": expanded_end,
                "reported_freq_range_mhz": window["reported_freq_range_mhz"],
                "intensity": int(window["intensity"]),
                "burstiness": window["burstiness"],
                "n_samples_all_frequencies": int(len(sub)),
                "n_frequencies_with_samples": int(sub["frequency_mhz"].nunique()) if not sub.empty else 0,
                "n_blind_rae2_bursts_all_frequencies": int(len(bsub)),
                "n_frequencies_with_blind_rae2_bursts": int(bsub["frequency_mhz"].nunique()) if not bsub.empty else 0,
            }
        )
        for freq in frequencies:
            fsub = sub[np.isclose(sub["frequency_mhz"], float(freq))]
            fb = bsub[np.isclose(pd.to_numeric(bsub.get("frequency_mhz", pd.Series(dtype=float)), errors="coerce"), float(freq))]
            local = pd.to_numeric(fsub["local_normalized_power"], errors="coerce") if not fsub.empty else pd.Series(dtype=float)
            relative = (
                pd.to_numeric(fsub["relative_raw_power_percent"], errors="coerce")
                if not fsub.empty
                else pd.Series(dtype=float)
            )
            power = pd.to_numeric(fsub["power"], errors="coerce") if not fsub.empty else pd.Series(dtype=float)
            freq_rows.append(
                {
                    "historical_window_id": str(window["historical_window_id"]),
                    "frequency_mhz": float(freq),
                    "n_samples": int(len(fsub)),
                    "n_blind_rae2_bursts": int(len(fb)),
                    "median_power": float(power.median()) if len(power.dropna()) else np.nan,
                    "max_power": float(power.max()) if len(power.dropna()) else np.nan,
                    "median_local_normalized_power": float(local.median()) if len(local.dropna()) else np.nan,
                    "max_local_normalized_power": float(local.max()) if len(local.dropna()) else np.nan,
                    "median_relative_raw_power_percent": float(relative.median()) if len(relative.dropna()) else np.nan,
                    "max_relative_raw_power_percent": float(relative.max()) if len(relative.dropna()) else np.nan,
                    "median_io_phase_deg": float(pd.to_numeric(fsub["io_phase_spice_deg"], errors="coerce").median())
                    if not fsub.empty
                    else np.nan,
                    "median_jupiter_cml_deg": float(pd.to_numeric(fsub["jupiter_cml_spice_deg"], errors="coerce").median())
                    if not fsub.empty
                    else np.nan,
                }
            )
    return pd.DataFrame(event_rows), pd.DataFrame(freq_rows)


def plot_summary_matrix(
    freq_summary: pd.DataFrame,
    windows: pd.DataFrame,
    value_col: str,
    title: str,
    cbar_label: str,
    out_dir: Path,
    filename: str,
    cmap: str,
    log_scale: bool = False,
    centered: bool = False,
) -> Path:
    order = windows["historical_window_id"].astype(str).tolist()
    mat = freq_summary.pivot_table(index="frequency_mhz", columns="historical_window_id", values=value_col, aggfunc="mean")
    mat = mat.reindex(index=sorted(freq_summary["frequency_mhz"].dropna().unique()), columns=order)
    values = mat.to_numpy(dtype=float)
    plot_values = values.copy()
    fig, ax = plt.subplots(figsize=(15.5, 5.8))
    if log_scale:
        from matplotlib.colors import LogNorm

        plot_values[plot_values <= 0] = np.nan
        finite = plot_values[np.isfinite(plot_values)]
        norm = LogNorm(vmin=1.0, vmax=max(1.0, float(np.nanmax(finite))) if finite.size else 1.0)
    elif centered:
        from matplotlib.colors import TwoSlopeNorm

        finite = plot_values[np.isfinite(plot_values)]
        vmax = max(1.0, float(np.nanpercentile(np.abs(finite), 98))) if finite.size else 1.0
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    else:
        from matplotlib.colors import Normalize

        finite = plot_values[np.isfinite(plot_values)]
        norm = Normalize(vmin=float(np.nanmin(finite)) if finite.size else 0.0, vmax=float(np.nanmax(finite)) if finite.size else 1.0)
    im = ax.imshow(plot_values, origin="lower", aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels([f"{freq:.2f}" for freq in mat.index])
    tick_step = max(1, int(np.ceil(len(order) / 18)))
    tick_positions = np.arange(0, len(order), tick_step)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([order[i] for i in tick_positions], rotation=90, fontsize=7)
    ax.set_xlabel("WDR interval")
    ax.set_ylabel("frequency (MHz)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.012, label=cbar_label)
    fig.tight_layout()
    path = out_dir / filename
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def write_report(
    out_dir: Path,
    windows: pd.DataFrame,
    event_summary: pd.DataFrame,
    freq_summary: pd.DataFrame,
    paths: list[Path],
    config: dict[str, object],
) -> Path:
    n_events = int(len(event_summary))
    n_with_samples = int((event_summary["n_samples_all_frequencies"] > 0).sum()) if n_events else 0
    n_with_bursts = int((event_summary["n_blind_rae2_bursts_all_frequencies"] > 0).sum()) if n_events else 0
    total_samples = int(event_summary["n_samples_all_frequencies"].sum()) if n_events else 0
    padding_min = float(config["padding_min"])
    plot_reading = [
        "- The x-axis is minutes since the WDR reported interval start.",
        "- The orange shaded span is the reported WDR active interval; the unshaded margin is the +/- padding.",
        "- Red downward ticks mark blind RAE-2 burst peaks from the existing per-frequency burst catalog.",
    ]
    if not bool(config.get("relative_only", False)):
        plot_reading.extend(
            [
                "- Raw-power plots use a separate log y-scale per frequency.",
                "- Local-normalized plots use a separate centered y-scale per frequency.",
            ]
        )
    plot_reading.append(
        "- Relative raw-power plots use `(raw power / centered local median raw power - 1) * 100%`; the local median is computed in a 20 minute centered window, i.e. +/-10 minutes."
    )
    lines = [
        f"# WDR +/-{padding_min:g} Minute RAE-2 All-Frequency Sample Plots",
        "",
        "Input WDR intervals are the manually checked Warwick/Dulk/Riddle overlap rows.",
        "Each interval is expanded by the configured padding, and the plots show upper-V samples that already pass the Jupiter beam-gain cut.",
        "",
        "## Summary",
        "",
        f"- WDR intervals plotted: {n_events}",
        f"- Intervals with at least one padded-window sample: {n_with_samples}",
        f"- Intervals with at least one blind RAE-2 burst peak in the padded window: {n_with_bursts}",
        f"- Total padded-window samples across all frequencies: {total_samples:,}",
        "",
        "## Plot Reading",
        "",
        *plot_reading,
        "",
        "## Configuration",
        "",
        *[f"- `{k}`: `{v}`" for k, v in config.items() if k != "software_versions"],
        "",
        "## Event Summary",
        "",
        event_summary[
            [
                "historical_window_id",
                "event_start_time",
                "event_end_time",
                "reported_freq_range_mhz",
                "intensity",
                "burstiness",
                "n_samples_all_frequencies",
                "n_frequencies_with_samples",
                "n_blind_rae2_bursts_all_frequencies",
            ]
        ].to_string(index=False),
        "",
        "## Files",
        "",
        *[f"- `{path}`" for path in paths],
    ]
    pad_label = f"{padding_min:g}".replace(".", "p")
    path = out_dir / f"wdr_rae2_padded{pad_label}min_all_frequency_sample_plot_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-samples", type=Path, default=DEFAULT_SELECTED)
    parser.add_argument("--windows", type=Path, default=DEFAULT_WINDOWS)
    parser.add_argument("--burst-catalog", type=Path, default=DEFAULT_BURSTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--padding-min", type=float, default=5.0)
    parser.add_argument("--skip-per-window-plots", action="store_true")
    parser.add_argument(
        "--relative-only",
        action="store_true",
        help="Only write relative raw-power per-window plots, plus summary matrices.",
    )
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    config = {
        "selected_samples": str(args.selected_samples),
        "windows": str(args.windows),
        "burst_catalog": str(args.burst_catalog),
        "padding_min": float(args.padding_min),
        "antenna": "upper V / rv1_coarse",
        "sample_selection": "Jupiter visible and beam-gain selected from existing upper-V table",
        "relative_raw_power_definition": "(power / local_normalization_center_power - 1) * 100",
        "relative_only": bool(args.relative_only),
        "software_versions": software_versions(),
    }
    write_json(out_dir / "run_config.json", config)

    samples = load_selected_samples(args.selected_samples)
    bursts = load_burst_catalog(args.burst_catalog)
    windows = load_windows(args.windows, padding_min=float(args.padding_min))
    frequencies = np.array(sorted(samples["frequency_mhz"].dropna().unique()), dtype=float)

    event_summary, freq_summary = summarize_windows(windows, samples, bursts, frequencies)
    pad_label = f"{float(args.padding_min):g}".replace(".", "p")
    event_summary_name = f"wdr_rae2_padded{pad_label}min_event_sample_summary.csv"
    freq_summary_name = f"wdr_rae2_padded{pad_label}min_frequency_sample_summary.csv"
    event_summary.to_csv(out_dir / event_summary_name, index=False)
    freq_summary.to_csv(out_dir / freq_summary_name, index=False)

    paths: list[Path] = [
        out_dir / "run_config.json",
        out_dir / event_summary_name,
        out_dir / freq_summary_name,
    ]
    paths.append(
        plot_summary_matrix(
            freq_summary,
            windows,
            value_col="n_samples",
            title=f"RAE-2 upper-V beam-selected sample counts in WDR +/-{float(args.padding_min):g} minute windows",
            cbar_label="samples",
            out_dir=out_dir,
            filename=f"wdr_rae2_padded{pad_label}min_sample_counts_by_event_frequency.png",
            cmap="viridis",
            log_scale=True,
        )
    )
    paths.append(
        plot_summary_matrix(
            freq_summary,
            windows,
            value_col="max_local_normalized_power",
            title=f"Peak local-normalized RAE-2 sample in WDR +/-{float(args.padding_min):g} minute windows",
            cbar_label="max local-normalized power",
            out_dir=out_dir,
            filename=f"wdr_rae2_padded{pad_label}min_peak_local_normalized_power_by_event_frequency.png",
            cmap="coolwarm",
            centered=True,
        )
    )
    paths.append(
        plot_summary_matrix(
            freq_summary,
            windows,
            value_col="n_blind_rae2_bursts",
            title=f"Blind RAE-2 burst peaks in WDR +/-{float(args.padding_min):g} minute windows",
            cbar_label="blind RAE-2 burst peaks",
            out_dir=out_dir,
            filename=f"wdr_rae2_padded{pad_label}min_blind_burst_counts_by_event_frequency.png",
            cmap="magma",
            log_scale=True,
        )
    )
    paths.append(
        plot_summary_matrix(
            freq_summary,
            windows,
            value_col="max_relative_raw_power_percent",
            title="Peak relative raw power in WDR padded windows, local median from +/-10 min",
            cbar_label="max relative raw power (%)",
            out_dir=out_dir,
            filename=f"wdr_rae2_padded{pad_label}min_peak_relative_raw_power_percent_by_event_frequency.png",
            cmap="coolwarm",
            centered=True,
        )
    )

    if not args.skip_per_window_plots:
        for _, window in windows.iterrows():
            expanded = sample_window(samples, pd.Timestamp(window["expanded_start_time"]), pd.Timestamp(window["expanded_end_time"]))
            expanded_bursts = burst_window(bursts, pd.Timestamp(window["expanded_start_time"]), pd.Timestamp(window["expanded_end_time"]))
            if not args.relative_only:
                paths.append(
                    plot_window_metric(
                        window,
                        expanded,
                        expanded_bursts,
                        frequencies,
                        out_dir,
                        value_col="power",
                        ylabel="raw power",
                        filename_suffix=f"allfreq_raw_power_padded{int(args.padding_min)}min",
                        yscale="log",
                    )
                )
                paths.append(
                    plot_window_metric(
                        window,
                        expanded,
                        expanded_bursts,
                        frequencies,
                        out_dir,
                        value_col="local_normalized_power",
                        ylabel="local z",
                        filename_suffix=f"allfreq_local_normalized_power_padded{int(args.padding_min)}min",
                        centered=True,
                    )
                )
            paths.append(
                plot_window_metric(
                    window,
                    expanded,
                    expanded_bursts,
                    frequencies,
                    out_dir,
                    value_col="relative_raw_power_percent",
                    ylabel="relative raw (%)",
                    filename_suffix=f"allfreq_relative_raw_power_percent_padded{int(args.padding_min)}min",
                    centered=True,
                )
            )

    report = write_report(out_dir, windows, event_summary, freq_summary, paths, config)
    print(report)


if __name__ == "__main__":
    main()
