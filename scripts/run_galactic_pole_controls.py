#!/usr/bin/env python
"""Run fast stacked-profile controls for the Galactic poles."""

from __future__ import annotations

import argparse
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
from rylevonberg.stacking import detrend_profile_values  # noqa: E402
from rylevonberg.util import datetime_ns, ensure_dir, robust_sigma, write_json  # noqa: E402


ANTENNA_LABELS = {
    "rv1_coarse": "upper V",
    "rv2_coarse": "lower V",
}


def _read(path: Path, **kwargs) -> pd.DataFrame:
    return read_table(path, low_memory=False, **kwargs)


def _source_title(name: str) -> str:
    return name.replace("_", " ").title()


def _channel_arrays(clean: pd.DataFrame) -> dict[tuple[int, str], dict[str, np.ndarray]]:
    out: dict[tuple[int, str], dict[str, np.ndarray]] = {}
    for (freq, ant), grp in clean.groupby(["frequency_band", "antenna"], sort=True):
        grp = grp.sort_values("time")
        out[(int(freq), str(ant))] = {
            "time_ns": datetime_ns(pd.DatetimeIndex(grp["time"])),
            "power": grp["power"].to_numpy(dtype=float),
            "valid": grp["is_valid"].to_numpy(dtype=bool) if "is_valid" in grp else np.isfinite(grp["power"].to_numpy(dtype=float)),
        }
    return out


def _empty_accumulator() -> dict[str, float]:
    return {
        "sum": 0.0,
        "sum2": 0.0,
        "n_samples": 0,
        "template_dot_value": 0.0,
        "template_dot_template": 0.0,
    }


def stack_quiet_controls(
    clean: pd.DataFrame,
    events: pd.DataFrame,
    window_s: float,
    bin_s: float,
    baseline_mode: str,
    sideband_exclusion_s: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    channels = _channel_arrays(clean)
    accum: dict[tuple, dict[str, float]] = {}
    event_sets: dict[tuple, set[int]] = {}
    event_counts: dict[tuple, int] = {}

    half_ns = int(float(window_s) * 1e9)
    for ev in events.itertuples(index=False):
        source = str(ev.source_name)
        event_type = str(ev.event_type)
        freq = int(ev.frequency_band)
        ant = str(ev.antenna)
        event_id = int(ev.event_id)
        chan = channels.get((freq, ant))
        if chan is None:
            continue
        event_ns = pd.Timestamp(ev.predicted_event_time).value
        time_ns = chan["time_ns"]
        lo = int(np.searchsorted(time_ns, event_ns - half_ns, side="left"))
        hi = int(np.searchsorted(time_ns, event_ns + half_ns, side="right"))
        if hi <= lo:
            continue
        tr = (time_ns[lo:hi] - event_ns).astype(float) / 1e9
        y = chan["power"][lo:hi]
        keep = (np.abs(tr) <= float(window_s)) & chan["valid"][lo:hi] & np.isfinite(y)
        if int(np.count_nonzero(keep)) < 6:
            continue
        tr = tr[keep]
        y = y[keep]
        profile, template, _, _ = detrend_profile_values(
            tr,
            y,
            event_type,
            baseline_mode=baseline_mode,
            sideband_exclusion_seconds=sideband_exclusion_s,
            normalize=True,
        )
        t_bins = np.round(tr / float(bin_s)) * float(bin_s)
        group_key = (source, event_type, freq, ant)
        event_counts[group_key] = event_counts.get(group_key, 0) + 1
        event_sets.setdefault(group_key, set()).add(event_id)
        for t_bin, value, tmpl in zip(t_bins, profile, template):
            key = (source, event_type, freq, ant, float(t_bin))
            slot = accum.setdefault(key, _empty_accumulator())
            slot["sum"] += float(value)
            slot["sum2"] += float(value * value)
            slot["n_samples"] += 1
            slot["template_dot_value"] += float(value * tmpl)
            slot["template_dot_template"] += float(tmpl * tmpl)

    rows = []
    for (source, event_type, freq, ant, t_bin), vals in sorted(accum.items()):
        n = int(vals["n_samples"])
        mean = vals["sum"] / n if n else np.nan
        var = (vals["sum2"] - vals["sum"] ** 2 / n) / (n - 1) if n > 1 else np.nan
        rows.append(
            {
                "source_name": source,
                "event_type": event_type,
                "frequency_band": freq,
                "frequency_mhz": FREQUENCY_MAP_MHZ.get(freq, np.nan),
                "antenna": ant,
                "antenna_label": ANTENNA_LABELS.get(ant, ant),
                "t_bin_sec": t_bin,
                "mean": mean,
                "sem": float(np.sqrt(var / n)) if np.isfinite(var) and n else np.nan,
                "n_samples": n,
                "n_events": len(event_sets.get((source, event_type, freq, ant), set())),
            }
        )
    stacked = pd.DataFrame(rows)

    summary_rows = []
    for group_key in sorted(event_sets):
        source, event_type, freq, ant = group_key
        sub_keys = [k for k in accum if k[:4] == group_key]
        values = np.array([accum[k]["sum"] / accum[k]["n_samples"] for k in sub_keys if accum[k]["n_samples"] > 0], dtype=float)
        # Use bin-centered templates for a compact stack score.
        t_bins = np.array([k[4] for k in sub_keys if accum[k]["n_samples"] > 0], dtype=float)
        template = np.where(t_bins < 0.0, 0.5, -0.5) if event_type == "disappearance" else np.where(t_bins < 0.0, -0.5, 0.5)
        denom = float(np.dot(template, template))
        amp = float(np.dot(values, template) / denom) if denom > 0 and values.size else np.nan
        sig = robust_sigma(values)
        snr = float(amp * np.sqrt(denom) / sig) if np.isfinite(sig) and sig > 0 else np.nan
        summary_rows.append(
            {
                "source_name": source,
                "event_type": event_type,
                "frequency_band": freq,
                "frequency_mhz": FREQUENCY_MAP_MHZ.get(freq, np.nan),
                "antenna": ant,
                "antenna_label": ANTENNA_LABELS.get(ant, ant),
                "n_events": len(event_sets[group_key]),
                "n_event_windows_used": event_counts.get(group_key, 0),
                "stacked_amplitude": amp,
                "stacked_snr": snr,
            }
        )
    summary = pd.DataFrame(summary_rows)
    return stacked, summary


def plot_profile_grid(stacked: pd.DataFrame, source: str, out_dir: Path) -> Path:
    source_df = stacked[stacked["source_name"].eq(source)].copy()
    freqs = sorted(source_df["frequency_band"].dropna().astype(int).unique())
    ants = ["rv2_coarse", "rv1_coarse"]
    fig, axes = plt.subplots(len(freqs), len(ants), figsize=(10, max(12, 1.35 * len(freqs))), sharex=True, sharey=True)
    if len(freqs) == 1:
        axes = np.array([axes])
    colors = {"disappearance": "#4c78a8", "reappearance": "#d95f02"}
    for i, freq in enumerate(freqs):
        for j, ant in enumerate(ants):
            ax = axes[i, j]
            sub = source_df[(source_df["frequency_band"].eq(freq)) & (source_df["antenna"].eq(ant))]
            for event_type, grp in sub.groupby("event_type", sort=True):
                grp = grp.sort_values("t_bin_sec")
                label = str(event_type) if i == 0 and j == 0 else None
                x = grp["t_bin_sec"].to_numpy(dtype=float) / 60.0
                y = grp["mean"].to_numpy(dtype=float)
                ax.plot(x, y, marker="o", ms=2.5, lw=1.2, color=colors.get(str(event_type), "black"), label=label)
                if "sem" in grp:
                    sem = grp["sem"].to_numpy(dtype=float)
                    ax.fill_between(x, y - sem, y + sem, color=colors.get(str(event_type), "black"), alpha=0.14, linewidth=0)
            ax.axvline(0, color="black", lw=0.8)
            ax.axhline(0, color="0.5", lw=0.6)
            if i == 0:
                ax.set_title(ANTENNA_LABELS.get(ant, ant))
            if j == 0:
                ax.set_ylabel(f"{FREQUENCY_MAP_MHZ.get(freq, np.nan):.2f} MHz\nnorm. power")
            if i == len(freqs) - 1:
                ax.set_xlabel("minutes from predicted event")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=False)
    fig.suptitle(f"{_source_title(source)} quiet-control stacked profiles", y=0.995)
    fig.tight_layout(rect=(0, 0, 0.98, 0.985))
    path = out_dir / f"{source}_stacked_profile_grid_600s.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_snr_spectrum(summary: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    colors = {"north_galactic_pole": "#4c78a8", "south_galactic_pole": "#d95f02"}
    for ax, ant in zip(axes, ["rv2_coarse", "rv1_coarse"]):
        sub = summary[summary["antenna"].eq(ant)].copy()
        combo = (
            sub.groupby(["source_name", "frequency_band", "frequency_mhz"], as_index=False)
            .agg(combined_abs_snr=("stacked_snr", lambda x: float(np.nanmax(np.abs(pd.to_numeric(x, errors="coerce"))))))
            .sort_values("frequency_mhz")
        )
        for source, grp in combo.groupby("source_name", sort=True):
            ax.plot(grp["frequency_mhz"], grp["combined_abs_snr"], marker="o", lw=1.5, color=colors.get(source), label=_source_title(source))
        ax.axhline(3.0, color="0.4", lw=0.8, ls="--", label="|SNR|=3" if ant == "rv2_coarse" else None)
        ax.set_title(ANTENNA_LABELS.get(ant, ant))
        ax.set_xlabel("frequency (MHz)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("max |stacked SNR| across event types")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Galactic-pole quiet-control stack spectrum")
    fig.tight_layout()
    path = out_dir / "galactic_pole_quiet_control_snr_spectrum.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_report(out_dir: Path, summary: pd.DataFrame, plot_paths: list[Path], args: argparse.Namespace) -> Path:
    top = summary.assign(abs_snr=summary["stacked_snr"].abs()).sort_values("abs_snr", ascending=False).head(12)
    lines = [
        "# Galactic Pole Quiet-Control Stack Report",
        "",
        "Purpose: run the same occultation-stack machinery on the North and South Galactic Poles, which should be comparatively radio quiet high-Galactic-latitude controls.",
        "",
        "## Run Settings",
        "",
        f"- Cleaned data: `{args.cleaned}`",
        f"- Events: `{args.events}`",
        f"- Window half-width: `{args.window_seconds}` s",
        f"- Bin size: `{args.bin_seconds}` s",
        f"- Baseline mode: `{args.baseline_mode}`",
        f"- Sideband exclusion: `{args.sideband_exclusion_seconds}` s",
        "",
        "## Strongest Quiet-Control Stack Scores",
        "",
        top[["source_name", "event_type", "frequency_mhz", "antenna_label", "n_events", "stacked_amplitude", "stacked_snr"]].to_string(index=False),
        "",
        "## Plots",
        "",
    ]
    lines.extend([f"- `{p}`" for p in plot_paths])
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "These are not source-detection plots. They are null/quiet-region controls. A clean source-detection pipeline should not produce a coherent Earth-like occultation profile at both Galactic poles.",
            "Residual nonzero SNR can still occur from baseline structure, antenna beam gradients, sampling cadence, or sky-background gradients. The relevant comparison is whether pole-control stacks are weaker and less coherent than candidate source stacks.",
        ]
    )
    path = out_dir / "galactic_pole_quiet_control_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cleaned", default="outputs/control_survey_earth_sun_postnov1974_v2/01_ingest/cleaned_timeseries.csv")
    parser.add_argument("--events", default="outputs/quiet_galactic_poles_control_v1/02_events/predicted_events.csv")
    parser.add_argument("--output-dir", default="outputs/quiet_galactic_poles_control_v1/04_fast_stack")
    parser.add_argument("--window-seconds", type=float, default=600.0)
    parser.add_argument("--bin-seconds", type=float, default=60.0)
    parser.add_argument("--baseline-mode", default="sideband_linear")
    parser.add_argument("--sideband-exclusion-seconds", type=float, default=120.0)
    args = parser.parse_args()

    out_dir = ensure_dir(ROOT / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir))
    clean = _read(ROOT / args.cleaned if not Path(args.cleaned).is_absolute() else Path(args.cleaned), parse_dates=["time"])
    events = _read(ROOT / args.events if not Path(args.events).is_absolute() else Path(args.events), parse_dates=["predicted_event_time"])
    stacked, summary = stack_quiet_controls(
        clean,
        events,
        window_s=args.window_seconds,
        bin_s=args.bin_seconds,
        baseline_mode=args.baseline_mode,
        sideband_exclusion_s=args.sideband_exclusion_seconds,
    )
    stacked.to_csv(out_dir / "galactic_pole_stacked_profiles.csv", index=False)
    summary.to_csv(out_dir / "galactic_pole_stack_summary.csv", index=False)
    paths = []
    for source in sorted(stacked["source_name"].unique()):
        paths.append(plot_profile_grid(stacked, source, out_dir))
    paths.append(plot_snr_spectrum(summary, out_dir))
    report = write_report(out_dir, summary, paths, args)
    write_json(out_dir / "run_config.json", vars(args))
    print(report)


if __name__ == "__main__":
    main()
