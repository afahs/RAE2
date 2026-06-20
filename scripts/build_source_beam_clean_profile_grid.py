#!/usr/bin/env python
"""Build beam-clean all-frequency profile grids for a selected source.

This applies the same physically informed diffuse-Galaxy event removal used for
Earth to fixed-source occultation targets such as Fornax A.  The cleaning is
per source/frequency/antenna/event-type row and uses a beam-weighted PySM
synchrotron prediction to select events with low diffuse level and low diffuse
slope.
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

from rylevonberg.util import ensure_dir, software_versions, write_json  # noqa: E402
from scripts.build_all_frequency_occultation_profile_grids import (  # noqa: E402
    BRIGHT_EVENTS,
    CLEAN,
    _read,
    collect_profiles,
    summarize_profiles,
)
from rylevonberg.geometry import normalize_vectors  # noqa: E402
from rylevonberg.util import datetime_ns  # noqa: E402
from scripts.build_earth_beam_clean_profile_grid import (  # noqa: E402
    MODEL_NSIDE,
    _beam_weighted_sky,
    _load_beam,
    _load_sky_i,
    _nearest_beam,
    _pixel_fk4_vectors,
    _robust_slope,
    _select_clean_events,
)


ANT_LABEL = {"rv1_coarse": "upper V", "rv2_coarse": "lower V"}
ANT_COLOR = {"rv1_coarse": "#4c78a8", "rv2_coarse": "#d95f02"}


def _source_title(source: str) -> str:
    return {"fornax_a": "Fornax A", "cyg_a": "Cyg A", "cas_a": "Cas A"}.get(source, source.replace("_", " ").title())


def _cached_event_model_metrics(clean: pd.DataFrame, events: pd.DataFrame, antenna: str, window_s: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute diffuse model metrics by caching model values on the clean time grid."""
    rows = []
    input_rows = []
    pixel_vecs = _pixel_fk4_vectors(MODEL_NSIDE)
    clean = clean[clean["antenna"].astype(str).eq(antenna)].copy()
    for (band, freq), band_events in events.groupby(["frequency_band", "frequency_mhz"], sort=True):
        channel = clean[
            clean["frequency_band"].astype(int).eq(int(band))
            & clean["antenna"].astype(str).eq(antenna)
        ].sort_values("time").reset_index(drop=True)
        if channel.empty:
            continue
        sky_i, sky_path = _load_sky_i(float(freq))
        beam_freq, eplane, hplane = _nearest_beam(float(freq))
        beam_angles, beam_gains = _load_beam(eplane, hplane)
        input_rows.append(
            {
                "frequency_band": int(band),
                "frequency_mhz": float(freq),
                "sky_map": str(sky_path),
                "beam_model_frequency_mhz": float(beam_freq),
                "eplane_beam": str(eplane),
                "hplane_beam": str(hplane),
                "nside": MODEL_NSIDE,
            }
        )
        pos = channel[["position_x", "position_y", "position_z"]].to_numpy(dtype=float)
        lower_axes = normalize_vectors(-pos)
        axes = lower_axes if antenna == "rv2_coarse" else -lower_axes
        model_series = _beam_weighted_sky(axes, sky_i, pixel_vecs, beam_angles, beam_gains)
        time_ns = datetime_ns(channel["time"])
        for ev in band_events.itertuples(index=False):
            event_ns = pd.Timestamp(ev.predicted_event_time).value
            half_ns = int(float(window_s) * 1e9)
            lo = int(np.searchsorted(time_ns, event_ns - half_ns, side="left"))
            hi = int(np.searchsorted(time_ns, event_ns + half_ns, side="right"))
            if hi <= lo:
                continue
            t = (time_ns[lo:hi] - event_ns).astype(float) / 1e9
            model = model_series[lo:hi]
            keep = np.isfinite(t) & np.isfinite(model) & (np.abs(t) <= float(window_s))
            if np.count_nonzero(keep) < 8:
                continue
            level = float(np.nanmedian(model[keep]))
            slope = _robust_slope(t[keep], model[keep])
            frac_slope = float(slope / level) if np.isfinite(level) and level != 0 else np.nan
            rows.append(
                {
                    "event_id": ev.event_id,
                    "event_type": ev.event_type,
                    "predicted_event_time": ev.predicted_event_time,
                    "frequency_band": int(band),
                    "frequency_mhz": float(freq),
                    "antenna": antenna,
                    "model_diffuse_level": level,
                    "model_diffuse_abs_frac_slope_per_window": abs(frac_slope) if np.isfinite(frac_slope) else np.nan,
                    "model_diffuse_frac_slope_per_window": frac_slope,
                    "n_model_samples": int(np.count_nonzero(keep)),
                    "beam_model_frequency_mhz": float(beam_freq),
                    "sky_map": str(sky_path),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(input_rows)


def _plot_grid(summary: pd.DataFrame, source: str, antenna: str, out_dir: Path, window_s: float) -> Path:
    freqs = sorted(summary["frequency_mhz"].dropna().unique())
    fig, axes = plt.subplots(len(freqs), 2, figsize=(12, max(10, 1.35 * len(freqs))), sharex=True)
    if len(freqs) == 1:
        axes = np.asarray([axes])
    for i, freq in enumerate(freqs):
        for j, event_type in enumerate(["disappearance", "reappearance"]):
            ax = axes[i, j]
            sub = summary[
                np.isclose(summary["frequency_mhz"], freq)
                & summary["event_type"].astype(str).eq(event_type)
                & summary["antenna"].astype(str).eq(antenna)
            ].sort_values("t_bin_sec")
            if not sub.empty:
                ax.errorbar(
                    sub["t_bin_sec"],
                    sub["median_z_power"],
                    yerr=sub["median_z_power_err"],
                    marker="o",
                    markersize=2.5,
                    linewidth=1.2,
                    elinewidth=0.65,
                    capsize=1.2,
                    color=ANT_COLOR.get(antenna, "#333333"),
                    ecolor=ANT_COLOR.get(antenna, "#333333"),
                )
                ax.text(
                    0.02,
                    0.94,
                    f"median n/bin={np.nanmedian(sub['n_events']):.0f}",
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=7,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.65),
                )
            ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
            ax.axhline(0, color="0.6", linewidth=0.7)
            ax.set_title(f"{freq:.2f} MHz {event_type}", fontsize=9)
            if j == 0:
                ax.set_ylabel("normalized power")
            if i == len(freqs) - 1:
                ax.set_xlabel("seconds from predicted event")
    fig.suptitle(
        f"{_source_title(source)} {ANT_LABEL.get(antenna, antenna)} all-frequency normalized profiles: beam-clean diffuse-Galaxy filter\n"
        "Filter uses digitized beam cuts + PySM sky map; no trendline subtraction",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    path = out_dir / f"{source}_{antenna}_beam_clean_all_frequency_profile_grid_{int(window_s)}s.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _count_summary(all_events: pd.DataFrame, selected: pd.DataFrame, points: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for freq in sorted(all_events["frequency_mhz"].dropna().unique()):
        for event_type in ["disappearance", "reappearance"]:
            pred = all_events[
                np.isclose(all_events["frequency_mhz"], freq) & all_events["event_type"].astype(str).eq(event_type)
            ]["event_id"].nunique()
            kept = selected[
                np.isclose(selected["frequency_mhz"], freq) & selected["event_type"].astype(str).eq(event_type)
            ]["event_id"].nunique()
            prof = (
                points[
                    np.isclose(points.get("frequency_mhz", pd.Series(dtype=float)), freq)
                    & points.get("event_type", pd.Series(dtype=object)).astype(str).eq(event_type)
                ]["event_id"].nunique()
                if not points.empty
                else 0
            )
            med = np.nan
            if not summary.empty:
                sub = summary[np.isclose(summary["frequency_mhz"], freq) & summary["event_type"].astype(str).eq(event_type)]
                if not sub.empty:
                    med = float(np.nanmedian(sub["n_events"]))
            rows.append(
                {
                    "frequency_mhz": float(freq),
                    "event_type": event_type,
                    "predicted_unique_events": int(pred),
                    "beam_clean_selected_events": int(kept),
                    "events_with_profile_points": int(prof),
                    "median_events_per_time_bin": med,
                }
            )
    return pd.DataFrame(rows)


def _write_report(
    out: Path,
    source: str,
    antenna: str,
    plot_path: Path,
    metrics: pd.DataFrame,
    selected: pd.DataFrame,
    counts: pd.DataFrame,
    keep_fraction: float,
    max_level_rank: float,
    max_slope_rank: float,
) -> None:
    lines = [
        f"# {_source_title(source)} Beam-Clean Profile Grid",
        "",
        "## Method",
        "",
        "This applies a physically informed diffuse-Galaxy removal before stacking:",
        "",
        "- compute beam-weighted PySM synchrotron brightness through the selected antenna beam;",
        "- estimate each event/frequency row's diffuse level and absolute fractional diffuse slope;",
        "- keep rows with low diffuse level and low diffuse slope within each frequency/event-type group;",
        "- stack the retained rows using the same local normalized raw-power profile method.",
        "",
        "The beam remains approximate because only 1D E/H-plane digitized cuts are available.",
        "",
        "## Selection",
        "",
        f"- source: `{source}`",
        f"- antenna: `{antenna}` ({ANT_LABEL.get(antenna, antenna)})",
        f"- requested maximum level rank: {max_level_rank:.2f}",
        f"- requested maximum slope rank: {max_slope_rank:.2f}",
        f"- fallback keep fraction: {keep_fraction:.2f}",
        f"- model rows computed: {len(metrics)}",
        f"- selected event/frequency rows: {len(selected)}",
        f"- selected unique events: {selected['event_id'].nunique()}",
        "",
        "## Count Summary",
        "",
        counts.to_string(index=False),
        "",
        "## Outputs",
        "",
        f"- `{plot_path}`",
        f"- `{source}_{antenna}_beam_clean_model_metrics.csv`",
        f"- `{source}_{antenna}_beam_clean_selected_events.csv`",
        f"- `{source}_{antenna}_beam_clean_profile_points.csv`",
        f"- `{source}_{antenna}_beam_clean_profile_summary.csv`",
    ]
    (out / f"{source}_{antenna}_beam_clean_profile_grid_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="fornax_a")
    parser.add_argument("--antenna", default="rv2_coarse", choices=["rv1_coarse", "rv2_coarse"])
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--keep-fraction", type=float, default=0.25)
    parser.add_argument("--max-level-rank", type=float, default=0.35)
    parser.add_argument("--max-slope-rank", type=float, default=0.35)
    parser.add_argument("--window-s", type=float, default=900.0)
    parser.add_argument("--bin-s", type=float, default=60.0)
    parser.add_argument("--inner-s", type=float, default=15.0)
    args = parser.parse_args()

    source = str(args.source).lower()
    antenna = str(args.antenna)
    out = ensure_dir(Path(args.out_dir) if args.out_dir else ROOT / f"outputs/{source}_{antenna}_beam_clean_profile_grid_v1")
    clean = _read(CLEAN, parse_dates=["time"])
    events = _read(BRIGHT_EVENTS, parse_dates=["predicted_event_time"])
    events = events[
        events["source_name"].astype(str).str.lower().eq(source)
        & events["antenna"].astype(str).eq(antenna)
    ].copy()
    if events.empty:
        raise RuntimeError(f"No events found for source={source} antenna={antenna}")

    metrics, model_inputs = _cached_event_model_metrics(clean, events, antenna, args.window_s)
    selected_metrics = _select_clean_events(metrics, args.keep_fraction, args.max_level_rank, args.max_slope_rank)
    selected_events = events.merge(
        selected_metrics[["event_id", "frequency_band", "frequency_mhz", "beam_clean_score", "level_rank", "slope_rank", "beam_clean_selection_mode"]],
        on=["event_id", "frequency_band", "frequency_mhz"],
        how="inner",
    )
    points = collect_profiles(clean, selected_events, source, args.window_s, args.bin_s, args.inner_s)
    points = points[points["antenna"].astype(str).eq(antenna)].copy() if not points.empty else points
    summary = summarize_profiles(points)
    counts = _count_summary(events, selected_metrics, points, summary)

    write_json(
        out / "run_config.json",
        {
            "source_name": source,
            "antenna": antenna,
            "window_s": float(args.window_s),
            "bin_s": float(args.bin_s),
            "inner_s": float(args.inner_s),
            "keep_fraction": float(args.keep_fraction),
            "max_level_rank": float(args.max_level_rank),
            "max_slope_rank": float(args.max_slope_rank),
            "event_table": str(BRIGHT_EVENTS.relative_to(ROOT)),
            "software_versions": software_versions(),
        },
    )
    model_inputs.to_csv(out / f"{source}_{antenna}_beam_clean_model_inputs.csv", index=False)
    metrics.to_csv(out / f"{source}_{antenna}_beam_clean_model_metrics.csv", index=False)
    selected_metrics.to_csv(out / f"{source}_{antenna}_beam_clean_selected_metrics.csv", index=False)
    selected_events.to_csv(out / f"{source}_{antenna}_beam_clean_selected_events.csv", index=False)
    points.to_csv(out / f"{source}_{antenna}_beam_clean_profile_points.csv", index=False)
    summary.to_csv(out / f"{source}_{antenna}_beam_clean_profile_summary.csv", index=False)
    counts.to_csv(out / f"{source}_{antenna}_beam_clean_count_summary.csv", index=False)
    plot_path = _plot_grid(summary, source, antenna, out, args.window_s)
    _write_report(out, source, antenna, plot_path, metrics, selected_metrics, counts, args.keep_fraction, args.max_level_rank, args.max_slope_rank)
    print(plot_path)
    print(out / f"{source}_{antenna}_beam_clean_profile_grid_report.md")


if __name__ == "__main__":
    main()
