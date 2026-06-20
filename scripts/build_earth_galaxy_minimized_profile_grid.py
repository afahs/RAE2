#!/usr/bin/env python
"""Build Earth all-frequency grids after minimizing diffuse-Galaxy beam pickup.

The filter is intentionally based on the lower-V boresight, not only on the
Earth position.  For lunar occultations the lower V points toward the Moon, so
diffuse Galactic contamination is controlled by what the Moon-facing beam sees.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import astropy.units as u
from astropy.coordinates import CartesianRepresentation, FK4, Galactic, SkyCoord
from astropy.time import Time
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.util import ensure_dir, software_versions, write_json  # noqa: E402
from scripts.build_all_frequency_occultation_profile_grids import (  # noqa: E402
    CLEAN,
    EARTH_EVENTS,
    _read,
    collect_profiles,
    summarize_profiles,
)


LOWER_V = "rv2_coarse"


def _annotate_boresight_geometry(events: pd.DataFrame) -> pd.DataFrame:
    """Attach lower-V/Moon-center Galactic geometry to each unique event."""
    unique = events.drop_duplicates("event_id").copy()
    rep = CartesianRepresentation(
        pd.to_numeric(unique["moon_center_x"], errors="coerce").to_numpy(dtype=float) * u.one,
        pd.to_numeric(unique["moon_center_y"], errors="coerce").to_numpy(dtype=float) * u.one,
        pd.to_numeric(unique["moon_center_z"], errors="coerce").to_numpy(dtype=float) * u.one,
    )
    gal = SkyCoord(rep, frame=FK4(equinox=Time("B1950"))).galactic
    gal_center = SkyCoord(l=0.0 * u.deg, b=0.0 * u.deg, frame=Galactic())
    unique["lower_v_boresight_gal_l_deg"] = np.asarray(gal.l.deg, dtype=float)
    unique["lower_v_boresight_gal_b_deg"] = np.asarray(gal.b.deg, dtype=float)
    unique["lower_v_boresight_abs_gal_b_deg"] = np.abs(unique["lower_v_boresight_gal_b_deg"])
    unique["lower_v_boresight_gal_center_sep_deg"] = np.asarray(gal.separation(gal_center).deg, dtype=float)
    return events.merge(
        unique[
            [
                "event_id",
                "lower_v_boresight_gal_l_deg",
                "lower_v_boresight_gal_b_deg",
                "lower_v_boresight_abs_gal_b_deg",
                "lower_v_boresight_gal_center_sep_deg",
            ]
        ],
        on="event_id",
        how="left",
    )


def _robust_count_summary(events: pd.DataFrame, filtered: pd.DataFrame, points: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for freq in sorted(events["frequency_mhz"].dropna().unique()):
        for event_type in ["disappearance", "reappearance"]:
            pred = events[
                np.isclose(events["frequency_mhz"], freq) & events["event_type"].astype(str).eq(event_type)
            ]["event_id"].nunique()
            kept = filtered[
                np.isclose(filtered["frequency_mhz"], freq) & filtered["event_type"].astype(str).eq(event_type)
            ]["event_id"].nunique()
            prof = points[
                np.isclose(points.get("frequency_mhz", pd.Series(dtype=float)), freq)
                & points.get("event_type", pd.Series(dtype=object)).astype(str).eq(event_type)
            ]["event_id"].nunique() if not points.empty else 0
            median_bin_events = np.nan
            if not summary.empty:
                sub = summary[
                    np.isclose(summary["frequency_mhz"], freq)
                    & summary["event_type"].astype(str).eq(event_type)
                ]
                if not sub.empty:
                    median_bin_events = float(np.nanmedian(sub["n_events"]))
            rows.append(
                {
                    "frequency_mhz": float(freq),
                    "event_type": event_type,
                    "predicted_unique_events": int(pred),
                    "galaxy_minimized_unique_events": int(kept),
                    "events_with_profile_points": int(prof),
                    "median_events_per_time_bin": median_bin_events,
                }
            )
    return pd.DataFrame(rows)


def _plot_lower_v_grid(summary: pd.DataFrame, out_dir: Path, window_s: float) -> Path:
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
                & summary["antenna"].astype(str).eq(LOWER_V)
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
                    color="#d95f02",
                    ecolor="#d95f02",
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
        "Earth lower-V all-frequency normalized profiles: Galaxy-minimized boresight filter\n"
        "Filter: lower-V/Moon boresight away from Galactic plane and Galactic center; no trendline subtraction",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    path = out_dir / f"earth_lower_v_galaxy_minimized_all_frequency_profile_grid_{int(window_s)}s.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_report(
    out_dir: Path,
    plot_path: Path | None,
    all_events: pd.DataFrame,
    filtered_events: pd.DataFrame,
    count_summary: pd.DataFrame,
    min_abs_b: float,
    min_gc_sep: float,
    window_s: float,
    bin_s: float,
    inner_s: float,
) -> None:
    all_unique = all_events.drop_duplicates("event_id")
    filtered_unique = filtered_events.drop_duplicates("event_id")
    lines = [
        "# Earth Galaxy-Minimized Lower-V Profile Grid",
        "",
        "## Event Capture Audit",
        "",
        f"- Earth lower-V predicted event/frequency rows in source table: {len(all_events)}",
        f"- Earth unique occultation events in source table: {all_events['event_id'].nunique()}",
        f"- Lower-V event/frequency rows before Galaxy filter: {len(all_events)}",
        "",
        "The event predictor is producing the full post-November Earth event set. The smaller counts in profile plots are",
        "per time-bin stack counts after asking for valid samples in a +/- window around each predicted event.",
        "",
        "## Galaxy-Minimized Filter",
        "",
        "The filter is applied per unique event, then all lower-V frequency rows for retained events are stacked.",
        "",
        f"- lower-V/Moon-center boresight `|Galactic b| >= {min_abs_b:.1f} deg`",
        f"- lower-V/Moon-center boresight separation from Galactic center `>= {min_gc_sep:.1f} deg`",
        "",
        f"- retained unique Earth events: {len(filtered_unique)} / {len(all_unique)}",
        f"- retained lower-V event/frequency rows: {len(filtered_events)}",
        "",
        "This is a proxy for minimizing diffuse Galactic pickup in the Moon-facing beam. It is more physically relevant",
        "than the earlier Earth-position-only positive-latitude cut because the contaminating sky is seen by the antenna beam,",
        "not by the Earth position alone.",
        "",
        "## Counts By Frequency/Event Type",
        "",
        count_summary.to_string(index=False),
        "",
        "## Outputs",
        "",
        f"- `{plot_path}`" if plot_path else "- No plot generated.",
        "- `earth_galaxy_minimized_events.csv`",
        "- `earth_galaxy_minimized_profile_points.csv`",
        "- `earth_galaxy_minimized_profile_summary.csv`",
        "- `earth_galaxy_minimized_count_summary.csv`",
        "",
        f"Window: {window_s:.0f} s. Bin size: {bin_s:.0f} s. Normalization side region: |t| >= {inner_s:.0f} s.",
    ]
    (out_dir / "earth_galaxy_minimized_profile_grid_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(ROOT / "outputs/earth_galaxy_minimized_profile_grid_v1"))
    parser.add_argument("--min-abs-boresight-gal-b-deg", type=float, default=30.0)
    parser.add_argument("--min-boresight-gal-center-sep-deg", type=float, default=60.0)
    parser.add_argument("--window-s", type=float, default=900.0)
    parser.add_argument("--bin-s", type=float, default=60.0)
    parser.add_argument("--inner-s", type=float, default=15.0)
    args = parser.parse_args()

    out_dir = ensure_dir(Path(args.out_dir))
    clean = _read(CLEAN, parse_dates=["time"])
    events = _read(EARTH_EVENTS, parse_dates=["predicted_event_time"])
    events = events[events["source_name"].astype(str).str.lower().eq("earth")].copy()
    events = _annotate_boresight_geometry(events)
    lower_v_events = events[events["antenna"].astype(str).eq(LOWER_V)].copy()
    filtered_events = lower_v_events[
        pd.to_numeric(lower_v_events["lower_v_boresight_abs_gal_b_deg"], errors="coerce").ge(args.min_abs_boresight_gal_b_deg)
        & pd.to_numeric(lower_v_events["lower_v_boresight_gal_center_sep_deg"], errors="coerce").ge(args.min_boresight_gal_center_sep_deg)
    ].copy()

    write_json(
        out_dir / "run_config.json",
        {
            "source_name": "earth",
            "antenna": LOWER_V,
            "min_abs_boresight_gal_b_deg": float(args.min_abs_boresight_gal_b_deg),
            "min_boresight_gal_center_sep_deg": float(args.min_boresight_gal_center_sep_deg),
            "window_s": float(args.window_s),
            "bin_s": float(args.bin_s),
            "inner_s": float(args.inner_s),
            "event_table": str(EARTH_EVENTS.relative_to(ROOT)),
            "cleaned_timeseries": str(CLEAN.relative_to(ROOT)),
            "software_versions": software_versions(),
        },
    )

    events.to_csv(out_dir / "earth_events_with_lower_v_boresight_galaxy_geometry.csv", index=False)
    filtered_events.to_csv(out_dir / "earth_galaxy_minimized_events.csv", index=False)

    points = collect_profiles(clean, filtered_events, "earth", args.window_s, args.bin_s, args.inner_s)
    points = points[points["antenna"].astype(str).eq(LOWER_V)].copy() if not points.empty else points
    summary = summarize_profiles(points)
    points.to_csv(out_dir / "earth_galaxy_minimized_profile_points.csv", index=False)
    summary.to_csv(out_dir / "earth_galaxy_minimized_profile_summary.csv", index=False)
    count_summary = _robust_count_summary(lower_v_events, filtered_events, points, summary)
    count_summary.to_csv(out_dir / "earth_galaxy_minimized_count_summary.csv", index=False)
    plot_path = _plot_lower_v_grid(summary, out_dir, args.window_s) if not summary.empty else None
    if plot_path:
        print(plot_path)
    _write_report(
        out_dir,
        plot_path,
        lower_v_events,
        filtered_events,
        count_summary,
        args.min_abs_boresight_gal_b_deg,
        args.min_boresight_gal_center_sep_deg,
        args.window_s,
        args.bin_s,
        args.inner_s,
    )


if __name__ == "__main__":
    main()
