#!/usr/bin/env python
"""Build all-frequency event profile grids for before/after visual inspection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rylevonberg.table_io import read_table

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.util import datetime_ns, ensure_dir, robust_sigma, software_versions, write_json  # noqa: E402


CLEAN = ROOT / "outputs/control_survey_earth_sun_postnov1974_v2/01_ingest/cleaned_timeseries.csv"
SUN_EVENTS = ROOT / "outputs/pipeline_confidence_audit_v2/sun_audit_input_events.csv"
BRIGHT_EVENTS = ROOT / "outputs/control_survey_bright_sources_postnov1974_v1/02_events/predicted_events.csv"
EARTH_EVENTS = ROOT / "outputs/planetary_confirmation_survey_science_baseline_v2/events/all_planet_predicted_events.csv"
ANT_LABEL = {"rv1_coarse": "upper V", "rv2_coarse": "lower V"}
ANT_COLOR = {"rv1_coarse": "#4c78a8", "rv2_coarse": "#d95f02"}
PLANETARY_SCIENCE_BASELINE_SOURCES = {"earth", "jupiter"}
BRIGHT_SOURCE_NAMES = {"fornax_a", "cyg_a", "cas_a"}


def _read(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return read_table(path, low_memory=False, **kwargs)


def _events_for_source(source: str) -> pd.DataFrame:
    """Load the intended event table for one source.

    This avoids mixing filtered and unfiltered versions of the same moving-body
    event list. In particular, the Sun all-frequency grid should use the
    Earth-limb-excluded Sun event table, not the unfiltered Sun rows that are
    present in the broader planetary science-baseline table.
    """
    name = source.lower()
    if name == "sun":
        events = _read(SUN_EVENTS, parse_dates=["predicted_event_time"])
    elif name in PLANETARY_SCIENCE_BASELINE_SOURCES:
        events = _read(EARTH_EVENTS, parse_dates=["predicted_event_time"])
    elif name in BRIGHT_SOURCE_NAMES:
        events = _read(BRIGHT_EVENTS, parse_dates=["predicted_event_time"])
    else:
        events = pd.DataFrame()
    if events.empty or "source_name" not in events.columns:
        return events
    return events[events["source_name"].astype(str).str.lower().eq(name)].copy()


def _make_groups(clean: pd.DataFrame) -> dict[tuple[int, str], tuple[pd.DataFrame, np.ndarray]]:
    groups = {}
    for (band, antenna), grp in clean.groupby(["frequency_band", "antenna"], sort=True):
        g = grp.sort_values("time").reset_index(drop=True)
        groups[(int(band), str(antenna))] = (g, datetime_ns(g["time"]))
    return groups


def _event_window(group: pd.DataFrame, group_ns: np.ndarray, event_time: pd.Timestamp, window_s: float) -> tuple[np.ndarray, np.ndarray] | None:
    event_ns = pd.Timestamp(event_time).value
    half_ns = int(float(window_s) * 1e9)
    lo = int(np.searchsorted(group_ns, event_ns - half_ns, side="left"))
    hi = int(np.searchsorted(group_ns, event_ns + half_ns, side="right"))
    if hi <= lo:
        return None
    local = group.iloc[lo:hi]
    t = (datetime_ns(local["time"]) - event_ns).astype(float) / 1e9
    y = pd.to_numeric(local["power"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(y) & (np.abs(t) <= float(window_s))
    if "is_valid" in local.columns:
        valid &= local["is_valid"].to_numpy(dtype=bool)
    if np.count_nonzero(valid) < 8:
        return None
    order = np.argsort(t[valid])
    return t[valid][order], y[valid][order]


def _normalized_points(t: np.ndarray, y: np.ndarray, inner_s: float) -> tuple[np.ndarray, np.ndarray] | None:
    side = np.abs(t) >= float(inner_s)
    if np.count_nonzero(side) < 6:
        return None
    center = float(np.nanmedian(y[side]))
    scale = robust_sigma(y[side] - center)
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.nanstd(y[side]))
    if not np.isfinite(scale) or scale <= 0:
        return None
    return t, (y - center) / scale


def _robust_standard_error(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size <= 1:
        return np.nan
    scale = robust_sigma(vals - np.nanmedian(vals))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.nanstd(vals, ddof=1))
    if not np.isfinite(scale) or scale <= 0:
        return np.nan
    return float(scale / np.sqrt(vals.size))


def collect_profiles(
    clean: pd.DataFrame,
    events: pd.DataFrame,
    source: str,
    window_s: float,
    bin_s: float,
    inner_s: float,
    time_shift_s: float = 0.0,
) -> pd.DataFrame:
    groups = _make_groups(clean)
    work = events[events["source_name"].astype(str).str.lower().eq(source.lower())].copy()
    bins = np.arange(-float(window_s), float(window_s) + float(bin_s), float(bin_s))
    rows = []
    for _, ev in work.iterrows():
        band = int(ev["frequency_band"])
        antenna = str(ev["antenna"])
        payload = groups.get((band, antenna))
        if payload is None:
            continue
        group, group_ns = payload
        event_time = pd.Timestamp(ev["predicted_event_time"]) + pd.to_timedelta(float(time_shift_s), unit="s")
        local = _event_window(group, group_ns, event_time, window_s)
        if local is None:
            continue
        normalized = _normalized_points(local[0], local[1], inner_s)
        if normalized is None:
            continue
        t, z = normalized
        bin_idx = np.digitize(t, bins) - 1
        for idx in sorted(set(bin_idx)):
            if idx < 0 or idx >= len(bins) - 1:
                continue
            mask = bin_idx == idx
            if np.count_nonzero(mask) == 0:
                continue
            rows.append(
                {
                    "source_name": source.lower(),
                    "event_id": ev.get("event_id"),
                    "event_type": str(ev["event_type"]),
                    "frequency_band": band,
                    "frequency_mhz": float(ev["frequency_mhz"]),
                    "antenna": antenna,
                    "time_shift_s": float(time_shift_s),
                    "t_bin_sec": float(0.5 * (bins[idx] + bins[idx + 1])),
                    "z_power": float(np.nanmedian(z[mask])),
                    "n_samples": int(np.count_nonzero(mask)),
                }
            )
    return pd.DataFrame.from_records(rows)


def summarize_profiles(points: pd.DataFrame) -> pd.DataFrame:
    if points.empty:
        return points
    rows = []
    by = ["source_name", "event_type", "frequency_band", "frequency_mhz", "antenna", "t_bin_sec"]
    for keys, grp in points.groupby(by, sort=True, dropna=False):
        vals = pd.to_numeric(grp["z_power"], errors="coerce")
        rows.append(
            {
                **dict(zip(by, keys)),
                "median_z_power": float(np.nanmedian(vals)),
                "median_z_power_err": _robust_standard_error(vals),
                "n_events": int(grp["event_id"].nunique()),
                "n_points": int(len(grp)),
            }
        )
    return pd.DataFrame.from_records(rows)


def plot_grid(summary: pd.DataFrame, source: str, window_s: float, out_dir: Path) -> Path:
    freqs = sorted(summary["frequency_mhz"].dropna().unique())
    event_types = ["disappearance", "reappearance"]
    fig, axes = plt.subplots(len(freqs), 2, figsize=(12, max(10, 1.35 * len(freqs))), sharex=True, sharey=False)
    if len(freqs) == 1:
        axes = np.asarray([axes])
    for i, freq in enumerate(freqs):
        for j, event_type in enumerate(event_types):
            ax = axes[i, j]
            sub = summary[
                np.isclose(summary["frequency_mhz"], float(freq))
                & summary["event_type"].astype(str).eq(event_type)
            ].copy()
            for antenna, grp in sub.groupby("antenna", sort=True):
                grp = grp.sort_values("t_bin_sec")
                ax.errorbar(
                    grp["t_bin_sec"],
                    grp["median_z_power"],
                    yerr=grp["median_z_power_err"],
                    marker="o",
                    markersize=2.5,
                    linewidth=1.2,
                    elinewidth=0.65,
                    capsize=1.3,
                    alpha=0.9,
                    color=ANT_COLOR.get(str(antenna)),
                    ecolor=ANT_COLOR.get(str(antenna)),
                    label=ANT_LABEL.get(str(antenna), str(antenna)),
                )
            ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
            ax.axhline(0, color="0.6", linewidth=0.7)
            ax.set_title(f"{freq:.2f} MHz {event_type}", fontsize=9)
            if j == 0:
                ax.set_ylabel("normalized power")
            if i == len(freqs) - 1:
                ax.set_xlabel("seconds from predicted event")
            if i == 0 and j == 1:
                ax.legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle(
        f"{source}: all-frequency normalized profiles, {int(window_s)} s window\n"
        "Disappearance should tend high-before/low-after; reappearance low-before/high-after for a positive occulted source.",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    path = out_dir / f"{source}_all_frequency_profile_grid_{int(window_s)}s.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report(out_dir: Path, paths: list[Path], window_s: float, bin_s: float, time_shift_s: float) -> None:
    lines = [
        "# All-Frequency Occultation Profile Grids",
        "",
        "These plots show every frequency on the same figure for each source.",
        "",
        "How to read them:",
        "",
        "- x-axis is seconds from predicted disappearance/reappearance;",
        "- y-axis is local normalized raw power, with no trendline removal;",
        "- each row is a frequency;",
        "- columns separate disappearance and reappearance;",
        "- blue is upper V and orange is lower V;",
        "- vertical error bars are robust event-to-event standard errors in each time bin;",
        "- a source-like disappearance should look higher before the event and lower after;",
        "- a source-like reappearance should look lower before the event and higher after;",
        "- narrow one-bin excursions should be treated as spike-like, even if an SNR metric is high.",
        "",
        f"Window: {window_s:.0f} s. Bin size: {bin_s:.0f} s.",
        f"Event-time shift applied before extracting windows: {time_shift_s:.0f} s.",
        "",
        "Generated plots:",
        "",
    ]
    lines.extend(f"- `{path.name}`" for path in paths)
    (out_dir / "all_frequency_profile_grid_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(ROOT / "outputs/all_frequency_profile_grids_v1"))
    parser.add_argument("--sources", default="earth,sun,fornax_a,cyg_a,cas_a")
    parser.add_argument("--window-s", type=float, default=900.0)
    parser.add_argument("--bin-s", type=float, default=60.0)
    parser.add_argument("--inner-s", type=float, default=15.0)
    parser.add_argument("--time-shift-s", type=float, default=0.0)
    args = parser.parse_args()
    out_dir = ensure_dir(args.out_dir)
    sources = [x.strip().lower() for x in str(args.sources).split(",") if x.strip()]
    write_json(
        out_dir / "run_config.json",
        {
            "sources": sources,
            "window_s": float(args.window_s),
            "bin_s": float(args.bin_s),
            "inner_s": float(args.inner_s),
            "time_shift_s": float(args.time_shift_s),
            "software_versions": software_versions(),
        },
    )
    clean = _read(CLEAN, parse_dates=["time"])
    paths = []
    event_status = []
    for source in sources:
        events = _events_for_source(source)
        event_status.append(
            {
                "source_name": source,
                "n_event_rows": int(len(events)),
                "event_table": (
                    str(SUN_EVENTS.relative_to(ROOT))
                    if source == "sun"
                    else str(EARTH_EVENTS.relative_to(ROOT))
                    if source in PLANETARY_SCIENCE_BASELINE_SOURCES
                    else str(BRIGHT_EVENTS.relative_to(ROOT))
                    if source in BRIGHT_SOURCE_NAMES
                    else ""
                ),
                "limb_exclusion_deg": (
                    float(pd.to_numeric(events["limb_exclusion_deg"], errors="coerce").dropna().iloc[0])
                    if "limb_exclusion_deg" in events.columns
                    and not pd.to_numeric(events["limb_exclusion_deg"], errors="coerce").dropna().empty
                    else np.nan
                ),
            }
        )
        points = collect_profiles(clean, events, source, args.window_s, args.bin_s, args.inner_s, args.time_shift_s)
        points.to_csv(out_dir / f"{source}_all_frequency_profile_points_{int(args.window_s)}s.csv", index=False)
        summary = summarize_profiles(points)
        summary.to_csv(out_dir / f"{source}_all_frequency_profile_summary_{int(args.window_s)}s.csv", index=False)
        if not summary.empty:
            paths.append(plot_grid(summary, source, args.window_s, out_dir))
            print(paths[-1])
    pd.DataFrame(event_status).to_csv(out_dir / "all_frequency_profile_event_inputs.csv", index=False)
    write_report(out_dir, paths, args.window_s, args.bin_s, args.time_shift_s)


if __name__ == "__main__":
    main()
