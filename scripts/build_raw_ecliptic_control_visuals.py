#!/usr/bin/env python
"""Build raw-power visual controls for Earth/Sun low-frequency behavior."""

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

from rylevonberg.constants import FREQUENCY_MAP_MHZ  # noqa: E402
from rylevonberg.util import datetime_ns, ensure_dir, software_versions, write_json  # noqa: E402


CLEAN = ROOT / "outputs/control_survey_earth_sun_postnov1974_v2/01_ingest/cleaned_timeseries.csv"
SUN_EVENTS = ROOT / "outputs/pipeline_confidence_audit_v2/sun_audit_input_events.csv"
PLANET_EVENTS = ROOT / "outputs/planetary_confirmation_survey_science_baseline_v2/events/all_planet_predicted_events.csv"
BRIGHT_EVENTS = ROOT / "outputs/control_survey_bright_sources_postnov1974_v1/02_events/predicted_events.csv"
ECLIPTIC_EVENTS = ROOT / "outputs/ecliptic_control_points_sun_earth_limb5_v1/ecliptic_control_predicted_events.csv"

ANTENNA = "rv2_coarse"
ANTENNA_LABEL = "lower V"
FREQS_MHZ = [0.45, 0.70, 0.90, 1.31, 2.20]
FREQ_TO_BAND = {v: k for k, v in FREQUENCY_MAP_MHZ.items()}
EVENT_TYPES = ["disappearance", "reappearance"]

GROUP_ORDER = [
    "Earth moving track",
    "Sun moving track",
    "Fixed ecliptic plane",
    "Fixed near ecliptic (+/-10 deg)",
    "Fixed off ecliptic (+/-60 deg)",
    "Fornax A fixed source",
]
GROUP_COLORS = {
    "Earth moving track": "#1f77b4",
    "Sun moving track": "#d62728",
    "Fixed ecliptic plane": "#2ca02c",
    "Fixed near ecliptic (+/-10 deg)": "#ff7f0e",
    "Fixed off ecliptic (+/-60 deg)": "#c51b7d",
    "Fornax A fixed source": "#9467bd",
}


def _read(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return read_table(path, low_memory=False, **kwargs)


def _load_clean_subset(bands: set[int]) -> pd.DataFrame:
    """Load only lower-V rows needed for these plots."""
    frames = []
    usecols = ["time", "frequency_band", "antenna", "power", "is_valid"]
    for chunk in read_table(CLEAN, usecols=usecols, chunksize=750_000, low_memory=False):
        mask = chunk["antenna"].astype(str).eq(ANTENNA) & chunk["frequency_band"].astype(int).isin(bands)
        if not mask.any():
            continue
        sub = chunk.loc[mask].copy()
        sub["time"] = pd.to_datetime(sub["time"], errors="coerce")
        sub["power"] = pd.to_numeric(sub["power"], errors="coerce")
        if "is_valid" in sub:
            if sub["is_valid"].dtype != bool:
                sub["is_valid"] = sub["is_valid"].astype(str).str.lower().isin(["true", "1", "yes"])
        else:
            sub["is_valid"] = np.isfinite(sub["power"])
        sub = sub[sub["time"].notna()]
        frames.append(sub)
    if not frames:
        return pd.DataFrame(columns=usecols)
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["frequency_band", "antenna", "time"]).reset_index(drop=True)


def _channel_groups(clean: pd.DataFrame) -> dict[int, tuple[pd.DataFrame, np.ndarray]]:
    groups = {}
    for band, grp in clean.groupby("frequency_band", sort=True):
        g = grp.sort_values("time").reset_index(drop=True)
        groups[int(band)] = (g, datetime_ns(g["time"]))
    return groups


def _source_events(name: str, bands: set[int]) -> pd.DataFrame:
    if name == "earth":
        events = _read(PLANET_EVENTS, parse_dates=["predicted_event_time"])
        events = events[events["source_name"].astype(str).str.lower().eq("earth")]
        group_label = "Earth moving track"
    elif name == "sun":
        events = _read(SUN_EVENTS, parse_dates=["predicted_event_time"])
        events = events[events["source_name"].astype(str).str.lower().eq("sun")]
        group_label = "Sun moving track"
    elif name == "fornax_a":
        events = _read(BRIGHT_EVENTS, parse_dates=["predicted_event_time"])
        events = events[events["source_name"].astype(str).str.lower().eq("fornax_a")]
        group_label = "Fornax A fixed source"
    else:
        raise ValueError(name)
    events = events[
        events["antenna"].astype(str).eq(ANTENNA)
        & events["frequency_band"].astype(int).isin(bands)
        & events["event_type"].astype(str).isin(EVENT_TYPES)
    ].copy()
    events["plot_group"] = group_label
    events["control_name_for_plot"] = events["source_name"].astype(str)
    return events


def _ecliptic_events(bands: set[int]) -> pd.DataFrame:
    events = _read(ECLIPTIC_EVENTS, parse_dates=["predicted_event_time"])
    if events.empty:
        return events
    events = events[
        events["antenna"].astype(str).eq(ANTENNA)
        & events["frequency_band"].astype(int).isin(bands)
        & events["event_type"].astype(str).isin(EVENT_TYPES)
    ].copy()
    label_map = {
        "ecliptic_plane": "Fixed ecliptic plane",
        "near_ecliptic": "Fixed near ecliptic (+/-10 deg)",
        "off_ecliptic": "Fixed off ecliptic (+/-60 deg)",
    }
    events["plot_group"] = events["control_class"].astype(str).map(label_map)
    events["control_name_for_plot"] = events["source_name"].astype(str)
    return events[events["plot_group"].notna()].copy()


def _all_events(bands: set[int]) -> pd.DataFrame:
    frames = [
        _source_events("earth", bands),
        _source_events("sun", bands),
        _source_events("fornax_a", bands),
        _ecliptic_events(bands),
    ]
    events = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    events["event_uid"] = (
        events["plot_group"].astype(str)
        + "|"
        + events["control_name_for_plot"].astype(str)
        + "|"
        + events["event_id"].astype(str)
        + "|"
        + events["frequency_band"].astype(str)
        + "|"
        + events["event_type"].astype(str)
    )
    return events


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
    keep = np.isfinite(y) & (np.abs(t) <= float(window_s))
    if "is_valid" in local.columns:
        keep &= local["is_valid"].to_numpy(dtype=bool)
    if np.count_nonzero(keep) < 4:
        return None
    order = np.argsort(t[keep])
    return t[keep][order], y[keep][order]


def _collect_raw_samples(clean: pd.DataFrame, events: pd.DataFrame, window_s: float, max_events_per_group: int) -> pd.DataFrame:
    groups = _channel_groups(clean)
    rows = []
    rng = np.random.default_rng(12345)
    work = events.copy()
    sampled_parts = []
    by = ["plot_group", "frequency_band", "event_type"]
    for _, grp in work.groupby(by, sort=False):
        event_uids = np.array(sorted(grp["event_uid"].dropna().unique()))
        if max_events_per_group > 0 and event_uids.size > max_events_per_group:
            event_uids = np.sort(rng.choice(event_uids, size=max_events_per_group, replace=False))
        sampled_parts.append(grp[grp["event_uid"].isin(event_uids)])
    work = pd.concat(sampled_parts, ignore_index=True) if sampled_parts else pd.DataFrame()
    for ev in work.itertuples(index=False):
        band = int(ev.frequency_band)
        payload = groups.get(band)
        if payload is None:
            continue
        local = _event_window(payload[0], payload[1], pd.Timestamp(ev.predicted_event_time), window_s)
        if local is None:
            continue
        t, y = local
        freq = FREQUENCY_MAP_MHZ[band]
        t_bin = np.round(t / 60.0) * 60.0
        for tt, yy, tb in zip(t, y, t_bin):
            rows.append(
                {
                    "plot_group": ev.plot_group,
                    "control_name_for_plot": ev.control_name_for_plot,
                    "event_uid": ev.event_uid,
                    "event_type": ev.event_type,
                    "predicted_event_time": ev.predicted_event_time,
                    "frequency_band": band,
                    "frequency_mhz": freq,
                    "antenna": ANTENNA,
                    "t_rel_sec": float(tt),
                    "t_bin_sec": float(tb),
                    "raw_power": float(yy),
                }
            )
    return pd.DataFrame.from_records(rows)


def _binned_raw_summary(samples: pd.DataFrame) -> pd.DataFrame:
    if samples.empty:
        return samples
    rows = []
    by = ["plot_group", "frequency_band", "frequency_mhz", "event_type", "t_bin_sec"]
    for keys, grp in samples.groupby(by, sort=True):
        vals = pd.to_numeric(grp["raw_power"], errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue
        rows.append(
            {
                **dict(zip(by, keys)),
                "median_raw_power": float(np.nanmedian(vals)),
                "q25_raw_power": float(np.nanpercentile(vals, 25)),
                "q75_raw_power": float(np.nanpercentile(vals, 75)),
                "n_raw_samples": int(vals.size),
                "n_events": int(grp["event_uid"].nunique()),
                "n_controls_or_sources": int(grp["control_name_for_plot"].nunique()),
            }
        )
    return pd.DataFrame.from_records(rows)


def _format_freq(freq: float) -> str:
    return f"{freq:.2f}".replace(".", "p")


def _panel_ylim(vals: pd.Series) -> tuple[float, float] | None:
    arr = pd.to_numeric(vals, errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size < 2:
        return None
    lo, hi = np.nanpercentile(arr, [2, 98])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return None
    pad = 0.08 * (hi - lo)
    return float(lo - pad), float(hi + pad)


def _plot_frequency_grid(samples: pd.DataFrame, summary: pd.DataFrame, freq: float, out_dir: Path) -> Path | None:
    sub_samples = samples[np.isclose(samples["frequency_mhz"], freq)].copy()
    sub_summary = summary[np.isclose(summary["frequency_mhz"], freq)].copy()
    if sub_samples.empty or sub_summary.empty:
        return None
    fig, axes = plt.subplots(len(GROUP_ORDER), 2, figsize=(13.5, 2.15 * len(GROUP_ORDER)), sharex=True)
    rng = np.random.default_rng(20240523)
    for row, group in enumerate(GROUP_ORDER):
        for col, event_type in enumerate(EVENT_TYPES):
            ax = axes[row, col]
            raw = sub_samples[
                sub_samples["plot_group"].eq(group)
                & sub_samples["event_type"].astype(str).eq(event_type)
            ].copy()
            med = sub_summary[
                sub_summary["plot_group"].eq(group)
                & sub_summary["event_type"].astype(str).eq(event_type)
            ].sort_values("t_bin_sec")
            if raw.empty or med.empty:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            else:
                if len(raw) > 2200:
                    raw_plot = raw.iloc[np.sort(rng.choice(len(raw), size=2200, replace=False))]
                else:
                    raw_plot = raw
                ax.plot(
                    raw_plot["t_rel_sec"] / 60.0,
                    raw_plot["raw_power"],
                    ".",
                    color="0.25",
                    alpha=0.055,
                    markersize=2.4,
                    rasterized=True,
                )
                color = GROUP_COLORS.get(group, "black")
                x = med["t_bin_sec"].to_numpy(dtype=float) / 60.0
                ax.fill_between(
                    x,
                    med["q25_raw_power"].to_numpy(dtype=float),
                    med["q75_raw_power"].to_numpy(dtype=float),
                    color=color,
                    alpha=0.16,
                    linewidth=0,
                )
                ax.plot(x, med["median_raw_power"], color=color, lw=1.8)
                ylim = _panel_ylim(raw["raw_power"])
                if ylim is not None:
                    ax.set_ylim(*ylim)
                title = f"{event_type}"
                if row == 0:
                    ax.set_title(title)
                if col == 0:
                    events = int(raw["event_uid"].nunique())
                    controls = int(raw["control_name_for_plot"].nunique())
                    max_bin_events = int(med["n_events"].max())
                    ax.set_ylabel(f"{group}\nraw power")
                    ax.text(
                        0.01,
                        0.96,
                        f"{events} events, {controls} track(s)\nmax {max_bin_events} events/bin",
                        ha="left",
                        va="top",
                        transform=ax.transAxes,
                        fontsize=7,
                        color="0.25",
                    )
            ax.axvline(0.0, color="black", lw=0.85, ls="--")
            ax.grid(alpha=0.18)
            if row == len(GROUP_ORDER) - 1:
                ax.set_xlabel("minutes from predicted event")
    fig.suptitle(
        f"{freq:.2f} MHz {ANTENNA_LABEL}: telemetry-valid raw power around predicted occultations\n"
        "Black dots are raw samples; colored line is per-bin median raw power; shaded band is the middle 50% of raw samples.",
        y=0.997,
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.968])
    path = out_dir / f"raw_power_{_format_freq(freq)}mhz_lower_v_source_control_grid.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _plot_ecliptic_longitude_grid(samples: pd.DataFrame, freq: float, out_dir: Path) -> Path | None:
    sub = samples[
        np.isclose(samples["frequency_mhz"], freq)
        & samples["plot_group"].eq("Fixed ecliptic plane")
    ].copy()
    if sub.empty:
        return None
    sub["lon_label"] = sub["control_name_for_plot"].str.extract(r"lon(\d{3})", expand=False)
    lon_labels = sorted([x for x in sub["lon_label"].dropna().unique()])
    if not lon_labels:
        return None
    rows = int(np.ceil(len(lon_labels) / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(13.5, 2.6 * rows), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, lon in zip(axes, lon_labels):
        lon_sub = sub[sub["lon_label"].eq(lon)]
        for event_type, color in [("disappearance", "#4c78a8"), ("reappearance", "#d95f02")]:
            g = lon_sub[lon_sub["event_type"].astype(str).eq(event_type)]
            if g.empty:
                continue
            med = (
                g.groupby("t_bin_sec", as_index=False)
                .agg(median_raw_power=("raw_power", "median"), n_events=("event_uid", "nunique"))
                .sort_values("t_bin_sec")
            )
            ax.plot(med["t_bin_sec"] / 60.0, med["median_raw_power"], lw=1.3, color=color, label=event_type)
        ax.axvline(0.0, color="black", lw=0.75, ls="--")
        ax.set_title(f"ecliptic lon {int(lon):03d} deg")
        ax.grid(alpha=0.18)
    for ax in axes[len(lon_labels) :]:
        ax.axis("off")
    axes[0].legend(frameon=False, fontsize=8)
    for ax in axes[-3:]:
        ax.set_xlabel("minutes from predicted event")
    fig.suptitle(
        f"Fixed ecliptic-plane controls by longitude, {freq:.2f} MHz {ANTENNA_LABEL}\n"
        "Lines are medians of telemetry-valid raw power; no baseline subtraction or normalization.",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = out_dir / f"raw_power_{_format_freq(freq)}mhz_fixed_ecliptic_longitude_grid.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _plot_single_event_examples(samples: pd.DataFrame, freq: float, out_dir: Path) -> Path | None:
    groups = ["Earth moving track", "Sun moving track", "Fixed ecliptic plane", "Fornax A fixed source"]
    sub = samples[np.isclose(samples["frequency_mhz"], freq) & samples["plot_group"].isin(groups)].copy()
    if sub.empty:
        return None
    fig, axes = plt.subplots(len(groups), 2, figsize=(13.5, 2.5 * len(groups)), sharex=True)
    rng = np.random.default_rng(331)
    for row, group in enumerate(groups):
        for col, event_type in enumerate(EVENT_TYPES):
            ax = axes[row, col]
            part = sub[sub["plot_group"].eq(group) & sub["event_type"].astype(str).eq(event_type)]
            uids = np.array(sorted(part["event_uid"].unique()))
            if uids.size == 0:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            else:
                chosen = uids if uids.size <= 12 else np.sort(rng.choice(uids, size=12, replace=False))
                for uid in chosen:
                    g = part[part["event_uid"].eq(uid)].sort_values("t_rel_sec")
                    ax.plot(g["t_rel_sec"] / 60.0, g["raw_power"], "-", lw=0.8, alpha=0.45)
                ax.text(
                    0.01,
                    0.95,
                    f"{len(chosen)} individual events",
                    ha="left",
                    va="top",
                    transform=ax.transAxes,
                    fontsize=7,
                    color="0.25",
                )
            ax.axvline(0.0, color="black", lw=0.85, ls="--")
            ax.grid(alpha=0.18)
            if row == 0:
                ax.set_title(event_type)
            if col == 0:
                ax.set_ylabel(f"{group}\nraw power")
            if row == len(groups) - 1:
                ax.set_xlabel("minutes from predicted event")
    fig.suptitle(
        f"Individual raw event traces, {freq:.2f} MHz {ANTENNA_LABEL}\n"
        "These are actual event windows, not fitted or normalized profiles.",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = out_dir / f"raw_power_{_format_freq(freq)}mhz_individual_event_examples.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _write_report(out_dir: Path, paths: list[Path], samples: pd.DataFrame, summary: pd.DataFrame, args: argparse.Namespace) -> Path:
    counts = (
        samples.groupby(["plot_group", "frequency_mhz", "event_type"], as_index=False)
        .agg(n_events=("event_uid", "nunique"), n_raw_samples=("raw_power", "size"), n_tracks=("control_name_for_plot", "nunique"))
        .sort_values(["plot_group", "frequency_mhz", "event_type"])
    )
    lines = [
        "# Raw Ecliptic-Control Visual Audit",
        "",
        "This output intentionally avoids source-like contrast, SNR, or score columns.",
        "The purpose is to inspect the raw telemetry behavior that generated the previous claims.",
        "",
        "## What Is Plotted",
        "",
        "- Antenna: lower V (`rv2_coarse`).",
        "- Frequencies: " + ", ".join(f"{f:.2f} MHz" for f in FREQS_MHZ) + ".",
        "- x-axis: minutes from predicted lunar-limb event.",
        "- y-axis: original Ryle-Vonberg power values.",
        "- Black dots: deterministic subset of telemetry-valid raw samples.",
        "- Colored line: median raw power in each 60 s bin, drawn only as a visual guide.",
        "- Shaded region: middle 50% of raw samples in that bin.",
        "- No baseline subtraction, trendline removal, per-event normalization, or source-sign transformation is applied.",
        "",
        "Telemetry-flagged invalid samples are excluded so single bad jumps do not dominate the y-axis.",
        "Panel y-limits are clipped to the local 2nd-98th percentiles for display only.",
        "",
        "## Event Counts In These Visuals",
        "",
        counts.to_string(index=False),
        "",
        "## How To Read The Plots",
        "",
        "These plots should be used to check shape, not to quote a detection statistic.",
        "If Earth/Sun behavior were a generic fixed-ecliptic effect, the fixed ecliptic-plane rows should visually resemble Earth/Sun rows at the same frequency and event type.",
        "If the behavior is tied to moving-body time selection or observing geometry, Earth/Sun can look similar to each other while fixed ecliptic rows look mixed or longitude-dependent.",
        "",
        "## Visual Claims From The Current Plots",
        "",
        "- At 0.90 and 1.31 MHz, the Earth and Sun raw medians both rise through disappearance windows and fall through reappearance windows. This is visible before any baseline subtraction, trendline removal, normalization, or sign convention.",
        "- Fornax A at the same frequencies shows the opposite visual behavior: the raw median falls through disappearance and rises through reappearance. That is the cleaner fixed-source occultation pattern.",
        "- Fixed ecliptic-plane controls sometimes resemble part of the Earth/Sun behavior, especially at 0.90 and 1.31 MHz, but the longitude panels show this is not uniform around the ecliptic.",
        "- At 2.20 MHz, the fixed ecliptic controls are much flatter than the Sun raw profile, so fixed ecliptic latitude alone is not a complete explanation.",
        "- At 0.45 MHz, the raw plots are dominated by broad scatter and occasional high-power excursions. The median curves are less visually diagnostic there than at 0.90 and 1.31 MHz.",
        "",
        "The current plot-based conclusion is therefore: the Earth/Sun reversal is not created by baseline subtraction, but the assumption that every fixed point near the ecliptic should behave the same way is too strong. A moving-track/time-selection/background-coupling explanation remains more plausible than a generic ecliptic-coordinate explanation.",
        "",
        "## Generated Figures",
        "",
    ]
    lines.extend(f"- `{path}`" for path in paths)
    lines.extend(
        [
            "",
            "## Run Configuration",
            "",
            f"- window_s: {args.window_s}",
            f"- max_events_per_group: {args.max_events_per_group}",
            "- software versions saved in `run_config.json`.",
            "",
        ]
    )
    path = out_dir / "raw_ecliptic_control_visual_audit.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(ROOT / "outputs/raw_ecliptic_control_visuals_v1"))
    parser.add_argument("--window-s", type=float, default=900.0)
    parser.add_argument("--max-events-per-group", type=int, default=260)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    bands = {FREQ_TO_BAND[f] for f in FREQS_MHZ}
    write_json(
        out_dir / "run_config.json",
        {
            "antenna": ANTENNA,
            "antenna_label": ANTENNA_LABEL,
            "frequencies_mhz": FREQS_MHZ,
            "window_s": float(args.window_s),
            "max_events_per_group": int(args.max_events_per_group),
            "software_versions": software_versions(),
        },
    )

    clean = _load_clean_subset(bands)
    events = _all_events(bands)
    samples = _collect_raw_samples(clean, events, args.window_s, args.max_events_per_group)
    summary = _binned_raw_summary(samples)
    samples.to_csv(out_dir / "raw_power_event_samples.csv", index=False)
    summary.to_csv(out_dir / "raw_power_binned_summary.csv", index=False)

    paths: list[Path] = []
    for freq in FREQS_MHZ:
        path = _plot_frequency_grid(samples, summary, freq, out_dir)
        if path is not None:
            paths.append(path)
    for freq in [0.90, 1.31]:
        path = _plot_ecliptic_longitude_grid(samples, freq, out_dir)
        if path is not None:
            paths.append(path)
    for freq in [0.90, 1.31]:
        path = _plot_single_event_examples(samples, freq, out_dir)
        if path is not None:
            paths.append(path)
    report = _write_report(out_dir, paths, samples, summary, args)
    print(report)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
