#!/usr/bin/env python
"""Build normalized-power source/control grids for low-frequency occultation checks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.util import ensure_dir, robust_sigma, software_versions, write_json  # noqa: E402

from build_raw_ecliptic_control_visuals import (  # noqa: E402
    ANTENNA,
    ANTENNA_LABEL,
    FREQS_MHZ,
    FREQ_TO_BAND,
    GROUP_COLORS,
    GROUP_ORDER,
    EVENT_TYPES,
    _all_events,
    _collect_raw_samples,
    _format_freq,
    _load_clean_subset,
    _panel_ylim,
)


def _normalize_samples(samples: pd.DataFrame, inner_s: float) -> pd.DataFrame:
    """Attach per-event local z-normalized power.

    This is not a detection statistic. It only puts event windows on comparable
    vertical scales for visual inspection.
    """
    rows = []
    for _, grp in samples.groupby("event_uid", sort=False):
        t = pd.to_numeric(grp["t_rel_sec"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(grp["raw_power"], errors="coerce").to_numpy(dtype=float)
        side = np.isfinite(t) & np.isfinite(y) & (np.abs(t) >= float(inner_s))
        if np.count_nonzero(side) < 6:
            continue
        center = float(np.nanmedian(y[side]))
        scale = robust_sigma(y[side] - center)
        if not np.isfinite(scale) or scale <= 0:
            scale = float(np.nanstd(y[side], ddof=1))
        if not np.isfinite(scale) or scale <= 0:
            continue
        out = grp.copy()
        out["normalization_center_raw_power"] = center
        out["normalization_scale_raw_power"] = scale
        out["normalized_power"] = (pd.to_numeric(out["raw_power"], errors="coerce") - center) / scale
        rows.append(out)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _binned_normalized_summary(samples: pd.DataFrame) -> pd.DataFrame:
    if samples.empty:
        return samples
    rows = []
    by = ["plot_group", "frequency_band", "frequency_mhz", "event_type", "t_bin_sec"]
    for keys, grp in samples.groupby(by, sort=True):
        vals = pd.to_numeric(grp["normalized_power"], errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue
        rows.append(
            {
                **dict(zip(by, keys)),
                "median_normalized_power": float(np.nanmedian(vals)),
                "q25_normalized_power": float(np.nanpercentile(vals, 25)),
                "q75_normalized_power": float(np.nanpercentile(vals, 75)),
                "n_samples": int(vals.size),
                "n_events": int(grp["event_uid"].nunique()),
                "n_controls_or_sources": int(grp["control_name_for_plot"].nunique()),
            }
        )
    return pd.DataFrame.from_records(rows)


def _plot_frequency_grid(samples: pd.DataFrame, summary: pd.DataFrame, freq: float, out_dir: Path) -> Path | None:
    sub_samples = samples[np.isclose(samples["frequency_mhz"], freq)].copy()
    sub_summary = summary[np.isclose(summary["frequency_mhz"], freq)].copy()
    if sub_samples.empty or sub_summary.empty:
        return None
    fig, axes = plt.subplots(len(GROUP_ORDER), 2, figsize=(13.5, 2.15 * len(GROUP_ORDER)), sharex=True)
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
                ax.plot(
                    raw["t_rel_sec"] / 60.0,
                    raw["normalized_power"],
                    ".",
                    color="0.25",
                    alpha=0.018 if len(raw) > 20_000 else 0.035,
                    markersize=1.25 if len(raw) > 20_000 else 1.7,
                    rasterized=True,
                )
                color = GROUP_COLORS.get(group, "black")
                x = med["t_bin_sec"].to_numpy(dtype=float) / 60.0
                ax.fill_between(
                    x,
                    med["q25_normalized_power"].to_numpy(dtype=float),
                    med["q75_normalized_power"].to_numpy(dtype=float),
                    color=color,
                    alpha=0.16,
                    linewidth=0,
                )
                ax.plot(x, med["median_normalized_power"], color=color, lw=1.8)
                ylim = _panel_ylim(raw["normalized_power"])
                if ylim is not None:
                    ax.set_ylim(*ylim)
                if row == 0:
                    ax.set_title(event_type)
                if col == 0:
                    events = int(raw["event_uid"].nunique())
                    controls = int(raw["control_name_for_plot"].nunique())
                    max_bin_events = int(med["n_events"].max())
                    ax.set_ylabel(f"{group}\nnormalized power")
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
            ax.axhline(0.0, color="0.45", lw=0.6)
            ax.grid(alpha=0.18)
            if row == len(GROUP_ORDER) - 1:
                ax.set_xlabel("minutes from predicted event")
    fig.suptitle(
        f"{freq:.2f} MHz {ANTENNA_LABEL}: locally normalized power around predicted occultations\n"
        "Dots are normalized samples; colored line is per-bin median; shaded band is the middle 50%.",
        y=0.997,
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.968])
    path = out_dir / f"normalized_power_{_format_freq(freq)}mhz_lower_v_source_control_grid.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _write_report(out_dir: Path, paths: list[Path], samples: pd.DataFrame, args: argparse.Namespace) -> Path:
    counts = (
        samples.groupby(["plot_group", "frequency_mhz", "event_type"], as_index=False)
        .agg(n_events=("event_uid", "nunique"), n_samples=("normalized_power", "size"), n_tracks=("control_name_for_plot", "nunique"))
        .sort_values(["plot_group", "frequency_mhz", "event_type"])
    )
    lines = [
        "# Normalized Ecliptic-Control Visual Audit",
        "",
        "These source-control grids show locally normalized power, not raw power and not a detection score.",
        "",
        "Normalization for each individual event window:",
        "",
        "    normalized_power = (power - median(sideband power)) / robust_sigma(sideband power)",
        "",
        f"The sideband is all telemetry-valid samples with |dt| >= {args.inner_s:g} s inside the plotted window.",
        "This removes arbitrary vertical scale differences between events and sources while preserving the before/after shape.",
        "",
        "What this does not do:",
        "",
        "- no baseline slope is fit or removed;",
        "- no event-type sign convention is applied;",
        "- no source-like contrast or SNR is computed;",
        "- no per-source decision is made from these plots.",
        "",
        "The fixed ecliptic controls are read from the Sun/Earth limb-vetoed control event table:",
        "",
        "    outputs/ecliptic_control_points_sun_earth_limb5_v1/ecliptic_control_predicted_events.csv",
        "",
        "That table rejects fixed-control events where either the Sun or Earth is within 5 deg of the lunar limb.",
        "",
        "## What Fixed Near-Ecliptic Controls Mean",
        "",
        "`Fixed near ecliptic (+/-10 deg)` controls are artificial fixed sky positions placed at ecliptic latitude +10 deg and -10 deg, with ecliptic longitudes spaced every 60 deg. They are transformed into FK4/B1950 and passed through the same lunar-limb event prediction as real fixed sources.",
        "",
        "They are not moving like the Sun or Earth. They test whether simply being close to the ecliptic plane is enough to create similar stacked behavior. If Earth/Sun-like behavior appears only in moving tracks and not in these fixed near-ecliptic controls, then ecliptic latitude alone is not the explanation.",
        "",
        "## Event Counts",
        "",
        counts.to_string(index=False),
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
            f"- antenna: {ANTENNA} ({ANTENNA_LABEL})",
            f"- window_s: {args.window_s}",
            f"- inner_s: {args.inner_s}",
            f"- max_events_per_group: {args.max_events_per_group} (0 means all available events)",
            "- software versions saved in `run_config.json`.",
            "",
        ]
    )
    path = out_dir / "normalized_ecliptic_control_visual_audit.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(ROOT / "outputs/normalized_ecliptic_control_visuals_v1"))
    parser.add_argument("--window-s", type=float, default=900.0)
    parser.add_argument("--inner-s", type=float, default=15.0)
    parser.add_argument("--max-events-per-group", type=int, default=0, help="0 or negative means use all available events.")
    parser.add_argument("--write-samples", action="store_true")
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
            "inner_s": float(args.inner_s),
            "max_events_per_group": int(args.max_events_per_group),
            "write_samples": bool(args.write_samples),
            "software_versions": software_versions(),
        },
    )

    clean = _load_clean_subset(bands)
    events = _all_events(bands)
    raw_samples = _collect_raw_samples(clean, events, args.window_s, args.max_events_per_group)
    samples = _normalize_samples(raw_samples, args.inner_s)
    summary = _binned_normalized_summary(samples)
    if args.write_samples:
        samples.to_csv(out_dir / "normalized_power_event_samples.csv", index=False)
    summary.to_csv(out_dir / "normalized_power_binned_summary.csv", index=False)

    paths: list[Path] = []
    for freq in FREQS_MHZ:
        path = _plot_frequency_grid(samples, summary, freq, out_dir)
        if path is not None:
            paths.append(path)
    report = _write_report(out_dir, paths, samples, args)
    print(report)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
