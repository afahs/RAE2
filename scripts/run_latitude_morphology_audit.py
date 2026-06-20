#!/usr/bin/env python
"""Compare normalized occultation morphology across Galactic/ecliptic latitude controls."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import astropy.units as u
from astropy.coordinates import BarycentricTrueEcliptic, FK4, Galactic, SkyCoord
from astropy.time import Time
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rylevonberg.table_io import read_table

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.events import predict_events  # noqa: E402
from rylevonberg.constants import EARTH_UNIT_COLUMNS, SPACECRAFT_COLUMNS  # noqa: E402
from rylevonberg.sources import load_source_list  # noqa: E402
from rylevonberg.util import ensure_dir, software_versions, write_json  # noqa: E402
from scripts.build_all_frequency_occultation_profile_grids import CLEAN  # noqa: E402
from scripts.build_ecliptic_controls_with_sun_earth_limb_filter import _limb_exclusion_sources  # noqa: E402
from scripts.build_normalized_ecliptic_control_visuals import _binned_normalized_summary, _normalize_samples  # noqa: E402
from scripts.build_raw_ecliptic_control_visuals import (  # noqa: E402
    ANTENNA,
    ANTENNA_LABEL,
    BRIGHT_EVENTS,
    EVENT_TYPES,
    FREQS_MHZ,
    FREQ_TO_BAND,
    PLANET_EVENTS,
    SUN_EVENTS,
    _collect_raw_samples,
    _format_freq,
    _load_clean_subset,
    _panel_ylim,
    _read,
)


LOW_BANDS = [FREQ_TO_BAND[f] for f in FREQS_MHZ]
LATITUDE_COLORS = {
    "Earth moving track": "#1f77b4",
    "Sun moving track": "#d62728",
    "Galactic b=-60 deg": "#5e3c99",
    "Galactic b=-30 deg": "#8e63b0",
    "Galactic b=-10 deg": "#b2abd2",
    "Galactic b=+00 deg": "#2ca02c",
    "Galactic b=+10 deg": "#a6dba0",
    "Galactic b=+30 deg": "#fdb863",
    "Galactic b=+60 deg": "#e66101",
    "Fornax A fixed source": "#9467bd",
}
POINT_COLORS = {
    "Earth moving track": "#1f77b4",
    "Sun moving track": "#d62728",
    "tau_a": "#2ca02c",
    "vir_a": "#ff7f0e",
    "sgr_a": "#8c564b",
    "galactic_center": "#e377c2",
    "Fornax A fixed source": "#9467bd",
}


def _fk4_to_ecliptic_galactic(ra_deg: float, dec_deg: float) -> tuple[float, float, float, float]:
    coord = SkyCoord(ra=float(ra_deg) * u.deg, dec=float(dec_deg) * u.deg, frame=FK4(equinox=Time("B1950")))
    ecl = coord.transform_to(BarycentricTrueEcliptic(equinox=Time("B1950")))
    gal = coord.galactic
    return float(ecl.lon.deg), float(ecl.lat.deg), float(gal.l.deg), float(gal.b.deg)


def _galactic_to_fk4(lon_deg: float, lat_deg: float) -> tuple[float, float]:
    coord = SkyCoord(l=float(lon_deg) * u.deg, b=float(lat_deg) * u.deg, frame=Galactic())
    fk4 = coord.transform_to(FK4(equinox=Time("B1950")))
    return float(fk4.ra.deg), float(fk4.dec.deg)


def _galactic_control_sources(latitudes: list[float], lon_step_deg: float) -> pd.DataFrame:
    rows = []
    for lat in latitudes:
        for lon in np.arange(0.0, 360.0, float(lon_step_deg)):
            ra, dec = _galactic_to_fk4(lon, lat)
            ecl_lon, ecl_lat, gal_lon, gal_lat = _fk4_to_ecliptic_galactic(ra, dec)
            rows.append(
                {
                    "source_name": f"gal_lat{lat:+05.1f}_lon{lon:06.1f}".replace("+", "p").replace("-", "m").replace(".", "p"),
                    "kind": "fixed",
                    "body_name": "",
                    "ra_deg": ra,
                    "dec_deg": dec,
                    "frame": "fk4",
                    "galactic_l_deg": gal_lon,
                    "galactic_b_deg": gal_lat,
                    "ecliptic_lon_deg": ecl_lon,
                    "ecliptic_lat_deg": ecl_lat,
                    "plot_group": f"Galactic b={lat:+03.0f} deg",
                    "control_class": "galactic_latitude",
                }
            )
    return pd.DataFrame(rows)


def _load_prediction_geometry() -> pd.DataFrame:
    """Load one row per timestamp with the geometry columns needed for event prediction."""
    usecols = ["time", *SPACECRAFT_COLUMNS, *EARTH_UNIT_COLUMNS]
    frames = []
    for chunk in read_table(CLEAN, usecols=usecols, chunksize=750_000, low_memory=False):
        chunk["time"] = pd.to_datetime(chunk["time"], errors="coerce")
        chunk = chunk[chunk["time"].notna()]
        frames.append(chunk.drop_duplicates("time"))
    if not frames:
        return pd.DataFrame(columns=usecols)
    out = pd.concat(frames, ignore_index=True).drop_duplicates("time").sort_values("time").reset_index(drop=True)
    return out


def _fixed_source_coordinate_table() -> pd.DataFrame:
    src = load_source_list(ROOT / "configs/bright_sources.csv")
    fixed = src[src["kind"].astype(str).str.lower().eq("fixed")].copy()
    rows = []
    for _, row in fixed.iterrows():
        ecl_lon, ecl_lat, gal_lon, gal_lat = _fk4_to_ecliptic_galactic(row["ra_deg"], row["dec_deg"])
        rows.append(
            {
                "source_name": row["source_name"],
                "ra_fk4_deg": float(row["ra_deg"]),
                "dec_fk4_deg": float(row["dec_deg"]),
                "ecliptic_lon_deg": ecl_lon,
                "ecliptic_lat_deg": ecl_lat,
                "galactic_l_deg": gal_lon,
                "galactic_b_deg": gal_lat,
                "abs_ecliptic_lat_deg": abs(ecl_lat),
                "abs_galactic_lat_deg": abs(gal_lat),
            }
        )
    return pd.DataFrame(rows).sort_values("abs_ecliptic_lat_deg")


def _predict_or_read_controls(
    clean: pd.DataFrame,
    sources: pd.DataFrame,
    out_dir: Path,
    filename: str,
    limb_exclusion_deg: float,
) -> pd.DataFrame:
    path = out_dir / filename
    if path.exists() and path.stat().st_size > 0:
        return _read(path, parse_dates=["predicted_event_time"])
    events, _states = predict_events(
        clean,
        sources,
        target_frame="fk4",
        equinox="B1950",
        ephemeris="builtin",
        max_gap_seconds=600.0,
        prediction_cadence_seconds=300.0,
        frequencies=LOW_BANDS,
        antennas=[ANTENNA],
        limb_exclusion_sources_df=_limb_exclusion_sources(),
        limb_exclusion_deg=float(limb_exclusion_deg),
    )
    if not events.empty:
        meta_cols = [c for c in sources.columns if c not in set(events.columns) or c in {"source_name", "plot_group"}]
        events = events.merge(sources[meta_cols].drop_duplicates("source_name"), on="source_name", how="left")
    events.to_csv(path, index=False)
    return events


def _moving_and_fornax_events() -> pd.DataFrame:
    frames = []
    for source, path, label in [
        ("earth", PLANET_EVENTS, "Earth moving track"),
        ("sun", SUN_EVENTS, "Sun moving track"),
    ]:
        ev = _read(path, parse_dates=["predicted_event_time"])
        ev = ev[
            ev["source_name"].astype(str).str.lower().eq(source)
            & ev["antenna"].astype(str).eq(ANTENNA)
            & ev["frequency_band"].astype(int).isin(LOW_BANDS)
            & ev["event_type"].astype(str).isin(EVENT_TYPES)
        ].copy()
        ev["plot_group"] = label
        ev["control_name_for_plot"] = ev["source_name"].astype(str)
        frames.append(ev)
    bright = _read(BRIGHT_EVENTS, parse_dates=["predicted_event_time"])
    fornax = bright[
        bright["source_name"].astype(str).str.lower().eq("fornax_a")
        & bright["antenna"].astype(str).eq(ANTENNA)
        & bright["frequency_band"].astype(int).isin(LOW_BANDS)
        & bright["event_type"].astype(str).isin(EVENT_TYPES)
    ].copy()
    fornax["plot_group"] = "Fornax A fixed source"
    fornax["control_name_for_plot"] = "fornax_a"
    frames.append(fornax)
    out = pd.concat(frames, ignore_index=True)
    return _assign_event_uid(out)


def _assign_event_uid(events: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    out["control_name_for_plot"] = out.get("control_name_for_plot", out["source_name"]).astype(str)
    out["event_uid"] = (
        out["plot_group"].astype(str)
        + "|"
        + out["control_name_for_plot"].astype(str)
        + "|"
        + out["event_id"].astype(str)
        + "|"
        + out["frequency_band"].astype(str)
        + "|"
        + out["event_type"].astype(str)
    )
    return out


def _prepare_control_events(events: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    out["control_name_for_plot"] = out["source_name"].astype(str)
    return _assign_event_uid(out)


def _near_ecliptic_point_sources(coord: pd.DataFrame, threshold_deg: float) -> pd.DataFrame:
    src = load_source_list(ROOT / "configs/bright_sources.csv")
    names = coord[coord["abs_ecliptic_lat_deg"].le(float(threshold_deg))]["source_name"].tolist()
    out = src[src["source_name"].isin(names)].copy()
    out = out.merge(coord, on="source_name", how="left")
    out["plot_group"] = out["source_name"].astype(str)
    return out


def _prepare_point_source_events(events: pd.DataFrame, coord: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    out["plot_group"] = out["source_name"].astype(str)
    out["control_name_for_plot"] = out["source_name"].astype(str)
    out = out.merge(
        coord[["source_name", "ecliptic_lat_deg", "galactic_b_deg", "abs_ecliptic_lat_deg"]],
        on="source_name",
        how="left",
    )
    return _assign_event_uid(out)


def _summary_counts(samples: pd.DataFrame) -> pd.DataFrame:
    return (
        samples.groupby(["plot_group", "frequency_mhz", "event_type"], as_index=False)
        .agg(n_events=("event_uid", "nunique"), n_samples=("normalized_power", "size"), n_tracks=("control_name_for_plot", "nunique"))
        .sort_values(["plot_group", "frequency_mhz", "event_type"])
    )


def _plot_group_grid(
    samples: pd.DataFrame,
    summary: pd.DataFrame,
    freq: float,
    groups: list[str],
    colors: dict[str, str],
    title_prefix: str,
    out_dir: Path,
    stem: str,
) -> Path | None:
    sub_samples = samples[np.isclose(samples["frequency_mhz"], freq)].copy()
    sub_summary = summary[np.isclose(summary["frequency_mhz"], freq)].copy()
    if sub_samples.empty or sub_summary.empty:
        return None
    fig, axes = plt.subplots(len(groups), 2, figsize=(13.6, max(4.0, 2.05 * len(groups))), sharex=True)
    if len(groups) == 1:
        axes = np.asarray([axes])
    for row, group in enumerate(groups):
        for col, event_type in enumerate(EVENT_TYPES):
            ax = axes[row, col]
            raw = sub_samples[sub_samples["plot_group"].eq(group) & sub_samples["event_type"].astype(str).eq(event_type)].copy()
            med = sub_summary[sub_summary["plot_group"].eq(group) & sub_summary["event_type"].astype(str).eq(event_type)].sort_values("t_bin_sec")
            if raw.empty or med.empty:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            else:
                ax.plot(
                    raw["t_rel_sec"] / 60.0,
                    raw["normalized_power"],
                    ".",
                    color="0.25",
                    alpha=0.018 if len(raw) > 20_000 else 0.035,
                    markersize=1.2 if len(raw) > 20_000 else 1.6,
                    rasterized=True,
                )
                color = colors.get(group, "black")
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
                if col == 0:
                    ax.set_ylabel(f"{group}\nnormalized power")
                    ax.text(
                        0.01,
                        0.96,
                        f"{raw['event_uid'].nunique()} events, {raw['control_name_for_plot'].nunique()} track(s)\n"
                        f"max {int(med['n_events'].max())} events/bin",
                        ha="left",
                        va="top",
                        transform=ax.transAxes,
                        fontsize=7,
                        color="0.25",
                    )
            if row == 0:
                ax.set_title(event_type)
            ax.axvline(0.0, color="black", lw=0.85, ls="--")
            ax.axhline(0.0, color="0.45", lw=0.6)
            ax.grid(alpha=0.18)
            if row == len(groups) - 1:
                ax.set_xlabel("minutes from predicted event")
    fig.suptitle(
        f"{title_prefix}, {freq:.2f} MHz {ANTENNA_LABEL}\n"
        "Per-event locally normalized power; no source-like sign convention or detection score.",
        y=0.997,
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.966])
    path = out_dir / f"{stem}_{_format_freq(freq)}mhz_lower_v.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _plot_latitude_trend(summary: pd.DataFrame, out_dir: Path) -> Path:
    ctrl = summary[summary["plot_group"].str.startswith("Galactic b=", na=False)].copy()
    ctrl["galactic_b_bin_deg"] = ctrl["plot_group"].str.extract(r"([+-]\d+) deg", expand=False).astype(float)
    rows = []
    for keys, grp in ctrl.groupby(["plot_group", "galactic_b_bin_deg", "frequency_mhz", "event_type"], sort=True):
        group, bdeg, freq, event_type = keys
        pre = grp[(grp["t_bin_sec"] >= -180) & (grp["t_bin_sec"] <= -60)]["median_normalized_power"].median()
        post = grp[(grp["t_bin_sec"] >= 60) & (grp["t_bin_sec"] <= 180)]["median_normalized_power"].median()
        rows.append({"plot_group": group, "galactic_b_bin_deg": bdeg, "frequency_mhz": freq, "event_type": event_type, "post_minus_pre": post - pre})
    trend = pd.DataFrame(rows)
    trend.to_csv(out_dir / "galactic_latitude_prepost_visual_index.csv", index=False)
    fig, axes = plt.subplots(1, len(FREQS_MHZ), figsize=(16, 3.8), sharey=True)
    for ax, freq in zip(axes, FREQS_MHZ):
        sub = trend[np.isclose(trend["frequency_mhz"], freq)]
        for event_type, color in [("disappearance", "#4c78a8"), ("reappearance", "#d95f02")]:
            g = sub[sub["event_type"].eq(event_type)].sort_values("galactic_b_bin_deg")
            ax.plot(g["galactic_b_bin_deg"], g["post_minus_pre"], marker="o", lw=1.4, color=color, label=event_type)
        ax.axhline(0.0, color="black", lw=0.8)
        ax.axvline(0.0, color="0.6", lw=0.6)
        ax.set_title(f"{freq:.2f} MHz")
        ax.set_xlabel("Galactic latitude bin (deg)")
        ax.grid(alpha=0.22)
    axes[0].set_ylabel("post - pre visual index\nfrom median normalized profile")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("Fixed Galactic-latitude controls: visual before/after trend by latitude")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    path = out_dir / "galactic_latitude_visual_index_by_frequency.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _write_report(
    out_dir: Path,
    paths: list[Path],
    coord: pd.DataFrame,
    near_sources: pd.DataFrame,
    gal_counts: pd.DataFrame,
    point_counts: pd.DataFrame,
    args: argparse.Namespace,
) -> Path:
    lines = [
        "# Latitude Morphology Audit",
        "",
        "Purpose: test whether the Earth/Sun-like low-frequency normalized profile morphology tracks ecliptic latitude, Galactic latitude, or only moving-body geometry.",
        "",
        "All plotted profiles use lower V, per-event local normalization, no baseline slope removal, no source-like sign convention, and no SNR.",
        "",
        "## Real Fixed-Source Coordinates",
        "",
        coord.to_string(index=False),
        "",
        f"Near-ecliptic fixed sources with |ecliptic latitude| <= {args.near_ecliptic_threshold_deg:g} deg:",
        "",
        near_sources[["source_name", "ecliptic_lat_deg", "galactic_b_deg"]].to_string(index=False) if not near_sources.empty else "None",
        "",
        "Interpretation note: `tau_a`, `sgr_a`, and `galactic_center` are close to the ecliptic and also close to the Galactic plane. `vir_a` is near the ecliptic but at high Galactic latitude, making it a useful discriminator.",
        "",
        "## Galactic Latitude Control Counts",
        "",
        gal_counts.to_string(index=False),
        "",
        "## Near-Ecliptic Point Source Counts",
        "",
        point_counts.to_string(index=False) if not point_counts.empty else "No near-ecliptic point-source events available.",
        "",
        "## Claims To Check Visually",
        "",
        "- If morphology is mainly an ecliptic-coordinate effect, fixed near-ecliptic point sources should resemble Earth/Sun more than off-ecliptic sources.",
        "- If morphology is mainly Galactic-background replacement, controls near Galactic b=0 deg and point sources near the Galactic plane should show the strongest shared behavior.",
        "- If only Earth/Sun retain the behavior after these controls, moving-body time selection remains the better explanation.",
        "",
        "## Generated Figures",
        "",
    ]
    lines.extend(f"- `{p}`" for p in paths)
    lines.extend(
        [
            "",
            "## Run Configuration",
            "",
            f"- galactic_latitudes_deg: {args.galactic_latitudes}",
            f"- galactic_lon_step_deg: {args.galactic_lon_step_deg}",
            f"- limb_exclusion_deg: {args.limb_exclusion_deg}",
            f"- max_events_per_group: {args.max_events_per_group}",
            "- software versions saved in `run_config.json`.",
            "",
        ]
    )
    path = out_dir / "latitude_morphology_audit_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(ROOT / "outputs/latitude_morphology_audit_v1"))
    parser.add_argument("--galactic-latitudes", default="-60,-30,-10,0,10,30,60")
    parser.add_argument("--galactic-lon-step-deg", type=float, default=45.0)
    parser.add_argument("--limb-exclusion-deg", type=float, default=5.0)
    parser.add_argument("--near-ecliptic-threshold-deg", type=float, default=15.0)
    parser.add_argument("--window-s", type=float, default=900.0)
    parser.add_argument("--inner-s", type=float, default=15.0)
    parser.add_argument("--max-events-per-group", type=int, default=360)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    latitudes = [float(x.strip()) for x in str(args.galactic_latitudes).split(",") if x.strip()]
    write_json(
        out_dir / "run_config.json",
        {
            "galactic_latitudes_deg": latitudes,
            "galactic_lon_step_deg": float(args.galactic_lon_step_deg),
            "limb_exclusion_deg": float(args.limb_exclusion_deg),
            "near_ecliptic_threshold_deg": float(args.near_ecliptic_threshold_deg),
            "window_s": float(args.window_s),
            "inner_s": float(args.inner_s),
            "max_events_per_group": int(args.max_events_per_group),
            "frequencies_mhz": FREQS_MHZ,
            "antenna": ANTENNA,
            "software_versions": software_versions(),
        },
    )

    bands = set(LOW_BANDS)
    raw_clean = _load_clean_subset(bands)
    prediction_geometry = _load_prediction_geometry()
    gal_sources = _galactic_control_sources(latitudes, args.galactic_lon_step_deg)
    gal_sources.to_csv(out_dir / "galactic_latitude_control_sources.csv", index=False)
    gal_events = _predict_or_read_controls(prediction_geometry, gal_sources, out_dir, "galactic_latitude_control_events.csv", args.limb_exclusion_deg)
    gal_events = _prepare_control_events(gal_events)

    coord = _fixed_source_coordinate_table()
    coord.to_csv(out_dir / "fixed_source_coordinate_audit.csv", index=False)
    near_sources = _near_ecliptic_point_sources(coord, args.near_ecliptic_threshold_deg)
    near_sources.to_csv(out_dir / "near_ecliptic_point_sources.csv", index=False)
    point_events = _predict_or_read_controls(prediction_geometry, near_sources, out_dir, "near_ecliptic_point_source_events.csv", args.limb_exclusion_deg) if not near_sources.empty else pd.DataFrame()
    point_events = _prepare_point_source_events(point_events, coord) if not point_events.empty else pd.DataFrame()

    moving_fornax = _moving_and_fornax_events()

    gal_plot_events = pd.concat([moving_fornax, gal_events], ignore_index=True)
    raw_gal = _collect_raw_samples(raw_clean, gal_plot_events, args.window_s, args.max_events_per_group)
    norm_gal = _normalize_samples(raw_gal, args.inner_s)
    gal_summary = _binned_normalized_summary(norm_gal)
    gal_summary.to_csv(out_dir / "galactic_latitude_normalized_profile_summary.csv", index=False)

    point_plot_events = pd.concat([moving_fornax, point_events], ignore_index=True) if not point_events.empty else moving_fornax
    raw_point = _collect_raw_samples(raw_clean, point_plot_events, args.window_s, args.max_events_per_group)
    norm_point = _normalize_samples(raw_point, args.inner_s)
    point_summary = _binned_normalized_summary(norm_point)
    point_summary.to_csv(out_dir / "near_ecliptic_point_source_normalized_profile_summary.csv", index=False)

    gal_groups = ["Earth moving track", "Sun moving track"] + [f"Galactic b={lat:+03.0f} deg" for lat in latitudes] + ["Fornax A fixed source"]
    point_groups = ["Earth moving track", "Sun moving track"] + near_sources["source_name"].astype(str).tolist() + ["Fornax A fixed source"]

    paths: list[Path] = []
    for freq in FREQS_MHZ:
        p = _plot_group_grid(norm_gal, gal_summary, freq, gal_groups, LATITUDE_COLORS, "Galactic latitude control morphology", out_dir, "galactic_latitude_profile_grid")
        if p is not None:
            paths.append(p)
        p = _plot_group_grid(norm_point, point_summary, freq, point_groups, POINT_COLORS, "Near-ecliptic point-source morphology", out_dir, "near_ecliptic_point_source_profile_grid")
        if p is not None:
            paths.append(p)
    if not gal_summary.empty:
        paths.append(_plot_latitude_trend(gal_summary, out_dir))

    gal_counts = _summary_counts(norm_gal)
    point_counts = _summary_counts(norm_point)
    gal_counts.to_csv(out_dir / "galactic_latitude_profile_counts.csv", index=False)
    point_counts.to_csv(out_dir / "near_ecliptic_point_source_profile_counts.csv", index=False)
    report = _write_report(out_dir, paths, coord, near_sources, gal_counts, point_counts, args)
    print(report)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
