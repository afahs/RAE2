#!/usr/bin/env python
"""Resolve fixed Galactic-latitude controls by longitude.

This is a visual diagnostic, not a detection-score script. It asks whether the
latitude-control morphology is smooth around a Galactic-latitude ring or is
dominated by particular Galactic longitudes.
"""

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

from rylevonberg.constants import SPACECRAFT_COLUMNS  # noqa: E402
from rylevonberg.events import contaminant_limb_angles, spacecraft_positions  # noqa: E402
from rylevonberg.frames import fixed_source_unit_vector, repeated_unit_vector  # noqa: E402
from rylevonberg.geometry import find_limb_transitions, moon_angular_radius_deg, moon_center_direction, moon_limb_angle_deg  # noqa: E402
from rylevonberg.util import datetime_ns, ensure_dir, software_versions, write_json  # noqa: E402
from scripts.build_ecliptic_controls_with_sun_earth_limb_filter import _limb_exclusion_sources  # noqa: E402
from scripts.build_normalized_ecliptic_control_visuals import _normalize_samples  # noqa: E402
from scripts.build_raw_ecliptic_control_visuals import (  # noqa: E402
    ANTENNA,
    ANTENNA_LABEL,
    EVENT_TYPES,
    FREQS_MHZ,
    FREQ_TO_BAND,
    _collect_raw_samples,
    _format_freq,
    _load_clean_subset,
    _panel_ylim,
    _read,
)
from scripts.run_latitude_morphology_audit import (  # noqa: E402
    _galactic_control_sources,
    _load_prediction_geometry,
)


def _predict_or_read(
    clean_geom: pd.DataFrame,
    sources: pd.DataFrame,
    out_dir: Path,
    limb_exclusion_deg: float,
) -> pd.DataFrame:
    path = out_dir / "galactic_longitude_control_events.csv"
    if path.exists() and path.stat().st_size > 0:
        return _read(path, parse_dates=["predicted_event_time"])
    events = _predict_fixed_controls_fast(clean_geom, sources, float(limb_exclusion_deg))
    if not events.empty:
        meta = sources[
            [
                "source_name",
                "galactic_l_deg",
                "galactic_b_deg",
                "ecliptic_lon_deg",
                "ecliptic_lat_deg",
                "plot_group",
            ]
        ].drop_duplicates("source_name")
        events = events.merge(meta, on="source_name", how="left")
    events.to_csv(path, index=False)
    return events


def _downsample_prediction_grid(clean_geom: pd.DataFrame, cadence_s: float = 300.0) -> pd.DataFrame:
    base = clean_geom.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    if len(base) <= 2:
        return base
    times = pd.DatetimeIndex(base["time"])
    elapsed = (datetime_ns(times) - pd.Timestamp(times[0]).value).astype(float) / 1e9
    bucket = np.floor(elapsed / float(cadence_s)).astype(np.int64)
    keep = np.r_[True, bucket[1:] != bucket[:-1]]
    keep[-1] = True
    return base.loc[keep].reset_index(drop=True)


def _interpolate_transition_value(times: pd.DatetimeIndex, values: np.ndarray, transition: pd.Series) -> float:
    pre_idx = int(transition["pre_idx"])
    post_idx = int(transition["post_idx"])
    if pre_idx < 0 or post_idx >= len(times):
        return np.nan
    t0 = times[pre_idx].value
    t1 = times[post_idx].value
    te = pd.Timestamp(transition["predicted_event_time"]).value
    if t1 == t0:
        return float(values[post_idx])
    frac = float((te - t0) / (t1 - t0))
    return float(values[pre_idx] + frac * (values[post_idx] - values[pre_idx]))


def _predict_fixed_controls_fast(clean_geom: pd.DataFrame, sources: pd.DataFrame, limb_exclusion_deg: float) -> pd.DataFrame:
    """Predict fixed-control lunar events while reusing shared Sun/Earth veto geometry."""
    base = _downsample_prediction_grid(clean_geom, cadence_s=300.0)
    times = pd.DatetimeIndex(base["time"])
    sc = spacecraft_positions(base)
    center = moon_center_direction(sc)
    moon_radius = moon_angular_radius_deg(sc)
    exclusion_limb = contaminant_limb_angles(
        base,
        _limb_exclusion_sources(),
        target_frame="fk4",
        equinox="B1950",
        ephemeris="builtin",
    )
    freq_values = [FREQ_TO_BAND[f] for f in FREQS_MHZ]
    rows = []
    for _, source in sources.iterrows():
        vec = fixed_source_unit_vector(
            float(source["ra_deg"]),
            float(source["dec_deg"]),
            source_frame=str(source.get("frame", "fk4") or "fk4"),
            target_frame="fk4",
            equinox="B1950",
        )
        target = repeated_unit_vector(vec, len(times))
        limb = moon_limb_angle_deg(sc, target)
        transitions = find_limb_transitions(times, limb, max_gap_seconds=600.0)
        if transitions.empty:
            continue
        for event_id, ev in transitions.reset_index(drop=True).iterrows():
            exclusion_values = {
                name: _interpolate_transition_value(times, values, ev)
                for name, values in exclusion_limb.items()
            }
            finite_abs = {name: abs(value) for name, value in exclusion_values.items() if np.isfinite(value)}
            nearest_source = min(finite_abs, key=finite_abs.get) if finite_abs else ""
            nearest_abs = float(finite_abs[nearest_source]) if nearest_source else np.nan
            if np.isfinite(nearest_abs) and nearest_abs <= float(limb_exclusion_deg):
                continue
            idx = int(ev["post_idx"])
            for freq in freq_values:
                rows.append(
                    {
                        "event_id": int(event_id),
                        "source_name": source["source_name"],
                        "source_ra_deg": float(source["ra_deg"]),
                        "source_dec_deg": float(source["dec_deg"]),
                        "frame": source.get("frame", "fk4"),
                        "event_type": ev["event_type"],
                        "predicted_event_time": ev["predicted_event_time"],
                        "frequency_band": freq,
                        "frequency_mhz": float(FREQS_MHZ[freq_values.index(freq)]),
                        "antenna": ANTENNA,
                        "limb_angle_deg": 0.0,
                        "pre_limb_angle_deg": ev["pre_limb_angle_deg"],
                        "post_limb_angle_deg": ev["post_limb_angle_deg"],
                        "moon_center_x": float(center[idx, 0]),
                        "moon_center_y": float(center[idx, 1]),
                        "moon_center_z": float(center[idx, 2]),
                        "moon_angular_radius_deg": float(moon_radius[idx]),
                        "gap_seconds": ev["gap_seconds"],
                        "limb_exclusion_deg": limb_exclusion_deg,
                        "limb_exclusion_nearest_source": nearest_source,
                        "limb_exclusion_nearest_abs_deg": nearest_abs,
                        "quality_flags": "",
                    }
                )
    return pd.DataFrame.from_records(rows)


def _prepare_events(events: pd.DataFrame) -> pd.DataFrame:
    out = events[
        events["antenna"].astype(str).eq(ANTENNA)
        & events["frequency_band"].astype(int).isin([FREQ_TO_BAND[f] for f in FREQS_MHZ])
        & events["event_type"].astype(str).isin(EVENT_TYPES)
    ].copy()
    out["galactic_l_bin_deg"] = pd.to_numeric(out["galactic_l_deg"], errors="coerce").round(6)
    out["galactic_b_bin_deg"] = pd.to_numeric(out["galactic_b_deg"], errors="coerce").round(6)
    out["control_name_for_plot"] = out["source_name"].astype(str)
    out["longitude_group"] = out["galactic_l_bin_deg"].map(lambda x: f"l={x:05.1f} deg")
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


def _sample_event_windows(events: pd.DataFrame, max_events_per_group: int, seed: int = 12345) -> pd.DataFrame:
    """Deterministically cap windows per longitude/frequency/event-type group."""
    if max_events_per_group <= 0 or events.empty:
        return events
    rng = np.random.default_rng(seed)
    parts = []
    by = ["plot_group", "galactic_l_bin_deg", "frequency_band", "event_type"]
    for _, grp in events.groupby(by, sort=True, dropna=False):
        event_uids = np.array(sorted(grp["event_uid"].dropna().unique()))
        if event_uids.size > max_events_per_group:
            event_uids = np.sort(rng.choice(event_uids, size=max_events_per_group, replace=False))
        parts.append(grp[grp["event_uid"].isin(event_uids)])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=events.columns)


def _summary_by_longitude(samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    by = [
        "plot_group",
        "galactic_l_bin_deg",
        "galactic_b_bin_deg",
        "longitude_group",
        "frequency_band",
        "frequency_mhz",
        "event_type",
        "t_bin_sec",
    ]
    for keys, grp in samples.groupby(by, sort=True, dropna=False):
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


def _visual_index(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    by = ["plot_group", "galactic_l_bin_deg", "galactic_b_bin_deg", "frequency_mhz", "event_type"]
    for keys, grp in summary.groupby(by, sort=True, dropna=False):
        pre = grp[(grp["t_bin_sec"] >= -180) & (grp["t_bin_sec"] <= -60)]["median_normalized_power"].median()
        post = grp[(grp["t_bin_sec"] >= 60) & (grp["t_bin_sec"] <= 180)]["median_normalized_power"].median()
        rows.append(
            {
                **dict(zip(by, keys)),
                "post_minus_pre": float(post - pre) if np.isfinite(pre) and np.isfinite(post) else np.nan,
            }
        )
    return pd.DataFrame.from_records(rows)


def _plot_lat_freq_grid(samples: pd.DataFrame, summary: pd.DataFrame, lat: float, freq: float, out_dir: Path) -> Path | None:
    sub_samples = samples[
        np.isclose(samples["galactic_b_bin_deg"], lat) & np.isclose(samples["frequency_mhz"], freq)
    ].copy()
    sub_summary = summary[
        np.isclose(summary["galactic_b_bin_deg"], lat) & np.isclose(summary["frequency_mhz"], freq)
    ].copy()
    if sub_samples.empty or sub_summary.empty:
        return None
    longs = sorted(sub_summary["galactic_l_bin_deg"].dropna().unique())
    fig, axes = plt.subplots(len(longs), 2, figsize=(13.4, max(4.5, 1.75 * len(longs))), sharex=True)
    if len(longs) == 1:
        axes = np.asarray([axes])
    for row, lon in enumerate(longs):
        for col, event_type in enumerate(EVENT_TYPES):
            ax = axes[row, col]
            raw = sub_samples[np.isclose(sub_samples["galactic_l_bin_deg"], lon) & sub_samples["event_type"].eq(event_type)]
            med = sub_summary[np.isclose(sub_summary["galactic_l_bin_deg"], lon) & sub_summary["event_type"].eq(event_type)].sort_values("t_bin_sec")
            if raw.empty or med.empty:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            else:
                ax.plot(
                    raw["t_rel_sec"] / 60.0,
                    raw["normalized_power"],
                    ".",
                    color="0.25",
                    alpha=0.05 if len(raw) < 8000 else 0.025,
                    markersize=1.5,
                    rasterized=True,
                )
                x = med["t_bin_sec"].to_numpy(dtype=float) / 60.0
                ax.fill_between(
                    x,
                    med["q25_normalized_power"].to_numpy(dtype=float),
                    med["q75_normalized_power"].to_numpy(dtype=float),
                    color="#4c78a8",
                    alpha=0.16,
                    linewidth=0,
                )
                ax.plot(x, med["median_normalized_power"], color="#4c78a8", lw=1.55)
                ylim = _panel_ylim(raw["normalized_power"])
                if ylim is not None:
                    ax.set_ylim(*ylim)
                if col == 0:
                    ax.set_ylabel(f"l={lon:05.1f} deg\nnormalized power")
                    ax.text(
                        0.01,
                        0.96,
                        f"{raw['event_uid'].nunique()} events",
                        ha="left",
                        va="top",
                        transform=ax.transAxes,
                        fontsize=7,
                        color="0.25",
                    )
            ax.axvline(0.0, color="black", lw=0.8, ls="--")
            ax.axhline(0.0, color="0.45", lw=0.55)
            ax.grid(alpha=0.18)
            if row == 0:
                ax.set_title(event_type)
            if row == len(longs) - 1:
                ax.set_xlabel("minutes from predicted event")
    fig.suptitle(
        f"Fixed Galactic controls by longitude: b={lat:+.0f} deg, {freq:.2f} MHz {ANTENNA_LABEL}\n"
        "Per-event locally normalized power; no source-like sign convention or SNR.",
        y=0.997,
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.958])
    stem = f"galactic_longitude_profiles_b{lat:+.0f}_{_format_freq(freq)}mhz".replace("+", "p").replace("-", "m")
    path = out_dir / f"{stem}.png"
    fig.savefig(path, dpi=165)
    plt.close(fig)
    return path


def _plot_visual_index_heatmaps(index: pd.DataFrame, out_dir: Path) -> Path | None:
    if index.empty:
        return None
    freqs = list(FREQS_MHZ)
    fig, axes = plt.subplots(2, len(freqs), figsize=(17.0, 6.1), sharex=True, sharey=True)
    vmax = np.nanpercentile(np.abs(index["post_minus_pre"]), 95)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    for row, event_type in enumerate(EVENT_TYPES):
        for col, freq in enumerate(freqs):
            ax = axes[row, col]
            sub = index[np.isclose(index["frequency_mhz"], freq) & index["event_type"].eq(event_type)].copy()
            pivot = sub.pivot_table(
                index="galactic_b_bin_deg",
                columns="galactic_l_bin_deg",
                values="post_minus_pre",
                aggfunc="median",
            ).sort_index(ascending=True)
            if pivot.empty:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
                continue
            im = ax.imshow(
                pivot.to_numpy(dtype=float),
                aspect="auto",
                origin="lower",
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
                extent=[
                    float(pivot.columns.min()) - 22.5,
                    float(pivot.columns.max()) + 22.5,
                    float(pivot.index.min()) - 10,
                    float(pivot.index.max()) + 10,
                ],
            )
            ax.set_title(f"{freq:.2f} MHz")
            if col == 0:
                ax.set_ylabel(f"{event_type}\nGalactic b (deg)")
            if row == 1:
                ax.set_xlabel("Galactic longitude l (deg)")
            ax.set_xticks([0, 90, 180, 270, 315])
            ax.set_yticks(sorted(index["galactic_b_bin_deg"].dropna().unique()))
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82, pad=0.012)
    cbar.set_label("post - pre median normalized power")
    fig.suptitle(
        "Longitude-resolved Galactic controls: before/after visual index\n"
        "Positive means higher after than before; compare longitude structure before averaging a latitude ring.",
        y=0.99,
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 0.965, 0.91])
    path = out_dir / "galactic_longitude_visual_index_heatmaps.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _write_report(out_dir: Path, args: argparse.Namespace, sources: pd.DataFrame, events: pd.DataFrame, paths: list[Path]) -> Path:
    counts = (
        events.groupby(["plot_group", "galactic_l_bin_deg", "frequency_mhz", "event_type"], as_index=False)
        .agg(n_event_rows=("event_id", "size"), n_unique_events=("event_id", "nunique"))
        .sort_values(["plot_group", "galactic_l_bin_deg", "frequency_mhz", "event_type"])
    )
    counts.to_csv(out_dir / "galactic_longitude_event_counts.csv", index=False)
    lines = [
        "# Galactic Longitude Morphology Audit",
        "",
        "Purpose: test whether fixed Galactic-latitude control morphology depends on Galactic longitude.",
        "",
        "Control construction:",
        "",
        f"- fixed Galactic latitudes: `{args.galactic_latitudes}` deg;",
        f"- fixed Galactic longitudes spaced by `{args.galactic_lon_step_deg:g}` deg;",
        "- each `(l, b)` point is transformed to FK4/B1950 RA/Dec;",
        "- predicted lunar occultation events are generated with the same lower-V low-frequency setup as the latitude audit;",
        f"- events are vetoed when Sun or Earth are within `{args.limb_exclusion_deg:g}` deg of the lunar limb;",
        "- plots use locally normalized raw power, not SNR and not sign-corrected contrast.",
        "",
        "The key diagnostic is whether one or two longitudes dominate a latitude bin. If yes, the previous latitude-averaged morphology can hide longitude-dependent sky structure.",
        "",
        "Generated source table:",
        "",
        f"- `{out_dir / 'galactic_longitude_control_sources.csv'}`",
        "",
        "Generated figures:",
        "",
    ]
    lines.extend(f"- `{p}`" for p in paths)
    lines.extend(
        [
            "",
            "Generated tables:",
            "",
            f"- `{out_dir / 'galactic_longitude_normalized_samples.csv'}`",
            f"- `{out_dir / 'galactic_longitude_normalized_profile_summary.csv'}`",
            f"- `{out_dir / 'galactic_longitude_visual_index.csv'}`",
            f"- `{out_dir / 'galactic_longitude_event_counts.csv'}`",
            "",
            "Interpretation guide:",
            "",
            "- If the heatmap is roughly uniform in longitude at a fixed latitude, latitude is a reasonable summary variable.",
            "- If the heatmap changes sign or amplitude with longitude, longitude-dependent diffuse sky structure or geometry is important.",
            "- A strong longitude dependence near `b=0` is more consistent with Galactic-background structure than a simple generic pipeline artifact.",
            "",
        ]
    )
    path = out_dir / "galactic_longitude_morphology_audit_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(ROOT / "outputs/galactic_longitude_morphology_audit_v1"))
    parser.add_argument("--galactic-latitudes", default="-60,-30,-10,0,10,30,60")
    parser.add_argument("--galactic-lon-step-deg", type=float, default=45.0)
    parser.add_argument("--limb-exclusion-deg", type=float, default=5.0)
    parser.add_argument("--window-s", type=float, default=900.0)
    parser.add_argument("--inner-s", type=float, default=15.0)
    parser.add_argument("--max-events-per-group", type=int, default=80)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    latitudes = [float(x.strip()) for x in str(args.galactic_latitudes).split(",") if x.strip()]
    write_json(
        out_dir / "run_config.json",
        {
            "galactic_latitudes_deg": latitudes,
            "galactic_lon_step_deg": float(args.galactic_lon_step_deg),
            "limb_exclusion_deg": float(args.limb_exclusion_deg),
            "window_s": float(args.window_s),
            "inner_s": float(args.inner_s),
            "max_events_per_group": int(args.max_events_per_group),
            "frequencies_mhz": FREQS_MHZ,
            "antenna": ANTENNA,
            "software_versions": software_versions(),
        },
    )

    sources = _galactic_control_sources(latitudes, args.galactic_lon_step_deg)
    sources.to_csv(out_dir / "galactic_longitude_control_sources.csv", index=False)
    clean_geom = _load_prediction_geometry()
    events = _predict_or_read(clean_geom, sources, out_dir, args.limb_exclusion_deg)
    events = _prepare_events(events)
    all_count = int(events["event_uid"].nunique()) if not events.empty else 0
    events = _sample_event_windows(events, args.max_events_per_group)
    sampled_count = int(events["event_uid"].nunique()) if not events.empty else 0
    events.to_csv(out_dir / "galactic_longitude_control_events_prepared.csv", index=False)

    clean = _load_clean_subset({FREQ_TO_BAND[f] for f in FREQS_MHZ})
    raw = _collect_raw_samples(clean, events, args.window_s, args.max_events_per_group)
    if raw.empty:
        raise RuntimeError("No raw samples collected for longitude controls")
    meta = events[
        [
            "control_name_for_plot",
            "galactic_l_bin_deg",
            "galactic_b_bin_deg",
            "longitude_group",
            "ecliptic_lon_deg",
            "ecliptic_lat_deg",
        ]
    ].drop_duplicates("control_name_for_plot")
    raw = raw.merge(meta, on="control_name_for_plot", how="left")
    norm = _normalize_samples(raw, args.inner_s)
    norm.to_csv(out_dir / "galactic_longitude_normalized_samples.csv", index=False)
    summary = _summary_by_longitude(norm)
    summary.to_csv(out_dir / "galactic_longitude_normalized_profile_summary.csv", index=False)
    index = _visual_index(summary)
    index.to_csv(out_dir / "galactic_longitude_visual_index.csv", index=False)

    paths: list[Path] = []
    heatmap = _plot_visual_index_heatmaps(index, out_dir)
    if heatmap is not None:
        paths.append(heatmap)
        print(heatmap)
    for lat in latitudes:
        for freq in FREQS_MHZ:
            path = _plot_lat_freq_grid(norm, summary, lat, freq, out_dir)
            if path is not None:
                paths.append(path)
                print(path)
    (out_dir / "sampling_summary.txt").write_text(
        "\n".join(
            [
                f"unique_event_windows_before_sampling={all_count}",
                f"unique_event_windows_after_sampling={sampled_count}",
                f"max_events_per_longitude_frequency_event_type={args.max_events_per_group}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    report = _write_report(out_dir, args, sources, events, paths)
    print(report)


if __name__ == "__main__":
    main()
