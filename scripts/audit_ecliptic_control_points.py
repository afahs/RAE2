#!/usr/bin/env python
"""Test whether fixed points near the ecliptic reproduce Earth/Sun low-frequency reversal."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rylevonberg.table_io import read_table
from astropy import units as u
from astropy.coordinates import BarycentricTrueEcliptic, FK4, SkyCoord
from astropy.time import Time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.events import predict_events  # noqa: E402
from rylevonberg.util import datetime_ns, ensure_dir, robust_sigma  # noqa: E402
from scripts.build_all_frequency_occultation_profile_grids import CLEAN  # noqa: E402


OUT = ROOT / "outputs/ecliptic_control_points_v1"
LOW_FREQS = [0.70, 0.90, 1.31, 2.20]
PLOT_FREQS = [0.45, 0.70, 0.90, 1.31, 2.20]
EXPECTED_SIGN = {"disappearance": -1.0, "reappearance": 1.0}


def _read(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return read_table(path, low_memory=False, **kwargs)


def _ecliptic_to_fk4(lon_deg: float, lat_deg: float) -> tuple[float, float]:
    coord = SkyCoord(
        lon=float(lon_deg) * u.deg,
        lat=float(lat_deg) * u.deg,
        frame=BarycentricTrueEcliptic(equinox=Time("B1950")),
    )
    fk4 = coord.transform_to(FK4(equinox=Time("B1950")))
    return float(fk4.ra.deg), float(fk4.dec.deg)


def _source_table() -> pd.DataFrame:
    rows = []
    specs = []
    for lon in range(0, 360, 30):
        specs.append((lon, 0.0, "ecliptic_plane"))
    for lon in range(0, 360, 60):
        for lat in [-10.0, 10.0]:
            specs.append((lon, lat, "near_ecliptic"))
    for lon in range(0, 360, 60):
        for lat in [-60.0, 60.0]:
            specs.append((lon, lat, "off_ecliptic"))
    for lon, lat, klass in specs:
        ra, dec = _ecliptic_to_fk4(lon, lat)
        rows.append(
            {
                "source_name": f"{klass}_lon{int(lon):03d}_lat{int(lat):+03d}".replace("+", "p").replace("-", "m"),
                "kind": "fixed",
                "ra_deg": ra,
                "dec_deg": dec,
                "frame": "fk4",
                "ecliptic_lon_deg": float(lon),
                "ecliptic_lat_deg": float(lat),
                "control_class": klass,
            }
        )
    return pd.DataFrame(rows)


def _predict_or_read(clean: pd.DataFrame, sources: pd.DataFrame) -> pd.DataFrame:
    path = OUT / "ecliptic_control_predicted_events.csv"
    state_path = OUT / "ecliptic_control_limb_visibility_states.csv"
    if path.exists():
        return _read(path, parse_dates=["predicted_event_time"])
    events, states = predict_events(
        clean,
        sources,
        target_frame="fk4",
        equinox="B1950",
        ephemeris="builtin",
        max_gap_seconds=600.0,
        prediction_cadence_seconds=300.0,
        frequencies=list(range(1, 10)),
        antennas=["rv2_coarse"],
    )
    events = events.merge(
        sources[["source_name", "ecliptic_lon_deg", "ecliptic_lat_deg", "control_class"]],
        on="source_name",
        how="left",
    )
    events.to_csv(path, index=False)
    states.to_csv(state_path, index=False)
    return events


def _central_contrasts(summary: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for keys, grp in summary.groupby(["source_name", "frequency_mhz", "antenna", "event_type"], sort=True):
        source, freq, antenna, event_type = keys
        pre = grp[(grp["t_bin_sec"] >= -180) & (grp["t_bin_sec"] <= -60)]["median_z_power"].median()
        post = grp[(grp["t_bin_sec"] >= 60) & (grp["t_bin_sec"] <= 180)]["median_z_power"].median()
        outer_pre = grp[(grp["t_bin_sec"] >= -900) & (grp["t_bin_sec"] <= -600)]["median_z_power"].median()
        outer_post = grp[(grp["t_bin_sec"] >= 600) & (grp["t_bin_sec"] <= 900)]["median_z_power"].median()
        sign = EXPECTED_SIGN[str(event_type)]
        rows.append(
            {
                "run_label": label,
                "source_name": source,
                "frequency_mhz": float(freq),
                "antenna": antenna,
                "event_type": event_type,
                "central_source_like_contrast": float(sign * (post - pre)),
                "central_post_minus_pre": float(post - pre),
                "outer_source_like_contrast": float(sign * (outer_post - outer_pre)),
                "n_events": int(grp["n_events"].max()),
            }
        )
    return pd.DataFrame(rows)


def _channel_groups(clean: pd.DataFrame) -> dict[tuple[int, str], tuple[pd.DataFrame, np.ndarray]]:
    groups = {}
    for (band, antenna), grp in clean.groupby(["frequency_band", "antenna"], sort=True):
        g = grp.sort_values("time").reset_index(drop=True)
        groups[(int(band), str(antenna))] = (g, datetime_ns(g["time"]))
    return groups


def _event_window(group: pd.DataFrame, group_ns: np.ndarray, event_time: pd.Timestamp, window_s: float) -> tuple[np.ndarray, np.ndarray] | None:
    event_ns = pd.Timestamp(event_time).value
    half_ns = int(window_s * 1e9)
    lo = int(np.searchsorted(group_ns, event_ns - half_ns, side="left"))
    hi = int(np.searchsorted(group_ns, event_ns + half_ns, side="right"))
    if hi <= lo:
        return None
    local = group.iloc[lo:hi]
    t = (datetime_ns(local["time"]) - event_ns).astype(float) / 1e9
    y = pd.to_numeric(local["power"], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(y) & (np.abs(t) <= window_s)
    if "is_valid" in local.columns:
        keep &= local["is_valid"].to_numpy(dtype=bool)
    if np.count_nonzero(keep) < 8:
        return None
    order = np.argsort(t[keep])
    return t[keep][order], y[keep][order]


def _event_contrast(t: np.ndarray, y: np.ndarray) -> dict[str, float] | None:
    pre = (t >= -180.0) & (t <= -60.0)
    post = (t >= 60.0) & (t <= 180.0)
    outer_pre = (t >= -900.0) & (t <= -600.0)
    outer_post = (t >= 600.0) & (t <= 900.0)
    side = np.abs(t) >= 15.0
    if np.count_nonzero(pre) < 1 or np.count_nonzero(post) < 1 or np.count_nonzero(side) < 6:
        return None
    center = float(np.nanmedian(y[side]))
    sigma = robust_sigma(y[side] - center)
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.nanstd(y[side]))
    if not np.isfinite(sigma) or sigma <= 0:
        return None
    z = (y - center) / sigma
    pre_z = float(np.nanmedian(z[pre]))
    post_z = float(np.nanmedian(z[post]))
    outer_pre_z = float(np.nanmedian(z[outer_pre])) if np.count_nonzero(outer_pre) else np.nan
    outer_post_z = float(np.nanmedian(z[outer_post])) if np.count_nonzero(outer_post) else np.nan
    return {
        "post_minus_pre": post_z - pre_z,
        "outer_post_minus_pre": outer_post_z - outer_pre_z if np.isfinite(outer_pre_z) and np.isfinite(outer_post_z) else np.nan,
        "n_pre": int(np.count_nonzero(pre)),
        "n_post": int(np.count_nonzero(post)),
    }


def _build_ecliptic_contrasts(clean: pd.DataFrame, events: pd.DataFrame, sources: pd.DataFrame) -> pd.DataFrame:
    contrast_path = OUT / "ecliptic_control_central_contrasts.csv"
    event_path = OUT / "ecliptic_control_event_contrasts.csv"
    if contrast_path.exists():
        return _read(contrast_path)

    groups = _channel_groups(clean)
    rows = []
    work = events[
        events["antenna"].astype(str).eq("rv2_coarse")
        & events["frequency_mhz"].astype(float).isin(PLOT_FREQS)
    ].copy()
    for ev in work.itertuples(index=False):
        key = (int(ev.frequency_band), str(ev.antenna))
        payload = groups.get(key)
        if payload is None:
            continue
        local = _event_window(payload[0], payload[1], pd.Timestamp(ev.predicted_event_time), 900.0)
        if local is None:
            continue
        c = _event_contrast(local[0], local[1])
        if c is None:
            continue
        sign = EXPECTED_SIGN[str(ev.event_type)]
        rows.append(
            {
                "source_name": ev.source_name,
                "event_id": ev.event_id,
                "frequency_band": int(ev.frequency_band),
                "frequency_mhz": float(ev.frequency_mhz),
                "antenna": str(ev.antenna),
                "event_type": str(ev.event_type),
                "central_source_like_contrast": float(sign * c["post_minus_pre"]),
                "central_post_minus_pre": float(c["post_minus_pre"]),
                "outer_source_like_contrast": float(sign * c["outer_post_minus_pre"]) if np.isfinite(c["outer_post_minus_pre"]) else np.nan,
                "n_pre": c["n_pre"],
                "n_post": c["n_post"],
            }
        )
    event_contrast = pd.DataFrame(rows).merge(
        sources[["source_name", "ecliptic_lon_deg", "ecliptic_lat_deg", "control_class"]],
        on="source_name",
        how="left",
    )
    event_contrast.to_csv(event_path, index=False)

    agg_rows = []
    by = ["source_name", "frequency_band", "frequency_mhz", "antenna", "event_type", "ecliptic_lon_deg", "ecliptic_lat_deg", "control_class"]
    for keys, grp in event_contrast.groupby(by, sort=True):
        agg_rows.append(
            {
                **dict(zip(by, keys)),
                "run_label": "fixed_ecliptic_controls",
                "central_source_like_contrast": float(np.nanmedian(grp["central_source_like_contrast"])),
                "central_post_minus_pre": float(np.nanmedian(grp["central_post_minus_pre"])),
                "outer_source_like_contrast": float(np.nanmedian(grp["outer_source_like_contrast"])),
                "n_events": int(grp["event_id"].nunique()),
            }
        )
    contrast = pd.DataFrame(agg_rows)
    contrast.to_csv(contrast_path, index=False)
    return contrast


def _reference_contrasts() -> pd.DataFrame:
    frames = []
    for source, label, klass in [
        ("earth", "earth", "moving_body"),
        ("sun", "sun", "moving_body"),
        ("fornax_a", "fornax_a", "fixed_source"),
        ("cas_a", "cas_a", "fixed_source"),
        ("cyg_a", "cyg_a", "fixed_source"),
    ]:
        path = ROOT / f"outputs/all_frequency_profile_grids_v1/{source}_all_frequency_profile_summary_900s.csv"
        if not path.exists():
            continue
        summary = _read(path)
        con = _central_contrasts(summary, label)
        con["control_class"] = klass
        frames.append(con)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _offset_track_summary() -> pd.DataFrame:
    path = ROOT / "outputs/moving_body_geometry_audit_v1/moving_body_offset_track_contrasts.csv"
    if not path.exists():
        return pd.DataFrame()
    df = _read(path)
    return (
        df[df["frequency_mhz"].isin(LOW_FREQS)]
        .groupby(["source_name", "offset_deg", "frequency_mhz", "event_type"], as_index=False)
        .agg(
            median_source_like_contrast=("median_source_like_contrast", "median")
            if "median_source_like_contrast" in df.columns
            else ("source_like_contrast", "median"),
            n_rows=("frequency_mhz", "size"),
        )
    )


def _aggregate_controls(contrast: pd.DataFrame) -> pd.DataFrame:
    return (
        contrast.groupby(["control_class", "ecliptic_lat_deg", "frequency_mhz", "event_type"], as_index=False)
        .agg(
            n_controls=("source_name", "nunique"),
            median_source_like_contrast=("central_source_like_contrast", "median"),
            frac_negative=("central_source_like_contrast", lambda x: float(np.mean(np.asarray(x) < 0))),
            q25=("central_source_like_contrast", lambda x: float(np.nanpercentile(x, 25))),
            q75=("central_source_like_contrast", lambda x: float(np.nanpercentile(x, 75))),
        )
    )


def _plot_control_spectrum(agg: pd.DataFrame, refs: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    class_styles = {
        "ecliptic_plane": ("#2ca02c", "o"),
        "near_ecliptic": ("#ff7f0e", "s"),
        "off_ecliptic": ("#7f7f7f", "^"),
    }
    for ax, event_type in zip(axes, ["disappearance", "reappearance"]):
        sub = agg[agg["event_type"].eq(event_type)]
        for klass, grp in sub.groupby("control_class", sort=True):
            g = grp.groupby("frequency_mhz", as_index=False).agg(
                median=("median_source_like_contrast", "median"),
                q25=("q25", "median"),
                q75=("q75", "median"),
            )
            color, marker = class_styles.get(klass, ("black", "o"))
            ax.plot(g["frequency_mhz"], g["median"], marker=marker, color=color, lw=1.6, label=klass)
            ax.fill_between(g["frequency_mhz"], g["q25"], g["q75"], color=color, alpha=0.12)
        for source, color in [("earth", "#1f77b4"), ("sun", "#d62728"), ("fornax_a", "#9467bd")]:
            r = refs[
                refs["source_name"].eq(source)
                & refs["antenna"].eq("rv2_coarse")
                & refs["event_type"].eq(event_type)
            ].sort_values("frequency_mhz")
            if not r.empty:
                ax.plot(r["frequency_mhz"], r["central_source_like_contrast"], color=color, lw=1.1, ls="--", label=source)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xscale("log")
        ax.set_xticks(PLOT_FREQS)
        ax.set_xticklabels([f"{f:.2f}" for f in PLOT_FREQS], rotation=45)
        ax.set_title(event_type)
        ax.set_xlabel("frequency (MHz)")
        ax.grid(alpha=0.22, which="both")
    axes[0].set_ylabel("source-like contrast, lower V")
    axes[1].legend(frameon=False, fontsize=7, ncols=2)
    fig.suptitle("Fixed ecliptic controls do not reproduce the Earth/Sun low-frequency reversal")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = OUT / "fixed_ecliptic_controls_contrast_spectrum.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _plot_longitude_heatmap(contrast: pd.DataFrame) -> Path:
    sub = contrast[
        contrast["control_class"].eq("ecliptic_plane")
        & contrast["frequency_mhz"].isin(LOW_FREQS)
        & contrast["event_type"].isin(["disappearance", "reappearance"])
    ].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, event_type in zip(axes, ["disappearance", "reappearance"]):
        p = sub[sub["event_type"].eq(event_type)].pivot_table(
            index="ecliptic_lon_deg",
            columns="frequency_mhz",
            values="central_source_like_contrast",
            aggfunc="median",
        )
        p = p.reindex(index=sorted(p.index), columns=LOW_FREQS)
        data = p.to_numpy(dtype=float)
        im = ax.imshow(data, aspect="auto", cmap="coolwarm", vmin=-0.5, vmax=0.5)
        ax.set_title(event_type)
        ax.set_xticks(np.arange(len(LOW_FREQS)))
        ax.set_xticklabels([f"{f:.2f}" for f in LOW_FREQS])
        ax.set_yticks(np.arange(len(p.index)))
        ax.set_yticklabels([f"{int(v):03d}" for v in p.index], fontsize=8)
        ax.set_xlabel("frequency (MHz)")
    axes[0].set_ylabel("fixed ecliptic longitude (deg)")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="source-like contrast", shrink=0.85)
    fig.suptitle("Fixed ecliptic-plane controls by longitude")
    path = OUT / "fixed_ecliptic_plane_longitude_heatmap.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_offset_track_summary(offset: pd.DataFrame) -> Path | None:
    if offset.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5), sharey=True)
    for ax, source in zip(axes, ["earth", "sun"]):
        sub = offset[offset["source_name"].eq(source)]
        for off, grp in sub.groupby("offset_deg", sort=True):
            g = grp.groupby("frequency_mhz", as_index=False)["median_source_like_contrast"].median().sort_values("frequency_mhz")
            ax.plot(g["frequency_mhz"], g["median_source_like_contrast"], marker="o", lw=1.2, label=f"{off:g} deg")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(f"{source} moving offset tracks")
        ax.set_xlabel("frequency (MHz)")
        ax.grid(alpha=0.22)
    axes[0].set_ylabel("median source-like contrast")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Moving-track offset controls preserve the low-frequency reversal")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = OUT / "moving_offset_track_lowfreq_summary.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _write_report(
    sources: pd.DataFrame,
    agg: pd.DataFrame,
    refs: pd.DataFrame,
    offset: pd.DataFrame,
    paths: list[Path],
) -> Path:
    low_agg = agg[agg["frequency_mhz"].isin(LOW_FREQS)].sort_values(["control_class", "ecliptic_lat_deg", "frequency_mhz", "event_type"])
    ref_low = refs[
        refs["antenna"].eq("rv2_coarse")
        & refs["frequency_mhz"].isin(LOW_FREQS)
        & refs["source_name"].isin(["earth", "sun", "fornax_a", "cas_a", "cyg_a"])
    ][["source_name", "frequency_mhz", "event_type", "central_source_like_contrast"]].sort_values(["source_name", "frequency_mhz", "event_type"])
    offset_low = offset[offset["frequency_mhz"].isin(LOW_FREQS)] if not offset.empty else pd.DataFrame()
    lines = [
        "# Ecliptic Control Points Audit",
        "",
        "Question: if Earth and Sun reverse sign at 0.70-2.20 MHz, do other points near the ecliptic do the same?",
        "",
        "## Control Construction",
        "",
        "- Fixed ecliptic-plane controls: ecliptic latitude 0 deg, longitude every 30 deg.",
        "- Near-ecliptic fixed controls: latitude +/-10 deg, longitude every 60 deg.",
        "- Off-ecliptic fixed controls: latitude +/-60 deg, longitude every 60 deg.",
        "- Coordinates were transformed into FK4/B1950 before event prediction.",
        "- Profiles use lower V only and the same all-frequency 900 s profile method.",
        "",
        f"Total fixed control sources: {len(sources)}",
        "",
        "## Result",
        "",
        "Fixed points near the ecliptic do **not** reproduce the coherent Earth/Sun low-frequency reversal.",
        "This corrects the simple assumption that ecliptic latitude alone is enough.",
        "",
        "The existing moving offset-track controls around Earth/Sun remain more consistent with the reversal.",
        "That points to a moving-body track/time-selection response rather than a generic fixed ecliptic sky-position response.",
        "",
        "## Physical Interpretation",
        "",
        "A fixed source near the ecliptic and a moving body on the ecliptic are not equivalent tests.",
        "A fixed source samples lunar occultations when the Moon passes that fixed sky position.",
        "Earth and Sun events instead select times when the moving Earth/Sun direction intersects the lunar limb,",
        "which couples to spacecraft orbit, lunar phase, antenna orientation, and time-variable low-frequency background conditions.",
        "",
        "Therefore, if fixed ecliptic controls do not reverse but moving Earth/Sun tracks and nearby moving offsets do,",
        "the evidence favors a moving-track or observing-geometry systematic/contrast effect over a simple ecliptic-plane astrophysical effect.",
        "",
        "## Fixed Ecliptic Control Summary",
        "",
        low_agg[
            [
                "control_class",
                "ecliptic_lat_deg",
                "frequency_mhz",
                "event_type",
                "n_controls",
                "median_source_like_contrast",
                "frac_negative",
                "q25",
                "q75",
            ]
        ].to_string(index=False),
        "",
        "## Reference Source Low-Frequency Lower-V Contrasts",
        "",
        ref_low.to_string(index=False),
        "",
    ]
    if not offset_low.empty:
        lines.extend(
            [
                "## Moving Offset Track Summary",
                "",
                offset_low.sort_values(["source_name", "offset_deg", "frequency_mhz", "event_type"]).to_string(index=False),
                "",
            ]
        )
    lines.extend(
        [
            "## How Else To Verify",
            "",
            "1. Generate additional moving-track controls that follow the Sun path with larger ecliptic-longitude phase shifts, not just fixed sky positions.",
            "2. Repeat the same fixed-control test for ecliptic latitudes +/-20 and +/-30 deg to map the transition from ecliptic to off-ecliptic behavior.",
            "3. Compare event-time distributions: if moving-track controls reverse because they select different months/local times, fixed ecliptic controls should not match.",
            "4. Build a channel-sign model: estimate empirical sign per source/frequency/antenna from controls, then require sign stability across event type and offset tracks.",
            "5. Test raw receiver housekeeping or global band medians around Earth/Sun event times; a receiver-state effect should appear even for nearby pseudo-events.",
            "",
            "## Generated Plots",
            "",
        ]
    )
    lines.extend(f"- `{p}`" for p in paths if p is not None)
    path = OUT / "ecliptic_control_points_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    ensure_dir(OUT)
    clean = _read(CLEAN, parse_dates=["time"])
    sources = _source_table()
    sources.to_csv(OUT / "ecliptic_control_source_list.csv", index=False)
    events = _predict_or_read(clean, sources)
    contrast = _build_ecliptic_contrasts(clean, events, sources)
    agg = _aggregate_controls(contrast)
    refs = _reference_contrasts()
    offset = _offset_track_summary()

    agg.to_csv(OUT / "ecliptic_control_aggregate_contrasts.csv", index=False)
    refs.to_csv(OUT / "reference_source_contrasts.csv", index=False)
    offset.to_csv(OUT / "moving_offset_track_lowfreq_summary.csv", index=False)

    paths = [_plot_control_spectrum(agg, refs), _plot_longitude_heatmap(contrast)]
    off_plot = _plot_offset_track_summary(offset)
    if off_plot is not None:
        paths.append(off_plot)
    report = _write_report(sources, agg, refs, offset, paths)
    print(report)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
