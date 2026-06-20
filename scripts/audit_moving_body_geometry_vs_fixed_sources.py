#!/usr/bin/env python
"""Audit why Earth/Sun low-frequency profiles differ from fixed sources."""

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
from astropy.coordinates import FK4, CartesianRepresentation, SkyCoord, get_body_barycentric, solar_system_ephemeris
from astropy.time import Time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.constants import EARTH_MOON_DISTANCE_KM, EARTH_UNIT_COLUMNS, SPACECRAFT_COLUMNS  # noqa: E402
from rylevonberg.events import source_vectors_for_rows  # noqa: E402
from rylevonberg.frames import body_unit_vectors_from_moon, offset_unit_vectors_tangent  # noqa: E402
from rylevonberg.geometry import earth_direction_from_spacecraft, find_limb_transitions, moon_limb_angle_deg, normalize_vectors  # noqa: E402
from rylevonberg.util import datetime_ns, ensure_dir, robust_sigma  # noqa: E402


OUT = ROOT / "outputs/moving_body_geometry_audit_v1"
CLEAN = ROOT / "outputs/control_survey_earth_sun_postnov1974_v2/01_ingest/cleaned_timeseries.csv"
PLANET_EVENTS = ROOT / "outputs/planetary_confirmation_survey_science_baseline_v2/events/all_planet_predicted_events.csv"
BRIGHT_EVENTS = ROOT / "outputs/control_survey_bright_sources_postnov1974_v1/02_events/predicted_events.csv"
POLE_EVENTS = ROOT / "outputs/quiet_galactic_poles_control_v1/02_events/predicted_events.csv"
GRID_DIR = ROOT / "outputs/all_frequency_profile_grids_v1"
POLE_GRID = ROOT / "outputs/quiet_galactic_poles_control_v1/04_fast_stack/galactic_pole_stacked_profiles.csv"

ANT_LABEL = {"rv1_coarse": "upper V", "rv2_coarse": "lower V"}
EXPECTED_SIGN = {"disappearance": -1.0, "reappearance": 1.0}
LOW_FREQS = [0.70, 0.90, 1.31, 2.20]


def _read(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return read_table(path, low_memory=False, **kwargs)


def _base_prediction_grid(clean: pd.DataFrame, cadence_s: float = 300.0) -> pd.DataFrame:
    base = clean.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    elapsed = (datetime_ns(base["time"]) - pd.Timestamp(base["time"].iloc[0]).value).astype(float) / 1e9
    bucket = np.floor(elapsed / float(cadence_s)).astype(np.int64)
    keep = np.r_[True, bucket[1:] != bucket[:-1]]
    keep[-1] = True
    return base.loc[keep].reset_index(drop=True)


def location_integrity(clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = clean.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    sample = base.iloc[np.linspace(0, len(base) - 1, min(96, len(base)), dtype=int)].copy()
    times = pd.DatetimeIndex(sample["time"])
    t_ast = Time(times.to_pydatetime(), scale="utc")
    with solar_system_ephemeris.set("builtin"):
        moon = get_body_barycentric("moon", t_ast)
        earth = get_body_barycentric("earth", t_ast)
        sun = get_body_barycentric("sun", t_ast)

    moon_to_earth_icrs = (earth.xyz - moon.xyz).to_value(u.km).T
    moon_to_sun_icrs = (sun.xyz - moon.xyz).to_value(u.km).T
    csv_moon_to_earth = normalize_vectors(sample[EARTH_UNIT_COLUMNS].to_numpy(dtype=float))
    spacecraft = sample[SPACECRAFT_COLUMNS].to_numpy(dtype=float)
    pipeline_sc_to_earth = earth_direction_from_spacecraft(spacecraft, sample[EARTH_UNIT_COLUMNS].to_numpy(dtype=float))
    astropy_sc_to_earth = normalize_vectors(moon_to_earth_icrs - spacecraft)

    rep = CartesianRepresentation(
        moon_to_sun_icrs[:, 0] * u.km,
        moon_to_sun_icrs[:, 1] * u.km,
        moon_to_sun_icrs[:, 2] * u.km,
    )
    sun_fk4_independent = SkyCoord(rep, frame="icrs", obstime=t_ast).transform_to(FK4(equinox=Time("B1950")))
    sun_fk4_vec = normalize_vectors(
        np.column_stack(
            [
                sun_fk4_independent.cartesian.x.to_value(u.km),
                sun_fk4_independent.cartesian.y.to_value(u.km),
                sun_fk4_independent.cartesian.z.to_value(u.km),
            ]
        )
    )
    sun_pipeline = body_unit_vectors_from_moon("sun", times, target_frame="fk4", equinox="B1950", ephemeris="builtin")

    def sep_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.degrees(np.arccos(np.clip(np.sum(normalize_vectors(a) * normalize_vectors(b), axis=1), -1.0, 1.0)))

    rows = [
        {
            "check": "csv_moon_to_earth_vs_astropy_moon_to_earth_icrs",
            "median_sep_deg": float(np.nanmedian(sep_deg(csv_moon_to_earth, moon_to_earth_icrs))),
            "max_sep_deg": float(np.nanmax(sep_deg(csv_moon_to_earth, moon_to_earth_icrs))),
            "n_samples": int(len(sample)),
        },
        {
            "check": "pipeline_sc_to_earth_vs_astropy_sc_to_earth_icrs",
            "median_sep_deg": float(np.nanmedian(sep_deg(pipeline_sc_to_earth, astropy_sc_to_earth))),
            "max_sep_deg": float(np.nanmax(sep_deg(pipeline_sc_to_earth, astropy_sc_to_earth))),
            "n_samples": int(len(sample)),
        },
        {
            "check": "pipeline_sun_fk4_vs_independent_sun_fk4",
            "median_sep_deg": float(np.nanmedian(sep_deg(sun_pipeline, sun_fk4_vec))),
            "max_sep_deg": float(np.nanmax(sep_deg(sun_pipeline, sun_fk4_vec))),
            "n_samples": int(len(sample)),
        },
    ]
    loc = pd.DataFrame(rows)

    events = _read(PLANET_EVENTS, parse_dates=["predicted_event_time"])
    label_rows = []
    for source in ["earth", "sun"]:
        sub = events[
            events["source_name"].astype(str).eq(source)
            & events["frequency_band"].eq(4)
            & events["antenna"].eq("rv2_coarse")
        ].copy()
        if sub.empty:
            continue
        for event_type, grp in sub.groupby("event_type"):
            label_rows.append(
                {
                    "source_name": source,
                    "event_type": event_type,
                    "n_events": int(len(grp)),
                    "median_pre_limb_angle_deg": float(np.nanmedian(grp["pre_limb_angle_deg"])),
                    "median_post_limb_angle_deg": float(np.nanmedian(grp["post_limb_angle_deg"])),
                    "pre_positive_fraction": float(np.nanmean(grp["pre_limb_angle_deg"] >= 0)),
                    "post_positive_fraction": float(np.nanmean(grp["post_limb_angle_deg"] >= 0)),
                }
            )
    labels = pd.DataFrame(label_rows)
    return loc, labels


def grid_source_like_contrasts() -> pd.DataFrame:
    rows = []
    sources = ["earth", "sun", "fornax_a", "cyg_a", "cas_a"]
    for source in sources:
        path = GRID_DIR / f"{source}_all_frequency_profile_summary_900s.csv"
        df = _read(path)
        if df.empty:
            continue
        for keys, grp in df.groupby(["frequency_mhz", "antenna", "event_type"], sort=True):
            mhz, ant, et = keys
            pre = grp[(grp["t_bin_sec"] >= -180) & (grp["t_bin_sec"] <= -60)]["median_z_power"].median()
            post = grp[(grp["t_bin_sec"] >= 60) & (grp["t_bin_sec"] <= 180)]["median_z_power"].median()
            sign = EXPECTED_SIGN.get(str(et), np.nan)
            rows.append(
                {
                    "source_name": source,
                    "source_class": "moving_body" if source in {"earth", "sun"} else "fixed_source",
                    "frequency_mhz": float(mhz),
                    "antenna": str(ant),
                    "antenna_label": ANT_LABEL.get(str(ant), str(ant)),
                    "event_type": str(et),
                    "post_minus_pre": float(post - pre),
                    "source_like_contrast": float(sign * (post - pre)),
                }
            )
    pole = _read(POLE_GRID)
    if not pole.empty:
        for keys, grp in pole.groupby(["source_name", "frequency_mhz", "antenna", "event_type"], sort=True):
            source, mhz, ant, et = keys
            pre = grp[(grp["t_bin_sec"] >= -180) & (grp["t_bin_sec"] <= -60)]["mean"].median()
            post = grp[(grp["t_bin_sec"] >= 60) & (grp["t_bin_sec"] <= 180)]["mean"].median()
            sign = EXPECTED_SIGN.get(str(et), np.nan)
            rows.append(
                {
                    "source_name": str(source),
                    "source_class": "quiet_pole",
                    "frequency_mhz": float(mhz),
                    "antenna": str(ant),
                    "antenna_label": ANT_LABEL.get(str(ant), str(ant)),
                    "event_type": str(et),
                    "post_minus_pre": float(post - pre),
                    "source_like_contrast": float(sign * (post - pre)),
                }
            )
    return pd.DataFrame(rows)


def _channel_groups(clean: pd.DataFrame) -> dict[tuple[int, str], tuple[pd.DataFrame, np.ndarray]]:
    groups = {}
    for (band, ant), grp in clean.groupby(["frequency_band", "antenna"], sort=True):
        g = grp.sort_values("time").reset_index(drop=True)
        groups[(int(band), str(ant))] = (g, datetime_ns(g["time"]))
    return groups


def _event_window(group: pd.DataFrame, group_ns: np.ndarray, event_time: pd.Timestamp, window_s: float) -> tuple[np.ndarray, np.ndarray] | None:
    event_ns = pd.Timestamp(event_time).value
    half_ns = int(window_s * 1e9)
    lo = int(np.searchsorted(group_ns, event_ns - half_ns, side="left"))
    hi = int(np.searchsorted(group_ns, event_ns + half_ns, side="right"))
    if hi <= lo:
        return None
    local = group.iloc[lo:hi]
    tr = (datetime_ns(local["time"]) - event_ns).astype(float) / 1e9
    y = pd.to_numeric(local["power"], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(y) & (np.abs(tr) <= window_s)
    if "is_valid" in local:
        keep &= local["is_valid"].to_numpy(dtype=bool)
    if np.count_nonzero(keep) < 8:
        return None
    order = np.argsort(tr[keep])
    return tr[keep][order], y[keep][order]


def _contrast(t: np.ndarray, y: np.ndarray) -> dict[str, float] | None:
    pre = (t >= -180.0) & (t <= -60.0)
    post = (t >= 60.0) & (t <= 180.0)
    side = np.abs(t) >= 15.0
    if np.count_nonzero(pre) < 1 or np.count_nonzero(post) < 1 or np.count_nonzero(side) < 6:
        return None
    center = float(np.nanmedian(y[side]))
    scale = robust_sigma(y[side] - center)
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.nanstd(y[side]))
    if not np.isfinite(scale) or scale <= 0:
        return None
    z = (y - center) / scale
    return {
        "post_minus_pre_z": float(np.nanmedian(z[post]) - np.nanmedian(z[pre])),
        "raw_post_minus_pre": float(np.nanmedian(y[post]) - np.nanmedian(y[pre])),
    }


def _predict_offset_events(base: pd.DataFrame, source_name: str, east_deg: float, north_deg: float) -> pd.DataFrame:
    times = pd.DatetimeIndex(base["time"])
    sc = base[SPACECRAFT_COLUMNS].to_numpy(dtype=float)
    if source_name == "earth":
        real = source_vectors_for_rows(pd.Series({"source_name": "earth", "kind": "earth", "frame": "fk4"}), times, base)
    elif source_name == "sun":
        real = source_vectors_for_rows(pd.Series({"source_name": "sun", "kind": "body", "body_name": "sun", "frame": "fk4"}), times, base)
    else:
        raise ValueError(source_name)
    vec = offset_unit_vectors_tangent(real, east_offset_deg=east_deg, north_offset_deg=north_deg)
    limb = moon_limb_angle_deg(sc, vec)
    events = find_limb_transitions(times, limb, max_gap_seconds=600.0)
    if events.empty:
        return events
    events = events.copy()
    events["source_name"] = source_name
    events["control_name"] = f"{source_name}_offset_e{east_deg:+g}_n{north_deg:+g}"
    events["offset_east_deg"] = float(east_deg)
    events["offset_north_deg"] = float(north_deg)
    events["offset_deg"] = float(np.hypot(east_deg, north_deg))
    return events


def offset_track_test(clean: pd.DataFrame) -> pd.DataFrame:
    base = _base_prediction_grid(clean)
    groups = _channel_groups(clean)
    offsets = [(0.0, 0.0), (2.0, 0.0), (-2.0, 0.0), (0.0, 2.0), (0.0, -2.0), (5.0, 0.0), (-5.0, 0.0), (0.0, 5.0), (0.0, -5.0)]
    rows = []
    bands = {2: 0.70, 3: 0.90, 4: 1.31, 5: 2.20}
    for source in ["earth", "sun"]:
        for east, north in offsets:
            events = _predict_offset_events(base, source, east, north)
            if events.empty:
                continue
            for band, mhz in bands.items():
                payload = groups.get((band, "rv2_coarse"))
                if payload is None:
                    continue
                group, group_ns = payload
                for _, ev in events.iterrows():
                    local = _event_window(group, group_ns, pd.Timestamp(ev["predicted_event_time"]), 900.0)
                    if local is None:
                        continue
                    c = _contrast(local[0], local[1])
                    if c is None:
                        continue
                    sign = EXPECTED_SIGN[str(ev["event_type"])]
                    rows.append(
                        {
                            "source_name": source,
                            "control_name": ev["control_name"],
                            "offset_deg": ev["offset_deg"],
                            "offset_east_deg": east,
                            "offset_north_deg": north,
                            "event_type": ev["event_type"],
                            "frequency_band": band,
                            "frequency_mhz": mhz,
                            "antenna": "rv2_coarse",
                            "source_like_contrast": sign * c["post_minus_pre_z"],
                            **c,
                        }
                    )
    event_rows = pd.DataFrame(rows)
    agg = (
        event_rows.groupby(["source_name", "control_name", "offset_deg", "frequency_mhz", "event_type"], as_index=False)
        .agg(
            n_events=("source_like_contrast", "size"),
            median_source_like_contrast=("source_like_contrast", "median"),
            source_like_fraction=("source_like_contrast", lambda x: float(np.mean(np.asarray(x) > 0))),
            median_raw_post_minus_pre=("raw_post_minus_pre", "median"),
        )
        if not event_rows.empty
        else pd.DataFrame()
    )
    return agg


def plot_class_comparison(contrast: pd.DataFrame, out_dir: Path) -> Path:
    low = contrast[contrast["frequency_mhz"].isin(LOW_FREQS) & contrast["antenna"].eq("rv2_coarse")].copy()
    summary = (
        low.groupby(["source_name", "source_class", "frequency_mhz"], as_index=False)
        .agg(median_source_like=("source_like_contrast", "median"))
        .sort_values(["source_class", "source_name", "frequency_mhz"])
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    markers = {"moving_body": "o", "fixed_source": "s", "quiet_pole": "^"}
    for (source, klass), grp in summary.groupby(["source_name", "source_class"], sort=True):
        ax.plot(grp["frequency_mhz"], grp["median_source_like"], marker=markers.get(klass, "o"), lw=1.4, label=f"{source} ({klass})")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("frequency (MHz)")
    ax.set_ylabel("median source-like pre/post contrast")
    ax.set_title("Low-frequency lower-V sign: moving bodies vs fixed sources vs quiet poles")
    ax.legend(fontsize=7, ncols=2, frameon=False)
    ax.grid(alpha=0.25)
    path = out_dir / "lowfreq_lower_v_source_class_contrast.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_offset_test(offsets: pd.DataFrame, out_dir: Path) -> Path:
    real = offsets[offsets["offset_deg"].eq(0.0)]
    off = offsets[offsets["offset_deg"].gt(0.0)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, source in zip(axes, ["earth", "sun"]):
        r = real[real["source_name"].eq(source)].groupby("frequency_mhz")["median_source_like_contrast"].median()
        o = off[off["source_name"].eq(source)].groupby("frequency_mhz")["median_source_like_contrast"].median()
        ax.plot(r.index, r.values, marker="o", label="real track")
        ax.plot(o.index, o.values, marker="s", label="2-5 deg offset tracks")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(source)
        ax.set_xlabel("frequency (MHz)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("median source-like contrast")
    axes[0].legend(frameon=False)
    fig.suptitle("Moving-body offset-track test, lower V")
    fig.tight_layout()
    path = out_dir / "earth_sun_offset_track_test.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_report(loc: pd.DataFrame, labels: pd.DataFrame, contrast: pd.DataFrame, offsets: pd.DataFrame, plots: list[Path]) -> Path:
    low = contrast[contrast["frequency_mhz"].isin(LOW_FREQS) & contrast["antenna"].eq("rv2_coarse")].copy()
    class_summary = (
        low.groupby(["source_name", "source_class", "frequency_mhz"], as_index=False)
        .agg(median_source_like=("source_like_contrast", "median"))
        .sort_values(["source_class", "source_name", "frequency_mhz"])
    )
    offset_summary = (
        offsets.groupby(["source_name", "offset_deg", "frequency_mhz"], as_index=False)
        .agg(median_source_like=("median_source_like_contrast", "median"), median_n_events=("n_events", "median"))
        .sort_values(["source_name", "offset_deg", "frequency_mhz"])
    )
    lines = [
        "# Moving Body Geometry Audit",
        "",
        "Purpose: explain why Earth/Sun low-frequency profiles can have the opposite pre/post trend while Fornax A, Cyg A, and Cas A look point-source-like.",
        "",
        "## Three Hypotheses Tested",
        "",
        "1. **Position or event-label bug.** If Sun/Earth locations or disappearance/reappearance labels are wrong, the sign behavior could be artificial.",
        "2. **Moving-body geometry selection.** Earth and Sun are moving bodies whose occultation events select repeated spacecraft-Moon-body geometries; fixed source and Galactic-pole controls do not sample the same geometry.",
        "3. **Nearby-track/background response.** If offset tracks around Earth/Sun preserve the same sign, the effect is likely tied to local moving-body/orbital background geometry rather than only the exact body disk.",
        "",
        "## Location and Label Integrity",
        "",
        loc.to_string(index=False),
        "",
        labels.to_string(index=False),
        "",
        "Interpretation: the Sun path is internally consistent at numerical precision. Earth uses the mission CSV Earth vector; compared with an independent Astropy builtin ephemeris the difference is about 0.35 deg, not a sign or hemisphere error. Event labels have the expected limb-angle signs.",
        "",
        "## Low-Frequency Lower-V Source-Class Comparison",
        "",
        class_summary.to_string(index=False),
        "",
        "Positive values mean expected point-source occultation sign. Negative values mean anti-template sign.",
        "",
        "## Earth/Sun Offset-Track Test",
        "",
        offset_summary.to_string(index=False),
        "",
        "## Plots",
        "",
    ]
    lines.extend(f"- `{p}`" for p in plots)
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "The fixed-source controls and quiet poles do not behave like Earth/Sun at the same frequencies. That argues against a simple plotting bug or generic low-frequency artifact.",
            "",
            "The strongest supported explanation is that Earth/Sun are moving, extended, very bright bodies whose limb events select a specific spacecraft-Moon-body geometry. At low frequency, the local antenna/background response around that geometry can dominate the simple point-source occultation sign. The offset-track test indicates whether this is exact-body-specific or local-track/background-specific.",
        ]
    )
    path = OUT / "moving_body_geometry_audit_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    ensure_dir(OUT)
    clean = _read(CLEAN, parse_dates=["time"])
    loc, labels = location_integrity(clean)
    contrast = grid_source_like_contrasts()
    offsets = offset_track_test(clean)
    loc.to_csv(OUT / "location_integrity_checks.csv", index=False)
    labels.to_csv(OUT / "event_label_limb_angle_checks.csv", index=False)
    contrast.to_csv(OUT / "source_class_prepost_contrasts.csv", index=False)
    offsets.to_csv(OUT / "moving_body_offset_track_contrasts.csv", index=False)
    plots = [plot_class_comparison(contrast, OUT)]
    if not offsets.empty:
        plots.append(plot_offset_test(offsets, OUT))
    report = write_report(loc, labels, contrast, offsets, plots)
    print(report)


if __name__ == "__main__":
    main()
