#!/usr/bin/env python
"""Compare local MIE lunar-occultation results against RAE-2 profile morphology."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rylevonberg.table_io import read_table


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/mie_vs_rae_interpretation_v1"
MIE_DIR = Path(os.environ.get("RAE2_MIE_DIR", "data/mie"))
MIE_TIME = Path(os.environ.get("RAE2_MIE_TIME_CSV", MIE_DIR / "figures/lunar_source_occultation_700khz_time.csv"))
PROFILE_DIR = ROOT / "outputs/all_frequency_profile_grids_v1"
LONGITUDE_INDEX = ROOT / "outputs/galactic_longitude_morphology_audit_v1/galactic_longitude_visual_index.csv"
FRESNEL_NOTE = ROOT / "outputs/fornax_fresnel_timescale_note.md"

LOW_FREQS = [0.70, 0.90, 1.31, 2.20]
SOURCES = ["earth", "sun", "fornax_a", "cas_a", "cyg_a"]


def _ensure_out() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    return OUT


def _read_mie() -> pd.DataFrame:
    df = read_table(MIE_TIME)
    for col in [
        "time_min",
        "source_angle_from_shadow_axis_deg",
        "occultation_angle_deg",
        "beam_gain_dB",
        "diffraction_only_dB",
        "diffraction_only_over_intrinsic",
        "observed_with_beam_dB",
        "observed_with_beam_over_intrinsic",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["geometric_shadow"] = df["geometric_shadow"].astype(str).str.lower().eq("true")
    return df


def _event_relative_mie(df: pd.DataFrame) -> pd.DataFrame:
    # The generated curve is symmetric about mid-occultation. The two geometric
    # limb crossings are nearest occultation_angle=0 on negative/positive time.
    ingress = df[df["time_min"].lt(0)].iloc[(df[df["time_min"].lt(0)]["occultation_angle_deg"].abs()).argmin()]
    egress = df[df["time_min"].gt(0)].iloc[(df[df["time_min"].gt(0)]["occultation_angle_deg"].abs()).argmin()]
    parts = []
    for event_type, row in [("disappearance", ingress), ("reappearance", egress)]:
        work = df.copy()
        work["event_type"] = event_type
        work["t_rel_s"] = (work["time_min"] - float(row["time_min"])) * 60.0
        # Keep only the local profile around that limb crossing.
        work = work[work["t_rel_s"].between(-900.0, 900.0)].copy()
        parts.append(work)
    return pd.concat(parts, ignore_index=True)


def _profile_contrasts() -> pd.DataFrame:
    rows = []
    for source in SOURCES:
        path = PROFILE_DIR / f"{source}_all_frequency_profile_summary_900s.csv"
        if not path.exists():
            continue
        df = read_table(path)
        df = df[df["antenna"].astype(str).eq("rv2_coarse") & df["frequency_mhz"].isin(LOW_FREQS)].copy()
        for keys, grp in df.groupby(["source_name", "frequency_mhz", "event_type"], sort=True):
            source_name, freq, event_type = keys
            pre = grp[grp["t_bin_sec"].between(-180, -60)]["median_z_power"].median()
            post = grp[grp["t_bin_sec"].between(60, 180)]["median_z_power"].median()
            if not np.isfinite(pre) or not np.isfinite(post):
                continue
            if event_type == "disappearance":
                source_like = pre - post
                post_minus_pre = post - pre
            else:
                source_like = post - pre
                post_minus_pre = post - pre
            rows.append(
                {
                    "source_name": source_name,
                    "frequency_mhz": float(freq),
                    "event_type": event_type,
                    "pre_median_norm_power": float(pre),
                    "post_median_norm_power": float(post),
                    "post_minus_pre": float(post_minus_pre),
                    "source_like_contrast": float(source_like),
                    "sign_class": "source_like" if source_like > 0 else "anti_template" if source_like < 0 else "zero",
                }
            )
    return pd.DataFrame(rows)


def _mie_contrasts(mie_rel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for event_type, grp in mie_rel.groupby("event_type", sort=True):
        pre = grp[grp["t_rel_s"].between(-180, -60)]
        post = grp[grp["t_rel_s"].between(60, 180)]
        for quantity in ["diffraction_only_over_intrinsic", "observed_with_beam_over_intrinsic", "diffraction_only_dB", "observed_with_beam_dB"]:
            pre_med = pre[quantity].median()
            post_med = post[quantity].median()
            if event_type == "disappearance":
                source_like = pre_med - post_med
            else:
                source_like = post_med - pre_med
            rows.append(
                {
                    "event_type": event_type,
                    "quantity": quantity,
                    "pre_median": float(pre_med),
                    "post_median": float(post_med),
                    "source_like_contrast": float(source_like),
                }
            )
    return pd.DataFrame(rows)


def _mie_thresholds(df: pd.DataFrame) -> pd.DataFrame:
    # Use the ingress side, where time increases from outside shadow into occultation.
    ingress = df[df["time_min"].lt(0)].sort_values("time_min")
    limb = ingress.iloc[(ingress["occultation_angle_deg"].abs()).argmin()]
    rows = []
    for quantity in ["diffraction_only_dB", "observed_with_beam_dB"]:
        for threshold in [-3, -10, -20, -40, -60, -80]:
            after = ingress[ingress["time_min"].ge(float(limb["time_min"]))]
            hit = after[after[quantity].le(threshold)]
            if hit.empty:
                continue
            r = hit.iloc[0]
            rows.append(
                {
                    "quantity": quantity,
                    "threshold_dB": threshold,
                    "occultation_angle_deg": float(r["occultation_angle_deg"]),
                    "seconds_after_limb": float((r["time_min"] - limb["time_min"]) * 60.0),
                }
            )
    return pd.DataFrame(rows)


def _galactic_dominance_table() -> pd.DataFrame:
    # Standard low-frequency sky scaling; sufficient for order-of-magnitude comparison.
    # T_sky = 180 K (nu / 180 MHz)^-2.6.
    omega_beam = 2.0 * np.pi
    sun_radius_rad = np.deg2rad(0.25)
    earth_radius_rad = np.arcsin(6378.137 / 384400.0)
    omega_sun = np.pi * sun_radius_rad**2
    omega_earth = np.pi * earth_radius_rad**2
    rows = []
    for freq in [0.45, 0.70, 0.90, 1.31, 2.20, 4.70, 9.18]:
        t_sky = 180.0 * (freq / 180.0) ** (-2.6)
        rows.append(
            {
                "frequency_mhz": freq,
                "t_gal_sky_K_order": t_sky,
                "quiet_body_brightness_needed_sun_K_for_2pi_beam": t_sky * omega_beam / omega_sun,
                "quiet_body_brightness_needed_earth_K_for_2pi_beam": t_sky * omega_beam / omega_earth,
                "beam_to_sun_solid_angle_ratio": omega_beam / omega_sun,
                "beam_to_earth_solid_angle_ratio": omega_beam / omega_earth,
            }
        )
    return pd.DataFrame(rows)


def _longitude_dependence_summary() -> pd.DataFrame:
    if not LONGITUDE_INDEX.exists():
        return pd.DataFrame()
    df = read_table(LONGITUDE_INDEX)
    rows = []
    for keys, grp in df[df["frequency_mhz"].isin(LOW_FREQS)].groupby(["galactic_b_bin_deg", "frequency_mhz", "event_type"], sort=True):
        vals = pd.to_numeric(grp["post_minus_pre"], errors="coerce").dropna()
        if vals.empty:
            continue
        rows.append(
            {
                "galactic_b_deg": float(keys[0]),
                "frequency_mhz": float(keys[1]),
                "event_type": keys[2],
                "n_longitudes": int(vals.size),
                "median_post_minus_pre": float(vals.median()),
                "min_post_minus_pre": float(vals.min()),
                "max_post_minus_pre": float(vals.max()),
                "longitude_range": float(vals.max() - vals.min()),
            }
        )
    return pd.DataFrame(rows).sort_values("longitude_range", ascending=False)


def _plot_mie_vs_profiles(mie_rel: pd.DataFrame, profiles: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for ax, event_type in zip(axes[0], ["disappearance", "reappearance"]):
        sub = mie_rel[mie_rel["event_type"].eq(event_type)].sort_values("t_rel_s")
        ax.plot(sub["t_rel_s"] / 60.0, sub["diffraction_only_dB"], label="MIE diffraction only, dB", lw=2)
        ax.plot(sub["t_rel_s"] / 60.0, sub["observed_with_beam_dB"], label="MIE + beam, dB", lw=1.5, alpha=0.8)
        ax.axvline(0, color="black", ls="--", lw=0.9)
        ax.axhline(0, color="0.5", lw=0.7)
        ax.set_title(f"MIE 700 kHz {event_type}")
        ax.set_xlabel("minutes from geometric limb")
        ax.set_ylabel("relative power (dB)")
        ax.set_ylim(-90, 5)
        ax.legend(frameon=False, fontsize=8)

    for ax, event_type in zip(axes[1], ["disappearance", "reappearance"]):
        sub = profiles[profiles["event_type"].eq(event_type)].copy()
        pivot = sub.pivot_table(index="frequency_mhz", columns="source_name", values="source_like_contrast", aggfunc="median").sort_index()
        for source in [s for s in SOURCES if s in pivot.columns]:
            ax.plot(pivot.index, pivot[source], marker="o", label=source)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(f"RAE lower-V normalized profile sign: {event_type}")
        ax.set_xlabel("frequency (MHz)")
        ax.set_ylabel("source-like contrast\npositive = ordinary occultation sign")
        ax.legend(frameon=False, fontsize=8, ncol=3)
        ax.grid(alpha=0.25)
    path = OUT / "mie_vs_rae_profile_signs.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _write_report(
    mie_thresholds: pd.DataFrame,
    mie_contrasts: pd.DataFrame,
    profiles: pd.DataFrame,
    dominance: pd.DataFrame,
    longitude: pd.DataFrame,
    plot_path: Path,
) -> Path:
    profile_pivot = (
        profiles.groupby(["source_name", "frequency_mhz"], as_index=False)["source_like_contrast"]
        .median()
        .pivot(index="frequency_mhz", columns="source_name", values="source_like_contrast")
        .reset_index()
    )
    low_moving = profiles[profiles["source_name"].isin(["earth", "sun"]) & profiles["frequency_mhz"].isin([0.90, 1.31, 2.20])]
    frac_anti = (low_moving["source_like_contrast"] < 0).mean() if not low_moving.empty else np.nan
    lines = [
        "# MIE vs RAE-2 Occultation Interpretation",
        "",
        "## Question",
        "",
        "Does the MIE lunar diffraction result support the earlier explanation that the broad, nearly straight RAE-2 occultation profiles are caused by diffraction + beam + sparse sampling? And can the same framework explain the low-frequency Earth/Sun anti-template behavior?",
        "",
        "## Short Answer",
        "",
        "Only partly. The MIE calculation supports a non-instantaneous lunar shadow, but it does **not** support diffraction as the dominant cause of the several-minute normalized-power ramps or the Earth/Sun sign reversal.",
        "",
        "The MIE curve is close to a straight line mainly when plotted in dB inside the deep shadow. In linear intensity, the ordinary source contribution collapses rapidly after the geometric limb. The RAE profile grids are locally normalized linear power, so the MIE dB-linearity is not directly the same thing as the RAE straight-looking profile lines.",
        "",
        "## MIE Timescale Evidence",
        "",
        "Threshold crossing after the geometric limb for the 700 kHz lunar-orbiter MIE light curve:",
        "",
        mie_thresholds.to_string(index=False),
        "",
        "Pre/post source-like contrasts in a +/-180 s local window around the MIE limb crossing:",
        "",
        mie_contrasts.to_string(index=False),
        "",
        "Interpretation: MIE + beam keeps the ordinary positive occultation sign. It does not flip disappearance into a reappearance-like profile or vice versa.",
        "",
        "## RAE Profile Sign Evidence",
        "",
        "Median lower-V source-like contrast by source and frequency, combining disappearance and reappearance:",
        "",
        profile_pivot.to_string(index=False),
        "",
        f"For Earth/Sun at 0.90, 1.31, and 2.20 MHz, the fraction of event-type rows with anti-template sign is {frac_anti:.2f}.",
        "",
        "This is not what a positive compact source plus MIE diffraction predicts. Fornax A and Cas A mostly retain ordinary point-source-like signs in these bands, while Earth and Sun are consistently negative-contrast/anti-template.",
        "",
        "## Galactic Background Scale",
        "",
        "Order-of-magnitude diffuse Galactic sky comparison using `T_sky = 180 K (nu/180 MHz)^-2.6` and a broad 2pi beam:",
        "",
        dominance.to_string(index=False, formatters={
            "t_gal_sky_K_order": "{:.3e}".format,
            "quiet_body_brightness_needed_sun_K_for_2pi_beam": "{:.3e}".format,
            "quiet_body_brightness_needed_earth_K_for_2pi_beam": "{:.3e}".format,
            "beam_to_sun_solid_angle_ratio": "{:.3e}".format,
            "beam_to_earth_solid_angle_ratio": "{:.3e}".format,
        }),
        "",
        "At 0.7-2.2 MHz, a wide-beam radiometer is naturally dominated by diffuse Galactic synchrotron unless the compact body has an enormous effective brightness temperature over its small solid angle. The Sun/Earth can still matter, but their signal is a small differential term on top of a large, structured background.",
        "",
        "## Longitude Dependence",
        "",
        "Largest longitude-dependent fixed Galactic-control bins:",
        "",
        longitude.head(12).to_string(index=False) if not longitude.empty else "No longitude audit table found.",
        "",
        "This shows that Galactic longitude matters. A simple Galactic-latitude average can hide sign changes around the same latitude ring.",
        "",
        "## Physical Interpretation",
        "",
        "The best-supported picture is:",
        "",
        "1. MIE diffraction makes the mathematical edge non-ideal, and in dB the deep-shadow tail can look linear.",
        "2. Sparse grouped RAE sampling can make a sharp or moderately smeared transition look visually line-like when plotted with connected binned medians.",
        "3. But neither MIE diffraction nor sparse sampling should reverse the sign of a positive compact source. Existing sampled-template tests also retain the expected sign.",
        "4. The Earth/Sun low-frequency anti-template behavior is more consistent with negative effective contrast relative to a dominant diffuse Galactic/antenna background, selected by moving-body orbital geometry.",
        "5. The sign may involve spacecraft/orbital rotation through the beam, but not rotation alone. If it were just a generic rotation artifact, fixed sources and quiet-pole controls should show the same behavior. They do not.",
        "",
        "For Earth/Sun, disappearance increasing power and reappearance decreasing power means the aligned event is removing/reintroducing a deficit relative to the local beam-weighted sky. That deficit could come from body/plasma absorption or scattering of bright low-frequency background, or from a local moving-track background/beam-gradient term that is locked to the Sun/Earth geometry rather than to the exact disk.",
        "",
        "## Consequence For The Pipeline",
        "",
        "Do not model Earth/Sun low-frequency channels as ordinary positive compact-source occultations by default. The pipeline should report signed contrast: positive-template and anti-template are physically different outcomes. For fixed point sources, the ordinary positive template remains appropriate unless the data demonstrate otherwise.",
        "",
        "## Generated Plot",
        "",
        f"- `{plot_path}`",
        "",
        "## Related Existing Evidence",
        "",
        f"- `{FRESNEL_NOTE}`",
        "- `outputs/sampling_smearing_diagnostics_v2/sampling_smearing_report.md`",
        "- `outputs/low_frequency_reversal_audit_v1/low_frequency_reversal_audit_report.md`",
        "- `outputs/moving_body_geometry_audit_v1/moving_body_geometry_audit_report.md`",
        "- `outputs/galactic_longitude_morphology_audit_v1/galactic_longitude_morphology_findings.md`",
        "",
    ]
    path = OUT / "mie_vs_rae_interpretation_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    _ensure_out()
    mie = _read_mie()
    mie_rel = _event_relative_mie(mie)
    thresholds = _mie_thresholds(mie)
    mie_contrasts = _mie_contrasts(mie_rel)
    profiles = _profile_contrasts()
    dominance = _galactic_dominance_table()
    longitude = _longitude_dependence_summary()
    plot_path = _plot_mie_vs_profiles(mie_rel, profiles)
    thresholds.to_csv(OUT / "mie_threshold_timescales.csv", index=False)
    mie_contrasts.to_csv(OUT / "mie_local_prepost_contrasts.csv", index=False)
    profiles.to_csv(OUT / "rae_lower_v_profile_sign_contrasts.csv", index=False)
    dominance.to_csv(OUT / "galactic_background_order_of_magnitude.csv", index=False)
    longitude.to_csv(OUT / "galactic_longitude_dependence_summary.csv", index=False)
    report = _write_report(thresholds, mie_contrasts, profiles, dominance, longitude, plot_path)
    print(report)


if __name__ == "__main__":
    main()
