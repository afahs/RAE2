#!/usr/bin/env python
"""Second-pass tests for why the Sun does not produce a clean positive detection."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rylevonberg.table_io import read_table
from astropy import units as u
from astropy.constants import R_sun
from astropy.coordinates import get_body_barycentric, solar_system_ephemeris
from astropy.time import Time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.events import predict_events
from rylevonberg.util import ensure_dir, write_json


CLEAN = ROOT / "outputs/control_survey_earth_sun_postnov1974_v2/01_ingest/cleaned_timeseries.csv"
BLOCKER_SCRIPT = ROOT / "scripts" / "run_solar_detection_blocker_diagnostics.py"
ANT_LABEL = {"rv1_coarse": "upper V", "rv2_coarse": "lower V"}


def _load_blocker_helpers():
    spec = importlib.util.spec_from_file_location("solar_blocker_helpers", BLOCKER_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load helpers from {BLOCKER_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BLOCK = _load_blocker_helpers()


def _read(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return read_table(path, low_memory=False, **kwargs)


def _source_table() -> pd.DataFrame:
    fixed = _read(ROOT / "configs/bright_sources.csv")
    keep_fixed = fixed[fixed["source_name"].isin(["cas_a", "cyg_a", "tau_a", "vir_a", "fornax_a", "sgr_a", "galactic_center"])].copy()
    planets = pd.DataFrame(
        [
            {"source_name": name, "kind": "body", "body_name": name, "frame": "fk4"}
            for name in ["mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune"]
        ]
    )
    earth = pd.DataFrame([{"source_name": "earth", "kind": "earth", "body_name": "earth", "frame": "fk4"}])
    return pd.concat([earth, planets, keep_fixed], ignore_index=True, sort=False)


def _sun_source() -> pd.DataFrame:
    return pd.DataFrame([{"source_name": "sun", "kind": "body", "body_name": "sun", "frame": "fk4"}])


def _parse_angle_payload(payload: str) -> dict[str, float]:
    out: dict[str, float] = {}
    if pd.isna(payload) or not str(payload):
        return out
    for part in str(payload).split(";"):
        if ":" not in part:
            continue
        name, value = part.split(":", 1)
        try:
            out[name] = float(value)
        except ValueError:
            continue
    return out


def _solar_radius_deg(times: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    t = Time(pd.DatetimeIndex(times).to_pydatetime(), scale="utc")
    with solar_system_ephemeris.set("builtin"):
        moon = get_body_barycentric("moon", t)
        sun = get_body_barycentric("sun", t)
    dist_km = np.linalg.norm((sun.xyz - moon.xyz).to_value(u.km).T, axis=1)
    return np.degrees(np.arcsin(np.clip(R_sun.to_value(u.km) / dist_km, 0.0, 1.0)))


def _finite_disk_summary(events: pd.DataFrame) -> pd.DataFrame:
    unique = events[["event_id", "event_type", "predicted_event_time", "pre_limb_angle_deg", "post_limb_angle_deg", "gap_seconds"]].drop_duplicates("event_id").copy()
    radius = _solar_radius_deg(unique["predicted_event_time"])
    speed = np.abs(unique["post_limb_angle_deg"].to_numpy(dtype=float) - unique["pre_limb_angle_deg"].to_numpy(dtype=float)) / np.maximum(unique["gap_seconds"].to_numpy(dtype=float), 1e-9)
    unique["solar_angular_radius_deg"] = radius
    unique["limb_speed_deg_s"] = speed
    unique["solar_radius_crossing_time_s"] = radius / speed
    unique["solar_diameter_crossing_time_s"] = 2.0 * unique["solar_radius_crossing_time_s"]
    return unique


def _annotate_contaminants(clean: pd.DataFrame, retained_events: pd.DataFrame) -> pd.DataFrame:
    annotated, _states = predict_events(
        clean,
        _sun_source(),
        target_frame="fk4",
        equinox="B1950",
        ephemeris="builtin",
        max_gap_seconds=600.0,
        prediction_cadence_seconds=300.0,
        frequencies=[8],
        antennas=["rv2_coarse"],
        limb_exclusion_sources_df=_source_table(),
        limb_exclusion_deg=None,
    )
    keep_ids = set(retained_events["event_id"].astype(int).drop_duplicates())
    annotated = annotated[annotated["event_id"].astype(int).isin(keep_ids)].copy()
    rows = []
    for _, row in annotated.iterrows():
        angles = _parse_angle_payload(row.get("limb_exclusion_source_angles_deg", ""))
        for name, angle in angles.items():
            rows.append(
                {
                    "event_id": int(row["event_id"]),
                    "event_type": row["event_type"],
                    "predicted_event_time": row["predicted_event_time"],
                    "contaminant_source": name,
                    "limb_angle_deg": angle,
                    "abs_limb_angle_deg": abs(angle),
                }
            )
    return pd.DataFrame.from_records(rows)


def _contaminant_summary(contam: pd.DataFrame, contrasts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if contam.empty:
        return pd.DataFrame(), pd.DataFrame()
    summary_rows = []
    for source, grp in contam.groupby("contaminant_source", sort=True):
        summary_rows.append(
            {
                "contaminant_source": source,
                "n_events_within_1deg": int(grp[grp["abs_limb_angle_deg"] <= 1.0]["event_id"].nunique()),
                "n_events_within_3deg": int(grp[grp["abs_limb_angle_deg"] <= 3.0]["event_id"].nunique()),
                "n_events_within_5deg": int(grp[grp["abs_limb_angle_deg"] <= 5.0]["event_id"].nunique()),
                "min_abs_limb_angle_deg": float(grp["abs_limb_angle_deg"].min()),
                "median_abs_limb_angle_deg": float(grp["abs_limb_angle_deg"].median()),
            }
        )
    summary = pd.DataFrame.from_records(summary_rows).sort_values(["n_events_within_3deg", "min_abs_limb_angle_deg"], ascending=[False, True])

    target = contrasts[
        contrasts["window_s"].eq(300.0)
        & contrasts["normalize"].eq("zscore")
        & contrasts["time_shift_s"].eq(0.0)
        & contrasts["frequency_band"].astype(int).eq(8)
        & contrasts["antenna"].astype(str).eq("rv2_coarse")
    ].copy()
    near_ids = set(contam[contam["abs_limb_angle_deg"] <= 3.0]["event_id"].astype(int))
    target["has_any_contaminant_within_3deg"] = target["event_id"].astype(int).isin(near_ids)
    split = BLOCK._summarize(
        target,
        ["has_any_contaminant_within_3deg"],
        256,
        20260514,
    )
    return summary, split


def _antenna_summary(channel_summary: pd.DataFrame) -> pd.DataFrame:
    sub = channel_summary[
        channel_summary["time_shift_s"].eq(0.0)
        & channel_summary["window_s"].eq(300.0)
        & channel_summary["normalize"].eq("zscore")
    ].copy()
    piv = sub.pivot_table(index=["frequency_band", "frequency_mhz"], columns="antenna", values="bootstrap_snr", aggfunc="first").reset_index()
    piv["upper_v_snr"] = piv.get("rv1_coarse")
    piv["lower_v_snr"] = piv.get("rv2_coarse")
    piv["same_sign"] = np.sign(piv["upper_v_snr"]) == np.sign(piv["lower_v_snr"])
    piv["lower_minus_upper_snr"] = piv["lower_v_snr"] - piv["upper_v_snr"]
    return piv[["frequency_band", "frequency_mhz", "upper_v_snr", "lower_v_snr", "same_sign", "lower_minus_upper_snr"]]


def _earth_channel_comparison(clean: pd.DataFrame, sun_summary: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    earth_events = _read(ROOT / "outputs/planetary_confirmation_survey_science_baseline_v2/events/earth_predicted_events.csv", parse_dates=["predicted_event_time"])
    earth = BLOCK._collect(
        clean,
        earth_events[earth_events["source_name"].astype(str).eq("earth")].copy(),
        windows=[300.0],
        normalizes=["zscore"],
        shifts=[0.0],
        inner_s=60.0,
        outer_fraction=0.8,
    )
    earth_summary = BLOCK._summarize(
        earth,
        ["frequency_band", "frequency_mhz", "antenna", "window_s", "time_shift_s", "normalize"],
        256,
        20260515,
    )
    earth_summary.to_csv(out_dir / "earth_all_channel_contrast_summary.csv", index=False)
    sun = sun_summary[
        sun_summary["time_shift_s"].eq(0.0)
        & sun_summary["window_s"].eq(300.0)
        & sun_summary["normalize"].eq("zscore")
    ][["frequency_band", "frequency_mhz", "antenna", "bootstrap_snr"]].rename(columns={"bootstrap_snr": "sun_snr"})
    earth = earth_summary[["frequency_band", "frequency_mhz", "antenna", "bootstrap_snr"]].rename(columns={"bootstrap_snr": "earth_snr"})
    comp = sun.merge(earth, on=["frequency_band", "frequency_mhz", "antenna"], how="outer")
    comp["sun_sign"] = np.sign(comp["sun_snr"])
    comp["earth_sign"] = np.sign(comp["earth_snr"])
    comp["same_sign"] = comp["sun_sign"].eq(comp["earth_sign"])
    return comp.sort_values(["frequency_band", "antenna"]).reset_index(drop=True)


def _plot_earth_sun_sign(comp: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for antenna, grp in comp.groupby("antenna", sort=True):
        grp = grp.sort_values("frequency_mhz")
        ax.plot(grp["frequency_mhz"], grp["sun_snr"], marker="o", lw=1.7, label=f"Sun {ANT_LABEL.get(antenna, antenna)}")
        ax.plot(grp["frequency_mhz"], grp["earth_snr"], marker="s", lw=1.2, ls="--", label=f"Earth {ANT_LABEL.get(antenna, antenna)}")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Simple contrast bootstrap SNR")
    ax.set_title("Sun versus Earth same-channel sign check")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    path = out_dir / "sun_earth_same_channel_sign_check.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_report(
    out_dir: Path,
    finite: pd.DataFrame,
    antenna: pd.DataFrame,
    contam_summary: pd.DataFrame,
    contam_split: pd.DataFrame,
    earth_comp: pd.DataFrame,
    timing_summary: pd.DataFrame,
    figures: list[Path],
) -> None:
    fd = finite["solar_diameter_crossing_time_s"].describe(percentiles=[0.1, 0.5, 0.9]).to_frame().T
    timing_focus = timing_summary[
        timing_summary["window_s"].eq(300.0)
        & timing_summary["normalize"].eq("zscore")
    ].copy()
    timing_focus["abs_snr"] = timing_focus["bootstrap_snr"].abs()
    earth_key = earth_comp[earth_comp["frequency_band"].isin([2, 3, 4, 5, 8])].copy()
    lines = [
        "# Solar Hypothesis Tests",
        "",
        "These tests follow the blocker diagnostic and evaluate the specific proposed explanations.",
        "",
        "## 1. Finite Solar Disk Timing",
        "",
        "The point-source Sun event time may be wrong if the finite solar disk matters. The table below estimates the time for the lunar limb to cross one solar diameter using the predicted limb-angle speed.",
        "",
        "```\n" + fd.to_string(index=False) + "\n```",
        "",
        "Strongest selected-channel timing rows:",
        "",
        "```\n"
        + timing_focus.sort_values("abs_snr", ascending=False)[
            ["window_s", "time_shift_s", "n_events", "median_contrast", "bootstrap_snr", "positive_fraction"]
        ].head(10).to_string(index=False)
        + "\n```",
        "",
        "Interpretation: if the finite disk were the dominant issue, timing shifts comparable to the disk crossing time should recover a positive sign. They do not.",
        "",
        "## 2. Antenna Behavior",
        "",
        "```\n" + antenna.to_string(index=False) + "\n```",
        "",
        "Interpretation: several bands show negative behavior in both antennas, which argues against a single lower-V-only sign convention mistake. Directional antenna response may still affect amplitudes, but it does not by itself explain the broad anti-template sign.",
        "",
        "## 3. Remaining Lunar-Limb Contaminants",
        "",
        "Nearest contaminant-source counts among retained Sun events:",
        "",
        "```\n" + contam_summary.head(15).to_string(index=False) + "\n```",
        "",
        "Selected 6.55 MHz lower-V simple contrast split by whether any tested contaminant is within 3 degrees of the lunar limb:",
        "",
        "```\n" + (contam_split.to_string(index=False) if not contam_split.empty else "No contaminant split rows.") + "\n```",
        "",
        "Interpretation: remaining known-source limb coincidences exist but do not cleanly explain the anti-template behavior unless the contaminated subset alone carries the negative signal.",
        "",
        "## 4. Channel Sign / Calibration Check",
        "",
        "Sun and Earth are compared in the same bands/antennas using the same simple contrast statistic:",
        "",
        "```\n" + earth_key.to_string(index=False) + "\n```",
        "",
        "Interpretation: this does not support a single global radiometer sign inversion. The key 6.55 MHz lower-V solar channel is negative for the Sun but positive for Earth, while some lower-frequency channels are negative for both Earth and Sun. That means channel-dependent morphology/calibration can still matter, but it does not by itself rescue the Sun as a clean positive occultation detection.",
        "",
        "## Figures",
        "",
        *[f"- {path}" for path in figures],
        "",
        "## Conclusion",
        "",
        "The tested explanations do not rescue a positive solar occultation detection. The strongest remaining explanation is physical/systematic morphology: the Sun-associated response is real in the data but has anti-template sign under the Earth-validated occultation convention. The next useful improvement is an antenna/extended-source forward model, not more scalar SNR cuts.",
    ]
    (out_dir / "solar_hypothesis_tests.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", default=str(CLEAN.relative_to(ROOT)))
    parser.add_argument("--events", default="outputs/planetary_confirmation_survey_sun_earth_excluded_v1/events/sun_predicted_events.csv")
    parser.add_argument("--blocker-root", default="outputs/solar_detection_blocker_diagnostics_v1")
    parser.add_argument("--output-dir", default="outputs/solar_hypothesis_tests_v1")
    args = parser.parse_args()

    out_dir = ensure_dir(ROOT / args.output_dir)
    clean = _read(ROOT / args.clean, parse_dates=["time"])
    events = _read(ROOT / args.events, parse_dates=["predicted_event_time"])
    events = events[events["source_name"].astype(str).eq("sun")].copy()
    channel_summary = _read(ROOT / args.blocker_root / "sun_all_channel_contrast_summary.csv")
    timing_summary = _read(ROOT / args.blocker_root / "sun_selected_channel_timing_summary.csv")
    contrasts = _read(ROOT / args.blocker_root / "sun_all_channel_event_contrasts.csv", parse_dates=["predicted_event_time"])

    finite = _finite_disk_summary(events)
    finite.to_csv(out_dir / "solar_finite_disk_timing_summary.csv", index=False)
    antenna = _antenna_summary(channel_summary)
    antenna.to_csv(out_dir / "solar_antenna_sign_summary.csv", index=False)
    contaminants = _annotate_contaminants(clean, events)
    contaminants.to_csv(out_dir / "solar_retained_event_contaminant_limb_angles.csv", index=False)
    contam_summary, contam_split = _contaminant_summary(contaminants, contrasts)
    contam_summary.to_csv(out_dir / "solar_contaminant_limb_summary.csv", index=False)
    contam_split.to_csv(out_dir / "solar_contaminant_split_contrast_summary.csv", index=False)
    earth_comp = _earth_channel_comparison(clean, channel_summary, out_dir)
    earth_comp.to_csv(out_dir / "sun_earth_same_channel_sign_comparison.csv", index=False)
    figures = [_plot_earth_sun_sign(earth_comp, out_dir)]
    _write_report(out_dir, finite, antenna, contam_summary, contam_split, earth_comp, timing_summary, figures)
    write_json(out_dir / "run_config.json", vars(args))
    print(out_dir / "solar_hypothesis_tests.md")
    print(out_dir / "sun_earth_same_channel_sign_comparison.csv")


if __name__ == "__main__":
    main()
