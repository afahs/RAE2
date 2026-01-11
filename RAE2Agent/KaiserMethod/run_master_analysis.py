#!/usr/bin/env python3
"""
Run Kaiser stacking on the interpolated RAE2 master file for
all planets and selected fixed sources.
"""
import os
from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd
from astropy.coordinates import FK4, get_body
from astropy.time import Time
from astropy import units as u

ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


oa = _load_module("occultation_analysis", ROOT / "RAE2Agent" / "occultation_analysis.py")
ks = _load_module("occultation_stack", ROOT / "RAE2Agent" / "KaiserMethod" / "occultation_stack.py")


DATA_PATH = "/global/cfs/projectdirs/m4895/RAE2Data/interpolatedRAE2MasterFile.csv"
OUTPUT_DIR = Path("RAE2Agent/KaiserMethod/output_master")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TIME_COL = "time"
CHANNEL_COL = "frequency_band"
POWER_COL = "rv2_coarse"

GEOM_COLS = [
    "position_x", "position_y", "position_z",
    "earth_unit_vector_x", "earth_unit_vector_y", "earth_unit_vector_z",
]

CONFUSION_BODIES = [
    "sun", "mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune",
]


def compute_target_uvecs(target, times, orb_pos, earth_uvec):
    n = len(times)
    vec_moon_earth = earth_uvec * oa.DIST_MOON_EARTH_KM

    if target.kind == "fixed":
        uvec = oa.ra_dec_to_unit_vector(target.ra_icrs, target.dec_icrs)
        return np.repeat(uvec[None, :], n, axis=0)

    if target.kind == "earth":
        vec_orb_target = vec_moon_earth - orb_pos
    else:
        target_vec = get_body(target.body_name, times).transform_to(FK4(equinox="B1950"))
        target_vec = target_vec.cartesian.xyz.to(u.km).value.T
        vec_moon_target = target_vec + vec_moon_earth
        vec_orb_target = vec_moon_target - orb_pos

    norm = np.linalg.norm(vec_orb_target, axis=1)
    norm = np.where(norm == 0, np.nan, norm)
    return vec_orb_target / norm[:, None]


def compute_confusion_separations(events_df, geom_df, target):
    times = Time(events_df["event_time"].dt.to_pydatetime())
    geom_idx = geom_df.index.get_indexer(events_df["event_time"], method="nearest")
    geom_rows = geom_df.iloc[geom_idx]

    orb_pos = geom_rows[["position_x", "position_y", "position_z"]].to_numpy()
    earth_uvec = geom_rows[["earth_unit_vector_x", "earth_unit_vector_y", "earth_unit_vector_z"]].to_numpy()
    vec_moon_earth = earth_uvec * oa.DIST_MOON_EARTH_KM

    target_uvecs = compute_target_uvecs(target, times, orb_pos, earth_uvec)

    for body in CONFUSION_BODIES:
        if target.kind == "planet" and body == target.body_name:
            events_df[f"{body}_sep_deg"] = 0.0
            continue
        if target.kind == "earth" and body == "earth":
            events_df[f"earth_sep_deg"] = 0.0
            continue

        if body == "earth":
            vec_orb_body = vec_moon_earth - orb_pos
        else:
            body_vec = get_body(body, times).transform_to(FK4(equinox="B1950"))
            body_vec = body_vec.cartesian.xyz.to(u.km).value.T
            vec_moon_body = body_vec + vec_moon_earth
            vec_orb_body = vec_moon_body - orb_pos

        norm = np.linalg.norm(vec_orb_body, axis=1)
        norm = np.where(norm == 0, np.nan, norm)
        uvec_body = vec_orb_body / norm[:, None]

        dots = np.einsum("ij,ij->i", target_uvecs, uvec_body)
        dots = np.clip(dots, -1.0, 1.0)
        sep_deg = np.degrees(np.arccos(dots))
        events_df[f"{body}_sep_deg"] = sep_deg

    return events_df


def main():
    print("Loading data...", flush=True)
    usecols = [TIME_COL, CHANNEL_COL, POWER_COL] + GEOM_COLS
    data_df = pd.read_csv(DATA_PATH, usecols=usecols, parse_dates=[TIME_COL])
    data_df = data_df.sort_values(TIME_COL)

    geom_df = data_df[[TIME_COL] + GEOM_COLS].groupby(TIME_COL).first()

    meas_df = data_df[[TIME_COL, CHANNEL_COL, POWER_COL]].copy()
    meas_df = meas_df.rename(columns={TIME_COL: "time", CHANNEL_COL: "channel_id", POWER_COL: "power"})
    meas_df["channel_id"] = meas_df["channel_id"].astype(int)

    channel_freqs = ks.infer_default_channel_frequencies(meas_df["channel_id"].unique())
    meas_df["frequency_mhz"] = meas_df["channel_id"].map(channel_freqs)
    meas_df["power_db"] = ks.to_db(meas_df["power"].to_numpy(dtype=float), "linear")
    meas_df = meas_df.drop(columns=["power"]).dropna(subset=["power_db"]).sort_values("time")

    all_targets = oa.build_default_targets()
    requested = {
        "Fornax-A", "Cygnus-A", "Sag-A",
        "Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Earth",
    }
    targets = [t for t in all_targets if t.name in requested]

    summary_rows = []
    window_cfg = ks.WindowConfig(default_window_min=4.0)

    for target in targets:
        print(f"Processing {target.name}...", flush=True)
        if target.kind == "fixed":
            uvec = oa.ra_dec_to_unit_vector(target.ra_icrs, target.dec_icrs)
            occ_df = oa.compute_occultation_fixed(geom_df, uvec)
        elif target.kind == "planet":
            occ_df = oa.compute_occultation_planet(geom_df, target.body_name)
        elif target.kind == "earth":
            occ_df = oa.compute_occultation_earth(geom_df)
        else:
            continue

        events = oa.find_discrete_events(occ_df["is_blocked"], max_time_gap_seconds=180.0)
        if events.empty:
            print(f"  No events found for {target.name}", flush=True)
            continue

        events = events.copy()
        events["event_time"] = events["pre_event_time"] + (events["post_event_time"] - events["pre_event_time"]) / 2
        events["event_type"] = events["event_type"].map({"Ingress": "DISAPPEARANCE", "Egress": "REAPPEARANCE"})
        events["event_id"] = [f"{target.name}_{i:04d}" for i in range(len(events))]
        if target.kind == "planet":
            events["planet"] = target.body_name
        else:
            events["planet"] = target.name

        events = compute_confusion_separations(events, geom_df, target)
        filtered_events = ks.filter_events_by_confusion(events, require_separations=True)

        delta_df = ks.compute_event_channel_deltas(
            meas_df,
            filtered_events,
            window_config=window_cfg,
            min_samples_per_side=2,
            statistic="mean",
            trim_fraction=0.1,
        )

        s_df = ks.compute_S_statistics(delta_df)
        candidates_df = ks.find_candidate_events(delta_df, min_run=3, delta_threshold_db=0.5)

        target_dir = OUTPUT_DIR / ks.sanitize_filename(target.name)
        target_dir.mkdir(parents=True, exist_ok=True)
        delta_df.to_csv(target_dir / "event_channel_deltas.csv", index=False)
        s_df.to_csv(target_dir / "stack_statistics.csv", index=False)
        candidates_df.to_csv(target_dir / "candidate_events.csv", index=False)
        filtered_events.to_csv(target_dir / "events_filtered.csv", index=False)

        s_df = s_df.sort_values("channel_id")
        s_flag = s_df["S"] >= 2.0
        max_run = 0
        run = 0
        last_ch = None
        for ch, flag in zip(s_df["channel_id"], s_flag):
            if flag and (last_ch is None or ch == last_ch + 1):
                run += 1
            elif flag:
                run = 1
            else:
                run = 0
            last_ch = ch
            max_run = max(max_run, run)

        summary_rows.append(
            {
                "target": target.name,
                "n_events": int(filtered_events.shape[0]),
                "n_event_channel": int(delta_df.shape[0]),
                "n_channels": int(s_df.shape[0]),
                "n_channels_S_ge_2": int(s_flag.sum()),
                "max_S": float(s_df["S"].max()) if not s_df.empty else float("nan"),
                "max_S_run": int(max_run),
                "candidate_events": int(candidates_df.shape[0]),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    print("Done. Summary saved to", OUTPUT_DIR / "summary.csv", flush=True)


if __name__ == "__main__":
    main()
