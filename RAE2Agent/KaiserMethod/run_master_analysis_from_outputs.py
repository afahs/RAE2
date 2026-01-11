#!/usr/bin/env python3
"""
Compute Kaiser stacking statistics using existing event-window outputs
from RAE2Agent/output_event_full_limbcache (derived from the interpolated
master file).
"""
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
OUTPUT_ROOT = Path("RAE2Agent/output_event_full_limbcache")
OUTPUT_DIR = Path("RAE2Agent/KaiserMethod/output_master_from_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SIG_FILE = "significance_per_band_rv2_coarse.csv"

GEOM_COLS = [
    "position_x", "position_y", "position_z",
    "earth_unit_vector_x", "earth_unit_vector_y", "earth_unit_vector_z",
]

CONFUSION_BODIES = [
    "sun", "mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune",
]

TARGETS = {
    "Fornax-A": {"kind": "fixed", "ra": "03h22m41.7s", "dec": "-37d12m30s"},
    "Cygnus-A": {"kind": "fixed", "ra": "19h59m28.356s", "dec": "+40d44m02.1s"},
    "Sag-A": {"kind": "fixed", "ra": "17h45m40.0409s", "dec": "-29d00m28.118s"},
    "Mercury": {"kind": "planet", "body": "mercury"},
    "Venus": {"kind": "planet", "body": "venus"},
    "Mars": {"kind": "planet", "body": "mars"},
    "Jupiter": {"kind": "planet", "body": "jupiter"},
    "Saturn": {"kind": "planet", "body": "saturn"},
    "Uranus": {"kind": "planet", "body": "uranus"},
    "Neptune": {"kind": "planet", "body": "neptune"},
    "Earth": {"kind": "earth"},
}


def compute_target_uvecs(target, times, orb_pos, earth_uvec):
    n = len(times)
    vec_moon_earth = earth_uvec * oa.DIST_MOON_EARTH_KM

    if target["kind"] == "fixed":
        uvec = oa.ra_dec_to_unit_vector(target["ra"], target["dec"])
        return np.repeat(uvec[None, :], n, axis=0)

    if target["kind"] == "earth":
        vec_orb_target = vec_moon_earth - orb_pos
    else:
        target_vec = get_body(target["body"], times).transform_to(FK4(equinox="B1950"))
        target_vec = target_vec.cartesian.xyz.to(u.km).value.T
        vec_moon_target = target_vec + vec_moon_earth
        vec_orb_target = vec_moon_target - orb_pos

    norm = np.linalg.norm(vec_orb_target, axis=1)
    norm = np.where(norm == 0, np.nan, norm)
    return vec_orb_target / norm[:, None]


def compute_confusion_separations(events_df, geom_df, target):
    times = Time(events_df["event_time"].dt.to_pydatetime())
    geom_rows = geom_df.loc[events_df["event_time"]]

    orb_pos = geom_rows[["position_x", "position_y", "position_z"]].to_numpy()
    earth_uvec = geom_rows[["earth_unit_vector_x", "earth_unit_vector_y", "earth_unit_vector_z"]].to_numpy()
    vec_moon_earth = earth_uvec * oa.DIST_MOON_EARTH_KM

    target_uvecs = compute_target_uvecs(target, times, orb_pos, earth_uvec)

    for body in CONFUSION_BODIES:
        if target["kind"] == "planet" and body == target["body"]:
            events_df[f"{body}_sep_deg"] = 0.0
            continue
        if target["kind"] == "earth" and body == "earth":
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


def load_significance(target_dir: Path) -> pd.DataFrame:
    sig_path = target_dir / "tables" / SIG_FILE
    if not sig_path.exists():
        return pd.DataFrame()
    sig_df = pd.read_csv(sig_path)
    sig_df["pre_event_time"] = pd.to_datetime(sig_df["pre_event_time"])
    sig_df["post_event_time"] = pd.to_datetime(sig_df["post_event_time"])
    return sig_df


def build_event_key(df: pd.DataFrame) -> pd.Series:
    return (
        df["pre_event_time"].astype(str)
        + "|"
        + df["post_event_time"].astype(str)
        + "|"
        + df["event_type"].astype(str)
    )


def main():
    sig_by_target = {}
    all_event_times = set()

    for target_name, target in TARGETS.items():
        target_dir = OUTPUT_ROOT / oa.sanitize_name(target_name)
        sig_df = load_significance(target_dir)
        if sig_df.empty:
            print(f"Missing significance file for {target_name}")
            continue
        sig_by_target[target_name] = sig_df
        all_event_times.update(sig_df["post_event_time"].unique())

    if not all_event_times:
        raise SystemExit("No events found in output_event_full_limbcache")

    print(f"Collecting geometry for {len(all_event_times)} event times...", flush=True)

    geom_rows = []
    event_times = set(pd.to_datetime(list(all_event_times)))
    usecols = ["time"] + GEOM_COLS
    for chunk in pd.read_csv(DATA_PATH, usecols=usecols, parse_dates=["time"], chunksize=250000):
        mask = chunk["time"].isin(event_times)
        if mask.any():
            geom_rows.append(chunk.loc[mask])

    if not geom_rows:
        raise SystemExit("No geometry rows matched event times in master file")

    geom_df = pd.concat(geom_rows, ignore_index=True).groupby("time").first()

    summary_rows = []

    for target_name, target in TARGETS.items():
        sig_df = sig_by_target.get(target_name)
        if sig_df is None or sig_df.empty:
            continue

        sig_df = sig_df.copy()
        sig_df["event_time"] = sig_df["post_event_time"]
        sig_df["event_type"] = sig_df["event_type"].map({"Ingress": "DISAPPEARANCE", "Egress": "REAPPEARANCE"})
        sig_df["event_key"] = build_event_key(sig_df)

        events_df = (
            sig_df[["event_type", "pre_event_time", "post_event_time", "event_time", "event_key"]]
            .drop_duplicates()
            .sort_values("event_time")
        )
        events_df["planet"] = target.get("body", target_name)

        missing_geom = events_df[~events_df["event_time"].isin(geom_df.index)]
        if not missing_geom.empty:
            print(f"Warning: {target_name} has {len(missing_geom)} events missing geometry rows")
            events_df = events_df[events_df["event_time"].isin(geom_df.index)]

        events_df = compute_confusion_separations(events_df, geom_df, target)
        events_df = ks.filter_events_by_confusion(events_df, require_separations=True)

        valid_keys = set(events_df["event_key"].unique())
        sig_df = sig_df[sig_df["event_key"].isin(valid_keys)]

        pre = sig_df["pre_median"].to_numpy(dtype=float)
        post = sig_df["post_median"].to_numpy(dtype=float)
        event_type = sig_df["event_type"].to_numpy()

        with np.errstate(divide="ignore", invalid="ignore"):
            unocc = np.where(event_type == "DISAPPEARANCE", pre, post)
            occ = np.where(event_type == "DISAPPEARANCE", post, pre)
            ratio = unocc / occ
            delta_db = 10.0 * np.log10(ratio)

        sign_bit = np.where(delta_db > 0, 1, -1)

        delta_df = pd.DataFrame(
            {
                "event_key": sig_df["event_key"].values,
                "event_time": sig_df["event_time"].values,
                "event_type": sig_df["event_type"].values,
                "planet": target_name,
                "channel_id": sig_df["frequency_band"].astype(int).values,
                "frequency_mhz": sig_df["frequency_band"].map(oa.band_to_freq_mhz).values,
                "delta_db": delta_db,
                "sign_bit": sign_bit,
            }
        )
        delta_df = delta_df[np.isfinite(delta_df["delta_db"])].copy()

        # Stack statistics
        s_rows = []
        for channel_id, group in delta_df.groupby("channel_id"):
            n = len(group)
            sum_bits = int(group["sign_bit"].sum())
            s_val = sum_bits / np.sqrt(n) if n > 0 else float("nan")
            freq = float(group["frequency_mhz"].iloc[0])
            s_rows.append(
                {
                    "channel_id": int(channel_id),
                    "frequency_mhz": freq,
                    "n_events": n,
                    "sum_bits": sum_bits,
                    "S": s_val,
                }
            )
        s_df = pd.DataFrame(s_rows)

        # Candidate events
        candidates = []
        for event_key, group in delta_df.groupby("event_key"):
            runs = ks.find_candidate_runs(
                group["channel_id"].to_numpy(),
                group["delta_db"].to_numpy(),
                group["sign_bit"].to_numpy(),
                min_run=3,
                delta_threshold_db=0.5,
            )
            for run in runs:
                candidates.append(
                    {
                        "event_key": event_key,
                        "event_time": group["event_time"].iloc[0],
                        "event_type": group["event_type"].iloc[0],
                        "planet": target_name,
                        **run,
                    }
                )
        candidates_df = pd.DataFrame(candidates)

        target_dir = OUTPUT_DIR / oa.sanitize_name(target_name)
        target_dir.mkdir(parents=True, exist_ok=True)
        delta_df.to_csv(target_dir / "event_channel_deltas.csv", index=False)
        s_df.to_csv(target_dir / "stack_statistics.csv", index=False)
        candidates_df.to_csv(target_dir / "candidate_events.csv", index=False)
        events_df.to_csv(target_dir / "events_filtered.csv", index=False)

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
                "target": target_name,
                "n_events": int(events_df.shape[0]),
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
    print("Done. Summary saved to", OUTPUT_DIR / "summary.csv")


if __name__ == "__main__":
    main()
