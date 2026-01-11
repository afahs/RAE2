#!/usr/bin/env python3
"""
Alternative occultation search pipeline for RAE2 data.

This script focuses on global blocked-vs-visible statistics (streaming across
the full dataset), includes fixed sources, planets, Earth, and random sky points,
and produces per-band and combined-band significance estimates.
"""
import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from astropy.coordinates import FK4, SkyCoord, get_body
from astropy.time import Time
from astropy import units as u
from scipy.stats import norm
import matplotlib.pyplot as plt


MOON_RADIUS_KM = 1737.4
DIST_MOON_EARTH_KM = 384400.0

DEFAULT_DATA_PATH = "/global/cfs/projectdirs/m4895/RAE2Data/interpolatedRAE2MasterFile.csv"
DEFAULT_START = "1974-09-07 14:00"
DEFAULT_END = "1975-06-27 16:00"

BAND_TO_FREQ_MHZ = {
    1: 0.45, 2: 0.70, 3: 0.90, 4: 1.31,
    5: 2.20, 6: 3.93, 7: 4.70, 8: 6.55, 9: 9.18,
}


@dataclass
class Target:
    name: str
    kind: str  # "fixed", "planet", "earth", "random"
    ra_icrs: Optional[str] = None
    dec_icrs: Optional[str] = None
    uvec_b1950: Optional[np.ndarray] = None
    body_name: Optional[str] = None


def band_to_freq_mhz(band):
    return BAND_TO_FREQ_MHZ.get(int(band), np.nan)


def sanitize_name(name):
    return name.lower().replace(" ", "_").replace("-", "_")


def build_default_fixed_sources() -> List[Target]:
    return [
        Target("Fornax-A", "fixed", "03h22m41.7s", "-37d12m30s"),
        Target("Cygnus-A", "fixed", "19h59m28.356s", "+40d44m02.1s"),
        Target("Sag-A", "fixed", "17h45m40.0409s", "-29d00m28.118s"),
    ]


def build_planets() -> List[Target]:
    return [
        Target("Mercury", "planet", body_name="mercury"),
        Target("Venus", "planet", body_name="venus"),
        Target("Mars", "planet", body_name="mars"),
        Target("Jupiter", "planet", body_name="jupiter"),
        Target("Saturn", "planet", body_name="saturn"),
        Target("Uranus", "planet", body_name="uranus"),
        Target("Neptune", "planet", body_name="neptune"),
    ]


def build_earth() -> Target:
    return Target("Earth", "earth")


def icrs_to_b1950_uvec(ra_icrs, dec_icrs):
    coord = SkyCoord(ra=ra_icrs, dec=dec_icrs, frame="icrs")
    coord_b1950 = coord.transform_to(FK4(equinox="B1950"))
    uvec = coord_b1950.cartesian.xyz.value
    return uvec / np.linalg.norm(uvec)


def generate_random_targets(n_random, seed=12345):
    rng = np.random.default_rng(seed)
    ra = rng.uniform(0.0, 360.0, size=n_random)
    sin_dec = rng.uniform(-1.0, 1.0, size=n_random)
    dec = np.degrees(np.arcsin(sin_dec))

    targets = []
    for i in range(n_random):
        ra_deg = ra[i]
        dec_deg = dec[i]
        ra_rad = np.radians(ra_deg)
        dec_rad = np.radians(dec_deg)
        uvec = np.array([
            np.cos(dec_rad) * np.cos(ra_rad),
            np.cos(dec_rad) * np.sin(ra_rad),
            np.sin(dec_rad),
        ])
        targets.append(Target(f"Random-{i:02d}", "random", uvec_b1950=uvec))
    return targets


def compute_cos_moon(dist_orb_moon):
    ratio = np.zeros_like(dist_orb_moon, dtype=np.float64)
    np.divide(MOON_RADIUS_KM, dist_orb_moon, out=ratio, where=dist_orb_moon > 0)
    ratio = np.clip(ratio, 0.0, 1.0)
    return np.sqrt(1.0 - ratio ** 2)


def build_ephemeris_grid(start_time, end_time, step_minutes, planet_targets):
    start = pd.to_datetime(start_time)
    end = pd.to_datetime(end_time)
    step = pd.Timedelta(minutes=step_minutes)
    times = pd.date_range(start, end, freq=step)
    times_astropy = Time(times.to_pydatetime())

    moon = get_body("moon", times_astropy).transform_to(FK4(equinox="B1950"))
    moon_vec = moon.cartesian.xyz.to(u.km).value.T

    sun = get_body("sun", times_astropy).transform_to(FK4(equinox="B1950"))
    sun_vec = sun.cartesian.xyz.to(u.km).value.T
    moon_to_sun = sun_vec - moon_vec

    planet_vecs = {}
    for target in planet_targets:
        planet = get_body(target.body_name, times_astropy).transform_to(FK4(equinox="B1950"))
        planet_vec = planet.cartesian.xyz.to(u.km).value.T
        planet_vecs[target.name] = planet_vec - moon_vec

    return times, planet_vecs, moon_to_sun


def time_to_grid_index(times, grid_start, step_seconds, n_grid):
    delta = times - grid_start
    if isinstance(delta, pd.Series):
        seconds = delta.dt.total_seconds().to_numpy(dtype=np.float64)
    else:
        seconds = delta.total_seconds().astype(np.float64)
    delta = seconds
    idx = np.rint(delta / step_seconds).astype(np.int64)
    idx = np.clip(idx, 0, n_grid - 1)
    return idx


def init_stats(n_targets, n_bands=10):
    zeros = np.zeros((n_targets, n_bands), dtype=np.float64)
    return {
        "count_blocked": zeros.copy(),
        "sum_blocked": zeros.copy(),
        "sumsq_blocked": zeros.copy(),
        "count_visible": zeros.copy(),
        "sum_visible": zeros.copy(),
        "sumsq_visible": zeros.copy(),
    }


def stats_to_per_band_df(stats, targets, data_col):
    rows = []
    for i, target in enumerate(targets):
        for band in range(1, 10):
            n_blk = stats["count_blocked"][i, band]
            n_vis = stats["count_visible"][i, band]
            if n_blk == 0 or n_vis == 0:
                continue

            mu_blk = stats["sum_blocked"][i, band] / n_blk
            mu_vis = stats["sum_visible"][i, band] / n_vis
            var_blk = stats["sumsq_blocked"][i, band] / n_blk - mu_blk ** 2
            var_vis = stats["sumsq_visible"][i, band] / n_vis - mu_vis ** 2
            var_blk = max(var_blk, 0.0)
            var_vis = max(var_vis, 0.0)

            diff = mu_vis - mu_blk
            se = np.sqrt(var_vis / n_vis + var_blk / n_blk)
            if se == 0:
                z = np.nan
                p = np.nan
            else:
                z = diff / se
                p = 2 * (1 - norm.cdf(abs(z)))

            rows.append(
                {
                    "target": target.name,
                    "kind": target.kind,
                    "data_col": data_col,
                    "frequency_band": band,
                    "frequency_mhz": band_to_freq_mhz(band),
                    "n_blocked": int(n_blk),
                    "n_visible": int(n_vis),
                    "mean_blocked": mu_blk,
                    "mean_visible": mu_vis,
                    "diff_visible_minus_blocked": diff,
                    "z_score": z,
                    "p_value": p,
                }
            )

    return pd.DataFrame(rows)


def update_stats_for_target(stats, target_idx, bands, values, blocked_mask):
    blocked_vals = values[blocked_mask]
    blocked_bands = bands[blocked_mask]
    visible_vals = values[~blocked_mask]
    visible_bands = bands[~blocked_mask]

    stats["count_blocked"][target_idx] += np.bincount(blocked_bands, minlength=10)
    stats["sum_blocked"][target_idx] += np.bincount(blocked_bands, weights=blocked_vals, minlength=10)
    stats["sumsq_blocked"][target_idx] += np.bincount(blocked_bands, weights=blocked_vals ** 2, minlength=10)

    stats["count_visible"][target_idx] += np.bincount(visible_bands, minlength=10)
    stats["sum_visible"][target_idx] += np.bincount(visible_bands, weights=visible_vals, minlength=10)
    stats["sumsq_visible"][target_idx] += np.bincount(visible_bands, weights=visible_vals ** 2, minlength=10)


def finalize_stats(stats, targets, data_col, output_dir):
    rows = []
    n_targets = len(targets)

    for i in range(n_targets):
        target = targets[i]
        for band in range(1, 10):
            n_blk = stats["count_blocked"][i, band]
            n_vis = stats["count_visible"][i, band]
            if n_blk == 0 or n_vis == 0:
                continue

            mu_blk = stats["sum_blocked"][i, band] / n_blk
            mu_vis = stats["sum_visible"][i, band] / n_vis
            var_blk = stats["sumsq_blocked"][i, band] / n_blk - mu_blk ** 2
            var_vis = stats["sumsq_visible"][i, band] / n_vis - mu_vis ** 2
            var_blk = max(var_blk, 0.0)
            var_vis = max(var_vis, 0.0)

            diff = mu_vis - mu_blk
            se = np.sqrt(var_vis / n_vis + var_blk / n_blk)
            if se == 0:
                z = np.nan
                p = np.nan
            else:
                z = diff / se
                p = 2 * (1 - norm.cdf(abs(z)))

            rows.append(
                {
                    "target": target.name,
                    "kind": target.kind,
                    "data_col": data_col,
                    "frequency_band": band,
                    "frequency_mhz": band_to_freq_mhz(band),
                    "n_blocked": int(n_blk),
                    "n_visible": int(n_vis),
                    "mean_blocked": mu_blk,
                    "mean_visible": mu_vis,
                    "diff_visible_minus_blocked": diff,
                    "z_score": z,
                    "p_value": p,
                }
            )

    df = pd.DataFrame(rows)
    per_band_path = os.path.join(output_dir, f"summary_per_band_{data_col}.csv")
    df.to_csv(per_band_path, index=False)

    combined_rows = []
    for i in range(n_targets):
        target = targets[i]
        n_blk = stats["count_blocked"][i, 1:].sum()
        n_vis = stats["count_visible"][i, 1:].sum()
        if n_blk == 0 or n_vis == 0:
            continue

        sum_blk = stats["sum_blocked"][i, 1:].sum()
        sum_vis = stats["sum_visible"][i, 1:].sum()
        sumsq_blk = stats["sumsq_blocked"][i, 1:].sum()
        sumsq_vis = stats["sumsq_visible"][i, 1:].sum()

        mu_blk = sum_blk / n_blk
        mu_vis = sum_vis / n_vis
        var_blk = sumsq_blk / n_blk - mu_blk ** 2
        var_vis = sumsq_vis / n_vis - mu_vis ** 2
        var_blk = max(var_blk, 0.0)
        var_vis = max(var_vis, 0.0)

        diff = mu_vis - mu_blk
        se = np.sqrt(var_vis / n_vis + var_blk / n_blk)
        if se == 0:
            z = np.nan
            p = np.nan
        else:
            z = diff / se
            p = 2 * (1 - norm.cdf(abs(z)))

        combined_rows.append(
            {
                "target": target.name,
                "kind": target.kind,
                "data_col": data_col,
                "n_blocked": int(n_blk),
                "n_visible": int(n_vis),
                "mean_blocked": mu_blk,
                "mean_visible": mu_vis,
                "diff_visible_minus_blocked": diff,
                "z_score": z,
                "p_value": p,
            }
        )

    combined_df = pd.DataFrame(combined_rows)
    combined_path = os.path.join(output_dir, f"summary_combined_{data_col}.csv")
    combined_df.to_csv(combined_path, index=False)

    return df, combined_df


def parse_args():
    parser = argparse.ArgumentParser(description="RAE2 occultation search (global stats)")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="Path to interpolatedRAE2MasterFile.csv")
    parser.add_argument("--output-dir", default="RAE2Agent/output_search", help="Output directory")
    parser.add_argument("--start", default=DEFAULT_START, help="Start datetime (inclusive)")
    parser.add_argument("--end", default=DEFAULT_END, help="End datetime (inclusive)")
    parser.add_argument("--data-cols", nargs="+", default=["rv2_coarse"], help="Data columns to analyze")
    parser.add_argument("--chunk-size", type=int, default=200000, help="CSV chunk size")
    parser.add_argument("--random-sky", type=int, default=500, help="Number of random sky points")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed for random sky points")
    parser.add_argument("--planet-step-minutes", type=int, default=30, help="Ephemeris grid step in minutes")
    parser.add_argument("--no-planets", action="store_true", help="Skip planets (fixed + Earth only)")
    parser.add_argument("--sun-limb-threshold", type=float, default=3.0, help="Sun limb threshold (deg)")
    parser.add_argument("--earth-limb-threshold", type=float, default=5.0, help="Earth limb threshold (deg)")
    parser.add_argument("--no-limb-filter", action="store_true", help="Disable Sun/Earth limb filtering")
    parser.add_argument("--downsample-minutes", type=float, default=0.0,
                        help="Optional time downsample (minutes). 0 disables.")
    parser.add_argument("--permute-reps", type=int, default=0,
                        help="Number of permutation reps for null test (0 disables).")
    parser.add_argument("--permute-window-minutes", type=float, default=60.0,
                        help="Time window for permutation shuffles (minutes).")
    parser.add_argument("--permute-seed", type=int, default=12345,
                        help="Random seed for permutation shuffles.")
    parser.add_argument("--no-plots", action="store_true", help="Disable plot generation")
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    fixed_targets = build_default_fixed_sources()
    for target in fixed_targets:
        target.uvec_b1950 = icrs_to_b1950_uvec(target.ra_icrs, target.dec_icrs)

    random_targets = generate_random_targets(args.random_sky, seed=args.seed)

    planet_targets = [] if args.no_planets else build_planets()
    earth_target = build_earth()

    fixed_like_targets = fixed_targets + random_targets
    all_targets = fixed_like_targets + planet_targets + [earth_target]
    real_targets = fixed_targets + planet_targets + [earth_target]
    target_index = {t.name: i for i, t in enumerate(all_targets)}

    planet_grid_times = None
    planet_vecs = {}
    sun_vec_grid = None
    need_ephemeris = bool(planet_targets) or not args.no_limb_filter
    if need_ephemeris:
        planet_grid_times, planet_vecs, sun_vec_grid = build_ephemeris_grid(
            args.start, args.end, args.planet_step_minutes, planet_targets
        )
        grid_start = planet_grid_times[0]
        step_seconds = args.planet_step_minutes * 60
        n_grid = len(planet_grid_times)
    else:
        grid_start = None
        step_seconds = None
        n_grid = None

    usecols = [
        "time", "frequency_band",
        "position_x", "position_y", "position_z",
        "earth_unit_vector_x", "earth_unit_vector_y", "earth_unit_vector_z",
    ] + args.data_cols

    stats_by_col: Dict[str, Dict[str, np.ndarray]] = {}
    for col in args.data_cols:
        stats_by_col[col] = init_stats(len(all_targets))

    permute_reps = int(args.permute_reps)
    perm_stats_by_col: Dict[str, List[Dict[str, np.ndarray]]] = {}
    if permute_reps > 0:
        for col in args.data_cols:
            perm_stats_by_col[col] = [init_stats(len(real_targets)) for _ in range(permute_reps)]
        permute_window_seconds = args.permute_window_minutes * 60.0
        perm_rng = np.random.default_rng(args.permute_seed)
    else:
        permute_window_seconds = None
        perm_rng = None

    start_time = pd.to_datetime(args.start)
    end_time = pd.to_datetime(args.end)
    downsample_minutes = float(args.downsample_minutes)
    seen_buckets = set()
    bucket_size_seconds = None
    if downsample_minutes > 0:
        bucket_size_seconds = downsample_minutes * 60.0

    fixed_uvecs = np.stack([t.uvec_b1950 for t in fixed_like_targets], axis=0)
    planet_vec_stack = None
    if planet_targets:
        planet_vec_stack = np.stack([planet_vecs[t.name] for t in planet_targets], axis=1)

    for chunk in pd.read_csv(args.data, usecols=usecols, chunksize=args.chunk_size, parse_dates=["time"]):
        chunk = chunk[(chunk["time"] >= start_time) & (chunk["time"] <= end_time)]
        if chunk.empty:
            continue

        if downsample_minutes > 0:
            time_seconds = (chunk["time"] - start_time).dt.total_seconds()
            bucket = (time_seconds // bucket_size_seconds).astype(np.int64)
            chunk = chunk.assign(bucket=bucket)
            chunk = chunk[~chunk["bucket"].isin(seen_buckets)]
            if chunk.empty:
                continue
            first_times = chunk.groupby("bucket")["time"].min()
            keep_times = set(first_times.values)
            chunk = chunk[chunk["time"].isin(keep_times)].drop(columns=["bucket"])
            seen_buckets.update(first_times.index.values)
            if chunk.empty:
                continue

        bands = chunk["frequency_band"].astype(int).values
        pos = chunk[["position_x", "position_y", "position_z"]].values.astype(np.float64)
        vec_orb_moon = -pos
        dist_orb_moon = np.linalg.norm(vec_orb_moon, axis=1)
        cos_moon = compute_cos_moon(dist_orb_moon)

        dot_fixed = vec_orb_moon @ fixed_uvecs.T
        with np.errstate(divide="ignore", invalid="ignore"):
            cos_theta_fixed = dot_fixed / dist_orb_moon[:, None]
        cos_theta_fixed = np.nan_to_num(cos_theta_fixed, nan=-1.0, posinf=-1.0, neginf=-1.0)
        blocked_fixed = cos_theta_fixed > cos_moon[:, None]

        grid_idx = None
        if need_ephemeris:
            grid_idx = time_to_grid_index(chunk["time"], grid_start, step_seconds, n_grid)

        planet_blocked = None
        if planet_targets:
            planet_vec = planet_vec_stack[grid_idx]
            vec_orb_planet = planet_vec - pos[:, None, :]
            norm_orb_planet = np.linalg.norm(vec_orb_planet, axis=2)
            dot_planet = np.sum(vec_orb_planet * vec_orb_moon[:, None, :], axis=2)
            with np.errstate(divide="ignore", invalid="ignore"):
                cos_theta_planet = dot_planet / (dist_orb_moon[:, None] * norm_orb_planet)
            cos_theta_planet = np.nan_to_num(cos_theta_planet, nan=-1.0, posinf=-1.0, neginf=-1.0)
            planet_blocked = cos_theta_planet > cos_moon[:, None]

        earth_uvec = chunk[["earth_unit_vector_x", "earth_unit_vector_y", "earth_unit_vector_z"]].values
        vec_moon_earth = earth_uvec * DIST_MOON_EARTH_KM
        vec_orb_earth = vec_moon_earth - pos
        norm_orb_earth = np.linalg.norm(vec_orb_earth, axis=1)
        dot_earth = np.einsum("ij,ij->i", vec_orb_moon, vec_orb_earth)
        with np.errstate(divide="ignore", invalid="ignore"):
            cos_theta_earth = dot_earth / (dist_orb_moon * norm_orb_earth)
        cos_theta_earth = np.nan_to_num(cos_theta_earth, nan=-1.0, posinf=-1.0, neginf=-1.0)
        blocked_earth = cos_theta_earth > cos_moon

        if args.no_limb_filter:
            limb_mask = None
        else:
            moon_ang = np.arccos(np.clip(cos_moon, -1.0, 1.0))
            sun_thresh = np.deg2rad(args.sun_limb_threshold)
            earth_thresh = np.deg2rad(args.earth_limb_threshold)
            cos_sun_thresh = np.cos(moon_ang + sun_thresh)
            cos_earth_thresh = np.cos(moon_ang + earth_thresh)

            sun_vec = sun_vec_grid[grid_idx]
            vec_orb_sun = sun_vec - pos
            norm_orb_sun = np.linalg.norm(vec_orb_sun, axis=1)
            dot_sun = np.einsum("ij,ij->i", vec_orb_moon, vec_orb_sun)
            with np.errstate(divide="ignore", invalid="ignore"):
                cos_theta_sun = dot_sun / (dist_orb_moon * norm_orb_sun)
            cos_theta_sun = np.nan_to_num(cos_theta_sun, nan=-1.0, posinf=-1.0, neginf=-1.0)

            sun_ok = cos_theta_sun <= cos_sun_thresh
            earth_ok = cos_theta_earth <= cos_earth_thresh
            limb_mask = sun_ok & earth_ok

        if limb_mask is None:
            bands_use = bands
            blocked_fixed_use = blocked_fixed
            blocked_earth_use = blocked_earth
            blocked_planet_use = planet_blocked
            perm_bucket_use = None
            if permute_reps > 0:
                perm_bucket_use = ((chunk["time"] - start_time).dt.total_seconds() // permute_window_seconds).astype(
                    np.int64
                ).values
        else:
            if not limb_mask.any():
                continue
            bands_use = bands[limb_mask]
            blocked_fixed_use = blocked_fixed[limb_mask]
            blocked_earth_use = blocked_earth[limb_mask]
            blocked_planet_use = planet_blocked[limb_mask] if planet_blocked is not None else None
            perm_bucket_use = None
            if permute_reps > 0:
                perm_bucket_use = (
                    (chunk["time"].iloc[limb_mask].reset_index(drop=True) - start_time)
                    .dt.total_seconds()
                    .floordiv(permute_window_seconds)
                    .astype(np.int64)
                    .values
                )

        blocked_real_use = None
        if permute_reps > 0:
            blocked_cols = [blocked_fixed_use]
            if blocked_planet_use is not None:
                blocked_cols.append(blocked_planet_use)
            blocked_cols.append(blocked_earth_use[:, None])
            blocked_real_use = np.concatenate(blocked_cols, axis=1)

        for data_col in args.data_cols:
            values = chunk[data_col].astype(np.float64).values
            if limb_mask is not None:
                values = values[limb_mask]

            stats = stats_by_col[data_col]

            for i, target in enumerate(fixed_like_targets):
                update_stats_for_target(stats, i, bands_use, values, blocked_fixed_use[:, i])

            offset = len(fixed_like_targets)
            if planet_targets and blocked_planet_use is not None:
                for j, target in enumerate(planet_targets):
                    update_stats_for_target(stats, offset + j, bands_use, values, blocked_planet_use[:, j])
                offset += len(planet_targets)

            update_stats_for_target(stats, offset, bands_use, values, blocked_earth_use)

            if permute_reps > 0 and blocked_real_use is not None:
                bucket_indices = []
                if perm_bucket_use is not None:
                    for bucket_id in np.unique(perm_bucket_use):
                        idx = np.where(perm_bucket_use == bucket_id)[0]
                        if len(idx) > 1:
                            bucket_indices.append(idx)

                for rep_idx in range(permute_reps):
                    perm_blocked = blocked_real_use.copy()
                    for idx in bucket_indices:
                        perm_order = perm_rng.permutation(len(idx))
                        perm_blocked[idx] = perm_blocked[idx][perm_order]

                    perm_stats = perm_stats_by_col[data_col][rep_idx]
                    for j, _ in enumerate(real_targets):
                        update_stats_for_target(perm_stats, j, bands_use, values, perm_blocked[:, j])

    summary_paths = []
    for data_col, stats in stats_by_col.items():
        per_band_df, combined_df = finalize_stats(stats, all_targets, data_col, args.output_dir)

        summary_paths.append(os.path.join(args.output_dir, f"summary_per_band_{data_col}.csv"))
        summary_paths.append(os.path.join(args.output_dir, f"summary_combined_{data_col}.csv"))

        random_df = per_band_df[per_band_df["kind"] == "random"].copy()
        real_df = per_band_df[per_band_df["kind"] != "random"].copy()
        random_df.to_csv(os.path.join(args.output_dir, f"random_per_band_{data_col}.csv"), index=False)
        real_df.to_csv(os.path.join(args.output_dir, f"real_per_band_{data_col}.csv"), index=False)

        if not args.no_plots:
            plots_dir = os.path.join(args.output_dir, "plots", data_col)
            os.makedirs(plots_dir, exist_ok=True)

            def plot_group(band_df, rand_band, title_suffix, filename):
                if band_df.empty:
                    return

                targets = sorted(band_df["target"].unique().tolist())
                colors = plt.cm.tab20(np.linspace(0, 1, max(len(targets), 1)))
                color_map = {name: colors[i] for i, name in enumerate(targets)}

                plt.figure(figsize=(8, 5))
                plt.hist(rand_band, bins=30, alpha=0.6, color="gray", label="Random sky")

                for _, row in band_df.iterrows():
                    plt.axvline(
                        row["z_score"],
                        linewidth=1.2,
                        alpha=0.8,
                        color=color_map[row["target"]],
                        label=row["target"],
                    )

                plt.title(title_suffix)
                plt.xlabel("Z-Score (visible - blocked)")
                plt.ylabel("Count")
                plt.legend(fontsize=8, ncol=2)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, filename), dpi=150)
                plt.close()

            for band in sorted(random_df["frequency_band"].unique()):
                rand_band = random_df[random_df["frequency_band"] == band]["z_score"].dropna()
                if rand_band.empty:
                    continue

                real_band = real_df[real_df["frequency_band"] == band]
                planets_band = real_band[real_band["kind"] == "planet"]
                non_planets_band = real_band[real_band["kind"] != "planet"]

                title_base = f"{data_col} Band {band} ({band_to_freq_mhz(band):.2f} MHz)"
                plot_group(
                    planets_band,
                    rand_band,
                    f"{title_base} Planets vs Random",
                    f"random_compare_band_{band}_planets.png",
                )
                plot_group(
                    non_planets_band,
                    rand_band,
                    f"{title_base} Non-Planets vs Random",
                    f"random_compare_band_{band}_nonplanets.png",
                )

            rand_comb = combined_df[combined_df["kind"] == "random"]["z_score"].dropna()
            if not rand_comb.empty:
                real_comb = combined_df[combined_df["kind"] != "random"]
                planets_comb = real_comb[real_comb["kind"] == "planet"]
                non_planets_comb = real_comb[real_comb["kind"] != "planet"]

                plot_group(
                    planets_comb,
                    rand_comb,
                    f"{data_col} Combined Bands Planets vs Random",
                    "random_compare_combined_planets.png",
                )
                plot_group(
                    non_planets_comb,
                    rand_comb,
                    f"{data_col} Combined Bands Non-Planets vs Random",
                    "random_compare_combined_nonplanets.png",
                )

        comparison_rows = []
        for band in sorted(per_band_df["frequency_band"].unique()):
            rand_z = random_df[random_df["frequency_band"] == band]["z_score"].dropna()
            if rand_z.empty:
                continue
            for _, row in real_df[real_df["frequency_band"] == band].iterrows():
                z = row["z_score"]
                if np.isnan(z):
                    continue
                pct = (rand_z < z).mean()
                comparison_rows.append(
                    {
                        "target": row["target"],
                        "kind": row["kind"],
                        "data_col": data_col,
                        "frequency_band": band,
                        "frequency_mhz": row["frequency_mhz"],
                        "z_score": z,
                        "random_percentile": pct,
                        "random_n": len(rand_z),
                    }
                )

        comparison_df = pd.DataFrame(comparison_rows)
        comparison_df.to_csv(
            os.path.join(args.output_dir, f"random_comparison_per_band_{data_col}.csv"),
            index=False,
        )

        if not args.no_plots and not comparison_df.empty:
            plots_dir = os.path.join(args.output_dir, "plots", data_col)
            os.makedirs(plots_dir, exist_ok=True)
            for target in sorted(comparison_df["target"].unique()):
                target_df = comparison_df[comparison_df["target"] == target].sort_values("frequency_mhz")
                if target_df.empty:
                    continue
                plt.figure(figsize=(7, 4))
                plt.plot(
                    target_df["frequency_mhz"].values,
                    target_df["random_percentile"].values,
                    marker="o",
                    linewidth=1.5,
                )
                plt.ylim(0, 1)
                plt.xlabel("Frequency (MHz)")
                plt.ylabel("Random Percentile")
                plt.title(f"{target} {data_col} Random Percentile vs Frequency")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                filename = f"random_percentile_{sanitize_name(target)}.png"
                plt.savefig(os.path.join(plots_dir, filename), dpi=150)
                plt.close()

        rand_comb = combined_df[combined_df["kind"] == "random"]["z_score"].dropna()
        real_comb = combined_df[combined_df["kind"] != "random"]
        combined_rows = []
        if not rand_comb.empty:
            for _, row in real_comb.iterrows():
                z = row["z_score"]
                if np.isnan(z):
                    continue
                pct = (rand_comb < z).mean()
                combined_rows.append(
                    {
                        "target": row["target"],
                        "kind": row["kind"],
                        "data_col": data_col,
                        "z_score": z,
                        "random_percentile": pct,
                        "random_n": len(rand_comb),
                    }
                )
        pd.DataFrame(combined_rows).to_csv(
            os.path.join(args.output_dir, f"random_comparison_combined_{data_col}.csv"),
            index=False,
        )

        if permute_reps > 0:
            perm_rows = []
            for rep_idx in range(permute_reps):
                perm_stats = perm_stats_by_col[data_col][rep_idx]
                perm_df = stats_to_per_band_df(perm_stats, real_targets, data_col)
                if perm_df.empty:
                    continue
                perm_df["perm_rep"] = rep_idx
                perm_rows.append(perm_df)

            if perm_rows:
                perm_all = pd.concat(perm_rows, ignore_index=True)
                perm_all.to_csv(
                    os.path.join(args.output_dir, f"perm_null_per_band_{data_col}.csv"),
                    index=False,
                )

                obs_df = per_band_df[per_band_df["kind"] != "random"][
                    ["target", "frequency_band", "z_score", "n_blocked", "n_visible"]
                ].rename(columns={"z_score": "obs_z"})

                merged = perm_all.merge(obs_df, on=["target", "frequency_band"], how="inner")
                grouped = merged.groupby(["target", "frequency_band"])
                comparison = grouped["z_score"].agg(["mean", "std"]).reset_index()
                comparison = comparison.merge(obs_df, on=["target", "frequency_band"], how="left")

                def percentile_fn(sub):
                    obs = sub["obs_z"].iloc[0]
                    return float((sub["z_score"] < obs).mean())

                percentiles = grouped.apply(percentile_fn).reset_index(name="perm_percentile")
                comparison = comparison.merge(percentiles, on=["target", "frequency_band"], how="left")
                comparison["perm_reps"] = permute_reps

                comparison.to_csv(
                    os.path.join(args.output_dir, f"permutation_comparison_per_band_{data_col}.csv"),
                    index=False,
                )

    print("Wrote summaries:")
    for path in summary_paths:
        print("  -", path)


if __name__ == "__main__":
    main()
