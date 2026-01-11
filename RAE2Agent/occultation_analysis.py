#!/usr/bin/env python3
"""
Occultation analysis for RAE2 data.

This script scans for lunar occultation on/off events for planets and fixed
point sources, then measures dip significance per band and combined bands.
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from astropy.coordinates import FK4, SkyCoord, get_body, get_sun
from astropy.time import Time
from astropy import units as u
import multiprocessing as mp


MOON_RADIUS_KM = 1737.4
DIST_MOON_EARTH_KM = 384400.0

DEFAULT_DATA_PATH = "/global/cfs/projectdirs/m4895/RAE2Data/interpolatedRAE2MasterFile.csv"
DEFAULT_START = "1974-09-07 14:00"
DEFAULT_END = "1975-06-27 16:00"

BAND_TO_FREQ_MHZ = {
    1: 0.45, 2: 0.70, 3: 0.90, 4: 1.31,
    5: 2.20, 6: 3.93, 7: 4.70, 8: 6.55, 9: 9.18,
}


class Target:
    def __init__(self, name, kind, ra_icrs=None, dec_icrs=None, body_name=None):
        self.name = name
        self.kind = kind  # "fixed", "planet", "earth", "random"
        self.ra_icrs = ra_icrs
        self.dec_icrs = dec_icrs
        self.body_name = body_name


def band_to_freq_mhz(band):
    if np.isscalar(band):
        return BAND_TO_FREQ_MHZ.get(int(band), np.nan)
    band = np.asarray(band)
    return np.vectorize(lambda b: BAND_TO_FREQ_MHZ.get(int(b), np.nan))(band)


def sanitize_name(name):
    return name.lower().replace(" ", "_").replace("-", "_")


def build_default_targets():
    # Fixed point sources in ICRS (J2000). Converted to FK4 B1950 for analysis.
    fixed_sources = [
        Target("Fornax-A", "fixed", "03h22m41.7s", "-37d12m30s"),
        Target("Cygnus-A", "fixed", "19h59m28.356s", "+40d44m02.1s"),
        Target("Sag-A", "fixed", "17h45m40.0409s", "-29d00m28.118s"),
    ]

    planets = [
        Target("Mercury", "planet", body_name="mercury"),
        Target("Venus", "planet", body_name="venus"),
        Target("Mars", "planet", body_name="mars"),
        Target("Jupiter", "planet", body_name="jupiter"),
        Target("Saturn", "planet", body_name="saturn"),
        Target("Uranus", "planet", body_name="uranus"),
        Target("Neptune", "planet", body_name="neptune"),
    ]

    earth = [Target("Earth", "earth")]
    return fixed_sources + planets + earth


def build_random_targets(n_random, seed=12345):
    rng = np.random.default_rng(seed)
    ra = rng.uniform(0.0, 360.0, size=n_random)
    sin_dec = rng.uniform(-1.0, 1.0, size=n_random)
    dec = np.degrees(np.arcsin(sin_dec))

    targets = []
    for i in range(n_random):
        targets.append(
            Target(
                f"Random-{i:03d}",
                "random",
                ra_icrs=f"{ra[i]:.6f}d",
                dec_icrs=f"{dec[i]:.6f}d",
            )
        )
    return targets


def time_to_grid_index(times, grid_start, step_seconds, n_grid):
    seconds = (times - grid_start) / np.timedelta64(1, "s")
    idx = np.rint(seconds / step_seconds).astype(np.int64)
    idx = np.clip(idx, 0, n_grid - 1)
    return idx


def build_planet_ephemeris_grid(start_time, end_time, step_minutes, planet_targets):
    start = pd.to_datetime(start_time)
    end = pd.to_datetime(end_time)
    step = pd.Timedelta(minutes=step_minutes)
    times = pd.date_range(start, end, freq=step)
    times_astropy = Time(times.to_pydatetime())

    planet_vecs = {}
    for target in planet_targets:
        planet = get_body(target.body_name, times_astropy).transform_to(FK4(equinox="B1950"))
        planet_vec = planet.cartesian.xyz.to(u.km).value.T
        planet_vecs[target.name] = planet_vec

    return times, planet_vecs


def load_data(path, start_time, end_time, usecols=None):
    df = pd.read_csv(path, usecols=usecols)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time")
    df = df[(df["time"] >= pd.to_datetime(start_time)) & (df["time"] <= pd.to_datetime(end_time))]
    df = df.set_index("time")
    return df


def build_geometry_frame(df):
    geom_cols = [
        "position_x", "position_y", "position_z",
        "earth_unit_vector_x", "earth_unit_vector_y", "earth_unit_vector_z",
    ]
    geom_df = df[geom_cols].groupby(level=0).first()
    return geom_df


def ra_dec_to_unit_vector(ra_icrs, dec_icrs):
    src_icrs = SkyCoord(ra=ra_icrs, dec=dec_icrs, frame="icrs")
    src_b1950 = src_icrs.transform_to(FK4(equinox="B1950"))
    uvec = src_b1950.cartesian.xyz.value
    return uvec / np.linalg.norm(uvec)


def _moon_angular_radius_deg(dist_orb_moon):
    ratio = np.clip(MOON_RADIUS_KM / dist_orb_moon, 0.0, 1.0)
    return np.degrees(np.arcsin(ratio))


def compute_occultation_fixed(geom_df, source_uvec):
    orb_pos = geom_df[["position_x", "position_y", "position_z"]].values
    vec_orb_moon = -orb_pos
    dist_orb_moon = np.linalg.norm(vec_orb_moon, axis=1)

    dot = np.einsum("ij,j->i", vec_orb_moon, source_uvec)
    cos_theta = np.clip(dot / dist_orb_moon, -1.0, 1.0)
    separation_deg = np.degrees(np.arccos(cos_theta))

    moon_ang_deg = _moon_angular_radius_deg(dist_orb_moon)
    is_blocked = separation_deg < moon_ang_deg

    return pd.DataFrame(
        {
            "separation_angle_deg": separation_deg,
            "is_blocked": is_blocked,
        },
        index=geom_df.index,
    )


def compute_occultation_planet(geom_df, body_name):
    orb_pos = geom_df[["position_x", "position_y", "position_z"]].values
    vec_orb_moon = -orb_pos
    dist_orb_moon = np.linalg.norm(vec_orb_moon, axis=1)

    earth_uvec = geom_df[["earth_unit_vector_x", "earth_unit_vector_y", "earth_unit_vector_z"]].values
    vec_moon_earth = earth_uvec * DIST_MOON_EARTH_KM

    times = Time(geom_df.index.to_pydatetime())
    planet_gcrs = get_body(body_name, times)
    planet_b1950 = planet_gcrs.transform_to(FK4(equinox="B1950"))

    planet_vec = planet_b1950.cartesian.xyz.to(u.km).value.T
    vec_moon_planet = planet_vec + vec_moon_earth

    vec_orb_planet = vec_moon_planet - orb_pos
    norm_orb_planet = np.linalg.norm(vec_orb_planet, axis=1)

    dot = np.einsum("ij,ij->i", vec_orb_moon, vec_orb_planet)
    cos_theta = np.clip(dot / (dist_orb_moon * norm_orb_planet), -1.0, 1.0)
    separation_deg = np.degrees(np.arccos(cos_theta))

    moon_ang_deg = _moon_angular_radius_deg(dist_orb_moon)
    is_blocked = separation_deg < moon_ang_deg

    return pd.DataFrame(
        {
            "separation_angle_deg": separation_deg,
            "is_blocked": is_blocked,
        },
        index=geom_df.index,
    )


def compute_occultation_planet_grid(geom_df, planet_vecs, grid_start, step_seconds, body_name):
    orb_pos = geom_df[["position_x", "position_y", "position_z"]].values
    vec_orb_moon = -orb_pos
    dist_orb_moon = np.linalg.norm(vec_orb_moon, axis=1)

    earth_uvec = geom_df[["earth_unit_vector_x", "earth_unit_vector_y", "earth_unit_vector_z"]].values
    vec_moon_earth = earth_uvec * DIST_MOON_EARTH_KM

    times = geom_df.index
    n_grid = planet_vecs[body_name].shape[0]
    idx = time_to_grid_index(times, grid_start, step_seconds, n_grid)

    moon_to_planet = planet_vecs[body_name][idx] + vec_moon_earth
    vec_orb_planet = moon_to_planet - orb_pos
    norm_orb_planet = np.linalg.norm(vec_orb_planet, axis=1)

    dot = np.einsum("ij,ij->i", vec_orb_moon, vec_orb_planet)
    cos_theta = np.clip(dot / (dist_orb_moon * norm_orb_planet), -1.0, 1.0)
    separation_deg = np.degrees(np.arccos(cos_theta))

    moon_ang_deg = _moon_angular_radius_deg(dist_orb_moon)
    is_blocked = separation_deg < moon_ang_deg

    return pd.DataFrame(
        {
            "separation_angle_deg": separation_deg,
            "is_blocked": is_blocked,
        },
        index=geom_df.index,
    )


def compute_occultation_earth(geom_df):
    orb_pos = geom_df[["position_x", "position_y", "position_z"]].values
    vec_orb_moon = -orb_pos
    dist_orb_moon = np.linalg.norm(vec_orb_moon, axis=1)

    earth_uvec = geom_df[["earth_unit_vector_x", "earth_unit_vector_y", "earth_unit_vector_z"]].values
    vec_moon_earth = earth_uvec * DIST_MOON_EARTH_KM
    vec_orb_earth = vec_moon_earth - orb_pos

    norm_orb_earth = np.linalg.norm(vec_orb_earth, axis=1)
    dot = np.einsum("ij,ij->i", vec_orb_moon, vec_orb_earth)
    cos_theta = np.clip(dot / (dist_orb_moon * norm_orb_earth), -1.0, 1.0)
    separation_deg = np.degrees(np.arccos(cos_theta))

    moon_ang_deg = _moon_angular_radius_deg(dist_orb_moon)
    is_blocked = separation_deg < moon_ang_deg

    return pd.DataFrame(
        {
            "separation_angle_deg": separation_deg,
            "is_blocked": is_blocked,
        },
        index=geom_df.index,
    )


def find_discrete_events(is_blocked, max_time_gap_seconds=None):
    is_blocked = is_blocked.astype(bool)
    prev_blocked = is_blocked.shift(1).fillna(is_blocked.iloc[0])

    ingress_mask = (~prev_blocked) & is_blocked
    egress_mask = prev_blocked & (~is_blocked)
    transitions = ingress_mask | egress_mask

    times = is_blocked.index
    locs = np.where(transitions.values)[0]
    events = []

    for loc in locs:
        if loc == 0:
            continue

        t_pre = times[loc - 1]
        t_post = times[loc]
        gap_seconds = (t_post - t_pre).total_seconds()

        if max_time_gap_seconds is not None and gap_seconds > max_time_gap_seconds:
            continue

        event_type = "Ingress" if ingress_mask.iloc[loc] else "Egress"
        events.append(
            {
                "event_type": event_type,
                "pre_event_time": t_pre,
                "post_event_time": t_post,
                "gap_seconds": gap_seconds,
            }
        )

    return pd.DataFrame(events)


def compute_limb_mask(
    geom_df,
    sun_limb_threshold,
    earth_limb_threshold,
    moon_radius_km=MOON_RADIUS_KM,
):
    orb_pos = geom_df[["position_x", "position_y", "position_z"]].values
    vec_orb_moon = -orb_pos
    dist_orb_moon = np.linalg.norm(vec_orb_moon, axis=1)

    earth_uvec = geom_df[["earth_unit_vector_x", "earth_unit_vector_y", "earth_unit_vector_z"]].values
    vec_moon_earth = earth_uvec * DIST_MOON_EARTH_KM
    vec_orb_earth = vec_moon_earth - orb_pos
    norm_orb_earth = np.linalg.norm(vec_orb_earth, axis=1)
    dot_earth = np.einsum("ij,ij->i", vec_orb_moon, vec_orb_earth)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos_theta_earth = dot_earth / (dist_orb_moon * norm_orb_earth)
    cos_theta_earth = np.nan_to_num(cos_theta_earth, nan=1.0, posinf=1.0, neginf=-1.0)
    theta_earth = np.degrees(np.arccos(np.clip(cos_theta_earth, -1.0, 1.0)))

    moon_ang_radius = np.degrees(np.arcsin(np.clip(moon_radius_km / dist_orb_moon, 0.0, 1.0)))
    earth_limb_dist = theta_earth - moon_ang_radius

    times = Time(geom_df.index.to_pydatetime())
    sun_gcrs = get_sun(times)
    sun_b1950 = sun_gcrs.transform_to(FK4(equinox="B1950"))
    sun_vec = sun_b1950.cartesian.xyz.to(u.km).value.T
    vec_moon_sun = sun_vec + vec_moon_earth
    vec_orb_sun = vec_moon_sun - orb_pos
    norm_orb_sun = np.linalg.norm(vec_orb_sun, axis=1)
    dot_sun = np.einsum("ij,ij->i", vec_orb_moon, vec_orb_sun)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos_theta_sun = dot_sun / (dist_orb_moon * norm_orb_sun)
    cos_theta_sun = np.nan_to_num(cos_theta_sun, nan=1.0, posinf=1.0, neginf=-1.0)
    theta_sun = np.degrees(np.arccos(np.clip(cos_theta_sun, -1.0, 1.0)))
    sun_limb_dist = theta_sun - moon_ang_radius

    limb_ok = (sun_limb_dist >= sun_limb_threshold) & (earth_limb_dist >= earth_limb_threshold)
    return pd.Series(limb_ok, index=geom_df.index)


def filter_events_by_geometry(
    geom_df,
    events_df,
    sun_limb_threshold=3.0,
    earth_limb_threshold=5.0,
    moon_radius_km=MOON_RADIUS_KM,
    limb_ok=None,
):
    if events_df.empty:
        return events_df.copy()

    if limb_ok is None:
        limb_ok = compute_limb_mask(
            geom_df,
            sun_limb_threshold=sun_limb_threshold,
            earth_limb_threshold=earth_limb_threshold,
            moon_radius_km=moon_radius_km,
        )

    mask = limb_ok.reindex(events_df["pre_event_time"]).fillna(False).values
    return events_df.loc[mask].copy()


def sigma_clip_stats(data, sigma=5, iterations=3):
    cleaned = np.array(data, dtype=float)
    if len(cleaned) == 0:
        return np.nan, np.nan, 0

    for _ in range(iterations):
        if len(cleaned) < 2:
            break
        median_val = np.median(cleaned)
        std_val = np.std(cleaned)
        if std_val == 0:
            break
        lower = median_val - (sigma * std_val)
        upper = median_val + (sigma * std_val)
        cleaned = cleaned[(cleaned >= lower) & (cleaned <= upper)]

    if len(cleaned) == 0:
        return np.nan, np.nan, 0

    return np.median(cleaned), np.std(cleaned), len(cleaned)


def calculate_event_significance(
    df,
    events_df,
    data_col,
    frequency_col="frequency_band",
    window_minutes=3.0,
    min_samples=5,
):
    results = []

    window_delta = pd.Timedelta(minutes=window_minutes)

    for _, event in events_df.iterrows():
        t_pre = event["pre_event_time"]
        t_post = event["post_event_time"]

        pre_mask = (df.index >= t_pre - window_delta) & (df.index <= t_pre)
        post_mask = (df.index >= t_post) & (df.index <= t_post + window_delta)

        pre_data = df.loc[pre_mask]
        post_data = df.loc[post_mask]

        if pre_data.empty and post_data.empty:
            continue

        if frequency_col and frequency_col in df.columns:
            bands = set(pre_data[frequency_col].unique()) | set(post_data[frequency_col].unique())
        else:
            bands = {"all"}

        for band in bands:
            row = {
                "event_type": event["event_type"],
                "pre_event_time": t_pre,
                "post_event_time": t_post,
                "frequency_band": band,
            }

            if frequency_col and frequency_col in df.columns:
                pre_vals = pre_data[pre_data[frequency_col] == band][data_col].dropna().values
                post_vals = post_data[post_data[frequency_col] == band][data_col].dropna().values
            else:
                pre_vals = pre_data[data_col].dropna().values
                post_vals = post_data[data_col].dropna().values

            mu_pre, sig_pre, n_pre = sigma_clip_stats(pre_vals)
            mu_post, sig_post, n_post = sigma_clip_stats(post_vals)

            row["pre_median"] = mu_pre
            row["post_median"] = mu_post

            if event["event_type"] == "Ingress":
                vis_mu, vis_sig, vis_n = mu_pre, sig_pre, n_pre
                blk_mu, blk_sig, blk_n = mu_post, sig_post, n_post
            else:
                vis_mu, vis_sig, vis_n = mu_post, sig_post, n_post
                blk_mu, blk_sig, blk_n = mu_pre, sig_pre, n_pre

            if np.isnan(vis_mu) or np.isnan(blk_mu) or vis_n < min_samples or blk_n < min_samples:
                row["dip_z_score"] = np.nan
                row["ttest_pvalue"] = np.nan
            else:
                diff = vis_mu - blk_mu
                se_diff = np.sqrt((vis_sig ** 2 / vis_n) + (blk_sig ** 2 / blk_n))
                row["dip_z_score"] = 0.0 if se_diff == 0 else diff / se_diff
                try:
                    _, p_val = stats.ttest_ind(pre_vals, post_vals, equal_var=False, nan_policy="omit")
                    row["ttest_pvalue"] = p_val
                except Exception:
                    row["ttest_pvalue"] = np.nan

            results.append(row)

    return pd.DataFrame(results)


def aggregate_significance(significance_df):
    rows = []
    if significance_df.empty:
        return pd.DataFrame(rows)

    grouped = significance_df.groupby(["frequency_band", "event_type"])
    for (band, event_type), group in grouped:
        z_vals = group["dip_z_score"].dropna()
        if len(z_vals) == 0:
            combined_z = np.nan
            mean_z = np.nan
        else:
            combined_z = np.sum(z_vals) / np.sqrt(len(z_vals))
            mean_z = np.mean(z_vals)
        rows.append(
            {
                "frequency_band": band,
                "event_type": event_type,
                "n_events": len(z_vals),
                "combined_z": combined_z,
                "mean_z": mean_z,
            }
        )
    return pd.DataFrame(rows)


def aggregate_combined_z(significance_df):
    rows = []
    if significance_df.empty:
        return pd.DataFrame(rows)

    for band, band_df in significance_df.groupby("frequency_band"):
        z_vals = band_df["dip_z_score"].dropna().values
        if len(z_vals) == 0:
            combined_z = np.nan
            mean_z = np.nan
        else:
            combined_z = np.sum(z_vals) / np.sqrt(len(z_vals))
            mean_z = np.mean(z_vals)
        rows.append(
            {
                "frequency_band": band,
                "frequency_mhz": band_to_freq_mhz(band),
                "n_events": len(z_vals),
                "combined_z": combined_z,
                "mean_z": mean_z,
            }
        )

    return pd.DataFrame(rows)


def subsample_events(events_df, stride=1, max_events=None):
    if events_df.empty:
        return events_df
    if stride is None or stride < 1:
        stride = 1
    events_sorted = events_df.sort_values("pre_event_time")
    if stride > 1:
        events_sorted = events_sorted.iloc[::stride]
    if max_events is not None and max_events > 0:
        events_sorted = events_sorted.iloc[:max_events]
    return events_sorted


def plot_zscore_histograms(significance_df, output_path, title_prefix, by_band=True, bins=30):
    if significance_df.empty:
        return

    if by_band:
        bands = list(significance_df["frequency_band"].unique())
    else:
        bands = ["all"]

    n_bands = len(bands)
    fig, axes = plt.subplots(n_bands, 1, figsize=(10, 6 * n_bands), constrained_layout=True)
    if n_bands == 1:
        axes = [axes]

    for ax, band in zip(axes, bands):
        if by_band:
            band_data = significance_df[significance_df["frequency_band"] == band]
            if isinstance(band, (int, np.integer, float, np.floating)):
                band_label = f"{band_to_freq_mhz(band):.2f} MHz"
            else:
                band_label = str(band)
        else:
            band_data = significance_df
            band_label = "Combined Bands"

        ingress = band_data[band_data["event_type"] == "Ingress"]["dip_z_score"].dropna()
        egress = band_data[band_data["event_type"] == "Egress"]["dip_z_score"].dropna()

        ax.hist(ingress, bins=bins, density=True, alpha=0.5, color="blue",
                label=f"Ingress (N={len(ingress)})")
        ax.hist(egress, bins=bins, density=True, alpha=0.5, color="orange",
                label=f"Egress (N={len(egress)})")

        all_z = pd.concat([ingress, egress])
        if not all_z.empty:
            x_min, x_max = min(all_z.min(), -4), max(all_z.max(), 4)
            x = np.linspace(x_min, x_max, 200)
            ax.plot(x, stats.norm.pdf(x, 0, 1), "k--", linewidth=2, label="Null N(0,1)")

        ax.set_title(f"{title_prefix} - {band_label}")
        ax.set_xlabel("Dip Z-Score (Positive = Signal Drop)")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Per AGENTS.md: create clearly named plots in subfolders.
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def select_targets(targets, selected_names):
    if not selected_names:
        return targets
    selected = {name.lower() for name in selected_names}
    return [t for t in targets if t.name.lower() in selected]


def analyze_target(
    target,
    data_df,
    geom_df,
    output_dir,
    data_cols,
    window_minutes,
    max_time_gap_seconds,
    sun_limb_threshold,
    earth_limb_threshold,
    combine_bands,
    min_samples,
    planet_grid=None,
    limb_ok=None,
    write_outputs=True,
    event_stride=1,
    max_events=None,
    random_event_stride=1,
    random_max_events=None,
):
    if target.kind == "fixed":
        uvec = ra_dec_to_unit_vector(target.ra_icrs, target.dec_icrs)
        occ_df = compute_occultation_fixed(geom_df, uvec)
    elif target.kind == "random":
        uvec = ra_dec_to_unit_vector(target.ra_icrs, target.dec_icrs)
        occ_df = compute_occultation_fixed(geom_df, uvec)
    elif target.kind == "planet":
        if planet_grid is None:
            occ_df = compute_occultation_planet(geom_df, target.body_name)
        else:
            occ_df = compute_occultation_planet_grid(
                geom_df,
                planet_grid["planet_vecs"],
                planet_grid["grid_start"],
                planet_grid["step_seconds"],
                target.name,
            )
    elif target.kind == "earth":
        occ_df = compute_occultation_earth(geom_df)
    else:
        raise ValueError(f"Unsupported target kind: {target.kind}")

    events = find_discrete_events(occ_df["is_blocked"], max_time_gap_seconds=max_time_gap_seconds)
    filtered_events = filter_events_by_geometry(
        geom_df,
        events,
        sun_limb_threshold=sun_limb_threshold,
        earth_limb_threshold=earth_limb_threshold,
        limb_ok=limb_ok,
    )
    if target.kind == "random":
        filtered_events = subsample_events(
            filtered_events,
            stride=random_event_stride,
            max_events=random_max_events,
        )
    else:
        filtered_events = subsample_events(
            filtered_events,
            stride=event_stride,
            max_events=max_events,
        )

    target_dir = os.path.join(output_dir, sanitize_name(target.name))
    tables_dir = os.path.join(target_dir, "tables")
    plots_dir = os.path.join(target_dir, "plots")
    if write_outputs:
        os.makedirs(tables_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)

    if write_outputs:
        events.to_csv(os.path.join(tables_dir, "events_all.csv"), index=False)
        filtered_events.to_csv(os.path.join(tables_dir, "events_filtered.csv"), index=False)

    aggregate_rows = []
    combined_z_rows = []

    for data_col in data_cols:
        if data_col not in data_df.columns:
            continue

        per_band = calculate_event_significance(
            data_df,
            filtered_events,
            data_col=data_col,
            frequency_col="frequency_band",
            window_minutes=window_minutes,
            min_samples=min_samples,
        )
        per_band["target"] = target.name
        per_band["data_col"] = data_col
        per_band["analysis_mode"] = "per_band"
        if write_outputs:
            per_band.to_csv(os.path.join(tables_dir, f"significance_per_band_{data_col}.csv"), index=False)

        per_band_agg = aggregate_significance(per_band)
        per_band_agg["target"] = target.name
        per_band_agg["data_col"] = data_col
        per_band_agg["analysis_mode"] = "per_band"
        if write_outputs:
            per_band_agg.to_csv(os.path.join(tables_dir, f"aggregate_per_band_{data_col}.csv"), index=False)
        aggregate_rows.append(per_band_agg)

        if write_outputs:
            plot_zscore_histograms(
                per_band,
                os.path.join(plots_dir, f"zscores_per_band_{data_col}.png"),
                f"{target.name} ({data_col})",
                by_band=True,
            )

        combined_z = aggregate_combined_z(per_band)
        if not combined_z.empty:
            combined_z["target"] = target.name
            combined_z["data_col"] = data_col
            combined_z["kind"] = target.kind
            combined_z_rows.append(combined_z)
            if write_outputs:
                combined_z.to_csv(
                    os.path.join(tables_dir, f"event_combined_z_{data_col}.csv"),
                    index=False,
                )

        if combine_bands and write_outputs:
            combined = calculate_event_significance(
                data_df,
                filtered_events,
                data_col=data_col,
                frequency_col=None,
                window_minutes=window_minutes,
                min_samples=min_samples,
            )
            combined["target"] = target.name
            combined["data_col"] = data_col
            combined["analysis_mode"] = "combined"
            combined.to_csv(os.path.join(tables_dir, f"significance_combined_{data_col}.csv"), index=False)

            combined_agg = aggregate_significance(combined)
            combined_agg["target"] = target.name
            combined_agg["data_col"] = data_col
            combined_agg["analysis_mode"] = "combined"
            combined_agg.to_csv(os.path.join(tables_dir, f"aggregate_combined_{data_col}.csv"), index=False)
            aggregate_rows.append(combined_agg)

            plot_zscore_histograms(
                combined,
                os.path.join(plots_dir, f"zscores_combined_{data_col}.png"),
                f"{target.name} ({data_col})",
                by_band=False,
            )

    if aggregate_rows:
        all_agg = pd.concat(aggregate_rows, ignore_index=True)
    else:
        all_agg = pd.DataFrame()

    if combined_z_rows:
        combined_z_df = pd.concat(combined_z_rows, ignore_index=True)
    else:
        combined_z_df = pd.DataFrame()

    return all_agg, combined_z_df


def parse_args():
    parser = argparse.ArgumentParser(description="RAE2 occultation detection pipeline")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="Path to interpolatedRAE2MasterFile.csv")
    parser.add_argument("--output-dir", default="RAE2Agent/output", help="Output directory for tables/plots")
    parser.add_argument("--start", default=DEFAULT_START, help="Start datetime (inclusive)")
    parser.add_argument("--end", default=DEFAULT_END, help="End datetime (inclusive)")
    parser.add_argument("--data-cols", nargs="+", default=["rv2_coarse"], help="Data columns to analyze")
    parser.add_argument("--window-minutes", type=float, default=3.0, help="Pre/post window size in minutes")
    parser.add_argument("--max-time-gap-seconds", type=float, default=180.0, help="Max gap to accept transitions")
    parser.add_argument("--sun-limb-threshold", type=float, default=3.0, help="Sun limb threshold (deg)")
    parser.add_argument("--earth-limb-threshold", type=float, default=5.0, help="Earth limb threshold (deg)")
    parser.add_argument("--min-samples", type=int, default=5, help="Minimum samples per window for z-scores")
    parser.add_argument("--random-sky", type=int, default=200, help="Number of random sky points")
    parser.add_argument("--random-seed", type=int, default=12345, help="Random seed for sky points")
    parser.add_argument("--planet-step-minutes", type=int, default=30, help="Ephemeris grid step in minutes")
    parser.add_argument("--no-planets", action="store_true", help="Skip planets")
    parser.add_argument("--event-stride", type=int, default=1, help="Use every Nth event (real targets)")
    parser.add_argument("--max-events", type=int, default=0, help="Limit number of events per real target (0 disables)")
    parser.add_argument("--random-event-stride", type=int, default=5, help="Use every Nth event (random targets)")
    parser.add_argument("--random-max-events", type=int, default=200, help="Limit events per random target (0 disables)")
    parser.add_argument("--n-processes", type=int, default=1, help="Parallel workers for target processing")
    parser.add_argument("--targets", nargs="*", help="Optional subset of targets by name")
    parser.add_argument("--no-combine-bands", action="store_true", help="Disable combined-band analysis")
    return parser.parse_args()


_WORKER_CFG = {}


def _init_worker(cfg):
    global _WORKER_CFG
    _WORKER_CFG = cfg


def _analyze_target_worker(target):
    cfg = _WORKER_CFG
    return analyze_target(
        target=target,
        data_df=cfg["data_df"],
        geom_df=cfg["geom_df"],
        output_dir=cfg["output_dir"],
        data_cols=cfg["data_cols"],
        window_minutes=cfg["window_minutes"],
        max_time_gap_seconds=cfg["max_time_gap_seconds"],
        sun_limb_threshold=cfg["sun_limb_threshold"],
        earth_limb_threshold=cfg["earth_limb_threshold"],
        combine_bands=cfg["combine_bands"],
        min_samples=cfg["min_samples"],
        planet_grid=cfg["planet_grid"],
        limb_ok=cfg["limb_ok"],
        write_outputs=(target.kind != "random"),
        event_stride=cfg["event_stride"],
        max_events=cfg["max_events"],
        random_event_stride=cfg["random_event_stride"],
        random_max_events=cfg["random_max_events"],
    )


def main():
    args = parse_args()

    usecols = [
        "time",
        "frequency_band",
        "position_x",
        "position_y",
        "position_z",
        "earth_unit_vector_x",
        "earth_unit_vector_y",
        "earth_unit_vector_z",
    ] + args.data_cols
    data_df = load_data(args.data, args.start, args.end, usecols=usecols)
    geom_df = build_geometry_frame(data_df)
    limb_ok = compute_limb_mask(
        geom_df,
        sun_limb_threshold=args.sun_limb_threshold,
        earth_limb_threshold=args.earth_limb_threshold,
    )

    base_targets = build_default_targets()
    if args.no_planets:
        base_targets = [t for t in base_targets if t.kind != "planet"]

    targets = select_targets(base_targets, args.targets)
    if not targets:
        raise SystemExit("No valid targets selected.")

    random_targets = build_random_targets(args.random_sky, seed=args.random_seed) if args.random_sky > 0 else []
    all_targets = targets + random_targets

    planet_targets = [t for t in targets if t.kind == "planet"]
    if planet_targets:
        grid_times, planet_vecs = build_planet_ephemeris_grid(
            args.start, args.end, args.planet_step_minutes, planet_targets
        )
        planet_grid = {
            "planet_vecs": planet_vecs,
            "grid_start": grid_times[0],
            "step_seconds": args.planet_step_minutes * 60,
        }
    else:
        planet_grid = None

    os.makedirs(args.output_dir, exist_ok=True)
    summary_rows = []
    combined_rows = []

    cfg = {
        "data_df": data_df,
        "geom_df": geom_df,
        "output_dir": args.output_dir,
        "data_cols": args.data_cols,
        "window_minutes": args.window_minutes,
        "max_time_gap_seconds": args.max_time_gap_seconds,
        "sun_limb_threshold": args.sun_limb_threshold,
        "earth_limb_threshold": args.earth_limb_threshold,
        "combine_bands": not args.no_combine_bands,
        "min_samples": args.min_samples,
        "planet_grid": planet_grid,
        "limb_ok": limb_ok,
        "event_stride": args.event_stride,
        "max_events": args.max_events if args.max_events > 0 else None,
        "random_event_stride": args.random_event_stride,
        "random_max_events": args.random_max_events if args.random_max_events > 0 else None,
    }

    if args.n_processes and args.n_processes > 1:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=args.n_processes, initializer=_init_worker, initargs=(cfg,)) as pool:
            results = pool.map(_analyze_target_worker, all_targets)
    else:
        _init_worker(cfg)
        results = [_analyze_target_worker(t) for t in all_targets]

    for target, result in zip(all_targets, results):
        agg_df, combined_df = result
        if target.kind != "random" and not agg_df.empty:
            summary_rows.append(agg_df)
        if not combined_df.empty:
            combined_rows.append(combined_df)

    if summary_rows:
        summary = pd.concat(summary_rows, ignore_index=True)
        summary.to_csv(os.path.join(args.output_dir, "summary_aggregates.csv"), index=False)

    if combined_rows:
        combined_all = pd.concat(combined_rows, ignore_index=True)
        for data_col in args.data_cols:
            subset = combined_all[combined_all["data_col"] == data_col]
            if subset.empty:
                continue
            subset.to_csv(
                os.path.join(args.output_dir, f"event_combined_z_all_{data_col}.csv"),
                index=False,
            )

            rand = subset[subset["kind"] == "random"]
            real = subset[subset["kind"] != "random"]
            rows = []
            for band, band_df in real.groupby("frequency_band"):
                rand_band = rand[rand["frequency_band"] == band]["combined_z"].dropna()
                if rand_band.empty:
                    continue
                for _, row in band_df.iterrows():
                    z = row["combined_z"]
                    if np.isnan(z):
                        continue
                    pct = float((rand_band < z).mean())
                    rows.append(
                        {
                            "target": row["target"],
                            "kind": row["kind"],
                            "data_col": data_col,
                            "frequency_band": band,
                            "frequency_mhz": row["frequency_mhz"],
                            "combined_z": z,
                            "random_percentile": pct,
                            "random_n": len(rand_band),
                        }
                    )

            percentile_df = pd.DataFrame(rows)
            percentile_df.to_csv(
                os.path.join(args.output_dir, f"event_random_percentile_per_band_{data_col}.csv"),
                index=False,
            )

            plots_dir = os.path.join(args.output_dir, "plots", data_col)
            os.makedirs(plots_dir, exist_ok=True)
            for target in sorted(percentile_df["target"].unique()):
                tdf = percentile_df[percentile_df["target"] == target].sort_values("frequency_mhz")
                if tdf.empty:
                    continue
                plt.figure(figsize=(7, 4))
                plt.plot(
                    tdf["frequency_mhz"].values,
                    tdf["random_percentile"].values,
                    marker="o",
                    linewidth=1.5,
                )
                plt.ylim(0, 1)
                plt.xlabel("Frequency (MHz)")
                plt.ylabel("Random Percentile")
                plt.title(f"{target} {data_col} Random Percentile vs Frequency")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(plots_dir, f"event_random_percentile_{sanitize_name(target)}.png"),
                    dpi=150,
                )
                plt.close()


if __name__ == "__main__":
    main()
