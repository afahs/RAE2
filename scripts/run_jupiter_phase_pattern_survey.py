#!/usr/bin/env python
"""Jupiter phase-pattern survey using all Ryle-Vonberg samples.

This is not an occultation-event detector. It asks whether raw radiometer
samples organize by Jupiter observer geometry:

- System-III central meridian longitude proxy;
- Io phase, with both normal and reversed historical conventions represented;
- Jupiter visible/occulted by the Moon;
- Earth visible/occulted by the Moon;
- lower and upper V antennas.

The plotted statistic is deliberately simple: per-day/channel normalized
log-power and the fraction of samples in the high-power tail. The daily
normalization reduces long-timescale receiver/background changes without
fitting an occultation baseline.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rylevonberg.table_io import read_table

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.constants import FREQUENCY_MAP_MHZ  # noqa: E402
from rylevonberg.util import ensure_dir, software_versions, write_json  # noqa: E402
from scripts.run_low_frequency_external_selectors import (  # noqa: E402
    MASER_MAPS,
    load_probability_map,
    sample_map_score,
)
from scripts.run_solar_burst_spice_jupiter_visibility import (  # noqa: E402
    _signed_phase_deg,
    _spice_time_strings,
    ensure_spice_kernels,
    spice,
)


DEFAULT_CLEAN = ROOT / "outputs/control_survey_earth_sun_postnov1974_v2/01_ingest/cleaned_timeseries.csv"
DEFAULT_JUPITER_STATES = ROOT / "outputs/planetary_confirmation_survey_science_baseline_v2/events/jupiter_limb_visibility_states.csv"
DEFAULT_EARTH_STATES = ROOT / "outputs/planetary_confirmation_survey_science_baseline_v2/events/earth_limb_visibility_states.csv"
DEFAULT_OUT = ROOT / "outputs/jupiter_phase_pattern_survey_v1"

ANTENNA_LABEL = {"rv1_coarse": "upper V", "rv2_coarse": "lower V"}
ANTENNA_COLOR = {"rv1_coarse": "#4c78a8", "rv2_coarse": "#d95f02"}
REGIME_LABEL = {
    "jupiter_visible": "Jupiter visible",
    "jupiter_visible_earth_occulted": "Jupiter visible, Earth occulted",
    "jupiter_occulted_control": "Jupiter occulted control",
}


def _read(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return read_table(path, low_memory=False, **kwargs)


def _load_visibility_states(jupiter_states: Path, earth_states: Path) -> pd.DataFrame:
    jup = _read(jupiter_states, parse_dates=["time"])
    earth = _read(earth_states, parse_dates=["time"])
    states = jup[["time", "visible_by_moon", "limb_angle_deg"]].rename(
        columns={
            "visible_by_moon": "jupiter_visible_by_moon",
            "limb_angle_deg": "jupiter_limb_angle_deg",
        }
    )
    states = states.merge(
        earth[["time", "visible_by_moon", "limb_angle_deg"]].rename(
            columns={
                "visible_by_moon": "earth_visible_by_moon",
                "limb_angle_deg": "earth_limb_angle_deg",
            }
        ),
        on="time",
        how="inner",
    )
    return states.sort_values("time").drop_duplicates("time").reset_index(drop=True)


def _annotate_spice_grid(states: pd.DataFrame, out_dir: Path, refresh: bool = False) -> pd.DataFrame:
    if spice is None:
        raise RuntimeError("spiceypy is required for Jupiter phase-pattern survey")
    out = states.copy()
    kernels = ensure_spice_kernels(out_dir, refresh=refresh)
    spice.kclear()
    for name in ["naif0012.tls", "pck00010.tpc", "jup100.bsp", "jup348.bsp"]:
        spice.furnsh(str(kernels[name]))
    try:
        et = np.asarray(spice.str2et(_spice_time_strings(out["time"])), dtype=float)
        earth_vec = np.asarray(spice.spkpos("EARTH", et, "J2000", "NONE", "JUPITER")[0], dtype=float)
        io_vec = np.asarray(spice.spkpos("IO", et, "J2000", "NONE", "JUPITER")[0], dtype=float)
        cml = np.full(len(out), np.nan, dtype=float)
        io_phase = np.full(len(out), np.nan, dtype=float)
        io_phase_reverse = np.full(len(out), np.nan, dtype=float)
        jupiter_range_au = np.linalg.norm(earth_vec, axis=1) / 149_597_870.7
        for idx, (e, ev, iv) in enumerate(zip(et, earth_vec, io_vec)):
            rot = np.asarray(spice.pxform("J2000", "IAU_JUPITER", float(e)), dtype=float)
            earth_body = rot @ ev
            io_body = rot @ iv
            lon_east = np.degrees(np.arctan2(earth_body[1], earth_body[0])) % 360.0
            cml[idx] = (360.0 - lon_east) % 360.0
            phase = float(_signed_phase_deg((-earth_body[:2]).reshape(1, 2), io_body[:2].reshape(1, 2))[0])
            io_phase[idx] = phase
            io_phase_reverse[idx] = (-phase) % 360.0
        out["jupiter_cml_spice_deg"] = cml
        out["io_phase_spice_deg"] = io_phase
        out["io_phase_spice_reverse_deg"] = io_phase_reverse
        out["jupiter_range_au"] = jupiter_range_au
    finally:
        spice.kclear()

    maps = {
        name: load_probability_map(name, meta, out_dir, refresh=refresh)
        for name, meta in MASER_MAPS.items()
    }
    for name, pmap in maps.items():
        normal = sample_map_score(pmap, out["jupiter_cml_spice_deg"], out["io_phase_spice_deg"])
        reversed_phase = sample_map_score(pmap, out["jupiter_cml_spice_deg"], out["io_phase_spice_reverse_deg"])
        stack = np.vstack([normal, reversed_phase])
        score = np.full(stack.shape[1], np.nan, dtype=float)
        finite = np.isfinite(stack).any(axis=0)
        score[finite] = np.nanmax(stack[:, finite], axis=0)
        out[f"{name}_score"] = score
    return out


def _read_clean(clean_path: Path) -> pd.DataFrame:
    usecols = ["time", "frequency_band", "frequency_mhz", "antenna", "power", "is_valid"]
    clean = read_table(clean_path, usecols=usecols, parse_dates=["time"], low_memory=False)
    clean = clean[
        clean["antenna"].astype(str).isin(["rv1_coarse", "rv2_coarse"])
        & clean["is_valid"].astype(bool)
        & pd.to_numeric(clean["power"], errors="coerce").gt(0)
    ].copy()
    clean["power"] = pd.to_numeric(clean["power"], errors="coerce")
    clean = clean.dropna(subset=["power"])
    clean["log_power"] = np.log(clean["power"].to_numpy(dtype=float))
    clean["date"] = clean["time"].dt.floor("D")
    return clean.sort_values("time").reset_index(drop=True)


def _add_daily_channel_normalization(clean: pd.DataFrame) -> pd.DataFrame:
    keys = ["date", "antenna", "frequency_band"]
    stats = (
        clean.groupby(keys, sort=True)["log_power"]
        .quantile([0.25, 0.5, 0.75])
        .unstack()
        .reset_index()
        .rename(columns={0.25: "day_q25", 0.5: "day_median", 0.75: "day_q75"})
    )
    stats["day_sigma"] = (stats["day_q75"] - stats["day_q25"]) / 1.349
    global_stats = (
        clean.groupby(["antenna", "frequency_band"], sort=True)["log_power"]
        .quantile([0.25, 0.5, 0.75])
        .unstack()
        .reset_index()
        .rename(columns={0.25: "global_q25", 0.5: "global_median", 0.75: "global_q75"})
    )
    global_stats["global_sigma"] = (global_stats["global_q75"] - global_stats["global_q25"]) / 1.349
    clean = clean.merge(stats[keys + ["day_median", "day_sigma"]], on=keys, how="left")
    clean = clean.merge(
        global_stats[["antenna", "frequency_band", "global_median", "global_sigma"]],
        on=["antenna", "frequency_band"],
        how="left",
    )
    bad_scale = ~np.isfinite(clean["day_sigma"]) | (clean["day_sigma"] <= 1e-9)
    clean.loc[bad_scale, "day_sigma"] = clean.loc[bad_scale, "global_sigma"]
    bad_center = ~np.isfinite(clean["day_median"])
    clean.loc[bad_center, "day_median"] = clean.loc[bad_center, "global_median"]
    clean["daily_z_log_power"] = (clean["log_power"] - clean["day_median"]) / clean["day_sigma"]
    clean = clean[np.isfinite(clean["daily_z_log_power"])].copy()
    return clean


def _merge_geometry(clean: pd.DataFrame, geom: pd.DataFrame, tolerance_s: float) -> pd.DataFrame:
    geom_cols = [
        "time",
        "jupiter_visible_by_moon",
        "earth_visible_by_moon",
        "jupiter_limb_angle_deg",
        "earth_limb_angle_deg",
        "jupiter_cml_spice_deg",
        "io_phase_spice_deg",
        "io_phase_spice_reverse_deg",
        "jupiter_range_au",
        "maser_zarka_full_score",
        "maser_zarka_io_score",
        "maser_leblanc_1978_score",
    ]
    merged = pd.merge_asof(
        clean.sort_values("time"),
        geom[geom_cols].sort_values("time"),
        on="time",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=float(tolerance_s)),
    )
    merged = merged.dropna(subset=["jupiter_cml_spice_deg", "io_phase_spice_deg"])
    for col in ["jupiter_visible_by_moon", "earth_visible_by_moon"]:
        merged[col] = merged[col].astype(bool)
    return merged


def _bin_phase_samples(samples: pd.DataFrame, phase_bin_deg: float, high_z: float, min_count: int) -> pd.DataFrame:
    out = samples.copy()
    out["cml_bin_deg"] = (np.floor(out["jupiter_cml_spice_deg"] / phase_bin_deg) * phase_bin_deg) + 0.5 * phase_bin_deg
    out["io_bin_deg"] = (np.floor(out["io_phase_spice_deg"] / phase_bin_deg) * phase_bin_deg) + 0.5 * phase_bin_deg
    out["io_reverse_bin_deg"] = (
        np.floor(out["io_phase_spice_reverse_deg"] / phase_bin_deg) * phase_bin_deg
    ) + 0.5 * phase_bin_deg
    out["high_power_tail"] = out["daily_z_log_power"] > float(high_z)

    regimes = {
        "jupiter_visible": out["jupiter_visible_by_moon"],
        "jupiter_visible_earth_occulted": out["jupiter_visible_by_moon"] & ~out["earth_visible_by_moon"],
        "jupiter_occulted_control": ~out["jupiter_visible_by_moon"],
    }
    rows = []
    group_cols = ["antenna", "frequency_band", "frequency_mhz", "cml_bin_deg", "io_bin_deg"]
    for regime, mask in regimes.items():
        sub = out[mask].copy()
        if sub.empty:
            continue
        grouped = sub.groupby(group_cols, sort=True, observed=True)
        agg = grouped.agg(
            n_samples=("daily_z_log_power", "size"),
            median_daily_z=("daily_z_log_power", "median"),
            q90_daily_z=("daily_z_log_power", lambda x: float(np.nanquantile(x, 0.90))),
            high_power_fraction=("high_power_tail", "mean"),
            median_zarka_io_score=("maser_zarka_io_score", "median"),
        ).reset_index()
        agg["regime"] = regime
        agg.loc[agg["n_samples"] < int(min_count), ["median_daily_z", "q90_daily_z", "high_power_fraction"]] = np.nan
        rows.append(agg)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _score_contrast_summary(samples: pd.DataFrame, high_z: float) -> pd.DataFrame:
    rows = []
    work = samples.copy()
    work["high_power_tail"] = work["daily_z_log_power"] > float(high_z)
    visible = work[work["jupiter_visible_by_moon"]].copy()
    occulted = work[~work["jupiter_visible_by_moon"]].copy()
    visible_score = pd.to_numeric(visible["maser_zarka_io_score"], errors="coerce")
    q75 = float(visible_score.quantile(0.75)) if not visible_score.dropna().empty else np.nan
    q25 = float(visible_score.quantile(0.25)) if not visible_score.dropna().empty else np.nan
    regimes = {
        "visible_high_maser_score": visible[visible["maser_zarka_io_score"] >= q75],
        "visible_low_maser_score": visible[visible["maser_zarka_io_score"] <= q25],
        "visible_earth_occulted_high_maser_score": visible[
            (~visible["earth_visible_by_moon"]) & (visible["maser_zarka_io_score"] >= q75)
        ],
        "jupiter_occulted_high_maser_score_control": occulted[occulted["maser_zarka_io_score"] >= q75],
    }
    for (antenna, band, freq), _grp in work.groupby(["antenna", "frequency_band", "frequency_mhz"], sort=True):
        row = {
            "antenna": antenna,
            "frequency_band": int(band),
            "frequency_mhz": float(freq),
            "maser_score_q25_visible": q25,
            "maser_score_q75_visible": q75,
        }
        for name, sub_all in regimes.items():
            sub = sub_all[
                sub_all["antenna"].astype(str).eq(str(antenna))
                & sub_all["frequency_band"].astype(int).eq(int(band))
            ]
            row[f"{name}_n"] = int(len(sub))
            row[f"{name}_median_z"] = float(np.nanmedian(sub["daily_z_log_power"])) if len(sub) else np.nan
            row[f"{name}_high_frac"] = float(np.nanmean(sub["high_power_tail"])) if len(sub) else np.nan
        row["high_minus_low_high_frac"] = (
            row["visible_high_maser_score_high_frac"] - row["visible_low_maser_score_high_frac"]
        )
        row["high_minus_occulted_high_frac"] = (
            row["visible_high_maser_score_high_frac"] - row["jupiter_occulted_high_maser_score_control_high_frac"]
        )
        row["earth_occulted_high_minus_occulted_high_frac"] = (
            row["visible_earth_occulted_high_maser_score_high_frac"]
            - row["jupiter_occulted_high_maser_score_control_high_frac"]
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _phase_map_matrix(summary: pd.DataFrame, antenna: str, band: int, regime: str, value_col: str) -> pd.DataFrame:
    sub = summary[
        summary["antenna"].astype(str).eq(antenna)
        & summary["frequency_band"].astype(int).eq(int(band))
        & summary["regime"].astype(str).eq(regime)
    ].copy()
    if sub.empty:
        return pd.DataFrame()
    return sub.pivot(index="io_bin_deg", columns="cml_bin_deg", values=value_col).sort_index(ascending=True)


def plot_phase_maps(summary: pd.DataFrame, out_dir: Path, regime: str, value_col: str, title_suffix: str) -> list[Path]:
    paths = []
    for antenna in ["rv1_coarse", "rv2_coarse"]:
        fig, axes = plt.subplots(3, 3, figsize=(13.4, 10.8), sharex=True, sharey=True)
        vals = summary[
            summary["antenna"].astype(str).eq(antenna)
            & summary["regime"].astype(str).eq(regime)
        ][value_col].dropna()
        if vals.empty:
            continue
        if value_col == "high_power_fraction":
            vmin, vmax = 0.0, max(0.02, float(vals.quantile(0.98)))
            cmap = "magma"
        else:
            lim = max(0.5, float(np.nanpercentile(np.abs(vals), 98)))
            vmin, vmax = -lim, lim
            cmap = "coolwarm"
        for ax, band in zip(axes.ravel(), sorted(FREQUENCY_MAP_MHZ)):
            mat = _phase_map_matrix(summary, antenna, band, regime, value_col)
            if mat.empty:
                ax.axis("off")
                continue
            im = ax.imshow(
                mat.to_numpy(dtype=float),
                origin="lower",
                extent=[
                    float(mat.columns.min() - 7.5),
                    float(mat.columns.max() + 7.5),
                    float(mat.index.min() - 7.5),
                    float(mat.index.max() + 7.5),
                ],
                aspect="auto",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_title(f"{FREQUENCY_MAP_MHZ[band]:.2f} MHz", fontsize=9)
            ax.grid(True, color="white", alpha=0.18, linewidth=0.45)
        for ax in axes[-1, :]:
            ax.set_xlabel("Jupiter CML proxy (deg)")
        for ax in axes[:, 0]:
            ax.set_ylabel("Io phase (deg)")
        fig.suptitle(
            f"Jupiter phase map: {ANTENNA_LABEL.get(antenna, antenna)}, {REGIME_LABEL.get(regime, regime)}\n"
            f"{title_suffix}; daily channel normalization, all valid samples",
            y=0.993,
        )
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82, pad=0.015)
        cbar.set_label(value_col.replace("_", " "))
        path = out_dir / f"jupiter_phase_map_{value_col}_{regime}_{antenna}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_antenna_score_contrasts(score: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9), sharey=True)
    for ax, metric, title in [
        (axes[0], "high_minus_low_high_frac", "high MASER score - low MASER score"),
        (axes[1], "high_minus_occulted_high_frac", "visible high-score - occulted high-score"),
    ]:
        ax.axhline(0, color="0.6", lw=0.8)
        for antenna, grp in score.groupby("antenna", sort=True):
            grp = grp.sort_values("frequency_mhz")
            ax.plot(
                grp["frequency_mhz"],
                grp[metric],
                marker="o",
                lw=1.6,
                color=ANTENNA_COLOR.get(str(antenna)),
                label=ANTENNA_LABEL.get(str(antenna), str(antenna)),
            )
        ax.set_xscale("log")
        ax.set_xticks(list(FREQUENCY_MAP_MHZ.values()))
        ax.set_xticklabels([f"{v:g}" for v in FREQUENCY_MAP_MHZ.values()], rotation=45, ha="right")
        ax.set_xlabel("frequency (MHz)")
        ax.set_title(title)
        ax.grid(True, color="0.9", lw=0.5)
    axes[0].set_ylabel("difference in high-power-tail fraction")
    axes[1].legend(frameon=False, fontsize=9)
    fig.suptitle("Jupiter phase-pattern contrast by antenna")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = out_dir / "jupiter_antenna_maser_score_contrast_spectrum.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def phase_map_correlations(summary: pd.DataFrame, regime: str = "jupiter_visible") -> pd.DataFrame:
    rows = []
    src = summary[summary["regime"].astype(str).eq(regime)].copy()
    for (antenna, band, freq), grp in src.groupby(["antenna", "frequency_band", "frequency_mhz"], sort=True):
        sub = grp[["high_power_fraction", "median_daily_z", "median_zarka_io_score"]].dropna().copy()
        if len(sub) < 20:
            rows.append(
                {
                    "antenna": antenna,
                    "frequency_band": int(band),
                    "frequency_mhz": float(freq),
                    "regime": regime,
                    "n_phase_bins": int(len(sub)),
                    "spearman_high_frac_vs_maser_score": np.nan,
                    "pearson_high_frac_vs_maser_score": np.nan,
                    "spearman_median_z_vs_maser_score": np.nan,
                }
            )
            continue
        rows.append(
            {
                "antenna": antenna,
                "frequency_band": int(band),
                "frequency_mhz": float(freq),
                "regime": regime,
                "n_phase_bins": int(len(sub)),
                "spearman_high_frac_vs_maser_score": float(
                    sub["high_power_fraction"].corr(sub["median_zarka_io_score"], method="spearman")
                ),
                "pearson_high_frac_vs_maser_score": float(
                    sub["high_power_fraction"].corr(sub["median_zarka_io_score"], method="pearson")
                ),
                "spearman_median_z_vs_maser_score": float(
                    sub["median_daily_z"].corr(sub["median_zarka_io_score"], method="spearman")
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_phase_map_correlations(corr: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9), sharey=True)
    metrics = [
        ("spearman_high_frac_vs_maser_score", "high-tail fraction vs MASER score"),
        ("spearman_median_z_vs_maser_score", "median daily z vs MASER score"),
    ]
    for ax, (metric, title) in zip(axes, metrics):
        ax.axhline(0, color="0.6", lw=0.8)
        for antenna, grp in corr.groupby("antenna", sort=True):
            grp = grp.sort_values("frequency_mhz")
            ax.plot(
                grp["frequency_mhz"],
                grp[metric],
                marker="o",
                lw=1.6,
                color=ANTENNA_COLOR.get(str(antenna)),
                label=ANTENNA_LABEL.get(str(antenna), str(antenna)),
            )
        ax.set_xscale("log")
        ax.set_xticks(list(FREQUENCY_MAP_MHZ.values()))
        ax.set_xticklabels([f"{v:g}" for v in FREQUENCY_MAP_MHZ.values()], rotation=45, ha="right")
        ax.set_xlabel("frequency (MHz)")
        ax.set_title(title)
        ax.grid(True, color="0.9", lw=0.5)
    axes[0].set_ylabel("Spearman correlation across CML/Io bins")
    axes[1].legend(frameon=False, fontsize=9)
    fig.suptitle("Does the observed Jupiter-visible phase map follow the MASER Io-CML map?")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = out_dir / "jupiter_phase_map_maser_score_correlation_spectrum.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_observed_vs_maser_scatter(summary: pd.DataFrame, corr: pd.DataFrame, out_dir: Path, top_n: int = 6) -> Path:
    ranked = corr.dropna(subset=["spearman_high_frac_vs_maser_score"]).copy()
    if ranked.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No valid phase-map correlations", ha="center", va="center")
        path = out_dir / "jupiter_observed_high_tail_vs_maser_score_scatter.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path
    ranked["abs_corr"] = ranked["spearman_high_frac_vs_maser_score"].abs()
    ranked = ranked.sort_values("abs_corr", ascending=False).head(int(top_n))
    fig, axes = plt.subplots(len(ranked), 1, figsize=(8.8, max(7.0, 2.35 * len(ranked))))
    if len(ranked) == 1:
        axes = [axes]
    for ax, (_, row) in zip(axes, ranked.iterrows()):
        sub = summary[
            summary["regime"].astype(str).eq(str(row["regime"]))
            & summary["antenna"].astype(str).eq(str(row["antenna"]))
            & summary["frequency_band"].astype(int).eq(int(row["frequency_band"]))
        ][["high_power_fraction", "median_zarka_io_score", "n_samples"]].dropna()
        ax.scatter(
            sub["median_zarka_io_score"],
            sub["high_power_fraction"],
            s=np.clip(sub["n_samples"] / 18.0, 8, 60),
            alpha=0.55,
            color=ANTENNA_COLOR.get(str(row["antenna"]), "black"),
            edgecolor="none",
        )
        ax.set_title(
            f"{ANTENNA_LABEL.get(str(row['antenna']), row['antenna'])} {float(row['frequency_mhz']):.2f} MHz: "
            f"Spearman r={float(row['spearman_high_frac_vs_maser_score']):.3f}"
        )
        ax.set_xlabel("binned MASER Io-CML map score")
        ax.set_ylabel("high-tail fraction")
        ax.grid(True, color="0.9", lw=0.5)
    fig.suptitle("Observed high-power-tail fraction versus MASER map score")
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    path = out_dir / "jupiter_observed_high_tail_vs_maser_score_scatter.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_io_phase_curves(samples: pd.DataFrame, out_dir: Path, high_z: float, phase_bin_deg: float) -> list[Path]:
    paths = []
    work = samples[samples["jupiter_visible_by_moon"]].copy()
    work["io_bin_deg"] = (np.floor(work["io_phase_spice_deg"] / phase_bin_deg) * phase_bin_deg) + 0.5 * phase_bin_deg
    work["high_power_tail"] = work["daily_z_log_power"] > float(high_z)
    grouped = (
        work.groupby(["antenna", "frequency_band", "frequency_mhz", "io_bin_deg"], sort=True)
        .agg(n_samples=("high_power_tail", "size"), high_power_fraction=("high_power_tail", "mean"))
        .reset_index()
    )
    for antenna, src in grouped.groupby("antenna", sort=True):
        fig, axes = plt.subplots(3, 3, figsize=(13.2, 9.5), sharex=True)
        for ax, band in zip(axes.ravel(), sorted(FREQUENCY_MAP_MHZ)):
            sub = src[src["frequency_band"].astype(int).eq(int(band))].sort_values("io_bin_deg")
            ax.plot(sub["io_bin_deg"], sub["high_power_fraction"], marker="o", lw=1.1, color=ANTENNA_COLOR.get(str(antenna), "black"))
            ax.set_title(f"{FREQUENCY_MAP_MHZ[band]:.2f} MHz", fontsize=9)
            ax.grid(True, color="0.9", lw=0.5)
        for ax in axes[-1, :]:
            ax.set_xlabel("Io phase (deg)")
        for ax in axes[:, 0]:
            ax.set_ylabel("high-tail fraction")
        fig.suptitle(f"Jupiter visible samples: Io-phase high-power fraction, {ANTENNA_LABEL.get(antenna, antenna)}")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        path = out_dir / f"jupiter_io_phase_high_tail_curves_{antenna}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def write_report(
    out_dir: Path,
    score: pd.DataFrame,
    corr: pd.DataFrame,
    paths: list[Path],
    config: dict[str, object],
    n_samples: int,
    n_geometry_samples: int,
) -> Path:
    display_cols = [
        "antenna",
        "frequency_mhz",
        "visible_high_maser_score_n",
        "visible_high_maser_score_high_frac",
        "visible_low_maser_score_high_frac",
        "jupiter_occulted_high_maser_score_control_high_frac",
        "high_minus_low_high_frac",
        "high_minus_occulted_high_frac",
    ]
    ranked = score.copy()
    ranked["abs_high_minus_occulted"] = ranked["high_minus_occulted_high_frac"].abs()
    top = ranked.sort_values("abs_high_minus_occulted", ascending=False).head(12)
    corr_top = corr.dropna(subset=["spearman_high_frac_vs_maser_score"]).copy()
    corr_top["abs_corr"] = corr_top["spearman_high_frac_vs_maser_score"].abs()
    corr_cols = [
        "antenna",
        "frequency_mhz",
        "n_phase_bins",
        "spearman_high_frac_vs_maser_score",
        "pearson_high_frac_vs_maser_score",
        "spearman_median_z_vs_maser_score",
    ]
    lines = [
        "# Jupiter Phase-Pattern Survey",
        "",
        "This run treats Jupiter as an episodic phase-organized source, not as an occultation-step source.",
        "",
        "## Method",
        "",
        "1. Read all valid Ryle-Vonberg samples for upper V and lower V.",
        "2. Convert raw power to log power.",
        "3. Normalize within each UTC day, antenna, and frequency band:",
        "",
        "   z = (log(power) - daily_channel_median) / daily_channel_IQR_sigma",
        "",
        "4. Mark high-power-tail samples using the configured z threshold.",
        "5. Attach SPICE-derived Jupiter CML and Io phase from the nearest geometry grid time.",
        "6. Bin samples by CML and Io phase, separately by antenna/frequency and visibility regime.",
        "7. Compare high-MASER-score phase regions against low-score and Jupiter-occulted controls.",
        "",
        "The observer vector is Earth-centered as a proxy for RAE-2/Moon. The Earth-Moon parallax is far below the 15 degree phase bins used here.",
        "",
        "## Configuration",
        "",
        *[f"- `{k}`: `{v}`" for k, v in config.items() if k != "software_versions"],
        "",
        f"Valid radiometer samples used after geometry merge: `{n_samples}`.",
        f"Geometry grid samples: `{n_geometry_samples}`.",
        "",
        "## Strongest Phase-Pattern Contrasts",
        "",
        top[display_cols].to_string(index=False) if not top.empty else "(none)",
        "",
        "## All Antenna/Frequency Contrast Summary",
        "",
        score[display_cols].sort_values(["antenna", "frequency_mhz"]).to_string(index=False) if not score.empty else "(none)",
        "",
        "## Phase-Map Correlation With MASER Io-CML Map",
        "",
        corr_top.sort_values("abs_corr", ascending=False).head(12)[corr_cols].to_string(index=False)
        if not corr_top.empty
        else "(none)",
        "",
        "## Plots",
        "",
        *[f"- `{p}`" for p in paths],
        "",
        "## Interpretation Notes",
        "",
        "- A Jovian phase pattern should appear preferentially when Jupiter is visible by the Moon.",
        "- If the same phase pattern is present in the Jupiter-occulted control, it is likely receiver/background/coverage structure rather than Jupiter.",
        "- A bursty signal is better represented by high-power-tail fraction than by median power alone.",
        "- Upper/lower V differences are allowed here; unlike lunar occultation geometry, this is a phase/burst survey and antenna pattern can matter.",
    ]
    path = out_dir / "jupiter_phase_pattern_survey_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", type=Path, default=DEFAULT_CLEAN)
    parser.add_argument("--jupiter-states", type=Path, default=DEFAULT_JUPITER_STATES)
    parser.add_argument("--earth-states", type=Path, default=DEFAULT_EARTH_STATES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--phase-bin-deg", type=float, default=15.0)
    parser.add_argument("--high-z", type=float, default=2.5)
    parser.add_argument("--min-count-per-phase-bin", type=int, default=30)
    parser.add_argument("--geometry-tolerance-s", type=float, default=360.0)
    parser.add_argument("--refresh-external", action="store_true")
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    config = {
        "clean": str(args.clean),
        "jupiter_states": str(args.jupiter_states),
        "earth_states": str(args.earth_states),
        "phase_bin_deg": float(args.phase_bin_deg),
        "high_z": float(args.high_z),
        "min_count_per_phase_bin": int(args.min_count_per_phase_bin),
        "geometry_tolerance_s": float(args.geometry_tolerance_s),
        "software_versions": software_versions(),
    }
    write_json(out_dir / "run_config.json", config)

    states = _load_visibility_states(args.jupiter_states, args.earth_states)
    geom = _annotate_spice_grid(states, out_dir, refresh=bool(args.refresh_external))
    geom.to_csv(out_dir / "jupiter_spice_visibility_geometry_grid.csv", index=False)

    clean = _read_clean(args.clean)
    clean = _add_daily_channel_normalization(clean)
    samples = _merge_geometry(clean, geom, float(args.geometry_tolerance_s))
    # Keep a compact table for downstream inspection without storing all 9M rows.
    sample_cols = [
        "time",
        "frequency_band",
        "frequency_mhz",
        "antenna",
        "log_power",
        "daily_z_log_power",
        "jupiter_visible_by_moon",
        "earth_visible_by_moon",
        "jupiter_cml_spice_deg",
        "io_phase_spice_deg",
        "maser_zarka_io_score",
    ]
    samples.sample(min(200_000, len(samples)), random_state=12345)[sample_cols].to_csv(
        out_dir / "jupiter_phase_pattern_sampled_points.csv",
        index=False,
    )

    phase_summary = _bin_phase_samples(
        samples,
        phase_bin_deg=float(args.phase_bin_deg),
        high_z=float(args.high_z),
        min_count=int(args.min_count_per_phase_bin),
    )
    phase_summary.to_csv(out_dir / "jupiter_phase_binned_summary.csv", index=False)
    score = _score_contrast_summary(samples, high_z=float(args.high_z))
    score.to_csv(out_dir / "jupiter_maser_score_contrast_summary.csv", index=False)
    corr = phase_map_correlations(phase_summary, regime="jupiter_visible")
    corr.to_csv(out_dir / "jupiter_phase_map_maser_score_correlation_summary.csv", index=False)

    paths: list[Path] = []
    paths.extend(
        plot_phase_maps(
            phase_summary,
            out_dir,
            regime="jupiter_visible",
            value_col="high_power_fraction",
            title_suffix=f"fraction with daily z > {float(args.high_z):g}",
        )
    )
    paths.extend(
        plot_phase_maps(
            phase_summary,
            out_dir,
            regime="jupiter_occulted_control",
            value_col="high_power_fraction",
            title_suffix=f"fraction with daily z > {float(args.high_z):g}",
        )
    )
    paths.extend(
        plot_phase_maps(
            phase_summary,
            out_dir,
            regime="jupiter_visible",
            value_col="median_daily_z",
            title_suffix="median daily-normalized log power",
        )
    )
    paths.append(plot_antenna_score_contrasts(score, out_dir))
    paths.append(plot_phase_map_correlations(corr, out_dir))
    paths.append(plot_observed_vs_maser_scatter(phase_summary, corr, out_dir))
    paths.extend(plot_io_phase_curves(samples, out_dir, float(args.high_z), float(args.phase_bin_deg)))

    report = write_report(out_dir, score, corr, paths, config, len(samples), len(geom))
    print(report)


if __name__ == "__main__":
    main()
