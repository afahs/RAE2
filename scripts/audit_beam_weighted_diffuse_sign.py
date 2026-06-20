#!/usr/bin/env python
"""Audit whether beam-weighted diffuse-sky changes predict raw sign classes.

This diagnostic tests a physically motivated version of the user's proposed
cut: instead of selecting events because the raw pre/post median already has
the desired sign, compute an independent lower-V diffuse-sky model across the
same event window and ask whether that model predicts source-like or
anti-template behavior.

The model is intentionally approximate.  It uses PySM synchrotron maps and the
axisymmetric mean of the available digitized Ryle-Vonberg E/H-plane beam cuts.
It is useful for testing whether broad diffuse Galactic pickup is a plausible
event-selection variable; it is not a calibrated receiver model.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import healpy as hp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rylevonberg.table_io import read_table

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.geometry import normalize_vectors  # noqa: E402
from rylevonberg.util import datetime_ns, ensure_dir, robust_sigma, software_versions, write_json  # noqa: E402
from scripts.build_earth_beam_clean_profile_grid import (  # noqa: E402
    MODEL_NSIDE,
    _beam_weighted_sky,
    _load_beam,
    _load_sky_i,
    _nearest_beam,
    _pixel_fk4_vectors,
)


CLEAN = ROOT / "outputs/control_survey_earth_sun_postnov1974_v2/01_ingest/cleaned_timeseries.csv"
RAW_SCORE_ROOT = ROOT / "outputs/lower_v_structure_selected_stacks_all_sources_v1"
OUT = ROOT / "outputs/beam_weighted_diffuse_sign_audit_v1"
LOWER_V = "rv2_coarse"
EXPECTED_SIGN = {"disappearance": -1.0, "reappearance": 1.0}
SOURCE_LABEL = {"sun": "Sun", "fornax_a": "Fornax A"}
CLASS_COLORS = {
    "all_usable": "0.35",
    "model_anti_strong": "#d62728",
    "model_source_strong": "#2ca02c",
    "model_weak_change": "#4c78a8",
}


def _read(path: Path, **kwargs) -> pd.DataFrame:
    return read_table(path, low_memory=False, **kwargs)


def _load_clean(bands: set[int]) -> pd.DataFrame:
    cols = [
        "time",
        "frequency_band",
        "frequency_mhz",
        "antenna",
        "power",
        "is_valid",
        "position_x",
        "position_y",
        "position_z",
    ]
    clean = _read(CLEAN, usecols=cols, parse_dates=["time"])
    clean = clean[
        clean["antenna"].astype(str).eq(LOWER_V)
        & clean["frequency_band"].astype(int).isin({int(b) for b in bands})
    ].copy()
    return clean.sort_values(["frequency_band", "time"]).reset_index(drop=True)


def _bool_array(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.to_numpy(dtype=bool)
    return series.astype(str).str.lower().isin(["true", "1", "yes"]).to_numpy(dtype=bool)


def _make_channel_cache(clean: pd.DataFrame) -> dict[int, dict[str, object]]:
    cache: dict[int, dict[str, object]] = {}
    pixel_vecs = _pixel_fk4_vectors(MODEL_NSIDE)
    for band, grp in clean.groupby("frequency_band", sort=True):
        channel = grp.sort_values("time").reset_index(drop=True)
        if channel.empty:
            continue
        freq = float(channel["frequency_mhz"].dropna().iloc[0])
        sky_i, sky_path = _load_sky_i(freq)
        beam_freq, eplane, hplane = _nearest_beam(freq)
        beam_angles, beam_gains = _load_beam(eplane, hplane)
        pos = channel[["position_x", "position_y", "position_z"]].to_numpy(dtype=float)
        axes = normalize_vectors(-pos)
        model = _beam_weighted_sky(axes, sky_i, pixel_vecs, beam_angles, beam_gains)
        cache[int(band)] = {
            "channel": channel,
            "time_ns": datetime_ns(channel["time"]),
            "model": model,
            "frequency_mhz": freq,
            "sky_map": str(sky_path),
            "beam_model_frequency_mhz": float(beam_freq),
            "eplane_beam": str(eplane),
            "hplane_beam": str(hplane),
        }
    return cache


def _event_slice(time_ns: np.ndarray, event_time: pd.Timestamp, window_s: float) -> slice:
    event_ns = pd.Timestamp(event_time).value
    half_ns = int(float(window_s) * 1e9)
    lo = int(np.searchsorted(time_ns, event_ns - half_ns, side="left"))
    hi = int(np.searchsorted(time_ns, event_ns + half_ns, side="right"))
    return slice(lo, hi)


def _prepost_masks(t: np.ndarray, prepost_s: float, inner_s: float) -> tuple[np.ndarray, np.ndarray]:
    pre = (t >= -float(prepost_s)) & (t <= -float(inner_s))
    post = (t >= float(inner_s)) & (t <= float(prepost_s))
    return pre, post


def _score_model_window(
    channel: pd.DataFrame,
    time_ns: np.ndarray,
    model: np.ndarray,
    row: pd.Series,
    window_s: float,
    prepost_s: float,
    inner_s: float,
    min_side_samples: int,
) -> dict[str, object]:
    event_time = pd.Timestamp(row["predicted_event_time"])
    sl = _event_slice(time_ns, event_time, window_s)
    if sl.stop <= sl.start:
        return {"model_usable": False, "model_failure": "no_samples"}
    event_ns = event_time.value
    t = (time_ns[sl] - event_ns).astype(float) / 1e9
    y = np.asarray(model[sl], dtype=float)
    keep = np.isfinite(t) & np.isfinite(y) & (np.abs(t) <= float(window_s))
    t = t[keep]
    y = y[keep]
    pre, post = _prepost_masks(t, prepost_s, inner_s)
    n_pre = int(np.count_nonzero(pre))
    n_post = int(np.count_nonzero(post))
    if n_pre < min_side_samples or n_post < min_side_samples:
        return {
            "model_usable": False,
            "model_failure": "too_few_model_side_samples",
            "model_n_pre": n_pre,
            "model_n_post": n_post,
        }
    pre_med = float(np.nanmedian(y[pre]))
    post_med = float(np.nanmedian(y[post]))
    delta = post_med - pre_med
    expected = EXPECTED_SIGN[str(row["event_type"])]
    signed_delta = expected * delta
    side = np.concatenate([y[pre], y[post]])
    level = float(np.nanmedian(np.abs(side)))
    frac = signed_delta / level if np.isfinite(level) and level > 0 else np.nan
    noise = robust_sigma(side - np.nanmedian(side))
    denom = noise * np.sqrt(1.0 / max(n_pre, 1) + 1.0 / max(n_post, 1)) if np.isfinite(noise) else np.nan
    z_like = signed_delta / denom if np.isfinite(denom) and denom > 0 else np.nan
    if np.count_nonzero(np.isfinite(t) & np.isfinite(y)) >= 6:
        slope = float(np.polyfit(t / float(window_s), y, 1)[0])
        frac_slope = slope / level if np.isfinite(level) and level > 0 else np.nan
    else:
        slope = np.nan
        frac_slope = np.nan
    return {
        "model_usable": True,
        "model_failure": "",
        "model_n_pre": n_pre,
        "model_n_post": n_post,
        "model_pre_median": pre_med,
        "model_post_median": post_med,
        "model_raw_delta_post_minus_pre": float(delta),
        "model_signed_delta": float(signed_delta),
        "model_fractional_signed_delta": float(frac) if np.isfinite(frac) else np.nan,
        "model_step_z_like": float(z_like) if np.isfinite(z_like) else np.nan,
        "model_level": level,
        "model_frac_slope_per_window": float(frac_slope) if np.isfinite(frac_slope) else np.nan,
    }


def _analysis_usable(rows: pd.DataFrame, min_side_samples: int) -> pd.Series:
    usable = rows["usable"].astype(bool)
    usable &= pd.to_numeric(rows["predicted_n_pre"], errors="coerce").ge(min_side_samples)
    usable &= pd.to_numeric(rows["predicted_n_post"], errors="coerce").ge(min_side_samples)
    usable &= pd.to_numeric(rows["predicted_signed_delta"], errors="coerce").notna()
    usable &= rows["model_usable"].astype(bool)
    usable &= pd.to_numeric(rows["model_fractional_signed_delta"], errors="coerce").notna()
    return usable


def _assign_model_classes(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    out["model_class"] = "model_other"
    out["model_abs_fractional_signed_delta"] = pd.to_numeric(out["model_fractional_signed_delta"], errors="coerce").abs()
    for _, idx in out.groupby(["frequency_mhz", "event_type"], sort=True).groups.items():
        sub = out.loc[idx]
        vals = pd.to_numeric(sub["model_fractional_signed_delta"], errors="coerce")
        abs_vals = vals.abs()
        finite = vals[np.isfinite(vals)]
        if finite.empty:
            continue
        q25 = float(np.nanquantile(finite, 0.25))
        q75 = float(np.nanquantile(finite, 0.75))
        q50_abs = float(np.nanquantile(abs_vals[np.isfinite(abs_vals)], 0.50)) if np.isfinite(abs_vals).any() else np.nan
        anti = vals.le(q25) & vals.lt(0)
        source = vals.ge(q75) & vals.gt(0)
        weak = abs_vals.le(q50_abs) if np.isfinite(q50_abs) else pd.Series(False, index=sub.index)
        out.loc[sub.index[weak], "model_class"] = "model_weak_change"
        out.loc[sub.index[source], "model_class"] = "model_source_strong"
        out.loc[sub.index[anti], "model_class"] = "model_anti_strong"
    return out


def _model_rows_for_source(source: str, cache: dict[int, dict[str, object]], args: argparse.Namespace) -> pd.DataFrame:
    score_path = RAW_SCORE_ROOT / source / "raw_occultation_candidate_scores.csv"
    scores = _read(score_path, parse_dates=["predicted_event_time"])
    scores = scores[scores["antenna"].astype(str).eq(LOWER_V)].copy()
    rows = []
    for score in scores.itertuples(index=False):
        band = int(score.frequency_band)
        cached = cache.get(band)
        base = score._asdict()
        if cached is None:
            base.update({"model_usable": False, "model_failure": "missing_clean_channel"})
        else:
            base.update(
                _score_model_window(
                    channel=cached["channel"],
                    time_ns=cached["time_ns"],
                    model=cached["model"],
                    row=pd.Series(base),
                    window_s=float(args.window_s),
                    prepost_s=float(args.prepost_s),
                    inner_s=float(args.inner_s),
                    min_side_samples=int(args.min_side_samples),
                )
            )
            base.update(
                {
                    "sky_map": cached["sky_map"],
                    "beam_model_frequency_mhz": cached["beam_model_frequency_mhz"],
                    "eplane_beam": cached["eplane_beam"],
                    "hplane_beam": cached["hplane_beam"],
                }
            )
        rows.append(base)
    out = pd.DataFrame(rows)
    out["analysis_usable"] = _analysis_usable(out, int(args.min_side_samples))
    out["observed_source_like"] = pd.to_numeric(out["predicted_signed_delta"], errors="coerce").gt(0)
    out["observed_anti_template"] = pd.to_numeric(out["predicted_signed_delta"], errors="coerce").lt(0)
    out["model_source_like"] = pd.to_numeric(out["model_signed_delta"], errors="coerce").gt(0)
    out["model_anti_template"] = pd.to_numeric(out["model_signed_delta"], errors="coerce").lt(0)
    sign_obs = np.sign(pd.to_numeric(out["predicted_signed_delta"], errors="coerce").to_numpy(dtype=float))
    sign_model = np.sign(pd.to_numeric(out["model_signed_delta"], errors="coerce").to_numpy(dtype=float))
    out["model_observed_sign_agree"] = sign_obs == sign_model
    out.loc[~np.isfinite(sign_obs) | ~np.isfinite(sign_model) | (sign_obs == 0) | (sign_model == 0), "model_observed_sign_agree"] = False
    return _assign_model_classes(out)


def _summaries(rows: pd.DataFrame, seed: int, n_perm: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    usable = rows[rows["analysis_usable"]].copy()
    summary_rows = []
    for keys, grp in usable.groupby(["source_name", "frequency_mhz", "event_type"], sort=True):
        baseline_anti = float(grp["observed_anti_template"].mean()) if len(grp) else np.nan
        baseline_source = float(grp["observed_source_like"].mean()) if len(grp) else np.nan
        corr = grp[["predicted_fractional_signed_delta", "model_fractional_signed_delta"]].corr(method="spearman").iloc[0, 1]
        for model_class, sub in grp.groupby("model_class", sort=True):
            summary_rows.append(
                {
                    "source_name": keys[0],
                    "frequency_mhz": float(keys[1]),
                    "event_type": keys[2],
                    "model_class": model_class,
                    "n_rows": int(len(sub)),
                    "baseline_n_rows": int(len(grp)),
                    "observed_source_like_fraction": float(sub["observed_source_like"].mean()),
                    "observed_anti_template_fraction": float(sub["observed_anti_template"].mean()),
                    "baseline_source_like_fraction": baseline_source,
                    "baseline_anti_template_fraction": baseline_anti,
                    "anti_template_lift_vs_baseline": float(sub["observed_anti_template"].mean() - baseline_anti),
                    "source_like_lift_vs_baseline": float(sub["observed_source_like"].mean() - baseline_source),
                    "sign_agreement_fraction": float(sub["model_observed_sign_agree"].mean()),
                    "spearman_observed_vs_model_fractional_delta": float(corr) if np.isfinite(corr) else np.nan,
                    "median_model_fractional_signed_delta": float(np.nanmedian(sub["model_fractional_signed_delta"])),
                    "median_observed_fractional_signed_delta": float(np.nanmedian(sub["predicted_fractional_signed_delta"])),
                }
            )
    class_summary = pd.DataFrame(summary_rows)

    overall_rows = []
    rng = np.random.default_rng(seed)
    for source, grp in usable.groupby("source_name", sort=True):
        model_anti = grp["model_class"].eq("model_anti_strong").to_numpy(dtype=bool)
        model_source = grp["model_class"].eq("model_source_strong").to_numpy(dtype=bool)
        anti_obs = grp["observed_anti_template"].to_numpy(dtype=bool)
        source_obs = grp["observed_source_like"].to_numpy(dtype=bool)
        baseline_anti = float(np.mean(anti_obs)) if len(anti_obs) else np.nan
        baseline_source = float(np.mean(source_obs)) if len(source_obs) else np.nan
        anti_lift = float(np.mean(anti_obs[model_anti]) - baseline_anti) if np.any(model_anti) else np.nan
        source_lift = float(np.mean(source_obs[model_source]) - baseline_source) if np.any(model_source) else np.nan
        perm_anti = []
        perm_source = []
        for _ in range(int(n_perm)):
            shuffled_anti = rng.permutation(anti_obs)
            shuffled_source = rng.permutation(source_obs)
            if np.any(model_anti):
                perm_anti.append(float(np.mean(shuffled_anti[model_anti]) - np.mean(shuffled_anti)))
            if np.any(model_source):
                perm_source.append(float(np.mean(shuffled_source[model_source]) - np.mean(shuffled_source)))
        perm_anti_arr = np.asarray(perm_anti, dtype=float)
        perm_source_arr = np.asarray(perm_source, dtype=float)
        overall_rows.append(
            {
                "source_name": source,
                "n_usable_rows": int(len(grp)),
                "baseline_anti_template_fraction": baseline_anti,
                "baseline_source_like_fraction": baseline_source,
                "model_anti_strong_rows": int(np.count_nonzero(model_anti)),
                "model_anti_strong_anti_fraction": float(np.mean(anti_obs[model_anti])) if np.any(model_anti) else np.nan,
                "model_anti_strong_anti_lift": anti_lift,
                "model_anti_permutation_p": float((1 + np.count_nonzero(perm_anti_arr >= anti_lift)) / (1 + len(perm_anti_arr)))
                if np.isfinite(anti_lift) and len(perm_anti_arr)
                else np.nan,
                "model_source_strong_rows": int(np.count_nonzero(model_source)),
                "model_source_strong_source_fraction": float(np.mean(source_obs[model_source])) if np.any(model_source) else np.nan,
                "model_source_strong_source_lift": source_lift,
                "model_source_permutation_p": float((1 + np.count_nonzero(perm_source_arr >= source_lift)) / (1 + len(perm_source_arr)))
                if np.isfinite(source_lift) and len(perm_source_arr)
                else np.nan,
                "overall_sign_agreement_fraction": float(grp["model_observed_sign_agree"].mean()),
                "overall_spearman_observed_vs_model_fractional_delta": float(
                    grp[["predicted_fractional_signed_delta", "model_fractional_signed_delta"]]
                    .corr(method="spearman")
                    .iloc[0, 1]
                ),
            }
        )
    overall = pd.DataFrame(overall_rows)
    event_summary = (
        usable.groupby(["source_name", "event_id", "event_type", "predicted_event_time"], as_index=False)
        .agg(
            observed_source_like_fraction=("observed_source_like", "mean"),
            observed_anti_template_fraction=("observed_anti_template", "mean"),
            model_signed_fractional_delta_median=("model_fractional_signed_delta", "median"),
            model_signed_fractional_delta_abs_median=("model_abs_fractional_signed_delta", "median"),
            sign_agreement_fraction=("model_observed_sign_agree", "mean"),
            n_frequency_rows=("frequency_mhz", "nunique"),
        )
        .sort_values(["source_name", "predicted_event_time"])
    )
    return class_summary, overall, event_summary


def _normalized_event_points(
    rows: pd.DataFrame,
    clean: pd.DataFrame,
    window_s: float,
    bin_s: float,
    inner_s: float,
) -> pd.DataFrame:
    groups = {}
    for band, grp in clean.groupby("frequency_band", sort=True):
        g = grp.sort_values("time").reset_index(drop=True)
        groups[int(band)] = (g, datetime_ns(g["time"]))
    bins = np.arange(-float(window_s), float(window_s) + float(bin_s), float(bin_s))
    point_rows = []
    usable = rows[rows["analysis_usable"]].copy()
    for row in usable.itertuples(index=False):
        band = int(row.frequency_band)
        payload = groups.get(band)
        if payload is None:
            continue
        channel, time_ns = payload
        event_time = pd.Timestamp(row.predicted_event_time)
        sl = _event_slice(time_ns, event_time, window_s)
        if sl.stop <= sl.start:
            continue
        event_ns = event_time.value
        local = channel.iloc[sl].copy()
        t = (time_ns[sl] - event_ns).astype(float) / 1e9
        power = pd.to_numeric(local["power"], errors="coerce").to_numpy(dtype=float)
        keep = np.isfinite(t) & np.isfinite(power) & (power > 0.0) & (np.abs(t) <= float(window_s))
        if "is_valid" in local.columns:
            keep &= _bool_array(local["is_valid"])
        if np.count_nonzero(keep) < 8:
            continue
        t = t[keep]
        power = power[keep]
        side = np.abs(t) >= float(inner_s)
        if np.count_nonzero(side) < 6:
            continue
        center = float(np.nanmedian(power[side]))
        scale = robust_sigma(power[side] - center)
        if not np.isfinite(scale) or scale <= 0:
            scale = float(np.nanstd(power[side], ddof=1)) if np.count_nonzero(side) > 1 else np.nan
        if not np.isfinite(scale) or scale <= 0:
            continue
        z = (power - center) / scale
        bin_idx = np.digitize(t, bins) - 1
        classes = ["all_usable", str(row.model_class)]
        for bidx in np.unique(bin_idx):
            if bidx < 0 or bidx >= len(bins) - 1:
                continue
            mask = bin_idx == bidx
            if not np.any(mask):
                continue
            for cls in classes:
                point_rows.append(
                    {
                        "source_name": row.source_name,
                        "event_id": row.event_id,
                        "event_type": row.event_type,
                        "frequency_band": int(row.frequency_band),
                        "frequency_mhz": float(row.frequency_mhz),
                        "antenna": LOWER_V,
                        "model_class": cls,
                        "t_bin_sec": float(0.5 * (bins[bidx] + bins[bidx + 1])),
                        "z_power": float(np.nanmedian(z[mask])),
                        "n_samples": int(np.count_nonzero(mask)),
                    }
                )
    return pd.DataFrame(point_rows)


def _profile_summary(points: pd.DataFrame) -> pd.DataFrame:
    if points.empty:
        return pd.DataFrame()
    rows = []
    by = ["source_name", "event_type", "frequency_band", "frequency_mhz", "antenna", "model_class", "t_bin_sec"]
    for keys, grp in points.groupby(by, sort=True):
        vals = pd.to_numeric(grp["z_power"], errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue
        err = robust_sigma(vals - np.nanmedian(vals)) / np.sqrt(vals.size) if vals.size > 1 else np.nan
        rows.append(
            {
                **dict(zip(by, keys)),
                "median_z_power": float(np.nanmedian(vals)),
                "median_z_power_err": float(err) if np.isfinite(err) else np.nan,
                "n_events": int(grp["event_id"].nunique()),
                "n_points": int(len(grp)),
            }
        )
    return pd.DataFrame(rows)


def _plot_scatter(rows: pd.DataFrame, source: str, out_dir: Path) -> Path:
    usable = rows[rows["analysis_usable"]].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    x_all = pd.to_numeric(usable["model_fractional_signed_delta"], errors="coerce").to_numpy(dtype=float)
    y_all = pd.to_numeric(usable["predicted_fractional_signed_delta"], errors="coerce").to_numpy(dtype=float)
    xlim = float(np.nanpercentile(np.abs(x_all[np.isfinite(x_all)]), 98)) if np.isfinite(x_all).any() else 1.0
    ylim = float(np.nanpercentile(np.abs(y_all[np.isfinite(y_all)]), 98)) if np.isfinite(y_all).any() else 1.0
    xlim = xlim if np.isfinite(xlim) and xlim > 0 else 1.0
    ylim = ylim if np.isfinite(ylim) and ylim > 0 else 1.0
    for ax, event_type in zip(axes, ["disappearance", "reappearance"]):
        sub = usable[usable["event_type"].astype(str).eq(event_type)].copy()
        if sub.empty:
            continue
        x = pd.to_numeric(sub["model_fractional_signed_delta"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(sub["predicted_fractional_signed_delta"], errors="coerce").to_numpy(dtype=float)
        clipped = int(np.count_nonzero(np.isfinite(x) & np.isfinite(y) & ((np.abs(x) > xlim) | (np.abs(y) > ylim))))
        freq = pd.to_numeric(sub["frequency_mhz"], errors="coerce")
        sc = ax.scatter(
            sub["model_fractional_signed_delta"],
            sub["predicted_fractional_signed_delta"],
            c=freq,
            s=13,
            alpha=0.55,
            cmap="viridis",
            linewidths=0,
        )
        ax.axhline(0, color="0.5", lw=0.8)
        ax.axvline(0, color="0.5", lw=0.8)
        ax.set_xlim(-xlim, xlim)
        ax.set_ylim(-ylim, ylim)
        ax.set_title(event_type)
        ax.set_xlabel("diffuse model expected-sign fractional change")
        ax.grid(alpha=0.2)
        ax.text(
            0.02,
            0.96,
            f"axes clipped at 98% abs; hidden={clipped}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
        )
    axes[0].set_ylabel("observed raw expected-sign fractional change")
    cbar = fig.colorbar(sc, ax=axes, shrink=0.9)
    cbar.set_label("frequency MHz")
    fig.suptitle(f"{SOURCE_LABEL.get(source, source)} lower V: observed raw sign vs beam-weighted diffuse model")
    path = out_dir / f"{source}_model_vs_observed_signed_delta_scatter.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_enrichment(class_summary: pd.DataFrame, source: str, out_dir: Path) -> Path:
    sub = class_summary[class_summary["source_name"].astype(str).str.lower().eq(source)].copy()
    sub = sub[sub["model_class"].isin(["model_anti_strong", "model_source_strong", "model_weak_change"])]
    if sub.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No class summary rows", ha="center", va="center")
        path = out_dir / f"{source}_model_class_enrichment.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path
    freqs = sorted(sub["frequency_mhz"].dropna().unique())
    fig, axes = plt.subplots(len(freqs), 2, figsize=(12, max(8, 1.3 * len(freqs))), sharey=True)
    if len(freqs) == 1:
        axes = np.asarray([axes])
    classes = ["model_anti_strong", "model_weak_change", "model_source_strong"]
    labels = ["model anti", "model weak", "model source"]
    colors = [CLASS_COLORS[c] for c in classes]
    for i, freq in enumerate(freqs):
        for j, event_type in enumerate(["disappearance", "reappearance"]):
            ax = axes[i, j]
            g = sub[np.isclose(sub["frequency_mhz"], freq) & sub["event_type"].astype(str).eq(event_type)]
            vals = []
            ns = []
            for cls in classes:
                row = g[g["model_class"].eq(cls)]
                vals.append(float(row["observed_anti_template_fraction"].iloc[0]) if not row.empty else np.nan)
                ns.append(int(row["n_rows"].iloc[0]) if not row.empty else 0)
            ax.bar(np.arange(len(classes)), vals, color=colors, alpha=0.85)
            base = g["baseline_anti_template_fraction"].dropna()
            if not base.empty:
                ax.axhline(float(base.iloc[0]), color="black", linestyle="--", lw=0.8, label="all usable")
            ax.set_xticks(np.arange(len(classes)), labels, rotation=25, ha="right", fontsize=7)
            ax.set_ylim(0, 1)
            ax.set_title(f"{freq:.2f} MHz {event_type}", fontsize=9)
            for x, val, n in zip(np.arange(len(classes)), vals, ns):
                if np.isfinite(val):
                    ax.text(x, min(0.96, val + 0.025), f"n={n}", ha="center", va="bottom", fontsize=7)
            if j == 0:
                ax.tick_params(axis="y", labelsize=8)
    fig.suptitle(f"{SOURCE_LABEL.get(source, source)}: does model diffuse class enrich anti-template raw behavior?", y=0.995)
    fig.supylabel("observed anti-template fraction", x=0.01)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    path = out_dir / f"{source}_model_class_anti_template_enrichment.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_profile_grid(summary: pd.DataFrame, source: str, out_dir: Path, window_s: float) -> Path:
    sub_all = summary[summary["source_name"].astype(str).str.lower().eq(source)].copy()
    freqs = sorted(sub_all["frequency_mhz"].dropna().unique())
    classes = ["all_usable", "model_anti_strong", "model_weak_change", "model_source_strong"]
    labels = {
        "all_usable": "all usable",
        "model_anti_strong": "diffuse anti strong",
        "model_weak_change": "diffuse weak",
        "model_source_strong": "diffuse source strong",
    }
    fig, axes = plt.subplots(len(freqs), 2, figsize=(13, max(10, 1.45 * len(freqs))), sharex=True)
    if len(freqs) == 1:
        axes = np.asarray([axes])
    for i, freq in enumerate(freqs):
        for j, event_type in enumerate(["disappearance", "reappearance"]):
            ax = axes[i, j]
            for cls in classes:
                sub = sub_all[
                    np.isclose(sub_all["frequency_mhz"], freq)
                    & sub_all["event_type"].astype(str).eq(event_type)
                    & sub_all["model_class"].astype(str).eq(cls)
                ].sort_values("t_bin_sec")
                if sub.empty:
                    continue
                alpha = 0.55 if cls == "all_usable" else 0.95
                lw = 1.0 if cls == "all_usable" else 1.6
                ax.errorbar(
                    sub["t_bin_sec"],
                    sub["median_z_power"],
                    yerr=sub["median_z_power_err"],
                    color=CLASS_COLORS.get(cls, "0.3"),
                    alpha=alpha,
                    linewidth=lw,
                    marker="o",
                    markersize=2.2,
                    elinewidth=0.55,
                    capsize=1.0,
                    label=labels.get(cls, cls),
                )
            ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
            ax.axhline(0, color="0.6", linewidth=0.7)
            ax.set_title(f"{freq:.2f} MHz {event_type}", fontsize=9)
            if i == len(freqs) - 1:
                ax.set_xlabel("seconds from predicted event")
            if i == 0 and j == 1:
                ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.suptitle(
        f"{SOURCE_LABEL.get(source, source)} lower V: raw profiles split by independent diffuse-model class\n"
        "Classes come from beam-weighted PySM pre/post change, not from raw power sign.",
        y=0.996,
    )
    fig.supylabel("normalized raw power", x=0.01)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    path = out_dir / f"{source}_raw_profile_grid_split_by_diffuse_model_class_{int(window_s)}s.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_time_frequency(rows: pd.DataFrame, source: str, out_dir: Path) -> Path:
    usable = rows[rows["analysis_usable"]].copy()
    if usable.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No usable rows", ha="center", va="center")
        path = out_dir / f"{source}_model_signed_delta_time_frequency.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path
    t = pd.to_datetime(usable["predicted_event_time"], errors="coerce")
    vals = pd.to_numeric(usable["model_fractional_signed_delta"], errors="coerce")
    lim = float(np.nanpercentile(np.abs(vals), 98)) if np.isfinite(vals).any() else 1.0
    if not np.isfinite(lim) or lim <= 0:
        lim = 1.0
    marker = np.where(usable["observed_source_like"], "o", "x")
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for m in ["o", "x"]:
        mask = marker == m
        sc = ax.scatter(
            t[mask],
            usable.loc[mask, "frequency_mhz"],
            c=vals[mask],
            cmap="coolwarm",
            vmin=-lim,
            vmax=lim,
            marker=m,
            s=24 if m == "o" else 34,
            alpha=0.75,
            linewidths=0.8,
            label="observed source-like" if m == "o" else "observed anti-template",
        )
    ax.set_xlabel("predicted event time")
    ax.set_ylabel("frequency MHz")
    ax.set_title(f"{SOURCE_LABEL.get(source, source)} lower V: diffuse-model sign over time")
    ax.legend(frameon=False, fontsize=8)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("model expected-sign fractional change")
    fig.autofmt_xdate()
    fig.tight_layout()
    path = out_dir / f"{source}_model_signed_delta_time_frequency.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_report(
    out_dir: Path,
    source: str,
    rows: pd.DataFrame,
    class_summary: pd.DataFrame,
    overall: pd.DataFrame,
    paths: list[Path],
    args: argparse.Namespace,
) -> Path:
    usable = rows[rows["analysis_usable"]].copy()
    ov = overall[overall["source_name"].astype(str).str.lower().eq(source)]
    if ov.empty:
        overall_text = "No overall summary."
    else:
        overall_text = ov.to_string(index=False)
    top = class_summary[
        class_summary["source_name"].astype(str).str.lower().eq(source)
        & class_summary["model_class"].isin(["model_anti_strong", "model_source_strong"])
    ].sort_values(["frequency_mhz", "event_type", "model_class"])
    lines = [
        f"# {SOURCE_LABEL.get(source, source)} Beam-Weighted Diffuse Sign Audit",
        "",
        "## Purpose",
        "",
        "This tests whether an independent lower-V beam-weighted diffuse-background model can explain or predict",
        "which raw occultation windows look source-like or anti-template.",
        "",
        "Positive model expected-sign fractional change means the diffuse model changes in the same direction as",
        "a real occultation signal would. Negative means the diffuse model predicts anti-template behavior.",
        "",
        "## Inputs",
        "",
        f"- source: `{source}`",
        f"- antenna: `{LOWER_V}` / lower V only",
        f"- window: +/- {float(args.window_s):.0f} s",
        f"- pre/post bands: {float(args.inner_s):.0f} to {float(args.prepost_s):.0f} s from event center",
        f"- scored rows: {len(rows)}",
        f"- analysis-usable rows: {len(usable)}",
        "- sky model: PySM synchrotron map, nearest integer MHz with sub-MHz clamped to 1 MHz",
        "- beam model: axisymmetric mean of digitized E/H-plane Ryle-Vonberg cuts",
        "",
        "## Overall Summary",
        "",
        overall_text,
        "",
        "## Model-Class Rows",
        "",
        top[
            [
                "frequency_mhz",
                "event_type",
                "model_class",
                "n_rows",
                "observed_anti_template_fraction",
                "baseline_anti_template_fraction",
                "anti_template_lift_vs_baseline",
                "observed_source_like_fraction",
                "baseline_source_like_fraction",
                "source_like_lift_vs_baseline",
                "sign_agreement_fraction",
                "spearman_observed_vs_model_fractional_delta",
            ]
        ].to_string(index=False)
        if not top.empty
        else "No class summary rows.",
        "",
        "## Interpretation Guardrails",
        "",
        "- A useful contamination cut should be based on `model_*` columns, not observed raw sign.",
        "- Strong model anti-template enrichment means diffuse pickup is likely biasing that event subset.",
        "- Weak or inconsistent enrichment means the current 1D-beam diffuse model is not enough to define a hard cut.",
        "- This audit does not prove a detection; it tests whether diffuse-background physics can guide event triage.",
        "",
        "## Diagnostic Plots",
        "",
    ]
    lines.extend(f"- `{path}`" for path in paths)
    report = out_dir / f"{source}_beam_weighted_diffuse_sign_audit_report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run_source(source: str, clean: pd.DataFrame, cache: dict[int, dict[str, object]], args: argparse.Namespace) -> Path:
    out_dir = ensure_dir(Path(args.out_dir) / source)
    rows = _model_rows_for_source(source, cache, args)
    class_summary, overall, event_summary = _summaries(rows, int(args.seed), int(args.n_permutations))
    points = _normalized_event_points(
        rows,
        clean,
        window_s=float(args.window_s),
        bin_s=float(args.bin_s),
        inner_s=float(args.inner_s),
    )
    profile_summary = _profile_summary(points)

    rows.to_csv(out_dir / f"{source}_beam_weighted_diffuse_row_audit.csv", index=False)
    class_summary.to_csv(out_dir / f"{source}_beam_weighted_diffuse_class_summary.csv", index=False)
    overall.to_csv(out_dir / f"{source}_beam_weighted_diffuse_overall_summary.csv", index=False)
    event_summary.to_csv(out_dir / f"{source}_beam_weighted_diffuse_event_summary.csv", index=False)
    points.to_csv(out_dir / f"{source}_raw_profile_points_by_diffuse_model_class.csv", index=False)
    profile_summary.to_csv(out_dir / f"{source}_raw_profile_summary_by_diffuse_model_class.csv", index=False)

    paths = [
        _plot_scatter(rows, source, out_dir),
        _plot_enrichment(class_summary, source, out_dir),
        _plot_time_frequency(rows, source, out_dir),
        _plot_profile_grid(profile_summary, source, out_dir, float(args.window_s)),
    ]
    return _write_report(out_dir, source, rows, class_summary, overall, paths, args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", default="sun,fornax_a")
    parser.add_argument("--out-dir", default=str(OUT))
    parser.add_argument("--window-s", type=float, default=600.0)
    parser.add_argument("--prepost-s", type=float, default=300.0)
    parser.add_argument("--inner-s", type=float, default=30.0)
    parser.add_argument("--bin-s", type=float, default=60.0)
    parser.add_argument("--min-side-samples", type=int, default=4)
    parser.add_argument("--n-permutations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260603)
    args = parser.parse_args()

    out_dir = ensure_dir(Path(args.out_dir))
    sources = [x.strip().lower() for x in str(args.sources).split(",") if x.strip()]
    all_scores = []
    for source in sources:
        score_path = RAW_SCORE_ROOT / source / "raw_occultation_candidate_scores.csv"
        if not score_path.exists():
            raise FileNotFoundError(score_path)
        score = _read(score_path, usecols=["frequency_band", "antenna"])
        score = score[score["antenna"].astype(str).eq(LOWER_V)]
        all_scores.append(score)
    bands = {int(x) for x in pd.concat(all_scores, ignore_index=True)["frequency_band"].dropna().astype(int).unique()}
    clean = _load_clean(bands)
    cache = _make_channel_cache(clean)
    write_json(
        out_dir / "run_config.json",
        {
            "sources": sources,
            "antenna": LOWER_V,
            "clean": str(CLEAN),
            "raw_score_root": str(RAW_SCORE_ROOT),
            "window_s": float(args.window_s),
            "prepost_s": float(args.prepost_s),
            "inner_s": float(args.inner_s),
            "bin_s": float(args.bin_s),
            "min_side_samples": int(args.min_side_samples),
            "n_permutations": int(args.n_permutations),
            "seed": int(args.seed),
            "model_nside": MODEL_NSIDE,
            "software_versions": software_versions(),
        },
    )

    reports = []
    for source in sources:
        reports.append(run_source(source, clean, cache, args))
    index = out_dir / "beam_weighted_diffuse_sign_audit_index.md"
    index.write_text(
        "# Beam-Weighted Diffuse Sign Audit\n\n" + "\n".join(f"- `{path}`" for path in reports) + "\n",
        encoding="utf-8",
    )
    print(index)


if __name__ == "__main__":
    main()
