#!/usr/bin/env python
"""Fit occultation templates after normalizing and stacking events."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rylevonberg.table_io import read_table

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.detection import baseline_matrix, event_template
from rylevonberg.stackfit import StackedStepFitConfig, fit_stacked_step, stacked_event_template
from rylevonberg.util import datetime_ns, ensure_dir, robust_sigma, write_json


CLEAN = ROOT / "outputs/control_survey_earth_sun_postnov1974_v2/01_ingest/cleaned_timeseries.csv"
ANT_LABEL = {"rv1_coarse": "upper V", "rv2_coarse": "lower V"}
VARIANT_LABELS = {
    "raw_fractional_no_baseline": "raw fractional, no drift subtraction",
    "raw_zscore_no_baseline": "raw z-score, no drift subtraction",
    "sideband_constant_subtracted": "sideband constant subtracted",
    "sideband_linear_subtracted": "sideband linear subtracted",
}
EVENT_COLORS = {"disappearance": "#4c78a8", "reappearance": "#d95f02", "combined": "#333333"}


def _read(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return read_table(path, low_memory=False, **kwargs)


def _antenna_label(antenna: str) -> str:
    return ANT_LABEL.get(str(antenna), str(antenna))


def _title_source(source: str) -> str:
    return "Jupiter" if str(source).lower() == "jupiter" else str(source).capitalize()


def _event_window(group: pd.DataFrame, group_ns: np.ndarray, event: pd.Series, window_s: float) -> tuple[np.ndarray, np.ndarray] | None:
    event_ns = pd.Timestamp(event["predicted_event_time"]).value
    half_ns = int(float(window_s) * 1e9)
    lo = int(np.searchsorted(group_ns, event_ns - half_ns, side="left"))
    hi = int(np.searchsorted(group_ns, event_ns + half_ns, side="right"))
    if hi <= lo:
        return None
    local = group.iloc[lo:hi]
    rel = (datetime_ns(local["time"]) - event_ns).astype(float) / 1e9
    y = pd.to_numeric(local["power"], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(y) & (np.abs(rel) <= float(window_s))
    if "is_valid" in local.columns:
        keep &= local["is_valid"].to_numpy(dtype=bool)
    if np.count_nonzero(keep) < 8:
        return None
    tr = rel[keep]
    yy = y[keep]
    order = np.argsort(tr)
    return tr[order], yy[order]


def _safe_scale(values: np.ndarray) -> float:
    sig = robust_sigma(values)
    if np.isfinite(sig) and sig > 0:
        return float(sig)
    std = float(np.nanstd(values))
    return std if np.isfinite(std) and std > 0 else 1.0


def _profile_variant(tr: np.ndarray, y: np.ndarray, variant: str, exclusion_s: float) -> tuple[np.ndarray, float, int]:
    side = np.abs(tr) >= float(exclusion_s)
    if np.count_nonzero(side) < 6:
        side = np.ones(len(y), dtype=bool)
    side_values = y[side]
    center = float(np.nanmedian(side_values))
    scale = _safe_scale(side_values - center)
    if variant == "raw_fractional_no_baseline":
        denom = center if np.isfinite(center) and abs(center) > 0 else float(np.nanmedian(y))
        if not np.isfinite(denom) or abs(denom) == 0:
            denom = 1.0
        return y / denom, abs(scale / denom), int(np.count_nonzero(side))
    if variant == "raw_zscore_no_baseline":
        return (y - center) / scale, 1.0, int(np.count_nonzero(side))
    if variant == "sideband_constant_subtracted":
        return (y - center) / scale, 1.0, int(np.count_nonzero(side))
    if variant == "sideband_linear_subtracted":
        B = baseline_matrix(tr[side], 1)
        beta, *_ = np.linalg.lstsq(B, y[side], rcond=None)
        baseline = baseline_matrix(tr, 1) @ beta
        resid = y - baseline
        return resid / _safe_scale(resid[side]), 1.0, int(np.count_nonzero(side))
    raise ValueError(f"unknown profile variant: {variant}")


def _collect_profiles(
    clean: pd.DataFrame,
    events: pd.DataFrame,
    source_row: pd.Series,
    variants: list[str],
    bin_s: float,
    sideband_exclusion_s: float,
) -> pd.DataFrame:
    source = str(source_row["source_name"])
    band = int(source_row["target_band"])
    antenna = str(source_row["target_antenna"])
    window_s = float(source_row["target_window_s"])
    group = clean[
        clean["frequency_band"].astype(int).eq(band)
        & clean["antenna"].astype(str).eq(antenna)
    ].sort_values("time").reset_index(drop=True)
    group_ns = datetime_ns(group["time"])
    sub_events = events[
        events["source_name"].astype(str).eq(source)
        & events["frequency_band"].astype(int).eq(band)
        & events["antenna"].astype(str).eq(antenna)
    ].copy()
    rows = []
    for _, ev in sub_events.iterrows():
        local = _event_window(group, group_ns, ev, window_s)
        if local is None:
            continue
        tr, y = local
        for variant in variants:
            profile, local_scale, n_side = _profile_variant(tr, y, variant, sideband_exclusion_s)
            tbin = np.round(tr / float(bin_s)) * float(bin_s)
            for trel, tb, val in zip(tr, tbin, profile):
                rows.append(
                    {
                        "source_name": source,
                        "event_id": ev.get("event_id"),
                        "event_type": ev["event_type"],
                        "predicted_event_time": ev["predicted_event_time"],
                        "month_block": pd.Timestamp(ev["predicted_event_time"]).strftime("%Y-%m"),
                        "frequency_band": band,
                        "frequency_mhz": float(source_row["target_frequency_mhz"]),
                        "antenna": antenna,
                        "window_s": window_s,
                        "variant": variant,
                        "t_rel_sec": float(trel),
                        "t_bin_sec": float(tb),
                        "profile_value": float(val),
                        "local_scale": float(local_scale),
                        "n_sideband_samples": int(n_side),
                    }
                )
    return pd.DataFrame.from_records(rows)


def _stack_profiles(profiles: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = [
        "source_name",
        "frequency_band",
        "frequency_mhz",
        "antenna",
        "window_s",
        "variant",
        "event_type",
        "t_bin_sec",
    ]
    for keys, grp in profiles.groupby(group_cols, dropna=False, sort=True):
        meta = dict(zip(group_cols, keys))
        vals = pd.to_numeric(grp["profile_value"], errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue
        rows.append(
            {
                **meta,
                "mean_profile": float(np.nanmean(vals)),
                "median_profile": float(np.nanmedian(vals)),
                "sem_profile": float(np.nanstd(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else np.nan,
                "robust_sem_profile": float(robust_sigma(vals) / np.sqrt(vals.size)) if vals.size > 1 else np.nan,
                "n_samples": int(vals.size),
                "n_events": int(grp["event_id"].nunique()),
            }
        )
    return pd.DataFrame.from_records(rows)


def _fit_summary(stacked: pd.DataFrame, profiles: pd.DataFrame, timing_offsets: list[float]) -> pd.DataFrame:
    rows = []
    cfg0 = StackedStepFitConfig(baseline_order=0, timing_offsets_seconds=tuple(timing_offsets))
    cfg1 = StackedStepFitConfig(baseline_order=1, timing_offsets_seconds=tuple(timing_offsets))
    group_cols = ["source_name", "frequency_band", "frequency_mhz", "antenna", "window_s", "variant"]
    event_counts = (
        profiles.groupby(group_cols, dropna=False, sort=True)["event_id"].nunique().rename("total_events_used").reset_index()
        if not profiles.empty
        else pd.DataFrame(columns=[*group_cols, "total_events_used"])
    )
    for keys, grp in stacked.groupby(group_cols, dropna=False, sort=True):
        meta = dict(zip(group_cols, keys))
        count_match = event_counts
        for key, value in meta.items():
            count_match = count_match[count_match[key].eq(value)]
        total_events_used = int(count_match["total_events_used"].iloc[0]) if not count_match.empty else 0
        for label, sub in [("combined", grp), *[(et, g) for et, g in grp.groupby("event_type", sort=True)]]:
            sub = sub.sort_values(["event_type", "t_bin_sec"])
            for model_name, cfg in [("constant_plus_step", cfg0), ("linear_plus_step", cfg1)]:
                fit = fit_stacked_step(
                    sub["t_bin_sec"].to_numpy(dtype=float),
                    sub["mean_profile"].to_numpy(dtype=float),
                    sub["event_type"].astype(str).to_numpy(),
                    uncertainty=sub["robust_sem_profile"].fillna(sub["sem_profile"]).to_numpy(dtype=float),
                    config=cfg,
                )
                rows.append(
                    {
                        **meta,
                        "stack_group": str(label),
                        "stack_model": model_name,
                        "total_events_used": total_events_used,
                        "median_events_per_bin": float(np.nanmedian(sub["n_events"])) if not sub.empty else np.nan,
                        "max_events_per_bin": int(sub["n_events"].max()) if not sub.empty else 0,
                        **fit,
                    }
                )
    return pd.DataFrame.from_records(rows)


def _stack_one_profile_table(profiles: pd.DataFrame) -> pd.DataFrame:
    if profiles.empty:
        return pd.DataFrame()
    rows = []
    for (event_type, tbin), grp in profiles.groupby(["event_type", "t_bin_sec"], dropna=False, sort=True):
        vals = pd.to_numeric(grp["profile_value"], errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue
        rows.append(
            {
                "event_type": event_type,
                "t_bin_sec": float(tbin),
                "mean_profile": float(np.nanmean(vals)),
                "sem_profile": float(np.nanstd(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else np.nan,
                "robust_sem_profile": float(robust_sigma(vals) / np.sqrt(vals.size)) if vals.size > 1 else np.nan,
                "n_events": int(grp["event_id"].nunique()),
            }
        )
    return pd.DataFrame.from_records(rows)


def _fit_stacked_table(table: pd.DataFrame, stack_model: str, timing_offsets: list[float]) -> dict[str, float]:
    if table.empty:
        return {}
    cfg = StackedStepFitConfig(
        baseline_order=1 if stack_model == "linear_plus_step" else 0,
        timing_offsets_seconds=tuple(timing_offsets),
    )
    table = table.sort_values(["event_type", "t_bin_sec"])
    return fit_stacked_step(
        table["t_bin_sec"].to_numpy(dtype=float),
        table["mean_profile"].to_numpy(dtype=float),
        table["event_type"].astype(str).to_numpy(),
        uncertainty=table["robust_sem_profile"].fillna(table["sem_profile"]).to_numpy(dtype=float),
        config=cfg,
    )


def _event_bootstrap_and_month_jackknife(
    profiles: pd.DataFrame,
    fits: pd.DataFrame,
    timing_offsets: list[float],
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate stack-fit stability by resampling/removing whole events."""
    if profiles.empty or fits.empty:
        return fits, pd.DataFrame()
    rng = np.random.default_rng(int(seed))
    fit_rows = []
    month_rows = []
    group_cols = ["source_name", "frequency_band", "frequency_mhz", "antenna", "window_s", "variant"]
    combined = fits[fits["stack_group"].eq("combined")].copy()
    for keys, prof_grp in profiles.groupby(group_cols, dropna=False, sort=True):
        meta = dict(zip(group_cols, keys))
        event_ids = prof_grp["event_id"].drop_duplicates().to_numpy()
        full_rows = combined
        for key, value in meta.items():
            full_rows = full_rows[full_rows[key].eq(value)]
        for _, full in full_rows.iterrows():
            stack_model = str(full["stack_model"])
            amps = []
            snrs = []
            offsets = []
            if n_bootstrap > 0 and event_ids.size > 2:
                by_event = {event_id: frame for event_id, frame in prof_grp.groupby("event_id", sort=False)}
                for _ in range(int(n_bootstrap)):
                    sampled = rng.choice(event_ids, size=event_ids.size, replace=True)
                    sample_profile = pd.concat([by_event[event_id] for event_id in sampled], ignore_index=True)
                    stack = _stack_one_profile_table(sample_profile)
                    boot_fit = _fit_stacked_table(stack, stack_model, timing_offsets)
                    if boot_fit:
                        amps.append(float(boot_fit.get("amplitude", np.nan)))
                        snrs.append(float(boot_fit.get("stack_fit_snr", np.nan)))
                        offsets.append(float(boot_fit.get("best_timing_offset_s", np.nan)))
            months = sorted(prof_grp["month_block"].dropna().unique())
            leave_amps = []
            leave_snrs = []
            for month in months:
                sub = prof_grp[~prof_grp["month_block"].eq(month)].copy()
                if sub.empty:
                    continue
                stack = _stack_one_profile_table(sub)
                leave_fit = _fit_stacked_table(stack, stack_model, timing_offsets)
                if not leave_fit:
                    continue
                leave_amps.append(float(leave_fit.get("amplitude", np.nan)))
                leave_snrs.append(float(leave_fit.get("stack_fit_snr", np.nan)))
                month_rows.append(
                    {
                        **meta,
                        "stack_model": stack_model,
                        "left_out_month": str(month),
                        "events_left": int(sub["event_id"].nunique()),
                        **leave_fit,
                    }
                )
            amp = float(full["amplitude"])
            boot_std = float(np.nanstd(amps, ddof=1)) if len(amps) > 1 else np.nan
            leave_amps_arr = np.asarray(leave_amps, dtype=float)
            leave_snrs_arr = np.asarray(leave_snrs, dtype=float)
            fit_rows.append(
                {
                    **full.to_dict(),
                    "event_bootstrap_amp_std": boot_std,
                    "event_bootstrap_snr": float(amp / boot_std) if np.isfinite(boot_std) and boot_std > 0 else np.nan,
                    "event_bootstrap_sign_fraction": float(np.nanmean(np.sign(amps) == np.sign(amp))) if len(amps) else np.nan,
                    "event_bootstrap_median_abs_timing_offset_s": float(np.nanmedian(np.abs(offsets))) if len(offsets) else np.nan,
                    "leave_one_month_amp_std": float(np.nanstd(leave_amps_arr, ddof=1)) if leave_amps_arr.size > 1 else np.nan,
                    "leave_one_month_min_abs_snr": float(np.nanmin(np.abs(leave_snrs_arr))) if leave_snrs_arr.size else np.nan,
                    "leave_one_month_sign_fraction": float(np.nanmean(np.sign(leave_amps_arr) == np.sign(amp))) if leave_amps_arr.size else np.nan,
                    "max_month_amp_leverage": float(np.nanmax(np.abs(leave_amps_arr - amp))) if leave_amps_arr.size else np.nan,
                }
            )
    enriched_combined = pd.DataFrame.from_records(fit_rows)
    other = fits[~fits["stack_group"].eq("combined")].copy()
    enriched = pd.concat([enriched_combined, other], ignore_index=True) if not enriched_combined.empty else fits
    return enriched, pd.DataFrame.from_records(month_rows)


def _month_cluster_fit_summary(profiles: pd.DataFrame, timing_offsets: list[float]) -> pd.DataFrame:
    """Fit combined stacks separately by calendar month for episodic-source checks."""
    rows = []
    if profiles.empty:
        return pd.DataFrame()
    group_cols = ["source_name", "frequency_band", "frequency_mhz", "antenna", "window_s", "variant", "month_block"]
    for keys, grp in profiles.groupby(group_cols, dropna=False, sort=True):
        meta = dict(zip(group_cols, keys))
        if grp["event_id"].nunique() < 3:
            continue
        stack = _stack_one_profile_table(grp)
        for model_name in ["constant_plus_step", "linear_plus_step"]:
            fit = _fit_stacked_table(stack, model_name, timing_offsets)
            rows.append(
                {
                    **meta,
                    "stack_model": model_name,
                    "total_events_used": int(grp["event_id"].nunique()),
                    "median_events_per_bin": float(np.nanmedian(stack["n_events"])) if not stack.empty else np.nan,
                    "max_events_per_bin": int(stack["n_events"].max()) if not stack.empty else 0,
                    **fit,
                }
            )
    return pd.DataFrame.from_records(rows)


def _model_for_plot(sub: pd.DataFrame, fit: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    t = sub["t_bin_sec"].to_numpy(dtype=float)
    y = sub["mean_profile"].to_numpy(dtype=float)
    et = sub["event_type"].astype(str).to_numpy()
    order = int(fit["baseline_order"])
    tmpl = stacked_event_template(t, et, timing_offset_sec=float(fit["best_timing_offset_s"]))
    B = baseline_matrix(t, order)
    X = np.column_stack([B, tmpl])
    sem = sub["robust_sem_profile"].fillna(sub["sem_profile"]).to_numpy(dtype=float)
    sem = np.where(np.isfinite(sem) & (sem > 0), sem, np.nanmedian(sem[np.isfinite(sem) & (sem > 0)]) if np.any(np.isfinite(sem) & (sem > 0)) else 1.0)
    w = 1.0 / sem**2
    beta, *_ = np.linalg.lstsq(X * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)
    return X @ beta, tmpl * float(beta[-1])


def _plot_source(source: str, stacked: pd.DataFrame, fits: pd.DataFrame, out_dir: Path) -> list[Path]:
    paths = []
    src_stack = stacked[stacked["source_name"].astype(str).eq(source)].copy()
    src_fits = fits[fits["source_name"].astype(str).eq(source)].copy()
    if src_stack.empty or src_fits.empty:
        return paths
    variants = [v for v in VARIANT_LABELS if v in set(src_stack["variant"])]
    nrows = len(variants)
    fig, axes = plt.subplots(nrows, 2, figsize=(13.2, max(3.2, 3.0 * nrows)), squeeze=False, sharex=True)
    for i, variant in enumerate(variants):
        for j, model_name in enumerate(["constant_plus_step", "linear_plus_step"]):
            ax = axes[i, j]
            sub = src_stack[src_stack["variant"].eq(variant)].copy()
            fit_rows = src_fits[
                src_fits["variant"].eq(variant)
                & src_fits["stack_group"].eq("combined")
                & src_fits["stack_model"].eq(model_name)
            ]
            for event_type, grp in sub.groupby("event_type", sort=True):
                grp = grp.sort_values("t_bin_sec")
                sem = grp["robust_sem_profile"].fillna(grp["sem_profile"]).fillna(0.0)
                x = grp["t_bin_sec"].to_numpy(dtype=float) / 60.0
                y = grp["mean_profile"].to_numpy(dtype=float)
                ax.plot(x, y, marker="o", ms=3, lw=1.3, color=EVENT_COLORS.get(str(event_type)), label=str(event_type))
                ax.fill_between(x, y - sem, y + sem, color=EVENT_COLORS.get(str(event_type)), alpha=0.13, linewidth=0)
            if not fit_rows.empty:
                fit = fit_rows.iloc[0]
                model_sub = sub.sort_values(["event_type", "t_bin_sec"])
                model, step_component = _model_for_plot(model_sub, fit)
                for event_type, grp in model_sub.assign(model=model, step_component=step_component).groupby("event_type", sort=True):
                    grp = grp.sort_values("t_bin_sec")
                    ax.plot(grp["t_bin_sec"] / 60.0, grp["model"], color=EVENT_COLORS.get(str(event_type)), lw=2.0, ls="--")
                ax.text(
                    0.02,
                    0.96,
                    f"A={float(fit['amplitude']):.3g}\nSNR={float(fit['stack_fit_snr']):.2f}\nDeltaBIC={float(fit['delta_bic']):.1f}\ndt={float(fit['best_timing_offset_s']):.0f}s",
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=8,
                    bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.8", "pad": 3},
                )
            ax.axvline(0.0, color="black", lw=0.9, ls=":")
            ax.set_title(f"{VARIANT_LABELS.get(variant, variant)} / {model_name.replace('_', ' ')}")
            ax.set_ylabel("Stacked normalized value")
            if i == 0 and j == 1:
                ax.legend(fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel("Relative time from predicted event (min)")
    meta = src_stack.iloc[0]
    fig.suptitle(
        f"{_title_source(source)} stack-first fits: {float(meta['frequency_mhz']):.2f} MHz {_antenna_label(str(meta['antenna']))}",
        fontsize=13,
    )
    fig.tight_layout()
    path = out_dir / source / f"{source}_stack_first_fit_grid.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)
    return paths


def _plot_month_clusters(source: str, month_fits: pd.DataFrame, out_dir: Path) -> Path | None:
    sub = month_fits[
        month_fits["source_name"].astype(str).eq(source)
        & month_fits["stack_model"].eq("constant_plus_step")
    ].copy()
    if sub.empty:
        return None
    # Use the most stable/no-baseline variant first for inspection.
    preferred = "raw_zscore_no_baseline" if "raw_zscore_no_baseline" in set(sub["variant"]) else str(sub["variant"].iloc[0])
    sub = sub[sub["variant"].eq(preferred)].sort_values("month_block")
    if sub.empty:
        return None
    fig, ax = plt.subplots(figsize=(9.5, 4.3))
    vals = pd.to_numeric(sub["stack_fit_snr"], errors="coerce").to_numpy(dtype=float)
    labels = sub["month_block"].astype(str).to_list()
    colors = ["#4c78a8" if v >= 0 else "#d95f02" for v in vals]
    ax.bar(labels, vals, color=colors)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(3, color="0.6", lw=0.8, ls="--")
    ax.axhline(-3, color="0.6", lw=0.8, ls="--")
    for i, (_, row) in enumerate(sub.iterrows()):
        ax.text(i, vals[i], f"n={int(row['total_events_used'])}", ha="center", va="bottom" if vals[i] >= 0 else "top", fontsize=8)
    ax.set_ylabel("Stack-first fit SNR")
    ax.set_title(f"{_title_source(source)} month-cluster stack fits ({VARIANT_LABELS.get(preferred, preferred)})")
    ax.tick_params(axis="x", labelrotation=35)
    fig.tight_layout()
    path = out_dir / source / f"{source}_month_cluster_stackfit.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_report(out_dir: Path, summary: pd.DataFrame, month_fits: pd.DataFrame, paths: dict[str, list[Path]]) -> None:
    best = (
        summary[summary["stack_group"].eq("combined")]
        .assign(abs_snr=lambda d: pd.to_numeric(d["stack_fit_snr"], errors="coerce").abs())
        .sort_values(["source_name", "abs_snr"], ascending=[True, False])
        .groupby("source_name")
        .head(6)
    )
    lines = [
        "# Stack-First Fit Diagnostics",
        "",
        "This run addresses the concern that sparse individual-event fits may be misleading.",
        "",
        "Workflow:",
        "",
        "1. Extract event windows for the selected source/channel.",
        "2. Normalize each event window.",
        "3. Stack samples by time relative to predicted occultation.",
        "4. Fit the occultation template to the stacked profile, not to individual events.",
        "",
        "Variants:",
        "",
        "- `raw_fractional_no_baseline`: divide each event by its local sideband median; no time-dependent baseline subtraction.",
        "- `raw_zscore_no_baseline`: subtract a local sideband median and divide by sideband robust sigma; no time-dependent baseline subtraction.",
        "- `sideband_constant_subtracted`: subtract a constant sideband level before stacking.",
        "- `sideband_linear_subtracted`: subtract a linear sideband baseline before stacking.",
        "",
        "The fit still has a signed amplitude. Positive means the stacked profile follows the expected positive-source occultation sign. Negative means the stacked profile changes opposite to that template.",
        "",
        "## Best Combined Stack Fits",
        "",
        "```\n"
        + best[
            [
                "source_name",
                "frequency_mhz",
                "antenna",
                "variant",
                "stack_model",
                "total_events_used",
                "median_events_per_bin",
                "max_events_per_bin",
                "amplitude",
                "uncertainty",
                "stack_fit_snr",
                "event_bootstrap_snr",
                "event_bootstrap_sign_fraction",
                "leave_one_month_min_abs_snr",
                "leave_one_month_sign_fraction",
                "delta_bic",
                "best_timing_offset_s",
            ]
        ].to_string(index=False)
        + "\n```",
        "",
        "## Figures",
        "",
    ]
    if not month_fits.empty:
        month_best = (
            month_fits.assign(abs_snr=lambda d: pd.to_numeric(d["stack_fit_snr"], errors="coerce").abs())
            .sort_values(["source_name", "abs_snr"], ascending=[True, False])
            .groupby("source_name")
            .head(5)
        )
        lines.extend(
            [
                "## Strongest Month/Date-Cluster Fits",
                "",
                "These are diagnostic for episodic behavior. A source that appears only in one month should not be treated like a stable continuum detection.",
                "",
                "```\n"
                + month_best[
                    [
                        "source_name",
                        "month_block",
                        "frequency_mhz",
                        "antenna",
                        "variant",
                        "stack_model",
                        "total_events_used",
                        "amplitude",
                        "stack_fit_snr",
                        "delta_bic",
                        "best_timing_offset_s",
                    ]
                ].to_string(index=False)
                + "\n```",
                "",
            ]
        )
    for source, source_paths in paths.items():
        lines.extend([f"### {_title_source(source)}", ""])
        for path in source_paths:
            lines.append(f"- {path}")
        lines.append("")
    (out_dir / "stack_first_fit_diagnostics_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--survey-root", default="outputs/planetary_confirmation_survey_science_baseline_v2")
    parser.add_argument("--selected-channels", default="outputs/fit_inspection_plots_lower_v_fitquality_v1/lower_v_selected_channels.csv")
    parser.add_argument("--output-dir", default="outputs/stack_first_fit_diagnostics_lower_v_v1")
    parser.add_argument("--sources", nargs="+", default=["earth", "sun", "jupiter"])
    parser.add_argument("--bin-seconds", type=float, default=60.0)
    parser.add_argument("--sideband-exclusion-seconds", type=float, default=120.0)
    parser.add_argument("--timing-offsets", nargs="+", type=float, default=[-300, -240, -180, -120, -90, -60, -30, 0, 30, 60, 90, 120, 180, 240, 300])
    parser.add_argument("--n-bootstrap", type=int, default=128)
    parser.add_argument("--bootstrap-seed", type=int, default=12345)
    parser.add_argument("--write-month-splits", action="store_true", help="Fit stacks separately by month/date block.")
    args = parser.parse_args()

    out_dir = ensure_dir(ROOT / args.output_dir)
    clean = _read(CLEAN, parse_dates=["time"])
    events = _read(ROOT / args.survey_root / "events" / "all_planet_predicted_events.csv", parse_dates=["predicted_event_time"])
    selected = _read(ROOT / args.selected_channels)
    selected = selected[selected["source_name"].astype(str).isin(args.sources)].copy()
    variants = list(VARIANT_LABELS)

    all_profiles = []
    all_stacks = []
    all_fits = []
    all_month_fits = []
    paths: dict[str, list[Path]] = {}
    for _, row in selected.iterrows():
        source = str(row["source_name"])
        source_dir = ensure_dir(out_dir / source)
        profiles = _collect_profiles(
            clean,
            events,
            row,
            variants,
            float(args.bin_seconds),
            float(args.sideband_exclusion_seconds),
        )
        profiles.to_csv(source_dir / f"{source}_stack_first_profiles.csv", index=False)
        stacked = _stack_profiles(profiles)
        stacked.to_csv(source_dir / f"{source}_stack_first_binned_profiles.csv", index=False)
        fits = _fit_summary(stacked, profiles, args.timing_offsets)
        fits, leave_month = _event_bootstrap_and_month_jackknife(
            profiles,
            fits,
            args.timing_offsets,
            n_bootstrap=int(args.n_bootstrap),
            seed=int(args.bootstrap_seed),
        )
        fits.to_csv(source_dir / f"{source}_stack_first_fit_summary.csv", index=False)
        leave_month.to_csv(source_dir / f"{source}_leave_one_month_stackfit.csv", index=False)
        month_fits = _month_cluster_fit_summary(profiles, args.timing_offsets) if args.write_month_splits else pd.DataFrame()
        month_fits.to_csv(source_dir / f"{source}_month_cluster_stackfit.csv", index=False)
        all_profiles.append(profiles)
        all_stacks.append(stacked)
        all_fits.append(fits)
        all_month_fits.append(month_fits)
        paths[source] = _plot_source(source, stacked, fits, out_dir)
        p = _plot_month_clusters(source, month_fits, out_dir)
        if p is not None:
            paths[source].append(p)

    profiles_out = pd.concat(all_profiles, ignore_index=True) if all_profiles else pd.DataFrame()
    stacks_out = pd.concat(all_stacks, ignore_index=True) if all_stacks else pd.DataFrame()
    fits_out = pd.concat(all_fits, ignore_index=True) if all_fits else pd.DataFrame()
    month_fits_out = pd.concat(all_month_fits, ignore_index=True) if all_month_fits else pd.DataFrame()
    profiles_out.to_csv(out_dir / "all_stack_first_profiles.csv", index=False)
    stacks_out.to_csv(out_dir / "all_stack_first_binned_profiles.csv", index=False)
    fits_out.to_csv(out_dir / "all_stack_first_fit_summary.csv", index=False)
    month_fits_out.to_csv(out_dir / "all_month_cluster_stackfit.csv", index=False)
    _write_report(out_dir, fits_out, month_fits_out, paths)
    write_json(out_dir / "run_config.json", vars(args))
    print(out_dir / "stack_first_fit_diagnostics_report.md")
    print(out_dir / "all_stack_first_fit_summary.csv")


if __name__ == "__main__":
    main()
