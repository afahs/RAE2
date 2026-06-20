#!/usr/bin/env python
"""Data-driven background subtraction for lower-V stack-first occultation profiles.

This script does not forward model the diffuse sky.  It estimates the
low-frequency nuisance component directly from wrong-time and wrong-sky stacks,
subtracts that empirical background from the real stack, and fits the residual
with the same finite-duration occultation template.  It also performs the same
subtraction on pseudo-real controls using leave-one-control-out backgrounds.
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
from rylevonberg.detection import baseline_matrix  # noqa: E402
from rylevonberg.stackfit import StackedStepFitConfig, fit_stacked_step, stacked_event_template  # noqa: E402
from rylevonberg.util import ensure_dir, robust_sigma, software_versions, write_json  # noqa: E402


STACKFIRST_ROOT = ROOT / "outputs/lower_v_stackfirst_detection_sun_fornax_v1"
STACKS = STACKFIRST_ROOT / "lower_v_stackfirst_control_stacks.csv"
DEFAULT_OUT = ROOT / "outputs/lower_v_empirical_background_subtraction_sun_fornax_v1"

SOURCE_LABEL = {"sun": "Sun", "fornax_a": "Fornax-A"}
METHOD_LABEL = {
    "time_shift": "time-shift background",
    "offsource": "off-source background",
    "hybrid_time_off": "hybrid time-shift + off-source background",
}
METHOD_FAMILIES = {
    "time_shift": ("time_shift",),
    "offsource": ("offsource",),
    "hybrid_time_off": ("time_shift", "offsource"),
}


def _read(path: Path, **kwargs) -> pd.DataFrame:
    return read_table(path, low_memory=False, **kwargs)


def _robust_sem(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size <= 1:
        return np.nan
    scale = robust_sigma(vals - np.nanmedian(vals))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.nanstd(vals, ddof=1))
    return float(scale / np.sqrt(vals.size)) if np.isfinite(scale) and scale > 0 else np.nan


def _background_pool(stacks: pd.DataFrame, source: str, families: tuple[str, ...]) -> pd.DataFrame:
    return stacks[
        stacks["analysis_source"].astype(str).eq(source)
        & stacks["control_family"].astype(str).isin(families)
    ].copy()


def _background_summary(pool: pd.DataFrame, method: str) -> pd.DataFrame:
    rows = []
    by = ["analysis_source", "frequency_band", "frequency_mhz", "event_type", "t_bin_sec"]
    for keys, grp in pool.groupby(by, sort=True, dropna=False):
        vals = pd.to_numeric(grp["median_z_power"], errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue
        spread = (np.nanquantile(vals, 0.75) - np.nanquantile(vals, 0.25)) / 1.349 if vals.size > 1 else np.nan
        rows.append(
            {
                **dict(zip(by, keys)),
                "background_method": method,
                "background_median": float(np.nanmedian(vals)),
                "background_q25": float(np.nanquantile(vals, 0.25)),
                "background_q75": float(np.nanquantile(vals, 0.75)),
                "background_robust_sigma": float(spread) if np.isfinite(spread) else np.nan,
                "n_background_curves": int(grp["control_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def _real_residuals(stacks: pd.DataFrame, bg: pd.DataFrame, method: str) -> pd.DataFrame:
    real = stacks[stacks["control_family"].astype(str).eq("real")].copy()
    keys = ["analysis_source", "frequency_band", "frequency_mhz", "event_type", "t_bin_sec"]
    merged = real.merge(bg, on=keys, how="inner")
    if merged.empty:
        return pd.DataFrame()
    sigma = np.sqrt(
        np.square(pd.to_numeric(merged["robust_sem_z_power"], errors="coerce").to_numpy(dtype=float))
        + np.square(pd.to_numeric(merged["background_robust_sigma"], errors="coerce").fillna(0.0).to_numpy(dtype=float))
    )
    fallback = np.nanmedian(sigma[np.isfinite(sigma) & (sigma > 0)])
    if not np.isfinite(fallback) or fallback <= 0:
        fallback = 1.0
    merged["residual_z_power"] = merged["median_z_power"] - merged["background_median"]
    merged["residual_uncertainty"] = np.where(np.isfinite(sigma) & (sigma > 0), sigma, fallback)
    merged["background_method"] = method
    merged["residual_role"] = "real"
    merged["pseudo_control_id"] = "real"
    return merged


def _pseudo_control_residuals(pool: pd.DataFrame, method: str) -> pd.DataFrame:
    rows = []
    keys = ["analysis_source", "frequency_band", "frequency_mhz", "event_type", "t_bin_sec"]
    for control_id, control in pool.groupby("control_id", sort=True, dropna=False):
        other = pool[~pool["control_id"].astype(str).eq(str(control_id))].copy()
        if other.empty:
            continue
        bg = _background_summary(other, method)
        merged = control.merge(bg, on=keys, how="inner")
        if merged.empty:
            continue
        sigma = np.sqrt(
            np.square(pd.to_numeric(merged["robust_sem_z_power"], errors="coerce").to_numpy(dtype=float))
            + np.square(pd.to_numeric(merged["background_robust_sigma"], errors="coerce").fillna(0.0).to_numpy(dtype=float))
        )
        fallback = np.nanmedian(sigma[np.isfinite(sigma) & (sigma > 0)])
        if not np.isfinite(fallback) or fallback <= 0:
            fallback = 1.0
        merged["residual_z_power"] = merged["median_z_power"] - merged["background_median"]
        merged["residual_uncertainty"] = np.where(np.isfinite(sigma) & (sigma > 0), sigma, fallback)
        merged["background_method"] = method
        merged["residual_role"] = "pseudo_control"
        merged["pseudo_control_id"] = str(control_id)
        rows.append(merged)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_residual_tables(stacks: pd.DataFrame, methods: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    residuals = []
    backgrounds = []
    for method in methods:
        families = METHOD_FAMILIES[method]
        for source in sorted(stacks["analysis_source"].dropna().astype(str).unique()):
            pool = _background_pool(stacks, source, families)
            if pool.empty:
                continue
            bg = _background_summary(pool, method)
            backgrounds.append(bg)
            real = _real_residuals(stacks[stacks["analysis_source"].astype(str).eq(source)].copy(), bg, method)
            if not real.empty:
                residuals.append(real)
            pseudo = _pseudo_control_residuals(pool, method)
            if not pseudo.empty:
                residuals.append(pseudo)
    out_res = pd.concat(residuals, ignore_index=True) if residuals else pd.DataFrame()
    out_bg = pd.concat(backgrounds, ignore_index=True) if backgrounds else pd.DataFrame()
    return out_res, out_bg


def fit_residuals(residuals: pd.DataFrame, timing_offsets: list[float], transition_durations: list[float]) -> pd.DataFrame:
    rows = []
    cfg = StackedStepFitConfig(
        baseline_order=0,
        timing_offsets_seconds=tuple(float(x) for x in timing_offsets),
        transition_durations_seconds=tuple(float(x) for x in transition_durations),
        min_bins=8,
    )
    group_cols = ["analysis_source", "background_method", "residual_role", "pseudo_control_id", "frequency_band", "frequency_mhz"]
    for keys, grp in residuals.groupby(group_cols, sort=True, dropna=False):
        meta = dict(zip(group_cols, keys))
        for stack_group, sub in [("combined", grp), *[(et, g) for et, g in grp.groupby("event_type", sort=True)]]:
            sub = sub.sort_values(["event_type", "t_bin_sec"])
            fit = fit_stacked_step(
                sub["t_bin_sec"].to_numpy(dtype=float),
                sub["residual_z_power"].to_numpy(dtype=float),
                sub["event_type"].astype(str).to_numpy(),
                uncertainty=sub["residual_uncertainty"].to_numpy(dtype=float),
                config=cfg,
            )
            rows.append(
                {
                    **meta,
                    "stack_group": str(stack_group),
                    "median_windows_per_bin": float(np.nanmedian(sub["n_windows"])) if "n_windows" in sub.columns and not sub.empty else np.nan,
                    **fit,
                }
            )
    return pd.DataFrame(rows)


def empirical_summary(fits: pd.DataFrame) -> pd.DataFrame:
    real = fits[(fits["residual_role"].eq("real")) & fits["stack_group"].eq("combined")].copy()
    pseudo = fits[(fits["residual_role"].eq("pseudo_control")) & fits["stack_group"].eq("combined")].copy()
    rows = []
    for _, row in real.iterrows():
        same = pseudo[
            pseudo["analysis_source"].eq(row["analysis_source"])
            & pseudo["background_method"].eq(row["background_method"])
            & pseudo["frequency_band"].eq(row["frequency_band"])
        ].copy()
        vals = pd.to_numeric(same["amplitude"], errors="coerce").dropna().to_numpy(dtype=float)
        abs_vals = np.abs(vals)
        real_amp = float(row["amplitude"])
        rows.append(
            {
                "analysis_source": row["analysis_source"],
                "background_method": row["background_method"],
                "frequency_band": int(row["frequency_band"]),
                "frequency_mhz": float(row["frequency_mhz"]),
                "real_residual_amplitude": real_amp,
                "real_residual_uncertainty": float(row["uncertainty"]),
                "real_residual_fit_snr": float(row["stack_fit_snr"]),
                "real_delta_bic": float(row["delta_bic"]),
                "real_best_transition_duration_s": float(row.get("best_transition_duration_s", np.nan)),
                "n_pseudo_controls": int(len(vals)),
                "pseudo_median_amplitude": float(np.nanmedian(vals)) if len(vals) else np.nan,
                "pseudo_q25_amplitude": float(np.nanquantile(vals, 0.25)) if len(vals) else np.nan,
                "pseudo_q75_amplitude": float(np.nanquantile(vals, 0.75)) if len(vals) else np.nan,
                "pseudo_abs_q75_amplitude": float(np.nanquantile(abs_vals, 0.75)) if len(abs_vals) else np.nan,
                "empirical_p_amp_ge_real": float((1 + np.count_nonzero(vals >= real_amp)) / (1 + len(vals))) if len(vals) else np.nan,
                "empirical_p_abs_amp_ge_real": float((1 + np.count_nonzero(abs_vals >= abs(real_amp))) / (1 + len(vals))) if len(vals) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _model_curve(sub: pd.DataFrame, fit: pd.Series) -> np.ndarray:
    t = sub["t_bin_sec"].to_numpy(dtype=float)
    event_types = sub["event_type"].astype(str).to_numpy()
    tmpl = stacked_event_template(
        t,
        event_types,
        timing_offset_sec=float(fit["best_timing_offset_s"]),
        transition_duration_sec=float(fit["best_transition_duration_s"]),
    )
    X = np.column_stack([baseline_matrix(t, 0), tmpl])
    y = sub["residual_z_power"].to_numpy(dtype=float)
    sigma = sub["residual_uncertainty"].to_numpy(dtype=float)
    fallback = np.nanmedian(sigma[np.isfinite(sigma) & (sigma > 0)])
    if not np.isfinite(fallback) or fallback <= 0:
        fallback = 1.0
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, fallback)
    w = 1.0 / sigma**2
    beta, *_ = np.linalg.lstsq(X * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)
    return X @ beta


def _pseudo_envelope(residuals: pd.DataFrame, source: str, method: str) -> pd.DataFrame:
    pseudo = residuals[
        residuals["analysis_source"].eq(source)
        & residuals["background_method"].eq(method)
        & residuals["residual_role"].eq("pseudo_control")
    ].copy()
    if pseudo.empty:
        return pd.DataFrame()
    rows = []
    by = ["analysis_source", "background_method", "frequency_band", "frequency_mhz", "event_type", "t_bin_sec"]
    for keys, grp in pseudo.groupby(by, sort=True, dropna=False):
        vals = pd.to_numeric(grp["residual_z_power"], errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue
        rows.append(
            {
                **dict(zip(by, keys)),
                "pseudo_median_residual": float(np.nanmedian(vals)),
                "pseudo_q25_residual": float(np.nanquantile(vals, 0.25)),
                "pseudo_q75_residual": float(np.nanquantile(vals, 0.75)),
                "n_pseudo_points": int(vals.size),
            }
        )
    return pd.DataFrame(rows)


def plot_residual_grids(residuals: pd.DataFrame, backgrounds: pd.DataFrame, stacks: pd.DataFrame, method: str, out_dir: Path) -> list[Path]:
    paths = []
    for source in sorted(residuals.loc[residuals["residual_role"].eq("real"), "analysis_source"].dropna().unique()):
        freqs = sorted(residuals.loc[residuals["analysis_source"].eq(source), "frequency_mhz"].dropna().unique())
        fig, axes = plt.subplots(len(freqs), 2, figsize=(13.5, max(10, 1.45 * len(freqs))), sharex=True, sharey=False)
        if len(freqs) == 1:
            axes = np.asarray([axes])
        for i, freq in enumerate(freqs):
            for j, event_type in enumerate(["disappearance", "reappearance"]):
                ax = axes[i, j]
                real_stack = stacks[
                    stacks["analysis_source"].eq(source)
                    & stacks["control_family"].eq("real")
                    & np.isclose(stacks["frequency_mhz"].astype(float), float(freq))
                    & stacks["event_type"].eq(event_type)
                ].sort_values("t_bin_sec")
                bg = backgrounds[
                    backgrounds["analysis_source"].eq(source)
                    & backgrounds["background_method"].eq(method)
                    & np.isclose(backgrounds["frequency_mhz"].astype(float), float(freq))
                    & backgrounds["event_type"].eq(event_type)
                ].sort_values("t_bin_sec")
                real_res = residuals[
                    residuals["analysis_source"].eq(source)
                    & residuals["background_method"].eq(method)
                    & residuals["residual_role"].eq("real")
                    & np.isclose(residuals["frequency_mhz"].astype(float), float(freq))
                    & residuals["event_type"].eq(event_type)
                ].sort_values("t_bin_sec")
                if not real_stack.empty:
                    ax.plot(real_stack["t_bin_sec"] / 60.0, real_stack["median_z_power"], color="0.2", lw=1.0, marker=".", ms=2.5, label="real stack")
                if not bg.empty:
                    x = bg["t_bin_sec"].to_numpy(dtype=float) / 60.0
                    ax.plot(x, bg["background_median"], color="#d95f02", lw=1.0, label="empirical background")
                    ax.fill_between(x, bg["background_q25"], bg["background_q75"], color="#d95f02", alpha=0.15, linewidth=0)
                if not real_res.empty:
                    ax.errorbar(
                        real_res["t_bin_sec"] / 60.0,
                        real_res["residual_z_power"],
                        yerr=real_res["residual_uncertainty"],
                        color="#1f78b4",
                        ecolor="#1f78b4",
                        marker="o",
                        markersize=2.4,
                        linewidth=1.35,
                        elinewidth=0.5,
                        capsize=1.0,
                        label="real - empirical background",
                    )
                ax.axvline(0.0, color="black", linestyle=":", linewidth=0.8)
                ax.axhline(0.0, color="0.7", linewidth=0.7)
                ax.set_title(f"{float(freq):.2f} MHz {event_type}", fontsize=8.5)
                if j == 0:
                    ax.set_ylabel("normalized lower-V power")
                if i == len(freqs) - 1:
                    ax.set_xlabel("minutes from predicted event")
                ax.grid(True, color="0.92", linewidth=0.5)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        fig.legend(by_label.values(), by_label.keys(), loc="upper center", ncol=min(3, len(by_label)), frameon=False)
        fig.suptitle(f"{SOURCE_LABEL.get(source, source)} empirical background subtraction: {METHOD_LABEL[method]}", y=0.996)
        fig.tight_layout(rect=[0, 0, 1, 0.965])
        path = out_dir / f"{source}_{method}_background_subtracted_grid.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_residual_amplitude_spectra(summary: pd.DataFrame, out_dir: Path) -> list[Path]:
    paths = []
    for source, src in summary.groupby("analysis_source", sort=True):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
        for ax, method in zip(axes, ["time_shift", "offsource", "hybrid_time_off"]):
            sub = src[src["background_method"].eq(method)].sort_values("frequency_mhz")
            if sub.empty:
                continue
            x = sub["frequency_mhz"].to_numpy(dtype=float)
            ax.axhline(0.0, color="0.7", lw=0.8)
            ax.fill_between(
                x,
                sub["pseudo_q25_amplitude"],
                sub["pseudo_q75_amplitude"],
                color="0.55",
                alpha=0.25,
                label="pseudo-control IQR",
            )
            ax.plot(x, sub["pseudo_median_amplitude"], color="0.45", lw=1.0, label="pseudo-control median")
            ax.errorbar(
                x,
                sub["real_residual_amplitude"],
                yerr=sub["real_residual_uncertainty"],
                color="black",
                marker="o",
                lw=1.4,
                capsize=2.0,
                label="real residual fit",
            )
            ax.set_xscale("log")
            ax.set_xticks(list(FREQUENCY_MAP_MHZ.values()))
            ax.set_xticklabels([f"{v:g}" for v in FREQUENCY_MAP_MHZ.values()], rotation=45, ha="right")
            ax.set_title(METHOD_LABEL[method], fontsize=9)
            ax.set_xlabel("frequency (MHz)")
            ax.grid(True, color="0.9", lw=0.5)
        axes[0].set_ylabel("residual positive-source template amplitude")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
        fig.suptitle(f"{SOURCE_LABEL.get(source, source)} empirical-background residual amplitude spectra", y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        path = out_dir / f"{source}_residual_amplitude_spectra_by_background.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_top_residual_fits(residuals: pd.DataFrame, fits: pd.DataFrame, summary: pd.DataFrame, method: str, out_dir: Path) -> list[Path]:
    paths = []
    for source, src in summary[summary["background_method"].eq(method)].groupby("analysis_source", sort=True):
        ranked = src.copy()
        ranked["score"] = ranked["real_residual_amplitude"].where(ranked["real_residual_amplitude"] > 0, -np.inf)
        ranked = ranked.sort_values(["score", "real_delta_bic"], ascending=[False, False]).head(4)
        if ranked.empty:
            continue
        pseudo_env = _pseudo_envelope(residuals, source, method)
        fig, axes = plt.subplots(len(ranked), 2, figsize=(13.5, max(3.0, 2.7 * len(ranked))), squeeze=False, sharex=True)
        for i, (_, row) in enumerate(ranked.iterrows()):
            freq = float(row["frequency_mhz"])
            fit = fits[
                fits["analysis_source"].eq(source)
                & fits["background_method"].eq(method)
                & fits["residual_role"].eq("real")
                & fits["stack_group"].eq("combined")
                & np.isclose(fits["frequency_mhz"].astype(float), freq)
            ].iloc[0]
            real = residuals[
                residuals["analysis_source"].eq(source)
                & residuals["background_method"].eq(method)
                & residuals["residual_role"].eq("real")
                & np.isclose(residuals["frequency_mhz"].astype(float), freq)
            ].copy()
            real = real.sort_values(["event_type", "t_bin_sec"])
            real["model"] = _model_curve(real, fit)
            for j, event_type in enumerate(["disappearance", "reappearance"]):
                ax = axes[i, j]
                sub = real[real["event_type"].eq(event_type)].sort_values("t_bin_sec")
                env = pseudo_env[
                    np.isclose(pseudo_env["frequency_mhz"].astype(float), freq)
                    & pseudo_env["event_type"].eq(event_type)
                ].sort_values("t_bin_sec")
                if not env.empty:
                    x_env = env["t_bin_sec"].to_numpy(dtype=float) / 60.0
                    ax.fill_between(x_env, env["pseudo_q25_residual"], env["pseudo_q75_residual"], color="0.65", alpha=0.28, linewidth=0, label="pseudo-control residual IQR")
                    ax.plot(x_env, env["pseudo_median_residual"], color="0.45", lw=0.9, label="pseudo-control median")
                if not sub.empty:
                    x = sub["t_bin_sec"].to_numpy(dtype=float) / 60.0
                    ax.errorbar(
                        x,
                        sub["residual_z_power"],
                        yerr=sub["residual_uncertainty"],
                        color="#1f78b4",
                        ecolor="#1f78b4",
                        marker="o",
                        ms=2.5,
                        lw=1.1,
                        elinewidth=0.55,
                        capsize=1.0,
                        label="real residual",
                    )
                    ax.plot(x, sub["model"], color="#e41a1c", lw=1.8, label="template fit")
                ax.axvline(0.0, color="black", ls=":", lw=0.8)
                ax.axhline(0.0, color="0.7", lw=0.7)
                ax.grid(True, color="0.92", lw=0.5)
                ax.set_title(event_type, fontsize=9)
                if j == 0:
                    ax.set_ylabel(f"{freq:.2f} MHz\nresidual power")
                if i == len(ranked) - 1:
                    ax.set_xlabel("minutes from predicted event")
                if i == 0 and j == 1:
                    ax.legend(frameon=False, fontsize=7)
            axes[i, 1].text(
                1.02,
                0.5,
                f"A={float(row['real_residual_amplitude']):.3f}\n"
                f"unc={float(row['real_residual_uncertainty']):.3f}\n"
                f"fitSNR={float(row['real_residual_fit_snr']):.2f}\n"
                f"DeltaBIC={float(row['real_delta_bic']):.1f}\n"
                f"tau={float(row['real_best_transition_duration_s']):.0f}s\n"
                f"p_amp={float(row['empirical_p_amp_ge_real']):.2f}\n"
                f"p_abs={float(row['empirical_p_abs_amp_ge_real']):.2f}",
                transform=axes[i, 1].transAxes,
                ha="left",
                va="center",
                fontsize=8,
            )
        fig.suptitle(f"{SOURCE_LABEL.get(source, source)} residual fits after {METHOD_LABEL[method]}", y=0.995)
        fig.tight_layout(rect=[0, 0, 0.91, 0.96])
        path = out_dir / f"{source}_{method}_top_residual_fit_profiles.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def write_report(out_dir: Path, summary: pd.DataFrame, paths: list[Path]) -> None:
    cols = [
        "analysis_source",
        "background_method",
        "frequency_mhz",
        "real_residual_amplitude",
        "real_residual_uncertainty",
        "real_residual_fit_snr",
        "real_delta_bic",
        "real_best_transition_duration_s",
        "n_pseudo_controls",
        "pseudo_median_amplitude",
        "pseudo_q25_amplitude",
        "pseudo_q75_amplitude",
        "empirical_p_amp_ge_real",
        "empirical_p_abs_amp_ge_real",
    ]
    lines = [
        "# Lower-V Empirical Background Subtraction Detection Attempt",
        "",
        "This is a data-driven subtraction run. It does not use a diffuse-sky or beam forward model.",
        "",
        "For each source/frequency/event type, the empirical background is estimated from control stacks and subtracted from the real stack:",
        "",
        "    residual(t) = real_stack(t) - median(control_stacks(t))",
        "",
        "The residual is then fit with the same finite-duration occultation template. Pseudo-control residuals are generated by subtracting a leave-one-control-out background from each control curve.",
        "",
        "The hybrid method uses both time-shift and off-source controls and is the main diagnostic.",
        "",
        "## Residual Fit Summary",
        "",
        summary[cols].sort_values(["analysis_source", "background_method", "frequency_mhz"]).to_string(index=False),
        "",
        "## Plots",
        "",
    ]
    lines.extend(f"- `{path.name}`" for path in paths)
    lines += [
        "",
        "## Reading The Plots",
        "",
        "- Background-subtracted grids show the real stack, empirical control background, and residual on the same axes.",
        "- Residual-fit plots show the residual data and template fit, with pseudo-control residual bands in gray.",
        "- A useful extraction should leave a residual with the expected disappearance/reappearance sign that is outside the pseudo-control residual band.",
    ]
    (out_dir / "empirical_background_subtraction_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack-root", default=str(STACKFIRST_ROOT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--methods", default="time_shift,offsource,hybrid_time_off")
    parser.add_argument("--timing-offsets-s", default="0")
    parser.add_argument("--transition-durations-s", default="0,120,300,600,900")
    args = parser.parse_args()

    stack_root = Path(args.stack_root)
    out_dir = ensure_dir(Path(args.out_dir))
    methods = [x.strip() for x in str(args.methods).split(",") if x.strip()]
    timing_offsets = [float(x.strip()) for x in str(args.timing_offsets_s).split(",") if x.strip()]
    transition_durations = [float(x.strip()) for x in str(args.transition_durations_s).split(",") if x.strip()]
    bad = sorted(set(methods) - set(METHOD_FAMILIES))
    if bad:
        raise SystemExit(f"Unknown method(s): {', '.join(bad)}")
    write_json(
        out_dir / "run_config.json",
        {
            "stack_root": str(stack_root),
            "methods": methods,
            "timing_offsets_s": timing_offsets,
            "transition_durations_s": transition_durations,
            "software_versions": software_versions(),
        },
    )

    stacks = _read(stack_root / "lower_v_stackfirst_control_stacks.csv")
    print("Building empirical background residuals...", flush=True)
    residuals, backgrounds = build_residual_tables(stacks, methods)
    residuals.to_csv(out_dir / "empirical_background_residual_profiles.csv", index=False)
    backgrounds.to_csv(out_dir / "empirical_background_profiles.csv", index=False)
    print("Fitting residual stacks...", flush=True)
    fits = fit_residuals(residuals, timing_offsets, transition_durations)
    fits.to_csv(out_dir / "empirical_background_residual_fit_by_group.csv", index=False)
    summary = empirical_summary(fits)
    summary.to_csv(out_dir / "empirical_background_residual_fit_summary.csv", index=False)
    print("Writing plots...", flush=True)
    paths: list[Path] = []
    for method in methods:
        paths.extend(plot_residual_grids(residuals, backgrounds, stacks, method, out_dir))
    paths.extend(plot_residual_amplitude_spectra(summary, out_dir))
    if "hybrid_time_off" in methods:
        paths.extend(plot_top_residual_fits(residuals, fits, summary, "hybrid_time_off", out_dir))
    write_report(out_dir, summary, paths)
    print(out_dir / "empirical_background_subtraction_report.md")


if __name__ == "__main__":
    main()
