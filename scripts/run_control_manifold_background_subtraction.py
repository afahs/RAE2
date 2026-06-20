#!/usr/bin/env python
"""Control-manifold background subtraction for lower-V occultation stacks.

This is a data-driven subtraction stage. It does not use a diffuse-sky or beam
forward model. Instead, it learns the nuisance profile shapes from wrong-time,
wrong-sky, and randomized control stacks. The real stack is projected onto that
control manifold using only sideband bins away from the predicted occultation
time, then the predicted nuisance profile is subtracted across the full window.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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
DEFAULT_OUT = ROOT / "outputs/lower_v_control_manifold_background_sun_fornax_v1"

SOURCE_LABEL = {"sun": "Sun", "fornax_a": "Fornax-A"}
FAMILY_SETS = {
    "time_off": ("time_shift", "offsource"),
    "all_controls": ("time_shift", "offsource", "randomized_time"),
}
EVENT_ORDER = {"disappearance": 0, "reappearance": 1}


@dataclass(frozen=True)
class ManifoldModel:
    columns: pd.DataFrame
    center: np.ndarray
    scale: np.ndarray
    pc_center: np.ndarray
    components: np.ndarray
    sideband_mask: np.ndarray
    n_controls: int
    n_components_available: int


def _read(path: Path, **kwargs) -> pd.DataFrame:
    return read_table(path, low_memory=False, **kwargs)


def _robust_column_scale(values: np.ndarray) -> np.ndarray:
    scales = []
    for j in range(values.shape[1]):
        col = values[:, j]
        col = col[np.isfinite(col)]
        if col.size <= 1:
            scales.append(np.nan)
            continue
        sigma = robust_sigma(col - np.nanmedian(col))
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = float(np.nanstd(col, ddof=1))
        scales.append(float(sigma) if np.isfinite(sigma) and sigma > 0 else np.nan)
    out = np.asarray(scales, dtype=float)
    fallback = np.nanmedian(out[np.isfinite(out) & (out > 0)])
    if not np.isfinite(fallback) or fallback <= 0:
        fallback = 1.0
    return np.where(np.isfinite(out) & (out > 0), out, fallback)


def _curve_matrix(
    stacks: pd.DataFrame,
    source: str,
    frequency_mhz: float,
    families: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub = stacks[
        stacks["analysis_source"].astype(str).eq(source)
        & np.isclose(stacks["frequency_mhz"].astype(float), float(frequency_mhz))
        & stacks["control_family"].astype(str).isin(families)
    ].copy()
    if sub.empty:
        return pd.DataFrame(), pd.DataFrame()
    sub["event_sort"] = sub["event_type"].astype(str).map(EVENT_ORDER).fillna(99).astype(int)
    sub = sub.sort_values(["control_family", "control_id", "event_sort", "t_bin_sec"])
    cols = (
        sub[["event_type", "t_bin_sec"]]
        .drop_duplicates()
        .sort_values(["event_type", "t_bin_sec"], key=lambda s: s.map(EVENT_ORDER).fillna(99) if s.name == "event_type" else s)
        .reset_index(drop=True)
    )
    rows = []
    meta = []
    for (family, control_id), grp in sub.groupby(["control_family", "control_id"], sort=True):
        merged = cols.merge(grp[["event_type", "t_bin_sec", "median_z_power"]], on=["event_type", "t_bin_sec"], how="left")
        vec = pd.to_numeric(merged["median_z_power"], errors="coerce").to_numpy(dtype=float)
        if np.count_nonzero(np.isfinite(vec)) < max(8, int(0.75 * len(vec))):
            continue
        rows.append(vec)
        meta.append({"control_family": str(family), "control_id": str(control_id)})
    if not rows:
        return pd.DataFrame(), cols
    matrix = pd.DataFrame(np.vstack(rows))
    meta_frame = pd.DataFrame(meta)
    return pd.concat([meta_frame, matrix], axis=1), cols


def _real_curve(stacks: pd.DataFrame, source: str, frequency_mhz: float) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    sub = stacks[
        stacks["analysis_source"].astype(str).eq(source)
        & np.isclose(stacks["frequency_mhz"].astype(float), float(frequency_mhz))
        & stacks["control_family"].astype(str).eq("real")
    ].copy()
    if sub.empty:
        return np.array([]), np.array([]), pd.DataFrame()
    sub["event_sort"] = sub["event_type"].astype(str).map(EVENT_ORDER).fillna(99).astype(int)
    sub = sub.sort_values(["event_sort", "t_bin_sec"]).reset_index(drop=True)
    y = pd.to_numeric(sub["median_z_power"], errors="coerce").to_numpy(dtype=float)
    sigma = pd.to_numeric(sub["robust_sem_z_power"], errors="coerce").to_numpy(dtype=float)
    fallback = np.nanmedian(sigma[np.isfinite(sigma) & (sigma > 0)])
    if not np.isfinite(fallback) or fallback <= 0:
        fallback = 1.0
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, fallback)
    return y, sigma, sub


def _build_model(control_matrix: pd.DataFrame, columns: pd.DataFrame, sideband_s: float) -> ManifoldModel | None:
    if control_matrix.empty:
        return None
    values = control_matrix.drop(columns=["control_family", "control_id"]).to_numpy(dtype=float)
    finite_cols = np.mean(np.isfinite(values), axis=0) >= 0.8
    if np.count_nonzero(finite_cols) < 8:
        return None
    values = values[:, finite_cols]
    cols = columns.loc[finite_cols].reset_index(drop=True)
    col_medians = np.nanmedian(values, axis=0)
    inds = np.where(~np.isfinite(values))
    values[inds] = np.take(col_medians, inds[1])
    center = np.nanmedian(values, axis=0)
    scale = _robust_column_scale(values)
    standardized = (values - center) / scale
    pc_center = np.nanmean(standardized, axis=0)
    standardized = standardized - pc_center
    _, _, vt = np.linalg.svd(standardized, full_matrices=False)
    sideband_mask = np.abs(cols["t_bin_sec"].to_numpy(dtype=float)) >= float(sideband_s)
    if np.count_nonzero(sideband_mask) < 6:
        return None
    return ManifoldModel(
        columns=cols,
        center=center,
        scale=scale,
        pc_center=pc_center,
        components=vt,
        sideband_mask=sideband_mask,
        n_controls=int(values.shape[0]),
        n_components_available=int(vt.shape[0]),
    )


def _predict_background(y: np.ndarray, model: ManifoldModel, n_components: int) -> tuple[np.ndarray, dict[str, float]]:
    k = min(int(n_components), int(model.n_components_available), max(int(model.n_controls) - 2, 1))
    z = (np.asarray(y, dtype=float) - model.center) / model.scale
    z_centered = z - model.pc_center
    mask = model.sideband_mask & np.isfinite(z)
    if np.count_nonzero(mask) < max(6, k + 2):
        return np.full_like(z, np.nan, dtype=float), {"sideband_rms": np.nan, "n_sideband_bins": int(np.count_nonzero(mask)), "n_components_used": int(k)}
    basis = model.components[:k]
    x = np.column_stack([np.ones(np.count_nonzero(mask)), basis[:, mask].T])
    beta, *_ = np.linalg.lstsq(x, z_centered[mask], rcond=None)
    pred_z = model.pc_center + beta[0] + beta[1:] @ basis
    pred = model.center + model.scale * pred_z
    side_resid = z_centered[mask] - (beta[0] + beta[1:] @ basis[:, mask])
    return pred, {
        "sideband_rms": float(np.sqrt(np.nanmean(side_resid**2))) if side_resid.size else np.nan,
        "n_sideband_bins": int(np.count_nonzero(mask)),
        "n_components_used": int(k),
    }


def _align_to_model(y: np.ndarray, sigma: np.ndarray, real_meta: pd.DataFrame, model: ManifoldModel) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    real = real_meta[["event_type", "t_bin_sec"]].copy()
    real["value"] = y
    real["sigma"] = sigma
    merged = model.columns.merge(real, on=["event_type", "t_bin_sec"], how="left")
    yy = pd.to_numeric(merged["value"], errors="coerce").to_numpy(dtype=float)
    ss = pd.to_numeric(merged["sigma"], errors="coerce").to_numpy(dtype=float)
    fallback = np.nanmedian(ss[np.isfinite(ss) & (ss > 0)])
    if not np.isfinite(fallback) or fallback <= 0:
        fallback = 1.0
    ss = np.where(np.isfinite(ss) & (ss > 0), ss, fallback)
    return yy, ss, merged[["event_type", "t_bin_sec"]].copy()


def _control_uncertainty(control_matrix: pd.DataFrame, columns: pd.DataFrame, model: ManifoldModel) -> np.ndarray:
    values = control_matrix.drop(columns=["control_family", "control_id"]).to_numpy(dtype=float)
    control_cols = columns.reset_index(drop=True)
    keep = model.columns.merge(control_cols.reset_index(), on=["event_type", "t_bin_sec"], how="left")["index"].to_numpy()
    values = values[:, keep]
    spread = _robust_column_scale(values)
    return np.where(np.isfinite(spread) & (spread > 0), spread, np.nanmedian(spread[np.isfinite(spread) & (spread > 0)]))


def _fit_residual_profile(profile: pd.DataFrame, timing_offsets: list[float], transition_durations: list[float]) -> dict[str, float]:
    cfg = StackedStepFitConfig(
        baseline_order=0,
        timing_offsets_seconds=tuple(float(x) for x in timing_offsets),
        transition_durations_seconds=tuple(float(x) for x in transition_durations),
        min_bins=8,
    )
    return fit_stacked_step(
        profile["t_bin_sec"].to_numpy(dtype=float),
        profile["residual_z_power"].to_numpy(dtype=float),
        profile["event_type"].astype(str).to_numpy(),
        uncertainty=profile["residual_uncertainty"].to_numpy(dtype=float),
        config=cfg,
    )


def _model_curve(profile: pd.DataFrame, fit: pd.Series | dict[str, float]) -> np.ndarray:
    t = profile["t_bin_sec"].to_numpy(dtype=float)
    event_types = profile["event_type"].astype(str).to_numpy()
    tmpl = stacked_event_template(
        t,
        event_types,
        timing_offset_sec=float(fit["best_timing_offset_s"]),
        transition_duration_sec=float(fit["best_transition_duration_s"]),
    )
    x = np.column_stack([baseline_matrix(t, 0), tmpl])
    y = profile["residual_z_power"].to_numpy(dtype=float)
    sigma = profile["residual_uncertainty"].to_numpy(dtype=float)
    fallback = np.nanmedian(sigma[np.isfinite(sigma) & (sigma > 0)])
    if not np.isfinite(fallback) or fallback <= 0:
        fallback = 1.0
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, fallback)
    w = 1.0 / sigma**2
    beta, *_ = np.linalg.lstsq(x * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)
    return x @ beta


def build_manifold_residuals(
    stacks: pd.DataFrame,
    family_set_name: str,
    n_components_list: list[int],
    sideband_s: float,
    timing_offsets: list[float],
    transition_durations: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    families = FAMILY_SETS[family_set_name]
    residual_rows: list[dict[str, object]] = []
    fit_rows: list[dict[str, object]] = []
    diag_rows: list[dict[str, object]] = []

    for source in sorted(stacks["analysis_source"].dropna().astype(str).unique()):
        freqs = sorted(stacks.loc[stacks["analysis_source"].astype(str).eq(source), "frequency_mhz"].dropna().astype(float).unique())
        for freq in freqs:
            controls, columns = _curve_matrix(stacks, source, freq, families)
            model = _build_model(controls, columns, sideband_s=sideband_s)
            if model is None:
                continue
            real_y, real_sigma, real_meta = _real_curve(stacks, source, freq)
            if real_meta.empty:
                continue
            y, sigma, meta = _align_to_model(real_y, real_sigma, real_meta, model)
            control_spread = _control_uncertainty(controls, columns, model)
            residual_groups = []
            for n_components in n_components_list:
                bg, pred_diag = _predict_background(y, model, n_components)
                uncertainty = np.sqrt(np.square(sigma) + np.square(control_spread))
                profile = meta.copy()
                profile["analysis_source"] = source
                profile["family_set"] = family_set_name
                profile["frequency_mhz"] = float(freq)
                profile["frequency_band"] = int(
                    stacks.loc[
                        stacks["analysis_source"].astype(str).eq(source)
                        & np.isclose(stacks["frequency_mhz"].astype(float), float(freq)),
                        "frequency_band",
                    ].iloc[0]
                )
                profile["n_components_requested"] = int(n_components)
                profile["n_components_used"] = int(pred_diag["n_components_used"])
                profile["residual_role"] = "real"
                profile["pseudo_control_id"] = "real"
                profile["real_stack_z_power"] = y
                profile["manifold_background_z_power"] = bg
                profile["residual_z_power"] = y - bg
                profile["residual_uncertainty"] = uncertainty
                profile["is_sideband_fit_bin"] = model.sideband_mask
                residual_rows.extend(profile.replace([np.inf, -np.inf], np.nan).to_dict("records"))
                residual_groups.append(profile)
                fit = _fit_residual_profile(profile, timing_offsets, transition_durations)
                fit_rows.append(
                    {
                        "analysis_source": source,
                        "family_set": family_set_name,
                        "frequency_mhz": float(freq),
                        "frequency_band": int(profile["frequency_band"].iloc[0]),
                        "n_components_requested": int(n_components),
                        "n_components_used": int(pred_diag["n_components_used"]),
                        "residual_role": "real",
                        "pseudo_control_id": "real",
                        "n_control_curves": int(model.n_controls),
                        **pred_diag,
                        **fit,
                    }
                )

            for control_idx, control_row in controls.iterrows():
                control_id = str(control_row["control_id"])
                family = str(control_row["control_family"])
                others = controls.drop(index=control_idx).reset_index(drop=True)
                loo_model = _build_model(others, columns, sideband_s=sideband_s)
                if loo_model is None:
                    continue
                vec = control_row.drop(labels=["control_family", "control_id"]).to_numpy(dtype=float)
                c_meta = columns.copy()
                c_y, c_sigma, c_meta2 = _align_to_model(vec, np.ones_like(vec, dtype=float), c_meta.assign(value=vec, sigma=1.0), loo_model)
                c_spread = _control_uncertainty(others, columns, loo_model)
                for n_components in n_components_list:
                    bg, pred_diag = _predict_background(c_y, loo_model, n_components)
                    c_profile = c_meta2.copy()
                    c_profile["analysis_source"] = source
                    c_profile["family_set"] = family_set_name
                    c_profile["frequency_mhz"] = float(freq)
                    c_profile["frequency_band"] = int(
                        stacks.loc[
                            stacks["analysis_source"].astype(str).eq(source)
                            & np.isclose(stacks["frequency_mhz"].astype(float), float(freq)),
                            "frequency_band",
                        ].iloc[0]
                    )
                    c_profile["n_components_requested"] = int(n_components)
                    c_profile["n_components_used"] = int(pred_diag["n_components_used"])
                    c_profile["residual_role"] = "pseudo_control"
                    c_profile["pseudo_control_id"] = control_id
                    c_profile["pseudo_control_family"] = family
                    c_profile["real_stack_z_power"] = c_y
                    c_profile["manifold_background_z_power"] = bg
                    c_profile["residual_z_power"] = c_y - bg
                    c_profile["residual_uncertainty"] = c_spread
                    c_profile["is_sideband_fit_bin"] = loo_model.sideband_mask
                    residual_rows.extend(c_profile.replace([np.inf, -np.inf], np.nan).to_dict("records"))
                    fit = _fit_residual_profile(c_profile, timing_offsets, transition_durations)
                    fit_rows.append(
                        {
                            "analysis_source": source,
                            "family_set": family_set_name,
                            "frequency_mhz": float(freq),
                            "frequency_band": int(c_profile["frequency_band"].iloc[0]),
                            "n_components_requested": int(n_components),
                            "n_components_used": int(pred_diag["n_components_used"]),
                            "residual_role": "pseudo_control",
                            "pseudo_control_id": control_id,
                            "pseudo_control_family": family,
                            "n_control_curves": int(loo_model.n_controls),
                            **pred_diag,
                            **fit,
                        }
                    )

            for n_components in n_components_list:
                real_fit = [r for r in fit_rows if r["analysis_source"] == source and r["family_set"] == family_set_name and np.isclose(r["frequency_mhz"], freq) and r["residual_role"] == "real" and r["n_components_requested"] == n_components]
                pseudo_fits = [
                    r
                    for r in fit_rows
                    if r["analysis_source"] == source
                    and r["family_set"] == family_set_name
                    and np.isclose(r["frequency_mhz"], freq)
                    and r["residual_role"] == "pseudo_control"
                    and r["n_components_requested"] == n_components
                ]
                if not real_fit:
                    continue
                vals = np.asarray([float(r.get("amplitude", np.nan)) for r in pseudo_fits], dtype=float)
                vals = vals[np.isfinite(vals)]
                real_amp = float(real_fit[0].get("amplitude", np.nan))
                diag_rows.append(
                    {
                        "analysis_source": source,
                        "family_set": family_set_name,
                        "frequency_mhz": float(freq),
                        "frequency_band": int(real_fit[0]["frequency_band"]),
                        "n_components_requested": int(n_components),
                        "n_components_used": int(real_fit[0]["n_components_used"]),
                        "n_control_curves": int(model.n_controls),
                        "n_pseudo_controls": int(vals.size),
                        "real_residual_amplitude": real_amp,
                        "real_residual_uncertainty": float(real_fit[0].get("uncertainty", np.nan)),
                        "real_residual_fit_snr": float(real_fit[0].get("stack_fit_snr", np.nan)),
                        "real_delta_bic": float(real_fit[0].get("delta_bic", np.nan)),
                        "real_best_transition_duration_s": float(real_fit[0].get("best_transition_duration_s", np.nan)),
                        "real_sideband_rms": float(real_fit[0].get("sideband_rms", np.nan)),
                        "pseudo_median_amplitude": float(np.nanmedian(vals)) if vals.size else np.nan,
                        "pseudo_q25_amplitude": float(np.nanquantile(vals, 0.25)) if vals.size else np.nan,
                        "pseudo_q75_amplitude": float(np.nanquantile(vals, 0.75)) if vals.size else np.nan,
                        "pseudo_abs_q75_amplitude": float(np.nanquantile(np.abs(vals), 0.75)) if vals.size else np.nan,
                        "empirical_p_amp_ge_real": float((1 + np.count_nonzero(vals >= real_amp)) / (1 + vals.size)) if vals.size and np.isfinite(real_amp) else np.nan,
                        "empirical_p_abs_amp_ge_real": float((1 + np.count_nonzero(np.abs(vals) >= abs(real_amp))) / (1 + vals.size)) if vals.size and np.isfinite(real_amp) else np.nan,
                    }
                )

    return pd.DataFrame(residual_rows), pd.DataFrame(fit_rows), pd.DataFrame(diag_rows)


def _pseudo_envelope(residuals: pd.DataFrame, source: str, family_set: str, n_components: int) -> pd.DataFrame:
    pseudo = residuals[
        residuals["analysis_source"].eq(source)
        & residuals["family_set"].eq(family_set)
        & residuals["n_components_requested"].eq(int(n_components))
        & residuals["residual_role"].eq("pseudo_control")
    ].copy()
    if pseudo.empty:
        return pd.DataFrame()
    rows = []
    by = ["analysis_source", "family_set", "frequency_mhz", "event_type", "t_bin_sec"]
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


def plot_residual_grid(residuals: pd.DataFrame, family_set: str, n_components: int, out_dir: Path) -> list[Path]:
    paths = []
    real = residuals[
        residuals["family_set"].eq(family_set)
        & residuals["n_components_requested"].eq(int(n_components))
        & residuals["residual_role"].eq("real")
    ].copy()
    for source in sorted(real["analysis_source"].dropna().unique()):
        src = real[real["analysis_source"].eq(source)]
        freqs = sorted(src["frequency_mhz"].dropna().astype(float).unique())
        fig, axes = plt.subplots(len(freqs), 2, figsize=(13.5, max(10, 1.45 * len(freqs))), sharex=True, sharey=False)
        if len(freqs) == 1:
            axes = np.asarray([axes])
        for i, freq in enumerate(freqs):
            for j, event_type in enumerate(["disappearance", "reappearance"]):
                ax = axes[i, j]
                sub = src[np.isclose(src["frequency_mhz"].astype(float), freq) & src["event_type"].eq(event_type)].sort_values("t_bin_sec")
                if not sub.empty:
                    x = sub["t_bin_sec"].to_numpy(dtype=float) / 60.0
                    ax.plot(x, sub["real_stack_z_power"], color="0.2", marker=".", ms=2.5, lw=1.0, label="real stack")
                    ax.plot(x, sub["manifold_background_z_power"], color="#d95f02", lw=1.2, label="control-manifold background")
                    ax.errorbar(
                        x,
                        sub["residual_z_power"],
                        yerr=sub["residual_uncertainty"],
                        color="#1f78b4",
                        ecolor="#1f78b4",
                        marker="o",
                        ms=2.4,
                        lw=1.2,
                        elinewidth=0.5,
                        capsize=1.0,
                        label="real - background",
                    )
                    side = sub["is_sideband_fit_bin"].astype(bool).to_numpy()
                    if np.any(side):
                        ax.scatter(x[side], sub["real_stack_z_power"].to_numpy(dtype=float)[side], facecolors="none", edgecolors="black", s=18, linewidths=0.6, label="sideband bins")
                ax.axvline(0.0, color="black", ls=":", lw=0.8)
                ax.axhline(0.0, color="0.7", lw=0.7)
                ax.set_title(f"{freq:.2f} MHz {event_type}", fontsize=8.5)
                if j == 0:
                    ax.set_ylabel("normalized lower-V power")
                if i == len(freqs) - 1:
                    ax.set_xlabel("minutes from predicted event")
                ax.grid(True, color="0.92", lw=0.5)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        fig.legend(by_label.values(), by_label.keys(), loc="upper center", ncol=min(4, len(by_label)), frameon=False)
        fig.suptitle(f"{SOURCE_LABEL.get(source, source)} control-manifold subtraction ({family_set}, k={n_components})", y=0.996)
        fig.tight_layout(rect=[0, 0, 1, 0.965])
        path = out_dir / f"{source}_{family_set}_k{n_components}_control_manifold_residual_grid.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_amplitude_spectra(summary: pd.DataFrame, family_set: str, out_dir: Path) -> list[Path]:
    paths = []
    sub = summary[summary["family_set"].eq(family_set)].copy()
    for source, src in sub.groupby("analysis_source", sort=True):
        fig, ax = plt.subplots(figsize=(9.5, 5.2))
        ax.axhline(0.0, color="0.7", lw=0.8)
        for k, grp in src.groupby("n_components_requested", sort=True):
            grp = grp.sort_values("frequency_mhz")
            x = grp["frequency_mhz"].to_numpy(dtype=float)
            y = grp["real_residual_amplitude"].to_numpy(dtype=float)
            err = grp["real_residual_uncertainty"].to_numpy(dtype=float)
            ax.errorbar(x, y, yerr=err, marker="o", lw=1.2, capsize=2.0, label=f"k={int(k)} real residual")
        k2 = src[src["n_components_requested"].eq(2)].sort_values("frequency_mhz")
        if not k2.empty:
            x = k2["frequency_mhz"].to_numpy(dtype=float)
            ax.fill_between(x, k2["pseudo_q25_amplitude"], k2["pseudo_q75_amplitude"], color="0.55", alpha=0.22, label="k=2 pseudo-control IQR")
            ax.plot(x, k2["pseudo_median_amplitude"], color="0.4", lw=1.0, label="k=2 pseudo-control median")
        ax.set_xscale("log")
        ax.set_xticks(list(FREQUENCY_MAP_MHZ.values()))
        ax.set_xticklabels([f"{v:g}" for v in FREQUENCY_MAP_MHZ.values()], rotation=45, ha="right")
        ax.set_xlabel("frequency (MHz)")
        ax.set_ylabel("positive-source residual template amplitude")
        ax.set_title(f"{SOURCE_LABEL.get(source, source)} control-manifold residual amplitude ({family_set})")
        ax.grid(True, color="0.9", lw=0.5)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        path = out_dir / f"{source}_{family_set}_control_manifold_amplitude_spectrum.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_top_fit_profiles(residuals: pd.DataFrame, fits: pd.DataFrame, summary: pd.DataFrame, family_set: str, n_components: int, out_dir: Path) -> list[Path]:
    paths = []
    main = summary[
        summary["family_set"].eq(family_set)
        & summary["n_components_requested"].eq(int(n_components))
    ].copy()
    for source, src in main.groupby("analysis_source", sort=True):
        ranked = src.copy()
        ranked["rank_score"] = ranked["real_residual_amplitude"].where(ranked["real_residual_amplitude"] > 0, -np.inf)
        ranked = ranked.sort_values(["rank_score", "real_delta_bic"], ascending=[False, False]).head(4)
        if ranked.empty:
            continue
        env = _pseudo_envelope(residuals, source, family_set, n_components)
        fig, axes = plt.subplots(len(ranked), 2, figsize=(13.5, max(3, 2.7 * len(ranked))), squeeze=False, sharex=True)
        for i, (_, row) in enumerate(ranked.iterrows()):
            freq = float(row["frequency_mhz"])
            fit_row = fits[
                fits["analysis_source"].eq(source)
                & fits["family_set"].eq(family_set)
                & fits["n_components_requested"].eq(int(n_components))
                & fits["residual_role"].eq("real")
                & np.isclose(fits["frequency_mhz"].astype(float), freq)
            ].iloc[0]
            prof = residuals[
                residuals["analysis_source"].eq(source)
                & residuals["family_set"].eq(family_set)
                & residuals["n_components_requested"].eq(int(n_components))
                & residuals["residual_role"].eq("real")
                & np.isclose(residuals["frequency_mhz"].astype(float), freq)
            ].copy()
            prof["model"] = _model_curve(prof, fit_row)
            for j, event_type in enumerate(["disappearance", "reappearance"]):
                ax = axes[i, j]
                sub = prof[prof["event_type"].eq(event_type)].sort_values("t_bin_sec")
                e = env[np.isclose(env["frequency_mhz"].astype(float), freq) & env["event_type"].eq(event_type)].sort_values("t_bin_sec")
                if not e.empty:
                    xe = e["t_bin_sec"].to_numpy(dtype=float) / 60.0
                    ax.fill_between(xe, e["pseudo_q25_residual"], e["pseudo_q75_residual"], color="0.65", alpha=0.28, linewidth=0, label="pseudo-control residual IQR")
                    ax.plot(xe, e["pseudo_median_residual"], color="0.45", lw=0.9, label="pseudo-control median")
                if not sub.empty:
                    x = sub["t_bin_sec"].to_numpy(dtype=float) / 60.0
                    ax.errorbar(x, sub["residual_z_power"], yerr=sub["residual_uncertainty"], color="#1f78b4", ecolor="#1f78b4", marker="o", ms=2.4, lw=1.1, elinewidth=0.55, capsize=1.0, label="real residual")
                    ax.plot(x, sub["model"], color="#e41a1c", lw=1.7, label="template fit")
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
        fig.suptitle(f"{SOURCE_LABEL.get(source, source)} residual fits after control-manifold subtraction ({family_set}, k={n_components})", y=0.995)
        fig.tight_layout(rect=[0, 0, 0.91, 0.96])
        path = out_dir / f"{source}_{family_set}_k{n_components}_top_control_manifold_residual_fits.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def write_report(out_dir: Path, summary: pd.DataFrame, plot_paths: list[Path], sideband_s: float) -> None:
    cols = [
        "analysis_source",
        "family_set",
        "frequency_mhz",
        "n_components_requested",
        "n_control_curves",
        "real_residual_amplitude",
        "real_residual_uncertainty",
        "real_residual_fit_snr",
        "real_delta_bic",
        "real_best_transition_duration_s",
        "real_sideband_rms",
        "n_pseudo_controls",
        "pseudo_median_amplitude",
        "pseudo_q25_amplitude",
        "pseudo_q75_amplitude",
        "empirical_p_amp_ge_real",
        "empirical_p_abs_amp_ge_real",
    ]
    lines = [
        "# Lower-V Control-Manifold Background Subtraction",
        "",
        "This run is data-driven and uses no diffuse-sky or beam forward model.",
        "",
        "For each source/frequency, control stacks are converted into a matrix of profile shapes. A low-rank SVD/PCA basis is learned from controls only. For the real source stack, the background coefficients are fit only using sideband bins away from the predicted event:",
        "",
        f"    sideband bins: |t| >= {float(sideband_s):.0f} s",
        "",
        "The fitted control-manifold background is then extrapolated through the event center and subtracted. Pseudo-control residuals are generated with leave-one-control-out training.",
        "",
        "Interpretation rule: a useful extraction should produce a source-like residual that is stable with component count and outside the pseudo-control residual distribution. A high fit SNR alone is not sufficient.",
        "",
        "## Summary",
        "",
        summary[cols].sort_values(["analysis_source", "family_set", "n_components_requested", "frequency_mhz"]).to_string(index=False),
        "",
        "## Plots",
        "",
    ]
    lines.extend(f"- `{p.name}`" for p in plot_paths)
    (out_dir / "control_manifold_background_subtraction_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack-root", default=str(STACKFIRST_ROOT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--family-sets", default="time_off,all_controls")
    parser.add_argument("--components", default="1,2,3")
    parser.add_argument("--main-components", type=int, default=2)
    parser.add_argument("--sideband-s", type=float, default=600.0)
    parser.add_argument("--timing-offsets-s", default="0")
    parser.add_argument("--transition-durations-s", default="0,120,300,600,900")
    args = parser.parse_args()

    stack_root = Path(args.stack_root)
    out_dir = ensure_dir(Path(args.out_dir))
    family_sets = [x.strip() for x in str(args.family_sets).split(",") if x.strip()]
    unknown = sorted(set(family_sets) - set(FAMILY_SETS))
    if unknown:
        raise SystemExit(f"Unknown family set(s): {', '.join(unknown)}")
    components = [int(x.strip()) for x in str(args.components).split(",") if x.strip()]
    timing_offsets = [float(x.strip()) for x in str(args.timing_offsets_s).split(",") if x.strip()]
    transition_durations = [float(x.strip()) for x in str(args.transition_durations_s).split(",") if x.strip()]

    write_json(
        out_dir / "run_config.json",
        {
            "stack_root": str(stack_root),
            "family_sets": family_sets,
            "components": components,
            "main_components": int(args.main_components),
            "sideband_s": float(args.sideband_s),
            "timing_offsets_s": timing_offsets,
            "transition_durations_s": transition_durations,
            "software_versions": software_versions(),
        },
    )

    stacks = _read(stack_root / "lower_v_stackfirst_control_stacks.csv")
    all_residuals = []
    all_fits = []
    all_summary = []
    for family_set in family_sets:
        print(f"Building control-manifold residuals for {family_set}...", flush=True)
        residuals, fits, summary = build_manifold_residuals(
            stacks,
            family_set_name=family_set,
            n_components_list=components,
            sideband_s=float(args.sideband_s),
            timing_offsets=timing_offsets,
            transition_durations=transition_durations,
        )
        all_residuals.append(residuals)
        all_fits.append(fits)
        all_summary.append(summary)
    residual_table = pd.concat(all_residuals, ignore_index=True) if all_residuals else pd.DataFrame()
    fit_table = pd.concat(all_fits, ignore_index=True) if all_fits else pd.DataFrame()
    summary_table = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
    residual_table.to_csv(out_dir / "control_manifold_residual_profiles.csv", index=False)
    fit_table.to_csv(out_dir / "control_manifold_fit_by_group.csv", index=False)
    summary_table.to_csv(out_dir / "control_manifold_fit_summary.csv", index=False)

    print("Writing control-manifold plots...", flush=True)
    paths: list[Path] = []
    for family_set in family_sets:
        paths.extend(plot_residual_grid(residual_table, family_set, int(args.main_components), out_dir))
        paths.extend(plot_amplitude_spectra(summary_table, family_set, out_dir))
        paths.extend(plot_top_fit_profiles(residual_table, fit_table, summary_table, family_set, int(args.main_components), out_dir))
    write_report(out_dir, summary_table, paths, sideband_s=float(args.sideband_s))
    print(out_dir / "control_manifold_background_subtraction_report.md")


if __name__ == "__main__":
    main()
