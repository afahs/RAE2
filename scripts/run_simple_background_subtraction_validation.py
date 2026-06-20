#!/usr/bin/env python
"""Simple lower-V background subtraction and transfer validation.

This is a deliberately conservative alternative to the SVD/control-manifold
background subtraction. It avoids learned high-rank background modes and tests
each simple subtraction method with injection recovery before any source result
is interpreted.
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
from rylevonberg.stackfit import StackedStepFitConfig, fit_stacked_step, stacked_event_template  # noqa: E402
from rylevonberg.util import ensure_dir, robust_sigma, software_versions, write_json  # noqa: E402


DEFAULT_OUT = ROOT / "outputs/lower_v_simple_background_validation_v1"
STACK_ROOTS = {
    "earth": ROOT / "outputs/lower_v_control_manifold_earth_positive_control_v1",
    "sun": ROOT / "outputs/lower_v_stackfirst_detection_sun_fornax_v1",
    "fornax_a": ROOT / "outputs/lower_v_stackfirst_detection_sun_fornax_v1",
}
SOURCE_LABEL = {"earth": "Earth", "sun": "Sun", "fornax_a": "Fornax-A"}
EVENT_ORDER = {"disappearance": 0, "reappearance": 1}
METHOD_LABEL = {
    "sideband_center": "real sideband centering",
    "control_median": "direct control median",
    "control_offset": "control median + sideband offset",
    "control_scale_offset": "control median + sideband scale/offset",
}
KEYS = ["analysis_source", "frequency_band", "frequency_mhz", "event_type", "t_bin_sec"]


def _read(path: Path, **kwargs) -> pd.DataFrame:
    return read_table(path, low_memory=False, **kwargs)


def _robust_scale(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size <= 1:
        return np.nan
    scale = robust_sigma(vals - np.nanmedian(vals))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.nanstd(vals, ddof=1))
    return float(scale) if np.isfinite(scale) and scale > 0 else np.nan


def _load_stacks(sources: list[str]) -> pd.DataFrame:
    frames = []
    for source in sources:
        root = STACK_ROOTS[source]
        path = root / "lower_v_stackfirst_control_stacks.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        df = _read(path)
        frames.append(df[df["analysis_source"].astype(str).eq(source)].copy())
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out["analysis_source"] = out["analysis_source"].astype(str)
    out["control_family"] = out["control_family"].astype(str)
    out["control_id"] = out["control_id"].astype(str)
    out["event_type"] = out["event_type"].astype(str)
    return out


def _control_summary(pool: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if pool.empty:
        return pd.DataFrame()
    for keys, grp in pool.groupby(KEYS, sort=True, dropna=False):
        vals = pd.to_numeric(grp["median_z_power"], errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue
        rows.append(
            {
                **dict(zip(KEYS, keys)),
                "control_median_z_power": float(np.nanmedian(vals)),
                "control_q25_z_power": float(np.nanquantile(vals, 0.25)),
                "control_q75_z_power": float(np.nanquantile(vals, 0.75)),
                "control_robust_sigma": _robust_scale(vals),
                "n_background_controls": int(grp[["control_family", "control_id"]].drop_duplicates().shape[0]),
            }
        )
    return pd.DataFrame(rows)


def _fit_scale_offset(y: np.ndarray, c: np.ndarray) -> tuple[float, float, str]:
    mask = np.isfinite(y) & np.isfinite(c)
    if np.count_nonzero(mask) < 4:
        return 0.0, 1.0, "fallback_too_few_sideband_bins"
    yy = y[mask]
    cc = c[mask]
    if np.nanstd(cc) < 1e-6:
        return float(np.nanmedian(yy - cc)), 1.0, "fallback_control_flat"
    x = np.column_stack([np.ones_like(cc), cc])
    beta, *_ = np.linalg.lstsq(x, yy, rcond=None)
    intercept = float(beta[0])
    slope = float(np.clip(beta[1], 0.0, 2.0))
    note = "least_squares_sideband"
    if not np.isclose(slope, float(beta[1])):
        note = "least_squares_sideband_slope_clipped"
    return intercept, slope, note


def _target_residuals(
    target: pd.DataFrame,
    control_pool: pd.DataFrame,
    method: str,
    sideband_s: float,
    residual_role: str,
    pseudo_control_id: str,
) -> pd.DataFrame:
    if target.empty:
        return pd.DataFrame()
    target = target.copy()
    target["event_sort"] = target["event_type"].map(EVENT_ORDER).fillna(99).astype(int)
    target = target.sort_values(["analysis_source", "frequency_mhz", "event_sort", "t_bin_sec"])
    rows = []
    control_bg = _control_summary(control_pool) if method != "sideband_center" else pd.DataFrame()
    for (source, band, freq, event_type), grp in target.groupby(
        ["analysis_source", "frequency_band", "frequency_mhz", "event_type"],
        sort=True,
        dropna=False,
    ):
        g = grp.copy().sort_values("t_bin_sec")
        y = pd.to_numeric(g["median_z_power"], errors="coerce").to_numpy(dtype=float)
        t = pd.to_numeric(g["t_bin_sec"], errors="coerce").to_numpy(dtype=float)
        side = np.abs(t) >= float(sideband_s)
        if method == "sideband_center":
            bg = np.full_like(y, np.nanmedian(y[side & np.isfinite(y)]) if np.any(side & np.isfinite(y)) else np.nan)
            bg_sigma = np.zeros_like(bg)
            bg_q25 = bg.copy()
            bg_q75 = bg.copy()
            n_bg = np.zeros_like(bg, dtype=int)
            intercept = float(bg[0]) if bg.size and np.isfinite(bg[0]) else np.nan
            slope = 0.0
            fit_note = "real_sideband_constant"
        else:
            template = control_bg[
                control_bg["analysis_source"].astype(str).eq(str(source))
                & control_bg["frequency_band"].astype(int).eq(int(band))
                & np.isclose(control_bg["frequency_mhz"].astype(float), float(freq))
                & control_bg["event_type"].astype(str).eq(str(event_type))
            ].copy()
            if template.empty:
                continue
            merged = g.merge(
                template[
                    KEYS
                    + [
                        "control_median_z_power",
                        "control_q25_z_power",
                        "control_q75_z_power",
                        "control_robust_sigma",
                        "n_background_controls",
                    ]
                ],
                on=KEYS,
                how="inner",
            ).sort_values("t_bin_sec")
            if merged.empty:
                continue
            g = merged
            y = pd.to_numeric(g["median_z_power"], errors="coerce").to_numpy(dtype=float)
            t = pd.to_numeric(g["t_bin_sec"], errors="coerce").to_numpy(dtype=float)
            c = pd.to_numeric(g["control_median_z_power"], errors="coerce").to_numpy(dtype=float)
            side = np.abs(t) >= float(sideband_s)
            if method == "control_median":
                intercept, slope, fit_note = 0.0, 1.0, "direct_control_median"
            elif method == "control_offset":
                resid = y - c
                intercept = float(np.nanmedian(resid[side & np.isfinite(resid)])) if np.any(side & np.isfinite(resid)) else 0.0
                slope, fit_note = 1.0, "sideband_median_offset"
            elif method == "control_scale_offset":
                intercept, slope, fit_note = _fit_scale_offset(y[side], c[side])
            else:
                raise ValueError(f"Unknown method: {method}")
            bg = intercept + slope * c
            control_sigma = pd.to_numeric(g["control_robust_sigma"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            bg_sigma = np.abs(slope) * control_sigma
            bg_q25 = intercept + slope * pd.to_numeric(g["control_q25_z_power"], errors="coerce").to_numpy(dtype=float)
            bg_q75 = intercept + slope * pd.to_numeric(g["control_q75_z_power"], errors="coerce").to_numpy(dtype=float)
            n_bg = pd.to_numeric(g["n_background_controls"], errors="coerce").fillna(0).to_numpy(dtype=int)
        sem = pd.to_numeric(g["robust_sem_z_power"], errors="coerce").to_numpy(dtype=float)
        fallback = np.nanmedian(sem[np.isfinite(sem) & (sem > 0)])
        if not np.isfinite(fallback) or fallback <= 0:
            fallback = 1.0
        sem = np.where(np.isfinite(sem) & (sem > 0), sem, fallback)
        residual_sigma = np.sqrt(np.square(sem) + np.square(np.where(np.isfinite(bg_sigma), bg_sigma, 0.0)))
        out = g.copy()
        out["background_method"] = method
        out["residual_role"] = residual_role
        out["pseudo_control_id"] = pseudo_control_id
        out["background_z_power"] = bg
        out["background_q25_z_power"] = bg_q25
        out["background_q75_z_power"] = bg_q75
        out["background_fit_intercept"] = intercept
        out["background_fit_slope"] = slope
        out["background_fit_note"] = fit_note
        out["n_background_controls"] = n_bg
        out["residual_z_power"] = y - bg
        out["residual_uncertainty"] = residual_sigma
        rows.append(out)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_residuals(
    stacks: pd.DataFrame,
    methods: list[str],
    control_families: list[str],
    sideband_s: float,
    include_pseudo: bool = True,
) -> pd.DataFrame:
    rows = []
    for source, src in stacks.groupby("analysis_source", sort=True):
        real = src[src["control_family"].eq("real")].copy()
        controls = src[src["control_family"].isin(control_families)].copy()
        for method in methods:
            real_res = _target_residuals(real, controls, method, sideband_s, "real", "real")
            if not real_res.empty:
                rows.append(real_res)
            if not include_pseudo:
                continue
            for (family, control_id), control in controls.groupby(["control_family", "control_id"], sort=True):
                other = controls[
                    ~(controls["control_family"].eq(str(family)) & controls["control_id"].eq(str(control_id)))
                ].copy()
                if method != "sideband_center" and other.empty:
                    continue
                pseudo = _target_residuals(
                    control,
                    other,
                    method,
                    sideband_s,
                    "pseudo_control",
                    f"{family}:{control_id}",
                )
                if not pseudo.empty:
                    rows.append(pseudo)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


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
            rows.append({**meta, "stack_group": str(stack_group), **fit})
    return pd.DataFrame(rows)


def _fixed_template_projection(profile: pd.DataFrame, tau_s: float) -> float:
    t = pd.to_numeric(profile["t_bin_sec"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(profile["residual_z_power"], errors="coerce").to_numpy(dtype=float)
    template = stacked_event_template(
        t,
        profile["event_type"].astype(str).to_numpy(),
        timing_offset_sec=0.0,
        transition_duration_sec=float(tau_s),
    )
    mask = np.isfinite(t) & np.isfinite(y) & np.isfinite(template)
    if np.count_nonzero(mask) < 8:
        return np.nan
    x = np.column_stack([np.ones(np.count_nonzero(mask)), template[mask]])
    beta, *_ = np.linalg.lstsq(x, y[mask], rcond=None)
    return float(beta[-1])


def summarize_fits(fits: pd.DataFrame) -> pd.DataFrame:
    real = fits[fits["residual_role"].eq("real") & fits["stack_group"].eq("combined")].copy()
    pseudo = fits[fits["residual_role"].eq("pseudo_control") & fits["stack_group"].eq("combined")].copy()
    rows = []
    for _, row in real.iterrows():
        same = pseudo[
            pseudo["analysis_source"].eq(row["analysis_source"])
            & pseudo["background_method"].eq(row["background_method"])
            & pseudo["frequency_band"].eq(row["frequency_band"])
        ].copy()
        vals = pd.to_numeric(same["amplitude"], errors="coerce").dropna().to_numpy(dtype=float)
        abs_vals = np.abs(vals)
        amp = float(row["amplitude"])
        rows.append(
            {
                "analysis_source": row["analysis_source"],
                "background_method": row["background_method"],
                "frequency_band": int(row["frequency_band"]),
                "frequency_mhz": float(row["frequency_mhz"]),
                "real_template_amplitude": amp,
                "real_template_uncertainty": float(row["uncertainty"]),
                "real_delta_bic": float(row["delta_bic"]),
                "real_best_transition_duration_s": float(row.get("best_transition_duration_s", np.nan)),
                "n_pseudo_controls": int(vals.size),
                "pseudo_median_amplitude": float(np.nanmedian(vals)) if vals.size else np.nan,
                "pseudo_abs_q75_amplitude": float(np.nanquantile(abs_vals, 0.75)) if vals.size else np.nan,
                "empirical_p_abs_amp_ge_real": float((1 + np.count_nonzero(abs_vals >= abs(amp))) / (1 + vals.size))
                if vals.size
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _inject_real(stacks: pd.DataFrame, source: str, amplitude: float, tau_s: float) -> pd.DataFrame:
    out = stacks.copy()
    mask = out["analysis_source"].astype(str).eq(source) & out["control_family"].astype(str).eq("real")
    if not mask.any():
        return out
    template = stacked_event_template(
        out.loc[mask, "t_bin_sec"].to_numpy(dtype=float),
        out.loc[mask, "event_type"].astype(str).to_numpy(),
        timing_offset_sec=0.0,
        transition_duration_sec=float(tau_s),
    )
    out.loc[mask, "median_z_power"] = pd.to_numeric(out.loc[mask, "median_z_power"], errors="coerce").to_numpy(dtype=float) + float(amplitude) * template
    if "mean_z_power" in out.columns:
        out.loc[mask, "mean_z_power"] = pd.to_numeric(out.loc[mask, "mean_z_power"], errors="coerce").to_numpy(dtype=float) + float(amplitude) * template
    return out


def injection_recovery(
    stacks: pd.DataFrame,
    methods: list[str],
    control_families: list[str],
    sideband_s: float,
    amplitudes: list[float],
    taus: list[float],
) -> pd.DataFrame:
    base = build_residuals(stacks, methods, control_families, sideband_s, include_pseudo=False)
    base_rows = []
    by = ["analysis_source", "background_method", "frequency_band", "frequency_mhz"]
    for keys, grp in base.groupby(by, sort=True, dropna=False):
        for tau in taus:
            base_rows.append({**dict(zip(by, keys)), "injection_tau_s": float(tau), "baseline_projection": _fixed_template_projection(grp, float(tau))})
    baseline = pd.DataFrame(base_rows)
    rows = []
    for source in sorted(stacks["analysis_source"].dropna().astype(str).unique()):
        source_stacks = stacks[stacks["analysis_source"].astype(str).eq(source)].copy()
        for tau in taus:
            for amp in amplitudes:
                injected = _inject_real(source_stacks, source, float(amp), float(tau))
                residuals = build_residuals(injected, methods, control_families, sideband_s, include_pseudo=False)
                for keys, grp in residuals.groupby(by, sort=True, dropna=False):
                    meta = dict(zip(by, keys))
                    proj = _fixed_template_projection(grp, float(tau))
                    b = baseline[
                        baseline["analysis_source"].eq(meta["analysis_source"])
                        & baseline["background_method"].eq(meta["background_method"])
                        & baseline["frequency_band"].astype(int).eq(int(meta["frequency_band"]))
                        & np.isclose(baseline["frequency_mhz"].astype(float), float(meta["frequency_mhz"]))
                        & np.isclose(baseline["injection_tau_s"].astype(float), float(tau))
                    ]
                    base_proj = float(b["baseline_projection"].iloc[0]) if not b.empty else np.nan
                    rec = (proj - base_proj) / float(amp) if np.isfinite(proj) and np.isfinite(base_proj) and amp != 0 else np.nan
                    rows.append(
                        {
                            **meta,
                            "injection_tau_s": float(tau),
                            "injection_amplitude": float(amp),
                            "baseline_projection": base_proj,
                            "injected_projection": proj,
                            "fixed_recovery_fraction": rec,
                        }
                    )
    return pd.DataFrame(rows)


def summarize_transfer(recovery: pd.DataFrame, main_tau: float) -> pd.DataFrame:
    rows = []
    by = ["analysis_source", "background_method", "frequency_band", "frequency_mhz", "injection_tau_s"]
    for keys, grp in recovery.groupby(by, sort=True, dropna=False):
        vals = pd.to_numeric(grp["fixed_recovery_fraction"], errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue
        med = float(np.nanmedian(vals))
        if not np.isfinite(med) or abs(med) > 3:
            flag = "unstable_or_nonphysical"
        elif med <= 0:
            flag = "negative_transfer"
        elif med < 0.1:
            flag = "low_recovery"
        elif med < 0.5:
            flag = "attenuated"
        elif med <= 1.5:
            flag = "usable"
        else:
            flag = "amplified"
        rows.append(
            {
                **dict(zip(by, keys)),
                "median_recovery_fraction": med,
                "q25_recovery_fraction": float(np.nanquantile(vals, 0.25)),
                "q75_recovery_fraction": float(np.nanquantile(vals, 0.75)),
                "transfer_stability_flag": flag,
                "is_main_tau": bool(np.isclose(float(keys[-1]), float(main_tau))),
            }
        )
    return pd.DataFrame(rows)


def plot_residual_grids(residuals: pd.DataFrame, stacks: pd.DataFrame, methods: list[str], out_dir: Path) -> list[Path]:
    paths = []
    real_residuals = residuals[residuals["residual_role"].eq("real")].copy()
    for source in sorted(real_residuals["analysis_source"].dropna().unique()):
        for method in methods:
            src_method = real_residuals[real_residuals["analysis_source"].eq(source) & real_residuals["background_method"].eq(method)].copy()
            if src_method.empty:
                continue
            freqs = sorted(src_method["frequency_mhz"].dropna().unique())
            fig, axes = plt.subplots(len(freqs), 2, figsize=(13.5, max(10, 1.45 * len(freqs))), sharex=True)
            if len(freqs) == 1:
                axes = np.asarray([axes])
            for i, freq in enumerate(freqs):
                for j, event_type in enumerate(["disappearance", "reappearance"]):
                    ax = axes[i, j]
                    raw = stacks[
                        stacks["analysis_source"].eq(source)
                        & stacks["control_family"].eq("real")
                        & np.isclose(stacks["frequency_mhz"].astype(float), float(freq))
                        & stacks["event_type"].eq(event_type)
                    ].sort_values("t_bin_sec")
                    res = src_method[
                        np.isclose(src_method["frequency_mhz"].astype(float), float(freq))
                        & src_method["event_type"].eq(event_type)
                    ].sort_values("t_bin_sec")
                    if not raw.empty:
                        ax.plot(raw["t_bin_sec"] / 60.0, raw["median_z_power"], color="0.25", marker=".", ms=2.2, lw=0.9, label="real stack")
                    if not res.empty:
                        x = res["t_bin_sec"].to_numpy(dtype=float) / 60.0
                        ax.plot(x, res["background_z_power"], color="#d95f02", lw=1.0, label="simple background")
                        ax.fill_between(x, res["background_q25_z_power"], res["background_q75_z_power"], color="#d95f02", alpha=0.12, linewidth=0)
                        ax.errorbar(
                            x,
                            res["residual_z_power"],
                            yerr=res["residual_uncertainty"],
                            color="#1f78b4",
                            ecolor="#1f78b4",
                            marker="o",
                            ms=2.2,
                            lw=1.15,
                            elinewidth=0.45,
                            capsize=0.8,
                            label="real - background",
                        )
                    ax.axvline(0, color="black", ls=":", lw=0.8)
                    ax.axhline(0, color="0.75", lw=0.7)
                    ax.grid(True, color="0.92", lw=0.5)
                    ax.set_title(f"{float(freq):.2f} MHz {event_type}", fontsize=8.5)
                    if j == 0 and i == len(freqs) // 2:
                        ax.set_ylabel("normalized lower-V power")
                    if i == len(freqs) - 1:
                        ax.set_xlabel("minutes from predicted event")
            handles, labels = axes[0, 0].get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            fig.legend(
                by_label.values(),
                by_label.keys(),
                loc="lower center",
                bbox_to_anchor=(0.5, 0.006),
                ncol=min(3, len(by_label)),
                frameon=False,
            )
            fig.suptitle(f"{SOURCE_LABEL.get(source, source)} lower-V simple background: {METHOD_LABEL[method]}", y=0.992)
            fig.tight_layout(rect=[0, 0.035, 1, 0.965])
            path = out_dir / f"{source}_{method}_simple_background_grid.png"
            fig.savefig(path, dpi=180)
            plt.close(fig)
            paths.append(path)
    return paths


def plot_transfer(transfer: pd.DataFrame, out_dir: Path, main_tau: float) -> Path:
    main = transfer[np.isclose(transfer["injection_tau_s"].astype(float), float(main_tau))].copy()
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), sharey=True)
    colors = {
        "sideband_center": "#333333",
        "control_median": "#d95f02",
        "control_offset": "#1f78b4",
        "control_scale_offset": "#7570b3",
    }
    for ax, source in zip(axes, ["earth", "sun", "fornax_a"]):
        sub = main[main["analysis_source"].eq(source)].sort_values("frequency_mhz")
        ax.axhline(1, color="0.55", lw=0.8, ls="--")
        ax.axhline(0, color="0.75", lw=0.8)
        for method in colors:
            m = sub[sub["background_method"].eq(method)].copy()
            if m.empty:
                continue
            y = pd.to_numeric(m["median_recovery_fraction"], errors="coerce").to_numpy(dtype=float)
            y_plot = np.where(np.isfinite(y) & (np.abs(y) <= 3), y, np.nan)
            ax.plot(m["frequency_mhz"], y_plot, marker="o", lw=1.3, color=colors[method], label=METHOD_LABEL[method])
            bad = m[~np.isfinite(y_plot)]
            if not bad.empty:
                ax.scatter(bad["frequency_mhz"], np.full(len(bad), -0.18), marker="x", color=colors[method], s=45)
        ax.set_xscale("log")
        ax.set_xticks(list(FREQUENCY_MAP_MHZ.values()))
        ax.set_xticklabels([f"{v:g}" for v in FREQUENCY_MAP_MHZ.values()], rotation=45, ha="right")
        ax.set_ylim(-0.25, 2.2)
        ax.set_title(SOURCE_LABEL.get(source, source))
        ax.set_xlabel("frequency (MHz)")
        ax.grid(True, color="0.9", lw=0.5)
    axes[0].set_ylabel("recovered / injected amplitude")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle(f"Simple-background transfer function for injected lower-V occultation templates (tau={main_tau:g} s)")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    path = out_dir / f"simple_background_transfer_tau{int(main_tau)}s.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report(
    out_dir: Path,
    fit_summary: pd.DataFrame,
    transfer: pd.DataFrame,
    plot_paths: list[Path],
    main_tau: float,
) -> None:
    main_transfer = transfer[np.isclose(transfer["injection_tau_s"].astype(float), float(main_tau))].copy()
    transfer_counts = (
        main_transfer.groupby(["analysis_source", "background_method", "transfer_stability_flag"])
        .size()
        .rename("n_channels")
        .reset_index()
        .sort_values(["analysis_source", "background_method", "transfer_stability_flag"])
    )
    fit_cols = [
        "analysis_source",
        "background_method",
        "frequency_mhz",
        "real_template_amplitude",
        "real_template_uncertainty",
        "real_delta_bic",
        "real_best_transition_duration_s",
        "n_pseudo_controls",
        "pseudo_abs_q75_amplitude",
        "empirical_p_abs_amp_ge_real",
    ]
    transfer_cols = [
        "analysis_source",
        "background_method",
        "frequency_mhz",
        "median_recovery_fraction",
        "transfer_stability_flag",
    ]
    lines = [
        "# Simple Lower-V Background Subtraction Validation",
        "",
        "This run compares simple empirical background-subtraction choices on the lower-V stack-first profiles. It is a method-validation run, not a source-detection claim.",
        "",
        "Methods:",
        "",
        "- `sideband_center`: subtract only a constant measured from the real sidebands. This is the least aggressive method.",
        "- `control_median`: subtract the bin-by-bin median control profile directly. This is simple but aggressive.",
        "- `control_offset`: subtract the control median after fitting one constant offset on sideband bins only.",
        "- `control_scale_offset`: subtract the control median after fitting one scale and one offset on sideband bins only.",
        "",
        "The transfer test injects a known positive-source occultation template into the real stack only, reruns the same background subtraction, and measures recovered / injected amplitude. A method is not trusted for weak-source claims in channels with negative, near-zero, or unstable transfer.",
        "",
        f"Main transfer duration shown: tau={float(main_tau):.0f} s",
        "",
        "## Transfer Stability Counts",
        "",
        transfer_counts.to_string(index=False),
        "",
        "## Main Transfer Table",
        "",
        main_transfer[transfer_cols].sort_values(["analysis_source", "background_method", "frequency_mhz"]).to_string(index=False),
        "",
        "## Residual Fit Summary",
        "",
        fit_summary[fit_cols].sort_values(["analysis_source", "background_method", "frequency_mhz"]).to_string(index=False),
        "",
        "## Plots",
        "",
    ]
    lines.extend(f"- `{p.name}`" for p in plot_paths)
    lines += [
        "",
        "## Practical Takeaway",
        "",
        "Prefer the simplest method that leaves the target residual visible while preserving injected templates. If a method weakens Earth or erases injected Fornax/Sun-like templates, it should not be used for weak-source non-detection claims.",
        "",
        "Recommended next extraction options:",
        "",
        "1. Use `sideband_center` or `control_offset` as the first-pass background treatment, with injection-recovery gating by frequency.",
        "2. Treat neighboring frequencies as positive coherence evidence, not as controls.",
        "3. Use off-source and shifted-time controls only for empirical p-values and transfer checks, not for shape-selected event cuts.",
        "4. Build geometry/quality subsets using independent variables only, then pre-register those cuts before looking at source contrast.",
        "5. Fit stacks first with fixed-duration templates and report raw, centered, and background-subtracted profiles side by side.",
        "6. If a Bayesian model is revisited, keep the background as per-event offsets or control-template nuisance terms with strong shrinkage, not free trend lines.",
    ]
    (out_dir / "simple_background_subtraction_validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--sources", default="earth,sun,fornax_a")
    parser.add_argument("--methods", default="sideband_center,control_median,control_offset,control_scale_offset")
    parser.add_argument("--control-families", default="time_shift,offsource,randomized_time")
    parser.add_argument("--sideband-s", type=float, default=600.0)
    parser.add_argument("--timing-offsets-s", default="0")
    parser.add_argument("--transition-durations-s", default="0,300,900")
    parser.add_argument("--injection-amplitudes", default="0.1,0.2,0.5")
    parser.add_argument("--injection-taus-s", default="0,300,900")
    parser.add_argument("--main-tau-s", type=float, default=300.0)
    args = parser.parse_args()

    out_dir = ensure_dir(Path(args.out_dir))
    sources = [x.strip() for x in str(args.sources).split(",") if x.strip()]
    methods = [x.strip() for x in str(args.methods).split(",") if x.strip()]
    control_families = [x.strip() for x in str(args.control_families).split(",") if x.strip()]
    timing_offsets = [float(x.strip()) for x in str(args.timing_offsets_s).split(",") if x.strip()]
    transition_durations = [float(x.strip()) for x in str(args.transition_durations_s).split(",") if x.strip()]
    amplitudes = [float(x.strip()) for x in str(args.injection_amplitudes).split(",") if x.strip()]
    injection_taus = [float(x.strip()) for x in str(args.injection_taus_s).split(",") if x.strip()]
    unknown = sorted(set(methods) - set(METHOD_LABEL))
    if unknown:
        raise SystemExit(f"Unknown methods: {', '.join(unknown)}")
    write_json(
        out_dir / "run_config.json",
        {
            "sources": sources,
            "methods": methods,
            "control_families": control_families,
            "sideband_s": float(args.sideband_s),
            "timing_offsets_s": timing_offsets,
            "transition_durations_s": transition_durations,
            "injection_amplitudes": amplitudes,
            "injection_taus_s": injection_taus,
            "main_tau_s": float(args.main_tau_s),
            "stack_roots": {k: str(v) for k, v in STACK_ROOTS.items()},
            "software_versions": software_versions(),
        },
    )
    print("Loading lower-V stack-first control tables...", flush=True)
    stacks = _load_stacks(sources)
    stacks.to_csv(out_dir / "input_lower_v_stackfirst_control_stacks.csv", index=False)
    print("Building simple residual profiles...", flush=True)
    residuals = build_residuals(stacks, methods, control_families, float(args.sideband_s), include_pseudo=True)
    residuals.to_csv(out_dir / "simple_background_residual_profiles.csv", index=False)
    print("Fitting simple residual profiles...", flush=True)
    fits = fit_residuals(residuals, timing_offsets, transition_durations)
    fits.to_csv(out_dir / "simple_background_residual_fit_by_group.csv", index=False)
    fit_summary = summarize_fits(fits)
    fit_summary.to_csv(out_dir / "simple_background_residual_fit_summary.csv", index=False)
    print("Running simple-background injection recovery...", flush=True)
    recovery = injection_recovery(stacks, methods, control_families, float(args.sideband_s), amplitudes, injection_taus)
    recovery.to_csv(out_dir / "simple_background_injection_recovery_by_amplitude.csv", index=False)
    transfer = summarize_transfer(recovery, float(args.main_tau_s))
    transfer.to_csv(out_dir / "simple_background_transfer_summary.csv", index=False)
    print("Writing plots...", flush=True)
    plot_paths = plot_residual_grids(residuals, stacks, methods, out_dir)
    plot_paths.append(plot_transfer(transfer, out_dir, float(args.main_tau_s)))
    write_report(out_dir, fit_summary, transfer, plot_paths, float(args.main_tau_s))
    print(out_dir / "simple_background_subtraction_validation_report.md")


if __name__ == "__main__":
    main()
