#!/usr/bin/env python
"""Injection-recovery through lower-V control-manifold subtraction.

This measures the transfer function of the empirical background subtraction:
how much of a known source-like occultation template survives after the
sideband-trained control-manifold background is fit and subtracted.
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
from rylevonberg.util import ensure_dir, software_versions, write_json  # noqa: E402
from scripts.run_control_manifold_background_subtraction import (  # noqa: E402
    FAMILY_SETS,
    _align_to_model,
    _build_model,
    _control_uncertainty,
    _curve_matrix,
    _predict_background,
    _real_curve,
)


DEFAULT_OUT = ROOT / "outputs/lower_v_control_manifold_injection_recovery_v1"
SOURCE_LABEL = {"earth": "Earth", "sun": "Sun", "fornax_a": "Fornax-A"}
STACK_ROOTS = {
    "earth": ROOT / "outputs/lower_v_control_manifold_earth_positive_control_v1",
    "sun": ROOT / "outputs/lower_v_stackfirst_detection_sun_fornax_v1",
    "fornax_a": ROOT / "outputs/lower_v_stackfirst_detection_sun_fornax_v1",
}


def _read(path: Path, **kwargs) -> pd.DataFrame:
    return read_table(path, low_memory=False, **kwargs)


def _fit_profile(profile: pd.DataFrame, timing_offsets: list[float], transition_durations: list[float]) -> dict[str, float]:
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


def _fixed_template_projection(profile: pd.DataFrame, tau_s: float) -> float:
    t = profile["t_bin_sec"].to_numpy(dtype=float)
    y = profile["residual_z_power"].to_numpy(dtype=float)
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


def _projection_table(profiles: pd.DataFrame, tau_s: float) -> pd.DataFrame:
    rows = []
    by = ["analysis_source", "family_set", "frequency_band", "frequency_mhz", "n_components_requested"]
    for keys, grp in profiles.groupby(by, sort=True, dropna=False):
        rows.append(
            {
                **dict(zip(by, keys)),
                "projection_tau_s": float(tau_s),
                "fixed_template_amplitude": _fixed_template_projection(grp, float(tau_s)),
            }
        )
    return pd.DataFrame(rows)


def _fit_real_manifold(
    stacks: pd.DataFrame,
    family_set: str,
    components: list[int],
    sideband_s: float,
    timing_offsets: list[float],
    transition_durations: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    families = FAMILY_SETS[family_set]
    profile_rows: list[dict[str, object]] = []
    fit_rows: list[dict[str, object]] = []
    for source in sorted(stacks["analysis_source"].dropna().astype(str).unique()):
        source_stacks = stacks[stacks["analysis_source"].astype(str).eq(source)].copy()
        for freq in sorted(source_stacks["frequency_mhz"].dropna().astype(float).unique()):
            controls, columns = _curve_matrix(source_stacks, source, freq, families)
            model = _build_model(controls, columns, sideband_s=sideband_s)
            if model is None:
                continue
            real_y, real_sigma, real_meta = _real_curve(source_stacks, source, freq)
            if real_meta.empty:
                continue
            y, sigma, meta = _align_to_model(real_y, real_sigma, real_meta, model)
            control_spread = _control_uncertainty(controls, columns, model)
            frequency_band = int(
                source_stacks.loc[np.isclose(source_stacks["frequency_mhz"].astype(float), float(freq)), "frequency_band"].iloc[0]
            )
            for n_components in components:
                bg, pred_diag = _predict_background(y, model, int(n_components))
                profile = meta.copy()
                profile["analysis_source"] = source
                profile["family_set"] = family_set
                profile["frequency_band"] = frequency_band
                profile["frequency_mhz"] = float(freq)
                profile["n_components_requested"] = int(n_components)
                profile["n_components_used"] = int(pred_diag["n_components_used"])
                profile["real_stack_z_power"] = y
                profile["manifold_background_z_power"] = bg
                profile["residual_z_power"] = y - bg
                profile["residual_uncertainty"] = np.sqrt(np.square(sigma) + np.square(control_spread))
                profile["is_sideband_fit_bin"] = model.sideband_mask
                fit = _fit_profile(profile, timing_offsets, transition_durations)
                fit_rows.append(
                    {
                        "analysis_source": source,
                        "family_set": family_set,
                        "frequency_band": frequency_band,
                        "frequency_mhz": float(freq),
                        "n_components_requested": int(n_components),
                        "n_components_used": int(pred_diag["n_components_used"]),
                        "sideband_rms": float(pred_diag["sideband_rms"]),
                        **fit,
                    }
                )
                profile_rows.extend(profile.replace([np.inf, -np.inf], np.nan).to_dict("records"))
    return pd.DataFrame(profile_rows), pd.DataFrame(fit_rows)


def _inject_source_template(stacks: pd.DataFrame, source: str, amplitude: float, tau_s: float) -> pd.DataFrame:
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


def run_injections(
    sources: list[str],
    family_set: str,
    components: list[int],
    amplitudes: list[float],
    injection_taus: list[float],
    sideband_s: float,
    timing_offsets: list[float],
    transition_durations: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_results = []
    all_profiles = []
    for source in sources:
        stack_root = STACK_ROOTS[source]
        stacks = _read(stack_root / "lower_v_stackfirst_control_stacks.csv")
        stacks = stacks[stacks["analysis_source"].astype(str).eq(source)].copy()
        if stacks.empty:
            continue
        baseline_profiles, baseline_fits = _fit_real_manifold(
            stacks,
            family_set=family_set,
            components=components,
            sideband_s=sideband_s,
            timing_offsets=timing_offsets,
            transition_durations=transition_durations,
        )
        baseline_fits = baseline_fits.rename(
            columns={
                "amplitude": "baseline_residual_amplitude",
                "uncertainty": "baseline_residual_uncertainty",
                "stack_fit_snr": "baseline_residual_fit_snr",
                "delta_bic": "baseline_delta_bic",
            }
        )
        baseline_profiles["injection_amplitude"] = 0.0
        baseline_profiles["injection_tau_s"] = np.nan
        baseline_profiles["profile_role"] = "baseline"
        all_profiles.append(baseline_profiles)
        key_cols = ["analysis_source", "family_set", "frequency_band", "frequency_mhz", "n_components_requested"]
        for tau in injection_taus:
            baseline_projection = _projection_table(baseline_profiles, float(tau)).rename(
                columns={"fixed_template_amplitude": "baseline_fixed_template_amplitude"}
            )
            for amp in amplitudes:
                injected = _inject_source_template(stacks, source, amplitude=float(amp), tau_s=float(tau))
                profiles, fits = _fit_real_manifold(
                    injected,
                    family_set=family_set,
                    components=components,
                    sideband_s=sideband_s,
                    timing_offsets=timing_offsets,
                    transition_durations=transition_durations,
                )
                injected_projection = _projection_table(profiles, float(tau)).rename(
                    columns={"fixed_template_amplitude": "injected_fixed_template_amplitude"}
                )
                profiles["injection_amplitude"] = float(amp)
                profiles["injection_tau_s"] = float(tau)
                profiles["profile_role"] = "injected"
                all_profiles.append(profiles)
                merged = fits.merge(
                    baseline_fits[
                        key_cols
                        + [
                            "baseline_residual_amplitude",
                            "baseline_residual_uncertainty",
                            "baseline_residual_fit_snr",
                            "baseline_delta_bic",
                        ]
                    ],
                    on=key_cols,
                    how="left",
                )
                merged = merged.merge(
                    baseline_projection[key_cols + ["projection_tau_s", "baseline_fixed_template_amplitude"]],
                    on=key_cols,
                    how="left",
                )
                merged = merged.merge(
                    injected_projection[key_cols + ["projection_tau_s", "injected_fixed_template_amplitude"]],
                    on=key_cols + ["projection_tau_s"],
                    how="left",
                )
                merged["injection_amplitude"] = float(amp)
                merged["injection_tau_s"] = float(tau)
                merged["recovered_delta_amplitude"] = merged["amplitude"] - merged["baseline_residual_amplitude"]
                merged["recovery_fraction"] = merged["recovered_delta_amplitude"] / float(amp)
                merged["attenuation_fraction"] = 1.0 - merged["recovery_fraction"]
                merged["fixed_recovered_delta_amplitude"] = (
                    merged["injected_fixed_template_amplitude"] - merged["baseline_fixed_template_amplitude"]
                )
                merged["fixed_recovery_fraction"] = merged["fixed_recovered_delta_amplitude"] / float(amp)
                merged["fixed_attenuation_fraction"] = 1.0 - merged["fixed_recovery_fraction"]
                all_results.append(merged)
    results = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    profiles = pd.concat(all_profiles, ignore_index=True) if all_profiles else pd.DataFrame()
    return results, profiles


def _load_fixed_false_floor(sources: list[str], family_set: str, component: int, tau_s: float) -> pd.DataFrame:
    rows = []
    for source in sources:
        if source == "earth":
            p = ROOT / "outputs/lower_v_control_manifold_earth_positive_control_v1/control_manifold_residual_profiles.csv"
        else:
            p = ROOT / "outputs/lower_v_control_manifold_background_sun_fornax_v1/control_manifold_residual_profiles.csv"
        if not p.exists():
            continue
        df = _read(p)
        sub = df[
            df["analysis_source"].astype(str).eq(source)
            & df["family_set"].astype(str).eq(family_set)
            & df["n_components_requested"].astype(int).eq(int(component))
            & df["residual_role"].astype(str).eq("pseudo_control")
        ].copy()
        if sub.empty:
            continue
        projected_rows = []
        by = ["analysis_source", "family_set", "frequency_band", "frequency_mhz", "pseudo_control_id"]
        for keys, grp in sub.groupby(by, sort=True, dropna=False):
            projected_rows.append(
                {
                    **dict(zip(by, keys)),
                    "fixed_pseudo_control_amplitude": _fixed_template_projection(grp, float(tau_s)),
                }
            )
        projected = pd.DataFrame(projected_rows)
        for keys, grp in projected.groupby(["analysis_source", "family_set", "frequency_band", "frequency_mhz"], sort=True, dropna=False):
            vals = pd.to_numeric(grp["fixed_pseudo_control_amplitude"], errors="coerce").dropna().to_numpy(dtype=float)
            if vals.size == 0:
                continue
            rows.append(
                {
                    **dict(zip(["analysis_source", "family_set", "frequency_band", "frequency_mhz"], keys)),
                    "fixed_pseudo_median_amplitude": float(np.nanmedian(vals)),
                    "fixed_pseudo_q25_amplitude": float(np.nanquantile(vals, 0.25)),
                    "fixed_pseudo_q75_amplitude": float(np.nanquantile(vals, 0.75)),
                    "fixed_pseudo_abs_q75_amplitude": float(np.nanquantile(np.abs(vals), 0.75)),
                    "n_fixed_pseudo_controls": int(vals.size),
                }
            )
    return pd.DataFrame(rows)


def summarize_sensitivity(results: pd.DataFrame, false_floor: pd.DataFrame, main_component: int) -> pd.DataFrame:
    main = results[results["n_components_requested"].astype(int).eq(int(main_component))].copy()
    rows = []
    by = ["analysis_source", "frequency_band", "frequency_mhz", "injection_tau_s"]
    for keys, grp in main.groupby(by, sort=True, dropna=False):
        rec = pd.to_numeric(grp["fixed_recovery_fraction"], errors="coerce").dropna().to_numpy(dtype=float)
        if rec.size == 0:
            continue
        source, band, freq, tau = keys
        floor = false_floor[
            false_floor["analysis_source"].astype(str).eq(str(source))
            & np.isclose(false_floor["frequency_mhz"].astype(float), float(freq))
        ]
        pseudo_abs_q75 = float(floor["fixed_pseudo_abs_q75_amplitude"].iloc[0]) if not floor.empty and "fixed_pseudo_abs_q75_amplitude" in floor.columns else np.nan
        median_recovery = float(np.nanmedian(rec))
        q25_recovery = float(np.nanquantile(rec, 0.25))
        q75_recovery = float(np.nanquantile(rec, 0.75))
        if not np.isfinite(median_recovery) or abs(median_recovery) > 3.0:
            stability_flag = "unstable_or_nonphysical"
        elif median_recovery <= 0.0:
            stability_flag = "negative_transfer"
        elif median_recovery < 0.1:
            stability_flag = "low_recovery"
        elif median_recovery < 0.5:
            stability_flag = "attenuated"
        elif median_recovery <= 1.5:
            stability_flag = "usable"
        else:
            stability_flag = "amplified"
        plot_ok = np.isfinite(median_recovery) and abs(median_recovery) <= 3.0
        sensitivity_ok = np.isfinite(median_recovery) and 0.0 < median_recovery <= 3.0
        rows.append(
            {
                "analysis_source": source,
                "frequency_band": int(band),
                "frequency_mhz": float(freq),
                "injection_tau_s": float(tau),
                "median_fixed_recovery_fraction": median_recovery,
                "q25_fixed_recovery_fraction": q25_recovery,
                "q75_fixed_recovery_fraction": q75_recovery,
                "transfer_stability_flag": stability_flag,
                "plot_median_fixed_recovery_fraction": median_recovery if plot_ok else np.nan,
                "plot_q25_fixed_recovery_fraction": q25_recovery if plot_ok else np.nan,
                "plot_q75_fixed_recovery_fraction": q75_recovery if plot_ok else np.nan,
                "fixed_pseudo_abs_q75_amplitude": pseudo_abs_q75,
                "rough_2iqr_detectable_injected_amplitude": float(2.0 * pseudo_abs_q75 / median_recovery)
                if np.isfinite(pseudo_abs_q75) and sensitivity_ok
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def plot_recovery_spectrum(summary: pd.DataFrame, out_dir: Path, main_tau: float) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), sharey=True)
    for ax, source in zip(axes, ["earth", "sun", "fornax_a"]):
        sub = summary[
            summary["analysis_source"].astype(str).eq(source)
            & np.isclose(summary["injection_tau_s"].astype(float), float(main_tau))
        ].sort_values("frequency_mhz")
        if sub.empty:
            ax.set_title(SOURCE_LABEL.get(source, source))
            continue
        plot_sub = sub[np.isfinite(pd.to_numeric(sub["plot_median_fixed_recovery_fraction"], errors="coerce"))].copy()
        bad_sub = sub[~np.isfinite(pd.to_numeric(sub["plot_median_fixed_recovery_fraction"], errors="coerce"))].copy()
        nonpositive_sub = plot_sub[pd.to_numeric(plot_sub["plot_median_fixed_recovery_fraction"], errors="coerce") <= 0.0].copy()
        positive_sub = plot_sub[pd.to_numeric(plot_sub["plot_median_fixed_recovery_fraction"], errors="coerce") > 0.0].copy()
        ax.axhline(1.0, color="0.55", lw=0.8, ls="--", label="no attenuation")
        ax.axhline(0.0, color="0.75", lw=0.8)
        if not positive_sub.empty:
            x = positive_sub["frequency_mhz"].to_numpy(dtype=float)
            ax.fill_between(
                x,
                positive_sub["plot_q25_fixed_recovery_fraction"].to_numpy(dtype=float),
                positive_sub["plot_q75_fixed_recovery_fraction"].to_numpy(dtype=float),
                color="#4c78a8",
                alpha=0.18,
            )
            ax.plot(x, positive_sub["plot_median_fixed_recovery_fraction"], marker="o", color="#1f78b4", lw=1.4)
        if not nonpositive_sub.empty:
            ax.scatter(
                nonpositive_sub["frequency_mhz"],
                np.full(len(nonpositive_sub), -0.08),
                marker="v",
                s=38,
                color="#d95f02",
                label="negative transfer",
                zorder=4,
            )
        if not bad_sub.empty:
            ax.scatter(
                bad_sub["frequency_mhz"],
                np.full(len(bad_sub), -0.16),
                marker="x",
                s=48,
                color="#b2182b",
                label="unstable",
                zorder=4,
            )
        ax.set_xscale("log")
        ax.set_xticks(list(FREQUENCY_MAP_MHZ.values()))
        ax.set_xticklabels([f"{v:g}" for v in FREQUENCY_MAP_MHZ.values()], rotation=45, ha="right")
        ax.set_ylim(-0.25, 3.1)
        ax.set_title(SOURCE_LABEL.get(source, source))
        ax.set_xlabel("frequency (MHz)")
        ax.grid(True, color="0.9", lw=0.5)
    axes[0].set_ylabel("fixed-template recovered / injected amplitude")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Control-manifold transfer function for injected positive occultation templates (tau={main_tau:g} s)")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    path = out_dir / f"control_manifold_transfer_function_tau{int(main_tau)}s.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_detectable_amplitude(summary: pd.DataFrame, out_dir: Path, main_tau: float) -> Path:
    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    for source, color in [("earth", "#333333"), ("sun", "#d95f02"), ("fornax_a", "#1f78b4")]:
        sub = summary[
            summary["analysis_source"].astype(str).eq(source)
            & np.isclose(summary["injection_tau_s"].astype(float), float(main_tau))
        ].sort_values("frequency_mhz")
        if sub.empty:
            continue
        sub = sub[np.isfinite(pd.to_numeric(sub["rough_2iqr_detectable_injected_amplitude"], errors="coerce"))].copy()
        if sub.empty:
            continue
        ax.plot(
            sub["frequency_mhz"],
            sub["rough_2iqr_detectable_injected_amplitude"],
            marker="o",
            lw=1.4,
            color=color,
            label=SOURCE_LABEL.get(source, source),
        )
    ax.set_xscale("log")
    ax.set_xticks(list(FREQUENCY_MAP_MHZ.values()))
    ax.set_xticklabels([f"{v:g}" for v in FREQUENCY_MAP_MHZ.values()], rotation=45, ha="right")
    ax.set_xlabel("frequency (MHz)")
    ax.set_ylabel("rough detectable injected amplitude")
    ax.set_title(f"Approximate sensitivity after manifold subtraction (2 x pseudo-control IQR / transfer, tau={main_tau:g} s)")
    ax.grid(True, color="0.9", lw=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = out_dir / f"control_manifold_rough_detectable_amplitude_tau{int(main_tau)}s.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_example_injection_profiles(profiles: pd.DataFrame, out_dir: Path, main_component: int, main_tau: float, example_amp: float) -> Path:
    examples = [("sun", 0.90), ("fornax_a", 2.20), ("earth", 3.93)]
    fig, axes = plt.subplots(len(examples), 2, figsize=(13.5, 8.5), sharex=True)
    for i, (source, freq) in enumerate(examples):
        base = profiles[
            profiles["analysis_source"].astype(str).eq(source)
            & profiles["n_components_requested"].astype(int).eq(int(main_component))
            & profiles["profile_role"].astype(str).eq("baseline")
            & np.isclose(profiles["frequency_mhz"].astype(float), float(freq))
        ].copy()
        inj = profiles[
            profiles["analysis_source"].astype(str).eq(source)
            & profiles["n_components_requested"].astype(int).eq(int(main_component))
            & profiles["profile_role"].astype(str).eq("injected")
            & np.isclose(profiles["frequency_mhz"].astype(float), float(freq))
            & np.isclose(profiles["injection_tau_s"].astype(float), float(main_tau))
            & np.isclose(profiles["injection_amplitude"].astype(float), float(example_amp))
        ].copy()
        for j, event_type in enumerate(["disappearance", "reappearance"]):
            ax = axes[i, j]
            b = base[base["event_type"].eq(event_type)].sort_values("t_bin_sec")
            g = inj[inj["event_type"].eq(event_type)].sort_values("t_bin_sec")
            if not b.empty:
                ax.errorbar(
                    b["t_bin_sec"] / 60.0,
                    b["residual_z_power"],
                    yerr=b["residual_uncertainty"],
                    color="0.35",
                    marker="o",
                    ms=2.3,
                    lw=1.0,
                    elinewidth=0.45,
                    capsize=1.0,
                    label="baseline residual",
                )
            if not g.empty:
                ax.errorbar(
                    g["t_bin_sec"] / 60.0,
                    g["residual_z_power"],
                    yerr=g["residual_uncertainty"],
                    color="#1f78b4",
                    marker="o",
                    ms=2.3,
                    lw=1.2,
                    elinewidth=0.45,
                    capsize=1.0,
                    label=f"+ injected A={example_amp:g}",
                )
            ax.axvline(0.0, color="black", ls=":", lw=0.8)
            ax.axhline(0.0, color="0.75", lw=0.7)
            ax.grid(True, color="0.92", lw=0.5)
            ax.set_title(f"{SOURCE_LABEL.get(source, source)} {freq:.2f} MHz {event_type}", fontsize=9)
            if j == 0:
                ax.set_ylabel("residual power")
            if i == len(examples) - 1:
                ax.set_xlabel("minutes from predicted event")
            if i == 0 and j == 1:
                ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Actual residual profiles before/after synthetic source injection (k={main_component}, tau={main_tau:g} s)")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = out_dir / f"example_injected_residual_profiles_tau{int(main_tau)}s_amp{str(example_amp).replace('.', 'p')}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report(out_dir: Path, summary: pd.DataFrame, plot_paths: list[Path], main_tau: float, main_component: int) -> None:
    main = summary[
        np.isclose(summary["injection_tau_s"].astype(float), float(main_tau))
    ].copy()
    cols = [
        "analysis_source",
        "frequency_mhz",
        "median_fixed_recovery_fraction",
        "q25_fixed_recovery_fraction",
        "q75_fixed_recovery_fraction",
        "transfer_stability_flag",
        "fixed_pseudo_abs_q75_amplitude",
        "rough_2iqr_detectable_injected_amplitude",
    ]
    lines = [
        "# Control-Manifold Injection-Recovery",
        "",
        "This run measures the transfer function of the lower-V empirical background subtraction.",
        "",
        "A known positive-source occultation template is injected into the real stack only, not into controls. The same sideband-trained control-manifold subtraction is then rerun, and the recovered amplitude change is measured.",
        "",
        "Definitions:",
        "",
        "- recovery fraction = fixed-template projection change after injection / injected amplitude;",
        "- values near 1 mean little attenuation;",
        "- values near 0 mean the subtraction erased most of the injected source template;",
        "- negative or nonphysical values mean the current manifold fit is not a valid sensitivity estimator for that channel;",
        "- rough detectable amplitude = 2 x pseudo-control absolute-IQR / recovery fraction.",
        "",
        f"Main plotted component count: k={int(main_component)}",
        f"Main plotted injection transition duration: tau={float(main_tau):.0f} s",
        "",
        "## Study-Level Conclusion",
        "",
        "The current control-manifold subtraction is useful as a diagnostic, but it is not yet a neutral weak-source detector. Several channels preserve only a small fraction of injected source-like contrast, and some channels return negative or nonphysical transfer values. Therefore weak Sun/Fornax-A residuals, or their absence after subtraction, should not be interpreted as definitive detections or definitive non-detections.",
        "",
        "At this stage the method is best treated as positive-control validation and background-systematics characterization. Source claims should be restricted to channels whose injection-recovery transfer is positive, bounded, and stable under controls.",
        "",
        "## Main Transfer/Sensitivity Summary",
        "",
        main[cols].sort_values(["analysis_source", "frequency_mhz"]).to_string(index=False),
        "",
        "## Plots",
        "",
    ]
    lines.extend(f"- `{path.name}`" for path in plot_paths)
    lines += [
        "",
        "## Interpretation",
        "",
        "- This test does not prove a Sun or Fornax-A detection.",
        "- It quantifies how much source signal the current subtraction would keep or suppress.",
        "- If the rough detectable amplitude is comparable to or larger than plausible Sun/Fornax residuals, then non-detection is not a strong astrophysical statement.",
        "- If Earth is recovered but weak injected templates are strongly attenuated, the current study is validated only for strong positives, not for weak-source limits.",
    ]
    (out_dir / "control_manifold_injection_recovery_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--sources", default="earth,sun,fornax_a")
    parser.add_argument("--family-set", default="all_controls")
    parser.add_argument("--components", default="1,2,3")
    parser.add_argument("--main-components", type=int, default=2)
    parser.add_argument("--amplitudes", default="0.05,0.1,0.2,0.5,1.0")
    parser.add_argument("--injection-taus-s", default="0,300,900")
    parser.add_argument("--main-tau-s", type=float, default=300.0)
    parser.add_argument("--example-amplitude", type=float, default=0.2)
    parser.add_argument("--sideband-s", type=float, default=600.0)
    parser.add_argument("--timing-offsets-s", default="0")
    parser.add_argument("--transition-durations-s", default="0,120,300,600,900")
    args = parser.parse_args()

    out_dir = ensure_dir(Path(args.out_dir))
    sources = [x.strip() for x in str(args.sources).split(",") if x.strip()]
    components = [int(x.strip()) for x in str(args.components).split(",") if x.strip()]
    amplitudes = [float(x.strip()) for x in str(args.amplitudes).split(",") if x.strip()]
    injection_taus = [float(x.strip()) for x in str(args.injection_taus_s).split(",") if x.strip()]
    timing_offsets = [float(x.strip()) for x in str(args.timing_offsets_s).split(",") if x.strip()]
    transition_durations = [float(x.strip()) for x in str(args.transition_durations_s).split(",") if x.strip()]
    write_json(
        out_dir / "run_config.json",
        {
            "sources": sources,
            "family_set": str(args.family_set),
            "components": components,
            "main_components": int(args.main_components),
            "amplitudes": amplitudes,
            "injection_taus_s": injection_taus,
            "main_tau_s": float(args.main_tau_s),
            "example_amplitude": float(args.example_amplitude),
            "sideband_s": float(args.sideband_s),
            "timing_offsets_s": timing_offsets,
            "transition_durations_s": transition_durations,
            "stack_roots": {k: str(v) for k, v in STACK_ROOTS.items()},
            "software_versions": software_versions(),
        },
    )

    print("Running control-manifold injection recovery...", flush=True)
    results, profiles = run_injections(
        sources=sources,
        family_set=str(args.family_set),
        components=components,
        amplitudes=amplitudes,
        injection_taus=injection_taus,
        sideband_s=float(args.sideband_s),
        timing_offsets=timing_offsets,
        transition_durations=transition_durations,
    )
    results.to_csv(out_dir / "control_manifold_injection_recovery_by_amplitude.csv", index=False)
    profiles.to_csv(out_dir / "control_manifold_injection_recovery_profiles.csv", index=False)
    false_floor = _load_fixed_false_floor(sources, str(args.family_set), int(args.main_components), float(args.main_tau_s))
    false_floor.to_csv(out_dir / "control_manifold_injection_false_floor_reference.csv", index=False)
    summary = summarize_sensitivity(results, false_floor, int(args.main_components))
    summary.to_csv(out_dir / "control_manifold_injection_recovery_summary.csv", index=False)
    print("Writing injection-recovery plots...", flush=True)
    paths = [
        plot_recovery_spectrum(summary, out_dir, float(args.main_tau_s)),
        plot_detectable_amplitude(summary, out_dir, float(args.main_tau_s)),
        plot_example_injection_profiles(profiles, out_dir, int(args.main_components), float(args.main_tau_s), float(args.example_amplitude)),
    ]
    write_report(out_dir, summary, paths, float(args.main_tau_s), int(args.main_components))
    print(out_dir / "control_manifold_injection_recovery_report.md")


if __name__ == "__main__":
    main()
