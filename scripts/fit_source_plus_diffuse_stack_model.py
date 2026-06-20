#!/usr/bin/env python
"""Fit stacked profiles with diffuse background plus source occultation.

The input is the same-window simulator audit table, where observed raw profiles
and diffuse-sky simulator profiles were computed on identical event windows.
For each source/frequency/antenna stack this script compares:

* constant-only;
* diffuse-only;
* source-occultation-only;
* diffuse plus source occultation.

The source term uses the stack-first finite-duration occultation template.  A
positive amplitude means disappearance decreases and reappearance increases.
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

from rylevonberg.stackfit import stacked_event_template  # noqa: E402
from rylevonberg.util import ensure_dir, software_versions, write_json  # noqa: E402


ANT_LABEL = {"rv1_coarse": "upper V", "rv2_coarse": "lower V"}
ANT_COLOR = {"rv1_coarse": "#4c78a8", "rv2_coarse": "#d95f02"}
DIFFUSE_LABEL = {
    "model_axisymmetric_mean": "old radial diffuse",
    "model_eh_azimuth_yaw13": "yawed E/H diffuse",
}


def _read(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return read_table(path, low_memory=False, **kwargs)


def _weighted_lstsq(X: np.ndarray, y: np.ndarray, sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    sigma = np.asarray(sigma, dtype=float)
    fallback = np.nanmedian(sigma[np.isfinite(sigma) & (sigma > 0)])
    if not np.isfinite(fallback) or fallback <= 0:
        fallback = 1.0
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, fallback)
    weights = 1.0 / np.maximum(sigma, fallback * 0.25) ** 2
    weights = weights / np.nanmedian(weights)
    w = np.sqrt(weights)
    beta, *_ = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)
    resid = y - X @ beta
    wrss = float(np.sum(weights * resid**2))
    dof = max(int(len(y) - X.shape[1]), 1)
    sigma2 = wrss / dof
    cov = np.linalg.pinv((X.T * weights) @ X) * sigma2
    return beta, resid, wrss, cov


def _bic(wrss: float, n: int, n_params: int) -> float:
    wrss = max(float(wrss), np.finfo(float).tiny)
    n = max(int(n), 1)
    return float(n * np.log(wrss / n) + int(n_params) * np.log(n))


def _design(
    t: np.ndarray,
    event_types: np.ndarray,
    diffuse: np.ndarray,
    model_kind: str,
    timing_offset_s: float = 0.0,
    transition_duration_s: float = 0.0,
) -> tuple[np.ndarray, list[str]]:
    cols = [np.ones_like(t, dtype=float)]
    names = ["intercept"]
    if "diffuse" in model_kind:
        cols.append(diffuse)
        names.append("diffuse_amplitude")
    if "source" in model_kind:
        tmpl = stacked_event_template(
            t,
            event_types,
            timing_offset_sec=float(timing_offset_s),
            transition_duration_sec=float(transition_duration_s),
        )
        cols.append(tmpl)
        names.append("source_amplitude")
    return np.column_stack(cols), names


def _fit_model_grid(
    t: np.ndarray,
    y: np.ndarray,
    sigma: np.ndarray,
    event_types: np.ndarray,
    diffuse: np.ndarray,
    model_kind: str,
    timing_offsets: list[float],
    transition_durations: list[float],
) -> dict[str, float]:
    if "source" in model_kind:
        grid = [(float(dt), float(tau)) for dt in timing_offsets for tau in transition_durations]
    else:
        grid = [(0.0, 0.0)]
    best: dict[str, float] | None = None
    for dt, tau in grid:
        X, names = _design(t, event_types, diffuse, model_kind, dt, tau)
        beta, resid, wrss, cov = _weighted_lstsq(X, y, sigma)
        bic = _bic(wrss, len(y), X.shape[1])
        row: dict[str, float] = {
            "model_kind": model_kind,
            "n_bins": int(len(y)),
            "n_params": int(X.shape[1]),
            "bic": float(bic),
            "weighted_rss": float(wrss),
            "residual_rms": float(np.sqrt(np.nanmean(resid**2))) if resid.size else np.nan,
            "best_timing_offset_s": float(dt) if "source" in model_kind else np.nan,
            "best_transition_duration_s": float(tau) if "source" in model_kind else np.nan,
        }
        for idx, name in enumerate(names):
            row[name] = float(beta[idx])
            var = float(cov[idx, idx]) if cov.size else np.nan
            row[f"{name}_uncertainty"] = float(np.sqrt(var)) if np.isfinite(var) and var >= 0 else np.nan
        if "source_amplitude" not in row:
            row["source_amplitude"] = np.nan
            row["source_amplitude_uncertainty"] = np.nan
        if "diffuse_amplitude" not in row:
            row["diffuse_amplitude"] = np.nan
            row["diffuse_amplitude_uncertainty"] = np.nan
        if best is None or row["bic"] < best["bic"]:
            best = row
    return best or {}


def _prepare_group(summary: pd.DataFrame, diffuse_method: str) -> pd.DataFrame:
    obs = summary[summary["method"].eq("observed_raw")].copy()
    diff = summary[summary["method"].eq(diffuse_method)].copy()
    keys = ["source_name", "event_type", "frequency_band", "frequency_mhz", "antenna", "t_bin_sec"]
    obs = obs.rename(columns={"median_z_power": "observed_z", "median_z_power_err": "observed_err"})
    diff = diff.rename(columns={"median_z_power": "diffuse_z", "median_z_power_err": "diffuse_err"})
    merged = obs[keys + ["observed_z", "observed_err", "n_events", "n_points"]].merge(
        diff[keys + ["diffuse_z", "diffuse_err"]],
        on=keys,
        how="inner",
    )
    return merged


def fit_source_plus_diffuse(
    summary: pd.DataFrame,
    source: str,
    diffuse_methods: list[str],
    timing_offsets: list[float],
    transition_durations: list[float],
    min_bins: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit_rows = []
    curve_rows = []
    for diffuse_method in diffuse_methods:
        merged = _prepare_group(summary, diffuse_method)
        merged = merged[merged["source_name"].astype(str).str.lower().eq(source.lower())].copy()
        group_cols = ["source_name", "frequency_band", "frequency_mhz", "antenna"]
        for keys, grp in merged.groupby(group_cols, sort=True, dropna=False):
            for stack_group, sub in [("combined", grp), *[(et, g) for et, g in grp.groupby("event_type", sort=True)]]:
                sub = sub.sort_values(["event_type", "t_bin_sec"]).copy()
                if len(sub) < int(min_bins):
                    continue
                t = sub["t_bin_sec"].to_numpy(dtype=float)
                y = sub["observed_z"].to_numpy(dtype=float)
                sigma = sub["observed_err"].to_numpy(dtype=float)
                diffuse = sub["diffuse_z"].to_numpy(dtype=float)
                event_types = sub["event_type"].astype(str).to_numpy()
                good = np.isfinite(t) & np.isfinite(y) & np.isfinite(diffuse)
                good &= np.isfinite(sigma) & (sigma > 0)
                if np.count_nonzero(good) < int(min_bins):
                    continue
                t = t[good]
                y = y[good]
                sigma = sigma[good]
                event_types = event_types[good]
                diffuse = diffuse[good]
                diffuse_center = float(np.nanmedian(diffuse))
                diffuse = diffuse - diffuse_center

                fits = {
                    kind: _fit_model_grid(
                        t,
                        y,
                        sigma,
                        event_types,
                        diffuse,
                        kind,
                        timing_offsets,
                        transition_durations,
                    )
                    for kind in ["constant", "diffuse", "source", "diffuse_source"]
                }
                full = fits["diffuse_source"]
                full_bic = full.get("bic", np.nan)
                diffuse_bic = fits["diffuse"].get("bic", np.nan)
                source_bic = fits["source"].get("bic", np.nan)
                constant_bic = fits["constant"].get("bic", np.nan)
                source_amp = full.get("source_amplitude", np.nan)
                source_unc = full.get("source_amplitude_uncertainty", np.nan)
                delta_vs_diffuse = float(diffuse_bic - full_bic) if np.isfinite(diffuse_bic) and np.isfinite(full_bic) else np.nan
                delta_vs_source = float(source_bic - full_bic) if np.isfinite(source_bic) and np.isfinite(full_bic) else np.nan
                delta_source_vs_constant = (
                    float(constant_bic - source_bic) if np.isfinite(constant_bic) and np.isfinite(source_bic) else np.nan
                )
                if np.isfinite(source_amp) and source_amp > 0 and np.isfinite(delta_vs_diffuse) and delta_vs_diffuse >= 10:
                    classification = "occultation_like_term_strongly_supported"
                elif np.isfinite(source_amp) and source_amp > 0 and np.isfinite(delta_vs_diffuse) and delta_vs_diffuse >= 2:
                    classification = "occultation_like_term_supported"
                elif np.isfinite(source_amp) and source_amp < 0 and np.isfinite(delta_vs_diffuse) and delta_vs_diffuse >= 6:
                    classification = "anti_template_occultation_like_term"
                else:
                    classification = "not_decisive"

                meta = dict(zip(group_cols, keys))
                fit_rows.append(
                    {
                        **meta,
                        "diffuse_method": diffuse_method,
                        "stack_group": str(stack_group),
                        "median_events_per_bin": float(np.nanmedian(sub.loc[good, "n_events"])),
                        "n_profile_bins": int(np.count_nonzero(good)),
                        "diffuse_center_subtracted": diffuse_center,
                        "constant_bic": constant_bic,
                        "diffuse_only_bic": diffuse_bic,
                        "source_only_bic": source_bic,
                        "diffuse_plus_source_bic": full_bic,
                        "delta_bic_source_plus_diffuse_vs_diffuse_only": delta_vs_diffuse,
                        "delta_bic_source_plus_diffuse_vs_source_only": delta_vs_source,
                        "delta_bic_source_only_vs_constant": delta_source_vs_constant,
                        "source_amplitude": source_amp,
                        "source_amplitude_uncertainty": source_unc,
                        "source_amplitude_over_uncertainty": (
                            float(source_amp / source_unc) if np.isfinite(source_unc) and source_unc > 0 else np.nan
                        ),
                        "diffuse_amplitude": full.get("diffuse_amplitude", np.nan),
                        "diffuse_amplitude_uncertainty": full.get("diffuse_amplitude_uncertainty", np.nan),
                        "best_timing_offset_s": full.get("best_timing_offset_s", np.nan),
                        "best_transition_duration_s": full.get("best_transition_duration_s", np.nan),
                        "full_model_residual_rms": full.get("residual_rms", np.nan),
                        "diffuse_only_residual_rms": fits["diffuse"].get("residual_rms", np.nan),
                        "source_only_residual_rms": fits["source"].get("residual_rms", np.nan),
                        "classification": classification,
                    }
                )

                # Save curves for the full and diffuse-only models on the same bins.
                X_diffuse, _ = _design(t, event_types, diffuse, "diffuse")
                beta_d = np.asarray(
                    [
                        fits["diffuse"].get("intercept", np.nan),
                        fits["diffuse"].get("diffuse_amplitude", np.nan),
                    ],
                    dtype=float,
                )
                diffuse_only_curve = X_diffuse @ beta_d
                X_full, _ = _design(
                    t,
                    event_types,
                    diffuse,
                    "diffuse_source",
                    full.get("best_timing_offset_s", 0.0),
                    full.get("best_transition_duration_s", 0.0),
                )
                beta_full = np.asarray(
                    [
                        full.get("intercept", np.nan),
                        full.get("diffuse_amplitude", np.nan),
                        full.get("source_amplitude", np.nan),
                    ],
                    dtype=float,
                )
                full_curve = X_full @ beta_full
                source_template = X_full[:, -1]
                for idx, row_idx in enumerate(np.flatnonzero(good)):
                    row = sub.iloc[int(row_idx)]
                    curve_rows.append(
                        {
                            **meta,
                            "diffuse_method": diffuse_method,
                            "stack_group": str(stack_group),
                            "event_type": str(row["event_type"]),
                            "t_bin_sec": float(row["t_bin_sec"]),
                            "observed_z": float(row["observed_z"]),
                            "observed_err": float(row["observed_err"]),
                            "diffuse_z_centered": float(diffuse[idx]),
                            "source_template": float(source_template[idx]),
                            "diffuse_only_curve": float(diffuse_only_curve[idx]),
                            "diffuse_plus_source_curve": float(full_curve[idx]),
                            "source_component": float(beta_full[-1] * source_template[idx]),
                            "n_events": int(row["n_events"]),
                        }
                    )
    return pd.DataFrame(fit_rows), pd.DataFrame(curve_rows)


def plot_heatmaps(fits: pd.DataFrame, source: str, diffuse_method: str, out_dir: Path) -> Path:
    sub = fits[
        fits["source_name"].astype(str).str.lower().eq(source.lower())
        & fits["diffuse_method"].eq(diffuse_method)
        & fits["stack_group"].eq("combined")
    ].copy()
    metrics = [
        ("source_amplitude", "source amplitude", "viridis"),
        ("delta_bic_source_plus_diffuse_vs_diffuse_only", "Delta BIC vs diffuse-only", "magma"),
        ("best_transition_duration_s", "transition duration (s)", "cividis"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 4.8), squeeze=False)
    for ax, (col, title, cmap) in zip(axes[0], metrics):
        pivot = sub.pivot_table(index="antenna", columns="frequency_mhz", values=col, aggfunc="first")
        order = [a for a in ["rv2_coarse", "rv1_coarse"] if a in pivot.index]
        pivot = pivot.loc[order]
        data = pivot.to_numpy(dtype=float)
        if col == "source_amplitude":
            vmax = np.nanmax(np.abs(data)) if np.isfinite(data).any() else 1.0
            vmin = -vmax
            cmap = "coolwarm"
        else:
            vmin = np.nanmin(data) if np.isfinite(data).any() else 0.0
            vmax = np.nanmax(data) if np.isfinite(data).any() else 1.0
        im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_yticks(np.arange(len(pivot.index)), [ANT_LABEL.get(str(x), str(x)) for x in pivot.index])
        ax.set_xticks(np.arange(len(pivot.columns)), [f"{float(x):.2f}" for x in pivot.columns], rotation=45, ha="right")
        for y in range(data.shape[0]):
            for x in range(data.shape[1]):
                val = data[y, x]
                if np.isfinite(val):
                    txt = f"{val:.1f}" if col != "source_amplitude" else f"{val:.2f}"
                    ax.text(x, y, txt, ha="center", va="center", fontsize=7, color="white" if col.startswith("delta") else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle(f"{source.replace('_', ' ').title()}: source + diffuse stack fit ({DIFFUSE_LABEL.get(diffuse_method, diffuse_method)})")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    path = out_dir / f"{source}_{diffuse_method}_source_plus_diffuse_heatmaps.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_profile_grid(curves: pd.DataFrame, fits: pd.DataFrame, source: str, diffuse_method: str, antenna: str, out_dir: Path) -> Path:
    sub = curves[
        curves["source_name"].astype(str).str.lower().eq(source.lower())
        & curves["diffuse_method"].eq(diffuse_method)
        & curves["antenna"].eq(antenna)
        & curves["stack_group"].eq("combined")
    ].copy()
    freqs = sorted(sub["frequency_mhz"].dropna().unique())
    fig, axes = plt.subplots(len(freqs), 2, figsize=(12.5, max(10, 1.45 * len(freqs))), sharex=True)
    if len(freqs) == 1:
        axes = np.asarray([axes])
    for i, freq in enumerate(freqs):
        fit_row = fits[
            fits["source_name"].astype(str).str.lower().eq(source.lower())
            & fits["diffuse_method"].eq(diffuse_method)
            & fits["antenna"].eq(antenna)
            & np.isclose(fits["frequency_mhz"], float(freq))
            & fits["stack_group"].eq("combined")
        ]
        fit_text = ""
        if not fit_row.empty:
            row = fit_row.iloc[0]
            fit_text = (
                f"A={row['source_amplitude']:.2f}, "
                f"dBIC={row['delta_bic_source_plus_diffuse_vs_diffuse_only']:.1f}, "
                f"tau={row['best_transition_duration_s']:.0f}s"
            )
        for j, event_type in enumerate(["disappearance", "reappearance"]):
            ax = axes[i, j]
            grp = sub[np.isclose(sub["frequency_mhz"], float(freq)) & sub["event_type"].eq(event_type)].sort_values("t_bin_sec")
            if grp.empty:
                continue
            x = grp["t_bin_sec"].to_numpy(dtype=float) / 60.0
            y = grp["observed_z"].to_numpy(dtype=float)
            err = grp["observed_err"].fillna(0.0).to_numpy(dtype=float)
            ax.errorbar(x, y, yerr=err, color="black", marker="o", markersize=2.3, linewidth=1.0, capsize=1.0, label="observed")
            ax.plot(x, grp["diffuse_only_curve"], color="#4c78a8", linestyle="--", linewidth=1.5, label="diffuse only")
            ax.plot(x, grp["diffuse_plus_source_curve"], color="#d95f02", linewidth=1.8, label="diffuse + source")
            ax.axvline(0, color="black", linestyle=":", linewidth=0.8)
            ax.axhline(0, color="0.65", linewidth=0.75)
            ax.set_title(f"{freq:.2f} MHz {event_type}" + (f"\n{fit_text}" if j == 0 and fit_text else ""), fontsize=8.5)
            if j == 0 and i == len(freqs) // 2:
                ax.set_ylabel("stacked normalized power")
            if i == len(freqs) - 1:
                ax.set_xlabel("minutes from predicted event")
            if i == 0 and j == 1:
                ax.legend(frameon=False, fontsize=7)
    fig.suptitle(
        f"{source.replace('_', ' ').title()} {ANT_LABEL.get(antenna, antenna)}: observed vs diffuse+source stack model",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    path = out_dir / f"{source}_{antenna}_{diffuse_method}_source_plus_diffuse_profile_grid.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report(out_dir: Path, source: str, diffuse_method: str, fits: pd.DataFrame, paths: list[Path], config: dict[str, object]) -> None:
    combined = fits[
        fits["source_name"].astype(str).str.lower().eq(source.lower())
        & fits["diffuse_method"].eq(diffuse_method)
        & fits["stack_group"].eq("combined")
    ].copy()
    display_cols = [
        "frequency_mhz",
        "antenna",
        "median_events_per_bin",
        "source_amplitude",
        "source_amplitude_uncertainty",
        "delta_bic_source_plus_diffuse_vs_diffuse_only",
        "delta_bic_source_plus_diffuse_vs_source_only",
        "best_timing_offset_s",
        "best_transition_duration_s",
        "classification",
    ]
    lower = combined[combined["antenna"].eq("rv2_coarse")].sort_values("frequency_mhz")
    lines = [
        "# Source Plus Diffuse Stack Model",
        "",
        "## Purpose",
        "",
        "This implements the next model after the diffuse-only simulator audit: fit the stacked observed profile as",
        "",
        "    observed(t) = intercept + beta_diffuse * diffuse_simulator(t) + A_source * occultation_template(t)",
        "",
        "The occultation-like term is fitted after stacking, not event by event.  Positive `A_source` means disappearance",
        "drops and reappearance rises in the expected source-like direction.  It should not be interpreted by itself as",
        "proof that the residual is Fornax-A emission; it can also represent another diffuse structure that is occulted,",
        "beam-weighted, or repeatedly aligned with the event geometry.",
        "",
        "## Configuration",
        "",
        pd.Series(config).to_string(),
        "",
        "## Lower-V Combined Fits",
        "",
        lower[display_cols].to_string(index=False) if not lower.empty else "No lower-V combined rows.",
        "",
        "## Interpretation",
        "",
        "Use `delta_bic_source_plus_diffuse_vs_diffuse_only` as the main model-comparison value.  Positive values mean",
        "the explicit source term improves the fit after penalizing the extra parameter.  Values above about 10 are strong",
        "model-comparison evidence for adding the source term; 2-10 is suggestive; below 2 is weak.",
        "",
        "This is still not a confirmed detection test by itself.  It says whether the observed stacked morphology is better",
        "represented by diffuse background plus an occultation-like residual term than by diffuse background alone.  For",
        "Fornax-A, broad transition durations should be treated as evidence for unresolved diffuse/beam/background structure",
        "unless off-source and sky-region controls show the same term is uniquely tied to the Fornax-A position.",
        "",
        "## Generated Plots",
        "",
    ]
    lines.extend(f"- `{path}`" for path in paths)
    (out_dir / "source_plus_diffuse_stack_model_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="fornax_a")
    parser.add_argument(
        "--summary",
        default=str(ROOT / "outputs/fornax_simulator_structure_audit_v1/fornax_a_same_window_observed_and_model_summary.csv"),
    )
    parser.add_argument("--diffuse-methods", default="model_eh_azimuth_yaw13,model_axisymmetric_mean")
    parser.add_argument("--primary-diffuse-method", default="model_eh_azimuth_yaw13")
    parser.add_argument("--timing-offsets", default="-180,-120,-60,-30,0,30,60,120,180")
    parser.add_argument("--transition-durations", default="0,120,240,360,600,900")
    parser.add_argument("--min-bins", type=int, default=10)
    parser.add_argument("--out-dir", default=str(ROOT / "outputs/source_plus_diffuse_stack_model_fornax_v1"))
    args = parser.parse_args()

    source = str(args.source).lower()
    out_dir = ensure_dir(args.out_dir)
    diffuse_methods = [x.strip() for x in str(args.diffuse_methods).split(",") if x.strip()]
    timing_offsets = [float(x.strip()) for x in str(args.timing_offsets).split(",") if x.strip()]
    transition_durations = [float(x.strip()) for x in str(args.transition_durations).split(",") if x.strip()]
    config = {
        "source": source,
        "summary_input": str(args.summary),
        "diffuse_methods": diffuse_methods,
        "primary_diffuse_method": str(args.primary_diffuse_method),
        "timing_offsets": timing_offsets,
        "transition_durations": transition_durations,
        "min_bins": int(args.min_bins),
        "software_versions": software_versions(),
    }
    write_json(out_dir / "run_config.json", config)

    summary = _read(Path(args.summary))
    if summary.empty:
        raise SystemExit(f"empty or missing summary table: {args.summary}")
    fits, curves = fit_source_plus_diffuse(
        summary,
        source=source,
        diffuse_methods=diffuse_methods,
        timing_offsets=timing_offsets,
        transition_durations=transition_durations,
        min_bins=int(args.min_bins),
    )
    fits.to_csv(out_dir / f"{source}_source_plus_diffuse_fit_summary.csv", index=False)
    curves.to_csv(out_dir / f"{source}_source_plus_diffuse_model_curves.csv", index=False)

    paths = []
    primary = str(args.primary_diffuse_method)
    paths.append(plot_heatmaps(fits, source, primary, out_dir))
    for antenna in ["rv2_coarse", "rv1_coarse"]:
        if antenna in set(curves["antenna"].astype(str)):
            paths.append(plot_profile_grid(curves, fits, source, primary, antenna, out_dir))
    write_report(out_dir, source, primary, fits, paths, config)
    print(out_dir / "source_plus_diffuse_stack_model_report.md")


if __name__ == "__main__":
    main()
