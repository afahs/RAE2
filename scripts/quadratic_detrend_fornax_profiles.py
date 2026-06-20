#!/usr/bin/env python
"""Polynomial whole-window detrending tests for occultation profile grids.

Two modes are intentionally separated:

1. pre_average_polynomial: fit/subtract a polynomial from each event profile
   before computing the stacked median profile.
2. post_average_polynomial: compute the ordinary stacked median profile first,
   then fit/subtract a polynomial from that averaged profile.

Both use the full +/-900 s window.  This is a diagnostic for whether the
straight-line-looking profile morphology is dominated by broad background
curvature rather than a local occultation transition.
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

from rylevonberg.util import ensure_dir, robust_sigma, software_versions, write_json  # noqa: E402


DEFAULT_SOURCE = "fornax_a"
PROFILE_DIR = ROOT / "outputs/all_frequency_profile_grids_v1"
POINTS = PROFILE_DIR / f"{DEFAULT_SOURCE}_all_frequency_profile_points_900s.csv"
SUMMARY = PROFILE_DIR / f"{DEFAULT_SOURCE}_all_frequency_profile_summary_900s.csv"
DEFAULT_OUT = ROOT / "outputs/fornax_quadratic_detrend_profiles_v1"

WINDOW_S = 900.0
MIN_BINS_FOR_FIT = 6
EXPECTED_SIGN = {"disappearance": -1.0, "reappearance": 1.0}
ANT_LABEL = {"rv1_coarse": "upper V", "rv2_coarse": "lower V"}
ANT_COLOR = {"rv1_coarse": "#4c78a8", "rv2_coarse": "#d95f02"}
SOURCE_LABEL = {
    "earth": "Earth",
    "sun": "Sun",
    "fornax_a": "Fornax A",
    "cyg_a": "Cyg A",
    "cas_a": "Cas A",
    "jupiter": "Jupiter",
}


def _read(path: Path, **kwargs) -> pd.DataFrame:
    return read_table(path, low_memory=False, **kwargs)


def _robust_se(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size <= 1:
        return np.nan
    scale = robust_sigma(vals - np.nanmedian(vals))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.nanstd(vals, ddof=1))
    if not np.isfinite(scale) or scale <= 0:
        return np.nan
    return float(scale / np.sqrt(vals.size))


def _degree_label(degree: int) -> str:
    return {1: "linear", 2: "quadratic"}.get(int(degree), f"degree{int(degree)}")


def _source_label(source: str) -> str:
    return SOURCE_LABEL.get(source.lower(), source.replace("_", " ").title())


def _poly_residual(t: np.ndarray, y: np.ndarray, degree: int) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(t) & np.isfinite(y)
    residual = np.full_like(y, np.nan, dtype=float)
    model = np.full_like(y, np.nan, dtype=float)
    if np.count_nonzero(good) < MIN_BINS_FOR_FIT:
        return residual, model
    x = t[good] / WINDOW_S
    coeff = np.polyfit(x, y[good], deg=int(degree))
    model_good = np.polyval(coeff, x)
    residual[good] = y[good] - model_good
    model[good] = model_good
    return residual, model


def pre_average_polynomial(points: pd.DataFrame, degree: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Subtract event-level polynomial fits before stacking."""
    by = ["source_name", "event_id", "event_type", "frequency_band", "frequency_mhz", "antenna"]
    residual_parts = []
    fit_rows = []
    for keys, grp in points.groupby(by, sort=True, dropna=False):
        g = grp.sort_values("t_bin_sec").copy()
        residual, model = _poly_residual(
            g["t_bin_sec"].to_numpy(dtype=float), g["z_power"].to_numpy(dtype=float), degree
        )
        g["polynomial_model_z_power"] = model
        g["polynomial_residual_z_power"] = residual
        residual_parts.append(g)
        fit_rows.append(
            {
                **dict(zip(by, keys)),
                "n_bins": int(np.isfinite(g["z_power"]).sum()),
                "fit_ok": bool(np.isfinite(residual).sum() >= MIN_BINS_FOR_FIT),
                "residual_rms": float(np.nanstd(residual)) if np.isfinite(residual).any() else np.nan,
            }
        )
    residual_points = pd.concat(residual_parts, ignore_index=True) if residual_parts else pd.DataFrame()
    fit_status = pd.DataFrame(fit_rows)
    return residual_points, fit_status


def summarize_residual_points(points: pd.DataFrame, value_col: str, method: str) -> pd.DataFrame:
    rows = []
    by = ["source_name", "event_type", "frequency_band", "frequency_mhz", "antenna", "t_bin_sec"]
    for keys, grp in points.groupby(by, sort=True, dropna=False):
        vals = pd.to_numeric(grp[value_col], errors="coerce").dropna()
        if vals.empty:
            continue
        rows.append(
            {
                **dict(zip(by, keys)),
                "method": method,
                "median_residual_z_power": float(vals.median()),
                "residual_z_power_err": _robust_se(vals),
                "n_events": int(grp["event_id"].nunique()) if "event_id" in grp else np.nan,
                "n_points": int(len(vals)),
            }
        )
    return pd.DataFrame(rows)


def post_average_polynomial(summary: pd.DataFrame, degree: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Subtract polynomial fits from already-stacked median profiles."""
    by = ["source_name", "event_type", "frequency_band", "frequency_mhz", "antenna"]
    parts = []
    fit_rows = []
    for keys, grp in summary.groupby(by, sort=True, dropna=False):
        g = grp.sort_values("t_bin_sec").copy()
        residual, model = _poly_residual(
            g["t_bin_sec"].to_numpy(dtype=float), g["median_z_power"].to_numpy(dtype=float), degree
        )
        label = _degree_label(degree)
        g["method"] = f"post_average_{label}"
        g["polynomial_model_z_power"] = model
        g["median_residual_z_power"] = residual
        g["residual_z_power_err"] = g["median_z_power_err"]
        parts.append(
            g[
                [
                    "source_name",
                    "event_type",
                    "frequency_band",
                    "frequency_mhz",
                    "antenna",
                    "t_bin_sec",
                    "method",
                    "median_residual_z_power",
                    "residual_z_power_err",
                    "n_events",
                    "n_points",
                    "polynomial_model_z_power",
                    "median_z_power",
                ]
            ]
        )
        fit_rows.append(
            {
                **dict(zip(by, keys)),
                "n_bins": int(np.isfinite(g["median_z_power"]).sum()),
                "fit_ok": bool(np.isfinite(residual).sum() >= MIN_BINS_FOR_FIT),
                "residual_rms": float(np.nanstd(residual)) if np.isfinite(residual).any() else np.nan,
            }
        )
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(), pd.DataFrame(fit_rows)


def prepost_contrast(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pre = (-180.0, -60.0)
    post = (60.0, 180.0)
    by = ["method", "frequency_band", "frequency_mhz", "antenna", "event_type"]
    for keys, grp in summary.groupby(by, sort=True, dropna=False):
        vals_pre = grp[(grp["t_bin_sec"] >= pre[0]) & (grp["t_bin_sec"] <= pre[1])]["median_residual_z_power"]
        vals_post = grp[(grp["t_bin_sec"] >= post[0]) & (grp["t_bin_sec"] <= post[1])]["median_residual_z_power"]
        if vals_pre.empty or vals_post.empty:
            continue
        event_type = str(keys[-1])
        delta = float(np.nanmedian(vals_post) - np.nanmedian(vals_pre))
        rows.append(
            {
                **dict(zip(by, keys)),
                "post_minus_pre_residual": delta,
                "source_like_residual_contrast": float(EXPECTED_SIGN[event_type] * delta),
            }
        )
    return pd.DataFrame(rows)


def plot_grid(summary: pd.DataFrame, method: str, out: Path, degree: int, source: str) -> Path:
    sub_all = summary[summary["method"].eq(method)].copy()
    freqs = sorted(sub_all["frequency_mhz"].dropna().unique())
    event_types = ["disappearance", "reappearance"]
    fig, axes = plt.subplots(len(freqs), 2, figsize=(12, max(10, 1.35 * len(freqs))), sharex=True)
    if len(freqs) == 1:
        axes = np.asarray([axes])
    for i, freq in enumerate(freqs):
        for j, event_type in enumerate(event_types):
            ax = axes[i, j]
            sub = sub_all[np.isclose(sub_all["frequency_mhz"], freq) & sub_all["event_type"].eq(event_type)]
            for antenna, grp in sub.groupby("antenna", sort=True):
                grp = grp.sort_values("t_bin_sec")
                ax.errorbar(
                    grp["t_bin_sec"],
                    grp["median_residual_z_power"],
                    yerr=grp["residual_z_power_err"],
                    color=ANT_COLOR.get(antenna),
                    ecolor=ANT_COLOR.get(antenna),
                    marker="o",
                    markersize=2.4,
                    linewidth=1.15,
                    elinewidth=0.55,
                    capsize=1.1,
                    alpha=0.9,
                    label=ANT_LABEL.get(antenna, antenna),
                )
            ax.axvline(0, color="black", linestyle="--", linewidth=0.75)
            ax.axhline(0, color="0.65", linewidth=0.7)
            ax.set_title(f"{freq:.2f} MHz {event_type}", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{_degree_label(degree)} residual")
            if i == len(freqs) - 1:
                ax.set_xlabel("seconds from predicted event")
            if i == 0 and j == 1:
                ax.legend(frameon=False, fontsize=8)
    title = {
        f"pre_average_{_degree_label(degree)}": f"{_source_label(source)}: event-level {_degree_label(degree)} removed before stacking",
        f"post_average_{_degree_label(degree)}": f"{_source_label(source)}: {_degree_label(degree)} removed after stacking",
        "original": f"{_source_label(source)}: original normalized profile",
    }.get(method, method)
    fig.suptitle(f"{title}\nWhole-window {_degree_label(degree)} fit over +/-{WINDOW_S:.0f} s", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = out / f"{source}_{method}_all_frequency_profile_grid.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_method_comparison(original: pd.DataFrame, detrended: pd.DataFrame, out: Path, degree: int, source: str) -> Path:
    rows = []
    orig = original.copy()
    orig["method"] = "original"
    orig = orig.rename(columns={"median_z_power": "median_residual_z_power", "median_z_power_err": "residual_z_power_err"})
    rows.append(orig[detrended.columns.intersection(orig.columns).tolist()])
    rows.append(detrended)
    combined = pd.concat(rows, ignore_index=True, sort=False)
    freqs = [0.70, 0.90, 1.31, 2.20, 3.93, 6.55]
    label = _degree_label(degree)
    methods = ["original", f"pre_average_{label}", f"post_average_{label}"]
    colors = {"original": "0.35", f"pre_average_{label}": "#009E73", f"post_average_{label}": "#CC79A7"}
    fig, axes = plt.subplots(len(freqs), 2, figsize=(12, 1.55 * len(freqs) + 2), sharex=True)
    for i, freq in enumerate(freqs):
        for j, event_type in enumerate(["disappearance", "reappearance"]):
            ax = axes[i, j]
            for method in methods:
                sub = combined[
                    np.isclose(combined["frequency_mhz"], freq)
                    & combined["event_type"].eq(event_type)
                    & combined["antenna"].eq("rv2_coarse")
                    & combined["method"].eq(method)
                ].sort_values("t_bin_sec")
                if sub.empty:
                    continue
                ax.plot(
                    sub["t_bin_sec"],
                    sub["median_residual_z_power"],
                    color=colors[method],
                    linewidth=1.45,
                    marker="o" if method == "original" else None,
                    markersize=2.0,
                    alpha=0.88,
                    label=method.replace("_", " "),
                )
            ax.axvline(0, color="black", linestyle="--", linewidth=0.7)
            ax.axhline(0, color="0.65", linewidth=0.7)
            ax.set_title(f"lower V {freq:.2f} MHz {event_type}", fontsize=9)
            if j == 0:
                ax.set_ylabel("profile value")
            if i == len(freqs) - 1:
                ax.set_xlabel("seconds from predicted event")
            if i == 0 and j == 1:
                ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"{_source_label(source)} lower-V comparison: original vs {_degree_label(degree)} residual profiles", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    path = out / f"{source}_lower_v_{label}_method_comparison.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report(
    fit_status_pre: pd.DataFrame,
    fit_status_post: pd.DataFrame,
    contrast: pd.DataFrame,
    paths: list[Path],
    out: Path,
    degree: int,
    source: str,
) -> None:
    low = contrast[contrast["frequency_mhz"].isin([0.70, 0.90, 1.31, 2.20])].copy()
    label = _degree_label(degree)
    lines = [
        f"# {_source_label(source)} {label.title()} Detrend Profile Test",
        "",
        "## Purpose",
        "",
        f"This tests whether broad {label} background structure across the full +/-900 s event window is responsible for",
        f"the apparently linear {_source_label(source)} all-frequency profiles.",
        "",
        "Two versions were run:",
        "",
        f"- `pre_average_{label}`: fit/subtract a {label} model from each event's binned profile before stacking;",
        f"- `post_average_{label}`: stack first, then fit/subtract a {label} model from the stacked median profile.",
        "",
        "The first version is stricter because each event contributes only its residual. The second version asks whether the",
        "already-averaged profile has a smooth low-order component that can be removed.",
        "",
        "## Fit Counts",
        "",
        f"Polynomial degree: {degree}",
        f"Pre-average event profiles attempted: {len(fit_status_pre)}",
        f"Pre-average successful fits: {int(fit_status_pre['fit_ok'].sum()) if not fit_status_pre.empty else 0}",
        f"Post-average profiles attempted: {len(fit_status_post)}",
        f"Post-average successful fits: {int(fit_status_post['fit_ok'].sum()) if not fit_status_post.empty else 0}",
        "",
        "## Low-Frequency Residual Pre/Post Contrasts",
        "",
        low[
            [
                "method",
                "frequency_mhz",
                "antenna",
                "event_type",
                "post_minus_pre_residual",
                "source_like_residual_contrast",
            ]
        ].to_string(index=False)
        if not low.empty
        else "No low-frequency contrast rows.",
        "",
        "## How To Interpret",
        "",
        f"If the apparent source-like/anti-template profile disappears after `pre_average_{label}`, then the morphology is",
        "mostly event-local background curvature and should not be treated as an occultation signature.",
        "",
        f"If it survives `pre_average_{label}` but weakens after `post_average_{label}`, the signal is partly encoded in the",
        "stacked shape but is entangled with smooth average background structure.",
        "",
        f"If it survives both, the residual structure is not just a whole-window {label} trend.",
        "",
        "## Generated Plots",
        "",
    ]
    lines.extend(f"- `{path}`" for path in paths)
    (out / f"{source}_{label}_detrend_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--points", default=None)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--degree", type=int, default=2, choices=[1, 2])
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    source = str(args.source).lower()
    label = _degree_label(args.degree)
    points_path = Path(args.points) if args.points else PROFILE_DIR / f"{source}_all_frequency_profile_points_900s.csv"
    summary_path = Path(args.summary) if args.summary else PROFILE_DIR / f"{source}_all_frequency_profile_summary_900s.csv"
    default_out = DEFAULT_OUT if source == DEFAULT_SOURCE and args.degree == 2 else ROOT / f"outputs/{source}_{label}_detrend_profiles_v1"
    out = ensure_dir(Path(args.out_dir) if args.out_dir else default_out)
    write_json(
        out / "run_config.json",
        {
            "source": source,
            "points": str(points_path),
            "summary": str(summary_path),
            "window_s": WINDOW_S,
            "polynomial_degree": int(args.degree),
            "min_bins_for_fit": MIN_BINS_FOR_FIT,
            "software_versions": software_versions(),
        },
    )
    points = _read(points_path)
    original = _read(summary_path)
    points = points[points["source_name"].astype(str).str.lower().eq(source)].copy()
    original = original[original["source_name"].astype(str).str.lower().eq(source)].copy()

    residual_points, fit_status_pre = pre_average_polynomial(points, args.degree)
    pre_summary = summarize_residual_points(residual_points, "polynomial_residual_z_power", f"pre_average_{label}")
    post_summary, fit_status_post = post_average_polynomial(original, args.degree)
    detrended = pd.concat([pre_summary, post_summary], ignore_index=True, sort=False)
    contrast = prepost_contrast(detrended)

    residual_points.to_csv(out / f"{source}_pre_average_{label}_residual_points.csv", index=False)
    pre_summary.to_csv(out / f"{source}_pre_average_{label}_summary.csv", index=False)
    post_summary.to_csv(out / f"{source}_post_average_{label}_summary.csv", index=False)
    fit_status_pre.to_csv(out / f"{source}_pre_average_{label}_fit_status.csv", index=False)
    fit_status_post.to_csv(out / f"{source}_post_average_{label}_fit_status.csv", index=False)
    contrast.to_csv(out / f"{source}_{label}_residual_prepost_contrast.csv", index=False)

    paths = [
        plot_grid(pre_summary, f"pre_average_{label}", out, args.degree, source),
        plot_grid(post_summary, f"post_average_{label}", out, args.degree, source),
        plot_method_comparison(original, detrended, out, args.degree, source),
    ]
    write_report(fit_status_pre, fit_status_post, contrast, paths, out, args.degree, source)
    print(out / f"{source}_{label}_detrend_report.md")


if __name__ == "__main__":
    main()
