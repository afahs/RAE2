#!/usr/bin/env python
"""Try lower-V Earth/Sun recovery using far-sideband background residuals and regimes.

The goal is visual extraction, not a new SNR score.  This script uses the
existing lower-V moving-body event samples and tests three physically motivated
views:

1. raw fractional profiles;
2. per-event far-sideband linear residual profiles;
3. the same residual profiles split by background-regime selections.

The line fit is trained only on |t| >= 300 s, so the near occultation bins do
not define the background model.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rylevonberg.table_io import read_table
from astropy import units as u
from astropy.coordinates import FK4, SkyCoord
from astropy.time import Time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.util import ensure_dir, robust_sigma, software_versions, write_json  # noqa: E402


POINTS = ROOT / "outputs/moving_body_stack_type_subset_tests_v1/moving_body_stack_points.csv"
METRICS = ROOT / "outputs/moving_body_regime_physical_differences_v1/moving_body_event_regime_geometry_table.csv"
OUT = ROOT / "outputs/lowfreq_regime_recovery_profiles_v1"

LOW_FREQS = [0.45, 0.70, 0.90, 1.31, 2.20]
FOCUS_SOURCES = ["earth", "sun"]
FAR_SIDEBAND_S = 300.0
PRE = (-180.0, -60.0)
POST = (60.0, 180.0)
EXPECTED_SIGN = {"disappearance": -1.0, "reappearance": 1.0}
COLORS = {
    "raw_fractional_all": "0.35",
    "far_line_residual_all": "#0072B2",
    "far_line_residual_low_abs_far_slope": "#009E73",
    "far_line_residual_favorable_far_slope": "#D55E00",
    "far_line_residual_high_abs_gal_b": "#CC79A7",
    "far_line_residual_very_high_abs_gal_b": "#F0E442",
}


def _read(path: Path, **kwargs) -> pd.DataFrame:
    return read_table(path, low_memory=False, **kwargs)


def _robust_sem(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size <= 1:
        return np.nan
    scale = robust_sigma(vals - np.nanmedian(vals))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.nanstd(vals, ddof=1))
    return float(scale / np.sqrt(vals.size)) if np.isfinite(scale) and scale > 0 else np.nan


def _fit_far_line(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, bool]:
    good = np.isfinite(t) & np.isfinite(y) & (np.abs(t) >= FAR_SIDEBAND_S)
    model = np.full_like(y, np.nan, dtype=float)
    if np.count_nonzero(good) < 6:
        return model, False
    x = t[good] / 900.0
    coeff = np.polyfit(x, y[good], deg=1)
    model[:] = np.polyval(coeff, t / 900.0)
    return model, True


def build_residual_points(points: pd.DataFrame, metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    points = points[
        points["source_name"].isin(FOCUS_SOURCES)
        & points["frequency_mhz"].isin(LOW_FREQS)
    ].copy()
    metrics = metrics[
        metrics["source_name"].isin(FOCUS_SOURCES)
        & metrics["frequency_mhz"].isin(LOW_FREQS)
    ].copy()

    meta_cols = [
        "source_name",
        "event_id",
        "event_type",
        "frequency_band",
        "frequency_mhz",
        "far_raw_slope_per_s",
        "near_raw_slope_per_s",
        "pre_far_median",
        "pre_sigma",
        "limb_rate_deg_s",
        "gap_seconds",
        "limb_position_angle_deg",
        "target_ra_deg",
        "target_dec_deg",
        "month",
    ]
    meta = metrics[meta_cols].drop_duplicates(["source_name", "event_id", "event_type", "frequency_band"])
    finite_coord = np.isfinite(meta["target_ra_deg"]) & np.isfinite(meta["target_dec_deg"])
    meta["target_galactic_b_deg"] = np.nan
    if finite_coord.any():
        coord = SkyCoord(
            ra=meta.loc[finite_coord, "target_ra_deg"].to_numpy(dtype=float) * u.deg,
            dec=meta.loc[finite_coord, "target_dec_deg"].to_numpy(dtype=float) * u.deg,
            frame=FK4(equinox=Time("B1950")),
        )
        meta.loc[finite_coord, "target_galactic_b_deg"] = np.asarray(coord.galactic.b.deg, dtype=float)
    points = points.merge(meta, on=["source_name", "event_id", "event_type", "frequency_band", "frequency_mhz", "month"], how="left")

    out_parts = []
    fit_rows = []
    group_cols = ["source_name", "event_id", "event_type", "frequency_band", "frequency_mhz"]
    for keys, grp in points.groupby(group_cols, sort=True, dropna=False):
        g = grp.sort_values("t_rel_sec").copy()
        t = g["t_rel_sec"].to_numpy(dtype=float)
        y = g["raw_fractional"].to_numpy(dtype=float)
        model, ok = _fit_far_line(t, y)
        g["far_line_model"] = model
        g["far_line_residual"] = y - model
        # Keep a constant-centered residual for plotting around zero.
        if np.isfinite(g["far_line_residual"]).any():
            center = float(np.nanmedian(g.loc[np.abs(g["t_rel_sec"]) >= FAR_SIDEBAND_S, "far_line_residual"]))
            g["far_line_residual"] = g["far_line_residual"] - center
        out_parts.append(g)
        fit_rows.append(
            {
                **dict(zip(group_cols, keys)),
                "fit_ok": bool(ok),
                "n_points": int(len(g)),
                "n_far_sideband_points": int(np.count_nonzero(np.isfinite(y) & (np.abs(t) >= FAR_SIDEBAND_S))),
            }
        )
    residual = pd.concat(out_parts, ignore_index=True) if out_parts else pd.DataFrame()
    fit_status = pd.DataFrame(fit_rows)
    return residual, fit_status


def add_regime_labels(points: pd.DataFrame) -> pd.DataFrame:
    out = points.copy()
    out["method"] = "raw_fractional_all"
    residual_all = out.copy()
    residual_all["method"] = "far_line_residual_all"
    residual_all["profile_value"] = residual_all["far_line_residual"]

    raw_all = out.copy()
    raw_all["profile_value"] = raw_all["raw_fractional"]

    pieces = [raw_all, residual_all]
    # Low absolute far slope: least background-dominated third within source/frequency/event type.
    low_slope = residual_all.copy()
    low_slope["abs_far_slope"] = low_slope["far_raw_slope_per_s"].abs()
    keep = []
    for _, grp in low_slope.groupby(["source_name", "frequency_mhz", "event_type"], sort=True):
        q = np.nanquantile(grp["abs_far_slope"], 1 / 3) if grp["abs_far_slope"].notna().any() else np.nan
        keep.append(grp[grp["abs_far_slope"].le(q)])
    if keep:
        ls = pd.concat(keep, ignore_index=True)
        ls["method"] = "far_line_residual_low_abs_far_slope"
        pieces.append(ls)

    # Favorable far slope: background far-side trend has the same sign as a positive occultation.
    fav = residual_all.copy()
    sign = fav["event_type"].map(EXPECTED_SIGN).astype(float)
    fav = fav[(sign * fav["far_raw_slope_per_s"]).ge(0)].copy()
    fav["method"] = "far_line_residual_favorable_far_slope"
    pieces.append(fav)

    high_b = residual_all[residual_all["target_galactic_b_deg"].abs().ge(30.0)].copy()
    high_b["method"] = "far_line_residual_high_abs_gal_b"
    pieces.append(high_b)

    very_high_b = residual_all[residual_all["target_galactic_b_deg"].abs().ge(45.0)].copy()
    very_high_b["method"] = "far_line_residual_very_high_abs_gal_b"
    pieces.append(very_high_b)

    return pd.concat(pieces, ignore_index=True, sort=False)


def summarize(points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["source_name", "method", "frequency_mhz", "event_type", "t_bin_sec"]
    for vals_key, grp in points.groupby(keys, sort=True, dropna=False):
        vals = pd.to_numeric(grp["profile_value"], errors="coerce").dropna()
        if vals.empty:
            continue
        rows.append(
            {
                **dict(zip(keys, vals_key)),
                "median_profile": float(vals.median()),
                "profile_err": _robust_sem(vals),
                "n_events": int(grp["event_id"].nunique()),
                "n_points": int(len(vals)),
            }
        )
    return pd.DataFrame(rows)


def contrast(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["source_name", "method", "frequency_mhz", "event_type"]
    for vals_key, grp in summary.groupby(keys, sort=True, dropna=False):
        pre = grp[(grp["t_bin_sec"] >= PRE[0]) & (grp["t_bin_sec"] <= PRE[1])]["median_profile"]
        post = grp[(grp["t_bin_sec"] >= POST[0]) & (grp["t_bin_sec"] <= POST[1])]["median_profile"]
        if pre.empty or post.empty:
            continue
        event_type = str(vals_key[-1])
        delta = float(np.nanmedian(post) - np.nanmedian(pre))
        rows.append(
            {
                **dict(zip(keys, vals_key)),
                "n_events": int(np.nanmedian(grp["n_events"])),
                "post_minus_pre": delta,
                "source_like_contrast": float(EXPECTED_SIGN[event_type] * delta),
            }
        )
    return pd.DataFrame(rows)


def plot_source_grid(summary: pd.DataFrame, source: str, out_dir: Path) -> Path:
    methods = [
        "raw_fractional_all",
        "far_line_residual_all",
        "far_line_residual_low_abs_far_slope",
        "far_line_residual_favorable_far_slope",
        "far_line_residual_high_abs_gal_b",
        "far_line_residual_very_high_abs_gal_b",
    ]
    freqs = [f for f in LOW_FREQS if f in set(summary["frequency_mhz"])]
    fig, axes = plt.subplots(len(freqs), 2, figsize=(13, max(10, 1.55 * len(freqs))), sharex=True)
    if len(freqs) == 1:
        axes = np.asarray([axes])
    for i, freq in enumerate(freqs):
        for j, event_type in enumerate(["disappearance", "reappearance"]):
            ax = axes[i, j]
            for method in methods:
                sub = summary[
                    summary["source_name"].eq(source)
                    & np.isclose(summary["frequency_mhz"], freq)
                    & summary["event_type"].eq(event_type)
                    & summary["method"].eq(method)
                ].sort_values("t_bin_sec")
                if sub.empty:
                    continue
                ax.errorbar(
                    sub["t_bin_sec"],
                    sub["median_profile"],
                    yerr=sub["profile_err"],
                    color=COLORS.get(method),
                    ecolor=COLORS.get(method),
                    marker="o",
                    markersize=2.0,
                    linewidth=1.2,
                    elinewidth=0.45,
                    capsize=0.9,
                    alpha=0.9,
                    label=method.replace("_", " "),
                )
            ax.axvline(0, color="black", linestyle="--", linewidth=0.7)
            ax.axhline(0, color="0.65", linewidth=0.7)
            ax.set_title(f"{freq:.2f} MHz {event_type}", fontsize=9)
            if j == 0:
                ax.set_ylabel("profile")
            if i == len(freqs) - 1:
                ax.set_xlabel("seconds from predicted event")
            if i == 0 and j == 1:
                ax.legend(frameon=False, fontsize=7)
    fig.suptitle(
        f"{source.title()} lower-V recovery profiles: raw vs far-sideband residual and background-regime subsets",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = out_dir / f"{source}_lowfreq_regime_recovery_profile_grid.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report(out_dir: Path, fit_status: pd.DataFrame, contrasts: pd.DataFrame, paths: list[Path]) -> None:
    lines = [
        "# Low-Frequency Regime Recovery Profiles",
        "",
        "## Method",
        "",
        "This tries to recover Earth/Sun lower-V low-frequency occultation morphology by removing broad event-local",
        "background trends with a line fit only to far sidebands, then by stacking physically motivated event subsets.",
        "",
        "Subset definitions:",
        "",
        "- `raw_fractional_all`: original per-event fractional profile;",
        "- `far_line_residual_all`: raw fractional profile minus a line fit using only |t| >= 300 s;",
        "- `far_line_residual_low_abs_far_slope`: same residual, but only the lowest third of absolute far-window slopes;",
        "- `far_line_residual_favorable_far_slope`: same residual, but only events where the far-window slope is not fighting the expected source sign.",
        "- `far_line_residual_high_abs_gal_b`: same residual, but only events where the target apparent |Galactic b| >= 30 deg.",
        "- `far_line_residual_very_high_abs_gal_b`: same residual, but only events where the target apparent |Galactic b| >= 45 deg.",
        "",
        "These subsets are based on broad-window background behavior, not on the near-event measured occultation sign.",
        "",
        "## Fit Counts",
        "",
        f"Event profiles attempted: {len(fit_status)}",
        f"Successful far-sideband line fits: {int(fit_status['fit_ok'].sum()) if not fit_status.empty else 0}",
        "",
        "## Source-Like Contrast Table",
        "",
        contrasts[
            [
                "source_name",
                "method",
                "frequency_mhz",
                "event_type",
                "n_events",
                "source_like_contrast",
            ]
        ].sort_values(["source_name", "frequency_mhz", "event_type", "method"]).to_string(index=False),
        "",
        "## Initial Interpretation",
        "",
        "- A successful recovery should visually resemble Earth 0.45 MHz: disappearance down across t=0 and reappearance up across t=0.",
        "- If low-absolute-slope subsets improve a bin, the previous failure is likely background-gradient dominated.",
        "- If favorable-slope subsets improve a bin but all-events do not, mixed event regimes are diluting or reversing the stack.",
        "- If neither subset improves the morphology, the relevant contaminating term is not captured by a scalar broad-window slope.",
        "",
        "## Plots",
        "",
    ]
    lines.extend(f"- `{path}`" for path in paths)
    (out_dir / "lowfreq_regime_recovery_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = ensure_dir(OUT)
    write_json(
        out_dir / "run_config.json",
        {
            "points": str(POINTS),
            "metrics": str(METRICS),
            "low_freqs": LOW_FREQS,
            "far_sideband_s": FAR_SIDEBAND_S,
            "software_versions": software_versions(),
        },
    )
    points = _read(POINTS)
    metrics = _read(METRICS, parse_dates=["predicted_event_time"])
    residual, fit_status = build_residual_points(points, metrics)
    method_points = add_regime_labels(residual)
    summary = summarize(method_points)
    contrasts = contrast(summary)

    residual.to_csv(out_dir / "event_far_line_residual_points.csv", index=False)
    fit_status.to_csv(out_dir / "far_line_fit_status.csv", index=False)
    method_points.to_csv(out_dir / "method_subset_profile_points.csv", index=False)
    summary.to_csv(out_dir / "method_subset_profile_summary.csv", index=False)
    contrasts.to_csv(out_dir / "method_subset_prepost_contrasts.csv", index=False)
    paths = [plot_source_grid(summary, source, out_dir) for source in FOCUS_SOURCES]
    write_report(out_dir, fit_status, contrasts, paths)
    print(out_dir / "lowfreq_regime_recovery_report.md")


if __name__ == "__main__":
    main()
