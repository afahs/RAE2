#!/usr/bin/env python
"""Lower-V profile grids after upper-V common-mode subtraction.

This is a diagnostic background-removal experiment.  The lower V points are
treated as the target channel, while the simultaneous upper V points from the
same event/frequency/time bin are used as a monitor for scan-synchronous
background structure.  For each event/frequency/event type, estimate a scalar
alpha from sideband bins:

    lower(t) ~= alpha * upper(t) + residual(t)

using only |t| >= sideband_s seconds.  Then stack residual(t).

If the low-frequency reversal is mainly a broad common-mode background scanned
through both antennas, this should make lower-V profiles more source-like.  If
it does not, the contamination is not well represented by a single simultaneous
upper-V monitor.
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


PROFILE_DIR = ROOT / "outputs/all_frequency_profile_grids_v1"
ANT_COLOR = {"original_lower_v": "0.35", "upper_v_common_mode_subtracted": "#009E73"}
EXPECTED_SIGN = {"disappearance": -1.0, "reappearance": 1.0}


def _read(path: Path) -> pd.DataFrame:
    return read_table(path, low_memory=False)


def _robust_sem(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size <= 1:
        return np.nan
    scale = robust_sigma(vals - np.nanmedian(vals))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.nanstd(vals, ddof=1))
    return float(scale / np.sqrt(vals.size)) if np.isfinite(scale) and scale > 0 else np.nan


def _alpha_from_sidebands(lower: np.ndarray, upper: np.ndarray) -> float:
    good = np.isfinite(lower) & np.isfinite(upper)
    if np.count_nonzero(good) < 4:
        return np.nan
    x = upper[good]
    y = lower[good]
    denom = float(np.nansum(x * x))
    if not np.isfinite(denom) or denom <= 0:
        return np.nan
    alpha = float(np.nansum(x * y) / denom)
    return alpha if np.isfinite(alpha) else np.nan


def common_mode_correct(points: pd.DataFrame, sideband_s: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    key = ["source_name", "event_id", "event_type", "frequency_band", "frequency_mhz", "t_bin_sec"]
    low = points[points["antenna"].eq("rv2_coarse")].copy()
    up = points[points["antenna"].eq("rv1_coarse")].copy()
    merged = low.merge(
        up[key + ["z_power"]].rename(columns={"z_power": "upper_z_power"}),
        on=key,
        how="inner",
    ).rename(columns={"z_power": "lower_z_power"})

    corrected_parts = []
    alpha_rows = []
    group_cols = ["source_name", "event_id", "event_type", "frequency_band", "frequency_mhz"]
    for keys, grp in merged.groupby(group_cols, sort=True, dropna=False):
        g = grp.sort_values("t_bin_sec").copy()
        side = np.abs(g["t_bin_sec"].to_numpy(dtype=float)) >= float(sideband_s)
        alpha = _alpha_from_sidebands(
            g.loc[side, "lower_z_power"].to_numpy(dtype=float),
            g.loc[side, "upper_z_power"].to_numpy(dtype=float),
        )
        if not np.isfinite(alpha):
            continue
        g["alpha_upper_to_lower"] = alpha
        g["corrected_z_power"] = g["lower_z_power"] - alpha * g["upper_z_power"]
        corrected_parts.append(g)
        alpha_rows.append(
            {
                **dict(zip(group_cols, keys)),
                "alpha_upper_to_lower": alpha,
                "n_sideband_bins": int(np.count_nonzero(side)),
            }
        )
    corrected = pd.concat(corrected_parts, ignore_index=True) if corrected_parts else pd.DataFrame()
    return corrected, pd.DataFrame(alpha_rows)


def summarize(points: pd.DataFrame, value_col: str, method: str) -> pd.DataFrame:
    rows = []
    group_cols = ["source_name", "event_type", "frequency_band", "frequency_mhz", "t_bin_sec"]
    for keys, grp in points.groupby(group_cols, sort=True, dropna=False):
        vals = pd.to_numeric(grp[value_col], errors="coerce").dropna()
        if vals.empty:
            continue
        rows.append(
            {
                **dict(zip(group_cols, keys)),
                "method": method,
                "median_z_power": float(vals.median()),
                "median_z_power_err": _robust_sem(vals),
                "n_events": int(grp["event_id"].nunique()),
                "n_points": int(len(vals)),
            }
        )
    return pd.DataFrame(rows)


def prepost_contrast(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pre = (-180.0, -60.0)
    post = (60.0, 180.0)
    group_cols = ["source_name", "method", "frequency_band", "frequency_mhz", "event_type"]
    for keys, grp in summary.groupby(group_cols, sort=True, dropna=False):
        before = grp[(grp["t_bin_sec"] >= pre[0]) & (grp["t_bin_sec"] <= pre[1])]["median_z_power"]
        after = grp[(grp["t_bin_sec"] >= post[0]) & (grp["t_bin_sec"] <= post[1])]["median_z_power"]
        if before.empty or after.empty:
            continue
        event_type = str(keys[-1])
        delta = float(np.nanmedian(after) - np.nanmedian(before))
        rows.append(
            {
                **dict(zip(group_cols, keys)),
                "n_events": int(np.nanmedian(grp["n_events"])),
                "post_minus_pre": delta,
                "source_like_contrast": float(EXPECTED_SIGN[event_type] * delta),
            }
        )
    return pd.DataFrame(rows)


def plot_comparison(original: pd.DataFrame, corrected: pd.DataFrame, source: str, out_dir: Path) -> Path:
    combined = pd.concat([original, corrected], ignore_index=True, sort=False)
    freqs = sorted(combined["frequency_mhz"].dropna().unique())
    fig, axes = plt.subplots(len(freqs), 2, figsize=(12, max(10, 1.35 * len(freqs))), sharex=True)
    if len(freqs) == 1:
        axes = np.asarray([axes])
    for i, freq in enumerate(freqs):
        for j, event_type in enumerate(["disappearance", "reappearance"]):
            ax = axes[i, j]
            sub = combined[np.isclose(combined["frequency_mhz"], freq) & combined["event_type"].eq(event_type)]
            for method, grp in sub.groupby("method", sort=True):
                grp = grp.sort_values("t_bin_sec")
                ax.errorbar(
                    grp["t_bin_sec"],
                    grp["median_z_power"],
                    yerr=grp["median_z_power_err"],
                    marker="o",
                    markersize=2.2,
                    linewidth=1.1,
                    elinewidth=0.55,
                    capsize=1.0,
                    alpha=0.88,
                    color=ANT_COLOR.get(method, None),
                    ecolor=ANT_COLOR.get(method, None),
                    label=method.replace("_", " "),
                )
            ax.axvline(0, color="black", linestyle="--", linewidth=0.75)
            ax.axhline(0, color="0.65", linewidth=0.7)
            ax.set_title(f"{freq:.2f} MHz {event_type}", fontsize=9)
            if j == 0:
                ax.set_ylabel("normalized profile")
            if i == len(freqs) - 1:
                ax.set_xlabel("seconds from predicted event")
            if i == 0 and j == 1:
                ax.legend(frameon=False, fontsize=8)
    fig.suptitle(
        f"{source.title()}: lower V original vs upper-V common-mode-subtracted profiles",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = out_dir / f"{source}_lower_v_common_mode_comparison_grid.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report(out_dir: Path, sources: list[str], sideband_s: float, contrasts: pd.DataFrame, paths: list[Path]) -> None:
    lines = [
        "# Upper-V Common-Mode Subtraction",
        "",
        "## Method",
        "",
        "For each event/frequency/event type, lower V is corrected by subtracting alpha times the simultaneous upper-V profile.",
        "The scalar alpha is fit only in sideband bins so the central occultation bins do not determine the correction.",
        "",
        f"Sideband threshold: |t| >= {sideband_s:.0f} s.",
        f"Sources: {', '.join(sources)}",
        "",
        "## Low-Frequency Contrasts",
        "",
    ]
    low = contrasts[contrasts["frequency_mhz"].isin([0.45, 0.70, 0.90, 1.31, 2.20])].copy()
    if low.empty:
        lines.append("No low-frequency rows.")
    else:
        lines.append(
            low[
                [
                    "source_name",
                    "method",
                    "frequency_mhz",
                    "event_type",
                    "n_events",
                    "post_minus_pre",
                    "source_like_contrast",
                ]
            ].to_string(index=False)
        )
    lines.extend(["", "## Plots", ""])
    lines.extend(f"- `{path}`" for path in paths)
    (out_dir / "antenna_common_mode_profile_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", default="earth,sun")
    parser.add_argument("--out-dir", default=str(ROOT / "outputs/antenna_common_mode_profiles_v1"))
    parser.add_argument("--sideband-s", type=float, default=300.0)
    args = parser.parse_args()

    sources = [x.strip().lower() for x in str(args.sources).split(",") if x.strip()]
    out_dir = ensure_dir(args.out_dir)
    write_json(
        out_dir / "run_config.json",
        {
            "sources": sources,
            "sideband_s": float(args.sideband_s),
            "profile_dir": str(PROFILE_DIR),
            "software_versions": software_versions(),
        },
    )

    summaries = []
    contrasts = []
    paths = []
    for source in sources:
        points = _read(PROFILE_DIR / f"{source}_all_frequency_profile_points_900s.csv")
        points = points[points["source_name"].astype(str).str.lower().eq(source)].copy()
        corrected, alpha = common_mode_correct(points, args.sideband_s)
        corrected.to_csv(out_dir / f"{source}_lower_v_common_mode_corrected_points.csv", index=False)
        alpha.to_csv(out_dir / f"{source}_upper_to_lower_alpha_by_event.csv", index=False)
        original_lower = points[points["antenna"].eq("rv2_coarse")].copy()
        original_summary = summarize(original_lower, "z_power", "original_lower_v")
        corrected_summary = summarize(corrected, "corrected_z_power", "upper_v_common_mode_subtracted")
        original_summary.to_csv(out_dir / f"{source}_lower_v_original_summary.csv", index=False)
        corrected_summary.to_csv(out_dir / f"{source}_lower_v_common_mode_summary.csv", index=False)
        source_summary = pd.concat([original_summary, corrected_summary], ignore_index=True, sort=False)
        source_contrast = prepost_contrast(source_summary)
        source_contrast.to_csv(out_dir / f"{source}_lower_v_common_mode_prepost_contrast.csv", index=False)
        paths.append(plot_comparison(original_summary, corrected_summary, source, out_dir))
        summaries.append(source_summary)
        contrasts.append(source_contrast)

    combined_summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    combined_contrast = pd.concat(contrasts, ignore_index=True) if contrasts else pd.DataFrame()
    combined_summary.to_csv(out_dir / "lower_v_common_mode_summary_all_sources.csv", index=False)
    combined_contrast.to_csv(out_dir / "lower_v_common_mode_prepost_contrast_all_sources.csv", index=False)
    write_report(out_dir, sources, args.sideband_s, combined_contrast, paths)
    print(out_dir / "antenna_common_mode_profile_report.md")


if __name__ == "__main__":
    main()
