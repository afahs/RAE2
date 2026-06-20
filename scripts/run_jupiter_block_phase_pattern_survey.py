#!/usr/bin/env python
"""Block-resolved Jupiter phase-pattern survey.

This extends ``run_jupiter_phase_pattern_survey.py`` by asking whether any
Jupiter-like CML/Io pattern appears only during specific date blocks. That is
more appropriate for Jupiter than repeated occultation-event fitting because
Jovian radio emission is bursty and episodic.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.constants import FREQUENCY_MAP_MHZ  # noqa: E402
from rylevonberg.util import ensure_dir, software_versions, write_json  # noqa: E402
from scripts.run_jupiter_phase_pattern_survey import (  # noqa: E402
    ANTENNA_COLOR,
    ANTENNA_LABEL,
    DEFAULT_CLEAN,
    DEFAULT_EARTH_STATES,
    DEFAULT_JUPITER_STATES,
    _add_daily_channel_normalization,
    _annotate_spice_grid,
    _load_visibility_states,
    _merge_geometry,
    _read_clean,
)


DEFAULT_OUT = ROOT / "outputs/jupiter_block_phase_pattern_survey_v1"


def _phase_bins(samples: pd.DataFrame, phase_bin_deg: float) -> pd.DataFrame:
    out = samples.copy()
    out["cml_bin_deg"] = (np.floor(out["jupiter_cml_spice_deg"] / phase_bin_deg) * phase_bin_deg) + 0.5 * phase_bin_deg
    out["io_bin_deg"] = (np.floor(out["io_phase_spice_deg"] / phase_bin_deg) * phase_bin_deg) + 0.5 * phase_bin_deg
    return out


def summarize_blocks(
    samples: pd.DataFrame,
    block_freq: str,
    high_z: float,
    phase_bin_deg: float,
    min_bin_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = _phase_bins(samples, phase_bin_deg)
    work["block"] = work["time"].dt.to_period(block_freq).astype(str)
    work["high_power_tail"] = work["daily_z_log_power"] > float(high_z)
    visible = work[work["jupiter_visible_by_moon"]].copy()
    q25 = float(visible["maser_zarka_io_score"].quantile(0.25))
    q75 = float(visible["maser_zarka_io_score"].quantile(0.75))
    work["maser_score_q25_visible_global"] = q25
    work["maser_score_q75_visible_global"] = q75

    contrast_rows = []
    for (block, antenna, band, freq), grp in work.groupby(["block", "antenna", "frequency_band", "frequency_mhz"], sort=True):
        vis = grp[grp["jupiter_visible_by_moon"]]
        high = vis[vis["maser_zarka_io_score"] >= q75]
        low = vis[vis["maser_zarka_io_score"] <= q25]
        earth_occ_high = high[~high["earth_visible_by_moon"]]
        occ = grp[~grp["jupiter_visible_by_moon"]]
        occ_high = occ[occ["maser_zarka_io_score"] >= q75]
        contrast_rows.append(
            {
                "block": block,
                "antenna": antenna,
                "frequency_band": int(band),
                "frequency_mhz": float(freq),
                "visible_high_maser_score_n": int(len(high)),
                "visible_low_maser_score_n": int(len(low)),
                "visible_earth_occulted_high_maser_score_n": int(len(earth_occ_high)),
                "jupiter_occulted_high_maser_score_control_n": int(len(occ_high)),
                "visible_high_maser_score_high_frac": float(high["high_power_tail"].mean()) if len(high) else np.nan,
                "visible_low_maser_score_high_frac": float(low["high_power_tail"].mean()) if len(low) else np.nan,
                "visible_earth_occulted_high_maser_score_high_frac": (
                    float(earth_occ_high["high_power_tail"].mean()) if len(earth_occ_high) else np.nan
                ),
                "jupiter_occulted_high_maser_score_control_high_frac": (
                    float(occ_high["high_power_tail"].mean()) if len(occ_high) else np.nan
                ),
                "high_minus_low_high_frac": (
                    float(high["high_power_tail"].mean() - low["high_power_tail"].mean())
                    if len(high) and len(low)
                    else np.nan
                ),
                "earth_occulted_high_minus_low_high_frac": (
                    float(earth_occ_high["high_power_tail"].mean() - low["high_power_tail"].mean())
                    if len(earth_occ_high) and len(low)
                    else np.nan
                ),
                "maser_score_q25_visible_global": q25,
                "maser_score_q75_visible_global": q75,
            }
        )
    contrast = pd.DataFrame(contrast_rows)

    phase_rows = []
    group_cols = ["block", "antenna", "frequency_band", "frequency_mhz", "cml_bin_deg", "io_bin_deg"]
    for keys, grp in visible.groupby(group_cols, sort=True, observed=True):
        if len(grp) < int(min_bin_count):
            continue
        phase_rows.append(
            {
                **dict(zip(group_cols, keys)),
                "n_samples": int(len(grp)),
                "high_power_fraction": float(grp["high_power_tail"].mean()),
                "median_daily_z": float(grp["daily_z_log_power"].median()),
                "median_maser_score": float(grp["maser_zarka_io_score"].median()),
            }
        )
    phase = pd.DataFrame(phase_rows)
    corr_rows = []
    if not phase.empty:
        for (block, antenna, band, freq), grp in phase.groupby(["block", "antenna", "frequency_band", "frequency_mhz"], sort=True):
            valid = grp[["high_power_fraction", "median_daily_z", "median_maser_score"]].dropna()
            corr_rows.append(
                {
                    "block": block,
                    "antenna": antenna,
                    "frequency_band": int(band),
                    "frequency_mhz": float(freq),
                    "n_phase_bins": int(len(valid)),
                    "n_phase_samples": int(grp["n_samples"].sum()),
                    "spearman_high_frac_vs_maser_score": (
                        float(valid["high_power_fraction"].corr(valid["median_maser_score"], method="spearman"))
                        if len(valid) >= 20
                        else np.nan
                    ),
                    "spearman_median_z_vs_maser_score": (
                        float(valid["median_daily_z"].corr(valid["median_maser_score"], method="spearman"))
                        if len(valid) >= 20
                        else np.nan
                    ),
                }
            )
    corr = pd.DataFrame(corr_rows)
    return contrast, corr, phase


def _heatmap_table(df: pd.DataFrame, antenna: str, value_col: str) -> pd.DataFrame:
    sub = df[df["antenna"].astype(str).eq(antenna)].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["freq_label"] = sub["frequency_mhz"].map(lambda x: f"{float(x):.2f}")
    return sub.pivot_table(index="freq_label", columns="block", values=value_col, aggfunc="median").sort_index(
        key=lambda idx: idx.astype(float)
    )


def plot_block_heatmaps(contrast: pd.DataFrame, corr: pd.DataFrame, out_dir: Path) -> list[Path]:
    paths = []
    specs = [
        (
            contrast,
            "high_minus_low_high_frac",
            "High MASER-score minus low-score high-tail fraction",
            "jupiter_block_high_minus_low_heatmap",
            "coolwarm",
            -0.08,
            0.08,
        ),
        (
            contrast,
            "earth_occulted_high_minus_low_high_frac",
            "Same, restricted to Jupiter-visible / Earth-occulted high-score samples",
            "jupiter_block_earth_occulted_high_minus_low_heatmap",
            "coolwarm",
            -0.12,
            0.12,
        ),
        (
            corr,
            "spearman_high_frac_vs_maser_score",
            "Spearman correlation: observed high-tail fraction vs MASER Io-CML map",
            "jupiter_block_maser_correlation_heatmap",
            "coolwarm",
            -0.45,
            0.45,
        ),
    ]
    for df, value_col, title, stem, cmap, vmin, vmax in specs:
        for antenna in ["rv1_coarse", "rv2_coarse"]:
            mat = _heatmap_table(df, antenna, value_col)
            if mat.empty:
                continue
            fig, ax = plt.subplots(figsize=(12.5, 5.8))
            im = ax.imshow(mat.to_numpy(dtype=float), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks(np.arange(mat.shape[1]))
            ax.set_xticklabels(mat.columns, rotation=45, ha="right")
            ax.set_yticks(np.arange(mat.shape[0]))
            ax.set_yticklabels([f"{float(x):g}" for x in mat.index])
            ax.set_xlabel("date block")
            ax.set_ylabel("frequency (MHz)")
            ax.set_title(f"{title}\n{ANTENNA_LABEL.get(antenna, antenna)}")
            ax.grid(False)
            cbar = fig.colorbar(im, ax=ax, pad=0.015)
            cbar.set_label(value_col)
            fig.tight_layout()
            path = out_dir / f"{stem}_{antenna}.png"
            fig.savefig(path, dpi=180)
            plt.close(fig)
            paths.append(path)
    return paths


def plot_top_block_phase_maps(phase: pd.DataFrame, corr: pd.DataFrame, out_dir: Path, top_n: int = 8) -> Path:
    ranked = corr.dropna(subset=["spearman_high_frac_vs_maser_score"]).copy()
    if ranked.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No valid block phase-map correlations", ha="center", va="center")
        path = out_dir / "jupiter_top_block_phase_maps.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path
    ranked["abs_corr"] = ranked["spearman_high_frac_vs_maser_score"].abs()
    ranked = ranked.sort_values("abs_corr", ascending=False).head(int(top_n)).reset_index(drop=True)
    ncols = 2
    nrows = int(np.ceil(len(ranked) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.5, max(6.5, 4.6 * nrows)), sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(-1)
    vals = phase["high_power_fraction"].dropna()
    vmax = max(0.02, float(vals.quantile(0.98))) if not vals.empty else 0.1
    for ax, (_, row) in zip(axes, ranked.iterrows()):
        sub = phase[
            phase["block"].astype(str).eq(str(row["block"]))
            & phase["antenna"].astype(str).eq(str(row["antenna"]))
            & phase["frequency_band"].astype(int).eq(int(row["frequency_band"]))
        ].copy()
        mat = sub.pivot(index="io_bin_deg", columns="cml_bin_deg", values="high_power_fraction").sort_index()
        if mat.empty:
            ax.axis("off")
            continue
        im = ax.imshow(
            mat.to_numpy(dtype=float),
            origin="lower",
            extent=[
                float(mat.columns.min() - 7.5),
                float(mat.columns.max() + 7.5),
                float(mat.index.min() - 7.5),
                float(mat.index.max() + 7.5),
            ],
            aspect="auto",
            cmap="magma",
            vmin=0.0,
            vmax=vmax,
        )
        ax.set_title(
            f"{row['block']} {ANTENNA_LABEL.get(str(row['antenna']), row['antenna'])} "
            f"{float(row['frequency_mhz']):.2f} MHz\n"
            f"r={float(row['spearman_high_frac_vs_maser_score']):.2f}, bins={int(row['n_phase_bins'])}"
        )
        ax.set_xlabel("CML (deg)")
        ax.set_ylabel("Io phase (deg)")
    for ax in axes[len(ranked) :]:
        ax.axis("off")
    fig.suptitle("Most MASER-correlated block-resolved Jupiter phase maps")
    cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.82, pad=0.015)
    cbar.set_label("high-tail fraction")
    fig.savefig(out_dir / "jupiter_top_block_phase_maps.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_dir / "jupiter_top_block_phase_maps.png"


def write_report(
    out_dir: Path,
    contrast: pd.DataFrame,
    corr: pd.DataFrame,
    paths: list[Path],
    config: dict[str, object],
    n_samples: int,
) -> Path:
    top_corr = corr.dropna(subset=["spearman_high_frac_vs_maser_score"]).copy()
    top_corr["abs_corr"] = top_corr["spearman_high_frac_vs_maser_score"].abs()
    corr_cols = [
        "block",
        "antenna",
        "frequency_mhz",
        "n_phase_bins",
        "spearman_high_frac_vs_maser_score",
        "spearman_median_z_vs_maser_score",
    ]
    top_contrast = contrast.copy()
    top_contrast["abs_high_minus_low"] = top_contrast["high_minus_low_high_frac"].abs()
    contrast_cols = [
        "block",
        "antenna",
        "frequency_mhz",
        "visible_high_maser_score_n",
        "visible_low_maser_score_n",
        "high_minus_low_high_frac",
        "earth_occulted_high_minus_low_high_frac",
    ]
    lines = [
        "# Block-Resolved Jupiter Phase-Pattern Survey",
        "",
        "This run checks whether a Jupiter phase pattern appears only in specific date blocks.",
        "",
        "## Method",
        "",
        "- Uses all valid upper-V and lower-V Ryle-Vonberg samples.",
        "- Uses the same daily channel-normalized log-power variable as the full phase survey.",
        "- Blocks are calendar months by default.",
        "- A high-power sample is `daily_z_log_power > high_z`.",
        "- CML/Io bins are compared with the MASER/PADC Io-CML probability map.",
        "- The main within-visible control is high-MASER-score bins versus low-MASER-score bins in the same RAE-2 data block.",
        "",
        "## Historical Data Context",
        "",
        "- Published and digitized NDA Jupiter event catalogs are openly available from 1978-1990, which starts after the main RAE-2 interval used here.",
        "- The Jupiter Probability Tool and MASER/PADC map collection provide phase-CML occurrence maps and explicitly use CML/Io geometry for observation interpretation.",
        "- Cassini/Voyager-style analyses commonly compute occurrence probability in CML/Io bins and use thresholded activity counts rather than occultation step fits.",
        "- Therefore these block plots should be read as occurrence-pattern diagnostics, not as per-event occultation detections.",
        "",
        "## Configuration",
        "",
        *[f"- `{k}`: `{v}`" for k, v in config.items() if k != "software_versions"],
        "",
        f"Samples used after geometry merge: `{n_samples}`.",
        "",
        "## Strongest Block Correlations",
        "",
        top_corr.sort_values("abs_corr", ascending=False).head(20)[corr_cols].to_string(index=False)
        if not top_corr.empty
        else "(none)",
        "",
        "## Strongest High-Score Versus Low-Score Block Contrasts",
        "",
        top_contrast.sort_values("abs_high_minus_low", ascending=False).head(20)[contrast_cols].to_string(index=False)
        if not top_contrast.empty
        else "(none)",
        "",
        "## Plots",
        "",
        *[f"- `{p}`" for p in paths],
    ]
    path = out_dir / "jupiter_block_phase_pattern_survey_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", type=Path, default=DEFAULT_CLEAN)
    parser.add_argument("--jupiter-states", type=Path, default=DEFAULT_JUPITER_STATES)
    parser.add_argument("--earth-states", type=Path, default=DEFAULT_EARTH_STATES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--block-freq", default="M", help="pandas Period frequency, e.g. M or 2M")
    parser.add_argument("--phase-bin-deg", type=float, default=15.0)
    parser.add_argument("--high-z", type=float, default=2.5)
    parser.add_argument("--min-count-per-phase-bin", type=int, default=10)
    parser.add_argument("--geometry-tolerance-s", type=float, default=360.0)
    parser.add_argument("--refresh-external", action="store_true")
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    config = {
        "clean": str(args.clean),
        "jupiter_states": str(args.jupiter_states),
        "earth_states": str(args.earth_states),
        "block_freq": str(args.block_freq),
        "phase_bin_deg": float(args.phase_bin_deg),
        "high_z": float(args.high_z),
        "min_count_per_phase_bin": int(args.min_count_per_phase_bin),
        "geometry_tolerance_s": float(args.geometry_tolerance_s),
        "software_versions": software_versions(),
    }
    write_json(out_dir / "run_config.json", config)

    states = _load_visibility_states(args.jupiter_states, args.earth_states)
    geom = _annotate_spice_grid(states, out_dir, refresh=bool(args.refresh_external))
    clean = _read_clean(args.clean)
    clean = _add_daily_channel_normalization(clean)
    samples = _merge_geometry(clean, geom, float(args.geometry_tolerance_s))

    contrast, corr, phase = summarize_blocks(
        samples=samples,
        block_freq=str(args.block_freq),
        high_z=float(args.high_z),
        phase_bin_deg=float(args.phase_bin_deg),
        min_bin_count=int(args.min_count_per_phase_bin),
    )
    contrast.to_csv(out_dir / "jupiter_block_maser_score_contrast_summary.csv", index=False)
    corr.to_csv(out_dir / "jupiter_block_phase_map_correlation_summary.csv", index=False)
    phase.to_csv(out_dir / "jupiter_block_phase_binned_summary.csv", index=False)

    paths = plot_block_heatmaps(contrast, corr, out_dir)
    paths.append(plot_top_block_phase_maps(phase, corr, out_dir))
    report = write_report(out_dir, contrast, corr, paths, config, len(samples))
    print(report)


if __name__ == "__main__":
    main()
