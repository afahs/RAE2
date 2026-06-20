#!/usr/bin/env python
"""Raw sample-level inspection plots for block-resolved Jupiter candidates."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rylevonberg.table_io import read_table


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLES = ROOT / "outputs/jupiter_phase_pattern_survey_v1/jupiter_phase_pattern_sampled_points.csv"
DEFAULT_CORR = ROOT / "outputs/jupiter_block_phase_pattern_survey_v1/jupiter_block_phase_map_correlation_summary.csv"
DEFAULT_OUT = ROOT / "outputs/jupiter_block_phase_pattern_survey_v1"

ANTENNA_LABEL = {
    "rv1_coarse": "upper V",
    "rv2_coarse": "lower V",
}


def _load_top_blocks(corr_path: Path, top_n: int) -> pd.DataFrame:
    corr = read_table(corr_path)
    corr = corr.dropna(subset=["spearman_high_frac_vs_maser_score"]).copy()
    corr["abs_corr"] = corr["spearman_high_frac_vs_maser_score"].abs()
    return corr.sort_values("abs_corr", ascending=False).head(int(top_n)).reset_index(drop=True)


def _read_candidate_samples(samples_path: Path, top: pd.DataFrame) -> pd.DataFrame:
    usecols = [
        "time",
        "frequency_band",
        "frequency_mhz",
        "antenna",
        "daily_z_log_power",
        "jupiter_visible_by_moon",
        "earth_visible_by_moon",
        "jupiter_cml_spice_deg",
        "io_phase_spice_deg",
        "maser_zarka_io_score",
    ]
    df = read_table(samples_path, usecols=usecols, parse_dates=["time"])
    df["block"] = df["time"].dt.to_period("M").astype(str)
    pieces = []
    for _, row in top.iterrows():
        sub = df[
            df["block"].astype(str).eq(str(row["block"]))
            & df["antenna"].astype(str).eq(str(row["antenna"]))
            & np.isclose(df["frequency_mhz"].astype(float), float(row["frequency_mhz"]))
            & df["jupiter_visible_by_moon"].astype(bool)
        ].copy()
        sub["block_corr"] = float(row["spearman_high_frac_vs_maser_score"])
        pieces.append(sub)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=usecols + ["block", "block_corr"])


def plot_time_strips(samples: pd.DataFrame, top: pd.DataFrame, out_dir: Path, high_z: float, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    n = len(top)
    fig, axes = plt.subplots(n, 1, figsize=(12.5, max(3.0 * n, 5.0)), sharex=False, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (_, row) in zip(axes, top.iterrows()):
        sub = samples[
            samples["block"].astype(str).eq(str(row["block"]))
            & samples["antenna"].astype(str).eq(str(row["antenna"]))
            & np.isclose(samples["frequency_mhz"].astype(float), float(row["frequency_mhz"]))
        ].copy()
        if len(sub) > 10000:
            sub = sub.iloc[rng.choice(len(sub), size=10000, replace=False)].sort_values("time")
        sc = ax.scatter(
            sub["time"],
            sub["daily_z_log_power"],
            c=sub["maser_zarka_io_score"],
            s=5,
            alpha=0.5,
            cmap="viridis",
            vmin=0,
            vmax=max(1.0, float(samples["maser_zarka_io_score"].quantile(0.995))),
            rasterized=True,
        )
        ax.axhline(high_z, color="crimson", lw=1.1, ls="--", label=f"high-power threshold z>{high_z:g}")
        ax.axhline(0.0, color="0.35", lw=0.8)
        ax.set_ylim(-4, 8)
        ax.set_ylabel("daily-normalized\nlog power")
        ax.set_title(
            f"{row['block']}  {ANTENNA_LABEL.get(str(row['antenna']), row['antenna'])}  "
            f"{float(row['frequency_mhz']):.2f} MHz  "
            f"Io-CML map correlation={float(row['spearman_high_frac_vs_maser_score']):+.2f}"
        )
    axes[-1].set_xlabel("UTC time within monthly block")
    fig.subplots_adjust(left=0.075, right=0.86, top=0.965, bottom=0.055, hspace=0.72)
    cax = fig.add_axes([0.88, 0.12, 0.018, 0.76])
    fig.colorbar(sc, cax=cax, label="MASER Io-CML probability score")
    fig.suptitle("Jupiter block candidates: actual Ryle-Vonberg samples")
    path = out_dir / "jupiter_top_block_raw_time_strips.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_phase_scatter(samples: pd.DataFrame, top: pd.DataFrame, out_dir: Path, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    ncols = 2
    nrows = int(np.ceil(len(top) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.0, max(4.4 * nrows, 5.5)), sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(-1)
    zclip = np.nanpercentile(samples["daily_z_log_power"].to_numpy(dtype=float), [1, 99])
    for ax, (_, row) in zip(axes, top.iterrows()):
        sub = samples[
            samples["block"].astype(str).eq(str(row["block"]))
            & samples["antenna"].astype(str).eq(str(row["antenna"]))
            & np.isclose(samples["frequency_mhz"].astype(float), float(row["frequency_mhz"]))
        ].copy()
        if len(sub) > 12000:
            sub = sub.iloc[rng.choice(len(sub), size=12000, replace=False)]
        sc = ax.scatter(
            sub["jupiter_cml_spice_deg"],
            sub["io_phase_spice_deg"],
            c=sub["daily_z_log_power"],
            s=5,
            alpha=0.55,
            cmap="coolwarm",
            vmin=float(zclip[0]),
            vmax=float(zclip[1]),
            rasterized=True,
        )
        hi = sub["maser_zarka_io_score"] >= sub["maser_zarka_io_score"].quantile(0.75)
        if hi.any():
            ax.scatter(
                sub.loc[hi, "jupiter_cml_spice_deg"],
                sub.loc[hi, "io_phase_spice_deg"],
                facecolors="none",
                edgecolors="black",
                linewidths=0.35,
                s=12,
                alpha=0.25,
                rasterized=True,
            )
        ax.set_xlim(0, 360)
        ax.set_ylim(0, 360)
        ax.set_title(
            f"{row['block']} {ANTENNA_LABEL.get(str(row['antenna']), row['antenna'])} "
            f"{float(row['frequency_mhz']):.2f} MHz\n"
            f"r={float(row['spearman_high_frac_vs_maser_score']):+.2f}"
        )
        ax.set_xlabel("Jupiter CML (deg)")
        ax.set_ylabel("Io phase (deg)")
    for ax in axes[len(top) :]:
        ax.axis("off")
    fig.colorbar(sc, ax=axes.tolist(), pad=0.012, label="daily-normalized log power")
    fig.suptitle("Jupiter block candidates in Io-CML space: sample-level data")
    fig.savefig(out_dir / "jupiter_top_block_raw_phase_scatter.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_dir / "jupiter_top_block_raw_phase_scatter.png"


def write_guide(out_dir: Path, paths: list[Path], top: pd.DataFrame) -> Path:
    path = out_dir / "jupiter_block_raw_inspection_guide.md"
    lines = [
        "# Jupiter Block Raw Inspection Guide",
        "",
        "These plots show the same block candidates as actual Ryle-Vonberg sample-level data.",
        "",
        "- The time-strip plot answers whether a candidate block is a few isolated high samples, a broad receiver/background drift, or repeated burst-like activity.",
        "- The phase-scatter plot shows where those samples fall in Jupiter CML and Io phase.",
        "- Black outlines mark the upper quartile of the MASER Io-CML probability score within each panel. A convincing Jupiter pattern would have high normalized powers preferentially inside those outlined regions.",
        "- The plotted power is daily-normalized log power, not an occultation-fit statistic.",
        "",
        "## Files",
        "",
        *[f"- `{p}`" for p in paths],
        "",
        "## Candidate Blocks Plotted",
        "",
        top[
            [
                "block",
                "antenna",
                "frequency_mhz",
                "n_phase_bins",
                "spearman_high_frac_vs_maser_score",
                "spearman_median_z_vs_maser_score",
            ]
        ].to_string(index=False),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--corr", type=Path, default=DEFAULT_CORR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--high-z", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=20260609)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    top = _load_top_blocks(args.corr, args.top_n)
    samples = _read_candidate_samples(args.samples, top)
    paths = [
        plot_time_strips(samples, top, args.out_dir, high_z=float(args.high_z), seed=int(args.seed)),
        plot_phase_scatter(samples, top, args.out_dir, seed=int(args.seed)),
    ]
    guide = write_guide(args.out_dir, paths, top)
    print(guide)


if __name__ == "__main__":
    main()
