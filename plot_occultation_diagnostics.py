#!/usr/bin/env python3
"""Visualization helpers for occultation significance CSV.

Run on an interactive node as noted in AGENTS.md (`salloc ...; conda activate
luseepy_env`) before generating figures.
"""

import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_occultation_significance import (
    EVENT_ORIENTATION,
    load_events,
    safe_divide,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create histograms, QQ plots, volcano plots, and stacked significance "
            "spectra from occultation significance outputs."
        )
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("RAE2/outputs/random_occultations"),
        help="Directory containing objective JSONs and the summary CSV.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Path to the CSV created by analyze_occultation_significance.py.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level used for BH and plotting thresholds.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for output figures (defaults to BASE_DIR/figures).",
    )
    return parser.parse_args()


def reshape_by_event_freq(
    df: pd.DataFrame, value_column: str, event_order: Sequence[str], freq_order: Sequence[str]
) -> np.ndarray:
    """Pivot long-form values to event x freq arrays following the supplied order."""
    arr = np.full((len(event_order), len(freq_order)), np.nan, dtype=float)
    event_index: Dict[str, int] = {name: idx for idx, name in enumerate(event_order)}
    freq_index: Dict[str, int] = {freq: idx for idx, freq in enumerate(freq_order)}
    for row in df.itertuples():
        arr[event_index[row.event], freq_index[row.freq_label]] = getattr(row, value_column)
    return arr


def gaussian_pdf(x: np.ndarray) -> np.ndarray:
    return (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x ** 2)


def empirical_pvalues(u_target: np.ndarray, u_reference: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return directional/two-sided/flip empirical p-values for target vs reference."""
    comparator = u_reference.shape[0] + 1.0
    ge_counts = (u_reference[:, None, :] >= u_target[None, :, :]).sum(axis=0)
    p_dir = (1.0 + ge_counts) / comparator

    abs_counts = (np.abs(u_reference)[:, None, :] >= np.abs(u_target)[None, :, :]).sum(axis=0)
    p_two = (1.0 + abs_counts) / comparator

    le_counts = (u_reference[:, None, :] <= u_target[None, :, :]).sum(axis=0)
    p_flip = np.ones_like(p_dir)
    neg_mask = u_target < 0
    p_flip[neg_mask] = (1.0 + le_counts[neg_mask]) / comparator
    return p_dir, p_two, p_flip


def plot_z_histograms(
    label: str,
    z_candidates: np.ndarray,
    z_random: np.ndarray,
    freq_labels: Sequence[str],
    out_dir: Path,
) -> None:
    """Histogram/PDF comparison for Z and Z_rand."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pooled_cand = z_candidates.ravel()
    pooled_rand = z_random.ravel()
    bins = np.linspace(-5, 5, 41)
    x = np.linspace(-5, 5, 400)

    plt.figure(figsize=(6, 4))
    plt.hist(pooled_rand, bins=bins, density=True, alpha=0.5, label="Random Z", color="gray")
    plt.hist(pooled_cand, bins=bins, density=True, alpha=0.5, label="Candidate Z", color="tab:orange")
    plt.plot(x, gaussian_pdf(x), "k--", label="N(0,1)")
    plt.xlabel("Z-score")
    plt.ylabel("Density")
    plt.title(f"{label}: Pooled Z-score Distributions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "z_scores_hist_pooled.png", dpi=200)
    plt.close()

    n_freq = len(freq_labels)
    ncols = 3
    nrows = int(np.ceil(n_freq / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3), sharex=True, sharey=True)
    axes = axes.ravel()
    for idx, freq in enumerate(freq_labels):
        ax = axes[idx]
        ax.hist(z_random[:, idx], bins=bins, density=True, alpha=0.5, label="Random", color="gray")
        ax.hist(z_candidates[:, idx], bins=bins, density=True, alpha=0.5, label="Candidate", color="tab:orange")
        ax.plot(x, gaussian_pdf(x), "k--")
        ax.set_title(f"freq={freq}")
        if idx % ncols == 0:
            ax.set_ylabel("Density")
        if idx >= (nrows - 1) * ncols:
            ax.set_xlabel("Z-score")
    for ax in axes[n_freq:]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle(f"{label}: Per-channel Z-score Histograms", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_dir / "z_scores_hist_per_channel.png", dpi=200)
    plt.close(fig)


def plot_qq(
    label: str,
    pvals_cand: np.ndarray,
    pvals_rand: np.ndarray,
    freq_per_point: Sequence[str],
    out_dir: Path,
) -> None:
    """QQ plot vs uniform for candidate and random-sky p-values."""
    out_dir.mkdir(parents=True, exist_ok=True)
    def prepare(pvals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        finite = np.isfinite(pvals)
        sorted_vals = np.sort(pvals[finite])
        quantiles = (np.arange(1, sorted_vals.size + 1) - 0.5) / sorted_vals.size
        return quantiles, sorted_vals

    finite = np.isfinite(pvals_cand)
    freq_arr = np.asarray(freq_per_point)
    candidate_vals = pvals_cand[finite]
    candidate_freqs = freq_arr[finite]
    order = np.argsort(candidate_vals)
    s_cand = candidate_vals[order]
    freq_sorted = candidate_freqs[order]
    q_cand = (np.arange(1, s_cand.size + 1) - 0.5) / max(s_cand.size, 1)

    q_rand, s_rand = prepare(pvals_rand)
    max_q = max(q_cand.max() if q_cand.size else 0, q_rand.max() if q_rand.size else 0, 1e-3)
    plt.figure(figsize=(5, 5))
    plt.plot([0, max_q], [0, max_q], "k--", label="Uniform reference")
    if q_rand.size:
        plt.plot(q_rand, s_rand, marker="o", linestyle="", markersize=3, label="Random sky")
    if q_cand.size:
        plt.scatter(q_cand, s_cand, marker="o", s=25, label="Candidates")
        for q, s, freq in zip(q_cand, s_cand, freq_sorted):
            plt.annotate(freq, (q, s), textcoords="offset points", xytext=(4, 2), fontsize=7)
    plt.xlabel("Uniform quantiles")
    plt.ylabel("Observed p-values")
    plt.title(f"{label}: QQ Plot of Directional p-values")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "qq_pvalues.png", dpi=200)
    plt.close()


def plot_volcano(
    label: str,
    u: np.ndarray,
    p_dir: np.ndarray,
    event_type: np.ndarray,
    freq_per_point: Sequence[str],
    alpha: float,
    out_dir: Path,
) -> None:
    """Volcano plot of oriented effect vs -log10 p_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    x = u.ravel()
    y = -np.log10(np.clip(p_dir.ravel(), 1e-12, 1.0))
    colors = np.where(event_type[:, None].repeat(p_dir.shape[1], axis=1).ravel() > 0, "tab:red", "tab:blue")
    freq_arr = np.asarray(freq_per_point)
    plt.figure(figsize=(6, 5))
    plt.scatter(x, y, c=colors, alpha=0.7, edgecolor="none")
    for xi, yi, freq in zip(x, y, freq_arr):
        plt.annotate(freq, (xi, yi), textcoords="offset points", xytext=(4, 2), fontsize=7)
    thresh = -np.log10(alpha)
    plt.axhline(thresh, color="k", linestyle="--", label=f"FDR α={alpha}")
    plt.xlabel("Oriented effect U")
    plt.ylabel("-log10 p_dir")
    plt.title(f"{label}: Volcano Plot (Ingress blue / Egress red)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "volcano_plot.png", dpi=200)
    plt.close()


def plot_stacked_spectra(
    label: str,
    freq_numeric: np.ndarray,
    p_dir: np.ndarray,
    event_names: Sequence[str],
    event_type: np.ndarray,
    alpha: float,
    out_dir: Path,
) -> None:
    """Stacked -log10(p_dir) vs frequency plots per event."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n_events = p_dir.shape[0]
    fig, axes = plt.subplots(max(n_events, 1), 1, figsize=(7, 1.8 * max(n_events, 1)), sharex=True)
    if n_events == 1:
        axes = [axes]
    threshold = -np.log10(alpha)
    for idx, ax in enumerate(axes):
        values = -np.log10(np.clip(p_dir[idx], 1e-12, 1.0))
        color = "tab:red" if event_type[idx] > 0 else "tab:blue"
        ax.plot(freq_numeric, values, marker="o", color=color)
        ax.axhline(threshold, color="k", linestyle="--")
        ax.set_ylabel(event_names[idx])
    axes[-1].set_xlabel("Frequency (MHz)")
    fig.suptitle(f"{label}: Stacked Directional Significance Spectra", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_dir / "stacked_significance_spectra.png", dpi=200)
    plt.close(fig)


def plot_fraction_significant(
    label: str,
    freq_labels: Sequence[str],
    p_dir: np.ndarray,
    alpha: float,
    out_dir: Path,
) -> None:
    """Fraction of events with p_dir < alpha as a function of frequency."""
    out_dir.mkdir(parents=True, exist_ok=True)
    freq_numeric = np.array([float(f) for f in freq_labels])
    frac = np.mean(p_dir < alpha, axis=0)
    plt.figure(figsize=(6, 3.5))
    plt.plot(freq_numeric, frac, marker="o")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Fraction significant")
    plt.title(f"{label}: Fraction with p_dir < {alpha}")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_dir / "fraction_significant_vs_frequency.png", dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()
    csv_path = (args.csv or base_dir / "occultation_significance.csv").resolve()
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    output_dir = (args.output_dir or (base_dir / "figures")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    # Ensure freq labels stay string-typed to match JSON metadata.
    df["freq_label"] = df["freq_label"].astype(str)

    # Load random events directly from JSON to lock in frequency ordering.
    (
        freq_labels_rand,
        rand_names,
        event_type_rand,
        delta_rand,
        sigma_rand,
    ) = load_events(base_dir, predicate=lambda name: name.startswith("random_"))
    csv_freq_set = set(df["freq_label"].unique())
    if csv_freq_set != set(freq_labels_rand):
        raise SystemExit(
            "CSV frequencies do not match random JSON files. "
            "Regenerate the CSV after running analyze_occultation_significance.py."
        )
    freq_labels = freq_labels_rand
    event_names = list(dict.fromkeys(df["event"]))
    freq_numeric = np.array([float(f) for f in freq_labels])
    event_type = (
        df.drop_duplicates("event")
            .set_index("event")["event_type"]
            .reindex(event_names)
            .to_numpy()
    )

    def reshape(col: str) -> np.ndarray:
        return reshape_by_event_freq(df, col, event_names, freq_labels)

    z_candidates = reshape("z_score")
    p_dir = reshape("p_dir")
    u = reshape("u_stat")

    z_rand = safe_divide(delta_rand, sigma_rand)
    u_rand = event_type_rand[:, None] * z_rand
    p_dir_rand, _, _ = empirical_pvalues(u_rand, u_rand)
    p_dir_rand_flat = p_dir_rand.ravel()

    groups: Dict[str, list] = defaultdict(list)
    for idx, name in enumerate(event_names):
        base = name.rsplit("_", 1)[0]
        groups[base].append(idx)

    for source, indices in groups.items():
        sub_dir = output_dir / source
        sub_dir.mkdir(parents=True, exist_ok=True)
        z_source = z_candidates[indices, :]
        p_source = p_dir[indices, :]
        u_source = u[indices, :]
        event_type_source = event_type[indices]
        event_names_source = [event_names[i] for i in indices]
        freq_per_point = np.tile(np.array(freq_labels), len(indices))

        plot_z_histograms(source, z_source, z_rand, freq_labels, sub_dir)
        plot_qq(source, p_source.ravel(), p_dir_rand_flat, freq_per_point, sub_dir)
        plot_volcano(source, u_source, p_source, event_type_source, freq_per_point, args.alpha, sub_dir)
        plot_stacked_spectra(
            source, freq_numeric, p_source, event_names_source, event_type_source, args.alpha, sub_dir
        )
        plot_fraction_significant(source, freq_labels, p_source, args.alpha, sub_dir)

    print(f"Figures saved to {output_dir} (per-source subdirectories).")


if __name__ == "__main__":
    main()
