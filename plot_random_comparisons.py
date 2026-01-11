#!/usr/bin/env python3
"""
Generate comparison plots for benchmark sources versus random sky directions.

This script reads the precomputed objective JSON files written by
``run_occultation_random.py`` and produces two figures (Fornax-A and Virgo-A)
showing ingress/egress objective metrics as a function of frequency. The
benchmark source is highlighted with solid, high-opacity lines, while all 50
random directions are drawn in the background with low opacity.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np

BENCHMARK_SOURCES = ("fornax_a", "virgo_a","crab_nebula")
RANDOM_PREFIX = "random_"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot benchmark ingress/egress curves against random sky realizations."
    )
    parser.add_argument(
        "--root", required=True, help="Path to the run_occultation_random.py output directory."
    )
    parser.add_argument(
        "--comparison-dir",
        default="comparisons_from_json",
        help="Subdirectory (under root) where figures will be saved.",
    )
    parser.add_argument(
        "--random-alpha",
        type=float,
        default=0.2,
        help="Opacity for random ingress/egress curves.",
    )
    return parser.parse_args()


def load_objective(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_series(results: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = results.get("rows", [])
    freq = []
    ingress = []
    egress = []

    for row in rows:
        freq.append(row.get("freq"))
        ingress.append(_to_float(row.get("ingress", {}).get("metric")))
        egress.append(_to_float(row.get("egress", {}).get("metric")))

    freq_arr = np.array(freq, dtype=float)
    order = np.argsort(freq_arr)

    return (
        freq_arr[order],
        np.array(ingress, dtype=float)[order],
        np.array(egress, dtype=float)[order],
    )


def _to_float(value):
    if value is None:
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def plot_comparison(
    focus_name: str,
    focus_results: Dict,
    random_results: Iterable[Dict],
    save_path: Path,
    *,
    random_alpha: float,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    # Plot random curves first.
    for entry in random_results:
        xs, ingress, egress = extract_series(entry)
        if np.isfinite(ingress).any():
            ax.plot(xs, ingress, "-", color="tab:gray", alpha=random_alpha, linewidth=1)
        if np.isfinite(egress).any():
            ax.plot(xs, egress, "-", color="tab:olive", alpha=random_alpha, linewidth=1)

    # Plot focus source.
    xs, ingress, egress = extract_series(focus_results)
    label_prefix = focus_name.replace("_", " ").title()

    if np.isfinite(ingress).any():
        ax.plot(xs, ingress, "-", color="tab:blue", linewidth=3, alpha=1.0, label=f"{label_prefix} Ingress")
    if np.isfinite(egress).any():
        ax.plot(xs, egress, "-", color="tab:orange", linewidth=3, alpha=1.0, label=f"{label_prefix} Egress")

    ax.set_title(f"{label_prefix} vs. random sky directions")
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Metric")
    ax.grid(True, alpha=0.25)

    handles = []
    if np.isfinite(ingress).any():
        handles.append(ax.lines[-2])  # ingress line
    if np.isfinite(egress).any():
        handles.append(ax.lines[-1])  # egress line
    ax.legend(handles, [h.get_label() for h in handles], frameon=False)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Input root {root} does not exist or is not a directory.")

    comparison_dir = root / args.comparison_dir
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # Load random results.
    random_results = []
    for random_dir in sorted(root.glob(f"{RANDOM_PREFIX}*")):
        obj_path = random_dir / f"{random_dir.name}_objective.json"
        if obj_path.is_file():
            random_results.append(load_objective(obj_path))

    if not random_results:
        raise SystemExit(f"No random objective JSONs found under {root}.")

    # Plot for each benchmark source.
    for focus_slug in BENCHMARK_SOURCES:
        obj_path = root / focus_slug / f"{focus_slug}_objective.json"
        if not obj_path.is_file():
            print(f"Skipping {focus_slug}: {obj_path} not found.")
            continue

        focus_results = load_objective(obj_path)
        plot_path = comparison_dir / f"{focus_slug}_vs_random_ingress_egress.png"
        plot_comparison(
            focus_name=focus_slug,
            focus_results=focus_results,
            random_results=random_results,
            save_path=plot_path,
            random_alpha=args.random_alpha,
        )
        print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
