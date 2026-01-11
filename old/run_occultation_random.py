#!/usr/bin/env python3
"""
Occultation analysis for randomly distributed sky locations plus benchmark sources.

This script augments ``run_ingress_egress.py`` by sampling a set of uniformly
distributed directions on the celestial sphere, running the standard
ingress/egress analytics for each, and comparing the resulting objective metrics
to those of Virgo-A and Fornax-A.  The goal is to build a null distribution that
quantifies the significance of detections.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord, FK4
from astropy.time import Time
from concurrent.futures import ProcessPoolExecutor, as_completed
from matplotlib.lines import Line2D

import RAEAnglesUtilities as rae
from ingress_egress_helper import (
    DEFAULT_DATA_PATH,
    compute_ingress_egress_objective_over_freqs,
    has_histogram_content,
    load_occultation_dataframe,
    plot_objective_vs_frequency,
    occultationStatisticsIngressEgressPairs,
)



DATA_START = "1974-01-01 14:00"
DATA_END = "1975-12-31 16:00"

STATIC_SOURCES: Dict[str, SkyCoord] = {
    "virgo_a": SkyCoord(
        ra="12h30m49.42338s",
        dec="+12d23m28.0439s",
        frame="fk4",
        equinox="B1950",
        unit=(u.hourangle, u.deg),
    ),
    "fornax_a": SkyCoord(
        ra="03h22m41s",
        dec="-37d12m30s",
        frame="fk4",
        equinox="B1950",
        unit=(u.hourangle, u.deg),
    ),
}


DATA_START = "1974-01-01 14:00"
DATA_END = "1975-12-31 16:00"

def slugify(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run occultation analysis on random sky positions and compare against benchmark sources."
    )
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH), help="Path to interpolated master CSV.")
    parser.add_argument("--output-dir", default="RAE2/outputs/random_occultations", help="Directory for outputs.")
    parser.add_argument("--random-root", help="Directory containing existing random *_objective.json files (defaults to output).")
    parser.add_argument("--sources-root", help="Directory containing benchmark source objective JSONs (virgo_a, fornax_a).")
    parser.add_argument("--num-random", type=int, default=50, help="Number of random sky directions to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the random generator.")
    parser.add_argument("--window-minutes", type=float, default=2.0, help="Ingress/egress half-window in minutes.")
    parser.add_argument("--antenna", default="rv2_coarse", help="Antenna column to analyse.")
    parser.add_argument(
        "--aggregate",
        choices=["sum", "mean", "weighted_sum", "weighted_mean"],
        default="weighted_sum",
        help="Aggregation mode for objective metrics.",
    )
    parser.add_argument("--no-std-weights", action="store_true", help="Disable inverse-std weighting.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument("--skip-random", action="store_true", help="Skip recomputing random positions (reuse existing).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    random_root = Path(args.random_root).expanduser().resolve() if args.random_root else output_root
    sources_root = Path(args.sources_root).expanduser().resolve() if args.sources_root else output_root

    if not args.skip_random:
        random_root.mkdir(parents=True, exist_ok=True)

    data = load_occultation_dataframe(
        args.data_path, start=DATA_START, end=DATA_END
    )
    data.sort_index(inplace=True)

    use_std_weights = not args.no_std_weights
    show_progress = not args.no_progress

    random_results: List[Dict] = []

    if not args.skip_random:
        random_targets = generate_random_targets(args.num_random, seed=args.seed)
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(
                    process_target,
                    name=f"random_{idx:02d}",
                    coord=coord,
                    data=None,
                    data_path=args.data_path,
                    start=DATA_START,
                    end=DATA_END,
                    output_root=random_root,
                    window_minutes=args.window_minutes,
                    antenna=args.antenna,
                    aggregate=args.aggregate,
                    use_std_weights=use_std_weights,
                    show_progress=show_progress,
                )
                for idx, coord in enumerate(random_targets)
            ]
            for future in as_completed(futures):
                random_results.append(future.result())
    else:
        random_dirs = sorted(random_root.glob("random_*/*_objective.json"))
        if not random_dirs:
            raise SystemExit(f"No existing random outputs found under {random_root}; rerun without --skip-random.")
        for path in random_dirs:
            random_results.append(load_results_from_json(path))

    benchmark_results: Dict[str, Dict] = {}
    for slug, coord in STATIC_SOURCES.items():
        print(f"Processing benchmark source {slug}")
        benchmark_results[slug] = process_target(
            name=slug,
            coord=coord,
            data=data,
            data_path=args.data_path,
            start=DATA_START,
            end=DATA_END,
            output_root=random_root,
            window_minutes=args.window_minutes,
            antenna=args.antenna,
            aggregate=args.aggregate,
            use_std_weights=use_std_weights,
            show_progress=show_progress,
        )

    comparison_dir = output_root / "comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    for slug, result in benchmark_results.items():
        plot_path = comparison_dir / f"{slug}_vs_random.png"
        plot_focus_vs_random(
            focus_name=slug,
            focus_results=result,
            random_results=random_results,
            save_path=plot_path,
            alpha_random=0.2,
        )
        print(f"Wrote comparison plot -> {plot_path}")


def generate_random_targets(n: int, seed: int) -> List[SkyCoord]:
    """Sample directions uniformly on the sphere by drawing RA uniformly and sin(Dec) uniformly."""

    rng = np.random.default_rng(seed)
    ra = rng.uniform(0.0, 360.0, size=n) * u.deg
    dec = np.arcsin(rng.uniform(-1.0, 1.0, size=n)) * u.rad
    coords = SkyCoord(ra=ra, dec=dec, frame="icrs")
    fk4_coords = coords.transform_to(FK4(equinox=Time("B1950")))
    return list(fk4_coords)
def _clean_results_for_json(results: Dict) -> Dict:
    cleaned = {"rows": []}
    for row in results.get("rows", []):
        entry = {
            "freq": _to_float(row.get("freq")),
            "freq_label": row.get("freq_label"),
            "freq_band": _to_float(row.get("freq_band")),
        }
        for label in ("ingress", "egress", "combined"):
            metrics = row.get(label) or {}
            entry[label] = {
                "metric": _to_float(metrics.get("metric")),
                "stderr": _to_float(metrics.get("stderr")),
                "n_used": _to_int(metrics.get("n_used")),
            }
        cleaned["rows"].append(entry)
    return cleaned


def process_target(
    name: str,
    coord: SkyCoord,
    data,
    data_path: str,
    start: str,
    end: str,
    output_root: Path,
    *,
    window_minutes: float,
    antenna: str,
    aggregate: str,
    use_std_weights: bool,
    show_progress: bool,
) -> Dict:
    source_dir = output_root / slugify(name)
    source_dir.mkdir(parents=True, exist_ok=True)

    if data is None:
        data = load_occultation_dataframe(data_path, start=start, end=end)
        data.sort_index(inplace=True)
    else:
        data = data.copy()

    vis_col = add_static_source_visibility(data, name, coord)
    stats = occultationStatisticsIngressEgressPairs(
        data,
        col=vis_col,
        window=pd.Timedelta(minutes=window_minutes),
        antenn=antenna,
        progress=show_progress,
    )

    usable_stats = {}
    min_bins = []
    for freq_key, freq_stats in stats.items():
        if not has_histogram_content(freq_stats):
            continue
        usable_stats[freq_key] = freq_stats
        min_bins.append(0.3)  # uniform percentile trim for randomness

    if not usable_stats:
        raise ValueError(f"No usable ingress/egress pairs for {name}.")

    results = compute_ingress_egress_objective_over_freqs(
        usable_stats,
        use_std_weights=use_std_weights,
        min_bin_percentage=min_bins,
        aggregate=aggregate,
    )

    objective_path = source_dir / f"{slugify(name)}_objective.json"
    plot_path = source_dir / f"{slugify(name)}_objective_vs_frequency.png"

    plot_objective_vs_frequency(results, save_path=str(plot_path), show=False)
    objective_path.write_text(json.dumps(_clean_results_for_json(results), indent=2), encoding="utf-8")

    return {"name": name, "results": _clean_results_for_json(results)}

def _to_int(value):
    if value is None:
        return None
    if isinstance(value, (np.generic,)):
        value = value.item()
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value):
    if value is None:
        return None
    if isinstance(value, (np.generic,)):
        value = value.item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_results_from_json(path: Path) -> Dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {"name": path.parent.name, "results": _clean_results_for_json(payload)}


def plot_focus_vs_random(
    focus_name: str,
    focus_results: Dict,
    random_results: Iterable[Dict],
    save_path: Path,
    *,
    alpha_random: float = 0.2,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    random_color_ingress = "tab:gray"
    random_color_egress = "tab:olive"

    for random_entry in random_results:
        xs, ingress, _, egress, _, _, _ = extract_series(random_entry["results"])
        ingress = np.asarray(ingress, dtype=float)
        egress = np.asarray(egress, dtype=float)
        if np.isfinite(ingress).any():
            ax.plot(xs, ingress, "-", color=random_color_ingress, alpha=alpha_random, linewidth=1)
        if np.isfinite(egress).any():
            ax.plot(xs, egress, "-", color=random_color_egress, alpha=alpha_random, linewidth=1)

    xs, ingress, ingress_err, egress, egress_err, _, _ = extract_series(focus_results)
    ingress = np.asarray(ingress, dtype=float)
    ingress_err = np.asarray(ingress_err, dtype=float)
    egress = np.asarray(egress, dtype=float)
    egress_err = np.asarray(egress_err, dtype=float)

    label_prefix = focus_name.replace('_', ' ').title()
    if np.isfinite(ingress).any():
        ax.errorbar(
            xs,
            ingress,
            yerr=ingress_err,
            fmt="o",
            color="tab:blue",
            linewidth=2,
            capsize=3,
            alpha=1.0,
            label=f"{label_prefix} Ingress",
        )
        ax.plot(xs, ingress, "-", color="tab:blue", linewidth=2, alpha=1.0)
    if np.isfinite(egress).any():
        ax.errorbar(
            xs,
            egress,
            yerr=egress_err,
            fmt="o",
            color="tab:orange",
            linewidth=2,
            capsize=3,
            alpha=1.0,
            label=f"{label_prefix} Egress",
        )
        ax.plot(xs, egress, "-", color="tab:orange", linewidth=2, alpha=1.0)

    ax.set_title(f"{label_prefix} vs. random sky locations")
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Metric (± jackknife SE)")
    ax.grid(True, alpha=0.25)

    legend_handles = [
        Line2D([0], [0], color=random_color_ingress, linewidth=1, alpha=alpha_random, label="Random Ingress"),
        Line2D([0], [0], color=random_color_egress, linewidth=1, alpha=alpha_random, label="Random Egress"),
        Line2D([0], [0], color="tab:blue", linewidth=3, alpha=1.0, label=f"{label_prefix} Ingress"),
        Line2D([0], [0], color="tab:orange", linewidth=3, alpha=1.0, label=f"{label_prefix} Egress"),
    ]
    ax.legend(legend_handles, [h.get_label() for h in legend_handles], frameon=False)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def extract_series(results: Dict) -> tuple:
    rows = results.get("rows", [])
    freq = []
    ingress = []
    ingress_err = []
    egress = []
    egress_err = []
    diff = []
    diff_err = []

    for row in rows:
        freq.append(row.get("freq"))
        in_entry = row.get("ingress", {})
        eg_entry = row.get("egress", {})
        ingress.append(in_entry.get("metric"))
        ingress_err.append(_abs_or_none(in_entry.get("stderr")))
        egress.append(eg_entry.get("metric"))
        egress_err.append(_abs_or_none(eg_entry.get("stderr")))
        if in_entry.get("metric") is not None and eg_entry.get("metric") is not None:
            diff.append(in_entry["metric"] - eg_entry["metric"])
            if in_entry.get("stderr") is not None and eg_entry.get("stderr") is not None:
                diff_err.append(np.sqrt(abs(in_entry["stderr"]) ** 2 + abs(eg_entry["stderr"]) ** 2))
            else:
                diff_err.append(None)
        else:
            diff.append(None)
            diff_err.append(None)

    xs = np.array(freq, dtype=float)
    order = np.argsort(xs)

    def reorder(values):
        arr = np.array([np.nan if v is None else v for v in values], dtype=float)
        return arr[order]

    xs_sorted = xs[order]

    return (
        xs_sorted,
        reorder(ingress),
        reorder([_abs_or_none(v) for v in ingress_err]),
        reorder(egress),
        reorder([_abs_or_none(v) for v in egress_err]),
        reorder(diff),
        reorder(diff_err),
    )


def _abs_or_none(value):
    if value is None:
        return None
    return abs(value)


def add_static_source_visibility(data, source_name: str, coord: SkyCoord) -> str:
    angle_col = f"{slugify(source_name)}_angle"
    vis_col = f"{slugify(source_name)}_vis"
    source_angle = [coord.ra, coord.dec]
    data[angle_col] = rae.raeAngFromSource(data, source_angle)
    data[vis_col] = rae.isVisible(data, data[angle_col])
    return vis_col


if __name__ == "__main__":
    main()
