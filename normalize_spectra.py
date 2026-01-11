#!/usr/bin/env python3
"""
Normalize occultation-derived spectra using the Sun as a flat-spectrum calibrator.

Workflow
--------
1. Load the Sun objective JSON produced by ``run_ingress_egress.py`` and build a
   per-frequency correction curve by normalizing the Sun's combined metric to its
   median value. Any chromatic structure is interpreted as beam gain.
2. For every other source directory (including optional null references), divide
   the ingress, egress, and combined metrics by the correction curve to recover
   beam-corrected spectra; errors are propagated in quadrature (σ / gain).
3. Emit per-source ``*_objective_corrected.json`` files alongside a global CSV
   summary and a CSV copy of the correction curve for downstream analysis.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

METRIC_FIELDS = ("metric", "stderr", "n_used")


def load_objective(path: Path) -> Dict[str, Dict[str, Dict[str, Optional[float]]]]:
    """Return objective rows keyed by frequency label with clean numeric values."""

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    rows: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    for row in payload.get("rows", []):
        freq_label = str(row.get("freq_label"))
        per_label: Dict[str, Dict[str, Optional[float]]] = {}
        for label in ("ingress", "egress", "combined"):
            entry = row.get(label)
            if not isinstance(entry, dict):
                continue
            per_label[label] = {
                field: _to_float(entry.get(field)) for field in METRIC_FIELDS
            }
        per_label["meta"] = {
            "freq": _to_float(row.get("freq")),
            "freq_band": _to_float(row.get("freq_band")),
        }
        rows[freq_label] = per_label
    return rows


def _to_float(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (np.generic,)):
        value = value.item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_correction_curve(
    sun_rows: Dict[str, Dict[str, Dict[str, Optional[float]]]]
) -> Dict[str, float]:
    """Normalize the Sun's combined metric to its median to obtain beam gain."""

    gains: Dict[str, float] = {}
    metrics: List[float] = []
    for freq_label, entry in sun_rows.items():
        metric = entry.get("combined", {}).get("metric")
        if metric is None or not np.isfinite(metric):
            continue
        metrics.append(metric)
        gains[freq_label] = metric

    if not metrics:
        raise ValueError("Sun objective file does not contain usable combined metrics.")

    baseline = float(np.nanmedian(metrics))
    if baseline == 0 or not np.isfinite(baseline):
        raise ValueError("Sun combined metrics median is zero or non-finite.")

    for freq_label, metric in gains.items():
        gains[freq_label] = metric / baseline
    return gains


def apply_correction(
    rows: Dict[str, Dict[str, Dict[str, Optional[float]]]],
    correction: Dict[str, float],
) -> Dict[str, Dict[str, Dict[str, Optional[float]]]]:
    """Return a deep copy with metrics divided by the Sun-derived gain."""

    corrected: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    for freq_label, entry in rows.items():
        gain = correction.get(freq_label)
        per_label: Dict[str, Dict[str, Optional[float]]] = {}
        for label, metrics in entry.items():
            if label == "meta":
                per_label[label] = dict(metrics)
                continue
            per_label[label] = {}
            for field, value in metrics.items():
                if field == "n_used" or gain is None or gain == 0:
                    per_label[label][field] = value
                    continue
                if value is None:
                    per_label[label][field] = None
                else:
                    per_label[label][field] = value / gain
        per_label["meta"] = dict(entry.get("meta", {}))
        corrected[freq_label] = per_label
    return corrected


def write_corrected_objective(
    dest: Path,
    rows: Dict[str, Dict[str, Dict[str, Optional[float]]]],
) -> None:
    data = {"rows": []}
    for freq_label, entry in rows.items():
        record = {
            "freq_label": freq_label,
            "freq": entry.get("meta", {}).get("freq"),
            "freq_band": entry.get("meta", {}).get("freq_band"),
        }
        for label in ("ingress", "egress", "combined"):
            if label in entry:
                record[label] = {
                    field: entry[label].get(field) for field in METRIC_FIELDS
                }
        data["rows"].append(record)
    dest.write_text(json.dumps(data, indent=2), encoding="utf-8")


def iter_source_dirs(root: Path) -> Iterable[Path]:
    for child in sorted(root.iterdir()):
        if child.is_dir():
            yield child


def locate_objective_file(directory: Path) -> Optional[Path]:
    candidates = sorted(directory.glob("*_objective.json"))
    return candidates[0] if candidates else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize occultation spectra by dividing out the Sun-derived beam chromaticity."
    )
    parser.add_argument("--input-root", required=True, help="Path to run_ingress_egress.py output directory")
    parser.add_argument("--sun-name", default="sun", help="Slugified directory name holding the Sun results")
    parser.add_argument("--summary-csv", default="corrected_summary.csv", help="Filename for consolidated CSV (written inside input root)")
    parser.add_argument("--curve-csv", default="sun_correction_curve.csv", help="Filename for the Sun correction curve CSV")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing corrected JSON files if present")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.input_root).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Input root {root} does not exist or is not a directory.")

    sun_dir = root / args.sun_name
    if not sun_dir.is_dir():
        raise SystemExit(f"Sun directory {sun_dir} not found.")

    sun_obj = locate_objective_file(sun_dir)
    if sun_obj is None:
        raise SystemExit(f"No objective JSON found in {sun_dir}.")

    print(f"Loading Sun objective file: {sun_obj}")
    sun_rows = load_objective(sun_obj)
    correction = build_correction_curve(sun_rows)

    write_correction_curve(root / args.curve_csv, correction, sun_rows)

    summary_rows: List[List[Optional[float]]] = []
    header = [
        "source",
        "freq_label",
        "freq_mhz",
        "ingress_metric",
        "ingress_stderr",
        "egress_metric",
        "egress_stderr",
        "combined_metric",
        "combined_stderr",
    ]

    for source_dir in iter_source_dirs(root):
        obj_path = locate_objective_file(source_dir)
        if obj_path is None:
            continue

        source_slug = source_dir.name
        corrected_path = source_dir / f"{obj_path.stem}_corrected.json"
        if corrected_path.exists() and not args.overwrite:
            print(f"Skipping {source_slug}: corrected file already exists (use --overwrite to replace).")
            continue

        rows = load_objective(obj_path)
        corrected = apply_correction(rows, correction)
        write_corrected_objective(corrected_path, corrected)
        print(f"Wrote corrected objective for {source_slug} -> {corrected_path.name}")

        plot_path = source_dir / f"{obj_path.stem}_corrected.png"
        plot_corrected_objective(rows, corrected, plot_path, title=f"{source_slug} (corrected)")
        print(f"Wrote corrected plot for {source_slug} -> {plot_path.name}")

        for freq_label, entry in corrected.items():
            meta = entry.get("meta", {})
            summary_rows.append(
                [
                    source_slug,
                    freq_label,
                    meta.get("freq"),
                    entry.get("ingress", {}).get("metric"),
                    entry.get("ingress", {}).get("stderr"),
                    entry.get("egress", {}).get("metric"),
                    entry.get("egress", {}).get("stderr"),
                    entry.get("combined", {}).get("metric"),
                    entry.get("combined", {}).get("stderr"),
                ]
            )

    if summary_rows:
        summary_path = root / args.summary_csv
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(summary_rows)
        print(f"Wrote summary CSV -> {summary_path}")
    else:
        print("No sources processed; summary CSV not created.")


def plot_corrected_objective(
    original: Dict[str, Dict[str, Dict[str, Optional[float]]]],
    corrected: Dict[str, Dict[str, Dict[str, Optional[float]]]],
    save_path: Path,
    *,
    title: str,
    xlabel: str = "Frequency (MHz)",
    ylabel: str = "Metric (± jackknife SE)",
    marker: str = "o",
    capsize: float = 3,
    ingress_color: str = "tab:blue",
    egress_color: str = "tab:orange",
    diff_color: str = "tab:green",
    sum_color: str = "tab:red",
) -> None:
    import numpy as np

    freq_vals = []
    ingress = []
    ingress_err = []
    egress = []
    egress_err = []
    sinks = []
    sinks_err = []
    sums = []
    sums_err = []

    for freq_label, entry in corrected.items():
        meta = entry.get("meta", {})
        freq_vals.append(meta.get("freq"))

        m_in = entry.get("ingress", {}).get("metric")
        s_in = entry.get("ingress", {}).get("stderr")
        m_eg = entry.get("egress", {}).get("metric")
        s_eg = entry.get("egress", {}).get("stderr")

        ingress.append(m_in)
        ingress_err.append(None if s_in is None else abs(s_in))
        egress.append(m_eg)
        egress_err.append(None if s_eg is None else abs(s_eg))

        if m_in is not None and m_eg is not None:
            diff = m_in - m_eg
            sums_val = m_in + m_eg
        else:
            diff = None
            sums_val = None
        sinks.append(diff)
        sums.append(sums_val)

        if s_in is not None and s_eg is not None:
            err = float(np.sqrt(abs(s_in) ** 2 + abs(s_eg) ** 2))
        else:
            err = None
        sinks_err.append(err)
        sums_err.append(err)

    xs = np.array(freq_vals, dtype=float)
    order = np.argsort(xs)
    xs = xs[order]

    def reorder(values):
        arr = np.array([np.nan if v is None else v for v in values], dtype=float)
        return arr[order]

    ingress = reorder(ingress)
    ingress_err = reorder([None if v is None else abs(v) for v in ingress_err])
    egress = reorder(egress)
    egress_err = reorder([None if v is None else abs(v) for v in egress_err])
    sinks = reorder(sinks)
    sinks_err = reorder([None if v is None else abs(v) for v in sinks_err])
    sums = reorder(sums)
    sums_err = reorder([None if v is None else abs(v) for v in sums_err])

    plt.figure(figsize=(9, 5))
    plt.errorbar(xs, ingress, yerr=ingress_err, fmt=marker, capsize=capsize, color=ingress_color, label="Ingress")
    plt.plot(xs, ingress, "-", color=ingress_color)

    plt.errorbar(xs, egress, yerr=egress_err, fmt=marker, capsize=capsize, color=egress_color, label="Egress")
    plt.plot(xs, egress, "-", color=egress_color)

    plt.errorbar(xs, sinks, yerr=sinks_err, fmt=marker, capsize=capsize, color=diff_color, alpha=0.6, label="Ingress - Egress")
    plt.plot(xs, sinks, "-", color=diff_color, alpha=0.6)

    plt.errorbar(xs, sums, yerr=sums_err, fmt=marker, capsize=capsize, color=sum_color, alpha=0.6, label="Ingress + Egress")
    plt.plot(xs, sums, "-", color=sum_color, alpha=0.6)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def write_correction_curve(
    path: Path,
    correction: Dict[str, float],
    sun_rows: Dict[str, Dict[str, Dict[str, Optional[float]]]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["freq_label", "freq_mhz", "gain"])
        for freq_label in sorted(correction, key=lambda k: float(_safe_float(sun_rows[k]["meta"].get("freq"), default=np.nan)) if sun_rows[k]["meta"].get("freq") is not None else freq_label):
            meta = sun_rows[freq_label].get("meta", {})
            writer.writerow([freq_label, meta.get("freq"), correction[freq_label]])
    print(f"Wrote Sun correction curve -> {path}")


def _safe_float(value, default=np.nan):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    main()
