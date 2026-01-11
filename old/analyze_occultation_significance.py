#!/usr/bin/env python3
"""Empirical significance evaluation for occultation events.

Per AGENTS.md we keep the logic transparent so it can be run on an interactive
node (e.g. `salloc ...; conda activate luseepy_env` before invoking the script).
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

# Explicit orientation as requested: ingress=-1, egress=+1 (AGENTS instruction).
EVENT_ORIENTATION = {"ingress": -1.0, "egress": 1.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute directional/two-sided empirical p-values and "
            "Benjamini–Hochberg detections for occultation events."
        )
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("RAE2/outputs/random_occultations"),
        help="Directory that holds location and random_* JSON subdirectories.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="FDR control level for Benjamini–Hochberg.",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help=(
            "Optional path for a persistent CSV summary. Defaults to "
            "BASE_DIR/occultation_significance.csv."
        ),
    )
    return parser.parse_args()


def iter_objective_files(base_dir: Path) -> Sequence[Tuple[str, Path]]:
    """Yield (name, objective_path) pairs for each subdirectory."""
    pairs: List[Tuple[str, Path]] = []
    for entry in sorted(base_dir.iterdir()):
        if not entry.is_dir():
            continue
        candidate = entry / f"{entry.name}_objective.json"
        if candidate.exists():
            pairs.append((entry.name, candidate))
            continue
        # Fallback: grab the first *_objective.json if naming deviates.
        matches = sorted(entry.glob("*_objective.json"))
        if matches:
            pairs.append((entry.name, matches[0]))
    return pairs


def extract_event_arrays(
    rows: Sequence[dict], target_freq_labels: Optional[Sequence[str]]
) -> Tuple[List[str], List[np.ndarray], List[np.ndarray]]:
    """Return freq labels plus ordered metric/sigma arrays for ingress/egress."""
    freq_labels: List[str] = []
    delta_by_event = {"ingress": [], "egress": []}
    sigma_by_event = {"ingress": [], "egress": []}

    for row in rows:
        freq_labels.append(str(row.get("freq_label", row.get("freq"))))
        for key in ("ingress", "egress"):
            section = row.get(key)
            if section is None:
                raise ValueError(f"Missing '{key}' section in row {row}")
            delta_by_event[key].append(float(section["metric"]))
            sigma_by_event[key].append(float(section["stderr"]))

    if target_freq_labels is not None and list(target_freq_labels) != freq_labels:
        raise ValueError(
            "Frequency labels mismatch; ensure every JSON uses the same channels."
        )

    delta_arrays = [
        np.asarray(delta_by_event["ingress"], dtype=float),
        np.asarray(delta_by_event["egress"], dtype=float),
    ]
    sigma_arrays = [
        np.asarray(sigma_by_event["ingress"], dtype=float),
        np.asarray(sigma_by_event["egress"], dtype=float),
    ]

    return freq_labels, delta_arrays, sigma_arrays


def load_events(
    base_dir: Path, predicate: Callable[[str], bool]
) -> Tuple[List[str], List[str], np.ndarray, np.ndarray, np.ndarray]:
    """Collect event metadata/arrays for directories that satisfy predicate."""
    freq_labels: Optional[List[str]] = None
    names: List[str] = []
    event_types: List[float] = []
    deltas: List[np.ndarray] = []
    sigmas: List[np.ndarray] = []

    for name, obj_path in iter_objective_files(base_dir):
        if not predicate(name):
            continue
        with obj_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        rows = payload.get("rows")
        if not rows:
            continue
        freq_labels, delta_pair, sigma_pair = extract_event_arrays(rows, freq_labels)
        for evt_key, delta_vec, sigma_vec in zip(
            ("ingress", "egress"), delta_pair, sigma_pair
        ):
            names.append(f"{name}_{evt_key}")
            event_types.append(EVENT_ORIENTATION[evt_key])
            deltas.append(delta_vec)
            sigmas.append(sigma_vec)

    if not names:
        raise ValueError(f"No events matched predicate in {base_dir}")

    return (
        freq_labels or [],
        names,
        np.asarray(event_types, dtype=float),
        np.vstack(deltas),
        np.vstack(sigmas),
    )


def benjamini_hochberg(pvals: np.ndarray, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return BH q-values and detection mask."""
    flat = pvals.ravel()
    order = np.argsort(flat)
    ranked = flat[order]
    n = ranked.size
    adjusted = np.empty(n, dtype=float)
    running = 1.0
    for idx in range(n - 1, -1, -1):
        rank = idx + 1
        running = min(running, ranked[idx] * n / rank)
        adjusted[idx] = min(running, 1.0)
    qvals = np.empty_like(adjusted)
    qvals[order] = adjusted
    qvals = qvals.reshape(pvals.shape)
    detections = qvals <= alpha
    return qvals, detections


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Element-wise divide while guarding against zero denominators."""
    out = np.zeros_like(numerator, dtype=float)
    np.divide(
        numerator,
        denominator,
        out=out,
        where=np.asarray(denominator, dtype=float) != 0.0,
    )
    return out


def write_csv(
    path: Path,
    freq_labels: Sequence[str],
    event_names: Sequence[str],
    event_type: np.ndarray,
    delta: np.ndarray,
    sigma: np.ndarray,
    z: np.ndarray,
    u: np.ndarray,
    p_dir: np.ndarray,
    p_two: np.ndarray,
    p_flip: np.ndarray,
    q_dir: np.ndarray,
    q_flip: np.ndarray,
    detected_dir: np.ndarray,
    detected_flip: np.ndarray,
) -> None:
    """Persist per-event/channel statistics for downstream analysis."""
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "event",
        "freq_label",
        "event_type",
        "delta",
        "sigma",
        "z_score",
        "u_stat",
        "p_dir",
        "p_two",
        "p_flip",
        "q_dir",
        "q_flip",
        "detected_dir",
        "detected_flip",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        for evt_idx, event in enumerate(event_names):
            for ch_idx, freq in enumerate(freq_labels):
                writer.writerow(
                    {
                        "event": event,
                        "freq_label": freq,
                        "event_type": event_type[evt_idx],
                        "delta": delta[evt_idx, ch_idx],
                        "sigma": sigma[evt_idx, ch_idx],
                        "z_score": z[evt_idx, ch_idx],
                        # Reference instruction: include oriented stat used for p-values.
                        "u_stat": u[evt_idx, ch_idx],
                        "p_dir": p_dir[evt_idx, ch_idx],
                        "p_two": p_two[evt_idx, ch_idx],
                        "p_flip": p_flip[evt_idx, ch_idx],
                        "q_dir": q_dir[evt_idx, ch_idx],
                        "q_flip": q_flip[evt_idx, ch_idx],
                        "detected_dir": int(detected_dir[evt_idx, ch_idx]),
                        "detected_flip": int(detected_flip[evt_idx, ch_idx]),
                    }
                )


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()
    if not base_dir.exists():
        raise SystemExit(f"{base_dir} does not exist.")

    freq_labels, event_names, event_type, delta, sigma = load_events(
        base_dir, predicate=lambda name: not name.startswith("random_")
    )
    (
        freq_labels_rand,
        rand_names,
        event_type_rand,
        delta_rand,
        sigma_rand,
    ) = load_events(base_dir, predicate=lambda name: name.startswith("random_"))

    if freq_labels_rand != freq_labels:
        raise ValueError("Random and non-random files have different channel grids.")

    z = safe_divide(delta, sigma)
    z_rand = safe_divide(delta_rand, sigma_rand)

    # Orientation step per instructions: align ingress/egress signs before testing.
    u = event_type[:, None] * z
    u_rand = event_type_rand[:, None] * z_rand

    rand_count = u_rand.shape[0]
    comparator = rand_count + 1  # add-one smoothing for empirical p-values

    ge_counts = (u_rand[:, None, :] >= u[None, :, :]).sum(axis=0)
    p_dir = (1.0 + ge_counts) / comparator

    abs_counts = (np.abs(u_rand)[:, None, :] >= np.abs(u)[None, :, :]).sum(axis=0)
    p_two = (1.0 + abs_counts) / comparator

    le_counts = (u_rand[:, None, :] <= u[None, :, :]).sum(axis=0)
    p_flip = np.ones_like(p_dir)
    neg_mask = u < 0
    p_flip[neg_mask] = (1.0 + le_counts[neg_mask]) / comparator

    q_dir, detected_dir = benjamini_hochberg(p_dir, args.alpha)
    q_flip, detected_flip = benjamini_hochberg(p_flip, args.alpha)

    csv_path = args.csv_out or (base_dir / "occultation_significance.csv")
    write_csv(
        csv_path,
        freq_labels,
        event_names,
        event_type,
        delta,
        sigma,
        z,
        u,
        p_dir,
        p_two,
        p_flip,
        q_dir,
        q_flip,
        detected_dir,
        detected_flip,
    )

    np.set_printoptions(precision=4, suppress=True)
    print(f"Frequencies ({len(freq_labels)}): {freq_labels}")
    print(f"Events ({len(event_names)}): {event_names}")
    print("\nPer-event/channel arrays (events x channels):")
    print("p_dir:\n", p_dir)
    print("p_two:\n", p_two)
    print("p_flip:\n", p_flip)
    print("\nBenjamini–Hochberg results (alpha = {:.3f}):".format(args.alpha))
    print("q_dir:\n", q_dir)
    print("detected_dir:\n", detected_dir)
    print("q_flip:\n", q_flip)
    print("detected_flip:\n", detected_flip)
    print(f"\nCSV summary written to: {csv_path}")


if __name__ == "__main__":
    main()
