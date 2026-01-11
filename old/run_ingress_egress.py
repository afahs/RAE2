#!/usr/bin/env python3
"""Parallel ingress/egress analysis for configured static and solar targets.

As requested in AGENTS.md, this script mirrors run_occultation_random.py's
parallel ProcessPool layout and helper functions while extending the target
catalog to cover every static/null and solar-system source.
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

from ingress_egress_helper import (
    DEFAULT_DATA_PATH,
    SOLAR_SYSTEM_BODIES,
    compute_ingress_egress_objective_over_freqs,
    compute_solar_system_visibility,
    compute_static_source_visibility,
    convert_band_to_freq_mhz,
    has_histogram_content,
    load_occultation_dataframe,
    occultationStatisticsIngressEgressPairs,
    plotIngressEgressDiffHistograms,
    plotIngressEgressHistograms,
    plot_objective_vs_frequency,
)

NON_PLANETARY_SOURCES: Dict[str, SkyCoord] = {
    "CAS-A": SkyCoord(ra="23h23m26s", dec="+58d48m00s", frame="fk4", equinox="B1950", unit=(u.hourangle, u.deg)),
    "Cygnus-A": SkyCoord(ra="19h59m28s", dec="+40d44m02s", frame="fk4", equinox="B1950", unit=(u.hourangle, u.deg)),
    "Sag-A": SkyCoord(ra="17h45m40s", dec="-29d00m28s", frame="fk4", equinox="B1950", unit=(u.hourangle, u.deg)),
    "Virgo A": SkyCoord(ra="12h30m49.42338s", dec="+12d23m28.0439s", frame="fk4", equinox="B1950", unit=(u.hourangle, u.deg)),
    "Tau-A": SkyCoord(ra="05h34m31.97s", dec="+22d00m52.1s", frame="fk4", equinox="B1950", unit=(u.hourangle, u.deg)),
    "Crab Nebula": SkyCoord(ra="05h34m31.97s", dec="+22d00m52.1s", frame="fk4", equinox="B1950", unit=(u.hourangle, u.deg)),
    "Fornax-A": SkyCoord(ra="03h22m41s", dec="-37d12m30s", frame="fk4", equinox="B1950", unit=(u.hourangle, u.deg)),
}

NULL_SOURCES: Dict[str, SkyCoord] = {
    "null_galactic_north": SkyCoord(ra="12h51m26.3s", dec="+27d07m42s", frame="fk4", equinox="B1950", unit=(u.hourangle, u.deg)),
    "null_lockman_hole": SkyCoord(ra="10h45m00s", dec="+57d20m00s", frame="fk4", equinox="B1950", unit=(u.hourangle, u.deg)),
}

HISTOGRAM_PERCENT_BY_BAND = {
    1: 0.5,
    2: 0.1,
    3: 0.1,
    4: 0.05,
    5: 0.5,
    6: 0.3,
    7: 0.3,
    8: 0.05,
    9: 0.05,
}
DIFF_PERCENT_BY_BAND = {
    1: 0.6,
    2: 0.2,
    3: 0.1,
    4: 0.1,
    5: 0.6,
    6: 0.3,
    7: 0.3,
    8: 0.1,
    9: 0.1,
}
HISTOGRAM_PERCENT_DEFAULT = 0.3
DIFF_PERCENT_DEFAULT = 0.3

EXCLUDED_SOLAR_SYSTEM_BODIES = {"Moon", "Pluto"}
SOURCE_GUARD_BODIES = ()#  ("Sun", "Earth")
SOURCE_GUARD_LIMB_DEGREES = 0.0
MOON_RADIUS_KM = 1740.0
MOON_WIDTH_COL = "_moon_half_width_deg"


@dataclass(frozen=True)
class TargetSpec:
    name: str
    kind: str  # "static" or "solar"
    coord: Optional[SkyCoord] = None


def slugify(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


def normalize(name: str) -> str:
    return name.lower().replace(" ", "").replace("-", "")


def _to_serializable_list(values: Sequence) -> List:
    output: List = []
    for value in values:
        if pd.isna(value):
            output.append(None)
        else:
            output.append(float(value))
    return output


def stats_to_serializable(stats: Dict) -> Dict:
    serializable = {}
    for freq_key, payload in stats.items():
        serializable[str(freq_key)] = {
            label: {name: _to_serializable_list(values) for name, values in components.items()}
            for label, components in payload.items()
        }
    return serializable


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


def _clean_results_for_json(results: Dict) -> Dict:
    cleaned = {"rows": []}
    for row in results.get("rows", []):
        entry = {
            "freq": _to_float(row.get("freq")),
            "freq_label": _to_float(row.get("freq_label")),
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


def freq_label_for_files(freq_band, freq_mhz) -> str:
    try:
        band_int = int(round(float(freq_band)))
    except Exception:
        band_int = None
    if freq_mhz is not None and freq_mhz >= 0:
        base = f"{freq_mhz:.2f}MHz".replace(".", "p")
        if band_int is not None:
            return f"band{band_int:02d}_{base}"
        return f"freq_{base}"
    if band_int is not None:
        return f"band{band_int:02d}"
    return f"band_{freq_band}"


def add_static_source_visibility(data: pd.DataFrame, source_name: str, coord: SkyCoord) -> str:
    angle_series, vis_series = compute_static_source_visibility(data, source_name, coord)
    data[angle_series.name] = angle_series
    data[vis_series.name] = vis_series
    return vis_series.name


def sort_frequency_keys(keys: Iterable) -> List:
    try:
        return sorted(keys, key=lambda k: float(k))
    except Exception:
        return sorted(keys, key=str)


def _moon_half_width_series(data: pd.DataFrame) -> pd.Series:
    existing = data.get(MOON_WIDTH_COL)
    if isinstance(existing, pd.Series):
        return existing

    pos_x = np.asarray(data["position_x"], dtype=float)
    pos_y = np.asarray(data["position_y"], dtype=float)
    pos_z = np.asarray(data["position_z"], dtype=float)
    moon_dist = np.sqrt(pos_x**2 + pos_y**2 + pos_z**2)
    moon_width = np.degrees(np.tan(MOON_RADIUS_KM / moon_dist))
    series = pd.Series(moon_width, index=data.index, name=MOON_WIDTH_COL)
    data[MOON_WIDTH_COL] = series
    return series


def _ensure_angle_series(data: pd.DataFrame, body_name: str) -> pd.Series:
    angle_col = f"{body_name}Angle"
    existing = data.get(angle_col)
    if isinstance(existing, pd.Series):
        return existing
    angle_series, vis_series = compute_solar_system_visibility(data, body_name)
    data[angle_series.name] = angle_series
    data[vis_series.name] = vis_series
    return angle_series


def _build_source_guard_mask(data: pd.DataFrame, target_name: str) -> Optional[pd.Series]:
    if normalize(target_name) in {"sun", "earth"}:
        return None

    limb_half_width = _moon_half_width_series(data)
    guard_mask = pd.Series(False, index=data.index, dtype=bool)

    for body in SOURCE_GUARD_BODIES:
        angle_series = _ensure_angle_series(data, body)
        limb_distance = angle_series.abs() - limb_half_width
        proximity_mask = (limb_distance < SOURCE_GUARD_LIMB_DEGREES).fillna(False)
        guard_mask |= proximity_mask

    # Source guarding per instructions: drop samples when Sun/Earth graze the lunar limb.
    return guard_mask


def _freq_to_band_int(freq_key) -> Optional[int]:
    try:
        return int(round(float(freq_key)))
    except Exception:
        return None


def _prepare_visibility_for_target(data: pd.DataFrame, spec: TargetSpec) -> Tuple[str, Optional[pd.Series]]:
    if spec.kind == "static":
        if spec.coord is None:
            raise ValueError(f"Static target {spec.name} is missing coordinates.")
        vis_col = add_static_source_visibility(data, spec.name, spec.coord)
        guard_mask = _build_source_guard_mask(data, spec.name)
        return vis_col, guard_mask

    if spec.kind == "solar":
        angle_series, vis_series = compute_solar_system_visibility(data, spec.name)
        data[angle_series.name] = angle_series
        data[vis_series.name] = vis_series
        guard_mask = _build_source_guard_mask(data, spec.name)
        return vis_series.name, guard_mask

    raise ValueError(f"Unknown target kind: {spec.kind}")


def process_target(
    spec: TargetSpec,
    *,
    data,
    data_path: Union[str, Path],
    start: str,
    end: str,
    output_root: Path,
    window_minutes: float,
    antenna: str,
    aggregate: str,
    use_std_weights: bool,
    show_progress: bool,
) -> Dict:
    source_dir = output_root / slugify(spec.name)
    source_dir.mkdir(parents=True, exist_ok=True)

    if data is None:
        data = load_occultation_dataframe(data_path, start=start, end=end)
        data.sort_index(inplace=True)
    else:
        data = data.copy()

    print(f"Processing {spec.name}")
    vis_col, exclude_mask = _prepare_visibility_for_target(data, spec)

    stats, filter_counts = occultationStatisticsIngressEgressPairs(
        data,
        col=vis_col,
        window=pd.Timedelta(minutes=window_minutes),
        antenn=antenna,
        progress=show_progress,
        exclude_mask=exclude_mask,
        return_counts=True,
    )

    usable_stats: Dict[Union[int, float], Dict] = {}
    agg_min_bins: List[float] = []
    freq_keys = sort_frequency_keys(stats.keys())

    for freq_key in freq_keys:
        freq_stats = stats[freq_key]
        if not has_histogram_content(freq_stats):
            print(f"  Skipping band {freq_key} for {spec.name}: no usable data")
            continue

        band_int = _freq_to_band_int(freq_key)
        hist_pct = HISTOGRAM_PERCENT_BY_BAND.get(band_int, HISTOGRAM_PERCENT_DEFAULT)
        diff_pct = DIFF_PERCENT_BY_BAND.get(band_int, DIFF_PERCENT_DEFAULT)

        freq_mhz = convert_band_to_freq_mhz(freq_key)
        label_for_title = f"{freq_mhz:.2f} MHz" if freq_mhz is not None else str(freq_key)
        freq_slug = freq_label_for_files(freq_key, freq_mhz)

        band_dir = source_dir / freq_slug
        band_dir.mkdir(parents=True, exist_ok=True)

        hist_path = band_dir / f"{slugify(spec.name)}_{freq_slug}_ingress_egress.png"
        diff_path = band_dir / f"{slugify(spec.name)}_{freq_slug}_diff.png"

        plotIngressEgressHistograms(
            freq_stats,
            use_std_weights=use_std_weights,
            min_bin_percentage=hist_pct,
            suptitle=f"{spec.name} – {label_for_title}",
            save_path=str(hist_path),
            show=False,
        )
        plotIngressEgressDiffHistograms(
            freq_stats,
            use_std_weights=use_std_weights,
            min_bin_percentage=diff_pct,
            suptitle=f"{spec.name} – {label_for_title}",
            save_path=str(diff_path),
            show=False,
        )

        usable_stats[freq_key] = freq_stats
        agg_min_bins.append(diff_pct)

    filters_json_path = source_dir / f"{slugify(spec.name)}_filters.json"
    filters_json_path.write_text(json.dumps(filter_counts, indent=2), encoding="utf-8")

    if not usable_stats:
        print(f"  No valid ingress/egress pairs for {spec.name}; skipping objective plot")
        filtered_total = filter_counts.get("filtered_candidates", 0)
        print(
            f"  Filtered candidates: {filtered_total} "
            f"(source_guard={filter_counts.get('guard_filtered_pairs', 0)}, "
            f"missing_data={filter_counts.get('missing_data_events', 0)}, "
            f"pairs={filter_counts.get('total_pairs', 0)})"
        )
        return {"name": spec.name, "results": {}, "filters": filter_counts}

    results = compute_ingress_egress_objective_over_freqs(
        usable_stats,
        use_std_weights=use_std_weights,
        min_bin_percentage=agg_min_bins,
        aggregate=aggregate,
    )

    stats_json_path = source_dir / f"{slugify(spec.name)}_stats.json"
    stats_json_path.write_text(json.dumps(stats_to_serializable(usable_stats), indent=2), encoding="utf-8")

    objective_path = source_dir / f"{slugify(spec.name)}_objective_vs_frequency.png"
    objective_json_path = source_dir / f"{slugify(spec.name)}_objective.json"

    plot_objective_vs_frequency(
        results,
        save_path=str(objective_path),
        show=False,
    )

    payload = _clean_results_for_json(results)
    objective_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    filtered_total = filter_counts.get("filtered_candidates", 0)
    print(
        f"  Filtered candidates: {filtered_total} "
        f"(source_guard={filter_counts.get('guard_filtered_pairs', 0)}, "
        f"missing_data={filter_counts.get('missing_data_events', 0)}, "
        f"pairs={filter_counts.get('total_pairs', 0)})"
    )
    print(f"  Finished {spec.name} -> {source_dir}")
    return {"name": spec.name, "results": payload, "filters": filter_counts}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ingress/egress histograms for multiple sources.")
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH), help="Path to the interpolated master CSV")
    parser.add_argument("--output-dir", default="RAE2/outputs/ingress_egress", help="Directory for generated plots")
    parser.add_argument("--window-minutes", type=float, default=2.0, help="Half-width (minutes) for ingress/egress windows")
    parser.add_argument("--antenna", default="rv2_coarse", help="Antenna column to analyse")
    parser.add_argument("--aggregate", choices=["sum", "mean", "weighted_sum", "weighted_mean"], default="weighted_sum")
    parser.add_argument("--no-std-weights", action="store_true", help="Disable inverse-std weighting in the metrics")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")
    parser.add_argument("--start", default="1974-01-01 14:00", help="Start timestamp (inclusive)")
    parser.add_argument("--end", default="1975-12-31 16:00", help="End timestamp (inclusive)")
    parser.add_argument("--sources", nargs="*", help="Optional subset of extra-galactic sources (case-insensitive)")
    parser.add_argument("--include-fornax", action="store_true", help="Also process Fornax-A (active in the notebook)")
    parser.add_argument("--solar-system", action="store_true", help="Also generate plots for predefined solar-system bodies")
    parser.add_argument("--bodies", nargs="*", help="Subset of solar-system bodies (requires --solar-system)")
    parser.add_argument("--all", action="store_true", help="Process all static sources and planets (excludes Moon/Pluto)")
    parser.add_argument("--null", action="store_true", help="Add null reference directions (galactic north pole and Lockman Hole)")
    parser.add_argument("--max-workers", type=int, help="Max concurrent processes (mirrors run_occultation_random.py parallelism)")
    return parser.parse_args()


def _build_target_specs(args: argparse.Namespace) -> List[TargetSpec]:
    selected_sources = dict(NON_PLANETARY_SOURCES)
    if args.include_fornax and "Fornax-A" not in selected_sources:
        selected_sources["Fornax-A"] = NON_PLANETARY_SOURCES["Fornax-A"]

    if args.null:
        selected_sources.update(NULL_SOURCES)

    if args.sources:
        lookup = {normalize(name): name for name in selected_sources.keys()}
        filtered: Dict[str, SkyCoord] = {}
        for raw in args.sources:
            key = normalize(raw)
            if key not in lookup:
                raise SystemExit(f"Unknown source requested: {raw}")
            canonical = lookup[key]
            filtered[canonical] = selected_sources[canonical]
        selected_sources = filtered

    if args.bodies and not args.solar_system:
        raise SystemExit("--bodies requires --solar-system")

    if args.all:
        args.solar_system = True

    specs: List[TargetSpec] = [TargetSpec(name=name, kind="static", coord=coord) for name, coord in selected_sources.items()]

    if args.solar_system:
        body_lookup = {normalize(name): name for name in SOLAR_SYSTEM_BODIES}
        if args.bodies:
            requested = []
            for raw in args.bodies:
                key = normalize(raw)
                if key not in body_lookup:
                    raise SystemExit(f"Unknown solar-system body requested: {raw}")
                requested.append(body_lookup[key])
            solar_targets = requested
        else:
            solar_targets = list(SOLAR_SYSTEM_BODIES)
        for body in solar_targets:
            if body in EXCLUDED_SOLAR_SYSTEM_BODIES:
                continue
            specs.append(
                TargetSpec(
                    name=body,
                    kind="solar",
                    coord=None,
                )
            )

    if not specs:
        raise SystemExit("No sources selected")

    return specs


def main() -> None:
    args = parse_args()
    specs = _build_target_specs(args)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    use_std_weights = not args.no_std_weights
    show_progress = not args.no_progress
    sequential = args.max_workers == 1 or len(specs) == 1

    if sequential:
        data = load_occultation_dataframe(args.data_path, start=args.start, end=args.end)
        data.sort_index(inplace=True)
        for spec in specs:
            process_target(
                spec,
                data=data,
                data_path=args.data_path,
                start=args.start,
                end=args.end,
                output_root=output_dir,
                window_minutes=args.window_minutes,
                antenna=args.antenna,
                aggregate=args.aggregate,
                use_std_weights=use_std_weights,
                show_progress=show_progress,
            )
        return

    # Parallelize per AGENTS.md by mirroring run_occultation_random.py's ProcessPool layout.
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(
                process_target,
                spec,
                data=None,
                data_path=args.data_path,
                start=args.start,
                end=args.end,
                output_root=output_dir,
                window_minutes=args.window_minutes,
                antenna=args.antenna,
                aggregate=args.aggregate,
                use_std_weights=use_std_weights,
                show_progress=show_progress,
            )
            for spec in specs
        ]
        for future in as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()
