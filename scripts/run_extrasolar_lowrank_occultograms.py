#!/usr/bin/env python
"""Build event-level low-rank foreground-residual occultograms for fixed sources.

The foreground model uses the 0.70, 0.90, 1.31, and 2.20 MHz bands and is
trained without the lunar block to which it is applied. Reappearances are
time-reversed and the source edge remains fixed at t=0. Antennas are analysed
separately and are not required to agree.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from rylevonberg.occultogram import fold_occultogram_points  # noqa: E402
from rylevonberg.table_io import write_table  # noqa: E402
from run_earth_lowrank_covariance import (  # noqa: E402
    DISPLAY_HALF_WINDOW_SECONDS,
    FIT_HALF_WINDOW_SECONDS,
    TRACER_FREQUENCIES,
    _available_frequencies,
    _block_profiles,
    _edge_amplitude,
    _fit_factor_family,
    _select_factor_rank,
    _wide_profiles,
)


DEFAULT_PROFILES = Path(
    "/pscratch/sd/a/afahs/RAE2/occultograms_morphology_v1/occultogram_event_profiles.parquet"
)
DEFAULT_OUT = Path("/pscratch/sd/a/afahs/RAE2/extrasolar_lowrank_occultograms_v1")
DEFAULT_SOURCES = ("fornax_a", "vir_a")
SOURCE_LABELS = {"fornax_a": "Fornax A", "vir_a": "Virgo A"}


def _physical_event_columns(points: pd.DataFrame) -> pd.DataFrame:
    out = points.copy()
    parts = out["event_uid"].astype(str).str.split("|", expand=True, regex=False)
    if parts.shape[1] < 3:
        raise RuntimeError("event_uid does not contain the expected source|time|type fields")
    out["physical_event_uid"] = parts[0] + "|" + parts[1] + "|" + parts[2]
    out["physical_event_time"] = pd.to_datetime(parts[1], errors="coerce")
    if out["physical_event_time"].isna().any():
        raise RuntimeError("one or more physical event timestamps could not be parsed")
    return out


def _event_wide(points: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    points = _physical_event_columns(points)
    keys = [
        "physical_event_uid",
        "physical_event_time",
        "lunar_block",
        "original_event_type",
        "antenna",
        "control_shift_s",
        "t_bin_sec",
        "frequency_mhz",
    ]
    profiles = (
        points.groupby(keys, observed=True, sort=False)
        .agg(value=("normalized_power", "median"))
        .reset_index()
    )
    event_metadata = (
        profiles.loc[
            :,
            [
                "physical_event_uid",
                "physical_event_time",
                "lunar_block",
                "original_event_type",
            ],
        ]
        .drop_duplicates("physical_event_uid")
        .sort_values(["physical_event_time", "original_event_type"])
        .reset_index(drop=True)
    )
    event_metadata.insert(0, "event_index", np.arange(len(event_metadata), dtype=np.int32))
    profiles = profiles.merge(
        event_metadata.loc[:, ["physical_event_uid", "event_index"]],
        on="physical_event_uid",
        how="left",
        validate="many_to_one",
    )
    wide = profiles.pivot_table(
        index=[
            "event_index",
            "lunar_block",
            "original_event_type",
            "antenna",
            "control_shift_s",
            "t_bin_sec",
        ],
        columns="frequency_mhz",
        values="value",
        aggfunc="median",
    ).reset_index()
    wide.columns.name = None
    wide = wide.sort_values(
        ["event_index", "antenna", "control_shift_s", "t_bin_sec"]
    ).reset_index(drop=True)
    return wide, event_metadata


def _model_parameter_frame(
    source: str,
    block: int,
    antenna: str,
    family: dict[float, object],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for target_frequency, model in family.items():
        for tracer_index, tracer_frequency in enumerate(model.tracer_frequencies):
            for factor_index in range(model.loadings.shape[1]):
                records.append(
                    {
                        "source_name": source,
                        "held_out_lunar_block": block,
                        "antenna": antenna,
                        "target_frequency_mhz": float(target_frequency),
                        "selected_rank": int(model.rank),
                        "tracer_frequency_mhz": float(tracer_frequency),
                        "tracer_center": float(model.center[tracer_index]),
                        "tracer_scale": float(model.scale[tracer_index]),
                        "factor_index": int(factor_index),
                        "factor_loading": float(model.loadings[tracer_index, factor_index]),
                        "regression_intercept": float(model.regression[0]),
                        "factor_regression_coefficient": float(
                            model.regression[factor_index + 1]
                        ),
                    }
                )
    return pd.DataFrame.from_records(records)


def _crossfit_event_residuals(
    source: str,
    block_wide: pd.DataFrame,
    event_wide: pd.DataFrame,
    frequencies: tuple[float, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    blocks = sorted(int(value) for value in pd.unique(event_wide["lunar_block"]))
    antennas = sorted(str(value) for value in pd.unique(event_wide["antenna"]))
    residual_frames: list[pd.DataFrame] = []
    rank_records: list[dict[str, object]] = []
    model_frames: list[pd.DataFrame] = []
    for block in blocks:
        training_blocks = [candidate for candidate in blocks if candidate != block]
        for antenna in antennas:
            rank, losses = _select_factor_rank(
                block_wide, training_blocks, antenna, frequencies
            )
            for record in losses:
                rank_records.append(
                    {
                        "source_name": source,
                        "held_out_lunar_block": block,
                        "antenna": antenna,
                        **record,
                        "selected": bool(record["rank"] == rank),
                    }
                )
            family = _fit_factor_family(
                event_wide, training_blocks, antenna, rank, frequencies
            )
            model_frames.append(_model_parameter_frame(source, block, antenna, family))
            current = event_wide[
                event_wide["lunar_block"].eq(block)
                & event_wide["antenna"].astype(str).eq(antenna)
                & np.isclose(event_wide["control_shift_s"], 0.0)
                & (np.abs(event_wide["t_bin_sec"]) <= DISPLAY_HALF_WINDOW_SECONDS)
            ].copy()
            metadata = current.loc[
                :,
                [
                    "event_index",
                    "lunar_block",
                    "original_event_type",
                    "antenna",
                    "t_bin_sec",
                ],
            ]
            for frequency in frequencies:
                prediction = family[frequency].predict(current)
                frame = metadata.copy()
                frame["source_name"] = source
                frame["frequency_mhz"] = float(frequency)
                frame["raw_value"] = current[frequency].to_numpy(dtype=float)
                frame["foreground_prediction"] = prediction
                frame["residual_value"] = frame["raw_value"] - prediction
                frame["available_tracer_count"] = np.sum(
                    np.isfinite(
                        current.loc[
                            :, list(family[frequency].tracer_frequencies)
                        ].to_numpy(dtype=float)
                    ),
                    axis=1,
                ).astype(np.int8)
                frame["selected_rank"] = int(rank)
                residual_frames.append(frame)
    return (
        pd.concat(residual_frames, ignore_index=True),
        pd.DataFrame.from_records(rank_records),
        pd.concat(model_frames, ignore_index=True),
    )


def _event_edge_amplitudes(profiles: pd.DataFrame) -> pd.DataFrame:
    use = profiles[np.abs(profiles["t_bin_sec"]) <= FIT_HALF_WINDOW_SECONDS]
    keys = [
        "source_name",
        "event_index",
        "lunar_block",
        "original_event_type",
        "antenna",
        "frequency_mhz",
    ]
    records: list[dict[str, object]] = []
    for group_key, group in use.groupby(keys, observed=True, sort=False):
        record = dict(zip(keys, group_key if isinstance(group_key, tuple) else (group_key,)))
        times = group["t_bin_sec"].to_numpy(dtype=float)
        record["raw_edge_amplitude"] = _edge_amplitude(
            times, group["raw_value"].to_numpy(dtype=float)
        )
        record["residual_edge_amplitude"] = _edge_amplitude(
            times, group["residual_value"].to_numpy(dtype=float)
        )
        records.append(record)
    return pd.DataFrame.from_records(records)


def _edge_summary(amplitudes: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    keys = ["source_name", "antenna", "frequency_mhz"]
    for group_key, group in amplitudes.groupby(keys, observed=True, sort=True):
        source, antenna, frequency = group_key
        for data_kind, column in (
            ("raw", "raw_edge_amplitude"),
            ("lowrank_residual", "residual_edge_amplitude"),
        ):
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            block_values = (
                group.assign(_value=pd.to_numeric(group[column], errors="coerce"))
                .groupby("lunar_block", observed=True)["_value"]
                .median()
                .dropna()
            )
            records.append(
                {
                    "source_name": str(source),
                    "antenna": str(antenna),
                    "frequency_mhz": float(frequency),
                    "data_kind": data_kind,
                    "event_median": float(values.median()),
                    "event_q16": float(values.quantile(0.16)),
                    "event_q84": float(values.quantile(0.84)),
                    "correct_sign_event_fraction": float(np.mean(values > 0.0)),
                    "block_median": float(block_values.median()),
                    "block_q16": float(block_values.quantile(0.16)),
                    "block_q84": float(block_values.quantile(0.84)),
                    "n_events": int(values.size),
                    "n_lunar_blocks": int(block_values.size),
                }
            )
    return pd.DataFrame.from_records(records)


def _tracer_coverage_summary(profiles: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    keys = ["source_name", "antenna", "frequency_mhz"]
    for group_key, group in profiles.groupby(keys, observed=True, sort=True):
        source, antenna, frequency = group_key
        counts = pd.to_numeric(group["available_tracer_count"], errors="coerce").dropna()
        possible = 3 if any(np.isclose(float(frequency), tracer) for tracer in TRACER_FREQUENCIES) else 4
        records.append(
            {
                "source_name": str(source),
                "antenna": str(antenna),
                "frequency_mhz": float(frequency),
                "possible_tracer_count": int(possible),
                "median_available_tracers": float(counts.median()),
                "fraction_with_any_tracer": float(np.mean(counts >= 1)),
                "fraction_with_two_or_more_tracers": float(np.mean(counts >= 2)),
                "fraction_with_all_tracers": float(np.mean(counts >= possible)),
                "n_target_bins": int(counts.size),
            }
        )
    return pd.DataFrame.from_records(records)


def _plot_occultograms(
    source: str,
    profiles: pd.DataFrame,
    frequencies: tuple[float, ...],
    output_dir: Path,
) -> list[Path]:
    label = SOURCE_LABELS.get(source, source.replace("_", " ").title())
    paths: list[Path] = []
    for antenna in sorted(profiles["antenna"].astype(str).unique()):
        antenna_profiles = profiles[profiles["antenna"].astype(str).eq(antenna)]
        event_indices = np.asarray(sorted(int(value) for value in antenna_profiles["event_index"].unique()))
        fig, axes = plt.subplots(
            len(frequencies), 2, figsize=(13, 2.15 * len(frequencies)), constrained_layout=True
        )
        for row, frequency in enumerate(frequencies):
            current = antenna_profiles[np.isclose(antenna_profiles["frequency_mhz"], frequency)]
            raw_grid = current.pivot_table(
                index="event_index", columns="t_bin_sec", values="raw_value", aggfunc="median"
            ).reindex(event_indices)
            residual_grid = current.pivot_table(
                index="event_index",
                columns="t_bin_sec",
                values="residual_value",
                aggfunc="median",
            ).reindex(index=event_indices, columns=raw_grid.columns)
            combined = np.concatenate(
                [raw_grid.to_numpy(dtype=float).ravel(), residual_grid.to_numpy(dtype=float).ravel()]
            )
            finite = combined[np.isfinite(combined)]
            limit = float(np.quantile(np.abs(finite), 0.98)) if finite.size else 1.0
            limit = max(limit, 1e-6)
            extent = [
                float(raw_grid.columns.min()),
                float(raw_grid.columns.max()),
                len(event_indices) - 0.5,
                -0.5,
            ]
            for column, (grid, title) in enumerate(
                ((raw_grid, "raw"), (residual_grid, "low-rank residual"))
            ):
                ax = axes[row, column]
                ax.imshow(
                    grid.to_numpy(dtype=float),
                    aspect="auto",
                    interpolation="nearest",
                    cmap="RdBu_r",
                    vmin=-limit,
                    vmax=limit,
                    extent=extent,
                    rasterized=True,
                )
                ax.axvline(0.0, color="black", linestyle="--", linewidth=0.8)
                if column == 0:
                    ax.set_ylabel(f"{frequency:.2f} MHz\n{len(event_indices)} events")
                ax.set_yticks([])
                if row == 0:
                    ax.set_title(title)
                if row == len(frequencies) - 1:
                    ax.set_xlabel("time from fixed edge (s)")
                else:
                    ax.set_xticklabels([])
        fig.suptitle(
            f"{label} — {antenna}: event-level cross-fitted foreground subtraction",
            fontsize=14,
        )
        path = output_dir / f"{source}_{antenna}_lowrank_occultograms.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def _plot_profiles(
    source: str,
    profiles: pd.DataFrame,
    frequencies: tuple[float, ...],
    output_dir: Path,
) -> list[Path]:
    label = SOURCE_LABELS.get(source, source.replace("_", " ").title())
    paths: list[Path] = []
    for antenna in sorted(profiles["antenna"].astype(str).unique()):
        antenna_profiles = profiles[profiles["antenna"].astype(str).eq(antenna)]
        fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharex=True, constrained_layout=True)
        for ax, frequency in zip(axes.ravel(), frequencies):
            current = antenna_profiles[np.isclose(antenna_profiles["frequency_mhz"], frequency)]
            raw = current.groupby("t_bin_sec", observed=True)["raw_value"].median().sort_index()
            residual_group = current.groupby("t_bin_sec", observed=True)["residual_value"]
            residual = residual_group.median().sort_index()
            q16 = residual_group.quantile(0.16).reindex(residual.index)
            q84 = residual_group.quantile(0.84).reindex(residual.index)
            times = residual.index.to_numpy(dtype=float)
            ax.plot(raw.index, raw.values, color="0.55", linewidth=1.1, label="raw")
            ax.plot(times, residual.values, color="#7b2cbf", linewidth=1.7, label="low-rank residual")
            ax.fill_between(times, q16.values, q84.values, color="#7b2cbf", alpha=0.14)
            ax.axvline(0.0, color="black", linestyle="--", linewidth=0.8)
            ax.axhline(0.0, color="0.4", linewidth=0.6)
            ax.set_title(f"{frequency:.2f} MHz")
            ax.grid(alpha=0.17)
        axes[0, 0].legend(frameon=False)
        for ax in axes[-1, :]:
            ax.set_xlabel("time from fixed edge (s)")
        for ax in axes[:, 0]:
            ax.set_ylabel("normalized power")
        fig.suptitle(f"{label} — {antenna}: raw and foreground-residual profiles")
        path = output_dir / f"{source}_{antenna}_lowrank_profiles.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return json.loads(frame.to_json(orient="records"))


def _run_source(source: str, profiles_path: Path, output_root: Path) -> dict[str, object]:
    output_dir = output_root / source
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "source_name",
        "event_uid",
        "event_order",
        "lunar_block",
        "event_type",
        "frequency_mhz",
        "antenna",
        "control_shift_s",
        "t_bin_sec",
        "normalized_power",
    ]
    points = pd.read_parquet(
        profiles_path, columns=columns, filters=[("source_name", "==", source)]
    )
    if points.empty:
        raise RuntimeError(f"no occultogram profiles found for {source}")
    points = points[np.isfinite(pd.to_numeric(points["normalized_power"], errors="coerce"))].copy()
    points = fold_occultogram_points(points)
    block_wide = _wide_profiles(_block_profiles(points))
    event_wide, event_metadata = _event_wide(points)
    frequencies = _available_frequencies(block_wide)
    missing = [
        frequency
        for frequency in TRACER_FREQUENCIES
        if not any(np.isclose(frequency, available) for available in frequencies)
    ]
    if missing:
        raise RuntimeError(f"{source} is missing foreground tracer frequencies: {missing}")
    blocks = sorted(int(value) for value in pd.unique(event_wide["lunar_block"]))
    if len(blocks) < 5:
        raise RuntimeError(f"{source} has only {len(blocks)} lunar blocks; nested cross-fitting requires five")

    residuals, rank_selection, model_parameters = _crossfit_event_residuals(
        source, block_wide, event_wide, frequencies
    )
    amplitudes = _event_edge_amplitudes(residuals)
    summary = _edge_summary(amplitudes)
    tracer_coverage = _tracer_coverage_summary(residuals)

    write_table(event_metadata, output_dir / "event_metadata.parquet", index=False)
    write_table(residuals, output_dir / "lowrank_event_profiles.parquet", index=False)
    write_table(rank_selection, output_dir / "lowrank_rank_selection.parquet", index=False)
    write_table(model_parameters, output_dir / "lowrank_model_parameters.parquet", index=False)
    write_table(amplitudes, output_dir / "event_edge_amplitudes.parquet", index=False)
    write_table(summary, output_dir / "edge_summary.parquet", index=False)
    write_table(tracer_coverage, output_dir / "tracer_coverage_summary.parquet", index=False)

    plot_paths = []
    plot_paths.extend(_plot_occultograms(source, residuals, frequencies, output_dir))
    plot_paths.extend(_plot_profiles(source, residuals, frequencies, output_dir))
    visual_data = {
        "config": {
            "source_name": source,
            "source_label": SOURCE_LABELS.get(source, source),
            "edge_time_s": 0.0,
            "antennas_fit_separately": True,
            "antenna_consistency_required": False,
            "crossfit_unit": "29.53059-day lunar block",
            "foreground_tracer_frequencies_mhz": TRACER_FREQUENCIES,
            "display_half_window_s": DISPLAY_HALF_WINDOW_SECONDS,
        },
        "edge_summary": _records(summary),
        "tracer_coverage": _records(tracer_coverage),
        "rank_selection": _records(rank_selection),
    }
    (output_dir / "visual_data.json").write_text(
        json.dumps(visual_data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "source_name": source,
        "source_label": SOURCE_LABELS.get(source, source),
        "input_profile_rows": int(len(points)),
        "physical_events": int(event_metadata["event_index"].nunique()),
        "lunar_blocks": len(blocks),
        "antennas": sorted(str(value) for value in pd.unique(event_wide["antenna"])),
        "frequencies_mhz": frequencies,
        "edge_time_fixed_seconds": 0.0,
        "crossfit_unit": "lunar_block",
        "antennas_fit_separately": True,
        "generated_plots": [path.name for path in plot_paths],
        "claim_status": "foreground-subtraction visual experiment; no detection claim",
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sources", nargs="+", default=list(DEFAULT_SOURCES))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifests = [
        _run_source(source, args.profiles, args.out_dir) for source in args.sources
    ]
    root_manifest = {
        "profiles": str(args.profiles),
        "output_directory": str(args.out_dir),
        "sources": manifests,
    }
    (args.out_dir / "run_manifest.json").write_text(
        json.dumps(root_manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(root_manifest, indent=2))


if __name__ == "__main__":
    main()
