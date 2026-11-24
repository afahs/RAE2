#!/usr/bin/env python
"""Stream 3: test the external Jovian latitude-beaming prediction.

Alexander et al. (1979) showed that roughly 1-MHz Jupiter occurrence curves
change strongly with observer jovigraphic latitude and reproduced the effect
with a narrow, warped emission sheet.  This stream computes RAE-2's actual
sub-observer latitude with SPICE, applies the paper's 9.6-degree tilted-dipole
geometry, and tests a fixed 10-degree-wide magnetic-latitude beam.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.jupiter_external_stream_common import (
    COLORS,
    DETECTORS,
    LABELS,
    band_gate,
    daily_roll_plan,
    detector_source,
    familywise_upper_p,
    log_rate_ratio,
    rolled_vectors,
    upper_tail_p,
)


DEFAULT_EVENTS = Path(
    "/pscratch/sd/a/afahs/RAE2/jupiter_burst_gated_phase_occurrence_v1/"
    "five_minute_channel_percentile_events.parquet"
)
DEFAULT_KERNELS = Path(
    "/pscratch/sd/a/afahs/RAE2/jupiter_phase_occurrence_svd_v1/spice_kernels"
)
DEFAULT_OUTPUT = Path("/pscratch/sd/a/afahs/RAE2/jupiter_external_stream3_latitude_v1")

PRIMARY_BANDS = (2, 3, 4)  # 0.70, 0.90, 1.31 MHz around the paper's ~1 MHz result.
PRIMARY_QUANTILE = 0.99
DIPOLE_TILT_DEG = 9.6
DIPOLE_POLE_WEST_LONGITUDE_DEG = 202.0
BEAM_CENTER_MAGNETIC_LATITUDE_DEG = 3.0
BEAM_HALF_WIDTH_DEG = 5.0


def furnish_kernels(kernel_dir: Path) -> None:
    import spiceypy as spice

    spice.kclear()
    for name in ("naif0012.tls", "pck00010.tpc", "jup100.bsp", "jup348.bsp"):
        path = kernel_dir / name
        if path.exists():
            spice.furnsh(str(path))
    if spice.ktotal("ALL") == 0:
        raise FileNotFoundError(f"No SPICE kernels furnished from {kernel_dir}")


def observer_latitude(times: pd.Series, kernel_dir: Path) -> pd.DataFrame:
    """Jovigraphic latitude of Earth as seen from Jupiter at emission time."""

    import spiceypy as spice

    furnish_kernels(kernel_dir)
    rows = []
    unique_times = pd.to_datetime(times).drop_duplicates().sort_values()
    for value in unique_times:
        utc = value.strftime("%Y-%m-%dT%H:%M:%S.%f")
        receive_et = spice.str2et(utc)
        earth_to_jupiter, light_time = spice.spkezr("JUPITER", receive_et, "J2000", "LT+S", "EARTH")
        emission_et = receive_et - float(light_time)
        jupiter_to_earth, _ = spice.spkezr("EARTH", emission_et, "J2000", "NONE", "JUPITER")
        rotation = spice.pxform("J2000", "IAU_JUPITER", emission_et)
        body = rotation @ np.asarray(jupiter_to_earth[:3], dtype=float)
        latitude = np.degrees(np.arctan2(body[2], np.hypot(body[0], body[1])))
        rows.append(
            {
                "time": value,
                "jupiter_observer_latitude_deg": float(latitude),
                "jupiter_light_time_s_recomputed": float(light_time),
            }
        )
    spice.kclear()
    return pd.DataFrame(rows)


def magnetic_latitude(cml_west_deg: np.ndarray, observer_latitude_deg: np.ndarray) -> np.ndarray:
    """Observer magnetic latitude for the paper's centered tilted dipole."""

    west = np.deg2rad(np.asarray(cml_west_deg, dtype=float))
    latitude = np.deg2rad(np.asarray(observer_latitude_deg, dtype=float))
    observer = np.column_stack(
        [
            np.cos(latitude) * np.cos(-west),
            np.cos(latitude) * np.sin(-west),
            np.sin(latitude),
        ]
    )
    pole_latitude = np.deg2rad(90.0 - DIPOLE_TILT_DEG)
    pole_east_longitude = np.deg2rad(-DIPOLE_POLE_WEST_LONGITUDE_DEG)
    pole = np.asarray(
        [
            np.cos(pole_latitude) * np.cos(pole_east_longitude),
            np.cos(pole_latitude) * np.sin(pole_east_longitude),
            np.sin(pole_latitude),
        ]
    )
    return np.degrees(np.arcsin(np.clip(observer @ pole, -1.0, 1.0)))


def beam_mask(magnetic_latitude_deg: np.ndarray) -> np.ndarray:
    return (
        np.abs(np.asarray(magnetic_latitude_deg, dtype=float) - BEAM_CENTER_MAGNETIC_LATITUDE_DEG)
        <= BEAM_HALF_WIDTH_DEG
    )


def add_latitude_geometry(events: pd.DataFrame, kernel_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    geometry = observer_latitude(events["time"], kernel_dir)
    result = events.merge(geometry, on="time", how="left", validate="many_to_one")
    return result, geometry


def analyze_detector(
    events: pd.DataFrame,
    detector: str,
    n_permutations: int,
    seed: int,
) -> tuple[dict[str, object], pd.DataFrame, np.ndarray, np.ndarray]:
    source = detector_source(events, detector, require_geometry=True)
    if "jupiter_observer_latitude_deg" not in source.columns:
        latitude_lookup = events[
            ["time", "jupiter_observer_latitude_deg"]
        ].drop_duplicates("time")
        source = source.merge(latitude_lookup, on="time", how="left", validate="one_to_one")
    gate = band_gate(source, PRIMARY_BANDS, PRIMARY_QUANTILE, mode="any").astype(np.int8)
    cml = source["jupiter_cml_system_iii_deg"].to_numpy(dtype=float)
    actual_latitude = source["jupiter_observer_latitude_deg"].to_numpy(dtype=float)
    static_latitude = np.full(len(source), float(np.median(actual_latitude)))
    actual_magnetic_latitude = magnetic_latitude(cml, actual_latitude)
    static_magnetic_latitude = magnetic_latitude(cml, static_latitude)
    actual_beam = beam_mask(actual_magnetic_latitude)
    static_beam = beam_mask(static_magnetic_latitude)

    observed_actual = float(log_rate_ratio(gate, actual_beam))
    observed_static = float(log_rate_ratio(gate, static_beam))
    observed_gain = observed_actual - observed_static
    rng = np.random.default_rng(int(seed))
    groups, offsets = daily_roll_plan(source, n_permutations, rng)
    rolled = rolled_vectors(gate, groups, offsets)
    null_actual = log_rate_ratio(rolled, actual_beam)
    null_static = log_rate_ratio(rolled, static_beam)
    null_gain = null_actual - null_static

    summary: dict[str, object] = {
        "detector": detector,
        "n_valid": int(len(source)),
        "n_candidates": int(gate.sum()),
        "observer_latitude_min_deg": float(np.min(actual_latitude)),
        "observer_latitude_median_deg": float(np.median(actual_latitude)),
        "observer_latitude_max_deg": float(np.max(actual_latitude)),
        "actual_beam_exposure_fraction": float(np.mean(actual_beam)),
        "static_beam_exposure_fraction": float(np.mean(static_beam)),
        "actual_beam_log_rate_ratio": observed_actual,
        "actual_beam_rate_ratio": float(np.exp(observed_actual)),
        "actual_beam_local_p": upper_tail_p(observed_actual, null_actual),
        "static_beam_log_rate_ratio": observed_static,
        "static_beam_rate_ratio": float(np.exp(observed_static)),
        "latitude_conditioning_gain": observed_gain,
        "latitude_conditioning_gain_local_p": upper_tail_p(observed_gain, null_gain),
    }

    edges = np.quantile(actual_latitude, [0.0, 1 / 3, 2 / 3, 1.0])
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    latitude_bin = np.clip(np.digitize(actual_latitude, edges[1:-1]), 0, 2)
    cml_bin = np.floor((cml % 360.0) / 20.0).astype(int)
    rows: list[dict[str, object]] = []
    for lat_bin in range(3):
        for longitude_bin in range(18):
            keep = (latitude_bin == lat_bin) & (cml_bin == longitude_bin)
            rows.append(
                {
                    "detector": detector,
                    "observer_latitude_bin": int(lat_bin + 1),
                    "observer_latitude_low_deg": float(edges[lat_bin]),
                    "observer_latitude_high_deg": float(edges[lat_bin + 1]),
                    "cml_center_deg": float(longitude_bin * 20 + 10),
                    "n_valid": int(keep.sum()),
                    "candidate_fraction": float(np.mean(gate[keep])) if np.any(keep) else np.nan,
                    "predicted_beam_fraction": float(np.mean(actual_beam[keep])) if np.any(keep) else np.nan,
                }
            )
    curves = pd.DataFrame(rows)
    return summary, curves, null_actual, null_gain


def plot_effects(summary: pd.DataFrame, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    x = np.arange(len(summary))
    width = 0.34
    ax.bar(
        x - width / 2,
        summary["actual_beam_log_rate_ratio"],
        width,
        color=[COLORS[value] for value in summary["detector"]],
        label="actual observer latitude",
    )
    ax.bar(
        x + width / 2,
        summary["static_beam_log_rate_ratio"],
        width,
        color="0.72",
        edgecolor="0.38",
        label="latitude held at median",
    )
    ax.axhline(0.0, color="0.25", lw=0.9)
    ax.set_xticks(x, [LABELS[value] for value in summary["detector"]])
    ax.set_ylabel("log burst-rate ratio inside predicted beam")
    ax.set_title("Stream 3 - external latitude/beaming prediction")
    ax.grid(True, axis="y", color="0.90", lw=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = output / "stream3_latitude_beam_effects.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def plot_curves(curves: pd.DataFrame, output: Path) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(9.6, 8.4), sharex=True)
    shades = (0.35, 0.62, 0.92)
    for ax, detector in zip(axes, DETECTORS):
        detector_curves = curves[curves["detector"].eq(detector)]
        for latitude_bin, alpha in zip((1, 2, 3), shades):
            sub = detector_curves[detector_curves["observer_latitude_bin"].eq(latitude_bin)]
            label = (
                f"D_E {sub['observer_latitude_low_deg'].iloc[0]:.2f} to "
                f"{sub['observer_latitude_high_deg'].iloc[0]:.2f} deg"
            )
            ax.plot(
                sub["cml_center_deg"],
                sub["candidate_fraction"],
                color=COLORS[detector],
                alpha=alpha,
                marker="o",
                ms=3.2,
                lw=1.25,
                label=label,
            )
        ax.set_ylabel("top-1%\nfraction")
        ax.set_title(LABELS[detector], loc="left", fontsize=10)
        ax.grid(True, color="0.92", lw=0.45)
        ax.legend(frameon=False, fontsize=8, ncol=3)
    axes[-1].set_xlabel("Jupiter System III central meridian longitude (west deg)")
    axes[-1].set_xlim(0, 360)
    axes[-1].set_xticks(np.arange(0, 361, 45))
    fig.suptitle("Low-frequency RAE-2 burst occurrence across observer-latitude terciles")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = output / "stream3_latitude_cml_curves.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def write_report(output: Path, summary: pd.DataFrame, files: list[Path], config: dict[str, object]) -> Path:
    columns = [
        "detector",
        "n_valid",
        "actual_beam_rate_ratio",
        "actual_beam_local_p",
        "actual_beam_fwer_p",
        "static_beam_rate_ratio",
        "latitude_conditioning_gain",
        "latitude_conditioning_gain_local_p",
        "latitude_conditioning_gain_fwer_p",
    ]
    lines = [
        "# Stream 3 - Jovian latitude/beaming prior",
        "",
        "The primary test uses RAE-2's 0.70, 0.90, and 1.31 MHz bands at a fixed top-1% daily gate. SPICE supplies the actual sub-observer jovigraphic latitude. A centered dipole tilted 9.6 degrees toward System III west longitude 202 degrees predicts magnetic latitude; the fixed beam is 10 degrees wide and centered at +3 degrees, following Alexander et al. (1979).",
        "",
        "## Results",
        "",
        summary[columns].to_string(index=False),
        "",
        "## Controls and interpretation",
        "",
        f"- {config['n_permutations']} within-day circular rolls preserve clustering, daily thresholds, and the strong latitude/time relation.",
        "- The static comparison holds observer latitude at its full-run median and isolates the value added by the time-varying geometry.",
        "- FWER p-values use the maximum statistic across RV1, RV2, and receiver coincidence.",
        "- RAE-2 spans only a modest positive-latitude range, so this is a partial test of the much wider multi-spacecraft relation in the source paper.",
        "",
        "## Files",
        "",
        *[f"- `{path.name}`" for path in files],
    ]
    report = output / "STREAM3_LATITUDE_REPORT.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--kernel-dir", type=Path, default=DEFAULT_KERNELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-permutations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=260825301)
    args = parser.parse_args()

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    events = pd.read_parquet(args.events)
    events_with_latitude, geometry = add_latitude_geometry(events, args.kernel_dir)
    config = {
        "events": str(args.events),
        "kernel_dir": str(args.kernel_dir),
        "primary_bands": list(PRIMARY_BANDS),
        "primary_quantile": PRIMARY_QUANTILE,
        "dipole_tilt_deg": DIPOLE_TILT_DEG,
        "dipole_pole_west_longitude_deg": DIPOLE_POLE_WEST_LONGITUDE_DEG,
        "beam_center_magnetic_latitude_deg": BEAM_CENTER_MAGNETIC_LATITUDE_DEG,
        "beam_half_width_deg": BEAM_HALF_WIDTH_DEG,
        "n_permutations": int(args.n_permutations),
        "seed": int(args.seed),
    }
    (output / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    summaries = []
    curves = []
    actual_nulls = []
    gain_nulls = []
    for detector_index, detector in enumerate(DETECTORS):
        summary, curve, actual_null, gain_null = analyze_detector(
            events_with_latitude,
            detector,
            int(args.n_permutations),
            int(args.seed) + detector_index,
        )
        summaries.append(summary)
        curves.append(curve)
        actual_nulls.append(actual_null)
        gain_nulls.append(gain_null)
    summary_table = pd.DataFrame(summaries)
    curve_table = pd.concat(curves, ignore_index=True)
    summary_table["actual_beam_fwer_p"] = familywise_upper_p(
        summary_table["actual_beam_log_rate_ratio"].to_numpy(dtype=float),
        np.vstack(actual_nulls),
    )
    summary_table["latitude_conditioning_gain_fwer_p"] = familywise_upper_p(
        summary_table["latitude_conditioning_gain"].to_numpy(dtype=float),
        np.vstack(gain_nulls),
    )
    summary_table.to_csv(output / "stream3_latitude_summary.csv", index=False)
    curve_table.to_csv(output / "stream3_latitude_cml_curves.csv", index=False)
    geometry.to_parquet(output / "stream3_observer_latitude_geometry.parquet", index=False)
    files = [
        plot_effects(summary_table, output),
        plot_curves(curve_table, output),
        output / "stream3_latitude_summary.csv",
        output / "stream3_observer_latitude_geometry.parquet",
    ]
    report = write_report(output, summary_table, files, config)
    print(report)


if __name__ == "__main__":
    main()
