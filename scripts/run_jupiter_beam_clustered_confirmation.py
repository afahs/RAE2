#!/usr/bin/env python
"""Confirm the historical RAE-2 Jupiter burst hint with stricter controls.

The historical upper-V analysis first selected samples for which Jupiter was
inside the antenna beam and then searched for clustered, single-frequency
bursts.  This script reproduces that design from the cleaned raw receiver
samples, extends it to both Ryle-Vonberg receivers, and replaces the iid
binomial test with a within-day circular-roll null.

The historical digitized E/H-plane curves are read from the RAE-2 antenna
digitization directory and combined exactly as in the earlier analysis.  The
4.70-MHz channel uses its historical nearest model (3.93 MHz), while 6.55 MHz
uses the 6.55-MHz model.  The fixed selection threshold is -10 dB.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.frames import body_unit_vectors_from_moon  # noqa: E402
from rylevonberg.util import ensure_dir  # noqa: E402
from scripts.jupiter_external_stream_common import (  # noqa: E402
    familywise_upper_p,
    log_rate_ratio,
    rolled_vectors,
    upper_tail_p,
)
from scripts.plot_jupiter_upper_v_beam_io_phase_frequency import (  # noqa: E402
    beam_half_width_deg_for_frequency,
    compute_beam_separation_deg,
    interpolate_cyclic,
    load_relative_beam_model,
    radec_unit_vectors,
)
from scripts.run_jupiter_beam_burst_phase_enrichment import (  # noqa: E402
    cluster_high_samples,
)


DEFAULT_RAW = Path(
    "/global/u2/a/afahs/RAE2/RyleVonberg/runtime_inputs/occultogram_v1/cleaned_timeseries.parquet"
)
DEFAULT_GEOMETRY = Path(
    "/pscratch/sd/a/afahs/RAE2/jupiter_burst_gated_phase_occurrence_v1/"
    "five_minute_channel_percentile_events.parquet"
)
DEFAULT_OUT = Path("/pscratch/sd/a/afahs/RAE2/jupiter_beam_clustered_confirmation_v1")
DEFAULT_BEAM_DIR = Path("/global/u2/a/afahs/antennaDigitization")

ANTENNAS = ("rv1_coarse", "rv2_coarse")
BAND_FREQUENCIES = {7: 4.70, 8: 6.55}
EXPECTED_IO_WINDOWS = ((80.0, 130.0), (205.0, 260.0))
PERIODS = ("full", "early", "late")
COLORS = {"rv1_coarse": "#3568a8", "rv2_coarse": "#d4681f"}


def in_expected_io_windows(phases_deg: np.ndarray) -> np.ndarray:
    """Return membership in the fixed historical Io-B/D and Io-A/C windows."""

    phase = np.asarray(phases_deg, dtype=float) % 360.0
    out = np.zeros(phase.shape, dtype=bool)
    for low, high in EXPECTED_IO_WINDOWS:
        out |= (phase >= low) & (phase < high)
    return out


def attach_local_normalization(
    samples: pd.DataFrame,
    window_seconds: float = 1200.0,
    min_periods: int = 6,
) -> pd.DataFrame:
    """Attach the historical centered rolling median/MAD raw-power z score."""

    out = samples.copy()
    out["local_normalization_center_power"] = np.nan
    out["local_normalization_scale_power"] = np.nan
    out["local_normalized_power"] = np.nan
    window = pd.to_timedelta(float(window_seconds), unit="s")
    for _, group in out.groupby(["antenna", "frequency_band"], sort=True):
        ordered = group.sort_values("time")
        power = pd.to_numeric(ordered["power"], errors="coerce").to_numpy(dtype=float)
        series = pd.Series(power, index=pd.DatetimeIndex(ordered["time"]))
        rolling = series.rolling(window=window, center=True, min_periods=int(min_periods))
        center = rolling.median()
        absolute_deviation = (series - center).abs()
        scale = 1.4826 * absolute_deviation.rolling(
            window=window,
            center=True,
            min_periods=int(min_periods),
        ).median()
        fallback = rolling.std(ddof=1)
        scale = scale.where(np.isfinite(scale) & (scale > 0.0), fallback)
        normalized = (series - center) / scale
        normalized = normalized.mask(~np.isfinite(scale) | (scale <= 0.0))
        out.loc[ordered.index, "local_normalization_center_power"] = center.to_numpy(dtype=float)
        out.loc[ordered.index, "local_normalization_scale_power"] = scale.to_numpy(dtype=float)
        out.loc[ordered.index, "local_normalized_power"] = normalized.to_numpy(dtype=float)
    return out


def load_raw_samples(path: Path, max_rows: int | None = None) -> pd.DataFrame:
    columns = [
        "time",
        "frequency_band",
        "antenna",
        "power",
        "right_ascension",
        "declination",
        "is_valid",
    ]
    samples = pd.read_parquet(
        path,
        columns=columns,
        filters=[("frequency_band", "in", list(BAND_FREQUENCIES))],
    )
    keep = samples["antenna"].astype(str).isin(ANTENNAS)
    keep &= samples["frequency_band"].isin(BAND_FREQUENCIES)
    keep &= samples["is_valid"].astype(bool)
    for column in ["power", "right_ascension", "declination"]:
        keep &= np.isfinite(pd.to_numeric(samples[column], errors="coerce"))
    keep &= pd.to_numeric(samples["power"], errors="coerce") > 0.0
    samples = samples.loc[keep, columns[:-1]].copy()
    samples["time"] = pd.to_datetime(samples["time"], utc=False)
    samples["frequency_band"] = samples["frequency_band"].astype(int)
    samples["frequency_mhz"] = samples["frequency_band"].map(BAND_FREQUENCIES).astype(float)
    if max_rows is not None:
        samples = (
            samples.sort_values("time")
            .groupby(["antenna", "frequency_band"], sort=True, group_keys=False)
            .head(int(max_rows))
        )
    return samples.sort_values(["antenna", "frequency_band", "time"]).reset_index(drop=True)


def load_geometry_with_vectors(path: Path) -> pd.DataFrame:
    columns = [
        "time",
        "io_phase_deg",
        "jupiter_cml_system_iii_deg",
        "jupiter_occulted",
    ]
    geom = pd.read_parquet(path, columns=columns)
    geom["time"] = pd.to_datetime(geom["time"], utc=False)
    geom = (
        geom.dropna(subset=["io_phase_deg", "jupiter_cml_system_iii_deg"])
        .sort_values("time")
        .drop_duplicates("time")
        .reset_index(drop=True)
    )
    vectors = body_unit_vectors_from_moon("jupiter", geom["time"], target_frame="fk4")
    geom["jupiter_unit_x"] = vectors[:, 0]
    geom["jupiter_unit_y"] = vectors[:, 1]
    geom["jupiter_unit_z"] = vectors[:, 2]
    return geom


def select_jupiter_beam_samples(
    samples: pd.DataFrame,
    geometry: pd.DataFrame,
    beam_dir: Path,
    gain_threshold_db: float = -10.0,
    tolerance_seconds: float = 360.0,
) -> tuple[pd.DataFrame, str]:
    """Merge SPICE geometry and apply the historical digitized-gain cut."""

    keep_columns = [
        "time",
        "io_phase_deg",
        "jupiter_cml_system_iii_deg",
        "jupiter_occulted",
        "jupiter_unit_x",
        "jupiter_unit_y",
        "jupiter_unit_z",
    ]
    merged = pd.merge_asof(
        samples.sort_values("time"),
        geometry[keep_columns].sort_values("time"),
        on="time",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=float(tolerance_seconds)),
    )
    merged = merged.dropna(subset=["io_phase_deg", "jupiter_unit_x"]).copy()
    beam_vectors, ra_units = radec_unit_vectors(
        merged["right_ascension"],
        merged["declination"],
        "auto",
    )
    jupiter_vectors = merged[
        ["jupiter_unit_x", "jupiter_unit_y", "jupiter_unit_z"]
    ].to_numpy(dtype=float)
    merged["jupiter_beam_separation_deg"] = compute_beam_separation_deg(
        beam_vectors,
        jupiter_vectors,
    )
    merged["beam_half_width_deg"] = merged["frequency_mhz"].map(
        beam_half_width_deg_for_frequency
    )
    merged["jupiter_beam_relative_gain_db"] = np.nan
    merged["beam_model_frequency_mhz"] = np.nan
    beam_models = {
        7: (3.93, beam_dir / "eplane_3p93MHz.csv", beam_dir / "hplane_3p93MHz.csv"),
        8: (6.55, beam_dir / "eplane_6p55MHz.csv", beam_dir / "hplane_6p55MHz.csv"),
    }
    for band, (model_frequency, eplane, hplane) in beam_models.items():
        angles, relative_gain_db = load_relative_beam_model(eplane, hplane)
        band_keep = merged["frequency_band"].eq(band)
        merged.loc[band_keep, "jupiter_beam_relative_gain_db"] = interpolate_cyclic(
            angles,
            relative_gain_db,
            merged.loc[band_keep, "jupiter_beam_separation_deg"].to_numpy(dtype=float),
        )
        merged.loc[band_keep, "beam_model_frequency_mhz"] = model_frequency
    merged["jupiter_passes_gain_beam_cut"] = (
        merged["jupiter_beam_relative_gain_db"] >= float(gain_threshold_db)
    )
    selected = merged.loc[
        ~merged["jupiter_occulted"].astype(bool)
        & merged["jupiter_passes_gain_beam_cut"]
        & np.isfinite(merged["local_normalized_power"])
    ].copy()
    selected["date"] = selected["time"].dt.floor("D")
    return (
        selected.sort_values(["antenna", "frequency_band", "time"]).reset_index(drop=True),
        ra_units,
    )


def mark_cluster_peak_candidates(
    source: pd.DataFrame,
    threshold: float,
    max_gap_seconds: float = 180.0,
    min_cluster_samples: int = 2,
    max_event_minutes: float = 30.0,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Make one candidate at the highest-z sample of every accepted cluster."""

    work = source.sort_values("time").reset_index(drop=True)
    gate = np.zeros(len(work), dtype=np.int8)
    rows: list[dict[str, object]] = []
    clusters = cluster_high_samples(
        work,
        threshold=float(threshold),
        max_gap_seconds=float(max_gap_seconds),
        min_cluster_samples=int(min_cluster_samples),
        max_event_minutes=float(max_event_minutes),
    )
    z = pd.to_numeric(work["local_normalized_power"], errors="coerce").to_numpy(dtype=float)
    for cluster_number, (start, end) in enumerate(clusters):
        positions = np.arange(start, end + 1, dtype=int)
        high = positions[z[positions] >= float(threshold)]
        if len(high) == 0:
            continue
        peak_position = int(high[np.nanargmax(z[high])])
        gate[peak_position] = 1
        peak = work.iloc[peak_position]
        rows.append(
            {
                "cluster_number": int(cluster_number),
                "antenna": str(peak["antenna"]),
                "frequency_band": int(peak["frequency_band"]),
                "frequency_mhz": float(peak["frequency_mhz"]),
                "burst_start_time": pd.Timestamp(work.iloc[start]["time"]),
                "burst_end_time": pd.Timestamp(work.iloc[end]["time"]),
                "burst_peak_time": pd.Timestamp(peak["time"]),
                "n_high_samples": int(len(high)),
                "peak_local_normalized_power": float(peak["local_normalized_power"]),
                "io_phase_deg": float(peak["io_phase_deg"]) % 360.0,
                "jupiter_cml_system_iii_deg": float(peak["jupiter_cml_system_iii_deg"]) % 360.0,
                "in_expected_io_window": bool(
                    in_expected_io_windows(np.array([float(peak["io_phase_deg"])]))[0]
                ),
                "jupiter_beam_separation_deg": float(peak["jupiter_beam_separation_deg"]),
                "jupiter_beam_relative_gain_db": float(peak["jupiter_beam_relative_gain_db"]),
                "beam_half_width_deg": float(peak["beam_half_width_deg"]),
            }
        )
    return gate, pd.DataFrame.from_records(rows)


def shared_daily_roll_plan(
    source: pd.DataFrame,
    date_uniforms: np.ndarray,
    date_columns: dict[pd.Timestamp, int],
) -> tuple[list[np.ndarray], np.ndarray]:
    """Map shared per-day uniform draws to legal circular offsets."""

    groups: list[np.ndarray] = []
    offsets: list[np.ndarray] = []
    for date, group in source.groupby("date", sort=True):
        indices = group.index.to_numpy(dtype=int)
        groups.append(indices)
        column = date_columns[pd.Timestamp(date)]
        offsets.append(np.floor(date_uniforms[:, column] * len(indices)).astype(np.int32))
    if not offsets:
        return groups, np.empty((date_uniforms.shape[0], 0), dtype=np.int32)
    return groups, np.column_stack(offsets)


def period_mask(times: pd.Series, period: str, split_time: pd.Timestamp) -> np.ndarray:
    values = pd.DatetimeIndex(times)
    if period == "full":
        return np.ones(len(values), dtype=bool)
    if period == "early":
        return np.asarray(values < split_time, dtype=bool)
    if period == "late":
        return np.asarray(values >= split_time, dtype=bool)
    raise ValueError(f"Unknown period: {period}")


def _stat_counts(gate: np.ndarray, target: np.ndarray) -> dict[str, object]:
    target = np.asarray(target, dtype=bool)
    return {
        "n_exposure": int(len(gate)),
        "n_target_exposure": int(target.sum()),
        "n_control_exposure": int((~target).sum()),
        "n_bursts": int(np.sum(gate)),
        "n_target_bursts": int(np.sum(gate[target])),
        "n_control_bursts": int(np.sum(gate[~target])),
        "target_rate_per_100k": float(np.mean(gate[target]) * 100000.0) if target.any() else np.nan,
        "control_rate_per_100k": float(np.mean(gate[~target]) * 100000.0) if (~target).any() else np.nan,
    }


def analyse_cell(
    source: pd.DataFrame,
    gate: np.ndarray,
    rolled: np.ndarray,
    split_time: pd.Timestamp,
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    rows: list[dict[str, object]] = []
    nulls: dict[str, np.ndarray] = {}
    for period in PERIODS:
        keep = period_mask(source["time"], period, split_time)
        target = in_expected_io_windows(source.loc[keep, "io_phase_deg"].to_numpy(dtype=float))
        observed = float(log_rate_ratio(gate[keep], target))
        null = np.asarray(log_rate_ratio(rolled[:, keep], target), dtype=float)
        nulls[period] = null
        row = _stat_counts(gate[keep], target)
        row.update(
            {
                "period": period,
                "observed_log_rate_ratio": observed,
                "observed_rate_ratio": float(np.exp(observed)) if np.isfinite(observed) else np.nan,
                "roll_p_upper": upper_tail_p(observed, null),
                "null_log_rate_ratio_q025": float(np.nanquantile(null, 0.025)),
                "null_log_rate_ratio_q500": float(np.nanquantile(null, 0.500)),
                "null_log_rate_ratio_q975": float(np.nanquantile(null, 0.975)),
            }
        )
        rows.append(row)
    return rows, nulls


def phase_curve(source: pd.DataFrame, gate: np.ndarray, phase_bin_deg: float) -> pd.DataFrame:
    phase = source["io_phase_deg"].to_numpy(dtype=float) % 360.0
    bin_left = np.floor(phase / float(phase_bin_deg)) * float(phase_bin_deg)
    rows = []
    for left in np.arange(0.0, 360.0, float(phase_bin_deg)):
        keep = np.isclose(bin_left, left)
        exposure = int(np.count_nonzero(keep))
        bursts = int(np.sum(gate[keep]))
        rows.append(
            {
                "io_phase_bin_left_deg": float(left),
                "io_phase_bin_center_deg": float(left + 0.5 * phase_bin_deg),
                "n_exposure": exposure,
                "n_bursts": bursts,
                "burst_rate_per_100k": bursts * 100000.0 / exposure if exposure else np.nan,
                "in_expected_io_window": bool(
                    in_expected_io_windows(np.array([left + 0.5 * phase_bin_deg]))[0]
                ),
            }
        )
    return pd.DataFrame(rows)


def add_familywise_p(summary: pd.DataFrame, null_by_cell: dict[tuple[str, int, str], np.ndarray]) -> pd.DataFrame:
    out = summary.copy()
    out["roll_p_fwer_across_4_cells"] = np.nan
    for period in PERIODS:
        peer = out[out["period"].eq(period)].sort_values(["antenna", "frequency_band"])
        observed = peer["observed_log_rate_ratio"].to_numpy(dtype=float)
        null = np.vstack(
            [
                null_by_cell[(str(row.antenna), int(row.frequency_band), period)]
                for row in peer.itertuples(index=False)
            ]
        )
        out.loc[peer.index, "roll_p_fwer_across_4_cells"] = familywise_upper_p(observed, null)
    return out


def replication_summary(
    summary: pd.DataFrame,
    null_by_cell: dict[tuple[str, int, str], np.ndarray],
) -> pd.DataFrame:
    """Require same-sign enrichment across receivers and time halves."""

    rows = []
    for band, frequency in BAND_FREQUENCIES.items():
        full_rows = summary[
            summary["frequency_band"].eq(band) & summary["period"].eq("full")
        ].sort_values("antenna")
        full_observed = float(full_rows["observed_log_rate_ratio"].min())
        full_null = np.min(
            np.vstack(
                [null_by_cell[(antenna, band, "full")] for antenna in ANTENNAS]
            ),
            axis=0,
        )
        component_rows = summary[
            summary["frequency_band"].eq(band) & summary["period"].isin(["early", "late"])
        ]
        chronological_observed = float(component_rows["observed_log_rate_ratio"].min())
        chronological_null = np.min(
            np.vstack(
                [
                    null_by_cell[(antenna, band, period)]
                    for antenna in ANTENNAS
                    for period in ("early", "late")
                ]
            ),
            axis=0,
        )
        rows.extend(
            [
                {
                    "frequency_band": band,
                    "frequency_mhz": frequency,
                    "replication_test": "both_receivers_full_min_log_rate_ratio",
                    "observed_min_log_rate_ratio": full_observed,
                    "observed_min_rate_ratio": float(np.exp(full_observed)),
                    "roll_p_upper": upper_tail_p(full_observed, full_null),
                    "null_q025": float(np.nanquantile(full_null, 0.025)),
                    "null_q975": float(np.nanquantile(full_null, 0.975)),
                    "n_components": 2,
                },
                {
                    "frequency_band": band,
                    "frequency_mhz": frequency,
                    "replication_test": "both_receivers_x_early_late_min_log_rate_ratio",
                    "observed_min_log_rate_ratio": chronological_observed,
                    "observed_min_rate_ratio": float(np.exp(chronological_observed)),
                    "roll_p_upper": upper_tail_p(chronological_observed, chronological_null),
                    "null_q025": float(np.nanquantile(chronological_null, 0.025)),
                    "null_q975": float(np.nanquantile(chronological_null, 0.975)),
                    "n_components": 4,
                },
            ]
        )
    return pd.DataFrame(rows)


def plot_effects(summary: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), sharey=True)
    for ax, (band, frequency) in zip(axes, BAND_FREQUENCIES.items()):
        sub = summary[summary["frequency_band"].eq(band)].copy()
        x_base = np.arange(len(PERIODS), dtype=float)
        for antenna, offset in zip(ANTENNAS, (-0.10, 0.10)):
            cell = sub[sub["antenna"].eq(antenna)].set_index("period").reindex(PERIODS)
            observed = cell["observed_log_rate_ratio"].to_numpy(dtype=float)
            low = cell["null_log_rate_ratio_q025"].to_numpy(dtype=float)
            high = cell["null_log_rate_ratio_q975"].to_numpy(dtype=float)
            ax.errorbar(
                x_base + offset,
                observed,
                yerr=np.vstack([observed - low, high - observed]),
                fmt="o",
                capsize=4,
                color=COLORS[antenna],
                label=antenna.replace("_coarse", "").upper(),
            )
        ax.axhline(0.0, color="#555", lw=1.0)
        ax.set_xticks(x_base, ["Full", "Early", "Late"])
        ax.set_title(f"{frequency:.2f} MHz")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)
    axes[0].set_ylabel("log rate ratio: canonical Io windows / outside")
    fig.suptitle("Digitized-beam clustered bursts vs within-day circular-roll null (95% interval)")
    fig.tight_layout()
    path = out_dir / "jupiter_beam_clustered_effects.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_phase_curves(curves: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.4), sharex=True)
    for ax, ((antenna, band), cell) in zip(
        axes.ravel(),
        curves.groupby(["antenna", "frequency_band"], sort=True),
    ):
        ax.plot(
            cell["io_phase_bin_center_deg"],
            cell["burst_rate_per_100k"],
            marker="o",
            lw=1.5,
            color=COLORS[str(antenna)],
        )
        for low, high in EXPECTED_IO_WINDOWS:
            ax.axvspan(low, high, color="#f1b44c", alpha=0.15, lw=0)
        ax.set_xlim(0, 360)
        ax.set_xticks(np.arange(0, 361, 60))
        ax.grid(alpha=0.22)
        ax.set_title(f"{str(antenna).replace('_coarse', '').upper()} · {BAND_FREQUENCIES[int(band)]:.2f} MHz")
        ax.set_ylabel("bursts / 100k samples")
    for ax in axes[-1, :]:
        ax.set_xlabel("Io phase (degrees)")
    fig.suptitle("Exposure-corrected clustered-burst occurrence by Io phase")
    fig.tight_layout()
    path = out_dir / "jupiter_beam_clustered_phase_curves.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report(
    out_dir: Path,
    summary: pd.DataFrame,
    replication: pd.DataFrame,
    thresholds: pd.DataFrame,
    config: dict[str, object],
) -> Path:
    lines = [
        "# Jupiter digitized-beam clustered-burst confirmation",
        "",
        "## Design",
        "",
        "Raw valid positive-power samples at 4.70 and 6.55 MHz were locally normalized "
        "with a centered 20-minute rolling median/MAD. Jupiter had to be unocculted and "
        "inside the historical digitized mean E/H-plane beam at relative gain ≥ -10 dB. "
        "Bursts were defined without reference to Io phase at max(channel q99.5, z=5), "
        "then clustered within 180 s with at least two high samples.",
        "",
        "The fixed target is the union of Io phase 80–130° and 205–260°. P-values use "
        "within-day circular rolls of the burst indicator. Early/late and RV1/RV2 "
        "replication tests use the minimum log rate ratio, so every component must have "
        "the same positive sign for a positive joint statistic.",
        "",
        "At 4.70 MHz this deliberately reproduces the old script's nearest-model rule "
        "and therefore uses the 3.93-MHz beam curve; 6.55 MHz uses its same-frequency curve.",
        "",
        "## Cell results",
        "",
        "| Receiver | MHz | Period | Bursts | RR | roll p | 4-cell FWER p |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in summary.sort_values(["frequency_band", "antenna", "period"]).itertuples(index=False):
        lines.append(
            f"| {str(row.antenna).replace('_coarse', '').upper()} | {row.frequency_mhz:.2f} | "
            f"{row.period} | {row.n_bursts} | {row.observed_rate_ratio:.3f} | "
            f"{row.roll_p_upper:.4f} | {row.roll_p_fwer_across_4_cells:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Replication tests",
            "",
            "| MHz | Test | min RR | roll p |",
            "|---:|---|---:|---:|",
        ]
    )
    for row in replication.itertuples(index=False):
        lines.append(
            f"| {row.frequency_mhz:.2f} | {row.replication_test} | "
            f"{row.observed_min_rate_ratio:.3f} | {row.roll_p_upper:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Burst thresholds",
            "",
            "| Receiver | MHz | Exposure | q99.5 z | threshold z | bursts |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in thresholds.itertuples(index=False):
        lines.append(
            f"| {str(row.antenna).replace('_coarse', '').upper()} | {row.frequency_mhz:.2f} | "
            f"{row.n_beam_selected_exposure:,} | {row.quantile_local_z:.3f} | "
            f"{row.threshold_local_z:.3f} | {row.n_clustered_bursts} |"
        )
    lines.extend(["", "## Configuration", "", "```json", json.dumps(config, indent=2), "```", ""])
    path = out_dir / "JUPITER_BEAM_CLUSTERED_CONFIRMATION.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--beam-dir", type=Path, default=DEFAULT_BEAM_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--local-window-seconds", type=float, default=1200.0)
    parser.add_argument("--local-min-periods", type=int, default=6)
    parser.add_argument("--geometry-tolerance-seconds", type=float, default=360.0)
    parser.add_argument("--gain-threshold-db", type=float, default=-10.0)
    parser.add_argument("--burst-quantile", type=float, default=0.995)
    parser.add_argument("--min-local-z", type=float, default=5.0)
    parser.add_argument("--cluster-gap-seconds", type=float, default=180.0)
    parser.add_argument("--min-cluster-samples", type=int, default=2)
    parser.add_argument("--max-event-minutes", type=float, default=30.0)
    parser.add_argument("--split-time", default="1975-02-28")
    parser.add_argument("--phase-bin-deg", type=float, default=20.0)
    parser.add_argument("--n-permutations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=260825075)
    parser.add_argument("--max-rows-per-cell", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = ensure_dir(args.out_dir)
    print(f"Reading raw receiver samples from {args.raw}", flush=True)
    samples = load_raw_samples(args.raw, max_rows=args.max_rows_per_cell)
    print(f"Locally normalizing {len(samples):,} raw samples", flush=True)
    samples = attach_local_normalization(
        samples,
        window_seconds=args.local_window_seconds,
        min_periods=args.local_min_periods,
    )
    print(f"Building Jupiter vectors from {args.geometry}", flush=True)
    geometry = load_geometry_with_vectors(args.geometry)
    selected, interpreted_ra_units = select_jupiter_beam_samples(
        samples,
        geometry,
        beam_dir=args.beam_dir,
        gain_threshold_db=args.gain_threshold_db,
        tolerance_seconds=args.geometry_tolerance_seconds,
    )
    print(f"Digitized-gain beam retained {len(selected):,} samples (RA interpreted as {interpreted_ra_units})", flush=True)

    all_dates = sorted(pd.Timestamp(value) for value in selected["date"].dropna().unique())
    date_columns = {date: index for index, date in enumerate(all_dates)}
    rng = np.random.default_rng(int(args.seed))
    date_uniforms = rng.random((int(args.n_permutations), len(all_dates)))
    split_time = pd.Timestamp(args.split_time)

    summary_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    burst_tables: list[pd.DataFrame] = []
    curve_tables: list[pd.DataFrame] = []
    null_by_cell: dict[tuple[str, int, str], np.ndarray] = {}

    for antenna in ANTENNAS:
        for band, frequency in BAND_FREQUENCIES.items():
            source = selected[
                selected["antenna"].eq(antenna) & selected["frequency_band"].eq(band)
            ].sort_values("time").reset_index(drop=True)
            local_z = source["local_normalized_power"].to_numpy(dtype=float)
            quantile_value = float(np.nanquantile(local_z, args.burst_quantile))
            threshold = max(quantile_value, float(args.min_local_z))
            gate, bursts = mark_cluster_peak_candidates(
                source,
                threshold=threshold,
                max_gap_seconds=args.cluster_gap_seconds,
                min_cluster_samples=args.min_cluster_samples,
                max_event_minutes=args.max_event_minutes,
            )
            groups, offsets = shared_daily_roll_plan(source, date_uniforms, date_columns)
            rolled = rolled_vectors(gate, groups, offsets)
            rows, period_nulls = analyse_cell(source, gate, rolled, split_time)
            for row in rows:
                row.update(
                    {
                        "antenna": antenna,
                        "frequency_band": band,
                        "frequency_mhz": frequency,
                        "threshold_local_z": threshold,
                    }
                )
                summary_rows.append(row)
            for period, null in period_nulls.items():
                null_by_cell[(antenna, band, period)] = null
            threshold_rows.append(
                {
                    "antenna": antenna,
                    "frequency_band": band,
                    "frequency_mhz": frequency,
                    "n_beam_selected_exposure": len(source),
                    "burst_quantile": float(args.burst_quantile),
                    "quantile_local_z": quantile_value,
                    "min_local_z": float(args.min_local_z),
                    "threshold_local_z": threshold,
                    "n_clustered_bursts": int(gate.sum()),
                    "beam_half_width_deg": beam_half_width_deg_for_frequency(frequency),
                    "beam_gain_threshold_db": float(args.gain_threshold_db),
                    "beam_model_frequency_mhz": 3.93 if band == 7 else 6.55,
                }
            )
            if not bursts.empty:
                burst_tables.append(bursts)
            curve = phase_curve(source, gate, args.phase_bin_deg)
            curve.insert(0, "frequency_mhz", frequency)
            curve.insert(0, "frequency_band", band)
            curve.insert(0, "antenna", antenna)
            curve_tables.append(curve)
            print(
                f"{antenna} {frequency:.2f} MHz: {len(source):,} exposure, "
                f"z>={threshold:.3f}, {int(gate.sum())} clustered bursts",
                flush=True,
            )

    summary = add_familywise_p(pd.DataFrame(summary_rows), null_by_cell)
    thresholds = pd.DataFrame(threshold_rows)
    bursts = pd.concat(burst_tables, ignore_index=True) if burst_tables else pd.DataFrame()
    curves = pd.concat(curve_tables, ignore_index=True)
    replication = replication_summary(summary, null_by_cell)

    summary.to_csv(out_dir / "jupiter_beam_clustered_cell_summary.csv", index=False)
    thresholds.to_csv(out_dir / "jupiter_beam_clustered_thresholds.csv", index=False)
    bursts.to_csv(out_dir / "jupiter_beam_clustered_bursts.csv", index=False)
    curves.to_csv(out_dir / "jupiter_beam_clustered_phase_curves.csv", index=False)
    replication.to_csv(out_dir / "jupiter_beam_clustered_replication_summary.csv", index=False)
    plot_effects(summary, out_dir)
    plot_phase_curves(curves, out_dir)

    config: dict[str, object] = {
        "raw": str(args.raw),
        "geometry": str(args.geometry),
        "beam_dir": str(args.beam_dir),
        "beam_selection": "historical_digitized_mean_eh_gain",
        "beam_gain_threshold_db": float(args.gain_threshold_db),
        "beam_model_frequency_mhz": {"7": 3.93, "8": 6.55},
        "io_windows_deg": list(EXPECTED_IO_WINDOWS),
        "split_time": str(split_time),
        "n_permutations": int(args.n_permutations),
        "seed": int(args.seed),
        "ra_units_interpreted": interpreted_ra_units,
        "arguments": vars(args),
    }
    config["arguments"] = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (out_dir / "jupiter_beam_clustered_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )
    report = write_report(out_dir, summary, replication, thresholds, config)
    print(f"Wrote {report}", flush=True)


if __name__ == "__main__":
    main()
