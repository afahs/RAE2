#!/usr/bin/env python
"""Blind per-frequency burst selection followed by Io-phase enrichment checks.

This script consumes the upper-V beam-selected sample table produced by
``plot_jupiter_upper_v_beam_io_phase_frequency.py``.  Bursts are selected using
only local-normalized power within each frequency channel, so no broadband
emission assumption is required.  Io phase is used only after the burst catalog
exists.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import numpy as np
import pandas as pd
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rylevonberg.util import ensure_dir, software_versions, write_json  # noqa: E402


DEFAULT_SELECTED = (
    ROOT
    / "outputs/jupiter_upper_v_beam_gain10db_io_phase_frequency_v1/jupiter_upper_v_beam_selected_samples.npy"
)
DEFAULT_OUT = ROOT / "outputs/jupiter_upper_v_beam_single_frequency_burst_phase_v1"
EXPECTED_IO_WINDOWS = [("Io-B/D", 80.0, 130.0), ("Io-A/C", 205.0, 260.0)]


def shifted_windows(shift_deg: float, windows: list[tuple[str, float, float]] = EXPECTED_IO_WINDOWS) -> list[tuple[str, float, float]]:
    return [(label, (lo + float(shift_deg)) % 360.0, (hi + float(shift_deg)) % 360.0) for label, lo, hi in windows]


def in_phase_windows(phases_deg: np.ndarray, windows: list[tuple[str, float, float]] = EXPECTED_IO_WINDOWS) -> np.ndarray:
    phases = np.asarray(phases_deg, dtype=float) % 360.0
    out = np.zeros(phases.shape, dtype=bool)
    for _, lo, hi in windows:
        lo = float(lo) % 360.0
        hi = float(hi) % 360.0
        if lo <= hi:
            out |= (phases >= lo) & (phases < hi)
        else:
            out |= (phases >= lo) | (phases < hi)
    return out


def phase_window_label(phase_deg: float, windows: list[tuple[str, float, float]] = EXPECTED_IO_WINDOWS) -> str:
    phase = float(phase_deg) % 360.0
    for label, lo, hi in windows:
        lo = float(lo) % 360.0
        hi = float(hi) % 360.0
        inside = (lo <= phase < hi) if lo <= hi else (phase >= lo or phase < hi)
        if inside:
            return label
    return "outside"


def phase_bin_centers(phase_bin_deg: float) -> np.ndarray:
    return np.arange(0.5 * float(phase_bin_deg), 360.0, float(phase_bin_deg))


def attach_phase_bins(df: pd.DataFrame, phase_bin_deg: float) -> pd.DataFrame:
    out = df.copy()
    phase = pd.to_numeric(out["io_phase_spice_deg"], errors="coerce").to_numpy(dtype=float) % 360.0
    out["io_phase_bin_deg"] = np.floor(phase / float(phase_bin_deg)) * float(phase_bin_deg) + 0.5 * float(phase_bin_deg)
    return out


def threshold_table(samples: pd.DataFrame, quantile: float, min_local_z: float) -> pd.DataFrame:
    rows = []
    for (band, freq), grp in samples.groupby(["frequency_band", "frequency_mhz"], sort=True):
        vals = pd.to_numeric(grp["local_normalized_power"], errors="coerce").dropna().to_numpy(dtype=float)
        q = float(np.nanquantile(vals, float(quantile))) if vals.size else np.nan
        threshold = max(q, float(min_local_z)) if np.isfinite(q) else float(min_local_z)
        rows.append(
            {
                "frequency_band": int(band),
                "frequency_mhz": float(freq),
                "n_exposure_samples": int(vals.size),
                "threshold_quantile": float(quantile),
                "quantile_local_normalized_power": q,
                "min_local_z_floor": float(min_local_z),
                "burst_threshold_local_normalized_power": float(threshold),
            }
        )
    return pd.DataFrame(rows)


def cluster_high_samples(
    group: pd.DataFrame,
    threshold: float,
    max_gap_seconds: float,
    min_cluster_samples: int,
    max_event_minutes: float,
) -> list[tuple[int, int]]:
    work = group.sort_values("time").reset_index(drop=True)
    z = pd.to_numeric(work["local_normalized_power"], errors="coerce").to_numpy(dtype=float)
    high_pos = np.where(z >= float(threshold))[0]
    if high_pos.size == 0:
        return []
    times = pd.DatetimeIndex(work["time"])
    clusters: list[tuple[int, int]] = []
    start = int(high_pos[0])
    last = int(high_pos[0])
    for pos_value in high_pos[1:]:
        pos = int(pos_value)
        if (times[pos] - times[last]).total_seconds() <= float(max_gap_seconds):
            last = pos
        else:
            clusters.append((start, last))
            start = last = pos
    clusters.append((start, last))

    kept: list[tuple[int, int]] = []
    for start, end in clusters:
        high_count = int(np.count_nonzero(z[start : end + 1] >= float(threshold)))
        duration_min = (times[end] - times[start]).total_seconds() / 60.0
        if high_count >= int(min_cluster_samples) and duration_min <= float(max_event_minutes):
            kept.append((start, end))
    return kept


def build_burst_catalog(
    samples: pd.DataFrame,
    thresholds: pd.DataFrame,
    max_gap_seconds: float,
    min_cluster_samples: int,
    max_event_minutes: float,
) -> pd.DataFrame:
    threshold_map = {
        int(row.frequency_band): float(row.burst_threshold_local_normalized_power)
        for row in thresholds.itertuples(index=False)
    }
    rows: list[dict[str, object]] = []
    for band, group in samples.groupby("frequency_band", sort=True):
        work = group.sort_values("time").reset_index(drop=True)
        threshold = threshold_map[int(band)]
        clusters = cluster_high_samples(
            work,
            threshold=threshold,
            max_gap_seconds=max_gap_seconds,
            min_cluster_samples=min_cluster_samples,
            max_event_minutes=max_event_minutes,
        )
        for start, end in clusters:
            segment = work.iloc[start : end + 1].copy()
            high = segment[pd.to_numeric(segment["local_normalized_power"], errors="coerce") >= threshold].copy()
            if high.empty:
                continue
            peak = high.loc[pd.to_numeric(high["local_normalized_power"], errors="coerce").idxmax()]
            peak_phase = float(peak["io_phase_spice_deg"]) % 360.0
            start_time = pd.Timestamp(segment["time"].iloc[0])
            end_time = pd.Timestamp(segment["time"].iloc[-1])
            rows.append(
                {
                    "frequency_band": int(peak["frequency_band"]),
                    "frequency_mhz": float(peak["frequency_mhz"]),
                    "burst_start_time": start_time,
                    "burst_end_time": end_time,
                    "burst_peak_time": pd.Timestamp(peak["time"]),
                    "duration_min": float((end_time - start_time).total_seconds() / 60.0),
                    "n_high_samples": int(len(high)),
                    "threshold_local_normalized_power": float(threshold),
                    "peak_power": float(peak["power"]),
                    "peak_local_normalized_power": float(peak["local_normalized_power"]),
                    "io_phase_spice_deg": peak_phase,
                    "io_phase_window": phase_window_label(peak_phase),
                    "in_expected_io_window": bool(in_phase_windows(np.array([peak_phase]))[0]),
                    "jupiter_cml_spice_deg": float(peak["jupiter_cml_spice_deg"]),
                    "jupiter_beam_relative_gain_db": float(peak["jupiter_beam_relative_gain_db"]),
                    "jupiter_beam_separation_deg": float(peak["jupiter_beam_separation_deg"]),
                }
            )
    out = pd.DataFrame.from_records(rows)
    if out.empty:
        return out
    out = out.sort_values(["burst_peak_time", "frequency_mhz"]).reset_index(drop=True)
    out.insert(0, "burst_id", np.arange(len(out), dtype=int))
    return out


def annotate_cross_frequency_groups(bursts: pd.DataFrame, tolerance_seconds: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if bursts.empty:
        out = bursts.copy()
        out["cross_frequency_group_id"] = pd.Series(dtype=int)
        return out, pd.DataFrame()
    work = bursts.sort_values("burst_peak_time").reset_index(drop=True).copy()
    times = pd.DatetimeIndex(work["burst_peak_time"])
    tol = pd.Timedelta(seconds=float(tolerance_seconds))
    group_ids = np.full(len(work), -1, dtype=int)
    group_rows: list[dict[str, object]] = []
    start = 0
    group_id = 0
    while start < len(work):
        anchor_time = times[start]
        end = start
        while end + 1 < len(work) and times[end + 1] - anchor_time <= tol:
            end += 1
        group = work.iloc[start : end + 1].copy()
        group_ids[start : end + 1] = group_id
        peak = group.loc[pd.to_numeric(group["peak_local_normalized_power"], errors="coerce").idxmax()]
        freqs = sorted(group["frequency_mhz"].dropna().unique())
        group_rows.append(
            {
                "cross_frequency_group_id": int(group_id),
                "group_start_time": pd.Timestamp(group["burst_peak_time"].min()),
                "group_end_time": pd.Timestamp(group["burst_peak_time"].max()),
                "group_duration_s": float((pd.Timestamp(group["burst_peak_time"].max()) - pd.Timestamp(group["burst_peak_time"].min())).total_seconds()),
                "n_group_bursts": int(len(group)),
                "n_group_frequencies": int(len(freqs)),
                "group_frequency_min_mhz": float(np.min(freqs)) if freqs else np.nan,
                "group_frequency_max_mhz": float(np.max(freqs)) if freqs else np.nan,
                "group_frequency_span_mhz": float(np.max(freqs) - np.min(freqs)) if len(freqs) else np.nan,
                "group_peak_burst_id": int(peak["burst_id"]),
                "group_peak_time": pd.Timestamp(peak["burst_peak_time"]),
                "group_peak_frequency_mhz": float(peak["frequency_mhz"]),
                "group_peak_local_normalized_power": float(peak["peak_local_normalized_power"]),
                "group_peak_io_phase_deg": float(peak["io_phase_spice_deg"]),
                "group_peak_io_phase_window": str(peak["io_phase_window"]),
                "group_peak_in_expected_io_window": bool(peak["in_expected_io_window"]),
            }
        )
        group_id += 1
        start = end + 1
    work["cross_frequency_group_id"] = group_ids
    groups = pd.DataFrame(group_rows)
    annotated = work.merge(groups, on="cross_frequency_group_id", how="left")
    annotated = annotated.sort_values("burst_id").reset_index(drop=True)
    return annotated, groups


def phase_exposure_summary(samples: pd.DataFrame, bursts: pd.DataFrame, phase_bin_deg: float) -> pd.DataFrame:
    sample_bins = attach_phase_bins(samples, phase_bin_deg)
    burst_bins = attach_phase_bins(bursts, phase_bin_deg) if not bursts.empty else bursts.copy()
    phases = phase_bin_centers(phase_bin_deg)
    freqs = sample_bins[["frequency_band", "frequency_mhz"]].drop_duplicates().sort_values("frequency_mhz")
    full = pd.MultiIndex.from_product(
        [freqs["frequency_band"].to_numpy(dtype=int), phases],
        names=["frequency_band", "io_phase_bin_deg"],
    ).to_frame(index=False)
    full = full.merge(freqs, on="frequency_band", how="left")
    exposure = (
        sample_bins.groupby(["frequency_band", "io_phase_bin_deg"], sort=True)
        .size()
        .rename("n_exposure_samples")
        .reset_index()
    )
    burst_counts = pd.DataFrame(columns=["frequency_band", "io_phase_bin_deg", "n_bursts"])
    if not burst_bins.empty:
        burst_counts = (
            burst_bins.groupby(["frequency_band", "io_phase_bin_deg"], sort=True)
            .size()
            .rename("n_bursts")
            .reset_index()
        )
    out = full.merge(exposure, on=["frequency_band", "io_phase_bin_deg"], how="left")
    out = out.merge(burst_counts, on=["frequency_band", "io_phase_bin_deg"], how="left")
    out[["n_exposure_samples", "n_bursts"]] = out[["n_exposure_samples", "n_bursts"]].fillna(0).astype(int)
    out["burst_rate_per_100k_samples"] = np.divide(
        out["n_bursts"] * 100000.0,
        out["n_exposure_samples"],
        out=np.full(len(out), np.nan),
        where=out["n_exposure_samples"].to_numpy(dtype=float) > 0.0,
    )
    totals = out.groupby("frequency_band")["n_exposure_samples"].transform("sum")
    out["exposure_fraction_in_frequency"] = np.divide(
        out["n_exposure_samples"],
        totals,
        out=np.full(len(out), np.nan, dtype=float),
        where=totals.to_numpy(dtype=float) > 0.0,
    )
    out["in_expected_io_window"] = in_phase_windows(out["io_phase_bin_deg"].to_numpy(dtype=float))
    out["io_phase_window"] = [phase_window_label(v) for v in out["io_phase_bin_deg"].to_numpy(dtype=float)]
    return out.sort_values(["frequency_mhz", "io_phase_bin_deg"]).reset_index(drop=True)


def _enrichment_row(label: str, frequency_band: object, frequency_mhz: object, samples: pd.DataFrame, bursts: pd.DataFrame) -> dict[str, object]:
    sample_expected = in_phase_windows(samples["io_phase_spice_deg"].to_numpy(dtype=float))
    n_sample_expected = int(np.count_nonzero(sample_expected))
    n_sample_control = int(len(samples) - n_sample_expected)
    if bursts.empty:
        n_burst_expected = 0
        n_burst_control = 0
    else:
        burst_expected = in_phase_windows(bursts["io_phase_spice_deg"].to_numpy(dtype=float))
        n_burst_expected = int(np.count_nonzero(burst_expected))
        n_burst_control = int(len(bursts) - n_burst_expected)
    n_bursts = n_burst_expected + n_burst_control
    n_samples = n_sample_expected + n_sample_control
    exposure_fraction = n_sample_expected / n_samples if n_samples else np.nan
    burst_fraction = n_burst_expected / n_bursts if n_bursts else np.nan
    expected_rate = n_burst_expected / n_sample_expected * 100000.0 if n_sample_expected else np.nan
    control_rate = n_burst_control / n_sample_control * 100000.0 if n_sample_control else np.nan
    rate_ratio = ((n_burst_expected + 0.5) / (n_sample_expected + 0.5)) / (
        (n_burst_control + 0.5) / (n_sample_control + 0.5)
    ) if n_sample_expected and n_sample_control else np.nan
    p_positive = binomtest(n_burst_expected, n_bursts, exposure_fraction, alternative="greater").pvalue if n_bursts else np.nan
    p_two_sided = binomtest(n_burst_expected, n_bursts, exposure_fraction, alternative="two-sided").pvalue if n_bursts else np.nan
    return {
        "group": label,
        "frequency_band": frequency_band,
        "frequency_mhz": frequency_mhz,
        "n_exposure_samples": n_samples,
        "n_expected_io_exposure_samples": n_sample_expected,
        "n_control_io_exposure_samples": n_sample_control,
        "expected_io_exposure_fraction": exposure_fraction,
        "n_bursts": n_bursts,
        "n_expected_io_bursts": n_burst_expected,
        "n_control_io_bursts": n_burst_control,
        "expected_io_burst_fraction": burst_fraction,
        "expected_io_burst_rate_per_100k_samples": expected_rate,
        "control_io_burst_rate_per_100k_samples": control_rate,
        "expected_over_control_rate_ratio_add_half": rate_ratio,
        "binomial_p_positive_enrichment": p_positive,
        "binomial_p_two_sided": p_two_sided,
    }


def enrichment_summary(samples: pd.DataFrame, bursts: pd.DataFrame) -> pd.DataFrame:
    rows = [_enrichment_row("all_frequencies", "all", "all", samples, bursts)]
    for (band, freq), grp in samples.groupby(["frequency_band", "frequency_mhz"], sort=True):
        burst_grp = bursts[bursts["frequency_band"].eq(int(band))] if not bursts.empty else bursts
        rows.append(_enrichment_row("frequency", int(band), float(freq), grp, burst_grp))
    return pd.DataFrame(rows)


def phase_shift_control_summary(samples: pd.DataFrame, bursts: pd.DataFrame, shift_step_deg: float) -> pd.DataFrame:
    rows = []
    sample_phase = samples["io_phase_spice_deg"].to_numpy(dtype=float)
    burst_phase = bursts["io_phase_spice_deg"].to_numpy(dtype=float) if not bursts.empty else np.array([], dtype=float)
    for shift in np.arange(0.0, 360.0, float(shift_step_deg)):
        windows = shifted_windows(float(shift))
        sample_expected = in_phase_windows(sample_phase, windows=windows)
        burst_expected = in_phase_windows(burst_phase, windows=windows) if burst_phase.size else np.array([], dtype=bool)
        n_sample_expected = int(np.count_nonzero(sample_expected))
        n_sample_control = int(len(samples) - n_sample_expected)
        n_burst_expected = int(np.count_nonzero(burst_expected))
        n_burst_control = int(len(bursts) - n_burst_expected)
        exposure_fraction = n_sample_expected / len(samples) if len(samples) else np.nan
        burst_fraction = n_burst_expected / len(bursts) if len(bursts) else np.nan
        expected_rate = n_burst_expected / n_sample_expected * 100000.0 if n_sample_expected else np.nan
        control_rate = n_burst_control / n_sample_control * 100000.0 if n_sample_control else np.nan
        rate_ratio = (
            ((n_burst_expected + 0.5) / (n_sample_expected + 0.5))
            / ((n_burst_control + 0.5) / (n_sample_control + 0.5))
            if n_sample_expected and n_sample_control
            else np.nan
        )
        p_positive = (
            binomtest(n_burst_expected, len(bursts), exposure_fraction, alternative="greater").pvalue
            if len(bursts)
            else np.nan
        )
        rows.append(
            {
                "shift_deg": float(shift),
                "n_expected_io_exposure_samples": n_sample_expected,
                "n_expected_io_bursts": n_burst_expected,
                "expected_io_exposure_fraction": exposure_fraction,
                "expected_io_burst_fraction": burst_fraction,
                "expected_io_burst_rate_per_100k_samples": expected_rate,
                "control_io_burst_rate_per_100k_samples": control_rate,
                "expected_over_control_rate_ratio_add_half": rate_ratio,
                "binomial_p_positive_enrichment": p_positive,
                "is_nominal_expected_window": bool(np.isclose(float(shift), 0.0)),
            }
        )
    return pd.DataFrame(rows)


def decorate_io_bands(ax: plt.Axes) -> None:
    colors = {"Io-B/D": "#d95f02", "Io-A/C": "#1b9e77"}
    for label, lo, hi in EXPECTED_IO_WINDOWS:
        color = colors[label]
        ax.axvspan(lo, hi, color=color, alpha=0.10, lw=0)
        ax.axvline(lo, color=color, lw=1.0, alpha=0.85)
        ax.axvline(hi, color=color, lw=1.0, alpha=0.85)
        ax.text(
            (lo + hi) / 2.0,
            0.98,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
            color=color,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.70, "pad": 1.0},
        )


def _fmt(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    if abs(value) >= 1:
        return f"{value:.2g}"
    return f"{value:.2f}"


def plot_burst_rate_heatmap(
    summary: pd.DataFrame,
    out_dir: Path,
    phase_bin_deg: float,
    filename: str = "jupiter_blind_burst_rate_io_phase_frequency_row_scaled.png",
    title: str = "Blind burst catalog: exposure-corrected burst rate by Io phase, row-scaled",
) -> Path:
    phases = phase_bin_centers(phase_bin_deg)
    mat = summary.pivot_table(
        index="frequency_mhz",
        columns="io_phase_bin_deg",
        values="burst_rate_per_100k_samples",
        aggfunc="first",
    ).reindex(columns=phases)
    vals = mat.to_numpy(dtype=float)
    scaled = np.full_like(vals, np.nan, dtype=float)
    ylabels = []
    for idx, freq in enumerate(mat.index):
        row = vals[idx]
        finite = row[np.isfinite(row)]
        if finite.size == 0:
            ylabels.append(f"{freq:.2f}\nno data")
            continue
        lo = float(np.nanmin(finite))
        hi = float(np.nanpercentile(finite, 98)) if finite.size >= 4 else float(np.nanmax(finite))
        if not np.isfinite(hi) or np.isclose(lo, hi):
            hi = float(np.nanmax(finite))
        if not np.isfinite(hi) or np.isclose(lo, hi):
            scaled[idx] = 0.0
            ylabels.append(f"{freq:.2f}\nflat")
        else:
            scaled[idx] = np.clip((row - lo) / (hi - lo), 0.0, 1.0)
            ylabels.append(f"{freq:.2f}\n{_fmt(lo)}-{_fmt(hi)}")
    fig, ax = plt.subplots(figsize=(12.8, 5.8))
    im = ax.imshow(
        scaled,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[0.0, 360.0, -0.5, len(mat.index) - 0.5],
        cmap="magma",
        norm=Normalize(vmin=0.0, vmax=1.0),
    )
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels(ylabels)
    ax.set_xlim(0, 360)
    ax.set_xticks(np.arange(0, 361, 30))
    ax.set_xlabel("Io phase (deg)")
    ax.set_ylabel("frequency (MHz)\nrow rate range")
    ax.set_title(title)
    decorate_io_bands(ax)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="row-scaled burst rate")
    fig.tight_layout()
    path = out_dir / filename
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_burst_phase_scatter(
    bursts: pd.DataFrame,
    out_dir: Path,
    filename: str = "jupiter_blind_burst_peak_io_phase_scatter.png",
    title: str = "Blind burst peak phases after upper-V beam selection",
) -> Path | None:
    if bursts.empty:
        return None
    fig, ax = plt.subplots(figsize=(12.8, 5.8))
    sizes = 18.0 + 34.0 * np.clip(np.log10(bursts["peak_local_normalized_power"].to_numpy(dtype=float) + 1.0), 0, 4) / 4.0
    sc = ax.scatter(
        bursts["io_phase_spice_deg"],
        bursts["frequency_mhz"],
        c=np.log10(bursts["peak_local_normalized_power"].to_numpy(dtype=float) + 1.0),
        s=sizes,
        cmap="viridis",
        alpha=0.72,
        edgecolor="white",
        linewidth=0.35,
    )
    decorate_io_bands(ax)
    ax.set_xlim(0, 360)
    ax.set_xticks(np.arange(0, 361, 30))
    freqs = sorted(bursts["frequency_mhz"].dropna().unique())
    ax.set_yticks(freqs)
    ax.set_yticklabels([f"{v:.2f}" for v in freqs])
    ax.set_xlabel("Io phase at burst peak (deg)")
    ax.set_ylabel("frequency (MHz)")
    ax.set_title(title)
    ax.grid(True, color="0.9", lw=0.45)
    fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.02, label="log10(peak local-normalized power + 1)")
    fig.tight_layout()
    path = out_dir / filename
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_burst_rate_phase_panels(
    summary: pd.DataFrame,
    out_dir: Path,
    phase_bin_deg: float,
    filename: str = "jupiter_blind_burst_rate_by_io_phase_per_frequency.png",
    title: str = "Per-frequency blind burst rate by Io phase",
) -> Path:
    freqs = summary[["frequency_band", "frequency_mhz"]].drop_duplicates().sort_values("frequency_mhz")
    n_freq = len(freqs)
    n_cols = 3
    n_rows = int(np.ceil(n_freq / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14.0, 3.0 * n_rows), sharex=True)
    axes_arr = np.asarray(axes).reshape(-1)
    bin_width = float(phase_bin_deg) * 0.86
    for ax, row in zip(axes_arr, freqs.itertuples(index=False)):
        band = int(row.frequency_band)
        freq = float(row.frequency_mhz)
        work = summary[summary["frequency_band"].eq(band)].sort_values("io_phase_bin_deg")
        phases = work["io_phase_bin_deg"].to_numpy(dtype=float)
        rates = work["burst_rate_per_100k_samples"].to_numpy(dtype=float)
        colors = np.where(work["in_expected_io_window"].to_numpy(dtype=bool), "#d95f02", "0.62")
        ax.bar(phases, rates, width=bin_width, color=colors, alpha=0.82, edgecolor="white", linewidth=0.35)
        finite = rates[np.isfinite(rates)]
        ymax = float(np.nanmax(finite)) if finite.size else 1.0
        ax.set_ylim(0.0, ymax * 1.18 if ymax > 0 else 1.0)
        decorate_io_bands(ax)
        ax.set_title(f"{freq:.2f} MHz")
        ax.grid(True, axis="y", color="0.90", lw=0.45)
        n_bursts = int(work["n_bursts"].sum())
        n_exposure = int(work["n_exposure_samples"].sum())
        ax.text(
            0.02,
            0.96,
            f"{n_bursts} bursts\n{n_exposure:,} samples",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.76, "pad": 1.0},
        )
    for ax in axes_arr[n_freq:]:
        ax.axis("off")
    for ax in axes_arr[:n_freq]:
        ax.set_xlim(0, 360)
        ax.set_xticks(np.arange(0, 361, 60))
    for ax in axes_arr[::n_cols]:
        ax.set_ylabel("bursts / 100k samples")
    for ax in axes_arr[max(0, (n_rows - 1) * n_cols) : n_rows * n_cols]:
        if ax.has_data():
            ax.set_xlabel("Io phase (deg)")
    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    path = out_dir / filename
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_phase_shift_control(
    shift_summary: pd.DataFrame,
    out_dir: Path,
    filename: str = "jupiter_blind_burst_shifted_io_window_control.png",
    title: str = "Shifted-window control for blind burst Io-phase enrichment",
) -> Path:
    fig, ax = plt.subplots(figsize=(11.2, 4.8))
    ax.plot(
        shift_summary["shift_deg"],
        shift_summary["expected_over_control_rate_ratio_add_half"],
        marker="o",
        ms=4.0,
        lw=1.25,
        color="#386cb0",
    )
    ax.axhline(1.0, color="0.35", lw=0.9)
    ax.axvline(0.0, color="#d95f02", lw=1.3, label="nominal Io windows")
    ax.set_xlim(0, 350)
    ax.set_xticks(np.arange(0, 360, 30))
    ax.set_xlabel("phase shift applied to both expected Io windows (deg)")
    ax.set_ylabel("expected-window / control burst-rate ratio")
    ax.set_title(title)
    ax.grid(True, color="0.9", lw=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = out_dir / filename
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_enrichment(
    enrich: pd.DataFrame,
    out_dir: Path,
    filename: str = "jupiter_blind_burst_expected_io_window_enrichment.png",
    title: str = "Blind burst enrichment in expected Io phase windows",
) -> Path:
    work = enrich[enrich["group"].eq("frequency")].copy()
    x = np.arange(len(work))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    ax.bar(
        x - width / 2,
        work["expected_io_burst_rate_per_100k_samples"],
        width=width,
        label="expected Io windows",
        color="#d95f02",
        alpha=0.82,
    )
    ax.bar(
        x + width / 2,
        work["control_io_burst_rate_per_100k_samples"],
        width=width,
        label="other Io phases",
        color="0.55",
        alpha=0.72,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{float(v):.2f}" for v in work["frequency_mhz"]])
    ax.set_xlabel("frequency (MHz)")
    ax.set_ylabel("bursts per 100k exposure samples")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(True, axis="y", color="0.88", lw=0.55)
    for idx, row in enumerate(work.itertuples(index=False)):
        p = float(row.binomial_p_positive_enrichment)
        if np.isfinite(p) and p < 0.05:
            ymax = max(float(row.expected_io_burst_rate_per_100k_samples), float(row.control_io_burst_rate_per_100k_samples))
            ax.text(idx, ymax * 1.05 if ymax > 0 else 0.05, f"p={p:.3g}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    path = out_dir / filename
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report(
    out_dir: Path,
    config: dict[str, object],
    thresholds: pd.DataFrame,
    enrich: pd.DataFrame,
    paths: list[Path],
) -> Path:
    lines = [
        "# Jupiter Per-Frequency Blind Burst Io-Phase Enrichment",
        "",
        "Bursts are selected without using Io phase and without requiring broadband coincidence.  Within each frequency, a sample is high if `local_normalized_power` exceeds the larger of the configured per-frequency quantile and the configured minimum local-z floor.  Nearby high samples are then clustered into burst events.",
        "",
        "After the burst catalog is fixed, Io phase is used for exposure-corrected tests.  The expected Io windows are `80-130 deg` and `205-260 deg`.",
        "",
        "## Configuration",
        "",
        *[f"- `{k}`: `{v}`" for k, v in config.items() if k != "software_versions"],
        "",
        "## Burst Thresholds",
        "",
        thresholds.to_string(index=False),
        "",
        "## Io-Window Enrichment",
        "",
        enrich.to_string(index=False),
        "",
        "## Files",
        "",
        *[f"- `{p}`" for p in paths],
    ]
    path = out_dir / "jupiter_blind_burst_phase_enrichment_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def read_selected_samples(path: Path) -> pd.DataFrame:
    cols = [
        "time",
        "frequency_band",
        "frequency_mhz",
        "power",
        "local_normalized_power",
        "io_phase_spice_deg",
        "jupiter_cml_spice_deg",
        "jupiter_beam_relative_gain_db",
        "jupiter_beam_separation_deg",
    ]
    if path.suffix.lower() == ".npy":
        arr = np.load(path, allow_pickle=True)
        missing = [col for col in cols if col not in arr.dtype.names]
        if missing:
            raise ValueError(f"Missing selected-sample columns in {path}: {missing}")
        samples = pd.DataFrame({col: arr[col] for col in cols})
        samples["time"] = pd.to_datetime(samples["time"])
    else:
        samples = pd.read_csv(path, usecols=cols, parse_dates=["time"], low_memory=False)
    for col in [
        "frequency_band",
        "frequency_mhz",
        "power",
        "local_normalized_power",
        "io_phase_spice_deg",
        "jupiter_cml_spice_deg",
        "jupiter_beam_relative_gain_db",
        "jupiter_beam_separation_deg",
    ]:
        samples[col] = pd.to_numeric(samples[col], errors="coerce")
    samples = samples.dropna(subset=["time", "frequency_band", "frequency_mhz", "local_normalized_power", "io_phase_spice_deg"])
    samples = samples[np.isfinite(samples["local_normalized_power"])].copy()
    samples["frequency_band"] = samples["frequency_band"].astype(int)
    samples["io_phase_spice_deg"] = samples["io_phase_spice_deg"] % 360.0
    return samples.sort_values(["frequency_band", "time"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-samples", type=Path, default=DEFAULT_SELECTED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--threshold-quantile", type=float, default=0.995)
    parser.add_argument("--min-local-z", type=float, default=5.0)
    parser.add_argument("--max-gap-seconds", type=float, default=180.0)
    parser.add_argument("--min-cluster-samples", type=int, default=2)
    parser.add_argument("--max-event-minutes", type=float, default=30.0)
    parser.add_argument("--phase-bin-deg", type=float, default=10.0)
    parser.add_argument("--phase-shift-step-deg", type=float, default=10.0)
    parser.add_argument("--cross-frequency-tolerance-seconds", type=float, default=300.0)
    parser.add_argument("--min-cross-frequency-bands", type=int, default=2)
    parser.add_argument(
        "--skip-cross-frequency-products",
        action="store_true",
        help="Only write the per-frequency burst catalog and Io-phase tests; do not summarize broadband burst coincidence.",
    )
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    config = {
        "selected_samples": str(args.selected_samples),
        "threshold_quantile": float(args.threshold_quantile),
        "min_local_z": float(args.min_local_z),
        "max_gap_seconds": float(args.max_gap_seconds),
        "min_cluster_samples": int(args.min_cluster_samples),
        "max_event_minutes": float(args.max_event_minutes),
        "phase_bin_deg": float(args.phase_bin_deg),
        "phase_shift_step_deg": float(args.phase_shift_step_deg),
        "cross_frequency_tolerance_seconds": float(args.cross_frequency_tolerance_seconds),
        "min_cross_frequency_bands": int(args.min_cross_frequency_bands),
        "skip_cross_frequency_products": bool(args.skip_cross_frequency_products),
        "expected_io_windows": EXPECTED_IO_WINDOWS,
        "software_versions": software_versions(),
    }
    write_json(out_dir / "run_config.json", config)

    print("Reading beam-selected upper-V samples...", flush=True)
    samples = read_selected_samples(args.selected_samples)
    print(f"Loaded {len(samples):,} normalization-valid samples", flush=True)
    print("Selecting blind burst candidates...", flush=True)
    thresholds = threshold_table(samples, quantile=float(args.threshold_quantile), min_local_z=float(args.min_local_z))
    bursts = build_burst_catalog(
        samples,
        thresholds,
        max_gap_seconds=float(args.max_gap_seconds),
        min_cluster_samples=int(args.min_cluster_samples),
        max_event_minutes=float(args.max_event_minutes),
    )
    print(f"Built {len(bursts):,} burst events", flush=True)
    phase_summary = phase_exposure_summary(samples, bursts, phase_bin_deg=float(args.phase_bin_deg))
    enrich = enrichment_summary(samples, bursts)
    enrich.insert(0, "event_set", "all_blind_bursts")
    combined_enrich = enrich.copy()
    shift_summary = phase_shift_control_summary(samples, bursts, shift_step_deg=float(args.phase_shift_step_deg))

    thresholds.to_csv(out_dir / "jupiter_blind_burst_thresholds.csv", index=False)
    bursts.to_csv(out_dir / "jupiter_blind_burst_catalog.csv", index=False)
    phase_summary.to_csv(out_dir / "jupiter_blind_burst_io_phase_exposure_summary.csv", index=False)
    combined_enrich.to_csv(out_dir / "jupiter_blind_burst_expected_io_enrichment.csv", index=False)
    shift_summary.to_csv(out_dir / "jupiter_blind_burst_shifted_io_window_control.csv", index=False)

    print("Making plots...", flush=True)
    paths: list[Path] = [
        out_dir / "run_config.json",
        out_dir / "jupiter_blind_burst_thresholds.csv",
        out_dir / "jupiter_blind_burst_catalog.csv",
        out_dir / "jupiter_blind_burst_io_phase_exposure_summary.csv",
        out_dir / "jupiter_blind_burst_expected_io_enrichment.csv",
        out_dir / "jupiter_blind_burst_shifted_io_window_control.csv",
    ]
    paths.append(plot_burst_rate_heatmap(phase_summary, out_dir, phase_bin_deg=float(args.phase_bin_deg)))
    paths.append(plot_burst_rate_phase_panels(phase_summary, out_dir, phase_bin_deg=float(args.phase_bin_deg)))
    scatter = plot_burst_phase_scatter(bursts, out_dir)
    if scatter is not None:
        paths.append(scatter)
    paths.append(plot_enrichment(enrich, out_dir))
    paths.append(plot_phase_shift_control(shift_summary, out_dir))
    if not args.skip_cross_frequency_products:
        bursts_with_groups, burst_groups = annotate_cross_frequency_groups(
            bursts,
            tolerance_seconds=float(args.cross_frequency_tolerance_seconds),
        )
        multi_bursts = bursts_with_groups[
            bursts_with_groups["n_group_frequencies"].ge(int(args.min_cross_frequency_bands))
        ].copy()
        multi_groups = burst_groups[burst_groups["n_group_frequencies"].ge(int(args.min_cross_frequency_bands))].copy()
        print(
            f"Cross-frequency diagnostic subset: {len(multi_bursts):,} burst events in {len(multi_groups):,} groups with "
            f">= {int(args.min_cross_frequency_bands)} frequency bands",
            flush=True,
        )
        multi_phase_summary = phase_exposure_summary(samples, multi_bursts, phase_bin_deg=float(args.phase_bin_deg))
        multi_enrich = enrichment_summary(samples, multi_bursts)
        multi_enrich.insert(0, "event_set", f"cross_frequency_ge_{int(args.min_cross_frequency_bands)}")
        multi_shift_summary = phase_shift_control_summary(samples, multi_bursts, shift_step_deg=float(args.phase_shift_step_deg))

        bursts_with_groups.to_csv(out_dir / "jupiter_blind_burst_cross_frequency_diagnostic_groups.csv", index=False)
        multi_bursts.to_csv(out_dir / "jupiter_blind_multifrequency_diagnostic_burst_catalog.csv", index=False)
        multi_groups.to_csv(out_dir / "jupiter_blind_multifrequency_diagnostic_groups.csv", index=False)
        multi_phase_summary.to_csv(
            out_dir / "jupiter_blind_multifrequency_diagnostic_io_phase_exposure_summary.csv", index=False
        )
        multi_shift_summary.to_csv(out_dir / "jupiter_blind_multifrequency_diagnostic_shifted_io_window_control.csv", index=False)
        pd.concat([enrich, multi_enrich], ignore_index=True).to_csv(
            out_dir / "jupiter_blind_burst_expected_io_enrichment_with_cross_frequency_diagnostic.csv",
            index=False,
        )
        paths.extend(
            [
                out_dir / "jupiter_blind_burst_cross_frequency_diagnostic_groups.csv",
                out_dir / "jupiter_blind_multifrequency_diagnostic_burst_catalog.csv",
                out_dir / "jupiter_blind_multifrequency_diagnostic_groups.csv",
                out_dir / "jupiter_blind_multifrequency_diagnostic_io_phase_exposure_summary.csv",
                out_dir / "jupiter_blind_multifrequency_diagnostic_shifted_io_window_control.csv",
                out_dir / "jupiter_blind_burst_expected_io_enrichment_with_cross_frequency_diagnostic.csv",
            ]
        )
        paths.append(
            plot_burst_rate_heatmap(
                multi_phase_summary,
                out_dir,
                phase_bin_deg=float(args.phase_bin_deg),
                filename="jupiter_blind_multifrequency_diagnostic_rate_io_phase_frequency_row_scaled.png",
                title="Blind multi-frequency diagnostic: exposure-corrected burst rate by Io phase, row-scaled",
            )
        )
        multi_scatter = plot_burst_phase_scatter(
            multi_bursts,
            out_dir,
            filename="jupiter_blind_multifrequency_diagnostic_peak_io_phase_scatter.png",
            title="Blind multi-frequency diagnostic burst peak phases after upper-V beam selection",
        )
        if multi_scatter is not None:
            paths.append(multi_scatter)
        paths.append(
            plot_enrichment(
                multi_enrich,
                out_dir,
                filename="jupiter_blind_multifrequency_diagnostic_expected_io_window_enrichment.png",
                title="Blind multi-frequency diagnostic enrichment in expected Io phase windows",
            )
        )
        paths.append(
            plot_phase_shift_control(
                multi_shift_summary,
                out_dir,
                filename="jupiter_blind_multifrequency_diagnostic_shifted_io_window_control.png",
                title="Shifted-window control for blind multi-frequency diagnostic Io-phase enrichment",
            )
        )
    report = write_report(out_dir, config=config, thresholds=thresholds, enrich=combined_enrich, paths=paths)
    print(f"Wrote {report}", flush=True)


if __name__ == "__main__":
    main()
