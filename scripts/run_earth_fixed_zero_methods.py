#!/usr/bin/env python
"""Demonstrate fixed-zero, morphology-preserving Earth detection methods."""

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

from rylevonberg.table_io import write_table  # noqa: E402
from rylevonberg.occultogram import fold_occultogram_points  # noqa: E402


DEFAULT_PROFILES = Path("/pscratch/sd/a/afahs/RAE2/occultograms_morphology_v1/occultogram_event_profiles.parquet")
DEFAULT_OUT = Path("/pscratch/sd/a/afahs/RAE2/earth_fixed_zero_methods_v2")
TRACER_FREQUENCIES = (0.70, 0.90, 1.31, 2.20)
CLEAR_EARTH_FREQUENCIES = (0.45, 3.93, 6.55, 9.18)
FOREGROUND_TARGETS = (0.45, 3.93, 6.55, 9.18)
EDGE_WINDOWS_SECONDS = (180.0, 300.0, 600.0, 1200.0)
PRIMARY_WINDOW_SECONDS = 600.0


def _huber_fit(x: np.ndarray, y: np.ndarray, max_iter: int = 40) -> np.ndarray:
    finite = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    x = x[finite]
    y = y[finite]
    if len(y) < x.shape[1] + 2:
        return np.full(x.shape[1], np.nan)
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    for _ in range(max_iter):
        residual = y - x @ beta
        center = np.median(residual)
        scale = 1.4826 * np.median(np.abs(residual - center))
        if not np.isfinite(scale) or scale <= 1e-10:
            break
        u = np.abs(residual - center) / (1.345 * scale)
        weights = np.ones_like(u)
        large = u > 1.0
        weights[large] = 1.0 / u[large]
        root_w = np.sqrt(weights)
        updated = np.linalg.lstsq(x * root_w[:, None], y * root_w, rcond=None)[0]
        if np.max(np.abs(updated - beta)) <= 1e-8 * (1.0 + np.max(np.abs(beta))):
            beta = updated
            break
        beta = updated
    return beta


def _block_profiles(points: pd.DataFrame, include_event_type: bool = False) -> pd.DataFrame:
    keys = ["lunar_block"]
    if include_event_type:
        keys.append("event_type")
    keys.extend(["frequency_mhz", "antenna", "control_shift_s", "t_bin_sec"])
    return (
        points.groupby(keys, observed=True, sort=False)
        .agg(value=("normalized_power", "median"), n_events=("event_uid", "nunique"))
        .reset_index()
    )


def _fixed_zero_amplitudes(
    profiles: pd.DataFrame,
    windows: tuple[float, ...],
    include_event_type: bool = False,
) -> pd.DataFrame:
    group_columns = ["lunar_block"]
    if include_event_type:
        group_columns.append("event_type")
    group_columns.extend(["frequency_mhz", "antenna", "control_shift_s"])
    records: list[dict[str, object]] = []
    for keys, group in profiles.groupby(group_columns, observed=True, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_columns, keys))
        t_all = pd.to_numeric(group["t_bin_sec"], errors="coerce").to_numpy(dtype=float)
        y_all = pd.to_numeric(group["value"], errors="coerce").to_numpy(dtype=float)
        for window in windows:
            use = np.isfinite(t_all) & np.isfinite(y_all) & (np.abs(t_all) <= window)
            t = t_all[use]
            y = y_all[use]
            edge = np.where(t < 0.0, 0.5, np.where(t > 0.0, -0.5, 0.0))
            design = np.column_stack([np.ones_like(t), t / window, edge])
            beta = _huber_fit(design, y)
            records.append(
                {
                    **base,
                    "window_s": float(window),
                    "edge_amplitude": float(beta[2]),
                    "baseline_slope_per_window": float(beta[1]),
                    "n_time_bins": int(np.isfinite(y).sum()),
                }
            )
    return pd.DataFrame.from_records(records)


def _channel_summary(amplitudes: pd.DataFrame) -> pd.DataFrame:
    keys = ["frequency_mhz", "antenna", "window_s"]
    records: list[dict[str, object]] = []
    for group_key, group in amplitudes.groupby(keys, observed=True, sort=True):
        frequency, antenna, window = group_key
        target = group[np.isclose(group["control_shift_s"], 0.0)]["edge_amplitude"].dropna()
        controls = group[~np.isclose(group["control_shift_s"], 0.0)]["edge_amplitude"].dropna()
        if target.empty or controls.empty:
            continue
        records.append(
            {
                "frequency_mhz": float(frequency),
                "antenna": str(antenna),
                "window_s": float(window),
                "target_q16": float(target.quantile(0.16)),
                "target_median": float(target.median()),
                "target_q84": float(target.quantile(0.84)),
                "control_q16": float(controls.quantile(0.16)),
                "control_median": float(controls.median()),
                "control_q84": float(controls.quantile(0.84)),
                "n_target_blocks": int(target.size),
            }
        )
    return pd.DataFrame.from_records(records)


def _empirical_null(amplitudes: pd.DataFrame, permutations: int, seed: int) -> pd.DataFrame:
    primary = amplitudes[np.isclose(amplitudes["window_s"], PRIMARY_WINDOW_SECONDS)]
    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    for (frequency, antenna), group in primary.groupby(["frequency_mhz", "antenna"], observed=True):
        pivot = group.pivot_table(
            index="lunar_block",
            columns="control_shift_s",
            values="edge_amplitude",
            aggfunc="median",
        ).dropna(axis=0, how="any")
        shifts = np.asarray(pivot.columns, dtype=float)
        target_indices = np.flatnonzero(np.isclose(shifts, 0.0))
        if pivot.empty or target_indices.size != 1:
            continue
        values = pivot.to_numpy(dtype=float)
        observed = float(np.median(values[:, target_indices[0]]))
        choices = rng.integers(0, values.shape[1], size=(permutations, values.shape[0]))
        selected = values[np.arange(values.shape[0])[None, :], choices]
        null_values = np.median(selected, axis=1)
        records.append(
            {
                "frequency_mhz": float(frequency),
                "antenna": str(antenna),
                "observed_amplitude": observed,
                "null_q16": float(np.quantile(null_values, 0.16)),
                "null_median": float(np.median(null_values)),
                "null_q84": float(np.quantile(null_values, 0.84)),
                "target_percentile": float(100.0 * (1 + np.sum(null_values <= observed)) / (permutations + 1)),
                "one_sided_empirical_p": float((1 + np.sum(null_values >= observed)) / (permutations + 1)),
                "n_complete_blocks": int(values.shape[0]),
                "permutations": int(permutations),
            }
        )
    return pd.DataFrame.from_records(records)


def _multiscale_summary(amplitudes: pd.DataFrame, permutations: int, seed: int) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for window in EDGE_WINDOWS_SECONDS:
        subset = amplitudes[np.isclose(amplitudes["window_s"], window)]
        current = _empirical_null(subset.assign(window_s=PRIMARY_WINDOW_SECONDS), permutations, seed + int(window))
        current["window_s"] = float(window)
        records.append(current)
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def _joint_block_consistency(type_amplitudes: pd.DataFrame) -> pd.DataFrame:
    selected = type_amplitudes[
        np.isclose(type_amplitudes["window_s"], PRIMARY_WINDOW_SECONDS)
        & type_amplitudes["frequency_mhz"].apply(
            lambda value: any(np.isclose(float(value), frequency) for frequency in CLEAR_EARTH_FREQUENCIES)
        )
    ].copy()
    return (
        selected.groupby(["lunar_block", "event_type", "control_shift_s"], observed=True)["edge_amplitude"]
        .median()
        .reset_index(name="joint_edge_amplitude")
    )


def _foreground_residuals(profiles: pd.DataFrame) -> pd.DataFrame:
    wide = profiles.pivot_table(
        index=["lunar_block", "antenna", "control_shift_s", "t_bin_sec"],
        columns="frequency_mhz",
        values="value",
        aggfunc="median",
    )
    frequency_columns = [float(column) for column in wide.columns]
    wide.columns = frequency_columns
    tracer_columns = [
        column for column in frequency_columns if any(np.isclose(column, tracer) for tracer in TRACER_FREQUENCIES)
    ]
    wide = wide.reset_index()
    wide["foreground_tracer"] = wide[tracer_columns].median(axis=1, skipna=True)
    records: list[dict[str, object]] = []
    for (block, antenna), group in wide.groupby(["lunar_block", "antenna"], observed=True, sort=False):
        tracer = pd.to_numeric(group["foreground_tracer"], errors="coerce").to_numpy(dtype=float)
        shift = pd.to_numeric(group["control_shift_s"], errors="coerce").to_numpy(dtype=float)
        time = pd.to_numeric(group["t_bin_sec"], errors="coerce").to_numpy(dtype=float)
        training_geometry = ~np.isclose(shift, 0.0) | (np.abs(time) >= PRIMARY_WINDOW_SECONDS)
        for requested_frequency in FOREGROUND_TARGETS:
            matching = [column for column in frequency_columns if np.isclose(column, requested_frequency)]
            if not matching:
                continue
            frequency = matching[0]
            y = pd.to_numeric(group[frequency], errors="coerce").to_numpy(dtype=float)
            train = training_geometry & np.isfinite(y) & np.isfinite(tracer)
            beta = _huber_fit(np.column_stack([np.ones(train.sum()), tracer[train]]), y[train])
            target = np.isclose(shift, 0.0) & np.isfinite(y) & np.isfinite(tracer)
            for t_value, raw_value, tracer_value in zip(time[target], y[target], tracer[target]):
                prediction = beta[0] + beta[1] * tracer_value
                records.append(
                    {
                        "lunar_block": block,
                        "antenna": str(antenna),
                        "frequency_mhz": float(frequency),
                        "t_bin_sec": float(t_value),
                        "raw_value": float(raw_value),
                        "foreground_prediction": float(prediction),
                        "residual_value": float(raw_value - prediction),
                    }
                )
    return pd.DataFrame.from_records(records)


def _foreground_summary(residuals: pd.DataFrame) -> pd.DataFrame:
    keys = ["antenna", "frequency_mhz", "t_bin_sec"]
    return (
        residuals.groupby(keys, observed=True)
        .agg(
            raw_median=("raw_value", "median"),
            raw_q16=("raw_value", lambda values: values.quantile(0.16)),
            raw_q84=("raw_value", lambda values: values.quantile(0.84)),
            residual_median=("residual_value", "median"),
            residual_q16=("residual_value", lambda values: values.quantile(0.16)),
            residual_q84=("residual_value", lambda values: values.quantile(0.84)),
            n_blocks=("lunar_block", "nunique"),
        )
        .reset_index()
    )


def _plot_fixed_zero(summary: pd.DataFrame, output: Path) -> None:
    primary = summary[np.isclose(summary["window_s"], PRIMARY_WINDOW_SECONDS)]
    antennas = sorted(primary["antenna"].unique())
    fig, axes = plt.subplots(1, len(antennas), figsize=(12, 4.5), sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, antenna in zip(axes, antennas):
        data = primary[primary["antenna"].eq(antenna)].sort_values("frequency_mhz")
        x = data["frequency_mhz"].to_numpy(dtype=float)
        target = data["target_median"].to_numpy(dtype=float)
        ax.fill_between(x, data["control_q16"], data["control_q84"], color="0.80", label="fixed-time controls")
        ax.plot(x, data["control_median"], color="0.45", linestyle="--", linewidth=1.2)
        ax.errorbar(
            x,
            target,
            yerr=np.vstack([target - data["target_q16"], data["target_q84"] - target]),
            color="#1f77b4",
            marker="o",
            capsize=3,
            label="predicted time",
        )
        ax.axhline(0.0, color="0.2", linewidth=0.8)
        ax.set_title(antenna)
        ax.set_xlabel("frequency (MHz)")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("fixed-zero edge amplitude\n(high-before minus low-after)")
    axes[0].legend(frameon=False)
    fig.suptitle("Earth: robust fixed-zero edge versus time-shift controls")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_empirical_null(nulls: pd.DataFrame, output: Path) -> None:
    antennas = sorted(nulls["antenna"].unique())
    fig, axes = plt.subplots(1, len(antennas), figsize=(12, 4.5), sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, antenna in zip(axes, antennas):
        data = nulls[nulls["antenna"].eq(antenna)].sort_values("frequency_mhz")
        x = data["frequency_mhz"].to_numpy(dtype=float)
        y = data["target_percentile"].to_numpy(dtype=float)
        ax.plot(x, y, color="#2ca02c", marker="o")
        ax.axhspan(95.0, 100.0, color="#2ca02c", alpha=0.12)
        ax.axhline(95.0, color="0.35", linestyle="--", linewidth=1.0)
        for x_value, y_value in zip(x, y):
            ax.annotate(f"{y_value:.0f}", (x_value, y_value), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8)
        ax.set_ylim(0.0, 103.0)
        ax.set_title(antenna)
        ax.set_xlabel("frequency (MHz)")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("predicted-time percentile\namong block-label permutations")
    fig.suptitle("Earth: empirical null with fixed edge time")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_multiscale(summary: pd.DataFrame, output: Path) -> None:
    antennas = sorted(summary["antenna"].unique())
    fig, axes = plt.subplots(1, len(antennas), figsize=(12, 6), constrained_layout=True)
    axes = np.atleast_1d(axes)
    image = None
    for ax, antenna in zip(axes, antennas):
        data = summary[summary["antenna"].eq(antenna)]
        pivot = data.pivot_table(index="frequency_mhz", columns="window_s", values="target_percentile").sort_index()
        image = ax.imshow(pivot.to_numpy(), aspect="auto", origin="lower", vmin=0, vmax=100, cmap="viridis")
        ax.set_xticks(range(len(pivot.columns)), [f"±{int(value / 60)}m" for value in pivot.columns])
        ax.set_yticks(range(len(pivot.index)), [f"{value:.2f}" for value in pivot.index])
        for row in range(len(pivot.index)):
            for column in range(len(pivot.columns)):
                value = pivot.iloc[row, column]
                ax.text(column, row, f"{value:.0f}", ha="center", va="center", color="white" if value < 45 else "black", fontsize=8)
        ax.set_title(antenna)
        ax.set_xlabel("fixed-zero fit half-window")
    axes[0].set_ylabel("frequency (MHz)")
    if image is not None:
        colorbar = fig.colorbar(image, ax=axes, shrink=0.85)
        colorbar.set_label("empirical control percentile")
    fig.suptitle("Earth: multiscale persistence of the zero-centered edge")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_joint_blocks(joint: pd.DataFrame, output: Path) -> None:
    event_types = sorted(joint["event_type"].astype(str).unique())
    fig, axes = plt.subplots(len(event_types), 1, figsize=(12, 3.2 * len(event_types)), sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, event_type in zip(axes, event_types):
        data = joint[joint["event_type"].astype(str).eq(event_type)].copy()
        target = data[np.isclose(data["control_shift_s"], 0.0)].sort_values("lunar_block")
        controls = data[~np.isclose(data["control_shift_s"], 0.0)]
        ax.scatter(controls["lunar_block"], controls["joint_edge_amplitude"], color="0.72", s=13, alpha=0.55, label="controls")
        ax.plot(target["lunar_block"], target["joint_edge_amplitude"], color="#d62728", marker="o", linewidth=1.4, label="predicted time")
        ax.axhline(0.0, color="0.2", linewidth=0.8)
        ax.set_ylabel("joint edge amplitude")
        ax.set_title(str(event_type))
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("29.53059-day lunar block")
    axes[0].legend(frameon=False)
    fig.suptitle("Earth: disappearance/reappearance and lunar-block consistency")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_foreground(summary: pd.DataFrame, output_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for antenna in sorted(summary["antenna"].unique()):
        subset = summary[summary["antenna"].eq(antenna)]
        frequencies = sorted(subset["frequency_mhz"].unique())
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, constrained_layout=True)
        for ax, frequency in zip(axes.ravel(), frequencies):
            data = subset[np.isclose(subset["frequency_mhz"], frequency)].sort_values("t_bin_sec")
            t = data["t_bin_sec"].to_numpy(dtype=float)
            ax.plot(t, data["raw_median"], color="0.55", linewidth=1.2, label="raw block median")
            ax.plot(t, data["residual_median"], color="#9467bd", linewidth=1.8, label="foreground residual")
            ax.fill_between(t, data["residual_q16"], data["residual_q84"], color="#9467bd", alpha=0.16)
            ax.axvline(0.0, color="0.15", linestyle="--", linewidth=0.9)
            ax.axhline(0.0, color="0.35", linewidth=0.6)
            ax.set_title(f"{frequency:.2f} MHz")
            ax.grid(alpha=0.18)
        axes[1, 0].set_xlabel("time from predicted edge (s)")
        axes[1, 1].set_xlabel("time from predicted edge (s)")
        axes[0, 0].set_ylabel("normalized power")
        axes[1, 0].set_ylabel("normalized power")
        axes[0, 0].legend(frameon=False)
        fig.suptitle(f"Earth {antenna}: cross-frequency Galactic foreground residual")
        path = output_dir / f"earth_{antenna}_foreground_residual.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return json.loads(frame.to_json(orient="records"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--permutations", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
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
    points = pd.read_parquet(args.profiles, columns=columns, filters=[("source_name", "==", "earth")])
    points = points[np.isfinite(pd.to_numeric(points["normalized_power"], errors="coerce"))].copy()
    points = fold_occultogram_points(points)
    points["event_type"] = points["original_event_type"]
    profiles = _block_profiles(points)
    type_profiles = _block_profiles(points, include_event_type=True)
    amplitudes = _fixed_zero_amplitudes(profiles, EDGE_WINDOWS_SECONDS)
    type_amplitudes = _fixed_zero_amplitudes(type_profiles, (PRIMARY_WINDOW_SECONDS,), include_event_type=True)
    channel_summary = _channel_summary(amplitudes)
    empirical_null = _empirical_null(amplitudes, args.permutations, args.seed)
    multiscale = _multiscale_summary(amplitudes, args.permutations, args.seed)
    joint = _joint_block_consistency(type_amplitudes)
    residuals = _foreground_residuals(profiles)
    foreground = _foreground_summary(residuals)

    write_table(amplitudes, args.out_dir / "fixed_zero_block_amplitudes.parquet", index=False)
    write_table(channel_summary, args.out_dir / "fixed_zero_channel_summary.parquet", index=False)
    write_table(empirical_null, args.out_dir / "fixed_zero_empirical_null.parquet", index=False)
    write_table(multiscale, args.out_dir / "fixed_zero_multiscale_summary.parquet", index=False)
    write_table(joint, args.out_dir / "fixed_zero_joint_block_consistency.parquet", index=False)
    write_table(residuals, args.out_dir / "cross_frequency_foreground_residuals.parquet", index=False)
    write_table(foreground, args.out_dir / "cross_frequency_foreground_summary.parquet", index=False)

    plot_paths = [
        args.out_dir / "earth_fixed_zero_robust_edge.png",
        args.out_dir / "earth_fixed_zero_empirical_null.png",
        args.out_dir / "earth_fixed_zero_multiscale.png",
        args.out_dir / "earth_fixed_zero_block_consistency.png",
    ]
    _plot_fixed_zero(channel_summary, plot_paths[0])
    _plot_empirical_null(empirical_null, plot_paths[1])
    _plot_multiscale(multiscale, plot_paths[2])
    _plot_joint_blocks(joint, plot_paths[3])
    plot_paths.extend(_plot_foreground(foreground, args.out_dir))

    visual_data = {
        "config": {
            "source": "earth",
            "edge_time_s": 0.0,
            "edge_windows_s": EDGE_WINDOWS_SECONDS,
            "primary_window_s": PRIMARY_WINDOW_SECONDS,
            "foreground_tracer_frequencies_mhz": TRACER_FREQUENCIES,
            "permutations": args.permutations,
            "independent_unit": "29.53059-day lunar block",
        },
        "fixed_zero": _records(channel_summary),
        "empirical_null": _records(empirical_null),
        "multiscale": _records(multiscale),
        "joint_blocks": _records(joint),
        "foreground_profiles": _records(foreground),
    }
    (args.out_dir / "earth_fixed_zero_visual_data.json").write_text(
        json.dumps(visual_data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "profiles": str(args.profiles),
        "output_directory": str(args.out_dir),
        "earth_profile_rows": int(len(points)),
        "lunar_blocks": int(points["lunar_block"].nunique()),
        "edge_time_fixed_seconds": 0.0,
        "generated_plots": [path.name for path in plot_paths],
        "claim_status": "positive-control demonstrations only; no method selected for other sources",
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
