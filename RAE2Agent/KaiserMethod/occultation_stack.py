#!/usr/bin/env python3
"""
Kaiser (1977) occultation stacking for RAE-2 total-power measurements.

Implements 1-bit stacking of pre/post averages with confusion rejection and
candidate-event listing based on >= 0.5 dB in >= 3 consecutive channels.
"""
import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_DATA_PATH = "/global/cfs/projectdirs/m4895/RAE2Data/interpolatedRAE2MasterFile.csv"

# 32-channel RAE-2 burst receiver center frequencies (MHz), from digitization tools.
DEFAULT_CHANNEL_FREQUENCIES_MHZ = [
    0.025, 0.035, 0.044, 0.055, 0.067, 0.083, 0.096, 0.110,
    0.130, 0.155, 0.185, 0.210, 0.250, 0.292, 0.360, 0.425,
    0.475, 0.600, 0.737, 0.870, 1.030, 1.27, 1.45, 1.85,
    2.20, 2.80, 3.93, 4.70, 6.55, 9.18, 11.8, 13.1,
]

RAE2_BAND_FREQ_MHZ = {
    1: 0.45, 2: 0.70, 3: 0.90, 4: 1.31,
    5: 2.20, 6: 3.93, 7: 4.70, 8: 6.55, 9: 9.18,
}

DEFAULT_POWER_COL = "rv2_coarse"
DEFAULT_TIME_COL = "time"
DEFAULT_CHANNEL_COL = "frequency_band"
DEFAULT_FREQUENCY_COL = "center_frequency_mhz"

CONFUSION_BODIES = [
    "sun",
    "mercury",
    "venus",
    "earth",
    "mars",
    "jupiter",
    "saturn",
    "uranus",
    "neptune",
]


@dataclass
class WindowConfig:
    default_window_min: float
    lowfreq_threshold_mhz: Optional[float] = None
    lowfreq_window_min: Optional[float] = None
    per_frequency: Optional[pd.DataFrame] = None


def load_window_config(path: str) -> pd.DataFrame:
    """Load a CSV with columns: min_freq_mhz,max_freq_mhz,window_min."""
    df = pd.read_csv(path)
    required = {"min_freq_mhz", "max_freq_mhz", "window_min"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Window config missing columns: {sorted(missing)}")
    return df


def get_window_minutes(freq_mhz: float, config: WindowConfig) -> float:
    if config.per_frequency is not None:
        match = config.per_frequency[
            (config.per_frequency["min_freq_mhz"] <= freq_mhz)
            & (config.per_frequency["max_freq_mhz"] >= freq_mhz)
        ]
        if not match.empty:
            return float(match.iloc[0]["window_min"])
    if (
        config.lowfreq_threshold_mhz is not None
        and config.lowfreq_window_min is not None
        and freq_mhz <= config.lowfreq_threshold_mhz
    ):
        return float(config.lowfreq_window_min)
    return float(config.default_window_min)


def normalize_event_type(value: str) -> str:
    val = str(value).strip().upper()
    if val in {"DISAPPEARANCE", "INGRESS"}:
        return "DISAPPEARANCE"
    if val in {"REAPPEARANCE", "EGRESS"}:
        return "REAPPEARANCE"
    raise ValueError(f"Unsupported event_type: {value}")


def sign_bit_from_delta(delta_db: float) -> int:
    """Map delta to +/-1 (zero treated as -1)."""
    return 1 if delta_db > 0 else -1


def compute_delta_db(mean_before: float, mean_after: float, event_type: str) -> float:
    event_type = normalize_event_type(event_type)
    if event_type == "DISAPPEARANCE":
        unocculted = mean_before
        occulted = mean_after
    else:
        unocculted = mean_after
        occulted = mean_before
    return float(unocculted - occulted)


def compute_statistic(values: np.ndarray, statistic: str, trim_fraction: float) -> float:
    if len(values) == 0:
        return float("nan")
    if statistic == "mean":
        return float(np.mean(values))
    if statistic == "median":
        return float(np.median(values))
    if statistic == "trimmed":
        if trim_fraction <= 0:
            return float(np.mean(values))
        values_sorted = np.sort(values)
        trim_n = int(math.floor(len(values_sorted) * trim_fraction))
        if trim_n * 2 >= len(values_sorted):
            return float(np.mean(values_sorted))
        trimmed = values_sorted[trim_n:-trim_n]
        return float(np.mean(trimmed))
    raise ValueError(f"Unsupported statistic: {statistic}")


def to_db(power: np.ndarray, unit: str) -> np.ndarray:
    if unit == "db":
        return power.astype(float)
    if unit == "linear":
        power = np.array(power, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            db = 10.0 * np.log10(power)
        db[~np.isfinite(db)] = np.nan
        return db
    raise ValueError(f"Unsupported power_unit: {unit}")


def load_channel_frequency_map(path: str) -> Dict[int, float]:
    df = pd.read_csv(path)
    required = {"channel_id", "frequency_mhz"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Channel frequency map missing columns: {sorted(missing)}")
    return {int(row.channel_id): float(row.frequency_mhz) for row in df.itertuples(index=False)}


def infer_default_channel_frequencies(channel_ids: Iterable[int]) -> Dict[int, float]:
    channel_ids = sorted({int(cid) for cid in channel_ids})
    if channel_ids and max(channel_ids) <= 31 and min(channel_ids) >= 0:
        return {idx: DEFAULT_CHANNEL_FREQUENCIES_MHZ[idx] for idx in channel_ids}
    if channel_ids and max(channel_ids) <= 9 and min(channel_ids) >= 1:
        return {idx: RAE2_BAND_FREQ_MHZ.get(idx, float("nan")) for idx in channel_ids}
    return {}


def load_data(
    path: str,
    time_col: str = DEFAULT_TIME_COL,
    channel_col: str = DEFAULT_CHANNEL_COL,
    frequency_col: str = DEFAULT_FREQUENCY_COL,
    power_col: str = DEFAULT_POWER_COL,
    power_unit: str = "db",
    channel_freqs: Optional[Dict[int, float]] = None,
) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    available = set(header.columns)

    usecols = {time_col, channel_col, power_col}
    if frequency_col and frequency_col in available:
        usecols.add(frequency_col)
    df = pd.read_csv(path, usecols=[c for c in usecols if c])

    if time_col not in df.columns:
        raise ValueError(f"Missing time column: {time_col}")
    if channel_col not in df.columns:
        raise ValueError(f"Missing channel column: {channel_col}")
    if power_col not in df.columns:
        raise ValueError(f"Missing power column: {power_col}")

    df = df.rename(
        columns={
            time_col: "time",
            channel_col: "channel_id",
            power_col: "power",
            frequency_col: "frequency_mhz" if frequency_col in df.columns else frequency_col,
        }
    )

    df["time"] = pd.to_datetime(df["time"])
    df["channel_id"] = df["channel_id"].astype(int)

    if "frequency_mhz" not in df.columns:
        if channel_freqs is None:
            channel_freqs = infer_default_channel_frequencies(df["channel_id"].unique())
        if not channel_freqs:
            raise ValueError("No frequency column and no channel frequency map provided.")
        df["frequency_mhz"] = df["channel_id"].map(channel_freqs)

    df["power_db"] = to_db(df["power"].to_numpy(dtype=float), power_unit)
    df = df.drop(columns=["power"])
    df = df.sort_values("time")
    return df


def normalize_separation_columns(events_df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for body in CONFUSION_BODIES:
        for col in [f"{body}_sep_deg", f"{body}_limb_sep_deg", f"{body}_limb_distance_deg"]:
            if col in events_df.columns:
                rename[col] = f"{body}_sep_deg"
    if rename:
        events_df = events_df.rename(columns=rename)
    return events_df


def load_events(
    path: str,
    time_col: str = "event_time",
    event_type_col: str = "event_type",
    planet_col: str = "planet",
    event_id_col: str = "event_id",
) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {time_col, event_type_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Events file missing columns: {sorted(missing)}")

    df = df.rename(
        columns={
            time_col: "event_time",
            event_type_col: "event_type",
            planet_col: "planet" if planet_col in df.columns else planet_col,
            event_id_col: "event_id" if event_id_col in df.columns else event_id_col,
        }
    )

    df["event_time"] = pd.to_datetime(df["event_time"])
    df["event_type"] = df["event_type"].apply(normalize_event_type)
    if "event_id" not in df.columns:
        df["event_id"] = np.arange(len(df), dtype=int)
    if "planet" not in df.columns:
        df["planet"] = "UNKNOWN"
    df = normalize_separation_columns(df)
    return df


def compute_spice_separations(
    events_df: pd.DataFrame,
    kernels: Sequence[str],
    observer: str,
    frame: str = "J2000",
    aberration: str = "LT+S",
) -> pd.DataFrame:
    try:
        import spiceypy as spice
    except ImportError as exc:
        raise ImportError("spiceypy is required for SPICE separations.") from exc

    spice.kclear()
    for kernel in kernels:
        spice.furnsh(kernel)

    try:
        rows = []
        for _, event in events_df.iterrows():
            et = spice.utc2et(event["event_time"].isoformat())
            target = str(event.get("planet", "")).strip().lower()
            if not target:
                target = "unknown"

            try:
                target_vec, _ = spice.spkpos(target, et, frame, aberration, observer)
            except Exception:
                target_vec = None

            row = {}
            for body in CONFUSION_BODIES:
                if body == target:
                    row[f"{body}_sep_deg"] = 0.0
                    continue
                try:
                    body_vec, _ = spice.spkpos(body, et, frame, aberration, observer)
                except Exception:
                    row[f"{body}_sep_deg"] = float("nan")
                    continue
                if target_vec is None:
                    row[f"{body}_sep_deg"] = float("nan")
                    continue
                angle_rad = spice.vsep(target_vec, body_vec)
                row[f"{body}_sep_deg"] = float(np.degrees(angle_rad))
            rows.append(row)
        sep_df = pd.DataFrame(rows)
    finally:
        spice.kclear()

    merged = events_df.reset_index(drop=True).copy()
    for col in sep_df.columns:
        if col not in merged.columns:
            merged[col] = sep_df[col].values
        else:
            merged[col] = merged[col].fillna(sep_df[col].values)
    return merged


def filter_events_by_confusion(
    events_df: pd.DataFrame,
    require_separations: bool = True,
) -> pd.DataFrame:
    if events_df.empty:
        return events_df.copy()

    missing_cols = [f"{body}_sep_deg" for body in CONFUSION_BODIES if f"{body}_sep_deg" not in events_df.columns]
    if missing_cols and require_separations:
        raise ValueError(
            "Missing separation columns for confusion filter: " + ", ".join(missing_cols)
        )

    keep_mask = []
    for _, event in events_df.iterrows():
        target = str(event.get("planet", "")).strip().lower()
        keep = True
        for body in CONFUSION_BODIES:
            if body == target:
                continue
            col = f"{body}_sep_deg"
            if col not in events_df.columns:
                continue
            sep = event[col]
            if not np.isfinite(sep):
                keep = False
                break
            threshold = 5.0 if body == "earth" else 3.0
            if sep < threshold:
                keep = False
                break
        keep_mask.append(keep)
    return events_df.loc[keep_mask].copy()


def extract_channel_window(
    channel_df: pd.DataFrame,
    t0: pd.Timestamp,
    t1: pd.Timestamp,
    include_end: bool = True,
) -> np.ndarray:
    if include_end:
        mask = (channel_df["time"] >= t0) & (channel_df["time"] <= t1)
    else:
        mask = (channel_df["time"] >= t0) & (channel_df["time"] < t1)
    return channel_df.loc[mask, "power_db"].dropna().to_numpy()


def compute_event_channel_deltas(
    measurements_df: pd.DataFrame,
    events_df: pd.DataFrame,
    window_config: WindowConfig,
    min_samples_per_side: int = 2,
    statistic: str = "mean",
    trim_fraction: float = 0.1,
) -> pd.DataFrame:
    if measurements_df.empty or events_df.empty:
        return pd.DataFrame()

    channel_groups = {
        int(ch): ch_df.sort_values("time")
        for ch, ch_df in measurements_df.groupby("channel_id")
    }

    rows = []
    for _, event in events_df.iterrows():
        event_time = event["event_time"]
        event_type = event["event_type"]
        for channel_id, ch_df in channel_groups.items():
            freq = ch_df["frequency_mhz"].iloc[0]
            window_min = get_window_minutes(freq, window_config)
            window_delta = pd.Timedelta(minutes=window_min)

            before = extract_channel_window(ch_df, event_time - window_delta, event_time, include_end=False)
            after = extract_channel_window(ch_df, event_time, event_time + window_delta, include_end=True)

            if len(before) < min_samples_per_side or len(after) < min_samples_per_side:
                continue

            mean_before = compute_statistic(before, statistic, trim_fraction)
            mean_after = compute_statistic(after, statistic, trim_fraction)
            delta_db = compute_delta_db(mean_before, mean_after, event_type)
            sign_bit = sign_bit_from_delta(delta_db)

            rows.append(
                {
                    "event_id": event["event_id"],
                    "event_time": event_time,
                    "event_type": event_type,
                    "planet": event.get("planet", "UNKNOWN"),
                    "channel_id": int(channel_id),
                    "frequency_mhz": float(freq),
                    "window_min": window_min,
                    "mean_before": mean_before,
                    "mean_after": mean_after,
                    "delta_db": delta_db,
                    "sign_bit": int(sign_bit),
                    "n_before": int(len(before)),
                    "n_after": int(len(after)),
                }
            )

    return pd.DataFrame(rows)


def compute_S_statistics(delta_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if delta_df.empty:
        return pd.DataFrame(rows)

    for channel_id, group in delta_df.groupby("channel_id"):
        n = len(group)
        sum_bits = int(group["sign_bit"].sum())
        s_val = sum_bits / math.sqrt(n) if n > 0 else float("nan")
        freq = float(group["frequency_mhz"].iloc[0])
        rows.append(
            {
                "channel_id": int(channel_id),
                "frequency_mhz": freq,
                "n_events": n,
                "sum_bits": sum_bits,
                "S": s_val,
            }
        )
    return pd.DataFrame(rows)


def find_candidate_runs(
    channel_ids: np.ndarray,
    deltas: np.ndarray,
    sign_bits: np.ndarray,
    min_run: int,
    delta_threshold_db: float,
) -> List[Dict[str, object]]:
    order = np.argsort(channel_ids)
    channel_ids = channel_ids[order]
    deltas = deltas[order]
    sign_bits = sign_bits[order]

    runs = []
    run_channels: List[int] = []
    run_deltas: List[float] = []
    last_channel: Optional[int] = None

    def flush_run() -> None:
        if len(run_channels) >= min_run:
            runs.append(
                {
                    "channel_start": int(run_channels[0]),
                    "channel_end": int(run_channels[-1]),
                    "run_length": int(len(run_channels)),
                    "channels": ",".join(str(ch) for ch in run_channels),
                    "min_delta_db": float(np.min(run_deltas)),
                    "max_delta_db": float(np.max(run_deltas)),
                    "mean_delta_db": float(np.mean(run_deltas)),
                }
            )

    for ch, delta, sign in zip(channel_ids, deltas, sign_bits):
        ok = (sign == 1) and (delta >= delta_threshold_db)
        if ok:
            if run_channels and last_channel is not None and ch == last_channel + 1:
                run_channels.append(int(ch))
                run_deltas.append(float(delta))
            else:
                if run_channels:
                    flush_run()
                run_channels = [int(ch)]
                run_deltas = [float(delta)]
            last_channel = int(ch)
        else:
            if run_channels:
                flush_run()
            run_channels = []
            run_deltas = []
            last_channel = None

    if run_channels:
        flush_run()

    return runs


def find_candidate_events(
    delta_df: pd.DataFrame,
    min_run: int = 3,
    delta_threshold_db: float = 0.5,
) -> pd.DataFrame:
    if delta_df.empty:
        return pd.DataFrame()

    rows = []
    for (event_id, event_time), group in delta_df.groupby(["event_id", "event_time"]):
        runs = find_candidate_runs(
            group["channel_id"].to_numpy(),
            group["delta_db"].to_numpy(),
            group["sign_bit"].to_numpy(),
            min_run,
            delta_threshold_db,
        )
        for run in runs:
            rows.append(
                {
                    "event_id": event_id,
                    "event_time": event_time,
                    "event_type": group["event_type"].iloc[0],
                    "planet": group["planet"].iloc[0],
                    **run,
                }
            )
    return pd.DataFrame(rows)


def plot_candidate_event(
    measurements_df: pd.DataFrame,
    event_row: pd.Series,
    channel_ids: Sequence[int],
    outpath: str,
    window_min: float,
) -> None:
    event_time = event_row["event_time"]
    t0 = event_time - pd.Timedelta(minutes=2 * window_min)
    t1 = event_time + pd.Timedelta(minutes=2 * window_min)

    fig, axes = plt.subplots(len(channel_ids), 1, figsize=(9, 2.4 * len(channel_ids)), sharex=True)
    if len(channel_ids) == 1:
        axes = [axes]

    for ax, channel_id in zip(axes, channel_ids):
        ch_df = measurements_df[
            (measurements_df["channel_id"] == channel_id)
            & (measurements_df["time"] >= t0)
            & (measurements_df["time"] <= t1)
        ]
        freq = ch_df["frequency_mhz"].iloc[0] if not ch_df.empty else float("nan")
        ax.plot(ch_df["time"], ch_df["power_db"], linewidth=1.0)
        ax.axvline(event_time, color="red", linestyle="--", linewidth=1.0)
        ax.set_ylabel(f"Ch {channel_id} ({freq:.3f} MHz)")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time")
    title = f"Event {event_row['event_id']} {event_row['event_type']} ({event_row['planet']})"
    fig.suptitle(title)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_dynamic_spectrum(
    measurements_df: pd.DataFrame,
    event_row: pd.Series,
    outpath: str,
    window_min: float,
) -> None:
    event_time = event_row["event_time"]
    t0 = event_time - pd.Timedelta(minutes=2 * window_min)
    t1 = event_time + pd.Timedelta(minutes=2 * window_min)

    slice_df = measurements_df[
        (measurements_df["time"] >= t0) & (measurements_df["time"] <= t1)
    ].copy()
    if slice_df.empty:
        return

    pivot = slice_df.pivot_table(
        index="time",
        columns="frequency_mhz",
        values="power_db",
        aggfunc="mean",
    ).sort_index()

    times = pivot.index.to_pydatetime()
    freqs = pivot.columns.to_numpy(dtype=float)
    power = pivot.to_numpy().T

    fig, ax = plt.subplots(figsize=(9, 4))
    mesh = ax.pcolormesh(times, freqs, power, shading="auto")
    ax.axvline(event_time, color="white", linestyle="--", linewidth=1.0)
    ax.set_ylabel("Frequency (MHz)")
    ax.set_xlabel("Time")
    ax.set_title(f"Dynamic Spectrum Event {event_row['event_id']}")
    fig.colorbar(mesh, ax=ax, label="Power (dB)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def sanitize_filename(name: str) -> str:
    keep = []
    for ch in str(name):
        if ch.isalnum() or ch in {"-", "_"}:
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kaiser (1977) occultation stacking analysis")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="Path to measurements CSV")
    parser.add_argument("--events", required=True, help="Path to predicted occultation events CSV")
    parser.add_argument("--outdir", default="RAE2Agent/KaiserMethod/output", help="Output directory")
    parser.add_argument("--planet", default=None, help="Filter events by target planet name")

    parser.add_argument("--time-col", default=DEFAULT_TIME_COL, help="Time column name")
    parser.add_argument("--channel-col", default=DEFAULT_CHANNEL_COL, help="Channel column name")
    parser.add_argument("--frequency-col", default=DEFAULT_FREQUENCY_COL, help="Frequency column name")
    parser.add_argument("--power-col", default=DEFAULT_POWER_COL, help="Power column name")
    parser.add_argument("--power-unit", default="db", choices=["db", "linear"], help="Power units")
    parser.add_argument("--channel-freqs", default=None, help="CSV mapping channel_id,frequency_mhz")

    parser.add_argument("--window-min", type=float, default=4.0, help="Default window size in minutes")
    parser.add_argument("--lowfreq-threshold-mhz", type=float, default=None,
                        help="Low-frequency threshold for larger window")
    parser.add_argument("--lowfreq-window-min", type=float, default=None,
                        help="Window size below low-frequency threshold")
    parser.add_argument("--window-config-csv", default=None,
                        help="CSV mapping min_freq_mhz,max_freq_mhz,window_min")
    parser.add_argument("--min-samples", type=int, default=2, help="Minimum samples per window side")
    parser.add_argument("--statistic", default="mean", choices=["mean", "median", "trimmed"],
                        help="Statistic for window averaging")
    parser.add_argument("--trim-fraction", type=float, default=0.1, help="Trim fraction for trimmed mean")

    parser.add_argument("--candidate-min-run", type=int, default=3, help="Minimum consecutive channels")
    parser.add_argument("--candidate-delta-db", type=float, default=0.5, help="Delta threshold (dB)")
    parser.add_argument("--s-threshold", type=float, default=2.0, help="S threshold flag for summary")
    parser.add_argument("--max-plot-events", type=int, default=20, help="Max candidate plots to generate")
    parser.add_argument("--plot-dynamic-spectrum", action="store_true", help="Plot dynamic spectra")

    parser.add_argument("--event-start", default=None, help="Keep events on/after this time")
    parser.add_argument("--event-end", default=None, help="Keep events on/before this time")
    parser.add_argument("--antenna-extension-date", default=None,
                        help="Alias for --event-start to filter pre-extension events")

    parser.add_argument("--spice-kernel", action="append", default=[], help="SPICE kernel to load")
    parser.add_argument("--spice-meta-kernel", default=None, help="SPICE meta-kernel to load")
    parser.add_argument("--spice-observer", default=None, help="SPICE observer name/ID")
    parser.add_argument("--spice-frame", default="J2000", help="SPICE frame")
    parser.add_argument("--spice-abcorr", default="LT+S", help="SPICE aberration correction")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    channel_freqs = load_channel_frequency_map(args.channel_freqs) if args.channel_freqs else None
    data_df = load_data(
        args.data,
        time_col=args.time_col,
        channel_col=args.channel_col,
        frequency_col=args.frequency_col,
        power_col=args.power_col,
        power_unit=args.power_unit,
        channel_freqs=channel_freqs,
    )

    events_df = load_events(args.events)

    if args.planet:
        planet = args.planet.strip().lower()
        events_df = events_df[events_df["planet"].str.lower() == planet]

    start = args.event_start or args.antenna_extension_date
    if start:
        events_df = events_df[events_df["event_time"] >= pd.to_datetime(start)]
    if args.event_end:
        events_df = events_df[events_df["event_time"] <= pd.to_datetime(args.event_end)]

    kernels = list(args.spice_kernel)
    if args.spice_meta_kernel:
        kernels.append(args.spice_meta_kernel)
    if kernels:
        if not args.spice_observer:
            raise ValueError("--spice-observer is required when using SPICE kernels")
        events_df = compute_spice_separations(
            events_df,
            kernels=kernels,
            observer=args.spice_observer,
            frame=args.spice_frame,
            aberration=args.spice_abcorr,
        )

    events_df = filter_events_by_confusion(events_df, require_separations=True)

    window_cfg = WindowConfig(
        default_window_min=args.window_min,
        lowfreq_threshold_mhz=args.lowfreq_threshold_mhz,
        lowfreq_window_min=args.lowfreq_window_min,
        per_frequency=load_window_config(args.window_config_csv) if args.window_config_csv else None,
    )

    delta_df = compute_event_channel_deltas(
        data_df,
        events_df,
        window_config=window_cfg,
        min_samples_per_side=args.min_samples,
        statistic=args.statistic,
        trim_fraction=args.trim_fraction,
    )

    s_stats = compute_S_statistics(delta_df)
    if not s_stats.empty:
        s_stats["S_flag"] = s_stats["S"] >= args.s_threshold

    candidates_df = find_candidate_events(
        delta_df,
        min_run=args.candidate_min_run,
        delta_threshold_db=args.candidate_delta_db,
    )

    os.makedirs(args.outdir, exist_ok=True)
    delta_path = os.path.join(args.outdir, "event_channel_deltas.csv")
    stack_path = os.path.join(args.outdir, "stack_statistics.csv")
    cand_path = os.path.join(args.outdir, "candidate_events.csv")

    delta_df.to_csv(delta_path, index=False)
    s_stats.to_csv(stack_path, index=False)
    candidates_df.to_csv(cand_path, index=False)

    if not candidates_df.empty and args.max_plot_events > 0:
        plots_dir = os.path.join(args.outdir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        ranked = candidates_df.sort_values("max_delta_db", ascending=False)
        to_plot = ranked.head(args.max_plot_events)
        for _, row in to_plot.iterrows():
            channels = [int(c) for c in str(row["channels"]).split(",") if c]
            filename = sanitize_filename(
                f"event_{row['event_id']}_ch{row['channel_start']}-{row['channel_end']}"
            )
            plot_path = os.path.join(plots_dir, f"{filename}.png")
            plot_candidate_event(
                data_df,
                row,
                channels,
                plot_path,
                window_min=args.window_min,
            )
            if args.plot_dynamic_spectrum:
                dyn_path = os.path.join(plots_dir, f"{filename}_dynamic.png")
                plot_dynamic_spectrum(
                    data_df,
                    row,
                    dyn_path,
                    window_min=args.window_min,
                )


if __name__ == "__main__":
    main()
