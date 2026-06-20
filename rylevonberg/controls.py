"""Controls and injection-recovery utilities."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .constants import add_frequency_mhz_column
from .detection import StepFitConfig, event_template, run_stepfit_detections
from .util import datetime_ns


def randomized_event_times(
    events_df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    seed: int = 12345,
    exclusion_seconds: float = 0.0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = events_df.copy()
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    span_ns = end_ts.value - start_ts.value
    real_times = pd.to_datetime(events_df["predicted_event_time"]) if "predicted_event_time" in events_df.columns else pd.Series([], dtype="datetime64[ns]")
    generated = []
    exclusion = pd.Timedelta(seconds=float(exclusion_seconds))
    for _ in range(len(out)):
        candidate = start_ts
        for _attempt in range(1000):
            candidate = pd.Timestamp(start_ts.value + int(rng.integers(0, max(span_ns, 1))))
            if exclusion <= pd.Timedelta(0):
                break
            if real_times.empty or (pd.Series(real_times).sub(candidate).abs() > exclusion).all():
                break
        generated.append(candidate)
    out["predicted_event_time"] = pd.to_datetime(generated)
    flags = out["quality_flags"].astype(str) if "quality_flags" in out.columns else pd.Series([""] * len(out), index=out.index)
    out["quality_flags"] = flags + ";negative_control_randomized_time"
    return out


def time_reversed_events(events_df: pd.DataFrame) -> pd.DataFrame:
    out = events_df.copy()
    mapping = {"disappearance": "reappearance", "reappearance": "disappearance"}
    out["event_type"] = out["event_type"].map(mapping).fillna(out["event_type"])
    flags = out["quality_flags"].astype(str) if "quality_flags" in out.columns else pd.Series([""] * len(out), index=out.index)
    out["quality_flags"] = flags + ";negative_control_time_reversed_template"
    return out


def wrong_frequency_events(events_df: pd.DataFrame, available_frequencies: list[int] | np.ndarray) -> pd.DataFrame:
    """Move each event to a different available frequency when possible."""
    choices = [int(v) for v in available_frequencies]
    out = events_df.copy()
    if not choices or "frequency_band" not in out.columns:
        return out
    shifted = []
    for value in out["frequency_band"]:
        current = int(value) if pd.notna(value) else None
        alternatives = [v for v in choices if v != current]
        shifted.append(alternatives[0] if alternatives else current)
    out["frequency_band"] = shifted
    flags = out["quality_flags"].astype(str) if "quality_flags" in out.columns else pd.Series([""] * len(out), index=out.index)
    out["quality_flags"] = flags + ";negative_control_wrong_frequency"
    return add_frequency_mhz_column(out)


def wrong_antenna_events(events_df: pd.DataFrame, available_antennas: list[str] | np.ndarray) -> pd.DataFrame:
    """Move each event to a different antenna/receiver column when possible."""
    choices = [str(v) for v in available_antennas]
    out = events_df.copy()
    if not choices or "antenna" not in out.columns:
        return out
    shifted = []
    for value in out["antenna"]:
        current = str(value) if pd.notna(value) else None
        alternatives = [v for v in choices if v != current]
        shifted.append(alternatives[0] if alternatives else current)
    out["antenna"] = shifted
    flags = out["quality_flags"].astype(str) if "quality_flags" in out.columns else pd.Series([""] * len(out), index=out.index)
    out["quality_flags"] = flags + ";negative_control_wrong_antenna"
    return out


def negative_control_event_ensemble(
    events_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    n_random: int = 100,
    seed: int = 12345,
    exclusion_seconds: float = 900.0,
    include_reversed: bool = True,
    include_wrong_frequency: bool = True,
    include_wrong_antenna: bool = True,
) -> pd.DataFrame:
    """Build a deterministic event table for null-control detection runs."""
    tables: list[pd.DataFrame] = []
    start = pd.Timestamp(clean_df["time"].min())
    end = pd.Timestamp(clean_df["time"].max())
    for idx in range(int(n_random)):
        rand = randomized_event_times(events_df, start, end, seed=int(seed) + idx, exclusion_seconds=exclusion_seconds)
        rand["control_type"] = "randomized_time"
        rand["control_id"] = idx
        tables.append(rand)
    next_id = int(n_random)
    if include_reversed:
        rev = time_reversed_events(events_df)
        rev["control_type"] = "time_reversed_template"
        rev["control_id"] = next_id
        tables.append(rev)
        next_id += 1
    if include_wrong_frequency and "frequency_band" in clean_df.columns:
        wf = wrong_frequency_events(events_df, sorted(clean_df["frequency_band"].dropna().unique()))
        wf["control_type"] = "wrong_frequency"
        wf["control_id"] = next_id
        tables.append(wf)
        next_id += 1
    if include_wrong_antenna and "antenna" in clean_df.columns:
        wa = wrong_antenna_events(events_df, sorted(clean_df["antenna"].dropna().unique()))
        wa["control_type"] = "wrong_antenna"
        wa["control_id"] = next_id
        tables.append(wa)
    return add_frequency_mhz_column(pd.concat(tables, ignore_index=True)) if tables else pd.DataFrame()


def subsample_control_events(
    controls_df: pd.DataFrame,
    max_per_group: int | None,
    seed: int = 12345,
) -> pd.DataFrame:
    """Deterministically cap large control ensembles per source/channel/control type."""
    if controls_df.empty or not max_per_group or int(max_per_group) <= 0:
        return controls_df
    group_cols = [c for c in ["control_type", "source_name", "event_type", "frequency_band", "antenna"] if c in controls_df.columns]
    rows = []
    for key, group in controls_df.groupby(group_cols, dropna=False, sort=True):
        if len(group) <= int(max_per_group):
            rows.append(group)
            continue
        stable_seed = int(hashlib.sha1(str(key).encode("utf-8")).hexdigest()[:8], 16) % 1_000_000
        rows.append(group.sample(n=int(max_per_group), random_state=int(seed) + stable_seed).sort_index())
    return add_frequency_mhz_column(pd.concat(rows, ignore_index=True)) if rows else controls_df.head(0)


def inject_synthetic_steps(
    clean_df: pd.DataFrame,
    events_df: pd.DataFrame,
    amplitude: float,
    width_seconds: float = 0.0,
) -> pd.DataFrame:
    out = clean_df.copy()
    out["power_original"] = out["power"]
    groups = {
        (freq, antenna): idx.to_numpy()
        for (freq, antenna), idx in out.groupby(["frequency_band", "antenna"], dropna=False, sort=True).groups.items()
    }
    for _, ev in events_df.iterrows():
        freq = ev.get("frequency_band")
        antenna = ev.get("antenna")
        idx = groups.get((freq, antenna))
        if idx is None or len(idx) == 0:
            continue
        t_rel = (datetime_ns(out.loc[idx, "time"]) - pd.Timestamp(ev["predicted_event_time"]).value).astype(float) / 1e9
        tmpl = event_template(t_rel, str(ev["event_type"]), smooth_seconds=width_seconds)
        out.loc[idx, "power"] = out.loc[idx, "power"].to_numpy(dtype=float) + float(amplitude) * tmpl
    return out


def injection_recovery(
    clean_df: pd.DataFrame,
    events_df: pd.DataFrame,
    amplitudes: list[float],
    config: StepFitConfig,
) -> pd.DataFrame:
    rows = []
    if not events_df.empty and {"frequency_band", "antenna"}.issubset(events_df.columns) and {"frequency_band", "antenna"}.issubset(clean_df.columns):
        keep = np.zeros(len(clean_df), dtype=bool)
        for freq, antenna in events_df[["frequency_band", "antenna"]].drop_duplicates().itertuples(index=False, name=None):
            keep |= (clean_df["frequency_band"].to_numpy() == freq) & (clean_df["antenna"].astype(str).to_numpy() == str(antenna))
        clean_work = clean_df.loc[keep].reset_index(drop=True)
    else:
        clean_work = clean_df
    for amp in amplitudes:
        injected = inject_synthetic_steps(clean_work, events_df, amplitude=float(amp), width_seconds=config.smooth_seconds)
        det = run_stepfit_detections(injected, events_df, config)
        if det.empty:
            rows.append({"injected_amplitude": float(amp), "n_recovered": 0, "median_snr": np.nan, "median_amplitude": np.nan})
            continue
        finite_snr = det["detection_snr"].to_numpy(dtype=float)
        finite_amp = det["amplitude"].to_numpy(dtype=float)
        finite_snr = finite_snr[np.isfinite(finite_snr)]
        finite_amp = finite_amp[np.isfinite(finite_amp)]
        rows.append(
            {
                "injected_amplitude": float(amp),
                "n_recovered": int(finite_snr.size),
                "median_snr": float(np.median(finite_snr)) if finite_snr.size else np.nan,
                "median_amplitude": float(np.median(finite_amp)) if finite_amp.size else np.nan,
            }
        )
    return add_frequency_mhz_column(pd.DataFrame.from_records(rows))


def injection_recovery_grid(
    clean_df: pd.DataFrame,
    events_df: pd.DataFrame,
    amplitudes: list[float],
    window_seconds: list[float],
    smooth_seconds: float = 0.0,
    min_samples_per_side: int = 4,
) -> pd.DataFrame:
    """Run injection recovery over amplitude/window/frequency/antenna cells."""
    rows: list[dict] = []
    if events_df.empty:
        return pd.DataFrame()
    group_cols = [c for c in ["frequency_band", "antenna"] if c in events_df.columns]
    grouped = events_df.groupby(group_cols, dropna=False, sort=True) if group_cols else [((), events_df)]
    for keys, event_group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        meta = dict(zip(group_cols, keys))
        for window in window_seconds:
            config = StepFitConfig(
                window_seconds=float(window),
                smooth_seconds=float(smooth_seconds),
                min_samples_per_side=int(min_samples_per_side),
            )
            clean_group = clean_df
            if {"frequency_band", "antenna"}.issubset(clean_df.columns) and {"frequency_band", "antenna"}.issubset(event_group.columns):
                freq = event_group["frequency_band"].iloc[0]
                antenna = event_group["antenna"].iloc[0]
                clean_group = clean_df[(clean_df["frequency_band"] == freq) & (clean_df["antenna"] == antenna)].reset_index(drop=True)
            recovered = injection_recovery(clean_group, event_group.reset_index(drop=True), amplitudes, config)
            for _, row in recovered.iterrows():
                rows.append({**meta, "window_seconds": float(window), **row.to_dict()})
    return pd.DataFrame.from_records(rows)
