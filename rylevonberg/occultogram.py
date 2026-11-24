"""Morphology-first lunar-occultation profiles and event-level contrasts.

The functions in this module intentionally do not calculate a sample-count
SNR.  Each heatmap row is one predicted event, and summaries therefore treat
events (or lunar-month blocks) as the independent units.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .constants import add_frequency_mhz_column
from .util import datetime_ns, robust_sigma


LUNAR_MONTH_DAYS = 29.53059
LUNAR_BLOCK_EPOCH = pd.Timestamp("1973-01-01")
COMPACT_PROFILE_COLUMNS = [
    "source_name",
    "event_uid",
    "event_order",
    "lunar_block",
    "event_type",
    "frequency_band",
    "frequency_mhz",
    "antenna",
    "control_shift_s",
    "t_bin_sec",
    "raw_power",
    "normalized_power",
    "n_samples",
]
PROFILE_VALUE_COLUMNS = {"t_bin_sec", "raw_power", "normalized_power", "n_samples"}


@dataclass(frozen=True)
class OccultogramConfig:
    """Extraction settings for raw, normalized, and control occultograms."""

    window_seconds: float = 2400.0
    bin_seconds: float = 60.0
    sideband_start_seconds: float = 1200.0
    control_shifts_seconds: tuple[float, ...] = (-1200.0, -600.0, 0.0, 600.0, 1200.0)
    minimum_window_samples: int = 8
    minimum_sideband_samples: int = 6
    validity_mode: str = "finite"

    def __post_init__(self) -> None:
        if self.window_seconds <= 0 or self.bin_seconds <= 0:
            raise ValueError("window_seconds and bin_seconds must be positive")
        if not 0 < self.sideband_start_seconds < self.window_seconds:
            raise ValueError("sideband_start_seconds must lie inside the extraction window")
        if not any(np.isclose(value, 0.0) for value in self.control_shifts_seconds):
            raise ValueError("control_shifts_seconds must include the predicted alignment at 0")
        if self.validity_mode not in {"finite", "pipeline"}:
            raise ValueError("validity_mode must be 'finite' or 'pipeline'")


def _event_uid(event: pd.Series) -> str:
    """Return a channel-specific event identifier that is globally stable."""

    stamp = pd.Timestamp(event["predicted_event_time"]).isoformat()
    return "|".join(
        [
            str(event.get("source_name", "")),
            stamp,
            str(event.get("event_type", "")),
            str(event.get("frequency_band", "")),
            str(event.get("antenna", "")),
            str(event.get("event_id", "")),
        ]
    )


def lunar_block_id(event_time: pd.Timestamp) -> int:
    """Assign an event to a fixed 29.53059-day block for block summaries."""

    elapsed_days = (pd.Timestamp(event_time) - LUNAR_BLOCK_EPOCH).total_seconds() / 86400.0
    return int(np.floor(elapsed_days / LUNAR_MONTH_DAYS))


def _clean_groups(clean_df: pd.DataFrame) -> dict[tuple[object, object], tuple[pd.DataFrame, np.ndarray]]:
    groups: dict[tuple[object, object], tuple[pd.DataFrame, np.ndarray]] = {}
    for key, group in clean_df.groupby(["frequency_band", "antenna"], dropna=False, sort=False):
        ordered = group.sort_values("time").reset_index(drop=True)
        groups[key] = (ordered, datetime_ns(ordered["time"]))
    return groups


def _window_values(
    group: pd.DataFrame,
    group_ns: np.ndarray,
    center_time: pd.Timestamp,
    config: OccultogramConfig,
) -> tuple[np.ndarray, np.ndarray] | None:
    center_ns = pd.Timestamp(center_time).value
    half_ns = int(config.window_seconds * 1e9)
    lo = int(np.searchsorted(group_ns, center_ns - half_ns, side="left"))
    hi = int(np.searchsorted(group_ns, center_ns + half_ns, side="right"))
    if hi <= lo:
        return None
    local = group.iloc[lo:hi]
    relative_seconds = (datetime_ns(local["time"]) - center_ns).astype(float) / 1e9
    power = pd.to_numeric(local["power"], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(power) & (np.abs(relative_seconds) <= config.window_seconds)
    if config.validity_mode == "pipeline" and "is_valid" in local.columns:
        keep &= local["is_valid"].fillna(False).to_numpy(dtype=bool)
    if np.count_nonzero(keep) < config.minimum_window_samples:
        return None
    order = np.argsort(relative_seconds[keep])
    return relative_seconds[keep][order], power[keep][order]


def _normalize_without_detrending(
    relative_seconds: np.ndarray,
    power: np.ndarray,
    config: OccultogramConfig,
) -> tuple[np.ndarray, float, float] | None:
    sideband = np.abs(relative_seconds) >= config.sideband_start_seconds
    if np.count_nonzero(sideband) < config.minimum_sideband_samples:
        return None
    center = float(np.nanmedian(power[sideband]))
    scale = robust_sigma(power[sideband] - center)
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.nanstd(power[sideband], ddof=1))
    if not np.isfinite(scale) or scale <= 0:
        return None
    return (power - center) / scale, center, scale


def extract_occultogram_points(
    clean_df: pd.DataFrame,
    events_df: pd.DataFrame,
    config: OccultogramConfig | None = None,
) -> pd.DataFrame:
    """Extract one binned raw and normalized profile per event and time control.

    ``validity_mode='finite'`` is the science default.  It retains finite
    samples even when the legacy ingest flags them as amplitude or jump
    outliers, preventing a real burst or sharp edge from being censored.
    ``validity_mode='pipeline'`` is available as a sensitivity check.
    """

    config = config or OccultogramConfig()
    if clean_df.empty or events_df.empty:
        return pd.DataFrame()
    required_clean = {"time", "frequency_band", "antenna", "power"}
    required_events = {"predicted_event_time", "event_type", "frequency_band", "antenna"}
    missing_clean = sorted(required_clean - set(clean_df.columns))
    missing_events = sorted(required_events - set(events_df.columns))
    if missing_clean or missing_events:
        raise ValueError(f"missing clean columns={missing_clean}; missing event columns={missing_events}")

    groups = _clean_groups(clean_df)
    rows: list[dict[str, object]] = []
    for event_order, (_, event) in enumerate(events_df.iterrows()):
        channel = groups.get((event["frequency_band"], event["antenna"]))
        if channel is None:
            continue
        group, group_ns = channel
        predicted_time = pd.Timestamp(event["predicted_event_time"])
        uid = _event_uid(event)
        block = lunar_block_id(predicted_time)
        for shift_seconds in config.control_shifts_seconds:
            center_time = predicted_time + pd.to_timedelta(float(shift_seconds), unit="s")
            window = _window_values(group, group_ns, center_time, config)
            if window is None:
                continue
            relative_seconds, power = window
            normalized = _normalize_without_detrending(relative_seconds, power, config)
            if normalized is None:
                # Raw power remains scientifically usable even when a flat or
                # undersampled sideband cannot supply a normalization scale.
                normalized_power = np.full_like(power, np.nan)
                normalization_center = normalization_scale = np.nan
                sideband_count = np.count_nonzero(np.abs(relative_seconds) >= config.sideband_start_seconds)
                normalization_status = ("insufficient_sideband_samples" if sideband_count < config.minimum_sideband_samples
                                        else "zero_or_invalid_sideband_scale")
            else:
                normalized_power, normalization_center, normalization_scale = normalized
                normalization_status = "ok"
            edges = np.arange(
                -config.window_seconds,
                config.window_seconds + config.bin_seconds,
                config.bin_seconds,
                dtype=float,
            )
            if edges[-1] < config.window_seconds:
                edges = np.append(edges, config.window_seconds)
            bin_index = np.digitize(relative_seconds, edges, right=False) - 1
            bin_index[relative_seconds == edges[-1]] = len(edges) - 2
            for index in np.unique(bin_index):
                if index < 0 or index >= len(edges) - 1:
                    continue
                mask = bin_index == index
                if not np.any(mask):
                    continue
                row = event.to_dict()
                row.update(
                    {
                        "event_uid": uid,
                        "event_order": int(event_order),
                        "lunar_block": int(block),
                        "original_predicted_event_time": predicted_time,
                        "alignment_time": center_time,
                        "control_shift_s": float(shift_seconds),
                        "alignment_type": "predicted" if np.isclose(shift_seconds, 0.0) else "fixed_time_control",
                        "t_bin_sec": float(0.5 * (edges[index] + edges[index + 1])),
                        "raw_power": float(np.nanmedian(power[mask])),
                        "normalized_power": float(np.nanmedian(normalized_power[mask])) if np.isfinite(normalized_power[mask]).any() else np.nan,
                        "n_samples": int(np.count_nonzero(mask)),
                        "normalization_center": normalization_center,
                        "normalization_scale": normalization_scale,
                        "normalization_status": normalization_status,
                        "normalization_method": "joint_outer_sideband_median_mad_no_detrend",
                        "validity_mode": config.validity_mode,
                    }
                )
                rows.append(row)
    return add_frequency_mhz_column(pd.DataFrame.from_records(rows))


def fold_occultogram_points(points: pd.DataFrame) -> pd.DataFrame:
    """Fold reappearances so every positive source is high-before/low-after."""

    if points.empty:
        return points.copy()
    event_type = points["event_type"].astype(str).str.lower()
    keep = event_type.isin(["disappearance", "reappearance"])
    folded = points.loc[keep].copy()
    folded["original_event_type"] = folded["event_type"]
    reappearance = folded["event_type"].astype(str).str.lower().eq("reappearance")
    folded.loc[reappearance, "t_bin_sec"] = -pd.to_numeric(
        folded.loc[reappearance, "t_bin_sec"], errors="coerce"
    )
    folded["event_type"] = "folded_source_like"
    return folded.sort_values(["event_order", "control_shift_s", "t_bin_sec"]).reset_index(drop=True)


def summarize_occultogram(points: pd.DataFrame) -> pd.DataFrame:
    """Summarize profiles using event rows and lunar blocks, never samples."""

    if points.empty:
        return pd.DataFrame()
    value_columns = [column for column in ["raw_power", "normalized_power"] if column in points.columns]
    group_columns = [
        column
        for column in [
            "source_name",
            "event_type",
            "frequency_band",
            "frequency_mhz",
            "antenna",
            "control_shift_s",
            "alignment_type",
            "t_bin_sec",
        ]
        if column in points.columns
    ]
    rows: list[dict[str, object]] = []
    for keys, group in points.groupby(group_columns, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: dict[str, object] = dict(zip(group_columns, keys))
        row["n_events"] = int(group["event_uid"].nunique())
        row["n_lunar_blocks"] = int(group["lunar_block"].nunique())
        for value_column in value_columns:
            values = pd.to_numeric(group[value_column], errors="coerce").dropna()
            row[f"n_events_{value_column}"] = int(group.loc[values.index, "event_uid"].nunique())
            row[f"median_{value_column}"] = float(values.median()) if not values.empty else np.nan
            row[f"event_q16_{value_column}"] = float(values.quantile(0.16)) if not values.empty else np.nan
            row[f"event_q84_{value_column}"] = float(values.quantile(0.84)) if not values.empty else np.nan
            block_values = (
                group.assign(_value=pd.to_numeric(group[value_column], errors="coerce"))
                .groupby("lunar_block", dropna=False)["_value"]
                .median()
                .dropna()
            )
            row[f"block_q16_{value_column}"] = float(block_values.quantile(0.16)) if not block_values.empty else np.nan
            row[f"block_q84_{value_column}"] = float(block_values.quantile(0.84)) if not block_values.empty else np.nan
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def event_signed_contrasts(
    points: pd.DataFrame,
    inner_seconds: float = 120.0,
    outer_seconds: float = 600.0,
) -> pd.DataFrame:
    """Return event-level, correctly signed pre/post contrasts.

    Positive values have the source-like sign: pre minus post for a
    disappearance, and post minus pre for a reappearance.  These values are
    supporting diagnostics for the visual occultograms, not detection SNRs.
    """

    if points.empty:
        return pd.DataFrame()
    if not 0 <= inner_seconds < outer_seconds:
        raise ValueError("contrast intervals require 0 <= inner_seconds < outer_seconds")
    id_columns = [
        column
        for column in [
            "source_name",
            "event_uid",
            "event_id",
            "event_order",
            "lunar_block",
            "original_predicted_event_time",
            "event_type",
            "frequency_band",
            "frequency_mhz",
            "antenna",
            "control_shift_s",
            "alignment_type",
        ]
        if column in points.columns
    ]
    rows: list[dict[str, object]] = []
    for keys, group in points.groupby(id_columns, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        meta = dict(zip(id_columns, keys))
        times = pd.to_numeric(group["t_bin_sec"], errors="coerce")
        pre = group[(times >= -outer_seconds) & (times <= -inner_seconds)]
        post = group[(times >= inner_seconds) & (times <= outer_seconds)]
        if pre.empty or post.empty:
            continue
        direction = 1.0 if str(meta.get("event_type", "")).lower() == "disappearance" else -1.0
        row = dict(meta)
        row["contrast_inner_s"] = float(inner_seconds)
        row["contrast_outer_s"] = float(outer_seconds)
        row["n_pre_bins"] = int(len(pre))
        row["n_post_bins"] = int(len(post))
        for value_column in ["raw_power", "normalized_power"]:
            if value_column not in group.columns:
                continue
            pre_values = pd.to_numeric(pre[value_column], errors="coerce").dropna()
            post_values = pd.to_numeric(post[value_column], errors="coerce").dropna()
            pre_value = float(pre_values.median()) if len(pre_values) else np.nan
            post_value = float(post_values.median()) if len(post_values) else np.nan
            row[f"pre_{value_column}"] = pre_value
            row[f"post_{value_column}"] = post_value
            row[f"signed_{value_column}_contrast"] = direction * (pre_value - post_value)
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def compact_occultogram_profiles(points: pd.DataFrame) -> pd.DataFrame:
    """Return the self-contained, non-lossy profile columns needed downstream."""

    if points.empty:
        return points.copy()
    missing = sorted(set(COMPACT_PROFILE_COLUMNS) - set(points.columns))
    if missing:
        raise ValueError(f"occultogram profile table is missing compact columns: {missing}")
    return points.loc[:, COMPACT_PROFILE_COLUMNS].copy()


def occultogram_event_metadata(points: pd.DataFrame) -> pd.DataFrame:
    """Return one metadata row per event/control alignment.

    Event geometry and normalization values do not vary with time bin. Keeping
    them separately prevents thousands of repeated strings and timestamps from
    inflating the profile table.
    """

    if points.empty:
        return points.copy()
    keys = ["event_uid", "control_shift_s"]
    missing = sorted(set(keys) - set(points.columns))
    if missing:
        raise ValueError(f"occultogram table is missing metadata keys: {missing}")
    metadata_columns = [column for column in points.columns if column not in PROFILE_VALUE_COLUMNS]
    metadata = points.loc[:, metadata_columns].drop_duplicates(keys, keep="first")
    if metadata.duplicated(keys).any():
        raise RuntimeError("occultogram metadata keys are not unique")
    return metadata.reset_index(drop=True)
