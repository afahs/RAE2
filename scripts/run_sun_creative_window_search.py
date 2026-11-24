#!/usr/bin/env python
"""Reassess the Sun without combining RAE-2 frequency channels.

Three independent approaches are implemented:

1. A multiscale, ephemeris-locked shift ridge on each channel's median profile.
2. The same channel-local ridge on q84-minus-median, which is sensitive to an
   intermittent upper-tail component rather than a persistent continuum.
3. Optional NOAA/RSTN timestamp gates on event-level RAE-2 contrasts.  The
   reported ground frequency is not required to overlap a RAE-2 channel: it is
   used as an independent solar-activity time marker, never as spectral proof.

No statistic pools frequencies or antennas.  Repetition across fit windows is
used only as a within-channel robustness check, and the second antenna is
reported as an independent replication check.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
DEFAULT_SUMMARY = WORKSPACE / "final_occultograms" / "occultogram_event_and_lunar_block_summary.parquet"
DEFAULT_OUT = WORKSPACE / "sun_detection_reanalysis_v1"

# Project-specific warning inherited from the Earth analysis: these bands can
# be dominated by the Galactic foreground in some intervals.  They are flagged
# rather than removed, because dominance is time dependent.
FOREGROUND_SENSITIVE_FREQUENCIES = (0.70, 0.90, 1.31, 2.20)
# Sixty-second profile bins leave only six points in a +/-180 s fit, which is
# too little leverage for a quadratic baseline plus an edge.  Start at 240 s.
EDGE_WINDOWS_SECONDS = (240.0, 300.0, 600.0)
PRIMARY_WINDOW_SECONDS = 300.0
DELTA_GRID_SECONDS = np.arange(-600.0, 600.0 + 1.0, 60.0)
RSTN_GATE_HALF_WINDOWS_HOURS = (0.25, 1.0, 6.0, 24.0)

STATISTIC_COLUMNS = {
    "median": "median_normalized_power",
    "upper_tail_excess": "upper_tail_excess",
    "lower_tail_excess": "lower_tail_excess",
}


def _robust_scale(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    center = float(np.median(values))
    return float(1.4826 * np.median(np.abs(values - center)))


def huber_fit(design: np.ndarray, values: np.ndarray, max_iter: int = 30) -> np.ndarray:
    """Return a small robust linear fit used by all edge statistics."""

    design = np.asarray(design, dtype=float)
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values) & np.all(np.isfinite(design), axis=1)
    x = design[finite]
    y = values[finite]
    if len(y) < x.shape[1] + 2:
        return np.full(x.shape[1], np.nan)
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    for _ in range(max_iter):
        residual = y - x @ beta
        center = float(np.median(residual))
        scale = _robust_scale(residual)
        if not np.isfinite(scale) or scale <= 1e-10:
            break
        standardized = np.abs(residual - center) / (1.345 * scale)
        weights = np.ones_like(standardized)
        large = standardized > 1.0
        weights[large] = 1.0 / standardized[large]
        root_weight = np.sqrt(weights)
        updated = np.linalg.lstsq(x * root_weight[:, None], y * root_weight, rcond=None)[0]
        if np.max(np.abs(updated - beta)) <= 1e-9 * (1.0 + np.max(np.abs(beta))):
            beta = updated
            break
        beta = updated
    return beta


def robust_edge_amplitude(
    time_seconds: np.ndarray,
    values: np.ndarray,
    center_seconds: float,
    half_window_seconds: float,
) -> float:
    """Fit a high-before/low-after edge with a local quadratic baseline."""

    time_seconds = np.asarray(time_seconds, dtype=float)
    values = np.asarray(values, dtype=float)
    relative = (time_seconds - float(center_seconds)) / float(half_window_seconds)
    use = np.isfinite(relative) & np.isfinite(values) & (np.abs(relative) <= 1.0)
    if np.count_nonzero(use) < 7:
        return np.nan
    x = relative[use]
    edge = np.where(x < 0.0, 0.5, np.where(x > 0.0, -0.5, 0.0))
    design = np.column_stack([np.ones_like(x), x, x * x, edge])
    return float(huber_fit(design, values[use])[-1])


def compute_shift_ridge_for_group(
    group: pd.DataFrame,
    value_column: str,
    half_window_seconds: float,
    delta_grid_seconds: np.ndarray = DELTA_GRID_SECONDS,
) -> pd.DataFrame:
    """Reconstruct a common original-time edge from all fixed alignments.

    A feature at original event offset ``delta`` appears at profile coordinate
    ``delta - control_shift``.  The five shifted summaries are repeated views
    of the same data, so their scatter is a reconstruction diagnostic and not
    an independent uncertainty estimate.
    """

    records: list[dict[str, float | int]] = []
    for delta in np.asarray(delta_grid_seconds, dtype=float):
        amplitudes = []
        for shift, shifted in group.groupby("control_shift_s", observed=True, sort=True):
            shifted = shifted.sort_values("t_bin_sec")
            center = float(delta) - float(shift)
            amplitude = robust_edge_amplitude(
                shifted["t_bin_sec"].to_numpy(dtype=float),
                shifted[value_column].to_numpy(dtype=float),
                center,
                half_window_seconds,
            )
            if np.isfinite(amplitude):
                amplitudes.append(amplitude)
        values = np.asarray(amplitudes, dtype=float)
        records.append(
            {
                "delta_s": float(delta),
                "ridge_amplitude": float(np.median(values)) if values.size else np.nan,
                "ridge_shift_mad": _robust_scale(values),
                "n_reconstructed_shifts": int(values.size),
            }
        )
    return pd.DataFrame.from_records(records)


def _prepare_summary(path: Path) -> pd.DataFrame:
    summary = pd.read_parquet(path)
    numeric = [
        "frequency_mhz",
        "control_shift_s",
        "t_bin_sec",
        "median_normalized_power",
        "event_q16_normalized_power",
        "event_q84_normalized_power",
    ]
    for column in numeric:
        summary[column] = pd.to_numeric(summary[column], errors="coerce")
    summary["upper_tail_excess"] = (
        summary["event_q84_normalized_power"] - summary["median_normalized_power"]
    )
    summary["lower_tail_excess"] = (
        summary["median_normalized_power"] - summary["event_q16_normalized_power"]
    )
    return summary


def build_shift_ridges(summary: pd.DataFrame) -> pd.DataFrame:
    keys = ["source_name", "frequency_mhz", "antenna"]
    records: list[pd.DataFrame] = []
    for key, group in summary.groupby(keys, observed=True, sort=True):
        source, frequency, antenna = key
        for statistic, column in STATISTIC_COLUMNS.items():
            for window in EDGE_WINDOWS_SECONDS:
                ridge = compute_shift_ridge_for_group(group, column, window)
                ridge.insert(0, "half_window_s", float(window))
                ridge.insert(0, "statistic", statistic)
                ridge.insert(0, "antenna", str(antenna))
                ridge.insert(0, "frequency_mhz", float(frequency))
                ridge.insert(0, "source_name", str(source))
                records.append(ridge)
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def _one_sided_lag_rank(values: pd.DataFrame, score_column: str, zero_column: str = "delta_s") -> float:
    zero = values[np.isclose(values[zero_column], 0.0)][score_column].dropna()
    off = values[~np.isclose(values[zero_column], 0.0)][score_column].dropna()
    if zero.empty or off.empty:
        return np.nan
    score = float(zero.iloc[0])
    return float((1 + np.sum(off.to_numpy(dtype=float) >= score)) / (1 + len(off)))


def summarize_channel_ridges(ridges: pd.DataFrame) -> pd.DataFrame:
    keys = ["source_name", "frequency_mhz", "antenna", "statistic", "half_window_s"]
    records = []
    for key, group in ridges.groupby(keys, observed=True, sort=True):
        group = group.sort_values("delta_s")
        zero = group[np.isclose(group["delta_s"], 0.0)]
        valid = group.dropna(subset=["ridge_amplitude"])
        if zero.empty or valid.empty:
            continue
        best = valid.loc[valid["ridge_amplitude"].idxmax()]
        records.append(
            {
                **dict(zip(keys, key)),
                "zero_amplitude": float(zero.iloc[0]["ridge_amplitude"]),
                "zero_shift_mad": float(zero.iloc[0]["ridge_shift_mad"]),
                "best_delta_s": float(best["delta_s"]),
                "best_amplitude": float(best["ridge_amplitude"]),
                "zero_one_sided_lag_rank": _one_sided_lag_rank(group, "ridge_amplitude"),
                "positive_delta_fraction": float(np.mean(valid["ridge_amplitude"] > 0.0)),
            }
        )
    return pd.DataFrame.from_records(records)


def _matches_frequency(value: float, requested: tuple[float, ...]) -> bool:
    return bool(np.any(np.isclose(float(value), np.asarray(requested, dtype=float))))


def build_channel_persistence(channel_summary: pd.DataFrame) -> pd.DataFrame:
    """Summarize multiscale persistence without pooling channels.

    A channel passes only if its own statistic is positive, zero-localized, and
    in the top translated-lag decile at every declared fit width.  Antennas are
    never averaged; the matching antenna is recorded as a replication check.
    """

    keys = ["source_name", "frequency_mhz", "antenna", "statistic"]
    records: list[dict[str, object]] = []
    expected = tuple(float(value) for value in EDGE_WINDOWS_SECONDS)
    for key, group in channel_summary.groupby(keys, observed=True, sort=True):
        source, frequency, antenna, statistic = key
        by_window = {
            float(row["half_window_s"]): row
            for _, row in group.iterrows()
            if np.isfinite(pd.to_numeric(row["half_window_s"], errors="coerce"))
        }
        present = [window for window in expected if window in by_window]
        rows = [by_window[window] for window in present]
        positives = [float(row["zero_amplitude"]) > 0.0 for row in rows]
        localized = [abs(float(row["best_delta_s"])) <= 120.0 for row in rows]
        ranked = [float(row["zero_one_sided_lag_rank"]) <= 0.10 for row in rows]
        complete = len(present) == len(expected)
        passes = bool(complete and all(positives) and all(localized) and all(ranked))
        record: dict[str, object] = {
            "source_name": str(source),
            "frequency_mhz": float(frequency),
            "antenna": str(antenna),
            "statistic": str(statistic),
            "n_windows_present": int(len(present)),
            "positive_all_windows": bool(complete and all(positives)),
            "localized_all_windows": bool(complete and all(localized)),
            "top_decile_all_windows": bool(complete and all(ranked)),
            "passes_all_windows": passes,
            "source_like_statistic": str(statistic) in {"median", "upper_tail_excess"},
            "foreground_sensitive_band": _matches_frequency(
                float(frequency), FOREGROUND_SENSITIVE_FREQUENCIES
            ),
            "median_zero_amplitude": (
                float(np.median([float(row["zero_amplitude"]) for row in rows]))
                if rows
                else np.nan
            ),
            "minimum_zero_amplitude": (
                float(np.min([float(row["zero_amplitude"]) for row in rows]))
                if rows
                else np.nan
            ),
            "maximum_abs_best_delta_s": (
                float(np.max([abs(float(row["best_delta_s"])) for row in rows]))
                if rows
                else np.nan
            ),
            "maximum_lag_rank": (
                float(np.max([float(row["zero_one_sided_lag_rank"]) for row in rows]))
                if rows
                else np.nan
            ),
        }
        for window in expected:
            label = str(int(window))
            row = by_window.get(window)
            record[f"zero_amplitude_{label}s"] = (
                float(row["zero_amplitude"]) if row is not None else np.nan
            )
            record[f"best_delta_{label}s"] = (
                float(row["best_delta_s"]) if row is not None else np.nan
            )
            record[f"lag_rank_{label}s"] = (
                float(row["zero_one_sided_lag_rank"]) if row is not None else np.nan
            )
        records.append(record)

    out = pd.DataFrame.from_records(records)
    if out.empty:
        return out
    lookup = {
        (str(row.source_name), float(row.frequency_mhz), str(row.statistic), str(row.antenna)):
        bool(row.passes_all_windows)
        for row in out.itertuples(index=False)
    }
    replications = []
    for row in out.itertuples(index=False):
        matching = [
            passed
            for (source, frequency, statistic, antenna), passed in lookup.items()
            if source == str(row.source_name)
            and np.isclose(frequency, float(row.frequency_mhz))
            and statistic == str(row.statistic)
            and antenna != str(row.antenna)
        ]
        replications.append(bool(any(matching)))
    out["other_antenna_passes_same_channel"] = replications
    out["candidate_class"] = np.where(
        out["passes_all_windows"] & out["source_like_statistic"],
        np.where(
            out["foreground_sensitive_band"],
            "persistent_but_foreground_sensitive",
            np.where(
                out["other_antenna_passes_same_channel"],
                "persistent_and_antenna_replicated",
                "persistent_single_antenna",
            ),
        ),
        np.where(
            out["passes_all_windows"] & out["statistic"].eq("lower_tail_excess"),
            "persistent_lower_tail_control",
            "not_persistent",
        ),
    )
    return out


def parse_rstn_time_token(token: object) -> float | None:
    digits = re.sub(r"[^0-9]", "", str(token or ""))
    if len(digits) < 4:
        return None
    hour = int(digits[:2])
    minute = int(digits[2:4])
    if hour > 23 or minute > 59:
        return None
    fractional_minute = int(digits[4]) / 10.0 if len(digits) >= 5 else 0.0
    return hour + (minute + fractional_minute) / 60.0


def parse_rstn_line(line: str) -> dict[str, object] | None:
    # Later archive years prepend compact status flags (for example ``S/F`` or
    # ``FAL``) directly to the YYMMDD field.  Locate the date/frequency/start
    # triplet instead of assuming that the date is the first whitespace token.
    match = re.search(
        r"(?<!\d)(?P<date>\d{6})\s+(?P<frequency>\d+)(?P<station>[A-Za-z]+)\s+"
        r"(?P<start>\d{4,5})",
        line,
    )
    if match is None:
        return None
    date_token = match.group("date")
    year_two_digit = int(date_token[:2])
    year = 1900 + year_two_digit if year_two_digit >= 50 else 2000 + year_two_digit
    month = int(date_token[2:4])
    day = int(date_token[4:6])
    start_hour = parse_rstn_time_token(match.group("start"))
    if start_hour is None:
        return None
    try:
        date = pd.Timestamp(year=year, month=month, day=day)
    except ValueError:
        return None
    return {
        "date": date,
        "start_time": date + pd.to_timedelta(int(round(start_hour * 3600.0)), unit="s"),
        "frequency_mhz_reported": float(match.group("frequency")),
        "station": match.group("station"),
    }


def summarize_rstn(external_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for path in sorted(external_dir.glob("solar-radio_bursts_fixed-frequency_*.txt")):
        for line in path.read_text(encoding="latin1", errors="replace").splitlines():
            parsed = parse_rstn_line(line)
            if parsed is not None:
                rows.append(parsed)
    events = pd.DataFrame.from_records(rows)
    if events.empty:
        return events, pd.DataFrame(), pd.DataFrame()
    events = events.sort_values("start_time").reset_index(drop=True)
    gaps = events["start_time"].diff().dt.total_seconds().fillna(np.inf)
    events["episode_id_30m"] = (gaps > 1800.0).cumsum().astype(int)
    events["month"] = events["date"].dt.to_period("M").astype(str)
    monthly = (
        events.groupby("month", observed=True)
        .agg(
            report_records=("start_time", "size"),
            clustered_episodes_30m=("episode_id_30m", "nunique"),
            active_days=("date", "nunique"),
            minimum_reported_frequency_mhz=("frequency_mhz_reported", "min"),
        )
        .reset_index()
    )
    yearly = (
        events.assign(year=events["date"].dt.year)
        .groupby("year", observed=True)
        .agg(
            report_records=("start_time", "size"),
            clustered_episodes_30m=("episode_id_30m", "nunique"),
            active_days=("date", "nunique"),
            records_at_or_below_30mhz=("frequency_mhz_reported", lambda x: int(np.sum(x <= 30.0))),
            direct_rae2_band_overlap_records=(
                "frequency_mhz_reported", lambda x: int(np.sum(x <= 9.18))
            ),
            minimum_reported_frequency_mhz=("frequency_mhz_reported", "min"),
        )
        .reset_index()
    )
    mission = events[
        (events["start_time"] >= pd.Timestamp("1974-11-01"))
        & (events["start_time"] <= pd.Timestamp("1975-06-28 23:59:59"))
    ]
    mission_row = pd.DataFrame(
        [
            {
                "year": "1974-11-01_to_1975-06-28",
                "report_records": int(len(mission)),
                "clustered_episodes_30m": int(mission["episode_id_30m"].nunique()),
                "active_days": int(mission["date"].nunique()),
                "records_at_or_below_30mhz": int(np.sum(mission["frequency_mhz_reported"] <= 30.0)),
                "direct_rae2_band_overlap_records": int(np.sum(mission["frequency_mhz_reported"] <= 9.18)),
                "minimum_reported_frequency_mhz": float(mission["frequency_mhz_reported"].min()),
            }
        ]
    )
    yearly = pd.concat([yearly, mission_row], ignore_index=True)
    return events, monthly, yearly


def _count_reports_near(
    event_times: pd.Series,
    report_times: pd.Series,
    half_window_hours: float,
) -> np.ndarray:
    """Count external reports in a symmetric window around each event."""

    events = pd.to_datetime(event_times, errors="coerce", utc=True)
    reports = pd.to_datetime(report_times, errors="coerce", utc=True).dropna()
    counts = np.zeros(len(events), dtype=int)
    valid = ~events.isna()
    if not np.any(valid) or reports.empty:
        return counts
    event_ns = events[valid].astype("int64").to_numpy()
    report_ns = np.sort(reports.astype("int64").to_numpy())
    half_ns = int(float(half_window_hours) * 3600.0 * 1e9)
    left = np.searchsorted(report_ns, event_ns - half_ns, side="left")
    right = np.searchsorted(report_ns, event_ns + half_ns, side="right")
    counts[np.asarray(valid)] = right - left
    return counts


def _rstn_window_label(hours: float) -> str:
    minutes = int(round(float(hours) * 60.0))
    return f"{minutes}m" if minutes < 60 else f"{int(round(float(hours)))}h"


def build_rstn_timing_gate_products(
    event_features: pd.DataFrame,
    rstn_events: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply NOAA report timestamps to every RAE-2 channel independently.

    Two external scopes are retained: every fixed-frequency report, and the
    subset reported at <=30 MHz (18 MHz in the mission-era files).  Neither is
    matched to the RAE-2 observing frequency.  The latter is a closer solar-
    burst prior; both remain timing selectors rather than spectral evidence.
    """

    required = {"predicted_event_time", "frequency_mhz", "source_like_log_contrast"}
    missing = sorted(required.difference(event_features.columns))
    if missing:
        raise ValueError("event feature table is missing columns: " + ", ".join(missing))
    annotated = event_features.copy()
    annotated["predicted_event_time"] = pd.to_datetime(
        annotated["predicted_event_time"], errors="coerce", utc=True
    ).dt.tz_convert(None)
    annotated["frequency_mhz"] = pd.to_numeric(annotated["frequency_mhz"], errors="coerce")
    annotated["source_like_log_contrast"] = pd.to_numeric(
        annotated["source_like_log_contrast"], errors="coerce"
    )
    if "analysis_source" not in annotated:
        annotated["analysis_source"] = "sun"
    if "antenna" not in annotated:
        annotated["antenna"] = "unspecified"
    if "control_family" not in annotated:
        annotated["control_family"] = "real"
    if "control_id" not in annotated:
        annotated["control_id"] = "real"

    if rstn_events.empty:
        scopes = {
            "all_reported_frequencies": pd.Series(dtype="datetime64[ns]"),
            "reported_le_30mhz": pd.Series(dtype="datetime64[ns]"),
        }
    else:
        report_frequency = pd.to_numeric(
            rstn_events["frequency_mhz_reported"], errors="coerce"
        )
        scopes = {
            "all_reported_frequencies": rstn_events["start_time"],
            "reported_le_30mhz": rstn_events.loc[report_frequency <= 30.0, "start_time"],
        }

    rows: list[dict[str, object]] = []
    sun = annotated[annotated["analysis_source"].astype(str).str.lower().eq("sun")]
    channel_keys = ["frequency_mhz", "antenna"]
    for scope_name, report_times in scopes.items():
        for half_window in RSTN_GATE_HALF_WINDOWS_HOURS:
            column = f"rstn_{scope_name}_{_rstn_window_label(half_window)}_count"
            annotated[column] = _count_reports_near(
                annotated["predicted_event_time"], report_times, half_window
            )
            sun = annotated[annotated["analysis_source"].astype(str).str.lower().eq("sun")]
            for key, channel in sun.groupby(channel_keys, observed=True, sort=True):
                selected = channel[channel[column].astype(int) > 0]
                real_total = channel[channel["control_family"].astype(str).eq("real")]
                real = selected[selected["control_family"].astype(str).eq("real")]
                controls = selected[~selected["control_family"].astype(str).eq("real")]
                real_values = real["source_like_log_contrast"].dropna().to_numpy(dtype=float)
                control_values = controls["source_like_log_contrast"].dropna().to_numpy(dtype=float)
                group_medians = (
                    controls.groupby(["control_family", "control_id"], observed=True)[
                        "source_like_log_contrast"
                    ]
                    .median()
                    .dropna()
                    .to_numpy(dtype=float)
                )
                real_median = float(np.median(real_values)) if len(real_values) else np.nan
                empirical = (
                    float((1 + np.sum(group_medians >= real_median)) / (1 + len(group_medians)))
                    if len(group_medians) and np.isfinite(real_median)
                    else np.nan
                )
                rows.append(
                    {
                        "frequency_mhz": float(key[0]),
                        "antenna": str(key[1]),
                        "rstn_scope": scope_name,
                        "gate_half_window_h": float(half_window),
                        "reported_frequency_matched_to_rae2": False,
                        "n_real_total": int(len(real_total)),
                        "n_real_selected": int(len(real_values)),
                        "real_selection_fraction": (
                            float(len(real_values) / len(real_total)) if len(real_total) else np.nan
                        ),
                        "real_median_log_contrast": real_median,
                        "real_positive_fraction": (
                            float(np.mean(real_values > 0.0)) if len(real_values) else np.nan
                        ),
                        "n_control_selected": int(len(control_values)),
                        "n_control_groups": int(len(group_medians)),
                        "control_group_empirical_p_median_ge_real": empirical,
                    }
                )
    return annotated, pd.DataFrame.from_records(rows)


def _plot_channel_persistence(
    ridges: pd.DataFrame,
    persistence: pd.DataFrame,
    output: Path,
) -> None:
    candidates = persistence[
        persistence["source_name"].eq("sun")
        & persistence["passes_all_windows"]
        & persistence["source_like_statistic"]
    ].sort_values(["frequency_mhz", "antenna", "statistic"])
    if candidates.empty:
        return
    # Plot one row per frequency/statistic lead.  Matching antennas are shown
    # for comparison but are never combined numerically.
    candidates = candidates.drop_duplicates(["frequency_mhz", "statistic"]).head(4)
    figure, axes = plt.subplots(
        len(candidates),
        len(EDGE_WINDOWS_SECONDS),
        figsize=(14.0, 3.5 * len(candidates)),
        sharex=True,
        squeeze=False,
        constrained_layout=True,
    )
    colors = {"rv1_coarse": "#1b9e77", "rv2_coarse": "#d95f02"}
    for row_index, candidate in enumerate(candidates.itertuples(index=False)):
        for column_index, window in enumerate(EDGE_WINDOWS_SECONDS):
            axis = axes[row_index, column_index]
            panel = ridges[
                ridges["source_name"].eq("sun")
                & np.isclose(ridges["frequency_mhz"], float(candidate.frequency_mhz))
                & ridges["statistic"].eq(str(candidate.statistic))
                & np.isclose(ridges["half_window_s"], float(window))
            ]
            for antenna, line in panel.groupby("antenna", observed=True, sort=True):
                line = line.sort_values("delta_s")
                axis.plot(
                    line["delta_s"],
                    line["ridge_amplitude"],
                    marker="o",
                    markersize=2.8,
                    linewidth=1.3,
                    color=colors.get(str(antenna), None),
                    label=str(antenna),
                )
            axis.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
            axis.axhline(0.0, color="0.5", linewidth=0.8)
            axis.grid(alpha=0.2)
            axis.set_title(
                f"{float(candidate.frequency_mhz):.2f} MHz, "
                f"{str(candidate.statistic).replace('_', ' ')}, ±{int(window)} s"
            )
            if row_index == len(candidates) - 1:
                axis.set_xlabel("offset from solar ephemeris (s)")
            if column_index == 0:
                axis.set_ylabel("edge amplitude (one channel)")
            if row_index == 0 and column_index == 0:
                axis.legend(frameon=False)
    figure.suptitle(
        "Sun channel-local shift ridges (antennas compared, never averaged)",
        fontsize=13,
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_rstn(monthly: pd.DataFrame, output: Path) -> None:
    if monthly.empty:
        return
    dates = pd.to_datetime(monthly["month"] + "-01")
    fig, ax = plt.subplots(figsize=(10.5, 4.7), constrained_layout=True)
    ax.bar(dates, monthly["clustered_episodes_30m"], width=22.0, color="#b2182b", alpha=0.8)
    ax.axvspan(pd.Timestamp("1974-11-01"), pd.Timestamp("1975-06-28"), color="#1b9e77", alpha=0.12)
    ax.set_ylabel("NOAA report clusters (30 min)")
    ax.set_xlabel("month")
    ax.set_title("NOAA fixed-frequency burst activity: timestamps remain useful below 18 MHz")
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def write_report(
    out_dir: Path,
    persistence: pd.DataFrame,
    timing_summary: pd.DataFrame,
    rstn_yearly: pd.DataFrame,
    plot_paths: list[Path],
    event_features_path: Path | None,
) -> Path:
    mission_rstn = rstn_yearly[rstn_yearly["year"].astype(str).eq("1974-11-01_to_1975-06-28")]
    mission_display_columns = [
        "year",
        "report_records",
        "clustered_episodes_30m",
        "active_days",
        "records_at_or_below_30mhz",
        "minimum_reported_frequency_mhz",
    ]
    mission_display = mission_rstn[
        [column for column in mission_display_columns if column in mission_rstn.columns]
    ]
    candidates = persistence[
        persistence["source_name"].eq("sun")
        & persistence["passes_all_windows"]
        & persistence["source_like_statistic"]
    ].sort_values(["frequency_mhz", "antenna", "statistic"])
    lower_controls = persistence[
        persistence["source_name"].eq("sun")
        & persistence["passes_all_windows"]
        & persistence["statistic"].eq("lower_tail_excess")
    ].sort_values(["frequency_mhz", "antenna"])

    candidate_table = [
        "| Frequency | Antenna/statistic | zero amplitude: ±240 / ±300 / ±600 s | best offsets | other antenna passes? | interpretation |",
        "|---:|---|---:|---:|:---:|---|",
    ]
    for row in candidates.itertuples(index=False):
        amplitudes = " / ".join(
            f"{float(getattr(row, f'zero_amplitude_{window}s')):.3g}"
            for window in (240, 300, 600)
        )
        offsets = " / ".join(
            f"{float(getattr(row, f'best_delta_{window}s')):.0f}"
            for window in (240, 300, 600)
        )
        if bool(row.foreground_sensitive_band):
            interpretation = "persistent, but foreground-sensitive"
        elif bool(row.other_antenna_passes_same_channel):
            interpretation = "persistent and antenna-replicated"
        else:
            interpretation = "persistent in one antenna only"
        candidate_table.append(
            f"| {float(row.frequency_mhz):.2f} MHz | {row.antenna}; "
            f"{str(row.statistic).replace('_', ' ')} | {amplitudes} | {offsets} s | "
            f"{'yes' if bool(row.other_antenna_passes_same_channel) else 'no'} | {interpretation} |"
        )
    if len(candidate_table) == 2:
        candidate_table.append("| — | — | — | — | — | no channel passed all windows |")

    antenna_checks: list[str] = []
    for row in candidates.itertuples(index=False):
        other = persistence[
            persistence["source_name"].eq("sun")
            & np.isclose(persistence["frequency_mhz"], float(row.frequency_mhz))
            & persistence["statistic"].eq(str(row.statistic))
            & ~persistence["antenna"].eq(str(row.antenna))
        ]
        if other.empty:
            continue
        other_row = other.iloc[0]
        antenna_checks.append(
            f"- {float(row.frequency_mhz):.2f} MHz {str(row.statistic).replace('_', ' ')}: "
            f"{row.antenna} passes, while {other_row['antenna']} does not "
            f"(300 s zero amplitude {float(other_row['zero_amplitude_300s']):.3g}, "
            f"best offset {float(other_row['best_delta_300s']):.0f} s)."
        )

    timing_lines: list[str]
    if timing_summary.empty:
        timing_lines = [
            "The timestamp gate was not executed because the local transfer has no event-level feature table. "
            "This is a missing-input limitation, not a frequency-overlap limitation.",
            "",
            "When supplied with `--event-features`, the script applies ±15 min, ±1 h, ±6 h, and ±24 h gates independently to every RAE-2 frequency/antenna. "
            "The first two are burst-coincidence windows; the latter two are activity-state windows.",
        ]
    else:
        timing_lines = [
            f"Event table: `{event_features_path}`.",
            "",
            "The channel-local output is `noaa_timing_gated_channel_summary.csv`. "
            "No reported frequency was matched to or pooled with a RAE-2 frequency.",
        ]
        candidate_timing = timing_summary[
            timing_summary["rstn_scope"].eq("reported_le_30mhz")
            & np.any(
                np.isclose(
                    timing_summary["frequency_mhz"].to_numpy(dtype=float)[:, None],
                    candidates["frequency_mhz"].to_numpy(dtype=float)[None, :],
                ),
                axis=1,
            )
        ] if not candidates.empty else pd.DataFrame()
        for row in candidate_timing.itertuples(index=False):
            timing_lines.append(
                f"- {float(row.frequency_mhz):.2f} MHz {row.antenna}, ±{float(row.gate_half_window_h):g} h: "
                f"selected {int(row.n_real_selected)}/{int(row.n_real_total)} real events; "
                f"median log contrast {float(row.real_median_log_contrast):.3g}, "
                f"positive fraction {float(row.real_positive_fraction):.3g}."
            )

    lower_text = (
        ", ".join(
            f"{float(row.frequency_mhz):.2f} MHz {row.antenna}"
            for row in lower_controls.itertuples(index=False)
        )
        if not lower_controls.empty
        else "none"
    )
    lines = [
        "# RAE-2 Sun Reanalysis — Channel-Local Methods",
        "",
        "**Result: no confirmed detection.** Two channel-local features survive the multiscale localization screen, but neither replicates in the other antenna; the lower-frequency one is also in a band known from the Earth analysis to become Galactic-foreground dominated.",
        "",
        "## Analysis contract",
        "",
        "1. Quiet component: a median-power occultation edge, fitted separately to every frequency and antenna.",
        "2. Bursty component: a q84-minus-median edge, again fitted separately to every frequency and antenna.",
        "3. External timing: NOAA fixed-frequency report timestamps applied to event contrasts within each RAE-2 channel, without requiring or implying spectral overlap.",
        "",
        "There is no cross-frequency or cross-antenna aggregation. Persistence means that the same channel passes the sign, |best offset| ≤120 s, and translated-lag-rank ≤0.10 gates at all three fit half-widths (240, 300, and 600 s). The second antenna is an independent replication check, not another sample in a pooled score.",
        "",
        "## Channel-local leads",
        "",
        *candidate_table,
        "",
        "The 0.70 MHz RV2 median feature is interesting because it remains centered at zero across window sizes, but it cannot be separated from a time-varying Galactic foreground with these aggregate profiles. The 4.70 MHz RV2 upper-tail feature is the stronger solar-burst follow-up because it is not in the foreground-warning list and remains exactly zero-centered, but RV1 has the opposite sign at the 240, 300, and 600 s windows.",
        "",
        *antenna_checks,
        "",
        f"Persistent lower-tail controls occur at: {lower_text}. That matters: a centered lower-tail effect is evidence that distributional/systematic structure can also localize at the ephemeris, so upper-tail localization alone is not sufficient.",
        "",
        "The translated-offset rank is only a localization diagnostic. Neighboring offsets are correlated, and the five stored shifts reuse samples; none of these ranks is a formal p-value.",
        "",
        "## NOAA timing gate",
        "",
        *timing_lines,
        "",
        "The mission-era NOAA archive reaches 18 MHz, while these Ryle–Vonberg channels stop at 9.18 MHz. That is not a reason to discard the timestamps. Type-III emission drifts rapidly from high to low frequency as the electron beam propagates outward, so an 18 MHz report can define a blinded time window for inspecting each lower-frequency RAE-2 channel. Lack of overlap means only that the report is an external activity marker—not proof of emission in a specific RAE-2 band.",
        "",
        mission_display.to_string(index=False) if not mission_display.empty else "(NOAA files unavailable)",
        "",
        "The interval approaches the March 1976 solar-cycle minimum, making a persistent quiet-Sun edge less favorable. Low-frequency type-III bursts are intermittent and directionally biased, while scattering broadens their apparent position and timing. Those facts favor event selection and upper-tail statistics over a naive all-event stack, and they favor nested timing windows rather than one arbitrary coincidence width.",
        "",
        "## What would confirm the lead",
        "",
        "The local copy retains the 36,000-row aggregate summary but not the individual event and lunar-block products. A defensible solar result therefore still requires:",
        "",
        "- running the RSTN gates on the full event-feature table, especially for 4.70 MHz RV2;",
        "- verifying disappearance and reappearance separately;",
        "- checking lunar-block stability and matched off-source controls within that same channel;",
        "- a block permutation that includes all channel/window choices;",
        "- channel-specific injection recovery.",
        "",
        "The existing `scripts/run_solar_burst_spice_jupiter_visibility.py` can build the event-level RSTN features. If the 32-step burst-receiver records (25 kHz–13.1 MHz) can be recovered, their denser cadence is the best route for resolving burst morphology; the present Ryle–Vonberg aggregate remains useful for timing and localization.",
        "",
        "## External sources used",
        "",
        "- [RAE-2 instrumentation and 25 kHz–13.1 MHz receiver coverage](https://ntrs.nasa.gov/citations/19750010212)",
        "- [NASA RAE program overview](https://ntrs.nasa.gov/citations/19900001489)",
        "- [NOAA fixed-frequency burst archive](https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-features/solar-radio/radio-bursts/reports/fixed-frequency-listings/)",
        "- [NOAA fixed-frequency record format](https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-features/solar-radio/radio-bursts/reports/fixed-frequency-listings/documentation/dataset-format_solar-radio_bursts_fixed-frequency_listings.pdf)",
        "- [Fitzenreiter, Fainberg & Bundy (1976), low-frequency type-III directivity](https://ntrs.nasa.gov/citations/19770027734)",
        "- [STEREO/WAVES summary of type-III drift and scattering](https://stereo-ssc.nascom.nasa.gov/instruments/documentation/swaves/swaves_science.html)",
        "- [SILSO solar-cycle minima](https://sidc.be/SILSO/cyclesmm)",
        "",
        "## Products",
        "",
        *[f"- `{path.name}`" for path in plot_paths],
        "- `shift_ridge_channel_summary.csv`",
        "- `channel_persistence_summary.csv`",
        "- `noaa_timing_gated_channel_summary.csv` (when event features are supplied)",
        "- `noaa_rstn_activity_summary.csv`",
    ]
    report = out_dir / "SUN_DETECTION_REANALYSIS.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--event-features",
        type=Path,
        default=None,
        help=(
            "Optional event-level CSV/parquet with predicted_event_time, frequency_mhz, "
            "and source_like_log_contrast. NOAA timestamps are applied per channel."
        ),
    )
    parser.add_argument(
        "--reuse-computed",
        action="store_true",
        help="Reuse existing channel ridge CSVs and refresh timing products, plots, and report.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary: pd.DataFrame | None = None
    if args.reuse_computed:
        required = {
            "ridges": args.out_dir / "shift_ridges.csv",
            "channel_summary": args.out_dir / "shift_ridge_channel_summary.csv",
        }
        missing = [str(path) for path in required.values() if not path.exists()]
        if missing:
            raise FileNotFoundError("cannot reuse computed products; missing: " + ", ".join(missing))
        tables = {name: pd.read_csv(path) for name, path in required.items()}
        ridges = tables["ridges"]
        channel_summary = tables["channel_summary"]
    else:
        summary = _prepare_summary(args.summary)
        ridges = build_shift_ridges(summary)
        channel_summary = summarize_channel_ridges(ridges)
    persistence = build_channel_persistence(channel_summary)
    rstn_events, rstn_monthly, rstn_yearly = summarize_rstn(args.out_dir / "external_data")

    timing_annotated = pd.DataFrame()
    timing_summary = pd.DataFrame()
    if args.event_features is not None:
        if not args.event_features.exists():
            raise FileNotFoundError(f"event feature table does not exist: {args.event_features}")
        if args.event_features.suffix.lower() in {".parquet", ".pq"}:
            event_features = pd.read_parquet(args.event_features)
        else:
            event_features = pd.read_csv(args.event_features)
        timing_annotated, timing_summary = build_rstn_timing_gate_products(
            event_features, rstn_events
        )

    ridges.to_csv(args.out_dir / "shift_ridges.csv", index=False)
    channel_summary.to_csv(args.out_dir / "shift_ridge_channel_summary.csv", index=False)
    persistence.to_csv(args.out_dir / "channel_persistence_summary.csv", index=False)
    if not timing_annotated.empty:
        timing_annotated.to_csv(args.out_dir / "noaa_timing_gated_event_features.csv", index=False)
        timing_summary.to_csv(args.out_dir / "noaa_timing_gated_channel_summary.csv", index=False)
    if not rstn_events.empty:
        rstn_events.to_csv(args.out_dir / "noaa_rstn_events_parsed.csv", index=False)
        rstn_monthly.to_csv(args.out_dir / "noaa_rstn_monthly_activity.csv", index=False)
        rstn_yearly.to_csv(args.out_dir / "noaa_rstn_activity_summary.csv", index=False)

    plot_paths = [
        args.out_dir / "sun_channel_persistence.png",
        args.out_dir / "noaa_rstn_activity_timeline.png",
    ]
    _plot_channel_persistence(ridges, persistence, plot_paths[0])
    _plot_rstn(rstn_monthly, plot_paths[1])
    plot_paths = [path for path in plot_paths if path.exists()]

    report = write_report(
        args.out_dir,
        persistence,
        timing_summary,
        rstn_yearly,
        plot_paths,
        args.event_features,
    )
    manifest = {
        "summary_input": str(args.summary),
        "summary_rows": int(len(summary)) if summary is not None else None,
        "sun_summary_rows": (
            int(np.sum(summary["source_name"].astype(str).eq("sun")))
            if summary is not None
            else None
        ),
        "event_features_input": str(args.event_features) if args.event_features else None,
        "edge_windows_seconds": EDGE_WINDOWS_SECONDS,
        "primary_window_seconds": PRIMARY_WINDOW_SECONDS,
        "rstn_gate_half_windows_hours": RSTN_GATE_HALF_WINDOWS_HOURS,
        "cross_frequency_aggregation": False,
        "cross_antenna_aggregation": False,
        "formal_inference": False,
        "limitation": (
            "aggregate profiles only; event dates and block profiles absent locally"
            if args.event_features is None
            else "event features supplied; individual lunar-block profiles still absent locally"
        ),
        "report": report.name,
        "plots": [path.name for path in plot_paths],
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    # These were generated by the earlier pooled analysis and are removed so
    # they cannot be mistaken for current evidence.
    deprecated = (
        "shift_ridge_composites.csv",
        "shift_ridge_composite_summary.csv",
        "earth_template_composites.csv",
        "earth_template_composite_summary.csv",
        "earth_template_projections.csv",
        "earth_template_injection_recovery.csv",
        "sun_shift_ridge_multiscale.png",
        "sun_earth_template_projection.png",
        "sun_earth_template_injection_recovery.png",
    )
    for name in deprecated:
        (args.out_dir / name).unlink(missing_ok=True)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
