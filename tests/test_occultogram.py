from __future__ import annotations

import numpy as np
import pandas as pd

from rylevonberg.occultogram import (
    OccultogramConfig,
    compact_occultogram_profiles,
    event_signed_contrasts,
    extract_occultogram_points,
    fold_occultogram_points,
    occultogram_event_metadata,
    summarize_occultogram,
)


def _synthetic_occultations() -> tuple[pd.DataFrame, pd.DataFrame]:
    times = pd.date_range("1974-01-01", periods=181, freq="1min")
    disappearance = pd.Timestamp("1974-01-01 00:45:00")
    reappearance = pd.Timestamp("1974-01-01 02:15:00")
    power = np.full(len(times), 5.0)
    power[times < disappearance] = 10.0
    power[times >= reappearance] = 10.0
    power += 0.05 * np.sin(np.arange(len(times)))
    clean = pd.DataFrame(
        {
            "time": times,
            "frequency_band": 1,
            "antenna": "rv2_coarse",
            "power": power,
            # The default morphology extraction must not censor finite edges.
            "is_valid": np.where(np.abs(np.gradient(power)) > 2.0, False, True),
        }
    )
    events = pd.DataFrame(
        [
            {
                "event_id": 1,
                "source_name": "earth",
                "event_type": "disappearance",
                "predicted_event_time": disappearance,
                "frequency_band": 1,
                "frequency_mhz": 0.45,
                "antenna": "rv2_coarse",
            },
            {
                "event_id": 2,
                "source_name": "earth",
                "event_type": "reappearance",
                "predicted_event_time": reappearance,
                "frequency_band": 1,
                "frequency_mhz": 0.45,
                "antenna": "rv2_coarse",
            },
        ]
    )
    return clean, events


def test_occultogram_has_correctly_signed_event_contrasts() -> None:
    clean, events = _synthetic_occultations()
    config = OccultogramConfig(
        window_seconds=1200,
        bin_seconds=60,
        sideband_start_seconds=600,
        control_shifts_seconds=(0.0,),
    )
    points = extract_occultogram_points(clean, events, config)
    contrasts = event_signed_contrasts(points, inner_seconds=120, outer_seconds=480)

    assert len(contrasts) == 2
    assert (contrasts["signed_raw_power_contrast"] > 4.5).all()
    assert set(points["validity_mode"]) == {"finite"}


def test_reappearance_folds_to_high_before_and_low_after() -> None:
    clean, events = _synthetic_occultations()
    config = OccultogramConfig(
        window_seconds=1200,
        bin_seconds=60,
        sideband_start_seconds=600,
        control_shifts_seconds=(0.0,),
    )
    folded = fold_occultogram_points(extract_occultogram_points(clean, events, config))
    reappearance_uid = folded.loc[folded["original_event_type"].eq("reappearance"), "event_uid"].iloc[0]
    profile = folded[folded["event_uid"].eq(reappearance_uid)]

    before = profile.loc[profile["t_bin_sec"] < -120, "raw_power"].median()
    after = profile.loc[profile["t_bin_sec"] > 120, "raw_power"].median()
    assert before > after
    assert set(folded["event_type"]) == {"folded_source_like"}


def test_summary_counts_events_and_lunar_blocks_not_samples() -> None:
    clean, events = _synthetic_occultations()
    config = OccultogramConfig(
        window_seconds=1200,
        bin_seconds=60,
        sideband_start_seconds=600,
        control_shifts_seconds=(0.0,),
    )
    summary = summarize_occultogram(fold_occultogram_points(extract_occultogram_points(clean, events, config)))

    central = summary[np.abs(summary["t_bin_sec"]) <= 90]
    assert central["n_events"].max() == 2
    assert central["n_lunar_blocks"].max() == 1
    assert "median_normalized_power" in summary.columns
    assert "block_q16_normalized_power" in summary.columns


def test_fixed_time_control_moves_edge_away_from_zero() -> None:
    clean, events = _synthetic_occultations()
    config = OccultogramConfig(
        window_seconds=1200,
        bin_seconds=60,
        sideband_start_seconds=600,
        control_shifts_seconds=(0.0, 600.0),
    )
    points = extract_occultogram_points(clean, events.iloc[[0]], config)

    edge_locations = {}
    for shift, profile in points.groupby("control_shift_s"):
        profile = profile.sort_values("t_bin_sec")
        jumps = np.abs(np.diff(profile["raw_power"].to_numpy(dtype=float)))
        index = int(np.argmax(jumps))
        times = profile["t_bin_sec"].to_numpy(dtype=float)
        edge_locations[float(shift)] = 0.5 * (times[index] + times[index + 1])

    assert abs(edge_locations[0.0]) <= 60
    assert abs(edge_locations[600.0] + 600.0) <= 60


def test_compact_profiles_separate_repeated_event_metadata() -> None:
    clean, events = _synthetic_occultations()
    config = OccultogramConfig(
        window_seconds=1200,
        bin_seconds=60,
        sideband_start_seconds=600,
        control_shifts_seconds=(0.0, 600.0),
    )
    points = extract_occultogram_points(clean, events, config)
    compact = compact_occultogram_profiles(points)
    metadata = occultogram_event_metadata(points)

    assert len(compact) == len(points)
    assert set(compact.columns) < set(points.columns)
    assert len(metadata) == points[["event_uid", "control_shift_s"]].drop_duplicates().shape[0]
    assert not metadata.duplicated(["event_uid", "control_shift_s"]).any()
