from __future__ import annotations

import numpy as np
import pandas as pd

from rylevonberg.blind import blind_changepoints, cluster_blind_constraints
from rylevonberg.controls import injection_recovery_grid, negative_control_event_ensemble
from rylevonberg.detection import StepFitConfig, run_matched_filter, run_stepfit_detections
from rylevonberg.ingest import IngestOptions, ingest_csv
from rylevonberg.stacking import aligned_profiles, stack_profiles


def _receiver_grid() -> tuple[pd.DataFrame, pd.DataFrame]:
    times = pd.date_range("1974-11-01", periods=50, freq="1min")
    event_time = pd.Timestamp("1974-11-01 00:25:00")
    rows = []
    event_rows = []
    event_id = 0
    for freq in [4, 8]:
        for antenna, scale in [("rv1_coarse", 1.0), ("rv2_coarse", 1.5)]:
            power = np.where(times < event_time, 100.0 * scale, 60.0 * scale)
            for t, y in zip(times, power):
                rows.append(
                    {
                        "time": t,
                        "frequency_band": freq,
                        "antenna": antenna,
                        "power": y,
                        "is_valid": True,
                        "position_x": 2000.0,
                        "position_y": 0.0,
                        "position_z": 0.0,
                    }
                )
            event_rows.append(
                {
                    "event_id": event_id,
                    "source_name": "synthetic",
                    "event_type": "disappearance",
                    "predicted_event_time": event_time,
                    "frequency_band": freq,
                    "antenna": antenna,
                }
            )
            event_id += 1
    return pd.DataFrame(rows), pd.DataFrame(event_rows)


def test_multiband_dual_antenna_stepfit_and_matched_filter() -> None:
    clean, events = _receiver_grid()
    config = StepFitConfig(window_seconds=900, min_samples_per_side=4, timing_grid_seconds=(0.0,))
    step = run_stepfit_detections(clean, events, config)
    matched = run_matched_filter(clean, events, window_seconds=900)
    assert len(step) == 4
    assert len(matched) == 4
    assert set(step["frequency_band"]) == {4, 8}
    assert set(step["antenna"]) == {"rv1_coarse", "rv2_coarse"}
    assert (step["amplitude"] > 0).all()
    assert (matched["matched_amp"] > 0).all()


def test_negative_control_ensemble_is_scaled_and_deterministic() -> None:
    clean, events = _receiver_grid()
    ctl1 = negative_control_event_ensemble(events, clean, n_random=5, seed=7)
    ctl2 = negative_control_event_ensemble(events, clean, n_random=5, seed=7)
    assert len(ctl1) == len(events) * 8
    assert set(ctl1["control_type"]) == {"randomized_time", "time_reversed_template", "wrong_frequency", "wrong_antenna"}
    pd.testing.assert_frame_equal(ctl1.reset_index(drop=True), ctl2.reset_index(drop=True))


def test_injection_recovery_grid_covers_frequency_antenna_window_amplitude() -> None:
    clean, events = _receiver_grid()
    grid = injection_recovery_grid(clean, events, amplitudes=[10.0, 30.0], window_seconds=[300.0, 900.0], min_samples_per_side=2)
    assert len(grid) == 16
    assert set(grid["frequency_band"]) == {4, 8}
    assert set(grid["antenna"]) == {"rv1_coarse", "rv2_coarse"}
    assert set(grid["window_seconds"]) == {300.0, 900.0}
    assert set(grid["injected_amplitude"]) == {10.0, 30.0}


def test_data_quality_regression_flags_gap_saturation_and_jump(tmp_path) -> None:
    path = tmp_path / "quality.csv"
    times = pd.to_datetime(
        [
            "1974-11-01 00:00:00",
            "1974-11-01 00:00:01",
            "1974-11-01 00:00:02",
            "1974-11-01 00:00:20",
            "1974-11-01 00:00:21",
            "1974-11-01 00:00:22",
        ]
    )
    pd.DataFrame(
        {
            "time": times,
            "frequency_band": [4] * 6,
            "position_x": [2000.0] * 6,
            "position_y": [0.0] * 6,
            "position_z": [0.0] * 6,
            "earth_unit_vector_x": [1.0] * 6,
            "earth_unit_vector_y": [0.0] * 6,
            "earth_unit_vector_z": [0.0] * 6,
            "right_ascension": [0.0] * 6,
            "declination": [0.0] * 6,
            "rv1_coarse": [10.0, 10.1, 10.2, 0.0, 10000.0, 10.1],
        }
    ).to_csv(path, index=False)
    clean, report = ingest_csv(path, IngestOptions(value_columns=("rv1_coarse",), use_existing_loader=False, artifact_sigma=3.0))
    flags = ";".join(clean["quality_flags"].astype(str))
    assert "gap_after_previous" in flags
    assert "saturation_or_nonpositive" in flags
    assert "telemetry_artifact_jump" in flags
    assert int(report.iloc[0]["n_gap_flags"]) >= 1


def test_long_window_integration_stacking_and_reproducibility() -> None:
    times = pd.date_range("1974-11-05", periods=360, freq="1min")
    event_times = [pd.Timestamp("1974-11-05 02:00:00"), pd.Timestamp("1974-11-05 04:00:00")]
    power = np.full(len(times), 100.0)
    power[times >= event_times[0]] -= 20.0
    power[times >= event_times[1]] += 20.0
    clean = pd.DataFrame({"time": times, "frequency_band": 4, "antenna": "rv2_coarse", "power": power, "is_valid": True})
    events = pd.DataFrame(
        [
            {"event_id": 0, "source_name": "long", "event_type": "disappearance", "predicted_event_time": event_times[0], "frequency_band": 4, "antenna": "rv2_coarse"},
            {"event_id": 1, "source_name": "long", "event_type": "reappearance", "predicted_event_time": event_times[1], "frequency_band": 4, "antenna": "rv2_coarse"},
        ]
    )
    profiles = aligned_profiles(clean, events, window_seconds=1800, bin_seconds=300)
    stacked1, summary1 = stack_profiles(profiles, n_bootstrap=32, seed=11)
    stacked2, summary2 = stack_profiles(profiles, n_bootstrap=32, seed=11)
    assert not stacked1.empty
    assert summary1.iloc[0]["n_events"] == 2
    pd.testing.assert_frame_equal(summary1.reset_index(drop=True), summary2.reset_index(drop=True))


def test_blind_search_finds_injected_step_and_clusters_constraint() -> None:
    rng = np.random.default_rng(2)
    times = pd.date_range("1974-11-10", periods=80, freq="1min")
    event_time = pd.Timestamp("1974-11-10 00:40:00")
    power = rng.normal(0.0, 0.5, len(times))
    power[times >= event_time] += 20.0
    clean = pd.DataFrame(
        {
            "time": times,
            "frequency_band": 4,
            "antenna": "rv2_coarse",
            "power": power,
            "is_valid": True,
            "position_x": 2000.0,
            "position_y": 0.0,
            "position_z": 0.0,
        }
    )
    candidates = blind_changepoints(clean, window_samples=4, snr_threshold=6.0)
    assert not candidates.empty
    nearest = pd.DatetimeIndex(candidates["candidate_time"]).to_series().sub(event_time).abs().min()
    assert nearest <= pd.Timedelta(minutes=2)
    clusters = cluster_blind_constraints(candidates, time_tolerance_seconds=180)
    assert not clusters.empty
