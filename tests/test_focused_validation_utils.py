from __future__ import annotations

import numpy as np
import pandas as pd

from rylevonberg.offsource import OffsourceConfig, generate_offsource_controls
from rylevonberg.quality import event_window_quality
from rylevonberg.source_summary import add_offsource_pvalues, aggregate_source_level, final_source_summary


def test_generate_offsource_controls_has_required_metadata() -> None:
    sources = pd.DataFrame(
        [{"source_name": "cyg_a", "kind": "fixed", "ra_deg": 299.0, "dec_deg": 40.0, "frame": "fk4"}]
    )
    controls = generate_offsource_controls(sources, OffsourceConfig(annulus_positions=4))
    assert not controls.empty
    assert {"parent_source", "control_name", "control_type", "offset_deg", "ra_fk4", "dec_fk4", "notes"}.issubset(controls.columns)
    assert set(controls["parent_source"]) == {"cyg_a"}
    assert controls["source_name"].is_unique


def test_event_window_quality_reports_specific_failures() -> None:
    times = pd.date_range("1974-01-01", periods=20, freq="1min")
    clean = pd.DataFrame(
        {
            "time": times,
            "frequency_band": 1,
            "frequency_mhz": 0.45,
            "antenna": "rv1_coarse",
            "power": np.ones(20),
            "is_valid": [True] * 8 + [False] * 4 + [True] * 8,
            "quality_flags": [""] * 8 + ["gap_after_previous"] * 4 + [""] * 8,
        }
    )
    events = pd.DataFrame(
        [
            {
                "source_name": "x",
                "event_type": "disappearance",
                "predicted_event_time": times[10],
                "frequency_band": 1,
                "frequency_mhz": 0.45,
                "antenna": "rv1_coarse",
            }
        ]
    )
    quality = event_window_quality(clean, events, 600)
    assert quality.iloc[0]["gap_fraction"] > 0
    assert quality.iloc[0]["primary_quality_failure"] == ""


def test_event_window_quality_rejects_severe_gaps() -> None:
    times = pd.date_range("1974-01-01", periods=20, freq="1min")
    clean = pd.DataFrame(
        {
            "time": times,
            "frequency_band": 1,
            "frequency_mhz": 0.45,
            "antenna": "rv1_coarse",
            "power": np.ones(20),
            "is_valid": [True] * 20,
            "quality_flags": ["gap_after_previous"] * 8 + [""] * 12,
        }
    )
    events = pd.DataFrame(
        [
            {
                "source_name": "x",
                "event_type": "disappearance",
                "predicted_event_time": times[10],
                "frequency_band": 1,
                "frequency_mhz": 0.45,
                "antenna": "rv1_coarse",
            }
        ]
    )
    quality = event_window_quality(clean, events, 600)
    assert quality.iloc[0]["primary_quality_failure"] == "severe_gap_fraction"


def test_event_window_quality_does_not_reject_window_for_two_jump_points() -> None:
    times = pd.date_range("1974-01-01", periods=30, freq="1min")
    flags = [""] * 30
    flags[4] = "telemetry_artifact_jump"
    flags[24] = "telemetry_artifact_jump"
    valid = [True] * 30
    valid[4] = False
    valid[24] = False
    clean = pd.DataFrame(
        {
            "time": times,
            "frequency_band": 1,
            "frequency_mhz": 0.45,
            "antenna": "rv1_coarse",
            "power": np.ones(30),
            "is_valid": valid,
            "quality_flags": flags,
        }
    )
    events = pd.DataFrame(
        [
            {
                "source_name": "x",
                "event_type": "disappearance",
                "predicted_event_time": times[15],
                "frequency_band": 1,
                "frequency_mhz": 0.45,
                "antenna": "rv1_coarse",
            }
        ]
    )
    quality = event_window_quality(clean, events, 900)
    row = quality.iloc[0]
    assert row["jump_count"] == 2
    assert row["jump_fraction"] > 0
    assert row["primary_quality_failure"] == ""


def test_source_level_summary_and_offsource_pvalue() -> None:
    events = pd.DataFrame(
        {
            "source_name": ["earth", "earth"],
            "event_type": ["disappearance", "reappearance"],
            "predicted_event_time": pd.to_datetime(["1974-01-01", "1974-01-02"]),
            "frequency_band": [7, 7],
            "frequency_mhz": [4.7, 4.7],
            "antenna": ["rv2_coarse", "rv2_coarse"],
        }
    )
    scored = events.copy()
    scored["detection_snr"] = [5.0, 6.0]
    scored["best_empirical_p"] = [0.01, 0.02]
    scored["timing_offset_sec"] = [0.0, 30.0]
    scored["quality_clean_fraction"] = [1.0, 1.0]
    stack = pd.DataFrame(
        [{"source_name": "earth", "frequency_band": 7, "frequency_mhz": 4.7, "antenna": "rv2_coarse", "n_events": 2, "stacked_amplitude": 1.0, "stacked_snr": 12.0}]
    )
    quality = pd.DataFrame(
        [{"source_name": "earth", "frequency_band": 7, "antenna": "rv2_coarse", "primary_quality_failure": ""}]
    )
    level = aggregate_source_level(events, scored, stack, quality=quality, window_s=600)
    off = pd.DataFrame(
        {
            "parent_source": ["earth", "earth"],
            "frequency_band": [7, 7],
            "antenna": ["rv2_coarse", "rv2_coarse"],
            "stacked_snr": [1.0, 2.0],
        }
    )
    level = add_offsource_pvalues(level, off)
    final = final_source_summary(level)
    assert level.iloc[0]["offsource_empirical_p"] < 1.0
    assert final.iloc[0]["final_status"] == "positive_control_confirmed"
