"""Regression coverage for the October source-sensitivity audit."""
from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd
import pytest

from rylevonberg.detection import StepFitConfig, event_template, run_stepfit_detections
from rylevonberg.ingest import validate_timeseries
from rylevonberg.occultogram import (OccultogramConfig, extract_occultogram_points,
                                    event_signed_contrasts, summarize_occultogram)
from rylevonberg.sample_quality import strict_power_mask
from rylevonberg.stacking import (aligned_profiles, detrend_profile_values, fast_stack_summary,
                                 stack_profiles, stack_cache_matches)
from rylevonberg.util import datetime_ns
from rylevonberg.source_summary import add_randomized_stack_pvalues


def channel(event_type="disappearance"):
    t = np.arange(-600., 601., 10.)
    center = pd.Timestamp("1974-01-01 12:00")
    data = pd.DataFrame({"time": center + pd.to_timedelta(t, unit="s"),
                         "power": 100. + 10 * event_template(t, event_type),
                         "frequency_band": 1, "antenna": "rv1_coarse"})
    events = pd.DataFrame([{"event_id": 0, "source_name": "synthetic", "event_type": event_type,
                            "predicted_event_time": center, "frequency_band": 1, "antenna": "rv1_coarse"}])
    return t, data, events


@pytest.mark.parametrize("event_type", ["disappearance", "reappearance"])
def test_quantized_edge_survives_ingest_and_detection(event_type):
    _, data, events = channel(event_type)
    clean, _ = validate_timeseries(data)
    assert clean.is_valid.all()
    assert clean.quality_flags.str.contains("high_power_percentile").any()
    fit = run_stepfit_detections(clean, events, StepFitConfig(timing_grid_seconds=(0.,))).iloc[0]
    assert fit.n_pre > 0 and fit.n_post > 0
    assert fit.amplitude == pytest.approx(10.)


def test_instrument_limit_rejects_saturation_but_default_retains_large_burst():
    _, data, _ = channel()
    data.loc[60, "power"] = 1e9
    default, _ = validate_timeseries(data)
    assert default.loc[60, "is_valid"]
    assert strict_power_mask(default)[60]
    limited, _ = validate_timeseries(data, saturation_power=1000.)
    assert not limited.loc[60, "is_valid"]
    data.loc[59, "power"] = np.inf
    bad, _ = validate_timeseries(data)
    assert not bad.loc[59, "is_valid"]


@pytest.mark.parametrize("event_type", ["disappearance", "reappearance"])
def test_default_stack_preserves_step_with_linear_trend_and_irregular_sampling(event_type):
    t, data, _ = channel(event_type)
    keep = (np.arange(len(t)) % 7) != 0
    t = t[keep]
    y = data.power.to_numpy()[keep] + .03 * t
    profile, template, _, _ = detrend_profile_values(t, y, event_type, normalize=False)
    assert profile @ template / (template @ template) == pytest.approx(10., abs=1e-10)


@pytest.mark.parametrize("baseline", ["joint_step_linear", "sideband_linear", "linear_all"])
def test_target_and_fast_control_paths_match_with_invalid_samples(baseline):
    t, data, events = channel()
    data.power += np.sin(t / 23.)
    data["is_valid"] = True
    data.loc[12:18, "is_valid"] = False
    data.loc[21, "power"] = 0.
    data.loc[30, "power"] = np.nan
    _, target = stack_profiles(aligned_profiles(data, events, baseline_mode=baseline), n_bootstrap=0)
    control = fast_stack_summary(data, events, 600., baseline_mode=baseline)
    for column in ("stacked_amplitude", "stacked_snr", "n_events"):
        assert target[column].iloc[0] == pytest.approx(control[column].iloc[0])


@pytest.mark.parametrize("flat", [True, False])
def test_raw_profile_survives_unavailable_normalization(flat):
    t, data, events = channel()
    data["power"] = np.where((t >= -100) & (t < 0), 20., 10.)
    if not flat:
        data = data[np.abs(t) < 300].copy()  # no outer-sideband support
    points = extract_occultogram_points(data, events, OccultogramConfig(
        window_seconds=600, sideband_start_seconds=300, control_shifts_seconds=(0.,)))
    assert not points.empty
    assert points.raw_power.max() == 20.
    assert points.normalized_power.isna().all()
    assert set(points.normalization_status) == {
        "zero_or_invalid_sideband_scale" if flat else "insufficient_sideband_samples"}
    summary = summarize_occultogram(points)
    assert summary.n_events_raw_power.eq(1).all()
    assert summary.n_events_normalized_power.eq(0).all()
    contrasts = event_signed_contrasts(points, inner_seconds=0., outer_seconds=90.)
    assert contrasts.signed_raw_power_contrast.iloc[0] > 0


def test_processing_metadata_rejects_legacy_or_mismatched_cache():
    _, data, events = channel()
    summary = fast_stack_summary(data, events, 600.)
    assert stack_cache_matches(summary, 600.)
    assert not stack_cache_matches(summary, 900.)
    assert not stack_cache_matches(summary, 600., baseline_mode="linear_all")
    assert not stack_cache_matches(summary.drop(columns="stack_processing_version"), 600.)


def test_current_target_cannot_be_scored_against_legacy_controls():
    t, data, events = channel()
    data.power += np.sin(t / 23.)
    target = fast_stack_summary(data, events, 600.)
    controls = target.assign(stacked_snr=0., control_type="randomized_time_stack", control_id=0)
    assert add_randomized_stack_pvalues(target, controls).randomized_empirical_p.notna().all()
    legacy = controls.drop(columns="stack_processing_version")
    assert add_randomized_stack_pvalues(target, legacy).randomized_empirical_p.isna().all()
    assert add_randomized_stack_pvalues(target.drop(columns="stack_processing_version"), controls).randomized_empirical_p.isna().all()


def test_legacy_random_stack_cache_is_regenerated(tmp_path, monkeypatch):
    import scripts.run_focused_validation as focused
    _, data, events = channel()
    path = tmp_path / "randomized.csv"
    pd.DataFrame({"control_id": [0], "n_input_events": [1], "stacked_snr": [99.]}).to_csv(path, index=False)
    current = fast_stack_summary(data, events, 600.).assign(control_id=0)
    calls = []
    def rebuild(*args, **kwargs):
        calls.append(1)
        return current
    monkeypatch.setattr(focused, "randomized_stack_distribution", rebuild)
    result = focused._load_or_build_randomized_stacks(data, events, 600., path, 1, 7)
    assert len(calls) == 1
    from rylevonberg.stacking import STACK_PROCESSING_VERSION
    assert result.stack_processing_version.eq(STACK_PROCESSING_VERSION).all()


def test_stack_score_does_not_count_injected_signal_as_noise():
    t, data, events = channel()
    scores = []
    for amplitude in (10., 20., 40.):
        data.power = 100. + amplitude * event_template(t, "disappearance") + np.sin(t / 23.)
        scores.append(fast_stack_summary(data, events, 600.).stacked_snr.iloc[0])
    assert scores[1] / scores[0] == pytest.approx(2., rel=.03)
    assert scores[2] / scores[1] == pytest.approx(2., rel=.03)


@pytest.fixture(scope="module")
def solar():
    path = Path(__file__).resolve().parents[2] / "perlmutter_sun_science_v1/run_sun_rv_external_occultation_analysis.py"
    if not path.exists():
        pytest.skip(f"workspace analysis script not present: {path}")
    spec = importlib.util.spec_from_file_location("solar_sensitivity_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_solar_burst_metric_and_transition_lookup_are_timestamp_unit_invariant(solar, unit):
    center = pd.Timestamp("1975-01-01 12:00")
    stamps = pd.date_range(center - pd.Timedelta(minutes=40), periods=81, freq="1min")
    power = 10. + .2 * np.sin(np.arange(81))
    power[40:44] += 10.
    dwell = pd.DataFrame({"time": stamps.as_unit(unit), "power": power})
    metric = solar.metric_at_center(dwell, datetime_ns(dwell.time), center)
    reference = dwell.assign(time=stamps.as_unit("ns"))
    assert metric == solar.metric_at_center(reference, datetime_ns(reference.time), center)
    assert metric["peak_z"] > 40.
    transitions = pd.DataFrame({"predicted_event_time": stamps[[10, 40, 70]].as_unit(unit),
                                "event_type": ["reappearance", "disappearance", "reappearance"]})
    assert solar.find_nearest_transition(center, transitions).predicted_event_time == center
    assert list(solar.sun_visibility_state(stamps[[20, 50]].as_unit("ns"), transitions)) == [True, False]


def test_solar_scan_phase_does_not_depend_on_timestamp_storage_unit():
    path = Path(__file__).resolve().parents[2] / "perlmutter_sun_science_v1/run_independent_peak_drift_analysis.py"
    if not path.exists():
        pytest.skip(f"workspace analysis script not present: {path}")
    spec = importlib.util.spec_from_file_location("drift_sensitivity_test", path)
    drift = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(drift)
    rows = []
    for cycle in range(10):
        for band, offset in [(9, 0.), (1, 120.)]:
            for sample in (0., 2.):
                rows.append({"time": pd.Timestamp("1975-01-01") + pd.Timedelta(seconds=138. * cycle + offset + sample),
                             "frequency_band": band, "antenna": "rv1_coarse"})
    frame = pd.DataFrame(rows)
    for unit in ("us", "ns"):
        frame["time"] = pd.DatetimeIndex(frame.time).as_unit(unit)
        summary, phase = drift.derive_scan_phase(frame, "finite")
        assert summary.scan_period_s.eq(138.).all()
        assert phase[("rv1_coarse", 1)] == 120.
