"""run-control-survey can reuse reviewed cleaned/event tables instead of re-deriving them."""

from __future__ import annotations

import argparse

import pandas as pd
import pytest

from rylevonberg.constants import GEOMETRY_VERSION
from rylevonberg.pipeline import _select_existing_events, build_parser


def _events(geometry_version: str | None) -> pd.DataFrame:
    rows = []
    for source in ["earth", "sun", "jupiter"]:
        for band in [1, 4]:
            for antenna in ["rv1_coarse", "rv2_coarse"]:
                rows.append({
                    "event_id": 0, "source_name": source, "event_type": "disappearance",
                    "predicted_event_time": pd.Timestamp("1975-01-01 00:00"),
                    "frequency_band": band, "antenna": antenna,
                })
    frame = pd.DataFrame(rows)
    if geometry_version is not None:
        frame["geometry_version"] = geometry_version
    return frame


def _args(path, **overrides) -> argparse.Namespace:
    base = {"events": str(path), "source": ["earth", "sun"], "frequency": ["4"], "antenna": ["rv2_coarse"]}
    base.update(overrides)
    return argparse.Namespace(**base)


def test_reused_events_are_filtered_to_requested_channels(tmp_path) -> None:
    path = tmp_path / "events.parquet"
    _events(GEOMETRY_VERSION).to_parquet(path)
    out = _select_existing_events(_args(path), tmp_path / "02_events")
    selected = pd.read_csv(out["predicted_events"])
    assert sorted(selected.source_name.unique()) == ["earth", "sun"]
    assert set(selected.frequency_band) == {4}
    assert set(selected.antenna) == {"rv2_coarse"}


@pytest.mark.parametrize("version", [None, "legacy"])
def test_reused_events_must_have_current_geometry(tmp_path, version) -> None:
    path = tmp_path / "events.parquet"
    _events(version).to_parquet(path)
    with pytest.raises(ValueError, match="Stale event geometry"):
        _select_existing_events(_args(path), tmp_path / "02_events")


def test_control_survey_accepts_cleaned_and_events_without_raw_data() -> None:
    args = build_parser().parse_args([
        "run-control-survey", "--output-dir", "out", "--cleaned", "c.parquet", "--events", "e.parquet",
    ])
    assert args.data is None and args.cleaned == "c.parquet" and args.events == "e.parquet"
