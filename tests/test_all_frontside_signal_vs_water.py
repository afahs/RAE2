from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "plot_all_frontside_signal_vs_water.py"
SPEC = importlib.util.spec_from_file_location("plot_all_frontside_signal_vs_water", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_merge_same_orbit_fragments_chooses_best_alignment() -> None:
    fragments = pd.DataFrame(
        {
            "pass_id": [10, 11, 12],
            "center_time": pd.to_datetime(
                ["1974-01-01 00:00", "1974-01-01 00:30", "1974-01-01 04:00"]
            ),
            "start_time": pd.to_datetime(
                ["1973-12-31 23:50", "1974-01-01 00:25", "1974-01-01 03:30"]
            ),
            "end_time": pd.to_datetime(
                ["1974-01-01 00:10", "1974-01-01 00:35", "1974-01-01 04:30"]
            ),
            "frontside_duration_s": [1200.0, 600.0, 3600.0],
            "n_geometry_rows": [20, 10, 60],
            "alignment_cosine": [0.8, 0.95, 0.7],
            "minimum_alignment_angle_deg": [36.9, 18.2, 45.6],
            "position_x": [1.0, 2.0, 3.0],
        }
    )
    result = MODULE.merge_same_orbit_fragments(fragments, 1.0)
    assert len(result) == 2
    assert result.loc[0, "center_time"] == pd.Timestamp("1974-01-01 00:30")
    assert result.loc[0, "position_x"] == 2.0
    assert result.loc[0, "fragment_count"] == 2
    assert result.loc[0, "observed_frontside_duration_s"] == 1800.0
    assert result.loc[0, "n_geometry_rows"] == 30
