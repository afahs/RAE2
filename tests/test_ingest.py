from __future__ import annotations

import pandas as pd

from rylevonberg.ingest import IngestOptions, ingest_csv


def test_ingest_flags_duplicate_and_malformed(tmp_path) -> None:
    path = tmp_path / "mini.csv"
    pd.DataFrame(
        {
            "time": ["1974-01-01 00:00:00", "1974-01-01 00:00:00", "bad"],
            "frequency_band": [1, 1, 1],
            "position_x": [2000.0, 2000.0, 2000.0],
            "position_y": [0.0, 0.0, 0.0],
            "position_z": [0.0, 0.0, 0.0],
            "earth_unit_vector_x": [1.0, 1.0, 1.0],
            "earth_unit_vector_y": [0.0, 0.0, 0.0],
            "earth_unit_vector_z": [0.0, 0.0, 0.0],
            "right_ascension": [0.0, 0.0, 0.0],
            "declination": [0.0, 0.0, 0.0],
            "rv1_coarse": [1.0, 2.0, 3.0],
            "rv2_coarse": [1.0, 2.0, None],
        }
    ).to_csv(path, index=False)
    clean, report = ingest_csv(path, IngestOptions(value_columns=("rv1_coarse", "rv2_coarse"), use_existing_loader=False))
    assert {"original_time", "antenna", "power", "quality_flags", "is_valid"}.issubset(clean.columns)
    assert clean["quality_flags"].str.contains("duplicate_timestamp").any()
    assert clean["quality_flags"].str.contains("malformed").any()
    assert not report.empty

