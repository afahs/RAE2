from __future__ import annotations

import pandas as pd

from rylevonberg.table_io import read_table, resolve_table_path, write_table


def test_read_table_falls_back_from_csv_to_npy(tmp_path):
    df = pd.DataFrame(
        {
            "time": ["1974-11-01 00:00:00", "1974-11-01 00:01:00"],
            "frequency_band": [1, 2],
            "power": [10.5, 11.5],
        }
    )
    npy_path = tmp_path / "large_table.npy"
    csv_path = tmp_path / "large_table.csv"
    write_table(df, npy_path)

    assert resolve_table_path(csv_path) == npy_path
    loaded = read_table(csv_path, parse_dates=["time"])

    assert list(loaded.columns) == list(df.columns)
    assert str(loaded["time"].dtype).startswith("datetime64")
    assert loaded["frequency_band"].tolist() == [1, 2]
    assert loaded["power"].tolist() == [10.5, 11.5]


def test_read_table_npy_usecols_and_chunks(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": ["x", "y", "z"]})
    path = tmp_path / "table.npy"
    write_table(df, path)

    loaded = read_table(path, usecols=["a", "c"])
    assert loaded.to_dict(orient="list") == {"a": [1, 2, 3], "c": ["x", "y", "z"]}

    chunks = list(read_table(path, chunksize=2))
    assert [len(chunk) for chunk in chunks] == [2, 1]
    assert chunks[0]["a"].tolist() == [1, 2]
