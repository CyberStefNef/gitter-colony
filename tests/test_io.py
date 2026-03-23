from __future__ import annotations

import pandas as pd

from gitter_py.io import plate_warnings, read_results_csv, write_results_csv


def test_csv_roundtrip(tmp_path):
    df = pd.DataFrame(
        {
            "row": [1, 1, 2],
            "col": [1, 2, 1],
            "size": [100, 0, 80],
            "circularity": [0.9, 0.2, 0.5],
            "flags": ["", "S,C", "C"],
        }
    )
    path = tmp_path / "sample.csv"
    write_results_csv(df, path)
    loaded = read_results_csv(path)
    assert list(loaded.columns) == ["row", "col", "size", "circularity", "flags"]
    assert loaded.shape == df.shape
    assert plate_warnings(loaded) is not None
