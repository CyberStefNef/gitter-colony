from __future__ import annotations

import pandas as pd

from gitter_py.io import gitter_read, gitter_write, plate_warnings


def test_dat_roundtrip(tmp_path):
    df = pd.DataFrame(
        {
            "row": [1, 1, 2],
            "col": [1, 2, 1],
            "size": [100, 0, 80],
            "circularity": [0.9, 0.2, 0.5],
            "flags": ["", "S,C", "C"],
        }
    )
    path = tmp_path / "sample.dat"
    gitter_write(df, path)
    loaded = gitter_read(path)
    assert list(loaded.columns) == ["row", "col", "size", "circularity", "flags"]
    assert loaded.shape == df.shape
    assert plate_warnings(loaded) is not None
