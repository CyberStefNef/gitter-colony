from __future__ import annotations

from pathlib import Path

from gitter_py.core import gitter


def test_gitter_sample_image_smoke():
    sample = Path("examples/extdata/sample.jpg")
    assert sample.exists()
    df = gitter(
        str(sample),
        plate_format=1536,
        verbose="n",
        grid_save=None,
        dat_save=None,
        _auto_plate_detector=False,
    )
    assert {"row", "col", "size", "circularity", "flags"}.issubset(df.columns)
    assert len(df) == 32 * 48
