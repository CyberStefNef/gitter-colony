from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from skimage import io as skio

from gitter_py import PlateBox, PlateDetections, PlateSplitter, gitter


class _FakeDetector:
    detector_version = "test"

    def detect(self, image: np.ndarray) -> PlateDetections:
        return PlateDetections(
            boxes=[
                PlateBox(
                    x_min=10,
                    x_max=69,
                    y_min=20,
                    y_max=89,
                    confidence=0.99,
                    polygon=((10.0, 20.0), (69.0, 20.0), (69.0, 89.0), (10.0, 89.0)),
                ),
                PlateBox(
                    x_min=80,
                    x_max=89,
                    y_min=10,
                    y_max=19,
                    confidence=0.40,
                    polygon=((80.0, 10.0), (89.0, 10.0), (89.0, 19.0), (80.0, 19.0)),
                ),
            ],
            source_image=image,
            source_rotation_degrees=0,
            overall_confidence=0.99,
            detector_name="FakeDetector",
            detector_version="test",
        )


def test_plate_splitter_filters_without_rotation():
    image = np.zeros((120, 100), dtype=float)
    image[20:90, 10:70] = 1.0

    splitter = PlateSplitter(detector=_FakeDetector(), min_confidence=0.95)
    result = splitter.split(image)

    assert result.source_name == "image"
    assert result.detector_name == "FakeDetector"
    assert len(result.plates) == 1
    plate = result.plates[0]
    assert plate.plate_index == 1
    assert plate.confidence == 0.99
    assert plate.crop.shape == (70, 60)
    assert plate.crop_rotation_degrees == 0


def test_plate_splitter_save_writes_expected_artifacts(tmp_path):
    image = np.zeros((120, 100), dtype=float)
    image[20:90, 10:70] = 1.0

    splitter = PlateSplitter(detector=_FakeDetector(), min_confidence=0.95)
    result = splitter.split(image)
    splitter.save(result, tmp_path)

    layout_path = tmp_path / "image__plate_layout.png"
    split_path = tmp_path / "image__split.json"
    crop_path = tmp_path / "image__plate_01.tiff"

    assert layout_path.exists()
    assert split_path.exists()
    assert crop_path.exists()

    payload = json.loads(split_path.read_text(encoding="utf-8"))
    assert payload["detected_plate_count"] == 1
    assert payload["plates"][0]["crop_file"] == crop_path.name


def test_gitter_accepts_split_crop_arrays_directly():
    sample = Path("examples/extdata/sample.jpg")
    image = skio.imread(sample)

    df = gitter(
        image,
        plate_format=1536,
        verbose="n",
        grid_save=None,
        dat_save=None,
    )

    assert len(df) == 1536
    assert df.attrs["file"] == "image"
