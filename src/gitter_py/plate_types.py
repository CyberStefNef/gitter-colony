from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlateBox:
    x_min: int
    x_max: int
    y_min: int
    y_max: int
    confidence: float
    label: str = "plate"
    polygon: tuple[tuple[float, float], ...] | None = None


@dataclass(frozen=True)
class PlateDetections:
    boxes: list[PlateBox]
    source_image: np.ndarray
    source_rotation_degrees: int
    overall_confidence: float
    detector_name: str
    detector_version: str | None = None


@dataclass(frozen=True)
class ExtractedPlate:
    plate_index: int
    confidence: float
    label: str
    bbox: tuple[int, int, int, int]
    polygon: tuple[tuple[float, float], ...] | None
    crop: np.ndarray
    crop_rotation_degrees: int


@dataclass(frozen=True)
class PlateSplitResult:
    source_name: str
    source_image: np.ndarray
    source_rotation_degrees: int
    detector_name: str
    detector_version: str | None
    overall_confidence: float
    plates: list[ExtractedPlate]
