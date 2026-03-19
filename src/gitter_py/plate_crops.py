from __future__ import annotations

import numpy as np


def validate_rotate_override(rotate_override: int | None) -> int | None:
    if rotate_override is None:
        return None
    if rotate_override not in {0, 90, 180, 270}:
        raise ValueError("rotate_override must be one of: 0, 90, 180, 270")
    return int(rotate_override)


def rotate_by_degrees(image: np.ndarray, degrees: int) -> np.ndarray:
    if degrees % 90 != 0:
        raise ValueError("Rotation degrees must be a multiple of 90")
    k = (degrees // 90) % 4
    if k == 0:
        return image
    return np.rot90(image, k=k)


def orient_crop(
    crop: np.ndarray,
    rotate: bool,
    rotate_override: int | None,
) -> tuple[np.ndarray, int]:
    if rotate_override is not None:
        return rotate_by_degrees(crop, rotate_override), rotate_override
    if rotate and crop.shape[0] > crop.shape[1]:
        return np.rot90(crop, k=1), 90
    return crop, 0
