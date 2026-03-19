from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from skimage import io as skio

from .plate_detection import PlateDetector
from .plate_detection_utils import (
    DEFAULT_MODEL_ROOT,
    clip_plate_polygon,
    extract_plate_crop,
    read_source_image,
    to_saveable_u8,
)
from .plate_types import ExtractedPlate, PlateBox, PlateSplitResult


class PlateSplitter:
    """Explicit multi-plate splitting workflow built on top of PlateDetector."""

    def __init__(
        self,
        detector: PlateDetector | None = None,
        *,
        model_bundle: str | Path = DEFAULT_MODEL_ROOT,
        min_confidence: float = 0.95,
    ) -> None:
        if min_confidence < 0.0 or min_confidence > 1.0:
            raise ValueError("min_confidence must be between 0.0 and 1.0")
        self.detector = detector if detector is not None else PlateDetector(model_bundle)
        self.min_confidence = float(min_confidence)

    def split(self, image: str | Path | np.ndarray) -> PlateSplitResult:
        source_name, source_image = read_source_image(image)
        detections = self.detector.detect(source_image)
        height, width = detections.source_image.shape[:2]

        plates: list[ExtractedPlate] = []
        for box in detections.boxes:
            x_min = int(np.clip(box.x_min, 0, max(width - 1, 0)))
            x_max = int(np.clip(box.x_max, 0, max(width - 1, 0)))
            y_min = int(np.clip(box.y_min, 0, max(height - 1, 0)))
            y_max = int(np.clip(box.y_max, 0, max(height - 1, 0)))
            if x_max <= x_min or y_max <= y_min:
                continue
            if float(box.confidence) < self.min_confidence:
                continue

            clipped_polygon = clip_plate_polygon(box.polygon, width=width, height=height)
            extracted = extract_plate_crop(
                detections.source_image,
                PlateBox(
                    x_min=x_min,
                    x_max=x_max,
                    y_min=y_min,
                    y_max=y_max,
                    confidence=float(box.confidence),
                    label=box.label,
                    polygon=clipped_polygon,
                ),
            )
            if extracted.size == 0:
                continue
            plates.append(
                ExtractedPlate(
                    plate_index=len(plates) + 1,
                    confidence=float(box.confidence),
                    label=box.label,
                    bbox=(x_min, x_max, y_min, y_max),
                    polygon=clipped_polygon,
                    crop=extracted,
                    crop_rotation_degrees=0,
                )
            )

        if not plates:
            raise ValueError(
                "No plate regions met the extraction confidence threshold "
                f"({self.min_confidence:.3f})"
            )

        return PlateSplitResult(
            source_name=source_name,
            source_image=detections.source_image,
            source_rotation_degrees=detections.source_rotation_degrees,
            detector_name=detections.detector_name,
            detector_version=detections.detector_version,
            overall_confidence=float(
                max((plate.confidence for plate in plates), default=detections.overall_confidence)
            ),
            plates=plates,
        )

    def save(self, result: PlateSplitResult, output_dir: str | Path) -> None:
        save_split_artifacts(
            output_dir=output_dir,
            source_name=result.source_name,
            source_image=result.source_image,
            source_rotation_degrees=result.source_rotation_degrees,
            detector_name=result.detector_name,
            detector_version=result.detector_version,
            overall_confidence=result.overall_confidence,
            boxes=[
                PlateBox(
                    x_min=plate.bbox[0],
                    x_max=plate.bbox[1],
                    y_min=plate.bbox[2],
                    y_max=plate.bbox[3],
                    confidence=plate.confidence,
                    label=plate.label,
                    polygon=plate.polygon,
                )
                for plate in result.plates
            ],
            crops=[plate.crop for plate in result.plates],
            crop_rotations=[plate.crop_rotation_degrees for plate in result.plates],
        )


def save_split_artifacts(
    *,
    output_dir: str | Path,
    source_name: str,
    source_image: np.ndarray,
    source_rotation_degrees: int,
    detector_name: str,
    detector_version: str | None,
    overall_confidence: float,
    boxes: list[PlateBox],
    crops: list[np.ndarray | None],
    crop_rotations: list[int | None],
) -> None:
    from matplotlib import pyplot as plt
    from matplotlib.patches import Polygon, Rectangle

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not (len(boxes) == len(crops) == len(crop_rotations)):
        raise ValueError("boxes, crops, and crop_rotations must have the same length")

    crop_files: list[str | None] = []
    for idx, crop in enumerate(crops, start=1):
        if crop is None:
            crop_files.append(None)
            continue
        crop_name = f"{source_name}__plate_{idx:02d}.tiff"
        crop_path = out_dir / crop_name
        skio.imsave(crop_path, to_saveable_u8(crop))
        crop_files.append(crop_name)

    fig, ax = plt.subplots(figsize=(12, 8))
    if source_image.ndim == 2:
        ax.imshow(source_image, cmap="gray")
    else:
        ax.imshow(source_image)
    for idx, box in enumerate(boxes, start=1):
        if box.polygon is not None:
            polygon = np.asarray(box.polygon, dtype=np.float32)
            ax.add_patch(
                Polygon(
                    polygon,
                    closed=True,
                    linewidth=3.0,
                    edgecolor="#ff9a00",
                    facecolor="none",
                )
            )
            label_x = float(np.min(polygon[:, 0]))
            label_y = float(np.min(polygon[:, 1]))
        else:
            width = box.x_max - box.x_min + 1
            height = box.y_max - box.y_min + 1
            ax.add_patch(
                Rectangle(
                    (box.x_min, box.y_min),
                    width,
                    height,
                    linewidth=3.0,
                    edgecolor="#ff9a00",
                    facecolor="none",
                )
            )
            label_x = float(box.x_min)
            label_y = float(box.y_min)
        ax.text(
            label_x + 8,
            label_y + 26,
            f"{idx} ({box.confidence:.2f})",
            color="black",
            fontsize=13,
            bbox={"facecolor": "#ffd44d", "edgecolor": "#111", "pad": 2},
        )
    ax.set_axis_off()
    overlay_path = out_dir / f"{source_name}__plate_layout.png"
    fig.savefig(overlay_path, dpi=180, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    payload = {
        "source_name": source_name,
        "source_rotation_degrees": int(source_rotation_degrees),
        "detector_name": detector_name,
        "detector_version": detector_version,
        "overall_confidence": float(overall_confidence),
        "detected_plate_count": len(boxes),
        "plates": [
            {
                "plate_index": idx,
                "label": box.label,
                "confidence": float(box.confidence),
                "bbox": {
                    "x_min": int(box.x_min),
                    "x_max": int(box.x_max),
                    "y_min": int(box.y_min),
                    "y_max": int(box.y_max),
                },
                "polygon": (
                    [{"x": float(x), "y": float(y)} for x, y in box.polygon]
                    if box.polygon is not None
                    else None
                ),
                "crop_rotation_degrees": (
                    int(crop_rotation) if crop_rotation is not None else None
                ),
                "crop_file": crop_file,
            }
            for idx, (box, crop_rotation, crop_file) in enumerate(
                zip(boxes, crop_rotations, crop_files, strict=True),
                start=1,
            )
        ],
    }
    metadata_path = out_dir / f"{source_name}__split.json"
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
