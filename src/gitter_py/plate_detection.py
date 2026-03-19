from __future__ import annotations

from pathlib import Path

import numpy as np

from .plate_detection_utils import (
    DEFAULT_MODEL_ROOT,
    YOLO_MODEL_KIND,
    as_numpy,
    polygon_bounds,
    prepare_ultralytics_image,
    read_metadata,
    refine_plate_polygon,
    resolve_label,
    resolve_model_paths,
)
from .plate_types import PlateBox, PlateDetections


def _require_ultralytics():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "PlateDetector requires the 'ultralytics' package and its PyTorch runtime. "
            "Install the project dependencies before using the detector."
        ) from exc
    return YOLO


class PlateDetector:
    """YOLO-backed plate detector with detector-internal preprocessing."""

    def __init__(
        self,
        model_bundle: str | Path = DEFAULT_MODEL_ROOT,
        *,
        confidence_threshold: float | None = None,
        nms_iou_threshold: float | None = None,
        max_plates: int | None = None,
        image_size: int | None = None,
    ) -> None:
        bundle_path = Path(model_bundle)
        metadata_path, weights_path = resolve_model_paths(bundle_path)
        metadata = read_metadata(metadata_path)

        model_kind = str(metadata.get("model_kind", YOLO_MODEL_KIND))
        if model_kind != YOLO_MODEL_KIND:
            if bundle_path.suffix.lower() == ".pt":
                metadata = {}
                metadata_path = None
                model_kind = YOLO_MODEL_KIND
            else:
                raise ValueError(
                    "Unsupported detector bundle "
                    f"model_kind={model_kind!r}; expected {YOLO_MODEL_KIND!r}"
                )
        if not weights_path.exists():
            raise ValueError(f"Detector weights do not exist: {weights_path}")

        YOLO = _require_ultralytics()
        self.model = YOLO(str(weights_path))
        self.bundle_path = bundle_path
        self.metadata_path = metadata_path
        self.weights_path = weights_path
        self.model_kind = model_kind
        self.input_size = int(image_size or metadata.get("input_size", 1024))
        self.confidence_threshold = float(
            confidence_threshold
            if confidence_threshold is not None
            else metadata.get("confidence_threshold", 0.25)
        )
        self.nms_iou_threshold = float(
            nms_iou_threshold
            if nms_iou_threshold is not None
            else metadata.get("nms_iou_threshold", 0.45)
        )
        self.max_plates = int(max_plates or metadata.get("max_plates", 4))
        self.detector_version = str(metadata.get("detector_version", "v1"))
        self.class_names = metadata.get("class_names", ["plate"])

    def detect(self, image: np.ndarray) -> PlateDetections:
        model_input = prepare_ultralytics_image(image)
        results = self.model.predict(
            source=model_input,
            conf=self.confidence_threshold,
            iou=self.nms_iou_threshold,
            imgsz=self.input_size,
            max_det=self.max_plates,
            verbose=False,
        )
        result = results[0] if results else None
        if result is None or getattr(result, "boxes", None) is None:
            return PlateDetections(
                boxes=[],
                source_image=image,
                source_rotation_degrees=0,
                overall_confidence=0.0,
                detector_name=type(self).__name__,
                detector_version=self.detector_version,
            )

        boxes_obj = result.boxes
        xyxy = as_numpy(getattr(boxes_obj, "xyxy", None)).reshape(-1, 4)
        conf = as_numpy(getattr(boxes_obj, "conf", None)).reshape(-1)
        cls = as_numpy(getattr(boxes_obj, "cls", None)).reshape(-1)
        names = (
            getattr(result, "names", None)
            or getattr(self.model, "names", None)
            or self.class_names
        )

        src_h, src_w = image.shape[:2]
        scored_boxes: list[PlateBox] = []
        for idx, coords in enumerate(xyxy):
            label = resolve_label(names, cls[idx] if idx < len(cls) else 0)
            if label != "plate":
                continue
            score = float(conf[idx]) if idx < len(conf) else 1.0
            x_min = int(np.floor(float(coords[0])))
            y_min = int(np.floor(float(coords[1])))
            x_max = int(np.ceil(float(coords[2])))
            y_max = int(np.ceil(float(coords[3])))
            x_min = int(np.clip(x_min, 0, max(src_w - 1, 0)))
            x_max = int(np.clip(x_max, 0, max(src_w - 1, 0)))
            y_min = int(np.clip(y_min, 0, max(src_h - 1, 0)))
            y_max = int(np.clip(y_max, 0, max(src_h - 1, 0)))
            if x_max <= x_min or y_max <= y_min:
                continue
            polygon = refine_plate_polygon(
                image,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
            )
            if polygon is not None:
                x_min, x_max, y_min, y_max = polygon_bounds(
                    np.asarray(polygon, dtype=np.float32),
                    src_w,
                    src_h,
                )
            scored_boxes.append(
                PlateBox(
                    x_min=x_min,
                    x_max=x_max,
                    y_min=y_min,
                    y_max=y_max,
                    confidence=score,
                    label=label,
                    polygon=polygon,
                )
            )

        scored_boxes = sorted(
            scored_boxes,
            key=lambda box: box.confidence,
            reverse=True,
        )[: self.max_plates]
        ordered_boxes = sorted(scored_boxes, key=lambda box: (box.y_min, box.x_min))
        overall_confidence = max((box.confidence for box in ordered_boxes), default=0.0)
        return PlateDetections(
            boxes=ordered_boxes,
            source_image=image,
            source_rotation_degrees=0,
            overall_confidence=float(overall_confidence),
            detector_name=type(self).__name__,
            detector_version=self.detector_version,
        )
