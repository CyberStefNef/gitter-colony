from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from skimage import io as skio
from skimage import util

YOLO_MODEL_KIND = "ultralytics_yolo_v1"
DEFAULT_MODEL_ROOT = Path("models/plate_detector_general/model/best.pt")


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


def _require_ultralytics():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "PlateDetector requires the 'ultralytics' package and its PyTorch runtime. "
            "Install the project dependencies before using the detector."
        ) from exc
    return YOLO


def _read_metadata(metadata_path: Path | None) -> dict[str, Any]:
    if metadata_path is None or not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _resolve_model_paths(model_bundle: str | Path) -> tuple[Path | None, Path]:
    path = Path(model_bundle)
    if path.is_dir():
        metadata_path = path / "metadata.json"
        if metadata_path.exists():
            metadata = _read_metadata(metadata_path)
            weights_name = str(metadata.get("weights_path", "best.pt"))
            weights_path = metadata_path.parent / weights_name
            return metadata_path, weights_path
        weights_path = path / "best.pt"
        if weights_path.exists():
            sibling_metadata = path / "metadata.json"
            return sibling_metadata if sibling_metadata.exists() else None, weights_path
        raise ValueError(f"No metadata.json or best.pt found under {path}")

    if path.suffix.lower() == ".json":
        metadata_path = path
        metadata = _read_metadata(metadata_path)
        weights_name = str(metadata.get("weights_path", "best.pt"))
        return metadata_path, metadata_path.parent / weights_name

    if path.suffix.lower() == ".pt":
        metadata_path = path.with_name("metadata.json")
        return metadata_path if metadata_path.exists() else None, path

    raise ValueError(
        "Unsupported detector model path: "
        f"{path}. Expected a model directory, metadata.json, or .pt weights."
    )


def _prepare_ultralytics_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        arr = arr[:, :, :3]
    else:
        raise ValueError(f"Unsupported image shape for detection: {arr.shape}")

    if np.issubdtype(arr.dtype, np.floating):
        max_value = float(np.nanmax(arr)) if arr.size else 1.0
        if max_value <= 1.5:
            arr = np.clip(arr, 0.0, 1.0) * 255.0
        else:
            arr = np.clip(arr, 0.0, 255.0)
    else:
        arr = np.clip(arr, 0, 255)

    return np.ascontiguousarray(arr.astype(np.uint8))


def _as_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([])
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def _resolve_label(names: Any, cls_id: float | int | None) -> str:
    if cls_id is None:
        return "plate"
    idx = int(round(float(cls_id)))
    if isinstance(names, dict):
        return str(names.get(idx, idx))
    if isinstance(names, (list, tuple)) and 0 <= idx < len(names):
        return str(names[idx])
    return "plate" if idx == 0 else str(idx)


def _to_gray_u8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3:
        arr = arr[:, :, :3].mean(axis=2)
    arr = np.nan_to_num(arr, nan=1.0, posinf=1.0, neginf=0.0)
    if np.issubdtype(arr.dtype, np.floating):
        max_value = float(np.nanmax(arr)) if arr.size else 1.0
        if max_value <= 1.5:
            arr = np.clip(arr, 0.0, 1.0) * 255.0
        else:
            arr = np.clip(arr, 0.0, 255.0)
    else:
        arr = np.clip(arr, 0, 255)
    return np.ascontiguousarray(arr.astype(np.uint8))


def _expand_bounds(
    x_min: int,
    x_max: int,
    y_min: int,
    y_max: int,
    width: int,
    height: int,
    fraction: float = 0.08,
) -> tuple[int, int, int, int]:
    pad_x = int(round((x_max - x_min + 1) * fraction))
    pad_y = int(round((y_max - y_min + 1) * fraction))
    return (
        max(0, x_min - pad_x),
        min(width - 1, x_max + pad_x),
        max(0, y_min - pad_y),
        min(height - 1, y_max + pad_y),
    )


def _polygon_bounds(
    polygon: np.ndarray,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    x_min = int(np.clip(np.floor(float(polygon[:, 0].min())), 0, max(image_width - 1, 0)))
    x_max = int(np.clip(np.ceil(float(polygon[:, 0].max())), 0, max(image_width - 1, 0)))
    y_min = int(np.clip(np.floor(float(polygon[:, 1].min())), 0, max(image_height - 1, 0)))
    y_max = int(np.clip(np.ceil(float(polygon[:, 1].max())), 0, max(image_height - 1, 0)))
    return x_min, x_max, y_min, y_max


def _bbox_iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    x_min = max(a[0], b[0])
    x_max = min(a[1], b[1])
    y_min = max(a[2], b[2])
    y_max = min(a[3], b[3])
    if x_max <= x_min or y_max <= y_min:
        return 0.0
    intersection = float((x_max - x_min) * (y_max - y_min))
    a_area = float(max(0.0, a[1] - a[0]) * max(0.0, a[3] - a[2]))
    b_area = float(max(0.0, b[1] - b[0]) * max(0.0, b[3] - b[2]))
    union = a_area + b_area - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def _refine_plate_polygon(
    image: np.ndarray,
    *,
    x_min: int,
    x_max: int,
    y_min: int,
    y_max: int,
) -> tuple[tuple[float, float], ...] | None:
    image_height, image_width = image.shape[:2]
    roi_x_min, roi_x_max, roi_y_min, roi_y_max = _expand_bounds(
        x_min, x_max, y_min, y_max, image_width, image_height
    )
    gray = _to_gray_u8(image)
    roi = gray[roi_y_min : roi_y_max + 1, roi_x_min : roi_x_max + 1]
    if roi.size == 0:
        return None
    if float(np.std(roi)) < 2.0:
        return None

    roi_h, roi_w = roi.shape[:2]
    roi_area = float(roi_h * roi_w)
    target_bounds = (float(x_min), float(x_max), float(y_min), float(y_max))
    target_area = float(max(1, (x_max - x_min + 1) * (y_max - y_min + 1)))
    target_center = np.array([(x_min + x_max) / 2.0, (y_min + y_max) / 2.0], dtype=np.float32)
    target_diag = float(np.hypot(x_max - x_min + 1, y_max - y_min + 1))

    blur = cv2.GaussianBlur(roi, (5, 5), 0)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    edges = cv2.Canny(blur, 40, 140)
    edges = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=1)

    best_polygon: np.ndarray | None = None
    best_score = float("-inf")
    masks = (thresh, edges)
    for mask in masks:
        contours_info = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]
        for contour in contours:
            contour_area = float(cv2.contourArea(contour))
            if contour_area < roi_area * 0.03:
                continue
            rect = cv2.minAreaRect(contour)
            (_, _), (rect_w, rect_h), _ = rect
            if rect_w < 5.0 or rect_h < 5.0:
                continue
            rect_area = float(rect_w * rect_h)
            if rect_area < roi_area * 0.08:
                continue
            aspect_ratio = max(rect_w, rect_h) / max(min(rect_w, rect_h), 1e-6)
            if aspect_ratio > 2.4:
                continue

            local_polygon = cv2.boxPoints(rect).astype(np.float32)
            global_polygon = local_polygon + np.array([roi_x_min, roi_y_min], dtype=np.float32)
            cand_x_min, cand_x_max, cand_y_min, cand_y_max = _polygon_bounds(
                global_polygon,
                image_width,
                image_height,
            )
            cand_bounds = (
                float(cand_x_min),
                float(cand_x_max),
                float(cand_y_min),
                float(cand_y_max),
            )
            iou = _bbox_iou(cand_bounds, target_bounds)
            center = np.array(
                [(cand_x_min + cand_x_max) / 2.0, (cand_y_min + cand_y_max) / 2.0],
                dtype=np.float32,
            )
            center_distance = float(np.linalg.norm(center - target_center) / max(target_diag, 1.0))
            fill_ratio = contour_area / max(rect_area, 1.0)
            size_penalty = abs(np.log(max(rect_area, 1.0) / target_area))
            score = (
                (2.2 * iou)
                + (0.45 * fill_ratio)
                - (0.35 * center_distance)
                - (0.2 * size_penalty)
            )
            if score > best_score:
                best_score = score
                best_polygon = global_polygon

    if best_polygon is None or best_score < 0.45:
        return None

    clipped = best_polygon.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0, max(image_width - 1, 0))
    clipped[:, 1] = np.clip(clipped[:, 1], 0, max(image_height - 1, 0))
    return tuple((float(x), float(y)) for x, y in clipped)


def _white_fill_value(image: np.ndarray) -> float | int:
    arr = np.asarray(image)
    if np.issubdtype(arr.dtype, np.floating):
        max_value = float(np.nanmax(arr)) if arr.size else 1.0
        return 1.0 if max_value <= 1.5 else 255.0
    return 255


def _order_quadrilateral(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if pts.shape != (4, 2):
        raise ValueError("Expected exactly four points for quadrilateral ordering")

    point_sums = pts.sum(axis=1)
    point_diffs = np.diff(pts, axis=1).reshape(-1)

    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(point_sums)]   # top-left
    ordered[2] = pts[np.argmax(point_sums)]   # bottom-right
    ordered[1] = pts[np.argmin(point_diffs)]  # top-right
    ordered[3] = pts[np.argmax(point_diffs)]  # bottom-left
    return ordered


def _warp_plate_polygon_crop(
    source_image: np.ndarray,
    polygon: np.ndarray,
) -> np.ndarray | None:
    ordered = _order_quadrilateral(polygon)

    width_top = float(np.linalg.norm(ordered[1] - ordered[0]))
    width_bottom = float(np.linalg.norm(ordered[2] - ordered[3]))
    height_right = float(np.linalg.norm(ordered[2] - ordered[1]))
    height_left = float(np.linalg.norm(ordered[3] - ordered[0]))

    target_width = max(1, int(round(max(width_top, width_bottom))) + 1)
    target_height = max(1, int(round(max(height_left, height_right))) + 1)
    if target_width < 2 or target_height < 2:
        return None

    destination = np.array(
        [
            [0.0, 0.0],
            [float(target_width - 1), 0.0],
            [float(target_width - 1), float(target_height - 1)],
            [0.0, float(target_height - 1)],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(ordered, destination)
    warped = cv2.warpPerspective(
        np.ascontiguousarray(source_image),
        transform,
        (target_width, target_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    if warped.size == 0:
        return None
    return warped


def _to_saveable_u8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
    if np.issubdtype(arr.dtype, np.floating):
        max_value = float(np.nanmax(arr)) if arr.size else 1.0
        if max_value <= 1.5:
            return util.img_as_ubyte(np.clip(arr, 0.0, 1.0))
        return np.clip(arr, 0.0, 255.0).astype(np.uint8)
    return np.clip(arr, 0, 255).astype(np.uint8)


def extract_plate_crop(
    source_image: np.ndarray,
    box: PlateBox,
    *,
    mask_outside_polygon: bool = True,
) -> np.ndarray:
    image = np.asarray(source_image)
    height, width = image.shape[:2]
    x_min = int(np.clip(box.x_min, 0, max(width - 1, 0)))
    x_max = int(np.clip(box.x_max, 0, max(width - 1, 0)))
    y_min = int(np.clip(box.y_min, 0, max(height - 1, 0)))
    y_max = int(np.clip(box.y_max, 0, max(height - 1, 0)))
    crop = image[y_min : y_max + 1, x_min : x_max + 1].copy()
    if crop.size == 0 or not mask_outside_polygon or box.polygon is None:
        return crop

    polygon = np.asarray(box.polygon, dtype=np.float32).copy()
    if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2:
        return crop

    if polygon.shape[0] == 4:
        warped = _warp_plate_polygon_crop(image, polygon)
        if warped is not None:
            return warped

    local_polygon = polygon
    local_polygon[:, 0] -= x_min
    local_polygon[:, 1] -= y_min

    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(local_polygon).astype(np.int32)], 255)
    fill_value = _white_fill_value(crop)
    crop[mask == 0] = fill_value
    return crop


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
        metadata_path, weights_path = _resolve_model_paths(bundle_path)
        metadata = _read_metadata(metadata_path)

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
        model_input = _prepare_ultralytics_image(image)
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
        xyxy = _as_numpy(getattr(boxes_obj, "xyxy", None)).reshape(-1, 4)
        conf = _as_numpy(getattr(boxes_obj, "conf", None)).reshape(-1)
        cls = _as_numpy(getattr(boxes_obj, "cls", None)).reshape(-1)
        names = (
            getattr(result, "names", None)
            or getattr(self.model, "names", None)
            or self.class_names
        )

        src_h, src_w = image.shape[:2]
        scored_boxes: list[PlateBox] = []
        for idx, coords in enumerate(xyxy):
            label = _resolve_label(names, cls[idx] if idx < len(cls) else 0)
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
            polygon = _refine_plate_polygon(
                image,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
            )
            if polygon is not None:
                x_min, x_max, y_min, y_max = _polygon_bounds(
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
        skio.imsave(crop_path, _to_saveable_u8(crop))
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
