from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "PlateDetector",
    "PlateBox",
    "PlateDetections",
    "PlateSplitter",
    "PlateSplitResult",
    "ExtractedPlate",
    "gitter",
    "gitter_batch",
    "gitter_read",
    "plate_warnings",
    "plot_gitter",
    "summary_gitter",
]


if TYPE_CHECKING:
    from .core import gitter, gitter_batch
    from .io import gitter_read, plate_warnings, summary_gitter
    from .plate_detection import PlateDetector
    from .plate_splitter import PlateSplitter
    from .plate_types import ExtractedPlate, PlateBox, PlateDetections, PlateSplitResult
    from .plotting import plot_gitter


def __getattr__(name: str):
    if name in {"gitter", "gitter_batch"}:
        from .core import gitter, gitter_batch

        return {"gitter": gitter, "gitter_batch": gitter_batch}[name]
    if name in {"gitter_read", "plate_warnings", "summary_gitter"}:
        from .io import gitter_read, plate_warnings, summary_gitter

        return {
            "gitter_read": gitter_read,
            "plate_warnings": plate_warnings,
            "summary_gitter": summary_gitter,
        }[name]
    if name in {
        "PlateDetector",
        "PlateBox",
        "PlateDetections",
        "PlateSplitter",
        "PlateSplitResult",
        "ExtractedPlate",
    }:
        from .plate_detection import PlateDetector
        from .plate_splitter import PlateSplitter
        from .plate_types import ExtractedPlate, PlateBox, PlateDetections, PlateSplitResult

        return {
            "PlateDetector": PlateDetector,
            "PlateBox": PlateBox,
            "PlateDetections": PlateDetections,
            "PlateSplitter": PlateSplitter,
            "PlateSplitResult": PlateSplitResult,
            "ExtractedPlate": ExtractedPlate,
        }[name]
    if name == "plot_gitter":
        from .plotting import plot_gitter

        return plot_gitter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
