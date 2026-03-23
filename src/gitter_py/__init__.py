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
    "read_results_csv",
    "write_results_csv",
    "plate_warnings",
    "plot_results",
    "plot_gitter",
    "render_grid_overlay",
    "save_grid_overlay",
    "summary_gitter",
]


if TYPE_CHECKING:
    from .core import gitter, gitter_batch
    from .io import plate_warnings, read_results_csv, summary_gitter, write_results_csv
    from .plate_detection import PlateDetector
    from .plate_splitter import PlateSplitter
    from .plate_types import ExtractedPlate, PlateBox, PlateDetections, PlateSplitResult
    from .plotting import plot_gitter, plot_results, render_grid_overlay, save_grid_overlay


def __getattr__(name: str):
    if name in {"gitter", "gitter_batch"}:
        from .core import gitter, gitter_batch

        return {"gitter": gitter, "gitter_batch": gitter_batch}[name]
    if name in {"read_results_csv", "write_results_csv", "plate_warnings", "summary_gitter"}:
        from .io import plate_warnings, read_results_csv, summary_gitter, write_results_csv

        return {
            "read_results_csv": read_results_csv,
            "write_results_csv": write_results_csv,
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
    if name in {"plot_results", "plot_gitter", "render_grid_overlay", "save_grid_overlay"}:
        from .plotting import plot_gitter, plot_results, render_grid_overlay, save_grid_overlay

        return {
            "plot_results": plot_results,
            "plot_gitter": plot_gitter,
            "render_grid_overlay": render_grid_overlay,
            "save_grid_overlay": save_grid_overlay,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
