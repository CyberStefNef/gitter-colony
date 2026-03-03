"""Shared constants for the gitter Python port."""

from __future__ import annotations

GITTER_VERSION = "1.1.4"

PLATE_FORMATS: dict[str, tuple[int, int]] = {
    "1536": (32, 48),
    "768": (32, 48),
    "384": (16, 24),
    "96": (8, 12),
}

FLAG_MAP: dict[str, str] = {
    "1": "high count of small colony sizes",
    "2": "high count of low colony circularity",
}

WARNING_PAT = "# Warning possible misgridding: "
FLAGS_HELP = "# Flags: S - Colony spill or edge interference, C - Low colony circularity"
