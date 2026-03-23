from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .constants import FLAG_MAP

RESULT_COLUMNS = ["row", "col", "size", "circularity", "flags"]


def _ensure_result_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    missing = [col for col in RESULT_COLUMNS if col not in out.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {', '.join(missing)}")
    return out


def get_flags(df: pd.DataFrame) -> list[str]:
    data = _ensure_result_columns(df)
    size_median = float(np.nanmedian(data["size"].to_numpy()))
    if size_median == 0:
        rel_size = data["size"].to_numpy(dtype=float)
    else:
        rel_size = data["size"].to_numpy(dtype=float) / size_median

    circ = data["circularity"].to_numpy(dtype=float)
    n = max(len(data), 1)
    flags: list[str] = []
    if (np.sum(rel_size < 0.1) / n) > 0.1:
        flags.append("1")
    if (np.sum(np.isnan(circ) | (circ < 0.6)) / n) > 0.1:
        flags.append("2")
    return flags


def write_results_csv(df: pd.DataFrame, path: str | Path) -> pd.DataFrame:
    target = Path(path)
    data = _ensure_result_columns(df)[RESULT_COLUMNS].copy()
    target.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(target, index=False)
    return data


def read_results_csv(path_or_df: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(path_or_df, pd.DataFrame):
        return _ensure_result_columns(path_or_df)

    path = Path(path_or_df)
    if not path.exists():
        raise FileNotFoundError(path)

    data = pd.read_csv(path)
    return _ensure_result_columns(data)


def plate_warnings(df: pd.DataFrame) -> list[str] | None:
    ids = [flag_id for flag_id in get_flags(df) if flag_id in FLAG_MAP]
    if not ids:
        return None
    return [FLAG_MAP[flag_id] for flag_id in ids]


def summary_gitter(df: pd.DataFrame) -> str:
    data = _ensure_result_columns(df)
    lines: list[str] = ["# gitter results #"]
    lines.append("Colony size statistics:")
    size_stats = data["size"].describe(percentiles=[0.25, 0.5, 0.75]).to_string()
    lines.append(size_stats)
    lines.append("Results (first 6 rows):")
    lines.append(data[RESULT_COLUMNS].head(6).to_string(index=False))
    warnings = plate_warnings(data)
    if warnings:
        lines.append("Plate warnings:")
        lines.append(", ".join(warnings))
    return "\n".join(lines)
