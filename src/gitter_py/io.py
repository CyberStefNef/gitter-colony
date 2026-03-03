from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import FLAG_MAP, FLAGS_HELP, GITTER_VERSION, WARNING_PAT


def get_flags(df: pd.DataFrame) -> list[str]:
    size_median = float(np.nanmedian(df["size"].to_numpy()))
    if size_median == 0:
        rel_size = df["size"].to_numpy(dtype=float)
    else:
        rel_size = df["size"].to_numpy(dtype=float) / size_median

    circ = df["circularity"].to_numpy(dtype=float)
    n = max(len(df), 1)
    flags: list[str] = []
    if (np.sum(rel_size < 0.1) / n) > 0.1:
        flags.append("1")
    if (np.sum(np.isnan(circ) | (circ < 0.6)) / n) > 0.1:
        flags.append("2")
    return flags


def _now_stamp() -> str:
    return datetime.now().strftime("%a %b %d %H:%M:%S %Y")


def gitter_write(df: pd.DataFrame, path: str | Path) -> pd.DataFrame:
    target = Path(path)
    data = df.copy()
    header = [f"# gitter v{GITTER_VERSION} data file generated on {_now_stamp()}"]

    ids = [i for i in get_flags(data) if i in FLAG_MAP]
    warnings = [FLAG_MAP[i] for i in ids]
    if warnings:
        data.attrs["warnings"] = warnings
        header.append(f"{WARNING_PAT}{', '.join(warnings)}")
    header.append(FLAGS_HELP)

    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        for line in header:
            f.write(f"{line}\n")
        f.write("# row\tcol\tsize\tcircularity\tflags\n")
        data.to_csv(f, sep="\t", index=False, header=False, na_rep="NA")
    return data


def _read_warnings(path: Path, probe_lines: int = 5) -> list[str]:
    warnings: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for _ in range(probe_lines):
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if line.startswith(WARNING_PAT):
                raw = line.replace(WARNING_PAT, "", 1)
                parts = [p.strip() for p in raw.split(",")]
                warnings = [p for p in parts if p in FLAG_MAP.values()]
                break
    return warnings


def gitter_read(path_or_df: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(path_or_df, pd.DataFrame):
        out = path_or_df.copy()
        required = ["row", "col", "size", "circularity", "flags"]
        if not all(c in out.columns for c in required):
            raise ValueError("DataFrame must contain row, col, size, circularity and flags columns")
        return out

    path = Path(path_or_df)
    if not path.exists():
        raise FileNotFoundError(path)

    dat = pd.read_csv(path, sep="\t", comment="#", header=None, dtype=str)
    if dat.shape[1] != 5:
        raise ValueError(f"Expected 5 columns in dat file, found {dat.shape[1]}")
    dat.columns = ["row", "col", "size", "circularity", "flags"]
    dat["row"] = pd.to_numeric(dat["row"], errors="coerce").astype("Int64")
    dat["col"] = pd.to_numeric(dat["col"], errors="coerce").astype("Int64")
    dat["size"] = pd.to_numeric(dat["size"], errors="coerce")
    dat["circularity"] = pd.to_numeric(dat["circularity"], errors="coerce")
    dat["flags"] = dat["flags"].fillna("")
    dat.attrs["warnings"] = _read_warnings(path)
    return dat


def plate_warnings(df: pd.DataFrame) -> list[str] | None:
    warnings = df.attrs.get("warnings")
    if warnings is None:
        return None
    return list(warnings)


def summary_gitter(df: pd.DataFrame) -> str:
    plate_format = df.attrs.get("format")
    call = df.attrs.get("call", "not available")
    elapsed = df.attrs.get("elapsed")
    lines: list[str] = [f"# gitter v{GITTER_VERSION} data file #"]
    lines.append(f"Function call: {call}")
    lines.append(f"Elapsed time: {elapsed} secs")
    if plate_format is not None:
        plate_total = int(plate_format[0] * plate_format[1])
        lines.append(
            f"Plate format: {plate_format[0]} x {plate_format[1]} ({plate_total})"
        )
    lines.append("Colony size statistics:")
    size_stats = df["size"].describe(percentiles=[0.25, 0.5, 0.75]).to_string()
    lines.append(size_stats)
    lines.append("Dat file (first 6 rows):")
    lines.append(df.head(6).to_string(index=False))
    warnings = plate_warnings(df)
    if warnings:
        lines.append("Plate warnings:")
        lines.append(", ".join(warnings))
    return "\n".join(lines)
