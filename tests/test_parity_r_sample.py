from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from gitter_py.core import gitter

BASE_IMAGE = Path("examples/extdata/sample.jpg")
BASE_DAT = Path("examples/extdata/sample.jpg.dat")


def _load_r_baseline() -> pd.DataFrame:
    baseline = pd.read_csv(
        BASE_DAT,
        sep="\t",
        comment="#",
        header=None,
        names=["row", "col", "size", "circularity", "flags"],
        na_values=["NA"],
    )
    baseline["row"] = pd.to_numeric(baseline["row"], errors="raise").astype(int)
    baseline["col"] = pd.to_numeric(baseline["col"], errors="raise").astype(int)
    baseline["size"] = pd.to_numeric(baseline["size"], errors="coerce")
    baseline["circularity"] = pd.to_numeric(baseline["circularity"], errors="coerce")
    baseline["flags"] = baseline["flags"].fillna("")
    return baseline


def _compute_metrics(py: pd.DataFrame, baseline: pd.DataFrame) -> dict[str, float]:
    merged = py.merge(
        baseline,
        on=["row", "col"],
        suffixes=("_py", "_r"),
        how="inner",
    )
    if len(merged) != len(baseline):
        raise AssertionError(
            "Failed to align all colonies by row/col: "
            f"matched={len(merged)}, expected={len(baseline)}"
        )

    size_err = (merged["size_py"] - merged["size_r"]).abs()
    circ_err = (merged["circularity_py"] - merged["circularity_r"]).abs()
    circ_non_na = merged[["circularity_py", "circularity_r"]].dropna()

    py_flagged = set(
        map(tuple, merged.loc[merged["flags_py"].fillna("") != "", ["row", "col"]].to_numpy())
    )
    r_flagged = set(
        map(tuple, merged.loc[merged["flags_r"].fillna("") != "", ["row", "col"]].to_numpy())
    )
    overlap = len(py_flagged & r_flagged)
    precision = overlap / len(py_flagged) if py_flagged else 1.0
    recall = overlap / len(r_flagged) if r_flagged else 1.0

    return {
        "n": float(len(merged)),
        "size_mae": float(size_err.mean()),
        "size_p99_abs_err": float(size_err.quantile(0.99)),
        "size_corr": float(merged["size_py"].corr(merged["size_r"])),
        "circularity_mae": float(circ_err.mean()),
        "circularity_p99_abs_err": float(circ_err.quantile(0.99)),
        "circularity_corr": float(circ_non_na["circularity_py"].corr(circ_non_na["circularity_r"])),
        "flag_precision": float(precision),
        "flag_recall": float(recall),
        "py_flagged_count": float(len(py_flagged)),
        "r_flagged_count": float(len(r_flagged)),
    }


@pytest.fixture(scope="session")
def parity_metrics() -> dict[str, float]:
    assert BASE_IMAGE.exists(), f"Missing sample image: {BASE_IMAGE}"
    assert BASE_DAT.exists(), f"Missing R baseline dat: {BASE_DAT}"

    py = gitter(
        str(BASE_IMAGE),
        plate_format=1536,
        verbose="n",
        grid_save=None,
        dat_save=None,
    )[["row", "col", "size", "circularity", "flags"]].copy()
    py["row"] = pd.to_numeric(py["row"], errors="raise").astype(int)
    py["col"] = pd.to_numeric(py["col"], errors="raise").astype(int)
    py["size"] = pd.to_numeric(py["size"], errors="coerce")
    py["circularity"] = pd.to_numeric(py["circularity"], errors="coerce")
    py["flags"] = py["flags"].fillna("")
    baseline = _load_r_baseline()
    return _compute_metrics(py=py, baseline=baseline)


def test_r_parity_shape_and_alignment(parity_metrics: dict[str, float]):
    assert parity_metrics["n"] == 1536.0


def test_r_parity_quantitative_metrics(parity_metrics: dict[str, float]):
    assert parity_metrics["size_mae"] <= 0.10, parity_metrics
    assert parity_metrics["size_p99_abs_err"] <= 1.0, parity_metrics
    assert parity_metrics["size_corr"] >= 0.999, parity_metrics
    assert parity_metrics["circularity_mae"] <= 0.01, parity_metrics
    assert parity_metrics["circularity_p99_abs_err"] <= 0.02, parity_metrics
    assert parity_metrics["circularity_corr"] >= 0.98, parity_metrics


def test_r_parity_flag_recall(parity_metrics: dict[str, float]):
    assert parity_metrics["flag_recall"] >= 0.99, parity_metrics


def test_r_parity_flag_precision_target(parity_metrics: dict[str, float]):
    assert parity_metrics["flag_precision"] >= 0.99, parity_metrics
