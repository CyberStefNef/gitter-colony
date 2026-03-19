from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from gitter_py.core import gitter

R_PARITY_IMAGE = os.environ.get("GITTER_R_PARITY_IMAGE", "gitter-r-parity:4.3.3")
R_SOURCE_REPO = os.environ.get("GITTER_R_SOURCE_REPO", "https://github.com/omarwagih/gitter.git")
R_SOURCE_REF = os.environ.get("GITTER_R_SOURCE_REF", "master")
R_SOURCE_DIR = os.environ.get("GITTER_R_SOURCE_DIR")


def _compute_metrics(py: pd.DataFrame, r_df: pd.DataFrame) -> dict[str, float]:
    merged = py.merge(
        r_df,
        on=["row", "col"],
        suffixes=("_py", "_r"),
        how="inner",
    )
    if len(merged) != len(r_df):
        raise AssertionError(
            "Failed to align all colonies by row/col: "
            f"matched={len(merged)}, expected={len(r_df)}"
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
    exact_flag_match = float(
        (merged["flags_py"].fillna("") == merged["flags_r"].fillna("")).mean()
    )

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
        "flag_exact_match": exact_flag_match,
        "py_flagged_count": float(len(py_flagged)),
        "r_flagged_count": float(len(r_flagged)),
    }


def _docker_ready() -> bool:
    docker = shutil.which("docker")
    if docker is None:
        return False
    try:
        proc = subprocess.run(
            [docker, "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


def _docker_image_available(image: str) -> bool:
    try:
        proc = subprocess.run(
            ["docker", "image", "inspect", image],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


def _has_required_r_sources(root: Path) -> bool:
    return all((root / "R" / name).exists() for name in ("Peaks.R", "Help.R", "Main.R"))


def _prepare_r_sources(repo_root: Path, scratch_dir: Path) -> Path:
    local_r_gitter = repo_root / "r_gitter"
    if _has_required_r_sources(local_r_gitter):
        return local_r_gitter

    if R_SOURCE_DIR:
        override = Path(R_SOURCE_DIR).expanduser().resolve()
        if _has_required_r_sources(override):
            return override
        raise RuntimeError(
            "GITTER_R_SOURCE_DIR is set, but required files were not found under "
            f"{override / 'R'}."
        )

    git_bin = shutil.which("git")
    if git_bin is None:
        raise RuntimeError(
            "git is required to fetch R sources automatically. "
            "Install git or set GITTER_R_SOURCE_DIR to a local checkout."
        )

    target = scratch_dir / "r_gitter_src"
    try:
        subprocess.run(
            [git_bin, "clone", "--depth", "1", R_SOURCE_REPO, str(target)],
            check=True,
            timeout=180,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        if R_SOURCE_REF:
            subprocess.run(
                [git_bin, "-C", str(target), "fetch", "--depth", "1", "origin", R_SOURCE_REF],
                check=True,
                timeout=180,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            subprocess.run(
                [git_bin, "-C", str(target), "checkout", "--detach", "FETCH_HEAD"],
                check=True,
                timeout=60,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Timed out while fetching R sources from {R_SOURCE_REPO}.") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip()
        raise RuntimeError(f"Failed to fetch R sources from {R_SOURCE_REPO}: {detail}") from exc

    if not _has_required_r_sources(target):
        raise RuntimeError(
            "Fetched repository does not contain required gitter R sources in R/."
        )

    return target


def _run_r_in_docker(repo_root: Path, r_source_root: Path, out_tsv: Path) -> None:
    script_path = repo_root / "tests" / "_tmp_run_gitter_r_live.R"
    script_rel = script_path.relative_to(repo_root)
    out_rel = out_tsv.relative_to(repo_root)
    image_rel = Path("examples/extdata/sample.jpg")

    script = f"""
suppressPackageStartupMessages(library(jpeg))
suppressPackageStartupMessages(library(tiff))
suppressPackageStartupMessages(library(logging))
suppressPackageStartupMessages(library(EBImage))

source("/r_gitter_src/R/Peaks.R")
source("/r_gitter_src/R/Help.R")
source("/r_gitter_src/R/Main.R")

dat <- gitter(
  image.file="/work/{image_rel.as_posix()}",
  plate.format=1536,
  verbose="n",
  grid.save=NULL,
  dat.save=NULL
)
dat <- dat[, c("row", "col", "size", "circularity", "flags")]
write.table(
  dat,
  file="/work/{out_rel.as_posix()}",
  sep="\\t",
  quote=FALSE,
  row.names=FALSE,
  col.names=TRUE,
  na="NA"
)
"""
    script_path.write_text(script.strip() + "\n", encoding="utf-8")
    try:
        subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{repo_root}:/work",
                "-v",
                f"{r_source_root}:/r_gitter_src:ro",
                "-w",
                "/work",
                R_PARITY_IMAGE,
                "Rscript",
                f"/work/{script_rel.as_posix()}",
            ],
            check=True,
            timeout=3600,
        )
    finally:
        script_path.unlink(missing_ok=True)


@pytest.mark.r_live
def test_parity_live_r_docker(tmp_path: Path):
    if os.environ.get("GITTER_ENABLE_R_LIVE") != "1":
        pytest.skip("Set GITTER_ENABLE_R_LIVE=1 to run live R parity test in Docker")
    if not _docker_ready():
        pytest.skip("Docker is not available or not running")
    if not _docker_image_available(R_PARITY_IMAGE):
        pytest.skip(
            "Docker image not found. Build it with "
            f"`docker build -t {R_PARITY_IMAGE} -f docker/r-parity.Dockerfile .`"
        )

    repo_root = Path(__file__).resolve().parents[1]
    try:
        r_source_root = _prepare_r_sources(repo_root=repo_root, scratch_dir=tmp_path)
    except RuntimeError as exc:
        pytest.skip(str(exc))
    out_tsv = repo_root / "tests" / "_tmp_r_live_output.tsv"
    if out_tsv.exists():
        out_tsv.unlink()

    _run_r_in_docker(repo_root=repo_root, r_source_root=r_source_root, out_tsv=out_tsv)
    assert out_tsv.exists(), "R live run did not produce output TSV"

    try:
        r_df = pd.read_csv(out_tsv, sep="\t", na_values=["NA"])
        r_df["row"] = pd.to_numeric(r_df["row"], errors="raise").astype(int)
        r_df["col"] = pd.to_numeric(r_df["col"], errors="raise").astype(int)
        r_df["size"] = pd.to_numeric(r_df["size"], errors="coerce")
        r_df["circularity"] = pd.to_numeric(r_df["circularity"], errors="coerce")
        r_df["flags"] = r_df["flags"].fillna("")

        py = gitter(
            str(repo_root / "examples/extdata/sample.jpg"),
            plate_format=1536,
            verbose="n",
            grid_save=None,
            dat_save=None,
            _auto_plate_detector=False,
        )[["row", "col", "size", "circularity", "flags"]].copy()
        py["row"] = pd.to_numeric(py["row"], errors="raise").astype(int)
        py["col"] = pd.to_numeric(py["col"], errors="raise").astype(int)
        py["size"] = pd.to_numeric(py["size"], errors="coerce")
        py["circularity"] = pd.to_numeric(py["circularity"], errors="coerce")
        py["flags"] = py["flags"].fillna("")

        metrics = _compute_metrics(py=py, r_df=r_df)
        assert metrics["n"] == 1536.0
        assert metrics["size_mae"] <= 2.0, metrics
        assert metrics["size_p99_abs_err"] <= 20.0, metrics
        assert metrics["size_corr"] >= 0.999, metrics
        assert metrics["circularity_mae"] <= 0.06, metrics
        assert metrics["circularity_p99_abs_err"] <= 0.20, metrics
        assert metrics["circularity_corr"] >= 0.88, metrics
        assert metrics["flag_precision"] >= 0.95, metrics
        assert metrics["flag_recall"] >= 0.95, metrics
        assert metrics["flag_exact_match"] >= 0.995, metrics
    finally:
        out_tsv.unlink(missing_ok=True)
