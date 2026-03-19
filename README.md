# gitter-colony

Python 3.13 rewrite of the original R `gitter` package for quantification of pinned microbial colonies.

Documentation: <https://cyberstefnef.github.io/gitter-colony/>

## Installation

Install with `pip`:

```bash
pip install gitter-colony
```

Install with `uv`:

```bash
uv add gitter-colony
```

## Python Usage

```python
from gitter_py import gitter, plot_gitter

df = gitter(
    image_file="examples/extdata/sample.jpg",
    plate_format=1536,
    verbose="n",
)

fig = plot_gitter(df, plot_type="heatmap", title="Sample")
fig.savefig("sample.png", dpi=200)
```

`gitter(...)` returns one row per colony position with:

- `row`
- `col`
- `size`
- `circularity`
- `flags`

`gitter(...)` is single-plate quantification only. If your source image contains
multiple plates, split it explicitly first with `PlateSplitter`.
It accepts either a file path or an in-memory `numpy.ndarray`.

## CLI (Optional)

```bash
gitter run examples/extdata/sample.jpg --plate-format 1536 --grid-save . --dat-save .
gitter read sample.jpg.dat
gitter plot sample.jpg.dat --plot-type heatmap --out sample.png
```

## Multi-plate Splitting

```python
from pathlib import Path

from gitter_py import PlateSplitter, gitter

splitter = PlateSplitter(min_confidence=0.95)
result = splitter.split(
    "examples/scanomatic/250417_saltLBtest_35_35_Bran_0060_37101.4909.tiff"
)
splitter.save(result, "split_save")

for plate_file in sorted(Path("split_save").glob("*__plate_*.tiff")):
    plate_df = gitter(
        image_file=str(plate_file),
        plate_format=1536,
        verbose="n",
        inverse=True,
        autorotate=True,
        grid_save=None,
        dat_save=None,
    )
    print(plate_df["size"].median())

for plate in result.plates:
    plate_df = gitter(
        image_file=plate.crop,
        plate_format=1536,
        verbose="n",
        inverse=True,
        autorotate=True,
        grid_save=None,
        dat_save=None,
    )
    print(plate_df["size"].median())
```

Splitter behavior:

- only plates with confidence `>= 0.95` are extracted by default
- set `autorotate=True` on `gitter(...)` to rotate portrait plate crops before quantification
- call `splitter.save(...)` to write layout and crop artifacts
- for scanomatic-style crops, pass `inverse=True` to `gitter(...)` when needed

## R Parity Tests (Optional)

```bash
docker build -t gitter-r-parity:4.3.3 -f docker/r-parity.Dockerfile .
uv run --extra dev pytest -q tests/test_parity_r_sample.py
GITTER_ENABLE_R_LIVE=1 uv run --extra dev pytest -q -m r_live tests/test_parity_r_live.py
```

The parity checks are intentionally calibrated for biological agreement with the
original R implementation, not exact pixel-for-pixel equality.
