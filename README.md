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

## CLI (Optional)

```bash
gitter run examples/extdata/sample.jpg --plate-format 1536 --grid-save . --dat-save .
gitter read sample.jpg.dat
gitter plot sample.jpg.dat --plot-type heatmap --out sample.png
```

## Multi-plate Detection

`PlateDetector` is loaded automatically when detector weights are available and
the input image contains multiple plates.

```python
from gitter_py import gitter

plates = gitter(
    image_file="examples/scanomatic/250417_saltLBtest_35_35_Bran_0060_37101.4909.tiff",
    plate_format=1536,
    split_save="split_save",
    verbose="n",
    grid_save=None,
    dat_save=None,
)

for plate_df in plates:
    print(plate_df["size"].median())
```

Detection behavior:

- only plates with confidence `>= 0.95` are extracted by default
- set `autorotate=True` to rotate portrait plate crops to landscape before quantification
- pass `split_save="some_dir"` to save layout and split artifacts

If you want the legacy single-image quantification path on a file like
`examples/extdata/sample.jpg`, use the image directly without split outputs.

## R Parity Tests (Optional)

```bash
docker build -t gitter-r-parity:4.3.3 -f docker/r-parity.Dockerfile .
uv run --extra dev pytest -q tests/test_parity_r_sample.py
GITTER_ENABLE_R_LIVE=1 uv run --extra dev pytest -q -m r_live tests/test_parity_r_live.py
```

The parity checks are intentionally calibrated for biological agreement with the
original R implementation, not exact pixel-for-pixel equality.
