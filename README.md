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

## Python Usage (Recommended)

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

## CLI (Optional)

```bash
gitter run examples/extdata/sample.jpg --plate-format 1536 --grid-save . --dat-save .
gitter read sample.jpg.dat
gitter plot sample.jpg.dat --plot-type heatmap --out sample.png
```

## Live R Parity Test (Optional)

```bash
docker build -t gitter-r-parity:4.3.3 -f docker/r-parity.Dockerfile .
GITTER_ENABLE_R_LIVE=1 uv run --extra dev pytest -q -m r_live tests/test_parity_r_live.py
```
