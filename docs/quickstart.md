# Quick Start

## Python workflow (recommended)

```python
from gitter_py import gitter, plot_gitter

df = gitter(
    image_file="examples/extdata/sample.jpg",
    plate_format=1536,
    verbose="n",
)

print(df[["row", "col", "size", "circularity", "flags"]].head())

fig = plot_gitter(df, plot_type="heatmap", title="Sample plate")
fig.savefig("sample-heatmap.png", dpi=200)
```

`gitter(...)` returns one row per colony position with the key columns:

- `row`, `col`: colony grid position
- `size`: quantified colony size
- `circularity`: morphology metric
- `flags`: quality/edge warnings

`gitter(...)` is for single-plate images. For multi-plate images, split first
with `PlateSplitter`, then quantify each extracted plate crop separately.
It accepts either an image path or a `numpy.ndarray`.

## Optional CLI workflow

```bash
gitter run examples/extdata/sample.jpg --plate-format 1536 --grid-save . --dat-save .
gitter read sample.jpg.dat
gitter plot sample.jpg.dat --plot-type heatmap --out sample-heatmap.png
```

If `gitter` is not on your PATH, use `uv run gitter ...` instead.
