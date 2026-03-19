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

For multi-plate images, `gitter(...)` can also return a list of per-plate
results when detector-backed extraction is active.

## Optional CLI workflow

```bash
gitter run examples/extdata/sample.jpg --plate-format 1536 --grid-save . --dat-save .
gitter read sample.jpg.dat
gitter plot sample.jpg.dat --plot-type heatmap --out sample-heatmap.png
```

If `gitter` is not on your PATH, use `uv run gitter ...` instead.
