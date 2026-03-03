# Python API

## Main functions

- `gitter`: quantify one image
- `gitter_batch`: quantify many images
- `gitter_read`: load DAT output
- `plot_gitter`: visualize colony metrics
- `plate_warnings`: plate-level warnings
- `summary_gitter`: summary statistics

## Single-image example

```python
from gitter_py import gitter, plot_gitter, plate_warnings

df = gitter(
    image_file="examples/extdata/sample.jpg",
    plate_format=1536,
    verbose="n",
    grid_save=None,
    dat_save=None,
)

print(plate_warnings(df))

fig = plot_gitter(df, plot_type="heatmap", title="Sample")
fig.savefig("sample.png", dpi=200)
```

## Batch example

```python
from gitter_py import gitter_batch

results = gitter_batch(
    image_files=[
        "examples/extdata/sample.jpg",
        "examples/extdata/sample_dead.jpg",
    ],
    plate_format=1536,
    verbose="n",
)

print(results[0].head())
```

## DataFrame shape

Each result row represents one colony position and includes:

- `row`
- `col`
- `size`
- `circularity`
- `flags`

Metadata is available in `df.attrs` (for example `elapsed`, `format`, `file`).
