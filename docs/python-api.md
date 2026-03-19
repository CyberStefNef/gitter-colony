# Python API

## Main functions

- `gitter`: quantify one image
- `gitter_batch`: quantify many images
- `gitter_read`: load DAT output
- `plot_gitter`: visualize colony metrics
- `plate_warnings`: plate-level warnings
- `summary_gitter`: summary statistics

`gitter(...)` returns:

- a single `DataFrame` for single-plate inputs
- a `list[DataFrame]` when detector-backed multi-plate extraction is used

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

## Multi-plate split example (`PlateDetector` default when detector weights are available)

```python
from gitter_py import gitter

plates = gitter(
    image_file="examples/scanomatic/250417_saltLBtest_35_35_Bran_0060_37101.4909.tiff",
    plate_format=1536,
    split_save="split_save",  # optional output directory
    verbose="n",
    grid_save=None,
    dat_save=None,
)

for plate_idx, plate_df in enumerate(plates, start=1):
    print(plate_idx, plate_df["size"].median())
```

By default, only detected plates with confidence `>= 0.95` are extracted and processed.
If extracted plate crops are portrait-oriented, pass `autorotate=True` to rotate them
to landscape before quantification.

Useful attrs on detector-extracted results include:

- `plate_index`
- `plate_confidence`
- `crop_rotation_degrees`
- `plate_inverse`
- `detector_name`

## Batch example

```python
from gitter_py import gitter_batch

gitter_batch(
    image_files=[
        "examples/extdata/sample.jpg",
        "examples/extdata/sample_dead.jpg",
    ],
    plate_format=1536,
    verbose="n",
)
```

## DataFrame shape

Each result row represents one colony position and includes:

- `row`
- `col`
- `size`
- `circularity`
- `flags`

Metadata is available in `df.attrs` (for example `elapsed`, `format`, `file`).
