# Python API

## Main functions

- `gitter`: quantify one single-plate image
- `gitter_batch`: quantify many images
- `PlateSplitter`: detect and extract plates from a multi-plate image
- `read_results_csv`: load CSV output
- `write_results_csv`: save CSV output
- `plot_results`: visualize colony metrics
- `plate_warnings`: plate-level warnings
- `summary_gitter`: summary statistics

`gitter(...)` always returns a single `DataFrame`.
It accepts either a file path or an in-memory `numpy.ndarray`.

## Single-image example

```python
from gitter_py import gitter, plot_results, plate_warnings

df = gitter(
    image_file="examples/extdata/sample.jpg",
    plate_format=1536,
    verbose="n",
)

print(plate_warnings(df))

fig = plot_results(df, kind="heatmap", title="Sample")
fig.savefig("sample.png", dpi=200)
```

## Multi-plate split example

```python
from gitter_py import PlateSplitter, gitter

splitter = PlateSplitter(min_confidence=0.95)
result = splitter.split(
    "examples/scanomatic/250417_saltLBtest_35_35_Bran_0060_37101.4909.tiff"
)
splitter.save(result, "split_save")

plate_df = gitter(
    image_file="split_save/250417_saltLBtest_35_35_Bran_0060_37101.4909.tiff__plate_01.tiff",
    plate_format=1536,
    verbose="n",
    inverse=True,
    autorotate=True,
)

for plate in result.plates:
    plate_df = gitter(
        image_file=plate.crop,
        plate_format=1536,
        verbose="n",
        inverse=True,
        autorotate=True,
    )
```

By default, only detected plates with confidence `>= 0.95` are extracted.
If extracted plate crops are portrait-oriented, pass `autorotate=True` to `gitter(...)`
to rotate them to landscape before you quantify them.
For scanomatic-style crops, pass `inverse=True` to `gitter(...)` when needed.

Useful `PlateSplitResult` fields include:

- `plates`
- `overall_confidence`
- `detector_name`

Each `ExtractedPlate` contains:

- `plate_index`
- `confidence`
- `bbox`
- `polygon`
- `crop`
- `crop_rotation_degrees`

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

## CSV export example

```python
from gitter_py import gitter, read_results_csv, write_results_csv

df = gitter("examples/extdata/sample.jpg", plate_format=1536, verbose="n")
write_results_csv(df, "sample.csv")
loaded = read_results_csv("sample.csv")
```
