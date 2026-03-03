# Reference Image Workflow

Use a reference image when target plates are sparse or partially empty and
direct grid detection may be unstable.

## Python example (recommended)

```python
from gitter_py import gitter_batch

gitter_batch(
    image_files=["examples/extdata/sample_dead.jpg"],
    ref_image_file="examples/extdata/sample.jpg",
    plate_format=1536,
    verbose="n",
)
```

## Optional CLI example

```bash
gitter batch \
  examples/extdata/sample_dead.jpg \
  --ref-image-file examples/extdata/sample.jpg \
  --plate-format 1536 \
  --verbose p
```

## Notes

- Grid parameters are computed from the reference image once.
- Target plates are aligned against that reference profile.
- Failed images are listed in `gitter_failed_images*.txt`.
