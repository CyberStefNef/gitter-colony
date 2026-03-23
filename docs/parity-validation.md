# Parity Validation

This project includes two parity layers against the original R implementation:

- Static baseline parity (`sample.jpg.dat` fixture only)
- Live R parity via Docker (`r_live` test)

The acceptance thresholds are intentionally based on biological agreement rather
than exact low-level equality. Python and R use different image-processing
libraries, so exact colony masks are not expected to be identical even when the
grids and scientific conclusions match closely.

## Live parity test

Build the pinned R image:

```bash
./scripts/build_r_parity_image.sh
```

Run:

```bash
GITTER_ENABLE_R_LIVE=1 uv run --extra dev pytest -q -m r_live tests/test_parity_r_live.py
```

If `r_gitter/` is not available locally, the test fetches R sources from:

- repo: `https://github.com/omarwagih/gitter.git`
- ref: `master`

Override defaults with:

- `GITTER_R_SOURCE_REPO`
- `GITTER_R_SOURCE_REF`
- `GITTER_R_SOURCE_DIR` (use a local checkout and skip network fetch)

The retained parity coverage is:

- `tests/test_parity_r_sample.py`: static sample baseline against the bundled R fixture `sample.jpg.dat`
- `tests/test_parity_r_live.py`: live Docker comparison against the R code

## Release thresholds

Both parity tests currently require:

- `size_corr >= 0.999`
- `size_mae <= 2.0`
- `size_p99_abs_err <= 20.0`
- `circularity_mae <= 0.06`
- `circularity_p99_abs_err <= 0.20`
- `circularity_corr >= 0.88`
- `flag_precision >= 0.95`
- `flag_recall >= 0.95`
- `flag_exact_match >= 0.995`
