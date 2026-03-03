# Parity Validation

This project includes two parity layers against the original R implementation:

- Static baseline parity (`sample.jpg.dat`)
- Live R parity via Docker (`r_live` test)

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

## Current sample parity (live R vs Python)

- `size_mae`: `0.01171875`
- `size_corr`: `0.9999994939`
- `circularity_mae`: `0.0002680782`
- `circularity_corr`: `0.9990531310`
- `flag_precision`: `1.0`
- `flag_recall`: `1.0`
- `flag_exact_match`: `1.0`

Values are measured on `examples/extdata/sample.jpg`.
