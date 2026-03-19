# CLI

The package installs a `gitter` command for optional terminal workflows.
If `gitter` is not on your PATH, prefix commands with `uv run`.

## Commands

## `gitter run`

Process one image.

```bash
gitter run IMAGE.jpg --plate-format 1536 --grid-save . --dat-save .
```

Common options:

- `--plate-format 1536` or `--plate-format ROWS,COLS`
- `--remove-noise`
- `--inverse`
- `--contrast INT`
- `--fast INT`
- `--verbose l|p|n`

For single-plate images like `examples/extdata/sample.jpg`, the CLI runs direct
quantification.

For multi-plate images, use the Python API if you need split artifacts or
control over detector-backed extraction.

## `gitter batch`

Process a directory or multiple files.

```bash
gitter batch examples/extdata --ref-image-file examples/extdata/sample.jpg
```

## `gitter read`

Show a DAT summary.

```bash
gitter read sample.jpg.dat
```

## `gitter plot`

Plot DAT results.

```bash
gitter plot sample.jpg.dat --plot-type heatmap --out sample.png
```
