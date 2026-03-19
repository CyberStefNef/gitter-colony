# gitter-colony

`gitter-colony` is a Python 3.13 rewrite of the original R `gitter` package for
quantification of pinned microbial colonies from plate images.

## Start Here

1. Install the package:
   [Installation](installation.md)
2. Run one image end-to-end in Python:
   [Quick Start](quickstart.md)
3. Learn the core functions:
   [Python API](python-api.md)

## What You Can Do

- Quantify colony sizes from a single image (`gitter`)
- Split multi-plate images explicitly with `PlateSplitter`
- Run batch quantification (`gitter_batch`)
- Read/write DAT-style results (`gitter_read`, `gitter run --dat-save`)
- Plot heatmap and bubble summaries (`plot_gitter`)
- Validate biological agreement against original R output (static + live Docker tests)

## Links

- GitHub: <https://github.com/CyberStefNef/gitter-colony>
- PyPI: <https://pypi.org/project/gitter-colony/>
- Docs: <https://cyberstefnef.github.io/gitter-colony/>
- Original R project: <https://omarwagih.github.io/gitter/>
