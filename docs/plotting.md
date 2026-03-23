# Plotting

Use `plot_results` to visualize colony results as:

- `heatmap`
- `bubble`

## Python example (recommended)

```python
from gitter_py import gitter, plot_results

df = gitter("examples/extdata/sample.jpg", plate_format=1536, verbose="n")

heatmap = plot_results(df, kind="heatmap", title="Colony size heatmap")
heatmap.savefig("heatmap.png", dpi=200)

bubble = plot_results(df, kind="bubble", title="Colony size bubble plot")
bubble.savefig("bubble.png", dpi=200)
```

Flagged colonies are overlaid when `show_flags=True` (default) and flag data exists.

## Optional CLI example

```bash
gitter run examples/extdata/sample.jpg --out sample.csv
gitter plot sample.csv --plot-type heatmap --out heatmap.png
gitter plot sample.csv --plot-type bubble --out bubble.png
```
