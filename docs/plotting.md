# Plotting

Use `plot_gitter` to visualize colony results as:

- `heatmap`
- `bubble`

## Python example (recommended)

```python
from gitter_py import gitter, plot_gitter

df = gitter("examples/extdata/sample.jpg", plate_format=1536, verbose="n")

heatmap = plot_gitter(df, plot_type="heatmap", title="Colony size heatmap")
heatmap.savefig("heatmap.png", dpi=200)

bubble = plot_gitter(df, plot_type="bubble", title="Colony size bubble plot")
bubble.savefig("bubble.png", dpi=200)
```

Flagged colonies are overlaid when `show_flags=True` (default) and flag data exists.

## Optional CLI example

```bash
gitter plot sample.jpg.dat --plot-type heatmap --out heatmap.png
gitter plot sample.jpg.dat --plot-type bubble --out bubble.png
```
