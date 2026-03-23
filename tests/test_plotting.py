from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gitter_py.plotting import plot_results


def test_heatmap_row_order_matches_r_plot():
    df = pd.DataFrame(
        {
            "row": [1, 2, 3],
            "col": [1, 1, 1],
            "size": [10.0, 20.0, 30.0],
            "circularity": [1.0, 1.0, 1.0],
            "flags": ["", "", ""],
        }
    )

    fig = plot_results(df, kind="heatmap", norm=False)
    try:
        grid = np.asarray(fig.axes[0].images[0].get_array())
        assert grid.shape == (3, 1)
        assert grid[0, 0] == 10.0
        assert grid[1, 0] == 20.0
        assert grid[2, 0] == 30.0
        assert tuple(fig.axes[0].images[0].get_extent()) == (0.5, 1.5, 3.5, 0.5)
    finally:
        plt.close(fig)
