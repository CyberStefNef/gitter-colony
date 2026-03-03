from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


def _middle_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return 1.0
    srt = np.sort(values)
    lo = int(0.4 * len(srt))
    hi = max(lo + 1, int(0.6 * len(srt)))
    mid = float(np.nanmean(srt[lo:hi]))
    if np.isnan(mid) or mid == 0:
        return 1.0
    return mid


def plot_gitter(
    x: pd.DataFrame,
    title: str = "",
    plot_type: str = "heatmap",
    low: str = "turquoise",
    mid: str = "black",
    high: str = "yellow",
    show_text: bool = False,
    text_color: str = "white",
    norm: bool = True,
    show_flags: bool = True,
    flag_color: str = "white",
):
    if not isinstance(x, pd.DataFrame):
        raise TypeError("Argument must be a gitter data object")
    if plot_type not in {"heatmap", "bubble"}:
        raise ValueError('Invalid plot type. Use "heatmap" or "bubble"')
    if x.shape[1] < 3 or x.shape[1] > 5:
        raise ValueError("Invalid number of columns for dat file")

    dat = x.iloc[:, :5].copy()
    cols = ["r", "c", "s"] + (["circularity", "flags"] if dat.shape[1] >= 5 else [])
    dat.columns = cols
    dat["r"] = pd.to_numeric(dat["r"], errors="coerce")
    dat["c"] = pd.to_numeric(dat["c"], errors="coerce")
    dat["s"] = pd.to_numeric(dat["s"], errors="coerce")
    dat = dat.dropna(subset=["r", "c", "s"])
    dat["r"] = dat["r"].astype(int)
    dat["c"] = dat["c"].astype(int)

    max_r = int(dat["r"].max())
    max_c = int(dat["c"].max())
    dat["r"] = (max_r + 1) - dat["r"]

    pmm = _middle_mean(dat["s"].to_numpy())
    if norm:
        dat["s"] = dat["s"] / pmm
        pmm = 1.0
        dat["s"] = dat["s"].clip(0.5, 1.5)

    flagged = None
    if "flags" in dat.columns:
        flagged = dat[dat["flags"].astype(str).str.len() > 0]

    cmap = LinearSegmentedColormap.from_list("gitter", [low, mid, high])
    vmin = float(np.nanmin(dat["s"]))
    vmax = float(np.nanmax(dat["s"]))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6
    norm_obj = TwoSlopeNorm(vmin=vmin, vcenter=pmm, vmax=vmax)

    fig, ax = plt.subplots(figsize=(10, 7))
    if plot_type == "heatmap":
        grid = (
            dat.pivot(index="r", columns="c", values="s")
            .reindex(index=np.arange(1, max_r + 1), columns=np.arange(1, max_c + 1))
            .to_numpy()
        )
        image = ax.imshow(grid, cmap=cmap, norm=norm_obj, origin="upper", aspect="equal")
        fig.colorbar(image, ax=ax, label="Size")
    else:
        sc = ax.scatter(
            dat["c"],
            dat["r"],
            c=dat["s"],
            s=np.clip(dat["s"] * 30.0, 8.0, 220.0),
            cmap=cmap,
            norm=norm_obj,
            alpha=0.8,
            marker="o",
        )
        fig.colorbar(sc, ax=ax, label="Size")

    if show_flags and flagged is not None and len(flagged) > 0:
        ax.scatter(flagged["c"], flagged["r"], c=flag_color, s=12)

    if show_text:
        for _, row in dat.iterrows():
            ax.text(
                row["c"],
                row["r"],
                f"{row['s']:.2f}",
                color=text_color,
                ha="center",
                va="center",
                fontsize=7,
            )

    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_xlim(0.5, max_c + 0.5)
    ax.set_ylim(max_r + 0.5, 0.5)
    ax.set_aspect("equal")
    return fig
