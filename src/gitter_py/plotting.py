from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from skimage import io as skio
from skimage import util


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


def plot_results(
    x: pd.DataFrame,
    title: str = "",
    kind: str = "heatmap",
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
    if kind not in {"heatmap", "bubble"}:
        raise ValueError('Invalid plot type. Use "heatmap" or "bubble"')
    if x.shape[1] < 3 or x.shape[1] > 5:
        raise ValueError("Invalid number of columns for results table")

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
    if kind == "heatmap":
        heatmap_dat = dat.copy()
        heatmap_dat["plot_r"] = (max_r + 1) - heatmap_dat["r"]
        grid = (
            heatmap_dat.pivot(index="plot_r", columns="c", values="s")
            .reindex(index=np.arange(1, max_r + 1), columns=np.arange(1, max_c + 1))
            .to_numpy()
        )
        image = ax.imshow(
            grid,
            cmap=cmap,
            norm=norm_obj,
            origin="upper",
            aspect="equal",
            extent=(0.5, max_c + 0.5, max_r + 0.5, 0.5),
        )
        fig.colorbar(image, ax=ax, label="Size", shrink=0.65, pad=0.04, fraction=0.05)
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
        fig.colorbar(sc, ax=ax, label="Size", shrink=0.65, pad=0.04, fraction=0.05)

    if show_flags and flagged is not None and len(flagged) > 0:
        y = flagged["r"]
        if kind == "heatmap":
            y = (max_r + 1) - y
        ax.scatter(flagged["c"], y, c=flag_color, s=12)

    if show_text:
        for _, row in dat.iterrows():
            plot_r = row["r"]
            if kind == "heatmap":
                plot_r = (max_r + 1) - plot_r
            ax.text(
                row["c"],
                plot_r,
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


def render_grid_overlay(
    image: np.ndarray,
    results: pd.DataFrame,
    color: tuple[float, float, float] = (1.0, 0.647, 0.0),
) -> np.ndarray:
    required = {"xl", "xr", "yt", "yb"}
    if not required.issubset(results.columns):
        raise ValueError("Results must contain xl, xr, yt, and yb columns to render overlays")

    if image.ndim == 2:
        overlay = np.repeat(image[:, :, None], 3, axis=2).copy()
    else:
        overlay = image[:, :, :3].copy()

    h, w = overlay.shape[:2]
    for row in results.itertuples(index=False):
        xl = int(np.clip(row.xl, 0, w - 1))
        xr = int(np.clip(row.xr, 0, w - 1))
        yt = int(np.clip(row.yt, 0, h - 1))
        yb = int(np.clip(row.yb, 0, h - 1))
        overlay[yt : yb + 1, xl, :] = color
        overlay[yt : yb + 1, xr, :] = color
        overlay[yt, xl : xr + 1, :] = color
        overlay[yb, xl : xr + 1, :] = color
    return overlay


def save_grid_overlay(
    image: np.ndarray,
    results: pd.DataFrame,
    path: str,
    color: tuple[float, float, float] = (1.0, 0.647, 0.0),
) -> None:
    overlay = render_grid_overlay(image, results, color=color)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    skio.imsave(target, util.img_as_ubyte(np.clip(overlay, 0.0, 1.0)))


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
    return plot_results(
        x=x,
        title=title,
        kind=plot_type,
        low=low,
        mid=mid,
        high=high,
        show_text=show_text,
        text_color=text_color,
        norm=norm,
        show_flags=show_flags,
        flag_color=flag_color,
    )
