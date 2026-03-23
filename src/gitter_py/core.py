from __future__ import annotations

import logging
import math
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage, signal
from skimage import io as skio
from skimage import morphology, transform, util

from .constants import GITTER_VERSION, PLATE_FORMATS
from .peaks import colony_peaks, colony_peaks_fixed, round_odd, split_half
from .plate_crops import orient_crop, validate_rotate_override

LOGGER = logging.getLogger("gitter")
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s:INFO:gitter:%(message)s")
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)


def _resize_by_width(image: np.ndarray, width: int) -> np.ndarray:
    width = int(width)
    if width <= 0:
        return image
    h, w = image.shape[:2]
    if w == width:
        return image
    new_h = max(1, int(round(h * (width / w))))
    if image.ndim == 2:
        out_shape = (new_h, width)
    else:
        out_shape = (new_h, width, image.shape[2])
    return transform.resize(image, out_shape, preserve_range=True, anti_aliasing=True).astype(float)


def _resize_to_shape(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return transform.resize(image, shape, preserve_range=True, anti_aliasing=True).astype(float)


def _read_image(file_path: str | Path) -> np.ndarray:
    image = skio.imread(file_path)
    image = util.img_as_float(image).astype(float)
    return image


def _coerce_image_input(image_file: str | Path | np.ndarray) -> tuple[np.ndarray, str, str | None]:
    if isinstance(image_file, np.ndarray):
        image = util.img_as_float(np.asarray(image_file)).astype(float)
        return image, "image", None
    image_path = Path(image_file)
    image = _read_image(image_path)
    return image, image_path.name, str(image_path)


def _set_contrast(image: np.ndarray, contrast: float = 10.0) -> np.ndarray:
    c = (100.0 + contrast) / 100.0
    out = ((image - 0.5) * c) + 0.5
    return np.clip(out, 0.0, 1.0)


def _padmatrix(mat: np.ndarray, level: int, pad_val: float) -> np.ndarray:
    return np.pad(mat, ((level, level), (level, level)), mode="constant", constant_values=pad_val)


def _unpadmatrix(mat: np.ndarray, level: int) -> np.ndarray:
    if level <= 0:
        return mat
    return mat[level:-level, level:-level]


def _find_optimal_threshold(
    x: np.ndarray,
    r: int = 2,
    lim: tuple[float, float] = (0.0, 0.4),
    cap: float = 0.2,
) -> float:
    values = np.asarray(x, dtype=float).copy()
    values[values < lim[0]] = 0
    values[values > lim[1]] = 0

    t = round(float(np.nanmean(values)), r)
    mu1 = float(np.nanmean(values[values >= t])) if np.any(values >= t) else 0.0
    mu2 = float(np.nanmean(values[values < t])) if np.any(values < t) else 0.0
    cmu1 = mu1
    cmu2 = mu2
    i = 1
    while True:
        if i > 1:
            cmu1 = float(np.nanmean(values[values >= t])) if np.any(values >= t) else 0.0
            cmu2 = float(np.nanmean(values[values < t])) if np.any(values < t) else 0.0
        if (i > 1 and cmu1 == mu1 and cmu2 == mu2) or t > 1:
            if t > cap:
                t = float(np.nanquantile(values, 0.9))
            return t
        mu1 = cmu1
        mu2 = cmu2
        t = round((mu1 + mu2) / 2.0, r)
        i += 1


def _threshold(
    im_gray: np.ndarray,
    nrow_plate: int,
    ncol_plate: int,
    fast: bool = True,
    f: int = 1000,
) -> np.ndarray:
    if fast:
        small = _resize_by_width(im_gray, f)
        si = round((small.shape[0] / max(nrow_plate, 1)) * 1.5)
        footprint = np.ones((round_odd(si), round_odd(si)), dtype=bool)
        opened = morphology.opening(small, footprint=footprint)
        opened = _resize_to_shape(opened, im_gray.shape)
        corr = im_gray - opened
        corr[corr < 0] = 0
        thresh = _find_optimal_threshold(_resize_by_width(corr, f))
        bw = (corr >= thresh).astype(float)
        return bw

    si = round((im_gray.shape[0] / max(nrow_plate, 1)) * 1.0)
    footprint = np.ones((round_odd(si), round_odd(si)), dtype=bool)
    tophat = morphology.white_tophat(im_gray, footprint=footprint)
    thresh = _find_optimal_threshold(tophat)
    return (tophat >= thresh).astype(float)


def _has_long_run(values: np.ndarray, cap: float) -> bool:
    if len(values) == 0:
        return False
    vals = values.astype(int)
    changes = np.diff(vals)
    run_starts = np.r_[0, np.where(changes != 0)[0] + 1]
    run_ends = np.r_[run_starts[1:], len(vals)]
    run_vals = vals[run_starts]
    run_lengths = run_ends - run_starts
    return bool(np.any((run_vals == 1) & (run_lengths > cap)))


def _rm_rle(image: np.ndarray, p: float = 0.2, margin: int = 1) -> np.ndarray:
    cap = p * (image.shape[0] if margin == 1 else image.shape[1])
    if margin == 1:
        mask = np.array([_has_long_run(row, cap) for row in image], dtype=bool)
        vec = image.sum(axis=1).astype(float)
    else:
        mask = np.array([_has_long_run(col, cap) for col in image.T], dtype=bool)
        vec = image.sum(axis=0).astype(float)
    left, right = split_half(mask.astype(bool))
    if np.any(left):
        vec[: np.max(np.where(left)[0]) + 1] = 0
    if np.any(right):
        vec[np.min(np.where(right)[0]) + len(left) :] = 0
    return vec


def _xl(vec: np.ndarray, minimum: int) -> int:
    if len(vec) == 0:
        return minimum
    idx = np.where(vec == np.nanmin(vec))[0]
    if len(idx) == 0:
        return minimum
    t = len(vec) - (idx[-1] + 1)
    if t < minimum:
        t = minimum
    return int(t)


def _xr(vec: np.ndarray, minimum: int) -> int:
    if len(vec) == 0:
        return minimum
    t = int(np.nanargmin(vec)) + 2
    if t < minimum:
        t = minimum
    return int(t)


def _mat_border(x: np.ndarray) -> dict[str, np.ndarray]:
    return {"l": x[:, 0], "r": x[:, -1], "t": x[0, :], "b": x[-1, :]}


def _spilled(x: np.ndarray, frac: float) -> bool:
    borders = _mat_border(x)
    return bool(
        (np.sum(borders["l"]) > x.shape[0] * frac)
        or (np.sum(borders["r"]) > x.shape[0] * frac)
        or (np.sum(borders["t"]) > x.shape[1] * frac)
        or (np.sum(borders["b"]) > x.shape[1] * frac)
    )


def _s1(z: np.ndarray) -> np.ndarray:
    return np.hstack([np.zeros((z.shape[0], 1), dtype=bool), z[:, :-1]])


def _s2(z: np.ndarray) -> np.ndarray:
    return np.hstack([z[:, 1:], np.zeros((z.shape[0], 1), dtype=bool)])


def _s3(z: np.ndarray) -> np.ndarray:
    return np.vstack([np.zeros((1, z.shape[1]), dtype=bool), z[:-1, :]])


def _s4(z: np.ndarray) -> np.ndarray:
    return np.vstack([z[1:, :], np.zeros((1, z.shape[1]), dtype=bool)])


def _edge(z: np.ndarray) -> np.ndarray:
    return z & ~(_s1(z) & _s2(z) & _s3(z) & _s4(z))


def _perimeter(z: np.ndarray) -> float:
    e = _edge(z)
    return float(
        (
            np.sum(e & _s1(e))
            + np.sum(e & _s2(e))
            + np.sum(e & _s3(e))
            + np.sum(e & _s4(e))
            + math.sqrt(2.0)
            * (
                np.sum(e & _s1(_s3(e)))
                + np.sum(e & _s1(_s4(e)))
                + np.sum(e & _s2(_s3(e)))
                + np.sum(e & _s2(_s4(e)))
            )
        )
        / 2.0
    )


def _circularity(z: np.ndarray) -> float:
    area = float(np.sum(z))
    perimeter = _perimeter(z.astype(bool))
    if perimeter == 0:
        return float("nan")
    return float((4.0 * math.pi * area) / (perimeter**2))


def _fit_rects(
    coords: pd.DataFrame,
    im_gray: np.ndarray,
    d: int,
    fixed_square: float = 2.0,
) -> pd.DataFrame:
    def r_slice_indices(start: int, end: int, n: int) -> np.ndarray:
        if start <= end:
            seq = np.arange(start, end + 1, dtype=int)
        else:
            seq = np.arange(start, end - 1, -1, dtype=int)
        seq = seq[seq != 0]
        seq = seq[(seq > 0) & (seq <= n)]
        return seq - 1

    def r_slice2d(mat: np.ndarray, yt: int, yb: int, xl: int, xr: int) -> np.ndarray:
        rows = r_slice_indices(yt, yb, mat.shape[0])
        cols = r_slice_indices(xl, xr, mat.shape[1])
        if len(rows) == 0 or len(cols) == 0:
            return np.zeros((0, 0), dtype=float)
        return mat[np.ix_(rows, cols)]

    minb = int(round(d / 3))
    rows: list[list[float]] = []

    h, w = im_gray.shape
    for row in coords.itertuples(index=False):
        x = int(row.x)
        y = int(row.y)
        xl = int(row.xl)
        xr = int(row.xr)
        yt = int(row.yt)
        yb = int(row.yb)
        if xl > xr:
            xl, xr = xr, xl
        if yt > yb:
            yt, yb = yb, yt

        x_r = x + 1
        y_r = y + 1
        xl_r = xl + 1
        xr_r = xr + 1
        yt_r = yt + 1
        yb_r = yb + 1

        cent_pixel = im_gray[y, x] if (0 <= y < h and 0 <= x < w) else 0
        spot_bw = r_slice2d(im_gray, yt_r, yb_r, xl_r, xr_r)
        if spot_bw.size == 0:
            rows.append([x, y, xl, xr, yt, yb, 0.0, float("nan"), 0.0])
            continue

        rs = np.sum(spot_bw, axis=1)
        cs = np.sum(spot_bw, axis=0)
        x_rel = x_r - xl_r
        y_rel = y_r - yt_r

        if cent_pixel == 0:
            z = np.repeat(int(round(minb * fixed_square)), 4)
        else:
            sp_y_l, sp_y_r = split_half(rs)
            sp_x_l, sp_x_r = split_half(cs)
            z = np.array(
                [
                    _xl(sp_x_l, minb),
                    _xr(sp_x_r, minb),
                    _xl(sp_y_l, minb),
                    _xr(sp_y_r, minb),
                ]
            )

        rect_rel = [x_rel - z[0], x_rel + z[1], y_rel - z[2], y_rel + z[3]]
        rect_rel = [int(rect_rel[0]), int(rect_rel[1]), int(rect_rel[2]), int(rect_rel[3])]
        if rect_rel[0] > rect_rel[1]:
            rect_rel[0], rect_rel[1] = rect_rel[1], rect_rel[0]
        if rect_rel[2] > rect_rel[3]:
            rect_rel[2], rect_rel[3] = rect_rel[3], rect_rel[2]

        cropped = r_slice2d(
            spot_bw,
            rect_rel[2],
            rect_rel[3],
            rect_rel[0],
            rect_rel[1],
        )
        spill = _spilled(cropped, 0.2)
        abs_rect = [x - z[0], x + z[1], y - z[2], y + z[3]]
        rows.append(
            [
                x,
                y,
                abs_rect[0],
                abs_rect[1],
                abs_rect[2],
                abs_rect[3],
                float(np.sum(cropped)),
                _circularity(cropped.astype(bool)),
                float(spill),
            ]
        )

    out = pd.DataFrame(
        rows,
        columns=["x", "y", "xl", "xr", "yt", "yb", "size", "circularity", "spill"],
    )
    flags = pd.DataFrame(
        {
            "spilled": out["spill"] == 1,
            "lowcirc": (out["circularity"] < 0.6) & out["circularity"].notna(),
        }
    )
    flag_ids = ["S", "C"]
    out["flags"] = flags.apply(
        lambda r: ",".join([flag_ids[i] for i, val in enumerate(r) if bool(val)]),
        axis=1,
    )
    return out.drop(columns=["spill"])


def _best_lag(a: np.ndarray, b: np.ndarray, lag_max: int) -> int:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    corr = signal.correlate(a, b, mode="full")
    lags = signal.correlation_lags(len(a), len(b), mode="full")
    mask = (lags >= -lag_max) & (lags <= lag_max)
    if not np.any(mask):
        return 0
    i = int(np.argmax(corr[mask]))
    return int(lags[mask][i])


def _register2d(
    image: np.ndarray,
    ref_row_sums: np.ndarray,
    ref_col_sums: np.ndarray,
    lag_max: int = 100,
) -> np.ndarray:
    row_shift = _best_lag(np.sum(image, axis=1), ref_row_sums, lag_max)
    col_shift = _best_lag(np.sum(image, axis=0), ref_col_sums, lag_max)
    return ndimage.shift(image, shift=(row_shift, col_shift), order=0, mode="nearest")


def _draw_rect(
    rects: pd.DataFrame,
    image: np.ndarray,
    color: tuple[float, float, float] = (1.0, 0.647, 0.0),
) -> np.ndarray:
    if image.ndim == 2:
        out = np.repeat(image[:, :, None], 3, axis=2).copy()
    else:
        out = image.copy()
    h, w = out.shape[:2]
    for r in rects.itertuples(index=False):
        xl = int(np.clip(r.xl, 0, w - 1))
        xr = int(np.clip(r.xr, 0, w - 1))
        yt = int(np.clip(r.yt, 0, h - 1))
        yb = int(np.clip(r.yb, 0, h - 1))
        out[yt : yb + 1, xl, :] = color
        out[yt : yb + 1, xr, :] = color
        out[yt, xl : xr + 1, :] = color
        out[yb, xl : xr + 1, :] = color
    return out


def _parse_plate_format(plate_format: int | tuple[int, int] | list[int]) -> tuple[int, int]:
    if isinstance(plate_format, int):
        key = str(plate_format)
        if key not in PLATE_FORMATS:
            raise ValueError(
                "Invalid plate density, please use 1536, 384 or 96. "
                "You can also specify a tuple of (rows, cols)."
            )
        return PLATE_FORMATS[key]
    if len(plate_format) != 2:
        raise ValueError("Invalid plate format, expected density or (rows, cols)")
    return int(plate_format[0]), int(plate_format[1])


def _timestamp() -> str:
    return datetime.now().strftime("%a %b %d %H:%M:%S %Y")


def gitter(
    image_file: str | Path | np.ndarray,
    plate_format: int | tuple[int, int] | list[int] = (32, 48),
    remove_noise: bool = False,
    autorotate: bool = False,
    inverse: bool = False,
    image_align: bool = True,
    verbose: str = "l",
    contrast: int | None = None,
    fast: int | None = None,
    plot: bool = False,
    start_coords: tuple[float, float] | list[float] | None = None,
    increment_coords: tuple[float, float] | list[float] | None = None,
    dilation_factor: int = 0,
    rotate: bool = False,
    rotate_override: int | None = None,
    _fx: float = 2.0,
    _is_ref: bool = False,
    _params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    if start_coords is not None:
        if len(start_coords) != 2:
            raise ValueError("start.coords must be a numeric vector of length 2")
        if increment_coords is None or len(increment_coords) != 2:
            raise ValueError("increment.coords must be provided with start.coords")
    if increment_coords is not None:
        if len(increment_coords) != 2:
            raise ValueError("increment.coords must be a numeric vector of length 2")
        if start_coords is None or len(start_coords) != 2:
            raise ValueError("start.coords must be provided with increment.coords")
    if dilation_factor < 0:
        raise ValueError("dilation.factor must be greater than or equal to 0")
    if verbose not in {"l", "p", "n"}:
        raise ValueError('Invalid verbose parameter, use "l", "p", or "n"')
    if contrast is not None and contrast <= 0:
        raise ValueError("Contrast value must be positive")
    if fast is not None and (fast < 1500 or fast > 4000):
        raise ValueError("Fast resize width must be between 1500-4000px")
    rotate_override = validate_rotate_override(rotate_override)
    should_auto_landscape = bool(autorotate or rotate)

    plate_rows, plate_cols = _parse_plate_format(plate_format)
    image, source_name, source_file = _coerce_image_input(image_file)
    if source_name.startswith("gridded_"):
        LOGGER.warning("Detected gridded image as input")

    if verbose == "l":
        LOGGER.setLevel(logging.INFO)
    else:
        LOGGER.setLevel(logging.ERROR)
    if verbose == "p":
        print(f"Processing {source_name} ...")

    t0 = perf_counter()
    image, _ = orient_crop(
        image,
        rotate=should_auto_landscape,
        rotate_override=rotate_override,
    )
    if fast is not None:
        image = _resize_by_width(image, fast)
    if image.ndim == 3:
        image = image[:, :, :3]
        im_gray = (image[:, :, 0] * 0.21) + (image[:, :, 1] * 0.72) + (image[:, :, 2] * 0.07)
    else:
        im_gray = image.copy()

    if contrast is not None:
        im_gray = _set_contrast(im_gray, contrast)
    if inverse:
        im_gray = 1.0 - im_gray

    is_ref = _params is None
    if not is_ref and image_align:
        im_gray = _register2d(im_gray, _params["row_sums"], _params["col_sums"])

    im_bw = _threshold(im_gray, plate_rows, plate_cols, fast=True, f=1000)
    if remove_noise:
        im_bw = morphology.opening(im_bw, footprint=morphology.diamond(1))

    sum_y = _rm_rle(im_bw, p=0.6, margin=1)
    sum_x = _rm_rle(im_bw, p=0.6, margin=2)

    params: dict[str, Any]
    if is_ref:
        if start_coords is not None and increment_coords is not None:
            cp_y = colony_peaks_fixed(plate_rows, start_coords[1] - 1, increment_coords[1])
            cp_x = colony_peaks_fixed(plate_cols, start_coords[0] - 1, increment_coords[0])
        else:
            cp_y = colony_peaks(
                sum_y,
                n=plate_rows,
                plate_size=plate_rows * plate_cols,
                plot=plot,
            )
            cp_x = colony_peaks(
                sum_x,
                n=plate_cols,
                plate_size=plate_rows * plate_cols,
                plot=plot,
            )

        window = int(round(np.mean([cp_x["window"], cp_y["window"]])))
        grid_x = np.tile(cp_x["peaks"], plate_rows)
        grid_y = np.repeat(cp_y["peaks"], plate_cols)
        coords = pd.DataFrame({"x": grid_x.astype(int), "y": grid_y.astype(int)})

        params = {
            "window": window,
            "coords": coords.copy(),
            "row_sums": np.sum(im_bw, axis=1),
            "col_sums": np.sum(im_bw, axis=0),
        }
    else:
        params = _params
        window = int(params["window"])
        coords = params["coords"].copy()

    d = round_odd(round(window * 1.5))
    im_pad = _padmatrix(im_bw, window, 1)
    coords = coords.copy()
    coords["x"] = coords["x"] + window
    coords["y"] = coords["y"] + window
    coords["xl"] = coords["x"] - d
    coords["xr"] = coords["x"] + d
    coords["yt"] = coords["y"] - d
    coords["yb"] = coords["y"] + d

    if dilation_factor >= 1:
        kernel = np.ones((round_odd(dilation_factor), round_odd(dilation_factor)), dtype=bool)
        op = morphology.dilation(im_pad, footprint=kernel)
    else:
        op = im_pad
    fit = _fit_rects(coords, op, d=window, fixed_square=_fx)

    im_bw = _unpadmatrix(im_pad, window)
    for col in ["x", "y", "xl", "xr", "yt", "yb"]:
        fit[col] = fit[col] - window
    fit[["x", "y", "xl", "xr", "yt", "yb"]] = fit[["x", "y", "xl", "xr", "yt", "yb"]].clip(lower=0)
    fit[["xl", "xr"]] = fit[["xl", "xr"]].clip(upper=im_bw.shape[1] - 1)
    fit[["yt", "yb"]] = fit[["yt", "yb"]].clip(upper=im_bw.shape[0] - 1)

    rows = np.repeat(np.arange(1, plate_rows + 1), plate_cols)
    cols = np.tile(np.arange(1, plate_cols + 1), plate_rows)
    rc = pd.DataFrame({"row": rows, "col": cols})

    results = pd.concat([rc, fit.reset_index(drop=True)], axis=1)
    results = results[
        ["row", "col", "size", "circularity", "flags", "x", "y", "xl", "xr", "yt", "yb"]
    ]

    elapsed = round(perf_counter() - t0, 5)

    results = results[["row", "col", "size", "circularity", "flags"]]

    results.attrs["params"] = params
    results.attrs["elapsed"] = elapsed
    results.attrs["call"] = f"gitter({image_file!r})"
    results.attrs["file"] = source_file if source_file is not None else source_name
    results.attrs["format"] = (plate_rows, plate_cols)
    return results


def gitter_batch(
    image_files: str | list[str],
    ref_image_file: str | None = None,
    verbose: str = "l",
    **kwargs: Any,
) -> None:
    failed_name = "gitter_failed_images"
    if any(Path(".").glob(f"{failed_name}*")):
        failed_name = f"{failed_name}_{datetime.now().strftime('%d-%m-%y_%H-%M-%S')}"
    failed_file = Path(f"{failed_name}.txt")

    if isinstance(image_files, str):
        image_list: list[str] = [image_files]
    else:
        image_list = list(image_files)
    if not image_list:
        raise ValueError("No image files were provided")

    first = Path(image_list[0])
    if len(image_list) == 1 and first.is_dir():
        image_list = [
            str(p)
            for p in sorted(first.iterdir())
            if p.suffix.lower() in {".jpg", ".jpeg", ".tiff", ".tif"}
            and not p.name.startswith("gridded_")
        ]
        if len(image_list) == 0:
            raise ValueError(
                "No images with JPEG/JPG/TIFF extension found. Convert non-supported images first."
            )

    missing = [f for f in image_list if not Path(f).exists()]
    if missing:
        raise FileNotFoundError(f'Files do not exist: {", ".join(missing)}')

    params = None
    if ref_image_file is not None:
        ref = gitter(ref_image_file, verbose=verbose, _is_ref=True, **kwargs)
        params = ref.attrs.get("params")

    failed: list[str] = []
    for image_file in image_list:
        try:
            gitter(image_file, _params=params, _is_ref=False, verbose=verbose, **kwargs)
        except Exception as exc:
            failed.append(f"{Path(image_file).name}\t{type(exc).__name__}: {exc}")
            if verbose == "p":
                print(f"Failed to process {image_file}, skipping ({type(exc).__name__}: {exc})")

    if failed:
        lines = [f"# gitter v{GITTER_VERSION} failed images generated on {_timestamp()}", *failed]
        failed_file.write_text("\n".join(lines), encoding="utf-8")
