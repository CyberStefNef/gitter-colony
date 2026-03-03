from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import signal
from scipy.stats import norm, spearmanr


def round_odd(value: float) -> int:
    value_i = int(math.floor(value))
    if value_i % 2 == 0:
        value_i += 1
    return max(value_i, 1)


def _ksmooth_normal(values: np.ndarray, bandwidth: float) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    if len(vals) <= 1:
        return vals.copy()
    bw = float(bandwidth)
    if bw <= 0:
        return vals.copy()
    # R::ksmooth("normal") bandwidth is broader than Gaussian sigma used in
    # scipy-style kernels. A quarter-bandwidth aligns the observed smoothing.
    sigma = max(bw / 4.0, 1e-6)
    radius = max(1, int(math.ceil(4.0 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    weights = np.exp(-0.5 * (offsets / sigma) ** 2)
    smooth = np.convolve(vals, weights, mode="same")
    norm_w = np.convolve(np.ones(len(vals), dtype=float), weights, mode="same")
    return smooth / np.maximum(norm_w, 1e-12)


def split_half(vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mid = int(math.ceil(len(vec) / 2.0))
    left = vec[: max(mid - 1, 0)]
    right = vec[mid:]
    return left, right


def shift(vec: np.ndarray, offset: int, pad: float = np.nan) -> np.ndarray:
    out = np.full(vec.shape, pad, dtype=float)
    idx = np.arange(len(vec)) + offset
    valid = (idx >= 0) & (idx < len(vec))
    out[valid] = vec[idx[valid]]
    return out


def _gaussian_smooth(values: np.ndarray, bw: float) -> np.ndarray:
    return _ksmooth_normal(values, bandwidth=bw)


def _local_maxima(values: np.ndarray) -> np.ndarray:
    if len(values) < 3:
        return np.array([], dtype=int)
    return np.where(np.diff(np.sign(np.diff(values))) < 0)[0] + 1


def estimate_window2(values: np.ndarray, bw: int = 15) -> int:
    if len(values) < 4:
        return 3
    lo = int(np.quantile(np.arange(len(values)), 0.25))
    hi = int(np.quantile(np.arange(len(values)), 0.75))
    center = values[lo : hi + 1]
    smooth = _gaussian_smooth(center.astype(float), bw=bw)
    peaks = _local_maxima(smooth)
    if len(peaks) < 2:
        return max(3, int(round(len(center) / 10)))
    dists = np.diff(peaks)
    dists = np.sort(dists)[::-1]
    top = dists[: max(1, len(dists) // 2)]
    return max(3, int(np.nanmedian(top)))


def sin_correlation(values: np.ndarray, window: int) -> np.ndarray:
    values = _gaussian_smooth(values.astype(float), bw=15)
    window = max(round_odd(window), 3)
    ref = np.sin(np.linspace(-np.pi, 2 * np.pi, window))
    corr = np.zeros(len(values), dtype=float)
    half = window // 2
    for i in range(len(values)):
        start = max(i - half, 0)
        end = min(i + half + 1, len(values))
        seg = values[start:end]
        if len(seg) < len(ref):
            seg = np.pad(seg, (0, len(ref) - len(seg)), mode="edge")
        seg = seg[: len(ref)]
        if np.all(seg == seg[0]):
            rho = 0.0
        else:
            rho, _ = spearmanr(ref, seg)
        if np.isnan(rho) or rho < 0.3:
            rho = 0.0
        corr[i] = float(rho)
    return corr


def get_peaks(values: np.ndarray, half_window_size: int) -> np.ndarray:
    if len(values) == 0:
        return np.array([], dtype=bool)
    half_window_size = max(1, int(half_window_size))
    local_max = np.zeros(len(values), dtype=bool)
    for i in range(half_window_size, len(values) - half_window_size):
        win = values[i - half_window_size : i + half_window_size + 1]
        mx = np.max(win)
        if values[i] == mx and np.argmax(win) == half_window_size:
            local_max[i] = True
    return local_max


def _safe_medfilt(values: np.ndarray, kernel_size: int) -> np.ndarray:
    if len(values) == 0:
        return values
    kernel_size = round_odd(kernel_size)
    if kernel_size > len(values):
        kernel_size = len(values) if len(values) % 2 == 1 else len(values) - 1
    if kernel_size <= 1:
        return values
    return signal.medfilt(values, kernel_size=kernel_size)


def colony_peaks(values: np.ndarray, n: int, plate_size: int, plot: bool = False) -> dict[str, Any]:
    del plot
    bw = 40 if plate_size < 500 else 15
    window = estimate_window2(values, bw=bw)
    signal_values = np.asarray(values, dtype=float)
    signal_values = signal_values * _safe_medfilt(signal_values, round_odd(window / 3))

    max_window = max(int(math.floor(len(signal_values) / max(n, 1))), 3)
    if window > max_window:
        window = max_window
    window = max(window, 3)

    padded = np.pad(signal_values, (window, window), mode="constant")
    corr = sin_correlation(padded, window=window)
    corr = corr[window:-window]
    idx = np.arange(len(corr))
    peak_mask = get_peaks(corr, half_window_size=max(1, int(math.floor(window / 3))))
    rp = idx[peak_mask]
    pv = corr[peak_mask]
    if len(rp) < n:
        raise ValueError("Not enough peaks found")

    peak_dist = shift(rp.astype(float), 1) - rp
    peak_height = shift(pv.astype(float), 1) - pv

    num_s = (len(rp) - n) + 1
    scores = np.full(num_s, -np.inf, dtype=float)
    for si in range(num_s):
        dist = peak_dist[si : si + (n - 1)]
        height = peak_height[si : si + (n - 1)]
        dist_m = np.nanmedian(dist)
        dist_sd = float(np.nanstd(dist, ddof=1))
        if dist_sd == 0 or np.isnan(dist_sd):
            dist_sd = 1e-6
        height_m = np.nanmedian(height)
        height_sd = float(np.nanstd(height, ddof=1))
        if height_sd == 0 or np.isnan(height_sd):
            height_sd = 1e-6
        p1 = norm.pdf(dist, loc=dist_m, scale=dist_sd)
        p2 = norm.pdf(height, loc=height_m, scale=height_sd)
        scores[si] = np.nansum(np.log((p1 * p2) + 1e-12))

    start = int(np.argmax(scores))
    peaks = rp[start : start + n]
    delta = np.nanmedian(shift(peaks.astype(float), 1) - peaks)
    if np.isnan(delta):
        delta = window
    return {"peaks": peaks.astype(int), "all_peaks": rp.astype(int), "window": float(delta / 2.0)}


def colony_peaks_fixed(n: int, start_index: float, increment_index: float) -> dict[str, Any]:
    start = int(round(start_index))
    inc = int(round(increment_index))
    peaks = np.array([start + (i * inc) for i in range(n)], dtype=int)
    return {"peaks": peaks, "all_peaks": peaks.copy(), "window": float(inc / 2.0)}
