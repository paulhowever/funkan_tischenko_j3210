from __future__ import annotations

import numpy as np

from case_6.distance import pairwise_euclidean
from case_6.kernels import KERNELS
from case_6.types import FloatArray


def _safe_weighted_average(weights: FloatArray, y_train: FloatArray) -> FloatArray:
    """Weighted mean per row; falls back to mean(y_train) for empty windows."""
    numer = weights @ y_train
    denom = np.sum(weights, axis=1)
    out = np.full(weights.shape[0], float(np.mean(y_train)), dtype=np.float64)
    mask = denom > 1e-12
    out[mask] = numer[mask] / denom[mask]
    return out


def nw_predict_fixed(
    x_train: FloatArray,
    y_train: FloatArray,
    x_query: FloatArray,
    h: float,
    kernel_name: str,
) -> FloatArray:
    if h <= 0:
        raise ValueError('h must be > 0')
    if kernel_name not in KERNELS:
        raise ValueError(f'unknown kernel: {kernel_name}')

    distances = pairwise_euclidean(x_train, x_query)
    kernel = KERNELS[kernel_name]
    weights = kernel(distances / h)
    return _safe_weighted_average(weights, y_train)


def nw_predict_variable(
    x_train: FloatArray,
    y_train: FloatArray,
    x_query: FloatArray,
    k: int,
    kernel_name: str,
) -> FloatArray:
    if k < 1 or k >= x_train.shape[0]:
        raise ValueError('k must be in [1, n_train - 1]')
    if kernel_name not in KERNELS:
        raise ValueError(f'unknown kernel: {kernel_name}')

    distances = pairwise_euclidean(x_train, x_query)
    sorted_dist = np.sort(distances, axis=1)
    h_values = sorted_dist[:, k - 1]
    h_values = np.where(h_values < 1e-12, 1e-12, h_values)

    kernel = KERNELS[kernel_name]
    scaled = distances / h_values[:, None]
    weights = kernel(scaled)
    return _safe_weighted_average(weights, y_train)
