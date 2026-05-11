from __future__ import annotations

import numpy as np

from case_6.distance import pairwise_euclidean
from case_6.kernels import KERNELS, quartic_kernel
from case_6.types import FloatArray


def robust_weight(u: FloatArray) -> FloatArray:
    """Bisquare = quartic kernel used for LOWESS reweighting."""
    return quartic_kernel(u)


def lowess_fit_predict(
    x_train: FloatArray,
    y_train: FloatArray,
    k: int,
    kernel_name: str,
    max_iter: int = 20,
    tol: float = 1e-5,
) -> tuple[FloatArray, FloatArray]:
    if k < 1 or k >= x_train.shape[0]:
        raise ValueError('k must be in [1, n_train - 1]')
    if kernel_name not in KERNELS:
        raise ValueError(f'unknown kernel: {kernel_name}')

    kernel = KERNELS[kernel_name]
    n = x_train.shape[0]
    y_train = y_train.astype(np.float64)
    gamma = np.ones(n, dtype=np.float64)

    distances = pairwise_euclidean(x_train, x_train)
    np.fill_diagonal(distances, np.inf)
    h_values = np.sort(distances, axis=1)[:, k - 1]
    h_values = np.where(h_values < 1e-12, 1e-12, h_values)
    np.fill_diagonal(distances, 0.0)

    y_hat = y_train.copy()
    prev_pred = y_train.copy()

    for _ in range(max_iter):
        scaled = distances / h_values[:, None]
        weights = kernel(scaled) * gamma[None, :]
        np.fill_diagonal(weights, 0.0)

        denom = np.sum(weights, axis=1)
        numer = weights @ y_train

        y_hat = np.full(n, float(np.mean(y_train)), dtype=np.float64)
        mask = denom > 1e-12
        y_hat[mask] = numer[mask] / denom[mask]

        residuals = np.abs(y_hat - y_train)
        med = float(np.median(residuals))
        if med < 1e-12:
            prev_pred = y_hat
            break

        gamma = robust_weight(residuals / (6.0 * med))

        if np.max(np.abs(y_hat - prev_pred)) < tol:
            prev_pred = y_hat
            break
        prev_pred = y_hat

    return prev_pred.astype(np.float64), gamma.astype(np.float64)


def lowess_predict_query(
    x_train: FloatArray,
    lowess_train_pred: FloatArray,
    x_query: FloatArray,
    k: int,
    kernel_name: str,
) -> FloatArray:
    """LOWESS inference on new points.

    Uses k-NN bandwidth at each query point and kernel-weighted average of
    the already-smoothed training predictions ``lowess_train_pred``.
    """
    if kernel_name not in KERNELS:
        raise ValueError(f'unknown kernel: {kernel_name}')
    if k < 1 or k >= x_train.shape[0]:
        raise ValueError('k must be in [1, n_train - 1]')
    if lowess_train_pred.shape[0] != x_train.shape[0]:
        raise ValueError('lowess_train_pred and x_train must have the same length')

    kernel = KERNELS[kernel_name]
    distances = pairwise_euclidean(x_train, x_query)
    sorted_dist = np.sort(distances, axis=1)
    k_idx = min(k, sorted_dist.shape[1] - 1)
    scale = sorted_dist[:, k_idx]
    scale = np.where(scale < 1e-12, 1e-12, scale)

    weights = kernel(distances / scale[:, None])
    numer = weights @ lowess_train_pred
    denom = np.sum(weights, axis=1)
    out = np.full(x_query.shape[0], float(np.mean(lowess_train_pred)), dtype=np.float64)
    mask = denom > 1e-12
    out[mask] = numer[mask] / denom[mask]
    return out.astype(np.float64)
