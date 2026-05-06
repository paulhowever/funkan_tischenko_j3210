from __future__ import annotations

import numpy as np

from case_6.distance import pairwise_euclidean
from case_6.kernels import KERNELS
from case_6.types import FloatArray


def _bisquare(u: FloatArray) -> FloatArray:
    values = (1.0 - u**2) ** 2
    return np.where(np.abs(u) < 1.0, values, 0.0)


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
    gamma = np.ones(x_train.shape[0], dtype=np.float64)
    distances = pairwise_euclidean(x_train, x_train)

    np.fill_diagonal(distances, np.inf)
    h_values = np.sort(distances, axis=1)[:, k - 1]
    h_values = np.where(h_values < 1e-12, 1e-12, h_values)
    np.fill_diagonal(distances, 0.0)

    prev_pred = np.zeros_like(y_train)
    for _ in range(max_iter):
        scaled = distances / h_values[:, None]
        weights = kernel(scaled) * gamma[None, :]
        np.fill_diagonal(weights, 0.0)

        denom = np.sum(weights, axis=1)
        numer = weights @ y_train

        y_hat = np.full_like(y_train, np.mean(y_train))
        mask = denom > 1e-12
        y_hat[mask] = numer[mask] / denom[mask]

        residuals = np.abs(y_hat - y_train)
        med = np.median(residuals)
        if med < 1e-12:
            break

        robust_input = residuals / (6.0 * med)
        gamma = _bisquare(robust_input)

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
    if kernel_name not in KERNELS:
        raise ValueError(f'unknown kernel: {kernel_name}')
    if k < 1 or k >= x_train.shape[0]:
        raise ValueError('k must be in [1, n_train - 1]')

    kernel = KERNELS[kernel_name]
    distances = pairwise_euclidean(x_train, x_query)
    sorted_dist = np.sort(distances, axis=1)
    scales = np.where(sorted_dist[:, k] < 1e-12, 1e-12, sorted_dist[:, k])
    weights = kernel(distances / scales[:, None])
    denom = np.sum(weights, axis=1)
    numer = weights @ lowess_train_pred
    fallback = np.full(x_query.shape[0], np.mean(lowess_train_pred), dtype=np.float64)
    mask = denom > 1e-12
    fallback[mask] = numer[mask] / denom[mask]
    return fallback.astype(np.float64)
