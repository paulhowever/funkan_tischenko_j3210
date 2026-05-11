from __future__ import annotations

import numpy as np

from case_6.distance import pairwise_euclidean
from case_6.kernels import KERNELS, quartic_kernel
from case_6.types import FloatArray


def robust_weight(u: FloatArray) -> FloatArray:
    """Bisquare (=quartic) weight used in LOWESS reweighting."""
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
    y_train: FloatArray,
    gamma: FloatArray,
    x_query: FloatArray,
    k: int,
    kernel_name: str,
) -> FloatArray:
    """Canonical LOWESS inference: kernel weights on query × robust gammas × y_train."""
    if kernel_name not in KERNELS:
        raise ValueError(f'unknown kernel: {kernel_name}')
    kernel = KERNELS[kernel_name]

    d = pairwise_euclidean(x_train, x_query)
    sorted_d = np.sort(d, axis=1)
    k_idx = min(k, sorted_d.shape[1] - 1)
    scale = sorted_d[:, k_idx]
    scale = np.where(scale < 1e-12, 1e-12, scale)
    w = kernel(d / scale[:, None]) * gamma[None, :]
    numer = w @ y_train
    denom = np.sum(w, axis=1)
    fallback = float(np.mean(y_train))
    return np.where(denom > 1e-12, numer / denom, fallback).astype(np.float64)
