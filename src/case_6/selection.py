from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from case_6.distance import pairwise_euclidean
from case_6.kernels import KERNELS
from case_6.metrics import rmse
from case_6.types import FloatArray


@dataclass(slots=True)
class SelectionResult:
    kernel_name: str
    param_name: str
    param_value: float
    score_rmse: float


def _loo_predictions(weights: FloatArray, y: FloatArray) -> FloatArray:
    """Given a square (n,n) kernel weight matrix, return LOO NW predictions."""
    w = weights.copy()
    np.fill_diagonal(w, 0.0)
    denom = np.sum(w, axis=1)
    numer = w @ y
    preds = np.full(y.shape[0], float(np.mean(y)), dtype=np.float64)
    mask = denom > 1e-12
    preds[mask] = numer[mask] / denom[mask]
    return preds


def loo_score_fixed(x: FloatArray, y: FloatArray, h: float, kernel_name: str) -> float:
    if h <= 0:
        raise ValueError('h must be > 0')
    if kernel_name not in KERNELS:
        raise ValueError(f'unknown kernel: {kernel_name}')
    d = pairwise_euclidean(x, x)
    w = KERNELS[kernel_name](d / h)
    preds = _loo_predictions(w, y)
    return rmse(y, preds)


def loo_score_variable(x: FloatArray, y: FloatArray, k: int, kernel_name: str) -> float:
    if k < 1 or k >= x.shape[0]:
        raise ValueError('k must be in [1, n-1]')
    if kernel_name not in KERNELS:
        raise ValueError(f'unknown kernel: {kernel_name}')
    d = pairwise_euclidean(x, x)
    d_for_h = d.copy()
    np.fill_diagonal(d_for_h, np.inf)
    h_values = np.sort(d_for_h, axis=1)[:, k - 1]
    h_values = np.where(h_values < 1e-12, 1e-12, h_values)
    w = KERNELS[kernel_name](d / h_values[:, None])
    preds = _loo_predictions(w, y)
    return rmse(y, preds)


def select_fixed_window(
    x: FloatArray, y: FloatArray, hs: list[float], kernels: list[str]
) -> SelectionResult:
    best: SelectionResult | None = None
    for kernel_name in kernels:
        for h in hs:
            score = loo_score_fixed(x, y, h, kernel_name)
            cand = SelectionResult(kernel_name, 'h', float(h), score)
            if best is None or cand.score_rmse < best.score_rmse:
                best = cand
    if best is None:
        raise RuntimeError('no candidates provided')
    return best


def select_variable_window(
    x: FloatArray, y: FloatArray, ks: list[int], kernels: list[str]
) -> SelectionResult:
    best: SelectionResult | None = None
    for kernel_name in kernels:
        for k in ks:
            score = loo_score_variable(x, y, k, kernel_name)
            cand = SelectionResult(kernel_name, 'k', float(k), score)
            if best is None or cand.score_rmse < best.score_rmse:
                best = cand
    if best is None:
        raise RuntimeError('no candidates provided')
    return best


def compare_kernel_impact_fixed(
    x: FloatArray, y: FloatArray, h: float, kernels: list[str]
) -> list[SelectionResult]:
    return [
        SelectionResult(k, 'h', float(h), loo_score_fixed(x, y, h, k))
        for k in kernels
    ]


def compare_window_impact_fixed(
    x: FloatArray, y: FloatArray, hs: list[float], kernel_name: str
) -> list[SelectionResult]:
    return [
        SelectionResult(kernel_name, 'h', float(h), loo_score_fixed(x, y, h, kernel_name))
        for h in hs
    ]
