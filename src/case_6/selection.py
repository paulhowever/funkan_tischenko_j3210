from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from case_6.metrics import rmse
from case_6.nadaraya_watson import nw_predict_fixed, nw_predict_variable
from case_6.types import FloatArray


@dataclass(slots=True)
class SelectionResult:
    kernel_name: str
    param_name: str
    param_value: float
    score_rmse: float


def loo_score_fixed(x: FloatArray, y: FloatArray, h: float, kernel_name: str) -> float:
    preds = np.zeros_like(y)
    for i in range(x.shape[0]):
        mask = np.ones(x.shape[0], dtype=bool)
        mask[i] = False
        preds[i] = nw_predict_fixed(x[mask], y[mask], x[i : i + 1], h, kernel_name)[0]
    return rmse(y, preds)


def loo_score_variable(x: FloatArray, y: FloatArray, k: int, kernel_name: str) -> float:
    preds = np.zeros_like(y)
    for i in range(x.shape[0]):
        mask = np.ones(x.shape[0], dtype=bool)
        mask[i] = False
        preds[i] = nw_predict_variable(x[mask], y[mask], x[i : i + 1], k, kernel_name)[0]
    return rmse(y, preds)


def select_fixed_window(x: FloatArray, y: FloatArray, hs: list[float], kernels: list[str]) -> SelectionResult:
    best: SelectionResult | None = None
    for kernel_name in kernels:
        for h in hs:
            score = loo_score_fixed(x, y, h, kernel_name)
            candidate = SelectionResult(kernel_name=kernel_name, param_name='h', param_value=h, score_rmse=score)
            if best is None or candidate.score_rmse < best.score_rmse:
                best = candidate
    if best is None:
        raise RuntimeError('no candidates provided')
    return best


def select_variable_window(x: FloatArray, y: FloatArray, ks: list[int], kernels: list[str]) -> SelectionResult:
    best: SelectionResult | None = None
    for kernel_name in kernels:
        for k in ks:
            score = loo_score_variable(x, y, k, kernel_name)
            candidate = SelectionResult(kernel_name=kernel_name, param_name='k', param_value=float(k), score_rmse=score)
            if best is None or candidate.score_rmse < best.score_rmse:
                best = candidate
    if best is None:
        raise RuntimeError('no candidates provided')
    return best


def compare_kernel_impact_fixed(
    x: FloatArray,
    y: FloatArray,
    h: float,
    kernels: list[str],
) -> list[SelectionResult]:
    return [
        SelectionResult(
            kernel_name=kernel_name,
            param_name='h',
            param_value=h,
            score_rmse=loo_score_fixed(x, y, h, kernel_name),
        )
        for kernel_name in kernels
    ]


def compare_window_impact_fixed(
    x: FloatArray,
    y: FloatArray,
    hs: list[float],
    kernel_name: str,
) -> list[SelectionResult]:
    return [
        SelectionResult(
            kernel_name=kernel_name,
            param_name='h',
            param_value=h,
            score_rmse=loo_score_fixed(x, y, h, kernel_name),
        )
        for h in hs
    ]
