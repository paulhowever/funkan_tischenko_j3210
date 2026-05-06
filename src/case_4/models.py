from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from case_4.types import FloatArray


@dataclass(slots=True)
class RegressionResult:
    beta: FloatArray
    y_pred: FloatArray


def ols_fit_predict(x: FloatArray, y: FloatArray) -> RegressionResult:
    if x.ndim != 2:
        raise ValueError('x must be 2D')
    if y.ndim != 1:
        raise ValueError('y must be 1D')
    if x.shape[0] != y.shape[0]:
        raise ValueError('x and y sizes must match')

    xtx = x.T @ x
    xty = x.T @ y
    beta = np.linalg.solve(xtx, xty)
    y_pred = x @ beta
    return RegressionResult(beta=beta.astype(np.float64), y_pred=y_pred.astype(np.float64))


def ridge_fit_predict(x: FloatArray, y: FloatArray, lam: float) -> RegressionResult:
    if lam < 0:
        raise ValueError('lam must be >= 0')
    if x.ndim != 2:
        raise ValueError('x must be 2D')
    if y.ndim != 1:
        raise ValueError('y must be 1D')
    if x.shape[0] != y.shape[0]:
        raise ValueError('x and y sizes must match')

    reg = np.eye(x.shape[1], dtype=np.float64)
    reg[0, 0] = 0.0

    beta = np.linalg.solve(x.T @ x + lam * reg, x.T @ y)
    y_pred = x @ beta
    return RegressionResult(beta=beta.astype(np.float64), y_pred=y_pred.astype(np.float64))


def mse(y_true: FloatArray, y_pred: FloatArray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: FloatArray, y_pred: FloatArray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def r2(y_true: FloatArray, y_pred: FloatArray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)
