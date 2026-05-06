from __future__ import annotations

import numpy as np

from case_6.types import FloatArray


def mae(y_true: FloatArray, y_pred: FloatArray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mse(y_true: FloatArray, y_pred: FloatArray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: FloatArray, y_pred: FloatArray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def r2(y_true: FloatArray, y_pred: FloatArray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot <= 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)
