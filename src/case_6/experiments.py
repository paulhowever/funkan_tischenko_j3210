from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from case_6.data import make_sinusoidal_split
from case_6.lowess import lowess_fit_predict, lowess_predict_query
from case_6.metrics import mae, r2, rmse
from case_6.nadaraya_watson import nw_predict_fixed, nw_predict_variable
from case_6.selection import (
    select_fixed_window,
    select_lowess,
    select_variable_window,
)
from case_6.types import FloatArray


@dataclass(slots=True)
class EvalMetrics:
    mae: float
    rmse: float
    r2: float


def evaluate_predictions(y_true: FloatArray, y_pred: FloatArray) -> EvalMetrics:
    return EvalMetrics(
        mae=mae(y_true, y_pred),
        rmse=rmse(y_true, y_pred),
        r2=r2(y_true, y_pred),
    )


def run_synthetic_comparison(
    seed: int = 42,
    train_outlier_fraction: float = 0.08,
) -> dict[str, EvalMetrics]:
    train_ds, test_ds = make_sinusoidal_split(
        n_samples=240,
        noise_std=0.12,
        train_outlier_fraction=train_outlier_fraction,
        seed=seed,
    )

    kernels = ['gaussian', 'epanechnikov', 'triangular', 'quartic']
    best_h = select_fixed_window(
        train_ds.x, train_ds.y, hs=[0.1, 0.2, 0.3, 0.5, 0.8], kernels=kernels
    )
    best_k = select_variable_window(
        train_ds.x, train_ds.y, ks=[5, 10, 15, 20], kernels=kernels
    )
    best_lowess = select_lowess(
        train_ds.x, train_ds.y, ks=[5, 10, 15, 20], kernels=['quartic', 'triangular']
    )

    pred_fixed = nw_predict_fixed(
        train_ds.x, train_ds.y, test_ds.x,
        h=best_h.param_value, kernel_name=best_h.kernel_name,
    )
    pred_variable = nw_predict_variable(
        train_ds.x, train_ds.y, test_ds.x,
        k=int(best_k.param_value), kernel_name=best_k.kernel_name,
    )

    _, gamma = lowess_fit_predict(
        train_ds.x, train_ds.y,
        k=int(best_lowess.param_value), kernel_name=best_lowess.kernel_name,
    )
    pred_lowess = lowess_predict_query(
        train_ds.x, train_ds.y, gamma, test_ds.x,
        k=int(best_lowess.param_value), kernel_name=best_lowess.kernel_name,
    )

    return {
        'nw_fixed': evaluate_predictions(test_ds.y, pred_fixed),
        'nw_variable': evaluate_predictions(test_ds.y, pred_variable),
        'lowess': evaluate_predictions(test_ds.y, pred_lowess),
    }
