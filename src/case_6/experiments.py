from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from case_6.data import make_sinusoidal_dataset, train_test_split
from case_6.lowess import lowess_fit_predict
from case_6.metrics import mae, r2, rmse
from case_6.nadaraya_watson import nw_predict_fixed, nw_predict_variable
from case_6.selection import select_fixed_window, select_variable_window


@dataclass(slots=True)
class EvalMetrics:
    mae: float
    rmse: float
    r2: float


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> EvalMetrics:
    return EvalMetrics(mae=mae(y_true, y_pred), rmse=rmse(y_true, y_pred), r2=r2(y_true, y_pred))


def run_synthetic_comparison(seed: int = 42) -> dict[str, EvalMetrics]:
    ds = make_sinusoidal_dataset(n_samples=220, noise_std=0.12, outlier_fraction=0.08, seed=seed)
    train_ds, test_ds = train_test_split(ds, test_size=0.25, seed=seed)

    kernels = ['gaussian', 'epanechnikov', 'triangular']
    best_h = select_fixed_window(train_ds.x, train_ds.y, hs=[0.1, 0.2, 0.3, 0.5, 0.8], kernels=kernels)
    best_k = select_variable_window(train_ds.x, train_ds.y, ks=[5, 10, 15, 20], kernels=kernels)

    pred_fixed = nw_predict_fixed(train_ds.x, train_ds.y, test_ds.x, h=best_h.param_value, kernel_name=best_h.kernel_name)
    pred_variable = nw_predict_variable(train_ds.x, train_ds.y, test_ds.x, k=int(best_k.param_value), kernel_name=best_k.kernel_name)

    lowess_train_pred, _ = lowess_fit_predict(train_ds.x, train_ds.y, k=10, kernel_name='triangular')
    from case_6.distance import pairwise_euclidean
    from case_6.kernels import KERNELS

    d_test = pairwise_euclidean(train_ds.x, test_ds.x)
    scale = np.sort(d_test, axis=1)[:, 10]
    scale = np.where(scale < 1e-12, 1e-12, scale)
    w_test = KERNELS['triangular'](d_test / scale[:, None])
    numer = w_test @ lowess_train_pred
    denom = np.sum(w_test, axis=1)
    pred_lowess = np.where(denom > 1e-12, numer / denom, np.mean(lowess_train_pred))

    return {
        'nw_fixed': evaluate_predictions(test_ds.y, pred_fixed),
        'nw_variable': evaluate_predictions(test_ds.y, pred_variable),
        'lowess': evaluate_predictions(test_ds.y, pred_lowess),
    }
