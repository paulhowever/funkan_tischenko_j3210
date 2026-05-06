from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from case_6.data import (
    load_california_dataset,
    load_diabetes_dataset,
    make_sinusoidal_dataset,
    train_test_split,
)
from case_6.lowess import lowess_fit_predict, lowess_predict_query
from case_6.metrics import mae, r2, rmse
from case_6.nadaraya_watson import nw_predict_fixed, nw_predict_variable
from case_6.selection import (
    compare_kernel_impact_fixed,
    compare_window_impact_fixed,
    select_fixed_window,
    select_variable_window,
)


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
    pred_lowess = lowess_predict_query(train_ds.x, lowess_train_pred, test_ds.x, k=10, kernel_name='triangular')

    return {
        'nw_fixed': evaluate_predictions(test_ds.y, pred_fixed),
        'nw_variable': evaluate_predictions(test_ds.y, pred_variable),
        'lowess': evaluate_predictions(test_ds.y, pred_lowess),
    }


def run_real_dataset_benchmark(seed: int = 42) -> dict[str, dict[str, EvalMetrics]]:
    datasets = {
        "diabetes": load_diabetes_dataset(standardize=True),
    }
    try:
        datasets["california"] = load_california_dataset(standardize=True, max_samples=3000)
    except RuntimeError:
        pass
    kernels = ['gaussian', 'epanechnikov', 'triangular']
    report: dict[str, dict[str, EvalMetrics]] = {}
    for name, ds in datasets.items():
        train_ds, test_ds = train_test_split(ds, test_size=0.25, seed=seed)
        best_h = select_fixed_window(train_ds.x, train_ds.y, hs=[0.3, 0.5, 0.8, 1.2], kernels=kernels)
        best_k = select_variable_window(train_ds.x, train_ds.y, ks=[5, 10, 15, 20], kernels=kernels)
        pred_fixed = nw_predict_fixed(train_ds.x, train_ds.y, test_ds.x, h=best_h.param_value, kernel_name=best_h.kernel_name)
        pred_variable = nw_predict_variable(train_ds.x, train_ds.y, test_ds.x, k=int(best_k.param_value), kernel_name=best_k.kernel_name)
        lowess_train_pred, _ = lowess_fit_predict(train_ds.x, train_ds.y, k=10, kernel_name='triangular')
        pred_lowess = lowess_predict_query(train_ds.x, lowess_train_pred, test_ds.x, k=10, kernel_name='triangular')
        report[name] = {
            "nw_fixed": evaluate_predictions(test_ds.y, pred_fixed),
            "nw_variable": evaluate_predictions(test_ds.y, pred_variable),
            "lowess": evaluate_predictions(test_ds.y, pred_lowess),
        }
    return report


def kernel_vs_window_impact(seed: int = 42) -> dict[str, list[float]]:
    ds = make_sinusoidal_dataset(n_samples=180, noise_std=0.12, seed=seed)
    kernel_scores = compare_kernel_impact_fixed(ds.x, ds.y, h=0.3, kernels=['gaussian', 'epanechnikov', 'triangular', 'quartic'])
    window_scores = compare_window_impact_fixed(ds.x, ds.y, hs=[0.1, 0.2, 0.3, 0.5, 0.8], kernel_name='triangular')
    return {
        "kernel_rmse": [item.score_rmse for item in kernel_scores],
        "window_rmse": [item.score_rmse for item in window_scores],
    }


def lowess_outlier_threshold_study(seed: int = 42) -> list[tuple[float, float, float]]:
    levels = [0.0, 0.03, 0.06, 0.1, 0.15, 0.2]
    results: list[tuple[float, float, float]] = []
    for level in levels:
        ds = make_sinusoidal_dataset(n_samples=220, noise_std=0.12, outlier_fraction=level, seed=seed)
        train_ds, test_ds = train_test_split(ds, test_size=0.25, seed=seed)
        pred_nw = nw_predict_fixed(train_ds.x, train_ds.y, test_ds.x, h=0.3, kernel_name='triangular')
        lowess_train_pred, _ = lowess_fit_predict(train_ds.x, train_ds.y, k=10, kernel_name='triangular')
        pred_lowess = lowess_predict_query(train_ds.x, lowess_train_pred, test_ds.x, k=10, kernel_name='triangular')
        results.append((level, rmse(test_ds.y, pred_nw), rmse(test_ds.y, pred_lowess)))
    return results


def synthetic_curve_artifacts(
    h_values: list[float],
    k_values: list[int],
    seed: int = 42,
) -> dict[str, np.ndarray]:
    ds = make_sinusoidal_dataset(n_samples=220, noise_std=0.12, seed=seed)
    train_ds, test_ds = train_test_split(ds, test_size=0.25, seed=seed)
    x_line = np.linspace(-3.0, 3.0, 240, dtype=np.float64)[:, None]
    y_true_line = np.sin(x_line[:, 0])

    pred_by_h: list[np.ndarray] = []
    for h in h_values:
        pred_by_h.append(nw_predict_fixed(train_ds.x, train_ds.y, x_line, h=h, kernel_name='triangular'))

    rmse_by_h = np.array(
        [rmse(test_ds.y, nw_predict_fixed(train_ds.x, train_ds.y, test_ds.x, h=h, kernel_name='triangular')) for h in h_values],
        dtype=np.float64,
    )
    rmse_by_k = np.array(
        [rmse(test_ds.y, nw_predict_variable(train_ds.x, train_ds.y, test_ds.x, k=k, kernel_name='triangular')) for k in k_values],
        dtype=np.float64,
    )
    return {
        "x_train": train_ds.x[:, 0].astype(np.float64),
        "y_train": train_ds.y.astype(np.float64),
        "x_line": x_line[:, 0].astype(np.float64),
        "y_true_line": y_true_line.astype(np.float64),
        "pred_by_h": np.vstack(pred_by_h).astype(np.float64),
        "rmse_by_h": rmse_by_h,
        "rmse_by_k": rmse_by_k,
    }


def lowess_diagnostic_artifacts(seed: int = 42) -> dict[str, np.ndarray]:
    ds = make_sinusoidal_dataset(n_samples=220, noise_std=0.12, outlier_fraction=0.1, seed=seed)
    train_ds, test_ds = train_test_split(ds, test_size=0.25, seed=seed)
    pred_nw = nw_predict_fixed(train_ds.x, train_ds.y, test_ds.x, h=0.3, kernel_name='triangular')
    lowess_train_pred, gamma = lowess_fit_predict(train_ds.x, train_ds.y, k=10, kernel_name='triangular')
    pred_lowess = lowess_predict_query(train_ds.x, lowess_train_pred, test_ds.x, k=10, kernel_name='triangular')

    return {
        "gamma": gamma.astype(np.float64),
        "x_train": train_ds.x[:, 0].astype(np.float64),
        "y_train": train_ds.y.astype(np.float64),
        "lowess_train_pred": lowess_train_pred.astype(np.float64),
        "test_err_nw": (test_ds.y - pred_nw).astype(np.float64),
        "test_err_lowess": (test_ds.y - pred_lowess).astype(np.float64),
    }


def variable_vs_fixed_win_map(
    noise_levels: list[float],
    seed: int = 42,
) -> list[tuple[float, float, float]]:
    rows: list[tuple[float, float, float]] = []
    for noise in noise_levels:
        ds = make_sinusoidal_dataset(n_samples=220, noise_std=noise, outlier_fraction=0.05, seed=seed)
        train_ds, test_ds = train_test_split(ds, test_size=0.25, seed=seed)
        pred_fixed = nw_predict_fixed(train_ds.x, train_ds.y, test_ds.x, h=0.3, kernel_name='triangular')
        pred_variable = nw_predict_variable(train_ds.x, train_ds.y, test_ds.x, k=10, kernel_name='triangular')
        rows.append((noise, rmse(test_ds.y, pred_fixed), rmse(test_ds.y, pred_variable)))
    return rows
