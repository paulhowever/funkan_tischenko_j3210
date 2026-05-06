from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from case_4.basis import BasisSystem, piecewise_indicator_basis, trigonometric_basis
from case_4.data import SyntheticConfig, make_synthetic_dataset, train_test_split
from case_4.features import add_intercept, build_feature_matrix, reconstruct_weight_function
from case_4.models import mse, ols_fit_predict, r2, ridge_fit_predict, rmse


@dataclass(slots=True)
class MetricReport:
    train_mse: float
    test_mse: float
    train_rmse: float
    test_rmse: float
    train_r2: float
    test_r2: float


@dataclass(slots=True)
class ExperimentResult:
    basis_name: str
    n_functionals: int
    lambda_value: float
    metrics: MetricReport
    beta: np.ndarray
    w_t: np.ndarray


def evaluate_single_setup(
    basis: BasisSystem,
    lambda_value: float,
    config: SyntheticConfig = SyntheticConfig(),
    split_seed: int = 42,
) -> ExperimentResult:
    dataset = make_synthetic_dataset(config)
    train_ds, test_ds = train_test_split(dataset, seed=split_seed)

    z_train = build_feature_matrix(train_ds, basis)
    z_test = build_feature_matrix(test_ds, basis)

    x_train = add_intercept(z_train)
    x_test = add_intercept(z_test)

    if lambda_value == 0.0:
        fit = ols_fit_predict(x_train, train_ds.y)
    else:
        fit = ridge_fit_predict(x_train, train_ds.y, lambda_value)

    y_test_pred = x_test @ fit.beta
    w_t = reconstruct_weight_function(fit.beta, basis)

    metrics = MetricReport(
        train_mse=mse(train_ds.y, fit.y_pred),
        test_mse=mse(test_ds.y, y_test_pred),
        train_rmse=rmse(train_ds.y, fit.y_pred),
        test_rmse=rmse(test_ds.y, y_test_pred),
        train_r2=r2(train_ds.y, fit.y_pred),
        test_r2=r2(test_ds.y, y_test_pred),
    )

    return ExperimentResult(
        basis_name=basis.name,
        n_functionals=basis.values.shape[0],
        lambda_value=lambda_value,
        metrics=metrics,
        beta=fit.beta,
        w_t=w_t,
    )


def compare_bases(config: SyntheticConfig = SyntheticConfig(), lambda_value: float = 0.0) -> list[ExperimentResult]:
    t = np.linspace(config.t_start, config.t_end, config.n_grid)
    bases = [
        piecewise_indicator_basis(t, m=8),
        trigonometric_basis(t, harmonics=4),
    ]
    return [evaluate_single_setup(basis=b, lambda_value=lambda_value, config=config) for b in bases]
