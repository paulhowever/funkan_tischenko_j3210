from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from case_4.basis import BasisSystem, piecewise_indicator_basis, trigonometric_basis
from case_4.data import (
    PiecewiseConfig,
    SyntheticConfig,
    create_uniform_grid,
    make_piecewise_dataset,
    make_synthetic_dataset,
    train_test_split,
)
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


@dataclass(slots=True)
class SweepPoint:
    value: float
    train_rmse: float
    test_rmse: float
    train_r2: float
    test_r2: float


@dataclass(slots=True)
class CoefficientStability:
    mean_std: float
    max_std: float
    beta_matrix: np.ndarray


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
    t = np.linspace(config.t_start, config.t_end, config.n_grid, dtype=np.float64)
    bases = [
        piecewise_indicator_basis(t, m=8),
        trigonometric_basis(t, harmonics=4),
    ]
    return [evaluate_single_setup(basis=b, lambda_value=lambda_value, config=config) for b in bases]


def sweep_functionals(
    m_values: list[int],
    config: SyntheticConfig = SyntheticConfig(),
    lambda_value: float = 0.0,
    basis_name: str = "trigonometric",
) -> list[SweepPoint]:
    t = create_uniform_grid(config.n_grid, config.t_start, config.t_end)
    dataset = make_synthetic_dataset(config)
    train_ds, test_ds = train_test_split(dataset, seed=config.seed)
    points: list[SweepPoint] = []

    for m in m_values:
        if basis_name == "trigonometric":
            basis = trigonometric_basis(t, harmonics=m)
        elif basis_name == "piecewise":
            basis = piecewise_indicator_basis(t, m=m)
        else:
            raise ValueError("unknown basis_name")
        z_train = build_feature_matrix(train_ds, basis)
        z_test = build_feature_matrix(test_ds, basis)
        x_train = add_intercept(z_train)
        x_test = add_intercept(z_test)
        fit = ridge_fit_predict(x_train, train_ds.y, lambda_value) if lambda_value > 0 else ols_fit_predict(x_train, train_ds.y)
        y_test_pred = x_test @ fit.beta
        points.append(
            SweepPoint(
                value=float(basis.values.shape[0]),
                train_rmse=rmse(train_ds.y, fit.y_pred),
                test_rmse=rmse(test_ds.y, y_test_pred),
                train_r2=r2(train_ds.y, fit.y_pred),
                test_r2=r2(test_ds.y, y_test_pred),
            )
        )
    return points


def sweep_lambda(
    lambda_values: list[float],
    config: SyntheticConfig = SyntheticConfig(),
    harmonics: int = 4,
) -> list[SweepPoint]:
    t = create_uniform_grid(config.n_grid, config.t_start, config.t_end)
    basis = trigonometric_basis(t, harmonics=harmonics)
    dataset = make_synthetic_dataset(config)
    train_ds, test_ds = train_test_split(dataset, seed=config.seed)
    z_train = build_feature_matrix(train_ds, basis)
    z_test = build_feature_matrix(test_ds, basis)
    x_train = add_intercept(z_train)
    x_test = add_intercept(z_test)
    points: list[SweepPoint] = []

    for lam in lambda_values:
        fit = ridge_fit_predict(x_train, train_ds.y, lam)
        y_test_pred = x_test @ fit.beta
        points.append(
            SweepPoint(
                value=lam,
                train_rmse=rmse(train_ds.y, fit.y_pred),
                test_rmse=rmse(test_ds.y, y_test_pred),
                train_r2=r2(train_ds.y, fit.y_pred),
                test_r2=r2(test_ds.y, y_test_pred),
            )
        )
    return points


def noise_stability_study(
    noise_levels: list[float],
    n_grid: int = 200,
    seed: int = 42,
) -> list[SweepPoint]:
    points: list[SweepPoint] = []
    for level in noise_levels:
        config = SyntheticConfig(n_samples=300, n_grid=n_grid, noise_x_std=level, noise_y_std=level, seed=seed)
        result = evaluate_single_setup(
            basis=trigonometric_basis(create_uniform_grid(n_grid), harmonics=4),
            lambda_value=0.01,
            config=config,
            split_seed=seed,
        )
        points.append(
            SweepPoint(
                value=level,
                train_rmse=result.metrics.train_rmse,
                test_rmse=result.metrics.test_rmse,
                train_r2=result.metrics.train_r2,
                test_r2=result.metrics.test_r2,
            )
        )
    return points


def grid_stability_study(
    grid_sizes: list[int],
    seed: int = 42,
) -> list[SweepPoint]:
    points: list[SweepPoint] = []
    for n_grid in grid_sizes:
        config = SyntheticConfig(n_samples=300, n_grid=n_grid, seed=seed)
        result = evaluate_single_setup(
            basis=trigonometric_basis(create_uniform_grid(n_grid), harmonics=4),
            lambda_value=0.01,
            config=config,
            split_seed=seed,
        )
        points.append(
            SweepPoint(
                value=float(n_grid),
                train_rmse=result.metrics.train_rmse,
                test_rmse=result.metrics.test_rmse,
                train_r2=result.metrics.train_r2,
                test_r2=result.metrics.test_r2,
            )
        )
    return points


def evaluate_piecewise_dataset(config: PiecewiseConfig = PiecewiseConfig()) -> ExperimentResult:
    dataset = make_piecewise_dataset(config)
    train_ds, test_ds = train_test_split(dataset, seed=config.seed)
    basis = piecewise_indicator_basis(dataset.t, m=10)
    z_train = build_feature_matrix(train_ds, basis)
    z_test = build_feature_matrix(test_ds, basis)
    x_train = add_intercept(z_train)
    x_test = add_intercept(z_test)
    fit = ridge_fit_predict(x_train, train_ds.y, 0.01)
    y_test_pred = x_test @ fit.beta
    return ExperimentResult(
        basis_name=basis.name,
        n_functionals=basis.values.shape[0],
        lambda_value=0.01,
        metrics=MetricReport(
            train_mse=mse(train_ds.y, fit.y_pred),
            test_mse=mse(test_ds.y, y_test_pred),
            train_rmse=rmse(train_ds.y, fit.y_pred),
            test_rmse=rmse(test_ds.y, y_test_pred),
            train_r2=r2(train_ds.y, fit.y_pred),
            test_r2=r2(test_ds.y, y_test_pred),
        ),
        beta=fit.beta,
        w_t=reconstruct_weight_function(fit.beta, basis),
    )


def coefficient_stability_study(
    seeds: list[int],
    lambda_value: float = 0.01,
    harmonics: int = 4,
    n_samples: int = 250,
    n_grid: int = 180,
) -> CoefficientStability:
    beta_values: list[np.ndarray] = []
    for seed in seeds:
        config = SyntheticConfig(n_samples=n_samples, n_grid=n_grid, seed=seed)
        t = create_uniform_grid(n_grid, config.t_start, config.t_end)
        basis = trigonometric_basis(t, harmonics=harmonics)
        result = evaluate_single_setup(basis=basis, lambda_value=lambda_value, config=config, split_seed=seed)
        beta_values.append(result.beta)
    beta_matrix = np.vstack(beta_values)
    std = np.std(beta_matrix, axis=0)
    return CoefficientStability(
        mean_std=float(np.mean(std)),
        max_std=float(np.max(std)),
        beta_matrix=beta_matrix,
    )
