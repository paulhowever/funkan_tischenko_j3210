import numpy as np

from case_4.basis import trigonometric_basis
from case_4.data import SyntheticConfig
from case_4.experiments import (
    coefficient_stability_study,
    evaluate_single_setup,
    grid_stability_study,
    noise_stability_study,
    sweep_functionals,
    sweep_lambda,
)


def test_single_setup_returns_valid_metrics() -> None:
    config = SyntheticConfig(n_samples=120, n_grid=150, seed=10)
    grid = np.linspace(config.t_start, config.t_end, config.n_grid)
    basis = trigonometric_basis(grid, harmonics=3)

    result = evaluate_single_setup(basis=basis, lambda_value=0.01, config=config)

    assert result.metrics.train_mse >= 0.0
    assert result.metrics.test_mse >= 0.0
    assert result.metrics.train_rmse >= 0.0
    assert result.metrics.test_rmse >= 0.0
    assert -1.0 <= result.metrics.train_r2 <= 1.0
    assert -1.0 <= result.metrics.test_r2 <= 1.0
    assert result.w_t.shape == (config.n_grid,)


def test_sweeps_return_points() -> None:
    config = SyntheticConfig(n_samples=100, n_grid=120, seed=15)
    m_points = sweep_functionals([2, 3, 4], config=config, lambda_value=0.01)
    l_points = sweep_lambda([0.0, 0.01, 0.1], config=config, harmonics=3)
    assert len(m_points) == 3
    assert len(l_points) == 3
    assert all(point.test_rmse >= 0.0 for point in m_points)
    assert all(point.test_rmse >= 0.0 for point in l_points)


def test_stability_studies_return_points() -> None:
    noise_points = noise_stability_study([0.01, 0.05, 0.1], n_grid=120, seed=5)
    grid_points = grid_stability_study([64, 128], seed=5)
    assert len(noise_points) == 3
    assert len(grid_points) == 2


def test_coefficient_stability_returns_valid_stats() -> None:
    stats = coefficient_stability_study([1, 2, 3], lambda_value=0.01, harmonics=3, n_samples=120, n_grid=120)
    assert stats.beta_matrix.shape[0] == 3
    assert stats.mean_std >= 0.0
    assert stats.max_std >= 0.0
