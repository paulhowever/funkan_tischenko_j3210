from case_4.basis import trigonometric_basis
from case_4.data import SyntheticConfig
from case_4.experiments import evaluate_single_setup


def test_single_setup_returns_valid_metrics() -> None:
    config = SyntheticConfig(n_samples=120, n_grid=150, seed=10)
    t = config.t_start + (config.t_end - config.t_start) * 0.0
    del t
    import numpy as np

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
