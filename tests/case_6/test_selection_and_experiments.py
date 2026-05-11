import numpy as np

from case_6.data import make_sinusoidal_dataset, make_sinusoidal_split
from case_6.experiments import run_synthetic_comparison
from case_6.selection import (
    loo_score_fixed,
    select_fixed_window,
    select_lowess,
    select_variable_window,
)


def test_parameter_selection_returns_candidate() -> None:
    ds = make_sinusoidal_dataset(n_samples=80, seed=7)
    fixed = select_fixed_window(ds.x, ds.y, hs=[0.2, 0.5], kernels=["gaussian", "triangular"])
    variable = select_variable_window(ds.x, ds.y, ks=[3, 5], kernels=["gaussian", "triangular"])
    assert fixed.param_name == "h" and fixed.param_value in (0.2, 0.5)
    assert variable.param_name == "k" and variable.param_value in (3.0, 5.0)


def test_select_lowess_runs_and_selects_something() -> None:
    ds = make_sinusoidal_dataset(n_samples=60, seed=3)
    res = select_lowess(ds.x, ds.y, ks=[5, 10], kernels=["quartic", "triangular"])
    assert res.param_name == "k"
    assert res.param_value in (5.0, 10.0)
    assert res.score_rmse > 0


def test_loo_score_constant_target_is_zero() -> None:
    x = np.linspace(0, 1, 20)[:, None]
    y = np.full(20, 4.0)
    score = loo_score_fixed(x, y, h=0.2, kernel_name="quartic")
    assert score < 1e-8


def test_synthetic_comparison_runs() -> None:
    results = run_synthetic_comparison(seed=11)
    assert set(results.keys()) == {"nw_fixed", "nw_variable", "lowess"}
    for metric in results.values():
        assert metric.mae >= 0.0
        assert metric.rmse >= 0.0
        assert metric.r2 <= 1.0


def test_make_sinusoidal_split_outliers_only_in_train() -> None:
    """Test set should be free of injected outliers."""
    train, test = make_sinusoidal_split(
        n_samples=400, noise_std=0.05, train_outlier_fraction=0.2,
        outlier_scale=5.0, seed=42,
    )
    # train residuals from sin(x) should have heavy tail
    train_resid = train.y - np.sin(train.x[:, 0])
    test_resid = test.y - np.sin(test.x[:, 0])
    # max absolute residual on train must be much larger than on test
    assert np.max(np.abs(train_resid)) > 5.0 * np.max(np.abs(test_resid))
