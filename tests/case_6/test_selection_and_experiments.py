from case_6.data import make_sinusoidal_dataset
from case_6.experiments import run_synthetic_comparison
from case_6.selection import select_fixed_window, select_variable_window


def test_parameter_selection_returns_candidate() -> None:
    ds = make_sinusoidal_dataset(n_samples=80, seed=7)
    fixed = select_fixed_window(ds.x, ds.y, hs=[0.2, 0.5], kernels=['gaussian', 'triangular'])
    variable = select_variable_window(ds.x, ds.y, ks=[3, 5], kernels=['gaussian', 'triangular'])

    assert fixed.param_name == 'h'
    assert fixed.param_value in [0.2, 0.5]
    assert variable.param_name == 'k'
    assert variable.param_value in [3.0, 5.0]


def test_synthetic_comparison_runs() -> None:
    results = run_synthetic_comparison(seed=11)
    assert set(results.keys()) == {'nw_fixed', 'nw_variable', 'lowess'}
    for metric in results.values():
        assert metric.mae >= 0.0
        assert metric.rmse >= 0.0
        assert -1.0 <= metric.r2 <= 1.0
