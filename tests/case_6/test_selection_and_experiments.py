from case_6.data import make_sinusoidal_dataset
from case_6.experiments import (
    lowess_diagnostic_artifacts,
    kernel_vs_window_impact,
    lowess_outlier_threshold_study,
    run_real_dataset_benchmark,
    run_synthetic_comparison,
    synthetic_curve_artifacts,
    variable_vs_fixed_win_map,
)
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


def test_impact_and_outlier_studies_run() -> None:
    impact = kernel_vs_window_impact(seed=3)
    threshold = lowess_outlier_threshold_study(seed=3)
    assert len(impact["kernel_rmse"]) >= 3
    assert len(impact["window_rmse"]) >= 3
    assert len(threshold) >= 3


def test_real_dataset_benchmark_runs() -> None:
    report = run_real_dataset_benchmark(seed=2)
    assert "diabetes" in report
    assert set(report.keys()).issubset({"diabetes", "california"})


def test_synthetic_curve_artifacts_shapes() -> None:
    out = synthetic_curve_artifacts([0.1, 0.3], [5, 10], seed=4)
    assert out["pred_by_h"].shape[0] == 2
    assert out["rmse_by_h"].shape == (2,)
    assert out["rmse_by_k"].shape == (2,)


def test_lowess_diagnostic_artifacts_shapes() -> None:
    out = lowess_diagnostic_artifacts(seed=4)
    assert out["gamma"].ndim == 1
    assert out["test_err_nw"].ndim == 1
    assert out["test_err_lowess"].ndim == 1


def test_variable_vs_fixed_win_map_runs() -> None:
    rows = variable_vs_fixed_win_map([0.05, 0.1], seed=4)
    assert len(rows) == 2
