from case_4.basis import (
    BasisSystem,
    check_functional_linearity,
    orthonormality_matrix,
    piecewise_indicator_basis,
    polynomial_basis,
    trigonometric_basis,
)
from case_4.data import PiecewiseConfig, SyntheticConfig, make_piecewise_dataset, make_synthetic_dataset, train_test_split
from case_4.experiments import (
    coefficient_stability_study,
    compare_bases,
    evaluate_piecewise_dataset,
    evaluate_single_setup,
    grid_stability_study,
    noise_stability_study,
    sweep_functionals,
    sweep_lambda,
)
from case_4.features import add_intercept, build_feature_matrix, reconstruct_weight_function
from case_4.models import mse, ols_fit_predict, r2, ridge_fit_predict, rmse
from case_4.types import FunctionalDataset

__all__ = [
    'BasisSystem',
    'FunctionalDataset',
    'PiecewiseConfig',
    'SyntheticConfig',
    'add_intercept',
    'build_feature_matrix',
    'check_functional_linearity',
    'coefficient_stability_study',
    'compare_bases',
    'evaluate_piecewise_dataset',
    'evaluate_single_setup',
    'grid_stability_study',
    'make_piecewise_dataset',
    'make_synthetic_dataset',
    'mse',
    'noise_stability_study',
    'ols_fit_predict',
    'orthonormality_matrix',
    'piecewise_indicator_basis',
    'polynomial_basis',
    'r2',
    'reconstruct_weight_function',
    'ridge_fit_predict',
    'rmse',
    'sweep_functionals',
    'sweep_lambda',
    'train_test_split',
    'trigonometric_basis',
]
