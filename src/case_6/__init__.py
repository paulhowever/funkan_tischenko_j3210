from case_6.data import (
    dataset_stats,
    inject_outliers,
    load_real_datasets,
    make_sinusoidal_dataset,
    make_sinusoidal_split,
    train_test_split,
)
from case_6.experiments import evaluate_predictions, run_synthetic_comparison
from case_6.kernels import KERNELS
from case_6.lowess import lowess_fit_predict, lowess_predict_query, robust_weight
from case_6.metrics import mae, mse, r2, rmse
from case_6.nadaraya_watson import nw_predict_fixed, nw_predict_variable
from case_6.selection import (
    SelectionResult,
    loo_score_fixed,
    loo_score_lowess,
    loo_score_variable,
    select_fixed_window,
    select_lowess,
    select_variable_window,
)
from case_6.types import TabularDataset

__all__ = [
    'KERNELS',
    'SelectionResult',
    'TabularDataset',
    'dataset_stats',
    'evaluate_predictions',
    'inject_outliers',
    'load_real_datasets',
    'loo_score_fixed',
    'loo_score_lowess',
    'loo_score_variable',
    'lowess_fit_predict',
    'lowess_predict_query',
    'mae',
    'make_sinusoidal_dataset',
    'make_sinusoidal_split',
    'mse',
    'nw_predict_fixed',
    'nw_predict_variable',
    'r2',
    'rmse',
    'robust_weight',
    'run_synthetic_comparison',
    'select_fixed_window',
    'select_lowess',
    'select_variable_window',
    'train_test_split',
]
