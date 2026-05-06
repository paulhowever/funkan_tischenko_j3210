from __future__ import annotations

import numpy as np

from case_4.basis import BasisSystem
from case_4.types import FloatArray, FunctionalDataset


def build_feature_matrix(dataset: FunctionalDataset, basis: BasisSystem) -> FloatArray:
    dataset.validate()
    basis.validate(dataset.t)

    weighted = dataset.x[:, None, :] * basis.values[None, :, :]
    z = np.trapezoid(weighted, dataset.t, axis=2)
    return z.astype(np.float64)


def add_intercept(z: FloatArray) -> FloatArray:
    if z.ndim != 2:
        raise ValueError('z must be 2D')
    ones = np.ones((z.shape[0], 1), dtype=np.float64)
    return np.hstack([ones, z]).astype(np.float64)


def reconstruct_weight_function(beta: FloatArray, basis: BasisSystem) -> FloatArray:
    if beta.ndim != 1:
        raise ValueError('beta must be 1D')
    if beta.size != basis.values.shape[0] + 1:
        raise ValueError('beta length must equal m + 1 (with intercept)')

    coefficients = beta[1:]
    w = np.sum(coefficients[:, None] * basis.values, axis=0)
    return w.astype(np.float64)
