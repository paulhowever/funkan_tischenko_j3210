from __future__ import annotations

import numpy as np

from case_6.types import FloatArray


def pairwise_euclidean(x_train: FloatArray, x_query: FloatArray) -> FloatArray:
    if x_train.ndim != 2 or x_query.ndim != 2:
        raise ValueError('x_train and x_query must be 2D')
    if x_train.shape[1] != x_query.shape[1]:
        raise ValueError('feature dimensions must match')

    diff = x_query[:, None, :] - x_train[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    return dist.astype(np.float64)
