from __future__ import annotations

import numpy as np

from case_6.types import FloatArray


def gaussian_kernel(r: FloatArray) -> FloatArray:
    return np.exp(-0.5 * (r**2)) / np.sqrt(2.0 * np.pi)


def triangular_kernel(r: FloatArray) -> FloatArray:
    return np.maximum(1.0 - np.abs(r), 0.0)


def epanechnikov_kernel(r: FloatArray) -> FloatArray:
    values = 0.75 * (1.0 - r**2)
    return np.where(np.abs(r) <= 1.0, values, 0.0)


def quartic_kernel(r: FloatArray) -> FloatArray:
    values = (15.0 / 16.0) * ((1.0 - r**2) ** 2)
    return np.where(np.abs(r) <= 1.0, values, 0.0)


KERNELS = {
    'gaussian': gaussian_kernel,
    'triangular': triangular_kernel,
    'epanechnikov': epanechnikov_kernel,
    'quartic': quartic_kernel,
}
