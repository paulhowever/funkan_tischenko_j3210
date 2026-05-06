from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from case_4.types import FloatArray


@dataclass(slots=True)
class BasisSystem:
    name: str
    values: FloatArray

    def validate(self, t: FloatArray) -> None:
        if self.values.ndim != 2:
            raise ValueError('basis matrix must be 2D (m, n_grid)')
        if self.values.shape[1] != t.size:
            raise ValueError('basis grid length must match t size')


def piecewise_indicator_basis(t: FloatArray, m: int) -> BasisSystem:
    if m < 1:
        raise ValueError('m must be >= 1')

    edges = np.linspace(t.min(), t.max(), m + 1)
    basis = np.zeros((m, t.size), dtype=np.float64)
    for k in range(m):
        left = edges[k]
        right = edges[k + 1]
        mask = (t >= left) & (t < right)
        if k == m - 1:
            mask = (t >= left) & (t <= right)
        basis[k, mask] = 1.0
    return BasisSystem(name='piecewise', values=basis)


def polynomial_basis(t: FloatArray, degree: int) -> BasisSystem:
    if degree < 0:
        raise ValueError('degree must be >= 0')

    powers = np.arange(degree + 1)
    basis = np.vstack([t ** power for power in powers]).astype(np.float64)
    return BasisSystem(name='polynomial', values=basis)


def trigonometric_basis(t: FloatArray, harmonics: int) -> BasisSystem:
    if harmonics < 1:
        raise ValueError('harmonics must be >= 1')

    vectors: list[FloatArray] = [np.ones_like(t, dtype=np.float64)]
    for h in range(1, harmonics + 1):
        vectors.append(np.sin(2.0 * np.pi * h * t))
        vectors.append(np.cos(2.0 * np.pi * h * t))
    basis = np.vstack(vectors).astype(np.float64)
    return BasisSystem(name='trigonometric', values=basis)


def l2_inner_product(phi_k: FloatArray, phi_l: FloatArray, t: FloatArray) -> float:
    return float(np.trapezoid(phi_k * phi_l, t))


def check_functional_linearity(
    x1: FloatArray,
    x2: FloatArray,
    phi: FloatArray,
    t: FloatArray,
    alpha: float = 0.7,
    beta: float = -1.2,
    atol: float = 1e-8,
) -> bool:
    left = float(np.trapezoid((alpha * x1 + beta * x2) * phi, t))
    right = alpha * float(np.trapezoid(x1 * phi, t)) + beta * float(np.trapezoid(x2 * phi, t))
    return bool(np.isclose(left, right, atol=atol))


def orthonormality_matrix(basis: BasisSystem, t: FloatArray) -> FloatArray:
    basis.validate(t)
    m = basis.values.shape[0]
    gram = np.zeros((m, m), dtype=np.float64)
    for i in range(m):
        for j in range(m):
            gram[i, j] = l2_inner_product(basis.values[i], basis.values[j], t)
    return gram
