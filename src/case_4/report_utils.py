from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(slots=True)
class FunctionalDataset:
    t: FloatArray
    x: FloatArray
    y: FloatArray
    y_true: FloatArray


def trapezoid_integral(values: FloatArray, t: FloatArray) -> FloatArray:
    return np.trapezoid(values, t, axis=-1).astype(np.float64)


def make_functional_dataset(
    n_samples: int = 260,
    n_grid: int = 300,
    noise_x: float = 0.06,
    noise_y: float = 0.10,
    seed: int = 42,
) -> FunctionalDataset:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n_grid, dtype=np.float64)

    a = rng.normal(0.0, 1.0, size=n_samples)
    b = rng.normal(0.0, 1.0, size=n_samples)
    c = rng.normal(0.0, 0.8, size=n_samples)
    phase = rng.uniform(-0.3, 0.3, size=n_samples)

    x_clean = (
        a[:, None] * np.sin(2.0 * np.pi * t[None, :] + phase[:, None])
        + b[:, None] * np.cos(2.0 * np.pi * t[None, :] - 0.5 * phase[:, None])
        + c[:, None] * t[None, :]
    )
    x = x_clean + rng.normal(0.0, noise_x, size=x_clean.shape)

    y_true = (
        2.0 * trapezoid_integral(x * np.sin(2.0 * np.pi * t[None, :]), t)
        - 1.0 * trapezoid_integral(x * t[None, :], t)
        + 0.25 * (trapezoid_integral(x * np.cos(4.0 * np.pi * t[None, :]), t) ** 2)
    )
    y = y_true + rng.normal(0.0, noise_y, size=n_samples)

    return FunctionalDataset(t=t, x=x.astype(np.float64), y=y.astype(np.float64), y_true=y_true.astype(np.float64))


def split_indices(n_samples: int, test_size: float = 0.25, seed: int = 42) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be in (0, 1)")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_samples)
    split = int(n_samples * (1.0 - test_size))
    return idx[:split], idx[split:]


def make_indicator_basis(m: int, t: FloatArray) -> tuple[list[Callable[[FloatArray], FloatArray]], list[str]]:
    edges = np.linspace(0.0, 1.0, m + 1, dtype=np.float64)
    basis: list[Callable[[FloatArray], FloatArray]] = []
    names: list[str] = []
    for k in range(m):
        left = edges[k]
        right = edges[k + 1]

        def phi(tt: FloatArray, l: float = left, r: float = right) -> FloatArray:
            return ((tt >= l) & (tt < r)).astype(np.float64)

        basis.append(phi)
        names.append(f"indicator_{k}")
    names[-1] = f"indicator_{m-1}"
    return basis, names


def make_polynomial_basis(m: int) -> tuple[list[Callable[[FloatArray], FloatArray]], list[str]]:
    basis: list[Callable[[FloatArray], FloatArray]] = []
    names: list[str] = []
    for k in range(m):
        basis.append(lambda tt, p=k: np.power(tt, p, dtype=np.float64))
        names.append(f"poly_{k}")
    return basis, names


def make_trigonometric_basis(m: int) -> tuple[list[Callable[[FloatArray], FloatArray]], list[str]]:
    basis: list[Callable[[FloatArray], FloatArray]] = [lambda tt: np.ones_like(tt, dtype=np.float64)]
    names = ["trig_const"]
    harmonic = 1
    while len(basis) < m:
        basis.append(lambda tt, h=harmonic: np.sin(2.0 * np.pi * h * tt))
        names.append(f"sin_{harmonic}")
        if len(basis) >= m:
            break
        basis.append(lambda tt, h=harmonic: np.cos(2.0 * np.pi * h * tt))
        names.append(f"cos_{harmonic}")
        harmonic += 1
    return basis[:m], names[:m]


def compute_feature_matrix(
    x: FloatArray, t: FloatArray, basis: list[Callable[[FloatArray], FloatArray]]
) -> FloatArray:
    features = np.zeros((x.shape[0], len(basis)), dtype=np.float64)
    for k, phi in enumerate(basis):
        phi_values = phi(t)
        features[:, k] = trapezoid_integral(x * phi_values[None, :], t)
    return features


def fit_linear_regression(x: FloatArray, y: FloatArray, ridge_lambda: float = 0.0) -> FloatArray:
    x_aug = np.column_stack([np.ones(x.shape[0], dtype=np.float64), x])
    xtx = x_aug.T @ x_aug
    if ridge_lambda > 0.0:
        penalty = ridge_lambda * np.eye(xtx.shape[0], dtype=np.float64)
        penalty[0, 0] = 0.0
        xtx = xtx + penalty
    xty = x_aug.T @ y
    beta = np.linalg.solve(xtx + 1e-10 * np.eye(xtx.shape[0]), xty)
    return beta.astype(np.float64)


def predict_linear(x: FloatArray, beta: FloatArray) -> FloatArray:
    x_aug = np.column_stack([np.ones(x.shape[0], dtype=np.float64), x])
    return (x_aug @ beta).astype(np.float64)


def mse(y_true: FloatArray, y_pred: FloatArray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: FloatArray, y_pred: FloatArray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: FloatArray, y_pred: FloatArray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: FloatArray, y_pred: FloatArray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot <= 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def recover_weight_function(t: FloatArray, beta: FloatArray, basis: list[Callable[[FloatArray], FloatArray]]) -> FloatArray:
    w = np.zeros_like(t, dtype=np.float64)
    for coeff, phi in zip(beta[1:], basis):
        w = w + coeff * phi(t)
    return w.astype(np.float64)
