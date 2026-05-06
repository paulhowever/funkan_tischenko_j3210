from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from case_4.types import FloatArray, FunctionalDataset


@dataclass(slots=True)
class SyntheticConfig:
    n_samples: int = 300
    n_grid: int = 200
    t_start: float = 0.0
    t_end: float = 1.0
    noise_x_std: float = 0.05
    noise_y_std: float = 0.05
    seed: int = 42


def create_uniform_grid(n_grid: int, t_start: float = 0.0, t_end: float = 1.0) -> FloatArray:
    if n_grid < 2:
        raise ValueError('n_grid must be >= 2')
    return np.linspace(t_start, t_end, n_grid, dtype=np.float64)


def _generate_functions(t: FloatArray, n_samples: int, rng: np.random.Generator, noise_x_std: float) -> FloatArray:
    a = rng.normal(0.0, 1.0, size=n_samples)
    b = rng.normal(0.0, 1.0, size=n_samples)
    c = rng.normal(0.0, 0.5, size=n_samples)
    noise = rng.normal(0.0, noise_x_std, size=(n_samples, t.size))

    x = (
        a[:, None] * np.sin(2.0 * np.pi * t)[None, :]
        + b[:, None] * np.cos(2.0 * np.pi * t)[None, :]
        + c[:, None] * t[None, :]
        + noise
    )
    return x.astype(np.float64)


def _integral(values: FloatArray, t: FloatArray) -> FloatArray:
    return np.trapezoid(values, t, axis=1).astype(np.float64)


def _generate_targets(x: FloatArray, t: FloatArray, rng: np.random.Generator, noise_y_std: float) -> FloatArray:
    term_1 = 2.0 * _integral(x * np.sin(2.0 * np.pi * t)[None, :], t)
    term_2 = _integral(x * t[None, :], t)
    eta = rng.normal(0.0, noise_y_std, size=x.shape[0])
    return (term_1 - term_2 + eta).astype(np.float64)


def make_synthetic_dataset(config: SyntheticConfig = SyntheticConfig()) -> FunctionalDataset:
    rng = np.random.default_rng(config.seed)
    t = create_uniform_grid(config.n_grid, config.t_start, config.t_end)
    x = _generate_functions(t, config.n_samples, rng, config.noise_x_std)
    y = _generate_targets(x, t, rng, config.noise_y_std)
    dataset = FunctionalDataset(t=t, x=x, y=y)
    dataset.validate()
    return dataset


def train_test_split(dataset: FunctionalDataset, test_size: float = 0.25, seed: int = 42) -> tuple[FunctionalDataset, FunctionalDataset]:
    if not (0.0 < test_size < 1.0):
        raise ValueError('test_size must be in (0, 1)')

    dataset.validate()
    rng = np.random.default_rng(seed)
    indices = rng.permutation(dataset.x.shape[0])
    split = int(dataset.x.shape[0] * (1.0 - test_size))

    train_idx = indices[:split]
    test_idx = indices[split:]

    train = FunctionalDataset(t=dataset.t, x=dataset.x[train_idx], y=dataset.y[train_idx])
    test = FunctionalDataset(t=dataset.t, x=dataset.x[test_idx], y=dataset.y[test_idx])
    train.validate()
    test.validate()
    return train, test
