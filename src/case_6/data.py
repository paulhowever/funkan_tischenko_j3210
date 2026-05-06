from __future__ import annotations

import numpy as np

from case_6.types import FloatArray, TabularDataset


def make_sinusoidal_dataset(
    n_samples: int = 200,
    x_min: float = -3.0,
    x_max: float = 3.0,
    noise_std: float = 0.15,
    outlier_fraction: float = 0.0,
    seed: int = 42,
) -> TabularDataset:
    rng = np.random.default_rng(seed)
    x = rng.uniform(x_min, x_max, size=n_samples)
    y_true = np.sin(x)
    y = y_true + rng.normal(0.0, noise_std, size=n_samples)

    if outlier_fraction > 0.0:
        n_out = max(1, int(n_samples * outlier_fraction))
        idx = rng.choice(n_samples, size=n_out, replace=False)
        y[idx] += rng.normal(0.0, 2.0, size=n_out)

    order = np.argsort(x)
    x_sorted = x[order][:, None].astype(np.float64)
    y_sorted = y[order].astype(np.float64)

    ds = TabularDataset(x=x_sorted, y=y_sorted)
    ds.validate()
    return ds


def train_test_split(dataset: TabularDataset, test_size: float = 0.25, seed: int = 42) -> tuple[TabularDataset, TabularDataset]:
    if not (0.0 < test_size < 1.0):
        raise ValueError('test_size must be in (0, 1)')

    dataset.validate()
    rng = np.random.default_rng(seed)
    idx = rng.permutation(dataset.x.shape[0])
    split = int(dataset.x.shape[0] * (1.0 - test_size))

    train = TabularDataset(x=dataset.x[idx[:split]], y=dataset.y[idx[:split]])
    test = TabularDataset(x=dataset.x[idx[split:]], y=dataset.y[idx[split:]])
    train.validate()
    test.validate()
    return train, test
