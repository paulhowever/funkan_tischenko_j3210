from __future__ import annotations

import numpy as np
from sklearn.datasets import fetch_california_housing, load_diabetes

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


def _standardize(x: FloatArray) -> FloatArray:
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    return ((x - mean) / std).astype(np.float64)


def load_diabetes_dataset(standardize: bool = True) -> TabularDataset:
    raw = load_diabetes()
    x = raw.data.astype(np.float64)
    y = raw.target.astype(np.float64)
    if standardize:
        x = _standardize(x)
    ds = TabularDataset(x=x, y=y)
    ds.validate()
    return ds


def load_california_dataset(standardize: bool = True, max_samples: int = 4000) -> TabularDataset:
    try:
        raw = fetch_california_housing()
        x = raw.data.astype(np.float64)
        y = raw.target.astype(np.float64)
    except Exception as exc:
        raise RuntimeError("Failed to load California Housing dataset") from exc
    if max_samples > 0 and max_samples < x.shape[0]:
        x = x[:max_samples]
        y = y[:max_samples]
    if standardize:
        x = _standardize(x)
    ds = TabularDataset(x=x, y=y)
    ds.validate()
    return ds
