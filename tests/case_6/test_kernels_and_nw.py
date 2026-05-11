import numpy as np
import pytest

from case_6.distance import pairwise_euclidean
from case_6.kernels import KERNELS
from case_6.nadaraya_watson import nw_predict_fixed, nw_predict_variable


def test_kernels_nonnegative_and_compact() -> None:
    r = np.linspace(-2.0, 2.0, 101)
    for name, k in KERNELS.items():
        vals = k(r)
        assert np.all(vals >= 0.0), f"{name} produced negative values"
        if name in ("epanechnikov", "triangular", "quartic"):
            assert np.all(vals[np.abs(r) > 1.0] == 0.0), f"{name} should vanish outside |r|<=1"


def test_kernels_integrate_to_one() -> None:
    """All four kernels should be densities."""
    from scipy.integrate import quad
    for name, k in KERNELS.items():
        lo = -10.0 if name == "gaussian" else -1.0
        hi = 10.0 if name == "gaussian" else 1.0
        val, _ = quad(lambda r: float(k(np.array([r]))[0]), lo, hi)
        assert abs(val - 1.0) < 1e-3, f"{name} integrates to {val}, expected 1"


def test_nw_fixed_predict_shape() -> None:
    x_train = np.array([[-1.0], [0.0], [1.0], [2.0]])
    y_train = np.array([-1.0, 0.0, 1.0, 2.0])
    x_query = np.array([[0.5], [1.5]])
    pred = nw_predict_fixed(x_train, y_train, x_query, h=0.8, kernel_name="gaussian")
    assert pred.shape == (2,)
    assert np.isfinite(pred).all()


def test_nw_fixed_huge_h_converges_to_mean() -> None:
    """With h -> infinity Gaussian kernel ≈ 1 everywhere, NW -> mean(y_train)."""
    rng = np.random.default_rng(0)
    x_train = rng.uniform(-1.0, 1.0, size=(50, 1))
    y_train = rng.normal(0.0, 1.0, size=50)
    pred = nw_predict_fixed(x_train, y_train, np.array([[0.0], [0.5]]),
                            h=1e6, kernel_name="gaussian")
    assert np.allclose(pred, np.mean(y_train), atol=1e-6)


def test_nw_fixed_constant_y_predicts_constant() -> None:
    x_train = np.linspace(0, 1, 30)[:, None]
    y_train = np.full(30, 3.14)
    pred = nw_predict_fixed(x_train, y_train, np.array([[0.5], [0.0]]),
                            h=0.1, kernel_name="quartic")
    assert np.allclose(pred, 3.14)


def test_nw_empty_window_falls_back_to_mean() -> None:
    """With compact kernel and tiny h, all weights are 0 → fallback to mean(y_train)."""
    x_train = np.linspace(0, 1, 5)[:, None]
    y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    pred = nw_predict_fixed(x_train, y_train, np.array([[10.0]]),
                            h=0.01, kernel_name="epanechnikov")
    assert np.isclose(pred[0], 3.0)


def test_nw_variable_predict_shape() -> None:
    x_train = np.array([[-1.0], [0.0], [1.0], [2.0], [3.0]])
    y_train = np.array([-1.0, 0.0, 1.0, 2.0, 3.0])
    x_query = np.array([[0.5], [1.5]])
    pred = nw_predict_variable(x_train, y_train, x_query, k=2, kernel_name="epanechnikov")
    assert pred.shape == (2,)
    assert np.isfinite(pred).all()


def test_nw_invalid_h() -> None:
    x_train = np.zeros((3, 1)); y_train = np.zeros(3)
    with pytest.raises(ValueError):
        nw_predict_fixed(x_train, y_train, x_train, h=0.0, kernel_name="gaussian")


def test_nw_invalid_kernel() -> None:
    x_train = np.zeros((3, 1)); y_train = np.zeros(3)
    with pytest.raises(ValueError):
        nw_predict_fixed(x_train, y_train, x_train, h=1.0, kernel_name="nope")


def test_pairwise_euclidean_matches_norm() -> None:
    rng = np.random.default_rng(7)
    a = rng.normal(size=(4, 3)); b = rng.normal(size=(6, 3))
    d = pairwise_euclidean(a, b)
    assert d.shape == (6, 4)
    for i in range(b.shape[0]):
        for j in range(a.shape[0]):
            assert abs(d[i, j] - np.linalg.norm(b[i] - a[j])) < 1e-12
