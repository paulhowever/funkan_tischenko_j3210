import numpy as np

from case_6.lowess import lowess_fit_predict


def test_lowess_returns_predictions_and_weights() -> None:
    x = np.linspace(-2.0, 2.0, 60)[:, None]
    y = np.sin(x[:, 0]) + 0.1 * np.random.default_rng(42).normal(size=60)

    pred, gamma = lowess_fit_predict(x, y, k=8, kernel_name='triangular')
    assert pred.shape == y.shape
    assert gamma.shape == y.shape
    assert np.isfinite(pred).all()
    assert np.isfinite(gamma).all()
    assert (gamma >= 0.0).all()
    assert (gamma <= 1.0).all()
