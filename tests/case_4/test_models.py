import numpy as np

from case_4.models import mse, ols_fit_predict, r2, ridge_fit_predict


def test_ols_recovers_simple_linear_relation() -> None:
    x = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    y = np.array([1.0, 3.0, 5.0, 7.0])
    result = ols_fit_predict(x, y)
    assert np.allclose(result.beta, np.array([1.0, 2.0]), atol=1e-8)
    assert mse(y, result.y_pred) < 1e-12
    assert r2(y, result.y_pred) > 0.999999


def test_ridge_keeps_prediction_finite() -> None:
    x = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    y = np.array([1.0, 3.0, 5.0, 7.0])
    result = ridge_fit_predict(x, y, lam=1.0)
    assert np.isfinite(result.beta).all()
    assert np.isfinite(result.y_pred).all()
