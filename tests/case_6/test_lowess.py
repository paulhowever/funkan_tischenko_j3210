import numpy as np

from case_6.lowess import lowess_fit_predict, lowess_predict_query


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


def test_lowess_predict_query_shape() -> None:
    x = np.linspace(-2.0, 2.0, 60)[:, None]
    y = np.sin(x[:, 0]) + 0.1 * np.random.default_rng(1).normal(size=60)
    pred_train, _ = lowess_fit_predict(x, y, k=8, kernel_name='triangular')
    x_query = np.array([[-1.5], [0.0], [1.3]])
    pred_query = lowess_predict_query(x, pred_train, x_query, k=8, kernel_name='triangular')
    assert pred_query.shape == (3,)
    assert np.isfinite(pred_query).all()
