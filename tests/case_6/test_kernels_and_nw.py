import numpy as np

from case_6.nadaraya_watson import nw_predict_fixed, nw_predict_variable


def test_nw_fixed_predict_shape() -> None:
    x_train = np.array([[-1.0], [0.0], [1.0], [2.0]])
    y_train = np.array([-1.0, 0.0, 1.0, 2.0])
    x_query = np.array([[0.5], [1.5]])

    pred = nw_predict_fixed(x_train, y_train, x_query, h=0.8, kernel_name='gaussian')
    assert pred.shape == (2,)
    assert np.isfinite(pred).all()


def test_nw_variable_predict_shape() -> None:
    x_train = np.array([[-1.0], [0.0], [1.0], [2.0], [3.0]])
    y_train = np.array([-1.0, 0.0, 1.0, 2.0, 3.0])
    x_query = np.array([[0.5], [1.5]])

    pred = nw_predict_variable(x_train, y_train, x_query, k=2, kernel_name='epanechnikov')
    assert pred.shape == (2,)
    assert np.isfinite(pred).all()
