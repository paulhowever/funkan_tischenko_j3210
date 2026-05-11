import numpy as np

from case_6.data import inject_outliers, make_sinusoidal_dataset, train_test_split
from case_6.lowess import lowess_fit_predict, lowess_predict_query
from case_6.metrics import rmse
from case_6.nadaraya_watson import nw_predict_variable


def test_lowess_returns_predictions_and_weights() -> None:
    x = np.linspace(-2.0, 2.0, 60)[:, None]
    y = np.sin(x[:, 0]) + 0.1 * np.random.default_rng(42).normal(size=60)
    pred, gamma = lowess_fit_predict(x, y, k=8, kernel_name="triangular")
    assert pred.shape == y.shape and gamma.shape == y.shape
    assert np.isfinite(pred).all() and np.isfinite(gamma).all()
    assert (gamma >= 0.0).all() and (gamma <= 1.0).all()


def test_lowess_constant_target_returns_constant() -> None:
    """Regression for prev_pred=0 bug: med~0 break must not zero predictions."""
    x = np.linspace(0, 1, 25)[:, None]
    y = np.full(25, 7.0)
    pred, _ = lowess_fit_predict(x, y, k=5, kernel_name="triangular")
    assert np.allclose(pred, 7.0)


def test_lowess_outlier_downweighted() -> None:
    """A single huge outlier in middle should receive gamma close to 0."""
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(-3, 3, 80))[:, None]
    y = np.sin(x[:, 0]) + 0.05 * rng.normal(size=80)
    out_idx = 40
    y[out_idx] += 10.0
    _, gamma = lowess_fit_predict(x, y, k=10, kernel_name="quartic")
    assert gamma[out_idx] < 0.1
    others = np.delete(gamma, out_idx)
    assert others.mean() > 0.5


def test_lowess_beats_nw_under_outliers() -> None:
    """With 15% outliers in train, LOWESS test RMSE should beat plain NW."""
    clean = make_sinusoidal_dataset(n_samples=240, noise_std=0.1, outlier_fraction=0.0, seed=1)
    tr, te = train_test_split(clean, test_size=0.25, seed=1)
    tr = inject_outliers(tr, fraction=0.15, scale=2.0, seed=2)

    pred_nw = nw_predict_variable(tr.x, tr.y, te.x, k=10, kernel_name="quartic")
    _, gamma = lowess_fit_predict(tr.x, tr.y, k=10, kernel_name="quartic")
    pred_low = lowess_predict_query(tr.x, tr.y, gamma, te.x, k=10, kernel_name="quartic")
    assert rmse(te.y, pred_low) < rmse(te.y, pred_nw)
