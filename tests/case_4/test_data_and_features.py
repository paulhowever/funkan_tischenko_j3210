import numpy as np

from case_4.basis import piecewise_indicator_basis
from case_4.data import SyntheticConfig, make_synthetic_dataset
from case_4.features import build_feature_matrix


def test_synthetic_dataset_shapes() -> None:
    ds = make_synthetic_dataset(SyntheticConfig(n_samples=20, n_grid=50, seed=1))
    assert ds.t.shape == (50,)
    assert ds.x.shape == (20, 50)
    assert ds.y.shape == (20,)


def test_feature_matrix_shape() -> None:
    ds = make_synthetic_dataset(SyntheticConfig(n_samples=12, n_grid=40, seed=2))
    basis = piecewise_indicator_basis(ds.t, m=5)
    z = build_feature_matrix(ds, basis)
    assert z.shape == (12, 5)
    assert np.isfinite(z).all()
