import numpy as np

from case_4.basis import (
    BasisSystem,
    check_functional_linearity,
    orthonormality_matrix,
    piecewise_indicator_basis,
)
from case_4.data import create_uniform_grid


def test_functional_linearity_holds() -> None:
    t = create_uniform_grid(200)
    x1 = np.sin(2.0 * np.pi * t)
    x2 = np.cos(2.0 * np.pi * t)
    phi = t
    assert check_functional_linearity(x1, x2, phi, t)


def test_orthonormality_matrix_shape() -> None:
    t = create_uniform_grid(100)
    basis = piecewise_indicator_basis(t, m=4)
    gram = orthonormality_matrix(basis, t)
    assert gram.shape == (4, 4)
    assert np.all(np.diag(gram) > 0)


def test_basis_validation() -> None:
    t = create_uniform_grid(20)
    basis = BasisSystem(name="broken", values=np.ones((2, 25), dtype=np.float64))
    try:
        basis.validate(t)
    except ValueError:
        assert True
    else:
        assert False
