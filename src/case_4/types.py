from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(slots=True)
class FunctionalDataset:
    t: FloatArray
    x: FloatArray
    y: FloatArray

    def validate(self) -> None:
        if self.t.ndim != 1:
            raise ValueError('t must be 1D')
        if self.x.ndim != 2:
            raise ValueError('x must be 2D (n_samples, n_grid)')
        if self.y.ndim != 1:
            raise ValueError('y must be 1D')
        if self.x.shape[0] != self.y.shape[0]:
            raise ValueError('x and y must have equal number of samples')
        if self.x.shape[1] != self.t.shape[0]:
            raise ValueError('x grid dimension must match t length')
