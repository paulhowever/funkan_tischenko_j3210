from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(slots=True)
class TabularDataset:
    x: FloatArray
    y: FloatArray

    def validate(self) -> None:
        if self.x.ndim != 2:
            raise ValueError('x must be 2D')
        if self.y.ndim != 1:
            raise ValueError('y must be 1D')
        if self.x.shape[0] != self.y.shape[0]:
            raise ValueError('x and y sample counts must match')
