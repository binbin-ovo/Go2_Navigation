"""Quantize in NumPy, evaluate cell eligibility once, then use native-int membership."""

import numpy as np
from .background import BackgroundMap


class FastBackgroundMap(BackgroundMap):
    def stable_mask(self, points, t):
        keys = np.floor(np.asarray(points) / self.size).astype(np.int64)
        stable = {
            k
            for k, e in self.cells.items()
            if e[0] >= self.min_observations
            and e[2] - e[1] >= self.min_span
            and t - e[2] <= self.ttl
        }
        return np.fromiter(
            (tuple(k) in stable for k in keys.tolist()), dtype=bool, count=len(keys)
        )
