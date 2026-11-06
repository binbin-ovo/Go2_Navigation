import numpy as np


def aabb(points):
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) == 0 or not np.isfinite(p).all():
        raise ValueError("AABB requires nonempty finite (N,3)")
    low, high = p.min(axis=0), p.max(axis=0)
    return low, high, (low + high) / 2
