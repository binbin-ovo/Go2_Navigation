import numpy as np


def validate_transform(transform):
    t = np.asarray(transform, dtype=float)
    if t.shape != (4, 4) or not np.isfinite(t).all():
        raise ValueError("Transform must be finite 4x4")
    if (
        not np.allclose(t[3], [0, 0, 0, 1])
        or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(t[:3, :3]), 1)
    ):
        raise ValueError("Transform must be rigid and right handed")
    return t


def transform_points(points, transform):
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("Expected (N,3) XYZ")
    t = validate_transform(transform)
    return p @ t[:3, :3].T + t[:3, 3]
