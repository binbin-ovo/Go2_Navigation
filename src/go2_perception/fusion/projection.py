import numpy as np
from go2_perception.core.transforms import transform_points


def project_points(points, calibration, min_depth=1e-6):
    """Return visible pixels, camera depths, and indices into input XYZ."""
    if min_depth <= 0:
        raise ValueError("min_depth must be positive")
    camera = transform_points(points, np.linalg.inv(calibration.T_lidar_camera))
    valid = np.isfinite(camera).all(axis=1) & (camera[:, 2] > min_depth)
    indices = np.flatnonzero(valid)
    camera = camera[valid]
    projected = camera @ calibration.K.T
    uv = projected[:, :2] / projected[:, 2, None]
    w, h = calibration.image_size
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    return uv[inside], camera[inside, 2], indices[inside]
