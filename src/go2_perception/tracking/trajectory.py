import numpy as np
from go2_perception.core.tracking_types import (
    TrajectoryPoint,
    finite_array,
    covariance as validate_covariance,
)
from .kalman_filter import transition


def predict_trajectory(
    state, covariance, z, timestamp, horizon=3.0, step=0.2, process_noise=0.4
):
    if not np.isfinite([z, timestamp, horizon, step]).all() or not 0 < step <= horizon:
        raise ValueError("Invalid trajectory time/height")
    state = finite_array(state, (4,), "state")
    covariance = validate_covariance(covariance, 4)
    result = []
    for dt in np.arange(step, horizon + 1e-9, step):
        f, q = transition(float(dt), process_noise)
        predicted = f @ state
        result.append(
            TrajectoryPoint(
                float(timestamp + dt),
                np.r_[predicted[:2], z],
                (f @ covariance @ f.T + q)[:2, :2],
            )
        )
    return result
