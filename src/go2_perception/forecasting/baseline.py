import numpy as np
from go2_perception.tracking.trajectory import predict_trajectory


def constant_velocity(observed, pred_steps, dt, method="mean", recent_intervals=3):
    x = np.asarray(observed, dtype=float)
    if (
        x.ndim != 3
        or x.shape[0] < 1
        or x.shape[-1] != 2
        or x.shape[1] < 2
        or not np.isfinite(x).all()
        or not np.isfinite(dt)
        or dt <= 0
        or pred_steps < 1
        or recent_intervals < 1
    ):
        raise ValueError("Invalid constant-velocity input")
    if method not in ("last", "mean"):
        raise ValueError("Unknown velocity method")
    n = 1 if method == "last" else min(recent_intervals, x.shape[1] - 1)
    velocity = (x[:, -1] - x[:, -1 - n]) / (n * dt)
    return (
        x[:, -1, None, :]
        + np.arange(1, pred_steps + 1)[None, :, None] * dt * velocity[:, None, :]
    )


def phase3_constant_velocity(observed, pred_steps, dt, **kwargs):
    first = constant_velocity(observed, 1, dt, **kwargs)[:, 0]
    velocity = (first - np.asarray(observed)[:, -1]) / dt
    return np.array(
        [
            [
                p.position[:2]
                for p in predict_trajectory(
                    np.r_[x[-1], v],
                    np.eye(4),
                    0,
                    0,
                    horizon=pred_steps * dt,
                    step=dt,
                    process_noise=0,
                )
            ]
            for x, v in zip(observed, velocity)
        ]
    )
