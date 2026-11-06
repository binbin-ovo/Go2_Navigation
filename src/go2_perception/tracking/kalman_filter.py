"""Planar constant velocity with continuous white acceleration process noise."""

import numpy as np
from go2_perception.core.tracking_types import covariance, finite_array

H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])


def transition(dt, process_noise):
    if (
        not np.isfinite(dt)
        or dt < 0
        or not np.isfinite(process_noise)
        or process_noise < 0
    ):
        raise ValueError("Invalid prediction interval/noise")
    f = np.eye(4)
    f[0, 2] = f[1, 3] = dt
    q = np.zeros((4, 4))
    for p, v in ((0, 2), (1, 3)):
        q[p, p] = dt**3 / 3
        q[p, v] = q[v, p] = dt**2 / 2
        q[v, v] = dt
    return f, q * process_noise


class ConstantVelocityFilter:
    def __init__(
        self,
        position,
        measurement_covariance,
        initial_velocity_variance=4.0,
        process_noise=0.4,
        min_dt=1e-5,
        max_prediction_interval=0.5,
    ):
        if (
            not np.isfinite(
                [
                    initial_velocity_variance,
                    process_noise,
                    min_dt,
                    max_prediction_interval,
                ]
            ).all()
            or initial_velocity_variance <= 0
            or process_noise < 0
            or min_dt <= 0
            or max_prediction_interval < min_dt
        ):
            raise ValueError("Invalid filter parameters")
        self.state = np.r_[finite_array(position, (2,), "position"), [0.0, 0.0]]
        self.P = np.diag(
            [1.0, 1.0, initial_velocity_variance, initial_velocity_variance]
        )
        self.P[:2, :2] = covariance(measurement_covariance, 2)
        self.process_noise, self.min_dt, self.max_prediction_interval = (
            process_noise,
            min_dt,
            max_prediction_interval,
        )

    def predict(self, dt):
        if not np.isfinite(dt) or dt < self.min_dt:
            raise ValueError("Non-increasing or too-close timestamp")
        # Integrate all elapsed time in bounded steps, including the final remainder.
        steps = int(np.ceil(dt / self.max_prediction_interval))
        f, q = transition(dt / steps, self.process_noise)
        for _ in range(steps):
            self.state = f @ self.state
            self.P = f @ self.P @ f.T + q
        self.P = (self.P + self.P.T) / 2

    def innovation(self, position, measurement_covariance):
        r = covariance(measurement_covariance, 2)
        residual = finite_array(position, (2,), "measurement") - H @ self.state
        return residual, H @ self.P @ H.T + r

    def mahalanobis(self, position, measurement_covariance):
        residual, s = self.innovation(position, measurement_covariance)
        return float(residual @ np.linalg.solve(s, residual))

    def update(self, position, measurement_covariance):
        r = covariance(measurement_covariance, 2)
        residual, s = self.innovation(position, r)
        k = np.linalg.solve(s, H @ self.P).T
        self.state += k @ residual
        a = np.eye(4) - k @ H
        self.P = a @ self.P @ a.T + k @ r @ k.T  # Joseph form
        self.P = (self.P + self.P.T) / 2
