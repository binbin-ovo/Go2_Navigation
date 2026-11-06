import numpy as np


def classify_motion(
    velocity, velocity_covariance, hits, age_s, misses, config, previous="UNCERTAIN"
):
    if hits < config.motion_min_hits or age_s < config.motion_min_age_s or misses:
        return "UNCERTAIN"
    speed = float(np.linalg.norm(velocity))
    std = float(np.sqrt(max(0, np.linalg.eigvalsh(velocity_covariance).max())))
    if std > config.max_velocity_std:
        return "UNCERTAIN"
    low, high = (
        max(0, speed - config.motion_sigma * std),
        speed + config.motion_sigma * std,
    )
    if low >= config.moving_speed:
        return "MOVING"
    if high <= config.stationary_speed:
        return "STATIONARY"
    if previous == "MOVING" and low > config.stationary_speed:
        return "MOVING"
    if previous == "STATIONARY" and high < config.moving_speed:
        return "STATIONARY"
    return "UNCERTAIN"
