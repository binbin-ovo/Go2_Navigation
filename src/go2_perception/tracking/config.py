from dataclasses import dataclass
import numpy as np
from go2_perception.core.transforms import validate_transform


@dataclass
class TrackerConfig:
    mode: str = "odom"
    tracking_frame: str = "odom"
    sensor_frame: str = "base_link"
    base_frame: str = "base_link"
    clock_domain: str | None = (
        None  # bind on first valid update, never change thereafter
    )
    T_base_sensor: list | None = None
    pose_tolerance_s: float = 0.05
    timestamp_tolerance_s: float = 1e-6
    process_noise: float = 0.04  # white acceleration spectral density m^2/s^3
    measurement_noise: float = 0.01  # isotropic XYZ variance m^2
    initial_velocity_variance: float = 4.0
    min_dt: float = 1e-5
    max_prediction_interval: float = 0.5  # substep cap; never truncates elapsed time
    max_distance: float = 1.5
    mahalanobis_gate: float = 13.82  # squared 2-DOF gate
    max_vertical_distance: float = 1.0
    distance_weight: float = 0.4
    mahalanobis_weight: float = 0.6
    iou_weight: float = 0.0
    min_hits: int = 3
    confirmation_window_s: float = 1.0
    lost_after_misses: int = 2
    max_misses: int = 10
    max_unobserved_s: float = 1.5
    duplicate_distance: float = 0.02
    birth_suppression_distance: float = 0.3
    history_size: int = 40
    moving_speed: float = 0.3
    stationary_speed: float = 0.15
    motion_min_hits: int = 5
    motion_min_age_s: float = 0.5
    max_velocity_std: float = 0.3
    motion_sigma: float = 1.0
    trajectory_horizon_s: float = 3.0
    trajectory_step_s: float = 0.2

    def __post_init__(self):
        if (
            self.mode not in ("odom", "fixed_sensor")
            or not self.tracking_frame
            or not self.sensor_frame
            or not self.base_frame
        ):
            raise ValueError("Invalid tracking mode/frames")
        if self.clock_domain is not None and not self.clock_domain:
            raise ValueError("Empty clock domain")
        nonnegative = (
            "pose_tolerance_s",
            "timestamp_tolerance_s",
            "process_noise",
            "duplicate_distance",
            "birth_suppression_distance",
            "distance_weight",
            "mahalanobis_weight",
            "iou_weight",
            "motion_min_age_s",
            "motion_sigma",
        )
        positive = (
            "confirmation_window_s",
            "measurement_noise",
            "initial_velocity_variance",
            "min_dt",
            "max_prediction_interval",
            "max_distance",
            "mahalanobis_gate",
            "max_vertical_distance",
            "max_unobserved_s",
            "moving_speed",
            "stationary_speed",
            "max_velocity_std",
            "trajectory_horizon_s",
            "trajectory_step_s",
        )
        for key in nonnegative + positive:
            value = getattr(self, key)
            if not np.isfinite(value) or value < 0 or (key in positive and value == 0):
                raise ValueError(f"Invalid {key}")
        for key in (
            "min_hits",
            "lost_after_misses",
            "max_misses",
            "history_size",
            "motion_min_hits",
        ):
            value = getattr(self, key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"Invalid {key}")
        if (
            self.max_prediction_interval < self.min_dt
            or self.moving_speed <= self.stationary_speed
        ):
            raise ValueError("Invalid time bounds or motion thresholds")
        if (
            self.lost_after_misses > self.max_misses
            or self.trajectory_step_s > self.trajectory_horizon_s
        ):
            raise ValueError("Invalid lifecycle/horizon")
        if self.distance_weight + self.mahalanobis_weight + self.iou_weight <= 0:
            raise ValueError("Zero association weights")
        if self.T_base_sensor is not None:
            validate_transform(self.T_base_sensor)
