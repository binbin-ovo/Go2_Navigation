"""Tracking contracts: metres, seconds, m/s; covariance ordering is explicit."""

from dataclasses import dataclass, field
import numpy as np
from go2_perception.core.transforms import validate_transform


def finite_array(value, shape, name):
    a = np.asarray(value, dtype=float)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return a.copy()


def covariance(value, size):
    a = finite_array(value, (size, size), "covariance")
    if not np.allclose(a, a.T, atol=1e-10) or np.linalg.eigvalsh(a).min() <= 0:
        raise ValueError("Covariance must be symmetric positive definite")
    return a


def metadata(timestamp, frame_id, clock_domain):
    if (
        timestamp is None
        or not np.isfinite(timestamp)
        or not frame_id
        or not clock_domain
    ):
        raise ValueError("Finite seconds timestamp and explicit frame/clock required")


@dataclass
class Detection3D:
    timestamp: float
    position: np.ndarray
    frame_id: str
    clock_domain: str
    detection_id: str | None = None
    bbox_min: np.ndarray | None = None
    bbox_max: np.ndarray | None = None
    class_name: str | None = None
    confidence: float | None = (
        None  # None for geometric clusters, not fabricated probability
    )
    measurement_covariance: np.ndarray | None = None  # XYZ, m^2

    def __post_init__(self):
        metadata(self.timestamp, self.frame_id, self.clock_domain)
        self.position = finite_array(self.position, (3,), "position")
        if (self.bbox_min is None) != (self.bbox_max is None):
            raise ValueError("Both box limits required")
        if self.bbox_min is not None:
            self.bbox_min = finite_array(self.bbox_min, (3,), "bbox_min")
            self.bbox_max = finite_array(self.bbox_max, (3,), "bbox_max")
            if np.any(self.bbox_max < self.bbox_min):
                raise ValueError("Reversed box bounds")
        if self.confidence is not None and (
            not np.isfinite(self.confidence) or not 0 <= self.confidence <= 1
        ):
            raise ValueError("Confidence must be in [0,1] or None")
        if self.measurement_covariance is not None:
            self.measurement_covariance = covariance(self.measurement_covariance, 3)


@dataclass
class RobotPose:
    timestamp: float
    T_parent_child: np.ndarray
    parent_frame: str
    child_frame: str
    clock_domain: str

    def __post_init__(self):
        metadata(self.timestamp, self.parent_frame, self.clock_domain)
        if not self.child_frame or self.child_frame == self.parent_frame:
            raise ValueError("Pose requires distinct parent/child frames")
        self.T_parent_child = validate_transform(self.T_parent_child).copy()


@dataclass
class TrajectoryPoint:
    timestamp: float
    position: np.ndarray  # XYZ, z held at last measured value
    position_covariance: np.ndarray  # XY covariance, m^2


@dataclass
class TrackOutput:
    track_id: int
    timestamp: float
    frame_id: str
    clock_domain: str
    velocity_reference: str  # odometry or fixed_sensor_relative
    position: np.ndarray
    velocity: np.ndarray  # vx,vy
    state_covariance: np.ndarray  # x,y,vx,vy
    bbox_min: np.ndarray | None
    bbox_max: np.ndarray | None
    class_name: str | None
    detection_confidence: float | None
    status: str
    motion_state: str
    last_observation_timestamp: float
    age_s: float
    hits: int
    consecutive_misses: int
    history: list[dict] = field(default_factory=list)
    predicted_positions: list[TrajectoryPoint] = field(default_factory=list)
