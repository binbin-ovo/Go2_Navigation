"""Transport-independent messages; no ROS dependency."""

from dataclasses import dataclass
import numpy as np
from .image_loader import load_image
from .pointcloud_loader import validate_points


@dataclass
class RGBMessage:
    image: np.ndarray
    timestamp: float
    width: int
    height: int
    K: np.ndarray
    frame_id: str
    clock_domain: str

    def __post_init__(self):
        self.image = load_image(self.image)
        self.K = np.asarray(self.K, dtype=float)
        if self.image.shape[:2] != (self.height, self.width):
            raise ValueError("Image dimensions mismatch")
        if self.K.shape != (3, 3) or not np.isfinite(self.K).all():
            raise ValueError("Invalid K")
        _metadata(self.timestamp, self.frame_id, self.clock_domain)


@dataclass
class LidarMessage:
    points: np.ndarray
    timestamp: float
    frame_id: str
    clock_domain: str

    def __post_init__(self):
        self.points = validate_points(self.points)
        _metadata(self.timestamp, self.frame_id, self.clock_domain)


def _metadata(timestamp, frame_id, clock_domain):
    if not np.isfinite(timestamp) or not frame_id or not clock_domain:
        raise ValueError(
            "Finite seconds timestamp and named frame/clock domain required"
        )


def process_rgb(detector, message):
    return detector.detect(
        message.image,
        timestamp=message.timestamp,
        frame_id=message.frame_id,
        clock_domain=message.clock_domain,
    )


def process_lidar(processor, message):
    return processor.process(
        message.points,
        timestamp=message.timestamp,
        frame_id=message.frame_id,
        clock_domain=message.clock_domain,
    )
