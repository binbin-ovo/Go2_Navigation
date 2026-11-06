"""Public result types. Coordinates in metres; timestamps in seconds."""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class Detection2D:
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]


@dataclass
class RGBResult:
    detections: list[Detection2D]
    image_size: tuple[int, int]  # width, height
    latency_ms: float = 0.0
    timestamp: float | None = None
    frame_id: str = "camera_optical"
    clock_domain: str | None = None


@dataclass
class Obstacle:
    cluster_id: int
    points: np.ndarray
    center: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray


@dataclass
class LidarResult:
    points: np.ndarray  # filtered/downsampled XYZ
    ground: np.ndarray
    unclassified: np.ndarray
    obstacles: list[Obstacle]
    labels: np.ndarray  # per processed point: -2 ground, -1 unclassified, >=0 cluster
    latency_ms: float = 0.0
    timestamp: float | None = None
    frame_id: str = "lidar"
    clock_domain: str | None = None


@dataclass
class FusedDetection:
    class_id: int
    class_name: str
    confidence: float
    cluster_id: int
    position: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    timestamp: float | None
    rgb_timestamp: float | None
    frame_id: str
    clock_domain: str | None
    association_score: float


@dataclass
class FusionResult:
    detections: list[FusedDetection]
    validation: str
    unmatched_rgb: list[int] = field(default_factory=list)
    unmatched_clusters: list[int] = field(default_factory=list)
