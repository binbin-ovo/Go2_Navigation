from dataclasses import dataclass
from pathlib import Path
import numpy as np
import yaml
from go2_perception.core.transforms import validate_transform


@dataclass
class Calibration:
    K: np.ndarray
    T_lidar_camera: np.ndarray  # camera -> lidar; inverse used for projection
    image_size: tuple[int, int]
    camera_frame: str = "camera_optical"
    lidar_frame: str = "lidar"
    distortion: tuple = ()
    rectified: bool = True
    synthetic: bool = True

    def __post_init__(self):
        self.K = np.asarray(self.K, dtype=float)
        self.T_lidar_camera = validate_transform(self.T_lidar_camera)
        self.image_size = tuple(self.image_size)
        if (
            self.K.shape != (3, 3)
            or not np.isfinite(self.K).all()
            or self.K[0, 0] <= 0
            or self.K[1, 1] <= 0
            or not np.allclose(self.K[2], [0, 0, 1])
        ):
            raise ValueError("Invalid camera intrinsics")
        if len(self.image_size) != 2 or any(
            int(v) != v or v <= 0 for v in self.image_size
        ):
            raise ValueError("image_size must be positive integer width,height")
        if not np.isfinite(self.distortion).all():
            raise ValueError("Invalid distortion")
        if not self.rectified or np.any(np.asarray(self.distortion) != 0):
            raise ValueError(
                "Supply rectified image and corresponding K; raw distortion unsupported"
            )

    @classmethod
    def from_yaml(cls, path):
        return cls(**yaml.safe_load(Path(path).read_text()))
