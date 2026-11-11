"""Camera intrinsics and rigid extrinsics with explicit provenance for the Go2 VLN interface.

Conventions (all enforced, see docs/GO2_VLN_INPUT_INTERFACE.md):
- Camera optical frame: x right, y down, z forward; pixel (0, 0) is the centre of the top-left pixel.
- ``T_target_source`` maps column vectors from ``source`` into ``target``:
  ``p_target = T[:3, :3] @ p_source + T[:3, 3]``. Translation in metres.
- ``status`` is ``measured`` only for calibration estimated from real Go2 data; ``synthetic`` is for
  tests and fixtures and is refused wherever real geometry is required.
"""

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

from ..core.transforms import validate_transform

CALIBRATION_SCHEMA = "go2-camera-calibration-v1"
MODELS = {"pinhole_radtan": 5, "fisheye_equidistant": 4}
STATUSES = {"measured", "synthetic"}


@dataclass(frozen=True)
class CameraIntrinsics:
    """Intrinsics of the image they were estimated on (raw, distorted Go2 frames)."""

    width: int
    height: int
    K: np.ndarray
    model: str
    D: np.ndarray
    status: str
    camera_frame: str = "go2_front_camera_optical"
    reprojection_rms_px: float | None = None
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        K = np.asarray(self.K, dtype=float)
        D = np.asarray(self.D, dtype=float).reshape(-1)
        if K.shape != (3, 3) or not np.isfinite(K).all():
            raise ValueError("K must be a finite 3x3 matrix")
        if K[0, 0] <= 0 or K[1, 1] <= 0 or not np.allclose(K[2], [0, 0, 1]) or K[1, 0] or K[2, 0]:
            raise ValueError("K must be upper triangular with positive focal lengths and K[2]=[0,0,1]")
        if self.model not in MODELS:
            raise ValueError(f"Unknown camera model {self.model!r}; expected one of {sorted(MODELS)}")
        if D.shape != (MODELS[self.model],) or not np.isfinite(D).all():
            raise ValueError(f"{self.model} needs {MODELS[self.model]} finite distortion coefficients")
        if self.status not in STATUSES:
            raise ValueError(f"status must be one of {sorted(STATUSES)}")
        if int(self.width) <= 0 or int(self.height) <= 0:
            raise ValueError("Image size must be positive")
        object.__setattr__(self, "K", K)
        object.__setattr__(self, "D", D)

    def scaled(self, new_width, new_height):
        """Intrinsics after resizing the whole image (no crop); distortion is unchanged."""
        sx, sy = new_width / self.width, new_height / self.height
        K = self.K.copy()
        K[0, :] *= sx  # fx, skew, cx
        K[1, 1:] *= sy  # fy, cy
        # Pixel-centre convention: a resize maps centre c to (c + 0.5) * s - 0.5.
        K[0, 2] = (self.K[0, 2] + 0.5) * sx - 0.5
        K[1, 2] = (self.K[1, 2] + 0.5) * sy - 0.5
        return _replace(self, width=int(new_width), height=int(new_height), K=K)

    def cropped(self, x0, y0, new_width, new_height):
        K = self.K.copy()
        K[0, 2] -= x0
        K[1, 2] -= y0
        return _replace(self, width=int(new_width), height=int(new_height), K=K)


@dataclass(frozen=True)
class RigidTransform:
    """``matrix`` = T_target_source. ``status``: measured, firmware, synthetic."""

    matrix: np.ndarray
    source_frame: str
    target_frame: str
    status: str
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.source_frame or not self.target_frame or self.source_frame == self.target_frame:
            raise ValueError("Transform needs two distinct named frames")
        if self.status not in {"measured", "firmware", "synthetic"}:
            raise ValueError("Transform status must be measured, firmware or synthetic")
        object.__setattr__(self, "matrix", validate_transform(self.matrix))

    def apply(self, points):
        p = np.asarray(points, dtype=float)
        if p.ndim != 2 or p.shape[1] != 3:
            raise ValueError("Expected (N,3) XYZ")
        return p @ self.matrix[:3, :3].T + self.matrix[:3, 3]

    def inverse(self):
        m = np.eye(4)
        m[:3, :3] = self.matrix[:3, :3].T
        m[:3, 3] = -self.matrix[:3, :3].T @ self.matrix[:3, 3]
        return RigidTransform(m, self.target_frame, self.source_frame, self.status, self.provenance)

    def compose(self, other):
        """self (B<-A) composed with other (A<-C) gives B<-C."""
        if other.target_frame != self.source_frame:
            raise ValueError(f"Cannot compose {self.target_frame}<-{self.source_frame} with "
                             f"{other.target_frame}<-{other.source_frame}")
        weakest = "synthetic" if "synthetic" in (self.status, other.status) else (
            "firmware" if "firmware" in (self.status, other.status) else "measured")
        return RigidTransform(self.matrix @ other.matrix, other.source_frame, self.target_frame,
                              weakest, {"composed": [self.provenance, other.provenance]})


def _replace(intr, **changes):
    values = {k: getattr(intr, k) for k in intr.__dataclass_fields__}
    values.update(changes)
    return CameraIntrinsics(**values)


def project_points(points_camera, intrinsics, min_depth_m=0.05):
    """Project camera-frame XYZ (metres) through the distortion model.

    Returns ``(uv, valid)``: uv is (N, 2) float pixels (NaN where invalid); valid requires depth
    above ``min_depth_m`` and a landing inside the image. Points behind the camera are never
    projected, which the raw OpenCV call would otherwise do silently.
    """
    p = np.asarray(points_camera, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("Expected (N,3) camera-frame XYZ")
    uv = np.full((len(p), 2), np.nan)
    front = np.isfinite(p).all(axis=1) & (p[:, 2] > min_depth_m)
    if front.any():
        obj = p[front].reshape(-1, 1, 3)
        zero = np.zeros(3)
        if intrinsics.model == "fisheye_equidistant":
            img, _ = cv2.fisheye.projectPoints(obj, zero, zero, intrinsics.K, intrinsics.D)
        else:
            img, _ = cv2.projectPoints(obj, zero, zero, intrinsics.K, intrinsics.D)
        uv[front] = img.reshape(-1, 2)
    inside = (uv[:, 0] >= -0.5) & (uv[:, 0] < intrinsics.width - 0.5) & \
             (uv[:, 1] >= -0.5) & (uv[:, 1] < intrinsics.height - 0.5)
    valid = front & inside
    uv[~valid] = np.nan
    return uv, valid


def load_calibration(path):
    """Load ``go2-camera-calibration-v1`` YAML -> (CameraIntrinsics, RigidTransform | None)."""
    data = yaml.safe_load(Path(path).read_text())
    if data.get("schema") != CALIBRATION_SCHEMA:
        raise ValueError(f"{path}: expected schema {CALIBRATION_SCHEMA}")
    intr = CameraIntrinsics(
        width=data["image_size"][0], height=data["image_size"][1], K=data["K"],
        model=data["model"], D=data["D"], status=data["status"],
        camera_frame=data.get("camera_frame", "go2_front_camera_optical"),
        reprojection_rms_px=data.get("reprojection_rms_px"), provenance=data.get("provenance", {}))
    ext = None
    if data.get("T_base_camera") is not None:
        e = data["T_base_camera"]
        ext = RigidTransform(e["matrix"], intr.camera_frame, "base_link", e["status"],
                             e.get("provenance", {}))
        if ext.status == "measured" and intr.status != "measured":
            raise ValueError("Measured extrinsics cannot rest on non-measured intrinsics")
    return intr, ext


def save_calibration(path, intrinsics, T_base_camera=None):
    data = {
        "schema": CALIBRATION_SCHEMA, "status": intrinsics.status,
        "camera_frame": intrinsics.camera_frame,
        "image_size": [int(intrinsics.width), int(intrinsics.height)],
        "model": intrinsics.model, "K": intrinsics.K.tolist(), "D": intrinsics.D.tolist(),
        "reprojection_rms_px": intrinsics.reprojection_rms_px,
        "units": {"K": "pixels", "translation": "metres"},
        "provenance": intrinsics.provenance,
        "T_base_camera": None if T_base_camera is None else {
            "matrix": T_base_camera.matrix.tolist(), "status": T_base_camera.status,
            "direction": "p_base = T @ p_camera", "provenance": T_base_camera.provenance},
    }
    Path(path).write_text(yaml.safe_dump(data, sort_keys=False))
