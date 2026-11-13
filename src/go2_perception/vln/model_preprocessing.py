"""Calibrated image views for offline models; never creates camera-to-robot geometry."""

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .calibration import CameraIntrinsics
from .observation import Go2VLNObservation


@dataclass
class CalibratedView:
    image: np.ndarray                  # RGB uint8; normalization belongs to model processor
    intrinsics: CameraIntrinsics      # pinhole with zero distortion, same optical axes
    valid_mask: np.ndarray             # remap support, not traversability
    metadata: dict


def calibrated_view(obs: Go2VLNObservation, *, allow_synthetic=False) -> CalibratedView:
    """Undistort using observation calibration, preserving the original observation.

    No spatial crop/resize here. Fisheye balance=0; R=I means unchanged *optical*
    axes, not a replacement for T_base_camera. Already rectified input must carry
    pinhole K and zero D and is copied without a second remap.
    """
    v = obs.visual
    if v is None:
        raise ValueError("RGB observation is missing")
    if v.image.dtype != np.uint8 or v.image.ndim != 3 or v.image.shape[2] != 3:
        raise ValueError("Expected unnormalized uint8 HxWx3 RGB")
    if v.intrinsics_status != "measured" and not (allow_synthetic and v.intrinsics_status == "synthetic"):
        raise ValueError("Measured image intrinsics required (synthetic only by explicit opt-in)")
    if v.intrinsics_size is None or tuple(v.intrinsics_size) != (v.width, v.height):
        raise ValueError("Intrinsics size does not match source image; no implicit scaling")
    if v.K is None or v.D is None:
        raise ValueError("Missing K or D")
    intr = CameraIntrinsics(v.width, v.height, v.K, v.distortion_model, v.D,
                            v.intrinsics_status, camera_frame=v.frame_id)
    size = (v.width, v.height)
    if v.rectified:
        if intr.model != "pinhole_radtan" or np.any(intr.D != 0):
            raise ValueError("Rectified input requires pinhole intrinsics with zero distortion")
        K = intr.K.copy()
        img = v.image.copy()
        valid = np.ones(v.image.shape[:2], dtype=bool)
        method = "already_rectified_copy"
    else:
        if intr.model == "fisheye_equidistant":
            K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                intr.K, intr.D, size, np.eye(3), balance=0.0, new_size=size)
            mx, my = cv2.fisheye.initUndistortRectifyMap(
                intr.K, intr.D, np.eye(3), K, size, cv2.CV_32FC1)
            method = "opencv_fisheye_balance0_same_optical_axes"
        else:
            K, _ = cv2.getOptimalNewCameraMatrix(intr.K, intr.D, size, 0, size)
            mx, my = cv2.initUndistortRectifyMap(intr.K, intr.D, np.eye(3), K, size, cv2.CV_32FC1)
            method = "opencv_radtan_alpha0_same_optical_axes"
        img = cv2.remap(v.image, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        valid = (mx >= 0) & (mx <= v.width - 1) & (my >= 0) & (my <= v.height - 1)
    out = CameraIntrinsics(v.width, v.height, K, "pinhole_radtan", np.zeros(5),
                           intr.status, camera_frame=intr.camera_frame)
    meta = {
        "source": {"frame_id": v.frame_id, "size_wh": list(size), "rectified": v.rectified,
                   "intrinsics_status": intr.status, "K": intr.K.tolist(), "D": intr.D.tolist(),
                   "distortion_model": intr.model, "stamp_host": asdict(v.stamp_host),
                   "stamp_robot": None if v.stamp_robot is None else asdict(v.stamp_robot),
                   "clock_epoch": obs.clock_epoch},
        "view": {"frame_id": "rectified_image", "optical_axes": "x right, y down, z forward",
                 "size_wh": list(size), "rectified": True, "K": K.tolist(),
                 "D": [0.0] * 5, "distortion_model": "pinhole_radtan", "method": method,
                 "crop_xywh": [0, 0, *size], "resize_scale_xy": [1.0, 1.0],
                 "padding_ltrb": [0, 0, 0, 0], "valid_fraction": float(valid.mean())},
        "history_frames_available": len(v.history), "history_frames_consumed": 0,
        "normalization_before_processor": "none (RGB uint8)",
    }
    return CalibratedView(img, out, valid, meta)


def processor_geometry(view: CalibratedView, pixel_mask) -> dict:
    """Recover exact resize/pad geometry from this model's single-image mask.

    GroundingDinoImageProcessor resizes without cropping and pads bottom/right.
    Reject other layouts rather than assigning a wrong K to them.
    """
    mask = np.asarray(pixel_mask, dtype=bool)
    if mask.ndim != 2 or not mask.any():
        raise ValueError("Expected nonempty 2D processor pixel mask")
    ys, xs = np.nonzero(mask)
    h, w = int(ys.max()) + 1, int(xs.max()) + 1
    if xs.min() != 0 or ys.min() != 0 or not mask[:h, :w].all() or mask.sum() != h * w:
        raise ValueError("Unsupported processor crop/padding layout")
    K = view.intrinsics.scaled(w, h).K
    return {"size_wh": [mask.shape[1], mask.shape[0]], "resized_size_wh": [w, h],
            "K": K.tolist(), "D": [0.0] * 5, "distortion_model": "pinhole_radtan",
            "crop_xywh": [0, 0, view.intrinsics.width, view.intrinsics.height],
            "resize_scale_xy": [w / view.intrinsics.width, h / view.intrinsics.height],
            "padding_ltrb": [0, 0, mask.shape[1] - w, mask.shape[0] - h],
            "pixel_convention": "integer pixel centers; c_new=(c_old+0.5)*scale-0.5"}
