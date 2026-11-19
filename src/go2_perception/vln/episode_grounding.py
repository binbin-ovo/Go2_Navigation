"""Re-identify a human-named object in the current rectified frame, allowing viewpoint changes."""

from hashlib import sha256
from pathlib import Path

import cv2
import numpy as np

from .motion_profile import Hold


def object_reference(path, roi, target, operator):
    path = Path(path)
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or len(roi) != 4 or not (0 <= roi[0] < roi[2] <= 1 and 0 <= roi[1] < roi[3] <= 1):
        raise Hold("invalid_reference_image_or_roi")
    keypoints, descriptors, size = features(image, roi)
    if descriptors is None or len(keypoints) < 12:
        raise Hold("reference_has_too_few_distinctive_features")
    return {
        "target": target,
        "operator": operator,
        "sha256": sha256(path.read_bytes()).hexdigest(),
        "roi": list(roi),
        "size": size,
        "points": keypoints,
        "descriptors": descriptors,
    }


def features(image, roi=None):
    scale = min(1.0, 720.0 / max(image.shape))
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h, w = image.shape
    mask = None
    if roi is not None:
        x0, y0, x1, y1 = np.asarray(roi) * [w, h, w, h]
        mask = np.zeros((h, w), np.uint8)
        mask[int(np.ceil(y0)) : int(y1), int(np.ceil(x0)) : int(x1)] = 255
    pts, desc = cv2.ORB_create(nfeatures=1500, edgeThreshold=12, fastThreshold=12).detectAndCompute(
        image, mask
    )
    return np.float32([p.pt for p in pts]).reshape(-1, 2), desc, (w, h)


def locate(ref, image, point):
    """Require distributed, mutual object-only matches and the current mark inside the warped ROI."""
    q, desc, (w, h) = features(image)
    if desc is None or len(q) < 12:
        return None
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(ref["descriptors"], desc, k=2)
    reverse = {
        (a.trainIdx, a.queryIdx)
        for a, b in matcher.knnMatch(desc, ref["descriptors"], k=2)
        if a.distance < 0.7 * b.distance
    }
    good = [a for a, b in pairs if a.distance < 0.7 * b.distance and (a.queryIdx, a.trainIdx) in reverse]
    if len(good) < 12:
        return None
    a = np.float32([ref["points"][m.queryIdx] for m in good])
    b = np.float32([q[m.trainIdx] for m in good])
    mat, inliers = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=2.5)
    if mat is None or inliers is None:
        return None
    keep = inliers.ravel().astype(bool)
    x0, y0, x1, y1 = np.asarray(ref["roi"]) * [*ref["size"], *ref["size"]]
    spread = cv2.contourArea(cv2.convexHull(a[keep])) / ((x1 - x0) * (y1 - y0)) if keep.sum() >= 3 else 0.0
    if keep.sum() < 12 or keep.mean() < 0.8 or spread < 0.12:
        return None
    residual = float(np.median(np.linalg.norm(cv2.transform(a[None], mat)[0][keep] - b[keep], axis=1)))
    corners = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    polygon = cv2.transform(corners[None], mat)[0]
    uv = (float(point[0] * w), float(point[1] * h))
    if residual > 2 or cv2.pointPolygonTest(polygon, uv, False) < 0:
        return None
    lo, hi = np.clip(polygon.min(axis=0) / [w, h], 0, 1), np.clip(polygon.max(axis=0) / [w, h], 0, 1)
    return {
        "method": "identified_instance_features",
        "operator": ref["operator"],
        "reference_sha256": ref["sha256"],
        "roi": [*lo.tolist(), *hi.tolist()],
        "inliers": int(keep.sum()),
        "inlier_fraction": float(keep.mean()),
        "feature_spread": float(spread),
        "median_residual_px": residual,
    }


def ground(pred, path, rectification, target, ref, rejected=None):
    view = rectification["view"]
    if not view["rectified"] or view["distortion_model"] != "pinhole_radtan":
        raise Hold("rectified_image_required")
    e = {
        "target": target,
        "image_sha256": pred["source_image"]["file_sha256"],
        "prediction_id": pred["prediction_id"],
        "camera": {"width": view["size_wh"][0], "fx": view["K"][0][0], "cx": view["K"][0][2]},
    }
    point = pred["target_fraction"]
    if point is None:
        return dict(
            e, status="absent" if pred["opos_not_visible"] else "uncertain", method="native_visibility"
        )
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise Hold("current_target_image_unavailable")
    match = locate(ref, image, point) if ref is not None and ref["target"] == target else None
    incompatible = rejected is not None and locate(rejected, image, point) is not None
    if match and incompatible:
        return dict(e, status="needs_identification", method="conflicting_instance_matches")
    if incompatible:
        return dict(e, status="uncertain", method="previously_rejected_instance")
    if match:
        return dict(e, status="supported", **match)
    return dict(e, status="needs_identification", method="unverified_native_mark")
