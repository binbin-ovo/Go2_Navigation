"""Re-identify a human-named object in the current rectified frame, allowing viewpoint changes."""

from hashlib import sha256
from pathlib import Path

import cv2
import numpy as np

from .motion_profile import Hold


def reference_matches_target(name, requested):
    """Allow case and whitespace differences. Every descriptive word still has to match."""
    if not isinstance(name, str) or not isinstance(requested, str):
        return False
    return bool(name.strip()) and name.casefold().split() == requested.casefold().split()


def valid_roi(roi):
    return (
        isinstance(roi, (list, tuple, np.ndarray))
        and np.shape(roi) == (4,)
        and all(
            isinstance(v, (int, float, np.integer, np.floating))
            and not isinstance(v, bool)
            and np.isfinite(v)
            for v in roi
        )
        and 0 <= roi[0] < roi[2] <= 1
        and 0 <= roi[1] < roi[3] <= 1
    )


def read_gray(path, unavailable, expected_sha=None):
    try:
        content = Path(path).read_bytes()
    except OSError as exc:
        raise Hold(unavailable) from exc
    image_sha = sha256(content).hexdigest()
    if expected_sha is not None and image_sha != expected_sha:
        raise Hold("current_target_image_changed")
    image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_GRAYSCALE) if content else None
    if image is None:
        raise Hold(unavailable)
    return image, image_sha


def remember_rejection(previous, new):
    """Retain at most five operator-rejected regions for this episode."""
    recent = [r for r in previous if (r["sha256"], r["roi"]) != (new["sha256"], new["roi"])]
    return [new, *recent[:4]]


def object_reference(path, roi, target, operator):
    if not valid_roi(roi):
        raise Hold("invalid_reference_image_or_roi")
    image, image_sha = read_gray(path, "invalid_reference_image_or_roi")
    keypoints, descriptors, size = features(image, roi)
    if descriptors is None or len(keypoints) < 12:
        raise Hold("reference_has_too_few_distinctive_features")
    return {
        "target": target,
        "operator": operator,
        "sha256": image_sha,
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


def localize(ref, current, point):
    """Locate an enrolled ROI before asking whether the current native mark agrees."""
    q, desc, (w, h) = current
    evidence = dict(
        reference_sha256=ref["sha256"],
        reference_roi=list(ref["roi"]),
        operator=ref["operator"],
        localized=False,
    )
    if desc is None or len(q) < 12:
        return dict(evidence, reason="too_few_current_features")
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(ref["descriptors"], desc, k=2)
    reverse = {
        (a.trainIdx, a.queryIdx)
        for a, b in matcher.knnMatch(desc, ref["descriptors"], k=2)
        if a.distance < 0.7 * b.distance
    }
    good = [a for a, b in pairs if a.distance < 0.7 * b.distance and (a.queryIdx, a.trainIdx) in reverse]
    evidence["mutual_matches"] = len(good)
    if len(good) < 12:
        return dict(evidence, reason="too_few_mutual_matches")
    a = np.float32([ref["points"][m.queryIdx] for m in good])
    b = np.float32([q[m.trainIdx] for m in good])
    mat, inliers = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=2.5)
    if mat is None or inliers is None:
        return dict(evidence, reason="no_consistent_transform")
    keep = inliers.ravel().astype(bool)
    x0, y0, x1, y1 = np.asarray(ref["roi"]) * [*ref["size"], *ref["size"]]
    spread = cv2.contourArea(cv2.convexHull(a[keep])) / ((x1 - x0) * (y1 - y0)) if keep.sum() >= 3 else 0.0
    evidence.update(inliers=int(keep.sum()), inlier_fraction=float(keep.mean()), feature_spread=float(spread))
    if keep.sum() < 12 or keep.mean() < 0.8 or spread < 0.12:
        return dict(evidence, reason="insufficient_distributed_inliers")
    residual = float(np.median(np.linalg.norm(cv2.transform(a[None], mat)[0][keep] - b[keep], axis=1)))
    evidence["median_residual_px"] = residual
    if residual > 2:
        return dict(evidence, reason="large_reprojection_residual")
    corners = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    polygon = cv2.transform(corners[None], mat)[0]
    frame = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    area, visible = cv2.intersectConvexConvex(polygon, frame)
    if area <= 0 or visible is None:
        return dict(evidence, reason="reference_outside_image")
    visible = np.clip(visible.reshape(-1, 2) / [w, h], 0, 1)
    inside = (
        None
        if point is None
        else cv2.pointPolygonTest(polygon, (float(point[0] * w), float(point[1] * h)), False) >= 0
    )
    return dict(
        evidence,
        localized=True,
        reason="localized",
        mark_inside=inside,
        roi=[*visible.min(axis=0).tolist(), *visible.max(axis=0).tolist()],
    )


def ground(pred, path, rectification, target, ref, rejected=None, *, seek_view=False):
    """Bind native visibility, positive localization and rejected-instance evidence to one image.

    Five-move episodes opt into a new SEARCH viewpoint on inconclusive enrolled identity.
    The default preserves the review behavior of the separate continuous replay.
    """
    view = rectification["view"]
    if not view["rectified"] or view["distortion_model"] != "pinhole_radtan":
        raise Hold("rectified_image_required")
    point = pred["target_fraction"]
    positive = ref is not None and reference_matches_target(ref["target"], target)
    rejected_refs = [rejected] if isinstance(rejected, dict) else (rejected or [])
    negatives = [r for r in rejected_refs if reference_matches_target(r["target"], target)]
    e = {
        "target": target,
        "image_sha256": pred["source_image"]["file_sha256"],
        "prediction_id": pred["prediction_id"],
        "native_visibility": "marked"
        if point is not None
        else "reported_absent"
        if pred["opos_not_visible"]
        else "unknown",
        "camera": {"width": view["size_wh"][0], "fx": view["K"][0][0], "cx": view["K"][0][2]},
    }
    if (positive or negatives) and (point is not None or seek_view and positive):
        image, _ = read_gray(path, "current_target_image_unavailable", e["image_sha256"])
        current = features(image)
        if positive:
            e["positive_reference"] = localize(ref, current, point)
        if negatives and point is not None:
            e["rejected_references"] = [localize(r, current, point) for r in negatives]
    match = e.get("positive_reference", {})
    if any(r.get("mark_inside") for r in e.get("rejected_references", [])):
        if match.get("mark_inside"):
            return dict(
                e,
                status="uncertain" if seek_view else "needs_identification",
                method="conflicting_instance_matches",
            )
        return dict(e, status="uncertain", method="previously_rejected_instance")
    if match.get("mark_inside"):
        return dict(
            e,
            status="supported",
            method="identified_instance_features",
            **{
                k: match[k]
                for k in (
                    "operator",
                    "reference_sha256",
                    "roi",
                    "inliers",
                    "inlier_fraction",
                    "feature_spread",
                    "median_residual_px",
                )
            },
        )
    if seek_view and match.get("localized"):
        return dict(e, status="uncertain", method="reference_visible_without_native_agreement")
    if point is None:
        return dict(
            e, status="absent" if pred["opos_not_visible"] else "uncertain", method="native_visibility"
        )
    if seek_view and positive:
        return dict(e, status="uncertain", method="reference_inconclusive_seek_view")
    return dict(e, status="needs_identification", method="unverified_native_mark")
