"""Conservative visual continuity of an operator-selected ROI; no semantic model."""
import hashlib
from pathlib import Path

import cv2
import numpy as np


def correspondence(review_path, fresh_path, target, point_fraction):
    """Register nearly stationary views; require support INSIDE the reviewed object.

    This can refuse textureless/occluded objects. It does not prove object identity;
    Claude must validate it before enabling physical use in the trial conditions.
    Repeated patterns, poor geometry and a point outside the registered ROI HOLD.
    """
    paths = [Path(review_path), Path(fresh_path)]
    hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    result = {"method": "local_feature_roi_v1", "instance_id": target["instance_id"],
              "reviewed_image_sha256": hashes[0], "fresh_image_sha256": hashes[1],
              "valid": False, "reason": "insufficient_correspondence"}
    if hashes[0] != target["reviewed_image_sha256"]:
        result["reason"] = "reviewed_image_hash_mismatch"
        return result
    images = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in paths]
    if any(im is None for im in images) or images[0].shape != images[1].shape or point_fraction is None:
        result["reason"] = "image_shape_or_target_point_unavailable"
        return result
    h, w = images[0].shape
    x0, y0, x1, y1 = np.asarray(target["roi_xyxy"]) * [w, h, w, h]
    detector = cv2.ORB_create(nfeatures=1600, edgeThreshold=12, fastThreshold=12)
    features = [detector.detectAndCompute(im, None) for im in images]
    if any(desc is None for _, desc in features):
        return result
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    forward = matcher.knnMatch(features[0][1], features[1][1], k=2)
    backward = matcher.knnMatch(features[1][1], features[0][1], k=2)
    reverse = {(a.queryIdx, a.trainIdx) for a, b in backward if a.distance < .7*b.distance}
    good = [a for a, b in forward if a.distance < .7*b.distance and (a.trainIdx, a.queryIdx) in reverse]
    if len(good) < 16:
        return result
    p = np.float32([features[0][0][m.queryIdx].pt for m in good])
    q = np.float32([features[1][0][m.trainIdx].pt for m in good])
    # An affine rigid/scale approximation is deliberately stricter than arbitrary homography.
    transform, inliers = cv2.estimateAffinePartial2D(p, q, method=cv2.RANSAC,
                                                   ransacReprojThreshold=2., maxIters=1000)
    if transform is None or inliers is None:
        return result
    mask = inliers.ravel().astype(bool)
    inside = (p[:, 0] > x0) & (p[:, 0] < x1) & (p[:, 1] > y0) & (p[:, 1] < y1) & mask
    corners = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    warped = cv2.transform(corners[None], transform)[0]
    drift = np.linalg.norm(warped-corners, axis=1).max() / max(w, h)
    point = tuple(float(v) for v in np.asarray(point_fraction)*[w, h])
    residual = float(np.median(np.linalg.norm(cv2.transform(p[None], transform)[0][mask]-q[mask], axis=1)))
    result.update(inliers=int(mask.sum()), inlier_fraction=float(mask.mean()),
                  target_inliers=int(inside.sum()), median_residual_px=residual,
                  max_corner_drift_fraction=float(drift), registered_roi_px=warped.tolist(),
                  fresh_point_inside=cv2.pointPolygonTest(warped, point, False) >= 0)
    result["valid"] = bool(mask.sum() >= 16 and mask.mean() >= .8 and inside.sum() >= 8 and
                           residual <= 1.5 and drift <= .02 and result["fresh_point_inside"])
    result["reason"] = "supported_local_visual_continuity" if result["valid"] else "changed_or_ambiguous_target_roi"
    return result


def check_evidence(evidence, target, fresh_hash):
    """Executor checks all diagnostic bounds, never just a browser-supplied true flag."""
    return (evidence.get("method") == "local_feature_roi_v1" and evidence.get("valid") is True and
            evidence.get("instance_id") == target["instance_id"] and
            evidence.get("reviewed_image_sha256") == target["reviewed_image_sha256"] and
            evidence.get("fresh_image_sha256") == fresh_hash and
            evidence.get("inliers", 0) >= 16 and evidence.get("inlier_fraction", 0) >= .8 and
            evidence.get("target_inliers", 0) >= 8 and evidence.get("median_residual_px", 99) <= 1.5 and
            evidence.get("max_corner_drift_fraction", 99) <= .02 and evidence.get("fresh_point_inside") is True)
