"""Re-identify a human-named object in the current rectified frame, allowing viewpoint changes."""

from hashlib import sha256
import json
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


def prepared_detail(pred, snapshot, job, native_path):
    """Bind the current prepared image to the exact uncropped native view and decision."""
    if pred.get("mock") or pred.get("backend") != "lightnav":
        return None
    native = Path(native_path)
    try:
        item = snapshot["history"][-1]
        src, frame, view = item["source"], job["frames"][-1], item["rectification"]["view"]
        display = pred["source_image"]
        index = snapshot["bounded_context"]["source_indices"][-1]
        size = view["size_wh"]
        if not (
            view["rectified"] is True
            and view["distortion_model"] == "pinhole_radtan"
            and view.get("crop_xywh") == [0, 0, *size]
            and view.get("padding_ltrb") == [0] * 4
            and view.get("resize_scale_xy") == [1, 1]
        ):
            return None  # The mapping to this view is unsupported. Native ORB remains available.
        if not (
            snapshot["current_source"] == src
            and all(
                src[k] == pred["source_frame"][k]
                for k in (
                    "frame_id",
                    "clock_epoch",
                    "observation_s",
                    "available_s",
                    "received_monotonic_s",
                    "exposure_monotonic_s",
                )
            )
            and src["observation_s"]
            == frame["observation_s"]
            <= src["available_s"]
            == frame["available_s"]
            <= job["decision_time_s"]
            == snapshot["source_cutoff_s"]
            and src["clock_epoch"] == job["clock_epoch"]
            and job["instruction"] == pred["instruction"]
            and len(snapshot["history"]) == len(job["frames"]) == display["job_frame_index"] + 1
            and index == display["causal_grid_index"]
            and native.name == f"input_{index:04d}.png"
        ):
            raise Hold("prepared_detail_frame_binding_mismatch")
        return dict(
            path=frame["path"], sha256=frame["sha256"], size=size, native_sha256=display["file_sha256"]
        )
    except (KeyError, IndexError, TypeError) as exc:
        raise Hold("incomplete_prepared_detail_binding") from exc


def request_detail(pred, request, native_path, snapshot=None):
    if pred.get("mock") or pred.get("backend") != "lightnav":
        return None
    if Path(native_path).resolve().parent != (request / "native").resolve():
        raise Hold("prepared_detail_request_mismatch")
    context = snapshot or json.loads((request / "snapshot.json").read_bytes())
    job = json.loads((request / "job.json").read_bytes())
    native = json.loads((request / "native/result.json").read_bytes())
    if not (
        native["status"] == "decoded"
        and native["native_output"] == pred["native_output"]
        and native["native_selected_frame_indices"] == context["bounded_context"]["source_indices"]
        and all(native[k] == job[k] for k in ("clock_epoch", "decision_id", "episode_id"))
    ):
        raise Hold("prepared_detail_native_decision_mismatch")
    detail = prepared_detail(pred, context, job, native_path)
    if (
        detail
        and Path(detail["path"]).resolve() != (request / f"frame_{len(job['frames']) - 1:04d}.png").resolve()
    ):
        raise Hold("prepared_detail_request_mismatch")
    return detail


def detail_features(detail, native_sha, roi=None):
    if detail["native_sha256"] != native_sha:
        raise Hold("prepared_detail_native_hash_mismatch")
    image, _ = read_gray(detail["path"], "prepared_detail_unavailable", detail["sha256"])
    h, w = image.shape
    if [w, h] != list(detail["size"]):
        raise Hold("prepared_detail_dimensions_changed")
    pts, desc = cv2.SIFT_create().detectAndCompute(image, None)
    if roi is not None and desc is not None:
        x0, y0, x1, y1 = np.asarray(roi) * [w, h, w, h]
        # Nominal radius of the rotated 4x4 OpenCV SIFT sampling grid, in keypoint diameters.
        support_radius = 3.75 * np.sqrt(2)
        keep = [
            i
            for i, p in enumerate(pts)
            if min(p.pt[0] - x0, x1 - p.pt[0], p.pt[1] - y0, y1 - p.pt[1]) > support_radius * p.size
        ]
        pts, desc = [pts[i] for i in keep], desc[keep]
    return np.float32([p.pt for p in pts]).reshape(-1, 2), desc, (w, h)


def object_reference(path, roi, target, operator, *, detail=None):
    if not valid_roi(roi):
        raise Hold("invalid_reference_image_or_roi")
    image, image_sha = read_gray(path, "invalid_reference_image_or_roi")
    if detail and image.shape != (256, 448):
        raise Hold("unexpected_native_image_size")
    keypoints, descriptors, size = features(image, roi)
    if descriptors is None or len(keypoints) < 12:
        raise Hold("reference_has_too_few_distinctive_features")
    reference = {
        "target": target,
        "operator": operator,
        "sha256": image_sha,
        "roi": list(roi),
        "size": size,
        "points": keypoints,
        "descriptors": descriptors,
    }
    if detail is not None:
        pts, desc, size = detail_features(detail, image_sha, roi)
        if desc is not None and len(pts) >= 12:
            reference["detail"] = dict(
                reference, points=pts, descriptors=desc, size=size, prepared_sha256=detail["sha256"]
            )
    return reference


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


def localize(ref, current, point, *, interior=False):
    """Locate an enrolled ROI before asking whether the current native mark agrees."""
    q, desc, (w, h) = current
    evidence = dict(
        reference_sha256=ref["sha256"],
        reference_roi=list(ref["roi"]),
        algorithm="interior_sift" if interior else "native_orb",
        operator=ref["operator"],
        localized=False,
    )
    if desc is None or len(q) < 12:
        return dict(evidence, reason="too_few_current_features")
    matcher = cv2.BFMatcher(cv2.NORM_L2 if interior else cv2.NORM_HAMMING)
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
    if interior:
        evidence["reference_prepared_sha256"] = ref["prepared_sha256"]
        evidence["mark_in_region"] = inside
        evidence["mark_in_inlier_hull"] = (
            None
            if point is None
            else cv2.pointPolygonTest(
                cv2.convexHull(b[keep]), (float(point[0] * w), float(point[1] * h)), False
            )
            >= 0
        )
        inside = inside and evidence["mark_in_inlier_hull"]
    return dict(
        evidence,
        localized=True,
        reason="localized",
        mark_inside=inside,
        roi=[*visible.min(axis=0).tolist(), *visible.max(axis=0).tolist()],
    )


def ground(pred, path, rectification, target, ref, rejected=None, *, seek_view=False, detail=None):
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
        if detail and image.shape != (256, 448):
            raise Hold("unexpected_native_image_size")
        current = features(image)
        detailed = None

        def match_instance(reference, *, negative=False):
            nonlocal detailed
            native_match = localize(reference, current, point)
            if (
                not seek_view
                or detail is None
                or "detail" not in reference
                or native_match.get("mark_inside")
                or native_match.get("localized")
                and not negative
            ):
                return native_match
            if detailed is None:
                detailed = detail_features(detail, e["image_sha256"])
            candidate = localize(reference["detail"], detailed, point, interior=True)
            candidate["current_prepared_sha256"] = detail["sha256"]
            if negative and candidate.get("mark_in_region"):
                # A rejected instance can veto anywhere in its projected region.
                candidate["mark_inside"] = True
            if candidate.get("localized"):
                return dict(candidate, native_orb=native_match)
            return dict(native_match, detail_attempt=candidate)

        if positive:
            e["positive_reference"] = match_instance(ref)
        if negatives and point is not None:
            e["rejected_references"] = [match_instance(r, negative=True) for r in negatives]
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
