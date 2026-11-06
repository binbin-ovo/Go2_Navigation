"""Explicit sensor-to-tracking-frame transforms; no implicit stationary-robot assumption."""

from dataclasses import replace
from itertools import product
import numpy as np
from go2_perception.core.transforms import transform_points


def transform_detection(detection, transform, target_frame, default_variance=0.01):
    r = transform[:3, :3]
    measurement = (
        np.eye(3) * default_variance
        if detection.measurement_covariance is None
        else detection.measurement_covariance
    )
    low = high = None
    if detection.bbox_min is not None:
        corners = np.array(list(product(*zip(detection.bbox_min, detection.bbox_max))))
        mapped = transform_points(corners, transform)
        low, high = mapped.min(0), mapped.max(0)
    return replace(
        detection,
        position=transform_points(detection.position[None], transform)[0],
        frame_id=target_frame,
        bbox_min=low,
        bbox_max=high,
        measurement_covariance=r @ measurement @ r.T,
    )


def normalize_detections(
    detections, timestamp, pose, config, source_frame, clock_domain
):
    for d in detections:
        if abs(d.timestamp - timestamp) > config.timestamp_tolerance_s:
            raise ValueError("Detection timestamp differs from update")
        if d.frame_id != source_frame or d.clock_domain != clock_domain:
            raise ValueError("Mixed detection frames or clocks")
    if config.mode == "fixed_sensor":
        if pose is not None:
            raise ValueError("Fixed-sensor mode must not consume robot poses")
        if source_frame != config.sensor_frame:
            raise ValueError("Fixed sensor frame changed")
        target = "fixed_sensor:" + source_frame
        transform = np.eye(4)
    else:
        target = config.tracking_frame
        if source_frame == target:
            transform = np.eye(4)
        else:
            if pose is None:
                raise ValueError("Robot pose required for odometry-frame tracking")
            if pose.parent_frame != target or pose.clock_domain != clock_domain:
                raise ValueError("Pose frame/clock mismatch")
            if abs(pose.timestamp - timestamp) > config.pose_tolerance_s:
                raise ValueError("Stale robot pose")
            if source_frame == pose.child_frame:
                transform = pose.T_parent_child
            elif (
                pose.child_frame == config.base_frame
                and source_frame == config.sensor_frame
                and config.T_base_sensor is not None
            ):
                transform = pose.T_parent_child @ np.asarray(config.T_base_sensor)
            else:
                raise ValueError("Missing sensor-to-pose extrinsic")
        if pose is not None:
            if (
                pose.parent_frame != target
                or pose.clock_domain != clock_domain
                or abs(pose.timestamp - timestamp) > config.pose_tolerance_s
            ):
                raise ValueError("Invalid supplied pose metadata")
    return [
        transform_detection(d, transform, target, config.measurement_noise)
        for d in detections
    ], target
