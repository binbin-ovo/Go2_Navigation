import numpy as np
from scipy.optimize import linear_sum_assignment
from go2_perception.core.types import FusedDetection, FusionResult
from .projection import project_points


class FusionProcessor:
    def __init__(
        self,
        sync_tolerance_s=0.05,
        min_points=3,
        min_fraction=0.5,
        require_timestamps=True,
    ):
        if sync_tolerance_s < 0 or min_points < 1 or not 0 < min_fraction <= 1:
            raise ValueError("Invalid fusion configuration")
        self.sync_tolerance_s, self.min_points = sync_tolerance_s, min_points
        self.min_fraction, self.require_timestamps = min_fraction, require_timestamps

    def fuse(self, rgb, lidar, calibration):
        if (
            rgb.frame_id != calibration.camera_frame
            or lidar.frame_id != calibration.lidar_frame
        ):
            raise ValueError("Sensor frames do not match calibration")
        if tuple(rgb.image_size) != calibration.image_size:
            raise ValueError("Calibration resolution mismatch")
        stamps = [rgb.timestamp, lidar.timestamp]
        if any(t is not None and not np.isfinite(t) for t in stamps):
            raise ValueError("Nonfinite timestamp")
        if any(t is None for t in stamps):
            if self.require_timestamps:
                raise ValueError("Both timestamps required")
            if any(t is not None for t in stamps):
                raise ValueError("Both timestamps must be supplied or omitted")
        else:
            if not rgb.clock_domain or rgb.clock_domain != lidar.clock_domain:
                raise ValueError("Timestamps must share a named clock domain")
            if abs(rgb.timestamp - lidar.timestamp) > self.sync_tolerance_s + 1e-12:
                raise ValueError("Sensor observations exceed synchronization tolerance")
        scores = np.zeros((len(rgb.detections), len(lidar.obstacles)))
        for j, obstacle in enumerate(lidar.obstacles):
            uv, _, _ = project_points(obstacle.points, calibration)
            if len(uv) < self.min_points:
                continue
            for i, detection in enumerate(rgb.detections):
                x1, y1, x2, y2 = detection.bbox_xyxy
                count = np.sum(
                    (uv[:, 0] >= x1)
                    & (uv[:, 0] <= x2)
                    & (uv[:, 1] >= y1)
                    & (uv[:, 1] <= y2)
                )
                fraction = count / len(uv)
                if count >= self.min_points and fraction >= self.min_fraction:
                    scores[i, j] = fraction
        rows, cols = linear_sum_assignment(-scores)
        matched_rgb, matched_lidar, fused = set(), set(), []
        for i, j in zip(rows, cols):
            if scores[i, j] == 0:
                continue
            d, o = rgb.detections[i], lidar.obstacles[j]
            fused.append(
                FusedDetection(
                    d.class_id,
                    d.class_name,
                    d.confidence,
                    o.cluster_id,
                    o.center,
                    o.bbox_min,
                    o.bbox_max,
                    lidar.timestamp,
                    rgb.timestamp,
                    lidar.frame_id,
                    lidar.clock_domain,
                    float(scores[i, j]),
                )
            )
            matched_rgb.add(int(i))
            matched_lidar.add(int(j))
        status = (
            "synthetic calibration; real sensor fusion NOT validated"
            if calibration.synthetic
            else "provided calibration; real sensor fusion NOT validated by this pipeline"
        )
        return FusionResult(
            fused,
            status,
            [i for i in range(len(rgb.detections)) if i not in matched_rgb],
            [
                o.cluster_id
                for j, o in enumerate(lidar.obstacles)
                if j not in matched_lidar
            ],
        )
