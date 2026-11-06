"""Offline schemas and existing Go2 recording adapter. Never invent real clocks or poses."""

from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from go2_perception.core.tracking_types import Detection3D, RobotPose
from go2_perception.adapters.go2.window import PoseBuffer, SlidingWindowAccumulator
from go2_perception.adapters.go2.recording import RecordingReader
from go2_perception.adapters.go2.pointcloud import to_lidar_message, quaternion_matrix
from go2_perception.lidar.processor import LidarProcessor
from .normalization import from_lidar


@dataclass
class TrackingFrame:
    timestamp: float
    frame_id: str
    clock_domain: str
    detections: list[Detection3D]
    robot_pose: RobotPose | None = None
    ground_truth: list | None = None


def frame_from_dict(row):
    t, frame, clock = row["timestamp"], row["frame_id"], row["clock_domain"]
    detections = [
        Detection3D(timestamp=t, frame_id=frame, clock_domain=clock, **d)
        for d in row["detections"]
    ]
    pose = None if row.get("robot_pose") is None else RobotPose(**row["robot_pose"])
    if not np.isfinite(t) or not frame or not clock:
        raise ValueError("Invalid frame metadata")
    return TrackingFrame(t, frame, clock, detections, pose, row.get("ground_truth"))


def ordered(rows, duplicate_policy="error"):
    if duplicate_policy not in ("error", "first"):
        raise ValueError("Invalid duplicate policy")
    rows = sorted(rows, key=lambda r: r.timestamp)
    result = []
    duplicates = 0
    for row in rows:
        if not np.isfinite(row.timestamp):
            raise ValueError("Invalid timestamp")
        if result and row.timestamp == result[-1].timestamp:
            if duplicate_policy == "error":
                raise ValueError(f"Duplicate timestamp {row.timestamp}")
            duplicates += 1
            continue
        result.append(row)
    return result, duplicates


def load_detection_sequence(path, duplicate_policy="error"):
    path = Path(path)
    if path.is_dir():
        path = path / "frames.jsonl"
    rows = [
        frame_from_dict(json.loads(line))
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("Sequence has no frames")
    return ordered(rows, duplicate_policy)


class RecordedPoses:
    """Validate metadata, reuse Phase 2 interpolation, and preserve exact samples at gaps."""

    def __init__(self, records, max_gap_s=0.1):
        self.buffer = PoseBuffer(max_age_s=float("inf"), max_gap_s=max_gap_s)
        self.exact = {}
        for r in sorted(records, key=lambda r: r["stamp"]):
            q = np.asarray(r["q_xyzw"], dtype=float)
            if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
                raise ValueError("Invalid odometry quaternion")
            t = r["stamp"]
            matrix = np.eye(4)
            matrix[:3, :3] = quaternion_matrix(q)
            matrix[:3, 3] = r["position"]
            pose = RobotPose(t, matrix, r["frame_id"], r["child_frame_id"], "go2_robot")
            if pose.parent_frame != "odom" or pose.child_frame != "base_link":
                raise ValueError("Recording requires odom -> base_link poses")
            if t in self.exact:
                raise ValueError("Duplicate odometry timestamp")
            self.exact[t] = matrix
            self.buffer.add(t, r["position"], q)

    def lookup(self, t):
        return self.exact[t].copy() if t in self.exact else self.buffer.lookup(t)


def recording_frames(path, config, lidar_config, window_s=0):
    reader = RecordingReader(path)
    frames = reader.lidar_frames("rt/utlidar/cloud_base")
    if not frames:
        raise ValueError("No cloud_base frames in recording")
    poses = RecordedPoses(
        reader.state("rt/utlidar/robot_odom"), 2 * config.pose_tolerance_s
    )
    window = SlidingWindowAccumulator(poses, window_s) if window_s else None
    if window and config.mode != "odom":
        raise ValueError("Windows require odometry mode")
    processor = LidarProcessor(**lidar_config)
    # Sort by declared acquisition midpoint (not host receipt or file name).
    frames = sorted((r.load() for r in frames), key=lambda f: f.acquisition_midpoint)
    last = None
    for frame in frames:
        t = frame.acquisition_midpoint
        if last is not None and t <= last:
            raise ValueError("Duplicate/nonmonotonic LiDAR midpoint")
        last = t
        robot_pose = None
        if config.mode == "odom":
            matrix = poses.lookup(t)
            if matrix is None:
                raise ValueError(f"No valid bracketing odometry for LiDAR at {t}")
            if window:
                if not window.add(frame.xyz, t):
                    raise ValueError("Cannot accumulate cloud without pose")
                merged, _, _, _ = window.merged()
                from dataclasses import replace

                frame = replace(frame, xyz=merged, intensity=None, time=None, stamp=t)
            # Level once, then use a yaw-only level->odom transform, avoiding double rotation.
            q = Rotation.from_matrix(matrix[:3, :3]).as_quat()
            message = to_lidar_message(frame, q)
            yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
            level = matrix.copy()
            level[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()
            robot_pose = RobotPose(t, level, "odom", "base_link_level", "go2_robot")
        else:
            message = to_lidar_message(frame)
        result = processor.process(
            message.points,
            timestamp=t,
            frame_id=message.frame_id,
            clock_domain=message.clock_domain,
        )
        yield TrackingFrame(
            t, message.frame_id, message.clock_domain, from_lidar(result), robot_pose
        )
