import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from go2_perception.core.tracking_types import Detection3D, RobotPose
from go2_perception.core.transforms import transform_points
from go2_perception.tracking.multi_object_tracker import MultiObjectTracker
from go2_perception.tracking.config import TrackerConfig
from go2_perception.localization.ego_motion import normalize_detections


@pytest.mark.parametrize("rotation", [False, True])
def test_static_world_object_under_ego_motion(rotation):
    tracker = MultiObjectTracker()
    for i in range(60):
        t = i * 0.1
        matrix = np.eye(4)
        matrix[:3, 3] = [0.5 * t, 0, 0.3]
        if rotation:
            matrix[:3, :3] = Rotation.from_euler("z", t * 0.3).as_matrix()
        body = transform_points([[4.0, 2.0, 1.0]], np.linalg.inv(matrix))[0]
        d = Detection3D(t, body, "base_link", "sim")
        tracks = tracker.update(
            [d], t, RobotPose(t, matrix, "odom", "base_link", "sim")
        )
    np.testing.assert_allclose(tracks[0].position, [4, 2, 1], atol=1e-6)
    np.testing.assert_allclose(tracks[0].velocity, [0, 0], atol=1e-6)
    assert tracks[0].motion_state == "STATIONARY"


def test_full_rotation_extrinsics_bbox_and_covariance():
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("xyz", [0.3, 0.2, 0.7]).as_matrix()
    transform[:3, 3] = [1, 2, 3]
    ext = np.eye(4)
    ext[0, 3] = 0.2
    c = TrackerConfig(sensor_frame="lidar", T_base_sensor=ext.tolist())
    d = Detection3D(
        1,
        [2, 0, 0],
        "lidar",
        "sim",
        bbox_min=[1, -1, -1],
        bbox_max=[3, 1, 1],
        measurement_covariance=np.diag([0.1, 0.2, 0.3]),
    )
    rows, frame = normalize_detections(
        [d], 1, RobotPose(1, transform, "odom", "base_link", "sim"), c, "lidar", "sim"
    )
    np.testing.assert_allclose(
        rows[0].position, transform_points([[2.2, 0, 0]], transform)[0]
    )
    np.testing.assert_allclose(
        rows[0].measurement_covariance,
        transform[:3, :3] @ d.measurement_covariance @ transform[:3, :3].T,
    )
    assert (
        frame == "odom"
        and np.all(rows[0].bbox_min <= rows[0].position)
        and np.all(rows[0].position <= rows[0].bbox_max)
    )


@pytest.mark.parametrize("case", ["missing", "stale", "clock", "parent", "child"])
def test_reject_bad_pose(case):
    pose = RobotPose(0, np.eye(4), "odom", "base_link", "sim")
    d = Detection3D(0, [2, 0, 0], "base_link", "sim")
    if case == "missing":
        pose = None
    if case == "stale":
        pose.timestamp = 1
    if case == "clock":
        pose.clock_domain = "host"
    if case == "parent":
        pose.parent_frame = "map"
    if case == "child":
        pose.child_frame = "wrong"
    with pytest.raises(ValueError):
        MultiObjectTracker().update([d], 0, pose)


def test_fixed_sensor_is_explicit_and_labelled():
    d = Detection3D(0, [2, 0, 0], "base_link", "sim")
    result = MultiObjectTracker({"mode": "fixed_sensor"}).update([d], 0)[0]
    assert result.frame_id == "fixed_sensor:base_link"
    assert result.velocity_reference == "fixed_sensor_relative"
    with pytest.raises(ValueError):
        MultiObjectTracker({"mode": "fixed_sensor"}).update(
            [d], 0, RobotPose(0, np.eye(4), "odom", "base_link", "sim")
        )


def test_moving_world_object_with_translating_rotating_robot():
    tracker = MultiObjectTracker()
    for i in range(70):
        t = i * 0.1
        matrix = np.eye(4)
        matrix[:3, :3] = Rotation.from_euler("xyz", [0.05, -0.1, 0.2 * t]).as_matrix()
        matrix[:3, 3] = [0.3 * t, -0.1 * t, 0.3]
        world = np.array([3 + 0.7 * t, 2 - 0.2 * t, 1.0])
        body = transform_points(world[None], np.linalg.inv(matrix))[0]
        output = tracker.update(
            [Detection3D(t, body, "base_link", "sim")],
            t,
            RobotPose(t, matrix, "odom", "base_link", "sim"),
        )
    np.testing.assert_allclose(output[0].velocity, [0.7, -0.2], atol=0.01)
    assert output[0].motion_state == "MOVING"


def test_extrinsics_require_matching_pose_child_frame():
    c = TrackerConfig(sensor_frame="lidar", T_base_sensor=np.eye(4).tolist())
    pose = RobotPose(0, np.eye(4), "odom", "other_robot", "sim")
    d = Detection3D(0, [1, 0, 0], "lidar", "sim")
    with pytest.raises(ValueError, match="extrinsic"):
        normalize_detections([d], 0, pose, c, "lidar", "sim")
