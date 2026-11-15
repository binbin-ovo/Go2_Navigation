"""Optional numeric MuJoCo checks and actual rendered landmark validation."""

import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

pytest.importorskip(
    "mujoco", reason="Use the isolated Phase 6C model/simulator environment"
)

from go2_perception.vln.simulation import CameraRoom  # noqa: E402
from go2_perception.vln.waypoints import PrefixLimits, to_world  # noqa: E402

SCENE = Path(__file__).resolve().parents[1] / "configs/simulation_vln_room.xml"


@pytest.mark.parametrize(
    "chunk",
    [
        [[0.4, 0, 0]],
        [[0, 0, 1]],
        [[0, 0, -1]],
        [[0.3, 0.1, 0.5]],
        [[0.1, 0, 0.1], [0.2, 0.1, 0.3], [0.3, 0.15, 0.5]],
        [[0, 0, 0], [0.2, 0, 0]],
    ],
)
def test_actual_proxy_follows_known_paths(chunk):
    room = CameraRoom(SCENE)
    try:
        room.pose = np.array([3.2, -0.2, -0.4])
        room._place()
        expected = to_world(chunk, room.pose)[-1]
        result = room.execute_prefix(chunk, limits=PrefixLimits(window_s=2))
        assert result["applied"]
        assert room.pose == pytest.approx(expected, abs=1e-10)
        assert result["path_error"]["max_cross_track_m"] < 1e-10
        assert room.clock == pytest.approx(2)
    finally:
        room.close()


def test_collision_interrupt_keeps_last_safe_pose_and_reports_rejected_candidate():
    room = CameraRoom(SCENE)
    try:
        room.pose = np.array([6.5, 1.5, 0.0])
        room._place()
        result = room.execute_prefix([[1, 0, 0]], limits=PrefixLimits(window_s=2))
        assert not result["applied"]
        assert result["constraint_rejection"]["contacts"]
        assert 6.5 < room.pose[0] < 6.9
        assert not room.contacts()
        assert 0 < room.clock < 2
        assert result["actual_endpoint"] != result["planned_endpoint"]
    finally:
        room.close()


def test_goal_is_reachable_by_known_geometry_diagnostic_only():
    room = CameraRoom(SCENE)
    try:
        room.pose = np.array([4.3, 0, 0.0])
        room._place()
        result = room.execute_prefix([[1.85, 0, 0]], limits=PrefixLimits(window_s=3))
        assert result["applied"]
        assert room.evaluation()["inside_stopping_region"]
        assert room.evaluation()["distance_to_goal_center_m"] < 1e-10
        stopped = room.execute_prefix(np.zeros((10, 3)), native_stop=True)
        assert stopped["simulated_dt_s"] == 0
        assert not room.contacts()
    finally:
        room.close()


def test_baseline_remains_equivalent_and_exposes_leading_zero_limitation():
    a, b = CameraRoom(SCENE), CameraRoom(SCENE)
    try:
        chunk = [[0.1, 0.03, -0.1]] * 10
        legacy = a.advance(chunk[0])
        traced = b.execute_waypoint_policy(chunk, "first_waypoint")
        assert np.array_equal(a.pose, b.pose) and a.clock == b.clock
        assert traced["legacy_baseline_result"] == legacy
        chunk = [[0, 0, 0]] + [[0.3, 0, 0]] * 9
        before = b.pose.copy()
        hold = b.execute_waypoint_policy(chunk, "first_waypoint")
        assert not hold["native_stop"] and np.array_equal(b.pose, before)
        forward = b.execute_waypoint_policy(chunk, "bounded_prefix")
        assert np.linalg.norm(b.pose[:2] - before[:2]) == pytest.approx(0.2)
        assert forward["selected_indices"] == [1]
    finally:
        a.close()
        b.close()


@pytest.mark.skipif(
    os.getenv("MUJOCO_RENDER_TEST") != "1", reason="Requires explicit OpenGL rendering"
)
def test_rendered_landmarks_camera_pose_pixel_axes_and_following(tmp_path):
    tree = ET.parse(SCENE)
    body = tree.getroot().find("worldbody")
    landmarks = [
        ([5.46, 0.4, 0.9], "1 0 0 1"),
        ([5.46, -0.4, 0.9], "0 1 0 1"),
        ([5.46, 0.4, 0.4], "0 0 1 1"),
        ([5.46, -0.4, 0.4], "1 1 0 1"),
    ]
    for i, (xyz, color) in enumerate(landmarks):
        ET.SubElement(
            body,
            "geom",
            name=f"axis_marker_{i}",
            type="sphere",
            size="0.04",
            pos=" ".join(map(str, xyz)),
            rgba=color,
        )
    scene = tmp_path / "axis_scene.xml"
    tree.write(scene)
    room = CameraRoom(scene)
    try:
        room.pose = np.array([4.3, 0, 0.0])
        room._place()
        for action in (None, [0.1, 0, 0.1]):
            if action:
                room.execute_prefix([action])
            rgb = room.render().astype(float)
            cid = room.model.camera("rgb").id
            camera_position = room.data.cam_xpos[cid].copy()
            rotation = room.data.cam_xmat[cid].reshape(3, 3)
            c, s = math.cos(room.pose[2]), math.sin(room.pose[2])
            assert camera_position == pytest.approx(
                [room.pose[0] + 0.16 * c, room.pose[1] + 0.16 * s, 0.65]
            )
            assert -rotation[:, 2] == pytest.approx([c, s, 0], abs=1e-10)
            f = 128 / math.tan(math.radians(30))
            r, g, b = rgb.transpose(2, 0, 1)
            masks = [
                (r > 2 * g) & (r > 2 * b) & (r > 70),
                (g > 2 * r) & (g > 2 * b) & (g > 70),
                (b > 2 * r) & (b > 2 * g) & (b > 70),
                (r > 2 * b) & (g > 2 * b) & (r > 70) & (g > 70),
            ]
            for (xyz, _), mask in zip(landmarks, masks):
                camera = rotation.T @ (np.array(xyz) - camera_position)
                u = 224 + f * camera[0] / -camera[2]
                v = 128 - f * camera[1] / -camera[2]
                # Restrict to expected marker neighborhood; furniture has other reds/blues.
                yy, xx = np.indices(mask.shape)
                pixels = np.where(mask & ((xx - u) ** 2 + (yy - v) ** 2 < 15**2))
                assert len(pixels[0]) > 30
                assert [pixels[1].mean(), pixels[0].mean()] == pytest.approx(
                    [u, v], abs=2.0
                )
    finally:
        room.close()
