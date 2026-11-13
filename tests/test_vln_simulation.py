from pathlib import Path
import os

import numpy as np
import pytest

mujoco = pytest.importorskip(
    "mujoco", reason="Optional isolated simulation environment is not installed in core"
)
from go2_perception.vln.simulation import CameraRoom  # noqa: E402

SCENE = Path(__file__).resolve().parents[1] / "configs/simulation_vln_room.xml"


def test_proxy_se2_uses_metres_radians_and_left_positive():
    room = CameraRoom(SCENE)
    try:
        room.pose = np.array([4.3, 0.0, np.pi / 2])
        room._place()
        result = room.advance([0.1, 0.05, -0.1])
        assert result["applied"]
        assert room.pose == pytest.approx([4.25, 0.1, np.pi / 2 - 0.1])
        assert room.clock == 0.25
        before = room.pose.copy()
        assert not room.advance([1, 0, 0])["applied"]
        assert np.array_equal(room.pose, before)
    finally:
        room.close()


def test_proxy_rejects_wall_contact_without_claiming_model_avoidance():
    room = CameraRoom(SCENE)
    try:
        room.pose = np.array([6.6, 1.5, 0.0])
        room._place()
        result = room.advance([0.2, 0, 0])
        assert not result["applied"]
        assert result["reason"] == "simulator_contact_constraint"
        assert room.pose[0] == 6.6
    finally:
        room.close()


@pytest.mark.skipif(
    os.getenv("MUJOCO_RENDER_TEST") != "1",
    reason="Needs an explicit local OpenGL rendering context",
)
def test_actual_camera_rendering():
    room = CameraRoom(SCENE)
    try:
        first = room.render()
        assert first.shape == (256, 448, 3) and first.dtype == np.uint8
        assert first.std() > 5
        room.advance([0, 0, 0.2])
        assert not np.array_equal(first, room.render())
    finally:
        room.close()
