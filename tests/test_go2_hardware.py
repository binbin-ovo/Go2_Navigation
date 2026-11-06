"""Live checks. Skipped unless GO2_HARDWARE=1, so offline runs don't fail without a robot.

Run: GO2_HARDWARE=1 .venv/bin/python -m pytest -q -m hardware tests/test_go2_hardware.py
Passive only: the test subscribes to streams and sends no commands.
"""

import os
import threading

import pytest

pytestmark = [
    pytest.mark.hardware,
    pytest.mark.skipif(os.environ.get("GO2_HARDWARE") != "1", reason="set GO2_HARDWARE=1"),
]


def test_live_lidar_state_and_camera_deliver_data():
    from go2_perception.adapters.go2.pointcloud import to_lidar_message
    from go2_perception.adapters.go2.receivers import (
        ODOM_TOPIC,
        Go2CameraReceiver,
        Go2DdsReceiver,
    )

    clouds, odom, frames, done = [], [], [], threading.Event()

    def check():
        if clouds and odom and frames:
            done.set()

    def on_lidar(frame):
        clouds.append(frame)
        check()

    def on_state(topic, record):
        if topic == ODOM_TOPIC:
            odom.append(record)
            check()

    def on_frame(frame):
        frames.append(frame)
        check()

    dds = Go2DdsReceiver("eno2", ["rt/utlidar/cloud_base"], on_lidar, on_state)
    camera = Go2CameraReceiver(on_frame=on_frame)
    dds.start()
    camera.start()
    try:
        assert done.wait(10), f"clouds={len(clouds)} odom={len(odom)} frames={len(frames)}"
    finally:
        camera.stop()
        camera.join(timeout=3)
        dds.stop()
    assert camera.exception is None
    assert clouds[-1].frame_id == "base_link" and len(clouds[-1].xyz) > 100
    assert to_lidar_message(clouds[-1], odom[-1]["q_xyzw"]).frame_id == "base_link_level"
    assert frames[-1].image.shape[2] == 3
