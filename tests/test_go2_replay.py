"""Exercise the merged recording-to-perception CLI without SDK, DDS, or hardware."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from go2_perception.adapters.go2.pointcloud import LidarFrame
from go2_perception.adapters.go2.recording import RecordingWriter

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("window_s", [0, 0.3])
def test_recorded_real_cloud_replays_through_cli(tmp_path, window_s):
    recording, output = tmp_path / "recording", tmp_path / "output"
    with np.load(ROOT / "tests/fixtures/go2/real_cloud_pair.npz") as fixture:
        frame = LidarFrame(
            "rt/utlidar/cloud_base",
            float(fixture["stamp"]),
            "base_link",
            0.0,
            fixture["base_xyzi"][:, :3].copy(),
            fixture["base_xyzi"][:, 3].copy(),
            fixture["base_time"].copy(),
        )
    writer = RecordingWriter(
        recording,
        notes="Real committed cloud; SYNTHETIC stationary odometry for CLI regression only.",
    )
    writer.write_lidar(frame)
    # No measured attitude accompanies this fixture. These identity poses only
    # exercise plumbing and must not be treated as motion-validation evidence.
    for offset in (-0.02, 0.02):
        writer.write_state(
            "rt/utlidar/robot_odom",
            {
                "stamp": frame.acquisition_midpoint + offset,
                "position": [0, 0, 0],
                "q_xyzw": [0, 0, 0, 1],
            },
        )
    writer.close()

    subprocess.run(
        [
            sys.executable,
            "scripts/go2_replay.py",
            "--recording",
            str(recording),
            "--output",
            str(output),
            "--no-rgb",
            "--window-s",
            str(window_s),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    summary = json.loads((output / "summary.json").read_text())["lidar"]
    assert summary["frames_input"] == summary["frames_processed"] == 1
    assert summary["frames_refused_no_pose"] == 0
    assert summary["frames_without_attitude"] == 0
    (result,) = json.loads((output / "lidar_obstacles.json").read_text())
    assert result["timestamp"] == pytest.approx(frame.acquisition_midpoint, abs=1e-6)
    assert result["frame_id"] == "base_link_level"
    assert result["clock_domain"] == "go2_robot"
    assert result["input_points"] == len(frame.xyz)
    assert result["ground_points"] > 50
    assert result["processed_points"] == (
        result["ground_points"]
        + result["unclassified_points"]
        + sum(obstacle["points"] for obstacle in result["obstacles"])
    )
    assert (output / "lidar/000000/lidar.png").is_file()
    assert (output / "lidar/000000/segmented.ply").is_file()
