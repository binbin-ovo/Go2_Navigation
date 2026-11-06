import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from go2_perception.tracking.synthetic import SCENARIOS, generate_sequence
from go2_perception.tracking.replay import (
    frame_from_dict,
    load_detection_sequence,
    recording_frames,
    RecordedPoses,
)
from go2_perception.tracking.multi_object_tracker import MultiObjectTracker
from go2_perception.tracking.config import TrackerConfig
from go2_perception.tracking.evaluation import evaluate
from go2_perception.tracking.normalization import from_lidar, from_fusion
from go2_perception.core.types import FusedDetection, FusionResult
from go2_perception.lidar.processor import LidarProcessor
from go2_perception.io.synthetic import make_scene
from go2_perception.visualization.bev_tracking import render_bev

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_end_to_end_synthetic(scenario):
    tracker = MultiObjectTracker()
    frames = [frame_from_dict(row) for row in generate_sequence(scenario)]
    outputs = [tracker.update(f.detections, f.timestamp, f.robot_pose) for f in frames]
    report = evaluate(frames, outputs, [1.0] * len(frames))
    assert report["position_mae_m"] < 0.2
    assert report["velocity_mae_mps"] < 0.2
    assert report["track_continuity"] > (0.55 if scenario == "sparse" else 0.85)
    if scenario != "crossing":
        assert report["id_switches"] == 0
    assert all(
        np.isfinite(t.state_covariance).all() for tracks in outputs for t in tracks
    )
    if scenario in ("stationary", "ego_motion"):
        assert outputs[-1][0].motion_state == "STATIONARY"
    if scenario == "moving":
        assert outputs[-1][0].motion_state == "MOVING"
        assert len(outputs[-1][0].predicted_positions) == 15


def test_replay_sort_duplicates_invalid_metadata(tmp_path):
    rows = generate_sequence("moving")[:3]
    p = tmp_path / "frames.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows[::-1] + rows[:1]))
    with pytest.raises(ValueError, match="Duplicate"):
        load_detection_sequence(tmp_path)
    frames, duplicates = load_detection_sequence(tmp_path, "first")
    assert duplicates == 1 and [f.timestamp for f in frames] == [0, 0.1, 0.2]
    rows[0]["clock_domain"] = ""
    with pytest.raises(ValueError):
        frame_from_dict(rows[0])


def test_lidar_and_fused_normalization():
    result = LidarProcessor().process(
        make_scene(), timestamp=1, frame_id="base_link", clock_domain="sim"
    )
    normalized = from_lidar(result)
    assert (
        len(normalized) == 2
        and normalized[0].class_name is None
        and normalized[0].confidence is None
    )
    fused = FusionResult(
        [
            FusedDetection(
                0,
                "person",
                0.8,
                1,
                np.ones(3),
                np.zeros(3),
                np.ones(3) * 2,
                1,
                1,
                "base_link",
                "sim",
                0.9,
            )
        ],
        "synthetic",
    )
    assert from_fusion(fused)[0].class_name == "person"


def test_unannotated_metrics_do_not_fabricate_accuracy():
    f = frame_from_dict(generate_sequence("stationary")[0])
    f.ground_truth = None
    report = evaluate([f], [[]], [1])
    assert (
        "position_mae_m" not in report
        and "unavailable" in report["ground_truth_metrics"]
    )


def test_cli_output_and_headless_bev(tmp_path):
    p = tmp_path / "frames.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in generate_sequence("moving")[:12]))
    out = tmp_path / "output"
    subprocess.run(
        [
            sys.executable,
            "scripts/run_tracking.py",
            "--input",
            str(p),
            "--output",
            str(out),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert len((out / "tracks.jsonl").read_text().splitlines()) == 12
    assert (out / "frames/000011.png").is_file()
    assert json.loads((out / "summary.json").read_text())["frames"] == 12
    assert render_bev([], 0, "fixed_sensor:base_link").shape == (750, 1000, 3)


def test_recorded_pose_validation_and_exact_sample_after_gap():
    rows = [
        dict(
            stamp=t,
            frame_id="odom",
            child_frame_id="base_link",
            position=[t, 0, 0],
            q_xyzw=[0, 0, 0, 1],
        )
        for t in (0, 1)
    ]
    poses = RecordedPoses(rows)
    assert poses.lookup(0.5) is None
    assert (
        poses.lookup(1)[0, 3] == 1
    )  # exact observed sample must not need interpolation
    with pytest.raises(ValueError):
        RecordedPoses(rows + rows)


def test_recorded_real_cloud_fixed_mode_and_missing_pose(tmp_path):
    from go2_perception.adapters.go2.pointcloud import LidarFrame
    from go2_perception.adapters.go2.recording import RecordingWriter
    from go2_perception.io.config import load_config

    with np.load(ROOT / "tests/fixtures/go2/real_cloud_pair.npz") as data:
        frame = LidarFrame(
            "rt/utlidar/cloud_base",
            float(data["stamp"]),
            "base_link",
            0.0,
            data["base_xyzi"][:, :3],
            data["base_xyzi"][:, 3],
            data["base_time"],
        )
    writer = RecordingWriter(
        tmp_path / "real", notes="Original committed frame, no pose fabricated"
    )
    writer.write_lidar(frame)
    writer.close()
    config = load_config(ROOT / "configs/go2_lidar.yaml")
    with pytest.raises(ValueError, match="odometry"):
        list(recording_frames(tmp_path / "real", TrackerConfig(), config))
    rows = list(
        recording_frames(tmp_path / "real", TrackerConfig(mode="fixed_sensor"), config)
    )
    assert len(rows) == 1 and rows[0].robot_pose is None and rows[0].detections
    tracks = MultiObjectTracker({"mode": "fixed_sensor"}).update(
        rows[0].detections, rows[0].timestamp
    )
    assert all(t.motion_state == "UNCERTAIN" for t in tracks)


def test_identity_metrics_count_switches_and_unmatched_predictions():
    frames = [frame_from_dict(row) for row in generate_sequence("stationary")[:10]]
    tracker = MultiObjectTracker({"min_hits": 1})
    outputs = [tracker.update(f.detections, f.timestamp, f.robot_pose) for f in frames]
    for tracks in outputs[5:]:
        tracks[0].track_id = 2
    report = evaluate(frames, outputs, [1.0] * 10)
    assert report["id_switches"] == 1
    assert report["idf1"] == pytest.approx(0.5)
    assert report["track_continuity"] == 1


def test_normalization_rejects_untimed_empty_lidar():
    result = LidarProcessor().process(np.empty((0, 3)))
    with pytest.raises(ValueError):
        from_lidar(result)


@pytest.mark.parametrize("window_s", [0, 0.3])
def test_raw_recording_levelling_is_not_applied_twice(tmp_path, window_s):
    from scipy.spatial.transform import Rotation
    from go2_perception.adapters.go2.pointcloud import LidarFrame
    from go2_perception.adapters.go2.recording import RecordingWriter
    from go2_perception.io.config import load_config

    yaw = 0.4
    rotation = Rotation.from_euler("xyz", [0.2, -0.1, yaw]).as_matrix()
    yaw_rotation = Rotation.from_euler("z", yaw).as_matrix()
    level_body = yaw_rotation.T @ rotation
    points = make_scene() @ level_body  # explicitly synthetic tilted-body scan
    writer = RecordingWriter(
        tmp_path / "synthetic_raw", notes="Synthetic poses and clouds"
    )
    for t in (1.0, 1.1):
        writer.write_lidar(
            LidarFrame("rt/utlidar/cloud_base", t, "base_link", None, points)
        )
    for t in (0.98, 1.02, 1.08, 1.12):
        writer.write_state(
            "rt/utlidar/robot_odom",
            dict(
                stamp=t,
                frame_id="odom",
                child_frame_id="base_link",
                position=[1, 2, 0.5],
                q_xyzw=Rotation.from_matrix(rotation).as_quat().tolist(),
            ),
        )
    writer.close()
    frames = list(
        recording_frames(
            tmp_path / "synthetic_raw",
            TrackerConfig(),
            load_config(ROOT / "configs/lidar.yaml"),
            window_s,
        )
    )
    tracker = MultiObjectTracker({"min_hits": 1})
    for f in frames:
        assert f.frame_id == "base_link_level"
        outputs = tracker.update(f.detections, f.timestamp, f.robot_pose)
        assert len(outputs) == 2
        expected = np.array(
            [[2.25, -0.75, 0.125], [4.25, 1.25, 0.225]]
        ) @ yaw_rotation.T + [1, 2, 0.5]
        for center in expected:
            assert min(np.linalg.norm(t.position - center) for t in outputs) < 0.09
    assert outputs[0].timestamp == 1.1
