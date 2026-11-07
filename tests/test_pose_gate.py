"""Pose gating of LiDAR frames against odometry. All clouds and poses here are synthetic."""

import json
from pathlib import Path
import subprocess
import sys
import pytest
from go2_perception.adapters.go2.pointcloud import LidarFrame
from go2_perception.adapters.go2.recording import RecordingWriter
from go2_perception.io.config import load_config
from go2_perception.io.synthetic import make_scene
from go2_perception.tracking.config import TrackerConfig
from go2_perception.tracking.multi_object_tracker import MultiObjectTracker
from go2_perception.tracking.pose_gate import (
    AFTER_LAST_ODOMETRY,
    BEFORE_FIRST_ODOMETRY,
    NO_ODOMETRY,
    ODOMETRY_GAP,
    ODOMETRY_WAIT_TIMEOUT,
    PoseAlignmentConfig,
    PoseGate,
)
from go2_perception.tracking.replay import RecordedPoses, recording_frames

ROOT = Path(__file__).resolve().parents[1]


def odom(t, x):
    return dict(
        stamp=t,
        frame_id="odom",
        child_frame_id="base_link",
        position=[x, 0.0, 0.0],
        q_xyzw=[0, 0, 0, 1],
    )


def gate_for(stamps, speed=1.0, **config):
    poses = RecordedPoses([odom(t, speed * t) for t in stamps])
    return PoseGate(poses, PoseAlignmentConfig(**config))


def test_lidar_before_first_odometry_waits_then_is_skipped():
    gate = gate_for([1.0, 1.05, 1.1])
    gate.push(0.98, "early")
    assert list(gate.release()) == []  # buffered: no odometry has arrived yet
    gate.note_pose(1.0)
    gate.push(1.02, "late")
    assert list(gate.release()) == []  # 1.02 still waits for odometry after it
    gate.note_pose(1.05)
    released = list(gate.release())
    assert [item for _, item, _ in released] == ["late"]
    assert [(s.timestamp, s.reason) for s in gate.skipped] == [(0.98, BEFORE_FIRST_ODOMETRY)]
    assert gate.held == 0


def test_interpolated_pose_between_bracketing_samples():
    gate = gate_for([1.0, 1.05], speed=2.0)
    gate.note_pose(1.0)
    gate.note_pose(1.05)
    gate.push(1.02, "frame")
    ((t, _, matrix),) = gate.release()
    assert t == 1.02 and matrix[0, 3] == pytest.approx(2.04)
    assert not gate.skipped


def test_prolonged_gap_skips_and_resumption_after_gap():
    stamps = [1.0, 1.05, 2.0, 2.05]  # 0.95 s outage, beyond the 0.1 s interpolation limit
    gate = gate_for(stamps)
    for t in stamps:
        gate.note_pose(t)
    for t in (1.02, 1.5, 1.9, 2.02):
        gate.push(t, t)
    released = [t for t, _, _ in gate.release()]
    assert released == [1.02, 2.02]
    assert [(s.timestamp, s.reason) for s in gate.skipped] == [
        (1.5, ODOMETRY_GAP),
        (1.9, ODOMETRY_GAP),
    ]


def test_live_wait_timeout_and_end_of_stream():
    gate = gate_for([1.0])
    gate.push(1.5, "stale", wait_start=10.0)
    assert list(gate.release(now=10.1)) == []
    assert list(gate.release(now=10.3)) == []
    assert gate.skipped[0].reason == ODOMETRY_WAIT_TIMEOUT
    gate.note_pose(1.0)
    gate.push(1.2, "tail")
    assert list(gate.release(final=True)) == []
    assert gate.skipped[-1].reason == AFTER_LAST_ODOMETRY
    empty = gate_for([1.0])
    empty.push(0.5, "x")
    assert list(empty.release(final=True)) == [] and empty.skipped[0].reason == NO_ODOMETRY


def test_backward_hold_requires_explicit_justification():
    with pytest.raises(ValueError, match="justification"):
        PoseAlignmentConfig(start_hold_s=0.05)
    gate = gate_for(
        [1.0, 1.05],
        start_hold_s=0.05,
        start_hold_justification="synthetic test: robot verified stationary",
    )
    gate.note_pose(1.0)
    gate.push(0.98, "held")
    gate.push(0.9, "too early")
    gate.note_pose(1.05)
    gate.push(1.0, "exact")
    released = list(gate.release())
    assert [item for _, item, _ in released] == ["held", "exact"]
    assert gate.held == 1 and gate.skipped[0].reason == BEFORE_FIRST_ODOMETRY


def write_recording(root, lidar_times, odom_times):
    writer = RecordingWriter(root, notes="Synthetic poses and clouds")
    for t in lidar_times:
        writer.write_lidar(LidarFrame("rt/utlidar/cloud_base", t, "base_link", None, make_scene()))
    for t in odom_times:
        writer.write_state("rt/utlidar/robot_odom", odom(t, 0.0))
    writer.close()


def test_recording_skips_early_and_gap_frames_and_resets_tracker(tmp_path):
    lidar = [0.95, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.2, 2.3, 2.4]
    odometry = [round(1.0 + 0.02 * i, 2) for i in range(26)]  # 1.00-1.50
    odometry += [round(2.0 + 0.02 * i, 2) for i in range(26)]  # 2.00-2.50
    write_recording(tmp_path / "rec", lidar, odometry)
    report = {}
    frames = list(
        recording_frames(
            tmp_path / "rec",
            TrackerConfig(),
            load_config(ROOT / "configs/lidar.yaml"),
            report=report,
        )
    )
    assert [f.timestamp for f in frames] == [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.2, 2.3, 2.4]
    assert [(s.timestamp, s.reason) for s in report["skipped"]] == [
        (0.95, BEFORE_FIRST_ODOMETRY)
    ]
    # 1.5 -> 2.2 has no skipped LiDAR frame, so it is not an odometry outage.
    assert not any(f.reset_before for f in frames)

    write_recording(tmp_path / "gap", [1.0, 1.1, 1.6, 1.7, 2.1, 2.2], odometry)
    report = {}
    frames = list(
        recording_frames(
            tmp_path / "gap",
            TrackerConfig(),
            load_config(ROOT / "configs/lidar.yaml"),
            report=report,
        )
    )
    assert [f.timestamp for f in frames] == [1.0, 1.1, 2.1, 2.2]
    assert [s.reason for s in report["skipped"]] == [ODOMETRY_GAP, ODOMETRY_GAP]
    resumed = frames[2]
    assert resumed.reset_before and resumed.pose_outage_s == pytest.approx(1.0)
    assert report["resets"] == [dict(timestamp=2.1, pose_outage_s=pytest.approx(1.0))]

    tracker = MultiObjectTracker({"min_hits": 1})
    first_ids = set()
    for f in frames:
        ended = tracker.reset() if f.reset_before else []
        tracks = tracker.update(f.detections, f.timestamp, f.robot_pose)
        if f.timestamp < 2:
            first_ids |= {t.track_id for t in tracks}
        else:
            assert {t.track_id for t in tracks}.isdisjoint(first_ids)
        if f.reset_before:
            assert set(ended) == first_ids


def test_recording_without_any_valid_pose_still_fails(tmp_path):
    write_recording(tmp_path / "rec", [0.5, 0.6], [1.0, 1.02])
    with pytest.raises(ValueError, match="odometry"):
        list(recording_frames(tmp_path / "rec", TrackerConfig(), load_config(ROOT / "configs/lidar.yaml")))


def test_cli_reports_skipped_frames(tmp_path):
    odometry = [round(1.0 + 0.02 * i, 2) for i in range(16)]
    write_recording(tmp_path / "rec", [0.97, 1.0, 1.1, 1.2, 1.3], odometry)
    out = tmp_path / "out"
    subprocess.run(
        [
            sys.executable,
            "scripts/run_tracking.py",
            "--input",
            str(tmp_path / "rec"),
            "--output",
            str(out),
            "--lidar-config",
            "configs/lidar.yaml",
            "--no-render",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    summary = json.loads((out / "summary.json").read_text())
    assert summary["frames"] == 4
    assert summary["pose_alignment"]["frames_input"] == 5
    assert summary["pose_alignment"]["skipped_by_reason"] == {BEFORE_FIRST_ODOMETRY: 1}
    skipped = [json.loads(line) for line in (out / "skipped_frames.jsonl").read_text().splitlines()]
    assert skipped == [dict(timestamp=0.97, reason=BEFORE_FIRST_ODOMETRY)]
    assert (tmp_path / "rec" / "lidar").is_dir()  # raw recording left in place
