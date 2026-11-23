"""Regressions from the recorded five-move trials. All motion and hardware opt-ins stay off."""

from copy import deepcopy
from hashlib import sha256
import json
import subprocess
import sys

import cv2
import numpy as np
import pytest

from go2_perception.vln.chunk_replay import chunk_directory, chunk_events
from go2_perception.vln.episode_grounding import detail_features, localize, prepared_detail, request_detail
from go2_perception.vln.interactive_lidar import LidarRecorder, LidarState
from go2_perception.vln.motion_profile import Hold
from go2_perception.vln.source_revision import clean_navigation, source_state


def test_prepared_detail_requires_exact_decision_and_view(tmp_path):
    root = tmp_path / "request"
    (root / "native").mkdir(parents=True)
    path = root / "native/input_0003.png"
    full = root / "frame_0000.png"
    cv2.imwrite(str(full), np.zeros((90, 160), dtype="uint8"))
    image_hash = sha256(full.read_bytes()).hexdigest()
    source = dict(
        frame_id="f3",
        clock_epoch="boot",
        observation_s=2.0,
        available_s=2.01,
        received_monotonic_s=22.0,
        exposure_monotonic_s=21.95,
    )
    image = dict(file_sha256="a" * 64, causal_grid_index=3, job_frame_index=0)
    pred = dict(
        backend="lightnav",
        instruction="Find the box",
        source_frame=source,
        source_image=image,
        native_output="native",
    )
    job = dict(
        instruction=pred["instruction"],
        decision_time_s=2.02,
        clock_epoch="boot",
        decision_id="d",
        episode_id="e",
        frames=[dict(path=str(full), sha256=image_hash, observation_s=2.0, available_s=2.01)],
    )
    view = dict(
        rectified=True,
        distortion_model="pinhole_radtan",
        size_wh=[160, 90],
        crop_xywh=[0, 0, 160, 90],
        padding_ltrb=[0] * 4,
        resize_scale_xy=[1, 1],
    )
    snap = dict(
        current_source=source,
        history=[dict(source=source, rectification=dict(view=view))],
        source_cutoff_s=2.02,
        bounded_context=dict(source_indices=[3]),
    )
    result = dict(
        status="decoded",
        native_output="native",
        native_selected_frame_indices=[3],
        clock_epoch="boot",
        decision_id="d",
        episode_id="e",
    )
    for name, value in (("job.json", job), ("native/result.json", result), ("snapshot.json", snap)):
        (root / name).write_text(json.dumps(value))
    evidence = request_detail(pred, root, path)
    assert detail_features(evidence, "a" * 64)[-1] == (160, 90)
    with pytest.raises(Hold, match="frame_binding"):
        prepared_detail(dict(pred, source_frame=dict(source, exposure_monotonic_s=21.96)), snap, job, path)
    with pytest.raises(Hold, match="frame_binding"):
        prepared_detail(pred, snap, job, root / "native/input_0002.png")
    with pytest.raises(Hold, match="native_decision"):
        request_detail(dict(pred, native_output="different"), root, path)
    with pytest.raises(Hold, match="native_hash_mismatch"):
        detail_features(evidence, "b" * 64)
    cropped = deepcopy(snap)
    cropped["history"][0]["rectification"]["view"]["crop_xywh"][0] = 1
    assert prepared_detail(pred, cropped, job, path) is None
    full.write_bytes(full.read_bytes() + b"x")
    with pytest.raises(Hold, match="image_changed"):
        detail_features(evidence, "a" * 64)


def test_interior_sift_requires_native_mark_on_current_inliers():
    # A translated texture supports the upper-left portion of a much larger box.
    pts = np.float32([(x, y) for y in (10, 24, 38, 52) for x in (11, 25, 39, 53)])
    desc = np.random.default_rng(51).normal(size=(len(pts), 128)).astype("float32")
    ref = dict(
        roi=[0, 0, 1, 1],
        size=(100, 100),
        points=pts,
        descriptors=desc,
        operator="reviewer",
        sha256="1" * 64,
        prepared_sha256="2" * 64,
    )
    current = (pts + 2, desc.copy(), (100, 100))
    for point, inside in (([0.3, 0.3], True), ([0.8, 0.8], False), (None, None)):
        evidence = localize(ref, current, point, interior=True)
        assert evidence["localized"] and evidence["inliers"] == 16
        assert evidence["mark_inside"] is inside and evidence["mark_in_inlier_hull"] is inside
        if point is not None:
            assert evidence["mark_in_region"]


def test_chunk_replay_round_trip_and_explicit_gap(tmp_path):
    root = tmp_path / "session/lidar"
    rec = LidarRecorder(root)
    xyz = np.float32([[2.0, -1.0, 0.4], [1.0, 2.0, 0.3]])
    rec.start("T1")
    try:
        for i in (1, 2):
            epoch = f"boot{i}"
            common = dict(wall_minus_monotonic_s=90.0, stamp=8.0 + i)
            rec.offer(
                "odom",
                dict(
                    common,
                    seq=20 + i,
                    host_recv_wall_s=100.0 + i,
                    position=[1.0, 2.0, 0.0],
                    q_xyzw=[0.0, 0.0, 0.0, 1.0],
                ),
                None,
                epoch,
                10.0 + i,
            )
            rec.offer(
                "scan",
                dict(common, seq=30 + i, host_recv_wall_s=100.01 + i, time_span_s=None),
                xyz * i,
                epoch,
                10.01 + i,
            )
    finally:
        rec.close()
    assert chunk_directory(tmp_path) == chunk_directory(tmp_path / "session") == chunk_directory(root) == root
    events = list(chunk_events(root))
    assert [e[1] for e in events] == ["odom", "scan", "replay_boundary", "odom", "scan"]
    assert events[2][2]["reason"] == "recorded_epoch_changed"
    assert events[1][0] == pytest.approx(0.01) and events[-1][0] == pytest.approx(1.01)
    assert events[0][2]["seq"] == 21 and "linear" not in events[0][2]
    assert events[-1][2]["recorded_robot_epoch"] == "boot2" and events[-1][2]["time_span_s"] is None
    assert np.array_equal(events[-1][3], xyz * 2)
    assert all(e[1] != "odom" for e in chunk_events(root, no_odom=True))
    state = LidarState()
    state.scan(dict(events[1][2], host_recv_wall_s=200.0, wall_minus_monotonic_s=180.0), xyz)
    state.replay_boundary()
    assert state.latest is None and state.last_odom is None and not state.pending_windows
    path = root / "index.jsonl"
    records = list(map(json.loads, path.read_text().splitlines()))
    path.write_text(
        "\n".join(
            json.dumps(r) for r in [records[1], dict(records[0], dropped_not_recorded=True, reason="quota")]
        )
    )
    assert next(chunk_events(root))[2]["reason"] == "quota"
    recfile = root / records[1]["file"]
    recfile.write_bytes(recfile.read_bytes() + b"x")
    with pytest.raises(ValueError, match="SHA-256"):
        list(chunk_events(root))


def test_clean_source_required_for_capability_and_live_launch(tmp_path):
    def git(*args):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)

    git("init")
    git("-c", "user.name=fixture", "-c", "user.email=f@local", "commit", "--allow-empty", "-m", "fixture")
    clean = clean_navigation(tmp_path)
    assert not clean["dirty"] and len(clean["commit"]) == 40
    capability_id = f"go2-hw-{clean['commit']}-12345"
    assert clean_navigation(tmp_path, capability_id) == clean
    with pytest.raises(Hold, match="source_mismatch"):
        clean_navigation(tmp_path, "go2-hw-123abcd-12345")
    (tmp_path / "navigation.py").write_text("edited")
    assert source_state(tmp_path)["status_porcelain"] == ["?? navigation.py"]
    with pytest.raises(Hold, match="clean_navigation"):
        clean_navigation(tmp_path, capability_id)
    git("add", "navigation.py")
    first_hash = source_state(tmp_path)["tracked_diff_sha256"]
    (tmp_path / "navigation.py").write_text("different edit")
    assert source_state(tmp_path)["tracked_diff_sha256"] != first_hash


def test_public_supervised_limits_are_forwarded_and_validated(monkeypatch):
    import app

    commands = []
    monkeypatch.setattr(app.os, "execl", lambda *args: commands.append(args))
    app.supervised_operator(
        [
            "--prepare-only",
            "--move-budget",
            "2",
            "--episode-seconds",
            "45",
            "--area-radius-m",
            "3",
            "--alignment-deg",
            "14",
        ]
    )
    assert commands[0][-8:] == (
        "--move-budget",
        "2",
        "--episode-seconds",
        "45.0",
        "--area-radius-m",
        "3.0",
        "--alignment-deg",
        "14.0",
    )
    for flags in (["--move-budget", "6"], ["--alignment-deg", "nan"], ["--episode-seconds", "361"]):
        with pytest.raises(SystemExit):
            app.supervised_operator(["--prepare-only", *flags])
    for args in (["--min-free-disk-gb", "0"], ["--run-quota-gb", "nan"]):
        with pytest.raises(SystemExit):
            app.main(args)
    assert len(commands) == 1
    app.supervised_operator(["--prepare-only", "--episode-seconds", "360"])  # room for re-identification
    assert commands[1][-5:-4] == ("360.0",)


@pytest.mark.parametrize("profile,version", [("legacy-first-row", 1), ("go2-effective-v1", 2)])
def test_generated_capability_with_full_commit_obeys_existing_contract(
    tmp_path, monkeypatch, profile, version
):
    monkeypatch.setattr(sys, "path", list(sys.path))
    from hardware import make_capability as builder
    from go2_perception.vln.supervised import validate_record

    heading = tmp_path / "heading.json"
    heading.write_text(json.dumps({"accepted_yaw_deg": [0.835]}))
    output = tmp_path / "capability.json"
    monkeypatch.setattr(builder, "CONTROL_ROOT", tmp_path)
    monkeypatch.setattr(builder, "CONTROL_LOGS", {})
    monkeypatch.setattr(builder, "HEADING_SHA256", sha256(heading.read_bytes()).hexdigest())
    monkeypatch.setattr(builder, "clean_navigation", lambda _: {"commit": "c" * 40})
    monkeypatch.setattr(
        builder, "git", lambda root, *args: builder.CONTROL_REVISION if args[0] == "rev-parse" else ""
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["make_capability", "--profile", profile, "--heading-file", str(heading), "--output", str(output)],
    )
    builder.main()
    cap = json.loads(output.read_bytes())
    validate_record("capability", cap, version)
    assert cap["capability_id"].startswith(f"go2-hw-{'c' * 40}-")
