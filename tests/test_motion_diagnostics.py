"""Offline regressions for the November 19 motion records. All inputs and routes are local fakes."""
from copy import deepcopy
import math
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from go2_perception.vln.interactive_lidar import LidarRecorder
from go2_perception.vln.interactive_trials import expand_target, read_jsonl
from go2_perception.vln.motion_profile import INSPECTED
from go2_perception.vln.supervised import Hold, digest, plan
from go2_perception.vln.supervised_app import NavigationUI
from go2_perception.vln.supervised_executor import result_record
from test_effective_motion import attach_camera_evidence, configure, proposed, straight
from test_hardware_guards import Clock, floor, make_guards, wall
from test_supervised_executor import hardware_fakes, idle_run
from test_supervised_navigation import ticket_for
from test_vln_interactive import make_app, odom_header, scan_header


@pytest.mark.parametrize("fault,predicate", [
    ("no_floor", "geometry_unavailable"), ("short", "insufficient_clearance"),
    ("old", "age_outside_window"), ("provider_error", None),
])
def test_exact_transient_guard_is_retained_before_cleanup(tmp_path, fault, predicate):
    motion, ticket, cancel, calls, clock = hardware_fakes(tmp_path)
    configure(motion.cap)
    attach_camera_evidence(motion, ticket)
    ticket["proposal"]["waypoints"] = straight() + [[1.05, 0.0, 0.0]] * 3
    action = plan(ticket["proposal"], motion.cap)
    ticket["action"] = ticket["authorization"]["envelope"] = action
    ticket["authorization"]["capability_sha256"] = digest(motion.cap)
    idle_run(motion, clock, 1.1)
    original = motion.guards.snapshot
    injected = []

    def snapshot():
        snap = original()
        if "velocity" in calls and not injected:
            geo = snap["geometry"]
            snap["scan_sources"] = {"scan_stamps_robot_s": [8.25], "scan_receipts_monotonic_s": [clock() - .03]}
            snap["geometry_detail"] = {"reason": "floor_not_observed" if fault == "no_floor" else "synthetic"}
            if fault == "no_floor":
                geo.update(available=False, forward_clear_m=0.0, directions=[])
            elif fault == "short":
                geo["forward_clear_m"] = .17
            elif fault == "old":
                geo["monotonic_s"] -= .9
            injected.append(deepcopy(snap))
            if fault == "provider_error":
                raise RuntimeError("injected_guard_loss")
        return snap

    motion.guards.snapshot = snapshot
    result = motion(ticket, action, result_record(ticket, False, clock()), cancel)
    rows = [e for e in result["events"] if e.get("event") == "guard_rejected"]
    assert len(rows) == 1 and calls.count("velocity") == 1
    assert calls[-4:] == ["stop", "stop", "stop", "release"]
    event = rows[0]
    assert event["request_id"] == ticket["request_id"] and event["nonzero_sends"] == 1
    assert event["reason"] in result["reason"] and snapshot()["geometry"]["available"]
    if fault == "provider_error":
        assert event["guards"] is None and event["stage"] == "snapshot"
        assert event["exception_type"] == "RuntimeError" and event["reason"] == "injected_guard_loss"
    else:
        assert event["stage"] == "current_geometry_and_pose" and event["guards"] == injected[0]
        assert event["details"]["predicate"] == predicate
        assert event["checked_monotonic_s"] <= event["monotonic_s"]
        if fault == "short":
            assert event["details"]["available"] == .17
            assert event["details"]["required"] == pytest.approx(.85)
        if fault == "old":
            assert event["checked_monotonic_s"] - event["guards"]["geometry"]["monotonic_s"] == pytest.approx(.9)


def test_forward_evidence_preserves_both_inputs_and_source_selection(tmp_path):
    clock = Clock()
    guards = make_guards(tmp_path, clock)

    def observe(x, distance):
        robot = clock() - 90
        guards.odom.update(robot, x, 0, 0, raw_pose=(x, 0, .3, 0, 0, 0, 1))
        points = floor() if distance is None else np.concatenate([floor(), wall(distance)])
        guards.scans.add(points, stamp_robot_s=robot - .012)

    observe(0, 2.)
    sample = guards.progress({"type": "forward", "direction": "forward"})
    saved = []
    for distance, expected_lidar, value, decision, selected in (
        (1.74, .26, .26, "accepted_max", "lidar"),
        (1.89, .11, .20, "accepted_max", "odom"),
        (.85, 1.15, .20, "disagreement_rejected", "odom"),
        (None, None, .20, "current_surface_unavailable", "odom"),
    ):
        clock.t += .6  # independent fresh windows, same robot epoch
        observe(.2, distance)
        assert sample() == pytest.approx(value)
        b = sample.evidence
        assert b["reference_odometry"]["stamp_robot_s"] == 10.
        assert b["odometry"]["position_xyz_q_xyzw"] == (.2, 0, .3, 0, 0, 0, 1)
        assert b["odometry"]["received_monotonic_s"] == clock()
        assert b["reference_scans"]["scan_stamps_robot_s"] == pytest.approx([9.988])
        assert b["scans"]["scan_stamps_robot_s"] == pytest.approx([clock() - 90.012])
        assert b["scans"]["scan_receipts_monotonic_s"] == [clock()]
        assert b["scans"]["scan_pose_stamps_robot_s"] == [clock() - 90]
        assert b["scans"]["scan_poses_xy_yaw"] == [(.2, 0, 0)]
        assert b["scans"]["current_pose_xy_yaw"] == (.2, 0, 0)
        assert b["ground_z_m"] == pytest.approx(b["reference_ground_z_m"])
        assert b["lidar_minus_odom_m"] == (None if expected_lidar is None else pytest.approx(expected_lidar-.2))
        assert b["reference_surface_x_m"] == pytest.approx(2.) and b["surface_x_m"] == (None if distance is None else pytest.approx(distance))
        assert b["odom_m"] == pytest.approx(.20) and b["fused_m"] == pytest.approx(value)
        assert b["lidar_m"] == (None if expected_lidar is None else pytest.approx(expected_lidar))
        assert (b["fusion_decision"], b["selected_input"], b["lidar_accepted"]) == (
            decision, selected, decision == "accepted_max")
        saved.append(b)
    clock.t += .7
    assert sample() is None
    assert sample.evidence["fusion_decision"] == "odometry_missing_or_stale" and sample.evidence["fused_m"] is None
    guards.odom.update(clock() - 90, .20, 0, 0)
    assert sample() == pytest.approx(.2) and sample.evidence["fusion_decision"] == "scan_window_unavailable"
    assert sample.evidence["scans"]["scan_receipts_monotonic_s"] == [clock() - .7]
    guards.odom.update(1., 0, 0, 0)
    assert sample() is None and sample.evidence["fusion_decision"] == "odometry_epoch_changed"
    assert saved[0]["fusion_decision"] == "accepted_max" and saved[0]["lidar_m"] == pytest.approx(.26)


@pytest.mark.parametrize("yaw", [math.radians(6), .2505, .3056, .3447, math.radians(19.99)])
@pytest.mark.parametrize("sign", [-1, 1])
def test_gap_never_becomes_straight_or_twenty_degrees(yaw, sign):
    direction = "left" if sign > 0 else "right"
    cap, p = proposed([[.10, 0., sign * yaw], [.39, .01 * sign, sign * yaw]],
                      (.2 if sign > 0 else .8, .5), INSPECTED)
    raw = deepcopy(p["waypoints"])
    with pytest.raises(Hold, match="heading_gap") as held:
        plan(p, cap)
    assert p["waypoints"] == raw
    assert held.value.details["native_direction"] == direction
    assert held.value.details["available_magnitude"] == pytest.approx(yaw)
    assert held.value.details["minimum"] == pytest.approx(math.radians(20))


@pytest.mark.parametrize("sign", [-1, 1])
def test_gap_boundaries_and_contiguous_support(sign):
    angle = math.radians(20)
    side = (.2 if sign > 0 else .8, .5)
    cap, p = proposed([[.1, 0, .22 * sign], [.2, .01 * sign, angle * sign]], side, INSPECTED)
    a = plan(p, cap)
    assert a["type"] == "turn" and a["magnitude"] == pytest.approx(angle) and a["speed"] == 2.
    p["waypoints"].insert(1, [0., 0., -.12 * sign])
    p["waypoints"].pop()
    with pytest.raises(Hold, match="heading_gap") as held:
        plan(p, cap)
    assert held.value.details["available_magnitude"] == pytest.approx(.22)
    cap, p = proposed([[.4, 0., sign * .10]])
    cap["mapping"].update(yaw_rad=0., yaw_uncertainty_rad=0.)
    assert plan(p, cap)["type"] == "forward"


def test_volume_limits_and_loss_metadata_under_stalled_disk(tmp_path):
    entered, release = threading.Event(), threading.Event()
    def quota():
        entered.set()
        return release.wait(5)

    rec = LidarRecorder(tmp_path, queue_size=1, max_chunk_bytes=48, max_messages=2, quota=quota)
    rec.start("refresh")
    xyz = np.ones((4, 3), np.float32)
    try:
        for i in range(2):
            rec.offer("scan", scan_header(i, i, 1100), xyz, "boot", 100)
        assert entered.wait(2)
        for i in range(2, 100):
            rec.offer("scan", scan_header(i, i, 1100), xyz, "boot", 100)
        rec.start("other")
        rec.offer("scan", scan_header(100, 1, 1100.5), np.ones((5, 3), np.float32), "other-boot", 100.5)
        rec.close(timeout=.01)
        assert len(rec.gaps) <= rec.MAX_GAP_ROWS and rec.gap_summary["n_chunks"] > 1
        assert rec.status()["queued"] <= 1 and not rec.status()["drain_complete"]
    finally:
        release.set()
        rec.close()
    rows = read_jsonl(tmp_path / "index.jsonl")
    group = next(r for r in rows if r.get("n_chunks"))
    assert group["robot_epoch"] is None and group["source_interval_robot_s"] is None
    assert group["host_receipt_interval_monotonic_s"] == [100., 100.5]
    assert rec.status()["drain_complete"] and sum(r["n_scans"] for r in rows) == 101
    assert sum(r.get("n_chunks", 1) for r in rows if r["dropped_not_recorded"]) == rec.counts["chunks_dropped"]
    assert sum(r["n_points"] for r in rows) == 405
    assert all(r["n_points"] == 4 for r in rows if r.get("file"))
    # Reopening the same index cannot reuse an ID hidden in a coalesced drop.
    resumed = LidarRecorder(tmp_path, max_messages=2)
    resumed.start("later")
    for i in range(5):
        resumed.offer("odom", odom_header(i, 2., 1200.), None, "later", 200)
    resumed.close()
    later = [r for r in read_jsonl(tmp_path / "index.jsonl") if r["trial_id"] == "later"]
    assert [r["n_odom"] for r in later] == [2, 2, 1]
    assert later[0]["chunk"] == 102


@pytest.mark.parametrize("continuous,settled", [(False, True), (True, True), (False, False)])
def test_cancel_keeps_raw_samples_through_cleanup(tmp_path, continuous, settled):
    app, clock, _, _ = make_app(tmp_path / "ui", record_lidar_continuously=continuous)
    c, _, _, _, ticket = ticket_for(tmp_path)
    messages = []
    executor = SimpleNamespace(submit=lambda _: None, poll=lambda: messages.pop(0) if messages else None,
                               process=SimpleNamespace(poll=lambda: None))
    ui = NavigationUI(app, c, executor)
    recorder = app.lidar_recorder = LidarRecorder(app.store.root / "lidar")
    try:
        trial = app.start_trial(expand_target("box"), clock(), single_answer=True)
        ui._dispatch(ticket, clock())
        assert app.trial is None
        ui.command({"action": "stop"}, clock(), clock())
        assert c.state == "STOPPING" and recorder.trial_id == trial.id
        # Simulated drive, first zero, settling and post-release dwell receipts.
        for i, t in enumerate((1000.02, 1000.15, 1001.30, 1004.30)):
            recorder.offer("odom", odom_header(i, t-900, t+1000), None, "boot", t)
            recorder.offer("scan", scan_header(i, t-900, t+1000), np.zeros((1, 3), np.float32), "boot", t)
        clock.t = 1004.35
        res = result_record(ticket, True, clock())
        res.update(started_monotonic_s=1000., status="cancelled", observed_stationary=True,
                   ownership_released=True, release_rpc_code=0, post_release_stationary=settled)
        messages.append({"type": "result", "result": res})
        ui.process(clock())
        if not settled:
            assert recorder.trial_id == trial.id and c.state == "HOLD"
            recorder.offer("odom", odom_header(4, 105., 2005.), None, "boot", 1005.)
            ui.finish_motion_recording(1005.1, "session_closed_with_unverified_cleanup", res)
        assert recorder.trial_id == ("between_trials" if continuous else None)
        assert ui.motion_recording is None
    finally:
        recorder.close()
    rows = read_jsonl(app.store.root / "lidar/index.jsonl")
    assert all(row["trial_id"] == trial.id for row in rows)
    times, odometry = [], []
    for row in rows:
        with np.load(app.store.root / "lidar" / row["file"]) as chunk:
            odometry.extend(chunk["odom_host_recv_monotonic_s"])
            times.extend(chunk["scan_host_recv_monotonic_s"])
    assert times == pytest.approx([1000.02, 1000.15, 1001.30, 1004.30])
    assert odometry == pytest.approx(times if settled else [*times, 1005.])
    expected = ["opened", "closed"] if settled else ["opened", "cleanup_unverified", "closed"]
    assert [e["phase"] for e in read_jsonl(app.store.events) if e["event"] == "motion_lidar_window"] == expected


def test_camera_bound_is_scoped_to_the_measured_boot():
    from test_hardware_guards import g

    meta = {"clock_epoch": "host_receipt_wall:bridge=fake:g0",
            "monotonic_mapping": "configured camera latency 0.048 s"}
    assert g.capture_bound(meta, -1766552958.008) == (.157, g.CAMERA["basis"])
    for older_boot in (-1766613707.168, -1766856813.961, -1766897811.159, -1766947283.136, -1766903986.310):
        assert g.capture_bound(meta, older_boot) == (None, None)
