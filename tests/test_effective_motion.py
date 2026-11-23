"""Offline profile, geometry, feedback and one-shot replan regressions. No SDK."""

from copy import deepcopy
import json
import math
from queue import Queue
from types import SimpleNamespace

import jsonschema
import numpy as np
import pytest

from go2_perception.vln.motion_profile import (
    DISC_RADIUS_M,
    INSPECTED,
    LIMITS,
    PROFILE,
    SENSOR,
    measure_outcome,
)
from go2_perception.vln.supervised import (
    CONTROL_REVISION,
    Cycle,
    Hold,
    capability_check,
    check_inspection,
    current_checks,
    digest,
    plan,
    revalidate,
    validate_record,
)
from go2_perception.vln.supervised_executor import result_record
from test_hardware_guards import Clock, floor, make_guards, wall
from test_supervised_executor import SAFETY, hardware_fakes, idle_run
from test_supervised_navigation import FIXTURE, ROOT


def configure(cap, scope=SENSOR):
    cap.update(motion_profile=PROFILE, turn_scope=scope)
    cap["limits"].update(**LIMITS, stationary_dwell_s=1.0, lateral_m=0.08, straight_yaw_rad=0.10)
    cap["mapping"].update(
        verified_actions=["forward"],
        translation_m=None,
        translation_bound_m=0.5,
        yaw_rad=0.02941,
        yaw_uncertainty_rad=0.05236,
        orientation_basis="relative_planar_heading_ccw",
    )
    return cap


def attach_camera_evidence(motion, ticket):
    source = ticket["proposal"]["identity"]
    snapshot = motion.guards.snapshot
    motion.guards.snapshot = lambda: dict(
        snapshot(),
        capture_clock_epoch=source["clock_epoch"],
        capture_bound_basis="invented-timing-for-mock",
        capture_uncertainty_bound_s=source["uncertainty_bound_s"],
    )


def proposed(rows, point=(0.5, 0.5), scope=SENSOR):
    cap, p = configure(deepcopy(FIXTURE["capability"]), scope), deepcopy(FIXTURE["proposal"])
    p["waypoints"] = (rows + [rows[-1]] * 10)[:10]
    p["point_fraction"] = list(point)
    geo = p["identity"]["geometry"]
    geo.update(
        forward_clear_m=2.0,
        directions=["forward"],
        turn_clear_rad=0.0,
        observed_turn_disc={"radius_m": DISC_RADIUS_M, "floor_returns": 9, "obstacle_returns": 0},
    )
    return cap, p


def target_for(proposal):
    return {
        "instance_id": "synthetic-box",
        "reviewed_image_sha256": proposal["identity"]["source_image_sha256"],
        "roi_xyxy": [0.1, 0.1, 0.9, 0.9],
        "confirmed_by": "synthetic-operator",
        "confirmed_monotonic_s": 100.0,
    }


def straight():
    return [[0.15 * i, 0.0, 0.0] for i in range(1, 8)]


def turning(sign):
    return [
        [0.0, 0.0, 0.0],
        [0.01, 0.0, sign * 0.26],
        [0.05, sign * 0.012, sign * 0.46],
        [0.12, sign * 0.025, sign * 0.56],
        [0.30, sign * 0.05, sign * 0.72],
    ]


def test_absolute_prefix_charges_origin_and_yaw_slack_preserves_raw():
    cap, p = proposed([[0.0, 0.0, 0.0]] + straight())
    before = deepcopy(p)
    a = plan(p, cap)
    seg = a["segment"]
    assert a["speed"] == 1.0 and a["magnitude"] == 0.6 and a["selected_row"] == 5
    assert seg["stop_trigger"] == 0.5 and seg["required_clearance"] == 0.85 and a["duration_s"] == 3.0
    assert seg["prefix_start"] == 1 and 0 < seg["end_fraction"] < 1
    assert seg["mapped_prefix"][5]["slack_m"] > 0.03 and a["residual_translation_m"] <= 0.08
    assert p == before
    independently_planned = plan(deepcopy(p), deepcopy(cap))
    assert independently_planned == a


@pytest.mark.parametrize("bad", [[0.48, 0.16, 0.28], [0.24, 0.0, 0.0], [-0.1, 0.0, 0.0], [0.5, 0.06, 0.0]])
def test_never_skips_an_intervening_bend_or_reversal(bad):
    cap, p = proposed([[0.20, 0.0, 0.0], [0.40, 0.0, 0.0], bad, [0.60, 0.0, 0.0], [0.8, 0.0, 0.0]])
    a = plan(p, cap)
    assert a["selected_row"] == 1 and 0.36 < a["magnitude"] < 0.4
    assert a["segment"]["boundary"] != "profile_limit"


def test_stop_absence_terminal_small_motion_and_uncertain_mapping():
    cap, p = proposed([[0.12, 0.0, 0.0]])
    with pytest.raises(Hold, match="below_effective_resolution"):
        plan(p, cap)
    p["native_stop"] = True
    with pytest.raises(Hold, match="native_stop"):
        plan(p, cap)
    p.update(native_stop=False, point_fraction=None)
    with pytest.raises(Hold, match="no_visible_target"):
        plan(p, cap)
    cap, p = proposed([[0.2, 0.0, 0.08], [0.45, 0.0, 0.08], [0.65, 0.0, 0.08]])
    # The unknown 0.5 m lever arm consumes lateral budget even for small native yaw.
    a = plan(p, cap)
    assert a["magnitude"] < 0.45 and a["residual_translation_m"] > 0.06
    cap["mapping"]["translation_bound_m"] = 2.0
    with pytest.raises(Hold):
        plan(p, cap)


def test_clearance_reserve_limits_destination_and_current_geometry_age():
    cap, p = proposed(straight())
    g = p["identity"]["geometry"]
    g["forward_clear_m"] = 0.73
    action = plan(p, cap)
    assert (
        action["magnitude"] == pytest.approx(0.48)
        and action["segment"]["boundary"] == "decision_geometry_limit"
    )
    current = deepcopy(p["identity"])
    current["geometry"]["monotonic_s"] -= 0.10
    with pytest.raises(Hold, match="geometry"):
        current_checks(current, action, cap, 100.0, capture=False)
    current_checks(current, action, cap, 100.0, capture=False, progress=0.11)
    g["forward_clear_m"] = 0.54
    with pytest.raises(Hold, match="insufficient_forward_clearance"):
        plan(p, cap)
    g.update(forward_clear_m=2.0, monotonic_s=95.0)
    with pytest.raises(Hold, match="not_bound"):
        plan(p, cap)


@pytest.mark.parametrize("sign,direction,lead", [(1, "left", 10), (-1, "right", 4)])
def test_only_authorized_heading_never_combined_translation(sign, direction, lead):
    point = (0.2 if sign > 0 else 0.8, 0.5)
    cap, p = proposed(turning(sign), point)
    with pytest.raises(Hold, match="geometry"):
        plan(p, cap)
    cap["turn_scope"] = INSPECTED
    a = plan(p, cap)
    assert (
        a["direction"] == direction
        and a["speed"] == 2.0
        and a["magnitude"] == pytest.approx(math.radians(30))
    )
    assert math.degrees(a["segment"]["stop_trigger"]) == pytest.approx(30 - lead)
    assert a["segment"]["basis"] == "yaw_only_stop_observe_replan" and a["residual_translation_m"] > 0.1
    assert p["identity"]["geometry"]["turn_clear_rad"] == 0
    p["point_fraction"][0] = 1.0 - point[0]
    with pytest.raises(Hold, match="inconsistent_with_off_centre"):
        plan(p, cap)


@pytest.mark.parametrize("change", [{"floor_returns": 0}, {"obstacle_returns": 1}, {"radius_m": 0.5}])
def test_inspection_cannot_replace_missing_or_obstructed_sensor_data(change):
    cap, p = proposed(turning(1), (0.2, 0.5), INSPECTED)
    p["identity"]["geometry"]["observed_turn_disc"].update(change)
    with pytest.raises(Hold, match="disc"):
        plan(p, cap)
    p["identity"]["geometry"]["available"] = False
    with pytest.raises(Hold, match="geometry"):
        plan(p, cap)


def test_inspection_bound_to_image_epoch_operator_drift_and_time():
    cap, p = proposed(turning(-1), (0.8, 0.5), INSPECTED)
    t = target_for(p)
    c = Cycle(cap, nonce="test-effective")
    a = c.preview(p, t, now=100.0, wall_s=1.0)
    with pytest.raises(Hold, match="inspection_required"):
        c.authorize(now=100.0, wall_s=1.0)
    with pytest.raises(Hold, match="operator"):
        c.inspect_turn(operator="other", now=100.0, current=p["identity"])
    c.inspect_turn(operator=t["confirmed_by"], now=100.0, current=p["identity"])
    auth = c.authorize(now=100.01, wall_s=1.0)
    validate_record("authorization", auth, 2)
    assert "session_nonce" not in c.public(100.01)["authorization"]["turn_inspection"]
    for key, val in (("session_nonce", "other"), ("preview_sha256", "a" * 64), ("direction", "left")):
        bad = deepcopy(auth)
        bad["turn_inspection"][key] = val
        with pytest.raises(Hold, match="inspection_required"):
            check_inspection(bad, a, cap, 101.0, p["identity"]["pose"])
    pose = deepcopy(p["identity"]["pose"])
    with pytest.raises(Hold, match="inspection_required"):
        check_inspection(auth, a, cap, 130.0, pose)
    pose["y_m"] += 0.11
    with pytest.raises(Hold, match="drift"):
        check_inspection(auth, a, cap, 101.0, pose, moving=True)
    pose.update(y_m=0.0, yaw_rad=0.35)
    check_inspection(auth, a, cap, 101.0, pose, moving=True)
    pose["robot_epoch"] = "another-epoch"
    with pytest.raises(Hold, match="drift"):
        check_inspection(auth, a, cap, 101.0, pose, moving=True)


def test_effective_limits_version_source_checks_and_timed_command_refusals(tmp_path):
    motion, _, _, _, _ = hardware_fakes(tmp_path)
    cap = configure(motion.cap)
    assert capability_check(cap, live=True, wall_s=1.0)["turn_radps"] == 2.0
    for key, value in (("forward_mps", 0.30), ("turn_radps", 1.0), ("duration_s", 4.0)):
        bad = deepcopy(cap)
        bad["limits"][key] = value
        with pytest.raises(Hold):
            capability_check(bad, live=True, wall_s=1.0)
    for mutate in (
        lambda x: x["control"].update(revision="742cfe4"),
        lambda x: x.pop("motion_profile"),
        lambda x: x["mapping"].update(translation_m=[0.0, 0.0]),
        lambda x: x["mapping"].update(yaw_uncertainty_rad=0.0),
    ):
        bad = deepcopy(cap)
        mutate(bad)
        with pytest.raises(Hold):
            capability_check(bad, live=True, wall_s=1.0)
    assert cap["control"]["revision"] == CONTROL_REVISION
    for vx, yaw, target in ((1.0, 0.0, 0.5), (0.0, -2.0, 0.45), (0.0, 2.0, 0.35)):
        SAFETY.validate(SAFETY.MoveRequest(vx, yaw, 3.0, 0.05, target))
        with pytest.raises(SAFETY.Refused):
            SAFETY.validate(SAFETY.MoveRequest(vx, yaw, 1.0, 0.05, 0.0))


@pytest.mark.parametrize(
    "response,expected,termination",
    [
        ("delayed", "target_within_tolerance", "feedback_trigger"),
        ("coast", "overshoot", "feedback_trigger"),
        ("stuck", "no_measurable_progress", "blocked_stall"),
        ("partial", "partial_progress", "blocked_stall"),
        ("lost", "measurement_unavailable", "progress unavailable"),
    ],
)
def test_real_audited_loop_scripted_delayed_response_or_stall(tmp_path, response, expected, termination):
    motion, ticket, cancelled, calls, clock = hardware_fakes(tmp_path)
    configure(motion.cap)
    attach_camera_evidence(motion, ticket)
    ticket["proposal"]["waypoints"] = straight() + [[1.05, 0.0, 0.0]] * 3
    a = plan(ticket["proposal"], motion.cap)
    ticket["action"] = ticket["authorization"]["envelope"] = a
    ticket["authorization"]["capability_sha256"] = digest(motion.cap)
    changed_camera = dict(motion.guards.snapshot(), capture_clock_epoch="restarted-camera")
    with pytest.raises(Hold, match="camera_epoch_changed"):
        revalidate(
            ticket["authorization"],
            ticket["proposal"],
            motion.cap,
            now=clock(),
            wall_s=1.0,
            nonce=ticket["nonce"],
            live=True,
            current=changed_camera,
        )
    idle_run(motion, clock, 1.2)
    start = clock()
    increments = {"delayed": 0.60, "coast": 0.80, "stuck": 0.0, "partial": 0.15, "lost": 0.0}

    def measured():
        if "stop" in calls:
            return None if response == "lost" else increments[response]
        elapsed = clock() - start
        if response == "lost" and calls.count("velocity") >= 3:
            return None
        if elapsed < 0.45 or response == "stuck":
            return 0.0
        return min(0.15 if response == "partial" else 0.51, (elapsed - 0.45) * 0.73)

    motion.guards.progress = lambda _: measured
    updates = []
    motion.feedback = updates.append
    result = motion(ticket, a, result_record(ticket, False, clock()), cancelled)
    m = result["motion_measurement"]
    assert (m["outcome"], m["termination"], m["stop_verified"]) == (expected, termination, True)
    assert m["interrupted_or_blocked"] == (termination != "feedback_trigger")
    assert result["status"] == ("completed" if expected == "target_within_tolerance" else "ended")
    assert calls.count("claim") == 1 and calls[-4:] == ["stop", "stop", "stop", "release"]
    assert all(
        e["vx"] == 1.0 and e["vyaw"] == 0.0 for e in result["events"] if e.get("event") == "send_velocity"
    )
    assert updates[-1]["phase"] == "settled"
    if termination == "feedback_trigger":
        assert 0.5 <= m["at_first_zero"] <= 0.51 and m["settled"] > m["at_first_zero"]
        assert calls.count("velocity") > 10
    validate_record("execution_result", result, 2)
    if result["status"] == "ended":
        with pytest.raises(Hold, match="completion_inconsistent"):
            validate_record("execution_result", dict(result, status="completed"), 2)


def test_provider_partial_disc_is_a_veto_and_delta_imu_wraps(tmp_path):
    clock = Clock()
    g = make_guards(tmp_path, clock)
    g.capability = {"motion_profile": PROFILE}
    g.odom.update(10.0, 0.0, 0.0, 0.0)
    g.scans.add(floor())
    g.compute()
    geo = g.snapshot()["geometry"]
    assert geo["turn_clear_rad"] == 0 and geo["observed_turn_disc"]["floor_returns"] > 2
    assert geo["observed_turn_disc"]["obstacle_returns"] == 0
    g.scans.add(np.concatenate([floor(), wall(0.59)]))  # beyond the body's reach plus drift
    g.compute()
    assert g.snapshot()["geometry"]["observed_turn_disc"]["obstacle_returns"] == 0
    g.scans.add(np.concatenate([floor(), wall(0.50)]))
    g.compute()
    assert g.snapshot()["geometry"]["observed_turn_disc"]["obstacle_returns"] > 0
    imu = [math.pi - 0.1]
    g.monitor = SimpleNamespace(state_age=lambda: 0.01, snapshot=lambda: {"rpy": [0.0, 0.0, imu[0]]})
    measure = g.progress({"type": "turn", "direction": "left"})
    imu[0] = -math.pi + 0.2
    assert measure() == pytest.approx(0.3) and measure.evidence["source"] == "relative_imu_yaw"
    clock.t += 0.3
    assert measure() is None


def test_stop_release_new_view_then_advisory_only(tmp_path):
    from go2_perception.vln.supervised_app import NavigationUI

    clock = Clock()
    cap, p = proposed(turning(1), (0.2, 0.5), INSPECTED)
    c = Cycle(cap, nonce="post-turn")
    p["identity"]["geometry"].update(monotonic_s=clock())
    t = target_for(p)
    a = c.preview(p, t, now=clock(), wall_s=1.0)
    c.inspect_turn(operator=t["confirmed_by"], now=clock(), current=p["identity"])
    c.authorize(now=clock(), wall_s=1.0)
    clock.t += 0.2
    fresh = deepcopy(p)
    fresh["identity"].update(
        observed_monotonic_s=clock(),
        received_monotonic_s=clock(),
        prediction_id="fresh",
        observation_id="fresh-img",
    )
    for k in ("pose", "geometry"):
        fresh["identity"][k]["monotonic_s"] = clock()
    comparison = {
        "valid": True,
        "method": "local_feature_roi_v1",
        "instance_id": t["instance_id"],
        "reviewed_image_sha256": t["reviewed_image_sha256"],
        "fresh_image_sha256": fresh["identity"]["source_image_sha256"],
    }
    ticket = c.refreshed(fresh, comparison, now=clock(), wall_s=1.0, current=fresh["identity"])
    bad = deepcopy(fresh)
    bad["point_fraction"] = [0.8, 0.5]
    with pytest.raises(Hold):
        revalidate(
            ticket["authorization"],
            bad,
            cap,
            now=clock(),
            wall_s=1.0,
            nonce=c.nonce,
            live=False,
            current=bad["identity"],
        )
    result = result_record(ticket, True, clock())
    result.update(
        status="completed",
        started_monotonic_s=clock(),
        stop_send_statuses=[0, 0, 0],
        release_rpc_code=0,
        observed_stationary=True,
        ownership_released=True,
        post_release_stationary=True,
    )
    result["motion_measurement"] = measure_outcome(
        a, 0.35, 0.50, 0.52, termination="feedback_trigger", stop_verified=True
    )
    c.completed(result)
    with pytest.raises(Hold):
        c.record_replan(fresh)
    clock.t += 0.2
    old = fresh["identity"]
    meta = {
        "frame_id": "stopped-img",
        "stamp_robot": {"t": clock()},
        "received_monotonic_s": clock(),
        "clock_epoch": old["clock_epoch"],
    }
    tap = SimpleNamespace(
        latest_id=meta["frame_id"],
        queue=Queue(),
        frame=lambda _: (np.zeros((64, 64, 3), dtype=np.uint8), meta),
    )
    trial = SimpleNamespace(id="new-advisory")
    starts, ends = [], []

    def start_trial(spec, now, single_answer):
        starts.append((spec["instruction"], single_answer))
        return trial

    app = SimpleNamespace(
        tap=tap,
        predictions={},
        store=SimpleNamespace(root=tmp_path),
        trial=trial,
        session=SimpleNamespace(replay_anchor=(0.0, 0.0, 1.0), session_id=old["session_id"]),
        start_trial=start_trial,
        end_trial=lambda why, *args, **kwargs: ends.append(why),
    )
    # This executor supplies observations only. A second submit would fail the test.
    executor = SimpleNamespace(
        guard_at=lambda stamp, gap: {
            "pose": dict(old["pose"], monotonic_s=stamp),
            "geometry": dict(old["geometry"], monotonic_s=stamp),
            "capture_uncertainty_bound_s": 0.01,
        },
        poll=lambda: None,
    )
    ui = NavigationUI(app, c, executor)
    ui.instruction_id = old["instruction_id"]
    ui.process(clock())
    assert c.state == "REPLANNING" and c.locked and starts == [(old["instruction"], True)]
    with pytest.raises(Hold, match="reused"):
        c.record_replan(fresh)
    clock.t += 0.2
    post = c.post["identity"]
    app.predictions["new-advice"] = {
        "trial_id": trial.id,
        "prediction_id": "new-advice",
        "instruction": old["instruction"],
        "source_frame": {
            "frame_id": "new-inference-img",
            "clock_epoch": old["clock_epoch"],
            "observation_s": clock(),
            "received_monotonic_s": clock(),
        },
        "source_image": {"path": "navigation/post_motion.png", "file_sha256": post["source_image_sha256"]},
        "request_directory": "new-request",
        "rejection_reason_at_receipt": None,
        "mock": True,
        "waypoint_convention": "absolute_ego_SE2_x_forward_y_left_yaw_CCW_metres_radians",
        "native_stop": True,
        "waypoints": [[0.0, 0.0, 0.0]] * 10,
        "target_fraction": [0.5, 0.5],
        "native_output": "synthetic STOP",
    }
    (tmp_path / "new-request").mkdir()
    (tmp_path / "new-request/snapshot.json").write_text(
        json.dumps({"history": [{"source": {"observation_s": clock()}}]})
    )
    ui.process(clock())
    assert ends == ["post_turn_read_only_replan"]
    assert json.loads((tmp_path / "navigation/post_turn_advisory.json").read_text()) == c.replan
    assert json.loads((tmp_path / "navigation/cycle_result.json").read_text())["state"] == "COMPLETE"
    assert c.replan["refusal"] == "native_stop_no_motion" and c.replan["execution_authorized"] is False
    assert c.state == "COMPLETE"
    with pytest.raises(Hold):
        c.preview(p, t, now=clock(), wall_s=1.0)
    archive = dict(
        schema="go2-supervised-navigation-v2",
        synthetic=True,
        capability=cap,
        proposal=fresh,
        authorization=ticket["authorization"],
        executed_action=ticket["action"],
        execution_result=result,
        post_observation=c.post,
    )
    schema = json.loads(
        (ROOT / "src/go2_perception/vln/schemas/supervised-navigation-v2.schema.json").read_text()
    )
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(archive, schema)


@pytest.mark.parametrize("sign", [-1, 1])
def test_inspected_turn_uses_pinned_loop_heading_feedback_and_zero_forward_speed(tmp_path, sign):
    motion, ticket, cancel, calls, clock = hardware_fakes(tmp_path)
    configure(motion.cap, INSPECTED)
    attach_camera_evidence(motion, ticket)
    _, example = proposed(turning(sign), (0.2 if sign > 0 else 0.8, 0.5), INSPECTED)
    proposal = ticket["proposal"]
    proposal.update(waypoints=example["waypoints"], point_fraction=example["point_fraction"])
    auth = ticket["authorization"]
    snapshot = motion.guards.snapshot

    def observed():
        newer = snapshot()
        newer["geometry"].update(
            directions=["forward"],
            turn_clear_rad=0.0,
            observed_turn_disc=deepcopy(example["identity"]["geometry"]["observed_turn_disc"]),
        )
        return newer

    motion.guards.snapshot = observed
    proposal["identity"]["geometry"].update(observed()["geometry"])
    a = plan(proposal, motion.cap)
    ticket["action"] = auth["envelope"] = a
    auth["capability_sha256"] = digest(motion.cap)
    auth["turn_inspection"] = {
        "scope": INSPECTED,
        "session_nonce": ticket["nonce"],
        "preview_sha256": digest(auth["reviewed_identity"]),
        "direction": a["direction"],
        "radius_m": DISC_RADIUS_M,
        "operator": auth["target"]["confirmed_by"],
        "pose": dict(auth["reviewed_identity"]["pose"], monotonic_s=auth["created_monotonic_s"] - 0.06),
        "inspected_s": auth["created_monotonic_s"] - 0.05,
    }
    idle_run(motion, clock, 1.2)
    motion.guards.progress = (
        lambda _: lambda: a["magnitude"] if "stop" in calls else 0.048 * calls.count("velocity")
    )
    result = motion(ticket, a, result_record(ticket, False, clock()), cancel)
    assert result["status"] == "completed" and result["motion_measurement"]["stop_verified"]
    sends = [e for e in result["events"] if e.get("event") == "send_velocity"]
    assert all(e["vx"] == 0.0 and e["vyaw"] == sign * 2.0 for e in sends)
    assert len(sends) == math.ceil(a["segment"]["stop_trigger"] / 0.048)
    assert result["motion_measurement"]["settled"] == a["magnitude"]
    assert calls[-4:] == ["stop", "stop", "stop", "release"]


def test_sector_veto_even_with_inspected_scope():
    cap, p = proposed(turning(1), (0.2, 0.5), INSPECTED)
    p["identity"]["geometry"].update(directions=["forward", "left"], turn_clear_rad=0.1)
    with pytest.raises(Hold, match="measured_turn_sector_restricts"):
        plan(p, cap)


def test_freshness_gate_waits_for_a_wholly_post_authorization_capture(tmp_path):
    from go2_perception.vln.supervised_app import NavigationUI

    cap, p = proposed(straight())
    c = Cycle(cap)
    c.preview(p, target_for(p), now=100.0, wall_s=1.0)
    c.authorize(now=100.0, wall_s=1.0)
    c.live = True
    frame = {"exposure_monotonic_s": 100.10, "clock_epoch": "camera-epoch"}
    tap = SimpleNamespace(latest_id="frame", frame=lambda _: (None, frame))
    snapshot = {
        "capture_uncertainty_bound_s": 0.148,
        "capture_bound_basis": "measured",
        "capture_clock_epoch": "camera-epoch",
    }
    app = SimpleNamespace(tap=tap, predictions={}, trial=None)
    executor = SimpleNamespace(latest_guards={"snapshot": snapshot}, poll=lambda: None)
    ui = NavigationUI(app, c, executor)
    ui.refresh_started = 100.0
    ui.pending_refresh = {"instruction": "box"}
    starts = []
    ui._start = lambda spec, now, intent: starts.append((spec, now, intent))
    ui.process(100.11)
    frame.update(exposure_monotonic_s=100.19, clock_epoch="different")
    ui.process(100.20)
    assert not starts
    frame["clock_epoch"] = "camera-epoch"
    ui.process(100.21)
    assert len(starts) == 1 and starts[0][-1] == "refresh" and ui.pending_refresh is None
