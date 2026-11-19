"""Continuous navigation uses the v2 proposal and simulated expiring velocity packets only."""

from copy import deepcopy
import json
import math
import os
import subprocess
from pathlib import Path
import sys

import numpy as np
import pytest

from go2_perception.vln.continuous import ContinuousNavigator, PERIOD_S, ZERO, progress, simulated_velocity
from go2_perception.vln.continuous_simulation import (
    CAMERA,
    START,
    effective_mock_capability,
    fixture_result,
    integrate,
    sensor,
    simulate,
)
from go2_perception.vln.motion_profile import Hold, SENSOR

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def fixture():
    return json.loads((ROOT / "examples/supervised_navigation/synthetic_cycle.json").read_bytes())


class Episode:
    def __init__(self, fixture, **bounds):
        self.template = deepcopy(fixture["proposal"])
        self.scope = dict(self.template["identity"], clock_epoch="sim-camera-g0")
        self.pose = dict(self.scope["pose"])
        self.cap = effective_mock_capability(deepcopy(fixture["capability"]))
        self.cap["turn_scope"] = SENSOR
        self.nav = ContinuousNavigator(
            self.cap,
            self.scope,
            target="chair",
            operator="fixture",
            origin=self.pose,
            started_s=START,
            **bounds,
        )
        self.sequence = 0

    def guard(self, t, **position):
        g = sensor(dict(self.pose, **position), t)
        g["prior_stillness"] = dict(
            still_since_monotonic_s=START - 1.2, last_idle_sample_s=START, monotonic_s=START
        )
        return g

    def result(self, t, *, stop=False, visible=True, latency=0.0, **position):
        self.sequence += 1
        image = np.zeros((101, CAMERA["width"]), np.uint8)
        if visible:
            h = 30 if not stop else 60
            image[50 - h // 2 : 50 + h // 2, 73:88] = 255
        return fixture_result(self.template, self.guard(t, **position), image, self.sequence, latency)

    def step(self, dt=0.0, **kwargs):
        t = START + dt
        return self.nav.tick(t, self.guard(t), **kwargs)


def test_compatible_plan_is_replaced_without_zero_and_uses_capture_pose(fixture):
    e = Episode(fixture)
    a = e.step(delivery=e.result(START))
    delayed = e.result(START + 0.01, latency=0.04, x_m=0.01)
    b = e.nav.tick(START + 0.05, e.guard(START + 0.05, x_m=0.05), delivery=delayed)
    assert a["velocity"] == b["velocity"] == [1.0, 0.0]
    assert b["progress"] == pytest.approx(0.04)
    assert b["source_observation_id"] == delayed["proposal"]["identity"]["observation_id"]
    assert b["capture_age_upper_s"] == pytest.approx(0.04)
    assert b["rollovers"] == 1 and b["plans"] == 2
    assert b["expires_s"] <= b["t"] + 0.10000001


@pytest.mark.parametrize(
    "edit,expected",
    [
        ("wrong_scope", "different_episode_instruction_or_epoch"),
        ("duplicate_id", "duplicate_or_superseded"),
        ("duplicate_frame", "duplicate_or_superseded"),
        ("older_stop", "duplicate_or_superseded"),
    ],
)
def test_superseded_or_foreign_result_cannot_steer_or_stop(fixture, edit, expected):
    e = Episode(fixture)
    initial = e.result(START)
    e.step(delivery=initial)
    q = e.result(START + 0.02, stop=True)
    ident = q["proposal"]["identity"]
    old = initial["proposal"]["identity"]
    if edit == "wrong_scope":
        ident["trial_id"] = "different"
    elif edit == "duplicate_id":
        ident["prediction_id"] = old["prediction_id"]
    elif edit == "duplicate_frame":
        ident["observation_id"] = old["observation_id"]
    else:
        ident["observed_monotonic_s"] = START
    b = e.step(0.05, delivery=q)
    assert b["velocity"] == [1.0, 0.0] and b["plans"] == 1
    assert b["events"][0]["ignored"] == expected


@pytest.mark.parametrize(
    "fault,reason",
    [
        ("lidar", "lidar_unavailable"),
        ("blocked", "required_geometry_unavailable_or_insufficient"),
        ("radio", "radio_receipt_time_missing"),
        ("stale_radio", "stale_or_future_radio"),
        ("stick", "operator_or_state_interlock"),
        ("unhealthy", "operator_or_state_interlock"),
        ("clock", "camera_epoch_changed"),
        ("odom_epoch", "sensor_delivery_frame_or_epoch_changed"),
        ("future", "stale_or_future_guard_delivery"),
        ("stale_pose", "stale_or_future_odometry"),
        ("capture_bound", "capture_bound_changed"),
        ("off_path", "forward_path_drift"),
        ("reverse", "motion_opposes_plan"),
        ("bad_payload", "malformed_input"),
    ],
)
def test_fresh_guard_vetoes_in_flight_plan_and_preserves_exact_evidence(fixture, fault, reason):
    e = Episode(fixture)
    a = e.step(delivery=e.result(START))
    g = e.guard(START + 0.05)
    snap = g["snapshot"]
    state = g["interlock"]
    if fault == "lidar":
        snap["geometry"]["available"] = False
    elif fault == "blocked":
        snap["geometry"]["forward_clear_m"] = 0.1
    elif fault == "radio":
        state["radio_s"] = None
    elif fault == "stale_radio":
        state["radio_s"] -= 0.4
    elif fault == "stick":
        state["neutral"] = False
    elif fault == "unhealthy":
        state["healthy"] = False
    elif fault == "clock":
        snap["capture_clock_epoch"] = "restarted"
    elif fault == "odom_epoch":
        snap["pose"]["robot_epoch"] = "restarted"
    elif fault == "future":
        g["available_s"] += 0.5
    elif fault == "stale_pose":
        snap["pose"]["monotonic_s"] -= 0.4
    elif fault == "capture_bound":
        snap["capture_uncertainty_bound_s"] = 0.2
    elif fault == "off_path":
        snap["pose"]["y_m"] += 0.12
    elif fault == "reverse":
        snap["pose"]["x_m"] -= 0.08
    elif fault == "bad_payload":
        del g["interlock"]
    b = e.nav.tick(START + 0.05, g)
    assert b["velocity"] == list(ZERO) and b["reason"] == reason
    assert b["veto"] == dict(
        reason=reason, details=b["veto"]["details"], guard=g, previous_prediction_id=a["prediction_id"]
    )
    assert e.step(0.1)["velocity"] == list(ZERO)


def test_incompatible_mode_brakes_and_requires_capture_after_continuous_settling(fixture):
    e = Episode(fixture)
    assert e.step(delivery=e.result(START))["mode"] == "APPROACH"
    b = e.step(0.05, delivery=e.result(START + 0.05, visible=False))
    assert b["reason"] == "mode_or_direction_change_requires_settled_view"
    for i in range(2, 24):
        q = e.step(i * PERIOD_S)
        assert q["velocity"] == list(ZERO)
    pre_dwell = e.result(START + 0.2, latency=1.0)
    assert e.step(1.2, delivery=pre_dwell)["reason"] == "awaiting_new_settled_observation"
    b = e.step(1.25, delivery=e.result(START + 1.25, visible=False))
    assert b["mode"] == "SEARCH" and b["velocity"] == [0.0, 2.0]
    assert b["rollovers"] == 0


def test_native_stop_even_during_geometry_loss_and_cancel_latch(fixture):
    e = Episode(fixture)
    e.step(delivery=e.result(START))
    b = e.nav.tick(START + 0.05, None, delivery=e.result(START + 0.05, stop=True))
    assert b["mode"] == "TERMINAL" and b["reason"] == "native_stop_arrival_unverified"
    assert e.step(0.1, delivery=e.result(START + 0.1))["velocity"] == list(ZERO)
    e = Episode(fixture)
    assert e.step(cancel=True, delivery=e.result(START))["reason"] == "cancelled"
    assert e.step(0.05, delivery=e.result(START + 0.05))["reason"] == "cancelled"


@pytest.mark.parametrize(
    "kind,expected",
    [
        ("camera_age", "stale_or_future_observation"),
        ("future_output", "noncausal_proposal_timestamps"),
        ("clock_order", "noncausal_proposal_timestamps"),
        ("no_bound", "unknown_capture_age_bound"),
        ("old_pose", "source_pose_or_geometry_unbound"),
        ("fake_real", "synthetic_source_mismatch"),
        ("unreviewed", "identify_or_reject_current_mark"),
        ("misbound_image", "grounding_for_wrong_target_or_image"),
    ],
)
def test_proposal_admission(fixture, kind, expected):
    e = Episode(fixture)
    q = e.result(START)
    ident = q["proposal"]["identity"]
    if kind == "camera_age":
        ident["observed_monotonic_s"] -= 2.95
        ident["uncertainty_bound_s"] = 0.1
    elif kind == "future_output":
        q["available_s"] += 0.1
    elif kind == "clock_order":
        ident["received_monotonic_s"] -= 0.05
    elif kind == "no_bound":
        ident["uncertainty_bound_s"] = None
    elif kind == "old_pose":
        ident["pose"]["monotonic_s"] -= 0.4
    elif kind == "fake_real":
        q["proposal"]["synthetic"] = False
    elif kind == "unreviewed":
        q["grounding"]["status"] = "needs_identification"
    elif kind == "misbound_image":
        q["grounding"]["image_sha256"] = "b" * 64
    b = e.step(delivery=q)
    assert b["reason"] == expected and b["plans"] == 0


def test_watchdog_independent_of_controller_and_zero_on_late_resume(fixture):
    e = Episode(fixture)
    packet = e.step(delivery=e.result(START))
    pose = integrate(e.pose, packet, START, START + 3)
    assert pose["x_m"] == pytest.approx(0.10) and simulated_velocity(packet, START + 3) == ZERO
    b = e.nav.tick(START + 0.25, e.guard(START + 0.25, x_m=0.10), delivery=e.result(START + 0.25, x_m=0.10))
    assert b["reason"] == "control_tick_gap" and b["velocity"] == list(ZERO)


def test_packet_cannot_outlive_due_sensor_or_remaining_prefix(fixture):
    e = Episode(fixture)
    t = START + 0.1
    g = e.guard(t)
    g["interlock"]["radio_s"] = t - e.cap["limits"]["state_age_s"] + 0.02
    packet = e.nav.tick(t, g, delivery=e.result(t))
    assert packet["expires_s"] == pytest.approx(t + 0.02)
    b = e.nav.tick(t + 0.05, e.guard(t + 0.05), delivery=e.result(t + 0.05))
    assert b["reason"] == "command_packet_expired"
    e = Episode(fixture)
    a = e.step(delivery=e.result(START))
    trigger = a["action"]["segment"]["stop_trigger"]
    b = e.nav.tick(START + 0.05, e.guard(START + 0.05, x_m=trigger - 0.01))
    assert b["expires_s"] == pytest.approx(START + 0.06)


def test_no_initial_motion_without_witness_or_consecutive_fresh_stationary_samples(fixture):
    e = Episode(fixture)
    for j in range(22):
        t = START + j * 0.05
        g = e.guard(t)
        del g["prior_stillness"]
        b = e.nav.tick(t, g, delivery=e.result(t))
        if j < 20:
            assert b["velocity"] == list(ZERO)
    assert b["velocity"] == [1.0, 0.0]


def test_source_frame_signed_turn_and_wrap(fixture):
    e = Episode(fixture)
    a = e.step(delivery=e.result(START, visible=False))["action"]
    origin = dict(e.pose, x_m=4.0, y_m=-3.0, yaw_rad=math.pi - 0.1)
    assert progress(a, origin, dict(origin, yaw_rad=-math.pi + 0.05), e.cap["limits"]) == pytest.approx(0.15)
    right = dict(a, direction="right")
    assert progress(right, origin, dict(origin, yaw_rad=math.pi - 0.25), e.cap["limits"]) == pytest.approx(
        0.15
    )
    with pytest.raises(Hold, match="turn_translation_drift"):
        progress(a, origin, dict(origin, x_m=4.12), e.cap["limits"])


def test_deadline_budget_lease_and_physical_profile_bounds(fixture):
    e = Episode(fixture, max_plans=1)
    e.step(delivery=e.result(START))
    b = e.step(0.05, delivery=e.result(START + 0.05))
    assert b["reason"] == "plan_budget"
    assert b["plans"] == 1 and b["velocity"] == list(ZERO)
    e = Episode(fixture, travel_m=0.12)
    e.step(delivery=e.result(START))
    assert e.nav.tick(START + 0.05, e.guard(START + 0.05, x_m=0.05))["reason"] == "travel_or_rotation_budget"
    e = Episode(fixture, duration_s=0.075)
    a = e.step(delivery=e.result(START))
    assert a["expires_s"] == pytest.approx(START + 0.075)
    assert e.step(0.1)["reason"] == "episode_deadline"
    e = Episode(fixture)
    e.step(delivery=e.result(START))
    for k in range(1, 14):
        b = e.step(k * 0.05)
    assert b["velocity"] == list(ZERO) and b["reason"] == "plan_lease_expired"
    with pytest.raises(ValueError):
        Episode(fixture, max_plans=6)


@pytest.mark.parametrize(
    "scenario",
    (
        "visible_after_search",
        "two_second_planner",
        "lidar_dropout",
        "blocked_corridor",
        "operator_cancel",
        "stopped_control_clock",
    ),
)
def test_closed_loop_inference_delay_and_faults(fixture, scenario):
    rows = []
    r = simulate(fixture, scenario, rows.append)
    assert 0 < r["plans"] <= 5 and r["physical_commands_sent"] == 0
    assert r["moving_intervals_while_inference_pending"] > 0
    packets = [row["packet"] for row in rows if row["kind"] == "tick"]
    assert {"SEARCH", "APPROACH"} <= {p["mode"] for p in packets}
    assert packets[-1]["velocity"] == list(ZERO) and r["end_reason"]
    if scenario == "visible_after_search":
        assert r["uninterrupted_replacements"] >= 1
        assert any(
            a["plans"] < b["plans"] and a["velocity"] == b["velocity"] == [1.0, 0.0]
            for a, b in zip(packets, packets[1:])
        )
    if scenario == "two_second_planner":
        assert r["uninterrupted_replacements"] == 0
    if scenario == "stopped_control_clock":
        assert r["watchdog_zero_intervals"] > 0
    if scenario == "operator_cancel":
        assert r["end_reason"] == "cancelled"
    if scenario in ("lidar_dropout", "blocked_corridor"):
        reason = (
            "lidar_unavailable"
            if scenario == "lidar_dropout"
            else "required_geometry_unavailable_or_insufficient"
        )
        first = next(i for i, p in enumerate(packets) if p["reason"] == reason)
        assert all(p["velocity"] == list(ZERO) for p in packets[first:])
    assert r["proxy_path_length_m"] >= 0 and r["proxy_absolute_yaw_rad"] > 0


def test_cli_refuses_hardware_opt_ins_and_reusing_output(tmp_path, monkeypatch):
    from run_vln_continuous import main

    dest = tmp_path / "one"
    monkeypatch.setenv("GO2_HARDWARE", "1")
    with pytest.raises(SystemExit):
        main(["simulate", "--output", str(dest)])
    assert not dest.exists()
    monkeypatch.setenv("GO2_HARDWARE", "0")
    monkeypatch.setenv("GO2_SUPERVISED_LIVE", "0")
    main(["simulate", "--scenario", "operator_cancel", "--output", str(dest)])
    with pytest.raises(FileExistsError):
        main(["simulate", "--output", str(dest)])
    assert json.loads((dest / "summary.json").read_bytes())["physical_commands_sent"] == 0


def test_offline_entry_has_no_sdk_socket_or_control_dependency(tmp_path):
    probe = """
import runpy,sys
def audit(event,args):
    if event in ('socket.connect','socket.bind','subprocess.Popen'):
        raise AssertionError(event)
    if event=='import' and args[0].split('.')[0] in ('cyclonedds','unitree_sdk2py','go2_control'):
        raise AssertionError(args[0])
sys.addaudithook(audit)
script,out=sys.argv[1:]
sys.argv=[script,'simulate','--scenario','operator_cancel','--output',out]
runpy.run_path(script,run_name='__main__')
"""
    # NumPy may query CPU information on first import; preload only common numeric libraries.
    probe = "import numpy,cv2\n" + probe
    subprocess.run(
        [sys.executable, "-c", probe, str(ROOT / "scripts/run_vln_continuous.py"), str(tmp_path / "out")],
        env={**os.environ, "GO2_HARDWARE": "0", "GO2_SUPERVISED_LIVE": "0"},
        check=True,
        timeout=15,
    )


def local_recording(name):
    root = Path(os.environ.get("GO2_WORKSPACE", "/nonexistent")) / "runs/effective_trials" / name / "session"
    if not (root / "summary.json").is_file():
        pytest.skip("closed private Go2 single-action recording unavailable")
    return root


@pytest.mark.parametrize("name", ["trial-C3jMfl", "trial-abZbVx"])
def test_closed_archive_uses_actual_receipt_times_and_exposes_absent_radio_clock(name):
    from run_vln_continuous import replay

    rows = []
    result = replay(local_recording(name), rows.append)
    assert result["source_hashes_verified_after_replay"] and result["grounding_status"] == "supported"
    assert result["historical_input_frames"] == 5 and result["inliers"] >= 12
    assert result["passes"]["recorded_only"]["nonzero_ticks"] == 0
    assert result["radio_receipt_timestamps_present"] == 0
    assert result["passes"]["recorded_only"]["zero_reasons"]["radio_receipt_time_missing"] > 0
    for row in rows[1:]:
        g = row["guard"]
        if g:
            assert g["available_s"] <= row["t"]
        if g and g["radio_time_assumed"]:
            assert row["counterfactual_pass"] == "assume_radio_sampled_at_guard_check"
        if row["offered_prediction_id"]:
            assert row["t"] >= rows[0]["delivery"]["available_s"]
        assert row["packet"]["rollovers"] == 0


@pytest.mark.parametrize(
    "fault,reason", [("future", "causal frame clock"), ("image", "archived model history image")]
)
def test_archived_future_frame_and_tampered_history_refused_in_memory(monkeypatch, fault, reason):
    from run_vln_continuous import archive
    import replay_effective_motion as reader

    original = reader.load

    def edited(path):
        doc = original(path)
        if path.name == "job.json":
            if fault == "future":
                doc["frames"][0]["available_s"] = doc["decision_time_s"] + 5
            else:
                doc["frames"][0]["sha256"] = "0" * 64
        return doc

    monkeypatch.setattr(reader, "load", edited)
    with pytest.raises(ValueError, match=reason):
        archive(local_recording("trial-C3jMfl"))


def test_terminal_time_and_wrong_instruction_fail_closed(fixture):
    e = Episode(fixture)
    e.step(delivery=e.result(START))
    assert e.nav.tick(START - 1, None)["reason"] == "invalid_control_clock"
    assert e.step(1.0)["velocity"] == list(ZERO)
    e = Episode(fixture)
    q = e.result(START)
    q["proposal"]["identity"]["instruction"] = "A different instruction"
    assert e.step(delivery=q)["plans"] == 0


def test_unaudited_inspected_turn_scope_fails_closed(fixture):
    e = Episode(fixture)
    e.nav.cap["turn_scope"] = "operator_inspected_static_disc"
    g = e.guard(START)
    g["snapshot"]["geometry"]["observed_turn_disc"] = {
        "radius_m": 0.65,
        "floor_returns": 20,
        "obstacle_returns": 0,
    }
    q = e.result(START, visible=False)
    q["proposal"]["identity"]["geometry"] = g["snapshot"]["geometry"]
    b = e.nav.tick(START, g, delivery=q)
    assert b["velocity"] == list(ZERO) and b["reason"] == "continuous_turn_requires_observed_sector"


def test_private_trace_cannot_be_placed_in_publishable_tree_or_source(tmp_path):
    from run_vln_continuous import main

    source = tmp_path / "source"
    source.mkdir()
    for out in (ROOT / "docs" / "continuous-private-test-output", source / "bad"):
        with pytest.raises(SystemExit):
            main(["replay", "--recording", str(source), "--output", str(out)])
        assert not out.exists()


def test_caller_mutation_cannot_change_accepted_plan_or_saved_veto(fixture):
    e = Episode(fixture)
    offer = e.result(START)
    a = e.step(delivery=offer)
    offer["proposal"]["identity"]["pose"]["yaw_rad"] = 1.2
    a["action"]["speed"] = 100.0
    a["events"][0]["selection"]["action"]["magnitude"] = 0.0
    assert e.step(0.05)["velocity"] == [1.0, 0.0]
    g = e.guard(START + 0.10)
    g["snapshot"]["geometry"]["available"] = False
    b = e.nav.tick(START + 0.10, g)
    g["snapshot"]["geometry"]["available"] = True
    assert b["veto"]["guard"]["snapshot"]["geometry"]["available"] is False


def test_operator_intervention_has_priority_during_lidar_loss(fixture):
    e = Episode(fixture)
    e.step(delivery=e.result(START))
    g = e.guard(START + 0.05)
    g["snapshot"]["geometry"]["available"] = False
    g["interlock"]["neutral"] = False
    assert e.nav.tick(START + 0.05, g)["reason"] == "operator_or_state_interlock"
    assert e.step(0.1, delivery=e.result(START + 0.1))["mode"] == "TERMINAL"
