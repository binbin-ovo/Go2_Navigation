"""Lightweight offline contracts/state-machine tests. No SDK, DDS, GPU or model."""
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading

import cv2
import numpy as np
import pytest

from go2_perception.vln.supervised import (Cycle, Hold, capability_check, mapped_pose, plan,
                                         validate_record)
from go2_perception.vln.supervised_executor import Executor, FakeGuards, live_preflight
from go2_perception.vln.supervised_target import correspondence

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((ROOT / "examples/supervised_navigation/synthetic_cycle.json").read_text())


class Clock:
    def __init__(self, t=100.):
        self.t = t

    def __call__(self):
        return self.t

    def sleep(self, dt):
        self.t += max(.000001, dt)


def setup_cycle(tmp_path):
    clock = Clock()
    cap, proposal = deepcopy(FIXTURE["capability"]), deepcopy(FIXTURE["proposal"])
    image = np.random.default_rng(123).integers(0, 256, (256, 448, 3), dtype=np.uint8)
    first, second = tmp_path / "review.png", tmp_path / "refresh.png"
    cv2.imwrite(str(first), image)
    cv2.imwrite(str(second), image)
    import hashlib
    sha = hashlib.sha256(first.read_bytes()).hexdigest()
    proposal["identity"]["source_image_sha256"] = sha
    proposal["identity"]["received_monotonic_s"] = clock()
    target = {"instance_id": "synthetic-box-instance-1", "reviewed_image_sha256": sha,
              "roi_xyxy": [.2, .2, .8, .8], "confirmed_by": "mock-operator", "confirmed_monotonic_s": clock()}
    cycle = Cycle(cap, nonce="test-session-nonce")
    cycle.preview(proposal, target, now=clock(), wall_s=1)
    guards = FakeGuards(clock)
    return cycle, proposal, guards, clock, first, second


def refresh(tmp_path, delay=30.):
    cycle, proposal, guards, clock, first, second = setup_cycle(tmp_path)
    clock.t += delay  # Human may inspect the preview for arbitrarily longer than 3 s.
    cycle.authorize(now=clock(), wall_s=1)
    clock.t += .1
    fresh = deepcopy(proposal)
    fresh["identity"].update(prediction_id="new-prediction", observation_id="new-observation",
                             trial_id="new-independent-trial", observed_monotonic_s=clock(),
                             received_monotonic_s=clock(), **{k: v for k, v in guards.snapshot().items()
                                                            if k in ("pose", "geometry")})
    match = correspondence(first, second, cycle.target, fresh["point_fraction"])
    assert match["valid"]
    return cycle, fresh, guards, clock, match


def ticket_for(tmp_path):
    c, p, g, clock, match = refresh(tmp_path)
    ticket = c.refreshed(p, match, now=clock(), wall_s=1, current=g.snapshot())
    return c, p, g, clock, ticket


def test_complete_delayed_review_refresh_one_action_stop_new_observation(tmp_path):
    cycle, proposal, guards, clock, ticket = ticket_for(tmp_path)
    engine = Executor(cycle.cap, cycle.nonce, guards=guards, monotonic=clock, wall=lambda: 1, sleep=clock.sleep)
    result = engine.execute(ticket)
    assert result["status"] == "completed" and result["synthetic"]
    assert result["measured_progress"] == pytest.approx(.1)
    assert result["stop_command_acknowledged"] is None
    assert result["observed_stationary"] and result["ownership_released"]
    cycle.completed(result)
    assert cycle.state == "AWAITING_OBSERVATION" and cycle.authorization is None
    with pytest.raises(Hold, match="not_new"):
        cycle.observe_after(proposal["identity"])
    clock.t += .1
    post = deepcopy(proposal["identity"])
    post.update(observed_monotonic_s=clock(), observation_id="post-motion", **{k: v for k, v in guards.snapshot().items() if k in ("pose", "geometry")})
    cycle.observe_after(post)
    assert cycle.state == "COMPLETE" and cycle.reviewed is None
    assert engine.execute(ticket)["status"] == "held"
    with pytest.raises(Hold):
        cycle.authorize(now=clock(), wall_s=1)


@pytest.mark.parametrize("field,value,reason", [
    ("instruction", "Approach another box.", "changed_instruction"),
    ("instruction_id", "other", "changed_instruction_id"),
    ("clock_epoch", "reboot", "changed_clock_epoch"),
    ("session_id", "another-app", "changed_session_id"),
    ("observation_id", "synthetic-observation_id", "reused_observation"),
    ("uncertainty_bound_s", None, "unknown_capture"),
    ("observed_monotonic_s", 90., "refresh_not_acquired"),
])
def test_changed_or_unavailable_refresh_holds(tmp_path, field, value, reason):
    c, p, g, clock, match = refresh(tmp_path)
    p["identity"][field] = value
    with pytest.raises(Hold, match=reason):
        c.refreshed(p, match, now=clock(), wall_s=1, current=g.snapshot())
    assert c.state == "HOLD"


@pytest.mark.parametrize("mutation,reason", [
    (lambda p: p["identity"]["pose"].update(x_m=.5), "material_pose_change"),
    (lambda p: p["identity"]["pose"].update(robot_epoch="other"), "robot_epoch"),
    (lambda p: p.update(native_stop=True), "native_stop"),
    (lambda p: p.update(waypoints=[[0., 0., -.2]]*10), "changed_action"),
    (lambda p: p.update(waypoints=[[.14, 0., 0.]]*10), "outside_authorized"),
])
def test_pose_or_action_changes_hold(tmp_path, mutation, reason):
    c, p, g, clock, match = refresh(tmp_path)
    mutation(p)
    with pytest.raises(Hold, match=reason):
        c.refreshed(p, match, now=clock(), wall_s=1, current=g.snapshot())


@pytest.mark.parametrize("part", ["instance_id", "fresh_image_sha256", "reviewed_image_sha256", "valid"])
def test_same_word_or_point_is_not_target_correspondence(tmp_path, part):
    c, p, g, clock, match = refresh(tmp_path)
    match[part] = False if part == "valid" else "another"
    with pytest.raises(Hold, match="physical_target"):
        c.refreshed(p, match, now=clock(), wall_s=1, current=g.snapshot())


@pytest.mark.parametrize("change,reason", [
    (lambda current: current["pose"].update(monotonic_s=0), "stale_or_future_robot_state"),
    (lambda current: current["geometry"].update(available=False), "geometry_unavailable"),
    (lambda current: current["geometry"].update(directions=[]), "geometry_unavailable"),
    (lambda current: current["geometry"].update(monotonic_s=0), "stale_or_future_geometry"),
    (lambda current: current["geometry"].update(forward_clear_m=.01), "geometry_unavailable"),
])
def test_independent_current_guards_required(tmp_path, change, reason):
    c, p, g, clock, match = refresh(tmp_path)
    current = g.snapshot()
    change(current)
    with pytest.raises(Hold, match=reason):
        c.refreshed(p, match, now=clock(), wall_s=1, current=current)


def test_expiry_and_no_duplicate_authorization(tmp_path):
    c, p, g, clock, match = refresh(tmp_path)
    with pytest.raises(Hold, match="duplicate"):
        c.authorize(now=clock(), wall_s=1)
    clock.t += 12
    with pytest.raises(Hold, match="expired"):
        c.refreshed(p, match, now=clock(), wall_s=1, current=g.snapshot())


def test_old_fresh_proposal_not_restamped(tmp_path):
    c, p, g, clock, match = refresh(tmp_path)
    clock.t += 3.01
    with pytest.raises(Hold, match="stale_or_future_observation"):
        c.refreshed(p, match, now=clock(), wall_s=1, current=g.snapshot())


def test_cancel_before_and_during_fake_execution(tmp_path):
    c, p, g, clock, ticket = ticket_for(tmp_path)
    cancelled = threading.Event()
    cancelled.set()
    engine = Executor(c.cap, c.nonce, guards=g, monotonic=clock, wall=lambda: 1, sleep=clock.sleep, cancelled=cancelled)
    held = engine.execute(ticket)
    assert held["started_monotonic_s"] is None and not held["command_send_statuses"]
    cancelled.clear()
    engine = Executor(c.cap, c.nonce, guards=g, monotonic=clock, wall=lambda: 1, cancelled=cancelled)

    def sleep(dt):
        clock.sleep(dt)
        cancelled.set()

    engine.sleep = sleep
    stopped = engine.execute(ticket)
    assert stopped["status"] == "cancelled" and stopped["observed_stationary"]
    assert stopped["measured_progress"] < ticket["action"]["magnitude"]


@pytest.mark.parametrize("field", ["observed_stationary", "ownership_released", "post_release_stationary"])
def test_stop_or_release_failure_locks_cycle(tmp_path, field):
    c, p, g, clock, ticket = ticket_for(tmp_path)
    engine = Executor(c.cap, c.nonce, guards=g, monotonic=clock, wall=lambda: 1, sleep=clock.sleep)
    result = engine.execute(ticket)
    result[field] = False
    c.completed(result)
    assert c.state == "HOLD" and c.locked


def test_mapping_conjugation_heading_wrap_and_no_summing():
    cap, p = deepcopy(FIXTURE["capability"]), deepcopy(FIXTURE["proposal"])
    cap["mapping"]["yaw_rad"] = math.pi/2
    assert mapped_pose([1., 0., 0.], cap["mapping"]) == pytest.approx([0., 1., 0.])
    cap["mapping"].update(yaw_rad=0., translation_m=[1., 0.])
    assert mapped_pose([0., 0., math.pi], cap["mapping"]) == pytest.approx([2., 0., math.pi])
    cap["mapping"]["translation_m"] = [0., 0.]
    p["waypoints"] = [[0., 0., 0.], [.1, 0., 0.]] + [[1., 0., 0.]]*8
    action = plan(p, cap)
    assert action["selected_row"] == 1 and action["magnitude"] == .1
    p["waypoints"] = [[0., 0., 2*math.pi-.2]]*10
    assert plan(p, cap)["direction"] == "right"


@pytest.mark.parametrize("row", [[-.1, 0., 0.], [0., .1, 0.], [.3, 0., .3]])
def test_unsupported_motion_holds(row):
    p = deepcopy(FIXTURE["proposal"])
    p["waypoints"] = [row]*10
    with pytest.raises(Hold, match="unsupported"):
        plan(p, FIXTURE["capability"])


def test_missing_registration_and_synthetic_live_refused():
    cap = deepcopy(FIXTURE["capability"])
    cap["mapping"] = None
    with pytest.raises(Hold, match="missing_frame"):
        capability_check(cap, live=False, wall_s=1)
    with pytest.raises(Hold, match="mixing"):
        live_preflight(FIXTURE["capability"], execute_live=True, exclusive_source=True, control_root=ROOT)
    assert "unitree_sdk2py" not in sys.modules and "cyclonedds" not in sys.modules


def test_runtime_schema_rejects_nonfinite_and_extra_fields():
    p = deepcopy(FIXTURE["proposal"])
    p["waypoints"][0][0] = float("nan")
    with pytest.raises(Hold):
        validate_record("proposal", p)
    p = deepcopy(FIXTURE["proposal"])
    p["trust_me"] = True
    with pytest.raises(Hold):
        validate_record("proposal", p)


def test_visual_change_or_textureless_target_holds(tmp_path):
    c, p, g, clock, first, second = setup_cycle(tmp_path)
    cv2.imwrite(str(second), np.zeros((256, 448, 3), np.uint8))
    assert not correspondence(first, second, c.target, [.5, .5])["valid"]
    cv2.imwrite(str(second), cv2.imread(str(first)))
    assert not correspondence(first, second, c.target, [.95, .95])["valid"]


def test_live_launcher_blocks_before_model_or_robot_import(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "GO2_HARDWARE": "0", "GO2_SUPERVISED_LIVE": "0"}
    result = subprocess.run([sys.executable, str(ROOT / "scripts/run_vln_supervised.py"), "--navigation-mode", "live"],
                            capture_output=True, text=True, env=env)
    assert result.returncode == 2 and "Live needs" in result.stderr


def test_unregistered_translation_is_bounded_not_zero():
    cap, p = deepcopy(FIXTURE["capability"]), deepcopy(FIXTURE["proposal"])
    cap["mapping"].update(translation_m=None, yaw_rad=0., verified_actions=["forward"])
    with pytest.raises(Hold, match="without_bound"):
        capability_check(cap, live=False, wall_s=1)
    cap["mapping"]["translation_bound_m"] = .5
    capability_check(cap, live=False, wall_s=1)
    p["waypoints"] = [[.3, 0., 0.]]*10       # pure translation: the origin cancels exactly
    assert plan(p, cap)["residual_translation_m"] == pytest.approx(0.)
    yaw = cap["limits"]["straight_yaw_rad"]
    p["waypoints"] = [[.3, 0., yaw]]*10      # heading change: worst case 2 sin(yaw/2) * bound
    worst = 2*math.sin(yaw/2)*.5
    if worst > cap["limits"]["lateral_m"]:
        with pytest.raises(Hold, match="unsupported"):
            plan(p, cap)
    else:
        action = plan(p, cap)
        assert action["residual_translation_m"] == pytest.approx(worst)
        assert action["magnitude"] == pytest.approx(min(.3 - worst, cap["limits"]["forward_m"]))
    cap["mapping"]["verified_actions"] = ["forward", "turn"]
    with pytest.raises(Hold, match="turn_requires_registered"):
        capability_check(cap, live=False, wall_s=1)


def test_mapping_yaw_uncertainty_widens_lateral_residual():
    cap, p = deepcopy(FIXTURE["capability"]), deepcopy(FIXTURE["proposal"])
    cap["mapping"].update(translation_m=[0., 0.], yaw_rad=0., verified_actions=["forward"])
    lateral = cap["limits"]["lateral_m"]
    p["waypoints"] = [[.2, lateral*.9, 0.]]*10
    assert plan(p, cap)["type"] == "forward"
    cap["mapping"]["yaw_uncertainty_rad"] = math.radians(3)
    with pytest.raises(Hold, match="unsupported"):
        plan(p, cap)
