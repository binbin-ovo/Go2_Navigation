"""Audited-loop wrapper tests with injected routes/clock; NEVER initialize DDS."""
from copy import deepcopy
import hashlib
import json
import os
import sys
import threading
import time

import pytest

from go2_perception.vln.supervised import CONTROL_REVISION, MODEL_REVISION, Hold, digest
from go2_perception.vln.supervised_executor import (Executor, ExecutorClient, HardwareMotion,
                                                  canonical_control_check, load_module, result_record)
from test_supervised_navigation import ROOT, ticket_for

PURE_LOOP = ROOT / "tests/fixtures/supervised/audited_safety.py"
assert hashlib.sha256(PURE_LOOP.read_bytes()).hexdigest() == "71f70f5706c3f8e93ede361d1a90ea74ad27d2d709a14b926d306ab002a36353"
SAFETY = load_module(PURE_LOOP, "supervised_test_audited_safety")


def hardware_fakes(tmp_path, *, failure=None):
    c, p, guards, clock, ticket = ticket_for(tmp_path)
    cap = deepcopy(c.cap)
    cap["kind"] = "hardware"
    cap["control"] = {"revision": CONTROL_REVISION, "root": "/synthetic/control", "python": "/synthetic/python",
                      "provider_path": "/synthetic/provider", "provider_sha256": "a"*64,
                      "shared_lock": "/synthetic/control/data/outputs/go2_control/.command.lock"}
    ticket["authorization"]["capability_sha256"] = digest(cap)
    ticket["proposal"]["synthetic"] = False
    ticket["proposal"]["model_identity"].update(revision=MODEL_REVISION, processor_revision=MODEL_REVISION)
    calls, progress = [], [0.]
    cancelled = threading.Event()

    class Monitor:
        @property
        def remote_t(self):
            return clock() - (1. if failure == "stale_remote" else .01)

        def state_age(self):
            return 1. if failure == "stale_state" else .01

        def operator_input(self):
            return False

        def is_still(self):
            return not (failure == "failed_stop" and "velocity" in calls)

        def snapshot(self):
            return {"error_code": 1002 if failure == "robot_error" else 100}

    class Route:
        def avoidance_enabled(self):
            return failure != "avoidance_disabled"

        def claim_api_control(self):
            calls.append("claim")
            if failure == "claim_exception":
                raise OSError("injected claim exception")
            return 0

        def send_velocity(self, vx, yaw):
            calls.append("velocity")
            progress[0] += .01
            if failure == "cancel":
                cancelled.set()
            if failure == "epoch_change":
                guards.epoch = "different-boot"
            return 2 if failure == "send_failed" else 0

        def send_stop(self):
            calls.append("stop")
            if failure == "stop_exception":
                raise OSError("injected stop send exception")
            return 0

        def release_api_control(self):
            calls.append("release")
            return 2 if failure == "release_failed" else 0

    guards.progress = lambda action: lambda: progress[0]
    if failure == "missing_geometry":
        old = guards.snapshot

        def snapshot():
            value = old()
            value["geometry"]["available"] = False
            return value

        guards.snapshot = snapshot
    motion = HardwareMotion(SAFETY, Route(), Monitor(), guards, cap,
                            heartbeat=lambda: clock() - (1. if failure == "heartbeat" else 0.),
                            monotonic=clock, sleep=clock.sleep, wall=lambda: 1)
    return motion, ticket, cancelled, calls, clock


def test_audited_wrapper_cadence_stop_dwell_and_release_separate(tmp_path):
    motion, ticket, cancel, calls, clock = hardware_fakes(tmp_path)
    result = motion(ticket, ticket["action"], result_record(ticket, False, clock()), cancel)
    assert calls[:2] == ["claim", "velocity"]
    assert calls[-4:] == ["stop", "stop", "stop", "release"]
    assert result["observed_stationary"] and result["post_release_stationary"] and result["ownership_released"]
    assert result["command_robot_acknowledged"] is None and result["stop_command_acknowledged"] is None
    assert "unitree_sdk2py" not in sys.modules and "cyclonedds" not in sys.modules


@pytest.mark.parametrize("failure", ["stale_remote", "stale_state", "missing_geometry", "heartbeat", "avoidance_disabled",
                                     "robot_error"])
def test_guard_failure_never_claims_or_moves(tmp_path, failure):
    motion, ticket, cancel, calls, clock = hardware_fakes(tmp_path, failure=failure)
    result = motion(ticket, ticket["action"], result_record(ticket, False, clock()), cancel)
    assert "claim" not in calls and "velocity" not in calls
    assert result["observed_stationary"] is None


@pytest.mark.parametrize("failure", ["cancel", "claim_exception", "send_failed", "stop_exception", "failed_stop", "release_failed", "epoch_change"])
def test_executor_failure_attempts_cleanup_and_preserves_unknowns(tmp_path, failure):
    motion, ticket, cancel, calls, clock = hardware_fakes(tmp_path, failure=failure)
    result = motion(ticket, ticket["action"], result_record(ticket, False, clock()), cancel)
    assert "stop" in calls and "release" in calls
    first_stop = calls.index("stop")
    assert "velocity" not in calls[first_stop:]
    assert result["stop_command_acknowledged"] is None
    if failure == "failed_stop":
        assert result["observed_stationary"] is False and result["post_release_stationary"] is False
    if failure == "release_failed":
        assert result["ownership_released"] is False
    if failure == "send_failed":
        assert result["command_send_statuses"] == [2]
    if failure == "epoch_change":
        assert calls.count("velocity") == 1 and "epoch" in result["reason"]


def test_independent_executor_cancel_while_parent_waits_for_model(tmp_path):
    c, p, guards, clock, ticket = ticket_for(tmp_path)
    # Translate synthetic monotonic stamps to the actual subprocess clock; no old
    # recording is restamped. This is a newly generated test message.
    now = time.monotonic()
    a = ticket["authorization"]
    a["created_monotonic_s"], a["expires_monotonic_s"] = now-.1, now+9.
    for identity in (ticket["proposal"]["identity"], a["reviewed_identity"]):
        identity["observed_monotonic_s"] = now-.05
        identity["received_monotonic_s"] = now-.04
        identity["pose"]["monotonic_s"] = now-.05
        identity["geometry"]["monotonic_s"] = now-.05
    cap_path = tmp_path / "capability.json"
    cap_path.write_text(json.dumps(c.cap))
    client = ExecutorClient([sys.executable, str(ROOT / "scripts/vln_supervised_executor.py"),
                             "--capability", str(cap_path), "--nonce", c.nonce, "--output", str(tmp_path/"child")],
                            nonce=c.nonce, env={**os.environ, "PYTHONPATH": str(ROOT/"src"),
                                              "GO2_HARDWARE": "0", "GO2_SUPERVISED_LIVE": "0"})
    try:
        end = time.monotonic()+3
        while not client.ready and time.monotonic() < end:
            time.sleep(.01)
        assert client.ready
        client.submit(ticket)
        # A concurrent model wait never owns the cancel pipe or command loop.
        cancelled_at = []

        def cancel_soon():
            time.sleep(.15)
            cancelled_at.append(time.monotonic())
            client.cancel()

        thread = threading.Thread(target=cancel_soon)
        thread.start()
        time.sleep(.5)  # stand-in for parent waiting on an unrelated model
        message = client.poll()
        assert message["type"] == "result"
        assert message["result"]["status"] == "cancelled"
        assert message["result"]["ended_monotonic_s"]-cancelled_at[0] < .15
        assert message["result"]["measured_progress"] < ticket["action"]["magnitude"]
        thread.join()
    finally:
        assert client.close()


def test_wrong_session_never_dispatches(tmp_path):
    c, p, guards, clock, ticket = ticket_for(tmp_path)
    engine = Executor(c.cap, "another-nonce", guards=guards, monotonic=clock, wall=lambda: 1, sleep=clock.sleep)
    result = engine.execute(ticket)
    assert result["status"] == "held" and not result["command_send_statuses"]


def test_other_worktree_cannot_choose_independent_lock(tmp_path):
    root = tmp_path / "canonical"
    other = tmp_path / "another-worktree"
    control = {"root": str(root), "shared_lock": str(other / "data/outputs/go2_control/.command.lock")}
    with pytest.raises(Hold, match="wrong_command_lock"):
        canonical_control_check(control, root)
    with pytest.raises(Hold, match="control_root_mismatch"):
        canonical_control_check(control, other)
    assert os.environ.get("GO2_SUPERVISED_LIVE", "0") == "0"


def test_hardware_environment_is_not_compared_by_shared_base_python(tmp_path):
    root = tmp_path / "canonical"
    control = {"root": str(root), "shared_lock": str(root / "data/outputs/go2_control/.command.lock"),
               "python": str(tmp_path / "wrong-venv/bin/python")}
    with pytest.raises(Hold, match="wrong_hardware_environment"):
        canonical_control_check(control, root)
