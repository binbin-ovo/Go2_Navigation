"""Isolated executor boundary. Hardware imports occur only after explicit live gates.

Nothing in the browser or model process imports go2_control/sdk_routes. The fake
executor exercises the same request/authorization checks without any robot SDK.
"""
from copy import deepcopy
from collections import deque
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
import threading
import time

from .motion_profile import measure_outcome, wrap
from .supervised import (CONTROL_REVISION, Hold, age_check, capability_check, check_inspection, contract_version, current_checks,
                         revalidate, validate_record)

# Sport-state error_code values seen while the robot stands and walks normally (Go2 Pro, 2026-11-15/18:
# every state sample of the canonical moves and statuses reads 100). The controller emergency stop sets
# 1002 (2026-11-18). Anything else HOLDs.
NOMINAL_ROBOT_ERROR_CODES = frozenset({100})
IDLE_SAMPLE_GAP_S = 0.1  # longest gap between idle stillness samples that still counts as continuous


def finite_json(value):
    """Keep unavailable raw telemetry explicit; never emit nonstandard JSON NaN/Inf."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [finite_json(v) for v in value]
    return value


def result_record(ticket, synthetic, now):
    return {"request_id": ticket["request_id"], "authorization_id": ticket["authorization"]["authorization_id"],
            "synthetic": synthetic, "status": "held", "reason": "not_started",
            "started_monotonic_s": None, "ended_monotonic_s": now, "command_send_statuses": [],
            "command_robot_acknowledged": None, "measured_progress": None, "stop_send_statuses": [],
            "stop_command_acknowledged": None, "observed_stationary": None,
            "ownership_released": None, "release_rpc_code": None, "post_release_stationary": None,
            "events": []}


def check_correspondence(ticket):
    # Stdlib checks in hardware env; core computed registration from archived pixels.
    c, a, p = ticket["correspondence"], ticket["authorization"], ticket["proposal"]
    if not (c.get("method") == "local_feature_roi_v1" and c.get("valid") is True and
            c.get("instance_id") == a["target"]["instance_id"] and
            c.get("reviewed_image_sha256") == a["target"]["reviewed_image_sha256"] and
            c.get("fresh_image_sha256") == p["identity"]["source_image_sha256"] and
            c.get("inliers", 0) >= 16 and c.get("inlier_fraction", 0) >= .8 and
            c.get("target_inliers", 0) >= 8 and 0 <= c.get("median_residual_px", 99) <= 1.5 and
            0 <= c.get("max_corner_drift_fraction", 99) <= .02 and c.get("fresh_point_inside") is True):
        raise Hold("unverified_target_correspondence")


class FakeGuards:
    """Synthetic only; deterministic pose and explicit invented geometry."""
    def __init__(self, monotonic=time.monotonic):
        self.clock = monotonic
        self.x = self.y = self.yaw = 0.
        self.epoch = "synthetic-boot"

    def snapshot(self, observed_s=None):
        now = self.clock() if observed_s is None else observed_s
        return {"pose": {"frame": "synthetic_odom", "robot_epoch": self.epoch,
                         "monotonic_s": now, "x_m": self.x, "y_m": self.y, "yaw_rad": self.yaw},
                "geometry": {"available": True, "monotonic_s": now, "frame": "synthetic_odom",
                             "robot_epoch": self.epoch, "source": "SYNTHETIC fake guard",
                             "directions": ["forward", "left", "right"], "forward_clear_m": 1., "turn_clear_rad": 1.},
                "capture_uncertainty_bound_s": .001}


class Executor:
    """Once-per-session dispatcher; duplicate IDs stay spent even on HOLD/errors."""
    def __init__(self, cap, nonce, *, guards, live=False, motion=None, monotonic=time.monotonic,
                 wall=time.time, sleep=time.sleep, cancelled=None):
        self.cap, self.nonce, self.guards, self.live = cap, nonce, guards, live
        self.motion, self.clock, self.wall, self.sleep = motion, monotonic, wall, sleep
        self.cancelled = cancelled or threading.Event()
        self.used = set()
        self.faulted = False

    def execute(self, ticket):
        result = result_record(ticket, not self.live, self.clock())
        auth = ticket["authorization"]["authorization_id"]
        try:
            if self.faulted or auth in self.used or ticket["nonce"] != self.nonce:
                raise Hold("executor_faulted_duplicate_or_wrong_session")
            self.used.add(auth)
            if self.cancelled.is_set():
                raise Hold("cancelled_before_dispatch")
            current = self.guards.snapshot()
            action = revalidate(ticket["authorization"], ticket["proposal"], self.cap,
                                now=self.clock(), wall_s=self.wall(), nonce=self.nonce,
                                live=self.live, current=current)
            if action != ticket["action"]:
                raise Hold("request_action_not_reproducible")
            check_correspondence(ticket)
            if self.live:
                if self.motion is None:
                    raise Hold("hardware_motion_unavailable")
                result = self.motion(ticket, action, result, self.cancelled)
            else:
                result = self._fake_motion(action, result)
        except Exception as error:
            result["reason"] = f"{type(error).__name__}: {error}"
            result["status"] = "held" if result["started_monotonic_s"] is None else "failed"
        result["ended_monotonic_s"] = self.clock()
        # Every request ends the process's ability to accept another physical move.
        # Even a rejected request requires a fresh supervised session; no retry loop.
        self.faulted = True
        validate_record("execution_result", result, contract_version(self.cap))
        return result

    def _fake_motion(self, action, result, *, area_guard=None):
        result["started_monotonic_s"] = self.clock()
        start, progress = self.clock(), 0.
        goal = action.get("segment", {}).get("stop_trigger", action["magnitude"])
        while self.clock()-start < action["duration_s"] and progress < goal:
            if self.cancelled.is_set():
                break
            current = self.guards.snapshot()
            now = self.clock()
            current_checks(current, action, self.cap, now, capture=False, progress=progress)
            if area_guard is not None:
                area_guard(current, progress)
            dt = min(self.cap["limits"]["resend_period_s"],
                     (goal-progress)/action["speed"],
                     action["duration_s"]-(now-start))
            self.sleep(dt)
            step = action["speed"]*dt
            if action["type"] == "forward":
                self.guards.x += step*math.cos(self.guards.yaw)
                self.guards.y += step*math.sin(self.guards.yaw)
            else:
                self.guards.yaw += step*(1 if action["direction"] == "left" else -1)
            progress += step
            result["command_send_statuses"].append(0)
        result.update(status="cancelled" if self.cancelled.is_set() else "completed",
                      reason="synthetic_cancel" if self.cancelled.is_set() else "synthetic_bounded_action",
                      measured_progress=progress, stop_send_statuses=[0, 0, 0], observed_stationary=True,
                      ownership_released=True, release_rpc_code=0, post_release_stationary=True)
        if "segment" in action:
            result["motion_measurement"] = measure_outcome(action, progress, progress, progress,
                termination="synthetic_cancel" if self.cancelled.is_set() else "synthetic_feedback_trigger", stop_verified=True)
            if not self.cancelled.is_set() and result["motion_measurement"]["outcome"] != "target_within_tolerance":
                result["status"] = "ended"
        result["events"].append({"synthetic": True, "note": "No hardware connection or commands."})
        return result


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def live_preflight(cap, *, execute_live, exclusive_source, control_root):
    """No SDK import or network action. Check ALL static gates first."""
    capability_check(cap, live=True, wall_s=time.time())
    if not execute_live or not exclusive_source or os.environ.get("GO2_SUPERVISED_LIVE") != "1":
        raise Hold("explicit_live_opt_in_and_exclusive_source_required")
    return canonical_control_check(cap["control"], control_root)


def canonical_control_check(control, control_root):
    """Read-only path/source checks, independently testable without enabling live opt-ins."""
    root = Path(control_root).resolve()
    if root != Path(control["root"]).resolve() or not Path(control["root"]).is_absolute():
        raise Hold("canonical_control_root_mismatch")
    expected_lock = root / "data/outputs/go2_control/.command.lock"
    if Path(control["shared_lock"]).resolve() != expected_lock.resolve():
        raise Hold("independent_or_wrong_command_lock")
    # Venv interpreters may symlink to the SAME base Python; compare the venv,
    # not just the dereferenced executable binary.
    expected_python = Path(control["python"])
    if (not expected_python.is_absolute() or Path(sys.prefix).resolve() != expected_python.parent.parent.resolve() or
            Path(sys.executable).absolute() != expected_python.absolute()):
        raise Hold("wrong_hardware_environment")
    revision = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    if revision != CONTROL_REVISION:
        raise Hold("control_revision_changed")
    for relative in ("scripts/go2_hl_control.py", "src/go2_control/safety.py", "src/go2_control/sdk_routes.py"):
        tracked = subprocess.run(["git", "-C", str(root), "show", CONTROL_REVISION + ":" + relative],
                                 capture_output=True, check=True).stdout
        if (root / relative).read_bytes() != tracked:
            raise Hold("audited_control_source_modified")
    provider = Path(control["provider_path"])
    if not provider.is_absolute() or hashlib.sha256(provider.read_bytes()).hexdigest() != control["provider_sha256"]:
        raise Hold("hardware_guard_provider_hash_mismatch")
    return root, provider


class HardwareMotion:
    """Wrap the unchanged audited loop; fresh guards/stop dwell/cancel are independent."""
    def __init__(self, safety, route, monitor, guards, cap, heartbeat, monotonic=time.monotonic,
                 sleep=time.sleep, wall=time.time, feedback=None):
        self.safety, self.route, self.monitor, self.guards = safety, route, monitor, guards
        self.cap, self.heartbeat, self.clock = cap, heartbeat, monotonic
        self.sleep, self.wall = sleep, wall
        self.feedback = feedback
        self.idle_still_since = self.idle_sampled_at = None

    def sample_idle(self):
        """Called by the executor's idle loop (every ~0.05 s) while no action is dispatched.

        Keeps the start of the current continuous run of fresh, still state samples, so the pre-claim
        dwell can count stillness that preceded dispatch instead of restarting after the model answer.
        Any moving or stale sample resets it.
        """
        now, age = self.clock(), self.monitor.state_age()
        fresh = age is not None and 0 <= age <= self.cap["limits"]["state_age_s"]
        if fresh and self.monitor.is_still() is True:
            self.idle_still_since = self.idle_still_since or now
        else:
            self.idle_still_since = None
        self.idle_sampled_at = now

    def __call__(self, ticket, action, result, cancelled, *, preflight=None, area_guard=None):
        outer = self
        lim = self.cap["limits"]
        reason = []
        unconfirmed_only = None  # (status, reason) when only the controller's settle window lapsed
        lease_deadline = [None]
        segment = action.get("segment")
        at_zero = peak = None
        last_value = last_stamp = None
        zero_started = False
        next_coast_sample = 0.
        termination = "interrupted"
        progress_fn = None

        def read_progress(phase):
            nonlocal peak, last_value, last_stamp
            value = progress_fn() if progress_fn else None
            stamp = self.clock()
            last_value = value if value is not None and math.isfinite(value) else None
            last_stamp = stamp
            if last_value is not None:
                peak = last_value if peak is None else max(peak, last_value)
            if segment:
                result["events"].append({"event": "motion_feedback", "phase": phase, "monotonic_s": stamp,
                                         "value": last_value, "unit": action["unit"],
                                         "basis": finite_json(getattr(progress_fn, "evidence", None))})
                if self.feedback is not None:
                    self.feedback({"request_id": ticket["request_id"], "phase": phase, "value": last_value,
                                   "unit": action["unit"], "monotonic_s": stamp})
            return last_value


        # Continue an unbroken idle stillness run only if it was sampled moments ago.
        continuing = (self.idle_still_since if self.idle_sampled_at is not None
                      and self.clock() - self.idle_sampled_at <= IDLE_SAMPLE_GAP_S else None)
        result["events"].append({"event": "idle_stillness_at_dispatch", "monotonic_s": self.clock(),
                                 "still_since_monotonic_s": continuing, "last_idle_sample_s": self.idle_sampled_at})

        def check_before_claim_or_first_send(current):
            if preflight is not None:
                preflight(current)
            else:
                revalidate(ticket["authorization"], ticket["proposal"], outer.cap, now=outer.clock(),
                           wall_s=outer.wall(), nonce=ticket["nonce"], live=True, current=current)

        class Monitor:
            stationary_since = continuing

            def state_age(self):
                age = outer.monitor.state_age()
                return age if age is not None and 0 <= age <= lim["state_age_s"] else None

            def operator_input(self):
                snap = state = checked_at = completed = None
                stage = "intervention"
                try:
                    if cancelled.is_set() or outer.clock()-outer.heartbeat() > .5:
                        raise Hold("cancel_or_parent_heartbeat_lost")
                    # Radio freshness is checked in addition to the audited state freshness.
                    age_check(outer.monitor.remote_t, outer.clock(), lim["state_age_s"], "remote_state")
                    stage = "snapshot"
                    snap = outer.guards.snapshot()
                    checked_at = outer.clock()
                    stage = "path_drift"
                    original = ticket["proposal"]["identity"]["pose"]
                    # Geometry is in the current body frame. Subtract only fresh
                    # odometry displacement in the original commanded direction.
                    completed = 0.
                    if segment and action["type"] == "forward":
                        pose = snap["pose"]
                        dx, dy = pose["x_m"]-original["x_m"], pose["y_m"]-original["y_m"]
                        c, s = math.cos(original["yaw_rad"]), math.sin(original["yaw_rad"])
                        completed = dx*c + dy*s
                        if (abs(dy*c-dx*s) > lim["lateral_m"] or
                                abs(wrap(pose["yaw_rad"]-original["yaw_rad"])) > lim["straight_yaw_rad"]):
                            raise Hold("forward_path_drift_exceeds_reviewed_corridor")
                    stage = "current_geometry_and_pose"
                    checked_at = outer.clock()
                    current_checks(snap, action, outer.cap, checked_at, capture=False, progress=completed)
                    stage = "inspection"
                    if area_guard is not None:
                        stage = "episode_area"
                        area_guard(snap, completed)
                    else:
                        check_inspection(ticket["authorization"], action, outer.cap, outer.clock(), snap["pose"],
                                         moving=bool(result["command_send_statuses"]))
                    stage = "robot_epoch"
                    if (snap["pose"]["frame"], snap["pose"]["robot_epoch"]) != (original["frame"], original["robot_epoch"]):
                        raise Hold("robot_epoch_or_pose_frame_changed_during_execution")
                    stage = "robot_state"
                    state = outer.monitor.snapshot()
                    result["events"].append(finite_json({"event": "current_guards", "monotonic_s": outer.clock(),
                                                        "guards": snap, "robot_state": state}))
                    if state.get("error_code") not in NOMINAL_ROBOT_ERROR_CODES:
                        raise Hold("robot_error_state")
                except Exception as error:
                    # Capture the object actually checked. The next cache read can
                    # already describe a different scan. No new sensor call or file I/O.
                    result["events"].append(finite_json({
                        "event": "guard_rejected", "request_id": ticket["request_id"],
                        "monotonic_s": outer.clock(), "checked_monotonic_s": checked_at,
                        "stage": stage, "nonzero_sends": len(result["command_send_statuses"]),
                        "direction": action["direction"], "clearance_progress_m": completed,
                        "reason": str(error), "exception_type": type(error).__name__,
                        "details": getattr(error, "details", {}), "guards": snap, "robot_state": state,
                    }))
                    reason.append(str(error))
                    return True
                return outer.monitor.operator_input()

            def is_still(self):
                nonlocal next_coast_sample
                if segment and zero_started and outer.clock() >= next_coast_sample:
                    try:
                        read_progress("settling")
                    except Exception as error:
                        result["events"].append({"settling_feedback_error": str(error)})
                    next_coast_sample = outer.clock()+.05
                age = self.state_age()
                still = outer.monitor.is_still()
                result["events"].append(finite_json({"event": "stationary_sample", "monotonic_s": outer.clock(),
                                                    "state_age_s": age, "still": still,
                                                    "robot_state": outer.monitor.snapshot()}))
                if age is None or age > lim["state_age_s"] or still is not True:
                    self.stationary_since = None
                    return False
                self.stationary_since = self.stationary_since or outer.clock()
                return outer.clock() - self.stationary_since >= lim["stationary_dwell_s"]

        class Route:
            def avoidance_enabled(self):
                return outer.route.avoidance_enabled()

            def claim_api_control(self):
                # Last no-motion checks after the RPC avoidance query and immediately before claim.
                current = outer.guards.snapshot()
                check_before_claim_or_first_send(current)
                if mon.operator_input():
                    raise Hold("intervention_before_claim")
                if mon.is_still() is not True:
                    raise Hold("not_stationary_before_claim")
                result["started_monotonic_s"] = outer.clock()
                code = outer.route.claim_api_control()
                lease_deadline[0] = min(outer.clock()+action["duration_s"],
                                        ticket["authorization"]["expires_monotonic_s"])
                return code

            def release_api_control(self):
                code = outer.route.release_api_control()
                result["release_rpc_code"], result["ownership_released"] = code, code == 0
                return code

            def send_velocity(self, vx, vyaw):
                if mon.operator_input():
                    raise Hold("current_guard_or_cancel_before_send")
                if lease_deadline[0] is None or outer.clock() >= lease_deadline[0]:
                    raise Hold("independent_command_deadline_expired")
                if not result["command_send_statuses"]:
                    current = outer.guards.snapshot()
                    check_before_claim_or_first_send(current)
                mon.stationary_since = None  # pre-motion dwell cannot count as post-motion dwell
                code = outer.route.send_velocity(vx, vyaw)
                result["command_send_statuses"].append(code)
                if code != 0:
                    raise Hold("velocity_local_send_failed")
                return code

            def send_stop(self):
                nonlocal at_zero, zero_started
                code = outer.route.send_stop()  # zero precedes telemetry and any UI update
                result["stop_send_statuses"].append(code)
                if not zero_started:
                    zero_started = True
                    if segment and last_stamp is not None and outer.clock()-last_stamp <= lim["state_age_s"]:
                        at_zero = last_value
                        result["events"].append({"event": "first_zero_feedback", "sample_s": last_stamp,
                                                 "zero_send_s": outer.clock(), "value": at_zero})
                return code

        mon, route = Monitor(), Route()
        vx = action["speed"] if action["type"] == "forward" else 0.
        vyaw = (action["speed"] if action["direction"] == "left" else -action["speed"]) if action["type"] == "turn" else 0.
        trigger = segment["stop_trigger"] if segment else action["magnitude"]
        request = self.safety.MoveRequest(vx, vyaw, action["duration_s"], lim["resend_period_s"], trigger)

        def progress():
            value = read_progress("driving")
            return value if value is not None and value >= -.02 else None

        def record_event(event):
            result["events"].append(event)
            if event.get("event") == "stop_observed":
                result["observed_stationary"] = event["confirmed"]

        try:
            wait_end = self.clock() + lim["stationary_dwell_s"] + .5
            while not mon.is_still():
                if mon.operator_input() or self.clock() >= wait_end:
                    raise Hold("initial_stationary_dwell_unverified")
                self.sleep(.02)
            progress_fn = self.guards.progress(action)  # reference pose after stationary dwell
            out = self.safety.execute(request, route, mon, confirm=self.safety.confirmation_phrase(request),
                                      execute_flag=True, verified_stop=True, other_source=False,
                                      progress=progress, settle_timeout=lim["stationary_dwell_s"]+1.,
                                      clock=self.clock, sleep=self.sleep,
                                      log=record_event)
            termination = ("feedback_trigger" if out.end_reason.startswith("target reached") else
                           "blocked_stall" if out.end_reason.startswith("stalled") else
                           "; ".join(reason) or out.end_reason)
            result.update(reason="; ".join(reason) or out.end_reason, observed_stationary=out.stop_confirmed,
                          status="cancelled" if cancelled.is_set() else "completed" if out.stop_confirmed else "failed")
            unconfirmed_only = (result["status"], result["reason"]) if out.stop_confirmed is False else None
        except BaseException as error:
            result.update(status="failed", reason=f"executor_exception: {error}")
            # The audited loop cleans up once its try block is entered. A claim RPC
            # exception occurs before that block; attempt bounded cleanup separately.
            if result["started_monotonic_s"] is not None and not result["stop_send_statuses"]:
                for _ in range(3):
                    try:
                        route.send_stop()
                    except Exception as stop_error:
                        result["events"].append({"stop_error": str(stop_error)})
                    self.sleep(.05)
            if result["started_monotonic_s"] is not None and result["ownership_released"] is not True:
                try:
                    route.release_api_control()
                except Exception as release_error:
                    result["events"].append({"release_error": str(release_error)})
        result["measured_progress"] = last_value
        if result["started_monotonic_s"] is not None and (len(result["stop_send_statuses"]) < 3 or
                                                         any(code != 0 for code in result["stop_send_statuses"])):
            result["status"] = "failed"
            result["reason"] += "; stop_local_sends_failed_or_unknown"
        # Audited evidence includes movement after release. Observe independently for
        # 3 s; any stale sample, or movement after the robot first comes to rest, makes the
        # post-release result false. It must then stay still for at least 2.5 s to the end
        # (2026-11-23 Sm23TA: one moving sample right at release, 56 still after it).
        if result["started_monotonic_s"] is not None:
            settled, rest_since = True, None
            end = self.clock() + 3.
            while self.clock() < end:
                now = self.clock()
                age = self.monitor.state_age()
                still = self.monitor.is_still()
                result["events"].append(finite_json({"event": "post_release_sample", "monotonic_s": now,
                                                    "state_age_s": age, "still": still,
                                                    "robot_state": self.monitor.snapshot()}))
                fresh = age is not None and 0 <= age <= lim["state_age_s"]
                if still is True and rest_since is None:
                    rest_since = now
                settled &= fresh and (still is True or rest_since is None)
                self.sleep(.05)
            result["post_release_stationary"] = bool(
                settled and rest_since is not None and self.clock() - rest_since >= 2.5)
            try:
                enabled = self.route.avoidance_enabled()
                result["events"].append({"avoidance_enabled_after": enabled})
                if enabled is not True:
                    result["status"], result["reason"] = "failed", "avoidance_after_unverified"
            except Exception as error:
                result["events"].append({"avoidance_after_error": str(error)})
                result["status"] = "failed"
        if segment and result["started_monotonic_s"] is not None:
            # A turn can still spring back slowly (about 0.03 rad/s, 2026-11-22) when the controller's
            # settle window closes. Three seconds of fresh, still samples after release then confirm
            # the stop; the settled value is read after them. An unknown (None) observation never does.
            if result["observed_stationary"] is False and result["post_release_stationary"] is True:
                result["events"].append({"event": "stop_verified_by", "source": "post_release_stationary"})  # schema-free log
                if unconfirmed_only is not None and unconfirmed_only == (result["status"], result["reason"]) \
                        and result["status"] == "failed":
                    result["status"] = "completed"  # nothing else failed after the settle window
            verified = (result["observed_stationary"] is not None and result["ownership_released"] is True and
                        result["post_release_stationary"] is True and len(result["stop_send_statuses"]) >= 3 and
                        all(code == 0 for code in result["stop_send_statuses"]))
            settled_value = None
            if verified:
                try:
                    settled_value = read_progress("settled")
                    if action["type"] == "turn":
                        final_pose = self.guards.snapshot()["pose"]
                        start_pose = ticket["proposal"]["identity"]["pose"]
                        age_check(final_pose["monotonic_s"], self.clock(), lim["state_age_s"], "settled_pose")
                        if ((start_pose["frame"], start_pose["robot_epoch"]) != (final_pose["frame"], final_pose["robot_epoch"]) or
                                math.hypot(final_pose["x_m"]-start_pose["x_m"], final_pose["y_m"]-start_pose["y_m"]) >
                                segment["translation_budget_m"]):
                            termination = "settled_turn_outside_translation_budget"
                except Exception as error:
                    result["events"].append({"settled_feedback_error": str(error)})
                    termination = "settled_feedback_unavailable"
            if result["status"] == "failed":
                termination = result["reason"]
            measurement = measure_outcome(action, at_zero, settled_value, peak, termination=termination, stop_verified=verified)
            result.update(motion_measurement=measurement, measured_progress=settled_value)
            if not verified:
                result["status"] = "failed"
            elif result["status"] == "completed" and (measurement["interrupted_or_blocked"] or
                                                       measurement["outcome"] != "target_within_tolerance"):
                result["status"] = "ended"
        return result


class ExecutorClient:
    """JSONL subprocess client. Reader/heartbeat/cancel do not wait for inference."""
    def __init__(self, command, *, env=None, nonce=None):
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, text=True, bufsize=1, env=env)
        self.messages, self.errors = Queue(), []
        self.lock, self.closed = threading.Lock(), threading.Event()
        self.latest_guards = None
        self.guard_history = deque(maxlen=200)
        self.nonce = nonce
        self.ready = False
        self.events = []
        threading.Thread(target=self._read, daemon=True, name="navigation-executor-reader").start()
        threading.Thread(target=self._stderr, daemon=True, name="navigation-executor-stderr").start()
        threading.Thread(target=self._beat, daemon=True, name="navigation-executor-heartbeat").start()

    def _read(self):
        for line in self.process.stdout:
            try:
                item = json.loads(line)
            except ValueError:
                self.errors.append("invalid_executor_stdout")
                continue
            if item.get("type") == "guards":
                if self.nonce is not None and item.get("nonce") != self.nonce:
                    self.errors.append("wrong_executor_guard_session")
                    continue
                self.latest_guards = item
                self.guard_history.append(item["snapshot"])
            elif item.get("type") == "ready":
                self.ready = True
            else:
                self.messages.put(item)

    def _stderr(self):
        for line in self.process.stderr:
            self.errors.append(line.rstrip())
            self.errors[:] = self.errors[-100:]

    def _beat(self):
        while not self.closed.wait(.1) and self.process.poll() is None:
            self.send({"type": "heartbeat"})

    def send(self, item):
        with self.lock:
            try:
                self.process.stdin.write(json.dumps(item, allow_nan=False) + "\n")
                self.process.stdin.flush()
            except (BrokenPipeError, OSError, ValueError):
                self.errors.append("executor_pipe_closed")

    def submit(self, ticket):
        self.send({"type": "execute", "ticket": ticket})

    def guard_at(self, observed_s, max_gap_s):
        history = list(self.guard_history)
        if not history:
            raise Hold("executor_pose_history_unavailable")
        match = min(history, key=lambda item: abs(item["pose"]["monotonic_s"]-observed_s))
        if abs(match["pose"]["monotonic_s"]-observed_s) > max_gap_s:
            raise Hold("pose_not_available_at_observation_time")
        return deepcopy(match)

    def cancel(self):
        self.send({"type": "cancel"})

    def poll(self):
        try:
            return self.messages.get_nowait()
        except Empty:
            return None

    def close(self):
        self.cancel()
        self.send({"type": "close"})
        self.closed.set()
        # Do NOT kill a live cleanup loop. Its own deadline/heartbeat owns shutdown.
        try:
            self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.errors.append("executor_exit_unverified_independent_intervention_required")
        if self.process.poll() is None:
            return False
        for pipe in (self.process.stdin, self.process.stdout, self.process.stderr):
            pipe.close()
        return True
