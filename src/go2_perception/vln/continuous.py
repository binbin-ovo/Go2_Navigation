"""Offline rolling SEARCH/APPROACH governor. Inputs and outputs are data on one monotonic clock."""

from copy import deepcopy
import math

from .episode_policy import area_check, select_action
from .motion_profile import LIMITS, PROFILE, TURN_DRIFT_M, Hold, wrap
from .supervised import FRESHNESS_S, MODEL_REVISION, age_check, current_checks, validate_record


PERIOD_S = LIMITS["resend_period_s"]
PACKET_TTL_S = 2 * PERIOD_S
PLAN_LEASE_S = 0.60  # From model result availability, never renewed by a control tick.
SCOPE_KEYS = ("session_id", "trial_id", "instruction_id", "instruction", "clock_epoch")
ZERO = (0.0, 0.0)


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def same_frame(a, b):
    return (a["frame"], a["robot_epoch"]) == (b["frame"], b["robot_epoch"])


def offsets(origin, pose):
    if not same_frame(origin, pose):
        raise Hold("pose_frame_or_epoch_changed")
    dx, dy = pose["x_m"] - origin["x_m"], pose["y_m"] - origin["y_m"]
    c, s = math.cos(origin["yaw_rad"]), math.sin(origin["yaw_rad"])
    return dx * c + dy * s, dy * c - dx * s, wrap(pose["yaw_rad"] - origin["yaw_rad"])


def progress(action, origin, pose, limits):
    along, lateral, yaw = offsets(origin, pose)
    if action["type"] == "forward":
        if abs(lateral) > limits["lateral_m"] or abs(yaw) > limits["straight_yaw_rad"]:
            raise Hold("forward_path_drift", lateral_m=lateral, yaw_rad=yaw)
        value = along
    else:
        if math.hypot(along, lateral) > TURN_DRIFT_M:
            raise Hold("turn_translation_drift")
        value = yaw if action["direction"] == "left" else -yaw
    if value < -action["segment"]["noise_floor"]:
        raise Hold("motion_opposes_plan", measured=value)
    return value


class ContinuousNavigator:
    """A single owner calls tick independently of inference. There is no transport or actuator."""

    def __init__(
        self,
        capability,
        scope,
        *,
        target,
        operator,
        origin,
        started_s,
        reference_sha256=None,
        area_radius_m=2.0,
        max_plans=5,
        duration_s=30.0,
        travel_m=1.5,
        turn_rad=math.pi / 2,
    ):
        validate_record("capability", capability, 2)
        validate_record("pose", origin)
        limits = capability["limits"]
        if (
            capability["motion_profile"] != PROFILE
            or any(limits[k] != v for k, v in LIMITS.items())
            or not 0 < limits["state_age_s"] <= 0.25
            or not 0 < limits["geometry_age_s"] <= 0.5
            or not 0 < limits["lateral_m"] <= 0.08
            or not 0 < limits["straight_yaw_rad"] <= 0.10
            or not 1 <= limits["stationary_dwell_s"] <= 2
        ):
            raise ValueError("unsupported motion profile")
        if not (
            type(max_plans) is int
            and 1 <= max_plans <= 5
            and number(started_s)
            and 0 < duration_s <= 60
            and 1.2 <= area_radius_m <= 3
            and 0 < travel_m <= 3
            and 0 < turn_rad <= math.pi
            and target
            and operator
        ):
            raise ValueError("invalid offline episode bounds")
        self.cap = deepcopy(capability)
        self.scope = {k: scope[k] for k in SCOPE_KEYS}
        self.reference_sha256 = reference_sha256
        self.grant = dict(
            target=target,
            operator=operator,
            origin=dict(origin),
            area_radius_m=area_radius_m,
            alignment_deg=15,
        )
        self.start, self.deadline = started_s, started_s + duration_s
        self.max_plans, self.max_travel, self.max_turn = max_plans, travel_m, turn_rad
        self.plans = self.rollovers = 0
        self.travel = self.turn = 0.0
        self.views = []
        self.last_pose, self.last_tick = dict(origin), None
        self.last_packet_expiry = None
        self.last_capture, self.seen, self.seen_frames = -math.inf, set(), set()
        self.needs_initial_dwell = True
        self.last_guard_times = None
        self.capture_evidence = None
        self.plan = self.terminal = self.braked_at = self.rest_anchor = None
        self.rest_ready = self.moving = False
        self.reason = "awaiting_first_decision"

    def _brake(self, reason, now):
        if self.moving or self.plan or self.braked_at is None:
            self.braked_at, self.rest_anchor, self.rest_ready = now, None, False
        self.plan = None
        self.moving = False
        self.reason = reason

    def _guard(self, now, guard):
        if guard is None:
            raise Hold("no_delivered_guard")
        available, snap = guard["available_s"], guard["snapshot"]
        age_check(available, now, self.cap["limits"]["state_age_s"], "guard_delivery")
        interlock = guard["interlock"]
        if interlock["neutral"] is not True or interlock["healthy"] is not True:
            raise Hold("operator_or_state_interlock")
        pose, geo = snap["pose"], snap["geometry"]
        validate_record("pose", pose)
        validate_record("geometry", geo, 2)
        for obj, limit, label in ((pose, "state_age_s", "odometry"), (geo, "geometry_age_s", "lidar")):
            age_check(obj["monotonic_s"], now, self.cap["limits"][limit], label)
            if obj["monotonic_s"] > available or not same_frame(self.grant["origin"], obj):
                raise Hold("sensor_delivery_frame_or_epoch_changed")
        if snap["capture_clock_epoch"] != self.scope["clock_epoch"]:
            raise Hold("camera_epoch_changed")
        signature = (snap.get("capture_bound_basis"), snap.get("capture_uncertainty_bound_s"))
        if self.capture_evidence is not None and signature != self.capture_evidence:
            raise Hold("capture_bound_changed")
        if (
            not isinstance(signature[0], str)
            or not signature[0].strip()
            or not number(signature[1])
            or not 0 <= signature[1] < FRESHNESS_S
        ):
            raise Hold("unknown_capture_age_bound")
        self.capture_evidence = signature
        if not geo["available"]:
            raise Hold("lidar_unavailable")
        if any(number(interlock[k]) and interlock[k] > available for k in ("radio_s", "state_s")):
            raise Hold("interlock_postdates_delivery")
        if interlock["radio_s"] is None:
            raise Hold("radio_receipt_time_missing")
        age_check(interlock["radio_s"], now, self.cap["limits"]["state_age_s"], "radio")
        age_check(interlock["state_s"], now, self.cap["limits"]["state_age_s"], "sport_state")
        if not self.plans and interlock["stationary"] is not True:
            self.needs_initial_dwell = True
        stamps = (available, pose["monotonic_s"], geo["monotonic_s"])
        if self.last_guard_times and any(b < a for a, b in zip(self.last_guard_times, stamps)):
            raise Hold("regressing_guard_sample")
        self.last_guard_times = stamps
        delta = offsets(self.last_pose, pose)
        if pose["monotonic_s"] < self.last_pose["monotonic_s"] or (
            pose["monotonic_s"] == self.last_pose["monotonic_s"] and any(abs(v) > 1e-9 for v in delta)
        ):
            raise Hold("regressing_or_conflicting_odometry")
        self.travel += math.hypot(*delta[:2])
        self.turn += abs(delta[2])
        self.last_pose = dict(pose)
        area_check(self.grant, pose, attempts=self.plans)
        if self.needs_initial_dwell:
            idle = guard.get("prior_stillness")
            if idle and interlock["stationary"] is True:
                since, last, checked = (
                    idle[k] for k in ("still_since_monotonic_s", "last_idle_sample_s", "monotonic_s")
                )
                if (
                    all(number(v) for v in (since, last, checked))
                    and since <= last <= checked <= available
                    and now - last <= self.cap["limits"]["state_age_s"]
                    and last - since >= self.cap["limits"]["stationary_dwell_s"]
                ):
                    self.needs_initial_dwell = False
        if self.needs_initial_dwell or self.braked_at is not None and pose["monotonic_s"] >= self.braked_at:
            if interlock["stationary"] is not True:
                self.rest_anchor, self.rest_ready = None, False
                return snap
            if self.rest_anchor is None:
                self.rest_anchor = dict(pose)
            along, side, yaw = offsets(self.rest_anchor, pose)
            if math.hypot(along, side) > 0.01 or abs(yaw) > 0.02:
                self.rest_anchor, self.rest_ready = dict(pose), False
            else:
                self.rest_ready = (
                    pose["monotonic_s"] - self.rest_anchor["monotonic_s"]
                    >= self.cap["limits"]["stationary_dwell_s"]
                )
            if self.rest_ready:
                self.needs_initial_dwell = False
        return snap

    def _offer(self, delivery, now, events):
        p, evidence, ready = delivery["proposal"], delivery["grounding"], delivery["available_s"]
        validate_record("proposal", p, 2)
        ident = p["identity"]
        stamp, received = ident["observed_monotonic_s"], ident["received_monotonic_s"]
        pid, bound = ident["prediction_id"], ident["uncertainty_bound_s"]
        if any(ident[k] != v for k, v in self.scope.items()):
            events.append({"ignored": "different_episode_instruction_or_epoch", "id": pid})
            return
        if pid in self.seen or ident["observation_id"] in self.seen_frames or stamp <= self.last_capture:
            events.append({"ignored": "duplicate_or_superseded", "id": pid})
            return
        if not number(ready) or not stamp <= received <= ready <= now:
            raise Hold("noncausal_proposal_timestamps")
        # Only accepted/observed in-scope arrivals affect the watermark.
        self.last_capture = stamp
        self.seen.add(pid)
        self.seen_frames.add(ident["observation_id"])
        if len(self.seen) > 1200:  # One fresh decision per tick for the 60 s maximum.
            raise Hold("decision_capacity_exceeded")
        if not number(bound) or bound < 0:
            raise Hold("unknown_capture_age_bound")
        age_check(stamp, now, FRESHNESS_S - bound, "observation")
        age_check(ready, now, PLAN_LEASE_S, "plan_result")
        if p["synthetic"] != (self.cap["kind"] == "synthetic"):
            raise Hold("synthetic_source_mismatch")
        if not p["synthetic"] and any(
            p["model_identity"][k] != MODEL_REVISION for k in ("revision", "processor_revision")
        ):
            raise Hold("model_identity_changed")
        for key, limit in (("pose", "state_age_s"), ("geometry", "geometry_age_s")):
            item = ident[key]
            if (
                not same_frame(self.grant["origin"], item)
                or item["monotonic_s"] > ready
                or abs(item["monotonic_s"] - stamp) > self.cap["limits"][limit]
            ):
                raise Hold("source_pose_or_geometry_unbound")
        if p["native_stop"]:
            self.terminal = "native_stop_arrival_unverified"
            return
        if self.needs_initial_dwell:
            raise Hold("initial_stationarity_unverified")
        if (
            not p["synthetic"]
            and evidence["status"] == "supported"
            and (not self.reference_sha256 or evidence.get("reference_sha256") != self.reference_sha256)
        ):
            raise Hold("different_identified_instance")
        old = self.plan
        selection = select_action(
            self.grant,
            self.cap,
            p,
            evidence,
            self.views,
            self.plans,
            bool(old and old["selection"]["aligned"]),
        )
        events.append({"selection": selection, "id": pid})
        action = selection["action"]
        if not action:
            raise Hold(selection["reason"])
        if action["type"] == "turn" and self.cap["turn_scope"] != "sensor_only":
            raise Hold("continuous_turn_requires_observed_sector")
        if old and (selection["mode"], action["direction"]) != (
            old["selection"]["mode"],
            old["action"]["direction"],
        ):
            raise Hold("mode_or_direction_change_requires_settled_view")
        if self.braked_at is not None and (
            not self.rest_ready
            or stamp - bound <= self.braked_at
            or self.rest_anchor is None
            or stamp - bound < self.rest_anchor["monotonic_s"] + self.cap["limits"]["stationary_dwell_s"]
        ):
            raise Hold("awaiting_new_settled_observation")
        if self.plans >= self.max_plans:
            self.terminal = "plan_budget"
            return
        self.plan = dict(
            action=deepcopy(action),
            selection=deepcopy(selection),
            identity=deepcopy(ident),
            ready=ready,
            issued=False,
        )

    def tick(self, now, guard, *, delivery=None, cancel=False):
        """Return one expiring, simulated velocity sample. A late call cannot extend an old packet."""
        if not number(now) or now < self.start or self.last_tick is not None and now <= self.last_tick:
            self.terminal = "invalid_control_clock"
        packet = {
            "t": now,
            "expires_s": now + PACKET_TTL_S if number(now) else None,
            "velocity": [0.0, 0.0],
            "mode": "HOLD",
            "prediction_id": None,
            "events": [],
        }
        if cancel:
            self.terminal = "cancelled"
        if number(now) and now >= self.deadline:
            self.terminal = self.terminal or "episode_deadline"
        gap = number(now) and number(self.last_tick) and now - self.last_tick > PACKET_TTL_S + 1e-8
        self.last_tick = now
        old_id = self.plan["identity"]["prediction_id"] if self.plan else None
        if not self.terminal:
            try:
                if delivery is not None and delivery["proposal"]["native_stop"]:
                    self._offer(delivery, now, packet["events"])
                    delivery = None
                if not self.terminal:
                    self._advance(now, guard, delivery, packet, gap)
            except (Hold, KeyError, TypeError, ValueError, IndexError) as exc:
                reason = str(exc) if isinstance(exc, Hold) else "malformed_input"
                packet.update(
                    veto={
                        "reason": reason,
                        "details": getattr(exc, "details", {}),
                        "guard": deepcopy(guard),
                        "previous_prediction_id": old_id,
                    }
                )
                if reason in {
                    "operator_or_state_interlock",
                    "camera_epoch_changed",
                    "sensor_delivery_frame_or_epoch_changed",
                    "regressing_or_conflicting_odometry",
                    "regressing_guard_sample",
                    "outside_inspected_area",
                    "decision_capacity_exceeded",
                    "capture_bound_changed",
                }:
                    self.terminal = reason
                if reason in {
                    "control_tick_gap",
                    "no_delivered_guard",
                    "lidar_unavailable",
                    "malformed_input",
                    "radio_receipt_time_missing",
                    "unknown_capture_age_bound",
                } or reason.startswith("stale_or_future_"):
                    if not self.plans:
                        self.needs_initial_dwell = True
                    self.rest_anchor = None
                    self.rest_ready = False
                # An empty initial mailbox is normal; do not fabricate a prior movement or dwell.
                if self.plan or self.moving or self.braked_at is not None:
                    self._brake(reason, now)
                else:
                    self.reason = reason
        if self.terminal:
            self._brake(self.terminal, now)
            packet.update(mode="TERMINAL", velocity=[0.0, 0.0])
        packet.update(
            reason=self.reason,
            plans=self.plans,
            rollovers=self.rollovers,
            odom_travel_m=self.travel,
            odom_turn_rad=self.turn,
        )
        return packet

    def _advance(self, now, guard, delivery, packet, gap):
        snap = self._guard(now, guard)
        if gap:
            raise Hold("control_tick_gap")
        if self.moving and self.last_packet_expiry is not None and now > self.last_packet_expiry + 1e-8:
            raise Hold("command_packet_expired")
        if self.travel >= self.max_travel or self.turn >= self.max_turn:
            self.terminal = "travel_or_rotation_budget"
        if self.plan and delivery is not None and not self.terminal:
            # A late replacement cannot bridge an expired lease or a passed stop trigger.
            self._check_plan(self.plan, snap, now)
        if delivery is not None and not self.terminal:
            self._offer(delivery, now, packet["events"])
        if self.plan and not self.terminal:
            self._emit(now, guard, snap, packet)

    def _emit(self, now, guard, snap, packet):
        plan = self.plan
        done = self._check_plan(plan, snap, now)
        a = plan["action"]
        limits = self.cap["limits"]
        expires = min(
            now + PACKET_TTL_S,
            self.deadline,
            plan["ready"] + PLAN_LEASE_S,
            plan["identity"]["observed_monotonic_s"] + FRESHNESS_S - plan["identity"]["uncertainty_bound_s"],
            snap["pose"]["monotonic_s"] + limits["state_age_s"],
            snap["geometry"]["monotonic_s"] + limits["geometry_age_s"],
            guard["interlock"]["radio_s"] + limits["state_age_s"],
            guard["interlock"]["state_s"] + limits["state_age_s"],
            guard["available_s"] + limits["state_age_s"],
            now + (a["segment"]["stop_trigger"] - done) / a["speed"],
        )
        if expires <= now:
            raise Hold("command_validity_elapsed")
        self.last_packet_expiry = expires
        if not plan["issued"]:
            self.plans += 1
            self.rollovers += int(self.moving)
            plan["issued"] = True
            self.views.append(
                dict(
                    plan["identity"]["pose"],
                    attempt={"mode": plan["selection"]["mode"], "direction": a["direction"]},
                )
            )
        self.moving, self.reason = True, "tracking"
        self.braked_at = self.rest_anchor = None
        self.rest_ready = False
        speed = a["speed"]
        packet.update(
            velocity=[speed, 0.0]
            if a["type"] == "forward"
            else [0.0, speed if a["direction"] == "left" else -speed],
            mode=plan["selection"]["mode"],
            prediction_id=plan["identity"]["prediction_id"],
            source_observation_id=plan["identity"]["observation_id"],
            capture_monotonic_s=plan["identity"]["observed_monotonic_s"],
            result_monotonic_s=plan["ready"],
            capture_age_upper_s=now
            - plan["identity"]["observed_monotonic_s"]
            + plan["identity"]["uncertainty_bound_s"],
            selection_source=plan["selection"]["source"],
            progress=done,
            action=deepcopy(a),
            expires_s=expires,
        )

    def _check_plan(self, plan, snap, now):
        i, a = plan["identity"], plan["action"]
        if (
            not snap.get("capture_bound_basis")
            or snap.get("capture_uncertainty_bound_s") != i["uncertainty_bound_s"]
        ):
            raise Hold("capture_bound_changed")
        age_check(i["observed_monotonic_s"], now, FRESHNESS_S - i["uncertainty_bound_s"], "observation")
        if now >= plan["ready"] + PLAN_LEASE_S:
            raise Hold("plan_lease_expired")
        done = progress(a, i["pose"], snap["pose"], self.cap["limits"])
        current_checks(snap, a, self.cap, now, capture=False, progress=done)
        area_check(self.grant, snap["pose"], a, progress=done, attempts=self.plans + int(not plan["issued"]))
        if done >= a["segment"]["stop_trigger"] - 1e-8:
            raise Hold("primitive_stop_trigger")
        linear = a["type"] == "forward"
        remaining = self.max_travel - self.travel if linear else self.max_turn - self.turn
        if remaining + 1e-9 < a["speed"] * PACKET_TTL_S:
            self.terminal = "travel_or_rotation_budget"
            raise Hold(self.terminal)
        return done


def simulated_velocity(packet, at_s):
    """Independent simulated sink watchdog. It has no external connection."""
    if (
        packet is None
        or not all(number(v) for v in (at_s, packet.get("t"), packet.get("expires_s")))
        or not packet["t"] <= at_s < packet["expires_s"]
    ):
        return ZERO
    return tuple(packet["velocity"])
