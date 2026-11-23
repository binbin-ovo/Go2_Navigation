"""One episode grant, one command owner, at most five single-use action leases."""

from copy import deepcopy
import math

from go2_perception.navigation import (MAX_EPISODE_SECONDS, area_check, bounded_number, image_bearing, select_action,
                                       supported_ground)
from .motion_profile import INSPECTED, PROFILE, Hold, wrap
from .supervised import (
    FRESHNESS_S,
    MODEL_REVISION,
    age_check,
    capability_check,
    current_checks,
    digest,
    pose_check,
    stop_and_release_verified,
    validate_record,
)
from .supervised_executor import Executor, FakeGuards, result_record

SCHEMA = "go2-five-move-episode-v1"
TRACK_HEADING_RAD = math.radians(6)
TRACK_TRANSLATION_M = 0.8
TRACK_MAX = 2
TERMINAL = {
    "GOAL_CONFIRMED",
    "MODEL_STOPPED_ARRIVAL_UNVERIFIED",
    "MOVE_BUDGET_EXHAUSTED",
    "BLOCKED_OR_NO_USEFUL_ACTION",
    "SENSOR_OR_CONTROL_FAILURE",
    "OPERATOR_ABORT",
    "EPISODE_DEADLINE",
}


class EpisodeFakeGuards(FakeGuards):
    """Invented observed room for the explicitly synthetic launcher."""

    def snapshot(self, observed_s=None):
        snap = super().snapshot(observed_s)
        snap["geometry"].update(
            forward_clear_m=3.0,
            observed_turn_disc={"radius_m": 0.65, "floor_returns": 50, "obstacle_returns": 0},
        )
        return snap


class EpisodeExecutor(Executor):
    """Authoritative episode state inside the existing lock-owning executor process.

    Only observations cross the dispatch boundary. The child selects and spends the
    action before invoking the audited per-action loop. A fresh process needs a new grant.
    """

    def __init__(self, *args, emit, **kwargs):
        super().__init__(*args, **kwargs)
        self.emit = emit
        self.retain_lock = False
        self.sensor_failure = None
        self.abort_reason = "episode_stop_latched"
        self.grant = None
        self.state = "IDLE"
        self.mode, self.reason, self.outcome = "SEARCH", "awaiting_area_authorization", None
        self.inspected = []  # At most 64 accepted decisions per episode.
        self.target_heading = None  # odometry yaw of the last verified instance
        self.target_xy, self.tracked = None, 0
        self.attempts, self.views, self.predictions, self.observations = [], [], set(), set()
        self.found, self.aligned = False, False
        self.after = 0.0
        self.last_target = self.last = self.selection = self.previous_outcome = None

    def arrival_review_available(self):
        if not self.last or self.cancelled.is_set() or self.retain_lock:
            return False
        eligible = self.state == "IDENTIFYING" or self.outcome in (
            "MODEL_STOPPED_ARRIVAL_UNVERIFIED",
            "MOVE_BUDGET_EXHAUSTED",
        )
        return eligible and self.last["exposure_lower_monotonic_s"] > self.after

    def public(self):
        return {
            "schema": SCHEMA,
            "episode_id": self.nonce,
            "state": self.state,
            "mode": self.mode,
            "reason": self.reason,
            "outcome": self.outcome,
            "cleanup_unverified": self.retain_lock,
            "previous_outcome": self.previous_outcome,
            "moves": len(self.attempts),
            "max_moves": self.grant["max_moves"] if self.grant else 5,
            "target": self.grant["target"] if self.grant else None,
            "target_found": self.found,
            "after_monotonic_s": self.after,
            "deadline_monotonic_s": self.grant["deadline"] if self.grant else None,
            "last_observation": self.last,
            "arrival_review_available": self.arrival_review_available(),
            "last_supported_target": self.last_target,
            "selected": self.selection,
            "attempts": deepcopy(self.attempts),
        }

    def publish(self):
        self.emit({"type": "episode_state", "nonce": self.nonce, "episode": self.public()})

    def finish(self, outcome, reason):
        if outcome not in TERMINAL:
            raise ValueError("unknown episode outcome")
        if self.outcome is None:
            self.outcome, self.state, self.reason = outcome, "TERMINAL", reason
            self.publish()

    def interrupt_sensor(self, reason):
        # Called by the IPC reader even while the audited motion loop owns the main thread.
        self.sensor_failure = str(reason)
        self.cancelled.set()

    def poll_radio(self):
        """Latch intervention between model decisions and at existing action guard checks."""
        if not self.live or not self.grant or self.outcome or self.cancelled.is_set():
            return
        remote = self.motion.monitor
        try:
            age_check(remote.remote_t, self.clock(), self.cap["limits"]["state_age_s"], "remote_state")
            if remote.operator_input():
                self.abort_reason = "operator_radio_input"
                self.cancelled.set()
        except (OSError, ValueError, RuntimeError) as exc:
            self.interrupt_sensor(str(exc))

    def tick(self):
        self.poll_radio()
        if self.cancelled.is_set() and self.outcome is None:
            self.finish(
                "SENSOR_OR_CONTROL_FAILURE" if self.sensor_failure else "OPERATOR_ABORT",
                self.sensor_failure or self.abort_reason,
            )
        elif self.grant and self.clock() >= self.grant["deadline"] and self.outcome is None:
            full = len(self.attempts) == self.grant["max_moves"]
            self.finish("MOVE_BUDGET_EXHAUSTED" if full else "EPISODE_DEADLINE", "episode_deadline_reached")

    def arm(self, grant):
        if self.state != "IDLE" or self.grant is not None or self.cancelled.is_set():
            raise Hold("episode_already_armed_or_closed")
        now = self.clock()
        capability_check(self.cap, live=self.live, wall_s=self.wall())
        if self.cap.get("motion_profile") != PROFILE or self.cap.get("turn_scope") != INSPECTED:
            raise Hold("effective_inspected_profile_required")
        if (
            self.cap["mapping"].get("orientation_basis") != "relative_planar_heading_ccw"
            or "forward" not in self.cap["mapping"]["verified_actions"]
        ):
            raise Hold("verified_forward_and_heading_mapping_required")
        if (
            grant["schema"] != SCHEMA
            or grant["episode_id"] != self.nonce
            or grant["capability_sha256"] != digest(self.cap)
            or grant["area_inspected"] is not True
            or grant["exclusive_and_intervention_confirmed"] is not True
        ):
            raise Hold("invalid_episode_grant")
        for key in (
            "target",
            "operator",
            "instruction",
            "session_id",
            "trial_id",
            "instruction_id",
            "clock_epoch",
        ):
            if not isinstance(grant[key], str) or not 1 <= len(grant[key].strip()) <= 512:
                raise Hold("missing_grant_" + key)
        if grant["instruction"] != f"Go to the {grant['target']} and stop.":
            raise Hold("episode_target_and_instruction_mismatch")
        if type(grant["max_moves"]) is not int or not 1 <= grant["max_moves"] <= 5:
            raise Hold("move_budget_out_of_range")
        bounded_number(grant["alignment_deg"], 8, 18, "alignment_band")
        bounded_number(grant["area_radius_m"], 1.2, 3.0, "area_radius")
        age_check(grant["issued"], now, 1.0, "episode_grant")
        bounded_number(grant["deadline"] - grant["issued"], 5.0, MAX_EPISODE_SECONDS, "episode_duration")
        snap = self.guards.snapshot()
        pose_check(grant["origin"], snap["pose"], self.cap["limits"])
        age_check(grant["origin"]["monotonic_s"], now, self.cap["limits"]["state_age_s"], "area_origin")
        area_check(grant, snap["pose"])
        if self.live and (
            not snap.get("capture_bound_basis") or grant["clock_epoch"] != snap.get("capture_clock_epoch")
        ):
            raise Hold("camera_epoch_or_bound_unverified")
        self.grant, self.after = deepcopy(grant), grant["issued"]
        self.state, self.reason = "OBSERVING", "awaiting_fresh_observation"
        self.emit({"type": "episode_authorization", "nonce": self.nonce, "grant": self.grant})
        self.publish()

    def check_observation(self, proposal, snap):
        g, i, now = self.grant, proposal["identity"], self.clock()
        validate_record("proposal", proposal, 2)
        if proposal["synthetic"] == self.live or any(
            i[k] != g[k] for k in ("session_id", "trial_id", "instruction_id", "instruction", "clock_epoch")
        ):
            raise Hold("episode_instruction_source_or_epoch_changed")
        if self.live and any(
            proposal["model_identity"][key] != MODEL_REVISION for key in ("revision", "processor_revision")
        ):
            raise Hold("unexpected_episode_model")
        bound = i["uncertainty_bound_s"]
        if (
            bound is None
            or not math.isfinite(bound)
            or bound < 0
            or (
                self.live
                and (
                    not snap.get("capture_bound_basis")
                    or snap.get("capture_clock_epoch") != i["clock_epoch"]
                    or snap.get("capture_uncertainty_bound_s") != bound
                )
            )
        ):
            raise Hold("camera_epoch_or_bound_unverified")
        if i["observed_monotonic_s"] - bound <= self.after:
            raise Hold("observation_precedes_settling_or_grant")
        age_check(i["observed_monotonic_s"], now, FRESHNESS_S - bound, "observation")
        age_check(i["received_monotonic_s"], now, FRESHNESS_S, "receipt")
        for key, ttl in (("pose", "state_age_s"), ("geometry", "geometry_age_s")):
            if min(i[key]["monotonic_s"], snap[key]["monotonic_s"]) <= self.after:
                raise Hold("guard_observation_precedes_settling_or_grant")
            if abs(i[key]["monotonic_s"] - i["observed_monotonic_s"]) > self.cap["limits"][ttl]:
                raise Hold("decision_guard_not_bound_to_source")
            age_check(snap[key]["monotonic_s"], now, self.cap["limits"][ttl], key)
        pose_check(i["pose"], snap["pose"], self.cap["limits"])
        area_check(g, snap["pose"], attempts=len(self.attempts))

    def track(self, proposal, evidence):
        """Keep an appearance-verified instance through views where features fail (a plain front).

        The native mark must point within TRACK_HEADING_RAD of the odometry heading where the
        instance was last verified, after less than TRACK_TRANSLATION_M of travel, for at most
        TRACK_MAX consecutive views before features must confirm it again (2026-11-23 6t7lPf).
        """
        point = proposal["point_fraction"]
        if (evidence.get("status") != "uncertain" or evidence.get("method") != "reference_inconclusive_seek_view"
                or self.target_heading is None or point is None or self.tracked >= TRACK_MAX):
            return evidence
        pose = proposal["identity"]["pose"]
        moved = math.hypot(pose["x_m"] - self.target_xy[0], pose["y_m"] - self.target_xy[1])
        error = wrap(pose["yaw_rad"] + image_bearing(point, evidence["camera"], self.cap["mapping"])
                     - self.target_heading)
        if moved > TRACK_TRANSLATION_M or abs(error) > TRACK_HEADING_RAD:
            return evidence
        x, y = point
        return dict(evidence, status="supported", method="tracked_instance_heading",
                    roi=[max(0.0, x - 0.05), max(0.0, y - 0.05), min(1.0, x + 0.05), min(1.0, y + 0.05)],
                    tracking={"from_prediction_id": self.last_target["prediction_id"], "heading_error_rad": error,
                              "translation_m": moved, "consecutive": self.tracked + 1})

    def observe(self, message):
        self.tick()
        if self.state != "OBSERVING" or message["episode_id"] != self.nonce:
            raise Hold("inactive_or_wrong_episode")
        p, evidence = message["proposal"], message["evidence"]
        ident = p["identity"]
        if ident["prediction_id"] in self.predictions or ident["observation_id"] in self.observations:
            raise Hold("decision_or_source_already_seen")
        if len(self.predictions) >= 64:
            return self.finish("BLOCKED_OR_NO_USEFUL_ACTION", "decision_limit")
        self.predictions.add(ident["prediction_id"])
        self.observations.add(ident["observation_id"])
        snap = self.guards.snapshot()
        try:
            self.check_observation(p, snap)
        except Hold as exc:
            if str(exc) in (
                "observation_precedes_settling_or_grant",
                "guard_observation_precedes_settling_or_grant",
                "stale_or_future_observation",
                "stale_or_future_receipt",
            ):
                self.reason = str(exc)
                self.emit(
                    {
                        "type": "episode_rejected_observation",
                        "nonce": self.nonce,
                        "reason": self.reason,
                        "prediction_id": ident["prediction_id"],
                    }
                )
                return self.publish()
            raise
        evidence = self.track(p, evidence)
        grounded = supported_ground(self.grant, p, evidence)
        self.inspected.append(dict(ident["pose"]))
        self.found |= grounded
        if not grounded:
            self.aligned = False
        self.mode = "APPROACH" if grounded else "SEARCH"
        self.last = {
            "prediction_id": ident["prediction_id"],
            "observation_id": ident["observation_id"],
            "image_sha256": ident["source_image_sha256"],
            "observed_monotonic_s": ident["observed_monotonic_s"],
            "exposure_lower_monotonic_s": ident["observed_monotonic_s"] - ident["uncertainty_bound_s"],
            "age_at_decision_s": self.clock() - ident["observed_monotonic_s"],
            "grounding": deepcopy(evidence),
            "native_stop": p["native_stop"],
            "point_fraction": p["point_fraction"],
            "waypoints": p["waypoints"],
        }
        if grounded:
            self.last_target = {
                k: self.last[k]
                for k in ("prediction_id", "observation_id", "grounding", "observed_monotonic_s")
            }
            # Odometry heading of the verified instance, so a later SEARCH turns back toward it.
            self.target_heading = ident["pose"]["yaw_rad"] + image_bearing(
                p["point_fraction"], evidence["camera"], self.cap["mapping"])
            self.target_xy = (ident["pose"]["x_m"], ident["pose"]["y_m"])
            self.tracked = self.tracked + 1 if evidence["method"] == "tracked_instance_heading" else 0
        self.emit({"type": "episode_observation", "nonce": self.nonce, "proposal": p, "evidence": evidence})
        if p["native_stop"]:
            return self.finish("MODEL_STOPPED_ARRIVAL_UNVERIFIED", "native_stop")
        if len(self.attempts) >= self.grant["max_moves"]:
            return self.finish("MOVE_BUDGET_EXHAUSTED", "final_view_after_last_attempt")
        try:
            selected = select_action(
                self.grant, self.cap, p, evidence, self.views, len(self.attempts), self.aligned,
                inspected=self.inspected, target_heading=self.target_heading,
            )
        except Hold as exc:
            return self.finish("BLOCKED_OR_NO_USEFUL_ACTION", str(exc))
        if selected.get("review"):
            self.state, self.reason = "IDENTIFYING", selected["reason"]
            return self.publish()
        selected["prediction_id"] = ident["prediction_id"]
        self.selection, self.aligned = selected, selected["aligned"]
        self.dispatch(p, evidence, selected)

    def dispatch(self, proposal, evidence, selected):
        i, action, g = proposal["identity"], selected["action"], self.grant
        slot = len(self.attempts) + 1
        if self.state != "OBSERVING" or slot > min(5, g["max_moves"]) or self.cancelled.is_set():
            raise Hold("episode_dispatch_refused")
        binding = {
            "grant": digest(g),
            "slot": slot,
            "proposal": digest(proposal),
            "grounding": digest(evidence),
            "selection": digest(selected),
        }
        action_id = digest(binding)
        if action_id in self.used:
            raise Hold("action_lease_reused")
        self.used.add(action_id)
        expires = min(g["deadline"], self.clock() + self.cap["limits"]["auth_ttl_s"])
        ticket = {
            "nonce": self.nonce,
            "request_id": action_id,
            "proposal": proposal,
            "action": action,
            "authorization": {"authorization_id": action_id, "expires_monotonic_s": expires},
        }
        row = {
            "slot": slot,
            "action_id": action_id,
            "prediction_id": i["prediction_id"],
            "direction": action["direction"],
            "requested": action["magnitude"],
            "unit": action["unit"],
            "mode": self.mode,
            "source": selected["source"],
        }
        if evidence["method"] == "reference_inconclusive_seek_view":
            row["inconclusive_reference"] = evidence["positive_reference"]["reference_sha256"]
        self.attempts.append(row)
        self.views.append({**i["pose"], "attempt": row})
        self.state, self.reason = "MOVING", selected["reason"]
        self.emit(
            {
                "type": "episode_dispatch",
                "nonce": self.nonce,
                "binding": binding,
                "ticket": ticket,
                "selection": selected,
            }
        )
        self.publish()

        def during(snap, progress):
            self.poll_radio()
            if self.cancelled.is_set():
                raise Hold(self.sensor_failure or self.abort_reason)
            if self.clock() >= expires:
                raise Hold("episode_action_deadline")
            area_check(g, snap["pose"], action, progress=progress or 0.0, attempts=slot)
            if (
                action["type"] == "turn"
                and math.hypot(snap["pose"]["x_m"] - i["pose"]["x_m"], snap["pose"]["y_m"] - i["pose"]["y_m"])
                > action["segment"]["translation_budget_m"]
            ):
                raise Hold("turn_drift_exceeds_inspected_envelope")
            if (
                action["type"] == "turn"
                and abs(math.remainder(snap["pose"]["yaw_rad"] - i["pose"]["yaw_rad"], 2 * math.pi))
                > action["segment"]["required_clearance"]
            ):
                raise Hold("turn_heading_exceeds_inspected_envelope")

        def preflight(snap):
            if self.cancelled.is_set():
                raise Hold("operator_abort")
            capability_check(self.cap, live=self.live, wall_s=self.wall())
            self.check_observation(proposal, snap)
            current_checks(snap, action, self.cap, self.clock(), capture=False)
            during(snap, 0.0)

        result = result_record(ticket, not self.live, self.clock())
        try:
            preflight(self.guards.snapshot())
            if self.live:
                result = self.motion(
                    ticket, action, result, self.cancelled, preflight=preflight, area_guard=during
                )
            else:
                result = self._fake_motion(action, result, area_guard=during)
        except Exception as exc:
            result.update(status="failed", reason=str(exc))
        result["ended_monotonic_s"] = self.clock()
        self.retain_lock = (
            self.live
            and result["started_monotonic_s"] is not None
            and not stop_and_release_verified(result)
        )
        self.emit({"type": "result", "nonce": self.nonce, "result": result})
        self.after = result["ended_monotonic_s"]
        m = result.get("motion_measurement", {})
        row.update(
            status=result["status"],
            reason=result["reason"],
            measured=m.get("settled"),
            termination=m.get("termination"),
            outcome=m.get("outcome"),
            no_progress=m.get("outcome") == "no_measurable_progress",
            stop_verified=m.get("stop_verified"),
        )
        validate_record("execution_result", result, 2)
        if self.cancelled.is_set():
            return self.tick()
        if result["status"] == "failed" or not m.get("stop_verified"):
            return self.finish("SENSOR_OR_CONTROL_FAILURE", result["reason"])
        if any(word in result["reason"].lower() for word in ("operator", "override", "intervention")):
            return self.finish("OPERATOR_ABORT", result["reason"])
        if "episode_action_deadline" in result["reason"] or self.clock() >= g["deadline"]:
            return self.finish("EPISODE_DEADLINE", "episode_or_action_deadline")
        if m.get("termination") not in ("feedback_trigger", "synthetic_feedback_trigger", "blocked_stall"):
            outcome = (
                "BLOCKED_OR_NO_USEFUL_ACTION"
                if any(term in result["reason"] for term in ("clearance", "inspected_area", "obstructed"))
                else "SENSOR_OR_CONTROL_FAILURE"
            )
            return self.finish(outcome, result["reason"])
        if m.get("outcome") in ("measurement_unavailable", "wrong_direction"):
            return self.finish("SENSOR_OR_CONTROL_FAILURE", m["outcome"])
        settled = self.guards.snapshot()["pose"]
        row.update(settled_pose=settled, decision_pose=i["pose"])
        try:
            area_check(g, settled, attempts=slot)
        except Hold as exc:
            return self.finish("BLOCKED_OR_NO_USEFUL_ACTION", str(exc))
        self.state, self.reason = "OBSERVING", "awaiting_view_after_verified_settling"
        self.publish()

    def review(self, message):
        if (
            self.grant is None
            or message.get("episode_id") != self.nonce
            or not self.last
            or message.get("operator") != self.grant["operator"]
            or message.get("prediction_id") != self.last["prediction_id"]
            or message.get("image_sha256") != self.last["image_sha256"]
        ):
            raise Hold("review_for_wrong_episode_image_or_operator")
        if message["type"] == "episode_identified" and self.state == "IDENTIFYING":
            self.after = self.clock()
            self.state, self.reason = "OBSERVING", "new_view_after_identification"
            self.publish()
        elif message["type"] == "episode_goal" and self.arrival_review_available():
            self.previous_outcome, self.outcome = self.outcome, None
            self.found = True
            self.emit(
                {
                    "type": "episode_human_arrival",
                    **{k: message[k] for k in ("episode_id", "operator", "prediction_id", "image_sha256")},
                }
            )
            self.finish("GOAL_CONFIRMED", "operator_confirmed_arrival_on_named_image")
        else:
            raise Hold("review_incompatible_with_episode_state")
