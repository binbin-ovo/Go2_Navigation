"""One reviewed primitive, one refresh, one execution. No SDK/model imports here."""
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re
import secrets

from .motion_profile import (DISC_RADIUS_M, INSPECTED, INSPECTION_TTL_S, Hold,
                             mapped_pose, mapping_slack, prefix_action, wrap)

FRESHNESS_S = 3.0  # unchanged advisory limit; uncertainty is included, never guessed
CONTROL_REVISION = "6d555d31c6eaa4ee194920925ab96238bf15b128"
MODEL_REVISION = "826dc5fbfa37afa8293d2e336d329b6ffc0bfb64"
EVIDENCE = ("frame_mapping", "intervention", "obstacle_handling", "stop_verification",
            "motion_bounds", "target_correspondence")
SCHEMA = Path(__file__).with_name("schemas") / "supervised-navigation-v1.schema.json"


def contract_version(cap):
    return 2 if "motion_profile" in cap else 1


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


@lru_cache(maxsize=2)
def _schema(version):
    return json.loads(SCHEMA.with_name(f"supervised-navigation-v{version}.schema.json").read_text())


def validate_record(kind, value, version=1):
    # Validate the deliberately small schema vocabulary using only stdlib in the
    # hardware child. The published schema is additionally checked by jsonschema tests.
    schema = _schema(version)

    def check(spec, item, path):
        if "$ref" in spec:
            return check(schema["$defs"][spec["$ref"].split("/")[-1]], item, path)
        if "anyOf" in spec:
            for branch in spec["anyOf"]:
                try:
                    check(branch, item, path)
                    return
                except Hold:
                    pass
            raise Hold("invalid_" + path)
        if "const" in spec and item != spec["const"] or "enum" in spec and item not in spec["enum"]:
            raise Hold("invalid_" + path)
        typ = spec.get("type")
        matches = {"object": isinstance(item, dict), "array": isinstance(item, list),
                   "number": type(item) in (int, float), "integer": type(item) is int,
                   "string": isinstance(item, str), "boolean": type(item) is bool, "null": item is None}
        if typ and not matches[typ]:
            raise Hold("invalid_type_" + path)
        if typ == "object":
            if any(k not in item for k in spec.get("required", [])):
                raise Hold("missing_fields_" + path)
            if spec.get("additionalProperties") is False and set(item) - spec["properties"].keys():
                raise Hold("unknown_fields_" + path)
            for k, sub in spec.get("properties", {}).items():
                if k in item:
                    check(sub, item[k], path + "." + k)
        elif typ == "array":
            if not spec.get("minItems", 0) <= len(item) <= spec.get("maxItems", math.inf):
                raise Hold("invalid_length_" + path)
            for sub in item:
                check(spec["items"], sub, path + "[]")
        elif typ in ("number", "integer"):
            if not math.isfinite(item) or not spec.get("minimum", -math.inf) <= item <= spec.get("maximum", math.inf):
                raise Hold("invalid_number_" + path)
        elif typ == "string" and (len(item) < spec.get("minLength", 0) or
                                  "pattern" in spec and not re.fullmatch(spec["pattern"], item)):
            raise Hold("invalid_string_" + path)

    check(schema["$defs"][kind], value, kind)
    if version == 2 and kind == "execution_result" and value["status"] in ("completed", "ended"):
        measured = value.get("motion_measurement")
        zeros = value["stop_send_statuses"]
        if (not measured or not measured["stop_verified"] or value["started_monotonic_s"] is None or
                value["release_rpc_code"] != 0 or len(zeros) < 3 or any(code != 0 for code in zeros) or
                not all(value[key] is True for key in ("observed_stationary", "ownership_released",
                                                       "post_release_stationary"))):
            raise Hold("settled_measurement_and_verified_release_required")
        if value["status"] == "completed" and (measured["outcome"] != "target_within_tolerance" or
                                               measured["interrupted_or_blocked"]):
            raise Hold("completion_inconsistent_with_settled_motion")


def age_check(stamp, now, limit, label):
    if stamp is None or not math.isfinite(stamp) or not 0 <= now - stamp <= limit:
        raise Hold(f"stale_or_future_{label}", predicate="age_outside_window", field=label,
                   sample_monotonic_s=stamp, checked_monotonic_s=now, maximum_age_s=limit)


def capability_check(cap, *, live, wall_s):
    version = contract_version(cap)
    validate_record("capability", cap, version)
    if cap["kind"] != ("hardware" if live else "synthetic"):
        raise Hold("synthetic_hardware_mixing")
    if wall_s >= cap["valid_until_wall_s"]:
        raise Hold("expired_capability")
    for key in EVIDENCE:
        if cap["evidence"][key]["status"] != "verified":
            raise Hold("unverified_" + key)
    if cap["mapping"] is None:
        raise Hold("missing_frame_mapping")
    if cap["mapping"]["translation_m"] is None:
        # An unregistered model origin is usable only where it cancels to a bounded residual.
        if "turn" in cap["mapping"]["verified_actions"]:
            raise Hold("turn_requires_registered_mapping_translation")
        if "translation_bound_m" not in cap["mapping"]:
            raise Hold("unknown_mapping_translation_without_bound")
    limits = cap["limits"]
    if limits is None or any(not math.isfinite(v) or v <= 0 for v in limits.values()):
        raise Hold("missing_or_invalid_physical_limits")
    # These are the audited utility's implementation ceilings, NOT verified safe limits.
    if ((version == 1 and (limits["forward_mps"] > .30 or limits["turn_radps"] > 1)) or
            limits["forward_m"] > 1.6 or limits["turn_rad"] > 1.6 or
            not 2 <= limits["duration_s"] <= 8 or
            not .05 <= limits["resend_period_s"] <= .5 or
            limits["state_age_s"] > .25 or limits["geometry_age_s"] > .5 or
            limits["auth_ttl_s"] > 15 or limits["stationary_dwell_s"] > 2):
        raise Hold("capability_exceeds_audited_implementation")
    if live and version == 2:
        mapping = cap["mapping"]
        if (mapping["translation_m"] is not None or mapping.get("translation_bound_m") != .5 or
                mapping["verified_actions"] != ["forward"] or mapping["yaw_rad"] != .02941 or
                mapping["yaw_uncertainty_rad"] != .05236):
            raise Hold("effective_profile_requires_recorded_forward_mapping")
        if limits["stationary_dwell_s"] < 1.:
            raise Hold("effective_profile_requires_measured_stationary_dwell")
    if live:
        control = cap["control"]
        if control is None or control["revision"] != CONTROL_REVISION:
            raise Hold("missing_or_unpinned_control")
    return limits


def plan(proposal, cap):
    validate_record("proposal", proposal, contract_version(cap))
    if proposal["native_stop"]:
        raise Hold("native_stop_no_motion")
    if proposal["point_fraction"] is None or any(not 0 <= x <= 1 for x in proposal["point_fraction"]):
        raise Hold("no_visible_target_point")
    m, lim = cap["mapping"], cap["limits"]
    if m is None or lim is None:
        raise Hold("missing_frame_mapping_or_limits")
    if contract_version(cap) == 2:
        validate_record("capability", cap, 2)
        ident = proposal["identity"]
        geo, pose = ident["geometry"], ident["pose"]
        if (abs(geo["monotonic_s"]-ident["observed_monotonic_s"]) > lim["geometry_age_s"] or
                abs(pose["monotonic_s"]-ident["observed_monotonic_s"]) > lim["state_age_s"]):
            raise Hold("decision_geometry_or_pose_not_bound_to_observation")
        action = prefix_action(proposal["waypoints"], proposal["point_fraction"], cap, geo["forward_clear_m"])
        check_clearance(geo, pose, action, cap)
        return action
    for i, row in enumerate(proposal["waypoints"]):
        # Skip native near-identity poses, never just interpret the first as STOP.
        if math.hypot(*row[:2]) <= .005 and abs(wrap(row[2])) <= .005:
            continue
        x, y, yaw = mapped_pose(row, m)
        slack = mapping_slack(row, m)
        residual = math.hypot(x, y) + slack
        if x - slack > .005 and abs(y) + slack <= lim["lateral_m"] and abs(yaw) <= lim["straight_yaw_rad"]:
            kind, direction, magnitude, speed, unit = "forward", "forward", min(x - slack, lim["forward_m"]), lim["forward_mps"], "m"
            residual, residual_yaw = abs(y) + slack, abs(yaw)
        elif abs(yaw) > .005 and residual <= lim["turn_translation_m"]:
            kind, direction, magnitude, speed, unit = "turn", "left" if yaw > 0 else "right", min(abs(yaw), lim["turn_rad"]), lim["turn_radps"], "rad"
            residual_yaw = 0.
        else:
            raise Hold("unsupported_reverse_lateral_or_combined_primitive")
        if kind not in m["verified_actions"]:
            raise Hold("mapping_not_verified_for_" + kind)
        return {"type": kind, "direction": direction, "magnitude": magnitude, "unit": unit,
                "speed": speed, "speed_unit": unit + "/s", "duration_s": lim["duration_s"],
                "selected_row": i, "mapped_pose": [x, y, yaw],
                "residual_translation_m": residual, "residual_yaw_rad": residual_yaw}
    raise Hold("near_zero_chunk_no_motion")


def pose_check(reference, current, limits):
    validate_record("pose", reference)
    validate_record("pose", current)
    if (reference["frame"], reference["robot_epoch"]) != (current["frame"], current["robot_epoch"]):
        raise Hold("changed_pose_frame_or_robot_epoch")
    if (math.hypot(current["x_m"] - reference["x_m"], current["y_m"] - reference["y_m"]) > limits["pose_translation_m"] or
            abs(wrap(current["yaw_rad"] - reference["yaw_rad"])) > limits["pose_yaw_rad"]):
        raise Hold("material_pose_change")


def check_clearance(geo, pose, action, cap, *, progress=0., age=0.):
    if not geo["available"]:
        raise Hold("required_geometry_unavailable_or_insufficient", predicate="geometry_unavailable")
    if (geo["frame"], geo["robot_epoch"]) != (pose["frame"], pose["robot_epoch"]):
        raise Hold("required_geometry_unavailable_or_insufficient", predicate="geometry_frame_or_epoch_mismatch")
    segment = action.get("segment", {})
    if action["type"] == "turn" and cap.get("turn_scope") == INSPECTED:
        # The ordinary forward guard must still be usable. Observed returns can
        # veto the turn. The rear and cable require a separate per-preview inspection.
        disc = geo.get("observed_turn_disc", {})
        if ("forward" not in geo["directions"] or geo["forward_clear_m"] < segment["translation_budget_m"] or
                disc.get("radius_m", 0) < DISC_RADIUS_M or disc.get("floor_returns", 0) < 2 or
                disc.get("obstacle_returns") != 0):
            raise Hold("observed_turn_disc_unavailable_or_obstructed", field="observed_turn_disc",
                       required_radius_m=DISC_RADIUS_M, min_floor_returns=2, max_obstacle_returns=0,
                       forward_clearance_required_m=segment["translation_budget_m"])
        if set(geo["directions"]) & {"left", "right"} and (
                action["direction"] not in geo["directions"] or
                geo["turn_clear_rad"] < segment["required_clearance"]+action["speed"]*age):
            raise Hold("measured_turn_sector_restricts_inspected_area", direction=action["direction"],
                       available_rad=geo["turn_clear_rad"], required_rad=segment["required_clearance"]+action["speed"]*age)
        return
    remain = max(0., action["magnitude"]-max(0., progress))
    required = remain + segment.get("clearance_reserve", 0.)
    if segment:
        required += age*action["speed"]
    key = "forward_clear_m" if action["type"] == "forward" else "turn_clear_rad"
    if action["direction"] not in geo["directions"] or required > geo[key]+1e-9:
        predicate = "direction_unavailable" if action["direction"] not in geo["directions"] else "insufficient_clearance"
        raise Hold("required_geometry_unavailable_or_insufficient", predicate=predicate, field=key,
                   available=geo[key], required=required, remaining=remain,
                   reserve=segment.get("clearance_reserve", 0.), geometry_age_s=age)


def check_inspection(auth, action, cap, now, pose, *, moving=False):
    if action["type"] != "turn" or cap.get("turn_scope") != INSPECTED:
        return
    i, old = auth.get("turn_inspection"), auth["reviewed_identity"]
    if (not i or i["preview_sha256"] != digest(old) or i["session_nonce"] != auth["session_nonce"] or
            i["direction"] != action["direction"] or i["operator"] != auth["target"]["confirmed_by"] or
            i["radius_m"] != DISC_RADIUS_M or i["scope"] != INSPECTED or
            not i["inspected_s"] <= auth["created_monotonic_s"] <= now < i["inspected_s"] + INSPECTION_TTL_S):
        raise Hold("fresh_turn_area_and_cable_inspection_required")
    origin = i["pose"]
    age_check(origin["monotonic_s"], i["inspected_s"], cap["limits"]["state_age_s"], "inspection_pose")
    pose_check(old["pose"], origin, cap["limits"])
    if not moving:
        pose_check(origin, pose, cap["limits"])
    elif ((origin["frame"], origin["robot_epoch"]) != (pose["frame"], pose["robot_epoch"]) or
          math.hypot(pose["x_m"]-origin["x_m"], pose["y_m"]-origin["y_m"]) >
          action["segment"]["translation_budget_m"] or
          abs(wrap(pose["yaw_rad"]-origin["yaw_rad"])) > action["segment"]["required_clearance"]):
        raise Hold("inspected_turn_drift_or_heading_bound_exceeded")


def current_checks(identity, action, cap, now, *, capture=True, progress=0.):
    lim = cap["limits"]
    if capture:
        bound = identity["uncertainty_bound_s"]
        if bound is None:
            raise Hold("unknown_capture_age_bound")
        age_check(identity["observed_monotonic_s"], now, FRESHNESS_S - bound, "observation")
        age_check(identity["received_monotonic_s"], now, FRESHNESS_S, "receipt")
    pose, geo = identity["pose"], identity["geometry"]
    validate_record("pose", pose)
    validate_record("geometry", geo, contract_version(cap))
    age_check(pose["monotonic_s"], now, lim["state_age_s"], "robot_state")
    age_check(geo["monotonic_s"], now, lim["geometry_age_s"], "geometry")
    check_clearance(geo, pose, action, cap, progress=progress, age=now-geo["monotonic_s"])


def revalidate(authorization, proposal, cap, *, now, wall_s, nonce, live, current=None):
    """Independent checks repeated by executor before any command-source claim."""
    validate_record("authorization", authorization, contract_version(cap))
    validate_record("proposal", proposal, contract_version(cap))
    limits = capability_check(cap, live=live, wall_s=wall_s)
    a, fresh = authorization, proposal["identity"]
    if a["session_nonce"] != nonce or a["capability_sha256"] != digest(cap) or not a["consumed"]:
        raise Hold("wrong_session_capability_or_unconsumed_authorization")
    if not a["created_monotonic_s"] <= now < a["expires_monotonic_s"] <= a["created_monotonic_s"] + limits["auth_ttl_s"]:
        raise Hold("expired_or_invalid_authorization")
    old = a["reviewed_identity"]
    for field in ("session_id", "instruction_id", "instruction", "clock_epoch"):
        if old[field] != fresh[field]:
            raise Hold("changed_" + field)
    for field in ("observation_id", "prediction_id"):
        if old[field] == fresh[field]:
            raise Hold("reused_" + field)
    # Equal pixel content is possible when stationary; source identities/times must be NEW.
    if fresh["observed_monotonic_s"] <= a["created_monotonic_s"]:
        raise Hold("refresh_not_acquired_after_authorization")
    if proposal["synthetic"] == live:
        raise Hold("synthetic_hardware_mixing")
    if live and (proposal["model_identity"]["revision"] != MODEL_REVISION or
                 proposal["model_identity"]["processor_revision"] != MODEL_REVISION):
        raise Hold("changed_model_revision")
    pose_check(old["pose"], fresh["pose"], limits)
    action = plan(proposal, cap)
    for field in ("type", "direction", "unit", "speed_unit"):
        if action[field] != a["envelope"][field]:
            raise Hold("changed_action_" + field)
    if "segment" in action:
        for field in ("profile", "policy", "basis", "stop_lead", "clearance_reserve", "translation_budget_m"):
            if action["segment"][field] != a["envelope"]["segment"][field]:
                raise Hold("changed_execution_policy_"+field)
    for field in ("magnitude", "speed", "duration_s"):
        if action[field] > a["envelope"][field] + 1e-9:
            raise Hold("action_outside_authorized_" + field)
    bound = fresh["uncertainty_bound_s"]
    if bound is None:
        raise Hold("unknown_capture_age_bound")
    age_check(fresh["observed_monotonic_s"], now, FRESHNESS_S - bound, "observation")
    age_check(fresh["received_monotonic_s"], now, FRESHNESS_S, "receipt")
    if abs(fresh["pose"]["monotonic_s"] - fresh["observed_monotonic_s"]) > limits["state_age_s"]:
        raise Hold("pose_not_bound_to_observation")
    if contract_version(cap) == 2 and live and fresh["observed_monotonic_s"]-bound <= a["created_monotonic_s"]:
        raise Hold("refresh_capture_interval_overlaps_authorization")
    if current is None:
        raise Hold("fresh_executor_guards_unavailable")
    if live and contract_version(cap) == 2 and (
            current.get("capture_clock_epoch") != fresh["clock_epoch"] or not current.get("capture_bound_basis") or
            current.get("capture_uncertainty_bound_s") != bound):
        raise Hold("capture_bound_or_camera_epoch_changed_before_dispatch")
    pose_check(fresh["pose"], current["pose"], limits)
    current_checks(current, action, cap, now, capture=False)
    check_inspection(a, action, cap, now, current["pose"])
    return action


class Cycle:
    """Owner-thread state machine. Authorization is burned at refresh, never reused."""

    def __init__(self, cap, *, live=False, nonce=None, event=lambda *a, **k: None):
        self.cap, self.live = deepcopy(cap), live
        self.nonce = nonce or secrets.token_hex(24)
        self.event = event
        self.state, self.reason = "IDLE", "Choose a target and obtain a proposal."
        self.reviewed = self.target = self.action = self.authorization = self.result = self.post = None
        self.request_id = None
        self.inspection = self.replan = self.feedback = self.last_proposal = None
        self.dispatched = None  # private immutable audit record, never reusable authorization
        self.used = set()
        self.locked = False

    def hold(self, reason):
        self.state, self.reason = "HOLD", str(reason)
        self.event("navigation_hold", reason=self.reason)

    def preview(self, proposal, target, *, now, wall_s):
        if self.locked or self.dispatched or self.state in ("EXECUTING", "STOPPING", "REFRESHING", "AWAITING_OBSERVATION"):
            raise Hold("cycle_busy_or_requires_intervention")
        self.reviewed = self.target = self.action = self.inspection = self.authorization = None
        validate_record("proposal", proposal, contract_version(self.cap))
        self.last_proposal = deepcopy(proposal)
        capability_check(self.cap, live=self.live, wall_s=wall_s)
        validate_record("target", target)
        if target["reviewed_image_sha256"] != proposal["identity"]["source_image_sha256"]:
            raise Hold("reviewed_target_image_mismatch")
        roi = target["roi_xyxy"]
        if not (0 <= roi[0] < roi[2] <= 1 and 0 <= roi[1] < roi[3] <= 1):
            raise Hold("invalid_target_roi")
        if proposal["native_stop"] or proposal["point_fraction"] is None:
            plan(proposal, self.cap)  # retain the precise STOP/absent refusal
        point = proposal["point_fraction"]
        if point is None or not (roi[0] <= point[0] <= roi[2] and roi[1] <= point[1] <= roi[3]):
            raise Hold("model_point_outside_reviewed_target")
        self.action = plan(proposal, self.cap)
        self.reviewed, self.target = deepcopy(proposal), deepcopy(target)
        self.authorization = self.result = self.post = None
        self.state, self.reason = "REVIEW", "Preview may age. Authorization requires ONE fresh re-evaluation."
        self.event("navigation_preview", proposal=proposal, target=target, action=self.action, at=now)
        return deepcopy(self.action)

    def inspect_turn(self, *, operator, now, current):
        if (self.state != "REVIEW" or self.cap.get("turn_scope") != INSPECTED or
                self.action["type"] != "turn" or operator != self.target["confirmed_by"]):
            raise Hold("inspection_requires_matching_turn_preview_and_operator")
        current_checks(current, self.action, self.cap, now, capture=False)
        identity = self.reviewed["identity"]
        pose_check(identity["pose"], current["pose"], self.cap["limits"])
        self.inspection = {"scope": INSPECTED, "session_nonce": self.nonce, "preview_sha256": digest(identity),
                           "direction": self.action["direction"], "radius_m": DISC_RADIUS_M, "operator": operator,
                           "pose": deepcopy(current["pose"]), "inspected_s": now}
        self.event("navigation_turn_disc_and_cable_inspected", inspection=self.inspection)

    def authorize(self, *, now, wall_s):
        if self.state != "REVIEW":
            raise Hold("no_reviewable_preview_or_duplicate_authorization")
        limits = capability_check(self.cap, live=self.live, wall_s=wall_s)
        self.authorization = {"authorization_id": secrets.token_hex(16), "session_nonce": self.nonce,
                              "created_monotonic_s": now, "expires_monotonic_s": now + limits["auth_ttl_s"],
                              "capability_sha256": digest(self.cap),
                              "reviewed_identity": deepcopy(self.reviewed["identity"]),
                              "target": deepcopy(self.target), "envelope": deepcopy(self.action), "consumed": True}
        if self.inspection:
            self.authorization["turn_inspection"] = deepcopy(self.inspection)
        check_inspection(self.authorization, self.action, self.cap, now, self.reviewed["identity"]["pose"])
        self.state, self.reason = "REFRESHING", "Authorization consumed; collecting new observations for ONE request."
        self.event("navigation_authorized_once", authorization=self.authorization)
        return deepcopy(self.authorization)

    def refreshed(self, proposal, correspondence, *, now, wall_s, current):
        if self.state != "REFRESHING":
            raise Hold("unsolicited_or_duplicate_refresh")
        try:
            action = revalidate(self.authorization, proposal, self.cap, now=now, wall_s=wall_s,
                                nonce=self.nonce, live=self.live, current=current)
            expected = {"reviewed_image_sha256": self.target["reviewed_image_sha256"],
                        "fresh_image_sha256": proposal["identity"]["source_image_sha256"],
                        "instance_id": self.target["instance_id"]}
            if (not correspondence.get("valid") or correspondence.get("method") != "local_feature_roi_v1" or
                    any(correspondence.get(k) != v for k, v in expected.items())):
                raise Hold("changed_or_unverified_physical_target")
            self.request_id = secrets.token_hex(16)
            self.execution_identity = deepcopy(proposal["identity"])
            ticket = {"request_id": self.request_id, "nonce": self.nonce,
                      "authorization": deepcopy(self.authorization), "proposal": deepcopy(proposal),
                      "correspondence": deepcopy(correspondence), "action": action}
            self.dispatched = deepcopy(ticket)
            self.used.add(self.authorization["authorization_id"])
            self.state, self.reason = "EXECUTING", "One bounded action dispatched; cancellation now requests STOP."
            self.event("navigation_dispatch", ticket=ticket)
            return ticket
        except (ValueError, KeyError, TypeError) as error:
            self.hold(error)
            raise

    def cancel(self):
        if self.state in ("COMPLETE", "HOLD", "AWAITING_OBSERVATION"):
            return "cancel"  # closing the UI must not erase a result/failed-stop diagnosis
        if self.state in ("EXECUTING", "STOPPING"):
            self.state, self.reason = "STOPPING", "Stop requested; physical stop is not yet verified."
            return "stop"
        self.authorization = None
        self.hold("pending_cycle_cancelled_no_command")
        return "cancel"

    def completed(self, result):
        validate_record("execution_result", result, contract_version(self.cap))
        if self.state not in ("EXECUTING", "STOPPING") or result["request_id"] != self.request_id or result["authorization_id"] != self.authorization["authorization_id"]:
            raise Hold("wrong_or_duplicate_executor_result")
        self.result = deepcopy(result)
        safe = (result["observed_stationary"] is True and result["ownership_released"] is True and
                result["post_release_stationary"] is True)
        self.event("navigation_execution_result", result=result)
        self.reviewed = self.target = self.action = self.authorization = None
        if result["started_monotonic_s"] is None:
            self.locked = True
            self.hold("executor_refused_before_motion: " + result["reason"])
        elif not safe or result["status"] == "failed":
            self.locked = True
            self.hold("stop_or_release_unverified_requires_independent_intervention: " + result["reason"])
        else:
            self.state, self.reason = "AWAITING_OBSERVATION", "Stopped/released; waiting for a strictly newer observation."

    def observe_after(self, identity):
        if self.state != "AWAITING_OBSERVATION":
            raise Hold("not_awaiting_post_motion_observation")
        validate_record("identity", identity, contract_version(self.cap))
        if identity["observed_monotonic_s"] <= self.result["ended_monotonic_s"]:
            raise Hold("post_motion_observation_not_new")
        if identity["clock_epoch"] != self.execution_identity["clock_epoch"] or (identity["pose"]["frame"], identity["pose"]["robot_epoch"]) != (self.execution_identity["pose"]["frame"], self.execution_identity["pose"]["robot_epoch"]):
            raise Hold("post_motion_epoch_or_frame_changed")
        if identity["observation_id"] == self.execution_identity["observation_id"]:
            raise Hold("reused_post_motion_observation")
        self.post = {"identity": deepcopy(identity), "after_request_id": self.request_id}
        self.state, self.reason = "COMPLETE", "One cycle ended. Another movement needs a new preview and authorization."
        self.locked = True  # the isolated executor has spent its one dispatch
        if self.dispatched["action"].get("segment", {}).get("basis") == "yaw_only_stop_observe_replan":
            self.state, self.reason = "REPLANNING", "Reading the stopped view for a single advisory proposal."
        measurement = self.result.get("motion_measurement")
        if measurement:
            self.reason += " Settled outcome: "+measurement["outcome"]+"."
        self.event("navigation_post_observation", post=self.post)

    def record_replan(self, proposal):
        if self.state != "REPLANNING" or self.replan is not None:
            raise Hold("unsolicited_post_turn_replan")
        ident, old = proposal["identity"], self.execution_identity
        for field in ("session_id", "instruction_id", "instruction", "clock_epoch"):
            if ident[field] != old[field]:
                raise Hold("post_turn_changed_"+field)
        if (ident["prediction_id"] == old["prediction_id"] or ident["observation_id"] == old["observation_id"] or
                ident["observed_monotonic_s"] <= self.result["ended_monotonic_s"]):
            raise Hold("post_turn_replan_reused_source")
        pose_check(self.post["identity"]["pose"], ident["pose"], self.cap["limits"])
        try:
            candidate, refusal = plan(proposal, self.cap), None
        except Hold as error:
            candidate, refusal = None, str(error)
        self.replan = {"proposal": deepcopy(proposal), "candidate": candidate, "refusal": refusal,
                       "execution_authorized": False}
        self.state, self.reason = "COMPLETE", "Turn cycle ended. The new view has an advisory result."
        self.event("navigation_post_turn_advisory", replan=self.replan)

    def public(self, now):
        authorization = None
        if self.authorization:
            authorization = {k: v for k, v in self.authorization.items() if k != "session_nonce"}
            if "turn_inspection" in authorization:
                authorization["turn_inspection"] = {k: v for k, v in authorization["turn_inspection"].items()
                                                    if k != "session_nonce"}
        return {"state": self.state, "reason": self.reason, "live": self.live, "locked": self.locked,
                "motion_profile": self.cap.get("motion_profile", "legacy-first-row"), "rates": {
                    k: self.cap["limits"][k] for k in ("forward_mps", "turn_radps")},
                "turn_scope": self.cap.get("turn_scope"), "turn_inspected": self.inspection is not None,
                "action": self.dispatched["action"] if self.dispatched else self.action,
                "native_waypoints": (self.dispatched["proposal"] if self.dispatched else self.last_proposal or {}).get("waypoints"),
                "feedback": self.feedback, "replan": self.replan, "target": self.target, "result": self.result, "post_observation": self.post,
                "preview_age_s": None if self.reviewed is None else now - self.reviewed["identity"]["observed_monotonic_s"],
                "authorization": authorization}
