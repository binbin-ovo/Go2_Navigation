"""Plan one supervised episode primitive in the observed body frame. Pure Python, metres/radians."""

import math

from .motion_profile import (
    DISC_RADIUS_M,
    FORWARD_CAP,
    MIN_FORWARD,
    MIN_TURN,
    PROFILE,
    RESPONSE,
    TURN_CAP,
    TURN_DRIFT_M,
    Hold,
    prefix_action,
    wrap,
)
from .supervised import check_clearance


def bounded_number(value, low, high, label):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise Hold("invalid_" + label)
    return value


def area_check(grant, pose, action=None, *, progress=0.0, attempts=0):
    """Contain the swept segment and stopping reserve within the operator's initial odom disc."""
    origin = grant["origin"]
    if (pose["frame"], pose["robot_epoch"]) != (origin["frame"], origin["robot_epoch"]):
        raise Hold("inspection_area_epoch_or_frame_changed")
    dx, dy = pose["x_m"] - origin["x_m"], pose["y_m"] - origin["y_m"]
    radial = math.hypot(dx, dy)
    # Footprint plus observed turn drift; include lateral corridor tolerance and an
    # increasing pilot odometry allowance after every dispatch, including stalls.
    footprint = DISC_RADIUS_M
    uncertainty = 0.15 + 0.05 * attempts
    if action and action["type"] == "forward":
        travel = max(0.0, action["magnitude"] - max(0.0, progress)) + action["segment"]["clearance_reserve"]
        radial = max(
            radial,
            math.hypot(dx + travel * math.cos(pose["yaw_rad"]), dy + travel * math.sin(pose["yaw_rad"])),
        )
    required = radial + footprint + uncertainty
    if not math.isfinite(required) or required > grant["area_radius_m"]:
        raise Hold(
            "outside_inspected_area",
            required_radius_m=required,
            inspected_radius_m=grant["area_radius_m"],
            uncertainty_m=uncertainty,
        )
    return required


def image_bearing(point, camera, mapping):
    """Rectified no-crop stretched image, positive left. Use the measured centre-ray heading."""
    w = bounded_number(camera["width"], 2, 8192, "image_width")
    fx = bounded_number(camera["fx"], 1, 1e5, "rectified_fx")
    cx = bounded_number(camera["cx"], 0, w, "rectified_cx")
    x = bounded_number(point[0], 0, 1, "target_x") * w - 0.5
    centre = (w - 1.0) / 2.0
    return mapping["yaw_rad"] + math.atan2(cx - x, fx) - math.atan2(cx - centre, fx)


def primitive(cap, direction, magnitude, basis):
    forward = direction == "forward"
    unit = "m" if forward else "rad"
    maximum = min(FORWARD_CAP if forward else TURN_CAP, cap["limits"]["forward_m" if forward else "turn_rad"])
    bounded_number(magnitude, MIN_FORWARD if forward else MIN_TURN, maximum, "episode_magnitude")
    lead, reserve, tolerance, noise = RESPONSE[direction]
    return {
        "type": "forward" if forward else "turn",
        "direction": direction,
        "magnitude": magnitude,
        "speed": cap["limits"]["forward_mps" if forward else "turn_radps"],
        "unit": unit,
        "speed_unit": unit + "/s",
        "duration_s": cap["limits"]["duration_s"],
        "segment": {
            "profile": PROFILE,
            "policy": "five-move-v1",
            "basis": basis,
            "stop_lead": lead,
            "stop_trigger": magnitude - lead,
            "clearance_reserve": reserve,
            "required_clearance": magnitude + reserve,
            "noise_floor": noise,
            "tolerance": tolerance,
            "translation_budget_m": 0.0 if forward else TURN_DRIFT_M,
            "sweep_radius_m": 0.0 if forward else DISC_RADIUS_M,
        },
    }


def native_intent(rows):
    """Short contiguous forward/concordant prefix. A later row cannot rescue a reversal."""
    prev_x = prev_y = yaw = extent = 0.0
    for x, y, angle in rows:
        dx, dy = x - prev_x, y - prev_y
        if (
            dx < -0.005
            or x < -0.005
            or abs(y) > 0.08
            or abs(dy) > 0.005 + max(0.0, dx) * math.tan(math.radians(15))
            or (abs(yaw) > 0.005 and (yaw * angle < 0 or abs(angle) < abs(yaw) - 0.02))
        ):
            break
        if abs(angle) > abs(yaw):
            yaw = angle
        extent = max(extent, x)
        prev_x, prev_y = x, y
    return extent, yaw


def supported_ground(grant, proposal, evidence):
    """Bind the semantic verification to this requested target, source image and model point."""
    ident = proposal["identity"]
    if (
        evidence.get("target") != grant["target"]
        or evidence.get("image_sha256") != ident["source_image_sha256"]
        or evidence.get("prediction_id") != ident["prediction_id"]
    ):
        raise Hold("grounding_for_wrong_target_or_image")
    status = evidence["status"]
    if status not in ("absent", "uncertain", "needs_identification", "supported"):
        raise Hold("invalid_grounding_status")
    point = proposal["point_fraction"]
    if status == "absent" and point is not None or status == "supported" and point is None:
        raise Hold("grounding_and_model_point_disagree")
    if status != "supported":
        return False
    roi = evidence["roi"]
    if len(roi) != 4 or not (
        0 <= roi[0] < roi[2] <= 1
        and 0 <= roi[1] < roi[3] <= 1
        and roi[0] <= point[0] <= roi[2]
        and roi[1] <= point[1] <= roi[3]
    ):
        raise Hold("unsupported_target_roi")
    if evidence["method"] == "synthetic_fixture" and proposal["synthetic"]:
        return True
    if evidence["method"] != "identified_instance_features" or not (
        evidence.get("operator") == grant["operator"]
        and len(evidence.get("reference_sha256", "")) == 64
        and evidence.get("inliers", 0) >= 12
        and evidence.get("inlier_fraction", 0) >= 0.8
        and evidence.get("feature_spread", 0) >= 0.12
        and 0 <= evidence.get("median_residual_px", 99) <= 2.0
    ):
        raise Hold("insufficient_instance_evidence")
    return True


def select_action(grant, cap, proposal, evidence, views, attempts, was_aligned):
    ident, rows = proposal["identity"], proposal["waypoints"]
    geo, pose = ident["geometry"], ident["pose"]
    found = supported_ground(grant, proposal, evidence)
    mode = "APPROACH" if found else "SEARCH"
    extent, native_yaw = native_intent(rows)
    selection = {"mode": mode, "found": found, "aligned": False, "bearing_rad": None, "action": None}
    if proposal["native_stop"]:
        return dict(selection, reason="native_stop", result="MODEL_STOPPED_ARRIVAL_UNVERIFIED")
    if evidence["status"] == "needs_identification":
        return dict(selection, reason="identify_or_reject_current_mark", review=True)

    def checked(action, source, reason):
        check_clearance(geo, pose, action, cap)
        area_check(grant, pose, action, attempts=attempts + 1)
        return dict(selection, action=action, source=source, reason=reason)

    if found:
        bearing = image_bearing(proposal["point_fraction"], evidence["camera"], cap["mapping"])
        band = math.radians(grant["alignment_deg"])
        aligned = abs(bearing) <= band + (math.radians(3) if was_aligned else 0.0)
        selection.update(bearing_rad=bearing, aligned=aligned)
        if aligned:
            try:
                native = prefix_action(rows, proposal["point_fraction"], cap, geo["forward_clear_m"])
            except Hold:
                native = None
            if native and native["type"] == "forward":
                return checked(native, "native_prefix", "grounded_target_ahead")
            # The proposed translation must support an approach. Geometry independently
            # supports a short straight corridor. Curved/reversing suffixes add no extent.
            if extent < MIN_FORWARD or abs(native_yaw) > MIN_TURN:
                raise Hold("native_path_does_not_support_forward_approach")
            return checked(
                primitive(cap, "forward", MIN_FORWARD, "short_observed_corridor"),
                "approach_policy",
                "grounded_within_alignment_band",
            )
        side = 1 if bearing > 0 else -1
        if side * native_yaw < -0.10:
            raise Hold("visible_target_and_native_heading_disagree")
        angle = min(TURN_CAP, max(MIN_TURN, abs(bearing)))
        residual = abs(abs(bearing) - angle)
        if angle > abs(native_yaw) + 1e-6 and residual + cap["mapping"]["yaw_uncertainty_rad"] > band:
            raise Hold("coarse_alignment_residual_too_large")
        selection.update(native_prefix_yaw_rad=native_yaw, expected_residual_rad=residual)
        direction = "left" if side > 0 else "right"
        return checked(
            primitive(cap, direction, angle, "coarse_target_alignment"),
            "approach_policy",
            "fresh_view_required_after_alignment",
        )

    # Inspect new headings at the current location. Measured poses, including stalls,
    # determine where we actually looked. A forward viewpoint is also a distinct move.
    preferred = "right" if native_yaw < -0.10 else "left"
    recent_search = [v["attempt"] for v in views if v["attempt"]["mode"] == "SEARCH"]
    if recent_search and abs(native_yaw) <= 0.10 and recent_search[-1]["direction"] in ("left", "right"):
        preferred = recent_search[-1]["direction"]
    opposite = "right" if preferred == "left" else "left"
    choices = [preferred, opposite, "forward"]
    if len(recent_search) >= 2 and extent >= MIN_FORWARD and recent_search[-1]["direction"] != "forward":
        choices = ["forward", preferred, opposite]
    reasons = []
    for direction in choices:
        forward = direction == "forward"
        # Attribute a search turn to the model only for an in-place native
        # prefix with coherent signed yaw. A translated viewpoint is policy-selected.
        supported_turn = (
            not forward
            and direction == preferred
            and MIN_TURN <= abs(native_yaw) <= TURN_CAP
            and all(math.hypot(x, y) <= 0.025 and ang * native_yaw >= 0.0 for x, y, ang in rows)
        )
        amount = MIN_FORWARD if forward else (abs(native_yaw) if supported_turn else TURN_CAP)
        predicted_heading = pose["yaw_rad"] + (amount if direction == "left" else -amount)
        if not forward and any(
            math.hypot(v["x_m"] - pose["x_m"], v["y_m"] - pose["y_m"]) < 0.25
            and abs(wrap(v["yaw_rad"] - predicted_heading)) < math.radians(12)
            for v in views
        ):
            reasons.append("heading_already_inspected")
            continue
        if (
            recent_search
            and recent_search[-1]["direction"] == direction
            and recent_search[-1].get("no_progress")
        ):
            reasons.append("previous_attempt_made_no_progress")
            continue
        try:
            return checked(
                primitive(cap, direction, amount, "observed_search_viewpoint"),
                "native_search_prefix" if supported_turn else "search_policy",
                "inspect_new_viewpoint",
            )
        except Hold as exc:
            reasons.append(str(exc))
    raise Hold("no_supported_search_viewpoint: " + ", ".join(reasons))
