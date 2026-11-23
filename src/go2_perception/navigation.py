"""Plan one supervised episode primitive in the observed body frame. Pure Python, metres/radians."""

import math
import re


from pathlib import Path
import sys

# The root-level link also supports a Python prompt in this checkout.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from go2_perception.vln.motion_profile import (
    DISC_RADIUS_M,
    FORWARD_CAP,
    INSPECTED,
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
from go2_perception.vln.supervised import check_clearance


TARGET_TEMPLATE_ID = "target_go_to_and_stop_v1"
TARGET_TEMPLATE = "Go to the {target} and stop."


def expand_target(text):
    """Documented template expansion; the operator's words stay a request, not a label."""
    target = " ".join(str(text).split())
    if not target or len(target) > 200:
        raise ValueError("Target text must be 1..200 characters")
    return {
        "kind": "target",
        "template_id": TARGET_TEMPLATE_ID,
        "template": TARGET_TEMPLATE,
        "operator_text": target,
        "instruction": TARGET_TEMPLATE.format(target=target),
    }


def as_instruction(text):
    instruction = " ".join(str(text).split())
    if not instruction or len(instruction) > 4096:
        raise ValueError("Instruction must be 1..4096 characters")
    return {
        "kind": "instruction",
        "template_id": None,
        "template": None,
        "operator_text": instruction,
        "instruction": instruction,
    }


# Bounds enforced independently by the executor. The public defaults match the episode CLI.
DEFAULT_MOVES = 5
DEFAULT_SECONDS = 60.0
DEFAULT_RADIUS_M = 2.0
DEFAULT_ALIGNMENT_DEG = 15.0


def target_spec(text):
    """Keep the full requested description and the original operator text."""

    if (
        not isinstance(text, str)
        or not text.strip()
        or text.strip().rstrip(".!").casefold() in ("go to", "go to the", "the")
    ):
        raise ValueError("Enter an object description")
    noun = re.sub(r"^go\s+to\s+(?:the\s+)?", "", text.strip(), flags=re.I)
    noun = re.sub(r"(?:\s+and\s+stop)?[.!]?$", "", noun, flags=re.I)
    noun = re.sub(r"^the\s+", "", noun.strip(), flags=re.I)
    return {**expand_target(noun), "entered_target_text": text}


def same_view(old, x, y, yaw, pose):
    """A local viewpoint in the same odometry frame and boot epoch."""
    return (
        (old["frame"], old["robot_epoch"]) == (pose["frame"], pose["robot_epoch"])
        and math.hypot(old["x_m"] - x, old["y_m"] - y) < 0.25
        and abs(wrap(old["yaw_rad"] - yaw)) < math.radians(12)
    )


def select_action(grant, cap, proposal, evidence, views, attempts, was_aligned, *, inspected=None):
    """Choose HOLD/review, SEARCH or APPROACH from a bound image, native path and fresh geometry.

    The result is a selection. The separate executor grants or rejects each dispatch.
    """
    ident, rows = proposal["identity"], proposal["waypoints"]
    geo, pose = ident["geometry"], ident["pose"]
    found = supported_ground(grant, proposal, evidence)
    mode = "APPROACH" if found else "SEARCH"
    extent, native_yaw = native_intent(rows)
    selection = {
        "mode": mode,
        "found": found,
        "aligned": False,
        "bearing_rad": None,
        "action": None,
    }
    if proposal["native_stop"]:
        return dict(
            selection, reason="native_stop", result="MODEL_STOPPED_ARRIVAL_UNVERIFIED"
        )
    if evidence["status"] == "needs_identification":
        return dict(selection, reason="identify_or_reject_current_mark", review=True)

    def checked(action, source, reason):
        if inspected is not None and action["type"] == "turn" and cap.get("turn_scope") == INSPECTED:
            if (
                grant.get("area_inspected") is not True
                or grant.get("exclusive_and_intervention_confirmed") is not True
            ):
                raise Hold("inspected_turn_requires_current_episode_authorization")
        try:
            check_clearance(geo, pose, action, cap)
        except Hold as exc:
            if inspected is None or action["type"] != "turn":
                raise
            disc = geo.get("observed_turn_disc") or {}
            raise Hold(
                f"{action['direction']} {exc}: floor={disc.get('floor_returns')}, "
                f"obstacles={disc.get('obstacle_returns')}, sectors={geo.get('directions')}, "
                f"turn_clear_rad={geo.get('turn_clear_rad')}, requested={math.degrees(action['magnitude']):.1f}deg",
                proposed_action=action,
                mode=mode,
                clearance=exc.details,
            ) from exc
        area_check(grant, pose, action, attempts=attempts + 1)
        return dict(selection, action=action, source=source, reason=reason)

    if found:
        bearing = image_bearing(
            proposal["point_fraction"], evidence["camera"], cap["mapping"]
        )
        band = math.radians(grant["alignment_deg"])
        aligned = abs(bearing) <= band + (math.radians(3) if was_aligned else 0.0)
        selection.update(bearing_rad=bearing, aligned=aligned)
        if aligned:
            try:
                native = prefix_action(
                    rows, proposal["point_fraction"], cap, geo["forward_clear_m"]
                )
            except Hold:
                native = None
            if native and native["type"] == "forward":
                return checked(native, "native_prefix", "grounded_target_ahead")
            if inspected is not None:
                if straight_corridor_prefix(rows, cap["mapping"], band):
                    return checked(
                        primitive(cap, "forward", MIN_FORWARD, "short_observed_corridor"),
                        "approach_policy",
                        "grounded_within_alignment_band",
                    )
                sweep = initial_heading_sweep(rows)
                error = abs(bearing)
                uncertainty = cap["mapping"]["yaw_uncertainty_rad"]
                residual = abs(error - MIN_TURN)
                # One coarse re-observation only when the native heading and calibrated
                # image bearing agree and the improvement exceeds the heading uncertainty.
                if (
                    sweep * bearing > 0
                    and abs(sweep) >= MIN_TURN
                    and residual + uncertainty <= band
                    and error - residual > uncertainty
                ):
                    selection.update(native_prefix_yaw_rad=sweep, expected_residual_rad=residual)
                    return checked(
                        primitive(
                            cap, "left" if bearing > 0 else "right", MIN_TURN, "coarse_target_alignment"
                        ),
                        "approach_policy",
                        "native_turn_and_offcentre_target_reobserve",
                    )
                # A straight native path rotated off body-forward still supports one short
                # step when its heading agrees with the verified target already in the band.
                heading = straight_native_heading(rows, cap["mapping"])
                if heading is not None and abs(wrap(heading - bearing)) <= band:
                    selection.update(native_heading_rad=heading)
                    return checked(
                        primitive(cap, "forward", MIN_FORWARD, "short_observed_corridor"),
                        "approach_policy",
                        "verified_target_and_straight_native_heading_agree",
                    )
                raise Hold("native_path_does_not_support_forward_approach")
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
        if inspected is not None:
            sweep = initial_heading_sweep(rows)
            if abs(sweep) > 0.10:
                native_yaw = sweep
        if side * native_yaw < -0.10:
            raise Hold("visible_target_and_native_heading_disagree")
        angle = min(TURN_CAP, max(MIN_TURN, abs(bearing)))
        residual = abs(abs(bearing) - angle)
        if (
            angle > abs(native_yaw) + 1e-6
            and residual + cap["mapping"]["yaw_uncertainty_rad"] > band
        ):
            raise Hold("coarse_alignment_residual_too_large")
        selection.update(
            native_prefix_yaw_rad=native_yaw, expected_residual_rad=residual
        )
        direction = "left" if side > 0 else "right"
        return checked(
            primitive(cap, direction, angle, "coarse_target_alignment"),
            "approach_policy",
            "fresh_view_required_after_alignment",
        )

    # Inspect new headings at the current location. Measured poses, including stalls,
    # determine where we actually looked. A forward viewpoint is also a distinct move.
    search_yaw = initial_heading_sweep(rows) if inspected is not None else native_yaw
    preferred = "right" if search_yaw < -0.10 else "left"
    recent_search = [v["attempt"] for v in views if v["attempt"]["mode"] == "SEARCH"]
    if (
        recent_search
        and abs(search_yaw) <= 0.10
        and recent_search[-1]["direction"] in ("left", "right")
    ):
        preferred = recent_search[-1]["direction"]
    opposite = "right" if preferred == "left" else "left"
    choices = [preferred, opposite, "forward"]
    if (
        len(recent_search) >= 2
        and extent >= MIN_FORWARD
        and recent_search[-1]["direction"] != "forward"
    ):
        choices = ["forward", preferred, opposite]
    # Five-move SEARCH remembers the fresh camera poses the executor accepted.
    # Re-identification is available again after one inconclusive viewpoint change.
    hint = None
    if inspected is not None:
        # Advancing along an unverified mark has no target-conditioned view benefit.
        choices = [preferred, opposite]
        match = evidence.get("positive_reference", {})
        if (
            evidence["method"] == "reference_inconclusive_seek_view"
            and views
            and match.get("reference_sha256")
            and views[-1]["attempt"].get("inconclusive_reference") == match["reference_sha256"]
            and not same_view(views[-1], pose["x_m"], pose["y_m"], pose["yaw_rad"], pose)
            and not evidence.get("previous_image_review")
        ):
            return dict(selection, reason="reidentify_after_new_view", review=True)
        if evidence["method"] == "reference_visible_without_native_agreement" and match.get(
            "localized"
        ):
            if not verified_features(match, grant) or match.get("mark_inside") is True:
                raise Hold("unbound_reference_viewpoint")
            x0, y0, x1, y1 = match["roi"]
            if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
                raise Hold("invalid_reference_viewpoint_roi")
            bearing = image_bearing(
                ((x0 + x1) / 2, (y0 + y1) / 2), evidence["camera"], cap["mapping"]
            )
            if abs(bearing) > math.radians(grant["alignment_deg"]):
                hint = "left" if bearing > 0 else "right"
                choices = [hint]

    reasons = []
    for direction in choices:
        forward = direction == "forward"
        # Attribute a search turn to the model only for an in-place native
        # prefix with coherent signed yaw. A translated viewpoint is policy-selected.
        supported_turn = (
            not forward
            and hint is None
            and direction == preferred
            and MIN_TURN <= abs(native_yaw) <= TURN_CAP
            and all(math.hypot(x, y) <= 0.025 and ang * native_yaw >= 0.0 for x, y, ang in rows)
        )
        amount = MIN_FORWARD if forward else (abs(native_yaw) if supported_turn else TURN_CAP)
        predicted_heading = pose["yaw_rad"] + (amount if direction == "left" else -amount)
        if inspected is not None:
            tx = pose["x_m"] + (amount * math.cos(pose["yaw_rad"]) if forward else 0)
            ty = pose["y_m"] + (amount * math.sin(pose["yaw_rad"]) if forward else 0)
            yaw = pose["yaw_rad"] if forward else predicted_heading
            if any(same_view(v, tx, ty, yaw, pose) for v in inspected):
                reasons.append(direction + "_view_already_inspected")
                continue
            if any(
                v["attempt"]["mode"] == "SEARCH"
                and v["attempt"]["direction"] == direction
                and v["attempt"].get("no_progress")
                and same_view(v, pose["x_m"], pose["y_m"], pose["yaw_rad"], pose)
                for v in views
            ):
                reasons.append(direction + "_stalled_from_this_view")
                continue
        elif not forward and any(
            math.hypot(v["x_m"] - pose["x_m"], v["y_m"] - pose["y_m"]) < 0.25
            and abs(wrap(v["yaw_rad"] - predicted_heading)) < math.radians(12)
            for v in views
        ):
            reasons.append("heading_already_inspected")
            continue
        if (
            inspected is None
            and recent_search
            and recent_search[-1]["direction"] == direction
            and recent_search[-1].get("no_progress")
        ):
            reasons.append("previous_attempt_made_no_progress")
            continue
        try:
            result = checked(
                primitive(cap, direction, amount, "observed_search_viewpoint"),
                "native_search_prefix" if supported_turn else "search_policy",
                "recenter_localized_reference" if hint else "inspect_new_viewpoint",
            )
            if reasons:
                result["skipped_viewpoints"] = list(reasons)
            return result
        except Hold as exc:
            reasons.append(str(exc))
    raise Hold("no_supported_search_viewpoint: " + ", ".join(reasons))


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
        travel = (
            max(0.0, action["magnitude"] - max(0.0, progress))
            + action["segment"]["clearance_reserve"]
        )
        radial = max(
            radial,
            math.hypot(
                dx + travel * math.cos(pose["yaw_rad"]),
                dy + travel * math.sin(pose["yaw_rad"]),
            ),
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
    maximum = min(
        FORWARD_CAP if forward else TURN_CAP,
        cap["limits"]["forward_m" if forward else "turn_rad"],
    )
    bounded_number(
        magnitude, MIN_FORWARD if forward else MIN_TURN, maximum, "episode_magnitude"
    )
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


def straight_corridor_prefix(rows, mapping, band):
    """A policy step along an observed corridor requires an aligned translation ray.

    A modest change of body yaw can coexist with collinear forward translations.
    Stop examining the chunk at the end of the requested step.
    """
    if "forward" not in mapping["verified_actions"]:
        return False
    px = py = 0.0
    tangent = None
    for x, y, yaw in rows:
        dx, dy = x - px, y - py
        if abs(wrap(yaw)) > band or dx < 0:
            return False
        if math.hypot(dx, dy) > 0.005:
            angle = math.atan2(dy, dx)
            if tangent is None:
                tangent = angle
            if (
                abs(wrap(angle + mapping["yaw_rad"])) + mapping["yaw_uncertainty_rad"] > band
                or abs(wrap(angle - tangent)) > mapping["yaw_uncertainty_rad"]
            ):
                return False
        if x >= MIN_FORWARD and tangent is not None:
            return True
        px, py = x, y
    return False


def straight_native_heading(rows, mapping):
    """Base-frame heading of a straight forward native prefix, or None.

    Every translation up to the step length must share one tangent, and the body yaw
    must follow that tangent, both within the heading uncertainty. Curves, reversals
    and sideways paths return None.
    """
    if "forward" not in mapping["verified_actions"]:
        return None
    uncertainty = mapping["yaw_uncertainty_rad"]
    px = py = 0.0
    tangent = None
    for x, y, yaw in rows:
        if not all(math.isfinite(v) for v in (x, y, yaw)):
            return None
        dx, dy = x - px, y - py
        if dx < 0:
            return None
        if math.hypot(dx, dy) > 0.005:
            angle = math.atan2(dy, dx)
            if tangent is None:
                tangent = angle
            if abs(wrap(angle - tangent)) > uncertainty or abs(wrap(yaw - tangent)) > uncertainty:
                return None
        if tangent is not None and math.hypot(x, y) >= MIN_FORWARD:
            return wrap(tangent + mapping["yaw_rad"])
        px, py = x, y
    return None


def initial_heading_sweep(rows):
    """Read signed relative heading from the first coherent prefix, including a curve.

    Heading is a SEARCH hint. A path starting in translation cannot acquire a turn
    hint from a distant bend. No translation from a curved chunk is executed here.
    """
    sweep = 0.0
    for x, y, yaw in rows:
        if not all(math.isfinite(v) for v in (x, y, yaw)) or x < -0.005:
            break
        angle = wrap(yaw)
        travel = math.hypot(x, y)
        if not sweep and travel > 0.025 and (abs(angle) <= 0.10 or travel >= MIN_FORWARD):
            break  # A later bend cannot turn an initially straight approach into a search hint.
        if abs(angle) < 0.005 and not sweep:
            continue
        if angle * sweep < 0 or abs(angle) + 0.02 < abs(sweep):
            break
        if abs(angle) > abs(sweep):
            sweep = angle
    return sweep


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


def verified_features(match, grant):
    """The executor independently checks the operator and quantitative feature evidence."""
    sha = match.get("reference_sha256")
    count = match.get("inliers")
    fraction, spread, residual = (match.get(k) for k in ("inlier_fraction", "feature_spread", "median_residual_px"))
    return (
        match.get("operator") == grant["operator"]
        and isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha) is not None
        and type(count) is int and count >= 12
        and all(type(v) in (int, float) for v in (fraction, spread, residual))
        and 0.8 <= fraction <= 1 and 0.12 <= spread <= 1 and 0 <= residual <= 2.0
    )


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
    if (
        status == "absent"
        and point is not None
        or status == "supported"
        and point is None
    ):
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
    if evidence["method"] != "identified_instance_features" or not verified_features(evidence, grant):
        raise Hold("insufficient_instance_evidence")
    match = evidence.get("positive_reference")
    if match is not None and (
        match.get("localized") is not True or match.get("mark_inside") is not True
        or not verified_features(match, grant)
        or any(match.get(k) != evidence[k] for k in ("reference_sha256", "roi"))
    ):
        raise Hold("reference_and_native_mark_disagree")
    if match and match.get("algorithm") == "interior_sift" and not (
        match.get("mark_in_inlier_hull") is True and match.get("mark_in_region") is True
        and all(re.fullmatch(r"[0-9a-f]{64}", str(match.get(k))) for k in
                ("current_prepared_sha256", "reference_prepared_sha256"))
    ):
        raise Hold("unsupported_detail_feature_match")
    return True
