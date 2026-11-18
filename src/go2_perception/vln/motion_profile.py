"""Effective Go2 trial profile. Distances/angles are absolute in the decision frame.

Constants below are declared trial settings informed by three controller records.
They are neither fitted dynamics nor a universal stopping guarantee.
"""

import math

PROFILE = "go2-effective-v1"
POLICY = "contiguous-prefix-v2"
SENSOR = "sensor_only"
INSPECTED = "operator_inspected_static_disc"
FORWARD_CAP = 0.60
TURN_CAP = math.radians(30)
RATES = {"forward_mps": 1.0, "turn_radps": 2.0}
LIMITS = {**RATES, "forward_m": FORWARD_CAP, "turn_rad": TURN_CAP, "duration_s": 3.0, "resend_period_s": 0.05}
# lead, reserve beyond the intended endpoint, scoring tolerance, observation noise floor.
# Angular quantities use radians throughout the contract.
RESPONSE = {
    "forward": (0.10, 0.25, 0.10, 0.02),
    "right": tuple(map(math.radians, (4, 8, 6, 2))),
    "left": tuple(map(math.radians, (10, 12, 6, 2))),
}
# Pilot admission floors: smaller raw endpoints remain visible. The motion at
# these speeds has been demonstrated only at the 0.60 m / 30 deg reference scale.
# Keep a positive useful trigger beyond the stopping allowance and scoring band.
MIN_FORWARD = 0.30
MIN_TURN = math.radians(20)
TURN_DRIFT_M = 0.10  # observed up to .07 m in the cited control tests
DISC_RADIUS_M = 0.55 + TURN_DRIFT_M
INSPECTION_TTL_S = 30.0
EPS_M = 0.005
EPS_YAW = 0.005


class Hold(ValueError):
    """Refusal with a specific reason and optional private diagnostic values."""

    def __init__(self, reason, **details):
        super().__init__(reason)
        self.details = details


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def mapped_pose(row, mapping):
    """Nominal T P T^-1. With unknown origin the omitted term has a separate bound."""
    x, y, yaw = row
    a = mapping["yaw_rad"]
    rx, ry = math.cos(a) * x - math.sin(a) * y, math.sin(a) * x + math.cos(a) * y
    if mapping["translation_m"] is not None:
        tx, ty = mapping["translation_m"]
        c, s = math.cos(yaw), math.sin(yaw)
        rx += tx - (c * tx - s * ty)
        ry += ty - (s * tx + c * ty)
    return [rx, ry, wrap(yaw)]


def mapping_slack(row, mapping):
    x, y, yaw = row
    slack = math.hypot(x, y) * 2 * math.sin(min(mapping.get("yaw_uncertainty_rad", 0.0), math.pi) / 2)
    if mapping["translation_m"] is None:
        slack += 2 * abs(math.sin(wrap(yaw) / 2)) * mapping["translation_bound_m"]
    return slack


def mapped_path(rows, mapping):
    return [
        {"row": i, "pose": mapped_pose(row, mapping), "slack_m": mapping_slack(row, mapping)}
        for i, row in enumerate(rows)
    ]


def prefix_action(rows, point, cap, forward_clear_m=None):
    """Kinematic candidate for replay or planning. Live callers also apply geometry.

    Advance from the decision origin until the first incompatible segment. If
    the cap falls between valid samples, the executable magnitude ends inside
    that segment. No new direction is inferred from a later row or image point.
    """
    lim, mapping = cap["limits"], cap["mapping"]
    mapped = mapped_path(rows, mapping)
    first = next(
        (i for i, (x, y, yaw) in enumerate(rows) if math.hypot(x, y) > EPS_M or abs(wrap(yaw)) > EPS_YAW),
        None,
    )
    if first is None:
        raise Hold("near_zero_chunk_no_motion")
    x, y, yaw = mapped[first]["pose"]
    slack = mapped[first]["slack_m"]
    forward = x - slack > EPS_M and abs(y) + slack <= lim["lateral_m"] and abs(yaw) <= lim["straight_yaw_rad"]
    if forward:
        if "forward" not in mapping["verified_actions"]:
            raise Hold("mapping_not_verified_for_forward")
        direction, kind, unit = "forward", "forward", "m"
        maximum = min(lim["forward_m"], FORWARD_CAP)
        if forward_clear_m is not None:
            maximum = min(maximum, forward_clear_m - RESPONSE[direction][1])
    else:
        if abs(yaw) <= lim["straight_yaw_rad"]:
            raise Hold("unsupported_reverse_or_lateral_prefix")
        if mapping.get("orientation_basis") != "relative_planar_heading_ccw":
            raise Hold("relative_yaw_mapping_unavailable")
        direction, kind, unit = ("left" if yaw > 0 else "right"), "turn", "rad"
        # Image evidence can veto the direction. It never supplies the angle.
        side = 1 if direction == "left" else -1
        if (0.5 - point[0]) * side <= 0.025:
            raise Hold("native_turn_inconsistent_with_off_centre_target")
        maximum = min(lim["turn_rad"], TURN_CAP)

    lead, reserve, tolerance, noise = RESPONSE[direction]
    best = None
    prev_x = prev_y = prev_yaw = greatest = previous_lower = 0.0
    boundary = "end_of_chunk"
    for entry in mapped[first:]:
        x, y, yaw = entry["pose"]
        slack = entry["slack_m"]
        dx, dy = x - prev_x, y - prev_y
        if dx < -EPS_M or abs(yaw - prev_yaw) >= math.pi:
            boundary = "reversal_or_heading_discontinuity"
            break
        if forward:
            amount = x - slack
            if (
                abs(y) + slack > lim["lateral_m"]
                or abs(yaw) > lim["straight_yaw_rad"]
                or abs(dy) > 0.005 + max(dx, 0.0) * math.tan(math.radians(15))
            ):
                boundary = "forward_prefix_bends"
                break
            residual_m, residual_yaw = abs(y) + slack, abs(yaw)
        else:
            amount = side * yaw
            # Translation is deferred. Reject sideways/reversing/non-concordant
            # travel. Relative yaw survives an unknown origin exactly; XY does not.
            native_x, native_y, _ = rows[entry["row"]]
            if (
                amount <= 0
                or side * dy < -0.02
                or native_x < -0.005
                or abs(native_y) > 0.025 + max(0.0, native_x)
            ):
                boundary = "turn_prefix_changes_intent"
                break
            residual_m, residual_yaw = math.hypot(x, y) + slack, 0.0
        if amount < greatest - (EPS_M if forward else 0.02):
            boundary = "prefix_reverses"
            break
        if amount > greatest:
            candidate = min(amount, maximum)
            # Fraction describes progress along the selected scalar (conservative
            # forward coordinate or signed heading), not a waypoint time.
            fraction = max(0.0, min(1.0, (candidate - previous_lower) / (amount - previous_lower)))
            best = (entry, candidate, fraction, residual_m, residual_yaw)
            greatest = amount
            if amount >= maximum:
                boundary = (
                    "decision_geometry_limit"
                    if forward and maximum < min(lim["forward_m"], FORWARD_CAP)
                    else "profile_limit"
                )
                break
        prev_x, prev_y, prev_yaw, previous_lower = x, y, yaw, amount

    minimum = MIN_FORWARD if forward else MIN_TURN
    if best is None or best[1] < minimum - 1e-9:
        candidate = 0.0 if best is None else max(0.0, best[1])
        reason = (
            "insufficient_forward_clearance_with_stopping_reserve"
            if boundary == "decision_geometry_limit"
            else "below_effective_resolution"
            if best
            else "unsupported_path_prefix"
        )
        raise Hold(
            f"{reason}: {candidate:.3f} {unit}, minimum {minimum:.3f} {unit}",
            prefix_limit=boundary,
            available_magnitude=candidate,
            minimum=minimum,
            unit=unit,
        )
    entry, magnitude, fraction, residual_m, residual_yaw = best
    speed = lim["forward_mps" if forward else "turn_radps"]
    # required_clearance is along the primitive. The inspected radial footprint
    # for yaw also has a separate translation budget in metres.
    return {
        "type": kind,
        "direction": direction,
        "magnitude": magnitude,
        "unit": unit,
        "speed": speed,
        "speed_unit": unit + "/s",
        "duration_s": lim["duration_s"],
        "selected_row": entry["row"],
        "mapped_pose": entry["pose"],
        "residual_translation_m": residual_m,
        "residual_yaw_rad": residual_yaw,
        "segment": {
            "profile": PROFILE,
            "policy": POLICY,
            "basis": "base_forward" if forward else "yaw_only_stop_observe_replan",
            "prefix_start": first,
            "end_fraction": fraction,
            "boundary": boundary,
            "mapped_prefix": mapped[: entry["row"] + 1],
            "stop_trigger": magnitude - lead,
            "stop_lead": lead,
            "clearance_reserve": reserve,
            "required_clearance": magnitude + reserve,
            "tolerance": tolerance,
            "noise_floor": noise,
            "sweep_radius_m": 0.0 if forward else DISC_RADIUS_M,
            "translation_budget_m": 0.0 if forward else TURN_DRIFT_M,
        },
    }


def measure_outcome(action, at_zero, settled, peak, *, termination, stop_verified):
    """Score only the settled measurement, independently of why the loop ended."""
    segment = action["segment"]
    measured = settled is not None and math.isfinite(settled)
    error = settled - action["magnitude"] if measured else None
    if not measured:
        outcome = "measurement_unavailable"
    elif abs(settled) <= segment["noise_floor"]:
        outcome = "no_measurable_progress"
    elif settled < 0:
        outcome = "wrong_direction"
    elif abs(error) <= segment["tolerance"] + 1e-9:
        outcome = "target_within_tolerance"
    elif error > 0:
        outcome = "overshoot"
    else:
        outcome = "partial_progress"
    return {
        "requested": action["magnitude"],
        "unit": action["unit"],
        "stop_trigger": segment["stop_trigger"],
        "at_first_zero": at_zero,
        "settled": settled,
        "peak_observed": peak,
        "signed_error": error,
        "tolerance": segment["tolerance"],
        "outcome": outcome,
        "termination": termination,
        "interrupted_or_blocked": termination not in ("feedback_trigger", "synthetic_feedback_trigger"),
        "stop_verified": stop_verified,
    }
