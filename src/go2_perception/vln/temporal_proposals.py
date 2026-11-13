"""Conservative parsing of native model text. No keyword-to-action navigation logic."""

import json
import math
import re

ACTIONS = {"ADVANCE", "TURN_LEFT", "TURN_RIGHT", "STOP", "UNRESOLVED"}
STATUSES = {
    "IN_PROGRESS",
    "COMPLETE",
    "AMBIGUOUS",
    "TARGET_NOT_FOUND",
    "INSUFFICIENT_OBSERVATION",
}


def parse_proposal(raw):
    result = {
        "proposal": "UNRESOLVED",
        "task_status": None,
        "parse_status": "rejected",
        "warnings": [],
        "reason": None,
        "native_action_reference_frame": None,
        "native_distance": None,
        "native_angle": None,
        "metric_goal": None,
        "execution_authorized": False,
        "completion_verified": False,
        "model_claims_complete": False,
    }
    text = raw.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    # Model-native explicit fields observed in synthetic development. This parses
    # an emitted category; it never derives an action from instruction keywords.
    fields = re.fullmatch(
        r"Proposal:\s*(ADVANCE|TURN_LEFT|TURN_RIGHT|STOP|UNRESOLVED)\s*\n"
        r"(?:Task status:\s*(IN_PROGRESS|COMPLETE|AMBIGUOUS|TARGET_NOT_FOUND|INSUFFICIENT_OBSERVATION)\s*\n)?"
        r"Reason:\s*([^\n]+)",
        text,
    )
    if fields:
        text = json.dumps(
            {"proposal": fields[1], "task_status": fields[2], "reason": fields[3]}
        )
        result["warnings"].append("Parsed explicit native field format; not JSON")
    try:

        def unique_pairs(pairs):
            d = {}
            for k, v in pairs:
                if k in d:
                    raise ValueError("Duplicate JSON key")
                d[k] = v
            return d

        value = json.loads(
            text,
            object_pairs_hook=unique_pairs,
            parse_constant=lambda v: (_ for _ in ()).throw(
                ValueError("Nonfinite JSON")
            ),
        )
        if not isinstance(value, dict) or set(value) - {
            "proposal",
            "task_status",
            "reason",
            "distance",
            "angle",
        }:
            raise ValueError("Expected one proposal object with documented fields only")
        if (
            value.get("proposal") not in ACTIONS
            or not isinstance(value.get("reason"), str)
            or not value["reason"].strip()
        ):
            raise ValueError("Missing/unsupported proposal or reason")
        status = value.get("task_status")
        if status is not None and status not in STATUSES:
            raise ValueError("Unsupported task_status")
        quantities = {}
        for key, units in [
            ("distance", {"m", "cm"}),
            ("angle", {"degrees", "radians"}),
        ]:
            q = value.get(key)
            if q is not None:
                if (
                    not isinstance(q, dict)
                    or set(q) != {"value", "unit"}
                    or q["unit"] not in units
                    or isinstance(q["value"], bool)
                    or not isinstance(q["value"], (float, int))
                    or not math.isfinite(q["value"])
                ):
                    raise ValueError("Invalid native quantity or unit")
                quantities["native_" + key] = {
                    **q,
                    "reference_frame": None,
                    "measured": False,
                }
        result.update(
            proposal=value["proposal"],
            task_status=status,
            reason=value["reason"],
            parse_status="accepted" if status is not None else "partial",
            model_claims_complete=status == "COMPLETE",
            **quantities,
        )
        if status is None:
            result["warnings"].append(
                "Task status absent; STOP does not establish completion"
            )
        if status == "COMPLETE" and value["proposal"] != "STOP":
            result["warnings"].append(
                "Potential inconsistency: COMPLETE with a non-STOP proposal"
            )
    except (ValueError, TypeError, KeyError) as e:
        result["warnings"].append(str(e))
    result["warnings"] += [
        "Native text has no documented robot action reference frame",
        "No measured camera-base registration or aligned depth; no metric goal",
        "Offline proposal only; geometry and traversability are unvalidated",
    ]
    return result
