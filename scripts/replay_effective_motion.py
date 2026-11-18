"""Trace saved LightNav predictions and controller feedback offline. Never dispatch."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path

from control_log_summary import odom_yaw, summarise
from make_capability import CONTROL_LOGS, HEADING_SHA256
from go2_perception.vln.motion_profile import Hold, PROFILE, mapped_path, prefix_action
from go2_perception.vln.shadow_display import target_fraction
from go2_perception.vln.supervised import MODEL_REVISION, plan, validate_record


def load(path):
    return json.loads(path.read_bytes())


def records(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def file_hash(path):
    return sha256(path.read_bytes()).hexdigest()


def require(condition, label):
    if not condition:
        raise ValueError("archive mismatch: " + label)


def prediction(root, prediction_id):
    pred_path = root / "trials" / prediction_id.rsplit("-P", 1)[0] / "predictions.jsonl"
    matched = [r for r in records(pred_path) if r["prediction_id"] == prediction_id]
    require(
        len(matched) == 1 and load(root / "summary.json").get("ended_wall_utc"),
        "open run or ambiguous prediction",
    )
    p = matched[0]
    request = root / p["request_directory"]
    native, snap, job = (load(request / n) for n in ("native/result.json", "snapshot.json", "job.json"))
    revision = load(root / "session/worker/model.json")
    require(not p["mock"] and revision["checkpoint"] == MODEL_REVISION, "model identity")
    require(
        native["status"] == "decoded"
        and native["waypoint_convention"]
        == p["waypoint_convention"]
        == "absolute_ego_SE2_x_forward_y_left_yaw_CCW_metres_radians",
        "waypoint convention",
    )
    for n, k in (
        ("waypoints", "waypoints"),
        ("native_output", "native_output"),
        ("pointing", "pointing"),
        ("decoder_stop", "native_stop"),
        ("explicit_stop_code", "explicit_stop_code"),
    ):
        require(native[n] == p[k], n)
    point = target_fraction(native["pointing"])
    require((list(point) if point is not None else None) == p["target_fraction"], "target point")
    require(
        all(native[k] == job[k] for k in ("decision_id", "episode_id", "clock_epoch"))
        and job["instruction"] == p["instruction"],
        "model job",
    )
    source = snap["current_source"]
    for k in ("frame_id", "clock_epoch", "observation_s", "available_s"):
        require(source[k] == p["source_frame"][k], k)
    for k in ("received_monotonic_s", "exposure_monotonic_s"):
        if k in p["source_frame"]:
            require(k in source and source[k] == p["source_frame"][k], k)
    require(source["clock_epoch"] == job["clock_epoch"], "source epoch")
    image = root / p["source_image"]["path"]
    require(
        image.parent == request / "native" and file_hash(image) == p["source_image"]["file_sha256"],
        "model image",
    )
    artifacts = (
        pred_path,
        request / "snapshot.json",
        request / "job.json",
        request / "native/result.json",
        image,
    )
    return p, {str(path): file_hash(path) for path in artifacts}


def trace(root, pid, cap, case, proposal=None):
    p, sources = prediction(root, pid)
    raw, point = p["waypoints"], p["target_fraction"]
    record = {
        "case": case,
        "prediction_id": pid,
        "source_frame": p["source_frame"],
        "sources": sources,
        "instruction": p["instruction"],
        "native_output": p["native_output"],
        "native_waypoints": raw,
        "point_fraction": point,
        "native_stop": p["native_stop"],
        "receipt_refusal": p["rejection_reason_at_receipt"],
        "mapped_path": mapped_path(raw, cap["mapping"]),
        "effective_profile": PROFILE,
        "candidate": None,
        "effective_motion_measured": None,
        "archived_geometry": None if proposal is None else proposal["identity"]["geometry"],
    }
    if proposal is not None:
        require(
            proposal["native_output"] == p["native_output"]
            and proposal["waypoints"] == raw
            and proposal["point_fraction"] == point
            and proposal["native_stop"] == p["native_stop"]
            and proposal["identity"]["observation_id"] == p["source_frame"]["frame_id"]
            and proposal["identity"]["source_image_sha256"] == p["source_image"]["file_sha256"],
            "bound proposal",
        )
    try:
        if p["native_stop"]:
            raise Hold("native_stop_no_motion")
        if point is None:
            raise Hold("no_visible_target_point")
        record["candidate"] = plan(proposal, cap) if proposal else prefix_action(raw, point, cap)
        record["gate"] = (
            "saved_decision_geometry_only" if proposal else "geometry_and_per_cycle_authorization_unavailable"
        )
    except Hold as error:
        record.update(gate=str(error), detail=error.details)
    record["execution_authorized"] = False
    return record


def legacy_measurements(cycle, executed_action):
    r = cycle["execution_result"]
    events = r["events"]
    sends = [e["t"] for e in events if e.get("event") == "send_velocity"]
    zero = next(e["t"] for e in events if e.get("event") == "send_stop")
    states = [
        (e["monotonic_s"], e["robot_state"]["lidar_odom"])
        for e in events
        if e.get("robot_state", {}).get("lidar_odom")
    ]
    origin = max((v for v in states if v[0] <= sends[0]), key=lambda v: v[0])[1]

    def distance(pose):
        dx, dy = pose[0] - origin[0], pose[1] - origin[1]
        yaw = odom_yaw(origin[3:7])
        return dx * math.cos(yaw) + dy * math.sin(yaw)

    stop_pose = min(states, key=lambda v: abs(v[0] - zero))[1]
    return {
        "historical_profile": "legacy-first-row",
        "native_selected_row": executed_action["selected_row"],
        "requested_m": executed_action["magnitude"],
        "rate_mps": executed_action["speed"],
        "sends": len(sends),
        "first_send_to_zero_s": zero - sends[0],
        "trigger_feedback_m": r["measured_progress"],
        "odom_near_zero_m": distance(stop_pose),
        "settled_odom_m": distance(states[-1][1]),
        "zero_send_statuses": r["stop_send_statuses"],
        "stationary": r["observed_stationary"],
        "release_code": r["release_rpc_code"],
        "post_release_stationary": r["post_release_stationary"],
    }


def replay(args):
    cap = load(args.capability)
    validate_record("capability", cap, 2)
    require(
        file_hash(args.readiness / "hw_readiness/camera_heading.json") == HEADING_SHA256, "heading evidence"
    )
    supervised = args.readiness / "supervised/claude_one_step_02"
    cycle_path = supervised / "navigation/cycle_record.json"
    cycle = load(cycle_path)
    events = records(supervised / "events.jsonl")
    auth = cycle["authorization"]
    reviewed = next(
        e["proposal"]
        for e in events
        if e["event"] == "navigation_preview" and e["proposal"]["identity"] == auth["reviewed_identity"]
    )
    ticket = next(
        e["ticket"]
        for e in events
        if e["event"] == "navigation_dispatch" and e["ticket"]["authorization"] == auth
    )
    require(
        ticket["proposal"] == cycle["proposal"]
        and ticket["request_id"] == cycle["execution_result"]["request_id"],
        "execution result belongs to dispatch",
    )
    traces = [
        trace(supervised, p["identity"]["prediction_id"], cap, label, p)
        for label, p in (("centred_preview", reviewed), ("centred_refresh", cycle["proposal"]))
    ]
    require(
        traces[1]["candidate"]["magnitude"] <= traces[0]["candidate"]["magnitude"] + 1e-9,
        "refresh exceeds preview",
    )
    require(
        plan(cycle["proposal"], cycle["capability"]) == ticket["action"]
        and plan(reviewed, cycle["capability"]) == auth["envelope"],
        "legacy plan not reproducible",
    )
    traces[1]["historical_actual_movement"] = legacy_measurements(cycle, ticket["action"])
    pred, _ = prediction(supervised, cycle["proposal"]["identity"]["prediction_id"])
    history = load(supervised / pred["request_directory"] / "snapshot.json")["history"]
    first_capture = min(entry["source"]["exposure_monotonic_s"] for entry in history)
    delta = first_capture - auth["created_monotonic_s"]
    bound = cycle["proposal"]["identity"]["uncertainty_bound_s"]
    traces[1]["historical_refresh_timing"] = {
        "earliest_estimated_capture_after_authorization_s": delta,
        "uncertainty_bound_s": bound,
        "earliest_capture_lower_bound_after_authorization_s": delta - bound,
        "all_inputs_provably_after_authorization_under_v2": delta > bound,
    }
    live14 = args.readiness / "vln_interactive/live_14"
    # Positions are from Claude's collection notes (readiness report section 8).
    positions = {
        1: "nearby",
        **dict.fromkeys(range(2, 7), "off_centre"),
        **dict.fromkeys(range(7, 10), "centred"),
    }
    for path in sorted((live14 / "trials").glob("*/predictions.jsonl")):
        for p in records(path):
            case = positions[int(p["prediction_id"].rsplit("-P", 1)[1])]
            traces.append(trace(live14, p["prediction_id"], cap, case))
    absent = args.interactive / "vln_interactive/live_08"
    pid = "T20261117-201136-06-7ce4-P006"
    tr = trace(absent, pid, cap, "absent_sofa")
    review = absent / "trials" / pid.rsplit("-P", 1)[0] / "human_labels.jsonl"
    label = [r for r in records(review) if r["prediction_id"] == pid][-1]
    require(
        label["confirmed_by_user"]
        and label["choice"] == "agree_not_in_view"
        and label["source"] == "human_operator"
        and label["source_image"]["file_sha256"] == tr["sources"][str(absent / label["source_image"]["path"])]
        and tr["point_fraction"] is None,
        "absent review",
    )
    tr.update(review_choice=label["choice"], review_file_sha256=file_hash(review))
    traces.append(tr)
    references = {}
    for direction, (run, sha) in CONTROL_LOGS.items():
        path = args.control / run / "events.jsonl"
        require(file_hash(path) == sha, run)
        references[direction] = {**summarise(path), "events_sha256": sha}
    return {
        "profile": PROFILE,
        "mode": "offline_counterfactual",
        "controller_only_references": references,
        "cycle_sha256": file_hash(cycle_path),
        "source_capability_sha256": file_hash(args.capability),
        "cases": traces,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for flag in ("capability", "readiness", "interactive", "control", "output"):
        ap.add_argument("--" + flag, type=Path, required=True)
    args = ap.parse_args()
    report = replay(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    for case in report["cases"]:
        a = case["candidate"]
        detail = (
            f"{a['direction']} {a['magnitude']:.3f} {a['unit']}, trigger {a['segment']['stop_trigger']:.3f}"
            if a
            else case["gate"]
        )
        print(case["case"], case["prediction_id"], detail, case["receipt_refusal"] or "")
    print(args.output)


if __name__ == "__main__":
    main()
