"""Explicit offline entry point. Replay saved proposals or use the deterministic image fixture."""

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
from tempfile import mkdtemp
from time import perf_counter

from go2_perception.vln.continuous import ContinuousNavigator, PERIOD_S
from go2_perception.vln.continuous_simulation import SCENARIOS, simulate


def archive(root):
    from replay_effective_motion import file_hash, load, prediction, require
    from go2_perception.vln.episode_grounding import ground, object_reference
    from go2_perception.vln.interactive_trials import expand_target
    from go2_perception.vln.supervised import validate_record

    path = root / "navigation/cycle_record.json"
    cycle = load(path)
    p, auth = cycle["proposal"], cycle["authorization"]
    validate_record("proposal", p, 2)
    ident, refid, human = p["identity"], auth["reviewed_identity"], auth["target"]
    pred, sources = prediction(root, ident["prediction_id"])
    reviewed, older_sources = prediction(root, refid["prediction_id"])
    trial_path = root / "trials" / ident["trial_id"] / "trial.json"
    trial = load(trial_path)
    ref_trial_path = root / "trials" / refid["trial_id"] / "trial.json"
    ref_trial = load(ref_trial_path)
    target = ref_trial["operator_text"]
    original_spec = expand_target(target)
    require(
        ident["session_id"] == trial["session_id"] == refid["session_id"]
        and ident["trial_id"] == trial["trial_id"] == pred["trial_id"]
        and refid["trial_id"] == ref_trial["trial_id"] == reviewed["trial_id"]
        and ref_trial["session_id"] == trial["session_id"]
        and ref_trial["kind"] == original_spec["kind"]
        and ref_trial["template_id"] == original_spec["template_id"]
        and ref_trial["instruction"] == original_spec["instruction"]
        and trial["operator_text"] == trial["instruction"]
        and ident["instruction"]
        == trial["instruction"]
        == pred["instruction"]
        == reviewed["instruction"]
        == ref_trial["instruction"]
        and ident["clock_epoch"]
        == refid["clock_epoch"]
        == pred["source_frame"]["clock_epoch"]
        == reviewed["source_frame"]["clock_epoch"]
        and ident["instruction_id"] == refid["instruction_id"]
        and ident["prediction_id"] != refid["prediction_id"],
        "session / instruction / reference identity",
    )
    require(
        p["native_output"] == pred["native_output"]
        and p["waypoints"] == pred["waypoints"]
        and p["point_fraction"] == pred["target_fraction"]
        and p["native_stop"] == pred["native_stop"]
        and ident["source_image_sha256"] == pred["source_image"]["file_sha256"]
        and ident["observation_id"] == pred["source_frame"]["frame_id"]
        and not p["synthetic"]
        and pred["rejection_reason_at_receipt"] is None,
        "original accepted proposal",
    )
    require(
        ident["observed_monotonic_s"] == pred["source_frame"]["exposure_monotonic_s"]
        and ident["received_monotonic_s"] == pred["source_frame"]["received_monotonic_s"]
        and reviewed["source_image"]["file_sha256"]
        == refid["source_image_sha256"]
        == human["reviewed_image_sha256"]
        and reviewed["source_frame"]["frame_id"] == refid["observation_id"]
        and reviewed["received_host_monotonic_s"]
        <= human["confirmed_monotonic_s"]
        <= ident["observed_monotonic_s"] - ident["uncertainty_bound_s"],
        "capture timeline or reviewed image",
    )
    ref = object_reference(
        root / reviewed["source_image"]["path"], human["roi_xyxy"], target, human["confirmed_by"]
    )
    request = root / pred["request_directory"]
    snapshot = load(request / "snapshot.json")
    job = load(request / "job.json")
    history = snapshot["history"]
    cutoff = snapshot["source_cutoff_s"]
    require(
        history
        and len(history) == len(job["frames"])
        and job["decision_time_s"] == cutoff
        and history[-1]["source"] == snapshot["current_source"]
        and snapshot["annotations_consumed"] is False
        and snapshot["previous_actions_supplied"] is False,
        "causal history shape",
    )
    source_paths = {**sources, **older_sources}
    previous = -math.inf
    for index, (row, frame) in enumerate(zip(history, job["frames"])):
        source = row["source"]
        require(
            source["clock_epoch"] == ident["clock_epoch"]
            and previous
            < source["observation_s"]
            == frame["observation_s"]
            <= frame["available_s"]
            == source["available_s"]
            <= cutoff
            and source["received_monotonic_s"]
            <= snapshot["host_submit_monotonic_s"]
            <= pred["received_host_monotonic_s"],
            "causal frame clock",
        )
        original = Path(frame["path"])
        local = request / f"frame_{index:04d}.png"
        require(
            original.name == local.name
            and original.parent.name == request.name
            and file_hash(local) == frame["sha256"],
            "archived model history image",
        )
        source_paths[str(local)] = frame["sha256"]
        previous = source["observation_s"]
    evidence = ground(pred, root / pred["source_image"]["path"], history[-1]["rectification"], target, ref)
    for local in (
        path,
        trial_path,
        ref_trial_path,
        root / "summary.json",
        root / "session/worker/model.json",
    ):
        source_paths[str(local)] = file_hash(local)
    rows = [e for e in cycle["execution_result"]["events"] if e.get("event") == "current_guards"]
    require(
        rows and all(b["monotonic_s"] >= a["monotonic_s"] for a, b in zip(rows, rows[1:])), "guard chronology"
    )
    return (
        cycle,
        dict(
            proposal=p,
            grounding=evidence,
            available_s=pred["received_host_monotonic_s"],
            archived_history_frames=len(history),
            history_policy=snapshot["history_policy"],
        ),
        rows,
        source_paths,
    )


def archived_guard(row, conditional):
    state, t = row["robot_state"], row["monotonic_s"]
    remote = state["remote"]
    # The original calls serialized the latest radio values, without the radio receipt clock.
    radio = state.get("remote_monotonic_s")
    assumed = radio is None and conditional
    neutral = (
        remote is not None
        and remote.get("buttons") == 0
        and all(
            type(remote.get(k)) in (int, float) and abs(remote[k]) <= 0.10 for k in ("lx", "ly", "rx", "ry")
        )
    )
    # StateMonitor.is_still() in the pinned canonical controller uses planar speed and yaw rate.
    stationary = (
        state.get("velocity") is not None
        and state.get("yaw_speed") is not None
        and len(state["velocity"]) >= 2
        and math.hypot(*state["velocity"][:2]) < 0.03
        and abs(state["yaw_speed"]) < 0.05
    )
    return dict(
        available_s=t,
        snapshot=row["guards"],
        interlock=dict(
            neutral=neutral,
            healthy=state.get("error_code") == 100,
            stationary=stationary,
            radio_s=t if assumed else radio,
            state_s=t - state["sport_age_s"] if state.get("sport_age_s") is not None else None,
        ),
        radio_time_assumed=assumed,
    )


def replay(root, record):
    from replay_effective_motion import file_hash

    cycle, delivery, rows, hashes = archive(root)
    proposal = delivery["proposal"]
    ident = proposal["identity"]
    start = delivery["available_s"]
    until = rows[-1]["monotonic_s"] + 0.55
    if not 0 <= until - start < 60:
        raise ValueError("recorded control window exceeds the offline episode budget")
    result = dict(
        source="recorded_single_action_open_loop",
        recording=str(root),
        sources=hashes,
        prediction_id=ident["prediction_id"],
        source_image_sha256=ident["source_image_sha256"],
        camera_epoch=ident["clock_epoch"],
        capture_uncertainty_s=ident["uncertainty_bound_s"],
        capture_to_result_s=start - ident["observed_monotonic_s"],
        guard_count=len(rows),
        historical_input_frames=delivery["archived_history_frames"],
        history_policy=delivery["history_policy"],
        grounding_status=delivery["grounding"]["status"],
        inliers=delivery["grounding"].get("inliers"),
        physical_commands_sent=0,
        passes={},
    )
    poses = [e["guards"]["pose"] for e in rows]
    result["recorded_guard_path_length_m"] = sum(
        math.hypot(b["x_m"] - a["x_m"], b["y_m"] - a["y_m"]) for a, b in zip(poses, poses[1:])
    )
    result["radio_receipt_timestamps_present"] = sum(
        e["robot_state"].get("remote_monotonic_s") is not None for e in rows
    )
    record(dict(kind="archived_proposal", delivery=delivery))
    idle = next(
        (e for e in cycle["execution_result"]["events"] if e.get("event") == "idle_stillness_at_dispatch"),
        None,
    )
    for name in ("recorded_only", "assume_radio_sampled_at_guard_check"):
        nav = ContinuousNavigator(
            cycle["capability"],
            ident,
            target=delivery["grounding"]["target"],
            operator=cycle["authorization"]["target"]["confirmed_by"],
            origin=ident["pose"],
            reference_sha256=cycle["authorization"]["target"]["reviewed_image_sha256"],
            started_s=start,
            duration_s=60,
        )
        cursor = nonzero = 0
        guard = None
        pending = delivery
        reasons = Counter()
        first = last = None
        for n in range(math.ceil((until - start) / PERIOD_S) + 1):
            now = start + n * PERIOD_S
            while cursor < len(rows) and rows[cursor]["monotonic_s"] <= now:
                guard = archived_guard(rows[cursor], name != "recorded_only")
                if idle and idle["monotonic_s"] <= guard["available_s"]:
                    guard["prior_stillness"] = idle
                cursor += 1
            # Keep only the latest available result while waiting for the first delivered guard.
            offer = pending if guard is not None else None
            if offer:
                pending = None
            packet = nav.tick(now, guard, delivery=offer)
            moving = any(packet["velocity"])
            if moving:
                nonzero += 1
                first = now if first is None else first
                last = now
            else:
                reasons[packet["reason"]] += 1
            record(
                dict(
                    kind="tick",
                    counterfactual_pass=name,
                    t=now,
                    guard=guard,
                    offered_prediction_id=ident["prediction_id"] if offer else None,
                    packet=packet,
                )
            )
        result["passes"][name] = dict(
            plans=nav.plans,
            nonzero_ticks=nonzero,
            zero_reasons=dict(reasons),
            first_nonzero_s=first,
            last_nonzero_s=last,
            odom_travel_on_eligible_samples_m=nav.travel,
            odom_turn_on_eligible_samples_rad=nav.turn,
            end_reason="recorded_window_ended",
            uninterrupted_replacements=nav.rollovers,
        )
    if any(file_hash(Path(path)) != initial for path, initial in hashes.items()):
        raise ValueError("an archived source changed during replay")
    result["source_hashes_verified_after_replay"] = True
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    sim = sub.add_parser("simulate", help="kinematic proxy and delayed synthetic visual decisions")
    sim.add_argument("--scenario", choices=(*SCENARIOS, "all"), default="all")
    sim.add_argument(
        "--wall-clock",
        action="store_true",
        help="pace the proxy on the host clock with one asynchronous fixture worker",
    )
    saved = sub.add_parser("replay", help="closed single-action sessions, read-only")
    saved.add_argument("--recording", type=Path, required=True, action="append", metavar="SESSION_DIR")
    for command in (sim, saved):
        command.add_argument(
            "--output", type=Path, help="fresh directory; default under GO2_WORKSPACE/runs/continuous"
        )
    args = parser.parse_args(argv)
    if any(os.environ.get(k, "0") != "0" for k in ("GO2_HARDWARE", "GO2_SUPERVISED_LIVE")):
        parser.error("disable GO2_HARDWARE and GO2_SUPERVISED_LIVE before using this offline utility")
    if args.mode == "replay":
        roots = [p.resolve(strict=True) for p in args.recording]
        if len(set(roots)) != len(roots):
            parser.error("a recording appears twice")
    dest = (args.output if args.output else Path(os.environ["GO2_WORKSPACE"]) / "runs/continuous").resolve()
    if args.mode == "replay":
        from go2_perception.viewer.privacy import require_private_dir

        if any(dest.is_relative_to(root) for root in roots):
            parser.error("output must be outside the input recordings")
        require_private_dir(dest)
    if args.output:
        dest.mkdir(parents=True, exist_ok=False, mode=0o700)
        out = dest
    else:
        dest.mkdir(parents=True, exist_ok=True, mode=0o700)
        out = Path(mkdtemp(prefix=args.mode + "-", dir=dest))
    if args.mode == "simulate":
        fixture_path = (
            Path(__file__).resolve().parents[1] / "examples/supervised_navigation/synthetic_cycle.json"
        )
        fixture = json.loads(fixture_path.read_bytes())
        jobs = SCENARIOS if args.scenario == "all" else [args.scenario]
    else:
        jobs = roots
    started = perf_counter()
    results = []
    for n, job in enumerate(jobs, 1):
        trace = f"trace-{n:02d}.jsonl"
        with (out / trace).open("x") as stream:

            def record(row):
                stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")

            result = (
                simulate(fixture, job, record, wall_clock=args.wall_clock)
                if args.mode == "simulate"
                else replay(job, record)
            )
        results.append(dict(result, trace=trace))
    summary = dict(
        schema="go2-continuous-offline-v1",
        mode=args.mode,
        physical_commands_sent=0,
        host_elapsed_s=perf_counter() - started,
        results=results,
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(out / "summary.json")


if __name__ == "__main__":
    main()
