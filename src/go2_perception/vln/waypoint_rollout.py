"""Static simulation loop with an observation-only inference callback."""

import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from go2_perception.vln.pretrained import SimulationRequest
from go2_perception.vln.waypoint_evaluation import GoalRegion, summarize_run
from go2_perception.vln.waypoints import PrefixLimits, waypoint_array


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run_episode(room, spec, case, policy, infer, destination, *, clock_epoch=None):
    """Execute paused inference -> bounded simulated action -> actual new image.

    ``infer(job, output_directory)`` receives ONLY the whitelisted instruction,
    identities, timestamps and causal RGB history. The parent CLI enforces a hard
    host-time bound on its child process; this loop also checks elapsed budgets.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    epoch = clock_epoch or f"simulation-{time.time_ns()}"
    episode = f"{case['id']}__{policy}"
    budget, steps, history = spec["budget"], [], []
    room.pose = np.asarray(case["pose"], dtype=float)
    room._place()
    initial = room.pose.copy()
    goal = GoalRegion(**spec["goal"])

    def evaluate():
        return room.evaluation() | {
            "goal_region": spec["goal"],
            "inside_stopping_region": goal.inside(room.pose),
            "distance_to_goal_center_m": goal.distance(room.pose),
        }

    started = time.monotonic()
    termination = "decision_budget"
    write_json(
        destination / "episode_declaration.json",
        {"case": case, "policy": policy, "clock_epoch": epoch, "spec": spec},
    )
    if room.contacts():
        raise ValueError("Declared starting pose is in collision")
    for i in range(budget["max_decisions"]):
        if room.clock >= budget["max_simulated_s"] - 1e-9:
            termination = "simulated_time_budget"
            break
        now = time.monotonic()
        if now - started >= budget["max_host_s"]:
            termination = "host_time_budget"
            break
        capture = now
        camera = destination / f"camera_{i:03d}.png"
        Image.fromarray(room.render()).save(camera)
        history.append(
            {
                "path": str(camera.resolve()),
                "sha256": hashlib.sha256(camera.read_bytes()).hexdigest(),
                "observation_s": room.clock,
                "available_s": room.clock,
            }
        )
        job = {
            "episode_id": episode,
            "clock_epoch": epoch,
            "decision_id": f"d{i}",
            "decision_time_s": room.clock,
            "instruction": spec["instruction"],
            "frames": list(history),
        }
        request = SimulationRequest(
            episode,
            epoch,
            i,
            capture,
            min(capture + budget["response_timeout_s"], started + budget["max_host_s"]),
        )
        row = {
            "index": i,
            "before": evaluate(),
            "simulated_observation_s": room.clock,
            "proposal": {"status": "pending"},
            "response_accepted": False,
            "action": None,
        }
        steps.append(row)
        write_json(destination / "steps.json", steps)
        write_json(destination / f"job_{i:03d}.json", job)
        try:
            proposal = infer(job, destination / "inference" / f"{i:03d}")
            row["proposal"] = proposal
            row["host_observation_age_at_response_s"] = time.monotonic() - capture
            if proposal.get("decision_id") != f"d{i}":
                raise ValueError("Response decision mismatch")
            request.accept({**proposal, "sequence": i}, time.monotonic())
            row["transport_accepted"] = True
            if proposal.get("status") != "decoded":
                termination = "model_or_decode_error"
            else:
                chunk = waypoint_array(proposal["waypoints"])
                if (
                    chunk.shape != (10, 3)
                    or type(proposal.get("decoder_stop")) is not bool
                ):
                    raise ValueError(
                        "Expected ten native waypoints and decoder STOP boolean"
                    )
                if proposal["decoder_stop"] and np.any(chunk != 0):
                    raise ValueError("Native STOP contradicts nonzero decoded chunk")
                row["response_accepted"] = True
                row["action"] = room.execute_waypoint_policy(
                    chunk,
                    policy,
                    native_stop=proposal["decoder_stop"],
                    limits=PrefixLimits(**spec["prefix_limits"]),
                )
                if row["action"]["constraint_rejection"]:
                    termination = "constraint_rejection"
                elif proposal["decoder_stop"]:
                    termination = "native_stop"
        except Exception as exc:
            row["error"] = repr(exc)
            termination = "response_or_execution_error"
        row["after"] = evaluate()
        write_json(destination / "steps.json", steps)
        # Native STOP or external safety/error termination, never evaluator goal entry.
        if termination != "decision_budget":
            break
    if (
        termination == "decision_budget"
        and room.clock >= budget["max_simulated_s"] - 1e-9
    ):
        termination = "simulated_time_budget"
    Image.fromarray(room.render()).save(destination / "camera_final.png")
    summary = summarize_run(
        initial, steps, termination, goal=GoalRegion(**spec["goal"])
    )
    summary.update(
        case=case["id"],
        policy=policy,
        instruction=spec["instruction"],
        mode="static_step_and_wait",
        host_elapsed_s=time.monotonic() - started,
        source="model_driven" if steps else "no_model_decisions",
        clock_epoch=epoch,
        human_behavior_review="pending",
        simulation_advanced_during_inference=False,
        failure_stage="none" if summary["learned_navigation_success"] else termination,
    )
    write_json(destination / "summary.json", summary)
    return summary
