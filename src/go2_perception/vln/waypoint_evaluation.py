"""Evaluator-only goal and termination accounting, never model context."""

from dataclasses import dataclass

import numpy as np

from go2_perception.vln.waypoints import pose_array, wrap_angle


@dataclass(frozen=True)
class GoalRegion:
    x_range: tuple = (5.8, 6.5)
    y_range: tuple = (-0.6, 0.6)

    def __post_init__(self):
        for interval in (self.x_range, self.y_range):
            if (
                len(interval) != 2
                or not np.isfinite(interval).all()
                or interval[0] >= interval[1]
            ):
                raise ValueError("Expected finite ordered goal intervals")

    def inside(self, pose):
        x, y, _ = pose_array(pose)
        return bool(
            self.x_range[0] <= x <= self.x_range[1]
            and self.y_range[0] <= y <= self.y_range[1]
        )

    def distance(self, pose):
        return float(
            np.linalg.norm(
                pose_array(pose)[:2] - [np.mean(self.x_range), np.mean(self.y_range)]
            )
        )


def summarize_run(initial_pose, steps, termination, *, goal=None):
    goal = goal or GoalRegion()
    current = pose_array(initial_pose)
    entered = goal.inside(current)
    first_entry_s = 0.0 if entered else None
    minimum = initial_distance = goal.distance(current)
    elapsed, length = 0.0, 0.0
    native_stop, stop_inside, premature, continued = False, False, False, False
    rejections, rejected_contacts = 0, 0
    cross_track, heading_errors, latencies, ages = [], [], [], []
    memory = {}
    for row in steps:
        proposal = row["proposal"]
        accepted = row.get("response_accepted", False)
        if proposal.get("decoder_stop") is True and accepted:
            native_stop = True
            stop_inside |= goal.inside(current)
            premature |= not goal.inside(current)
        action = row.get("action") or {}
        poses = action.get("actual_world_poses", [current.tolist()])
        times = action.get("actual_times_s", [0.0])
        if len(poses) != len(times):
            raise ValueError("Actual trajectory time/pose length mismatch")
        for p, t in zip(poses[1:], times[1:]):
            p = pose_array(p)
            distance = float(np.linalg.norm(p[:2] - current[:2]))
            turn = abs(float(wrap_angle(p[2] - current[2])))
            if entered and (distance > 1e-9 or turn > 1e-9):
                continued = True
            length += distance
            current = p
            minimum = min(minimum, goal.distance(p))
            if goal.inside(p) and not entered:
                entered = True
                first_entry_s = elapsed + t
        elapsed += action.get("simulated_dt_s", 0.0)
        rejection = action.get("constraint_rejection")
        rejections += int(rejection is not None)
        if rejection:
            rejected_contacts += len(rejection.get("contacts", []))
        errors = action.get("path_error", {})
        if errors:
            cross_track.append(errors["max_cross_track_m"])
            heading_errors.append(errors["max_heading_error_rad"])
        timing = proposal.get("timing", {})
        if "native_generate_s" in timing:
            latencies.append(timing["native_generate_s"])
        if "host_observation_age_at_response_s" in row:
            ages.append(row["host_observation_age_at_response_s"])
        for k, v in proposal.get("memory", {}).items():
            if k.endswith("_bytes") and v is not None:
                memory[k] = max(memory.get(k, 0), v)

    def stats(values):
        return (
            None
            if not values
            else {
                "mean": float(np.mean(values)),
                "min": min(values),
                "max": max(values),
            }
        )

    return {
        "entered_goal_region": entered,
        "first_goal_entry_sim_s": first_entry_s,
        "native_stop_received_and_accepted": native_stop,
        "native_stop_inside_goal": stop_inside,
        "premature_native_stop": premature,
        "continued_motion_after_goal_entry": continued,
        "termination": termination,
        "externally_terminated": termination != "native_stop",
        "initial_goal_distance_m": initial_distance,
        "final_goal_distance_m": goal.distance(current),
        "minimum_goal_distance_m": minimum,
        "final_pose": current.tolist(),
        "path_length_m": length,
        "simulated_travel_time_s": elapsed,
        "model_decisions": len(steps),
        "constraint_rejections": rejections,
        "rejected_contact_pairs": rejected_contacts,
        "max_path_cross_track_error_m": max(cross_track, default=0.0),
        "max_path_heading_error_rad": max(heading_errors, default=0.0),
        "native_generation_s": stats(latencies),
        "host_observation_age_at_response_s": stats(ages),
        "peak_memory": memory,
        "success_definition": "accepted native action STOP inside declared goal region",
        "learned_navigation_success": bool(stop_inside),
        "dynamic_safety_validated": False,
        "physical_execution_authorized": False,
    }
