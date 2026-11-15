"""Simulation-only fixed-frame SE(2) path diagnostics; no hardware transports.

Decoded rows are absolute poses in ONE decision frame, metres/radians. This is
a holonomic kinematic prefix follower, not the upstream MPC or robot dynamics.
"""

from dataclasses import asdict, dataclass
import math

import numpy as np


def wrap_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def pose_array(value):
    pose = np.asarray(value, dtype=float)
    if pose.shape != (3,) or not np.isfinite(pose).all():
        raise ValueError("Expected finite [x_m, y_m, heading_rad]")
    return pose.copy()


def waypoint_array(value):
    points = np.asarray(value, dtype=float)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or not len(points)
        or not np.isfinite(points).all()
    ):
        raise ValueError("Expected nonempty finite (N,3) absolute ego poses")
    return points.copy()


def to_world(waypoints, decision_pose):
    points, pose = waypoint_array(waypoints), pose_array(decision_pose)
    c, s = math.cos(pose[2]), math.sin(pose[2])
    points[:, :2] = points[:, :2] @ np.array([[c, s], [-s, c]]) + pose[:2]
    points[:, 2] = wrap_angle(points[:, 2] + pose[2])
    return points


@dataclass(frozen=True)
class PrefixLimits:
    linear_mps: float = 0.8
    angular_radps: float = 3.0
    window_s: float = 0.25
    substep_s: float = 0.01

    def __post_init__(self):
        values = asdict(self)
        if not all(math.isfinite(v) and v > 0 for v in values.values()):
            raise ValueError("Motion limits must be positive and finite")
        if self.substep_s > 0.01 or self.substep_s > self.window_s:
            raise ValueError("Collision sampling must be <=0.01 s and <= window")


def bounded_prefix(waypoints, decision_pose, *, native_stop=False, limits=None):
    """Plan a time-bounded prefix without changing origin between chunk rows.

    STOP is supplied by the native decoder, never inferred from an initial row.
    All arrays are returned as JSON-compatible lists for an auditable trace.
    """
    limits = limits or PrefixLimits()
    native, origin = waypoint_array(waypoints), pose_array(decision_pose)
    world = to_world(native, origin)
    if not isinstance(native_stop, bool):
        raise ValueError("native_stop must be a decoder boolean")
    if native_stop and np.any(native != 0):
        raise ValueError("Native STOP contradicts nonzero decoded chunk")
    times, poses, indices = [0.0], [origin.tolist()], []
    selected = []
    elapsed = 0.0
    current = origin.copy()
    if not native_stop:
        for i, target in enumerate(world):
            delta = target - current
            delta[2] = wrap_angle(delta[2])
            duration = max(
                np.linalg.norm(delta[:2]) / limits.linear_mps,
                abs(delta[2]) / limits.angular_radps,
            )
            if duration <= 1e-12:
                continue
            remaining = limits.window_s - elapsed
            if remaining <= 1e-12:
                break
            used = min(float(duration), remaining)
            segment_start = current.copy()
            segment_time = elapsed
            selected.append(i)
            for t in np.linspace(0, used, math.ceil(used / limits.substep_s) + 1)[1:]:
                pose = segment_start + (t / duration) * delta
                pose[2] = wrap_angle(pose[2])
                times.append(float(segment_time + t))
                poses.append(pose.tolist())
                indices.append(i)
            elapsed += used
            current = np.asarray(poses[-1])
            if used < duration:
                break
        # Finishing a short chunk is a hold, not a native STOP or task completion.
        if elapsed < limits.window_s - 1e-12:
            duration = limits.window_s - elapsed
            for t in np.linspace(
                0, duration, math.ceil(duration / limits.substep_s) + 1
            )[1:]:
                times.append(float(elapsed + t))
                poses.append(current.tolist())
                indices.append(None)
    return {
        "policy": "bounded_prefix",
        "native_stop": native_stop,
        "decision_pose": origin.tolist(),
        "native_chunk": native.tolist(),
        "world_chunk": world.tolist(),
        "limits": asdict(limits),
        "selected_indices": selected,
        "requested_world_targets": world[selected].tolist(),
        "planned_times_s": times,
        "planned_world_poses": poses,
        "substep_target_indices": indices,
        "planned_endpoint": poses[-1],
        "physical_execution_authorized": False,
    }


def path_errors(actual_poses, decision_pose, waypoints):
    """Distance to nearest segment, including origin; yaw error at that projection.

    This measures adherence to the supplied path, not navigation correctness.
    Pure-turn segments project by heading because their translation is constant.
    """
    actual = waypoint_array(actual_poses)
    path = np.vstack([pose_array(decision_pose), to_world(waypoints, decision_pose)])
    xy_errors, yaw_errors = [], []
    for pose in actual:
        choices = []
        for a, b in zip(path[:-1], path[1:]):
            d = b[:2] - a[:2]
            turn = float(wrap_angle(b[2] - a[2]))
            length2 = float(d @ d)
            if length2 > 1e-20:
                fraction = float(np.clip((pose[:2] - a[:2]) @ d / length2, 0, 1))
            else:
                fraction = (
                    0.0
                    if abs(turn) < 1e-12
                    else float(np.clip(wrap_angle(pose[2] - a[2]) / turn, 0, 1))
                )
            xy = float(np.linalg.norm(pose[:2] - (a[:2] + fraction * d)))
            yaw = abs(float(wrap_angle(pose[2] - a[2] - fraction * turn)))
            choices.append((xy, yaw))
        # Rounding at coincident points should not select the wrong turn heading.
        minimum = min(x[0] for x in choices)
        xy, yaw = min(
            (x for x in choices if x[0] <= minimum + 1e-10), key=lambda x: x[1]
        )
        xy_errors.append(xy)
        yaw_errors.append(yaw)
    return {
        "max_cross_track_m": max(xy_errors),
        "mean_cross_track_m": float(np.mean(xy_errors)),
        "max_heading_error_rad": max(yaw_errors),
        "mean_heading_error_rad": float(np.mean(yaw_errors)),
    }
