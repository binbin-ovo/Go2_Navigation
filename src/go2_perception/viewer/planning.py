"""Planner-facing reading of go2-scene-v1 (no control, no planner): what a future collision-aware
planner may conclude about a point, and which needs the v1 schema does not meet.

The only answers are occupied (fresh or stale evidence) or ``unknown``. v1 carries no
free-space or visibility evidence, so nothing is ever reported as free.
"""

from __future__ import annotations

import numpy as np

from .scene import SceneFrame

OBJECT_FRESH, OBJECT_STALE, CELL, UNKNOWN = "object_fresh", "object_stale", "occupied_cell", "unknown"


def query(frame: SceneFrame, xy, margin=0.0):
    """Classify a planar odom point. Moving objects are checked at their current box only."""
    xy = np.asarray(xy, float)
    for o in frame.objects:
        lo = o.center[:2] - o.dimensions[:2] / 2 - margin
        hi = o.center[:2] + o.dimensions[:2] / 2 + margin
        if (xy >= lo).all() and (xy <= hi).all():
            return OBJECT_FRESH if o.observed else OBJECT_STALE
    if len(frame.cell_centers):
        h = frame.voxel_size / 2 + margin
        d = np.abs(frame.cell_centers[:, :2] - xy)
        if ((d[:, 0] <= h) & (d[:, 1] <= h)).any():
            return CELL
    return UNKNOWN


REQUIREMENTS = [
    ("robot pose and timestamp", "robot_pose.T_odom_base + timestamp + clock_domain", "met"),
    ("history validity", "epoch (reset on discontinuity)", "met"),
    ("static occupancy", "background cells with evidence/recency score", "partial: score is not a calibrated probability"),
    ("freshness", "status, observation_age_s, last_observed, observation_score", "met"),
    ("moving obstacle state", "center, velocity, velocity_covariance, motion_state", "partial: covariance heuristic; identity prior before 4 observations"),
    ("object extent and heading", "dimensions (AABB); orientation_xyzw null", "partial: no heading; AABB only"),
    ("free space", "none; free_space_policy=unobserved_is_unknown", "gap: needs raycast/visibility evidence"),
    ("unknown / occluded regions", "none explicit (absence = unknown)", "gap: no sensor coverage / FOV mask or occlusion map"),
    ("semantic class", "none (geometric objects only)", "gap: camera calibration needed before camera semantics"),
    ("prediction horizon", "velocity only; no predicted trajectories", "gap: forecasting not connected (CV only, validated IDs missing)"),
    ("sensor latency", "diagnostics.processing_ms; acquisition times", "partial: transport/host latency not in the row"),
    ("robot footprint / safety margin", "none in scene", "gap: belongs to the planner config"),
]


def readiness(frames):
    """Field presence over a stream, plus the static requirement table."""
    n = len(frames)
    stats = dict(
        frames=n,
        with_pose=sum(f.pose is not None for f in frames),
        epochs=len({f.epoch for f in frames}),
        objects=sum(len(f.objects) for f in frames),
        stale_objects=sum(not o.observed for f in frames for o in f.objects),
        prior_velocity_objects=sum(o.velocity_is_prior for f in frames for o in f.objects),
        moving_objects=sum(o.motion_state == "MOVING" for f in frames for o in f.objects),
    )
    return dict(stats=stats, requirements=[dict(need=a, v1_fields=b, status=c) for a, b, c in REQUIREMENTS])
