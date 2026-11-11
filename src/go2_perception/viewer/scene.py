"""Parse go2-scene-v1 rows into renderer types (standard library + NumPy, no perception imports).

All geometry stays in ``odom`` metres exactly as received. ``T_odom_base`` is kept only to
place the robot proxy; it is never applied to scene points, cells or boxes.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SCHEMA = "go2-scene-v1"
MOTION_STATES = ("STATIONARY", "MOVING", "UNCERTAIN")
STATUSES = ("observed_filtered", "stale")
ROW_FIELDS = ("schema", "timestamp", "clock_domain", "frame_id", "units", "epoch", "robot_pose", "background", "objects", "diagnostics")
OBJECT_FIELDS = (
    "object_id", "track_id", "center", "dimensions", "orientation_xyzw", "velocity", "velocity_covariance",
    "motion_state", "status", "last_observed", "observation_age_s", "age_s", "observation_score",
    "observed_box", "display_compact",
)  # fmt: skip
CELL_FIELDS = ("center", "observation_count", "last_observed", "age_s", "evidence_score")


class SceneError(ValueError):
    """A row violates the go2-scene-v1 contract; the message names the field."""


def _reject_constant(name):
    raise SceneError(f"non-finite JSON constant {name}")


def loads(line):
    """``json.loads`` that rejects NaN/Infinity, which the standard parser would accept."""
    return json.loads(line, parse_constant=_reject_constant)


def _require(d, fields, where):
    if not isinstance(d, dict):
        raise SceneError(f"{where}: expected an object")
    missing = [f for f in fields if f not in d]
    if missing:
        raise SceneError(f"{where}: missing field(s) {', '.join(missing)}")


def _vec(v, n, where, positive=False):
    a = np.asarray(v, dtype=float)
    if a.shape != (n,) or not np.isfinite(a).all():
        raise SceneError(f"{where}: expected {n} finite numbers")
    if positive and (a <= 0).any():
        raise SceneError(f"{where}: must be positive")
    return a


def _num(v, where):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise SceneError(f"{where}: expected a finite number")
    return float(v)


@dataclass
class SceneObject:
    object_id: int
    track_id: int | None
    center: np.ndarray
    dimensions: np.ndarray
    velocity: np.ndarray
    velocity_cov: np.ndarray
    motion_state: str
    status: str
    last_observed: float
    observation_age_s: float
    age_s: float
    observation_score: float
    observed_center: np.ndarray | None
    observed_dimensions: np.ndarray | None
    observed_points: int | None
    display_compact: bool

    @property
    def observed(self):
        """Directly supported by points of the current scan (``observed_box`` present)."""
        return self.status == "observed_filtered" and self.observed_center is not None

    @property
    def speed(self):
        return float(np.hypot(*self.velocity[:2]))

    @property
    def speed_sigma(self):
        """1-sigma of planar speed from the (heuristic, uncalibrated) covariance."""
        return float(np.sqrt(max(np.trace(self.velocity_cov[:2, :2]), 0.0)))

    @property
    def velocity_is_prior(self):
        """5A emits identity covariance and zero velocity before regression has evidence."""
        return bool(np.allclose(self.velocity_cov, np.eye(3)) and not self.velocity.any())


@dataclass
class SceneFrame:
    timestamp: float
    clock_domain: str
    epoch: int
    pose: np.ndarray | None  # 4x4 T_odom_base, robot proxy only
    voxel_size: float
    cell_centers: np.ndarray  # (N, 3) odom
    cell_scores: np.ndarray  # (N,) evidence/recency, not probability
    cell_ages: np.ndarray
    objects: list[SceneObject]
    diagnostics: dict = field(default_factory=dict)

    @property
    def source_time(self):
        """LiDAR acquisition midpoint when the producer recorded it (5A replay), else timestamp."""
        return float(self.diagnostics.get("source_midpoint_timestamp", self.timestamp))


def parse_object(o, t, where="object"):
    _require(o, OBJECT_FIELDS, where)
    oid = o["object_id"]
    if isinstance(oid, bool) or not isinstance(oid, int):
        raise SceneError(f"{where}.object_id: expected an integer")
    if o["motion_state"] not in MOTION_STATES:
        raise SceneError(f"{where}.motion_state: {o['motion_state']!r}")
    if o["status"] not in STATUSES:
        raise SceneError(f"{where}.status: {o['status']!r}")
    cov = np.asarray(o["velocity_covariance"], dtype=float)
    if cov.shape != (3, 3) or not np.isfinite(cov).all() or not np.allclose(cov, cov.T) or np.linalg.eigvalsh(cov).min() < -1e-9:
        raise SceneError(f"{where}.velocity_covariance: symmetric PSD 3x3 required")
    box = o["observed_box"]
    if (o["status"] == "stale") != (box is None):
        raise SceneError(f"{where}.observed_box: stale objects must not carry an observed box and vice versa")
    obs_c = obs_d = None
    n = None
    if box is not None:
        _require(box, ("center", "dimensions", "point_count"), f"{where}.observed_box")
        obs_c = _vec(box["center"], 3, f"{where}.observed_box.center")
        obs_d = _vec(box["dimensions"], 3, f"{where}.observed_box.dimensions", positive=True)
        n = int(box["point_count"])
    age_obs = _num(o["observation_age_s"], f"{where}.observation_age_s")
    last = _num(o["last_observed"], f"{where}.last_observed")
    if age_obs < 0 or abs((t - last) - age_obs) > 1e-6:
        raise SceneError(f"{where}.observation_age_s: inconsistent with timestamp - last_observed")
    score = _num(o["observation_score"], f"{where}.observation_score")
    if not 0 <= score <= 1:
        raise SceneError(f"{where}.observation_score: outside [0, 1]")
    return SceneObject(
        object_id=oid,
        track_id=o["track_id"],
        center=_vec(o["center"], 3, f"{where}.center"),
        dimensions=_vec(o["dimensions"], 3, f"{where}.dimensions", positive=True),
        velocity=_vec(o["velocity"], 3, f"{where}.velocity"),
        velocity_cov=cov,
        motion_state=o["motion_state"],
        status=o["status"],
        last_observed=last,
        observation_age_s=age_obs,
        age_s=_num(o["age_s"], f"{where}.age_s"),
        observation_score=score,
        observed_center=obs_c,
        observed_dimensions=obs_d,
        observed_points=n,
        display_compact=bool(o["display_compact"]),
    )


def parse_row(row):
    """Validate one decoded JSON row and return a :class:`SceneFrame`. Unknown fields are ignored."""
    _require(row, ROW_FIELDS, "row")
    if row["schema"] != SCHEMA:
        raise SceneError(f"row.schema: unsupported {row['schema']!r}")
    if row["frame_id"] != "odom" or row["units"] != "m":
        raise SceneError("row: frame_id must be 'odom' and units 'm'")
    if not row["clock_domain"]:
        raise SceneError("row.clock_domain: empty")
    t = _num(row["timestamp"], "row.timestamp")
    epoch = row["epoch"]
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise SceneError("row.epoch: nonnegative integer required")
    pose = None
    if row["robot_pose"] is not None:
        rp = row["robot_pose"]
        _require(rp, ("timestamp", "parent_frame", "child_frame", "T_odom_base"), "row.robot_pose")
        m = np.asarray(rp["T_odom_base"], dtype=float)
        if m.shape != (4, 4) or not np.isfinite(m).all():
            raise SceneError("row.robot_pose.T_odom_base: finite 4x4 required")
        if rp["parent_frame"] != "odom" or rp["child_frame"] != "base_link":
            raise SceneError("row.robot_pose: odom <- base_link required")
        rot = m[:3, :3]
        if not np.allclose(m[3], [0, 0, 0, 1]) or not np.allclose(rot.T @ rot, np.eye(3), atol=1e-5) or np.linalg.det(rot) < 0.99:
            raise SceneError("row.robot_pose.T_odom_base: not a rigid transform")
        if abs(_num(rp["timestamp"], "row.robot_pose.timestamp") - t) > 1e-9:
            raise SceneError("row.robot_pose.timestamp: must equal row.timestamp")
        pose = m
    bg = row["background"]
    _require(bg, ("voxel_size", "free_space_policy", "cells"), "row.background")
    if bg["free_space_policy"] != "unobserved_is_unknown":
        raise SceneError("row.background.free_space_policy: only 'unobserved_is_unknown' is supported")
    voxel = _num(bg["voxel_size"], "row.background.voxel_size")
    if voxel <= 0:
        raise SceneError("row.background.voxel_size: must be positive")
    cells = bg["cells"]
    centers = np.empty((len(cells), 3))
    scores = np.empty(len(cells))
    ages = np.empty(len(cells))
    for i, c in enumerate(cells):
        _require(c, CELL_FIELDS, f"row.background.cells[{i}]")
        centers[i] = _vec(c["center"], 3, f"row.background.cells[{i}].center")
        scores[i] = _num(c["evidence_score"], f"row.background.cells[{i}].evidence_score")
        ages[i] = _num(c["age_s"], f"row.background.cells[{i}].age_s")
    if len(cells) and ((scores < 0) | (scores > 1)).any():
        raise SceneError("row.background.cells: evidence_score outside [0, 1]")
    objects = [parse_object(o, t, f"row.objects[{i}]") for i, o in enumerate(row["objects"])]
    ids = [o.object_id for o in objects]
    if len(set(ids)) != len(ids):
        raise SceneError("row.objects: duplicate object_id")
    if pose is None and any(o.observed for o in objects):
        raise SceneError("row: missing pose cannot carry observed object claims")
    diag = row["diagnostics"] if isinstance(row["diagnostics"], dict) else {}
    return SceneFrame(t, row["clock_domain"], epoch, pose, voxel, centers, scores, ages, objects, diag)


def read_frames(path, every=1, start=None, end=None):
    """Stream :class:`SceneFrame` from JSONL. ``start``/``end`` filter on source time (seconds)."""
    with Path(path).open() as f:
        for i, line in enumerate(f):
            if not line.strip() or i % every:
                continue
            frame = parse_row(loads(line))
            if start is not None and frame.source_time < start:
                continue
            if end is not None and frame.source_time > end:
                break
            yield frame
