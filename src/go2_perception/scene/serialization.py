"""Versioned, finite JSONL scene snapshots usable without the perception pipeline."""

import json
from pathlib import Path
import numpy as np
from . import SCHEMA_VERSION


def validate_scene(row):
    if (
        row["schema"] != SCHEMA_VERSION
        or row["units"] != "m"
        or not row["clock_domain"]
        or not row["frame_id"]
    ):
        raise ValueError("Scene version, metric frame and clock required")
    json.dumps(row, allow_nan=False)
    t = float(row["timestamp"])
    pose = row["robot_pose"]
    if pose is not None:
        matrix = np.asarray(pose["T_odom_base"])
        if (
            matrix.shape != (4, 4)
            or pose["timestamp"] != t
            or pose["parent_frame"] != row["frame_id"]
        ):
            raise ValueError("Invalid snapshot pose")
        if (
            not np.allclose(matrix[3], [0, 0, 0, 1])
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5)
            or np.linalg.det(matrix[:3, :3]) < 0.99
        ):
            raise ValueError("Pose must be rigid")
    if row["background"]["free_space_policy"] != "unobserved_is_unknown":
        raise ValueError("No inferred free space without visibility evidence")
    if row["background"]["voxel_size"] <= 0:
        raise ValueError("Positive voxel size required")
    for cell in row["background"]["cells"]:
        if (
            len(cell["center"]) != 3
            or cell["observation_count"] < 1
            or not 0 <= cell["evidence_score"] <= 1
            or cell["last_observed"] > t
            or not np.isclose(cell["age_s"], t - cell["last_observed"])
        ):
            raise ValueError("Invalid background evidence")
    ids = set()
    for obj in row["objects"]:
        if obj["object_id"] in ids:
            raise ValueError("Duplicate scene object ID")
        ids.add(obj["object_id"])
        if obj["motion_state"] not in ("STATIONARY", "MOVING", "UNCERTAIN") or obj[
            "status"
        ] not in ("observed_filtered", "stale"):
            raise ValueError("Invalid object status")
        if (
            len(obj["center"]) != 3
            or len(obj["dimensions"]) != 3
            or min(obj["dimensions"]) <= 0
            or len(obj["velocity"]) != 3
        ):
            raise ValueError("Invalid object geometry")
        cov = np.asarray(obj["velocity_covariance"])
        if (
            cov.shape != (3, 3)
            or not np.allclose(cov, cov.T)
            or np.linalg.eigvalsh(cov).min() < -1e-9
        ):
            raise ValueError("Invalid covariance")
        if (
            not 0 <= obj["observation_score"] <= 1
            or not np.isclose(obj["observation_age_s"], t - obj["last_observed"])
            or obj["observation_age_s"] < 0
            or obj["age_s"] < obj["observation_age_s"]
        ):
            raise ValueError("Invalid object age/evidence")
        if (obj["status"] == "stale") != (obj["observed_box"] is None):
            raise ValueError("Stale geometry must not be labelled directly observed")
    return row


def write_scenes(path, rows):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(validate_scene(row), allow_nan=False) + "\n")


def read_scenes(path):
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                yield validate_scene(json.loads(line))
