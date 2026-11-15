"""Freeze reviewed experiment settings before any model-driven decisions."""

import hashlib
import math
from pathlib import Path

from go2_perception.vln.pretrained import CODE_REVISION, MODEL_REVISION
from go2_perception.vln.waypoint_evaluation import GoalRegion
from go2_perception.vln.waypoints import PrefixLimits, pose_array


def digest(path):
    # Include size to reject transient early-EOF reads seen on this workstation.
    path = Path(path)
    before = path.stat()
    h, count = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            count += len(block)
            h.update(block)
    after = path.stat()
    if count != before.st_size or (
        before.st_size,
        before.st_mtime_ns,
        before.st_ino,
    ) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError(f"Incomplete or unstable source read: {path}")
    return h.hexdigest()


def validate_spec(spec, root, *, require_review=True):
    expected_keys = {
        "schema",
        "status",
        "model_revision",
        "code_revision",
        "portability_patch_sha256",
        "scene",
        "scene_sha256",
        "instruction",
        "policies",
        "prefix_limits",
        "goal",
        "budget",
        "cases",
        "history",
        "mode",
        "review",
        "physical_execution_authorized",
    }
    if set(spec) != expected_keys:
        raise ValueError("Unexpected experiment fields; no silently ignored settings")
    expected = {
        "schema": "go2-waypoint-validation-v1",
        "model_revision": MODEL_REVISION,
        "code_revision": CODE_REVISION,
        "portability_patch_sha256": "bec34b7d855f33ac94ec212a61ca507d2d677d82441add0cd1b2dfe7088e554c",
        "mode": "static_step_and_wait",
        "history": "episode_epoch_reset_actual_rgb_4hz_native_slowfast",
        "physical_execution_authorized": False,
    }
    for key, value in expected.items():
        if spec[key] != value:
            raise ValueError(f"Unsupported {key}")
    if spec["policies"] != ["first_waypoint", "bounded_prefix"]:
        raise ValueError("Preserve baseline and exactly one additional policy")
    limits = PrefixLimits(**spec["prefix_limits"])
    if limits.window_s != 0.25:
        raise ValueError("Runner captures actual images every 0.25 simulated seconds")
    GoalRegion(**spec["goal"])
    b = spec["budget"]
    if set(b) != {
        "max_simulated_s",
        "max_decisions",
        "max_host_s",
        "response_timeout_s",
    } or not all(math.isfinite(x) and x > 0 for x in b.values()):
        raise ValueError("Explicit positive finite budgets required")
    if type(b["max_decisions"]) is not int or b["max_decisions"] > 128:
        raise ValueError("Bounded suite requires 1..128 decisions")
    if (
        b["max_simulated_s"] != b["max_decisions"] * 0.25
        or b["response_timeout_s"] > b["max_host_s"]
    ):
        raise ValueError("Inconsistent simulated/decision/host budgets")
    if not isinstance(spec["instruction"], str) or not spec["instruction"].strip():
        raise ValueError("Instruction is required")
    if not 1 <= len(spec["cases"]) <= 8:
        raise ValueError("Small suite requires 1..8 declared cases")
    ids = []
    for case in spec["cases"]:
        if (
            set(case) != {"id", "pose"}
            or not case["id"].isascii()
            or not case["id"].replace("_", "").isalnum()
        ):
            raise ValueError("Safe case identifier and pose required")
        pose_array(case["pose"])
        ids.append(case["id"])
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate cases")
    scene = (Path(root) / spec["scene"]).resolve()
    scene.relative_to(Path(root).resolve())
    if digest(scene) != spec["scene_sha256"]:
        raise ValueError("Scene changed after declaration")
    if require_review:
        review = spec["review"]
        if spec["status"] != "reviewed_ready" or not isinstance(review, dict):
            raise ValueError(
                "Claude visual review is pending; do not freeze or run the model suite"
            )
        if set(review) != {
            "report_path",
            "report_sha256",
            "commit",
            "reviewed_scene_sha256",
            "scene_suitable",
            "preview_manifest_path",
            "preview_manifest_sha256",
        }:
            raise ValueError(
                "Bind the published review and actual model-input previews"
            )
        if (
            review["scene_suitable"] is not True
            or review["reviewed_scene_sha256"] != spec["scene_sha256"]
        ):
            raise ValueError("Scene not cleared by the recorded visual review")
        if len(review["commit"]) != 40 or any(
            c not in "0123456789abcdef" for c in review["commit"]
        ):
            raise ValueError("Published review commit SHA required")
        for key in ("report", "preview_manifest"):
            if digest(review[f"{key}_path"]) != review[f"{key}_sha256"]:
                raise ValueError("Review evidence changed")
    return scene
