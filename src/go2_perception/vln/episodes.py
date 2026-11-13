"""Versioned, local-only temporal episodes. Evaluation annotations are loaded separately."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMAS = Path(__file__).parent / "schemas"


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_schema(data, name):
    from jsonschema import Draft202012Validator
    errors = sorted(Draft202012Validator(json.loads((SCHEMAS / name).read_text())).iter_errors(data),
                    key=lambda e: str(list(e.path)))
    if errors:
        raise ValueError("Invalid contract: " + "; ".join(f"{list(e.path)}: {e.message}" for e in errors[:5]))
    # JSON Schema 'number' alone can accept Python's NaN; do not accept nonstandard JSON.
    json.dumps(data, allow_nan=False)


def local_path(root, relative):
    p = Path(relative)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError("Episode references must be package-relative local paths")
    path = (Path(root) / p).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("Episode reference escapes its package")
    return path


@dataclass(frozen=True)
class Episode:
    path: Path
    data: dict

    @property
    def root(self):
        return self.path.parent

    @property
    def sha256(self):
        return file_sha256(self.path)


def load_episode(path, *, verify_files=True):
    """Load a closed v1 manifest, reject mixed recordings/epochs and invalid time order.

    Missing images must be declared as image=null with a reason. An undeclared missing
    or changed file is an error when verify_files=True; no silent frame substitution.
    """
    path = Path(path).resolve()
    d = json.loads(path.read_text())
    _validate_schema(d, "vln_episode_v1.schema.json")
    start, end = d["bounds"]["start_s"], d["bounds"]["end_s"]
    if end <= start:
        raise ValueError("Episode boundaries must increase")
    ids, times, available = [], [], []
    for f in d["frames"]:
        ids.append(f["frame_id"])
        times.append(f["observation_s"])
        available.append(f["available_s"])
        if f["recording_id"] != d["recording"]["recording_id"] or f["clock_epoch"] != d["clock"]["epoch"]:
            raise ValueError("Frame crosses recording or clock epoch boundary")
        if not start <= f["observation_s"] <= end or f["available_s"] < f["observation_s"]:
            raise ValueError("Frame time outside episode or availability before observation")
        if (f["image"] is None) != (f["missing_reason"] is not None):
            raise ValueError("Missing frames require image=null and a reason; present frames require null reason")
    if len(ids) != len(set(ids)) or np.any(np.diff(times) <= 0) or np.any(np.diff(available) < 0):
        raise ValueError("Frame IDs must be unique and times strictly chronological (availability nondecreasing)")
    decisions = d["decisions"]
    if len({x['decision_id'] for x in decisions}) != len(decisions):
        raise ValueError("Duplicate decision ID")
    if any(not start <= x["time_s"] <= end for x in decisions) or np.any(np.diff([x["time_s"] for x in decisions]) <= 0):
        raise ValueError("Decision times must be ordered and inside episode")
    for g in d["gaps"]:
        if not start <= g["start_s"] < g["end_s"] <= end:
            raise ValueError("Invalid gap boundaries")
    refs = [d["calibration"]]
    refs += [f[key] for f in d["frames"] for key in ("image", "robot_state_ref") if f[key] is not None]
    for ref in refs:
        resource = local_path(path.parent, ref["path"])
        if verify_files and (not resource.is_file() or file_sha256(resource) != ref["sha256"]):
            raise ValueError(f"Missing or changed episode resource: {ref['path']}")
    return Episode(path, d)


def select_history(episode, decision_id, *, max_frames=4, horizon_s=6.0):
    """Uniform chronological subsample of available frames in a bounded past window.

    Includes the latest available frame. No repeated images or future padding. The
    caller may request max_frames=1 for a supported single-image ablation. Gaps,
    declared missing frames and stale observations remain explicit in the result.
    """
    if max_frames < 1 or horizon_s <= 0:
        raise ValueError("Positive frame budget and history horizon required")
    d = episode.data
    decision = next((x for x in d["decisions"] if x["decision_id"] == decision_id), None)
    if decision is None:
        raise ValueError(f"Unknown decision {decision_id}")
    cutoff = decision["time_s"]
    start = max(d["bounds"]["start_s"], cutoff - horizon_s)
    causal = [f for f in d["frames"] if start <= f["observation_s"] <= cutoff and f["available_s"] <= cutoff]
    valid = [f for f in causal if f["image"] is not None]
    chosen = []
    if valid:
        indices = [len(valid) - 1] if max_frames == 1 else np.linspace(0, len(valid) - 1, min(max_frames, len(valid))).round().astype(int)
        chosen = [valid[int(i)] for i in indices]
    # Clip gaps at the decision cutoff so future gap ends cannot leak into input.
    gaps = [{**g, "start_s": max(g["start_s"], start), "end_s": min(g["end_s"], cutoff)}
            for g in d["gaps"] if g["start_s"] < cutoff and g["end_s"] > start]
    return {"episode_id": d["episode_id"], "decision_id": decision_id, "decision_time_s": cutoff,
            "clock_epoch": d["clock"]["epoch"], "frames": chosen,
            "missing_frame_ids": [f["frame_id"] for f in causal if f["image"] is None],
            "gaps": gaps, "history_horizon_s": horizon_s, "max_frames": max_frames,
            "current_observation_age_s": None if not chosen else cutoff - chosen[-1]["observation_s"],
            "policy": "uniform_available_window_including_latest_v1"}


def load_annotations(path, episode):
    """Evaluation-only loader. The model adapter must never call this function."""
    a = json.loads(Path(path).read_text())
    _validate_schema(a, "vln_annotations_v1.schema.json")
    if a["episode_id"] != episode.data["episode_id"] or a["manifest_sha256"] != episode.sha256:
        raise ValueError("Annotation episode or manifest hash mismatch")
    ids = [x["decision_id"] for x in a["decisions"]]
    if len(ids) != len(set(ids)) or set(ids) != {x["decision_id"] for x in episode.data["decisions"]}:
        raise ValueError("Annotations must cover each episode decision exactly once")
    return a
