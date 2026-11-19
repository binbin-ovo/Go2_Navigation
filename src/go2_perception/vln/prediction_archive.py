"""Read and cross-check archived predictions without a model or hardware connection."""

from .episodes import file_sha256 as file_hash
import json

from .shadow_display import target_fraction
from .supervised import MODEL_REVISION


def load(path):
    return json.loads(path.read_bytes())


def iter_records(path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def records(path):
    return list(iter_records(path))


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
