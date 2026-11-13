"""Phase 6C contracts and predeclared diagnostics; no robot transport or motion API."""

from dataclasses import dataclass
import math
import re

import numpy as np

MODEL_REVISION = "826dc5fbfa37afa8293d2e336d329b6ffc0bfb64"
CODE_REVISION = "3015508b70fb6e30bbd66701e94c9b916dcb8df8"


def validate_runtime(config, hardware):
    """Only the audited HF BF16/SDPA, unquantized path is supported here.

    This rejects Turing, rather than pretending an FP8/4-bit setting is sufficient.
    MPS requires the recorded device/timing and adaptive-pooling portability patch.
    """
    for key, expected in (
        ("backend", "hf"),
        ("dtype", "bfloat16"),
        ("attention", "sdpa"),
        ("quantization", None),
    ):
        if config.get(key) != expected:
            raise ValueError(f"Unsupported {key}: expected {expected!r}")
    device = config.get("device")
    if device not in ("cpu", "mps", "cuda"):
        raise ValueError("Explicit cpu, mps or cuda device required")
    if device == "mps" and not hardware.get("mps_available"):
        raise ValueError("Metal is unavailable in this execution context")
    if device == "cuda":
        if not hardware.get("cuda_available"):
            raise ValueError("CUDA unavailable")
        if tuple(hardware.get("compute_capability", (0, 0))) < (8, 0):
            raise ValueError(
                "Pinned upstream BF16 path requires native BF16; Turing unsupported"
            )
        if hardware.get("free_vram_bytes", 0) < 14 * 2**30:
            raise ValueError(
                "Reserve at least 14 GiB free for weights, vision, KV and activations"
            )
    if not 1 <= config.get("max_new_tokens", 0) <= 64:
        raise ValueError("Generation budget must be in [1, 64]")
    if config.get("action_feedback") != "none_observation_only":
        raise ValueError("Recorded proposals cannot be supplied as executed actions")
    if config.get("execution_authorized") is not False:
        raise ValueError("Physical execution is disabled")


def validate_experiment_config(config):
    """Reject a YAML declaration that the pinned worker would silently ignore."""
    validate_runtime({**config, "device": "cpu"}, {})  # device checked in worker
    expected = {
        "schema": "go2-pretrained-vln-experiment-v1",
        "model": "LightOriginsHQ/LightNav-0",
        "model_revision": MODEL_REVISION,
        "code_repository": "lightorigins/LightNav-0",
        "code_revision": CODE_REVISION,
        "task": "vlnce_traj",
        "max_new_tokens": 64,
        "seed": 0,
        "input_policy": "causal_4hz_from_episode_start_then_native_slowfast",
        "reset_policy": "every_decision_reconstruct_causal_history",
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Declared {key} does not match the pinned worker")
    mapping = config["comparison"]
    if (
        mapping.get("version") != "first_waypoint_diagnostic_v1"
        or mapping.get("stop_requires_native_decoder_stop") is not True
        or mapping.get("simultaneous_turn_translation") != "unsupported"
    ):
        raise ValueError("Unsupported declared comparison policy")
    for key in (
        "turn_threshold_rad",
        "advance_threshold_m",
        "max_lateral_for_advance_m",
    ):
        if not math.isfinite(mapping[key]) or mapping[key] <= 0:
            raise ValueError(f"Positive finite comparison threshold required: {key}")


def causal_history(episode, decision_id, *, fps=4.0, max_gap_s=0.375):
    """Regular causal stream from episode start, then native SlowFast selects tiers.

    Use the latest exposure at/before each grid time; include the latest available
    exposure as the final grid anchor. No future/repeated/padded source frames.
    Native tubelet duplication is a separate, logged upstream processor operation.
    SlowFast indexes a regular stream, so sparse/gapped clocks are refused rather
    than silently compressed. Quantization error is retained per source frame.
    """
    if not math.isfinite(fps) or fps <= 0 or max_gap_s <= 0:
        raise ValueError("Positive finite sampling rate/gap required")
    d = episode.data
    decision = next(
        (x for x in d["decisions"] if x["decision_id"] == decision_id), None
    )
    if decision is None:
        raise ValueError("Unknown decision")
    cutoff = decision["time_s"]
    frames = [
        f
        for f in d["frames"]
        if f["observation_s"] <= cutoff and f["available_s"] <= cutoff
    ]
    if any(f["image"] is None for f in frames) or any(
        g["start_s"] < cutoff for g in d["gaps"]
    ):
        raise ValueError(
            "Native regular-stream replay does not support declared gaps/missing images"
        )
    if not frames:
        raise ValueError("No causal observation")
    if np.any(np.diff([f["observation_s"] for f in frames]) > max_gap_s):
        raise ValueError("Observation gap exceeds regular-stream tolerance")
    last = frames[-1]["observation_s"]
    count = int(math.floor((last - frames[0]["observation_s"]) * fps + 1e-8)) + 1
    grid = last - np.arange(count - 1, -1, -1) / fps
    times = np.array([f["observation_s"] for f in frames])
    indices = np.searchsorted(times, grid + 1e-9, side="right") - 1
    if np.any(indices < 0) or len(set(indices)) != len(indices):
        raise ValueError(
            "Sparse history would require repeated/fabricated source observations"
        )
    chosen = [frames[int(i)] for i in indices]
    return {
        "frames": chosen,
        "grid_s": grid.tolist(),
        "fps": fps,
        "clock_epoch": d["clock"]["epoch"],
        "decision_time_s": cutoff,
        "current_observation_age_s": cutoff - last,
        "timestamp_quantization_error_s": [
            float(t - f["observation_s"]) for t, f in zip(grid, chosen)
        ],
        "policy": "causal_4hz_latest_anchor_then_native_slowfast",
        "executed_action_feedback": False,
    }


def native_token_contract(text):
    """Check RVQ syntax before calling the OFFICIAL codebook decoder.

    Never extract an action by reading instruction words. Unknown/prose outputs
    remain available for review. This does not claim semantic correctness.
    """
    tokens = re.findall(r"<act_l(\d+)_(\d+)>", text)
    if len(tokens) != 3 or [int(a) for a, _ in tokens] != [0, 1, 2]:
        raise ValueError("Expected exactly one ordered code per RVQ level")
    codes = [int(b) for _, b in tokens]
    if any(c < 0 or c >= 256 for c in codes):
        raise ValueError("RVQ code outside checkpoint vocabulary")
    return codes


def comparison_action(
    waypoints, *, native_stop, turn_rad=math.radians(5), advance_m=0.02, lateral_m=0.02
):
    """Lossy first-waypoint diagnostic, never a replacement for the native chunk.

    Absolute ego SE(2) waypoints: +x forward, +y left, +yaw CCW, metres/radians.
    Mixed translation/turn, reverse, lateral-only and tiny motion remain unsupported.
    STOP means native decoder stop, never verified task completion.
    """
    wp = np.asarray(waypoints, dtype=float)
    if wp.shape != (10, 3) or not np.isfinite(wp).all():
        raise ValueError("Expected ten finite absolute ego SE(2) waypoints")
    if type(native_stop) is not bool:
        raise ValueError("Native stop must be explicit boolean")
    if native_stop:
        if not np.all(wp == 0):
            raise ValueError("Native stop disagrees with decoded trajectory")
        return "STOP"
    x, y, yaw = wp[0]
    if abs(x) <= advance_m and abs(y) <= lateral_m and abs(yaw) >= turn_rad:
        return "TURN_LEFT" if yaw > 0 else "TURN_RIGHT"
    if x > advance_m and abs(y) <= lateral_m and abs(yaw) < turn_rad:
        return "ADVANCE"
    return None


@dataclass(frozen=True)
class SimulationRequest:
    """An evaluator envelope, not model context. Monotonic host seconds only."""

    episode_id: str
    clock_epoch: str
    sequence: int
    observation_monotonic_s: float
    deadline_monotonic_s: float

    def __post_init__(self):
        if (
            not self.episode_id
            or not self.clock_epoch
            or type(self.sequence) is not int
            or self.sequence < 0
        ):
            raise ValueError("Episode, epoch and nonnegative integer sequence required")
        if (
            not all(
                math.isfinite(t)
                for t in (self.observation_monotonic_s, self.deadline_monotonic_s)
            )
            or self.deadline_monotonic_s < self.observation_monotonic_s
        ):
            raise ValueError("Finite ordered host monotonic timestamps required")

    def accept(self, response, now_monotonic_s):
        times = (
            self.observation_monotonic_s,
            self.deadline_monotonic_s,
            now_monotonic_s,
        )
        if not all(math.isfinite(t) for t in times):
            raise ValueError("Nonfinite request clock")
        if (
            not self.observation_monotonic_s
            <= now_monotonic_s
            <= self.deadline_monotonic_s
        ):
            raise ValueError("Stale response or mismatched monotonic clock")
        for key in ("episode_id", "clock_epoch", "sequence"):
            if response.get(key) != getattr(self, key):
                raise ValueError(f"Response {key} mismatch")
        if response.get("execution_authorized") is not False:
            raise ValueError("Only simulation proposals are accepted")
        return response
