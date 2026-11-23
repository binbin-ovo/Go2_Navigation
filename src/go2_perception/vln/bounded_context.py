"""Bounded visual context from the offline ablation, shared by the opt-in episode.

The checkpoint's native tier selector supplies the segment layout. Selection is
latest six + episode-start two (at most eight unique observations). Irregular
source timestamps are encoded as integer millisecond positions, NOT as 4 Hz
frame indices. This is an explicit custom sparse-buffer/time adapter, not an
input-equivalent optimization or a supported upstream streaming policy.
"""

import math

POLICY = "anchor2_recent6_actual_time_v1"
MAX_IMAGES = 8
TIMESTAMP_HZ = 1000
MAX_SOURCE_FRAMES = 1024
# Anchor age bound: a 120 s five-move episode plus its 10 s trial margin (2026-11-22, trial 3k486T
# ended at 60 s with history_limit_reached). About 4 frames/s keeps 130 s near 520 source frames.
MAX_HISTORY_S = 130.0
TIERS = [
    {"name": "current", "age_lo": 0, "age_hi": 1, "mode": "dense", "pool_spatial": 1},
    {"name": "fast", "age_lo": 2, "age_hi": 5, "mode": "dense", "pool_spatial": 2},
    {"name": "anchor", "age_lo": 0, "age_hi": 0, "mode": "anchor", "num_frames": 2, "pool_spatial": 4},
]


def select_context(frames, cutoff):
    """Select metadata only; caller supplies the unchanged causal-grid candidates.

    Missing frames and excessive session history fail closed, including omitted
    frames. This function owns no accumulated state or pixels. Instruction/epoch
    separation belongs to the immutable job and every-decision native reset.
    """
    if not 1 <= len(frames) <= MAX_SOURCE_FRAMES or not math.isfinite(cutoff):
        raise ValueError("Empty/over-limit history or invalid cutoff")
    previous = -math.inf
    for frame in frames:
        obs, available = frame["observation_s"], frame["available_s"]
        if not all(math.isfinite(t) for t in (obs, available)) or not previous < obs <= available <= cutoff:
            raise ValueError("History must be chronological, unique and causal")
        if frame.get("image", True) is None or frame.get("path", True) is None:
            raise ValueError("Missing source image")
        # The common preselection has already refused missing/gapped raw streams.
        if previous != -math.inf and obs - previous > 0.5:
            raise ValueError("Unexpected gap in causal-grid candidates")
        previous = obs
    origin = frames[0]["observation_s"]
    if frames[-1]["observation_s"] - origin > MAX_HISTORY_S:
        raise ValueError("Declared history-duration bound exceeded; no silent restart")
    indices = sorted(set(range(min(2, len(frames)))) | set(range(max(0, len(frames) - 6), len(frames))))
    stamps = [frames[i]["observation_s"] for i in indices]
    ticks = [round((t - origin) * TIMESTAMP_HZ) for t in stamps]
    if any(a >= b for a, b in zip(ticks, ticks[1:])):
        raise ValueError("Distinct frames collide at millisecond timestamp resolution")
    return {
        "policy": POLICY, "source_count": len(frames), "source_indices": indices,
        "origin_observation_s": origin, "selected_observation_s": stamps,
        "timestamp_ticks": ticks, "timestamp_hz": TIMESTAMP_HZ,
        "timestamp_rounding_error_s": [origin + tick / TIMESTAMP_HZ - t for tick, t in zip(ticks, stamps)],
        "omitted_source_indices": [i for i in range(len(frames)) if i not in indices],
        "max_unique_images": MAX_IMAGES, "max_native_slots": MAX_IMAGES,
    }


def validate_context(plan, job):
    """Reject extra context (labels/actions), identity drift and incorrect bounds."""
    expected = {"policy", "source_count", "source_indices", "origin_observation_s",
                "selected_observation_s", "timestamp_ticks", "timestamp_hz",
                "timestamp_rounding_error_s", "omitted_source_indices",
                "max_unique_images", "max_native_slots"}
    if set(plan) != expected or plan["policy"] != POLICY or plan["timestamp_hz"] != TIMESTAMP_HZ:
        raise ValueError("Unexpected bounded-context contract")
    n = plan["source_count"]
    if type(n) is not int or not 1 <= n <= MAX_SOURCE_FRAMES:
        raise ValueError("Invalid source count")
    ids = sorted(set(range(min(2, n))) | set(range(max(0, n - 6), n)))
    if (plan["source_indices"] != ids or len(job["frames"]) != len(ids)
            or plan["max_unique_images"] != MAX_IMAGES or plan["max_native_slots"] != MAX_IMAGES
            or plan["omitted_source_indices"] != [i for i in range(n) if i not in ids]):
        raise ValueError("Incorrect context budget, anchors or current frame")
    stamps = [f["observation_s"] for f in job["frames"]]
    if stamps != plan["selected_observation_s"] or stamps[0] != plan["origin_observation_s"]:
        raise ValueError("Context timestamps disagree with supplied images")
    origin = plan["origin_observation_s"]
    if not 0 <= stamps[-1] - origin <= MAX_HISTORY_S:
        raise ValueError("History duration exceeds bound")
    ticks = [round((t - origin) * TIMESTAMP_HZ) for t in stamps]
    errors = [origin + t / TIMESTAMP_HZ - s for t, s in zip(ticks, stamps)]
    if (ticks != plan["timestamp_ticks"] or errors != plan["timestamp_rounding_error_s"]
            or any(a >= b for a, b in zip(ticks, ticks[1:]))):
        raise ValueError("Invalid actual-time encoding")


def sample_factory(plan):
    """Return a bounded native sample builder (heavy imports are lazy)."""
    def build(agent, job, load_rgb):
        import torch
        from lightnav.inference.samples import build_vln_traj_sample
        from lightnav.slowfast import validate_slowfast_tiers

        validate_context(plan, job)
        agent.reset(instruction=job["instruction"])
        # Official conversion; no call to observe(), which allocates a full buffer
        # and stores raw copies. At most eight normalized frames are retained here.
        tensors = {i: agent._convert_frame(load_rgb(frame))
                   for i, frame in zip(plan["source_indices"], job["frames"], strict=True)}

        class SparseEpisode:
            shape = (plan["source_count"],)

            def __getitem__(self, indices):
                return torch.stack([tensors[i] for i in indices])

        bundle = agent.engine.bundle
        old_tiers = bundle.slowfast_tiers
        try:
            bundle.slowfast_tiers = validate_slowfast_tiers(TIERS)
            sample = build_vln_traj_sample(SparseEpisode(), job["instruction"], None, bundle)
        finally:
            bundle.slowfast_tiers = old_tiers
        native_ids = {i for s in sample["video_segments"] for i in s["frame_indices"]}
        if native_ids != set(tensors):
            raise ValueError("Pinned native tier selection differs from declared selection")
        if sum(len(s["frame_indices"]) for s in sample["video_segments"]) > MAX_IMAGES:
            raise ValueError("Native tubelet slot budget exceeded")
        tick_map = dict(zip(plan["source_indices"], plan["timestamp_ticks"]))
        for segment in sample["video_segments"]:
            segment["source_frame_indices"] = list(segment["frame_indices"])
            segment["frame_indices"] = [tick_map[i] for i in segment["source_frame_indices"]]
            # Positions on a millisecond clock. This is NOT a count of captured
            # images or a 1000 fps camera claim. No missing images are synthesized.
            segment["total_frames"] = plan["timestamp_ticks"][-1] + 1
        sample["video_fps"] = TIMESTAMP_HZ
        return sample, tensors, {**plan, "processor_timestamp_hz": TIMESTAMP_HZ,
                                 "native_padding_only": True, "action_feedback": False}
    return build
