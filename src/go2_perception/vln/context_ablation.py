"""Fixed offline F/B/T/P/TP input ablation. No model loader or hardware adapter.

T changes timestamps only; P changes pooling values only, retaining B's groups.
TP additionally restores the full baseline's segment partition after filtering to
the same selected images. That extra grouping/prompt change is explicit: TP is
not a clean two-factor interaction. Missing partners can change temporal tubelets.
"""

import copy

from .bounded_context import MAX_IMAGES, select_context, validate_context
from .bounded_context import sample_factory as bounded_factory

CONDITIONS = ("F", "B", "T", "P", "TP")
CASES = ("p6b_test_c_e1_turn_right_d2", "p6b_test_c_e2_turn_left_d2",
         "p6b_test_c_e3_forward_bottles_d2", "p6b_test_c_e3_forward_bottles_d0", "live_r3")
SEGMENT_KEYS = {"frame_indices", "total_frames", "pool_spatial", "pool_mode"}


def canonical_segments(sample):
    return [{k: copy.deepcopy(s[k]) for k in sorted(SEGMENT_KEYS)}
            for s in sample["video_segments"]]


def validate_contract(spec, job):
    if set(spec) != {"condition", "bounded_context", "baseline_segments"}:
        raise ValueError("Closed ablation contract excludes labels, actions and state")
    if spec["condition"] not in CONDITIONS:
        raise ValueError("Undeclared ablation condition")
    context = spec["bounded_context"]
    if spec["condition"] == "F":
        if select_context(job["frames"], job["decision_time_s"]) != context:
            raise ValueError("Full job does not match the original causal selection")
    else:
        validate_context(context, job)
    segments = spec["baseline_segments"]
    if not isinstance(segments, list) or not segments:
        raise ValueError("Original full native metadata is required")
    for s in segments:
        if set(s) != SEGMENT_KEYS or s["total_frames"] != context["source_count"]:
            raise ValueError("Unexpected baseline segment metadata")
        ids = s["frame_indices"]
        if (not ids or len(ids) % 2 or any(type(i) is not int or not 0 <= i < context["source_count"] for i in ids)
                or ids != sorted(ids) or type(s["pool_spatial"]) is not int or s["pool_spatial"] < 1
                or s["pool_mode"] not in ("avg", "max")):
            raise ValueError("Invalid baseline native positions/pooling")


def verify_native_layout(spec, bundle):
    """Compare saved *actual* metadata with the pinned native tier selector."""
    from lightnav.slowfast import slowfast_segments
    n = spec["bounded_context"]["source_count"]
    native = slowfast_segments(n-1, n, bundle.slowfast_tiers)
    expected = [{"frame_indices":s["frame_ids"], "total_frames":n,
                 "pool_spatial":s["pool_spatial"], "pool_mode":s["pool_mode"]} for s in native]
    if expected != spec["baseline_segments"] or bundle.video_fps != 4:
        raise ValueError("Saved full layout disagrees with pinned baseline configuration")


def pooling_assignments(baseline_segments):
    assignments = {}
    for segment in baseline_segments:
        value = (segment["pool_spatial"], segment["pool_mode"])
        for i in segment["frame_indices"]:
            if i in assignments and assignments[i] != value:
                raise ValueError("Conflicting native pooling assignments")
            assignments[i] = value
    return assignments


def filtered_full_layout(baseline_segments, selected):
    """Retain baseline segment membership; no crossing omitted segment boundaries.

    Remove original per-segment padding, filter physical observations, then use the
    native pad-to-even rule on each remaining segment. Partial temporal pairs may
    regroup; callers must report that rather than adding a missing partner image.
    """
    selected = set(selected)
    result = []
    for s in baseline_segments:
        ids = s["frame_indices"]
        unique = ids[:-1] if len(ids) > 1 and ids[-1] == ids[-2] else ids
        ids = [i for i in unique if i in selected]
        if not ids:
            continue
        if len(ids) % 2:
            ids.append(ids[-1])
        result.append({**s, "frame_indices":ids})
    if {i for s in result for i in s["frame_indices"]} != selected:
        raise ValueError("A selected image is absent from the original native layout")
    if sum(len(s["frame_indices"]) for s in result) > MAX_IMAGES:
        raise ValueError("Restoring native segment membership exceeds fixed eight-slot budget")
    return result


def layout_audit(sample, baseline_segments):
    """Explicit groups, native padding, time prefixes and changed pair membership."""
    original_pairs = {tuple(s["frame_indices"][i:i+2]) for s in baseline_segments
                      for i in range(0,len(s["frame_indices"]),2)}
    groups = []
    ref = max(i for s in sample["video_segments"] for i in s["frame_indices"])
    for s in sample["video_segments"]:
        ids = s.get("source_frame_indices", s["frame_indices"])
        pairs = [ids[i:i+2] for i in range(0,len(ids),2)]
        ticks = s["frame_indices"]
        times = [(ref - (ticks[i]+ticks[i+1])/2)/sample["video_fps"] for i in range(0,len(ticks),2)]
        groups.append({"source_indices":ids, "source_pairs":pairs,
                       "pairs_not_in_full_baseline":[p for p in pairs if tuple(p) not in original_pairs],
                       "pool_spatial":s["pool_spatial"],"pool_mode":s["pool_mode"],
                       "relative_pair_seconds":times,"timestamp_prefixes":[f"<{t:.1f} seconds>" for t in times]})
    return {"segments":groups, "segment_count":len(groups),
            "native_slots":sum(len(s["source_indices"]) for s in groups),
            "video_fps_metadata":sample["video_fps"],
            "conversations":copy.deepcopy(sample["conversations"])}


def sample_factory(spec):
    if spec["condition"] == "F":
        return None  # original run_job path, unchanged

    def build(agent, job, load_rgb):
        import torch
        from lightnav.prompts import UNIFIED_TRAJ_PROMPT_TEMPLATE, build_video_block, to_rvq_prompt

        validate_contract(spec,job)
        verify_native_layout(spec,agent.engine.bundle)
        sample,tensors,metadata = bounded_factory(spec["bounded_context"])(agent,job,load_rgb)
        condition = spec["condition"]
        n = spec["bounded_context"]["source_count"]
        if condition in ("T", "TP"):
            for s in sample["video_segments"]:
                s["frame_indices"] = list(s["source_frame_indices"])
                s["total_frames"] = n
            sample["video_fps"] = agent.engine.bundle.video_fps
        if condition == "P":
            assignments = pooling_assignments(spec["baseline_segments"])
            for s in sample["video_segments"]:
                if not set(s["source_frame_indices"]) <= assignments.keys():
                    raise ValueError("A selected image has no saved full-baseline pooling assignment")
                values = {assignments[i] for i in s["source_frame_indices"]}
                if len(values) != 1:
                    raise ValueError("P cannot restore pooling without changing B grouping for this case")
                s["pool_spatial"], s["pool_mode"] = values.pop()
        if condition == "TP":
            layout = filtered_full_layout(spec["baseline_segments"],tensors)
            sample["video_segments"] = [{**s, "source_frame_indices":list(s["frame_indices"]),
                                         "video":torch.stack([tensors[i] for i in s["frame_indices"]])} for s in layout]
            # Official prompt template; the number of segment placeholders changes
            # with baseline partition restoration. This is logged, not hidden.
            template = to_rvq_prompt(UNIFIED_TRAJ_PROMPT_TEMPLATE)
            sample["conversations"][0]["value"] = template.format(
                task=job["instruction"],videos=build_video_block(len(layout)),horizon=agent.engine.bundle.predict_horizon)
        metadata.update(ablation_condition=condition, processor_timestamp_hz=sample["video_fps"],
                        timestamp_encoding="native_4hz_grid_approximation" if condition in ("T","TP") else "actual_ms_ticks",
                        pooling_restored=condition in ("P","TP"),
                        grouping="filtered_full_partition" if condition=="TP" else "unchanged_B_partition",
                        layout_audit=layout_audit(sample,spec["baseline_segments"]))
        return sample,tensors,metadata
    return build
