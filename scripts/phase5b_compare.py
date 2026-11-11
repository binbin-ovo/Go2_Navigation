"""Original 0.3 s pipeline vs Phase 5A (and diagnostic variants) on A1/A2/B/C.

Two separate groups of measures, never merged into one score:
- perception: fresh coverage and moving recall (and the other 5A evaluator outputs);
- visual stability: displayed-box step jitter, appear/disappear counts, false MOVING,
  MOVING delay, stale/coasting durations, persistence through detection gaps, latency.
Annotations are LiDAR-derived, Claude-reviewed, not human-verified, and only used here.

  PYTHONPATH=src .venv/bin/python scripts/phase5b_compare.py --data ../Go2_Navigation/data \
      --variant 5A=../Go2_Navigation/data/outputs/phase5b/tap \
      --variant 5A_obj0.3=../Go2_Navigation/data/outputs/phase5b/tap_obj0.3 \
      --output ../Go2_Navigation/data/outputs/phase5b/comparison.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_stable_scene_real import evaluate  # noqa: E402
from go2_perception.tracking import ground_truth as gt  # noqa: E402
from go2_perception.viewer.privacy import require_private_dir  # noqa: E402

RECORDINGS = ["test_a1", "test_a2", "test_b", "test_c"]
GATES = dict(person=0.5, static_obstacle=0.15)
ROI_M = 3.5


def load_scene_light(path):
    """5A rows without background cells (not needed here; keeps memory small)."""
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            r["background"]["cells"] = []
            rows.append(r)
    return rows


def jsonl(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def displayed(rows, kind):
    """Unified per-frame display lists: (t_mid, robot_xy, [dict(id, xy, fresh, moving)])."""
    out = []
    for r in rows:
        if kind == "original":
            t = r["t"]
            objs = [dict(id=o[0], xy=np.array(o[1:3]), fresh=o[9] == 0 and o[7] == "C", moving=o[8] == "M") for o in r["tracks"] if o[7] in ("C", "L")]
            pose = r.get("_pose")
        else:
            t = r["diagnostics"]["source_midpoint_timestamp"]
            objs = [dict(id=o["object_id"], xy=np.array(o["center"][:2]), fresh=o["status"] == "observed_filtered", moving=o["status"] == "observed_filtered" and o["motion_state"] == "MOVING") for o in r["objects"]]
            pose = np.asarray(r["robot_pose"]["T_odom_base"])[:2, 3] if r["robot_pose"] else None
        out.append((t, pose, objs))
    return out


def stability(disp, doc, types):
    gtf = {round(f["t"], 3): f["objects"] for f in doc["frames"]}
    duration = disp[-1][0] - disp[0][0]
    first, last, life = {}, {}, {}
    stale_runs, run_start = [], {}
    false_moving_frames = moving_frames = 0
    per_obj = defaultdict(list)  # GT id -> [(t, matched_any_xy or None, matched_fresh)]
    prev_match = {}
    steps = []
    for t, rxy, objs in disp:
        near = [o for o in objs if rxy is None or np.hypot(*(o["xy"] - rxy)) <= ROI_M]
        for o in near:
            first.setdefault(o["id"], t)
            last[o["id"]] = t
        # stale / coasting run lengths per displayed id
        for o in objs:
            if not o["fresh"]:
                run_start.setdefault(o["id"], t)
            elif o["id"] in run_start:
                stale_runs.append(t - run_start.pop(o["id"]))
        ids = {o["id"] for o in objs}
        for i in list(run_start):
            if i not in ids:
                stale_runs.append(t - run_start.pop(i))
        states = [dict(s, type=types[s["id"]]) for s in gtf.get(round(t, 3), []) if gt.scorable(s)]
        moving_people = [s for s in states if types[s["id"]] == "person" and s.get("motion") == "moving"]
        for o in near:
            if o["moving"]:
                moving_frames += 1
                if not any(gt.state_distance(s, o["xy"]) <= 0.5 for s in moving_people):
                    false_moving_frames += 1
        for s in states:
            g = GATES.get(s["type"], 0.5)
            cand = sorted(objs, key=lambda o: gt.state_distance(s, o["xy"]))
            any_m = next((o for o in cand if gt.state_distance(s, o["xy"]) <= g), None)
            fresh_m = next((o for o in cand if o["fresh"] and gt.state_distance(s, o["xy"]) <= g), None)
            per_obj[s["id"]].append((t, any_m is not None, fresh_m is not None))
            if s["type"] == "static_obstacle":
                pm = prev_match.get(s["id"])
                if any_m is not None and pm is not None and pm[0] == any_m["id"] and t - pm[2] < 0.2:
                    steps.append(float(np.hypot(*(any_m["xy"] - pm[1]))))
                prev_match[s["id"]] = (any_m["id"], any_m["xy"], t) if any_m is not None else None
    for i in first:
        life[i] = last[i] - first[i]
    gaps = defaultdict(list)
    for key, seq in per_obj.items():
        cur = None
        for t, any_m, fresh_m in seq:
            if not fresh_m:
                if cur is None or t - cur["last"] > 0.2:
                    if cur is not None:
                        gaps[key].append(cur)
                    cur = dict(start=t, last=t, frames=0, held=0)
                cur["frames"] += 1
                cur["held"] += any_m
                cur["last"] = t
            elif cur is not None:
                gaps[key].append(cur)
                cur = None
        if cur is not None:
            gaps[key].append(cur)
    minutes = duration / 60
    births = len(first)
    return dict(
        duration_s=round(duration, 1),
        displayed_ids_within_3p5m=births,
        appear_per_min=round(births / minutes, 1),
        disappear_per_min=round(sum(1 for i in last if last[i] < disp[-1][0] - 0.1) / minutes, 1),
        short_lived_lt_0p5s=sum(v < 0.5 for v in life.values()),
        unsupported_moving_frame_fraction=round(false_moving_frames / moving_frames, 3) if moving_frames else None,
        unsupported_moving_frames_per_min=round(false_moving_frames / minutes, 1),
        static_box_step_m=dict(median=round(float(np.median(steps)), 4), p95=round(float(np.percentile(steps, 95)), 4), n=len(steps)) if steps else None,
        stale_or_coasting_run_s=dict(mean=round(float(np.mean(stale_runs)), 3), p95=round(float(np.percentile(stale_runs, 95)), 3), max=round(float(np.max(stale_runs)), 3), n=len(stale_runs)) if stale_runs else None,
        detection_gaps={
            k: dict(
                gaps=len(v),
                gap_s_median=round(float(np.median([g["last"] - g["start"] for g in v])), 3) if v else None,
                frames=int(sum(g["frames"] for g in v)),
                held_fraction=round(sum(g["held"] for g in v) / max(sum(g["frames"] for g in v), 1), 3),
            )
            for k, v in gaps.items()
        },
    )


def summarize_eval(ev):
    out = {}
    for key, o in ev["objects"].items():
        delays = [e["moving_delay_s"] for e in o["motion_events"] if not e["left_censored"]]
        out[key] = dict(
            fresh_coverage=round(o["coverage"], 3),
            moving_recall=None if o["moving_recall"] is None else round(o["moving_recall"], 3),
            false_moving_rate=None if o["false_moving_rate"] is None else round(o["false_moving_rate"], 3),
            residual_jitter_m=None if o["centroid_jitter_xy_m"] is None else round(o["centroid_jitter_xy_m"], 3),
            velocity_error_mps=None if o["velocity_error_mps"] is None else round(o["velocity_error_mps"], 3),
            moving_events=len(delays),
            moving_activated=sum(d is not None for d in delays),
            moving_delay_median_s=round(float(np.median([d for d in delays if d is not None])), 2) if any(d is not None for d in delays) else None,
            proposal_states=o["proposal_states"],
        )
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--variant", action="append", required=True, help="NAME=DIR with <recording>/scene.jsonl")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    data = Path(a.data)
    out = Path(a.output)
    require_private_dir(out.parent)
    variants = dict(v.split("=", 1) for v in a.variant)
    result = {}
    for name in RECORDINGS:
        doc = gt.load(data / "annotations" / f"{name}.json")
        types = {k: v["type"] for k, v in doc["objects"].items()}
        inputs = jsonl(data / "outputs/mac/tracker_comparison/inputs" / name / "frames.jsonl")
        old = jsonl(data / "outputs/mac/ground_truth_eval/tracks/baseline" / name / "tracks.jsonl")
        for r, i in zip(old, inputs):
            r["_pose"] = np.asarray(i["robot_pose"]["T_parent_child"])[:2, 3] if i.get("robot_pose") else None
        res = dict(original=dict(
            perception=summarize_eval(evaluate(doc, old, True, inputs)),
            availability=summarize_eval(evaluate(doc, old, True, inputs, include_stale=True)),
            stability=stability(displayed(old, "original"), doc, types),
            latency_ms=dict(kind="tracker only (cached)", mean=round(float(np.mean([r["latency_ms"] for r in old])), 2), p95=round(float(np.percentile([r["latency_ms"] for r in old], 95)), 2)),
        ))  # fmt: skip
        for v, root in variants.items():
            rows = load_scene_light(Path(root) / name / "scene.jsonl")
            summ = json.loads((Path(root) / name / "summary.json").read_text())
            ev = evaluate(doc, rows)
            res[v] = dict(
                perception=summarize_eval(ev),
                availability=summarize_eval(evaluate(doc, rows, include_stale=True)),
                stale_fraction=round(ev["stale_fraction"], 3),
                stability=stability(displayed(rows, "5a"), doc, types),
                latency_ms=dict(kind="5A compute per scan (this run; runs were parallel, see summary)", **{k: round(x, 2) for k, x in summ["processing_ms"].items()}),
                overrides=summ.get("overrides", []),
            )
            print(name, v, "done", flush=True)
        result[name] = res
    out.write_text(json.dumps(result, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
