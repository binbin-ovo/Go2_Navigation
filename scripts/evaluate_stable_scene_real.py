"""Offline annotated diagnostic comparison. Never passes annotations into perception."""

import argparse
from collections import defaultdict, Counter
import json
from pathlib import Path
import hashlib
import numpy as np
from scipy.optimize import linear_sum_assignment
from go2_perception.tracking import ground_truth as gt
from evaluate_ground_truth import detections_odom, proposal_state


def load(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def evaluate(doc, scenes, baseline=False, inputs=None, include_stale=False):
    records = {
        round(
            r["t"] if baseline else r["diagnostics"]["source_midpoint_timestamp"], 3
        ): r
        for r in scenes
    }
    cached = {round(r["timestamp"], 3): r for r in inputs or []}
    counts = defaultdict(Counter)
    residual = defaultdict(list)
    velocity = defaultdict(list)
    presence = defaultdict(list)
    proposals = defaultdict(Counter)
    states = defaultdict(list)
    stale = total = 0
    for f in doc["frames"]:
        r = records.get(round(f["t"], 3))
        if r is None:
            continue
        truth = [
            dict(s, type=doc["objects"][s["id"]]["type"])
            for s in f["objects"]
            if gt.scorable(s)
        ]
        if baseline:
            all_objects = [
                dict(
                    object_id=o[0],
                    center=o[1:4],
                    velocity=o[4:6],
                    motion_state={"M": "MOVING", "S": "STATIONARY", "U": "UNCERTAIN"}[
                        o[8]
                    ],
                    observed=o[9] == 0,
                )
                for o in r["tracks"]
            ]
            dets = (
                detections_odom(cached[round(f["t"], 3)])
                if round(f["t"], 3) in cached
                else []
            )
        else:
            all_objects = [
                dict(o, observed=o["status"] == "observed_filtered")
                for o in r["objects"]
            ]
            dets = []
            for b in r["diagnostics"]["proposal_boxes"]:
                if b is not None:
                    xy = np.asarray(b["center"])[:2]
                    size = np.asarray(b["dimensions"])[:2]
                    dets.append(
                        dict(xy=xy, lo=xy - size / 2, hi=xy + size / 2, size=size)
                    )
        stale += sum(not o["observed"] for o in all_objects)
        total += len(all_objects)
        objects = [o for o in all_objects if o["observed"] or include_stale]
        matches = {}
        if truth and objects:
            cost = np.array(
                [
                    [gt.state_distance(s, o["center"][:2]) for o in objects]
                    for s in truth
                ]
            )
            gates = np.array(
                [0.15 if s["type"] == "static_obstacle" else 0.5 for s in truth]
            )
            gated = np.where(cost <= gates[:, None], cost, 1e6)
            for i, j in zip(*linear_sum_assignment(gated)):
                if gated[i, j] < 1e5:
                    matches[truth[i]["id"]] = objects[j]
        for s in truth:
            key = s["id"]
            c = counts[key]
            c["detectable"] += 1
            o = matches.get(key)
            presence[key].append((f["t"], o is not None))
            n = sum(
                gt.state_distance(s, x["center"][:2])
                <= (0.15 if s["type"] == "static_obstacle" else 0.5)
                for x in objects
            )
            c["duplicate_frames"] += n > 1
            proposals[key][
                proposal_state(s, dets, [v for v in truth if v["id"] != key])
            ] += 1
            moving = s.get("motion") == "moving"
            stationary = s.get("motion") == "stationary"
            c["moving_truth"] += moving
            c["stationary_truth"] += stationary
            states[key].append(
                (
                    f["t"],
                    moving,
                    o is not None,
                    o is not None and o["motion_state"] == "MOVING",
                )
            )
            if o is None:
                continue
            c["observed"] += 1
            c["moving_recalled"] += moving and o["motion_state"] == "MOVING"
            c["stationary_matched"] += stationary
            c["false_moving"] += stationary and o["motion_state"] == "MOVING"
            if stationary:
                center = np.asarray(
                    s["xy"] if s.get("xy") is not None else np.mean(s["region"], axis=0)
                )
                residual[key].append(np.asarray(o["center"])[:2] - center)
            if s.get("velocity") is not None:
                velocity[key].append(
                    float(
                        np.linalg.norm(
                            np.asarray(o["velocity"])[:2]
                            - np.asarray(s["velocity"])[:2]
                        )
                    )
                )
    result = {}
    for key, c in counts.items():
        run = longest = 0
        last_presence = None
        for t, v in presence[key]:
            if last_presence is not None and t - last_presence > 0.2:
                run = 0
            run = run + 1 if v else 0
            longest = max(longest, run)
            last_presence = t
        events = []
        active = None
        previous = False
        last = None
        for t, moving, seen, recalled in states[key]:
            gap = last is not None and t - last > 0.2
            if active is not None and (not moving or gap):
                events.append(active)
                active = None
            if moving and (not previous or gap):
                active = dict(
                    onset_offset_s=t - doc["t0"],
                    detection_delay_s=None,
                    moving_delay_s=None,
                    left_censored=last is None or gap,
                )
            if active is not None:
                onset = active["onset_offset_s"] + doc["t0"]
                if seen and active["detection_delay_s"] is None:
                    active["detection_delay_s"] = t - onset
                if recalled and active["moving_delay_s"] is None:
                    active["moving_delay_s"] = t - onset
            previous = moving
            last = t
        if active is not None:
            events.append(active)
        result[key] = dict(
            counts={k: int(v) for k, v in c.items()},
            coverage=c["observed"] / c["detectable"],
            false_moving_rate=c["false_moving"] / c["stationary_matched"]
            if c["stationary_matched"]
            else None,
            moving_recall=c["moving_recalled"] / c["moving_truth"]
            if c["moving_truth"]
            else None,
            centroid_jitter_xy_m=float(np.linalg.norm(np.std(residual[key], axis=0)))
            if len(residual[key]) > 1
            else None,
            velocity_error_mps=float(np.mean(velocity[key])) if velocity[key] else None,
            longest_observed_fraction=longest / c["detectable"],
            proposal_states=dict(proposals[key]),
            motion_events=events,
        )
    return dict(
        objects=result,
        stale_fraction=stale / total if total else 0.0,
        box_frames=total,
        matching="Hungarian per-frame; .5m person point / .15m box polygon; IDs ignored",
        includes_stale_geometry=include_stale,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotations", required=True)
    p.add_argument("--comparison", required=True)
    p.add_argument("--scenes", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    result = {}
    for name in ["test_a1", "test_a2", "test_b", "test_c"]:
        annotation = Path(a.annotations) / f"{name}.json"
        doc = gt.load(annotation)
        comparison = Path(a.comparison)
        old = load(comparison / "baseline" / name / "tracks.jsonl")
        new = load(Path(a.scenes) / name / "scene.jsonl")
        inputs = load(comparison / "inputs" / name / "frames.jsonl")
        result[name] = dict(
            baseline=evaluate(doc, old, True, inputs),
            stable=evaluate(doc, new),
            baseline_availability=evaluate(doc, old, True, inputs, include_stale=True),
            stable_availability=evaluate(doc, new, include_stale=True),
            annotation_sha256=hashlib.sha256(annotation.read_bytes()).hexdigest(),
            baseline_source_sha256=hashlib.sha256(
                (comparison / "baseline" / name / "tracks.jsonl").read_bytes()
            ).hexdigest(),
            runtime=json.loads((Path(a.scenes) / name / "summary.json").read_text())[
                "processing_ms"
            ],
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
