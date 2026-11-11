"""Compare current evidence and bounded recent evidence separately; annotations never enter runtime."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from evaluate_stable_scene_real import evaluate
from go2_perception.tracking import ground_truth as gt


def compact_rows(path):
    rows = []
    with Path(path).open() as f:
        for line in f:
            r = json.loads(line)
            rows.append(
                dict(
                    timestamp=r["timestamp"],
                    objects=r["objects"],
                    diagnostics=r["diagnostics"],
                )
            )
    return rows


def recent_rows(rows, horizon=0.2):
    result = deepcopy(rows)
    for r in result:
        for o in r["objects"]:
            if o["observation_age_s"] <= horizon:
                o["status"] = (
                    "observed_filtered"  # Evaluation view ONLY; never serialize this as a scene.
                )
                evidence = o.get("motion_evidence", {})
                if (
                    evidence.get("valid")
                    and r["timestamp"] - evidence.get("updated_at", -1e9) <= horizon
                ):
                    o["motion_state"] = evidence["state"]
    return result


def identity_reliability(doc, rows):
    lookup = {round(r["diagnostics"]["source_midpoint_timestamp"], 3): r for r in rows}
    previous = {}
    switches = 0
    moving_after = 0
    moving_switches = 0
    gaps = []
    for f in doc["frames"]:
        r = lookup.get(round(f["t"], 3))
        if r is None:
            continue
        truth = [
            dict(s, type=doc["objects"][s["id"]]["type"])
            for s in f["objects"]
            if gt.scorable(s)
        ]
        objects = [o for o in r["objects"] if o["status"] == "observed_filtered"]
        if not truth or not objects:
            continue
        costs = np.array(
            [[gt.state_distance(s, o["center"][:2]) for o in objects] for s in truth]
        )
        gates = np.array(
            [0.15 if s["type"] == "static_obstacle" else 0.5 for s in truth]
        )
        costs = np.where(costs <= gates[:, None], costs, 1e6)
        for i, j in zip(*linear_sum_assignment(costs)):
            if costs[i, j] >= 1e5:
                continue
            s, o = truth[i], objects[j]
            prior = previous.get(s["id"])
            if prior is not None and prior[0] != o["object_id"]:
                switches += 1
                gaps.append(f["t"] - prior[1])
                if s.get("motion") == "moving":
                    moving_switches += 1
                    moving_after += o["motion_state"] == "MOVING"
            previous[s["id"]] = (o["object_id"], f["t"])
    return dict(
        matched_id_changes=switches,
        moving_id_changes=moving_switches,
        moving_label_at_first_observation_after_change=moving_after,
        mean_gap_s=float(np.mean(gaps)) if gaps else None,
        warning="Per-frame geometric matches only; not verified semantic identity.",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenes", required=True)
    p.add_argument("--annotations", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    results = {}
    for name in ["test_a1", "test_a2", "test_b", "test_c"]:
        root = Path(a.scenes) / name
        rows = compact_rows(root / "scene.jsonl")
        doc = gt.load(Path(a.annotations) / (name + ".json"))
        fresh = evaluate(doc, rows)
        recent = evaluate(doc, recent_rows(rows))
        available = evaluate(doc, rows, include_stale=True)
        results[name] = dict(
            current_direct=fresh,
            recent_0p2s=recent,
            available=available,
            identity=identity_reliability(doc, rows),
            runtime=json.loads((root / "summary.json").read_text()),
            self_excluded_points=sum(
                r["diagnostics"].get("self_excluded_points", 0) for r in rows
            ),
            sector_states={},
        )
        for r in rows:
            for o in r["objects"]:
                k = o.get("sector_evidence", {}).get("state", "unavailable")
                results[name]["sector_states"][k] = (
                    results[name]["sector_states"].get(k, 0) + 1
                )
    out.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
