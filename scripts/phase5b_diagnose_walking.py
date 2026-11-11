"""Debug-mode diagnosis: why does Phase 5A miss (or not mark MOVING) a walking person?

For every frame where the annotation says the person is visible and walking, the person
cylinder (0.45 m radius, floor+0.05..floor+2.0 m) selects points from the recorded 5A
intermediates (``phase5b_scene_tap.py`` output). Evaluation only: annotations are read after
perception and never influence it. Writes JSON with per-frame attributions and summaries.

  PYTHONPATH=src .venv/bin/python scripts/phase5b_diagnose_walking.py --recording test_b \
      --data ../Go2_Navigation/data --output ../Go2_Navigation/data/outputs/phase5b/diagnosis/test_b.json
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from go2_perception.diagnostics.attribution import attribute_miss, attribute_not_moving, in_cylinder  # noqa: E402
from go2_perception.tracking import ground_truth as gt  # noqa: E402
from go2_perception.viewer import sources as S  # noqa: E402
from go2_perception.viewer.privacy import require_private_dir  # noqa: E402

RADIUS, Z_LO, Z_HI = 0.45, 0.05, 2.0
GATES = dict(person=0.5, static_obstacle=0.15)


def ng_window(tap, k, window_s):
    t = tap.frame_t[k]
    j = k
    while j > 0 and tap.frame_t[j - 1] >= t - window_s - 0.1:
        j -= 1
    a, b = tap.scan_off[j], tap.scan_off[k + 1]
    keep = (tap.scan_acq[a:b] >= t - window_s) & tap.scan_ng[a:b]
    return tap.scan_xyz[a:b][keep].astype(float)


def match(states, candidates):
    """Hungarian per frame (IDs ignored), same gates as the 5A evaluator."""
    out = {}
    if not states or not candidates:
        return out
    cost = np.array([[gt.state_distance(s, c[:2]) for c in candidates] for s in states])
    gates = np.array([GATES.get(s["type"], 0.5) for s in states])
    gated = np.where(cost <= gates[:, None], cost, 1e6)
    for i, j in zip(*linear_sum_assignment(gated)):
        if gated[i, j] < 1e5:
            out[states[i]["id"]] = j
    return out


def interval_of(doc, t_rel):
    for iv in doc.get("intervals", []):
        if iv["start"] <= t_rel <= iv["end"]:
            return iv["name"]
    return "other"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--recording", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--tap", help="tap directory (default outputs/phase5b/tap/<recording>)")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    data = Path(a.data)
    tap_dir = Path(a.tap) if a.tap else data / "outputs" / "phase5b" / "tap" / a.recording
    out = Path(a.output)
    require_private_dir(out.parent)
    frames = S.load_scene(tap_dir / "scene.jsonl")
    tap = S.Tap(tap_dir)
    cfg = json.loads((tap_dir / "summary.json").read_text())["config"]
    min_points = cfg["proposals"]["min_points"]
    support_min = cfg.get("current_support_min", 3)
    doc = gt.load(data / "annotations" / f"{a.recording}.json")
    gt_frames = {round(f["t"], 3): f["objects"] for f in doc["frames"]}
    types = {k: v["type"] for k, v in doc["objects"].items()}
    p3 = S.Phase3(data / "outputs/mac/tracker_comparison/inputs" / a.recording, data / "outputs/mac/ground_truth_eval/tracks/baseline" / a.recording / "tracks.jsonl")
    rows = []
    for k, f in enumerate(frames):
        states = [dict(s, type=types[s["id"]]) for s in gt_frames.get(round(f.source_time, 3), []) if gt.scorable(s)]
        person = next((s for s in states if s["type"] == "person" and s.get("xy") is not None), None)
        if person is None or person.get("motion") != "moving" or f.pose is None:
            continue
        t_rel = f.source_time - doc["t0"]
        xy = np.asarray(person["xy"], float)
        floor = f.pose[2, 3] - 0.32
        zl, zh = floor + Z_LO, floor + Z_HI
        ti = tap.index(f.timestamp)
        scan, _, _ = tap.scan(ti)
        win, codes = tap.window(ti)
        sel_w = in_cylinder(win, xy, RADIUS, zl, zh)
        ng03 = ng_window(tap, ti, 0.3)
        props = []
        for pr in tap.records[ti]["proposals"]:
            c = (np.asarray(pr["low"]) + np.asarray(pr["high"])) / 2
            props.append(dict(dist=float(np.hypot(*(c[:2] - xy))), support=pr["support"]))
        fresh = [o for o in f.objects if o.observed]
        m = match(states, [o.center for o in fresh])
        obj = fresh[m["person"]] if "person" in m else None
        # original 0.3 s pipeline at the same LiDAR midpoint
        j = p3.index(f.source_time)
        orig_fresh = orig_moving = None
        if j is not None:
            obs = [t for t in p3.tracks(j) if t["observed"] and t["status"] == "C"]
            mo = match(states, [np.asarray(t["xyz"]) for t in obs])
            orig_fresh = "person" in mo
            orig_moving = orig_fresh and obs[mo["person"]]["motion"] == "MOVING"
        rec = dict(
            t_rel=round(t_rel, 3), stamp=f.source_time, interval=interval_of(doc, t_rel),
            scan_points=int(in_cylinder(scan, xy, RADIUS, zl, zh).sum()),
            window_points=int(sel_w.sum()), ng03_points=int(in_cylinder(ng03, xy, RADIUS, zl, zh).sum()),
            fresh=obj is not None, moving=obj is not None and obj.motion_state == "MOVING",
            stale_near=any(not o.observed and np.hypot(*(o.center[:2] - xy)) <= 0.5 for o in f.objects),
            orig_fresh=orig_fresh, orig_moving=orig_moving,
        )  # fmt: skip
        if obj is None:
            ev = dict(scan_n=rec["scan_points"], win_codes=codes[sel_w], ng03_n=rec["ng03_points"], proposals=props)
            rec["category"], rec["detail"] = attribute_miss(ev, min_points, support_min)
        elif not rec["moving"]:
            internals = tap.records[ti]["internals"].get(str(obj.object_id))
            rec["object_id"] = obj.object_id
            rec["category"], rec["detail"] = attribute_not_moving(internals, cfg["objects"]["activation_frames"]) if internals else ("motion_unknown", "")
        else:
            rec["category"], rec["detail"] = "recalled_moving", ""
        rows.append(rec)
    n = len(rows)
    present = [r for r in rows if r["scan_points"] >= support_min]
    miss = [r for r in present if not r["fresh"]]

    def summary(sel):
        c = Counter(r["category"] for r in sel)
        return {k: dict(frames=v, fraction=round(v / max(len(sel), 1), 3)) for k, v in c.most_common()}

    by_interval = defaultdict(list)
    for r in rows:
        by_interval[r["interval"]].append(r)
    # contiguous failure episodes (same category, gaps <= 0.2 s) for montage selection
    episodes = []
    for r in rows:
        if r["category"] in ("recalled_moving",):
            continue
        e = episodes[-1] if episodes else None
        if e and e["category"] == r["category"] and r["t_rel"] - e["end"] <= 0.2:
            e["end"] = r["t_rel"]
            e["frames"] += 1
        else:
            episodes.append(dict(category=r["category"], start=r["t_rel"], end=r["t_rel"], frames=1, detail=r["detail"], interval=r["interval"]))
    result = dict(
        recording=a.recording,
        tap=str(tap_dir),
        note="Evaluation-only attribution; annotations (LiDAR-derived, not human-verified) select points after perception.",
        person_cylinder=dict(radius_m=RADIUS, z_above_floor=[Z_LO, Z_HI], floor="robot z - 0.32 m"),
        walking_frames=n,
        present_in_current_scan=len(present),
        fresh_coverage_walking=round(sum(r["fresh"] for r in rows) / max(n, 1), 3),
        moving_recall_walking=round(sum(r["moving"] for r in rows) / max(n, 1), 3),
        original_fresh_coverage_walking=round(np.mean([bool(r["orig_fresh"]) for r in rows]), 3) if n else None,
        original_moving_recall_walking=round(np.mean([bool(r["orig_moving"]) for r in rows]), 3) if n else None,
        missed_while_present=len(miss),
        missed_while_present_original_had_fresh=sum(bool(r["orig_fresh"]) for r in miss),
        miss_attribution=summary(miss),
        all_frame_categories=summary(rows),
        not_moving_attribution=summary([r for r in rows if r["fresh"] and not r["moving"]]),
        by_interval={k: dict(frames=len(v), fresh=round(np.mean([r["fresh"] for r in v]), 3), moving=round(np.mean([r["moving"] for r in v]), 3), categories=dict(Counter(r["category"] for r in v))) for k, v in by_interval.items()},
        episodes=sorted(episodes, key=lambda e: -e["frames"]),
        frames=rows,
    )  # fmt: skip
    out.write_text(json.dumps(result, indent=1, default=float) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("frames", "episodes")}, indent=1, default=float))


if __name__ == "__main__":
    main()
