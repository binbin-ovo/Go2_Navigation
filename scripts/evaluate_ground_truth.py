"""Score tracker variants against local ground-truth annotations and diagnose ID failures.

Inputs (all local): annotations from ``scripts/annotate_ground_truth.py`` and a run of
``scripts/compare_tracking_real.py`` (compact per-variant ``tracks.jsonl`` plus the shared
cached detections it consumed). Outputs: a full local JSON with every failure event, an
aggregate JSON that is safe to commit (no positions, no per-frame data), and LiDAR-only
identity timelines.

``--oracle-segmentation DIR`` writes diagnostic tracker inputs: a re-extraction from the raw
recording in which the real LiDAR points inside each annotated object's gate are removed
before clustering and returned as one proposal (this undoes splits and merges with the
background or the other object, and invents no points). Everything else in the scene is
clustered exactly as before. Replay the tracker variants on it with
``compare_tracking_real.py --cache-root``. It uses ground truth to build inputs, so its
scores are a diagnostic upper bound for better proposals, never a tracker result.
"""

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from go2_perception.tracking import ground_truth as gt  # noqa: E402

LABELS = dict(
    baseline="original (ecfaae3, tracking.yaml)",
    improved="Phase 3.6 (tracking_robustness.yaml)",
    refined="Phase 3.7 (tracking_refinement.yaml)",
    current_default="current code, tracking.yaml",
)
GATES = dict(person=0.5, static_obstacle=0.15, other=0.5)


def load_records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def detections_odom(row):
    """Cached detection centroids and boxes transformed from base_link_level to odom."""
    m = np.asarray(row["robot_pose"]["T_parent_child"])
    out = []
    for d in row["detections"]:
        p = m[:3, :3] @ np.asarray(d["position"]) + m[:3, 3]
        corners = np.array(
            [[x, y, d["bbox_min"][2]] for x in (d["bbox_min"][0], d["bbox_max"][0]) for y in (d["bbox_min"][1], d["bbox_max"][1])]
        )
        c = corners @ m[:3, :3].T + m[:3, 3]
        out.append(dict(xy=p[:2], lo=c[:, :2].min(0), hi=c[:, :2].max(0), size=np.ptp(c[:, :2], 0)))
    return out


def subset(doc, start, end):
    return dict(doc, frames=[f for f in doc["frames"] if start <= f["t"] - doc["t0"] <= end])


def located(states):
    return [s for s in states if gt.scorable(s)]


def proposal_state(s, dets, others):
    """How the LiDAR proposals represent one annotated object in one frame.

    single / split: one / several detection centroids inside the object's gate.
    merged_other_object / merged_background: no centroid in the gate, but the object's centre
    lies inside a larger detection box, whose centroid is / is not near another annotated
    object. no_cluster: no detection covers the object at all.
    """
    n = sum(gt.state_distance(s, d["xy"]) <= GATES[s["type"]] for d in dets)
    if n:
        return "single" if n == 1 else "split"
    c = np.mean(s["region"], axis=0) if s.get("region") else np.asarray(s["xy"])
    covering = [d for d in dets if np.all(d["lo"] - 0.05 <= c) and np.all(c <= d["hi"] + 0.05)]
    if not covering:
        return "no_cluster"
    d = max(covering, key=lambda d: np.prod(d["size"]))
    near_other = any(gt.state_distance(o, d["xy"]) <= 0.6 for o in others)
    return "merged_other_object" if near_other else "merged_background"


def frame_proposals(f, row, types):
    states = [dict(s, type=types[s["id"]]) for s in f["objects"]]
    dets = detections_odom(row)
    found = located(states)
    return {
        s["id"]: proposal_state(s, dets, [o for o in found if o["id"] != s["id"]]) for s in found
    }


def proposal_quality(doc, inputs):
    """Per object: how often the cached LiDAR proposals are single, split, merged or absent."""
    rows = {round(r["timestamp"], 3): r for r in inputs}
    types = {k: v["type"] for k, v in doc["objects"].items()}
    out = collections.defaultdict(collections.Counter)
    for f in doc["frames"]:
        row = rows.get(round(f["t"], 3))
        if row is not None:
            for oid, state in frame_proposals(f, row, types).items():
                out[oid]["frames"] += 1
                out[oid][state] += 1
    return {k: dict(v) for k, v in out.items()}


def oracle_segmentation(args):
    """Re-extract inputs with each annotated object's own points as a single proposal."""
    import importlib.util

    from go2_perception.core.geometry import aabb
    from go2_perception.core.tracking_types import Detection3D
    from go2_perception.tracking import replay

    spec = importlib.util.spec_from_file_location("compare_tracking_real", ROOT / "scripts/compare_tracking_real.py")
    compare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compare)
    original = replay._tracking_frame
    ground_z = -0.32 + 0.06  # go2_lidar.yaml expected ground height plus the RANSAC distance
    # Capture every point inside the matching gate (people: slightly less than the gate, so a
    # wall behind the person is not pulled in); obstacles are processed before people.
    point_radius = dict(person=0.45, static_obstacle=GATES["static_obstacle"], other=0.45)
    for name in args.recordings:
        doc = gt.load(Path(args.annotations) / f"{name}.json")
        types = {k: v["type"] for k, v in doc["objects"].items()}
        frames = {round(f["t"], 3): f for f in doc["frames"]}
        stats = collections.Counter()

        def patched(processor, t, message, robot_pose, reset=False, outage=None, frames=frames, types=types, stats=stats):
            f = frames.get(round(t, 3))
            if f is None or robot_pose is None:
                return original(processor, t, message, robot_pose, reset, outage)
            m = robot_pose.T_parent_child
            pts = message.points
            odom = pts @ m[:3, :3].T + m[:3, 3]
            keep = np.ones(len(pts), bool)
            groups = []
            objects = located([dict(s, type=types[s["id"]]) for s in f["objects"]])
            for s in sorted(objects, key=lambda s: s["type"] != "static_obstacle"):
                r = point_radius[s["type"]]
                inside = keep & (pts[:, 2] > ground_z)
                inside &= gt.distances(s, odom) <= r
                keep &= ~inside
                if inside.sum() >= 3:
                    groups.append((s["id"], pts[inside]))
                stats["objects"] += 1
                stats["empty" if inside.sum() < 3 else "proposals"] += 1
            from dataclasses import replace

            frame = original(processor, t, replace(message, points=pts[keep]), robot_pose, reset, outage)
            for oid, p in groups:
                low, high, center = aabb(p)
                frame.detections.append(
                    Detection3D(t, center, message.frame_id, message.clock_domain, f"oracle_{oid}", low, high)
                )
            return frame

        replay._tracking_frame = patched
        try:
            target = Path(args.oracle_segmentation) / name / "frames.jsonl"
            compare.extract(Path(args.recordings_root) / name, target)
        finally:
            replay._tracking_frame = original
        print(name, dict(stats), flush=True)


def events(doc, records, inputs, selection):
    """Every change of the matched ID of an annotated object, with what the proposals and the
    tracker did between the last frame with the old ID and the first with the new one."""
    rows = {round(r["timestamp"], 3): r for r in inputs}
    types = {k: v["type"] for k, v in doc["objects"].items()}
    births = {}
    for r in records:
        for t in r["tracks"]:
            births.setdefault(t[0], r["t"])
    history = collections.defaultdict(list)  # object -> [(t, status, matched row, proposal, near ids)]
    for f, rec, states, _, matches, near in gt.match_sequence(doc, records, selection, GATES):
        row = rows.get(round(f["t"], 3))
        if row is None:
            continue
        proposals = frame_proposals(f, row, types)
        for s in states:
            ids = {r[0] for r in near.get(s["id"], [])}
            history[s["id"]].append((f["t"], s["status"], matches.get(s["id"]), proposals.get(s["id"]), ids, rec))
    found = []
    for oid, h in history.items():
        last = None
        for i, (t, status, row, _, _, rec) in enumerate(h):
            if row is None:
                continue
            if last is not None and row[0] != last[1][0]:
                j = last[0]
                old, new = last[1][0], row[0]
                span = h[j + 1 : i + 1]
                statuses = {x[1] for x in h[j : i + 1]}
                proposals = collections.Counter(x[3] for x in span if x[3] is not None)
                old_row = next((tr for tr in rec["tracks"] if tr[0] == old), None)
                # Was the new ID already a second track on this object (within the last 1 s)?
                coexisted = any(new in x[4] for x in h[:i] if t - x[0] <= 1.0)
                if "out_of_view" in statuses:
                    proposal = "exit (out of view)"
                elif "occluded" in statuses:
                    proposal = "occluded"
                elif proposals.get("split"):
                    proposal = "split"
                elif proposals.get("merged_other_object"):
                    proposal = "merged with other annotated object"
                elif proposals.get("merged_background"):
                    proposal = "merged with background"
                elif proposals.get("no_cluster"):
                    proposal = "no cluster"
                else:
                    proposal = "clean"
                if old_row is None:
                    tracker = "old ID deleted"
                elif coexisted:
                    tracker = "duplicate track took over" + (" (old Lost)" if old_row[7] == "L" else "")
                else:
                    tracker = "new track" + (" (old Lost)" if old_row[7] == "L" else " (old still Confirmed)")
                found.append(
                    dict(
                        object=oid,
                        t=round(t - doc["t0"], 2),
                        t_abs=t,
                        gap_s=round(t - h[j][0], 2),
                        old=old,
                        new=new,
                        new_age_s=round(t - births[new], 2),
                        returning=new in {x[2][0] for x in h[:j] if x[2] is not None},
                        proposal=proposal,
                        proposals=dict(proposals),
                        tracker=tracker,
                    )
                )
            last = (i, row)
    return found


def cross_object_ids(doc, records, selection):
    """Track IDs matched to more than one annotated object (identity transfer between objects)."""
    owners = collections.defaultdict(collections.Counter)
    for *_, matches, _ in gt.match_sequence(doc, records, selection, GATES):
        for oid, row in matches.items():
            owners[row[0]][oid] += 1
    return {tid: dict(c) for tid, c in owners.items() if len(c) > 1}


def robot_phase(doc, inputs):
    """Per annotated frame: still / rotating / translating from consecutive odometry poses."""
    rows = sorted(inputs, key=lambda r: r["timestamp"])
    t = np.array([r["timestamp"] for r in rows])
    m = np.array([r["robot_pose"]["T_parent_child"] for r in rows])
    yaw = np.unwrap(np.arctan2(m[:, 1, 0], m[:, 0, 0]))
    xy = m[:, :2, 3]
    out = {}
    for f in doc["frames"]:
        i = int(np.argmin(np.abs(t - f["t"])))
        a, b = max(i - 3, 0), min(i + 3, len(t) - 1)
        dt = max(t[b] - t[a], 1e-6)
        w = abs(yaw[b] - yaw[a]) / dt
        v = np.linalg.norm(xy[b] - xy[a]) / dt
        out[round(f["t"], 3)] = "rotating" if w > 0.15 else "translating" if v > 0.08 else "still"
    return out


def phase_proposals(doc, inputs, phase):
    """Proposal states and in-gate centroid offsets from the annotated centre, per robot phase
    (registration check for odometry: a static object's proposals should not move)."""
    rows = {round(r["timestamp"], 3): r for r in inputs}
    types = {k: v["type"] for k, v in doc["objects"].items()}
    states = collections.defaultdict(collections.Counter)
    offsets = collections.defaultdict(list)
    for f in doc["frames"]:
        row = rows.get(round(f["t"], 3))
        if row is None:
            continue
        p = phase[round(f["t"], 3)]
        for oid, state in frame_proposals(f, row, types).items():
            states[f"{oid} {p}"][state] += 1
        for s in located([dict(s, type=types[s["id"]]) for s in f["objects"]]):
            if s.get("region"):
                c = np.mean(s["region"], axis=0)
                for d in detections_odom(row):
                    if gt.state_distance(s, d["xy"]) <= GATES[s["type"]]:
                        offsets[f"{s['id']} {p}"].append(d["xy"] - c)
    return {
        k: dict(
            states=dict(states[k]),
            offset_median_xy=np.median(offsets[k], axis=0).round(3).tolist() if offsets[k] else None,
            offset_spread_m=float(np.median(np.linalg.norm(np.asarray(offsets[k]) - np.median(offsets[k], axis=0), axis=1))) if offsets[k] else None,
        )
        for k in states
    }


def timeline(out, name, doc, results, variants):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    objects = list(doc["objects"])
    fig, axes = plt.subplots(len(objects), 1, figsize=(13, 1.2 + 1.1 * len(variants) * len(objects)), squeeze=False)
    cmap = plt.get_cmap("tab20")
    for ax, oid in zip(axes[:, 0], objects):
        for k, label in enumerate(variants):
            seq = results[label]["sequence"][oid]
            order = {}
            for t, status, tid in seq:
                if status not in gt.DETECTABLE:
                    ax.barh(k, 0.07, left=t, height=0.8, color="0.85")
                    continue
                if tid is None:
                    ax.barh(k, 0.07, left=t, height=0.25, color="k")
                    continue
                ax.barh(k, 0.07, left=t, height=0.6, color=cmap(order.setdefault(tid, len(order)) % 20))
        ax.set_yticks(range(len(variants)), [LABELS.get(v, v).split(" (")[0] for v in variants])
        ax.set_title(f"{name} / {oid}: colour = matched track ID (order of appearance); black = detectable but unmatched; grey = occluded or out of view", fontsize=9)
        ax.set_xlabel("time since first LiDAR frame (s)")
    fig.tight_layout()
    path = out / f"gt_timeline_{name}.png"
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def sequences(doc, records, selection):
    seq = collections.defaultdict(list)
    for f, _, states, _, matches, _ in gt.match_sequence(doc, records, selection, GATES):
        for s in states:
            row = matches.get(s["id"])
            seq[s["id"]].append((f["t"] - doc["t0"], s["status"], None if row is None else row[0]))
    return seq


def strip(result):
    """Aggregate-only copy: drop recovery event times and IDs."""
    r = json.loads(json.dumps(result, default=float))
    for o in r["objects"].values():
        rec = o.pop("recovery")
        o["recovery_events"] = len(rec)
        o["recovery_kept"] = sum(e["kept"] for e in rec)
        o["recovery_by_kind"] = dict(collections.Counter(e["kind"] for e in rec))
    return r


def combine(args):
    """Committed metrics file: tracker and oracle aggregates plus run fingerprints."""
    tracker, oracle = (json.loads(Path(p).read_text()) for p in args.combine[:2])
    runs = [json.loads((Path(p) / "run.json").read_text()) for p in args.combine[2:4]]
    doc = dict(
        description="Phase 3.8 annotated evaluation on the private Go2 recordings (aggregate only; "
        "no positions, frames or images). See docs/TRACKING_GROUND_TRUTH.md.",
        annotation=dict(
            schema=gt.SCHEMA,
            annotator="Claude (Opus 5.5): LiDAR background subtraction, reviewed on BEV+camera "
            "contact sheets; not human-verified",
        ),
        tracker_runs={k: runs[0][k] for k in ("baseline_commit", "working_commit", "working_tree_dirty", "input_sha256")},
        meta=tracker["meta"],
        trackers=tracker["recordings"],
        oracle_segmentation_diagnostic=dict(
            note="Tracker inputs re-extracted with each annotated object's own LiDAR points (inside "
            "its matching gate) removed before clustering and returned as one proposal. Uses ground "
            "truth to build inputs: an upper bound for better proposals, not a tracker result.",
            input_sha256=runs[1]["input_sha256"],
            recordings=oracle["recordings"],
        ),
    )
    text = json.dumps(doc, indent=1)
    if "/Users" in text or "/home" in text:
        raise ValueError("Local paths in aggregate output")
    Path(args.combine[4]).write_text(text + "\n")
    print("Saved", args.combine[4])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--annotations", default="data/annotations")
    p.add_argument("--tracks", default="data/outputs/mac/ground_truth_eval/tracks")
    p.add_argument("--inputs", default="data/outputs/mac/tracker_comparison/inputs")
    p.add_argument("--variants", nargs="+", default=["baseline", "improved", "refined"])
    p.add_argument("--recordings", nargs="+", default=["test_b", "test_a2", "test_c", "test_a1"])
    p.add_argument("--output", default="data/outputs/mac/ground_truth_eval/evaluation.json")
    p.add_argument("--aggregate", help="also write an aggregate-only JSON (safe to commit)")
    p.add_argument("--plots", help="directory for LiDAR-only identity timelines")
    p.add_argument("--oracle-segmentation", help="write oracle-segmentation diagnostic inputs here and exit")
    p.add_argument("--recordings-root", default="data/recordings")
    p.add_argument(
        "--combine",
        nargs=5,
        metavar=("TRACKER_AGG", "ORACLE_AGG", "TRACKS_DIR", "ORACLE_TRACKS_DIR", "OUT"),
        help="write the committed metrics file from two aggregates and their runs, then exit",
    )
    args = p.parse_args()
    if args.combine:
        combine(args)
        return
    if args.oracle_segmentation:
        oracle_segmentation(args)
        return
    full, aggregate = {}, {}
    for name in args.recordings:
        doc = gt.load(Path(args.annotations) / f"{name}.json")
        inputs = load_records(Path(args.inputs) / name / "frames.jsonl")
        phase = robot_phase(doc, inputs)
        entry = dict(
            annotated_frames=len(doc["frames"]),
            intervals=doc["intervals"],
            objects=doc["objects"],
            annotation_counts={
                oid: dict(collections.Counter(s["status"] for f in doc["frames"] for s in f["objects"] if s["id"] == oid))
                for oid in doc["objects"]
            },
            motion_label_counts={
                oid: dict(collections.Counter(str(s.get("motion")) for f in doc["frames"] for s in f["objects"] if s["id"] == oid))
                for oid in doc["objects"]
            },
            proposal_quality=proposal_quality(doc, inputs),
            robot_phase_frames=dict(collections.Counter(phase.values())),
            variants={},
        )
        if any(x != "still" for x in phase.values()):
            entry["proposals_by_phase"] = phase_proposals(doc, inputs, phase)
        agg = dict(
            {k: v for k, v in entry.items() if k not in ("variants", "objects")},
            objects={k: v["type"] for k, v in doc["objects"].items()},
            variants={},
        )
        plot_data = {}
        for label in args.variants:
            records = load_records(Path(args.tracks) / label / name / "tracks.jsonl")
            v = dict(
                confirmed=gt.evaluate(doc, records, "confirmed", GATES),
                observed=gt.evaluate(doc, records, "observed", GATES),
                intervals={
                    i["name"]: gt.evaluate(subset(doc, i["start"], i["end"]), records, "confirmed", GATES)
                    for i in doc["intervals"]
                },
                events=events(doc, records, inputs, "confirmed"),
            )
            for e in v["events"]:
                e["robot_phase"] = phase[round(e["t_abs"], 3)]
            if any(x != "still" for x in phase.values()):
                by_phase = collections.defaultdict(list)
                for f in doc["frames"]:
                    by_phase[phase[round(f["t"], 3)]].append(f)
                v["phases"] = {
                    k: gt.evaluate(dict(doc, frames=fr), records, "confirmed", GATES) for k, fr in by_phase.items()
                }
                v["phase_frames"] = {k: len(fr) for k, fr in by_phase.items()}
            entry["variants"][label] = v
            plot_data[label] = dict(sequence=sequences(doc, records, "confirmed"))
            v["cross_object_ids"] = cross_object_ids(doc, records, "confirmed")
            causes = collections.Counter((e["object"], e["proposal"], e["tracker"]) for e in v["events"])
            agg["variants"][label] = dict(
                confirmed=strip(v["confirmed"]),
                observed=strip(v["observed"]),
                intervals={k: strip(x) for k, x in v["intervals"].items()},
                id_changes={f"{o} | {p} | {tr}": n for (o, p, tr), n in sorted(causes.items())},
                id_changes_returning=sum(e["returning"] for e in v["events"]),
                cross_object_ids=len(v["cross_object_ids"]),
                cross_object_frames=sum(min(c.values()) for c in v["cross_object_ids"].values()),
                **(
                    {
                        "phases": {k: strip(x) for k, x in v["phases"].items()},
                        "phase_frames": v["phase_frames"],
                        "id_changes_by_phase": dict(collections.Counter(e["robot_phase"] for e in v["events"])),
                    }
                    if "phases" in v
                    else {}
                ),
            )
        full[name], aggregate[name] = entry, agg
        if args.plots:
            Path(args.plots).mkdir(parents=True, exist_ok=True)
            print(timeline(Path(args.plots), name, doc, plot_data, args.variants))
        print("Evaluated", name, flush=True)
    meta = dict(
        gates_m=GATES,
        variants={v: LABELS.get(v, v) for v in args.variants},
        tracks=str(args.tracks),
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(dict(meta=meta, recordings=full), indent=1, default=float) + "\n")
    print("Saved", args.output)
    if args.aggregate:
        Path(args.aggregate).write_text(json.dumps(dict(meta=dict(meta, tracks="local"), recordings=aggregate), indent=1, default=float) + "\n")
        print("Saved", args.aggregate)


if __name__ == "__main__":
    main()
