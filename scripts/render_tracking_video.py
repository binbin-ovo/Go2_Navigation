"""Render synchronized RGB + LiDAR BEV + status diagnostic videos from private recordings.

All inputs are existing local outputs; nothing is re-detected or invented:
- tracker input frames (clusters, odometry poses) from the shared comparison cache;
- tracker outputs (compact ``tracks.jsonl``) from ``compare_tracking_real.py`` runs;
- raw LiDAR points from the annotation cache (``annotate_ground_truth.py cache``);
- camera frames and YOLO detections from ``go2_replay.py``;
- optional ground truth (``data/annotations/<recording>.json``) and its evaluation, which
  supplies ID switches confirmed against ground truth.

Outputs are home imagery: they must go to a Git-ignored path (checked before writing).

Examples:
  render_tracking_video.py --recording test_b --output data/outputs/mac/videos/test_b_original.mp4
  render_tracking_video.py --recording test_c --compare baseline improved refined \
      --output data/outputs/mac/videos/test_c_compare.mp4
"""

import argparse
import bisect
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from go2_perception.tracking import ground_truth as gt  # noqa: E402
from go2_perception.visualization import diagnostic_video as dv  # noqa: E402

VARIANT_LABEL = dict(
    baseline="original tracker (tracking.yaml, ecfaae3)",
    current_default="current default (tracking.yaml)",
    improved="Phase 3.6 (tracking_robustness.yaml)",
    refined="Phase 3.7 (tracking_refinement.yaml)",
)
GATES = dict(person=0.5, static_obstacle=0.15, other=0.5)


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


class Run:
    """Everything needed to render one recording with one or more tracker variants."""

    def __init__(self, args, variants):
        name = args.recording
        self.name = name
        self.inputs = jsonl(Path(args.inputs) / name / "frames.jsonl")
        self.times = [r["timestamp"] for r in self.inputs]
        self.t0 = self.times[0]
        self.records = {v: jsonl(Path(args.tracks) / v / name / "tracks.jsonl") for v in variants}
        for v, recs in self.records.items():
            if [round(r["t"], 6) for r in recs] != [round(t, 6) for t in self.times]:
                raise ValueError(f"{v}: tracker records do not match the input frames")
        cache = np.load(Path(args.annotations) / "cache" / f"{name}.npz")
        self.cloud_t, self.cloud_pts, self.cloud_off = cache["t"], cache["pts"], cache["off"]
        self.host_offset = float(cache["host_offset"])
        rec_dir = Path(args.recordings) / name
        self.rgb_index = jsonl(rec_dir / "rgb" / "index.jsonl")
        self.rgb_dir = rec_dir / "rgb"
        dets = json.loads((Path(args.replay_root) / f"{name}_replay" / "rgb_detections.json").read_text())
        self.rgb_dets = {d["file"]: d["detections"] for d in dets}
        self.rgb_idx, self.rgb_dt = dv.align_rgb(
            self.times, [r["host_recv_first"] for r in self.rgb_index], self.host_offset
        )
        odom = jsonl(rec_dir / "state" / "utlidar_robot_odom.jsonl")
        self.odom_t = np.array([o["stamp"] for o in odom])
        self.odom = odom
        report = json.loads((Path(args.inputs) / name / "pose_report.json").read_text())
        self.skipped = [s["timestamp"] for s in report.get("skipped", [])]
        self.poses = [np.asarray(r["robot_pose"]["T_parent_child"]) if r.get("robot_pose") else None for r in self.inputs]
        self.gt_doc = None
        gt_path = Path(args.annotations) / f"{name}.json"
        if gt_path.exists() and not args.no_ground_truth:
            self.gt_doc = gt.load(gt_path)
            self.gt_frames = {round(f["t"], 3): f["objects"] for f in self.gt_doc["frames"]}
        self.key = {}
        self.gt_events = {}
        eval_path = Path(args.evaluation)
        if self.gt_doc is not None and eval_path.exists():
            ev = json.loads(eval_path.read_text())["recordings"].get(name, {}).get("variants", {})
            for v in variants:
                at = {round(r["t"], 3): {row[0]: (row[1], row[2]) for row in r["tracks"]} for r in self.records[v]}
                self.gt_events[v] = [
                    dv.Event(
                        e["t_abs"],
                        "ID SWITCH vs GT",
                        f"ID SWITCH vs GT: {e['object']} #{e['old']} -> #{e['new']} (proposals: {e['proposal']})",
                        True,
                        at.get(round(e["t_abs"], 3), {}).get(e["new"]),
                        (e["old"], e["new"]),
                    )
                    for e in ev.get(v, {}).get("events", [])
                    if "exit" not in e["proposal"]
                ]

    def roi(self, t, x, y):
        k = min(bisect.bisect_left(self.times, t), len(self.times) - 1)
        pose = self.poses[k]
        return pose is None or np.hypot(x - pose[0, 3], y - pose[1, 3]) <= 3.5

    def key_ids(self, v):
        """Tracks matched to an annotated object at any time; without annotations, tracks that
        stay Confirmed for 3 s inside the region of interest."""
        if self.gt_doc is not None:
            return {row[0] for *_, matches, _ in gt.match_sequence(self.gt_doc, self.records[v], "confirmed", GATES) for row in matches.values()}
        first, last = {}, {}
        for r in self.records[v]:
            for row in r["tracks"]:
                if row[7] == "C" and self.roi(r["t"], row[1], row[2]):
                    first.setdefault(row[0], r["t"])
                    last[row[0]] = r["t"]
        return {i for i in first if last[i] - first[i] >= 3.0}

    def events(self, v):
        recs = [dict(t=r["t"], tracks=r["tracks"]) for r in self.records[v]]
        key = self.key[v]
        ev = [e for e in dv.heuristic_events(recs, roi=self.roi) if set(e.ids) & key]
        ev += self.gt_events.get(v, [])
        ev += [dv.Event(t, "ODOMETRY GAP", "ODOMETRY GAP: LiDAR frame skipped (no bracketing odometry)", False) for t in self.skipped]
        gaps = np.where(np.diff(self.odom_t) > 0.1)[0]
        ev += [dv.Event(float(self.odom_t[i]), "ODOMETRY GAP", f"ODOMETRY GAP {np.diff(self.odom_t)[i] * 1000:.0f} ms", False) for i in gaps]
        ev += self.uncertain_motion_events(v)
        return sorted(ev, key=lambda e: e.t)

    def uncertain_motion_events(self, v):
        """UNCERTAIN MOTION where ground truth says the person walks but the matched track
        stays UNCERTAIN for at least 1 s (GT-supported, so marked confirmed)."""
        if self.gt_doc is None:
            return []
        out, start = [], None
        for f, rec, states, _, matches, _ in gt.match_sequence(self.gt_doc, self.records[v], "confirmed", GATES):
            s = next((s for s in states if s["type"] == "person"), None)
            row = matches.get(s["id"]) if s else None
            bad = s is not None and s.get("motion") == "moving" and row is not None and row[8] == "U"
            if bad and start is None:
                start = f["t"]
            if not bad:
                start = None
            if bad and f["t"] - start >= 1.0 and (not out or out[-1].t < start):
                out.append(dv.Event(f["t"], "UNCERTAIN MOTION", f"UNCERTAIN MOTION: walking person, track #{row[0]} UNCERTAIN for 1 s", True, (row[1], row[2])))
        return out

    def points(self, k, pose):
        t = self.times[k]
        a = bisect.bisect_left(self.cloud_t.tolist(), t - 0.3)
        b = bisect.bisect_right(self.cloud_t.tolist(), t)
        if b <= a:
            return np.empty((0, 2))
        p = self.cloud_pts[self.cloud_off[a] : self.cloud_off[b]]
        z0 = pose[2, 3] if pose is not None else float(np.median(p[:, 2]))
        keep = (p[:, 2] > z0 - 0.25) & (p[:, 2] < z0 + 1.2)
        return p[keep, :2].astype(float)

    def clusters(self, k):
        r = self.inputs[k]
        if not r.get("robot_pose"):
            return []
        m = np.asarray(r["robot_pose"]["T_parent_child"])
        out = []
        for d in r["detections"]:
            c = np.array([[x, y, 0.0] for x in (d["bbox_min"][0], d["bbox_max"][0]) for y in (d["bbox_min"][1], d["bbox_max"][1])])
            c = c @ m[:3, :3].T + m[:3, 3]
            out.append((c[:, :2].min(0), c[:, :2].max(0)))
        return out

    def velocity(self, t):
        i = int(np.clip(np.searchsorted(self.odom_t, t), 0, len(self.odom_t) - 1))
        if abs(self.odom_t[i] - t) > 0.1:
            return None
        o = self.odom[i]
        return float(np.hypot(*o["linear"][:2])), float(o["angular"][2])

    def frame(self, k, v, trails, seen, events, title, with_rgb=True):
        t = self.times[k]
        pose = self.poses[k]
        planar = None
        if pose is not None:
            planar = np.eye(3)
            planar[:2, :2] = pose[:2, :2]
            planar[:2, 2] = pose[:2, 3]
        rec = self.records[v][k]
        tracks = [dv.Track.from_row(r) for r in rec["tracks"]]
        new_ids = 0
        for tr in tracks:
            if tr.status == "C" and tr.id not in seen:
                seen.add(tr.id)
                new_ids += 1
        shown_trails = dv.update_trails(trails, t, tracks)
        path = np.array([p[:2, 3] for p in self.poses[: k + 1] if p is not None]) if pose is not None else np.empty((0, 2))
        gt_states = []
        if self.gt_doc is not None:
            gt_states = self.gt_frames.get(round(t, 3), [])
        recent = [e for e in events if t - 4.0 < e.t <= t]
        markers = []
        for e in reversed(recent):  # newest first; one marker per 0.5 m so labels stay legible
            visible = t - e.t <= 1.0 and e.kind not in ("TRACK LOST", "TRACK ENDED") and e.xy is not None
            if visible and all(np.hypot(e.xy[0] - m.xy[0], e.xy[1] - m.xy[1]) > 0.5 for m in markers):
                markers.append(e)
        f = dv.FrameData(
            t=t,
            t_rel=t - self.t0,
            frame="odom",
            robot_pose=planar,
            robot_path=path[-2000:],
            points=self.points(k, pose),
            clusters=self.clusters(k),
            tracks=tracks,
            trails=shown_trails,
            gt=gt_states,
            events=[dv.Event(e.t - self.t0, e.kind, e.text, e.confirmed, e.xy if e in markers else None) for e in recent],
            robot_velocity=self.velocity(t),
            odometry_gap=any(abs(s - t) < 0.2 for s in self.skipped)
            or bool(np.any((np.diff(self.odom_t) > 0.1) & (np.abs(self.odom_t[:-1] - t) < 0.2))),
            new_ids=new_ids,
            title=title,
            key_ids=self.key.get(v),
        )
        idx = self.rgb_idx[k]
        if idx is not None:
            r = self.rgb_index[idx]
            f.rgb_index = idx
            f.rgb_time = t + self.rgb_dt[k]
        if with_rgb and idx is not None:
            f.rgb = cv2.imread(str(self.rgb_dir / r["file"]))
            f.rgb_detections = [
                (d["class_name"], d["confidence"], d["bbox_xyxy"]) for d in self.rgb_dets.get(r["file"], []) if d["confidence"] >= 0.3
            ]
        return f

    def view(self, k):
        """Fixed view per recording: centred ahead of the start pose, start heading up."""
        pose = next(p for p in self.poses if p is not None)
        yaw = np.arctan2(pose[1, 0], pose[0, 0])
        center = pose[:2, 3] + 1.6 * pose[:2, 0]
        return dv.BevView(center, yaw, 2.9, (dv.BEV_W, dv.TOP_H)), np.degrees(yaw)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--recording", required=True, help="e.g. test_b")
    p.add_argument("--variant", default="baseline", choices=list(VARIANT_LABEL))
    p.add_argument("--compare", nargs="+", choices=list(VARIANT_LABEL), help="side-by-side BEV of several variants")
    p.add_argument("--output", required=True)
    p.add_argument("--start", type=float, default=0.0, help="seconds since first LiDAR frame")
    p.add_argument("--end", type=float, default=1e9)
    p.add_argument("--title", default="")
    p.add_argument("--banner", default="", help="highlight caption; 'FAIL...' renders red, else green; '|' splits lines")
    p.add_argument("--title-card", type=float, default=0.0, help="seconds of a caption card (banner text) before the clip")
    p.add_argument("--fps", type=float, default=15.4, help="LiDAR rate gives real-time playback")
    p.add_argument("--inputs", default="data/outputs/mac/tracker_comparison/inputs")
    p.add_argument("--tracks", default="data/outputs/mac/ground_truth_eval/tracks")
    p.add_argument("--evaluation", default="data/outputs/mac/ground_truth_eval/evaluation.json")
    p.add_argument("--annotations", default="data/annotations")
    p.add_argument("--recordings", default="data/recordings")
    p.add_argument("--replay-root", default="data/outputs/mac")
    p.add_argument("--no-ground-truth", action="store_true")
    p.add_argument("--events-out", help="write the event log (JSON lines) here")
    p.add_argument("--forecasts", help="overlay.json from forecast_ground_truth.py")
    p.add_argument("--still", type=float, help="write one PNG at this time instead of a video")
    args = p.parse_args()
    if not dv.private_output_ok(args.output, ROOT):
        sys.exit(f"Refusing to write home imagery to a tracked path: {args.output}")
    variants = args.compare or [args.variant]
    run = Run(args, variants)
    run.key = {v: run.key_ids(v) for v in variants}
    events = {v: run.events(v) for v in variants}
    view, yaw_deg = run.view(0)
    t_rel = np.array(run.times) - run.t0
    frames = [k for k in range(len(run.times)) if args.start <= t_rel[k] <= args.end]
    if args.still is not None:
        frames = [int(np.argmin(np.abs(t_rel - args.still)))]
    t_end = t_rel[-1]
    timelines = {
        v: (0.0, t_end, [(e.t - run.t0, dv.RED if e.confirmed and "SWITCH" in e.kind else dv.AMBER) for e in events[v] if e.kind != "NEW TRACK"])
        for v in variants
    }
    forecasts = []
    if args.forecasts:
        forecasts = sorted(json.loads(Path(args.forecasts).read_text()).get(run.name, []), key=lambda x: x["origin"])
    trails = {v: {} for v in variants}
    seen = {v: set() for v in variants}
    # Warm up trails and seen IDs before the start time, so a clip starts with history.
    for k in range(frames[0]):
        for v in variants:
            run.frame(k, v, trails[v], seen[v], [], "", with_rgb=False)
    frame_label = dv.frame_label("odom", yaw_deg)
    writer = None if args.still is not None else dv.VideoWriter(args.output, args.fps)
    if writer is not None and args.title_card > 0:
        card = dv.title_card(args.banner or args.title or run.name)
        for _ in range(round(args.title_card * args.fps)):
            writer.write(card)
    for k in frames:
        if args.compare:
            fs = [run.frame(k, v, trails[v], seen[v], events[v], args.title or run.name, with_rgb=False) for v in variants]
            fs[0].banner = args.banner
            img = dv.compose_comparison(
                fs, [f"{VARIANT_LABEL[v]} | odom" for v in variants], [run.view(0)[0] for _ in variants], timelines[variants[0]]
            )
            dv._text(img, "shared timestamp and scale; event log and timeline below are for: " + VARIANT_LABEL[variants[0]], (14, dv.TOP_H + 84), 0.5, dv.DARK)
        else:
            f = run.frame(k, variants[0], trails[variants[0]], seen[variants[0]], events[variants[0]], f"{args.title or run.name} | {VARIANT_LABEL[variants[0]]}")
            f.banner = args.banner
            held = [x for x in forecasts if x["origin"] <= f.t + 1e-6 and f.t - x["origin"] <= 4.8]
            if held:
                fc = dict(held[-1], origin_rel=held[-1]["origin"] - run.t0)
                if fc.get("cv_tracker") is not None:
                    fc["tracker_last"] = None
                f.forecast = fc
            img = dv.compose(f, view, timelines[variants[0]], frame_label)
        if writer is None:
            cv2.imwrite(args.output, img)
        else:
            writer.write(img)
    if writer is not None:
        writer.close()
        print(f"Wrote {writer.frames} frames to {args.output}")
    if args.events_out:
        Path(args.events_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.events_out, "w") as s:
            for v in variants:
                for e in events[v]:
                    s.write(json.dumps(dict(variant=v, t=round(e.t - run.t0, 3), kind=e.kind, text=e.text, gt_supported=e.confirmed)) + "\n")


if __name__ == "__main__":
    main()
