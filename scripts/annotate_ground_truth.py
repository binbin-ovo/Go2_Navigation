"""Sensor-based ground-truth annotation for Go2 tracking recordings (local, private outputs).

Workflow (all files under the gitignored ``data/annotations/``):

1. ``cache``: transform every raw ``cloud_base`` frame to odom with recorded odometry.
2. ``propose``: from a hand-written spec (intervals, empty-scene reference, objects, review
   decisions) compute per-frame annotations at the tracker's frame times:
   - static obstacles: a footprint polygon measured once from the empty-scene points, and a
     visibility status from the fraction of its reference points seen in each 0.3 s window;
   - people: the median of background-subtracted foreground points in a 0.2 s window
     centred on the frame, smoothed over 0.3 s; velocity from a 0.6 s linear fit; moving /
     stationary only where the speed is clearly above / below the thresholds.
   The proposer uses raw points and an empty-scene background, never tracker output.
3. ``sheet``: render review contact sheets (BEV of raw and foreground points with the
   proposed annotation, plus the nearest camera frame) and a space-time overview.
4. Review the sheets; record decisions in the spec's ``review`` list (status, confidence
   or position overrides, or ``reject`` to drop a frame) and rerun ``propose``.

The output is ``data/annotations/<recording>.json`` (schema go2-tracking-gt-v1). Camera
frames are only displayed for review; they are never used for positions (no calibration).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from go2_perception.tracking import ground_truth as gt  # noqa: E402
from go2_perception.tracking.annotation_edits import apply_review  # noqa: E402

VOXEL = 0.1


def cache(args):
    from go2_perception.adapters.go2.recording import RecordingReader
    from go2_perception.tracking.replay import RecordedPoses

    reader = RecordingReader(args.recording)
    records = reader.state("rt/utlidar/robot_odom")
    poses = RecordedPoses(records, 0.1)
    frames = sorted(
        (f.load() for f in reader.lidar_frames("rt/utlidar/cloud_base")),
        key=lambda f: f.acquisition_midpoint,
    )
    times, clouds, matrices = [], [], []
    for f in frames:
        try:
            m = poses.lookup(f.acquisition_midpoint)
        except (KeyError, ValueError):
            m = None
        if m is None:  # no bracketing odometry: leave the frame out, never guess a pose
            continue
        times.append(f.acquisition_midpoint)
        clouds.append((f.xyz @ m[:3, :3].T + m[:3, 3]).astype(np.float32))
        matrices.append(m)
    offsets = np.cumsum([0] + [len(c) for c in clouds])
    host_offset = np.median([r["host_recv"] - r["stamp"] for r in records])
    out = Path(args.annotations) / "cache" / f"{Path(args.recording).name}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        t=np.array(times),
        pts=np.concatenate(clouds),
        off=offsets,
        pose=np.array(matrices),
        host_offset=host_offset,
    )
    print(f"Cached {len(times)} of {len(frames)} frames to {out}")


class Clouds:
    def __init__(self, path):
        d = np.load(path)
        self.t, self.pts, self.off, self.pose = d["t"], d["pts"], d["off"], d["pose"]
        self.host_offset = float(d["host_offset"])
        self.t0 = float(self.t[0])
        self.body_inv = np.linalg.inv(self.pose[0])

    def window(self, a, b):
        idx = np.where((self.t >= a) & (self.t <= b))[0]
        if not len(idx):
            return np.empty((0, 3))
        return self.pts[self.off[idx[0]] : self.off[idx[-1] + 1]].astype(float)

    def to_body(self, p):
        """Odom -> body frame of the first pose (only meaningful for a static robot)."""
        return p @ self.body_inv[:3, :3].T + self.body_inv[:3, 3]

    def to_odom(self, p):
        m = self.pose[0]
        return p @ m[:3, :3].T + m[:3, 3]


def components(points, eps):
    """Connected components of planar points with link distance ``eps``."""
    if not len(points):
        return np.empty(0, int)
    pairs = cKDTree(points).query_pairs(eps, output_type="ndarray")
    parent = np.arange(len(points))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return np.array([find(i) for i in range(len(points))])


def in_box(p, box, margin=0.0):
    (x0, x1), (y0, y1) = box
    return (
        (p[:, 0] >= x0 - margin)
        & (p[:, 0] <= x1 + margin)
        & (p[:, 1] >= y0 - margin)
        & (p[:, 1] <= y1 + margin)
    )


def body_points(clouds, a, b, z):
    p = clouds.to_body(clouds.window(a, b))
    return p[(p[:, 2] >= z[0]) & (p[:, 2] <= z[1])]


def measure_obstacle(clouds, spec, obj):
    """Footprint (body frame) of a static obstacle from empty-scene points near its seed."""
    a, b = (clouds.t0 + s for s in spec["empty_scene"])
    p = body_points(clouds, a, b, obj["z"])
    p = p[in_box(p, obj["search"])]
    labels = components(p[:, :2], 0.08)
    seed = np.asarray(obj["seed"])
    keep = labels == labels[np.argmin(np.linalg.norm(p[:, :2] - seed, axis=1))]
    q = p[keep]
    lo, hi = np.percentile(q[:, :2], 2, axis=0), np.percentile(q[:, :2], 98, axis=0)
    return ((float(lo[0]), float(hi[0])), (float(lo[1]), float(hi[1]))), int(keep.sum())


def frame_times(spec, tracker_inputs):
    """Annotate exactly the tracker's frame stamps (shared cached inputs) inside intervals."""
    stamps = [
        json.loads(line)["timestamp"]
        for line in Path(tracker_inputs).read_text().splitlines()
        if line.strip()
    ]
    t0 = stamps[0]
    keep = [
        t
        for t in stamps
        if any(i["start"] <= t - t0 <= i["end"] for i in spec["intervals"])
    ]
    return t0, keep


def registration(clouds, spec, obj, box, times, half=1.5, step=1.0):
    """Apparent displacement (start-pose body frame) of a static obstacle's front face.

    Measured from raw points near the footprint (grown by 0.3 m) in sliding windows, relative
    to the empty-scene reference: x from the 5th percentile (front face), y from the median of
    the face points. Interpolated to the annotated frame times.
    """

    def face(a, b):
        p = body_points(clouds, a, b, obj["z"])
        p = p[in_box(p, box, 0.3)]
        if len(p) < 30:
            return None
        x = np.percentile(p[:, 0], 5)
        return np.array([x, np.median(p[p[:, 0] < x + 0.08][:, 1])])

    a, b = (clouds.t0 + s for s in spec["empty_scene"])
    ref = face(a, b)
    grid = np.arange(times[0], times[-1] + step, step)
    samples = [(t, face(t - half, t + half)) for t in grid]
    samples = [(t, f - ref) for t, f in samples if f is not None]
    st = np.array([t for t, _ in samples])
    sv = np.array([v for _, v in samples])
    return np.c_[np.interp(times, st, sv[:, 0]), np.interp(times, st, sv[:, 1])]


def obstacle_states(clouds, spec, oid, obj, times, frame):
    box, n = measure_obstacle(clouds, spec, obj)
    a, b = (clouds.t0 + s for s in spec["empty_scene"])
    window = spec.get("visibility_window_s", 0.3)
    ref_counts = []
    for t in clouds.t[(clouds.t >= a + window) & (clouds.t <= b)]:
        p = body_points(clouds, t - window, t, obj["z"])
        ref_counts.append(int(in_box(p, box).sum()))
    ref = float(np.median(ref_counts))
    corners = np.array([[box[0][i], box[1][j], 0.0] for i, j in ((0, 0), (1, 0), (1, 1), (0, 1))])
    region = clouds.to_odom(corners)[:, :2].round(4).tolist()
    offsets = registration(clouds, spec, obj, box, times) if spec["robot"] == "moving" else None
    ratios, regions = {}, {}
    for k, t in enumerate(times):
        shifted = box
        if offsets is not None:
            # Moving robot: the obstacle is static, but odometry drift moves where it appears in
            # odom. Follow its measured front face (3 s windows) so the region stays registered.
            (dx, dy) = offsets[k]
            shifted = ((box[0][0] + dx, box[0][1] + dx), (box[1][0] + dy, box[1][1] + dy))
            moved = corners + np.array([dx, dy, 0.0])
            regions[t] = clouds.to_odom(moved)[:, :2].round(4).tolist()
        p = body_points(clouds, t - window, t, obj["z"])
        count = int(in_box(p, shifted).sum())
        ratios[t] = count / ref if ref else 0.0
    # The scan pattern is non-repetitive, so single-window counts are noisy (an unoccluded
    # obstacle drops to about 0.45 of its reference); use a 0.3 s running median.
    ts = np.array(times)
    raw = np.array([ratios[t] for t in times])
    lo, hi = obj.get("visibility_thresholds", (0.15, 0.5))
    states = {}
    for i, t in enumerate(times):
        ratio = float(np.median(raw[np.abs(ts - t) <= 0.15]))
        status = "visible" if ratio >= hi else "partial" if ratio >= lo else "occluded"
        near = min(abs(ratio - lo), abs(ratio - hi))
        states[t] = dict(
            id=oid,
            status=status,
            xy=None,
            region=None if status == "occluded" else regions.get(t, region),
            motion="stationary",
            confidence="high" if near > 0.1 else "medium",
            visible_ratio=round(ratio, 3),
            raw_ratio=round(float(raw[i]), 3),
        )
        if offsets is not None:
            states[t]["registration_offset"] = np.round(offsets[i], 3).tolist()
    meta = dict(
        footprint_body=box,
        reference_points=n,
        reference_window_count=ref,
        region_odom=region,
        frame=frame,
    )
    return states, meta


def person_states(clouds, spec, oid, obj, times):
    a, b = (clouds.t0 + s for s in spec["empty_scene"])
    background = set()
    for p in (body_points(clouds, a, b, obj["z"]),):
        background = {tuple(v) for v in np.floor(p / VOXEL).astype(int)}
    offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]

    def foreground(p):
        keys = np.floor(p / VOXEL).astype(int)
        mask = np.array(
            [not any((k[0] + o[0], k[1] + o[1], k[2] + o[2]) in background for o in offsets) for k in keys],
            bool,
        )
        return p[mask] if len(p) else p

    half = obj.get("window_s", 0.2) / 2
    raw = {}
    for t in times:
        p = body_points(clouds, t - half, t + half, obj["z"])
        p = foreground(p[in_box(p, obj["zone"])])
        if len(p) < obj.get("min_points", 4):
            raw[t] = (None, len(p))
            continue
        labels = components(p[:, :2], 0.35)
        values, counts = np.unique(labels, return_counts=True)
        q = p[labels == values[np.argmax(counts)]]
        raw[t] = (np.median(q[:, :2], axis=0), len(q))
    ts = np.array(times)
    states = {}
    smooth_s, fit_s = obj.get("smooth_s", 0.3), obj.get("velocity_fit_s", 0.6)
    located = np.array([raw[t][0] is not None for t in times])
    xy = np.array([raw[t][0] if raw[t][0] is not None else (np.nan, np.nan) for t in times])
    smoothed = xy.copy()
    for i, t in enumerate(ts):
        if not located[i]:
            continue
        near = located & (np.abs(ts - t) <= smooth_s / 2)
        smoothed[i] = np.median(xy[near], axis=0)
    lo_speed, hi_speed = obj.get("speed_thresholds", (0.12, 0.35))
    for i, t in enumerate(ts):
        point, n = raw[t]
        if point is None:
            # Nothing seen: outside the zone if the last sighting was near its edge, else
            # occluded. Either way no position is given.
            states[t] = dict(id=oid, status="out_of_view", xy=None, region=None, motion=None, confidence="medium")
            continue
        near = located & (np.abs(ts - t) <= fit_s / 2)
        velocity, motion, conf = None, None, "high"
        if near.sum() >= 5 and ts[near].ptp() >= 0.8 * fit_s:
            A = np.c_[ts[near] - t, np.ones(near.sum())]
            coef = np.linalg.lstsq(A, smoothed[near], rcond=None)[0]
            velocity = coef[0]
            speed = float(np.linalg.norm(velocity))
            motion = "moving" if speed > hi_speed else "stationary" if speed < lo_speed else None
        status = "visible" if n >= obj.get("visible_points", 10) else "partial"
        if status == "partial":
            conf = "medium"
        p_odom = clouds.to_odom(np.array([[*smoothed[i], 0.0]]))[0, :2]
        states[t] = dict(
            id=oid,
            status=status,
            xy=p_odom.round(4).tolist(),
            region=None,
            motion=motion,
            confidence=conf,
            points=int(n),
        )
        if velocity is not None:
            # Rotate body-frame velocity into odom (static robot: fixed rotation).
            states[t]["velocity"] = (clouds.pose[0][:2, :2] @ velocity).round(4).tolist()
    # Visit index: consecutive located runs separated by >= 1 s out of view.
    visit, last_seen = 0, None
    for t in ts:
        s = states[t]
        if s["xy"] is not None:
            if last_seen is None or t - last_seen >= 1.0:
                visit += 1
            last_seen = t
            s["visit"] = visit
    return states


def propose(args):
    spec = json.loads(Path(args.spec).read_text())
    name = spec["recording"]
    clouds = Clouds(Path(args.annotations) / "cache" / f"{name}.npz")
    t0, times = frame_times(spec, Path(args.inputs) / name / "frames.jsonl")
    per_object, meta = {}, {}
    for oid, obj in spec["objects"].items():
        if obj.get("manual"):  # declared for manual annotation only; nothing to propose
            continue
        if obj["type"] == "person":
            per_object[oid] = person_states(clouds, spec, oid, obj, times)
        else:
            per_object[oid], meta[oid] = obstacle_states(clouds, spec, oid, obj, times, "odom")
    frames = {t: [] for t in times}
    for oid, states in per_object.items():
        for t, s in states.items():
            if any(
                i["start"] <= t - t0 <= i["end"] and oid in i.get("objects", [oid])
                for i in spec["intervals"]
            ):
                frames[t].append(s)
    objects = {oid: dict(type=o["type"], description=o.get("description", "")) for oid, o in spec["objects"].items()}
    apply_review(frames, spec.get("review", []), t0, objects)
    doc = dict(
        schema=gt.SCHEMA,
        recording=name,
        frame="odom",
        clock_domain="go2_robot",
        t0=t0,
        annotator=spec.get("annotator", "unspecified"),
        method=spec.get("method", "lidar background subtraction, reviewed"),
        objects=objects,
        human_review_entries=len(spec.get("review", [])),
        intervals=spec["intervals"],
        review_log=spec.get("review_log", []),
        obstacle_measurements=meta,
        frames=[dict(t=t, objects=frames[t]) for t in times if frames[t]],
    )
    out = Path(args.annotations) / f"{name}.json"
    gt.save(doc, out)
    print(f"{name}: {len(doc['frames'])} annotated frames -> {out}")
    for oid in objects:
        statuses = [s["status"] for f in doc["frames"] for s in f["objects"] if s["id"] == oid]
        print(" ", oid, {k: statuses.count(k) for k in gt.STATUSES if statuses.count(k)})


def sheet(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    spec = json.loads(Path(args.spec).read_text())
    name = spec["recording"]
    doc = gt.load(Path(args.annotations) / f"{name}.json")
    clouds = Clouds(Path(args.annotations) / "cache" / f"{name}.npz")
    rgb_root = Path(args.recordings) / name / "rgb"
    rgb = [json.loads(line) for line in (rgb_root / "index.jsonl").read_text().splitlines()]
    rgb_t = np.array([r["host_recv_first"] for r in rgb]) - clouds.host_offset
    out = Path(args.annotations) / "review" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir(exist_ok=True)
    frames = doc["frames"]
    picks, last = [], -np.inf
    for f in frames:
        if f["t"] - last >= args.every:
            picks.append(f)
            last = f["t"]
    view = spec["view"]
    per_sheet = args.columns * args.rows
    for k in range(0, len(picks), per_sheet):
        fig, axes = plt.subplots(args.rows * 2, args.columns, figsize=(3.1 * args.columns, 5.6 * args.rows))
        for ax in axes.ravel():
            ax.axis("off")
        for n, f in enumerate(picks[k : k + per_sheet]):
            r, c = divmod(n, args.columns)
            ax = axes[2 * r, c]
            ax.axis("on")
            p = clouds.window(f["t"] - 0.15, f["t"] + 0.15)
            p = p[(p[:, 2] > view["z_odom"][0]) & (p[:, 2] < view["z_odom"][1])]
            pb = clouds.to_body(p)
            ax.scatter(pb[:, 1], pb[:, 0], s=0.6, c="0.4")
            for s in f["objects"]:
                colour = "r" if s["id"] == "person" else "b"
                if s.get("xy"):
                    q = clouds.to_body(np.array([[*s["xy"], 0]]))[0]
                    ax.plot(q[1], q[0], "o", mfc="none", mec=colour, ms=14, mew=1.5)
                    if s.get("velocity"):
                        v = clouds.body_inv[:2, :2] @ np.asarray(s["velocity"])
                        ax.arrow(q[1], q[0], v[1], v[0], color=colour, width=0.01)
                if s.get("region"):
                    reg = clouds.to_body(np.c_[np.asarray(s["region"]), np.zeros(4)])
                    ax.fill(reg[:, 1], reg[:, 0], fill=False, ec=colour, lw=1.2)
            ax.set_xlim(view["y"][1], view["y"][0])
            ax.set_ylim(*view["x"])
            ax.set_aspect("equal")
            ax.tick_params(labelsize=6)
            label = " ".join(
                f"{s['id'][:2]}:{s['status'][:3]}/{(s.get('motion') or '-')[:3]}/{s['confidence'][0]}"
                for s in f["objects"]
            )
            ax.set_title(f"{f['t'] - doc['t0']:.2f}s {label}", fontsize=6.5)
            i = int(np.argmin(np.abs(rgb_t - f["t"])))
            ax = axes[2 * r + 1, c]
            img = Image.open(rgb_root / rgb[i]["file"])
            img.thumbnail((400, 225))
            ax.imshow(img)
            ax.set_title(f"rgb {rgb_t[i] - doc['t0']:.2f}s", fontsize=6)
        fig.tight_layout()
        path = out / f"sheet_{k // per_sheet:02d}.png"
        fig.savefig(path, dpi=args.dpi)
        plt.close(fig)
        print(path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["cache", "propose", "sheet"])
    p.add_argument("--recording", help="cache: recording directory")
    p.add_argument("--spec", help="propose/sheet: annotation spec JSON")
    p.add_argument("--annotations", default="data/annotations")
    p.add_argument("--inputs", default="data/outputs/mac/tracker_comparison/inputs")
    p.add_argument("--recordings", default="data/recordings")
    p.add_argument("--every", type=float, default=0.5, help="sheet: seconds between panels")
    p.add_argument("--columns", type=int, default=6)
    p.add_argument("--rows", type=int, default=3)
    p.add_argument("--dpi", type=int, default=80)
    args = p.parse_args()
    dict(cache=cache, propose=propose, sheet=sheet)[args.command](args)


if __name__ == "__main__":
    main()
