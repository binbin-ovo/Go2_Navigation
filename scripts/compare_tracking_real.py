"""Replay a Git-baseline tracker and the working tracker on identical local Go2 recordings.

Extraction (LiDAR clusters, 0.3 s window, pose gating) runs once with code that is unchanged
between the baseline and the working tree; every tracker variant then consumes the same cached
frames in its own process. Outputs stay local. The recording-specific anchors below come from
the earlier report (tape-measured box, background-subtracted person path); they are proxies,
not annotated ground truth.
"""

import argparse
import collections
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# label: (source, config, overrides). "baseline" source means the archived Git revision.
VARIANTS = {
    "refined": ("working", "configs/tracking_refinement.yaml", {}),
    "baseline": ("baseline", "configs/tracking.yaml", {}),
    "current_default": ("working", "configs/tracking.yaml", {}),
    "improved": ("working", "configs/tracking_robustness.yaml", {}),
    "improved_no_time_expiry": (
        "working",
        "configs/tracking_robustness.yaml",
        dict(confirmed_time_expiry=False),
    ),
    "improved_no_adaptive_gate": (
        "working",
        "configs/tracking_robustness.yaml",
        dict(adaptive_gate_max_distance=0.0),
    ),
    "improved_no_fragments": (
        "working",
        "configs/tracking_robustness.yaml",
        dict(fragment_recovery=False),
    ),
    "improved_legacy_motion": (
        "working",
        "configs/tracking_robustness.yaml",
        dict(stationary_speed=0.15, motion_dwell_s=0.0),
    ),
}

# Proxies from docs/PHASE3_REAL_VALIDATION.md. "body" anchors are in the static robot's
# base_link (median odometry pose); "odom" anchors are world positions for the moving robot.
ANCHORS = {
    "test_a1": dict(box=((2.50, 0.01), 0.3, "body")),
    "test_a2": dict(box=((2.49, 0.00), 0.3, "body"), person_spot=((1.93, 0.61), 0.5)),
    "test_b": dict(box=((2.50, 0.00), 0.3, "body")),
    "test_c": dict(box=((1.95, -0.40), 0.3, "odom")),
}
B_PASSES = [(7.5, 17.5), (21.5, 34.5), (39.5, 53.5)]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract(recording, target):
    from go2_perception.io.config import load_config
    from go2_perception.io.serialization import jsonable
    from go2_perception.tracking.config import TrackerConfig
    from go2_perception.tracking.pose_gate import PoseAlignmentConfig
    from go2_perception.tracking.replay import recording_frames

    config = load_config(ROOT / "configs/tracking.yaml")
    report = {}
    rows = []
    for f in recording_frames(
        recording,
        TrackerConfig(**config["tracker"]),
        load_config(ROOT / "configs/go2_lidar.yaml"),
        0.3,
        PoseAlignmentConfig(**config["pose_alignment"]),
        report,
    ):
        row = jsonable(f)
        for d in row["detections"]:
            for key in ("timestamp", "frame_id", "clock_domain"):
                d.pop(key)
        rows.append(row)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in rows))
    (target.parent / "pose_report.json").write_text(
        json.dumps(jsonable(report), indent=2) + "\n"
    )


def worker(args):
    """Runs inside the variant's PYTHONPATH; writes compact per-frame tracks."""
    import go2_perception
    from go2_perception.io.config import load_config
    from go2_perception.tracking.multi_object_tracker import MultiObjectTracker
    from go2_perception.tracking.replay import frame_from_dict

    cfg = dict(load_config(args.config)["tracker"], **json.loads(args.overrides))
    for path in sorted(Path(args.cache).glob("*/frames.jsonl")):
        tracker = MultiObjectTracker(cfg)
        out = Path(args.output) / path.parent.name / "tracks.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as stream:
            for line in path.read_text().splitlines():
                row = json.loads(line)
                f = frame_from_dict(row)
                if row.get("reset_before"):
                    tracker.reset()
                start = perf_counter()
                tracks = tracker.update(
                    f.detections,
                    f.timestamp,
                    f.robot_pose,
                    frame_id=f.frame_id,
                    clock_domain=f.clock_domain,
                )
                latency = (perf_counter() - start) * 1000
                record = dict(
                    t=f.timestamp,
                    latency_ms=latency,
                    raw=len(f.detections),
                    fragments=len(getattr(tracker, "fragment_groups", [])),
                    tracks=[
                        [
                            t.track_id,
                            *np.round(t.position, 4).tolist(),
                            *np.round(t.velocity, 4).tolist(),
                            round(float(np.sqrt(np.linalg.eigvalsh(t.state_covariance[2:, 2:]).max())), 4),
                            t.status[0],
                            t.motion_state[0],
                            t.consecutive_misses,
                        ]
                        for t in tracks
                    ],
                )
                stream.write(json.dumps(record) + "\n")
    (Path(args.output) / "worker.json").write_text(
        json.dumps(
            dict(
                package_path=str(Path(go2_perception.__file__).resolve()),
                config=cfg,
            ),
            indent=2,
        )
        + "\n"
    )


def run(args):
    out = Path(args.output).resolve()
    if (out / "run.json").exists() or any((out / label).exists() for label in args.variants):
        raise FileExistsError("Use a fresh output directory; --analyze reads existing results")
    cache = Path(args.cache_root).resolve() if args.cache_root else out / "inputs"
    if args.cache_root and not args.reuse_cache:
        raise ValueError("External cache requires --reuse-cache")
    for name in args.recordings:
        target = cache / Path(name).name / "frames.jsonl"
        if args.cache_root and not target.exists():
            raise FileNotFoundError(target)
        if not (target.exists() and args.reuse_cache):
            print("Extracting LiDAR detections locally:", Path(name).name, flush=True)
            extract(name, target)
    baseline = subprocess.check_output(
        ["git", "rev-parse", args.baseline], cwd=ROOT, text=True
    ).strip()
    working = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain", "src", "configs"], cwd=ROOT, text=True
        ).strip()
    )
    with tempfile.TemporaryDirectory(prefix="go2-tracker-baseline-") as tmp:
        archive = subprocess.check_output(
            ["git", "archive", "--format=zip", baseline, "src", "configs"], cwd=ROOT
        )
        with zipfile.ZipFile(io.BytesIO(archive)) as z:
            if any(".." in Path(i.filename).parts for i in z.infolist()):
                raise ValueError("Unsafe archive path")
            z.extractall(tmp)
        jobs = []
        for label in args.variants:
            source, config, overrides = VARIANTS[label]
            base = Path(tmp) if source == "baseline" else ROOT
            env = dict(os.environ, PYTHONPATH=str(base / "src"))
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--cache",
                str(cache),
                "--config",
                str(base / config),
                "--overrides",
                json.dumps(overrides),
                "--output",
                str(out / label),
            ]
            print("Running", label, flush=True)
            jobs.append((label, base, subprocess.Popen(cmd, cwd=ROOT, env=env)))
            if args.serial:
                jobs[-1][2].wait()
        for label, base, job in jobs:
            if job.wait():
                raise RuntimeError(f"{label} failed")
            imported = Path(json.loads((out / label / "worker.json").read_text())["package_path"])
            if not imported.resolve().is_relative_to((base / "src").resolve()):
                raise RuntimeError(f"{label} imported {imported}")
    (out / "run.json").write_text(
        json.dumps(
            dict(
                baseline_commit=baseline,
                working_commit=working,
                working_tree_dirty=dirty,
                variants={k: VARIANTS[k] for k in args.variants},
                input_sha256={
                    p.parent.name: sha256(p) for p in sorted(cache.glob("*/frames.jsonl"))
                },
            ),
            indent=2,
        )
        + "\n"
    )
    print("Saved", out / "run.json", flush=True)


# ---------------------------------------------------------------- analysis


def load_tracks(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        r["tracks"] = [
            dict(
                id=t[0],
                p=np.array(t[1:3]),
                v=np.array(t[4:6]),
                vstd=t[6],
                status=t[7],
                motion=t[8],
                misses=t[9],
            )
            for t in r["tracks"]
        ]
        rows.append(r)
    return rows


class Recording:
    """Recording clock origin, odometry and the static-robot body transform."""

    def __init__(self, path):
        from scipy.spatial.transform import Rotation

        path = Path(path)
        index = path / "lidar/utlidar_cloud_base/index.jsonl"
        self.t0 = json.loads(index.read_text().splitlines()[0])["stamp"]
        odom = [json.loads(line) for line in (path / "state/utlidar_robot_odom.jsonl").read_text().splitlines()]
        self.ot = np.array([o["stamp"] for o in odom])
        self.P = np.array([o["position"] for o in odom])
        yaw = Rotation.from_quat([o["q_xyzw"] for o in odom]).as_euler("zyx")[:, 0]
        self.yaw = np.unwrap(yaw)
        self.w = np.abs([o["angular"][2] for o in odom])
        self.v = np.array([np.hypot(*o["linear"][:2]) for o in odom])
        self.p0 = np.median(self.P, 0)[:2]
        y = np.median(yaw)
        self.R = np.array([[np.cos(y), np.sin(y)], [-np.sin(y), np.cos(y)]])

    def body(self, xy):
        return self.R @ (np.asarray(xy) - self.p0)

    def phase(self, t):
        i = min(np.searchsorted(self.ot, t), len(self.ot) - 1)
        return "rotating" if self.w[i] > 0.15 else ("translating" if self.v[i] > 0.08 else "still")


def runs_of(seq):
    runs = []
    prev = "start"
    for t, i in seq:
        if i != prev:
            runs.append([i, t, t])
            prev = i
        else:
            runs[-1][2] = t
    return runs


def nearest(r, anchor, radius, to_xy, statuses="CT"):
    best = None
    for t in r["tracks"]:
        if t["status"] not in statuses:
            continue
        d = np.hypot(*(to_xy(t["p"]) - anchor))
        if d < radius and (best is None or d < best[0]):
            best = (d, t)
    return best


def object_proxy(rows, rec, anchor, radius, frame):
    """Non-Lost track nearest a fixed anchor each frame (the earlier report's definition)."""
    to_xy = rec.body if frame == "body" else (lambda p: p)
    seq = []
    picked = []
    competing = 0
    for r in rows:
        t = r["t"] - rec.t0
        best = nearest(r, anchor, radius, to_xy)
        competing += sum(
            1 for x in r["tracks"] if x["status"] in "CT" and np.hypot(*(to_xy(x["p"]) - anchor)) < radius
        ) > 1
        seq.append((t, best[1]["id"] if best else None))
        if best:
            picked.append((t, rec.phase(r["t"]), best[1], to_xy(best[1]["p"])))
    observed_ids = []
    for r in rows:
        observed_row = dict(r, tracks=[t for t in r["tracks"] if not t["misses"]])
        best = nearest(observed_row, anchor, radius, to_xy)
        if best:
            observed_ids.append(best[1]["id"])
    ids = [i for _, i in seq if i is not None]
    changes = sum(a != b for a, b in zip(ids, ids[1:]))
    seen = set(ids[:1])
    returns = 0
    for a, b in zip(ids, ids[1:]):
        if a != b:
            returns += b in seen  # flicker: an intruding ID took the anchor, then the old one came back
            seen.add(b)
    runs = [x for x in runs_of(seq) if x[0] is not None]
    pos = np.array([x[3] for x in picked])
    speed = np.array([np.hypot(*x[2]["v"]) for x in picked])
    vstd = np.array([x[2]["vstd"] for x in picked])
    result = dict(
        observed_coverage=round(len(observed_ids) / len(rows), 3),
        observed_dominant_id_share=(round(collections.Counter(observed_ids).most_common(1)[0][1] / len(observed_ids), 3) if observed_ids else None),
        frames=len(rows),
        coverage=round(len(picked) / len(rows), 3),
        unique_ids=len(set(ids)),
        id_changes=changes,
        returns_to_previous_id=returns,
        dominant_id_share=round(collections.Counter(ids).most_common(1)[0][1] / len(ids), 3),
        longest_run_s=round(max(b - a for _, a, b in runs), 1) if runs else 0.0,
        runs_over_1s=sum(b - a > 1 for _, a, b in runs),
        competing_frames=int(competing),
        position_std_m=np.round(pos.std(0), 3).tolist(),
        speed_median=round(float(np.median(speed)), 3),
        speed_p95=round(float(np.percentile(speed, 95)), 3),
        velocity_std_median=round(float(np.median(vstd)), 3),
        states=dict(collections.Counter(x[2]["motion"] for x in picked)),
    )
    if frame == "odom":
        by_phase = {}
        for ph in ("still", "rotating", "translating"):
            s = [x for x in picked if x[1] == ph]
            if not s:
                continue
            p = np.array([x[3] for x in s])
            sp = np.array([np.hypot(*x[2]["v"]) for x in s])
            by_phase[ph] = dict(
                n=len(s),
                std_m=np.round(p.std(0), 3).tolist(),
                speed_median=round(float(np.median(sp)), 3),
                speed_p95=round(float(np.percentile(sp, 95)), 3),
                moving=sum(x[2]["motion"] == "M" for x in s),
                stationary=sum(x[2]["motion"] == "S" for x in s),
            )
        # ID changes whose new ID appears while (or within 0.5 s after) the robot rotates.
        prev = None
        turn_changes = 0
        for t, ph, tr, _ in picked:
            if prev is not None and tr["id"] != prev:
                window = [
                    rec.phase(rec.t0 + t - dt) for dt in (0.0, 0.25, 0.5)
                ]
                turn_changes += "rotating" in window
            prev = tr["id"]
        a = pos[[x[0] < 5 for x in picked]].mean(0)
        b = pos[[x[0] > picked[-1][0] - 5 for x in picked]].mean(0)
        result.update(
            phases=by_phase,
            id_changes_near_rotation=turn_changes,
            drift_first_last_5s_m=round(float(np.hypot(*(b - a))), 3),
        )
    return result, seq


def person_b(rows, rec, ref):
    ok = np.isfinite(ref[:, 3])

    def ref_pos(t):
        i = np.argmin(np.abs(ref[:, 0] - t))
        return ref[i, 2:4] if ok[i] and abs(ref[i, 0] - t) < 0.15 else None

    def ref_vel(t):
        a, b = ref_pos(t - 0.3), ref_pos(t + 0.3)
        return None if a is None or b is None else (b - a) / 0.6

    assoc = []
    for r in rows:
        t = r["t"] - rec.t0
        rp = ref_pos(t)
        if rp is None:
            continue
        best = nearest(r, rp, 0.5, rec.body)
        assoc.append((t, rp, ref_vel(t), best[1] if best else None))
    passes = []
    for a, b in B_PASSES:
        seg = [x for x in assoc if a <= x[0] <= b]
        cov = [x for x in seg if x[3]]
        ids = [x[3]["id"] for x in cov]
        walk = [x for x in cov if x[2] is not None and np.hypot(*x[2]) > 0.4]
        pause = [x for x in cov if x[2] is not None and np.hypot(*x[2]) < 0.15]
        passes.append(
            dict(
                window_s=[a, b],
                coverage=round(len(cov) / max(1, len(seg)), 3),
                unique_ids=len(set(ids)),
                id_changes=sum(i != j for i, j in zip(ids, ids[1:])),
                walking_frames=len(walk),
                walking_reference_frames=sum(x[2] is not None and np.hypot(*x[2]) > 0.4 for x in seg),
                walking_ref_speed=round(float(np.median([np.hypot(*x[2]) for x in walk])), 3),
                walking_track_speed=round(
                    float(np.median([np.hypot(*(rec.R @ x[3]["v"])) for x in walk])), 3
                ),
                walking_velocity_error=round(
                    float(np.median([np.hypot(*(rec.R @ x[3]["v"] - x[2])) for x in walk])), 3
                ),
                walking_states=dict(collections.Counter(x[3]["motion"] for x in walk)),
                pausing_frames=len(pause),
                pausing_track_speed=round(
                    float(np.median([np.hypot(*x[3]["v"]) for x in pause])), 3
                ),
                pausing_states=dict(collections.Counter(x[3]["motion"] for x in pause)),
            )
        )
    covered = [x for x in assoc if x[3]]
    seq = [(x[0], x[3]["id"] if x[3] else None) for x in assoc]
    return dict(
        coverage=round(len(covered) / len(assoc), 3),
        unique_ids=len({x[3]["id"] for x in covered}),
        passes=passes,
    ), seq


def person_spot(rows, rec, anchor, radius):
    present = collections.defaultdict(list)
    for r in rows:
        for t in r["tracks"]:
            if t["status"] in "CT" and np.hypot(*(rec.body(t["p"]) - anchor)) < radius:
                present[t["id"]].append(r["t"] - rec.t0)
    spans = [max(v) - min(v) for v in present.values()]
    return dict(unique_ids=len(present), longest_span_s=round(max(spans), 1) if spans else 0.0)


def scene(rows, static_robot):
    """All-scene counts. Duplicates: two confirmed tracks closer than 0.3 m in one frame."""
    first = {}
    last = {}
    confirmed = set()
    dup_frames = 0
    dup_pairs = 0
    observed_dup_frames = 0
    last_seen = {}
    jumps = 0
    reactivations = 0
    reactivation_jumps = 0
    states = collections.Counter()
    ever_moving = set()
    for r in rows:
        conf = [t for t in r["tracks"] if t["status"] == "C"]
        for t in r["tracks"]:
            first.setdefault(t["id"], r["t"])
            last[t["id"]] = r["t"]
            if t["misses"]:
                continue
            if t["id"] in last_seen:
                t_prev, p_prev, gap = last_seen[t["id"]]
                d = np.hypot(*(t["p"] - p_prev))
                # A home object or person cannot cover >0.75 m at >2 m/s between observations.
                jumps += d > 0.75 and d / (r["t"] - t_prev) > 2.0
                if gap:
                    reactivations += 1
                    reactivation_jumps += d > 1.0
            last_seen[t["id"]] = (r["t"], t["p"], False)
        for t in r["tracks"]:
            if t["misses"] >= 2 and t["id"] in last_seen:
                last_seen[t["id"]] = (*last_seen[t["id"]][:2], True)
        for t in conf:
            confirmed.add(t["id"])
            states[t["motion"]] += 1
            if t["motion"] == "M":
                ever_moving.add(t["id"])
        pairs = sum(
            np.hypot(*(a["p"] - b["p"])) < 0.3
            for i, a in enumerate(conf)
            for b in conf[i + 1 :]
        )
        observed_conf = [t for t in conf if not t["misses"]]
        observed_dup_frames += any(
            np.hypot(*(a["p"] - b["p"])) < 0.3
            for i, a in enumerate(observed_conf) for b in observed_conf[i + 1:]
        )
        dup_pairs += pairs
        dup_frames += pairs > 0
    life = np.array([last[i] - first[i] for i in confirmed])
    lat = np.array([r["latency_ms"] for r in rows])
    return dict(
        created_ids=len(first),
        confirmed_ids=len(confirmed),
        confirmed_ids_lasting_5s=int((life >= 5).sum()),
        confirmed_lifetime_median_s=round(float(np.median(life)), 2),
        confirmed_tracks_per_frame=round(
            float(np.mean([sum(t["status"] == "C" for t in r["tracks"]) for r in rows])), 2
        ),
        lost_tracks_per_frame=round(
            float(np.mean([sum(t["status"] == "L" for t in r["tracks"]) for r in rows])), 2
        ),
        duplicate_pair_frames=int(dup_frames),
        duplicate_pairs=int(dup_pairs),
        observed_duplicate_pair_frames=int(observed_dup_frames),
        confirmed_states=dict(states),
        ids_ever_moving=len(ever_moving),
        fragment_groups=int(sum(r["fragments"] for r in rows)),
        implausible_jumps=int(jumps),
        reactivations=int(reactivations),
        reactivations_over_1m=int(reactivation_jumps),
        tracker_ms_mean=round(float(lat.mean()), 2),
        tracker_ms_p95=round(float(np.percentile(lat, 95)), 2),
    )


def historical_match(rows, path):
    """Exact-replay check against the original Phase 3 outputs (IDs, status, position)."""
    if not Path(path).exists():
        return None
    old = [json.loads(line) for line in Path(path).read_text().splitlines()]
    same = 0
    worst = 0.0
    for r, o in zip(rows, old):
        a = sorted((t["id"], t["status"], t["motion"]) for t in r["tracks"])
        b = sorted((t["track_id"], t["status"][0], t["motion_state"][0]) for t in o["tracks"])
        same += a == b and r["t"] == o["timestamp"]
        if a == b:
            pa = {t["id"]: t["p"] for t in r["tracks"]}
            for t in o["tracks"]:
                worst = max(worst, float(np.abs(pa[t["track_id"]] - t["position"][:2]).max()))
    return dict(frames_identical=same, frames=len(old), max_position_diff_m=worst)


def plot(out, name, rec, data, seqs, labels, frame):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    to_xy = rec.body if frame == "body" else (lambda p: p)
    fig = plt.figure(figsize=(13, 8.2))
    grid = fig.add_gridspec(2, 2, height_ratios=[3.2, 1])
    cmap = plt.get_cmap("tab20")
    (anchor, radius, _), = [ANCHORS[name]["box"]]
    for k, label in enumerate(labels):
        ax = fig.add_subplot(grid[0, k])
        tracks = collections.defaultdict(list)
        for r in data[label]:
            for t in r["tracks"]:
                if t["status"] == "C":
                    tracks[t["id"]].append(to_xy(t["p"]))
        long = {i: np.array(v) for i, v in tracks.items() if len(v) >= 8}
        for i, v in long.items():
            ax.plot(v[:, 1], v[:, 0], ".", ms=1.2, color=cmap(i % 20), alpha=0.5)
        ax.add_patch(plt.Circle((anchor[1], anchor[0]), radius, fill=False, color="k", lw=1.2))
        ax.plot(0, 0, "k^", ms=8) if frame == "body" else ax.plot(
            *(rec.P[:, 1], rec.P[:, 0]), "k-", lw=1
        )
        ax.set_xlim(3, -3)
        ax.set_ylim(-1, 5)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_xlabel("y left (m)" if frame == "body" else "odom y (m)")
        ax.set_ylabel("x forward (m)" if frame == "body" else "odom x (m)")
        ax.set_title(
            f"{label}: {len(tracks)} confirmed IDs, {len(long)} with >=8 frames",
            fontsize=10,
        )
    ax = fig.add_subplot(grid[1, :])
    for k, label in enumerate(labels):
        order = {}  # colour by order of first appearance so consecutive IDs never share a colour
        for i, a, b in (x for x in runs_of(seqs[label]) if x[0] is not None):
            colour = cmap(order.setdefault(i, len(order)) % 20)
            ax.barh(k, max(b - a, 0.07), left=a, height=0.6, color=colour)
            if b - a >= 3:
                ax.text((a + b) / 2, k, f"#{i}", ha="center", va="center", fontsize=7)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("time since first LiDAR frame (s); colour = box track ID, gaps = no non-Lost track in the circle")
    fig.suptitle(f"{name}: confirmed track positions and box-ID timeline (LiDAR only, no camera)")
    fig.tight_layout()
    path = out / "plots" / f"{name}_bev_compare.png"
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def analyze(args):
    out = Path(args.output)
    run_info = json.loads((out / "run.json").read_text())
    labels = list(run_info["variants"])
    report = dict(run=run_info, recordings={})
    for name in ANCHORS:
        rec_path = Path(args.recordings_root) / name
        if not rec_path.exists() or not all((out / label / name).exists() for label in labels):
            continue
        rec = Recording(rec_path)
        data = {k: load_tracks(out / k / name / "tracks.jsonl") for k in labels}
        anchor, radius, frame = ANCHORS[name]["box"]
        entry = {}
        box_seqs = {}
        for k in labels:
            box, box_seqs[k] = object_proxy(data[k], rec, np.array(anchor), radius, frame)
            e = dict(scene=scene(data[k], frame == "body"), box=box)
            if "person_spot" in ANCHORS[name]:
                e["person_spot"] = person_spot(data[k], rec, np.array(ANCHORS[name]["person_spot"][0]), ANCHORS[name]["person_spot"][1])
            ref = Path(args.outputs_root) / "test_b_tracking/raw_foreground_traj.npy"
            if name == "test_b" and ref.exists():
                e["person"], _ = person_b(data[k], rec, np.load(ref))
            entry[k] = e
        if "baseline" in data:
            entry["baseline"]["historical_match"] = historical_match(
                data["baseline"], Path(args.outputs_root) / f"{name}_tracking/tracks.jsonl"
            )
        entry["current_default_identical_to_baseline"] = all(
            [(t["id"], t["status"], t["motion"], *t["p"]) for t in a["tracks"]]
            == [(t["id"], t["status"], t["motion"], *t["p"]) for t in b["tracks"]]
            for a, b in zip(data["baseline"], data["current_default"])
        ) if "current_default" in data and "baseline" in data else None
        plot_labels = [label for label in ("improved", "refined") if label in data]
        if len(plot_labels) != 2:
            plot_labels = labels[:2]
        if len(plot_labels) == 2:
            plot(out, name, rec, data, box_seqs, plot_labels, frame)
        report["recordings"][name] = entry
        print("Analysed", name, flush=True)
    (out / "comparison_real.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    print("Saved", out / "comparison_real.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", default="ecfaae3")
    p.add_argument("--output", default="data/outputs/mac/tracker_comparison")
    p.add_argument(
        "--recordings",
        nargs="+",
        default=[f"data/recordings/{n}" for n in ANCHORS],
    )
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    p.add_argument("--reuse-cache", action="store_true")
    p.add_argument("--cache-root", help="Read existing extracted inputs without modifying them")
    p.add_argument("--serial", action="store_true", help="one variant at a time, for timing")
    p.add_argument("--analyze", action="store_true", help="only analyse an existing run")
    p.add_argument("--recordings-root", default="data/recordings")
    p.add_argument("--outputs-root", default="data/outputs/mac")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--cache", help=argparse.SUPPRESS)
    p.add_argument("--config", help=argparse.SUPPRESS)
    p.add_argument("--overrides", default="{}", help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.worker:
        worker(a)
    else:
        if not a.analyze:
            run(a)
        analyze(a)
