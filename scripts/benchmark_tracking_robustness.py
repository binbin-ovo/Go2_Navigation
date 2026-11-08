"""Compare a Git baseline with the working tracker on identical local inputs. Never uploads data."""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter
import zipfile
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in rows))


def metrics(frames, outputs, latencies, events):
    from go2_perception.tracking.evaluation import evaluate

    result = evaluate(frames, outputs, latencies, match_distance=0.5)
    seen = set()
    confirmed = set()
    tentative = set()
    duplicates = 0
    ids_by_gt = {}
    motion = {key: dict(tp=0, fp=0, fn=0) for key in ("MOVING", "STATIONARY")}
    states = {}
    speeds = []
    reactivated = 0
    last_status = {}
    for f, tracks in zip(frames, outputs):
        reliable = [t for t in tracks if t.status == "Confirmed"]
        for t in tracks:
            seen.add(t.track_id)
            if t.status == "Tentative":
                tentative.add(t.track_id)
            if t.status == "Confirmed":
                confirmed.add(t.track_id)
                states[t.motion_state] = states.get(t.motion_state, 0) + 1
                speeds.append(float(np.linalg.norm(t.velocity)))
                if last_status.get(t.track_id) == "Lost":
                    reactivated += 1
            last_status[t.track_id] = t.status
        if f.ground_truth:
            cost = np.array(
                [
                    [
                        np.linalg.norm(np.array(g["position"])[:2] - t.position[:2])
                        for t in reliable
                    ]
                    for g in f.ground_truth
                ]
            )
            if not reliable:
                continue
            # Count competing confirmed tracks within .3m of a unique nearest GT.
            owners = np.argmin(cost, axis=0)
            for gi in range(len(f.ground_truth)):
                duplicates += max(
                    0,
                    sum(
                        owners[j] == gi and cost[gi, j] < 0.3
                        for j in range(len(reliable))
                    )
                    - 1,
                )
            a, b = linear_sum_assignment(cost)
            for gi, tj in zip(a, b):
                if cost[gi, tj] > 0.5:
                    continue
                g, t = f.ground_truth[gi], reliable[tj]
                ids_by_gt.setdefault(g["object_id"], set()).add(t.track_id)
                truth = (
                    "MOVING" if np.linalg.norm(g["velocity"]) >= 0.3 else "STATIONARY"
                )
                for key, count in motion.items():
                    count["tp"] += int(t.motion_state == key and truth == key)
                    count["fp"] += int(t.motion_state == key and truth != key)
                    count["fn"] += int(t.motion_state != key and truth == key)
    result.update(
        created_tracks=len(seen),
        ever_tentative_tracks=len(tentative),
        ever_confirmed_tracks=len(confirmed),
        reactivation_events=reactivated,
        fragment_groups=events["fragment_groups"],
        confirmed_motion_counts=states,
        confirmed_speed_p95=float(np.percentile(speeds, 95)) if speeds else None,
    )
    if all(f.ground_truth is not None for f in frames):
        result.update(
            duplicate_track_frames=duplicates,
            identity_fragmentation=sum(max(0, len(v) - 1) for v in ids_by_gt.values()),
            motion_metrics={
                key: dict(
                    **v,
                    precision=v["tp"] / (v["tp"] + v["fp"])
                    if v["tp"] + v["fp"]
                    else None,
                    recall=v["tp"] / (v["tp"] + v["fn"]) if v["tp"] + v["fn"] else None,
                )
                for key, v in motion.items()
            },
        )
    else:
        result["proxy_only"] = True
        # Same fixed ROI for both variants, from the reported box about 2.5m ahead
        # at recording start. This is an unlabelled proxy, not ground truth identity.
        pose = frames[0].robot_pose
        if pose is not None:
            anchor = pose.T_parent_child @ np.array([2.5, 0.0, 0.0, 1.0])
            roi_ids = set()
            roi_speeds = []
            roi_states = {}
            covered = 0
            competition = 0
            proposals = 0
            for f, tracks in zip(frames, outputs):
                candidates = [
                    t
                    for t in tracks
                    if t.status == "Confirmed"
                    and np.linalg.norm(t.position[:2] - anchor[:2]) < 0.5
                ]
                competition += int(len(candidates) > 1)
                for d in f.detections:
                    p = f.robot_pose.T_parent_child @ np.r_[d.position, 1.0]
                    proposals += int(np.linalg.norm(p[:2] - anchor[:2]) < 0.5)
                if candidates:
                    covered += 1
                    t = min(
                        candidates,
                        key=lambda t: np.linalg.norm(t.position[:2] - anchor[:2]),
                    )
                    roi_ids.add(t.track_id)
                    roi_speeds.append(float(np.linalg.norm(t.velocity)))
                    roi_states[t.motion_state] = roi_states.get(t.motion_state, 0) + 1
            result["box_roi_proxy"] = dict(
                unique_nearest_ids=len(roi_ids),
                coverage=covered / len(frames),
                competing_track_frames=competition,
                raw_proposals=proposals,
                motion_counts=roi_states,
                speed_p95=float(np.percentile(roi_speeds, 95)) if roi_speeds else None,
            )
    return result


def worker(args):
    from go2_perception.tracking.replay import frame_from_dict
    from go2_perception.tracking.multi_object_tracker import MultiObjectTracker
    from go2_perception.io.config import load_config
    import go2_perception
    from go2_perception.io.serialization import jsonable

    cfg = load_config(args.config)["tracker"]
    report = {}
    for path in sorted(Path(args.cache).glob("*/frames.jsonl")):
        tracker = MultiObjectTracker(cfg)
        frames = []
        outputs = []
        latencies = []
        events = {"fragment_groups": 0}
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
            latencies.append((perf_counter() - start) * 1000)
            events["fragment_groups"] += len(getattr(tracker, "fragment_groups", []))
            frames.append(f)
            outputs.append(tracks)
        report[path.parent.name] = metrics(frames, outputs, latencies, events)
    Path(args.output).write_text(
        json.dumps(
            jsonable(
                dict(
                    package_path=str(Path(go2_perception.__file__).resolve()),
                    sequences=report,
                )
            ),
            indent=2,
        )
        + "\n"
    )


def main(args):
    from go2_perception.tracking.robustness_scenarios import scenarios
    from go2_perception.tracking.replay import recording_frames
    from go2_perception.tracking.config import TrackerConfig
    from go2_perception.tracking.pose_gate import PoseAlignmentConfig
    from go2_perception.io.config import load_config
    from go2_perception.io.serialization import jsonable

    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "inputs"
    expected = (
        {Path(name).name for name in args.recordings}
        if args.recordings
        else set(scenarios(args.seed))
    )
    if args.recordings and len(expected) != len(args.recordings):
        raise ValueError("Recording directory names must be unique")
    existing = {p.parent.name for p in cache.glob("*/frames.jsonl")}
    if existing - expected:
        raise ValueError(
            "Output contains other sequences; use a fresh output directory"
        )
    if args.recordings:
        config = load_config("configs/tracking.yaml")
        for name in args.recordings:
            target = cache / Path(name).name / "frames.jsonl"
            if target.exists() and args.reuse_cache:
                continue
            print("Extracting LiDAR detections locally:", Path(name).name, flush=True)
            report = {}
            rows = []
            for f in recording_frames(
                name,
                TrackerConfig(**config["tracker"]),
                load_config("configs/go2_lidar.yaml"),
                0.3,
                PoseAlignmentConfig(**config["pose_alignment"]),
                report,
            ):
                row = jsonable(f)
                for d in row["detections"]:
                    for key in ("timestamp", "frame_id", "clock_domain"):
                        d.pop(key)
                rows.append(row)
            write_rows(target, rows)
            (target.parent / "pose_report.json").write_text(
                json.dumps(jsonable(report), indent=2)
            )
    else:
        for name, rows in scenarios(args.seed).items():
            write_rows(cache / name / "frames.jsonl", rows)
    baseline = subprocess.check_output(
        ["git", "rev-parse", args.baseline], cwd=ROOT, text=True
    ).strip()
    with tempfile.TemporaryDirectory(prefix="go2-code-baseline-") as tmp:
        archive = subprocess.check_output(
            ["git", "archive", "--format=zip", baseline, "src", "configs"], cwd=ROOT
        )
        with zipfile.ZipFile(io.BytesIO(archive)) as z:
            for info in z.infolist():
                if (
                    ".." in Path(info.filename).parts
                    or Path(info.filename).is_absolute()
                ):
                    raise ValueError("Unsafe archive path")
            z.extractall(tmp)
        reports = {}
        for label, source, config in [
            ("baseline", Path(tmp) / "src", Path(tmp) / args.baseline_config),
            ("improved", ROOT / "src", ROOT / args.config),
        ]:
            print("Evaluating", label, flush=True)
            dest = out / f"{label}.json"
            env = dict(os.environ, PYTHONPATH=str(source))
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--cache",
                    str(cache),
                    "--config",
                    str(config),
                    "--output",
                    str(dest),
                ],
                cwd=ROOT,
                env=env,
                check=True,
            )
            worker_report = json.loads(dest.read_text())
            imported = Path(worker_report["package_path"]).resolve()
            if not imported.is_relative_to(source.resolve()):
                raise RuntimeError(f"Worker imported wrong source: {imported}")
            reports[label] = worker_report["sequences"]
    hashes = {
        p.parent.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(cache.glob("*/frames.jsonl"))
    }
    result = dict(
        baseline_commit=baseline,
        baseline_config=args.baseline_config,
        seed=None if args.recordings else args.seed,
        real_data=bool(args.recordings),
        input_sha256=hashes,
        profile=load_config(args.config),
        results=reports,
    )
    (out / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Comparison saved:", out / "comparison.json", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", default="ecfaae3")
    p.add_argument("--baseline-config", default="configs/tracking.yaml")
    p.add_argument("--config", default="configs/tracking_robustness.yaml")
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=36)
    p.add_argument("--recordings", nargs="+")
    p.add_argument("--reuse-cache", action="store_true")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--cache", help=argparse.SUPPRESS)
    a = p.parse_args()
    worker(a) if a.worker else main(a)
