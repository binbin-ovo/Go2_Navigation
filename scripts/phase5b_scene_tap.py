"""Replay a recording through the unchanged Phase 5A pipeline and record Debug intermediates.

Writes, to a new git-ignored directory:
  scene.jsonl   go2-scene-v1 rows (identical to scripts/run_stable_scene.py output)
  tap.npz       per-scan odom points, acquisition times, non-ground flags; object-window
                points with stage codes (ground / plane / compact / rejected / noise)
  tap.jsonl     per-frame clusters, proposals (with current-scan support) and motion internals
  summary.json  frame count, missing poses, processing time, config, determinism check

Never reads annotations or RGB. ``--reference`` compares every row (except processing_ms)
with an existing scene.jsonl, e.g. the Phase 5A release output.

  PYTHONPATH=src .venv/bin/python scripts/phase5b_scene_tap.py \
      --input ../Go2_Navigation/data/recordings/test_b --output data/outputs/phase5b/tap/test_b
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from go2_perception.adapters.go2.recording import RecordingReader  # noqa: E402
from go2_perception.diagnostics.scene_tap import SceneTap, strip_volatile  # noqa: E402
from go2_perception.io.config import load_config  # noqa: E402
from go2_perception.scene.serialization import validate_scene  # noqa: E402
from go2_perception.tracking.replay import RecordedPoses  # noqa: E402
from go2_perception.viewer.privacy import require_private_dir  # noqa: E402


def _jsonable(x):
    if isinstance(x, np.ndarray):
        return [round(float(v), 4) for v in x.ravel()]
    if isinstance(x, (np.floating, float)):
        return round(float(x), 4)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def set_override(config, item):
    key, value = item.split("=", 1)
    node = config
    parts = key.split(".")
    for p in parts[:-1]:
        node = node[p]
    node[parts[-1]] = type(node[parts[-1]])(float(value)) if not isinstance(node[parts[-1]], bool) else value == "true"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--config", default=str(ROOT / "configs/stable_lidar.yaml"))
    p.add_argument("--set", action="append", default=[], help="diagnostic override, e.g. object_window_s=0.3 (never a default change)")
    p.add_argument("--reference", help="existing scene.jsonl to compare against")
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        sys.exit(f"Refusing to overwrite {out}")
    require_private_dir(out)
    config = load_config(a.config)
    for item in a.set:
        set_override(config, item)
    reader = RecordingReader(a.input)
    poses = RecordedPoses(reader.state("rt/utlidar/robot_odom"))
    tap = SceneTap(config)
    ref = open(a.reference) if a.reference else None
    out.mkdir(parents=True)
    scans, acq, ng, wins, codes = [], [], [], [], []
    frame_t, frame_src = [], []
    latency, missing, mismatches = [], 0, []
    with open(out / "scene.jsonl", "w") as scene_f, open(out / "tap.jsonl", "w") as tap_f:
        for i, entry in enumerate(reader.lidar_frames("rt/utlidar/cloud_base")):
            frame = entry.load()
            if frame.frame_id != "base_link":
                raise ValueError("Raw base_link recording required")
            row, rec = tap.process(frame.xyz, frame.stamp, str(i), poses.lookup, clock_domain="go2_robot", point_offsets=frame.time)
            row["diagnostics"]["source_midpoint_timestamp"] = frame.acquisition_midpoint
            scene_f.write(json.dumps(validate_scene(row), allow_nan=False) + "\n")
            latency.append(row["diagnostics"]["processing_ms"])
            missing += row["robot_pose"] is None
            frame_t.append(row["timestamp"])
            frame_src.append(frame.acquisition_midpoint)
            empty3 = np.empty((0, 3), np.float32)
            scans.append(empty3 if rec.scan_xyz is None else rec.scan_xyz.astype(np.float32))
            acq.append(np.empty(0) if rec.scan_acquired is None else rec.scan_acquired)
            ng.append(np.empty(0, bool) if rec.scan_nonground is None else rec.scan_nonground)
            wins.append(empty3 if rec.window_xyz is None else rec.window_xyz.astype(np.float32))
            codes.append(np.empty(0, np.int8) if rec.window_code is None else rec.window_code)
            tap_f.write(json.dumps(_jsonable(dict(t=row["timestamp"], clusters=rec.clusters, proposals=rec.proposals, internals=rec.objects_internal))) + "\n")
            if ref is not None:
                expected = json.loads(ref.readline())
                if json.dumps(strip_volatile(expected), sort_keys=True) != json.dumps(strip_volatile(json.loads(json.dumps(row))), sort_keys=True):
                    mismatches.append(i)
            if (i + 1) % 200 == 0:
                print("processed", i + 1, flush=True)

    def offsets(parts):
        return np.r_[0, np.cumsum([len(x) for x in parts])]

    np.savez_compressed(
        out / "tap.npz",
        frame_t=np.array(frame_t), frame_src=np.array(frame_src),
        scan_off=offsets(scans), scan_xyz=np.concatenate(scans), scan_acq=np.concatenate(acq), scan_ng=np.concatenate(ng),
        win_off=offsets(wins), win_xyz=np.concatenate(wins), win_code=np.concatenate(codes),
    )  # fmt: skip
    summary = dict(
        frames=len(frame_t),
        missing_pose=int(missing),
        processing_ms=dict(mean=float(np.mean(latency)), p50=float(np.median(latency)), p95=float(np.percentile(latency, 95))),
        config=config,
        overrides=a.set,
        reference=a.reference,
        reference_mismatched_frames=None if ref is None else mismatches,
        git_commit=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=1))


if __name__ == "__main__":
    main()
