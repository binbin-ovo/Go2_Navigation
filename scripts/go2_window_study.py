#!/usr/bin/env python
"""Measure the tradeoff between single cloud_base frames and short compensated windows.

Reports density, spatial coverage (occupied 10 cm voxels), and clustering per window length,
plus convention checks against the robot's own rt/utlidar/cloud_deskewed output.
Usage: .venv/bin/python scripts/go2_window_study.py --recording R [--windows 0 0.13 0.3 0.65 1]
"""
import argparse
import json

import numpy as np

from go2_perception.adapters.go2.pointcloud import level_points
from go2_perception.adapters.go2.recording import RecordingReader
from go2_perception.adapters.go2.window import PoseBuffer, SlidingWindowAccumulator
from go2_perception.io.config import load_config
from go2_perception.lidar.processor import LidarProcessor

BASE, DESKEW, ODOM = "rt/utlidar/cloud_base", "rt/utlidar/cloud_deskewed", "rt/utlidar/robot_odom"


def occupied_voxels(points, size=0.1):
    return len(np.unique(np.floor(points / size).astype(np.int32), axis=0)) if len(points) else 0


def pose_check(reader, poses):
    """Does T_odom_base(odom) map a cloud_base frame onto the new part of cloud_deskewed?"""
    base = {round(f.record["stamp"], 6): f for f in reader.lidar_frames(BASE)}
    offsets_by_lag, used = {}, 0
    for replay in reader.lidar_frames(DESKEW)[10::10]:
        d = replay.load()
        new = d.xyz[10000:]
        candidates = [f for s, f in base.items() if len(f.load().xyz) == len(new)
                      and abs(s - d.stamp) < 0.3] if len(new) else []
        if not candidates:
            continue
        f = min(candidates, key=lambda c: abs(c.record["stamp"] - d.stamp)).load()
        pose = poses.lookup(f.acquisition_midpoint)
        if pose is None:
            continue
        mapped = f.xyz.astype(float) @ pose[:3, :3].T + pose[:3, 3]
        error = np.linalg.norm(mapped - new, axis=1)
        offsets_by_lag.setdefault("median_m", []).append(float(np.median(error)))
        offsets_by_lag.setdefault("p95_m", []).append(float(np.percentile(error, 95)))
        used += 1
    return {k: round(float(np.median(v)), 4) for k, v in offsets_by_lag.items()} | {"frames": used}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recording", required=True)
    p.add_argument("--windows", type=float, nargs="+", default=[0, 0.13, 0.3, 0.65, 1.0, 2.0])
    p.add_argument("--config", default="configs/go2_lidar.yaml")
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--json")
    a = p.parse_args()
    reader = RecordingReader(a.recording)
    odom = reader.state(ODOM)
    poses = PoseBuffer(max_age_s=1e9)
    for r in odom:
        poses.add(r["stamp"], r["position"], r["q_xyzw"])
    motion = np.array([r["position"] for r in odom])
    frames = [f.load() for f in reader.lidar_frames(BASE)]
    processor = LidarProcessor(**load_config(a.config))
    stamps = np.array([r["stamp"] for r in odom])
    rows = []
    for window_s in a.windows:
        counts, cover, clusters, ground, unclassified, latency = [], [], [], [], [], []
        acc = SlidingWindowAccumulator(poses, max(window_s, 1e-3), max_frames=100)
        for i, f in enumerate(frames):
            if not acc.add(f.xyz, f.acquisition_midpoint):
                continue
            if i % a.stride or f.stamp - frames[0].stamp < max(a.windows):
                continue
            merged, t, _, _ = acc.merged()
            q = odom[int(np.argmin(np.abs(stamps - t)))]["q_xyzw"]
            level = level_points(merged, q)
            result = processor.process(level, timestamp=t, frame_id="base_link_level", clock_domain="go2_robot")
            counts.append(len(merged))
            cover.append(occupied_voxels(merged))
            clusters.append(len(result.obstacles))
            ground.append(len(result.ground))
            unclassified.append(len(result.unclassified))
            latency.append(result.latency_ms)
        rows.append(
            {
                "window_s": window_s,
                "evaluated_frames": len(counts),
                "points_median": float(np.median(counts)),
                "occupied_10cm_voxels_median": float(np.median(cover)),
                "clusters_median": float(np.median(clusters)),
                "clusters_std": round(float(np.std(clusters)), 2),
                "ground_points_median": float(np.median(ground)),
                "unclassified_points_median": float(np.median(unclassified)),
                "processor_ms_median": round(float(np.median(latency)), 2),
            }
        )
    report = {
        "recording": a.recording,
        "odom_position_span_m": np.ptp(motion, axis=0).round(4).tolist(),
        "odom_to_deskewed_check": pose_check(reader, poses),
        "windows": rows,
    }
    print(json.dumps(report, indent=2))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
