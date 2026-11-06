#!/usr/bin/env python
"""Measure how a moving object smears in ego-compensated LiDAR windows.

Background is taken from an initial period with no motion, and points outside it are
foreground. Reported spread is the 5th-95th percentile extent of foreground points.
Usage: .venv/bin/python scripts/go2_moving_object_study.py --recording R --static-until 30
"""
import argparse
import json

import numpy as np

from go2_perception.adapters.go2.recording import RecordingReader
from go2_perception.adapters.go2.window import PoseBuffer, SlidingWindowAccumulator


def voxels(points, size):
    return np.floor(points / size).astype(np.int32)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recording", required=True)
    p.add_argument("--static-until", type=float, required=True, help="seconds after start")
    p.add_argument("--windows", type=float, nargs="+", default=[0, 0.13, 0.3, 0.65, 1.0, 2.0])
    p.add_argument("--voxel", type=float, default=0.2)
    p.add_argument("--min-foreground", type=int, default=15)
    p.add_argument("--json")
    a = p.parse_args()
    reader = RecordingReader(a.recording)
    poses = PoseBuffer(max_age_s=1e9)
    for r in reader.state("rt/utlidar/robot_odom"):
        poses.add(r["stamp"], r["position"], r["q_xyzw"])
    frames = [f.load() for f in reader.lidar_frames("rt/utlidar/cloud_base")]
    t0 = frames[0].stamp
    background = set()
    for f in frames:
        if f.stamp - t0 < a.static_until:
            background.update(map(tuple, voxels(f.xyz, a.voxel)))

    def foreground(points):
        keep = np.array([tuple(v) not in background for v in voxels(points, a.voxel)], bool)
        return points[keep]

    centers = []
    for f in frames:
        fg = foreground(f.xyz)
        centers.append(np.median(fg, axis=0) if len(fg) >= 5 else [np.nan] * 3)
    centers = np.array(centers)
    moving = ~np.isnan(centers[:, 0])
    rows = []
    for window_s in a.windows:
        acc = SlidingWindowAccumulator(poses, max(window_s, 1e-3), max_frames=100)
        spreads, counts = [], []
        for f in frames:
            if not acc.add(f.xyz, f.acquisition_midpoint):
                continue
            merged, *_ = acc.merged()
            if f.stamp - t0 < a.static_until:
                continue
            fg = foreground(merged)
            if len(fg) < a.min_foreground:
                continue
            xy = fg[:, :2]
            spreads.append(np.percentile(xy, 95, axis=0) - np.percentile(xy, 5, axis=0))
            counts.append(len(fg))
        rows.append(
            {
                "window_s": window_s,
                "evaluated_frames": len(spreads),
                "foreground_points_median": float(np.median(counts)) if counts else 0,
                "xy_spread_m_median": np.median(spreads, axis=0).round(3).tolist() if spreads else None,
                "xy_spread_m_p90": np.percentile(spreads, 90, axis=0).round(3).tolist() if spreads else None,
            }
        )
    stamps = np.array([f.stamp for f in frames])
    track = centers[moving, :2]
    speed = None
    if len(track) > 8:
        t = stamps[moving]
        keep = np.flatnonzero(np.diff(t) < 0.5)
        v = np.linalg.norm(np.diff(track, axis=0)[keep], axis=1) / np.diff(t)[keep]
        speed = {"median_mps": round(float(np.median(v)), 3), "p90_mps": round(float(np.percentile(v, 90)), 3)}
    report = {
        "recording": a.recording,
        "static_until_s": a.static_until,
        "background_voxels": len(background),
        "frames_with_foreground": int(moving.sum()),
        "foreground_activity_s": [round(float(stamps[moving][0] - t0), 1),
                                  round(float(stamps[moving][-1] - t0), 1)] if moving.any() else None,
        "single_frame_centroid_speed": speed,
        "windows": rows,
    }
    print(json.dumps(report, indent=1))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
