"""Check whether the Go2 LiDAR returns flat surfaces as flat (offline, read-only).

Peels the three largest planes off a stationary recording (RANSAC + SVD, 3 cm tolerance, points within
2.5 m) for raw `rt/utlidar/cloud` (sensor frame) and `rt/utlidar/cloud_base` (base_link), and prints
their normals and pairwise angles. In a room the floor should come out as one plane and a wall at 90°
to it; two near-parallel planes with the floor's normal mean the floor itself is returned bent.

Usage:
  PYTHONPATH=src python scripts/phase5d_lidar_flatness.py --recording data/recordings/p5d_extr_snap_wallcheck2 \
      --recording data/recordings/p5d_extr_snap_wall_A2 --output data/outputs/phase5d/lidar_flatness.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from go2_perception.vln.extrinsics import fit_plane


def load(root, topic, scans):
    rows = [json.loads(line) for line in open(root / "lidar" / topic / "index.jsonl")][:scans]
    return np.concatenate([np.load(root / "lidar" / topic / r["file"])[:, :3].astype(float) for r in rows])


def peel(points, count=3, tol=0.03):
    planes, rest = [], points
    for _ in range(count):
        n, d, inl, std = fit_plane(rest, iters=600, tol=tol)
        planes.append({"normal": n.round(4).tolist(), "d_m": round(d, 4), "points": int(inl.sum()),
                       "std_mm": round(std * 1000, 1)})
        rest = rest[~inl]
    pairs = {f"{i + 1}-{j + 1}": round(float(np.degrees(np.arccos(min(1.0, abs(
        np.dot(planes[i]["normal"], planes[j]["normal"])))))), 2) for i in range(count) for j in range(i + 1, count)}
    return planes, pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recording", type=Path, action="append", required=True)
    ap.add_argument("--scans", type=int, default=40)
    ap.add_argument("--max-range-m", type=float, default=2.5)
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    report = {}
    for root in a.recording:
        for topic in ("utlidar_cloud", "utlidar_cloud_base"):
            p = load(root, topic, a.scans)
            planes, pairs = peel(p[np.linalg.norm(p, axis=1) < a.max_range_m])
            report[f"{root.name}/{topic}"] = {"planes": planes, "pairwise_angle_deg": pairs}
            print(f"{root.name} {topic}: " + " | ".join(
                f"n {np.round(q['normal'], 2)} N {q['points']}" for q in planes) + f"\n    angles {pairs}")
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
