"""Raw odom-frame acquisition windows; GT regions are diagnostic measurement ROIs only."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time
import numpy as np
from go2_perception.adapters.go2.recording import RecordingReader
from go2_perception.tracking.replay import RecordedPoses
from go2_perception.tracking.ground_truth import distances
from go2_perception.scene.accumulation import TemporalAccumulator


def audit(root, annotation, stride=15):
    reader = RecordingReader(root)
    poses = RecordedPoses(reader.state("rt/utlidar/robot_odom"))
    a = TemporalAccumulator()
    gt = json.loads(Path(annotation).read_text())
    states = {round(f["t"], 3): f["objects"] for f in gt["frames"]}
    rows = {str(w): defaultdict(list) for w in [0.1, 0.3, 0.6, 1.0]}
    for i, entry in enumerate(reader.lidar_frames("rt/utlidar/cloud_base")):
        f = entry.load()
        new = a.add(
            f.xyz,
            f.stamp,
            str(i),
            poses.lookup,
            clock_domain="go2_robot",
            point_offsets=f.time,
        )
        if new is None or i % stride:
            continue
        for window in [0.1, 0.3, 0.6, 1.0]:
            t = time.perf_counter()
            p = a.snapshot(window, 0.05)
            elapsed = (time.perf_counter() - t) * 1000
            r = rows[str(window)]
            r["occupied_voxels"].append(len(p.xyz))
            r["source_intervals"].append(len(set(p.scan_ids)))
            r["window_build_ms"].append(elapsed)
            body = (p.xyz - a.last_pose[:3, 3]) @ a.last_pose[:3, :3]
            body = body[np.linalg.norm(body[:, :2], axis=1) > 0.5]
            angles = np.arctan2(body[:, 1], body[:, 0])
            r["azimuth_10deg_bins"].append(
                len(np.unique(np.floor((angles + np.pi) / np.deg2rad(10))))
            )
            box = next(
                (
                    s
                    for s in states.get(round(f.acquisition_midpoint, 3), [])
                    if s["id"] == "box" and s.get("region")
                ),
                None,
            )
            if box is not None:
                # Diagnostic oracle ROI only; never feeds the accumulator/proposal pipeline.
                q = p.xyz[
                    (distances(box, p.xyz) < 0.05)
                    & (p.xyz[:, 2] > 0.03)
                    & (p.xyz[:, 2] < 0.5)
                ]
                if len(q) >= 4:
                    r["roi_points"].append(len(q))
                    r["roi_center_residual"].append(
                        (
                            np.median(q[:, :2], axis=0) - np.mean(box["region"], axis=0)
                        ).tolist()
                    )
                    r["roi_xy_extent"].append(np.ptp(q[:, :2], axis=0).tolist())
    result = {}
    for window, values in rows.items():
        out = {}
        for key, v in values.items():
            x = np.asarray(v)
            if key == "roi_center_residual":
                out["static_roi_centroid_jitter_m"] = float(
                    np.linalg.norm(np.std(x, axis=0))
                )
            elif x.ndim == 2:
                out[key + "_mean"] = x.mean(0).tolist()
            else:
                out[key] = dict(
                    mean=float(x.mean()), p95=float(np.percentile(x, 95)), count=len(x)
                )
        result[window] = out
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recordings", required=True)
    p.add_argument("--annotations", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    result = {
        name: audit(Path(a.recordings) / name, Path(a.annotations) / f"{name}.json")
        for name in ["test_a1", "test_a2", "test_b", "test_c"]
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
