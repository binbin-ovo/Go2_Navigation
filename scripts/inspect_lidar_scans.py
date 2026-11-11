"""Read-only acquisition and overlap audit; exports aggregates, never private points/images."""

import argparse
from pathlib import Path
import json
import numpy as np
from go2_perception.adapters.go2.recording import RecordingReader
from go2_perception.tracking.replay import RecordedPoses


def describe(x):
    return (
        dict(
            min=float(np.min(x)),
            median=float(np.median(x)),
            p95=float(np.percentile(x, 95)),
            max=float(np.max(x)),
        )
        if len(x)
        else None
    )


def inspect(root):
    reader = RecordingReader(root)
    records = reader.lidar_frames("rt/utlidar/cloud_base")
    poses = RecordedPoses(reader.state("rt/utlidar/robot_odom"))
    lengths = []
    spans = []
    mins = []
    steps = []
    overlap = []
    delta = []
    missing = 0
    negative = 0
    prev = None
    stamps = []
    for record in records:
        f = record.load()
        lengths.append(len(f.xyz))
        stamps.append(f.stamp)
        if f.time is None:
            missing += 1
            continue
        spans.append(float(np.ptp(f.time)))
        mins.append(float(f.time.min()))
        negative += int((np.diff(f.time) < 0).sum())
        start, end = f.stamp + float(f.time.min()), f.stamp + float(f.time.max())
        a, b = poses.lookup(start), poses.lookup(end)
        if a is not None and b is not None:
            rotated = f.xyz @ a[:3, :3].T + a[:3, 3]
            late = f.xyz @ b[:3, :3].T + b[:3, 3]
            delta.append(
                float(np.percentile(np.linalg.norm(late - rotated, axis=1), 95))
            )
        rounded = set(map(tuple, np.round(f.xyz, 4)))
        if prev is not None:
            overlap.append(len(rounded & prev) / max(1, len(rounded)))
        prev = rounded
    stamps = np.array(stamps)
    steps = np.diff(stamps)
    windows = {
        str(w): describe(
            [int(((stamps <= t) & (stamps >= t - w)).sum()) for t in stamps]
        )
        for w in [0.1, 0.3, 0.6, 1.0]
    }
    return dict(
        frames=len(records),
        frame_ids=sorted({r.record["frame_id"] for r in records}),
        has_time_frames=len(records) - missing,
        point_counts=describe(lengths),
        message_step_s=describe(steps),
        point_time_span_s=describe(spans),
        point_time_min_s=describe(mins),
        within_message_time_decreases=negative,
        exact_xyz_overlap_rounded_0p1mm=describe(overlap),
        within_scan_pose_displacement_p95_m=describe(delta),
        frames_per_window=windows,
        interpretation="Pose-induced endpoint displacement is a distortion sensitivity bound, not measured scan distortion or deskew ground truth.",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recordings", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    result = {
        name: inspect(Path(a.recordings) / name)
        for name in ["test_a1", "test_a2", "test_b", "test_c"]
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
