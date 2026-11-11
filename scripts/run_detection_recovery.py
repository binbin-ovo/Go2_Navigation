"""Opt-in recovery/unchanged 5A comparison; private JSONL with compute, wall and RSS accounting."""

import argparse
import json
import resource
import time
from pathlib import Path
import numpy as np
from go2_perception.adapters.go2.recording import RecordingReader
from go2_perception.tracking.replay import RecordedPoses
from go2_perception.io.config import load_config
from go2_perception.scene.pipeline import StableScene
from go2_perception.scene.recovery import RecoveryScene
from go2_perception.viewer.privacy import require_private_dir


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--window", type=float, choices=[0.1, 0.3], default=0.3)
    p.add_argument("--baseline", action="store_true")
    p.add_argument("--config", default="configs/detection_recovery.yaml")
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    require_private_dir(out)
    out.mkdir(parents=True)
    c = load_config("configs/stable_lidar.yaml" if a.baseline else a.config)
    c["object_window_s"] = a.window
    pipeline = StableScene(c) if a.baseline else RecoveryScene(c)
    reader = RecordingReader(a.input)
    poses = RecordedPoses(reader.state("rt/utlidar/robot_odom"))
    compute, wall, age = [], [], []
    t0 = time.perf_counter()
    with (out / "scene.jsonl").open("w") as f:
        for i, entry in enumerate(reader.lidar_frames("rt/utlidar/cloud_base")):
            start = time.perf_counter()
            frame = entry.load()
            if frame.frame_id != "base_link":
                raise ValueError("Raw base_link required")
            row = pipeline.process(
                frame.xyz,
                frame.stamp,
                str(i),
                poses.lookup,
                clock_domain="go2_robot",
                point_offsets=frame.time,
            )
            row["diagnostics"]["source_midpoint_timestamp"] = frame.acquisition_midpoint
            compute.append(row["diagnostics"]["processing_ms"])
            age.extend(o["observation_age_s"] for o in row["objects"])
            f.write(json.dumps(row, allow_nan=False) + "\n")
            wall.append((time.perf_counter() - start) * 1000)
            if i % 200 == 0:
                print(i, round(wall[-1], 2), flush=True)

    def summary(v):
        return (
            dict(
                mean=float(np.mean(v)),
                p95=float(np.percentile(v, 95)),
                max=float(max(v)),
            )
            if v
            else None
        )

    stats = dict(
        frames=len(wall),
        processing_ms=summary(compute),
        wall_ms=summary(wall),
        observation_age_s=summary(age),
        peak_rss_platform_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        elapsed_s=time.perf_counter() - t0,
        config=c,
        baseline=a.baseline,
    )
    (out / "summary.json").write_text(json.dumps(stats, indent=2) + "\n")


if __name__ == "__main__":
    main()
