"""Replay raw incremental cloud_base into scene-v1; never consumes annotations or RGB."""

import argparse
import json
from pathlib import Path
from go2_perception.adapters.go2.recording import RecordingReader
from go2_perception.tracking.replay import RecordedPoses
from go2_perception.io.config import load_config
from go2_perception.scene.pipeline import StableScene
from go2_perception.scene.serialization import write_scenes


def replay(recording, config):
    reader = RecordingReader(recording)
    poses = RecordedPoses(reader.state("rt/utlidar/robot_odom"))
    pipeline = StableScene(config)
    for i, entry in enumerate(reader.lidar_frames("rt/utlidar/cloud_base")):
        frame = entry.load()
        if frame.frame_id != "base_link":
            raise ValueError("Raw base_link recording required")
        row = pipeline.process(
            frame.xyz,
            frame.stamp,
            str(i),
            poses.lookup,
            clock_domain="go2_robot",
            point_offsets=frame.time,
        )
        row["diagnostics"]["source_midpoint_timestamp"] = frame.acquisition_midpoint
        yield row


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--config", default="configs/stable_lidar.yaml")
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    # Real geometry remains local. Repository ignore rules cover data/outputs.
    import subprocess

    if (
        subprocess.run(["git", "check-ignore", "-q", str(out)], check=False).returncode
        != 0
    ):
        raise ValueError("Real replay outputs must be gitignored")
    c = load_config(a.config)
    out.mkdir(parents=True)
    stats = {"frames": 0, "missing_pose": 0, "latency_ms": []}

    def rows():
        for row in replay(a.input, c):
            stats["frames"] += 1
            stats["missing_pose"] += int(row["robot_pose"] is None)
            stats["latency_ms"].append(row["diagnostics"]["processing_ms"])
            if stats["frames"] % 100 == 0:
                print("processed", stats["frames"], flush=True)
            yield row

    write_scenes(out / "scene.jsonl", rows())
    import numpy as np

    latency = stats.pop("latency_ms")
    stats["processing_ms"] = {
        "mean": float(np.mean(latency)),
        "p95": float(np.percentile(latency, 95)),
    }
    stats["config"] = c
    (out / "summary.json").write_text(json.dumps(stats, indent=2) + "\n")
