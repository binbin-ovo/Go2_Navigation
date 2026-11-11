"""Synthetic scene stability and accumulation tradeoffs; no annotations enter perception."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
import numpy as np
from go2_perception.io.config import load_config
from go2_perception.scene.pipeline import StableScene
from go2_perception.scene.synthetic import sequence, SCENARIOS
from go2_perception.scene.evaluation import score_frames
from go2_perception.scene.accumulation import TemporalAccumulator
from go2_perception.lidar.processor import LidarProcessor
from go2_perception.tracking.multi_object_tracker import MultiObjectTracker
from go2_perception.core.tracking_types import Detection3D


def baseline(data):
    # Unchanged original processor and tracker; odom-normalized synthetic input.
    a = TemporalAccumulator(window_s=0.3)
    c = load_config("configs/go2_lidar.yaml")
    c["ground"]["expected_ground_z"] = 0.0
    processor = LidarProcessor(**c)
    tracker = MultiObjectTracker()
    rows = []
    for i, f in enumerate(data):
        start = time.perf_counter()
        t = f["timestamp"]
        a.add(f["points"], t, str(i), f["pose_lookup"], clock_domain="synthetic")
        cloud = a.snapshot()
        result = processor.process(
            cloud.xyz, timestamp=t, frame_id="odom", clock_domain="synthetic"
        )
        detections = [
            Detection3D(
                t,
                o.center,
                "odom",
                "synthetic",
                str(o.cluster_id),
                o.bbox_min,
                o.bbox_max,
            )
            for o in result.obstacles
        ]
        tracks = tracker.update(
            detections, t, frame_id="odom", clock_domain="synthetic"
        )
        objects = []
        for o in tracks:
            dimensions = (
                (o.bbox_max - o.bbox_min) if o.bbox_min is not None else np.ones(3)
            )
            objects.append(
                dict(
                    object_id=o.track_id,
                    center=o.position.tolist(),
                    dimensions=dimensions.tolist(),
                    velocity=[*o.velocity, 0.0],
                    motion_state=o.motion_state,
                    status="observed_filtered"
                    if o.consecutive_misses == 0
                    else "stale",
                )
            )
        rows.append(
            (
                dict(
                    timestamp=t,
                    objects=objects,
                    diagnostics={"processing_ms": (time.perf_counter() - start) * 1000},
                ),
                f["truth"],
            )
        )
    return score_frames(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    config = load_config("configs/stable_lidar.yaml")
    results = {}
    for name in SCENARIOS:
        data = list(sequence(name))
        results[name] = {"original_0.3": baseline(data)}
        for window in [0.1, 0.3, 0.6, 1.0]:
            c = deepcopy(config)
            c["object_window_s"] = window
            scene = StableScene(c)
            rows = []
            density = []
            intervals = []
            for i, f in enumerate(data):
                row = scene.process(
                    f["points"],
                    f["timestamp"],
                    str(i),
                    f["pose_lookup"],
                    clock_domain="synthetic",
                )
                rows.append((row, f["truth"]))
                density.append(row["diagnostics"].get("accumulated_points", 0))
                intervals.append(row["diagnostics"].get("source_intervals", 0))
            metrics = score_frames(rows)
            metrics["accumulated_points_mean"] = float(np.mean(density))
            metrics["source_intervals_mean"] = float(np.mean(intervals))
            results[name][str(window)] = metrics
            print(
                name, window, metrics["coverage"], metrics["moving_recall"], flush=True
            )
        (out / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
