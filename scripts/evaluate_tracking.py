"""Evaluate annotated detection JSONL; unannotated data gets runtime metrics only."""

import argparse
import json
from pathlib import Path
from time import perf_counter
from go2_perception.io.config import load_config
from go2_perception.io.serialization import save_json
from go2_perception.tracking.replay import load_detection_sequence
from go2_perception.tracking.multi_object_tracker import MultiObjectTracker
from go2_perception.tracking.evaluation import evaluate

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--config", default="configs/tracking.yaml")
    p.add_argument("--match-distance", type=float, default=1.0)
    a = p.parse_args()
    root = Path(a.input)
    paths = (
        [root]
        if root.is_file() or (root / "frames.jsonl").exists()
        else sorted(p.parent for p in root.glob("*/frames.jsonl"))
    )
    if not paths:
        raise ValueError("No detection sequences found")
    report = {}
    for path in paths:
        frames, _ = load_detection_sequence(path)
        tracker = MultiObjectTracker(load_config(a.config)["tracker"])
        outputs = []
        latencies = []
        for f in frames:
            start = perf_counter()
            outputs.append(
                tracker.update(
                    f.detections,
                    f.timestamp,
                    f.robot_pose,
                    frame_id=f.frame_id,
                    clock_domain=f.clock_domain,
                )
            )
            latencies.append((perf_counter() - start) * 1000)
        report[path.stem] = evaluate(frames, outputs, latencies, a.match_distance)
    save_json(a.output, report)
    print(json.dumps(report, indent=2))
