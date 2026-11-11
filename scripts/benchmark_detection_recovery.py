"""Synthetic 5A/recovery comparisons and deterministic viewer-compatible recovery fixture."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
from go2_perception.io.config import load_config
from go2_perception.scene.pipeline import StableScene
from go2_perception.scene.recovery import RecoveryScene
from go2_perception.scene.recovery_synthetic import sequence, SCENARIOS
from go2_perception.scene.evaluation import score_frames
from go2_perception.scene.serialization import write_scenes
from evaluate_detection_recovery import recent_rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--fixture")
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    results = {}
    for name in SCENARIOS:
        results[name] = {}
        data = list(sequence(name))
        for method in ["5a01", "5a03", "recovery01", "recovery03"]:
            recovery = method.startswith("recovery")
            c = load_config(
                "configs/detection_recovery.yaml"
                if recovery
                else "configs/stable_lidar.yaml"
            )
            c["object_window_s"] = 0.1 if method.endswith("01") else 0.3
            s = RecoveryScene(c) if recovery else StableScene(c)
            rows = []
            for i, f in enumerate(data):
                rows.append(
                    s.process(
                        f["points"],
                        f["timestamp"],
                        str(i),
                        f["pose_lookup"],
                        clock_domain="synthetic",
                    )
                )
            truth = [f["truth"] for f in data]
            results[name][method] = dict(
                current=score_frames(list(zip(rows, truth))),
                recent_0p2s=score_frames(list(zip(recent_rows(rows), truth))),
            )
            print(
                name,
                method,
                results[name][method]["current"]["moving_recall"],
                flush=True,
            )
            if a.fixture and name == "missing_sector" and method == "recovery03":
                fixture = deepcopy(rows[18:30])
                for r in fixture:
                    r["diagnostics"]["processing_ms"] = 0.0
                write_scenes(a.fixture, fixture)
        (out / "metrics.json").write_text(
            json.dumps(results, indent=2, allow_nan=False) + "\n"
        )


if __name__ == "__main__":
    main()
