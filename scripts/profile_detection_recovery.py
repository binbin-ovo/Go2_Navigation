"""Reproduce warm 0.3 s profiling and exact-mask microbenchmarks without saving geometry."""

import argparse
import cProfile
import itertools
import json
import pstats
import resource
import subprocess
import time
from pathlib import Path
import numpy as np
from go2_perception.scene import pipeline
from go2_perception.scene.background import BackgroundMap
from go2_perception.scene.fast_background import FastBackgroundMap
from go2_perception.scene.serialization import validate_scene
from go2_perception.io.config import load_config
from run_stable_scene import replay


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recording", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    namespace = {}
    source = subprocess.check_output(
        ["git", "show", "89d901d:src/go2_perception/scene/serialization.py"], text=True
    )
    exec(
        source.replace("from . import SCHEMA_VERSION", 'SCHEMA_VERSION="go2-scene-v1"'),
        namespace,
    )
    original = namespace["validate_scene"]
    pipeline.validate_scene = original
    c = load_config("configs/stable_lidar.yaml")
    c["object_window_s"] = 0.3
    rows = replay(a.recording, c)
    for _ in range(50):
        next(rows)
    pr = cProfile.Profile()
    pr.enable()
    times = []
    for row in itertools.islice(rows, 150):
        times.append(row["diagnostics"]["processing_ms"])
    pr.disable()
    pr.dump_stats(str(out / "original.prof"))
    with (out / "profile.txt").open("w") as f:
        pstats.Stats(pr, stream=f).sort_stats("tottime").print_stats(30)
    results = dict(
        profile_compute_mean_ms=float(np.mean(times)),
        profile_compute_p95_ms=float(np.percentile(times, 95)),
        peak_rss_platform_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    points = np.random.default_rng(12).uniform(-2, 2, (6000, 3))
    old, new = BackgroundMap(), FastBackgroundMap()
    for i in range(5):
        for m in [old, new]:
            m.update(points, i * 0.1, str(i), np.zeros(3))
    np.testing.assert_array_equal(
        old.stable_mask(points, 0.4), new.stable_mask(points, 0.4)
    )
    fixture = json.loads(
        Path("tests/fixtures/scene/renderer.jsonl").read_text().splitlines()[-1]
    )
    fixture["background"]["cells"] *= 500
    jobs = {
        "original_mask": lambda: old.stable_mask(points, 0.4),
        "optimized_mask": lambda: new.stable_mask(points, 0.4),
        "original_validator": lambda: original(fixture),
        "scalar_validator": lambda: validate_scene(fixture),
    }
    for name, fun in jobs.items():
        elapsed = []
        for _ in range(30):
            t = time.perf_counter()
            fun()
            elapsed.append((time.perf_counter() - t) * 1000)
        results[name + "_median_ms"] = float(np.median(elapsed))
    (out / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
