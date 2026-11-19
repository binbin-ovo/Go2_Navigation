"""Exercise the controller and bounded recorder repeatedly without loading a model or any sensor."""

import argparse
import gc
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp
import tracemalloc

import psutil

from go2_perception.vln.continuous_recording import TraceRecorder
from go2_perception.vln.continuous_simulation import simulate


def measure(fixture, output, episodes):
    if not 2 <= episodes <= 256:
        raise ValueError("choose 2 to 256 measured episodes")
    proc = psutil.Process()
    totals = dict(episodes=episodes, packets=0, plans=0, moving_replacements=0, peak_trace_queue_rows=0)

    def run(n, folder):
        with TraceRecorder(folder / f"trace-{n}.jsonl") as record:
            result = simulate(fixture, "visible_after_search", record)
        return result, record.peak

    with TemporaryDirectory(dir=output) as temp:
        folder = Path(temp)
        tracemalloc.start()
        try:
            for i in range(4):
                run(-1 - i, folder)
            gc.collect()
            initial = tracemalloc.get_traced_memory()[0]
            baseline_rss = peak_rss = proc.memory_info().rss
            tracemalloc.reset_peak()
            for i in range(episodes):
                result, peak = run(i, folder)
                totals["packets"] += result["control_ticks"]
                totals["plans"] += result["plans"]
                totals["moving_replacements"] += result["uninterrupted_replacements"]
                totals["peak_trace_queue_rows"] = max(peak, totals["peak_trace_queue_rows"])
                del result
                peak_rss = max(peak_rss, proc.memory_info().rss)
            gc.collect()
            final, peak = tracemalloc.get_traced_memory()
            totals.update(
                retained_python_delta_bytes=final - initial,
                peak_python_over_baseline_bytes=peak - initial,
                rss_delta_bytes=proc.memory_info().rss - baseline_rss,
                peak_end_of_episode_rss_over_baseline_bytes=peak_rss - baseline_rss,
                queue_capacity=TraceRecorder.capacity,
            )
        finally:
            tracemalloc.stop()
    return totals


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=64)
    args = parser.parse_args()
    if any(os.getenv(k, "0") != "0" for k in ("GO2_HARDWARE", "GO2_SUPERVISED_LIVE")):
        parser.error("disable physical opt-ins")
    parent = Path(os.environ["GO2_WORKSPACE"]) / "runs/continuous"
    parent.mkdir(parents=True, exist_ok=True)
    dest = Path(mkdtemp(prefix="memory-", dir=parent))
    fixture = json.loads(
        (
            Path(__file__).resolve().parents[1] / "examples/supervised_navigation/synthetic_cycle.json"
        ).read_bytes()
    )
    report = measure(fixture, dest, args.episodes)
    (dest / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(dest / "summary.json")
