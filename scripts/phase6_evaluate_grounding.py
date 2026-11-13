"""Run a declared offline case list. Semantic judgments must come from separate review."""

import argparse
import json
from pathlib import Path
import re
import statistics

import yaml

from go2_perception.vln.grounding import GroundingDino, peak_rss_bytes
from go2_perception.vln.grounding_demo import replay_source, run_observation, sha256, write_json
from go2_perception.vln.replay import Go2RecordingReplay

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recordings", type=Path, required=True)
    ap.add_argument("--cases", type=Path, default=ROOT / "configs/vln_grounding_eval.yaml")
    ap.add_argument("--config", type=Path, default=ROOT / "configs/vln_grounding.yaml")
    ap.add_argument("--calibration", type=Path, default=ROOT / "configs/go2_front_camera_calibration.yaml")
    ap.add_argument("--cache-dir", type=Path, default=ROOT / "data/outputs/phase6/hf_cache")
    ap.add_argument("--allow-download", action="store_true")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists() and any(a.output.iterdir()):
        ap.error("use a new or empty output directory")
    declared = yaml.safe_load(a.cases.read_text())
    cases = declared["cases"]
    ids = [c["id"] for c in cases]
    if len(ids) != len(set(ids)) or not all(re.fullmatch(r"[a-z0-9_]+", i) for i in ids):
        ap.error("case IDs must be unique safe directory names")
    a.output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(a.config.read_text())
    # Freeze the declaration and thresholds before the model is loaded/run.
    write_json(a.output / "declared_cases.json", {**declared, "model_config": config,
               "cases_sha256": sha256(a.cases), "calibration_sha256": sha256(a.calibration)})
    model = GroundingDino(config, a.cache_dir, allow_download=a.allow_download)
    write_json(a.output / "model.json", model.info)
    replays, summaries = {}, []
    for c in cases:
        if c["recording"] not in replays:
            replays[c["recording"]] = Go2RecordingReplay(a.recordings / c["recording"], calibration=a.calibration)
        r = replays[c["recording"]]
        source = replay_source(r, c["index"], history=2)
        source["calibration_sha256"] = sha256(a.calibration)
        o = r.observation(c["index"], history=2, instruction=c["instruction"])
        p = run_observation(o, model, a.output / c["id"], source=source)
        row = {"id": c["id"], "category": c["category"], "status": p["status"],
               "candidate_count": len(p["candidates"]), "rejected": p["rejected_candidates"],
               "forward_s": p["timing"]["forward_s"], "semantic_review": "pending independent review"}
        summaries.append(row)
        write_json(a.output / "progress.json", summaries)
        print(json.dumps(row), flush=True)
    latencies = [s["forward_s"] for s in summaries if s["forward_s"] is not None]
    write_json(a.output / "summary.json", {"cases": summaries,
               "mean_forward_s": statistics.mean(latencies) if latencies else None,
               "process_lifetime_peak_rss_bytes": peak_rss_bytes(),
               "semantic_accuracy": None, "semantic_accuracy_note": "Not inferred from model confidence or format validity",
               "runtime_note": "Sequential CPU; first call included, no benchmark warmup; includes no robot execution"})


if __name__ == "__main__":
    main()
