"""Offline Phase 6A terminal instruction -> inspected image-space proposals."""

import argparse
from pathlib import Path

import yaml

from go2_perception.vln import load
from go2_perception.vln.grounding import GroundingDino
from go2_perception.vln.grounding_demo import replay_source, run_observation, sha256
from go2_perception.vln.replay import Go2RecordingReplay

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--recording", type=Path)
    group.add_argument("--observation", type=Path, help="saved go2-vln-obs-v1 NPZ")
    ap.add_argument("--index", type=int, default=200, help="LiDAR cloud_base reference scan index")
    ap.add_argument("--history", type=int, default=2)
    ap.add_argument("--instruction", required=True)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/vln_grounding.yaml")
    ap.add_argument("--calibration", type=Path, default=ROOT / "configs/go2_front_camera_calibration.yaml")
    ap.add_argument("--cache-dir", type=Path, default=ROOT / "data/outputs/phase6/hf_cache")
    ap.add_argument("--allow-download", action="store_true", help="download pinned public model; images stay local")
    ap.add_argument("--allow-synthetic", action="store_true", help="fixture tests only")
    ap.add_argument("--output", required=True, type=Path)
    a = ap.parse_args()
    if a.history < 0:
        ap.error("history must be nonnegative")
    if a.output.exists() and any(a.output.iterdir()):
        ap.error("output must be a new or empty directory")
    if a.recording:
        replay = Go2RecordingReplay(a.recording, calibration=a.calibration, allow_synthetic=a.allow_synthetic)
        source = replay_source(replay, a.index, a.history)
        source["calibration_file"] = str(a.calibration.resolve())
        source["calibration_sha256"] = sha256(a.calibration)
        obs = replay.observation(a.index, history=a.history, instruction=a.instruction)
    else:
        obs = load(a.observation)
        obs.instruction = a.instruction
        source = {"observation_file": str(a.observation.resolve()), "sha256": sha256(a.observation)}
    model = GroundingDino(yaml.safe_load(a.config.read_text()), a.cache_dir, allow_download=a.allow_download)
    p = run_observation(obs, model, a.output, source=source, allow_synthetic=a.allow_synthetic)
    print(f"{p['status']}: {len(p['candidates'])} image candidate(s); executable=False")
    print(a.output.resolve() / "demo.png")


if __name__ == "__main__":
    main()
