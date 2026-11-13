"""Unchanged episode packages -> calibrated causal inputs -> isolated local model worker.

Annotations are intentionally not accepted. Run evaluate_pretrained_vln.py separately
after inference. The worker imports no robot client and cannot execute a proposal.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess

import yaml

from go2_perception.vln.episodes import load_episode, file_sha256
from go2_perception.vln.pretrained import validate_experiment_config
from go2_perception.vln.pretrained_inputs import prepare_job

ROOT = Path(__file__).resolve().parents[1]


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episode", type=Path, action="append", required=True)
    ap.add_argument("--decision")
    ap.add_argument("--assets", type=Path, required=True)
    ap.add_argument("--model-python", type=Path, required=True)
    ap.add_argument("--device", choices=["cpu", "mps", "cuda"], required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--allow-synthetic", action="store_true")
    a = ap.parse_args()
    config = ROOT / "configs/pretrained_vln.yaml"
    configuration = yaml.safe_load(config.read_text())
    validate_experiment_config(configuration)
    episodes = [load_episode(p) for p in a.episode]
    if len({e.data["episode_id"] for e in episodes}) != len(episodes):
        ap.error("Duplicate episode")
    a.output.mkdir(parents=True, exist_ok=False)
    declaration = {
        "config": configuration,
        "config_sha256": file_sha256(config),
        "device": a.device,
        "episodes": [{"path": str(e.path), "sha256": e.sha256} for e in episodes],
        "input_protocol": "native LightNav SlowFast following causal 4Hz resampling",
        "comparison_mapping_frozen_before_inference": True,
        "jobs": [],
    }
    jobs = []
    for ep in episodes:
        for decision in ep.data["decisions"]:
            if a.decision and decision["decision_id"] != a.decision:
                continue
            i = len(jobs)
            folder = a.output / "prepared" / f"{i:03d}"
            job, provenance = prepare_job(
                ep, decision["decision_id"], folder, allow_synthetic=a.allow_synthetic
            )
            dump(folder / "provenance.json", provenance)
            jobs.append(job)
            declaration["jobs"].append(
                {
                    "index": i,
                    "episode_id": ep.data["episode_id"],
                    "decision_id": decision["decision_id"],
                    "manifest_sha256": ep.sha256,
                }
            )
    if not jobs:
        ap.error("No matching decisions")
    jobs_path = a.output / "jobs.json"
    dump(jobs_path, {"schema": "lightnav-local-jobs-v1", "jobs": jobs})
    declaration["jobs_sha256"] = file_sha256(jobs_path)
    declaration["code_hashes"] = {
        str(p.relative_to(ROOT)): file_sha256(p)
        for p in [
            ROOT / "scripts/lightnav_worker.py",
            ROOT / "scripts/run_pretrained_vln.py",
            ROOT / "src/go2_perception/vln/pretrained.py",
            ROOT / "src/go2_perception/vln/pretrained_inputs.py",
        ]
    }
    dump(a.output / "run_declaration.json", declaration)
    print(f"Prepared {len(jobs)} causal jobs; no annotations loaded", flush=True)
    if a.prepare_only:
        return
    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(ROOT / "src"),
        HF_HUB_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="4",
        PYTORCH_MPS_HIGH_WATERMARK_RATIO="0.5",
        PYTORCH_MPS_LOW_WATERMARK_RATIO="0.45",
    )
    command = [
        # Keep the venv launcher path: resolve() follows its symlink to the base
        # interpreter and silently drops the model environment's site-packages.
        str(a.model_python.absolute()),
        "-u",
        str(ROOT / "scripts/lightnav_worker.py"),
        "--assets",
        str(a.assets.resolve()),
        "--jobs",
        str(jobs_path.resolve()),
        "--device",
        a.device,
        "--output",
        str((a.output / "inference").resolve()),
    ]
    dump(
        a.output / "worker_command.json",
        {
            "argv": command,
            "environment_overrides": {
                k: env[k]
                for k in (
                    "PYTHONPATH",
                    "HF_HUB_OFFLINE",
                    "OMP_NUM_THREADS",
                    "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
                    "PYTORCH_MPS_LOW_WATERMARK_RATIO",
                )
            },
        },
    )
    with (a.output / "worker.log").open("x") as log:
        result = subprocess.run(
            command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=7200
        )
    if result.returncode:
        raise RuntimeError(
            f"Model worker exit {result.returncode}; inspect {a.output / 'worker.log'}"
        )
    print(f"Saved inference to {a.output / 'inference'}", flush=True)


if __name__ == "__main__":
    main()
