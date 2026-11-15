"""Freeze a reviewed Phase 6D suite, then run every declared simulation condition.

No robot/network clients. Each model run is a child process with a hard host-time
budget. Existing files, failed runs and native outputs are never overwritten.
"""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml

from go2_perception.vln.waypoint_declaration import digest, validate_spec
from go2_perception.vln.waypoint_evaluation import GoalRegion, summarize_run
from go2_perception.vln.waypoint_rollout import run_episode, write_json

ROOT = Path(__file__).resolve().parents[1]
CODE_FILES = [
    "scripts/run_waypoint_suite.py",
    "scripts/lightnav_worker.py",
    "src/go2_perception/vln/simulation.py",
    "src/go2_perception/vln/waypoints.py",
    "src/go2_perception/vln/waypoint_rollout.py",
    "src/go2_perception/vln/waypoint_declaration.py",
    "src/go2_perception/vln/waypoint_evaluation.py",
    "src/go2_perception/vln/native_history.py",
    "src/go2_perception/vln/pretrained.py",
]
RUNTIME = {
    "device": "mps",
    "backend": "hf",
    "dtype": "bfloat16",
    "attention": "sdpa",
    "quantization": None,
    "max_new_tokens": 64,
    "action_feedback": "none_observation_only",
    "execution_authorized": False,
}


def freeze(config, destination):
    spec = yaml.safe_load(config.read_text())
    validate_spec(spec, ROOT)
    declared = {
        "schema": "go2-waypoint-suite-frozen-v1",
        "spec": spec,
        "runtime": RUNTIME,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "implementation": {path: digest(ROOT / path) for path in CODE_FILES},
        "source_config_sha256": digest(config),
        "git_commit_at_freeze": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    }
    # Exclusive output creation binds the complete suite before the first result.
    with destination.open("x") as stream:
        json.dump(declared, stream, indent=2, allow_nan=False)
        stream.write("\n")


def load_declaration(path):
    data = json.loads(path.read_text())
    if (
        data.get("schema") != "go2-waypoint-suite-frozen-v1"
        or data.get("runtime") != RUNTIME
    ):
        raise ValueError("Unsupported frozen experiment/runtime")
    validate_spec(data["spec"], ROOT)
    if set(data["implementation"]) != set(CODE_FILES):
        raise ValueError("Missing implementation identities")
    for name, sha in data["implementation"].items():
        if digest(ROOT / name) != sha:
            raise ValueError(f"Implementation changed after declaration: {name}")
    return data


def child(declaration, assets, output, key):
    data = load_declaration(declaration)
    spec = data["spec"]
    conditions = {
        f"{case['id']}__{policy}": (case, policy)
        for case in spec["cases"]
        for policy in spec["policies"]
    }
    if key not in conditions:
        raise ValueError("Undeclared condition")
    case, policy = conditions[key]
    output.mkdir(parents=True, exist_ok=False)
    os.environ.update(
        HF_HUB_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        TOKENIZERS_PARALLELISM="false",
        LIGHTNAV_ATTN="sdpa",
        VLN_EVAL_TEMPERATURE="0",
        VLN_EVAL_TRAJ_TOP1="0",
        VLN_TIMESTAMP_RELATIVE="1",
    )
    import torch
    from lightnav_worker import MemorySampler, run_job, verify_assets
    from go2_perception.vln.native_history import NativeObservationSession
    from go2_perception.vln.pretrained import validate_runtime
    from go2_perception.vln.simulation import CameraRoom

    validate_runtime(RUNTIME, {"mps_available": torch.backends.mps.is_available()})
    source, checkpoint = verify_assets(assets)
    if source["patch_sha256"] != spec["portability_patch_sha256"]:
        raise ValueError("Runtime patch changed")
    sys.path.insert(0, str(assets / "source-local-v2/src"))
    from lightnav.tracking import build_tracking_agent

    torch.manual_seed(0)
    torch.set_num_threads(4)
    sampler = MemorySampler("mps")
    sampler.start()
    started = time.monotonic()
    agent = build_tracking_agent(
        model_path=str(assets / "checkpoint"),
        backend="hf",
        device="mps",
        max_new_tokens=64,
        task_key="vlnce",
    )
    write_json(
        output / "model.json",
        {
            "runtime": RUNTIME,
            "model_revision": checkpoint["revision"],
            "code_revision": source["upstream_revision"],
            "patch_sha256": source["patch_sha256"],
            "processor": type(agent.engine.bundle.processor).__name__,
            "load_s": time.monotonic() - started,
            "load_memory": sampler.finish(),
            "versions": {
                p: importlib.metadata.version(p)
                for p in ("torch", "transformers", "numpy", "mujoco")
            },
        },
    )
    stream = NativeObservationSession()
    room = CameraRoom(ROOT / spec["scene"])
    try:
        summary = run_episode(
            room,
            spec,
            case,
            policy,
            lambda job, dest: run_job(agent, job, dest, "mps", stream=stream),
            output / "rollout",
        )
        print(json.dumps(summary), flush=True)
    finally:
        room.close()


def suite(declaration, assets, output):
    data = load_declaration(declaration)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "frozen_declaration.json", data)
    spec, summaries = data["spec"], []
    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(ROOT / "src"),
        HF_HUB_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        MUJOCO_GL="cgl",
        PYTORCH_MPS_HIGH_WATERMARK_RATIO="0.5",
        PYTORCH_MPS_LOW_WATERMARK_RATIO="0.45",
    )
    for case in spec["cases"]:
        for policy in spec["policies"]:
            key = f"{case['id']}__{policy}"
            destination = output / key
            started = time.monotonic()
            timeout, returncode = False, None
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "run",
                "--declaration",
                str(declaration.resolve()),
                "--assets",
                str(assets.resolve()),
                "--output",
                str(destination.resolve()),
                "--child",
                key,
            ]
            with (output / f"{key}.log").open("x") as log:
                try:
                    result = subprocess.run(
                        command,
                        cwd=ROOT,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=spec["budget"]["max_host_s"],
                        check=False,
                    )
                    returncode = result.returncode
                except subprocess.TimeoutExpired:
                    timeout = True  # subprocess.run kills and waits for its child.
            summary_path = destination / "rollout/summary.json"
            if summary_path.exists() and not timeout and returncode == 0:
                summary = json.loads(summary_path.read_text())
            else:
                steps_path = destination / "rollout/steps.json"
                try:
                    rows = (
                        json.loads(steps_path.read_text())
                        if steps_path.exists()
                        else []
                    )
                except json.JSONDecodeError:
                    rows = []  # retain partial file; don't invent a completed decision.
                termination = "host_time_budget" if timeout else "worker_failure"
                summary = summarize_run(
                    case["pose"], rows, termination, goal=GoalRegion(**spec["goal"])
                )
                summary.update(
                    case=case["id"],
                    policy=policy,
                    incomplete_artifacts=True,
                    failure_stage=termination,
                )
            summary.update(
                host_process_elapsed_s=time.monotonic() - started,
                child_exit_code=returncode,
                hard_host_timeout=timeout,
                artifact_path=str(destination.resolve()),
            )
            summaries.append(summary)
            write_json(
                output / "aggregate.json",
                {
                    "schema": "go2-waypoint-suite-results-v1",
                    "declaration_sha256": digest(declaration),
                    "declared_conditions": len(spec["cases"]) * len(spec["policies"]),
                    "completed_or_failed_conditions": len(summaries),
                    "runs": summaries,
                    "physical_execution_authorized": False,
                    "mode": "static_step_and_wait",
                },
            )
            print(
                json.dumps(
                    {
                        "condition": key,
                        "termination": summary["termination"],
                        "native_stop_inside_goal": summary["native_stop_inside_goal"],
                    }
                ),
                flush=True,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    p = modes.add_parser("freeze")
    p.add_argument(
        "--config", type=Path, default=ROOT / "configs/waypoint_validation.yaml"
    )
    p.add_argument("--output", required=True, type=Path)
    p = modes.add_parser("run")
    p.add_argument("--declaration", required=True, type=Path)
    p.add_argument("--assets", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--child", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.mode == "freeze":
        freeze(args.config, args.output)
    elif args.child:
        child(args.declaration, args.assets, args.output, args.child)
    else:
        suite(args.declaration, args.assets, args.output)


if __name__ == "__main__":
    main()
