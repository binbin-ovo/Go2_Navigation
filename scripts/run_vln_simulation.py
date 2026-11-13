"""One small synthetic camera room: render smoke, optionally static closed loop.

No Isaac installation, networking or physical Go2. A kinematic proxy is not validated
quadruped locomotion. Simulation pauses during inference; no latency-aware claim.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

from go2_perception.vln.pretrained import SimulationRequest, validate_runtime
from go2_perception.vln.simulation import CameraRoom

ROOT = Path(__file__).resolve().parents[1]


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--closed-loop", action="store_true")
    ap.add_argument("--assets", type=Path)
    ap.add_argument("--device", choices=["cpu", "mps", "cuda"], default="mps")
    ap.add_argument("--steps", type=int, default=16)
    a = ap.parse_args()
    if not 1 <= a.steps <= 32:
        ap.error("Bounded demo supports 1..32 steps")
    if a.closed_loop and not a.assets:
        ap.error("Closed loop requires audited local assets")
    a.output.mkdir(parents=True, exist_ok=False)
    scene = ROOT / "configs/simulation_vln_room.xml"
    t = time.monotonic()
    room = CameraRoom(scene)
    try:
        first = room.render()
        init_s = time.monotonic() - t
        Image.fromarray(first).save(a.output / "camera_initial.png")
        count, equal = 0, 0
        t = time.monotonic()
        while time.monotonic() - t < 5.0:
            frame = room.render()
            count += 1
            equal += int(np.array_equal(first, frame))
        render_s = time.monotonic() - t
        if first.shape != (256, 448, 3) or first.dtype != np.uint8 or first.std() < 5:
            raise ValueError("Invalid/blank RGB camera")
        import mujoco

        summary = {
            "simulator": "MuJoCo",
            "version": mujoco.__version__,
            "scene_sha256": hashlib.sha256(scene.read_bytes()).hexdigest(),
            "initialization_and_first_camera_s": init_s,
            "render_frames": count,
            "render_elapsed_s": render_s,
            "render_fps": count / render_s,
            "camera_wh": [448, 256],
            "rgb_std": float(first.std()),
            "repeat_static_frames_equal": equal == count,
            "render_settings": "OpenGL RGB, one camera/room/proxy, shadow map 1024",
            "nvidia_vram": None,
            "mode": "static_step_and_wait" if a.closed_loop else "render_smoke",
            "physical_execution_authorized": False,
            "closed_loop_executed": False,
            "sim_camera": {
                "ideal_pinhole": True,
                "fovy_degrees": 60,
                "height_m": 0.65,
                "axes": "world/base +x forward,+y left,+z up; camera optical right/down/forward",
                "not_real_go2_calibration": True,
            },
        }
        dump(a.output / "render_smoke.json", summary)
        if not a.closed_loop:
            print(json.dumps(summary, indent=2))
            return
        # The model experiment and rendering have independently passed before this mode.
        os.environ.update(
            HF_HUB_OFFLINE="1",
            TOKENIZERS_PARALLELISM="false",
            LIGHTNAV_ATTN="sdpa",
            VLN_EVAL_TEMPERATURE="0",
            VLN_EVAL_TRAJ_TOP1="0",
            VLN_TIMESTAMP_RELATIVE="1",
        )
        import torch
        from lightnav_worker import verify_assets, run_job, MemorySampler

        source, checkpoint = verify_assets(a.assets)
        config = {
            "device": a.device,
            "backend": "hf",
            "dtype": "bfloat16",
            "attention": "sdpa",
            "quantization": None,
            "max_new_tokens": 64,
            "action_feedback": "none_observation_only",
            "execution_authorized": False,
        }
        hardware = {
            "mps_available": torch.backends.mps.is_available(),
            "cuda_available": torch.cuda.is_available(),
        }
        if a.device == "cuda" and hardware["cuda_available"]:
            hardware.update(
                compute_capability=torch.cuda.get_device_capability(),
                free_vram_bytes=torch.cuda.mem_get_info()[0],
            )
        validate_runtime(config, hardware)
        sys.path.insert(0, str(a.assets / "source-local-v2/src"))
        from lightnav.tracking import build_tracking_agent

        torch.manual_seed(0)
        torch.set_num_threads(4)
        sampler = MemorySampler(a.device)
        sampler.start()
        agent = build_tracking_agent(
            model_path=str(a.assets / "checkpoint"),
            backend="hf",
            device=a.device,
            max_new_tokens=64,
            task_key="vlnce",
        )
        summary["model"] = {
            "revision": checkpoint["revision"],
            "code_revision": source["upstream_revision"],
            "patch_sha256": source["patch_sha256"],
            "runtime": config,
        }
        instruction = (
            "Turn right and approach the doorway, then stop before the doorway."
        )
        summary["instruction"] = instruction
        epoch = f"sim_{time.time_ns()}"
        history = []
        steps = []
        termination = "step_budget_exhausted"
        for i in range(a.steps):
            capture_t = time.monotonic()
            rgb = room.render()
            path = a.output / f"camera_{i:03d}.png"
            Image.fromarray(rgb).save(path)
            history.append(
                {
                    "path": str(path.resolve()),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "observation_s": room.clock,
                    "available_s": room.clock,
                }
            )
            job = {
                "instruction": instruction,
                "episode_id": "synthetic_room",
                "clock_epoch": epoch,
                "decision_id": f"s{i:03d}",
                "decision_time_s": room.clock,
                "frames": list(history),
            }
            request = SimulationRequest(
                "synthetic_room", epoch, i, capture_t, capture_t + 120.0
            )
            result = run_job(agent, job, a.output / "inference" / f"{i:03d}", a.device)
            row = {
                "step": i,
                "before": room.evaluation(),
                "proposal": result,
                "host_observation_age_at_response_s": time.monotonic() - capture_t,
                "sim_time_during_inference_advanced": False,
            }
            try:
                request.accept({**result, "sequence": i}, time.monotonic())
            except ValueError as e:
                row["action"] = {"applied": False, "reason": str(e)}
                termination = "stale_response"
            else:
                if result["status"] != "decoded":
                    row["action"] = {
                        "applied": False,
                        "reason": "unsupported_native_output",
                    }
                    termination = "unsupported_native_output"
                elif result["decoder_stop"]:
                    row["action"] = {
                        "applied": False,
                        "reason": "model_stop",
                        "actual": [0.0, 0.0, 0.0],
                    }
                    termination = "model_stop"
                else:
                    row["action"] = room.advance(result["waypoints"][0])
                    if not row["action"]["applied"]:
                        termination = "simulator_constraint"
            row["after"] = room.evaluation()
            steps.append(row)
            dump(a.output / "steps.json", steps)
            print(
                json.dumps(
                    {
                        "step": i,
                        "native": result.get("native_output"),
                        "action": row["action"],
                        "distance": row["after"]["distance_to_goal_center_m"],
                    }
                ),
                flush=True,
            )
            if termination != "step_budget_exhausted":
                break
        Image.fromarray(room.render()).save(a.output / "camera_final.png")
        summary.update(
            closed_loop_executed=True,
            termination=termination,
            steps=len(steps),
            final=room.evaluation(),
            model_stopped_in_region=bool(
                termination == "model_stop"
                and room.evaluation()["inside_stopping_region"]
            ),
            combined_memory=sampler.finish(),
            latency_aware_test_executed=False,
            limitations=[
                "Synthetic visual domain; one development scene",
                "Kinematic camera proxy, no Go2 locomotion",
                "Static paused inference, no dynamic collision-avoidance evidence",
                "Simulator contact rejection is external constraint logic, not learned avoidance",
            ],
        )
        dump(a.output / "summary.json", summary)
        print(json.dumps(summary, indent=2))
    finally:
        room.close()


if __name__ == "__main__":
    main()
