"""Local inference-only LightNav worker in its isolated Python 3.11 environment.

Reads a whitelisted JSON job file; writes native outputs/tensors/metadata. No server,
SDK, ROS, robot client, annotation loader or motion transport is imported.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback

import numpy as np
from PIL import Image
import psutil
import torch

try:
    import resource
except ImportError:  # Native Windows: sampled process memory remains available.
    resource = None

from go2_perception.vln.pretrained import (
    CODE_REVISION,
    MODEL_REVISION,
    native_token_contract,
    validate_runtime,
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


class MemorySampler:
    def __init__(self, device):
        self.device = device
        self.stop = threading.Event()
        self.peak = {}
        self.thread = threading.Thread(target=self._run, daemon=True)

    def sample(self):
        values = {"sampled_rss_bytes": psutil.Process().memory_info().rss}
        if self.device == "mps":
            values.update(
                mps_allocated_bytes=torch.mps.current_allocated_memory(),
                mps_driver_bytes=torch.mps.driver_allocated_memory(),
            )
        if self.device == "cuda":
            values.update(
                cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
            )
        for key, value in values.items():
            self.peak[key] = max(value, self.peak.get(key, 0))

    def _run(self):
        while not self.stop.wait(0.02):
            self.sample()

    def start(self):
        self.sample()
        self.thread.start()

    def finish(self):
        self.stop.set()
        self.thread.join()
        self.sample()
        peak = (
            None
            if resource is None
            else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        )
        return {
            **self.peak,
            "process_lifetime_peak_rss_bytes": None
            if peak is None
            else peak
            if sys.platform == "darwin"
            else peak * 1024,
            "sampling_interval_s": 0.02,
            "mps_peak_is_sampled_lower_bound": True,
        }


def verify_assets(root):
    source = json.loads((root / "source_local_v2_identity.json").read_text())
    checkpoint = json.loads((root / "checkpoint_identity.json").read_text())
    if (
        source["upstream_revision"] != CODE_REVISION
        or checkpoint["revision"] != MODEL_REVISION
    ):
        raise ValueError("Unexpected code/checkpoint revision")
    for identity, folder in ((source, "source-local-v2"), (checkpoint, "checkpoint")):
        for f in identity["files"]:
            if sha(root / folder / f["path"]) != f["sha256"]:
                raise ValueError(f"Changed {folder} asset: {f['path']}")
    return source, checkpoint


def validate_jobs(data):
    if set(data) != {"schema", "jobs"} or data["schema"] != "lightnav-local-jobs-v1":
        raise ValueError("Unsupported job contract")
    for job in data["jobs"]:
        if set(job) != {
            "instruction",
            "episode_id",
            "clock_epoch",
            "decision_id",
            "decision_time_s",
            "frames",
        }:
            raise ValueError(
                "Job whitelist excludes annotations, state and action feedback"
            )
        if (
            not isinstance(job["instruction"], str)
            or not job["instruction"].strip()
            or not job["frames"]
        ):
            raise ValueError("Instruction and causal frames required")
        previous = -float("inf")
        for frame in job["frames"]:
            if set(frame) != {"path", "sha256", "observation_s", "available_s"}:
                raise ValueError("Unexpected frame context")
            obs, available, cutoff = (
                frame["observation_s"],
                frame["available_s"],
                job["decision_time_s"],
            )
            if (
                not all(np.isfinite([obs, available, cutoff]))
                or not previous < obs <= available <= cutoff
            ):
                raise ValueError("Noncausal/invalid frame time")
            if sha(frame["path"]) != frame["sha256"]:
                raise ValueError("Prepared frame changed")
            previous = obs


def tensor_archive(values):
    arrays, types = {}, {}
    for key, value in values.items():
        if not isinstance(value, torch.Tensor):
            continue
        tensor = value.detach().contiguous().cpu()
        types[key] = {"dtype": str(tensor.dtype), "shape": list(tensor.shape)}
        # Preserve exact BF16 bits (NumPy has no native BF16); do not convert to FP16.
        arrays[key] = (
            tensor.view(torch.uint16).numpy()
            if tensor.dtype == torch.bfloat16
            else tensor.numpy()
        )
    return arrays, types


def run_job(agent, job, dest, device):
    from lightnav.inference.samples import build_vln_traj_sample
    from lightnav.vln_utils import (
        safe_parse_apos_token,
        safe_parse_opos_token,
        posxy_channels,
        posxy_sentinels,
    )

    dest.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    sampler = MemorySampler(device)
    sampler.start()
    result = {
        "episode_id": job["episode_id"],
        "clock_epoch": job["clock_epoch"],
        "decision_id": job["decision_id"],
        "execution_authorized": False,
        "status": "error",
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    # Reset also between decisions: reconstruct real causal observations only.
    agent.reset(instruction=job["instruction"])
    try:
        for frame in job["frames"]:
            agent.observe(np.asarray(Image.open(frame["path"]).convert("RGB")))
        sample = build_vln_traj_sample(
            agent._get_video_tensor(), job["instruction"], None, agent.engine.bundle
        )
        segments = [
            {k: v for k, v in s.items() if k != "video"}
            for s in sample["video_segments"]
        ]
        dump(
            dest / "native_sample.json",
            {k: v for k, v in sample.items() if k != "video_segments"}
            | {"video_segments": segments},
        )
        # Actual model-space input images, with upstream normalization inverted once.
        selected_ids = sorted(
            {i for s in sample["video_segments"] for i in s["frame_indices"]}
        )
        for i in selected_ids:
            rgb = agent._get_video_tensor()[i].permute(1, 2, 0).numpy()
            Image.fromarray(
                np.clip(np.rint((rgb + 1) * 127.5), 0, 255).astype(np.uint8)
            ).save(dest / f"input_{i:04d}.png")
        result["native_selected_frame_indices"] = selected_ids
        result["native_segments"] = segments
        model = agent.engine.bundle.model
        original_generate = model.generate
        captured, types = {}, {}

        def generate(**kwargs):
            nonlocal captured, types
            captured, types = tensor_archive(kwargs)
            ids = kwargs["input_ids"]
            if ids.shape[-1] > agent.engine.bundle.max_seq_len:
                raise ValueError("Native context exceeds checkpoint sequence budget")
            (dest / "prompt.txt").write_text(
                agent.engine.bundle.tokenizer.decode(ids[0], skip_special_tokens=False)
            )
            out = original_generate(**kwargs)
            captured["generated_ids"] = out.detach().cpu().numpy()
            return out

        model.generate = generate
        prep_done = time.monotonic()
        try:
            text, native_latency_ms = agent.engine.generate(sample)
        finally:
            model.generate = original_generate
        inference_done = time.monotonic()
        (dest / "native_output.txt").write_text(text)
        result.update(
            status="generated",
            native_output=text,
            timing={
                "prepare_s": prep_done - started,
                "engine_call_s": inference_done - prep_done,
                "native_generate_s": native_latency_ms / 1000,
                "replay_age_at_output_s": job["decision_time_s"]
                - job["frames"][-1]["observation_s"]
                + inference_done
                - started,
            },
        )
        np.savez_compressed(dest / "model_inputs_and_generated_ids.npz", **captured)
        dump(dest / "tensor_types.json", types)
        result["pointing"] = {
            "apos": safe_parse_apos_token(text),
            "opos": safe_parse_opos_token(text),
            "posxy": posxy_channels(text),
            "sentinels": posxy_sentinels(text),
        }
        try:
            codes = native_token_contract(text)
            wp, _ = agent.decode_waypoints(text)
            result.update(
                status="decoded",
                rvq_codes=codes,
                waypoints=wp.tolist(),
                decoder_stop=bool(np.all(wp == 0)),
                explicit_stop_code=bool(agent.rvq.is_stop(codes)),
                waypoint_convention="absolute_ego_SE2_x_forward_y_left_yaw_CCW_metres_radians",
                waypoint_time_base=None,
                completion_verified=False,
            )
        except ValueError as e:
            result["decode_error"] = str(e)
    except Exception as e:
        result["error"] = repr(e)
        (dest / "error.txt").write_text(traceback.format_exc())
    finally:
        result["memory"] = sampler.finish()
        result["wall_s_including_evidence"] = time.monotonic() - started
        dump(dest / "result.json", result)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--assets", type=Path, required=True)
    ap.add_argument("--jobs", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", choices=["cpu", "mps", "cuda"], required=True)
    a = ap.parse_args()
    # Offline flags apply even when the process itself has network access.
    os.environ.update(
        HF_HUB_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        TOKENIZERS_PARALLELISM="false",
        LIGHTNAV_ATTN="sdpa",
        VLN_EVAL_TEMPERATURE="0",
        VLN_EVAL_TRAJ_TOP1="0",
        VLN_TIMESTAMP_RELATIVE="1",
    )
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
    data = json.loads(a.jobs.read_text())
    validate_jobs(data)
    source, checkpoint = verify_assets(a.assets)
    a.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(a.assets / "source-local-v2/src"))
    from lightnav.tracking import build_tracking_agent

    torch.set_num_threads(4)
    torch.manual_seed(0)
    sampler = MemorySampler(a.device)
    sampler.start()
    t = time.monotonic()
    agent = build_tracking_agent(
        model_path=str(a.assets / "checkpoint"),
        backend="hf",
        device=a.device,
        max_new_tokens=64,
        task_key="vlnce",
    )
    identity = {
        "config": config,
        "model": checkpoint["model"],
        "model_revision": MODEL_REVISION,
        "code_revision": CODE_REVISION,
        "portability_patch_sha256": source["patch_sha256"],
        "processor": type(agent.engine.bundle.processor).__name__,
        "load_s": time.monotonic() - t,
        "load_memory": sampler.finish(),
        "versions": {
            p: importlib.metadata.version(p)
            for p in (
                "torch",
                "torchvision",
                "transformers",
                "numpy",
                "pillow",
                "safetensors",
            )
        },
    }
    dump(a.output / "model.json", identity)
    for i, job in enumerate(data["jobs"]):
        result = run_job(agent, job, a.output / f"{i:03d}", a.device)
        print(
            json.dumps(
                {
                    "index": i,
                    "status": result["status"],
                    "native": result.get("native_output"),
                    "timing": result.get("timing"),
                    "error": result.get("error"),
                }
            ),
            flush=True,
        )
        if result["status"] == "error":
            raise RuntimeError(
                "Inference failed; remaining jobs not attempted (see saved error)"
            )


if __name__ == "__main__":
    main()
