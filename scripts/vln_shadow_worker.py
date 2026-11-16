"""Internal pipe-only LightNav process. Uses the existing pinned inference worker."""

import argparse
from contextlib import redirect_stdout
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--assets", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--device", choices=["cpu", "mps", "cuda"], required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--profile-stages", action="store_true")
    ap.add_argument("--preparation-mode", choices=["baseline", "cached"], default="baseline")
    ap.add_argument("--tensor-evidence", choices=["record", "skip"], default="record",
                    help="skip (cached mode only): do not archive model-input tensors")
    args = ap.parse_args()
    wire = sys.stdout

    def send(data):
        wire.write(json.dumps(data, allow_nan=False) + "\n")
        wire.flush()

    os.environ.update(
        HF_HUB_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        TOKENIZERS_PARALLELISM="false",
        LIGHTNAV_ATTN="sdpa",
        VLN_EVAL_TEMPERATURE="0",
        VLN_EVAL_TRAJ_TOP1="0",
        VLN_TIMESTAMP_RELATIVE="1",
    )
    with redirect_stdout(sys.stderr):
        import torch
        import yaml
        from lightnav_worker import (
            MemorySampler,
            dump,
            run_job,
            validate_jobs,
            verify_assets,
        )
        from go2_perception.vln.pretrained import (
            validate_experiment_config,
            validate_runtime,
        )

        config = yaml.safe_load(args.config.read_text())
        validate_experiment_config(config)
        runtime = {**config, "device": args.device}
        hardware = {
            "mps_available": torch.backends.mps.is_available(),
            "cuda_available": torch.cuda.is_available(),
        }
        if args.device == "cuda" and hardware["cuda_available"]:
            hardware.update(
                compute_capability=torch.cuda.get_device_capability(),
                free_vram_bytes=torch.cuda.mem_get_info()[0],
            )
        validate_runtime(runtime, hardware)
        source, checkpoint = verify_assets(args.assets)
        sys.path.insert(0, str(args.assets / "source-local-v2/src"))
        from lightnav.tracking import build_tracking_agent

        torch.set_num_threads(4)
        torch.manual_seed(0)
        sampler = MemorySampler(args.device)
        sampler.start()
        started = time.monotonic()
        agent = build_tracking_agent(
            model_path=str(args.assets / "checkpoint"),
            backend="hf",
            device=args.device,
            max_new_tokens=64,
            task_key="vlnce",
        )
        dump(
            args.output / "model.json",
            {
                "runtime": runtime,
                "checkpoint": checkpoint["revision"],
                "upstream_code": source["upstream_revision"],
                "patch_sha256": source["patch_sha256"],
                "processor": type(agent.engine.bundle.processor).__name__,
                "load_s": time.monotonic() - started,
                "load_memory": sampler.finish(),
                "versions": {
                    p: importlib.metadata.version(p)
                    for p in ("torch", "transformers", "numpy")
                },
                "reset_policy": "reconstruct_complete_causal_instruction_history_per_request",
                "action_feedback": False,
                "physical_execution_authorized": False,
                "preparation_mode": args.preparation_mode,
                "stage_profiling": args.profile_stages,
                "tensor_evidence": args.tensor_evidence,
            },
        )
        send({"type": "ready"})
        from go2_perception.vln.shadow_evidence import SkippedTensorEvidence, TensorEvidenceWriter
        if args.tensor_evidence == "skip" and args.preparation_mode != "cached":
            raise ValueError("Tensor evidence can only be skipped in cached preparation mode")
        writer = None
        if args.preparation_mode == "cached":
            writer = TensorEvidenceWriter() if args.tensor_evidence == "record" else SkippedTensorEvidence()
        try:
            serve(agent, args, send, writer, run_job, validate_jobs)
        finally:
            if writer is not None:
                try:
                    writer.close()
                finally:
                    dump(args.output / "tensor_evidence_summary.json", writer.status())


def serve(agent, args, send, writer, run_job, validate_jobs):
    for line in sys.stdin:
        if writer is not None:
            writer.check()
        message = json.loads(line)
        if set(message) != {"identity", "job", "destination"}:
            raise ValueError("Unsupported advisory worker envelope")
        job = message["job"]
        validate_jobs({"schema": "lightnav-local-jobs-v1", "jobs": [job]})
        start = time.monotonic()
        # Existing Phase 6C reset/reconstruct path: complete causal stream, native
        # SlowFast (including anchors), no KV/action feedback from other requests.
        result = run_job(agent, job, Path(message["destination"]), args.device,
                         profile_stages=args.profile_stages, evidence_writer=writer)
        end = time.monotonic()
        send(
            {
                "identity": message["identity"],
                "result": result,
                "host_inference_start_s": start,
                "host_inference_end_s": end,
                "inference_duration_s": end - start,
                "backend": "lightnav",
            }
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"type": "error", "error": repr(error)}), flush=True)
        raise
