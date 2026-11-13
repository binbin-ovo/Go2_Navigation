"""One pinned local SmolVLM2 model, used as a VLM navigation-proposal baseline."""

from datetime import datetime, timezone
import importlib.metadata
import platform
import random
import re
import time

import numpy as np

from .grounding import peak_rss_bytes
from .temporal_inputs import build_model_input
from .temporal_proposals import parse_proposal


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class SmolTemporalModel:
    def __init__(self, config, cache_dir, *, allow_download=False):
        import psutil
        import torch
        from huggingface_hub import snapshot_download
        from transformers import SmolVLMForConditionalGeneration, SmolVLMProcessor

        self.config = dict(config)
        if (
            config["device"] != "cpu"
            or config["dtype"] != "float32"
            or config["attention_implementation"] != "eager"
        ):
            raise ValueError("This experiment validates CPU FP32 eager only")
        if config["do_image_splitting"] or config["image_longest_edge"] != 384:
            raise ValueError(
                "Only the declared unsplit 384px input geometry is supported"
            )
        for k in ("revision", "processor_revision"):
            if not re.fullmatch("[a-f0-9]{40}", config[k]):
                raise ValueError("Pin immutable checkpoint and processor revisions")
        if (
            config["model_id"] != config["processor_id"]
            or config["revision"] != config["processor_revision"]
        ):
            raise ValueError("Use matching checkpoint and processor")
        available = psutil.virtual_memory().available
        if available < config["minimum_available_memory_gib"] * 2**30:
            raise MemoryError(
                f"Need {config['minimum_available_memory_gib']} GiB available; measured {available / 2**30:.2f}"
            )
        torch.set_num_threads(config["cpu_threads"])
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        started = time.perf_counter()
        snapshot = snapshot_download(
            config["model_id"],
            revision=config["revision"],
            cache_dir=str(cache_dir),
            local_files_only=not allow_download,
            allow_patterns=["*.json", "*.txt", "*.safetensors", "*.jinja"],
            max_workers=2,
        )
        # Explicit official class avoids AutoProcessor silently falling back to only a tokenizer.
        self.processor = SmolVLMProcessor.from_pretrained(
            snapshot, local_files_only=True
        )
        # Set processor/image-processor geometry explicitly, not tokenizer init kwargs.
        # Transformers 4.51.3 duplicates `size` if it leaks through from_pretrained.
        self.processor.do_image_splitting = False
        self.processor.image_processor.do_image_splitting = False
        self.processor.image_size = {"longest_edge": 384}
        self.processor.image_processor.size = {"longest_edge": 384}
        if self.processor.do_image_splitting or self.processor.image_size != {
            "longest_edge": 384
        }:
            raise ValueError("Processor override not applied")
        self.model = SmolVLMForConditionalGeneration.from_pretrained(
            snapshot,
            local_files_only=True,
            use_safetensors=True,
            torch_dtype=torch.float32,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        ).eval()
        self.info = {
            "model_id": config["model_id"],
            "revision": config["revision"],
            "processor_id": config["processor_id"],
            "processor_revision": config["processor_revision"],
            "model_class": type(self.model).__name__,
            "processor_class": type(self.processor).__name__,
            "capability_label": "VLM navigation-proposal baseline, not a reproduced VLN policy",
            "device": "cpu",
            "dtype": "float32",
            "attention_implementation": "eager",
            "seed": config["seed"],
            "cpu_threads": torch.get_num_threads(),
            "parameter_count": sum(p.numel() for p in self.model.parameters()),
            "load_s": time.perf_counter() - started,
            "available_memory_before_load_bytes": available,
            "rss_after_load_bytes": psutil.Process().memory_info().rss,
            "versions": {
                n: importlib.metadata.version(n)
                for n in [
                    "torch",
                    "transformers",
                    "huggingface-hub",
                    "tokenizers",
                    "safetensors",
                    "accelerate",
                    "numpy",
                    "Pillow",
                    "opencv-python",
                    "num2words",
                ]
            },
            "platform": platform.platform(),
            "python": platform.python_version(),
        }

    def infer(
        self,
        episode,
        decision_id,
        prompt_path,
        *,
        mode="temporal",
        allow_synthetic=False,
    ):
        import psutil
        import torch

        if mode not in {"temporal", "current_only"}:
            raise ValueError("Unsupported input mode")
        start_wall = time.time()
        start_utc = utc_now()
        start = time.perf_counter()
        prepared = build_model_input(
            episode,
            decision_id,
            self.processor,
            prompt_path,
            max_frames=self.config["max_frames"] if mode == "temporal" else 1,
            horizon_s=self.config["history_horizon_s"],
            allow_synthetic=allow_synthetic,
        )
        preprocess_s = time.perf_counter() - start
        vision_calls = []

        def hook(module, args, kwargs):
            pixels = kwargs.get("pixel_values", args[0] if args else None)
            if pixels is not None:
                vision_calls.append(list(pixels.shape))

        handle = self.model.model.vision_model.register_forward_pre_hook(
            hook, with_kwargs=True
        )
        gen_start = time.perf_counter()
        gen_start_utc = utc_now()
        try:
            with torch.inference_mode():
                output_ids = self.model.generate(
                    **prepared.inputs,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=self.config["max_new_tokens"],
                    use_cache=True,
                )
        finally:
            handle.remove()
        end_wall = time.time()
        gen_s = time.perf_counter() - gen_start
        end_utc = utc_now()
        generated = output_ids[0, prepared.inputs.input_ids.shape[-1] :].cpu()
        native = self.processor.tokenizer.decode(generated, skip_special_tokens=True)
        native_tokens = self.processor.tokenizer.decode(
            generated, skip_special_tokens=False
        )
        proposal = parse_proposal(native)
        consumed = sum(s[0] for s in vision_calls)
        if consumed != len(prepared.views):
            raise RuntimeError(
                f"Vision encoder consumed {consumed} images, expected {len(prepared.views)}"
            )
        latest = prepared.selection["frames"][-1]
        elapsed = time.perf_counter() - start
        proposal.update(
            {
                "schema_version": "go2-temporal-proposal-v1",
                "episode_id": episode.data["episode_id"],
                "decision_id": decision_id,
                "instruction": episode.data["instruction"],
                "mode": mode,
                "native_model_output": native,
                "native_output_with_special_tokens": native_tokens,
                "model": self.info,
                "model_executed": True,
                "selected_observations": prepared.selection,
                "vision_encoder_calls": vision_calls,
                "history_images_consumed": max(0, consumed - 1),
                "generated_tokens": len(generated),
                "hit_generation_limit": len(generated) >= self.config["max_new_tokens"],
                "timing": {
                    "inference_start_utc": start_utc,
                    "generation_start_utc": gen_start_utc,
                    "inference_end_utc": end_utc,
                    "start_unix_s": start_wall,
                    "end_unix_s": end_wall,
                    "preprocessing_s": preprocess_s,
                    "generation_s": gen_s,
                    "inference_pipeline_s": elapsed,
                    "replay_observation_age_at_output_s": prepared.selection[
                        "current_observation_age_s"
                    ]
                    + preprocess_s
                    + gen_s,
                    "historical_host_age_at_output_s": end_wall
                    - latest["host_receipt_s"],
                },
                "memory": {
                    "rss_after_inference_bytes": psutil.Process().memory_info().rss,
                    "process_lifetime_peak_rss_bytes": peak_rss_bytes(),
                },
            }
        )
        if proposal["hit_generation_limit"]:
            proposal["warnings"].append(
                "Generation hit token limit; output may be truncated"
            )
        proposal["warnings"].append(
            "Recorded future follows the operator, not this proposal; counterfactual outcome unobserved"
        )
        raw = {k: v.cpu().numpy() for k, v in prepared.inputs.items()}
        raw["generated_token_ids"] = generated.numpy()
        return proposal, prepared, raw
