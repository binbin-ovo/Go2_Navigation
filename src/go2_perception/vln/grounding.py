"""Grounding DINO adapter: offline image regions only, never executable robot goals."""

from dataclasses import dataclass
import importlib.metadata
import platform
import re
import resource
import time

import numpy as np

from .model_preprocessing import CalibratedView, processor_geometry


def format_instruction(instruction: str, max_characters=240) -> dict:
    """A declared lexical wrapper, not learned navigation instruction following.

    Accept noun phrases or Find/Locate/Show me + noun phrase. Relational questions
    and navigation commands remain unresolved; no guessed destination substitution.
    """
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("A nonempty instruction is required")
    text = " ".join(instruction.lower().split()).strip(" .!?")
    if len(instruction) > max_characters:
        return {"supported": False, "query": None, "reason": "instruction exceeds configured limit"}
    text = re.sub(r"^(find|locate|show me)\s+", "", text)
    unsupported = re.search(
        r"\b(go|move|turn|walk|navigate|which|where|why|beside|behind|between|nearest|"
        r"closest|left|right|toward|towards|then|previous|remember|and|or)\b", text)
    if unsupported or not re.fullmatch(r"[a-z][a-z0-9 '’-]*", text):
        return {"supported": False, "query": None,
                "reason": "Only simple target phrases are supported; navigation/relations need another model"}
    return {"supported": True, "query": text + ".",
            "method": "fixed Find/Locate/Show me wrapper removal; lowercase; terminal period"}


def parse_candidates(boxes, scores, labels, size_wh) -> dict:
    """Validate official postprocessor output (xyxy edge pixels in rectified view).

    Invalid entries are rejected and counted; slight out-of-image boxes are clipped
    explicitly. Multiple candidates stay unresolved rather than choosing by score.
    """
    width, height = size_wh
    boxes, scores = np.asarray(boxes, float), np.asarray(scores, float)
    if boxes.size == 0:
        boxes = boxes.reshape(0, 4)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or scores.shape != (len(boxes),) or len(labels) != len(boxes):
        raise ValueError("Malformed detection arrays: expected Nx4 boxes, N scores and N labels")
    candidates, rejected, clipped = [], 0, 0
    for box, score, label in zip(boxes, scores, labels):
        if (not np.isfinite(box).all() or not np.isfinite(score) or not 0 <= score <= 1
                or not isinstance(label, str) or not label.strip()
                or box[2] <= box[0] or box[3] <= box[1]):
            rejected += 1
            continue
        b = np.clip(box, [0, 0, 0, 0], [width, height, width, height])
        if b[2] <= b[0] or b[3] <= b[1]:
            rejected += 1
            continue
        clipped += int(not np.array_equal(b, box))
        candidates.append({"label": label, "score": float(score), "box_xyxy": b.tolist(),
                           "coordinate_frame": "rectified_image", "units": "pixels",
                           "point_base_m": None})
    warnings = []
    if rejected:
        warnings.append(f"rejected {rejected} malformed model candidates")
    if clipped:
        warnings.append(f"clipped {clipped} model boxes to image edges")
    status = "candidate" if len(candidates) == 1 else "unresolved_multiple" if candidates else "unresolved_no_candidate"
    if not candidates and rejected:
        status = "invalid_output"
    if len(candidates) > 1:
        warnings.append("multiple candidates (possibly duplicates); target identity unresolved")
    return {"status": status, "candidates": candidates, "rejected_candidates": rejected,
            "warnings": warnings, "proposal_only": True, "executable": False,
            "metric_goal": None, "action": None,
            "box_convention": "xyxy image edges; range [0,width] x [0,height]; not raw fisheye pixels",
            "semantic_correctness": "requires independent review; scores are not calibrated probabilities"}


def peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if platform.system() == "Darwin" else value * 1024)


@dataclass
class GroundingResult:
    proposal: dict
    raw: dict
    processor_metadata: dict
    model_input_rgb: np.ndarray
    timing: dict


class GroundingDino:
    """Pinned official Transformers implementation. Local cache only by default.

    CPU FP32 is the tested runtime. Other backends require a separate experiment.
    Imports are lazy so the core sensor interface does not require Transformers.
    """

    def __init__(self, config, cache_dir, *, allow_download=False):
        import psutil
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.config = dict(config)
        if self.config["device"] != "cpu":
            raise ValueError("Phase 6A baseline validates CPU only; do not silently change backend")
        for key in ("revision", "processor_revision"):
            if not re.fullmatch(r"[0-9a-f]{40}", self.config[key]):
                raise ValueError(f"{key} must be an immutable commit SHA")
        for key in ("box_threshold", "text_threshold"):
            if not 0 < float(self.config[key]) < 1:
                raise ValueError(f"{key} must be in (0,1)")
        torch.set_num_threads(int(config["cpu_threads"]))
        torch.manual_seed(0)
        before = psutil.Process().memory_info().rss
        start = time.perf_counter()
        opts = dict(cache_dir=str(cache_dir), local_files_only=not allow_download, trust_remote_code=False)
        self.processor = AutoProcessor.from_pretrained(config["processor_id"],
                                                       revision=config["processor_revision"], **opts)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            config["model_id"], revision=config["revision"], use_safetensors=True, **opts).cpu().eval()
        ip = self.processor.image_processor
        # Explicit assumptions used to carry K through the official processor.
        if getattr(ip, "do_center_crop", False) or not ip.do_resize or not ip.do_pad:
            raise ValueError("Unsupported image processor geometry")
        if not ip.do_rescale or not ip.do_normalize or ip.rescale_factor != 1 / 255:
            raise ValueError("Unexpected pinned processor normalization")
        self.info = {"model_id": config["model_id"], "revision": config["revision"],
                     "processor_id": config["processor_id"], "processor_revision": config["processor_revision"],
                     "processor_class": type(self.processor).__name__,
                     "image_processor": ip.to_dict(), "model_class": type(self.model).__name__,
                     "parameter_count": sum(p.numel() for p in self.model.parameters()),
                     "device": "cpu", "dtype": str(next(self.model.parameters()).dtype),
                     "cpu_threads": torch.get_num_threads(), "load_s": time.perf_counter() - start,
                     "rss_before_load_bytes": before, "rss_after_load_bytes": psutil.Process().memory_info().rss,
                     "versions": {p: importlib.metadata.version(p) for p in
                                  ["torch", "transformers", "huggingface-hub", "tokenizers", "safetensors",
                                   "numpy", "opencv-python", "Pillow", "psutil"]},
                     "platform": platform.platform(), "python": platform.python_version()}

    def predict(self, view: CalibratedView, instruction: str) -> GroundingResult:
        import psutil
        import torch
        from PIL import Image

        formatted = format_instruction(instruction, self.config["max_instruction_characters"])
        start = time.perf_counter()
        if not formatted["supported"]:
            proposal = parse_candidates([], [], [], (view.intrinsics.width, view.intrinsics.height))
            proposal.update(status="unresolved_unsupported_instruction", instruction=instruction,
                            instruction_format=formatted, model_executed=False)
            proposal["warnings"].append(formatted["reason"])
            return GroundingResult(proposal, {}, {}, view.image.copy(), {"forward_s": None})
        # Uint8 PIL input: official rescaling and normalization are applied exactly once.
        inputs = self.processor(images=Image.fromarray(view.image), text=formatted["query"], return_tensors="pt")
        if int(inputs["input_ids"].shape[-1]) > self.model.config.max_text_len:
            raise ValueError("Instruction exceeds model token limit")
        geometry = processor_geometry(view, inputs["pixel_mask"][0].numpy())
        preprocess_s = time.perf_counter() - start
        forward_start = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model(**inputs)
        forward_s = time.perf_counter() - forward_start
        post_start = time.perf_counter()
        post = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=self.config["box_threshold"],
            text_threshold=self.config["text_threshold"],
            target_sizes=[(view.intrinsics.height, view.intrinsics.width)])[0]
        proposal = parse_candidates(post["boxes"].numpy(), post["scores"].numpy(),
                                    post["text_labels"], (view.intrinsics.width, view.intrinsics.height))
        proposal.update(instruction=instruction, instruction_format=formatted, model_executed=True)
        raw = {"logits": outputs.logits.cpu().numpy(), "pred_boxes": outputs.pred_boxes.cpu().numpy(),
               "input_ids": inputs.input_ids.numpy(), "attention_mask": inputs.attention_mask.numpy(),
               "pixel_mask": inputs.pixel_mask.numpy()}
        # This is a preview reconstructed from the exact normalized tensor, including padding.
        ip = self.processor.image_processor
        pixels = inputs.pixel_values[0].permute(1, 2, 0).numpy()
        pixels = pixels * np.asarray(ip.image_std) + np.asarray(ip.image_mean)
        preview = np.clip(np.rint(pixels / ip.rescale_factor), 0, 255).astype(np.uint8)
        meta = {"geometry": geometry, "image_processor": ip.to_dict(),
                "tensor_shape_bchw": list(inputs.pixel_values.shape),
                "input_rgb_dtype": "uint8", "normalization_owner": "official AutoProcessor only",
                "output_boxes_target_size_wh": [view.intrinsics.width, view.intrinsics.height]}
        timing = {"processor_s": preprocess_s, "forward_s": forward_s,
                  "postprocess_and_parse_s": time.perf_counter() - post_start,
                  "model_pipeline_s": time.perf_counter() - start,
                  "rss_after_forward_bytes": psutil.Process().memory_info().rss,
                  "process_lifetime_peak_rss_bytes": peak_rss_bytes()}
        return GroundingResult(proposal, raw, meta, preview, timing)
