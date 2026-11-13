# Phase 6A model selection

Decision made before checkpoint download, 2026-11-12. Machine: Apple M3 Pro,
36 GiB unified memory, arm64 macOS. Use CPU FP32 for the reproducible first run;
CUDA is unavailable, and MPS acceleration is not validated by this experiment.

| Candidate | Required input / preprocessing | Actual output / frame | Availability / license | Resource estimate and decision |
|---|---|---|---|---|
| IDEA-Research/grounding-dino-tiny | One RGB image and lower-case, period-terminated target text. Official Transformers AutoProcessor handles aspect-preserving resize, padding, rescaling and normalization. No history, pose, panorama or depth required. | Text-conditioned boxes, token phrases and scores in image pixels; no metric distance or actions. | Public safetensors; Apache-2.0 card/repository. Revision `a2bb814dd30d776dcf7e30523b00659f4f141c71`. | 172,249,090 FP32 parameters (~0.69 GB weights), estimate 2–6 GiB process RAM with activations; feasible on this CPU. **Selected: grounding baseline, not a navigation policy.** |
| HuggingFaceTB/SmolVLM-256M-Instruct | Image/text sequence through official chat template and processor; tiled 512px patches, model-specific normalization. Single image sufficient; history optional. No mandatory depth, pose or panorama. | Generated text; no guaranteed region format, metric frame, or navigation action semantics. | Public checkpoint, Apache-2.0. | ~0.5 GB BF16 / ~1 GB FP32 weights estimate, plus activations/cache. Card reports under 1 GB GPU memory for one image (vendor measurement, not ours). Likely fits CPU memory; not installed or measured. DINO gives more directly inspectable spatial proposals. |
| a8cheng/navila-llama3-8b-8f | Current RGB plus sampled historical RGB and instruction, VILA/SigLIP image processor + conversation prompt. Released eight-frame checkpoint. High-level VLA does not require a panorama, depth or pose; the separate locomotion system requires additional observations. | Egocentric language motion increments (forward distance in cm, turn angle in degrees, stop). These are model proposals, not measured base_link waypoints. | Public checkpoint linked by official repo; repo Apache-2.0, checkpoint page has **no model card/license declaration**. Llama 3 upstream terms also need review. | ~16 GB FP16 weights alone, plus image activations/KV cache (estimate). Released evaluation setup targets Linux/CUDA/FlashAttention, patched Transformers 4.37.2 and Habitat. Not a drop-in Mac runtime; no weights downloaded, no deployment claim. |

Official sources (accessed 2026-11-12):
- Grounding DINO [paper](https://arxiv.org/abs/2303.05499),
  [authors' repository](https://github.com/IDEA-Research/GroundingDINO),
  [checkpoint and example](https://huggingface.co/IDEA-Research/grounding-dino-tiny),
  [Transformers 4.49 API](https://huggingface.co/docs/transformers/v4.49.0/en/model_doc/grounding-dino).
- SmolVLM [paper](https://arxiv.org/abs/2504.05299),
  [model card and official inference](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct),
  [authors' code](https://github.com/huggingface/smollm).
- NaVILA [paper](https://arxiv.org/abs/2412.04453),
  [official code and evaluation instructions](https://github.com/AnjieCheng/NaVILA),
  [checkpoint](https://huggingface.co/a8cheng/navila-llama3-8b-8f).

These estimates are selection budgets, not measured latency or memory. Actual local
measurements and semantic review belong in VLN_INPUT_VALIDATION.md after execution.
No private images are submitted to a hosted inference service.

Post-selection execution confirmed the selected model fits: 12 real-image forwards
completed, mean 1.959 s, process peak 3.003 GiB. These are local CPU measurements,
not estimates for the two unexecuted alternatives. Semantic performance was weak,
especially absent-target rejection; see [actual results](VLN_INPUT_VALIDATION.md).
