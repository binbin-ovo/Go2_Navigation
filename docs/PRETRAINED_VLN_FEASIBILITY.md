# Phase 6C: navigation-trained model feasibility

Audit date: 2026-11-13. Branch `phase6-pretrained-vln`, isolated worktree based on
`7e2989ad54b9c484d75c029ff0141834880852b6`. Phase 6B, Claude's packages/calibration,
Phase 5A.1, main and operational defaults are preserved. Physical execution is disabled.

## Actual accessible machine

Apple M3 Pro (18 GPU cores), macOS 26.6.2 arm64, 36 GiB unified RAM. Initial available
RAM 14.15 GiB and free disk 14.04 GiB. The existing Phase 6B environment is Python
3.12.11 / Torch 2.6.0; CUDA runtime is null and CUDA unavailable. No `nvidia-smi`,
`nvcc` or WSL executable exists. Metal is built but hidden inside the sandbox;
a read-only check outside it reports MPS available, recommended maximum allocation
30,150,672,384 bytes. That allocator limit is NOT free system memory.

No Isaac installation was found in the checked Mac paths (`/Applications/Isaac Sim.app`,
`~/isaacsim`, `~/IsaacLab`, `~/.local/share/ov/pkg`, `~/Omniverse/pkg`); this bounded
search is not proof about every disk location. Windows was not accessible. Its
driver, toolkit, actual free VRAM, PyTorch runtime and existing simulator remain
pending the [Windows preflight](WINDOWS_SIMULATION_FEASIBILITY.md).

## Exactly two candidates audited

| | StreamVLN real-world | **LightNav-0: selected** |
|---|---|---|
| Official code | [InternRobotics/StreamVLN](https://github.com/InternRobotics/StreamVLN), `e48f6ff7e9201d93aae003e8f64b04c00cec13bc` | [lightorigins/LightNav-0](https://github.com/lightorigins/LightNav-0), `3015508b70fb6e30bbd66701e94c9b916dcb8df8` |
| Navigation checkpoint | [mengwei0427/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_real_world](https://huggingface.co/mengwei0427/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_real_world), `7b0269ad28e7039a32627b950766c09b5a646f8e` | [LightOriginsHQ/LightNav-0](https://huggingface.co/LightOriginsHQ/LightNav-0), `826dc5fbfa37afa8293d2e336d329b6ffc0bfb64` |
| Count / weights | 8,030,345,248 stored BF16 tensor elements; ~14.96 GiB checkpoint | 4,847,825,408 stored BF16 tensor elements; **4,448,342,528 unique parameters inferred** after the declared embedding/head tie (~8.286 GiB BF16 weights). 9,695,744,034-byte safetensors (~9.03 GiB on disk). Includes vision; “4B” is backbone shorthand |
| Training / backbone | Navigation-trained LLaVA-Video / Qwen-family custom architecture, real-world adaptation | Navigation-aligned Qwen3-VL-4B-Instruct; extended pointing/RVQ vocabulary, not the generic Qwen checkpoint |
| License | Repository CC-BY-NC-SA-4.0. Real-world checkpoint README is empty and card supplies no separate terms; inherited model/component terms need resolution before redistribution | Code and checkpoint Apache-2.0; [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) also Apache-2.0. Keep upstream notices; do not redistribute weights here |
| Inference dependencies | README Python 3.9, Torch 2.1.2, advertised CUDA 12.4; requirements actually pin CUDA 12.1 runtime packages, Transformers 4.45.1; custom model, Habitat evaluation stack | Python 3.11 only, Torch >=2.9, torchvision >=0.24, Transformers exactly 5.8.0, tokenizers 0.22.x, NumPy >=2.0. HF or vLLM 0.19.1 (latter verified with Torch 2.10) |
| OS support evidence | Linux CUDA/Conda/Habitat recipe; no verified macOS/native Windows path | Documented CUDA/Python and Linux Docker stack. HF source has CPU fallback, but unconditional CUDA timing calls break it. macOS/native Windows are not upstream-validated |
| Attention / precision | Official server hardcodes FlashAttention-2 and BF16 | Official HF defaults SDPA and hardcodes BF16; optional FlashAttention, not required. No upstream HF dtype/4-bit/offload configuration |
| Quantization | bitsandbytes dependency does not establish a supported quantized inference path | Only documented code option `fp8_llm_only` in vLLM (vision stays BF16); full-model FP8 refused. Not a Turing/8 GB solution. No quantization used here |
| Auxiliary decoder | Text/arrow action parser; unknown outputs upstream fall back to STOP (must not count as valid stopping) | Required checkpoint `action_tokenizer/manifest.json`, 3×256 codebooks, Jacobian weights; RVQ `se2_diff` decoded by official code, no extra neural decoder |
| Native output | Action chunks: forward 0.25 m, left/right 15°, STOP | Dual pointing plus `<act_l0_…><act_l1_…><act_l2_…>`, ten **absolute ego SE(2) waypoints**, +x forward/+y left/+yaw CCW, metres/radians, no intrinsic time base |
| Native temporal policy | Sliding-window KV/conversation plus slow history; 32-frame reset/8-history defaults; generated action tokens retained in subsequent context | Full observation buffer at 4 Hz; native SlowFast current/fast/mid/long/anchor tiers, post-ViT spatial pooling, relative timestamps, checkpoint 256×448 H×W, max 8192 tokens |
| Replay feedback | Generated action/conversation context assumes the rollout history; no faithful operator-route equivalence established | Observation + instruction only. No previous proposal/executed-action inputs; each HF generate starts fresh. Reset all buffers/caches between episodes/epochs |
| Decision | Not downloaded/run: weights alone exceed current free RAM and 8 GB GPU, CUDA-only reference path; port would be larger | Selected for a bounded local experiment due to smaller weights, official SDPA, observation-only replay and explicit RVQ decoder |

Source evidence retained privately under `data/outputs/phase6c/audit/` (API revisions,
file sizes, model configs) and `upstream/` (pinned inference source). Only these two
navigation candidates were audited. No generic VLM substitution or paid compute.

## Resource budgets (estimates, not measurements)

Parameter inference uses pinned tensor shapes and `tie_word_embeddings=true` in the
model/text configs. The 156048×2560 embedding/head duplication accounts for 399,482,880
stored elements. This is metadata-based inference, not a separate live parameter
census; `audit/parameter_count_from_metadata.json` records it. Weight-file bytes are
not runtime memory. Measured process/MPS peaks are separate in the experiment report.

| Workload | Separate budget / feasibility gate |
|---|---|
| LightNav unquantized inference | ~8.286 GiB unique BF16 weights inferred from the declared tie (9.03 GiB checkpoint on disk) + up to 1.125 GiB BF16 KV at 8192 tokens (2×36 layers×8 KV heads×128×8192×2 bytes), plus vision/language activations, processor/buffers and allocator overhead. Plan 12–16 GiB; measure before large histories |
| StreamVLN inference | ~14.96 GiB weights plus vision/KV/activations, roughly 18–24+ GiB; not attempted |
| Windows RTX 2070 Super | User reports ~8 GB, actual inventory pending. Unquantized LightNav weights alone exceed capacity. Turing is not the standard FlashAttention-2 or native BF16 path; no compatible quantified memory/offload solution established |
| Isaac camera rendering | Version-dependent; current documented baseline is 16 GB VRAM/32 GB RAM/50 GB SSD, not an 8 GB guarantee. Existing Windows version must be identified first |
| Simultaneous model + simulator | No shared-GPU capacity inferred by adding theoretical figures. Requires independent successful tests and a measured combined run; see the experiment report for the separate Mac result. Windows remains pending |

The standard [FlashAttention-2 upstream documentation](https://github.com/Dao-AILab/flash-attention)
targets Ampere or newer; its separate Turing project is not automatically compatible
with these policies. No FlashAttention installation, driver change, simulator reinstall,
WSL enablement or hidden FP32 promotion is part of this work.

## Isolation and reproduction

Core episode/calibration/evaluation uses a fresh `core-env` constrained to the tested
Phase 6B dependencies; the existing Phase 6B environment is untouched.
LightNav gets a **separate** Python 3.11 environment; do not install its NumPy 2 /
Transformers 5.8 dependencies into the perception environment. A file-based local
worker boundary carries only instruction and causal rectified RGB, with provenance.
The model never receives annotations, future frames, odometry or perfect geometry.

```bash
cd /Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6c-worktree
ARTIFACTS=/Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6c
UV_PYTHON_INSTALL_DIR="$ARTIFACTS/python" uv venv --python 3.11.13 "$ARTIFACTS/model-env"
uv pip install --python "$ARTIFACTS/model-env/bin/python" -r requirements-pretrained-vln-macos-tested.txt
HF_HUB_DISABLE_XET=1 "$ARTIFACTS/model-env/bin/python" scripts/prepare_pretrained_vln.py \
  --root "$ARTIFACTS/lightnav" --weights
```

The preparer downloads only pinned inference source (no robot clients) and ONE model,
verifies Git blob hashes and LFS SHA256, includes the RVQ bundle, and requires 3 GiB
disk reserve after download. Existing changed source files are refused. Source-only
is the default without `--weights`. Actual runs, portability deviations, latency,
memory and failures belong in [PRETRAINED_VLN_EXPERIMENTS](PRETRAINED_VLN_EXPERIMENTS.md).

## Portable HF execution: explicitly a local deviation

Unmodified upstream CPU inference loaded the checkpoint but failed at an unconditional
CUDA timing barrier. `scripts/patch_lightnav_local_runtime.py` derives a separate,
hashed `source-local-v2` from verified source; pristine source and the first local fork
remain intact. The local changes expose CPU/MPS device selection, synchronize the
selected accelerator for timing, and run **only non-divisible MPS adaptive average
pooling** on CPU in the same BF16 dtype before returning its result to MPS. The latter
fix became necessary for the native long/anchor history tiers, not the initial short
smoke. Native CPU versus MPS-fallback embeddings, grids and deepstack features are
bit-identical in the opt-in test. No history, prompt, weights, decoder or image geometry
was changed to avoid that unsupported kernel. Attention remains official SDPA.

The model's native 4 Hz stream is reconstructed causally using the last available
exposure as the grid anchor. Each grid sample selects a real exposure at/before it;
source exposure/availability and quantization error are retained. Gaps, missing frames
and repeated-source padding are refused. Native SlowFast may duplicate an odd tubelet
frame; its actual segment indices/pooling are saved separately. Raw fisheye images,
measured rectification and official 448×256 bilinear stretch each retain their geometry.
`T_base_camera` stays null: native ego waypoints have **not** been registered to Go2.

The official policy stores RGB history and a vision cache, not previously executed
actions. Replay resets before each decision and reconstructs only the actual causal
observations. It never treats a new proposal as the cause of an operator-recorded frame.
Simulation has its own explicit applied-action record. STOP from the native decoder
(coarse RVQ stop code or near-zero trajectory) is not proof of task completion.
