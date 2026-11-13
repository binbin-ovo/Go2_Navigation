# Phase 6B temporal VLN experiment — 2026-11-12

**Result:** a real pretrained model consumed calibrated Go2 visual history and
produced inspectable offline proposals. It did **not** establish reliable instruction
following or a benefit from history. All eight temporal decisions proposed TURN_RIGHT,
including an explicit left-turn instruction. This is a **VLM navigation-proposal
baseline**, not a reproduced VLN policy. No robot connection, command, fine-tuning,
main merge or Phase 5A.1/calibration change was performed.

## Reproducible scope

Branch `phase6-temporal-vln`, verified base
`c13f1dc442fa157b31043613d3cb2b0c30ceab8a`. The episode interface and synthetic fixture
were committed and pushed first at `5c7f586aa2536a8eda9167e66d5050a4a91dbac2`, followed
by the exact-SHA handoff. No Claude-prepared private episode package was available
locally during this run. The declared fallback uses existing recordings; all human
annotations remain pending. They are separate from manifests and model inputs.

The only temporal checkpoint executed was `HuggingFaceTB/SmolVLM2-2.2B-Instruct` at
`be92141c8d0ceb98596a923c308ac2bbb4a7c466`, with the processor at the same revision.
Official model/processor, 2,246,784,880 parameters, ~8.99 GB FP32 weights, Apache-2.0.
See [candidate audit and official sources](TEMPORAL_VLN_ARCHITECTURE.md).
NaVILA and SmolVLM2-500M were audited but not downloaded or run. The separate
Grounding DINO regression below preserves Phase 6A, not a second temporal candidate.

Machine: Apple M3 Pro, 36 GiB RAM, macOS 26.6.2 arm64, Python 3.12.11. Actual device:
**CPU FP32, eager attention, four threads**; CUDA and MPS unavailable in this process.
Torch 2.6.0, Transformers 4.51.3, HF Hub 0.30.2, tokenizers 0.21.4, safetensors 0.5.3,
accelerate 1.6.0, NumPy 1.26.4, OpenCV 4.11.0.86, Pillow 11.1.0, num2words 0.5.14.
Full installed constraints: [requirements-temporal-vln-tested.txt](../requirements-temporal-vln-tested.txt).
Greedy decoding, seed 0, one beam, max 160 new tokens; no sampling or training.
All inference was local. Private images were not sent to a hosted model.

## Predeclared pilot

`configs/temporal_vln_pilot.yaml` fixes the following eight cutoffs before evaluation.
Offsets are seconds from the source recording's first estimated RGB exposure.
Each cutoff is run with both supported inputs: four chronological images over six
seconds including current, and the same current image alone. The package samples
raw recording frames on a 0.5 s exposure grid; selection subsequently enforces
availability at the decision cutoff. There are 120 packaged frames, four episodes,
three recordings, eight unique decisions, and **16 inference calls**.

| Episode | Recording | Episode offsets (s) | Decision offsets (s) | Instruction |
|---|---|---|---|---|
| visible_refrigerator | p5d_static_02 | 2–15 | 8, 12 | Approach the refrigerator and stop before reaching it. |
| turn_to_racket | test_c | 10–27 | 18, 24 | Turn left to face the tennis racket, then stop. |
| similar_cabinets | test_c | 35–52 | 43, 49 | Approach the white cabinet and stop in front of it. |
| absent_red_door | test_b | 2–15 | 8, 12 | Go through the red doorway, turn into the hallway, and stop. |

These are highly correlated home-interior views, mostly stationary/yaw motion.
Case names/categories are agent-selected sampling intentions, not human-certified
visibility/ambiguity labels. The operator did **not** record executions of these new
instructions. This is a pilot, not a generalization benchmark.

Source frames, calibration, manifest and selected image hashes are saved privately.
Exposure is receipt minus per-recording host/robot offset minus 48 ms, with about
16 ms estimated timing uncertainty. Availability uses last RTP receipt minus the
offset plus an explicitly **assumed** 100 ms decode margin. This is a receipt-based
proxy, not a measured decode-completion bound or hardware synchronization. No source
gaps above 0.25 s were found in these four intervals; missing/gap paths are unit-tested.
Latest selected images were 0.510–0.569 s old at the cutoff.

Every image uses the unchanged Phase 6A measured fisheye-to-pinhole preprocessing.
The official SmolVLM processor is explicitly unsplit at 384 px: aspect-preserving
1280×720 → 384×216, then its unsplit 384×384 resize. Appearance is stretched in that
second step; both transforms and K are retained. Mean/std 0.5 normalization occurs
once. This custom bounded multi-image protocol is not the model's default 1536px
image tiling or a reproduced navigation-specific video policy.

Actual forward hooks recorded **[4, 3, 384, 384]** for every temporal vision-encoder
call, with 324 image tokens (81 per image); single-image calls recorded
**[1, 3, 384, 384]** with 81 image tokens. The four images are three past observations
plus current. Full tensors, tokens and frame identities are retained. History really
entered the model; this does not prove useful temporal reasoning.

## Freeze and development changes

Before any private pilot inference, these SHA256 values were frozen in
`audit/pilot_frozen_files.json` and checked unchanged after evaluation:

| File | SHA256 |
|---|---|
| configs/temporal_vln_prompt.txt | `93f6b36d692dcc72ccd46ac8832f70e989fc257705d8fe05800fb45f246dc00d` |
| configs/temporal_vln.yaml | `95eaa88651def2875082068094f0423b82e3c09bf9683504d7b579e0c9fadcb1` |
| configs/temporal_vln_pilot.yaml | `c6b02f0cf236f799f3ec1d936ec72b08da9283ba7b05bbeb3c42f2f5c0ae16bd` |
| vln/temporal_proposals.py | `a5a0ad418fa65986f06bd3432531b3578ac3e309370ca548214016339fc64874` |
| vln/temporal_inputs.py | `803061fe74577c6d151fb60d85b07c8b865828c4ae1f96269c4245c1d5f1a3e6` |
| vln/temporal_model.py | `92947de2e5cb3532c5e790e3a5692ed45036ea6bb9988a627c5c8ba4d2a3d1b6` |

Three development runs precede the pilot; they are **not** accuracy measurements:

1. `synthetic_smoke/`: failed before forward. Passing `size` through processor
   `from_pretrained` duplicated the image-processor argument in Transformers 4.51.3.
   Fixed with explicit official SmolVLMProcessor, then documented geometry settings.
   An earlier AutoProcessor lookup returned only a tokenizer; no mock fallback used.
2. `synthetic_smoke_v2/`: real four-image forward with prompt v1; model copied entire
   enum lists, rejected by parser. One prompt revision removed the inline JSON enum
   example and requested one label. Original prompt is preserved in the private audit.
3. `synthetic_smoke_v3/`: real four-image forward with prompt v2; native `Proposal:` /
   `Reason:` fields were rejected by the then JSON-only parser. Before pilot freeze,
   parsing added that exact anchored field format, keeping absent task status unknown.
   The synthetic response also invented a right-turn instruction; this is a recorded
   semantic failure, not evidence of model accuracy.

Private real results did not trigger any prompt, parser, model or input changes.
After evaluation, reporting alone clarified that completion counts concern structured
status, and added a regression test so missing model status cannot count as correct
incompletion. Original pilot summaries are preserved; `audit/pilot_analysis.json`
contains the reporting-only update. No model responses were altered or regenerated.

The fresh editable environment initially had a hidden `.pth` flag that Python 3.12
skipped. It was corrected only in the new environment; `PYTHONPATH=src` is also a
reproduction workaround. `uv pip check` passed for all 88 installed packages after
using a writable local cache. No shared calibration environment was modified.

## Observed results

| Measurement | Four-image temporal | Current image only |
|---|---:|---:|
| Inputs accepted / calls executed | 8/8 | 8/8 |
| Strict JSON outputs accepted | 8/8 | 8/8 |
| Partial / rejected / token-truncated | 0 / 0 / 0 | 0 / 0 / 0 |
| TURN_RIGHT / TURN_LEFT / UNRESOLVED | 8 / 0 / 0 | 4 / 2 / 2 |
| STOP / ADVANCE | 0 / 0 | 0 / 0 |
| Structured task status IN_PROGRESS | 8 | 8 |
| Structured task status COMPLETE | 0 | 0 |
| Repeated proposal across each episode's two cutoffs | 4/4 | 4/4 |
| Opposite consecutive turn pairs within episode/mode | 0 | 0 |
| Human-reviewed action comparisons | 0 (pending) | 0 (pending) |

Paired modes differ at 4/8 cutoffs. Temporal history did not yield a demonstrated
quality gain; it produced a single action category across every case. Repeated
proposals and direction differences are review flags, not automatic unsafe-action
scores. The recorded view follows the operator independently of the model.

**Human evaluation remains pending:** action agreement, instruction-progress
consistency, verified false completion and acceptable behavior for absent/ambiguous
targets cannot be scored against human ground truth. Zero structured COMPLETE fields
is **not** zero false completion: the native explanation for `absent_red_door/d00`
claims the instruction is completed while its field says IN_PROGRESS. The automatic
metric only counts structured status; prose must be inspected separately.

Codex inspected six representative d00 panels and all 16 native responses. This is
explicitly **agent review**, not human verification or a semantic accuracy score:

| Example (relative to pilot_v1/) | Agent review |
|---|---|
| turn_to_racket/d00/current_only/review.png | Limited success: TURN_LEFT matches the literal instruction and racket visible on the left. Reason echoes instruction, so learned visual reasoning is not established. |
| turn_to_racket/d00/temporal/review.png | Failure: TURN_RIGHT contradicts the explicit left instruction and racket location. |
| visible_refrigerator/d00/current_only/review.png | Unresolved despite a visible refrigerator; the model calls the destination unclear. Visibility does not establish a safe approach. |
| visible_refrigerator/d00/temporal/review.png | TURN_RIGHT plus copied instruction does not demonstrate grounded progress reasoning. |
| similar_cabinets/d00/temporal/review.png | Several white storage/cabinet surfaces are visible; response does not disambiguate a referent and repeats the instruction. Human acceptable-action set remains pending. |
| absent_red_door/d00/temporal/review.png | No red doorway apparent in supplied history; native text claims completion without support and contradicts its status field. |

All responses, including these failures, appear in the private video/panels.
`audit/agent_review.json` records reviewer type and scope. No agent labels were inserted
into human annotation files or fed back into the model.

## Timing and memory

Pilot wall time: 2026-11-13 03:02:54.993–03:05:13.793 UTC (November 12, New York).
One loaded model, fixed temporal-then-single ordering at each cutoff, no randomized
runtime experiment. Times are wall-clock local CPU observations, not a real-time claim.

| Seconds | Temporal mean / median / range | Single mean / median / range |
|---|---|---|
| Preprocessing | 0.142 / 0.134 / 0.126–0.189 | 0.036 / 0.034 / 0.033–0.043 |
| Generation | 10.171 / 9.586 / 9.303–12.000 | 6.221 / 6.208 / 5.838–6.666 |
| Inference pipeline | 10.313 / 9.736 / 9.463–12.143 | 6.257 / 6.246 / 5.871–6.700 |
| Replay observation age at model output | 10.848 / 10.261 / 9.973–12.653 | 6.791 / 6.772 / 6.440–7.212 |

Replay age = decision cutoff − latest estimated exposure + preprocessing + generation.
It excludes review/video/disk-write time and scheduling between decisions. Historical
host age is separately recorded against the old recording receipt; never subtract a
robot timestamp from today's host clock. Output age is not measured live latency.

Process lifetime peak RSS: **10,397,810,688 bytes = 9.684 GiB**, including model,
activations, processor and review work; this is not GPU VRAM. Both mode summaries
inherit a shared process peak, not independent per-mode memory benchmarks. Available
RAM before load was 12.583 GiB. Cached/mmap model construction took 0.458 s and initially
reported 0.480 GiB RSS; lazily resident weights make that a misleading total memory
estimate, so use the observed inference peak instead. Generation lengths were 42–59
new tokens, below the 160 cap.

## Automated and compatibility validation

Final full command (inside the isolated worktree):

```bash
ARTIFACTS=/Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6b
PYTHONPATH=src SMOLVLM_MODEL_CACHE="$ARTIFACTS/hf_cache" HF_HUB_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false MPLCONFIGDIR="$ARTIFACTS/mpl" \
  .venv/bin/python -m pytest -q -rs
```

**470 passed, 1 skipped in 36.12 s.** Only skipped test:
`tests/test_go2_hardware.py:18` requires `GO2_HARDWARE=1`. No hardware test was enabled.
This run includes both actual-model integration tests: exact pinned processor and
four-image vision forward, bounded to 16 generation tokens for the test. It makes
no synthetic semantic-accuracy assertion. Before the final reporting regression,
normal offline suite: 467 passed, 3 skipped in 22.10 s (hardware plus two opt-in model
tests); separate model tests had also passed, 2 in 12.93 s. Final normal suite has
one additional reporting test; the final all-model run above is authoritative.

Ruff 0.11.13 check and format-check passed on all 12 new/modified Python files.
Tests cover strict manifests, file hashes, cross-epoch/recording rejection, both
exposure/availability cutoffs, missing frames/gaps, calibration per image, normalization,
annotation exclusion, native parsing/rejection, STOP vs completion, unknown frames,
serialization and panel/video decode. Synthetic fixture generation remains byte-for-byte
reproducible and the published schema/fixture is unchanged.

Existing Grounding DINO Tiny at `a2bb814dd30d776dcf7e30523b00659f4f141c71` was run with
Transformers 4.51.3 on the saved Phase 6A `s_fridge` observation. It returned the exact
same label, score 0.9389241933822632 and box as the Phase 6A output. This is a single
compatibility smoke, not a repeat of the twelve-case grounding evaluation.

```bash
PYTHONPATH=src HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  MPLCONFIGDIR="$ARTIFACTS/mpl" .venv/bin/python scripts/phase6_grounding_demo.py \
  --observation /private/tmp/go2-phase6/data/outputs/phase6/evaluation/s_fridge/observation.vlnobs.npz \
  --instruction 'Find the refrigerator.' \
  --cache-dir /private/tmp/go2-phase6/data/outputs/phase6/hf_cache \
  --output "$ARTIFACTS/grounding_compatibility"
```

The new compatibility output, including the source observation NPZ, is persistent.
The command above records the actual original input/cache locations; the main Phase 6B
evidence does not depend on `/private/tmp` for persistence.

## Artifacts, reproduction and remaining validation

Persistent private root:
`/Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6b/`.
`episodes/` has source packages; `pilot_v1/` has every call's exact input tensors,
generated IDs, raw text, metadata and review PNG; `pilot_v1/review.mp4` is H.264
1650×990, 16 frames, 16 s at 1 fps. This presentation speed is not robot replay time.
`audit/` contains frozen hashes, logs, development prompts and agent review.
`hf_cache/` retains pinned weights. All are private and Git-ignored.

[Demo/reproduction](TEMPORAL_VLN_DEMO.md) gives installation, prepare/download/run,
single-decision and test commands. [Architecture](TEMPORAL_VLN_ARCHITECTURE.md) gives
input/output semantics. [Episode interface](VLN_EPISODE_INTERFACE.md) is Claude's
stable contract. Do not overwrite evaluated packages or run outputs.

Human references and diverse held-out episodes are the next research requirement.
Useful temporal reasoning, absent-target resolution and progress tracking remain
unverified. Before metric grounding or planning: measured held-out-validated
T_base_camera and aligned depth, timing/odometry validation, documented action frame,
independent traversability/collision checks, then a separate controlled closed-loop
study with explicit authorization. Recorded replay cannot establish goal-reaching,
collision avoidance, safe execution or recovery after counterfactual actions.
