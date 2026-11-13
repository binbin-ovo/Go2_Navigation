# Phase 6B temporal instruction-following proposals

Isolated branch `phase6-temporal-vln`, based on verified Phase 6A `c13f1dc`.
The published [episode contract](VLN_EPISODE_INTERFACE.md) is stable v1.
No calibration, tracking, locomotion or operational-default changes.

## Model feasibility audit (before weights, 2026-11-12)

Only these three candidates were audited. Actual machine: Apple M3 Pro, 36 GiB
unified RAM, macOS arm64. CPU is available; CUDA unavailable, MPS unavailable in
the tested process. CPU FP32 is selected. No hosted inference or private-image upload.

| Candidate | Inputs / preprocessing | Native output / semantics | Availability/license/resources | Decision |
|---|---|---|---|---|
| NaVILA, `a8cheng/navila-llama3-8b-8f` | Current plus sampled historical RGB; VILA/SigLIP conversation formatting, eight-frame release. High-level VLA needs no depth/panorama/pose; separate locomotion layer has other inputs. | Egocentric textual increments, forward cm / turns degrees / stop. Not a measured base_link waypoint. | Public checkpoint; code Apache-2.0 but checkpoint has no model card/license declaration, upstream Llama terms need verification. ~16 GB FP16 weights estimate plus activations/cache. Released evaluation requires Linux CUDA FlashAttention, patched Transformers 4.37.2/Habitat. | Navigation-specific but no verified Mac installation path; porting/licensing unresolved. Not downloaded or executed. |
| SmolVLM2-500M-Video-Instruct | RGB image/multiple images/video + arbitrary text, official SmolVLM processor/chat template. No depth, panorama or pose. | Free-form text; no native robot action frame. | Public Apache-2.0; roughly 2 GB FP32 weights estimate, vendor reports 1.8 GB GPU video memory (not our measurement). Transformers supports CPU eager attention. | Feasible smaller candidate; not executed. |
| **SmolVLM2-2.2B-Instruct** | Same official multi-image/video input capability, arbitrary instructions. Native video settings: 1 fps, max 64, 384px longest edge; image defaults split 1536px into tiles. | Free-form instruction-conditioned text. Adapter action labels have no pretrained robot-frame guarantee. | Public Apache-2.0, checkpoint+processor revision `be92141c8d0ceb98596a923c308ac2bbb4a7c466`. Hub metadata: 2,246,784,880 FP32 parameters (~8.99 GB weight bytes). Estimated 10–18 GiB process RAM with bounded 4-frame input. Vendor's 5.2 GB GPU figure uses a different runtime. | **Selected for actual execution:** generic **VLM navigation-proposal baseline**, not a reproduced VLN policy. CPU eager FP32, low-memory checkpoint loading. |

Official sources: [NaVILA code/setup](https://github.com/AnjieCheng/NaVILA),
[NaVILA checkpoint](https://huggingface.co/a8cheng/navila-llama3-8b-8f),
[NaVILA paper](https://arxiv.org/abs/2412.04453),
[SmolVLM2 500M card/examples](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct),
[SmolVLM2 2.2B card/examples](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct),
[SmolVLM paper](https://arxiv.org/abs/2504.05299),
[official implementation](https://github.com/huggingface/transformers/tree/v4.51.3/src/transformers/models/smolvlm),
[4.51.3 API](https://huggingface.co/docs/transformers/v4.51.3/en/model_doc/smolvlm).
The model's research baseline use is offline; its card does not establish safety for
robot decisions. Resource estimates are distinct from the actual experiment report.

## Temporal input design

The selected model natively supports interleaved multiple images and text, and a
single image. Use that supported interface for both conditions: at most four
chronological available frames over six seconds, including latest, versus latest
only at the same cutoff. This bounded sampling is our declared adapter policy,
not a reproduction of a trained navigation policy's history protocol. No fabricated
frames, panorama or depth. Gaps and relative times are supplied, human annotations
are not. Each decision is a fresh prompt; no teacher-forced labels or previous
model actions are inserted. This avoids inventing the counterfactual camera route.

Every image is independently rectified using Phase 6A code and measured intrinsics.
Use the official processor with explicit image size longest-edge 384 and image
splitting disabled to bound memory, as supported by the processor (matching native
video resolution, but using multi-image chat). This is a recorded override of the
1536px image defaults. Official implementation first resizes preserving aspect,
then squashes each unsplit image to 384×384; record both resize stages and their K.
No crop or spatial pad for these same-size inputs. Normalization is official
mean/std 0.5/0.5, **not** Grounding DINO's ImageNet values. Never normalize twice.

Raw images, rectified views, processor tensor previews, source/processed intrinsics,
FOV changes, token/image counts, hashes, exact chat prompt and processor settings
are retained per call. Passing historical arrays to the vision encoder, with
matching image tokens, is required to claim history consumption. Whether the model
uses history *usefully* is a separate question requiring evaluation.

## Components and data flow

```text
private recording snippets -> prepare_temporal_vln_pilot.py -> episode.json + hashed images/calibration
                                                                 |
                          load_episode -> select_history(cutoff, horizon, budget)
                                                                 |
                    calibrated_view per image -> official SmolVLMProcessor -> tensors + provenance
                                                                 |
                 SmolVLM2.generate + vision-encoder hook -> native output + conservative parser
                                                                 |
                                 proposal.json + tensors/tokens + review PNG/video
                                                                 |
separate annotations -> load_annotations (after inference only) -> evaluation panel / summarize
```

`episodes.py` validates the versioned JSON contracts and package-relative hashes;
`temporal_inputs.py` owns causal selection integration, rectification and official
chat/image processing. `temporal_model.py` loads pinned local weights, checks the
CPU/memory/geometry configuration and records actual vision-encoder input shapes.
`temporal_proposals.py` parses native fields conservatively. `temporal_review.py`
renders evidence and evaluates only separately supplied human-reviewed references.
`scripts/run_temporal_vln.py` freezes declarations and persists every call; it refuses
to overwrite a result directory. The preparation script copies selected private
recording bytes and never edits source recordings.

### Python entry points

```python
from pathlib import Path
import yaml
from go2_perception.vln.episodes import load_episode, select_history
from go2_perception.vln.temporal_model import SmolTemporalModel

episode = load_episode(Path('private_episode/episode.json'))
selection = select_history(episode, 'd00', max_frames=4, horizon_s=6.0)
config = yaml.safe_load(Path('configs/temporal_vln.yaml').read_text())
model = SmolTemporalModel(config, Path('private_cache'))  # no download by default
proposal, prepared, tensors = model.infer(
    episode, 'd00', Path('configs/temporal_vln_prompt.txt'), mode='temporal')
assert proposal['execution_authorized'] is False
```

`prepared` is a TemporalInput: selection, calibrated views, annotation-free messages,
official prompt text, model tensors, provenance and exact tensor previews. `tensors`
also includes generated token IDs for replay inspection. Use `mode='current_only'`
for the native single-image ablation, never a synthetic video padded with duplicates.
A decision with no available image raises an explicit input error; it does not run
the model on a fabricated frame. Invalid manifests/changed images fail before forward.

### Time and geometry semantics

An episode contains exactly one source recording and clock epoch. History must
satisfy both exposure <= cutoff and availability <= cutoff, inside episode bounds
and six-second lookback. A uniform chronological subsample includes the latest
available image. No cross-episode state or future action feedback is kept. Explicit
missing frame IDs and gaps are retained; gap ends are clipped at the cutoff.

Raw Go2 frames use measured fisheye K/D at 1280×720. Each frame becomes a rectified
pinhole image with its own K and zero distortion; processor resizes carry pixel-center
intrinsics through both stages. Raw/view/tensor dimensions, FOV, crop/padding,
normalization settings, times, image hashes and clocks remain in input provenance.
Optical axes are right/down/forward; base_link is forward/left/up. **T_base_camera is
unknown**, not identity. This adapter has no aligned depth or navigability layer.

The offline pilot's exposure and availability are estimates with explicitly recorded
provenance. Clock epoch and uncertainty are not removed by batching. Runtime start/end
are UTC/Unix host times plus monotonic elapsed seconds. Replay age adds elapsed
preprocessing/generation to source age at cutoff; historical host age is separate.
No runtime subtraction mixes host and robot clock domains.

### Proposal semantics

`go2-temporal-proposal-v1` is the saved result version (the episode JSON Schema is
independent). It retains episode/decision/instruction, selected observations,
model/checkpoint/processor identities, native text and generated IDs, parse result,
start/end timestamps, preprocessing/generation/pipeline latency and process RSS.
Actual forward-hook shapes and image-token counts prove history consumption.

Parser vocabulary is ADVANCE / TURN_LEFT / TURN_RIGHT / STOP / UNRESOLVED.
Task status is separately IN_PROGRESS / COMPLETE / AMBIGUOUS / TARGET_NOT_FOUND /
INSUFFICIENT_OBSERVATION, or null if absent. Accepted means valid syntax and known
fields, **not** correct instruction following. Supported formats are one strict JSON
object or the anchored explicit native fields observed in synthetic development.
Malformed/free prose/unknown actions stay rejected or unresolved, preserving raw text.
The model receives the full instruction; no keyword-to-action logic is used.

Native action frame is null: this generic VLM has no documented physical action
reference. Explicit native quantities retain units (m/cm or degrees/radians) but are
unmeasured and unregistered. Missing quantities stay null. `metric_goal=null`,
`completion_verified=false`, `execution_authorized=false` on every path. STOP never
implies completion; COMPLETE remains a model claim. Prose may contradict a structured
field and is always retained for separate review. There is no robot-command adapter.

### Evaluation boundary and verified result

Only reviewed human annotations contribute to human agreement/completion metrics.
Agent reviews and pending annotations cannot enter those denominators. Unknown model
task status is not scored as verified incompletion. Progress consistency requires
separate adjudication of native text; no keyword heuristic claims to measure it.
Structured completion counts do not include semantic claims hidden in prose.

The actual pilot executed 16 calls (four episodes, eight cutoffs, two input modes).
All temporal forwards encoded four images; all temporal actions were TURN_RIGHT.
Human semantic metrics remain pending; this is no evidence of a history benefit.
Full tests including actual weights: 470 passed, only hardware skipped. Read
[TEMPORAL_VLN_EXPERIMENTS](TEMPORAL_VLN_EXPERIMENTS.md) for failures, measurements,
freeze hashes and exact scope, and [TEMPORAL_VLN_DEMO](TEMPORAL_VLN_DEMO.md) for commands.

This isolated optional `temporal-vln` dependency extra uses Transformers 4.51.3 instead
of Phase 6A's 4.49.0. Existing detector/tracker/sensor/calibration code and configs are
unchanged. A real Grounding DINO compatibility call in the new environment reproduced
the prior image candidate exactly. No operational defaults are promoted.
