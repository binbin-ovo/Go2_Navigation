# Bounded visual context experiment — 2026-11-16

Branch `feature/vln-bounded-context`, separate worktree
`/Users/yushan/Desktop/Go2_Navigation/data/outputs/vln-bounded-context-worktree`,
based exactly on `1f93d7eff57906ce26c4b3ab4109effb4ca71edd`
(`feature/vln-cached-live-validation`). Main, operational defaults, the native
checkpoint, source-local-v2 pooling patch, hardware/calibration environments,
Claude's packages and all historical results remain unchanged.

**Status: implementation and contract verification complete; learned comparison
and full regression pending a new shared-Mac profiling window.** The earlier
authorization covered the completed 5/17/37 latency experiment. A new coordination
question was sent before this experiment; no heavy profiling has started. No robot
connection, DDS initialization, command-client import or physical command is used.
This branch adds an offline experiment, not a live-context default or motion path.

## Evidence motivating the experiment

[Claude's cached live validation](VLN_CACHED_LIVE_VALIDATION.md) supplied 5/19/39
frames and produced estimated exposure-to-display ages of 3.651/5.033/8.540 s, all
stale under the unchanged **3 s** limit. Caching and asynchronous evidence writing
worked. The operator judged the doorway targets appropriate in that scene. These
are historical measurements, not results of this new candidate.

## Pinned native temporal contract

Checkpoint/processor/RVQ: `LightOriginsHQ/LightNav-0`, revision
`826dc5fbfa37afa8293d2e336d329b6ffc0bfb64`. Source revision
`3015508b70fb6e30bbd66701e94c9b916dcb8df8`; existing source-local-v2 patch SHA256
`bec34b7d855f33ac94ec212a61ca507d2d677d82441add0cd1b2dfe7088e554c`.
No upstream asset was edited. Inspected actual cached implementation:

- `inference/policies.py`: `observe` stores raw RGB copies and a growing normalized
  full-episode tensor. The view returned by `_get_video_tensor` is already O(1).
  `reset` clears images, IDs, vision cache and engine episode state. This replay
  integration resets/reconstructs **each decision**, not the instruction's temporal
  origin, and supplies no proposed/executed actions.
- Checkpoint `eval_config.json`: current ages 0–1, dense fast ages 2–33, mid burst
  ages 34–89 (pair stride 6), long span ages 90–1,000,000 (14 pairs), absolute
  anchors 0/1. Spatial pooling factors 1/2/2/4/4; post-ViT; nominal 4 Hz;
  relative timestamps; unified RVQ navigation prompt; 256×448 input.
- `slowfast.py`: native `validate_slowfast_tiers` accepts ordered, disjoint dense,
  burst, span and anchor tiers. Dense observations take precedence over anchors.
  Segments are chronological; odd segments repeat their last image only for the
  model's two-frame tubelet padding. This is disclosed processor padding, not a
  newly fabricated observation. Counts distinguish unique images and padded slots.
- `inference/samples.py`: the SlowFast builder ignores its `frame_ids` argument,
  indexes a full buffer by absolute logical index, uses the official prompt and
  marks `slowfast_abs_frame_indices=True`. Simply giving it a truncated/reindexed
  buffer would collapse time and change the episode anchors.
- `data_processor.py`: explicit segment positions must be integers; resampling,
  resizing and renormalization are disabled after official frame conversion.
  Segment positions and processor `video_fps` become native `VideoMetadata`.
- `processing.py`: relative time is `(global_latest_index - index) / video_fps`.
  Each temporal patch averages the first/last timestamp; prompt prefixes render
  one decimal place. No irregular-timestamp field exists in the policy interface.
  HF performs the vision forward every request; vLLM's feature cache is not used.

The unchanged baseline uses the existing latest-anchored causal grid selection:
latest real observation at or before each 4 Hz grid point, no future/duplicate
source frames, raw gaps refused. Actual stamps and quantization errors remain in
provenance. Its native timestamp text still describes the grid approximation;
that inherited approximation is not an assertion of uniform physical capture.

## The ONE candidate, frozen before model outputs

Identifier **`anchor2_recent6_actual_time_v1`**. From the same validated causal-grid
candidate list, keep its first two images and its most recent six. Overlap is
deduplicated. Maximum **eight unique images and eight native tubelet slots**.
The latest available observation is always included. New native tier configuration:

| Tier | Selection | Pooling |
|---|---|---:|
| Current | ages 0–1 | 1 |
| Recent | ages 2–5 | 2 |
| Anchor | absolute indices 0/1, unless already in a dense tier | 4 |

No mid/long images are retained. This sacrifices intermediate visual context;
anchors alone cannot reconstruct its content. No alternative budget or policy is
being searched. The baseline remains the checkpoint's full tier configuration.

**Custom sparse-buffer and timestamp adaptation, not an official streaming
configuration:** the native tier validator/selector and sample builder index a
sparse view whose logical length is the original causal list length. Only the
selected eight images are converted using the official `_convert_frame`; no full
raw/normalized episode buffer is allocated for this candidate. All requested
native indices must exist in the selection; otherwise it fails closed.

Actual observation timestamps are encoded as rounded integer millisecond ticks
since the original first observation, with a **1000 ticks/s metadata timebase**.
This is not a 1000 fps camera or a claim that irregular frames are uniform.
`total_frames` is the extent of that virtual clock, not captured-image count.
The processor's existing index/frequency arithmetic generates actual relative
times, with ≤0.5 ms encoding error before its normal one-decimal prompt rounding.
Omitted time is retained. Logical source indices, exact observation/availability
times, tick mapping, rounding errors and omitted indices are saved separately.
All native validators run; no validator or tokenizer is bypassed.

The processor timestamp rate is temporarily scoped to the candidate call and
restored even if generation fails. Spatial preprocessing, measured calibration,
normalization, conversation template, native decoder and fixed weights stay the
same. This comparison changes **context selection and its timestamp representation**;
it is not input-equivalent or a controlled weights-only comparison. Short cases
may contain the same images but different timestamp tokens.

## Fixed matrix and interpretation

The private immutable declaration was written at `2026-11-16T17:25:48.807935+00:00`:

`/Users/yushan/Desktop/Go2_Navigation/data/outputs/vln_bounded_context/experiment_20261116_01/plan.json`

SHA256 `55c73c298afe3df47ba58ac9109d39509d485d96d0a1091e0320518be50b408d`.
It binds inference/preparation/evaluation code and **863** existing source files.

| Source/case | Original supplied candidate counts | Repeats per condition |
|---|---|---:|
| `cached_live_01` requests 1/2/3 | 5 / 19 / 39 | 2 |
| E1 right turn, d0/d1/d2 | 6 / 12 / 29 | 1 |
| E2 left turn, d0/d1/d2 | 6 / 10 / 36 | 1 |
| E3 approach bottles, d0/d1/d2 | 8 / 16 / 25 | 1 |
| E4 absent sofa, d0/d1 | 4 / 17 | 1 |

Total **34 requests**, no selective retry. Live repeat 0 uses full→bounded for
each cutoff, repeat 1 bounded→full. Pilot condition order alternates by decision.
One fixed model load, then sequential requests. The first request is identified
as cold after load; subsequent requests are warmed. Two repetitions do not support
robust percentiles. No instruction, decision cutoff or human annotation is edited.
E1–E4 are correlated `test_c` development examples. E1–E3 are retrospective
proposals along an operator route; E4 is a static absent-target probe.

Execution budgets: load 150 s, each request 45 s, total host 900 s. No simultaneous
model workers. Runtime stays MPS BF16, SDPA, no quantization, greedy seed 0,
64 generated-token cap and the existing same-dtype CPU pooling fallback.

Both conditions use existing calibrated preprocessing and compression-zero PNGs.
Live jobs use independent prepared-image caches per condition/repeat; episode jobs
prepare their selected calibrated views anew using `prepare_views`. There is no
background camera arrival in this fixed-cutoff experiment. Timings include
preparation, child interval, native processor/generation/vision, response delivery
and local review-file creation. Browser paint/live arrival latency is not measured.
The plan's shorthand “cached preparation” refers to the live jobs, not a new
episode-image cache. Stage timers are nested, not additive.

Reconstructed live ages add the **original estimated exposure-to-submit offset**
to the newly measured offline request-to-review-file interval. Original exposure
is receipt minus an assumed 48 ms (σ16 ms, previous single-boot measurement), not
hardware-synchronized on this boot. Episode ages use declared cutoff minus latest
estimated exposure plus processing time. Neither is measured live latency.
The 3 s advisory threshold is never raised.

Declared material latency effect: at least 20% reduction in median 39-frame
request-to-review-file time. Behavior checks: no lost E1–E3 reference action
agreements, no new wrong-direction or missed-native-STOP cases, plus inspection of
**all** changed native pointing/chunks. Human acceptability of new outputs remains
review pending until reviewed; syntax validity alone cannot satisfy it.

Comparison retains the existing lossy first-waypoint mapping (5° turn, 2 cm
advance/lateral thresholds; mixed motion unsupported), all ten native absolute
ego SE(2) waypoints, target tokens, native decoded STOP and APOS1299 completion
claims. A first yaw opposite a single-direction human reference is flagged;
continued first-yaw turning at STOP-only completed references is flagged separately.
These diagnostics are not executed behavior. No real `T_base_camera` is invented.

E4 retains the historical reference permitting either search turn. Agreement is
reported only as **search-action agreement**, never absent-target recognition.
The earlier generic prompt's UNRESOLVED-for-absence mismatch is preserved; no
historical score is overwritten. STOP and task-completion claims remain distinct.

## Resource bounds and failure behavior

- Eight candidate image tensors (at 3×256×448 float32, 10.5 MiB), plus at most
  eight segment tensor slots, transient source/preprocessor tensors and the model.
  This is not a bound on model activations or total RSS; those must be measured.
- Metadata/source history: ≤1024 candidates and ≤60 s; over-limit history is
  rejected, never silently restarted. This is a bounded offline diagnostic, not
  indefinitely running sparse streaming. Baseline's growing full buffer is unchanged.
- Existing live calibrated cache: ≤128 links /256 MiB each; four declared independent
  caches at most. Retained evidence and original recordings are outside cache bounds.
- One inference in flight; no pending request queue; existing IPC result queue two
  messages; one asynchronous tensor archive, ≤256 MiB owned-array cap. Writer
  backpressure/errors and final completion are retained. Full-run artifact growth
  is bounded by the fixed 34 requests, not an automatic deletion policy.
- Missing/gapped/noncausal data fail closed; source hashes are checked before/after.
  Partial failures and remaining requests are recorded; nothing is rerun until useful.
- New contexts use every-decision reset with the original instruction and original
  time origin. No history, previous proposals, labels or executed-action claim is
  carried between instruction/epoch jobs. There is no streaming/live CLI integration.

## Implemented files and actual checks

- `src/go2_perception/vln/bounded_context.py`: pure selection/validation and lazy
  native sample factory. Ordinary imports do not load a model or robot adapter.
- `scripts/run_vln_bounded_context.py`: declare/run with fixed source binding,
  sequential fixed-cutoff preparation and inherited local process worker bounds.
- `scripts/vln_bounded_worker.py`: inference-only pipe worker using the existing
  pinned shadow bootstrap; label/action whitelist retained.
- `scripts/summarize_vln_bounded_context.py`: separate evaluation-only reader,
  exact baseline tensor-bit/native-output comparison against original cached live
  results, selected-image pixel comparison, all-row review links and score summary.
- Minimal optional hooks in `lightnav_worker.py`, `vln_shadow_worker.py` and
  `shadow_worker.py`; ordinary full-history/default invocations keep their protocol.

**Actual lightweight tests:** 125 passed /0 failed /0 skipped in 15.03 s for
`test_vln_bounded_context.py`, `test_vln_latency.py`, `test_vln_shadow.py`,
`test_vln_live_shadow.py`. This includes 33 new core tests and 92 prior advisory tests.
Ten isolated `test_bounded_native_integration.py` tests passed /0 failed /0 skipped
in 4.00 s; source/processor/config loaded, **no neural weights or model inference**.
They verify native tier agreement, current/anchor preservation, the actual irregular
timestamp prompt, reset/buffer bounds and restoration of the baseline after a
candidate or mocked generation failure. A mock bundle initially omitted the native
tier normalization step; that test fixture was corrected. No candidate tuning was
based on a model outcome.

**Not run:** the 34 learned requests, their behavior/latency/memory comparison,
the full offline regression suite, or any new live test. Accordingly no speedup,
sub-3-second freshness, acceptable navigation behavior or categorical experiment
decision is claimed. No live retest is recommended without offline evidence.

## Exact continuation commands

Use the existing worktree and environments; no installation/download needed.
After the operator confirms the new shared-Mac window, run sequentially:

```bash
cd /Users/yushan/Desktop/Go2_Navigation/data/outputs/vln-bounded-context-worktree
env -u LIGHTNAV_ASSETS -u SMOLVLM_MODEL_CACHE GO2_HARDWARE=0 MUJOCO_RENDER_TEST=0 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
  TOKENIZERS_PARALLELISM=false PYTHONPATH=src \
  /Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6c/core-env/bin/python -u \
  scripts/run_vln_bounded_context.py --run \
  --model-python /Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6c/model-env/bin/python \
  --assets /Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6c/lightnav \
  --output /Users/yushan/Desktop/Go2_Navigation/data/outputs/vln_bounded_context/experiment_20261116_01

PYTHONPATH=src /Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6c/core-env/bin/python \
  scripts/summarize_vln_bounded_context.py \
  --run /Users/yushan/Desktop/Go2_Navigation/data/outputs/vln_bounded_context/experiment_20261116_01

env -u LIGHTNAV_ASSETS GO2_HARDWARE=0 MUJOCO_RENDER_TEST=0 HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false \
  SMOLVLM_MODEL_CACHE=/Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6b/hf_cache \
  PYTHONPATH=src /Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6c/core-env/bin/python \
  -m pytest -q -rs

env -u SMOLVLM_MODEL_CACHE GO2_HARDWARE=0 MUJOCO_RENDER_TEST=0 HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 \
  LIGHTNAV_ASSETS=/Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6c/lightnav \
  PYTHONPATH=src /Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6c/model-env/bin/python \
  -m pytest -q -rs tests/test_bounded_native_integration.py \
  tests/test_lightnav_runtime_integration.py tests/test_waypoint_native_integration.py \
  tests/test_vln_simulation.py tests/test_waypoint_simulation.py
```

For legacy viewer subprocess tests the ignored worktree `.venv` may link to the
existing core-env (no environment edits). Inspect hardware test selection first;
`GO2_HARDWARE=0` and `MUJOCO_RENDER_TEST=0` must remain set. Optional native tests
skip genuinely when `LIGHTNAV_ASSETS` is absent from the core run. Capture fresh logs.
Do not run these suites concurrently with model profiling or physical testing.

To reproduce the declaration into a **new** destination:

```bash
PYTHONPATH=src /Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6c/core-env/bin/python \
  scripts/run_vln_bounded_context.py --declare-only \
  --source-run /Users/yushan/Desktop/Go2_Navigation/data/outputs/vln_live/cached_live_01 \
  --episodes /Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6b/episodes \
  --output /Users/yushan/Desktop/Go2_Navigation/data/outputs/vln_bounded_context/NEW_UNIQUE_RUN
```

Existing plan/source drift causes refusal. Preserve failed/partial directories;
do not remove them to retry invisibly. New outputs will be under `results/` (all
native evidence and local review pages) and `evaluation/` (human-reference-only
summary). They do not exist until the corresponding stages actually run. Only
safe code/docs are published; private images, annotations and artifacts stay ignored.
