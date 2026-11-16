# Bounded visual context experiment — 2026-11-16

Branch `feature/vln-bounded-context`, separate worktree
`/Users/yushan/Desktop/Go2_Navigation/data/outputs/vln-bounded-context-worktree`,
based exactly on `1f93d7eff57906ce26c4b3ab4109effb4ca71edd`
(`feature/vln-cached-live-validation`). Main, operational defaults, the native
checkpoint, source-local-v2 pooling patch, hardware/calibration environments,
Claude's packages and all historical results remain unchanged.

**Decision: the candidate is faster, but behavior diagnostics degrade. Do not
promote it to the live advisory or recommend a live retest on this evidence.**
All 34 declared requests ran once, without retries, on implementation
`28affec52677a61e6e5695002681872023c182b9`. The 39-frame case's median delivery
duration fell from **8.052 to 2.193 s (72.8%)**. E1–E3 reference agreement fell from
**3/9 to 2/9**, a completed-turn case acquired an additional initial turn, and a
clearly visible shoe-rack target lost its native visible-target prediction.
Both conditions still missed the two STOP-only completion references. These are
small, correlated development diagnostics with a lossy action mapping, not a
general navigation-quality estimate.

The operator confirmed stationary Go2, completed cleanup and paused physical
testing for this new window. Model profiling, then core regression, then optional
native/numeric regression ran sequentially. **716 core tests passed /19 skipped;
27 optional tests passed /2 rendering tests skipped; zero failures.** No robot
connection, DDS initialization, command-client import or physical command was used.
This remains an offline experiment, not a live-context default or motion path.

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
same. Images assigned to the anchor tier receive its factor-4 post-ViT pooling,
even when full history would still place those images in the factor-2 fast tier.
This comparison changes **context selection, tier assignment and timestamp representation**;
it is not input-equivalent or a controlled weights-only comparison. Short cases
may contain the same images but different timestamp tokens or pooling. Anchors
are the first two entries of the common latest-anchored causal sample list, not
guaranteed to be the exact first two raw camera captures. Their actual IDs/stamps
are recorded; upstream resampling can change earlier selections across cutoffs.

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

## Measured performance and all declared requests

Actual host: Apple M3 Pro, macOS 26.6.2 arm64, 36 GiB unified RAM. PyTorch 2.10.0,
Transformers 5.8.0, NumPy 2.4.6 in the model environment; existing core environment
performed calibrated preprocessing. BF16/MPS, SDPA, unquantized, fixed checkpoint
and pooling patch as above. Approximately 24.6 GiB free disk before the run.
The model loaded once in **45.564 s**, measured separately. Full run host time,
including startup/loading, requests, archive drain and preservation check: **159.643 s**.
No timeout, decode error, unattempted request or retry. All **34/34** archives completed.

Every live timing repeat appears below. Units seconds, format `repeat 0 / repeat 1`.
Delivery means preparation start through local review-file creation, not browser
paint. `r1/full/repeat 0` is the only cold-after-load request; it is not a fair
standalone comparison against an already warmed candidate.

| Saved live cutoff | Full supplied / native unique / padded slots | Bounded supplied / native unique / padded slots | Full delivery | Bounded delivery | Full reconstructed age | Bounded reconstructed age |
|---|---|---|---|---|---|---|
| r1, ~1.12 s history | 5 / 5 / 6 | 5 / 5 / 6 | 3.027 / 1.869 | 1.899 / 1.841 | 3.154 / 1.995 | 2.025 / 1.967 |
| r2, ~4.62 s history | 19 / 19 / 20 | 8 / 8 / 8 | 4.450 / 4.254 | 2.245 / 2.022 | 4.599 / 4.402 | 2.394 / 2.170 |
| r3, ~9.52 s history | 39 / 38 / 38 | 8 / 8 / 8 | 8.037 / 8.066 | 2.164 / 2.221 | 8.172 / 8.201 | 2.299 / 2.356 |

The warmed r1 comparison is **1.869 vs 1.841 s**, little difference when five images
are supplied to both. The paired median 19-frame delivery falls **51.0%**, and the
39-frame delivery falls **72.8%**. With only two repeats there is no p95 or broad
variability claim. Native outputs are identical across the two repetitions within
each case/condition.

Median stage durations (nested stages, **do not sum** processor/generation/vision):

| Case/condition | Parent prepare | Child interval | Native generate | Vision + pooling | Official processor | Response delivery | Local HTML write |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5/full, cold included | 0.186 | 2.252 | 2.077 | 0.830 | 0.021 | 0.0014 | 0.0014 |
| 5/bounded | 0.165 | 1.692 | 1.597 | 0.575 | 0.005 | 0.0021 | 0.0013 |
| 19/full | 0.471 | 3.848 | 3.522 | 1.905 | 0.010 | 0.0038 | 0.0024 |
| 19/bounded | 0.192 | 1.927 | 1.780 | 0.730 | 0.005 | 0.0017 | 0.0013 |
| 39/full | 1.104 | 6.873 | 6.120 | 3.682 | 0.050 | 0.0048 | 0.0053 |
| 39/bounded | 0.218 | 1.956 | 1.815 | 0.762 | 0.005 | 0.0045 | 0.0028 |

Dispatch/pipe/worker pre-validation contributes another median **0.065 vs 0.011 s**
in the 39-frame comparison (outside the child timer, included in total delivery).
Its full/candidate history reconstruction was 0.332/0.065 s; sample/model-image
evidence was 0.318/0.063 s. Asynchronous tensor archive writes took 0.116–0.963 s
across requests, outside delivery unless bounded writer backpressure occurs.

All remaining 22 requests are recorded here; `U` means the fixed first-waypoint
mapping is unsupported, not a fabricated discrete label. Native chunks are retained
in full in each request. Times are full/bounded seconds.

| Pilot cutoff | Supplied full→bounded | Delivery full / bounded | Reconstructed age full / bounded | Mapped action full→bounded | Existing-reference agreement full→bounded |
|---|---|---|---|---|---|
| E1 d0 | 6→6 | 1.952 / 1.983 | 2.073 / 2.104 | U→U | no→no |
| E1 d1 | 12→8 | 3.172 / 2.428 | 3.323 / 2.579 | ADVANCE→ADVANCE | no→no |
| E1 d2 | 29→8 | 6.247 / 2.130 | 6.358 / 2.241 | ADVANCE→U | no→no |
| E2 d0 | 6→6 | 1.802 / 1.848 | 1.909 / 1.955 | U→U | no→no |
| E2 d1 | 10→8 | 2.585 / 2.088 | 2.746 / 2.249 | ADVANCE→U | no→no |
| E2 d2 | 36→8 | 7.528 / 2.087 | 7.646 / 2.204 | ADVANCE→ADVANCE | no→no |
| E3 d0 | 8→8 | 2.199 / 2.262 | 2.354 / 2.417 | ADVANCE→ADVANCE | yes→yes |
| E3 d1 | 16→8 | 4.093 / 2.092 | 4.216 / 2.214 | ADVANCE→ADVANCE | yes→yes |
| E3 d2 | 25→8 | 5.692 / 2.208 | 5.847 / 2.364 | ADVANCE→U | yes→no |
| E4 d0 | 4→4 | 1.413 / 1.520 | 1.538 / 1.644 | TURN_RIGHT→TURN_RIGHT | search yes→yes |
| E4 d1 | 17→8 | 4.024 / 2.198 | 4.123 / 2.297 | TURN_RIGHT→TURN_RIGHT | search yes→yes |

Every supplied pilot image was natively used. Odd full histories 29/25/17 each
received one native tubelet-padding slot (30/26/18 total); bounded slots never
exceeded eight. All selected source timestamps and omissions are in `preparation.json`;
actual processor positions are in `native/native_sample.json` and tokenized prompt.
For example, the 39-frame candidate preserves relative-to-first observations at
**0.000, 0.138, 8.261, 8.404, 8.682, 8.963, 9.241, 9.522 s** after millisecond
rounding. The omitted interval is not compressed into a two-second clip.

At the unchanged 3 s threshold, reconstructed ages were below it for **17/17
bounded requests vs 6/17 full requests**; for the live-recording subset, **6/6 vs
1/6**. This is fixed-cutoff offline evidence with original age offsets, not a new
live arrival/queue test. It does not supersede Claude's actual all-stale live run.

## Behavior, native pointing and stopping

All 14 unique case pairs were inspected using current native inputs, full waypoint
chunks and original human references; timing repetitions had identical native
outputs. This inspection is by Codex, **not new human-reviewed ground truth**.

| Diagnostic | Full | Bounded |
|---|---:|---:|
| E1–E3 human-reference agreement | 3/9 | 2/9 |
| Supported discrete mappings within E1–E3 | 7/9 | 4/9 |
| Opposite initial yaw at single-direction references | 0 | 0 |
| Missed native STOP at STOP-only completed references | 2/2 | 2/2 |
| Initial turn ≥5° at those completed references | 0 | 1 |
| Native decoded STOP, all requests | 0/17 | 0/17 |
| APOS1299 completion/stop claim, all requests | 0/17 | 0/17 |
| E4 permitted search-action agreement | 2/2 | 2/2 |

- **Saved live r1/r2:** OPOS target tokens 501/502 and all ten waypoints stay equal;
  APOS affordance changes from 885/837 to 933. **r3:** full history changes to a
  straight ~1.51 m chunk with OPOS504/APOS1273; bounded retains the earlier
  ~1.45 m forward, 0.39 m left chunk with ~15° yaw and OPOS501/APOS933. Target
  points appear on the doorway in both; this does not validate either motion path.
- **E1 d0:** both predict the same right-curving mixed motion (correct yaw sign,
  unsupported by the discrete mapping) and OPOS0. **d1:** both initially advance,
  outside the STOP/right reference, but select different counter-area target points
  (OPOS363→414) and later right-turn chunks. **d2:** both miss STOP; bounded adds
  an initial −6.1° right turn versus approximately 0° and increases the last lateral
  displacement from −0.14 to −0.54 m. OPOS512→467 stays on the cabinet/counter area.
  The predeclared repeated-initial-turn flag is triggered; this was not executed.
- **E2 d0:** identical left-curving mixed motion, correct yaw sign, OPOS0. **d1:**
  both retain OPOS0 despite the reviewed target's visibility; final yaw increases
  from +8.3° to +58.5° in the candidate. **d2:** the shoe rack is clearly visible
  in the image. Full history points to it (OPOS501); bounded emits **OPOS0 = target
  not visible**. Both initially advance and miss STOP, with candidate last forward
  displacement ~0.95 vs ~0.41 m and final yaw +41.8° vs +14.8°.
- **E3 d0:** same OPOS797 bottle target and ADVANCE mapping, but bounded shortens
  the last forward coordinate from 1.50 to 0.95 m and introduces final yaw −34.5°.
  Even without omitting images at this eight-image cutoff, timestamp/pooling tier
  changes alter the native prediction. **d1:** bottle target moves OPOS847→849;
  both map ADVANCE, with different shorter trajectories. **d2:** OPOS848→845 still
  lies in the bottle area; bounded emits APOS1298 (rotate-right affordance), first
  forward/yaw ~0.036 m/−8.8°, last forward ~0.12 m versus full ~0.68 m. This is mixed
  turn/translation, so the frozen mapping becomes unsupported and loses agreement.
  It is not proof that any small right turn is inherently wrong; the human reference
  allows right turns, and the lossy mapping has an explicitly documented limit.
- **E4 d0:** identical right-search chunk, OPOS0 and APOS1298. **d1:** same invisible
  target/right-search signals, but bounded's last yaw is −90° versus −150°.
  OPOS0 is consistent with a target absent from the image, not proof of global
  absence or successful searching. E2's false negative also shows why OPOS0 alone
  cannot establish robust absence recognition. No STOP/completion or navigation
  success is credited. The historical generic-prompt/reference mismatch remains.

Thus the predeclared latency criterion passed, but behavior retention failed. The
direction of the latency effect is clear in these long-history cases; an acceptable
navigation tradeoff has **not** been established. Preserve this negative diagnostic,
keep the full-history operational policy, and obtain human review of the saved
changes before considering a separately declared follow-up. No additional candidate,
prompt adjustment, checkpoint change, live retest or architecture migration was run.

## Measured memory, equivalence and preservation

The child process's lifetime peak RSS was **11,633,770,496 bytes (10.83 GiB)**,
reached during load. Per-request sampled RSS peaked at 1,246,724,096 bytes for full
and 1,171,308,544 bytes for bounded; MPS driver allocations are separate accounting
on unified memory, so low post-load RSS does not mean the weights fit in that RSS.
Sampled maximum MPS allocated bytes: full **9,966,786,304**, bounded **8,987,931,904**;
maximum driver bytes: full **12,963,840,000**, bounded **12,963,872,768**. Allocator
caches persist across the interleaved conditions, and 20 ms samples are lower
bounds, not isolated-condition peak-memory benchmarks. Do not add RSS and MPS
figures. Parent lifetime RSS peak: **208,748,544 bytes (~199 MiB)**.

Prepared live cache peaks per repetition/condition: full **53 entries /146,811,696
bytes**, bounded **18 entries /49,860,576 bytes**, no evictions or bypasses. Four
separate caches prevent cross-condition/repetition reuse; each remains within its
128-entry/256-MiB bound. Across each live sequence, full had 10 hits/53 misses;
bounded 3 hits/18 misses. There was one request in flight, no queued request and
one outstanding tensor archive at most. Peak writer-owned arrays: **26,164,683
bytes**, below 256 MiB; 34 submitted, 34 completed, zero outstanding/errors at close.

**All six full-history live replays exactly reproduce the original cached-live
tensor archives, including BF16 bits and generated token IDs.** Native text,
pointing, complete waypoints and STOP flags match the historical result as well.
This verifies the unchanged full-history baseline after interleaved candidate calls.
All **17 paired comparisons** retain identical shared calibrated pixels and
instruction/episode/clock/cutoff identity. The **101** shared normalized model-space
images across unique case pairs also match exactly. The candidate's context tokens,
pooling layout and many native outputs differ as intended; no full input/output
equivalence claim is made. All **863** original source files passed post-run hash
checks, with zero changed files. Original annotations/results were never overwritten.

The inherited `results/worker/model.json` describes the baseline bootstrap config;
`plan.json`, each `preparation.json` and each bounded result's `context_policy` record
the actual per-request candidate override. The fixed model identity remains common.

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
  Partial failures and remaining requests are recorded; no selective retries.
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

**Subsequent authorized verification:** the 34 learned requests and separate
evaluation completed, followed by **716 passed /0 failed /19 skipped in 62.49 s**
for the full core suite, then **27 passed /0 failed /2 skipped in 3.58 s** in the
isolated native/numeric suite. Core skips: one physical Go2 test, sixteen optional
native runtime tests, two module-level MuJoCo availability skips. The native cases
and numeric simulator cases pass in the second suite; two OpenGL rendering tests
remain intentionally skipped. No physical or rendering opt-in was enabled. Tests
were selected/inspected before execution; no hardware module's test body ran.

**Not run:** a new live test, physical execution, human review of new model outputs,
dynamic-safety evaluation or an indefinitely running bounded-memory stream. The
negative behavior result does not justify another live test or a default change.

## Exact execution and reproduction commands

These stages were executed sequentially using the existing environments, after
operator confirmation. No installation/download. The existing result directories
are immutable: rerunning the first/evaluation command there will refuse to overwrite
them. To reproduce, first declare into a new unique destination as shown below and
use that destination consistently. Repeating heavy work requires resource coordination.

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
  PYTHONPATH=src .venv/bin/python \
  -m pytest -q -rs \
  --junitxml=/Users/yushan/Desktop/Go2_Navigation/data/outputs/vln_bounded_context/experiment_20261116_01/full_regression.xml

env -u SMOLVLM_MODEL_CACHE GO2_HARDWARE=0 MUJOCO_RENDER_TEST=0 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
  TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 \
  LIGHTNAV_ASSETS=/Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6c/lightnav \
  PYTHONPATH=src /Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6c/model-env/bin/python \
  -m pytest -q -rs tests/test_bounded_native_integration.py \
  tests/test_lightnav_runtime_integration.py tests/test_waypoint_native_integration.py \
  tests/test_vln_simulation.py tests/test_waypoint_simulation.py \
  --junitxml=/Users/yushan/Desktop/Go2_Navigation/data/outputs/vln_bounded_context/experiment_20261116_01/optional_regression.xml
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
do not remove them to retry invisibly. Actual artifacts in the declared private root:

- `run.log`, `preflight.json`, `results/summary.json`, `results/worker/model.json`.
- `results/rows.json`, all 34 request folders (`job.json`, `preparation.json`,
  `response.json`, `timing.json`, `native/`, `index.html`); full native tensors,
  prompts, current/history images, pointing and all waypoint chunks retained.
- `evaluation/comparison.json` and `evaluation/index.html`: all rows and paired
  checks, with existing human references confined to evaluation-only panels.
- `evaluation/visual_inspection/comparison_01.png` through `comparison_04.png`:
  all 14 unique case pairs, OPOS/APOS overlays and native chunks. No fabricated
  success or failure omitted. `visual_inspection_notes.json` labels the reviewer as
  Codex, not a human annotator; `normalized_image_checks.json` records 101 equal images.
- `inspect_outputs.py`: retained private postprocessing script. The command actually
  used was `PYTHONPATH=src .venv/bin/python <private-root>/inspect_outputs.py`.
  Its existing-directory guard refuses to overwrite the sheets; to reproduce them,
  copy the script and `evaluation/comparison.json` into the same relative locations
  in a fresh derived-output root, then run the copied script. Existing absolute
  input references remain read-only; outputs go beneath that fresh root.
- `evaluation.log`, `full_regression.log/.xml`, `optional_regression.log/.xml`, and
  earlier `native_contract_tests.log`. Test logs are captured stdout/stderr, not estimates.

Only safe code/docs and aggregate findings are published; private images,
annotations, packages and artifacts remain local and Git-ignored.
