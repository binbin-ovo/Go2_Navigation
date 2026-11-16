# Input-equivalent advisory latency work — 2026-11-16

Branch `feature/vln-live-latency`, worktree
`/Users/yushan/Desktop/Go2_Navigation/data/outputs/vln-live-latency-worktree`, based
exactly on `feature/vln-live-shadow` publication
`eaec3314f4ed933623bd8aad1daca944d4575363`. Main and operational defaults are unchanged.
The new path is explicitly selected with **`--preparation-mode cached`**; the default
remains `baseline`. No history-policy experiment was added.

The operator confirmed stationary Go2, completed stop/release cleanup and no physical
testing during this window. All six model requests and subsequent regressions ran
sequentially. No camera/robot connection, DDS, ownership, motion, stop, posture or
avoidance request was made. Existing environments, weights and calibration were reused.

## Diagnosis and native-contract audit

The existing parent reloads each selected observation archive, rectifies its RGB,
PNG-encodes it, and serializes a full causal job on every request. At 37 frames this
took 1.196 s before dispatch: 0.148 s source hashing/loading, 0.260 s rectification,
0.783 s image encoding/writing, 0.005 s job serialization. The child then spent
0.876 s synchronously compressing exact tensor evidence before returning its result.

Pinned source inspected under `phase6c/lightnav/source-local-v2`:

- `inference/frame_preprocessing.py`: independent RGB → normalized model-frame
  conversion. Processor/normalization/resolution remain unchanged here.
- `inference/policies.py`: SlowFast already uses a preallocated full-history buffer;
  `_get_video_tensor()` is an O(1) view. Repeated stacking is **not** a bottleneck.
- `inference/samples.py` and `slowfast.py`: current/fast/mid/long/anchor tiers and
  absolute frame indices remain the trained sample contract.
- `inference/engine.py`: HF calls `data_processor.process_sample` followed by the
  full model's `generate`. Its visual-feature/tubelet-cache path is for `vllm_local`,
  not the fixed HF runtime. No feature, KV or tubelet cache was enabled.

The latest-anchored 4 Hz selection changes earlier selected indices. The 17-frame
job shares only two frames with the 5-frame job; the 37-frame job reuses six unique
frames from the union of preceding jobs. Simply appending to native history would
change inputs. Full selection and native reset/reconstruction are retained on every
request, including all native anchors, temporal ordering and original timestamps.

## Implemented changes

1. `shadow_cache.PreparedFrameCache`: a private, run-local LRU of calibrated RGB PNGs,
   bounded to 128 entries / 256 MiB of cache links. Cache identity binds the actual
   observation archive SHA256 (pixels, calibration, stamps and epoch), source frame
   ID, clock epoch, preprocessing-code hashes, OpenCV/NumPy versions, synthetic
   opt-in and explicit PNG settings. Calibration values remain in each snapshot.
   Every source and cached image is hash-checked before reuse. PNG compression level
   zero changes encoded bytes, not decoded pixels. Request-local hard links preserve
   evidence across eviction and instruction/epoch/cancellation resets. This bound
   covers the cache, not retained run evidence; existing overall run limits still apply.
2. `shadow_evidence.TensorEvidenceWriter`: only compressed tensor archives move off
   the response path. Native text, decoder results, model-space PNGs, prompts and
   input provenance remain available synchronously. At most **one** archive can be
   outstanding, with an explicit 256 MiB owned-array cap. Arrays are copied before
   enqueue. Full queues/errors are reported, never silently dropped. Per-request
   `tensor_evidence.json` records queued/writing/complete/error. EOF drains the owned
   worker before termination; incomplete archival is also propagated to final session
   status/events. Forced shutdown cannot be reported as completed archival.
3. Stage timing in the existing worker/session and two offline profiling/comparison
   scripts. Device synchronization around the visual forward/pooling measurement is
   **profiling-only**, enabled for both declared passes; the live command below does
   not enable it. No upstream source or same-dtype pooling patch was edited.

No pose, LiDAR, goal truth, action references or evaluation labels enter model jobs.
Instruction changes, epochs, cancellation, one request in flight, history pause rules,
camera stride/inbox and the **three-second advisory threshold** are unchanged.

## Declared experiment and actual execution

Source: Claude's unchanged private `vln_live/live_session_03`, requests 000001–000003.
The recorded snapshots select 5, 17 and 37 frames at their original cutoffs. A plan
was saved before inference: **one baseline pass, then one cached pass; one request
per length in 5 → 17 → 37 order; no retries**. Each pass loaded the model once.
The first 5-frame request is cold after loading; 17 and 37 are warmed requests, with
an accumulating prepared-image cache only in the optimized pass.

These are fixed-cutoff offline measurements. All 66 source observations up to the
last cutoff were ingested, with original pixels/timestamps preserved. The next
recorded batch was supplied only between requests. This does **not** measure a live
receiver or concurrent-inference camera queue. New host delivery times describe
replay, not camera exposure. Original source files and all prior results were preserved:
**408 source-run files** match before/after hashes in both passes.

Runtime: Apple M3 Pro Mac, macOS 26.6.2 arm64; existing core Python 3.12.11 / NumPy
1.26.4 / OpenCV 4.11.0.86; isolated model Python 3.11.13 / Torch 2.10.0 /
Transformers 5.8.0 / NumPy 2.4.6. MPS, BF16, SDPA, no quantization.
Checkpoint/processor/RVQ `LightOriginsHQ/LightNav-0@826dc5fbfa37afa8293d2e336d329b6ffc0bfb64`;
upstream `3015508b70fb6e30bbd66701e94c9b916dcb8df8`; unchanged pooling patch
`bec34b7d855f33ac94ec212a61ca507d2d677d82441add0cd1b2dfe7088e554c`.
Asset verification precedes model load. Measured load: **43.985 / 43.855 s**,
baseline/cached, separate from all request times below.

### Measured request times (seconds; baseline → cached)

| Stage | 5 frames | 17 frames | 37 frames |
|---|---:|---:|---:|
| Parent preparation + dispatch | 0.165 → 0.160 | 0.532 → 0.458 | 1.214 → 1.035 |
| Child history reconstruction, including image reads | 0.069 → 0.070 | 0.209 → 0.132 | 0.461 → 0.289 |
| Native sample + model-space image evidence | 0.047 → 0.044 | 0.122 → 0.126 | 0.263 → 0.301 |
| Official processor | 0.020 → 0.047 | 0.008 → 0.008 | 0.014 → 0.014 |
| Vision forward + post-ViT pooling | 1.061 → 0.946 | 1.597 → 1.687 | 3.243 → 3.405 |
| Native generation, including vision | 2.398 → 2.334 | 3.046 → 3.219 | 5.414 → 5.688 |
| Tensor archive on response path | 0.120 → 0.001 | 0.390 → 0.002 | 0.876 → 0.003 |
| Complete worker response interval | 2.765 → 2.604 | 3.809 → 3.522 | 7.066 → 6.329 |
| Result delivery | 0.0009 → 0.0027 | 0.0032 → 0.0070 | 0.0024 → 0.0002 |
| Display rendering | 0.0024 → 0.0040 | 0.0044 → 0.0072 | 0.0101 → 0.0110 |
| **Submit → display** | **2.942 → 2.779** | **4.358 → 4.005** | **8.305 → 7.388** |

Nested intervals must not be summed: vision is part of generation; processor and
generation are inside the engine call. Full raw timing is retained. Source ingestion
and storage averaged **57.0 / 59.6 ms per observation**, baseline/cached (66 each);
this path was not optimized. Parent job serialization was 1–6 ms. Actual background
archive writes took **0.133 / 0.430 / 0.949 s** and completed successfully; that work
was moved, not eliminated. Model-space evidence still costs time synchronously.

Observed submit-to-display reductions are 5.5%, 8.1%, 11.0%. One sample per length/mode
does not establish a timing distribution: **no meaningful median/p95 or variability
estimate** is reported. The native GPU timings vary despite identical inputs; no
model-compute acceleration is claimed. Small first-request changes are not evidence
of robust freshness. Comparison image reads were performed after dispatch and are
included in end-to-end timing; the raw `post_dispatch_equivalence_check_s` field starts
at submit, so includes preparation too and is not an independent stage to sum.

### Freshness, queues, memory and behavior

Fast replay has no capture-to-current-host mapping. Both runs recorded one accepted
and two stale responses under their elapsed-delivery/unchanged-age gates, no
superseded responses and no history pauses. Those accepted replay results are **not
measured fresh live proposals**.

For a separately labeled counterfactual, add each original recorded estimated
exposure-to-submit offset to the newly measured submit-to-display interval:

| Frames | Baseline estimated age | Cached estimated age | Under unchanged 3 s limit |
|---|---:|---:|---|
| 5 | 3.064 s | 2.902 s | Cached only, one sample |
| 17 | 4.533 s | 4.180 s | Neither |
| 37 | 8.430 s | 7.513 s | Neither |

This retains the historical 48 ms camera-latency estimate and its 16 ms uncertainty;
it is not re-measured exposure or live latency. No images/proposals were restamped.
Steady-state freshness remains unresolved. One in-flight request and one pending flag
were retained; each pass coalesced 63 scheduling notices. There was no continuous
receiver in this profile, so queue-overflow/drop-rate performance is **not measured**.

Cache: 51 misses / 8 hits across 59 selected-frame uses, maximum 51 entries /
141,271,632 bytes of prepared PNGs. No cache eviction in these short jobs; eviction,
byte limits and resets are tested synthetically. Writer: three submitted/completed,
zero lost/outstanding/errors, maximum owned tensor bytes **24,787,719**.

Child lifetime peak RSS (including startup): **11,868,241,920 / 11,805,310,976 bytes**.
Maximum sampled MPS driver allocations: **11,223,990,272 / 11,228,184,576 bytes**.
These are baseline/cached; MPS samples are lower bounds and are not additive with
RSS on unified memory. Parent CPU peak RSS was not instrumented. The cache is on disk;
the writer adds one bounded owned-array copy, not a second model or GPU runtime.

All three pairs passed exact comparisons of selected frames/timestamps, rectified
RGB pixels, official processed tensors (including BF16 bit patterns), tensor types,
native sample/history metadata, saved model-space images, generated token IDs,
native pointing, decoded waypoints and STOP flags. Outputs also match Claude's
original three live outputs. Native STOP/completion remain false. Earlier operator
target review is historical; no new navigation-success or semantic benchmark is claimed.

## Tests and remaining limitations

New `tests/test_vln_latency.py` covers cache invalidation (pixels/calibration/time/
epoch/version), tampering, eviction and byte bounds, input equivalence under changing
latest-anchored sampling, latest-frame/anchor preservation, resets/cancellation,
unchanged staleness, array ownership, writer backpressure/errors/drain, final archive
error reporting, and absence of robot/network imports in the profiling paths.

Initial targeted run: one invalid test fixture supplied nonzero distortion while
claiming already-rectified input; the existing validator correctly rejected it.
The fixture was corrected to raw input. The original failure log is retained.
Full core regression initially passed **682 tests, 0 failures, 9 skips in 58.69 s**.
A final regression after the added last-archive-error propagation check passed
**683 tests, 0 failures, 9 skips in 42.29 s**, including 20 latency tests. The separate native/numeric suite passed **17, 0 failures, 2 skips
in 2.62 s**. Core skips: one physical Go2 test, six isolated native-runtime tests,
two module-level MuJoCo skips. Native/numeric cases passed separately; two rendering
tests stayed disabled. Hardware test selection was inspected before execution;
`GO2_HARDWARE=0` and `MUJOCO_RENDER_TEST=0` throughout.

Remaining limits: generation/vision still grows with intended history; low cache
overlap limits preparation savings. No incremental history, feature cache, changed
history window, larger age limit or new checkpoint was introduced. The opt-in path
needs Claude's separately coordinated stationary live test to measure actual receiver
backpressure and age. Real `T_base_camera`, exposure calibration uncertainty, and
near-goal stopping limitations remain. This work authorizes no physical execution.

## Reproduction and local artifacts

Private root:
`/Users/yushan/Desktop/Go2_Navigation/data/outputs/vln_live_latency/profile_20261116_01/`.
`plan.json` predates inference. `baseline/` and `cached/` contain declarations/source
hashes, saved observations, requests/native tensors/images, model identity/load and
memory, per-stage profiles, HTML review and preservation checks. `comparison.json`
contains all six outcomes and bit-exact comparisons. Logs/XML:
`full_regression_final.*`, `optional_regression.*`; initial failures/results retained.
No private imagery or tensors are published.

`reconstructed_baseline_source/` (8 files) and `reconstructed_cached_source/`
(10 files) preserve the exact profiled source. They were reconstructed after the
runs from the verified base plus instrumentation; **every byte hash matches its
pre-inference declaration**, as recorded in each `PROVENANCE.json`. Overlaying
these files on a separate checkout of `eaec331` reproduces the respective source
state. The final branch also includes the subsequently tested shutdown-error
reporting fix; no model call was repeated for that fix.

Offline reproduction (no robot access; choose a fresh directory; sequential only):

```bash
(
set -e
cd /Users/yushan/Desktop/Go2_Navigation/data/outputs/vln-live-latency-worktree
CORE=/Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6c/core-env/bin/python
MODEL=/Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6c/model-env/bin/python
ASSETS=/Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6c/lightnav
SOURCE=/Users/yushan/Desktop/Go2_Navigation/data/outputs/vln_live/live_session_03
OUT=/Users/yushan/Desktop/Go2_Navigation/data/outputs/vln_live_latency/my_new_profile
mkdir "$OUT"
export GO2_HARDWARE=0 MUJOCO_RENDER_TEST=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=src
unset LIGHTNAV_ASSETS SMOLVLM_MODEL_CACHE
for MODE in baseline cached; do
  "$CORE" -u scripts/profile_vln_latency.py --source-run "$SOURCE" \
    --output "$OUT/$MODE" --mode "$MODE" --model-python "$MODEL" --assets "$ASSETS" \
    > "$OUT/$MODE.log" 2>&1
done
"$CORE" scripts/summarize_vln_latency.py --root "$OUT" --output "$OUT/comparison.json"
SMOLVLM_MODEL_CACHE=/Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6b/hf_cache \
  "$CORE" -m pytest -q -rs
LIGHTNAV_ASSETS="$ASSETS" "$MODEL" -m pytest -q -rs \
  tests/test_lightnav_runtime_integration.py tests/test_waypoint_native_integration.py \
  tests/test_vln_simulation.py tests/test_waypoint_simulation.py
)
```

The existing ignored `.venv` link to core-env is used by older subprocess tests; no
environment was changed. The baseline and cached passes retain their exact source
hashes; final code additionally reports incomplete evidence during shutdown. This
does not change native inputs or the successful profiling paths. MPS/ffmpeg need
the existing unrestricted local execution context, not new dependencies.

### Claude's next stationary read-only live test — prepared, NOT executed here

After separate operator coordination and confirmation of stationary Go2/cleanup:

```bash
cd /Users/yushan/Desktop/Go2_Navigation/data/outputs/vln-live-latency-worktree
env -u LIGHTNAV_ASSETS -u SMOLVLM_MODEL_CACHE GO2_HARDWARE=0 MUJOCO_RENDER_TEST=0 \
  HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=src \
  /Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6c/core-env/bin/python -u scripts/run_vln_shadow.py \
  --go2-camera --hardware-python /Users/yushan/Desktop/Go2_Navigation/.venv-go2/bin/python \
  --instruction 'Approach the doorway and stop before entering.' --backend lightnav \
  --model-python /Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6c/model-env/bin/python \
  --assets /Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6c/lightnav \
  --runtime-config configs/pretrained_vln.yaml --device mps --preparation-mode cached \
  --advisory-age-s 3 --max-requests 3 --max-observations 600 --history-seconds 20 \
  --max-host-seconds 180 \
  --output /Users/yushan/Desktop/Go2_Navigation/data/outputs/vln_live/latency_stationary_NEW
```

Use a new output path. Inspect `index.html`, `latency.jsonl`, each preparation profile,
native output and `tensor_evidence.json`, plus final writer status. Record every
request, drops/gaps, queue pauses and uncertainty. Keep stale results stale. No
motion option or connection to the hardware-control utility is added by this flag.
