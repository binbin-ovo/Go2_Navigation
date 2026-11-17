# Context regression isolation — completed offline experiment (2026-11-17)

**All 25 frozen Stage B requests ran once, sequentially, and decoded successfully.**
The operator confirmed that the live demo/physical tests were paused, Go2 was
stationary and cleanup was complete. Inference finished and the worker closed
before regression tests began. No Go2 connection, DDS initialization, control
client or physical command was used. Stage A was completed on November 16;
the newly authorized Stage B window was November 17.

**Result: reject the tested bounded corrections for operational promotion.**
Restoring pooling recovers the same-eight-image control, but E1 still adds an
initial right turn and misses STOP, and E2 still predicts the visible rack as
not visible. The E3 approach changes remain human-review pending. None of the
25 outputs produces native STOP or a completion claim. The old favorable doorway
proposal is preserved exactly. Faster offline delivery does not resolve these
behavior limitations or establish live freshness/navigation success.

## Identity and isolation

- Branch: `feature/vln-context-ablation`.
- Exact base: `d3c227a112436d1b93c2d9b67bb69d5b89288295`.
- Implemented/tested code: `956bbdf25088ec437e84a7deaec9312c8804ec8f`.
- Stage A publication used for execution: `c13d24b575b5fb9a5c65eea1d48c546649b025b9`.
  The inference code/configuration still matches every frozen hash. Only an
  evaluation-only results auditor, its tests and documentation were added after
  execution; they did not construct or influence any model request.
- Worktree: `<repository-root>/data/outputs/vln-context-ablation-worktree`.
- Read the [completed bounded experiment](VLN_BOUNDED_CONTEXT_EXPERIMENT.md),
  current handoffs and [Claude's pinned human review](https://github.com/binbin-ovo/Go2_Navigation/blob/bc8815319da554c6a3dee2523da9f2a72cf352aa/docs/VLN_BOUNDED_CONTEXT_REVIEW.md).
  No duplicate ablation branch/implementation existed when inspected.
- The existing bounded/full builders, model worker, calibration, operational
  configuration and Claude's interactive worktree are unchanged. Main remains
  `dc0f85b73433a49e57d73ff91b72e9309af862e3`; no merge or live-policy promotion.

Private experiment root:
`<repository-root>/data/outputs/vln_context_ablation/<run-id>`.
The exact local root, declaration time and `plan.json` SHA256 are recorded in the
Git-ignored `stage_a/LOCAL_REPRODUCTION.md` beside the run's artifacts.
It binds code/configuration hashes, the previous plan, actual native metadata and
**2,270** original source/result files. All 2,270 remain unchanged after Stage B
and regression. The Stage A input gate still passes.
The previous 34-request experiment and all packages/annotations remain intact.
Private imagery, prompts, tensors and panels are Git-ignored; only code, tests
and this aggregate report are published.

## Stage B findings and equivalence

All **25/25** runtime input audits pass: exact `model.generate` tensor bits and
types match Stage A; calibrated-file hashes, capture/availability timestamps,
instruction/episode/epoch/cutoff, canonical native metadata and normalized-image
evidence also match. All **10/10** repeated F/B requests reproduce the original
input tensors, generated tokens and complete decoded outputs. There were no
inference failures, retries, extra requests or source changes. All 25 background
tensor archives completed, with no outstanding archive at shutdown.

**E3 d0 TP = F** through calibrated pixels, normalized/processor tensors, temporal
positions, prompt tokens, native groups, post-ViT configuration, actual generation
inputs, generated tokens and complete native output. P also produces F's complete
output here despite different input metadata; T produces B's output. This supports
pooling as a sufficient correction for this particular same-image control.
Missing intermediate images cannot explain that control's original difference.
It does not establish that pooling explains the other regressions.

The expected no-op controls also reproduce exactly: **E2 d2 and doorway P=B,
TP=T**, including actual inputs and generated outputs. T changes actual inputs
in all five cases, so its unchanged outputs below are not a construction no-op.
Generated suffix tokens are compared separately from full token sequences
containing the prompt; equal output text is not called equal input.

| Case | F | B | T | P | TP |
|---|---|---|---|---|---|
| E1 d2 | Reproduces old F; misses STOP | Initial right yaw −6.09°; misses STOP | Exact B output | Exact B output | Initial right yaw −6.37°; misses STOP |
| E2 d2 | Reproduces old visible-rack target prediction | Target not visible | Exact B output | Exact B output | Exact B output |
| E3 d2 | Forward endpoint 0.685 m; final yaw +1.94° | 0.121 m; −13.53° | 0.169 m; −13.96° | 0.751 m; −28.84° | 0.946 m; −34.51° |
| E3 d0 | Straight 1.505 m chunk, nearly zero heading | 0.946 m; final yaw −34.51° | Exact B output | Exact F output; inputs differ | Exact F inputs and output |
| Doorway | Reproduces old straight proposal | Reproduces previously preferred bounded proposal | Exact B output | Exact B output | Exact B output |

Distances/headings above summarize **predicted** absolute ego-frame chunks, not
executed motion. All ten positions and headings remain in the private viewer and
machine-readable report. For E3 d2, P/TP increase forward displacement but retain
substantial heading changes; TP's initial yaw is near zero while its final yaw is
−34.51°. A favorable first-waypoint mapping is insufficient to call this a plausible
complete approach. All new outputs remain **human-review pending**, including
repeated outputs identical to historical ones. The old review is displayed
separately and is not silently transferred into new human labels.

No condition recovers E1's stopping requirement or E2's target prediction. In E2,
pooling was already unchanged and timestamp restoration is insufficient; remaining
history/representation differences are implicated, but this experiment does not
identify missing history alone as the cause. TP's grouping/pair confounds below
still apply to the longer histories.

Native decoding validity is **25/25**; native STOP, explicit STOP code and APOS1299
completion claims are each **0/25**. Every condition misses both STOP-only human
references (E1 d2/E2 d2). Secondary, lossy first-waypoint agreement against the
four unchanged reviewed references is F **2/4**, B **1/4**, T **1/4**, P **1/4**,
TP **2/4**. The doorway has no action reference. These correlated development
counts do not override complete-path review or constitute an accuracy estimate.

## Stage B timing and memory

Actual runtime: Apple M3 Pro Mac, existing isolated Python 3.11.13,
Torch 2.10.0/Transformers 5.8.0, MPS BF16/SDPA, no quantization, cached fixed
checkpoint, preserved CPU pooling fallback. The sandbox hides Metal; the approved
run used normal host Metal access with hardware/rendering opt-ins disabled.
One model load: **43.444 s**. Whole run including load/shutdown: **122.768 s**.
Only E1 d2 F (request 0) was cold-after-load. No other model job ran concurrently.

Offline request-to-review-file seconds, one observation per cell:

| Case | F | B | T | P | TP |
|---|---:|---:|---:|---:|---:|
| E1 d2 | 6.839* | 2.231 | 2.092 | 2.246 | 2.166 |
| E2 d2 | 7.625 | 2.163 | 2.131 | 2.044 | 2.070 |
| E3 d2 | 5.412 | 2.503 | 2.215 | 2.195 | 2.170 |
| E3 d0 | 2.099 | 2.149 | 2.091 | 2.217 | 2.170 |
| Doorway | 7.742 | 2.093 | 2.199 | 2.111 | 2.144 |

\* Cold-after-load; loading itself is excluded from request durations. There are
no repeated samples here and no robust percentile claim. The eight-image full
control has essentially the same cost as bounded input; do not infer a benefit
from tiny differences between those single observations.

Reconstructed offline ages are **2.161–2.658 s** for all 20 bounded-condition
requests (20/20 below the unchanged 3.0 s threshold). F is below it in 1/5 requests:
ages in case order are 6.950, 7.742, 5.568, 2.254, 7.877 s. These use the original
capture-to-submit offsets plus current processing time; they are not measurements
of a new live camera session.

Per-request preparation, official processor, vision/pooling, native generation,
worker, response-delivery, review-write, total/age and sampled memory are retained
for all 25 rows in `analysis_stage_b.json`, with full original profiles in each
`native/result.json`. Doorway F versus TP illustrates the measured breakdown:

| Component | F | TP |
|---|---:|---:|
| Calibrated preparation + job persistence | 1.258 s | 0.263 s |
| Official processor | 0.016 s | 0.004 s |
| Vision + pooling | 3.420 s | 0.717 s |
| Native generation, including vision | 5.749 s | 1.725 s |
| Entire worker request | 6.410 s | 1.864 s |
| Response delivery | 4.84 ms | 3.43 ms |
| Review-file write | 4.78 ms | 1.20 ms |
| Total | 7.742 s | 2.144 s |

These intervals are nested, not additive; total also includes pipe/validation
overhead outside the worker interval. Model generation remains the main cost.
Peak child lifetime RSS was **11.804 GB** (load sampled RSS 11.776 GB); maximum
sampled MPS allocation **9.804 GB**, driver allocation **11.195 GB**. Parent peak
RSS was 208.929 MB. The archive writer owned at most 26.165 MB and one outstanding
archive. Unified-memory figures are **not additive**, GPU peaks are sampled lower
bounds, and model caches persist across conditions. This is not a measured
independent memory budget for each condition.

## Stage B regression and review artifacts

Tests ran after the inference worker exited, sequentially in the existing isolated
environments. Selection was inspected before execution. `GO2_HARDWARE=0` and
`MUJOCO_RENDER_TEST=0` throughout. The full core run included the cached SmolVLM
processor/model regressions on synthetic fixtures; no download or shared-environment
modification occurred.

| Actual invocation | Result |
|---|---|
| Full core suite | 737 passed, 1 launcher failure, 26 skipped; 69.00 s |
| Focused recheck after local launcher fix | 1 passed; 0.34 s |
| Isolated native/numeric suite, including MPS pooling | 34 passed, 2 OpenGL rendering skips; 3.82 s |
| New evaluation-auditor tests | 5 passed; 0.21 s |

**777 distinct tests passed across these sequential invocations; no unresolved
test failure.** The original failing test was
`test_fsd_viewer_refuses_tracked_output_for_recordings`: it launches
`.venv/bin/python` if available, otherwise an absent bare `python`. An ignored
worktree-local `.venv` symlink to the already existing core environment resolved
the launcher; the actual privacy assertion then passed. No test assertion,
application or environment package was changed. The full suite was not rerun
after this isolated fix; its original log/failure and passing focused recheck
are both preserved.

The core's 26 skips were one physical Go2 test, 23 optional native tests and two
module-level MuJoCo availability skips. The isolated suite exercised those native
and numeric simulator cases; the remaining intentional test skips are the
physical Go2 test and two OpenGL rendering tests. No physical client was initialized.

Private artifacts, relative to the frozen experiment root:

- `results/rows.json`, `results/summary.json`, `results/worker/model.json`,
  `stage_b_run.log`: all requests, load/runtime, shutdown/archive completion.
- `results/<number>_<case>_<condition>/`: complete native outputs, all ten
  waypoints/yaws, images, source/timing/memory evidence and tensor archives.
- `index_stage_b.html`, `review_stage_b.json`: all 25 panels using the existing
  review tools; actual history, pointing/path, latency/age and separate unchanged
  evaluation-only references. All new human-review statuses remain pending.
- `analysis_stage_b.json`: all exact input/output comparisons, native diagnostics,
  secondary reference scores and per-request latency/memory breakdowns. Produced
  only after inference by `scripts/summarize_vln_context_results.py`.
- `verification_stage_b/`: full collection logs/JUnit results, launcher recheck,
  native/numeric and new auditor tests, final source/input preservation check.
  Exact local paths and executed commands are in `LOCAL_STAGE_B_HANDOFF.md` there.

All 25 viewer panels and 346 links pass static checks. Browser appearance remains
unverified because the tool's local-file URL policy blocked preview in Stage A;
no workaround was attempted. No private media, raw outputs, paths or run identifiers
are added to public Git history.

## Prior evidence and question

Claude's review concerned the **old** outputs: E1 d2's extra bounded turn/path was
inappropriate, E2 d2 lost the visible rack target, and E3 d2's complete bounded
path was rejected while full was acceptable. The old bounded doorway proposal
was preferred over full. That last comparison did not separately validate target
pointing/path or establish navigation success. The old full baseline also missed
STOP references. These judgements are not transferred to new outputs.

One timing correction to the review narrative: the old 39-frame full request
was warm (request 4). Only request 0, the five-frame full job, was cold-after-load.
This follows the original `results/rows.json`; it does not alter Claude's human
judgements or historical timings/scores.

The experiment asks whether timestamp encoding, spatial pooling or omitted
history/remaining representation differences accompany the observed regressions.
E1–E3 are correlated retrospective proposal cases from `test_c`, and the doorway
is one recorded cutoff. These are development diagnostics, not held-out trials.

## Frozen conditions and native representation audit

All conditions retain the checkpoint, processor, native normalization, calibrated
pixels, instruction, cutoff and every-decision observation-only reset. No action
feedback or generated proposal enters another request. Selection is causal by
both observation and availability time. B/T/P/TP use exactly the same first two
plus latest six observations, at most eight unique images/eight native slots.

| Condition | Actual change |
|---|---|
| F | Original full-history `run_job` path, no custom sample factory. |
| B | Calls the unchanged `bounded_context.sample_factory`: actual millisecond ticks, native validated bounded tiers, original B groups/pools. |
| T | B with original logical source IDs, original `total_frames`, and 4 Hz native-grid metadata. B grouping, pixels, pools and prompt template remain fixed. |
| P | B with each retained image's pool taken from the **saved full native metadata**. B groups, temporal pairs and millisecond encoding remain fixed. Mixed baseline pool assignments within a B segment fail closed rather than silently splitting it. |
| TP | T plus original pools **and the filtered full segment partition**, including the official prompt's segment placeholders. Empty segments are dropped; original padding is removed before filtering, then native even-length padding is reapplied if needed. |

T uses logical index zero as the original grid origin, not a compressed eight-frame
timeline. The 4 Hz grid is the original policy's **approximation**, not a corrected
claim about measured capture intervals. Actual capture/availability timestamps
and original source IDs remain in `job.json`/`preparation.json`. B/P use 1,000 Hz
integer **timestamp ticks**, not a 1,000 fps camera. All five T inputs differ from B
at the actual model-input level; rounded timestamp prefixes change, e.g. recent
history's `<1.2 seconds>`/`<0.7 seconds>` becomes `<1.1 seconds>`/`<0.6 seconds>`.

TP is **not a clean two-factor interaction**: it additionally restores segment
grouping and prompt structure. This is necessary to make the same-image control
equal F. The original pinned tier selector is still verified against the saved
full metadata before construction. No native validator is bypassed, pool factors
are not guessed, and omitted images are never fabricated.

| Case | Full candidates / native unique / native slots | B selected source IDs | Full segment pools | B → P pools | TP groups |
|---|---|---|---|---|---|
| E1 d2 | 29 / 29 / 30 | 0,1,23,24,25,26,27,28 | 2,1 | 4,2,1 → 2,2,1 | 6+2 |
| E2 d2 | 36 / 36 / 36 | 0,1,30,31,32,33,34,35 | 4,2,1 | unchanged 4,2,1 | 2+4+2 |
| E3 d2 | 25 / 25 / 26 | 0,1,19,20,21,22,23,24 | 2,1 | 4,2,1 → 2,2,1 | 6+2 |
| E3 d0 | 8 / 8 / 8 | 0,1,2,3,4,5,6,7 | 2,1 | 4,2,1 → 2,2,1 | 6+2 |
| Doorway | 39 / 38 / 38 | 0,1,33,34,35,36,37,38 | 4,2,2,1 | unchanged 4,2,1 | 2+4+2 |

All B groups are 2+4+2. In E1 d2/E3 d2, filtering a full segment changes temporal
pairing when the original partner was omitted. For example, E1's full pairs
include (22,23), (24,25), (26,26 padding); TP has (23,24), (25,26).
Retaining the missing partners would violate the declared image selection/budget.
The audit records changed pairs explicitly. Thus F versus TP on longer histories
does not isolate image omission from all representation effects. E3 d0 has no
omitted images and cannot be explained by missing history.

The checkpoint's HF engine drops processor `position_ids` and computes its own
stock-Qwen3-VL mRoPE from the grid. Stage A compares both the processor's full
tensors (including positions) and the exact HF generation inputs, plus the
separately consumed post-ViT grid/pooling state. BF16 values are compared as exact
uint16 bits, not lossy FP16 conversions. Equal token counts alone are insufficient.

## Frozen Stage B matrix and executed runtime

Exactly 25 requests, one model loaded, sequential, no retries or additional budgets:

| Request numbers (zero-based) | Case | Order |
|---|---|---|
| 0–4 | E1 d2 completed counter turn | F, B, T, P, TP |
| 5–9 | E2 d2 visible shoe rack | B, T, P, TP, F |
| 10–14 | E3 d2 bottle approach | T, P, TP, F, B |
| 15–19 | E3 d0 same-eight-image control | P, TP, F, B, T |
| 20–24 | Saved 39-frame doorway cutoff | TP, F, B, T, P |

Fixed model `LightOriginsHQ/LightNav-0`; checkpoint/processor/RVQ
`826dc5fbfa37afa8293d2e336d329b6ffc0bfb64`; official source
`3015508b70fb6e30bbd66701e94c9b916dcb8df8`; preserved source-local-v2 identity
SHA256 `bec34b7d855f33ac94ec212a61ca507d2d677d82441add0cd1b2dfe7088e554c`.
Executed Mac runtime: MPS BF16, SDPA, no quantization, seed 0, greedy,
64 new tokens, unchanged same-dtype CPU pooling fallback. No model/runtime patch
or weight download occurred in this task.

Budgets: load 150 s, per request 45 s, total host 600 s including load. Expiry
stops the experiment, records the failure/unattempted requests, and does not retry.
Only request 0 is marked cold-after-load. Each condition has its own fresh prepared
frame cache, with the same existing settings. Model caches persist as in the
existing worker; one observation per cell does not support robust percentiles.

Record preparation, native processor/vision/generation, delivery, review-file
write, observation age and sampled memory separately. The advisory threshold stays
**3.0 s**. Reconstructed offline ages are not live camera-to-display measurements.
Native target/affordance tokens, all ten waypoints/yaws, native decoder STOP,
explicit STOP code and APOS1299 completion claims remain separate. Historical human
references appear only in the evaluation viewer. New outputs, including newly
repeated F/B, remain human-review pending. Human review should answer the five
case questions in the request using complete paths/pointing, not only the existing
lossy first-waypoint action mapping. No operational promotion follows from timing
alone or matching the full baseline.

## Stage A measured results

All **25** native input constructions completed on CPU with two Torch threads and
no model/decoder weights or neural forward pass. Existing isolated model environment:
Python 3.11.13, Torch 2.10.0, Transformers 5.8.0, NumPy 2.4.6, Pillow 13.0.0.
Preparation/mocked tests used the existing core environment (Python 3.12.11,
NumPy 1.26.4, OpenCV 4.11.0.86, pytest 8.3.5). No shared environment changed.

- **10/10** F/B HF generation-input archives exactly match the previously saved
  corresponding original inputs (generated IDs excluded; no new IDs exist yet).
- **25/25** preserve original calibrated pixels, instruction, episode/epoch,
  cutoff, capture timestamps and availability timestamps.
- Shared normalized-image tensors match F in every B/T/P/TP case.
- **E3 d0 TP = F exactly**: calibrated pixels, normalized tensors, every processor
  tensor, temporal/positional inputs, prompt tokens, canonical segment metadata,
  HF generation tensors and post-ViT state. Generated-token/full-output equivalence
  was not established by Stage A alone; the Stage B checks above now establish it.
- **E2 d2 and doorway: P = B, TP = T** at the actual HF-input/post-ViT level.
  Their pooling was already unchanged; these intentional no-op controls can expose
  runtime nondeterminism later. Pooling-only restoration cannot independently
  recover E2 target loss when it changes no model input.
- T versus B changes actual input tensors in all five cases; longer-history TP
  differs from F as expected. Equal token lengths do not imply equal inputs.

Actual HF input token counts (generation batch size 1):

| Case | F | B | T | P | TP |
|---|---:|---:|---:|---:|---:|
| E1 d2 | 699 | 283 | 283 | 303 | 303 |
| E2 d2 | 789 | 285 | 285 | 285 | 285 |
| E3 d2 | 631 | 287 | 287 | 307 | 307 |
| E3 d0 | 307 | 287 | 287 | 307 | 307 |
| Doorway | 825 | 285 | 285 | 285 | 285 |

Evidence under the experiment root:

- `stage_a/prepared_jobs.json`, `prepared_manifest.json`: all closed input jobs,
  source provenance and plan binding; no evaluation labels in jobs.
- `stage_a/input_checks.json`: passing gate, all tensor signatures and case checks.
- Each numbered Stage A folder: calibrated images, native normalized images,
  `native/native_sample.json`, `native/prompt.txt`, `native/generation_inputs.npz`,
  `construction.json`. No native model outputs are fabricated.
- `index_stage_a.html`, `review_stage_a.json`: five-condition comparison using the
  existing pointing/path helpers, expandable native history/grouping/timing panels,
  pending new output panels and separate historical-output/human-reference panels.
- `stage_a/final_preservation.json`: 2,270 original files unchanged; input gate
  passed; 25 HTML panels and 346 local links verified, none missing.

The CPU audit took 24.06 s wall time according to `/usr/bin/time`; this is **input
construction**, not inference latency. Its external `time -l` wrapper exited 1
because the sandbox denied `sysctl kern.clockrate` after Python completed; the
saved 25-case audit and subsequent independent gate both passed. No peak-memory
measurement is claimed. The browser tool rejected `file:` preview by URL security
policy; no workaround was attempted. HTML/links were checked statically, but
browser appearance remains unverified.

## Stage A tests (historical)

Selection was inspected with `--collect-only` before execution. Both suites ran
sequentially with `GO2_HARDWARE=0`, `MUJOCO_RENDER_TEST=0`, plugin autoload disabled;
native checks were offline CPU-only. No physical, renderer, MPS-kernel or neural
integration test was selected.

| Suite | Result |
|---|---|
| `test_vln_context_ablation.py` + existing `test_vln_bounded_context.py` | **55 passed, 0 failed, 0 skipped**, 0.30 s |
| `test_context_native_integration.py` + existing `test_bounded_native_integration.py` | **17 passed, 0 failed, 0 skipped**, 3.88 s |
| Full regression and 25 learned requests | Subsequently executed in Stage B; see measured results above |

The first lightweight run had one test-fixture setup failure (a mocked agent was
`None`); correcting that fixture yielded the passing result above. No model result
was inspected or selected to tune inputs. Existing full-suite counts in older
documents describe their historical branches, not new validation of this branch.

## Reproduction and continuation

Use the existing environments and cached assets; do not install into the sensor
or calibration environment. All output directories are immutable: use a **new**
root for an independent reproduction, never overwrite the published declaration.
The commands below use portable path aliases. Set the repository/run locations
from the private local handoff; no user-specific filesystem paths are published.

```bash
GO2_CTX_REPO=/path/to/Go2_Navigation
cd "$GO2_CTX_REPO/data/outputs/vln-context-ablation-worktree"
export PYTHONPATH="$PWD/src:$PWD/scripts"
export GO2_HARDWARE=0 MUJOCO_RENDER_TEST=0 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
GO2_CTX_CORE="$GO2_CTX_REPO/data/outputs/phase6c/core-env/bin/python"
GO2_CTX_MODEL="$GO2_CTX_REPO/data/outputs/phase6c/model-env/bin/python"
GO2_CTX_ASSETS="$GO2_CTX_REPO/data/outputs/phase6c/lightnav"
GO2_CTX_PREVIOUS="$GO2_CTX_REPO/data/outputs/vln_bounded_context/<previous-run-id>"
GO2_CTX_RUN="$GO2_CTX_REPO/data/outputs/vln_context_ablation/<run-id>"

# Actual lightweight suites (also ran these with --collect-only first):
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$GO2_CTX_CORE" -m pytest -q \
  tests/test_vln_context_ablation.py tests/test_vln_bounded_context.py
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 LIGHTNAV_ASSETS="$GO2_CTX_ASSETS" \
  "$GO2_CTX_MODEL" -m pytest -q \
  tests/test_context_native_integration.py tests/test_bounded_native_integration.py

# Commands used to create Stage A. These refuse the already existing output.
"$GO2_CTX_CORE" scripts/run_vln_context_ablation.py --declare-only \
  --previous-run "$GO2_CTX_PREVIOUS" --output "$GO2_CTX_RUN"
"$GO2_CTX_CORE" scripts/run_vln_context_ablation.py --prepare-inputs --output "$GO2_CTX_RUN"
"$GO2_CTX_MODEL" scripts/check_vln_context_inputs.py --assets "$GO2_CTX_ASSETS" --run "$GO2_CTX_RUN"
"$GO2_CTX_CORE" scripts/review_vln_context_ablation.py --inputs-only --run "$GO2_CTX_RUN"
```

The following records the authorized Stage B execution. It is **complete**, not
an instruction to repeat it: the original `results/` exists and the runner refuses
to overwrite it. Any independent reproduction requires a new declaration/output
and separately agreed resource window. The frozen plan retains its historical
Stage A status string; `results/summary.json` and this report record completion.
The commands reverify code/source hashes and the input gate before loading one
model. No competing heavy job is permitted.

```bash
"$GO2_CTX_CORE" scripts/run_vln_context_ablation.py --run \
  --resource-window-confirmed --output "$GO2_CTX_RUN" \
  --model-python "$GO2_CTX_MODEL" --assets "$GO2_CTX_ASSETS"
"$GO2_CTX_CORE" scripts/review_vln_context_ablation.py --run "$GO2_CTX_RUN"

# After worker shutdown: collect then run the complete core suite, using cached SmolVLM.
GO2_CTX_SMOL_CACHE="$GO2_CTX_REPO/data/outputs/phase6b/hf_cache"
env -u LIGHTNAV_ASSETS PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 SMOLVLM_MODEL_CACHE="$GO2_CTX_SMOL_CACHE" \
  "$GO2_CTX_CORE" -m pytest --collect-only -q
env -u LIGHTNAV_ASSETS PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 SMOLVLM_MODEL_CACHE="$GO2_CTX_SMOL_CACHE" \
  "$GO2_CTX_CORE" -m pytest -q -rs
# If .venv is absent: this worktree-local alias fixes the legacy subprocess launcher.
ln -s "$GO2_CTX_REPO/data/outputs/phase6c/core-env" .venv
"$GO2_CTX_CORE" -m pytest -q tests/test_fsd_viewer.py::test_fsd_viewer_refuses_tracked_output_for_recordings

# Then isolated native/numeric checks. Rendering and hardware flags remain zero.
env -u SMOLVLM_MODEL_CACHE PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 LIGHTNAV_ASSETS="$GO2_CTX_ASSETS" \
  "$GO2_CTX_MODEL" -m pytest -q -rs tests/test_context_native_integration.py \
  tests/test_bounded_native_integration.py tests/test_lightnav_runtime_integration.py \
  tests/test_waypoint_native_integration.py tests/test_vln_simulation.py tests/test_waypoint_simulation.py

# Evaluation-only audit added after all 25 requests; no new neural inference.
"$GO2_CTX_CORE" -m pytest -q tests/test_context_result_audit.py
"$GO2_CTX_CORE" scripts/summarize_vln_context_results.py --run "$GO2_CTX_RUN"
```

Retain any `results/failure.json`, partial native evidence and
`results/summary.json` unattempted list. Do not retry selectively. The Stage B
viewer is `index_stage_b.html`; it preserves complete outputs and checks E3 d0
TP/F generated IDs plus full decoded output equality, and the two no-op controls.
If inference/decoding fails, report it rather than inferring equality from absent
fields. New human review remains required; do not change old annotation packages.

**Decision after Stage B:** reject these bounded corrections for operational
promotion. Keep the existing full-history advisory default, with its documented
staleness and stopping limitations. The control verifies input construction and a
specific pooling effect; it does not recover E1/E2 or validate E3's complete approach.
New human review is pending. The declared experiment is finished; do not run another
model/context budget, live robot test or training loop, change Claude's demo,
connect proposals to movement, or merge a policy into operational defaults.
