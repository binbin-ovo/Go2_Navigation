# Read-only advisory verification — 2026-11-15

Branch `feature/vln-shadow-demo`. **Tested implementation:**
`b3c3dffdaf1a663f554b29d8dc2ca19aa1460e8d`. This follow-up changes documentation only;
application code, model, calibration, operational defaults and main are unchanged.

The operator explicitly confirmed that physical motion testing was paused/finished,
the Go2 was stationary, and stop/release cleanup was complete. Authorization covered
one local LightNav replay and offline regressions, sequentially. No Go2 connection,
DDS initialization, command ownership, control utility, or robot command was used.
No environment/package/weight installation was performed.

## Actual LightNav smoke result

One request, no retry or threshold adjustment. The application received the first
24 existing E1 observations, selected five actual causal frames on its native 4 Hz
input grid, and ran the pinned official processor/decoder. All five calibrated
prepared images exactly match the existing `calibrated_view` output (zero pixel
difference). All native-selected indices 0–4 have saved 448×256 model-space images.
Exposure and availability precede the snapshot cutoff. No annotations, pose,
LiDAR, action reference, future frame or proposed-action feedback entered the job.

**Inference succeeded; the advisory was correctly rejected as stale.** Native output
decoded to ten finite absolute ego SE(2) waypoints. Native STOP was false; APOS
completion was false; goal completion remains unverified. This is pipeline evidence,
not an instruction-following accuracy, navigation-success or physical-motion test.
The E1 package remains a retrospective proposal episode from correlated `test_c`.
No human references were loaded or historical scores changed.

| Measurement | Actual result |
|---|---:|
| Native generation | 2.444501 s |
| Native engine call (including its preprocessing) | 2.586740 s |
| Complete child `run_job` interval, including evidence writes | 2.851785 s |
| Host submit → receipt | 3.142678 s |
| Paced-replay observation age at receipt/display | **3.544899 s** |
| Advisory age limit | 3.000000 s, unchanged |
| Rejection | `stale_advisory_age` |
| Saved final-page observation age | 3.571435 s |
| Source timestamp uncertainty | 0.016125 s, retained |
| Model load | 45.007259 s |
| Whole application, including verification/load | 57.192424 s |
| Lifetime peak process RSS | 11,720,278,016 bytes |
| Maximum sampled MPS driver allocation | 9,528,197,120 bytes |

MPS memory is sampled and may miss the true peak; it is not additive with RSS on
unified memory. The native worker's `replay_age_at_output_s=2.808510` excludes
parent-side input preparation/dispatch and is a **lower bound**, not full display
age. The complete generation-completion instant relative to capture was not saved;
do not relabel the measured receipt/display age as that instant. Terminal status
printed slightly later has a larger age. All these are **paced-replay diagnostic
times, not measured live Go2 latency**. No robot-clock/host-wall subtraction is used.

Runtime: macOS 26.6.2 arm64, Apple MPS, Python 3.11.13, Torch 2.10.0,
Transformers 5.8.0, NumPy 2.4.6; BF16, SDPA, no quantization. Checkpoint/processor/RVQ
`LightOriginsHQ/LightNav-0@826dc5fbfa37afa8293d2e336d329b6ffc0bfb64`; upstream
`3015508b70fb6e30bbd66701e94c9b916dcb8df8`; unchanged same-dtype CPU pooling patch
`bec34b7d855f33ac94ec212a61ca507d2d677d82441add0cd1b2dfe7088e554c`.
Asset hashes were verified by the existing worker before model loading.

## Test selection and results

Before execution, collection inspected all **655 core-environment test nodes**.
The sole hardware test had an active skip condition with `GO2_HARDWARE=0`; its
receiver/DDS imports are inside the skipped test body. No other physical-test opt-in
was found in this branch. The camera silence/rejoin test uses loopback interface
127.0.0.1 and a non-robot test port; it opens no Go2 connection. Claude's hardware
branch and control tests were not merged or imported.

The LightNav smoke exited before the core suite began. The optional native/numeric
suite ran only after the core suite exited. No competing heavy jobs were launched.

| Run | Passed | Failed | Skipped | Time |
|---|---:|---:|---:|---:|
| Full core regression, including cached SmolVLM tests | **648** | **0** | **9** | 44.73 s |
| Separate existing native-runtime/numeric-simulator regressions | **17** | **0** | **2** | 2.68 s |

Core skips: one physical Go2 test; six LightNav source/runtime tests intentionally
left out of the incompatible core runtime; two module-level skips because MuJoCo
is installed only in the isolated model environment. The six LightNav cases and
numeric simulator cases subsequently passed in that environment. Its two camera
rendering tests remained skipped because `MUJOCO_RENDER_TEST=0`; this does not claim
that rendering is unavailable on the Mac. Across environments, the remaining
unexecuted cases are the physical test and those two rendering tests.

The full core result includes all 57 advisory tests: bounded queues, immutable causal
history, observation/instruction provenance, epoch resets, delayed/stale/wrong-session
results, supersession, cancellation during generation, new instructions, invalid data,
native STOP versus completion, worker shutdown and absence of robot-control imports
or execution flags. Delay/cancellation/instruction-change checks use deterministic
mock workers; they were not extra real-model generations. The real call independently
exercised actual calibrated inference and stale rejection.

## Private artifacts and reproduction

All new artifacts are local and Git-ignored under:
`/Users/yushan/Desktop/Go2_Navigation/data/outputs/vln_shadow/verification_20261115_01/`.

- `real_model_smoke_01/index.html`: readable advisory, native path/history and timing.
- `real_model_smoke_01/advisory.png`: inspected private camera/input panel; not published.
- `real_model_smoke_01/requests/000001/`: immutable job/snapshot, response receipt,
  and `native/` outputs, images, exact tensors, timing/memory and prompt.
- `real_model_smoke_01/worker/model.json`: exact model/runtime identity and load metrics.
- `audit/`: authorization, preflight, test selection, logs/XML, input/output audit
  and metrics. Both test runs report zero failures.

All 107 E1 source images, calibration and manifest match the pre-run hashes and
existing package manifest. Other recordings/packages were not written. The input
audit also verifies causal fields, exact rectification, native input images, and
the HTML's instruction, native chunk, input timestamps and observation-age fields.
Private media and scene logs remain outside Git; no new private screenshot is published.

Exact invocations below retain the recorded `VERIFY` value. For reproduction,
change it to an unused directory. The enclosing shell stops at `mkdir` if the
directory already exists, before redirecting any logs or running inference.

```bash
(
set -e
cd /Users/yushan/Desktop/Go2_Navigation/data/outputs/vln-shadow-worktree
VERIFY=/Users/yushan/Desktop/Go2_Navigation/data/outputs/vln_shadow/verification_20261115_01
CORE=/Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6c/core-env/bin/python
MODEL=/Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6c/model-env/bin/python
ASSETS=/Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6c/lightnav
mkdir "$VERIFY"
mkdir "$VERIFY/audit"

env -u LIGHTNAV_ASSETS -u SMOLVLM_MODEL_CACHE \
  GO2_HARDWARE=0 MUJOCO_RENDER_TEST=0 HF_HUB_OFFLINE=1 \
  HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=src \
  "$CORE" scripts/run_vln_shadow.py \
  --episode /Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6b/episodes/p6b_test_c_e1_turn_right/model_input/episode.json \
  --instruction 'Turn right toward the kitchen counter.' \
  --backend lightnav --model-python "$MODEL" --assets "$ASSETS" \
  --runtime-config configs/pretrained_vln.yaml --device mps \
  --mode paced --max-observations 24 --max-requests 1 \
  --output "$VERIFY/real_model_smoke_01" > "$VERIFY/audit/real_model_smoke_01.log" 2>&1

# Only after the smoke and its child have exited:
env -u LIGHTNAV_ASSETS GO2_HARDWARE=0 MUJOCO_RENDER_TEST=0 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
  TOKENIZERS_PARALLELISM=false \
  SMOLVLM_MODEL_CACHE=/Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6b/hf_cache \
  PYTHONPATH=src "$CORE" -m pytest -q -rs \
  --junitxml="$VERIFY/audit/full_regression.xml" > "$VERIFY/audit/full_regression.log" 2>&1

# Only after core regressions have exited:
env -u SMOLVLM_MODEL_CACHE GO2_HARDWARE=0 MUJOCO_RENDER_TEST=0 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
  TOKENIZERS_PARALLELISM=false LIGHTNAV_ASSETS="$ASSETS" PYTHONPATH=src \
  "$MODEL" -m pytest -q -rs tests/test_lightnav_runtime_integration.py \
  tests/test_waypoint_native_integration.py tests/test_vln_simulation.py \
  tests/test_waypoint_simulation.py --junitxml="$VERIFY/audit/optional_regression.xml" \
  > "$VERIFY/audit/optional_regression.log" 2>&1
)
```

The core invocation actually used the new worktree's ignored `.venv` symlink to
that same `core-env`; no packages were changed. The output creation guard above is
provided for safe reproduction. MPS and the full suite's
ffmpeg shared-memory access require the existing unrestricted local tool context.
Keep hardware opt-ins disabled and coordinate again if physical testing resumes.

## Remaining read-only live work

Connect Claude's already-tested RGB receiver to `ObservationInbox` in a separately
coordinated **read-only** test. Validate exposure/receipt semantics, availability,
clock epochs and host-monotonic mapping/uncertainty against actual arrival, then
measure display age and bounded buffering under live cadence. This smoke did not
produce a fresh advisory under the default limit; freshness in live operation is
not established. No threshold was relaxed to turn that result into a success.

`T_base_camera` and reliable near-goal stopping remain unresolved. They prevent
claims of registered metric commands or autonomous completion; they do not prevent
an explicitly unregistered, RGB-only advisory display. Optional pose/LiDAR remain
unknown/separate when absent. [VLN_CONTROL_HANDOFF](VLN_CONTROL_HANDOFF.md) now cites
Claude's published reports at `742cfe4fb1b71f9d19d9d2d62be56f03b6b03707`, including
unverified operator takeover and obstacle-stopping behavior. Nothing here authorizes
or connects model output to that control route.
