# One supervised model-directed step

Branch: `feature/vln-supervised-navigation`. Base: interactive
`fdb5451e25cdc1f5971bbf0e1c7887ec51ff634a`. Initial shared contract publication:
`2213ced` (read the full SHA with `git rev-parse 2213ced`). Contract, schema and
fixture were pushed before implementing the executor/UI.

**Implemented and verified offline:** one preview → human target-instance/ROI
confirmation → one consumed authorization → a new independent observation/model
request → one bounded action through an isolated fake executor → separate stop
and release results → a newer observation. **Physical readiness is gated.** No
model was loaded, no SDK/DDS initialized and no Go2 command sent in this work.
Claude owns hardware evidence, guard-provider integration and the physical trial.

Read [the shared contract](SUPERVISED_NAVIGATION_CONTRACT.md) first. This guide
describes the implemented interface, including its remaining limitations.

## Components and unchanged defaults

| Component | Implementation |
|---|---|
| Versioned records | `vln/schemas/supervised-navigation-v1.schema.json` |
| Restricted planner, approvals and state transitions | `vln/supervised.py` |
| Conservative reviewed-image correspondence | `vln/supervised_target.py` |
| Subprocess client, fake executor and audited-loop wrapper | `vln/supervised_executor.py` |
| Separate hardware/mock process | `scripts/vln_supervised_executor.py` |
| Existing four panels plus minimal supervised controls | `vln/supervised_app.py` |
| Opt-in launcher | `scripts/run_vln_supervised.py` |
| Automated synthetic HTTP/worker/IPC demonstration | `scripts/demo_vln_supervised_mock.py` |

Paths under `vln/` are relative to `src/go2_perception/`. The existing
`run_vln_interactive.py` remains read-only by default. Its optional app factory is
used only by the new launcher. The original page, target-review flow, worker,
calibration, model settings, three-second freshness limit and defaults are
preserved. `/api/navigation` is absent for an ordinary read-only application.
The old `/api/guided` endpoint is disabled in a supervised session to prevent a
second, unbound instruction-control path. Main and Claude's worktrees are unchanged.

The new application uses the existing worker, observation inbox, measured camera
preprocessing, trial archives and recording paths. A preview and its refresh are
two explicitly linked **independent short-observation trials**; neither carries
model conversational/action memory into the next. Model inputs remain RGB plus
instruction. Targets, approvals, geometry and evaluator data are not model inputs.

## States and operation

1. **IDLE:** choose a target and obtain one preview. A mock run uses a fixed noisy
   synthetic image and scripted native proposal, with no semantic/model claim.
2. On the exact loaded model-input image in panel B, drag a target ROI and enter
   the physical instance description and operator name. Confirming produces
   **REVIEW** with one bounded action. Image IDs/hashes are checked. The displayed
   preview age keeps increasing; review does not restamp it.
3. Check the per-cycle operating-zone/intervention/source statement and authorize
   once. **REFRESHING** immediately consumes the authorization, with a maximum
   15-second TTL (10 seconds in the synthetic example). The app drains old queued
   camera frames, starts a new trial and makes one new request. It additionally
   checks that every source image in that request follows authorization.
4. Revalidation must retain the exact instruction/session/epoch, reviewed instance,
   permitted pose drift and primitive direction. Magnitude, rate and duration
   cannot exceed the reviewed envelope. The image's age including a justified
   uncertainty bound must remain ≤3 seconds, including immediately before the
   first command after ownership acquisition. Unknown capture bounds HOLD.
5. **EXECUTING:** the isolated child independently reproduces the action and
   rechecks authorization, nonce, state and geometry. It is the only command
   process. **STOPPING** means a stop was requested, not that it happened.
6. **AWAITING_OBSERVATION:** only after fresh measured stationary, release and
   post-release stationary are separately verified. A new camera frame must be
   strictly later than the executor's end time and remain in the same epochs.
7. **COMPLETE:** the software cycle is closed; old proposal/approval are invalid.
   This is not navigation/goal success. The first-pilot executor accepts at most
   one dispatch attempt per process. Start a new session for another physical
   cycle, and obtain another preview/authorization.

**HOLD** reports its specific reason. Refresh failures never cause automatic
inference/execution retries. A new human action may start another review while no
dispatch has occurred; cancelled executor sessions and dispatched sessions need
a new session. Cleanup/stop uncertainty locks the session and requires Claude's
verified independent intervention procedure.

Browser/terminal `cancel` discards pending work; if movement is active it requests
STOP from the child. The UI labels pending cancellation and active STOP separately.
HTTP cancellation reaches the child pipe immediately rather than waiting for the
model/preparation loop. Closing the application requests cleanup and waits; it
does not forcibly kill a live cleanup process. SIGINT/SIGTERM and parent-heartbeat
loss also request child cancellation. No software can guarantee delivery after
process/OS/network failure; do not substitute these for independent intervention.

## Restricted path conversion and target comparison

The first meaningful native row is selected after skipping only near-identity
rows (5e-3 m/rad). Each row is an absolute SE(2) pose in the fixed decision frame.
`T_base_model @ P @ inverse(T_base_model)` maps the displacement; the mapping must
be validated for the selected primitive. No identity transform is guessed.
Forward requires positive x and lateral/yaw residuals within the evidence limits.
An in-place turn requires translation within its evidence tolerance. Other
combinations HOLD. Selected row, mapped pose, ignored residual, clipped magnitude
and full native proposal are recorded separately. Native STOP/near-zero whole
chunk result in no motion. The planner does not use goal coordinates or integrate
waypoints as velocities. Timing comes from verified hardware limits.

The initial target-continuity gate uses mutual ORB feature matches (ratio <0.7),
an affine partial fit with ≤2 px RANSAC residual, ≥16 inliers, ≥80% inlier fraction,
≥8 inliers inside the reviewed ROI, ≤1.5 px median residual and ≤2% corner drift.
The fresh model point must fall inside the registered ROI. This deliberately
refuses textureless/ambiguous/changed images. It is a conservative visual
continuity heuristic **requiring hardware-trial validation**, not proof of
physical identity, semantic correctness or collision safety. Equal target words,
copied instance IDs and OPOS/APOS tokens alone cannot pass it.

## Executor boundary and measured stop

The model/browser process never imports `sdk_routes`. After explicit live gates,
the child uses the hardware Python, the canonical control checkout at
`742cfe4fb1b71f9d19d9d2d62be56f03b6b03707`, and SDK
`814556d15970dd2ecf1c9984e845ca02ab07e206`. It verifies tracked source content and
the guard-provider hash, then acquires the canonical utility's existing
`CommandLock` before initializing subscribers. No hardware environment is changed.

The unchanged `go2_control.safety.execute` owns the audited 1004 claim/release,
1003 velocity/zero route and cadence. The wrapper additionally checks current
geometry, radio-state age, robot errors, initial stationary dwell, parent heartbeat
(0.5 s), cancel, independent deadline and first-send freshness. It aborts on a
failed local velocity send. A fresh stationary dwell is required after movement;
pre-motion dwell is reset. The audited monitor uses <0.03 m/s planar speed and
<0.05 rad/s yaw speed; these are observed-state thresholds, not exact zero motion.
Current guards, raw robot state, stationary samples and post-release samples are
retained in private executor events. It observes for three further seconds after release,
because the earlier hardware audit reported post-release movement. These are
measurements/checks, not guaranteed emergency-stop performance.

The child reports local sends, robot acknowledgements, progress, stop sends,
stationary, release RPC and post-release stationary independently. 1003 uses
`_CallNoReply`: robot/stop acknowledgement stays null, including when local send
returns zero. Stop-send failures and RPC exceptions retain errors and unknowns.
No avoidance-disable method, gait/posture action or SportClient fallback exists.

**One canonical commanding location:** all commands in the trial must use the
same control checkout and its `data/outputs/go2_control/.command.lock`. The child
holds that lock for the entire session. Do not use utilities from another
worktree. The audited utility's `stop` command bypasses its lock; do not launch it
as a competing command source. Use the child's stop channel and the independently
verified intervention procedure. Unverified stop/release retains the lock for
human investigation; never delete it automatically to continue. A file lock does
not exclude unrelated SDK clients, so exclusive-source confirmation is required.

## Claude's capability and guard-provider checklist

No real capability file or guard provider is supplied by this milestone. The
synthetic fixture is explicitly rejected by live mode. Before one physical trial,
Claude must supply and review:

- A `kind: hardware` capability following `$defs.capability`, with configuration,
  expiry, evidence references (include local log/file hashes, reviewer and measured
  conditions), exact canonical paths and source versions.
- Validated planar registration/conjugation for each permitted action, including
  signs, heading, offsets and acceptable approximation residuals. No invented
  `T_base_camera` or simulator limits.
- Verified low-speed magnitudes/rates/deadlines, braking margin/overshoot and pose
  tolerances for the static operating zone. Existing implementation ceilings are
  ceilings only, not evidence of safe limits.
- Independent intervention while API owns commands, obstacle handling for the
  selected direction, measured stop/release and post-release behavior. Previous
  hardware reports leave intervention and obstacle stopping unresolved.
- Validation of the target-continuity rule in the trial conditions, and a capture
  age upper bound valid for the **current camera epoch**. The existing 0.048 s
  estimate/0.016 s sigma is not automatically a guaranteed upper bound.
- One pinned, reviewed hardware guard module at an absolute private path. Its
  SHA-256 is in `control.provider_sha256`; it must not send any commands.

The guard module exports:

```python
def create_guards(*, monitor, capability, monotonic, observation_root):
    # Return the hardware-validated, read-only provider described below.
    ...
```

This is an interface signature, not an implemented live provider. It must return
an object with:

- `snapshot()` → `{"pose": <schema pose>, "geometry": <schema geometry>,
  "capture_uncertainty_bound_s": <validated bound or None>,
  "capture_clock_epoch": <camera epoch>, "capture_bound_basis": <evidence reference>}`.
  Pose/geometry keep their actual acquisition timestamps. The child samples at
  ~20 Hz, and the core retains up to 200 snapshots to bind observation-time pose.
  This method must read a bounded, thread-safe cache, not block on RPC/network IO.
  Stop/intervention/state checks must remain responsive if a sensor disappears.
  `observation_root` points to the current private RunStore, so a background reader
  can bind the camera bridge/session epoch and calibration metadata to validated
  timing evidence. Reading a camera epoch is not validating its capture bound;
  return None until the independent evidence applies to that epoch/configuration.
- `progress(action)` → a callable measuring signed distance/heading since action
  start, or None when measurement is unavailable. Use validated sensor progress,
  never nominal speed × elapsed time. Preserve the audited range/odom/yaw caveats;
  do not silently fall back to unvalidated geometry. The wrapper rejects missing,
  nonfinite and negative progress. Provider reset/gap/discontinuity behavior must
  be validated by Claude.

Geometry must represent measured/verified clearance for the actual forward or
turn envelope in the pose frame/robot epoch. Zero returns, a target point, an old
evidence file or a nominal field of view are not free-space evidence. Missing
coverage yields `available: false`. A provider must report clock resets, not
reuse old poses/timestamps. The pinned provider runs only in the isolated child;
it is the remaining hardware integration boundary, not a second motion controller.

## Reproduction

Use the existing isolated **core** Python; no new installations are required.
All commands below are from the integration worktree. Set `CORE_PY` to that
environment's Python. Outputs must be fresh private/Git-ignored directories.

```bash
export PYTHONPATH=src:scripts
export GO2_HARDWARE=0 GO2_SUPERVISED_LIVE=0 MUJOCO_RENDER_TEST=0
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

"$CORE_PY" scripts/demo_vln_supervised_mock.py \
  --output data/outputs/supervised/mock_cycle_new

# Interactive synthetic browser demonstration (no model or robot):
"$CORE_PY" scripts/run_vln_supervised.py \
  --output data/outputs/supervised/mock_browser_new --port 8766

"$CORE_PY" -m pytest -q \
  tests/test_supervised_contract.py tests/test_supervised_navigation.py \
  tests/test_supervised_executor.py tests/test_supervised_integration.py \
  tests/test_vln_interactive.py tests/test_vln_shadow.py tests/test_vln_live_shadow.py
```

The separately gated **live command is for Claude only, after all evidence and
resource coordination are complete. It was not executed here.** Define
`CAPABILITY`, `CONTROL_ROOT`, `HARDWARE_PY`, `SDK_ROOT`, `MODEL_PY`, `ASSETS`,
`GO2_IFACE` and `CAMERA_HOST_IP` from Claude's verified local setup. The hardware
Python must exactly match the capability's venv path; two venvs pointing to the
same base interpreter are not treated as interchangeable.

```bash
GO2_HARDWARE=0 GO2_SUPERVISED_LIVE=1 PYTHONPATH=src:scripts \
"$CORE_PY" scripts/run_vln_supervised.py \
  --navigation-mode live --execute-live --exclusive-command-source \
  --capability "$CAPABILITY" --control-root "$CONTROL_ROOT" \
  --hardware-python "$HARDWARE_PY" --unitree-sdk "$SDK_ROOT" \
  --camera go2 --lidar go2 --iface "$GO2_IFACE" --camera-host-ip "$CAMERA_HOST_IP" \
  --backend lightnav --model-python "$MODEL_PY" --assets "$ASSETS" \
  --preparation-mode cached --tensor-evidence skip \
  --output data/outputs/supervised/claude_one_step_new --port 8766
```

Launching is not motion authorization. The operator must still inspect the exact
image/ROI/action and authorize one refresh/action through the UI. All physical
pytest opt-ins remain disabled. Do not enable ordinary hardware tests to use this
launcher. If a gate fails, preserve the HOLD and its evidence; do not weaken it.

## Verification and retained limitations

Execution machine: existing Apple Silicon Mac/core Python 3.12.11; CPU-only mocks,
no PyTorch inference, GPU use, SDK/DDS initialization or robot commands.

- Initial publication: 3 contract tests passed.
- First selected regression run: **156 passed in 48.53 s**, no failures/skips.
- Subsequent focused integration run: **60 passed in 11.42 s**, no failures/skips.
  Includes a real localhost HTTP cycle and separate executor process. Final
  selected regression: **165 passed in 57.20 s**, 0 failed, 0 skipped. Final
  executor + UI/schema follow-up: **25 passed in 10.25 s**, 0 failures/skips,
  including final epoch/telemetry, canonical cycle archive and mock-native archive
  consistency checks. Two regression cases were added after the broad run; its
  historical 165-test count is not relabeled as a later full-suite run.
- Durable manual-driver result: `data/outputs/supervised/mock_cycle_07/`, completed
  synthetic 0.08 m primitive, two mock requests (preview + refresh), 3.25 s review
  delay, one dispatch and a strictly newer observation. Main artifact:
  `navigation/cycle_result.json`; canonical schema-valid record:
  `navigation/cycle_record.json`; image: `navigation/post_motion.png`; full
  snapshots/events/native inputs remain in that ignored run; the canned native
  archive matches the returned proposal. Earlier demo archives are retained
  unchanged. This is not a
  trained-model or physical motion result.
- Earlier ignored runs are retained: `mock_cycle_01` was socket-sandbox blocked;
  `02` caught the native point tuple/JSON-list boundary; `03` and `04` caught
  freshness checks using a time sampled before a later guard sample. Correcting
  the check ordering did not relax timing thresholds. The real-clock process
  regression now covers the complete path.

No full heavyweight regression, actual LightNav refresh, live guard provider,
hardware motion or physical-stop trial was run. Hardware tests were excluded
from the selected lightweight suite, not reported as passes. Browser controls
were exercised through their HTTP API; no manual browser visual validation is
claimed. The model/semantic quality, actual target-continuity reliability,
calibration, capture-age bound, independent intervention, obstacle behavior,
physical limits and post-release stability remain Claude's verification work.
