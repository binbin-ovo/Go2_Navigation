# Supervised model-directed Go2 navigation

## Next integrated trial on `feature/vln-effective-motion`

Claude completed the first physical model-directed step at 0.30 m/s. The
`go2-effective-v1` profile adds a contiguous lookahead and measured stop
trigger on top of the published hardware-readiness branch
`24f12ca80605325e88c976e1af5849077a918a55`. The updated canonical control
checkout is `6d555d31c6eaa4ee194920925ab96238bf15b128`. The profile was
verified offline against original LightNav archives and the pinned pure control loop.
A physical outcome for the effective integrated profile remains unmeasured.
[Contract and units](SUPERVISED_NAVIGATION_CONTRACT.md) · [Hardware trials, sections 8–10](GO2_HARDWARE_READINESS.md)

**One proposed trial.** From Claude's working Mac shell with its camera network
permission, place the box directly ahead at about 2.45 m in the same static
clear corridor. Use one target ROI and one authorization. Read the rate,
destination, trigger and clearance in the viewer before authorizing. For a repeat
of the archived path, expect 0.60 m forward at 1.0 m/s, zero requested at 0.50 m
measured progress, a 3 s drive deadline and ≥0.85 m usable forward clearance
plus the current scan-age margin. The new model answer determines the actual
request. A smaller, blocked or absent request stays as observed.

```bash
cd /Users/yushan/Desktop/Go2_Navigation/data/outputs/vln-effective-profile-worktree
GO2_SUPERVISED_LIVE=1 bash scripts/launch_effective_trial.sh --live-exclusive sensor_only
```

The launcher checks the existing controller and evidence files, generates a fresh
capability, and uses the original hardware/model environments, cached weights,
interface `en9`, host `192.168.123.222`, configured camera timing and port 8766.
It prints a unique private trial directory under `data/outputs/effective_trials`.
The viewer still requires explicit review and fresh authorization. The native
avoidance check, shared command lock, independent intervention, nominal error code
100 and continuous pre-claim stationary dwell remain in effect. Physical pytest
opt-ins stay disabled. For a file-only preflight, invoke the same script with
`--prepare-only sensor_only` and `GO2_SUPERVISED_LIVE=0`.

**The one measurement to collect** is progress at first zero and after verified
settling for the adjusted trigger. Inspect the recorded feedback basis, peak,
signed error, zero sends, fresh stationary dwell, release and post-release state.
The proposed forward tolerance is ±0.10 m. Keep partial progress, stalls and
overshoot as outcomes. The boot-specific 0.148 s capture bound must still match
the active clocks. A changed boot holds arming until its timing bound is established.
At this verification the Mac had about 14 GiB free and the launcher requires 10 GiB.

### Evidence behind the trial settings

| Recorded condition | Measurement near first zero | Peak / settled response | First still after zero |
|---|---|---|---:|
| Forward 1.0 m/s, measured trigger 0.60 m | 0.610 m odometry, 0.661 m fused feedback | 0.91 m/s peak speed, 0.714 m settled forward odometry | 0.148 s |
| Right −2.0 rad/s, measured trigger 30° | −31.8° IMU | −37.2° peak, −34.4° settled, 6.0 cm planar drift | 0.506 s |
| Left +2.0 rad/s, measured trigger 30° | +31.0° IMU | +41.0° peak, +40.8° settled, 1.0 cm planar drift | 0.337 s |

There is one controller-only record per condition. Run IDs, hashes and original
request parameters can be verified by the replay command below. The forward
odometer under-read tape in earlier slower runs. Absolute fast travel has no
independent tape measurement. The leads (0.10 m, 4° right, 10° left), reserves
and tolerance are declared pilot settings. Different floors, battery state,
clearance or startup response may yield different endpoints.

### Original predictions through the new conversion

Twelve saved prediction records were read, including the two decisions from the
same executed cycle. The replay checks the archived model job, native decoder
output, snapshot source, image SHA-256, target point, captured epochs and
published controller event hashes. Only the original legacy execution has a
measured model-directed outcome. Each counterfactual retains a null measured
movement field in the private trace.

| Archived case | Native evidence | Effective conversion | Actual motion evidence |
|---|---|---|---|
| Distant centred box, supervised preview and refresh | Both raw paths extend to x = 0.945 m | Both select 0.600 m at 1.0 m/s, trigger 0.500 m. Endpoint lies 19.7% into the interval ending at row 4, zero-based. Reviewed/refreshed clearances 1.957/1.927 m | V1 selected row 0 for 0.143 m at 0.30 m/s. 23 sends in 1.26 s, 0.154 m fused at trigger, 0.129 m odometry near zero and 0.152 m settled. Three zero sends, release code 0, still afterward |
| Three centred `live_14` decisions | Consistent forward prefixes | All three reach the 0.600 m profile bound | Observation-only |
| Five off-centre `live_14` decisions | Turn and travel | Three right candidates (30°, 29.7°, 30°), one left (30°). The remaining 15.0° right prefix stays below the 20° admission floor | Observation-only. No turn-clearance attestation or subsequent physical replan exists in those records |
| Nearby box at about 1.1 m | Native STOP in a stale answer | No motion, stale receipt retained | Observation-only |
| Reviewed absent sofa | OPOS-0, nonzero native path, operator agreed absent | No target point, no action | Observation-only |

V1's selection of the first row accounts for the short requested distance on
the successful trial. Its 4 s deadline was still far away when measured feedback
ended the velocity phase. The original recording's first refresh image was
estimated 0.047 s after authorization. Subtracting its 0.148 s measured uncertainty
leaves an overlap of 0.101 s. V2 starts the fresh history once the camera capture
lower bound follows authorization, then checks every selected image. The offline
comparison reproduces the spatial plan only. Another live answer can differ.

The read-only `live_14` and sofa archives lack a decision-bound supervisor guard
record and one-use authorization. Their traces show kinematic candidates or
refusals. A turn on the current provider holds in the default `sensor_only`
scope. In a separately chosen static inspected scope, the operator confirms the
full rear/side disc and cable at the reviewed pose. After verified stop/release
the system obtains a fresh view and makes one read-only replan. Full registration,
sensor-certified rear clearance and physical model-directed turning remain open.

### Reproduce the offline analysis

From the integration worktree, the following commands create new private files.
The original camera, model and controller recordings are inputs.

```bash
base="$HOME/Desktop/Go2_Navigation"
ready="$HOME/Desktop/Go2_Navigation-hw-readiness"
control="$HOME/Desktop/Go2_Navigation-go2-control"
interactive="$HOME/Desktop/Go2_Navigation-vln-interactive"
core="$base/data/outputs/phase6c/core-env/bin/python"
export GO2_HARDWARE=0 GO2_SUPERVISED_LIVE=0 MUJOCO_RENDER_TEST=0
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src:scripts:hardware
mkdir -p data/outputs/effective_replays
out=$(mktemp -d "$PWD/data/outputs/effective_replays/trace-XXXXXX")
"$core" hardware/make_capability.py --profile go2-effective-v1 --turn-scope sensor_only \
  --heading-file "$ready/data/outputs/hw_readiness/camera_heading.json" --output "$out/capability.json"
"$core" scripts/replay_effective_motion.py --capability "$out/capability.json" \
  --readiness "$ready/data/outputs" --interactive "$interactive/data/outputs" \
  --control "$control/data/outputs/go2_control" --output "$out/trace.json"
"$core" -m pytest -q -rs -p no:cacheprovider -m 'not hardware and not model_integration' \
  tests/test_effective_motion.py tests/test_supervised_{contract,navigation,executor,integration}.py \
  tests/test_hardware_guards.py tests/test_vln_{interactive,shadow,live_shadow}.py
PYTHONPATH="$control/src" "$core" -m pytest -q -rs -p no:cacheprovider "$control/tests/test_go2_control.py"
```

Retained private replay in this worktree: `data/outputs/effective_motion_release/trace.json`.
The capability-only launcher, assembled viewer JavaScript, synthetic localhost
HTTP cycle and pure audited control loop were exercised offline. The regression
selection passed **213 tests** (zero failures/skips, 57.91 s). The separate
canonical controller suite passed **42**, with one SDK-dependent helper test
skipped in the core environment. High-rate behavior under actual load remains
a physical trial question.

## Shared components and existing defaults

| Component | Implementation |
|---|---|
| Versioned records | `vln/schemas/supervised-navigation-v{1,2}.schema.json` |
| Prefix selection, approvals and state transitions | `vln/motion_profile.py`, `vln/supervised.py` |
| Conservative reviewed-image correspondence | `vln/supervised_target.py` |
| Subprocess client, fake executor and audited-loop wrapper | `vln/supervised_executor.py` |
| Separate hardware/mock process | `scripts/vln_supervised_executor.py` |
| Existing four panels plus minimal supervised controls | `vln/supervised_app.py` |
| Opt-in launcher | `scripts/run_vln_supervised.py` |
| Automated synthetic HTTP/worker/IPC demonstration | `scripts/demo_vln_supervised_mock.py` |

Paths under `vln/` are relative to `src/go2_perception/`. The existing
`run_vln_interactive.py` remains read-only by default. Its optional app factory is
used only by the supervised launcher. The original page, target-review flow, worker,
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

The legacy v1 profile selects the first meaningful native row after skipping only near-identity
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
continuity heuristic tested on the recorded box views in section 8 of the hardware report. Other scenes and targets still require review. Equal target words,
copied instance IDs and OPOS/APOS tokens alone cannot pass it.

## Executor boundary and measured stop

The model/browser process never imports `sdk_routes`. After explicit live gates,
the child uses the hardware Python, the canonical control checkout at
`6d555d31c6eaa4ee194920925ab96238bf15b128`, and SDK
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

## Hardware capability and guard provider

The [generator](../hardware/make_capability.py) binds controller and provider
source versions, evidence references, expiry, the recorded forward mapping and
the explicitly selected profile. For the effective profile it verifies the
heading evidence file and all three canonical controller logs by SHA-256.
The provider checks the boot-specific capture bound, ongoing odometry, scan
coverage and progress. The schema and launcher reject incompatible capabilities.
The synthetic fixture is reserved for portable mock testing.

The guard module exports:

```python
def create_guards(*, monitor, capability, monotonic, observation_root):
    # Return the hardware-validated, read-only provider described below.
    ...
```

Claude implemented this interface in [supervised_guards.py](../hardware/supervised_guards.py). Its object exposes the following methods.

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
its interface separates sensing from the motion controller.

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

For the next physical trial use the effective-profile launcher at the top of this guide.
The commands above reproduce the original synthetic demonstration.

## Historical v1 software verification

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

Claude subsequently ran live LightNav with the guard provider, verified intervention
and recorded the first supervised forward step. The current evidence, scoped
limitations and next experiment are at the top of this guide.
