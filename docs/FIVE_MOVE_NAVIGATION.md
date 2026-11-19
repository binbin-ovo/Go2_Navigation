# Supervised object search and approach

This opt-in episode gives one operator authority to search for one named object within an inspected static area. The executor accepts at most five movement attempts. Finding the object changes the mode to APPROACH and preserves the number of attempts already spent.

The [November 19 box review](TARGET_GROUNDING_REVIEW.md) traces a completed live five-move trial. The original development runs and commands below describe the baseline publication.

## Reproduce the offline trial

Run from the dedicated worktree on the Mac. The driver starts the real loopback viewer, recorder and executor IPC with a synthetic camera, LiDAR and mock decisions. It authorizes once, checks the fixture's expected outcome and closes its child processes. The output directory must be new.

```bash
export GO2_WORKSPACE=/Users/yushan/Robotics/Go2
cd "$GO2_WORKSPACE/code/worktrees/codex-motion-diagnostics"
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/hardware"
export GO2_HARDWARE=0 GO2_SUPERVISED_LIVE=0 MUJOCO_RENDER_TEST=0 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$GO2_WORKSPACE/runs/five_move"
episode_dir=$(mktemp -d "$GO2_WORKSPACE/runs/five_move/mock-XXXXXX")
"$GO2_WORKSPACE/envs/core/bin/python" scripts/demo_vln_supervised_mock.py \
  --five-move-episode --output "$episode_dir/session"
```

For an interactive synthetic trial use the same exports and a fresh directory. This entry point prints the local viewer URL. Enter `go to the chair`, use a synthetic operator name, check both area acknowledgements and select **Authorize episode**. Panel B uses the exact archived model input. Panel C shows the native chunk and the selected primitive, drawn with an orange dashed line or yaw arc. The status panel records requested and settled movement in metres or radians. **STOP EPISODE** and the terminal command `stop` latch cancellation independently of inference.

```bash
"$GO2_WORKSPACE/envs/core/bin/python" -u scripts/run_vln_supervised.py \
  --five-move-episode --profile go2-effective-v1 --lidar synthetic --port 0 \
  --output "$episode_dir/browser" --max-host-seconds 300 \
  --min-available-memory-gb 0 --min-free-disk-gb 0
```

The fixture begins with an empty view. Its scripted chair appears after the fake search turn reports settled progress. A second turn aligns the mark, a forward step brings the illustrated chair closer, and the next mock decision emits native STOP. Human identification can affect executor grounding. Only the synthetic worker can use the scripted fixture view. A real LightNav job contains the instruction, causal image files and their timing. [Synthetic fixture](../examples/five_move_navigation/synthetic_episode.json)

## Episode contract

The version identifier is `go2-five-move-episode-v1`. The JSON Schema covers a grant, a state snapshot and the synthetic fixture. The executor enforces time, identity and cross-record invariants in addition to the schema. The existing v2 proposal retains the entire native output, ten absolute ego-frame waypoints, point, STOP, model identity, frame identity and image hash. [Episode schema](../src/go2_perception/vln/schemas/five-move-episode-v1.schema.json) · [Per-action evidence contract](SUPERVISED_NAVIGATION_CONTRACT.md)

| Setting | Default | Accepted values |
|---|---:|---|
| `--move-budget` | 5 | 1–5 nonzero dispatches, turns and translations combined |
| `--episode-seconds` | 60 | 5–60 host-monotonic seconds from authorization |
| `--area-radius-m` | 2.0 | 1.2–3.0 m around the authorized odometry pose |
| `--alignment-deg` | 15 | 8–18° to enter alignment, with 3° hysteresis to leave |

1. **IDLE → OBSERVING.** A named operator confirms the static disc, blind rear, cable, independent intervention and exclusive command source. The grant binds one executor nonce, session, trial, instruction, target, camera epoch, capability SHA-256, starting pose, limits, issue time and deadline. One executor holds the canonical process lock for the session.
2. **OBSERVING → IDENTIFYING or MOVING.** The executor checks the proposal, image hash binding, pose, geometry, clock epoch and capture bound. The lower bound on exposure time and the pose and geometry stamps must postdate authorization, identification or the previous action's completed settling. The source/receipt age ceiling is 3 s. A duplicate decision or source cannot buy a second dispatch.
3. **MOVING → OBSERVING.** Before calling the audited action loop, the executor spends a slot and records a unique digest of the grant, slot, raw proposal, current grounding and selected action. The radio and its freshness are polled between decisions and during movement. The cancellation latch, parent heartbeat, avoidance precheck, current pose and LiDAR guards remain in the execution path. After zero sends, stationary dwell, release and post-release verification, a fresh view can authorize the next decision under the existing grant. A stopped stall or modest endpoint error consumes its slot. The reason and settled estimate remain visible.
4. **TERMINAL.** A final view is permitted after the last slot. The executor cannot dispatch again or rearm. An explicit image-bound human arrival assessment can update an eligible result and preserves its previous outcome. It requires an image taken after the last movement. Unverified cleanup is recorded separately and the canonical lock is retained when ownership or stillness is uncertain after a claim.

The browser sends observations and image-bound reviews to the executor. Raw model waypoints never go to a velocity client. Source names in each decision distinguish `native_prefix`, `native_search_prefix`, `search_policy` and `approach_policy`. The audit joins `episode_authorization`, `episode_observation`, `episode_dispatch`, per-action `result` and `episode_state` by episode, prediction and single-use action IDs. [Executor](../src/go2_perception/vln/five_move.py) · [Selection policy](../src/go2_perception/vln/episode_policy.py)

## Target, history and motion

SEARCH accepts an absent or uncertain target. It can follow a coherent in-place model turn, select a new inspected heading, or change viewpoint with a 0.30 m observed-corridor step. Measured poses, previous outcomes and the last supported target are retained. A heading already inspected within 12° and 0.25 m is avoided. Missing floor returns terminate the observable forward corridor.

APPROACH requires an independent instance reference. Without a reference image the first unverified mark pauses the episode. In panel B the operator can identify a tight region containing the mark, reject the instance or select **Unsure · seek another view**. An instance decision grants no additional motion. The next fresh mark must fall within the re-identified object. The matcher uses mutual ORB correspondences inside the named region and a RANSAC similarity transform. It requires at least 12 inliers, 80% inliers, distributed reference features and a median residual at most 2 pixels. A different or ambiguous object cannot authorize an approach. A changed viewpoint can be identified again. For a pre-identified local photograph, supply `--target-reference FILE --reference-target chair --reference-roi X0 Y0 X1 Y1` with normalized coordinates in that photo. The target may still be absent from the starting view. [Grounding implementation](../src/go2_perception/vln/episode_grounding.py)

The bearing uses the rectified image's horizontal focal length, principal point and measured centre-ray heading. Planar axes are forward x, left y and positive counterclockwise yaw. The real camera translation and target range are unknown. Within the alignment band, a compatible forward prefix can run. A short separately logged 0.30 m approach also requires contiguous forward intent, lateral and curvature checks, and enough observed stopping room. Outside the band the policy can select a 20–30° turn. When it enlarges a nominal turn, the predicted residual plus heading uncertainty must fit the band. Every turn requires re-observation. The real one-action baseline keeps its original heading gap.

The area remains anchored at the original odometry pose after translation. The executor checks both the current position and projected remaining path including the forward stopping reserve, the 0.65 m swept footprint and a pilot uncertainty allowance of `0.15 + 0.05 × attempted moves` metres. It repeats this check during execution. Current observed floor/obstacle evidence can veto a turn even inside the area. This margin is a declared pilot assumption. Real slip, scene changes and unobserved obstacles limit what odometry and a forward-facing sensor can establish.

The effective profile uses 1.0 m/s forward and 2.0 rad/s for turns, 0.30–0.60 m forward, a 3 s per-action deadline and 0.05 s resend period. Forward stopping lead/reserve are 0.10/0.25 m. Left lead/reserve are 10°/12° and right lead/reserve are 4°/8°. The per-action guards and independent stopping checks also apply to exploration. [Effective profile](../src/go2_perception/vln/motion_profile.py)

One warmed model worker serves the trial. Images continue to enter the causal history during movement and settling. Each decision reconstructs the first two and latest six frames of the available causal 4 Hz grid, deduplicated, with original elapsed times encoded at millisecond resolution. Native sample construction resets per decision. Executed actions, evaluator state, human labels and synthetic fixture metadata are excluded from the real model input. Old bounded-context experiments showed recognition and path regressions. This episode's offline measurements cover the processor and orchestration. They do not establish learned search accuracy. [Original bounded-context experiment](https://github.com/binbin-ovo/Go2_Navigation/blob/6e7438a3201643e4db2bd294d77b3542b6d8a4b5/docs/VLN_CONTEXT_ABLATION.md)

## Interpreting outcomes and recordings

| Outcome | Evidence |
|---|---|
| `GOAL_CONFIRMED` | Explicit operator arrival confirmation on an eligible, named post-settle image |
| `MODEL_STOPPED_ARRIVAL_UNVERIFIED` | The native decoder emitted STOP |
| `MOVE_BUDGET_EXHAUSTED` | All allowed movement slots were spent. `target_found` reports whether supported evidence occurred |
| `BLOCKED_OR_NO_USEFUL_ACTION` | The proposed primitive, inspected area or observed space cannot support another move |
| `SENSOR_OR_CONTROL_FAILURE` | Timing, state, recording, execution or cleanup failed validation. Inspect the recorded reason |
| `OPERATOR_ABORT` | The operator interrupted the episode |
| `EPISODE_DEADLINE` | A time bound ended the episode |

Private run artifacts include `episode_summary.json`, `executor/executor_events.jsonl`, `session/requests/*/{job,snapshot,receipt}.json`, the trial's predictions and human labels, and `lidar/index.jsonl` with asynchronous scan/odometry chunks. Inspect `guard_rejected` for the exact veto snapshot and `motion_feedback.basis` for raw odometry, LiDAR and fusion decisions across driving, stopping and settling. Check chunk drops and writer drain in `summary.json`. The run can be reopened with `scripts/run_vln_interactive.py --review-run PATH`.

## Smallest physical pilot for Claude

First run `bash scripts/launch_five_move_episode.sh --prepare-only` in this worktree. It verifies local capability evidence and exits before starting hardware. The 0.159 s uncertainty bound applies only to the November 19 00:06 measurement and its matching boot. The later box trial used its own 11:31 measurement of 0.140 s. A boot mismatch holds the episode. Use the project's existing boot-specific readiness procedure if necessary. [Mac workspace and sensor setup](MAC_CONNECTION.md) · [November 19 evidence](AGENT_HANDOFF.md#two-counted-forward-runs)

For a separately supervised hardware window, use one distinctive chair just outside the starting view, a clear static 2 m radius and a two-move budget. Aim to observe one search and one grounded approach. The operator can administer the first identification when the chair comes into view. Use a pre-identified image if that is more practical. The operator's emergency intervention must be available throughout. The hardware launch below is for Claude's pilot.

```bash
GO2_SUPERVISED_LIVE=1 bash scripts/launch_five_move_episode.sh --live-exclusive \
  --move-budget 2 --episode-seconds 45 --area-radius-m 2 --alignment-deg 15
```

Record the run path, target/image correspondence before and after each attempt, selection source, measured sign and approximate progress, guard reason, verified stop/release and operator judgment of whether the chair was approached or reached. If alignment consumes both slots, retain that result. The two prior measured-by-sensor forward runs remain separate historical trials. Independent distance measurement can refine the motion diagnostics later.

## Offline verification

The selected core regressions passed 259 cases in 50.84 s. After adding the idle radio and stale-remote cases, the final episode module passed 31 cases in 0.64 s. The two runs cover 261 distinct core tests. The isolated model environment passed all 10 native processor tests in 5.33 s using CPU and mock generation. There were zero failures or skips in these selections. Hardware tests were excluded.

The browser fixture produced one grant, a 30° search turn, a 20° coarse alignment turn, a 0.30 m forward approach and native STOP. The fake settled feedback was 20°, 10° and 0.20 m. All four inputs and three attempts remain visible in the private review. The terminal result was `MODEL_STOPPED_ARRIVAL_UNVERIFIED`. A separate native processor test exercised actual sparse inputs with synthetic pixels and mocked generation. At that publication physical behavior and live learned recognition awaited testing. Later closed box episodes and their grounding outcomes are recorded in the [target review](TARGET_GROUNDING_REVIEW.md).

```bash
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
"$GO2_WORKSPACE/envs/core/bin/python" -m pytest -q -rs -p no:cacheprovider \
  -m 'not hardware and not model_integration' \
  tests/test_five_move_navigation.py tests/test_motion_diagnostics.py tests/test_effective_motion.py \
  tests/test_hardware_guards.py tests/test_supervised_contract.py tests/test_supervised_executor.py \
  tests/test_supervised_navigation.py tests/test_supervised_integration.py tests/test_vln_interactive.py \
  tests/test_vln_shadow.py tests/test_pretrained_worker.py

LIGHTNAV_ASSETS="$GO2_WORKSPACE/weights/lightnav" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  "$GO2_WORKSPACE/envs/model/bin/python" -m pytest -q -p no:cacheprovider \
  tests/test_bounded_native_integration.py
```
