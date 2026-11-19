# Go2 navigation handoff

## Current handoff — readable application adopted for hardware, November 19

Open `$GO2_WORKSPACE/START_HERE.md` (a link to `code/navigation/START_HERE.md`). The working checkout `code/navigation` is on `hardware/readable-navigation-integration`: Codex's `codex/readable-navigation-app` at `283b4b5b953a6d4a79efa286ec55941dbcf4e956` merged with `hardware/five-move-validation` (`96282e9`). The only hardware change not already in the refactor was the boot-specific camera bound: 0.140 s at boot offset −1766856813.961 s from `evidence/hw_readiness/capture_latency_20261119_113103.json`, with the matching regression test. It applies only to that robot boot and holds otherwise; re-measure after any reboot. Earlier bounds (0.159, 0.148, 0.172 s) remain historical. The controller stays `code/control` at `6d555d3` with its single `.command.lock`; the SDK stays `814556d`.

**What was checked offline (no robot connected, no movement).**

- Core suite: 558 passed, 13 skipped (12 native model cases, 1 physical test).
- Model environment: all 12 native cases pass. With `tests/test_bounded_native_integration.py` first, one `test_lightnav_runtime_integration.py` case fails, because each file puts a different LightNav source copy on `sys.path`. Either file alone, or the reverse order, passes. This test-isolation issue already existed before the refactor.
- Five-move mock: SEARCH left 30°, APPROACH left 20°, APPROACH forward 0.30 m, then native STOP recorded as `MODEL_STOPPED_ARRIVAL_UNVERIFIED`, with zero physical sends.
- Continuous offline trace: SHA-256 `d41e2829…` is identical to Codex's record.
- Real LightNav replay on recorded box inputs (`scripts/compare_grounding_prompts.py`, `runs/target_grounding/prompts-xvqwacul`): all 13 decoded outputs and input hashes match the pre-refactor run (`prompts-h_7tnoou`, `d0ff6ac`). The four plain "box" cases reproduce the original live-episode outputs exactly.
- Website (`app.py`): observation, five-move mock and continuous modes all opened in one browser session. The stop control cancelled the continuous run. Shutdown left no child processes.
- Full target description: *large brown cardboard box on the floor* reached the stored instruction intact, and the mark stayed labelled unverified.
- `sensors.py`: start, `latest_frame`, `latest_cloud`, `pose`, `status` and `stop` work on the recorded RGB episode and recorded LiDAR fixtures. The session refuses a restart after stop.
- `navigation.py`: `select_action`, `supported_ground`, `area_check`, `image_bearing`, `native_intent`, `primitive` and `bounded_number` are identical to the November 19 five-move policy by syntax-tree comparison.
  - `target_spec` additionally refuses empty or "go to the" text and keeps the typed text.
  - Instance matching now ignores case and whitespace in the target name only. It also validates the drawn region, which fixes the `NoneType` error a reject without a region produced on hardware.
  - Unchanged: motion profile, guards, capability generator, executor, bridges, calibration and runtime configs.
- Hardware launch routes: `scripts/launch_five_move_episode.sh`, `scripts/launch_effective_trial.sh` and `app.py supervised` all pass `--prepare-only`. Each resolves `code/control`, `envs/hw`, the SDK, `weights/lightnav` and this checkout's guard provider at 1.0 m/s and 2.0 rad/s. No active subprocess or launcher references a removed path. The four missing script names are provenance notes in configs and docs.

**Sensor interface differences.** `sensors.py` is a thin owner over the previous sources:

- Live RGB still comes from the `go2_camera_bridge.py` child (RTP/H.264, `envs/hw`), and LiDAR/odometry from the `go2_lidar_bridge.py` DDS child. Replay uses a paced feeder and `synthetic_lidar_bridge.py --recording`.
- New: public reads return copies, and camera status adds `alive` and `error`.
- There is no "current geometry" accessor. Clearance geometry is still computed only by the executor's hash-pinned guard provider, a separate process.
- Recorded LiDAR replay reads the `go2-recording-v1` layout. The `lidar/` chunk files written by five-move sessions use a different format, so they cannot yet be replayed through `sensors.py`.
- `T_base_camera` remains unmeasured.

**Next physical session (not run).**

1. Connect the robot and run a read-only boot check. Re-measure the camera bound if the boot changed.
2. Run `bash scripts/launch_five_move_episode.sh --prepare-only`.
3. Launch with `GO2_SUPERVISED_LIVE=1 bash scripts/launch_five_move_episode.sh --live-exclusive --move-budget 5 --episode-seconds 60 --area-radius-m 3 --alignment-deg 15`, then give one browser authorization per episode.

The refactored tree has not moved the robot. The first live session on it is also its first hardware check. The November 19 physical episode outcomes and operator wording are in `$GO2_WORKSPACE/runs/five_move/trial-*/operator_assessment.json`, and the formal five-move validation report is still pending. The continuous prototype remains offline-only.

## Previous application handoff

Open the workspace's `START_HERE.md`. The development checkout is `code/worktrees/codex-motion-diagnostics` on `codex/readable-navigation-app`, based on `91e98390baea3ec535a538d1fa5f3edae7024ae4`. From this checkout, activate `envs/core` and run `python app.py`. Camera and LiDAR, the browser, learned inference and command cadence keep their independent scheduling boundaries. The website starts in observation. Its five-move executor is synthetic and its continuous chair simulation is offline. The hardware operator procedure remains a separate explicit invocation.

Offline verification passed 558 core tests. Thirteen skipped there, including twelve native cases that passed separately in the model environment. The physical test remained disabled.

The public modules, settings and before/after behavior checks are documented in the [owner guide](../START_HERE.md), [20-minute walkthrough](CODE_WALKTHROUGH.md) and [verification record](APPLICATION_REFACTOR.md). Keep `code/navigation` on Claude's hardware branch and `code/control` at its canonical revision. For a future physical review, compare Claude's guard and regression changes for the current robot boot before any integration. The refactor has only been exercised offline. The [source inventory](CODE_INVENTORY.md) identifies old scripts that moved and the archive commit required for earlier experiments.

## Previous handoff — grounding diagnosis and offline review

Begin with [START_HERE](../START_HERE.md), the [original source guide](https://github.com/binbin-ovo/Go2_Navigation/blob/91e98390baea3ec535a538d1fa5f3edae7024ae4/docs/CODE_WALKTHROUGH.md) and the [box evidence report](TARGET_GROUNDING_REVIEW.md). Review branch `feature/vln-target-grounding-review` builds on continuous publication `ebefe9d25c820765d66a363d3d30a17368b3b5d6`. Claude's checkout remains on `hardware/five-move-validation` at `918f48c` with its private boot-specific guard and corresponding test edits. The controller is `6d555d3`. The midday box episode used a measured 0.140 s boot-specific bound. The 0.159 s value in the earlier handoff describes a different boot.

The closed `trial-THKUcr` run contains two operator-role rejections, two dispatched SEARCH steps, zero supported approaches, and an episode deadline. P002 was the wrong hallway mark on the exact model frame. On three saved histories, four descriptions and one repeat produced 13 decoded offline outputs. Extra detail suppressed a distractor but missed the edge box. The source-verified trace, side-by-side panels and exact replay commands are in the evidence report. New archives retain the originally entered text. The distinction between raw model marks, matched references, search and approach now appears in the viewer.

**Verification.** The focused offline selection passed 217 tests in 35.61 s. The study used one isolated fixed-model worker and its own private output directory. The replay checked 123 original source files before and after. The closed logs supersede the older “physical five-move run pending” notes as evidence of these trials. Independent distance, formal validation of the continuous evidence contract and physical activation of the continuous prototype remain pending. The next diagnostic is a cross-view test of the identified reference on saved images. [Study and reproduction](TARGET_GROUNDING_REVIEW.md)

## Continuous navigation handoff for Claude

This section records the earlier continuous-navigation publication.

`feature/vln-continuous-navigation` continues the first controller publication `93b08825c4eb702c1d8b9af4cc2df2a54dc3d712` in `$GO2_WORKSPACE/code/worktrees/codex-motion-diagnostics`. The stable five-move branch below is the physical-test baseline. Use the existing core and model environments. The single controller emits expiring packets only to the kinematic proxy.

**Measured feasibility.** The 150 ms visual fixture produced one forward handover while moving. In two 12 s simulated-camera runs, the pinned LightNav worker returned ten results, all indicating absence. The fixture chair was drawn in six of those inputs. The accepted motions were five policy-selected search turns. Their 0.175 s horizons expired before another result arrived. Both runs ended at the deadline, with zero moving replacements and zero forward travel. Saved Go2 forward jobs took 1.676 and 1.728 s to infer, compared with a conditional 0.500 s forward horizon. Both strict replays held at zero because the individual radio receipt clock was absent. [Results, limitations and commands](CONTINUOUS_NAVIGATION.md#what-the-measurements-establish)

**Offline verification.** The core suite passed 942 tests, with two optional MuJoCo skips and 19 hardware or model-integration cases deselected. All 67 focused cases passed. Both learned runs used the isolated model worker. [Test selection and commands](CONTINUOUS_NAVIGATION.md#regression-results)

**Review.** Use `scripts/run_vln_interactive.py --review-run RUN --port 0` in the core environment. It can follow an active offline run or scrub a closed trace. Under `$GO2_WORKSPACE/runs/continuous`, use `simulate-tz26_wqj` for the successful handover (case 0, tick 58) and unavailable continuation (case 7, tick 63), `simulate-8w_itx2j` for the learned worker, and `replay-bsgvnqxg` for the original recordings. [Viewer and artifact index](CONTINUOUS_NAVIGATION.md#private-artifacts)

**Physical evidence pending.** Claude's existing preparation record is `$GO2_WORKSPACE/evidence/five_move_pending_trial.md`. When the agreed baseline trial is complete, provide its exact commits and a private report using the published template. Include which motions were actually tested, independent distance where available, stopping/settling and release, camera/LiDAR continuity and boot, guard failures and radio receipt times, search/identity/arrival reviews, and unsupported capabilities. The verifier binds the report to local hashed artifacts and the closed five-move session. Its result is for human review. Physical activation requires a separate decision. [Schema and verification](CONTINUOUS_NAVIGATION.md#evidence-interface-for-claude)

## Baseline handoff for Claude — bounded object navigation, November 19

Claude’s validation checkout at `$GO2_WORKSPACE/code/navigation` was at five-move publication `918f48c7880d1ade20034a1ab37664fd3fa2188e` on `hardware/five-move-validation` when inspected. The published fallback branch is `feature/vln-five-move-navigation`. It builds on the published effective-motion diagnostics at `f444bc73432fa8401fe41b066a9392dfb5452c7d`, with the canonical controller at `6d555d3` and the SDK at `814556d`. The new episode runs only with `--five-move-episode`. [Operator guide and exact launch commands](FIVE_MOVE_NAVIGATION.md) · [Episode schema](../src/go2_perception/vln/schemas/five-move-episode-v1.schema.json) · [Synthetic chair fixture](../examples/five_move_navigation/synthetic_episode.json)

One area authorization permits up to five single-use turns and forward steps combined. The executor waits for verified settling and a fresh camera, pose and geometry sample after each movement. SEARCH can operate with an absent mark. APPROACH needs the named instance to be supported in the current image. The synthetic browser run found the illustrated chair, performed two turns and one forward approach, then recorded `MODEL_STOPPED_ARRIVAL_UNVERIFIED`. Live learned search and physical episode outcomes await Claude's trial.

**Smallest next trial.** In a dedicated hardware window, inspect a clear 2 m radius and use a distinctive chair just outside the starting view. Select two movements and a 45 s deadline, confirm the area once, and retain the outcome even if both slots go to turns. Record the initial absence, first supported mark, requested and settled primitives, guard and stop/release results, and the operator's assessment of approach or arrival. Keep raw records local. A tape measurement can support the separate displacement study. [Pilot procedure](FIVE_MOVE_NAVIGATION.md#smallest-physical-pilot-for-claude)

## Previous handoff — November 19 single-action diagnostics

Development branch `codex/motion-diagnostics` starts at navigation publication
`a1d7ad580b5ec23c3c5c1a40481b07e49718fb3c`. The local root is
`$GO2_WORKSPACE` (default `~/Robotics/Go2`). The canonical controller is
`code/control@6d555d3` and the SDK is `sdk/unitree_sdk2_python@814556d`.
The diagnostic worktree is `code/worktrees/codex-motion-diagnostics`.
Use the existing core environment for offline review. The [workspace layout](MAC_CONNECTION.md#local-workspace-since-2026-11-18)
and [operator guide](SUPERVISED_NAVIGATION.md) identify the live checkout and environments.

### Two counted forward runs

Both requested 0.600 m at 1.0 m/s with a 0.500 m feedback trigger. Both archived a
verified stop, release code 0 and stillness after release. The values below are sensor
estimates in metres. Neither run has an independent physical travel measurement.

| Private trial | Feedback at first zero | Settled odometry | Settled LiDAR | Settled fused | Fused minus odometry | Archived feedback outcome |
|---|---:|---:|---:|---:|---:|---|
| `trial-C3jMfl` | 0.522 | 0.604 | 0.700 | 0.700 | +0.096 | Within ±0.10 m, by 0.00018 m |
| `trial-abZbVx` | 0.522 | 0.616 | 0.791 | 0.791 | +0.175 | Overshoot by 0.191 m relative to request |

The recorded lateral displacement/yaw were +2.5 cm/+3.6° and +4.2 cm/+4.3°,
respectively. The accepted LiDAR value determined both settled fused values.
These two runs cannot establish which sensor was closer to the true endpoint.

**Diagnostic archive only.** The operator excluded the first run, `trial-DQAArT`,
because the robot had been moved after standing. Its selected extent was 0.566 m.
An in-motion geometry guard interrupted it at 0.146 m feedback. Settled odometry
was 0.213 m and fused progress was 0.275 m. The last passing guards showed
1.5–1.8 m clearance. The failing snapshot and post-dispatch raw scans were missing
from that recording. The cause remains unresolved. Keep the original under
`$GO2_WORKSPACE/runs/effective_trials`, outside performance totals.

**Headings.** Two observation-only sessions, `trial-6OyRDD` and `trial-2ktkjj`,
held rightward native prefixes around 0.345 and 0.306 rad (19.8° and 17.5°).
The single-action policy is HOLD for a turn-directed prefix above the 0.10 rad straight
limit and below 20°. Show the actual direction and available angle. The operator
may review a fresh stationary observation. A longer *contiguous*, same-direction
native prefix can qualify under existing turn, geometry and inspection gates.
The 2.0 rad/s profile has no independent small-turn validation.

**Camera timing.** The 0.159 s bound applies to the robot boot measured at
00:06 on November 19 and the recorded camera latency mapping. Evidence is
`$GO2_WORKSPACE/evidence/hw_readiness/capture_latency_20261119_000619.json`.
A different or unknown boot requires its own evidence.

### Optional measurements for a separate displacement trial

1. Keep the current profile and a fixed target surface. After the standing pose is
   settled, mark the floor projection of a repeatable rigid body reference and
   the initial forward axis. Measure the same reference when stationary after
   release. Record longitudinal and lateral displacement, heading change, instrument
   resolution and estimated measurement uncertainty. A tape start/end pair measures
   total travel only.
2. If available, use a fixed, scaled external video with a common timestamped
   cue to locate the first zero as well as the final pose. Record the alignment
   uncertainty. This separates motion before zero from coast. Record which
   surface supplied the front return, whether the box moved, and the floor/slope.
3. Save trial ID, request ID, boot/camera evidence ID and the private session.
   Join the guard and feedback events to the raw chunks by source stamps and
   host-monotonic receipt times. Check dropped chunks, bridge sequence gaps and
   writer drain status before interpreting gaps. Compare both sensor estimates
   with the external measurement before revisiting the 0.10 m stopping lead.

To inspect the first rejected guard and final raw/fused values locally, set
`session` to the private trial’s `session` directory and run:

```bash
jq -c 'select(.type=="result") | .result.events[]
  | select(.event=="guard_rejected" or (.event=="motion_feedback" and .phase=="settled"))' \
  "$session/executor/executor_events.jsonl"
```

The [diagnostic fields and recorder limits](SUPERVISED_NAVIGATION_CONTRACT.md#motion-diagnostic-records)
give the exact private file paths. Changes to the guard provider change its
hash. The ordinary launcher generates a fresh capability against the integrated
provider on the next authorized run. The [offline commands](SUPERVISED_NAVIGATION.md#reproduce-the-offline-analysis)
exercise the patch without using the hardware or model environments.

### Historical camera boots and readiness work, November 18

The evening 0.172 s bound belongs to `capture_latency_20261118_222418.json`.
The earlier 0.148 s bound belongs to `capture_latency_20261118_103131.json`.
`trial-HZbrPb` was held before claim while an iCloud-offloaded schema delayed
revalidation. `trial-qzUkyd` lost its executor during model loading. The project
then moved its runtime to the local Robotics workspace. The earlier one-step
result and controller-only measurements are described below.

## Go2 hardware readiness (Claude, 2026-11-18) — branch `feature/go2-hardware-readiness`

Report: [GO2_HARDWARE_READINESS](GO2_HARDWARE_READINESS.md) (section 8 = live results). Published at `24f12ca80605325e88c976e1af5849077a918a55`. Recordings/logs are local and git-ignored (`data/outputs/hw_readiness/`, `data/outputs/
vln_interactive/live_14`, `data/outputs/supervised/claude_one_step_0{1,2}` in the readiness worktree;
canonical control logs under `~/Desktop/Go2_Navigation-go2-control/data/outputs/go2_control/`).

- **First model-directed step done (`claude_one_step_02`)**: box dead centre ~2.5 m, operator ROI +
  one authorization → refreshed LightNav answer → forward 0.143 m at 0.30 m/s → 1.27 s of Move, stop
  still in 0.27 s, release 0, avoidance on; ~0.19 m actual (odometry 0.152 m, LiDAR ~0.2 m), heading
  −1.1°. Operator: clean small step. The capability (`hardware/make_capability.py`) passes the live check
  for **forward only**.
- **Live evidence**: emergency button while the API owns control freezes the robot standing
  (`error_code` 100 → 1002, cleared to 100); provider clearance tracks a placed box; capture-age bound
  0.148 s on this boot (`hardware/capture_latency.py`, screen flashes; boot-specific); correspondence on
  the box 20/24 same-view, 0/6 outside, 0/5 moved box.
- **Executor fixes** (tested): nominal `error_code` is 100 on this robot; the pre-claim stillness dwell
  now continues the idle stillness run (was restarting after the model answer → 3.02 s frame → HOLD).
- **Model behaviour**: STOP at ~1.1 m; the recorded off-centre cases gave turn+travel and the historical v1 planner held;
  answers 1.94–2.04 s after the frame once warm. The viewer/app must be launched from Claude's shell, not
  the Terminal panel: the panel-launched camera child received 0 RTP packets (likely macOS local-network
  permission).
- **Calibration**: measured intrinsics reused unchanged (sha `f5dedd82…`); live pipeline loads them and
  rectifies (fisheye, balance 0, no crop) then stretches to 448×256. `T_base_camera` stays null.
- **Forward mapping (verified for forward only)**: camera yaw +0.8° from the static wall capture
  (`hardware/camera_heading.py`), model yaw 0.0294 rad incl. the 0.85° image-centre ray, declared ±3°;
  model-frame origin **unknown** (`translation_m: null`, bound 0.5 m), which cancels for a pure translation
  and is charged as worst-case slack otherwise. Small contract change in `supervised.py` + schema
  (`mapping_slack`); null translation can never verify `turn`. Scale unverified, capped.
- **Turn**: only 30° heading targets at 1.0 rad/s are measured (→ 35.1° / 35.4°, coast 2.6–4.9°, ≤ 4 cm
  translation); 0.5 rad/s timed turns sprang back up to 8°. No smaller turn is claimed. `cloud_base` has
  0 returns in the rear of the r = 0.55 m swept disc (processing removal); the raw cloud has sparse
  rear returns to ~1 m (unvalidated); nothing beyond ~1.3 m behind (occlusion). Provider reports
  `turn_clear_rad = 0`. Next turn step is a controller-only, operator-inspected control test.
- **Correspondence gate offline**: stationary 21/24 valid, moved 0/18, outside-ROI 0/42.
- **Provider** `hardware/supervised_guards.py` (read-only); capability generator
  `hardware/make_capability.py` computes versions/hashes and names the evidence for each status.
- **Controller-only 15° right turn** (control test, `20261118_111433_move`): stop request at −16.7°,
  settled −17.4° (peak −19.4°, springback 2.0°), still 0.18 s, release 0; **translation 7.0 cm** (> 0.05 m
  turn limit), not explained by a fixed rotation centre. Operator: clean turn.
- **Faster caps (user request)**: control checkout `6d555d3` raises the utility to forward 1.0 m/s
  (distance target required above 0.30) and turn 2.0 rad/s. Controller-only: 1.0 m/s → peak 0.91 m/s,
  0.71 m odom for a 0.6 m stop, still 0.15 s; 30° at 2.0 rad/s → right −34.4° (6.0 cm), left +40.8°
  (1.0 cm); operator: all visible and clean. The effective integration now pins the updated controller.
- **Continuing work.** Use the integrated trial at the top of this handoff. Rear
  swept-disc coverage and the model origin are still open for general motion.
- **Mac**: ~/Desktop is iCloud-offloaded (disk 96 % full); hardware imports are local, recordings and
  the model env are being restored. A launch stalling at 0 % CPU means an offloaded file.

## Supervised navigation → Claude hardware handoff (2026-11-17)

Branch **`feature/vln-supervised-navigation`**, based on interactive
`fdb5451e25cdc1f5971bbf0e1c7887ec51ff634a`. Initial contract/schema/synthetic fixture
publication: **`2213cedae80d6eafca4a50eef93c6730c6c14dcc`**. Main, operational
read-only defaults, sensor/calibration environments and Claude's worktrees remain
unchanged. No hardware command or trained-model inference ran in this task.

Read first:

1. [SUPERVISED_NAVIGATION_CONTRACT](SUPERVISED_NAVIGATION_CONTRACT.md) — canonical v1.
2. [SUPERVISED_NAVIGATION](SUPERVISED_NAVIGATION.md) — actual implementation, states,
   exact mock/live launch commands and the capability/guard checklist.
3. [Schema](../src/go2_perception/vln/schemas/supervised-navigation-v1.schema.json)
   and [synthetic fixture](../examples/supervised_navigation/synthetic_cycle.json).
4. [VLN_CONTROL_HANDOFF](VLN_CONTROL_HANDOFF.md) and the pinned hardware branch's
   GO2_HIGH_LEVEL_CONTROL/GO2_NATIVE_AVOIDANCE_AUDIT — outstanding physical evidence.

**Implemented:** one reviewed target ROI/instance and bounded forward or turn;
consumed single-use authorization; one independent fresh model request; exact
instruction/epoch/pose/action checks; conservative image correspondence; isolated
executor with canonical lock, current guards, deadline, cancel, stationary dwell,
release and post-release measurement; new observation; concrete HOLDs. The existing
four panels/worker/calibrated preparation/recording are reused. The old read-only
launcher cannot accept navigation actions. The executor uses only the audited
obstacles_avoid route and never disables avoidance or falls back to another route.

**Verified:** the automated localhost mock cycle completed after a 3.25-second
review delay, with two mock requests, one synthetic 0.08 m segment, separate
stationary/release results and a newer observation. Local retained run:
`data/outputs/supervised/mock_cycle_07/` in this integration worktree. No private
Go2 recording was copied or changed. Earlier failed software checks are retained
alongside it and described in the guide. Initial regression: 156 passed; subsequent
focused integration: 60 passed. Final selected regression: **165 passed in 57.20 s**, 0 failed, 0 skipped.
Final executor + UI/schema follow-up: **25 passed in 10.25 s**
(0 failures/skips), including epoch changes and archive consistency.
Hardware/heavyweight tests were not selected.

**Subsequent hardware handoff.** Claude fulfilled the first-step evidence
collection and executed the first model-directed step. Refer to the current
handoff above for the effective trial, controller pin and command.

**Meaning of results:** `_CallNoReply` local send code zero is not an ACK; command
and stop robot acknowledgements remain null. Stationary, release and post-release
stationary are independent fields. Missing state/geometry, unknown capture timing,
changed target/pose/instruction/epoch, stale/expired/duplicate requests or cleanup
failure HOLD without retries. No dynamic obstacle, goal-success or autonomous
roaming claim. Hardware tests/full model regression remain unrun.

**Paired-view work is already complete separately:** evaluator branch
`feature/vln-viewpoint-evaluation` at
`1a0ac0e9f2cc0ca2688bbd7651383c90e86251fb` contains the recorder compatibility,
method-v2 importer and four real pair reports/handoff. It is not merged here and
must not delay the navigation interface. Its recognition findings do not validate
capture timing, motion safety or physical execution.

## Browser one-answer workflow + paired views → next agent (2026-11-17)

Same branch/worktree as below (`feature/vln-interactive-review`). Read
[VLN_INTERACTIVE_REVIEW](VLN_INTERACTIVE_REVIEW.md) sections "Step-by-step test" and
"Paired-view experiment" first.

- **Launch live** from a process with macOS Local Network access (not the app's terminal
  panel): the command in the doc, plus `--record-lidar-continuously` for paired views.
  Operators act only in the browser Test box; the terminal is a log; **Close tool** quits.
- **Records:** one trial per Ask. `predictions.jsonl` holds model output;
  `human_labels.jsonl` holds the operator verdict and verbatim comment, bound to the
  trial, prediction and image hash. `evaluation/paired_views.jsonl` (from
  `scripts/link_paired_views.py`) links a before/after pair with the measured turn
  interval and the post-turn check.
- **live_08** results are in PROJECT_STATE and the doc (6 tests, all CURRENT; operator
  agreed 5/6). Private run directories stay under `data/outputs/vln_interactive/`.
- **Do not:** enable the bounded-context policy, change the 3 s limit or the model, or
  connect waypoints to robot commands (a separate decision for the motion-control work).

## Interactive four-panel review tool → next agent (2026-11-16)

Branch **`feature/vln-interactive-review`** (worktree
`/Users/yushan/Desktop/Go2_Navigation-vln-interactive`), exact base `1f93d7e` from
`feature/vln-cached-live-validation`. Main and the motion-control PR are not merged or
imported. Read [VLN_INTERACTIVE_REVIEW](VLN_INTERACTIVE_REVIEW.md) first.

**What it is:** `scripts/run_vln_interactive.py` is a terminal plus a loopback-only browser
page with (A) the live camera, (B) the native target point on the exact model input,
(C) all native waypoints and headings in the model-local frame, and (D) a LiDAR top-down
view (latest / 0.5 s / 1.0 s, odometry-compensated `cloud_base`). Commands: `target`,
`instruction`, `mark`, `note`, `end`, `review`, `status`, `save`, `quit`, `help`. Each
trial is saved to a private, append-only run directory (images once by hash, LiDAR
chunks with source intervals, native outputs, marks and notes, confirmed human reviews
and click corrections in separate files). Reopen a run with
`--review-run <run>`. The model path is unchanged: existing `ShadowSession` +
`ProcessWorker`, full-history policy, cached preparation and the unchanged 3 s limit.
The bounded-context policy is **not** enabled. New opt-out: `--tensor-evidence skip`
(cached mode only; the default here; `run_vln_shadow.py` defaults unchanged).
LiDAR/odom come from a new read-only DDS-subscriber child
(`scripts/go2_lidar_bridge.py`, hardware env); robot-clock epochs and the per-epoch
host−robot estimate are display-only. There is no RGB–LiDAR fusion.

**Verified (no hardware, no model):** synthetic camera and LiDAR, a recorded public
LiDAR fixture and one private recorded episode, all with the mock worker. 22 new tests;
full core **702 passed / 0 failed / 11 environment skips**. App RSS over six short
synthetic trials: 98 → 111 → 112.5 MiB (flat after the first). Robot commands sent: 0.

**Live (2026-11-17, read-only, Go2 standing and stationary, operator-confirmed):** session
`live_03`, 4 trials and 19 LightNav requests (3 CURRENT, 12 STALE, 4 SUPERSEDED); camera
~7.1 Hz and `cloud_base` ~15.4 Hz with 0 gaps; app RSS ≤ 220 MiB; worker MPS driver memory
grew 8.9 → 11.8 GiB; disk ~0.45 GB/trial; 0 robot commands. Two operator-confirmed reviews:
the door target was correct; the chair target was wrong (it pointed at the shoe rack). The
bottle trial gave OPOS-0 and is unreviewed. Live launch needs `--unitree-sdk` and must run
from a process with macOS Local Network access (the app's terminal panel received nothing;
see the doc). Never connect native waypoints to any Go2 command.

## Cached preparation, live verification → next agent (2026-11-16)

Branch **`feature/vln-cached-live-validation`** (worktree
`/Users/yushan/Desktop/Go2_Navigation-vln-cached`), exact base `d173082` from
`feature/vln-live-latency`; docs only, no code change. Read
[VLN_CACHED_LIVE_VALIDATION](VLN_CACHED_LIVE_VALIDATION.md).

**Verified on hardware (stationary, read-only, operator-confirmed):** one bounded
`--preparation-mode cached` session, 3 requests, 0 robot commands. Jobs of 5/19/39
frames: prep 0.19/0.52/1.16 s, generate 3.0/4.0/6.5 s, estimated exposure→display
3.65/5.03/8.54 s — **all STALE** under the unchanged 3 s limit. Cache hits only 5 per
request (latest-anchored resampling), so live preparation is about the same as baseline;
no controlled speedup claimed. 0 inbox overflow/gaps/RTP loss; tensor archives 3/3
complete, normal shutdown. Display keeps latest-camera and model-input stamps separate.
Operator judged doorway targets (opos_501/502/504) correct — qualitative only.
Targeted tests 92 passed.

**Next:** freshness remains unresolved; native generation growing with history is the
dominant cost. Any further change (history policy, feature caching) is a separate,
explicitly approved decision. Do not connect waypoints to any Go2 command.

## Advisory latency → Claude / next agent (2026-11-16)

Branch **`feature/vln-live-latency`**, worktree
`/Users/yushan/Desktop/Go2_Navigation/data/outputs/vln-live-latency-worktree`, exact base
`eaec3314f4ed933623bd8aad1daca944d4575363` from `feature/vln-live-shadow`. Read
[VLN_LIVE_LATENCY](VLN_LIVE_LATENCY.md) first for contracts, all results, commands and
limitations; then [Claude's live validation](VLN_LIVE_SHADOW_VALIDATION.md).
Main, operational defaults, hardware/calibration environments and earlier results
are preserved. The hardware-control PR is not merged/imported.

Implemented **opt-in `--preparation-mode cached`**: hash-verified lossless calibrated
PNG reuse (128 entries / 256 MiB of cache links), compression level zero, and a
single bounded background tensor archive. Native model-space PNGs/text/prompt/decoder
outputs remain synchronous. Snapshots retain calibration and original timestamps;
archive status/errors and shutdown completion are explicit. Default `baseline`,
three-second threshold, receiver/stride/inbox, history policy, checkpoint/processor,
MPS BF16/SDPA and the same-dtype CPU pooling patch are unchanged.

**Actual verification:** after explicit operator resource coordination, six model
requests ran sequentially: recorded run 3's 5/17/37-frame jobs, baseline once then
cached once. All input tensors (BF16 bits included), history metadata, generated IDs,
pointing, waypoints and STOP flags match exactly; outputs also match the original
live results. Submit → display improved **2.942→2.779 / 4.358→4.005 / 8.305→7.388 s**.
These are fixed-cutoff offline profiles, not measured live latency. Fast replay
accepted one/rejected two in each pass; adding the original estimated exposure-to-
submit offsets gives 0/3 vs 1/3 under three seconds. Longer histories remain stale.
One sample/length/mode supports no percentile or robust variability claim. All
408 source-run files remained unchanged. Writer: 3/3 archives complete, no losses.

**Final tests:** 683 core passes / 0 failures / 9 skips in 42.29 s; 17 separate
native/numeric passes / 0 failures / 2 rendering skips in 2.62 s. The 20 new latency
tests cover invalidation, equivalent inputs, changing sampling prefixes, reset/
cancellation/staleness, bounded writing and shutdown/error reporting. Hardware opt-ins
were disabled and the physical test confirmed skipped before execution. No Go2
connection, DDS, ownership, stop or other physical command was used.

**Private artifacts:**
`/Users/yushan/Desktop/Go2_Navigation/data/outputs/vln_live_latency/profile_20261116_01/`.
`plan.json` predates inference; `baseline/` and `cached/` hold all requests and local
review pages; `comparison.json` has exact equivalence and timing; final test logs/XML
are `full_regression_final.*` and `optional_regression.*`. Initial fixture failure
and earlier passing suite are retained. The two `reconstructed_*_source/` directories
match every pre-inference source hash and permit exact source reconstruction on the
verified base; see their provenance notes. Existing isolated environments were reused;
the ignored `.venv` here only links to core-env. No installation or new weights.

**Next:** Claude may run the report's prepared three-request stationary **read-only**
command after a new operator-coordinated window. It explicitly enables cached mode;
this optimized path has not been tested live. Measure actual arrival/queue pressure,
archive completion and age; do not raise the threshold or connect outputs to control.
The pinned HF backend does not use its vLLM feature-cache path. Latest-anchored
sampling changes prior selections, so simply appending native frames is invalid.
No history truncation/reset shortcut was introduced. Real extrinsics, exposure
uncertainty and near-goal stopping limitations remain unresolved.

## Previous milestone: live read-only VLN advisory (2026-11-16)

Branch `feature/vln-live-shadow` (worktree `/Users/yushan/Desktop/Go2_Navigation-vln-live`),
based on advisory commit `b47f07d`, not main; the control PR was not merged or imported.
The existing read-only RTP camera receiver now feeds the advisory app through a
hardware-env child (`scripts/go2_camera_bridge.py`) and `vln/shadow_live.py`, which
produce the existing `Go2VLNObservation` for `ObservationInbox`. Launch with
`run_vln_shadow.py --go2-camera`. RGB-only: host-receipt stamps, estimated exposure
(receipt − 48 ms, single-boot), pose/LiDAR/`T_base_camera` None. Zero robot commands;
no DDS/SDK loaded.

**Verified on hardware (robot stationary, read-only):** real frames 720×1280 RGB with
measured fisheye calibration, unique IDs, 14.3 Hz decoded / 7.1 Hz forwarded, 0 RTP loss.
Three bounded LightNav sessions (model load 44.6–44.7 s). Sensor path ≈65 ms (48 ms of it
assumed); native generation 2.3–6.9 s, growing with instruction history. One request
was current at receipt (2.959 s); in steady state every proposal is **stale** under the
unchanged 3 s limit. Instruction change, cancel, history limit and clean shutdown observed
live; clock-step and interruption with mocks. Targets for doorway/refrigerator look
judged correct by the operator (qualitative, one scene). Tests: 15 new mocked tests; core 663 passed / 9 skipped.
Details: [VLN_LIVE_SHADOW_VALIDATION](VLN_LIVE_SHADOW_VALIDATION.md).

Next agent: do not connect native waypoints to any Go2 command. Open questions are
freshness (generation grows with full-history rebuild; a separate decision, not started),
re-measuring camera latency on the current boot, and broader (multi-scene) qualitative review.
Private runs: `/Users/yushan/Desktop/Go2_Navigation/data/outputs/vln_live/` (git-ignored).

## Read-only VLN advisory → Claude / next agent (2026-11-15)

Branch **`feature/vln-shadow-demo`**, worktree
`/Users/yushan/Desktop/Go2_Navigation/data/outputs/vln-shadow-worktree`, exact base
`1cc74909e489e1037c9f6f0051ab34e855d144dc`. Use `git rev-parse HEAD` for the publication
commit. Main, previous experiments and operational defaults are preserved.

Read first: [VLN_SHADOW_DEMO](VLN_SHADOW_DEMO.md),
[actual verification](VLN_SHADOW_VERIFICATION.md),
[VLN_CONTROL_HANDOFF](VLN_CONTROL_HANDOFF.md), then the existing
[observation contract](GO2_VLN_INPUT_INTERFACE.md) and
[native contract audit](WAYPOINT_EXECUTION_AUDIT.md).

One entry point: `scripts/run_vln_shadow.py`. It accepts terminal instruction plus
existing `model_input/episode.json` or deterministic synthetic images and creates a
private directory containing refreshing `index.html`, PNG/status/summary, append-only
events, original observation NPZ files, immutable native jobs/snapshots and receipts.
The internal `vln_shadow_worker.py` reuses existing LightNav `run_job` in the isolated
model environment. It loads once, resets/rebuilds the complete causal instruction
history per request, and runs official preprocessing/SlowFast/native decoding.
It receives no annotations, poses, LiDAR, action feedback or goal truth.

Session behavior is in `vln/shadow.py`: disk-backed bounded history, one in-flight
request, one coalesced dirty flag, strict instruction/epoch/generation/sequence and
monotonic-time checks. New instructions clear history and begin with the next actual
frame; no old frame is restamped. Native episode anchors are retained until a limit
or gap explicitly pauses inference. Late cancelled results are historical, never
current. `shadow_display.py` separates the latest camera from actual model inputs,
plots all native waypoints/yaw in their own unregistered local frame, and distinguishes
native STOP, model completion claim and unknown verified/physical completion.

**No physical command path exists.** There is no `--live`, `--execute`, `--arm`,
motion client, avoidance toggle, command publisher or simulator execution backend.
Even zero velocity is not sent. Cancelling means advisory cancellation only.
`shadow_sources.ObservationInbox` is the future read-only consumer; it opens no
connection and preserves the existing `Go2VLNObservation` payload. See the guide for
availability domains, clock-epoch/uncertainty requirements and overflow handling.

**Initial lightweight verification:** 114 selected tests passed in 1.56 s (57 new advisory
cases, 57 existing contracts), no skips in this set. Synthetic paced mocked demos
accepted 3/3 proposals; a deliberately strict age test rejected 2/2 stale proposals.
Private E1 package replay used 24 existing images with calibrated rectification and
two mocked proposals; all files remained local. Ruff/diff checks pass. Tests are
reproduced with core-env Python and `PYTHONPATH=src`, selecting `test_vln_shadow.py`,
`test_pretrained_vln.py`, `test_vln_episodes.py`.

**Operator-authorized verification, completed:** tested implementation
`b3c3dffdaf1a663f554b29d8dc2ca19aa1460e8d` ran one real LightNav replay, then the full
core suite, then the optional native/numeric suite, sequentially. Five causal,
calibrated inputs produced ten valid native waypoints. Native generation took
2.444501 s; the child interval took 2.851785 s; paced observation age at receipt was
3.544899 s. The unchanged 3 s advisory limit correctly rejected it as STALE: zero
fresh accepted, one rejected. These are replay timings, not measured live latency.
No extra real call or relaxed threshold was used. Core: **648 passed, 0 failed,
9 skipped** in 44.73 s. Separate environment: **17 passed, 0 failed, 2 skipped** in
2.68 s. Physical Go2 and two camera-rendering tests remain unexecuted; environment
specific core skips were subsequently covered. See the verification report for
exact selection, skip reasons and commands.

The operator confirmed Go2 stationary and physical testing paused/finished with
stop/release cleanup complete. No Go2 connection, DDS, control utility or robot
command was used. Existing `phase6c/core-env` and `phase6c/model-env` were reused
unchanged. No new model or dependency was installed. The worktree's ignored `.venv`
is only a symlink to core-env for existing subprocess tests. Live sensor connection
remains pending. Re-coordinate before heavy work if physical testing resumes.

Artifacts at `/Users/yushan/Desktop/Go2_Navigation/data/outputs/vln_shadow/`:
`synthetic_paced_v1/`, `synthetic_paced_v2/`, `synthetic_stale_v1/`,
`private_e1_mock_v1/`; logs and selected test XML under `audit/`. Each run has its
exact arguments in `run.json`. Use fresh output names for reproduction. Private
frames, scene logs, videos and weights stay ignored. Only the generated synthetic
`docs/assets/vln_shadow_synthetic.png` is published.

The final-code demo is `published_demo/` (32 frames, three mock proposals). Its
`run.json` also hashes the exact application/native-worker source and runtime config.
The input audit verifies all 107 E1 images and calibration against Claude's unchanged
manifest; annotations were not loaded. See `audit/input_and_environment_audit.json`.

The real verification is in the fresh ignored `verification_20261115_01/` under that
artifact root. Open `real_model_smoke_01/index.html`; original outputs/tensors are
under `real_model_smoke_01/requests/000001/native/`. `audit/` contains complete test
logs/XML, metrics and causal/calibration/preservation checks. All 107 E1 images,
manifest and calibration still match; every prepared frame matches the existing
calibrated preprocessing exactly. Private imagery remains local and unpublished.

Claude's published `GO2_HIGH_LEVEL_CONTROL.md` and `GO2_NATIVE_AVOIDANCE_AUDIT.md`
were read at `742cfe4fb1b71f9d19d9d2d62be56f03b6b03707` in
`Go2_Navigation-go2-control`. [VLN_CONTROL_HANDOFF](VLN_CONTROL_HANDOFF.md) links the
pinned reports and attributes hardware findings to Claude/operator. Reported
avoidance-service velocity commands and routine stop are distinct from unverified
remote override under API ownership, obstacle stopping and robot firmware. Native
absolute waypoints are not velocity commands. No control code was imported, executed
or merged. **Manufacturer obstacle avoidance must remain enabled.**

Next: a separately coordinated read-only receiver → `ObservationInbox` connection;
validate clock epochs, exposure/receipt semantics, host-monotonic mapping and actual
display age. The smoke produced no fresh advisory at the default limit, so live
freshness remains unestablished. Real `T_base_camera` and near-goal stopping are
unresolved requirements for future metric/physical use, not blockers to an explicitly
unregistered RGB-only advisory display. Never substitute identity poses or connect
model output to the control utility.

## Phase 6D → next agent: measured waypoint validation (2026-11-15)

Worktree `/Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6d-worktree`, branch
`phase6-waypoint-validation`, exact base `45154f6`. Main and all previous experiments
remain intact. Read [WAYPOINT_EXECUTION_AUDIT](WAYPOINT_EXECUTION_AUDIT.md) then
[PHASE6D_NAVIGATION_EXPERIMENTS](PHASE6D_NAVIGATION_EXPERIMENTS.md) for contracts,
limits, tests, artifact paths and reproduction commands.

Implemented: opt-in fixed-frame bounded-prefix simulation follower; unchanged
first-waypoint baseline exposed with complete traces; append-only native history
resetting on episode/epoch boundaries; reviewed/hashed experiment declarations;
simulation-only child runners with hard host limits; separate reaching, stopping
and external-termination metrics; full native/selected/executed path panels/videos.
The old replay worker defaults and operational defaults are unchanged.

Claude's review at `4a168e165e8eee58b7fcd4fc5a192e9791d3c278` is read from his branch,
not merged or edited. Human-recognizable doorway supports keeping the original scene.
Eight distinct native pixel tensors at old steps 1–8 rule out identical saved inputs
as the explanation for repeated outputs; HF bypasses the episode ViT cache. No
semantic/model root cause or controller defect is proven by these checks.

Actual verification: **591 core passes, nine genuine skips**; 19 optional native/simulator
cases passed across separate runs. Eight successful known-path/stop cases, one
intentional contact rejection and one reachable-goal controller diagnostic are saved
in `data/outputs/phase6d/waypoint_validation/controller_diagnostics_v1/`. These are
**not learned navigation**. Separately, the full model-driven suite **ran all eight
conditions once**, 121 valid native decodes. Both policies succeed by the declared
native-STOP-inside-goal criterion for ahead/left/right starts. Both initially-near
cases fail: baseline contact-constraint termination without STOP; prefix native STOP
after leaving the goal and passing through the doorway. No goal override, retried
condition, budget extension or private recording substitution. All new behavior
human review remains pending. Do not reinterpret 6/8 on these correlated synthetic
conditions as broad success, or initial goal presence as a completed approach.

**Execution code commit:** `d1590eb0c63046b3899325b70d6908e88a59e453`; published before
results. Later publication updates results/plots and fixes only an evaluator label:
an outside STOP after goal exit was incorrectly called premature. The correct v2
rescore changes that single flag for near/prefix; paths and success counts are
unchanged. Original summaries/declaration are intact, and the corrected source no
longer matches their frozen source hash. Use the execution commit for exact original
source replay, or freeze a new declaration with identical reviewed settings for the
corrected scorer. Do not weaken the implementation-hash check.

**Exact artifacts under our root:**
- `frozen_suite_v1.json`, SHA256 `0c26c9678692955de374a52d5e5f5c7c29bcc33a72778a3d7b97bd7f96f34f4e`;
  reviewed source settings in `reviewed_suite_v1.yaml`.
- `suite_v1/<ahead|left|right|near>__<first_waypoint|bounded_prefix>/rollout/`:
  raw images/jobs, per-decision `inference/NNN/` native outputs/tensors and state,
  actual path traces, unmodified first-version summaries. Per-condition logs/model
  identities are one directory above; original aggregate is `suite_v1/aggregate.json`.
- **Use `suite_v1_rescored_v2.json`** for corrected stopping metrics; it binds raw
  hashes and records the only semantic correction. Tiny distance rounding differences
  from NumPy 1.26 vs 2.4 are <3e-16 m and change no outcome.
- `suite_v1_review_v2/index.json`: all eight 1920×1080 H.264 videos (129 panels),
  including both failures. Representative success: `ahead__first_waypoint/rollout.mp4`;
  failures: `near__first_waypoint/rollout.mp4`, `near__bounded_prefix/rollout.mp4`.
- `audit/`: 591-pass final log/XML; separate optional-case results; original tensor
  repetition and same-image/different-instruction comparison; runtime inventory;
  before/after preservation snapshots; video verification. All 727 package/baseline
  files plus 273 old-simulation/review files match.

**Where the code lives:** `vln/waypoints.py` owns fixed-frame bounded interpolation;
`simulation.py` adds opt-in execution without changing `advance()`; `native_history.py`
owns causal append/reset semantics; `waypoint_rollout.py` separates model jobs and
evaluator state; `waypoint_declaration.py` validates review/code/scene bindings;
`waypoint_evaluation.py` distinguishes entering/stopping/termination.
`scripts/run_waypoint_suite.py` freezes and executes all conditions in bounded child
processes; `summarize_waypoint_suite.py` audits/rescores without inference;
`review_waypoint_validation.py` produces evaluation-only panels/videos.

Next: human-review the **new** close-up/late-stop/contact failures, then declare a
small fixed-checkpoint view/history ablation. Do not infer a camera-domain root cause
or tune the current development cases into a benchmark. Preserve Claude's earlier
review and the old E4 prompt/reference mismatch. Reuse `phase6c/model-env` and
`phase6c/core-env`; `.venv` here is only an ignored
alias to core-env. Do not install anything into Claude's environments. Real
`T_base_camera` remains missing. Physical execution disabled on every new path.

Artifact root `/Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6d/waypoint_validation/`
is ours; Claude owns sibling `panels/`, `human_review/` and `MANIFEST.json`. All remain
ignored. Never overwrite those siblings, Phase 6B packages/baselines or the Phase 6C
failed rollout. Both pre-run and post-run preservation snapshots match.

Known environment issues/reproduction: restricted sandbox ffmpeg/ffprobe can fail
with OpenMP shared-memory error 179; run the existing video/full-suite commands
outside that sandbox, without replacing dependencies. A combined optional run
stalled in a Transformers file read after 18 passes; its remaining MPS pooling case
passed separately. Retain that interrupted log/XML, not an invented combined pass.
No new code defect remains known from the completed tests. Real `T_base_camera`,
live robot validation and Windows/Isaac remain outside this verified milestone.

## Phase 6C → Claude / next agent: navigation-trained diagnostic (2026-11-13)

Implementation commit: [`0a442f4984c98578f32c5f406bcdfbac01b150a2`](https://github.com/binbin-ovo/Go2_Navigation/commit/0a442f4984c98578f32c5f406bcdfbac01b150a2).
Later documentation clarification distinguishes stored checkpoint elements from unique
parameters after embedding/head tying; inference artifacts are unchanged.

**Branch:** `phase6-pretrained-vln`, separate worktree
`/Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6c-worktree`, based exactly on
`7e2989ad54b9c484d75c029ff0141834880852b6`. Use `git rev-parse HEAD` to identify the
checked-out publication. No main merge or operational-default change. Original
checkouts, private recordings/annotations, calibration environment, Phase 5A.1,
Grounding DINO and saved SmolVLM results are not overwritten.

**Read first:** [PRETRAINED_VLN_FEASIBILITY](PRETRAINED_VLN_FEASIBILITY.md),
[PRETRAINED_VLN_EXPERIMENTS](PRETRAINED_VLN_EXPERIMENTS.md),
[WINDOWS_SIMULATION_FEASIBILITY](WINDOWS_SIMULATION_FEASIBILITY.md), then the existing
[episode contract](VLN_EPISODE_INTERFACE.md). The experiment report contains exact
installation, inference, evaluation, simulation, tests and preservation commands.

**What Codex implemented:** pinned-source/checkpoint preparation, read-only Mac/Windows
inventory, isolated local LightNav inference worker, causal native-history bridge,
strict RVQ/runtime contracts, fixed diagnostic action mapping, evaluator-only
SmolVLM/human comparison panels, and an optional synthetic MuJoCo camera proxy.
These extend existing calibrated views, episode loaders and review-video tools;
sensor/tracker internals were not rewritten. No upstream real-robot client, network
motion interface or execution-authorized path exists in these additions.

**Actual model/runtime:** LightOriginsHQ/LightNav-0
`826dc5fbfa37afa8293d2e336d329b6ffc0bfb64`; code
`3015508b70fb6e30bbd66701e94c9b916dcb8df8`; Apple M3 Pro, MPS BF16/SDPA, no quantization.
Use **two new environments**: `phase6c/core-env` (Phase 6B-constrained Python 3.12)
and `phase6c/model-env` (Python 3.11, Torch 2.10, Transformers 5.8, NumPy 2.4).
The ignored `.venv` alias in this new worktree points at the new `core-env` because an
existing viewer test requires that relative launcher. It does not point to or alter
Claude's environment. Both new environments pass dependency checks.
Do not install the latter stack into recording/calibration/perception environments.
The source is private under `phase6c/lightnav/`: pristine `source`, first `source-local`
fork and final `source-local-v2` all retained. Final patch SHA256
`bec34b7d855f33ac94ec212a61ca507d2d677d82441add0cd1b2dfe7088e554c`; changes are device
selection/timing and exact BF16 CPU fallback for unsupported non-divisible MPS pooling.
`scripts/patch_lightnav_local_runtime.py` recreates it from verified pinned source.

**API and frames:** `pretrained_inputs.prepare_job(episode, decision_id, destination)`
creates only instruction/identity/time and causal rectified images. `pretrained.causal_history`
uses actual exposure plus availability, a 4 Hz grid and native SlowFast afterwards.
Raw fisheye, rectified camera and official 448×256 model-space metadata remain distinct.
Worker jobs whitelist fields and hash images. The native processor/template/normalization
and RVQ decoder remain official. Native ten-waypoint chunks are absolute ego SE(2),
metres/radians, x forward/y left/yaw CCW, **without a time base**. `T_base_camera` is
still unknown; do not treat these as Go2 base-frame commands or fabricate registration.

The policy requires RGB history, not prior executed actions. It resets native buffers,
vision cache and engine state before every decision/episode/epoch, then reconstructs
causal observations. Never feed a proposal back as executed while following an operator
recording. `SimulationRequest` uses a separate host-monotonic deadline and rejects
wrong episode/epoch/sequence or stale results. The local static simulator applies the
first native waypoint over an explicitly chosen 0.25 simulated seconds with limits;
its geometry/contact/goal truth remains evaluator-only.

**Inputs and private artifacts:** use the exact Claude packages under
`/Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6b/episodes/`; do not rebuild,
change instructions/cutoffs or relabel. Their model/evaluation separation is unchanged.
Original baseline: `phase6b/claude_pilot_v1/`. New artifact root:
`/Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6c/`:

- `lightnav/`: hashed code/checkpoint/processor/RVQ assets; weights stay out of Git.
- `claude_pilot_v1/`: preparation/launcher failure retained; `claude_pilot_v2/`: five
  successful calls and native MPS pooling failure retained.
- `claude_pilot_v3/`: complete 11-call inference, native outputs/tensors/memory/latency.
  It reuses v2's exact hashed jobs/prepared inputs; `run_declaration.json` records that
  provenance, unchanged manifests, preparation code and new worker code identity.
- `claude_pilot_v3_review_v2/`: final evaluation-only panels, all actual native historical
  input images, video, human references, full chunks and baseline comparisons. Original
  review remains as well; only plot scale/yaw display was improved, without inference
  reruns or score changes.
- `sim_closed_loop_v1/`: one actual failed 16-step static rollout; full camera/native
  proposal/actual action/evaluator trace. `sim_closed_loop_v1_review_full_chunks/` contains
  all 16 panels and video, including the failure.
- `audit/`: official failure/success logs, tests, hardware and source-preservation hashes.

**Validation:** full core suite **541 passed, 4 skipped in 41.58 s**, including actual
cached SmolVLM inference; all five optional native/MuJoCo cases passed in separate
model-environment runs. Only physical Go2 remains unexecuted across these environments.
63 new core cases include configuration/worker identity and premature-EOF hashing
regressions. Existing tests were not modified. See `audit/full_regression_verified.log`.

**Preservation audit caveat:** all 434 package files and 292 baseline files match the
original preflight. One baseline PNG had an invalid empty-file hash from a transient
short read. It was not rewritten: two fresh readers and an in-memory run of its original
rectification/PNG producer agree on the intact bytes. Keep the failed inventory and
`baseline_short_read_recovery.json`; do not pretend the original digest was valid.
Two strict post-run snapshots agree on all 727 files. Use
`audit/inputs_final_verified.json` for subsequent comparisons, not the flawed initial
`inputs_before_verified.json`. The auditor now rejects partial/unstable reads and
saves mismatch reports instead of silently accepting a truncated digest.

**Results and limits:** native RVQ valid 11/11; fixed discrete comparison coverage 9/11,
five matches/four mismatches/two unsupported mixed chunks. E1–E3 on the same seven
mapped cutoffs: LightNav 3/7 vs Smol temporal 4/7 and current-only 5/7. Initial yaw sign
follows right/left, but no STOP on any cutoff; no overall gain established. E4 outputs
a native not-visible token and search turn, not verified global absence or task success.
Historical E4 UNRESOLVED/search-turn mismatch is preserved. All examples are one
correlated test_c group; new semantic/evidence review remains pending.

**Simulation result:** renderer and model can coexist in this tiny Mac case. The
16-step model rollout turns too long, then moves away, exhausting its budget without
stopping (distance 1.850 → 2.722 m). No contacts; no learned avoidance or Go2 locomotion
claim. No latency-aware/dynamic trial. Windows remains inaccessible: run the PowerShell
preflight before assessing its existing Isaac version. Its reported 8 GB is below
LightNav's unquantized weight size; do not assume BF16/FlashAttention-2/quantization
support or shared model+Isaac capacity. No driver/WSL/install change was made.

**Next steps, in order:** independently review pointing/chunk/stopping failures using
the local panels, design diverse reviewed episodes outside test_c, run the Windows
inventory on the actual laptop, then inspect that installed Isaac version/checker.
Any subsequent scene/model experiment should use a new output folder and declared
conditions. Preserve this failed rollout and benchmark. Resolve measured camera/base
registration, timing and navigation behavior before considering physical integration;
physical execution stays disabled. Current tests and reproduction details are in the
experiment report. A local file-I/O issue caused stalls/transient read errors; successful
retries and failures are recorded, with no private-source substitution.

## Phase 6B → next agent: Claude-package evaluation (2026-11-13)

Evaluation implementation commit: [`4d5e49642da03558830dba017f5aa150d9fdc684`](https://github.com/binbin-ovo/Go2_Navigation/commit/4d5e49642da03558830dba017f5aa150d9fdc684).

**Branch:** `phase6-temporal-vln`; follow-up to `d2112268b1863185d39f272a10d5bb80cbefaaaa`.
Claude's packages are from `phase6-vln-episode-validation` commit
`3a84f57b8753b0e5c169f53ae361a33f9ddbc3fe`. His files/checkout were not modified or
merged. Main remains unchanged. No physical robot connection or executable commands.

**Read first:** [TEMPORAL_VLN_CLAUDE_PILOT](TEMPORAL_VLN_CLAUDE_PILOT.md) for exact paths,
commands, actual scores and caveats; [VLN_EPISODE_INTERFACE](VLN_EPISODE_INTERFACE.md)
for the stable contract. The next section below documents the older Codex-authored
pilot and retains its separate pending-label status.

**Inputs used, unchanged:** root
`/Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6b/episodes/`, packages
`p6b_test_c_e1_turn_right`, `p6b_test_c_e2_turn_left`,
`p6b_test_c_e3_forward_bottles`, `p6b_test_c_e4_absent_sofa`.
Each exact manifest is `<package>/model_input/episode.json`; corresponding human
references are `<package>/evaluation_only/annotations.json`. Review logs, provenance,
odometry slices and label panels remain evaluator-only. No builder was run; no labels
or cutoffs were changed. All supplied hashes matched and all 434 package files were
verified unchanged after inference. Keep the declared 0.050 s decode margin.

**Review provenance:** all 11 action references are reviewed human labels, one repository
owner accepting Claude's drafts unchanged. Seven are marked ambiguous. There are two
complete / six incomplete / three unknown completion references. Do not reclassify
unknown as incomplete, substitute agent labels, or call these independent blind labels.
E1–E3 instructions were written retrospectively after viewing the operator route.
E4 is a static absent-target probe. All share test_c, overlap earlier pilot footage,
and must remain a single correlated group. No similar-target case exists here.

**Actual execution/results:** same SmolVLM2-2.2B checkpoint/processor revision
`be92141c8d0ceb98596a923c308ac2bbb4a7c466`, CPU FP32/eager, frozen prompt/parser/input
pipeline. 11 cutoffs × two modes = 22 calls. Every temporal encoder call used four
images; message/provenance audit confirmed only instruction plus causal observations
entered the model. Human action agreement: E1–E3 5/9 temporal vs 7/9 current-only;
E4 2/2 vs 0/2 (two malformed current outputs rejected); overall 7/11 each. No history
advantage is established. Both modes miss stopping references; the one temporal
COMPLETE claim is unscorable because that human completion reference is unknown.
E4's accepted search-turn sets also differ from the frozen absent-target UNRESOLVED
prompt policy; neither labels nor prompt were changed to improve scores.

**Code/API changes:** `scripts/run_temporal_vln.py --episode PATH` can now be repeated
for exact nested packages, avoiding automatic discovery of neighboring old episodes.
`scripts/evaluate_temporal_vln.py --run RUN --annotations-name ../evaluation_only/annotations.json
--static-episode p6b_test_c_e4_absent_sofa --output NEW_JSON` evaluates saved proposals
without model loading. It validates input/manifest binding, records annotation hashes,
keeps recordings grouped and static cases separate, and excludes missing/pending/agent
references from human denominators. It does not alter source data or inference files.
Unknown reference completion and malformed model statuses do not enter completion
consistency. Free-text progress consistency still requires separate adjudication.

**Private artifacts:**
`/Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6b/claude_pilot_v1/`.
Start at `evaluation_grouped.json` and `review.mp4` (22 s, 1650×990). Every per-call
folder retains native output, exact tensors/prompt, frame provenance and labeled panel.
Sibling `audit/claude_pilot_v1_preflight.json` / `claude_pilot_v1_postflight.json` prove
source integrity and causal inputs. Original `evaluation.json` is retained; the grouped
report removes inapplicable sequence flags from ambiguity-filtered subsets, with all
action scores unchanged. No inference was rerun for that reporting correction.

**Validation:** full suite with SMOLVLM_MODEL_CACHE set: **478 passed, one hardware test
skipped**; all four modified/new Python files pass Ruff check/format. Log:
`audit/claude_pilot_full_tests_final.log`. The earlier two model tests still execute
real weights. Normal tests use synthetic fixtures. The isolated environment and
pinned dependencies are unchanged; use the existing installation in the demo guide.

**Next:** do not rebuild/relabel this supplied pilot or tune against it as held-out
validation. Plan a separate experiment on independently reviewed diverse recordings
with explicit completion conditions. Preserve the one-recording grouping, human draft
acceptance provenance and static E4 separation in downstream comparisons. Registration,
metric depth, timing/odometry and traversability remain unresolved; controlled closed-loop
validation needs separate authorization. Keep every proposal execution_authorized=false.

## Phase 6B → Claude / next agent: temporal instruction-following pilot

Implementation commit: [`e9fa7a99c0182af150ffaccc58a83ef54f1a6b0d`](https://github.com/binbin-ovo/Go2_Navigation/commit/e9fa7a99c0182af150ffaccc58a83ef54f1a6b0d).

**Branch:** `phase6-temporal-vln`, verified base
`c13f1dc442fa157b31043613d3cb2b0c30ceab8a`. Original checkout, Claude's calibration
environment, Phase 5A.1 and main were left unchanged. No robot connection or commands.
Implementation is local offline code plus a completed private pilot; human semantic
validation and navigation success are not complete.

**Read first:** [TEMPORAL_VLN_EXPERIMENTS](TEMPORAL_VLN_EXPERIMENTS.md) for actual
failures/measurements, [TEMPORAL_VLN_ARCHITECTURE](TEMPORAL_VLN_ARCHITECTURE.md) for
model/input/output semantics, [TEMPORAL_VLN_DEMO](TEMPORAL_VLN_DEMO.md) for exact commands,
and [VLN_EPISODE_INTERFACE](VLN_EPISODE_INTERFACE.md) before preparing new episodes.
The Phase 6A and earlier handoffs below retain their original historical scope.

### Published episode contract — stable v1

Contract commit: **`5c7f586aa2536a8eda9167e66d5050a4a91dbac2`**, pushed before model
implementation. Fixture: `tests/fixtures/vln/episode_v1/episode.json`; separate
annotations: `tests/fixtures/vln/episode_v1/annotations.json`. Schemas:
`src/go2_perception/vln/schemas/`; loader: `src/go2_perception/vln/episodes.py`.
These bytes/schema remain compatible with the first publication. Fixture generator:
`scripts/generate_vln_episode_fixture.py`; inspect with:

```bash
python scripts/inspect_vln_episode.py tests/fixtures/vln/episode_v1/episode.json --decision d0
```

One episode = one recording and boot/clock epoch. Preserve exposure versus availability,
ordered images, explicit bounds/gaps and image/calibration hashes. Camera optical axes
right/down/forward; base_link forward/left/up. T_base_camera remains missing. Never use
identity as a placeholder or treat floor-seen cells as safe free space. Source timestamps
are seconds with documented clock semantics; runtime host times remain separate.

### Implemented and exercised by Codex

- `vln/temporal_inputs.py`: causal four-frame/six-second policy, independent Phase 6A
  calibrated rectification per image, official SmolVLM multi-image template/processor,
  resized K/FOV/crop/padding metadata and exact tensors/hashes.
- `vln/temporal_model.py`: one pinned SmolVLM2-2.2B checkpoint, CPU FP32/eager, deterministic
  greedy inference; vision-encoder hook verifies every historical image enters the model.
- `vln/temporal_proposals.py`: native text retained, strict JSON/anchored native field
  parsing; action separate from task status, unknown frame/quantities preserved.
- `vln/temporal_review.py`: current/history panels, native responses, labeled evaluation,
  review video and metrics restricted to reviewed human references. Prose claims need
  separate review; structured COMPLETE counts do not cover them.
- `scripts/prepare_temporal_vln_pilot.py`, `scripts/run_temporal_vln.py`: deterministic
  private packages and immutable run directories; configs `temporal_vln*.yaml` / prompt.

**Actual model:** `HuggingFaceTB/SmolVLM2-2.2B-Instruct`, processor and checkpoint
`be92141c8d0ceb98596a923c308ac2bbb4a7c466`. Generic **VLM navigation-proposal baseline**,
not a reproduced VLN policy. No keyword navigation parser or training. Four episodes,
eight decisions, 16 temporal/single-image calls, 16 valid structured responses.
Every temporal call encoded [4,3,384,384] images / 324 image tokens; single calls
encoded [1,3,384,384] / 81. No label data entered prompts or inference.

**Important failure:** temporal always TURN_RIGHT (8/8), including left instruction;
single outputs 4 right, 2 left, 2 unresolved. One native reason falsely implies task
completion while structured status remains IN_PROGRESS. Human references are pending.
Six panels/all native texts were agent-reviewed with explicit reviewer type; no human
semantic score or closed-loop success is claimed. Frozen input/model/parser/prompt
hashes are in the experiment report; do not silently retune and reuse this pilot as
held-out evaluation. Generation mean 10.171/6.221 s temporal/single; process peak 9.684 GiB.

### Installation, tests and private evidence

Separate worktree: `/Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6b-worktree`.
Private evidence: `/Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6b/`.
This persistent location is independent of `/private/tmp`; all media and weights are
Git-ignored. Source recordings were copied/read, not modified.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -c requirements-temporal-vln-tested.txt \
  -e '.[test,temporal-vln]' av==17.1.0 ruff==0.11.13
export PYTHONPATH=src
ARTIFACTS=/Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6b
SMOLVLM_MODEL_CACHE="$ARTIFACTS/hf_cache" HF_HUB_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false MPLCONFIGDIR="$ARTIFACTS/mpl" \
  .venv/bin/python -m pytest -q -rs
```

**470 passed, 1 hardware skipped, 36.12 s** including both actual model tests. Without
SMOLVLM_MODEL_CACHE, the two heavyweight model tests skip; no automatic model download.
Ruff check/format passed on 12 Phase 6B files. Grounding DINO compatibility was exercised
on a real saved image and exactly reproduced the old candidate. Full constraints record
this Mac/Python environment, not Linux/CUDA verification. Do not install the older
`vln` extra and `temporal-vln` simultaneously (different Transformers pins).
If a Mac editable `.pth` is hidden, `PYTHONPATH=src` works; the demo documents the isolated
flag fix. No remaining installation blocker in the tested environment.

Start demo review at `pilot_v1/review.mp4` (16 s, one panel per call, not robot timing).
Successful literal-direction example: `pilot_v1/turn_to_racket/d00/current_only/review.png`;
failed opposite direction: same episode/decision `temporal/review.png`; unresolved:
`visible_refrigerator/d00/current_only/review.png`; ambiguous referent:
`similar_cabinets/d00/temporal/review.png`; unsupported completion prose:
`absent_red_door/d00/temporal/review.png`. Review findings are agent-only in
`audit/agent_review.json`. `audit/full_tests_with_model.log` records the final suite.

### Exact next steps / ownership

1. Claude/human: prepare private real episodes using the published v1 contract.
   Existing `episodes/*/annotations.pending.json` are empty human-review placeholders.
   Write separate reviewed annotation files with reviewer identity/status and binding
   manifest SHA. Do not change an evaluated manifest or insert labels into prompts.
2. Score saved outputs using `temporal_review.summarize` with validated references
   (example in the demo); human instruction-progress and prose-completion consistency
   need separate adjudication. Do not count agent notes as human ground truth.
3. Collect held-out diverse turn/landmark, ambiguous and absent-target episodes.
   Diagnose right-turn bias in a separately named, predeclared development run;
   checkpoint/input/prompt changes must get new output directories and explicit scope.
4. Claude owns hardware/calibration/timing work. Missing T_base_camera and registered
   depth remain blockers for metric goals; moving odometry and traversability still
   require validation. No metric goal or action reference is invented here.
5. Only a separately authorized closed-loop experiment can assess goal reaching,
   collisions, safe execution or recovery after a proposed action. This baseline
   provides no control interface and is not recommended for robot commands.

## Phase 6A → next agent: calibrated offline grounding (Codex, 2026-11-12)

Implementation commit: [`dcc5de1c0d899d5fb8159c5f6b0905ab67b8cf01`](https://github.com/binbin-ovo/Go2_Navigation/commit/dcc5de1c0d899d5fb8159c5f6b0905ab67b8cf01).

- **Branch:** `phase6-vln-input-validation`, base
  `e4e0d3b01a16a14f52f7d039b27009ed3e0e0f96` from Claude's calibration branch.
  No main merge. No robot connection or commands. `phase5-detection-recovery` and
  Claude's worktree/environment were not modified. This base has 5A/5B, not 5A.1.
- **Capability actually tested:** Grounding DINO Tiny text-conditioned image-region
  proposals from measured-calibration Go2 recordings. Not a VLN navigation policy.
- **Read first:** [VLN_INPUT_VALIDATION](VLN_INPUT_VALIDATION.md) for actual measurements,
  failures and calibration limits; [VLN_OFFLINE_DEMO](VLN_OFFLINE_DEMO.md) for exact
  commands/artifacts; [VLN_MODEL_SELECTION](VLN_MODEL_SELECTION.md) for the checkpoint audit;
  [GO2_VLN_INPUT_INTERFACE](GO2_VLN_INPUT_INTERFACE.md) for the unchanged sensor contract.

**Implementation:** `vln/model_preprocessing.py` rectifies raw RGB and carries calibrated
K through official processing; `vln/grounding.py` loads pinned pretrained weights and
returns image proposals; `vln/grounding_demo.py` saves evidence and a separate LiDAR grid.
CLIs: `scripts/phase6_grounding_demo.py`, `scripts/phase6_evaluate_grounding.py`.
Config/cases: `configs/vln_grounding.yaml`, `configs/vln_grounding_eval.yaml`.
Regression tests: `tests/test_vln_grounding.py` (27 tests).

**API:** `view = calibrated_view(obs)`; `result = model.predict(view, instruction)`;
`run_observation(obs, model, output, source=...)` writes the full experiment.
Use a `Go2VLNObservation` with raw RGB uint8 and measured K/D of matching dimensions.
The model gets one current image, preserves but ignores history, and applies its own
normalization once. Output boxes use rectified-image edge pixels, optical axes remain
right/down/forward. No conversion to base_link forward/left/up or metric distance.
All `metric_goal`, `action`, `point_base_m` fields remain null even if future callers
supply extrinsics. The right-hand LiDAR plot is a separate observation, not a fused map.

**Important correction to older guidance below:** floor-seen cells and sector floor
returns do **not** establish traversability or a safe swept corridor. The historical
Phase 5D symbolic-action conversion paragraph is not an approved Phase 6 execution
path. Pixel proposals do not become camera-relative or base-relative metric waypoints.
No executable-goal adapter exists here.

**Install / verify:** in this branch, `uv venv --python 3.12 .venv`,
`uv pip install --python .venv/bin/python -e '.[test,vln]'`, optionally install `av==17.1.0`
for inherited decoder tests, then `.venv/bin/python -m pytest -q -rs`.
Actual result: **421 passed, 1 hardware skipped**, 30.71 s. No automated tests weakened.
Model + processor pinned to `IDEA-Research/grounding-dino-tiny` revision
`a2bb814dd30d776dcf7e30523b00659f4f141c71`; CPU FP32 / four threads.

**Local evidence:** `/private/tmp/go2-phase6/data/outputs/phase6/evaluation/` (ignored).
Each case has raw observation/model output, source hashes/clocks, preprocessing,
proposal JSON and demo PNG. Start with `s_fridge/demo.png` (correct) and `m_fridge/demo.png`
(wrong door region), then `s_dog`, `m_bicycle`, `m_sofa` false positives and `m_shoes`
miss. `semantic_review.json` records Codex visual review; human review is pending.
The 12-case declaration predates inference; thresholds were not retuned. 6 acceptable,
1 partial, 5 failed. Output validity was 12/12; semantic correctness differs.
NPZ replay reproduced raw arrays exactly. A navigation instruction was rejected as
unsupported without forwarding through the model. All private media, outputs and
weights remain out of Git. Local temporary paths may eventually be cleaned; rerun
from the private recordings and pinned model using the documented commands.

**Next work:**
1. Human-review the local panels and create diverse held-out object/absent/relational
   grounding annotations. Record failures; do not optimize on these 12 cases.
2. Claude: pursue measured, held-out validated camera/base/LiDAR registration and
   timing/odometry validation; preserve unknown fields until verified.
3. Select a true temporal/relational VLN model in a separate experiment if needed;
   its output still requires a documented frame and independent geometry checks.
4. Only after validated registration/range: design metric grounding and a separate
   collision-aware planner. No locomotion/control work is authorized by this milestone.

## Phase 5D → Codex handoff: Go2 VLN inputs (Claude, 2026-11-11)

- **Branch:** `phase5-go2-vln-sensor-calibration`, based on `89d901dc` (5B). Do not merge main.
  5A/5A.1 perception modules, configs and defaults are unchanged.
- **Interface:** `go2_perception.vln`, schema `go2-vln-obs-v1`. Spec:
  [GO2_VLN_INPUT_INTERFACE](GO2_VLN_INPUT_INTERFACE.md).

**Available now (real data):**
| Input | Shape / units | Frame | Time |
|---|---|---|---|
| RGB | uint8 (720, 1280, 3) **RGB**, raw/distorted (fisheye K, D measured), ~14.3 Hz, + history | `go2_front_camera_unrectified` | host receipt, and estimated robot-clock exposure (host − offset − 48 ms, σ 16 ms) |
| Obstacle points | float32 (~11 k, 3) m, 0.5 s window, odom-compensated | `base_link` of the reference scan | scan midpoint, robot clock |
| Display points | float32 (~21 k, 3) m, 1.0 s window | same | same |
| Grid | int8 (100, 100), 0.1 m, x −3…7 m, y −5…5 m; −1 unknown / 0 floor seen / 1 occupied | `base_link` | same |
| Elevation | float32 (100, 100) m, NaN unknown | `base_link` | same |
| Sector ranges | (72,) m per 5° CCW from +x; NaN unknown; `sector_clear_to` floor evidence | `base_link` | same |
| Pose / twist | 4×4 `T_odom_base`; twist frame unverified | `odom` ← `base_link` | robot clock |

**Calibrated vs uncertain:** camera intrinsics **measured** (`configs/go2_front_camera_calibration.yaml`,
fisheye, held-out 0.47 px; pass it as `calibration=`); `T_base_camera` **missing**, so
`projected_depth` is `None`. `T_base_lidar` comes from firmware only. Camera latency was
measured once. Odometry accuracy under motion is unverified. The floor-height assumption (−0.33 m)
holds only while standing. The robot clock resets each boot: compare stamps only within one `clock_epoch`.

**Feasible VLN types now:** RGB-only video models producing symbolic or mid-level actions
(NaVid / NaVILA / StreamVLN style), gated by the LiDAR grid. After calibration: RGB + geometric map
(VLFM-style value map on our grid, the preferred approach). RGB-D/3D models need calibration plus
dense depth. See [VLN_MODEL_COMPATIBILITY](VLN_MODEL_COMPATIBILITY.md).

**Model outputs → goals without inventing geometry:** a symbolic action becomes `T_odom_base · T(θ, d)`.
Accept it only if the target cell and corridor are FREE (floor seen) and within `sector_clear_to`;
UNKNOWN is not free. Pixel targets now give a calibrated *ray direction* in the camera frame, but
turning it into a metric goal needs `T_base_camera`, so they are unsupported today. 3D boxes
and model confidence never count as clearance. Nothing here sends commands.

**Reproduce (this Mac):**
```bash
cd /Users/yushan/Desktop/Go2_Navigation-phase5d
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e '.[test,forecasting]'
export PYTHONPATH=src
D=/Users/yushan/Desktop/Go2_Navigation/data
.venv/bin/python -m pytest -q                                    # 381 passed, 3 skipped
.venv/bin/python scripts/phase5d_vln_replay_demo.py --synthetic --output /tmp/vln_demo
.venv/bin/python scripts/phase5d_vln_replay_demo.py --recording $D/recordings/p5d_static_02 --index 200 --output $D/outputs/phase5d/demo
.venv/bin/python scripts/phase5d_lidar_accumulation.py --recording $D/recordings/p5d_static_02 --output $D/outputs/phase5d/accumulation
.venv/bin/python scripts/phase5d_timing_report.py --data $D/recordings --recordings test_c p5d_static_02 --output $D/outputs/phase5d/timing.json
.venv/bin/python scripts/phase5d_make_charuco.py --screen 2560x1600 --output $D/outputs/phase5d/targets
```
Live capture (robot stationary, user present): `.venv-go2` in the main checkout, `--iface en9`,
see [MAC_CONNECTION](MAC_CONNECTION.md).

**Claude/user next:**
1. ~~Camera intrinsics~~ done 2026-11-11: [GO2_CAMERA_CALIBRATION](GO2_CAMERA_CALIBRATION.md).
2. Extrinsics: stopped 2026-11-12 after two failed sessions; read "Session 2" and "What would plausibly
   work next" in [GO2_LIDAR_CAMERA_EXTRINSICS](GO2_LIDAR_CAMERA_EXTRINSICS.md) (≥ 60 × 80 cm target with
   larger squares, poses at one range, plane-level gate, PCA not 3-point RANSAC; or model the LiDAR's
   range bias first).
3. Re-measure camera latency on a new boot (operator-driven yaw).

## Phase 5B → Codex handoff (Claude, 2026-11-11)

- **Branch:** `phase5-fsd-visualization`, based on `be4b9de` (Phase 5A).
- **Do not merge main.** Nothing in `scene/`, any existing config or any default was changed; the only new config is the prepared held-out protocol.
- **Read:** [PHASE5_REAL_VALIDATION](PHASE5_REAL_VALIDATION.md) for the evidence and the fix list;
  [FSD_STYLE_VISUALIZATION](FSD_STYLE_VISUALIZATION.md) for the viewer.

**Fixes for Codex, ordered by evidence (development data A1–C; validate on held-out H1–H3):**
1. **Freshness from recent coverage.** Refresh an object from ≥3 returns in the last 2–3 messages, or
   by swept-sector coverage, rather than the latest ~62 ms message alone. In B, 88 of 91 "absent"
   walking frames have zero person returns in the latest message, yet ≥8 non-ground points at 0.3 s
   (89/91). At the person's bearing the message has no non-floor returns in 94.5% of them: a sampling
   gap, not occlusion.
2. **Motion gate robust to σ.** `speed − 2σ > 0.25` blocks 56 fresh walking frames: median speed
   0.44 m/s, median σ 0.23 m/s. Use a displacement/direction test, `speed − 1σ`, or σ from
   `observed_box` centres.
3. **Inherit motion history** when a reborn object re-associates within 1 s of an expired one (23 B
   frames).
4. **Robot self-filter** in proposal generation (fresh object within 0.45 m of `base_link` in about
   98% of frames).
5. **Leg clustering** (14 B frames left as noise or merged off-centre).
6. **Vectorize `BackgroundMap.stable_mask`.** With `object_window_s=0.3` a replay ran at about
   9–18 s/frame, so the counterfactual was stopped.

**Reproduce (on this Mac):**

```bash
cd /Users/yushan/Desktop/Go2_Navigation-phase5b    # worktree on phase5-fsd-visualization
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e '.[test,forecasting]'
export PYTHONPATH=src
D=/Users/yushan/Desktop/Go2_Navigation/data
O=$D/outputs/phase5b
.venv/bin/python scripts/phase5b_scene_tap.py --input $D/recordings/test_b --output $O/tap_new/test_b   # fresh dir; optional --set key=value diagnostics
.venv/bin/python scripts/phase5b_diagnose_walking.py --recording test_b --data $D --tap $O/tap_new/test_b --output $O/diagnosis_new/test_b.json
.venv/bin/python scripts/phase5b_compare.py --data $D --variant 5A=$O/tap --variant fix=$O/tap_new --output $O/comparison_new.json
.venv/bin/python scripts/fsd_viewer.py --recording test_b --data $D --mode debug --tap $O/tap_new/test_b --scene $O/tap_new/test_b/scene.jsonl
.venv/bin/python -m pytest -q
```

`phase5b_compare.py` expects all four recordings under each variant directory. `phase5b_scene_tap.py`
writes the same `scene.jsonl` as `run_stable_scene.py` plus Debug sidecars, and `--reference` checks
row identity.

**Claude/user next:**
1. Record held-out H1–H3 per `configs/phase5_heldout_protocol.yaml` (robot moved by hand only).
2. Verify `cloud_base` sector coverage on hardware.
3. Calibrate the camera.
4. Build a live read-only viewer with end-to-end latency measurement.

**Limitations:**
- Annotations are not human-verified; everything here is development data.
- The camera sync is approximate.
- The viewer's floor is assumed at robot z − 0.32 m.
- The 0.3 s 5A counterfactual was not completed.

## Phase 5A → Phase 5B Claude Handoff

- **Branch:** `phase5-stable-lidar-perception`.
- **Complete interface commit:** `5e269061c3da70ffe3d00798eb9bf05977869797` (includes initial fixture/schema version commit `9b99375`). This immutable commit contains every renderer-facing interface file; this handoff note is a subsequent documentation commit on the same branch.
- **Contract:** `docs/SCENE_REPRESENTATION_INTERFACE.md`.
- **Versioned machine schema:** `schemas/go2-scene-v1.schema.json` (`go2-scene-v1`, JSON Schema 2020-12).
- **Exact deterministic fixture:** `tests/fixtures/scene/renderer.jsonl` (six JSONL rows, synthetic only).
- **Minimal consumer:** `scripts/read_scene.py`, standard library only; `scripts/generate_scene_fixture.py` reproduces the fixture. Semantic validator: `src/go2_perception/scene/serialization.py`; tests: `tests/test_scene_interface.py` (**8 passed**).

Fetch the branch, then consume the fixture without waiting for perception implementation:

```bash
git fetch origin phase5-stable-lidar-perception
# Read from a separate checkout/worktree to preserve your current annotation/video work.
git worktree add ../Go2_scene_renderer origin/phase5-stable-lidar-perception
cd ../Go2_scene_renderer
python scripts/read_scene.py tests/fixtures/scene/renderer.jsonl
```

All scene geometry is already in `odom`, in meters, right-handed x-forward/y-left/z-up at odometry origin. Timestamps are seconds in the explicitly named clock domain; fixture clock is `synthetic`, real records use `go2_robot`. Row-major `T_odom_base` maps base column vectors into odom: apply it only to the robot mesh, never again to the scene points/boxes. Clear renderer history on epoch changes. Null pose is unknown, not identity. Preserve observed-vs-filtered-vs-stale and UNCERTAIN state; absence/expired cells are unknown, not free space. The last two fixture frames deliberately mark the static box stale while the second object remains observed/MOVING.

`display_compact` controls drawing only; raw measurements must still enter perception. Scores are defined evidence/recency ratios, not probabilities. This delivery contains no camera calibration, private recordings, control commands or promise of real geometry accuracy. Claude may implement the renderer against v1 immediately. Codex will continue Phase 5A on the same branch, preserving required fields and semantics. Additive optional fields are allowed; a breaking change requires explicit coordination with Claude and a new version. User selected GitHub branch publication as the delivery channel; Claude acknowledgement remains pending.

## Phase 5A perception continuation

Implementation and experiments are complete as an **experimental** opt-in pipeline, not a validated perception upgrade. See STABLE_LIDAR_ARCHITECTURE.md and STABLE_LIDAR_EXPERIMENTS.md before consuming real output. `StableScene.process` accepts raw Nx3 base_link points, seconds, unique interval IDs, same-clock pose lookup and optional per-point second offsets. It emits the unchanged published v1 contract. Do not retransform scene geometry or feed accumulated cloud_deskewed data. Reset renderer history on epoch changes; stale boxes must stay visibly stale/UNCERTAIN.

- Run: `PYTHONPATH=src .venv/bin/python scripts/run_stable_scene.py --input /path/to/recording --output data/outputs/scene_trial`. The output directory must be new and gitignored. No annotation input is accepted by runtime.
- Synthetic comparison: `PYTHONPATH=src .venv/bin/python scripts/benchmark_stable_scene.py --output data/outputs/scene_synthetic_trial`.
- Tests: `PYTHONPATH=src .venv/bin/python -m pytest -q`: **310 passed, 1 hardware skipped**. Full Ruff has 33 inherited hardware-script findings; Phase 5A scoped lint passes. Exact setup/report commands are in STABLE_LIDAR_EXPERIMENTS.md. New jsonschema test dependency is pinned to 4.26.0.
- Baseline comparison and aggregate evidence: STABLE_LIDAR_RESULTS.md / STABLE_LIDAR_METRICS.json. Real Phase 5A moving recall A2=0.121, B=0.058; fresh box coverage is much lower than original. Roughly 69% of emitted boxes are stale. Do not promote defaults or describe smooth geometry as reliable pedestrian perception.
- Source ownership: Codex added `scene/`, stable scene scripts, config and tests. Claude retains Go2 access, timing verification, annotation review and renderer. Existing tracker/adapter/forecasting code was not rewritten. No raw private data was committed.
- Next steps: consume the fixture on the branch; verify stale/epoch rendering; human-review development annotations; collect held-out scenes including nearby/occluded people; verify instantaneous point-frame semantics before enabling deskew; improve proposal coverage and compare observed-only metrics before changing operational settings.
- Known failure reproduction: run synthetic `near_furniture` through the benchmark (moving recall 0.189 at 0.1 s, 0 at 1.0 s). Replay A1/A2/B/C locally with the documented evaluator to reproduce low fresh coverage and low moving recall. Missing odometry emits empty geometry; weak evidence emits UNCERTAIN. Covariance is heuristic, not a calibrated safety bound. No live/held-out Phase 5A validation has been performed.

# Agent handoff — 2026-11-09 (macOS update)

## Phase 3.8 visual ground-truth handoff (integrated history)
Branch `phase3-visual-groundtruth`. It contains `79bb0f9` (ground truth) and the merged Phase
4.1 branch `5979a76`. Do not merge main or change defaults without approval. Read
[VIDEO_ANNOTATION](VIDEO_ANNOTATION.md), then [TRACKING_GROUND_TRUTH](TRACKING_GROUND_TRUTH.md).

**Code added:**
- `visualization/diagnostic_video.py`: rendering, camera alignment, trails, heuristic events,
  H.264 writer, private-path guard.
- `scripts/render_tracking_video.py`: single, comparison and forecast-overlay videos, stills,
  event logs.
- `scripts/render_highlights.py`: the captioned montage.
- `tracking/annotation_edits.py` and `scripts/annotation_editor.py`: manual corrections as
  review entries, CSV round trip, `identity_uncertain`.
- `scripts/forecast_ground_truth.py`: 8+12 windows at 0.4 s from validated trajectories,
  through the Phase 4 adapter checks; CV only.
- 13 synthetic tests.

**Videos are private and local:** `/Users/yushan/Desktop/Go2_Navigation/data/outputs/mac/videos/`.

**For Codex:**
- Use the videos to inspect cluster merging. Most red events coincide with the box or the
  person merging into a background cluster.
- The proposal-level work from the ground-truth handoff remains the priority.
- Do not tune on A1, A2, B or C.

**For Claude and the user:**
1. Human review in the editor (A2, B, C first).
2. Record held-out H1–H3.
3. Place the Phase 4 / 4.1 checkpoints under `data/outputs/forecasting/` and add learned
   predictions to `forecast_ground_truth.py`, through the existing adapters only.
4. Camera calibration, before any camera↔LiDAR association.

**Forecasting blocker:** only 6 independent identity-validated windows (one person, B and A2).
That supports plumbing checks only.

## Phase 3.8 ground-truth handoff (2026-11-08)
Branch `phase3-ground-truth-validation`, from `c643422`. Do not merge main or change the default
without approval. Read [TRACKING_GROUND_TRUTH](TRACKING_GROUND_TRUTH.md) first. The Phase 3.7
section below is historical.

**What exists now:**
- An annotation workflow, `scripts/annotate_ground_truth.py` (`cache` / `propose` / `sheet`). It
  uses LiDAR background subtraction, a measured static footprint and LiDAR-defined visibility,
  and the camera only for review.
- The schema and metrics, `src/go2_perception/tracking/ground_truth.py`: CLEAR-MOT-style matching
  that keeps the previous match, IDF1, IDF1 per visit, switches, fragmentations, duplicates,
  velocity error, motion confusion and recovery after occlusion.
- `scripts/evaluate_ground_truth.py`: scoring, proposal-quality and ID-change diagnosis, and an
  oracle-segmentation diagnostic.
- Annotations for B, A2, C and A1. They are local only, in `data/annotations/`; the specs there
  record intervals and review notes.
- 11 synthetic tests.

**Measured, original → 3.6 → 3.7:**
- IDF1: B 0.33/0.40/0.39, A2 0.30/0.60/0.53, C 0.29/0.60/0.41, A1 0.90/0.90/0.86.
- 3.6 is the best identity profile; 3.7 has fewer duplicates but lower coverage and more A1
  flicker.
- None is default-ready. Keep `tracking.yaml` as the default.

**Bottleneck: proposals.**
- Box merged with background in 14–33% of visible frames, and with the person in another 12%
  in B.
- Person merged in 23% of frames in A2.
- Splits are rare. Odometry is ruled out as a major cause in C.
- **Oracle segmentation diagnostic.** The untuned original tracker on oracle proposals reaches
  IDF1 B 0.67, A2 0.69, C 0.54 and A1 0.98.
  - That beats every tuned profile on real proposals in B, A2 and A1.
  - 3.6 and 3.7 do worse than the original on clean proposals; for example, A1 3.6 has 44 flicker
    switches.

**Next for Codex.** Improve LiDAR proposals before more tracker tuning:
1. Separate low objects from walls and counters: height-profile or background-aware splitting,
   a smaller or adaptive `eps` near large planar clusters.
2. Separate a person from an adjacent object.
3. Then re-check the tracker profiles on the improved proposals. On clean proposals the
   original tracker beats 3.6 and 3.7.
4. Do not tune on A1, A2, B or C again. They are development data now.

**Next for Claude and the user:**
1. A human pass over the review sheets (time ranges listed in the report).
2. Record held-out H1–H3 (two people, a taped start/end pose), annotate with unchanged
   thresholds, and score once.
3. Camera calibration, to enable semantic gating.

## Phase 4.1 forecasting handoff (merged from `phase4-social-forecasting` at 5979a76)

Branch `phase4-social-forecasting`, based on verified Phase 4 `447f164e808fee381fd3e9dbe8244583a0bed2f4`. No merge into main. Implemented observation-only multi-pedestrian extraction with partial-history masks, an 87,128-parameter shared temporal encoder with relative-position/mean neighbor pooling, dedicated checkpoint/training/evaluation interfaces, robustness studies, research plots, and a guarded offline multi-track adapter. Original forecasting, TrackOutput/ForecastOutput, tracking internals and Go2 hardware adapters are unchanged.

Completed: all five scene-identifier-held-out folds for original and social models with seeds 42/43/44 (15,061 paired windows per seed), same-weight social-disabled diagnostics, twenty validation robustness cases across clean/joint/noise-only/mask-only training, and separate CPU latency measurement. Every single-agent prediction and target/key population was checked against the independent Phase 4 evaluator. Full suite: **259 passed, one hardware skipped in 17.37 s**; Ruff passes on forecasting code/scripts/tests. No dependency changes (Python 3.12.11, torch 2.6.0, numpy 1.26.4, matplotlib 3.11.2).

**Negative result:** macro ADE/FDE, mean ± sample SD over three seeds: social 0.5812 ± 0.0162 / 1.1951 ± 0.0247 m; original 0.5565 ± 0.0118 / 1.1534 ± 0.0200 m; CV 0.5214 / 1.0933 m. Disabling social input improves macro errors in every seed. Noise-only augmentation improves one-fold noisy validation ADE to 0.3924 m, still behind noisy CV 0.3371 m. No claim of overall learned-model improvement or independent-window statistical significance.

Read [social architecture](SOCIAL_FORECASTING_ARCHITECTURE.md), [actual experiments and commands](SOCIAL_FORECASTING_EXPERIMENTS.md), [predeclared protocol and follow-up controls](SOCIAL_FORECASTING_PROTOCOL.md), [metrics](SOCIAL_FORECASTING_METRICS.json), and [Go2 contract](FORECASTING_INTEGRATION.md). Published artifacts are public-data aggregates/selected plots; raw annotations, weights, predictions and private recordings remain ignored. Known limitations: weak mean pooling, one-seed robustness controls, dependent windows, scene/domain shift, unverified dataset redistribution rights, no clean-install/GPU/live-Go2 validation. No navigation control.

Next: keep CV as the reference; Claude validates real pedestrian identity segments and stable metric frames. Then run matched held-out offline Go2 comparisons with explicit externally validated IDs. Repeat noise-only augmentation across seeds before considering more architecture. Historical phase states follow; their current/next claims apply to their original milestones.

# Phase 4 historical state — 2026-11-09

Branch `phase4-trajectory-forecasting`, based on Phase 3.7 `c6434229c54ee445763d8ceb1f5ef56b0f5c50da`; main is not merged. Added public ETH/UCY trajectory preparation, independent CV and Phase 3 baseline adapter, lightweight PyTorch Transformer, deterministic training/checkpoints, paired evaluation, validation ablations, plotting and guarded offline TrackOutput integration. Tracking/sensor internals remain unchanged; Claude's Phase 3.8 work is separate.

Verified: five fixed scene-identifier-held-out CPU runs (15,061 windows), validation-only history/capacity/representation/noise/missing-input studies, synthetic adapter demo and 231 tests passed/1 hardware skipped. The learned model **loses overall**: macro ADE/FDE 0.5701/1.1762 m vs CV 0.5214/1.0933 m. No Go2 forecasting accuracy or GPU validation is claimed. Data/weights remain local, public aggregate results and selected plots are committed. Python 3.12.11, torch 2.6.0, numpy 1.26.4, optional matplotlib 3.11.2; existing core pins unchanged.

Read [forecasting architecture](FORECASTING_ARCHITECTURE.md), [dataset protocol](FORECASTING_DATASETS.md), [commands and actual experiments](FORECASTING_EXPERIMENTS.md), and [integration contract](FORECASTING_INTEGRATION.md). Known limitations: one seed, correlated windows, no social context or uncertainty, unverified dataset redistribution rights, no clean-install test, and untrusted real tracker identities. Do not train from unvalidated Go2 tracks. Next: Claude validates identities/frames/visibility, then evaluate public-trained models and CV on held-out ground truth; extend forecasting research using validation-only decisions. Earlier phase sections below are historical.

## Phase 3.7 refinement handoff (historical)
Branch `phase3-tracking-refinement` starts at `de043d1`, containing Claude's real comparison and
Codex's Phase 3.6 work. Do not merge main or promote a profile to default without approval.
Read [TRACKING_REFINEMENT](TRACKING_REFINEMENT.md) first, then the unchanged
[TRACKING_REAL_COMPARISON](TRACKING_REAL_COMPARISON.md) for the regression diagnosis.

New code constrains reactivation by last observation/time/speed; suppresses repeatedly overlapping,
velocity-consistent weaker tracks; and separates moving activation/release from stationary dwell.
Use `configs/tracking_refinement.yaml`; both earlier profiles remain available. Public sensor and
tracking interfaces are unchanged. Source tests: 192 passed/1 hardware skipped. Seed 42's split
failure persists; first-miss Lost status reduces occlusion IDF1, and irregular-motion recall has small regressions. Dependencies unchanged.

The exact `compare_tracking_real.py --reuse-cache --serial` command is in TRACKING_REFINEMENT.
Use its new read-only --cache-root and a fresh output directory: existing run outputs now refuse
overwrite; --analyze reads them. Explicit PYTHONPATH=src avoids this Mac's stale installed package.
No private samples are committed. Claude owns hardware/calibration verification and annotation;
next label identities/visibility, record two close people and validate transitions on held-out data.
Do not start learned forecasting using these unverified identity targets. Earlier handoffs below
are historical; their 'next' tasks should be read in that context.

Final offline replay: 3,924 frames; original/default outputs remain identical. Walking MOVING
proxy improves about 39%→52%, and long reactivations fall in all recordings. Much of the duplicate
count reduction is removal of coasting Confirmed states; observed-only duplicates barely change.
A2/C dominant-ID continuity regresses relative to Phase 3.6, and A1 flicker increases. This is
not a uniform fix or a default-ready profile; labels/held-out testing remain required. Full measured
results and runtime are in TRACKING_REFINEMENT. No private samples are committed.

## Phase 3.6 robustness handoff (historical)
Branch `phase3-tracking-robustness` starts at `feature/macos-go2-phase3-validation` / `ecfaae3`.
Do not merge main without approval. Read [TRACKING_ROBUSTNESS](TRACKING_ROBUSTNESS.md) first,
then [TRACKING_ARCHITECTURE](TRACKING_ARCHITECTURE.md) and Claude's unchanged
[PHASE3_REAL_VALIDATION](PHASE3_REAL_VALIDATION.md). The previous macOS handoff below is historical.

Codex added opt-in `configs/tracking_robustness.yaml`, conservative historical-box fragment
recovery, time expiry for confirmed tracks, adaptive gating, motion dwell, lifecycle diagnostics,
26 regressions and `scripts/benchmark_tracking_robustness.py`. No sensor/hardware code changed.
`PYTHONPATH=src .venv/bin/python -m pytest -q`: **161 passed, 1 hardware skipped**.
Always set PYTHONPATH on this Mac to avoid the old installed package; `.venv-go2` can be used
instead. Install instructions and pinned dependencies remain in SETUP.md.

Use the exact synthetic/private-replay commands and aggregate comparisons in TRACKING_ROBUSTNESS.
Private cache and per-frame results live under ignored `data/outputs/robustness/`; no recordings
are needed for the synthetic benchmark. Seed 42 still has three split-object switches; brief-stop
classification recall regresses. Baseline defaults are deliberately unchanged.

Claude owns physical sensor access, timing/calibration validation and labelled real evaluation.
Next: annotate A2/B/C locally, inspect recovery/association failures with raw proposals, run D
(two nearby people), measure slow-motion/stop transitions and pose drift. Keep private sensor
files out of Git. Preserve pose-gate skips/resets. Integration continues through Detection3D,
RobotPose and the unchanged tracker.update signature; no detector internals need modification.
Phase 4 dataset tooling can use current outputs, but do not treat unstable IDs as reliable learned
forecasting targets. Hardware live validation and calibrated RGB semantic fusion remain outstanding.

**Real-data comparison done (Claude):** [TRACKING_REAL_COMPARISON](TRACKING_REAL_COMPARISON.md),
harness `scripts/compare_tracking_real.py`, local outputs in `data/outputs/mac/tracker_comparison/`.
The default path is bit-identical to `ecfaae3`; the improved profile is ready as opt-in, not as
default. Ablations attribute the effects:
- **Time expiry** gives the box-continuity gain, but also the extra duplicates and flicker.
- **The 2.5 m adaptive gate** causes most reactivations more than 1 m from the last observation.
- **Dwell plus the 0.25 m/s threshold** removes false MOVING labels, but also cuts walking MOVING
  recall from 62% to 39%.

Next for Codex:
1. Suppress or merge a Lost track that overlaps a Confirmed track.
2. Bound the reactivation radius by elapsed time × a plausible speed.
3. Re-tune dwell for walking people.
4. Rerun `compare_tracking_real.py --reuse-cache --serial`. It needs the local recordings, so it
   runs on the Mac.

Next for Claude/user: hand-label box and person IDs on A2/B/C so these proxies can be checked.

## macOS validation handoff (historical baseline)
Branch `feature/macos-go2-phase3-validation`; do not merge without user approval. The Mac reads the
Go2 natively; setup, rates and pitfalls are in [MAC_CONNECTION](MAC_CONNECTION.md). Use
`.venv-go2` for hardware and `--iface en9` (the config default is Linux `eno2`). Real tracking
results and gaps: [PHASE3_REAL_VALIDATION](PHASE3_REAL_VALIDATION.md).

```bash
.venv-go2/bin/python scripts/go2_record.py --iface en9 --seconds 60 --name test_a
.venv-go2/bin/python scripts/run_tracking.py --input data/recordings/test_a --output data/outputs/mac/test_a --window-s 0.3 --video
.venv-go2/bin/python -m pytest -q
```
Pose gating: `configs/tracking.yaml` `pose_alignment`; skipped frames in `skipped_frames.jsonl`.

**Next for Codex** (evidence: PHASE3_REAL_VALIDATION, recordings `test_a1`, `test_a2`, `test_b` on
the Mac): (1) person identity: one walking person became 19 IDs; examine leg clustering and
coasting; (2) occlusion and viewpoint: a static box is re-identified each time a person stands in front,
and every 5-15 s while the robot turns (Test C, 22 IDs) even though its odom position holds to 6-9 cm;
(3) motion classifier: steady velocity sigma about 0.14 m/s makes STATIONARY need speed under about
0.01 m/s; (4) clutter fragments on walls give false MOVING. Keep synthetic metrics intact and get
labels before tuning. **Next for Claude/user:**
Test D (two people), a Test C repeat with a taped end pose for odometry drift, Mac
interface override for `go2_live.py`/hardware test, camera calibration.

## Phase 3 handoff (2026-11-06)

Changes are committed on `phase3-tracking` from main `dc0f85b`; do not merge without user approval.
New packages: tracking, localization and visualization; core/tracking_types.py; three tracking
scripts and configs/tracking.yaml. All original modules/tests remain intact. Validation:
127 passed/1 hardware skipped. Reference synthetic IDs are stable; additional noisy seeds have
four switches across forty sequences. No real temporal or live tracking accuracy is claimed.

```bash
source .venv/bin/activate
uv pip install -r requirements.txt
python scripts/generate_tracking_test_data.py
python scripts/run_tracking.py --input data/sample/tracking_sequence/crossing --config configs/tracking.yaml --output data/outputs/tracking_run --video
python scripts/evaluate_tracking.py --input data/sample/tracking_sequence --output data/outputs/tracking_eval.json
pytest -q
# On the recording workstation, with real bracketed odometry:
python scripts/run_tracking.py --input data/recordings/stage_c_person --output data/outputs/tracking_person --window-s 0.3
```
Choose fresh output directories. *(2026-11-07: unposed frames are now skipped with a reason and
reported; a recording with no posable frame still raises. See TRACKING_ARCHITECTURE.md.)*
Do not substitute identity poses. Use `--mode fixed_sensor` only with an explicit fixed-sensor
assumption, and interpret velocities as sensor-relative. Full JSONL input, types, covariance,
frame conventions, config meanings and output contracts: TRACKING_ARCHITECTURE.md.

**Claude's exact next steps:**
1. Keep adapter/hardware ownership. Record cloud_base and odometry with real acquisition stamps;
   provide `go2_robot` clock and `T_odom_base` (xyzw source quaternion). No host/RTP substitution.
2. For live normalized detections, call `from_lidar` / `from_fusion` then the persistent
   MultiObjectTracker. Supply a RobotPose mapping the **actual detection frame** to odom at
   that timestamp. If detections were levelled to base_link_level, use yaw+translation, not
   a second roll/pitch rotation. Buffer/interpolate odometry upstream with validated gaps.
3. Validate stationary world objects while translating/rotating the robot, then known moving
   targets. Measure velocity error and ID continuity against annotated ground truth.
4. Review the documented exact-sample-after-gap PoseBuffer issue. Phase 3 handles exact observed
   samples in its replay wrapper; shared hardware code was not silently patched.
5. Validate dropouts, clock discontinuities, scan de-skew assumptions, odometry drift and dynamic
   ground height. Ground/clustering defaults still depend on standing height/scene density.
6. Complete camera intrinsics, distortion/rectification, extrinsics and capture-time alignment
   before associating RGB semantics with LiDAR trajectories. No synthetic calibration on real data.

**Phase 4 interface:** consume TrackOutput history, timestamp/frame, velocity/covariance/status;
replace `predict_trajectory` with a learned predictor returning timestamped positions and
uncertainty in the same frame. Establish annotated metrics and robustness first. Robot control,
learned networks, planners and navigation are not implemented by Phase 3.

Historical handoffs below describe their original phases, not the current task status.

## Main integration handoff
Local merge verification: **70 passed, 1 hardware test skipped** (macOS CPU).
Work from `main`, which contains both phase commits plus the merge validation tests/docs.
`tests/test_go2_replay.py` now exercises the real committed cloud through recording and
the replay CLI, with and without a window. Its stationary identity poses are synthetic
test scaffolding, not measured motion evidence. See [MERGE_VALIDATION](MERGE_VALIDATION.md).
Adapter/hardware ownership remains unchanged. Current priority: measured calibration and
synchronization, then labelled clustering and motion validation.

## What Codex should do next
Claude's Phase 2 is on `feature/go2-sensor-integration`. Read [PROJECT_STATE](PROJECT_STATE.md), then
[SENSOR_VALIDATION](SENSOR_VALIDATION.md). The original Phase 2 priorities are in
[INTEGRATION_REPORT](INTEGRATION_REPORT.md): replay regression test (now added for the committed fixture), `RGBMessage.K` decision, cluster
tuning on real frames, odometry-based ground height, false-positive handling, then tracking.

**Do not** use `rt/utlidar/cloud_deskewed` as a scan, or use RTP timestamps as capture times. **Do not**
use synthetic calibration on real data, or add any publisher, RPC client or motion command to
`adapters/go2`. Claude keeps ownership of `adapters/go2`, `scripts/go2_*.py`, `configs/go2_*.yaml`, and
hardware validation; coordinate before changing them. Perception modules remain yours, unchanged.

Recordings are local-only and contain private home images. They are not on GitHub. Without the
capture workstation, record your own with `scripts/go2_record.py`, or use
`tests/fixtures/go2/real_cloud_pair.npz` (one real LiDAR frame pair).

Start here on the workstation:
```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/go2_replay.py --recording data/recordings/stage_c_person --window-s 0.3 --output data/outputs/replay
```
Hardware work still pending from Claude and the user: link-drop test, calibration, motion validation.

# Phase 1 handoff record (historical; superseded by Phase 2 above)
## Purpose and ownership
Build an offline RGB + LiDAR perception foundation for future Go2 dynamic navigation.
Codex implemented repository architecture, detectors/processors, geometric fusion, CLIs,
synthetic fixtures, automated tests and documentation. Claude independently owns Ethernet,
DDS/SDK/ROS 2 connection, verified RGB/LiDAR/IMU/odometry streams and live validation.
No Claude-owned integration files existed when this repository was initialized. Coordinate
before changing such files once integration begins. No robot commands belong in Phase 1.

## Current repository
Branch `codex/phase1-perception`; implementation in src/go2_perception, CLI scripts in scripts,
YAML in configs, tests in tests. Everything operates without hardware. Real calibration is
absent; synthetic calibration is conspicuously marked. No ROS runtime or fabricated topics.
Results: 44 unit tests passed (final run 3.37s); lint and dependency compatibility passed; CPU YOLO processed a real image (6 detections); synthetic
LiDAR found 2 clusters; synthetic fusion matched 2; public PCL cloud processed but default
segmentation yielded one broad cluster/no ground. No real paired fusion or live validation.
See PROJECT_STATE.md and EXPERIMENTS.md for exact versions, sources, hashes and measurements.

## Install/run
From repository root:
```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
pytest -q
python scripts/generate_test_data.py
python scripts/run_lidar.py --input data/sample/test.pcd --output data/outputs/lidar
python scripts/run_rgb.py --input data/sample/bus.jpg --output data/outputs/rgb
python scripts/run_fusion.py --image data/sample/synthetic.jpg --points data/sample/test.pcd --detections data/sample/synthetic_detections.json --calibration configs/synthetic_calibration.yaml --rgb-timestamp 1 --lidar-timestamp 1 --clock-domain synthetic --output data/outputs/fusion
```
Download bus.jpg / yolov8n.pt with exact commands in EXPERIMENTS.md. Outputs and inputs are
ignored by Git and must be generated/downloaded on a new checkout. For macOS ARM64 exact
transitive versions, use `uv pip install -r requirements-macos-arm64.lock`. Linux/CUDA has
not been tested. Existing local .venv uses a runtime under /private/tmp; recreate if broken.

## Integration steps for Claude
1. Read SENSOR_INTERFACE.md in full, then core/types.py, io/sensor_interface.py and
   fusion/calibration.py. ARCHITECTURE.md describes every public entry point.
2. Verify actual SDK/ROS topics, data layout/endian, units, BGR/RGB encoding, timestamps,
   clock domains, frame axes, scan duration and calibration source. Document evidence.
3. Rectify images, provide corresponding K, and measure camera-to-LiDAR extrinsics
   (`T_lidar_camera`, NOT its inverse). Use explicit measured frame names/resolution.
4. Convert observations to RGBMessage/LidarMessage; call process_rgb/process_lidar with
   persistent detector/processor objects, then FusionProcessor.fuse. Check message K
   matches calibration. Pair timestamps in a verified common clock upstream.
5. Align scan +Z with gravity, configure expected ground height, and tune sparse-cloud
   filtering/DBSCAN. Motion-compensate accumulated windows before passing them to core.
6. Record an offline Go2 pair and run overlays before any live loop. Add integration tests
   for this evidence and update PROJECT_STATE.md, AGENT_HANDOFF.md and EXPERIMENTS.md.

## Open questions and issues
Actual topic names, QoS, sensor clock conversions, frame tree, optical distortion model,
intrinsics/extrinsics, mounting height and per-point timing are unknown. Determine whether
streams provide raw or already transformed points/images. No live synchronization queue,
clock conversion, de-skew or tracking is implemented. Nonzero distortion raises ValueError.
PLY/PCD intensity is not retained; cluster boxes describe only observed points.

No failing unit tests remain. Regression reproduction:
`pytest -q tests/test_lidar.py` covers nearby elevated obstacles incorrectly treated as ground
by the initial implementation; the height-band support fix now passes. Parameter sensitivity
is intentionally demonstrated by the nearby-cluster test (large eps merges them).
`python scripts/run_lidar.py --input data/sample/table_scene_lms400.pcd --output data/outputs/lidar_real`
reproduces the public scene's one broad cluster/no ground with defaults; do not call this
validated segmentation. First YOLO import may report a nonfatal font-cache warning.

The next milestone is **recorded, synchronized Go2 RGB–LiDAR replay with measured calibration**,
followed by live read-only inference. Autonomous locomotion, learned forecasting and planning
remain out of scope.

## Phase 4 continuation checklist

1. Use the `phase4-trajectory-forecasting` branch and read FORECASTING_EXPERIMENTS.md for the complete installation/reproduction commands. Run the synthetic smoke configuration before downloading annotations.
2. Codex owns `src/go2_perception/forecasting/`, `scripts/*forecasting*.py`, forecasting configs and tests. Core tracking and Go2 adapters were not rewritten. Claude retains sensor access, real calibration, ground truth and hardware validation.
3. For a real offline trial, establish stable metric odometry coordinates and reliable pedestrian IDs, retain at least 2.8 seconds of observed history, then use the explicit bounded resampling adapter. Keep all real recordings local.
4. Compare CV and learned forecasts on the same manually validated held-out windows. Never treat the public-data results or synthetic demo as proof of Go2 forecasting accuracy. Resolve fragmentation/ID switches before collecting training labels.
5. Reproduce known limitations with the documented five-fold command (model loses overall) and the noise ablation (model error rises substantially). Unknown data redistribution rights, GPU execution and clean-install portability remain unresolved. Do not merge into main or promote the learned model to navigation without separate approval and validation.

## Phase 4.1 continuation checklist

- Code belongs to the independent Phase 4.1 branch; do not merge main without approval. Claude's `phase3-ground-truth-validation` checkout and sensor/annotation responsibilities are separate. No hardware or tracker internals were modified.
- Start with SOCIAL_FORECASTING_EXPERIMENTS.md for exact install/reproduction commands. Reproduce Phase 4 checkpoints first; then use `run_social_forecasting.py --stage single`, `--stage social`, `--stage robustness`, and `--stage robustness_controls`. Use fresh output directories. Summary auditing requires local prediction NPZ files from both runs.
- Use `social_training.load_social_checkpoint` for social weights. Existing Phase 4 weights/API remain supported by the original loader. `forecast_multitrack` returns the unchanged ForecastOutput plus explicit context/masks/rejection reasons. Pass checkpoint social settings and externally reviewed `validated_ids` for the current snapshot; never infer reliable pedestrian identity from anonymous clusters or Confirmed status alone.
- The model loses overall. Reproduce the five-fold/three-seed comparison and noisy validation table before making a research claim. Do not choose a winning seed or tune against those held-out results. A future experiment needs a declared validation-only decision process.
- Physical Go2 identity quality, camera–LiDAR calibration, GPU execution and clean-install portability remain unresolved. Complete Claude's ground-truth work before domain-transfer evaluation; do not train on unannotated tracks or issue movement commands.
