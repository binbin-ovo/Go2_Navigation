# Experimental continuous navigation

A fast, offline controller can renew a SEARCH or APPROACH primitive while the proxy moves. It accepts the existing supervised v2 proposal, including the ten native LightNav waypoints and source-pose identity. A separate command-line entry point selects the experiment. Its only velocity consumer integrates expiring packets into a local kinematic proxy. The five-move viewer, executor and launch flags still select the published stop–observe–act policy. [Five-move publication](https://github.com/binbin-ovo/Go2_Navigation/commit/918f48c7880d1ade20034a1ab37664fd3fa2188e) · [Baseline operator guide](FIVE_MOVE_NAVIGATION.md)

## Reproduce the offline runs

Use the migrated Mac workspace and the existing core interpreter. Each command prints a new private `summary.json` path. The replay command consumes the original closed sessions in place.

```bash
export GO2_WORKSPACE=/Users/yushan/Robotics/Go2
cd "$GO2_WORKSPACE/code/worktrees/codex-motion-diagnostics"
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/hardware"
export GO2_HARDWARE=0 GO2_SUPERVISED_LIVE=0 PYTHONDONTWRITEBYTECODE=1
export MUJOCO_RENDER_TEST=0 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
py="$GO2_WORKSPACE/envs/core/bin/python"
"$py" scripts/run_vln_continuous.py simulate
"$py" scripts/run_vln_continuous.py replay \
  --recording "$GO2_WORKSPACE/runs/effective_trials/trial-C3jMfl/session" \
  --recording "$GO2_WORKSPACE/runs/effective_trials/trial-abZbVx/session"
"$py" scripts/run_vln_continuous.py simulate --scenario two_second_planner --wall-clock
"$py" -m pytest -q -p no:cacheprovider tests/test_continuous_navigation.py tests/test_continuous_integration.py
```

For the full offline selection, set `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `TOKENIZERS_PARALLELISM=false`, `OMP_NUM_THREADS=1` and `VECLIB_MAXIMUM_THREADS=1`. Run `"$py" -m pytest -q -rs -p no:cacheprovider -m 'not hardware and not model_integration'`. The mock browser tests use local loopback. MuJoCo rendering and pretrained model integrations have their own environments.

Use `--scenario NAME` to replay one synthetic case, or `--output NEW_DIRECTORY` to choose the destination. The recording importer checks the destination's Git privacy and rejects writes inside the input session. Output consists of a summary and a JSONL trace per case. Each tick carries the measured/virtual time, pose and geometry, a hypothetical packet, and any selection or guard veto. The veto contains the snapshot checked at that tick. The capture and delivery rows retain image identity, selection evidence, the native proposal and capture/result times. The recorded pass retains file hashes. [Offline entry point](../scripts/run_vln_continuous.py)

## Asynchronous LightNav, handovers and the viewer

On this branch the optional learned backend runs the pinned LightNav checkpoint on MPS in the existing model subprocess. The control thread polls one future and owns the single active velocity packet. The planner's background thread prepares one immutable request, validates the returned session, decision, decoder and coordinate contract, and attaches the original capture pose. The model receives the instruction and causal RGB history. The controller receives the native chunk, grounded mark, independent pose, LiDAR and operator interlock. Turn and forward primitives remain separate. [Asynchronous planner](../src/go2_perception/vln/continuous_planner.py) · [Controller](../src/go2_perception/vln/continuous.py)

The simulated camera records at 4 Hz even during inference. Each learned decision uses the five-move policy of two initial and six recent unique frames with their original millisecond timing. The planner retains up to eight PNGs in the camera cache and one in-flight request can hold up to eight source PNGs until completion. The experiment caps capture metadata at 241 frames. The recorder has a 16-row queue and a 2 MiB serialized row limit. A storage error or saturated host-paced queue ends the offline run. The viewer indexes complete trace lines and reads images on demand. [Bounded context](../src/go2_perception/vln/bounded_context.py) · [Trace recorder](../src/go2_perception/vln/continuous_recording.py)

Each issued packet reports remaining metres or radians to the selected endpoint, the remaining distance or angle to its braking trigger, and the time to that trigger at the configured speed. The plan horizon also accounts for the result lease, image age and episode deadline. The packet has its own shorter expiry. These horizons are controller calculations. LightNav's absolute SE(2) chunk supplies no time scale. On delivery, the governor subtracts measured progress in the original capture frame and checks current geometry and drift. A result arriving after its worker lease expires cannot revive the path. Native STOP ends an episode with arrival unverified.

### Execution and review commands

Use the environment exports and `py` from the reproduction section above. Each run creates a fresh private directory under `$GO2_WORKSPACE/runs/continuous`. The runner prints that directory before the worker loads.

```bash
"$py" scripts/run_vln_continuous.py simulate --planner lightnav --wall-clock --scenario visible_after_search
"$py" scripts/run_vln_continuous.py simulate --scenario visible_after_search
"$py" scripts/run_vln_continuous.py simulate --scenario no_continuation
"$py" scripts/run_vln_continuous.py simulate --scenario target_loss
"$py" scripts/check_continuous_memory.py --episodes 64
"$py" -m pytest -q -p no:cacheprovider tests/test_continuous_navigation.py tests/test_continuous_integration.py
```

Open the existing loopback viewer from a second terminal. Use the runner's printed directory to follow a new run. The commands below reopen the archived demonstrations.

```bash
"$py" scripts/run_vln_interactive.py --review-run "$GO2_WORKSPACE/runs/continuous/simulate-tz26_wqj" --port 0
"$py" scripts/run_vln_interactive.py --review-run "$GO2_WORKSPACE/runs/continuous/simulate-8w_itx2j" --port 0
"$py" scripts/run_vln_interactive.py --review-run "$GO2_WORKSPACE/runs/continuous/replay-bsgvnqxg" --port 0
```

Open the printed URL. **Follow latest** reads the growing trace during execution. Select a run and pass, scrub by tick, or step between decisions and guard events. The camera and the proposal's source image have independent identities. The incoming chunk is drawn in its capture frame, and the active primitive, measured path and current heading are drawn in odometry coordinates. The controller panel includes commanded linear and angular velocity, remaining support, handover and stop reason. The continuous review routes accept only reads. The pre-existing interactive viewer page is byte-identical for other run types. [Review implementation](../src/go2_perception/vln/continuous_review.py)

### What the measurements establish

On November 19 two 12-second closed-loop runs used the cached `LightOriginsHQ/LightNav-0` revision `826dc5fbfa37afa8293d2e336d329b6ffc0bfb64`. Both used the isolated Mac model environment, PyTorch 2.10.0, Transformers 5.8.0, MPS BF16, SDPA and the existing local runtime patch. The simple, synthetic pinhole scene was fed back after the proxy's actual motion. No Go2 camera transform was used.

| Trial | Results received / jobs submitted | Worker inference (s) | Capture to control poll (s) | Accepted motions | Stops without continuation |
|---|---:|---:|---:|---|---:|
| Initial instrumented run | 4 / 5 | 1.682–4.761 | 1.701–4.805 | Two SEARCH turns | 2 |
| Worker-finish and receipt instrumented separately | 6 / 7 | 1.696–2.506 | 1.750–2.555 | Three SEARCH turns | 3 |

The first result in the initial run exceeded the 3 s capture-age bound. In the second run the delay from worker finish to the parent preparing the result was 0.6–4.2 ms. Each accepted turn selected 0.524 rad with 0.349 rad remaining to brake at 2 rad/s, a **0.175 s motion horizon**. No incoming decision arrived while one of those plans remained active. Across the two runs there were five path-exhaustion stops, zero moving replacements and zero forward travel. A moving-handover fraction has no denominator in these learned runs. All ten delivered results reported target absence. The chair silhouette was drawn in six of their exact source images. All accepted motions came from the SEARCH policy. Both episodes ended at the evaluator deadline without a native STOP. The last outstanding request in each run did not enter control.

The second run performed 241 guard ticks. The p95/max tick gaps were 53.6/54.9 ms, and the proxy rotated 1.047 rad in total. Cold model load took 66.52 and 53.54 s in the two launches. Peak sampled MPS driver allocation during delivered inference reached 8.82 GiB. The first process reported a 10.35 GiB lifetime peak RSS at load. These counters measure different allocations. The read-only loopback viewer also returned successive control ticks during the second run.

| Other diagnostic | Available motion and result |
|---|---|
| 150 ms fixture | Two new deliveries arrived while moving. One became a compatible forward replacement. Two successive forward plans had 0.470 and 0.320 s to brake when issued. The second had 0.420 m remaining to its endpoint after accounting for pose change. |
| Missing continuation | The proxy exhausted the last forward path after 0.470 m, commanded zero and held through the deadline. No motion budget or native waypoint was extended. |
| Target disappears | APPROACH braked, waited for the stationary dwell and a new observation, and returned to SEARCH. All five allowed plan slots were consumed. |
| Recorded forward trials | Saved model-worker durations were 1.676 and 1.728 s. Capture-to-result ages were 1.965 and 2.002 s. The strict replay had zero admissible motion. In the explicitly conditional radio-timestamp pass, both plans had 0.600 m remaining to endpoint and 0.500 s to brake at issuance. Each archive supplied only one action refresh. |
| Memory stress | 64 measured fixture episodes after four warm-ups, 4,096 ticks, 192 plans and 64 moving replacements. Retained traced Python memory grew by 15,338 bytes after collection. RSS grew by 49,152 bytes. The writer queue peaked at 9 of 16 rows. This test measures the controller and recorder. Long-duration model-residency stability is unmeasured. |

In these samples even the fastest learned inference exceeded the admitted motion horizon. The synthetic forward replacement demonstrates controller compatibility. It supplies no learned or physical continuous-motion success rate. The open-loop recorded poses and LiDAR describe the historical action only.

### Private artifacts

All paths below are relative to `$GO2_WORKSPACE/runs/continuous`. Each run's `summary.json` names its trace files. Learned runs retain the worker identity, original frames, jobs, context, native text, waypoints, memory and timing under `model-01/`.

| Evidence | Directory or location |
|---|---|
| Eight fixture scenarios | `simulate-tz26_wqj/`. Case 0, tick 58 is the moving handover. Case 7, tick 63 is the exhausted forward path. Case 6 contains target loss. Tick numbers here are zero-based. |
| Learned closed loop | `simulate-vfqw4f2q/` and `simulate-8w_itx2j/` |
| Recorded comparisons | `replay-bsgvnqxg/`. Use the pass selector to inspect the strict and conditional results. |
| Memory benchmark | `memory-ji6tjk08/summary.json` |
| Verified browser captures | `viewer-uaae7ohb/forward-handover.png`, `path-exhausted.png`, `learned-search.png` |

### Evidence interface for Claude

The five-move hardware publication is `918f48c7880d1ade20034a1ab37664fd3fa2188e`. Claude's existing private preparation record is `$GO2_WORKSPACE/evidence/five_move_pending_trial.md`. The published template currently returns `awaiting_claude`. Use a private copy for the completed report. [Versioned JSON Schema](../src/go2_perception/vln/schemas/continuous-evidence-v1.schema.json) · [Pending template](../examples/continuous_navigation/pending_evidence.json)

```bash
"$py" scripts/check_continuous_evidence.py examples/continuous_navigation/pending_evidence.json
# After Claude supplies the report and bundle
"$py" scripts/check_continuous_evidence.py "$private_report" --bundle-root "$private_bundle"
```

Record the tested navigation, canonical controller and SDK commits in full, the episode ID and reviewer. Each assessed topic includes its status, notes, measurements and references to SHA-256-verified local files. Omit unknown measurements and mark unassessed topics explicitly.

| Topic | Measurements for the handoff |
|---|---|
| Tested motion | Actually attempted primitives, rates, budget, attempts, independent travel if measured |
| Stopping | Why movement ended, measured coast, zero confirmation, release result and post-release stillness |
| Camera and LiDAR continuity | Separate frame/scan counts, gaps, drops, boot/clock epochs and measured exposure uncertainty |
| Control and guard failures | Rejection reasons, exact veto snapshots, individual radio receipt timestamps and any control failure |
| Recognition and search | Search turns, independently grounded approaches, human instance and arrival reviews |
| Unsupported capabilities | Explicit gaps, including simultaneous forward/yaw, any unobserved turn sector and untested stopping conditions |

For a physical report the artifact IDs `run_manifest`, `episode_summary` and `session_summary` point to the closed session's `run.json`, `episode_summary.json` and `summary.json`. The validator cross-checks live mode, backend, controller and navigation revisions, terminal state and episode identity. Other artifact IDs can identify scans, model inputs, measured distances, review notes and command/guard logs. The verifier reports human-review eligibility and always returns `physical_activation: false`. Replay or synthetic bundles have a distinct disposition. [Evidence verifier](../scripts/check_continuous_evidence.py)

## Controller contract

One owner calls `ContinuousNavigator.tick(now, guard, delivery=..., cancel=...)` on a host-monotonic clock in seconds. The optional delivery contains a v2 proposal, grounded target evidence tied to its image hash and the time the result became available. The guard contains a separately delivered snapshot of pose, directional LiDAR clearance, camera clock epoch and capture uncertainty. The interlock carries the radio receipt time, neutral-input status, sport-state time, health and stationary status. An older verified idle-dwell record can establish initial stillness. A changed instruction, trial, session or camera epoch requires a new navigator. [Controller](../src/go2_perception/vln/continuous.py) · [V2 proposal format](SUPERVISED_NAVIGATION_CONTRACT.md)

| Parameter | Offline setting |
|---|---|
| Control cadence and packet lifetime | 20 Hz, maximum 100 ms per packet. Sensor expiry, source age, deadline and remaining primitive can shorten it. |
| Result lease and image age | 600 ms from worker completion when recorded, otherwise from logged result availability. The 3 s image-age limit includes capture uncertainty and continues throughout inference. |
| Motion | Separate forward or signed yaw packets at the existing 1.0 m/s and 2.0 rad/s. Waypoints are absolute in one capture-time ego frame, with metres forward/left and radians counterclockwise. |
| Episode bounds | At most five issued plans, including moving replacements. Default 30 s, 2 m inspected radius, 1.5 m cumulative odometry travel and 90° cumulative absolute rotation. The fixture uses a 12 s deadline. |
| Settling | An initial stationary dwell and a fresh settled view after braking. The November 19 profile and fixture use a one-second dwell. |
| Target and clearance | Five-move grounding, alignment and prefix checks, then independent current-pose, directional LiDAR and swept-area checks. A continuous turn requires observed sector coverage. |

A fresh proposal for the same mode and direction can replace the current plan while both pass the guards. The governor measures consumed forward distance or signed rotation from each proposal's capture-time pose. It checks the old plan first so a later result cannot revive expired movement. It charges a slot only when the new plan produces its first nonzero packet. Different directions or a SEARCH/APPROACH transition brake first, then require a new observation acquired after settling. Budget/deadline, operator intervention and epoch changes latch termination. A fresh in-scope native STOP latches `native_stop_arrival_unverified`. A command packet expiring while callbacks stall independently returns the proxy's velocity to zero. Odometry bounds here do not measure physical travel. [Shared SEARCH/APPROACH policy](../src/go2_perception/vln/episode_policy.py) · [Motion profile](../src/go2_perception/vln/motion_profile.py)

The fixture renders a simple chair into a synthetic pinhole image. A deterministic visual rule reads the mark and extent from those pixels. Each subsequent capture follows the proxy's actual integrated motion. The simulator advances while a result is pending. With `--wall-clock`, one background fixture worker waits independently while the main thread checks guards and records host timing. The two-second delay approximates the observed end-to-end age of these archived results. The fixture backend uses no model weights. The learned backend uses the cached, pinned worker described above. [Simulator](../src/go2_perception/vln/continuous_simulation.py)

## Measured results on November 19

These results are from the first controller publication, `93b08825`. The extended eight-case suite, learned runs and followable viewer are documented above.

The six scenarios below use the virtual clock. Every case entered SEARCH and APPROACH. All made the same 0.349 rad initial left turn. The path length is simulator truth, including travel during the final interval before a guard veto.

| Scenario | Plans issued | Replacements while moving | Path (m) | Time (s) | Outcome |
|---|---:|---:|---:|---:|---|
| Visible after search, 150 ms worker | 3 | 1 | 0.500 | 3.15 | The next projected segment exceeded the 2 m inspected-area envelope |
| Two-second worker | 2 | 0 | 0.470 | 10.00 | Inspected-area limit |
| Corridor becomes obstructed | 2 | 0 | 0.100 | 12.00 | First obstructed tick vetoed translation. Held through deadline |
| LiDAR drops out | 2 | 0 | 0.100 | 12.00 | First unavailable tick returned zero. Held through deadline |
| Operator cancellation | 2 | 0 | 0.100 | 2.75 | Cancellation latched on the next tick |
| Control callbacks interrupted | 3 | 0 | 0.350 | 4.70 | Watchdog gave six zero-velocity samples in the gap. A fresh settled plan resumed later and reached the area limit |

The short-delay run spent eight moving intervals awaiting another decision. The two-second run spent fourteen, with zero uninterrupted plan replacements. The inspected-area veto includes the next segment, stopping reserve, footprint and growing odometry allowance. Both workers stopped short of modeled arrival. These runs measure control behavior with a visual fixture. No learned closed-loop navigation success was measured.

**Host-paced check.** On the Apple Silicon Mac, the two-second background fixture produced observed capture-to-poll ages of 2.048–2.058 s. The guard loop ran 206 ticks over 10.26 s. Its 95th-percentile and maximum intervals were 56.93 and 60.00 ms, respectively. The proxy traveled 0.470 m and the area check ended the run. This is a measurement of one host run with logging enabled.

**Saved LightNav decisions.** Both recorded refreshes used five causal frames. Replay checked the original worker job, native result, image hashes, observation/receipt clocks, cutoff and earlier reviewed image/ROI. It re-ran instance matching on the original rectified model images. The source target entry and later instruction refresh have different trial IDs. Their instruction, session, review and camera epoch bind them. Each pass joined logged guards only once their check time was available and held on stale samples. The final 0.55 s tail exercises expiry using the last available sample. Twenty source files per session matched their recorded or pre-replay hashes at the end.

| Closed single-action session | Capture to saved result (s) | ORB inliers | Guard records | Strict nonzero ticks | Conditional nonzero ticks |
|---|---:|---:|---:|---:|---:|
| `trial-C3jMfl` | 1.965 | 152 | 39 | 0 | 10 |
| `trial-abZbVx` | 2.002 | 294 | 43 | 0 | 9 |

Both original traces store the radio stick/button values and omit the individual receipt timestamp. The strict pass emitted zero throughout. The separately labeled conditional pass explicitly assigned the guard check time to the radio sample, retained its saved values and all other checks, and emitted one hypothetical forward plan per recording. Both plan leases expired. The original camera bound of 0.159 s is specific to the measured boot. Each archive contains only one refresh for this action. Its later odometry and LiDAR follow the historical action. They cannot demonstrate moving policy replacement, actuator stopping distance or arrival. The operator-excluded earlier forward run was outside this comparison.

The private output root is `$GO2_WORKSPACE/runs/continuous`. The verified virtual cases are in `simulate-rfa3vdx7`, the two recorded replays in `replay-xqis_ctz`, and the measured host-paced case in `simulate-og31wcgb`. Open `summary.json` in each directory to locate every trace. Public Git content contains code, tests and aggregate results.

## Regression results

On the completed branch, **942 passed, 0 failed, 2 skipped and 19 were deselected** in 91.23 s. The focused controller, planner, private replay, recorder, evidence and viewer selection passed all **67 tests in 1.85 s**. The two skips are the optional MuJoCo modules in the core environment. The marker selection deselected one physical Go2 case and 18 model-integration cases. The two learned runs above exercised the cached model separately. [Focused controller cases](../tests/test_continuous_navigation.py) · [Planner, viewer and handoff cases](../tests/test_continuous_integration.py)

At the first controller publication, the macOS 26.6.2 arm64 Python 3.12.11 core environment passed all 926 selected tests in 86.20 s. Two optional MuJoCo modules skipped because the core environment lacks MuJoCo. The marker selection excluded one physical hardware test and 18 model-integration cases. The suite includes 51 continuous-navigation cases, including real-archive provenance and tamper tests on this Mac. On a machine without the private sessions, those four archive cases skip with their reason.

The first suite run found that a legacy viewer privacy test invoked a missing executable named `python` after the workspace migration. The test now starts the active interpreter with a known synthetic scene and verifies the actual privacy refusal. [Regression cases](../tests/test_continuous_navigation.py) · [Viewer privacy test](../tests/test_fsd_viewer.py)

## Hardware handoff

Claude's next physical evidence belongs to the pinned five-move baseline. The agreed pilot calls for a closed two-move, 45 s run in an inspected static area. Retain the original target image and review, model result and capture bound for that boot, requested versus settled motion, in-motion guard vetoes, stop/release and post-release stillness, and the operator's arrival assessment. For continuous-mode review, retain the radio receipt timestamp with every fast guard, record actual sensing and result gaps during motion, and independently measure travel and coast. Review that evidence before designing a physical command interface. [Exact published five-move pilot and launch procedure](FIVE_MOVE_NAVIGATION.md#smallest-physical-pilot-for-claude)
