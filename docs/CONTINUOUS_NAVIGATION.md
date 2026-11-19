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
"$py" -m pytest -q -p no:cacheprovider tests/test_continuous_navigation.py
```

For the full offline selection, set `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `TOKENIZERS_PARALLELISM=false`, `OMP_NUM_THREADS=1` and `VECLIB_MAXIMUM_THREADS=1`. Run `"$py" -m pytest -q -rs -p no:cacheprovider -m 'not hardware and not model_integration'`. The mock browser tests use local loopback. MuJoCo rendering and pretrained model integrations have their own environments.

Use `--scenario NAME` to replay one synthetic case, or `--output NEW_DIRECTORY` to choose the destination. The recording importer checks the destination's Git privacy and rejects writes inside the input session. Output consists of a summary and a JSONL trace per case. Each tick carries the measured/virtual time, pose and geometry, a hypothetical packet, and any selection or guard veto. The veto contains the snapshot checked at that tick. The capture and delivery rows retain image identity, selection evidence, the native proposal and capture/result times. The recorded pass retains file hashes. [Offline entry point](../scripts/run_vln_continuous.py)

## Controller contract

One owner calls `ContinuousNavigator.tick(now, guard, delivery=..., cancel=...)` on a host-monotonic clock in seconds. The optional delivery contains a v2 proposal, grounded target evidence tied to its image hash and the time the result became available. The guard contains a separately delivered snapshot of pose, directional LiDAR clearance, camera clock epoch and capture uncertainty. The interlock carries the radio receipt time, neutral-input status, sport-state time, health and stationary status. An older verified idle-dwell record can establish initial stillness. A changed instruction, trial, session or camera epoch requires a new navigator. [Controller](../src/go2_perception/vln/continuous.py) · [V2 proposal format](SUPERVISED_NAVIGATION_CONTRACT.md)

| Parameter | Offline setting |
|---|---|
| Control cadence and packet lifetime | 20 Hz, maximum 100 ms per packet. Sensor expiry, source age, deadline and remaining primitive can shorten it. |
| Result lease and image age | 600 ms from result availability. The existing 3 s image-age limit includes the capture uncertainty. Both run during inference. |
| Motion | Separate forward or signed yaw packets at the existing 1.0 m/s and 2.0 rad/s. Waypoints are absolute in one capture-time ego frame, with metres forward/left and radians counterclockwise. |
| Episode bounds | At most five issued plans, including moving replacements. Default 30 s, 2 m inspected radius, 1.5 m cumulative odometry travel and 90° cumulative absolute rotation. The fixture uses a 12 s deadline. |
| Settling | An initial stationary dwell and a fresh settled view after braking. The November 19 profile and fixture use a one-second dwell. |
| Target and clearance | Five-move grounding, alignment and prefix checks, then independent current-pose, directional LiDAR and swept-area checks. A continuous turn requires observed sector coverage. |

A fresh proposal for the same mode and direction can replace the current plan while both pass the guards. The governor measures consumed forward distance or signed rotation from each proposal's capture-time pose. It checks the old plan first so a later result cannot revive expired movement. It charges a slot only when the new plan produces its first nonzero packet. Different directions or a SEARCH/APPROACH transition brake first, then require a new observation acquired after settling. Budget/deadline, operator intervention and epoch changes latch termination. A fresh in-scope native STOP latches `native_stop_arrival_unverified`. A command packet expiring while callbacks stall independently returns the proxy's velocity to zero. Odometry bounds here do not measure physical travel. [Shared SEARCH/APPROACH policy](../src/go2_perception/vln/episode_policy.py) · [Motion profile](../src/go2_perception/vln/motion_profile.py)

The fixture renders a simple chair into a synthetic pinhole image. A deterministic visual rule reads the mark and extent from those pixels. Each subsequent capture follows the proxy's actual integrated motion. The simulator advances while a result is pending. With `--wall-clock`, one background fixture worker waits independently while the main thread checks guards and records host timing. The two-second delay approximates the observed end-to-end age of these archived results. Caches, model weights, calibration and DDS are outside this runner. [Simulator](../src/go2_perception/vln/continuous_simulation.py)

## Measured results on November 19

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

On macOS 26.6.2 arm64, the existing Python 3.12.11 core environment passed all 926 selected tests in 86.20 s. Two optional MuJoCo modules skipped because the core environment lacks MuJoCo. The marker selection excluded one physical hardware test and 18 model-integration cases. The suite includes 51 continuous-navigation cases, including real-archive provenance and tamper tests on this Mac. On a machine without the private sessions, those four archive cases skip with their reason.

The first suite run found that a legacy viewer privacy test invoked a missing executable named `python` after the workspace migration. The test now starts the active interpreter with a known synthetic scene and verifies the actual privacy refusal. [Regression cases](../tests/test_continuous_navigation.py) · [Viewer privacy test](../tests/test_fsd_viewer.py)

## Hardware handoff

Claude's next physical evidence belongs to the pinned five-move baseline. The agreed pilot calls for a closed two-move, 45 s run in an inspected static area. Retain the original target image and review, model result and capture bound for that boot, requested versus settled motion, in-motion guard vetoes, stop/release and post-release stillness, and the operator's arrival assessment. For continuous-mode review, retain the radio receipt timestamp with every fast guard, record actual sensing and result gaps during motion, and independently measure travel and coast. Review that evidence before designing a physical command interface. [Exact published five-move pilot and launch procedure](FIVE_MOVE_NAVIGATION.md#smallest-physical-pilot-for-claude)
