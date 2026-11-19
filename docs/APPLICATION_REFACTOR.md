# Application refactor and verification

Branch `codex/readable-navigation-app` starts at `91e98390baea3ec535a538d1fa5f3edae7024ae4`. The complete previous tree, including historical CLIs and tests in their original locations, is available at [the published parent](https://github.com/binbin-ovo/Go2_Navigation/tree/91e98390baea3ec535a538d1fa5f3edae7024ae4). The branch reuses the existing development worktree.

## Source tree

Counts below use tracked Python, shell and PowerShell source paths. The two root aliases and four legacy tools have their own rows. HTML, CSS, configuration, documentation and binary data are outside this measure.

| Location | Before | After | Purpose |
|---|---:|---:|---|
| Root Python launchers | 0 | 1 | `app.py` |
| Root links to packaged source | 0 | 2 | `sensors.py`, `navigation.py` |
| `src/` | 155 | 87 | Packaged code, including common loaders and retained regression dependencies |
| `scripts/` | 106 | 20 | Process entries, operator compatibility commands and offline tools |
| `hardware/` | 6 | 7 | Pinned guards, evidence tools and one common operator shell launcher |
| Application source excluding links, tests and legacy | **267** | **115** | Unique implementation files present in the application checkout |
| `legacy/` | 0 | 4 | Historical CLI paths still exercised by current checks |
| `tests/` including Python fixtures | 66 | 34 | Current application and supporting regressions |

The potential launch dependency graph across all modes contains 69 source paths in the parent and 77 on this branch. This count includes conditional Python imports, package initializers, operator shell scripts and separately launched workers. Extracting the sensor, session and UI code gives those responsibilities explicit filenames. The larger checkout reduction comes from taking separate experiments off the application branch. The [inventory](CODE_INVENTORY.md) gives the original paths, classifications and reasons.

| Decision | Files and behavior |
|---|---|
| Merge target decisions | `navigation.py` owns target normalization and the SEARCH/APPROACH policy. `interactive_trials.py` uses the same target helpers. The previous `vln.episode_policy` import path re-exports the public functions. |
| Merge sensor setup | `sensors.py` owns source settings, source creation, lifecycle and read-only snapshots. The bounded camera ring and paced feeder moved out of `interactive_app.py`. RTP and DDS reception retain their separate interpreter processes. |
| Merge runtime and display plumbing | `session_runner.py` serves observation and supervised sessions. `bound_proposal.py` checks evidence for both supervised strategies. The episode UI no longer inherits a one-action state machine or constructs an imitation cycle. One HTML template defines the panels and image-binding hooks. |
| Merge common operator arguments | Both published shell commands call `hardware/launch_supervised.sh`. Both published Python CLI paths still work. The command owner, lock, measured guard provider, native model worker, decoder and calibrated preprocessing retain their existing implementation. |
| Retain distinct policies | Single-action review and refresh, the five-move grant and settled observations, and the continuous offline segment lifecycle retain separate implementations. |
| Separate earlier work | Forecast training, tracking studies, scene rendering, generic grounding experiments and earlier waypoint suites remain together on the published parent. Four historical tools required for current regression checks moved to [legacy](../legacy/README.md). |

## What was compared

| Evidence | Result |
|---|---|
| Identical focused selection before and after | 297 passed before (66.01 s) and 297 passed on the final tree (65.22 s). Includes mock supervision, image binding, real-cloud replay, stale inputs, interruptions, guard failures and stop/release. |
| Same five-move mock through browser API, worker and executor | Four current-image hashes, instructions, native tokens, native waypoint arrays, grounding, selected magnitudes and cleanup flags matched. SEARCH left 30°, APPROACH left 20°, APPROACH forward 0.30 m, then native STOP with arrival unverified. Zero physical sends. The published regression fixture comes from the parent run. |
| Two freshly paced mock episodes | Session IDs, host times and some frame-grid indices differed as the real-time feeder ran independently. These quantities are retained in each archive. The deterministic comparison checks the model's current pixels and decisions. |
| Closed physical box recording read offline | Six source-checked decisions produced the same digest before and after, over the episode summary, exact instructions, selected frame histories and hashes, native outputs, reviews and executor evidence. P002 remained a rejected instance. P003 remained a SEARCH forward. The original data stayed in the workspace. |
| Deterministic continuous fixture | Complete JSONL traces were byte-identical, SHA-256 `d41e28294067e4e130dbf60d16db95b77dd9fd4ad53f4564bb43bc77fb406db7`. The proxy ran 64 control ticks, accepted three plans and made one moving replacement. It stopped at the inspected-area bound. |
| Website in headless Chrome | The default site showed the camera. Selecting five-move started an idle episode on the same address. Typing *chair* produced a displayed mock forward selection. Continuous mode displayed its proxy camera and geometry, and its stop control produced `cancelled`. |
| Resources and supervision | The local source imports resolved to this checkout. All 19 cached checkpoint paths and sizes matched the inventory. Checksums matched all 17 files smaller than 1 MiB. The large weights were not loaded. The core, hardware and model interpreters, SDK, checkpoint identity and cached model source were present. Both shell routes passed file-only preparation with a temporary output workspace. The actual workspace capability check resolved the single controller lock and matched the 1.0 m/s and 2.0 rad/s rates. The browser rejected an unknown live mode and discarded old session credentials. |
| Complete retained core suite | **558 passed, 0 failed, 13 skipped in 83.69 s**. The 13 skips comprised 12 isolated native-runtime checks and one physical Go2 test. |
| Isolated model-environment checks | **12 passed in 5.75 s**, offline. These exercised native frame selection, history/reset, preprocessing and small CPU/MPS pooling tensors. Learned generation and physical activation were outside this refactor. |

### Reproduce the checks

Start in the checkout with the environment in [the walkthrough](CODE_WALKTHROUGH.md#choose-inputs). Use fresh directories for mock and continuous artifacts.

```bash
export PYTHONDONTWRITEBYTECODE=1 MUJOCO_RENDER_TEST=0 HF_HUB_OFFLINE=1
python -m pytest -q -rs -p no:cacheprovider
python -m pytest -q -p no:cacheprovider tests/test_application.py
scratch=$(mktemp -d)
python scripts/demo_vln_supervised_mock.py --five-move-episode --output "$scratch/five"
python scripts/run_vln_continuous.py simulate --scenario visible_after_search --output "$scratch/continuous"
shasum -a 256 "$scratch/continuous/trace-01.jsonl"
LIGHTNAV_ASSETS="$GO2_WORKSPACE/weights/lightnav" TRANSFORMERS_OFFLINE=1 \
  "$GO2_WORKSPACE/envs/model/bin/python" -m pytest -q -rs -p no:cacheprovider \
  tests/test_bounded_native_integration.py tests/test_lightnav_runtime_integration.py
```

The exact paired selection used in the parent and refactor was the following.

```bash
python -m pytest -q -p no:cacheprovider -m 'not hardware and not model_integration' \
  tests/test_vln_interactive.py tests/test_five_move_navigation.py tests/test_effective_motion.py \
  tests/test_motion_diagnostics.py tests/test_hardware_guards.py tests/test_supervised_contract.py \
  tests/test_supervised_executor.py tests/test_supervised_integration.py tests/test_supervised_navigation.py \
  tests/test_continuous_navigation.py tests/test_continuous_integration.py tests/test_target_grounding_review.py \
  tests/test_vln_live_shadow.py tests/test_go2_replay.py
```

### Intentional user-visible changes

The default application now opens a loopback website in observation mode and accepts browser input even when stdin is closed. A mode switch rotates the session token, resets history and selects a fresh run directory. The public stop control cancels the applicable advisory, mock episode or simulation. Technical telemetry is expandable. Sensor access returns copies to public callers. Camera status adds liveness and error fields, and continuous review also exposes saved geometry and grounding when available. The previous operator CLIs, grant checks, bounded action rules and recorded native evidence retain their meanings.

### Boundaries for the handoff

Claude's active checkout still owns the local boot-specific timing adjustment. The box trial used its measured 0.140 s bound. The development provider's 0.159 s value belongs to an earlier boot. No boot-specific change was copied, no calibration was performed, and physical activation of this branch remains untested. Full RGB–LiDAR registration, independent distance measurement and real continuous handovers remain unresolved. The model's raw mark alone cannot identify a target. Historical native inference and physical outcomes remain attributable to their original runs.
