# Five-move navigation · hardware validation

This record covers the November 19 supervised box episodes, the offline readiness of the integrated application and the procedure for the next physical session. Hardware observations are kept apart from interpretation. The operator's words are quoted exactly. Private images, recordings and logs stay under `$GO2_WORKSPACE/runs/five_move/`.

## Source and configuration tested on November 19

| Item | Value |
|---|---|
| Navigation | `918f48c` (`feature/vln-five-move-navigation`) plus the then-uncommitted boot bound, since committed as `96282e9`. `run.json` records `git_commit: 918f48c` and the implementation hashes. |
| Controller / SDK | `code/control` `6d555d3`, SDK `814556d`, single `.command.lock` |
| Profile | `go2-effective-v1`: forward 1.0 m/s, turn 2.0 rad/s, separate primitives, native avoidance on |
| Camera bound | 0.140 s at boot offset −1766856813.961 s (`evidence/hw_readiness/capture_latency_20261119_113103.json`), valid only for that boot |
| Model | LightNav-0, revision `826dc5fb`, MPS, bfloat16, bounded history (first two + latest six frames) |

The robot had rebooted since the earlier 0.159 s measurement, so the bound was re-measured before the first episode.

## Episodes

"Correct" and "wrong" mark judgments are Claude's reading of the exact model image (panel B) during the episode. They are not an independent detector. Operator-role identify and reject clicks were made by Claude on the user's behalf, because the episode deadline left no time to hand them over.

| Run (`runs/five_move/…`) | Setup · limits | Native marks (image x, y) | Identity decisions | Movements executed | Outcome | Operator's words |
|---|---|---|---|---|---|---|
| `trial-cXouta` | Box out of view · 2 moves, 45 s, 2 m | P001 (0.51, 0.61), P002 (0.51, 0.65) on a bag pile at the hallway end: **wrong** | P002 awaiting identification | none | `SENSOR_OR_CONTROL_FAILURE · camera_epoch_changed`. The Mac wall clock stepped −80 ms against the bridge's 20 ms tolerance (macOS time correction). | none (no motion) |
| `trial-cdyo3G` | Same | P001–P003 (0.51, 0.57), P004–P005 (0.51, 0.54) in the hallway: **wrong** | P002 rejected. P003/P004 then `previously_rejected_instance`, so no approach was authorized. | SEARCH forward 0.30 m ×2 (feedback 0.415 m overshoot, 0.382 m). Stops verified. | `MOVE_BUDGET_EXHAUSTED`. The box never entered view. | "Never saw the box" · "The Go2 detect wrong object and not able to locate the box, which is now on its right side" |
| `trial-231Apz` | Box visible near centre, about 2 m · 5 moves, 60 s, 2 m | P001–P003 (0.51, 0.61) **on the box**. P004–P005 (0.55, 0.69) **on the box** after moving. | P002 identified. P003 `supported` by instance features. P004 needed re-identification (matcher did not carry over the closer view). P005 supported. | APPROACH `native_prefix` forward 0.60 m: feedback 0.675 m, odometry 0.585 m, heading change +0.003 rad. Stop verified. | `BLOCKED_OR_NO_USEFUL_ACTION · outside_inspected_area`. The next step's projected path plus allowances exceeded 2 m. No native STOP. | "Good approach" |
| `trial-bYheuc` | Box about 14° right · 5 moves, 60 s, 3 m | P001–P003 (0.656, 0.611) **on the box**. Native path yaw −0.31 rad at row 0, curving to about −0.5 rad. | P002 identified. P003 `supported`. Bearing 0.245 rad, inside the ±15° band, so treated as aligned. | none | `BLOCKED_OR_NO_USEFUL_ACTION · native_path_does_not_support_forward_approach` | "Didn't move" |
| `trial-ivBOES` | Box at the far right edge of the live view · 5 moves, 60 s, 3 m | P001–P002 (0.53, 0.54) on the hallway floor: **wrong**. The box was outside the rectified model image. | none | none | `OPERATOR_ABORT`. A person walked into the inspected area, and Claude pressed STOP EPISODE. | none |
| `trial-THKUcr` | Box at about 0.94 of live width · 5 moves, 60 s, 3 m | P001 **not visible** (`opos_0`), but its path turned right toward the box (row 0 yaw −0.31, end yaw −1.06 rad). P002–P003 (0.53, 0.54), P004 (0.59, 0.54), P005 (0.51, 0.54), P006 (0.45, 0.65) in the hallway: **wrong**. | P002 rejected. P004 recorded as inconclusive ("seek new view"). P006 unanswered. | SEARCH forward 0.30 m ×2 (feedback 0.340, 0.317 m). Yaw drifted +0.05 then +0.16 rad, to the left and away from the box. Stops verified. | `EPISODE_DEADLINE` (60 s) at 2/5 moves | "wrong box detected." |

Across all six runs:

- No guard rejections occurred during motion.
- Every executed move ended with verified stop and settling.
- LiDAR chunks were recorded through every episode.
- Inference took about 2.0 s per decision. The first decision of a runtime took about 2.6 s, and once 14 s.

The first runtime launch stopped before authorization: available memory dipped to 1.9 GiB while the model loaded. A relaunch loaded cleanly.

### What the hardware showed

- **Target marks.** Marks fell on the box in 2 of 6 runs, both with the box near the image centre. In all three runs where the box was out of view or at the edge, LightNav still placed a mark, usually near image column 0.51 at the hallway end. It reported "not visible" only once (THKUcr P001). The x coordinate was exactly 0.510 in 11 of 23 decisions and 0.531 in 4 more, across different scenes.
- **Identity gate.** Every wrong mark was stopped before APPROACH: by operator rejection, then by the matcher's `previously_rejected_instance` on later views. No wrong mark authorized movement toward it. The automatic matcher did not re-identify the box after a 0.6 m approach (231Apz P004), so a second human identification was needed.
- **SEARCH.** The search policy chose only straight 0.30 m `inspect_new_viewpoint` steps (4 executed). It never chose a turn, including when the box was known to be on the right.
- **APPROACH.** One forward approach worked as intended (231Apz). A target 14° off-centre, inside the alignment band but with a turning native path, produced no action at all (bYheuc).
- **Area check.** A 2 m radius with a box about 2 m away leaves room for only one approach step, once the stopping reserve (0.25 m), footprint (0.65 m) and uncertainty allowance (0.15 + 0.05 × moves) are counted.
- **Operator timing.** Each identification pause costs 5–10 s of the 60 s episode. At most 2 of 5 moves fit in one episode.
- **Not established.** Independent arrival at the box (no native STOP occurred, and there was no external measurement), combined turn-plus-forward motion, and any continuous-motion behaviour. Feedback distances are the controller's own estimates.

## Offline readiness of the integrated application (current checkout)

`code/navigation` is on `hardware/readable-navigation-integration` at `2fa98f2`. The integration checks are listed in `AGENT_HANDOFF.md`. Re-checked on that checkout:

- `app.py --source replay --episode tests/fixtures/vln/episode_v1/episode.json --lidar-recording tests/fixtures/vln/synthetic_recording` serves the website with replayed RGB and LiDAR, in REPLAY · ADVISORY ONLY mode. It shuts down on SIGTERM with no child processes left.
- The panels keep the raw model mark ("unverified model proposal") separate from instance identification.
- The five-move mock records zero physical sends. The continuous mode runs only on its proxy and has no executor.
- `bash scripts/launch_five_move_episode.sh --prepare-only` writes a capability that pins this checkout's `hardware/supervised_guards.py`, `code/control` and `envs/hw`.

**Blocking item, not a code fault.** The Mac has **17 GiB free (97% full)**. The observation website's default floor is 20 GiB, and `app.py` has no option to change it. New observation trials therefore show `STOPPED: free disk … below limit` until about 3 GiB is freed. The supervised launcher passes `--min-free-disk-gb 10` and is not blocked. The historical Desktop repository (about 50 GB of history) is the largest candidate to free, and needs the user's decision.

## Next physical session

Movement requires a new, explicitly arranged supervised session and one start confirmation per episode.

**Before any episode**

1. Free disk to at least 20 GiB (the website) or 10 GiB (the supervised launcher alone). Close large apps so that at least 12 GiB of memory is available while the model loads.
2. Power on and stand the Go2, connect `en9`, and ping 192.168.123.161.
3. **Boot check (read-only).** Compare `host_monotonic − rt/utlidar/robot_odom stamp` against the bound in `hardware/supervised_guards.py`. If it differs by more than 5 s, place the Mac screen in front of the camera and run `hardware/capture_latency.py` (envs/hw). Then update `CAMERA` and its test, commit, and re-run preparation. Never reuse an earlier boot's value.
4. `cd ~/Robotics/Go2/code/navigation && bash scripts/launch_five_move_episode.sh --prepare-only`
5. Launch: `GO2_SUPERVISED_LIVE=1 bash scripts/launch_five_move_episode.sh --live-exclusive --move-budget 5 --episode-seconds 60 --area-radius-m 3 --alignment-deg 15`. Open http://127.0.0.1:8766 and confirm that camera, LiDAR and the model are ready.

**Scenarios.** Use a printed or textured box, because the matcher needs ≥ 12 feature inliers. Keep everyone outside a clear 3 m radius, hold the cable slack and the remote stop, and relaunch between episodes (each executor grant is single-use).

| # | Scenario | Placement | What it tests |
|---|---|---|---|
| 1 | Visible distinctive box | About 1.3–1.5 m ahead, near centre | Mark → identification → APPROACH, and whether a second step fits in the area |
| 2 | Box near the image edge | About 25–35° right, fully inside the **model** view (live x ≈ 0.80–0.85, not 0.95) | Coarse turn selection, then re-observation and approach |
| 3 | Wrong-object distractor | Box visible plus a second similar box or bag at a similar distance | Rejection of the wrong instance, and whether the matcher keeps it rejected |
| 4 | Target initially out of view | Box beside or behind the start, about 90° | Whether SEARCH ever turns toward an unseen target |

**During each episode**

- Answer identification prompts quickly. Draw the region first, then choose a button.
- Never approve a mark that is not on the requested object.
- Press STOP EPISODE for any person, cable or scene change in the area.

**After each episode**

- Stop the runtime (`quit`).
- Ask the user: correct object? useful views? progress? wasted moves? stopping reasonable? Save their words verbatim in `operator_assessment.json`, separately from the interpretation.

## Evidence captured for Codex

| Item | Where it is recorded |
|---|---|
| Typed instruction and normalized target | `session/trials/*/trial.json` and `predictions.jsonl` (`instruction`). The refactored `target_spec` also keeps `entered_target_text`. |
| Images sent to the model | `session/requests/NNNNNN/native/input_*.png`, with SHA-256 in each prediction's `source_image` |
| Native point, absence, waypoints, STOP, tokens | `predictions.jsonl` (`native_output`, `target_fraction`, `opos_not_visible`, `waypoints`, `native_stop`) |
| Identification or rejection | `executor/executor_events.jsonl` `episode_observation` (`grounding.status`, `method`, `operator`) |
| SEARCH/APPROACH selection | `episode_state.selected` and `episode_dispatch` (source, basis, magnitude) |
| Movement and stopping | `episode_summary.json` `attempts[]` (requested, feedback, termination, stop verification, decision and settled odometry), `executor/*`, `lidar/index.jsonl` |
| Human judgment | `operator_assessment.json` in each run folder |
| Source and configuration | `run.json` (`git_commit`, `implementation_sha256`, runtime/calibration hashes, `canonical_control_revision`) and `capability.json` |
| Errors | `events.jsonl`, `session/events.jsonl` (for example `source_epoch_reset`), executor `guard_rejected` |
| Offline review | `scripts/review_target_grounding.py --session RUN/session --prediction ID`, or `scripts/run_vln_interactive.py --review-run RUN/session` |

A plausible native trajectory is not evidence that the correct object was found. Only an image-bound identification or the operator's judgment establishes the target.

## Compatibility notes for Codex

- The source commit in `run.json` does not show uncommitted edits. Commit boot-specific guard changes before a session, so the recorded commit is exact.
- `sensors.py` cannot replay the LiDAR recorded by five-move sessions (`lidar/chunk_*.npz` plus `index.jsonl`). It reads only `go2-recording-v1`. Offline grounding replays of these sessions therefore have RGB and decisions but no replayed LiDAR stream.
- `app.py` exposes no `--min-free-disk-gb` or memory limits, so the 20 GiB website floor currently blocks observation on this Mac.
- `app.py supervised --live-exclusive` cannot pass episode limits; it always uses 5 moves, 60 s, 2 m and 15°. Use the shell launcher for the 3 m area.
- Model tests: running `tests/test_bounded_native_integration.py` before `tests/test_lightnav_runtime_integration.py` in one process fails one case, because the two files load different LightNav source copies. Run them separately or in reverse order.
- The live camera epoch is tied to the host wall clock with a 20 ms step tolerance. A macOS time correction ends an episode as a sensor failure.
- A new target-verification behaviour must change only `navigation.py` and `episode_grounding.py` (or new modules behind them). The guard provider, motion profile and executor are hash-pinned in the capability. Any change to them needs a new capability and a hardware check before use on the robot.
