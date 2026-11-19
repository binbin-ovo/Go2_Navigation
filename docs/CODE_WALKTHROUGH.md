# From a camera image to a Go2 decision

One model answers a visual question and proposes a path. Separate Python code decides whether the named object has been identified, whether a movement would be useful, and whether the available sensors support it. The historical box episode used the bounded five-move controller. The continuous controller has only been exercised offline.

## A 25-minute reading route

| Minutes | Read in this order | Follow one input through to its output |
|---|---|---|
| 0–4 | [EpisodeUI.authorize and target_spec](../src/go2_perception/vln/episode_app.py) | A person enters an object description. The episode strips the optional “Go to the … and stop” wrapper, keeps the descriptive words, and records the instruction before inference. “brown cardboard box on the floor” becomes “Go to the brown cardboard box on the floor and stop.” New archives also keep the original text field. |
| 4–8 | [frame_observation](../src/go2_perception/vln/shadow_live.py), [PreparedFrameCache.prepare](../src/go2_perception/vln/shadow_cache.py), [select_context and sample_factory](../src/go2_perception/vln/bounded_context.py) | Timestamped distorted RGB plus measured camera intrinsics produce a rectified frame. The history policy samples a causal 4 Hz grid and keeps the first two and latest six distinct frames. For P002 these are indices 0, 1, 10–15. A frame arriving later cannot change that saved decision. |
| 8–12 | [run_job](../scripts/lightnav_worker.py), [target_fraction](../src/go2_perception/vln/shadow_display.py), [prediction_source_image](../src/go2_perception/vln/interactive_app.py) | LightNav uses the instruction and RGB history. Its trained weights predict a target grid cell and trajectory codes. The native decoder yields ten positions and headings in one decision frame. The on-screen mark for OPOS 698 is at (238, 137.48) on the actual 448 × 256 image, with the origin at top left. OPOS 0 means the model reports absence. |
| 12–16 | [ground, object_reference and locate](../src/go2_perception/vln/episode_grounding.py) | An image-bound human identification or a pre-named reference photo supplies features inside a rectangle. The grayscale ORB matcher tests whether enough of those features reappear around the new model point. A rejected instance and an identified instance have different roles. A reference named “box” cannot authorize “large brown box.” |
| 16–21 | [supported_ground and select_action](../src/go2_perception/vln/episode_policy.py), [EpisodeExecutor.observe and dispatch](../src/go2_perception/vln/five_move.py) | Python checks the image, target, pose, freshness, clearance and budget. An unverified mark can pause for identification. SEARCH may turn or take a bounded step for another view. APPROACH needs a currently supported instance. P003 produced a 0.30 m SEARCH step. A native STOP ends the episode with arrival unverified. |
| 21–25 | [Guards.snapshot and forward_clearance](../hardware/supervised_guards.py), [HardwareMotion](../src/go2_perception/vln/supervised_executor.py), [offline evidence reader](../src/go2_perception/vln/grounding_review.py) | LiDAR floor/obstacle returns and odometry constrain the next step and monitor progress while moving. Only the supervised executor can dispatch via its canonical command owner. For a replay, join the immutable observation, dispatch ticket and result by prediction and request ID. A send return, an odometry estimate, and an independent tape measurement answer different questions. |

### Read the box episode

Use the [offline replay command](../START_HERE.md). The HTML opens P002 and includes the original native PNG, all selected historical model frames, decoded coordinates and headings, the exact tokenized prompt as text, and per-prediction reviews and dispatch. The companion `trace.json` lists paths, hashes, timestamps and send returns. The files are written in a fresh `runs/target_grounding/trace-*` directory.

| Saved decision | What it shows |
|---|---|
| P001 | LightNav reported absence. The executor recorded `stale_or_future_observation` and held. |
| P002 | The stored request was “box”. OPOS 698 points into the hallway; the visible box is at the far right. The ten-point path ends about 1.507 m ahead in local model coordinates. The executor requested identification. The archived operator-role review rejected the marked instance. No movement was dispatched for P002. |
| P003 | The same mark reappeared on a fresh image. Its features matched the previously rejected instance. The controller chose SEARCH. It requested 0.30 m forward and recorded 11 movement sends and a settled estimate of 0.340 m. |
| P004 and P005 | P004 marked another hallway location and was rejected. P005 remained uncertain after that review. A second 0.30 m SEARCH step produced 12 movement sends and a 0.317 m settled estimate. |
| P006 | The next mark still needed identification. The 60 s deadline ended the episode with two of five slots used and zero supported target approaches. |

The two dispatch snapshots recorded zero turn clearance and forward clearance of 1.8 m and 2.0 m. Both moves reported stationary, released ownership and post-release stillness. The initial `run.json` describes startup. The final `summary.json` and executor results record 23 movement sends. Neither send counts nor settled estimates establish an independently measured travel distance. [Grounding evidence and limitations](TARGET_GROUNDING_REVIEW.md)

### What the object description enforces

The words *brown*, *cardboard*, *large*, *on the floor* and *near the refrigerator* go into the model prompt. The current pipeline has no independent category, color, material, physical-size or landmark-relation measurement for that point. LiDAR checks the traversable corridor. It does not establish that a particular marked pixel belongs to a box.

The reference check asks whether the **same named instance** was re-identified on the current image. A human can drag a tight region around the proposed point and choose **This is the requested target**, **This is a different object**, or **Unsure · seek another view**. Identify and Reject need a region containing the point. Unsure uses the image alone. A pre-identified local photo uses `--target-reference FILE --reference-target "brown cardboard box" --reference-roi X0 Y0 X1 Y1` on the supervised application. Coordinates are fractions of that photo. Match the full description, or identify the instance again under its new name. A plain cardboard face may have too few distinctive features. A printed face can help re-identification. Match counts describe texture correspondences and confer no semantic confidence. [Reference matcher](../src/go2_perception/vln/episode_grounding.py)

## Workspace and launch modes

On November 19 the workspace root was `/Users/yushan/Robotics/Go2`. The three Git worktrees share the repository in `code/navigation/.git`.

| Location | Contents at inspection |
|---|---|
| `code/navigation` | Claude's active `hardware/five-move-validation` at `918f48c7880d1ade20034a1ab37664fd3fa2188e`. Two local edits, `hardware/supervised_guards.py` and `tests/test_motion_diagnostics.py`, carry the boot-specific 0.140 s guard change. The recorded provider hash matches the active file. |
| `code/control` | Clean canonical controller `feature/go2-high-level-control-audit` at `6d555d31c6eaa4ee194920925ab96238bf15b128`. Its command lock location is `data/outputs/go2_control/.command.lock` under this checkout. Controller logs here are private. |
| `code/worktrees/codex-motion-diagnostics` | Review branch `feature/vln-target-grounding-review`, based on saved continuous publication `ebefe9d25c820765d66a363d3d30a17368b3b5d6`. Read the guide and run the offline tools here. |
| `runs` and `recordings/desktop_worktrees` | Current closed sessions, comparisons and earlier preserved recordings. The demonstrated session is in `runs/five_move/trial-THKUcr/session`. |
| `evidence`, `weights/lightnav`, `sdk`, `envs` | Per-boot timing/capability records, pinned LightNav weights and source, Unitree SDK `814556d`, and separate core 3.12.11, hardware 3.10.17 and model 3.11.13 interpreters. |

Before this review the workspace root README led with a motion-capable launcher and the repository README opened with the experimental controller. The root now points to the version-controlled start guide. The repository README links the walkthrough and a [historical phase index](PROJECT_HISTORY.md). All three worktrees had a distinct purpose. The controller has unique ignored logs, the active checkout has unpublished edits and installed-package metadata, and the recording archive contains unique preserved data. No directory needed retirement.

Use these environment settings for offline commands, including after opening a new terminal.

```bash
export GO2_WORKSPACE=/Users/yushan/Robotics/Go2
cd "$GO2_WORKSPACE/code/worktrees/codex-motion-diagnostics"
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/hardware" GO2_HARDWARE=0 GO2_SUPERVISED_LIVE=0
py="$GO2_WORKSPACE/envs/core/bin/python"
```

| Mode | Entry point and effect |
|---|---|
| Offline immutable replay | The [start guide](../START_HERE.md) generates a private HTML/JSON report. It reads the originals and starts neither sensors nor inference. |
| Interactive read-only advisory | `"$py" scripts/run_vln_interactive.py --review-run "$GO2_WORKSPACE/runs/five_move/trial-THKUcr/session" --port 0` opens a loopback viewer. Its legacy annotation workflow can append review events, refresh a contact sheet and save corrections. Use the immutable replay to preserve source bytes. The advisory application has no command client. |
| Experimental continuous controller | `"$py" scripts/run_vln_continuous.py simulate --scenario visible_after_search` exercises a local simulated camera and motion proxy. Separate opt-in model inference is documented in the [continuous guide](CONTINUOUS_NAVIGATION.md). |
| Single supervised action, **motion-capable** | From `$GO2_WORKSPACE/code/navigation`, `GO2_SUPERVISED_LIVE=1 bash scripts/launch_effective_trial.sh --live-exclusive sensor_only`. One review and fresh authorization can issue one guarded primitive. [One-action operator guide](SUPERVISED_NAVIGATION.md) |
| Bounded five-move episode, **motion-capable** | From the same active checkout, `GO2_SUPERVISED_LIVE=1 bash scripts/launch_five_move_episode.sh --live-exclusive --move-budget 5 --episode-seconds 60 --area-radius-m 3 --alignment-deg 15` matches the archived settings. This instance used one grant and the physical executor. Use the [operator procedure](FIVE_MOVE_NAVIGATION.md) for any separately scheduled trial. |

The live invocations above were checked against the launcher source and the saved manifest. This review ran only offline commands. The archived arguments establish the five-move application and launch profile. Shell command history was not preserved with the trial.
