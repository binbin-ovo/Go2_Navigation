# Running a supervised navigation session yourself

From power-on to a finished episode, using the code on `hardware/physical-evidence-integration`. Run every command in your own macOS **Terminal app**. A shell without macOS local-network permission receives no camera or LiDAR data (the Claude desktop Terminal panel is one). The first run may ask for that permission; allow it.

```bash
export GO2_WORKSPACE=$HOME/Robotics/Go2
cd "$GO2_WORKSPACE/code/navigation"
git status --short   # must print nothing; supervised startup refuses an edited checkout
```

## 1. Calibrate camera timing (after every robot power-on)

The camera timing bound is valid only for the robot boot on which it was measured. On a new boot every episode HOLDs until it is re-measured.

1. Stand the Go2 still. Put the Mac screen 20–40 cm in front of its front camera, filling the view, brightness at maximum.
2. Measure (read-only, about 40 s; the screen flashes black and white):

   ```bash
   PYTHONPATH=src:$GO2_WORKSPACE/sdk/unitree_sdk2_python "$GO2_WORKSPACE/envs/hw/bin/python" hardware/capture_latency.py --iface en9
   ```

   The last line is the evidence path, for example `.../evidence/hw_readiness/capture_latency_20261123_115758.json`.
3. Pin it (file-only). If it says the boot is already pinned, skip to step 2 below.

   ```bash
   "$GO2_WORKSPACE/envs/core/bin/python" hardware/update_camera_bound.py <evidence path from step 2>
   ```

4. Check and commit:

   ```bash
   GO2_HARDWARE=0 PYTHONPATH=src:scripts:hardware "$GO2_WORKSPACE/envs/core/bin/python" -m pytest -q -p no:cacheprovider tests/test_motion_diagnostics.py tests/test_hardware_guards.py
   ```

   ```bash
   git add hardware/supervised_guards.py && git commit -m "Pin the camera capture bound for this robot boot"
   ```

## 2. Prepare (file-only)

```bash
"$GO2_WORKSPACE/envs/core/bin/python" app.py supervised --prepare-only --move-budget 5 --episode-seconds 360 --area-radius-m 3 --alignment-deg 15
```

It must end with a `Trial directory:` line.

## 3. Run an episode

Place the Go2 about 2 m from the target, facing it, with nothing within about 0.6 m of the robot. Keep the remote in reach but untouched: any stick or button input stops the episode.

```bash
GO2_SUPERVISED_LIVE=1 "$GO2_WORKSPACE/envs/core/bin/python" app.py supervised --live-exclusive --move-budget 5 --episode-seconds 360 --area-radius-m 3 --alignment-deg 15
```

Wait for `Display: http://127.0.0.1:8766/` (the model takes about 45 s to load), then open that page:

1. Type the target (for example `dishwasher`) and your name, tick both boxes after inspecting the area, and press **Authorize episode**.
2. When the page says *identify_or_reject_current_mark*, drag a tight rectangle around the target on image B. If the box does not appear, click an empty part of the page and drag again. Then press **This is the requested target**, or **This is a different object**.
3. If it later says *reidentify_after_new_view*, do the same on the new image.
4. **STOP EPISODE** stops the robot at any time.
5. If the model stops at the target (*MODEL_STOPPED_ARRIVAL_UNVERIFIED*) and the Go2 is really there, press **I confirm arrival on this image** (`GOAL_CONFIRMED`).

Limits per episode: 5 moves, 360 s, 3 m from the start, forward steps 0.30–0.60 m at 1.0 m/s, turns 20–30° at 2.0 rad/s. The Go2's own obstacle avoidance stays on.

## 4. Finish

Type `quit` and Enter in the Terminal, or press Ctrl-C. The record is in `$GO2_WORKSPACE/runs/five_move/trial-*/` (private, never committed).

If the next launch fails with `another command source holds .../.command.lock`, the previous move ended without a verified stop. Look at the robot first. Only if it is standing still and no session is running (`pgrep -f run_vln_supervised` prints nothing), remove the lock:

```bash
rm "$GO2_WORKSPACE/code/control/data/outputs/go2_control/.command.lock"
```
