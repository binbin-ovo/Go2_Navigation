# Interactive four-panel VLN review and data collection

Branch `feature/vln-interactive-review` (worktree
`/Users/yushan/Desktop/Go2_Navigation-vln-interactive`), based exactly on
`1f93d7eff57906ce26c4b3ab4109effb4ca71edd` (`feature/vln-cached-live-validation`).
Main and the high-level motion-control PR are neither merged nor imported.

**ADVISORY ONLY — NO ROBOT COMMANDS.** The operator types a target in the terminal; a
loopback-only browser page shows the live camera, the model's native target point,
all native waypoints and headings, and a LiDAR top-down view. The operator marks what
they see, ends the trial, answers review questions, and can reopen any saved trial
with its comments later.

**Status (2026-11-16):** built and verified with synthetic fixtures, a public synthetic
recording, one private recorded episode and the mock worker only. **Nothing has run on
the Go2 or with LightNav yet.** That needs the operator's approval and a stationary,
posture-confirmed robot (see [Next: hardware session](#next-hardware-session-not-run)).
No human review of real trials exists yet; the synthetic review records created by
automated tests are test fixtures, not human labels.

![Synthetic contact sheet, mock worker, no model](assets/vln_interactive_synthetic_contact_sheet.png)

*Synthetic imagery and a canned mock that alternates a fixed point and OPOS-0. It shows
the layout only; it says nothing about model accuracy.*

## What is reused and what is new

| Piece | Reused unchanged | New |
|---|---|---|
| Camera | `scripts/go2_camera_bridge.py` + `LiveCameraSource` (read-only multicast, measured fisheye calibration, stride 2, inbox 16) | `TapInbox`: the same bounded inbox plus a 24-frame display ring (panel A updates on the reader thread, even while a request is being prepared) |
| Observation | `Go2VLNObservation`, `ObservationInbox`, `Delivery`; no schema change | — |
| Model | `ShadowSession`, `ProcessWorker`, `vln_shadow_worker.py`, `lightnav_worker.run_job`; **full-history** reconstruction per request, cached preparation, 3 s advisory limit, stale/superseded/cancel rules | `--tensor-evidence skip` (cached mode only; default in this tool). The status file records `not_recorded_tensor_evidence_opt_in_disabled`. Defaults of `run_vln_shadow.py` are unchanged |
| Native decoding | `shadow_display.target_fraction`, `validate_proposal` | — |
| LiDAR/odom | `Go2DdsReceiver`, `PoseBuffer`, `SlidingWindowAccumulator`, 0.5 s / 1.0 s windows from `configs/go2_vln_sensors.yaml` | `scripts/go2_lidar_bridge.py` (hardware env), `vln/interactive_lidar.py` (core env), `scripts/synthetic_lidar_bridge.py` (synthetic or recorded replay, same protocol) |
| App | — | `vln/interactive_app.py`, `interactive_trials.py`, `interactive_server.py`, `interactive_page.py`, `scripts/run_vln_interactive.py` |

The experimental 8-image bounded-context policy is **not** used (`run.json`:
`temporal_policy` … "bounded-context policy NOT enabled"). Weights, prompts, temporal policy
and the 3 s advisory limit are unchanged; the limit is hard-coded, not exposed as an option.

## Launch

Use the core environment (the ignored `.venv` link points at `phase6c/core-env`). Each run
needs a **new** output directory, and it must be Git-ignored. The default is
`data/outputs/vln_interactive/run_<time>`.

Synthetic, no hardware, no model (development and demos):

```bash
cd /Users/yushan/Desktop/Go2_Navigation-vln-interactive
PYTHONPATH=src .venv/bin/python -u scripts/run_vln_interactive.py \
  --camera synthetic --lidar synthetic --backend mock --mock-pointing alternate \
  --reviewer YOUR_INITIALS
```

Recorded episode (private images, mock model):
`--camera episode --episode <…>/model_input/episode.json --backend mock`.
Recorded LiDAR: `--lidar recording --lidar-recording <go2-recording-v1 dir>` (replay
LiDAR may not be mixed with the live camera, and vice versa).

Prepared live command (**not run yet**; it requires the operator's approval, a stationary
robot with posture confirmed, and no other heavy job on the Mac):

```bash
cd /Users/yushan/Desktop/Go2_Navigation-vln-interactive
env -u LIGHTNAV_ASSETS -u SMOLVLM_MODEL_CACHE GO2_HARDWARE=0 MUJOCO_RENDER_TEST=0 \
  HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=src \
  .venv/bin/python -u scripts/run_vln_interactive.py \
  --camera go2 --lidar go2 --iface en9 \
  --hardware-python /Users/yushan/Desktop/Go2_Navigation/.venv-go2/bin/python \
  --backend lightnav \
  --model-python /Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6c/model-env/bin/python \
  --assets /Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6c/lightnav \
  --runtime-config configs/pretrained_vln.yaml --device mps --preparation-mode cached \
  --reviewer YOUR_INITIALS \
  --output data/outputs/vln_interactive/live_01
```

Open the printed `http://127.0.0.1:8765/` (loopback only; `--port 0` picks a free port).

### Terminal commands (`help`)

```text
target TEXT        new trial; expands to "Go to the TEXT and stop." (saved before inference)
instruction TEXT   new trial with TEXT verbatim as the instruction
mark [NOTE]        bind a mark to the prediction + camera frame the browser shows now
note TEXT          verbatim note bound to the current trial and displayed IDs
end                end the trial (advisory cancel only; no robot command) and review it
review [ID|close]  list trials, open one (contact sheet + new append-only review), or close
status             print a compact status line
save               flush records and regenerate the contact sheet
quit               end everything (no robot command is ever sent)
help               this text
During a review dialog every line is an answer; `!cancel` discards it, `!quit` quits.
```

Template `target_go_to_and_stop_v1` = `Go to the {target} and stop.` The template ID, the
operator's words and the expanded instruction are written to `trial.json` **before** the
instruction reaches the session. The operator's words are a request, never a verified
label (`operator_text_is_verified_label: false`).

Starting a new `target`/`instruction` while a trial is active ends that trial
(`superseded_by_new_trial`); it can be reviewed later with `review ID`. `end` and `quit`
call only the session's advisory `cancel` (`physical_robot_stop_sent: false`).

## Window

The header shows the mode in large type: **LIVE**, **REPLAY**, **SYNTHETIC** or **REVIEW**.
The status strip shows the trial, model state, last inference time, observation age,
recording, memory (per process, never summed with MPS), queue fill and disk.

- **A · Live camera:** the latest RGB frame with its frame ID, host-receipt time (clock and
  semantics), robot-clock stamp (NONE for live RGB, never relabelled), clock epoch, receipt
  fps, RAW/RECTIFIED, receipt gaps > 0.375 s, INTERRUPTED after 1 s of silence, and estimated
  exposure (UNKNOWN unless the configured latency model is used).
- **B · Model target:** the exact model-space input image of the displayed prediction, with
  its native point as a `+`. The image and the point come from **one** prediction record and
  are drawn together only after that image has loaded, so an old point is never drawn on a
  newer frame. OPOS-0 is shown as "model says target NOT VISIBLE" (a model claim, not proof
  of absence). The panel also shows the prediction ID, source frame ID/time/clock, age and
  CURRENT/STALE/SUPERSEDED/REJECTED (re-evaluated on every refresh). No boxes, no
  confidences.
- **C · Native waypoints:** all 10 native SE(2) waypoints and headings of the same
  prediction, with a table, in the model-local frame (metres/radians). Native STOP, explicit
  STOP code, model completion claim, "verified completion UNKNOWN" and
  `NOT EXECUTED — advisory only` are shown separately. No time base or speeds, and no
  registration to LiDAR, base or camera.
- **D · LiDAR top-down:** `base_link`, with views for the latest scan, 0.5 s and 1.0 s.
  Accumulated views appear only with odometry compensation; otherwise "UNAVAILABLE" with
  the reason. The panel shows host age, robot and host receipt intervals, azimuth/grid
  coverage (cells without returns stay UNKNOWN), pose only while odometry is fresh (≤ 0.25 s),
  robot clock epoch, sync status, gaps and refusals. The 0.3 m blind radius is shaded.

In REVIEW the contact sheet sits above the live panels. Prediction tiles carry their own
point. Clicking a tile stores an optional correction, after a browser confirmation.

## Process boundaries and clocks

| Process | Environment | Talks to | Never |
|---|---|---|---|
| `run_vln_interactive.py` | core (3.12) | owns everything below via pipes; loopback HTTP | imports no SDK/DDS/Torch |
| `go2_camera_bridge.py` | hardware (3.10) | multicast **read** socket | DDS, SDK, outgoing datagrams |
| `go2_lidar_bridge.py` | hardware (3.10) | DDS **subscribers** (`cloud_base` + receiver state topics; only odom forwarded) | publishers, clients, `cloud_deskewed` |
| `vln_shadow_worker.py` | model env | stdin/stdout JSONL | network, sensors, robot |

- **Camera:** host-receipt wall clock (`stamp_host`) and decode completion (`available_s`).
  A per-frame wall−monotonic sample maps them to this host's monotonic clock. A new bridge
  or a > 20 ms wall step starts a new epoch, which resets history.
- **LiDAR/odom:** robot-clock stamps per *robot epoch*. A backwards stamp jump > 1 s,
  a > 2 s jump in host−robot, or a new bridge process starts a new epoch. That resets
  the pose buffer, windows and offset. host−robot is the lower envelope of odom receipt
  minus stamp, **estimated** only after 50 samples in the epoch. It is display only and
  never reused across epochs or boots. A host wall-clock step discards it. With no
  valid estimate the page says "streams shown separately". There is **no RGB–LiDAR
  fusion** in any case (no measured `T_base_camera`).
- **Model:** host monotonic submit/start/end; ages as in the existing session.
- **Records:** wall UTC strings for people, host monotonic seconds for ordering. Recorded
  and synthetic sources keep their original stamps; receipt is this process's own receipt.

Rendering does not depend on inference. The camera tap and the LiDAR reader run on
their own threads; the server reads snapshots; the main loop never waits for the model
(one request in flight, one coalesced pending flag, 0.25 s minimum interval, at most
`--max-requests-per-trial`, default 6). The terminal reader is its own thread, with a
bounded 64-line queue.

## Recording (private, Git-ignored)

```text
<run>/run.json              model identity file, runtime-config hash, calibration, clocks, limits,
                            code hashes, git commit, robot_commands_sent 0
<run>/events.jsonl          app events (append-only)
<run>/session/              existing ShadowSession evidence: observations/*.vlnobs.npz (lossless),
                            requests/<seq>/ (job, snapshot, native outputs, receipt), worker/model.json
<run>/blobs/sha256/ab/<sha>.png   lossless images stored once, keyed by pixel SHA-256
<run>/lidar/chunk_*.npz + index.jsonl   ≤1 s chunks: points, per-scan stamps/receipts/seq, odom;
                            index rows carry robot/host source intervals, epoch, trial, sha256,
                            or dropped_not_recorded with the interval
<run>/trials/index.jsonl
<run>/trials/<id>/trial.json         instruction, template, temporal policy (written pre-inference)
<run>/trials/<id>/predictions.jsonl  native outputs as received (model output, never labels)
<run>/trials/<id>/marks.jsonl, notes.jsonl     verbatim operator text bound to exact IDs
<run>/trials/<id>/human_labels.jsonl          confirmed human reviews (separate file)
<run>/trials/<id>/corrections.jsonl           optional click-to-correct targets (separate file)
<run>/trials/<id>/trial_end.json, contact_sheet.png
<run>/summary.json          termination, counts, final resources + history, robot_commands_sent 0
```

- **Images once, by hash:** marked frames that are already in the session's lossless
  observation archive are referenced (path + file hash + pixel hash). Only other frames
  are written as PNG blobs. On every read the pixel hash is checked.
- **Marks:** the browser reports what it is displaying (prediction ID, camera frame ID)
  to a token-protected loopback endpoint. `mark` binds to that view when the report
  is ≤ 2 s old (`browser_reported_view`); otherwise it binds to the server's latest
  view and records `server_latest_browser_view_not_reported_recently`.
- **Review:** after `end` the terminal asks for a verbatim comment, target
  present/absent/uncertain, marking correct/wrong/ambiguous/n.a., path
  plausible/inappropriate/uncertain, stopping appropriate/inappropriate/uncertain/n.a.,
  and the reviewer. It then prints the structured record and saves it only on `yes`.
  Records are append-only; reopening adds a new review and never edits an old one.
- **Corrections:** the clicked fraction and pixel on that prediction's own input image,
  the image hash and size, and the coordinate convention (origin top-left, x right,
  y down).
- **Feedback never reaches the model** (`model_feedback: false`). The worker receives only
  instruction + causal images.
- **Tensors** are recorded only with `--tensor-evidence record`.
- **Late results** are written to the trial whose instruction revision produced them,
  labelled SUPERSEDED with the reason. The contact sheet is regenerated if that trial is
  under review.

## Bounds, rates, memory and disk

| Bound | Default |
|---|---|
| Camera inbox / display ring / bridge queue | 16 / 24 frames / 2 (existing) |
| LiDAR bridge queues | 8 scans / 256 odom (drops counted, visible as seq gaps) |
| LiDAR recorder | 1 writer, queue 4 chunks; full queue → explicit gap row |
| History | 20 s, 1024 frames per trial (then inference pauses and frame archiving stops) |
| Trial | 45 s, 6 requests, 20 trials per run, 3600 s host time |
| Resources | run quota 6 GiB, free disk ≥ 20 GiB, available memory ≥ 2 GiB, app RSS ≤ 4 GiB |

When a resource limit trips, the active trial ends (`resource_limit`), new trials are
refused and LiDAR recording stops. Nothing is deleted, and the display and review stay
available.

Rates: earlier hardware validations measured 14.3 Hz decoded / 7.1 Hz forwarded camera
frames, ~15.6 Hz `cloud_base` (1,166–1,600 points) and ~149 Hz `robot_odom`. **This tool
has not yet measured them itself.** Synthetic sources: 8 Hz camera, 15 Hz scans, 50 Hz odom.

Measured memory (synthetic camera and LiDAR, mock worker, six 8 s trials, one process,
RSS from psutil): app parent **98.0 MiB** at start, 111.2 after trial 1, then 111.7 /
112.0 / 112.2 / 112.5 / 112.5 MiB. Growth flattens after the first trial (about
0.3 MiB per trial). Synthetic LiDAR bridge about 40 MiB. LightNav was not loaded; earlier
runs measured 11.2 GB worker RSS and 8.9 GB MPS allocated at load. Those two figures
overlap and are never added. Disk for the six synthetic trials was 78 MiB (448×256
images). Live 720p archives were about 0.94 MB per stored frame at 7.1 Hz in earlier
runs, so expect roughly 7 MB/s while a trial is active (an estimate, not a measurement
from this tool). The disk counter counts prepared-cache hard links once.

## Tests

`tests/test_vln_interactive.py`, 22 tests, no hardware, model or DDS:

- Template saved before inference.
- Target change keeps the late result with its original trial (SUPERSEDED).
- `end` cancels the advisory only; a late result is labelled `cancelled_or_closed`.
- A label ages from CURRENT to STALE at the unchanged 3 s limit, and the served image
  hash equals the recorded one.
- OPOS-0 is explicit, with no execution, time base or completion claims.
- Request, trial-time and trial-count limits; resource limit.
- Camera interruption and history limit stop archiving.
- Mark binding (browser view vs fallback), session-archive image references and hash
  checks.
- Blob dedupe and overwrite refusal.
- Review dialog: no save before `yes`, comment kept verbatim.
- Click mapping (fractions → pixels), separate storage, REVIEW only.
- LiDAR: reboot epoch, host step, bridge restart and offset reset; refusal without
  pose; latest-only fallback; ageing after the stream stops (recorded replay keeps all
  12/138 messages and original stamps).
- Bounded recorder with explicit gap rows.
- Synthetic bridge with and without odometry.
- Loopback server: token, Host check, path safety.
- Tensor opt-out.
- Full CLI: trial → mark → end → review → reopen with `--review-run` (append-only).
- AST check: no control route in any new file; the LiDAR bridge subscribes to
  `cloud_base` only.

Results: `22 passed`. Full core suite on this branch: **702 passed, 0 failed, 11 skipped**
in 51.9 s. The skips are environment-only (pinned LightNav source/RVQ, SmolVLM cache,
simulator, `GO2_HARDWARE=1`), listed in
`data/outputs/vln_interactive/audit/full_core_regression.log` (private). Ruff is clean on
the new files.

## Reopen and replay

```bash
PYTHONPATH=src .venv/bin/python -u scripts/run_vln_interactive.py --review-run <run directory>
```

This starts no sensors and no model. `review` lists trials; `review ID` shows the contact
sheet, predictions, marks, notes, corrections and earlier reviews, and offers a new
append-only review. Exact model-input replay uses the saved
`session/requests/<seq>/job.json`, a hash-bound job for the unchanged
`lightnav_worker.py`, the same way as in [VLN_LIVE_LATENCY](VLN_LIVE_LATENCY.md).

## Next: hardware session (not run)

1. The operator approves, confirms the Go2 is stationary with posture stated, and confirms
   no movement process is running. Check that no other agent is running heavy model or
   physical tests on this Mac.
2. Start the prepared live command. Model load is about 45 s. Panels A and D update
   during the load.
3. Run short trials with immediate review: a **visible** target, an **absent** target and an
   **ambiguous** target. Mark at least once per trial, `end`, and answer the questions.
4. Record measured camera/LiDAR rates, app/worker RSS and worker MPS (separately) across
   trials, and disk per trial second, in this document.

## Limits

- Live freshness is unresolved. Under the full-history policy, generation grows with
  history (earlier: 3.0–6.5 s), so most live proposals will show **STALE**; this tool
  does not change that.
- No `T_base_camera`, so no RGB–LiDAR fusion and no path registration. Camera exposure
  time is an estimate from one earlier boot.
- LiDAR coverage is returns-only. Cells and sectors without returns are UNKNOWN, not
  free, and the floor height assumes a standing robot.
- The browser view report proves only what the page loaded, not that the operator was
  looking at it.
- The review questions are per trial; marks and corrections carry per-prediction detail.
- Mock modes are UI checks only and are labelled MOCK everywhere.
