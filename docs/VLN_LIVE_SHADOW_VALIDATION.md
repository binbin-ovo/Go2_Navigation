# Live read-only VLN advisory — validation (2026-11-16)

Branch `feature/vln-live-shadow`, based on the verified advisory commit
`b47f07ddea005acfd537b2acf4c4aa739ed26c67` (`feature/vln-shadow-demo`), not on main.
Worktree `/Users/yushan/Desktop/Go2_Navigation-vln-live`. The control PR
(binbin-ovo/Go2_Navigation#1) was neither merged nor imported.

Flow: typed instruction → live Go2 front-camera RGB + causal history → pinned LightNav
→ local path / STOP / status display. **LIVE INPUT — ADVISORY ONLY — NO ROBOT COMMANDS.**

Evidence levels: **[HW]** = observed on the real Go2 in this session; **[MOCK]** =
verified with a fake bridge in tests; **[SRC]** = from source inspection only.

## What was added

| Piece | Environment | Role |
|---|---|---|
| `scripts/go2_camera_bridge.py` | hardware `.venv-go2` (Py 3.10) | Owned child. Runs the existing `Go2CameraReceiver` (multicast **read** socket + PyAV). Converts decoder BGR → RGB once, forwards every Nth frame (declared stride), bounded queue, JSON header + raw bytes on stdout. Exits on stdin EOF. |
| `src/go2_perception/vln/shadow_live.py` | core env (Py 3.12) | `frame_observation()` → existing `Go2VLNObservation`; `LiveCameraSource` owns the child, offers to the existing `ObservationInbox`, counts gaps/overflow, detects clock steps, closes cleanly. Imports no socket/SDK. |
| `scripts/run_vln_shadow.py --go2-camera` | core env | New source; mode `live_unvalidated`. Bridge starts **after** model readiness. Interruption watchdog, live observation budget, paused-history exit, per-request `latency.jsonl`. |
| `scripts/check_go2_camera_frame.py` | core env | One-off frame + metadata check, no model. |
| `shadow.py` / `shadow_display.py` | core env | Live banner, receipt time in each receipt, live camera status panel, cancel wording. |

No new observation schema, model, planner, calibration or registration. Model input is
still only the instruction plus causal calibrated RGB history.

### Observation semantics

- RGB uint8 720×1280×3, **RGB** order, raw/distorted (`rectified=False`), measured
  fisheye calibration from `configs/go2_front_camera_calibration.yaml` (hash in each
  observation's calibration block). Rectified **once**, by the existing `calibrated_view`,
  when a job is prepared. [HW][MOCK]
- Frame IDs `bridge_id:seq:rtpTS`, unique per bridge process. [HW]
- `stamp_host` = first-RTP-packet host receipt (wall), clock `host_receipt`. `stamp_robot`
  is **None**: no robot clock is subscribed, and host time is never relabelled.
  `available_s` = decode-completion wall time, same clock. [HW]
- Host monotonic: each frame carries a same-host wall−monotonic sample (sampling spread
  ~1e-7 s observed) so receipt is converted to the parent's monotonic domain. A hello
  bracket check refuses a child whose monotonic clock is not this host's. [HW]
- **Estimated exposure** (`Delivery.exposure_monotonic_s`) = receipt − 0.048 s (σ 0.016 s)
  from `configs/go2_vln_sensors.yaml`. That figure is one boot's test_c measurement
  relative to odometry stamps and excludes unknown odometry transport latency; the mapping
  string says so. `--camera-latency unknown` leaves exposure age unknown (not zero). [SRC][MOCK]
- Clock epoch `host_receipt_wall:bridge=<id>:g<n>`. A new bridge process, or a wall−monotonic
  step > 20 ms, starts a new epoch, and the session resets history. [MOCK]
- Pose, LiDAR, `T_base_camera`: **None**. Recorded as unavailable, never identity. [HW]
- No robot-time minus host-wall arithmetic anywhere.

### Bounded buffering and loss

Inbox capacity 16 (≈2.2 s at the forwarded rate), bridge queue 2. Overflow, bridge drops
and seq gaps are counted and logged as events, never silent. History is not padded or
re-fed: the existing session rule pauses inference on a stamp gap > 0.375 s; a camera
silence > 1 s is reported (`camera_interrupted`) and also pauses inference. A new
instruction starts a new history from the next real frame. Paused live history ends the
run after 15 s (`--paused-exit-s`) so frames are not stored indefinitely.

## Hardware session (read-only)

Operator confirmations (relayed from the control session in the user's words): "The Go2
is stationary and no movement process running". The control session reported movement
testing finished and API ownership released (code 0). `pgrep` found no control process
before access. The robot was not moved or commanded; the camera view was a low,
stationary viewpoint down a hallway toward the apartment door. **Posture
(lying/standing) was not confirmed by the operator.**

Access used: ICMP ping, then only the RTP multicast read socket. No DDS participant,
no SDK import (bridge reports `robot_client_modules_loaded: []` in every stats/bye
message), no `UseRemoteCommandFromApi`, no move/stop/zero-velocity/posture/avoidance
request. Obstacle avoidance was not touched. **Robot commands sent: 0.** [HW]

### 1. Frame check — `frame_check_01` [HW]

20 frames received in 5.5 s; validation issues `[]`; unique IDs; RGB verified visually
(the natural-colour PNG); RTP lost/reordered 0; decoder errors 0; 27 frames discarded
while waiting for the first keyframe (expected). Calibrated view: `opencv_fisheye_balance0`.
Receipt → decoded median 4.0 ms; bridge → parent read median 3.0 ms.

### 2. Rates [HW]

| Stream | Rate |
|---|---|
| Decoded camera frames | 14.3 Hz (3524 decoded in run 1) |
| Forwarded to the app (stride 2) | 7.1 Hz (median interval 0.140 s, max 0.148 s) |
| Model requests (one in flight) | one every 2.8–9 s; native generation 2.3–6.9 s |
| Model load (once per run) | 44.7 s, 44.6 s (RSS 11.8 GB, MPS ~9.3 GB) |

Stride 2 is declared: per-frame storage of the existing NPZ + latest PNG costs about
60 ms on this Mac, so at 14.3 Hz the event loop could not keep up. 7.1 Hz still gives
at least one actual frame per 0.25 s native grid slot.

### 3. Live sessions

Every run directory is under `/Users/yushan/Desktop/Go2_Navigation/data/outputs/vln_live/`
(git-ignored, private). Logs are in `audit/`.

| Run | Requests | Result | Notes |
|---|---|---|---|
| `live_session_01` | 2 | 1 STALE (3.11 s), 1 rejected | **Failure, kept.** The original 2-slot inbox overflowed while the next 17-frame job was being prepared (main loop blocked ~0.55 s). Two frames were lost, which made a 0.43 s stamp gap; inference paused correctly. The run then stored frames until the 300 s host budget (1.6 GB). Fixed: inbox 16, live observation budget, paused-history exit. |
| `live_session_02` | 7 | 1 **CURRENT** at receipt (2.959 s), 3 STALE, 3 rejected/superseded | Session-behaviour run (see §5). Display delay 0.5–2.1 s was an artefact of loop order (next job prepared before render). Fixed: render first. |
| `live_session_03` | 3 | 3 STALE (3.25, 5.02, 8.54 s) | Confirming run on the final code; display 3–13 ms after receipt; ended by request budget; both children exited. |

Exact verified launch command (run 3; use a new `--output` each time):

```bash
cd /Users/yushan/Desktop/Go2_Navigation-vln-live
env -u LIGHTNAV_ASSETS -u SMOLVLM_MODEL_CACHE GO2_HARDWARE=0 MUJOCO_RENDER_TEST=0 \
  HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=src \
  .venv/bin/python -u scripts/run_vln_shadow.py --go2-camera \
  --hardware-python /Users/yushan/Desktop/Go2_Navigation/.venv-go2/bin/python \
  --instruction 'Approach the doorway and stop before entering.' \
  --backend lightnav \
  --model-python /Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6c/model-env/bin/python \
  --assets /Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6c/lightnav \
  --runtime-config configs/pretrained_vln.yaml --device mps \
  --max-requests 3 --max-observations 600 --history-seconds 20 --max-host-seconds 180 \
  --output /Users/yushan/Desktop/Go2_Navigation/data/outputs/vln_live/<NEW_RUN>
```

`.venv` is an ignored symlink to `phase6c/core-env`. Frame check:
`PYTHONPATH=src .venv/bin/python -u scripts/check_go2_camera_frame.py --hardware-python
/Users/yushan/Desktop/Go2_Navigation/.venv-go2/bin/python --output <NEW_DIR> --frames 20`.
Open `<run>/index.html` in a browser; terminal commands are `instruction TEXT`, `cancel`,
`status`, `quit`.

## 4. Latency: where the time goes [HW]

All intervals are same-host monotonic. "Exposure" is the estimate (receipt − 48 ms).
Unchanged 3 s advisory limit; the replay baseline was 3.545 s.

Run 3 (final code):

| Stage | Req 1 | Req 2 | Req 3 |
|---|---:|---:|---:|
| estimated exposure → sensor receipt | 0.048 (assumed) | 0.048 | 0.048 |
| receipt → decoded (bridge) | 0.002 | 0.010 | 0.002 |
| decoded → parent read (pipe) | 0.003 | 0.003 | 0.003 |
| parent read → session receive | 0.010 | 0.002 | 0.005 |
| receive → submit (scheduling) | 0.059 | 0.112 | 0.068 |
| input preparation + dispatch | 0.175 | 0.580 | 1.210 |
| child inference (native generate) | 2.952 (2.598) | 4.239 (3.441) | 7.194 (5.552) |
| inference end → parent receipt | 0.002 | 0.023 | 0.011 |
| receipt → display | 0.003 | 0.007 | 0.013 |
| **est. exposure → display** | **3.255** | **5.024** | **8.554** |
| frames in job | 5 | 17 | 37 |
| source lag at receipt | 3.081 | 4.900 | 8.406 |
| result | STALE | STALE | STALE |

Summary across runs 2–3 (10 answered LightNav requests on final or near-final code):
1 accepted as current at receipt (run 2 request 1: 2.959 s, which aged past 3 s within
about 0.05 s and is then shown as STALE), 6 stale, 3 superseded/invalid by design.
**In steady state every proposal is stale.**

Where the time goes:
- Sensor path (exposure estimate + receipt + decode + pipe + inbox): about **65 ms**.
  Of that, 48 ms is the assumed camera latency; the measured part is ~10–20 ms.
- **Native generation dominates and grows with instruction history**: 2.3–2.6 s with
  5 input frames, 3.4 s at 17, 5.6 s at 37, 6.9 s at 63. The existing policy rebuilds
  the complete causal history from instruction start on every request, so model input
  grows with elapsed time.
- Input preparation also grows linearly (~30 ms per frame: rectify + PNG): 0.18 s → 2.1 s.
- Scheduling, receipt and display are below 0.12 s each after the render-order fix.

Clock-conversion uncertainty: wall→monotonic sample spread ≤ 3e-7 s; the dominant
uncertainty is the assumed exposure latency (σ 16 ms, single-boot, excludes odometry
transport). Dropped frames in runs 2–3: 0 (overflow 0, bridge drops 0, RTP loss 0).
History gaps: run 1 only (one 0.43 s gap from overflow). No optimization or replacement
was started; that is a separate decision.

## 5. Session behaviour [HW unless noted]

| Behaviour | Evidence |
|---|---|
| Changed instruction invalidates pending | Run 2: request 4 (r1) was in flight when "Turn toward the refrigerator." arrived; its result was rejected `superseded_instruction`; history restarted from the next real frame. |
| Cancel prevents late results becoming current | Run 2: `cancel` sent while request 5 (r2) was generating (`physical_robot_stop_sent: false`); its late result arrived 6.9 s later and was rejected, never current. [MOCK]: `test_cancel_during_generation_does_not_accept_or_send_stop`. |
| Clock/epoch change resets history | [MOCK] `test_clock_epoch_change_resets_history`, `test_bridge_overflow_gaps_epoch_and_clean_shutdown` (500 ms wall step → `g1`). Not provoked on hardware. |
| Camera interruption reported, not filled | Run 1 (overflow gap → paused). [MOCK] 1.6 s silence → `camera_interrupted`, history paused, no re-fed frames. A physical unplug was not tested. |
| History limit | Run 2: `history_limit_reached_start_new_instruction` at 20 s; request 7 rejected `invalid_history`. |
| UI separates current vs model input | `advisory.png`/`index.html`: latest raw fisheye frame vs exact rectified 448×256 model inputs with their own timestamps; path drawn in its own local chart, never on the camera image. |
| Shutdown | All 3 runs: `camera_bridge_closed` then `session_closed`; bridge `receiver_alive: false`; `pgrep` found no bridge/worker left. Quit, request budget and host budget each exercised. |
| Zero robot requests | No SDK/DDS module loaded in the bridge; static AST test of live files and the receiver; `robot_commands_sent: 0` in every summary. |

## 6. Qualitative target check (human review pending)

Native target pointing on the actual model input (`review/targets_r1_r4_r5.png`):

| Request | Instruction | Native | My reading — **awaiting operator review** |
|---|---|---|---|
| run 2 #1, #2, #6; run 3 #1, #2 | Approach the doorway… | `opos_501`, path 1.45 m fwd / 0.39 m left, yaw +0.26, STOP false | Target lies on the hallway door; leftward path matches the door being left of centre. No STOP while the door is far: plausible. |
| run 2 #4, #7; run 3 #3 | Approach the doorway… | `opos_549/550/502`, straight 1.51 m, STOP false | Target on/near the door. |
| run 2 #5 | Turn toward the refrigerator. | `opos_415`, yaw −0.60 (right), 0.95 m | Target on the refrigerator door, right turn matches its position. |

These are stationary-robot observations of one scene. They are not a benchmark, are not
registered to the robot base, and claim no navigation success or goal completion. Native
STOP and the model completion claim were false in every request.

## Tests

- New `tests/test_vln_live_shadow.py`: 15 mocked tests (conversion, unknown exposure,
  bad frames, single rectification + live age, epoch reset, gap pause, overflow/seq-gap/
  clock-step/clean shutdown with a fake bridge child, hung-bridge kill, bridge stride/RGB,
  live entry point with interruption + latency + shutdown, paused-exit and observation
  budget, no-control-route AST check). Fixture `tests/fixtures/vln_live/fake_bridge.py`
  speaks the bridge protocol with no socket.
- Existing `tests/test_vln_shadow.py`: 57 passed, unchanged.
- Full core regression on the final code: **663 passed, 0 failed, 9 skipped** in 56.3 s
  (same 9 environment/physical skips as the advisory baseline). Ruff clean on changed files.
  Logs: `data/outputs/vln_live/audit/full_core_regression_final.{log,xml}`.

## Limitations

- **Freshness is not achieved**: with the unchanged 3 s limit, only a first request with 5
  frames can land just under the limit; later requests in an instruction are stale because
  native generation and preparation grow with history. Not optimized here.
- Exposure time is an estimate from one earlier boot; not re-measured on this boot.
- RGB-only: pose and LiDAR were not subscribed; `T_base_camera` unresolved; the path is
  unregistered.
- Stride-2 decimation; a 14.3 Hz input would need cheaper per-frame storage.
- Clock-step and camera-unplug handling verified only with mocks.
- Single static scene, stationary robot, posture unconfirmed; target interpretation is
  qualitative and pending operator review.
- `live_session_01` holds 1.6 GB of private frames; delete it when no longer needed.
