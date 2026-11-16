# Cached preparation on live camera input — validation (2026-11-16)

Branch `feature/vln-cached-live-validation` (worktree
`/Users/yushan/Desktop/Go2_Navigation-vln-cached`), based exactly on Codex's verified
publication `d173082e887f66a81d81eadf52c332b7f5a3c655` (`feature/vln-live-latency`).
Main and the hardware-control PR were not merged or imported. **No software change:**
this branch adds only this report and the handoff update.

## Physical coordination and access

Before connecting, the operator confirmed in chat: Go2 stationary, physical movement
testing paused/finished, no software movement process active, and the earlier command
session's stop/release cleanup complete. No other agent was running model/benchmark
work on this Mac (only idle Codex app processes). A read-only ping to 192.168.123.161
preceded the run.

Access was the existing read-only multicast bridge only (`multicast_read_socket_only_no_dds_no_commands`).
Bridge `robot_client_modules_loaded: []`; `robot_commands_sent: 0`. No DDS, ownership,
move, stop, zero-velocity, posture or avoidance request. Manufacturer avoidance untouched.

## Exact command (one attempt, no retries)

The prepared command from [VLN_LIVE_LATENCY](VLN_LIVE_LATENCY.md), with only the
worktree and a fresh private output path changed:

```bash
cd /Users/yushan/Desktop/Go2_Navigation-vln-cached
env -u LIGHTNAV_ASSETS -u SMOLVLM_MODEL_CACHE GO2_HARDWARE=0 MUJOCO_RENDER_TEST=0 \
  HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=src \
  /Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6c/core-env/bin/python -u scripts/run_vln_shadow.py \
  --go2-camera --hardware-python /Users/yushan/Desktop/Go2_Navigation/.venv-go2/bin/python \
  --instruction 'Approach the doorway and stop before entering.' --backend lightnav \
  --model-python /Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6c/model-env/bin/python \
  --assets /Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6c/lightnav \
  --runtime-config configs/pretrained_vln.yaml --device mps --preparation-mode cached \
  --advisory-age-s 3 --max-requests 3 --max-observations 600 --history-seconds 20 \
  --max-host-seconds 180 \
  --output /Users/yushan/Desktop/Go2_Navigation/data/outputs/vln_live/cached_live_01
```

Same LightNav checkpoint/runtime, measured fisheye preprocessing, native image-history
policy, 3 s advisory threshold, stride 2, inbox 16. Model loaded once; only newly
received camera frames were used. Exit 0, termination `request_budget`, host 81.8 s.

## Per-request results (verified on hardware, live)

Ages are *estimated* exposure ages: host receipt − configured 48 ms (σ 16 ms, from the
2026-11-07 recording). Not hardware-synchronized; single boot; unvalidated on this boot.

| Req | Frames in job (native used) | Source frame / cutoff wall s | Cache hit/miss (this req) | Prep+dispatch s | Child inference s | Native generate s | Receipt→display s | Est. exposure→display s | Status |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 5 (5) | #9 / 1794847594.627 | 0 / 5 | 0.186 | 3.331 | 2.999 | 0.007 | 3.651 | STALE (rejected `stale_advisory_age`) |
| 2 | 19 (19) | #34 / 1794847598.130 | 5 / 14 | 0.520 | 4.296 | 3.963 | 0.068 | 5.033 | STALE |
| 3 | 39 (38) | #69 / 1794847603.030 | 5 / 34 | 1.160 | 7.155 | 6.483 | 0.090 | 8.540 | STALE |

History starts at frame #1 (1794847593.505) for all three; exact per-frame stamps are
in each `snapshot.json`. Sensor receipt → session receive ≈ 16 ms; uncertainty of the
exposure estimate ±16 ms (1σ). Request 3's 39-frame job feeds 38 frames: the pinned
SlowFast sampler's tiers are {0,1},{3,4},{5–36},{37,38}, skipping index 2 — same
behaviour as the baseline run (37 → 36), not a cache defect.

Preparation breakdown (req 3): hash/load 0.145, rectification 0.247, PNG encode/write
0.699, serialization 0.005 s. Cache: 53 entries / 147 MB at end, 0 evictions, 0 bypasses.

## Camera buffering and archive writing

- Bridge: 258 decoded, 129 forwarded (stride 2), 0 bridge-queue drops, 0 RTP loss/
  reorder, 0 decoder errors, 25 packets dropped until first keyframe, 0 socket rejoins.
- Parent: 129 read/offered, 0 inbox overflow, 0 seq gaps, 0 rejected, 0 clock-epoch
  changes, 0 events lost. No `camera_interrupted`, no history gap/pause (`history_error: null`).
- Tensor evidence: 3/3 `tensor_evidence.json` `complete` (4.1/13.8/26.2 MB, write
  0.14/0.47/0.93 s, off the response path). Shutdown: submitted 3, completed 3,
  outstanding 0, max outstanding 1, peak 26.2 MB / 256 MiB cap, error null.
  The shutdown was normal (stdin EOF drain), not forced.
- At close the session logged `advisory_cancelled` with `physical_robot_stop_sent: false`
  (session teardown; nothing in flight) and `session_closed`. No leftover processes.

## Display checks

`index.html` shows the live banner, "Latest camera view — may be newer" (frame #129,
its own host-receipt stamp) separately from "Image actually used by this proposal"
(cutoff 1794847603.030 s with exact input stamps). The displayed proposal is labelled
STALE with `rejection: stale_advisory_age`; it is never shown as newly inferred. Path is
not registered to the camera image (no `T_base_camera`).

## Interpretation

- **The optimized live path functions correctly**: cached mode ran end-to-end on real
  arrivals, with hash-verified reuse, complete bounded background archival, no
  buffering loss and correct stale labelling.
- **Freshness is unresolved; it did not improve.** All three valid proposals are stale
  under the unchanged 3 s limit. Request 1 (3.65 s) is just over the limit; earlier
  baseline run 2 had one request current at receipt (2.96 s), so a 5-frame request sits
  at the threshold either way.
- **Live cache reuse is low**: only 5 frames per request were hits because latest-anchored
  4 Hz selection re-picks frames as time advances (as Codex's audit predicted).
  Preparation stayed 0.19 / 0.52 / 1.16 s vs baseline run 3's 0.175 / 0.58 / 1.21 s at
  5 / 17 / 37 frames. Jobs are unmatched (different images and 19/39 vs 17/37 frames),
  so **no controlled live speedup is claimed**. Native generation (3.0 → 6.5 s, growing
  with history) dominates; the offline 2.94→2.78 / 4.36→4.01 / 8.31→7.39 s submit-to-
  display improvements came mainly from moving archival off the response path and are
  not live exposure-to-display measurements.
- One sample per history length; no percentiles or variability claims.

## Operator-reviewed observations (qualitative, one scene)

Model-input frames with native targets: `opos_501`, `opos_502`, `opos_504`, all on the
doorway at the end of the hallway. Native local paths: r1/r2 end 1.45 m forward,
0.39 m left, yaw +0.26 rad; r3 1.51 m straight ahead. Native STOP false in every request.
The operator judged the targets correct. This is not navigation success and authorizes
no movement.

## Tests (after the live session ended)

`tests/test_vln_live_shadow.py tests/test_vln_shadow.py tests/test_vln_latency.py`:
**92 passed, 0 failed** in 14.6 s (core env, hardware opt-ins off). No full-suite rerun
(no code changed; Codex's 683-pass regression stands for this commit).

## Local private artifacts (git-ignored, not published)

`/Users/yushan/Desktop/Go2_Navigation/data/outputs/vln_live/`:
`cached_live_01/` (index.html, latency.jsonl, events.jsonl, summary.json, requests/
with preparation_profile.json, native outputs, tensor_evidence.json, prepared_cache/,
observations/), `cached_live_01.log`, `cached_live_01_targeted_tests.log`,
`review/cached_targets_r1_r2_r3.png`.

## Limitations

Estimated (not measured) exposure latency; no real `T_base_camera`; RGB-only, pose/LiDAR
absent; one scene, one session, three requests. Native waypoints are connected to no
Go2 command and no physical execution path was added.
