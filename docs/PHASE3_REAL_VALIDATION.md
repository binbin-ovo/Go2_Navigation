# Phase 3 real-world validation (started 2026-11-07, macOS)
Real Go2 data captured on the Mac ([MAC_CONNECTION](MAC_CONNECTION.md)). Recordings are local and
gitignored (they include private home images); `data/outputs/mac/*` is also local. Only the two
LiDAR/BEV images in `docs/media` are committed.

## Status by planned test
| Test | Status |
|---|---|
| A: stationary robot, stationary obstacle | **Preliminary only**: 10 s static cluttered kitchen, no designated obstacle or ground truth (below) |
| B: stationary robot, moving person | Not recorded yet |
| C: moving robot, stationary obstacle | Not attempted (needs human-controlled motion; odometry under motion unverified) |
| D: two objects crossing | Not recorded yet |

## Odometry start-up fix
The first replay of `mac_smoke_01` crashed: the first `cloud_base` midpoint (…543.758) preceded
the first odometry sample (…543.803) by 45 ms, and the replay raised on any unposed frame. The
recorder starts all subscribers together, so this occurs at the start of real recordings on any
OS. `tracking/pose_gate.py` now holds each frame until odometry at or after it exists, then poses
it by bracketing interpolation or skips it with a reason. No backward extrapolation by default; a
held start pose requires `start_hold_s` **and** a written justification. Odometry gaps longer than
the 0.1 s interpolation limit skip frames; if posed frames resume more than 0.5 s later, all
tracks end (no stationary-robot assumption, no ID reuse). Details: [TRACKING_ARCHITECTURE](TRACKING_ARCHITECTURE.md).
Synthetic tests: `tests/test_pose_gate.py`.

## Preliminary Test A: `mac_smoke_01`
Robot standing still (odometry extent < 1 mm over 10 s), static kitchen, nobody moving.
```bash
.venv-go2/bin/python scripts/run_tracking.py --input data/recordings/mac_smoke_01 \
    --output data/outputs/mac/tracking_smoke --window-s 0.3 --video
```
| Quantity | Result |
|---|---|
| LiDAR frames input / tracked / skipped | **156 / 155 / 1** (`before_first_odometry`) |
| Tracker resets | 0 |
| Output frame | `odom` for all 155 rows, poses from bracketing odometry |
| Track IDs (confirmed) | 98 (92) in 10 s for a static scene |
| Confirmed track-frames by state | UNCERTAIN 2,854; MOVING 217 (6.8%); STATIONARY 128 |
| IDs ever MOVING | 46 of 92 confirmed |
| Speed, all confirmed | median 0.115, p95 1.37, max 7.76 m/s |
| Tracks lasting >= 5 s | 24; speed median 0.098, p95 0.63 m/s; final state 21 UNCERTAIN, 3 STATIONARY |
| Their position std (median) | 0.10 m |
| Timing | `tracker.update` mean 34 ms; end to end 8.5 frames/s incl. LiDAR processing and BEV rendering |

**Interpretation.** Velocities are numerically valid and expressed in the odom frame, but here
the robot is static, so odom equals the body frame up to a constant and **ego-motion compensation
is not exercised**. Against the static-scene expectation (speed near zero, STATIONARY), the
tracker mostly reports UNCERTAIN, and short-lived fragments on cluttered walls and cabinets produce
false MOVING tracks up to several m/s. Long tracks drift about 0.1 m/s. Cause (inferred, not
proven): DBSCAN clusters on extended surfaces split/merge as the non-repetitive scan samples
different parts each frame, so centroids jump; Phase 3's documented cluster fragmentation
limitation. A track near (0.44, 0.0) m may be robot self-return; unverified. No ground truth exists,
so these are consistency measurements, not accuracy.

Evidence (local): `data/outputs/mac/tracking_smoke/{tracks.jsonl, skipped_frames.jsonl,
summary.json, config.json, static_scene_stats.json, frames/, tracking.avi}`, logs in
`data/outputs/mac/*.log`. Committed: [BEV, last frame](media/mac_static_tracking_bev.png),
[LiDAR clusters, 0.3 s window](media/mac_go2_lidar_window0p3.png).

## Perception on the same recording (`go2_replay.py`, unmodified processors)
- YOLO (yolov8n, `--device mps`): 110/110 frames; refrigerator, oven, microwave correct; a
  dishwasher labelled `refrigerator` (0.57), the known false-positive class. 10.4 ms median.
- LiDAR 0.3 s window: 155/156 frames (1 refused, no pose); clusters median 18 (9-24); ground
  median 2,140 points; 40.5 ms median.

## Synchronization and calibration
| Item | Status |
|---|---|
| LiDAR-odometry time base | Same robot clock; frame midpoint interpolation verified in tests only |
| Camera time | Host receipt only; capture latency unknown; ICMP offset unavailable on macOS |
| Camera intrinsics, distortion, `T_lidar_camera` | **Missing** — fusion not attempted, no synthetic calibration used |
| `T_base_lidar` | Firmware-derived (Phase 2), unchanged |
| Odometry under motion | Unverified |

Real RGB-LiDAR tracking is therefore **not** claimed; all tracking here is LiDAR-only.

## Next
Record Test A with a designated object and measured position (60 s), Test B with one person
crossing (ID continuity, nonzero speed), Test D with two people. Then tune DBSCAN / tracker gating
on these labelled sequences, and add a minimum-extent or persistence filter for wall fragments.
