# Phase 3 real-world validation (2026-11-07, macOS)
Real Go2 data captured on the Mac ([MAC_CONNECTION](MAC_CONNECTION.md)). Recordings are local and
gitignored (they include private home images); `data/outputs/mac/*` is also local. Only the two
LiDAR/BEV images in `docs/media` are committed.

## Status by planned test
| Test | Status |
|---|---|
| A1: stationary robot, stationary obstacle | **Done** (`test_a1`, 60 s): one ID for 60 s, 4-5 cm jitter; rarely labelled STATIONARY |
| A2: stationary robot, person near obstacle | **Done** (`test_a2`, 50 s): YOLO detects the person; LiDAR fragments the person into ~16 short IDs |
| B: stationary robot, moving person | **Done** (`test_b`, 65 s): 96% coverage, speed roughly right, 3/4/10 IDs per pass |
| C: moving robot, stationary obstacle | **Done** (`test_c`, 80 s, human-driven): box fixed in odom to 6-9 cm through ±23° turns and 0.5 m walks; IDs still fragment |
| D: two objects crossing | Not recorded (needs a second person) |
| 10 s smoke run | Static kitchen, no designated object (section below) |

A1, A2, B: robot standing still (odometry extent < 3 mm); C: human-driven (below), `--window-s 0.3`, default
`configs/tracking.yaml` and `configs/go2_lidar.yaml`, Mac `.venv-go2`. Ground truth is limited to a
tape-measured box position and, for B, a reference trajectory from background-subtracted raw
points; there are no labelled boxes or IDs. Positions are in `base_link` (body centre, x forward,
y left), obtained from the odom-frame tracks with the static recorded odometry pose.

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

## 10 s smoke run: `mac_smoke_01`
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

## Test A1: water-bottle cases about 2 m ahead (`test_a1`)
Cases placed straight ahead, near face about 2.0 m from directly below the camera/LiDAR (user
measurement), stack about 0.3 m tall. Recording started about 6 s after the user's stopwatch; a
person passed close to the robot's right side at about 24 s (1.5 s in camera view).
| Quantity | Result |
|---|---|
| Frames input / tracked / skipped | 925 / 924 / 1 (`before_first_odometry`) |
| Box identity | **one ID (#13) in all 924 frames, 60.1 s, 0 switches**, including during the person's pass |
| Box centroid | (2.50, 0.01) m, std (0.036, 0.051) m; box 0.77 x 0.85 x 0.29 m |
| Box speed (truth 0) | median 0.068, p95 0.16 m/s; 0.45 m/s peak while the person passed |
| Box motion state | UNCERTAIN 870, STATIONARY 52, MOVING 2; Lost in 95 frames (missed detections) |

Centroid versus tape: the measured 2.0 m is to the near face from the sensor; the track is the
cluster centroid from the body centre, so sensor offset plus half the box depth account for the
difference qualitatively. Neither offset was measured, so this is not a range-accuracy result.
**Why STATIONARY is rare:** the steady-state velocity standard deviation is about 0.14 m/s, and
`classify_motion` needs speed + 1 sigma <= `stationary_speed` (0.15), i.e. speed <= ~0.01 m/s; any
missed frame also forces UNCERTAIN. A config/classifier issue, not a sensor one.

## Test A2: person beside the cases (`test_a2`)
Intended: stand still beside the cases 10-40 s. Camera frames show walking at about 8, 13 and 22 s
and standing just left of the cases at about 36 s; not a clean standing test.
| Quantity | Result |
|---|---|
| Frames input / tracked / skipped | 769 / 769 / 0 |
| YOLO person | 405 of 711 frames, median confidence 0.89, legs only (low camera) |
| LiDAR sees the person | yes: 95 points per 0.3 s at (1.93, +0.61) m versus 12-16 when absent |
| Person tracks | ~16 IDs within 0.5 m of the standing spot, longest 2.4 s, alternating Confirmed/Lost |
| Box identity | 6 IDs (#22, 71, 134, 140, 146, 247) while the person moved nearby; #247 stable 26-50 s |

## Test B: person walking in front of the cases (`test_b`)
The person entered from the robot's right three times (about 8-17, 22-34 and 40-53 s), walked
to the centre about 1.85 m ahead, paused 3-5 s just in front of the cases, and left to the right.
Reference: per-0.2 s median of points not in the empty-scene background (first 6 s, 10 cm voxels,
region x 0.8-2.15 m, |y| < 2 m); reference velocity by 0.6 s finite difference. This reference is
itself noisy and is not ground truth.
| Pass | Coverage by a track within 0.5 m | IDs (changes) | Walking: reference / track speed, median vel. error | Pausing: track speed, states |
|---|---|---|---|---|
| 1 | 93% | 3 (3) | 0.65 / 0.77 m/s, 0.25 m/s | 0.20 m/s; mostly UNCERTAIN |
| 2 | 98% | 4 (7) | 0.74 / 0.89 m/s, 0.37 m/s | 0.12 m/s; mostly UNCERTAIN |
| 3 | 98% | 10 (24) | 0.80 / 0.80 m/s, 0.40 m/s | 0.17 m/s; mostly UNCERTAIN |
Overall 96% coverage and 19 IDs for one person. While walking, 55-73% of track-frames are MOVING.
The cases (2.5, 0) were re-identified once per pass (main IDs #21, #101, #337, #505; 16 IDs in
total, 81% coverage): the person standing 0.3-0.6 m in front occludes or merges with them.

## Test C: robot moved by hand, cases static (`test_c`)
The user drove the robot with the remote at low speed; nothing was sent from the Mac. Motion from
`robot_odom`: yaw to -24 deg, back, to +21 deg, back; forward about 0.47 m; further turns
(-11 to -21 deg) and sideways steps; end 0.33 m from the start at -1.9 deg. Pitch/roll stayed
within ±3 deg; one odometry gap > 30 ms (30.7 ms). The operator stood behind the robot.
| Quantity | Result |
|---|---|
| Frames input / tracked / skipped | 1,231 / 1,230 / 1 (`before_first_odometry`), no tracker resets |
| Box apparent motion **in the robot frame** (what an uncompensated tracker would see) | x 1.89-2.53 m, **y -0.98 to +1.23 m** |
| Box front cluster in **odom**, robot still / rotating / translating | std (0.085, 0.067) / (0.059, 0.062) / (0.061, 0.066) m |
| Box speed, robot still / rotating / translating | median 0.068 / 0.071 / 0.056 m/s; p95 0.63 / 0.28 / 0.28 m/s |
| MOVING labels on the box while the robot rotated or translated | **0** (of 163 frames) |
| Box odom drift, first 5 s to last 5 s | 0.11 m (mostly y) over 80 s |
| Box identity | 22 IDs and 30 changes in 955 covered frames; longest runs: #79 (6-16 s, through -23 deg), #179 (18-34 s, through +21 deg), #430 (44-72 s) |

**Ego-motion compensation works on real data at this scale:** a 2.2 m lateral sweep in the robot
frame becomes a 6-9 cm spread in odom, and robot motion does not create false velocity or MOVING
labels. The 0.11 m drift may be odometry drift or the visible box face changing with viewpoint;
the two cannot be separated without external ground truth. Identity, not geometry, fails: the box
is re-identified roughly every 5-15 s, often during turns, and a second fragment about 0.35 m behind
its front face (#280, #420) appears intermittently. Odometry accuracy itself was not measured
independently (no motion capture or tape check of the end pose).

## Findings across A1, A2, B and C
1. A solid static object, unoccluded, is tracked with one stable ID and centimetre jitter.
2. A walking person is followed (96% coverage) and speed magnitude is roughly right (median
   error 0.25-0.40 m/s against a noisy reference), so **nonzero, sensor-relative velocities are
   real**; with the robot static, odom-frame equals sensor-relative motion.
3. Identity is the main weakness: people fragment into many short IDs (sparse leg clusters,
   split/merge), and occlusion or proximity re-identifies nearby static objects.
4. Motion states are dominated by UNCERTAIN because of the classifier thresholds above.
5. Ego-motion compensation holds under real human-driven turns (±23 deg) and short walks: the
   static box stays within 6-9 cm in odom and gets no MOVING labels (Test C).
Suggested Codex work, on these recordings: cluster association for people (leg merging, eps/min
points), longer coasting before deletion, occlusion-aware matching, classifier thresholds tied to
the actual velocity covariance. Do not tune on these alone without labels.

Evidence (local): `data/outputs/mac/test_{a1,a2,b,c}_tracking/` (tracks.jsonl, frames/, tracking.avi,
summary.json, skipped_frames.jsonl, `box_track_stats.json`, `person_stats.json`,
`person_eval.txt`, `raw_foreground_traj.npy`, `ego_eval.txt`), `data/outputs/mac/test_*_replay/` (YOLO), logs
`data/outputs/mac/*.log`. Committed BEV frames: [A1](media/mac_test_a1_bev.png),
[A2](media/mac_test_a2_bev.png), [B](media/mac_test_b_bev.png), [C, robot turned](media/mac_test_c_bev_turned.png).

## Original vs improved tracker on these recordings (added 2026-11-07)
The sections above describe the original tracker, and their numbers are unchanged. The improved
opt-in profile (`a4496a3` with `configs/tracking_robustness.yaml`) was replayed on identical
cached inputs. A fresh baseline replay matched the original outputs above in every frame (3,924
of 3,924). Full report: [TRACKING_REAL_COMPARISON](TRACKING_REAL_COMPARISON.md). There are no
annotations; all values are proxies.

| | A1 | A2 | B | C |
|---|---|---|---|---|
| Created IDs | 514 → 261 | 464 → 243 | 624 → 333 | 699 → 334 |
| Box dominant-ID share | 0.99 → 0.99 | **0.50 → 0.96** | **0.31 → 0.49** | **0.33 → 0.59** |
| Box ID changes (returns to an old ID) | 4 (2) → 6 (3) | 15 (5) → 28 (17) | 13 (1) → 14 (5) | 30 (9) → 30 (22) |
| Duplicate-pair frames | 365 → 483 | 277 → 383 | 380 → 504 | 682 → 832 |
| Reactivations > 1 m away | 77 → 99 | 62 → 114 | 96 → 163 | 88 → 93 |
| Box MOVING frames | 5 → 0 | 18 → 0 | 50 → 2 | 39 → 2 |
| `tracker.update` mean | 23.8 → 33.0 ms | 23.4 → 32.9 ms | 23.8 → 33.0 ms | 29.3 → 40.7 ms |

**Better.**
- The box keeps its main ID through the person's visits (A2), the first two occlusion passes (B)
  and the ±23° turns (C).
- False MOVING labels on static structure almost disappear: A1 confirmed MOVING frames
  1,550 → 127.

**Worse.**
- More duplicate and flickering tracks, and more long-range reactivations.
- Person identity is not improved (B: 19 → 20 IDs).
- Walking-person MOVING recall falls from 62% to 39%.
- About 40% more tracker time.

**Unchanged.**
- Velocity estimates: static box speed about 0.07 m/s, σ about 0.14 m/s.
- Ego-motion compensation: 0 MOVING labels while the robot moves; box odom spread unchanged.

Comparison BEVs: [A1](media/compare_test_a1_bev.png), [A2](media/compare_test_a2_bev.png),
[B](media/compare_test_b_bev.png), [C](media/compare_test_c_bev.png).

## Perception on the 10 s smoke recording (`go2_replay.py`, unmodified processors)
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
Test D needs two people. Test C could be repeated with a taped end position to check odometry drift. Labelled IDs/positions (even hand-annotated BEV) are needed before tuning.
