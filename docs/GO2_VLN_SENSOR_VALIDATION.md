# Go2 VLN sensor validation (Phase 5D, 2026-11-11)

Session on the physical Go2 Pro from macOS, robot **stationary**, read-only DDS subscribers only.
**No locomotion command, no robot control, no setting changed.** Each physical step was confirmed by
the user first. Raw data stay local and git-ignored (`data/recordings`, `data/outputs/phase5d`).

## Physical steps performed
| Step | Result |
|---|---|
| Link check | `en9` active, 192.168.123.222/24; robot 192.168.123.161 answered 3/3 pings, ~1 ms |
| Discovery (10 s) | 95 publications, same as 2026-11-07 |
| Probe (15 s) | all streams as before (table below); `lidar_state`, `frontvideostream`, `voxel_map`, `grid_map`, `uslam/cloud_map` silent |
| Recording `p5d_static_02` (20 s) | 282 RGB (lossless PNG), 308 × 3 LiDAR topics, odom, IMU, sport state; 0 RTP loss, 0 decoder errors, 0 writer drops |

`data/recordings/p5d_static_01` (8 s, 12:36 the same day) already existed and was not created by this
session; it was left untouched and only read for timing.

## LiDAR (re-verified)
| Topic | Rate | Points / msg | Frame | Overlap with previous |
|---|---|---|---|---|
| `cloud` | 15.47 Hz | 3 895–4 177 | `utlidar_lidar` | 0% |
| `cloud_base` | 15.6 Hz | 1 166–1 600 (mean 1 372) | `base_link` | 0% |
| `cloud_deskewed` | 15.6 Hz | ~11 372 | `odom` | 88% (rolling buffer) |

Fields: x y z intensity (float32), ring (uint16, always 1), time (float32 s since stamp, 0–62 ms;
absent in `cloud_deskewed`). Units metres. Stamp = scan start (inferred). LiDAR model still not
self-reported (`lidar_state` silent); probably Unitree L2 (inferred in Phase 2). `cloud_base`
self-returns: 1 point within the body box over 425 k accumulated points; nothing within 0.3 m XY.

**Partial scans confirmed:** one `cloud_base` scan covers 190° of azimuth (r > 0.5 m); 0.3 s → 305°,
1 s → 315°. **No double accumulation:** the interface accumulates `cloud_base` only.

### Accumulation interval (user decision)
`scripts/phase5d_lidar_accumulation.py` on `p5d_static_02`, six non-overlapping reference times:
| Window | Points | Completeness vs 20 s union | Repeatability Jaccard median / min |
|---|---|---|---|
| 1 scan | 1.3 k | 10% | 0.10 / 0.09 |
| 0.3 s | 6.9 k | 34% | 0.36 / 0.32 |
| **0.5 s** | 11.0 k | 45% | 0.49 / 0.38 |
| **1.0 s** | 21.3 k | 55% | 0.63 / 0.41 |
| 1.5 s | 31.8 k | 60% | 0.63 / 0.44 |
| 2.0 s | 42.7 k | 63% | 0.64 / 0.47 |
| 3.0 s | 63.5 k | 75% | 0.76 / 0.71 |

The user reviewed the BEV and obstacle-grid comparison images and chose **1.0 s for display and 0.5 s
for obstacle detection** (`configs/go2_vln_sensors.yaml`). Static-scene metrics only; from Phase 2,
a walking person spreads 0.37 m at 0.3 s, 0.54 m at 0.65 s and 0.69 m at 1 s. Phase 5A's own 0.3 s
configuration is unchanged.

### Representations
| Representation | Status |
|---|---|
| Raw single scan (`cloud`, `cloud_base`) | exists (robot) |
| Accumulated cloud (0.5 / 1.0 s, odom-compensated) | **implemented** (`vln.replay`) |
| 2D occupancy (unknown / floor seen / occupied) | **implemented** (`vln.geometry.occupancy`) |
| 2.5D elevation (max z per cell) | **implemented** |
| Obstacle distance per 5° sector | **implemented** (`vln.geometry.sectors`) |
| Projected depth in the image | implemented, **blocked** by missing calibration |
| Robot height map (`rt/utlidar/height_map_array`, 15.9 Hz) | published, format not decoded or validated |
| `voxel_map`, `grid_map`, `uslam/cloud_map` | advertised, 0 messages |

## Camera
See [GO2_CAMERA_CALIBRATION](GO2_CAMERA_CALIBRATION.md). Intrinsics: **not measured** (no factory
data; no printed target available to the user today). Screen and print ChArUco targets prepared.

## Extrinsics
See [GO2_LIDAR_CAMERA_EXTRINSICS](GO2_LIDAR_CAMERA_EXTRINSICS.md). **Not measured.** Method and
acceptance criteria documented; requires intrinsics and a large flat board.

## Timing
See [GO2_SENSOR_TIMING](GO2_SENSOR_TIMING.md). New measurements: robot clock resets each boot;
host − robot offset per recording (odom floor, ~10 ppm drift); camera receipt latency 48 ms (26–58)
from `test_c` yaw cross-correlation.

## Offline demo on real data
`PYTHONPATH=src .venv/bin/python scripts/phase5d_vln_replay_demo.py --recording
data/recordings/p5d_static_02 --index 200 --output data/outputs/phase5d/demo` →
validation issues: none; npz round trip OK (3.2 MB with 2 history frames); warning: projected depth
unavailable (calibration missing). Field report:
| Field | Status |
|---|---|
| visual.image | ok, 720×1280×3 RGB + 2 history frames |
| visual.stamp_robot | estimated, σ 16 ms |
| visual.intrinsics | **missing** |
| geometric.points / grid | ok: 11 292 points from 8 scans; 8 674 unknown / 1 017 floor seen / 309 occupied cells |
| geometric.sector_range | partial: 45 of 72 sectors have an obstacle return |
| geometric.projected_depth | **missing** (needs calibration) |
| robot.T_odom_base | ok (accuracy under motion unverified) |
| robot.velocity_frame | uncertain |

The same script with `--synthetic` exercises the full path, including a LiDAR overlay and projected
depth with the synthetic calibration (tests check that the box face projects to its expected depth).

## Dynamic-obstacle collision monitor: feasibility
Supported by the interface today: per-cell occupied / floor-seen / unknown with explicit unknown;
coverage awareness (one scan covers ~190°, so a monitor must use ≥ 0.3–0.5 s or track per-sector
age); sector nearest-obstacle ranges; ego-motion-compensated windows; consistent robot-clock stamps.
Temporal change can be computed by differencing successive 0.5 s grids, but a cell that turns
unknown is **not** evidence that an obstacle left.

Not solved here: moving-object recall. Phase 5B showed that 38% of walking-person frames have no
person returns in the latest scan, so a short window misses people and a long one smears them (0.5 s
≈ 0.25 m at 0.5 m/s). Near-field (< 0.3 m) and low (< 10 cm) obstacles are invisible to `cloud_base`;
raw `cloud` with a body self-filter would be needed. A conservative monitor would: stop-request when
any sector within the safety distance is occupied **or unknown** in the travel direction; use the
0.5 s window; age out evidence per sector; never use model output or 3D boxes as clearance. No
planner, monitor or autonomy was implemented in this phase.

## Tests
`tests/test_vln_interface.py`: 30 passed (RGB decode/colour/shape, timestamp provenance and clock
conversion, causal camera selection, LiDAR decoding and windows, grid and sector semantics,
transforms and invalid transforms, projection and both distortion models, intrinsics under
resize/letterbox, tensors, missing calibration, synthetic-calibration opt-in, projected depth, missing
odometry, missing RGB, clock-domain mismatch, validation, npz round trip, unknown schema,
calibration metadata/units, recorded user choice, fixture reproducibility).
Full suite (`PYTHONPATH=src .venv/bin/python -m pytest -q`): **381 passed, 3 skipped** (351 + 3
before this phase).

## Limitations
- No camera intrinsics, no extrinsics, no projected depth or RGB–LiDAR fusion on real data.
- Camera latency measured once (one boot, yaw only).
- Robot stationary for all new data; nothing validates behaviour while walking.
- Floor height −0.33 m assumes a standing robot.
- LiDAR model identity, intensity units and the odometry twist frame remain unconfirmed.
