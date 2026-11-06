# Phase 2 integration report (2026-11-06)
Branch `feature/go2-sensor-integration`, based on `codex/phase1-perception`. Evidence is in
[SENSOR_VALIDATION](SENSOR_VALIDATION.md); setup is in [GO2_CONNECTION](GO2_CONNECTION.md).

## Definition of done
| Requirement | Status |
|---|---|
| Real RGB retrieved and saved | **Done** |
| Real LiDAR retrieved and saved | **Done** |
| Offline replay | **Done** |
| Existing YOLO on real RGB | **Done**; correct person boxes visually confirmed |
| Existing LiDAR processor on real clouds | **Done**; plausible but no ground truth |
| Interfaces and limits documented | **Done** |
| Existing tests still pass | **Done**: all 44 original tests are in the 68 passing |
| State and handoff docs updated | **Done** |
| Fusion (additional objective) | **Blocked**: no calibration; synchronization unverified |

## Files
**Added: adapter** (`src/go2_perception/adapters/go2/`): `pointcloud.py`, `rtp_h264.py`, `receivers.py`,
`window.py`, `recording.py`, `stats.py`, `clock.py`, `dds_types.py`.

**Added: tools** (`scripts/`): `go2_discover.py`, `go2_probe.py`, `go2_record.py`, `go2_replay.py`,
`go2_live.py`, `go2_view_cloud.py`, `go2_window_study.py`, `go2_moving_object_study.py`.

**Added: other:** `configs/go2_sensors.yaml`, `go2_lidar.yaml`, `go2_rgb.yaml`;
`tests/test_go2_adapter.py`, `test_go2_hardware.py`, `fixtures/go2/real_cloud_pair.npz` (86 KB of
real LiDAR data; no images); `docs/media/` (LiDAR images only); the three new docs.

**Modified:** `pyproject.toml` (`go2` extra, `hardware` marker), `.gitignore`, `README.md` and four
Phase 1 docs (additions and corrections).

**Kept unchanged from a prior session:** `scripts/test_camera.py`, `test_lidar.py`, `lidar_state.py`,
`lidar_diag.py`, `go2_ros_env.sh`, `install_sdk.sh`; `output/` (ignored).

**Codex modules: no changes.** No perception source, Phase 1 test, or Phase 1 config was touched.

## How the adapter works
```text
RTP multicast -> parse_rtp -> AccessUnitAssembler -> H264Decoder -> CameraFrame -> RGBDetector.detect
DDS PointCloud2_ -> frame_from_msg -> LidarFrame -> to_lidar_message -> process_lidar -> LidarProcessor
DDS Odometry_ -> PoseBuffer -> SlidingWindowAccumulator (optional) -----^
```
- **Decoding** follows each message's field table, not a hard-coded layout.
- **Frame safety:** `to_lidar_message` rejects `cloud` (tilted sensor axes) and `cloud_deskewed`
  (accumulated, in `odom`). Only body-frame clouds match the Phase 1 axis contract.
- **Levelling:** with an attitude supplied, roll and pitch are removed and the frame becomes
  `base_link_level`, meeting the gravity assumption while the robot is reasonably level.
- **Timestamps:** LiDAR uses the frame midpoint on clock `go2_robot`. RGB uses first-packet receipt
  on clock `host_receipt`. The names differ on purpose, so `FusionProcessor` rejects the pair until a
  real clock conversion exists.
- **RGB bypasses `RGBMessage`:** its `K` is mandatory and no real intrinsics exist. Calling
  `RGBDetector.detect` directly is an existing public API and needs no invented matrix.
- **Damaged video:** frames are skipped until the next complete IDR, instead of passing corrupted
  pixels to the detector.

## Issues found
**Phase 1:** no defects found. Two contract points are worth noting, neither requiring a code change:
1. `RGBMessage` requires `K`, even for detection-only use. Reproduce: construct one without `K`.
   The adapter works around it; Codex may consider making `K` optional.
2. `configs/lidar.yaml` has `expected_ground_z: -0.5`, illustrative per the docs. The real standing
   value is about -0.33, supplied by `configs/go2_lidar.yaml` without modifying the original.

**Fixed in this phase's adapter:** decoder errors from joining mid-GOP.

**Remaining limitations:**
- `go2_lidar.yaml` ground height fits a **standing** robot only; a lying robot needs another value.
- `refrigerator` false positives in this home on COCO yolov8n at 0.25 confidence.
- DBSCAN parameters are Phase 1 defaults, and cluster quality has no ground truth.
- Accumulated frames lose their per-point `time`, and `level_points` uses one attitude per frame.
- RSS peak reporting via `ru_maxrss` is misleading when launched via `exec`.
- Not implemented: rosbag output (no ROS 2 installed), RViz2 view (Open3D instead), near-field
  self-filter for raw `cloud`.

## Calibration blockers (fusion)
Required, in order:
1. **Camera intrinsics and distortion.** Capture a checkerboard or ChArUco board using the recorder,
   then compute `K` and a fisheye or Brown-Conrady model.
2. **Rectification** of images with the matching rectified `K`, as `Calibration` requires.
3. **`T_lidar_camera`,** measured with a target visible in both sensors. Use `cloud` or
   `T_base_lidar` to express it in the chosen LiDAR frame, and check it with projected-point overlays.
4. **Camera capture latency,** for example with a sharp event seen by both sensors, because RTP gives
   no capture time. Robot stamps can then be converted using the measured clock offset.

## Recommended next tasks
**Codex:**
1. Add a replay-based regression test from a short non-image LiDAR recording.
2. Decide whether `RGBMessage.K` should be optional, and add a contract test.
3. Tune `go2_lidar.yaml` clustering with labelled recorded frames, for sparse real data and 0.3 s windows.
4. Add a ground-height input from odometry (about `-position.z`) instead of a fixed standing value.
5. Decide how to filter the `refrigerator` false positives or how to handle indoor classes.
6. Start tracking from `stage_c_person`-style sequences using frame-level detections.

**Hardware (Claude and the user):**
1. Physical link-drop test.
2. Calibration sessions above.
3. Motion validation with the user driving or carrying the robot, then re-run the window study.
4. Optional, with approval: install system ROS 2 Jazzy for rosbag2 and RViz2.
