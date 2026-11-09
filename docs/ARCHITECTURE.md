## Phase 4.1 social forecasting (2026-11-09)

The separate `phase4-social-forecasting` branch adds masked multi-pedestrian context, an 87,128-parameter social Transformer, three-seed/five-scene comparisons, validation robustness controls and an optional offline multi-track adapter. **The social model loses overall:** macro ADE 0.5812 m vs original 0.5565 m and CV 0.5214 m. Full suite: 259 passed/one hardware skipped. No tracker/hardware changes or navigation control.

See [social architecture](SOCIAL_FORECASTING_ARCHITECTURE.md), [results and exact commands](SOCIAL_FORECASTING_EXPERIMENTS.md), and [integration requirements](FORECASTING_INTEGRATION.md). Earlier milestones remain documented below.

## Phase 4 offline forecasting (2026-11-09)

Public annotated pedestrian forecasting is implemented on `phase4-trajectory-forecasting`; tracking and Go2 sensor APIs remain unchanged. A 78,680-parameter Transformer was trained and evaluated against constant velocity across five held-out scene identifiers. It did not win overall (macro ADE 0.5701 m vs CV 0.5214 m). Full suite: 231 passed, one hardware test skipped. No live forecasting validation or navigation control is claimed.

See [architecture](FORECASTING_ARCHITECTURE.md), [datasets](FORECASTING_DATASETS.md), [installation, commands and results](FORECASTING_EXPERIMENTS.md), and [Claude integration contract](FORECASTING_INTEGRATION.md). Existing phase descriptions below retain their historical validation scope.

# Architecture

Phase 3.7 adds a separate experimental tracking refinement profile. See
[the refinement evaluation](TRACKING_REFINEMENT.md) for measured gains, regressions and reproduction.

Phase 3.6 adds opt-in robustness within the existing tracker, preserving all sensor/perception
interfaces. See [TRACKING_ARCHITECTURE](TRACKING_ARCHITECTURE.md) for data flow and
[TRACKING_ROBUSTNESS](TRACKING_ROBUSTNESS.md) for the profile, benchmark and limitations.

**Phase 3:** timestamped LiDAR/fused results now feed `tracking.MultiObjectTracker` (import
from `tracking.multi_object_tracker`), with explicit odometry or fixed-sensor normalization,
Kalman velocity estimation, lifecycle/motion states, CV trajectories and headless BEV. See
[TRACKING_ARCHITECTURE](TRACKING_ARCHITECTURE.md) for the complete current interface.
The original perception and hardware code is unchanged; the sections below describe that core.

**Phase 2:** `adapters/go2` connects the real robot to the APIs below, without changing them.
```text
Go2 RTP video -> adapters/go2 (decode, keep receipt time) -> RGBDetector.detect -> RGBResult
Go2 DDS cloud_base + odometry -> adapters/go2 (decode, level, optional window)
    -> LidarMessage -> process_lidar -> LidarResult
RGB + LiDAR + Calibration -> FusionProcessor   [BLOCKED on real data: no calibration]
Recordings (JPEG/PNG, NPY, JSONL) replay through the same path without the robot or the SDK.
```
Only `receivers.py` and `dds_types.py` import the SDK; only `H264Decoder` imports PyAV. Core and
perception modules remain free of transport imports. Design details and reasons are in
[INTEGRATION_REPORT](INTEGRATION_REPORT.md). The rest of this file describes the Phase 1 core.

Images → `load_image` → `RGBDetector.detect` → `RGBResult`.
Clouds → `load_pointcloud` → `LidarProcessor.process` → `LidarResult`.
Results + `Calibration` → `FusionProcessor.fuse` → `FusionResult`.
Transport-independent messages wrap the same API; no sensor acquisition runs in core.

## Public APIs
- `RGBDetector(model, device, confidence, nms_iou, image_size)`: exact defaults in
  configs/rgb.yaml. `detect(path_or_bgr, timestamp=None, frame_id='camera_optical',
  clock_domain=None)` returns one RGBResult. `detect_batch(iterable)` returns results
  in input order. Batch metadata must be assigned by caller. Inference latency is wall
  time including preprocessing/postprocessing and first-call overhead, excluding model
  construction. Batch latency is amortized, not individual-image timing.
- `LidarProcessor(preprocessing={}, ground={}, clustering={}).process(points,
  timestamp=None, frame_id='lidar', clock_domain=None)` returns filtered XYZ, ground,
  unclassified points, obstacles, labels and elapsed time. Constructor dictionaries
  map directly to the named functions and configs/lidar.yaml keys.
- `Calibration(K, T_lidar_camera, image_size, camera_frame, lidar_frame, distortion,
  rectified, synthetic)` or `Calibration.from_yaml(path)` validates rigid extrinsics.
- `project_points(XYZ, calibration, min_depth=1e-6)` returns visible UV, optical depth
  and original point indices. Camera-to-LiDAR extrinsics are inverted internally.
- `FusionProcessor(**configs/fusion.yaml).fuse(rgb, lidar, calibration)` validates
  compatibility and returns fused detections plus unmatched input indices/cluster IDs.
- `load_pointcloud(path_or_array, bin_fields=4)`, `load_image(path_or_array)`,
  `save_json(path, result)` and message helpers are transport-independent.

Results are dataclasses in core/types.py. Boxes are `[xmin,ymin,xmax,ymax]` pixels
or XYZ minimum/maximum in the LiDAR frame; 3D position is observed AABB midpoint,
not an estimated complete physical-object center. Timestamp of a fused 3D detection
is the LiDAR observation time; RGB time is retained separately.

## Geometry and limitations
Ground RANSAC samples planes with seeded NumPy RNG, constrains normal tilt to +/−Z
and plane height at the sensor origin. Samples must also have observed support within
the expected height band, avoiding spurious extrapolated ground from elevated objects. Set expected_ground_z from actual mounting /
pose; defaults are illustrative. No qualifying plane means no ground removal, rather
than silently labeling a wall. This assumes a gravity-aligned scan; steep terrain and
body roll need upstream rotation or a future gravity input. No de-skew is implemented.

Open3D voxel downsampling creates centroids and drops intensity. Labels index processed
points, not raw input. DBSCAN uses fixed metric radius / minimum support; sparse points
remain unclassified, and close/touching objects may merge. No automotive density or full
360-degree scan is assumed. Short accumulated windows must already share a frame/time
reference and be motion compensated by the producer. The Phase 1 core has no temporal tracker; Phase 3 now consumes its outputs in a separate package.

Association scores are fractions of visible cluster points inside each RGB box; require
minimum support and fraction. SciPy Hungarian assignment maximizes total score one-to-one.
This is a baseline: no occlusion model, instance splitting, semantic LiDAR classifier,
or calibrated association probability. RGB confidence is preserved, not multiplied by
association score. Unmatched clusters remain geometric obstacles. Real fusion is always
reported as unvalidated by this offline implementation, even with user-provided calibration.

## Dependencies and configuration
PyTorch/torchvision + Ultralytics for inference; OpenCV for image I/O/annotation;
NumPy/Open3D for geometry and cloud I/O; SciPy for assignment; PyYAML for config.
Heavy imports are lazy except where explicitly needed. Core has no ROS imports.
`configs/default.yaml` selects component paths for the fusion CLI; individual CLIs
accept their own `--config`. Run scripts from repo root. Unknown constructor keys fail
rather than silently disappearing. Full direct pins are in pyproject.toml; the tested
transitive environment is recorded separately. ROS adapter is intentionally documentation
(Phase 2 uses the Unitree SDK directly because no system ROS 2 is installed; see `adapters/go2`).
