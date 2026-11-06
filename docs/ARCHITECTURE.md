# Architecture
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
reference and be motion compensated by the producer. No temporal tracker exists.

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
until Claude confirms topics and message formats.
