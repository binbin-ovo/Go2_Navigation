# Phase 3 tracking architecture

Phase 3 adds a transport-independent tracker after Phase 1 perception. It does not change
Claude's SDK communication, original perception internals, or Phase 2 live scripts. The actual
Phase 2 transport is Unitree DDS + RTP (not a mandatory ROS 2 integration).

```text
LiDARResult / FusionResult / timestamped detection JSONL
 -> tracking.normalization -> Detection3D
 -> localization.ego_motion -> odom-frame measurements + covariance
 -> gated Hungarian association -> planar constant-velocity filters
 -> lifecycle / motion classification -> TrackOutput / CV trajectories
 -> headless OpenCV BEV, JSONL, optional MJPG AVI
```

## Public API and units
```python
from go2_perception.io.config import load_config
from go2_perception.tracking.multi_object_tracker import MultiObjectTracker
from go2_perception.tracking.normalization import from_lidar, from_fusion

tracker = MultiObjectTracker(load_config('configs/tracking.yaml')['tracker'])
# lidar_result and pose must refer to the same acquisition clock and compatible frames.
tracks = tracker.update(from_lidar(lidar_result), lidar_result.timestamp, pose,
                        frame_id=lidar_result.frame_id,
                        clock_domain=lidar_result.clock_domain)
```
Reuse one tracker per sequence. IDs increase monotonically within that instance, and restart
on a new instance. `from_fusion(fusion_result)` preserves RGB class/confidence for matched
3D detections. Add only **unmatched** LiDAR clusters if combining the two sources: never
submit the same cluster twice. RGB-only boxes provide no depth and cannot be tracked in 3D.

`core/tracking_types.py` defines:
- `Detection3D(timestamp, position, frame_id, clock_domain, detection_id=None,
  bbox_min=None, bbox_max=None, class_name=None, confidence=None,
  measurement_covariance=None)`. XYZ metres; confidence optional [0,1]; covariance is
  finite symmetric positive-definite 3x3 XYZ in m². Missing covariance uses configured
  isotropic variance; unknown semantic class and geometric confidence stay `None`.
  Boxes require both finite ordered XYZ bounds. Duplicate non-null source IDs in a frame
  are rejected. Source IDs identify measurements, **not** persistent objects.
- `RobotPose(timestamp, T_parent_child, parent_frame, child_frame, clock_domain)`.
  Matrix is right-handed rigid 4x4, mapping child points to parent; translation in metres.
- `TrackOutput`: persistent ID, current timestamp/frame/clock, position XYZ, velocity
  `[vx,vy]` m/s, covariance ordered `[x,y,vx,vy]`, translated observed AABB, class and
  latest detection confidence, lifecycle status, motion state, last observation timestamp,
  age in seconds, total hits, consecutive misses, recent filtered position history with
  observed/predicted flags, and predicted positions/covariances. History is a detached
  bounded snapshot. Detection confidence is not an invented tracking probability.
  Reliability is expressed through status, observation counts and covariance.
- `TrajectoryPoint`: absolute future timestamp, XYZ position (z held), 2x2 XY covariance.

Empty detections represent a missed observation; call update at the real observation time.
Pass explicit `frame_id` and `clock_domain` for empty frames. Reject nonfinite, duplicate,
out-of-order or intervals below `min_dt`, before changing tracker state. The clock binds on
first valid update (or is configured), then must remain equal. Detection times must agree
with update time within 1e-6 s by default. No clocks or missing timestamps are synthesized.

## Frames and ego-motion
Default mode `odom`: already-odom detections need no pose. For body detections use
`p_odom = T_odom_base @ p_base`. For a distinct configured sensor frame, supply
`T_base_sensor` and a pose whose child equals configured `base_frame`:
`p_odom = T_odom_base @ T_base_sensor @ p_sensor`.
No implicit inversion, guessed extrinsic, fallback identity, or frame renaming is allowed.
Pose clock, parent, child and age are validated (default age tolerance 0.05 s).
Full 3D rotations/translations are supported. Rotate measurement covariance by `R C Rᵀ`;
transform all eight AABB corners and recompute enclosing bounds. This can enlarge boxes.

The tracking frame is **local odometry**, not a globally consistent map. Odometry drift and
pose/calibration uncertainty are not included in filter covariance: results are conditional
on provided poses. The implementation does not estimate localization or rectify camera data.

Without poses, explicitly choose `mode: fixed_sensor`. Inputs must remain in `sensor_frame`;
outputs are labelled `fixed_sensor:<frame>` with `velocity_reference=fixed_sensor_relative`.
This is appropriate for a known stationary sensor or a deliberately relative offline study.
It does not establish that a real robot is stationary. Supplying a pose in this mode is rejected.

## Kalman model
State `[x,y,vx,vy]`, measurement `[x,y]`; z is the most recent measurement, not filtered or
predicted vertical dynamics. Use actual elapsed time. `max_prediction_interval` subdivides
prediction into bounded steps without discarding elapsed time. Long unobserved gaps delete
tracks before association; stale IDs are never resurrected by a late detection.

For each axis, white acceleration spectral density q produces
`Q = q * [[dt³/3, dt²/2], [dt²/2, dt]]` in position/velocity ordering.
Default q=.04 m²/s³; default measurement variance=.01 m²; initial velocity variance=4 (m/s)².
Innovation solves use `numpy.linalg.solve`, and updates use Joseph covariance form plus
symmetrization. Positive-definite input covariances are required; no explicit matrix inverse
is used in the filter. Existing geometry utilities remain the source of rigid transforms.

## Association and lifecycle
Before Hungarian assignment, gate planar Euclidean distance (1.5 m), squared Mahalanobis
innovation (13.82), vertical separation (1 m) and incompatible known semantic classes.
Unknown classes can match known or unknown classes. Valid cost is a normalized weighted sum
of position distance (.4), Mahalanobis distance (.6), and optional 1−3D-AABB-IoU (0 by default).
Dummy columns allow unmatched tracks, with penalties chosen to maximize feasible match count
before minimizing cost. No appearance model or re-identification embedding is used.

Nearly identical within-frame centers (<.02 m) with compatible classes and overlapping boxes
(IoU>.9; or both boxes absent) are deduplicated. An unmatched observation within .3 m of a
missed confirmed track with compatible class and IoU>.3 does not immediately create a duplicate
track. Both distances can be disabled with 0; this conservative heuristic can suppress a true
new object extremely close to another. Output reports the number suppressed each update.

- New observations create `Tentative` tracks with zero mean initial velocity and high uncertainty.
- Three hits in a rolling 1-second confirmation window produce `Confirmed`; the hits need not
  be consecutive, so intermittent LiDAR observations can establish identity.
- One missed update retains status but reports `UNCERTAIN` motion and no future trajectory.
- Two consecutive misses produce `Lost`; prediction and gated reacquisition retain the same ID.
- More than ten missed updates or >1.5 s since last observation produces `Deleted`.
  Deleted objects are removed; their IDs are emitted in `tracker.deleted_ids` for that update.
- Reacquired previously confirmed tracks return to Confirmed. History stores at most 40 updates.

## Motion and trajectories
Require confirmation, at least 5 hits, age >=.5 s, a current observation, and maximum velocity
standard deviation <=.3 m/s. Let speed=norm(v), std=sqrt(max eigenvalue of velocity covariance),
and interval `[max(0,speed−sigma*std), speed+sigma*std]` (sigma=1 by default).
MOVING requires lower bound >=.3 m/s; STATIONARY requires upper bound <=.15 m/s. Hysteresis
retains a previous moving/stationary state only across the intermediate band. Otherwise
UNCERTAIN. In fixed mode these are **relative** motion classifications.

`tracking.trajectory.predict_trajectory(state, covariance, z, timestamp, horizon=3,
step=.2, process_noise=.4)` is a standalone replaceable baseline. The tracker passes its
configured q=.04. Confirmed currently observed, non-uncertain tracks get 15 samples over
3 seconds with covariance propagation. Lost/tentative/uncertain tracks get an empty prediction,
rather than a falsely precise forecast. Stationary tracks use their small estimated velocity
with increasing uncertainty; z remains fixed. A future learned predictor should consume the
same history/frame/time fields and return equivalent trajectory records.

## File formats and replay
Preferred directory contains `frames.jsonl` and optional `manifest.json` with
`{"schema":"go2-tracking-v1","synthetic":true|false}`. Each JSONL row is one observation:
```json
{"timestamp":1.0,"frame_id":"base_link","clock_domain":"synthetic","detections":[{"detection_id":"scan1:cluster0","position":[3,1,0.5],"bbox_min":[2.8,0.8,0],"bbox_max":[3.2,1.2,1],"class_name":null,"confidence":null}],"robot_pose":{"timestamp":1.0,"T_parent_child":[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],"parent_frame":"odom","child_frame":"base_link","clock_domain":"synthetic"}}
```
This example is synthetic, not robot calibration. Per-detection frame/time/clock are inherited
from the row. Optional `measurement_covariance` is 3x3. `robot_pose` may be absent for already
odom inputs or explicit fixed-sensor mode. Optional `ground_truth` is a list of
`{object_id, position:[x,y,z], velocity:[vx,vy], frame_id:"odom"}`. Do not add ground truth to
unannotated recordings. The loader sorts frames by time; duplicates fail by default. Explicit
`--duplicate-policy first` retains the first input row and reports the number discarded.
Near-duplicate intervals below min_dt still fail. Missing rows use variable dt; explicit empty
rows count as misses. Invalid measurements fail visibly, not silently disappearing.

Raw Phase 2 directories (`manifest.schema=go2-recording-v1`) reuse RecordingReader,
LidarProcessor and to_lidar_message. Only `cloud_base` is accepted. Load recorded odometry
and interpolate translation/quaternion using existing PoseBuffer with bracket gaps <=.1 s.
Exact observed pose samples do not require interpolation. Missing bracketing pose is an error,
including at recording endpoints; fixed mode is a deliberate alternative for stationary data.

Raw points are levelled once using interpolated attitude for ground segmentation. Their output
frame is `base_link_level`; the subsequent transform to odom is **yaw + translation only**.
Applying the full roll/pitch again would be wrong. Optional `--window-s 0.3` reuses Phase 2's
SlidingWindowAccumulator before levelling. It removes frame-level ego motion, not per-point
scan distortion or object motion; overlapping windows create correlated measurements not
represented in the Kalman covariance. Single scans are the default for tracking.

The raw path carries real LiDAR midpoint timestamps and robot clock; it never uses RTP timing
or host time for object velocity. Recordings without calibration can still track geometric
LiDAR clusters. Camera fusion is not enabled by this work.

## BEV and performance
OpenCV renders headlessly: +x right, +y up, robot-forward x and robot-left y; configurable bounds
and image size in YAML. Robot heading, AABB footprints, IDs/status, velocity arrows (1 second
of displacement), filtered history and CV paths are shown. MOVING orange, STATIONARY green,
UNCERTAIN gray. Frames carry the coordinate label and no-global-map caveat. Fixed mode marks
the sensor origin and explicitly labels relative velocity. PNGs always work without a GUI.
`--video` uses OpenCV MJPG AVI; if unavailable it reports failure and PNGs remain. Video frame
rate is for display, not an assertion of uniform acquisition timing; timestamps are overlaid.
Output directories must be empty to avoid overwriting experiments. JSONL has track snapshots,
deleted IDs and duplicate suppression counts. `summary.json` separates tracking-only latency
from end-to-end time including perception/loading/rendering/serialization after imports.

## Existing adapter issue observed, not silently patched
Phase 2 PoseBuffer rejects an exact non-first sample when the preceding gap exceeds max_gap_s:
```python
from go2_perception.adapters.go2.window import PoseBuffer
p=PoseBuffer(max_gap_s=.1)
p.add(0,[0,0,0],[0,0,0,1]); p.add(1,[1,0,0],[0,0,0,1])
assert p.lookup(1) is None  # current Phase 2 behavior, despite an exact sample
```
Phase 3's RecordedPoses validates records and returns stored exact transforms before calling
the existing interpolator. A regression test checks the correct exact-sample behavior and
continued refusal to interpolate across gaps. Claude should decide whether to fix the shared
PoseBuffer; no hardware-owned module was modified. PoseBuffer interpolation under real robot
motion remains unverified.
