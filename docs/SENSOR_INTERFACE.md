# Sensor integration contract for Claude
**Phase 3:** tracking consumes existing results through `tracking.normalization` and explicit
`RobotPose` transforms; see [TRACKING_ARCHITECTURE](TRACKING_ARCHITECTURE.md) for timestamps,
frames, covariance, JSONL schema and the live-adapter contract. Camera calibration remains
required only for real fusion, not LiDAR-only tracking. No hardware transport code changed.

**Phase 2 update:** real topics are verified. `adapters/go2` implements this contract, with measured
values in [SENSOR_VALIDATION](SENSOR_VALIDATION.md). How real data maps to each field:

| Contract field | Real Go2 source | Status |
|---|---|---|
| `LidarMessage.points` | `rt/utlidar/cloud_base`, XYZI metres | Verified |
| `LidarMessage.frame_id` | `base_link`, or `base_link_level` when levelled with odometry attitude | Verified while stationary |
| `LidarMessage.timestamp` | Frame midpoint: `header.stamp` + half of the `time` span | Declared convention |
| `LidarMessage.clock_domain` | `go2_robot` | Not host time |
| RGB image | H.264 RTP, decoded to BGR, 1280x720 | Verified |
| RGB timestamp | First-packet receipt, clock `host_receipt` | Not capture time; latency unknown |
| `RGBMessage.K` | **None exists** | Blocked; adapter calls `RGBDetector.detect` directly |
| `Calibration` | **None exists** | Fusion blocked |

`rt/utlidar/cloud` and `rt/utlidar/cloud_deskewed` violate this contract's axes and single-scan
assumption, so `to_lidar_message` refuses them. The contract text below is unchanged from Phase 1.

## RGBMessage
`go2_perception.io.sensor_interface.RGBMessage(image, timestamp, width, height, K,
frame_id, clock_domain)`:
- image: nonempty H×W×3 NumPy uint8 **BGR**, owned/stable for the duration of inference.
  Convert SDK RGB or ROS rgb8 explicitly. Decode compressed images before calling.
- timestamp: finite float seconds at image acquisition, not callback receipt.
- width/height: actual rectified image resolution; must match image and calibration.
- K: finite 3×3 rectified intrinsics in pixel units. Keep crop/resize changes consistent.
- frame_id: exact camera optical frame name, x right, y down, z forward.
- clock_domain: nonempty identifier for a shared clock, e.g. a verified robot clock.

## LidarMessage
`LidarMessage(points, timestamp, frame_id, clock_domain)`:
- points: N×3 XYZ or N×4 XYZI; metres. Intensity is optional in column 4, retained by
  array/BIN loading but currently dropped during geometric processing. Invalid rows
  (including invalid intensity) are removed. Empty arrays have shape (0,3)/(0,4).
- timestamp: finite seconds, convention must be declared by producer (scan midpoint
  recommended). Record scan duration/per-point timing externally when available.
- frame_id: exact LiDAR frame; processing assumes x forward, y left, z up and gravity
  alignment. Transform first if SDK axes differ. Set expected_ground_z accordingly.
- clock_domain: must equal RGB clock domain after verified clock conversion.

PLY/PCD loading uses Open3D XYZ (intensity not preserved). BIN is little-endian float32
records `[x,y,z,intensity]` by default; XYZ requires explicit `bin_fields=3` / `--bin-fields 3`.
NPY uses non-pickle loading. Empty/unreadable PLY/PCD raises ValueError because Open3D
cannot reliably distinguish them; empty arrays and BIN/NPY scans are supported.

## Calibration
Supply `Calibration` with K, `(width,height)`, distortion information, frame names and
`T_lidar_camera`: a 4×4 right-handed rigid **camera-to-LiDAR** transform.
For column vectors: `p_lidar = T_lidar_camera @ p_camera_homogeneous`;
projection uses its inverse. Rotation is dimensionless, translation is metres.
Raw distorted images are deliberately rejected: rectify upstream and pass corresponding
K, `rectified=True`, and empty/zero distortion coefficients. Archive raw calibration,
distortion model and rectification procedure externally. Set `synthetic=False` only for
measured calibration. This flag alone does not certify validation.
`configs/synthetic_calibration.yaml` is a test fixture, **never Go2 calibration**.

## Processing example
```python
import numpy as np
from go2_perception.io.sensor_interface import process_rgb, process_lidar
from go2_perception.rgb.detector import RGBDetector
from go2_perception.lidar.processor import LidarProcessor
from go2_perception.fusion.association import FusionProcessor

# rgb_message, lidar_message and calibration are supplied by transport integration.
assert np.allclose(rgb_message.K, calibration.K)
rgb_result = process_rgb(RGBDetector(), rgb_message)
lidar_result = process_lidar(LidarProcessor(), lidar_message)
fused_result = FusionProcessor(sync_tolerance_s=0.05).fuse(
    rgb_result, lidar_result, calibration)
```
Construct detector/processor once and reuse. Transport must check message K against
calibration as shown; RGBResult carries resolution/frame, not K. Queue/pair observations
upstream; this library does not wait, interpolate, convert clocks or synchronize a queue.
Fusion rejects missing timestamps by default, differing named clocks, nonfinite timestamps,
frame/resolution mismatches, and absolute skew > configured tolerance. For deliberately
untimed offline pairs only, `require_timestamps=False` permits **both** timestamps absent;
one missing timestamp still fails. Never use this opt-out for live integration.

## Optional future inputs and responsibilities
Claude owns Ethernet/DDS/SDK/ROS 2 setup and verification of actual streams. Report topic
names, types, encodings, field offsets/endian, units, frame tree, clocks, rates, scan timing,
QoS and calibration availability. Do not modify perception internals to accommodate transport.
An optional ROS bridge can convert Image/CameraInfo/PointCloud2 into these messages; no ROS
package is required here. Coordinate before modifying integration files once ownership begins.

Future IMU should carry timestamp/domain, frame, acceleration (m/s²), angular velocity
(rad/s), orientation quaternion convention and covariance. Odometry/pose need parent/child
frames, metres, quaternion ordering and uncertainty. A timestamped transform tree is needed
for gravity alignment, moving-frame transforms and accumulated-window motion compensation.
These inputs are not consumed by Phase 1. No command or locomotion API is provided.
