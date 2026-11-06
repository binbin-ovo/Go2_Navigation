# Go2 sensor validation (2026-11-06)
**Measured** means computed from messages received from the real robot in this session.
**Inferred** means reasoned from measurements. Recordings are local and gitignored, so the
paths below exist only on the capture workstation.

The robot stood still throughout (odometry moved about 1 mm in the static run). **Nothing here
validates behaviour while the robot walks.**

## Recordings
| Name | Duration | Contents | Scene |
|---|---|---|---|
| `data/recordings/stage_b_01` | 20 s | RGB 274; cloud, cloud_base, cloud_deskewed 309 each; IMU, odom, sport | Static room |
| `data/recordings/stage_b_02` | 12 s | Decoder-fix check | Static room |
| `data/recordings/stage_b_dynamic` | 106 s | RGB 1502; cloud, cloud_base 1636 each | Static: no person ever entered view |
| `data/recordings/stage_c_person` | 74 s | RGB 1053; cloud_base 1141 | Static about 35 s, then a person walks in view |

Camera recordings show a private home and are not to be committed. No rosbag exists because ROS 2
is not installed; the format is in `adapters/go2/recording.py`.

## RGB
| Property | Measured |
|---|---|
| Transport | H.264 RTP multicast, source 192.168.123.161, no RTCP on port 1721 |
| Resolution | 1280x720, decoded to BGR uint8 |
| Frame rate | **14.28-14.29 Hz** (receipt count in all timed recordings) |
| Keyframe interval | 2.0 s, RTP clock; receipt time is about 5% longer |
| Packet loss | **0** of 14,307 RTP packets (106 s run); 0 reordered |
| Frame gap at host | median 70 ms, p99 79 ms, max 90 ms (106 s run) |
| Decoder errors | 0 after the mid-GOP fix; up to 2 s of frames are skipped until the first IDR |
| Timestamps | **RTP only, and not capture times** (below) |
| Frame identifier | None published. `go2_front_camera_unrectified` is our own label |
| Distortion | Visibly wide-angle and distorted; no CameraInfo is published |

**RTP timestamps do not measure time.** Every frame advances exactly 6000 ticks (66.667 ms at
90 kHz), yet frames arrive about every 69.7 ms: 18.20 s of RTP time spanned 19.11 s of receipt time
(ratio 1.050). The counter is a nominal 15 fps, so it gives ordering only. Use `host_recv_*` as the
timing reference. True capture latency is **unknown**, since it has never been measured.

## LiDAR model
**Most probably a Unitree L2 (inferred, not confirmed).** `rt/utlidar/lidar_state` is advertised but
delivered 0 messages, so the sensor does not self-report here.
| Quantity | Measured | L2 published | L1 published |
|---|---|---|---|
| Vertical sweep | **216.3 Hz** (FFT of elevation) | 216 Hz | 180 Hz |
| Sample spacing | **7.6-7.9 us**, about 129k samples/s | about 128k/s (reseller figure) | 43.2k/s (reseller figure) |
| Delivered points | **about 59k-61k pts/s** | 64k pts/s effective | below measured value |
| Azimuth rotation | **about 4.4-4.5 Hz** (crude) | 5.55 Hz | not confirmed in search |

The vertical rate and the sampling rate agree with L2 and rule out the L1 figures found. The azimuth
estimate is below the L2 figure and remains an **open discrepancy**, though the estimate is crude.
Reseller pages disagree on which LiDAR ships with the Go2 Pro. To confirm, read the sensor's
label or the purchase documents.
Sources: [Unitree L2 manual](https://oss-global-cdn.unitree.com/static/Unitree%204D%20LiDAR%20L2%20User%20Manual.pdf),
[reseller L2 specs](https://uk.robotshop.com/products/unitree-4d-lidar-l2),
[reseller Go2 Pro variants](https://openelab.io/fr/products/unitree-go2-pro-quadruped-robot).

## LiDAR topics
All three are `sensor_msgs::msg::dds_::PointCloud2_`, Reliable writers, little-endian, `height=1`,
`point_step=32`, `is_dense=true`, 0 stamp gaps above threshold and 0 non-increasing stamps in the
recordings. The first probe, which subscribed to about 20 topics at once in Python, saw isolated stamp gaps
up to 259 ms on `cloud_deskewed`; the cause is unknown, so keep subscriptions minimal.
| Topic | Frame | Measured rate | Points/msg | Meaning |
|---|---|---|---|---|
| `rt/utlidar/cloud` | `utlidar_lidar` | 15.41-15.43 Hz | 3873-4128 (mean 3992) | **Individual scan frame**, raw sensor frame; 0% consecutive overlap |
| `rt/utlidar/cloud_base` | `base_link` | 15.41-15.43 Hz | 1117-1340 (mean 1232) | **Individual frame**, thinned subset of `cloud` in body axes |
| `rt/utlidar/cloud_deskewed` | `odom` | 15.43 Hz | 11117-11340 (mean 11232) | **Rolling accumulation; 89% consecutive overlap** |

**The 15.4 Hz rate of `cloud_deskewed` is not new-scene density.** Each message is the prior 10,000
points followed by the newest `cloud_base` frame at index 10000 onward. Measured directly: 1166
new points of 11166, at indices 10000-11165.

Field table, in bytes: `x` 0, `y` 4, `z` 8, `intensity` 16 (float32); `ring` 20 (uint16); `time` 24
(float32). `cloud_deskewed` has no `ring` or `time`.
- `ring` is always 1, so it is not a useful channel on this robot.
- `intensity` ranged 23-230 in the sampled frame. Its units are undocumented.
- `time` is seconds since `header.stamp`, 0 to about 0.062 s. It is usually sorted, but small
  reversals up to 1.9 ms occur.
- `header.stamp` is the **frame start** (inferred): frames span 62.3 ms and arrive 69 ms after it.

### Spatial coverage (measured)
A single frame sees only part of the horizon, because of the scan pattern and thinning.
| Frames (about 65 ms each) | `cloud_base` azimuth coverage at r>0.5 m, median (min) |
|---|---|
| 1 | **190 deg (150)** |
| 2 | 280 deg (260) |
| 3 | 290 deg (260) |
| 5 | 300 deg (280) |
| 15 | 310 deg (300) |

In raw `cloud`, one frame covers about 280 deg in the sensor frame and two cover 360 deg. Remaining
gaps may be real occlusion or an unobserved region; ground truth is unavailable.

### What `cloud_base` removes (measured)
Share of raw points kept, by XY distance in `base_link`:
| <0.3 m | 0.3-0.4 | 0.4-0.6 | 0.6-0.8 | 0.8-1 | 1-1.5 | 1.5-2 | 2-3 | 3-5 |
|---|---|---|---|---|---|---|---|---|
| **0%** | 13% | 27% | 38% | 57% | 71% | 84% | 98% | 100% |

So **nothing within 0.3 m XY exists in `cloud_base`**, and the near field is strongly thinned.
Close-range obstacle detection would need `cloud` plus a body self-filter, which is not implemented.

## Frames and transforms
- No TF is published. The frame names above come from message headers.
- `cloud_base = T_base_lidar * cloud`: residual below 1e-6 m for all 320 stationary message pairs,
  with per-element standard deviation about 1e-8. The value is in `configs/go2_sensors.yaml`. It was
  **recovered from firmware output**, not independently surveyed, and only while stationary.
- Convention check: odometry applied to `cloud_base` reproduces the new part of `cloud_deskewed`
  with median 0.0 mm and p95 0.1 mm error over 30 checks. This confirms the quaternion order and
  direction (`T_odom_base`, xyzw). The robot was stationary, so it says **nothing about odometry
  accuracy under motion**.
- Ground in `base_link` is at about z = -0.33 m with the robot standing; odometry z = 0.31 m.

## Clocks
| Clock | Used by | Relation to host |
|---|---|---|
| Robot | All DDS stamps | Host minus robot = **23,568,393.043-.047 s** (about 273 days) |
| RTP counter | Camera | Arbitrary origin, nominal rate; unrelated to real time (above) |
| Host | Our receipt times | `time.time()`; not synchronized to the robot |

The offset comes from ICMP timestamps (1 ms resolution, mod 24 h), with whole days resolved from
message receipt. It rose from .043 to .047 s over about 9 minutes, close to the resolution. **Re-measure
rather than treating it as constant.**

## Other streams (20 s and 106 s runs)
| Topic | Rate | Latency: receipt minus stamp, median (max) | Notes |
|---|---|---|---|
| `rt/utlidar/cloud` | 15.4 Hz | 69 ms (75) | Includes about 62 ms of scan, since stamp is frame start |
| `rt/utlidar/cloud_base` | 15.4 Hz | 70 ms (76) | |
| `rt/utlidar/cloud_deskewed` | 15.4 Hz | 17.5 ms (34) | Stamp evidently set later |
| `rt/utlidar/imu` | 251 Hz | 1.6 ms (16) | Frame `utlidar_imu`; mean accel norm 10.43 m/s^2 at rest, so biased or uncalibrated |
| `rt/utlidar/robot_odom` | 149-150 Hz | 1.6 ms (14) | `odom` to `base_link`, with twist |
| `rt/utlidar/robot_pose` | 19 Hz | not separately reported | `odom` |
| `rt/lf/sportmodestate` | 20 Hz | 3.7 ms (17) | Mode 0, error_code 100, meaning not looked up |
| `rt/sportmodestate`, `rt/lowstate` | about 294, 499 Hz | not reported | Heavy in Python; not recorded |

Latency depends on the ICMP-derived offset, so treat it as +/- about 1 ms. Odometry showed about 1.2%
and IMU 0.3% estimated missed messages by stamp gaps. Whether the robot or the SDK's queue caused
those is **unknown**. Point-cloud topics and RTP showed none.

## Resource use (measured, 50 s runs unless noted)
- Recorder with camera and one LiDAR topic: RSS flat at 109-114 MB, no leak trend.
- Recorder without camera, three LiDAR topics: RSS flat at 91-97 MB.
- CPU: 24-37% of one core while recording.
- Disk: about 3-7 MB/s depending on topics.
- A reported `max_rss_mb` of 507.5 in `stage_b_dynamic` and `stage_c_person` is probably inherited
  across shell `exec`, since exactly those two runs used it and share an identical value.
- Live perception: 348% of one core (Open3D and YOLO threads), 13 Hz RGB and 14 Hz LiDAR results.

## Window study (`scripts/go2_window_study.py`, `go2_moving_object_study.py`)
`stage_b_01`, static scene, default processor settings from `configs/go2_lidar.yaml`:
| Window | Points | 10 cm voxels | Clusters median (std) | Ground points | Processor |
|---|---|---|---|---|---|
| single frame | 1,227 | 410 | 7 (1.91) | 466 | 20 ms |
| 0.13 s | 3,673 | 921 | 11 (1.84) | 1,066 | 22 ms |
| **0.3 s** | 6,159 | 1,274 | 9 (**1.03**) | 1,506 | 24 ms |
| 0.65 s | 13,561 | 1,662 | 6 (1.13) | 2,119 | 49 ms |
| 1 s | 19,715 | 1,850 | 5 (1.30) | 2,410 | 57 ms |
| 2 s | 38,217 | 2,260 | 4 (1.19) | 3,066 | 82 ms |

`stage_c_person`, foreground outside a static background, 5th-95th percentile spread:
| Window | single | 0.13 s | **0.3 s** | 0.65 s | 1 s | 2 s |
|---|---|---|---|---|---|---|
| Foreground points | 37 | 61 | 115 | 251 | 370 | 704 |
| X spread | 0.24 m | 0.30 m | **0.37 m** | 0.54 m | 0.69 m | 1.14 m |

**Recommendation: 0.3 s.** It has the lowest cluster-count variance among those tested, about 290
deg coverage, and three times the foreground points of a single frame, with moderate smear. Fewer
clusters at long windows may mean real merging, not improvement; **no ground truth was
available, so this is stability and coverage evidence, not accuracy**. Foreground may also include a
door that appears to move in the images, so these spreads are approximate. The centroid-speed estimate
from single frames was too noisy to use.

The window requires a bracketing pose, refuses frames without one, leaves raw frames untouched,
and never writes a map. It compensates ego-motion per frame, not per point, using odometry that is
**untested under motion**.

## Perception results
`stage_c_person`, unmodified `RGBDetector` (yolov8n, GPU) and `LidarProcessor`:
- **Person:** 274 frames, all during the walk; median confidence 0.89. Visually confirmed correct in
  sampled frames, including close-range legs (0.52).
- **False positives seen:** `refrigerator` on door and drawers (often about 0.27-0.6), and a 0.29
  `person` box on a blurred frame. Classes `surfboard` (9), `bottle` (2) and `chair` (1) were not
  inspected.
- **Bed:** correct box at about 0.85. No formal accuracy measure exists because there are no labels.
- Inference: 8.2-8.5 ms median, 1.4-1.5 s first call.
- **LiDAR:** ground found at the right height; the largest cluster agrees with the bed's direction.
  1141/1141 frames processed. Clusters: single frame median 7 (range 1-14); 0.3 s median 9 (range
  4-15). Processor median 18 ms and 23 ms.
- Live: `go2_live.py` ran 30 s at 12.9 Hz RGB, 13.9 Hz LiDAR, 0 stale warnings.

## Calibration status
| Item | Status |
|---|---|
| Camera intrinsics and distortion | **Missing** |
| Rectified images | **Missing**; `Calibration` rejects unrectified input by design |
| `T_lidar_camera` | **Missing** |
| LiDAR to base | Available from firmware; not independently validated |
| Camera/LiDAR synchronization | **Unverified**: capture latency unknown |

Fusion is **blocked and was not attempted**. No calibration was invented, and
`configs/synthetic_calibration.yaml` was not used on real data.

## Tests performed
| Test | Result |
|---|---|
| Phase 1 suite on Linux, before changes | 44 passed |
| Full suite after changes (offline) | 68 passed, 1 skipped (hardware) |
| `GO2_HARDWARE=1` live test | 1 passed |
| Phase 1 `run_lidar.py` / `run_rgb.py` on real files | Worked |
| SIGTERM clean stop during recording | Worked; exit 0, statistics saved |
| Silent stream: camera rejoin | Passed, offline only |

## Verified versus unverified
**Verified:** connection, DDS discovery and data, RGB and LiDAR capture, replay, YOLO and LiDAR
processors on real data, live inference, person detection in one scene, clean shutdown.

**Unverified:** physical link-drop recovery; any behaviour while the robot moves; odometry accuracy;
windowing under ego-motion; camera capture latency; LiDAR model identity; detection accuracy
against labels; fusion; long-duration runs (longest here is 106 s); performance outdoors or in
other scenes.
