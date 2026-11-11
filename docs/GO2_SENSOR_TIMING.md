# Go2 sensor timing (Phase 5D, 2026-11-11)

**Measured** = computed from real recordings on this Mac (`scripts/phase5d_timing_report.py`,
output `data/outputs/phase5d/timing.json`, local). **Assumed** is marked. Recordings: `mac_smoke_01`,
`test_a1`, `test_a2`, `test_b`, `test_c` (robot boot of 2026-11-07) and `p5d_static_01`,
`p5d_static_02` (boot of 2026-11-11). The robot was stationary except in `test_c` (hand-driven yaw).

## Clock domains
| Stream | Timestamp field | Clock | Semantics |
|---|---|---|---|
| `rt/utlidar/cloud`, `cloud_base` | `header.stamp` + per-point `time` (0–62 ms) | robot | stamp = scan **start** (inferred: receipt floor is 64.7 ms after odom's) |
| `rt/utlidar/cloud_deskewed` | `header.stamp` | robot | set ~10 ms after odom's floor; rolling buffer, not a scan time |
| `rt/utlidar/robot_odom` | `header.stamp` | robot | 148–149 Hz |
| `rt/utlidar/imu` | `header.stamp` | robot | 249–252 Hz |
| Camera (RTP H.264) | `host_recv_first` | **host** | first RTP packet of the frame received; the RTP 90 kHz counter runs at **0.952×** real time (all 7 recordings), so it is ordering only |

**The robot clock resets at every boot (measured).** Robot stamps on 2026-11-11 (~1 770 427 9xx s)
are *smaller* than on 2026-11-07 (~1 770 430 xxx s), and host − robot jumped by 336 141.5 s
(3.89 days) between the two boots. Robot stamps are therefore only comparable within one boot;
`Go2VLNObservation.clock_epoch` records the per-recording offset for this reason.

## Host − robot offset
macOS cannot use the Linux ICMP-timestamp method (`ping` timestamp option unsupported), so the offset is
estimated per recording as the **lower envelope of `robot_odom` (host_recv − stamp)**. That is the
offset plus odom's minimum transport latency (on Linux, ICMP measured odom latency 1.6 ms median), so
it is biased high by roughly 1–2 ms; we carry σ = 2 ms.

| Recording | Offset (s) | Odom above floor median / p95 (ms) | Drift host vs robot (ppm) |
|---|---|---|---|
| mac_smoke_01 | 23 654 671.9201 | 0.4 / 4.6 | 18.0 |
| test_a1 | 23 654 671.9427 | 0.8 / 4.7 | 11.9 |
| test_a2 | 23 654 671.9492 | 0.8 / 3.6 | 13.0 |
| test_b | 23 654 671.9545 | 0.8 / 4.0 | 11.9 |
| test_c | 23 654 671.9588 | 0.9 / 3.7 | 11.6 |
| p5d_static_01 | 23 990 813.4818 | 0.3 / 90.7 (stall) | 7.1 |
| p5d_static_02 | 23 990 813.4817 | 0.3 / 1.7 | 1.0 |

Within the 2026-11-07 boot the offset rose 38.7 ms over 3 731 s (≈ 10 ppm), consistent with the
in-recording drift. So the offset must be re-estimated per recording (or continuously, live), never
copied between sessions. Some recordings contain one host-side stall of 0.26–0.38 s that delays all
streams together (start of `mac_smoke_01`, `p5d_static_01`, `test_a1`); it is a receipt artefact, not
lost data (0 non-increasing stamps, 0 RTP loss).

## Camera latency (measured once)
Method: phase-correlation horizontal image motion between consecutive frames vs odometry yaw rate,
cross-correlated over −300…+300 ms lags (`test_c`, robot yawed ~±30° by the operator, 1 118 frames).
| Quantity | Value |
|---|---|
| Lag of camera host receipt behind odometry stamp (mapped by the odom floor) | **48 ms** |
| Peak \|correlation\| | 0.979 |
| First / second half of the recording | 50 ms / 36 ms |
| Lags within 1 % of the peak correlation | 26–58 ms |

Interpretation: exposure (mid-frame) → host receipt ≈ 48 ms + odom's minimum latency (~1–2 ms).
Used as `camera_latency_s: {value: 0.048, sigma: 0.016}` in `configs/go2_vln_sensors.yaml`. Limits:
one recording, one boot, rotation only (no translation), and frames are 70 ms apart, so the
estimate is quantised; rolling shutter and exposure time are not separated. **Re-measure** on
the next boot with a deliberate yaw sweep (operator-driven, coordinated with the user).
The robot does not publish exposure timestamps (no `CameraInfo`, `rt/frontvideostream` silent).

## Receipt behaviour
| Stream | Rate | Receipt interval median / p95 / max (ms) |
|---|---|---|
| RGB | 14.28–14.30 Hz | 69.7–70.1 / 75.6–77.0 / 82.6–90.6 |
| cloud_base | 15.4 Hz | scan every 64.95 ms by stamp; 0 non-increasing stamps |
| cloud_base receipt − scan start (above odom floor) | | median 64.7 ms ≈ 62 ms scan + ~3 ms |

The camera stream has no B-frames (`has_b_frames=0`, 1 reference frame), so the decoder adds no
reorder delay; frame assembly from RTP packets takes 0.27 ms median, 7.7 ms max.

## Nearest timestamp is not synchronization
Earlier phases paired camera and LiDAR by nearest host receipt: median 17 ms, p95 31 ms, max 33–53
ms apart. That pairs a camera frame exposed ~48 ms earlier with a LiDAR scan whose *start* was ~65 ms
before its receipt, and it ignores the scan's 62 ms duration. The VLN interface instead converts each
camera frame to an **estimated robot-clock exposure time**
`t_robot = host_recv_first − offset − camera_latency` (σ ≈ 16 ms) and selects the latest frame at or
before the LiDAR scan midpoint (causal). Residual error: σ ≈ 16 ms latency + 2 ms offset + half a
camera frame (35 ms) of selection quantisation. At 0.5 m/s that is ~2.5 cm of motion per 50 ms;
at 1 rad/s yaw it is ~3° per 50 ms, so do not fuse pixel-accurate geometry during fast turns.

## What remains unverified
- Camera latency on another boot, under translation, and its variance over time.
- Odometry twist frame (the interface labels it `base_link` per the nav_msgs convention, unverified).
- Odometry accuracy while walking.
- Whether the DDS stamp of `cloud` is scan start (inferred from the 64.7 ms receipt floor and the
  62 ms per-point time span, not documented by Unitree).
