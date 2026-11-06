# Project state — 2026-11-06

## Phase 3 — temporal tracking (current branch: `phase3-tracking`)
Implemented on integrated baseline `dc0f85b`; not merged into main. Adds validated timestamped
3D detections, odometry/fixed-sensor normalization, planar CV Kalman tracking, Hungarian gating,
persistent IDs/lifecycle, velocity uncertainty, motion classification, 3-second trajectories,
raw Go2/detection replay, headless BEV/video and synthetic evaluation. No SDK communication or
existing perception code changed; no new runtime dependencies. Phase 2 uses DDS/RTP directly,
not a mandatory ROS 2 adapter.

Verified: baseline 70 passed/1 skipped; expanded suite **127 passed/1 hardware skipped**. All
8 reference scenarios run with zero ID switches, XY MAE .0181–.0598 m and velocity MAE
.0496–.1164 m/s. Additional 40 noisy-seed sequences have four ID switches; this remains a known
baseline limitation. A single real cloud produced four tentative uncertain tracks in explicitly
fixed-sensor mode; real temporal accuracy and Phase 3 live hardware behavior remain unverified.
Real fusion still requires calibration and verified time alignment. Covariance assumes exact
poses; sparse/merged/partially observed clusters can fragment IDs. No locomotion or planning.

Read [TRACKING_ARCHITECTURE](TRACKING_ARCHITECTURE.md), [TRACKING_EVALUATION](TRACKING_EVALUATION.md),
and the Phase 3 handoff below/at AGENT_HANDOFF.md. Next: labelled temporal Go2 replay, motion/pose
validation, calibration and synchronization, then replaceable learned prediction experiments.
All Phase 1/2 sections below are historical records and retain their original validation scope.

## Main integration
Local merge verification: **70 passed, 1 hardware test skipped** (macOS CPU).
`main` combines Phase 1 (`359fea8`) and Phase 2 (`72585e0`). Phase 2 descends directly
from Phase 1, so both histories and all source changes are preserved without conflict.
The merge adds offline recording-to-CLI regression tests and reconciles status/setup docs;
Claude-owned adapter source, hardware scripts and Go2 configurations are unchanged.
See [MERGE_VALIDATION](MERGE_VALIDATION.md) for local verification, separate from the
hardware-workstation results below. Real calibrated fusion remains blocked.

## Latest verified state: Phase 2 hardware integration
Branch `feature/go2-sensor-integration`, based on `codex/phase1-perception`. **Read-only** access to a
real Go2 Pro over Ethernet works; no movement or control functionality exists.

**Verified on the real robot (stationary):** DDS and multicast streams; RGB 1280x720 at 14.3 Hz;
LiDAR single frames at 15.4 Hz; IMU 251 Hz; odometry 150 Hz; recording and offline replay; unmodified
YOLO and LiDAR processors on real data; live inference at about 13 Hz RGB and 14 Hz LiDAR; correct
person detection in one scene. Linux/CUDA is verified. Tests: **68 passed, 1 hardware test skipped
offline, and that test passes live.**

**Blocked:** RGB-LiDAR fusion, for lack of camera calibration and verified synchronization.
**Unverified:** link-drop recovery, all behaviour under robot motion, LiDAR model (probably L2;
inferred only), detection accuracy against labels, runs longer than 106 s.

**Facts a new agent must not miss:**
- `rt/utlidar/cloud_deskewed` is a rolling 10,000-point buffer plus the newest frame, not a scan.
- One `cloud_base` frame covers only about 190 deg of azimuth; use the 0.3 s compensated window.
- `cloud_base` contains nothing within 0.3 m XY and is heavily thinned at close range.
- The robot clock is about 273 days behind the host; RTP timestamps are not capture times.

Evidence and commands: [GO2_CONNECTION](GO2_CONNECTION.md), [SENSOR_VALIDATION](SENSOR_VALIDATION.md),
[INTEGRATION_REPORT](INTEGRATION_REPORT.md). Next steps: [AGENT_HANDOFF](AGENT_HANDOFF.md).

The Phase 1 record below is historical. Phase 2 supersedes its live-access, Linux/CUDA,
and temporal-window limitations; its old next-task list is not the current backlog.

# Phase 1 record
## Phase
Phase 1 offline foundation implemented on branch `codex/phase1-perception`.
Synthetic validation and separate real RGB/cloud smoke tests completed. Live Go2 and
real calibrated sensor fusion **not validated**. No movement/control functionality exists.

## Completed and verified
- Installable src-layout Python package and pinned direct dependencies; fresh macOS ARM64
  Python 3.12.11 environment installation succeeded.
- COCO YOLOv8n CPU detection on a real bus image: 6 detections, JSON + annotated JPEG.
- Cloud loaders, finite/range filtering, Open3D voxel reduction, constrained seeded RANSAC,
  DBSCAN, observed AABBs, headless XY views, colored PLY and machine-readable results.
- Synthetic two-obstacle scene: 2 clusters; real PCL cloud processed (1 broad cluster;
  its segmentation quality is not validated).
- Rectified projection, explicit camera-to-LiDAR transform, one-to-one association,
  clock/frame/resolution/skew validation. Synthetic demo: 2 associations.
- Transport-independent RGBMessage/LidarMessage API; ROS boundary documentation.
- Automated suite: **44 passed in 3.37s** after ground extrapolation regression fix.
  Final Ruff lint and dependency compatibility checks also passed. Full commands/provenance: [EXPERIMENTS](EXPERIMENTS.md).

## Dependency versions
Python 3.12.11 tested; supported range 3.10–3.12. torch 2.6.0; torchvision 0.21.0;
ultralytics 8.3.100; checkpoint yolov8n.pt from official assets v8.3.0; numpy 1.26.4;
open3d 0.19.0; scipy 1.15.2; opencv-python 4.11.0.86; PyYAML 6.0.2; pytest 8.3.5.
See [pyproject](../pyproject.toml), [macOS environment lock](../requirements-macos-arm64.lock)
and [model hash](EXPERIMENTS.md). No training was performed.

## Unverified and limitations
- *(Superseded by Phase 2: Linux/CUDA, live streams and topic names are now verified.)* Linux/CUDA and live Go2 streams not tested; no actual topic names or sensor parameters known.
- Real calibration unavailable; example intrinsics/extrinsics are synthetic only.
- Raw distorted images are rejected; upstream rectification and matching K required.
- Ground requires gravity-aligned coordinates and known approximate height; seed points
  must be in the expected height band. Steep or elevated terrain may remain unclassified
  or cluster as obstacles. No terrain-adaptive model or IMU compensation yet.
- DBSCAN can merge close objects or reject sparse objects; tune radius/support per data.
- Fixed geometric association does not resolve occlusion/merged clusters; no tracking.
- No scan de-skew, temporal accumulation implementation, clock conversion or synchronization queue.
- PLY/PCD intensity is not preserved; preprocessing discards intensity. Empty PLY/PCD is
  rejected, while empty arrays/NPY/BIN work. JSON includes processed points and may be large.
- Local Python runtime is under /private/tmp; recreate the venv if removed. Nonfatal
  Fontconfig cache warning occurred during first YOLO import.
- No outstanding failing test; no perception accuracy or real-time guarantee claimed.

## Decisions and next tasks
Use existing repository root, lazy heavy imports, dataclass outputs and pure transport
boundary; no ROS placeholder subscriber. Calibration never defaults to invented Go2 data.
Ground support guard was added after a real regression test failure.

1. Claude verifies Linux hardware access and records message fields, rates, clocks and frames.
2. Capture synchronized RGB/XYZ and measured calibration; archive provenance.
3. Add a recorded Go2 replay fixture and verify overlay alignment, skew, gravity and units.
4. Tune ground/clustering against annotated sparse Go2 scenes and add integration regression tests.
5. Verify Linux/CUDA packaging/performance before considering any later navigation phase.

Read [ARCHITECTURE](ARCHITECTURE.md), [SENSOR_INTERFACE](SENSOR_INTERFACE.md),
[SETUP](SETUP.md), and [AGENT_HANDOFF](AGENT_HANDOFF.md). Update this file and handoff
whenever significant implementation or validation status changes.
