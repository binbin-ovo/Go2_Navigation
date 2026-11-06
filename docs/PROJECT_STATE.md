# Project state — 2026-11-06
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
- Linux/CUDA and live Go2 streams not tested; no actual topic names or sensor parameters known.
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
