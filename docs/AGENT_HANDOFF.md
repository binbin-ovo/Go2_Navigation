# Agent handoff — 2026-11-06 (Phase 2 update)
## What Codex should do next
Claude's Phase 2 is on `feature/go2-sensor-integration`. Read [PROJECT_STATE](PROJECT_STATE.md), then
[SENSOR_VALIDATION](SENSOR_VALIDATION.md). Your prioritized tasks are in
[INTEGRATION_REPORT](INTEGRATION_REPORT.md): replay regression test, `RGBMessage.K` decision, cluster
tuning on real frames, odometry-based ground height, false-positive handling, then tracking.

**Do not** use `rt/utlidar/cloud_deskewed` as a scan, or use RTP timestamps as capture times. **Do not**
use synthetic calibration on real data, or add any publisher, RPC client or motion command to
`adapters/go2`. Claude keeps ownership of `adapters/go2`, `scripts/go2_*.py`, `configs/go2_*.yaml`, and
hardware validation; coordinate before changing them. Perception modules remain yours, unchanged.

Recordings are local-only and contain private home images. They are not on GitHub. Without the
capture workstation, record your own with `scripts/go2_record.py`, or use
`tests/fixtures/go2/real_cloud_pair.npz` (one real LiDAR frame pair).

Start here on the workstation:
```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/go2_replay.py --recording data/recordings/stage_c_person --window-s 0.3 --output data/outputs/replay
```
Hardware work still pending from Claude and the user: link-drop test, calibration, motion validation.

# Phase 1 handoff record (Codex to Claude)
## Purpose and ownership
Build an offline RGB + LiDAR perception foundation for future Go2 dynamic navigation.
Codex implemented repository architecture, detectors/processors, geometric fusion, CLIs,
synthetic fixtures, automated tests and documentation. Claude independently owns Ethernet,
DDS/SDK/ROS 2 connection, verified RGB/LiDAR/IMU/odometry streams and live validation.
No Claude-owned integration files existed when this repository was initialized. Coordinate
before changing such files once integration begins. No robot commands belong in Phase 1.

## Current repository
Branch `codex/phase1-perception`; implementation in src/go2_perception, CLI scripts in scripts,
YAML in configs, tests in tests. Everything operates without hardware. Real calibration is
absent; synthetic calibration is conspicuously marked. No ROS runtime or fabricated topics.
Results: 44 unit tests passed (final run 3.37s); lint and dependency compatibility passed; CPU YOLO processed a real image (6 detections); synthetic
LiDAR found 2 clusters; synthetic fusion matched 2; public PCL cloud processed but default
segmentation yielded one broad cluster/no ground. No real paired fusion or live validation.
See PROJECT_STATE.md and EXPERIMENTS.md for exact versions, sources, hashes and measurements.

## Install/run
From repository root:
```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
pytest -q
python scripts/generate_test_data.py
python scripts/run_lidar.py --input data/sample/test.pcd --output data/outputs/lidar
python scripts/run_rgb.py --input data/sample/bus.jpg --output data/outputs/rgb
python scripts/run_fusion.py --image data/sample/synthetic.jpg --points data/sample/test.pcd --detections data/sample/synthetic_detections.json --calibration configs/synthetic_calibration.yaml --rgb-timestamp 1 --lidar-timestamp 1 --clock-domain synthetic --output data/outputs/fusion
```
Download bus.jpg / yolov8n.pt with exact commands in EXPERIMENTS.md. Outputs and inputs are
ignored by Git and must be generated/downloaded on a new checkout. For macOS ARM64 exact
transitive versions, use `uv pip install -r requirements-macos-arm64.lock`. Linux/CUDA has
not been tested. Existing local .venv uses a runtime under /private/tmp; recreate if broken.

## Integration steps for Claude
1. Read SENSOR_INTERFACE.md in full, then core/types.py, io/sensor_interface.py and
   fusion/calibration.py. ARCHITECTURE.md describes every public entry point.
2. Verify actual SDK/ROS topics, data layout/endian, units, BGR/RGB encoding, timestamps,
   clock domains, frame axes, scan duration and calibration source. Document evidence.
3. Rectify images, provide corresponding K, and measure camera-to-LiDAR extrinsics
   (`T_lidar_camera`, NOT its inverse). Use explicit measured frame names/resolution.
4. Convert observations to RGBMessage/LidarMessage; call process_rgb/process_lidar with
   persistent detector/processor objects, then FusionProcessor.fuse. Check message K
   matches calibration. Pair timestamps in a verified common clock upstream.
5. Align scan +Z with gravity, configure expected ground height, and tune sparse-cloud
   filtering/DBSCAN. Motion-compensate accumulated windows before passing them to core.
6. Record an offline Go2 pair and run overlays before any live loop. Add integration tests
   for this evidence and update PROJECT_STATE.md, AGENT_HANDOFF.md and EXPERIMENTS.md.

## Open questions and issues
Actual topic names, QoS, sensor clock conversions, frame tree, optical distortion model,
intrinsics/extrinsics, mounting height and per-point timing are unknown. Determine whether
streams provide raw or already transformed points/images. No live synchronization queue,
clock conversion, de-skew or tracking is implemented. Nonzero distortion raises ValueError.
PLY/PCD intensity is not retained; cluster boxes describe only observed points.

No failing unit tests remain. Regression reproduction:
`pytest -q tests/test_lidar.py` covers nearby elevated obstacles incorrectly treated as ground
by the initial implementation; the height-band support fix now passes. Parameter sensitivity
is intentionally demonstrated by the nearby-cluster test (large eps merges them).
`python scripts/run_lidar.py --input data/sample/table_scene_lms400.pcd --output data/outputs/lidar_real`
reproduces the public scene's one broad cluster/no ground with defaults; do not call this
validated segmentation. First YOLO import may report a nonfatal font-cache warning.

The next milestone is **recorded, synchronized Go2 RGB–LiDAR replay with measured calibration**,
followed by live read-only inference. Autonomous locomotion, learned forecasting and planning
remain out of scope.
