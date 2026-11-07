# Agent handoff — 2026-11-07 (macOS update)

## macOS validation handoff (current)
Branch `feature/macos-go2-phase3-validation`; do not merge without user approval. The Mac reads the
Go2 natively; setup, rates and pitfalls are in [MAC_CONNECTION](MAC_CONNECTION.md). Use
`.venv-go2` for hardware and `--iface en9` (the config default is Linux `eno2`). Real tracking
results and gaps: [PHASE3_REAL_VALIDATION](PHASE3_REAL_VALIDATION.md).

```bash
.venv-go2/bin/python scripts/go2_record.py --iface en9 --seconds 60 --name test_a
.venv-go2/bin/python scripts/run_tracking.py --input data/recordings/test_a --output data/outputs/mac/test_a --window-s 0.3 --video
.venv-go2/bin/python -m pytest -q
```
Pose gating: `configs/tracking.yaml` `pose_alignment`; skipped frames in `skipped_frames.jsonl`.

**Next for Codex:** (1) reduce false motion on static scenes: cluster split/merge on walls
(98 IDs in 10 s, 46 ever MOVING) — e.g. extent/persistence filtering or association on
geometry, tuned on labelled Go2 sequences, without breaking synthetic metrics; (2) evaluate Tests
B and D once recorded; (3) consider velocity-variance-aware motion gating. **Next for Claude/user:**
Test A with a measured object, Tests B and D, odometry under human-controlled motion (Test C), Mac
interface override for `go2_live.py`/hardware test, camera calibration.

## Phase 3 handoff (2026-11-06)

Changes are committed on `phase3-tracking` from main `dc0f85b`; do not merge without user approval.
New packages: tracking, localization and visualization; core/tracking_types.py; three tracking
scripts and configs/tracking.yaml. All original modules/tests remain intact. Validation:
127 passed/1 hardware skipped. Reference synthetic IDs are stable; additional noisy seeds have
four switches across forty sequences. No real temporal or live tracking accuracy is claimed.

```bash
source .venv/bin/activate
uv pip install -r requirements.txt
python scripts/generate_tracking_test_data.py
python scripts/run_tracking.py --input data/sample/tracking_sequence/crossing --config configs/tracking.yaml --output data/outputs/tracking_run --video
python scripts/evaluate_tracking.py --input data/sample/tracking_sequence --output data/outputs/tracking_eval.json
pytest -q
# On the recording workstation, with real bracketed odometry:
python scripts/run_tracking.py --input data/recordings/stage_c_person --output data/outputs/tracking_person --window-s 0.3
```
Choose fresh output directories. *(2026-11-07: unposed frames are now skipped with a reason and
reported; a recording with no posable frame still raises. See TRACKING_ARCHITECTURE.md.)*
Do not substitute identity poses. Use `--mode fixed_sensor` only with an explicit fixed-sensor
assumption, and interpret velocities as sensor-relative. Full JSONL input, types, covariance,
frame conventions, config meanings and output contracts: TRACKING_ARCHITECTURE.md.

**Claude's exact next steps:**
1. Keep adapter/hardware ownership. Record cloud_base and odometry with real acquisition stamps;
   provide `go2_robot` clock and `T_odom_base` (xyzw source quaternion). No host/RTP substitution.
2. For live normalized detections, call `from_lidar` / `from_fusion` then the persistent
   MultiObjectTracker. Supply a RobotPose mapping the **actual detection frame** to odom at
   that timestamp. If detections were levelled to base_link_level, use yaw+translation, not
   a second roll/pitch rotation. Buffer/interpolate odometry upstream with validated gaps.
3. Validate stationary world objects while translating/rotating the robot, then known moving
   targets. Measure velocity error and ID continuity against annotated ground truth.
4. Review the documented exact-sample-after-gap PoseBuffer issue. Phase 3 handles exact observed
   samples in its replay wrapper; shared hardware code was not silently patched.
5. Validate dropouts, clock discontinuities, scan de-skew assumptions, odometry drift and dynamic
   ground height. Ground/clustering defaults still depend on standing height/scene density.
6. Complete camera intrinsics, distortion/rectification, extrinsics and capture-time alignment
   before associating RGB semantics with LiDAR trajectories. No synthetic calibration on real data.

**Phase 4 interface:** consume TrackOutput history, timestamp/frame, velocity/covariance/status;
replace `predict_trajectory` with a learned predictor returning timestamped positions and
uncertainty in the same frame. Establish annotated metrics and robustness first. Robot control,
learned networks, planners and navigation are not implemented by Phase 3.

Historical handoffs below describe their original phases, not the current task status.

## Main integration handoff
Local merge verification: **70 passed, 1 hardware test skipped** (macOS CPU).
Work from `main`, which contains both phase commits plus the merge validation tests/docs.
`tests/test_go2_replay.py` now exercises the real committed cloud through recording and
the replay CLI, with and without a window. Its stationary identity poses are synthetic
test scaffolding, not measured motion evidence. See [MERGE_VALIDATION](MERGE_VALIDATION.md).
Adapter/hardware ownership remains unchanged. Current priority: measured calibration and
synchronization, then labelled clustering and motion validation.

## What Codex should do next
Claude's Phase 2 is on `feature/go2-sensor-integration`. Read [PROJECT_STATE](PROJECT_STATE.md), then
[SENSOR_VALIDATION](SENSOR_VALIDATION.md). The original Phase 2 priorities are in
[INTEGRATION_REPORT](INTEGRATION_REPORT.md): replay regression test (now added for the committed fixture), `RGBMessage.K` decision, cluster
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

# Phase 1 handoff record (historical; superseded by Phase 2 above)
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
