# Go2 Dynamic Perception and Navigation
**Phase 2 adds read-only access to a real Go2 Pro:** record, replay and run live YOLO and LiDAR
perception. Fusion on real data is blocked until calibration exists. The robot is never commanded.
```bash
.venv/bin/python scripts/go2_record.py --seconds 20 --name sample   # needs the robot
.venv/bin/python scripts/go2_replay.py --recording data/recordings/sample --window-s 0.3
.venv/bin/python scripts/go2_live.py --seconds 30 --display          # needs the robot
.venv/bin/python scripts/go2_view_cloud.py --recording data/recordings/sample --index 100 --window-s 0.3
```
Setup: [GO2_CONNECTION](docs/GO2_CONNECTION.md). Evidence: [SENSOR_VALIDATION](docs/SENSOR_VALIDATION.md).
Status: [INTEGRATION_REPORT](docs/INTEGRATION_REPORT.md). The LiDAR window image is in
[docs/media](docs/media/go2_lidar_clusters_window0p3.png).

Phase 1: offline RGB detection, geometric LiDAR processing, and preliminary calibrated
RGB–LiDAR association. No robot connection, movement commands, training, or ROS installation
is needed. **Real Go2 sensor fusion is not validated.**

## Install and run
Python 3.10–3.12; tested environment and results: [PROJECT_STATE](docs/PROJECT_STATE.md).
Run from the repository root:
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
Download the real image separately as described in [EXPERIMENTS](docs/EXPERIMENTS.md).
YOLO downloads `yolov8n.pt` on first use; pre-cache it for disconnected operation. The fusion
command uses hand-authored synthetic detections to test geometry, **not YOLO accuracy**.
Omit `--detections` to run YOLO on the supplied image; use matching measured calibration
and synchronized point-cloud data for real inputs. CPU defaults; RGB supports `--device 0`
for CUDA and `--model /path/to/yolov8n.pt`. No CUDA hardware validation is claimed.

## Capabilities
- COCO YOLO detection, typed results, BGR image/path/batch input, JSON and labeled images.
- XYZ/XYZI arrays, NPY, PLY, PCD and explicitly laid-out BIN input; finite/range filtering,
  optional voxel sampling, gravity-constrained RANSAC, DBSCAN, 3D axis-aligned boxes.
- Ground, unclassified points and obstacle clusters remain distinct. Clusters have no
  semantic class until associated with an RGB detection.
- Rectified pinhole projection and one-to-one geometric association with strict frame,
  image resolution, clock-domain and synchronization checks.
- Headless three-panel XY LiDAR visualization, colored PLY, fusion overlays and JSON.

## Layout
```text
configs/                 YAML defaults and explicitly synthetic calibration
src/go2_perception/
  core/                  result types, AABBs, rigid transforms
  rgb/                   lazy Ultralytics wrapper and annotation
  lidar/                 filtering, RANSAC, DBSCAN, processor, visualization
  fusion/                calibration, projection, association, visualization
  io/                    loaders, messages, serialization, configuration, fixtures
  adapters/ros2/         verified integration boundary documentation
scripts/                 RGB, LiDAR, fusion CLIs and deterministic fixture generator
tests/                   geometry, LiDAR, fusion, I/O and RGB wrapper tests
data/sample/             generated/downloaded ignored inputs
data/outputs/            generated ignored outputs
docs/                    architecture, setup, sensor contract, experiments, handoff
```
The existing repository is used directly rather than adding a nested repository. Small
processor, serialization and fixture modules were added for reuse. No empty transport
bridge is supplied before real Go2 messages are verified.

Start integration with [SENSOR_INTERFACE](docs/SENSOR_INTERFACE.md),
[AGENT_HANDOFF](docs/AGENT_HANDOFF.md), and [ARCHITECTURE](docs/ARCHITECTURE.md).
Ultralytics code/checkpoints carry upstream licensing requirements (AGPL-3.0 or an
applicable enterprise license); review upstream terms before redistribution.
