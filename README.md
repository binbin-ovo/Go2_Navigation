## Phase 6B temporal instruction-following pilot (2026-11-12)

`phase6-temporal-vln` adds a local pretrained **VLM navigation-proposal baseline** using
calibrated Go2 RGB history, separate human annotation files, exact model/input evidence,
and offline review videos. SmolVLM2-2.2B actually encoded four images per temporal call
on CPU. Four episodes × two cutoffs × two input modes = 16 calls. All parsed, but every
temporal proposal was TURN_RIGHT; reliable instruction following and history benefit
are **not established**. Human action/progress evaluation is pending. No robot commands
or main merge. **470 tests passed, only the hardware test skipped**, including actual
pretrained model tests; existing Grounding DINO compatibility was checked on a real image.

Start with [install/run/demo](docs/TEMPORAL_VLN_DEMO.md),
[actual successes and failures](docs/TEMPORAL_VLN_EXPERIMENTS.md),
[architecture/model audit](docs/TEMPORAL_VLN_ARCHITECTURE.md),
[episode contract](docs/VLN_EPISODE_INTERFACE.md), and [handoff](docs/AGENT_HANDOFF.md).
Earlier phase sections retain their historical branch results.

## Phase 6A calibrated offline grounding (2026-11-12)

`phase6-vln-input-validation` adds a real pretrained Grounding DINO Tiny demonstration
on calibrated private Go2 RGB recordings. It saves image-space proposals alongside
independent LiDAR geometry, exact preprocessing/timestamp metadata and raw outputs.
It is a grounding baseline, not autonomous navigation. **421 tests pass; 1 hardware
test skipped.** Real-image review includes false positives and misses (6/12 acceptable,
1 partial, 5 failed; human review pending). No main merge or robot commands.

Start with [install and run](docs/VLN_OFFLINE_DEMO.md),
[measured results and limitations](docs/VLN_INPUT_VALIDATION.md),
[model audit](docs/VLN_MODEL_SELECTION.md), and [current handoff](docs/AGENT_HANDOFF.md).
Earlier phase sections below describe their historical branches and results.

## Phase 4.1 social forecasting (2026-11-09)

The separate `phase4-social-forecasting` branch adds masked multi-pedestrian context, an 87,128-parameter social Transformer, three-seed/five-scene comparisons, validation robustness controls and an optional offline multi-track adapter. **The social model loses overall:** macro ADE 0.5812 m vs original 0.5565 m and CV 0.5214 m. Full suite: 259 passed/one hardware skipped. No tracker/hardware changes or navigation control.

See [social architecture](docs/SOCIAL_FORECASTING_ARCHITECTURE.md), [results and exact commands](docs/SOCIAL_FORECASTING_EXPERIMENTS.md), and [integration requirements](docs/FORECASTING_INTEGRATION.md). Earlier milestones remain documented below.

## Phase 4 offline forecasting (2026-11-09)

Public annotated pedestrian forecasting is implemented on `phase4-trajectory-forecasting`; tracking and Go2 sensor APIs remain unchanged. A 78,680-parameter Transformer was trained and evaluated against constant velocity across five held-out scene identifiers. It did not win overall (macro ADE 0.5701 m vs CV 0.5214 m). Full suite: 231 passed, one hardware test skipped. No live forecasting validation or navigation control is claimed.

See [architecture](docs/FORECASTING_ARCHITECTURE.md), [datasets](docs/FORECASTING_DATASETS.md), [installation, commands and results](docs/FORECASTING_EXPERIMENTS.md), and [Claude integration contract](docs/FORECASTING_INTEGRATION.md). Existing phase descriptions below retain their historical validation scope.

# Go2 Dynamic Perception and Navigation

Phase 3.7 adds a separate experimental tracking refinement profile. See
[the refinement evaluation](docs/TRACKING_REFINEMENT.md) for measured gains, regressions and reproduction.

Phase 3.6 adds an opt-in tracking robustness profile and reproducible before/after benchmarks.
See [tracking robustness](docs/TRACKING_ROBUSTNESS.md) for verified results and remaining limitations.

**macOS:** the Go2 is read natively from an Apple Silicon Mac with the Phase 2 adapter
(`.venv-go2`, Python 3.10). Setup: [MAC_CONNECTION](docs/MAC_CONNECTION.md); real tracking
results: [PHASE3_REAL_VALIDATION](docs/PHASE3_REAL_VALIDATION.md). Tests: 135 passed, 1 hardware skipped.

**Phase 3 adds offline temporal 3D tracking** on branch `phase3-tracking`: persistent IDs,
ego-motion compensation, planar velocity/covariance, moving/stationary/uncertain states,
constant-velocity trajectories, and headless BEV/video. This branch is not merged into main.
```bash
python scripts/generate_tracking_test_data.py
python scripts/run_tracking.py --input data/sample/tracking_sequence/crossing --config configs/tracking.yaml --output data/outputs/tracking_run --video
python scripts/evaluate_tracking.py --input data/sample/tracking_sequence --output data/outputs/tracking_eval.json
```
Install the package first (below); use a new output directory for each run. Real recording replay
uses the same CLI with a Phase 2 recording directory and recorded odometry. No calibration is
invented; real RGB–LiDAR fusion remains blocked. Phase 3 tests: 127 passed, 1 hardware skipped.
See [tracking architecture](docs/TRACKING_ARCHITECTURE.md),
[evaluation and limitations](docs/TRACKING_EVALUATION.md), and
[agent handoff](docs/AGENT_HANDOFF.md). No real-world tracking accuracy is claimed.

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
for CUDA and `--model /path/to/yolov8n.pt`. Phase 2 records Linux/CUDA validation on the hardware workstation; see
[GO2_CONNECTION](docs/GO2_CONNECTION.md).

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
  tracking/              filtering, assignment, lifecycle, trajectories, replay/evaluation
  localization/          explicit detection frame/pose normalization
  visualization/         headless tracking BEV
  adapters/go2/          read-only DDS/RTP access, recording, replay and cloud windows
  adapters/ros2/         optional ROS integration boundary documentation
scripts/                 offline pipelines, fixtures, Go2 recording/replay/live tools
tests/                   core/adapter/replay tests and opt-in hardware checks
data/sample/             generated/downloaded ignored inputs
data/outputs/            generated ignored outputs
data/recordings/         private local recordings (ignored)
docs/                    architecture, setup, sensor contract, experiments, handoff
```
The existing repository is used directly rather than adding a nested repository. Small
processor, serialization and fixture modules were added for reuse. No placeholder
ROS bridge is supplied; the verified Go2 integration uses the optional SDK adapter.

Start integration with [SENSOR_INTERFACE](docs/SENSOR_INTERFACE.md),
[AGENT_HANDOFF](docs/AGENT_HANDOFF.md), and [ARCHITECTURE](docs/ARCHITECTURE.md).
Ultralytics code/checkpoints carry upstream licensing requirements (AGPL-3.0 or an
applicable enterprise license); review upstream terms before redistribution.

## Phase 5A experimental scene representation

Independent opt-in accumulation, background voxels, filtered boxes and motion evidence are available on `phase5-stable-lidar-perception`. The original tracker and live defaults are preserved. Read [architecture](docs/STABLE_LIDAR_ARCHITECTURE.md) and [measured limitations](docs/STABLE_LIDAR_EXPERIMENTS.md) before use: real fresh coverage and moving recall regress, so this is not an operational replacement. Claude can consume the frozen [scene-v1 interface](docs/SCENE_REPRESENTATION_INTERFACE.md) and synthetic fixture independently.
