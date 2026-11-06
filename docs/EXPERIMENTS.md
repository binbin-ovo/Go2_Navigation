# Experiment log — 2026-11-06
All experiments below actually ran on macOS ARM64, CPython 3.12.11, CPU. No Go2 was used.

## Environment and reproducibility
Initial host Python: 3.14.0 (unsupported). A fresh Python 3.12 environment was created:
```bash
UV_CACHE_DIR=/private/tmp/go2-uv-cache UV_PYTHON_INSTALL_DIR=/private/tmp/go2-python uv venv --python 3.12 .venv
UV_CACHE_DIR=/private/tmp/go2-uv-cache uv pip install --python .venv/bin/python -r requirements.txt
```
Both succeeded; 77 packages installed. Sandbox network access was enabled for installation
and sample retrieval. Direct versions: torch 2.6.0, torchvision 0.21.0, ultralytics 8.3.100,
Open3D 0.19.0, NumPy 1.26.4, SciPy 1.15.2, opencv-python 4.11.0.86, PyYAML 6.0.2,
pytest 8.3.5. `requirements-macos-arm64.lock` records transitive versions; its editable path
was normalized to `-e .[test]` for portability. Linux/CUDA installation remains untested.
The local runtime is in /private/tmp; recreate .venv if that runtime is cleaned up.

## Inputs and exact sources
```bash
curl -fL --retry 2 https://raw.githubusercontent.com/ultralytics/assets/main/im/bus.jpg -o data/sample/bus.jpg
curl -fL --retry 2 https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt -o yolov8n.pt
curl -fL --retry 2 https://raw.githubusercontent.com/PointCloudLibrary/data/master/tutorials/table_scene_lms400.pcd -o data/sample/table_scene_lms400.pcd
shasum -a 256 yolov8n.pt data/sample/bus.jpg data/sample/table_scene_lms400.pcd
```
All three downloads succeeded. SHA-256:
```text
f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36  yolov8n.pt
c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63  bus.jpg
e285d415641e0d9de695b611db874cc8fe995e8089b77a50d6056d24d8cbcc58  table_scene_lms400.pcd
```
Checkpoint: Ultralytics **YOLOv8n COCO detection**, exact identifier `yolov8n.pt`, official
v8.3.0 assets release. Source documentation: https://docs.ultralytics.com/models/yolov8/.
Image: Ultralytics assets public bus example. Cloud: PCL public tutorial scene from
PointCloudLibrary/data (not a Go2 recording). Moving source branches are verified by the
hashes above. Downloaded binaries are ignored in Git; respect upstream asset licenses.

## Level 1 — synthetic automated validation: passed
```bash
.venv/bin/python -m pytest -q
```
Initial result: 43 passed, 1 failed. `test_edge_scenes[nearby]` exposed a tilted-plane
extrapolation that mislabeled elevated points as ground. Ground RANSAC now samples only
observed support within the expected height band. Subsequent result: **44 passed in 1.33s**.
Coverage includes PLY/PCD/BIN/NPY round trips, empty/malformed inputs, NaN and range filtering,
voxel downsampling, flat ground and wall rejection, nearby merging sensitivity, partial/sparse
scans, AABBs, rigid transforms, camera-to-LiDAR direction, projection/depth rejection,
one-to-one association, frame/resolution/clock/skew checks and mocked RGB wrapper behavior.
Mocked RGB tests are not model accuracy evaluation.

```bash
.venv/bin/python scripts/generate_test_data.py
.venv/bin/python scripts/run_lidar.py --input data/sample/test.pcd --output data/outputs/lidar
.venv/bin/python scripts/run_fusion.py --image data/sample/synthetic.jpg --points data/sample/test.pcd --detections data/sample/synthetic_detections.json --calibration configs/synthetic_calibration.yaml --rgb-timestamp 1 --lidar-timestamp 1 --clock-domain synthetic --output data/outputs/fusion
```
Generator seed 7; ground + two objects = 2040 input points. Final LiDAR run: 1985 processed
points, 1557 ground, **2 clusters**, 0 unclassified; 7.54 ms pipeline wall time.
Fusion: **2 associations**, synthetic calibration and synthetic class fixtures. JSON and
image output saved. The three-panel LiDAR visualization and fusion overlay were visually
inspected. This validates geometry only; no RGB accuracy claim.

## Level 2 — offline real-data inference: smoke tests passed
```bash
YOLO_CONFIG_DIR=/private/tmp/go2-yolo-settings MPLCONFIGDIR=/private/tmp/go2-matplotlib .venv/bin/python scripts/run_rgb.py --input data/sample/bus.jpg --output data/outputs/rgb
.venv/bin/python scripts/run_lidar.py --input data/sample/table_scene_lms400.pcd --output data/outputs/lidar_real
```
RGB config: configs/rgb.yaml, CPU, 640, conf .25, NMS IoU .45. Result: **6 detections**:
bus .873; persons .866, .853, .825, .261; stop sign .255 (rounded). 2411.06 ms first-call
wall time including initialization within predict; not a steady-state benchmark. Structured
JSON and annotated JPEG saved and inspected. A low-confidence prediction is not verified
truth; no mAP/accuracy estimate was computed. Fontconfig emitted a nonfatal cache-directory
warning on this sandboxed host; inference completed successfully.

Cloud config: configs/lidar.yaml unchanged. Result: 2202 processed points, **0 ground,
1 broad cluster, 2 unclassified**, 30.83 ms in the final run. This is a real-file processing
smoke, not successful scene segmentation or object-count validation. The scene's native
frame and geometry have not been calibrated to this project's ground assumptions; parameters
need scene-specific verification. RGB/cloud samples are unrelated and were **not fused**.

## Level 3 — live Go2 inference: not run
Blocked by unavailable robot streams, measured calibration, verified frame conventions,
and common-clock synchronization. CUDA inference, Linux install and real paired RGB–LiDAR
fusion also remain untested. No physical connectivity or robot behavior is claimed.

## Final verification
After formatting and removing one unused import:
```bash
.venv/bin/python -m pytest -q
UV_CACHE_DIR=/private/tmp/go2-uv-cache UV_TOOL_DIR=/private/tmp/go2-tools uvx --offline --from ruff==0.11.2 ruff check src scripts tests
UV_CACHE_DIR=/private/tmp/go2-uv-cache uv pip check --python .venv/bin/python
git diff --check
```
Results: **44 passed in 3.37s**; Ruff all checks passed; all 77 installed packages compatible;
no whitespace errors. Ruff 0.11.2 was run as an isolated tool, not added to runtime dependencies.
