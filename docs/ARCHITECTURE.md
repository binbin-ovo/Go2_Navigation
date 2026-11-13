## Phase 6C isolated navigation-trained inference (2026-11-13)

`phase6-pretrained-vln` adds a separate file-based LightNav worker, leaving the existing
sensor/perception/tracking/SmolVLM paths and operational defaults unchanged. Core
`pretrained_inputs.prepare_job` reuses the episode contract and measured calibrated
views. `pretrained.causal_history` creates a real causal 4 Hz observation stream; the
official LightNav processor then applies its own SlowFast policy and 448×256 resize.
The Python 3.11 / Torch 2.10 / Transformers 5.8 model environment is isolated from the
Phase 6B-compatible core environment. No evaluation-only field can enter a worker job.

`scripts/lightnav_worker.py` loads only pinned inference code, processor and RVQ assets,
resets native buffers/caches, captures exact tensors/native chunks and records runtime
identity/timing/memory. Observation-only replay does not feed proposals back as executed
actions. `evaluate_pretrained_vln.py` binds frozen manifest/cutoff identities and exact
prepared-image hashes, then loads human references separately. Full package/baseline
preservation is checked by `audit_pretrained_inputs.py`. All original benchmark results
and experimental failures remain intact. Native ego SE(2) is not registered to Go2.

`vln/simulation.py` is an optional MuJoCo-only camera proxy with explicit local actions,
contact constraints and host-clock response validation; no hardware/network transport.
Its evaluator-only scene geometry never reaches the model. The executed 16-step static
rollout failed to reach/stop at the destination; no latency-aware or dynamic-safety
claim. Windows/Isaac validation is pending actual machine access. Read
[feasibility/native protocols](PRETRAINED_VLN_FEASIBILITY.md),
[experiments, public synthetic panels and commands](PRETRAINED_VLN_EXPERIMENTS.md),
and [Windows preflight](WINDOWS_SIMULATION_FEASIBILITY.md).

## Phase 6B prepared-episode scoring update (2026-11-13)

The runner accepts repeated exact `--episode` manifests, including separate
`model_input/` and `evaluation_only/` folders. The model path is unchanged: instruction
and causal image/time context only. `scripts/evaluate_temporal_vln.py` reads saved
outputs, validates manifest/cutoff binding, and scores only reviewed human references.
It groups results by source recording, separates explicitly identified static probes,
and reports pending references without inserting them into scored denominators.
Sequence flags are omitted after ambiguity filtering, which can remove intervening
cutoffs. No episode schema, detector, model, prompt, parser or sensor defaults changed.

The exact Claude packages were processed unchanged: 11 decisions / 22 calls, one
correlated test_c group. E1–E3 reference agreement 5/9 temporal vs 7/9 current-only;
static E4 2/2 vs 0/2. Full suite 478 passed, one hardware skipped. See
[TEMPORAL_VLN_CLAUDE_PILOT](TEMPORAL_VLN_CLAUDE_PILOT.md) for actual provenance,
limitations and reproduction. These scores are not closed-loop navigation success.

## Phase 6B temporal instruction following (2026-11-12)

The optional `temporal-vln` runtime adds strict episode manifests and separate
annotation contracts, causal availability-aware RGB history, per-image calibrated
rectification and an official multi-image SmolVLM2-2.2B adapter. Native text and exact
tensors/provenance are preserved; parsing produces offline categories with unknown
action frame and execution_authorized=false. Human annotations are only loaded after
inference for evaluation. No sensor/tracker/calibration code or defaults changed.

The real pilot encoded four images per temporal call, but all eight temporal proposals
were TURN_RIGHT. Human semantic evaluation is pending; this is a VLM navigation-proposal
baseline, not a validated VLN policy. Full suite including actual model: 470 passed,
only hardware skipped. Read [architecture and public APIs](TEMPORAL_VLN_ARCHITECTURE.md),
[experiments/failures](TEMPORAL_VLN_EXPERIMENTS.md), [reproduction](TEMPORAL_VLN_DEMO.md),
and [Claude's episode contract](VLN_EPISODE_INTERFACE.md). Earlier sections are historical.

## Phase 6A optional offline grounding

The unchanged `go2-vln-obs-v1` observation feeds a separate model path:
`Go2RecordingReplay` / saved NPZ → `model_preprocessing.calibrated_view` → official
pinned Grounding DINO processor/model → validated image-region proposals.
`grounding_demo.run_observation` saves exact input/output evidence and draws the
independent existing LiDAR grid alongside RGB. Core sensor adapters and perception
internals are unchanged; Transformers is an optional `vln` dependency imported only
when constructing the model.

The raw fisheye view keeps its measured K/D. Rectification creates a new pinhole view
with its own K, and processor resize/padding produces separately recorded tensor K.
Source clocks/epoch and history survive; only the current RGB image enters the model.
Normalization belongs to the official processor. Text wrappers are lexical, not a
learned navigation instruction parser. No depth, extrinsics, poses or LiDAR cells
are supplied to the model, and no metric goal is emitted. Geometry warnings remain
independent of semantic scores. All model outputs are proposals only.

Public entry points: `calibrated_view`, `processor_geometry` in
`vln/model_preprocessing.py`; `GroundingDino.predict`, `format_instruction`,
`parse_candidates` in `vln/grounding.py`; `run_observation`, `replay_source` in
`vln/grounding_demo.py`. Configuration and exact coordinate/artifact definitions:
[VLN_OFFLINE_DEMO](VLN_OFFLINE_DEMO.md). Actual validation and unresolved problems:
[VLN_INPUT_VALIDATION](VLN_INPUT_VALIDATION.md). Main and operational defaults are unchanged.

## Phase 4.1 social forecasting (2026-11-09)

The separate `phase4-social-forecasting` branch adds masked multi-pedestrian context, an 87,128-parameter social Transformer, three-seed/five-scene comparisons, validation robustness controls and an optional offline multi-track adapter. **The social model loses overall:** macro ADE 0.5812 m vs original 0.5565 m and CV 0.5214 m. Full suite: 259 passed/one hardware skipped. No tracker/hardware changes or navigation control.

See [social architecture](SOCIAL_FORECASTING_ARCHITECTURE.md), [results and exact commands](SOCIAL_FORECASTING_EXPERIMENTS.md), and [integration requirements](FORECASTING_INTEGRATION.md). Earlier milestones remain documented below.

## Phase 4 offline forecasting (2026-11-09)

Public annotated pedestrian forecasting is implemented on `phase4-trajectory-forecasting`; tracking and Go2 sensor APIs remain unchanged. A 78,680-parameter Transformer was trained and evaluated against constant velocity across five held-out scene identifiers. It did not win overall (macro ADE 0.5701 m vs CV 0.5214 m). Full suite: 231 passed, one hardware test skipped. No live forecasting validation or navigation control is claimed.

See [architecture](FORECASTING_ARCHITECTURE.md), [datasets](FORECASTING_DATASETS.md), [installation, commands and results](FORECASTING_EXPERIMENTS.md), and [Claude integration contract](FORECASTING_INTEGRATION.md). Existing phase descriptions below retain their historical validation scope.

# Architecture

Phase 3.7 adds a separate experimental tracking refinement profile. See
[the refinement evaluation](TRACKING_REFINEMENT.md) for measured gains, regressions and reproduction.

Phase 3.6 adds opt-in robustness within the existing tracker, preserving all sensor/perception
interfaces. See [TRACKING_ARCHITECTURE](TRACKING_ARCHITECTURE.md) for data flow and
[TRACKING_ROBUSTNESS](TRACKING_ROBUSTNESS.md) for the profile, benchmark and limitations.

**Phase 3:** timestamped LiDAR/fused results now feed `tracking.MultiObjectTracker` (import
from `tracking.multi_object_tracker`), with explicit odometry or fixed-sensor normalization,
Kalman velocity estimation, lifecycle/motion states, CV trajectories and headless BEV. See
[TRACKING_ARCHITECTURE](TRACKING_ARCHITECTURE.md) for the complete current interface.
The original perception and hardware code is unchanged; the sections below describe that core.

**Phase 2:** `adapters/go2` connects the real robot to the APIs below, without changing them.
```text
Go2 RTP video -> adapters/go2 (decode, keep receipt time) -> RGBDetector.detect -> RGBResult
Go2 DDS cloud_base + odometry -> adapters/go2 (decode, level, optional window)
    -> LidarMessage -> process_lidar -> LidarResult
RGB + LiDAR + Calibration -> FusionProcessor   [BLOCKED on real data: no calibration]
Recordings (JPEG/PNG, NPY, JSONL) replay through the same path without the robot or the SDK.
```
Only `receivers.py` and `dds_types.py` import the SDK; only `H264Decoder` imports PyAV. Core and
perception modules remain free of transport imports. Design details and reasons are in
[INTEGRATION_REPORT](INTEGRATION_REPORT.md). The rest of this file describes the Phase 1 core.

Images → `load_image` → `RGBDetector.detect` → `RGBResult`.
Clouds → `load_pointcloud` → `LidarProcessor.process` → `LidarResult`.
Results + `Calibration` → `FusionProcessor.fuse` → `FusionResult`.
Transport-independent messages wrap the same API; no sensor acquisition runs in core.

## Public APIs
- `RGBDetector(model, device, confidence, nms_iou, image_size)`: exact defaults in
  configs/rgb.yaml. `detect(path_or_bgr, timestamp=None, frame_id='camera_optical',
  clock_domain=None)` returns one RGBResult. `detect_batch(iterable)` returns results
  in input order. Batch metadata must be assigned by caller. Inference latency is wall
  time including preprocessing/postprocessing and first-call overhead, excluding model
  construction. Batch latency is amortized, not individual-image timing.
- `LidarProcessor(preprocessing={}, ground={}, clustering={}).process(points,
  timestamp=None, frame_id='lidar', clock_domain=None)` returns filtered XYZ, ground,
  unclassified points, obstacles, labels and elapsed time. Constructor dictionaries
  map directly to the named functions and configs/lidar.yaml keys.
- `Calibration(K, T_lidar_camera, image_size, camera_frame, lidar_frame, distortion,
  rectified, synthetic)` or `Calibration.from_yaml(path)` validates rigid extrinsics.
- `project_points(XYZ, calibration, min_depth=1e-6)` returns visible UV, optical depth
  and original point indices. Camera-to-LiDAR extrinsics are inverted internally.
- `FusionProcessor(**configs/fusion.yaml).fuse(rgb, lidar, calibration)` validates
  compatibility and returns fused detections plus unmatched input indices/cluster IDs.
- `load_pointcloud(path_or_array, bin_fields=4)`, `load_image(path_or_array)`,
  `save_json(path, result)` and message helpers are transport-independent.

Results are dataclasses in core/types.py. Boxes are `[xmin,ymin,xmax,ymax]` pixels
or XYZ minimum/maximum in the LiDAR frame; 3D position is observed AABB midpoint,
not an estimated complete physical-object center. Timestamp of a fused 3D detection
is the LiDAR observation time; RGB time is retained separately.

## Geometry and limitations
Ground RANSAC samples planes with seeded NumPy RNG, constrains normal tilt to +/−Z
and plane height at the sensor origin. Samples must also have observed support within
the expected height band, avoiding spurious extrapolated ground from elevated objects. Set expected_ground_z from actual mounting /
pose; defaults are illustrative. No qualifying plane means no ground removal, rather
than silently labeling a wall. This assumes a gravity-aligned scan; steep terrain and
body roll need upstream rotation or a future gravity input. No de-skew is implemented.

Open3D voxel downsampling creates centroids and drops intensity. Labels index processed
points, not raw input. DBSCAN uses fixed metric radius / minimum support; sparse points
remain unclassified, and close/touching objects may merge. No automotive density or full
360-degree scan is assumed. Short accumulated windows must already share a frame/time
reference and be motion compensated by the producer. The Phase 1 core has no temporal tracker; Phase 3 now consumes its outputs in a separate package.

Association scores are fractions of visible cluster points inside each RGB box; require
minimum support and fraction. SciPy Hungarian assignment maximizes total score one-to-one.
This is a baseline: no occlusion model, instance splitting, semantic LiDAR classifier,
or calibrated association probability. RGB confidence is preserved, not multiplied by
association score. Unmatched clusters remain geometric obstacles. Real fusion is always
reported as unvalidated by this offline implementation, even with user-provided calibration.

## Dependencies and configuration
PyTorch/torchvision + Ultralytics for inference; OpenCV for image I/O/annotation;
NumPy/Open3D for geometry and cloud I/O; SciPy for assignment; PyYAML for config.
Heavy imports are lazy except where explicitly needed. Core has no ROS imports.
`configs/default.yaml` selects component paths for the fusion CLI; individual CLIs
accept their own `--config`. Run scripts from repo root. Unknown constructor keys fail
rather than silently disappearing. Full direct pins are in pyproject.toml; the tested
transitive environment is recorded separately. ROS adapter is intentionally documentation
(Phase 2 uses the Unitree SDK directly because no system ROS 2 is installed; see `adapters/go2`).

## Phase 5A experimental scene representation

Independent opt-in accumulation, background voxels, filtered boxes and motion evidence are available on `phase5-stable-lidar-perception`. The original tracker and live defaults are preserved. Read [architecture](STABLE_LIDAR_ARCHITECTURE.md) and [measured limitations](STABLE_LIDAR_EXPERIMENTS.md) before use: real fresh coverage and moving recall regress, so this is not an operational replacement. Claude can consume the frozen [scene-v1 interface](SCENE_REPRESENTATION_INTERFACE.md) and synthetic fixture independently.
