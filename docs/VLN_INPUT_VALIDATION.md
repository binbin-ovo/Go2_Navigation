# Phase 6A calibrated input validation

## Scope and calibration

Base commit: `e4e0d3b01a16a14f52f7d039b27009ed3e0e0f96` on
`phase5-go2-vln-sensor-calibration`, fetched and inspected 2026-11-12.
Development branch: `phase6-vln-input-validation`, separate worktree.
The original checkout, Claude's calibration environment, `phase5-detection-recovery`,
operational defaults and `main` are unchanged. The calibration branch contains 5A/5B;
it does **not** integrate the separate 5A.1 recovery branch.

Measured 1280×720 fisheye intrinsics are reused directly from
`configs/go2_front_camera_calibration.yaml` (held-out RMS ~0.47 px). No calibration
was fitted or invented here. `T_base_camera` and measured RGB-aligned depth remain
unavailable. Two unsuccessful extrinsics sessions are documented by Claude in
[GO2_LIDAR_CAMERA_EXTRINSICS](GO2_LIDAR_CAMERA_EXTRINSICS.md).

## Image contract and transformations

`calibrated_view(obs)` accepts `go2-vln-obs-v1` RGB uint8 and requires measured K/D,
matching calibration/image dimensions. Synthetic calibration is allowed only by
explicit opt-in. It copies the source into a separate view without modifying the
observation or its history. For raw fisheye input:

1. OpenCV `fisheye.estimateNewCameraMatrixForUndistortRectify`, balance=0,
   output 1280×720, unchanged optical axes.
2. `fisheye.initUndistortRectifyMap` + bilinear `remap`, constant border.
3. New pinhole K, zero distortion, rectified=true, remap validity mask. `R=I` here
   is the same-camera rectification rotation; **it is not T_base_camera**.
4. Official pinned `GroundingDinoProcessor` gets RGB uint8 PIL, lower-case target
   text with terminal period. No generic `vln.tensors.to_tensor` call is used.
5. The processor resizes (short edge 800 / long-edge cap 1333), bilinear sampling,
   pads bottom/right when necessary, rescales by 1/255, normalizes using its saved
   mean `[.485,.456,.406]` and std `[.229,.224,.225]`. Normalization occurs once.
6. Actual resized content and padded dimensions come from the processor's mask;
   unsupported masks/crops are rejected. K scales using the interface's pixel-center
   rule `(c+.5)*s-.5`. Crop is explicitly none, padding offsets are recorded.
   The official box postprocessor returns edge-coordinate boxes in the rectified
   1280×720 view, not tensor pixels or raw fisheye pixels.

Already rectified inputs must specify pinhole K and zero D; no second undistortion.
Source host receipt, estimated robot exposure with uncertainty, exact frame hashes,
clock epoch, calibrated view, processor metadata and untouched history are saved.
The model consumes only the current image; two historical frames are preserved for
future adapters. Sensor timing is not upgraded to hardware synchronization.

CPU FP32 Grounding DINO is a grounding baseline, not a trained navigation policy.
Unknown targets may produce no candidate **or a false positive**. All proposals
require review; none has a robot metric position or clearance claim. Even supplying
future measured extrinsics does not enable metric goals in this adapter.

## Reproducible evidence

See [VLN_OFFLINE_DEMO](VLN_OFFLINE_DEMO.md) for exact commands and artifact definitions.
Results below are updated only after actual execution and visual review.

### Actual run: 2026-11-12

Installed from a new Python 3.12.11 venv, `uv pip install -e '.[test,vln]'`, then
`av==17.1.0` to exercise both optional recording decoder tests. Core pins unchanged:
NumPy 1.26.4, OpenCV 4.11.0.86, SciPy 1.15.2, Open3D 0.19.0, Torch 2.6.0,
torchvision 0.21.0, Ultralytics 8.3.100. Grounding dependencies: Transformers 4.49.0,
huggingface-hub 0.29.3, tokenizers 0.21.4, safetensors 0.5.3, Pillow 11.1.0,
psutil 7.2.2, matplotlib 3.11.2. Full environment freeze is in the private output root; the same versions without
the editable local path are published as `requirements-vln-tested.txt` constraints.

Actual pretrained model and processor: `IDEA-Research/grounding-dino-tiny`, both at
`a2bb814dd30d776dcf7e30523b00659f4f141c71`, official Transformers classes,
safetensors, 172,249,090 parameters, CPU FP32 / four threads. No model training.
No substitute detector, fabricated predictions or hosted image processing.

Commands executed in `/private/tmp/go2-phase6` (cache populated once from the
pinned public repository):

```bash
MPLCONFIGDIR=/private/tmp/go2-phase6-mpl HF_HUB_OFFLINE=1 \
  .venv/bin/python -m pytest -q -rs

HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
  MPLCONFIGDIR=/private/tmp/go2-phase6-mpl \
  .venv/bin/python scripts/phase6_evaluate_grounding.py \
  --recordings /Users/yushan/Desktop/Go2_Navigation/data/recordings \
  --output data/outputs/phase6/evaluation

HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
  MPLCONFIGDIR=/private/tmp/go2-phase6-mpl \
  .venv/bin/python scripts/phase6_grounding_demo.py \
  --observation data/outputs/phase6/evaluation/s_fridge/observation.vlnobs.npz \
  --instruction 'Locate the fridge.' --output data/outputs/phase6/replay_fridge

HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
  MPLCONFIGDIR=/private/tmp/go2-phase6-mpl \
  .venv/bin/python scripts/phase6_grounding_demo.py \
  --observation data/outputs/phase6/evaluation/s_fridge/observation.vlnobs.npz \
  --instruction 'Go toward the kitchen entrance.' \
  --output data/outputs/phase6/unsupported_command
```

Tests: inherited baseline **392 passed / 3 skipped** (75.55 s, PyAV absent at that
point). New focused tests **27 passed** (5.12 s). Final installed environment:
**421 passed / 1 skipped**, 30.71 s. Only `test_go2_hardware.py` skipped because
`GO2_HARDWARE=1` was not set; no live connection/control was attempted. The optional
PyAV decoder tests ran successfully. No existing test was removed or weakened.
Tests cover calibration/view K, rectified-input guards, uint8/single normalization,
processor resize/padding K, query format, unsupported commands, malformed boxes,
ambiguity, missing extrinsics, causal frame provenance, and observation roundtrip.

The actual NPZ replay produced array-identical raw logits, boxes, input IDs and
masks to the corresponding `s_fridge_paraphrase` recording run. The unsupported
navigation command produced no forward pass, no candidate and no metric goal.

### Exact input observations

Two viewpoints of one private indoor environment, not 12 independent scenes.
The case list was declared from local recording previews before inference; the
exact causal selected frames were retained and visually checked. `m_*` uses a
recording with robot motion; this does not validate its odometry.

| Cases | Reference cloud_base index / robot time (s) | Actual RGB file / index | Host receipt (s) | Estimated robot exposure (s) |
|---|---|---|---|---|
| `s_*` | `p5d_static_02`, 200 / 1770427946.8355477 | `000181.png` / 181 | 1794418760.3026161 | 1770427946.7729251 |
| `m_*` | `test_c`, 300 / 1770431294.397286 | `000254.jpg` / 254 | 1794085966.391912 | 1770431294.3850987 |

Exposure uncertainty ~0.01612 s. RGB is respectively 0.06262 s and 0.01219 s before
the reference LiDAR scan. Epochs are the replay's per-recording identifiers
`go2_boot:host_minus_robot=23990813.5` and `go2_boot:host_minus_robot=23654672.0`.
Do not compare robot stamps across these recordings. Per-run `source.json` stores
exact paths, index metadata, hashes, LiDAR window inputs and odometry provenance.

Actual rectified K:
`[[769.779479,0,650.872024],[0,771.925472,371.813802],[0,0,1]]`.
Processor tensor shape **[1,3,750,1333]**, no crop or padding for these single-image
runs. Tensor-space K:
`[[801.653161,0,677.842897],[0,804.089034,387.326877],[0,0,1]]`.
These differ from raw fisheye K by design. Full-precision matrices and transforms
are in each `preprocessing.json`.

### Semantic review, output validity and runtime are different results

Reviewer: **Codex visual inspection**, separate from the Grounding DINO model.
Human verification is **pending**. Review used original/rectified images and drawn
boxes; it did not accept the model's own confidence as evidence. Rubric: acceptable
means correct target region, appropriate multiple-target ambiguity, or absent-target
abstention; partial means correct identity with substantial omitted visible extent;
failed means wrong object, absent-target false positive, or visible-target miss.
Fixed thresholds 0.35 box / 0.25 text were not tuned after seeing these results.

| Case | Model result | Review |
|---|---|---|
| s_fridge | 1 refrigerator, 0.939 | Acceptable, correct region |
| s_fridge_paraphrase | 1 fridge, 0.863 | Acceptable, same target under synonym |
| s_racket | 1 tennis racket, 0.361 | **Partial:** head found, much of handle omitted |
| s_shoes | 4 shoe regions, unresolved multiple | Acceptable ambiguity; overlapping/grouped boxes are not unique object counts |
| s_bicycle | no candidate | Acceptable abstention |
| s_dog | 1 dog, 0.486 | **Failed:** hallway object/shoes falsely labeled dog |
| m_door | 1 door, 0.829 | Acceptable, entrance door |
| m_door_paraphrase | 1 door | Acceptable; fixed wrapper yields identical model query, so this is a formatting check, not learned paraphrase reasoning |
| m_fridge | 1 refrigerator, 0.435 | **Failed:** box on door; real fridge visible at right edge |
| m_shoes | no candidate | **Failed:** visible shoes missed |
| m_bicycle | 1 bicycle, 0.537 | **Failed:** racket/shoe-rack region, no bicycle |
| m_sofa | 1 sofa, 0.477 | **Failed:** black bag, no sofa |

Descriptive case review: **6/12 acceptable, 1/12 partial, 5/12 failed**. This is not
held-out model accuracy or a navigation success rate: one environment, two correlated
views, no independently annotated boxes, agent reviewer, no human adjudication.
Absent-target abstention was only **1/4**. Multiple-target behavior was appropriate
in 1/2 cases. Both wording pairs were consistent, but one pair shares the exact same
model query. Output format was valid in **12/12** runs with zero malformed candidates;
format validity plainly does not establish semantic correctness.

Measured on Apple M3 Pro, 36 GiB, macOS 26.6.2 arm64:
- Model load 0.504 s from populated local cache (no download included; safetensors
  memory mapping means RSS immediately after load understates eventual residency).
- Forward time min/median/mean/max: **1.808 / 1.849 / 1.959 / 2.983 s**; first call
  included, no warmup. Mean processor+forward+postprocessing 1.996 s.
- Process lifetime peak RSS **3,224,846,336 bytes = 3.003 GiB**; whole evaluation
  process including replay and plotting, not isolated model allocation or GPU VRAM.
- Recording initialization experienced multi-minute local filesystem read delays.
  Latencies above exclude recording IO, serialization and rendering. This was an
  interactive shared machine, not an isolated throughput benchmark. No real-time claim.

Private outputs: `/private/tmp/go2-phase6/data/outputs/phase6/`, including
`evaluation/{case}/demo.png`, `evaluation/declared_cases.json`, `model.json`,
`summary.json`, separate `semantic_review.json`, exact raw outputs, logs and environment.
All remain ignored; no images, observation NPZs, weights or raw outputs are published.

## Remaining work

1. Human review of the local cases and a larger, diverse annotated grounding set;
   quantify false positives, visibility, duplicates and abstention on held-out scenes.
2. A true relational/instruction-following or temporal VLN model if that capability
   is required; this adapter intentionally supports only simple target phrases.
3. Measured/held-out validated camera-to-base/LiDAR registration; suitable registered
   depth or another explicitly validated range source for metric grounding.
4. Better exposure synchronization and moving-platform odometry validation.
5. A separately validated collision-aware planner, robot-footprint/swept-volume checks,
   blind-zone handling and execution safeguards. Floor-seen cells alone are insufficient.

No physical robot validation, metric grounding, traversability validation or autonomous
navigation was completed in Phase 6A. The milestone demonstrated actual calibrated Go2
RGB consumption and inspectable semantic proposals, including honest negative results.
