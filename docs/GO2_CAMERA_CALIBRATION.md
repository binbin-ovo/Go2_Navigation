# Go2 front camera: audit and intrinsic calibration (Phase 5D)

**Status (2026-11-11): intrinsics MEASURED** → `configs/go2_front_camera_calibration.yaml`
(`go2-camera-calibration-v1`, `status: measured`). No factory calibration is published; these values
come only from our ChArUco captures below. Extrinsics are still missing.

## Result
| Quantity | Value |
|---|---|
| Model | `fisheye_equidistant` (OpenCV `cv2.fisheye`), raw 1280×720 frames |
| fx, fy | 831.4, 833.7 px (bootstrap σ 7.4 px ≈ 0.9%) |
| cx, cy | 646.2, 369.6 px (σ 1.5, 2.1 px) |
| D (k1…k4) | −0.0548, −0.1079, 0.0812, −0.0119 |
| Field of view (from the model) | ≈ 94° horizontal × 50° vertical |
| Fit RMS (177 views) | 0.53 px |
| **Held-out RMS** (58 views, 1 599 corners) | **0.47 px**; p95 0.95 px; max 2.97 px |
| Held-out centre / border RMS | 0.48 / 0.43 px (355 border corners) |
| `pinhole_radtan` on the same split | 0.82 px held-out, and implausible fx ≈ 1 693 px → rejected |
| Distortion valid to image corner | yes: radius increases to 1 259 px > corner 744 px |
| Bootstrap (20 refits) ray-direction spread | median 0.18°, p95 0.29° after removing a common rotation |
| Independent check | yaw cross-correlation on `test_c` gave ≈ 836 px/rad of image motion near the centre, consistent with fx |

Leave-one-recording-out (fit on two passes, score the whole third):
| Left out | fx | Held-out RMS | Ray angle vs all, median / p95 |
|---|---|---|---|
| `p5d_charuco_01` (centre, mid band) | 825.9 | 0.47 px | 0.28° / 0.50° |
| `p5d_charuco_02` (bottom corners) | 851.6 | 0.81 px | 0.68° / 1.07° |
| `p5d_charuco_03` (top band) | 800.8 | 0.54 px | 0.91° / 1.31° |

Dropping a region shifts fx by up to 4% because focal length and distortion trade off; with all
regions the bootstrap spread is 0.9%. **Coverage of every image region is required**: the first pass
alone (top third empty) produced a distortion curve that folds back before the image corner, which
the tool now refuses. Straight edges (fridge, door frame, counter) are straight after undistortion
(visual check only).

Captures (local, git-ignored): `p5d_charuco_01` (80 s, 478 detected frames), `_02` (70 s, bottom
corners, 226), `_03` (55 s, top band with the board at knee/chair height, 586); 235 distinct views.
Target: A4 ChArUco 9×6, DICT_5X5_100, printed at 96.7% (measured square 29 mm, marker 21 mm; ±1 mm
ruler). The square size does not affect intrinsics (tested); it matters for extrinsics, which should
use a 9-square measurement.

Reproduce:
```bash
D=/Users/yushan/Desktop/Go2_Navigation/data
PYTHONPATH=src .venv/bin/python scripts/phase5d_calibrate_intrinsics.py \
  --recording $D/recordings/p5d_charuco_01 $D/recordings/p5d_charuco_02 $D/recordings/p5d_charuco_03 \
  --board $D/outputs/phase5d/targets/charuco_a4_9x6_30mm.json --square-mm 29 \
  --output configs/go2_front_camera_calibration.yaml --report $D/outputs/phase5d/intrinsics_final.json
```

## Verified stream properties (recording `p5d_static_02`, 282 frames, lossless PNG)
| Property | Value | How verified |
|---|---|---|
| Source | H.264 over RTP multicast 230.1.1.1:1720 from 192.168.123.161 | socket sender |
| Codec | H.264 High, level 4.0, progressive, `has_b_frames=0`, 1 ref frame | ffprobe / PyAV on `camera.h264` |
| Resolution | 1280 × 720 every frame, sample aspect ratio N/A (square pixels assumed) | index + decode |
| Pixel format | yuv420p; colour range / matrix / primaries **unspecified** in the stream | ffprobe |
| Rate | 14.28 Hz host receipt; RTP counter nominal 15 Hz, runs at 0.952× | index |
| GOP | IDR every 2.0 s; up to 2 s of frames dropped after (re)joining until the first IDR | recorder stats |
| Loss | 0 of 2 756 RTP packets lost / reordered; 0 decoder errors | recorder stats |
| Decoder output | BGR uint8 (OpenCV order) | `rtp_h264.py` |
| On disk | PNG (this recording) or JPEG q95 (older ones, re-compressed, lossy) | index `file` |
| Rectification | **None**: visibly strong barrel / fisheye distortion | images |
| Intrinsics / distortion | None published (no `CameraInfo`; `rt/frontvideostream` silent); **measured by us**, below | probe |
| Exposure timestamp | **Not available**; only host receipt | see [GO2_SENSOR_TIMING](GO2_SENSOR_TIMING.md) |
| Field of view | ≈ 94° × 50° from the measured model (below) | calibration |

Static-scene noise: mean absolute grey-level change between frames 1.4 DN. Colour conversion from
YUV uses the decoder's default (BT.601 limited range), which is an assumption.

## Conversion to model inputs
- `VisualObservation.image` is **RGB** uint8 H×W×3 (`cv2.cvtColor(BGR2RGB)` in the replay builder).
  Verified: PIL reading the PNG equals the OpenCV BGR array with channels reversed.
- Tensor: `vln.tensors.to_tensor` → float32 3×H×W in [0, 1], then ImageNet mean/std by default.
- Sequence: `vln.tensors.stack_history` → T×H×W×3, oldest first; frames are ~70 ms apart, not 66.7.
- Resize: `vln.tensors.resize` returns new intrinsics with the pixels. A uniform scale s maps
  `fx, fy → s·fx, s·fy` and `c → (c + 0.5)·s − 0.5` (pixel-centre convention); letterboxing adds
  the pad to `c`. A non-uniform resize (e.g. 1280×720 → 224×224 without letterbox) changes the aspect
  ratio and makes fx ≠ fy relative to the scene; prefer letterbox for geometry. Distortion
  coefficients do not change under resize because they act on normalized coordinates.
- Undistortion before a pinhole-only model must use the measured model; the interface keeps raw
  images and marks `rectified=False`.

## Calibration procedure (as run)
Target: ChArUco, DICT_5X5_100, generated by `scripts/phase5d_make_charuco.py`.
- Printed: A3 11×8 × 35 mm (preferred), A4 9×6 × 30 mm, Letter 8×6 × 30 mm PDFs at exact page size.
- Screen: `--screen WxH` writes a full-screen PNG (9×6 for wide screens, 8×6 for 4:3). Show it at
  native resolution with no scaling; **measure 5+ squares with a ruler** and record the square size.
  A screen is flat to well below a millimetre, which suits intrinsics.

Steps, robot stationary, one capture at a time with the user:
1. Record 30–60 s while the user moves the target slowly: centre, all four corners and edges (the
   distortion is strongest there), tilts of ±30–45°, distances 0.3–1.2 m. Hold still ~1 s per pose.
2. Select ≥ 40 sharp frames with ≥ 12 detected corners, spread over the image.
3. Fit both `fisheye_equidistant` (4 coefficients) and `pinhole_radtan` (5, or rational 8). Choose by
   held-out reprojection error, not training error.
4. Validate on ≥ 10 held-out frames, report RMS and per-region error (centre vs border), and check
   straight edges after undistortion.
5. Save `configs/go2_front_camera_calibration.yaml` (`go2-camera-calibration-v1`, `status: measured`,
   image size 1280×720, K in pixels, the model and D, RMS, date, target, square size, frame count).

Acceptance target: held-out RMS < 0.5 px centre, < 1.0 px border. Anything worse is reported, not hidden.

Tool (steps 2–5): `scripts/phase5d_calibrate_intrinsics.py --recording … --board <target>.json
--square-mm <measured> --output configs/go2_front_camera_calibration.yaml --report …json`
(`go2_perception.vln.intrinsics`). It drops near-duplicate views, holds out every 4th distinct view,
fits both models, re-estimates only the board pose on held-out views, and reports RMS / p95 / max,
centre vs border error and a 4×3 image coverage grid. Synthetic check
(`tests/test_vln_intrinsics.py`): ChArUco views rendered through a known fisheye camera are recovered
to fx/fy within 1% and principal point within 1.5 px, held-out RMS < 0.45 px. In that test both
models fit equally well over a ~100° field, so model choice on the real camera must come from the
held-out border error, not from the shape of the lens.

## File format (`go2-camera-calibration-v1`)
```yaml
schema: go2-camera-calibration-v1
status: measured            # or synthetic (tests only)
camera_frame: go2_front_camera_optical   # x right, y down, z forward
image_size: [1280, 720]     # width, height the intrinsics belong to
model: fisheye_equidistant  # or pinhole_radtan
K: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]   # pixels, pixel-centre origin
D: [k1, k2, k3, k4]         # radtan: [k1, k2, p1, p2, k3]
reprojection_rms_px: ...
units: {K: pixels, translation: metres}
provenance: {...}
T_base_camera: null         # or {matrix, status, direction: "p_base = T @ p_camera", provenance}
```
Loaded by `go2_perception.vln.calibration.load_calibration`, which rejects measured extrinsics
built on non-measured intrinsics. The synthetic example is `tests/fixtures/vln/synthetic_recording/synthetic_calibration.yaml`.
