## Phase 6A → next agent: calibrated offline grounding (Codex, 2026-11-12)

Implementation commit: [`dcc5de1c0d899d5fb8159c5f6b0905ab67b8cf01`](https://github.com/binbin-ovo/Go2_Navigation/commit/dcc5de1c0d899d5fb8159c5f6b0905ab67b8cf01).

- **Branch:** `phase6-vln-input-validation`, base
  `e4e0d3b01a16a14f52f7d039b27009ed3e0e0f96` from Claude's calibration branch.
  No main merge. No robot connection or commands. `phase5-detection-recovery` and
  Claude's worktree/environment were not modified. This base has 5A/5B, not 5A.1.
- **Capability actually tested:** Grounding DINO Tiny text-conditioned image-region
  proposals from measured-calibration Go2 recordings. Not a VLN navigation policy.
- **Read first:** [VLN_INPUT_VALIDATION](VLN_INPUT_VALIDATION.md) for actual measurements,
  failures and calibration limits; [VLN_OFFLINE_DEMO](VLN_OFFLINE_DEMO.md) for exact
  commands/artifacts; [VLN_MODEL_SELECTION](VLN_MODEL_SELECTION.md) for the checkpoint audit;
  [GO2_VLN_INPUT_INTERFACE](GO2_VLN_INPUT_INTERFACE.md) for the unchanged sensor contract.

**Implementation:** `vln/model_preprocessing.py` rectifies raw RGB and carries calibrated
K through official processing; `vln/grounding.py` loads pinned pretrained weights and
returns image proposals; `vln/grounding_demo.py` saves evidence and a separate LiDAR grid.
CLIs: `scripts/phase6_grounding_demo.py`, `scripts/phase6_evaluate_grounding.py`.
Config/cases: `configs/vln_grounding.yaml`, `configs/vln_grounding_eval.yaml`.
Regression tests: `tests/test_vln_grounding.py` (27 tests).

**API:** `view = calibrated_view(obs)`; `result = model.predict(view, instruction)`;
`run_observation(obs, model, output, source=...)` writes the full experiment.
Use a `Go2VLNObservation` with raw RGB uint8 and measured K/D of matching dimensions.
The model gets one current image, preserves but ignores history, and applies its own
normalization once. Output boxes use rectified-image edge pixels, optical axes remain
right/down/forward. No conversion to base_link forward/left/up or metric distance.
All `metric_goal`, `action`, `point_base_m` fields remain null even if future callers
supply extrinsics. The right-hand LiDAR plot is a separate observation, not a fused map.

**Important correction to older guidance below:** floor-seen cells and sector floor
returns do **not** establish traversability or a safe swept corridor. The historical
Phase 5D symbolic-action conversion paragraph is not an approved Phase 6 execution
path. Pixel proposals do not become camera-relative or base-relative metric waypoints.
No executable-goal adapter exists here.

**Install / verify:** in this branch, `uv venv --python 3.12 .venv`,
`uv pip install --python .venv/bin/python -e '.[test,vln]'`, optionally install `av==17.1.0`
for inherited decoder tests, then `.venv/bin/python -m pytest -q -rs`.
Actual result: **421 passed, 1 hardware skipped**, 30.71 s. No automated tests weakened.
Model + processor pinned to `IDEA-Research/grounding-dino-tiny` revision
`a2bb814dd30d776dcf7e30523b00659f4f141c71`; CPU FP32 / four threads.

**Local evidence:** `/private/tmp/go2-phase6/data/outputs/phase6/evaluation/` (ignored).
Each case has raw observation/model output, source hashes/clocks, preprocessing,
proposal JSON and demo PNG. Start with `s_fridge/demo.png` (correct) and `m_fridge/demo.png`
(wrong door region), then `s_dog`, `m_bicycle`, `m_sofa` false positives and `m_shoes`
miss. `semantic_review.json` records Codex visual review; human review is pending.
The 12-case declaration predates inference; thresholds were not retuned. 6 acceptable,
1 partial, 5 failed. Output validity was 12/12; semantic correctness differs.
NPZ replay reproduced raw arrays exactly. A navigation instruction was rejected as
unsupported without forwarding through the model. All private media, outputs and
weights remain out of Git. Local temporary paths may eventually be cleaned; rerun
from the private recordings and pinned model using the documented commands.

**Next work:**
1. Human-review the local panels and create diverse held-out object/absent/relational
   grounding annotations. Record failures; do not optimize on these 12 cases.
2. Claude: pursue measured, held-out validated camera/base/LiDAR registration and
   timing/odometry validation; preserve unknown fields until verified.
3. Select a true temporal/relational VLN model in a separate experiment if needed;
   its output still requires a documented frame and independent geometry checks.
4. Only after validated registration/range: design metric grounding and a separate
   collision-aware planner. No locomotion/control work is authorized by this milestone.

## Phase 5D → Codex handoff: Go2 VLN inputs (Claude, 2026-11-11)

- **Branch:** `phase5-go2-vln-sensor-calibration`, based on `89d901dc` (5B). Do not merge main.
  5A/5A.1 perception modules, configs and defaults are unchanged.
- **Interface:** `go2_perception.vln`, schema `go2-vln-obs-v1`. Spec:
  [GO2_VLN_INPUT_INTERFACE](GO2_VLN_INPUT_INTERFACE.md).

**Available now (real data):**
| Input | Shape / units | Frame | Time |
|---|---|---|---|
| RGB | uint8 (720, 1280, 3) **RGB**, raw/distorted (fisheye K, D measured), ~14.3 Hz, + history | `go2_front_camera_unrectified` | host receipt, and estimated robot-clock exposure (host − offset − 48 ms, σ 16 ms) |
| Obstacle points | float32 (~11 k, 3) m, 0.5 s window, odom-compensated | `base_link` of the reference scan | scan midpoint, robot clock |
| Display points | float32 (~21 k, 3) m, 1.0 s window | same | same |
| Grid | int8 (100, 100), 0.1 m, x −3…7 m, y −5…5 m; −1 unknown / 0 floor seen / 1 occupied | `base_link` | same |
| Elevation | float32 (100, 100) m, NaN unknown | `base_link` | same |
| Sector ranges | (72,) m per 5° CCW from +x; NaN unknown; `sector_clear_to` floor evidence | `base_link` | same |
| Pose / twist | 4×4 `T_odom_base`; twist frame unverified | `odom` ← `base_link` | robot clock |

**Calibrated vs uncertain:** camera intrinsics **measured** (`configs/go2_front_camera_calibration.yaml`,
fisheye, held-out 0.47 px; pass it as `calibration=`); `T_base_camera` **missing**, so
`projected_depth` is `None`. `T_base_lidar` comes from firmware only. Camera latency was
measured once. Odometry accuracy under motion is unverified. The floor-height assumption (−0.33 m)
holds only while standing. The robot clock resets each boot: compare stamps only within one `clock_epoch`.

**Feasible VLN types now:** RGB-only video models producing symbolic or mid-level actions
(NaVid / NaVILA / StreamVLN style), gated by the LiDAR grid. After calibration: RGB + geometric map
(VLFM-style value map on our grid, the preferred approach). RGB-D/3D models need calibration plus
dense depth. See [VLN_MODEL_COMPATIBILITY](VLN_MODEL_COMPATIBILITY.md).

**Model outputs → goals without inventing geometry:** a symbolic action becomes `T_odom_base · T(θ, d)`.
Accept it only if the target cell and corridor are FREE (floor seen) and within `sector_clear_to`;
UNKNOWN is not free. Pixel targets now give a calibrated *ray direction* in the camera frame, but
turning it into a metric goal needs `T_base_camera`, so they are unsupported today. 3D boxes
and model confidence never count as clearance. Nothing here sends commands.

**Reproduce (this Mac):**
```bash
cd /Users/yushan/Desktop/Go2_Navigation-phase5d
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e '.[test,forecasting]'
export PYTHONPATH=src
D=/Users/yushan/Desktop/Go2_Navigation/data
.venv/bin/python -m pytest -q                                    # 381 passed, 3 skipped
.venv/bin/python scripts/phase5d_vln_replay_demo.py --synthetic --output /tmp/vln_demo
.venv/bin/python scripts/phase5d_vln_replay_demo.py --recording $D/recordings/p5d_static_02 --index 200 --output $D/outputs/phase5d/demo
.venv/bin/python scripts/phase5d_lidar_accumulation.py --recording $D/recordings/p5d_static_02 --output $D/outputs/phase5d/accumulation
.venv/bin/python scripts/phase5d_timing_report.py --data $D/recordings --recordings test_c p5d_static_02 --output $D/outputs/phase5d/timing.json
.venv/bin/python scripts/phase5d_make_charuco.py --screen 2560x1600 --output $D/outputs/phase5d/targets
```
Live capture (robot stationary, user present): `.venv-go2` in the main checkout, `--iface en9`,
see [MAC_CONNECTION](MAC_CONNECTION.md).

**Claude/user next:**
1. ~~Camera intrinsics~~ done 2026-11-11: [GO2_CAMERA_CALIBRATION](GO2_CAMERA_CALIBRATION.md).
2. Extrinsics: stopped 2026-11-12 after two failed sessions; read "Session 2" and "What would plausibly
   work next" in [GO2_LIDAR_CAMERA_EXTRINSICS](GO2_LIDAR_CAMERA_EXTRINSICS.md) (≥ 60 × 80 cm target with
   larger squares, poses at one range, plane-level gate, PCA not 3-point RANSAC; or model the LiDAR's
   range bias first).
3. Re-measure camera latency on a new boot (operator-driven yaw).

## Phase 5B → Codex handoff (Claude, 2026-11-11)

- **Branch:** `phase5-fsd-visualization`, based on `be4b9de` (Phase 5A).
- **Do not merge main.** Nothing in `scene/`, any existing config or any default was changed; the only new config is the prepared held-out protocol.
- **Read:** [PHASE5_REAL_VALIDATION](PHASE5_REAL_VALIDATION.md) for the evidence and the fix list;
  [FSD_STYLE_VISUALIZATION](FSD_STYLE_VISUALIZATION.md) for the viewer.

**Fixes for Codex, ordered by evidence (development data A1–C; validate on held-out H1–H3):**
1. **Freshness from recent coverage.** Refresh an object from ≥3 returns in the last 2–3 messages, or
   by swept-sector coverage, rather than the latest ~62 ms message alone. In B, 88 of 91 "absent"
   walking frames have zero person returns in the latest message, yet ≥8 non-ground points at 0.3 s
   (89/91). At the person's bearing the message has no non-floor returns in 94.5% of them: a sampling
   gap, not occlusion.
2. **Motion gate robust to σ.** `speed − 2σ > 0.25` blocks 56 fresh walking frames: median speed
   0.44 m/s, median σ 0.23 m/s. Use a displacement/direction test, `speed − 1σ`, or σ from
   `observed_box` centres.
3. **Inherit motion history** when a reborn object re-associates within 1 s of an expired one (23 B
   frames).
4. **Robot self-filter** in proposal generation (fresh object within 0.45 m of `base_link` in about
   98% of frames).
5. **Leg clustering** (14 B frames left as noise or merged off-centre).
6. **Vectorize `BackgroundMap.stable_mask`.** With `object_window_s=0.3` a replay ran at about
   9–18 s/frame, so the counterfactual was stopped.

**Reproduce (on this Mac):**

```bash
cd /Users/yushan/Desktop/Go2_Navigation-phase5b    # worktree on phase5-fsd-visualization
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e '.[test,forecasting]'
export PYTHONPATH=src
D=/Users/yushan/Desktop/Go2_Navigation/data
O=$D/outputs/phase5b
.venv/bin/python scripts/phase5b_scene_tap.py --input $D/recordings/test_b --output $O/tap_new/test_b   # fresh dir; optional --set key=value diagnostics
.venv/bin/python scripts/phase5b_diagnose_walking.py --recording test_b --data $D --tap $O/tap_new/test_b --output $O/diagnosis_new/test_b.json
.venv/bin/python scripts/phase5b_compare.py --data $D --variant 5A=$O/tap --variant fix=$O/tap_new --output $O/comparison_new.json
.venv/bin/python scripts/fsd_viewer.py --recording test_b --data $D --mode debug --tap $O/tap_new/test_b --scene $O/tap_new/test_b/scene.jsonl
.venv/bin/python -m pytest -q
```

`phase5b_compare.py` expects all four recordings under each variant directory. `phase5b_scene_tap.py`
writes the same `scene.jsonl` as `run_stable_scene.py` plus Debug sidecars, and `--reference` checks
row identity.

**Claude/user next:**
1. Record held-out H1–H3 per `configs/phase5_heldout_protocol.yaml` (robot moved by hand only).
2. Verify `cloud_base` sector coverage on hardware.
3. Calibrate the camera.
4. Build a live read-only viewer with end-to-end latency measurement.

**Limitations:**
- Annotations are not human-verified; everything here is development data.
- The camera sync is approximate.
- The viewer's floor is assumed at robot z − 0.32 m.
- The 0.3 s 5A counterfactual was not completed.

## Phase 5A → Phase 5B Claude Handoff

- **Branch:** `phase5-stable-lidar-perception`.
- **Complete interface commit:** `5e269061c3da70ffe3d00798eb9bf05977869797` (includes initial fixture/schema version commit `9b99375`). This immutable commit contains every renderer-facing interface file; this handoff note is a subsequent documentation commit on the same branch.
- **Contract:** `docs/SCENE_REPRESENTATION_INTERFACE.md`.
- **Versioned machine schema:** `schemas/go2-scene-v1.schema.json` (`go2-scene-v1`, JSON Schema 2020-12).
- **Exact deterministic fixture:** `tests/fixtures/scene/renderer.jsonl` (six JSONL rows, synthetic only).
- **Minimal consumer:** `scripts/read_scene.py`, standard library only; `scripts/generate_scene_fixture.py` reproduces the fixture. Semantic validator: `src/go2_perception/scene/serialization.py`; tests: `tests/test_scene_interface.py` (**8 passed**).

Fetch the branch, then consume the fixture without waiting for perception implementation:

```bash
git fetch origin phase5-stable-lidar-perception
# Read from a separate checkout/worktree to preserve your current annotation/video work.
git worktree add ../Go2_scene_renderer origin/phase5-stable-lidar-perception
cd ../Go2_scene_renderer
python scripts/read_scene.py tests/fixtures/scene/renderer.jsonl
```

All scene geometry is already in `odom`, in meters, right-handed x-forward/y-left/z-up at odometry origin. Timestamps are seconds in the explicitly named clock domain; fixture clock is `synthetic`, real records use `go2_robot`. Row-major `T_odom_base` maps base column vectors into odom: apply it only to the robot mesh, never again to the scene points/boxes. Clear renderer history on epoch changes. Null pose is unknown, not identity. Preserve observed-vs-filtered-vs-stale and UNCERTAIN state; absence/expired cells are unknown, not free space. The last two fixture frames deliberately mark the static box stale while the second object remains observed/MOVING.

`display_compact` controls drawing only; raw measurements must still enter perception. Scores are defined evidence/recency ratios, not probabilities. This delivery contains no camera calibration, private recordings, control commands or promise of real geometry accuracy. Claude may implement the renderer against v1 immediately. Codex will continue Phase 5A on the same branch, preserving required fields and semantics. Additive optional fields are allowed; a breaking change requires explicit coordination with Claude and a new version. User selected GitHub branch publication as the delivery channel; Claude acknowledgement remains pending.

## Phase 5A perception continuation

Implementation and experiments are complete as an **experimental** opt-in pipeline, not a validated perception upgrade. See STABLE_LIDAR_ARCHITECTURE.md and STABLE_LIDAR_EXPERIMENTS.md before consuming real output. `StableScene.process` accepts raw Nx3 base_link points, seconds, unique interval IDs, same-clock pose lookup and optional per-point second offsets. It emits the unchanged published v1 contract. Do not retransform scene geometry or feed accumulated cloud_deskewed data. Reset renderer history on epoch changes; stale boxes must stay visibly stale/UNCERTAIN.

- Run: `PYTHONPATH=src .venv/bin/python scripts/run_stable_scene.py --input /path/to/recording --output data/outputs/scene_trial`. The output directory must be new and gitignored. No annotation input is accepted by runtime.
- Synthetic comparison: `PYTHONPATH=src .venv/bin/python scripts/benchmark_stable_scene.py --output data/outputs/scene_synthetic_trial`.
- Tests: `PYTHONPATH=src .venv/bin/python -m pytest -q`: **310 passed, 1 hardware skipped**. Full Ruff has 33 inherited hardware-script findings; Phase 5A scoped lint passes. Exact setup/report commands are in STABLE_LIDAR_EXPERIMENTS.md. New jsonschema test dependency is pinned to 4.26.0.
- Baseline comparison and aggregate evidence: STABLE_LIDAR_RESULTS.md / STABLE_LIDAR_METRICS.json. Real Phase 5A moving recall A2=0.121, B=0.058; fresh box coverage is much lower than original. Roughly 69% of emitted boxes are stale. Do not promote defaults or describe smooth geometry as reliable pedestrian perception.
- Source ownership: Codex added `scene/`, stable scene scripts, config and tests. Claude retains Go2 access, timing verification, annotation review and renderer. Existing tracker/adapter/forecasting code was not rewritten. No raw private data was committed.
- Next steps: consume the fixture on the branch; verify stale/epoch rendering; human-review development annotations; collect held-out scenes including nearby/occluded people; verify instantaneous point-frame semantics before enabling deskew; improve proposal coverage and compare observed-only metrics before changing operational settings.
- Known failure reproduction: run synthetic `near_furniture` through the benchmark (moving recall 0.189 at 0.1 s, 0 at 1.0 s). Replay A1/A2/B/C locally with the documented evaluator to reproduce low fresh coverage and low moving recall. Missing odometry emits empty geometry; weak evidence emits UNCERTAIN. Covariance is heuristic, not a calibrated safety bound. No live/held-out Phase 5A validation has been performed.

# Agent handoff — 2026-11-09 (macOS update)

## Phase 3.8 visual ground-truth handoff (integrated history)
Branch `phase3-visual-groundtruth`. It contains `79bb0f9` (ground truth) and the merged Phase
4.1 branch `5979a76`. Do not merge main or change defaults without approval. Read
[VIDEO_ANNOTATION](VIDEO_ANNOTATION.md), then [TRACKING_GROUND_TRUTH](TRACKING_GROUND_TRUTH.md).

**Code added:**
- `visualization/diagnostic_video.py`: rendering, camera alignment, trails, heuristic events,
  H.264 writer, private-path guard.
- `scripts/render_tracking_video.py`: single, comparison and forecast-overlay videos, stills,
  event logs.
- `scripts/render_highlights.py`: the captioned montage.
- `tracking/annotation_edits.py` and `scripts/annotation_editor.py`: manual corrections as
  review entries, CSV round trip, `identity_uncertain`.
- `scripts/forecast_ground_truth.py`: 8+12 windows at 0.4 s from validated trajectories,
  through the Phase 4 adapter checks; CV only.
- 13 synthetic tests.

**Videos are private and local:** `/Users/yushan/Desktop/Go2_Navigation/data/outputs/mac/videos/`.

**For Codex:**
- Use the videos to inspect cluster merging. Most red events coincide with the box or the
  person merging into a background cluster.
- The proposal-level work from the ground-truth handoff remains the priority.
- Do not tune on A1, A2, B or C.

**For Claude and the user:**
1. Human review in the editor (A2, B, C first).
2. Record held-out H1–H3.
3. Place the Phase 4 / 4.1 checkpoints under `data/outputs/forecasting/` and add learned
   predictions to `forecast_ground_truth.py`, through the existing adapters only.
4. Camera calibration, before any camera↔LiDAR association.

**Forecasting blocker:** only 6 independent identity-validated windows (one person, B and A2).
That supports plumbing checks only.

## Phase 3.8 ground-truth handoff (2026-11-08)
Branch `phase3-ground-truth-validation`, from `c643422`. Do not merge main or change the default
without approval. Read [TRACKING_GROUND_TRUTH](TRACKING_GROUND_TRUTH.md) first. The Phase 3.7
section below is historical.

**What exists now:**
- An annotation workflow, `scripts/annotate_ground_truth.py` (`cache` / `propose` / `sheet`). It
  uses LiDAR background subtraction, a measured static footprint and LiDAR-defined visibility,
  and the camera only for review.
- The schema and metrics, `src/go2_perception/tracking/ground_truth.py`: CLEAR-MOT-style matching
  that keeps the previous match, IDF1, IDF1 per visit, switches, fragmentations, duplicates,
  velocity error, motion confusion and recovery after occlusion.
- `scripts/evaluate_ground_truth.py`: scoring, proposal-quality and ID-change diagnosis, and an
  oracle-segmentation diagnostic.
- Annotations for B, A2, C and A1. They are local only, in `data/annotations/`; the specs there
  record intervals and review notes.
- 11 synthetic tests.

**Measured, original → 3.6 → 3.7:**
- IDF1: B 0.33/0.40/0.39, A2 0.30/0.60/0.53, C 0.29/0.60/0.41, A1 0.90/0.90/0.86.
- 3.6 is the best identity profile; 3.7 has fewer duplicates but lower coverage and more A1
  flicker.
- None is default-ready. Keep `tracking.yaml` as the default.

**Bottleneck: proposals.**
- Box merged with background in 14–33% of visible frames, and with the person in another 12%
  in B.
- Person merged in 23% of frames in A2.
- Splits are rare. Odometry is ruled out as a major cause in C.
- **Oracle segmentation diagnostic.** The untuned original tracker on oracle proposals reaches
  IDF1 B 0.67, A2 0.69, C 0.54 and A1 0.98.
  - That beats every tuned profile on real proposals in B, A2 and A1.
  - 3.6 and 3.7 do worse than the original on clean proposals; for example, A1 3.6 has 44 flicker
    switches.

**Next for Codex.** Improve LiDAR proposals before more tracker tuning:
1. Separate low objects from walls and counters: height-profile or background-aware splitting,
   a smaller or adaptive `eps` near large planar clusters.
2. Separate a person from an adjacent object.
3. Then re-check the tracker profiles on the improved proposals. On clean proposals the
   original tracker beats 3.6 and 3.7.
4. Do not tune on A1, A2, B or C again. They are development data now.

**Next for Claude and the user:**
1. A human pass over the review sheets (time ranges listed in the report).
2. Record held-out H1–H3 (two people, a taped start/end pose), annotate with unchanged
   thresholds, and score once.
3. Camera calibration, to enable semantic gating.

## Phase 4.1 forecasting handoff (merged from `phase4-social-forecasting` at 5979a76)

Branch `phase4-social-forecasting`, based on verified Phase 4 `447f164e808fee381fd3e9dbe8244583a0bed2f4`. No merge into main. Implemented observation-only multi-pedestrian extraction with partial-history masks, an 87,128-parameter shared temporal encoder with relative-position/mean neighbor pooling, dedicated checkpoint/training/evaluation interfaces, robustness studies, research plots, and a guarded offline multi-track adapter. Original forecasting, TrackOutput/ForecastOutput, tracking internals and Go2 hardware adapters are unchanged.

Completed: all five scene-identifier-held-out folds for original and social models with seeds 42/43/44 (15,061 paired windows per seed), same-weight social-disabled diagnostics, twenty validation robustness cases across clean/joint/noise-only/mask-only training, and separate CPU latency measurement. Every single-agent prediction and target/key population was checked against the independent Phase 4 evaluator. Full suite: **259 passed, one hardware skipped in 17.37 s**; Ruff passes on forecasting code/scripts/tests. No dependency changes (Python 3.12.11, torch 2.6.0, numpy 1.26.4, matplotlib 3.11.2).

**Negative result:** macro ADE/FDE, mean ± sample SD over three seeds: social 0.5812 ± 0.0162 / 1.1951 ± 0.0247 m; original 0.5565 ± 0.0118 / 1.1534 ± 0.0200 m; CV 0.5214 / 1.0933 m. Disabling social input improves macro errors in every seed. Noise-only augmentation improves one-fold noisy validation ADE to 0.3924 m, still behind noisy CV 0.3371 m. No claim of overall learned-model improvement or independent-window statistical significance.

Read [social architecture](SOCIAL_FORECASTING_ARCHITECTURE.md), [actual experiments and commands](SOCIAL_FORECASTING_EXPERIMENTS.md), [predeclared protocol and follow-up controls](SOCIAL_FORECASTING_PROTOCOL.md), [metrics](SOCIAL_FORECASTING_METRICS.json), and [Go2 contract](FORECASTING_INTEGRATION.md). Published artifacts are public-data aggregates/selected plots; raw annotations, weights, predictions and private recordings remain ignored. Known limitations: weak mean pooling, one-seed robustness controls, dependent windows, scene/domain shift, unverified dataset redistribution rights, no clean-install/GPU/live-Go2 validation. No navigation control.

Next: keep CV as the reference; Claude validates real pedestrian identity segments and stable metric frames. Then run matched held-out offline Go2 comparisons with explicit externally validated IDs. Repeat noise-only augmentation across seeds before considering more architecture. Historical phase states follow; their current/next claims apply to their original milestones.

# Phase 4 historical state — 2026-11-09

Branch `phase4-trajectory-forecasting`, based on Phase 3.7 `c6434229c54ee445763d8ceb1f5ef56b0f5c50da`; main is not merged. Added public ETH/UCY trajectory preparation, independent CV and Phase 3 baseline adapter, lightweight PyTorch Transformer, deterministic training/checkpoints, paired evaluation, validation ablations, plotting and guarded offline TrackOutput integration. Tracking/sensor internals remain unchanged; Claude's Phase 3.8 work is separate.

Verified: five fixed scene-identifier-held-out CPU runs (15,061 windows), validation-only history/capacity/representation/noise/missing-input studies, synthetic adapter demo and 231 tests passed/1 hardware skipped. The learned model **loses overall**: macro ADE/FDE 0.5701/1.1762 m vs CV 0.5214/1.0933 m. No Go2 forecasting accuracy or GPU validation is claimed. Data/weights remain local, public aggregate results and selected plots are committed. Python 3.12.11, torch 2.6.0, numpy 1.26.4, optional matplotlib 3.11.2; existing core pins unchanged.

Read [forecasting architecture](FORECASTING_ARCHITECTURE.md), [dataset protocol](FORECASTING_DATASETS.md), [commands and actual experiments](FORECASTING_EXPERIMENTS.md), and [integration contract](FORECASTING_INTEGRATION.md). Known limitations: one seed, correlated windows, no social context or uncertainty, unverified dataset redistribution rights, no clean-install test, and untrusted real tracker identities. Do not train from unvalidated Go2 tracks. Next: Claude validates identities/frames/visibility, then evaluate public-trained models and CV on held-out ground truth; extend forecasting research using validation-only decisions. Earlier phase sections below are historical.

## Phase 3.7 refinement handoff (historical)
Branch `phase3-tracking-refinement` starts at `de043d1`, containing Claude's real comparison and
Codex's Phase 3.6 work. Do not merge main or promote a profile to default without approval.
Read [TRACKING_REFINEMENT](TRACKING_REFINEMENT.md) first, then the unchanged
[TRACKING_REAL_COMPARISON](TRACKING_REAL_COMPARISON.md) for the regression diagnosis.

New code constrains reactivation by last observation/time/speed; suppresses repeatedly overlapping,
velocity-consistent weaker tracks; and separates moving activation/release from stationary dwell.
Use `configs/tracking_refinement.yaml`; both earlier profiles remain available. Public sensor and
tracking interfaces are unchanged. Source tests: 192 passed/1 hardware skipped. Seed 42's split
failure persists; first-miss Lost status reduces occlusion IDF1, and irregular-motion recall has small regressions. Dependencies unchanged.

The exact `compare_tracking_real.py --reuse-cache --serial` command is in TRACKING_REFINEMENT.
Use its new read-only --cache-root and a fresh output directory: existing run outputs now refuse
overwrite; --analyze reads them. Explicit PYTHONPATH=src avoids this Mac's stale installed package.
No private samples are committed. Claude owns hardware/calibration verification and annotation;
next label identities/visibility, record two close people and validate transitions on held-out data.
Do not start learned forecasting using these unverified identity targets. Earlier handoffs below
are historical; their 'next' tasks should be read in that context.

Final offline replay: 3,924 frames; original/default outputs remain identical. Walking MOVING
proxy improves about 39%→52%, and long reactivations fall in all recordings. Much of the duplicate
count reduction is removal of coasting Confirmed states; observed-only duplicates barely change.
A2/C dominant-ID continuity regresses relative to Phase 3.6, and A1 flicker increases. This is
not a uniform fix or a default-ready profile; labels/held-out testing remain required. Full measured
results and runtime are in TRACKING_REFINEMENT. No private samples are committed.

## Phase 3.6 robustness handoff (historical)
Branch `phase3-tracking-robustness` starts at `feature/macos-go2-phase3-validation` / `ecfaae3`.
Do not merge main without approval. Read [TRACKING_ROBUSTNESS](TRACKING_ROBUSTNESS.md) first,
then [TRACKING_ARCHITECTURE](TRACKING_ARCHITECTURE.md) and Claude's unchanged
[PHASE3_REAL_VALIDATION](PHASE3_REAL_VALIDATION.md). The previous macOS handoff below is historical.

Codex added opt-in `configs/tracking_robustness.yaml`, conservative historical-box fragment
recovery, time expiry for confirmed tracks, adaptive gating, motion dwell, lifecycle diagnostics,
26 regressions and `scripts/benchmark_tracking_robustness.py`. No sensor/hardware code changed.
`PYTHONPATH=src .venv/bin/python -m pytest -q`: **161 passed, 1 hardware skipped**.
Always set PYTHONPATH on this Mac to avoid the old installed package; `.venv-go2` can be used
instead. Install instructions and pinned dependencies remain in SETUP.md.

Use the exact synthetic/private-replay commands and aggregate comparisons in TRACKING_ROBUSTNESS.
Private cache and per-frame results live under ignored `data/outputs/robustness/`; no recordings
are needed for the synthetic benchmark. Seed 42 still has three split-object switches; brief-stop
classification recall regresses. Baseline defaults are deliberately unchanged.

Claude owns physical sensor access, timing/calibration validation and labelled real evaluation.
Next: annotate A2/B/C locally, inspect recovery/association failures with raw proposals, run D
(two nearby people), measure slow-motion/stop transitions and pose drift. Keep private sensor
files out of Git. Preserve pose-gate skips/resets. Integration continues through Detection3D,
RobotPose and the unchanged tracker.update signature; no detector internals need modification.
Phase 4 dataset tooling can use current outputs, but do not treat unstable IDs as reliable learned
forecasting targets. Hardware live validation and calibrated RGB semantic fusion remain outstanding.

**Real-data comparison done (Claude):** [TRACKING_REAL_COMPARISON](TRACKING_REAL_COMPARISON.md),
harness `scripts/compare_tracking_real.py`, local outputs in `data/outputs/mac/tracker_comparison/`.
The default path is bit-identical to `ecfaae3`; the improved profile is ready as opt-in, not as
default. Ablations attribute the effects:
- **Time expiry** gives the box-continuity gain, but also the extra duplicates and flicker.
- **The 2.5 m adaptive gate** causes most reactivations more than 1 m from the last observation.
- **Dwell plus the 0.25 m/s threshold** removes false MOVING labels, but also cuts walking MOVING
  recall from 62% to 39%.

Next for Codex:
1. Suppress or merge a Lost track that overlaps a Confirmed track.
2. Bound the reactivation radius by elapsed time × a plausible speed.
3. Re-tune dwell for walking people.
4. Rerun `compare_tracking_real.py --reuse-cache --serial`. It needs the local recordings, so it
   runs on the Mac.

Next for Claude/user: hand-label box and person IDs on A2/B/C so these proxies can be checked.

## macOS validation handoff (historical baseline)
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

**Next for Codex** (evidence: PHASE3_REAL_VALIDATION, recordings `test_a1`, `test_a2`, `test_b` on
the Mac): (1) person identity: one walking person became 19 IDs; examine leg clustering and
coasting; (2) occlusion and viewpoint: a static box is re-identified each time a person stands in front,
and every 5-15 s while the robot turns (Test C, 22 IDs) even though its odom position holds to 6-9 cm;
(3) motion classifier: steady velocity sigma about 0.14 m/s makes STATIONARY need speed under about
0.01 m/s; (4) clutter fragments on walls give false MOVING. Keep synthetic metrics intact and get
labels before tuning. **Next for Claude/user:**
Test D (two people), a Test C repeat with a taped end pose for odometry drift, Mac
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

## Phase 4 continuation checklist

1. Use the `phase4-trajectory-forecasting` branch and read FORECASTING_EXPERIMENTS.md for the complete installation/reproduction commands. Run the synthetic smoke configuration before downloading annotations.
2. Codex owns `src/go2_perception/forecasting/`, `scripts/*forecasting*.py`, forecasting configs and tests. Core tracking and Go2 adapters were not rewritten. Claude retains sensor access, real calibration, ground truth and hardware validation.
3. For a real offline trial, establish stable metric odometry coordinates and reliable pedestrian IDs, retain at least 2.8 seconds of observed history, then use the explicit bounded resampling adapter. Keep all real recordings local.
4. Compare CV and learned forecasts on the same manually validated held-out windows. Never treat the public-data results or synthetic demo as proof of Go2 forecasting accuracy. Resolve fragmentation/ID switches before collecting training labels.
5. Reproduce known limitations with the documented five-fold command (model loses overall) and the noise ablation (model error rises substantially). Unknown data redistribution rights, GPU execution and clean-install portability remain unresolved. Do not merge into main or promote the learned model to navigation without separate approval and validation.

## Phase 4.1 continuation checklist

- Code belongs to the independent Phase 4.1 branch; do not merge main without approval. Claude's `phase3-ground-truth-validation` checkout and sensor/annotation responsibilities are separate. No hardware or tracker internals were modified.
- Start with SOCIAL_FORECASTING_EXPERIMENTS.md for exact install/reproduction commands. Reproduce Phase 4 checkpoints first; then use `run_social_forecasting.py --stage single`, `--stage social`, `--stage robustness`, and `--stage robustness_controls`. Use fresh output directories. Summary auditing requires local prediction NPZ files from both runs.
- Use `social_training.load_social_checkpoint` for social weights. Existing Phase 4 weights/API remain supported by the original loader. `forecast_multitrack` returns the unchanged ForecastOutput plus explicit context/masks/rejection reasons. Pass checkpoint social settings and externally reviewed `validated_ids` for the current snapshot; never infer reliable pedestrian identity from anonymous clusters or Confirmed status alone.
- The model loses overall. Reproduce the five-fold/three-seed comparison and noisy validation table before making a research claim. Do not choose a winning seed or tune against those held-out results. A future experiment needs a declared validation-only decision process.
- Physical Go2 identity quality, camera–LiDAR calibration, GPU execution and clean-install portability remain unresolved. Complete Claude's ground-truth work before domain-transfer evaluation; do not train on unannotated tracks or issue movement commands.
