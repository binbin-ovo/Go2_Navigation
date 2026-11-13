## Phase 6C — navigation-trained checkpoint and small-scene experiment (2026-11-13)

**Branch:** `phase6-pretrained-vln`; isolated from verified Phase 6B
`7e2989ad54b9c484d75c029ff0141834880852b6`. Main, original worktrees, Claude's recording
and calibration environment, Phase 5A.1, Grounding DINO, SmolVLM experiments and
operational defaults are preserved. No robot connection, commands, private upload,
training, Windows driver/WSL change or simulator replacement.

**Implemented and actually executed:** selected one navigation-trained model,
LightOriginsHQ/LightNav-0 revision `826dc5fbfa37afa8293d2e336d329b6ffc0bfb64`, official
code `3015508b70fb6e30bbd66701e94c9b916dcb8df8`. Synthetic official-CLI inference and
11 Claude-package decisions ran on Apple M3 Pro MPS, BF16/SDPA, no quantization. Two
small device/timing changes plus a bit-tested CPU adaptive-pooling fallback are an
explicit local portability fork. Exact native RVQ chunks, pointing, source images,
history/tensors, errors, latency and memory are retained. Separate Python environments
preserve the existing perception installation; model versions are pinned in
`requirements-pretrained-vln-macos-tested.txt`.

**Research outcome:** 11/11 native RVQ decodes; nine discretely comparable outputs,
five human-reference matches, four mismatches and two unsupported mixed chunks. On
the seven comparable E1–E3 cutoffs: LightNav 3/7, saved SmolVLM temporal 4/7,
current-only 5/7. Initial native left/right yaw directions improve over the temporal
baseline, but **no overall gain is established**. No STOP at any cutoff. E4 emits a
not-visible token plus a search turn; search agreement is not absence/task success.
One correlated test_c development group, retrospective E1–E3 and static E4; historical
prompt/reference mismatch and all old scores retained. New semantic review is pending.

**Simulation:** one MuJoCo 3.3.7 camera room renders on the Mac. A 16-step static
step-and-wait closed loop actually ran and **failed**: repeated turning, then movement
away from the doorway, no stop, distance increased from 1.850 to 2.722 m. All failure
panels/video are retained, including public synthetic examples. This is a kinematic
camera proxy; no Go2 locomotion or collision-avoidance validation. Windows RTX 2070
Super / installed Isaac access remains pending the supplied read-only preflight.

**Resources:** private native generation mean 2.905 s; replay image age mean 3.369 s
before evidence serialization. Peak process RSS 11.055 GiB; sampled MPS driver peak
10.982 GiB (overlapping unified memory, not additive). Mac model+small-renderer peak
RSS 11.059 GiB. The 9.03 GiB BF16 weights alone exceed an 8 GB GPU; no supported
quantized/offloaded Turing path was established. Standard FlashAttention-2 is not assumed.

**Verified tests:** full suite **541 passed, 4 skipped (41.58 s)**, including cached
real SmolVLM inference and 63 new contract cases. The optional native reset/pooling
and MuJoCo geometry/contact/camera tests pass separately in the isolated runtime;
the physical Go2 test stays disabled. Source-preservation details and commands are in the
[experiment report](PRETRAINED_VLN_EXPERIMENTS.md). Intermittent local file-read stalls
were observed. One initial baseline-PNG hash was an invalid empty read; the untouched
PNG matches the original producer in memory, and strict repeat inventories agree on
all 727 files. The failed initial digest is retained, not silently repaired.

**Read first:** [handoff](AGENT_HANDOFF.md), [model feasibility](PRETRAINED_VLN_FEASIBILITY.md),
[experiments/reproduction](PRETRAINED_VLN_EXPERIMENTS.md),
[Windows feasibility](WINDOWS_SIMULATION_FEASIBILITY.md). All private artifacts remain
under `/Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6c/` and outside Git.
Next: independent review of native chunks/target visibility and stopping failures,
diverse separately annotated episodes, then the actual Windows preflight. Do not tune
or claim generalization on this single recording; do not connect proposals to Go2.
Earlier sections below retain their historical milestone scope.

## Phase 6B — Claude's prepared pilot evaluated (2026-11-13)

Evaluation implementation commit: [`4d5e49642da03558830dba017f5aa150d9fdc684`](https://github.com/binbin-ovo/Go2_Navigation/commit/4d5e49642da03558830dba017f5aa150d9fdc684).

On `phase6-temporal-vln`, the exact four user-supplied packages from Claude's
`phase6-vln-episode-validation` commit `3a84f57b8753b0e5c169f53ae361a33f9ddbc3fe`
were validated and processed unchanged. All supplied manifest/annotation hashes
match; **434 package files remained byte-identical**. No rebuilding, relabeling or
substitution. Main, Claude's checkout, calibration and operational defaults are unchanged.

**Verified:** 399 source frames, 11 decision times, 22 actual SmolVLM2-2.2B CPU calls
with the same pinned model/prompt/parser/preprocessing as the initial pilot. Each
temporal call encoded four causal images; exact message/tensor/frame audits excluded
future observations, evaluation labels and operator odometry. All **11 action
references are human-reviewed**, accepted by one owner from Claude drafts; seven
are ambiguous. No action-reference evaluation remains pending for this follow-up.

**Reference agreement:** E1–E3 retrospective navigation proposals: **5/9 temporal vs
7/9 current-only**. E4 static absent-target probe: **2/2 vs 0/2**, the latter two outputs
malformed and rejected. Overall 7/11 in both modes is descriptive of **one correlated
test_c group**, not independent episodes or navigation success. No temporal advantage
is established. Both modes miss the provided terminal stopping references. A temporal
COMPLETE claim occurs where human completion is unknown, so it remains unverified.

**Tests:** full suite including actual pretrained tests **478 passed, one hardware
skipped**; four changed/new Python files pass Ruff check/format. New CLI accepts
repeated exact `--episode` paths. New `scripts/evaluate_temporal_vln.py` scores saved
outputs with manifest binding, human-only references, recording grouping and separate
static/ambiguity strata. Model/sensor/core schema code is unchanged.

**Runtime:** mean generation 11.125 s temporal / 5.557 s current-only; median 9.191 /
5.463 s. Estimated replay image age at output mean 11.381 / 5.720 s. Peak shared
process RSS 9.727 GiB. Slow initialization and first forward are documented, not hidden.
Private results and readable 22-frame video:
`/Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6b/claude_pilot_v1/`.

**Read first:** [Claude-package report, commands and limitations](TEMPORAL_VLN_CLAUDE_PILOT.md),
[handoff](AGENT_HANDOFF.md), [stable episode contract](VLN_EPISODE_INTERFACE.md).
The earlier Codex pilot below is a separate historical experiment; its pending labels
were not replaced by these packages. Next: diverse independently reviewed episodes
with explicit completion criteria; metric geometry and closed-loop behavior remain
unvalidated. No robot connection or command was enabled.

## Phase 6B temporal VLN — completed offline pilot (2026-11-12)

Implementation commit: [`e9fa7a99c0182af150ffaccc58a83ef54f1a6b0d`](https://github.com/binbin-ovo/Go2_Navigation/commit/e9fa7a99c0182af150ffaccc58a83ef54f1a6b0d).

Branch **`phase6-temporal-vln`**, based on verified Phase 6A
`c13f1dc442fa157b31043613d3cb2b0c30ceab8a`. Contract was published first at
`5c7f586aa2536a8eda9167e66d5050a4a91dbac2`; v1 schemas/fixture remain compatible.
No main merge, hardware connection/control, shared-environment change, calibration
change, operational-default change or Phase 5A.1 modification.

**Implemented and verified:** versioned episode + separate annotation contracts,
causal exposure/availability-aware history, measured calibration on each image,
official multi-image preprocessing with complete provenance, pinned pretrained model,
conservative native-text parsing, evidence panels/video and human-reference-only
metrics. Actual SmolVLM2-2.2B inference ran locally on private Go2 RGB: four episodes,
eight cutoffs, both current-only and temporal conditions, **16 calls**. Every temporal
forward encoded four images (three history + current), not just a stored history list.

**Result:** 16/16 outputs parsed, but all eight temporal proposals were TURN_RIGHT,
including explicit left-turn instructions. Single-image outputs were TURN_RIGHT ×4,
TURN_LEFT ×2, UNRESOLVED ×2. Agent review found instruction-direction failures,
undemonstrated grounding and an unsupported completion claim in native prose despite
IN_PROGRESS in the structured field. **Human action/progress/completion evaluation
remains pending.** No reliable instruction following or temporal benefit is established.
This is a **VLM navigation-proposal baseline**, not a reproduced VLN policy.

**Validation:** final full suite including actual pretrained processor/model tests:
**470 passed, 1 hardware skipped (36.12 s)**. Ruff check/format passed on 12 Phase 6B
Python files. Existing Grounding DINO real-image smoke reproduced its prior label,
score and box exactly in the new environment. A separate environment was installed;
all 88 installed packages pass dependency compatibility checks. No existing tests
were weakened. Published synthetic fixture/schema remain unchanged.

**Runtime:** `HuggingFaceTB/SmolVLM2-2.2B-Instruct` and processor revision
`be92141c8d0ceb98596a923c308ac2bbb4a7c466`; CPU FP32/eager, four threads, seed 0,
384px unsplit multi-image input. Python 3.12.11, Torch 2.6.0, Transformers 4.51.3,
HF Hub 0.30.2; exact constraints in
[requirements-temporal-vln-tested.txt](../requirements-temporal-vln-tested.txt).
Generation mean 10.171 s temporal vs 6.221 s single; replay image age at output mean
10.848 vs 6.791 s. Process lifetime peak 9.684 GiB (not GPU memory or a per-mode measure).

**Known limits:** same-home correlated pilot, no human action labels, no closed-loop
outcomes, CPU latency, stretched 384×384 model images, estimated exposure/availability,
missing T_base_camera/aligned depth, unvalidated moving odometry and traversability.
No-image decisions fail explicitly. Model-native action frame is unknown; no distances
or metric goals invented. Every result has execution_authorized=false. STOP is not
completion. Zero structured COMPLETE fields does not rule out false claims in prose.

**Read:** [architecture/model audit](TEMPORAL_VLN_ARCHITECTURE.md),
[actual experiments/failures](TEMPORAL_VLN_EXPERIMENTS.md),
[installation/demo](TEMPORAL_VLN_DEMO.md), [episode interface](VLN_EPISODE_INTERFACE.md),
[current handoff](AGENT_HANDOFF.md). Persistent private evidence:
`/Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6b/`.

**Next:** Claude/human reviewers prepare reviewed episode annotations without changing
model inputs; evaluate held-out diverse environments and investigate the right-turn
bias in a new declared experiment. Metric grounding needs measured registration,
depth/timing/odometry validation and independent geometry checks. Navigation outcomes
need a separately authorized controlled closed-loop experiment. Earlier sections
below are historical milestone reports, not the current branch's full test count.

## Phase 6A calibrated RGB grounding — 2026-11-12 (Codex)

Implementation commit: [`dcc5de1c0d899d5fb8159c5f6b0905ab67b8cf01`](https://github.com/binbin-ovo/Go2_Navigation/commit/dcc5de1c0d899d5fb8159c5f6b0905ab67b8cf01).

Branch `phase6-vln-input-validation`, based on verified calibration commit
`e4e0d3b01a16a14f52f7d039b27009ed3e0e0f96`. Separate worktree/environment; no main
merge, no changes to Claude's checkout, Phase 5A.1 recovery branch, sensor defaults
or robot control. The base contains 5A/5B but not the separate 5A.1 changes.

**Implemented and exercised:** measured fisheye → pinhole preprocessing with new K,
source image/time/history preservation, exact official processor resize/padding K,
optional pinned Grounding DINO Tiny CPU adapter, proposal-only image boxes, terminal
instructions, recorded-observation replay, raw/parsed outputs and RGB/LiDAR review panels.
`go2-vln-obs-v1` is unchanged. Missing extrinsics are never replaced with identity.

**Verified:** clean Python 3.12 environment installation; 12 actual pretrained inference
runs on two private Go2 views; NPZ replay has identical raw predictions; unsupported
navigation command stays unresolved. **421 tests passed / 1 live-hardware test skipped**.
Model: `IDEA-Research/grounding-dino-tiny` at
`a2bb814dd30d776dcf7e30523b00659f4f141c71`; Torch 2.6.0 / Transformers 4.49.0.
CPU mean forward 1.959 s, process peak 3.003 GiB (not a real-time benchmark).

**Limitations:** visual case review found 6 acceptable, 1 partial, 5 failed; human review
pending. False positives on absent targets, visible-object misses, overlapping boxes,
and noun-phrase-only input prevent a navigation-policy claim. This is not learned
relational reasoning or metric grounding. T_base_camera and aligned depth are still
unavailable; exposure times estimated, moving odometry unverified, floor seen not
traversable. No live inference/control test was attempted.

Read [validation and failures](VLN_INPUT_VALIDATION.md), [model audit](VLN_MODEL_SELECTION.md),
[installation and demo](VLN_OFFLINE_DEMO.md), and the Phase 6A handoff below
[AGENT_HANDOFF](AGENT_HANDOFF.md). Next: human/held-out grounding review and separately
validated registration/range before any metric or collision-aware planning work.
This Phase 6A section retains its historical results; the Phase 6B section above
describes the current state of this branch.

## Phase 5D VLN sensor calibration and interface — 2026-11-11 (Claude)

Branch `phase5-go2-vln-sensor-calibration`, from Phase 5B `89d901dc` (contains 5A `be4b9de`). Main is
not merged; Codex's `phase5-detection-recovery` is untouched; no 5A config or default changed. Physical
session on the Go2 with the robot stationary, read-only subscribers only, no control.

**Done:**
- Live re-verification of link, discovery and every stream; 20 s stationary recording `p5d_static_02`
  (lossless PNG RGB).
- LiDAR accumulation study. The user chose **1.0 s display / 0.5 s obstacle** windows
  (`configs/go2_vln_sensors.yaml`).
- Camera audit: H.264 High, no B-frames, 1280×720, no colour metadata, no published intrinsics.
- Timing: the robot clock **resets each boot**. Host − robot offset is estimated per recording (~10 ppm
  drift). Camera receipt latency is **48 ms** (26–58), measured by yaw cross-correlation on `test_c`.
- New package `go2_perception.vln`: `Go2VLNObservation` (`go2-vln-obs-v1`) with visual / geometric /
  robot state, explicit clocks, frames and calibration status; replay from recordings; conservative
  grid (unknown / floor seen / occupied), 2.5D elevation, sector ranges; npz serialization; validation;
  tensor/resize helpers that carry intrinsics; synthetic fixture; 30 tests.
- ChArUco targets (print PDFs and screen PNGs), `scripts/phase5d_calibrate_intrinsics.py` (held-out,
  leave-one-recording-out, bootstrap, corner-validity checks) and the extrinsics procedure.
- VLN compatibility report (RGB-only, RGB + map preferred, RGB-D).

- **Camera intrinsics measured** (three ChArUco passes, robot stationary): fisheye, fx/fy 831.4/833.7 px,
  held-out RMS 0.47 px, ≈ 94° × 50° FOV → `configs/go2_front_camera_calibration.yaml`.

**Not done:** LiDAR–camera extrinsics. Two capture sessions (2026-11-11/12) failed the held-out gate
(best 11° / 75 mm); the LiDAR returns a flat floor bent by a few cm with range, the per-point noise on
the board is ~25 mm and the A4 grid reads only to ~1.1 m. Projected depth and fusion remain unavailable
on real data.

Read [GO2_VLN_SENSOR_VALIDATION](GO2_VLN_SENSOR_VALIDATION.md), [GO2_VLN_INPUT_INTERFACE](GO2_VLN_INPUT_INTERFACE.md),
[VLN_MODEL_COMPATIBILITY](VLN_MODEL_COMPATIBILITY.md), [GO2_SENSOR_TIMING](GO2_SENSOR_TIMING.md),
[GO2_CAMERA_CALIBRATION](GO2_CAMERA_CALIBRATION.md), [GO2_LIDAR_CAMERA_EXTRINSICS](GO2_LIDAR_CAMERA_EXTRINSICS.md).

Tests: 386 passed, 3 skipped.

## Phase 5B FSD-style visualization and real validation — 2026-11-11 (Claude)

Branch `phase5-fsd-visualization`, from Phase 5A `be4b9deb3d3f0f8fc7e111c45464a1922db761ef`, which contains
`phase3-visual-groundtruth`. Main is not merged. No default changed; no robot control.

**Built:**
- A go2-scene-v1 viewer: a NumPy/OpenCV software 3D renderer. Open3D offscreen fails on macOS, and
  rerun has no headless export.
- Clean, Hybrid and Debug modes with keyed layers; split view with recorded RGB.
- Causal playback with optional delayed pose interpolation, and a smoothed camera.
- An instrumented replay of the unchanged 5A pipeline for Debug layers: rows match the 5A release
  exactly on all 3,926 scans.
- Walking-person failure attribution, original-vs-5A comparison, reels.
- Planner-readiness queries (occupied or unknown, never free).

Read [FSD_STYLE_VISUALIZATION](FSD_STYLE_VISUALIZATION.md), then
[PHASE5_REAL_VALIDATION](PHASE5_REAL_VALIDATION.md).

**Results:**
- 5A removes almost all false MOVING: unsupported MOVING frames/min fall from about 600–840 to 0–5.
  Static-box step p95 falls from 51–69 mm to 20–38 mm.
- Fresh coverage and moving recall regress: B person 0.905/0.575 → 0.499/0.058. Median MOVING delay
  rises from 0.23 s to about 1.35 s.
- 69% of emitted boxes are stale; much of the steadiness is held stale geometry.
- Test B diagnosis: 38% of walking frames have **no person returns in the latest ~62 ms scan**,
  although 0.3 s windows contain the person. 5A's freshness rule (≥3 latest-message points) turns this
  sampling gap into stale boxes.
- Fresh but not MOVING: the speed − 2σ gate is the top cause (heuristic σ median 0.23 m/s), followed
  by object rebirth (too young) and the activation streak. A2 replicates the pattern.
- 5A also reports a fresh object inside the robot footprint in about 98% of frames (self/near-field
  returns).

**Tests:** 351 passed, 3 skipped in 25 s (1 hardware; 2 need the optional `av` package from the `go2` extra); 43 new viewer tests. Ruff passes on
the new files; the 33 pre-existing findings in the old hardware scripts are unchanged.

**Videos (private, local):** `/Users/yushan/Desktop/Go2_Navigation/data/outputs/phase5b/videos/`.

**Next:**
1. Codex: freshness from recent coverage, a motion gate robust to σ, history inheritance, a self-filter.
2. Record held-out H1–H3 under `configs/phase5_heldout_protocol.yaml`.
3. Calibrate the camera.
4. Build and measure a live read-only viewer.

## Phase 5A implementation — 2026-11-11 (experimental)

Branch `phase5-stable-lidar-perception`, based on integrated `9068088542067fc117535279b412fbf2f808fdf6`. Scene accumulation, expiring background voxels, observation-supported filtered boxes, uncertainty-aware motion rules and JSONL replay are implemented. Existing tracking, adapters, forecasting and operational configs remain unchanged. No main merge.

**Verified:** 310 tests passed, one hardware test skipped (39.65 s); new Phase 5A files pass Ruff 0.11.2. Full-repository Ruff finds 33 pre-existing issues in four unchanged hardware diagnostic scripts. Six synthetic scenarios × five variants and 3,926 real raw scans were processed. Python 3.12.11, NumPy 1.26.4, SciPy 1.15.2, Open3D 0.19.0, torch 2.6.0, pytest 8.3.5, jsonschema 4.26.0; report Matplotlib 3.11.2. This used existing dependency files through an isolated runtime, not a new clean install.

**Results and limitations:** some static jitter/false-motion metrics improve, but fresh box coverage falls substantially. Person moving recall is only 0.121 on A2 and 0.058 on B, versus original 0.620/0.575. Approximately 69% of scene boxes are stale. Reduced merges often become misses. This profile is not recommended as an operational replacement. Real annotations are development-only and not human-verified; no held-out or live Phase 5A validation. Midpoint pose is used on real scans; point-level deskew is synthetic-tested only. Overlap separation, odometry drift and covariance calibration remain unresolved.

**Read:** [architecture](STABLE_LIDAR_ARCHITECTURE.md), [experiments and exact commands](STABLE_LIDAR_EXPERIMENTS.md), [full tables](STABLE_LIDAR_RESULTS.md), [aggregate JSON](STABLE_LIDAR_METRICS.json), [scene interface](SCENE_REPRESENTATION_INTERFACE.md). Runtime: `src/go2_perception/scene/`; opt-in config: `configs/stable_lidar.yaml`; replay: `scripts/run_stable_scene.py`. Raw outputs and private recordings remain gitignored.

**Next:** Claude can consume the published v1 fixture immediately. Review annotations, collect held-out two-object scenes, verify point-frame timing during robot motion, and improve current-observation coverage before tuning thresholds or promoting defaults. No new motion/control code or training labels are provided.

# Project state — 2026-11-09

## Phase 3.8 visual ground truth (integrated history)
Branch `phase3-visual-groundtruth`. It contains the Phase 3.8 ground-truth work (`79bb0f9`)
plus a merge of `phase4-social-forecasting` (`5979a76`; code and documents only, no tracker
default changed). Main is unchanged. Read [VIDEO_ANNOTATION](VIDEO_ANNOTATION.md).

**Synchronized diagnostic videos.** `scripts/render_tracking_video.py`, H.264 1920×1080 at
real time, written to the ignored `data/outputs/mac/videos/`.
- **Left:** camera with YOLO boxes, labelled not calibrated to LiDAR.
- **Right:** odom BEV with raw points, clusters, robot path, tracks coloured by motion
  (observed filled, predicted hollow), trails, velocity, a 4.8 s constant-velocity future, and
  the annotations.
- **Bottom:** clocks with approximate camera−LiDAR offset (p95 34 ms), counts, robot
  velocity, warnings, and an event timeline and log. ID switches are red only when the
  annotations support them; other events are amber heuristics.

**Outputs:**
- Full videos for A1, A2, B and C;
- three-variant side-by-side comparisons;
- forecast overlays for B and A2;
- a 70 s highlight montage.

All decode on macOS (QuickLook) and were checked with ffprobe.

**Also added:**
- **Manual annotation editing:** an OpenCV editor plus CSV round trip. Edits are stored as
  review entries; `identity_uncertain` states are don't-care in scoring.
- **Forecasting readiness:** `scripts/forecast_ground_truth.py`.

**Results:**
- **Ground-truth scores unchanged.** IDF1 B 0.33/0.40/0.39, A2 0.30/0.60/0.53,
  C 0.29/0.60/0.41, A1 0.90/0.90/0.86 (original / 3.6 / 3.7).
- **Phase 4.2 eligibility.**
  - Annotated windows (8+12 at 0.4 s): B 200, A2 139, only 6 independent in total.
  - Identity-validated tracker windows: B 25/29/40, A2 12/25/12.
  - CV ADE/FDE with annotated history: B 0.55/1.28 m, A2 1.18/2.81 m.
  - No learned checkpoint is on this Mac.

Tests: **283 passed, 1 hardware skipped**, in both venvs.

**Next:**
1. Human review of the A2, B and C annotations with the editor.
2. Held-out recordings H1–H3 (with two people and taped poses).
3. Copy the Phase 4/4.1 checkpoints to this Mac to evaluate learned models on validated windows.
4. Camera calibration.

## Phase 3.8 ground-truth validation (2026-11-08)
Branch `phase3-ground-truth-validation`, from `phase3-tracking-refinement` at `c643422`. Main is
unchanged and no tracker code changed. Read [TRACKING_GROUND_TRUTH](TRACKING_GROUND_TRUTH.md).

**What was added:**
- An annotation workflow (`scripts/annotate_ground_truth.py`) and schema, matching and metrics
  (`tracking/ground_truth.py`).
- An evaluator (`scripts/evaluate_ground_truth.py`).
- Sensor-based annotations of B, A2, C and A1: 3,069 tracker frames; person point and velocity,
  box footprint region, visibility, motion and confidence.
  - The annotations came from LiDAR background subtraction and were reviewed by Claude on
    BEV-plus-camera contact sheets. No human has verified them yet.
  - Annotations, specs, sheets and caches stay local in the gitignored `data/annotations/`.

**Results, annotated IDF1 (Confirmed tracks), original → 3.6 → 3.7:**

| Recording | IDF1 |
|---|---|
| B | 0.33 → 0.40 → 0.39 |
| A2 | 0.30 → 0.60 → 0.53 |
| C | 0.29 → 0.60 → 0.41 |
| A1 | 0.90 → 0.90 → 0.86 |

- **Best current identity configuration: 3.6.** 3.7 is between the original and 3.6. Walking
  MOVING recall (B) is 0.61 → 0.43 → 0.50.
- **Not default-ready.** IDF1 on the person recordings is at most 0.60, and the static box
  changes ID 6–55 times per recording.

**Bottleneck:** the LiDAR proposals, in particular clusters merging (DBSCAN `eps` 0.3 m).
- The box loses its own proposal in 14–33% of visible frames by merging with the counter or wall,
  and in B another 12% by merging with the person.
- The person in A2 merges in 23% of frames.
- Most ID changes follow such a merge or a duplicate track born during one. Odometry is not a
  significant cause in C.
- **Oracle segmentation diagnostic.** With each annotated object's own points as one proposal,
  the unchanged original tracker reaches IDF1 B 0.67, A2 0.69, C 0.54 and A1 0.98, above every
  tuned profile in three of four recordings. 3.6 and 3.7 lose identity on clean proposals.

**Next:**
1. Proposal-level work: a height- or background-aware split of object clusters from walls, and
   person–object separation.
2. Then record the held-out protocol H1–H3 in the report, with a second person and a taped pose,
   and score it once.
3. Have a person review the annotations.

Tests: **203 passed, 1 hardware skipped**, in both venvs.

## Phase 4.1 forecasting state (merged from `phase4-social-forecasting` at 5979a76)

Branch `phase4-social-forecasting`, based on verified Phase 4 `447f164e808fee381fd3e9dbe8244583a0bed2f4`. No merge into main. Implemented observation-only multi-pedestrian extraction with partial-history masks, an 87,128-parameter shared temporal encoder with relative-position/mean neighbor pooling, dedicated checkpoint/training/evaluation interfaces, robustness studies, research plots, and a guarded offline multi-track adapter. Original forecasting, TrackOutput/ForecastOutput, tracking internals and Go2 hardware adapters are unchanged.

Completed: all five scene-identifier-held-out folds for original and social models with seeds 42/43/44 (15,061 paired windows per seed), same-weight social-disabled diagnostics, twenty validation robustness cases across clean/joint/noise-only/mask-only training, and separate CPU latency measurement. Every single-agent prediction and target/key population was checked against the independent Phase 4 evaluator. Full suite: **259 passed, one hardware skipped in 17.37 s**; Ruff passes on forecasting code/scripts/tests. No dependency changes (Python 3.12.11, torch 2.6.0, numpy 1.26.4, matplotlib 3.11.2).

**Negative result:** macro ADE/FDE, mean ± sample SD over three seeds: social 0.5812 ± 0.0162 / 1.1951 ± 0.0247 m; original 0.5565 ± 0.0118 / 1.1534 ± 0.0200 m; CV 0.5214 / 1.0933 m. Disabling social input improves macro errors in every seed. Noise-only augmentation improves one-fold noisy validation ADE to 0.3924 m, still behind noisy CV 0.3371 m. No claim of overall learned-model improvement or independent-window statistical significance.

Read [social architecture](SOCIAL_FORECASTING_ARCHITECTURE.md), [actual experiments and commands](SOCIAL_FORECASTING_EXPERIMENTS.md), [predeclared protocol and follow-up controls](SOCIAL_FORECASTING_PROTOCOL.md), [metrics](SOCIAL_FORECASTING_METRICS.json), and [Go2 contract](FORECASTING_INTEGRATION.md). Published artifacts are public-data aggregates/selected plots; raw annotations, weights, predictions and private recordings remain ignored. Known limitations: weak mean pooling, one-seed robustness controls, dependent windows, scene/domain shift, unverified dataset redistribution rights, no clean-install/GPU/live-Go2 validation. No navigation control.

Next: keep CV as the reference; Claude validates real pedestrian identity segments and stable metric frames. Then run matched held-out offline Go2 comparisons with explicit externally validated IDs. Repeat noise-only augmentation across seeds before considering more architecture. Historical phase states follow; their current/next claims apply to their original milestones.

## Phase 4 historical state (from `phase4-trajectory-forecasting`)

Branch `phase4-trajectory-forecasting`, based on Phase 3.7 `c6434229c54ee445763d8ceb1f5ef56b0f5c50da`; main is not merged. Added public ETH/UCY trajectory preparation, independent CV and Phase 3 baseline adapter, lightweight PyTorch Transformer, deterministic training/checkpoints, paired evaluation, validation ablations, plotting and guarded offline TrackOutput integration. Tracking/sensor internals remain unchanged; Claude's Phase 3.8 work is separate.

Verified: five fixed scene-identifier-held-out CPU runs (15,061 windows), validation-only history/capacity/representation/noise/missing-input studies, synthetic adapter demo and 231 tests passed/1 hardware skipped. The learned model **loses overall**: macro ADE/FDE 0.5701/1.1762 m vs CV 0.5214/1.0933 m. No Go2 forecasting accuracy or GPU validation is claimed. Data/weights remain local, public aggregate results and selected plots are committed. Python 3.12.11, torch 2.6.0, numpy 1.26.4, optional matplotlib 3.11.2; existing core pins unchanged.

Read [forecasting architecture](FORECASTING_ARCHITECTURE.md), [dataset protocol](FORECASTING_DATASETS.md), [commands and actual experiments](FORECASTING_EXPERIMENTS.md), and [integration contract](FORECASTING_INTEGRATION.md). Known limitations: one seed, correlated windows, no social context or uncertainty, unverified dataset redistribution rights, no clean-install test, and untrusted real tracker identities. Do not train from unvalidated Go2 tracks. Next: Claude validates identities/frames/visibility, then evaluate public-trained models and CV on held-out ground truth; extend forecasting research using validation-only decisions. Earlier phase sections below are historical.

## Phase 3.7 refinement (historical: `phase3-tracking-refinement`)

Branch `phase3-tracking-refinement`, based on Claude's latest integrated comparison `de043d1`.
Adds opt-in `tracking_refinement.yaml`: physical reacquisition limits, repeated-evidence duplicate
suppression and asymmetric motion dwell/release. Original and Phase 3.6 profiles are unchanged;
Go2 adapters, perception, pose gating and TrackOutput remain compatible.
Tests: **192 passed, 1 hardware skipped**. Synthetic identity counts remain stable, but Lost-status timing changes coverage scores;
moving recall improves with documented precision/coverage tradeoffs. No learned forecasting or live test.
Read [TRACKING_REFINEMENT](TRACKING_REFINEMENT.md) for exact real comparisons, limitations and
commands. Historical results below remain evidence for their own configurations.

Final offline replay: 3,924 frames; original/default outputs remain identical. Walking MOVING
proxy improves about 39%→52%, and long reactivations fall in all recordings. Much of the duplicate
count reduction is removal of coasting Confirmed states; observed-only duplicates barely change.
A2/C dominant-ID continuity regresses relative to Phase 3.6, and A1 flicker increases. This is
not a uniform fix or a default-ready profile; labels/held-out testing remain required. Full measured
results and runtime are in TRACKING_REFINEMENT. No private samples are committed.

## Phase 3.6 robustness (historical: `phase3-tracking-robustness`)
Based on latest macOS validation `ecfaae3`; main is unchanged. Opt-in profile adds time-based
confirmed-track expiry, capped covariance gating, mature-box fragment recovery and motion dwell.
Legacy defaults, raw perception, Go2 adapters, pose gating and public TrackOutput remain intact.
Verified source tests: **161 passed, 1 hardware skipped** (baseline 135/1). Forty deterministic
benchmark cases give 61→7 ID switches; four remaining switches are intentional pose resets,
three are split-recovery failures on seed 42. Nearby objects stay separate. Brief-stop recall
regresses with dwell, so this is an experimental profile, not a new default.
See [TRACKING_ROBUSTNESS](TRACKING_ROBUSTNESS.md) for the quantitative comparisons, local
real-recording results, exact commands and limitations. Dependencies are unchanged. No new live
robot validation or calibrated fusion. Next: label A2/B/C identity/visibility, validate two-person
separation and slow motion, then create quality-filtered forecasting targets. Historical sections
below describe their original branches and validation dates, not the current implementation.

**Real-data comparison (Claude, Mac, offline replay).** The improved profile was compared with
the original on identical inputs from A1/A2/B/C (3,924 frames); see
[TRACKING_REAL_COMPARISON](TRACKING_REAL_COMPARISON.md).
- **Checks:** the baseline replay matched the original Phase 3 outputs in every frame, and the
  working code with default `tracking.yaml` is bit-identical to `ecfaae3`.
- **Better:**
  - Box main-ID share after occlusion and turns: A2 0.50 → 0.96, B 0.31 → 0.49, C 0.33 → 0.59.
  - False MOVING on static structure nearly gone (A1 1,550 → 127 frames).
- **Worse:**
  - Duplicate-pair frames +22–38%, and more ID flicker (C box returns 9 → 22).
  - More long-range reactivations.
  - Person IDs unchanged (B 19 → 20).
  - Walking MOVING recall 62% → 39%.
  - Tracker time +40%.
- **Verdict:** ready to merge as an opt-in profile, not ready to become the default.
  No annotations exist; all values are proxies.

## macOS hardware integration (branch `feature/macos-go2-phase3-validation`)
Based on `phase3-tracking` (8c35a39); not merged into main. **Verified on a real Go2 from an Apple
Silicon Mac:** native read-only access with the unchanged Phase 2 adapter over `en9`
(Python 3.10 `.venv-go2`, Cyclone DDS wheel, SDK 814556d): RGB 14.3 Hz 1280x720, LiDAR about
15.4 Hz, IMU 259 Hz, odometry 153 Hz, robot state; recording, YOLO (MPS) and LiDAR replay on real
frames. Phase 3 replay now gates frames on odometry (`tracking/pose_gate.py`): frames before the
first odometry sample or inside gaps are skipped with reasons, never extrapolated or filled;
long gaps end all tracks. Tests: **135 passed, 1 hardware skipped** (both venvs).

Real tracking, robot static (A1/A2/B, 50-65 s each): an unoccluded box keeps one ID for 60 s
(4-5 cm jitter); a walking person is covered 96% with roughly correct speed (0.77-0.89 vs about
0.65-0.80 m/s reference), but people fragment into many IDs (19 for one person in B), nearby
static objects are re-identified when occluded, and STATIONARY is rarely assigned because of
classifier thresholds. Clutter in a 10 s static run gave false speeds up to 7.8 m/s. **Test C (human-driven
turns ±23 deg, 0.5 m walk):** the static box stays within 6-9 cm in odom with no MOVING labels, so
ego-motion compensation works on real data; its ID still changes every 5-15 s. Test D not recorded. No
calibration exists, so RGB and LiDAR remain separate. macOS lacks the ICMP clock offset.
Evidence: [MAC_CONNECTION](MAC_CONNECTION.md), [PHASE3_REAL_VALIDATION](PHASE3_REAL_VALIDATION.md).

## Phase 3 — temporal tracking (current branch: `phase3-tracking`)
Implemented on integrated baseline `dc0f85b`; not merged into main. Adds validated timestamped
3D detections, odometry/fixed-sensor normalization, planar CV Kalman tracking, Hungarian gating,
persistent IDs/lifecycle, velocity uncertainty, motion classification, 3-second trajectories,
raw Go2/detection replay, headless BEV/video and synthetic evaluation. No SDK communication or
existing perception code changed; no new runtime dependencies. Phase 2 uses DDS/RTP directly,
not a mandatory ROS 2 adapter.

Verified: baseline 70 passed/1 skipped; expanded suite **127 passed/1 hardware skipped**. All
8 reference scenarios run with zero ID switches, XY MAE .0181–.0598 m and velocity MAE
.0496–.1164 m/s. Additional 40 noisy-seed sequences have four ID switches; this remains a known
baseline limitation. A single real cloud produced four tentative uncertain tracks in explicitly
fixed-sensor mode; real temporal accuracy and Phase 3 live hardware behavior remain unverified.
Real fusion still requires calibration and verified time alignment. Covariance assumes exact
poses; sparse/merged/partially observed clusters can fragment IDs. No locomotion or planning.

Read [TRACKING_ARCHITECTURE](TRACKING_ARCHITECTURE.md), [TRACKING_EVALUATION](TRACKING_EVALUATION.md),
and the Phase 3 handoff below/at AGENT_HANDOFF.md. Next: labelled temporal Go2 replay, motion/pose
validation, calibration and synchronization, then replaceable learned prediction experiments.
All Phase 1/2 sections below are historical records and retain their original validation scope.

## Main integration
Local merge verification: **70 passed, 1 hardware test skipped** (macOS CPU).
`main` combines Phase 1 (`359fea8`) and Phase 2 (`72585e0`). Phase 2 descends directly
from Phase 1, so both histories and all source changes are preserved without conflict.
The merge adds offline recording-to-CLI regression tests and reconciles status/setup docs;
Claude-owned adapter source, hardware scripts and Go2 configurations are unchanged.
See [MERGE_VALIDATION](MERGE_VALIDATION.md) for local verification, separate from the
hardware-workstation results below. Real calibrated fusion remains blocked.

## Latest verified state: Phase 2 hardware integration
Branch `feature/go2-sensor-integration`, based on `codex/phase1-perception`. **Read-only** access to a
real Go2 Pro over Ethernet works; no movement or control functionality exists.

**Verified on the real robot (stationary):** DDS and multicast streams; RGB 1280x720 at 14.3 Hz;
LiDAR single frames at 15.4 Hz; IMU 251 Hz; odometry 150 Hz; recording and offline replay; unmodified
YOLO and LiDAR processors on real data; live inference at about 13 Hz RGB and 14 Hz LiDAR; correct
person detection in one scene. Linux/CUDA is verified. Tests: **68 passed, 1 hardware test skipped
offline, and that test passes live.**

**Blocked:** RGB-LiDAR fusion, for lack of camera calibration and verified synchronization.
**Unverified:** link-drop recovery, all behaviour under robot motion, LiDAR model (probably L2;
inferred only), detection accuracy against labels, runs longer than 106 s.

**Facts a new agent must not miss:**
- `rt/utlidar/cloud_deskewed` is a rolling 10,000-point buffer plus the newest frame, not a scan.
- One `cloud_base` frame covers only about 190 deg of azimuth; use the 0.3 s compensated window.
- `cloud_base` contains nothing within 0.3 m XY and is heavily thinned at close range.
- The robot clock is about 273 days behind the host; RTP timestamps are not capture times.

Evidence and commands: [GO2_CONNECTION](GO2_CONNECTION.md), [SENSOR_VALIDATION](SENSOR_VALIDATION.md),
[INTEGRATION_REPORT](INTEGRATION_REPORT.md). Next steps: [AGENT_HANDOFF](AGENT_HANDOFF.md).

The Phase 1 record below is historical. Phase 2 supersedes its live-access, Linux/CUDA,
and temporal-window limitations; its old next-task list is not the current backlog.

# Phase 1 record
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
- *(Superseded by Phase 2: Linux/CUDA, live streams and topic names are now verified.)* Linux/CUDA and live Go2 streams not tested; no actual topic names or sensor parameters known.
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
