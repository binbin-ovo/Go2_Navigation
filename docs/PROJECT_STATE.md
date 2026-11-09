# Phase 4.1 current state — 2026-11-09

Branch `phase4-social-forecasting`, based on verified Phase 4 `447f164e808fee381fd3e9dbe8244583a0bed2f4`. No merge into main. Implemented observation-only multi-pedestrian extraction with partial-history masks, an 87,128-parameter shared temporal encoder with relative-position/mean neighbor pooling, dedicated checkpoint/training/evaluation interfaces, robustness studies, research plots, and a guarded offline multi-track adapter. Original forecasting, TrackOutput/ForecastOutput, tracking internals and Go2 hardware adapters are unchanged.

Completed: all five scene-identifier-held-out folds for original and social models with seeds 42/43/44 (15,061 paired windows per seed), same-weight social-disabled diagnostics, twenty validation robustness cases across clean/joint/noise-only/mask-only training, and separate CPU latency measurement. Every single-agent prediction and target/key population was checked against the independent Phase 4 evaluator. Full suite: **259 passed, one hardware skipped in 17.37 s**; Ruff passes on forecasting code/scripts/tests. No dependency changes (Python 3.12.11, torch 2.6.0, numpy 1.26.4, matplotlib 3.11.2).

**Negative result:** macro ADE/FDE, mean ± sample SD over three seeds: social 0.5812 ± 0.0162 / 1.1951 ± 0.0247 m; original 0.5565 ± 0.0118 / 1.1534 ± 0.0200 m; CV 0.5214 / 1.0933 m. Disabling social input improves macro errors in every seed. Noise-only augmentation improves one-fold noisy validation ADE to 0.3924 m, still behind noisy CV 0.3371 m. No claim of overall learned-model improvement or independent-window statistical significance.

Read [social architecture](SOCIAL_FORECASTING_ARCHITECTURE.md), [actual experiments and commands](SOCIAL_FORECASTING_EXPERIMENTS.md), [predeclared protocol and follow-up controls](SOCIAL_FORECASTING_PROTOCOL.md), [metrics](SOCIAL_FORECASTING_METRICS.json), and [Go2 contract](FORECASTING_INTEGRATION.md). Published artifacts are public-data aggregates/selected plots; raw annotations, weights, predictions and private recordings remain ignored. Known limitations: weak mean pooling, one-seed robustness controls, dependent windows, scene/domain shift, unverified dataset redistribution rights, no clean-install/GPU/live-Go2 validation. No navigation control.

Next: keep CV as the reference; Claude validates real pedestrian identity segments and stable metric frames. Then run matched held-out offline Go2 comparisons with explicit externally validated IDs. Repeat noise-only augmentation across seeds before considering more architecture. Historical phase states follow; their current/next claims apply to their original milestones.

# Phase 4 historical state — 2026-11-09

Branch `phase4-trajectory-forecasting`, based on Phase 3.7 `c6434229c54ee445763d8ceb1f5ef56b0f5c50da`; main is not merged. Added public ETH/UCY trajectory preparation, independent CV and Phase 3 baseline adapter, lightweight PyTorch Transformer, deterministic training/checkpoints, paired evaluation, validation ablations, plotting and guarded offline TrackOutput integration. Tracking/sensor internals remain unchanged; Claude's Phase 3.8 work is separate.

Verified: five fixed scene-identifier-held-out CPU runs (15,061 windows), validation-only history/capacity/representation/noise/missing-input studies, synthetic adapter demo and 231 tests passed/1 hardware skipped. The learned model **loses overall**: macro ADE/FDE 0.5701/1.1762 m vs CV 0.5214/1.0933 m. No Go2 forecasting accuracy or GPU validation is claimed. Data/weights remain local, public aggregate results and selected plots are committed. Python 3.12.11, torch 2.6.0, numpy 1.26.4, optional matplotlib 3.11.2; existing core pins unchanged.

Read [forecasting architecture](FORECASTING_ARCHITECTURE.md), [dataset protocol](FORECASTING_DATASETS.md), [commands and actual experiments](FORECASTING_EXPERIMENTS.md), and [integration contract](FORECASTING_INTEGRATION.md). Known limitations: one seed, correlated windows, no social context or uncertainty, unverified dataset redistribution rights, no clean-install test, and untrusted real tracker identities. Do not train from unvalidated Go2 tracks. Next: Claude validates identities/frames/visibility, then evaluate public-trained models and CV on held-out ground truth; extend forecasting research using validation-only decisions. Earlier phase sections below are historical.

# Project state — 2026-11-07

## Phase 3.7 refinement (historical)
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
