# Agent handoff — 2026-11-08 (macOS update)

## Phase 3.8 ground-truth handoff (current)
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

# Agent handoff — 2026-11-07 (macOS update)

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
