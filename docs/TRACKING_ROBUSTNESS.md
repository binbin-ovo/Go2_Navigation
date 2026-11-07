# Phase 3.6 tracking robustness — 2026-11-07

Branch: `phase3-tracking-robustness`, based on macOS validation commit `ecfaae3`, not main.
The original [real validation report](PHASE3_REAL_VALIDATION.md) is unchanged. This is an
opt-in tracker experiment, not a replacement calibration or a claim of reliable person identity.
No new dependencies, hardware commands, forecasting models, or sensor-adapter changes.

## Evidence and root causes

- **Fragmentation (A2/B):** Claude observed multiple clusters/IDs for one person (19 IDs in B).
  The tracker associates one proposal to one track, so split shapes can create births and
  alternating matches. Without labels we cannot attribute every real ID to DBSCAN splitting
  rather than occlusion or clutter. We leave DBSCAN and raw clusters unchanged. A mature full
  box can now recover compatible contained fragments; legs split since first observation
  cannot safely be inferred to be one person. RGB semantics remain unavailable without calibration.
- **Occlusion:** baseline `max_misses=10` expires at the 11th missed frame, about .71 s at
  15.4 Hz, before `max_unobserved_s=1.5`. Confirmed tracks can now use time-only expiry.
  Lost tracks re-enter ordinary gated Hungarian matching and retain their ID. This is not
  long-term re-identification. Tentatives keep the existing frame-count limit.
- **Turning (C):** recorded box positions held within 6–9 cm in odom despite ID changes;
  the evidence does not justify changing transform direction. Covariance grows during gaps,
  but the old fixed 1.5 m gate can reject otherwise plausible innovations. A capped adaptive
  radius now supplements the unchanged Mahalanobis, class, and vertical gates. Detection
  latency and nonuniform partial views remain possible real contributors, not proven causes.
  Odometry interpolation, clock assumptions, pose skips and >.5 s gap reset are unchanged.
  A synthetic yaw crossing pi preserves the stationary ID and does not create world motion.
- **Motion (A1):** recorded sigma_v around .14 m/s combined with stationary threshold .15
  requires estimated speed around .01 m/s or below. The optional threshold .25 permits about
  .11 m/s at that uncertainty. A .25 s candidate dwell reduces immediate label changes;
  insufficient evidence and missed observations immediately produce UNCERTAIN. This threshold
  is informed by measured noise but is not calibrated on labelled Go2 motion.

## Algorithm and configuration

Use `configs/tracking_robustness.yaml`; the original `tracking.yaml` and constructor defaults
retain baseline behavior. New options in `TrackerConfig`:

| Field | Default | Experimental profile | Meaning |
|---|---:|---:|---|
| confirmed_time_expiry | false | true | Confirmed tracks expire by elapsed unobserved seconds |
| adaptive_gate_max_distance | 0 | 2.5 m | Zero disables; bounded covariance-based Euclidean radius |
| fragment_recovery | false | true | Recover fragments using an existing full box |
| fragment_min_hits | 5 | 5 | Minimum prior observations and confirmed status |
| fragment_margin | .05 m | .10 m | Containment tolerance (recorded jitter is several cm) |
| fragment_min_iou | .6 | .6 | Minimum union/predicted box 3D IoU |
| fragment_max_extent | 1.5 m | 1.5 m | Largest allowed prior box dimension |
| motion_dwell_s | 0 | .25 s | Continuous candidate evidence before emitting label |
| stationary_speed (existing) | .15 m/s | .25 m/s | Upper uncertainty-aware stationary threshold |

Radius = max(base_distance, min(cap, sqrt(mahalanobis_gate * lambda_max(Pxy + Rxy)))).
Mahalanobis gating still uses the full innovation covariance; uncertainty never bypasses the
2.5 m cap. Normalized distance cost uses the same radius. Longer coasting can increase clutter
matches; it is not evidence of identity by itself.

Fragment recovery runs after pose normalization and Kalman prediction, before association.
It requires at least two boxed proposals inside a mature compact track's predicted box,
exclusive ownership, compatible known classes and sufficient union IoU. A competing confirmed
track inside the union vetoes recovery. The union midpoint is the measurement; isotropic
covariance uses the largest eigenvalue among fragments (correlated fragments do not gain
false confidence). It preserves source provenance in `fragment_groups`. Two nearby objects
and fragments without a previous full box remain separate in the regression tests. Ambiguous
partial visibility, a wrong mature box, and noisy containment boundaries can still fail.

The Python update signature and `TrackOutput` fields are unchanged. CLI `tracks.jsonl` rows
add `raw_detection_count`, `created_ids`, `reactivated_ids`, and `fragment_groups`; raw LiDAR
proposals remain separately countable. Tracks with missing observations have no confident
trajectory. Pose-gate resets keep IDs increasing; a reset is not bridged by longer coasting.

## Validation and protocol

Baseline: **135 passed, 1 hardware skipped in 11.05 s**. Final source suite:
**161 passed, 1 hardware skipped in 12.04 s** on macOS ARM64, Python 3.12.11.
Pinned dependencies unchanged; see SETUP and earlier TRACKING_EVALUATION. Ruff passed for
changed Python files. The hardware-marked test requires live Go2 access; no live robot test
was attempted in this phase. Existing adapter, replay, geometry, RGB and LiDAR tests pass.

The benchmark archives baseline source/configs from Git into a temporary directory, then
runs baseline and current tracker in separate Python processes on identical cached detections.
Explicit PYTHONPATH prevents a stale installed package from silently changing the comparison.
Each input JSONL has a SHA256 fingerprint. Raw extraction uses unchanged Go2 LiDAR settings,
.3 s accumulation and existing pose gating; cache files remain ignored and local. Benchmark
runtime measures tracker.update only, including copies/history output, excluding extraction,
serialization, visualization and evaluation. Runs shared the machine with tests/extraction,
so timing is descriptive, not a controlled throughput claim.

Synthetic matching uses the existing evaluator with a **.5 m XY** gate (earlier Phase 3 used
1 m). IDF1 is planar distance-based identity assignment, not MOTChallenge box-IoU scoring.
Position/velocity MAE are matched coordinate errors. Identity fragmentation counts extra
unique matched IDs per GT. Duplicate-track frames count extra confirmed tracks within .3 m
of their unique nearest GT. Motion precision/recall is **conditional on spatially matched
confirmed outputs**; UNCERTAIN counts as a recall miss, missing/unmatched tracks do not enter
that conditional metric. GT speed >= .3 m/s is MOVING; fixtures have no ambiguous slow motion.
Created, ever-tentative and ever-confirmed counts are unique IDs, not cumulative frame counts.
Reactivation counts Lost→Confirmed events; repeated events can exceed unique track count.

### Synthetic comparison (seed 36)

Values are baseline → experimental profile. All ten fixtures have deterministic GT IDs.

| Case | ID switches | IDF1 | Created IDs | Position MAE m | Velocity MAE m/s | Duplicate frames |
|---|---:|---:|---:|---:|---:|---:|
| brief_stop | 0 → 0 | 0.990 → 0.990 | 1 → 1 | 0.017 → 0.017 | 0.067 → 0.067 | 0 → 0 |
| crossing | 0 → 0 | 0.987 → 0.987 | 2 → 2 | 0.016 → 0.016 | 0.047 → 0.047 | 0 → 0 |
| fragmentation | 11 → 0 | 0.698 → 0.990 | 2 → 1 | 0.073 → 0.009 | 0.024 → 0.019 | 83 → 0 |
| irregular | 0 → 0 | 0.987 → 0.987 | 1 → 1 | 0.018 → 0.018 | 0.051 → 0.051 | 0 → 0 |
| nearby_objects | 0 → 0 | 0.990 → 0.990 | 2 → 2 | 0.009 → 0.009 | 0.019 → 0.019 | 0 → 0 |
| occlusion | 1 → 0 | 0.575 → 0.907 | 2 → 1 | 0.011 → 0.010 | 0.032 → 0.023 | 0 → 0 |
| odometry_gap | 1 → 1 | 0.651 → 0.651 | 2 → 2 | 0.009 → 0.009 | 0.030 → 0.030 | 0 → 0 |
| outliers | 0 → 0 | 0.990 → 0.990 | 9 → 9 | 0.008 → 0.008 | 0.018 → 0.018 | 0 → 0 |
| rotation_wrap | 0 → 0 | 0.990 → 0.990 | 1 → 1 | 0.010 → 0.010 | 0.023 → 0.023 | 0 → 0 |
| stationary_noise | 0 → 0 | 0.990 → 0.990 | 1 → 1 | 0.017 → 0.017 | 0.037 → 0.037 | 0 → 0 |

The occlusion spans 16 missed scans (1.04 s); reactivation events rise 0→1.
The odometry-gap identity change is intentional: neither variant bridges an unobservable pose gap.
Noisy stationary recall rises .592→.837 with precision 1.0 in this fixture. Brief-stop stationary
recall falls .516→.484 and moving recall .761→.642; dwell delays both transitions. Brief-stop
stationary precision falls .762→.750, while moving precision rises .864→.878. Do not interpret
more stationary labels as a universal classification improvement.

Additional fixed seeds 7, 13, 42 yield split-object switches 7→0, 16→0, **19→3**. Seed 42 still
creates two IDs and retains duplicate observations; no settings were changed to eliminate this
failure. Across all four seeds/ten cases: switches 61→7 (four are intentional pose resets).
All nearby-object cases retain two distinct IDs. These are controlled mechanism tests, not a
simulation of Go2 accuracy. Full per-case motion, latency/throughput, lifecycle and error metrics:
[TRACKING_ROBUSTNESS_METRICS.json](TRACKING_ROBUSTNESS_METRICS.json).

## Local real-recording comparison (completed offline)

A1/A2/B/C were processed **locally**, without uploading recordings or connecting to the robot.
The exact retained total is **3,924 frames** (924 + 769 + 1001 + 1230).
Existing pose gating excludes initial/unobservable frames. Both workers used the same extracted
proposals and reset flags; their imported paths were checked as archived baseline versus working
source. This establishes offline execution on Go2 recordings, not new live hardware validation.

The recordings have no complete identity/visibility ground truth. We therefore report **no real
IDF1, ID-switch accuracy, position/velocity error or classification precision/recall**. Fewer IDs
or MOVING labels may also result from incorrect associations, merged objects or suppressed motion.
All-scene counts include clutter, walls and partial objects, not just the person/box in Claude's
report. The baseline is a fresh replay of ecfaae3 with shared extraction, not the historical outputs.

| Recording | Frames | Created / ever tentative IDs | Ever confirmed IDs | Lost→confirmed events | Fragment groups | Tracker ms/frame | Tracker Hz |
|---|---:|---:|---:|---:|---:|---:|---:|
| test_a1 | 924 | 514 → 261 | 471 → 240 | 1991 → 2559 | 0 → 48 | 21.34 → 33.00 | 46.87 → 30.31 |
| test_a2 | 769 | 464 → 243 | 421 → 222 | 1548 → 1981 | 0 → 44 | 21.06 → 30.26 | 47.49 → 33.04 |
| test_b | 1001 | 624 → 333 | 576 → 301 | 1902 → 2502 | 0 → 83 | 21.10 → 30.24 | 47.40 → 33.07 |
| test_c | 1230 | 699 → 334 | 652 → 315 | 2926 → 3847 | 0 → 27 | 25.72 → 37.39 | 38.88 → 26.75 |

The fixed ROI is a deliberately simple **unannotated proxy**: .5 m XY radius around a point
2.5 m ahead of the first valid robot pose (from the approximate box location in the report).
It can include a passing person or clutter and need not equal the true box center. Each frame
selects the nearest confirmed track. The ROI does not follow a labelled object and its IDs
must not be compared directly to Claude's manually selected box IDs. Competing-track frames
are not necessarily duplicates; they can represent distinct objects. Coverage includes up to
one missed frame before the existing Lost transition. These caveats apply to both variants.

| Recording | Unique nearest ROI IDs | ROI coverage | Competing-track frames | ROI STATIONARY frames | ROI MOVING frames |
|---|---:|---:|---:|---:|---:|
| test_a1 | 10 → 7 | 0.930 → 0.926 | 37 → 40 | 52 → 446 | 7 → 0 |
| test_a2 | 19 → 19 | 0.919 → 0.923 | 44 → 72 | 36 → 248 | 38 → 13 |
| test_b | 24 → 18 | 0.896 → 0.895 | 89 → 96 | 89 → 366 | 73 → 2 |
| test_c | 34 → 13 | 0.859 → 0.870 | 114 → 137 | 88 → 345 | 63 → 3 |

The real comparison is mixed: A2's ROI unique-ID count is unchanged; competing-track frames
increase in every recording, A1/B coverage slightly decreases, and mean tracker latency rises
by about 9–12 ms. The profile remains experimental. Aggregate reports, including all-scene
motion counts and ROI speed percentiles: [TRACKING_ROBUSTNESS_REAL_PROXIES.json](TRACKING_ROBUSTNESS_REAL_PROXIES.json).
No camera image, raw point, pose, timestamp sequence or per-object trajectory is in this report.

## Reproduction and Claude handoff

From the repository root (explicit PYTHONPATH also works around the documented macOS stale
editable-package issue):

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
PYTHONPATH=src .venv/bin/python scripts/benchmark_tracking_robustness.py --baseline ecfaae3 --output data/outputs/robustness/synthetic --seed 36
# Repeat with --seed 7, 13, 42 and separate output directories.
PYTHONPATH=src .venv/bin/python scripts/benchmark_tracking_robustness.py --baseline ecfaae3 --output data/outputs/robustness/real --recordings data/recordings/test_a1 data/recordings/test_a2 data/recordings/test_b data/recordings/test_c
# Add --reuse-cache to compare exactly the existing extracted detections without re-extraction.
PYTHONPATH=src .venv-go2/bin/python scripts/run_tracking.py --input data/recordings/test_c --config configs/tracking_robustness.yaml --window-s 0.3 --output data/outputs/mac/test_c_robustness --no-render
```

A 100-frame fragmentation CLI smoke test also passed, with structured lifecycle diagnostics:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_tracking.py --input data/outputs/robustness/synthetic/inputs/fragmentation --config configs/tracking_robustness.yaml --output data/outputs/robustness/cli_smoke --no-render
```

The hardened benchmark was rerun at seed 36 with output `data/outputs/robustness/harness_check`;
source-path verification passed. Use a fresh CLI output directory. `.venv-go2` can replace `.venv` for the benchmark/tests on
Claude's Mac. No robot connection is needed for replay. Do not stage data/recordings or
per-frame outputs, cache files, images, point clouds, poses or trajectories. Published JSON
reports contain only aggregate metrics/configuration and input digests, no sensor samples.

Claude should annotate box/person identities and visibility on A2/B/C locally, review split
and merge events against raw clusters, then rerun with the same cached detections. Test two
close people (D), long/ambiguous occlusion and a taped end pose for drift. Tune dwell and slow
motion thresholds on a separate labelled set. Verify physical timestamp/clock alignment and
calibration before enabling RGB semantic association. Preserve the hardware adapter ownership.

## Remaining limitations and Phase 4

No appearance model, semantic human detector in LiDAR, long-term re-identification, or
identity guarantee during merged/ambiguous observations. Recovery needs a previously complete
compact box; conservative vetoes leave split-from-birth objects unresolved. Longer track life
can retain clutter. Seed 42 demonstrates containment sensitivity. Raised stationary threshold
can mislabel slow motion, and dwell delays genuine stop/start transitions. Pose covariance
is still treated as exact. No new live validation, calibrated RGB–LiDAR fusion or end-to-end
real-time guarantee. Synthetic tests and unlabelled replay cannot establish real ID accuracy.

The stable output interface is ready for offline forecasting dataset tooling, but reliable
learned forecasting targets require labelled identity/visibility and motion-quality filtering
first. Do not begin large-scale learned forecasting or navigation on these noisy IDs.
