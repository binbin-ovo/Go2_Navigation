# Phase 3.7 tracking refinement — 2026-11-07

Branch `phase3-tracking-refinement` starts at Claude's integrated comparison commit `de043d1`.
Main is unchanged. This is an opt-in experiment: original `tracking.yaml` and Phase 3.6
`tracking_robustness.yaml` remain unchanged. Historical real reports and outputs are preserved.

## Diagnosis and changes

Claude's [real comparison](TRACKING_REAL_COMPARISON.md) attributed better static continuity and
more duplicates to longer confirmed-track lifetime, long reactivation jumps to adaptive gates,
and reduced walking MOVING recall to symmetric dwell and the raised stationary threshold.
The refinement retains time expiry, fragment recovery and the covariance gate, with three
additional constraints. No RGB, LiDAR, hardware, calibration or pose-gate code was modified.

1. **Physical displacement bound.** On reacquisition after missed observations, compare the new
   measurement to the last measured XY position, not only the extrapolated state. Require
   displacement <= `slack + min(max_speed, last_estimated_speed + velocity_margin) * elapsed`.
   The profile uses .25 m slack, 2 m/s maximum speed and 1.5 m/s velocity margin. Mahalanobis,
   adaptive-distance, class and vertical gates still apply. Covariance alone cannot authorize
   a large relocation. This is an indoor experimental limit, not a semantic speed estimate;
   faster objects/abrupt acceleration can lose their IDs. The slack admits measurement jitter,
   so the speed limit is a travel bound with tolerance, not an instantaneous speed assertion.
2. **Repeated duplicate evidence.** After association, require three consecutive overlapping
   updates, center distance <= .25 m, 3D box IoU >= .6, velocity difference <= .3 m/s and
   compatible known classes. Prefer the track with more observations (age/ID break ties).
   It must have a current observation before suppressing a weaker confirmed or Lost track.
   A coasting overlap is marked Lost without deleting its ID; two missing tracks are not merged.
   Unobserved strong tracks cannot suppress observed objects. No covariance averaging or
   invented observation occurs; removed IDs appear in existing `deleted_ids`. Evidence resets
   on separation, incompatibility or pose reset. Close centers alone are insufficient.
3. **Asymmetric motion response.** MOVING requires .065 s candidate evidence instead of .25 s;
   STATIONARY retains .25 s dwell. The moving uncertainty lower bound must exceed .3 m/s for
   activation; a previously MOVING track can persist above a separate .15 m/s release bound.
   Stationary upper bound remains .25 m/s. Misses and unreliable covariance immediately emit
   UNCERTAIN and no trajectory. MOVING additionally needs at least two observed history entries
   within .5 s and >= .15 m filtered XY displacement from the earliest of those entries. This
   prevents a velocity-only decision without recent displacement support; it does not prove
   that a noisy LiDAR cluster is a physical moving object.
4. **Observation status.** The profile sets existing `lost_after_misses=1`: a missed confirmed
   track immediately becomes Lost, but remains eligible for recovery for 1.5 s. Global Hungarian
   assignment is retained. This removes coasting predictions from confirmed-object counts;
   it must not be confused with resolving duplicate identities.

New settings live in `configs/tracking_refinement.yaml`. Constructor defaults disable these
features: `reactivation_max_speed=0`, `duplicate_track_hits=0`, `moving_dwell_s=None`,
`moving_release_speed=None`. Other new defaults: reactivation_slack=.25 m,
reactivation_velocity_margin=.5 m/s, duplicate_track_iou=.6, duplicate_track_distance=.25 m,
duplicate_track_velocity=.3 m/s, moving_min_displacement=0 (disabled),
motion_evidence_window_s=.5 s. Interfaces and TrackOutput schema are unchanged.

## Evaluation protocol

Use the same source-isolated comparison workers and identical cached detections as Claude.
The real harness now accepts a read-only `--cache-root`, includes a `refined` variant and refuses
to overwrite existing run/variant outputs. `--analyze` remains the explicit read-existing mode.
Original baseline and Phase 3.6 profiles are evaluated alongside the refinement. The synthetic
harness also accepts `--baseline-config` so Phase 3.6 can be the direct reference.

Synthetic metrics use the earlier .5 m planar matching protocol and GT identities. Motion recall
is conditional on matched confirmed outputs; UNCERTAIN is a recall miss. Real metrics use
Claude's fixed box anchors and noisy foreground person reference, without changing definitions.
Duplicate-pair frames mean two confirmed centers within .3 m, which need not be physical
duplicates. Reactivations over 1 m need not be physically impossible during long gaps. Person
MOVING fraction is a proxy, not annotated recall. No real IDF1 or accuracy is claimed.

## Automated and synthetic results

Baseline: **161 passed, 1 hardware skipped in 10.89 s**. Final source suite:
**192 passed, 1 hardware skipped in 11.38 s**, macOS ARM64 / Python 3.12.11. All 31 added
regressions pass, including duplicate evidence and vetoes, physically impossible reacquisition,
short/long occlusion, first-miss status, static motion support, all ten scenarios, config validation,
output protection and observed/coasting metric separation. Ruff passes for changed Python files.
No live Go2 test was attempted. Dependencies remain unchanged.

Seed 36, Phase 3.6 → refinement:

| Scenario | ID switches | IDF1 | Position MAE m | Velocity MAE m/s | MOVING recall |
|---|---:|---:|---:|---:|---:|
| brief_stop | 0 → 0 | 0.990 → 0.990 | 0.017 → 0.017 | 0.067 → 0.067 | 0.642 → 0.716 |
| crossing | 0 → 0 | 0.987 → 0.987 | 0.016 → 0.016 | 0.047 → 0.047 | 0.923 → 0.949 |
| fragmentation | 0 → 0 | 0.990 → 0.990 | 0.009 → 0.009 | 0.019 → 0.019 | — → — |
| irregular | 0 → 0 | 0.987 → 0.987 | 0.018 → 0.018 | 0.051 → 0.051 | 0.949 → 0.936 |
| nearby_objects | 0 → 0 | 0.990 → 0.990 | 0.009 → 0.009 | 0.019 → 0.019 | — → — |
| occlusion | 0 → 0 | 0.907 → 0.901 | 0.010 → 0.010 | 0.023 → 0.023 | 0.819 → 0.817 |
| odometry_gap | 1 → 1 | 0.651 → 0.651 | 0.009 → 0.009 | 0.030 → 0.030 | — → — |
| outliers | 0 → 0 | 0.990 → 0.990 | 0.008 → 0.008 | 0.018 → 0.018 | — → — |
| rotation_wrap | 0 → 0 | 0.990 → 0.990 | 0.010 → 0.010 | 0.023 → 0.023 | — → — |
| stationary_noise | 0 → 0 | 0.990 → 0.990 | 0.017 → 0.017 | 0.037 → 0.037 | — → — |

Forty scenarios (seeds 36, 7, 13, 42) retain **7→7 switches**: four intentional pose resets and
three unresolved seed-42 split-object switches. Nearby objects stay separate; short occlusions
retain the original ID, long occlusions expire. Brief-stop MOVING recall improves .642→.716
and precision .878→.889 at seed 36; all four seeds improve these brief-stop metrics. This still
does not restore the original pre-3.6 brief-stop recall (.761).

**Synthetic regressions:** occlusion IDF1 falls .9071→.9011 because the first missed observation
is now Lost rather than Confirmed. Matched coordinate-error averages shift slightly because that
coasting frame is excluded. Irregular MOVING recall falls .949→.936 at seeds 36 and 13; it rises
to .962 at seeds 7 and 42. Occlusion MOVING recall slightly decreases at seeds 36/7/13. These
tradeoffs are preserved in [TRACKING_REFINEMENT_SYNTHETIC.json](TRACKING_REFINEMENT_SYNTHETIC.json),
which includes all motion metrics, lifecycle counts, timing, profiles and input fingerprints.

## Final local real-data comparison

Completed on all **3,924 pose-valid frames** of A1/A2/B/C, with original baseline, current-default,
Phase 3.6 and refinement workers executed serially after tests and synthetic runs finished.
Inputs match Claude's cache by SHA256. The default output remains identical to the archived
baseline on every frame; baseline IDs/status/motion match the original historical replay on
every frame too. No Go2 connection was attempted. All outputs are local under
`data/outputs/mac/tracker_refinement_final`. Source fingerprints and full aggregate results are
in [TRACKING_REFINEMENT_REAL.json](TRACKING_REFINEMENT_REAL.json). The working tree was dirty
because this evaluation preceded the implementation commit; fingerprints identify the evaluated code.

Values below are original → Phase 3.6 → refinement unless specified. These are unannotated proxies.

| Test | Duplicate-pair frames | Reactivations >1 m | Box dominant-ID share | Box ID changes | Box coverage |
|---|---:|---:|---:|---:|---:|
| test_a1 | 365 → 483 → 139 | 77 → 99 → 42 | 0.993 → 0.995 → 0.979 | 4 → 6 → 19 | 0.904 → 0.903 → 0.818 |
| test_a2 | 277 → 383 → 115 | 62 → 114 → 58 | 0.498 → 0.96 → 0.843 | 15 → 28 → 21 | 0.853 → 0.856 → 0.779 |
| test_b | 380 → 504 → 137 | 96 → 163 → 66 | 0.311 → 0.489 → 0.493 | 13 → 14 → 8 | 0.808 → 0.806 → 0.727 |
| test_c | 682 → 832 → 275 | 88 → 93 → 45 | 0.327 → 0.585 → 0.484 | 30 → 30 → 31 | 0.776 → 0.783 → 0.676 |

**Status-policy confound:** Phase 3.6 allowed a first missed observation to remain Confirmed.
The refinement reports it as Lost but retains its ID. Much of the large reduction in duplicate-pair
frames is removal of coasting predictions, not proof of merged duplicate identities. The following
controls count observed tracks only, keeping the comparison meaningful across status policies:

| Test | Observed duplicate frames (3.6 → refined) | Observed box dominant share | Observed box coverage | Box-ROI MOVING frames |
|---|---:|---:|---:|---:|
| test_a1 | 141 → 139 | 0.993 → 0.979 | 0.818 → 0.818 | 0 → 1 |
| test_a2 | 112 → 115 | 0.952 → 0.843 | 0.78 → 0.779 | 0 → 4 |
| test_b | 134 → 137 | 0.491 → 0.493 | 0.732 → 0.727 | 2 → 1 |
| test_c | 278 → 275 | 0.568 → 0.484 | 0.689 → 0.676 | 2 → 2 |

Observed-duplicate counts change only slightly, including small increases in A2/B. Therefore,
**duplicate physical identities are not demonstrated to be solved**. The profile improves
reliability of status reporting and limits some ambiguous matches, but cannot infer true identity
from nearby unlabeled clusters. A2/C box dominant share remains below Phase 3.6, although above
the original baseline; A1 ID flicker increases. This fails a strict "no sacrificed improvements"
criterion, so the profile must remain experimental. No blanket real-world success is claimed.

### Person and motion proxies

| Variant | B person IDs | Person coverage | Walking MOVING / covered walking | MOVING / all walking reference frames |
|---|---:|---:|---:|---:|
| baseline | 19 | 0.958 | 119/193 (61.7%) | 119/206 (57.8%) |
| improved | 20 | 0.948 | 74/189 (39.2%) | 74/206 (35.9%) |
| refined | 18 | 0.919 | 95/183 (51.9%) | 95/206 (46.1%) |

The final column has the same noisy foreground-reference denominator for every variant, preventing
exclusion of Lost tracks from artificially inflating the apparent recall gain. Walking response
improves over Phase 3.6 but is still below the original tracker. Static box-ROI MOVING counts
remain small but A2 rises 0→4. A passing person can contaminate the box ROI; these are not labelled
classification false-positive rates. Scene-wide MOVING counts, pause states, velocity proxies and
C still/turning/translating breakdowns remain in the JSON. No real precision/recall or IDF1 is asserted.

### Serial tracker-only runtime

| Test | Mean ms (original → 3.6 → refined) | p95 ms | Refined throughput Hz |
|---|---:|---:|---:|
| test_a1 | 20.79 → 29.09 → 21.4 | 25.95 → 35.78 → 26.63 | 46.7 |
| test_a2 | 20.7 → 29.16 → 21.34 | 26.32 → 36.46 → 27.7 | 46.9 |
| test_b | 20.61 → 29.09 → 21.5 | 26.11 → 36.08 → 26.86 | 46.5 |
| test_c | 25.47 → 36.07 → 26.31 | 36.22 → 49.74 → 36.69 | 38.0 |

Physical rejection occurs before expensive covariance gating for implausible Lost candidates.
These timings exclude LiDAR extraction, visualization and metric calculation. They do not establish
end-to-end live throughput or a real-time deadline. Earlier trial timings had concurrent tests or
workers and are not used for this final latency comparison.

## Reproduction on Mac

From the repository root, with the existing cached comparison and recordings present:

```bash
PYTHONPATH=src .venv-go2/bin/python -m pytest -q
PYTHONPATH=src .venv-go2/bin/python scripts/compare_tracking_real.py \
  --reuse-cache --serial \
  --cache-root data/outputs/mac/tracker_comparison/inputs \
  --variants baseline current_default improved refined \
  --output data/outputs/mac/tracker_refinement_repeat
PYTHONPATH=src .venv-go2/bin/python scripts/benchmark_tracking_robustness.py \
  --baseline de043d1 --baseline-config configs/tracking_robustness.yaml \
  --config configs/tracking_refinement.yaml --seed 36 \
  --output data/outputs/refinement/synthetic_repeat
```

Commands were checked against the actual CLI; local execution uses `.venv/bin/python` with
explicit PYTHONPATH, avoiding the Mac's stale installed package. Choose a new output directory
on every real run. Without a cache, omit --cache-root and --reuse-cache to extract locally.
The person proxy requires the existing `data/outputs/mac/test_b_tracking/raw_foreground_traj.npy`;
otherwise person results are unavailable, not inferred. No robot connection is required.
All private cache, compact tracks, plots and recordings stay local and gitignored. Only
aggregate metrics/configuration are published. No learned forecasting or control was added.

## Development evidence and selection limits

Six local candidates were checked on the same four recordings. These are development inputs,
not a held-out benchmark, and the selected parameters are not calibrated. The initial narrow
physical gate (.5 m/s velocity margin, applied to all missed tracks) reduced A1 dominant share
from .995 to .554 and increased duplicates; it was rejected. Strict confirmed-before-Lost
assignment reduced duplicates but also fragmented static tracks, so it was removed. The final
code retains global assignment, a physically bounded but less restrictive velocity allowance,
and explicit Lost status after the first missed observation. Prior trial outputs remain local
under `data/outputs/mac/tracker_refinement`, `_v2` through `_v6`; final serial results are separate.

The selected profile is **not a uniform correction of all real regressions**. It is an opt-in
tradeoff pending annotations. In particular, dominant static-box continuity can remain worse
than Phase 3.6 even while long-distance reassociations and moving recall improve. Do not
promote it to default or claim that identity robustness is solved.

## Remaining evidence required

Conservative overlap cannot distinguish two truly co-located similarly moving objects. It can
leave less-overlapping duplicates unresolved or allow rebirth after suppression. It deliberately
does not transfer an observed younger track's measurement into a lost older identity. Long
occlusions beyond 1.5 s still expire, and missing odometry resets all tracks. Speed limits can
reject genuine abrupt motion. Shorter MOVING dwell can restore false positives as well as
recall. Reactivation limits and changed observation status can reduce continuity/coverage. Repeated-evidence processing adds work; measured end-to-end real-time operation is
not established. No calibration or live hardware validation is added.

Annotate identities/visibility on A2/B/C, validate two close people (D), and test slow starts,
pauses, acceleration and drift on held-out recordings. Keep the profile experimental and defer
learned trajectory forecasting until target identity quality is established.
