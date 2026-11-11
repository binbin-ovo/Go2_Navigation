# Phase 5B real-recording validation — 2026-11-11

Development recordings A1, A2, B and C, replayed through the original 0.3 s pipeline and through
Phase 5A, then rendered with the [FSD-style viewer](FSD_STYLE_VISUALIZATION.md). All four recordings
are development data; none is held out. Annotations are LiDAR-derived and reviewed by Claude, **not
human-verified**. They are read only after perception, for scoring and for the labelled retrospective
overlay. No robot was commanded; nothing here changes an operational default.

**Bottom line.** 5A looks calmer, and part of that is real: it removes almost every false MOVING label,
and residual static jitter halves on A1, B and C (A2 is slightly worse). But much of the visual
steadiness comes from **held stale geometry**, not from better observation. Fresh coverage and moving
recall regress badly. The Test B diagnosis traces most misses to one 5A design rule: an object counts
as fresh only with ≥3 points in the **latest ~62 ms incremental scan**, and that scan often does not
sweep the person at all. The viewer shows this as dashed gray boxes; it does not hide it.

## Inputs and provenance

- 5A outputs regenerated with `scripts/phase5b_scene_tap.py` (instrumented replay of the unchanged
  `StableScene`, commit `be4b9de`). All **3,926 rows match Codex's release `scene.jsonl` exactly**
  (processing time excluded); `reference_mismatched_frames` is empty for all four recordings.
- Original pipeline: the cached baseline tracker runs (`ground_truth_eval/tracks/baseline`, `ecfaae3`)
  and their cached inputs (`tracker_comparison/inputs`), 0.3 s odom windows.
- Matching is the 5A evaluator's (`evaluate_stable_scene_real.evaluate`): per-frame Hungarian, 0.5 m to
  a person point, 0.15 m to the box polygon, IDs ignored. The perception numbers below reproduce
  [STABLE_LIDAR_RESULTS](STABLE_LIDAR_RESULTS.md) exactly.

## Perception (fresh coverage and moving recall, kept separate)

Each cell reads original → 5A.

| Recording/object | Fresh coverage | Moving recall | False MOVING (matched stationary) | MOVING delay median s (activated/events) |
|---|---|---|---|---|
| A1 box | 0.854 → 0.171 | — | 0.018 → 0.000 | — |
| A2 box | 0.758 → 0.129 | — | 0.047 → 0.000 | — |
| A2 person | 0.735 → 0.455 | 0.620 → 0.121 | 0.379 → 0.000 | 0.23 (12/18) → 1.33 (6/18) |
| B box | 0.677 → 0.090 | — | 0.098 → 0.000 | — |
| B person | 0.905 → 0.499 | 0.575 → 0.058 | 0.209 → 0.000 | 0.23 (8/14) → 1.36 (6/14) |
| C box | 0.693 → 0.381 | — | 0.049 → 0.000 | — |

Delays are from the annotated moving onset, excluding left-censored events. A null delay means the
event never activated; it is not zero.

## Visual stability (what a viewer sees; not accuracy)

| Recording | Static-box step p95 mm | Static-box residual jitter m | Appear /min (≤3.5 m) | IDs living <0.5 s | Unsupported MOVING frames/min | Stale/coasting run mean / max s | Box not-fresh frames (fraction still displayed) | Person not-fresh frames (fraction displayed) |
|---|---|---|---|---|---|---|---|---|
| A1 | 51 → 23 | 0.101 → 0.048 | 298 → 319 | 36 → 8 | 607 → 1 | 0.25/0.66 → 0.20/0.59 | 135 (1.00) → 759 (0.86) | — |
| A2 | 67 → 22 | 0.077 → 0.084 | 338 → 372 | 37 → 4 | 730 → 5 | 0.26/0.66 → 0.20/0.59 | 138 (0.94) → 485 (0.78) | 152 (0.50) → 270 (0.89) |
| B | 69 → 20 | 0.100 → 0.046 | 333 → 377 | 38 → 2 | 724 → 0 | 0.26/0.66 → 0.20/0.59 | 214 (0.72) → 575 (0.51) | 56 (0.77) → 278 (0.84) |
| C | 63 → 38 | 0.089 → 0.051 | 301 → 329 | 35 → 6 | 841 → 0 | 0.26/0.66 → 0.19/0.59 | 293 (0.89) → 581 (0.97) | — |

Definitions:

- **Step:** frame-to-frame XY movement of the displayed estimate matched to the static box (same ID,
  fresh or stale). 5A's median step is 0 mm **because the box is stale most of the time**: held
  geometry does not move.
- **Residual jitter:** the evaluator's std of fresh matched centres around the annotated box.
- **Appear /min:** displayed IDs born within 3.5 m. Both pipelines churn about 300/min. 5A's short-lived
  flicker drops, but its total births are slightly higher.
- **Unsupported MOVING:** MOVING-coloured displays within 3.5 m that are not within 0.5 m of an
  annotated walking person. The original paints walls and clutter MOVING at about 600–840 frames/min;
  5A at 0–5.
- **Not-fresh frames:** annotated object visible but no fresh match. The fraction is how often a stale
  or coasting box was still displayed within the gate. 5A has 2–5× more not-fresh frames for the box
  and person, bridged by stale boxes. In the viewer they are dashed and gray, never solid.
- Stale share of all emitted 5A boxes: 0.69 in every recording (original about 0.45 coasting).

## Runtime

| Measure | Value |
|---|---|
| 5A perception compute per scan (four replays in parallel) | mean 20.7–22.4 ms, p50 18.3–18.7 ms, p95 23.3–24.1 ms |
| Original tracker only (cached, not end to end) | mean 23.3–28.9 ms, p95 28.9–40.9 ms |
| Viewer render, 1920×1080, unloaded | Clean about 17 fps, Debug about 10 fps (`fsd_viewer.py` export log) |
| Viewer render under 8 parallel jobs | about 3 fps (A2 full export) |

Real-time playback is 15.4 fps. Perception latency excludes JSON serialization, file I/O and
transport; neither pipeline has a measured end-to-end live latency.

## Test B Debug diagnosis: walking person present, no fresh box

`scripts/phase5b_diagnose_walking.py` checks every frame where the annotation marks the person
visible and walking: 240 frames in B, 321 in A2. A cylinder (0.45 m radius, floor + 0.05 to 2.0 m)
around the annotated position selects points from the recorded 5A intermediates: current scan,
0.1 s object window with stage codes, 0.3 s non-ground window, proposals with current-scan support,
and motion-gate internals. The first failing stage is reported. Categories are defined in
`diagnostics/attribution.py`.

| Test B, 240 walking frames | Frames | Share |
|---|---|---|
| **Absent from the current scan** (<3 current-scan returns on the person) | 91 | 0.379 |
| Fresh but not MOVING: speed − 2σ gate | 56 | 0.233 |
| Fresh but not MOVING: object too young (<4 observations over 0.4 s) | 23 | 0.096 |
| Fresh but not MOVING: activation streak (<3 consecutive) | 21 | 0.087 |
| Missed while present: short-window point support (<8 non-ground points at 0.1 s, ≥8 at 0.3 s) | 15 | 0.062 |
| Missed while present: foreground clustering (DBSCAN noise or off-centre merge) | 14 | 0.058 |
| Recalled MOVING | 14 | 0.058 |
| Missed while present: ground/background (vertical-plane) suppression | 3 | 0.013 |
| Missed while present: lifetime/stale (absorbed by an older object) | 2 | 0.008 |
| Jump/shape gate | 1 | 0.004 |

- AABB/proposal rejection and minimum-support thresholds did not occur in B; A2 had 4
  minimum-support frames.
- Fresh coverage on these frames is 0.479 (original 0.883); moving recall is 0.058 (original 0.579).
- Of the 148 frames where the person **is** in the current scan, 114 get a fresh box. Of the 34 that
  do not, the original had a fresh detection in 30.

**Evidence for the dominant cause.**

- In 88 of the 91 absent frames the current scan has **zero** returns on the person.
- 89 of them still have ≥8 non-ground person points in the 0.3 s window, 61 even in the 0.1 s window,
  and the original pipeline had a fresh detection in 81.
- At the person's bearing (±8°), the current message has a median of 0 non-floor returns, and nothing
  behind the person either (94.5% of absent frames). Present frames have a median of 50 non-floor
  returns at the same bearing.
- So the incremental `cloud_base` message did not sweep that sector above the floor. This is a sensor
  sampling gap, not occlusion. 5A's rule that a box is refreshed only by ≥3 latest-message points
  turns it into stale boxes; 79 of these frames had a stale box near the person.

**Motion gates.**

- On fresh, matched walking frames blocked by the speed gate, the median regressed speed is 0.44 m/s
  (82% above 0.25 m/s), but the heuristic σ has median 0.23 m/s. Speed − 2σ is therefore about −0.05,
  far below 0.25.
- The second cause is object re-creation. After a stale gap the object is reborn, its history resets,
  and it needs four observations over 0.4 s plus three consecutive candidates.

**A2 replicates the pattern** (321 walking frames):

| A2 category | Share |
|---|---|
| Absent from current scan | 0.377 |
| Speed gate | 0.156 |
| Recalled MOVING | 0.121 |
| Too young | 0.090 |
| Activation streak | 0.084 |
| Short window | 0.062 |
| Clustering | 0.053 |
| Jump/shape | 0.025 |
| Ground/background | 0.019 |
| Minimum support | 0.012 |

Timestamped frames, details and the longest episodes are in the local JSON files
`data/outputs/phase5b/diagnosis/test_b.json` and `test_a2.json`. The failure montage shows one episode
per category; the montage table below lists their times.

### Other findings from Debug mode

- **Self-return object.** In 98% of A1, B and C frames, 5A emits a *fresh* object centred within
  0.45 m of `base_link`, about 0.43 × 0.75 m and 0.06–0.24 m tall, at floor height. These are
  near-field floor or leg returns. A planner would read them as an obstacle at the robot's own
  footprint. The original pipeline also tracks a cluster there.
- **Ghost background cells.** In the synthetic `start_move` case (and on real data in Hybrid mode),
  cells from an object that started moving stay in the background map until the 2 s TTL. This is
  documented in 5A and visible in the viewer.
- **Unmeasured velocity.** About 54% of all emitted objects (sampled every 5th frame) carry 5A's
  identity-covariance prior with zero velocity. The viewer shows `v n/a (prior)` and draws no arrow.

## Recommended fixes for Codex (ordered by evidence)

1. **Freshness from recent coverage, not from the single latest message.** Refresh an object when it has
   ≥`current_support_min` returns in the last *k* incremental messages (2–3, about 0.13–0.2 s), or
   track per-azimuth/elevation sector coverage. If the object's sector was not swept by the current
   message, keep the previous state and mark it "not re-observed yet", not stale. This targets 38% of
   B and A2 walking frames, where the person is plainly present at 0.3 s. Keep `observed_box` limited
   to points that are really current, so freshness stays honest.
2. **Motion activation that survives realistic σ.**
   - Replace `speed − 2σ > 0.25` by a displacement test over the full window (e.g. net displacement
     ≥ 0.25 m over ≥ 0.4 s with a consistent direction), or use `speed − 1σ`.
   - Estimate σ from `observed_box` centres, which are less smeared than accumulated-proposal centres.
   - Validate any change on held-out data. B's median σ of 0.23 m/s makes 82% of truly fast frames fail.
3. **Keep motion history across short stale gaps.** If a reborn object associates within the gate to a
   recently expired one (<1 s), inherit its history instead of resetting. 23 B frames (10%) failed
   only because the object was too young.
4. **Robot self-filter before proposals.** Add a configured base_link footprint crop (Go2 body and
   legs, e.g. |x| < 0.40 m, |y| < 0.22 m, z below the body) to proposal generation, not to the raw
   recording or the background evidence. Report it as a separate diagnostic count.
5. **Clustering for legs.** 14 B frames are DBSCAN noise or off-centre merges with ≥8 points. Try a
   per-height-band or slightly larger eps for components of person-like height. Measure split and
   merge on held-out H2 (two people) before adopting.
6. **Do not lengthen object windows blindly.** See the 0.3 s note below. Any window change
   must be reported with both coverage/recall and dimension smearing (5A's synthetic study shows
   1.0 s doubling dimension error).

### Diagnostic counterfactual: object window 0.3 s (not a default change)

This was attempted with `phase5b_scene_tap.py --set object_window_s=0.3`, then **stopped
unfinished**. After an hour each replay had processed only about 200 of roughly 1,000 frames (about
9–18 s per frame, using about 4.6 cores, while other jobs ran), versus about 20 ms per frame at 0.1 s.
The likely hot path is 5A's novel-split check, `BackgroundMap.stable_mask`, which loops over points in
Python and receives far more points from larger windows. That is not profiled. Either way, a 0.3 s
object window is not usable live without vectorizing it. No 0.3 s 5A numbers are claimed.

What *is* measured about 0.3 s:

- The diagnosis counts non-ground person points in a 0.3 s window directly: 89 of the 91 "absent"
  frames would have ≥8.
- The original pipeline is itself a 0.3 s window pipeline: fresh person coverage 0.905 and moving
  recall 0.575 in B. It pays with false MOVING on clutter (about 724 frames/min) and higher box jitter.
- 5A's own synthetic study shows longer windows smear moving objects (dimension error 0.124 m at
  0.1 s → 0.275 m at 1.0 s) and delay MOVING.

Longer is therefore not simply better. Fix 1 (recent-coverage freshness) addresses the measured cause
without enlarging the clustering window.

## Videos (local, private, git-ignored)

All are 1920×1080 H.264 (yuv420p) in `/Users/yushan/Desktop/Go2_Navigation/data/outputs/phase5b/videos/`:

| File | Content |
|---|---|
| `test_a1_clean.mp4`, `test_a2_clean.mp4`, `test_b_clean.mp4`, `test_c_clean.mp4` | Full-length Clean-mode split view (RGB left, 3D right) |
| `highlights.mp4` | Successes section, then a separate failures section, captioned |
| `raw_vs_accumulated_vs_boxes.mp4` | Three panels at the same instant: current scan / 1.0 s accumulation (age colours) / 5A boxes; B pass 1 and C turns |
| `original_vs_5a.mp4` | Original clusters + tracks vs 5A boxes, B and A2 full length, with a labelled retrospective annotation ring |
| `walking_person_failures.mp4` | One clip per failure category from the B diagnosis, title cards, evidence lines |
| `static_box_stability.mp4` | A1 and C static box, original vs 5A, captioned with step jitter *and* fresh coverage |
| `clips/successes/`, `clips/failures/` | The individual clips, kept apart |

Failure montage clips (B; each runs from 1.5 s before to 1.5 s after the episode):

| Category | Episode (s from start) |
|---|---|
| Absent from current scan | 14.68–15.01 |
| Speed gate | 41.20–41.39 |
| Too young | 22.48–22.67 |
| Activation streak | 40.81–40.87 |
| Short window | 39.38–39.44 |
| Clustering | 52.37–52.44 |
| Ground/background | 50.68 |
| Lifetime/stale | 40.48 |

## Sensor, calibration and scan-timing limits

- **Scan timing.** `cloud_base` messages are thinned incremental scans spanning about 62 ms, about 65 ms
  apart, with per-point offsets (5A audit). This diagnosis shows they are also **angularly partial**:
  a given bearing is often not swept above the floor in one message. Firmware point-frame semantics
  during motion are unverified. 5A uses midpoint poses on real data; point-level deskew is only
  synthetic-tested. C has potential within-message motion distortion (p95 sensitivity about 0.1 m).
- **Camera.** Uncalibrated (no intrinsics or extrinsics). Frames are aligned by host receipt; the
  median offset is about 15 ms, p95 about 34 ms, maximum 92–137 ms. No camera↔LiDAR association is
  drawn.
- **Odometry.** Drift is unmeasured. The C box moves up to about 0.2 m in odom (Phase 3.8). Missing
  odometry (one frame each in A1 and C) advances `epoch`, and the viewer clears its history.
- **Annotations.** LiDAR-derived and not human-verified. The cylinder test is an evaluation-time oracle
  selection, not segmentation.

## Planner readiness (no planner, no control)

`viewer/planning.py` answers only occupied (fresh object, stale object, background cell) or
**unknown**; it never answers free. Stale and missing detections can therefore never read as free
space. What a future collision-aware planner needs, against v1:

| Need | v1 | Status |
|---|---|---|
| Robot pose and timestamp | `robot_pose`, `timestamp`, `clock_domain` | met |
| History validity | `epoch` | met |
| Static occupancy | background cells, evidence/recency score | partial: the score is not a calibrated probability; cells expire to unknown |
| Freshness | `status`, `observation_age_s`, `last_observed` | met (but see the 62 ms freshness rule above) |
| Moving obstacles | `center`, `velocity`, `velocity_covariance`, `motion_state` | partial: about 54% of objects carry the prior; the covariance is heuristic; recall is 6–12% |
| Extent and heading | AABB `dimensions`; `orientation_xyzw` null | partial |
| Free space | none (`unobserved_is_unknown`) | **gap**: needs raycast or visibility evidence |
| Unknown / occluded regions | implicit only | **gap**: no FOV, coverage or occlusion mask; the per-message sector coverage found above should be exported |
| Semantic class | none | **gap**: needs camera calibration first |
| Prediction | velocity only | **gap**: forecasting needs validated identities |
| Self-returns | not separated | **gap**: the fresh object inside the robot footprint (above) |

These gaps can be closed with **additive** optional fields (e.g. a `coverage` sector mask, a `free_cells`
list with raycast evidence, `self_filtered_points`) without breaking v1, as the interface allows.

## Held-out H1–H3 (prepared, not recorded, not tuned on)

The frozen protocol is in [`configs/phase5_heldout_protocol.yaml`](../configs/phase5_heldout_protocol.yaml).
It inherits H1–H3 from [TRACKING_GROUND_TRUTH](TRACKING_GROUND_TRUTH.md#held-out-evaluation-protocol-task-5),
adds an azimuth-sampling walk around the robot to H1, freezes the pipelines, renderer settings and
metrics, and pre-declares the gates for proposing any default change. The robot is moved by hand
only; there is no motion command of any kind.

## Remaining work before any live deployment

1. Codex fixes 1–4 above, then rerun this exact comparison and the diagnosis on A1–C (development) and
   **once** on H1–H3.
2. Record H1–H3 with taped poses; human-review at least 10% of the annotations.
3. Verify `cloud_base` sector coverage and point-frame timing during motion on the robot (Claude,
   hardware).
4. Camera intrinsics and extrinsics before any semantic labels or camera↔LiDAR association.
5. Live read-only viewer: feed `StableScene.process` from the existing live adapter, rendering in a
   separate process. Measure end-to-end latency (sensor stamp → display) and dropped frames. Not done
   here.
6. Additive schema fields for coverage, free space and self-filter, before a planner consumes v1.
