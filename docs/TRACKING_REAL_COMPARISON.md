# Original vs improved tracker on real Go2 recordings — 2026-11-07

This report compares the original tracker (`ecfaae3` code with `configs/tracking.yaml`) against the
improved profile (`a4496a3` code with `configs/tracking_robustness.yaml`). Both were replayed
offline on the private recordings A1, A2, B and C, on the Mac. No robot was connected. The
recordings and per-frame outputs stay local and gitignored.

What is committed is limited to:
- this report;
- aggregate metrics in [TRACKING_REAL_COMPARISON_METRICS.json](TRACKING_REAL_COMPARISON_METRICS.json);
- four LiDAR-only plots;
- the harness, `scripts/compare_tracking_real.py`.

Nothing committed contains camera images.

## Identical inputs, verified
- **Inputs.** LiDAR clusters were extracted once, with the unchanged `go2_lidar.yaml`, a 0.3 s
  window and the existing pose gate. The extraction code is identical in `ecfaae3` and `a4496a3`
  (`git diff` is empty for replay, pose gate, adapters and LiDAR code). Every variant then
  replays the same cached frames in its own process.
- **Separate code paths.** The baseline variant imports a `git archive` copy of `ecfaae3`; every
  other variant imports the working `src/`. The harness checks the imported package path for each.
- **Input fingerprints.** The SHA-256 hashes of the four input files match Codex's cache in
  `data/outputs/robustness/real` exactly, so extraction is deterministic.
- **Baseline matches the original runs.** The fresh baseline replay reproduces the original Phase 3
  outputs in `data/outputs/mac/test_*_tracking/tracks.jsonl` frame for frame. Track ID sets,
  status and motion label agree in **924/924, 769/769, 1001/1001 and 1230/1230** frames; the
  largest position difference is 5e-5 m, which is output rounding.
- **No regression in the default path.** Running the working code with the default
  `tracking.yaml` gives output identical to `ecfaae3` in every frame of every recording.
- **Scope.** The working tree was clean at `a4496a3`. The comparison covers 3,924 pose-valid frames.

## Ground truth and proxies
There are **no annotated identities, boxes or labels** for these recordings, so this report gives
no IDF1, MOTA or precision/recall. Every number below is an observable proxy:
- **Box.** In each frame, take the nearest non-Lost track within 0.3 m of the box position. The
  positions come from the original report:
  - A1: (2.50, 0.01) m;
  - A2: (2.49, 0.00) m;
  - B: (2.50, 0.00) m, all three in the static robot's `base_link`;
  - C: (1.95, −0.40) m in odom.
- **Box metrics:**
  - *Coverage:* fraction of frames with such a track.
  - *ID changes:* how often the selected ID differs from the previous frame's.
  - *Returns:* changes back to an ID already seen, meaning an intruding track briefly took the box
    and the original ID came back.
  - *Dominant share:* the most common ID's fraction of covered frames.
- **Person (B).** A per-0.2 s median of foreground LiDAR points, using the same background
  subtraction as the original report, with speed from a 0.6 s finite difference. This reference is
  noisy and is not ground truth.
- **Scene-wide:**
  - *Duplicates:* frames containing two Confirmed tracks closer than 0.3 m.
  - *Long reactivation jumps:* a track that returns from Lost more than 1 m from where it was last
    observed.
- **Motion.** The box and most of the scene are physically static, so MOVING on them is a false
  label. That covers all of A1, A2 and B apart from the people, and all of C in odom.

## Results (baseline → improved)

### Identity and continuity
| Test | Created IDs | Confirmed IDs | Box dominant-ID share | Box ID changes (returns) | Box unique IDs | Box coverage |
|---|---:|---:|---:|---:|---:|---:|
| A1 | 514 → 261 | 471 → 240 | 0.993 → 0.995 | 4 (2) → 6 (3) | 3 → 4 | 0.904 → 0.903 |
| A2 | 464 → 243 | 421 → 222 | **0.498 → 0.960** | 15 (5) → **28 (17)** | 11 → 12 | 0.853 → 0.856 |
| B | 624 → 333 | 576 → 301 | **0.311 → 0.489** | 13 (1) → 14 (5) | 13 → 10 | 0.808 → 0.806 |
| C | 699 → 334 | 652 → 315 | **0.327 → 0.585** | 30 (9) → **30 (22)** | 22 → 9 | 0.776 → 0.783 |

- **After occlusion.**
  - A2: the box keeps #22 through the person's three visits; the original went through #22, #146,
    #140 and #247.
  - B: the box keeps #21 through passes 1 and 2, to about 29 s, where the original switched to
    #101 at about 10 s. Pass 3 still changes the ID (#171, then #211).
- **During turns (C).** The box is mostly #16, with #44 from 17 to 33 s, through the −23° and +21°
  turns. The original changed through #16, #79, #179 and #430.
- **The changes count is unchanged because of flicker.** Brief intruding tracks take the anchor
  for a frame or two, then the main ID returns: returns rise 9 → 22 in C and 5 → 17 in A2.
- **Person fragmentation is not fixed:**
  - B: coverage 0.958 → 0.948, person IDs **19 → 20**, ID changes per pass 3/7/24 → 7/8/20.
  - A2 standing spot: 29 → 24 IDs within 0.5 m.

### Duplicates and association errors
| Test | Duplicate-pair frames | Lost tracks per frame | Reactivations > 1 m from last observation | Fragment merges |
|---|---:|---:|---:|---:|
| A1 | 365 → 483 | 11.9 → 18.6 | 77 → 99 | 0 → 48 |
| A2 | 277 → 383 | 12.2 → 19.2 | 62 → 114 | 0 → 44 |
| B | 380 → 504 | 11.9 → 19.2 | 96 → 163 | 0 → 83 |
| C | 682 → 832 | 12.8 → 20.1 | 88 → 93 | 0 → 27 |

Example (A2): track #61 was born at (3.8, −1.9) m, then reappeared at (1.3, 1.15) m. It lasted 43 s
in total, 217 frames of it Lost, and ended at (1.4, −0.4) m. Without labels it is unknown which
physical objects it followed. This is the long-reactivation pattern the adaptive gate makes possible.

### Motion and stationary velocity
| Quantity | Baseline | Improved |
|---|---:|---:|
| A1 confirmed track-frames MOVING (static scene plus one person pass) | 1,550 (7.8%) | **127 (0.6%)** |
| A1 IDs ever MOVING | 231 | 36 |
| A1 box STATIONARY / MOVING frames | 52 / 5 | **446 / 0** |
| C box MOVING: robot still / rotating / translating | 39 / 0 / 0 | **2 / 0 / 0** |
| C box STATIONARY frames | 86 | 342 |
| B box MOVING frames | 50 | 2 |
| Box speed median, A1 / C (truth 0) | 0.069 / 0.067 m/s | 0.070 / 0.061 m/s |
| Box velocity σ (median), all tests | 0.139–0.141 m/s | 0.139–0.142 m/s |
| C box odom std (still), drift first→last 5 s | (0.085, 0.067) m, 0.110 m | (0.083, 0.067) m, 0.121 m |
| B walking person labelled MOVING (3 passes) | 119/193 (62%) | **74/189 (39%)** |
| B pausing person: STATIONARY / MOVING | 11 / 27 of 150 | 53 / 6 of 148 |
| B walking speed, track vs reference, per pass | 0.77/0.89/0.80 vs 0.65/0.74/0.80 | 0.68/0.81/0.72 vs 0.65/0.77/0.80 |
| B walking velocity error (median), per pass | 0.25/0.37/0.40 m/s | 0.26/0.30/0.45 m/s |

The velocity estimates themselves barely change: static speed is about 0.07 m/s and σ about
0.14 m/s in both. The gain in STATIONARY and the loss of false MOVING come from the label rule
(`stationary_speed` 0.25 m/s and 0.25 s dwell). The `improved_legacy_motion` ablation restores
the baseline labels exactly: A1 box 49 STATIONARY, B walking MOVING 122/189.

### Runtime (`tracker.update`, serial run, same machine)
| Test | Mean ms | p95 ms |
|---|---:|---:|
| A1 | 23.8 → 33.0 | 29.2 → 40.2 |
| A2 | 23.4 → 32.9 | 29.6 → 41.0 |
| B | 23.8 → 33.0 | 30.2 → 40.8 |
| C | 29.3 → 40.7 | 41.4 → 55.5 |

That is about 40% slower. The 15.4 Hz LiDAR period is 65 ms, and LiDAR processing already takes
about 40 ms.

## Ablations: which feature does what
Each ablation is the improved profile with one feature turned off. The metric values are in the
JSON.
- **Confirmed-track time expiry.** This drives both the continuity gain and the extra duplicates.
  - Without it, the dominant share falls back to baseline (A2 0.494, C 0.266).
  - Duplicates also return to baseline (A2 268, C 654), and created IDs return to about 515/473/610/710.
- **Adaptive gate.** This causes most of the long reactivation jumps.
  - Without it they fall to 66/65/116/64, at or below baseline.
  - C flicker drops (changes 22, competing 9).
  - Box continuity worsens in A1/A2 (dominant share 0.565/0.562).
- **Fragment recovery.** It has a small, mixed effect on real data.
  - Without it: A1 box changes 10 vs 6; B dominant share 0.655 vs 0.489; C 0.458 vs 0.585.
- **Motion rule (threshold plus dwell).** All of the label change comes from this rule; it has no
  effect on identity.

## Regressions introduced by the improved profile
1. **Flicker and duplicate tracks near objects.**
   - Duplicate-pair frames rise 22–38% in every recording, and Lost tracks per frame rise by about 60%.
   - Returns to an old box ID rise 2→3, 5→17, 1→5 and 9→22.
   - Competing tracks at the A2 box rise 9→22.
2. **Long-range re-association.** Reactivations more than 1 m from the last observation rise in
   all four recordings (largest: A2 62→114, B 96→163), mainly from the 2.5 m adaptive gate.
3. **Walking-person MOVING recall drops 62% → 39%** because of dwell and the higher stationary
   threshold. This matches the synthetic brief-stop regression.
4. **Latency rises about 40%** (C p95 55.5 ms).
5. Small: A1 box ID changes 4→6; B person passes 1 and 2 have more ID changes (3→7, 7→8); C box
   drift 0.110→0.121 m (not significant without ground truth).

There were no regressions in ego-motion compensation: no MOVING labels on the box while the robot
moves, and the same odom spread. Pose gating and resets are unchanged, and the default
configuration is bit-identical to `ecfaae3`.

## Recommendation
- **Merging the code as an opt-in profile: ready.** The default path is verified bit-identical on
  all real recordings and the test suite passes (161 passed, 1 hardware skipped). The profile
  clearly improves static-object continuity after occlusion and turns, and removes most false
  MOVING labels on static structure.
- **Making `tracking_robustness.yaml` the default: not ready.** On real data it trades fewer IDs
  for more duplicate and flickering tracks and long-range re-associations. It does not improve
  person identity, and it lowers MOVING recall for a walking person. All evidence is unlabelled
  proxies.

Suggested before promotion:
- suppress or merge a Lost track that overlaps a Confirmed one;
- cap the adaptive radius for reactivation by elapsed time times a plausible speed;
- revisit dwell and threshold for walking people;
- hand-label box and person IDs on A2/B/C so these proxies can be confirmed.

## Reproduce
```bash
PYTHONPATH=src .venv-go2/bin/python scripts/compare_tracking_real.py \
    --output data/outputs/mac/tracker_comparison --serial
# re-analyse without re-running: add --analyze; reuse extracted inputs: add --reuse-cache
PYTHONPATH=src:third_party/unitree_sdk2_python .venv-go2/bin/python -m pytest -q
```

Local outputs are in `data/outputs/mac/tracker_comparison/`:
- `inputs/`: shared frames and pose reports;
- `<variant>/<test>/tracks.jsonl`;
- `run.json`;
- `comparison_real.json`;
- `plots/`.

The original outputs in `data/outputs/mac/test_*_tracking/` and `data/outputs/robustness/` were not
modified.

Plots (left: baseline, right: improved; bottom: the box ID over time, labelled on runs of 3 s or
longer): [A1](media/compare_test_a1_bev.png), [A2](media/compare_test_a2_bev.png),
[B](media/compare_test_b_bev.png), [C](media/compare_test_c_bev.png).
