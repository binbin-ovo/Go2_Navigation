# What changed after the six November 19 box trials

On the saved post-approach image, the application now recognizes the enrolled box. The recorded 2 m area still blocks the next step. The four earlier forward SEARCH dispatches also hold when evaluated against their original turn-clearance evidence. No new physical trial was run.

The development branch is `codex/physical-evidence-navigation`. Its ancestry includes Claude’s integrated `d62d936c67fed4ce10d0f364188bbaa596fee970` and the preceding offline study `ba8a7b692d45afe6bb7a7a2228826e34ca14571f`. The evaluator loads the two baseline policy/grounding modules from `2fa98f20`. Their contents are identical at `d62d936c`. The physical records separately identify `918f48c` and the then-uncommitted boot timing change. [Physical report](FIVE_MOVE_NAVIGATION_VALIDATION.md) · [Prior ten-case study](OBJECT_GUIDED_NAVIGATION.md)

## From raw mark to movement

1. LightNav consumes the target instruction and causal rectified RGB history. The saved OPOS, native image, waypoints and STOP remain the model evidence.
2. A nonzero OPOS makes an object candidate. An operator can enroll or reject the marked object on its exact image. An offline category detector can propose another region for review.
3. The identity gate locates the enrolled instance independently in the current view. Native ORB remains the first check. When this gives insufficient evidence, five-move mode can match interior SIFT features on the official processor’s source image, using the job, cutoff, frame ID, epoch, hashes and uncropped rectification to bind it to the displayed mark. It selects keypoints whose nominal descriptor footprint fits inside the enrolled ROI and searches across the full current image. A positive SIFT mark must fall in both the projected ROI and the convex hull of the current inliers. Conflicting positive and rejected instances yield uncertainty. External reference photos without this binding retain the native matcher. [OpenCV 4.11 descriptor geometry](https://github.com/opencv/opencv/blob/4.11.0/modules/features2d/src/sift.simd.hpp#L646-L660)
4. APPROACH requires supported identity. A compatible straight prefix or collinear forward corridor can support a forward step. A sign-consistent near-term heading and an off-centre image bearing can support one separately checked 20° alignment turn if it improves the residual by more than the measured heading uncertainty and leaves it within the band. Small errors and contradictory curved paths hold. Fresh observation follows settling.
5. On inconclusive identity, five-move SEARCH inspects unvisited headings. A located off-centre reference can supply a direction. Otherwise the policy reads the first coherent native heading or uses scan history. It never takes native curved translation as permission for a turn. When both turns lack clearance or a useful new view, it records the refusal and ends the bounded attempt. A new mark without an enrollment can still require operator identification. The executor retains its separate dispatch checks. [Public policy](../navigation.py) · [Instance matcher](../src/go2_perception/vln/episode_grounding.py)

## Recorded causes and remaining uncertainty

The audit checked all **23** native predictions in six closed sessions against their saved jobs, selected frames, prompt identity, archived native output and displayed point. The native grid has 48 columns. Eleven outputs selected column 24 (centre fraction 0.5104), four selected column 25 (0.53125), and one returned absence. Six were ineligible at receipt or executor freshness checks. No native STOP occurred. These totals include rejected decisions. Image-based findings use the recorded calibrated 448 × 256 model view. The outer edge of the live camera can fall outside that view.

The saved token arithmetic, displayed point and processor image agree. The model itself repeatedly selected those cells. Previous matched instruction tests showed that descriptive wording could suppress both a hallway false positive and a visible edge target. To isolate history, three new local MPS calls retained the identical current image hash, target text, cutoff and checkpoint, and included only that current frame. The checkpoint was `LightOriginsHQ/LightNav-0@826dc5fbfa37afa8293d2e336d329b6ffc0bfb64`, official source `3015508b70fb6e30bbd66701e94c9b916dcb8df8`, with its recorded local patch, `VLNQwen3VLProcessor`, MPS BF16 and SDPA. Each decision reset the model. The official sampler then represented a one-frame history. This changes the temporal input and its model representation together.

| Same current image and “box” instruction | Eight archived frames | One frame | Interpretation |
|---|---:|---:|---|
| Hallway, `cdyo3G` P002 | 745 | 745 | The distractor persisted with history removed. |
| Edge, `THKUcr` P002 | 698 | 0 | Removing history suppressed the wrong hallway point. The edge box still lacked a correct native mark. |
| Centered box, `231Apz` P003 | 793 | 793 | The on-box point persisted. |

This isolates a history effect in one case. A general learned centre bias, attention to the hallway’s vanishing point and the separate contributions of individual past frames remain hypotheses. [History experiment](../scripts/diagnose_box_history.py) · [Prompt and camera review](TARGET_GROUNDING_REVIEW.md)

## Same-input before/after comparison

The frozen ten-case matrix reused the saved checkpoint outputs, all 80 selected image slots and the original reference timing checks. Each decision uses its own historical pose, area grant, inspection evidence and prior attempts. The two cover conditions are deliberate pixel interventions on every selected frame. The original images and reviews remain intact. Earlier cross-session enrollments and edited conditions are labeled counterfactual. “Review” indicates an image-bound operator decision is needed.

| Case | OPOS | Integrated baseline | Feature branch on the same inputs |
|---|---:|---|---|
| Hallway, no reference | 745 | Review | Review. The original operator rejected the mark. |
| Absent hallway after rejection | 745 | SEARCH forward | Rejected appearance recurs. Both turns blocked, HOLD. |
| Same hallway, descriptive instruction | 0 | SEARCH forward | Native absence, both turns blocked, HOLD. |
| Centered, after enrollment | 793 | APPROACH forward | Same 0.60 m native forward selection. |
| Closer box after the approach | 891 | Review | Correctly re-identified. HOLD at inspected-area boundary. |
| Clear right box, earlier cross-session photo | 800 | Review | Correctly identified. Requested right turn vetoed by 153 obstacle returns. |
| Clear right box, local enrollment | 800 | HOLD on forward-path mismatch | Requested 20° right turn vetoed by 143 returns. |
| Edge target, native mark on hallway | 698 | Review | Insufficient identity at the mark. Both turns blocked, HOLD. |
| Partially covered box | 793 | Review | Reference localized, but the covered mark lies outside observed inliers. HOLD. |
| Fully covered box | 745 | Review | Insufficient identity at the hallucinated mark. HOLD. |

**Re-identification.** At `231Apz` P004, native ORB had 12 inliers in 23 mutual matches (52.2%). The detail fallback had 23/24 (95.8%), feature spread 0.1213 and median residual 0.604 px. Earlier photo to `bYheuc` P002 gave 28/31 (90.3%), spread 0.1843 and residual 0.403 px. The acceptance thresholds remain ≥12 inliers, ≥0.80 inlier fraction, ≥0.12 spread across the enrolled ROI and ≤2 px median residual. The closer case is just above the spread threshold. In the partial cover, 18/19 inliers located the visible surface but could not support the mark on the cover. These are development images used to choose and check the matcher.

**Semantic check.** Grounding DINO tiny `a2bb814dd30d776dcf7e30523b00659f4f141c71` ran locally, CPU FP32, four threads, query `box.`, box threshold 0.35, text threshold 0.25. It included both operator-identified marks and excluded both operator-rejected marks. It found a box at the edge while LightNav pointed at the hallway. A category region alone cannot identify one particular box. The new instance gate supported both reviewed positives with a prior enrollment and withheld APPROACH on both reviewed negatives. Only four distinct original marks have these image-bound labels in the comparison. Session descriptions and synthetic covers do not increase that denominator. DINO remains a review diagnostic. [Category experiment](../scripts/compare_box_regions.py)

**Timing.** Across the 23 physical-run outputs, original LightNav inference was 1.92–14.03 s, median 2.03 s. The saved ten-case inference median was 1.80 s. In the matched offline run the fast ORB decisions took about 6 ms and the five attempts using detail took 65–79 ms, including current-image read and feature extraction. Enrollment is a separate one-time cost. These timings exclude capture and dispatch. The independent CPU detector took 1.87–2.18 s per image, median 1.90 s. Adding the separately measured inference times gives 3.65–4.81 s, above the 3 s freshness threshold in every case. The live application now records time spent verifying each observation. [Private artifact names and reproduction below](#reproduce-on-the-mac)

## What the guards saw

| Recorded decision | Physical dispatch | Observed turn-disc obstacle returns | New selection with recorded geometry |
|---|---|---:|---|
| `cdyo3G` P003 / P004 | SEARCH forward twice | 97 / 130 | HOLD / HOLD |
| `THKUcr` P003 / P005 | SEARCH forward twice | 209 / 247 | HOLD / HOLD |
| `231Apz` P003 | APPROACH forward, requested 0.60 m | 156 | Same forward approach on its independently clear forward corridor |

All four SEARCH snapshots reported only the forward sector and zero measured turn clearance. The operator-inspected static-disc capability still requires floor support, zero observed obstacles in the swept disc, forward drift reserve and the current area grant. Missing rear observations supply no new clearance. The logs establish the guard predicates. They cannot distinguish every return on the robot, cable, floor clutter and real objects.

The `THKUcr` P001 native chunk did supply a rightward search hint (−1.06 rad). Its recorded observation failed executor freshness and never authorized a turn. Later forward SEARCH dispatches used different observations. In `bYheuc` P003 the target was about 14° right, with a curved native path starting about 18° right. Changing **only** the 143 recorded obstacle returns to zero in a controller diagnostic produced a 20° right turn at 2 rad/s. The recorded case still held. In `THKUcr` the image-bound event log records a reject at P004 and the seek-new-view interaction at P005. Use those event IDs for replay where the narrative table groups them differently.

The separate closed-loop **scripted chair fixture** turned left to SEARCH, observed the chair, selected a forward APPROACH and then received a fixture STOP. Its arrival remained unmeasured. The public mock five-move browser regression also retained the four-frame reference sequence, including a turn to align, forward motion and cleanup. Replay makes single-decision comparisons along the operator’s historical path. It cannot establish how a new physical turn would change the next observation.

## Reproduce on the Mac

```bash
export GO2_WORKSPACE=/Users/yushan/Robotics/Go2
cd "$GO2_WORKSPACE/code/worktrees/codex-motion-diagnostics"
export GO2_HARDWARE=0 GO2_SUPERVISED_LIVE=0 MUJOCO_RENDER_TEST=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/hardware"
py="$GO2_WORKSPACE/envs/core/bin/python"
runs="$GO2_WORKSPACE/runs/target_grounding"
"$py" app.py review --session "$GO2_WORKSPACE/runs/five_move/trial-231Apz/session"
"$py" scripts/evaluate_object_navigation.py --hardware-report \
  --reuse-inference "$runs/object-navigation-5ggvlnu9" \
  --dino-snapshot /private/tmp/go2-object.NHTKO8K7/public-detector/a2bb814dd30d776dcf7e30523b00659f4f141c71
"$py" scripts/diagnose_box_history.py
"$py" scripts/simulate_object_search.py --backend fixture
```

The evaluator accepts any identical local DINO snapshot named for its revision and verifies the weight checksum and processor files. Omitting `--dino-snapshot` runs the recorded image/geometry comparison without loading either model. The history probe requires the existing cached LightNav MPS environment and checks its identity against each original worker. Run the jobs sequentially.

Private reports under `runs/target_grounding` are `object-navigation-7n88hu29/index.html`, `results.json` and `six-trial-audit.json` in that same directory, `box-history-c1jo5gil/results.json`, `trace-k79ee3te/trace.html` and `object-sim-fixture-m34rvgc9/index.html`. Each command prints its newly created report location.

The loopback replay command in the walkthrough displayed scan 2585 from `231Apz` chunk 1 with its original robot stamp, epoch and monotonic receipt time. The public state showed a 10 GiB disk floor, 1 GiB quota and zero robot commands. The RGB shown in that joint interface smoke test came from the fixture. The separate historical HTML shows the actual box images. All six LiDAR archives were also read to completion, totaling 3,218 scans and 31,284 odometry records. The reader validates chunk hashes, offsets, sequence and timing metadata, streams one chunk at a time, and clears accumulation at recorded gaps and epoch transitions.

```bash
LIGHTNAV_ASSETS= "$py" -m pytest -q -rs -p no:cacheprovider \
  tests/test_navigation_evidence.py tests/test_five_move_navigation.py \
  tests/test_object_navigation_offline.py tests/test_target_grounding_review.py tests/test_application.py
LIGHTNAV_ASSETS= "$py" -m pytest -q -rs -p no:cacheprovider
modelpy="$GO2_WORKSPACE/envs/model/bin/python"
export LIGHTNAV_ASSETS="$GO2_WORKSPACE/weights/lightnav"
"$modelpy" -m pytest -q -rs -p no:cacheprovider tests/test_bounded_native_integration.py tests/test_lightnav_runtime_integration.py
"$modelpy" -m pytest -q -rs -p no:cacheprovider tests/test_lightnav_runtime_integration.py tests/test_bounded_native_integration.py
```

Focused tests passed 80/80. The full core run passed 576 with 13 skips. Twelve skipped tests require the isolated model runtime and passed 12/12 in each file order. The remaining skipped test subscribes to physical Go2 streams.

## Claude’s hardware handoff

Review the feature commit and use a clean checkout. Both preparation and live supervised startup verify navigation provenance. The existing capability ID now carries the full clean navigation SHA, and supervised startup compares it with the running checkout. Both contract versions retain their published field sets. Each new run records HEAD, Git status and, for tracked edits in offline runs, a digest of the working diff. The previous development commit introduced inspected-view memory in the five-move executor. Carry the reviewed versions together and generate a new capability. The canonical controller, guard provider and motion profile retain their hardware baselines.

```bash
"$py" app.py supervised --prepare-only --move-budget 5 --episode-seconds 60 \
  --area-radius-m 2 --alignment-deg 15
```

Preparation is file-only. For a later authorized session, verify the boot-specific capture-time evidence, exclusive area/cable inspection, manufacturer avoidance, control ownership, cancellation and stopping. Record a genuinely clear initial off-centre turn and an obstructed refusal, the enrollment and settled post-movement image hashes, instance evidence, the selected primitive, raw guard snapshot and settled heading. Measure physical distance independently if reporting travel. Count requested, executed and stopped moves separately. Evaluate identity on new boxes, backgrounds, occlusions and lighting before estimating generalization. The five-move limits remain 1.0 m/s forward, 2.0 rad/s turn and a default 15° alignment preference. Continuous execution remains offline.
