# Real-world tracking ground truth — Phase 3.8 (2026-11-08)

Branch `phase3-ground-truth-validation` is based on `phase3-tracking-refinement` at `c643422`.
This phase adds no tracker change. It adds:
- a local annotation workflow;
- sensor-based annotations of the A1, A2, B and C recordings;
- an evaluator;
- a measured comparison of the original, Phase 3.6 and Phase 3.7 trackers against those
  annotations.

The recordings, camera frames, annotation files and per-frame outputs stay local and gitignored.
What is committed: code, this report, aggregate metrics in
[TRACKING_GROUND_TRUTH_METRICS.json](TRACKING_GROUND_TRUTH_METRICS.json), LiDAR-only identity
timelines, and synthetic tests.

## Verdict
- **Best current configuration for identity: Phase 3.6 (`tracking_robustness.yaml`).** It has the
  highest annotated IDF1 on B, A2 and C, and ties the original on A1. Phase 3.7
  (`tracking_refinement.yaml`) is better than the original on B, A2 and C, but worse than 3.6 on
  IDF1 in every recording. It also lowers coverage, because a first miss now marks the track Lost.
- **No configuration is good enough to promote to the default.**
  - Recording-level IDF1 (person and box together) is 0.30–0.60 on B and A2 for every tracker.
    Per-visit IDF1, which forgives the expected new ID after the person leaves view, is
    0.42–0.72.
  - The static box changes ID 6–55 times per recording.
- **Keep the default `tracking.yaml`.** Use 3.6 for identity-sensitive experiments and 3.7 only
  where its lower duplicate count matters. 3.6 has lower walking MOVING recall (B 0.43 against
  0.61 for the original); 3.7 sits between them at 0.50.
- **The remaining bottleneck is the LiDAR proposals, not tracker tuning.**
  - On the static box, the clustering gives no proposal of its own in 14–33% of visible frames:
    the box merges into a background cluster, or into the person.
  - Most remaining ID changes follow one of these proposal failures, or are a duplicate track
    briefly taking over.
  - With oracle segmentation of the annotated objects, the unchanged original tracker reaches
    IDF1 0.67 (B), 0.69 (A2), 0.54 (C) and 0.98 (A1), higher than any tuned profile in three of
    four recordings.

## Annotation workflow
Code:
- `scripts/annotate_ground_truth.py` (`cache`, `propose`, `sheet`);
- `src/go2_perception/tracking/ground_truth.py` (schema, matching, metrics);
- `scripts/evaluate_ground_truth.py` (scoring, failure diagnosis, oracle diagnostic).

There were no annotation tools before; `visualization/bev_tracking.py` only draws tracker
output.

1. **`cache`.** Transform every raw `cloud_base` frame to odom with the recorded odometry.
   Frames without bracketing odometry are left out, never extrapolated.
2. **`propose`.** Build per-frame annotations from a hand-written spec (intervals, empty-scene
   window, objects, review decisions). Annotations are made at the exact timestamps of the
   tracker's input frames, from raw points only. No tracker output and no detector proposal is
   used.
   - **Static obstacle.** A footprint polygon is measured once from the empty-scene points
     (connected component at the seed; 2nd–98th percentile extent; z from −0.24 to 0.10 m in the
     body frame). Visibility comes from the fraction of the empty-scene point count seen in each
     0.3 s window (a 0.3 s running median): **visible** ≥ 0.5, **partial** 0.15–0.5,
     **occluded** < 0.15. Motion label: stationary.
     - When the robot moves (C), the obstacle does not move, but odometry drift moves where it
       appears in odom. The region therefore follows the measured front-face displacement of the
       raw points near the footprint, in 3 s windows at 1 s steps; the offset is stored per frame
       as `registration_offset`.
   - **Person.** 10 cm voxel background subtraction against the empty-scene window, inside an
     annotation zone (body x 1.0–2.36 m, y −2.4 to 1.5 m). Position: the median of the largest
     foreground component (0.35 m linkage) in a 0.2 s window centred on the frame, smoothed by a
     0.3 s running median. Velocity: linear fit over 0.6 s. **moving** above 0.35 m/s,
     **stationary** below 0.12 m/s, unlabelled in between. Status: visible with 10 or more
     points, partial with 4–9, **out_of_view** with none.
     - The `visit` index increments after at least 1 s out of view.
   - **Confidence.** high or medium (medium near a visibility threshold, or for a sparse
     person). Low is reserved for manual review.
3. **`sheet`.** Render review contact sheets: raw and foreground BEV with the proposed
   annotation, plus the nearest camera frame (host-time aligned through the odometry clock
   offset; camera latency is unmeasured). The camera is used only to check identity and
   occlusion, never for positions, because there is no calibration.
4. **Review.** Decisions go into the spec's `review` list (status, confidence or position
   overrides, or rejection), with a `review_log`; then rerun `propose`.

**Schema `go2-tracking-gt-v1`** (validated on load and save). Each annotated frame has the
tracker-frame timestamp (go2_robot clock) and, per object:
- **id**, persistent within the recording;
- **type**: person, static_obstacle or other;
- **status**: visible, partial, occluded or out_of_view;
- **position**, when it can be located: `xy` (point) or `region` (polygon) in odom. No position
  is given when the object is occluded or out of view;
- optional **velocity** (point objects only);
- **motion**: moving, stationary or null;
- **confidence**: high, medium or low.

Visibility is defined for the LiDAR, which is what the tracker uses. The camera often shows the
box hidden behind the person's legs while the LiDAR still sees it.

**Who annotated, and how.** The annotations were proposed from LiDAR and reviewed by Claude on
contact sheets: B 82 panels, A2 57 and C 95. A1, which has only the unoccluded box, was
checked only on the footprint and visibility plots. No human has verified
them, and no review override was needed. The review log is stored in each spec and annotation
file.
- One person appears in every camera frame that has LiDAR foreground (B and A2).
- The person's position sits on the leg points in every reviewed panel.
- Box occlusion coincides with the person standing between the robot and the cases.
- In C the scene stays registered through every turn. The box drifts slowly in odom, and the
  registered region encloses its points in every panel.
- Between passes the person leaves the LiDAR's view: with a wider zone, foreground disappears
  past y ≈ −2.5 m. **out_of_view** is therefore a real sensor exit.

**Annotation quality.**
- **Person position.** The median deviation from the neighbour average is 0.015 m in B and
  0.026 m in A2 (p95 0.08 m).
- **Person velocity.** The annotated speed on frames labelled stationary has median 0.06–0.07 m/s
  (p95 0.11). That is the noise floor of the velocity labels; it includes real body sway.
- **Box footprint.** Body-frame front face at x = 2.38–2.39 m in all four recordings. The depth
  varies from 2.63 to 2.85 m: the sparse top of a second case behind the first.
- **Odometry drift in C.** The box's apparent odom position moves by up to (−0.11, −0.20) m by
  62 s, about 3 mm/s, which is odometry drift plus a change of visible face; registration follows
  it. In-gate box proposals then stay within 3–4 cm of their median offset in every robot
  phase.

## Annotated intervals
Intervals are relative to the first LiDAR frame. Frame counts are tracker frames, at about
15.4 Hz.

| Recording | Intervals (s) | Frames | Objects | What they cover |
|---|---|---:|---|---|
| B (robot still) | 6.5–18.5, 20.5–35.5, 38.5–54.0 | 654 | person, box | Three walking passes, each with a 3–6 s pause directly in front of the box; box occlusion and recovery; the most fragmented pass (3). |
| A2 (robot still) | 3–17, 18.5–29.5, 31–42.5 | 561 | person, box | Three passes (the planned 30 s stand did not happen); pauses in front of and to the left of the box. |
| C (robot driven by hand) | 0.5–6.5, 6.5–26.5, 32.5–40, 47–68, 73–79.5 | 938 | box | Still reference; ±30° turns; 0.5 m forward motion; turns and side steps; final still period. |
| A1 (robot still) | 0.5–60 | 916 | box | Unoccluded control. |

Annotation counts:

| Recording | Person: visible / partial / out of view | Person motion: moving / stationary / unlabelled | Box: visible / partial / occluded |
|---|---|---|---|
| B | 526 / 29 / 99 | 240 / 155 / 259 | 548 / 84 / 22 |
| A2 | 466 / 29 / 66 | 321 / 52 / 188 | 505 / 52 / 4 |
| C | – | – | 938 / 0 / 0 |
| A1 | – | – | 915 / 1 / 0 |

## Metrics
Tracker outputs are matched to annotated objects frame by frame. An object keeps its previous
track while that track stays inside the gate (as in CLEAR-MOT); otherwise Hungarian matching on
distance applies.
- **Gates:** 0.5 m from a person's point; 0.15 m from the box polygon (0 inside it).
- **Ignored tracks:** tracks outside every gate are unannotated scene, neither true nor false
  positives.
- **Primary track set:** Confirmed tracks.
- **Control track set:** observed tracks (updated by a detection in this frame, any status).
  This removes 3.7's change of reporting the first missed observation as Lost.

**Metrics that need annotated positions:**
- **Coverage:** matched / detectable frames, where detectable means visible or partial.
- **GT ID switches:** a change of the matched track ID.
- **Fragmentations:** tracked, then untracked while detectable, then tracked again.
- **IDF1** (Ristani et al.): over all frames, counting every track inside an object's gate as a
  hypothesis. **IDF1 per visit** treats each re-entry of the person as a new identity.
- **Duplicate frames:** more than one track inside an object's gate.
- **Track IDs:** distinct IDs matched to the object.
- **Position error:** person only; the box is a region, so its error is 0 by definition.
- **Velocity error:** person, against the annotated velocity.
- **Static speed:** box, where the truth is 0.

**Metrics that need only identity, visibility, motion labels and a fixed static region:**
- **Recovery after occlusion/partial view:** the same ID within 1 s of reappearing. Exits from
  view are excluded, because no variant attempts re-identification.
- **MOVING recall and precision** against moving labels.
- **STATIONARY recall** and **false MOVING rate** against stationary labels.
- UNCERTAIN counts as no decision.

## Results (original → Phase 3.6 → Phase 3.7)
All three run on identical cached inputs: the shared `tracker_comparison/inputs`, whose SHA-256
fingerprints match earlier runs. The run is serial and source-isolated, with working code at
clean commit `c643422`. The original is the archived `ecfaae3`; the current-default replay was
also run.

### Identity (Confirmed tracks)
| Recording | IDF1 | IDF1 per visit | Object | Coverage | GT ID switches | Fragmentations | Track IDs | Dominant-ID share | Duplicate frames |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| B | 0.33 → **0.40** → 0.39 | 0.48 → **0.58** → 0.55 | person | 0.93 → 0.92 → 0.90 | 24 → 22 → 26 | 16 → 14 → 26 | 16 → 14 → 14 | 0.28 → 0.34 → 0.34 | 42 → 43 → 14 |
| | | | box | 0.74 → 0.75 → 0.67 | 16 → 6 → 10 | 39 → 39 → 61 | 14 → 6 → 6 | 0.46 → 0.56 → 0.56 | 3 → 3 → 0 |
| A2 | 0.30 → **0.60** → 0.53 | 0.42 → **0.72** → 0.65 | person | 0.74 → 0.77 → 0.72 | 18 → 8 → 26 | 17 → 16 → 27 | 15 → 8 → 11 | 0.22 → 0.32 → 0.33 | 7 → 20 → 14 |
| | | | box | 0.84 → 0.84 → 0.76 | 12 → 15 → 12 | 25 → 22 → 45 | 10 → 8 → 7 | 0.44 → 0.95 → 0.88 | 6 → 21 → 1 |
| C | 0.29 → **0.60** → 0.41 | – | box | 0.78 → 0.79 → 0.69 | 15 → 16 → 18 | 62 → 62 → 92 | 12 → 8 → 8 | 0.33 → 0.68 → 0.50 | 15 → 8 → 0 |
| A1 | **0.90 → 0.90** → 0.86 | – | box | 0.94 → 0.94 → 0.85 | 30 → 22 → 55 | 27 → 28 → 75 | 10 → 8 → 17 | 0.95 → 0.95 → 0.93 | 58 → 51 → 16 |

Observed-only control (no coasting tracks, so the three status policies are comparable):

| Recording | IDF1 | Object | Coverage | GT ID switches | Track IDs |
|---|---:|---|---:|---:|---:|
| B | 0.32 → 0.39 → 0.39 | person | 0.89 → 0.89 → 0.90 | 34 → 36 → 26 | 20 → 18 → 14 |
| | | box | 0.65 → 0.66 → 0.67 | 14 → 6 → 10 | 13 → 6 → 6 |
| A2 | 0.29 → 0.56 → 0.53 | person | 0.69 → 0.73 → 0.72 | 17 → 8 → 26 | 15 → 8 → 11 |
| | | box | 0.75 → 0.77 → 0.76 | 10 → 23 → 12 | 9 → 8 → 7 |
| C | 0.26 → 0.55 → 0.41 | box | 0.69 → 0.69 → 0.69 | 21 → 20 → 18 | 14 → 9 → 8 |
| A1 | 0.87 → 0.87 → 0.86 | box | 0.85 → 0.85 → 0.85 | 42 → 42 → 55 | 13 → 12 → 17 |

What the annotations show:
- **3.7's duplicate reduction is real but small.** In the observed-only control it changes
  coverage little; its lower Confirmed coverage (B box 0.75 → 0.67, C 0.79 → 0.69) comes from
  the first-miss Lost policy.
- **3.7 loses continuity relative to 3.6.** IDF1 falls in A2 and C, and switches on the A1
  control rise from 22 to 55. This confirms the proxy finding with labels.
- **Many ID changes are short flickers.** The original ID returns in 12/40, 7/30, 4/15 and
  21/30 of the original tracker's changes (B, A2, C, A1). For 3.7 that is 18/36, 22/38, 11/18
  and 39/55. One frame of a missed or merged proposal lets a duplicate track hold the object, then
  the main ID returns.
- **Per pass (B, Confirmed IDF1):** 0.64/0.65/0.33 → 0.78/0.72/0.42 → 0.73/0.70/0.46. Pass 3,
  the long pause in front of the box, is the hardest for every variant.
- **Identity transfer between the person and the box.** In B the original tracker matched track
  #101 to the box for 215 frames and to the person for 39. In 3.6, #211 served the person for
  104 frames and the box for 78; in 3.7, #211 served them for 81 and 75. In A2 3.7, #60 was
  matched to the box for 370 frames and to the person for 15.

[Timelines (LiDAR only)](media/gt_timeline_test_b.png): [B](media/gt_timeline_test_b.png),
[A2](media/gt_timeline_test_a2.png), [C](media/gt_timeline_test_c.png),
[A1](media/gt_timeline_test_a1.png). Colours mark matched IDs per row, black means detectable
but unmatched, grey means occluded or out of view.

### Position, velocity, motion and recovery
| Recording / object | Position error median (m) | Velocity error median (m/s) | Static speed median (m/s) | MOVING recall | MOVING precision | STATIONARY recall | False MOVING on stationary | Recovery kept |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B person | 0.15 → 0.16 → 0.16 | 0.26 → 0.25 → 0.25 | – | 0.61 → 0.43 → 0.50 | 0.81 → 0.98 → 0.96 | 0.11 → 0.41 → 0.44 | 0.21 → 0.013 → 0.034 | 6/6 → 6/6 → 6/6 |
| B box | – | – | 0.087 → 0.080 → 0.081 | – | – | 0.13 → 0.37 → 0.46 | 0.077 → 0.004 → 0.005 | 5/11 → 9/11 → 8/10 |
| A2 person | 0.15 → 0.14 → 0.14 | 0.33 → 0.33 → 0.31 | – | 0.71 → 0.70 → 0.69 | 0.95 → 0.98 → 0.97 | 0.00 → 0.00 → 0.00 | 0.37 → 0.13 → 0.21 | 7/7 → 7/7 → 7/7 |
| A2 box | – | – | 0.082 → 0.077 → 0.078 | – | – | 0.06 → 0.41 → 0.49 | 0.041 → 0 → 0.007 | 1/5 → 4/5 → 2/4 |
| C box | – | – | 0.061 → 0.061 → 0.056 | – | – | 0.09 → 0.38 → 0.45 | 0.042 → 0.003 → 0.003 | no occlusion |
| A1 box | – | – | 0.072 → 0.072 → 0.072 | – | – | 0.06 → 0.52 → 0.55 | 0.010 → 0 → 0.001 | no occlusion |

- **Position error.** The person's error of about 0.15 m is the leg-cluster offset within the
  0.5 m gate; it does not separate the variants.
- **Velocity error.** 0.25–0.33 m/s against a label noise floor of about 0.06 m/s; the same for
  all three.
- **Motion labels.** 3.6 and 3.7 trade walking MOVING recall for far fewer false MOVING labels
  and much better STATIONARY recall. The person counts are small: A2 has only 52 stationary
  person frames.
- **Recovery after occlusion.** These are box events: the person in front of the cases. 3.6
  keeps the ID best (9/11 and 4/5).
- **Robot motion (C).** ID changes are not concentrated in motion. Of the annotated C frames,
  752 are still, 82 rotating and 104 translating.
  - Original: 14 ID changes while still, 1 while rotating, 0 while translating.
  - 3.6: 16, 0 and 0. 3.7: 18, 0 and 0.

## Where fragmentation comes from (Task 4)
For each annotated frame the evaluator classifies the cached LiDAR proposals near each object:
- **single:** one centroid in the gate;
- **split:** several centroids in the gate;
- **merged:** the object's centre lies inside a larger cluster whose centroid is outside the gate,
  near the other annotated object or near background;
- **no cluster:** nothing covers the object.

These are properties of the detector's input to every tracker. No tracker affects them.

| Object | Frames | Single | Split | Merged with background | Merged with the other object | No cluster |
|---|---:|---:|---:|---:|---:|---:|
| A1 box | 916 | 759 (83%) | 25 | **131 (14%)** | – | 1 |
| C box | 938 | 627 (67%) | 0 | **308 (33%)** | – | 3 |
| B box | 632 | 425 (67%) | 0 | 67 (11%) | **74 (12%)** | 66 (10%, mostly partial occlusion) |
| A2 box | 557 | 427 (77%) | 1 | **113 (20%)** | 10 | 6 |
| B person | 555 | 499 (90%) | 19 | 20 | 4 | 13 |
| A2 person | 495 | 354 (72%) | 14 | **114 (23%)** | 1 | 12 |

When the box loses its own proposal in the unoccluded A1, the median cluster that swallows it is
3.5 × 2.6 m: the cases join the counter and wall about 0.15–0.3 m behind them, because the
DBSCAN `eps` is 0.3 m. In C, merged clusters are 1.5 × 1.3 m.

**Every ID change, by proposal state over the changeover and by what the tracker did.** Counts
are for the original tracker; 3.6 and 3.7 are in the JSON.
- **B (40 changes):**
  - 21 had clean proposals: 14 a duplicate track took over, 7 a new track was born.
  - 13 involved a merge with the person or background.
  - 3 were exits.
  - 2 were occlusions; 1 had no cluster.
- **A2 (30):** 11 merged with background, 2 merged with the person, 12 clean, 4 exits, 1
  occlusion.
- **C (15):** 6 merged with background, 1 no cluster, 8 clean (6 duplicate took over).
- **A1 (30):** 15 merged with background, 1 split, 14 clean (12 duplicate took over).

A "clean" changeover is usually a duplicate track that had existed nearby for some time. Those
duplicates are born during earlier merges and splits. When the main track's proposal merges for
a frame or two, its next update misses and the duplicate takes the measurement.

Representative cases (original tracker unless noted):
1. **Box merged with background (A1 15.2 and 29.8 s; C 16.3, 35.9, 47.8 and 74.1 s).** The
   proposal disappears into the counter and wall cluster for 1–11 frames.
   - The main track (#13 in A1) coasts and goes Lost; a new or duplicate track holds the box;
     #13 returns.
   - In 3.6, C box #16 hands over at 16.3, 38.8, 51.3, 62.6 and 67.4 s, each time during merged
     frames. It comes back after 0.1–0.3 s, except after 16.3 s, when #44 holds the box until
     33.1 s.
2. **Person merged with background, then fragment tracks (B 8.45–8.58 s).** For three frames the
   person's only nearby proposal is a 2.0 × 2.2 m cluster 0.9 m away.
   - Main track #101 goes Lost, and fragment track #104 is born from the merged centroid.
   - Other short tracks (#107, #109, #111) are born from leg and background fragments 0.5–0.9 m
     away. They coast into the person's gate, and at 9.42 s #107 takes the person's proposal for
     one frame (101 → 107 → 101).
3. **Person standing in front of the box: merge and identity transfer (B 25.3–31.3 s).**
   - The box proposal merges with the person (`merged_other_object`); the box track goes Lost,
     then the box is occluded from 29.8 s.
   - Original: box #101 → #270 → #101 → #312 → #329. Track #101 also held the person earlier.
   - 3.6 keeps the box as #21 until the occlusion at 29.8 s, then #171, apart from one 0.3 s
     flicker at 25.3 s.
4. **Person beside the box in A2 (8.6–10.5 s, 38.9 s).**
   - At 38.9 s, 47–48 consecutive merged frames end the person's ID in all three variants.
   - At 8.6–10.5 s (19 merged frames) the original deletes #105 (→ #113).
   - A2's 23% person merge rate is the largest person failure.
5. **Exits (B 21.8 and 39.1 s; A2 18.9 and 32.2 s).** The person is out of view for 4–5 s, past
   every lifetime limit, so a new ID is expected.

**Answers to the five hypotheses:**
1. **LiDAR cluster splitting/merging: primary.** Merges dominate (11–33% of visible box frames).
   Splits are rare (at most 3.4%).
2. **Data association: secondary.** It shows up as duplicate tracks taking over, but those
   duplicates exist because of earlier proposal failures. The 3.7 duplicate suppression lowers
   duplicate frames, but not the switches.
3. **Track lifecycle: secondary.** The 3.6 time-based expiry rescues some merges (higher IDF1).
   3.7's first-miss Lost policy turns every merged frame into a coverage gap (A1 fragmentations
   27 → 75).
4. **Odometry and transforms: not a significant cause.**
   - In C, 14–18 of the 15–18 ID changes per variant happen while the robot is still.
   - The apparent drift of the box in odom is about 3 mm/s, far below any gate. After
     registration, in-gate proposals stay within 3–4 cm in every phase.
   - Merges happen in every phase (still 34%, rotating 20%, translating 38%).
5. **Missing RGB semantics: contributes in mixed scenes.** Person↔box identity transfer
   occurred in B and A2 for every variant (above). A class label would forbid matching a person
   proposal to a box track, but without camera calibration this cannot be tested here.

### Oracle segmentation: would better proposals help more than tuning?
This is a diagnostic, not a tracker result. `--oracle-segmentation` re-extracts the inputs from
the raw recordings, at the same 0.3 s window and with the same pose gate. In annotated frames,
each annotated object's own LiDAR points are removed before clustering and returned as one
proposal: points inside its gate (box 0.15 m, person 0.45 m), above the ground band, box first.
- Nothing is invented: frames with fewer than 3 points get no oracle proposal (16 of 4,093).
- Every other cluster is unchanged. Unannotated frames are byte-identical to the shared inputs in
  all four recordings.
- The same three trackers were replayed on it. Because the inputs use ground-truth positions,
  the scores are an upper bound for "perfect proposals for these objects".

| Recording | IDF1 real proposals (orig → 3.6 → 3.7) | IDF1 oracle proposals (orig → 3.6 → 3.7) | Box ID switches, real → oracle (original) | Person ID switches, real → oracle (original) |
|---|---|---|---:|---:|
| B | 0.33 → 0.40 → 0.39 | **0.67** → 0.46 → 0.47 | 16 → 1 | 24 → 5 |
| A2 | 0.30 → 0.60 → 0.53 | **0.69** → 0.50 → 0.48 | 12 → 4 | 18 → 8 |
| C | 0.29 → 0.60 → 0.41 | 0.54 → 0.42 → **0.70** | 15 → 3 | – |
| A1 | 0.90 → 0.90 → 0.86 | **0.98** → 0.82 → 0.58 | 30 → 6 | – |

With oracle proposals, for the original tracker:
- **Coverage** rises to 0.98–1.00 for every object.
- **Per-visit person IDF1** is 0.89 (B) and 0.97 (A2).
- **Box recovery after occlusion** is 11/11 (B) and 7/8 (A2).
- **Motion labels:** walking MOVING recall in B is 0.73 (real 0.61), and box STATIONARY recall
  is 0.28–0.66 (real 0.06–0.13).

**Conclusions:**
1. **Better proposals help more than any tracker profile has.** The untuned original tracker on
   oracle proposals beats the best tuned profile on real proposals in B, A2 and A1: 0.67 against
   0.40, 0.69 against 0.60, 0.98 against 0.90. In C its box switches fall from 15 to 3, but IDF1
   (0.54) stays below 3.6 on real proposals (0.60), because one ID change splits the box into two
   long segments. 3.7 on oracle proposals reaches 0.70.
2. **The 3.6 and 3.7 mechanisms compensate for bad proposals but cost identity on good ones.**
   On oracle inputs, 3.6 adds short duplicate tracks that flicker on the box: A1 has 44 switches,
   43 of them returns to the old ID. Retune the tracker after the proposals improve, not before.
3. **What remains after the oracle is duplicate-track flicker.**
   - On oracle proposals, 3.7 still has 11 C switches (7 returns) and 3.6 has 44 on A1.
   - Person legs can fall outside 0.45 m: B and A2 each have 33–35 split frames, and C has 15.
   - These are a second, smaller target: duplicate suppression, and gating by object extent
     rather than by centroid.

## Held-out evaluation protocol (Task 5)
A1, A2, B and C have now been used to select parameters in Phases 3.6 and 3.7, and to design
this evaluation. They are development data. Proposed held-out set, to be recorded once and
scored once:
- **H1: one person, robot standing still (90 s).**
  - Walk in, stop for 3 s, walk diagonally toward the robot, stop and turn on the spot.
  - Accelerate slowly, cross in front of the box at 0.3–0.5 m, and walk out.
  - Repeat with a 1 s stop, to test brief stops.
- **H2: two people, robot standing still (90 s).**
  - They cross in front of the robot at 0.5–1.0 m separation and then walk side by side.
  - Then one stands still while the other passes behind them, a full occlusion of 1–2 s.
  - This is test D, never recorded.
- **H3: robot turning and walking, with stationary objects (90 s).**
  - Yaw ±45° in place at two speeds, then walk 1 m forward and back.
  - Three stationary objects: one free-standing at 1.5 m, one within 0.3 m of a wall (to provoke
    merging), and one partially behind another.
  - Tape the start and end footprint, so odometry drift is measured independently.

**Procedure:**
1. Before recording, freeze and record the code commit and the three profiles to be compared.
2. Record with `scripts/go2_record.py`. Write the plan in the manifest notes, and mark the
   recordings `held_out` in their annotation specs.
3. Tape-measure object positions relative to the robot's start footprint. That gives an absolute
   position check the A–C recordings lack.
4. Annotate with the same tool and thresholds. Do not change thresholds after looking at tracker
   output.
5. Run `compare_tracking_real.py` and `evaluate_ground_truth.py` once and report the result as
   is. Any parameter change after that needs a new held-out recording.
6. Have a person review at least 10% of the contact sheets and every out-of-view or occlusion
   transition.

## Reproduce (Mac with the private recordings)
```bash
# 1. Annotations. Specs live in data/annotations/specs/ (local); see the field descriptions above.
for r in test_a1 test_a2 test_b test_c; do
  PYTHONPATH=src .venv-go2/bin/python scripts/annotate_ground_truth.py cache --recording data/recordings/$r
  PYTHONPATH=src .venv-go2/bin/python scripts/annotate_ground_truth.py propose --spec data/annotations/specs/$r.json
  PYTHONPATH=src .venv-go2/bin/python scripts/annotate_ground_truth.py sheet --spec data/annotations/specs/$r.json --every 0.5
done
# 2. Tracker outputs on the shared cached inputs (fresh output directory)
PYTHONPATH=src .venv-go2/bin/python scripts/compare_tracking_real.py --reuse-cache --serial \
  --cache-root data/outputs/mac/tracker_comparison/inputs \
  --variants baseline current_default improved refined \
  --output data/outputs/mac/ground_truth_eval/tracks
# 3. Scoring, failure diagnosis, aggregate JSON and timelines
G=data/outputs/mac/ground_truth_eval
PYTHONPATH=src .venv-go2/bin/python scripts/evaluate_ground_truth.py \
  --plots $G/plots --aggregate $G/aggregate.json
# 4. Oracle-segmentation diagnostic (about 6 min extraction per recording)
PYTHONPATH=src .venv-go2/bin/python scripts/evaluate_ground_truth.py \
  --oracle-segmentation $G/oracle_segmentation_inputs
PYTHONPATH=src .venv-go2/bin/python scripts/compare_tracking_real.py --reuse-cache \
  --cache-root $G/oracle_segmentation_inputs \
  --variants baseline improved refined --output $G/oracle_segmentation_tracks
PYTHONPATH=src .venv-go2/bin/python scripts/evaluate_ground_truth.py \
  --tracks $G/oracle_segmentation_tracks --inputs $G/oracle_segmentation_inputs \
  --output $G/evaluation_oracle_segmentation.json --aggregate $G/aggregate_oracle_segmentation.json
# 5. Committed metrics file (aggregates only; refuses local paths)
PYTHONPATH=src .venv-go2/bin/python scripts/evaluate_ground_truth.py --combine \
  $G/aggregate.json $G/aggregate_oracle_segmentation.json $G/tracks $G/oracle_segmentation_tracks \
  docs/TRACKING_GROUND_TRUTH_METRICS.json
PYTHONPATH=src:third_party/unitree_sdk2_python .venv-go2/bin/python -m pytest -q
```

## Remaining manual annotation needs
- **Human review of the annotations.** A person should check at least the B and A2 sheets around
  the pauses in front of the box (B 10–13, 24–31 and 43–50 s; A2 8–13 and 35–38 s). Visibility
  and the person's position there are the least certain.
- **Partial versus occluded thresholds.** These come from LiDAR point fractions, not from
  ray-casting. A human check of 20–30 transitions would calibrate them.
- **Absolute positions.** Annotated positions come from the same LiDAR, in odom. They are not
  checked against a tape measure or motion capture.
- **Unannotated objects.** The A1 person pass at about 24 s, the operator in C, and every
  background structure have no labels. That is why tracks away from annotated objects are
  ignored rather than counted as false positives.
- **Unannotated time.** B 0–6.5 and 54–65 s, A2 0–3 and 42.5–50 s, and C 26.5–32.5, 40–47 and
  68–73 s (transitions between manoeuvres) are not annotated.
