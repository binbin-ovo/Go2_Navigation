# Box recognition and bounded navigation

The clearest failure is visible on the exact image that LightNav consumed. The box occupied a strip along the right edge. The model placed OPOS 698 at image fraction (0.53125, 0.53704), on the hallway. The operator-role review rejected the candidate. Two later forward dispatches were SEARCH steps. The original episode ended at its deadline with zero supported approaches. The stored target, expanded instruction, worker prompt, frame selection and decoded pixel agree. More descriptive wording in the earlier experiment sometimes suppressed the correct edge target too. [Original review and frame hash](TARGET_GROUNDING_REVIEW.md)

## What the five-move application does

1. The camera and LiDAR can run while the website awaits a target. Entering `box` produces `Go to the box and stop.` The pinned worker receives the instruction and causal images. It returns a native target point or absence, ten local waypoints and a STOP indication.
2. On the bound model image, the application independently locates the operator-named reference and checks whether the native mark lies inside its transformed polygon. It records native visibility, mutual match count, inlier fraction, feature spread, residual and the reason matching stopped. A correct-looking raw point still requires identification if no positive reference has been enrolled. Five recent rejected regions can veto recurring distractors.
3. In a five-move episode, inconclusive identity with an enrolled reference selects SEARCH. A strongly localized reference away from the native mark can guide a bounded search turn toward a new view. That turn still requires clearance. If identity remains inconclusive after a different viewpoint, the operator can re-identify. Contradictory positive and rejected reference evidence cannot support APPROACH.
4. SEARCH uses the actual poses of fresh accepted images and prior dispatches. It avoids revisiting the same position and heading, including after ±π wrap, and remembers a stalled direction at the pose where it failed. A forward search step needs a new viewpoint, sufficient observed corridor and room inside the inspected area. When the choices fail, the episode holds or terminates with their reasons.
5. APPROACH requires agreement on the current target and image, compatible native path, and fresh guards. The alignment band is ±15° with 3° of existing hysteresis. Forward and turn primitives use their separately supported 1.0 m/s and 2.0 rad/s profiles. Each action consumes a slot and requires a settled fresh observation. Five attempted movements end the action budget. Native STOP and human-confirmed arrival have separate outcomes. [Decision code](../navigation.py) · [Reference verification](../src/go2_perception/vln/episode_grounding.py) · [Admission and dispatch](../src/go2_perception/vln/five_move.py)

The matcher retains the published gates of 12 mutual ORB inliers, 0.70 ratio in both directions, ≥0.80 inlier fraction, ≥0.12 spread across the reference ROI and ≤2 px median residual in the scaled feature image. The affine RANSAC tolerance is 2.5 px. A measured transform can persist when the native mark is missing or elsewhere. The evidence authorizing a real-image approach must still come from the named operator's reference. A coarse feature match cannot establish material, physical size or an arbitrary requested relation.

## Fixed comparison on recorded images

The executable baseline was `hardware/readable-navigation-integration@2fa98f20b415003bdb6dae3518ea422a757f45ec`. Claude published the documentation-only follow-up `d62d936c67fed4ce10d0f364188bbaa596fee970` during verification. This feature includes its [hardware validation report](FIVE_MOVE_NAVIGATION_VALIDATION.md). The evaluator loads the two pure baseline modules from that Git object and runs both policies on the same model output, references, recorded decision pose, clearance, prior attempts and area grant. For the cross-session clear and edge rows, both replay policies received the same earlier enrolled photo. The original live P002 decisions had no such positive reference. The evaluator admits enrollment only after checking the review against the conservatively bounded current capture time. It computes one decision per case. A counterfactual SEARCH selection does not generate the next recorded frame. The archives contain the actual historical selections and dispatches separately. [Evaluation script and exact session/prediction IDs](../scripts/evaluate_object_navigation.py)

| Case | Native OPOS | Image evidence and verifier | Baseline → new selection | DINO regions containing the native mark |
|---|---:|---|---|---|
| Hallway candidate, no reference | 745 | Archived **reject**. First candidate needs an image-bound review. | REVIEW → REVIEW | None |
| Absent hallway after rejection | 745 | Earlier rejected appearance recurred. Operator reported the box remained out of view in the session. | SEARCH forward → SEARCH forward | None |
| Same absent view, descriptive instruction | 0 | Model reported absence. The human rejection belongs to the original plain-box mark. | SEARCH forward → SEARCH forward | No native mark, zero candidate boxes |
| Centered box, fresh view after enrollment | 793 | Earlier enrollment, then 57 inliers on this fresh image. The archive contains a supported approach. | APPROACH forward 0.60 m → same | Yes |
| Closer centered box | 891 | Archived **identify**. Only 12 of 23 mutual matches were inliers. Turn clearance and the area bound prevented a new search movement. | REVIEW → HOLD | Yes, two category boxes total |
| Clear box, positive reference from an earlier session | 800 | Archived **identify**. Only two mutual matches with that earlier reference. | REVIEW → SEARCH forward | Yes |
| Clear box, fresh frame with local enrollment | 800 | Instance verified with 33 inliers. Native path turned too far to support forward travel. | HOLD → HOLD | Yes |
| Box partly visible at the right edge | 698 | Archived hallway **reject**. No mutual matches from the earlier clear-box reference. | REVIEW → SEARCH forward | No, one box at the right edge |
| Controlled partial cover on centered history | 793 | Mark fell on the cover. Fifteen inliers occupied only 0.096 of the reference region. | REVIEW → SEARCH forward | No, one box on the exposed portion |
| Controlled full cover on centered history | 745 | Model placed a mark on the opaque cover. Zero mutual matches. | REVIEW → SEARCH forward | No, zero boxes |

A separate synthetic test displayed two cartons with matching color, silhouette and label layout but different printed textures. The earlier reference localized on the intended carton and withheld support for a native mark on its neighbor. Appearance alone cannot reliably separate visually indistinguishable copies.

The two covers were applied before official model preprocessing to **every** selected frame, including anchors. Derived images, original hashes, timestamps, prompts, decoded tokens and waypoints are stored per request. Three shape conditions use the same center decision. The four source sessions are correlated private development data. Four distinct untouched marked images had current image-bound reviews, two positive and two negative. Claude made those clicks on the operator’s behalf. Independent frame-by-frame human labels are unavailable. Other evidence comes from earlier enrollment, the operator's session assessment, controlled cover geometry, and the current matcher. Changed prompts and covers have no fresh human labels. [Original review provenance](TARGET_GROUNDING_REVIEW.md)

Comparing both policies with identical evidence, four REVIEW decisions changed to bounded SEARCH and one changed to HOLD. Both policies selected the same single forward APPROACH. **No additional false mark was accepted.** The four directly reviewed marks show how category agreement compares with the reviewed labels. Both reviewed positives lay inside a category box and both reviewed negatives lay outside every category box. A point-only gate would also accept both reviewed negatives. Neither policy selected APPROACH on any of those four reviewed marks. It supported a forward approach on a separate, subsequent frame with the earlier enrollment.

## Independent category check

Grounding DINO tiny used the previously tested revision `a2bb814dd30d776dcf7e30523b00659f4f141c71`, CPU FP32, four threads, text `box.`, box threshold 0.35 and text threshold 0.25. The previous scratch cache retained snapshot links but had lost its weight blobs. The same public revision was restored to isolated scratch storage and verified against its Git object IDs and weight SHA-256. The detector consumed the exact current 448 × 256 LightNav PNG through its own image processor. At the edge it found a region around x=0.92–1.00 at score 0.65. It missed the native hallway mark, correctly excluding it from the category region. On the partial cover it found the exposed right-hand portion but excluded the covered mark. The two candidates in the closer view show why a category match alone cannot identify the requested instance.

**Runtime.** On the first fixed matrix, LightNav took 1.75–2.98 s per request (median 1.80 s). DINO took 1.82–1.96 s per image including preprocessing and postprocessing (median 1.85 s). Adding the separately measured times gives 3.59–4.94 s before other application overhead. A repeat of all ten decisions reproduced all 80 selected image slots and their hashes, token strings, points and waypoints. Median LightNav and DINO times in that repeat were 1.80 s and 1.99 s. The sampled peak MPS driver allocation was 8.94 GiB. The existing freshness limit is 3 s. DINO remains a review tool. Live asynchronous category guidance would need its own measured timing, image binding and stale-result handling. The continuous experiment previously measured a 0.175 s turn horizon and 0.500 s conditional forward horizon. [Detector comparison](../scripts/compare_box_regions.py) · [Continuous measurements](CONTINUOUS_NAVIGATION.md)

## Recovery and movement

**Controlled recorded-image reveal, real model.** A separate two-decision probe used the six most recent actual source frames from centered P003. The first three were fully covered. The next three revealed the original recorded pixels. The second decision retained the same covered prefix, with its original observation and availability times. LightNav first marked the occluder at OPOS 696. After the reveal it placed OPOS 793 on the independently enrolled box. The final image matched the archived current-image hash, 57 inliers supported identity, and the original final geometry admitted a 0.60 m forward APPROACH. There was no archived guard observation at the first cutoff, so it has no movement selection. The two neural calls took 2.37 and 1.83 s. This tests visual recovery on a controlled history. The recording did not move in response to either proposal. [Reveal replay](../scripts/replay_object_reveal.py)

**Closed-loop pinhole simulation.** The existing synthetic chair scene was rendered from the ideal proxy after each five-move selection. A settled fresh frame followed each action. A fixed pixel fixture selected SEARCH left 30°, exposed the chair, selected APPROACH forward 0.57 m, and returned its fixture STOP. The actual LightNav model started at the same pose, received six decision images and five bounded SEARCH turns. The renderer drew the chair on decision two. LightNav reported absence on every decision, made zero approaches, returned no STOP and reached the five-move limit. Neither run measured arrival. Synthetic image-region evidence stayed outside the model input. This is a static, step-and-wait simulation. [Simulation](../scripts/simulate_object_search.py)

## Reproduce locally

From the dedicated worktree on this Mac:

```bash
export GO2_WORKSPACE=/Users/yushan/Robotics/Go2
cd "$GO2_WORKSPACE/code/worktrees/codex-motion-diagnostics"
export GO2_HARDWARE=0 GO2_SUPERVISED_LIVE=0 MUJOCO_RENDER_TEST=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/hardware"
py="$GO2_WORKSPACE/envs/core/bin/python"
"$py" scripts/evaluate_object_navigation.py
```

The command prints a fresh private HTML report. For the category comparison, use the same command with `--dino-snapshot /private/tmp/go2-object.NHTKO8K7/public-detector/a2bb814dd30d776dcf7e30523b00659f4f141c71` or a local copy of the identical pinned snapshot. Use `--reuse-inference RUN` on a completed run with `inference_hashes.json` to regenerate the analysis in a new directory. Only the instruction and causal images and their timing enter the local LightNav subprocess.

```bash
"$py" scripts/replay_object_reveal.py
"$py" scripts/simulate_object_search.py --backend fixture
"$py" scripts/simulate_object_search.py --backend lightnav
```

The camera/model worker uses the cached LightNav-0 revision `826dc5fbfa37afa8293d2e336d329b6ffc0bfb64` and upstream `3015508b70fb6e30bbd66701e94c9b916dcb8df8`, unchanged local pooling patch, MPS BF16, SDPA, unquantized, PyTorch 2.10.0 and Transformers 5.8.0. All seven original `box` controls reproduced their archived native tokens, points and ten waypoints. The fixed matrix preserves the published anchors plus six most recent images and their actual timestamps. The reveal has its own declared six-frame history. The original 1280 × 720 measured fisheye rectification precedes native resizing on the recordings. The simulator has its own pinhole camera.

Private artifacts under `$GO2_WORKSPACE/runs/target_grounding/`:

| Evidence | Local path |
|---|---|
| Fixed matrix, preserved native requests | `object-navigation-nex99ynf/` |
| First complete category/instance comparison | `object-navigation-nspgc_zg/index.html` and `results.json` |
| Repeated fixed matrix and HTML with region overlays | `object-navigation-5ggvlnu9/index.html`, `results.json` and `inference_hashes.json` |
| Causal reveal | `object-reveal-2luoufdn/reveal.json` and `cutoff-{3,6}/native/` |
| Closed-loop pixel fixture and LightNav | `object-sim-fixture-q6cg_hv1/index.html`, `object-sim-lightnav-8611jpzp/index.html` |

## Verification

The core regression suite passed **566** tests in 80.58 s with 13 skips. Twelve skips require the isolated model runtime. All twelve passed together there in 4.32 s. The sole physical camera/LiDAR case stayed disabled. The focused policy, grounding and offline demo selection passed **58** tests. The fixed matrix was executed twice, totaling 20 LightNav and 20 Grounding DINO calls. The reveal contributed two LightNav calls and the model-driven simulation six. The pixel fixture used three decisions. A saved-run verification checked 122 input/output manifest entries and reproduced all ten before/after selections in a new private report.

One pretrained reset test initially failed on its exact scalar-white assertion. The pinned float32 resize of its 90 × 160 white input produces a minimum of 0.9999998808 and maximum of 1.0000002384 on this Mac. The test now compares every pixel of the new history to the independently processed new white image, retaining exact equality and the existing empty-cache, cleared-buffer and instruction checks.

```bash
LIGHTNAV_ASSETS= "$py" -m pytest -q -rs -p no:cacheprovider \
  tests/test_object_navigation_offline.py tests/test_five_move_navigation.py tests/test_target_grounding_review.py
LIGHTNAV_ASSETS= "$py" -m pytest -q -rs -p no:cacheprovider
LIGHTNAV_ASSETS="$GO2_WORKSPACE/weights/lightnav" "$GO2_WORKSPACE/envs/model/bin/python" \
  -m pytest -q -rs -p no:cacheprovider tests/test_bounded_native_integration.py tests/test_lightnav_runtime_integration.py
```

## Next evidence for Claude

On the next independently authorized supervised session, retain the exact model input and current image-bound identity review for the centered, edge, lookalike and lost/reacquired views. Mark the visible target region separately from the rejected native point. Record the pre-enrollment identity and source image, current pose/epoch, per-frame capture and receipt bounds, native completion time, and ORB match/inlier/spread diagnostics. For each SEARCH and APPROACH retain the source-pose heading, chosen direction, measured settled heading, travel measured independently of fused odometry, post-settle frame ID, clearance rejection and stop outcome. These measurements distinguish a missed target, a lost instance reference, an unavailable turn and real progress toward the intended object. The feature branch remains an offline evaluation until hardware review.
