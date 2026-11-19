# Box grounding review — November 19, 2026

The recorded failure starts with the model's visual interpretation. In the edge view the box is partly visible at the far right of the **actual model input**, while LightNav places its mark near the middle of the hallway. The archived image-bound reviewer rejected it. The subsequent forward movements were search steps to acquire new views. This episode never produced a reference-supported approach.

## Trace and provenance

The most recent closed run with the operator's wrong-box assessment is `runs/five_move/trial-THKUcr/session`, trial `T20261119-120816-01-3a48`. It ran `go2-five-move-episode-v1` with `go2-effective-v1` in live mode on navigation `918f48c` and controller `6d555d3`. The capability binds the locally modified guard provider at SHA-256 `3c0e7c56c5c21ccbf91f6ae89300d8d6e86167d6318c81f4ddb12f44001919bf`. The 0.140 s exposure uncertainty belongs to the 11:31 boot measurement. [Workspace inventory](CODE_WALKTHROUGH.md#workspace-and-launch-modes)

P002 recorded the normalized target `box` and the instruction `Go to the box and stop.` in the trial, worker job and actual prompt. Original whitespace or an optional typed wrapper was not separately saved. The selected camera frames used one instruction revision and clock epoch. P002 used grid indices `[0, 1, 10, 11, 12, 13, 14, 15]`. Its exact 448 × 256 input is `session/requests/000002/native/input_0015.png` relative to the session root, SHA-256 `0226c295a4b6f919dc42f417cef2f1dccaead9f31eca98e4eddb771eba999a10`. The estimated observation age at display was 2.419 s. The executor accepted this observation for identification.

The P002 native output was `<apos_888><opos_698><act_l0_0><act_l1_30><act_l2_18><|im_end|>`. OPOS 698 decodes to the center of column 25, row 14 (zero-based) on the pinned 48 × 27 grid. It lands at (0.53125, 0.53704) on that image. The full ten-point proposal ends approximately 1.507 m ahead in decision-frame coordinates. The original raw RGB was 1280 × 720. Calibrated fisheye rectification preserved the measured optical axes, then the official processor resized the rectified view. The overlay and decoder agree on the archived native PNG. The arrow shows a proposed trajectory, with no physical speed or time assigned by the model.

The image-bound reviews rejected P002 and P004. P003 matched the earlier rejected appearance; P005 had inconclusive identity evidence after the next rejection. Both were routed to SEARCH. Each dispatch selected a 0.30 m forward viewpoint step from `search_policy`. Both recorded zero available turn clearance. The other decision with a stale-observation veto was P001. The logs support an incorrect model mark and an ambiguous display. They give no evidence of discarded target words, a stale P002 instruction, an overlay offset, a stale positive reference, or an identity-gate bypass. [All six decisions and their recorded results](CODE_WALKTHROUGH.md#read-the-box-episode)

## Fixed prompt comparison

Three P002 decisions were chosen before inference. The right-edge and hallway images have archived rejected candidates. The clear view has an archived identified box region. The session-level operator assessment reports the box remained out of view in the hallway trial. Descriptions of image content, the edge-box region and judgments about the new counterfactual points are **agent review**. Alternative prompts received no human reviews. These are related views in the same home.

The template was `Go to the {description} and stop.` The landmark variant was `brown cardboard box on the floor near the stainless steel refrigerator`. Each row retained its own original causal history and timestamps across all four columns. Entries are native OPOS codes. Zero means that the model reports absence.

| Recorded view | box | brown cardboard box | + on the floor | + near the stainless steel refrigerator |
|---|---:|---:|---:|---:|
| Box at right edge · `trial-THKUcr` | 698, rejected hallway object | 0, missed box | 0, missed box | 0, missed box |
| Clear box · `trial-bYheuc` | 800, identified box | 800, same box | 800, same box | 800, same box |
| Hallway distractor · `trial-cdyo3G` | 745, rejected distractor | 0, consistent with this view | 0, consistent with this view | 649, another hallway mark |

Color and material wording suppressed the false positive in the distractor view. The same wording lost the edge target. The landmark brought the distractor mark back. On the clear box all prompts marked (0.65625, 0.61111), while the trajectory codes changed with wording. A correct point alone cannot validate a path, stopping, size, material or navigation success.

The experiment used the cached `LightOriginsHQ/LightNav-0` revision `826dc5fbfa37afa8293d2e336d329b6ffc0bfb64`, upstream `3015508b70fb6e30bbd66701e94c9b916dcb8df8`, the archived local pooling patch, MPS BF16, SDPA, no quantization, seed 0, temperature 0 and a 64-token limit. The model interpreter was Python 3.11.13 with PyTorch 2.10.0 and Transformers 5.8.0. A single worker reset its decision state between requests. All 13 decoded, including one repeated edge baseline. Every selected native image hash and time segment matched its archived counterpart. The full native prompt text differed only in the requested instruction. The three baseline token strings matched the archives, and the repeated baseline matched its first run. This checks repeatability on this Mac for these controls. Bitwise determinism across devices or runs is unmeasured.

## Inspect or reproduce

Run the [immutable box replay](../START_HERE.md) or use the environment exports from the [walkthrough](CODE_WALKTHROUGH.md#workspace-and-launch-modes). To repeat only this 13-request experiment, run:

```bash
"$GO2_WORKSPACE/envs/core/bin/python" scripts/compare_grounding_prompts.py
```

Each run prints a new private directory. The measured outputs are in `$GO2_WORKSPACE/runs/target_grounding/prompts-h_7tnoou/`. The saved presentation with explicit absence labels is `panels-1mlvwwxl/comparison.html` in that directory. To regenerate those panels from the saved outputs, run `"$GO2_WORKSPACE/envs/core/bin/python" scripts/compare_grounding_prompts.py --render-run "$GO2_WORKSPACE/runs/target_grounding/prompts-h_7tnoou"`. It displays the four marks side by side for each view. Individual `native/prompt.txt` and `native/result.json` files contain full prompts, tokens, points, STOP and all waypoints. `design.json` records the fixed cases; `results.jsonl` includes the repeated control. The private annotated trace is `$GO2_WORKSPACE/runs/target_grounding/trace-iyz44g6h/trace.html`. The replay verified 123 source files before and after.

## Focused verification

Headless Chrome verified image-bound support, rejection controls, the SEARCH overlay and stale-image clearing. The core interpreter passed 217 tests in 35.61 s, zero failed or skipped in this selection. Reproduce from the worktree with the environment from the walkthrough:

```bash
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1
"$GO2_WORKSPACE/envs/core/bin/python" -m pytest -q -rs -p no:cacheprovider \
  -m 'not hardware and not model_integration' \
  tests/test_target_grounding_review.py tests/test_five_move_navigation.py tests/test_vln_interactive.py \
  tests/test_motion_diagnostics.py tests/test_continuous_navigation.py tests/test_effective_motion.py \
  tests/test_supervised_navigation.py
```

## Changes and next decision

The viewer now labels the raw mark, the named reference status on the bound image, and the selected SEARCH or APPROACH action with its source. Receipt freshness has its own label. Late image loads cannot update the current panel. The closed-trial review joins identification, rejection, dispatch and result by ID. New five-move trials preserve the entered text as well as its normalized form; old trials retain their original evidence. A missing region on Identify/Reject gives actionable feedback. Reference names must match every descriptor, allowing case and spacing differences. Color, category, physical size and landmark relation remain unverified by grayscale features.

The next useful experiment is to evaluate a human-identified reference across multiple saved edge, central and distractor views, retaining the existing thresholds and reporting “insufficient features” separately. That would distinguish a lost visual identity from a poor model proposal before changing the movement policy.
