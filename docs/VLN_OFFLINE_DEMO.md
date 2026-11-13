# Phase 6A offline grounding demo

This is text-conditioned image-region grounding with a real pretrained Grounding
DINO model. It is **not** a VLN policy, a metric destination estimator, a planner,
or a robot controller. Nothing in these scripts connects to the robot. Images and
model outputs stay local. See [model selection](VLN_MODEL_SELECTION.md) and
[measured validation](VLN_INPUT_VALIDATION.md).

## Install in a separate checkout

```bash
git fetch origin
git worktree add ../Go2_Navigation-phase6 phase6-vln-input-validation
cd ../Go2_Navigation-phase6
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[test,vln]'
# Optional: exercise the inherited video decoder tests, without DDS or a robot.
uv pip install --python .venv/bin/python av==17.1.0
.venv/bin/python -m pytest -q -rs
```

`vln` is an optional dependency group. CPU FP32 / four PyTorch threads is the
validated backend. Python 3.12 is recommended (the pinned plotting dependency
needs Python >=3.11). No existing environment or operational defaults need changing.
For the exact tested macOS/Python 3.12 transitive versions, install with
`uv pip install --python .venv/bin/python -c requirements-vln-tested.txt -e ".[test,vln]" av==17.1.0`.
The constraints record the tested machine; other platforms need their own validation.
The private recordings are **not** in Git. Set the path to your local copy below.

## Terminal instruction and recording

```bash
.venv/bin/python scripts/phase6_grounding_demo.py \
  --recording /Users/yushan/Desktop/Go2_Navigation/data/recordings/p5d_static_02 \
  --index 200 --history 2 \
  --calibration configs/go2_front_camera_calibration.yaml \
  --instruction 'Find the refrigerator.' \
  --allow-download \
  --output data/outputs/phase6/terminal_fridge
```

First execution downloads only the public pinned checkpoint and processor (~690 MB
weights). `--allow-download` is explicit; subsequent runs default to local cache only.
For enforced offline execution set `HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1`
and omit `--allow-download`. Cache defaults to ignored `data/outputs/phase6/hf_cache`.
Use a new output directory for each run; existing experiment results are not overwritten.

For an already serialized observation:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python scripts/phase6_grounding_demo.py \
  --observation data/outputs/phase6/terminal_fridge/observation.vlnobs.npz \
  --instruction 'Locate the fridge.' \
  --output data/outputs/phase6/replayed_fridge
```

The NPZ already contains calibration; `--calibration` is used only for recording
input. Synthetic intrinsics require `--allow-synthetic`; synthetic results cannot
establish real-image model accuracy. The tested configuration is
[configs/vln_grounding.yaml](../configs/vln_grounding.yaml).

Supported instructions are simple noun phrases, optionally wrapped in `Find`,
`Locate`, or `Show me`. The lexical adapter lowercases and adds a final period per
the model example. This wrapper is **not learned instruction following**. It does
not translate a kitchen destination into a refrigerator, resolve relational
questions, or claim that a room is reachable. E.g. `Go toward the kitchen entrance`
and `Which chair is beside the table?` return `unresolved_unsupported_instruction`
without a model forward pass. Phrases outside this grammar can be semantically
unsuitable even if syntactically accepted: inspect every result.

## Artifacts and interpretation

| File | Meaning |
|---|---|
| `observation.vlnobs.npz` | Original `go2-vln-obs-v1`, including raw RGB/history, geometry, robot pose, clocks and calibration status; no pickle |
| `original.png`, `rectified.png` | Preserved source RGB and calibrated pinhole view |
| `original_vs_rectified.png` | Side-by-side local review, raw left, rectified right |
| `model_input.png` | Exact processor tensor inverted through its own normalization for display, including padding; do not feed this preview back into the pipeline |
| `rectification_valid_mask.npy` | Pixels supported by the raw image remap; not free-space evidence |
| `preprocessing.json` | Raw/view/model-input K, size, distortion, crop, resize, padding, normalization owner and timestamp provenance |
| `source.json` | Exact RGB paths/index rows/SHA256, LiDAR window sources, odometry hash; NPZ-input runs instead reference the input NPZ hash |
| `raw_model_output.npz` | All query logits and normalized center/width/height boxes before thresholding, tokenizer IDs, attention mask and image mask; for unsupported commands it is intentionally empty |
| `proposal.json` | Original instruction, actual model query, pinned checkpoint/processor, versions, candidates, thresholds, timing and warnings |
| `demo.png` | Rectified RGB + instruction/boxes beside independent LiDAR grid, clocks and pose |

Boxes are `xyxy` **image-edge** coordinates in the *rectified* view, bounded by
`[0,width] × [0,height]`. Intrinsics use integer pixel **centers**. Raw model boxes
are normalized `cx,cy,w,h`; the official postprocessor maps them to view pixels.
Do not use these boxes on the raw fisheye image without a calibrated inverse map.
One valid candidate means only that one model region survived filtering. Multiple
candidates stay unresolved, including duplicate boxes; no score-based target choice.
No candidate is not proof that the target is absent. Confidence is not a calibrated
probability. `metric_goal`, `action`, and every `point_base_m` are always null;
`proposal_only=true`, `executable=false`.

The right-hand plot uses the existing 0.5 s obstacle grid: unknown / floor seen /
occupied. The 1.0 s display cloud is retained in the NPZ. Floor seen does not mean
traversable. The orange circle is the near-field blind zone. There is no semantic
LiDAR overlay or inferred connection between the image boxes and occupied cells.
Moving-platform odometry and exposure timing remain uncertain.

## Declared evaluation and independent review

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
  .venv/bin/python scripts/phase6_evaluate_grounding.py \
  --recordings /Users/yushan/Desktop/Go2_Navigation/data/recordings \
  --cases configs/vln_grounding_eval.yaml \
  --output data/outputs/phase6/evaluation
```

The runner freezes the case list and thresholds before inference, uses one loaded
model, and records every case, including failures. It computes runtime and output
validity only. It deliberately leaves semantic accuracy unset. A reviewer must
inspect the images/proposals: confirm object identity and region coverage, check
absent targets for false proposals, and confirm multiple-target ambiguity and
paraphrase consistency. Record reviewer identity/type and reasons separately.
The initial review is a Codex visual review, not independent human ground truth;
human review is still pending. Do not tune thresholds on these 12 cases and call
the resulting score a held-out benchmark.

Keep artifacts under ignored `data/outputs/` and recordings under ignored
`data/recordings/`. If choosing another directory, ensure it is outside Git or ignored.
