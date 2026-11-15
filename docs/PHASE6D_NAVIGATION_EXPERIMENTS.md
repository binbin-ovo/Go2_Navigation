# Phase 6D waypoint execution and visible-goal navigation

Date: 2026-11-15. Branch `phase6-waypoint-validation`; isolated worktree
`/Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6d-worktree`, based exactly on
Phase 6C publication `45154f6b9f05227de74254cb4562d7e0ad07b7dd`. No main merge or
operational-default change. See [contract audit](WAYPOINT_EXECUTION_AUDIT.md).

## Fixed model, runtime and preservation

LightOriginsHQ/LightNav-0 checkpoint/processor/RVQ
`826dc5fbfa37afa8293d2e336d329b6ffc0bfb64`, official code
`3015508b70fb6e30bbd66701e94c9b916dcb8df8`. Existing MPS BF16/SDPA runtime, no
quantization, same-BF16 CPU pooling fallback patch
`bec34b7d855f33ac94ec212a61ca507d2d677d82441add0cd1b2dfe7088e554c` unchanged.
No new weights, installations, real-robot client, network motion transport or training.

Actual Mac: Apple M3 Pro 18-core GPU, macOS 26.6.2 arm64, 36 GiB unified RAM.
Preflight available RAM 15,329,476,608 bytes, free disk 22,953,807,872 bytes.
Python 3.11.13 / Torch 2.10.0 / Transformers 5.8.0 / NumPy 2.4.6 / MuJoCo 3.3.7.
MPS available outside the sandbox. No NVIDIA driver/toolkit, CUDA runtime null.
Existing isolated `phase6c/model-env` and `phase6c/core-env` reused unchanged.
The new worktree's ignored `.venv` symlink targets the latter; no sensor/calibration
environment changes. Windows inventory remains independent and unexecuted here.

Artifact root (local, ignored):
`/Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6d/waypoint_validation/`.
The four Claude packages (434 files) and saved SmolVLM baseline (293 files) match
the strict Phase 6C final inventory. A separate 273-file snapshot protects the old
failed simulation, its full-chunk review, and Claude's new review files. Source
files and old results are read in place, not rebuilt, relabeled or overwritten.

## Pre-model evidence

- Native units/frame/STOP audit: no decoder or coordinate defect established.
  Original first-waypoint baseline retained. One added `bounded_prefix` policy
  follows fixed-frame poses under explicit speed/rate limits; no evaluator goal
  appears in its inputs. It is a holonomic camera proxy, not upstream unicycle MPC.
- Ten model-free numeric cases and four camera previews saved under
  `controller_diagnostics_v1/`; video under `controller_diagnostics_v1_review/`.
  The deliberate wall intrusion is rejected after 0.29 s at the last safe pose.
  Known-path doorway approach reaches the centre in 2.3125 s, then holds to 3 s.
  Endpoint/cross-track errors are numerical roundoff. Geometry-assisted paths are
  controller diagnostics, never model-generated navigation.
- Actual colored-landmark rendering verifies camera axes, image orientation and
  proxy following. Source and native processed previews are pixel-identical.
  Pinhole simulated inputs never pass through measured Go2 fisheye rectification.
- Claude's review, commit `4a168e165e8eee58b7fcd4fc5a192e9791d3c278`, finds the
  original doorway recognizable. Keep the same scene/camera; new close-up views
  remain a limitation, not a human-certified stopping observation. Review report:
  [PHASE6D_BEHAVIOR_REVIEW](https://github.com/binbin-ovo/Go2_Navigation/blob/4a168e165e8eee58b7fcd4fc5a192e9791d3c278/docs/PHASE6D_BEHAVIOR_REVIEW.md).
- Saved repetition audit: eight distinct pixel tensors at old steps 1–8, despite
  byte-identical outputs. HF does not use the episode ViT cache. No stale-image
  cause established; model interpretation/repeated action selection remain open.

## Declared diagnostic suite (before model results)

Four starts × two policies = **eight conditions**, one run each, no outcome-based
extensions or retries. This is development diagnostics, not broad generalization.
Instruction, identical for all: **“Approach the open doorway and stop just before
passing through it.”** It is changed once from the old right-turn prompt before
the new suite. Old scores and failed rollout remain historical evidence.

| Case | Initial world pose [x m, y m, yaw rad] | Purpose |
|---|---|---|
| ahead | [4.3, 0, 0] | visible doorway ahead |
| left | [4.3, 0, -0.4363323129985824] | same doorway initially left |
| right | [4.3, 0, +0.4363323129985824] | same doorway initially right |
| near | [6.05, 0, 0] | already in the evaluator stop region |

Scene: unchanged `configs/simulation_vln_room.xml`, SHA256
`a3b752beaf74151ffb1489dd425c69aa66cda43fbdc375b7a9eae2e765c666e2`.
Camera: 448×256, fovy 60 degrees, height 0.65 m, body x offset 0.16 m, forward
along body +x. Goal rectangle x=[5.8,6.5], y=[-0.6,0.6], no orientation requirement;
centre [6.15,0]. Only evaluator/constraint code receives geometry.

Policies: unchanged first waypoint, at most 0.25 m / 32 degrees over 0.25 s;
new ordered prefix at 0.8 m/s and 3 rad/s, <=0.01 s collision samples, same 0.25 s
replanning window. Native chunk and selected targets are retained separately from
actual safe motion. The prefix can interrupt at the last safe substep; baseline
atomic rejection/rollback remains its original behavior.

Uniform budget: **24 simulated seconds, 96 decisions, 900 host seconds** per
condition, including asset verification/model loading in the process-level limit.
Responses older than 120 host seconds are rejected before motion. The longest
known geometry route is 1.85 m with <=25-degree initial heading error: <=2.458 s
for a turn then approach at prefix limits (2.3125 s if concurrent). The uniform
24 s budget is >9× that geometric feasibility time; it cannot guarantee a learned
policy makes progress. No failed run will be selectively extended.

Static step-and-wait: capture, pause for inference, execute chosen motion, render
again. Actual simulated images at 4 Hz enter native SlowFast; reset only at episode
or clock-epoch boundaries. No previous proposals/executed actions, labels, pose,
target coordinates, shortest paths or evaluator completion are passed to LightNav.
Host latency and simulated time are separate. No dynamic obstacle or live-latency claim.

Success requires **accepted native action STOP inside the rectangle**. Goal entry
alone is separate. APOS completion and RVQ stop evidence are preserved separately.
Also report premature STOP, post-entry movement, and external budget/safety/error
termination. Goal entry never triggers an execution override. “Continued after
entry” includes any translation or rotation after the first in-region sample,
even motion still within the rectangle. Path error is adherence to the selected
native path, not goal correctness or dynamic tracking performance.

## Results

Model-driven suite not yet executed at this implementation checkpoint. It must
run from the hashed reviewed declaration; results will be appended without
changing the declared settings. No navigation success is claimed yet.

## Verified automated tests

Full suite: **590 passed, 9 skipped in 30.67 s**, including cached SmolVLM inference.
Core skips: physical Go2 disabled; two optional MuJoCo modules; six native-runtime
cases requiring the separate model environment. Optional environment: 18 cases
passed before a Transformers source-read stall interrupted the last pooling test;
that single same-dtype CPU/MPS check passed separately in 34.46 s. Thus all 19
optional cases executed successfully across those runs. The interrupted combined
run is not reported as a completed all-pass run. Logs/XML: `audit/`.

The first sandbox full-suite run had 587 passes, two failures, nine skips: existing
ffmpeg/ffprobe checks could not open OpenMP shared memory. After running outside
that restricted sandbox, with no test weakening or environment install, the full
suite passed; one new stale-response case was added before the final run.

## Reproduction

Use existing environments. Every artifact output directory/file must be new.

```bash
cd /Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6d-worktree
P6C=/Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6c
P6D=/Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6d/waypoint_validation
PYTHONPATH=src MUJOCO_GL=cgl "$P6C/model-env/bin/python" scripts/diagnose_waypoint_execution.py \
  --assets "$P6C/lightnav" --output "$P6D/controller_diagnostics_repeat"
PYTHONPATH=src .venv/bin/python scripts/review_waypoint_validation.py \
  --diagnostics "$P6D/controller_diagnostics_repeat" --output "$P6D/controller_diagnostics_repeat_review"
PYTHONPATH=src GO2_HARDWARE=0 HF_HUB_OFFLINE=1 \
  SMOLVLM_MODEL_CACHE=/Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6b/hf_cache \
  .venv/bin/python -m pytest -q -rs
PYTHONPATH=src MUJOCO_GL=cgl MUJOCO_RENDER_TEST=1 LIGHTNAV_ASSETS="$P6C/lightnav" \
  "$P6C/model-env/bin/python" -m pytest -q -rs tests/test_waypoint_native_integration.py \
  tests/test_waypoint_simulation.py tests/test_vln_simulation.py tests/test_lightnav_runtime_integration.py
```

The checked-in YAML is a draft template. A reviewed copy must bind the report's
commit, SHA256, reviewed scene SHA256 and actual processor-preview manifest. Set
its `status` to `reviewed_ready` only after reading the evidence. Do not fabricate
review fields. `freeze` refuses the draft. It hashes implementation and scene;
subsequent changes are rejected rather than silently applied to a frozen experiment.

```bash
PYTHONPATH=src .venv/bin/python scripts/run_waypoint_suite.py freeze \
  --config "$P6D/reviewed_suite_v1.yaml" --output "$P6D/frozen_suite_v1.json"
PYTHONPATH=src MUJOCO_GL=cgl HF_HUB_OFFLINE=1 \
  PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.45 \
  "$P6C/model-env/bin/python" -u scripts/run_waypoint_suite.py run \
  --declaration "$P6D/frozen_suite_v1.json" --assets "$P6C/lightnav" --output "$P6D/suite_v1"
PYTHONPATH=src .venv/bin/python scripts/review_waypoint_validation.py \
  --suite "$P6D/suite_v1" --output "$P6D/suite_v1_review"
```

Native data: `<suite>/<case>__<policy>/rollout/inference/NNN/`; all source images,
jobs, chunks and actual-path traces remain alongside. Each run has an isolated
process so a model stall cannot exceed its host budget or leak history to another
condition. The parent records every declared condition, including worker failures.
Raw frames are synthetic; private Go2 images/labels remain outside these new jobs.
