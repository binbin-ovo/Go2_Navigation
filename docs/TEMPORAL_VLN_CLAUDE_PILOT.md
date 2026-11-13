# Phase 6B — Claude's human-reviewed episodes (2026-11-13)

Evaluation implementation commit: [`4d5e49642da03558830dba017f5aa150d9fdc684`](https://github.com/binbin-ovo/Go2_Navigation/commit/4d5e49642da03558830dba017f5aa150d9fdc684).

This follow-up evaluates the exact prepared packages supplied by the user. It does
not rebuild episodes, change cutoffs, relabel decisions, or substitute the earlier
Codex pilot. E1–E3 are retrospectively authored navigation-proposal episodes; E4 is a
static absent-target probe. All originate from **test_c** and remain one correlated
group, including overlap with footage previously used during Phase 6B. These are not
independent held-out examples or closed-loop navigation-success tests.

## Package verification and references

Claude's package/documentation branch is `phase6-vln-episode-validation`, commit
`3a84f57b8753b0e5c169f53ae361a33f9ddbc3fe`. Model execution starts from
`phase6-temporal-vln` commit `d2112268b1863185d39f272a10d5bb80cbefaaaa`.
No merge of the episode builder or modification to Claude's checkout is needed.

Exact persistent root:
`/Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6b/episodes/`.
For each package below, the manifest is `model_input/episode.json`, with images and
calibration in the same folder. Evaluator-only labels are
`evaluation_only/annotations.json`; review logs, provenance, operator odometry and
labeling panels stay under `evaluation_only/` and never become model inputs.

| Package | Frames | Decisions | Role |
|---|---:|---:|---|
| p6b_test_c_e1_turn_right | 107 | 3 | Retrospective turn toward counter |
| p6b_test_c_e2_turn_left | 135 | 3 | Retrospective turn toward shoe rack |
| p6b_test_c_e3_forward_bottles | 93 | 3 | Retrospective approach to bottles |
| p6b_test_c_e4_absent_sofa | 64 | 2 | Static absent-target probe |

All four manifests are accessible and pass `load_episode`, including hashes of all
399 images and calibration files, ordered observations, one source/clock epoch,
explicit bounds, no gaps, and null robot_state_ref. All annotation files pass
`load_annotations`, including manifest binding and exact decision coverage. The
provided eight manifest/annotation SHA256 values match. Every decision's review-log
label equals its annotation entry and includes a review timestamp and owner's reply.

**All 11 action references are human-reviewed**, not pending or agent-only. They were
drafted by Claude from causal panels and accepted unchanged by one repository-owner
reviewer in chat. This is human acceptance of agent drafts, not blind independent
annotation; anchoring and single-reviewer limitations apply. Seven decisions are
marked ambiguous. Reference completion is complete for two, incomplete for six,
and unknown for three. Unknown completion is excluded from that comparison; it is
not relabeled incomplete. No case distinguishes similar-looking destinations.

The private `audit/claude_pilot_v1_preflight.json` retains exact paths, hashes,
selected frame IDs, 434 package-file hashes, reference coverage and frozen source
files. It is separate from all packages. Original recordings and packages are read-only
inputs to this run.

## Fixed model and causal input

The previous checkpoint, prompt, parser and preprocessing are unchanged:
`HuggingFaceTB/SmolVLM2-2.2B-Instruct` and processor revision
`be92141c8d0ceb98596a923c308ac2bbb4a7c466`, CPU FP32/eager, four threads, seed 0,
greedy generation with at most 160 new tokens. This remains a **VLM navigation-proposal
baseline**, not a reproduced VLN policy.

The run declares 11 existing cutoffs × two native input modes = 22 inference calls.
Temporal mode takes at most four chronological available observations over six
seconds, including current; current-only takes the same latest available image.
Earlier decisions have shorter available history rather than fabricated padding.
Only full instruction, causal RGB history and relative observation times enter the
fixed model prompt. Episode IDs, action references, completion/progress labels,
operator odometry and future images are excluded. References are read by the runner
only after inference and shown in a labeled evaluation panel.

Preserve the package's **0.050 s assumed decode margin**; do not replace it with the
earlier Codex pilot's 0.1 s. Exposure uses the 48 ms estimate calibrated on test_c
itself, so timing validation is not independent. Availability is a receipt-based
estimate, not hardware synchronization. Calibration remains measured fisheye to
rectified pinhole, then the existing official 384×384 unsplit SmolVLM input with
per-frame intrinsics/provenance. T_base_camera and registered depth remain unknown.

The frozen prompt asks for UNRESOLVED when a target is absent, whereas the provided
E4 references accept either search turn. Preserve both unchanged: report action
agreement against the supplied references and this policy/reference mismatch, rather
than silently changing prompts or labels. Agreement is not evidence that a search
turn is safe or that the target was found.

## Reproduction from the prepared packages

Use the existing isolated environment from [TEMPORAL_VLN_DEMO](TEMPORAL_VLN_DEMO.md).
The repeatable `--episode` option selects exact nested manifests; it does not discover
neighboring packages. No builder invocation is part of this workflow.

```bash
cd /Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6b-worktree
ARTIFACTS=/Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6b
EPISODES="$ARTIFACTS/episodes"
export PYTHONPATH=src HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MPLCONFIGDIR="$ARTIFACTS/mpl" XDG_CACHE_HOME="$ARTIFACTS/cache"
.venv/bin/python -u scripts/run_temporal_vln.py \
  --episode "$EPISODES/p6b_test_c_e1_turn_right/model_input/episode.json" \
  --episode "$EPISODES/p6b_test_c_e2_turn_left/model_input/episode.json" \
  --episode "$EPISODES/p6b_test_c_e3_forward_bottles/model_input/episode.json" \
  --episode "$EPISODES/p6b_test_c_e4_absent_sofa/model_input/episode.json" \
  --mode both --annotations-name ../evaluation_only/annotations.json \
  --cache-dir "$ARTIFACTS/hf_cache" --output "$ARTIFACTS/claude_pilot_v1" --video
.venv/bin/python scripts/evaluate_temporal_vln.py \
  --run "$ARTIFACTS/claude_pilot_v1" \
  --annotations-name ../evaluation_only/annotations.json \
  --static-episode p6b_test_c_e4_absent_sofa \
  --output "$ARTIFACTS/claude_pilot_v1/evaluation_grouped.json"
```

Both commands refuse output overwrite. To repeat, choose a new run directory and
point the evaluator there. The evaluator loads only declared packages and saved
results, verifies manifest/decision binding, scores reviewed human action sets, and
reports missing/pending/agent-only references as pending. It never loads a model,
changes labels or rebuilds frames. Summaries group by recording and separate static
cases, navigation proposals and human-marked ambiguity; counts do not imply independent
samples. Full per-decision labels/results stay private.

## Actual results

All **22 calls completed**: 11 temporal, 11 current-only. Every temporal call encoded
four actual images (vision hook [4,3,384,384], 324 image tokens); current-only encoded
one (81 tokens). A post-run audit compared each saved message list with the whitelist
construction from the instruction and causal selection, checked source frame IDs,
exposure/availability cutoffs, tensors and manifest hashes. No annotations or future
observations entered model inputs. All **434 package files are byte-for-byte unchanged**.
The six previously frozen inference files and the run's CLI hash also match preflight.

All 11 action references were usable and human-reviewed: **zero pending action
references**, zero substituted labels, zero missing calls. Human instruction-progress
labels are retained, but semantic comparison of free-text rationales still requires
adjudication; this run does not create new human labels or independently relabel cases.

| Supplied-reference action agreement | Temporal | Current only |
|---|---:|---:|
| E1: retrospective right turn | 2/3 | 2/3 |
| E2: retrospective left turn | 0/3 | 2/3 |
| E3: retrospective approach | 3/3 | 3/3 |
| **E1–E3 navigation proposals** | **5/9 (55.6%)** | **7/9 (77.8%)** |
| **E4 static absent-target probe** | **2/2** | **0/2** |
| All correlated test_c examples, descriptive only | 7/11 (63.6%) | 7/11 (63.6%) |
| Human-marked unambiguous subset | 2/4 | 3/4 |
| Human-marked ambiguous subset | 5/7 | 4/7 |

These denominators count decision-level action-set comparisons, not independent
recordings. There is one correlated recording group, no confidence interval or
significance claim. E4 is deliberately separated: its matches cannot be treated as
navigation success or used to hide the lower E1–E3 temporal agreement.

Temporal outputs: TURN_RIGHT ×8, ADVANCE ×3; 11/11 parsed. Current-only: TURN_RIGHT ×3,
TURN_LEFT ×3, ADVANCE ×3 and **two rejected responses**; 9/11 parsed. Both rejected E4
outputs ended inside a JSON reason string. No generation hit the 160-token limit, so
these are malformed model outputs, not a caller truncation. Their raw text is saved;
the evaluator counts them as action disagreements and never extracts a convenient
TURN_RIGHT from invalid JSON. Their adapter category is UNRESOLVED with unknown status.

Both modes agree with the provided E3 action sets, but their reasons largely echo
the instruction. Temporal mode proposes right turns throughout E2. Both modes keep
turning at the supplied terminal E1/E2 stopping references. E4 temporal responses
match an allowed search-turn label while their reasons copy instruction/history-time
text; that match does not establish target grounding, safety or successful search.
There is **no demonstrated temporal advantage** on these packages.

Temporal mode repeats the same category in all seven within-episode consecutive
pairs; current-only repeats in six evaluable pairs (E4's pair is rejected). Neither
has opposite consecutive turn categories. These are descriptive sequence flags, not
counterfactual trajectory failures. Sequence metrics are omitted for ambiguity-filtered
subsets because filtering can make nonadjacent cutoffs appear adjacent.

### Completion and progress limits

One temporal response declares COMPLETE while still proposing TURN_RIGHT. Its human
reference completion is **unknown**, so its completion claim is unverified and cannot
be scored as true or false. Do not relabel that reference. Neither mode declares
COMPLETE at the two human-labeled completed decisions. Against known reference
completion with parseable status, consistency is 6/8 temporal and 4/6 current-only;
three unknown reference states and two malformed current-only outputs are excluded
as applicable. Both modes have zero *adjudicable* COMPLETE claims, so the verified
false-completion count remains null, not a claim of zero false completions.

The human labels concern the operator's recorded visual progress, not the outcome
of executing model proposals. Goal reaching, collision avoidance, safe execution and
recovery after counterfactual actions remain UNKNOWN. All output paths retain
execution_authorized=false, metric_goal=null and unknown native action frame.

## Runtime and artifacts

Actual call interval: **2026-11-13 04:26:10.945–04:29:24.845 America/New_York**
(09:26:10.945–09:29:24.845 UTC). Same Mac M3 Pro / 36 GiB, Python 3.12.11,
Torch 2.6.0, Transformers 4.51.3, local CPU FP32. Exact versions are in `model.json`
and the unchanged tested dependency constraints.

| Seconds | Temporal mean / median / range | Current mean / median / range |
|---|---|---|
| Preprocessing | 0.127 / 0.127 / 0.125–0.129 | 0.033 / 0.033 / 0.032–0.035 |
| Generation | 11.125 / 9.191 / 8.870–29.250 | 5.557 / 5.463 / 5.283–6.123 |
| Inference pipeline | 11.252 / 9.318 / 8.998–29.379 | 5.590 / 5.496 / 5.315–6.156 |
| Estimated replay image age at output | 11.381 / 9.440 / 9.148–29.499 | 5.720 / 5.650 / 5.426–6.255 |

Image age at cutoff was 0.099–0.161 s. Replay age adds per-call preprocessing and
generation; it does not claim measured live latency or mix host and robot clocks.
Peak process RSS was **9.727 GiB**, a shared lifetime peak, not independent per-mode
memory or GPU memory. The slow first temporal forward is retained in all statistics.
Calls run temporal then current-only; no randomized performance experiment is claimed.

Initialization was initially delayed during local Transformers source-file reads,
observed through process sampling. A separate import-only diagnostic completed; the
original inference process resumed, without a restart, discarded call or changed
model. Per-call timing excludes dependency imports and model construction. Diagnostic
logs are retained privately; no installation or shared calibration environment changed.

Persistent private results:
`/Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6b/claude_pilot_v1/`.

- `evaluation_grouped.json`: authoritative grouped human-reference scoring with source
  annotation hashes, per-decision outcomes, E4 separation and pending-reference counts.
- `evaluation.json`: initial scoring artifact retained; identical action scores, but
  includes sequence flags within ambiguity subsets. The grouped revision omits those
  inapplicable sequence flags. This reporting-only correction did not rerun inference.
- `summary.json`, `summary_by_mode.json`: runner's original summaries, preserved.
- `<episode>/<decision>/<mode>/`: original native response, proposal, exact prompt,
  selected-image provenance, normalized tensors/tokens, labeled review panel and previews.
- `review.mp4`: H.264, 1650×990, 22 frames / 22 s at one panel per second, not robot timing.
  Every frame decoded successfully. Five representative panels were visually checked
  for readability; no new semantic labels were assigned.
- Sibling `audit/claude_pilot_v1_preflight.json` and `claude_pilot_v1_postflight.json`:
  package hash inventory, selected frames, model-input/freeze audit and timing checks.

Private images, video, annotations, responses, odometry and weights remain Git-ignored.
Only safe code, aggregate documentation and synthetic tests are published.

## Code and automated validation

The runner now accepts repeated `--episode` exact paths, preserving the existing
single-episode and root-discovery interfaces. The saved-output evaluator adds explicit
recording grouping, static-case separation, manifest/cutoff checks and human-only
scoring. No sensor adapter, model, prompt, parser, schema, calibration or operational
default changes were made. Episode contract/fixture from `5c7f586` remain unchanged.

Eight new synthetic regressions cover exact nested selection without neighboring
packages, labels excluded from the pre-inference path, correlated/static grouping,
pending/agent/missing references, changed input provenance and incomplete inference.
Normal unit tests never need private packages or pretrained weights.

```bash
SMOLVLM_MODEL_CACHE="$ARTIFACTS/hf_cache" HF_HUB_OFFLINE=1 \
  PYTHONPATH=src TOKENIZERS_PARALLELISM=false MPLCONFIGDIR="$ARTIFACTS/mpl" \
  .venv/bin/python -m pytest -q -rs
.venv/bin/ruff check scripts/run_temporal_vln.py scripts/evaluate_temporal_vln.py \
  tests/test_temporal_vln_runner.py tests/test_temporal_vln_evaluation.py
.venv/bin/ruff format --check scripts/run_temporal_vln.py scripts/evaluate_temporal_vln.py \
  tests/test_temporal_vln_runner.py tests/test_temporal_vln_evaluation.py
```

Full tests including actual pretrained model tests: **478 passed, one hardware test
skipped in 30.85 s**. Ruff check and format-check passed on all four changed/new Python files.
The skipped test is `tests/test_go2_hardware.py`, requiring GO2_HARDWARE=1. No hardware
connection was attempted. The final full-run log is `audit/claude_pilot_full_tests_final.log`.

Next research work should use newly collected, independently reviewed and diverse
episodes with explicit completion criteria, while preserving this correlated pilot
as a fixed diagnostic. Prompt/model changes need a new declared experiment. Metric
grounding still requires measured extrinsics/depth and validated timing/odometry;
closed-loop behavior requires a separately authorized controlled experiment.
