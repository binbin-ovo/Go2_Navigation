# Offline paired-view evaluation

Branch: `feature/vln-viewpoint-evaluation`. Exact base:
`73c83f19cf83ee01640e467e09085dd669e51763`, the published
`feature/vln-interactive-review` head checked at task start. This branch adds an
offline reader and reports; it does not change the website, receiver, calibration,
model configuration, history policy, control code, main, or operational defaults.

The browser-guided implementation in [VLN_INTERACTIVE_REVIEW](VLN_INTERACTIVE_REVIEW.md)
supersedes the older terminal description lower in the handoff. Each Ask creates a
new trial/revision with fresh instruction history, collects a short observation,
submits one request, and waits for the operator's verdict. The experiment therefore
links **two independent queries for evaluation**. There is no cross-view model
memory, autonomous search, continuous navigation, or navigation-success measurement.
The evaluator is optional after collection; it is not a prerequisite for Claude's
live experiment.

**Completed integration:** four real pairs from `live_09`/`live_10` have now been
analyzed against the current recorder and method-v2 linkage. See
[the aggregate findings and exact inspected versions](VLN_PAIRED_VIEW_RESULTS.md).
Real paired evidence is no longer pending.

## One command after collection

Use the existing isolated core environment, not the sensor/calibration or model
environment. Commands below run from this feature worktree. Set `CORE_PY` to that
environment's Python executable and `RUN` to the **completed** interactive archive.
The archive must contain a final `summary.json` with `termination` and
`ended_wall_utc`, plus both `trial_end.json` files. A trial end alone is insufficient:
late responses can still be appended while its run is active. Do not invoke the
evaluator on a run being reopened for review or otherwise written.

```bash
env GO2_HARDWARE=0 MUJOCO_RENDER_TEST=0 PYTHONDONTWRITEBYTECODE=1 \
  "$CORE_PY" scripts/analyze_vln_viewpoint_trial.py \
  --run "$RUN" \
  --before-trial "$BEFORE_TRIAL_ID" \
  --after-trial "$AFTER_TRIAL_ID" \
  --annotation "$PAIR_ANNOTATION" \
  --output data/outputs/vln_viewpoint/completed_pair_01
```

The output directory must be new, separate from the archive, and Git-ignored (or
outside Git entirely). Use a new suffix for another analysis. `--annotation` is
optional: without turn evidence, the report still compares reviewed images but
states that no post-turn observation has been verified. `--before-prediction` and
`--after-prediction` select explicitly among multiple archived predictions. When a
trial has multiple predictions and no selection, all are displayed and the paired
conclusion is insufficient evidence. Nothing silently selects the latest answer.

Outputs are `index.html`, `summary.json`, `timeline.json`, and local image copies in
`assets/`. Open `index.html` locally. It has no server, remote assets, or JavaScript.
It shows the instruction, each answer's own marked native input, all actually
selected model-input images, native tokens/waypoints/STOP claims, every review and
its provenance, source clocks, arrival times, and review ages. Native paths occupy
a separate local-coordinate chart labelled **UNEXECUTED**. They are never projected
onto LiDAR or used to compute clearance.

Private imagery, human comments and detailed reports stay local and ignored. The
JSON preserves verbatim reviews; do not publish it as an aggregate report.

## Import the recorder's completed pair linkage

For Claude's existing append-only `go2-vln-paired-view-evaluation-v1` records, use
this **instead of** `--annotation`:

```bash
env GO2_HARDWARE=0 MUJOCO_RENDER_TEST=0 PYTHONDONTWRITEBYTECODE=1 \
  "$CORE_PY" scripts/analyze_vln_viewpoint_trial.py \
  --run "$RUN" --before-trial "$BEFORE_TRIAL_ID" --after-trial "$AFTER_TRIAL_ID" \
  --linked-pairs evaluation/paired_views.jsonl --method-version 2 \
  --output data/outputs/vln_viewpoint/completed_pair_02
```

The importer chooses the requested pair and explicitly declared method version,
then validates the referenced prediction IDs against the original archive. Earlier
records for that pair remain in `pair_annotation.linkage_import`, with source-file,
line, record SHA-256 and full-file SHA-256 provenance. If there are multiple records
for the same pair and method, use `--linkage-line`; there is no silent latest-wins
selection. One relinked pair remains one example. Copied label summaries are never
used as source judgments. This adapter supports declared method v2; earlier
undeclared records are retained as historical evidence, not reinterpreted as v2.

`receipt_ordering_evidence` recomputes the margin against the selected v2 interval
union from original snapshot receipt times. It checks source/session/robot epochs,
input gaps, index continuity and archived camera counters. The raw yaw estimator is
not rerun. The linker's `verified` flag does **not** supply a camera-exposure bound.
Imported turn bounds remain null, and the existing strict capture-time checks remain
unchanged. `outcome.reviewed_image_outcome` is the separately supported image
judgment, including `persistent_false_absence_in_reviewed_images` when both native
OPOS-0 outputs receive explicit human disagreement. The older `outcome.code` and
timing description are retained for compatibility; do not use them to erase the
image judgment when exposure timing remains incomplete.

Combine the existing reports without running analysis or inference again:

```bash
"$CORE_PY" scripts/summarize_vln_viewpoint_reports.py \
  "$CHAIR_REPORT" "$OVEN_REPORT" "$MICROWAVE_REPORT" "$DISH_WASHER_REPORT" \
  --output data/outputs/vln_viewpoint/combined_pair_comparison
```

The combined HTML/JSON/Markdown table rejects duplicate archive/trial pairs. Exact
resolved local IDs and commands for the completed four-pair analysis are preserved
in the Git-ignored `paired_integration_01/reproduce.sh` and `commands-final.json`
artifacts described in the results document. No repeat collection is needed.

## Evaluation-only annotation contract

The sidecar is never supplied to inference. Both trial IDs must match the CLI, and
both queries must have the same exact saved instruction/operator wording. Different
trial IDs and instruction revisions are expected and validated separately. Version
`go2-vln-viewpoint-pair-v1` supports this shape (values below are **synthetic**, not
real turn times or calibration):

```json
{
  "schema": "go2-vln-viewpoint-pair-v1",
  "episode_id": "synthetic-chair-pair",
  "target_wording": "chair",
  "before_trial_id": "synthetic-before",
  "after_trial_id": "synthetic-after",
  "turn": {
    "direction": "left",
    "provenance": {
      "source": "synthetic example; replace with actual operator/video annotation",
      "annotator": "synthetic_fixture"
    },
    "start": {
      "time_s": 110.0,
      "clock": "host_monotonic",
      "clock_epoch": "synthetic-pair-session",
      "uncertainty_s": 0.01
    },
    "end": {
      "time_s": 112.0,
      "clock": "host_monotonic",
      "clock_epoch": "synthetic-pair-session",
      "uncertainty_s": 0.01
    }
  }
}
```

`end` means the manual turn has finished **and the operator has stopped**. Direction
is the operator's annotation, not inferred odometry. Provenance should identify who
annotated, from which saved evidence, and how the timing bound was obtained. Unknown
times/bounds may be omitted or null; they are not replaced with zero. The IDs and
target can still link the views without a complete `turn` object.

For archived host events, replace `time_s` in a start/end stamp with:

```json
"event_ref": {"path": "events.jsonl", "line": 12, "event": "operator_stopped"}
```

The referenced event must actually exist at that one-based line with that exact
event name; the reader resolves `host_monotonic_s` and retains the record hash and
line. Event names here are examples, **not events assumed to exist in the recorder**.
Keep `clock`, `clock_epoch`, `uncertainty_s` and annotation provenance. A generic
trial-end event is not evidence of a physical stop. Do not fabricate an event or
edit the source archive to make this example work.

Clock rules:

- All numeric times are seconds. `host_monotonic` is scoped to the saved trial's
  `session_id` (use that value as `clock_epoch`), not to another run/machine. Source
  camera/robot stamps retain their recorded clock and source epoch. This first
  version pairs trials within **one** completed run; it does not infer cross-run
  clock continuity.
- The marked image is identified by `source_image.job_frame_index`, checked against
  the native selected indices and its PNG hash. Its timestamp comes from the matching
  `snapshot.history[index].source`, not from prediction arrival or an unselected
  `current_source`. Every selected history image and prepared image is checked.
- `Stamp.uncertainty_s` is a **one-sigma guess** in the existing sensor contract.
  It is preserved as `source_uncertainty_sigma_s` and in the original stamp, never
  silently treated as a bound. A source exposure stamp can establish ordering within
  its domain/epoch only with an explicit
  `provenance.exposure_source_error_bound_s`. An archived
  `exposure_monotonic_s` plus `monotonic_mapping` provides
  an exposure estimate in the same session's host clock. Its bound is unknown unless
  the archive explicitly supplies `provenance.exposure_monotonic_error_bound_s`.
  **The current live recorder does not supply these bounds.** The earlier latency
  calibration sigma is not treated as a guaranteed error bound. This reader does
  not modify the recorder or invent a bound.
- Explicit cross-clock mappings can be supplied in a top-level `clock_mappings`
  list. Each entry requires `source_clock`, `source_epoch`, `target_clock`,
  `target_epoch`, `offset_s`, `uncertainty_s` (nonnegative), and `provenance`.
  Target time = source time + offset; uncertainty bounds add. Only one direct
  mapping per source domain/epoch is supported. These are evaluation assertions,
  never automatic robot↔host alignment or reused calibration. A mapping cannot
  turn an unknown receipt-to-exposure delay into a known exposure time.
- `before`/`after` require nonoverlapping bounded capture and turn intervals.
  `during`, `overlaps_uncertain_turn_boundary`, and `unknown` stay distinct. A
  clean paired-turn conclusion also requires all selected before-history frames
  before the turn and all after-history frames after it. Mixed history is flagged.
- `CURRENT`/`STALE` at receipt stays separate from exposure age at human review.
  A stale proposal may still be reviewed for grounding in its own saved image;
  staleness never becomes navigation authorization. Unknown age is JSON `null`.
  An exposure point estimate without an error bound is explicitly labelled estimated,
  and cannot establish verified post-turn timing. The three-second live threshold
  is not changed or recomputed into a new benchmark here.

With the present live metadata, a genuine reviewed pair may therefore demonstrate
wrong→correct **in the two reviewed images** while temporal verification remains
incomplete. That is an intentional evidence limit, not a reason to delay collection
or invent calibration.

## Archive and review validation

The reader follows the published `run.json`, `trial.json`, `trial_end.json`,
`predictions.jsonl`, `human_labels.jsonl`, request `job.json`/`snapshot.json`/
`receipt.json`, native `result.json`, prepared PNG and selected native PNG layouts.
Request identity, episode, sequence, source epoch, instruction/revision, causal
availability and own-trial history are checked. Each response is compared with its
own trial; the pair's revisions need not match. Superseded, cancelled, rejected or
mismatched responses remain visible but are excluded from pair conclusions.

`source_frame` required fields are compared individually with the snapshot source;
the current recorder's optional receipt/exposure monotonic timestamps are checked
explicitly when present. Additional fields are preserved. The review's optional
`displayed_image_sha256` must also match the prediction image. For the live path,
fresh-history admission is checked using archived `session_receive_monotonic_s`
where available; first sensor receipt can precede Ask for a frame admitted after
reset. That earlier receipt is retained and flagged, not relabelled as capture or
discarded to force a timing result. Legacy records without the admission timestamp
retain the conservative receipt-based check.

Missing images/results are reported, never reconstructed or inferred again. Missing
required run/trial closure causes a clear refusal. A missing native `result.json`
can use the identical result archived in `receipt.json`, with a warning; a missing
receipt, snapshot, or selected image excludes that prediction. Absolute job paths
from the original recorder are relocated only to the canonical request layout
inside this run, never followed into another recording. Escaping paths are refused.
Consumed files are hashed and rechecked before report publication; this detects
concurrent changes but is not a lock against another writer. Operator-provided
closed-archive discipline remains required. Large raw NPZ/LiDAR files are not read
or revalidated; saved source provenance is retained and selected/prepared PNGs are
checked. No model weights, hardware receiver, DDS, or control library is loaded.

Review semantics:

| Saved feedback | Supported conclusion in the reviewed image |
| --- | --- |
| Guided `correct` | Correct target mark; target present |
| Guided `wrong` | Wrong target mark; visibility elsewhere remains unknown |
| Guided `target_not_in_view` | Human absence judgment; not an OPOS-0 agreement |
| `agree_not_in_view` / `disagree_target_in_view` | Explicit OPOS-0 agreement/disagreement and absence/presence, separately |
| `unsure`, `no_point`, or no review | No confirmed marking judgment |
| Older `go2-vln-human-review-v1` with one explicit prediction | Preserve its legacy scope and separately stated marking, presence, path and stopping fields |
| Older review covering a prediction set/trial | Retain as group evidence; never copy onto each prediction |

Guided reviews must match the prediction ID and exact `source_image`, be explicitly
confirmed and `model_feedback: false`. File/line/hash provenance is recorded because
guided records do not have review IDs. Repeated reviews remain separate. Opposing
known judgments become `conflicting`; a known plus unsure judgment is `ambiguous`.
No latest-wins rule, comment interpretation, or automatically inferred verdict from
correction clicks. Marks, notes and corrections are retained as additional original
evaluation records. Guided feedback leaves path, stopping and navigation **unknown**.
Native STOP/completion tokens are model claims, not human-verified completion.

The summary separates the reviewed-image comparison from temporal verification.
It reports supported wrong→correct changes, visibility changes, persistent wrong
marking, unavailable post-turn evidence, ambiguous/missing human feedback, or
insufficient evidence. Initially absent→visible/correct is a visibility change,
not automatically improved recognition accuracy. No general success rate is computed
from correlated samples in one room.

## Synthetic example and actual verification

```bash
env GO2_HARDWARE=0 MUJOCO_RENDER_TEST=0 PYTHONDONTWRITEBYTECODE=1 \
  "$CORE_PY" scripts/analyze_vln_viewpoint_trial.py \
  --run tests/fixtures/vln_viewpoint/run \
  --before-trial synthetic-before --after-trial synthetic-after \
  --annotation tests/fixtures/vln_viewpoint/pair.json \
  --output data/outputs/vln_viewpoint/synthetic_example
```

The fixture, verdicts, clocks, native tokens and drawings are entirely synthetic;
there is no model inference or real motion. The expected outcome is wrong marking
becoming correct in the reviewed new view, relative to the synthetic declared turn.
The fixture generator refuses existing destinations and can reproduce the archive
at a fresh location. See [fixture README](../tests/fixtures/vln_viewpoint/README.md).

Focused test command (CPU only; no model/hardware/full regression):

```bash
env GO2_HARDWARE=0 MUJOCO_RENDER_TEST=0 PYTHONDONTWRITEBYTECODE=1 \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src:scripts \
  "$CORE_PY" -m pytest -q tests/test_vln_viewpoint.py \
  tests/test_vln_viewpoint_linkage.py \
  tests/test_vln_interactive.py::test_guided_one_answer_then_wait_then_save \
  tests/test_vln_interactive.py::test_review_dialog_requires_confirmation_and_keeps_comment_verbatim \
  tests/test_vln_interactive.py::test_target_change_keeps_late_result_with_original_trial \
  tests/test_vln_interactive.py::test_end_cancels_advisory_only_and_late_result_is_superseded \
  tests/test_vln_interactive.py::test_stale_label_and_point_bound_to_its_own_image \
  tests/test_vln_interactive.py::test_opos_zero_is_explicit_not_visible \
  tests/test_vln_interactive.py::test_template_expansion_is_documented_and_saved_before_inference
```

Final focused verification: **90 passed, 0 failed, 0 skipped in 2.99 s** on the
existing Python 3.12.11 core environment: 83 evaluator/linkage tests and seven existing
mocked interactive-contract tests. Ruff and `git diff --check` pass. The synthetic
report's eight image instances and ten local file links resolve; browser visual
appearance was not independently verified.
Tests cover delayed pre-turn arrivals, exact selected-image provenance, post-turn and
mixed histories, wrong marking persisting, mismatched/cancelled/superseded responses,
unknown clocks/age, documented mapping, OPOS-0, new and legacy review scopes,
conflicts, missing/deleted artifacts, changed archives, report escaping, and the
no-model/no-hardware import boundary. Full regression and heavyweight checks were
deliberately not run during Claude's live-use window.

The six completed stationary `live_08` trials passed archive/selected-image/guided
schema checks (six predictions and six reviews); all 128 consumed files were
unchanged. This is **schema compatibility only**, not a manual-turn experiment or
a navigation result. The four subsequently completed real pairs are now analyzed;
see [VLN_PAIRED_VIEW_RESULTS](VLN_PAIRED_VIEW_RESULTS.md) for the supported findings
and unresolved exposure-time qualifications. No private images or comments are
included in this branch.

Implementation: [reader](../src/go2_perception/vln/viewpoint.py),
[static report](../src/go2_perception/vln/viewpoint_report.py),
[CLI](../scripts/analyze_vln_viewpoint_trial.py),
[focused tests](../tests/test_vln_viewpoint.py). `Archive` + `inspect_trial` also
permit schema-only checks of individual completed trials without inventing a pair.
