# Completed paired-view integration and analysis

Development branch: `feature/vln-viewpoint-evaluation`. No merge into main, changes
to Claude's worktree, model/configuration changes, new inference, hardware connection,
SDK/DDS initialization, calibration session, or repeated collection.

## Exact source versions

Remote fetch confirmed these publications at integration start:

| Component | Commit |
| --- | --- |
| Evaluator starting publication | `77400cfe56f1ac20379f41d10bfd05db7f8f97b3` |
| Interactive branch / linker code inspected | `fdb5451e25cdc1f5971bbf0e1c7887ec51ff634a` |
| Recorder commit saved by `live_09/run.json` | `5057c748a0804635042b28ce3bbbdf51dc68726d` |
| Recorder commit saved by `live_10/run.json` | `5f38424988a2935fd7f925899014f65d489a00cf` |

The recordings predate the final linker publication. The analysis selects the
explicitly declared **method version 2** in each archive's
`evaluation/paired_views.jsonl`, preserving all earlier records and their source
file/line/hash. Four trial pairs are four examples, not seven linkage records.
Copied prediction/review summaries are references only; the evaluator validates the
original trial, request, prediction, selected images and human reviews. Actual local
IDs, paths, commands, hashes and detailed reports stay in Git-ignored artifacts.

## Results from the original human reviews

| Run / target | Before: model / human judgment | After: model / human judgment | Supported reviewed-image outcome | Post-turn receipt margin |
| --- | --- | --- | --- | --- |
| `live_09` / chair | OPOS-0 / target **is visible**, disagree | OPOS-0 / target **is visible**, disagree | Persistent false absence in both reviewed images | 28.066 s |
| `live_10` / oven | OPOS-0 / target absent, agree | Target point / correct | Absent→visible and correctly marked | 43.013 s |
| `live_10` / microwave oven | OPOS-0 / target absent, agree | Target point / correct | Absent→visible and correctly marked | 33.243 s |
| `live_10` / dish washer | Target point / target absent | Target point / correct | Absent→visible and correctly marked | 135.547 s |

The dish-washer before-feedback is a visibility judgment, not a newly invented
marking label. Comments are retained only in private reports. These are two
independent queries per pair, with separate instruction revisions/history. They are
correlated observations in one room, not independent held-out trials. No accuracy
gain, general success rate, autonomous search, navigation success or executed path
is claimed. Guided feedback leaves **path quality, stopping and navigation unknown**.
The first chair proposal was STALE at receipt; its exact reviewed image remains
valid evidence of grounding/visibility. Other proposals were CURRENT at receipt.
Receipt status never substitutes for image age at review.

## Timing and continuity qualifications

All four pairs support **receipt-time ordering** relative to the archived method-v2
odometry annotation. Margins above were recomputed from the original snapshots, not
accepted from copied summaries. v2 uses the union of the deviation-based and
rate-based turn intervals for the ordering check. The raw odometry yaw estimator
was not rerun; this integration imports the existing annotation and audits its
references, interval arithmetic, recorded index/counter metadata and clock epochs.

All four remain **incomplete for guaranteed camera-exposure timing**. The linker's
`verified` flag means receipt ordering. Camera latency, odometry transport, threshold
and sampling error do not have guaranteed bounds. No bound is invented, and the
sensor contract's one-sigma uncertainty is never treated as a hard bound. This
limitation does not erase the separately supported comparison of reviewed images.

There was one camera epoch/source per run, one robot epoch per run, no reported
camera epoch changes, delivery sequence/receipt gaps or inbox overflow, no indexed
dropped LiDAR chunks, and no odometry sequence discontinuities in the index.
All indexed chunk files were present. Raw clouds/odometry arrays were not rehashed
or reprocessed by this lightweight analysis.

The **closing** transport counters must be distinguished from delivery continuity:
`live_10` reports 38 RTP packets lost and two incomplete access units at closure,
whereas its periodic reports through the last status still report zero for both.
That last periodic status is after all paired inputs. The report preserves both
sets of counters; it does not repeat a blanket zero-transport-loss claim or diagnose
the cause of the closing-counter difference. `live_09` closes with zero for both.

One dish-washer before-frame reached the sensor receiver **1.185 ms before Ask**
but was admitted to the new instruction history **15.379 ms after Ask**. The
evaluator now distinguishes the recorded application-admission timestamp from
transport receipt, checks admission after reset/before submission, and retains the
earlier sensor timestamp plus a warning. This fixes a clock-semantics assumption;
it does not move the frame in time or grant a stronger turn-verification label.

Both runs have final summaries and all eight trials have closure records. The tested
linkage/event files had no open holders in the local check. All **2,796** source-file
size/mtime entries were unchanged (655 + 2,141). The **154 present consumed evidence
files** were rehashed; 24 expected-absent entries stayed absent. The source archives
were not modified. This is evidence of stability during analysis, not a filesystem lock.

## Implementation and verification

- Required `source_frame` fields are validated individually. Supported optional
  `received_monotonic_s` and `exposure_monotonic_s` must match their snapshot source,
  including null/unknown values. Additional fields remain preserved.
- The explicit JSONL importer selects pair + method version, retains prior records,
  rejects ambiguous same-method duplicates unless a line is specified, and never
  promotes receipt `verified` into an exposure bound.
- The new `outcome.reviewed_image_outcome` remains independent of timing status and
  includes persistent false absence. Existing summary fields remain available.
- A static combined report rejects duplicate archive/trial pairs and preserves links
  to all four private reports. No aggregate success rate is calculated.

Final focused verification on the unchanged Python 3.12.11 core environment:
**90 passed, 0 failed, 0 skipped in 2.99 s** (83 evaluator/linkage tests plus seven
existing mocked interactive contracts). Ruff and diff checks pass. Full/heavy,
model and hardware suites were not selected; physical opt-ins were disabled. Four
real HTML/JSON/timeline reports and a combined HTML/JSON/Markdown comparison were
produced using saved evidence only. This is not synthetic substitution for real data.
All 48 image instances and 61 local file links across the four pair reports and
combined page resolve. Browser visual appearance was not independently verified.

Use the [evaluator guide](VLN_VIEWPOINT_EVALUATION.md) for CLI syntax. Private
artifacts are under `data/outputs/vln_viewpoint/paired_integration_01/`: `verified/`
contains four pair directories and `comparison/`; `reproduce.sh` contains the exact
resolved commands and accepts a fresh output root. `commands-final.json` records the
commands actually executed. Source manifests, file inventories and test logs are
kept alongside them. Do not publish these private artifacts.

This milestone is complete. Remaining limitations are evidence qualifications,
not requests for more collection, calibration, model inference or robot execution.
