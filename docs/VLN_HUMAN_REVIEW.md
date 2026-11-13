# Phase 6B human review of Phase 6A grounding panels

Branch `phase6-vln-episode-validation` (base `a5de582`, episode contract `5c7f586`, Phase 6A
`c13f1dc`). This file holds aggregate results only. Panels, images and per-case JSON stay
private and Git-ignored.

## Evidence preservation

Phase 6A outputs lived only in `/private/tmp/go2-phase6/data/outputs/phase6/` (at risk of
cleanup). On 2026-11-12 Claude copied 99 of 172 files (38.6 MB) into the ignored directory
`data/outputs/phase6b/phase6a_evidence/` of the main checkout. `PROVENANCE.json` records the
source path, SHA-256 and size of all 172 source files, and each copy was re-hashed. These
were copied: all 12 `demo.png`/`proposal.json`/`source.json`/`preprocessing.json`/
`raw_model_output.npz`; original and rectified images for the six reviewed cases; run
metadata, logs and `semantic_review.json`. These were hashed but not copied, because they
can be regenerated from the recordings and the pinned model: observation NPZs, model-input
previews and masks. The public model cache was neither hashed nor copied. The originals were
not moved or deleted.

## Review protocol

- One panel at a time, shown locally to the user (repository owner, git user `YushanZzzz`)
  in chat on 2026-11-12. The reviewer was asked about target identity, presence and the
  correct region.
- Human results are in `phase6a_evidence/human_review.json`, alongside Codex's unmodified
  `evaluation/semantic_review.json` (its hash is recorded). An item becomes `reviewed` only
  after the user answers. The user's words are quoted in each item's notes.

## Results

| Case | Codex (agent) | Human: target present | Human: box | Notes |
|---|---|---|---|---|
| s_fridge | acceptable | yes | correct | |
| s_dog | failed | no | incorrect | Boxed region is a shoe shelf. |
| m_fridge | failed | not stated | incorrect | Boxed region is the main (entrance) door. Fridge visibility at the right edge was not confirmed. |
| m_bicycle | failed | no | incorrect | Absent-target false positive. |
| m_sofa | failed | no | incorrect | Absent-target false positive. |
| m_shoes | failed | yes | incorrect (miss) | User: the shoes are not marked. Recorded as a visible-target miss; presence comes from that reply. |

The human review agrees with every one of Codex's six judgments: 1 correct, 5 incorrect
(3 absent-target false positives, 1 wrong region, 1 miss). The other six 6A cases
(`s_fridge_paraphrase`, `s_racket`, `s_shoes`, `s_bicycle`, `m_door`, `m_door_paraphrase`)
are **not human-reviewed**.

These results come from 6 hand-picked cases drawn from two correlated viewpoints of one home.
They are not a held-out accuracy estimate.

## Phase 6B episode decision labels

The same reviewer labelled all 11 decisions of the 4 private `test_c` pilot episodes on
2026-11-13. Each package's `evaluation_only/human_review_log.json` records the reviewer's
answers. Aggregate counts and the protocol are in
[VLN_EPISODE_VALIDATION](VLN_EPISODE_VALIDATION.md#human-labels-2026-11-13).

Claude drafted each label and the reviewer accepted it. These labels therefore reflect
one person's judgment and are anchored on Claude's drafts. Only the reviewer can resolve
the 7 ambiguous decisions. No second reviewer was used, so inter-annotator agreement is not
measured.
