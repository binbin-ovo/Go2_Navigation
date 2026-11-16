# Bounded visual context — human review of saved outputs (2026-11-16)

Review of Codex's [bounded-context experiment](VLN_BOUNDED_CONTEXT_EXPERIMENT.md)
(`feature/vln-bounded-context` at `d3c227a112436d1b93c2d9b67bb69d5b89288295`),
candidate `anchor2_recent6_actual_time_v1`. Review-only: **no new inference, no robot
connection, no model-setting change, no rebuilt episodes, no candidate tuning.** The
repository owner reviewed four existing full/bounded pairs, one at a time, in a Claude
session. Answers are stored in a new private, Git-ignored review file with verbatim
wording and timestamps. Original annotations, model outputs and scores are unchanged.

Panels were composed offline from saved outputs by
`scripts/compose_vln_bounded_review_panels.py`. Each panel showed the instruction, the
current model-input image with native OPOS/APOS, both conditions' selected history
(actual relative time and pooling factor), all ten native waypoints and offline timings.
The existing human reference was shown in a separate evaluation-only box. Images stay
private.

## What the reviewer confirmed

| Pair | Reported difference | Reviewer answer | Status |
|---|---|---|---|
| E2 d2 "Turn left until you face the shoe rack" | Full points at the visible rack (OPOS501); bounded emits OPOS0 (target not visible) | Rack identifiable; **full target correct, bounded wrong** | **Confirmed regression** (target grounding) |
| E1 d2 "Turn right toward the kitchen counter" | Both miss STOP; bounded adds an initial −6.1° right turn and a more rightward path | Bounded turn/path does **not** suit the instruction; **should stop** | **Confirmed regression** (full also misses STOP; full not judged separately) |
| E3 d2 "Walk toward the water bottles…" | Full ~0.7 m forward; bounded ~15° right turn with ~0.12 m advance | Judged on complete paths: **full acceptable, bounded not** | **Confirmed regression**, not a mapping artifact |
| Saved 39-frame live cutoff "Approach the doorway…" | Full straight ~1.51 m, OPOS504; bounded ~1.45 m forward /0.39 m left, OPOS501 | **Bounded more plausible** than full | Comparative only; does not favour full |

## Ambiguous or mapping-dependent

- **E2 d2 path.** The reviewer also noted that bounded's waypoints move closer to the
  rack while full's stay nearly in place. The earlier human reference says the turn was
  complete and STOP was the only acceptable action, and the instruction asks to *face*
  the rack, not approach it. This path preference is recorded as an observation. It
  does not relabel the reference, and it does not offset the confirmed target loss.
- **Live cutoff.** The reviewer gave a comparison only: target point and path were not
  judged separately, and neither condition was called implausible on its own. There is
  no per-request human label. The two conditions also had different history (38 unique
  images vs 2 anchors + 6 recent) and timing (full 8.04 s, the only cold-after-load
  request, vs 2.16 s). Both timings are offline.
- **Discrete-mapping counts.** The drop in supported mappings (7/9→4/9) and part of the
  agreement drop (3/9→2/9) come from the first-waypoint mapping, which cannot represent
  mixed turn/translation. For E3 d2 the reviewer judged the complete paths and still
  rejected bounded, so that case stands without relying on the mapping.
- **Not reviewed:** E1 d0/d1, E2 d0/d1, E3 d0/d1, E4 and live r1/r2. Their Codex
  diagnostics remain un-reviewed.

## Decision

**Keep the candidate experimental.** It cuts median offline delivery by 51% (19
frames) to 73% (39 frames). However, on three of the four pairs the reviewer confirmed it got
worse in the case studied (target grounding in E2 d2; path and stopping in E1 d2 and
E3 d2). The one favourable comparison (live cutoff) has no ground truth and is a
single scene. Do not promote it to the live advisory or a default, and do not use it
for control. The full-history policy stays operational.

## Question a follow-up must isolate

The candidate changed three things together: **image count/selection** (2 anchors + 6
recent), **timestamp encoding** (actual millisecond ticks instead of the native 4 Hz
grid) and **pooling tier** (anchors at ×4 instead of ×2). E3 d0 already gives identical
8 images to both conditions and still gets different outputs, so image count alone
cannot explain every difference.

> With the same bounded 8-image selection, does restoring the native timestamp
> encoding and native anchor pooling recover the E2 d2 target (OPOS501) and the
> E1 d2 / E3 d2 paths? Or does dropping the middle history cause the regression?

A follow-up would be a separately declared experiment with a minimal ablation, for
example:
- bounded selection + native timestamps
- bounded selection + native pooling
- full history + actual-ms timestamps

It would be scored by human review of the same pairs. Nothing here authorizes running
it. It needs its own declaration and resource coordination.
