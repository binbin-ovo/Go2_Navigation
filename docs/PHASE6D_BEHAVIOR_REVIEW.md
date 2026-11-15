# Phase 6D: human review of LightNav grounding, waypoints and stopping

Branch `phase6-vln-behavior-review`, based on the Phase 6C publication `45154f6`
(implementation `0a442f4`). Review date: 2026-11-15. Claude prepared the panels, and the
repository owner (git user `YushanZzzz`) reviewed them in chat, one question at a time.

**What was done:** saved Phase 6C traces were re-plotted read-only. Nothing else was run or
changed:
- no inference, simulation or calibration was rerun
- no model parameters, controller files or Phase 6B/6C artifacts were changed
- no episodes were rebuilt or relabelled

The only material in Git is aggregate findings and tooling. Panels, model input images,
trajectories and the per-question review files stay local and Git-ignored.

## Sources (read in place)

| Source | Used for |
|---|---|
| `phase6c/sim_closed_loop_v1/steps.json`, `summary.json` | per-step pose, native output, 10-waypoint chunk, executed action, stop flags, evaluator state |
| `phase6c/sim_closed_loop_v1/inference/NNN/input_*.png` | the exact images the model received at each step |
| `phase6c/claude_pilot_v3/inference/NNN/{result.json,input_*.png}` | LightNav outputs and inputs for the recorded E1–E4 decisions |
| `phase6c/claude_pilot_v3_review_v2/NNN/evaluation_reference.json` | existing Phase 6B human references (displayed only, not modified) |
| `phase6c/lightnav/source-local-v2/src/lightnav/vln_utils.py` | official decoding of the pointing tokens |

Pointing tokens are decoded on the upstream 48×27 grid, which is relative to the image
frame:
- `apos`: 1297, 1298 and 1299 are the rotate-left, rotate-right and stop directives; 0 means
  no label.
- `opos`: 0 means target not visible.
- Edge cells may be clamped from points outside the frame.

Simulator waypoints are converted to world coordinates using only the pose before each step,
with no other registration. Recorded-route waypoints are plotted in the native ego frame and
are unregistered to Go2.

Private outputs, all under `data/outputs/phase6d/`:
- `panels/`
- `human_review/sim_rollout_v1_review.json`
- `human_review/lightnav_recorded_review.json`
- `MANIFEST.json`, holding SHA-256 hashes of the review files

Each review entry records:
- the source run and step or decision
- hashes of the artifacts it refers to
- the exact question, and the reviewer's verbatim answer
- the reviewer and timestamp
- what remains uncertain

`agent_review` entries are Claude's interpretation and are kept separate from human answers.

`data/outputs/phase6d/waypoint_validation/` belongs to another agent. This review did not
read or modify it.

Tooling, all in `scripts/`:
- `phase6d_sim_review_panels.py`
- `phase6d_recorded_review_panels.py`
- `phase6d_record_review.py`

## A. Failed static rollout (`sim_closed_loop_v1`)

Instruction: "Turn right and approach the doorway, then stop before the doorway." The robot
starts at (4.30, 0.00) with a heading of +25°. The evaluator-only doorway centre is 25° to its
right, and the robot is 1.850 m from the centre of the stopping region.

| Step | Human-reviewed visual finding | Model output (trace) | Executed (trace) |
|---|---|---|---|
| 0 | Doorway **recognizable**, on the **right** (agrees with evaluator geometry) | apos 1298 rotate-right; **opos 0 (not visible)**; chunk turns −90° in place, then 0.9 m along the new heading | waypoint 1: −18.0°, 0.5 mm |
| 1–8 | (step 2) Doorway **roughly straight ahead** | **Identical output at every step**: apos 1296 (bottom-right edge cell), opos 0, chunk ending at (+0.53 m, −0.91 m, −60°) | −18.0° per step; 0.1–0.5 mm translation |
| 9 | Doorway **not visible** (passed between steps 1 and 2; now about 137° behind) | apos 1273 (bottom-centre), opos 0, chunk 1.51 m straight ahead | +151 mm |
| 10 | Model's only target point (opos 313, col 24 row 6) is **not on the doorway** (blank wall/corner) | straight chunk | +151 mm |
| 15 | Final view **does not satisfy** "stop before the doorway" | no decoder stop, no stop code; straight chunk | +151 mm; final distance 2.722 m |

Findings:
- **Target recognizable in the simulator:** yes, according to the human reviewer at steps 0
  and 2. A gross visual domain problem (E) is therefore not supported as the primary cause.
  The synthetic-to-real domain gap itself was not assessed.
- **Target interpretation (A):** confirmed failure. The model said "not visible" while the
  doorway was visible on the right (step 0) and roughly ahead (step 2). Its single positive
  target point (step 10) marked a non-doorway location.
- **Native proposal before execution (B):**
  - At step 0, the proposed direction (right) is correct, but the chunk's turn (−90°)
    overshoots a 25° offset.
  - From step 1 on, the chunk is unchanged even though the images change, and it keeps
    turning past the doorway.
  - At step 9 the model commits to moving straight ahead while the doorway is out of view.
- **Proposal vs. execution (C):** no discrepancy was found in the trace. At every step the
  applied action equals waypoint 1 exactly, with no clamping, rejection or contact. The
  overshoot is the consequence of repeatedly executing the first waypoint of an identical
  chunk, −18° nine times (−162° in total). That is consistent with the documented design
  (first waypoint over 0.25 simulated seconds, chunk without a time base). It is **not** a
  proven controller bug.
- **Stopping (D):** the model gave no stop output at any of the 16 steps. It never entered the
  stopping region, and the human-reviewed final view does not meet the stop condition.

## B. Recorded Go2 decisions (`claude_pilot_v3`, offline, not executed)

These are the frames along the operator-driven route. No model action was executed, so the
reference is the existing Phase 6B human labels.

| Decision | Target visible (human) | Pointing | Proposed direction (human) | Stopping |
|---|---|---|---|---|
| E1 d0 (turn right) | visible | **incorrect**: opos 0 although visible (derived) | **appropriate** (curve ≈61° right over ≈1 m) | not expected; none output |
| E1 d2 (terminal) | visible (implied) | **correct**: on the counter (human) | not asked | no STOP output; human: **acceptable / unsure** |
| E2 d0 (turn left) | visible | **incorrect**: opos 0 although visible (derived) | **appropriate** (curve ≈65° left over ≈1 m) | not expected; none output |
| E2 d2 (terminal) | visible (implied) | **correct**: on the shoe rack (human) | not asked | no STOP output; human: **acceptable / unsure** |
| E3 d1 (approach) | visible (implied) | **correct**: on the bottles (human) | **appropriate** (0.68 m straight, then ≈40° right) | not expected; none output |

"Derived" means an `agent_review` entry that follows from the human visibility answer. It
was not asked as a separate question.

Pattern: every pointing error is a **missed visible target** (opos 0). Each time the model
did emit a target point on a recorded frame, it was correct (3/3). The initial turn
directions were appropriate even when the target was reported as not visible.

**Stopping:** LightNav did not emit STOP on any decision. At both terminal cases (E1 d2 and
E2 d2), the reviewer chose "acceptable / unsure" when asked whether not stopping was
inappropriate. The Phase 6B references (STOP; E2 d2 not ambiguous) are unchanged. Whether
LightNav fails to stop on the recordings is therefore **not human-confirmed** in this review.
The two judgments differ in emphasis and are reported as such, not reconciled.

**E4 (absent sofa):** not human-reviewed in Phase 6D. Both decisions emit apos 1298 and
opos 0, which matches the search-turn references. That is not evidence of absent-target
recognition, because the same opos 0 was emitted for clearly visible targets. The historical
mismatch is preserved: the SmolVLM prompt mapped absence to UNRESOLVED, while the references
accept search turns.

## Human-reviewed vs. pending

- **Human-reviewed: 16 answers.** Six are from the rollout (steps 0, 0, 2, 9, 10, 15) and ten
  are from recorded decisions (E1 d0 ×2, E1 d2 ×2, E2 d0 ×2, E2 d2 ×2, E3 d1 ×2).
- **Agent-only: 8 entries.** These are the trace analysis, two derived pointing judgments,
  three non-terminal stopping notes, the step 0 output summary and the E4 note.
- **Not reviewed in 6D:**
  - rollout steps 1, 3–8 and 11–14 (identical outputs to the reviewed neighbouring steps)
  - E1 d1, E2 d1, E3 d0, E3 d2 and E4 d0/d1
  - direction at the terminal decisions
  - realism of the synthetic scene beyond doorway recognizability

## Open questions for code inspection (Codex)

1. Why do rollout steps 1–8 produce byte-identical token sequences and chunks while the
   images change and the history grows? Check whether the image, the SlowFast selection and
   pooling, or a cache actually changes between calls.
2. Is executing waypoint 1 of each re-planned chunk the intended way to use LightNav chunks?
   The upstream intent might instead be a time-scaled chunk, a later waypoint, or executing
   several waypoints per call. With an unchanging chunk, the current scheme accumulates
   −18° per step without bound.
3. Why does opos 0 dominate for visible targets? This happened at the simulator doorway
   (steps 0–9), E1 d0 and E2 d0. Check prompt, resolution, 448×256 preprocessing or rectified
   input, and the history length when the first frames are presented.
4. What decides the switch from turning to straight-ahead at rollout step 9? The doorway
   was not in view.
5. When, if ever, should the decoder or stop code fire? No STOP was emitted across 27 calls
   (11 recorded decisions and 16 simulator steps).

No claim is made of real-world navigation success, collision avoidance, correct metric
grounding on Go2, or a proven controller bug.
