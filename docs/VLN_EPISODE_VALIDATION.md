# Phase 6B private pilot episodes (go2-vln-episode-v1)

Branch `phase6-vln-episode-validation`, based on `a5de582` (Codex `phase6-temporal-vln`,
contract `5c7f586`). These episodes follow [VLN_EPISODE_INTERFACE](VLN_EPISODE_INTERFACE.md)
unchanged; no competing schema. Packages are private and Git-ignored. Only this aggregate
description and the builder `scripts/phase6b_build_episodes.py` are in Git.

## Source inventory (2026-11-12)

Of the 27 local recordings, **only `test_c` contains robot motion**. The operator drove it
with the remote while the robot was supervised. All other recordings are stationary:
`test_a1/a2/b`, the `p5d_*` calibration/snapshot captures and `mac_smoke_01`. They can serve
only as observation-only or absent-target probes, not navigation episodes.

`test_c` odometry contradicts its manifest note about turn order. Measured yaw (+left,
base_link z up), in seconds after the first odometry stamp:
- 7–13 s: **right** ~23°.
- 17–25 s: left to +21°.
- 33–37 s: forward ~0.45 m.
- 37–79 s: small irregular repositioning (yaw −10…−24°, ≤0.4 m).

The camera frames agree with the odometry.

## Episodes

| Episode | Interval (s after first odom) | Instruction (retrospective) | Decisions | Coverage |
|---|---|---|---|---|
| `p6b_test_c_e1_turn_right` | 5.0–12.5 | Turn right toward the kitchen counter. | 3 | turn instruction |
| `p6b_test_c_e2_turn_left` | 15.0–24.5 | Turn left until you face the shoe rack. | 3 | turn toward a visible landmark |
| `p6b_test_c_e3_forward_bottles` | 31.0–37.5 | Walk toward the water bottles in front of the fridge. | 3 | approach a visible landmark (only ~0.45 m; arrival not recorded) |
| `p6b_test_c_e4_absent_sofa` | 2.0–6.5 | Find the sofa. | 2 | absent target, stationary, observation-only |

**Every instruction is retrospective.** Claude wrote each one after viewing the operator's
recorded route, and the user approved the draft. None was given to the operator at
recording time. Each package's `evaluation_only/provenance.json` records this.

**Not covered:** distinguishing similar targets. The only candidate in this footage is
fridge vs dishwasher, which is too weak to label, so no episode was built for it. A new
recording is not proposed yet. A doorway is visible in the background throughout `test_c`,
but the robot never approaches it.

## Human labels (2026-11-13)

All 11 decisions have human labels from the repository owner (git user `YushanZzzz`). The
owner reviewed one causal decision panel at a time in chat. Claude proposed draft labels
from the same panel, and the owner either accepted or corrected them. The owner accepted
every draft. Each package's `evaluation_only/human_review_log.json` records the review time,
the owner's reply and the label. A file is marked `reviewed` only after every one of its
decisions has been answered. Panels show only frames that `select_history` returns
(max 4 frames, 6 s horizon), so no frame after the cutoff and no odometry was shown.

| Episode | Decisions | Target visible / absent | Single acceptable action | Complete / incomplete / unknown | Ambiguous |
|---|---|---|---|---|---|
| e1_turn_right | 3 | 3 / 0 | 2 | 1 / 1 / 1 | 3 |
| e2_turn_left | 3 | 3 / 0 | 2 | 1 / 1 / 1 | 1 |
| e3_forward_bottles | 3 | 3 / 0 | 2 | 0 / 2 / 1 | 1 |
| e4_absent_sofa | 2 | 0 / 2 | 0 | 0 / 2 / 0 | 2 |
| **Total** | **11** | **9 / 2** | **6** | **2 / 6 / 3** | **7** |

No decision was marked `insufficient_information`. Where a decision has more than one
acceptable action, the others are close to completion: either `STOP` or continuing the turn
is acceptable. Exceptions:
- In e3 at d2, `ADVANCE` and `TURN_RIGHT` are both acceptable, so the bottles can be
  re-centred.
- In e4, either search turn is acceptable.

Sources of ambiguity:
- Turn instructions do not say exactly how far to turn ("toward the counter").
- "Walk toward" has no stopping point, and the recording ends before arrival.
- When the target is absent, nothing in view says which way to search.

Distances mentioned in notes are visual estimates. Only the per-package annotation files
hold per-decision labels. They are private and not in Git.

## Package layout

```
data/outputs/phase6b/episodes/<episode_id>/
  model_input/episode.json        # go2-vln-episode-v1 manifest (instruction + frames + decisions)
  model_input/images/NNNNNN.jpg   # raw fisheye RGB bytes copied unmodified from the recording
  model_input/calibration.yaml    # measured go2-camera-calibration-v1 (Phase 5D, held-out 0.47 px)
  evaluation_only/annotations.json   # go2-vln-annotations-v1, bound to the manifest hash
  evaluation_only/provenance.json    # source paths, offsets, latency, odometry at bounds/decisions
  evaluation_only/odom_slice.jsonl   # operator's recorded route (evaluator-only)
  evaluation_only/decision_panels/   # causal labelling panels (select_history at each cutoff)
```

The manifest never references anything in `evaluation_only/`. `robot_state_ref` is null for
every frame, so the odometry, which records the operator's route including future poses,
cannot reach a model through the manifest. The decision panels are built only from
`select_history` output: the 4 frames available within 6 s before each cutoff.

## Timing and calibration

- One recording and one boot per episode. The epoch is
  `go2_boot:host_minus_robot=<offset>`, with the offset estimated as in
  `vln.replay.Go2RecordingReplay` (lower envelope of odometry `host_recv − stamp`).
- `observation_s = host_recv_first − offset − 0.048` (`exposure_estimated`, σ ≈ 16 ms).
  **The 48 ms latency was measured on `test_c` itself**, so it is not an independent check
  for these episodes.
- `available_s = host_recv_last − offset + 0.050 s`. The 0.050 s decode margin is declared,
  not measured. Availability is receipt-based, not hardware-synchronized.
- Intrinsics are the measured Phase 5D fisheye calibration. `T_base_camera` is still
  missing, so there are no metric goals.
- `test_c` has no incomplete frames and no inter-frame interval above 1.5× nominal (max 0.091 s,
  nominal 0.070 s), so the episodes declare no gaps.

## What is not known

- Motor commands are not recorded and are not inferred from video or odometry.
- Whether a model's alternative action would reach the goal or avoid a collision is
  **UNKNOWN** for prerecorded data. Offline output does not change the recorded route.
- Any divergence from the operator's route is not, by itself, an error.
