# Phase 5A.1 detection recovery architecture

Opt-in branch `phase5-detection-recovery`, based on verified Phase 5B `89d901d`.
`RecoveryScene` and `RecoveryObjects` are separate implementations under `scene/`; the
existing `StableScene`, original tracker, hardware adapters, forecasting and 3D viewer remain
available. No live default, schema version or robot-control behavior changes.

## Observation semantics

The candidate uses a 0.3 s proposal window and a 0.2 s recent-support window (approximately
three incremental messages). It requires **at least one genuinely new current-scan return**
and three recent nonground support points inside a compact proposal. A retained old window
without any new support never refreshes v1 `last_observed` or `observed_box`.

This replaces the old three-points-in-the-latest-message rule. A single new point supports
existence, not a reliable new center/velocity sample. Such a point can refresh `observed_box`
(minimum dimension remains 0.03 m), but direct motion regression requires at least three
current returns. Filtered shape remains derived from the proposal window. Users must not
interpret a one-point observed box as complete physical dimensions.

v1 fields keep their meaning: `observed_filtered` has current support; no new measurement
means `stale`, `observed_box=null`, and `motion_state=UNCERTAIN`. The unchanged viewer therefore
remains conservative. The following **optional additive object fields** expose more detail:

| Field | Meaning |
|---|---|
| `evidence_state` | `observed_current`, `observed_recently` (last direct support ≤0.2 s), or `stale_unknown` |
| `recent_support_horizon_s` | Explicit recent-evidence age bound |
| `last_direct_scan_timestamp` | Supporting scan end timestamp, equal to v1 `last_observed`; not an invented current observation |
| `sector_evidence` | Angular return-cone overlap classification, return count, `proves_visible=false` |
| `motion_evidence` | State at `updated_at`, valid flag, observation count, reason; when fitted, span, coherence, displacement and speed |

A consumer must check the **motion evidence timestamp**, not just the object's timestamp.
Recent valid motion may be used as a bounded-age hypothesis, never a newly measured velocity.
After 0.6 s without support geometry expires from the output; history may persist internally
for up to 1 s solely for guarded re-association. Expiry is unknown, never free.

`sector_evidence.state` is one of `no_sector_returns`,
`foreground_returns_occlusion_possible`, or `sector_returns_without_object`. It compares
actual return directions/ranges with the object's angular bounding cone in the current body
frame. No returns do **not** prove an unsampled sector: unavailable firing schedules, dropouts,
reflectivity and occlusion remain possible. Foreground returns suggest possible occlusion,
not verified occlusion. Background returns do not certify absence of every point on an
object. This avoids over-interpreting the Phase 5B diagnostic evidence as calibrated visibility.
No angular free-space ray map is invented. `diagnostics.scan_coverage` explicitly records
return-only evidence and uncalibrated visibility.

## Motion and continuity

The direct-center history contains only genuinely supported current observations with at
least three points. Median pairwise slopes over time separations ≥0.2 s estimate velocity;
a 1 s history needs four observations over 0.35 s. Residual-derived diagonal covariance uses
a 0.025 m floor; it is heuristic, not statistically calibrated, and is not used as a hard
`speed - 2σ` gate. Coherence is clipped `1 - residual_XY_energy / centered_XY_energy`.

MOVING requires speed ≥0.28 m/s (0.18 release), fitted displacement ≥0.18 m and coherence
≥0.55, over two supporting observations. STATIONARY requires speed <0.12 m/s and position
spread <0.08 m. Otherwise state remains UNCERTAIN. Primary motion thresholds and history
lengths are configurable in `configs/detection_recovery.yaml`; fixed floor/stationary and
ambiguity constants are explicit in `recovery_objects.py`. New optional direct settings
are independent of inherited Phase 5A settings, retained for baseline geometry behavior.
Direction coherence suppresses alternating centroid jumps but cannot guarantee immunity to
consistent biased sampling. Long histories can delay stopping/restarting; evaluate both.

Hungarian matching still uses geometry. Recovery requires full 3D size ratios <2.5 and
vertical-center difference <0.4 m, avoiding the old XY-only person/low-box compatibility.
Gaps >0.2 s require distance within 0.3 m of the association estimate (ordinary gate 0.6 m).
Near-tied proposals/objects (cost margin <0.15) discard direct motion history and reset
activation. Unambiguous brief gaps preserve history; after 1 s it is discarded. This is
not semantic identity tracking. Ambiguous crossing occupants may merge or receive new IDs;
occupancy remains conservative instead of declaring a trustworthy continuous trajectory.
No existing Phase 3 tracker is rewritten, and this history is not forecasting supervision.

## Self-filter and geometry

The experimental exclusion volume is a deliberately small **inner body proxy** in raw
`base_link`: x=[−0.25,0.25], y=[−0.12,0.12], z=[−0.20,0.12] m. It is configurable, not a
verified full Go2 hull. It avoids a broad radius crop that would delete close external
obstacles. Points are masked in a copy before odom accumulation; input arrays and recordings
remain unchanged. Original per-point timestamps and scan-end pose timing are retained.
`self_excluded_points` reports the per-frame count. Both proposal/map inputs omit these points.

Rotation/translation and a synthetic crouched/pitched pose are tested. A fixed body-frame
box does not model moving legs, floor reflections or every crouched stance; floor-height
self artifacts can remain outside it, and real objects intruding inside it would be removed.
Do not enlarge it to cover all nearby clutter without measured robot geometry and a
near-obstacle validation set. The renderer's robot mesh is not calibration evidence.

DBSCAN eps remains 0.3 m, min_points 8; extended plane rejection and background-aware novel
component splitting are preserved. This experiment tests added sampling support rather
than globally shrinking eps. Wall/legs and two-person crossing synthetic cases measure
remaining merge/split behavior. No annotation geometry enters proposals. The evaluation-only
oracle diagnostics in Phase 5B remain available unchanged.

## Runtime and compatibility

`FastBackgroundMap.stable_mask` quantizes in NumPy, evaluates eligible map keys once, and
uses native-integer tuple membership. It preserves reference masks exactly in regression
tests. A sorted structured-array approach was tested and rejected as slower. Real profiling
found per-cell scalar `np.isclose` validation to be a larger cost than `stable_mask`; it is
replaced by the equivalent finite-scalar tolerance expression, with boundary regression
checks. This shared validator optimization does not change serialized output semantics.

Use `scripts/run_detection_recovery.py --input ... --output ... --window 0.3`.
`--baseline` selects unchanged Phase 5A with the specified window; `--window 0.1` gives the
controlled short-window comparison. Full raw geometry outputs must remain private/ignored.
The CLI records compute time, load/process/JSON-write wall time, RSS, age distribution and
config. It is offline; no end-to-end live sensor latency is implied.

See DETECTION_RECOVERY_EXPERIMENTS.md for measured results and the candidate freeze manifest.
The deterministic recovery fixture supplements, and does not replace, the original six-row
renderer fixture. `viewer.scene.parse_row` and schema validation accept the additive fields;
old renderers ignore them and continue to show stale geometry gray/dashed.
