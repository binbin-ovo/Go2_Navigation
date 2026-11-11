# Scene representation interface — go2-scene-v1

First Phase 5A delivery for Claude, 2026-11-10. This contract and the committed synthetic JSONL fixture are independent of the perception implementation. No direct Claude messaging channel was found; publication on `phase5-stable-lidar-perception` is a handoff, not a claim that Claude has acknowledged agreement. Additive optional fields are allowed; semantic/breaking changes require a new schema version.

## Transport and frames

One self-contained JSON object per line. `schema="go2-scene-v1"`, `timestamp` finite seconds, `clock_domain` explicit (real recordings: `go2_robot`; fixture: `synthetic`), `frame_id="odom"`, `units="m"`. Axes right-handed, x forward/y left/z up at odometry origin. No global map/geographic accuracy claim. `epoch` increments when history is invalidated; never join geometry or IDs across epochs. Camera coordinates are absent; calibration is unverified.

`robot_pose` is null if no valid pose is available, otherwise `{timestamp, parent_frame, child_frame, T_odom_base}` at the snapshot time. The 4×4 row-major matrix maps base_link column vectors to odom. Points already in odom are not transformed again. A missing-pose snapshot must contain no observed object claim from that cloud and report the reason in diagnostics.

## Background

`background`: `{voxel_size, free_space_policy:"unobserved_is_unknown", cells:[...]}`. Cells carry `center:[x,y,z]`, `observation_count` (distinct scan observations, not point count), `last_observed`, `age_s`, and `evidence_score` in [0,1]. Evidence score is `min(observation_count/min_observations,1) * max(0,1-age_s/ttl_s)`; it is a bounded persistence/recency score, **not a calibrated occupancy probability**. Snapshot diagnostics specify min_observations/ttl_s. Expiration means unknown, not verified free space. Cell size is meters; walls/furniture need not become compact objects.

## Objects

`objects` contain:
- `object_id`: scene-local continuity handle; `track_id`: optional existing tracker ID or null. IDs are metadata, not semantic identities.
- `center`, positive `dimensions`: filtered axis-aligned 3D box in odom meters. `orientation_xyzw`: null in v1 implementation; do not infer reliable heading from an AABB.
- `velocity`: [vx,vy,vz] m/s; `velocity_covariance`: symmetric PSD 3×3 (m/s)². Implemented estimator: observation-supported proposal-center temporal regression with residual-derived uncertainty plus a configured measurement-noise floor. This is model-based uncertainty, not empirically calibrated confidence.
- `motion_state`: STATIONARY, MOVING or UNCERTAIN. Missing evidence must remain UNCERTAIN.
- `status`: `observed_filtered` or `stale`. `observed_box` is `{center, dimensions, point_count}` for the current directly supported proposal, otherwise null. Runtime observed bounds use current raw support; filtered bounds use accumulated proposals. Filtered and observed geometry are deliberately separate. No extrapolated boxes are represented as current measurements.
- `last_observed`, `observation_age_s`, `age_s`: seconds. `observation_score` is the same explicitly configured persistence/recency score as background, not semantic confidence.
- `display_compact`: suggestion to replace redundant raw-point drawing with this box. It NEVER authorizes dropping points from sensing, proposal generation or updates. Conflicting/new objects inside old boxes must remain observable.

`diagnostics` is an extensible object: synthetic flag, reset/rejection reason, accumulation spans/counts, timestamp/deskew assumption, proposal statistics, processing milliseconds and evidence-score parameters. Unknown fields should be ignored by renderers. A renderer can show observed boxes solid, filtered centers distinct, stale boxes faded/dashed, and unknown motion gray. Do not silently hide stale/uncertain status.

## Ready-to-use deterministic fixture

`tests/fixtures/scene/renderer.jsonl`: six frames with a static box, moving object, changing robot pose, static voxel geometry and a partial occlusion that makes the box stale. All geometry is synthetic. Load as plain JSON; no PyTorch, Open3D, SDK, recording or model checkpoint required. Optional Python validation: `scene.serialization.read_scenes` (NumPy only).

```bash
PYTHONPATH=src python scripts/generate_scene_fixture.py --output /tmp/go2-renderer-fixture.jsonl
```

Do not overwrite existing fixtures silently. This schema is not a ROS topic and does not imply a live Go2 connection. Claude can build the renderer now; perception implementation and measured diagnostic comparisons are available on the same branch (see STABLE_LIDAR_EXPERIMENTS.md). Runtime annotations are prohibited; GT is evaluation-only.

## Machine schema and minimal consumer

The normative structural schema is [`schemas/go2-scene-v1.schema.json`](../schemas/go2-scene-v1.schema.json), JSON Schema draft 2020-12. Semantic validation (finite values, rigid transforms, PSD covariance, age consistency and duplicate IDs) is additionally performed by `scene.serialization.validate_scene`. JSON Schema's numeric type alone does not reject a nonstandard parser's NaN; consumers must reject nonfinite values.

A renderer can begin with the **standard-library-only** example:

```bash
python scripts/read_scene.py tests/fixtures/scene/renderer.jsonl
```

It prints six summaries; the last two frames contain one directly supported filtered box and one stale box. Draw scene points/boxes directly in odom. Apply `T_odom_base` only to the robot model; applying it to scene points again would double-transform them. Treat `display_compact` as rendering guidance only. The source measurements must still reach perception unchanged. Optional new fields may be ignored; changes to existing fields, units, semantics or required structure require coordination with Claude and a new schema version.

Runtime covariance starts as identity (m/s)^2 until regression has enough observations; it is a heuristic prior, not measured certainty. Runtime score parameters are `background_min_observations`, `background_ttl_s`, `object_min_observations` and `object_ttl_s` in diagnostics. The synthetic fixture uses its documented common `min_observations`/`ttl_s` values. These diagnostic additions do not change required v1 fields.

## Phase 5A.1 optional evidence fields

The required v1 schema and original fixture remain unchanged. Recovery producers add
`evidence_state`, `recent_support_horizon_s`, `last_direct_scan_timestamp`, `sector_evidence`
and timestamped `motion_evidence`. See [DETECTION_RECOVERY_ARCHITECTURE](DETECTION_RECOVERY_ARCHITECTURE.md)
for exact meanings and [the optional companion schema](../schemas/go2-scene-v1-recovery.schema.json)
for field types (validate alongside v1). Old consumers may ignore these fields. New consumers
must not promote `observed_recently` to a current observation or infer free space from no returns.
The deterministic `tests/fixtures/scene/recovery.jsonl` supplements the original renderer fixture.
