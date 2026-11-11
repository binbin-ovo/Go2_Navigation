# Navigation scene readiness — Phase 5A.1

Long-term objective: language-guided navigation with dynamic scene understanding and
predictive collision avoidance. This branch implements **no navigation policy, velocity
publisher, locomotion command, language grounding or autonomous control**.

| Future planner need | Current evidence | Remaining gap |
|---|---|---|
| Robot pose/time | Same-clock odom pose, epoch reset, null on missing pose | Drift and live latency unmeasured |
| Static occupied regions | Expiring voxel returns, explicitly defined recency score | Not a calibrated occupancy probability or full static map |
| Current obstacle support | Current return plus bounded recent support, actual observed box | Sparse samples can support incomplete/wrongly merged geometry |
| Temporarily unobserved obstacles | v1 stale geometry, direct age, optional recent evidence and sector-return state | No verified firing/visibility model; do not infer disappearance |
| Velocity and uncertainty | Direct-center regression, timestamped optional motion evidence, covariance | Residual covariance not calibrated; semantic identity ambiguous |
| Scan coverage | Positive angular return overlap | No-return directions may be unsampled, occluded or missing |
| Known free space | **Not provided** | Requires measured rays, sensor origin/timing and validated visibility integration |
| Unknown/occluded space | Missing cells/objects and expired evidence remain unknown | Cannot distinguish all occlusions from sampling gaps |
| Semantic destinations/classes | None from anonymous LiDAR boxes | Requires localization, semantic mapping and language grounding |
| Forecasting | Existing public-data forecasting modules preserved | Recovery boxes are not trustworthy pedestrian training labels |

`viewer.planning.query` remains usable: occupied from current/stale box or background cell,
otherwise unknown. It never returns free. A future 2D footprint projection must preserve
that distinction; absence of a polygon is not collision-free space. No new fabricated
free-space layer is added.

Consumers using `motion_evidence` must check `valid`, `updated_at`, same clock/epoch and age.
The proposed recent bound is 0.2 s, not permission to call a stale observation fresh. Old
renderers safely ignore the new fields; they continue to show gray stale boxes. Do not
convert the evidence score to a collision probability or covariance to a guaranteed safety
margin. No navigation readiness or safety certification follows from passing unit tests.

Before planning: human-review annotations, evaluate the frozen candidate on held-out H1–H3,
verify raw point-frame semantics and robot exclusion geometry, measure live latency/drift,
calibrate uncertainty, add validated visibility/free-space evidence, and separately design
control authorization and safe-stop behavior. Claude retains hardware validation, renderer
integration and held-out recording ownership.
