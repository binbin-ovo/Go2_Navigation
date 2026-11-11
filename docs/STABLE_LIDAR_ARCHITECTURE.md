# Phase 5A stable LiDAR scene architecture

Experimental, opt-in implementation on `phase5-stable-lidar-perception`, based on integrated
`9068088542067fc117535279b412fbf2f808fdf6`. It preserves the existing tracker, forecasting,
RGB/fusion, recording, annotation and visualization modules. No live defaults or robot
commands change. Results do **not** justify promotion to operational perception.

## Data flow and public API

`RecordingReader → TemporalAccumulator → ground/range filtering → compact proposals →
SceneObjects → BackgroundMap → scene-v1 JSONL`. Runtime never loads annotations or RGB.
`scene.serialization` validates every emitted frame. All geometry is in odometry meters;
see [renderer contract](SCENE_REPRESENTATION_INTERFACE.md).

```python
from go2_perception.io.config import load_config
from go2_perception.scene.pipeline import StableScene

pipeline = StableScene(load_config("configs/stable_lidar.yaml"))
# xyz: finite or partially invalid Nx3 raw incremental base_link points.
# lookup(t): rigid T_odom_base at seconds t in the same clock, or None.
row = pipeline.process(xyz, stamp_seconds, unique_scan_id, lookup,
                       clock_domain="go2_robot", point_offsets=offset_seconds)
```

`point_offsets` is optional Nx1 represented as an `(N,)` array, nonnegative seconds since
header timestamp; offsets need not be sorted. Nx4 recording intensity is separated by the
existing loader. Supply XYZ to this API. Explicit `reset=True` invalidates all history.
`deskew=True` additionally requires `instantaneous_point_frame=True`, verified instantaneous
point-frame semantics, per-point times and valid poses at every used acquisition time.
That flag is **not enabled on the real replay CLI**.

## Timing and transformations

The four development recordings contain per-point times spanning about 62 ms per message,
with about 65 ms between messages. `cloud_base` is a thinned incremental observation, not
an entire uniformly sampled sweep. `cloud_deskewed` contains old accumulated points in odom
and must not be fed through this accumulator. Stationary frame evidence does not establish
firmware point-frame semantics during motion.

`scene/accumulation.py` transforms each message once with its midpoint pose by default.
Point timestamps remain exact metadata despite this geometric approximation. With no
point timestamps, all points use the message stamp and age is approximate. Row time and
robot pose use the final acquisition time; acquisition points and robot mesh thus describe
different explicitly stated times. The real `RecordedPoses` adapter interpolates only
bracketed samples within its existing 0.1 s limit; it never silently extrapolates.

`TemporalAccumulator.add` accepts only raw incremental `base_link` input. The lower-level
API rejects preaccumulated or odom clouds, repeated interval IDs/times and invalid transforms.
`snapshot(window_s, voxel_size)` exposes XYZ, acquired times, interval IDs, snapshot time
and ages. Retention supports 0.1/0.3/0.6/1.0 s; voxelization keeps the latest actual point,
not a centroid with an invented time. The 300,000-point limit bounds history. Clock changes,
rewinds, explicit odometry reset, gaps over 0.5 s, pose jumps over 0.5 m/45 degrees, missing
poses and oversized scans clear history and advance `epoch`. Missing pose emits null pose
and empty geometry. Clear renderer trails whenever epoch changes. Smooth odometry drift
is not detectable by these guards and remains a limitation.

## Background and proposals

`scene/background.py` maintains 0.1 m local voxels within 8 m, requiring three distinct scan
hits spanning at least 0.2 s. Overlapping accumulated windows do not multiply evidence.
Cells expire after 2 s; count is lifetime distinct hits while the cell exists, not a sliding
window count. There is no raycasting or free-space evidence: disappeared/expired regions
are **unknown**. Walls and extended furniture remain in this layer even when unsuitable
for compact boxes. Known MOVING boxes mask the map update and remove their previous cells,
but only **after every new point enters proposal/motion processing**. Moving returns can
enter the map before motion is established and leave ghosts until TTL expiration.

`scene/proposals.py` uses existing DBSCAN at eps 0.3 m, min_points 8. It does not globally
reduce eps. A deterministic vertical-plane RANSAC removes extended planar surfaces from
compact proposals, not from raw sensor input or the background map. Components wider
than 1.5 m in XY or taller than 2.4 m are not compact boxes. Novel voxels relative to stable
background may split an extended component. Bounds are 2–98 percentiles, minimum 0.03 m.
This is geometric segmentation, not semantic classification. Adjacent people/furniture
remain difficult; rejecting extended components can turn a merge into a miss.

Ground uses the existing constrained RANSAC with odom z expected at robot z minus 0.32 m,
0.06 m tolerance and 100 iterations. That assumes a locally level floor and known mount
height; stairs/slopes and arbitrary support planes are not validated. Parameters live in
[stable_lidar.yaml](../configs/stable_lidar.yaml); the original profile is untouched.

## Object geometry and motion

`scene/objects.py` associates proposals using Hungarian XY distance (0.6 m gate) and a
size-ratio check. IDs are scene continuity metadata, with no claim of pedestrian identity.
No existing Kalman tracker is rewritten. Position EMA alpha is 0.35 (0.7 once moving),
dimension alpha 0.2; no reliable orientation is estimated. An object can only be refreshed
when at least three **current scan** nonground points support its proposal. Old accumulated
points alone cannot refresh `last_observed`. The directly supported `observed_box` uses
current raw points; the filtered box uses the accumulated proposal. Last-observed time is
the supporting scan's end time, not the precise last point time of that object.

Velocity regresses accumulated-proposal centers over up to 0.6 s, requiring four
observations spanning 0.4 s. Covariance is diagonal residual variance plus a (0.025 m)^2
floor, divided by temporal regression energy. Before sufficient evidence, velocity is zero
and covariance is identity (m/s)^2 as an explicit conservative placeholder, not a measured
uncertainty. Overlapping windows correlate observations; covariance is heuristic and not
calibrated. Neither the score nor covariance should be used as a navigation safety bound.

MOVING requires speed minus two heuristic standard deviations above 0.25 m/s (0.12 when
already moving), net XY displacement at least 0.18 m, centroid jump below 0.4 m and size
change below 0.35 m. STATIONARY requires speed plus two deviations below 0.16 m/s and
net displacement below 0.12 m. Activation needs three candidate observations; stationary
release needs four. Otherwise state is UNCERTAIN. Short missed scans preserve internal
candidate evidence; misses over 0.4 s clear it. Every stale output is still UNCERTAIN.
This prevents sparse scan misses from erasing all evidence, but real moving recall remains
poor and is explicitly reported in the experiments.

Boxes persist up to 0.6 s without observation, at held filtered geometry, marked stale.
No extrapolated box is passed off as observed. Association may use a short predicted center,
but serialization never silently presents it as a measurement. `track_id` stays null.

Both evidence scores are `min(hits/min_hits, 1) * max(0, 1-age/ttl)`, not probabilities.
Object min_hits=3; background min_hits=3 by default. Runtime diagnostic names are
`background_min_observations`, `background_ttl_s`, `object_min_observations`, `object_ttl_s`.
`display_compact` is only a renderer hint after three hits over 0.2 s. It authorizes no
measurement suppression. Tests exercise new conflicting geometry inside an existing box;
this does not establish reliable separation in all overlap cases.

## Integration boundaries

- Required scene-v1 fields and fixture remain compatible with published commit
  `5e269061c3da70ffe3d00798eb9bf05977869797`; new diagnostics are additive.
- `scripts/run_stable_scene.py`: offline recorded Go2 input, gitignored JSONL and summary.
- `scripts/benchmark_stable_scene.py`: deterministic synthetic baseline and four horizons.
- `scripts/inspect_lidar_scans.py`, `benchmark_lidar_windows.py`: timing/window diagnostics.
- `scripts/evaluate_stable_scene_real.py`: annotations only after perception, geometry/motion
  comparison with original cached results. Unannotated background is not false-positive GT.
- Claude owns sensor access, firmware semantics, calibration, annotation review and renderer.
  Use `scripts/read_scene.py` and the six-row synthetic fixture without perception imports.
- Forecasting needs separately validated pedestrian identity/history; these anonymous boxes
  are not training labels. No navigation or movement controller is included.
