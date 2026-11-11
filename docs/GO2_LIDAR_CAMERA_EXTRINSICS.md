# Go2 LiDAR–camera extrinsics (Phase 5D)

**Status (2026-11-11): NOT measured.** Intrinsics are now measured
([GO2_CAMERA_CALIBRATION](GO2_CAMERA_CALIBRATION.md)); this step still needs a large flat board and
captures. No hand-tuned or visually aligned transform exists or may be added.

## What is known
| Transform | Value | Status |
|---|---|---|
| `T_base_lidar` (utlidar_lidar → base_link) | `configs/go2_sensors.yaml` | **firmware**: recovered from the robot's own `cloud`→`cloud_base` output (residual < 1e-6 m over 320 pairs); not surveyed |
| `T_base_camera` (camera optical → base_link) | — | **missing** |
| `T_odom_base` | `rt/utlidar/robot_odom`, quaternion xyzw | verified convention while stationary; accuracy under motion unverified |

Direction convention everywhere: `T_target_source`, `p_target = R p_source + t`, metres.
The interface stores `T_base_camera` because geometry is in `base_link`; `T_lidar_camera =
inv(T_base_lidar) @ T_base_camera` follows by `RigidTransform.compose`, inheriting the weakest status.

## LiDAR properties that shape the method (measured)
- `cloud_base`: single scans, 1.1–1.6 k points, ~190° per scan (median), > 300° after 5 scans; no points
  within 0.3 m XY; 0% overlap between consecutive messages. Do **not** accumulate `cloud_deskewed`,
  which already repeats ~88% of the previous message.
- Accumulating a stationary target for 1–3 s gives 15–46 scans and dense planar evidence (see
  `scripts/phase5d_lidar_accumulation.py`; the user chose 1.0 s for display and 0.5 s for obstacles,
  but calibration captures should use the full stationary hold, e.g. 3 s, since nothing moves).
- Intensity 23–230, units undocumented; `ring` is always 1 (no beam index).

## Planned method (target-based, not visual tuning)
Robot stationary throughout; target moved by the user, never the robot.
1. Board: a large rigid plane (≥ 60 × 90 cm: foam board, cardboard sheet, table leaf) with the
   ChArUco target fixed on it (print, or a tablet/laptop screen taped flat). Measure with a tape the
   ChArUco origin's offset to the big board's edges (mm).
2. For each of ≥ 8 poses (distances 0.8–3 m, left/centre/right, tilted about two axes, the board
   off the floor so its lower edge is visible), hold still 3 s: record RGB + `cloud_base` + odom.
3. Camera: ChArUco pose with the **measured** intrinsics → board plane (n_c, d_c) and board edges in
   the camera frame.
4. LiDAR: accumulate the hold's scans in base_link; segment the board by RANSAC plane within a
   region; fit (n_l, d_l) and its boundary edges.
5. Solve: rotation from normals (Kabsch over ≥ 3 non-parallel normals), then translation from
   plane distances; refine jointly with point-to-plane residuals and board-edge constraints.
   Report the condition number; with fewer than 3 well-separated orientations it is
   **underconstrained** and must be reported as such, not forced.
6. Validate on held-out poses and scenes: point-to-plane residual (mm), reprojected LiDAR board
   edge vs image edge (px), per image region (centre / corners) and per distance bin; door frames and
   table edges in the room as independent checks. A pretty overlay is not acceptance.
7. Save into the calibration YAML as `T_base_camera` with `status: measured` and provenance (poses,
   residuals, date). `T_base_lidar` remains firmware unless independently checked by the same data.

Acceptance (proposal): held-out point-to-plane median < 15 mm, edge reprojection median < 3 px at
1–3 m. Expected limits: the per-point LiDAR noise of a few cm and the 48 ± 16 ms camera latency.
Static captures make timing irrelevant for calibration; it matters only for later fusion during motion.

## Measurements to request from the user when the time comes
- Square size on the target (ruler, ≥ 5 squares).
- Board outer dimensions and ChArUco offset to the board edges.
- Optional cross-check: tape-measured camera lens position relative to a body reference (front
  face of the head) — used only as a sanity bound, never as the calibration.
