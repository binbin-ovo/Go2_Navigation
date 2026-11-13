# Go2 LiDAR–camera extrinsics (Phase 5D)

**Status (2026-11-12): NOT measured; stopped by the user after a second session.** Intrinsics are
measured ([GO2_CAMERA_CALIBRATION](GO2_CAMERA_CALIBRATION.md)). Two sessions of board captures did not
pass the acceptance gate, so `T_base_camera` stays `null`. No hand-tuned or visually aligned transform
exists or may be added.

## Session 2 (2026-11-12): static snapshots, then a wall
Robot stationary throughout; each capture is a separate 4–8 s read-only recording taken after the
user placed the target and stepped out of view (`data/recordings/p5d_extr_snap_*`, local only).

**Board snapshots (7 usable, `p5d_extr_snap_01` … `07`).** Board 43.5 × 28.5 cm with the A4 ChArUco
(29 mm squares, 26 × 17 cm grid) standing on the floor or on a tilted stand: upright facing, turned
±20–25°, shifted left/right, and tilted back ~33° (two poses, for height). Camera: detected in every
frame of every snapshot (23–40 corners, PnP σ ≤ 1.5 mm within a snapshot) at 0.83–1.25 m. A first
attempt at position 4 (board far left, facing forward) gave 7 markers and was re-taken
(`p5d_extr_snap_04_rejected`, unused).
- LiDAR board points are noisy: **within a single scan the residual σ is 23–28 mm** (unimodal), with
  scan-to-scan offset σ 8–15 mm. The board is a ~5 cm thick slab of ~2–3 k points.
- 3-point RANSAC (`fit_plane`) locks onto the scan-line pattern on such a slab and returned normals
  ~20° off (±44° yaw instead of ±22°); a plain or Tukey-weighted PCA in a tight box agreed with the
  camera on yaw (+21.7° / −21.7° vs ≈ ±20–22°).
- Segmentation was the second failure mode: grid crops through the coarse seed, or connected
  components (0.5 m wide for a 0.43 m board), picked up the stand, the water pack and the floor.
- Result (Tukey planes on components, single solve): reduced χ² 170, leave-one-out normal error
  median 11.3° (3.5–22.9°), plane offset median 75 mm (17–198 mm). Coarse-to-fine re-cropping diverged
  (step > 1 m by pass 3). **Rejected; nothing saved.**
- The gate's point metric (held-out median |point-to-plane| ≤ 15 mm) cannot pass with 25 mm per-point
  noise even for a perfect transform (expected ≈ 17 mm). Future gates should use plane-level metrics
  (signed plane offset and normal angle), which average the noise out.

**Wall (`p5d_extr_snap_wallcheck`, `wallcheck2`, `wall_A`, `wall_A2`).** The robot faced an empty
wall; the grid was taped flat on it. LiDAR: the wall is a clean plane (7–23 k points, σ 14–18 mm).
Camera: at 1.25 m (`wall_A`) the 21 mm markers are ~13 px and only 2–9 markers were found, also on
2–3× upscaled frames; at 1.06 m (`wall_A2`) 62/93 frames detected with a median of 15 corners (minimum
12). **This grid is readable only up to ~1.1 m.** A wall gives one normal; a second wall (corner) and a
grid on the floor were planned but not captured.

**LiDAR geometry finding** (`scripts/phase5d_lidar_flatness.py` → `data/outputs/phase5d/lidar_flatness.json`,
plus a floor-height-vs-range profile): relative to the best floor plane, `cloud_base` floor returns lie
**~3–4 cm low at 0.25–0.5 m range, ≈ 0 at 0.75–1.25 m and ~2–4 cm high at 1.5–2.5 m** (medians over 40
scans; the far bins include some skirting-board points). Plane peeling therefore splits one floor into
planes 2–10° apart, in raw `cloud` as well as `cloud_base`, so the bend is in the sensor's returns, not
the firmware transform. Whole-plane fits at similar ranges are fine: wall vs floor 89.3–89.8° with the
wall at 0.7 m. (A first estimate of 84° from a 20 cm strip of that wall was a fitting artefact and is
withdrawn.) Consequence: planes measured at different ranges or elevations carry a few-cm / few-degree
bias that one rigid transform cannot absorb. Together with the 25 mm noise and a grid unreadable
beyond 1.1 m, this is why ≤ 2° / 15 mm was not reached.

**What would plausibly work next** (not attempted): a target of ≥ 60 × 80 cm with larger printed
squares (readable at 1.5 m), all poses at a similar range (1.0–1.5 m) so the LiDAR bias is common,
plane-level gates, and Tukey/PCA plane fits instead of 3-point RANSAC. Alternatively, model the LiDAR's
range/elevation bias first (floor and walls at several ranges).

**Also fixed this session:** `fit_plane` computed a full N×N SVD (NumPy's default
`full_matrices=True`). With ~60 k floor points that is ~29 GB: on 2026-11-12 a check script reached
30.3 GB and macOS jettisoned ~20 background services (JetsamEvent 16:29:26). It now uses
`full_matrices=False`, with a regression test; all later checks peaked below 160 MB.

## Session 1 (2026-11-11, paused by the user)
- Solver: `go2_perception.vln.extrinsics` (Kabsch on plane normals + least squares on offsets, then a
  plane-to-plane refinement weighted by expected noise; refuses normals that span < 3 directions) and
  `scripts/phase5d_solve_extrinsics.py` (still-period detection from ChArUco poses, background
  subtraction, LiDAR plane per pose, leave-one-pose-out validation, overlays). Synthetic tests:
  `tests/test_vln_extrinsics.py` (exact recovery < 0.3° / 1 cm; with 0.5° / 5 mm camera-plane noise
  < 1° / 2 cm and a matching reported σ). A point-to-plane refinement was tried first and **rejected**:
  on a 0.4 m board it let rotation trade against translation (0.6–3° worse than closed form).
- Captures (local): `p5d_extr_background` (10 s, empty scene), `p5d_extr_test` (15 s, board upright
  1.1 m: 198/198 frames detected, ~2 000 LiDAR board inliers, plane σ 11 mm), `p5d_extr_main` (stopped at
  ~82 s: only 2 usable still poses, both upright at ~1.1 m).
- Board: 43.5 × 28.5 cm with the A4 ChArUco (29 mm squares). It cannot stand turned or leaned on its
  own, so the next capture is **hand-held**: the user kneels behind it holding the side edges; the
  LiDAR sees mostly the board (body occluded), and points are restricted to the camera-predicted
  board plane. Still periods: ≥ 3 s within 10 mm / 2.5° (above single-frame PnP noise of 3.5 mm /
  1.4° p95 measured on the static test).
- Hand-held capture `p5d_extr_handheld` (73 s): board detected only at 0.8–1.0 m (t < 29 s; at ~1.7 m
  the 21 mm markers are ~10 px and not detected, later the board lay flat, edge-on). 7 still windows of
  1.1–3.0 s (≤ 2 cm / 3°), well spread (normal singular values 2.37 / 0.92 / 0.73). LiDAR points are
  restricted to the printed grid in board coordinates (hands and the water-bottle pack below the board
  contaminated a radius-based crop), coarse-to-fine over six passes.
  **Result rejected by the acceptance gate:** leave-one-pose-out median 11.7 mm but signed offset
  10.4 mm and normal error 2.8° median (1.0–6.3°), reduced χ² 11.7, σ ≈ 1–1.4° / 8–22 mm, and the
  translation still moved several cm between passes. Short swaying holds with ~200 noisy points per pose
  are not enough.
- **No `T_base_camera` exists yet.** Next capture: static poses of ≥ 6 s with nobody in view, the board
  leaned against a heavy object (e.g. the water-bottle pack) at different yaw / lean angles, 0.7–1.2 m.
  `scripts/phase5d_solve_extrinsics.py` refuses to save unless held-out plane distance and offset are
  ≤ 15 mm and the held-out normal error is ≤ 2° (`--max-loo-m`, `--max-loo-deg`).

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
