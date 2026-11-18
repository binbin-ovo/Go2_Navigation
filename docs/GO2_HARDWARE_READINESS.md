# Go2 hardware readiness for supervised navigation (Claude, 2026-11-18)

Branch `feature/go2-hardware-readiness`, based on Codex's `feature/vln-supervised-navigation`
@ `4063c73` (contract `2213ced`). Canonical command checkout: `~/Desktop/Go2_Navigation-go2-control`
@ `742cfe4`, lock `data/outputs/go2_control/.command.lock`; SDK `814556d`. Main is unchanged.
Recordings and logs stay local and git-ignored; only aggregates are published here.

**Status: no physical action has been taken in this task yet.** Model-directed forward motion is
blocked by three facts listed under "Blocking the first model-directed forward action".

## 1. Calibration reuse (no new lens calibration)

| Item | Evidence | Status |
|---|---|---|
| Intrinsics + fisheye distortion, 1280×720 | `configs/go2_front_camera_calibration.yaml`, sha256 `f5dedd82…`, held-out RMS 0.47 px | **measured**, reused unchanged |
| Live pipeline loads it | `run_vln_interactive.py` refuses non-`measured` intrinsics; recorded live run `live_10/run.json` names this file and sha | confirmed |
| Rectification | `model_preprocessing.calibrated_view`: OpenCV fisheye, balance 0, R = I, same 1280×720, no crop | confirmed by code; rectified K fx 769.8, cx 650.9, cy 371.8 |
| Model input geometry | 1280×720 rectified → 448×256 official bilinear stretch, no crop/pad; x/y scale 0.350 / 0.356 (1.6 % anisotropy) | confirmed by code |
| Rectified field of view | 79.5° × 50.0° | computed from the measured model |
| Image-centre ray vs optical axis | rectified cx is 11.4 px right of the image centre → the centre column looks 0.85° **left** of the optical axis | computed; enters the mapping yaw below |
| `T_base_camera` | `null`; two extrinsic sessions failed their gates (GO2_LIDAR_CAMERA_EXTRINSICS) | **missing; not filled here** |

No incompatibility or defect was found, so the lens calibration was not repeated.

## 2. Mapping needed for the forward primitive

LightNav rows are absolute SE(2) poses in the chunk-start ego frame (x forward, y left, yaw CCW, metres).
The contract maps a displacement P by `T_base_model · P · T_base_model⁻¹`. For T = (t, θ) and a row
(p, ψ) this gives translation `R_θ p + (I − R_ψ) t` and yaw ψ. Consequences:

- **Forward needs only θ.** For a pure translation (ψ = 0) the model-frame origin t cancels exactly.
  For a nearly straight row it contributes at most `2|sin(ψ/2)|·|t|`.
- **The model-frame origin t is unknown.** The upstream code does not store it (WAYPOINT_EXECUTION_AUDIT),
  the upstream MuJoCo demo used a camera 0.16 m ahead of its body, and the Go2 camera position is not
  registered. Camera heading does not establish it. It is kept **unknown** (`translation_m: null`) with
  a declared bound of 0.5 m: any plausible ego origin of a robot carrying this head camera lies within
  0.5 m of base_link. This is a bound on an unknown, not a measurement.
- **Turns need t** (the `(I − R_ψ) t` term is the whole translation residual), so a null translation
  can never verify `turn`. Turn mapping is out of scope until t is registered.
- **Scale**: the model's metric scale on this camera/height is unverified. The executed distance is
  `min(mapped x − slack, forward_m, current LiDAR clearance)`, so a scale error changes the distance
  within those caps but not the direction. The forward primitive executes along base +x only.
- **Signs**: +x forward, +yaw CCW/left on both sides (control: HW-verified; model: source audit).

### Contract change (small, reviewable)
`supervised.py` + schema: `mapping.translation_m` may be `null` with a required `translation_bound_m`,
only when `verified_actions` excludes `turn`; optional `yaw_uncertainty_rad`. The planner charges both as
a worst-case slack (`mapping_slack`): the lateral residual becomes `|y| + slack`, the forward magnitude
`x − slack`. With a known translation and no yaw uncertainty the behaviour is unchanged. Tests:
`test_unregistered_translation_is_bounded_not_zero`, `test_mapping_yaw_uncertainty_widens_lateral_residual`.

### Camera heading θ from existing recordings
`hardware/camera_heading.py` reuses the Phase 5D static captures (7 board snapshots + wall), which hold
the same physical plane as a ChArUco normal in the camera and as a LiDAR plane in base_link. A
rotation-only Kabsch fit over the normals gives the optical axis heading; only its yaw is used.
Translation is not estimated and nothing becomes a `T_base_camera`.

| Capture | Camera frames / LiDAR scans | LiDAR plane | Yaw (az. LiDAR − az. camera) | Used |
|---|---|---|---|---|
| `p5d_extr_snap_wall_A2` (taped grid on a wall, 1.03 m) | 62 / 124 | 7 369 pts, σ 24 mm | **+0.8°** | yes |
| 7 board snapshots (A4 board on stand, 0.85–1.16 m) | 29–114 / 61–122 | 1.1–2.3 k pts, σ 20–26 mm | −20.2 … +16.8° | no |

The board snapshots reproduce the Phase 5D segmentation failure: the LiDAR fit of a 43 × 29 cm slab
merges with its stand and the background, so its azimuth stays near the background's (the turned boards
give ±11–20° while the camera sees the turn). Elevation/position checks could not separate them, so only
an unambiguous large plane (≥ 4 000 inliers) is accepted. The 3-D rotation fit over all normals is
ill-conditioned (singular values 7.11 / 0.49 / 0.19) and is not used.

**Camera optical-axis yaw = +0.8° (left of base +x).** Declared uncertainty **±3°**: one accepted
capture; Phase 5D's tight-box board fit agreed with the camera on board yaw within ~2°; the model's
image-centre convention is assumed, not confirmed. Mapping recorded in the capability:
`yaw_rad = 0.0294` (+0.8° camera + 0.85° centre ray), `yaw_uncertainty_rad = 0.0524`,
`translation_m = null`, `translation_bound_m = 0.5`, `verified_actions = ["forward"]`.
Effect on a forward decision: lateral slack `|p|·2 sin(1.5°) + 2|sin(ψ/2)|·0.5 m`, e.g. 6.6 cm for a
0.3 m row with ψ = 0.10 rad, inside `lateral_m = 0.08`.

Why not forward probes: walking adds ±3–7° of heading change and 3–12 cm of lateral drift over
0.3–1.3 m (control logs, below), pitch/roll oscillation and timing error; a focus-of-expansion estimate
would need several metres of straight travel to beat the static estimate. It stays a cross-check.

## 3. Control evidence reused (canonical logs, `hardware/control_log_summary.py`)

| Run | Request | Measured (IMU / LiDAR odom) | Stop |
|---|---|---|---|
| C4 | fwd 0.30 m/s, 2 s | 0.29 m odom (tape 0.37), heading +1.7°, lateral +3.5 cm | still 0.37 s after zero |
| X2 | fwd 0.30 m/s, target 0.5 m | 0.44 m odom (LiDAR 0.505), heading +2.8°, lateral +5.7 cm | still 0.33 s |
| C5 | fwd 0.30 m/s, target 1.5 m, 6.5 s | 1.27 m odom (LiDAR 1.495), heading +6.8°, lateral +12 cm | still 0.22 s |
| T1 | turn +0.5 rad/s, 2 s timed | peak +40.3°, settled +38.3° (2° springback), 7 cm sideways | still 0.37 s |
| T2 | turn −0.5 rad/s, 1.4 s timed | peak −20.6°, settled −12.6° (**8° springback**) | still 0.08 s |
| D3r | turn −1.0 rad/s, target 30° | −32.8° at stop, settled **−35.4°**, translation 4 cm | still 0.43 s |
| D4 | turn +1.0 rad/s, target 30° | +30.2° at stop, settled **+35.1°**, translation 2 cm | still 0.19 s |

Forward walking veers left (heading +1.7 to +6.8°) and the realized speed is ~70 % of command; sport
odometry under-reads ~20 % (tape/LiDAR). Turns at 1.0 rad/s with a heading target coast 2.6–4.9° after
the stop command; the only sub-30° data (0.5 rad/s timed) sprang back by 8°. There is **no measurement
of a heading target below 30°**, so no smaller turn is claimed. Remote override while the API owns
control was never tested (B3 claim read all-zero sticks with no nudge).

## 4. Turn clearance: what the sensors can and cannot see

Swept region of an in-place turn: the standing footprint (~0.70 × 0.40 m including feet) about base_link
reaches r ≈ 0.40 m; with 0.15 m margin the turn needs a clear disc of **r = 0.55 m**, plus the Ethernet
cable. Static 2 s capture `p5d_extr_snap_01`, 40 scans, returns per 45° sector (rear = ±180°):

| Ring | `cloud_base` (provider input) | raw `rt/utlidar/cloud` → base |
|---|---|---|
| r < 0.30 m (under the body) | **0 everywhere**: removed by processing | 27 k, floor height |
| 0.30–0.50 m, rear sectors (135–225°) | **0**: removed by processing | 677 + 583 (sparse, floor height) |
| 0.30–0.50 m, side-rear (90–135°) | 250 / 305 | 821 / 985 |
| 0.50–0.80 m, rear | 140 / 15 | 698 / 410 |
| behind x < −0.35, \|y\| < 0.35 | 0 (documented "blind rear corridor") | 1 468 to x = −0.95 m, nothing beyond ~1.3 m |

So the rear/side-rear part of the swept disc is: (a) **missing in `cloud_base` because of processing**
(the firmware conversion drops r < 0.3 m and most rear returns), (b) **present but sparse in the raw
cloud** to about 1 m behind the body, unvalidated for obstacle detection, and (c) **missing coverage**
beyond ~1.3 m behind (body occlusion). The cable is never sensed. The current provider therefore reports
`turn_clear_rad = 0` (HOLD for any model-directed turn).

**Smallest step towards sensor-based turn clearance**: a stationary check that a box placed 0.4–0.5 m
behind / beside-behind the robot appears in the raw cloud within 0.5 s; if so, a raw-cloud swept-disc
check can be added to the provider and validated. Until then a turn is a **controller-only test with
operator inspection**, run with the audited `go2_hl_control.py move` (no executor, no model), labelled as
a control test. If an operator-cleared model-directed turn is wanted later, the smallest change is an
explicit capability scope (e.g. `turn_clearance_basis: "operator_inspection_experiment"`) whose geometry
record names that basis and an attestation separate from the authorization click; normal sensor-based
behaviour stays the default. Not implemented.

## 5. Target-correspondence gate (offline, recorded trial conditions)

`hardware/correspondence_check.py` on live_09/live_10 (rectified source images, odometry-labelled pairs,
three ROIs): stationary pairs (< 3 cm, < 2°) **21/24 valid**; moved/turned pairs (≥ 10 cm or ≥ 5°)
**0/18 valid**; model point outside the ROI **0/42 valid**. Refusals are conservative HOLDs. Limited to
one room and generic ROIs; it must be re-observed on the actual target at the trial.

## 6. Capture-age bound

The configured camera latency (0.048 s, σ 0.016) was measured relative to odometry stamps and excludes
odometry transport latency, so it is not an upper bound. The provider HOLDs until a bound is measured
on the current boot. Smallest experiment (stationary): the Go2 camera views the Mac screen showing a
millisecond clock; receipt time − displayed time over ~20 s bounds exposure-to-receipt latency from
above (display latency only adds).

## 7. Mac data availability (found 2026-11-18)

`~/Desktop` is iCloud-managed and the disk is 96 % full (18 GiB free), so macOS had offloaded
(`dataless`) many files, including 12 327 in `.venv-go2`, 9 in the SDK, 3 611 in the LightNav model env
and most Phase 5D recordings; reading one blocks until iCloud restores it (about 10 files/min). The
hardware imports (numpy, cyclonedds, unitree_sdk2py) load in 0.3 s, so the runtime files are local; a
background read is restoring the rest. If a launch stalls with no CPU, an offloaded file is the first
suspect. Freeing disk space or marking these folders "Keep Downloaded" in Finder prevents it (user's call).

## 8. Live evidence and the first model-directed step (2026-11-18)

All on one robot boot (`go2_odom_boot:-1766947283`), Go2 standing in the kitchen, avoidance read enabled
before and after every step, no `SportClient.Move`, no `SwitchSet`. Logs are local and git-ignored.

| Item | Experiment | Result |
|---|---|---|
| Intervention under API ownership | `ownership --execute --seconds 40`, no Move; operator pressed the controller emergency button | press seen at 17.6 s, `error_code` 100 → 1002 at 18.9 s, operator saw the robot **freeze standing**; release code 0; avoidance enabled; operator cleared → 100 |
| Obstacle handling (forward corridor) | provider live, stationary, box placed | open: nearest 2.88 m / clear 1.80 m; box centred ~1.1 m from the nose: 1.53 / 1.03 m; box 20 cm left: 1.58 / 1.08 m; geometry age ≤ 0.11 s |
| Capture-age bound | `hardware/capture_latency.py`: Mac screen flashes black/white at random 0.4–0.9 s in front of the camera; switch time taken before drawing | 39/39 switches; switch→first-RTP receipt median 0.117 s, max 0.176 s (includes display lag and up to one 0.07 s frame); bound = 0.176 − 0.048 + 0.020 = **0.148 s** for this boot |
| Target correspondence on the trial target | read-only viewer `live_14`, box as target, operator ROI around the box | operator judged 9/9 model points on the box; same view 0.3–1.1 s apart 20/24 valid (box at the left image edge refused); point outside the ROI 0/6; box moved between requests 0/5 |
| Model answers vs the forward primitive | same session | box ~1.1 m: native STOP, zero waypoints; box off-centre: every answer combined a 15–30° turn with travel → planner HOLD (`unsupported_reverse_lateral_or_combined_primitive`); box dead centre ~2.5 m: 3/3 straight (first row 0.15 m, yaw ≤ 0.01 rad) → forward 0.143–0.145 m |
| Answer delay | same session | first request 3.07 s (warm-up, STALE); then 1.94–2.04 s frame-to-answer (inference ~1.67 s) |

**Executor fixes found by the live runs** (both small, tested, committed):
- `error_code` nominal value on this robot is **100**, not 0; the executor's `!= 0` check would have
  held every live step before claim. Only 100 is accepted; 1002 (emergency stop) and anything else HOLD.
- First supervised cycle `claude_one_step_01` held before claim with `stale_or_future_observation`: the
  1.0 s pre-claim stationary dwell only started after the model answer, so the frame was 3.02 s old at
  the last check (limit 3.0 − 0.148). The idle loop now samples fresh stillness every ~0.05 s and the
  dwell continues that run only if the last sample is ≤ 0.1 s old; any moving/stale sample resets it.
  Dwell length and the 3.0 s limit are unchanged.

**First integrated step, `claude_one_step_02`** (operator: preview → ROI "cardboard box" → confirm →
authorize once; box dead centre ~2.45 m from base_link):

| Stage | Value |
|---|---|
| Authorized envelope | forward 0.143 m, 0.30 m/s, 4.0 s |
| Refreshed answer | forward 0.143 m, row 0 mapped (0.151, 0.0045, −0.00003) |
| Frame age at claim | 2.03 s (limit 2.852 s) |
| Velocity phase | 23 Move commands over 1.27 s, then zero ×3 |
| Stop | fused progress 0.154 m ("target reached"); still 0.27 s after the first zero |
| Release / after | release 0, still before and after release, avoidance enabled; post-step frame captured |
| Displacement | odometry 0.129 m at stop, 0.152 m settled (odometry under-reads ~20 % → ~0.18–0.19 m); LiDAR clearance 1.93–1.96 → 1.74 m (~0.19–0.22 m); heading −1.1°, lateral −4 mm |
| Operator | "clean small step" |

A 0.14 m request stops inside the walking ramp and coasts ~4–5 cm: the magnitude is a requested bound,
not a precise stop. The planner uses only the first non-identity waypoint (0.15 m minus mapping slack),
so a model-directed step is currently ≤ ~0.15 m requested.

## Status after the live session

The forward primitive is verified end to end for one supervised step. Remaining before turns:
yaw response and stop below 30°, springback, translation, clearance of body/legs/cable over the swept
disc (sections 3–4), and the model/body mapping translation (`translation_m` null). A turn is next
tested **controller-only** with the audited utility (no model, no executor), labelled a control test.

## 9. Controller-only turn test (control test, not model-directed; 2026-11-18)

Audited utility only (`go2_hl_control.py move --vyaw -1.0 --turn-deg 15 --duration 3`), no model, no
executor; operator inspected a clear r = 0.55 m disc and laid the cable with slack. Log
`go2_control/20261118_111433_move`, summarised with `hardware/control_log_summary.py`.

| | 15° right (11-18) | 30° right (11-15, D3r) | 30° left (11-15, D4) |
|---|---|---|---|
| Move commands | 11 (0.55 s) at 1.0 rad/s | – | – |
| Heading at stop request | −16.7° | −32.8° | +30.2° |
| Peak / settled | −19.4° / **−17.4°** | −35.8° / −35.4° | +35.1° / +35.1° |
| Springback | 2.0° | 0.4° | 0.0° |
| Translation (fwd, left) | **4.7, 5.2 cm (7.0 cm)** | 3.6, 2.0 cm | −0.7, 2.0 cm |
| Still after first zero | 0.18 s | 0.43 s | 0.19 s |
| Release / avoidance after | 0 / enabled | 0 / enabled | 0 / enabled |

Operator: clean turn. Overshoot behaves like a roughly fixed coast after the stop (+2.4° here, ~+5° at
30°), not a fraction of the target; one run, so no claim below 15°. The 7 cm translation exceeds the
planner's `turn_translation_m` (0.05 m), so a model-directed 15° turn would currently HOLD on its own
residual even with clearance. A single fixed rotation centre does not explain it (17.4° with 5.2 cm left
implies ~0.17 m offset, the 30° runs imply ~0.03 m, and neither predicts the 4.7 cm forward), so it is
more likely start/stop stepping than a geometric offset; the turning centre relative to the odometry
point and to the model origin (`translation_m`) remains **unknown**.

Missing for a model-directed turn: repeated runs per direction at 15° (spread of settled heading and
translation), the relation of the turning centre to base_link and to the model origin, sensor coverage
of the rear/side-rear swept disc (section 4), and cable handling.

## Historical: what blocked the first model-directed forward action (all resolved above)

1. **Intervention while the API owns control** is unverified. The user's controller has an emergency
   button; it has never been exercised while `UseRemoteCommandFromApi(True)` is held. Smallest
   experiment: stationary `ownership --execute --seconds 20`, the user presses the emergency button at
   ~10 s, the log shows the state change and release.
2. **Capture-age upper bound** for the current boot (section 6).
3. **Live provider check** in `.venv-go2` (subscribers only): pose/epoch rate and, with a box placed
   ~1 m ahead, `forward_clear_m` ≈ box distance − 0.5 m; this is the obstacle-handling evidence for the
   forward primitive (stationary, no command).

Obstacle handling for the forward primitive rests on the provider's LiDAR clearance (re-checked every
loop) plus onboard avoidance, which stays enabled; avoidance's own stopping is not claimed.
