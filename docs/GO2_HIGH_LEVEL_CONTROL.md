# Go2 high-level control: forward / turn / stop with onboard avoidance enabled (2026-11-15)

**Branch:** `feature/go2-high-level-control-audit`, worktree
`~/Desktop/Go2_Navigation-go2-control`, based on `45154f6b9f05227de74254cb4562d7e0ad07b7dd`
(`phase6-pretrained-vln`, contains Phase 5D `e4e0d3b`). Not merged into main.

**Robot:** Unitree Go2 Pro, Ethernet tether to a Mac (`en9`, 192.168.123.222 → robot
192.168.123.161), Cyclone DDS. SDK: `unitree_sdk2_python` `814556d` (from the main
checkout's `third_party`), Python 3.10.17 (`.venv-go2`). Firmware version was not read
through the API. The service versions answered on the robot: `obstacles_avoid` 1.0.0.2,
`sport` 1.0.0.1, `motion_switcher` 1.0.0.1, `robot_state` 1.0.0.2. `CheckMode` returns
`mcf`. The Pro answers these RPCs.

**Scope:** only the supported motion route, with obstacle avoidance **on** for every
command. Avoidance was never disabled. VLN outputs are not connected to motion. Every
movement and every control-setting command was approved by the operator, one physical
step at a time. Private telemetry is in `data/outputs/go2_control/` (git-ignored).

## Evidence levels

- **HW**: verified on hardware in this session (logs plus LiDAR, IMU or tape, and the operator's eyes where stated).
- **HW-1**: seen on hardware once, or seen but not confirmed visually.
- **SRC**: from SDK source or Unitree docs only.

## Command route used

All motion goes through the **`obstacles_avoid` service** (`rt/api/obstacles_avoid/request`),
not `sport`:

1. `SwitchGet` (1002) must return `{"enable":true}`, otherwise the move is refused.
2. `UseRemoteCommandFromApi(True)` (1004) gives command source to the API.
3. `Move` (1003), velocity mode (`mode 0`), resent at 20 Hz.
4. Stop: zero `Move` ×3, wait until the robot is still.
5. `UseRemoteCommandFromApi(False)` returns command source to the remote.
6. `SwitchGet` again to record the avoidance state after the move.

`SportClient.Move` (sport 1008) is **never** used, and there is no silent fallback to it.

## Command reference

| Command | Client / API id | Args, units | Frame and signs | Velocity / position | Update rate | Lifetime | Ack | Stop behaviour | Avoidance interaction | Evidence |
|---|---|---|---|---|---|---|---|---|---|---|
| Forward | `obstacles_avoid` Move 1003 | `x` m/s, `y`=0, `yaw`=0, `mode` 0 | body; **+x = forward** | velocity | 20 Hz (0.05 s) | about 0.9–1.0 s per command | none (`_CallNoReply` reports only that it was sent) | zero Move or deadline | avoidance active; realized speed is below the command (mean about 0.22 m/s at 0.30) | **HW** (C4 tape 37 cm, LiDAR 36 cm; C5 1.495 m LiDAR) |
| Backward | same | `x` < 0 m/s | **−x = backward** | velocity | 20 Hz | same | none | same | at 0.25–0.50 m/s, avoidance holds or counter-pushes after 0.1–0.3 m; at **0.75 m/s, 3 of 3 clean** | **HW** (R4 1.87 m LiDAR) |
| Turn left | same | `yaw` rad/s > 0 | **+yaw = CCW from above = left** | velocity | 20 Hz | same | none | heading-target stop (IMU) | none observed | **HW** (D4 +35.1° IMU; D1 +38.3°); operator's visual check pending |
| Turn right | same | `yaw` rad/s < 0 | **−yaw = CW = right** | velocity | 20 Hz | same | none | heading-target stop | none observed | **HW** (D3 retry −35.4° IMU; operator saw D3 turn right before the link loss) |
| Stop | same | `x=y=yaw=0`, ×3 at 0.05 s | – | velocity zero | burst | – | none | still in **0.2–0.4 s** from 0.3–0.65 m/s (≥8 trials) | – | **HW** |
| Claim command source | `obstacles_avoid` 1004 | `{"is_remote_commands_from_api": true}` | – | setting | once | until released | code 0 | – | required before Move | **HW** |
| Release command source | `obstacles_avoid` 1004 | `false` | – | setting | once | – | code 0 | remote drives again | – | **HW** (B3 sideways stick after release) |
| Avoidance status | `obstacles_avoid` SwitchGet 1002 | – | – | read | on demand | – | `{"enable":true}` | – | read before and after each move; always true | **HW** |
| Sport StopMove | `sport` 1003 | – | – | – | – | – | – | resets parameters | not used | **SRC** |
| Sport Move | `sport` 1008 | vx, vy, vyaw | body | velocity | – | 1 s (docs) | – | – | **not used**; route not tested | **SRC** |
| Position modes | `obstacles_avoid` Move `mode` 1 or 2 | increment or absolute pose | – | position | – | – | – | – | not used | **SRC** |

Utility limits (`src/go2_control/safety.py`):
- forward ≤ 1.0 m/s; above 0.30 m/s only with a distance target (raised 2026-11-18 at the user's request)
- reverse ≤ 0.75 m/s, and only with a distance target
- |yaw| ≤ 2.0 rad/s (was 1.0 until 2026-11-18); above 0.5 rad/s only with a heading target
- targets ≤ 1.6 m or ≤ 1.6 rad
- duration ≤ 2 s, or ≤ 8 s with a target
- no combined translation and turn

## Measured behaviour (hardware)

**Forward:**
- At 0.15 m/s (C, C2) and at 0.30 m/s for 1.5 s (C3), the robot stepped but did **not visibly walk**. The speed ramps up slowly, so C3 ended while still in the ramp.
- At 0.30 m/s for 2 s (C4), it walked: tape 37 cm, LiDAR 36.2 cm, odom 29.4 cm. **Sport odometry under-reads by about 20 %.**
- A 1.5 m target (C5) reached 1.495 m by LiDAR (deadline-limited).
- A 0.50 m target (X2) stopped at 0.507 m.

**Backward:** the rear corridor is **blind to the LiDAR** (|y| < 0.35, out to −4 m, 0 returns).
- At 0.25 m/s: 0.33 m (R1), then 0.28 m followed by a hard hold (R2), then no motion or a ~1 s forward/back cycle (R2 retry, X3b).
- At 0.35 m/s: 0.19 m, then a hold.
- At 0.50 m/s: one clean run (0.64 m) and one hold after 0.12 m.
- At **0.75 m/s: 0.69, 0.78 and 1.87 m (target 1.5), all clean.** Reverse at 0.75 overshoots by about 0.2–0.37 m: the speed is about 0.65 m/s and LiDAR progress jumps in steps.
- With the **remote**, full back stick in the held spot (X1) was also blocked at first, then broke through in bursts of up to 0.9 m/s. So the hold is not specific to the API route.
- Interpretation (inference, not proven): the onboard avoidance or locomotion limits slow reverse into unseen space.

**Turns:**
- At 0.5 rad/s (timed): left +38.3°, right only −20° peak, springing back to −11°. Right was slower than left, and the robot drifted back after stopping.
- At **1.0 rad/s with a 30° heading stop: right −35.4°, left +35.1°**. That is symmetric, with about 5° of overshoot and no spring-back. Heading-target turns at 1.0 rad/s are the recommended form.

**Stop:**
- A zero Move stops the robot in about 0.2–0.4 s.
- When commands simply end, the robot stops by itself about 0.9–1.0 s after the last one.

## Command source and operator stop

- With the remote owning control, releasing the sticks stops the robot in 0.44 s (B2, **HW**).
- After `UseRemoteCommandFromApi(False)`, the remote drives the robot again (B3, D3 cleanup, **HW**).
- **Unverified:** whether the remote can override while the API holds control. During the B3 claim, `wireless_remote` read all zeros, and no operator nudge was made to tell masked input from no input. The utility aborts on any stick input it sees, but it may be blind to the remote while the API holds control. The operator's backup is the app or the robot's own controls.
- Three of five early runs showed a small backward step (2–5 cm) right after the release. Unexplained.

## Timeout and disconnect

- The robot-side lifetime of a single command is about 0.9–1.0 s (C, C2: the robot stopped while still being commanded at 5 Hz or once; **HW-1**).
- The utility aborts when state is older than 0.25 s and then sends no further velocity.
- **D3 incident, an accidental link loss mid-turn** (16:38):
  - The Ethernet link dropped 0.72 s into a −1.0 rad/s turn. The cable was most likely pulled by the turn.
  - The utility aborted on stale state at 0.97 s. Its zeros and release were undelivered (release code 3104).
  - The operator reported the robot stopped and safe. Software control stayed claimed until step S (zero ×3 + release, codes 0) cleared it after reconnecting.
  - Lesson: route the cable with slack for every turn, and after any link loss run `stop --execute` before further motion.
- No deliberate disconnect tests were done.

## Utility

`scripts/go2_hl_control.py` (package `src/go2_control/`, isolated from the read-only sensor adapters):

| Mode | Sends commands? | What it does |
|---|---|---|
| `status` | no | read-only RPCs and state |
| `watch` | no | read-only state stream |
| `range` | no | read-only front LiDAR range |
| `preview` | no | dry run; prints the confirmation phrase |
| `ownership` | control setting | claim and release only |
| `move` | yes: one bounded move | needs `--execute --stop-verified --confirm "<phrase>"`; always stops and releases |
| `stop` | yes | zero ×3, then release |

Other safeguards:
- A lock file allows only one commanding process at a time.
- Importing the package does nothing.
- `SwitchSet` (1001) is never registered, so it cannot be called.

```bash
PYTHONPATH=src:$HOME/Desktop/Go2_Navigation/third_party/unitree_sdk2_python ~/Desktop/Go2_Navigation/.venv-go2/bin/python scripts/go2_hl_control.py preview --vyaw 1.0 --turn-deg 30 --duration 3.0 --resend-period 0.05
```

Tests: `tests/test_go2_control.py`, 40 mock tests pass and 1 is skipped. No robot is needed.

## Open items

- Visual confirmation of D3 retry and D4 turn magnitudes (IMU says about 35°).
- Remote override while the API holds control (needs a deliberate, approved test).
- Reverse below 0.75 m/s is unreliable. Reverse overshoot at 0.75 m/s needs earlier braking or a slower final approach. The rear is blind to the LiDAR, so the operator must check rear clearance.
- Small post-release backward step; slow realized forward speed (about 70 % of command).
- Avoidance-off behaviour: see [GO2_NATIVE_AVOIDANCE_AUDIT](GO2_NATIVE_AVOIDANCE_AUDIT.md). Interface exists, not tested.
