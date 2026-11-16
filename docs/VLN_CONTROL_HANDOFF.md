# VLN advisory → future hardware handoff

Date: 2026-11-15. The application on `feature/vln-shadow-demo` is **read-only**.
It contains no motion route, command publisher, avoidance toggle or robot backend.
Cancelling its instruction does not send a stop, including zero velocity. Model
STOP is a proposal, not proof that the goal is reached or the physical Go2 stopped.

## Evidence inspected

The isolated demo starts from completed Phase 6D publication
`1cc74909e489e1037c9f6f0051ab34e855d144dc`. It preserves the native semantics audited
in [WAYPOINT_EXECUTION_AUDIT](WAYPOINT_EXECUTION_AUDIT.md): ten absolute SE(2) poses
in **one decision-time local frame**, x forward/y left/yaw CCW, metres/radians,
without an intrinsic waypoint time base. No summing absolute poses as deltas; no
camera-optical-to-robot-base substitution. Real `T_base_camera` remains unresolved.

Claude's completed reports were read from his clean worktree
`/Users/yushan/Desktop/Go2_Navigation-go2-control`, branch
`feature/go2-high-level-control-audit`, commit
**`742cfe4fb1b71f9d19d9d2d62be56f03b6b03707`**:
[GO2_HIGH_LEVEL_CONTROL](https://github.com/binbin-ovo/Go2_Navigation/blob/742cfe4fb1b71f9d19d9d2d62be56f03b6b03707/docs/GO2_HIGH_LEVEL_CONTROL.md)
and [GO2_NATIVE_AVOIDANCE_AUDIT](https://github.com/binbin-ovo/Go2_Navigation/blob/742cfe4fb1b71f9d19d9d2d62be56f03b6b03707/docs/GO2_NATIVE_AVOIDANCE_AUDIT.md).
They supersede the initially unavailable-report status at advisory implementation
commit `b3c3dff`. **Hardware findings below are attributed to Claude/operator's
reports, not independently repeated by Codex.** No control code was imported,
executed, merged or made an advisory dependency.

Reported setup: Go2 Pro, Ethernet to the Mac, SDK `814556d`, isolated Python 3.10.17.
Robot firmware version was not read; reported service versions are
`obstacles_avoid` 1.0.0.2, `sport` 1.0.0.1, `motion_switcher` 1.0.0.1 and
`robot_state` 1.0.0.2. Use the pinned reports for private telemetry locations.

## Capability checklist owned by Claude/operator

| Requirement | Evidence reported by Claude/operator | Remaining limits / integration requirement |
|---|---|---|
| Selected high-level command route/method | `obstacles_avoid` velocity Move, API 1003; no sport-Move fallback. | No physical route in this advisory app. Robot firmware version unknown. |
| Velocity vs position semantics | Mode 0 velocity used; position modes were not tested. | Native absolute poses cannot be copied into velocity fields. |
| Translation/yaw units and axes/signs | m/s, rad/s; +x forward; +yaw CCW/left, −yaw right, supported by IMU/operator observations. | Lateral semantics not established by these tests. Some visual turn checks remain pending. Real model-to-base registration missing. |
| Command lifetime/update cadence | Tested resend 20 Hz; reported expiry about 0.9–1.0 s, limited evidence. | Not a guaranteed watchdog specification across failures/firmware. |
| Ownership and operator takeover | API claim/release returned success; remote drove again after release. | **Override while API owns control remains unverified**; remote state may be masked. |
| Routine stop | Reported zero-velocity routine stopped within about 0.2–0.4 s in observed trials. | This app never invokes it; native STOP is not a physical stop. |
| Independent operator stop | Releasing remote sticks stopped when remote owned control. | Independent override/stop under API ownership is not established. |
| Timeout/disconnection behavior | One accidental mid-turn link loss: stale-state abort; stop/release undelivered; operator reported stopped; ownership cleared after reconnection cleanup. | Deliberate disconnect tests not done; release is not assured on broken link. |
| Manufacturer obstacle avoidance | Selected route is the avoidance service; status true before/after reported runs. | **No obstacle was placed in the path: obstacle-stopping is unverified.** Reverse holds may reflect avoidance/locomotion; mechanism unknown. Must remain enabled. |
| Tested hardware limits | Utility bounds: forward ≤0.30 m/s, reverse ≤0.75 m/s only with distance target, yaw magnitude ≤1.0 rad/s with heading target above 0.5; no combined translation/turn. | These are reported utility limits, not a validated safe envelope. Reverse overshoot/blind rear and unexplained post-release steps remain. Do not import simulator defaults. |
| Model-frame → command-frame transform | **Unvalidated** | Measured `T_base_camera`, frame/axis tests and correct decision-time pose handling. |
| Sensor → advisory connection | **Pending** | Operator-coordinated read-only test after control testing, recording clock/calibration/source provenance. |

The future requirement is **manufacturer obstacle avoidance remains enabled**.
There is no avoidance-disable implementation here. If a proposed route cannot
demonstrate that requirement, record the incompatibility and stop integration;
do not switch routes silently. This checklist grants no physical execution approval.

## Exact sensor input boundary

Use the existing [Go2 observation contract](GO2_VLN_INPUT_INTERFACE.md), not native
robot SDK objects inside the model code. `ObservationInbox.offer` in
`src/go2_perception/vln/shadow_sources.py` accepts the current `Go2VLNObservation`
plus delivery/session metadata. It opens no connection. The detailed example and
clock rules are in [VLN_SHADOW_DEMO](VLN_SHADOW_DEMO.md#future-sensor-consumer).

- RGB: uint8 H×W×3 **RGB**, named optical frame, exposure/receipt `Stamp`s with
  uncertainty, image dimensions, calibrated K/D/model/size, raw/rectified flag.
  Receiver conversion must preserve the measured fisheye calibration and avoid
  a second rectification. Synthetic pinhole input must not use Go2 fisheye parameters.
- Timing: unique clock epoch, explicit exposure and availability semantics,
  availability in the visual stamp's domain; optional same-host monotonic exposure
  mapping only with documented conversion/uncertainty. No robot-time minus host-wall
  arithmetic. Incomplete mapping means exposure age unknown, not zero.
- Pose: optional `RobotState` with a real timestamp/frame and `T_odom_base`, or
  `None`. No identity substitute. Pose is preserved for inspection, never model input.
- LiDAR: optional separate `GeometricObservation` with its own timestamp/frame.
  It is not fused or classified by this advisory application. Missing returns do
  not imply free space; floor-seen cells are not certified traversable.
- Ownership/backpressure: caller supplies unique source/frame IDs; inbox copies
  data and returns `False` on overflow. Report loss/invalidate history explicitly.
  Session mutation belongs to the application event loop, not receiver callbacks.

All inference receives only terminal instruction and causal calibrated RGB history.
Pose, LiDAR, hardware state, semantic labels, references and evaluator goals do not
enter the native model job. No proposal is fed back as an executed action during replay.

## Next operator-coordinated steps

1. The reports above and the operator-approved offline verification window are now
   recorded. [One real-model replay and offline regressions](VLN_SHADOW_VERIFICATION.md)
   completed sequentially; the real proposal was correctly marked stale. Re-coordinate
   if physical testing resumes. A Git worktree is not resource isolation.
2. Connect the already-tested read-only receiver via the existing observation
   type. Keep output inspection-only and verify clocks, epochs, cancellation and age.
3. Consider physical integration only as a separate authorized milestone after
   frame registration, takeover/stop behavior and enabled-avoidance route evidence.
   Near-goal stopping failures remain unresolved; do not credit an external stop
   rule as native LightNav completion.
