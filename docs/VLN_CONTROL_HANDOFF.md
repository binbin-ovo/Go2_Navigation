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

Claude's worktree was checked read-only:
`/Users/yushan/Desktop/Go2_Navigation-go2-control`, branch
`feature/go2-high-level-control-audit`, HEAD
`45154f6b9f05227de74254cb4562d7e0ad07b7dd`.
It contains uncommitted control work (`scripts/go2_hl_control.py`, `src/go2_control/`,
`tests/test_go2_control.py`). Neither `docs/GO2_HIGH_LEVEL_CONTROL.md` nor
`docs/GO2_NATIVE_AVOIDANCE_AUDIT.md` was present at inspection. No corresponding
remote branch was present after the initial fetch. **No published hardware result
or verified capability commit is available to cite.** Source filenames and SDK
method names are not evidence of a tested command route. That work was not modified,
imported, executed, merged or treated as an implementation dependency.

When Claude publishes those two reports, record their exact commit, test machine,
robot firmware, route and evidence artifacts here before discussing integration.
This document's “unknown” entries must not be filled with simulator settings.

## Capability checklist owned by Claude/operator

| Requirement | Verified now | Evidence needed before later physical integration |
|---|---|---|
| Selected high-level command route/method | **Unknown** | Report commit and actual tested route, firmware, client version; no silent fallback route. |
| Velocity vs position semantics | **Unknown** | Measured response, absolute/relative interpretation and reference frame. |
| Translation/yaw units and axes/signs | **Unknown on hardware** | Separate forward/left/right tests, documented units and command frame. Native model semantics alone are insufficient. |
| Command lifetime/update cadence | **Unknown** | Observed hold/expiry behavior and required send rate, including interrupted sender. |
| Ownership and operator takeover | **Unknown** | Tested acquisition, competing-client behavior, release and independent takeover. |
| Routine stop | **Unknown** | Tested method and behavior; model STOP token is not this method. |
| Independent operator stop | **Unknown** | Actual tested independent stop path and operator procedure. |
| Timeout/disconnection behavior | **Unknown** | Evidence for lost application, lost link, blocked inference, sender crash and restart. |
| Manufacturer obstacle avoidance | **Unknown for any selected command route** | Must remain **enabled**. Status plus evidence that the chosen route actually uses it. A status flag alone is insufficient. |
| Tested hardware limits | **Unknown** | Validated translational/angular limits, operating mode and environmental constraints. No simulator 0.8 m/s or 3 rad/s defaults. |
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

1. Publish Claude's two hardware reports and exact verified capability commit.
2. Confirm the Mac is free of active physical-control tests before model loading
   or full regression tests. A Git worktree is not resource isolation.
3. Run the documented **one-job local replay smoke**, retaining every native result
   and timing/error. Real inference through this new app is currently pending.
4. Later, connect the already-tested read-only receiver via the existing observation
   type. Keep output inspection-only and verify clocks, epochs, cancellation and age.
5. Consider physical integration only as a separate authorized milestone after
   frame registration, takeover/stop behavior and enabled-avoidance route evidence.
   Near-goal stopping failures remain unresolved; do not credit an external stop
   rule as native LightNav completion.
