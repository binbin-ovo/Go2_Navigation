# Go2 native obstacle avoidance audit (2026-11-15)

Companion to [GO2_HIGH_LEVEL_CONTROL](GO2_HIGH_LEVEL_CONTROL.md).
- **HW** = verified on the Go2 Pro in this session.
- **SRC** = from SDK source or Unitree documentation only.

## Status reading (HW)

`obstacles_avoid` SwitchGet (API 1002) returned `{"enable":true}`:
- on every read: at startup, before and after every one of the ~25 moves, and after the D3 link-loss cleanup
- with code 0, first attempt

ServiceList shows `obstacles_avoid` with status 0, the same as the other running services. `sportmodestate.range_obstacle` is always `[0,0,0,0]` on this firmware, so it cannot be used as an avoidance signal.

## How avoidance showed up in motion (HW, interpretation marked)

| Observation | Evidence | Interpretation |
|---|---|---|
| Forward speed is below the command (about 0.22 m/s mean at 0.30) | C5 LiDAR / time | could be route speed shaping or gait ramp; not attributed |
| Slow reverse held after 0.1–0.3 m, or cycled forward/back about every 1 s | R2, X3b, X3c, R3 | **inference:** avoidance or locomotion resisting reverse into LiDAR-blind space |
| Same hold with the remote (full back stick), then bursts up to 0.9 m/s | X1 (watch only) | not specific to the API route |
| Reverse at 0.75 m/s clean 3/3 | X3e, X3f, R4 | the speed threshold matters; mechanism unknown |
| No intervention seen during turns at 0.5–1.0 rad/s | D1–D4 | – |

**LiDAR coverage:** the `cloud_base` rear corridor (|y| < 0.35 m, x from −0.3 to −4 m) had **0 returns**. Sectors with |azimuth| > 150° are sparse. The rear is effectively blind, so rear clearance must come from the operator.

No test placed an obstacle in the robot's path, so **stopping before an obstacle is not verified**.

## Avoidance-off interface: documentation only

**Interface exists; hardware disable behavior not tested.**

- **Route (SRC):** `obstacles_avoid` `SwitchSet`, API id **1001**, parameter `{"enable": false|true}`, called with a reply (`_Call`). The SDK is `unitree_sdk2_python` `814556d`, `unitree_sdk2py/go2/obstacles_avoid/obstacles_avoid_client.py`. The official C++ `unitree_sdk2` header exposes the same method. The service API version is 1.0.0.2, which matches the robot.
- **Not called:** this project's restricted client (`src/go2_control/sdk_routes.py`) registers only 1002, 1003 and 1004. So `SwitchSet` cannot be sent through the client's registry. A test (`tests/test_go2_control.py`) enforces this: no code line may build an `{"enable"` request.
- **Unknown (not tested):**
  - whether the setting persists across reboots
  - whether it affects the remote or only the API route
  - whether `Move` on this route still runs with avoidance off
  - how it interacts with `sport` SwitchAvoidMode (2058) / FreeAvoid (2048), which are a separate "avoid" concept in the sport service

### Future comparison modes (proposal; each needs separate written approval and a safety plan)

1. **Avoidance on, API route.** This is the current, tested baseline: same commands, obstacle placed ahead at a known distance, measure the stop distance.
2. **Avoidance off, API route, no obstacle.** Open area, low speed, operator ready. Compare speed tracking and reverse behaviour with mode 1. This tells whether the reverse hold and the speed shaping come from avoidance.
3. **Avoidance off with a soft obstacle.** Only if mode 2 is safe, and only with external stop measures. This shows what protection is lost. The setting must be restored and read back with SwitchGet true afterwards.

None of these ran. Disabling avoidance requires the user's explicit decision. This audit does not recommend it for routine use.
