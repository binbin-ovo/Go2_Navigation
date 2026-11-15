# Phase 6D native waypoint execution audit

Audit date: 2026-11-15. Written before changing execution. Base publication:
`45154f6b9f05227de74254cb4562d7e0ad07b7dd`; branch `phase6-waypoint-validation`.
LightNav checkpoint/processor/RVQ remain
`LightOriginsHQ/LightNav-0@826dc5fbfa37afa8293d2e336d329b6ffc0bfb64`.
All source references below are pinned to
`3015508b70fb6e30bbd66701e94c9b916dcb8df8`. Source was inspected, not launched;
no robot client or motion transport is used.

## Contract established from code

- [traj_vocab.py](https://github.com/lightorigins/LightNav-0/blob/3015508b70fb6e30bbd66701e94c9b916dcb8df8/src/lightnav/traj_vocab.py),
  `compose_to_abs` and `RVQBundle.decode_waypoints`: the stored `se2_diff` codewords
  are composed by the decoder into **ten absolute planar poses in the chunk-start
  ego frame**. Do not compose or sum the decoded poses again. The origin is not
  stored. Row zero is the first future pose, not a mandatory origin/anchor.
- [tracking.py](https://github.com/lightorigins/LightNav-0/blob/3015508b70fb6e30bbd66701e94c9b916dcb8df8/src/lightnav/tracking.py):
  translation is metres, x forward, y left; yaw is heading in radians,
  counterclockwise positive, relative to the chunk-start heading. It is an
  orientation prediction, not a requirement that yaw equal the XY path tangent.
  `compose_to_abs` numerically wraps to `[-pi, pi)` (its docstring says otherwise
  at the exact boundary). Controllers must use shortest wrapped angle differences.
- `RVQBundle.is_stop` uses the manifest's **level-zero code 6**. The pinned
  `TrackingAgent` returns an exactly zero whole chunk for that code, or if the
  entire decoded chunk is within `5e-3` of zero. A near-zero first point alone is
  not STOP. Preserve explicit-code and decoder-zero evidence separately.
- [serving/protocol.py](https://github.com/lightorigins/LightNav-0/blob/3015508b70fb6e30bbd66701e94c9b916dcb8df8/src/lightnav/serving/protocol.py):
  the native response's top-level stop comes from the whole decoded chunk. An
  APOS stop directive (`1299`) is a separate pointing signal and does not override
  the action stop. Preserve both, including contradictory predictions.
- There are no timestamps in the decoded chunk. Native image sampling is 4 Hz;
  this is **not** a waypoint execution clock. The external MuJoCo controller
  declares a 10 Hz control rate, 0.1 s MPC step, five-reference horizon, 0.8 m/s
  navigation speed and 3 rad/s angular limit. `WAYPOINT_DT_S=0.1` is declared in
  that module; reference selection itself is pose-based, not elapsed-index-based.

## Reference execution and comparison with Phase 6C

The pinned [MuJoCo MPC](https://github.com/lightorigins/LightNav-0/blob/3015508b70fb6e30bbd66701e94c9b916dcb8df8/mujoco_demo/vln_mujoco/mpc.py)
projects every row using one capture-time pose, finds the nearest weighted pose
in that world path, chooses the following five rows (repeating the last as
necessary), unwraps their headings, and tracks them with a constrained unicycle
MPC. Its objective/limits include accelerations. The simpler `control.py` helper
uses a spatial target at least 0.35 m away, but the current demo uses the MPC.
Neither implementation reanchors every row to the moving robot.

The [Habitat policy](https://github.com/lightorigins/LightNav-0/blob/3015508b70fb6e30bbd66701e94c9b916dcb8df8/src/lightnav/habitat/policy.py)
instead chooses the first velocity-relevant row, skips leading origin rows,
drops lateral displacement, and divides forward/yaw by the environment-supplied
step duration. Thus upstream has multiple execution policies; no single timing
or controller is intrinsic to the checkpoint.

Phase 6C transforms its first row correctly through the decision-time simulator
pose, retains lateral motion for its declared holonomic camera proxy, and applies
it over an external 0.25 s. The source audit establishes **no decoder, axis, sign,
or absolute-versus-incremental defect** in that path. It ignores later rows, so a
nearly stationary first row can conceal later motion. That is a policy limitation,
not evidence that the retained 16-step failure was caused by an implementation bug.
The original baseline script, scene, outputs and defaults remain intact.

## One additional diagnostic policy, selected before model results

Add `bounded_prefix`: traverse the decoded poses in order, with the origin frozen
at the decision-time simulator pose. Interpolate each segment in XY and shortest
yaw, with duration `max(distance / 0.8 m/s, abs(yaw_delta) / 3 rad/s)`. Execute at
most 0.25 s before the next observation/decision, checking collisions at <=0.01 s
substeps. Skip truly zero segments, never convert a leading zero to STOP, and hold
at a finished chunk for any unused window time. Record selected prefix, target
index, requested motion, bounded motion, and constraint interruption separately.

This is a deterministic, speed-bounded **holonomic kinematic proxy** path follower,
not the upstream unicycle MPC, Go2 dynamics, or learned avoidance. It adopts the
reference's fixed capture frame, multi-row path and navigation rate limits without
adding CasADi or replacing the existing simulator. It has no acceleration model.
It does not inspect evaluator goals or repair a bad model path. The 0.25 s replan
window is explicitly our simulator choice, not inferred trajectory timing.

The `first_waypoint` policy remains the Phase 6C baseline: 0.25 m / 32 degrees
per 0.25 s, rejection outside bounds. New diagnostics will expose its requested
and executed path without changing its historical interpretation.

## Frames, camera and streaming

Compose only through the known synthetic proxy pose. Real `T_base_camera` remains
missing. The original MuJoCo camera has body offset `[0.16, 0, 0.4]` above a proxy
at z=0.25 m; it looks forward along body +X, right along body -Y and up along +Z.
MuJoCo camera axes are right/up/back; RGB optical axes are right/down/forward.
Synthetic pinhole frames go straight to the official image processor. No measured
Go2 fisheye rectification applies to them. Numeric and rendered checks follow.

The native `NavigationPolicy` owns episode history and vision caches; reset clears
them and engine episode state. HF generation starts its own conversation each call;
no prior proposed/executed action is a required input. Phase 6C deliberately reset
and rebuilt all causal images at each decision. Phase 6D's opt-in streaming mode
will reset only at episode/clock-epoch boundaries, append newly rendered 4 Hz
observations and preserve native history selection. The old replay mode stays intact.

## Validation and independent visual review

Claude's published [human review at 4a168e1](https://github.com/binbin-ovo/Go2_Navigation/blob/4a168e165e8eee58b7fcd4fc5a192e9791d3c278/docs/PHASE6D_BEHAVIOR_REVIEW.md)
was read before freezing the next experiment. Its 18-file local manifest verifies.
The owner recognized the original doorway at steps 0 and 2; the saved execution
matches the first waypoint exactly. Missed visible targets and an incorrect positive
target point are human-supported findings. This supports keeping the original scene;
it does not establish broad domain suitability or approve every new camera pose.
The near-goal preview crops much of the doorway frame; retain that limitation when
interpreting its stopping probe. New-run behavior has not been human-reviewed.

`audit_lightnav_repetition.py` inspected all 16 retained native tensor archives:
all 16 current input images and pixel tensors differ. Steps 1–8 have eight distinct
pixel tensors despite one repeated output. Native SlowFast selection retains current
observations. The pinned engine's HF `generate` branch processes each sample anew;
the episode ViT cache path is in the vLLM branch. Thus this evidence does not support
stale images or that cache as the cause. It does not explain the model's semantic
decision boundary or prove every upstream operation correct.

Model-free execution: eight successful known-path/stop cases, one correctly
interrupted wall-contact case, and one reachable-goal diagnostic. Endpoint error
is <=4.45e-16 in metres/radians; maximum cross-track error <=1.12e-16 m, maximum
heading error <=3.61e-16 rad. These are kinematic arithmetic checks, not dynamic
tracking accuracy. Rendered colored landmarks agree with pinhole projection before
and after proxy motion to within the 2-pixel test tolerance; image right/up and
camera-follow geometry pass. Four actual native-processor previews have zero pixel
difference from the already-native-sized 448×256 RGB input. No Go2 rectification.

Full core regression: 590 passed, 9 skipped. Optional simulator/native checks:
18 passed before a Transformers file-read stall interrupted the last test; the
remaining CPU/MPS pooling test passed separately. No existing test was weakened.
One restricted-sandbox run failed two ffmpeg/ffprobe tests on OpenMP shared-memory
access; the full suite passes outside that sandbox. Logs and exact commands are in
[PHASE6D_NAVIGATION_EXPERIMENTS](PHASE6D_NAVIGATION_EXPERIMENTS.md).

The additional policy changes both path consumption and rate limits (0.8 m/s,
3 rad/s versus the baseline's effective 1 m/s and 2.234 rad/s caps). A between-policy
outcome difference cannot isolate either factor. No tuning on model outcomes is
allowed within the frozen suite.
