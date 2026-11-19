# Supervised navigation contracts

## Version 2 for the effective Go2 profile

The opt-in `go2-effective-v1` profile uses the v2 cycle schema on
`feature/vln-effective-motion`. The one-shot executor imports the audited control loop at
`6d555d31c6eaa4ee194920925ab96238bf15b128`. Version 1 still describes the original
first-row records and synthetic fixture. An archive declares its version in `schema`.
The launcher requires its explicit profile to match the capability. [Version 2 schema](../src/go2_perception/vln/schemas/supervised-navigation-v2.schema.json)

### From native path to one bounded primitive

Each LightNav row is an **absolute** local pose in the same decision frame, in metres
and radians, x forward, y left and counterclockwise yaw. Ten rows have no intrinsic
execution time. The preview and refreshed proposal are planned independently by
the same function. The child reproduces the refreshed plan before claiming control.
Native STOP and an absent target point both end the attempt.

The recorded forward mapping rotates by 0.02941 rad with angular uncertainty
0.05236 rad. Translation remains null with a 0.50 m bound. For native translation
`p` and heading `ψ`, the worst-case positional slack is
`2‖p‖sin(δθ/2) + 2(0.50 m)|sin(ψ/2)|`. The second term accounts for the unknown
origin in planar displacement conjugation. The projected forward distance uses
the lower bound `x − slack`. Full camera/body registration and metric model scale
remain unresolved.

The planner skips leading near-identity poses (5 mm and 0.005 rad), then walks
the rows in order from the original decision pose. Forward motion requires
positive conservative progress, lateral offset plus slack ≤0.08 m, absolute heading
≤0.10 rad, and a local lateral slope ≤15° with 5 mm jitter allowance. A reversal,
bend or incompatible row ends the prefix. The maximum accepted endpoint is
0.60 m, further limited by measured forward clearance after the stopping reserve.
`selected_row` is the last checked bracket row. `segment.end_fraction` locates
the endpoint within its scalar progress interval. The selected straight body
primitive has zero lateral speed and zero yaw rate. All native rows survive
independently of this approximation.

For a turning prefix, the planar rotation in `T P T⁻¹` is always `ψ` regardless
of the unknown origin. V2 records this narrow basis as `relative_planar_heading_ccw`
alongside `verified_actions=["forward"]`. It follows only the sign-consistent
heading of the contiguous native prefix, capped at 30°. The visible target must
be on the corresponding side of the image, outside a 2.5% centre deadband.
The image can veto a turn. Its pixel offset supplies no physical angle. Reversing,
sideways or changing-intent segments end the prefix. All predicted travel is
deferred, including the displacement residual and its mapping uncertainty.
Smaller terminal requests stay in the archive. The pilot admission floors of
0.30 m and 20° prevent low-amplitude retries or inflation. These are engineering
thresholds. The fast hardware reference conditions were 0.60 m and 30°.

A turn-directed native prefix in (0.10 rad, 20°) produces a
`below_effective_resolution: heading_gap` HOLD with its original direction, angle
and limiting boundary. Keep the full native chunk. The operator may review a
fresh stationary observation. Only a same-direction contiguous native extension
can support a larger angle under the existing clearance and inspection policy.

### Command and stop contract

| Primitive | Command | Max destination | Send zero at measured destination minus | Clearance beyond destination | Settled scoring tolerance | Measurement noise band |
|---|---:|---:|---:|---:|---:|---:|
| Forward | 1.0 m/s | 0.60 m | 0.10 m | 0.25 m | ±0.10 m | ±0.02 m |
| Right | −2.0 rad/s | 30° | 4° | 8° | ±6° | ±2° |
| Left | +2.0 rad/s | 30° | 10° | 12° | ±6° | ±2° |

The envelope stores a positive magnitude and separate direction. Wire angles use
radians. All three commands have a 3 s maximum drive duration and 0.05 s resend
period. Faster commands require a positive audited distance or heading target.
The child uses the adjusted trigger as that target, and keeps the intended destination
for scoring. Forward progress comes from odometry and consistent front-surface
LiDAR. Effective turn progress uses wrapped IMU yaw in the commanded direction,
with a fresh, same-epoch odometry check. Forward clearance is recomputed in the
current body frame. It must cover the remaining distance, the reserve, and
command speed × current geometry age. Forward lateral/heading drift outside
the reviewed corridor interrupts execution. Stalls, expired deadlines, unavailable
feedback and operator intervention end the velocity phase directly.

A zero send starts settling. The record gives the last fresh feedback sample around
first zero with both timestamps, sampled peak, and a new settled sample after
stationarity, release RPC and the three-second post-release observation.
`motion_measurement.outcome` scores the settled sample against the intended
destination using `target_within_tolerance`, `partial_progress`,
`no_measurable_progress`, `overshoot`, `wrong_direction` or
`measurement_unavailable`. `termination`, `interrupted_or_blocked`, and
`stop_verified` are independent. A stalled or off-tolerance run with verified
cleanup has status `ended`. A sent zero has no robot ACK on the native no-reply route.

### Motion diagnostic records

All paths below are relative to the private trial’s `session/` directory. The
executor and bridge subscribe independently. Their callback receipt times can
differ for the same robot header stamp.

| File | Evidence |
|---|---|
| `executor/executor_events.jsonl` | The terminal `type=result` contains `result.events`. A `guard_rejected` event freezes the guard object checked at the failing iteration, its request ID, stage, check time, nonzero send count, projected clearance progress, exception reason and numerical details. The cached odometry includes raw orientation when received. `guards.geometry_detail` contains the provider’s floor/corridor evidence when available. A provider exception before a snapshot leaves a null guard. |
| Same result, `motion_feedback` | Each driving, settling or settled sample has a host-monotonic time, value and unit. Forward `basis` retains reference/current odometry (XYZ and quaternion xyzw when provided, yaw, frame, robot/receipt stamps), contributing scan/window stamps and the planar poses used for compensation, start/current ground z and front surface x, signed `odom_m`, `lidar_m`, `fused_m`, `lidar_accepted`, `fusion_decision` and `selected_input`. The first-zero event identifies the feedback sample available at that send. |
| `events.jsonl` | `motion_lidar_window` links the executor request to the refresh model trial. Recording spans dispatch through the matching terminal result, including first zero, stationary dwell, release and the post-release observation period. If cleanup is unverified or the executor disappears, the raw recorder stays active until session shutdown. Its closing reason identifies missing or unverified evidence. |
| `lidar/index.jsonl`, `lidar/chunk_*.npz` | Per-trial/epoch chunk, source and host intervals, counts, hash and dropped-data flags. Raw `points_xyz_base_link` split at `scan_offsets`, scan/odom sequence numbers and robot stamps, raw odometry position and quaternion, wall receipts and offsets survive. `scan_host_recv_monotonic_s` and `odom_host_recv_monotonic_s` are bridge wall receipt minus its sampled wall-minus-monotonic offset, in seconds. |
| `summary.json` | `lidar_recording` reports chunks written/dropped, writer errors, queue/size limits and `drain_complete`. Check bridge gap counts and source stamp coverage as well. |

Forward odometry is displacement projected onto the initial heading in odom.
LiDAR progress is reference minus current front-surface range in the body frame.
The pinned controller accepts two available estimates within 0.15 m + 30% of
absolute odometry and returns their maximum. A rejected disagreement selects
odometry. A missing surface is distinguished from disagreement. Missing/stale
odometry or a changed epoch can yield an unavailable sample. A null value stays
unknown. Guard window robot stamps identify scans usable for an offline re-fit.

The recorder uses a separate writer with four queued chunks, one filling chunk
and one being written. Each holds at most one second of arrivals, 4 MiB of raw
point payload and 512 scan/odom messages. The cap excludes compression buffers
and bounded per-message metadata. The bridge reader offers chunks without waiting
for disk or a free queue slot. Oversize scans, writer backpressure and write failures
produce gap records. During a writer stall the recorder retains 64 detailed gap rows
plus one summary with exact dropped counts and an enclosing host interval. Mixed
epochs/trials lose their individual attribution in that summary. Compare intervals
only within matching streams/boots. An interrupted process or an index write error
can leave the archive incomplete. Guard and feedback events are persisted with the
executor result after cleanup.

### Turn scope and post-turn replan

The ordinary `sensor_only` scope requires measured clearance in the requested
direction. Claude's current cloud provider reports zero turn coverage. For a
static supervised experiment only, `operator_inspected_static_disc` requires
the operator to inspect the entire 0.65 m radius around the body, the blind rear,
legs and cable slack. The radius includes a 0.10 m drift allowance on the
0.55 m footprint used in the control trials. The separate viewer button binds
operator, session, reviewed image/identity, turn direction, pose, epoch, radius
and inspection time for at most 30 s. Authorization still expires within 15 s.
The ordinary fresh state and forward geometry must remain usable, with ≥0.10 m
forward clearance. The observed disc needs floor returns and zero obstacle-height
returns. An available restrictive turn-sector measurement vetoes an inspection.
This partial sensor check cannot certify empty space behind the robot.

After verified stop, release and a newer capture in the same epochs, the app
opens one fresh independent RGB/instruction request for an advisory replan. Every
image's capture lower bound must follow the end of the stopped cycle. The executor
is already spent. The post-turn native proposal, selected candidate or refusal
and `execution_authorized=false` are retained in `navigation/post_turn_advisory.json`.
Another physical movement requires a new session, target review and authorization.

## Historical version 1

Canonical branch: `feature/vln-supervised-navigation`. This is the shared software
interface for Codex and Claude, not permission to move a robot. Schema:
[`supervised-navigation-v1.schema.json`](../src/go2_perception/vln/schemas/supervised-navigation-v1.schema.json).
Fixture: [`synthetic_cycle.json`](../examples/supervised_navigation/synthetic_cycle.json).
The fixture describes an invented robot, scene, registration and capabilities;
**none is Go2 hardware evidence**. It must be rejected by live execution.

## Ownership and publication

Codex owns this versioned contract, planner, approval state machine, isolated
executor interface and mock verification. Claude owns the hardware evidence,
hardware guard integration and physical trial. Additive optional fields may be
introduced in v1; changing units, semantics or required fields needs a coordinated
new version. Preserve raw model output, earlier evidence and all refusal records.

Inspected upstream versions: interactive `fdb5451e25cdc1f5971bbf0e1c7887ec51ff634a`,
control `742cfe4fb1b71f9d19d9d2d62be56f03b6b03707`, paired-view evaluator
`1a0ac0e9f2cc0ca2688bbd7651383c90e86251fb`. Main is not an integration target.

## Message and evidence identities

The schema defines capability, proposal, authorization, execution result and
post-observation records, collected in a cycle for examples/archives. A proposal
binds session, trial, instruction ID **and exact text**, prediction, observation,
source image SHA-256, source clock epoch, robot clock epoch, observation/receipt
monotonic times, uncertainty bound, pose and native output. Do not use a target
word or OPOS/APOS token as a physical object ID. Preserve the original prediction
archive and model/checkpoint/processor identity.

All `_monotonic_s` values use the same computer/boot monotonic clock, seconds.
They are not comparable across computers or boot epochs. Wall time is used only
for evidence expiry/audit. Unknown timing uncertainty is null, never zero.
Execution requires a justified upper bound on observation age, including timing
uncertainty, at most the unchanged **3.0 seconds**. A receipt timestamp alone does
not establish that bound. Poses are metres/radians in a declared local frame and
robot epoch; comparisons across frames/epochs are refused.

## Action mapping and restricted follower

LightNav outputs ten **absolute** SE(2) poses in one fixed decision frame, x
forward, y left, yaw heading counterclockwise in radians, translation in metres.
There is no native time base. Four-Hz image sampling is not command timing.
Never sum rows as displacements or move the origin between rows. Native decoder
STOP is separate from a small first row or a task-completion/pointing token.

A mapping record must validate the rigid planar displacement convention for each
allowed action. Its translation/yaw describe `T_base_model`; displacement mapping
is `T_base_model @ P_native @ inverse(T_base_model)`. This conjugation assumes
rigidly attached planar frames and must be demonstrated by Claude; it is **not**
an invented physical camera transform. Missing registration is unavailable,
including when a convenient identity would produce a plausible preview.

The v1 follower skips only near-identity rows (5e-3 m/rad), selects the first
meaningful mapped row, and either takes a bounded forward segment (small declared
lateral/heading residual) or an in-place turn (small declared translation
residual). Other combinations HOLD. Keep the full raw chunk, selected row, ignored
residuals, proposed primitive and bounded request distinct. No reverse, lateral,
combined motion or hidden goal-directed replacement. Bounds and tolerances come
from validated capability evidence, not simulator constants. Both magnitude
target and wall-clock duration terminate the action; neither guarantees exact
physical travel or a safe stopping distance.

## Preview, authorization and one refresh

Preview may age visibly. The operator identifies a physical target instance and
selects its ROI on the exact reviewed source image. Authorization binds that
image/ROI, target instance, instruction/text, pose/frame/epochs, direction,
maximum magnitude/rate/duration, capability hash and mapping. It expires after the
declared short TTL (maximum 15 seconds). It is consumed **once when refresh starts**,
including on failure/cancel; duplicates never retry motion.

After authorization, collect a new independent short observation and make ONE
new model request with the existing worker/preprocessing. All refresh images
must be acquired after authorization. Preserve the existing independent
short-observation policy; there is no cross-query model memory claim. A refresh
must have new observation/prediction source identities, identical instruction,
same epochs, and pose drift within explicit evidence tolerances. A fresh primitive
must have the same type/direction and no greater magnitude, speed or duration.
Changed geometry/target, STOP, stale result, missing bound or failed refresh HOLD;
no automatic retries. Identical pixels may have the same content hash; new source
frame IDs and acquisition times, not a changed hash, establish a new observation.
The human may begin another reviewed cycle explicitly.

Same-target comparison requires the reviewed instance/ROI plus independent image
correspondence at the nearly unchanged pose, and the fresh model point within
the corresponding ROI. The initial implementation uses conservative local
feature registration, records its evidence and refuses insufficient/ambiguous
support. This is a visual continuity gate, not a semantic identity guarantee.
Hardware use requires Claude's validation of this gate in the trial conditions.
A matching word, point token or server-copied instance ID alone cannot pass it.

## Capability and current guards

Capability evidence names exact control/source/configuration versions, expiry,
validated mapping and per-action physical limits. Required evidence covers
frame mapping, independent intervention, obstacle handling, stop verification,
motion bounds and target correspondence. Unknown or unverified entries refuse
live arming. Report onboard avoidance enabled both before and after; never switch
it off or fall back to SportClient/another route.

The executor obtains current pose, state age, operator intervention, geometry and
measured progress through a hardware-environment guard provider. Its file hash
is pinned in the capability. Geometry must describe observed/verified clearance
for the actual action, with age, frame and scope; absent returns are not free
space. The provider must independently acquire/validate it, not echo browser or
model assertions. The executor rechecks it during movement. A static evidence
file is not a live state feed. Missing provider/geometry means HOLD. Claude must
integrate and validate the provider using the verified sensor environment; the
synthetic provider is deliberately unavailable in live mode.

## Process boundary and canonical lock

Model/browser and executor are separate processes/environments. Only the explicitly
armed executor may lazily import the audited hardware SDK. It reuses
`go2_control.safety.execute` and `AvoidRoute` from the pinned canonical control
checkout. It owns cadence, deadlines, fresh state, cancel handling, stationary
measurement and ownership release. An inference hang must not block those duties.

Set ONE canonical control checkout for the entire trial. Reuse its existing
`data/outputs/go2_control/.command.lock` (O_EXCL), not a second worktree's lock.
Every commanding process used in the trial must use that same checkout/lock.
The existing audited `stop` subcommand does not take this lock: do not run it as a
competing process; use the running executor's cancel/stop channel and the verified
independent intervention procedure. Do not run old utilities from other worktrees.
A file lock cannot prevent an unrelated SDK client from commanding the robot;
exclusive-source confirmation and intervention evidence remain necessary.

Live startup requires all of: explicit live launcher opt-in, separate hardware
Python/canonical checkout, verified evidence and pinned guard provider, exclusive
command-source confirmation, a session nonce, and fresh single-use operator
authorization. An evidence file alone cannot arm execution. No ordinary test may
initialize DDS or import `sdk_routes`.

## Results, stop and next observation

Record command local-send results, robot acknowledgement, measured movement,
stop-send results, observed stationary (fresh measurements over a declared dwell),
release RPC result and post-release stationary status separately. Audited
`Move`/zero use `_CallNoReply`: robot acknowledgement is **null/unknown**, even
when local send returns zero. Never relabel that status as acknowledged.

Cancel before dispatch discards the one-use authorization without a robot command.
Cancel during execution requests bounded stop/release through the independent
executor. UI cancellation is not proof of a physical stop. Failures retain
unknown/false stop and release status, lock the cycle against further movement,
and require the independently verified intervention procedure.

After stop/release, obtain a strictly newer observation in the same declared
epochs. Invalidate all earlier proposals/authorizations. No second movement is
automatic. A successful software cycle means one bounded action, separately
verified stop/release and a new observation; it is not navigation success.

HOLD records include concrete reasons such as missing/unverified mapping or
capability, expired evidence/auth, missing geometry, stale state/observation,
unknown capture-age bound, changed target/pose/instruction/epoch, unsupported
primitive, duplicate authorization, worker error, executor failure, unverified
stop/release or missing post-motion observation. Preserve these as evidence.
