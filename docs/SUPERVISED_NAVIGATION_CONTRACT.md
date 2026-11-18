# Supervised navigation contract v1

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
