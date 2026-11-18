## Offline paired-view evaluator → Claude / next agent (2026-11-17)

Branch **`feature/vln-viewpoint-evaluation`**, isolated worktree, exact base
`73c83f19cf83ee01640e467e09085dd669e51763` from the published interactive-review
branch. Main/defaults and Claude's active worktree are unchanged. This is an
optional **post-collection** analysis tool; it does not block or operate the demo.

Read [VLN_VIEWPOINT_EVALUATION](VLN_VIEWPOINT_EVALUATION.md) first, then the newer
browser-guided workflow and stationary `live_08` report in
[VLN_INTERACTIVE_REVIEW](VLN_INTERACTIVE_REVIEW.md). The older terminal description
below is historical. Browser Ask starts a fresh trial/revision/history and makes one
request; a before/after experiment consists of two independent queries. Do not call
it continuous navigation, retained memory, autonomous search or navigation success.

Implemented `scripts/analyze_vln_viewpoint_trial.py`, the read-only archive reader
and static HTML/JSON/timeline report in `vln/viewpoint*.py`, deterministic synthetic
archive/annotation under `tests/fixtures/vln_viewpoint/`, its generator, and focused
tests. Use existing isolated core Python; no dependency installation, model, GPU,
receiver, DDS, control client or physical command is needed. Exact commands and the
evaluation-only sidecar contract are in the guide.

Inputs: one **closed** interactive run (`summary.json` plus both trial ends), explicit
before/after trial IDs, optionally exact prediction IDs and an evaluation-only pair
sidecar. Use the same saved target/instruction. A fresh Git-ignored output directory
receives `index.html`, `summary.json`, `timeline.json`, and image copies. Consumed
source files are hash-checked for concurrent changes; originals are never modified.
Do not analyze an archive still being written or reopened for review. Missing or
intentionally deleted inputs remain missing and are reported, never reconstructed.

Verified **69 passed / 0 failed / 0 skipped in 2.89 s**: 62 new CPU-only tests and
seven existing mocked interactive contracts. Ruff/diff checks pass. Static report
links/images resolve; browser visual appearance was not independently verified.
The synthetic example produces wrong→correct in its two reviewed views relative to
its synthetic turn annotation; this is not model performance. The six completed
stationary `live_08` trials (six predictions, six guided reviews) pass schema/input
compatibility checks; 128 consumed files unchanged. No private images/comments are
published. Full regression and inference were deliberately not run during the
current lightweight-only window; hardware tests were not selected and opt-ins were
disabled.

Important evidence limits: each mark belongs to its own selected native image;
arrival after a turn does not prove post-turn capture. Preserve original clock
domains/epochs and unknown bounds. Current live exposure mappings provide point
estimates without a bounded capture error, so turn verification can remain incomplete
even with a valid reviewed pair. Never substitute zero uncertainty or calibration
sigma as a guaranteed bound. Receipt freshness and review age are separate. Guided
target-only labels do not review path/stopping/navigation. Group reviews are never
distributed across predictions, and conflicting/repeated reviews remain distinct.
All native paths are **UNEXECUTED**, without real metric registration.

**Pending / next step:** Claude completes the operator-guided before/turn/stop/after
recording and identifies the two trials and available annotation provenance. Close
the run, then run the documented command locally. If no reliable turn timing is
available, omit it and report only reviewed-image change with timing unverified.
No suitable explicitly linked completed pair was available at implementation time;
`live_08` is stationary compatibility evidence only. Stop at this scope and wait for
the real pair; do not start another model experiment or change the live website.

## Interactive four-panel review tool → next agent (2026-11-16)

Branch **`feature/vln-interactive-review`** (worktree
`/Users/yushan/Desktop/Go2_Navigation-vln-interactive`), exact base `1f93d7e` from
`feature/vln-cached-live-validation`. Main and the motion-control PR are not merged or
imported. Read [VLN_INTERACTIVE_REVIEW](VLN_INTERACTIVE_REVIEW.md) first.

**What it is:** `scripts/run_vln_interactive.py` is a terminal plus a loopback-only browser
page with (A) the live camera, (B) the native target point on the exact model input,
(C) all native waypoints and headings in the model-local frame, and (D) a LiDAR top-down
view (latest / 0.5 s / 1.0 s, odometry-compensated `cloud_base`). Commands: `target`,
`instruction`, `mark`, `note`, `end`, `review`, `status`, `save`, `quit`, `help`. Each
trial is saved to a private, append-only run directory (images once by hash, LiDAR
chunks with source intervals, native outputs, marks and notes, confirmed human reviews
and click corrections in separate files). Reopen a run with
`--review-run <run>`. The model path is unchanged: existing `ShadowSession` +
`ProcessWorker`, full-history policy, cached preparation and the unchanged 3 s limit.
The bounded-context policy is **not** enabled. New opt-out: `--tensor-evidence skip`
(cached mode only; the default here; `run_vln_shadow.py` defaults unchanged).
LiDAR/odom come from a new read-only DDS-subscriber child
(`scripts/go2_lidar_bridge.py`, hardware env); robot-clock epochs and the per-epoch
host−robot estimate are display-only. There is no RGB–LiDAR fusion.

**Verified (no hardware, no model):** synthetic camera and LiDAR, a recorded public
LiDAR fixture and one private recorded episode, all with the mock worker. 22 new tests;
full core **702 passed / 0 failed / 11 environment skips**. App RSS over six short
synthetic trials: 98 → 111 → 112.5 MiB (flat after the first). Robot commands sent: 0.

**Live (2026-11-17, read-only, Go2 standing and stationary, operator-confirmed):** session
`live_03`, 4 trials and 19 LightNav requests (3 CURRENT, 12 STALE, 4 SUPERSEDED); camera
~7.1 Hz and `cloud_base` ~15.4 Hz with 0 gaps; app RSS ≤ 220 MiB; worker MPS driver memory
grew 8.9 → 11.8 GiB; disk ~0.45 GB/trial; 0 robot commands. Two operator-confirmed reviews:
the door target was correct; the chair target was wrong (it pointed at the shoe rack). The
bottle trial gave OPOS-0 and is unreviewed. Live launch needs `--unitree-sdk` and must run
from a process with macOS Local Network access (the app's terminal panel received nothing;
see the doc). Never connect native waypoints to any Go2 command.

## Cached preparation, live verification → next agent (2026-11-16)

Branch **`feature/vln-cached-live-validation`** (worktree
`/Users/yushan/Desktop/Go2_Navigation-vln-cached`), exact base `d173082` from
`feature/vln-live-latency`; docs only, no code change. Read
[VLN_CACHED_LIVE_VALIDATION](VLN_CACHED_LIVE_VALIDATION.md).

**Verified on hardware (stationary, read-only, operator-confirmed):** one bounded
`--preparation-mode cached` session, 3 requests, 0 robot commands. Jobs of 5/19/39
frames: prep 0.19/0.52/1.16 s, generate 3.0/4.0/6.5 s, estimated exposure→display
3.65/5.03/8.54 s — **all STALE** under the unchanged 3 s limit. Cache hits only 5 per
request (latest-anchored resampling), so live preparation is about the same as baseline;
no controlled speedup claimed. 0 inbox overflow/gaps/RTP loss; tensor archives 3/3
complete, normal shutdown. Display keeps latest-camera and model-input stamps separate.
Operator judged doorway targets (opos_501/502/504) correct — qualitative only.
Targeted tests 92 passed.

**Next:** freshness remains unresolved; native generation growing with history is the
dominant cost. Any further change (history policy, feature caching) is a separate,
explicitly approved decision. Do not connect waypoints to any Go2 command.

## Advisory latency → Claude / next agent (2026-11-16)

Branch **`feature/vln-live-latency`**, worktree
`/Users/yushan/Desktop/Go2_Navigation/data/outputs/vln-live-latency-worktree`, exact base
`eaec3314f4ed933623bd8aad1daca944d4575363` from `feature/vln-live-shadow`. Read
[VLN_LIVE_LATENCY](VLN_LIVE_LATENCY.md) first for contracts, all results, commands and
limitations; then [Claude's live validation](VLN_LIVE_SHADOW_VALIDATION.md).
Main, operational defaults, hardware/calibration environments and earlier results
are preserved. The hardware-control PR is not merged/imported.

Implemented **opt-in `--preparation-mode cached`**: hash-verified lossless calibrated
PNG reuse (128 entries / 256 MiB of cache links), compression level zero, and a
single bounded background tensor archive. Native model-space PNGs/text/prompt/decoder
outputs remain synchronous. Snapshots retain calibration and original timestamps;
archive status/errors and shutdown completion are explicit. Default `baseline`,
three-second threshold, receiver/stride/inbox, history policy, checkpoint/processor,
MPS BF16/SDPA and the same-dtype CPU pooling patch are unchanged.

**Actual verification:** after explicit operator resource coordination, six model
requests ran sequentially: recorded run 3's 5/17/37-frame jobs, baseline once then
cached once. All input tensors (BF16 bits included), history metadata, generated IDs,
pointing, waypoints and STOP flags match exactly; outputs also match the original
live results. Submit → display improved **2.942→2.779 / 4.358→4.005 / 8.305→7.388 s**.
These are fixed-cutoff offline profiles, not measured live latency. Fast replay
accepted one/rejected two in each pass; adding the original estimated exposure-to-
submit offsets gives 0/3 vs 1/3 under three seconds. Longer histories remain stale.
One sample/length/mode supports no percentile or robust variability claim. All
408 source-run files remained unchanged. Writer: 3/3 archives complete, no losses.

**Final tests:** 683 core passes / 0 failures / 9 skips in 42.29 s; 17 separate
native/numeric passes / 0 failures / 2 rendering skips in 2.62 s. The 20 new latency
tests cover invalidation, equivalent inputs, changing sampling prefixes, reset/
cancellation/staleness, bounded writing and shutdown/error reporting. Hardware opt-ins
were disabled and the physical test confirmed skipped before execution. No Go2
connection, DDS, ownership, stop or other physical command was used.

**Private artifacts:**
`/Users/yushan/Desktop/Go2_Navigation/data/outputs/vln_live_latency/profile_20261116_01/`.
`plan.json` predates inference; `baseline/` and `cached/` hold all requests and local
review pages; `comparison.json` has exact equivalence and timing; final test logs/XML
are `full_regression_final.*` and `optional_regression.*`. Initial fixture failure
and earlier passing suite are retained. The two `reconstructed_*_source/` directories
match every pre-inference source hash and permit exact source reconstruction on the
verified base; see their provenance notes. Existing isolated environments were reused;
the ignored `.venv` here only links to core-env. No installation or new weights.

**Next:** Claude may run the report's prepared three-request stationary **read-only**
command after a new operator-coordinated window. It explicitly enables cached mode;
this optimized path has not been tested live. Measure actual arrival/queue pressure,
archive completion and age; do not raise the threshold or connect outputs to control.
The pinned HF backend does not use its vLLM feature-cache path. Latest-anchored
sampling changes prior selections, so simply appending native frames is invalid.
No history truncation/reset shortcut was introduced. Real extrinsics, exposure
uncertainty and near-goal stopping limitations remain unresolved.

## Previous milestone: live read-only VLN advisory (2026-11-16)

Branch `feature/vln-live-shadow` (worktree `/Users/yushan/Desktop/Go2_Navigation-vln-live`),
based on advisory commit `b47f07d`, not main; the control PR was not merged or imported.
The existing read-only RTP camera receiver now feeds the advisory app through a
hardware-env child (`scripts/go2_camera_bridge.py`) and `vln/shadow_live.py`, which
produce the existing `Go2VLNObservation` for `ObservationInbox`. Launch with
`run_vln_shadow.py --go2-camera`. RGB-only: host-receipt stamps, estimated exposure
(receipt − 48 ms, single-boot), pose/LiDAR/`T_base_camera` None. Zero robot commands;
no DDS/SDK loaded.

**Verified on hardware (robot stationary, read-only):** real frames 720×1280 RGB with
measured fisheye calibration, unique IDs, 14.3 Hz decoded / 7.1 Hz forwarded, 0 RTP loss.
Three bounded LightNav sessions (model load 44.6–44.7 s). Sensor path ≈65 ms (48 ms of it
assumed); native generation 2.3–6.9 s, growing with instruction history. One request
was current at receipt (2.959 s); in steady state every proposal is **stale** under the
unchanged 3 s limit. Instruction change, cancel, history limit and clean shutdown observed
live; clock-step and interruption with mocks. Targets for doorway/refrigerator look
judged correct by the operator (qualitative, one scene). Tests: 15 new mocked tests; core 663 passed / 9 skipped.
Details: [VLN_LIVE_SHADOW_VALIDATION](VLN_LIVE_SHADOW_VALIDATION.md).

Next agent: do not connect native waypoints to any Go2 command. Open questions are
freshness (generation grows with full-history rebuild; a separate decision, not started),
re-measuring camera latency on the current boot, and broader (multi-scene) qualitative review.
Private runs: `/Users/yushan/Desktop/Go2_Navigation/data/outputs/vln_live/` (git-ignored).

## Read-only VLN advisory → Claude / next agent (2026-11-15)

Branch **`feature/vln-shadow-demo`**, worktree
`/Users/yushan/Desktop/Go2_Navigation/data/outputs/vln-shadow-worktree`, exact base
`1cc74909e489e1037c9f6f0051ab34e855d144dc`. Use `git rev-parse HEAD` for the publication
commit. Main, previous experiments and operational defaults are preserved.

Read first: [VLN_SHADOW_DEMO](VLN_SHADOW_DEMO.md),
[actual verification](VLN_SHADOW_VERIFICATION.md),
[VLN_CONTROL_HANDOFF](VLN_CONTROL_HANDOFF.md), then the existing
[observation contract](GO2_VLN_INPUT_INTERFACE.md) and
[native contract audit](WAYPOINT_EXECUTION_AUDIT.md).

One entry point: `scripts/run_vln_shadow.py`. It accepts terminal instruction plus
existing `model_input/episode.json` or deterministic synthetic images and creates a
private directory containing refreshing `index.html`, PNG/status/summary, append-only
events, original observation NPZ files, immutable native jobs/snapshots and receipts.
The internal `vln_shadow_worker.py` reuses existing LightNav `run_job` in the isolated
model environment. It loads once, resets/rebuilds the complete causal instruction
history per request, and runs official preprocessing/SlowFast/native decoding.
It receives no annotations, poses, LiDAR, action feedback or goal truth.

Session behavior is in `vln/shadow.py`: disk-backed bounded history, one in-flight
request, one coalesced dirty flag, strict instruction/epoch/generation/sequence and
monotonic-time checks. New instructions clear history and begin with the next actual
frame; no old frame is restamped. Native episode anchors are retained until a limit
or gap explicitly pauses inference. Late cancelled results are historical, never
current. `shadow_display.py` separates the latest camera from actual model inputs,
plots all native waypoints/yaw in their own unregistered local frame, and distinguishes
native STOP, model completion claim and unknown verified/physical completion.

**No physical command path exists.** There is no `--live`, `--execute`, `--arm`,
motion client, avoidance toggle, command publisher or simulator execution backend.
Even zero velocity is not sent. Cancelling means advisory cancellation only.
`shadow_sources.ObservationInbox` is the future read-only consumer; it opens no
connection and preserves the existing `Go2VLNObservation` payload. See the guide for
availability domains, clock-epoch/uncertainty requirements and overflow handling.

**Initial lightweight verification:** 114 selected tests passed in 1.56 s (57 new advisory
cases, 57 existing contracts), no skips in this set. Synthetic paced mocked demos
accepted 3/3 proposals; a deliberately strict age test rejected 2/2 stale proposals.
Private E1 package replay used 24 existing images with calibrated rectification and
two mocked proposals; all files remained local. Ruff/diff checks pass. Tests are
reproduced with core-env Python and `PYTHONPATH=src`, selecting `test_vln_shadow.py`,
`test_pretrained_vln.py`, `test_vln_episodes.py`.

**Operator-authorized verification, completed:** tested implementation
`b3c3dffdaf1a663f554b29d8dc2ca19aa1460e8d` ran one real LightNav replay, then the full
core suite, then the optional native/numeric suite, sequentially. Five causal,
calibrated inputs produced ten valid native waypoints. Native generation took
2.444501 s; the child interval took 2.851785 s; paced observation age at receipt was
3.544899 s. The unchanged 3 s advisory limit correctly rejected it as STALE: zero
fresh accepted, one rejected. These are replay timings, not measured live latency.
No extra real call or relaxed threshold was used. Core: **648 passed, 0 failed,
9 skipped** in 44.73 s. Separate environment: **17 passed, 0 failed, 2 skipped** in
2.68 s. Physical Go2 and two camera-rendering tests remain unexecuted; environment
specific core skips were subsequently covered. See the verification report for
exact selection, skip reasons and commands.

The operator confirmed Go2 stationary and physical testing paused/finished with
stop/release cleanup complete. No Go2 connection, DDS, control utility or robot
command was used. Existing `phase6c/core-env` and `phase6c/model-env` were reused
unchanged. No new model or dependency was installed. The worktree's ignored `.venv`
is only a symlink to core-env for existing subprocess tests. Live sensor connection
remains pending. Re-coordinate before heavy work if physical testing resumes.

Artifacts at `/Users/yushan/Desktop/Go2_Navigation/data/outputs/vln_shadow/`:
`synthetic_paced_v1/`, `synthetic_paced_v2/`, `synthetic_stale_v1/`,
`private_e1_mock_v1/`; logs and selected test XML under `audit/`. Each run has its
exact arguments in `run.json`. Use fresh output names for reproduction. Private
frames, scene logs, videos and weights stay ignored. Only the generated synthetic
`docs/assets/vln_shadow_synthetic.png` is published.

The final-code demo is `published_demo/` (32 frames, three mock proposals). Its
`run.json` also hashes the exact application/native-worker source and runtime config.
The input audit verifies all 107 E1 images and calibration against Claude's unchanged
manifest; annotations were not loaded. See `audit/input_and_environment_audit.json`.

The real verification is in the fresh ignored `verification_20261115_01/` under that
artifact root. Open `real_model_smoke_01/index.html`; original outputs/tensors are
under `real_model_smoke_01/requests/000001/native/`. `audit/` contains complete test
logs/XML, metrics and causal/calibration/preservation checks. All 107 E1 images,
manifest and calibration still match; every prepared frame matches the existing
calibrated preprocessing exactly. Private imagery remains local and unpublished.

Claude's published `GO2_HIGH_LEVEL_CONTROL.md` and `GO2_NATIVE_AVOIDANCE_AUDIT.md`
were read at `742cfe4fb1b71f9d19d9d2d62be56f03b6b03707` in
`Go2_Navigation-go2-control`. [VLN_CONTROL_HANDOFF](VLN_CONTROL_HANDOFF.md) links the
pinned reports and attributes hardware findings to Claude/operator. Reported
avoidance-service velocity commands and routine stop are distinct from unverified
remote override under API ownership, obstacle stopping and robot firmware. Native
absolute waypoints are not velocity commands. No control code was imported, executed
or merged. **Manufacturer obstacle avoidance must remain enabled.**

Next: a separately coordinated read-only receiver → `ObservationInbox` connection;
validate clock epochs, exposure/receipt semantics, host-monotonic mapping and actual
display age. The smoke produced no fresh advisory at the default limit, so live
freshness remains unestablished. Real `T_base_camera` and near-goal stopping are
unresolved requirements for future metric/physical use, not blockers to an explicitly
unregistered RGB-only advisory display. Never substitute identity poses or connect
model output to the control utility.

## Phase 6D → next agent: measured waypoint validation (2026-11-15)

Worktree `/Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6d-worktree`, branch
`phase6-waypoint-validation`, exact base `45154f6`. Main and all previous experiments
remain intact. Read [WAYPOINT_EXECUTION_AUDIT](WAYPOINT_EXECUTION_AUDIT.md) then
[PHASE6D_NAVIGATION_EXPERIMENTS](PHASE6D_NAVIGATION_EXPERIMENTS.md) for contracts,
limits, tests, artifact paths and reproduction commands.

Implemented: opt-in fixed-frame bounded-prefix simulation follower; unchanged
first-waypoint baseline exposed with complete traces; append-only native history
resetting on episode/epoch boundaries; reviewed/hashed experiment declarations;
simulation-only child runners with hard host limits; separate reaching, stopping
and external-termination metrics; full native/selected/executed path panels/videos.
The old replay worker defaults and operational defaults are unchanged.

Claude's review at `4a168e165e8eee58b7fcd4fc5a192e9791d3c278` is read from his branch,
not merged or edited. Human-recognizable doorway supports keeping the original scene.
Eight distinct native pixel tensors at old steps 1–8 rule out identical saved inputs
as the explanation for repeated outputs; HF bypasses the episode ViT cache. No
semantic/model root cause or controller defect is proven by these checks.

Actual verification: **591 core passes, nine genuine skips**; 19 optional native/simulator
cases passed across separate runs. Eight successful known-path/stop cases, one
intentional contact rejection and one reachable-goal controller diagnostic are saved
in `data/outputs/phase6d/waypoint_validation/controller_diagnostics_v1/`. These are
**not learned navigation**. Separately, the full model-driven suite **ran all eight
conditions once**, 121 valid native decodes. Both policies succeed by the declared
native-STOP-inside-goal criterion for ahead/left/right starts. Both initially-near
cases fail: baseline contact-constraint termination without STOP; prefix native STOP
after leaving the goal and passing through the doorway. No goal override, retried
condition, budget extension or private recording substitution. All new behavior
human review remains pending. Do not reinterpret 6/8 on these correlated synthetic
conditions as broad success, or initial goal presence as a completed approach.

**Execution code commit:** `d1590eb0c63046b3899325b70d6908e88a59e453`; published before
results. Later publication updates results/plots and fixes only an evaluator label:
an outside STOP after goal exit was incorrectly called premature. The correct v2
rescore changes that single flag for near/prefix; paths and success counts are
unchanged. Original summaries/declaration are intact, and the corrected source no
longer matches their frozen source hash. Use the execution commit for exact original
source replay, or freeze a new declaration with identical reviewed settings for the
corrected scorer. Do not weaken the implementation-hash check.

**Exact artifacts under our root:**
- `frozen_suite_v1.json`, SHA256 `0c26c9678692955de374a52d5e5f5c7c29bcc33a72778a3d7b97bd7f96f34f4e`;
  reviewed source settings in `reviewed_suite_v1.yaml`.
- `suite_v1/<ahead|left|right|near>__<first_waypoint|bounded_prefix>/rollout/`:
  raw images/jobs, per-decision `inference/NNN/` native outputs/tensors and state,
  actual path traces, unmodified first-version summaries. Per-condition logs/model
  identities are one directory above; original aggregate is `suite_v1/aggregate.json`.
- **Use `suite_v1_rescored_v2.json`** for corrected stopping metrics; it binds raw
  hashes and records the only semantic correction. Tiny distance rounding differences
  from NumPy 1.26 vs 2.4 are <3e-16 m and change no outcome.
- `suite_v1_review_v2/index.json`: all eight 1920×1080 H.264 videos (129 panels),
  including both failures. Representative success: `ahead__first_waypoint/rollout.mp4`;
  failures: `near__first_waypoint/rollout.mp4`, `near__bounded_prefix/rollout.mp4`.
- `audit/`: 591-pass final log/XML; separate optional-case results; original tensor
  repetition and same-image/different-instruction comparison; runtime inventory;
  before/after preservation snapshots; video verification. All 727 package/baseline
  files plus 273 old-simulation/review files match.

**Where the code lives:** `vln/waypoints.py` owns fixed-frame bounded interpolation;
`simulation.py` adds opt-in execution without changing `advance()`; `native_history.py`
owns causal append/reset semantics; `waypoint_rollout.py` separates model jobs and
evaluator state; `waypoint_declaration.py` validates review/code/scene bindings;
`waypoint_evaluation.py` distinguishes entering/stopping/termination.
`scripts/run_waypoint_suite.py` freezes and executes all conditions in bounded child
processes; `summarize_waypoint_suite.py` audits/rescores without inference;
`review_waypoint_validation.py` produces evaluation-only panels/videos.

Next: human-review the **new** close-up/late-stop/contact failures, then declare a
small fixed-checkpoint view/history ablation. Do not infer a camera-domain root cause
or tune the current development cases into a benchmark. Preserve Claude's earlier
review and the old E4 prompt/reference mismatch. Reuse `phase6c/model-env` and
`phase6c/core-env`; `.venv` here is only an ignored
alias to core-env. Do not install anything into Claude's environments. Real
`T_base_camera` remains missing. Physical execution disabled on every new path.

Artifact root `/Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6d/waypoint_validation/`
is ours; Claude owns sibling `panels/`, `human_review/` and `MANIFEST.json`. All remain
ignored. Never overwrite those siblings, Phase 6B packages/baselines or the Phase 6C
failed rollout. Both pre-run and post-run preservation snapshots match.

Known environment issues/reproduction: restricted sandbox ffmpeg/ffprobe can fail
with OpenMP shared-memory error 179; run the existing video/full-suite commands
outside that sandbox, without replacing dependencies. A combined optional run
stalled in a Transformers file read after 18 passes; its remaining MPS pooling case
passed separately. Retain that interrupted log/XML, not an invented combined pass.
No new code defect remains known from the completed tests. Real `T_base_camera`,
live robot validation and Windows/Isaac remain outside this verified milestone.

## Phase 6C → Claude / next agent: navigation-trained diagnostic (2026-11-13)

Implementation commit: [`0a442f4984c98578f32c5f406bcdfbac01b150a2`](https://github.com/binbin-ovo/Go2_Navigation/commit/0a442f4984c98578f32c5f406bcdfbac01b150a2).
Later documentation clarification distinguishes stored checkpoint elements from unique
parameters after embedding/head tying; inference artifacts are unchanged.

**Branch:** `phase6-pretrained-vln`, separate worktree
`/Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6c-worktree`, based exactly on
`7e2989ad54b9c484d75c029ff0141834880852b6`. Use `git rev-parse HEAD` to identify the
checked-out publication. No main merge or operational-default change. Original
checkouts, private recordings/annotations, calibration environment, Phase 5A.1,
Grounding DINO and saved SmolVLM results are not overwritten.

**Read first:** [PRETRAINED_VLN_FEASIBILITY](PRETRAINED_VLN_FEASIBILITY.md),
[PRETRAINED_VLN_EXPERIMENTS](PRETRAINED_VLN_EXPERIMENTS.md),
[WINDOWS_SIMULATION_FEASIBILITY](WINDOWS_SIMULATION_FEASIBILITY.md), then the existing
[episode contract](VLN_EPISODE_INTERFACE.md). The experiment report contains exact
installation, inference, evaluation, simulation, tests and preservation commands.

**What Codex implemented:** pinned-source/checkpoint preparation, read-only Mac/Windows
inventory, isolated local LightNav inference worker, causal native-history bridge,
strict RVQ/runtime contracts, fixed diagnostic action mapping, evaluator-only
SmolVLM/human comparison panels, and an optional synthetic MuJoCo camera proxy.
These extend existing calibrated views, episode loaders and review-video tools;
sensor/tracker internals were not rewritten. No upstream real-robot client, network
motion interface or execution-authorized path exists in these additions.

**Actual model/runtime:** LightOriginsHQ/LightNav-0
`826dc5fbfa37afa8293d2e336d329b6ffc0bfb64`; code
`3015508b70fb6e30bbd66701e94c9b916dcb8df8`; Apple M3 Pro, MPS BF16/SDPA, no quantization.
Use **two new environments**: `phase6c/core-env` (Phase 6B-constrained Python 3.12)
and `phase6c/model-env` (Python 3.11, Torch 2.10, Transformers 5.8, NumPy 2.4).
The ignored `.venv` alias in this new worktree points at the new `core-env` because an
existing viewer test requires that relative launcher. It does not point to or alter
Claude's environment. Both new environments pass dependency checks.
Do not install the latter stack into recording/calibration/perception environments.
The source is private under `phase6c/lightnav/`: pristine `source`, first `source-local`
fork and final `source-local-v2` all retained. Final patch SHA256
`bec34b7d855f33ac94ec212a61ca507d2d677d82441add0cd1b2dfe7088e554c`; changes are device
selection/timing and exact BF16 CPU fallback for unsupported non-divisible MPS pooling.
`scripts/patch_lightnav_local_runtime.py` recreates it from verified pinned source.

**API and frames:** `pretrained_inputs.prepare_job(episode, decision_id, destination)`
creates only instruction/identity/time and causal rectified images. `pretrained.causal_history`
uses actual exposure plus availability, a 4 Hz grid and native SlowFast afterwards.
Raw fisheye, rectified camera and official 448×256 model-space metadata remain distinct.
Worker jobs whitelist fields and hash images. The native processor/template/normalization
and RVQ decoder remain official. Native ten-waypoint chunks are absolute ego SE(2),
metres/radians, x forward/y left/yaw CCW, **without a time base**. `T_base_camera` is
still unknown; do not treat these as Go2 base-frame commands or fabricate registration.

The policy requires RGB history, not prior executed actions. It resets native buffers,
vision cache and engine state before every decision/episode/epoch, then reconstructs
causal observations. Never feed a proposal back as executed while following an operator
recording. `SimulationRequest` uses a separate host-monotonic deadline and rejects
wrong episode/epoch/sequence or stale results. The local static simulator applies the
first native waypoint over an explicitly chosen 0.25 simulated seconds with limits;
its geometry/contact/goal truth remains evaluator-only.

**Inputs and private artifacts:** use the exact Claude packages under
`/Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6b/episodes/`; do not rebuild,
change instructions/cutoffs or relabel. Their model/evaluation separation is unchanged.
Original baseline: `phase6b/claude_pilot_v1/`. New artifact root:
`/Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6c/`:

- `lightnav/`: hashed code/checkpoint/processor/RVQ assets; weights stay out of Git.
- `claude_pilot_v1/`: preparation/launcher failure retained; `claude_pilot_v2/`: five
  successful calls and native MPS pooling failure retained.
- `claude_pilot_v3/`: complete 11-call inference, native outputs/tensors/memory/latency.
  It reuses v2's exact hashed jobs/prepared inputs; `run_declaration.json` records that
  provenance, unchanged manifests, preparation code and new worker code identity.
- `claude_pilot_v3_review_v2/`: final evaluation-only panels, all actual native historical
  input images, video, human references, full chunks and baseline comparisons. Original
  review remains as well; only plot scale/yaw display was improved, without inference
  reruns or score changes.
- `sim_closed_loop_v1/`: one actual failed 16-step static rollout; full camera/native
  proposal/actual action/evaluator trace. `sim_closed_loop_v1_review_full_chunks/` contains
  all 16 panels and video, including the failure.
- `audit/`: official failure/success logs, tests, hardware and source-preservation hashes.

**Validation:** full core suite **541 passed, 4 skipped in 41.58 s**, including actual
cached SmolVLM inference; all five optional native/MuJoCo cases passed in separate
model-environment runs. Only physical Go2 remains unexecuted across these environments.
63 new core cases include configuration/worker identity and premature-EOF hashing
regressions. Existing tests were not modified. See `audit/full_regression_verified.log`.

**Preservation audit caveat:** all 434 package files and 292 baseline files match the
original preflight. One baseline PNG had an invalid empty-file hash from a transient
short read. It was not rewritten: two fresh readers and an in-memory run of its original
rectification/PNG producer agree on the intact bytes. Keep the failed inventory and
`baseline_short_read_recovery.json`; do not pretend the original digest was valid.
Two strict post-run snapshots agree on all 727 files. Use
`audit/inputs_final_verified.json` for subsequent comparisons, not the flawed initial
`inputs_before_verified.json`. The auditor now rejects partial/unstable reads and
saves mismatch reports instead of silently accepting a truncated digest.

**Results and limits:** native RVQ valid 11/11; fixed discrete comparison coverage 9/11,
five matches/four mismatches/two unsupported mixed chunks. E1–E3 on the same seven
mapped cutoffs: LightNav 3/7 vs Smol temporal 4/7 and current-only 5/7. Initial yaw sign
follows right/left, but no STOP on any cutoff; no overall gain established. E4 outputs
a native not-visible token and search turn, not verified global absence or task success.
Historical E4 UNRESOLVED/search-turn mismatch is preserved. All examples are one
correlated test_c group; new semantic/evidence review remains pending.

**Simulation result:** renderer and model can coexist in this tiny Mac case. The
16-step model rollout turns too long, then moves away, exhausting its budget without
stopping (distance 1.850 → 2.722 m). No contacts; no learned avoidance or Go2 locomotion
claim. No latency-aware/dynamic trial. Windows remains inaccessible: run the PowerShell
preflight before assessing its existing Isaac version. Its reported 8 GB is below
LightNav's unquantized weight size; do not assume BF16/FlashAttention-2/quantization
support or shared model+Isaac capacity. No driver/WSL/install change was made.

**Next steps, in order:** independently review pointing/chunk/stopping failures using
the local panels, design diverse reviewed episodes outside test_c, run the Windows
inventory on the actual laptop, then inspect that installed Isaac version/checker.
Any subsequent scene/model experiment should use a new output folder and declared
conditions. Preserve this failed rollout and benchmark. Resolve measured camera/base
registration, timing and navigation behavior before considering physical integration;
physical execution stays disabled. Current tests and reproduction details are in the
experiment report. A local file-I/O issue caused stalls/transient read errors; successful
retries and failures are recorded, with no private-source substitution.

## Phase 6B → next agent: Claude-package evaluation (2026-11-13)

Evaluation implementation commit: [`4d5e49642da03558830dba017f5aa150d9fdc684`](https://github.com/binbin-ovo/Go2_Navigation/commit/4d5e49642da03558830dba017f5aa150d9fdc684).

**Branch:** `phase6-temporal-vln`; follow-up to `d2112268b1863185d39f272a10d5bb80cbefaaaa`.
Claude's packages are from `phase6-vln-episode-validation` commit
`3a84f57b8753b0e5c169f53ae361a33f9ddbc3fe`. His files/checkout were not modified or
merged. Main remains unchanged. No physical robot connection or executable commands.

**Read first:** [TEMPORAL_VLN_CLAUDE_PILOT](TEMPORAL_VLN_CLAUDE_PILOT.md) for exact paths,
commands, actual scores and caveats; [VLN_EPISODE_INTERFACE](VLN_EPISODE_INTERFACE.md)
for the stable contract. The next section below documents the older Codex-authored
pilot and retains its separate pending-label status.

**Inputs used, unchanged:** root
`/Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6b/episodes/`, packages
`p6b_test_c_e1_turn_right`, `p6b_test_c_e2_turn_left`,
`p6b_test_c_e3_forward_bottles`, `p6b_test_c_e4_absent_sofa`.
Each exact manifest is `<package>/model_input/episode.json`; corresponding human
references are `<package>/evaluation_only/annotations.json`. Review logs, provenance,
odometry slices and label panels remain evaluator-only. No builder was run; no labels
or cutoffs were changed. All supplied hashes matched and all 434 package files were
verified unchanged after inference. Keep the declared 0.050 s decode margin.

**Review provenance:** all 11 action references are reviewed human labels, one repository
owner accepting Claude's drafts unchanged. Seven are marked ambiguous. There are two
complete / six incomplete / three unknown completion references. Do not reclassify
unknown as incomplete, substitute agent labels, or call these independent blind labels.
E1–E3 instructions were written retrospectively after viewing the operator route.
E4 is a static absent-target probe. All share test_c, overlap earlier pilot footage,
and must remain a single correlated group. No similar-target case exists here.

**Actual execution/results:** same SmolVLM2-2.2B checkpoint/processor revision
`be92141c8d0ceb98596a923c308ac2bbb4a7c466`, CPU FP32/eager, frozen prompt/parser/input
pipeline. 11 cutoffs × two modes = 22 calls. Every temporal encoder call used four
images; message/provenance audit confirmed only instruction plus causal observations
entered the model. Human action agreement: E1–E3 5/9 temporal vs 7/9 current-only;
E4 2/2 vs 0/2 (two malformed current outputs rejected); overall 7/11 each. No history
advantage is established. Both modes miss stopping references; the one temporal
COMPLETE claim is unscorable because that human completion reference is unknown.
E4's accepted search-turn sets also differ from the frozen absent-target UNRESOLVED
prompt policy; neither labels nor prompt were changed to improve scores.

**Code/API changes:** `scripts/run_temporal_vln.py --episode PATH` can now be repeated
for exact nested packages, avoiding automatic discovery of neighboring old episodes.
`scripts/evaluate_temporal_vln.py --run RUN --annotations-name ../evaluation_only/annotations.json
--static-episode p6b_test_c_e4_absent_sofa --output NEW_JSON` evaluates saved proposals
without model loading. It validates input/manifest binding, records annotation hashes,
keeps recordings grouped and static cases separate, and excludes missing/pending/agent
references from human denominators. It does not alter source data or inference files.
Unknown reference completion and malformed model statuses do not enter completion
consistency. Free-text progress consistency still requires separate adjudication.

**Private artifacts:**
`/Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6b/claude_pilot_v1/`.
Start at `evaluation_grouped.json` and `review.mp4` (22 s, 1650×990). Every per-call
folder retains native output, exact tensors/prompt, frame provenance and labeled panel.
Sibling `audit/claude_pilot_v1_preflight.json` / `claude_pilot_v1_postflight.json` prove
source integrity and causal inputs. Original `evaluation.json` is retained; the grouped
report removes inapplicable sequence flags from ambiguity-filtered subsets, with all
action scores unchanged. No inference was rerun for that reporting correction.

**Validation:** full suite with SMOLVLM_MODEL_CACHE set: **478 passed, one hardware test
skipped**; all four modified/new Python files pass Ruff check/format. Log:
`audit/claude_pilot_full_tests_final.log`. The earlier two model tests still execute
real weights. Normal tests use synthetic fixtures. The isolated environment and
pinned dependencies are unchanged; use the existing installation in the demo guide.

**Next:** do not rebuild/relabel this supplied pilot or tune against it as held-out
validation. Plan a separate experiment on independently reviewed diverse recordings
with explicit completion conditions. Preserve the one-recording grouping, human draft
acceptance provenance and static E4 separation in downstream comparisons. Registration,
metric depth, timing/odometry and traversability remain unresolved; controlled closed-loop
validation needs separate authorization. Keep every proposal execution_authorized=false.

## Phase 6B → Claude / next agent: temporal instruction-following pilot

Implementation commit: [`e9fa7a99c0182af150ffaccc58a83ef54f1a6b0d`](https://github.com/binbin-ovo/Go2_Navigation/commit/e9fa7a99c0182af150ffaccc58a83ef54f1a6b0d).

**Branch:** `phase6-temporal-vln`, verified base
`c13f1dc442fa157b31043613d3cb2b0c30ceab8a`. Original checkout, Claude's calibration
environment, Phase 5A.1 and main were left unchanged. No robot connection or commands.
Implementation is local offline code plus a completed private pilot; human semantic
validation and navigation success are not complete.

**Read first:** [TEMPORAL_VLN_EXPERIMENTS](TEMPORAL_VLN_EXPERIMENTS.md) for actual
failures/measurements, [TEMPORAL_VLN_ARCHITECTURE](TEMPORAL_VLN_ARCHITECTURE.md) for
model/input/output semantics, [TEMPORAL_VLN_DEMO](TEMPORAL_VLN_DEMO.md) for exact commands,
and [VLN_EPISODE_INTERFACE](VLN_EPISODE_INTERFACE.md) before preparing new episodes.
The Phase 6A and earlier handoffs below retain their original historical scope.

### Published episode contract — stable v1

Contract commit: **`5c7f586aa2536a8eda9167e66d5050a4a91dbac2`**, pushed before model
implementation. Fixture: `tests/fixtures/vln/episode_v1/episode.json`; separate
annotations: `tests/fixtures/vln/episode_v1/annotations.json`. Schemas:
`src/go2_perception/vln/schemas/`; loader: `src/go2_perception/vln/episodes.py`.
These bytes/schema remain compatible with the first publication. Fixture generator:
`scripts/generate_vln_episode_fixture.py`; inspect with:

```bash
python scripts/inspect_vln_episode.py tests/fixtures/vln/episode_v1/episode.json --decision d0
```

One episode = one recording and boot/clock epoch. Preserve exposure versus availability,
ordered images, explicit bounds/gaps and image/calibration hashes. Camera optical axes
right/down/forward; base_link forward/left/up. T_base_camera remains missing. Never use
identity as a placeholder or treat floor-seen cells as safe free space. Source timestamps
are seconds with documented clock semantics; runtime host times remain separate.

### Implemented and exercised by Codex

- `vln/temporal_inputs.py`: causal four-frame/six-second policy, independent Phase 6A
  calibrated rectification per image, official SmolVLM multi-image template/processor,
  resized K/FOV/crop/padding metadata and exact tensors/hashes.
- `vln/temporal_model.py`: one pinned SmolVLM2-2.2B checkpoint, CPU FP32/eager, deterministic
  greedy inference; vision-encoder hook verifies every historical image enters the model.
- `vln/temporal_proposals.py`: native text retained, strict JSON/anchored native field
  parsing; action separate from task status, unknown frame/quantities preserved.
- `vln/temporal_review.py`: current/history panels, native responses, labeled evaluation,
  review video and metrics restricted to reviewed human references. Prose claims need
  separate review; structured COMPLETE counts do not cover them.
- `scripts/prepare_temporal_vln_pilot.py`, `scripts/run_temporal_vln.py`: deterministic
  private packages and immutable run directories; configs `temporal_vln*.yaml` / prompt.

**Actual model:** `HuggingFaceTB/SmolVLM2-2.2B-Instruct`, processor and checkpoint
`be92141c8d0ceb98596a923c308ac2bbb4a7c466`. Generic **VLM navigation-proposal baseline**,
not a reproduced VLN policy. No keyword navigation parser or training. Four episodes,
eight decisions, 16 temporal/single-image calls, 16 valid structured responses.
Every temporal call encoded [4,3,384,384] images / 324 image tokens; single calls
encoded [1,3,384,384] / 81. No label data entered prompts or inference.

**Important failure:** temporal always TURN_RIGHT (8/8), including left instruction;
single outputs 4 right, 2 left, 2 unresolved. One native reason falsely implies task
completion while structured status remains IN_PROGRESS. Human references are pending.
Six panels/all native texts were agent-reviewed with explicit reviewer type; no human
semantic score or closed-loop success is claimed. Frozen input/model/parser/prompt
hashes are in the experiment report; do not silently retune and reuse this pilot as
held-out evaluation. Generation mean 10.171/6.221 s temporal/single; process peak 9.684 GiB.

### Installation, tests and private evidence

Separate worktree: `/Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6b-worktree`.
Private evidence: `/Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6b/`.
This persistent location is independent of `/private/tmp`; all media and weights are
Git-ignored. Source recordings were copied/read, not modified.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -c requirements-temporal-vln-tested.txt \
  -e '.[test,temporal-vln]' av==17.1.0 ruff==0.11.13
export PYTHONPATH=src
ARTIFACTS=/Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6b
SMOLVLM_MODEL_CACHE="$ARTIFACTS/hf_cache" HF_HUB_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false MPLCONFIGDIR="$ARTIFACTS/mpl" \
  .venv/bin/python -m pytest -q -rs
```

**470 passed, 1 hardware skipped, 36.12 s** including both actual model tests. Without
SMOLVLM_MODEL_CACHE, the two heavyweight model tests skip; no automatic model download.
Ruff check/format passed on 12 Phase 6B files. Grounding DINO compatibility was exercised
on a real saved image and exactly reproduced the old candidate. Full constraints record
this Mac/Python environment, not Linux/CUDA verification. Do not install the older
`vln` extra and `temporal-vln` simultaneously (different Transformers pins).
If a Mac editable `.pth` is hidden, `PYTHONPATH=src` works; the demo documents the isolated
flag fix. No remaining installation blocker in the tested environment.

Start demo review at `pilot_v1/review.mp4` (16 s, one panel per call, not robot timing).
Successful literal-direction example: `pilot_v1/turn_to_racket/d00/current_only/review.png`;
failed opposite direction: same episode/decision `temporal/review.png`; unresolved:
`visible_refrigerator/d00/current_only/review.png`; ambiguous referent:
`similar_cabinets/d00/temporal/review.png`; unsupported completion prose:
`absent_red_door/d00/temporal/review.png`. Review findings are agent-only in
`audit/agent_review.json`. `audit/full_tests_with_model.log` records the final suite.

### Exact next steps / ownership

1. Claude/human: prepare private real episodes using the published v1 contract.
   Existing `episodes/*/annotations.pending.json` are empty human-review placeholders.
   Write separate reviewed annotation files with reviewer identity/status and binding
   manifest SHA. Do not change an evaluated manifest or insert labels into prompts.
2. Score saved outputs using `temporal_review.summarize` with validated references
   (example in the demo); human instruction-progress and prose-completion consistency
   need separate adjudication. Do not count agent notes as human ground truth.
3. Collect held-out diverse turn/landmark, ambiguous and absent-target episodes.
   Diagnose right-turn bias in a separately named, predeclared development run;
   checkpoint/input/prompt changes must get new output directories and explicit scope.
4. Claude owns hardware/calibration/timing work. Missing T_base_camera and registered
   depth remain blockers for metric goals; moving odometry and traversability still
   require validation. No metric goal or action reference is invented here.
5. Only a separately authorized closed-loop experiment can assess goal reaching,
   collisions, safe execution or recovery after a proposed action. This baseline
   provides no control interface and is not recommended for robot commands.

## Phase 6A → next agent: calibrated offline grounding (Codex, 2026-11-12)

Implementation commit: [`dcc5de1c0d899d5fb8159c5f6b0905ab67b8cf01`](https://github.com/binbin-ovo/Go2_Navigation/commit/dcc5de1c0d899d5fb8159c5f6b0905ab67b8cf01).

- **Branch:** `phase6-vln-input-validation`, base
  `e4e0d3b01a16a14f52f7d039b27009ed3e0e0f96` from Claude's calibration branch.
  No main merge. No robot connection or commands. `phase5-detection-recovery` and
  Claude's worktree/environment were not modified. This base has 5A/5B, not 5A.1.
- **Capability actually tested:** Grounding DINO Tiny text-conditioned image-region
  proposals from measured-calibration Go2 recordings. Not a VLN navigation policy.
- **Read first:** [VLN_INPUT_VALIDATION](VLN_INPUT_VALIDATION.md) for actual measurements,
  failures and calibration limits; [VLN_OFFLINE_DEMO](VLN_OFFLINE_DEMO.md) for exact
  commands/artifacts; [VLN_MODEL_SELECTION](VLN_MODEL_SELECTION.md) for the checkpoint audit;
  [GO2_VLN_INPUT_INTERFACE](GO2_VLN_INPUT_INTERFACE.md) for the unchanged sensor contract.

**Implementation:** `vln/model_preprocessing.py` rectifies raw RGB and carries calibrated
K through official processing; `vln/grounding.py` loads pinned pretrained weights and
returns image proposals; `vln/grounding_demo.py` saves evidence and a separate LiDAR grid.
CLIs: `scripts/phase6_grounding_demo.py`, `scripts/phase6_evaluate_grounding.py`.
Config/cases: `configs/vln_grounding.yaml`, `configs/vln_grounding_eval.yaml`.
Regression tests: `tests/test_vln_grounding.py` (27 tests).

**API:** `view = calibrated_view(obs)`; `result = model.predict(view, instruction)`;
`run_observation(obs, model, output, source=...)` writes the full experiment.
Use a `Go2VLNObservation` with raw RGB uint8 and measured K/D of matching dimensions.
The model gets one current image, preserves but ignores history, and applies its own
normalization once. Output boxes use rectified-image edge pixels, optical axes remain
right/down/forward. No conversion to base_link forward/left/up or metric distance.
All `metric_goal`, `action`, `point_base_m` fields remain null even if future callers
supply extrinsics. The right-hand LiDAR plot is a separate observation, not a fused map.

**Important correction to older guidance below:** floor-seen cells and sector floor
returns do **not** establish traversability or a safe swept corridor. The historical
Phase 5D symbolic-action conversion paragraph is not an approved Phase 6 execution
path. Pixel proposals do not become camera-relative or base-relative metric waypoints.
No executable-goal adapter exists here.

**Install / verify:** in this branch, `uv venv --python 3.12 .venv`,
`uv pip install --python .venv/bin/python -e '.[test,vln]'`, optionally install `av==17.1.0`
for inherited decoder tests, then `.venv/bin/python -m pytest -q -rs`.
Actual result: **421 passed, 1 hardware skipped**, 30.71 s. No automated tests weakened.
Model + processor pinned to `IDEA-Research/grounding-dino-tiny` revision
`a2bb814dd30d776dcf7e30523b00659f4f141c71`; CPU FP32 / four threads.

**Local evidence:** `/private/tmp/go2-phase6/data/outputs/phase6/evaluation/` (ignored).
Each case has raw observation/model output, source hashes/clocks, preprocessing,
proposal JSON and demo PNG. Start with `s_fridge/demo.png` (correct) and `m_fridge/demo.png`
(wrong door region), then `s_dog`, `m_bicycle`, `m_sofa` false positives and `m_shoes`
miss. `semantic_review.json` records Codex visual review; human review is pending.
The 12-case declaration predates inference; thresholds were not retuned. 6 acceptable,
1 partial, 5 failed. Output validity was 12/12; semantic correctness differs.
NPZ replay reproduced raw arrays exactly. A navigation instruction was rejected as
unsupported without forwarding through the model. All private media, outputs and
weights remain out of Git. Local temporary paths may eventually be cleaned; rerun
from the private recordings and pinned model using the documented commands.

**Next work:**
1. Human-review the local panels and create diverse held-out object/absent/relational
   grounding annotations. Record failures; do not optimize on these 12 cases.
2. Claude: pursue measured, held-out validated camera/base/LiDAR registration and
   timing/odometry validation; preserve unknown fields until verified.
3. Select a true temporal/relational VLN model in a separate experiment if needed;
   its output still requires a documented frame and independent geometry checks.
4. Only after validated registration/range: design metric grounding and a separate
   collision-aware planner. No locomotion/control work is authorized by this milestone.

## Phase 5D → Codex handoff: Go2 VLN inputs (Claude, 2026-11-11)

- **Branch:** `phase5-go2-vln-sensor-calibration`, based on `89d901dc` (5B). Do not merge main.
  5A/5A.1 perception modules, configs and defaults are unchanged.
- **Interface:** `go2_perception.vln`, schema `go2-vln-obs-v1`. Spec:
  [GO2_VLN_INPUT_INTERFACE](GO2_VLN_INPUT_INTERFACE.md).

**Available now (real data):**
| Input | Shape / units | Frame | Time |
|---|---|---|---|
| RGB | uint8 (720, 1280, 3) **RGB**, raw/distorted (fisheye K, D measured), ~14.3 Hz, + history | `go2_front_camera_unrectified` | host receipt, and estimated robot-clock exposure (host − offset − 48 ms, σ 16 ms) |
| Obstacle points | float32 (~11 k, 3) m, 0.5 s window, odom-compensated | `base_link` of the reference scan | scan midpoint, robot clock |
| Display points | float32 (~21 k, 3) m, 1.0 s window | same | same |
| Grid | int8 (100, 100), 0.1 m, x −3…7 m, y −5…5 m; −1 unknown / 0 floor seen / 1 occupied | `base_link` | same |
| Elevation | float32 (100, 100) m, NaN unknown | `base_link` | same |
| Sector ranges | (72,) m per 5° CCW from +x; NaN unknown; `sector_clear_to` floor evidence | `base_link` | same |
| Pose / twist | 4×4 `T_odom_base`; twist frame unverified | `odom` ← `base_link` | robot clock |

**Calibrated vs uncertain:** camera intrinsics **measured** (`configs/go2_front_camera_calibration.yaml`,
fisheye, held-out 0.47 px; pass it as `calibration=`); `T_base_camera` **missing**, so
`projected_depth` is `None`. `T_base_lidar` comes from firmware only. Camera latency was
measured once. Odometry accuracy under motion is unverified. The floor-height assumption (−0.33 m)
holds only while standing. The robot clock resets each boot: compare stamps only within one `clock_epoch`.

**Feasible VLN types now:** RGB-only video models producing symbolic or mid-level actions
(NaVid / NaVILA / StreamVLN style), gated by the LiDAR grid. After calibration: RGB + geometric map
(VLFM-style value map on our grid, the preferred approach). RGB-D/3D models need calibration plus
dense depth. See [VLN_MODEL_COMPATIBILITY](VLN_MODEL_COMPATIBILITY.md).

**Model outputs → goals without inventing geometry:** a symbolic action becomes `T_odom_base · T(θ, d)`.
Accept it only if the target cell and corridor are FREE (floor seen) and within `sector_clear_to`;
UNKNOWN is not free. Pixel targets now give a calibrated *ray direction* in the camera frame, but
turning it into a metric goal needs `T_base_camera`, so they are unsupported today. 3D boxes
and model confidence never count as clearance. Nothing here sends commands.

**Reproduce (this Mac):**
```bash
cd /Users/yushan/Desktop/Go2_Navigation-phase5d
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e '.[test,forecasting]'
export PYTHONPATH=src
D=/Users/yushan/Desktop/Go2_Navigation/data
.venv/bin/python -m pytest -q                                    # 381 passed, 3 skipped
.venv/bin/python scripts/phase5d_vln_replay_demo.py --synthetic --output /tmp/vln_demo
.venv/bin/python scripts/phase5d_vln_replay_demo.py --recording $D/recordings/p5d_static_02 --index 200 --output $D/outputs/phase5d/demo
.venv/bin/python scripts/phase5d_lidar_accumulation.py --recording $D/recordings/p5d_static_02 --output $D/outputs/phase5d/accumulation
.venv/bin/python scripts/phase5d_timing_report.py --data $D/recordings --recordings test_c p5d_static_02 --output $D/outputs/phase5d/timing.json
.venv/bin/python scripts/phase5d_make_charuco.py --screen 2560x1600 --output $D/outputs/phase5d/targets
```
Live capture (robot stationary, user present): `.venv-go2` in the main checkout, `--iface en9`,
see [MAC_CONNECTION](MAC_CONNECTION.md).

**Claude/user next:**
1. ~~Camera intrinsics~~ done 2026-11-11: [GO2_CAMERA_CALIBRATION](GO2_CAMERA_CALIBRATION.md).
2. Extrinsics: stopped 2026-11-12 after two failed sessions; read "Session 2" and "What would plausibly
   work next" in [GO2_LIDAR_CAMERA_EXTRINSICS](GO2_LIDAR_CAMERA_EXTRINSICS.md) (≥ 60 × 80 cm target with
   larger squares, poses at one range, plane-level gate, PCA not 3-point RANSAC; or model the LiDAR's
   range bias first).
3. Re-measure camera latency on a new boot (operator-driven yaw).

## Phase 5B → Codex handoff (Claude, 2026-11-11)

- **Branch:** `phase5-fsd-visualization`, based on `be4b9de` (Phase 5A).
- **Do not merge main.** Nothing in `scene/`, any existing config or any default was changed; the only new config is the prepared held-out protocol.
- **Read:** [PHASE5_REAL_VALIDATION](PHASE5_REAL_VALIDATION.md) for the evidence and the fix list;
  [FSD_STYLE_VISUALIZATION](FSD_STYLE_VISUALIZATION.md) for the viewer.

**Fixes for Codex, ordered by evidence (development data A1–C; validate on held-out H1–H3):**
1. **Freshness from recent coverage.** Refresh an object from ≥3 returns in the last 2–3 messages, or
   by swept-sector coverage, rather than the latest ~62 ms message alone. In B, 88 of 91 "absent"
   walking frames have zero person returns in the latest message, yet ≥8 non-ground points at 0.3 s
   (89/91). At the person's bearing the message has no non-floor returns in 94.5% of them: a sampling
   gap, not occlusion.
2. **Motion gate robust to σ.** `speed − 2σ > 0.25` blocks 56 fresh walking frames: median speed
   0.44 m/s, median σ 0.23 m/s. Use a displacement/direction test, `speed − 1σ`, or σ from
   `observed_box` centres.
3. **Inherit motion history** when a reborn object re-associates within 1 s of an expired one (23 B
   frames).
4. **Robot self-filter** in proposal generation (fresh object within 0.45 m of `base_link` in about
   98% of frames).
5. **Leg clustering** (14 B frames left as noise or merged off-centre).
6. **Vectorize `BackgroundMap.stable_mask`.** With `object_window_s=0.3` a replay ran at about
   9–18 s/frame, so the counterfactual was stopped.

**Reproduce (on this Mac):**

```bash
cd /Users/yushan/Desktop/Go2_Navigation-phase5b    # worktree on phase5-fsd-visualization
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e '.[test,forecasting]'
export PYTHONPATH=src
D=/Users/yushan/Desktop/Go2_Navigation/data
O=$D/outputs/phase5b
.venv/bin/python scripts/phase5b_scene_tap.py --input $D/recordings/test_b --output $O/tap_new/test_b   # fresh dir; optional --set key=value diagnostics
.venv/bin/python scripts/phase5b_diagnose_walking.py --recording test_b --data $D --tap $O/tap_new/test_b --output $O/diagnosis_new/test_b.json
.venv/bin/python scripts/phase5b_compare.py --data $D --variant 5A=$O/tap --variant fix=$O/tap_new --output $O/comparison_new.json
.venv/bin/python scripts/fsd_viewer.py --recording test_b --data $D --mode debug --tap $O/tap_new/test_b --scene $O/tap_new/test_b/scene.jsonl
.venv/bin/python -m pytest -q
```

`phase5b_compare.py` expects all four recordings under each variant directory. `phase5b_scene_tap.py`
writes the same `scene.jsonl` as `run_stable_scene.py` plus Debug sidecars, and `--reference` checks
row identity.

**Claude/user next:**
1. Record held-out H1–H3 per `configs/phase5_heldout_protocol.yaml` (robot moved by hand only).
2. Verify `cloud_base` sector coverage on hardware.
3. Calibrate the camera.
4. Build a live read-only viewer with end-to-end latency measurement.

**Limitations:**
- Annotations are not human-verified; everything here is development data.
- The camera sync is approximate.
- The viewer's floor is assumed at robot z − 0.32 m.
- The 0.3 s 5A counterfactual was not completed.

## Phase 5A → Phase 5B Claude Handoff

- **Branch:** `phase5-stable-lidar-perception`.
- **Complete interface commit:** `5e269061c3da70ffe3d00798eb9bf05977869797` (includes initial fixture/schema version commit `9b99375`). This immutable commit contains every renderer-facing interface file; this handoff note is a subsequent documentation commit on the same branch.
- **Contract:** `docs/SCENE_REPRESENTATION_INTERFACE.md`.
- **Versioned machine schema:** `schemas/go2-scene-v1.schema.json` (`go2-scene-v1`, JSON Schema 2020-12).
- **Exact deterministic fixture:** `tests/fixtures/scene/renderer.jsonl` (six JSONL rows, synthetic only).
- **Minimal consumer:** `scripts/read_scene.py`, standard library only; `scripts/generate_scene_fixture.py` reproduces the fixture. Semantic validator: `src/go2_perception/scene/serialization.py`; tests: `tests/test_scene_interface.py` (**8 passed**).

Fetch the branch, then consume the fixture without waiting for perception implementation:

```bash
git fetch origin phase5-stable-lidar-perception
# Read from a separate checkout/worktree to preserve your current annotation/video work.
git worktree add ../Go2_scene_renderer origin/phase5-stable-lidar-perception
cd ../Go2_scene_renderer
python scripts/read_scene.py tests/fixtures/scene/renderer.jsonl
```

All scene geometry is already in `odom`, in meters, right-handed x-forward/y-left/z-up at odometry origin. Timestamps are seconds in the explicitly named clock domain; fixture clock is `synthetic`, real records use `go2_robot`. Row-major `T_odom_base` maps base column vectors into odom: apply it only to the robot mesh, never again to the scene points/boxes. Clear renderer history on epoch changes. Null pose is unknown, not identity. Preserve observed-vs-filtered-vs-stale and UNCERTAIN state; absence/expired cells are unknown, not free space. The last two fixture frames deliberately mark the static box stale while the second object remains observed/MOVING.

`display_compact` controls drawing only; raw measurements must still enter perception. Scores are defined evidence/recency ratios, not probabilities. This delivery contains no camera calibration, private recordings, control commands or promise of real geometry accuracy. Claude may implement the renderer against v1 immediately. Codex will continue Phase 5A on the same branch, preserving required fields and semantics. Additive optional fields are allowed; a breaking change requires explicit coordination with Claude and a new version. User selected GitHub branch publication as the delivery channel; Claude acknowledgement remains pending.

## Phase 5A perception continuation

Implementation and experiments are complete as an **experimental** opt-in pipeline, not a validated perception upgrade. See STABLE_LIDAR_ARCHITECTURE.md and STABLE_LIDAR_EXPERIMENTS.md before consuming real output. `StableScene.process` accepts raw Nx3 base_link points, seconds, unique interval IDs, same-clock pose lookup and optional per-point second offsets. It emits the unchanged published v1 contract. Do not retransform scene geometry or feed accumulated cloud_deskewed data. Reset renderer history on epoch changes; stale boxes must stay visibly stale/UNCERTAIN.

- Run: `PYTHONPATH=src .venv/bin/python scripts/run_stable_scene.py --input /path/to/recording --output data/outputs/scene_trial`. The output directory must be new and gitignored. No annotation input is accepted by runtime.
- Synthetic comparison: `PYTHONPATH=src .venv/bin/python scripts/benchmark_stable_scene.py --output data/outputs/scene_synthetic_trial`.
- Tests: `PYTHONPATH=src .venv/bin/python -m pytest -q`: **310 passed, 1 hardware skipped**. Full Ruff has 33 inherited hardware-script findings; Phase 5A scoped lint passes. Exact setup/report commands are in STABLE_LIDAR_EXPERIMENTS.md. New jsonschema test dependency is pinned to 4.26.0.
- Baseline comparison and aggregate evidence: STABLE_LIDAR_RESULTS.md / STABLE_LIDAR_METRICS.json. Real Phase 5A moving recall A2=0.121, B=0.058; fresh box coverage is much lower than original. Roughly 69% of emitted boxes are stale. Do not promote defaults or describe smooth geometry as reliable pedestrian perception.
- Source ownership: Codex added `scene/`, stable scene scripts, config and tests. Claude retains Go2 access, timing verification, annotation review and renderer. Existing tracker/adapter/forecasting code was not rewritten. No raw private data was committed.
- Next steps: consume the fixture on the branch; verify stale/epoch rendering; human-review development annotations; collect held-out scenes including nearby/occluded people; verify instantaneous point-frame semantics before enabling deskew; improve proposal coverage and compare observed-only metrics before changing operational settings.
- Known failure reproduction: run synthetic `near_furniture` through the benchmark (moving recall 0.189 at 0.1 s, 0 at 1.0 s). Replay A1/A2/B/C locally with the documented evaluator to reproduce low fresh coverage and low moving recall. Missing odometry emits empty geometry; weak evidence emits UNCERTAIN. Covariance is heuristic, not a calibrated safety bound. No live/held-out Phase 5A validation has been performed.

# Agent handoff — 2026-11-09 (macOS update)

## Phase 3.8 visual ground-truth handoff (integrated history)
Branch `phase3-visual-groundtruth`. It contains `79bb0f9` (ground truth) and the merged Phase
4.1 branch `5979a76`. Do not merge main or change defaults without approval. Read
[VIDEO_ANNOTATION](VIDEO_ANNOTATION.md), then [TRACKING_GROUND_TRUTH](TRACKING_GROUND_TRUTH.md).

**Code added:**
- `visualization/diagnostic_video.py`: rendering, camera alignment, trails, heuristic events,
  H.264 writer, private-path guard.
- `scripts/render_tracking_video.py`: single, comparison and forecast-overlay videos, stills,
  event logs.
- `scripts/render_highlights.py`: the captioned montage.
- `tracking/annotation_edits.py` and `scripts/annotation_editor.py`: manual corrections as
  review entries, CSV round trip, `identity_uncertain`.
- `scripts/forecast_ground_truth.py`: 8+12 windows at 0.4 s from validated trajectories,
  through the Phase 4 adapter checks; CV only.
- 13 synthetic tests.

**Videos are private and local:** `/Users/yushan/Desktop/Go2_Navigation/data/outputs/mac/videos/`.

**For Codex:**
- Use the videos to inspect cluster merging. Most red events coincide with the box or the
  person merging into a background cluster.
- The proposal-level work from the ground-truth handoff remains the priority.
- Do not tune on A1, A2, B or C.

**For Claude and the user:**
1. Human review in the editor (A2, B, C first).
2. Record held-out H1–H3.
3. Place the Phase 4 / 4.1 checkpoints under `data/outputs/forecasting/` and add learned
   predictions to `forecast_ground_truth.py`, through the existing adapters only.
4. Camera calibration, before any camera↔LiDAR association.

**Forecasting blocker:** only 6 independent identity-validated windows (one person, B and A2).
That supports plumbing checks only.

## Phase 3.8 ground-truth handoff (2026-11-08)
Branch `phase3-ground-truth-validation`, from `c643422`. Do not merge main or change the default
without approval. Read [TRACKING_GROUND_TRUTH](TRACKING_GROUND_TRUTH.md) first. The Phase 3.7
section below is historical.

**What exists now:**
- An annotation workflow, `scripts/annotate_ground_truth.py` (`cache` / `propose` / `sheet`). It
  uses LiDAR background subtraction, a measured static footprint and LiDAR-defined visibility,
  and the camera only for review.
- The schema and metrics, `src/go2_perception/tracking/ground_truth.py`: CLEAR-MOT-style matching
  that keeps the previous match, IDF1, IDF1 per visit, switches, fragmentations, duplicates,
  velocity error, motion confusion and recovery after occlusion.
- `scripts/evaluate_ground_truth.py`: scoring, proposal-quality and ID-change diagnosis, and an
  oracle-segmentation diagnostic.
- Annotations for B, A2, C and A1. They are local only, in `data/annotations/`; the specs there
  record intervals and review notes.
- 11 synthetic tests.

**Measured, original → 3.6 → 3.7:**
- IDF1: B 0.33/0.40/0.39, A2 0.30/0.60/0.53, C 0.29/0.60/0.41, A1 0.90/0.90/0.86.
- 3.6 is the best identity profile; 3.7 has fewer duplicates but lower coverage and more A1
  flicker.
- None is default-ready. Keep `tracking.yaml` as the default.

**Bottleneck: proposals.**
- Box merged with background in 14–33% of visible frames, and with the person in another 12%
  in B.
- Person merged in 23% of frames in A2.
- Splits are rare. Odometry is ruled out as a major cause in C.
- **Oracle segmentation diagnostic.** The untuned original tracker on oracle proposals reaches
  IDF1 B 0.67, A2 0.69, C 0.54 and A1 0.98.
  - That beats every tuned profile on real proposals in B, A2 and A1.
  - 3.6 and 3.7 do worse than the original on clean proposals; for example, A1 3.6 has 44 flicker
    switches.

**Next for Codex.** Improve LiDAR proposals before more tracker tuning:
1. Separate low objects from walls and counters: height-profile or background-aware splitting,
   a smaller or adaptive `eps` near large planar clusters.
2. Separate a person from an adjacent object.
3. Then re-check the tracker profiles on the improved proposals. On clean proposals the
   original tracker beats 3.6 and 3.7.
4. Do not tune on A1, A2, B or C again. They are development data now.

**Next for Claude and the user:**
1. A human pass over the review sheets (time ranges listed in the report).
2. Record held-out H1–H3 (two people, a taped start/end pose), annotate with unchanged
   thresholds, and score once.
3. Camera calibration, to enable semantic gating.

## Phase 4.1 forecasting handoff (merged from `phase4-social-forecasting` at 5979a76)

Branch `phase4-social-forecasting`, based on verified Phase 4 `447f164e808fee381fd3e9dbe8244583a0bed2f4`. No merge into main. Implemented observation-only multi-pedestrian extraction with partial-history masks, an 87,128-parameter shared temporal encoder with relative-position/mean neighbor pooling, dedicated checkpoint/training/evaluation interfaces, robustness studies, research plots, and a guarded offline multi-track adapter. Original forecasting, TrackOutput/ForecastOutput, tracking internals and Go2 hardware adapters are unchanged.

Completed: all five scene-identifier-held-out folds for original and social models with seeds 42/43/44 (15,061 paired windows per seed), same-weight social-disabled diagnostics, twenty validation robustness cases across clean/joint/noise-only/mask-only training, and separate CPU latency measurement. Every single-agent prediction and target/key population was checked against the independent Phase 4 evaluator. Full suite: **259 passed, one hardware skipped in 17.37 s**; Ruff passes on forecasting code/scripts/tests. No dependency changes (Python 3.12.11, torch 2.6.0, numpy 1.26.4, matplotlib 3.11.2).

**Negative result:** macro ADE/FDE, mean ± sample SD over three seeds: social 0.5812 ± 0.0162 / 1.1951 ± 0.0247 m; original 0.5565 ± 0.0118 / 1.1534 ± 0.0200 m; CV 0.5214 / 1.0933 m. Disabling social input improves macro errors in every seed. Noise-only augmentation improves one-fold noisy validation ADE to 0.3924 m, still behind noisy CV 0.3371 m. No claim of overall learned-model improvement or independent-window statistical significance.

Read [social architecture](SOCIAL_FORECASTING_ARCHITECTURE.md), [actual experiments and commands](SOCIAL_FORECASTING_EXPERIMENTS.md), [predeclared protocol and follow-up controls](SOCIAL_FORECASTING_PROTOCOL.md), [metrics](SOCIAL_FORECASTING_METRICS.json), and [Go2 contract](FORECASTING_INTEGRATION.md). Published artifacts are public-data aggregates/selected plots; raw annotations, weights, predictions and private recordings remain ignored. Known limitations: weak mean pooling, one-seed robustness controls, dependent windows, scene/domain shift, unverified dataset redistribution rights, no clean-install/GPU/live-Go2 validation. No navigation control.

Next: keep CV as the reference; Claude validates real pedestrian identity segments and stable metric frames. Then run matched held-out offline Go2 comparisons with explicit externally validated IDs. Repeat noise-only augmentation across seeds before considering more architecture. Historical phase states follow; their current/next claims apply to their original milestones.

# Phase 4 historical state — 2026-11-09

Branch `phase4-trajectory-forecasting`, based on Phase 3.7 `c6434229c54ee445763d8ceb1f5ef56b0f5c50da`; main is not merged. Added public ETH/UCY trajectory preparation, independent CV and Phase 3 baseline adapter, lightweight PyTorch Transformer, deterministic training/checkpoints, paired evaluation, validation ablations, plotting and guarded offline TrackOutput integration. Tracking/sensor internals remain unchanged; Claude's Phase 3.8 work is separate.

Verified: five fixed scene-identifier-held-out CPU runs (15,061 windows), validation-only history/capacity/representation/noise/missing-input studies, synthetic adapter demo and 231 tests passed/1 hardware skipped. The learned model **loses overall**: macro ADE/FDE 0.5701/1.1762 m vs CV 0.5214/1.0933 m. No Go2 forecasting accuracy or GPU validation is claimed. Data/weights remain local, public aggregate results and selected plots are committed. Python 3.12.11, torch 2.6.0, numpy 1.26.4, optional matplotlib 3.11.2; existing core pins unchanged.

Read [forecasting architecture](FORECASTING_ARCHITECTURE.md), [dataset protocol](FORECASTING_DATASETS.md), [commands and actual experiments](FORECASTING_EXPERIMENTS.md), and [integration contract](FORECASTING_INTEGRATION.md). Known limitations: one seed, correlated windows, no social context or uncertainty, unverified dataset redistribution rights, no clean-install test, and untrusted real tracker identities. Do not train from unvalidated Go2 tracks. Next: Claude validates identities/frames/visibility, then evaluate public-trained models and CV on held-out ground truth; extend forecasting research using validation-only decisions. Earlier phase sections below are historical.

## Phase 3.7 refinement handoff (historical)
Branch `phase3-tracking-refinement` starts at `de043d1`, containing Claude's real comparison and
Codex's Phase 3.6 work. Do not merge main or promote a profile to default without approval.
Read [TRACKING_REFINEMENT](TRACKING_REFINEMENT.md) first, then the unchanged
[TRACKING_REAL_COMPARISON](TRACKING_REAL_COMPARISON.md) for the regression diagnosis.

New code constrains reactivation by last observation/time/speed; suppresses repeatedly overlapping,
velocity-consistent weaker tracks; and separates moving activation/release from stationary dwell.
Use `configs/tracking_refinement.yaml`; both earlier profiles remain available. Public sensor and
tracking interfaces are unchanged. Source tests: 192 passed/1 hardware skipped. Seed 42's split
failure persists; first-miss Lost status reduces occlusion IDF1, and irregular-motion recall has small regressions. Dependencies unchanged.

The exact `compare_tracking_real.py --reuse-cache --serial` command is in TRACKING_REFINEMENT.
Use its new read-only --cache-root and a fresh output directory: existing run outputs now refuse
overwrite; --analyze reads them. Explicit PYTHONPATH=src avoids this Mac's stale installed package.
No private samples are committed. Claude owns hardware/calibration verification and annotation;
next label identities/visibility, record two close people and validate transitions on held-out data.
Do not start learned forecasting using these unverified identity targets. Earlier handoffs below
are historical; their 'next' tasks should be read in that context.

Final offline replay: 3,924 frames; original/default outputs remain identical. Walking MOVING
proxy improves about 39%→52%, and long reactivations fall in all recordings. Much of the duplicate
count reduction is removal of coasting Confirmed states; observed-only duplicates barely change.
A2/C dominant-ID continuity regresses relative to Phase 3.6, and A1 flicker increases. This is
not a uniform fix or a default-ready profile; labels/held-out testing remain required. Full measured
results and runtime are in TRACKING_REFINEMENT. No private samples are committed.

## Phase 3.6 robustness handoff (historical)
Branch `phase3-tracking-robustness` starts at `feature/macos-go2-phase3-validation` / `ecfaae3`.
Do not merge main without approval. Read [TRACKING_ROBUSTNESS](TRACKING_ROBUSTNESS.md) first,
then [TRACKING_ARCHITECTURE](TRACKING_ARCHITECTURE.md) and Claude's unchanged
[PHASE3_REAL_VALIDATION](PHASE3_REAL_VALIDATION.md). The previous macOS handoff below is historical.

Codex added opt-in `configs/tracking_robustness.yaml`, conservative historical-box fragment
recovery, time expiry for confirmed tracks, adaptive gating, motion dwell, lifecycle diagnostics,
26 regressions and `scripts/benchmark_tracking_robustness.py`. No sensor/hardware code changed.
`PYTHONPATH=src .venv/bin/python -m pytest -q`: **161 passed, 1 hardware skipped**.
Always set PYTHONPATH on this Mac to avoid the old installed package; `.venv-go2` can be used
instead. Install instructions and pinned dependencies remain in SETUP.md.

Use the exact synthetic/private-replay commands and aggregate comparisons in TRACKING_ROBUSTNESS.
Private cache and per-frame results live under ignored `data/outputs/robustness/`; no recordings
are needed for the synthetic benchmark. Seed 42 still has three split-object switches; brief-stop
classification recall regresses. Baseline defaults are deliberately unchanged.

Claude owns physical sensor access, timing/calibration validation and labelled real evaluation.
Next: annotate A2/B/C locally, inspect recovery/association failures with raw proposals, run D
(two nearby people), measure slow-motion/stop transitions and pose drift. Keep private sensor
files out of Git. Preserve pose-gate skips/resets. Integration continues through Detection3D,
RobotPose and the unchanged tracker.update signature; no detector internals need modification.
Phase 4 dataset tooling can use current outputs, but do not treat unstable IDs as reliable learned
forecasting targets. Hardware live validation and calibrated RGB semantic fusion remain outstanding.

**Real-data comparison done (Claude):** [TRACKING_REAL_COMPARISON](TRACKING_REAL_COMPARISON.md),
harness `scripts/compare_tracking_real.py`, local outputs in `data/outputs/mac/tracker_comparison/`.
The default path is bit-identical to `ecfaae3`; the improved profile is ready as opt-in, not as
default. Ablations attribute the effects:
- **Time expiry** gives the box-continuity gain, but also the extra duplicates and flicker.
- **The 2.5 m adaptive gate** causes most reactivations more than 1 m from the last observation.
- **Dwell plus the 0.25 m/s threshold** removes false MOVING labels, but also cuts walking MOVING
  recall from 62% to 39%.

Next for Codex:
1. Suppress or merge a Lost track that overlaps a Confirmed track.
2. Bound the reactivation radius by elapsed time × a plausible speed.
3. Re-tune dwell for walking people.
4. Rerun `compare_tracking_real.py --reuse-cache --serial`. It needs the local recordings, so it
   runs on the Mac.

Next for Claude/user: hand-label box and person IDs on A2/B/C so these proxies can be checked.

## macOS validation handoff (historical baseline)
Branch `feature/macos-go2-phase3-validation`; do not merge without user approval. The Mac reads the
Go2 natively; setup, rates and pitfalls are in [MAC_CONNECTION](MAC_CONNECTION.md). Use
`.venv-go2` for hardware and `--iface en9` (the config default is Linux `eno2`). Real tracking
results and gaps: [PHASE3_REAL_VALIDATION](PHASE3_REAL_VALIDATION.md).

```bash
.venv-go2/bin/python scripts/go2_record.py --iface en9 --seconds 60 --name test_a
.venv-go2/bin/python scripts/run_tracking.py --input data/recordings/test_a --output data/outputs/mac/test_a --window-s 0.3 --video
.venv-go2/bin/python -m pytest -q
```
Pose gating: `configs/tracking.yaml` `pose_alignment`; skipped frames in `skipped_frames.jsonl`.

**Next for Codex** (evidence: PHASE3_REAL_VALIDATION, recordings `test_a1`, `test_a2`, `test_b` on
the Mac): (1) person identity: one walking person became 19 IDs; examine leg clustering and
coasting; (2) occlusion and viewpoint: a static box is re-identified each time a person stands in front,
and every 5-15 s while the robot turns (Test C, 22 IDs) even though its odom position holds to 6-9 cm;
(3) motion classifier: steady velocity sigma about 0.14 m/s makes STATIONARY need speed under about
0.01 m/s; (4) clutter fragments on walls give false MOVING. Keep synthetic metrics intact and get
labels before tuning. **Next for Claude/user:**
Test D (two people), a Test C repeat with a taped end pose for odometry drift, Mac
interface override for `go2_live.py`/hardware test, camera calibration.

## Phase 3 handoff (2026-11-06)

Changes are committed on `phase3-tracking` from main `dc0f85b`; do not merge without user approval.
New packages: tracking, localization and visualization; core/tracking_types.py; three tracking
scripts and configs/tracking.yaml. All original modules/tests remain intact. Validation:
127 passed/1 hardware skipped. Reference synthetic IDs are stable; additional noisy seeds have
four switches across forty sequences. No real temporal or live tracking accuracy is claimed.

```bash
source .venv/bin/activate
uv pip install -r requirements.txt
python scripts/generate_tracking_test_data.py
python scripts/run_tracking.py --input data/sample/tracking_sequence/crossing --config configs/tracking.yaml --output data/outputs/tracking_run --video
python scripts/evaluate_tracking.py --input data/sample/tracking_sequence --output data/outputs/tracking_eval.json
pytest -q
# On the recording workstation, with real bracketed odometry:
python scripts/run_tracking.py --input data/recordings/stage_c_person --output data/outputs/tracking_person --window-s 0.3
```
Choose fresh output directories. *(2026-11-07: unposed frames are now skipped with a reason and
reported; a recording with no posable frame still raises. See TRACKING_ARCHITECTURE.md.)*
Do not substitute identity poses. Use `--mode fixed_sensor` only with an explicit fixed-sensor
assumption, and interpret velocities as sensor-relative. Full JSONL input, types, covariance,
frame conventions, config meanings and output contracts: TRACKING_ARCHITECTURE.md.

**Claude's exact next steps:**
1. Keep adapter/hardware ownership. Record cloud_base and odometry with real acquisition stamps;
   provide `go2_robot` clock and `T_odom_base` (xyzw source quaternion). No host/RTP substitution.
2. For live normalized detections, call `from_lidar` / `from_fusion` then the persistent
   MultiObjectTracker. Supply a RobotPose mapping the **actual detection frame** to odom at
   that timestamp. If detections were levelled to base_link_level, use yaw+translation, not
   a second roll/pitch rotation. Buffer/interpolate odometry upstream with validated gaps.
3. Validate stationary world objects while translating/rotating the robot, then known moving
   targets. Measure velocity error and ID continuity against annotated ground truth.
4. Review the documented exact-sample-after-gap PoseBuffer issue. Phase 3 handles exact observed
   samples in its replay wrapper; shared hardware code was not silently patched.
5. Validate dropouts, clock discontinuities, scan de-skew assumptions, odometry drift and dynamic
   ground height. Ground/clustering defaults still depend on standing height/scene density.
6. Complete camera intrinsics, distortion/rectification, extrinsics and capture-time alignment
   before associating RGB semantics with LiDAR trajectories. No synthetic calibration on real data.

**Phase 4 interface:** consume TrackOutput history, timestamp/frame, velocity/covariance/status;
replace `predict_trajectory` with a learned predictor returning timestamped positions and
uncertainty in the same frame. Establish annotated metrics and robustness first. Robot control,
learned networks, planners and navigation are not implemented by Phase 3.

Historical handoffs below describe their original phases, not the current task status.

## Main integration handoff
Local merge verification: **70 passed, 1 hardware test skipped** (macOS CPU).
Work from `main`, which contains both phase commits plus the merge validation tests/docs.
`tests/test_go2_replay.py` now exercises the real committed cloud through recording and
the replay CLI, with and without a window. Its stationary identity poses are synthetic
test scaffolding, not measured motion evidence. See [MERGE_VALIDATION](MERGE_VALIDATION.md).
Adapter/hardware ownership remains unchanged. Current priority: measured calibration and
synchronization, then labelled clustering and motion validation.

## What Codex should do next
Claude's Phase 2 is on `feature/go2-sensor-integration`. Read [PROJECT_STATE](PROJECT_STATE.md), then
[SENSOR_VALIDATION](SENSOR_VALIDATION.md). The original Phase 2 priorities are in
[INTEGRATION_REPORT](INTEGRATION_REPORT.md): replay regression test (now added for the committed fixture), `RGBMessage.K` decision, cluster
tuning on real frames, odometry-based ground height, false-positive handling, then tracking.

**Do not** use `rt/utlidar/cloud_deskewed` as a scan, or use RTP timestamps as capture times. **Do not**
use synthetic calibration on real data, or add any publisher, RPC client or motion command to
`adapters/go2`. Claude keeps ownership of `adapters/go2`, `scripts/go2_*.py`, `configs/go2_*.yaml`, and
hardware validation; coordinate before changing them. Perception modules remain yours, unchanged.

Recordings are local-only and contain private home images. They are not on GitHub. Without the
capture workstation, record your own with `scripts/go2_record.py`, or use
`tests/fixtures/go2/real_cloud_pair.npz` (one real LiDAR frame pair).

Start here on the workstation:
```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/go2_replay.py --recording data/recordings/stage_c_person --window-s 0.3 --output data/outputs/replay
```
Hardware work still pending from Claude and the user: link-drop test, calibration, motion validation.

# Phase 1 handoff record (historical; superseded by Phase 2 above)
## Purpose and ownership
Build an offline RGB + LiDAR perception foundation for future Go2 dynamic navigation.
Codex implemented repository architecture, detectors/processors, geometric fusion, CLIs,
synthetic fixtures, automated tests and documentation. Claude independently owns Ethernet,
DDS/SDK/ROS 2 connection, verified RGB/LiDAR/IMU/odometry streams and live validation.
No Claude-owned integration files existed when this repository was initialized. Coordinate
before changing such files once integration begins. No robot commands belong in Phase 1.

## Current repository
Branch `codex/phase1-perception`; implementation in src/go2_perception, CLI scripts in scripts,
YAML in configs, tests in tests. Everything operates without hardware. Real calibration is
absent; synthetic calibration is conspicuously marked. No ROS runtime or fabricated topics.
Results: 44 unit tests passed (final run 3.37s); lint and dependency compatibility passed; CPU YOLO processed a real image (6 detections); synthetic
LiDAR found 2 clusters; synthetic fusion matched 2; public PCL cloud processed but default
segmentation yielded one broad cluster/no ground. No real paired fusion or live validation.
See PROJECT_STATE.md and EXPERIMENTS.md for exact versions, sources, hashes and measurements.

## Install/run
From repository root:
```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
pytest -q
python scripts/generate_test_data.py
python scripts/run_lidar.py --input data/sample/test.pcd --output data/outputs/lidar
python scripts/run_rgb.py --input data/sample/bus.jpg --output data/outputs/rgb
python scripts/run_fusion.py --image data/sample/synthetic.jpg --points data/sample/test.pcd --detections data/sample/synthetic_detections.json --calibration configs/synthetic_calibration.yaml --rgb-timestamp 1 --lidar-timestamp 1 --clock-domain synthetic --output data/outputs/fusion
```
Download bus.jpg / yolov8n.pt with exact commands in EXPERIMENTS.md. Outputs and inputs are
ignored by Git and must be generated/downloaded on a new checkout. For macOS ARM64 exact
transitive versions, use `uv pip install -r requirements-macos-arm64.lock`. Linux/CUDA has
not been tested. Existing local .venv uses a runtime under /private/tmp; recreate if broken.

## Integration steps for Claude
1. Read SENSOR_INTERFACE.md in full, then core/types.py, io/sensor_interface.py and
   fusion/calibration.py. ARCHITECTURE.md describes every public entry point.
2. Verify actual SDK/ROS topics, data layout/endian, units, BGR/RGB encoding, timestamps,
   clock domains, frame axes, scan duration and calibration source. Document evidence.
3. Rectify images, provide corresponding K, and measure camera-to-LiDAR extrinsics
   (`T_lidar_camera`, NOT its inverse). Use explicit measured frame names/resolution.
4. Convert observations to RGBMessage/LidarMessage; call process_rgb/process_lidar with
   persistent detector/processor objects, then FusionProcessor.fuse. Check message K
   matches calibration. Pair timestamps in a verified common clock upstream.
5. Align scan +Z with gravity, configure expected ground height, and tune sparse-cloud
   filtering/DBSCAN. Motion-compensate accumulated windows before passing them to core.
6. Record an offline Go2 pair and run overlays before any live loop. Add integration tests
   for this evidence and update PROJECT_STATE.md, AGENT_HANDOFF.md and EXPERIMENTS.md.

## Open questions and issues
Actual topic names, QoS, sensor clock conversions, frame tree, optical distortion model,
intrinsics/extrinsics, mounting height and per-point timing are unknown. Determine whether
streams provide raw or already transformed points/images. No live synchronization queue,
clock conversion, de-skew or tracking is implemented. Nonzero distortion raises ValueError.
PLY/PCD intensity is not retained; cluster boxes describe only observed points.

No failing unit tests remain. Regression reproduction:
`pytest -q tests/test_lidar.py` covers nearby elevated obstacles incorrectly treated as ground
by the initial implementation; the height-band support fix now passes. Parameter sensitivity
is intentionally demonstrated by the nearby-cluster test (large eps merges them).
`python scripts/run_lidar.py --input data/sample/table_scene_lms400.pcd --output data/outputs/lidar_real`
reproduces the public scene's one broad cluster/no ground with defaults; do not call this
validated segmentation. First YOLO import may report a nonfatal font-cache warning.

The next milestone is **recorded, synchronized Go2 RGB–LiDAR replay with measured calibration**,
followed by live read-only inference. Autonomous locomotion, learned forecasting and planning
remain out of scope.

## Phase 4 continuation checklist

1. Use the `phase4-trajectory-forecasting` branch and read FORECASTING_EXPERIMENTS.md for the complete installation/reproduction commands. Run the synthetic smoke configuration before downloading annotations.
2. Codex owns `src/go2_perception/forecasting/`, `scripts/*forecasting*.py`, forecasting configs and tests. Core tracking and Go2 adapters were not rewritten. Claude retains sensor access, real calibration, ground truth and hardware validation.
3. For a real offline trial, establish stable metric odometry coordinates and reliable pedestrian IDs, retain at least 2.8 seconds of observed history, then use the explicit bounded resampling adapter. Keep all real recordings local.
4. Compare CV and learned forecasts on the same manually validated held-out windows. Never treat the public-data results or synthetic demo as proof of Go2 forecasting accuracy. Resolve fragmentation/ID switches before collecting training labels.
5. Reproduce known limitations with the documented five-fold command (model loses overall) and the noise ablation (model error rises substantially). Unknown data redistribution rights, GPU execution and clean-install portability remain unresolved. Do not merge into main or promote the learned model to navigation without separate approval and validation.

## Phase 4.1 continuation checklist

- Code belongs to the independent Phase 4.1 branch; do not merge main without approval. Claude's `phase3-ground-truth-validation` checkout and sensor/annotation responsibilities are separate. No hardware or tracker internals were modified.
- Start with SOCIAL_FORECASTING_EXPERIMENTS.md for exact install/reproduction commands. Reproduce Phase 4 checkpoints first; then use `run_social_forecasting.py --stage single`, `--stage social`, `--stage robustness`, and `--stage robustness_controls`. Use fresh output directories. Summary auditing requires local prediction NPZ files from both runs.
- Use `social_training.load_social_checkpoint` for social weights. Existing Phase 4 weights/API remain supported by the original loader. `forecast_multitrack` returns the unchanged ForecastOutput plus explicit context/masks/rejection reasons. Pass checkpoint social settings and externally reviewed `validated_ids` for the current snapshot; never infer reliable pedestrian identity from anonymous clusters or Confirmed status alone.
- The model loses overall. Reproduce the five-fold/three-seed comparison and noisy validation table before making a research claim. Do not choose a winning seed or tune against those held-out results. A future experiment needs a declared validation-only decision process.
- Physical Go2 identity quality, camera–LiDAR calibration, GPU execution and clean-install portability remain unresolved. Complete Claude's ground-truth work before domain-transfer evaluation; do not train on unannotated tracks or issue movement commands.
