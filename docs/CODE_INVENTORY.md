# Navigation source and evidence inventory

The inventory starts at [the complete research publication](https://github.com/binbin-ovo/Go2_Navigation/tree/91e98390baea3ec535a538d1fa5f3edae7024ae4). Each archived filename still resolves at that commit. The operator checkout, controller, SDK, environments and recordings are outside this table.

ACTIVE files participate in a navigation launch. SHARED includes retained libraries, independent checks, test fixtures and utilities. HISTORICAL files remain reproducible together at the pinned publication. An absent parent import did not establish dead code. UNKNOWN identifies legacy field utilities whose independent operator usage was not observed in this offline session. Their original coordinates are available in the published tree and operational checkout. No file was retired as proven dead solely from an absence of imports.

I traced Python imports (including imports inside functions), package initializers, the shell launchers, interpreter subprocesses, dynamic guard and SDK loads, filename literals and test fixtures. The owner or consumer shown is an example, not an exhaustive list. Application-level regressions and recorded evidence are described in [the walkthrough](CODE_WALKTHROUGH.md).

## New application paths

| Path | Classification and use |
|---|---|
| `app.py` | ACTIVE · loopback session lifecycle, mode selection, gated operator entry |
| `navigation.py` | ACTIVE · root link to the packaged policy implementation |
| `sensors.py` | ACTIVE · root link to the packaged sensor implementation |
| `src/go2_perception/navigation.py` | SHARED · requested text, independent grounding and movement selection shared by five-move and continuous |
| `src/go2_perception/sensors.py` | SHARED · named configuration, capture lifecycle, timestamped RGB, cloud and pose for observation and supervision |
| `src/go2_perception/vln/sensor_buffers.py` | SHARED · bounded camera tap and paced replay feeder |
| `src/go2_perception/vln/session_runner.py` | SHARED · core process, sensor/session startup, recording and orderly shutdown |
| `src/go2_perception/vln/supervised_launch.py` | SHARED · capability validation and construction of each supervised strategy |
| `src/go2_perception/vln/bound_proposal.py` | SHARED · one-action and episode image, capture and model identity |
| `src/go2_perception/vln/web_modes.py` | ACTIVE · cancels and closes the offline session before switching modes |
| `src/go2_perception/vln/mode_controls.html` | ACTIVE · loopback mode selector, stop control, error and transition status |
| `src/go2_perception/vln/viewer.html` | SHARED · four panels and explicit extension points |
| `src/go2_perception/vln/review.css` | SHARED · viewer and trace presentation |
| `hardware/launch_supervised.sh` | SHARED · original live opt-ins and evidence arguments for both operator shortcuts |
| `tests/test_application.py` | SHARED · public lifecycle, replay, mode transitions, stale sessions and baseline equivalence |
| `tests/fixtures/supervised/five_move_behavior.json` | SHARED · expected public synthetic evidence captured before refactoring |

## Source, scripts and tests

| Original path | Class | Disposition and evidence |
|---|---|---|
| `hardware/camera_heading.py` | SHARED | Camera optical-axis heading in base_link from existing static plane recordings (read-only, offline) · referenced by `hardware/make_capability.py` |
| `hardware/capture_latency.py` | SHARED | Upper bound on Go2 front-camera exposure-to-receipt latency (read-only; hardware environment) · referenced by `hardware/supervised_guards.py` |
| `hardware/control_log_summary.py` | ACTIVE | Summarise audited go2_hl_control move logs (read-only, no SDK) · referenced by `scripts/replay_effective_motion.py` |
| `hardware/correspondence_check.py` | SHARED | Offline check of the target-correspondence gate (local_feature_roi_v1) on recorded live sessions |
| `hardware/make_capability.py` | ACTIVE | The two live shell launchers generate a local hash-pinned capability manifest here |
| `hardware/supervised_guards.py` | ACTIVE | Capability-pinned guard provider loaded with importlib in the executor |
| `scripts/annotate_ground_truth.py` | HISTORICAL | Sensor-based ground-truth annotation for Go2 tracking recordings (local, private outputs) · referenced by `scripts/annotation_editor.py` · pinned research tree |
| `scripts/annotation_editor.py` | HISTORICAL | Interactive ground-truth editor: step through RGB + BEV frames and correct annotations · pinned research tree |
| `scripts/audit_lightnav_repetition.py` | HISTORICAL | Read-only comparison of saved native tensors for repeated LightNav outputs · pinned research tree |
| `scripts/audit_pretrained_inputs.py` | HISTORICAL | Validate Claude's exact packages and hash every package/baseline file, read-only · pinned research tree |
| `scripts/benchmark_lidar_windows.py` | HISTORICAL | Raw odom-frame acquisition windows; GT regions are diagnostic measurement ROIs only · pinned research tree |
| `scripts/benchmark_social_forecasting.py` | HISTORICAL | Dedicated quiet-process warm timing; run after training jobs have finished · pinned research tree |
| `scripts/benchmark_stable_scene.py` | HISTORICAL | Synthetic scene stability and accumulation tradeoffs; no annotations enter perception · pinned research tree |
| `scripts/benchmark_tracking_robustness.py` | HISTORICAL | Compare a Git baseline with the working tracker on identical local inputs. Never uploads data · pinned research tree |
| `scripts/check_continuous_evidence.py` | SHARED | Inventory a five-move evidence bundle for human review. No control or sensor imports · referenced by `tests/test_continuous_integration.py` |
| `scripts/check_continuous_memory.py` | SHARED | Exercise the controller and bounded recorder repeatedly without loading a model or any sensor · referenced by `tests/test_continuous_integration.py` |
| `scripts/check_go2_camera_frame.py` | HISTORICAL | Verify real Go2 camera frames and metadata through the advisory bridge, no model · pinned research tree |
| `scripts/compare_grounding_prompts.py` | SHARED | Fixed, sequential replay of the November 19 box prompt comparison. Model only |
| `scripts/compare_tracking_real.py` | HISTORICAL | Replay a Git-baseline tracker and the working tracker on identical local Go2 recordings · referenced by `scripts/evaluate_ground_truth.py` · pinned research tree |
| `scripts/demo_go2_forecasting.py` | HISTORICAL | Offline demonstration only; never trains on Go2 tracks or sends robot commands · pinned research tree |
| `scripts/demo_vln_supervised_mock.py` | SHARED | Drive ONE synthetic cycle through the real loopback UI/worker/executor IPC · referenced by `tests/test_supervised_integration.py` |
| `scripts/diagnose_forecasting.py` | HISTORICAL | Inspect frozen original checkpoints without changing the held-out protocol · pinned research tree |
| `scripts/diagnose_waypoint_execution.py` | HISTORICAL | Model-free numeric paths, original-scene previews and official input images · pinned research tree |
| `scripts/download_forecasting_data.py` | HISTORICAL | Download the authors' public annotation archive, checksum it, extract only manifest files · pinned research tree |
| `scripts/evaluate_forecasting.py` | HISTORICAL | evaluate forecasting module · pinned research tree |
| `scripts/evaluate_ground_truth.py` | HISTORICAL | Score tracker variants against local ground-truth annotations and diagnose ID failures · referenced by `scripts/evaluate_stable_scene_real.py` · pinned research tree |
| `scripts/evaluate_pretrained_vln.py` | HISTORICAL | Evaluation-only comparison and local panels; never loads a learned model · pinned research tree |
| `scripts/evaluate_stable_scene_real.py` | HISTORICAL | Offline annotated diagnostic comparison. Never passes annotations into perception · referenced by `scripts/phase5b_compare.py` · pinned research tree |
| `scripts/evaluate_temporal_vln.py` | HISTORICAL | Score saved proposals against separate references; keep correlated recordings grouped · referenced by `tests/test_temporal_vln_evaluation.py` · pinned research tree |
| `scripts/evaluate_tracking.py` | HISTORICAL | Evaluate annotated detection JSONL; unannotated data gets runtime metrics only · pinned research tree |
| `scripts/forecast_ground_truth.py` | HISTORICAL | Phase 4.2 readiness: forecasting windows on identity-validated real pedestrian trajectories · pinned research tree |
| `scripts/fsd_viewer.py` | HISTORICAL | FSD-style 3D viewer for go2-scene-v1 (interactive window, MP4 export or PNG still) · referenced by `scripts/phase5b_render_reels.py` · pinned research tree |
| `scripts/generate_scene_fixture.py` | HISTORICAL | Standalone deterministic renderer fixture; no sensor/perception imports needed · referenced by `tests/test_scene_interface.py` · pinned research tree |
| `scripts/generate_test_data.py` | HISTORICAL | generate test data module · pinned research tree |
| `scripts/generate_tracking_test_data.py` | HISTORICAL | Generate explicitly synthetic timestamped detections, poses and truth · pinned research tree |
| `scripts/generate_vln_episode_fixture.py` | SHARED | Moved to `legacy/generate_vln_episode_fixture.py`.  Generate the deterministic synthetic episode contract fixture (no model required) · referenced by `tests/test_vln_episodes.py` |
| `scripts/go2_camera_bridge.py` | ACTIVE | Read-only RTP camera subprocess |
| `scripts/go2_discover.py` | UNKNOWN | External hardware-session use unverified.  Passive DDS discovery of the Go2: lists every remote publication (topic, type, QoS) · pinned research tree |
| `scripts/go2_lidar_bridge.py` | ACTIVE | Read-only DDS subprocess for LiDAR and odometry |
| `scripts/go2_live.py` | HISTORICAL | Live read-only perception: latest RGB frame -> YOLO; latest cloud_base -> LiDAR processor · pinned research tree |
| `scripts/go2_moving_object_study.py` | HISTORICAL | Measure how a moving object smears in ego-compensated LiDAR windows · pinned research tree |
| `scripts/go2_probe.py` | UNKNOWN | External hardware-session use unverified.  Read-only Go2 stream probe. Subscribes to sensor topics via the Unitree SDK and reports · pinned research tree |
| `scripts/go2_record.py` | UNKNOWN | External hardware-session use unverified.  Record a short read-only Go2 sequence and measure the streams while doing so · pinned research tree |
| `scripts/go2_replay.py` | SHARED | Moved to `legacy/go2_replay.py`.  Run the existing YOLO and LiDAR processors offline on a recording; no robot needed · referenced by `tests/test_go2_replay.py` |
| `scripts/go2_ros_env.sh` | UNKNOWN | External hardware-session use unverified.  go2 ros env module · referenced by `scripts/test_lidar.py` · pinned research tree |
| `scripts/go2_view_cloud.py` | HISTORICAL | Show a recorded LiDAR frame with clusters and boxes in Open3D (RViz2 is not installed) · pinned research tree |
| `scripts/go2_window_study.py` | HISTORICAL | Measure the tradeoff between single cloud_base frames and short compensated windows · pinned research tree |
| `scripts/inspect_lidar_scans.py` | HISTORICAL | Read-only acquisition and overlap audit; exports aggregates, never private points/images · pinned research tree |
| `scripts/inspect_vln_episode.py` | HISTORICAL | Minimal contract consumer; loads no model and no evaluation annotations · pinned research tree |
| `scripts/install_sdk.sh` | UNKNOWN | External hardware-session use unverified.  install sdk module · pinned research tree |
| `scripts/launch_effective_trial.sh` | ACTIVE | Operator entry for the separately authorized single-action strategy |
| `scripts/launch_five_move_episode.sh` | ACTIVE | Operator entry for the bounded live executor and capability preparation |
| `scripts/lidar_diag.py` | UNKNOWN | External hardware-session use unverified.  lidar diag module · pinned research tree |
| `scripts/lidar_state.py` | UNKNOWN | External hardware-session use unverified.  Read-only: prints the Go2 LiDAR's self-reported status via the Unitree SDK. Subscribes only · pinned research tree |
| `scripts/lightnav_worker.py` | ACTIVE | Native inference in the isolated model interpreter |
| `scripts/link_paired_views.py` | SHARED | Link two single-answer trials taken before and after a MANUAL viewpoint change · referenced by `tests/test_vln_interactive.py` |
| `scripts/make_pretrained_vln_smoke.py` | HISTORICAL | Deterministic two-frame public synthetic input, not a navigation accuracy test · pinned research tree |
| `scripts/patch_lightnav_local_runtime.py` | HISTORICAL | Create a separate, hash-audited CPU/MPS portability copy of pinned LightNav · pinned research tree |
| `scripts/phase5b_compare.py` | HISTORICAL | Original 0.3 s pipeline vs Phase 5A (and diagnostic variants) on A1/A2/B/C · pinned research tree |
| `scripts/phase5b_diagnose_walking.py` | HISTORICAL | Debug-mode diagnosis: why does Phase 5A miss (or not mark MOVING) a walking person? · pinned research tree |
| `scripts/phase5b_render_reels.py` | HISTORICAL | Phase 5B comparison videos from private recordings (1920x1080 H.264, git-ignored outputs) · pinned research tree |
| `scripts/phase5b_scene_tap.py` | HISTORICAL | Replay a recording through the unchanged Phase 5A pipeline and record Debug intermediates · referenced by `src/go2_perception/viewer/sources.py` · pinned research tree |
| `scripts/phase5b_synthetic_demo.py` | HISTORICAL | Synthetic viewer demo: run 5A's deterministic synthetic scenarios through the instrumented · pinned research tree |
| `scripts/phase5d_calibrate_intrinsics.py` | HISTORICAL | Calibrate Go2 front-camera intrinsics from a recording of a ChArUco target (offline) · pinned research tree |
| `scripts/phase5d_lidar_accumulation.py` | HISTORICAL | Compare ego-motion-compensated cloud_base accumulation windows on one recording · pinned research tree |
| `scripts/phase5d_lidar_flatness.py` | HISTORICAL | Check whether the Go2 LiDAR returns flat surfaces as flat (offline, read-only) · pinned research tree |
| `scripts/phase5d_make_charuco.py` | HISTORICAL | Write a printable ChArUco calibration target (PDF at exact scale) and its machine spec · pinned research tree |
| `scripts/phase5d_solve_extrinsics.py` | SHARED | Solve T_base_camera (camera optical -> base_link) from a static multi-pose ChArUco board capture · referenced by `hardware/camera_heading.py` |
| `scripts/phase5d_timing_report.py` | HISTORICAL | Measure Go2 stream timing from local recordings: rates, jitter, host-robot offset, drift and, · pinned research tree |
| `scripts/phase5d_vln_replay_demo.py` | HISTORICAL | Offline Go2 VLN observation demo: replay a recording, build, validate and serialize one · pinned research tree |
| `scripts/phase6_evaluate_grounding.py` | HISTORICAL | Run a declared offline case list. Semantic judgments must come from separate review · pinned research tree |
| `scripts/phase6_grounding_demo.py` | HISTORICAL | Offline Phase 6A terminal instruction -> inspected image-space proposals · pinned research tree |
| `scripts/prepare_pretrained_vln.py` | HISTORICAL | Download ONE pinned public inference implementation/checkpoint; never robot clients · pinned research tree |
| `scripts/prepare_temporal_vln_pilot.py` | HISTORICAL | Copy bounded private recording snippets into v1 episodes. No labels enter manifests · pinned research tree |
| `scripts/pretrained_vln_preflight.py` | HISTORICAL | Read-only runtime inventory. Does not load weights, install software or contact Go2 · pinned research tree |
| `scripts/profile_vln_latency.py` | SHARED | Moved to `legacy/profile_vln_latency.py`.  Offline fixed-cutoff profiling of saved advisory observations; no sensor access · referenced by `tests/test_vln_latency.py` |
| `scripts/read_scene.py` | HISTORICAL | Minimal standard-library-only renderer consumer; no perception imports required · referenced by `tests/test_scene_interface.py` · pinned research tree |
| `scripts/render_highlights.py` | HISTORICAL | Highlight montage of tracking successes and failures (private output, Git-ignored) · pinned research tree |
| `scripts/render_tracking_video.py` | HISTORICAL | Render synchronized RGB + LiDAR BEV + status diagnostic videos from private recordings · referenced by `scripts/annotation_editor.py` · pinned research tree |
| `scripts/replay_effective_motion.py` | ACTIVE | Trace saved LightNav predictions and controller feedback offline. Never dispatch · referenced by `scripts/run_vln_continuous.py` |
| `scripts/report_stable_scene.py` | HISTORICAL | Publish aggregate-only Phase 5A tables and figure; never read raw scene geometry · pinned research tree |
| `scripts/review_target_grounding.py` | SHARED | Write an immutable offline review of a closed five-move trial into a fresh private folder · referenced by `scripts/compare_grounding_prompts.py` |
| `scripts/review_vln_simulation.py` | HISTORICAL | Render evaluation-only panels for a saved synthetic step-and-wait rollout · pinned research tree |
| `scripts/review_waypoint_validation.py` | HISTORICAL | Evaluation-only panels/videos for every Phase 6D run, including failures · pinned research tree |
| `scripts/run_forecasting_ablation.py` | HISTORICAL | Predeclared small matrix, validation-only by design. Test evaluation is a separate command · pinned research tree |
| `scripts/run_forecasting_crossvalidation.py` | HISTORICAL | Fixed five scene-identifier-held-out runs; next scene is validation, remaining three train · referenced by `scripts/run_social_forecasting.py` · pinned research tree |
| `scripts/run_fusion.py` | HISTORICAL | run fusion module · pinned research tree |
| `scripts/run_lidar.py` | HISTORICAL | run lidar module · pinned research tree |
| `scripts/run_pretrained_vln.py` | HISTORICAL | Unchanged episode packages -> calibrated causal inputs -> isolated local model worker · pinned research tree |
| `scripts/run_rgb.py` | HISTORICAL | run rgb module · pinned research tree |
| `scripts/run_social_forecasting.py` | HISTORICAL | Fixed Phase 4.1 experiments; preserve original runs and never select on test scores · pinned research tree |
| `scripts/run_stable_scene.py` | HISTORICAL | Replay raw incremental cloud_base into scene-v1; never consumes annotations or RGB · referenced by `scripts/phase5b_scene_tap.py` · pinned research tree |
| `scripts/run_temporal_vln.py` | HISTORICAL | Offline causal episode -> pretrained temporal text proposal -> saved review evidence · referenced by `tests/test_temporal_vln_runner.py` · pinned research tree |
| `scripts/run_tracking.py` | HISTORICAL | Track timestamped detections or a Phase 2 Go2 recording, entirely offline · referenced by `tests/test_pose_gate.py` · pinned research tree |
| `scripts/run_vln_continuous.py` | ACTIVE | Supported offline simulation and immutable replay command |
| `scripts/run_vln_interactive.py` | ACTIVE | Existing command line preserved for observation and annotated review |
| `scripts/run_vln_shadow.py` | SHARED | Terminal instruction -> local observations -> read-only LightNav advisory · referenced by `tests/test_vln_live_shadow.py` |
| `scripts/run_vln_simulation.py` | SHARED | Moved to `legacy/run_vln_simulation.py`.  One small synthetic camera room: render smoke, optionally static closed loop · referenced by `tests/test_pretrained_worker.py` |
| `scripts/run_vln_supervised.py` | ACTIVE | Existing command line preserved for separately authorized supervised launch |
| `scripts/run_waypoint_suite.py` | HISTORICAL | Freeze a reviewed Phase 6D suite, then run every declared simulation condition · pinned research tree |
| `scripts/summarize_social_forecasting.py` | HISTORICAL | Aggregate seed variability without treating overlapping windows as independent samples · pinned research tree |
| `scripts/summarize_vln_latency.py` | HISTORICAL | Compare the two declared offline passes, including exact native tensor bits · pinned research tree |
| `scripts/summarize_waypoint_suite.py` | HISTORICAL | Read-only rescore/audit of saved Phase 6D traces; never inference or execution · pinned research tree |
| `scripts/synthetic_lidar_bridge.py` | ACTIVE | Synthetic or recorded LiDAR child |
| `scripts/test_camera.py` | UNKNOWN | External hardware-session use unverified.  Passive Go2 front-camera test: reads the H.264 RTP multicast stream and saves frames · pinned research tree |
| `scripts/test_lidar.py` | UNKNOWN | External hardware-session use unverified.  Read-only Go2 LiDAR test: lists topics, subscribes to the point cloud, reports stats, saves PLY + top-down PNG · pinned research tree |
| `scripts/train_forecasting.py` | HISTORICAL | Train on declared training scenes; select checkpoint on validation only · pinned research tree |
| `scripts/visualize_forecasting.py` | HISTORICAL | visualize forecasting module · pinned research tree |
| `scripts/visualize_social_forecasting.py` | HISTORICAL | Plot paired social forecasts with only observed neighbor context · pinned research tree |
| `scripts/vln_shadow_worker.py` | ACTIVE | Spawned by ProcessWorker across an interpreter boundary |
| `scripts/vln_supervised_executor.py` | ACTIVE | Executor child loads pinned guard and canonical controller by path |
| `scripts/windows_vln_preflight.ps1` | HISTORICAL | windows vln preflight module · pinned research tree |
| `src/go2_perception/__init__.py` | SHARED | Go2 offline perception components · referenced by `hardware/make_capability.py` |
| `src/go2_perception/adapters/__init__.py` | SHARED | Go2 offline perception components · referenced by `hardware/supervised_guards.py` |
| `src/go2_perception/adapters/go2/__init__.py` | SHARED | Read-only Go2 sensor adapter: real DDS/RTP messages -> Phase 1 perception inputs · referenced by `hardware/supervised_guards.py` |
| `src/go2_perception/adapters/go2/clock.py` | SHARED | Robot/host clock offset from ICMP timestamps (read-only ping; integer ms, mod 24 h) · referenced by `tests/test_go2_adapter.py` |
| `src/go2_perception/adapters/go2/dds_types.py` | ACTIVE | IDL for rt/utlidar/imu. The installed SDK lacks sensor_msgs Imu_; layout is standard · referenced by `src/go2_perception/adapters/go2/receivers.py` |
| `src/go2_perception/adapters/go2/pointcloud.py` | SHARED | PointCloud2 decoding and conversion to the Phase 1 LidarMessage contract · referenced by `hardware/supervised_guards.py` |
| `src/go2_perception/adapters/go2/receivers.py` | SHARED | Read-only live receivers. Requires eno2 link, the Unitree SDK (DDS) and PyAV (camera) · referenced by `scripts/go2_camera_bridge.py` |
| `src/go2_perception/adapters/go2/recording.py` | SHARED | Portable recording format: JPEG/PNG + NumPy + JSONL, loadable without ROS or the SDK · referenced by `scripts/go2_replay.py` |
| `src/go2_perception/adapters/go2/rtp_h264.py` | SHARED | Passive H.264-over-RTP depacketizing (RFC 6184 single NAL, STAP-A, FU-A) and decoding · referenced by `src/go2_perception/adapters/go2/receivers.py` |
| `src/go2_perception/adapters/go2/stats.py` | SHARED | Measured per-stream statistics. Only observed values are reported · referenced by `tests/test_go2_adapter.py` |
| `src/go2_perception/adapters/go2/window.py` | SHARED | Ego-motion-compensated short sliding window of body-frame LiDAR clouds · referenced by `src/go2_perception/vln/interactive_lidar.py` |
| `src/go2_perception/adapters/ros2/__init__.py` | HISTORICAL | Go2 offline perception components · pinned research tree |
| `src/go2_perception/core/__init__.py` | SHARED | Go2 offline perception components · referenced by `src/go2_perception/vln/calibration.py` |
| `src/go2_perception/core/geometry.py` | SHARED | geometry module · referenced by `src/go2_perception/lidar/clustering.py` |
| `src/go2_perception/core/tracking_types.py` | HISTORICAL | Tracking contracts: metres, seconds, m/s; covariance ordering is explicit · referenced by `scripts/benchmark_stable_scene.py` · pinned research tree |
| `src/go2_perception/core/transforms.py` | SHARED | transforms module · referenced by `src/go2_perception/vln/calibration.py` |
| `src/go2_perception/core/types.py` | SHARED | Public result types. Coordinates in metres; timestamps in seconds · referenced by `src/go2_perception/lidar/clustering.py` |
| `src/go2_perception/diagnostics/__init__.py` | HISTORICAL | Offline diagnostics around Phase 5A perception (evaluation side; never used at runtime) · referenced by `scripts/phase5b_diagnose_walking.py` · pinned research tree |
| `src/go2_perception/diagnostics/attribution.py` | HISTORICAL | Attribute a missed or non-MOVING walking person to the first failing Phase 5A stage · referenced by `scripts/phase5b_diagnose_walking.py` · pinned research tree |
| `src/go2_perception/diagnostics/scene_tap.py` | HISTORICAL | Instrumented replay of the unchanged Phase 5A StableScene · referenced by `scripts/phase5b_scene_tap.py` · pinned research tree |
| `src/go2_perception/forecasting/__init__.py` | HISTORICAL | Offline deterministic pedestrian forecasting; no hardware or control dependencies · referenced by `scripts/benchmark_social_forecasting.py` · pinned research tree |
| `src/go2_perception/forecasting/adapter.py` | HISTORICAL | Read-only offline adapter. Observed flags mean filter estimates supported by measurements · referenced by `scripts/demo_go2_forecasting.py` · pinned research tree |
| `src/go2_perception/forecasting/baseline.py` | HISTORICAL | baseline module · referenced by `scripts/benchmark_social_forecasting.py` · pinned research tree |
| `src/go2_perception/forecasting/data.py` | HISTORICAL | Strict metric trajectories, disjoint recording splits and gap-free common-cohort windows · referenced by `scripts/diagnose_forecasting.py` · pinned research tree |
| `src/go2_perception/forecasting/evaluation.py` | HISTORICAL | Paired held-out evaluation. No optimization, checkpoint selection or test-time sampling · referenced by `scripts/benchmark_social_forecasting.py` · pinned research tree |
| `src/go2_perception/forecasting/geometry.py` | HISTORICAL | Observation-only normalization; row-vector rotation, no fitted statistics · referenced by `src/go2_perception/forecasting/model.py` · pinned research tree |
| `src/go2_perception/forecasting/metrics.py` | HISTORICAL | Euclidean distances in metres, not coordinate-wise MSE or best-of-K metrics · referenced by `scripts/diagnose_forecasting.py` · pinned research tree |
| `src/go2_perception/forecasting/model.py` | HISTORICAL | Direct multi-horizon offsets: all attention is within observed history · referenced by `src/go2_perception/forecasting/social_model.py` · pinned research tree |
| `src/go2_perception/forecasting/social_adapter.py` | HISTORICAL | Optional offline multi-track context with explicit external identity-quality attestation · referenced by `tests/test_social_forecasting.py` · pinned research tree |
| `src/go2_perception/forecasting/social_data.py` | HISTORICAL | Observation-only synchronized neighbor context; target eligibility is unchanged · referenced by `scripts/benchmark_social_forecasting.py` · pinned research tree |
| `src/go2_perception/forecasting/social_evaluation.py` | HISTORICAL | Paired social diagnostics; corruption selection is validation-only in the experiment runner · referenced by `scripts/benchmark_social_forecasting.py` · pinned research tree |
| `src/go2_perception/forecasting/social_model.py` | HISTORICAL | Shared temporal encoder and masked mean social context, deterministic direct forecasts · referenced by `src/go2_perception/forecasting/social_training.py` · pinned research tree |
| `src/go2_perception/forecasting/social_training.py` | HISTORICAL | Social training is separate so Phase 4 checkpoint behavior remains unchanged · referenced by `scripts/benchmark_social_forecasting.py` · pinned research tree |
| `src/go2_perception/forecasting/social_visualization.py` | HISTORICAL | Public-annotation examples with observed neighbors and both successful/failed forecasts · referenced by `scripts/visualize_social_forecasting.py` · pinned research tree |
| `src/go2_perception/forecasting/training.py` | HISTORICAL | CPU-portable checkpoints; validation-only selection; test data is never loaded here · referenced by `scripts/benchmark_social_forecasting.py` · pinned research tree |
| `src/go2_perception/forecasting/visualization.py` | HISTORICAL | Headless metric XY research plots; same +x right/+y up convention as Phase 3 BEV · referenced by `scripts/visualize_forecasting.py` · pinned research tree |
| `src/go2_perception/fusion/__init__.py` | HISTORICAL | Go2 offline perception components · referenced by `scripts/generate_test_data.py` · pinned research tree |
| `src/go2_perception/fusion/association.py` | HISTORICAL | association module · referenced by `scripts/run_fusion.py` · pinned research tree |
| `src/go2_perception/fusion/calibration.py` | HISTORICAL | calibration module · referenced by `scripts/generate_test_data.py` · pinned research tree |
| `src/go2_perception/fusion/projection.py` | HISTORICAL | projection module · referenced by `scripts/generate_test_data.py` · pinned research tree |
| `src/go2_perception/fusion/visualization.py` | HISTORICAL | visualization module · referenced by `scripts/run_fusion.py` · pinned research tree |
| `src/go2_perception/io/__init__.py` | SHARED | Go2 offline perception components · referenced by `src/go2_perception/adapters/go2/pointcloud.py` |
| `src/go2_perception/io/config.py` | SHARED | config module · referenced by `scripts/go2_replay.py` |
| `src/go2_perception/io/image_loader.py` | SHARED | image loader module · referenced by `src/go2_perception/io/sensor_interface.py` |
| `src/go2_perception/io/pointcloud_loader.py` | SHARED | pointcloud loader module · referenced by `src/go2_perception/io/sensor_interface.py` |
| `src/go2_perception/io/sensor_interface.py` | SHARED | Transport-independent messages; no ROS dependency · referenced by `src/go2_perception/adapters/go2/pointcloud.py` |
| `src/go2_perception/io/serialization.py` | SHARED | serialization module · referenced by `scripts/go2_replay.py` |
| `src/go2_perception/io/synthetic.py` | SHARED | Deterministic fixtures, never an RGB accuracy benchmark · referenced by `tests/test_lidar.py` |
| `src/go2_perception/lidar/__init__.py` | SHARED | Go2 offline perception components · referenced by `scripts/go2_replay.py` |
| `src/go2_perception/lidar/clustering.py` | SHARED | clustering module · referenced by `src/go2_perception/lidar/processor.py` |
| `src/go2_perception/lidar/ground_segmentation.py` | SHARED | Seeded, gravity-constrained RANSAC; a wall is never selected as ground · referenced by `src/go2_perception/lidar/processor.py` |
| `src/go2_perception/lidar/preprocessing.py` | SHARED | preprocessing module · referenced by `src/go2_perception/lidar/processor.py` |
| `src/go2_perception/lidar/processor.py` | SHARED | processor module · referenced by `scripts/go2_replay.py` |
| `src/go2_perception/lidar/visualization.py` | SHARED | Headless XY views plus colored PLY usable in Open3D's interactive viewer · referenced by `scripts/go2_replay.py` |
| `src/go2_perception/localization/__init__.py` | HISTORICAL | Phase 3 offline temporal perception · referenced by `src/go2_perception/tracking/multi_object_tracker.py` · pinned research tree |
| `src/go2_perception/localization/ego_motion.py` | HISTORICAL | Explicit sensor-to-tracking-frame transforms; no implicit stationary-robot assumption · referenced by `src/go2_perception/tracking/multi_object_tracker.py` · pinned research tree |
| `src/go2_perception/rgb/__init__.py` | SHARED | Go2 offline perception components · referenced by `scripts/go2_replay.py` |
| `src/go2_perception/rgb/detector.py` | SHARED | detector module · referenced by `scripts/go2_replay.py` |
| `src/go2_perception/rgb/visualization.py` | SHARED | visualization module · referenced by `scripts/go2_replay.py` |
| `src/go2_perception/scene/__init__.py` | HISTORICAL | Offline stable scene perception; independent of hardware and robot control · referenced by `scripts/benchmark_lidar_windows.py` · pinned research tree |
| `src/go2_perception/scene/accumulation.py` | HISTORICAL | Incremental scans in odom, with original per-point acquisition provenance · referenced by `scripts/benchmark_lidar_windows.py` · pinned research tree |
| `src/go2_perception/scene/background.py` | HISTORICAL | Local persistent-return voxels; absence is unknown, never certified free space · referenced by `src/go2_perception/scene/pipeline.py` · pinned research tree |
| `src/go2_perception/scene/evaluation.py` | HISTORICAL | Motion-centric geometry metrics. Truth is used only here, after perception · referenced by `scripts/benchmark_stable_scene.py` · pinned research tree |
| `src/go2_perception/scene/objects.py` | HISTORICAL | Geometry-supported object estimates; continuity IDs are metadata, never suppression masks · referenced by `src/go2_perception/scene/pipeline.py` · pinned research tree |
| `src/go2_perception/scene/pipeline.py` | HISTORICAL | Opt-in stable scene pipeline. Every new scan is processed before rendering decisions · referenced by `scripts/benchmark_stable_scene.py` · pinned research tree |
| `src/go2_perception/scene/proposals.py` | HISTORICAL | Keep raw measurements; separate extended vertical planes only for compact-object proposals · referenced by `src/go2_perception/diagnostics/scene_tap.py` · pinned research tree |
| `src/go2_perception/scene/serialization.py` | HISTORICAL | Versioned, finite JSONL scene snapshots usable without the perception pipeline · referenced by `scripts/generate_scene_fixture.py` · pinned research tree |
| `src/go2_perception/scene/synthetic.py` | HISTORICAL | Deterministic physical-scene fixtures; truth never enters runtime perception · referenced by `scripts/benchmark_stable_scene.py` · pinned research tree |
| `src/go2_perception/tracking/__init__.py` | HISTORICAL | Phase 3 offline temporal perception · referenced by `scripts/annotate_ground_truth.py` · pinned research tree |
| `src/go2_perception/tracking/annotation_edits.py` | HISTORICAL | Manual corrections to ground-truth annotations, kept as reviewable overrides · referenced by `scripts/annotate_ground_truth.py` · pinned research tree |
| `src/go2_perception/tracking/association.py` | HISTORICAL | association module · referenced by `src/go2_perception/tracking/fragments.py` · pinned research tree |
| `src/go2_perception/tracking/config.py` | HISTORICAL | config module · referenced by `scripts/benchmark_tracking_robustness.py` · pinned research tree |
| `src/go2_perception/tracking/evaluation.py` | HISTORICAL | Geometric GT matching, explicit denominators, and global identity F1 (IDF1) · referenced by `scripts/benchmark_tracking_robustness.py` · pinned research tree |
| `src/go2_perception/tracking/fragments.py` | HISTORICAL | Recover a split of a previously observed compact object, never proximity-only leg merging · referenced by `src/go2_perception/tracking/multi_object_tracker.py` · pinned research tree |
| `src/go2_perception/tracking/ground_truth.py` | HISTORICAL | Ground-truth annotations for real tracking recordings, and evaluation against them · referenced by `scripts/annotate_ground_truth.py` · pinned research tree |
| `src/go2_perception/tracking/kalman_filter.py` | HISTORICAL | Planar constant velocity with continuous white acceleration process noise · referenced by `src/go2_perception/tracking/multi_object_tracker.py` · pinned research tree |
| `src/go2_perception/tracking/motion_classifier.py` | HISTORICAL | motion classifier module · referenced by `src/go2_perception/tracking/multi_object_tracker.py` · pinned research tree |
| `src/go2_perception/tracking/multi_object_tracker.py` | HISTORICAL | multi object tracker module · referenced by `scripts/benchmark_stable_scene.py` · pinned research tree |
| `src/go2_perception/tracking/normalization.py` | HISTORICAL | Convert existing LiDAR/fused results without changing their source modules · referenced by `src/go2_perception/tracking/replay.py` · pinned research tree |
| `src/go2_perception/tracking/pose_gate.py` | HISTORICAL | Hold LiDAR frames until odometry can pose them; skip, never invent, otherwise · referenced by `scripts/benchmark_tracking_robustness.py` · pinned research tree |
| `src/go2_perception/tracking/replay.py` | HISTORICAL | Offline schemas and existing Go2 recording adapter. Never invent real clocks or poses · referenced by `scripts/annotate_ground_truth.py` · pinned research tree |
| `src/go2_perception/tracking/robustness_scenarios.py` | HISTORICAL | Deterministic measured-failure-inspired fixtures, not replicas of private sensor data · referenced by `scripts/benchmark_tracking_robustness.py` · pinned research tree |
| `src/go2_perception/tracking/synthetic.py` | HISTORICAL | Eight deterministic noisy scenarios. Ground truth is odometry-frame, explicitly synthetic · referenced by `scripts/generate_tracking_test_data.py` · pinned research tree |
| `src/go2_perception/tracking/track_duplicates.py` | HISTORICAL | Conservative repeated-evidence suppression; never merge filter covariances · referenced by `src/go2_perception/tracking/multi_object_tracker.py` · pinned research tree |
| `src/go2_perception/tracking/trajectory.py` | HISTORICAL | trajectory module · referenced by `src/go2_perception/forecasting/baseline.py` · pinned research tree |
| `src/go2_perception/viewer/__init__.py` | SHARED | Phase 5B FSD-style scene viewer. Consumes go2-scene-v1 only; never imports perception code · referenced by `scripts/run_vln_continuous.py` |
| `src/go2_perception/viewer/camera.py` | HISTORICAL | Pinhole camera for the software renderer, and a smoothed chase camera · referenced by `src/go2_perception/viewer/render.py` · pinned research tree |
| `src/go2_perception/viewer/layers.py` | HISTORICAL | Named, toggleable render layers and the three presentation modes · referenced by `scripts/fsd_viewer.py` · pinned research tree |
| `src/go2_perception/viewer/planning.py` | HISTORICAL | Planner-facing reading of go2-scene-v1 (no control, no planner): what a future collision-aware · referenced by `tests/test_fsd_viewer.py` · pinned research tree |
| `src/go2_perception/viewer/privacy.py` | SHARED | Private-output guard: real imagery, geometry and videos only go to git-ignored paths · referenced by `scripts/run_vln_continuous.py` |
| `src/go2_perception/viewer/render.py` | HISTORICAL | FSD-inspired 3D scene rendering with NumPy + OpenCV (headless, deterministic, macOS-safe) · referenced by `src/go2_perception/viewer/session.py` · pinned research tree |
| `src/go2_perception/viewer/scene.py` | HISTORICAL | Parse go2-scene-v1 rows into renderer types (standard library + NumPy, no perception imports) · referenced by `src/go2_perception/viewer/planning.py` · pinned research tree |
| `src/go2_perception/viewer/session.py` | HISTORICAL | Causal playback session: scene rows (+ optional Debug sources) -> composed 1920x1080 frames · referenced by `scripts/fsd_viewer.py` · pinned research tree |
| `src/go2_perception/viewer/sources.py` | HISTORICAL | Load real-recording inputs for the viewer: 5A scene rows, Debug tap, Phase 3 cache, RGB · referenced by `scripts/fsd_viewer.py` · pinned research tree |
| `src/go2_perception/viewer/state.py` | HISTORICAL | Causal display state: what each emitted object looks like, and renderer-side memory · referenced by `scripts/fsd_viewer.py` · pinned research tree |
| `src/go2_perception/visualization/__init__.py` | HISTORICAL | Phase 3 offline temporal perception · referenced by `scripts/annotation_editor.py` · pinned research tree |
| `src/go2_perception/visualization/bev_tracking.py` | HISTORICAL | Headless OpenCV BEV: x to the right, y up, all distances in metres · referenced by `scripts/run_tracking.py` · pinned research tree |
| `src/go2_perception/visualization/diagnostic_video.py` | HISTORICAL | Synchronized RGB + LiDAR BEV + status diagnostic frames for recorded tracking runs · referenced by `scripts/annotation_editor.py` · pinned research tree |
| `src/go2_perception/vln/__init__.py` | SHARED | Go2 vision-language-navigation sensor interface (go2-vln-obs-v1). Read-only; no control · referenced by `hardware/make_capability.py` |
| `src/go2_perception/vln/bounded_context.py` | ACTIVE | Bounded visual context from the offline ablation, shared by the opt-in episode · referenced by `scripts/run_vln_supervised.py` |
| `src/go2_perception/vln/calibration.py` | SHARED | Camera intrinsics and rigid extrinsics with explicit provenance for the Go2 VLN interface · referenced by `scripts/run_vln_interactive.py` |
| `src/go2_perception/vln/continuous.py` | ACTIVE | Offline rolling SEARCH/APPROACH governor. Inputs and outputs are data on one monotonic clock · referenced by `scripts/run_vln_continuous.py` |
| `src/go2_perception/vln/continuous_planner.py` | ACTIVE | One asynchronous pinned LightNav job over the moving, synthetic pinhole camera · referenced by `scripts/run_vln_continuous.py` |
| `src/go2_perception/vln/continuous_recording.py` | SHARED | Bounded offline control recording and streaming feasibility counters · referenced by `scripts/run_vln_continuous.py` |
| `src/go2_perception/vln/continuous_review.py` | ACTIVE | Read-only continuous traces in the project's loopback VLN viewer · referenced by `scripts/run_vln_interactive.py` |
| `src/go2_perception/vln/continuous_simulation.py` | SHARED | Virtual-time kinematic proxy and image-based fixture. No model weights or robot I/O · referenced by `scripts/run_vln_continuous.py` |
| `src/go2_perception/vln/episode_app.py` | ACTIVE | Browser and archive integration for the opt-in episode. One trial owns its entire history · referenced by `scripts/run_vln_supervised.py` |
| `src/go2_perception/vln/episode_grounding.py` | ACTIVE | Re-identify a human-named object in the current rectified frame, allowing viewpoint changes · referenced by `scripts/run_vln_continuous.py` |
| `src/go2_perception/vln/episode_mock.py` | ACTIVE | Explicitly synthetic decisions for exercising the same episode IPC and recorder · referenced by `scripts/run_vln_supervised.py` |
| `src/go2_perception/vln/episode_page.py` | ACTIVE | Add episode controls to the existing camera, native image, trajectory and LiDAR viewer · referenced by `src/go2_perception/vln/episode_app.py` |
| `src/go2_perception/vln/episode_policy.py` | ACTIVE | Published import path retained; SEARCH and APPROACH implementation moved to navigation.py |
| `src/go2_perception/vln/episodes.py` | SHARED | Versioned, local-only temporal episodes. Evaluation annotations are loaded separately · referenced by `scripts/run_vln_interactive.py` |
| `src/go2_perception/vln/extrinsics.py` | SHARED | Target-based camera -> base_link extrinsics from planar board poses (no visual hand-tuning) · referenced by `scripts/phase5d_solve_extrinsics.py` |
| `src/go2_perception/vln/five_move.py` | ACTIVE | One episode grant, one command owner, at most five single-use action leases · referenced by `scripts/run_vln_supervised.py` |
| `src/go2_perception/vln/geometry.py` | SHARED | Conservative 2D / 2.5D geometry from accumulated base_link points · referenced by `src/go2_perception/vln/replay.py` |
| `src/go2_perception/vln/grounding.py` | HISTORICAL | Grounding DINO adapter: offline image regions only, never executable robot goals · referenced by `scripts/phase6_evaluate_grounding.py` · pinned research tree |
| `src/go2_perception/vln/grounding_demo.py` | HISTORICAL | Local, inspectable replay artifacts. No networking or robot command interface · referenced by `scripts/phase6_evaluate_grounding.py` · pinned research tree |
| `src/go2_perception/vln/grounding_review.py` | SHARED | Reconstruct five-move evidence from closed local files. No model or sensor imports · referenced by `src/go2_perception/vln/interactive_trials.py` |
| `src/go2_perception/vln/interactive_app.py` | ACTIVE | Interactive four-panel VLN review and data collection. ADVISORY ONLY — NO ROBOT COMMANDS · referenced by `scripts/run_vln_interactive.py` |
| `src/go2_perception/vln/interactive_guided.py` | ACTIVE | One-answer guided test, driven from the browser. ADVISORY ONLY — NO ROBOT COMMANDS · referenced by `src/go2_perception/vln/interactive_app.py` |
| `src/go2_perception/vln/interactive_lidar.py` | ACTIVE | Read-only LiDAR/odometry consumer for the interactive review tool (core environment) · referenced by `scripts/run_vln_interactive.py` |
| `src/go2_perception/vln/interactive_page.py` | ACTIVE | Single static page for the interactive review tool. Plain JS + canvas, no frameworks · referenced by `src/go2_perception/vln/continuous_review.py` |
| `src/go2_perception/vln/interactive_server.py` | ACTIVE | Loopback-only display server for the interactive review tool (standard library only) · referenced by `scripts/run_vln_interactive.py` |
| `src/go2_perception/vln/interactive_trials.py` | SHARED | Private, append-only trial recording and human review for the interactive tool · referenced by `scripts/run_vln_continuous.py` |
| `src/go2_perception/vln/intrinsics.py` | SHARED | ChArUco intrinsic calibration with held-out validation (no values are ever assumed) · referenced by `hardware/camera_heading.py` |
| `src/go2_perception/vln/model_preprocessing.py` | SHARED | Calibrated image views for offline models; never creates camera-to-robot geometry · referenced by `src/go2_perception/vln/shadow.py` |
| `src/go2_perception/vln/motion_profile.py` | ACTIVE | Effective Go2 trial profile. Distances/angles are absolute in the decision frame · referenced by `hardware/make_capability.py` |
| `src/go2_perception/vln/native_history.py` | SHARED | Append-only native observation sessions for simulation; never action feedback · referenced by `tests/test_native_observation_session.py` |
| `src/go2_perception/vln/observation.py` | SHARED | Versioned, model-agnostic Go2 observation for vision-language navigation (go2-vln-obs-v1) · referenced by `src/go2_perception/vln/__init__.py` |
| `src/go2_perception/vln/prediction_archive.py` | SHARED | Read and cross-check archived predictions without a model or hardware connection · referenced by `scripts/replay_effective_motion.py` |
| `src/go2_perception/vln/pretrained.py` | SHARED | Phase 6C contracts and predeclared diagnostics; no robot transport or motion API · referenced by `scripts/lightnav_worker.py` |
| `src/go2_perception/vln/pretrained_inputs.py` | HISTORICAL | Reuse calibrated Phase 6B views, with LightNav's regular causal observation stream · referenced by `scripts/run_pretrained_vln.py` · pinned research tree |
| `src/go2_perception/vln/replay.py` | SHARED | Build Go2VLNObservation from a local go2-recording-v1 directory (offline, read-only) · referenced by `tests/test_vln_interface.py` |
| `src/go2_perception/vln/shadow.py` | SHARED | Bounded read-only advisory sessions. Existing native jobs/results stay unchanged · referenced by `scripts/run_vln_interactive.py` |
| `src/go2_perception/vln/shadow_cache.py` | ACTIVE | Bounded, run-local cache of lossless calibrated RGB, never temporal features · referenced by `src/go2_perception/vln/shadow.py` |
| `src/go2_perception/vln/shadow_display.py` | SHARED | Small local HTML/status display. No server, commands, path-on-camera registration · referenced by `src/go2_perception/vln/continuous_planner.py` |
| `src/go2_perception/vln/shadow_evidence.py` | ACTIVE | One bounded background tensor archive; failures remain explicit on disk · referenced by `scripts/vln_shadow_worker.py` |
| `src/go2_perception/vln/shadow_live.py` | SHARED | Live read-only Go2 camera -> existing Go2VLNObservation -> ObservationInbox · referenced by `scripts/run_vln_interactive.py` |
| `src/go2_perception/vln/shadow_sources.py` | SHARED | Local observation sources. The payload is the existing Go2VLNObservation · referenced by `scripts/run_vln_interactive.py` |
| `src/go2_perception/vln/shadow_worker.py` | SHARED | One local inference subprocess, or an explicitly mocked deterministic worker · referenced by `scripts/run_vln_interactive.py` |
| `src/go2_perception/vln/simulation.py` | SHARED | One synthetic MuJoCo camera proxy. No robot hardware/network dependencies · referenced by `scripts/run_vln_simulation.py` |
| `src/go2_perception/vln/supervised.py` | ACTIVE | One reviewed primitive, one refresh, one execution. No SDK/model imports here · referenced by `hardware/make_capability.py` |
| `src/go2_perception/vln/supervised_app.py` | ACTIVE | Opt-in supervised extension of the existing four-panel viewer; defaults unchanged · referenced by `scripts/run_vln_supervised.py` |
| `src/go2_perception/vln/supervised_executor.py` | ACTIVE | Isolated executor boundary. Hardware imports occur only after explicit live gates · referenced by `scripts/run_vln_supervised.py` |
| `src/go2_perception/vln/supervised_target.py` | SHARED | Conservative visual continuity of an operator-selected ROI; no semantic model · referenced by `src/go2_perception/vln/supervised_app.py` |
| `src/go2_perception/vln/synthetic.py` | SHARED | Tiny public-safe synthetic go2-recording-v1 plus synthetic calibration, for tests and demos · referenced by `tests/test_vln_interface.py` |
| `src/go2_perception/vln/temporal_inputs.py` | HISTORICAL | Causal calibrated multi-image inputs for the official SmolVLM processor · referenced by `src/go2_perception/vln/pretrained_inputs.py` · pinned research tree |
| `src/go2_perception/vln/temporal_model.py` | HISTORICAL | One pinned local SmolVLM2 model, used as a VLM navigation-proposal baseline · referenced by `scripts/run_temporal_vln.py` · pinned research tree |
| `src/go2_perception/vln/temporal_proposals.py` | HISTORICAL | Conservative parsing of native model text. No keyword-to-action navigation logic · referenced by `src/go2_perception/vln/temporal_model.py` · pinned research tree |
| `src/go2_perception/vln/temporal_review.py` | HISTORICAL | Offline panels/videos and reference-aware metrics, independent of model inputs · referenced by `scripts/evaluate_pretrained_vln.py` · pinned research tree |
| `src/go2_perception/vln/tensors.py` | SHARED | Model-input conversion for VisualObservation without hidden geometry changes · referenced by `tests/test_vln_interface.py` |
| `src/go2_perception/vln/waypoint_declaration.py` | HISTORICAL | Freeze reviewed experiment settings before any model-driven decisions · referenced by `scripts/audit_lightnav_repetition.py` · pinned research tree |
| `src/go2_perception/vln/waypoint_evaluation.py` | HISTORICAL | Evaluator-only goal and termination accounting, never model context · referenced by `scripts/run_waypoint_suite.py` · pinned research tree |
| `src/go2_perception/vln/waypoint_rollout.py` | HISTORICAL | Static simulation loop with an observation-only inference callback · referenced by `scripts/run_waypoint_suite.py` · pinned research tree |
| `src/go2_perception/vln/waypoints.py` | SHARED | Simulation-only fixed-frame SE(2) path diagnostics; no hardware transports · referenced by `src/go2_perception/vln/simulation.py` |
| `tests/fixtures/supervised/audited_safety.py` | SHARED | File-backed test input for the supervised fixture family · referenced by `tests/test_supervised_executor.py` |
| `tests/fixtures/vln_live/fake_bridge.py` | SHARED | File-backed test input for the vln_live fixture family · referenced by `tests/test_vln_live_shadow.py` |
| `tests/test_bounded_native_integration.py` | SHARED | Optional pinned processor checks with synthetic pixels; no neural inference · retained regression |
| `tests/test_continuous_integration.py` | SHARED | Offline learned-worker protocol, observed handovers, review and hardware evidence boundary · retained regression |
| `tests/test_continuous_navigation.py` | SHARED | Continuous navigation uses the v2 proposal and simulated expiring velocity packets only · referenced by `tests/test_continuous_integration.py` · retained regression |
| `tests/test_diagnostic_video.py` | HISTORICAL | Diagnostic video rendering, event heuristics, annotation edits and privacy on synthetic data · pinned research tree |
| `tests/test_effective_motion.py` | SHARED | Offline profile, geometry, feedback and one-shot replan regressions. No SDK · referenced by `tests/test_five_move_navigation.py` · retained regression |
| `tests/test_ego_motion.py` | HISTORICAL | Checks static world object under ego motion and related cases · pinned research tree |
| `tests/test_five_move_navigation.py` | SHARED | Offline episode behavior, using synthetic poses, images, clocks and controller routes · retained regression |
| `tests/test_forecasting.py` | HISTORICAL | Checks window extraction and scene separation and related cases · pinned research tree |
| `tests/test_fsd_viewer.py` | HISTORICAL | Phase 5B viewer: schema parsing, transforms, timing, display state, colours, layers, export, · pinned research tree |
| `tests/test_fusion.py` | HISTORICAL | Checks projection invalid depth boundaries and related cases · pinned research tree |
| `tests/test_geometry.py` | SHARED | Checks aabb and related cases · retained regression |
| `tests/test_go2_adapter.py` | SHARED | Offline Go2 adapter tests. None need the robot, the Unitree SDK, or DDS · retained regression |
| `tests/test_go2_hardware.py` | SHARED | Live checks. Skipped unless GO2_HARDWARE=1, so offline runs don't fail without a robot · retained regression |
| `tests/test_go2_replay.py` | SHARED | Exercise the merged recording-to-perception CLI without SDK, DDS, or hardware · retained regression |
| `tests/test_ground_truth.py` | HISTORICAL | Ground-truth schema and evaluation on small synthetic sequences (no recordings needed) · pinned research tree |
| `tests/test_hardware_guards.py` | SHARED | Mock tests for the read-only supervised guard provider. No SDK, DDS or robot · referenced by `tests/test_effective_motion.py` · retained regression |
| `tests/test_io.py` | SHARED | Checks roundtrip and related cases · retained regression |
| `tests/test_kalman_filter.py` | HISTORICAL | Checks prediction uses full irregular interval and covariance and related cases · pinned research tree |
| `tests/test_lidar.py` | SHARED | Checks two obstacles and bounds and related cases · retained regression |
| `tests/test_lightnav_runtime_integration.py` | SHARED | Opt-in tests of the actual pinned upstream processor's CPU/MPS pooling path · retained regression |
| `tests/test_motion_classifier.py` | HISTORICAL | Checks evidence uncertainty and hysteresis and related cases · pinned research tree |
| `tests/test_motion_diagnostics.py` | SHARED | Offline regressions for the November 19 motion records. All inputs and routes are local fakes · retained regression |
| `tests/test_multi_object_tracker.py` | HISTORICAL | Checks lifecycle and reacquisition and deletion and related cases · pinned research tree |
| `tests/test_native_observation_session.py` | SHARED | Checks native buffer appends once then resets at boundaries and related cases · retained regression |
| `tests/test_pose_gate.py` | HISTORICAL | Pose gating of LiDAR frames against odometry. All clouds and poses here are synthetic · pinned research tree |
| `tests/test_pretrained_orchestration.py` | HISTORICAL | Regression checks for environment identity and immutable benchmark inputs · pinned research tree |
| `tests/test_pretrained_vln.py` | SHARED | No weights/downloads/hardware needed for Phase 6C interface regressions · retained regression |
| `tests/test_pretrained_worker.py` | SHARED | Checks model job whitelist excludes label and action leakage and related cases · retained regression |
| `tests/test_rgb.py` | SHARED | Wrapper tests mock model predictions; real model smoke is recorded separately · retained regression |
| `tests/test_scene_evaluation.py` | HISTORICAL | Evaluation regressions: stale geometry and annotation gaps must stay explicit · pinned research tree |
| `tests/test_scene_interface.py` | HISTORICAL | Checks fixture is deterministic and standalone and related cases · pinned research tree |
| `tests/test_social_forecasting.py` | HISTORICAL | Checks simultaneous neighbors partial and radius and related cases · pinned research tree |
| `tests/test_stable_scene.py` | HISTORICAL | Checks frame motion rotation translation and provenance and related cases · pinned research tree |
| `tests/test_supervised_contract.py` | SHARED | The public example is synthetic evidence, never a hardware capability · retained regression |
| `tests/test_supervised_executor.py` | SHARED | Audited-loop wrapper tests with injected routes/clock; NEVER initialize DDS · referenced by `tests/test_effective_motion.py` · retained regression |
| `tests/test_supervised_integration.py` | SHARED | CPU-only UI/protocol regression checks; localhost only, no model or robot · retained regression |
| `tests/test_supervised_navigation.py` | SHARED | Lightweight offline contracts/state-machine tests. No SDK, DDS, GPU or model · referenced by `tests/test_effective_motion.py` · retained regression |
| `tests/test_target_grounding_review.py` | SHARED | Checks typed and normalized description and related cases · retained regression |
| `tests/test_temporal_vln.py` | HISTORICAL | Offline Phase 6B contracts. Fixture responses are not model accuracy evidence · pinned research tree |
| `tests/test_temporal_vln_evaluation.py` | HISTORICAL | Synthetic saved-output evaluation; no private labels or pretrained inference · pinned research tree |
| `tests/test_temporal_vln_model_integration.py` | HISTORICAL | Opt-in actual pretrained processor/model tests. Never download during pytest · pinned research tree |
| `tests/test_temporal_vln_runner.py` | HISTORICAL | Exact-package CLI selection; stubs stop before inference, never model evidence · pinned research tree |
| `tests/test_tracking_association.py` | HISTORICAL | Checks hungarian gating unknown and known classes and related cases · pinned research tree |
| `tests/test_tracking_comparison_cli.py` | HISTORICAL | Protect historical private replay outputs when invoking the comparison harness · pinned research tree |
| `tests/test_tracking_pipeline.py` | HISTORICAL | Checks end to end synthetic and related cases · pinned research tree |
| `tests/test_tracking_refinement.py` | HISTORICAL | Checks duplicate requires repeated overlap and preserves stronger id and related cases · pinned research tree |
| `tests/test_tracking_robustness.py` | HISTORICAL | Regressions derived from A1/A2/B/C failure mechanisms; no private sensor inputs · pinned research tree |
| `tests/test_trajectory.py` | HISTORICAL | Checks constant velocity prediction and uncertainty and related cases · pinned research tree |
| `tests/test_vln_episodes.py` | SHARED | Offline contract tests; no model, private data, network or robot needed · retained regression |
| `tests/test_vln_extrinsics.py` | SHARED | Plane-based camera -> base_link extrinsics recover a known transform and refuse degenerate data · retained regression |
| `tests/test_vln_grounding.py` | HISTORICAL | Phase 6A contracts. Synthetic unit tests do not measure model semantic accuracy · pinned research tree |
| `tests/test_vln_interactive.py` | SHARED | Interactive four-panel review tool: synthetic fixtures, injected clocks, mock worker · referenced by `tests/test_motion_diagnostics.py` · retained regression |
| `tests/test_vln_interface.py` | SHARED | Phase 5D: Go2 VLN observation interface, calibration types, replay and timing provenance · retained regression |
| `tests/test_vln_intrinsics.py` | SHARED | ChArUco intrinsic calibration recovers a known synthetic fisheye camera from rendered views · retained regression |
| `tests/test_vln_latency.py` | SHARED | Lossless cache, causal reconstruction and bounded archival; no model or sensor · retained regression |
| `tests/test_vln_live_shadow.py` | SHARED | Live read-only camera bridge -> Go2VLNObservation -> ObservationInbox, mocked · retained regression |
| `tests/test_vln_shadow.py` | SHARED | Lightweight deterministic advisory tests: no torch, model, simulator or robot · retained regression |
| `tests/test_vln_simulation.py` | HISTORICAL | Checks proxy se2 uses metres radians and left positive and related cases · pinned research tree |
| `tests/test_waypoint_declaration.py` | HISTORICAL | Checks draft does not authorize model runs and related cases · pinned research tree |
| `tests/test_waypoint_evaluation.py` | HISTORICAL | Checks entry is not native stop or success and budget is external and related cases · pinned research tree |
| `tests/test_waypoint_execution.py` | HISTORICAL | Checks known endpoint and motion limits and related cases · pinned research tree |
| `tests/test_waypoint_native_integration.py` | HISTORICAL | Pinned decoder and processor, without neural inference or robot clients · pinned research tree |
| `tests/test_waypoint_rollout.py` | HISTORICAL | Loop logic is tested with an explicit fake predictor, never scored as LightNav · pinned research tree |
| `tests/test_waypoint_simulation.py` | HISTORICAL | Optional numeric MuJoCo checks and actual rendered landmark validation · pinned research tree |

## Configuration, schemas and fixture files

| Original path | Class | Disposition and evidence |
|---|---|---|
| `configs/default.yaml` | HISTORICAL | Versioned settings for default · pinned research tree |
| `configs/forecasting/eth_ucy_manifest.yaml` | HISTORICAL | Versioned settings for eth_ucy_manifest · pinned research tree |
| `configs/forecasting_baseline.yaml` | HISTORICAL | Versioned settings for forecasting_baseline · pinned research tree |
| `configs/forecasting_smoke.yaml` | HISTORICAL | Versioned settings for forecasting_smoke · pinned research tree |
| `configs/forecasting_social.yaml` | HISTORICAL | Versioned settings for forecasting_social · pinned research tree |
| `configs/forecasting_transformer.yaml` | HISTORICAL | Versioned settings for forecasting_transformer · pinned research tree |
| `configs/fusion.yaml` | HISTORICAL | Versioned settings for fusion · pinned research tree |
| `configs/go2_front_camera_calibration.yaml` | SHARED | Versioned settings for go2_front_camera_calibration |
| `configs/go2_lidar.yaml` | SHARED | Versioned settings for go2_lidar |
| `configs/go2_rgb.yaml` | SHARED | Versioned settings for go2_rgb |
| `configs/go2_sensors.yaml` | SHARED | Versioned settings for go2_sensors |
| `configs/go2_vln_sensors.yaml` | SHARED | Versioned settings for go2_vln_sensors |
| `configs/lidar.yaml` | SHARED | Versioned settings for lidar |
| `configs/phase5_heldout_protocol.yaml` | HISTORICAL | Versioned settings for phase5_heldout_protocol · pinned research tree |
| `configs/pretrained_vln.yaml` | SHARED | Versioned settings for pretrained_vln |
| `configs/rgb.yaml` | SHARED | Versioned settings for rgb |
| `configs/simulation_vln_room.xml` | SHARED | Versioned settings for simulation_vln_room |
| `configs/stable_lidar.yaml` | HISTORICAL | Versioned settings for stable_lidar · pinned research tree |
| `configs/synthetic_calibration.yaml` | SHARED | Versioned settings for synthetic_calibration |
| `configs/temporal_vln.yaml` | HISTORICAL | Versioned settings for temporal_vln · pinned research tree |
| `configs/temporal_vln_pilot.yaml` | HISTORICAL | Versioned settings for temporal_vln_pilot · pinned research tree |
| `configs/temporal_vln_prompt.txt` | HISTORICAL | Versioned settings for temporal_vln_prompt · pinned research tree |
| `configs/tracking.yaml` | HISTORICAL | Versioned settings for tracking · pinned research tree |
| `configs/tracking_refinement.yaml` | HISTORICAL | Versioned settings for tracking_refinement · pinned research tree |
| `configs/tracking_robustness.yaml` | HISTORICAL | Versioned settings for tracking_robustness · pinned research tree |
| `configs/vln_grounding.yaml` | HISTORICAL | Versioned settings for vln_grounding · pinned research tree |
| `configs/vln_grounding_eval.yaml` | HISTORICAL | Versioned settings for vln_grounding_eval · pinned research tree |
| `configs/waypoint_validation.yaml` | HISTORICAL | Versioned settings for waypoint_validation · pinned research tree |
| `examples/continuous_navigation/pending_evidence.json` | SHARED | Example data for continuous_navigation |
| `examples/five_move_navigation/synthetic_episode.json` | SHARED | Example data for five_move_navigation |
| `examples/supervised_navigation/synthetic_cycle.json` | SHARED | Example data for supervised_navigation |
| `schemas/go2-scene-v1.schema.json` | HISTORICAL | Versioned scene data contract · pinned research tree |
| `src/go2_perception/adapters/ros2/README.md` | HISTORICAL | README module · pinned research tree |
| `src/go2_perception/vln/schemas/continuous-evidence-v1.schema.json` | SHARED | Versioned JSON contract loaded by schema name |
| `src/go2_perception/vln/schemas/five-move-episode-v1.schema.json` | SHARED | Versioned JSON contract loaded by schema name |
| `src/go2_perception/vln/schemas/supervised-navigation-v1.schema.json` | SHARED | Versioned JSON contract loaded by schema name |
| `src/go2_perception/vln/schemas/supervised-navigation-v2.schema.json` | SHARED | Versioned JSON contract loaded by schema name |
| `src/go2_perception/vln/schemas/vln_annotations_v1.schema.json` | SHARED | Versioned JSON contract loaded by schema name |
| `src/go2_perception/vln/schemas/vln_episode_v1.schema.json` | SHARED | Versioned JSON contract loaded by schema name |
| `tests/fixtures/forecasting/manifest.yaml` | HISTORICAL | File-backed test input for the forecasting fixture family · pinned research tree |
| `tests/fixtures/forecasting/test_scene.txt` | HISTORICAL | File-backed test input for the forecasting fixture family · pinned research tree |
| `tests/fixtures/forecasting/train_scene.txt` | HISTORICAL | File-backed test input for the forecasting fixture family · pinned research tree |
| `tests/fixtures/forecasting/validation_scene.txt` | HISTORICAL | File-backed test input for the forecasting fixture family · pinned research tree |
| `tests/fixtures/go2/real_cloud_pair.npz` | SHARED | File-backed test input for the go2 fixture family |
| `tests/fixtures/scene/renderer.jsonl` | HISTORICAL | File-backed test input for the scene fixture family · pinned research tree |
| `tests/fixtures/supervised/README.md` | SHARED | File-backed test input for the supervised fixture family |
| `tests/fixtures/supervised/audited_safety.py` | SHARED | File-backed test input for the supervised fixture family · referenced by `tests/test_supervised_executor.py` |
| `tests/fixtures/vln/episode_v1/annotations.json` | SHARED | File-backed test input for the vln fixture family |
| `tests/fixtures/vln/episode_v1/calibration.yaml` | SHARED | File-backed test input for the vln fixture family |
| `tests/fixtures/vln/episode_v1/episode.json` | SHARED | File-backed test input for the vln fixture family |
| `tests/fixtures/vln/episode_v1/images/000.png` | SHARED | File-backed test input for the vln fixture family |
| `tests/fixtures/vln/episode_v1/images/001.png` | SHARED | File-backed test input for the vln fixture family |
| `tests/fixtures/vln/episode_v1/images/003.png` | SHARED | File-backed test input for the vln fixture family |
| `tests/fixtures/vln/episode_v1/images/004.png` | SHARED | File-backed test input for the vln fixture family |
| `tests/fixtures/vln/episode_v1/images/005.png` | SHARED | File-backed test input for the vln fixture family |
| `tests/fixtures/vln/synthetic_recording/lidar/utlidar_cloud_base/000000.npy` | SHARED | File-backed test input for the vln fixture family |
| `tests/fixtures/vln/synthetic_recording/lidar/utlidar_cloud_base/000000_time.npy` | SHARED | File-backed test input for the vln fixture family |
| `tests/fixtures/vln/synthetic_recording/lidar/utlidar_cloud_base/000001.npy` | SHARED | File-backed test input for the vln fixture family |
| `tests/fixtures/vln/synthetic_recording/lidar/utlidar_cloud_base/000001_time.npy` | SHARED | File-backed test input for the vln fixture family |
| `tests/fixtures/vln/synthetic_recording/lidar/utlidar_cloud_base/000002.npy` | SHARED | File-backed test input for the vln fixture family |
| `tests/fixtures/vln/synthetic_recording/lidar/utlidar_cloud_base/000002_time.npy` | SHARED | File-backed test input for the vln fixture family |
| `tests/fixtures/vln/synthetic_recording/lidar/utlidar_cloud_base/000003.npy` | SHARED | File-backed test input for the vln fixture family |
| `tests/fixtures/vln/synthetic_recording/lidar/utlidar_cloud_base/000003_time.npy` | SHARED | File-backed test input for the vln fixture family |
| `tests/fixtures/vln/synthetic_recording/lidar/utlidar_cloud_base/000004.npy` | SHARED | File-backed test input for the vln fixture family |
| `tests/fixtures/vln/synthetic_recording/lidar/utlidar_cloud_base/000004_time.npy` | SHARED | File-backed test input for the vln fixture family |
| `tests/fixtures/vln/synthetic_recording/lidar/utlidar_cloud_base/000005.npy` | SHARED | File-backed test input for the vln fixture family |
| `tests/fixtures/vln/synthetic_recording/lidar/utlidar_cloud_base/000005_time.npy` | SHARED | File-backed test input for the vln fixture family |
| `tests/fixtures/vln/synthetic_recording/lidar/utlidar_cloud_base/000006.npy` | SHARED | File-backed test input for the vln fixture family |
| `tests/fixtures/vln/synthetic_recording/lidar/utlidar_cloud_base/000006_time.npy` | SHARED | File-backed test input for the vln fixture family |
| `tests/fixtures/vln/synthetic_recording/lidar/utlidar_cloud_base/000007.npy` | SHARED | File-backed test input for the vln fixture family |
| `tests/fixtures/vln/synthetic_recording/lidar/utlidar_cloud_base/000007_time.npy` | SHARED | File-backed test input for the vln fixture family |
| `tests/fixtures/vln/synthetic_recording/lidar/utlidar_cloud_base/000008.npy` | SHARED | File-backed test input for the vln fixture family |
| `tests/fixtures/vln/synthetic_recording/lidar/utlidar_cloud_base/000008_time.npy` | SHARED | File-backed test input for the vln fixture family |
| `tests/fixtures/vln/synthetic_recording/lidar/utlidar_cloud_base/000009.npy` | SHARED | File-backed test input for the vln fixture family |
| `tests/fixtures/vln/synthetic_recording/lidar/utlidar_cloud_base/000009_time.npy` | SHARED | File-backed test input for the vln fixture family |
| `tests/fixtures/vln/synthetic_recording/lidar/utlidar_cloud_base/000010.npy` | SHARED | File-backed test input for the vln fixture family |
| `tests/fixtures/vln/synthetic_recording/lidar/utlidar_cloud_base/000010_time.npy` | SHARED | File-backed test input for the vln fixture family |
| `tests/fixtures/vln/synthetic_recording/lidar/utlidar_cloud_base/000011.npy` | SHARED | File-backed test input for the vln fixture family |
| `tests/fixtures/vln/synthetic_recording/lidar/utlidar_cloud_base/000011_time.npy` | SHARED | File-backed test input for the vln fixture family |
| `tests/fixtures/vln/synthetic_recording/lidar/utlidar_cloud_base/index.jsonl` | SHARED | File-backed test input for the vln fixture family |
| `tests/fixtures/vln/synthetic_recording/manifest.json` | SHARED | File-backed test input for the vln fixture family |
| `tests/fixtures/vln/synthetic_recording/rgb/000000.png` | SHARED | File-backed test input for the vln fixture family |
| `tests/fixtures/vln/synthetic_recording/rgb/000001.png` | SHARED | File-backed test input for the vln fixture family |
| `tests/fixtures/vln/synthetic_recording/rgb/000002.png` | SHARED | File-backed test input for the vln fixture family |
| `tests/fixtures/vln/synthetic_recording/rgb/000003.png` | SHARED | File-backed test input for the vln fixture family |
| `tests/fixtures/vln/synthetic_recording/rgb/000004.png` | SHARED | File-backed test input for the vln fixture family |
| `tests/fixtures/vln/synthetic_recording/rgb/000005.png` | SHARED | File-backed test input for the vln fixture family |
| `tests/fixtures/vln/synthetic_recording/rgb/000006.png` | SHARED | File-backed test input for the vln fixture family |
| `tests/fixtures/vln/synthetic_recording/rgb/000007.png` | SHARED | File-backed test input for the vln fixture family |
| `tests/fixtures/vln/synthetic_recording/rgb/000008.png` | SHARED | File-backed test input for the vln fixture family |
| `tests/fixtures/vln/synthetic_recording/rgb/000009.png` | SHARED | File-backed test input for the vln fixture family |
| `tests/fixtures/vln/synthetic_recording/rgb/000010.png` | SHARED | File-backed test input for the vln fixture family |
| `tests/fixtures/vln/synthetic_recording/rgb/index.jsonl` | SHARED | File-backed test input for the vln fixture family |
| `tests/fixtures/vln/synthetic_recording/state/utlidar_robot_odom.jsonl` | SHARED | File-backed test input for the vln fixture family |
| `tests/fixtures/vln/synthetic_recording/synthetic_calibration.yaml` | SHARED | File-backed test input for the vln fixture family |
| `tests/fixtures/vln_live/fake_bridge.py` | SHARED | File-backed test input for the vln_live fixture family · referenced by `tests/test_vln_live_shadow.py` |
