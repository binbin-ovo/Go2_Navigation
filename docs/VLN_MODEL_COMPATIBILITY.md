# VLN model compatibility with the Go2 sensor interface (Phase 5D, 2026-11-11)

Nothing below has been run on our Go2. "Reported" means stated by the cited authors. Our available
inputs come from [GO2_VLN_INPUT_INTERFACE](GO2_VLN_INPUT_INTERFACE.md): 1280×720 RGB at 14.3 Hz
(uncalibrated), LiDAR `cloud_base` (0.5 s / 1.0 s windows), a conservative 2D grid, a 2.5D elevation
grid, sector obstacle ranges, odometry. Missing: intrinsics, `T_base_camera`, RGB-aligned depth.

## Approach A — RGB-only video VLM / VLA
| Model | Inputs | Output | Go2 relevance |
|---|---|---|---|
| NaVid (RSS 2024, [arXiv 2402.15852](https://arxiv.org/abs/2402.15852)) | instruction + monocular RGB video history | language actions with amounts (forward / turn / stop) | RGB-only; trained in Habitat VLN-CE |
| Uni-NaVid ([arXiv 2412.06224](https://arxiv.org/pdf/2412.06224)) | instruction + RGB video | discrete navigation actions across several nav tasks | RGB-only |
| NaVILA ([arXiv 2412.04453](https://arxiv.org/html/2412.04453v2), [project](https://navila-bot.github.io/)) | instruction + RGB video frames | **mid-level language actions** (e.g. "turn right 30°"), executed by a separate locomotion policy | authors report deployment on Unitree Go2, H1, Booster T1; 88% on 25 real instructions |
| StreamVLN ([arXiv 2507.05240](https://arxiv.org/abs/2507.05240), [code](https://github.com/OpenRobotLab/StreamVLN)) | instruction + streaming RGB, slow-fast context | short action chunks | authors report a Go2 with an added RGB-D camera, inference on a remote RTX 4090 (~0.27 s per 4 actions, + 0.2–1.0 s network) |

- **Our inputs:** sufficient in format: `image` (letterbox via `tensors.resize`), `stack_history`,
  ~14 Hz is above what these models sample (they subsample history). Field of view matters: these
  models were trained mostly on simulator pinhole cameras (VLN-CE / Habitat, about 90° HFOV, around
  human chest height). Our
  camera is wide and strongly distorted at ~0.3 m height; the domain gap is unmeasured. Undistorting
  to a pinhole crop needs **measured intrinsics**.
- **Outputs:** symbolic, relative motions. Convert to an `odom` goal by composing with `T_odom_base`
  at decision time, then check against the geometry (below). No metric map is produced.
- **Training:** zero-shot use is possible for experiments; fine-tuning on Go2-view data would need
  our own episodes. Not required for the interface.
- **Blockers:** intrinsics (for undistortion / FOV match); GPU host (models are 7B-class); measured
  end-to-end latency. Collision safety is not provided by these models.

## Approach B — RGB + geometric map (preferred)
| Model | Inputs | Output | Notes |
|---|---|---|---|
| VLFM (ICRA 2024, [arXiv 2312.03275](https://arxiv.org/abs/2312.03275)) | RGB + depth + pose; depth → occupancy and frontiers; VLM scores RGB for a language value map | frontier / waypoint in the map | zero-shot; authors report a Spot deployment |
| VLMaps (ICRA 2023, [arXiv 2210.05714](https://arxiv.org/abs/2210.05714)) | posed RGB-D, open-vocabulary features painted onto a top-down map | map-coordinate goals from language | needs depth registered to RGB |

- **Our inputs:** the geometric half exists now: LiDAR `grid` (unknown/floor seen/occupied),
  `sector_range`, `elevation_max`, odometry pose. This can replace VLFM's depth-derived occupancy and
  frontiers (frontier = FREE cell adjacent to UNKNOWN). The semantic half (painting VLM scores into the
  map) needs the camera's viewing cone in `base_link`: **measured intrinsics (HFOV at least) and
  `T_base_camera`**. VLMaps-style per-pixel painting needs `projected_depth`, i.e. full calibration.
- **Outputs:** metric goals in `odom`; checkable against the grid before any use.
- **Why preferred:** language and semantics come from the VLM while geometry, free space and
  collision come from measured LiDAR, so a VLM hallucination cannot create free space.
- **Blockers:** intrinsics + extrinsics; a frontier/value-map module (not implemented); odometry
  accuracy under motion (unverified); LiDAR misses low (< 10 cm) and near (< 0.3 m) obstacles.

## Approach C — RGB-D / 3D-aware
| Model | Inputs | Output | Notes |
|---|---|---|---|
| ETPNav (TPAMI 2024, [arXiv 2304.03047](https://arxiv.org/abs/2304.03047)) | RGB-D panoramas + pose, online topological map | waypoint nodes | trained in VLN-CE |
| Dynam3D (NeurIPS 2025, [arXiv 2505.11383](https://arxiv.org/abs/2505.11383)) | posed RGB-D; CLIP features lifted into 3D patch/instance/zone tokens | actions from a 3D-token VLM | authors report a Stretch 3; 20 trials per setting |

- **Our inputs:** require dense depth aligned with RGB. LiDAR projected into the image gives sparse
  depth (~11 k points per 0.5 s over a wide image) only **after calibration**; densification would be a
  new, unvalidated component. Panoramic inputs (ETPNav) do not exist on the Go2 (single front camera).
- **Blockers:** calibration, dense depth, likely retraining for sparse LiDAR depth. Largest gap.

## Converting model outputs without inventing geometry
1. Symbolic action (turn θ, forward d) → candidate pose `T_odom_goal = T_odom_base · T(θ, d)`.
2. Pixel / image-space target → needs measured intrinsics + `T_base_camera`; intersect the ray with
   LiDAR `projected_depth` or with the floor plane only where the grid shows floor seen. Not possible today.
3. Map goal (Approach B) → already in `odom`.
4. Accept a goal only if its cell and the straight corridor to it are FREE (floor seen) and every
   sector it crosses has `sector_clear_to ≥` distance; UNKNOWN or OCCUPIED → reject or ask for
   exploration. Keep a margin ≥ robot half-width plus the 0.3 m blind radius. Never treat model
   confidence or a 3D box as free-space evidence.
5. Execution is out of scope: this repository sends no commands.

## Recommendation
Start with **B**: LiDAR geometry from this interface + a VLM scoring camera views, after the
intrinsic and extrinsic calibration. **A** (NaVILA-style mid-level actions) is the quickest experiment
and runs on the current uncalibrated images, gated by the geometric check above. **C** waits for
calibration plus a depth-completion study.
