# Phase 6C Windows/NVIDIA and small-scene feasibility

## Windows result: pending, not a successful hardware test

This session can access the Apple M3 Pro Mac, not the Windows laptop. The user's
RTX 2070 Super / approximately 8 GB description is a planning input, not a measured
inventory. Existing Isaac Sim / Isaac Lab version, Windows vs WSL configuration,
driver, CUDA toolkit, PyTorch CUDA runtime and free VRAM remain **unverified**.
No driver, WSL setting, existing simulator or shared environment was changed.

Read-only preflight from the Windows checkout of `phase6-pretrained-vln`:

```powershell
# Run from the repository root. Use the actual Python executable to be assessed.
powershell -NoProfile -File .\scripts\windows_vln_preflight.ps1 `
  -Python python -Output .\data\outputs\phase6c-windows-preflight

# For an EXISTING environment/installation at nonstandard locations, supply its paths.
# Replace these example paths with real locations; this does not install anything.
.\scripts\windows_vln_preflight.ps1 `
  -Python 'C:\existing-environment\python.exe' `
  -IsaacRoot @('C:\existing-isaacsim', 'C:\existing-IsaacLab') `
  -Output .\data\outputs\phase6c-windows-existing-env
```

Outputs: `windows.json` (OS/architecture, RAM, disks, displays, existing WSL listing,
bounded installation search) and `python.json` (actual interpreter/build, CUDA/MPS,
GPU total/free memory, compute capability, package/version files). Each output folder
must be new. It does not recursively scan the whole computer, launch a simulator,
load weights, enable WSL, change execution policy or install dependencies.
PowerShell execution itself has **not** been tested on Windows in this session.
Run separately for each intended environment; a native Windows report cannot certify
a different Python inside WSL. The Python helper can also run inside an existing
Linux/WSL environment with `python scripts/pretrained_vln_preflight.py --output report.json`.

Interpret three different fields separately:

- `nvidia-smi` banner CUDA version: maximum API compatibility advertised by the driver.
- `nvcc --version`: an installed toolkit, if one is on PATH.
- `torch.version.cuda`: runtime used by that PyTorch wheel; CUDA availability and a
  compute capability query belong to that specific environment.

CIM `AdapterRAM` is deliberately not used to establish VRAM (it can overflow).
An absent `nvcc` does not by itself mean a CUDA PyTorch wheel cannot run.

## Isaac decision gate

The [current official requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html)
list Windows 11, RTX 4080 / 16 GB VRAM, 32 GB RAM and 50 GB SSD as minimum baseline.
The inspected [5.1 requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)
also list 16 GB VRAM but include Windows 10/11; that archived release is marked
unsupported. These facts do not establish the requirements of the user's unknown
existing version. An 8 GB GPU is below those published baselines. Headless physics
success does not demonstrate RGB rendering capability or spare memory for a model.

After the preflight identifies the actual installation, consult THAT version's
requirements and [official compatibility checker instructions](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_workstation.html).
For an existing binary installation that provides the checker:

```powershell
Set-Location 'C:\actual-isaacsim-installation'
Get-Content .\VERSION
if (!(Test-Path .\isaac-sim.compatibility_check.bat)) { throw 'Inspect this version; do not install or replace it blindly.' }
.\isaac-sim.compatibility_check.bat
```

Preserve the report, including GPU/driver/VRAM findings. Test Kit initializes a small
headless application; it is not a camera test. Only after compatibility is established,
try one room, one kinematic proxy and one 448×256 RGB camera, no RL batch/RTX LiDAR.
Record version/settings, initialization, before/after free VRAM, camera image and
short-run FPS/stability. Do not launch the model on that GPU during the first test.
No version-specific Isaac scene script is represented as tested while the installation
is inaccessible. No blind downgrade/reinstallation is recommended.

## Separate resource budgets

| Component | Evidence / budget |
|---|---|
| LightNav model alone | ~9.03 GiB BF16 weights including vision, plus up to 1.125 GiB KV at 8192 tokens and activations/processor/allocator. Plan 12–16 GiB. Actual Mac inference in the experiment report |
| Windows model | 8 GB cannot hold unquantized weights alone. Pinned upstream BF16 path is not a native Turing path; standard FlashAttention-2 is unsuitable. No verified 4-bit/offload adaptation; configuration validator refuses these claims |
| Isaac renderer alone | Unknown installed version; modern documented baseline 16 GB. Initialization, VRAM and RGB results pending |
| Combined Windows GPU | Not tested and not assumed feasible. Neither a 4-bit filename nor a headless physics run proves capacity |
| Mac lightweight renderer | MuJoCo 3.3.7, OpenGL/CGL, synthetic room below. Renderer-only maximum RSS 109.4 MiB; no separate OpenGL VRAM measurement. Not Isaac and not an RTX test |
| Combined Mac run | One 16-step static rollout executed: maximum RSS 11.059 GiB; sampled MPS driver peak 10.550 GiB (overlapping unified memory). It failed navigation; this does not establish Windows capacity |

## Lightweight Mac scene

The selected model's [bundled MuJoCo path](https://github.com/lightorigins/LightNav-0/tree/3015508b70fb6e30bbd66701e94c9b916dcb8df8/mujoco_demo)
was inspected: model server and renderer/client are separate; its ProcTHOR scene and
THOR assets have CC-BY attribution terms. We did not need its larger scene, ROS clients,
MPC server or optional biped policy. This repository instead supplies one tiny original
MuJoCo room (`configs/simulation_vln_room.xml`) with a doorway, chair, table and bottle,
one RGB camera and a kinematic camera proxy. No third-party scene assets are vendored.

Measured `sim_render_smoke_v2`: 5,413 frames in 5.00036 s (~1,082.5 render calls/s),
448×256 RGB, initialization plus first image 0.135 s, all repeated static frames
identical; RGB standard deviation 66.72. This is a **five-second static render
microbenchmark**, not navigation FPS, real-time model throughput or long-run stability.
The initial darker lighting smoke is retained separately, then lighting/framing were
adjusted before any learned rollout. The improved view was visually inspected.
No NVIDIA VRAM exists on this Mac; do not report shared RSS as VRAM.

```bash
cd /Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6c-worktree
ARTIFACTS=/Users/yushan/Desktop/Go2_Navigation/data/outputs/phase6c
PYTHONPATH=src MUJOCO_GL=cgl "$ARTIFACTS/model-env/bin/python" scripts/run_vln_simulation.py \
  --output "$ARTIFACTS/sim_render_smoke_repeat"
```

`CameraRoom` uses only MuJoCo/NumPy and has no transport to robot hardware. Dynamics
contact lists omit static/mocap pairs, so a regression test exposed a false-clear
case; the adapter now checks signed geom distances and rejects penetrating substeps.
That is evaluator constraint logic, not learned collision avoidance. Local native
waypoints are metres/radians, +x forward/+y left/+yaw CCW; only the first waypoint
is proposed for a 0.25 s simulation step. Bounds reject excessive steps rather than
silently executing a different chunk.

Known simulator camera transforms and the stopping region remain exclusively in the
simulator/evaluator. The policy receives RGB and the instruction; never target
coordinates, contact flags or semantic ground truth. Simulation pauses during
inference; stale host-clock responses are still rejected. A paused scene provides no
dynamic-safety evidence. Closed-loop execution/result, if completed, is recorded in
[PRETRAINED_VLN_EXPERIMENTS](PRETRAINED_VLN_EXPERIMENTS.md), separately from this renderer test.
