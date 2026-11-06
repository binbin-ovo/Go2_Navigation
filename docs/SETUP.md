# Setup
For Phase 2 live access, use [GO2_CONNECTION](GO2_CONNECTION.md), which records the
verified Linux/CUDA workstation and pinned SDK installation. The base package below
also runs recorded RGB/LiDAR replay without DDS or the SDK. For offline video-decoder
tests, optionally install `av==17.1.0`; live hardware tests remain opt-in.

Use Python 3.10–3.12 (3.12 recommended). The installed host Python 3.14 is outside this
project's supported range. On macOS ARM and Linux x86_64 use a fresh virtual environment:
```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
pytest -q
```
With an existing Python 3.12, `python3.12 -m venv .venv` and
`python -m pip install -r requirements.txt` are equivalent. Requirements install the editable
package plus pytest; dependencies are pinned in pyproject.toml. `requirements-macos-arm64.lock`
records the entire tested environment when available; it is platform-specific, not a Linux
CUDA lock. Download weights once and keep an explicit local model path for offline operation.

CPU is default. CUDA users must install a torch/torchvision build compatible with their
NVIDIA driver and record the resulting environment; use `--device 0`. CUDA cannot be tested on this macOS host; Phase 2 records successful validation
on its Linux workstation. Open3D wheels/platform compatibility must be checked before
changing Python. Linux may require distribution OpenGL libraries (`libGL.so.1`, libgomp);
headless OpenCV visualization does not launch an Open3D GUI.

Inputs and weights can be downloaded with the commands in EXPERIMENTS.md. Running the
pipeline on cached inputs is fully offline. `.env.example` documents an optional writable
Ultralytics settings directory; no dotenv loader is used (export variables in the shell).

Sources: [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/),
[Open3D installation](https://www.open3d.org/docs/release/getting_started.html).

If macOS reports `ModuleNotFoundError` after an editable install, inspect the `.pth`
file with `ls -lO .venv/lib/python3.12/site-packages/__editable__*`. Python skips
files carrying the macOS hidden flag. Remove that flag from the affected file with
`chflags nohidden .venv/lib/python3.12/site-packages/__editable__.go2_dynamic_perception-0.1.0.pth`
(adjust Python version if needed); this repairs local environment metadata only.

If a local tool keeps reapplying the hidden flag, use a regular wheel install instead:
`uv pip install --reinstall-package go2-dynamic-perception '.[test]'`.
Reinstall after source changes when using this non-editable mode. This was the verified
workaround during the main-branch integration check.
