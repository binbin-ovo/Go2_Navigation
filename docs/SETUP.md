# Setup
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
NVIDIA driver and record the resulting environment; use `--device 0`. CUDA has not been
validated on this macOS host. Open3D wheels/platform compatibility must be checked before
changing Python. Linux may require distribution OpenGL libraries (`libGL.so.1`, libgomp);
headless OpenCV visualization does not launch an Open3D GUI.

Inputs and weights can be downloaded with the commands in EXPERIMENTS.md. Running the
pipeline on cached inputs is fully offline. `.env.example` documents an optional writable
Ultralytics settings directory; no dotenv loader is used (export variables in the shell).

Sources: [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/),
[Open3D installation](https://www.open3d.org/docs/release/getting_started.html).
