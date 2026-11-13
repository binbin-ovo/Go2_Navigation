"""Read-only runtime inventory. Does not load weights, install software or contact Go2."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def command(args):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=20)
        return {"returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"unavailable": str(e)}


def inventory(isaac_roots=()):
    d = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "os": platform.platform(),
        "architecture": platform.machine(),
        "python": sys.version,
        "executable": sys.executable,
        "wsl_environment": bool(os.getenv("WSL_DISTRO_NAME")),
        "disk_free_bytes": shutil.disk_usage(Path.cwd()).free,
        "driver_advertised_cuda": command(["nvidia-smi"]),
        "gpu_inventory": command(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,driver_version,compute_cap",
                "--format=csv,noheader",
            ]
        ),
        "installed_cuda_toolkit": command(["nvcc", "--version"]),
        "packages": {},
        "isaac_installations": [],
    }
    for package in (
        "torch",
        "torchvision",
        "transformers",
        "numpy",
        "isaacsim",
        "isaaclab",
        "mujoco",
    ):
        try:
            d["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            d["packages"][package] = None
    try:
        import psutil

        m = psutil.virtual_memory()
        d["ram"] = {"total_bytes": m.total, "available_bytes": m.available}
    except ImportError:
        d["ram"] = {
            "pending": "psutil absent; Windows wrapper also collects CIM memory"
        }
    try:
        import torch

        t = {
            "build": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
            "gpus": [],
        }
        for i in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            t["gpus"].append(
                {
                    "name": prop.name,
                    "compute_capability": [prop.major, prop.minor],
                    "total_bytes": total,
                    "free_bytes": free,
                    "native_bf16_architecture": prop.major >= 8,
                }
            )
        if t["mps_available"]:
            t["mps_recommended_max_bytes"] = torch.mps.recommended_max_memory()
        d["torch"] = t
    except Exception as e:
        d["torch"] = {"error": repr(e)}
    for root in isaac_roots:
        root = Path(root)
        entry = {"path": str(root), "exists": root.exists(), "version_files": {}}
        for filename in (
            "VERSION",
            "version.txt",
            "kit/VERSION",
            "source/isaaclab/config/extension.toml",
        ):
            path = root / filename
            if path.is_file():
                entry["version_files"][filename] = path.read_text(errors="replace")[
                    :8000
                ]
        entry["compatibility_checkers"] = [str(p) for p in root.glob("*compatibility*")]
        d["isaac_installations"].append(entry)
    if platform.system() == "Darwin":
        d["apple_gpu"] = command(["system_profiler", "SPDisplaysDataType"])
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--isaac-root", action="append", default=[])
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    text = json.dumps(inventory(a.isaac_root), indent=2) + "\n"
    if a.output:
        with a.output.open("x") as f:
            f.write(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
