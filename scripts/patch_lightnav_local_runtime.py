"""Create a separate, hash-audited CPU/MPS portability copy of pinned LightNav.

Explicit device selection, two HF timing barriers, and a CPU fallback for MPS's
unsupported non-divisible adaptive-average pooling. Pooling mathematics, processor
geometry, prompt, BF16 dtype, SDPA, sampling, history and RVQ decoder stay intact.
"""

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil

REVISION = "3015508b70fb6e30bbd66701e94c9b916dcb8df8"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root", type=Path, required=True, help="prepare_pretrained_vln output"
    )
    a = ap.parse_args()
    identity = json.loads((a.root / "source_identity.json").read_text())
    if identity["revision"] != REVISION:
        raise ValueError("Unsupported upstream revision")
    for f in identity["files"]:
        if (
            hashlib.sha256((a.root / "source" / f["path"]).read_bytes()).hexdigest()
            != f["sha256"]
        ):
            raise ValueError(f"Pristine source changed: {f['path']}")
    dest = a.root / "source-local-v2"
    shutil.copytree(
        a.root / "source", dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    changes = {}
    path = dest / "src/lightnav/inference/model.py"
    before = path.read_text()
    old = 'device = torch.device(config.device if torch.cuda.is_available() else "cpu")'
    if before.count(old) != 1:
        raise ValueError("Unexpected HF device selection site")
    after = before.replace(old, "device = torch.device(config.device)")
    path.write_text(after)
    changes[str(path.relative_to(dest))] = (before, after)
    path = dest / "src/lightnav/processing.py"
    before = path.read_text()
    if before.count("F.adaptive_avg_pool2d(") != 2:
        raise ValueError("Unexpected post-ViT pooling sites")
    helper = """def _portable_adaptive_avg_pool2d(x, output_size):
    # PyTorch MPS does not implement non-divisible adaptive pooling. Preserve
    # the same BF16 adaptive-average operation on CPU, then restore the device.
    h, w = output_size
    if x.device.type == "mps" and (x.shape[-2] % h or x.shape[-1] % w):
        return F.adaptive_avg_pool2d(x.to("cpu"), output_size).to(x.device)
    return F.adaptive_avg_pool2d(x, output_size)


"""
    after = before.replace("F.adaptive_avg_pool2d(", "_portable_adaptive_avg_pool2d(")
    after = after.replace(
        "def post_vit_spatial_pool(", helper + "def post_vit_spatial_pool("
    )
    path.write_text(after)
    changes[str(path.relative_to(dest))] = (before, after)
    path = dest / "src/lightnav/inference/engine.py"
    before = path.read_text()
    begin, end = (
        before.index("    def _generate_hf("),
        before.index("    def _vit_forward_from_inputs("),
    )
    segment = before[begin:end]
    if segment.count("        torch.cuda.synchronize()") != 2:
        raise ValueError("Unexpected HF synchronization sites")
    segment = segment.replace(
        "        torch.cuda.synchronize()",
        "        if self.bundle.device.type == 'cuda':\n            torch.cuda.synchronize()\n        elif self.bundle.device.type == 'mps':\n            torch.mps.synchronize()",
    )
    after = before[:begin] + segment + before[end:]
    path.write_text(after)
    changes[str(path.relative_to(dest))] = (before, after)
    diff = "".join(
        "".join(
            difflib.unified_diff(
                old.splitlines(True),
                new.splitlines(True),
                fromfile=f"a/{name}",
                tofile=f"b/{name}",
            )
        )
        for name, (old, new) in changes.items()
    )
    (a.root / "local_runtime_v2.patch").write_text(diff)
    records = [
        {
            "path": f["path"],
            "sha256": hashlib.sha256((dest / f["path"]).read_bytes()).hexdigest(),
        }
        for f in identity["files"]
    ]
    (a.root / "source_local_v2_identity.json").write_text(
        json.dumps(
            {
                "upstream_revision": REVISION,
                "patch_sha256": hashlib.sha256(diff.encode()).hexdigest(),
                "files": records,
                "changed_files": list(changes),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Created {dest}; changed only {list(changes)}")


if __name__ == "__main__":
    main()
