"""Download ONE pinned public inference implementation/checkpoint; never robot clients.

Run in the isolated LightNav Python environment. Source-only is the default.
Existing changed files are refused. No model is executed by this script.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

REPO = "lightorigins/LightNav-0"
CODE_REVISION = "3015508b70fb6e30bbd66701e94c9b916dcb8df8"
MODEL = "LightOriginsHQ/LightNav-0"
MODEL_REVISION = "826dc5fbfa37afa8293d2e336d329b6ffc0bfb64"


def fetch(url):
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def git_blob_sha(data):
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--weights", action="store_true")
    a = p.parse_args()
    a.root.mkdir(parents=True, exist_ok=True)
    tree = json.loads(
        fetch(
            f"https://api.github.com/repos/{REPO}/git/trees/{CODE_REVISION}?recursive=1"
        )
    )
    if tree.get("truncated"):
        raise ValueError("Incomplete upstream tree at pinned revision")
    entries = [
        x
        for x in tree["tree"]
        if x["type"] == "blob"
        and (
            x["path"].startswith("src/lightnav/")
            or x["path"]
            in {"pyproject.toml", "README.md", "LICENSE", "THIRD_PARTY_NOTICES.md"}
        )
    ]

    def source(entry):
        path = a.root / "source" / entry["path"]
        data = (
            path.read_bytes()
            if path.exists()
            else fetch(
                f"https://raw.githubusercontent.com/{REPO}/{CODE_REVISION}/{entry['path']}"
            )
        )
        if git_blob_sha(data) != entry["sha"]:
            raise ValueError(f"Upstream or existing source changed: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(data)
        return {"path": entry["path"], "sha256": hashlib.sha256(data).hexdigest()}

    with ThreadPoolExecutor(max_workers=6) as pool:
        files = list(pool.map(source, entries))
    (a.root / "source_identity.json").write_text(
        json.dumps(
            {"repository": REPO, "revision": CODE_REVISION, "files": files}, indent=2
        )
        + "\n"
    )
    print(
        f"Verified {len(files)} inference source files at {CODE_REVISION}", flush=True
    )
    if not a.weights:
        return
    metadata = json.loads(
        fetch(
            f"https://huggingface.co/api/models/{MODEL}/revision/{MODEL_REVISION}?blobs=true"
        )
    )
    if metadata["sha"] != MODEL_REVISION:
        raise ValueError("Checkpoint revision mismatch")
    required = [x for x in metadata["siblings"] if x["rfilename"] != "wechat_group.png"]
    needed = sum(
        x["size"]
        for x in required
        if not (a.root / "checkpoint" / x["rfilename"]).exists()
    )
    if shutil.disk_usage(a.root).free < needed + 3 * 2**30:
        raise RuntimeError(
            "Insufficient disk: retain 3 GiB free AFTER the checkpoint download"
        )
    from huggingface_hub import snapshot_download

    snapshot_download(
        MODEL,
        revision=MODEL_REVISION,
        local_dir=a.root / "checkpoint",
        allow_patterns=[x["rfilename"] for x in required],
        max_workers=2,
    )
    # Full weight hashing, not only a manifest name or a matching file length.
    verified = []
    for entry in required:
        path = a.root / "checkpoint" / entry["rfilename"]
        h = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
                h.update(block)
        if path.stat().st_size != entry["size"]:
            raise ValueError(f"Checkpoint size mismatch: {path}")
        expected = entry.get("lfs", {}).get("sha256")
        if expected and h.hexdigest() != expected:
            raise ValueError(f"Checkpoint SHA256 mismatch: {path}")
        verified.append(
            {"path": entry["rfilename"], "sha256": h.hexdigest(), "size": entry["size"]}
        )
    (a.root / "checkpoint_identity.json").write_text(
        json.dumps(
            {"model": MODEL, "revision": MODEL_REVISION, "files": verified}, indent=2
        )
        + "\n"
    )
    print(f"Verified checkpoint and RVQ decoder at {MODEL_REVISION}", flush=True)


if __name__ == "__main__":
    main()
