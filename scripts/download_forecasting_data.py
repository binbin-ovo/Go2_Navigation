"""Download the authors' public annotation archive, checksum it, extract only manifest files."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile
from go2_perception.io.config import load_config


def download(root, manifest):
    spec = load_config(manifest)
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(
            "Use an empty output directory; existing data is never overwritten"
        )
    request = urllib.request.Request(
        spec["archive_url"], headers={"User-Agent": "Go2-forecasting-research/1"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read(20_000_001)
    digest = hashlib.sha256(data).hexdigest()
    if digest != spec["archive_sha256"]:
        raise ValueError(
            "Dataset checksum mismatch; do not silently accept changed data"
        )
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for paths in spec["scenes"].values():
            for name in paths:
                if Path(name).is_absolute() or ".." in Path(name).parts:
                    raise ValueError("Unsafe path")
                dest = root / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(archive.read("datasets/" + name))
    (root / "download.json").write_text(
        json.dumps(dict(url=spec["archive_url"], sha256=digest), indent=2)
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="data/public/eth_ucy/datasets")
    p.add_argument("--manifest", default="configs/forecasting/eth_ucy_manifest.yaml")
    a = p.parse_args()
    download(a.output, a.manifest)
