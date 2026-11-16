"""Bounded, run-local cache of lossless calibrated RGB, never temporal features."""

from collections import OrderedDict
import copy
import hashlib
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np

from .episodes import file_sha256
from .model_preprocessing import calibrated_view
from .observation import load


class PreparedFrameCache:
    """Hash-verified source archives bind pixels, calibration, stamps and epoch.

    Entries are private lossless PNG files. Request-local hard links keep immutable
    evidence alive after LRU eviction or instruction/epoch reset. No arrays/features
    are kept in RAM. History selection still runs in full on every request.
    """

    def __init__(self, root, *, max_entries=128, max_bytes=256 * 2**20,
                 allow_synthetic=False):
        if not 1 <= max_entries <= 4096 or not 1 <= max_bytes <= 512 * 2**20:
            raise ValueError("Invalid prepared-cache bounds")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.max_entries, self.max_bytes = max_entries, max_bytes
        self.allow_synthetic = allow_synthetic
        self.settings = {
            "version": "lossless_calibrated_rgb_v1", "opencv": cv2.__version__,
            "numpy": np.__version__, "png_compression": 0,
            "preprocessor_sha256": file_sha256(Path(__file__).with_name("model_preprocessing.py")),
            "cache_code_sha256": file_sha256(__file__),
            "allow_synthetic": allow_synthetic,
        }
        self.entries = OrderedDict()
        self.bytes = 0
        self.counts = dict(hits=0, misses=0, evicted=0, resets=0, bypassed=0)

    def clear(self):
        for entry in self.entries.values():
            entry["path"].unlink()
        self.entries.clear()
        self.bytes = 0
        self.counts["resets"] += 1

    def status(self):
        return {**self.counts, "entries": len(self.entries), "bytes": self.bytes,
                "max_entries": self.max_entries, "max_bytes": self.max_bytes}

    def prepare(self, source, destination):
        started = time.monotonic()
        actual = file_sha256(source["observation_path"])
        if actual != source["observation_sha256"]:
            raise ValueError("Stored observation changed")
        identity = {"source_archive_sha256": actual, "frame_id": source["frame_id"],
                    "clock_epoch": source["clock_epoch"], "settings": self.settings}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        stages = {"source_hash_and_load_s": time.monotonic() - started,
                  "rectification_s": 0.0, "image_encoding_and_write_s": 0.0}
        entry = self.entries.get(key)
        hit = entry is not None
        stage = time.monotonic()
        if hit:
            if file_sha256(entry["path"]) != entry["sha256"]:
                raise ValueError("Prepared cache evidence changed")
            self.entries.move_to_end(key)
            self.counts["hits"] += 1
        else:
            observation = load(source["observation_path"])
            stages["source_hash_and_load_s"] += time.monotonic() - stage
            stage = time.monotonic()
            view = calibrated_view(observation, allow_synthetic=self.allow_synthetic)
            stages["rectification_s"] = time.monotonic() - stage
            stage = time.monotonic()
            path = self.root / f"{key}.png"
            if not cv2.imwrite(str(path), cv2.cvtColor(view.image, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_PNG_COMPRESSION, 0]):
                raise OSError("Cannot save cached prepared frame")
            entry = {"path": path, "sha256": file_sha256(path),
                     "metadata": copy.deepcopy(view.metadata), "size": path.stat().st_size}
            self.counts["misses"] += 1
            # Calibration is explicitly retained in the provenance; its bytes,
            # image identity and timestamps are all bound by the source archive hash.
            while self.entries and (len(self.entries) >= self.max_entries or
                                    self.bytes + entry["size"] > self.max_bytes):
                _, old = self.entries.popitem(last=False)
                old["path"].unlink()
                self.bytes -= old["size"]
                self.counts["evicted"] += 1
            if entry["size"] <= self.max_bytes:
                self.entries[key] = entry
                self.bytes += entry["size"]
            else:
                self.counts["bypassed"] += 1
        os.link(entry["path"], destination)
        if not hit and key not in self.entries:
            entry["path"].unlink()  # request evidence is still linked
        stages["image_encoding_and_write_s"] = time.monotonic() - stage
        return (entry["sha256"], copy.deepcopy(entry["metadata"]),
                {"key": key, "hit": hit, "identity": identity,
                 "calibration": copy.deepcopy(entry["metadata"]["source"])}, stages)
