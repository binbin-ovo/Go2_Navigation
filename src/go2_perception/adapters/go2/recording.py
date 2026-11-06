"""Portable recording format: JPEG/PNG + NumPy + JSONL, loadable without ROS or the SDK.

<root>/manifest.json, stats.json, camera.h264 (raw Annex-B, lossless original)
<root>/rgb/index.jsonl, rgb/000000.jpg
<root>/lidar/<topic-name>/index.jsonl, 000000.npy (N x 4 float32 XYZI), 000000_time.npy
<root>/state/<topic-name>.jsonl
Each ``.npy`` cloud is directly loadable by ``load_pointcloud`` and holds the published data.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA = "go2-recording-v1"


def topic_dirname(topic):
    return topic.removeprefix("rt/").replace("/", "_")


class RecordingWriter:
    def __init__(self, root, notes="", rgb_format="jpg", jpeg_quality=95):
        if rgb_format not in ("jpg", "png"):
            raise ValueError("rgb_format must be jpg or png")
        self.root = Path(root)
        if self.root.exists() and any(self.root.iterdir()):
            raise FileExistsError(f"Refusing to overwrite non-empty {self.root}")
        (self.root / "rgb").mkdir(parents=True, exist_ok=True)
        self.rgb_format, self.jpeg_quality = rgb_format, jpeg_quality
        self._rgb_index = open(self.root / "rgb" / "index.jsonl", "w")
        self._rgb_count = 0
        self._lidar = {}
        self._state = {}
        self.manifest = {
            "schema": SCHEMA,
            "created_host_time": time.time(),
            "notes": notes,
            "clocks": {
                "stamp": "go2_robot: header.stamp, not synchronized to the host",
                "host_recv": "host time.time() at receipt",
                "rtp_timestamp": "camera 90 kHz RTP counter, arbitrary origin",
            },
        }
        self._write_manifest()

    def _write_manifest(self):
        (self.root / "manifest.json").write_text(json.dumps(self.manifest, indent=2) + "\n")

    def write_rgb(self, image, rtp_timestamp, host_recv_first, host_recv_last, complete=True):
        import cv2

        name = f"{self._rgb_count:06d}.{self.rgb_format}"
        params = (
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
            if self.rgb_format == "jpg"
            else []
        )
        if not cv2.imwrite(str(self.root / "rgb" / name), image, params):
            raise OSError(f"Could not write {name}")
        record = {
            "file": name,
            "rtp_timestamp": int(rtp_timestamp),
            "host_recv_first": host_recv_first,
            "host_recv_last": host_recv_last,
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "complete": bool(complete),
            "encoding": "bgr8",
            "frame_id": "go2_front_camera_unrectified",
        }
        self._rgb_index.write(json.dumps(record) + "\n")
        self._rgb_count += 1

    def write_lidar(self, frame):
        key = topic_dirname(frame.topic)
        if key not in self._lidar:
            folder = self.root / "lidar" / key
            folder.mkdir(parents=True, exist_ok=True)
            self._lidar[key] = [open(folder / "index.jsonl", "w"), 0, folder]
        index, count, folder = self._lidar[key]
        intensity = (
            frame.intensity
            if frame.intensity is not None
            else np.full(len(frame.xyz), np.nan, np.float32)
        )
        cloud = np.column_stack((frame.xyz, intensity)).astype(np.float32)
        np.save(folder / f"{count:06d}.npy", cloud, allow_pickle=False)
        if frame.time is not None:
            np.save(folder / f"{count:06d}_time.npy", frame.time, allow_pickle=False)
        record = {
            "file": f"{count:06d}.npy",
            "topic": frame.topic,
            "stamp": frame.stamp,
            "host_recv": frame.host_recv,
            "frame_id": frame.frame_id,
            "points": len(frame.xyz),
            "has_intensity": frame.intensity is not None,
            "has_time": frame.time is not None,
            "ring_values": None
            if frame.ring is None
            else sorted(int(v) for v in np.unique(frame.ring)),
        }
        index.write(json.dumps(record) + "\n")
        self._lidar[key][1] = count + 1

    def write_state(self, topic, record):
        key = topic_dirname(topic)
        if key not in self._state:
            (self.root / "state").mkdir(exist_ok=True)
            self._state[key] = open(self.root / "state" / f"{key}.jsonl", "w")
        self._state[key].write(json.dumps(record) + "\n")

    def close(self, stats=None):
        self._rgb_index.close()
        for index, _, _ in self._lidar.values():
            index.close()
        for f in self._state.values():
            f.close()
        self.manifest["rgb_frames"] = self._rgb_count
        self.manifest["lidar_frames"] = {k: v[1] for k, v in self._lidar.items()}
        self._write_manifest()
        if stats is not None:
            (self.root / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")


@dataclass
class ReplayLidarFrame:
    record: dict
    path: Path

    def load(self):
        """Return a LidarFrame identical to what was recorded."""
        from .pointcloud import LidarFrame

        cloud = np.load(self.path, allow_pickle=False)
        time_path = self.path.with_name(self.path.stem + "_time.npy")
        return LidarFrame(
            self.record["topic"],
            self.record["stamp"],
            self.record["frame_id"],
            self.record["host_recv"],
            cloud[:, :3],
            cloud[:, 3] if self.record["has_intensity"] else None,
            np.load(time_path, allow_pickle=False) if time_path.exists() else None,
        )


class RecordingReader:
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest.get("schema") != SCHEMA:
            raise ValueError(f"Unsupported schema: {self.manifest.get('schema')}")

    @staticmethod
    def _jsonl(path):
        if not path.exists():
            return []
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]

    def rgb_records(self):
        return self._jsonl(self.root / "rgb" / "index.jsonl")

    def rgb_path(self, record):
        return self.root / "rgb" / record["file"]

    def lidar_topics(self):
        folder = self.root / "lidar"
        return sorted(p.name for p in folder.iterdir()) if folder.exists() else []

    def lidar_frames(self, topic):
        folder = self.root / "lidar" / topic_dirname(topic)
        return [
            ReplayLidarFrame(r, folder / r["file"])
            for r in self._jsonl(folder / "index.jsonl")
        ]

    def state(self, topic):
        return self._jsonl(self.root / "state" / f"{topic_dirname(topic)}.jsonl")
