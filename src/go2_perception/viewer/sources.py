"""Load real-recording inputs for the viewer: 5A scene rows, Debug tap, Phase 3 cache, RGB.

Everything here is file reading and time alignment; no perception code is imported or run.
All geometry is already in odom. Phase 3 detection boxes are stored in ``base_link_level``
with their planar pose, so their corners are transformed exactly once, here.
"""

from __future__ import annotations

import bisect
import json
from pathlib import Path

import numpy as np

from .scene import read_frames

MOTION = {"M": "MOVING", "S": "STATIONARY", "U": "UNCERTAIN"}


def jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class Tap:
    """Debug sidecar written by scripts/phase5b_scene_tap.py."""

    def __init__(self, root):
        root = Path(root)
        z = np.load(root / "tap.npz")
        self.frame_t = z["frame_t"]
        self.scan_off, self.scan_xyz, self.scan_acq, self.scan_ng = z["scan_off"], z["scan_xyz"], z["scan_acq"], z["scan_ng"]
        self.win_off, self.win_xyz, self.win_code = z["win_off"], z["win_xyz"], z["win_code"]
        self.records = jsonl(root / "tap.jsonl")
        self._time_index = {round(float(t), 6): i for i, t in enumerate(self.frame_t)}

    def index(self, t):
        return self._time_index.get(round(float(t), 6))

    def scan(self, k):
        a, b = self.scan_off[k], self.scan_off[k + 1]
        return self.scan_xyz[a:b].astype(float), self.scan_acq[a:b], self.scan_ng[a:b]

    def accumulated(self, k, window_s=1.0):
        """Union of the transformed scans within ``window_s`` before frame k (exactly what the
        5A accumulator retains, before voxelization). Returns (xyz, ages)."""
        t = self.frame_t[k]
        j = k
        while j > 0 and self.frame_t[j - 1] >= t - window_s - 0.1:
            j -= 1
        a, b = self.scan_off[j], self.scan_off[k + 1]
        acq = self.scan_acq[a:b]
        keep = acq >= t - window_s
        return self.scan_xyz[a:b][keep].astype(float), (t - acq)[keep]

    def window(self, k):
        a, b = self.win_off[k], self.win_off[k + 1]
        return self.win_xyz[a:b].astype(float), self.win_code[a:b]

    def clusters(self, k):
        out = []
        for c in self.records[k]["clusters"]:
            out.append(dict(c, lo=np.asarray(c["lo"]), hi=np.asarray(c["hi"]), centroid=np.asarray(c["centroid"])))
        return out


class Phase3:
    """Original 0.3 s pipeline: cached tracker inputs (clusters + pose) and tracker outputs."""

    def __init__(self, inputs_dir, tracks_file):
        self.inputs = jsonl(Path(inputs_dir) / "frames.jsonl")
        self.records = jsonl(tracks_file)
        self.times = [r["timestamp"] for r in self.inputs]
        if [round(r["t"], 6) for r in self.records] != [round(t, 6) for t in self.times]:
            raise ValueError("Phase 3 tracker records do not match their input frames")
        self._by_t = {round(t, 3): i for i, t in enumerate(self.times)}

    def index(self, t_mid):
        """Exact frame at a LiDAR acquisition midpoint (both pipelines use the same stamps)."""
        return self._by_t.get(round(float(t_mid), 3))

    def latest_before(self, t_mid):
        k = bisect.bisect_right(self.times, t_mid + 1e-6) - 1
        return k if k >= 0 else None

    def boxes(self, k):
        r = self.inputs[k]
        if not r.get("robot_pose"):
            return []
        m = np.asarray(r["robot_pose"]["T_parent_child"])
        out = []
        for d in r["detections"]:
            lo, hi = d["bbox_min"], d["bbox_max"]
            c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
            out.append(c @ m[:3, :3].T + m[:3, 3])
        return out

    def tracks(self, k):
        out = []
        for row in self.records[k]["tracks"]:
            out.append(dict(id=row[0], xyz=row[1:4], v=row[4:6], vstd=row[6], status=row[7], motion=MOTION[row[8]], observed=row[9] == 0 and row[7] != "L", misses=row[9]))
        return out

    def latency_ms(self, k):
        return self.records[k].get("latency_ms")


class Rgb:
    """Recorded camera frames mapped to the robot clock (approximate; host receipt, not exposure)."""

    def __init__(self, recording_dir, lidar_times, host_offset, detections_file=None, max_dt=0.15):
        from go2_perception.visualization.diagnostic_video import align_rgb

        rec = Path(recording_dir)
        self.dir = rec / "rgb"
        self.index = jsonl(self.dir / "index.jsonl")
        self.idx, self.dt = align_rgb(lidar_times, [r["host_recv_first"] for r in self.index], host_offset, max_dt)
        self.dets = {}
        if detections_file and Path(detections_file).exists():
            self.dets = {d["file"]: d["detections"] for d in json.loads(Path(detections_file).read_text())}

    def get(self, k):
        """(BGR image or None, camera index or None, dt seconds or None, detections)."""
        import cv2

        i = self.idx[k]
        if i is None:
            return None, None, self.dt[k], []
        r = self.index[i]
        dets = [(d["class_name"], d["confidence"], d["bbox_xyxy"]) for d in self.dets.get(r["file"], []) if d["confidence"] >= 0.3]
        return cv2.imread(str(self.dir / r["file"])), i, self.dt[k], dets


def host_offset_from_cache(annotation_cache):
    return float(np.load(annotation_cache)["host_offset"])


def load_annotations(path):
    """Retrospective, evaluation-only annotation states keyed by rounded LiDAR midpoint."""
    doc = json.loads(Path(path).read_text())
    return doc, {round(f["t"], 3): f["objects"] for f in doc["frames"]}


def load_scene(path, **kw):
    return list(read_frames(path, **kw))
