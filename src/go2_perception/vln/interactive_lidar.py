"""Read-only LiDAR/odometry consumer for the interactive review tool (core environment).

The DDS subscribers live in a separate hardware-environment child
(``scripts/go2_lidar_bridge.py``); this module only parses its stdout and opens no
socket or DDS participant. Process boundary and clocks:

- Scans are single ``rt/utlidar/cloud_base`` messages in base_link. Accumulation
  uses the existing ``SlidingWindowAccumulator`` (odometry ego-motion compensation)
  over 0.5 s and 1.0 s windows of scan midpoints. ``cloud_deskewed`` is never used,
  so nothing is accumulated twice. Scans without a bracketing pose are refused.
- Robot stamps stay in the robot clock of one *robot epoch*. A backwards stamp jump
  or a jump in host-minus-robot starts a new epoch (reboot): pose buffer, windows and
  the offset estimate are reset. An offset is never reused across epochs.
- host-minus-robot is only an *estimate* (lower envelope of odom receipt - stamp,
  including odom's minimum transport latency). It is displayed, never used to fuse
  LiDAR with RGB. Without a valid estimate the streams are shown separately only.
"""

from collections import deque
import hashlib
import io
import json
import math
import os
from pathlib import Path
from queue import Empty, Full, Queue
import subprocess
import threading
import time

import numpy as np

from ..adapters.go2.window import PoseBuffer, SlidingWindowAccumulator

WINDOWS_S = (0.5, 1.0)  # obstacle / display windows of configs/go2_vln_sensors.yaml
SECTORS = 72
GRID = {"cell_m": 0.1, "extent_x": (-3.0, 7.0), "extent_y": (-5.0, 5.0)}
SYNC_MIN_SAMPLES = 50
REBOOT_JUMP_S = 1.0
HOST_STEP_S = 0.02


def coverage(xyz, blind_radius_m=0.3):
    """Azimuth sectors and grid cells holding any return. Everything else is UNKNOWN."""
    if len(xyz) == 0:
        return {"sectors_with_returns": 0, "sectors": SECTORS, "sector_fraction": 0.0,
                "grid_cells_with_returns": 0, "grid_cells": 10000, "grid_fraction": 0.0}
    r = np.hypot(xyz[:, 0], xyz[:, 1])
    p = xyz[r >= blind_radius_m]
    s = np.floor((np.degrees(np.arctan2(p[:, 1], p[:, 0])) % 360) / (360 / SECTORS)).astype(int) % SECTORS
    nx = ny = 100
    i = np.floor((p[:, 0] - GRID["extent_x"][0]) / GRID["cell_m"]).astype(int)
    j = np.floor((p[:, 1] - GRID["extent_y"][0]) / GRID["cell_m"]).astype(int)
    inside = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
    cells = len(np.unique(i[inside] * ny + j[inside]))
    sectors = len(np.unique(s))
    return {"sectors_with_returns": int(sectors), "sectors": SECTORS,
            "sector_fraction": sectors / SECTORS, "grid_cells_with_returns": int(cells),
            "grid_cells": nx * ny, "grid_fraction": cells / (nx * ny)}


def downsample(xyz, limit=3000):
    if len(xyz) > limit:
        xyz = xyz[np.linspace(0, len(xyz) - 1, limit).astype(int)]
    return np.round(xyz.astype(float), 2).tolist()


def yaw_of(q):
    x, y, z, w = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class LidarState:
    """Latest scan, accumulated windows, pose validity, intervals and clock epochs."""

    def __init__(self, *, monotonic=time.monotonic, pose_stale_s=0.25):
        self.monotonic = monotonic
        self.pose_stale_s = pose_stale_s
        self.robot_epoch = 0
        self.host_epoch = 0
        self.bridge_id = None
        self.source = None
        self.counts = {"scans": 0, "odom": 0, "scan_seq_gaps": 0, "scans_missing": 0,
                       "odom_seq_gaps": 0, "odom_missing": 0, "robot_epoch_changes": 0,
                       "host_clock_steps": 0, "out_of_order_scans": 0,
                       "refused_no_pose": {str(w): 0 for w in WINDOWS_S}}
        self.last_seq = {"scan": None, "odom": None}
        self._reset_epoch()
        self.host_offset = None

    def _reset_epoch(self):
        self.poses = PoseBuffer(max_age_s=5.0, max_gap_s=0.1)
        self.windows = {w: SlidingWindowAccumulator(self.poses, w, max_frames=64) for w in WINDOWS_S}
        self.latest = None
        self.last_odom = None
        self.last_stamp = None
        self.offset_min, self.offset_samples, self.epoch_started_mono = None, 0, None
        self.scan_robot_t = deque(maxlen=64)
        self.scan_host_mono = deque(maxlen=64)
        self.pending_windows = []  # scans waiting (briefly) for a bracketing odom sample

    def epoch_label(self):
        return f"{self.source or 'unknown'}:{self.bridge_id or 'none'}:robot_epoch{self.robot_epoch}"

    def _check_clocks(self, header, now):
        host_mono = header["host_recv_wall_s"] - header["wall_minus_monotonic_s"]
        offset = header["wall_minus_monotonic_s"]
        if self.host_offset is not None and abs(offset - self.host_offset) > HOST_STEP_S:
            # Host wall clock stepped: host-minus-robot is no longer comparable.
            self.counts["host_clock_steps"] += 1
            self.host_epoch += 1
            self.offset_min, self.offset_samples = None, 0
        self.host_offset = offset
        stamp = header["stamp"]
        rough = header["host_recv_wall_s"] - stamp
        reboot = self.last_stamp is not None and stamp < self.last_stamp - REBOOT_JUMP_S
        jump = self.offset_min is not None and abs(rough - self.offset_min) > 2.0
        if reboot or jump:
            self.robot_epoch += 1
            self.counts["robot_epoch_changes"] += 1
            self._reset_epoch()
            return host_mono, "robot_epoch_changed"
        if self.epoch_started_mono is None:
            self.epoch_started_mono = host_mono
        self.last_stamp = stamp if self.last_stamp is None else max(self.last_stamp, stamp)
        return host_mono, None

    def _seq(self, kind, seq):
        last = self.last_seq[kind]
        if last is not None and seq != last + 1:
            self.counts[f"{kind}_seq_gaps"] += 1
            self.counts["scans_missing" if kind == "scan" else "odom_missing"] += max(0, seq - last - 1)
        self.last_seq[kind] = seq

    def hello(self, header):
        if self.bridge_id is not None and header.get("bridge_id") != self.bridge_id:
            self.robot_epoch += 1
            self.counts["robot_epoch_changes"] += 1
            self._reset_epoch()
        self.bridge_id, self.source = header.get("bridge_id"), header.get("source")
        self.last_seq = {"scan": None, "odom": None}

    def odom(self, header, now=None):
        host_mono, change = self._check_clocks(header, now)
        self._seq("odom", header["seq"])
        self.counts["odom"] += 1
        self.poses.add(header["stamp"], header["position"], header["q_xyzw"])
        rough = header["host_recv_wall_s"] - header["stamp"]
        if self.offset_min is None or rough < self.offset_min:
            self.offset_min = rough
        self.offset_samples += 1
        self.last_odom = {"stamp": header["stamp"], "host_mono": host_mono,
                          "position": header["position"], "q_xyzw": header["q_xyzw"]}
        return change

    def scan(self, header, xyz, now=None):
        host_mono, change = self._check_clocks(header, now)
        self._seq("scan", header["seq"])
        self.counts["scans"] += 1
        span = header.get("time_span_s")
        t_mid = header["stamp"] + (0.5 * span if span else 0.0)
        if self.latest is not None and t_mid <= self.latest["t_mid"]:
            self.counts["out_of_order_scans"] += 1
            return change
        self.latest = {"xyz": xyz, "seq": header["seq"], "stamp": header["stamp"], "t_mid": t_mid,
                       "time_span_s": span, "host_mono": host_mono, "n": int(len(xyz)),
                       "frame_id": header.get("frame_id")}
        self.scan_robot_t.append(t_mid)
        self.scan_host_mono.append(host_mono)
        self.pending_windows.append((t_mid, xyz))
        self._drain_pending()
        return change

    def _drain_pending(self):
        # A scan can only be compensated once odometry brackets its midpoint; retry
        # briefly (odom arrives at ~50 Hz) instead of merging uncompensated points.
        keep = []
        newest_odom = self.last_odom["stamp"] if self.last_odom else None
        for t_mid, xyz in self.pending_windows:
            if newest_odom is not None and newest_odom >= t_mid:
                for w, acc in self.windows.items():
                    if not acc.add(xyz, t_mid):
                        self.counts["refused_no_pose"][str(w)] += 1
            elif newest_odom is None or t_mid - newest_odom > 0.3 or len(keep) >= 8:
                for w in self.windows:
                    self.counts["refused_no_pose"][str(w)] += 1
            else:
                keep.append((t_mid, xyz))
        self.pending_windows = keep

    def after_odom(self):
        if self.pending_windows:
            self._drain_pending()

    def sync(self):
        enough = self.offset_samples >= SYNC_MIN_SAMPLES
        return {
            "status": "estimated" if enough else "invalid",
            "reason": None if enough else f"{self.offset_samples}/{SYNC_MIN_SAMPLES} odom samples in this robot epoch",
            "host_minus_robot_s": self.offset_min if enough else None,
            "robot_epoch": self.epoch_label(),
            "host_epoch": self.host_epoch,
            "provenance": "lower envelope of odom (host wall receipt - robot stamp) within this "
                          "robot epoch; includes odom's minimum transport latency; not hardware "
                          "synchronized; display only, never used to fuse LiDAR with RGB",
        }

    def snapshot(self, now):
        latest = self.latest
        pose = {"valid": False, "reason": "no odometry in this robot epoch"}
        if self.last_odom is not None:
            age = now - self.last_odom["host_mono"]
            if age > self.pose_stale_s:
                pose = {"valid": False, "reason": f"odometry silent for {age:.2f} s (host)"}
            else:
                p = self.last_odom["position"]
                pose = {"valid": True, "frame": "odom (robot-reported; origin at odometry start)",
                        "x": p[0], "y": p[1], "z": p[2], "yaw_rad": yaw_of(self.last_odom["q_xyzw"]),
                        "stamp_robot": self.last_odom["stamp"], "host_age_s": age}
        views = {}
        windows = {}
        if latest is not None:
            views["latest"] = latest["xyz"]
        for w, acc in self.windows.items():
            pts, newest, frames, span = acc.merged()
            key = str(w)
            if frames == 0:
                windows[key] = {"available": False, "reason": "no pose-compensated scans (pose unknown or none yet)",
                                "frames": 0}
                continue
            views[key] = pts.astype(np.float32)
            windows[key] = {"available": True, "frames": frames, "span_robot_s": span,
                            "newest_t_mid_robot": newest, "points": int(len(pts)),
                            "frame": "base_link of the newest scan in the window"}

        def intervals(values):
            v = np.diff(np.asarray(values))
            if len(v) == 0:
                return None
            return {"median_s": float(np.median(v)), "max_s": float(np.max(v)), "n": int(len(v))}

        latest_info = None
        if latest is not None:
            latest_info = {k: latest[k] for k in ("seq", "stamp", "t_mid", "time_span_s", "n", "frame_id")}
            latest_info["host_age_s"] = now - latest["host_mono"]
            latest_info["robot_age_vs_latest_odom_s"] = (
                None if self.last_odom is None else self.last_odom["stamp"] - latest["t_mid"])
        return {
            "source": self.source,
            "robot_epoch": self.epoch_label(),
            "latest_scan": latest_info,
            "robot_scan_intervals": intervals(self.scan_robot_t),
            "host_receipt_intervals": intervals(self.scan_host_mono),
            "windows": windows,
            "pose": pose,
            "sync": self.sync(),
            "coverage": {k: coverage(v) for k, v in views.items()},
            "counts": json.loads(json.dumps(self.counts)),
            "unknowns": [
                "no returns inside 0.3 m (cloud_base blind radius)",
                "cells/sectors without returns are UNKNOWN, never free",
                "no registration to the camera or to model waypoints (no measured T_base_camera)",
            ],
        }, {k: downsample(v) for k, v in views.items()}


class LidarRecorder:
    """Record on a separate writer. Bound scan bytes, messages, queue and pending gap metadata."""

    MAX_GAP_ROWS = 64

    def __init__(self, root, *, queue_size=4, chunk_s=1.0, quota=None,
                 max_chunk_bytes=4 * 1024 * 1024, max_messages=512):
        if min(queue_size, chunk_s, max_chunk_bytes, max_messages) <= 0:
            raise ValueError("recorder bounds must be positive")
        self.max_chunk_bytes, self.max_messages = max_chunk_bytes, max_messages
        self.closing = False
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.queue = Queue(maxsize=queue_size)
        self.chunk_s, self.quota = chunk_s, quota
        self.lock = threading.Lock()
        self.trial_id = None
        self.chunk = None
        self.index = self.root / "index.jsonl"
        self.counts = {"chunks_written": 0, "chunks_dropped": 0, "bytes": 0, "errors": 0}
        self.n = 0
        if self.index.is_file():
            with self.index.open() as stream:
                for line in stream:
                    row = json.loads(line)
                    self.n = max(self.n, row.get("last_chunk", row["chunk"]))
        self.gap_summary = None
        self.index_error = None
        self.gaps = []
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def start(self, trial_id):
        with self.lock:
            if self.closing:
                raise RuntimeError("LiDAR recorder is closing")
            self._flush_locked()
            self.trial_id = trial_id

    def stop(self):
        with self.lock:
            self._flush_locked()
            self.trial_id = None

    def _new_chunk(self, epoch, now):
        return {"started": now, "epoch": epoch, "trial_id": self.trial_id,
                "scans": [], "odom": [], "point_bytes": 0}

    def offer(self, kind, header, xyz, epoch, now):
        size = xyz.nbytes if kind == "scan" else 0
        with self.lock:
            if self.trial_id is None or self.closing:
                return
            c = self.chunk
            if c is not None and (c["epoch"] != epoch or now - c["started"] >= self.chunk_s or
                                  c["point_bytes"] + size > self.max_chunk_bytes or
                                  len(c["scans"]) + len(c["odom"]) >= self.max_messages):
                self._flush_locked()
            if size > self.max_chunk_bytes:
                self.n += 1
                row = self._new_chunk(epoch, now)
                row.update(number=self.n, scans=[(header, xyz)])
                self._drop_locked(row, "single_scan_exceeds_byte_limit")
                return
            if self.chunk is None:
                self.chunk = self._new_chunk(epoch, now)
            if kind == "scan":
                self.chunk["scans"].append((header, xyz))
                self.chunk["point_bytes"] += size
            else:
                self.chunk["odom"].append(header)

    def _flush_locked(self):
        chunk, self.chunk = self.chunk, None
        if chunk is None or not (chunk["scans"] or chunk["odom"]):
            return
        self.n += 1
        chunk["number"] = self.n
        try:
            self.queue.put_nowait(chunk)
        except Full:
            self._drop_locked(chunk, "writer_queue_full")

    def _drop_locked(self, chunk, reason):
        self.counts["chunks_dropped"] += 1
        row = self._describe(chunk, dropped=True, reason=reason)
        if len(self.gaps) < self.MAX_GAP_ROWS:
            self.gaps.append(row)
            return
        # During an extended disk stall, retain totals and a conservative host
        # interval. Individual source IDs may span epochs/trials and cannot be inferred.
        if self.gap_summary is None:
            self.gap_summary = dict(row, reason="coalesced_recorder_drops", n_chunks=1, last_chunk=row["chunk"],
                                    scan_seq=None, odom_seq=None)
            return
        group = self.gap_summary
        group["n_chunks"] += 1
        group["last_chunk"] = row["chunk"]
        for key in ("n_scans", "n_odom", "n_points"):
            group[key] += row[key]
        for key in ("trial_id", "robot_epoch"):
            if group[key] != row[key]:
                group[key] = None
        for key in ("host_receipt_interval_wall_s", "host_receipt_interval_monotonic_s", "source_interval_robot_s"):
            if key == "source_interval_robot_s" and group["robot_epoch"] is None:
                group[key] = None
            else:
                group[key] = [min(group[key][0], row[key][0]), max(group[key][1], row[key][1])]

    @staticmethod
    def _describe(chunk, dropped=False, **extra):
        scans, odom = chunk["scans"], chunk["odom"]
        stamps = [h["stamp"] for h, _ in scans] + [h["stamp"] for h in odom]
        headers = [h for h, _ in scans] + odom
        hosts = [h["host_recv_wall_s"] for h in headers]
        mono = [h["host_recv_wall_s"] - h["wall_minus_monotonic_s"] for h in headers]
        return {
            "chunk": chunk["number"], "trial_id": chunk["trial_id"], "robot_epoch": chunk["epoch"],
            "dropped_not_recorded": dropped,
            "source_interval_robot_s": [min(stamps), max(stamps)] if stamps else None,
            "host_receipt_interval_wall_s": [min(hosts), max(hosts)] if hosts else None,
            "host_receipt_interval_monotonic_s": [min(mono), max(mono)] if mono else None,
            "scan_seq": [scans[0][0]["seq"], scans[-1][0]["seq"]] if scans else None,
            "odom_seq": [odom[0]["seq"], odom[-1]["seq"]] if odom else None,
            "n_scans": len(scans), "n_odom": len(odom),
            "n_points": int(sum(len(x) for _, x in scans)), **extra,
        }

    def _run(self):
        while True:
            try:
                chunk = self.queue.get(timeout=0.1)
            except Empty:
                with self.lock:
                    done = self.closing and self.queue.empty()
                self._write_index()
                if done:
                    return
                continue
            try:
                if self.quota is not None and not self.quota():
                    raise OSError("recording quota reached")
                scans, odom = chunk["scans"], chunk["odom"]
                arrays = {
                    "points_xyz_base_link": np.concatenate([x for _, x in scans]).astype(np.float32)
                    if scans else np.empty((0, 3), np.float32),
                    "scan_offsets": np.cumsum([0] + [len(x) for _, x in scans]).astype(np.int64),
                    "scan_stamp_robot": np.array([h["stamp"] for h, _ in scans], float),
                    "scan_time_span_s": np.array([np.nan if h.get("time_span_s") is None else h["time_span_s"]
                                                  for h, _ in scans], float),
                    "scan_host_recv_wall": np.array([h["host_recv_wall_s"] for h, _ in scans], float),
                    "scan_wall_minus_monotonic": np.array([h["wall_minus_monotonic_s"] for h, _ in scans], float),
                    "scan_seq": np.array([h["seq"] for h, _ in scans], np.int64),
                    "odom_stamp_robot": np.array([h["stamp"] for h in odom], float),
                    "odom_host_recv_wall": np.array([h["host_recv_wall_s"] for h in odom], float),
                    "odom_wall_minus_monotonic": np.array([h["wall_minus_monotonic_s"] for h in odom], float),
                    "odom_position": np.array([h["position"] for h in odom], float).reshape(-1, 3),
                    "odom_q_xyzw": np.array([h["q_xyzw"] for h in odom], float).reshape(-1, 4),
                    "odom_seq": np.array([h["seq"] for h in odom], np.int64),
                }
                for kind in ("scan", "odom"):
                    arrays[kind + "_host_recv_monotonic_s"] = (
                        arrays[kind + "_host_recv_wall"] - arrays[kind + "_wall_minus_monotonic"])
                buffer = io.BytesIO()
                np.savez_compressed(buffer, **arrays)
                data = buffer.getvalue()
                name = f"chunk_{chunk['number']:06d}.npz"
                tmp = self.root / (name + ".tmp")
                tmp.write_bytes(data)
                tmp.replace(self.root / name)
                row = self._describe(chunk, file=name, sha256=hashlib.sha256(data).hexdigest(),
                                     bytes=len(data))
                self.counts["chunks_written"] += 1
                self.counts["bytes"] += len(data)
            except Exception as error:
                with self.lock:
                    self.counts["errors"] += 1
                    self.counts["chunks_dropped"] += 1
                row = self._describe(chunk, dropped=True, reason="write_failed", error=repr(error))
            self._write_index(row)

    def _write_index(self, row=None):
        with self.lock:
            rows, self.gaps = self.gaps, []
            if self.gap_summary is not None:
                rows.append(self.gap_summary)
                self.gap_summary = None
        if row is not None:
            rows.insert(0, row)
        if rows:
            try:
                with self.index.open("a") as stream:
                    for item in rows:
                        stream.write(json.dumps(item, allow_nan=False) + "\n")
            except OSError as error:
                self.counts["errors"] += 1
                self.index_error = repr(error)

    def close(self, timeout=5.0):
        with self.lock:
            self._flush_locked()
            self.trial_id = None
            self.closing = True
        self.thread.join(timeout)

    def status(self):
        return {**self.counts, "queued": self.queue.qsize(), "queue_max": self.queue.maxsize,
                "chunk_point_bytes_max": self.max_chunk_bytes, "chunk_messages_max": self.max_messages,
                "drain_complete": self.closing and not self.thread.is_alive(), "index_error": self.index_error,
                "recording_trial": self.trial_id}


class LidarSource:
    """Owned bridge child (hardware or synthetic) + one reader thread. Never a robot client."""

    def __init__(self, command, output, *, monotonic=time.monotonic, snapshot_hz=5.0, env=None):
        self.command = [str(c) for c in command]
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.monotonic = monotonic
        self.state = LidarState(monotonic=monotonic)
        self.recorder = None
        self.lock = threading.Lock()
        self.snapshot_period = 1.0 / snapshot_hz
        self.snapshot_value, self.views = None, {}
        self.last_snapshot = -math.inf
        self.hello = self.stats = self.bye = None
        self.error = None
        self.events = deque(maxlen=256)
        self.events_lost = 0
        self.process = self.reader = self.log = None
        self.closed = False
        self.env = env

    def start(self):
        self.log = (self.output / "bridge.log").open("w")
        root = Path(__file__).resolve().parents[3]
        env = {**os.environ, "PYTHONPATH": str(root / "src"), **(self.env or {})}
        self.spawn_monotonic = self.monotonic()
        self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.log, env=env)
        self.reader = threading.Thread(target=self._read, daemon=True, name="lidar-reader")
        self.reader.start()

    def _event(self, kind, **fields):
        with self.lock:
            if len(self.events) == self.events.maxlen:
                self.events_lost += 1
            self.events.append({"event": "lidar_" + kind, **fields})

    def drain_events(self):
        with self.lock:
            out = list(self.events)
            self.events.clear()
        return out

    def _read(self):
        try:
            stream = self.process.stdout
            while True:
                line = stream.readline()
                if not line:
                    break
                now = self.monotonic()
                header = json.loads(line)
                kind = header.get("type")
                if kind == "hello":
                    if not self.spawn_monotonic <= header["monotonic_s"] <= now:
                        raise RuntimeError("LiDAR bridge monotonic clock is not this host's domain")
                    with self.lock:
                        self.hello = header
                        self.state.hello(header)
                    self._event("bridge_hello", hello=header)
                elif kind in ("scan", "odom"):
                    xyz = None
                    if kind == "scan":
                        payload = stream.read(header["nbytes"])
                        if len(payload) != header["nbytes"] or header["nbytes"] != 12 * header["n"]:
                            raise RuntimeError("Truncated LiDAR scan payload")
                        xyz = np.frombuffer(payload, "<f4").reshape(-1, 3).copy()
                    if self.hello is None:
                        raise RuntimeError("LiDAR data before bridge hello")
                    with self.lock:
                        if kind == "scan":
                            change = self.state.scan(header, xyz, now)
                        else:
                            change = self.state.odom(header, now)
                            self.state.after_odom()
                        epoch = self.state.epoch_label()
                        recorder = self.recorder
                    if change:
                        self._event(change, epoch=epoch, stamp=header["stamp"])
                    if recorder is not None:
                        recorder.offer(kind, header, xyz, epoch, now)
                    if now - self.last_snapshot >= self.snapshot_period:
                        self._snapshot(now)
                elif kind in ("stats", "bye", "error"):
                    with self.lock:
                        if kind == "bye":
                            self.bye = header
                        else:
                            self.stats = header
                    if kind == "error":
                        self._event("bridge_error", error=header.get("error"))
        except Exception as error:  # surfaced to the app, never silent
            self.error = repr(error)
            self._event("reader_error", error=self.error)

    def _snapshot(self, now):
        with self.lock:
            snap, views = self.state.snapshot(now)
            self.snapshot_value, self.views, self.last_snapshot = snap, views, now

    def snapshot(self, now):
        """Thread-safe status. Rebuilt when older than the snapshot period, so pose validity
        and ages keep advancing (to UNKNOWN) even when the bridge has gone silent."""
        if self.hello is not None and now - self.last_snapshot >= self.snapshot_period:
            self._snapshot(now)
        with self.lock:
            snap = None if self.snapshot_value is None else json.loads(json.dumps(self.snapshot_value))
            latest = self.state.latest
            stats = self.stats
        if snap is not None and latest is not None and snap["latest_scan"]:
            snap["latest_scan"]["host_age_s"] = now - latest["host_mono"]
        return {
            "access": None if self.hello is None else self.hello.get("access"),
            "source": None if self.hello is None else self.hello.get("source"),
            "bridge_alive": self.alive,
            "error": self.error,
            "ended": self.bye is not None,
            "bridge_stats": stats,
            "events_lost": self.events_lost,
            "state": snap,
            "recording": None if self.recorder is None else self.recorder.status(),
        }

    def view(self, name):
        with self.lock:
            return self.views.get(name)

    @property
    def alive(self):
        return self.process is not None and self.process.poll() is None

    def close(self, timeout=3.0):
        """EOF on stdin, then terminate/kill. Sends nothing to the robot."""
        if self.closed:
            return
        self.closed = True
        if self.process is None:
            return
        try:
            self.process.stdin.close()
        except OSError:
            pass
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=timeout)
        self.reader.join(timeout=timeout)
        self.process.stdout.close()
        self.log.close()
