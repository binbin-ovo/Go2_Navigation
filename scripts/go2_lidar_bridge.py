"""Read-only Go2 LiDAR/odometry bridge: the interactive review tool's DDS child.

Runs the existing ``Go2DdsReceiver`` in the hardware Python (3.10, Cyclone DDS) with
subscribers only: ``rt/utlidar/cloud_base`` (single scans in base_link) and the
receiver's state topics, of which only ``rt/utlidar/robot_odom`` is forwarded.
``cloud_deskewed`` is never subscribed, so nothing is accumulated twice. No
publisher, DataWriter, RPC/sport/avoidance client or outgoing datagram is created;
nothing is sent to the robot.

Protocol on stdout (same style as go2_camera_bridge.py): one JSON header line per
message; ``scan`` headers are followed by exactly ``nbytes`` little-endian float32
x,y,z bytes. Robot stamps stay in the robot clock; host receipt stays host wall
time, with a same-host wall-minus-monotonic sample. The bridge exits on stdin EOF.
"""

import argparse
import json
import os
from queue import Empty, Full, Queue
import sys
import threading
import time
import uuid

import numpy as np

from go2_perception.adapters.go2.receivers import ODOM_TOPIC, Go2DdsReceiver

SCAN_TOPIC = "rt/utlidar/cloud_base"


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--iface", default="en9", help="DDS network interface (Mac USB adapter: en9)")
    p.add_argument("--scan-queue", type=int, default=8)
    p.add_argument("--odom-queue", type=int, default=256)
    p.add_argument("--stats-interval-s", type=float, default=1.0)
    return p


class Bridge:
    def __init__(self, out, scan_capacity, odom_capacity):
        if not 1 <= scan_capacity <= 32 or not 1 <= odom_capacity <= 1024:
            raise ValueError("Scan queue 1..32 and odom queue 1..1024 required")
        self.out = out
        self.lock = threading.Lock()
        self.queue = Queue(maxsize=scan_capacity + odom_capacity)
        self.scan_capacity, self.odom_capacity = scan_capacity, odom_capacity
        self.queued = {"scan": 0, "odom": 0}
        self.bridge_id = uuid.uuid4().hex[:12]
        self.counts = {"scans": 0, "odom": 0, "scans_dropped": 0, "odom_dropped": 0,
                       "other_state_ignored": 0}
        self.seq = {"scan": 0, "odom": 0}
        self.closed = threading.Event()

    def write(self, header, payload=b""):
        with self.lock:
            self.out.write((json.dumps(header, allow_nan=False) + "\n").encode())
            if payload:
                self.out.write(payload)
            self.out.flush()

    def _clock(self):
        mono0 = time.monotonic()
        wall = time.time()
        mono1 = time.monotonic()
        return wall, 0.5 * (mono0 + mono1), 0.5 * (mono1 - mono0)

    def _enqueue(self, kind, header, payload=b""):
        # DDS callback threads must never block; loss is counted and visible as seq gaps.
        with self.lock:
            full = self.queued[kind] >= (self.scan_capacity if kind == "scan" else self.odom_capacity)
            if not full:
                self.queued[kind] += 1
        if full:
            self.counts["scans_dropped" if kind == "scan" else "odom_dropped"] += 1
            return
        try:
            self.queue.put_nowait((kind, header, payload))
        except Full:
            with self.lock:
                self.queued[kind] -= 1
            self.counts["scans_dropped" if kind == "scan" else "odom_dropped"] += 1

    def on_lidar(self, frame):
        if frame.topic != SCAN_TOPIC:
            return
        wall, mono, conversion = self._clock()
        xyz = np.ascontiguousarray(frame.xyz[:, :3], dtype="<f4")
        span = None if frame.time is None or not len(frame.time) else float(np.max(frame.time))
        self.counts["scans"] += 1
        self.seq["scan"] += 1
        header = {
            "type": "scan", "bridge_id": self.bridge_id, "seq": self.seq["scan"],
            "topic": frame.topic, "frame_id": frame.frame_id,
            "stamp": frame.stamp, "time_span_s": span,
            "host_recv_wall_s": frame.host_recv, "wall_minus_monotonic_s": wall - mono,
            "conversion_uncertainty_s": conversion, "n": int(len(xyz)),
            "dtype": "<f4", "fields": ["x", "y", "z"], "nbytes": int(xyz.nbytes),
        }
        self._enqueue("scan", header, xyz.tobytes())

    def on_state(self, topic, record):
        if topic != ODOM_TOPIC:
            self.counts["other_state_ignored"] += 1
            return
        wall, mono, conversion = self._clock()
        self.counts["odom"] += 1
        self.seq["odom"] += 1
        header = {
            "type": "odom", "bridge_id": self.bridge_id, "seq": self.seq["odom"],
            "stamp": record["stamp"], "host_recv_wall_s": record["host_recv"],
            "wall_minus_monotonic_s": wall - mono, "conversion_uncertainty_s": conversion,
            "frame_id": record["frame_id"], "child_frame_id": record["child_frame_id"],
            "position": record["position"], "q_xyzw": record["q_xyzw"],
            "linear": record["linear"], "angular": record["angular"],
        }
        self._enqueue("odom", header)

    def writer(self):
        while not self.closed.is_set():
            try:
                kind, header, payload = self.queue.get(timeout=0.2)
            except Empty:
                continue
            with self.lock:
                self.queued[kind] -= 1
            header["write_monotonic_s"] = time.monotonic()
            try:
                self.write(header, payload)
            except (BrokenPipeError, OSError, ValueError):
                self.closed.set()


def main(argv=None):
    a = parser().parse_args(argv)
    bridge = Bridge(sys.stdout.buffer, a.scan_queue, a.odom_queue)
    receiver = Go2DdsReceiver(a.iface, [SCAN_TOPIC], bridge.on_lidar, bridge.on_state)

    def stats(kind):
        return {
            "type": kind, "bridge_id": bridge.bridge_id, "monotonic_s": time.monotonic(),
            **bridge.counts, "decode_errors": receiver.decode_errors,
            "last_recv_wall_s": dict(receiver.last_recv),
            # DDS is required here; evidence that only the receiver's subscribers exist is
            # the static test of this file and receivers.py (no publisher/client names).
            "access": "dds_subscribers_only_cloud_base_and_state_no_publishers",
            "control_modules_loaded": sorted(
                m for m in sys.modules if m.startswith(("go2_control", "go2_hl_control"))
                or m.endswith(("sport_client", "obstacles_avoid_client"))
            ),
        }

    bridge.write({
        "type": "hello", "bridge_id": bridge.bridge_id, "monotonic_s": time.monotonic(),
        "wall_s": time.time(), "source": "go2_dds_read_only", "iface": a.iface,
        "topics": [SCAN_TOPIC, ODOM_TOPIC], "python": sys.version.split()[0],
        "robot_clock": "go2_robot header.stamp (resets each boot)",
        "host_clock": "host wall receipt (time.time) + wall-minus-monotonic sample",
        "access": "dds_subscribers_only_cloud_base_and_state_no_publishers",
    })
    threading.Thread(target=lambda: (sys.stdin.read(), bridge.closed.set()), daemon=True).start()
    writer = threading.Thread(target=bridge.writer, daemon=True)
    writer.start()
    code = 0
    try:
        receiver.start()
        while not bridge.closed.wait(a.stats_interval_s):
            bridge.write(stats("stats"))
    except (BrokenPipeError, OSError, KeyboardInterrupt):
        pass
    except Exception as error:  # reported to the parent, never silent
        code = 1
        try:
            bridge.write({"type": "error", "bridge_id": bridge.bridge_id, "error": repr(error)})
        except (BrokenPipeError, OSError):
            pass
    finally:
        bridge.closed.set()
        try:
            receiver.stop()
        except Exception:
            pass
        writer.join(timeout=1)
        try:
            bridge.write(stats("bye"))
        except (BrokenPipeError, OSError, ValueError):
            pass
    sys.stdout.flush()
    # Cyclone DDS teardown can hang at interpreter exit (docs/GO2_CONNECTION.md).
    os._exit(code)


if __name__ == "__main__":
    main()
