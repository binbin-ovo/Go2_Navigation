"""Read-only Go2 front-camera bridge: the advisory app's hardware-environment child.

Runs the existing ``Go2CameraReceiver`` (one multicast *read* socket, PyAV decode) in
the hardware Python and streams decoded RGB frames to the parent over stdout. No DDS,
SDK client, RPC, DataWriter or outgoing datagram is created; nothing is sent to the
robot. The parent (core environment) owns observation conversion and the session.

Protocol on stdout: one JSON header line per message; ``frame`` headers are followed
by exactly ``nbytes`` raw uint8 RGB bytes. The bridge exits when stdin closes.
"""

import argparse
import json
from queue import Empty, Full, Queue
import sys
import threading
import time
import uuid

from go2_perception.adapters.go2.receivers import Go2CameraReceiver


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host-ip", default="192.168.123.222")
    p.add_argument("--group", default="230.1.1.1")
    p.add_argument("--port", type=int, default=1720)
    p.add_argument("--stride", type=int, default=2, help="Forward every Nth decoded frame")
    p.add_argument("--queue", type=int, default=2)
    p.add_argument("--stats-interval-s", type=float, default=1.0)
    return p


class Bridge:
    def __init__(self, out, stride, capacity):
        if not 1 <= stride <= 8 or not 1 <= capacity <= 8:
            raise ValueError("Stride 1..8 and queue 1..8 required")
        self.out, self.stride = out, stride
        self.lock = threading.Lock()
        self.queue = Queue(maxsize=capacity)
        self.bridge_id = uuid.uuid4().hex[:12]
        self.decoded = self.forwarded = self.decimated = self.dropped = 0
        self.closed = threading.Event()

    def write(self, header, payload=b""):
        with self.lock:
            self.out.write((json.dumps(header, allow_nan=False) + "\n").encode())
            if payload:
                self.out.write(payload)
            self.out.flush()

    def on_frame(self, frame):
        # Called on the receiver thread; must never block the socket loop.
        mono0 = time.monotonic()
        wall = time.time()
        mono1 = time.monotonic()
        self.decoded += 1
        if (self.decoded - 1) % self.stride:
            self.decimated += 1
            return
        self.forwarded += 1
        image = frame.image[:, :, ::-1].copy()  # decoder BGR -> RGB, once, here
        mono = 0.5 * (mono0 + mono1)
        header = {
            "type": "frame",
            "bridge_id": self.bridge_id,
            "seq": self.forwarded,
            "decoded_index": self.decoded,
            "rtp_timestamp": int(frame.rtp_timestamp),
            "complete": bool(frame.complete),
            "host_recv_first_wall_s": frame.host_recv_first,
            "host_recv_last_wall_s": frame.host_recv_last,
            "decoded_wall_s": wall,
            "decoded_monotonic_s": mono,
            "wall_minus_monotonic_s": wall - mono,
            "conversion_uncertainty_s": 0.5 * (mono1 - mono0),
            "shape": list(image.shape),
            "dtype": "uint8",
            "channel_order": "RGB",
            "nbytes": image.nbytes,
        }
        try:
            self.queue.put_nowait((header, image))
        except Full:
            self.dropped += 1  # seq gap tells the parent; counted here too

    def writer(self):
        while not self.closed.is_set():
            try:
                header, image = self.queue.get(timeout=0.2)
            except Empty:
                continue
            header["write_monotonic_s"] = time.monotonic()
            try:
                self.write(header, image.tobytes())
            except (BrokenPipeError, OSError):
                self.closed.set()


def main(argv=None):
    a = parser().parse_args(argv)
    bridge = Bridge(sys.stdout.buffer, a.stride, a.queue)
    receiver = Go2CameraReceiver(
        iface_ip=a.host_ip, group=a.group, port=a.port, on_frame=bridge.on_frame
    )

    def stats(kind):
        r, asm = receiver, receiver.assembler
        return {
            "type": kind,
            "bridge_id": bridge.bridge_id,
            "monotonic_s": time.monotonic(),
            "decoded": bridge.decoded,
            "forwarded": bridge.forwarded,
            "decimated_by_stride": bridge.decimated,
            "dropped_bridge_queue": bridge.dropped,
            "receiver_frames": r.frames,
            "rtp_packets": asm.packets,
            "rtp_lost": asm.lost_packets,
            "rtp_reordered": asm.reordered_packets,
            "incomplete_units": r.incomplete_units,
            "dropped_until_keyframe": r.dropped_until_keyframe,
            "socket_rejoins": r.reconnects,
            "decoder_errors": r.decoder_errors,
            "senders": sorted(r.senders),
            "rcvbuf_bytes": r.rcvbuf_bytes,
            "receiver_alive": r.is_alive(),
            "receiver_exception": None if r.exception is None else repr(r.exception),
            # Evidence for the read-only claim: no SDK/DDS/control module is loaded.
            "robot_client_modules_loaded": sorted(
                m for m in sys.modules if m.startswith(("unitree", "cyclonedds", "go2_control"))
            ),
        }

    bridge.write(
        {
            "type": "hello",
            "bridge_id": bridge.bridge_id,
            "monotonic_s": time.monotonic(),
            "wall_s": time.time(),
            "stride": a.stride,
            "queue": a.queue,
            "group": a.group,
            "port": a.port,
            "host_ip": a.host_ip,
            "python": sys.version.split()[0],
            "access": "multicast_read_socket_only_no_dds_no_commands",
        }
    )
    threading.Thread(
        target=lambda: (sys.stdin.read(), bridge.closed.set()), daemon=True
    ).start()
    writer = threading.Thread(target=bridge.writer, daemon=True)
    writer.start()
    receiver.start()
    try:
        while not bridge.closed.wait(a.stats_interval_s):
            if not receiver.is_alive():
                break
            bridge.write(stats("stats"))
    except (BrokenPipeError, OSError, KeyboardInterrupt):
        pass
    finally:
        bridge.closed.set()
        receiver.stop()
        receiver.join(timeout=2)
        writer.join(timeout=1)
        try:
            bridge.write(stats("bye"))
        except (BrokenPipeError, OSError):
            pass
    return 0 if receiver.exception is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
