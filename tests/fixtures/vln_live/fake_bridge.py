"""Test double for scripts/go2_camera_bridge.py: same protocol, no socket, no robot.

Usage: fake_bridge.py PLAN_JSON. PLAN: {"frames": [{"seq", "dt", "offset_jump",
"pause_after_s", "shape"}...], "ignore_eof": bool, "fps": float}.
"""

import json
import sys
import threading
import time

import numpy as np

plan = json.loads(sys.argv[-1])
out = sys.stdout.buffer
closed = threading.Event()


def write(header, payload=b""):
    out.write((json.dumps(header) + "\n").encode())
    out.write(payload)
    out.flush()


if not plan.get("ignore_eof"):
    threading.Thread(target=lambda: (sys.stdin.read(), closed.set()), daemon=True).start()
write({"type": "hello", "bridge_id": plan.get("bridge_id", "fakebridge01"),
       "monotonic_s": time.monotonic(), "wall_s": time.time(), "stride": 1})
offset = time.time() - time.monotonic()
for i, f in enumerate(plan["frames"]):
    if closed.is_set():
        break
    time.sleep(f.get("dt", 1 / plan.get("fps", 10)))
    offset += f.get("offset_jump", 0.0)
    h, w = f.get("shape", [720, 1280])
    image = np.zeros((h, w, 3), np.uint8)
    image[..., 0] = 200  # pure red in RGB order
    image[0, 0] = (i % 256, 1, 2)
    mono = time.monotonic()
    wall = mono + offset
    write({"type": "frame", "bridge_id": plan.get("bridge_id", "fakebridge01"),
           "seq": f.get("seq", i + 1), "decoded_index": i + 1, "rtp_timestamp": 9000 * i,
           "complete": True, "host_recv_first_wall_s": wall - 0.012,
           "host_recv_last_wall_s": wall - 0.010, "decoded_wall_s": wall,
           "decoded_monotonic_s": mono, "wall_minus_monotonic_s": offset,
           "conversion_uncertainty_s": 1e-6, "shape": [h, w, 3], "dtype": "uint8",
           "channel_order": "RGB", "nbytes": image.nbytes, "write_monotonic_s": time.monotonic()},
          image.tobytes())
    if f.get("pause_after_s"):
        closed.wait(f["pause_after_s"])
write({"type": "stats", "bridge_id": "fakebridge01", "monotonic_s": time.monotonic()})
while not closed.is_set():
    time.sleep(0.05)
write({"type": "bye", "bridge_id": "fakebridge01", "monotonic_s": time.monotonic()})
