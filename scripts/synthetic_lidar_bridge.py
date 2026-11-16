"""Synthetic or recorded LiDAR/odometry source speaking the go2_lidar_bridge.py protocol.

For development, tests and REVIEW-mode demos without hardware. Opens no socket and no
DDS participant. ``--recording DIR`` replays a local go2-recording-v1 directory's
``cloud_base`` scans and odometry at their recorded pace with their ORIGINAL robot
stamps; the host receipt is this replay's own receipt time (the recorded receipt is
kept alongside). Without ``--recording`` a deterministic box room is generated in a
``synthetic_robot`` clock that never claims to be a Go2 clock.
"""

import argparse
import json
from pathlib import Path
import sys
import threading
import time
import uuid

import numpy as np


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recording", type=Path)
    p.add_argument("--scan-hz", type=float, default=15.0)
    p.add_argument("--odom-hz", type=float, default=50.0)
    p.add_argument("--seconds", type=float, default=3600.0)
    p.add_argument("--no-odom", action="store_true", help="Pose unknown (accumulation refused)")
    p.add_argument("--reboot-after-s", type=float, help="Restart the synthetic robot clock once")
    p.add_argument("--pause-after-s", type=float, help="Stop sending for --pause-s once")
    p.add_argument("--pause-s", type=float, default=1.5)
    p.add_argument("--stamp-base", type=float, default=1000.0)
    return p


def room_scan(i, rng, n=900):
    """Partial (~190 deg) rotating sweep of a 7 x 6 m box room with one box obstacle."""
    start = (i * 37.0) % 360.0
    az = np.radians(start + rng.uniform(0, 190, n))
    elevation = rng.uniform(-0.45, 0.25, n)
    d = np.stack([np.cos(az), np.sin(az)], 1)
    # Ray to the walls x in [-2, 5], y in [-3, 3].
    with np.errstate(divide="ignore"):
        tx = np.where(d[:, 0] > 0, 5.0 / d[:, 0], -2.0 / d[:, 0])
        ty = np.where(d[:, 1] > 0, 3.0 / d[:, 1], -3.0 / d[:, 1])
    r = np.minimum(np.abs(tx), np.abs(ty))
    # Box obstacle centred (2.0, 0.8), 0.5 m square.
    box = (np.abs(d[:, 1] * 2.0 - d[:, 0] * 0.8) < 0.25 * np.hypot(d[:, 0], d[:, 1])) & (d[:, 0] > 0)
    r = np.where(box & (elevation > -0.3), np.minimum(r, 1.8), r)
    z = np.tan(elevation) * r
    floor = z < -0.33
    r = np.where(floor, -0.33 / np.tan(elevation), r)
    z = np.where(floor, -0.33, np.minimum(z, 1.0))
    xyz = np.stack([d[:, 0] * r, d[:, 1] * r, z], 1)
    keep = np.hypot(xyz[:, 0], xyz[:, 1]) >= 0.3  # cloud_base blind radius
    return xyz[keep].astype("<f4")


def main(argv=None):
    a = parser().parse_args(argv)
    out = sys.stdout.buffer
    lock = threading.Lock()
    closed = threading.Event()
    bridge_id = "synth-" + uuid.uuid4().hex[:8]

    def write(header, payload=b""):
        with lock:
            out.write((json.dumps(header, allow_nan=False) + "\n").encode())
            out.write(payload)
            out.flush()

    def clock():
        mono = time.monotonic()
        return time.time(), mono

    threading.Thread(target=lambda: (sys.stdin.read(), closed.set()), daemon=True).start()
    source = "recording_replay" if a.recording else "synthetic"
    write({"type": "hello", "bridge_id": bridge_id, "monotonic_s": time.monotonic(),
           "wall_s": time.time(), "source": source,
           "recording": None if a.recording is None else str(a.recording),
           "topics": ["rt/utlidar/cloud_base", "rt/utlidar/robot_odom"],
           "robot_clock": "recorded go2_robot stamps (original)" if a.recording
           else "synthetic_robot (not a Go2 clock)",
           "access": "no_socket_no_dds_" + source})
    events = []  # (time offset s, kind, header, payload)
    if a.recording:
        root = a.recording
        scan_dir = root / "lidar" / "utlidar_cloud_base"
        scans = [json.loads(x) for x in (scan_dir / "index.jsonl").read_text().splitlines() if x]
        odom = [json.loads(x) for x in (root / "state/utlidar_robot_odom.jsonl").read_text().splitlines() if x]
        t0 = min([s["host_recv"] for s in scans] + [o["host_recv"] for o in odom])
        for s in scans:
            xyz = np.load(scan_dir / s["file"])[:, :3].astype("<f4")
            tpath = scan_dir / s["file"].replace(".npy", "_time.npy")
            span = float(np.load(tpath).max()) if s.get("has_time") and tpath.exists() else None
            events.append((s["host_recv"] - t0, "scan", {"stamp": s["stamp"], "time_span_s": span,
                           "frame_id": s.get("frame_id", "base_link"),
                           "recorded_host_recv_wall_s": s["host_recv"]}, xyz))
        if not a.no_odom:
            for o in odom:
                events.append((o["host_recv"] - t0, "odom", {
                    k: o[k] for k in ("stamp", "frame_id", "child_frame_id", "position",
                                      "q_xyzw", "linear", "angular")} | {
                    "recorded_host_recv_wall_s": o["host_recv"]}, None))
        events.sort(key=lambda e: e[0])
    seq = {"scan": 0, "odom": 0}
    start = time.monotonic()
    rng = np.random.default_rng(7)
    stamp_base, rebooted, paused = a.stamp_base, False, False
    next_scan, next_odom, i = 0.0, 0.0, 0
    replay_index = 0
    while not closed.is_set():
        elapsed = time.monotonic() - start
        if elapsed > a.seconds:
            break
        if a.pause_after_s is not None and not paused and elapsed > a.pause_after_s:
            paused = True
            closed.wait(a.pause_s)
            continue
        if a.recording:
            if replay_index >= len(events):
                break
            due, kind, fields, xyz = events[replay_index]
            if elapsed < due:
                time.sleep(min(0.005, due - elapsed))
                continue
            replay_index += 1
        else:
            if a.reboot_after_s is not None and not rebooted and elapsed > a.reboot_after_s:
                rebooted, stamp_base = True, a.stamp_base - 900.0  # robot clock restarts
                start_reboot = elapsed
            robot_t = stamp_base + elapsed - (start_reboot if rebooted else 0.0)
            if not a.no_odom and elapsed >= next_odom:
                next_odom += 1.0 / a.odom_hz
                kind, xyz = "odom", None
                fields = {"stamp": robot_t, "frame_id": "odom", "child_frame_id": "base_link",
                          "position": [0.0, 0.0, 0.31], "q_xyzw": [0.0, 0.0, 0.0, 1.0],
                          "linear": [0.0, 0.0, 0.0], "angular": [0.0, 0.0, 0.0]}
            elif elapsed >= next_scan:
                next_scan += 1.0 / a.scan_hz
                i += 1
                kind, xyz = "scan", room_scan(i, rng)
                fields = {"stamp": robot_t - 0.07, "time_span_s": 0.06, "frame_id": "base_link"}
            else:
                time.sleep(0.002)
                continue
        wall, mono = clock()
        seq[kind] += 1
        header = {"type": kind, "bridge_id": bridge_id, "seq": seq[kind],
                  "host_recv_wall_s": wall, "wall_minus_monotonic_s": wall - mono,
                  "conversion_uncertainty_s": 1e-6, "write_monotonic_s": mono, **fields}
        if kind == "scan":
            header.update(topic="rt/utlidar/cloud_base", n=int(len(xyz)), dtype="<f4",
                          fields=["x", "y", "z"], nbytes=int(xyz.nbytes))
            try:
                write(header, xyz.tobytes())
            except (BrokenPipeError, OSError):
                break
        else:
            try:
                write(header)
            except (BrokenPipeError, OSError):
                break
    try:
        write({"type": "bye", "bridge_id": bridge_id, "monotonic_s": time.monotonic(),
               "scans": seq["scan"], "odom": seq["odom"]})
    except (BrokenPipeError, OSError):
        pass
    closed.wait()  # like the hardware bridge: exit on stdin EOF


if __name__ == "__main__":
    main()
