"""Upper bound on Go2 front-camera exposure-to-receipt latency (read-only; hardware environment).

The Mac screen, placed in front of the stationary robot's camera, alternates full-screen black and white
at random 0.4-0.9 s intervals (below 3 Hz). Each switch time is taken *before* the draw call, so the
light can only appear later; the first received frame showing the new state was exposed after the
switch. receipt(first RTP packet) - switch therefore over-estimates that frame's true latency, and display
lag only adds. The observation pipeline estimates exposure as receipt - configured latency, so the extra
age it can miss is max(latency) - configured; that, plus MARGIN_S, is the bound.

Nothing is sent to the robot: one multicast camera socket and one odometry subscriber (for the boot
offset the provider compares against). Output (git-ignored): data/outputs/hw_readiness/capture_latency_*.
"""
import argparse
import json
import os
import random
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from go2_perception.adapters.go2.receivers import Go2CameraReceiver  # noqa: E402

CONFIGURED_LATENCY_S = 0.048  # configs/go2_vln_sensors.yaml, what the pipeline subtracts
MARGIN_S = 0.020
SMALL = (160, 90)


def odom_offset(iface, wait_s=3.0):
    """host_monotonic - robot odom stamp, as hardware/supervised_guards.OdomTrack computes it."""
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.nav_msgs.msg.dds_ import Odometry_
    ChannelFactoryInitialize(0, iface)
    seen = []

    def on_odom(m):
        seen.append(time.monotonic() - (m.header.stamp.sec + m.header.stamp.nanosec * 1e-9))

    sub = ChannelSubscriber("rt/utlidar/robot_odom", Odometry_)
    sub.Init(on_odom, 10)
    time.sleep(wait_s)
    return (min(seen), len(seen)) if seen else (None, 0)


def analyse(frames, flips):
    recv = np.array([f[0] for f in frames])
    small = np.stack([f[1] for f in frames]).astype(np.float32)
    roi = small.std(0) >= np.percentile(small.std(0), 95)  # pixels that follow the screen
    lum = small[:, roi].mean(1)
    lo, hi = np.percentile(lum, 10), np.percentile(lum, 90)
    threshold = (lo + hi) / 2
    rows = []
    for t, white in flips[1:]:  # the first draw has no prior state to change from
        after = np.nonzero((recv > t) & ((lum > threshold) == white))[0]
        before = np.nonzero(recv > t)[0]
        if len(after) == 0 or len(before) == 0:
            continue
        rows.append({"switch_wall_s": t, "to_white": white, "latency_s": float(recv[after[0]] - t),
                     "frames_skipped": int(after[0] - before[0])})
    lat = np.array([r["latency_s"] for r in rows])
    stats = None if len(lat) == 0 else {
        "n": int(len(lat)), "min_s": float(lat.min()), "median_s": float(np.median(lat)),
        "p95_s": float(np.percentile(lat, 95)), "max_s": float(lat.max())}
    return {"contrast": float(hi - lo), "threshold": float(threshold), "roi_pixels": int(roi.sum()),
            "frame_interval_median_s": float(np.median(np.diff(recv))), "stats": stats, "switches": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", default="en9")
    ap.add_argument("--host-ip", default="192.168.123.222")
    ap.add_argument("--flips", type=int, default=40)
    ap.add_argument("--out", type=Path, default=Path(os.environ.get("GO2_WORKSPACE", Path.home() / "Robotics/Go2")) / "evidence/hw_readiness")
    a = ap.parse_args()

    offset, n_odom = odom_offset(a.iface)
    frames, lock = [], threading.Lock()

    def on_frame(frame):
        small = cv2.resize(cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY), SMALL, interpolation=cv2.INTER_AREA)
        with lock:
            frames.append((frame.host_recv_first, small))

    receiver = Go2CameraReceiver(iface_ip=a.host_ip, on_frame=on_frame)
    receiver.start()
    cv2.namedWindow("latency", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("latency", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    black, white = np.zeros((90, 160), np.uint8), np.full((90, 160), 255, np.uint8)
    flips, state, rng = [], False, random.Random(0)
    cv2.imshow("latency", black)
    cv2.waitKey(2000)  # settle on black before the first switch
    for _ in range(a.flips):
        state = not state
        t = time.time()  # before drawing: the light cannot appear earlier than this
        cv2.imshow("latency", white if state else black)
        flips.append((t, state))
        cv2.waitKey(int(1000 * rng.uniform(0.4, 0.9)))
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    receiver._stop_event.set()
    with lock:
        frames = list(frames)
    if len(frames) < 50:
        sys.exit(f"only {len(frames)} camera frames received")

    result = analyse(frames, flips)
    s = result["stats"]
    bound = None if s is None else round(max(0.0, s["max_s"] - CONFIGURED_LATENCY_S) + MARGIN_S, 3)
    record = {"created_wall_s": time.time(), "configured_latency_s": CONFIGURED_LATENCY_S, "margin_s": MARGIN_S,
              "bound_s": bound, "boot_offset_s": offset, "odom_messages": n_odom, "frames": len(frames),
              "receiver": {"incomplete_units": receiver.incomplete_units, "reconnects": receiver.reconnects,
                           "decoder_errors": receiver.decoder_errors}, **result}
    a.out.mkdir(parents=True, exist_ok=True)
    path = a.out / f"capture_latency_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(record, indent=1))
    print(json.dumps({k: record[k] for k in ("bound_s", "boot_offset_s", "frames", "contrast", "roi_pixels",
                                            "frame_interval_median_s", "stats", "receiver")}, indent=1))
    print(path)


if __name__ == "__main__":
    main()
