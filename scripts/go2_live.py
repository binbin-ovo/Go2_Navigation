#!/usr/bin/env python
"""Live read-only perception: latest RGB frame -> YOLO; latest cloud_base -> LiDAR processor.

Usage: .venv/bin/python scripts/go2_live.py --seconds 30 [--display] [--output DIR]
Sends no commands. Runs on the newest data and skips frames instead of queuing latency.
RGB and LiDAR results stay separate: fusion is blocked until a real calibration exists.
"""
import argparse
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from go2_perception.adapters.go2.pointcloud import to_lidar_message
from go2_perception.adapters.go2.receivers import (
    ODOM_TOPIC,
    Go2CameraReceiver,
    Go2DdsReceiver,
)
from go2_perception.adapters.go2.window import PoseBuffer, SlidingWindowAccumulator
from go2_perception.io.config import load_config
from go2_perception.io.sensor_interface import LidarMessage, process_lidar
from go2_perception.lidar.processor import LidarProcessor
from go2_perception.lidar.visualization import save_visualization
from go2_perception.rgb.detector import RGBDetector
from go2_perception.rgb.visualization import draw_detections

TOPIC = "rt/utlidar/cloud_base"


def main():
    cfg = load_config("configs/go2_sensors.yaml")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seconds", type=float, default=30)
    p.add_argument("--display", action="store_true")
    p.add_argument("--output", default="data/outputs/live")
    p.add_argument("--rgb-config", default="configs/go2_rgb.yaml")
    p.add_argument("--lidar-config", default="configs/go2_lidar.yaml")
    p.add_argument("--stale-after", type=float, default=1.0)
    p.add_argument("--window-s", type=float, default=0.3,
                   help="ego-compensated window; 0 processes single frames, which cover about 190 deg")
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)

    latest = {"rgb": None, "cloud": None, "q": None}
    lock = threading.Lock()
    poses = PoseBuffer(max_age_s=5.0)
    window = SlidingWindowAccumulator(poses, a.window_s) if a.window_s else None

    def on_frame(f):
        with lock:
            latest["rgb"] = f

    def on_lidar(f):
        with lock:
            latest["cloud"] = f

    def on_state(topic, record):
        if topic == ODOM_TOPIC:
            with lock:
                latest["q"] = record["q_xyzw"]
                poses.add(record["stamp"], record["position"], record["q_xyzw"])

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    detector = RGBDetector(**load_config(a.rgb_config))
    processor = LidarProcessor(**load_config(a.lidar_config))
    dds = Go2DdsReceiver(cfg["network"]["iface"], [TOPIC], on_lidar, on_state)
    dds.start()
    camera = Go2CameraReceiver(
        cfg["network"]["host_ip"], cfg["camera"]["multicast_group"], cfg["camera"]["port"], on_frame
    )
    camera.start()

    rgb_ms, lidar_ms, ages, clusters, stale_events = [], [], [], [], 0
    last_warn, seen_rgb, seen_cloud, saved = 0.0, None, None, 0
    t0, cpu0 = time.time(), sum(os.times()[:2])
    while time.time() - t0 < a.seconds and not stop.is_set():
        with lock:
            frame, cloud, q = latest["rgb"], latest["cloud"], latest["q"]
        now = time.time()
        newest = max(frame.host_recv_last if frame else 0, cloud.host_recv if cloud else 0)
        if now - t0 > 3 and now - newest > a.stale_after and now - last_warn > 2:
            stale_events += 1
            last_warn = now
            print(f"[{now - t0:6.1f}s] WAITING: no sensor data for {now - newest:.1f}s", flush=True)
        worked = False
        if frame is not None and frame is not seen_rgb:
            seen_rgb, worked = frame, True
            result = detector.detect(
                frame.image, timestamp=frame.host_recv_first,
                frame_id="go2_front_camera_unrectified", clock_domain="host_receipt",
            )
            rgb_ms.append(result.latency_ms)
            ages.append((time.time() - frame.host_recv_last) * 1e3)
            canvas = draw_detections(frame.image, result)
            if a.display:
                cv2.imshow("Go2 RGB + YOLO", canvas)
                if cv2.waitKey(1) == 27:
                    stop.set()
            if len(rgb_ms) % 60 == 1:
                cv2.imwrite(str(out / "rgb_latest.jpg"), canvas)
        if cloud is not None and cloud is not seen_cloud and q is not None:
            seen_cloud, worked = cloud, True
            message = to_lidar_message(cloud, q)
            with lock:
                accepted = window is not None and window.add(cloud.xyz, cloud.acquisition_midpoint)
                merged = window.merged() if accepted else None
            if merged is not None:
                level = to_lidar_message(
                    type(cloud)(cloud.topic, merged[1], cloud.frame_id, cloud.host_recv, merged[0]), q
                )
                message = LidarMessage(level.points, merged[1], level.frame_id, level.clock_domain)
            elif window is not None:
                continue  # no bracketing pose yet; retry with the next frame, not uncompensated
            result = process_lidar(processor, message)
            lidar_ms.append(result.latency_ms)
            clusters.append(len(result.obstacles))
            if saved < 3 and len(lidar_ms) % 60 == 1:
                save_visualization(result.points, result, out / "lidar_latest")
                saved += 1
        if not worked:
            time.sleep(0.002)
    wall, cpu = time.time() - t0, sum(os.times()[:2]) - cpu0
    camera.stop()
    camera.join(timeout=3)
    dds.stop()
    if a.display:
        cv2.destroyAllWindows()
    report = {
        "wall_s": round(wall, 2),
        "rgb_inferences": len(rgb_ms),
        "rgb_inference_hz": round(len(rgb_ms) / wall, 2),
        "rgb_latency_ms_median": round(float(np.median(rgb_ms[1:] or [0])), 2),
        "rgb_frame_age_at_result_ms_median": round(float(np.median(ages or [0])), 2),
        "lidar_results": len(lidar_ms),
        "lidar_result_hz": round(len(lidar_ms) / wall, 2),
        "lidar_latency_ms_median": round(float(np.median(lidar_ms or [0])), 2),
        "clusters_median": float(np.median(clusters or [0])),
        "stale_warnings": stale_events,
        "camera_rejoins": camera.reconnects,
        "window_s": a.window_s,
        "frames_refused_no_pose": window.refused_no_pose if window else 0,
        "process_cpu_percent_of_one_core": round(100 * cpu / wall, 1),
    }
    (out / "live_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    sys.stdout.flush()
    os._exit(0)  # cyclonedds may hang interpreter teardown


if __name__ == "__main__":
    main()
