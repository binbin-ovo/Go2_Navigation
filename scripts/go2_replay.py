#!/usr/bin/env python
"""Run the existing YOLO and LiDAR processors offline on a recording; no robot needed.

Usage: .venv/bin/python scripts/go2_replay.py --recording data/recordings/sample
Writes detections and clusters as JSON plus annotated images under <recording>/perception.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from go2_perception.adapters.go2.pointcloud import to_lidar_message
from go2_perception.adapters.go2.recording import RecordingReader
from go2_perception.adapters.go2.window import PoseBuffer, SlidingWindowAccumulator
from go2_perception.io.config import load_config
from go2_perception.io.image_loader import load_image
from go2_perception.io.sensor_interface import LidarMessage, process_lidar
from go2_perception.io.serialization import jsonable
from go2_perception.lidar.processor import LidarProcessor
from go2_perception.lidar.visualization import save_visualization
from go2_perception.rgb.detector import RGBDetector
from go2_perception.rgb.visualization import draw_detections

ODOM = "rt/utlidar/robot_odom"
TOPIC = "rt/utlidar/cloud_base"


def nearest_attitude(odom, t):
    stamps = np.array([r["stamp"] for r in odom])
    i = int(np.argmin(np.abs(stamps - t)))
    return odom[i]["q_xyzw"] if abs(stamps[i] - t) < 0.05 else None


def run_rgb(reader, out, args):
    records = reader.rgb_records()[:: args.rgb_stride]
    if not records:
        return {"frames": 0}
    config = load_config(args.rgb_config)
    if args.device:
        config["device"] = args.device
    detector = RGBDetector(**config)
    results, latencies, classes = [], [], {}
    frames_with_detections = 0
    (out / "rgb").mkdir(parents=True, exist_ok=True)
    for i, record in enumerate(records):
        image = load_image(reader.rgb_path(record))
        # No RGBMessage: its K is mandatory and the camera has no measured intrinsics.
        result = detector.detect(
            image,
            timestamp=record["host_recv_first"],
            frame_id=record["frame_id"],
            clock_domain="host_receipt",
        )
        latencies.append(result.latency_ms)
        results.append({"file": record["file"], "rtp_timestamp": record["rtp_timestamp"],
                        **jsonable(result)})
        frames_with_detections += bool(result.detections)
        for d in result.detections:
            classes[d.class_name] = classes.get(d.class_name, 0) + 1
        if i % args.save_every == 0 or (result.detections and frames_with_detections <= 5):
            cv2.imwrite(str(out / "rgb" / record["file"]), draw_detections(image, result))
    (out / "rgb_detections.json").write_text(json.dumps(results, indent=1) + "\n")
    warm = latencies[1:] or latencies
    return {
        "frames": len(records),
        "frames_with_detections": frames_with_detections,
        "class_counts": classes,
        "latency_ms_median_excl_first": round(float(np.median(warm)), 2),
        "latency_ms_first": round(latencies[0], 2),
    }


def run_lidar(reader, out, args):
    frames = reader.lidar_frames(TOPIC)[:: args.lidar_stride]
    odom = reader.state(ODOM)
    processor = LidarProcessor(**load_config(args.lidar_config))
    poses = PoseBuffer(max_age_s=1e9)
    for r in odom:
        poses.add(r["stamp"], r["position"], r["q_xyzw"])
    accumulator = SlidingWindowAccumulator(poses, args.window_s) if args.window_s else None
    rows, clusters, ground, latencies, unlevelled = [], [], [], [], 0
    for i, replay in enumerate(frames):
        frame = replay.load()
        q = nearest_attitude(odom, frame.acquisition_midpoint) if odom else None
        unlevelled += q is None
        message = to_lidar_message(frame, q)
        points = message.points
        if accumulator:
            if not accumulator.add(frame.xyz, frame.acquisition_midpoint):
                continue
            merged, t, _, _ = accumulator.merged()
            q = nearest_attitude(odom, t)
            merged, frame_id = to_lidar_message(
                type(frame)(frame.topic, t, frame.frame_id, frame.host_recv, merged), q
            ).points, "base_link_level"
            message = LidarMessage(merged, t, frame_id, message.clock_domain)
            points = merged
        result = process_lidar(processor, message)
        latencies.append(result.latency_ms)
        clusters.append(len(result.obstacles))
        ground.append(len(result.ground))
        rows.append(
            {
                "file": replay.record["file"],
                "timestamp": result.timestamp,
                "frame_id": result.frame_id,
                "clock_domain": result.clock_domain,
                "input_points": len(points),
                "processed_points": len(result.points),
                "ground_points": len(result.ground),
                "unclassified_points": len(result.unclassified),
                "latency_ms": round(result.latency_ms, 2),
                "obstacles": [
                    {
                        "cluster_id": o.cluster_id,
                        "points": len(o.points),
                        "center": o.center.round(3).tolist(),
                        "bbox_min": o.bbox_min.round(3).tolist(),
                        "bbox_max": o.bbox_max.round(3).tolist(),
                    }
                    for o in result.obstacles
                ],
            }
        )
        if i % args.save_every == 0:
            save_visualization(points, result, out / "lidar" / replay.record["file"][:-4])
    (out / "lidar_obstacles.json").write_text(json.dumps(rows, indent=1) + "\n")
    return {
        "frames_processed": len(rows),
        "frames_input": len(frames),
        "window_s": args.window_s,
        "frames_without_attitude": int(unlevelled),
        "frames_refused_no_pose": accumulator.refused_no_pose if accumulator else 0,
        "clusters_median_min_max": [
            float(np.median(clusters)), int(min(clusters)), int(max(clusters))
        ] if clusters else None,
        "ground_points_median": float(np.median(ground)) if ground else None,
        "latency_ms_median": round(float(np.median(latencies)), 2) if latencies else None,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recording", required=True)
    p.add_argument("--output")
    p.add_argument("--rgb-config", default="configs/go2_rgb.yaml")
    p.add_argument("--lidar-config", default="configs/go2_lidar.yaml")
    p.add_argument("--device", help="override the YOLO device, for example cpu")
    p.add_argument("--rgb-stride", type=int, default=1)
    p.add_argument("--lidar-stride", type=int, default=1)
    p.add_argument("--save-every", type=int, default=30)
    p.add_argument("--window-s", type=float, default=0,
                   help="ego-compensated window; 0 processes single frames")
    p.add_argument("--no-rgb", action="store_true")
    p.add_argument("--no-lidar", action="store_true")
    a = p.parse_args()
    reader = RecordingReader(a.recording)
    out = Path(a.output or Path(a.recording) / "perception")
    out.mkdir(parents=True, exist_ok=True)
    report = {"recording": str(a.recording)}
    if not a.no_rgb:
        report["rgb"] = run_rgb(reader, out, a)
    if not a.no_lidar:
        report["lidar"] = run_lidar(reader, out, a)
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
