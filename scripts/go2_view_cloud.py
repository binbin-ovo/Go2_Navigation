#!/usr/bin/env python
"""Show a recorded LiDAR frame with clusters and boxes in Open3D (RViz2 is not installed).

Usage: .venv/bin/python scripts/go2_view_cloud.py --recording R [--index 0] [--window-s 0.3]
       [--screenshot out.png]   # renders once, saves the image, and closes
"""
import argparse

import numpy as np
import open3d as o3d

from go2_perception.adapters.go2.pointcloud import to_lidar_message
from go2_perception.adapters.go2.recording import RecordingReader
from go2_perception.adapters.go2.window import PoseBuffer, SlidingWindowAccumulator
from go2_perception.io.config import load_config
from go2_perception.io.sensor_interface import LidarMessage, process_lidar
from go2_perception.lidar.processor import LidarProcessor

PALETTE = np.array([[1, 0.3, 0.1], [0.2, 0.5, 1], [1, 0.7, 0.1], [0.8, 0.2, 0.8], [0.1, 0.8, 0.8]])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recording", required=True)
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--topic", default="rt/utlidar/cloud_base")
    p.add_argument("--window-s", type=float, default=0)
    p.add_argument("--config", default="configs/go2_lidar.yaml")
    p.add_argument("--screenshot")
    a = p.parse_args()
    reader = RecordingReader(a.recording)
    frames = reader.lidar_frames(a.topic)
    odom = reader.state("rt/utlidar/robot_odom")
    stamps = np.array([r["stamp"] for r in odom])
    frame = frames[a.index].load()

    def attitude(t):
        return odom[int(np.argmin(np.abs(stamps - t)))]["q_xyzw"]

    message = to_lidar_message(frame, attitude(frame.acquisition_midpoint))
    if a.window_s:
        poses = PoseBuffer(max_age_s=1e9)
        for r in odom:
            poses.add(r["stamp"], r["position"], r["q_xyzw"])
        window = SlidingWindowAccumulator(poses, a.window_s)
        for replay in frames[: a.index + 1]:
            f = replay.load()
            if f.stamp >= frame.stamp - a.window_s - 0.1:
                window.add(f.xyz, f.acquisition_midpoint)
        merged, t, _, _ = window.merged()
        leveled = to_lidar_message(type(frame)(frame.topic, t, frame.frame_id, 0.0, merged), attitude(t))
        message = LidarMessage(leveled.points, t, leveled.frame_id, leveled.clock_domain)
    result = process_lidar(LidarProcessor(**load_config(a.config)), message)

    colors = np.full((len(result.points), 3), 0.6)
    colors[result.labels == -2] = [0.2, 0.7, 0.2]
    for o in result.obstacles:
        colors[result.labels == o.cluster_id] = PALETTE[o.cluster_id % len(PALETTE)]
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(result.points))
    cloud.colors = o3d.utility.Vector3dVector(colors)
    shapes = [cloud, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4)]
    for o in result.obstacles:
        box = o3d.geometry.AxisAlignedBoundingBox(o.bbox_min, o.bbox_max)
        box.color = (1, 1, 1)
        shapes.append(box)
    print(
        f"{len(result.points)} points; {len(result.ground)} ground; "
        f"{len(result.obstacles)} clusters; frame {result.frame_id}"
    )
    if not a.screenshot:
        o3d.visualization.draw_geometries(shapes, window_name="Go2 LiDAR clusters")
        return
    view = o3d.visualization.Visualizer()
    view.create_window(visible=False, width=1280, height=800)
    view.get_render_option().background_color = np.array([0.05, 0.05, 0.08])
    view.get_render_option().point_size = 3
    for shape in shapes:
        view.add_geometry(shape)
    view.poll_events()
    view.update_renderer()
    view.capture_screen_image(a.screenshot, do_render=True)
    view.destroy_window()
    print(f"saved {a.screenshot}")


if __name__ == "__main__":
    main()
