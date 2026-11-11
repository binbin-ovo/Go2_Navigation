"""Tiny public-safe synthetic ``go2-recording-v1`` plus synthetic calibration, for tests and demos.

Scene in base_link: floor at z = -0.33, a wall at x = 3 m, a 0.4 m box centred at (1.6, 0.5).
The robot is stationary. Clocks follow the real layout: robot stamps, host receipt = stamp + offset
(+ camera latency for RGB). The calibration is SYNTHETIC: never use it on real data.
"""

import json
from pathlib import Path

import cv2
import numpy as np

from .calibration import CameraIntrinsics, RigidTransform, project_points, save_calibration

OFFSET_S = 1000.0
CAMERA_LATENCY_S = 0.048
SCAN_DT = 0.065
BOX_CENTER = np.array([1.6, 0.5, -0.13])
BOX_HALF = 0.2
SIZE = (160, 120)

# Camera 0.3 m ahead of base_link, 0.1 m up, looking along +x: optical z = base x, x = -y, y = -z.
R_BASE_CAM = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
T_BASE_CAMERA = np.eye(4)
T_BASE_CAMERA[:3, :3] = R_BASE_CAM
T_BASE_CAMERA[:3, 3] = [0.3, 0.0, 0.1]


def synthetic_calibration():
    K = np.array([[80.0, 0, 79.5], [0, 80.0, 59.5], [0, 0, 1]])
    intr = CameraIntrinsics(SIZE[0], SIZE[1], K, "pinhole_radtan", np.zeros(5), "synthetic",
                            reprojection_rms_px=0.0, provenance={"generator": __name__})
    ext = RigidTransform(T_BASE_CAMERA, intr.camera_frame, "base_link", "synthetic",
                         {"generator": __name__})
    return intr, ext


def scene_points(rng, n_floor=300, n_wall=200, n_box=150):
    floor = np.c_[rng.uniform(0.4, 2.9, n_floor), rng.uniform(-1.5, 1.5, n_floor), np.full(n_floor, -0.33)]
    wall = np.c_[np.full(n_wall, 3.0), rng.uniform(-1.5, 1.5, n_wall), rng.uniform(-0.33, 1.0, n_wall)]
    # Box faces visible from the robot: front face x = 1.4 and top face.
    f = np.c_[np.full(n_box, BOX_CENTER[0] - BOX_HALF),
              BOX_CENTER[1] + rng.uniform(-BOX_HALF, BOX_HALF, n_box),
              BOX_CENTER[2] + rng.uniform(-BOX_HALF, BOX_HALF, n_box)]
    return np.vstack([floor, wall, f])


def box_corners():
    c = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    return BOX_CENTER + BOX_HALF * c


def render_image(intr, ext):
    img = np.full((intr.height, intr.width, 3), 90, np.uint8)
    cam = ext.inverse().apply(box_corners())
    uv, valid = project_points(cam, intr)
    if valid.all():
        x0, y0 = np.floor(uv.min(0)).astype(int)
        x1, y1 = np.ceil(uv.max(0)).astype(int)
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), -1)  # BGR red box
    return img


def write_recording(root, scans=12, seed=0):
    root = Path(root)
    rng = np.random.default_rng(seed)
    intr, ext = synthetic_calibration()
    (root / "lidar" / "utlidar_cloud_base").mkdir(parents=True, exist_ok=False)
    (root / "rgb").mkdir()
    (root / "state").mkdir()
    t0 = 1000.0
    lidar_rows, rgb_rows, odom_rows = [], [], []
    for k in range(scans):
        stamp = t0 + k * SCAN_DT
        pts = scene_points(rng)
        times = np.sort(rng.uniform(0, 0.062, len(pts))).astype(np.float32)
        arr = np.c_[pts, rng.uniform(20, 200, len(pts))].astype(np.float32)
        np.save(root / "lidar" / "utlidar_cloud_base" / f"{k:06d}.npy", arr)
        np.save(root / "lidar" / "utlidar_cloud_base" / f"{k:06d}_time.npy", times)
        lidar_rows.append({"file": f"{k:06d}.npy", "topic": "rt/utlidar/cloud_base", "stamp": stamp,
                           "host_recv": stamp + OFFSET_S + 0.070, "frame_id": "base_link",
                           "points": len(arr), "has_intensity": True, "has_time": True, "ring_values": [1]})
    for k in range(int(scans * SCAN_DT / 0.07)):
        exposure = t0 + 0.02 + k * 0.07
        name = f"{k:06d}.png"
        cv2.imwrite(str(root / "rgb" / name), render_image(intr, ext))
        recv = exposure + OFFSET_S + CAMERA_LATENCY_S
        rgb_rows.append({"file": name, "rtp_timestamp": 6000 * k, "host_recv_first": recv,
                         "host_recv_last": recv + 0.0003, "width": SIZE[0], "height": SIZE[1],
                         "complete": True, "encoding": "bgr8", "frame_id": "go2_front_camera_unrectified",
                         "synthetic_exposure_robot_s": exposure})
    for k in range(int(scans * SCAN_DT / 0.0066) + 20):
        stamp = t0 - 0.05 + k * 0.0066
        odom_rows.append({"stamp": stamp, "host_recv": stamp + OFFSET_S, "frame_id": "odom",
                          "child_frame_id": "base_link", "position": [0.0, 0.0, 0.31],
                          "q_xyzw": [0.0, 0.0, 0.0, 1.0], "linear": [0.0, 0.0, 0.0],
                          "angular": [0.0, 0.0, 0.0]})
    for path, rows in [(root / "lidar" / "utlidar_cloud_base" / "index.jsonl", lidar_rows),
                       (root / "rgb" / "index.jsonl", rgb_rows),
                       (root / "state" / "utlidar_robot_odom.jsonl", odom_rows)]:
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    (root / "manifest.json").write_text(json.dumps({
        "schema": "go2-recording-v1", "notes": "SYNTHETIC fixture, public-safe, no real sensor data",
        "clocks": {"stamp": "synthetic go2_robot", "host_recv": f"stamp + {OFFSET_S}"},
        "rgb_frames": len(rgb_rows), "lidar_frames": {"utlidar_cloud_base": scans}}, indent=2))
    save_calibration(root / "synthetic_calibration.yaml", intr, ext)
    return root
