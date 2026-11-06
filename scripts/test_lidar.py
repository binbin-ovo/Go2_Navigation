#!/usr/bin/env python
"""Read-only Go2 LiDAR test: lists topics, subscribes to the point cloud, reports stats, saves PLY + top-down PNG.
Publishes nothing. Usage: source scripts/go2_ros_env.sh && $GO2_PY scripts/test_lidar.py [--topic T] [--seconds N]"""
import argparse, os, time
import numpy as np, cv2, rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2

p = argparse.ArgumentParser()
p.add_argument("--topic", default="/utlidar/cloud")
p.add_argument("--seconds", type=float, default=8)
p.add_argument("--out", default="output")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
rclpy.init()
n = rclpy.create_node("go2_lidar_test")

t0 = time.time()
while time.time() - t0 < 6:
    rclpy.spin_once(n, timeout_sec=0.2)
topics = dict(n.get_topic_names_and_types())
print(f"Discovered {len(topics)} topics")
for t in sorted(topics):
    if any(k in t for k in ("lidar", "cloud", "video", "camera", "lio", "odom")):
        print(" ", t, topics[t])

stamps, sizes, msgs = [], [], []
def cb(m):
    stamps.append(time.time()); sizes.append(m.width * m.height); msgs.append(m)
    del msgs[:-30]
qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE,
                 history=HistoryPolicy.KEEP_LAST, depth=10)
n.create_subscription(PointCloud2, a.topic, cb, qos)
t0 = time.time()
while time.time() - t0 < a.seconds:
    rclpy.spin_once(n, timeout_sec=0.1)
if not msgs:
    raise SystemExit(f"FAIL: no messages on {a.topic} in {a.seconds}s")

m = msgs[-1]
print(f"{a.topic}: {len(stamps)} msgs, {len(stamps)/a.seconds:.1f} Hz, frame={m.header.frame_id}, "
      f"points/msg avg={np.mean(sizes):.0f} max={max(sizes)}, step={m.point_step}, "
      f"fields={[(f.name, f.offset, f.datatype) for f in m.fields]}")
off = {f.name: f.offset for f in m.fields}
def xyz(m):
    d = np.frombuffer(bytes(m.data), np.uint8).reshape(-1, m.point_step)
    return np.stack([np.frombuffer(d[:, off[k]:off[k] + 4].tobytes(), "<f4") for k in "xyz"], 1)
pts = np.concatenate([xyz(x) for x in msgs])
pts = pts[np.isfinite(pts).all(1)]
r = np.linalg.norm(pts, axis=1)
print(f"Accumulated {len(pts)} pts over {len(msgs)} msgs; min={pts.min(0).round(2)} max={pts.max(0).round(2)} "
      f"range median={np.median(r):.2f} p95={np.percentile(r, 95):.2f}")
ply = os.path.join(a.out, "lidar.ply")
with open(ply, "w") as f:
    f.write(f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n")
    np.savetxt(f, pts, fmt="%.4f")
# top-down view, coloured by height, centred on the median of the cloud
S, c = 800, np.median(pts[:, :2], 0)
span = max(np.percentile(np.abs(pts[:, :2] - c), 98) * 2.2, 1)
u = ((pts[:, 0] - c[0]) / span + .5) * S; v = (.5 - (pts[:, 1] - c[1]) / span) * S
z = np.clip((pts[:, 2] - np.percentile(pts[:, 2], 2)) / (np.ptp(np.percentile(pts[:, 2], [2, 98])) + 1e-6), 0, 1)
col = cv2.applyColorMap((z * 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3)
img = np.full((S, S, 3), 20, np.uint8); ok = (u >= 0) & (u < S) & (v >= 0) & (v < S)
img[v[ok].astype(int), u[ok].astype(int)] = col[ok]
cv2.putText(img, f"{a.topic} top-down, {span:.1f} m across, frame {m.header.frame_id}", (10, 25), 0, .55, (255, 255, 255), 1)
png = os.path.join(a.out, "lidar_topdown.png"); cv2.imwrite(png, img)
print(f"PASS: saved {ply}, {png}")
n.destroy_node(); rclpy.shutdown()
