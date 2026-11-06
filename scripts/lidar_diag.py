import time, numpy as np, rclpy, sys
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped
rclpy.init(); n = rclpy.create_node("go2_lidar_diag")
q = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST, depth=10)
T = ["/utlidar/cloud", "/utlidar/cloud_deskewed", "/utlidar/cloud_base"]
D = {t: [] for t in T}; C = {t: 0 for t in T}; other = {}
def mk(t):
    def cb(m):
        C[t] += 1; D[t].append(m); del D[t][:-150]
    return cb
for t in T: n.create_subscription(PointCloud2, t, mk(t), q)
for t, ty in (("/utlidar/server_log", String), ("/utlidar/switch", String), ("/utlidar/range_info", PointStamped)):
    n.create_subscription(ty, t, lambda m, t=t: other.setdefault(t, []).append(m), q)
t0 = time.time()
while time.time() - t0 < 12: rclpy.spin_once(n, timeout_sec=.05)
for t in T:
    if not D[t]: print(t, "NO DATA"); continue
    m = D[t][-1]; off = {f.name: f.offset for f in m.fields}
    P = []
    for x in D[t]:
        d = np.frombuffer(bytes(x.data), np.uint8).reshape(-1, x.point_step)
        P.append(np.stack([np.frombuffer(d[:, off[k]:off[k]+4].tobytes(), "<f4") for k in "xyz"], 1))
    P = np.concatenate(P); P = P[np.isfinite(P).all(1)]
    r = np.linalg.norm(P, axis=1); rxy = np.linalg.norm(P[:, :2], axis=1)
    az = np.degrees(np.arctan2(P[:, 1], P[:, 0])); el = np.degrees(np.arctan2(P[:, 2], rxy))
    print(f"\n{t}: {C[t]/12:.1f} Hz frame={m.header.frame_id} pts={len(P)}")
    print(f"  x[{P[:,0].min():.2f},{P[:,0].max():.2f}] y[{P[:,1].min():.2f},{P[:,1].max():.2f}] z[{P[:,2].min():.2f},{P[:,2].max():.2f}]")
    print("  range pct 5/25/50/75/95/99.9:", np.percentile(r, [5,25,50,75,95,99.9]).round(2), " frac<0.3m:", round((r<.3).mean(),3))
    print("  elevation pct 1/50/99:", np.percentile(el,[1,50,99]).round(1), " azimuth 12-bin hist:", np.histogram(az,12,(-180,180))[0].tolist())
for t, v in other.items():
    print(t, len(v), "msgs; last:", getattr(v[-1], "data", None) if hasattr(v[-1],"data") else v[-1].point)
print("no msgs:", [t for t in ("/utlidar/server_log","/utlidar/switch","/utlidar/range_info") if t not in other])
