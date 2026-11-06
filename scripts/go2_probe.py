#!/usr/bin/env python
"""Read-only Go2 stream probe. Subscribes to sensor topics via the Unitree SDK and reports
measured rates, formats, stamps and frames. Creates subscribers only; never a publisher or
RPC client. Usage: .venv/bin/python scripts/go2_probe.py [--iface eno2] [--seconds 15] [--json F]
"""
import argparse
import json
import os
import sys
import threading
import time

import numpy as np
from cyclonedds.idl.annotations import key  # noqa: F401
from cyclonedds.idl.types import array, float32, float64, sequence  # noqa: F401
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import PointStamped_, PoseStamped_
from unitree_sdk2py.idl.nav_msgs.msg.dds_ import Odometry_
from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import (
    Go2FrontVideoData_,
    HeightMap_,
    LidarState_,
    LowState_,
    SportModeState_,
)

CLOUDS = {
    "rt/utlidar/cloud",
    "rt/utlidar/cloud_base",
    "rt/utlidar/cloud_deskewed",
    "rt/utlidar/grid_map",
    "rt/utlidar/height_map",
    "rt/utlidar/range_map",
    "rt/utlidar/voxel_map",
    "rt/uslam/cloud_map",
}
OTHER = {
    "rt/utlidar/lidar_state": LidarState_,
    "rt/utlidar/robot_odom": Odometry_,
    "rt/utlidar/robot_pose": PoseStamped_,
    "rt/utlidar/range_info": PointStamped_,
    "rt/utlidar/height_map_array": HeightMap_,
    "rt/lf/lowstate": LowState_,
    "rt/lowstate": LowState_,
    "rt/lf/sportmodestate": SportModeState_,
    "rt/sportmodestate": SportModeState_,
    "rt/frontvideostream": Go2FrontVideoData_,
    "rt/servicestate": String_,
    "rt/multiplestate": String_,
    "rt/utlidar/switch": String_,
    "rt/selftest": String_,
}


def stamp_of(m):
    h = getattr(m, "header", None)
    if h is not None:
        return h.stamp.sec + h.stamp.nanosec * 1e-9, h.frame_id
    s = getattr(m, "stamp", None)
    if isinstance(s, float):
        return s, None
    if s is not None:
        return s.sec + s.nanosec * 1e-9, None
    return None, None


def cloud_xyz(m):
    off = {f.name: f.offset for f in m.fields}
    d = np.frombuffer(bytes(m.data), np.uint8).reshape(-1, m.point_step)
    return np.stack(
        [np.frombuffer(d[:, off[k] : off[k] + 4].tobytes(), "<f4") for k in "xyz"], 1
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iface", default="eno2")
    p.add_argument("--seconds", type=float, default=15)
    p.add_argument("--json")
    p.add_argument("--clouds-npz", help="save the last clouds of each topic for analysis")
    a = p.parse_args()
    ChannelFactoryInitialize(0, a.iface)
    rec, lock = {}, threading.Lock()

    def handler(topic):
        def cb(m):
            t = time.time()
            with lock:
                r = rec.setdefault(topic, {"recv": [], "stamp": [], "keep": []})
                r["recv"].append(t)
                s, fr = stamp_of(m)
                r["stamp"].append(s)
                r["frame"] = fr
                r["keep"].append(m)
                del r["keep"][:-20]
                if topic in CLOUDS:
                    r.setdefault("n", []).append(m.width * m.height)

        return cb

    subs = []
    for t in CLOUDS:
        s = ChannelSubscriber(t, PointCloud2_)
        s.Init(handler(t), 10)
        subs.append(s)
    for t, ty in OTHER.items():
        s = ChannelSubscriber(t, ty)
        s.Init(handler(t), 10)
        subs.append(s)
    time.sleep(a.seconds)
    with lock:
        snap = {k: dict(v) for k, v in rec.items()}
    report = {}
    npz = {}
    for t in sorted(set(CLOUDS) | set(OTHER)):
        r = snap.get(t)
        if not r:
            print(f"{t:36s} NO MESSAGES in {a.seconds:.0f}s")
            report[t] = {"count": 0}
            continue
        recv = np.array(r["recv"])
        o = {"count": len(recv), "hz_recv": round(len(recv) / a.seconds, 2), "frame": r.get("frame")}
        if len(recv) > 2:
            d = np.diff(recv)
            o["gap_ms_median_max"] = [round(float(np.median(d) * 1e3), 1), round(float(d.max() * 1e3), 1)]
        st = [s for s in r["stamp"] if s is not None]
        if st:
            st = np.array(st)
            o["last_stamp"] = float(st[-1])
            o["host_minus_stamp_s"] = round(float(recv[-1] - st[-1]), 4)
            if len(st) > 2:
                ds = np.diff(st)
                o["stamp_dt_ms_median_min_max"] = [round(float(x) * 1e3, 2) for x in (np.median(ds), ds.min(), ds.max())]
                o["nonincreasing_stamps"] = int((ds <= 0).sum())
        if t in CLOUDS:
            m = r["keep"][-1]
            o.update(
                points_mean=round(float(np.mean(r["n"])), 1),
                points_min=int(min(r["n"])),
                points_max=int(max(r["n"])),
                height=m.height,
                width=m.width,
                point_step=m.point_step,
                row_step=m.row_step,
                is_dense=bool(m.is_dense),
                is_bigendian=bool(m.is_bigendian),
                fields=[[f.name, f.offset, f.datatype, f.count] for f in m.fields],
            )
            xs = [cloud_xyz(x) for x in r["keep"] if x.width * x.height]
            if xs:
                P = np.concatenate(xs)
                fin = P[np.isfinite(P).all(1)]
                o["bbox_min"] = np.round(fin.min(0), 3).tolist()
                o["bbox_max"] = np.round(fin.max(0), 3).tolist()
                ov = []
                for u, v in zip(xs, xs[1:]):
                    su = {tuple(x) for x in np.round(u, 4)}
                    ov.append(sum(tuple(x) in su for x in np.round(v, 4)) / max(len(v), 1))
                o["consecutive_exact_point_overlap_mean"] = round(float(np.mean(ov)), 4) if ov else None
                npz[t.replace("/", "_")] = np.concatenate(xs[-5:])
        print(f"{t:36s} {json.dumps(o)}")
        report[t] = o
    for t in ("rt/utlidar/lidar_state", "rt/servicestate", "rt/multiplestate", "rt/selftest", "rt/utlidar/switch"):
        r = snap.get(t)
        if r:
            print(f"\nLAST {t}:\n{str(r['keep'][-1])[:1800]}")
            report[t]["last"] = str(r["keep"][-1])[:4000]
    if a.json:
        with open(a.json, "w") as f:
            json.dump(report, f, indent=2)
    if a.clouds_npz and npz:
        np.savez_compressed(a.clouds_npz, **npz)

    sys.stdout.flush()
    os._exit(0)  # cyclonedds listener threads may block interpreter shutdown


if __name__ == "__main__":
    main()
