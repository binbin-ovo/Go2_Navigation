#!/usr/bin/env python
"""Read-only: prints the Go2 LiDAR's self-reported status via the Unitree SDK. Subscribes only.
Usage: ~/anaconda3/envs/go2/bin/python scripts/lidar_state.py [iface] [seconds]"""
import sys, time
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LidarState_
from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_

iface = sys.argv[1] if len(sys.argv) > 1 else "eno2"
secs = float(sys.argv[2]) if len(sys.argv) > 2 else 6
ChannelFactoryInitialize(0, iface)
states, clouds = [], []
a = ChannelSubscriber("rt/utlidar/lidar_state", LidarState_); a.Init(states.append, 10)
b = ChannelSubscriber("rt/utlidar/cloud", PointCloud2_); b.Init(lambda m: clouds.append(m.width * m.height), 10)
time.sleep(secs)
print(f"cloud via SDK: {len(clouds)} msgs in {secs:.0f}s, {len(clouds)/secs:.1f} Hz, "
      f"avg points {sum(clouds)/max(len(clouds),1):.0f}")
print(f"lidar_state: {len(states)} msgs")
if states:
    print(states[-1])
    print("error_state values seen:", sorted({s.error_state for s in states}))
else:
    print("FAIL: no lidar_state messages")
