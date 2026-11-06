# Go2 connection (verified 2026-11-06)
Every value here was observed on this robot and workstation unless marked otherwise.
The access path is **read-only**: DDS subscribers plus a multicast read socket.

## Hardware and host
| Item | Value |
|---|---|
| Robot | Unitree Go2 Pro (as stated by the user) |
| LiDAR | Probably Unitree L2; inferred, not self-reported. See [SENSOR_VALIDATION](SENSOR_VALIDATION.md) |
| Robot address | 192.168.123.161 (MAC 7e:1d:75:60:f5:89), Ethernet |
| Host | Ubuntu 24.04.4 LTS, kernel 6.17.0-22-generic, 16 CPU threads, 31 GiB RAM |
| GPU | NVIDIA RTX 2070 Super, 8 GiB, driver 595.58.03 (CUDA verified with torch 2.6.0, CUDA 12.4) |
| Interface | `eno2` (altname enp61s0), 1000 Mb/s |
| Host address | 192.168.123.222/24, NetworkManager profile `go2-wired`: manual, never-default, autoconnect priority 50. Pre-existing; not changed in Phase 2 |
| Internet | Stays on `wlo1`; the default route is unaffected |

`go2-wired` must outrank `netplan-eno2` (DHCP). A prior session found that the DHCP profile
can otherwise take over after a link drop.

## Software
| Item | Status |
|---|---|
| Project env | `.venv` (Python 3.10.22, gitignored): repo pins + cyclonedds 0.10.2 + PyAV 17.1.0 + editable SDK |
| Unitree SDK | `third_party/unitree_sdk2_python` commit 814556d, unitree_sdk2py 1.0.1, gitignored |
| DDS | Cyclone DDS 0.10.2 in Python, domain 0, interface bound to `eno2` |
| System ROS 2 | **Not installed**, so there is no `ros2` CLI, rosbag2 or RViz2. Not installed in Phase 2 |
| Earlier env `go2` | Python 3.10, numpy 2.2.6, OpenCV 5.0: conflicts with repo pins; left untouched |
| Earlier env `env_isaaclab` | Isaac Sim's bundled ROS 2 Humble rclpy (`scripts/go2_ros_env.sh`); not used by this pipeline |
| Base conda Python | 3.13.9: outside the supported range; not used |

### Rebuild the environment
```bash
cd ~/code/Go2Navigation
git clone https://github.com/unitreerobotics/unitree_sdk2_python third_party/unitree_sdk2_python
git -C third_party/unitree_sdk2_python checkout 814556d
~/anaconda3/envs/go2/bin/python -m venv .venv   # any Python 3.10-3.12 works
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e '.[go2]'
.venv/bin/python -m pip install --no-deps -e third_party/unitree_sdk2_python
curl -fL https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt -o yolov8n.pt
sha256sum yolov8n.pt  # f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36
```
`--no-deps` keeps the SDK from changing the pinned numpy and OpenCV versions.

## Reconnection procedure
Run in order; stop at the first failure.
```bash
cat /sys/class/net/eno2/carrier             # 1 = link. 0 = check robot power and the cable
ip -br addr show eno2                       # must show 192.168.123.222/24
ping -c 3 -I eno2 192.168.123.161           # about 0.2 ms; reachability only, not streams
.venv/bin/python scripts/go2_discover.py    # DDS works: expect about 95 publications
GO2_HARDWARE=1 .venv/bin/python -m pytest -q -m hardware tests/test_go2_hardware.py  # real data
```
Allow about a minute after power-on. If eno2 has link but the wrong address, run
`nmcli con up go2-wired`. That changes host network state, so it is not part of normal startup.

## Interfaces
Full formats and measurements are in [SENSOR_VALIDATION](SENSOR_VALIDATION.md).
| Stream | Access | Verified |
|---|---|---|
| RGB | H.264 RTP multicast 230.1.1.1:1720 from .161, joined on eno2 | Yes |
| LiDAR single frames | DDS `rt/utlidar/cloud`, `rt/utlidar/cloud_base` | Yes |
| LiDAR rolling buffer | DDS `rt/utlidar/cloud_deskewed` | Yes |
| IMU | DDS `rt/utlidar/imu` (type defined locally; the SDK lacks `Imu_`) | Yes |
| Odometry | DDS `rt/utlidar/robot_odom`, `rt/utlidar/robot_pose` | Yes |
| Body state | DDS `rt/lf/sportmodestate`, `rt/lf/lowstate` and full-rate variants | Yes |
| LiDAR status | DDS `rt/utlidar/lidar_state` | **Advertised, 0 messages** |
| DDS video | DDS `rt/frontvideostream` | **Advertised, 0 messages** |
| Other point clouds | `grid_map`, `height_map`, `range_map`, `voxel_map`, `uslam/cloud_map` | **0 messages in 15 s** |
| TF | No `tf` topic advertised | **Unavailable** |
| SDK `VideoClient` JPEG | RPC request | **Deliberately untested**: it creates a request publisher |
| WebRTC community stacks | | **Not needed, not tried** |

The camera address came from earlier local work and was confirmed by receiving real video here. It
was not found in the SDK, whose camera example uses `VideoClient`, or in a documentation search.

## Safety boundary
The robot also advertises command topics, including `rt/api/sport/request`, `rt/lowcmd`,
`rt/api/motion_switcher/request` and `rt/utlidar/switch`. **This adapter never writes to them.**
`tests/test_go2_adapter.py` fails if the adapter source starts using `ChannelPublisher`,
`SportClient`, `VideoClient`, `DataWriter(` or `.sendto(`. That is a guard against accidents, not a
proof. The only packets sent on purpose, apart from DDS discovery and IGMP joins, are ICMP echoes
used for reachability and clock measurement. No firmware, robot configuration or network setting
was changed.

## Known issues
- **Link recovery after a cable pull is untested.** The camera receiver rejoins after silence
  (offline test only); DDS rediscovery after a real link drop has not been observed.
- **Clock mismatch:** the robot clock is about 273 days behind the host. Never compare robot stamps
  with host time directly. See [SENSOR_VALIDATION](SENSOR_VALIDATION.md).
- **Small socket buffer:** `SO_RCVBUF` was clamped to 425,984 bytes rather than the requested 4 MiB.
  There was no RTP loss at that size. Raising it needs `sysctl`, so it was left alone.
- **SDK queue drops:** the SDK's `BQueue` silently drops the oldest sample when full. Keep DDS
  callbacks fast; heavy work belongs on another thread.
- **Shutdown:** recording scripts end with `os._exit` after flushing, to avoid Cyclone DDS teardown
  hangs that were guarded against but not separately reproduced.
- **Stop scripts by PID:** `pkill -f` with a script name can match the invoking shell.
