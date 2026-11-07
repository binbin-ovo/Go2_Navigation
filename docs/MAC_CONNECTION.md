# Go2 connection from macOS (verified 2026-11-07)
Every value here was observed on this Mac and robot unless marked otherwise. Access is
**read-only**: DDS subscribers, DDS discovery readers and one multicast read socket. No robot
command, firmware or robot-side setting was changed. Linux reference: [GO2_CONNECTION](GO2_CONNECTION.md).

## Result
Native macOS works with the **existing Phase 2 adapter, unchanged** (option B). No ROS 2, VM or
Linux bridge is needed. Cyclone DDS (Python wheel) and the Unitree SDK run on Apple Silicon.

| Item | Value |
|---|---|
| Host | Apple Silicon Mac, arm64, macOS 26.6.2 (25G83) |
| Robot | Go2 Pro at 192.168.123.161; same robot as the Linux docs (same Ethernet MAC) |
| Interface | `en9`, USB gigabit adapter, service "USB 10/100/1000 LAN", 1000baseT full duplex |
| Host address | 192.168.123.222/24, manual, **no router** (default route stays on Wi-Fi) |
| Python | `.venv-go2`: CPython 3.10.17 (uv-managed), arm64, gitignored |
| Packages | repo pins + `cyclonedds==0.10.2` (cp310 macOS arm64 wheel) + `av==17.1.0` |
| Unitree SDK | `third_party/unitree_sdk2_python` at 814556d, `--no-deps` editable, gitignored |

The adapter's interface name is a parameter: pass `--iface en9` (DDS) and the host IP
(`--host-ip`, multicast join) instead of the Linux `eno2` default in `configs/go2_sensors.yaml`.

## Why a second environment
`.venv` (Python 3.12) cannot install `cyclonedds==0.10.2`: PyPI has macOS arm64 wheels only for
cp38-cp310, so 3.12 would build from source against a native Cyclone C library. A 3.10 env
reuses the official wheel and matches the Linux workstation's Python. Torch 2.6.0 and every
other pin installed from wheels. `.venv` stays the offline-only environment.

## Setup (reproducible)
```bash
uv venv --python 3.10 .venv-go2
uv pip install --python .venv-go2/bin/python -e '.[test,go2]'
git clone https://github.com/unitreerobotics/unitree_sdk2_python third_party/unitree_sdk2_python
git -C third_party/unitree_sdk2_python checkout 814556d
uv pip install --python .venv-go2/bin/python --no-deps -e third_party/unitree_sdk2_python
chflags nohidden .venv-go2/lib/python3.10/site-packages/*.pth   # see Problems
networksetup -setmanual "USB 10/100/1000 LAN" 192.168.123.222 255.255.255.0   # needs approval
```
Undo the address with `networksetup -setdhcp "USB 10/100/1000 LAN"`. The service name differs
per adapter; list it with `networksetup -listallhardwareports`.

## Reconnection procedure (verified)
Run in order; stop at the first failure.
```bash
ifconfig en9 | grep -E "status|inet "        # status: active, inet 192.168.123.222
ping -c 3 192.168.123.161                    # first try right after addressing may fail (ARP)
.venv-go2/bin/python scripts/go2_discover.py --iface en9 --seconds 10   # 95 publications
.venv-go2/bin/python scripts/go2_probe.py --iface en9 --seconds 15      # live rates
.venv-go2/bin/python scripts/go2_record.py --iface en9 --seconds 10 --name mac_smoke_01
```

## Measured streams (robot standing still)
`go2_probe.py` 15 s (receipt rate) and `go2_record.py` 10 s (`mac_smoke_01`, stats.json).
| Stream | Rate | Notes |
|---|---|---|
| RGB, H.264 RTP 230.1.1.1:1720 | **14.30 Hz** | 1280x720 BGR; 0 lost / 0 reordered of 1,283 RTP packets; 0 decoder errors; keyframes every 2.0 s; first 25 frames dropped until IDR |
| `rt/utlidar/cloud` | 15.4 Hz | `utlidar_lidar`, mean 3,993 pts, point_step 32 |
| `rt/utlidar/cloud_base` | 15.5 Hz | `base_link`, mean 1,349 pts |
| `rt/utlidar/cloud_deskewed` | 15.6 Hz | `odom`, mean 11,350 pts, 88% overlap: rolling buffer |
| `rt/utlidar/imu` | 258.8 Hz | 9 estimated missed of 2,533 |
| `rt/utlidar/robot_odom` | 152.7 Hz | `odom`->`base_link`; 16 estimated missed, max receipt gap 20 ms |
| `rt/utlidar/robot_pose` | 19.2 Hz | `odom` |
| `rt/lf/sportmodestate`, `rt/lf/lowstate` | 20 Hz | harmless status read verified |
| `rt/sportmodestate`, `rt/lowstate` | 295, 500 Hz | |
| `rt/utlidar/height_map_array`, `range_info` | 15.9 Hz | received, not used |
| `rt/frontvideostream`, `lidar_state`, `selftest`, `servicestate`, `switch`, grid/height/range/voxel maps, `uslam/cloud_map` | **0 messages** | as on Linux; LiDAR model still unconfirmed |

Ping RTT 0.5-1.7 ms through the USB adapter (Linux on-board NIC: about 0.2 ms). The camera
socket obtained the full 4 MiB `SO_RCVBUF` (Linux was clamped to 416 KiB). Recorder: 28% of one
core, writer queue never dropped.

## Clocks
All DDS stamps are the robot clock (`go2_robot`); RGB has only host receipt time. On this Mac the
receipt-minus-stamp offset was about **23,654,671.9 s** (robot about 273.8 days behind), not the
Linux workstation's 23,568,393.04 s: the offset is host-specific and must be re-measured.
`clock.py` (ICMP timestamps) returns `null` on macOS: unprivileged macOS `ping -M time` fails with
`sendto: Invalid argument`, and Linux `-T tsonly` does not exist. Recordings therefore have no
ICMP offset. LiDAR tracking is unaffected (LiDAR and odometry share the robot clock); RGB-LiDAR
pairing would need the offset plus the unknown camera latency.

## Problems encountered
- **Hidden `.pth` files:** the new venv's `.pth` files carried the macOS `hidden` flag, which
  `site.py` skips, so editable installs (project and SDK) silently failed to import from
  scripts. Fixed with `chflags nohidden`. Re-check after any reinstall.
- **Old `.venv` runs a stale package copy:** it contains a non-editable `go2_perception` in
  site-packages, so its tests import old code, not `src/`. Use `PYTHONPATH=src` or reinstall
  editable; it was not modified here.
- `go2_live.py` and `tests/test_go2_hardware.py` hard-code `eno2` via the config; on the Mac
  they need an interface override (not yet added). The recorder's `max_rss_mb` is wrong on
  macOS (`ru_maxrss` is bytes there, not KiB): 132,704 means about 130 MB.
- Unverified on the Mac: link-drop recovery, runs longer than 15 s, live `go2_live.py`.
