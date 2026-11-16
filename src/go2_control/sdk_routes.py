"""Unitree SDK bindings for the guarded utility. Import only from the CLI, never at startup of a
recorder or viewer. Each client registers only the API IDs it is allowed to call, so the SDK's
own registry check (`Client.__CheckApi`) rejects anything else with RPC_ERR_CLIENT_API_NOT_REG.

API IDs come from third_party/unitree_sdk2_python at 814556d:
  obstacles_avoid 1.0.0.2: 1001 SwitchSet, 1002 SwitchGet, 1003 Move (mode 0 velocity,
      1 increment position, 2 absolute position), 1004 UseRemoteCommandFromApi
  sport 1.0.0.1: 1003 StopMove, 1008 Move, ... (gait/posture IDs never registered here)
  motion_switcher: 1001 CheckMode
  robot_state 1.0.0.1: 1003 ServiceList
Internal ID 1 (server API version) is always allowed by the SDK and is read-only.
"""
from __future__ import annotations

import json
import struct
import threading
import time
from typing import Optional

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.nav_msgs.msg.dds_ import Odometry_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_, SportModeState_
from unitree_sdk2py.rpc.client import Client

from .safety import STICK_DEADBAND

AVOID_SWITCH_SET = 1001  # never registered anywhere in this package
AVOID_SWITCH_GET = 1002
AVOID_MOVE = 1003
AVOID_USE_REMOTE_FROM_API = 1004
SPORT_STOPMOVE = 1003
MS_CHECK_MODE = 1001
RS_SERVICE_LIST = 1003

_initialized = False


def init_dds(iface: str) -> None:
    global _initialized
    if not _initialized:
        ChannelFactoryInitialize(0, iface)
        _initialized = True


class _Restricted(Client):
    """A client whose registry holds only the given API IDs."""

    def __init__(self, service: str, version: str, api_ids, timeout: float):
        super().__init__(service, False)
        self._SetApiVerson(version)
        for api in api_ids:
            self._RegistApi(api, 0)
        self.SetTimeout(timeout)

    def call(self, api_id: int, params: dict):
        return self._Call(api_id, json.dumps(params))


def timed(fn, *args):
    t0 = time.monotonic()
    try:
        code, data = fn(*args)
        err = None
    except Exception as exc:  # report, never hide
        code, data, err = None, None, f"{type(exc).__name__}: {exc}"
    return {"code": code, "data": data, "error": err, "seconds": round(time.monotonic() - t0, 4)}


def read_only_queries(timeout: float = 1.0, attempts: int = 2) -> dict:
    """Read-only RPCs only: server API versions, SwitchGet, CheckMode, ServiceList."""
    clients = {
        "obstacles_avoid": _Restricted("obstacles_avoid", "1.0.0.2", [AVOID_SWITCH_GET], timeout),
        "sport": _Restricted("sport", "1.0.0.1", [], timeout),
        "motion_switcher": _Restricted("motion_switcher", "1.0.0.1", [MS_CHECK_MODE], timeout),
        "robot_state": _Restricted("robot_state", "1.0.0.1", [RS_SERVICE_LIST], timeout),
    }
    plan = [(name, "server_api_version", c.GetServerApiVersion) for name, c in clients.items()]
    plan += [
        ("obstacles_avoid", "SwitchGet(1002)", lambda: clients["obstacles_avoid"].call(AVOID_SWITCH_GET, {})),
        ("motion_switcher", "CheckMode(1001)", lambda: clients["motion_switcher"].call(MS_CHECK_MODE, {})),
        ("robot_state", "ServiceList(1003)", lambda: clients["robot_state"].call(RS_SERVICE_LIST, {})),
    ]
    results = []
    for service, query, fn in plan:
        tries = []
        for _ in range(attempts):
            r = timed(fn)
            tries.append(r)
            if r["code"] == 0:
                break
        results.append({"service": service, "query": query, "attempts": tries})
    return {"timeout_s": timeout, "attempts_max": attempts, "results": results}


def parse_switch_get(code, data) -> Optional[bool]:
    if code != 0 or data is None:
        return None
    try:
        value = json.loads(data)["enable"]
    except (ValueError, KeyError, TypeError):
        return None
    return value if isinstance(value, bool) else None


class AvoidRoute:
    """obstacles_avoid velocity route. SwitchSet (1001) is not registered: it cannot be called."""

    def __init__(self, timeout: float = 1.0):
        self._c = _Restricted(
            "obstacles_avoid", "1.0.0.2",
            [AVOID_SWITCH_GET, AVOID_MOVE, AVOID_USE_REMOTE_FROM_API], timeout,
        )

    def avoidance_enabled(self) -> Optional[bool]:
        return parse_switch_get(*self._c.call(AVOID_SWITCH_GET, {}))

    def claim_api_control(self) -> int:
        return self._c.call(AVOID_USE_REMOTE_FROM_API, {"is_remote_commands_from_api": True})[0]

    def release_api_control(self) -> int:
        return self._c.call(AVOID_USE_REMOTE_FROM_API, {"is_remote_commands_from_api": False})[0]

    def send_velocity(self, vx: float, vyaw: float) -> int:
        params = json.dumps({"x": float(vx), "y": 0.0, "yaw": float(vyaw), "mode": 0})
        return self._c._CallNoReply(AVOID_MOVE, params)

    def send_stop(self) -> int:
        # Zero velocity on the same route. Whether it stops motion is a hardware question (step E).
        return self._c._CallNoReply(AVOID_MOVE, json.dumps({"x": 0.0, "y": 0.0, "yaw": 0.0, "mode": 0}))


def decode_remote(raw) -> dict:
    """LowState.wireless_remote layout from the SDK's wireless_controller example."""
    b = bytes(raw)
    lx, rx, ry = struct.unpack_from("<fff", b, 4)
    (ly,) = struct.unpack_from("<f", b, 20)
    return {"lx": lx, "rx": rx, "ry": ry, "ly": ly, "buttons": (b[2] << 8) | b[3]}


class StateMonitor:
    """Subscribers only. Keeps the latest sport state and remote input; callbacks stay tiny."""

    def __init__(self, still_speed: float = 0.03, still_yaw: float = 0.05):
        self._lock = threading.Lock()
        self.sport = None
        self.sport_t = None
        self.remote = None
        self.remote_t = None
        self.odom = None  # rt/utlidar/robot_odom pose; matched sport state position exactly on 2026-11-15, so not independent
        self.still_speed = still_speed
        self.still_yaw = still_yaw
        self._subs = []

    def start(self, sport_topic: str = "rt/sportmodestate", low_topic: str = "rt/lf/lowstate"):
        s = ChannelSubscriber(sport_topic, SportModeState_)
        s.Init(self._on_sport, 10)
        l = ChannelSubscriber(low_topic, LowState_)
        l.Init(self._on_low, 10)
        o = ChannelSubscriber("rt/utlidar/robot_odom", Odometry_)
        o.Init(self._on_odom, 10)
        self._subs = [s, l, o]
        return self

    def _on_odom(self, m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        with self._lock:
            self.odom = [p.x, p.y, p.z, q.x, q.y, q.z, q.w]

    def _on_sport(self, m):
        with self._lock:
            self.sport, self.sport_t = m, time.monotonic()

    def _on_low(self, m):
        r = decode_remote(m.wireless_remote)
        with self._lock:
            self.remote, self.remote_t = r, time.monotonic()

    def state_age(self) -> Optional[float]:
        with self._lock:
            return None if self.sport_t is None else time.monotonic() - self.sport_t

    def operator_input(self) -> bool:
        with self._lock:
            r = self.remote
        if r is None:
            return True  # unknown counts as busy: refuse rather than guess
        sticks = max(abs(r["lx"]), abs(r["ly"]), abs(r["rx"]), abs(r["ry"]))
        return sticks > STICK_DEADBAND or r["buttons"] != 0

    def is_still(self) -> Optional[bool]:
        with self._lock:
            m = self.sport
        if m is None:
            return None
        speed = (m.velocity[0] ** 2 + m.velocity[1] ** 2) ** 0.5
        return speed < self.still_speed and abs(m.yaw_speed) < self.still_yaw

    def snapshot(self) -> dict:
        with self._lock:
            m, r, odom = self.sport, self.remote, self.odom
            age = None if self.sport_t is None else time.monotonic() - self.sport_t
        out = {"sport_age_s": age, "remote": r, "lidar_odom": odom}
        if m is not None:
            out.update(
                error_code=int(m.error_code), mode=int(m.mode), gait_type=int(m.gait_type),
                position=[float(v) for v in m.position], velocity=[float(v) for v in m.velocity],
                yaw_speed=float(m.yaw_speed), rpy=[float(v) for v in m.imu_state.rpy],
                body_height=float(m.body_height), range_obstacle=[float(v) for v in m.range_obstacle],
            )
        return out


def front_surface_x(xyz, ground_z: float, half_width: float = 0.35, min_height: float = 0.10):
    """Median x (base frame) of the nearest surface in the corridor ahead, or None."""
    import numpy as np

    ahead = xyz[(xyz[:, 0] > 0.3) & (np.abs(xyz[:, 1]) < half_width) & (xyz[:, 2] > ground_z + min_height)]
    if len(ahead) < 20:
        return None
    first = np.percentile(ahead[:, 0], 5)
    return float(np.median(ahead[ahead[:, 0] < first + 0.15, 0]))


class FrontRange:
    """Live LiDAR range to the surface ahead (rt/utlidar/cloud_base, ~15 Hz). Subscriber only.

    Progress = direction * (start range - current range), so reversing away from the surface
    ahead counts too. On 2026-11-15 this read 0.362 m for a move a tape measured at 0.37 m, while
    sport-state odometry read 0.294 m.
    """

    def __init__(self, ground_z: float = -0.325, window: float = 0.3, max_age: float = 0.5, direction: int = 1):
        self._lock = threading.Lock()
        self.ground_z = ground_z  # base-frame ground height measured on this robot (histogram mode)
        # Single frames left the corridor empty half the time; a 0.3 s window (~5 frames) smears
        # by at most ~0.09 m at 0.3 m/s, which under-reads progress (the safe direction).
        self.window = window
        self.max_age = max_age
        self._frames = []
        self.latest = None
        self.latest_t = None
        self.start = None
        self.direction = 1 if direction >= 0 else -1

    def start_sub(self):
        from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_

        from go2_perception.adapters.go2.pointcloud import frame_from_msg

        import numpy as np

        def on_cloud(m):
            now = time.monotonic()
            xyz = frame_from_msg(m, "rt/utlidar/cloud_base", time.time()).xyz
            with self._lock:
                self._frames = [(t, f) for t, f in self._frames if now - t <= self.window] + [(now, xyz)]
                frames = [f for _, f in self._frames]
            x = front_surface_x(np.concatenate(frames), self.ground_z)
            if x is not None:
                with self._lock:
                    self.latest, self.latest_t = x, now

        sub = ChannelSubscriber("rt/utlidar/cloud_base", PointCloud2_)
        sub.Init(on_cloud, 10)
        self._sub = sub
        return self

    def current(self) -> Optional[float]:
        with self._lock:
            x, t = self.latest, self.latest_t
        if x is None or t is None or time.monotonic() - t > self.max_age:
            return None
        return x

    def mark_start(self) -> Optional[float]:
        self.start = self.current()
        return self.start

    def progress(self) -> Optional[float]:
        x = self.current()
        if x is None or self.start is None:
            return None
        return self.direction * (self.start - x)


def odom_progress(monitor: StateMonitor, direction: int = 1):
    """Progress along the start heading (sign = direction) from sport-state position. Under-read ~15-20% here."""
    import math

    snap = monitor.snapshot()
    if "position" not in snap:
        return None
    x0, y0, yaw0 = snap["position"][0], snap["position"][1], snap["rpy"][2]

    def progress():
        s = monitor.snapshot()
        if "position" not in s:
            return None
        dx, dy = s["position"][0] - x0, s["position"][1] - y0
        return (1 if direction >= 0 else -1) * (dx * math.cos(yaw0) + dy * math.sin(yaw0))

    return progress


def fuse_progress(lidar: Optional[float], odom: Optional[float]):
    """Combine LiDAR and odometry progress; returns (progress, lidar_used).

    LiDAR is the accurate reference (36 vs 37 cm tape) but its "first surface ahead" can switch
    objects: in X3 (2026-11-15) it jumped +2.4 m in one step while odometry read 0.05 m. LiDAR is
    used only while it agrees with odometry within 0.15 m + 30 % (odometry under-read ~15-20 %).
    """
    if odom is None:
        return lidar, lidar is not None
    if lidar is not None and abs(lidar - odom) <= 0.15 + 0.3 * abs(odom):
        return max(lidar, odom), True
    return odom, False


def yaw_progress(monitor: StateMonitor, direction: int):
    """Heading change since the call, signed so the commanded turn direction is positive (IMU rpy)."""
    import math

    snap = monitor.snapshot()
    if "rpy" not in snap:
        return None
    yaw0 = snap["rpy"][2]
    sign = 1 if direction >= 0 else -1

    def progress():
        s = monitor.snapshot()
        if "rpy" not in s:
            return None
        d = s["rpy"][2] - yaw0
        return sign * math.atan2(math.sin(d), math.cos(d))

    return progress

