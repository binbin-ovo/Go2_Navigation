"""Read-only Go2 guard provider for the supervised executor (contracts v1/v2, `create_guards`).

Subscribers and file reads only: no RPC, no Move, no ownership call, no avoidance setting.
Loaded by `scripts/vln_supervised_executor.py` from an absolute path whose SHA-256 is pinned
in the hardware capability, after the canonical command lock is held and DDS is initialised.

What it supplies, and from where (all times are host-monotonic receipt times, seconds):
- pose: `rt/utlidar/robot_odom` (x, y, yaw in the robot's odom frame). The robot epoch is
  the host-minus-robot stamp offset of the first message; a backwards stamp or an offset step
  above EPOCH_STEP_S starts a new epoch, so poses never compare across a robot reboot.
- geometry: `rt/utlidar/cloud_base` scans from the last WINDOW_S, each moved into the
  *current* base frame through the odom pose at its receipt. Forward clearance is
  min(nearest obstacle, end of contiguously observed floor) - FRONT_M - MARGIN_M inside a
  corridor of half-width HALF_WIDTH_M. Bins without returns end the observed region:
  absent returns are never free space. The rear is blind. V2 reports observed disc returns
  as a veto for supervised inspections. Sensor turn clearance remains zero.
- capture bound: only when the newest camera observation in the run uses the configured
  latency mapping CAMERA.latency_config_s, and the robot boot matches the boot on which
  the bound was measured (CAMERA.boot_offset_s). Otherwise None, which HOLDs.
- progress: forward = audited fuse of LiDAR front-surface change and odom along the start
  heading. Legacy turns use odom yaw. The effective profile uses relative IMU heading
  with fresh same-epoch odometry and retains both measurements.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from go2_perception.vln.motion_profile import DISC_RADIUS_M, PROFILE

# Robot footprint (Go2: ~0.70 m long, ~0.31 m wide, base_link near the trunk centre).
FRONT_M = 0.40  # base_link to the front of the head, rounded up
HALF_WIDTH_M = 0.25  # half body width 0.155 m + 0.095 m margin
MARGIN_M = 0.10  # subtracted from clearance; the executor additionally needs magnitude <= clearance
WINDOW_S = 0.5  # LiDAR accumulation; the user's obstacle window (Phase 5D)
BIN_M = 0.10
MIN_BIN_POINTS = 2  # floor or obstacle returns that make a corridor bin "observed"
OBSTACLE_MIN_DZ = 0.08  # above ground; cloud_base floor bias is +/-4 cm (Phase 5D)
OBSTACLE_MAX_DZ = 0.70  # above ground; the Go2 stands ~0.4 m
OBSTACLE_RANK = 3  # nth-nearest obstacle return, rejects single noise points
NEAR_LIMIT_M = 0.30  # cloud_base has no points within 0.3 m XY
MAX_RANGE_M = 3.0
GROUND_Z_NOMINAL = -0.325  # base-frame ground measured on this robot (2026-11-15)
EPOCH_STEP_S = 1.0
POSE_FRAME = "go2_robot_odom"

# Filled from the measured evidence; None keeps the capture bound unavailable (HOLD).
CAMERA = {
    "latency_config_s": 0.048,  # what the camera bridge subtracts from receipt (configs/go2_vln_sensors.yaml)
    # Re-measured for the robot boot seen at 22:24 on 2026-11-18 (the 10:31 measurement, bound 0.148 s at
    # boot_offset -1766947283.136, belongs to the earlier boot and is not reused). Screen-flash test
    # (hardware/capture_latency.py): 39/39 switches, switch-to-receipt median 0.144 s, max 0.200 s
    # (includes display lag and up to one 0.07 s frame interval); bound = max - 0.048 configured + 0.020 margin.
    "bound_s": 0.172,  # validated extra age on top of the configured latency
    "boot_offset_s": -1766903986.310,  # host_monotonic - robot_odom_stamp on the measured boot
    "boot_tolerance_s": 5.0,
    "basis": ("evidence/hw_readiness/capture_latency_20261118_222418.json sha256 "
              "77bbbc5a50eac0cbea5f29fad8cd69a66c6968658ccc40206c5b3d717e6841f8"),  # private, git-ignored
}


def yaw_from_quaternion(qx, qy, qz, qw):
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def to_current_base(points, pose_then, pose_now):
    """Planar move of base-frame points seen at pose_then into the base frame at pose_now."""
    (x0, y0, a0), (x1, y1, a1) = pose_then, pose_now
    c0, s0 = math.cos(a0), math.sin(a0)
    wx = x0 + c0 * points[:, 0] - s0 * points[:, 1]
    wy = y0 + s0 * points[:, 0] + c0 * points[:, 1]
    c1, s1 = math.cos(a1), math.sin(a1)
    dx, dy = wx - x1, wy - y1
    out = np.empty_like(points)
    out[:, 0] = c1 * dx + s1 * dy
    out[:, 1] = -s1 * dx + c1 * dy
    out[:, 2] = points[:, 2]
    return out


def ground_plane(points):
    """Robust floor plane z = a + b*x + c*y in base_link, or None.

    The floor is not level in base_link: standing pitch plus the LiDAR's range-dependent bias put
    it at -0.42 m at 0.3 m and -0.24 m at 1.9 m (live_10, 2026-11-17). Two trimming passes.
    """
    cand = points[(points[:, 0] > NEAR_LIMIT_M) & (points[:, 0] < 2.5) & (np.abs(points[:, 1]) < 0.8) &
                  (points[:, 2] < -0.10)]
    keep = cand
    for tolerance in (None, 0.10, 0.05):
        if len(keep) < 100:
            return None
        design = np.column_stack([np.ones(len(keep)), keep[:, 0], keep[:, 1]])
        coef = np.linalg.lstsq(design, keep[:, 2], rcond=None)[0]
        residual = cand[:, 2] - (coef[0] + coef[1] * cand[:, 0] + coef[2] * cand[:, 1])
        keep = cand[np.abs(residual) < (tolerance or 0.10)]
    if len(keep) < 100 or abs(coef[1]) > 0.25 or abs(coef[2]) > 0.25:
        return None
    return coef


def height_above_ground(points, coef):
    return points[:, 2] - (coef[0] + coef[1] * points[:, 0] + coef[2] * points[:, 1])


def forward_clearance(points):
    """(clear_m, detail) for the forward corridor; clear_m 0.0 when nothing can be verified."""
    coef = ground_plane(points)
    if coef is None:
        return 0.0, {"reason": "floor_not_observed"}
    corridor = points[(np.abs(points[:, 1]) < HALF_WIDTH_M) & (points[:, 0] > NEAR_LIMIT_M) &
                      (points[:, 0] < MAX_RANGE_M)]
    dz = height_above_ground(corridor, coef)
    floor = corridor[np.abs(dz) < 0.06]
    obstacle = corridor[(dz > OBSTACLE_MIN_DZ) & (dz < OBSTACLE_MAX_DZ)]
    edges = np.arange(NEAR_LIMIT_M, MAX_RANGE_M + 1e-9, BIN_M)
    counts = np.histogram(floor[:, 0], edges)[0] + np.histogram(obstacle[:, 0], edges)[0]
    observed_to = NEAR_LIMIT_M
    for i, n in enumerate(counts):
        if n < MIN_BIN_POINTS:
            break
        observed_to = float(edges[i + 1])
    nearest = float(np.sort(obstacle[:, 0])[OBSTACLE_RANK - 1]) if len(obstacle) >= OBSTACLE_RANK else math.inf
    limit = min(nearest, observed_to)
    clear = max(0.0, limit - FRONT_M - MARGIN_M)
    return clear, {"ground_coef": [round(float(c), 4) for c in coef], "observed_to_m": round(observed_to, 3),
                   "nearest_obstacle_m": None if math.isinf(nearest) else round(nearest, 3),
                   "floor_points": int(len(floor)), "obstacle_points": int(len(obstacle))}


def ground_z_at_front(points):
    """Ground height for the audited front_surface_x (a single z), taken 1 m ahead."""
    coef = ground_plane(points)
    return GROUND_Z_NOMINAL if coef is None else float(coef[0] + coef[1] * 1.0)


class OdomTrack:
    """Latest odom pose with its host receipt time and robot boot epoch."""

    def __init__(self, monotonic=time.monotonic, history=400):
        self.clock = monotonic
        self.lock = threading.Lock()
        self.latest = None  # (t_host, stamp_robot, x, y, yaw)
        self.history = deque(maxlen=history)
        self.offset0 = None
        self.generation = 0
        self.last_stamp = None

    def update(self, stamp_robot, x, y, yaw, t_host=None):
        t_host = self.clock() if t_host is None else t_host
        with self.lock:
            offset = t_host - stamp_robot
            if (self.offset0 is None or abs(offset - self.offset0) > EPOCH_STEP_S or
                    (self.last_stamp is not None and stamp_robot < self.last_stamp)):
                if self.offset0 is not None:
                    self.generation += 1
                    self.history.clear()
                self.offset0 = offset
            self.last_stamp = stamp_robot
            self.latest = (t_host, stamp_robot, x, y, yaw)
            self.history.append(self.latest)

    def epoch(self):
        return None if self.offset0 is None else f"go2_odom_boot:{self.offset0:.0f}:g{self.generation}"

    def pose_at(self, t_host, max_gap=0.05):
        with self.lock:
            items = list(self.history)
        if not items:
            return None
        best = min(items, key=lambda item: abs(item[0] - t_host))
        return best if abs(best[0] - t_host) <= max_gap else None

    def snapshot(self):
        with self.lock:
            return self.latest, self.epoch(), self.offset0


class ScanWindow:
    """Recent cloud_base scans with the odom pose at receipt."""

    def __init__(self, odom, monotonic=time.monotonic):
        self.odom, self.clock = odom, monotonic
        self.lock = threading.Lock()
        self.scans = deque()  # (t_host, pose(x,y,yaw), epoch, xyz)

    def add(self, xyz, t_host=None):
        t_host = self.clock() if t_host is None else t_host
        match = self.odom.pose_at(t_host)
        epoch = self.odom.epoch()
        if match is None or epoch is None:
            return False
        with self.lock:
            self.scans.append((t_host, match[2:5], epoch, np.asarray(xyz, dtype=float)))
            while self.scans and t_host - self.scans[0][0] > WINDOW_S:
                self.scans.popleft()
        return True

    def current(self):
        """(points in the current base frame, newest scan time, epoch) or None."""
        latest, epoch, _ = self.odom.snapshot()
        with self.lock:
            scans = list(self.scans)
        if latest is None or not scans:
            return None
        now_pose = latest[2:5]
        same = [s for s in scans if s[2] == epoch]
        if not same or self.clock() - same[-1][0] > WINDOW_S:
            return None
        points = np.concatenate([to_current_base(xyz, pose, now_pose) for _, pose, _, xyz in same])
        return points, same[-1][0], epoch


class CameraEpoch:
    """Newest camera observation in the run directory (bounded, polled by a background thread)."""

    def __init__(self, observation_root):
        self.dir = Path(observation_root) / "session" / "observations"
        self.lock = threading.Lock()
        self.latest = None
        self.last_name = ""

    def poll(self):
        try:
            names = [e.name for e in os.scandir(self.dir) if e.name.endswith(".vlnobs.json")]
        except OSError:
            return
        newer = sorted(n for n in names if n > self.last_name)
        if not newer:
            return
        try:
            meta = json.loads((self.dir / newer[-1]).read_text())
        except (OSError, ValueError):
            return
        with self.lock:
            self.last_name = newer[-1]
            self.latest = {"clock_epoch": meta.get("clock_epoch"),
                           "monotonic_mapping": meta.get("monotonic_mapping") or ""}

    def get(self):
        with self.lock:
            return None if self.latest is None else dict(self.latest)


def capture_bound(camera_meta, boot_offset, config=CAMERA):
    """(bound, basis) only when the measured evidence applies to this epoch and boot."""
    if camera_meta is None or not camera_meta.get("clock_epoch"):
        return None, None
    if config["bound_s"] is None or config["boot_offset_s"] is None or not config["basis"]:
        return None, None
    if f"configured camera latency {config['latency_config_s']:.3f} s" not in camera_meta["monotonic_mapping"]:
        return None, None
    if boot_offset is None or abs(boot_offset - config["boot_offset_s"]) > config["boot_tolerance_s"]:
        return None, None
    return float(config["bound_s"]), config["basis"]


class Guards:
    """Bounded, thread-safe cache. snapshot() never blocks on DDS, RPC or disk."""

    def __init__(self, *, monitor, capability, monotonic, observation_root, front_surface_x, fuse_progress,
                 compute_period=0.05):
        self.monitor, self.capability, self.clock = monitor, capability, monotonic
        self.front_surface_x, self.fuse_progress = front_surface_x, fuse_progress
        self.odom = OdomTrack(monotonic)
        self.scans = ScanWindow(self.odom, monotonic)
        self.camera = CameraEpoch(observation_root)
        self.lock = threading.Lock()
        self.cached = None
        self.detail = None
        self.stop = threading.Event()
        self.period = compute_period

    def start(self):
        threading.Thread(target=self._loop, daemon=True, name="supervised-guards").start()
        return self

    def _loop(self):
        last_camera = 0.0
        while not self.stop.wait(self.period):
            try:
                if self.clock() - last_camera > 0.5:
                    self.camera.poll()
                    last_camera = self.clock()
                self.compute()
            except Exception as error:  # keep the cache explicit, never stale-valid
                with self.lock:
                    self.cached, self.detail = None, {"error": f"{type(error).__name__}: {error}"}

    def compute(self):
        latest, epoch, offset = self.odom.snapshot()
        if latest is None:
            with self.lock:
                self.cached = None
            return
        t_pose, _, x, y, yaw = latest
        pose = {"frame": POSE_FRAME, "robot_epoch": epoch, "monotonic_s": t_pose, "x_m": x, "y_m": y, "yaw_rad": yaw}
        window = self.scans.current()
        if window is None:
            geometry = {"available": False, "monotonic_s": t_pose, "frame": POSE_FRAME, "robot_epoch": epoch,
                        "source": "cloud_base window unavailable", "directions": [],
                        "forward_clear_m": 0.0, "turn_clear_rad": 0.0}
            detail = {"reason": "no_current_scans"}
        else:
            points, t_scan, _ = window
            clear, detail = forward_clearance(points)
            geometry = {"available": clear > 0.0, "monotonic_s": t_scan, "frame": POSE_FRAME, "robot_epoch": epoch,
                        "source": (f"cloud_base {WINDOW_S:.1f} s window, corridor +/-{HALF_WIDTH_M} m, "
                                   f"front {FRONT_M} m + margin {MARGIN_M} m; turns unverified (rear blind)"),
                        "directions": ["forward"] if clear > 0.0 else [],
                        "forward_clear_m": round(clear, 4), "turn_clear_rad": 0.0}
            if self.capability.get("motion_profile") == PROFILE and "ground_coef" in detail:
                radius = np.hypot(points[:, 0], points[:, 1])
                part = points[(radius >= NEAR_LIMIT_M) & (radius <= DISC_RADIUS_M)]
                height = height_above_ground(part, detail["ground_coef"])
                geometry["observed_turn_disc"] = {
                    "radius_m": DISC_RADIUS_M, "floor_returns": int(np.sum(np.abs(height) < .06)),
                    "obstacle_returns": int(np.sum((height > OBSTACLE_MIN_DZ) & (height < OBSTACLE_MAX_DZ)))}
        bound, basis = capture_bound(self.camera.get(), offset)
        camera = self.camera.get()
        snap = {"pose": pose, "geometry": geometry, "capture_uncertainty_bound_s": bound,
                "capture_clock_epoch": None if camera is None else camera["clock_epoch"],
                "capture_bound_basis": basis, "geometry_detail": detail}
        with self.lock:
            self.cached = snap

    def snapshot(self):
        with self.lock:
            if self.cached is None:
                raise RuntimeError("guards_unavailable")
            return json.loads(json.dumps(self.cached))

    def progress(self, action):
        latest, epoch0, _ = self.odom.snapshot()
        if latest is None or not 0 <= self.clock()-latest[0] <= .25:
            return lambda: None
        _, _, x0, y0, yaw0 = latest
        if action["type"] == "turn":
            sign = 1 if action["direction"] == "left" else -1

            effective = self.capability.get("motion_profile") == PROFILE

            def imu_yaw():
                age = self.monitor.state_age()
                rpy = self.monitor.snapshot().get("rpy")
                if age is None or not 0 <= age <= .25 or not rpy or len(rpy) != 3 or not math.isfinite(rpy[2]):
                    return None
                return rpy[2]

            start_imu = imu_yaw() if effective else None

            def turn_progress():
                now, epoch, _ = self.odom.snapshot()
                if now is None or epoch != epoch0 or not 0 <= self.clock() - now[0] <= 0.25:
                    return None
                if effective:
                    current_imu = imu_yaw()
                    if start_imu is None or current_imu is None:
                        return None
                    turn_progress.evidence = {"source": "relative_imu_yaw", "odom_rad": sign*wrap(now[4]-yaw0)}
                    return sign * wrap(current_imu - start_imu)
                return sign * wrap(now[4] - yaw0)

            return turn_progress
        window = self.scans.current()
        start_x = None if window is None else self.front_surface_x(window[0], ground_z_at_front(window[0]))

        def forward_progress():
            now, epoch, _ = self.odom.snapshot()
            if now is None or epoch != epoch0 or not 0 <= self.clock() - now[0] <= 0.25:
                return None
            odom = (now[2] - x0) * math.cos(yaw0) + (now[3] - y0) * math.sin(yaw0)
            lidar = None
            if start_x is not None:
                current = self.scans.current()
                if current is not None:
                    x = self.front_surface_x(current[0], ground_z_at_front(current[0]))
                    lidar = None if x is None else start_x - x
            fused, lidar_used = self.fuse_progress(lidar, odom)
            forward_progress.evidence = {"source": "lidar_and_odom" if lidar_used else "odom", "lidar_m": lidar, "odom_m": odom}
            return fused

        return forward_progress


def create_guards(*, monitor, capability, monotonic, observation_root):
    """Contract entry point. Runs only inside the armed executor child (hardware environment)."""
    from unitree_sdk2py.core.channel import ChannelSubscriber
    from unitree_sdk2py.idl.nav_msgs.msg.dds_ import Odometry_
    from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_

    from go2_control.sdk_routes import front_surface_x, fuse_progress
    from go2_perception.adapters.go2.pointcloud import frame_from_msg

    guards = Guards(monitor=monitor, capability=capability, monotonic=monotonic, observation_root=observation_root,
                    front_surface_x=front_surface_x, fuse_progress=fuse_progress)

    def on_odom(m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        stamp = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        guards.odom.update(stamp, float(p.x), float(p.y), yaw_from_quaternion(q.x, q.y, q.z, q.w))

    def on_cloud(m):
        guards.scans.add(frame_from_msg(m, "rt/utlidar/cloud_base", time.time()).xyz)

    odom_sub = ChannelSubscriber("rt/utlidar/robot_odom", Odometry_)
    odom_sub.Init(on_odom, 10)
    cloud_sub = ChannelSubscriber("rt/utlidar/cloud_base", PointCloud2_)
    cloud_sub.Init(on_cloud, 10)
    guards._subs = [odom_sub, cloud_sub]
    return guards.start()
