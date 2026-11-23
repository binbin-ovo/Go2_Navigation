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

from go2_perception.vln.motion_profile import DISC_RADIUS_M, PROFILE, TURN_DRIFT_M

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
DISC_SECTOR_DEG = 30  # turn-disc floor refit sectors
# Body rectangle about base_link for the swept-turn veto: head 0.40 m ahead (FRONT_M), 0.70 m long,
# legs about 0.20 m to each side. Turns are at most 30 deg; 35 covers the measured 32 deg peak.
BODY_FRONT_M, BODY_REAR_M, BODY_HALF_WIDTH_M = 0.40, 0.30, 0.20
TURN_SWEEP_DEG = 35
MAX_RANGE_M = 3.0
GROUND_Z_NOMINAL = -0.325  # base-frame ground measured on this robot (2026-11-15)
EPOCH_STEP_S = 1.0
POSE_FRAME = "go2_robot_odom"

# Filled from the measured evidence; None keeps the capture bound unavailable (HOLD).
CAMERA = {
    "latency_config_s": 0.048,  # what the camera bridge subtracts from receipt (configs/go2_vln_sensors.yaml)
    # Re-measured for the robot boot seen at 11:57 on 2026-11-23 (earlier records: 0.184 s at boot_offset
    # -1766613707.168, 0.140 s at -1766856813.961, 0.159 s at -1766897811.159, 0.148 s at -1766947283.136 and
    # 0.172 s at -1766903986.310 belong to earlier boots and are not reused). Screen-flash test
    # (hardware/capture_latency.py): 39 switches, switch-to-receipt median 0.127 s, max 0.185 s (includes display
    # lag and up to one 0.07 s frame interval); bound = max - 0.048 configured + 0.020 margin.
    "bound_s": 0.157,  # validated extra age on top of the configured latency
    "boot_offset_s": -1766552958.008,  # host_monotonic - robot_odom_stamp on the measured boot
    "boot_tolerance_s": 5.0,
    "basis": ("evidence/hw_readiness/capture_latency_20261123_115758.json sha256 "
              "9de4777357186c01afa9b921bd1e25df7b17a4c952a7d63a61ab88b6a39b34d9"),  # private, git-ignored
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


def body_radius(bearing_rad):
    """Distance from base_link to the edge of the body rectangle along a bearing."""
    c, s = np.cos(bearing_rad), np.sin(bearing_rad)
    along = np.where(c > 0, BODY_FRONT_M / np.maximum(c, 1e-9), BODY_REAR_M / np.maximum(-c, 1e-9))
    return np.minimum(along, BODY_HALF_WIDTH_M / np.maximum(np.abs(s), 1e-9))


def turn_reach(bearing_deg):
    """Farthest body point on each bearing during an in-place turn of up to TURN_SWEEP_DEG, plus drift."""
    offsets = np.radians(np.arange(-TURN_SWEEP_DEG, TURN_SWEEP_DEG + 0.5, 1.0))
    bearings = np.radians(np.asarray(bearing_deg, float))[..., None] - offsets
    return body_radius(bearings).max(axis=-1) + TURN_DRIFT_M


# Reach depends only on bearing; a 1-degree table keeps the guard update near 1 ms instead of ~70 ms
# (per-point evaluation over 71 turn angles delayed snapshots: stale_or_future_pose, 2026-11-23 N7NzcG).
_REACH_BY_DEGREE = turn_reach(np.arange(-180, 181))


def turn_disc(points, coef):
    """(published disc fields, diagnostics): returns in the swept disc, measured from the floor beside the robot.

    The forward plane models the LiDAR's range bias only along x. At the sides (range 0.3-0.65 m,
    x near 0) the bare floor reads +0.05 to +0.07 m above it with +/-0.05 m spread (2026-11-22),
    so every disc held "obstacles". Each sector refits floor height against range from floor-like
    returns out to 1 m. The correction is clipped to [-0.05, +0.10] m, so an object filling a
    sector still shows. A sector needs 15 floor returns for its own fit, uses the fit pooled over
    the sides with 8-14, and keeps the forward plane with fewer. A return vetoes the turn only
    inside the body's swept reach (turn_reach): an in-place turn reaches 0.55 m only near the
    corners and about 0.33 m beside the robot.
    """
    radius = np.hypot(points[:, 0], points[:, 1])
    bearing = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
    height = height_above_ground(points, coef)
    local = height.copy()
    ring = (radius >= NEAR_LIMIT_M) & (radius <= DISC_RADIUS_M)
    floorlike = (radius >= NEAR_LIMIT_M) & (radius <= 1.0) & (height > -0.10) & (height < 0.12)

    def range_fit(context, minimum):
        keep, fit = context, None
        for tolerance in (0.06, 0.05, 0.03):
            if keep.sum() < minimum:
                return None
            fit = np.linalg.lstsq(np.column_stack([np.ones(keep.sum()), radius[keep]]), height[keep], rcond=None)[0]
            keep = context & (np.abs(height - fit[0] - fit[1] * radius) < tolerance)
        return fit

    # Sparse sectors fit from 15 floor returns; below that they borrow the fit pooled over the sides.
    pooled = range_fit(floorlike & (np.abs(bearing) >= 30), 30)
    obstacle = np.zeros(len(points), bool)
    corrected = pooled_used = 0
    for start in range(-180, 180, DISC_SECTOR_DEG):
        sector = (bearing >= start) & (bearing < start + DISC_SECTOR_DEG)
        context = sector & floorlike
        fit = range_fit(context, 15)
        if fit is None and pooled is not None and context.sum() >= 8:
            fit, pooled_used = pooled, pooled_used + 1
        cell = sector & ring
        if fit is not None:
            local[cell] = height[cell] - np.clip(fit[0] + fit[1] * radius[cell], -0.05, 0.10)
            corrected += 1
        above = cell & (local > OBSTACLE_MIN_DZ) & (local < OBSTACLE_MAX_DZ)
        above &= radius <= _REACH_BY_DEGREE[np.rint(bearing).astype(int) + 180]  # where the turning body reaches
        if above.sum() >= OBSTACLE_RANK:  # Same isolated-return rejection as the forward corridor.
            obstacle |= above
    disc = {"radius_m": DISC_RADIUS_M, "floor_returns": int(np.sum(ring & (np.abs(local) < .06))),
            "obstacle_returns": int(obstacle.sum())}  # the published proposal fields only
    diagnostics = {"unfiltered_obstacle_returns": int(np.sum(ring & (local > OBSTACLE_MIN_DZ) & (local < OBSTACLE_MAX_DZ))),
                   "floor_model": f"sector_local_range_fit/{DISC_SECTOR_DEG}deg", "sectors_refit": corrected, "sectors_pooled": pooled_used}
    return disc, diagnostics


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
        self.raw_pose = None  # optional original (x, y, z, qx, qy, qz, qw)
        self.history = deque(maxlen=history)
        self.offset0 = None
        self.generation = 0
        self.last_stamp = None

    def update(self, stamp_robot, x, y, yaw, t_host=None, *, raw_pose=None):
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
            self.raw_pose = raw_pose
            self.latest = (t_host, stamp_robot, x, y, yaw)
            self.history.append(self.latest)

    def epoch(self):
        return None if self.offset0 is None else f"go2_odom_boot:{self.offset0:.0f}:g{self.generation}"

    def pose_with_epoch_at(self, t_host, max_gap=0.05):
        with self.lock:
            best = min(self.history, key=lambda item: abs(item[0] - t_host), default=None)
            return (best if best and abs(best[0] - t_host) <= max_gap else None), self.epoch()

    def pose_at(self, t_host, max_gap=0.05):
        return self.pose_with_epoch_at(t_host, max_gap)[0]

    def snapshot(self, *, evidence=None):
        with self.lock:
            epoch = self.epoch()
            if evidence is not None and self.latest is not None:
                received, stamp, x, y, yaw = self.latest
                evidence.update(frame=POSE_FRAME, robot_epoch=epoch, stamp_robot_s=stamp,
                                received_monotonic_s=received, x_m=x, y_m=y, yaw_rad=yaw,
                                position_xyz_q_xyzw=self.raw_pose)
            return self.latest, epoch, self.offset0


class ScanWindow:
    """Recent cloud_base scans with the odom pose at receipt."""

    def __init__(self, odom, monotonic=time.monotonic):
        self.odom, self.clock = odom, monotonic
        self.lock = threading.Lock()
        self.scans = deque()  # (t_host, matched odom sample, epoch, xyz, robot header stamp)

    def add(self, xyz, t_host=None, *, stamp_robot_s=None):
        t_host = self.clock() if t_host is None else t_host
        match, epoch = self.odom.pose_with_epoch_at(t_host)
        if match is None or epoch is None:
            return False
        with self.lock:
            self.scans.append((t_host, match, epoch, np.asarray(xyz, dtype=float), stamp_robot_s))
            while self.scans and t_host - self.scans[0][0] > WINDOW_S:
                self.scans.popleft()
        return True

    def current(self, *, evidence=None):
        """(points in the current base frame, newest scan time, epoch) or None."""
        latest, epoch, _ = self.odom.snapshot()
        with self.lock:
            scans = list(self.scans)
        if latest is None or not scans:
            return None
        now_pose = latest[2:5]
        same = [s for s in scans if s[2] == epoch]
        if not same:
            return None
        if evidence is not None:
            evidence.update(robot_epoch=epoch, scan_stamps_robot_s=[s[4] for s in same],
                            scan_receipts_monotonic_s=[s[0] for s in same],
                            scan_pose_stamps_robot_s=[s[1][1] for s in same],
                            scan_pose_receipts_monotonic_s=[s[1][0] for s in same],
                            scan_poses_xy_yaw=[s[1][2:5] for s in same], current_pose_xy_yaw=now_pose,
                            pose_stamp_robot_s=latest[1], pose_received_monotonic_s=latest[0])
        if self.clock() - same[-1][0] > WINDOW_S:
            return None
        points = np.concatenate([to_current_base(xyz, sample[2:5], now_pose) for _, sample, _, xyz, _ in same])
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
        odom_sample = {}
        latest, epoch, offset = self.odom.snapshot(evidence=odom_sample)
        if latest is None:
            with self.lock:
                self.cached, self.detail = None, {"reason": "odometry_unavailable"}
            return
        t_pose, _, x, y, yaw = latest
        pose = {"frame": POSE_FRAME, "robot_epoch": epoch, "monotonic_s": t_pose, "x_m": x, "y_m": y, "yaw_rad": yaw}
        scan_sources = {}
        window = self.scans.current(evidence=scan_sources)
        if window is None or window[2] != epoch:
            geometry = {"available": False, "monotonic_s": t_pose, "frame": POSE_FRAME, "robot_epoch": epoch,
                        "source": "cloud_base window unavailable", "directions": [],
                        "forward_clear_m": 0.0, "turn_clear_rad": 0.0}
            detail = {"reason": "no_current_scans" if window is None else "scan_pose_epoch_mismatch"}
        else:
            points, t_scan, _ = window
            clear, detail = forward_clearance(points)
            geometry = {"available": clear > 0.0, "monotonic_s": t_scan, "frame": POSE_FRAME, "robot_epoch": epoch,
                        "source": (f"cloud_base {WINDOW_S:.1f} s window, corridor +/-{HALF_WIDTH_M} m, "
                                   f"front {FRONT_M} m + margin {MARGIN_M} m; turns unverified (rear blind)"),
                        "directions": ["forward"] if clear > 0.0 else [],
                        "forward_clear_m": round(clear, 4), "turn_clear_rad": 0.0}
            if self.capability.get("motion_profile") == PROFILE and "ground_coef" in detail:
                geometry["observed_turn_disc"], detail["turn_disc"] = turn_disc(points, detail["ground_coef"])
        camera = self.camera.get()
        bound, basis = capture_bound(camera, offset)
        snap = {"pose": pose, "geometry": geometry, "capture_uncertainty_bound_s": bound,
                "capture_clock_epoch": None if camera is None else camera["clock_epoch"],
                "capture_bound_basis": basis, "geometry_detail": detail, "scan_sources": scan_sources or None,
                "odometry": odom_sample}
        with self.lock:
            self.cached, self.detail = snap, None

    def snapshot(self):
        with self.lock:
            if self.cached is None:
                cause = (self.detail or {}).get("error") or (self.detail or {}).get("reason")
                raise RuntimeError("guards_unavailable" if cause is None else f"guards_unavailable: {cause}")
            return json.loads(json.dumps(self.cached))

    def progress(self, action):
        reference_odom = {}
        latest, epoch0, _ = self.odom.snapshot(evidence=reference_odom)
        if latest is None or not 0 <= self.clock()-latest[0] <= .25:
            def unavailable():
                return None
            unavailable.evidence = {"fusion_decision": "reference_odometry_unavailable",
                                    "reference_odometry": reference_odom or None,
                                    "odom_m": None, "lidar_m": None, "fused_m": None}
            return unavailable
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
                odom_sample = {}
                now, epoch, _ = self.odom.snapshot(evidence=odom_sample)
                turn_progress.evidence = {"reference_odometry": reference_odom, "odometry": odom_sample or None,
                                          "source": None, "odom_rad": None}
                if now is None or epoch != epoch0 or not 0 <= self.clock() - now[0] <= 0.25:
                    return None
                if effective:
                    current_imu = imu_yaw()
                    if start_imu is None or current_imu is None:
                        return None
                    turn_progress.evidence.update(source="relative_imu_yaw", odom_rad=sign*wrap(now[4]-yaw0),
                                                  reference_imu_yaw_rad=start_imu, imu_yaw_rad=current_imu)
                    return sign * wrap(current_imu - start_imu)
                return sign * wrap(now[4] - yaw0)

            return turn_progress
        reference_scans = {}
        window = self.scans.current(evidence=reference_scans)
        start_x = reference_ground_z = None
        if window is not None and window[2] == epoch0:
            reference_ground_z = ground_z_at_front(window[0])
            start_x = self.front_surface_x(window[0], reference_ground_z)

        def forward_progress():
            odom_sample = {}
            now, epoch, _ = self.odom.snapshot(evidence=odom_sample)
            # A new object per call keeps saved samples immutable, even when a
            # subsequent read is stale or unavailable.
            evidence = {"reference_odometry": reference_odom, "odometry": odom_sample or None,
                        "reference_scans": reference_scans or None, "scans": None,
                        "reference_surface_x_m": start_x, "surface_x_m": None,
                        "reference_ground_z_m": reference_ground_z, "ground_z_m": None,
                        "lidar_minus_odom_m": None,
                        "odom_m": None, "lidar_m": None, "fused_m": None,
                        "lidar_accepted": None, "selected_input": None, "source": None}
            forward_progress.evidence = evidence
            if now is None or epoch != epoch0 or not 0 <= self.clock() - now[0] <= .25:
                evidence["fusion_decision"] = ("odometry_epoch_changed" if epoch != epoch0 else
                                               "odometry_missing_or_stale")
                return None
            odom = (now[2] - x0) * math.cos(yaw0) + (now[3] - y0) * math.sin(yaw0)
            evidence["odom_m"] = odom
            lidar = None
            lidar_state = "reference_surface_unavailable" if window is not None else "reference_window_unavailable"
            if window is not None and window[2] != epoch0:
                evidence["fusion_decision"] = "scan_epoch_changed"
                return None
            if start_x is not None:
                scan_sources = {}
                current = self.scans.current(evidence=scan_sources)
                lidar_state = "scan_window_unavailable"
                evidence["scans"] = scan_sources or None
                if current is not None:
                    if current[2] != epoch0:
                        evidence["fusion_decision"] = "scan_epoch_changed"
                        return None
                    evidence["ground_z_m"] = ground_z_at_front(current[0])
                    x = self.front_surface_x(current[0], evidence["ground_z_m"])
                    evidence["surface_x_m"] = x
                    lidar = None if x is None else start_x - x
                    lidar_state = "current_surface_unavailable" if x is None else "available"
            fused, lidar_used = self.fuse_progress(lidar, odom)
            # The canonical routine returns max(lidar, odom) when it accepts the
            # pair. Acceptance alone does not identify which estimate it returned.
            decision = ("accepted_max" if lidar_used else
                        "disagreement_rejected" if lidar is not None else lidar_state)
            selected = ("equal" if lidar_used and lidar == odom else
                        "lidar" if lidar_used and fused == lidar else "odom")
            evidence.update(source="lidar_and_odom" if lidar_used else "odom", lidar_m=lidar, fused_m=fused,
                            lidar_minus_odom_m=None if lidar is None else lidar-odom,
                            lidar_accepted=lidar_used, fusion_decision=decision, selected_input=selected)
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
        guards.odom.update(stamp, float(p.x), float(p.y), yaw_from_quaternion(q.x, q.y, q.z, q.w),
                           raw_pose=tuple(map(float, (p.x, p.y, p.z, q.x, q.y, q.z, q.w))))

    def on_cloud(m):
        frame = frame_from_msg(m, "rt/utlidar/cloud_base", time.time())
        guards.scans.add(frame.xyz, stamp_robot_s=frame.stamp)

    odom_sub = ChannelSubscriber("rt/utlidar/robot_odom", Odometry_)
    odom_sub.Init(on_odom, 10)
    cloud_sub = ChannelSubscriber("rt/utlidar/cloud_base", PointCloud2_)
    cloud_sub.Init(on_cloud, 10)
    guards._subs = [odom_sub, cloud_sub]
    return guards.start()
