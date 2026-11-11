"""Incremental scans in odom, with original per-point acquisition provenance."""

from dataclasses import dataclass
from collections import deque
import numpy as np


def rigid(matrix):
    m = np.asarray(matrix, dtype=float)
    if (
        m.shape != (4, 4)
        or not np.isfinite(m).all()
        or not np.allclose(m[3], [0, 0, 0, 1])
        or not np.allclose(m[:3, :3].T @ m[:3, :3], np.eye(3), atol=1e-5)
        or np.linalg.det(m[:3, :3]) < 0.99
    ):
        raise ValueError("Finite rigid T_odom_base required")
    return m


@dataclass
class Accumulated:
    xyz: np.ndarray
    acquired: np.ndarray
    scan_ids: np.ndarray
    timestamp: float

    @property
    def ages(self):
        return self.timestamp - self.acquired

    def voxelized(self, size):
        if size < 0:
            raise ValueError("Nonnegative voxel size required")
        if size == 0 or not len(self.xyz):
            return self
        # Keep latest actual sample, not a centroid with an invented acquisition time.
        order = np.argsort(-self.acquired, kind="stable")
        _, first = np.unique(
            np.floor(self.xyz[order] / size).astype(np.int64), axis=0, return_index=True
        )
        keep = order[first]
        return Accumulated(
            self.xyz[keep], self.acquired[keep], self.scan_ids[keep], self.timestamp
        )


class TemporalAccumulator:
    def __init__(
        self,
        window_s=1.0,
        reset_gap_s=0.5,
        max_pose_jump_m=0.5,
        max_pose_angle_deg=45.0,
        max_points=300000,
    ):
        if (
            min(window_s, reset_gap_s, max_pose_jump_m, max_pose_angle_deg, max_points)
            <= 0
        ):
            raise ValueError("Invalid accumulation parameters")
        self.window_s = window_s
        self.reset_gap_s = reset_gap_s
        self.max_pose_jump_m = max_pose_jump_m
        self.max_pose_angle_deg = max_pose_angle_deg
        self.max_points = max_points
        self.epoch = 0
        self.frames = deque()
        self.clock = None
        self.last_t = None
        self.last_pose = None
        self.last_id = None
        self.reset_reason = None

    def clear(self, reason):
        self.frames.clear()
        self.epoch += 1
        self.last_t = None
        self.last_pose = None
        self.last_id = None
        self.reset_reason = reason

    def add(
        self,
        xyz,
        stamp,
        scan_id,
        pose_lookup,
        *,
        clock_domain,
        point_offsets=None,
        frame_id="base_link",
        deskew=False,
        instantaneous_point_frame=False,
        incremental=True,
        reset=False,
    ):
        if not incremental or frame_id != "base_link":
            raise ValueError(
                "Only incremental raw base_link scans accepted; no preaccumulated/leveled/odom clouds"
            )
        if not clock_domain or not np.isfinite(stamp):
            raise ValueError("Finite timestamp and declared clock required")
        scan_id = str(scan_id)
        if not scan_id:
            raise ValueError("Nonempty acquisition interval ID required")
        p = np.asarray(xyz, dtype=float)
        if p.ndim != 2 or p.shape[1] != 3:
            raise ValueError("Expected Nx3")
        offsets = (
            np.zeros(len(p))
            if point_offsets is None
            else np.asarray(point_offsets, dtype=float)
        )
        if (
            offsets.shape != (len(p),)
            or not np.isfinite(offsets).all()
            or (offsets < 0).any()
        ):
            raise ValueError("Finite nonnegative per-point second offsets required")
        if deskew and (point_offsets is None or not instantaneous_point_frame):
            raise ValueError(
                "Deskew requires verified instantaneous point-frame semantics and per-point timestamps"
            )
        acquired = stamp + offsets
        now = float(acquired.max()) if len(p) else float(stamp)
        midpoint = stamp + (float(offsets.min() + offsets.max()) / 2 if len(p) else 0.0)
        if self.clock is not None and clock_domain != self.clock:
            self.clear("clock_domain_changed")
        elif reset:
            self.clear("explicit_odometry_reset")
        elif self.last_t is not None and now < self.last_t:
            self.clear("clock_rewind")
        elif self.last_t is not None and now - self.last_t > self.reset_gap_s:
            self.clear("timestamp_gap")
        else:
            self.reset_reason = None
        self.clock = clock_domain
        if self.last_t == now or any(f[0] == scan_id for f in self.frames):
            raise ValueError("Duplicate scan/time")
        pose = pose_lookup(midpoint)
        end_pose = pose_lookup(now)
        if pose is None or end_pose is None:
            self.clear("missing_odometry")
            return None
        pose, end_pose = rigid(pose), rigid(end_pose)
        if self.last_pose is not None:
            angle = np.degrees(
                np.arccos(
                    np.clip(
                        (np.trace(self.last_pose[:3, :3].T @ end_pose[:3, :3]) - 1) / 2,
                        -1,
                        1,
                    )
                )
            )
            if (
                np.linalg.norm(end_pose[:3, 3] - self.last_pose[:3, 3])
                > self.max_pose_jump_m
                or angle > self.max_pose_angle_deg
            ):
                self.clear("odometry_discontinuity")
        valid = np.isfinite(p).all(1)
        p = p[valid]
        acquired = acquired[valid]
        if deskew:
            transformed = np.empty_like(p)
            # Per-point poses are explicit; never invent offsets or use a held pose across gaps.
            for t in np.unique(acquired):
                m = pose_lookup(float(t))
                if m is None:
                    self.clear("missing_point_odometry")
                    return None
                m = rigid(m)
                mask = acquired == t
                transformed[mask] = p[mask] @ m[:3, :3].T + m[:3, 3]
        else:
            transformed = p @ pose[:3, :3].T + pose[:3, 3]
        self.frames.append((str(scan_id), now, transformed, acquired))
        while self.frames and self.frames[0][1] < now - self.window_s:
            self.frames.popleft()
        while (
            len(self.frames) > 1
            and sum(len(v[2]) for v in self.frames) > self.max_points
        ):
            self.frames.popleft()
        if len(transformed) > self.max_points:
            self.clear("scan_exceeds_memory_limit")
            return None
        self.last_t = now
        self.last_pose = end_pose
        self.last_id = str(scan_id)
        return Accumulated(transformed, acquired, np.full(len(p), str(scan_id)), now)

    def snapshot(self, window_s=None, voxel_size=0.0):
        window = self.window_s if window_s is None else window_s
        if not 0 < window <= self.window_s:
            raise ValueError("Window outside retained horizon")
        now = self.last_t
        if now is None:
            return Accumulated(
                np.empty((0, 3)), np.empty(0), np.empty(0, dtype=str), 0.0
            )
        parts = []
        times = []
        ids = []
        for identity, _, p, t in self.frames:
            keep = t >= now - window
            parts.append(p[keep])
            times.append(t[keep])
            ids.extend([identity] * int(keep.sum()))
        return Accumulated(
            np.concatenate(parts),
            np.concatenate(times),
            np.asarray(ids, dtype=str),
            now,
        ).voxelized(voxel_size)
