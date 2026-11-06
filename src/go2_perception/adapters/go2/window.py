"""Ego-motion-compensated short sliding window of body-frame LiDAR clouds.

Each frame is mapped through odometry into the newest frame's ``base_link``. This removes
the robot's motion, not an object's: a moving obstacle still smears by speed x window, so
keep the window short and retain raw frames. Frames without a usable pose are refused
instead of being merged uncompensated. Nothing is written to a persistent map.
"""

from bisect import bisect_left
from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


class PoseBuffer:
    """Timestamped T_odom_base samples; interpolates only between close neighbours."""

    def __init__(self, max_age_s=5.0, max_gap_s=0.1):
        self.max_age_s, self.max_gap_s = max_age_s, max_gap_s
        self._t, self._p, self._q = deque(), deque(), deque()

    def add(self, t, position, q_xyzw):
        if self._t and t <= self._t[-1]:
            return
        self._t.append(t)
        self._p.append(np.asarray(position, dtype=float))
        self._q.append(np.asarray(q_xyzw, dtype=float))
        while self._t and self._t[0] < t - self.max_age_s:
            self._t.popleft(), self._p.popleft(), self._q.popleft()

    def lookup(self, t):
        """Return 4x4 T_odom_base at t, or None when t is not bracketed by nearby poses."""
        times = list(self._t)
        i = bisect_left(times, t)
        if i == 0 or i == len(times):
            if i < len(times) and times[i] == t:
                return _matrix(self._p[i], Rotation.from_quat(self._q[i]))
            return None
        t0, t1 = times[i - 1], times[i]
        if t1 - t0 > self.max_gap_s:
            return None
        alpha = (t - t0) / (t1 - t0)
        p = (1 - alpha) * self._p[i - 1] + alpha * self._p[i]
        slerp = Slerp([0, 1], Rotation.from_quat([self._q[i - 1], self._q[i]]))
        return _matrix(p, slerp(alpha))


def _matrix(p, rotation):
    m = np.eye(4)
    m[:3, :3] = rotation.as_matrix()
    m[:3, 3] = p
    return m


class SlidingWindowAccumulator:
    """Merge recent frames into the newest body frame using poses from a PoseBuffer."""

    def __init__(self, poses, window_s=0.3, max_frames=10):
        if window_s <= 0 or max_frames < 1:
            raise ValueError("Invalid accumulator configuration")
        self.poses, self.window_s, self.max_frames = poses, window_s, max_frames
        self._frames = deque()
        self.refused_no_pose = 0

    def add(self, xyz, t):
        """Add a body-frame cloud. Return False, without storing it, if no pose exists."""
        pose = self.poses.lookup(t)
        if pose is None:
            self.refused_no_pose += 1
            return False
        if self._frames and t <= self._frames[-1][0]:
            return False
        self._frames.append((t, pose, np.asarray(xyz, dtype=float).copy()))
        newest = self._frames[-1][0]
        while self._frames and (
            newest - self._frames[0][0] > self.window_s
            or len(self._frames) > self.max_frames
        ):
            self._frames.popleft()
        return True

    def merged(self):
        """Return (points in newest base_link, newest time, frames merged, span seconds)."""
        if not self._frames:
            return np.empty((0, 3)), None, 0, 0.0
        newest_t, newest_pose, _ = self._frames[-1]
        inverse = np.linalg.inv(newest_pose)
        parts = []
        for _, pose, xyz in self._frames:
            relative = inverse @ pose
            parts.append(xyz @ relative[:3, :3].T + relative[:3, 3])
        return (
            np.concatenate(parts),
            newest_t,
            len(self._frames),
            newest_t - self._frames[0][0],
        )
