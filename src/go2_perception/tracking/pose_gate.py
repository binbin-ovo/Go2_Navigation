"""Hold LiDAR frames until odometry can pose them; skip, never invent, otherwise.

A frame is resolved only once odometry at or after its timestamp has arrived, so a frame
that precedes the first odometry sample waits briefly and is then skipped instead of being
extrapolated backward. Skips keep their reason; raw recordings are never modified.
"""

from collections import deque
from dataclasses import dataclass
import numpy as np

NO_ODOMETRY = "no_odometry"
BEFORE_FIRST_ODOMETRY = "before_first_odometry"
ODOMETRY_GAP = "odometry_gap"
AFTER_LAST_ODOMETRY = "after_last_odometry"
ODOMETRY_WAIT_TIMEOUT = "odometry_wait_timeout"


@dataclass
class PoseAlignmentConfig:
    max_wait_s: float = 0.2  # live only: wait for bracketing odometry, in the push clock
    start_hold_s: float = 0.0  # 0 = never pose frames earlier than the first odometry sample
    start_hold_justification: str | None = None  # required whenever start_hold_s > 0
    reset_after_gap_s: float = 0.5  # tracker restarts if posed frames are further apart

    def __post_init__(self):
        for key in ("max_wait_s", "start_hold_s", "reset_after_gap_s"):
            value = getattr(self, key)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {key}")
        if self.max_wait_s == 0 or self.reset_after_gap_s == 0:
            raise ValueError("max_wait_s and reset_after_gap_s must be positive")
        if self.start_hold_s > 0 and not (self.start_hold_justification or "").strip():
            raise ValueError("start_hold_s > 0 needs an explicit start_hold_justification")


@dataclass
class SkippedFrame:
    timestamp: float
    reason: str


class PoseGate:
    """FIFO of frames awaiting T_odom_base from ``poses.lookup(t)`` (None when unbracketed)."""

    def __init__(self, poses, config=None):
        self.poses = poses
        self.config = config or PoseAlignmentConfig()
        self.first_pose_t = self.latest_pose_t = None
        self.skipped = []
        self.held = 0  # frames posed by the explicit start hold
        self._pending = deque()

    def note_pose(self, t):
        """Record that odometry stamped ``t`` is now available from ``poses``."""
        if self.first_pose_t is None:
            self.first_pose_t = t
        self.latest_pose_t = t if self.latest_pose_t is None else max(self.latest_pose_t, t)

    def push(self, t, item, wait_start=None):
        self._pending.append((t, item, wait_start))

    def lookup(self, t):
        matrix = self.poses.lookup(t)
        first = self.first_pose_t
        if matrix is None and first is not None and 0 < first - t <= self.config.start_hold_s:
            return self.poses.lookup(first)  # explicit, justified zero-order hold only
        return matrix

    def release(self, now=None, final=False):
        """Yield ``(t, item, T_odom_base)`` in arrival order; record skips in ``skipped``.

        ``now`` (the clock of ``wait_start``, e.g. host receipt) enables the live timeout.
        ``final`` declares that no more odometry will arrive, as at the end of a recording.
        """
        while self._pending:
            t, item, wait_start = self._pending[0]
            if self.latest_pose_t is None or self.latest_pose_t < t:
                if final:
                    reason = NO_ODOMETRY if self.latest_pose_t is None else AFTER_LAST_ODOMETRY
                elif (
                    now is not None
                    and wait_start is not None
                    and now - wait_start > self.config.max_wait_s
                ):
                    reason = ODOMETRY_WAIT_TIMEOUT
                else:
                    return
                self._pending.popleft()
                self.skipped.append(SkippedFrame(t, reason))
                continue
            self._pending.popleft()
            matrix = self.lookup(t)
            if matrix is None:
                reason = BEFORE_FIRST_ODOMETRY if t < self.first_pose_t else ODOMETRY_GAP
                self.skipped.append(SkippedFrame(t, reason))
                continue
            if t < self.first_pose_t:
                self.held += 1
            yield t, item, matrix
