"""Causal display state: what each emitted object looks like, and renderer-side memory.

Rules (docs/FSD_STYLE_VISUALIZATION.md):
- ``fresh``: status observed_filtered with an observed box from the current scan; solid.
- ``stale``: emitted but not supported by the current scan; dashed and fading with
  observation age, drawn at most ``stale_max_s`` (never longer than the producer emits it).
- ``ghost``: the producer dropped the object; the renderer may keep a dotted "unseen" footprint
  for ``ghost_hold_s``. It is labelled unknown, never drawn as an object or as free space.
- Motion colour: orange only for fresh MOVING, blue for fresh STATIONARY, otherwise gray.
- Nothing is extrapolated into a box; the optional constant-velocity path is a labelled dashed
  prediction with a widening uncertainty band.
- Renderer memory (trails, ghosts, robot path) is cleared when ``epoch`` changes.
Everything here uses only frames at or before the display time (causal).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .scene import SceneFrame, SceneObject

FRESH, STALE, GHOST = "fresh", "stale", "ghost"


@dataclass
class DisplayConfig:
    stale_max_s: float = 0.6  # draw stale boxes up to this observation age
    stale_min_alpha: float = 0.15  # opacity reached at stale_max_s
    ghost_hold_s: float = 1.0  # 0 disables renderer-held "unseen" footprints
    trail_s: float = 3.0
    cv_horizon_s: float = 1.5  # constant-velocity path length; 0 disables
    cv_step_s: float = 0.25
    min_arrow_speed: float = 0.05
    interpolate_pose: bool = False  # causal: displays at t - delay, never uses future frames
    interpolation_delay_s: float = 0.07


@dataclass
class DisplayItem:
    obj: SceneObject
    kind: str  # fresh / stale / ghost
    alpha: float
    colour_key: str  # MOVING / STATIONARY / UNCERTAIN
    trail: list = field(default_factory=list)  # [(x, y, z)] observed centres
    cv_path: np.ndarray | None = None  # (K, 3) predicted centres
    cv_sigma: np.ndarray | None = None  # (K,) 1-sigma radius from velocity covariance
    unseen_s: float = 0.0  # ghosts: time since the producer last emitted the object


def colour_key(obj: SceneObject, kind: str):
    if kind == FRESH and obj.motion_state in ("MOVING", "STATIONARY"):
        return obj.motion_state
    return "UNCERTAIN"


def stale_alpha(age, cfg: DisplayConfig):
    if cfg.stale_max_s <= 0:
        return cfg.stale_min_alpha
    f = min(max(age / cfg.stale_max_s, 0.0), 1.0)
    return 1.0 - f * (1.0 - cfg.stale_min_alpha)


def cv_prediction(obj: SceneObject, cfg: DisplayConfig):
    """Constant-velocity centres and 1-sigma radii; None unless fresh MOVING with measured velocity."""
    if cfg.cv_horizon_s <= 0 or obj.velocity_is_prior or not obj.observed or obj.motion_state != "MOVING":
        return None, None
    k = np.arange(1, int(round(cfg.cv_horizon_s / cfg.cv_step_s)) + 1) * cfg.cv_step_s
    path = obj.center[None, :] + k[:, None] * np.r_[obj.velocity[:2], 0.0][None, :]
    return path, k * obj.speed_sigma


class DisplayState:
    """Feed frames in time order with :meth:`update`; read :attr:`items` and robot history."""

    def __init__(self, cfg: DisplayConfig | None = None):
        self.cfg = cfg or DisplayConfig()
        self.epoch = None
        self.trails = {}  # object_id -> [(t, xyz)]
        self.last_seen = {}  # object_id -> (t, SceneObject) last emitted
        self.robot_path = []  # [(t, xyz)]
        self.items: list[DisplayItem] = []
        self.frame: SceneFrame | None = None
        self.resets = 0

    def clear(self):
        self.trails.clear()
        self.last_seen.clear()
        self.robot_path.clear()

    def update(self, frame: SceneFrame):
        cfg = self.cfg
        if self.frame is not None and frame.timestamp < self.frame.timestamp:
            raise ValueError("Frames must be fed in time order (causal display)")
        if self.epoch is not None and frame.epoch != self.epoch:
            self.clear()
            self.resets += 1
        self.epoch = frame.epoch
        self.frame = frame
        t = frame.timestamp
        if frame.pose is not None:
            self.robot_path.append((t, frame.pose[:3, 3].copy()))
        items = []
        present = set()
        for obj in frame.objects:
            present.add(obj.object_id)
            self.last_seen[obj.object_id] = (t, obj)
            trail = self.trails.setdefault(obj.object_id, [])
            if obj.observed:
                trail.append((t, obj.center.copy()))
            trail[:] = [p for p in trail if t - p[0] <= cfg.trail_s]
            if obj.observed:
                kind, alpha = FRESH, 1.0
            else:
                if obj.observation_age_s > cfg.stale_max_s:
                    continue
                kind, alpha = STALE, stale_alpha(obj.observation_age_s, cfg)
            path, sigma = cv_prediction(obj, cfg) if kind == FRESH else (None, None)
            items.append(DisplayItem(obj, kind, alpha, colour_key(obj, kind), [p for _, p in trail], path, sigma))
        for oid in list(self.last_seen):
            if oid in present:
                continue
            seen_t, obj = self.last_seen[oid]
            unseen = t - seen_t
            if unseen > cfg.ghost_hold_s:
                del self.last_seen[oid]
                self.trails.pop(oid, None)
                continue
            items.append(DisplayItem(obj, GHOST, 0.5 * (1 - unseen / max(cfg.ghost_hold_s, 1e-9)), "UNCERTAIN", unseen_s=unseen))
        self.robot_path = [p for p in self.robot_path if t - p[0] <= 120.0]
        self.items = items
        return items


def interpolate_pose(a, b, ta, tb, t):
    """Pose at ``t`` between two validated poses (translation lerp, rotation slerp)."""
    if tb <= ta:
        return b.copy()
    f = min(max((t - ta) / (tb - ta), 0.0), 1.0)
    out = np.eye(4)
    out[:3, 3] = (1 - f) * a[:3, 3] + f * b[:3, 3]
    r = a[:3, :3].T @ b[:3, :3]
    angle = np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1))
    if angle < 1e-9:
        out[:3, :3] = b[:3, :3]
        return out
    axis = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]]) / (2 * np.sin(angle))
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    th = f * angle
    out[:3, :3] = a[:3, :3] @ (np.eye(3) + np.sin(th) * k + (1 - np.cos(th)) * k @ k)
    return out


class CausalPlayer:
    """Maps a playback clock to displayed data using only frames already received.

    Without interpolation the display holds the latest frame with ``timestamp <= clock``.
    With interpolation the whole display is delayed by ``delay``: objects come from the latest
    frame at or before ``clock - delay`` and the robot pose is interpolated between the two
    validated poses bracketing ``clock - delay`` *only if both were received by* ``clock``
    and belong to the same epoch; otherwise the pose is held.
    """

    def __init__(self, frames, cfg: DisplayConfig | None = None):
        self.frames = list(frames)
        self.times = np.array([f.timestamp for f in self.frames])
        if len(self.times) > 1 and (np.diff(self.times) < 0).any():
            raise ValueError("Frames must be in time order")
        self.cfg = cfg or DisplayConfig()

    @property
    def delay(self):
        return self.cfg.interpolation_delay_s if self.cfg.interpolate_pose else 0.0

    def index_at(self, clock):
        """Index of the frame displayed at ``clock`` or -1 before the first frame."""
        return int(np.searchsorted(self.times, clock - self.delay, side="right")) - 1

    def pose_at(self, clock):
        k = self.index_at(clock)
        if k < 0:
            return None, "no data"
        cur = self.frames[k]
        if not self.cfg.interpolate_pose:
            return cur.pose, "held"
        t = clock - self.delay
        if k + 1 >= len(self.frames):
            return cur.pose, "held"
        nxt = self.frames[k + 1]
        if nxt.timestamp > clock or cur.pose is None or nxt.pose is None or nxt.epoch != cur.epoch:
            return cur.pose, "held"
        return interpolate_pose(cur.pose, nxt.pose, cur.timestamp, nxt.timestamp, t), "interpolated"
