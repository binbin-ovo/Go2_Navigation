"""Synchronized RGB + LiDAR BEV + status diagnostic frames for recorded tracking runs.

Pure rendering on numpy images (OpenCV drawing), so it can be tested on synthetic data. The
CLI ``scripts/render_tracking_video.py`` loads real recordings and tracker outputs.

Conventions (see docs/VIDEO_ANNOTATION.md):
- motion colour: blue STATIONARY, orange MOVING, gray UNCERTAIN;
- filled circle: Confirmed and observed this frame; hollow circle: Lost/coasting (the drawn
  position is a prediction, labelled ``pred``); small square: Tentative;
- dashed line: constant-velocity future, from the current estimate (not a measurement);
- magenta: ground-truth annotation (LiDAR-derived; reviewed as documented, not calibrated
  to the camera).
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

WIDTH, HEIGHT = 1920, 1080
TOP_H = 800
RGB_W = 1080
BEV_W = WIDTH - RGB_W
FONT = cv2.FONT_HERSHEY_SIMPLEX

# BGR
BLUE = (215, 120, 30)
ORANGE = (0, 140, 255)
GRAY = (150, 150, 150)
DARK = (40, 40, 40)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
MAGENTA = (200, 40, 200)
RED = (40, 40, 220)
AMBER = (0, 175, 230)
GREEN = (60, 160, 60)
MOTION_COLOUR = {"M": ORANGE, "S": BLUE, "U": GRAY}
MOTION_NAME = {"M": "MOVING", "S": "STATIONARY", "U": "UNCERTAIN"}
CV_STEP_S, CV_STEPS = 0.4, 12  # Phase 4 public-data grid: 12 future positions at 0.4 s


@dataclass
class Track:
    """One tracker output row: [id, x, y, z, vx, vy, vstd, status, motion, misses]."""

    id: int
    x: float
    y: float
    vx: float
    vy: float
    status: str  # C(onfirmed) / T(entative) / L(ost)
    motion: str  # M / S / U
    misses: int

    @classmethod
    def from_row(cls, r):
        return cls(int(r[0]), float(r[1]), float(r[2]), float(r[4]), float(r[5]), r[7], r[8], int(r[9]))

    @property
    def observed(self):
        """Position supported by a detection in this frame (not a coasting prediction)."""
        return self.misses == 0 and self.status != "L"


@dataclass
class Event:
    t: float
    kind: str  # e.g. NEW TRACK, TRACK LOST, POSSIBLE FRAGMENTATION, ID SWITCH (GT), ODOMETRY GAP
    text: str
    confirmed: bool = False  # True only when supported by ground-truth identity annotations
    xy: tuple | None = None
    ids: tuple = ()  # track IDs involved


@dataclass
class FrameData:
    t: float  # LiDAR/tracker stamp, go2_robot clock
    t_rel: float
    frame: str  # tracking frame name, e.g. "odom"
    robot_pose: np.ndarray | None  # 3x3 planar odom<-base, None if unavailable
    robot_path: np.ndarray  # (K, 2) past robot positions
    points: np.ndarray  # (N, 2) LiDAR points in the tracking frame
    clusters: list  # [(lo_xy, hi_xy)] detection boxes in the tracking frame
    tracks: list[Track]
    trails: dict = field(default_factory=dict)  # id -> [(x, y, observed)]
    gt: list = field(default_factory=list)  # annotation states (schema go2-tracking-gt-v1)
    events: list[Event] = field(default_factory=list)  # recent events to display
    rgb: np.ndarray | None = None  # BGR camera image, None if missing
    rgb_detections: list = field(default_factory=list)  # [(class, conf, (x0,y0,x1,y1))]
    rgb_index: int | None = None
    rgb_time: float | None = None  # host receipt time mapped to the robot clock
    robot_velocity: tuple | None = None  # (v m/s, yaw rate rad/s) from odometry
    odometry_gap: bool = False
    new_ids: int = 0
    title: str = ""
    banner: str = ""  # highlight caption
    forecast: dict | None = None  # held forecast overlay (see draw_forecast)
    key_ids: set | None = None  # tracks to label fully (e.g. matched to annotated objects); None = all


def align_rgb(lidar_times, rgb_host_times, host_offset, max_dt=0.15):
    """Nearest camera frame for each LiDAR stamp, after mapping host receipt to robot clock.

    ``host_offset`` = median(host_recv - robot stamp) of the odometry stream. Returns
    (indices, dt) where dt = camera - lidar in seconds; index is None when no frame lies
    within ``max_dt`` (treated as a missing camera frame). Approximate: receipt is not exposure.
    """
    cam = np.asarray(rgb_host_times, float) - host_offset
    order = np.argsort(cam)
    cam_sorted = cam[order]
    out_idx, out_dt = [], []
    for t in lidar_times:
        if not len(cam_sorted):
            out_idx.append(None)
            out_dt.append(None)
            continue
        k = int(np.clip(np.searchsorted(cam_sorted, t), 1, len(cam_sorted) - 1)) if len(cam_sorted) > 1 else 0
        k = k if len(cam_sorted) == 1 or abs(cam_sorted[k] - t) < abs(cam_sorted[k - 1] - t) else k - 1
        dt = float(cam_sorted[k] - t)
        if abs(dt) <= max_dt:
            out_idx.append(int(order[k]))
            out_dt.append(dt)
        else:
            out_idx.append(None)
            out_dt.append(dt)
    return out_idx, out_dt


def update_trails(trails, t, tracks, keep_s=3.0):
    """Per-ID history across frames: {id: [(t, x, y, observed)]}. Returns the trails of the
    tracks present now ({id: [(x, y, observed)]}); an ID keeps its own trail while it exists,
    and the trail of a vanished ID is dropped once older than ``keep_s``."""
    alive = set()
    for tr in tracks:
        trails.setdefault(tr.id, []).append((t, tr.x, tr.y, tr.observed))
        alive.add(tr.id)
    shown = {}
    for i in list(trails):
        trails[i] = [p for p in trails[i] if t - p[0] <= keep_s]
        if i in alive:
            shown[i] = [(x, y, o) for _, x, y, o in trails[i]]
        elif not trails[i]:
            del trails[i]
    return shown


def frame_label(frame, yaw_deg):
    return f"frame: {frame} (metres), view rotated {-yaw_deg:+.1f} deg so the start heading points up"


def cv_forecast(track, step_s=CV_STEP_S, steps=CV_STEPS):
    """Constant-velocity future positions (steps, 2) from the current estimate."""
    k = np.arange(1, steps + 1)[:, None] * step_s
    return np.array([track.x, track.y]) + k * np.array([track.vx, track.vy])


class BevView:
    """Maps tracking-frame XY to pixels. ``yaw`` rotates the view so that heading points up."""

    def __init__(self, center, yaw, half_extent_m, size_px, origin=(0, 0)):
        self.center = np.asarray(center, float)
        self.c, self.s = np.cos(yaw), np.sin(yaw)
        self.scale = size_px[1] / (2 * half_extent_m)
        self.size = size_px
        self.origin = origin

    def inverse(self, uv):
        """Pixel (u, v) -> tracking-frame XY (for interactive annotation)."""
        u, v = uv
        left = -(u - self.origin[0] - self.size[0] / 2) / self.scale
        fwd = -(v - self.origin[1] - self.size[1] * 0.62) / self.scale
        x = fwd * self.c - left * self.s
        y = fwd * self.s + left * self.c
        return np.array([x, y]) + self.center

    def px(self, xy):
        p = np.atleast_2d(np.asarray(xy, float)) - self.center
        fwd = p[:, 0] * self.c + p[:, 1] * self.s  # along view heading
        left = -p[:, 0] * self.s + p[:, 1] * self.c
        u = self.origin[0] + self.size[0] / 2 - left * self.scale
        v = self.origin[1] + self.size[1] * 0.62 - fwd * self.scale
        return np.c_[u, v]


def _text(img, s, xy, scale=0.55, colour=DARK, thick=1, bg=None):
    x, y = int(xy[0]), int(xy[1])
    if bg is not None:
        (w, h), b = cv2.getTextSize(s, FONT, scale, thick)
        cv2.rectangle(img, (x - 2, y - h - 3), (x + w + 2, y + b), bg, -1)
    cv2.putText(img, s, (x, y), FONT, scale, colour, thick, cv2.LINE_AA)


def _dashed(img, pts, colour, thick=2, dash=8):
    pts = np.asarray(pts, float)
    for a, b in zip(pts[:-1], pts[1:]):
        length = np.linalg.norm(b - a)
        n = max(int(length // dash), 1)
        for i in range(0, n, 2):
            p = a + (b - a) * i / n
            q = a + (b - a) * min(i + 1, n) / n
            cv2.line(img, tuple(np.int32(p)), tuple(np.int32(q)), colour, thick, cv2.LINE_AA)


def draw_bev(img, view, f: FrameData, label="", forecast_min_speed=0.2):
    """Draw into ``img`` at ``view.origin`` (clipped to ``view.size``)."""
    x0, y0 = view.origin
    w, h = view.size
    canvas = np.full((h, w, 3), 255, np.uint8)
    local = BevView(view.center, 0, 1, (w, h))
    local.c, local.s, local.scale = view.c, view.s, view.scale
    _draw_bev(canvas, local, f, label, forecast_min_speed)
    img[y0 : y0 + h, x0 : x0 + w] = canvas


def _draw_bev(img, view, f, label, forecast_min_speed):
    w, h = view.size
    key = f.key_ids
    # 1 m grid
    span = int(np.ceil(max(w, h) / view.scale)) + 2
    base = np.round(view.center)
    for k in range(-span, span + 1):
        for line in (((k, -span), (k, span)), ((-span, k), (span, k))):
            a, b = view.px(base + np.array(line[0])), view.px(base + np.array(line[1]))
            cv2.line(img, tuple(np.int32(a[0])), tuple(np.int32(b[0])), (232, 232, 232), 1)
    if len(f.points):
        p = np.int32(view.px(f.points))
        ok = (p[:, 0] >= 1) & (p[:, 0] < w - 1) & (p[:, 1] >= 1) & (p[:, 1] < h - 1)
        for du, dv_ in ((0, 0), (1, 0), (0, 1), (1, 1)):
            img[p[ok, 1] + dv_, p[ok, 0] + du] = (135, 135, 135)
    for lo, hi in f.clusters:
        corners = view.px([lo, (hi[0], lo[1]), hi, (lo[0], hi[1])])
        cv2.polylines(img, [np.int32(corners)], True, (200, 200, 200), 1, cv2.LINE_AA)
    for s in f.gt:
        if s.get("region"):
            cv2.polylines(img, [np.int32(view.px(s["region"]))], True, MAGENTA, 2, cv2.LINE_AA)
            q = view.px(np.max(s["region"], axis=0))[0]
            _text(img, f"GT {s['id']}", (q[0] + 6, q[1] - 6), 0.5, MAGENTA)
        elif s.get("xy"):
            q = view.px(s["xy"])[0]
            cv2.drawMarker(img, tuple(np.int32(q)), MAGENTA, cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA)
            _text(img, f"GT {s['id']}", (q[0] - 70, q[1] + 4), 0.5, MAGENTA)
    gt_text = ", ".join(f"{s['id']} {s['status']}" + (f"/{s['motion']}" if s.get("motion") else "") for s in f.gt)
    if gt_text:
        _text(img, "GT: " + gt_text, (10, 68), 0.5, MAGENTA)
    if len(f.robot_path) > 1:
        cv2.polylines(img, [np.int32(view.px(f.robot_path))], False, GREEN, 2, cv2.LINE_AA)
    if f.robot_pose is not None:
        o = f.robot_pose[:2, 2]
        heading = f.robot_pose[:2, 0]
        side = f.robot_pose[:2, 1]
        tri = view.px([o + 0.3 * heading, o - 0.15 * heading + 0.12 * side, o - 0.15 * heading - 0.12 * side])
        cv2.fillPoly(img, [np.int32(tri)], GREEN, cv2.LINE_AA)
    # background tracks first, key tracks on top
    order = sorted(f.tracks, key=lambda t: key is None or t.id in key)
    for t in order:
        is_key = key is None or t.id in key
        colour = MOTION_COLOUR.get(t.motion, GRAY)
        if not is_key:
            colour = tuple(int(c * 0.45 + 255 * 0.55) for c in colour)
        trail = f.trails.get(t.id, [])
        if len(trail) > 1 and (is_key or t.motion == "M"):
            pts = view.px([(a, b) for a, b, _ in trail])
            for (a, b), (_, _, obs) in zip(zip(pts[:-1], pts[1:]), trail[1:]):
                cv2.line(img, tuple(np.int32(a)), tuple(np.int32(b)), colour if obs else (215, 215, 215), 2 if obs else 1, cv2.LINE_AA)
        q = np.int32(view.px((t.x, t.y))[0])
        if t.status == "T":
            cv2.rectangle(img, tuple(q - 3), tuple(q + 3), colour, 1, cv2.LINE_AA)
            continue
        speed = np.hypot(t.vx, t.vy)
        r = 7 if is_key else 4
        if t.observed:
            cv2.circle(img, tuple(q), r, colour, -1, cv2.LINE_AA)
        else:
            cv2.circle(img, tuple(q), r + 1, colour, 1, cv2.LINE_AA)
        if not is_key:
            continue
        if t.observed and t.status == "C":
            if speed >= 0.05:
                v_end = view.px((t.x + t.vx, t.y + t.vy))[0]
                cv2.arrowedLine(img, tuple(q), tuple(np.int32(v_end)), DARK, 1, cv2.LINE_AA, tipLength=0.2)
            if speed >= forecast_min_speed and is_key:
                _dashed(img, np.vstack([view.px((t.x, t.y)), view.px(cv_forecast(t))]), colour, 2)
        if is_key:
            tag = f"#{t.id}" + ("" if t.observed else (" pred LOST" if t.status == "L" else " pred"))
            _text(img, tag, (q[0] + 9, q[1] - 6), 0.52, colour if t.observed else (110, 110, 110), 1)
            if t.observed and t.status == "C":
                _text(img, f"{speed:.2f} m/s {MOTION_NAME.get(t.motion, '')}", (q[0] + 9, q[1] + 12), 0.42, DARK)
    if f.forecast:
        draw_forecast(img, view, f.forecast, f.t_rel)
    for e in f.events:
        if e.xy is not None:
            q = view.px(e.xy)[0]
            colour = RED if e.confirmed else AMBER
            cv2.circle(img, tuple(np.int32(q)), 22, colour, 2, cv2.LINE_AA)
            _text(img, e.kind, (q[0] - 70, q[1] - 28), 0.5, WHITE, 1, bg=colour)
    # header and legend
    _text(img, label or f"LiDAR BEV, frame: {f.frame}", (10, 24), 0.55, BLACK, 1, bg=(250, 250, 250))
    _text(img, "1 m grid; gray dots: LiDAR (0.3 s); gray boxes: clusters; faded: other tracks", (10, 46), 0.45, DARK, 1, bg=(250, 250, 250))
    legend = [
        ("STATIONARY", BLUE, "dot"),
        ("MOVING", ORANGE, "dot"),
        ("UNCERTAIN", GRAY, "dot"),
        ("observed this frame (filled)", DARK, "dot"),
        ("coasting/lost: predicted (hollow)", DARK, "ring"),
        ("tentative", DARK, "square"),
        ("const.-velocity 4.8 s (dashed)", DARK, "dash"),
        ("velocity, 1 s (arrow)", DARK, "arrow"),
        ("ground truth (LiDAR-derived)", MAGENTA, "cross"),
        ("robot + path (odometry)", GREEN, "tri"),
    ]
    lx, ly = 18, h - 20 * len(legend) - 6
    cv2.rectangle(img, (lx - 10, ly - 16), (lx + 300, h - 4), (248, 248, 248), -1)
    cv2.rectangle(img, (lx - 10, ly - 16), (lx + 300, h - 4), (200, 200, 200), 1)
    for i, (name, colour, kind) in enumerate(legend):
        c = (lx + 6, ly + 20 * i)
        if kind == "dot":
            cv2.circle(img, c, 6, colour, -1, cv2.LINE_AA)
        elif kind == "ring":
            cv2.circle(img, c, 6, colour, 1, cv2.LINE_AA)
        elif kind == "square":
            cv2.rectangle(img, (c[0] - 4, c[1] - 4), (c[0] + 4, c[1] + 4), colour, 1)
        elif kind == "dash":
            _dashed(img, [(c[0] - 8, c[1]), (c[0] + 10, c[1])], colour, 2, 4)
        elif kind == "arrow":
            cv2.arrowedLine(img, (c[0] - 8, c[1]), (c[0] + 10, c[1]), colour, 1, cv2.LINE_AA, tipLength=0.4)
        elif kind == "cross":
            cv2.drawMarker(img, c, colour, cv2.MARKER_CROSS, 12, 2)
        else:
            cv2.fillPoly(img, [np.int32([(c[0], c[1] - 6), (c[0] - 5, c[1] + 5), (c[0] + 5, c[1] + 5)])], colour)
        _text(img, name, (lx + 20, ly + 20 * i + 5), 0.45, DARK)


CYAN = (200, 160, 0)


def draw_forecast(img, view, fc, t_rel):
    """Held forecast: origin time, annotated history (solid magenta), annotated future (thin
    magenta dots, shown for evaluation only), constant-velocity forecasts from the annotated
    history (dashed magenta) and from the tracker history (dashed cyan, if identity-validated)."""
    hist = view.px(fc["gt_history"])
    cv2.polylines(img, [np.int32(hist)], False, MAGENTA, 3, cv2.LINE_AA)
    for q in view.px(fc["future"]):
        cv2.circle(img, tuple(np.int32(q)), 3, MAGENTA, -1, cv2.LINE_AA)
    _dashed(img, np.vstack([hist[-1:], view.px(fc["cv_gt"])]), MAGENTA, 2, 10)
    if fc.get("cv_tracker") is not None:
        start = view.px(fc["tracker_last"]) if fc.get("tracker_last") is not None else hist[-1:]
        _dashed(img, np.vstack([start, view.px(fc["cv_tracker"])]), CYAN, 2, 10)
    o = hist[-1]
    _text(img, f"forecast origin {fc['origin_rel']:.2f} s (held {t_rel - fc['origin_rel']:.1f} s)", (o[0] + 12, o[1] + 34), 0.5, WHITE, 1, bg=MAGENTA)
    _text(img, "forecast overlay (0.4 s grid, 8 obs + 12 pred):", (10, 92), 0.48, DARK, 1, bg=(250, 250, 250))
    _text(img, "magenta dashed: CV from annotated history; cyan dashed: CV from tracker history", (10, 114), 0.45, DARK, 1, bg=(250, 250, 250))
    _text(img, "magenta dots: annotated future (evaluation only); learned models: no checkpoint on this Mac", (10, 136), 0.45, DARK, 1, bg=(250, 250, 250))


def draw_rgb(img, f: FrameData, origin=(0, 0), size=(RGB_W, TOP_H)):
    x0, y0 = origin
    w, h = size
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (30, 30, 30), -1)
    _text(img, "RGB detections - not calibrated to LiDAR tracks", (x0 + 10, y0 + 26), 0.65, WHITE)
    if f.rgb is None:
        _text(img, "CAMERA FRAME MISSING (no frame within 0.15 s)", (x0 + 40, y0 + h // 2), 0.9, (80, 80, 255), 2)
        return
    ih, iw = f.rgb.shape[:2]
    s = min(w / iw, (h - 80) / ih)
    tw, th = int(iw * s), int(ih * s)
    top = y0 + 40
    img[top : top + th, x0 : x0 + tw] = cv2.resize(f.rgb, (tw, th), interpolation=cv2.INTER_AREA)
    for name, conf, (a, b, c, d) in f.rgb_detections:
        p, q = (int(x0 + a * s), int(top + b * s)), (int(x0 + c * s), int(top + d * s))
        colour = (0, 220, 255) if name == "person" else (0, 200, 0)
        cv2.rectangle(img, p, q, colour, 2, cv2.LINE_AA)
        _text(img, f"{name} {conf:.2f}", (p[0] + 2, max(p[1] - 6, top + 14)), 0.5, BLACK, 1, bg=colour)
    if f.banner:
        _banner(img, f.banner, (x0 + 10, y0 + h - 100), w - 20)
    info = f"camera frame {f.rgb_index}   camera t {f.rgb_time - f.t + f.t_rel:.2f} s (host receipt, approx.)"
    _text(img, info, (x0 + 10, top + th + 26), 0.55, WHITE)


def _banner(img, text, xy, width, scale=0.85):
    x, y = xy
    cv2.rectangle(img, (x, y), (x + width, y + 80), (30, 30, 160) if text.startswith("FAIL") else (40, 110, 40), -1)
    lines = text.split("|")
    for i, line in enumerate(lines[:2]):
        sc, th = (scale, 2) if i == 0 else (0.62, 1)
        while sc > 0.3 and cv2.getTextSize(line.strip(), FONT, sc, th)[0][0] > width - 24:
            sc -= 0.02
        _text(img, line.strip(), (x + 12, y + 32 + 34 * i), sc, WHITE, th)


def title_card(text, size=(WIDTH, HEIGHT)):
    """Plain caption frame (no imagery) shown before a highlight clip."""
    img = np.full((size[1], size[0], 3), 25, np.uint8)
    for i, line in enumerate(text.split("|")):
        _text(img, line.strip(), (120, 420 + 70 * i), 1.5 if i == 0 else 1.0, WHITE, 3 if i == 0 else 2)
    return img


def draw_status(img, f: FrameData, timeline, origin=(0, TOP_H), size=(WIDTH, HEIGHT - TOP_H)):
    x0, y0 = origin
    w, h = size
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (245, 245, 245), -1)
    conf = sum(t.status == "C" for t in f.tracks)
    lost = sum(t.status == "L" for t in f.tracks)
    tent = sum(t.status == "T" for t in f.tracks)
    dt = "n/a" if f.rgb_time is None else f"{(f.rgb_time - f.t) * 1000:+.0f} ms"
    vel = "n/a" if f.robot_velocity is None else f"v {f.robot_velocity[0]:.2f} m/s, yaw rate {f.robot_velocity[1]:+.2f} rad/s"
    lines = [
        f"{f.title}   t = {f.t_rel:6.2f} s   LiDAR stamp {f.t:.3f} (go2_robot)   camera-LiDAR dt {dt} (approximate sync)",
        f"tracks: confirmed {conf}   lost/coasting {lost}   tentative {tent}   new IDs this frame {f.new_ids}   robot: {vel}",
    ]
    for i, s in enumerate(lines):
        _text(img, s, (x0 + 14, y0 + 28 + 26 * i), 0.62, BLACK)
    warn = []
    if f.rgb_index is None:
        warn.append("CAMERA MISSING")
    if f.robot_pose is None:
        warn.append("NO POSE")
    if f.odometry_gap:
        warn.append("ODOMETRY GAP")
    if warn:
        _text(img, "  ".join(warn), (x0 + w - 420, y0 + 28), 0.7, WHITE, 2, bg=(60, 60, 220))
    # timeline: event ticks over the whole run, cursor at now
    if timeline is not None:
        t0, t1, ticks = timeline
        tx0, tx1, ty = x0 + 14, x0 + w // 2 - 20, y0 + 110
        cv2.line(img, (tx0, ty), (tx1, ty), DARK, 1)
        for t, colour in ticks:
            u = int(tx0 + (t - t0) / max(t1 - t0, 1e-6) * (tx1 - tx0))
            cv2.line(img, (u, ty - 12), (u, ty + 12), colour, 2)
        u = int(tx0 + (f.t_rel - t0) / max(t1 - t0, 1e-6) * (tx1 - tx0))
        cv2.line(img, (u, ty - 22), (u, ty + 22), BLACK, 3)
        _text(img, f"{t0:.0f} s", (tx0, ty + 40), 0.45, DARK)
        _text(img, f"{t1:.0f} s", (tx1 - 40, ty + 40), 0.45, DARK)
        _text(img, "event timeline: red = ID switch vs ground truth, amber = heuristic warning", (tx0, ty + 66), 0.48, DARK)
    # event log
    ex = x0 + w // 2
    _text(img, "events (red: GT-supported, amber: heuristic, unconfirmed)", (ex, y0 + 98), 0.5, DARK)
    for i, e in enumerate(f.events[-5:]):
        colour = RED if e.confirmed else (0, 120, 170)
        _text(img, f"{e.t:6.2f} s  {e.text}", (ex, y0 + 124 + 24 * i), 0.55, colour, 1)


def compose(f: FrameData, view, timeline=None, bev_label=""):
    img = np.full((HEIGHT, WIDTH, 3), 255, np.uint8)
    draw_rgb(img, f)
    view.origin = (RGB_W, 0)
    view.size = (BEV_W, TOP_H)
    draw_bev(img, view, f, bev_label)
    draw_status(img, f, timeline)
    cv2.line(img, (RGB_W, 0), (RGB_W, TOP_H), DARK, 2)
    cv2.line(img, (0, TOP_H), (WIDTH, TOP_H), DARK, 2)
    return img


def compose_comparison(frames, labels, views, timeline=None):
    """Side-by-side BEV of several tracker variants at one timestamp (shared view scale)."""
    img = np.full((HEIGHT, WIDTH, 3), 255, np.uint8)
    w = WIDTH // len(frames)
    for k, (f, label, view) in enumerate(zip(frames, labels, views)):
        view.origin = (k * w, 0)
        view.size = (w, TOP_H)
        draw_bev(img, view, f, label)
        cv2.line(img, (k * w, 0), (k * w, TOP_H), DARK, 2)
    draw_status(img, frames[0], timeline)
    if frames[0].banner:
        _banner(img, frames[0].banner, (WIDTH // 2, TOP_H - 90), WIDTH // 2 - 10)
    return img


def private_output_ok(path, repo_root):
    """True if ``path`` is outside the repository or ignored by Git (never commit home video)."""
    path, repo_root = Path(path).resolve(), Path(repo_root).resolve()
    if not path.is_relative_to(repo_root):
        return True
    rel = path.relative_to(repo_root)
    r = subprocess.run(["git", "check-ignore", "-q", "--no-index", str(rel)], cwd=repo_root, check=False)
    return r.returncode == 0


class VideoWriter:
    """H.264 MP4 (yuv420p, faststart) through ffmpeg, which QuickTime opens on macOS."""

    def __init__(self, path, fps, size=(WIDTH, HEIGHT), crf=20):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{size[0]}x{size[1]}", "-r", f"{fps}",
            "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf), "-preset", "medium",
            "-movflags", "+faststart", str(path),
        ]  # fmt: skip
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self.size = size
        self.frames = 0

    def write(self, img):
        if img.shape[:2] != (self.size[1], self.size[0]):
            raise ValueError("Frame size mismatch")
        self.proc.stdin.write(np.ascontiguousarray(img).tobytes())
        self.frames += 1

    def close(self):
        self.proc.stdin.close()
        if self.proc.wait():
            raise RuntimeError("ffmpeg failed")


def heuristic_events(records, roi=None, min_life_s=1.0, near_m=0.6, window_s=1.5, lost_frames=5):
    """Track-lifecycle warnings from tracker output alone (never ground truth).

    ``records``: [{"t", "tracks": rows}] in time order. Only "significant" tracks (Confirmed
    for at least ``min_life_s``) inside ``roi(t, x, y)`` generate events:
    NEW TRACK, TRACK LOST (unobserved for ``lost_frames`` frames), TRACK ENDED, POSSIBLE
    FRAGMENTATION (a new track near one that was lost or ended within ``window_s``), POSSIBLE
    ID SWITCH (a new track near one still coasting), POSSIBLE DUPLICATE (two observed
    Confirmed tracks within 0.3 m for 3 frames). All are heuristics: confirmed=False.
    """
    roi = roi or (lambda t, x, y: True)
    first_c, last_c = {}, {}
    for r in records:
        for row in r["tracks"]:
            if row[7] == "C":
                first_c.setdefault(row[0], r["t"])
                last_c[row[0]] = r["t"]
    significant = {i for i in first_c if last_c[i] - first_c[i] >= min_life_s}
    events, announced, lost_at, ended_at, pair_seen, last_pair_event = [], set(), {}, {}, {}, {}
    previous = {}
    for r in records:
        t = r["t"]
        rows = {row[0]: row for row in r["tracks"] if row[0] in significant}
        for i, row in rows.items():
            xy = (row[1], row[2])
            if not roi(t, *xy):
                continue
            if row[7] == "C" and i not in announced:
                announced.add(i)
                text, kind = f"NEW TRACK #{i}", "NEW TRACK"
                for j, (tj, xyj) in {**ended_at, **lost_at}.items():
                    if j != i and t - tj <= window_s and np.hypot(xy[0] - xyj[0], xy[1] - xyj[1]) <= near_m:
                        coasting = j in rows and rows[j][7] == "L"
                        kind = "POSSIBLE ID SWITCH" if coasting else "POSSIBLE FRAGMENTATION"
                        text = f"{kind} #{j} -> #{i}"
                        break
                ids = (i,) if kind == "NEW TRACK" else (j, i)
                events.append(Event(t, kind, text, False, xy, ids))
            if row[9] == lost_frames and i in announced:
                lost_at[i] = (t, xy)
                events.append(Event(t, "TRACK LOST", f"TRACK #{i} LOST (unobserved {lost_frames} frames)", False, xy, (i,)))
        for i, row in previous.items():
            if i not in rows and i in announced and roi(t, row[1], row[2]):
                ended_at[i] = (t, (row[1], row[2]))
                events.append(Event(t, "TRACK ENDED", f"TRACK #{i} ENDED", False, (row[1], row[2]), (i,)))
        observed = [row for row in rows.values() if row[7] == "C" and row[9] == 0 and roi(t, row[1], row[2])]
        current = set()
        for a in range(len(observed)):
            for b in range(a + 1, len(observed)):
                p, q = observed[a], observed[b]
                if np.hypot(p[1] - q[1], p[2] - q[2]) <= 0.3:
                    key = (min(p[0], q[0]), max(p[0], q[0]))
                    current.add(key)
                    pair_seen[key] = pair_seen.get(key, 0) + 1
                    if pair_seen[key] == 3 and t - last_pair_event.get(key, -1e9) > 2.0:
                        last_pair_event[key] = t
                        events.append(Event(t, "POSSIBLE DUPLICATE", f"POSSIBLE DUPLICATE #{key[0]} / #{key[1]}", False, (p[1], p[2]), key))
        for key in list(pair_seen):
            if key not in current:
                pair_seen[key] = 0
        previous = rows
    return events
