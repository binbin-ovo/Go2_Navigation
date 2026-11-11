"""FSD-inspired 3D scene rendering with NumPy + OpenCV (headless, deterministic, macOS-safe).

Every scene coordinate is drawn exactly as received in ``odom``. ``T_odom_base`` is used in one
place only, :func:`draw_robot`, to place the robot proxy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .camera import Camera
from .layers import Layers
from .state import FRESH, GHOST, STALE, DisplayItem

FONT = cv2.FONT_HERSHEY_SIMPLEX
# BGR, dark theme
BG = (24, 20, 17)
GRID = np.array([78, 72, 66], float)
TEXT = (235, 235, 235)
DIM = (150, 150, 150)
MOTION_COLOUR = {"MOVING": (30, 145, 255), "STATIONARY": (235, 165, 60), "UNCERTAIN": (165, 165, 165)}
ROBOT = (240, 240, 240)
ROBOT_ACCENT = (255, 205, 60)
RAW_COLOUR = (255, 255, 120)
P3_COLOUR = (60, 220, 240)
CLUSTER_OK = (120, 220, 120)
CLUSTER_REJECT = (90, 90, 235)
GT_COLOUR = (220, 80, 220)
FLOOR_OFFSET = 0.32  # base_link height above the floor (5A sensor_height_m); only for floor drawing

BOX_EDGES = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
BOX_FACES = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
FACE_NORMALS = np.array([[0, 0, -1], [0, 0, 1], [0, -1, 0], [0, 1, 0], [-1, 0, 0], [1, 0, 0]], float)


@dataclass
class Overlays:
    """Optional debug inputs, all already in odom (never transformed here)."""

    raw_points: np.ndarray | None = None  # current incremental scan
    accum_points: np.ndarray | None = None
    accum_ages: np.ndarray | None = None  # seconds before the frame time
    p3_boxes: list = field(default_factory=list)  # [(8, 3) corners]
    p3_tracks: list = field(default_factory=list)  # [dict(id, xyz, v, status, motion, observed)]
    clusters: list = field(default_factory=list)  # [dict(lo, hi, centroid, accepted, n)]
    gt: list = field(default_factory=list)  # retrospective annotation states
    floor_z: float | None = None
    notes: list = field(default_factory=list)  # extra HUD lines


def box_corners(center, dims):
    c, h = np.asarray(center, float), np.asarray(dims, float) / 2
    sign = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    return c + sign * h


def _pt(p):
    return (int(round(p[0])), int(round(p[1])))


def line3d(img, cam: Camera, a, b, colour, thick=1, dash=None):
    """Draw a 3D segment (near-plane clipped). ``dash`` (m) draws a dashed line."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if dash:
        n = max(int(np.linalg.norm(b - a) / dash), 1)
        for i in range(0, n, 2):
            line3d(img, cam, a + (b - a) * i / n, a + (b - a) * min(i + 1, n) / n, colour, thick)
        return
    seg = cam.clip_segment(a, b)
    if seg is None or not np.isfinite(seg).all() or np.abs(seg).max() > 1e5:
        return
    cv2.line(img, _pt(seg[0]), _pt(seg[1]), colour, thick, cv2.LINE_AA)


def polyline3d(img, cam, pts, colour, thick=1, dash=None, closed=False):
    pts = np.asarray(pts, float)
    for a, b in zip(pts[:-1], pts[1:]):
        line3d(img, cam, a, b, colour, thick, dash)
    if closed and len(pts) > 2:
        line3d(img, cam, pts[-1], pts[0], colour, thick, dash)


def blend_polys(img, polys, colour, alpha):
    """Fill 2D polygons with opacity ``alpha`` (ROI-limited blending)."""
    polys = [np.asarray(p, float) for p in polys if p is not None and len(p) >= 3 and np.isfinite(p).all()]
    if not polys or alpha <= 0:
        return
    allp = np.vstack(polys)
    x0, y0 = np.floor(allp.min(0)).astype(int)
    x1, y1 = np.ceil(allp.max(0)).astype(int) + 1
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, img.shape[1]), min(y1, img.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    over = roi.copy()
    cv2.fillPoly(over, [np.int32(np.round(p - [x0, y0])) for p in polys], colour, cv2.LINE_AA)
    cv2.addWeighted(over, alpha, roi, 1 - alpha, 0, dst=roi)


def mix(colour, alpha, bg=BG):
    return tuple(int(c * alpha + b * (1 - alpha)) for c, b in zip(colour, bg))


def text(img, s, xy, scale=0.5, colour=TEXT, thick=1, bg=None, pad=3):
    x, y = int(xy[0]), int(xy[1])
    if bg is not None:
        (w, h), base = cv2.getTextSize(s, FONT, scale, thick)
        cv2.rectangle(img, (x - pad, y - h - pad), (x + w + pad, y + base + pad // 2), bg, -1)
    cv2.putText(img, s, (x, y), FONT, scale, colour, thick, cv2.LINE_AA)


def label_block(img, lines, anchor, colour, alpha=1.0, scale=0.45, taken=None):
    """Small dark pill with a coloured left bar above ``anchor``. With ``taken`` (list of
    rectangles already used), a label overlapping an earlier one is skipped (returns False)."""
    if anchor is None or not np.isfinite(anchor).all():
        return False
    sizes = [cv2.getTextSize(s, FONT, scale, 1)[0] for s in lines]
    w = max(s[0] for s in sizes) + 14
    lh = max(s[1] for s in sizes) + 7
    x, y = int(anchor[0]) - w // 2, int(anchor[1]) - lh * len(lines) - 6
    if x > img.shape[1] or y > img.shape[0] or x + w < 0 or y + lh * len(lines) < 0:
        return False
    rect = (x, y, x + w, y + lh * len(lines) + 4)
    if taken is not None:
        for r in taken:
            if rect[0] < r[2] and r[0] < rect[2] and rect[1] < r[3] and r[1] < rect[3]:
                return False
        taken.append(rect)
    blend_polys(img, [np.array([(x, y), (x + w, y), (x + w, y + lh * len(lines) + 4), (x, y + lh * len(lines) + 4)])], (12, 12, 12), 0.72 * alpha)
    cv2.rectangle(img, (x, y), (x + 3, y + lh * len(lines) + 4), mix(colour, alpha), -1)
    for i, s in enumerate(lines):
        text(img, s, (x + 8, y + lh * (i + 1) - 2), scale, mix(TEXT if i == 0 else DIM, alpha))
    return True


def splat(img, cam, pts, colours, size=1):
    """Vectorized point splats, far to near. ``colours``: (N, 3) uint8 or one BGR tuple."""
    if pts is None or not len(pts):
        return 0
    uv, z = cam.project(pts)
    ok = np.isfinite(uv).all(1)
    ok &= (uv[:, 0] >= 0) & (uv[:, 0] < cam.w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < cam.h - 1)
    if not ok.any():
        return 0
    idx = np.flatnonzero(ok)
    idx = idx[np.argsort(-z[idx])]
    u = uv[idx, 0].astype(int)
    v = uv[idx, 1].astype(int)
    col = np.broadcast_to(np.asarray(colours, np.uint8), (len(pts), 3))[idx] if np.ndim(colours) == 2 else np.asarray(colours, np.uint8)
    offs = [(0, 0)] if size <= 1 else [(0, 0), (1, 0), (0, 1), (1, 1)]
    for du, dv in offs:
        img[v + dv, u + du] = col
    return len(idx)


def age_colours(ages, horizon=1.0):
    """New points bright yellow, old points dark purple (TURBO colormap, reversed)."""
    a = np.clip(1.0 - np.asarray(ages, float) / horizon, 0, 1)
    lut = cv2.applyColorMap(np.arange(256, dtype=np.uint8)[:, None], cv2.COLORMAP_PLASMA)[:, 0]
    return lut[(a * 255).astype(int)]


def floor_z_of(pose, overlays: Overlays | None = None):
    if overlays is not None and overlays.floor_z is not None:
        return overlays.floor_z
    return None if pose is None else float(pose[2, 3] - FLOOR_OFFSET)


def draw_grid(img, cam, center_xy, floor_z, radius=9, step=1.0):
    cx, cy = np.round(np.asarray(center_xy) / step) * step
    ticks = np.arange(-radius, radius + step, step)
    for along in (0, 1):
        for k in ticks:
            for j0 in np.arange(-radius, radius, 1.0):
                mid = math.hypot(k, j0 + 0.5)
                fade = max(0.0, 1 - mid / radius) ** 1.4
                if fade <= 0.02:
                    continue
                if along == 0:
                    a, b = (cx + k, cy + j0, floor_z), (cx + k, cy + j0 + 1, floor_z)
                else:
                    a, b = (cx + j0, cy + k, floor_z), (cx + j0 + 1, cy + k, floor_z)
                line3d(img, cam, a, b, tuple(int(c) for c in BG + (GRID - BG) * fade), 1)


def draw_occupancy_footprint(img, cam, frame, floor_z):
    """Flat floor tiles where persistent returns exist. Absence is unknown, not free (no tile)."""
    if not len(frame.cell_centers):
        return 0
    s = frame.voxel_size
    keys = np.floor(frame.cell_centers[:, :2] / s).astype(np.int64)
    order = np.argsort(-frame.cell_scores)
    _, first = np.unique(keys[order], axis=0, return_index=True)
    sel = order[first]
    polys_hi, polys_lo = [], []
    for i in sel:
        x, y = (keys[i] + 0.5) * s
        h = s * 0.46
        quad = np.array([(x - h, y - h, floor_z), (x + h, y - h, floor_z), (x + h, y + h, floor_z), (x - h, y + h, floor_z)])
        uv, z = cam.project(quad)
        if not np.isfinite(uv).all():
            continue
        (polys_hi if frame.cell_scores[i] >= 0.5 else polys_lo).append(uv)
    blend_polys(img, polys_lo, (125, 122, 118), 0.22)
    blend_polys(img, polys_hi, (165, 162, 158), 0.40)
    return len(sel)


def draw_voxels(img, cam, frame):
    if not len(frame.cell_centers):
        return 0
    s = frame.voxel_size * 0.5
    polys = {0: [], 1: []}
    for c, sc in zip(frame.cell_centers, frame.cell_scores):
        top = np.array([(c[0] - s, c[1] - s, c[2] + s), (c[0] + s, c[1] - s, c[2] + s), (c[0] + s, c[1] + s, c[2] + s), (c[0] - s, c[1] + s, c[2] + s)])
        uv, _ = cam.project(top)
        if np.isfinite(uv).all():
            polys[int(sc >= 0.5)].append(uv)
    blend_polys(img, polys[0], (125, 115, 100), 0.30)
    blend_polys(img, polys[1], (175, 155, 125), 0.50)
    return len(frame.cell_centers)


def draw_robot(img, cam, pose, floor_z):
    """Go2 proxy: the only place T_odom_base is applied."""
    if pose is None:
        return
    R, t = pose[:3, :3], pose[:3, 3]

    def tf(p):
        return np.asarray(p, float) @ R.T + t

    body = tf(box_corners((0.0, 0.0, 0.01), (0.62, 0.27, 0.15)))
    faces = [body[list(f)] for f in BOX_FACES]
    blend_polys(img, [cam.project(f)[0] for f in faces], ROBOT, 0.35)
    for a, b in BOX_EDGES:
        line3d(img, cam, body[a], body[b], ROBOT, 2)
    head = tf(box_corners((0.33, 0.0, 0.04), (0.08, 0.2, 0.1)))
    for a, b in BOX_EDGES:
        line3d(img, cam, head[a], head[b], ROBOT_ACCENT, 2)
    for sx in (-0.21, 0.21):
        for sy in (-0.12, 0.12):
            hip = tf((sx, sy, -0.05))
            knee = tf((sx + 0.06, sy * 1.1, -0.19))
            foot = tf((sx, sy * 1.1, -0.30))
            line3d(img, cam, hip, knee, ROBOT, 2)
            line3d(img, cam, knee, foot, ROBOT, 2)
    if floor_z is not None:
        h = R[:, 0].copy()
        h[2] = 0
        h /= max(np.linalg.norm(h), 1e-9)
        side = np.array([-h[1], h[0], 0])
        base = np.array([t[0], t[1], floor_z])
        chevron = [base + 0.55 * h + 0.0 * side, base + 0.38 * h + 0.12 * side, base + 0.38 * h - 0.12 * side]
        blend_polys(img, [cam.project(chevron)[0]], ROBOT_ACCENT, 0.8)


def draw_box(img, cam, center, dims, colour, alpha=1.0, fill=0.0, thick=2, dash=None):
    corners = box_corners(center, dims)
    if fill > 0:
        faces = []
        for f, n in zip(BOX_FACES, FACE_NORMALS):
            fc = corners[list(f)].mean(0)
            if np.dot(n, cam.eye - fc) > 0:  # visible face
                uv, _ = cam.project(corners[list(f)])
                faces.append(uv)
        blend_polys(img, faces, colour, fill * alpha)
    for a, b in BOX_EDGES:
        line3d(img, cam, corners[a], corners[b], mix(colour, alpha), thick, dash)
    return corners


def draw_brackets(img, cam, center, dims, colour, frac=0.25):
    """Corner brackets: the raw observed box, distinct from the filtered box."""
    c = box_corners(center, dims)
    for a, b in BOX_EDGES:
        d = c[b] - c[a]
        line3d(img, cam, c[a], c[a] + frac * d, colour, 1)
        line3d(img, cam, c[b], c[b] - frac * d, colour, 1)


def floor_circle(center_xy, r, floor_z, n=20):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.c_[center_xy[0] + r * np.cos(th), center_xy[1] + r * np.sin(th), np.full(n, floor_z)]


def draw_velocity(img, cam, item: DisplayItem, floor_z, alpha):
    o = item.obj
    if o.velocity_is_prior or o.speed < 0.05:
        return
    colour = mix(MOTION_COLOUR[item.colour_key], alpha)
    base = np.array([o.center[0], o.center[1], floor_z + 0.02])
    v = np.r_[o.velocity[:2], 0.0]
    tip = base + v  # 1 s of travel
    seg = cam.clip_segment(base, tip)
    if seg is not None and np.isfinite(seg).all():
        cv2.arrowedLine(img, _pt(seg[0]), _pt(seg[1]), colour, 3, cv2.LINE_AA, tipLength=0.18)
    # velocity uncertainty: 2-sigma circle at the 1 s tip and the implied heading spread
    sig2 = 2 * o.speed_sigma
    polyline3d(img, cam, floor_circle(tip[:2], max(sig2, 0.02), tip[2]), mix(colour, 0.6), 1, closed=True)
    if o.speed > 1e-6 and sig2 < o.speed:
        spread = math.asin(min(sig2 / o.speed, 1.0))
        for s in (-spread, spread):
            c, si = math.cos(s), math.sin(s)
            d = np.array([v[0] * c - v[1] * si, v[0] * si + v[1] * c, 0.0])
            line3d(img, cam, base, base + d, mix(colour, 0.45), 1, dash=0.08)


def draw_cv_path(img, cam, item: DisplayItem, floor_z):
    if item.cv_path is None:
        return
    colour = MOTION_COLOUR[item.colour_key]
    pts = np.c_[item.cv_path[:, :2], np.full(len(item.cv_path), floor_z + 0.02)]
    start = np.array([item.obj.center[0], item.obj.center[1], floor_z + 0.02])
    polyline3d(img, cam, np.vstack([start, pts]), mix(colour, 0.85), 2, dash=0.1)
    for p, s in zip(pts[1::2], item.cv_sigma[1::2]):
        polyline3d(img, cam, floor_circle(p[:2], max(s, 0.03), p[2], 16), mix(colour, 0.35), 1, closed=True)


def object_label(item: DisplayItem, show_id):
    o = item.obj
    head = "OBJ"  # geometric object; no semantic class is claimed
    if show_id:
        head += f" #{o.object_id}"
    if item.kind == GHOST:
        return [f"{head} unseen {item.unseen_s:.1f}s", "unknown - not free"]
    if o.velocity_is_prior:
        motion = "v n/a (prior)"
    else:
        motion = f"{o.speed:.2f}+-{2 * o.speed_sigma:.2f} m/s"
    state = o.motion_state if item.kind == FRESH else "UNCERTAIN"
    first = f"{head} {state}"
    second = f"{motion}  evid {o.observation_score:.2f}"
    third = f"obs age {o.observation_age_s:.2f}s" + ("  STALE" if item.kind == STALE else "")
    return [first, second, third]


LABEL_PRIORITY = {FRESH: 0, STALE: 2, GHOST: 3}


def draw_items(img, cam, items, layers: Layers, floor_z, robot_xy=None, label_range=5.0):
    """Geometry far-to-near, then labels by priority (fresh MOVING first, nearest first),
    skipping labels that would overlap. Labels beyond ``label_range`` of the robot are
    dropped unless the object is fresh MOVING; the geometry is always drawn."""

    def depth(it):
        return float(cam.to_cam(it.obj.center)[0, 2])

    drawn = dict(fresh=0, stale=0, ghost=0)
    labels = []
    for it in sorted(items, key=depth, reverse=True):
        o = it.obj
        colour = MOTION_COLOUR[it.colour_key]
        if it.kind == FRESH and not layers.fresh_boxes:
            continue
        if it.kind == STALE and not layers.stale_boxes:
            continue
        if it.kind == GHOST and not layers.lost_coasting:
            continue
        if layers.trails and len(it.trail) > 1:
            tr = np.array(it.trail)
            tr[:, 2] = floor_z + 0.01
            polyline3d(img, cam, tr, mix(colour, 0.5), 1)
        if it.kind == FRESH:
            draw_box(img, cam, o.center, o.dimensions, colour, 1.0, fill=0.28, thick=2)
            if layers.raw_detection_boxes and o.observed_center is not None:
                draw_brackets(img, cam, o.observed_center, o.observed_dimensions, (255, 255, 255))
        elif it.kind == STALE:
            draw_box(img, cam, o.center, o.dimensions, colour, it.alpha, fill=0.06, thick=1, dash=0.06)
        else:
            lo, hi = o.center[:2] - o.dimensions[:2] / 2, o.center[:2] + o.dimensions[:2] / 2
            quad = [(lo[0], lo[1], floor_z), (hi[0], lo[1], floor_z), (hi[0], hi[1], floor_z), (lo[0], hi[1], floor_z)]
            polyline3d(img, cam, quad, mix(colour, max(it.alpha, 0.2)), 1, dash=0.03, closed=True)
        drawn[it.kind] += 1
        if layers.motion and it.kind == FRESH:
            draw_velocity(img, cam, it, floor_z, 1.0)
        if layers.cv_path and it.kind == FRESH:
            draw_cv_path(img, cam, it, floor_z)
        if layers.labels:
            dist = 0.0 if robot_xy is None else float(np.hypot(*(o.center[:2] - robot_xy)))
            moving = it.kind == FRESH and it.colour_key == "MOVING"
            if dist <= label_range or moving:
                labels.append(((0 if moving else 1 + LABEL_PRIORITY[it.kind], dist), it))
    taken = []
    for _, it in sorted(labels, key=lambda x: x[0]):
        o = it.obj
        top = np.array([o.center[0], o.center[1], o.center[2] + o.dimensions[2] / 2 + 0.05])
        if it.kind == GHOST:
            top[2] = floor_z + 0.05
        uv, _ = cam.project(top)
        lines = object_label(it, layers.track_ids)
        if not layers.motion:
            lines = lines[:1] + lines[2:]
        if it.kind != FRESH:
            lines = lines[:1] + lines[2:] if it.kind == STALE else lines
        label_block(img, lines, uv[0], MOTION_COLOUR[it.colour_key], 1.0 if it.kind == FRESH else max(it.alpha, 0.5), taken=taken)
    return drawn


def draw_p3(img, cam, ov: Overlays, layers: Layers, floor_z):
    if layers.phase3_detections:
        for corners in ov.p3_boxes:
            for a, b in BOX_EDGES:
                line3d(img, cam, corners[a], corners[b], P3_COLOUR, 1)
    if layers.phase3_tracks:
        for tr in ov.p3_tracks:
            if tr["status"] == "T":
                continue
            colour = MOTION_COLOUR[tr["motion"]]
            xy = np.asarray(tr["xyz"][:2], float)
            ring = floor_circle(xy, 0.16, floor_z + 0.015, 16)
            if tr["observed"]:
                blend_polys(img, [cam.project(ring)[0]], colour, 0.75)
            else:
                polyline3d(img, cam, ring, colour, 1, dash=0.04, closed=True)
            line3d(img, cam, (*xy, floor_z), (*xy, tr["xyz"][2] + 0.3), mix(colour, 0.6), 1)
            if layers.motion and tr["observed"]:
                v = np.asarray(tr["v"], float)
                if np.hypot(*v) > 0.05:
                    seg = cam.clip_segment((*xy, floor_z + 0.02), (*(xy + v), floor_z + 0.02))
                    if seg is not None and np.isfinite(seg).all():
                        cv2.arrowedLine(img, _pt(seg[0]), _pt(seg[1]), colour, 1, cv2.LINE_AA, tipLength=0.2)
            uv, _ = cam.project([(*xy, tr["xyz"][2] + 0.35)])
            tag = "P3" + (f" #{tr['id']}" if layers.track_ids else "")
            tag += "" if tr["observed"] else (" pred LOST" if tr["status"] == "L" else " pred")
            if np.isfinite(uv).all():
                text(img, tag, uv[0] + [6, 0], 0.4, mix(colour, 0.9 if tr["observed"] else 0.6))


def draw_clusters(img, cam, ov: Overlays):
    for c in ov.clusters:
        colour = CLUSTER_OK if c["accepted"] else CLUSTER_REJECT
        lo, hi = np.asarray(c["lo"]), np.asarray(c["hi"])
        draw_box(img, cam, (lo + hi) / 2, np.maximum(hi - lo, 0.02), colour, 0.8, fill=0.0, thick=1, dash=None if c["accepted"] else 0.1)
        uv, _ = cam.project([c["centroid"]])
        if np.isfinite(uv).all():
            cv2.drawMarker(img, _pt(uv[0]), colour, cv2.MARKER_CROSS, 10, 1, cv2.LINE_AA)
            if not c["accepted"]:
                text(img, c.get("reason", "rejected"), uv[0] + [8, -6], 0.38, colour)


def draw_gt(img, cam, ov: Overlays, floor_z):
    for s in ov.gt:
        if s.get("region"):
            reg = np.asarray(s["region"], float)
            polyline3d(img, cam, np.c_[reg, np.full(len(reg), floor_z + 0.02)], GT_COLOUR, 2, closed=True)
            anchor = np.r_[reg.mean(0), floor_z + 0.05]
        elif s.get("xy") is not None:
            xy = np.asarray(s["xy"], float)
            polyline3d(img, cam, floor_circle(xy, 0.5, floor_z + 0.02, 24), GT_COLOUR, 1, dash=0.05, closed=True)
            line3d(img, cam, (*xy, floor_z), (*xy, floor_z + 1.7), GT_COLOUR, 1)
            anchor = np.r_[xy, floor_z + 1.75]
        else:
            continue
        uv, _ = cam.project([anchor])
        if np.isfinite(uv).all():
            text(img, f"GT {s['id']} {s.get('motion') or ''}".strip(), uv[0] + [6, 0], 0.42, GT_COLOUR)


def render_scene(size, cam: Camera, frame, items, layers: Layers, overlays: Overlays | None = None, display_pose=None, robot_path=None, hud=None):
    """Render one 3D panel. ``display_pose`` (interpolated or held) places the robot only."""
    ov = overlays or Overlays()
    w, h = size
    img = np.empty((h, w, 3), np.uint8)
    img[:] = BG
    pose = display_pose if display_pose is not None else (frame.pose if frame is not None else None)
    floor_z = floor_z_of(pose, ov)
    if floor_z is None:
        floor_z = 0.0
    center = pose[:2, 3] if pose is not None else np.zeros(2)
    draw_grid(img, cam, center, floor_z)
    stats = dict(footprint=0, voxels=0, raw=0, accumulated=0)
    if frame is not None and layers.occupancy_footprint:
        stats["footprint"] = draw_occupancy_footprint(img, cam, frame, floor_z)
    if layers.accumulated and ov.accum_points is not None:
        if layers.point_age and ov.accum_ages is not None:
            cols = age_colours(ov.accum_ages)
        else:
            cols = np.array((105, 100, 95), np.uint8)
        stats["accumulated"] = splat(img, cam, ov.accum_points, cols, 1)
    if frame is not None and layers.background_voxels:
        stats["voxels"] = draw_voxels(img, cam, frame)
    if layers.robot_path and robot_path is not None and len(robot_path) > 1:
        rp = np.array(robot_path, float)
        rp[:, 2] = floor_z + 0.01
        polyline3d(img, cam, rp[-600:], mix(ROBOT_ACCENT, 0.55), 2)
    if layers.raw_incremental and ov.raw_points is not None:
        stats["raw"] = splat(img, cam, ov.raw_points, RAW_COLOUR, 2)
    if layers.clusters:
        draw_clusters(img, cam, ov)
    draw_p3(img, cam, ov, layers, floor_z)
    draw_robot(img, cam, pose, floor_z)
    stats.update(draw_items(img, cam, items, layers, floor_z, None if pose is None else pose[:2, 3]))
    if layers.retrospective_gt and ov.gt:
        draw_gt(img, cam, ov, floor_z)
        text(img, "RETROSPECTIVE annotation overlay: evaluation only, not runtime data", (14, h - 16), 0.5, GT_COLOUR, 1, bg=(30, 20, 30))
    if hud:
        for i, s in enumerate(hud):
            text(img, s, (14, 26 + 22 * i), 0.52, TEXT if i == 0 else DIM, 1)
    return img, stats
