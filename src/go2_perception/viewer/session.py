"""Causal playback session: scene rows (+ optional Debug sources) -> composed 1920x1080 frames.

The playback clock only ever moves through frames already received (``timestamp <= clock``).
Retrospective inputs (annotations) are drawn only by the explicit ``retrospective_gt`` layer.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import render as R
from .camera import Camera, ChaseCamera
from .layers import LAYER_INFO, Layers, mode_layers
from .state import CausalPlayer, DisplayConfig, DisplayState

WIDTH, HEIGHT = 1920, 1080
LEFT_W = 768
RGB_H = 432
PANEL = (WIDTH - LEFT_W, HEIGHT)


@dataclass
class Sources:
    tap: object | None = None  # sources.Tap
    phase3: object | None = None  # sources.Phase3
    rgb: object | None = None  # sources.Rgb (indexed like the scene frames)
    gt_frames: dict | None = None  # rounded midpoint -> annotation states (retrospective)
    title: str = ""
    accumulate_s: float = 1.0
    banner: str = ""
    notes: list = field(default_factory=list)


class Session:
    def __init__(self, frames, sources: Sources | None = None, mode="clean", cfg: DisplayConfig | None = None, cam_mode="chase", fps=15.4, size=(WIDTH, HEIGHT)):
        self.frames = frames
        self.src = sources or Sources()
        self.cfg = cfg or DisplayConfig()
        self.player = CausalPlayer(frames, self.cfg)
        self.mode = mode
        self.layers: Layers = mode_layers(mode)
        self.camera = ChaseCamera(PANEL, mode=cam_mode)
        self.fps = fps
        self.size = size
        self.t0 = frames[0].timestamp if frames else 0.0
        self._reset_state()
        self.render_ms = []

    def _reset_state(self):
        self.state = DisplayState(self.cfg)
        self.fed = -1
        self.camera.reset()

    def set_mode(self, mode):
        self.mode = mode
        retro = self.layers.retrospective_gt
        self.layers = mode_layers(mode)
        self.layers.retrospective_gt = retro

    def _advance(self, k):
        """Feed frames up to index k (causal). Seeking backwards replays from the start."""
        if k < self.fed:
            self._reset_state()
        while self.fed < k:
            self.fed += 1
            f = self.frames[self.fed]
            self.state.update(f)
            if self.fed > 0 and f.epoch != self.frames[self.fed - 1].epoch:
                self.camera.reset()
            self.camera.update(f.pose, f.timestamp)

    def overlays(self, k, layers=None):
        f = self.frames[k]
        ov = R.Overlays()
        L = layers or self.layers
        tap = self.src.tap
        if tap is not None:
            i = tap.index(f.timestamp)
            if i is not None:
                if L.raw_incremental:
                    xyz, _, _ = tap.scan(i)
                    ov.raw_points = xyz
                if L.accumulated:
                    ov.accum_points, ov.accum_ages = tap.accumulated(i, self.src.accumulate_s)
                if L.clusters:
                    ov.clusters = tap.clusters(i)
        p3 = self.src.phase3
        if p3 is not None and (L.phase3_detections or L.phase3_tracks):
            j = p3.index(f.source_time)
            if j is None:
                j = p3.latest_before(f.source_time)
            if j is not None:
                ov.p3_boxes = p3.boxes(j) if L.phase3_detections else []
                ov.p3_tracks = p3.tracks(j) if L.phase3_tracks else []
        if L.retrospective_gt and self.src.gt_frames:
            ov.gt = self.src.gt_frames.get(round(f.source_time, 3), [])
        return ov

    def render(self, clock):
        """Compose the frame displayed at playback ``clock`` (seconds, scene clock)."""
        start = time.perf_counter()
        k = self.player.index_at(clock)
        img = np.zeros((self.size[1], self.size[0], 3), np.uint8)
        img[:] = R.BG
        if k < 0:
            R.text(img, "waiting for first scene frame", (LEFT_W + 40, 80), 0.8)
            return img, None
        self._advance(k)
        frame = self.frames[k]
        pose, pose_mode = self.player.pose_at(clock)
        cam = self.camera.view(pose, clock - self.player.delay)
        ov = self.overlays(k)
        hud = [
            f"3D scene | go2-scene-v1 in odom | {self.mode.upper()} mode",
            f"display: {'latest scan held' if pose_mode == 'held' else 'robot pose ' + pose_mode} | camera {self.camera.mode}, smoothed viewpoint only",
        ]
        if self.player.delay:
            hud.append(f"causal pose interpolation: display delayed {self.player.delay * 1000:.0f} ms")
        panel, stats = R.render_scene(PANEL, cam, frame, self.state.items, self.layers, ov, pose, [p for _, p in self.state.robot_path], hud)
        img[:, LEFT_W:] = panel
        self._left(img, k, frame, clock, stats, pose)
        cv2.line(img, (LEFT_W, 0), (LEFT_W, self.size[1]), (70, 70, 70), 2)
        self.render_ms.append((time.perf_counter() - start) * 1000)
        return img, stats

    def render_panels(self, clock, specs, status=(), rgb_thumb=True):
        """Several 3D panels side by side at one playback time (shared causal state and
        viewpoint smoothing), e.g. raw vs accumulated vs boxes. ``specs``: [(label, Layers)]."""
        k = self.player.index_at(clock)
        img = np.zeros((self.size[1], self.size[0], 3), np.uint8)
        img[:] = R.BG
        if k < 0:
            return img
        self._advance(k)
        frame = self.frames[k]
        pose, _ = self.player.pose_at(clock)
        n = len(specs)
        strip = 200
        w, h = self.size[0] // n, self.size[1] - strip
        for i, (label, layers) in enumerate(specs):
            pc = copy.copy(self.camera)
            pc.cam = Camera((w, h), 58 if n > 2 else 52)
            cam = pc.view(pose, clock - self.player.delay)
            panel, stats = R.render_scene((w, h), cam, frame, self.state.items, layers, self.overlays(k, layers), pose, [p for _, p in self.state.robot_path], [label])
            img[:h, i * w : (i + 1) * w] = panel
            if i:
                cv2.line(img, (i * w, 0), (i * w, h), (70, 70, 70), 2)
        cv2.line(img, (0, h), (self.size[0], h), (70, 70, 70), 2)
        x = 14
        if rgb_thumb and self.src.rgb is not None:
            im, idx, dt, _ = self.src.rgb.get(k)
            tw, th = int((strip - 16) * 16 / 9), strip - 16
            if im is not None:
                img[h + 8 : h + 8 + th, 8 : 8 + tw] = cv2.resize(im, (tw, th), interpolation=cv2.INTER_AREA)
            else:
                R.text(img, "camera frame missing", (20, h + strip // 2), 0.5, (80, 80, 255))
            x = tw + 24
        lines = [f"{self.src.title}   t = {clock - self.t0:6.2f} s   stamp {frame.timestamp:.3f} ({frame.clock_domain})   epoch {frame.epoch}", *status]
        for i, s in enumerate(lines):
            R.text(img, s, (x, h + 30 + 26 * i), 0.55 if i == 0 else 0.5, R.TEXT if i == 0 else R.DIM)
        if self.src.banner:
            col = (40, 40, 170) if self.src.banner.startswith("FAIL") else (40, 110, 40)
            y = h + strip - 62
            cv2.rectangle(img, (x - 6, y), (self.size[0] - 10, y + 54), col, -1)
            for i, line in enumerate(self.src.banner.split("|")[:2]):
                R.text(img, line.strip(), (x + 6, y + 22 + 24 * i), 0.6 if i == 0 else 0.5, R.TEXT, 2 if i == 0 else 1)
        return img

    # ---- left column: camera, status, legend ----
    def _left(self, img, k, frame, clock, stats, pose):
        x0 = 0
        R.text(img, "CAMERA (recorded RGB) - camera-only boxes, not calibrated to LiDAR", (x0 + 12, 28), 0.5, R.TEXT)
        rgb = self.src.rgb
        y = 44
        if rgb is not None:
            im, idx, dt, dets = rgb.get(k)
            if im is None:
                cv2.rectangle(img, (x0, y), (LEFT_W, y + RGB_H), (40, 40, 40), -1)
                R.text(img, "CAMERA FRAME MISSING (none within 0.15 s)", (x0 + 60, y + RGB_H // 2), 0.7, (80, 80, 255), 2)
            else:
                ih, iw = im.shape[:2]
                s = min(LEFT_W / iw, RGB_H / ih)
                tw, th = int(iw * s), int(ih * s)
                img[y : y + th, x0 : x0 + tw] = cv2.resize(im, (tw, th), interpolation=cv2.INTER_AREA)
                for name, conf, (a, b, c, d) in dets:
                    col = (0, 220, 255) if name == "person" else (0, 200, 0)
                    p, q = (int(x0 + a * s), int(y + b * s)), (int(x0 + c * s), int(y + d * s))
                    cv2.rectangle(img, p, q, col, 2, cv2.LINE_AA)
                    R.text(img, f"{name} {conf:.2f}", (p[0] + 2, max(p[1] - 5, y + 12)), 0.42, (0, 0, 0), 1, bg=col)
                R.text(img, f"camera frame {idx}  camera-LiDAR dt {dt * 1000:+.0f} ms (host receipt, approximate)", (x0 + 12, y + RGB_H + 22), 0.45, R.DIM)
        else:
            cv2.rectangle(img, (x0, y), (LEFT_W, y + RGB_H), (36, 32, 30), -1)
            R.text(img, "no camera stream (synthetic or not loaded)", (x0 + 140, y + RGB_H // 2), 0.6, R.DIM)
        if self.src.banner:
            col = (40, 40, 170) if self.src.banner.startswith("FAIL") else (40, 110, 40)
            cv2.rectangle(img, (x0 + 8, y + RGB_H - 74), (LEFT_W - 8, y + RGB_H - 6), col, -1)
            for i, line in enumerate(self.src.banner.split("|")[:2]):
                R.text(img, line.strip(), (x0 + 18, y + RGB_H - 46 + 30 * i), 0.62 if i == 0 else 0.5, R.TEXT, 2 if i == 0 else 1)
        y = y + RGB_H + 50
        objs = frame.objects
        fresh = sum(o.observed for o in objs)
        stale = len(objs) - fresh
        moving = sum(o.observed and o.motion_state == "MOVING" for o in objs)
        lat = frame.diagnostics.get("processing_ms")
        lines = [
            (f"{self.src.title}", R.TEXT, 0.62),
            (f"t = {clock - self.t0:6.2f} s   scene stamp {frame.timestamp:.3f} ({frame.clock_domain})   epoch {frame.epoch}", R.TEXT, 0.48),
            (f"objects: {fresh} fresh  {stale} stale  {stats.get('ghost', 0)} unseen   MOVING {moving}", R.TEXT, 0.48),
            (f"background cells {len(frame.cell_centers)} (absence = unknown, never free)", R.DIM, 0.45),
            (f"perception {lat:.1f} ms/scan (5A compute only)" if lat is not None else "perception latency n/a", R.DIM, 0.45),
        ]
        if self.render_ms:
            lines.append((f"render {np.mean(self.render_ms[-30:]):.0f} ms/frame (this viewer)", R.DIM, 0.45))
        for s, col, sc in lines:
            R.text(img, s, (x0 + 14, y), sc, col)
            y += 24
        warn = []
        if pose is None:
            warn.append("NO POSE: robot hidden, no object claims")
        if frame.diagnostics.get("reset_reason"):
            warn.append(f"EPOCH RESET: {frame.diagnostics['reset_reason']}")
        if frame.diagnostics.get("synthetic"):
            warn.append("SYNTHETIC DATA")
        for s in warn:
            R.text(img, s, (x0 + 14, y + 4), 0.5, (255, 255, 255), 1, bg=(40, 40, 190))
            y += 28
        y += 8
        legend = [
            ("fresh: solid box, current-scan support", R.MOTION_COLOUR["UNCERTAIN"], "solid"),
            ("stale: dashed, fades to 0.6 s", R.MOTION_COLOUR["UNCERTAIN"], "dash"),
            ("unseen: dotted footprint = unknown", R.MOTION_COLOUR["UNCERTAIN"], "dot"),
            ("MOVING (fresh only)", R.MOTION_COLOUR["MOVING"], "solid"),
            ("STATIONARY", R.MOTION_COLOUR["STATIONARY"], "solid"),
            ("UNCERTAIN / no motion evidence", R.MOTION_COLOUR["UNCERTAIN"], "solid"),
            ("arrow 1 s of travel, ring 2-sigma", R.TEXT, "arrow"),
            ("dashed path: const.-velocity prediction", R.MOTION_COLOUR["MOVING"], "dash"),
        ]
        if self.layers.raw_detection_boxes:
            legend.append(("white brackets: raw observed box", (255, 255, 255), "bracket"))
        if self.layers.phase3_detections or self.layers.phase3_tracks:
            legend.append(("cyan-yellow: original Phase 3 (0.3 s)", R.P3_COLOUR, "solid"))
        if self.layers.clusters:
            legend.append(("green/red: 5A compact/rejected cluster", R.CLUSTER_OK, "solid"))
        if self.layers.raw_incremental:
            legend.append(("cyan points: current incremental scan", R.RAW_COLOUR, "solid"))
        for i, (name, col, kind) in enumerate(legend):
            cy = y + 20 * i
            if kind == "solid":
                cv2.rectangle(img, (x0 + 16, cy - 10), (x0 + 34, cy + 2), col, 2)
            elif kind == "dash":
                for s in range(0, 18, 6):
                    cv2.line(img, (x0 + 16 + s, cy - 4), (x0 + 19 + s, cy - 4), col, 2)
            elif kind == "dot":
                for s in range(0, 18, 4):
                    cv2.circle(img, (x0 + 17 + s, cy - 4), 1, col, -1)
            elif kind == "arrow":
                cv2.arrowedLine(img, (x0 + 14, cy - 4), (x0 + 36, cy - 4), col, 2, tipLength=0.35)
            else:
                cv2.line(img, (x0 + 16, cy - 10), (x0 + 16, cy - 4), col, 1)
                cv2.line(img, (x0 + 16, cy - 10), (x0 + 22, cy - 10), col, 1)
            R.text(img, name, (x0 + 44, cy), 0.43, R.TEXT)
        y += 20 * len(legend) + 10
        if self.mode == "debug":
            R.text(img, "Debug layers (key: on/off)", (x0 + 14, y), 0.45, R.TEXT)
            y += 20
            names = [n for n in LAYER_INFO]
            for i, n in enumerate(names):
                key, lab = LAYER_INFO[n]
                on = getattr(self.layers, n)
                col = (120, 220, 120) if on else (110, 110, 110)
                cx = x0 + 14 + (i % 3) * 252
                cy = y + 17 * (i // 3)
                R.text(img, f"[{key}] {'ON ' if on else 'off'} {lab[:27]}", (cx, cy), 0.36, col)
        for i, s in enumerate(self.src.notes[:3]):
            R.text(img, s, (x0 + 14, self.size[1] - 14 - 20 * i), 0.42, (120, 200, 255))
