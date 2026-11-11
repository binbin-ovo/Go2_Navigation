"""Pinhole camera for the software renderer, and a smoothed chase camera.

Smoothing applies only to the *viewpoint*. Scene geometry is never smoothed or moved.
"""

from __future__ import annotations

import copy
import math

import numpy as np

NEAR = 0.15


class Camera:
    def __init__(self, size, fov_y_deg=50.0):
        self.w, self.h = size
        self.f = 0.5 * self.h / math.tan(math.radians(fov_y_deg) / 2)
        self.R = np.eye(3)  # rows: camera x (right), y (down), z (forward) in odom
        self.eye = np.zeros(3)

    def look_at(self, eye, target, up=(0.0, 0.0, 1.0)):
        eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
        fwd = target - eye
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, up)
        if np.linalg.norm(right) < 1e-9:
            right = np.array([1.0, 0.0, 0.0])
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        self.R = np.vstack([right, down, fwd])
        self.eye = eye
        return self

    def to_cam(self, p):
        return (np.atleast_2d(np.asarray(p, float)) - self.eye) @ self.R.T

    def project(self, p):
        """(N, 3) odom -> (uv (N, 2), depth (N,)). Points with depth < NEAR get NaN uv."""
        c = self.to_cam(p)
        z = c[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.f * c[:, 0] / z + self.w / 2
            v = self.f * c[:, 1] / z + self.h / 2
        uv = np.c_[u, v]
        uv[z < NEAR] = np.nan
        return uv, z

    def clip_segment(self, a, b):
        """Clip a 3D segment to the near plane; returns projected endpoints or None."""
        ca, cb = self.to_cam([a, b])
        za, zb = ca[2], cb[2]
        if za < NEAR and zb < NEAR:
            return None
        if za < NEAR or zb < NEAR:
            s = (NEAR - za) / (zb - za)
            p = ca + s * (cb - ca)
            if za < NEAR:
                ca = p
            else:
                cb = p
        out = []
        for c in (ca, cb):
            out.append((self.f * c[0] / c[2] + self.w / 2, self.f * c[1] / c[2] + self.h / 2))
        return np.array(out)


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class ChaseCamera:
    """Follows the robot from behind/above with exponential smoothing of position and yaw.

    ``mode``: ``chase`` (follows robot heading), ``fixed`` (start heading, follows position
    only) or ``static`` (first pose, never moves). ``orbit_deg``/``pitch_offset``/``zoom`` are
    user offsets for the interactive viewer.
    """

    def __init__(self, size, fov_y_deg=50.0, back=4.2, height=3.3, ahead=2.2, tau_s=0.6, mode="chase"):
        self.cam = Camera(size, fov_y_deg)
        self.back, self.height, self.ahead, self.tau = back, height, ahead, tau_s
        self.mode = mode
        self.pos = None
        self.yaw = None
        self.t = None
        self.orbit_deg = 0.0
        self.pitch_offset = 0.0
        self.zoom = 1.0

    def reset(self):
        self.pos = self.yaw = self.t = None

    def view(self, pose, t):
        """Camera at time ``t`` without changing the smoothing state, so a rendered image depends
        only on the scene frames fed through :meth:`update`, never on how often it is drawn."""
        return copy.copy(self).update(pose, t)

    def update(self, pose, t):
        """Advance the smoothed viewpoint to time ``t`` given the displayed robot pose (or None)."""
        if pose is not None:
            p = pose[:3, 3]
            yaw = math.atan2(pose[1, 0], pose[0, 0])
            if self.pos is None:
                self.pos, self.yaw, self.t = p.copy(), yaw, t
            elif self.mode != "static":
                dt = max(t - self.t, 0.0)
                a = 1.0 - math.exp(-dt / self.tau) if self.tau > 0 else 1.0
                self.pos = self.pos + a * (p - self.pos)
                if self.mode == "chase":
                    self.yaw = self.yaw + a * _wrap(yaw - self.yaw)
        self.t = t if self.t is None else max(t, self.t)
        if self.pos is None:
            self.cam.look_at((-self.back, 0, self.height), (self.ahead, 0, 0))
            return self.cam
        yaw = self.yaw + math.radians(self.orbit_deg)
        h = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        ground = self.pos - np.array([0, 0, 0.32])
        eye = ground - self.zoom * self.back * h + np.array([0, 0, self.zoom * self.height + self.pitch_offset])
        target = ground + self.ahead * h
        return self.cam.look_at(eye, target)
